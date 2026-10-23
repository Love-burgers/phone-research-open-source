"""Shared answer generation, evidence validation and extractive fallback."""

import json
import logging
import re

from .chat_schemas import AnswerBlock, ChatMessage, ModelAnswer
from .chat_prompts import ANSWER_INSTRUCTIONS
from .chat_intent import TOPIC_WORDS
from .source_rules import official_phone_url
from .selection_catalog import budget_position
from .selection_prices import unsupported_prices
from .selection_presentation import configuration_block, configuration_facts, conversation_guidance, presentation_copy, quoted_price
from .selection_requirements import card_matches, game_in_text, measured_game_parts, topic_observations, recommendation_gaps
from .tool_loop import ExecutionLimit
from .evidence_rules import unsupported_parameters, photo_observation, unsupported_game_averages, comparison_scope_mismatch, hardware_extrapolation
from .conversation_context import asks_price, recent_dialogue, focused_games, evidence_excerpt
from .config import get_settings

logger = logging.getLogger(__name__)

class AnswerComposer:
    """Used by both the Agent and offline route, with identical evidence checks."""

    def __init__(self, assistant, state, model_client=None):
        self.assistant, self.state = assistant, state
        self.client = model_client or self.assistant.model
        self.profile, self.sources, self.catalog = self.state["session"].profile, self.state["sources"], self.state["catalog"]
        registry=list(self.sources)
        self.recommending = self.state.get("recommendation_turn", not self.profile.focus_phone_id and
            bool(re.search(r"推荐|重新选|换.{0,4}(?:款|批)|预算|价位|左右", self.state["message"])))
        self.price_intent = self.recommending or asks_price(self.state['message'])
        self.gaming = "游戏" in self.profile.priorities or bool(self.profile.games) or any(word in self.state["message"] for word in TOPIC_WORDS["游戏"])
        # Every topical non-purchase answer follows the current semantic scope,
        # even when the extractor also stores a subject/duration requirement.
        scope_topics=self.state.get('focus_topics') or [topic for topic,words in TOPIC_WORDS.items()
                                                       if any(word in self.state['message'] for word in words)]
        self.detail_followup=not self.recommending and not self.price_intent and bool(scope_topics or self.state.get('detail_followup'))
        self.response_focus = self.state.get('response_focus', self.state['message'])
        self.focus_topics = scope_topics or self.profile.priorities or list(TOPIC_WORDS)
        self.answer_games = focused_games(self.profile, self.state) if self.detail_followup else self.profile.games
        self.evidence_profile = self.profile.model_copy(update={'games': self.answer_games})
        if self.detail_followup:
            self.price_intent = False
            self.gaming = '游戏' in self.focus_topics
            self.sources=[s for s in self.sources if s.evidence_kind!='price']
        if self.gaming:
            focused = []
            for phone_id in self.state["selected"]:
                available = [source for source in self.sources if source.phone_id == phone_id]
                relevant = [source for source in available if (source.evidence_kind in {"review", "price"}
                            or (self.recommending or self.detail_followup or self.profile.high_refresh_required) and source.evidence_kind == "spec"
                            and bool(configuration_facts(self.catalog[phone_id],self.focus_topics,[source]) or self.profile.high_refresh_required))
                            and source.source_type != "snippet"]
                # Once measured/price evidence exists, unrelated spec snippets must
                # not distract the answer or be mistaken for measured performance.
                focused.extend(relevant or available)
            self.sources = focused
        self.cards = [self.catalog[item].card(self.profile, self.sources) for item in self.profile.candidate_ids]
        if self.gaming and not self.price_intent and any(source.evidence_kind == "review" for source in self.sources):
            self.sources = [source for source in self.sources if source.evidence_kind == "review" or
                       (self.detail_followup or self.profile.high_refresh_required) and source.evidence_kind == "spec"]
        # The full registry remains in the reply/trace; keep the final model
        # request bounded and focused so useful work doesn't end in context_limit.
        self.registered_sources = registry
        ordered = sorted(self.sources, key=lambda s: (s.source_type == "snippet",
                         ({"price": 0, "spec": 1, "review": 2} if self.recommending else
                          {"review": 0, "price": 1, "spec": 2}).get(s.evidence_kind, 3)))
        counts = {}
        self.sources = []
        for source in ordered:
            counts[source.phone_id] = counts.get(source.phone_id, 0) + 1
            if counts[source.phone_id] <= 4 and len(self.sources) < 16:
                self.sources.append(source)
        self.blocks = []
        self.unusable_ids=[]
        self.unanswered_refinement=[]
        self.configuration_only=[]

    def run(self):
        if self.recommending and not self.profile.explicit_candidate_ids and not self.profile.focus_phone_id:
            rejected={pid:recommendation_gaps(self.catalog[pid],self.profile,self.registered_sources)
                      for pid in self.state['selected']}
            rejected={pid:gaps for pid,gaps in rejected.items() if gaps}
            self.unusable_ids.extend(rejected)
            self.state['selected']=[pid for pid in self.state['selected'] if pid not in rejected]
            self.profile.candidate_ids=[pid for pid in self.profile.candidate_ids if pid not in rejected]
            if rejected:self.state['recommendation_gaps']=rejected
            if not self.state['selected']:
                self.sources=self.registered_sources
                return self.build_reply()
        self.generate()
        self.sources = self.registered_sources
        self.fill_missing_blocks()
        self.complete_topic_coverage()
        self.merge_blocks()
        return self.build_reply()

    def request_source(self, source):
        """Compact request evidence; the complete registry is kept for validation."""
        record=source.model_dump(include={'id','phone_id','source_type','evidence_kind','market','published_on',
            'checked_on','reference_price','price_variant','price_options'},exclude_none=True)
        record={key:value for key,value in record.items() if value!=[]}
        quote=source.quote
        if self.detail_followup and source.evidence_kind=='spec':
            facts=configuration_facts(self.catalog[source.phone_id],self.focus_topics,[source])
            parts=re.split(r' / |[。；]',quote)
            relevant=[part for part in parts if any(value.removeprefix('官方标称') in part for value,_ in facts)]
            quote=' / '.join(relevant) if relevant else quote
        record['quote']=evidence_excerpt(quote,1000 if source.evidence_kind=='review' else 400)
        return record

    def generate(self):
        if self.client.available and not self.state.get("model_failed") and self.sources:
            answer_stage = "request"
            try:
                payload = {"本轮": self.state["message"], "条件": self.profile.model_dump(), "本轮是否询价或重新选购": self.price_intent,
                           "候选核对": [{'id':card.id,'name':card.name,'matches':[label[:100] for label in card.matches]}
                                        for card in self.cards],
                           "机型": {phone_id: self.catalog[phone_id].name for phone_id in self.state["selected"]},
                           "引用范围": {phone_id: [source.id for source in self.sources if source.phone_id == phone_id]
                                        for phone_id in self.state["selected"]},
                           "历史": recent_dialogue(self.state['session'].messages, characters=min(6000, get_settings().chat_history_characters)),
                           "本轮关注游戏": self.answer_games,
                           "可引用配置摘要": {pid: configuration_block(self.catalog[pid],
                                self.profile.model_copy(update={'priorities':self.focus_topics}) if self.detail_followup else self.profile,self.sources)
                                             for pid in self.state["selected"]},
                           "展示策略": "本轮是初筛推荐：正文以具体配置、一个适合预算的版本价与取舍为主；缺实测和未知价格由界面详情统一说明，不逐款重复。" if self.recommending else "直接回答当前问题",
                           "本轮对话动作": self.state.get('conversation_route'),
                           "细节追问": "用户细化："+self.response_focus+"。围绕当前候选接着回答这个细节，保留其他条件。有对应评测用评测，否则只用相关参数，不把参数当体验；不报价格，不重播推荐，不逐款说缺实测。" if self.detail_followup else None,
                           "来源": [self.request_source(source) for source in self.sources]}
                raw_draft = self.client.complete_json([
                    {"role": "system", "content": ANSWER_INSTRUCTIONS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False,separators=(',',':'))},
                ])
                self.state["answer_draft"] = raw_draft
                answer_stage = "schema"
                draft = ModelAnswer.model_validate(raw_draft)
                by_id = {source.id: source for source in self.registered_sources}
                for block in draft.blocks:
                    self.blocks.extend(self.checked_parts(block, by_id))
                self.state["answer_generated"] = bool(self.blocks)
                if not self.blocks:
                    self.state["model_failed"] = True
            except Exception as error:
                logger.warning("answer_failed stage=%s error_type=%s", answer_stage, type(error).__name__)
                self.state["answer_failure"] = {"stage": answer_stage, "error_type": type(error).__name__}
                self.state["answer_limited"] = isinstance(error, ExecutionLimit)
                self.state["model_failed"] = not self.state["answer_limited"]

    def checked_parts(self, block, by_id):
        """Recover independent grounded sentences; never salvage invalid citations."""
        if block.phone_id not in self.state['selected'] or any(
                sid not in by_id or by_id[sid].phone_id != block.phone_id for sid in block.evidence_ids):
            self.validate_block(block, by_id)
            return []
        original = block.text
        if comparison_scope_mismatch(block.text, len(self.state['selected'])):
            block = block.model_copy(update={'text': ''.join(clause for clause in re.split(r'(?<=[，,。；;])', block.text)
                if not comparison_scope_mismatch(clause,len(self.state['selected']))).strip('，,；; ')})
            if not block.text:
                return []
        if self.detail_followup:
            # Remove only pricing clauses. Keep subject headings and factual
            # clauses even when a price was appended to the same sentence.
            clauses = re.split(r'(?<=[，,。；;])', block.text)
            block = block.model_copy(update={'text': ''.join(part for part in clauses if not re.search(
                r'\d+\s*元|建议零售价|官网标价|官方价|预算内|预算余量', part)).strip('，,；; ')})
            if not block.text:
                return []
            block.evidence_ids = [sid for sid in block.evidence_ids if by_id[sid].evidence_kind != 'price']
            if not block.evidence_ids:
                return []
        if self.validate_block(block, by_id) or self.repair_citations(block, by_id):
            if block.text != original:
                self.state.setdefault('answer_repairs', []).append({'phone_id': block.phone_id, 'kind': 'scope_clauses'})
            return [block]
        failure = self.state.get('answer_failure', {}).copy()
        if failure.get('stage') not in {'parameter_evidence', 'recommendation_evidence', 'photo_scope', 'photo_evidence', 'game_measurement', 'hardware_inference'}:
            return []
        # Keep complete sentences so durations/measurement conditions survive.
        # For an explicit suitability extrapolation, remove that clause only;
        # its neighbouring sample observation must still pass all checks.
        text = block.text
        if failure.get('stage') == 'hardware_inference':
            text = ''.join(part for part in re.split(r'(?<=[，,。；;])', text) if not hardware_extrapolation(part))
            recovered = block.model_copy(update={'text':text.strip('，,；; ')})
            if recovered.text and self.validate_block(recovered,by_id):
                self.state['answer_failure']=failure
                self.state.setdefault('answer_repairs',[]).append({'phone_id':block.phone_id,'kind':'hardware_inference_clauses'})
                return [recovered]
        if failure.get('stage') == 'photo_scope':
            text = ''.join(part for part in re.split(r'(?<=[，,。；;])', text)
                           if not self.photo_extrapolation(part, [by_id[sid] for sid in block.evidence_ids]))
        parts = []
        previous_rejected = False
        for sentence in re.split(r'(?<=[。；;])', text):
            if not sentence.strip():
                continue
            if previous_rejected and re.match(r'\s*(?:这与|这并|这不|这是|这仅|因此|所以|上述|该测试|该结果)', sentence):
                continue
            candidate = block.model_copy(update={'text': sentence.strip()})
            if self.validate_block(candidate, by_id) or self.repair_citations(candidate, by_id):
                parts.append(candidate)
                previous_rejected = False
            else:
                previous_rejected = True
        # A recovered sample statement must retain the source's scope limit,
        # even if the rejected sentence contained the model's version of it.
        if parts and failure.get('stage') == 'photo_scope':
            boundaries = [s.quote.split('资料边界：', 1)[1].split('；', 1)[0].strip()
                          for s in (by_id[sid] for sid in block.evidence_ids)
                          if s.evidence_kind == 'review' and '样张' in s.quote and '资料边界：' in s.quote]
            if boundaries:
                parts.append(block.model_copy(update={'text': '来源边界：'+'；'.join(dict.fromkeys(boundaries))+'。'}))
        self.state['answer_failure'] = failure
        if parts:
            self.state.setdefault('answer_repairs', []).append({'phone_id': block.phone_id, 'kind': 'validated_sentences',
                                                              'failure': failure.get('stage')})
        return parts

    @staticmethod
    def photo_extrapolation(text, cited):
        sample_only = any(s.evidence_kind == 'review' and '样张' in s.quote and
                          re.search(r'不能证明|不可外推|仅限|静态', s.quote) for s in cited)
        return sample_only and (bool(re.search(r'不要期望.{0,20}(?:拍准|抓拍)|建议.{0,10}当作.{0,12}机', text)) or any(re.search(
            r'(?:适合|擅长|保证|确保|胜任|能拍准)[^。；]{0,24}(?:人像|舞台|抓拍|夜景|运动|成片)', clause)
            and not re.search(r'不能|无法|不保证|不适合|不要|未能|尚不能', clause)
            for clause in re.split(r'[，,。；;]', text)))

    def complete_topic_coverage(self):
        """A valid partial draft must not suppress other retained buying needs."""
        if not self.recommending:
            return
        by_id = {s.id: s for s in self.registered_sources}
        for pid in self.state['selected']:
            existing = [b for b in self.blocks if b.phone_id == pid]
            if not existing:
                continue
            for topic in self.profile.priorities:
                covered = False
                for block in existing:
                    cited = [by_id[sid] for sid in block.evidence_ids]
                    normalize = lambda value: re.sub(r'\s+', '', value).casefold()
                    literal_facts = configuration_facts(self.catalog[pid], [topic], cited, limit=8)
                    if any(normalize(value) in normalize(block.text) for value, _ in literal_facts) or topic_observations(
                            pid, topic, [s.model_copy(update={'quote': block.text}) for s in cited], self.profile.games):
                        covered = True
                        break
                if covered:
                    continue
                facts = configuration_facts(self.catalog[pid], [topic], self.registered_sources, limit=1)
                if facts:
                    candidate = AnswerBlock(phone_id=pid, text=topic+'配置参考：'+facts[0][0]+'。', evidence_ids=[facts[0][1]])
                    if self.validate_block(candidate, by_id):
                        self.blocks.append(candidate)
                        self.state.setdefault('answer_repairs', []).append({'phone_id': pid, 'kind': 'topic_configuration', 'topic': topic})

    def merge_blocks(self):
        """Combine same-phone parts without truncating facts or citation lists."""
        merged = []
        for block in self.blocks:
            if any(b.phone_id == block.phone_id and b.text == block.text for b in merged):
                continue
            previous = next((b for b in reversed(merged) if b.phone_id == block.phone_id), None)
            ids = list(dict.fromkeys([*(previous.evidence_ids if previous else []), *block.evidence_ids]))
            if previous and len(previous.text)+len(block.text)+1 <= 600 and len(ids) <= 8:
                previous.text += '\n'+block.text
                previous.evidence_ids = ids
            else:
                merged.append(block.model_copy(deep=True))
        self.blocks = merged

    def repair_citations(self, block, by_id):
        """Repair only missing literal parameter citations, without rewriting claims."""
        if self.state.get('answer_failure',{}).get('stage')!='parameter_evidence':return False
        ids=list(block.evidence_ids)
        if any(sid not in by_id or by_id[sid].phone_id!=block.phone_id for sid in ids):return False
        for source in self.registered_sources:
            if source.phone_id!=block.phone_id or source.source_type=='snippet' or source.id in ids:continue
            before=unsupported_parameters(block.text,[by_id[sid] for sid in ids],self.profile.requirements)
            after=unsupported_parameters(block.text,[by_id[sid] for sid in ids]+[source],self.profile.requirements)
            if len(after)<len(before) and len(ids)<8:ids.append(source.id)
        repaired=block.model_copy(update={'evidence_ids':ids})
        if ids!=block.evidence_ids and self.validate_block(repaired,by_id):
            added=ids[len(block.evidence_ids):]
            block.evidence_ids=ids
            self.state.pop('answer_failure',None)
            self.state.setdefault('citation_repairs',[]).append({'phone_id':block.phone_id,'added_ids':added})
            return True
        return False

    def validate_block(self, block, by_id):
        if block.phone_id not in self.state["selected"] or any(
            item not in by_id or by_id[item].phone_id != block.phone_id for item in block.evidence_ids
        ):
            self.state["answer_failure"] = {"stage": "citation_scope", "error_type": "ValueError"}
            return False
        if any(source.id in block.text for source in self.sources):
            self.state["answer_failure"] = {"stage": "internal_identifier", "error_type": "ValueError"}
            return False
        cited = [by_id[item] for item in block.evidence_ids]
        if comparison_scope_mismatch(block.text,len(self.state['selected'])):
            self.state['answer_failure']={'stage':'comparison_scope','error_type':'ValueError'}
            return False
        if not any(s.source_type != 'snippet' for s in cited):
            self.state['answer_failure'] = {'stage': 'evidence_body', 'error_type': 'ValueError'}
            return False
        if self.detail_followup and re.search(r'\d+\s*元|建议零售价|官网标价',block.text):
            self.state['answer_failure']={'stage':'answer_scope','error_type':'ValueError'}
            return False
        if self.photo_extrapolation(block.text, cited):
            self.state['answer_failure'] = {'stage': 'photo_scope', 'error_type': 'ValueError'}
            return False
        if photo_observation(block.text) and not any(s.evidence_kind == 'review' and s.source_type != 'snippet'
                                                    and photo_observation(s.quote) for s in cited):
            self.state['answer_failure'] = {'stage': 'photo_evidence', 'error_type': 'ValueError'}
            return False
        if unsupported_parameters(block.text, cited, self.profile.requirements):
            self.state['answer_failure'] = {'stage': 'parameter_evidence', 'error_type': 'ValueError'}
            return False
        if self.detail_followup and self.state.get('focus_terms') and not any(
                game_in_text(term, block.text) for term in self.state['focus_terms']):
            self.state['answer_failure']={'stage':'question_focus','error_type':'ValueError'}
            return False
        # A quoted number may be real while the suggested purchase
        # violates the budget, or falsely upgrades a review to an
        # official price. Reject that block and preserve its draft.
        over_budget = False
        if self.price_intent and self.profile.budget_max:
            for match in re.finditer(r"(?<!\d)(\d{3,5})\s*元", block.text):
                nearby = block.text[max(0, match.start()-35):match.end()+25]
                if int(match[1]) > self.profile.budget_max and not re.search(r"超出|超过|超预算|加预算|提高预算|放宽|不在|无法|不能|需.*确认", nearby):
                    over_budget = True
        official_claim = bool(re.search(r"官网.{0,4}(?:价|\d)|官方售价|官方建议零售价", block.text))
        official_evidence = any(s.evidence_kind == "price" and s.source_type != "snippet"
            and s.market == "中国大陆" and official_phone_url(self.catalog[block.phone_id].brand, s.url) for s in cited)
        if over_budget or official_claim and not official_evidence:
            self.state["answer_failure"] = {"stage": "budget_scope" if over_budget else "price_provenance", "error_type": "ValueError"}
            return False
        if unsupported_prices(block.text,cited,self.profile.budget_max):
            self.state['answer_failure']={'stage':'price_evidence','error_type':'ValueError'}
            return False
        labels, _ = card_matches(block.phone_id, self.evidence_profile, cited)
        experience_topics = [topic for topic,pattern in (
            ('续航',r"续航|使用时长"),('拍照',r"拍照|夜景"))
            if any(re.search(pattern,clause) and re.search(r"最(?:强|好|突出|优秀)|更(?:强|好)|领先|持久|优秀",clause)
                   for clause in re.split(r'[，。；]',block.text))]
        if any(not any(label.startswith(topic) and '实测资料' in label for label in labels)
               for topic in experience_topics):
            self.state['answer_failure']={'stage':'experience_evidence','error_type':'ValueError'}
            return False
        # A preliminary shortlist may describe hardware without a
        # benchmark. Reject unsupported experience claims, rather
        # than rejecting every useful gaming configuration answer.
        game_claim = bool(re.search(r"(?:游戏|帧率|运行).{0,12}(?:满帧|低温|不发热|不卡顿|不掉帧)|(?:游戏|运行)(?:表现|非常|很)?(?:流畅|稳定)|流畅(?:运行|玩)|满帧|\d+(?:\.\d+)?\s*(?:帧|fps)", block.text, re.I))
        game_review = any(label.startswith("游戏表现") and "实测资料" in label for label in labels)
        if (self.gaming and game_claim and not game_review):
            self.state["answer_failure"] = {"stage": "recommendation_evidence", "error_type": "ValueError"}
            return False
        if self.gaming and unsupported_game_averages(block.text, cited, self.answer_games):
            self.state['answer_failure'] = {'stage':'game_measurement','error_type':'ValueError'}
            return False
        if self.gaming and any(hardware_extrapolation(clause) for clause in re.split(r'[，,。；;]',block.text)):
            self.state['answer_failure']={'stage':'hardware_inference','error_type':'ValueError'}
            return False
        facts = configuration_facts(self.catalog[block.phone_id], self.profile.priorities, cited)
        if (self.recommending and facts and not re.search(r"\d", block.text)
            and re.search(r"缺|暂无|没有|待核对", block.text)):
            self.state["answer_failure"] = {"stage": "basic_information", "error_type": "ValueError"}
            return False
        return True

    def fill_missing_blocks(self):
        if not self.blocks or any(not any(block.phone_id == item for block in self.blocks)
                             for item in self.state["selected"] if any(s.phone_id == item for s in self.sources)):
            # Extractive source mode is usable without keys, but is labelled honestly.
            topics = [topic for topic, words in TOPIC_WORDS.items()
                      if any(word in self.state["message"] for word in words)] or self.profile.priorities
            if self.detail_followup:
                topics = self.focus_topics
            elif self.price_intent:
                topics = list(self.profile.priorities) or topics
            for phone_id in self.state["selected"]:
                self.fallback_for_phone(phone_id, topics)

    def fallback_for_phone(self, phone_id, topics):
        if any(block.phone_id == phone_id for block in self.blocks):
            return
        self.gaming = "游戏" in topics
        available = sorted((source for source in self.sources if source.phone_id == phone_id),
                           key=lambda source: (
                               {"review": 0, "price": 1}.get(source.evidence_kind, 2) if self.gaming else 0,
                               {"page": 0, "knowledge": 1, "snippet": 2}[source.source_type]))
        checked, _ = card_matches(phone_id, self.evidence_profile, available)
        if self.detail_followup:
            if self.fallback_aspects(phone_id, topics, available):
                return
            observations=[(source,part) for topic in topics
                          for source,part in topic_observations(phone_id,topic,available,self.answer_games)]
            if observations:
                source=observations[0][0]
                parts=list(dict.fromkeys(part for item,part in observations if item.id==source.id))
                facts=configuration_facts(self.catalog[phone_id],topics,available,limit=2)
                configuration=('；已确认的相关参数：'+'、'.join(value for value,_ in facts)) if facts else ''
                boundary='这是来源中的样张观察，不能直接保证本轮场景下的成片效果。' if '拍照' in topics else ''
                self.blocks.append(AnswerBlock(phone_id=phone_id,text='资料记录：'+'；'.join(parts[:2])[:180]+configuration+'。'+boundary,
                    evidence_ids=list(dict.fromkeys([source.id,*[sid for _,sid in facts]]))))
                return
            facts=configuration_facts(self.catalog[phone_id],topics,available)
            if facts:
                self.blocks.append(AnswerBlock(phone_id=phone_id,text='针对'+self.response_focus+'，可参考的配置是'+
                    '、'.join(value for value,_ in facts)+'。',evidence_ids=list(dict.fromkeys(sid for _,sid in facts))))
                self.configuration_only.append(phone_id)
                return
            self.unanswered_refinement.append(phone_id)
            return
        if self.price_intent and not self.recommending:
            price=quoted_price(self.catalog[phone_id],self.profile,available)
            if price:
                self.blocks.append(AnswerBlock(phone_id=phone_id,text=price[0]+'。',evidence_ids=[price[1]]))
                return
        if self.recommending:
            summary = configuration_block(self.catalog[phone_id], self.profile, available)
            if summary:
                text, ids = summary
                observation = next((label.split("实测资料 · ", 1)[1] for label in checked
                                    if "实测资料 · " in label), None)
                if observation and len(text) < 105:
                    text += "测试记录：" + observation[:65] + ("…" if len(observation) > 65 else "") + "。"
                    ids += [s.id for s in available if s.evidence_kind == "review" and s.source_type != "snippet"][:1]
                self.blocks.append(AnswerBlock(phone_id=phone_id, text=text, evidence_ids=list(dict.fromkeys(ids))))
                return
            if (not self.profile.explicit_candidate_ids and not self.profile.focus_phone_id
                and not any('实测资料' in label for label in checked)):
                self.unusable_ids.append(phone_id)
                return
        if self.price_intent and available and topics and topics[0] in {"拍照", "续航"}:
            # A review can contain both tests and spec cells. Reuse the
            # per-dimension checks instead of copying its first lines.
            statements = []
            for topic in topics[:2]:
                label = next((item for item in checked if item.startswith(topic + "：")), None)
                if label and "实测资料 · " in label:
                    observation = label.split("实测资料 · ", 1)[1]
                    statements.append(topic + "测试记录：" + observation[:65] + ("…" if len(observation) > 65 else ""))
                elif topic in {"拍照", "续航"}:
                    statements.append(topic + "还缺对应实测")
            if statements:
                text = self.catalog[phone_id].name + "：" + "；".join(statements) + "。"
                if not any(s.evidence_kind == "price" and s.reference_price for s in available):
                    text += "售价待核对。"
                self.blocks.append(AnswerBlock(phone_id=phone_id, text=text,
                    evidence_ids=[s.id for s in available if s.source_type != "snippet"][:4] or [available[0].id]))
                return
        experience = (self.gaming or bool(re.search(r"实测|效果|怎么样|夜景|拍人|帧率|发热|多久", self.state["message"]))
            or self.price_intent and any(t in {"拍照", "续航", "游戏"} for t in topics))
        if experience and available and not any("实测资料" in label or "有实测资料" in label for label in checked):
            topics_missing = "、".join(t for t in topics if t in {"拍照", "续航", "游戏"}) or "当前问题"
            price_limit = ""
            phone = self.catalog[phone_id]
            if self.price_intent and phone.price and self.profile.budget_max and phone.price > self.profile.budget_max:
                price_limit = f"该版本页面参考价超出{self.profile.budget_max}元预算{phone.price-self.profile.budget_max}元。"
            self.blocks.append(AnswerBlock(phone_id=phone_id,
                text=f"{phone.name}：还缺{topics_missing}的对应实测，暂作待核对备选。{price_limit}",
                evidence_ids=[available[0].id]))
            return
        game_reviews=[source for source in available if source.evidence_kind=='review'
            and any(not self.answer_games or any(game_in_text(game,part) for game in self.answer_games)
                    for part in measured_game_parts(source.quote))]
        selected_sources = game_reviews[:1] if self.gaming and game_reviews else available[:1]
        if self.gaming and selected_sources and selected_sources[0].evidence_kind == "review":
            price = next((source for source in available if source.evidence_kind == "price"), None)
            if price:
                selected_sources.append(price)
        for source in selected_sources[:1]:
            sentences = re.split(r"(?<=[。；])", source.quote)
            relevant = [sentence for sentence in sentences if any(
                word in sentence for topic in topics for word in TOPIC_WORDS[topic])]
            # Measured conditions and price dates must survive extractive fallback.
            excerpt = source.quote if self.gaming and source.evidence_kind in {"review", "price"} else (
                "".join(relevant) if relevant else "".join(sentences[:2]))
            if self.gaming and source.evidence_kind == "review":
                parts = measured_game_parts(source.quote)
                chosen_parts = []
                for game in self.answer_games:
                    part = next((p for p in parts if game_in_text(game, p)), None)
                    if part and part not in chosen_parts:
                        chosen_parts.append(part)
                excerpt = (f"{source.published_on}发布的测试记录：" if source.published_on else "") + " / ".join(chosen_parts or parts[:2])
            label = "搜索摘要，尚未核对正文：" if source.source_type == "snippet" else "资料摘录："
            direction = ""
            if self.gaming and source.evidence_kind == "review":
                direction = "游戏初筛：有对应游戏实测，可优先核对。"
                if self.catalog[phone_id].price is None:
                    direction += "购买前先确认具体版本售价在预算内。"
            elif self.gaming and source.evidence_kind == "price":
                price = self.catalog[phone_id].price
                if price is not None and self.profile.budget_max and price > self.profile.budget_max:
                    direction = f"预算取舍：页面参考价超出{self.profile.budget_max}元预算{price - self.profile.budget_max}元。"
            # Keep full passages in sources; fallback is a short excerpt,
            # not a duplicate of the expandable source document.
            short = re.split(r"(?<=[。；])", excerpt or source.quote)
            excerpt = "".join(short[:2])
            if len(excerpt) > 180:
                excerpt = excerpt[:175] + "…"
            self.blocks.append(AnswerBlock(phone_id=source.phone_id,
                                      text=(self.catalog[phone_id].name + "：" + direction + label + excerpt),
                                      evidence_ids=[source.id]))

    def fallback_aspects(self, phone_id, topics, available):
        """Preserve multiple quoted concerns when generation/tools cannot answer."""
        aspects = [s.strip() for s in re.split(r'[、，,；;]', self.response_focus) if s.strip()]
        if not 2 <= len(aspects) <= 4 or any(len(s) > 60 for s in aspects):
            return False
        by_id = {s.id: s for s in self.registered_sources}
        blocks = []
        for aspect in aspects:
            aspect_topics = [t for t in topics if any(w in aspect for w in TOPIC_WORDS[t])] or topics
            observations = [(s, part) for t in aspect_topics for s, part in topic_observations(
                phone_id, t, available, self.answer_games)]
            # Lexical overlap selects an excerpt, never establishes that a
            # neighbouring scene was measured. Keep the source's scope below.
            terms = {aspect[i:i+2] for i in range(len(aspect)-1)}
            ranked = sorted(observations, key=lambda pair: -sum(term in pair[1] for term in terms))
            observation = next(((s, part) for s, part in ranked if any(term in part for term in terms)), None)
            facts = configuration_facts(self.catalog[phone_id], aspect_topics, available, limit=2)
            ids, parts = [], []
            if observation:
                source, part = observation
                parts.append('可参考的来源观察：'+part.strip())
                ids.append(source.id)
                if '资料边界：' in source.quote:
                    parts.append('来源边界：'+source.quote.split('资料边界：', 1)[1].split('；', 1)[0].strip())
            if facts:
                parts.append('已确认配置：'+'、'.join(value for value, _ in facts))
                ids.extend(sid for _, sid in facts)
            if not parts:
                continue
            parts.append('尚未核实到该题材或负载的对应测试，不能据此推断实际表现')
            candidate = AnswerBlock(phone_id=phone_id, text=aspect+'：'+'；'.join(parts)+'。',
                                    evidence_ids=list(dict.fromkeys(ids)))
            if self.validate_block(candidate, by_id):
                blocks.append(candidate)
        if len(blocks) != len(aspects):
            return False
        self.blocks.extend(blocks)
        return True

    def build_reply(self):
        self.blocks.sort(key=lambda block: self.state["selected"].index(block.phone_id))
        phone_count = len({block.phone_id for block in self.blocks})
        for block in self.blocks:
            if comparison_scope_mismatch(block.text,phone_count):
                block.text=''.join(clause for clause in re.split(r'(?<=[，,。；;])',block.text)
                    if not comparison_scope_mismatch(clause,phone_count)).strip('，,；; ')
        if self.unusable_ids:
            self.cards=[card for card in self.cards if card.id not in self.unusable_ids]
        intro = (f"根据你的需求，先推荐这{phone_count}款供你比较：" if phone_count > 1 else
                 "根据你的需求，可以先看这款：") if self.recommending and self.blocks else ""
        if self.recommending and self.blocks and phone_count < self.profile.recommendation_count:
            intro = f"先给你这{phone_count}款有资料可参考的备选；其他候选还没核实齐："
        if self.recommending and self.blocks and len(self.profile.priorities) > 1:
            intro = f"结合{'、'.join(self.profile.priorities)}需求，先推荐这{phone_count}款备选："
        if self.detail_followup and self.blocks:
            intro='按'+self.response_focus+('来继续看当前这几款：' if phone_count>1 else '来继续看这款：')
            if self.configuration_only:
                intro+='现有资料未包含“'+self.state['message'][:100]+'”对应的实测，下面仅列已确认配置，不能据此判断实际表现或持续使用情况。'
            elif self.gaming and self.answer_games and any(not topic_observations(
                    pid, '游戏', self.registered_sources, self.answer_games) for pid in self.state['selected']):
                durations = re.findall(r'(?:\d+(?:\.\d+)?|[一二两三四五六七八九十百半]+)\s*(?:分钟|小时)',
                                       ' '.join([self.state['message'], *self.profile.requirements]))
                intro += '部分候选未取得'+ '、'.join(self.answer_games)+('在'+ '、'.join(dict.fromkeys(durations))+'负载下' if durations else '')+'的对应实测，不能推算实际帧率或耗电。'
        if self.detail_followup and not self.blocks:
            intro='已记住'+self.response_focus+'；这轮还没有可引用的新信息。'
        if not self.blocks and self.unusable_ids:
            intro='目前还没核实到同时符合预算、硬条件和用途的机型，暂时无法给出有依据的推荐。'
        if any("最多" in item for item in self.state["acknowledged"]):
            intro = "；".join(item for item in self.state["acknowledged"] if "最多" in item) + "。"
        details = []
        for pid,gaps in self.state.get('recommendation_gaps',{}).items():
            details.append(self.catalog[pid].name+'：'+'；'.join(gaps))
        if self.unanswered_refinement:
            details.append('、'.join(self.catalog[pid].name for pid in self.unanswered_refinement)+'本轮没有可引用的对应细节资料，未用价格或无关资料代替。')
        explicit_test_question=bool(re.search(r'实测|测试|帧率|多少帧|发热|温度|耗电|夜景|实拍|成片|怎么样|多久',self.state['message']))
        if self.recommending or self.detail_followup or not explicit_test_question:
            for block in self.blocks:
                block.text,notices=presentation_copy(block.text, hide_test_gaps=not self.detail_followup)
                details.extend(notices)
        if self.recommending and self.profile.budget_max and self.profile.price_preference!='save_money':
            for block in self.blocks:
                if budget_position(self.catalog[block.phone_id].price,self.profile)==2 and '省钱' not in block.text:
                    block.text='省钱备选：'+block.text
            if self.blocks and all(budget_position(self.catalog[b.phone_id].price,self.profile)==2 for b in self.blocks):
                intro=f'这轮先列出这{len(self.blocks)}款省钱备选：'
                details.append('当前候选尚未覆盖预算附近的价位，低价方案不代表该预算档的全部选择。')
        if self.recommending and any("待补实测" in label for card in self.cards for label in card.matches):
            details.append("部分介绍依据配置做初筛，未取得对应实测；芯片、像素、防抖和电池容量不代表实际帧率、成片或使用时长。")
        if self.state.get("answer_limited"):
            details.append("本轮达到执行用量或次数边界，先按已核对资料回答；模型服务未被判定为故障。")
        if self.profile.focus_phone_id:
            if self.catalog[self.profile.focus_phone_id].price is None:
                details.append("当前售价未核对，还不能判断是否符合预算。")
        else:
            if any(self.catalog[item].price is None for item in self.state["selected"]):
                details.append("部分候选的当前售价未核对，不能确认是否符合预算。")
        if not self.sources:
            intro += "这轮还没有找到能支撑推荐的资料，暂不把旧候选当作推荐。"
            details.append("当前检索没有相关资料，不能给出配置或购买结论。")
        else:
            missing = [self.catalog[item].name for item in self.state["selected"] if not any(s.phone_id == item for s in self.sources)]
            if missing:
                intro += "未查到" + "、".join(missing) + "的完整型号资料，暂不能评价这款。"
        if any(self.catalog[item].price is not None for item in self.state["selected"]):
            details.append("价格来自有日期的页面展示记录，下单前需要核对。")
        self.gaming = "游戏" in self.profile.priorities or bool(self.profile.games)
        topic = self.profile.priorities[0] if self.profile.priorities else "配置"
        followup_card = next((card for card in self.cards if card.id == self.profile.focus_phone_id),None) or next((card for card in self.cards if self.gaming and any(
            source.phone_id == card.id and source.evidence_kind == "review" for source in self.sources)), self.cards[0] if self.cards else None)
        game_question = ('玩'+ '、'.join(self.profile.games)+'的帧率和发热怎么样？' if self.profile.games else '的游戏帧率和发热怎么样？')
        followup = (followup_card.name+game_question if self.gaming else
                    f"{self.cards[0].name}的{topic}怎么样？") if self.cards else (
                        "继续查" + "、".join(self.profile.games) + "的帧率和发热" if self.profile.games else
                        f"继续查{topic}的相关证据")
        recommend = f"按预算{self.profile.budget_max}重新推荐" if self.profile.budget_max else "按当前条件重新推荐"
        if self.gaming:
            recommend += "游戏手机"
        suggestions = [followup, "换几款看看" if self.cards and not self.profile.explicit_candidate_ids else recommend]
        if len(self.cards) > 1:
            suggestions.append(f"不考虑{self.cards[-1].name}了")
        reply = ChatMessage(role="assistant", content=intro, blocks=self.blocks,
                            followup=conversation_guidance(self.profile) if self.recommending and self.blocks else "",
                            sources=self.sources, cards=self.cards, suggestions=suggestions, mode=self.assistant.mode(self.state), details=details)
        return {"reply": reply}

