"""Model-directed tools inside the existing multi-turn selection conversation."""

import json
import logging
import re
from time import monotonic
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from .chat_schemas import ChatMessage, ChatSource, AnswerBlock
from .agent_tools import TOOL_DEFINITIONS
from .retrieval import exact_model_in_text, focus_aliases
from .tool_loop import BudgetedJsonClient, build_tool_loop, PROJECT_LIMITS, project_limit_notice, ExecutionLimit
from .selection_requirements import discovery_needed, missing_evidence, local_only, card_matches, recommendation_gaps, recommendation_notes, topic_observations, check_requirement
from .source_retention import retained_sources
from .evidence_registry import session_evidence, merge_sources, refresh_kinds, finish_refresh, missing_reply, evidence_gaps
from .selection_catalog import budget_position, minimum_storage, storage_gb
from .selection_catalog import eligible, variant_eligible
from .phone_names import parse_phone_names, register_phone, checked_today
from .model_schema import compact_schema
from .phone_identity import identity_key as normalized_name, source_identity_matches, series_excluded
from .selection_presentation import configuration_facts, quoted_price
from .source_rules import official_phone_url
from .selection_prices import select_offer, source_price_options, price_lookup_notes, historical_price_references
from .conversation_context import recent_dialogue, focused_games, query_memory, evidence_excerpt
from .config import get_settings

from .chat_prompts import CHAT_AGENT_INSTRUCTIONS, COMPACT_AGENT_INSTRUCTIONS, CANDIDATE_SELECTION_INSTRUCTIONS, PROMPT_VERSION

logger = logging.getLogger(__name__)

class ChatDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["answer", "clarify", "acknowledge"]
    message: str = Field(default="", max_length=600)
    candidate_names: list[str] | None = Field(default=None, max_length=5)
    clarification_target: Literal["budget", "usage", "phone", "requirement", "other"] | None = None
    blocks: list[dict] | None = Field(default=None,max_length=5)


def pending_page_links(actions, catalog, recommending):
    """Keep useful links for each model when completed exchanges are compacted."""
    fetched = {a.get("arguments", {}).get("url") for a in actions if a.get("name") == "fetch_page"}
    fetched.update(step.get('url') for a in actions if a.get('name')=='fetch_page'
                   for step in a.get('result',{}).get('lookup_steps',[]))
    fetched.update(a['result']['resolved_url'] for a in actions if a.get('name')=='fetch_page'
                   and a.get('result',{}).get('resolved_url'))
    pending = {}
    for sequence, action in enumerate(actions):
        if action.get("action") != "tool":
            continue
        for candidate in action.get("result", {}).get("candidates", []):
            name = candidate.get("phone_name") or action.get("arguments", {}).get("phone_name")
            url = candidate["url"]
            if not name or url in fetched:
                continue
            phone = next((p for p in catalog.values() if normalized_name(p.name) == normalized_name(name)), None)
            if phone and not source_identity_matches(phone.name,candidate.get('title',''),''):continue
            official = phone and official_phone_url(phone.brand, url)
            spec = bool(re.search(r"参数|规格|spec|param", candidate.get("title", "") + url, re.I))
            review = bool(re.search(r"评测|实测|article|review", candidate.get("title", "") + url, re.I))
            navigation = candidate.get("purpose") == "official_directory"
            resolved_target = action.get('result', {}).get('status') == 'redirect_only'
            rank = ((0 if official and spec else 1 if official and not navigation else 2 if spec else 4 if navigation else 3)
                    if recommending else (0 if review else 1 if spec else 2))
            if resolved_target:
                rank = -1  # Preserve the newly observed destination before old wrappers.
            pending[(normalized_name(name),url)] = (rank,-sequence,{"phone_name":name,"title":candidate.get("title", "")[:100],"url":url})
    selected, counts = [], {}
    for _, _, candidate in sorted(pending.values(),key=lambda p:p[:2]):
        name = normalized_name(candidate['phone_name'])
        if counts.get(name,0) < 2:
            selected.append(candidate);counts[name]=counts.get(name,0)+1
        if len(selected) >= 8:
            break
    return selected


class ChatAgentRun:
    """One turn: explicit dependencies, tool state and independently callable stages."""

    def __init__(self, assistant, turn):
        self.assistant = assistant
        self.turn = turn
        self.response_focus = turn.get('response_focus') or turn['message']
        self.client, self.limits = self.assistant.model, self.assistant.agent_policy
        self.catalog = self.turn["catalog"]
        self.scope = [] if self.turn.get("clarification") else list(self.turn["selected"])
        self.profile = self.turn["session"].profile
        self.focus_games = focused_games(self.profile, self.turn)
        self.locked_scope = not self.turn.get('recommendation_turn', True)
        self.detail_response = self.locked_scope and bool(self.scope)
        self.alternative_ids = set(self.turn.get("alternative_ids", []))
        self.scope = [pid for pid in self.scope if pid not in self.alternative_ids]
        self.offline_only = self.turn.get("search_mode") == "local" or local_only(self.turn["message"])
        self.needs_discovery = self.turn.get("recommendation_turn",True) and not self.offline_only and not self.turn.get("clarification") and (bool(self.alternative_ids) or discovery_needed(self.profile, self.catalog))
        if self.needs_discovery:
            self.scope = []
        # Discovered leads remain queryable in later turns even if they were not yet
        # suitable recommendations. Old below-tier defaults do not consume this turn.
        for phone in self.turn["session"].discovered_phones:
            if (not self.locked_scope and phone.id not in self.alternative_ids and not self.turn.get("clarification") and self.phone_allowed(self.catalog[phone.id])
                and (phone.price is None or not self.profile.budget_max or phone.price <= self.profile.budget_max)
                and (self.profile.price_preference != "near_budget" or phone.price is None or not self.profile.budget_max or phone.price >= self.profile.budget_max * .8)
                and phone.id not in self.scope):
                self.scope.append(phone.id)
        self.local_names = {normalized_name(chunk.phone_name) for chunk in self.assistant.knowledge.index.chunks}
        self.requested_dimensions = focus_aliases(" ".join([*self.profile.priorities, *self.profile.games,
                                                       "屏幕" if self.profile.high_refresh_required else ""]))

        self.refresh_kinds = refresh_kinds(self.turn["message"])
        self.refresh = bool(self.refresh_kinds)


    def phone_allowed(self,phone):
        return eligible(phone,self.profile) and not series_excluded(phone.name,self.turn.get('scope_series',[]))

    def relevant_body(self, pid, sources):
        # Relevance admits a provisional comparison; it never proves a match.
        # A gaming-only article must not fill a photography recommendation slot.
        return (bool(configuration_facts(self.catalog[pid],self.profile.priorities,sources)) or
                any(s.phone_id == pid and s.source_type != "snippet" and (not self.requested_dimensions or
                    any(alias in s.quote for group in self.requested_dimensions for alias in group)) for s in sources))

    def schemas(self, _):
        definitions = []
        for name, (arguments, description) in TOOL_DEFINITIONS.items():
            if self.locked_scope and name in {'discover_phones','consider_phones'}:
                continue
            if self.offline_only and name in {"web_search", "discover_phones", "fetch_page"}:
                continue
            search_exhausted = _.get("search_calls",0) >= self.limits.max_search_calls
            fetch_exhausted = _.get("fetch_calls",0) >= self.limits.max_fetch_calls
            if name == "web_search" and search_exhausted or name == "discover_phones" and search_exhausted and fetch_exhausted:
                continue
            if name == "fetch_page" and fetch_exhausted:
                continue
            parameters = compact_schema(arguments.model_json_schema())
            if name == "discover_phones" and search_exhausted:
                parameters['required'] = [*parameters.get('required', []), 'url']
                parameters['properties']['url'] = {'type':'string','minLength':1,'maxLength':2000,
                    'description':'搜索额度已用完，只能打开先前发现的目录URL'}
            elif name == "discover_phones" and fetch_exhausted:
                parameters['properties']['url'] = {'type':'null','description':'网页读取额度已用完，仅可搜索'}
            if name == "search_phone_knowledge":
                available = [self.catalog[pid].name for pid in self.scope if normalized_name(self.catalog[pid].name) in self.local_names]
                if not available:
                    continue
                parameters["properties"]["phone_name"]["enum"] = available
            definitions.append({"type": "function", "function": {"name": name, "description": description,
                                "parameters": parameters}})
        return definitions

    def initialize(self, _):
        self.history_actions = [action for message in self.turn['session'].messages[-get_settings().chat_history_messages:] if message.role == 'assistant'
                                for action in (message.agent_trace or {}).get('actions', [])]
        pending_history = [c for c in pending_page_links(self.history_actions, self.catalog, self.turn.get('recommendation_turn',False))
                           if any(c['phone_name'] == self.catalog[pid].name for pid in self.scope)]
        reused = session_evidence(self.turn,self.catalog,self.scope)
        previous = reused['sources']
        conditions = self.turn["session"].profile.model_dump()
        if self.needs_discovery:
            conditions["candidate_ids"] = []
        context = {"本轮": self.turn["message"], "本轮日期": checked_today(), "条件": conditions,
                   "可查询机型": {pid: self.catalog[pid].name for pid in self.scope},
                   "澄清提示": self.turn.get("clarification") or None,
                   "最近对话": recent_dialogue(self.turn['session'].messages),
                   "本轮关注游戏": self.focus_games,
                   "已有查询反馈": query_memory(self.turn['session'].messages, {self.catalog[pid].name for pid in self.scope}),
                   "可继续打开的同型号链接": pending_history,
                   "可复用来源": [{**s.model_dump(), "quote": evidence_excerpt(s.quote, 900 if s.evidence_kind == "review" else 350)}
                              for s in previous.values()],
                   "已有证据核对": {self.catalog[pid].name: card_matches(pid, self.profile, list(previous.values()))[0] for pid in self.scope}}
        context['本轮刷新字段']=sorted(self.refresh_kinds)
        context['本轮执行额度']={'工具总次数':self.limits.max_tool_calls,'模型总次数含最终回答':self.limits.max_model_calls,
            '公开搜索次数':self.limits.max_search_calls,'网页读取次数':self.limits.max_fetch_calls}
        context["候选价位覆盖不足"] = self.needs_discovery
        context["本地有资料的机型"] = [self.catalog[pid].name for pid in self.scope if normalized_name(self.catalog[pid].name) in self.local_names]
        context["暂避上一批"] = [self.catalog[pid].name for pid in self.alternative_ids if pid in self.catalog]
        context['本轮暂避系列']=self.turn.get('scope_series',[])
        context["必须核对"] = {"游戏": self.profile.games, "高刷": self.profile.high_refresh_required,
                              "推荐数量": self.profile.recommendation_count, "仅本地": self.offline_only,
                              "用户开启联网搜索": self.turn.get("search_mode") == "web" and not self.offline_only}
        context["查询选择"] = ("用户已开启联网搜索。本轮结束前请按完整需求选择一次有价值的公开查询；具体工具、查询词和网页由你决定，不必重复查已有事实。" if self.turn.get("search_mode") == "web" and not self.offline_only else
                              "只使用本地和已有资料。" if self.offline_only else "按需决定是否补查公开资料。")
        context['本轮对话动作']=self.turn.get('conversation_route')
        context['本轮是细节追问']=self.detail_response
        if self.detail_response:
            context['回答重点']='用户细化了'+self.response_focus+'，接着回答当前候选与此细节相关的证据或参数，保留其他条件。不要重播通用推荐、报价或查新价位榜单；不要把参数写成实测。'
        context['预算定位']='省钱优先' if self.profile.price_preference=='save_money' else '优先覆盖预算所在价位，低价仅作有理由的省钱备选；不要用整批低价旧型号替代这个价位。'
        usage = self.turn.get("intent_usage")
        return {"messages": [SystemMessage(content=CHAT_AGENT_INSTRUCTIONS),
                             HumanMessage(content=json.dumps(context, ensure_ascii=False))],
                "sources": previous, "allowed_urls": {(s.phone_id, s.url) for s in previous.values()} |
                    {(pid,c['url']) for c in pending_history for pid in self.scope if self.catalog[pid].name==c['phone_name']} |
                    {(pid, str(url)) for pid in self.scope for url in self.catalog[pid].source_urls},
                "executed": [], "tool_calls_used": 0, "model_calls": 1,
                "charged_tokens": usage.get("total_tokens", 2000) if usage else 2000,
                "usage": [usage], "no_progress": 0, "stop_reason": "",
                "actions": [{"action": "conversation_route", **self.turn.get('conversation_route', {})}] + ([{"action": "intent_fallback", "error_type": self.turn["intent_failure"]}]
                            if self.turn.get("intent_failure") else []) +
                           ([{"action": "intent_grounding_repair", "fields": self.turn['intent_repairs']}]
                            if self.turn.get('intent_repairs') else []) +
                           ([{"action": "reuse_sources", "source_ids": list(previous)}] if previous else []),
                "started_at": self.turn.get("turn_started_at", monotonic()), "nodes": ["initialize"],
                "search_calls": 0, "fetch_calls": 0, "knowledge_calls": 0, "discovery_urls": [],'local_archive_gaps':{}}

    def execute(self, state, name, arguments):
        if name not in TOOL_DEFINITIONS:
            return {"status": "invalid_tool"}
        if self.locked_scope and name in {'discover_phones', 'consider_phones'}:
            return {"status": "invalid_tool", "error": "本轮细化当前候选的用途，请围绕当前候选查询。"}
        try:
            args = TOOL_DEFINITIONS[name][0].model_validate(arguments)
        except (ValueError, TypeError):
            return {"status": "invalid_arguments"}
        lookup = ('fetch' if name == 'fetch_page' or name == 'discover_phones' and args.url
                  else 'search' if name in {'web_search','discover_phones'} else None)
        if lookup and state[lookup+'_calls'] >= getattr(self.limits,'max_'+lookup+'_calls'):
            return {"status":"lookup_limit","error":"本轮对应的公开查询额度已用完，请复用现有资料。"}
        if self.offline_only and name in {"discover_phones", "web_search", "fetch_page"}:
            return {"status": "local_only", "error": "用户本轮要求不联网"}
        handlers = {
            "search_local_phones": self.search_local_phones,
            "consider_phones": self.consider_phones,
            "discover_phones": self.discover_phones,
        }
        handler = handlers.get(name)
        return handler(state, args) if handler else self.execute_scoped(state, name, args)

    def search_local_phones(self, state, args):
        key=json.dumps(['search_local_phones',args.query],ensure_ascii=False)
        if key in state['executed']:
            return {"status":"duplicate"}
        state['executed'].append(key);state['knowledge_calls']+=1
        hits=self.assistant.knowledge.discover(" ".join([args.query,*self.profile.priorities,*self.profile.games,*self.profile.requirements]))
        names=list(dict.fromkeys(h['phone_name'] for h in hits))
        if self.turn.get('recommendation_turn'):
            # Price records and useful configurations can have no lexical hit
            # for a short homepage query. Inspect the bounded inventory too.
            names=list(dict.fromkeys([*names,*(c.phone_name for c in self.assistant.knowledge.index.chunks)]))
        if self.locked_scope:
            names=[n for n in names if any(normalized_name(self.catalog[pid].name)==normalized_name(n) for pid in self.scope)]
        def price_rank(name):
            known=next((p.price for p in self.catalog.values() if normalized_name(p.name)==normalized_name(name)),None)
            offers=[c.reference_price for c in self.assistant.knowledge.index.chunks if normalized_name(c.phone_name)==normalized_name(name)
                and c.evidence_kind=='price' and c.reference_price and (not self.profile.budget_max or c.reference_price<=self.profile.budget_max)
                and (storage_gb(c.price_variant or '') or 0)>=minimum_storage(self.profile)]
            price=(max(offers) if self.profile.price_preference=='near_budget' else min(offers)) if offers else known
            position=budget_position(price,self.profile)
            rank={0:0,2:1,1:2,3:3}[position]
            return (rank,price or 100000) if self.profile.price_preference=='save_money' else (rank,)
        if self.turn.get('recommendation_turn'):names.sort(key=price_rank)
        returned=[];selected=[];pending=[]; screened_out=[]
        original_scope=set(self.scope)
        original_sources={sid:source.model_dump() for sid,source in state['sources'].items()}
        for candidate in names:
            mentions=parse_phone_names(candidate)
            if len(mentions)!=1:continue
            phone=register_phone(mentions[0],self.catalog)
            if phone.id in self.alternative_ids or not self.phone_allowed(phone):continue
            if phone.id not in self.catalog:
                if len(self.turn['session'].discovered_phones)>=20:continue
                self.catalog[phone.id]=phone
            ranked=[h for h in hits if h['phone_name']==candidate]
            pool=[*ranked,*({'text':c.text,**c.model_dump(exclude={'text','vector'})}
                for c in self.assistant.knowledge.index.chunks if normalized_name(c.phone_name)==normalized_name(candidate))]
            def source_from_hit(hit):
                return ChatSource(id=f"{phone.id}:{hit['chunk_id']}",phone_id=phone.id,title=hit['title'],url=hit['source_url'],
                    quote=hit['text'],source_type='knowledge',checked_on=hit.get('checked_on') or checked_today(),
                    market=hit.get('market') or '地区未确认',evidence_kind=hit.get('evidence_kind','unknown'),
                    reference_price=hit.get('reference_price'),price_variant=hit.get('price_variant'),
                    published_on=hit.get('published_on'),price_options=[{'variant':hit['price_variant'],'price':hit['reference_price']}]
                    if hit.get('price_variant') and hit.get('reference_price') else [])
            related=[]
            for topic in self.profile.priorities:
                found=next((h for h in pool if configuration_facts(phone,[topic],[source_from_hit(h)])),None)
                if found and all(h['chunk_id']!=found['chunk_id'] for h in related):related.append(found)
                review=next((h for h in pool if topic_observations(phone.id,topic,[source_from_hit(h)],self.profile.games)),None)
                if review and all(h['chunk_id']!=review['chunk_id'] for h in related):related.append(review)
            for requirement in [*self.profile.requirements,*(['刷新率至少90Hz'] if self.profile.high_refresh_required else [])]:
                found=next((h for h in pool if check_requirement(requirement,[source_from_hit(h)]).status!='unknown'),None)
                if found and all(h['chunk_id']!=found['chunk_id'] for h in related):related.append(found)
            prices=[c for c in self.assistant.knowledge.index.chunks if normalized_name(c.phone_name)==normalized_name(candidate)
                    and c.evidence_kind=='price' and c.reference_price]
            prices.sort(key=lambda c:(bool(minimum_storage(self.profile) and (storage_gb(c.price_variant or '') or 0)<minimum_storage(self.profile)),
                bool(self.profile.budget_max and c.reference_price>self.profile.budget_max),
                -(int(c.checked_on.replace('-','')) if c.checked_on else 0),
                abs(c.reference_price-(self.profile.budget_max or c.reference_price)) if self.profile.price_preference=='near_budget' else c.reference_price))
            price_hits=[{'text':c.text,**c.model_dump(exclude={'text','vector'})} for c in prices[:2]]
            related=[*price_hits,*related]
            related += [h for h in ranked if all(old['chunk_id']!=h['chunk_id'] for old in related)][:max(0,8-len(related))]
            incoming=list({source.id:source for source in (source_from_hit(h) for h in related)}.values())[:8]
            # Resolve offers locally before deciding which results deserve the
            # bounded candidate/evidence slots. Do not fill them with unknowns.
            offers=[offer for source in incoming if source.market=='中国大陆' for offer in source_price_options(source)]
            if offers:
                chosen=select_offer(offers,self.profile)
                phone.price,phone.variant=chosen.price,chosen.variant
                phone.price_options=offers[:12]
                self.catalog[phone.id]=phone
            phone=self.catalog[phone.id]
            if not variant_eligible(phone,self.profile) or any('不符合' in label for label in card_matches(phone.id,self.profile,incoming)[0]):
                screened_out.append({'name':phone.name,'status':'unsatisfied',
                    'reasons':recommendation_gaps(phone,self.profile,incoming)})
                continue
            if phone.price and self.profile.budget_max and (phone.price>self.profile.budget_max or
                self.profile.price_preference=='near_budget' and phone.price<self.profile.budget_max*.8):continue
            gaps=recommendation_gaps(phone,self.profile,incoming)
            row={'phone_id':phone.id,'name':phone.name,'price':phone.price,'variant':phone.variant,
                             'budget_role':'省钱备选' if budget_position(phone.price,self.profile)==2 else '当前预算价位' if phone.price and budget_position(phone.price,self.profile)==0 else '版本价格待核对',
                             'matches':card_matches(phone.id,self.profile,incoming)[0],
                             'configuration_facts':configuration_facts(phone,self.profile.priorities,incoming),
                             'requirements':self.profile.priorities,'qualification':'pending' if gaps else 'ready',
                             'qualification_gaps':gaps}
            if gaps:
                all_body=[source_from_hit(h) for h in pool]
                local_gaps=recommendation_gaps(phone,self.profile,all_body)
                row['local_archive_gaps']=local_gaps
                state.setdefault('local_archive_gaps',{})[phone.id]=local_gaps
                if local_gaps:
                    row['local_archive_status']='已扫描该型号全部本地片段；这些缺口需新来源，重复换词不能补出未收录事实'
            if gaps:
                if self.relevant_body(phone.id,incoming) and len(pending)<args.limit:pending.append((phone,incoming,row))
                continue
            if not self.keep_local_candidate(state,phone,incoming):continue
            returned.extend(s.model_dump() for s in incoming)
            selected.append(row)
            if len(selected)>=args.limit:break
        kept_pending=[]
        for phone,incoming,row in pending:
            if not self.keep_local_candidate(state,phone,incoming):continue
            kept_pending.append(row)
            returned.extend(s.model_dump() for s in incoming)
        returned=[s for s in returned if s['id'] in state['sources']]
        return {'status':'ok' if selected or pending else 'empty','local_candidates':selected,
                'pending_candidates':kept_pending,'sources':returned,
                'screened_out':screened_out[:8], 'archive_coverage':self.assistant.knowledge.coverage(),
                'coverage_note':'仅代表当前资料能确认的候选；缺价格或实测不是产品不符合，也不是全市场排序。',
                'lookup_origin':'local','added_candidates':sum(pid not in original_scope for pid in self.scope),
                'added_evidence':sum(original_sources.get(s['id'])!=s for s in returned),
                'fact_progress':sum(row['phone_id'] not in original_scope for row in selected)}

    def keep_local_candidate(self,state,phone,sources):
        if not any(p.id==phone.id for p in self.turn['session'].discovered_phones) and phone.id not in self.assistant.catalog:
            if len(self.turn['session'].discovered_phones)>=20:return False
            self.turn['session'].discovered_phones.append(phone)
        if phone.id not in self.scope:self.scope.append(phone.id)
        for offset in range(0,min(len(sources),8),4):
            self.record_sources(state,phone.id,sources[offset:offset+4])
        return True

    def consider_phones(self, state, args):
        hypotheses, rejected = [], []
        for candidate in args.names:
            mentions = parse_phone_names(candidate)
            if len(mentions) != 1 or normalized_name(mentions[0].original) != normalized_name(candidate):
                rejected.append({"name": candidate, "reason": "无法识别为已支持品牌的完整数字型号"})
                continue
            phone = register_phone(mentions[0], self.catalog)
            if phone.id in self.alternative_ids or not self.phone_allowed(phone):
                rejected.append({"name": candidate, "reason": "与当前品牌或排除条件冲突"})
                continue
            if phone.id not in self.catalog:
                if len(self.turn["session"].discovered_phones) >= 20:
                    continue
                phone.note = "模型提出的未验证型号假设；存在、价格与用途证据均需查询正文。"
                self.catalog[phone.id] = phone
                self.turn["session"].discovered_phones.append(phone)
            elif phone.id not in self.assistant.catalog and not any(p.id==phone.id for p in self.turn['session'].discovered_phones):
                if len(self.turn['session'].discovered_phones)<20:self.turn['session'].discovered_phones.append(phone)
            if phone.id not in self.scope:
                self.scope.append(phone.id)
                hypotheses.append({"phone_name": phone.name, "status": "unverified_hypothesis",
                                   "local_knowledge_available": normalized_name(phone.name) in self.local_names})
        return {"status": "ok" if hypotheses else "empty", "hypotheses": hypotheses, "rejected": rejected,
                "added_candidates": len(hypotheses)}

    def discover_phones(self, state, args):
        key = json.dumps(['discover_phones', args.query, args.url, args.backend], ensure_ascii=False)
        if key in state["executed"]:
            return {"status": "duplicate"}
        state["executed"].append(key)
        if args.url:
            if args.url not in state["discovery_urls"]:
                return {"status": "invalid_url", "error": "目录URL必须来自先前发现工具结果"}
            state["fetch_calls"] += 1
            leads = self.assistant.source_loader.discover(args.query, budget_max=self.profile.budget_max, url=args.url,
                **({'refresh':True} if self.refresh else {}))
        else:
            state["search_calls"] += 1
            options = {"backend": args.backend} if args.backend != "bing" else {}
            leads = self.assistant.source_loader.discover(args.query, budget_max=self.profile.budget_max, **options)
        navigation = []
        for lead in leads[:10]:
            if lead["url"] not in state["discovery_urls"]:
                state["discovery_urls"].append(lead["url"])
            navigation.append({"title": lead["title"], "url": lead["url"], "snippet": lead.get("snippet", ""),
                               "status": "navigation_only"})
        candidates = []
        returned_leads = set()
        added_leads = 0
        for lead in leads[:10]:
            for mention in parse_phone_names(lead["title"] + " " + lead.get("snippet", "")):
                phone = register_phone(mention, self.catalog)
                if phone.id in self.alternative_ids or not self.phone_allowed(phone):
                    continue
                if phone.id not in self.catalog:
                    if len(self.turn["session"].discovered_phones) >= 20:
                        continue
                    phone.note = "公开搜索发现的机型线索，价格、配置和实测按来源逐项核对。"
                    self.catalog[phone.id] = phone
                    self.turn["session"].discovered_phones.append(phone)
                if phone.id not in self.scope:
                    self.scope.append(phone.id)
                lead_key = (phone.id, lead["url"])
                if lead_key in returned_leads:
                    continue
                returned_leads.add(lead_key)
                if lead_key not in state["allowed_urls"]:
                    added_leads += 1
                state["allowed_urls"].add((phone.id, lead["url"]))
                candidates.append({"phone_name": phone.name, **lead, "status": "待核对正文与价位",
                                   "local_knowledge_available": normalized_name(phone.name) in self.local_names})
        return {"status": getattr(leads,'status',None) or ("ok" if candidates else "empty"), "candidates": candidates[:12],
                "lookup_origin": getattr(leads, "lookup_origin", "unknown"),
                **{key:getattr(leads,key) for key in ('requested_url','resolved_url','redirect_chain','lookup_steps','network_operations','cache_operations','fetch_feedback') if hasattr(leads,key)},
                "navigation": navigation, "added_candidates": added_leads,
                **({"hint": "未发现可用型号；可改查询词或用url打开返回的目录/文章。"} if not candidates else {})}

    def execute_scoped(self, state, name, args):
        matches = [pid for pid in self.scope if args.phone_name.casefold() in
                   {alias.casefold() for alias in [self.catalog[pid].name, *self.catalog[pid].aliases]}]
        if len(matches) != 1:
            return {"status": "invalid_scope", "error": "只能查询本轮已确认的机型；指代不明时先澄清"}
        phone_id = matches[0]
        phone = self.catalog[phone_id]
        key = json.dumps([name, phone_id, args.model_dump()], sort_keys=True, ensure_ascii=False)
        if key in state["executed"]:
            return {"status": "duplicate"}
        state["executed"].append(key)
        incoming, candidates, added_candidates = [], [], 0
        lookup_origin = "local" if name == "search_phone_knowledge" else "unknown"
        search_feedback = None
        fetch_feedback = None
        lookup_details = {}
        lookup_status = None
        if name == "search_phone_knowledge":
            target=(phone_id,tuple(self.turn.get('task',{}).get('fields',[])),tuple(self.focus_games),tuple(self.turn.get('focus_topics',[])))
            if target in state.get('scanned_local_targets',[]):
                return {'status':'duplicate','fact_progress':0,'target_gaps':evidence_gaps(self.turn,phone,list(state['sources'].values())),
                    'local_archive_gaps':state.get('local_archive_gaps',{}).get(phone_id,[]),
                    'hint':'该型号本地档案已扫描；目标缺口没有收录，换相近词不会产生新事实。可自主换新来源或如实结束。'}
            state["knowledge_calls"] += 1
            hits = self.assistant.knowledge.as_tool().invoke({"phone_name": phone.name, "need": args.query})
            incoming = [ChatSource(id=f"{phone_id}:{hit['chunk_id']}", phone_id=phone_id,
                title=hit["title"], url=hit["source_url"], quote=hit["text"],
                checked_on=hit.get("checked_on") or phone.checked_on, published_on=hit.get("published_on"),
                market=hit.get("market") or "地区未确认", evidence_kind=hit.get("evidence_kind", "unknown"),
                reference_price=hit.get("reference_price"), price_variant=hit.get("price_variant"))
                for hit in hits[:4] if hit.get("phone_name", "").casefold() == phone.name.casefold()]
            if self.locked_scope and not configuration_facts(phone,self.turn.get('focus_topics') or self.profile.priorities,incoming):
                for chunk in self.assistant.knowledge.index.chunks:
                    if normalized_name(chunk.phone_name)!=normalized_name(phone.name):continue
                    companion=ChatSource(id=f'{phone_id}:{chunk.chunk_id}',phone_id=phone_id,title=chunk.title,url=chunk.source_url,
                        quote=chunk.text,checked_on=chunk.checked_on or checked_today(),source_type='knowledge',
                        evidence_kind=chunk.evidence_kind,market=chunk.market or '地区未确认')
                    if configuration_facts(phone,self.turn.get('focus_topics') or self.profile.priorities,[companion]):
                        incoming=[companion,*incoming][:4]
                        break
            archive=[ChatSource(id=f'{phone_id}:{c.chunk_id}',phone_id=phone_id,title=c.title,url=c.source_url,
                quote=c.text,checked_on=c.checked_on or checked_today(),evidence_kind=c.evidence_kind,
                market=c.market,reference_price=c.reference_price,price_variant=c.price_variant)
                for c in self.assistant.knowledge.index.chunks if normalized_name(c.phone_name)==normalized_name(phone.name)]
            if self.turn.get('recommendation_turn'):
                companions=[]
                for topic in self.profile.priorities:
                    match=next((s for s in archive if configuration_facts(phone,[topic],[s]) or topic_observations(phone_id,topic,[s],self.profile.games)),None)
                    if match and match.id not in {s.id for s in companions}:companions.append(match)
                price=next((s for s in archive if any(o.price==phone.price and o.variant==phone.variant for o in source_price_options(s,phone.name))),None)
                if price and price.id not in {s.id for s in companions}:companions.append(price)
                incoming=list({s.id:s for s in [*companions,*incoming]}.values())[:4]
            gaps=evidence_gaps(self.turn,phone,archive)
            if gaps:
                state.setdefault('scanned_local_targets',[]).append(target)
                state.setdefault('local_archive_gaps',{})[phone_id]=gaps
        elif name == "web_search":
            if not exact_model_in_text(phone.name, args.query):
                return {"status": "invalid_arguments", "error": "公开查询须含完整机型"}
            state["search_calls"] += 1
            options = {"backend": args.backend} if args.backend != "bing" else {}
            result = self.assistant.source_loader.search(phone, args.query, refresh=self.refresh, **options)
            lookup_status = result.get('status') if isinstance(result,dict) else None
            lookup_origin = result.get("lookup_origin", "unknown") if isinstance(result, dict) else "unknown"
            search_feedback = result.get("search_feedback") if isinstance(result, dict) else None
            incoming = result.get("sources", []) if isinstance(result, dict) else result
            candidates = result.get("candidates", []) if isinstance(result, dict) else []
            if candidates and not incoming and all(c.get('status') == 'navigation_only' for c in candidates):
                lookup_status = 'navigation_only'
            for candidate in candidates[:6]:
                key = (phone_id, candidate["url"])
                if key not in state["allowed_urls"]:
                    added_candidates += 1
                    state["allowed_urls"].add(key)
        else:
            if (phone_id, args.url) not in state["allowed_urls"]:
                return {"status": "invalid_url", "error": "URL必须来自本会话同机型工具结果"}
            state["fetch_calls"] += 1
            if hasattr(self.assistant.source_loader, "fetch_result"):
                reader = getattr(self.assistant.source_loader,'fetch_following_result',self.assistant.source_loader.fetch_result)
                result = reader(phone, args.url, args.focus or self.turn["query"], refresh=self.refresh)
                lookup_details={k:result[k] for k in ('requested_url','resolved_url','redirect_chain','lookup_steps','network_operations','cache_operations') if k in result}
                lookup_origin = result.get("lookup_origin", "unknown")
                fetch_feedback = result.get('fetch_feedback')
                lookup_status = result.get('status')
                incoming, candidates = result["sources"], result.get("candidates", [])
                for candidate in candidates[:6]:
                    key = (phone_id, candidate["url"])
                    if key not in state["allowed_urls"]:
                        added_candidates += 1
                        state["allowed_urls"].add(key)
            else:
                incoming = self.assistant.source_loader.fetch(phone, args.url, args.focus or self.turn["query"], refresh=self.refresh)
        if name in {'web_search','fetch_page'}:state.setdefault('attempted_phone_ids',set()).add(phone_id)
        before_gaps=evidence_gaps(self.turn,phone,list(state['sources'].values()))
        returned, added = self.record_sources(state, phone_id, incoming, fresh=name=="fetch_page")
        after_gaps=evidence_gaps(self.turn,phone,list(state['sources'].values()))

        return {**self.candidate_state(state, phone_id), "status": lookup_status or ("ok" if returned or candidates else "empty"),
                "sources": returned, "candidates": candidates[:6], "added_candidates": added_candidates,
                "added_evidence": added,"fact_progress":len(set(before_gaps)-set(after_gaps)),"target_gaps":after_gaps, "lookup_origin": lookup_origin, "search_feedback": search_feedback,
                **lookup_details,
                **({'fetch_feedback':fetch_feedback} if fetch_feedback else {})}

    def record_sources(self, state, phone_id, incoming, *, fresh=False):
        result=merge_sources(state,phone_id,incoming[:4],self.catalog,self.profile,
            focus_topics=self.turn.get('focus_topics',()),refresh=self.refresh_kinds,fresh=fresh)
        self.assistant._remember_sources(self.turn,list(state['sources'].values()))
        return result

    def candidate_state(self, state, phone_id):
        current = self.catalog[phone_id]
        matches, _ = card_matches(phone_id, self.profile, list(state["sources"].values()))
        facts = configuration_facts(current, self.profile.priorities, list(state["sources"].values()))
        remaining = [label for label in matches if "待" in label or "冲突" in label or "不符合" in label]
        # Test gaps remain visible, but do not consume all discovery attempts
        # when parameters already support an initial, provisional comparison.
        if self.turn.get("recommendation_turn"):
            remaining = [label for label in remaining if "待补实测" not in label]
            remaining.extend(topic + "具体配置待核对" for topic in self.profile.priorities
                if not configuration_facts(current, [topic], list(state["sources"].values())))
        if self.profile.budget_max and current.price is None:
            remaining.append("版本价格待核对")
        elif self.profile.budget_max and current.price > self.profile.budget_max:
            remaining.append(f"该版本超预算{current.price-self.profile.budget_max}元，需核对其他版本或另找机型")
        elif self.profile.price_preference == "near_budget" and self.profile.budget_max and current.price < self.profile.budget_max * .8:
            remaining.append("低于目标价位，需解释省钱取舍或另找该价位机型")

        qualification=recommendation_gaps(current,self.profile,list(state['sources'].values()))
        return {"candidate_state": {"name": current.name, "price": current.price, "variant": current.variant,
                            'target_gaps':evidence_gaps(self.turn,current,list(state['sources'].values())),
                            "qualification":'pending' if qualification else 'ready',"qualification_gaps":qualification,
                            'recommendation_notes':recommendation_notes(current,self.profile,list(state['sources'].values())),
                            'local_archive_gaps':state.get('local_archive_gaps',{}).get(phone_id,[]),
                            'requirement_checks':{r:{'status':c.status,'source_ids':list(c.source_ids),'urls':list(c.urls)}
                                for r in self.profile.requirements
                                for c in [check_requirement(r,[s for s in state['sources'].values() if s.phone_id==phone_id])]},
                            "configuration_facts": facts, "quoted_price": quoted_price(current, self.profile, list(state["sources"].values())),
                            'price_lookup_notes':price_lookup_notes(current,list(state['sources'].values())),
                            "matches": matches, "remaining": remaining}}


    def validate_finish(self, state, response):
        if self.turn.get("clarification") or self.offline_only or state.get("finish_retries", 0) >= 1:
            return None
        try:
            decision = ChatDecision.model_validate_json(response.content)
        except (ValueError, TypeError):
            return "结束格式有误，请按约定JSON返回。"
        if decision.kind == "clarify":
            known = {"budget": self.profile.budget_max is not None or self.profile.budget_unlimited,
                     "usage": bool(self.profile.priorities or self.profile.games), "phone": bool(self.profile.focus_phone_id)}
            if decision.clarification_target is None or known.get(decision.clarification_target, False):
                return "你要澄清的字段未说明或已经确认。请用clarification_target说明真正缺少的条件，或直接围绕已知机型、预算和用途查询/回答，不重复确认。"
            return None
        if decision.kind == "acknowledge" and not re.search(r"推荐|想要|想买|预算|价位|左右|用途|拍照|续航|屏幕|轻便|游戏|高刷", self.turn["message"]):
            if not self.alternative_ids:
                return None
        tools = [a.get("name") for a in state["actions"] if a.get("action") == "tool"]
        if (self.turn.get("search_mode") == "web" and not any(t in tools for t in ("web_search", "discover_phones", "fetch_page"))):
            return "用户开启了联网搜索。本轮还没有查询公开资料，请根据完整需求自行选择工具和查询内容；失败时保留已知事实，不能声称已经联网。"
        if self.needs_discovery and not any(t in tools for t in ("discover_phones", "consider_phones", "search_local_phones")):
            return "还不能完成选购：旧候选没有覆盖用户价位。请用discover_phones搜索该价位，再核对候选；不要只确认预算。"
        if self.locked_scope:
            gaps={self.catalog[pid].name:evidence_gaps(self.turn,self.catalog[pid],list(state['sources'].values())) for pid in self.scope}
            if any(gaps.values()) and not state['search_calls'] and not state['fetch_calls']:
                return '当前问题仍有缺口：'+json.dumps(gaps,ensure_ascii=False)+'。可自主补查公开来源，或在确无可用线索时说明本轮具体缺口后结束；不要用旧话题代替。'
            return None
        check_ids = self.turn["selected"]
        if decision.candidate_names is not None:
            check_ids = [pid for pid in self.scope if normalized_name(self.catalog[pid].name) in
                         {normalized_name(n) for n in decision.candidate_names}]
        if self.turn.get('recommendation_turn'):
            gaps={self.catalog[pid].name:recommendation_gaps(self.catalog[pid],self.profile,list(state['sources'].values())) for pid in check_ids}
            if any(gaps.values()):return '这些仍是待核对线索，不能占合格推荐名额：'+json.dumps(gaps,ensure_ascii=False)+'。自主选择补查或减少数量；不足时直接说明，不凑数。'
        if (self.turn.get('recommendation_turn') and self.profile.budget_max and self.profile.price_preference!='save_money'
            and check_ids and all(budget_position(self.catalog[pid].price,self.profile)==2 for pid in check_ids)):
            return '本轮只覆盖了明显低于预算价位的选择。保留有理由的省钱备选，但请用当前完整需求查找预算附近的候选；不足时说明价位覆盖缺口，不把整批低价款当作该预算档。'
        if decision.kind == "answer" and decision.candidate_names == []:
            useful = [pid for pid in self.scope if self.phone_allowed(self.catalog[pid])
                and not recommendation_gaps(self.catalog[pid],self.profile,list(state['sources'].values()))
                and self.relevant_body(pid,list(state['sources'].values()))]
            if useful:
                return "已有同时通过预算、硬条件和用途依据核对的机型，请依据已核实事实决定保留哪些；不必为了凑数量选择其他线索。"
        if (not self.profile.explicit_candidate_ids and not self.profile.focus_phone_id
            and len(check_ids) < self.profile.recommendation_count and state["tool_calls_used"] < self.limits.max_tool_calls
            and (not state["search_calls"] or not any(a.get("name") == "fetch_page" for a in state["actions"]))):
            return "用户需要多款备选，目前不足目标数量。请利用现有线索寻找其他候选并核对正文；若确实查不到，说明失败原因后结束，不能只承诺将查询。"
        required = [pid for pid in check_ids if evidence_gaps(self.turn,self.catalog[pid],list(state['sources'].values()))]
        if self.turn.get("recommendation_turn"):
            required = [pid for pid in required if
                not configuration_facts(self.catalog[pid], self.profile.priorities, list(state["sources"].values()))
                or self.catalog[pid].price is None
                or any("冲突" in label or "不符合" in label for label in card_matches(pid, self.profile, list(state["sources"].values()))[0])]
        if (required
            and not state["search_calls"]):
            return "当前用途或具体要求仍缺证据或来源冲突。请自行选择搜索词、来源并核对正文；配置不能替代实测，缺资料不等于不满足。"
        if required:
            fetched = {a.get("arguments", {}).get("url") for a in state["actions"] if a.get("name") == "fetch_page"}
            pending = [c for a in state["actions"] if a.get("action") == "tool"
                       for c in a.get("result", {}).get("candidates", []) if c.get("url") not in fetched
                       and (c.get("purpose") in {"review", "spec_price"} or re.search(r"评测|实测|参数|规格", c.get("title", "")))]
            if pending:
                return "当前主要用途仍缺证据或来源冲突；工具结果中还有未核对的评测/规格正文线索。请结合这些反馈决定继续核对、换来源或说明为何停止，不能将标题或宣传当实测。"
        return None

    def finish(self, state, budget, charge):
        updated, events = dict(state), list(state["actions"])
        finish_refresh(self.turn,state,attempted=state.get('attempted_phone_ids',set()))
        sources = list(state["sources"].values())
        self.assistant._remember_sources(self.turn, sources)

        decision = self.read_decision(state, updated, events, sources, budget, charge)
        self.select_candidates(decision, sources)
        if self.turn.get('backfilled_candidate_ids'):
            events.append({'action':'grounded_candidate_backfill','phone_ids':self.turn['backfilled_candidate_ids'],
                'reason':'requested_count_not_met; same budget, scope, body and hard-gate checks'})
        collected_sources = sources
        self.turn['background_sources']=[s for s in collected_sources if s.phone_id not in self.turn['selected'] and s.source_type!='snippet'][:4]
        if self.turn.get('recommendation_turn'):
            self.turn['qualification_rejections']={pid:gaps for pid in self.scope
                if any(s.phone_id==pid and s.source_type!='snippet' for s in collected_sources)
                if (gaps:=recommendation_gaps(self.catalog[pid],self.profile,collected_sources))}
            self.turn['recommendation_gaps']=self.turn['qualification_rejections']
        sources = [s for s in sources if s.phone_id in self.turn["selected"]]
        if not sources and not self.profile.explicit_candidate_ids and not self.profile.focus_phone_id:
            self.profile.candidate_ids = []
            self.turn["selected"] = []
        self.assistant._remember_sources(self.turn, sources)
        reply = self.compose_reply(state, decision, sources, updated, events, budget, charge)
        if self.turn.get('recommendation_turn') and not sources and any(s.source_type!='snippet' for s in collected_sources):
            gaps=list(dict.fromkeys(g for values in self.turn['qualification_rejections'].values() for g in values))
            reply.content='已查到相关机型资料，但还没有核实到同时符合预算、硬条件和各用途的'+('其他备选' if self.alternative_ids else '机型')+'。'+('当前缺口或限制：'+'；'.join(gaps[:4])+'。' if gaps else '')
            reply.details=[d for d in reply.details if '当前检索没有相关资料' not in d]
            reply.details.extend(self.catalog[pid].name+'：'+'；'.join([*gaps,*price_lookup_notes(self.catalog[pid],collected_sources)])
                for pid,gaps in self.turn.get('qualification_rejections',{}).items())
        if state["stop_reason"] not in {"model_finished", "model_error"}:
            reasons={'no_progress':'连续查询未增加可核验事实','model_call_limit':'模型调用次数达到上限','tool_call_limit':'工具调用次数达到上限','deadline':'达到查询时限','token_budget':'达到用量边界','context_limit':'达到上下文边界'}
            used=(f"（本轮工具{state['tool_calls_used']}/{self.limits.max_tool_calls}次，模型{updated['model_calls']}/{self.limits.max_model_calls}次，公开搜索{state['search_calls']}/{self.limits.max_search_calls}次，网页读取{state['fetch_calls']}/{self.limits.max_fetch_calls}次）")
            reply.details.append('本轮查询停止原因：'+reasons.get(state['stop_reason'],state['stop_reason'])+used+'；已保留查到的来源。')
        if self.turn.get("search_mode") == "web" and not self.offline_only and not state["search_calls"] and not state["fetch_calls"]:
            reply.details.append("本轮开启了联网搜索，但查询未执行完成；当前回答仍依据已有资料，原因见查询过程。")
        self.explain_public_lookup(reply, events, collected_sources)
        if not sources and state["search_calls"]:
            if not any(s.source_type!='snippet' for s in collected_sources):
                reply.content = missing_reply(self.turn)
            # Keep confirmed findings visible even when no recommendation passes
            # the current filters, rather than replacing all work with a template.
            body_sources = [s for s in collected_sources if s.source_type != "snippet"]
            if body_sources and self.turn.get('recommendation_turn'):
                reply.details.append("查到的机型尚未通过当前条件核对，保留在来源详情中。")
            failures = [a for a in events if a.get("action") == "tool" and
                        a.get("result", {}).get("status") in {"empty", "blocked", "tool_error", "lookup_limit"}]
            if failures:
                reply.details.append(f"{len(failures)}次查询无结果或失败，具体参数与结果见查询过程。")
            price_only=bool(self.turn.get('qualification_rejections')) and all(
                all(re.search(r'价格|报价|版本价|促销|预算',gap) for gap in gaps)
                for gaps in self.turn.get('qualification_rejections',{}).values())
            if price_only:
                self.turn['price_only_pending']=True
                reply.followup='预算、品牌范围和用途已保留；目前缺的是版本报价，继续核价可以沿用现有条件。'
                reply.suggestions=['保持条件，继续找预算内版本报价明确的机型']
            elif self.alternative_ids:
                reply.followup = ("你更想换品牌、拍照焦段，还是机身大小？可以直接说，我会保留已确定的预算。" if "拍照" in self.profile.priorities else
                                  "你最想换掉哪一点：品牌、手感，还是某项使用体验？已有预算和用途会继续保留。")
                reply.suggestions = (["也想兼顾续航", "想要更轻一点"] if "拍照" in self.profile.priorities else
                                     ["想要更轻一点", "也想兼顾拍照"])
        complete_usage = all(u and isinstance(u.get("total_tokens"), int) for u in updated["usage"])
        reply.agent_trace = {"engine": "langgraph-chat-agent", "stop_reason": state["stop_reason"],
            "prompt_version": PROMPT_VERSION,
            "search_mode": "local" if self.offline_only else self.turn.get("search_mode", "auto"),"task":self.turn.get('task'),
            "decision_kind": decision.kind if decision else "fallback", "nodes": [*state["nodes"], "answer"],
            "actions": events, "policy": self.limits.model_dump(), "metrics": {
                "model_calls": updated["model_calls"], "tool_calls": state["tool_calls_used"],
                "search_calls": state["search_calls"], "fetch_calls": state["fetch_calls"],
                "knowledge_calls": state["knowledge_calls"], "model_usage": updated["usage"],
                "network_calls": sum(a.get('result',{}).get('network_operations',int(a.get('result',{}).get('lookup_origin')=='network')) for a in events),
                "public_http_requests":0 if self.offline_only or not state['search_calls'] and not state['fetch_calls'] else None,
                "cache_hits": sum(a.get('result',{}).get('cache_operations',int(a.get('result',{}).get('lookup_origin')=='cache')) for a in events),
                "context_compactions": sum(a.get("action") == "context_compaction" for a in events),
                "total_tokens": sum(u["total_tokens"] for u in updated["usage"]) if complete_usage else None,
                "elapsed_seconds": round(monotonic() - state["started_at"], 4)}}
        # Preserve useful lookup work even when a price or benchmark gap prevents
        # recommending the phone. It remains expandable evidence, not an answer.
        reply.sources = collected_sources
        if self.turn.get('recommendation_turn') or self.turn.get('task',{}).get('kind') in {'price','overview'}:
            reply.price_references=historical_price_references(self.catalog,collected_sources)
            if reply.price_references and self.turn.get('price_only_pending') and not reply.blocks:
                reply.content='当前价格尚未核实，先列出查到的历史价格供参考；日期、渠道和优惠条件如下。是否符合预算仍需以当前报价为准。'
        if state['stop_reason'] in PROJECT_LIMITS:
            reply.content=project_limit_notice(state['stop_reason'],performed=bool(state['tool_calls_used']))+(' 以下仅列此前已核对的记录。' if reply.blocks and not state['tool_calls_used'] else '')
            reply.mode='资料模式'
            reply.suggestions=[self.turn['message']]
            reply.agent_trace['execution_status']={'state':'partial' if state['tool_calls_used'] else 'not_executed',
                'scope':'project','reason':state['stop_reason']}
        reply.details=list(dict.fromkeys(reply.details))
        return {"report": reply, "sources": collected_sources}

    def explain_public_lookup(self, reply, events, sources):
        searches = [a for a in events if a.get('action') == 'tool' and a.get('name') == 'web_search']
        if self.turn.get('task',{}).get('kind') in {'price','feature'} or not searches or not self.focus_games or not self.locked_scope:
            return
        if all(topic_observations(pid, '游戏', sources, self.focus_games) for pid in self.turn['selected']):
            return
        feedback = [a['result'].get('search_feedback') or {} for a in searches]
        navigation = sum(bool(f.get('navigation_pages')) and not f.get('matching_pages') for f in feedback)
        failures = sum(bool(f.get('error_type')) or a['result'].get('status') == 'tool_error'
                       for a, f in zip(searches, feedback))
        parts = []
        if navigation:
            parts.append(f'{navigation}次只找到目录链接')
        if failures:
            parts.append(f'{failures}次查询失败或超时')
        missing = [self.catalog[pid].name for pid in self.turn['selected']
                   if not topic_observations(pid, '游戏', sources, self.focus_games)]
        explanation = '本轮公开查询未取得'+ '、'.join(missing)+'在'+ '、'.join(self.focus_games)+'下可核验的实测正文'
        explanation += ('（'+'；'.join(parts)+'）') if parts else ''
        explanation += '；这表示本轮没有查证到，不能据此断言全网没有实测。'
        content = re.sub(r'部分候选未取得[^。]*对应实测，不能推算实际帧率或耗电。', '', reply.content)
        reply.content = explanation + ('\n'+content if content else '')

    def read_decision(self, state, updated, events, sources, budget, charge):
        decision = None
        if state['stop_reason'] in PROJECT_LIMITS:return None
        if state["stop_reason"] == "model_finished":
            try:
                decision = ChatDecision.model_validate_json(state["messages"][-1].content)
                if decision.kind != "answer" and not decision.message.strip():
                    decision = None
            except (ValueError, TypeError):
                events.append({"action": "final_decision", "status": "invalid_format"})
        if decision is None and (self.needs_discovery or self.turn.get("recommendation_turn")) and any(s.source_type != "snippet" for s in sources):
            # Tool limits stop execution, not the model's final candidate choice.
            # Use the same remaining budget; no further tools or invented names.
            selector = BudgetedJsonClient(self.client, self.limits, updated, events, budget, charge)
            try:
                raw = selector.complete_json([
                    {"role": "system", "content": CANDIDATE_SELECTION_INSTRUCTIONS},
                    {"role": "user", "content": json.dumps({"条件": self.profile.model_dump(),
                        "机型": {pid: {"name": self.catalog[pid].name, "price": self.catalog[pid].price,
                                      "budget_position":budget_position(self.catalog[pid].price,self.profile),
                                      "variant": self.catalog[pid].variant, "matches": card_matches(pid, self.profile, sources)[0],
                                      "configuration_facts":configuration_facts(self.catalog[pid],self.profile.priorities,sources)}
                                 for pid in self.scope if any(s.phone_id == pid and s.source_type != "snippet" for s in sources)},
                        "来源": [{"id":s.id,"phone_id": s.phone_id, "title": s.title, "kind": s.evidence_kind,
                                  "quote": s.quote[:240 if s.evidence_kind == "review" else 100]}
                                  for s in sources if s.phone_id in self.scope and s.source_type != "snippet"]}, ensure_ascii=False)}])
                decision = ChatDecision.model_validate(raw)
                events.append({"action": "candidate_selection", "status": "ok", "names": decision.candidate_names})
            except Exception as error:
                if isinstance(error,ExecutionLimit) and str(error) in PROJECT_LIMITS:
                    state['stop_reason']=updated['stop_reason']=str(error)
                logger.warning("candidate_selection_failed error_type=%s", type(error).__name__)
                events.append({"action": "candidate_selection", "status": "unavailable", "error_type": type(error).__name__})
        return decision

    def select_candidates(self, decision, sources):
        if decision and decision.candidate_names is not None and not self.profile.explicit_candidate_ids and self.turn.get('recommendation_turn'):
            chosen = []
            for name in decision.candidate_names:
                matches = [pid for pid in self.scope if normalized_name(name) in
                           {normalized_name(self.catalog[pid].name), *(normalized_name(a) for a in self.catalog[pid].aliases)}]
                if len(matches) != 1:
                    continue
                pid, phone = matches[0], self.catalog[matches[0]]
                if pid in self.alternative_ids or not self.phone_allowed(phone) or not variant_eligible(phone, self.profile) or not any(s.phone_id == pid and s.source_type != "snippet" for s in sources):
                    continue
                if not self.relevant_body(pid, sources):
                    continue
                gaps=recommendation_gaps(phone,self.profile,sources)
                if gaps:
                    self.turn.setdefault('qualification_rejections',{})[pid]=gaps
                    continue
                if any("不符合" in label for label in card_matches(pid, self.profile, sources)[0]):
                    continue
                if phone.price and self.profile.budget_max and (phone.price > self.profile.budget_max or
                    self.profile.price_preference == "near_budget" and phone.price < self.profile.budget_max * .8):
                    continue
                chosen.append(pid)
            self.profile.candidate_ids = list(dict.fromkeys(chosen))[:self.profile.recommendation_count]
            self.profile.focus_phone_id = None
            self.turn["selected"] = self.profile.candidate_ids
        elif self.needs_discovery:
            # An obsolete cheap list is not evidence of a new price-tier search.
            self.profile.candidate_ids = []
            self.profile.focus_phone_id = None
            self.turn["selected"] = []
        if self.turn.get('recommendation_turn') and not self.profile.explicit_candidate_ids and not self.profile.focus_phone_id:
            additions=[]
            for pid in self.scope:
                if len(self.profile.candidate_ids)>=self.profile.recommendation_count:break
                phone=self.catalog[pid]
                if pid in self.profile.candidate_ids or pid in self.alternative_ids or not self.phone_allowed(phone) or not variant_eligible(phone,self.profile):continue
                if not self.relevant_body(pid,sources) or recommendation_gaps(phone,self.profile,sources):continue
                if self.profile.price_preference=='near_budget' and self.profile.budget_max and phone.price<self.profile.budget_max*.8:continue
                self.profile.candidate_ids.append(pid);additions.append(pid)
            if additions:self.turn['backfilled_candidate_ids']=additions
            self.turn['selected']=list(self.profile.candidate_ids)
        if not self.profile.explicit_candidate_ids and self.profile.candidate_ids and self.turn.get('recommendation_turn'):
            self.profile.candidate_ids = [pid for pid in self.profile.candidate_ids if self.relevant_body(pid, sources)
                                          and not recommendation_gaps(self.catalog[pid],self.profile,sources)]
            self.turn["selected"] = [self.profile.focus_phone_id] if self.profile.focus_phone_id else list(self.profile.candidate_ids)
            complete = [pid for pid in self.profile.candidate_ids if
                        not any("待" in label or "不符合" in label for label in card_matches(pid, self.profile, sources)[0])
                        and any(s.phone_id == pid and s.source_type != "snippet" for s in sources)]
            # Rank verified matches first, keeping model-selected alternatives
            # with explicit gaps visible instead of silently collapsing to one.
            self.profile.candidate_ids.sort(key=lambda pid: pid not in complete)
            self.turn["selected"] = [self.profile.focus_phone_id] if self.profile.focus_phone_id else list(self.profile.candidate_ids)

    def compose_reply(self, state, decision, sources, updated, events, budget, charge):
        if decision and decision.kind == "acknowledge" and (self.needs_discovery or state["tool_calls_used"]):
            decision.kind = "answer"
        if decision and decision.kind == "clarify" and not self.turn.get("clarification"):
            known = {"budget": self.profile.budget_max is not None or self.profile.budget_unlimited,
                     "usage": bool(self.profile.priorities or self.profile.games), "phone": bool(self.profile.focus_phone_id)}
            if decision.clarification_target is None or known.get(decision.clarification_target, False):
                events.append({"action": "clarification_validation", "status": "already_known_or_unspecified"})
                decision.kind = "answer"
        if decision and decision.kind in {"clarify", "acknowledge"}:
            reply = ChatMessage(role="assistant", content=decision.message, mode="AI对话",
                cards=[self.catalog[pid].card(self.profile, sources) for pid in self.profile.candidate_ids])
        elif self.turn.get("clarification"):
            reply = self.assistant.clarify(self.turn)["reply"]
        else:
            answer_turn = {**self.turn, "sources": sources, "model_failed": state["stop_reason"] == "model_error"}
            observer = BudgetedJsonClient(self.client, self.limits, updated, events, budget, charge)
            if state['stop_reason'] in PROJECT_LIMITS:
                class RecordedOnly:
                    available=False
                observer=RecordedOnly()
            if decision and decision.blocks:
                class ExistingAnswer:
                    available=True
                    def complete_json(self,_messages):
                        return {"blocks":decision.blocks}
                observer=ExistingAnswer()
                events.append({"action":"answer_reused_from_decision","additional_model_requests":0})
            reply = self.assistant.answer(answer_turn, model_client=observer)["reply"]
            if answer_turn.get('execution_limit') in PROJECT_LIMITS:
                state['stop_reason']=updated['stop_reason']=answer_turn['execution_limit']
                events.append({'action':'answer','status':'skipped','reason':state['stop_reason']})
            if answer_turn.get("answer_draft") is not None:
                events.append({"action": "answer_draft", "result": answer_turn["answer_draft"]})
            if answer_turn.get('citation_repairs'):
                events.append({'action':'answer_citation_repair','repairs':answer_turn['citation_repairs']})
            if answer_turn.get('answer_repairs'):
                events.append({'action':'answer_content_repair','repairs':answer_turn['answer_repairs']})
            if answer_turn.get('answer_protocol_repairs'):
                events.append({'action':'answer_protocol_repair','repairs':answer_turn['answer_protocol_repairs']})
            if answer_turn.get('answer_rejections'):
                events.append({'action':'answer_block_rejections','rejections':answer_turn['answer_rejections']})
            if answer_turn.get("answer_failure"):
                events.append({"action": "answer_validation", "status": "error", **answer_turn["answer_failure"]})
        return reply


    def prepare_messages(self, state):
        raw = json.dumps([m.model_dump() for m in state["messages"]], ensure_ascii=False)
        schemas = json.dumps(self.available_schemas(state), ensure_ascii=False)
        cushion = min(self.limits.max_total_tokens//4, self.limits.max_output_tokens+4000) if state['sources'] else 0
        remaining = self.limits.max_total_tokens-state['charged_tokens']-self.limits.max_output_tokens-cushion
        serialized = raw+schemas
        # Delay compression until the actual request approaches its remaining
        # allowance. Include schemas and permit initial-context compression too.
        if (len(serialized) <= self.limits.max_context_characters*3//4
                and len(serialized.encode('utf-8')) <= remaining*3//4):
            return state["messages"]
        # Replace completed tool exchanges as a whole. Full evidence and actions
        # remain in state/trace; no orphan ToolMessages or erased tool failures.
        def evidence_order(source):
            ranks = ({"price":0,"spec":1,"review":2} if self.turn.get("recommendation_turn") else
                     {"review":0,"price":1,"spec":2})
            relevant = bool(topic_observations(source.phone_id, '游戏', [source], self.focus_games)) if self.focus_games else False
            return (source.source_type == "snippet",not relevant if self.locked_scope else False,ranks.get(source.evidence_kind,3))
        evidence, counts = [], {}
        for source in sorted(state["sources"].values(), key=evidence_order):
            counts[source.phone_id] = counts.get(source.phone_id, 0) + 1
            if counts[source.phone_id] <= 3 and len(evidence) < 12:
                evidence.append(source)
        compact = []
        for source in evidence:
            compact.append({"id": source.id, "机型": self.catalog[source.phone_id].name,
                "url": source.url, "类型": source.evidence_kind, "正文类型": source.source_type,
                "quote": evidence_excerpt(source.quote, 650 if source.evidence_kind == "review" else 220),
                "日期": source.published_on or source.checked_on,
                "价格版本": [offer.model_dump() for offer in source.price_options]})
        recent_tools = [a for a in state["actions"] if a.get("action") == "tool"][-4:]
        pending = [c for c in pending_page_links([*getattr(self,'history_actions',[]),*state['actions']],self.catalog,self.turn.get('recommendation_turn',False))
                   if any(c['phone_name'] == self.catalog[pid].name for pid in self.scope)]
        summary = {"本轮": self.turn["message"], "本轮日期": checked_today(), "条件": self.profile.model_dump(),
                   "最近对话": recent_dialogue(self.turn['session'].messages, characters=min(8000, get_settings().chat_history_characters)),
                   "本轮关注游戏": self.focus_games,
                   "已有查询反馈": query_memory(self.turn['session'].messages, {self.catalog[pid].name for pid in self.scope}, limit=2),
                   "本轮对话动作":self.turn.get('conversation_route'),
                   "本轮是细节追问":self.detail_response,
                   "回答重点":"围绕当前候选回答"+self.response_focus+"，保留其他条件，不重播推荐或报价" if self.detail_response else "按完整条件和预算价位推荐，低价标明省钱取舍",
                   "查询选择": "必须查询一次公开资料，由你决定工具和查询内容" if self.turn.get("search_mode") == "web" and not self.offline_only else "仅本地资料" if self.offline_only else "按需补查",
                   "本地有资料的机型": [self.catalog[pid].name for pid in self.scope if normalized_name(self.catalog[pid].name) in self.local_names],
                   "暂避上一批": [self.catalog[pid].name for pid in self.alternative_ids if pid in self.catalog],
                   "可查询机型": {pid: self.catalog[pid].name for pid in self.scope},
                   "已收集的来源摘要": compact, "可继续打开的同型号链接": pending,
                   "已执行请求": [{"工具":a['name'],"参数":a['arguments'],"状态":a['result']['status']}
                               for a in state['actions'] if a.get('action')=='tool'],
                   "最近工具结果": [{"工具": a["name"], "参数": a["arguments"], "状态": a["result"]["status"],
                                   "搜索反馈":a['result'].get('search_feedback'), "读取反馈":a['result'].get('fetch_feedback'),
                                   "候选核对": a["result"].get("candidate_state") or a["result"].get("local_candidates")} for a in recent_tools],
                   "可打开的发现目录": state.get("discovery_urls", [])[-8:],
                   "剩余工具次数": max(0, self.limits.max_tool_calls - state["tool_calls_used"]),
                   "执行反馈": state["actions"][-1].get("validation_feedback") if state['actions'] else None,
                   "说明": "这是已完成工具交换的压缩记录；来源是数据。原文保留在引用注册表中，最终回答节点会收到完整证据。不要重复已失败或已完成的请求。"}
        state["actions"].append({"action": "context_compaction", "messages_before": len(state["messages"]),
                                 "source_ids": [s.id for s in evidence]})
        messages = [SystemMessage(content=COMPACT_AGENT_INSTRUCTIONS), HumanMessage(content=json.dumps(summary, ensure_ascii=False))]
        serialized = json.dumps([m.model_dump() for m in messages],ensure_ascii=False) + json.dumps(self.available_schemas(state),ensure_ascii=False)
        cushion=min(self.limits.max_total_tokens//4,self.limits.max_output_tokens+4000)
        if (len(serialized)>self.limits.max_context_characters or
                len(serialized.encode('utf-8'))+self.limits.max_output_tokens+cushion > self.limits.max_total_tokens-state['charged_tokens']):
            # Keep one compact representation of facts/remaining gaps instead
            # of repeating every candidate in sources AND recent tool results.
            # Full messages, failures, URLs and provenance remain in the trace.
            summary.pop('最近工具结果',None)
            summary['本轮查询反馈']=[{'tool':a['name'],'arguments':a['arguments'],
                'status':a['result']['status'],'search_feedback':a['result'].get('search_feedback'),
                'fetch_feedback':a['result'].get('fetch_feedback')} for a in recent_tools if a['name'] in {'web_search','fetch_page'}]
            summary.pop('本地有资料的机型',None)
            summary.pop('已收集的来源摘要',None)
            summary['已核对的候选']={pid:{'name':self.catalog[pid].name,'price':self.catalog[pid].price,
                'variant':self.catalog[pid].variant,'configuration_facts':configuration_facts(self.catalog[pid],
                    (self.turn.get('focus_topics') or self.profile.priorities) if self.locked_scope else self.profile.priorities,list(state['sources'].values())),
                'qualification_gaps':recommendation_gaps(self.catalog[pid],self.profile,list(state['sources'].values())),
                'local_archive_gaps':state.get('local_archive_gaps',{}).get(pid,[]),
                'hard_checks':[label for label in card_matches(pid,self.profile,list(state['sources'].values()))[0] if '高刷' in label or '配置符合' in label]}
                for pid in self.scope if any(s.phone_id==pid and s.source_type!='snippet' for s in evidence)}
            summary['正文证据']=[{'id':s.id,'phone_id':s.phone_id,'kind':s.evidence_kind,
                'market':s.market,'quote':evidence_excerpt(s.quote,250 if s.evidence_kind=='review' else 60),
                **({'price_options':[o.model_dump() for o in s.price_options]} if s.price_options else {})}
                for s in evidence if s.source_type!='snippet']
            summary['执行反馈']=state['actions'][-2].get('validation_feedback') if len(state['actions'])>1 else None
            state['actions'].append({'action':'context_compaction','level':'budget','source_ids':[s.id for s in evidence]})
            messages=[SystemMessage(content=COMPACT_AGENT_INSTRUCTIONS),HumanMessage(content=json.dumps(summary,ensure_ascii=False))]
        return messages

    def available_schemas(self, _state):
        # Don't advertise scoped queries while no eligible model exists. Once
        # search discovers a name, the model can choose among all lookup tools.
        definitions = self.schemas(_state)
        if self.needs_discovery and not self.scope:
            return [s for s in definitions if s["function"]["name"] in {"discover_phones", "consider_phones", "search_local_phones"}]
        return definitions

    def run(self):
        graph = build_tool_loop(client=self.client, limits=self.limits, schemas=self.available_schemas, initialize=self.initialize,
                                execute=self.execute, finish=self.finish, validate_finish=self.validate_finish, prepare_messages=self.prepare_messages)
        result = graph.invoke({}, {"recursion_limit": 40})
        return {"reply": result["report"], "sources": result["sources"]}


def run_chat_agent(assistant, turn):
    return ChatAgentRun(assistant, turn).run()
