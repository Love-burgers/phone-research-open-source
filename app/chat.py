"""Multi-turn selection with model-directed tools and an offline fallback.

LangGraph controls each turn. SQLite owns durable conversation state; this is
not a LangGraph checkpoint/resume implementation. Explicit models may use public sources.
"""

from time import monotonic
from threading import Lock
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import logging
from pathlib import Path
import re
from typing import TypedDict
from pydantic import HttpUrl

from langgraph.graph import END, START, StateGraph

from .chat_schemas import ChatMessage, ChatRequest, ChatSession, ChatSource
from .chat_store import ChatStore
from .config import get_settings
from .execution_context import turn_deadline
from .chat_agent import run_chat_agent
from .chat_answer import AnswerComposer
from .tool_loop import AgentPolicy, project_limit_notice
from .chat_intent import TOPIC_WORDS, understand_intent, conversation_candidates, resolve_references, mentioned_phones, literal_candidate_scope, series_scope, update_brand_scope, general_assessment
from .conversation_policy import route_turn
from .knowledge import KnowledgeBase
from .model_api import ModelClient
from .selection_catalog import (DATA_DIR, eligible, load_catalog, shortlist, select_variant, variant_eligible,
                               minimum_storage, storage_gb, select_offer, source_price_options)
from .selection_catalog import CatalogPhone
from .phone_names import parse_phone_names, register_phone
from .phone_identity import series_excluded
from .phone_sources import PublicPhoneSources, source_market
from .selection_prices import dated_promotion
from .source_rules import official_phone_url
from .selection_requirements import literal_requirements, card_matches, local_only
from .selection_presentation import configuration_facts
from .evidence_rules import game_in_text
from .evidence_registry import (session_evidence, merge_sources, refresh_kinds, public_allowed,
    evidence_gaps, finish_refresh)
from .feature_requirements import field_keys, update_requirements, RequirementOverflow
from .conversation_context import asks_price, answer_requirements
from .selection_prices import sanitize_session_prices

logger = logging.getLogger(__name__)

class TurnState(TypedDict, total=False):
    session: ChatSession
    message: str
    clarification: str
    selected: list[str]
    sources: list[ChatSource]
    reply: ChatMessage
    model_failed: bool
    acknowledged: list[str]
    query: str
    catalog: dict[str, CatalogPhone]
    answer_generated: bool
    intent_usage: dict | None
    intent_model_attempts: int
    turn_started_at: float
    alternative_ids: list[str]
    intent_failure: str | None
    intent_repairs: list[str]
    recommendation_turn: bool
    search_mode: str
    detail_followup: bool
    response_focus: str
    focus_terms: list[str]
    focus_topics: list[str]
    conversation_route: dict
    evidence_actions: list[dict]
    task: dict
    execution_limit: str
    scope_series: list[str]


class SelectionAssistant:
    def __init__(self, store: ChatStore, model_client=None, catalog=None, knowledge=None, source_loader=None,
                 *, agent_enabled=None, agent_policy=None):
        self.store = store
        settings = get_settings()
        self.catalog = catalog if catalog is not None else load_catalog()
        self._allow_inventory_catalog = catalog is None
        self.model = model_client if model_client is not None else ModelClient()
        bundled_knowledge = lambda: KnowledgeBase.from_path(DATA_DIR / "selection_index.json")
        self.knowledge = knowledge if knowledge is not None else (
            KnowledgeBase.from_path(Path(settings.chat_knowledge_index_path),
                                    search_method=settings.chat_retrieval_method)
            if settings.chat_knowledge_index_path else bundled_knowledge())
        self.source_loader = source_loader if source_loader is not None else PublicPhoneSources(
            store.path.parent / "phone-source-cache")
        self.agent_enabled = settings.chat_agent_enabled if agent_enabled is None else agent_enabled
        # Discovery + page verification needs room beyond the old local-only path.
        self.agent_policy = agent_policy or AgentPolicy(max_tool_calls=settings.chat_turn_tool_calls,
            max_model_calls=settings.chat_turn_model_calls,
            max_search_calls=settings.chat_turn_search_calls, max_fetch_calls=settings.chat_turn_fetch_calls,
            max_total_tokens=settings.chat_turn_token_budget,
            max_context_characters=settings.chat_context_characters)
        graph = StateGraph(TurnState)
        for name in ("understand", "clarify", "retrieve", "answer", "agent_turn", "execution_stopped"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "understand")
        graph.add_conditional_edges("understand", lambda state: (
            'execution_stopped' if state.get('execution_limit') else 'clarify' if state.get('clarification')
            else "agent_turn" if self.agent_enabled and self.model.available else "retrieve"))
        graph.add_edge('execution_stopped',END)
        graph.add_edge("agent_turn", END)
        graph.add_edge("retrieve", "answer")
        graph.add_edge("clarify", END)
        graph.add_edge("answer", END)
        self.graph = graph.compile()

    def turn(self, request: ChatRequest) -> ChatSession:
        session = self.store.load(request.session_id) if request.session_id else self.store.create()
        with turn_deadline(self.agent_policy.deadline_seconds):
            result = self.graph.invoke({"session": session, "message": request.message,
                                        "search_mode": request.search_mode,
                                        "model_failed": False, "acknowledged": [], "turn_started_at": monotonic()})
        for pid in session.profile.candidate_ids:
            if pid not in self.catalog and not any(p.id==pid for p in session.discovered_phones) and len(session.discovered_phones)<20:
                if any(s.phone_id==pid and s.source_type!='snippet' for s in result['reply'].sources):
                    session.discovered_phones.append(result['catalog'][pid].model_copy(deep=True))
        session.messages += [ChatMessage(role="user", content=request.message), result["reply"]]
        saved=self.store.save(session)
        if saved.messages[-1].agent_trace is not None:
            try:update=self.knowledge.remember_sources(result.get('sources',[]),result.get('catalog',{}))
            except Exception as error:
                logger.warning('knowledge_update_failed error_type=%s',type(error).__name__)
                update={'status':'error','error_type':type(error).__name__}
            try:self.store.record_knowledge_update(saved,update)
            except Exception as error:logger.warning('knowledge_trace_save_failed error_type=%s',type(error).__name__)
        return saved

    def local_inventory_catalog(self):
        """Price/configuration leads from the index, without inventing facts or IO."""
        if not self._allow_inventory_catalog or not self.agent_enabled or not self.model.available or not hasattr(self.knowledge,'index'):
            return {pid:p.model_copy(deep=True) for pid,p in self.catalog.items()}
        key=tuple(c.chunk_id for c in self.knowledge.index.chunks)
        cached=getattr(self,'_inventory_seed',None)
        if cached and cached[0]==key:return {pid:p.model_copy(deep=True) for pid,p in cached[1].items()}
        catalog={pid:p.model_copy(deep=True) for pid,p in self.catalog.items()}
        grouped={}
        for chunk in self.knowledge.index.chunks:grouped.setdefault(chunk.phone_name,[]).append(chunk)
        for name,chunks in grouped.items():
            if not any(c.evidence_kind=='price' for c in chunks):continue
            mentions=parse_phone_names(name)
            if len(mentions)!=1:continue
            phone=register_phone(mentions[0],catalog)
            sources=[ChatSource(id=phone.id+':'+c.chunk_id,phone_id=phone.id,title=c.title,url=c.source_url,
                quote=c.text,checked_on=c.checked_on or phone.checked_on,published_on=c.published_on,
                source_type='knowledge',market=c.market,evidence_kind=c.evidence_kind,
                reference_price=c.reference_price,price_variant=c.price_variant,
                price_options=[{'price':c.reference_price,'variant':c.price_variant}] if c.reference_price and c.price_variant else []) for c in chunks]
            records=[(source,offer) for source in sources if source.market=='中国大陆' for offer in source_price_options(source,phone.name)]
            if not records:continue
            offers=list({(o.variant,o.price):o for s,o in records}.values())
            if phone.price is None:
                chosen=min(offers,key=lambda o:o.price)
                source=next(s for s,o in records if o==chosen)
                phone.price,phone.variant,phone.price_url=chosen.price,chosen.variant,HttpUrl(source.url)
                phone.checked_on=source.checked_on;phone.price_options=offers[:12]
                phone.note='本地索引中的版本参考报价；按来源日期与条件初筛，不保证当前成交价或库存。'
            phone.topics=list(dict.fromkeys([*phone.topics,*(topic for topic in TOPIC_WORDS if configuration_facts(phone,[topic],sources))]))
            catalog[phone.id]=phone
        self._inventory_seed=(key,catalog)
        return {pid:p.model_copy(deep=True) for pid,p in catalog.items()}

    def understand(self, state: TurnState):
        session = state["session"]
        sanitize_session_prices(session)
        profile = session.profile
        if 'answer_requirements' not in profile.model_fields_set:
            for message in session.messages:
                if message.role=='user':profile.answer_requirements=answer_requirements(profile.answer_requirements,message.content)
        # Migrate only fields absent from older saved profiles. Reconstruct clear
        # user-stated requirements, without changing any saved conversation text.
        missing_fields = {"price_preference", "recommendation_count", "high_refresh_required", "games"} - profile.model_fields_set
        if missing_fields:
            restored = profile.model_copy(deep=True)
            for message in session.messages:
                if message.role == "user":
                    literal_requirements(restored, message.content)
            for field in missing_fields:
                setattr(profile, field, getattr(restored, field))
        # A new phone belongs to this session, never to the service's shared catalog.
        catalog = self.local_inventory_catalog()
        catalog.update((phone.id, CatalogPhone.model_validate(phone.model_dump()))
                       for phone in session.discovered_phones)
        if 'brand_scope' not in profile.model_fields_set:
            for message in session.messages:
                if message.role=='user':update_brand_scope(profile,message.content,catalog)
        # Migrate reusable configuration tags as well as measured-review tags.
        # Without these, a follow-up discarded verified parameter-only choices
        # as if their main use had never been researched.
        saved_sources=list(session_evidence({'session':session,'message':'','search_mode':'local'},
            catalog,list(catalog))['sources'].values())
        self._remember_sources({'session':session,'catalog':catalog},saved_sources)
        for phone in session.discovered_phones:
            configured = [topic for topic in TOPIC_WORDS if configuration_facts(phone, [topic], saved_sources)]
            phone.topics = list(dict.fromkeys([*phone.topics, *configured]))
            catalog[phone.id].topics = list(phone.topics)
        text, ack = state["message"], []
        # Normalize old persisted compounds without rewriting historical messages.
        try:profile.requirements=update_requirements(profile.requirements,'')
        except RequirementOverflow:pass  # Preserve oversized legacy compounds; never truncate them.
        previous_profile = profile.model_dump()
        overflow = False
        for mention in parse_phone_names(text):
            phone = register_phone(mention, catalog)
            if phone.id not in catalog:
                if len(session.discovered_phones) >= 20:
                    overflow = True
                    continue
                catalog[phone.id] = phone
                session.discovered_phones.append(phone)
        intent = understand_intent(profile, text, catalog, self.model, session.messages)
        scope_series=series_scope(profile,text,session.messages,catalog)
        if scope_series:intent.candidate_action='alternatives';intent.reselect=True
        selected_mentions = intent.selected or [pid for pid,_,excluded in mentioned_phones(text,catalog) if not excluded]
        for phone in catalog.values():
            select_variant(phone, profile)
        model_failed, ack = intent.model_failed, intent.acknowledged or []
        # Explicit model-name detection remains available for reference resolution,
        # while condition changes come from exactly one parser per turn.
        mentions = any(pid in selected_mentions and not excluded for pid, _, excluded in mentioned_phones(text, catalog))
        available_ids = conversation_candidates(profile, session.messages, catalog)
        # Rotate away from the current batch; previously considered phones are
        # still eligible on later turns unless the user explicitly excludes them.
        alternative_ids = conversation_candidates(profile,session.messages,catalog,last_displayed=True) if intent.candidate_action == "alternatives" else []
        alternative_ids=list(dict.fromkeys([*alternative_ids,*[pid for pid,p in catalog.items() if series_excluded(p.name,scope_series)]]))
        clarification = intent.clarification
        if not selected_mentions and not clarification:
            selected_mentions, clarification = resolve_references(text, profile, session.messages, catalog)
            selected_mentions = [pid for pid in selected_mentions if pid not in profile.excluded_phones]
        selected_mentions = list(dict.fromkeys(selected_mentions))
        if intent.candidate_scope == 'retain':
            intent.candidate_scope = literal_candidate_scope(text,selected_mentions,catalog)
        known_conflicts = {pid for pid in catalog if any('不符合' in label
                           for label in card_matches(pid, profile, saved_sources)[0])}
        incompatible = any(pid in catalog and (not variant_eligible(catalog[pid], profile)
                           or pid in known_conflicts) for pid in available_ids)
        route = route_turn(previous_profile, profile, intent, has_candidates=bool(available_ids),
                           selected=selected_mentions, incompatible=incompatible)
        # Rechecking constraints must not erase a user-pinned comparison list.
        # Only a fresh unscoped recommendation or an explicit scope operation
        # releases it; price-tier revisions still validate known constraints.
        restrict_scope = intent.candidate_scope == 'restrict' and bool(selected_mentions)
        fresh_recommendation = intent.turn_action == 'recommend' and not selected_mentions and not restrict_scope
        reselect = intent.reselect or intent.candidate_scope == 'release' or fresh_recommendation or route.operation == 'alternatives'
        detail_followup = route.operation == 'refine'
        added_games = [g for g in profile.games if g not in previous_profile['games']]
        # Exact identity matters for named games, not for a whole prose topic.
        # A grounded focus quote may be paraphrased naturally in the answer.
        focus_terms = list(dict.fromkeys([*added_games,*(game for game in profile.games
            if intent.response_focus and game_in_text(game,intent.response_focus))]))
        response_focus = intent.response_focus or '、'.join(added_games) or text
        focus_topics = intent.focus_topics or [topic for topic, words in TOPIC_WORDS.items() if any(word in response_focus for word in words)]
        # Older clients may omit semantic topic metadata. Keep an unresolved
        # detail narrow instead of turning every remembered preference into it.
        focus_topics = focus_topics or (['游戏'] if added_games else list(profile.priorities[:1] if detail_followup else profile.priorities))
        if reselect:
            profile.explicit_candidate_ids = []
            profile.focus_phone_id = None
            # A fresh recommendation uses constraints, not an extractor's copied
            # reference to a phone from an earlier question.
            if fresh_recommendation or intent.candidate_scope == 'release':
                selected_mentions = []
        if alternative_ids:
            selected_mentions = []
            clarification = ""
        if route.recommends or (not profile.candidate_ids and (profile.budget_max or profile.budget_unlimited)):
            if not profile.explicit_candidate_ids:
                profile.focus_phone_id = None
                retained = [pid for pid in profile.candidate_ids if pid in catalog and eligible(catalog[pid], profile)
                            and pid not in known_conflicts
                            and variant_eligible(catalog[pid], profile)
                            and (catalog[pid].price is None or not profile.budget_max or catalog[pid].price <= profile.budget_max)
                            and (profile.price_preference != "near_budget" or not profile.budget_max
                                 or catalog[pid].price is None or catalog[pid].price >= profile.budget_max * .8)]
                rerank = incompatible or any(previous_profile[field] != getattr(profile, field) for field in (
                    "budget_max", "budget_unlimited", "price_preference", "priorities", "preferred_brands",
                    "excluded_brands", "excluded_phones", "requirements", "high_refresh_required",'excluded_series','brand_scope'))
                profile.candidate_ids = (retained[:profile.recommendation_count] if retained and not reselect and
                                         not rerank else shortlist({pid:p for pid,p in catalog.items() if pid not in known_conflicts and not series_excluded(p.name,scope_series)}, profile))
            else:
                profile.candidate_ids = list(profile.explicit_candidate_ids)
        if selected_mentions:
            if len(selected_mentions) > 5:
                ack.append("每轮最多比较5款，已先保留你列出的前5款")
                selected_mentions = selected_mentions[:5]
            rejected = [item for item in selected_mentions if not eligible(catalog[item], profile)]
            if rejected:
                clarification = "这款与你已确认的排除或品牌条件冲突。可以说‘重新考虑" + catalog[rejected[0]].name + "’，或取消品牌限制。"
            else:
                if len(selected_mentions) > 1:
                    profile.focus_phone_id = None
                    if restrict_scope:
                        profile.candidate_ids = selected_mentions
                        profile.explicit_candidate_ids = selected_mentions
                    elif not profile.candidate_ids:
                        profile.candidate_ids = selected_mentions
                else:
                    profile.focus_phone_id = selected_mentions[0]
                    if selected_mentions[0] not in profile.candidate_ids:
                        profile.candidate_ids = list(dict.fromkeys(selected_mentions + profile.candidate_ids))[:5]
                    if restrict_scope:
                        if selected_mentions[0] not in profile.explicit_candidate_ids:
                            if len(profile.explicit_candidate_ids) >= 5:
                                ack.append("候选最多保留5款，已加入本款并移出最后一款候选")
                            profile.explicit_candidate_ids = list(dict.fromkeys(
                                selected_mentions + profile.explicit_candidate_ids))[:5]
                        profile.candidate_ids = list(profile.explicit_candidate_ids)
        profile.candidate_ids = [item for item in profile.candidate_ids if item in catalog and eligible(catalog[item], profile)]
        profile.explicit_candidate_ids = [item for item in profile.explicit_candidate_ids
                                        if item in catalog and eligible(catalog[item], profile)]
        if profile.focus_phone_id and (profile.focus_phone_id not in catalog or not eligible(catalog[profile.focus_phone_id], profile)):
            profile.focus_phone_id = None
        if overflow:
            clarification = "本对话已记录20款新型号，暂不能再添加。请新建对话查询新型号，已有条件和机型仍保留。"
        if not clarification and not selected_mentions and not profile.explicit_candidate_ids and not profile.focus_phone_id:
            if profile.budget_max is None and not profile.budget_unlimited:
                clarification = "你希望预算控制在多少元？也可以直接指定想了解的机型。"
            elif not profile.priorities and not profile.games:
                clarification = "预算已记住。你主要看重拍照、续航、游戏，还是小屏手感？可以说一到两项。"
            elif not profile.candidate_ids and not (self.agent_enabled and self.model.available and not model_failed):
                clarification = "现有资料范围内，没有参考价与品牌条件同时符合的机型。我保留了你的条件；可以调整预算或品牌，也可以指定机型核对资料。"
        if alternative_ids:
            profile.candidate_ids = [pid for pid in profile.candidate_ids if pid not in alternative_ids]
        selected = selected_mentions or ([profile.focus_phone_id] if profile.focus_phone_id else profile.candidate_ids)
        recommending=route.recommends and not selected_mentions and not profile.explicit_candidate_ids
        fields=sorted(field_keys(text)) if not recommending else []
        asking_price=not recommending and bool(re.search(r'价格|售价|价钱|多少钱|报价|到手价|卖多少|现价',text))
        comparing=bool(profile.explicit_candidate_ids or len(selected)>1) and (route.recommends or asks_price(text))
        kind='recommend' if recommending else 'price' if asking_price else 'feature' if fields else 'compare' if comparing else 'detail'
        if kind=='detail' and general_assessment(text,selected,catalog):
            kind='overview';response_focus='、'.join(profile.priorities)+'适配与取舍' if profile.priorities else '整体配置与取舍'
            focus_topics=list(profile.priorities) or list(TOPIC_WORDS)
        task={'kind':kind,'phone_ids':list(selected),'count':profile.recommendation_count if recommending else len(selected),
            'excluded_series':list(dict.fromkeys([*profile.excluded_series,*scope_series])),
            'fields':fields,'topics':focus_topics if kind not in {'price','feature'} else [],
            'games':focus_terms or list(profile.games) if kind not in {'price','feature'} else [],
            'evidence_requirements':list(profile.answer_requirements),
            'refresh':sorted(refresh_kinds(text)),'public_allowed':state.get('search_mode')!='local' and not local_only(text)}
        query = " ".join([text, *profile.priorities, *profile.games, *profile.requirements,
                          "高刷" if profile.high_refresh_required else "", "配置" if detail_followup else "配置 价格"])
        if kind in {'price','feature'}:query=text
        return {"clarification": clarification, "selected": selected, "model_failed": model_failed,
                "acknowledged": ack, "query": query, "catalog": catalog, "answer_generated": False,
                "intent_usage": getattr(self.model, "last_usage", None), "intent_model_attempts":intent.model_attempts, "alternative_ids": alternative_ids,
                'execution_limit':intent.execution_limit,'scope_series':scope_series,
                "recommendation_turn": recommending,"task":task,
                "detail_followup": detail_followup,
                "response_focus": response_focus, "focus_terms": [] if kind=='overview' else focus_terms, "focus_topics": focus_topics,
                "conversation_route": {"operation": route.operation, "changed_fields": list(route.changed_fields),
                                       "focus": response_focus, "reason": route.reason},
                "intent_failure": intent.error_type, "intent_repairs": intent.grounding_repairs or []}

    def clarify(self, state: TurnState):
        return {"reply": ChatMessage(role="assistant", content="。".join(state["acknowledged"] + [state["clarification"]]),
                                      suggestions=["预算4000，主要拍照和续航", "预算3000，主要玩游戏"],
                                      mode=self.mode(state),agent_trace={'engine':'conversation-control','decision_kind':'clarify',
                                      'stop_reason':'clarification_required','task':state.get('task'),
                                      'actions':[],'metrics':{'model_calls':state.get('intent_model_attempts',0),'tool_calls':0,'search_calls':0,
                                      'fetch_calls':0,'network_calls':0,'public_http_requests':0}})}

    def execution_stopped(self,state):
        reason=state['execution_limit']
        return {'reply':ChatMessage(role='assistant',mode='资料模式',content=project_limit_notice(reason) or '本轮请求被执行边界阻止，条件已保留。',
            suggestions=[state['message']],
            details=state.get('acknowledged',[]),agent_trace={'engine':'conversation-control','stop_reason':reason,
                'execution_status':{'state':'not_executed','scope':'project','reason':reason},'task':state.get('task'),
                'actions':[{'action':'intent','status':'skipped','reason':reason}],
                'metrics':{'model_calls':state.get('intent_model_attempts',0),'tool_calls':0,'search_calls':0,
                    'fetch_calls':0,'network_calls':0,'public_http_requests':0}})}

    def retrieve(self, state: TurnState):
        catalog=state['catalog']; profile=state['session'].profile
        search=self.knowledge.as_tool(); refresh=refresh_kinds(state['message'])
        counts={'search':0,'fetch':0}; lookup_lock=Lock()
        def reserve(kind):
            with lookup_lock:
                limit=self.agent_policy.max_search_calls if kind=='search' else self.agent_policy.max_fetch_calls
                if counts[kind]>=limit or sum(counts.values())>=self.agent_policy.max_tool_calls:return False
                counts[kind]+=1
                return True
        def collect(phone_id):
            registry=session_evidence(state,catalog,[phone_id])
            try:
                matches=search.invoke({'phone_name':catalog[phone_id].name,'need':state['query']})
                incoming=[ChatSource(id=f"{phone_id}:{hit['chunk_id']}",phone_id=phone_id,
                    title=hit['title'],url=hit['source_url'],quote=hit['text'],
                    checked_on=hit.get('checked_on') or catalog[phone_id].checked_on,
                    published_on=hit.get('published_on'),market=hit.get('market') or '地区未确认',
                    evidence_kind=hit.get('evidence_kind','unknown'),reference_price=hit.get('reference_price'),
                    price_variant=hit.get('price_variant')) for hit in matches]
                merge_sources(registry,phone_id,incoming,catalog,profile,
                    focus_topics=state.get('focus_topics',()),refresh=refresh)
                registry['actions'].append({'action':'local_retrieval','phone_id':phone_id,'source_ids':[s.id for s in incoming]})
            except Exception as error:
                logger.warning('retrieval_failed origin=local error_type=%s',type(error).__name__)
                registry['actions'].append({'action':'local_retrieval','phone_id':phone_id,'error_type':type(error).__name__})
            attempted=False
            gaps=evidence_gaps(state,catalog[phone_id],list(registry['sources'].values()))
            if public_allowed(state) and (state.get('search_mode')=='web' or refresh or gaps):
                attempted=True
                try:
                    if hasattr(self.source_loader,'collect_result'):
                        result=self.source_loader.collect_result(catalog[phone_id],state['query'],refresh=bool(refresh),reserve=reserve)
                        incoming=result['sources']; registry['actions'].extend(result.get('actions',[]))
                    else:
                        incoming=self.source_loader.collect(catalog[phone_id],state['query'],refresh=bool(refresh))
                        registry['actions'].append({'action':'public_lookup','phone_id':phone_id,'status':'adapter_result','lookup_origin':'unknown'})
                    merge_sources(registry,phone_id,incoming,catalog,profile,
                        focus_topics=state.get('focus_topics',()),refresh=refresh,fresh=True)
                except Exception as error:
                    logger.warning('retrieval_failed origin=public error_type=%s',type(error).__name__)
                    registry['actions'].append({'action':'public_lookup','phone_id':phone_id,'status':'tool_error','error_type':type(error).__name__})
            finish_refresh(state,registry,attempted=attempted)
            return registry
        combined={'sources':{},'allowed_urls':set(),'actions':[]}
        with ThreadPoolExecutor(max_workers=min(3,len(state['selected'])) or 1) as pool:
            futures=[pool.submit(copy_context().run,collect,pid) for pid in state['selected']]
            for phone_id,future in zip(state['selected'],futures):
                registry=future.result()
                merge_sources(combined,phone_id,list(registry['sources'].values()),catalog,profile,
                    focus_topics=state.get('focus_topics',()))
                combined['actions'].extend(registry['actions'])
        sources=list(combined['sources'].values()); self._remember_sources(state,sources)
        return {'sources':sources,'evidence_actions':combined['actions']}

    def _remember_sources(self, state, sources):
        catalog = state["catalog"]
        profile = state["session"].profile
        targets={p.id:p for p in catalog.values()}
        targets.update((p.id,p) for p in state['session'].discovered_phones)
        for phone in targets.values():
            matched = [source for source in sources if source.phone_id == phone.id]
            if matched:
                phone.source_urls = [HttpUrl(url) for url in list(dict.fromkeys(source.url for source in matched))[:8]]
                phone.checked_on = max(source.checked_on for source in matched)
                catalog[phone.id].source_urls = phone.source_urls
                catalog[phone.id].checked_on = phone.checked_on
                evidence_profile = profile.model_copy(update={"priorities": list(TOPIC_WORDS),
                    "games": [], "requirements": [], "high_refresh_required": False})
                labels, _ = card_matches(phone.id, evidence_profile, matched)
                supported = [topic for topic in TOPIC_WORDS if any(label.startswith(topic)
                    and "待" not in label and "冲突" not in label for label in labels)]
                supported += [topic for topic in TOPIC_WORDS if configuration_facts(phone, [topic], matched)]
                phone.topics = list(dict.fromkeys([*phone.topics, *supported]))
                catalog[phone.id].topics = list(phone.topics)
                def price_source_order(source):
                    chosen=select_offer(source_price_options(source),profile)
                    allowed=chosen and (not profile.budget_max or chosen.price<=profile.budget_max) and (
                        (storage_gb(chosen.variant) or 0)>=minimum_storage(profile))
                    value=chosen.price if allowed else 100001
                    return (dated_promotion(source),not official_phone_url(phone.brand,source.url),not bool(allowed),
                            -int(source.checked_on.replace('-','')), -value if profile.price_preference=='near_budget' and allowed else value)
                for source in sorted(matched, key=price_source_order):
                    if (source.market == "中国大陆"
                        and (official_phone_url(phone.brand, source.url) or source_price_options(source)
                             and source_market(source.url,source.title,source.quote)=='中国大陆')):
                        offers = source_price_options(source)
                        if not offers:
                            continue
                        phone.price_options = list(offers)
                        phone.price_url = HttpUrl(source.url)
                        select_variant(phone, profile)
                        phone.checked_on = source.checked_on
                        phone.note = ("官方页面的有日期展示价；其他渠道、库存和优惠仍需核对。"
                            if official_phone_url(phone.brand, source.url) else
                            "来源正文记录的版本参考价，非官方报价；适用渠道、日期及优惠条件见来源。")
                        catalog[phone.id] = CatalogPhone.model_validate(phone.model_dump())
                        break
    def agent_turn(self, state):
        result = run_chat_agent(self, state)
        self._remember_sources(state, result["sources"])
        return {"reply": result["reply"], "sources": result["sources"]}

    def mode(self, state):
        if state.get("answer_limited"):
            return "资料模式"
        if state.get("answer_failure", {}).get("stage") not in {None, "request"} and not state.get("answer_generated"):
            return "资料模式（回答未通过核对）"
        return ("资料模式（模型暂不可用）" if state.get("model_failed") else
                "AI对话" if state.get("answer_generated") else "资料模式")

    def answer(self, state: TurnState, model_client=None):
        return AnswerComposer(self, state, model_client).run()


@lru_cache(maxsize=4)
def _service(path: str) -> SelectionAssistant:
    return SelectionAssistant(ChatStore(Path(path)))


_service_lock = Lock()


def get_chat_service() -> SelectionAssistant:
    with _service_lock:
        return _service(get_settings().chat_storage_path)
