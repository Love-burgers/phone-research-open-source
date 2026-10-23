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
from .tool_loop import AgentPolicy
from .chat_intent import TOPIC_WORDS, understand_intent, conversation_candidates, resolve_references, mentioned_phones
from .conversation_policy import route_turn
from .knowledge import KnowledgeBase
from .model_api import ModelClient
from .selection_catalog import (DATA_DIR, eligible, load_catalog, shortlist, select_variant, variant_eligible,
                               minimum_storage, storage_gb, select_offer, source_price_options)
from .selection_catalog import CatalogPhone
from .phone_names import parse_phone_names, register_phone
from .phone_sources import PublicPhoneSources
from .source_rules import official_phone_url
from .selection_requirements import literal_requirements, card_matches
from .selection_presentation import configuration_facts
from .evidence_rules import game_in_text

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


class SelectionAssistant:
    def __init__(self, store: ChatStore, model_client=None, catalog=None, knowledge=None, source_loader=None,
                 *, agent_enabled=None, agent_policy=None):
        self.store = store
        settings = get_settings()
        self.catalog = catalog if catalog is not None else load_catalog()
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
        self.agent_policy = agent_policy or AgentPolicy(max_tool_calls=8, max_model_calls=6,
            max_total_tokens=settings.chat_turn_token_budget,
            max_context_characters=settings.chat_context_characters)
        graph = StateGraph(TurnState)
        for name in ("understand", "clarify", "retrieve", "answer", "agent_turn"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "understand")
        graph.add_conditional_edges("understand", lambda state: (
            "agent_turn" if self.agent_enabled and self.model.available
            else "clarify" if state.get("clarification") else "retrieve"))
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
        session.messages += [ChatMessage(role="user", content=request.message), result["reply"]]
        return self.store.save(session)

    def understand(self, state: TurnState):
        session = state["session"]
        profile = session.profile
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
        catalog = {pid: phone.model_copy(deep=True) for pid, phone in self.catalog.items()}
        catalog.update((phone.id, CatalogPhone.model_validate(phone.model_dump()))
                       for phone in session.discovered_phones)
        # Migrate reusable configuration tags as well as measured-review tags.
        # Without these, a follow-up discarded verified parameter-only choices
        # as if their main use had never been researched.
        saved_sources = [source for message in session.messages[-8:] if message.role == "assistant"
                         for source in message.sources]
        for phone in session.discovered_phones:
            configured = [topic for topic in TOPIC_WORDS if configuration_facts(phone, [topic], saved_sources)]
            phone.topics = list(dict.fromkeys([*phone.topics, *configured]))
            catalog[phone.id].topics = list(phone.topics)
        text, ack = state["message"], []
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
        selected_mentions = intent.selected
        for phone in catalog.values():
            select_variant(phone, profile)
        model_failed, ack = intent.model_failed, intent.acknowledged or []
        # Explicit model-name detection remains available for reference resolution,
        # while condition changes come from exactly one parser per turn.
        mentions = any(pid in selected_mentions and not excluded for pid, _, excluded in mentioned_phones(text, catalog))
        available_ids = conversation_candidates(profile, session.messages, catalog)
        # Rotate away from the current batch; previously considered phones are
        # still eligible on later turns unless the user explicitly excludes them.
        alternative_ids = list(available_ids) if intent.candidate_action == "alternatives" else []
        clarification = ''
        if not selected_mentions:
            selected_mentions, clarification = resolve_references(text, profile, session.messages, catalog)
            selected_mentions = [pid for pid in selected_mentions if pid not in profile.excluded_phones]
        selected_mentions = list(dict.fromkeys(selected_mentions))
        known_conflicts = {pid for pid in catalog if any('不符合' in label
                           for label in card_matches(pid, profile, saved_sources)[0])}
        incompatible = any(pid in catalog and (not variant_eligible(catalog[pid], profile)
                           or pid in known_conflicts) for pid in available_ids)
        route = route_turn(previous_profile, profile, intent, has_candidates=bool(available_ids),
                           selected=selected_mentions, incompatible=incompatible)
        # Rechecking constraints must not erase a user-pinned comparison list.
        # Only an explicit fresh selection (or a newly requested price tier)
        # releases it; ordinary revisions still validate its known constraints.
        reselect = intent.reselect or route.operation == 'alternatives' or (
            profile.price_preference == 'near_budget' and any(previous_profile[field] != getattr(profile, field)
                for field in ('budget_max', 'price_preference')))
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
                    "excluded_brands", "excluded_phones", "requirements", "high_refresh_required"))
                profile.candidate_ids = (retained[:profile.recommendation_count] if retained and not reselect and
                                         not rerank else shortlist({pid:p for pid,p in catalog.items() if pid not in known_conflicts}, profile))
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
                    profile.candidate_ids = selected_mentions
                    profile.focus_phone_id = None
                    if mentions:
                        profile.explicit_candidate_ids = selected_mentions
                else:
                    profile.focus_phone_id = selected_mentions[0]
                    if selected_mentions[0] not in profile.candidate_ids:
                        profile.candidate_ids = list(dict.fromkeys(selected_mentions + profile.candidate_ids))[:5]
                    if mentions:
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
        if not clarification and not selected_mentions and not profile.explicit_candidate_ids:
            if profile.budget_max is None and not profile.budget_unlimited:
                clarification = "你希望预算控制在多少元？也可以直接指定想了解的机型。"
            elif not profile.priorities and not profile.games:
                clarification = "预算已记住。你主要看重拍照、续航、游戏，还是小屏手感？可以说一到两项。"
            elif not profile.candidate_ids and not (self.agent_enabled and self.model.available and not model_failed):
                clarification = "现有资料范围内，没有参考价与品牌条件同时符合的机型。我保留了你的条件；可以调整预算或品牌，也可以指定机型核对资料。"
        if alternative_ids:
            profile.candidate_ids = [pid for pid in profile.candidate_ids if pid not in alternative_ids]
        selected = ([profile.focus_phone_id] if profile.focus_phone_id else profile.candidate_ids)
        query = " ".join([text, *profile.priorities, *profile.games, *profile.requirements,
                          "高刷" if profile.high_refresh_required else "", "配置" if detail_followup else "配置 价格"])
        return {"clarification": clarification, "selected": selected, "model_failed": model_failed,
                "acknowledged": ack, "query": query, "catalog": catalog, "answer_generated": False,
                "intent_usage": getattr(self.model, "last_usage", None), "alternative_ids": alternative_ids,
                "recommendation_turn": route.recommends and not selected_mentions,
                "detail_followup": detail_followup,
                "response_focus": response_focus, "focus_terms": focus_terms, "focus_topics": focus_topics,
                "conversation_route": {"operation": route.operation, "changed_fields": list(route.changed_fields),
                                       "focus": response_focus, "reason": route.reason},
                "intent_failure": intent.error_type, "intent_repairs": intent.grounding_repairs or []}

    def clarify(self, state: TurnState):
        return {"reply": ChatMessage(role="assistant", content="。".join(state["acknowledged"] + [state["clarification"]]),
                                      suggestions=["预算4000，主要拍照和续航", "预算3000，主要玩游戏"],
                                      mode=self.mode(state))}

    def retrieve(self, state: TurnState):
        catalog, sources = state["catalog"], []
        search = self.knowledge.as_tool()
        refresh = bool(re.search(r"刷新|重新查|更新资料|最新资料|最新价格|今天.*价", state["message"]))
        gaming = "游戏" in state["session"].profile.priorities or bool(state["session"].profile.games) or any(
            word in state["message"] for word in TOPIC_WORDS["游戏"])
        def collect(phone_id):
            local = []
            try:
                matches = search.invoke({"phone_name": catalog[phone_id].name, "need": state["query"]})
                for match in matches:
                    local.append(ChatSource(id=f"{phone_id}:{match['chunk_id']}", phone_id=phone_id,
                                            title=match["title"], url=match["source_url"], quote=match["text"],
                                            checked_on=match.get("checked_on") or catalog[phone_id].checked_on,
                                            published_on=match.get("published_on"), market=match.get("market") or "地区未确认",
                                            evidence_kind=match.get("evidence_kind", "unknown"),
                                            reference_price=match.get("reference_price"), price_variant=match.get("price_variant")))
            except Exception as error:
                logger.warning("retrieval_failed origin=local error_type=%s", type(error).__name__)
            has_review = any(source.evidence_kind == "review" for source in local)
            has_price = catalog[phone_id].price is not None or any(source.reference_price for source in local)
            from .selection_requirements import local_only
            public_allowed = state.get("search_mode") != "local" and not local_only(state["message"])
            if public_allowed and (state.get("search_mode") == "web" or not local or refresh or gaming and (not has_review or not has_price)):
                try:
                    public = self.source_loader.collect(catalog[phone_id], state["query"], refresh=refresh)
                    local.extend(source for source in public if source.phone_id == phone_id)
                except Exception as error:
                    logger.warning("retrieval_failed origin=public error_type=%s", type(error).__name__)
            # A known price/review must survive a failed refresh; avoid duplicate URL/text evidence.
            return list({(source.url, source.quote): source for source in local}.values())
        with ThreadPoolExecutor(max_workers=min(3, len(state["selected"])) or 1) as pool:
            futures=[pool.submit(copy_context().run,collect,pid) for pid in state['selected']]
            for future in futures:
                sources.extend(future.result())
        self._remember_sources(state, sources)
        return {"sources": sources}

    def _remember_sources(self, state, sources):
        catalog = state["catalog"]
        profile = state["session"].profile
        for phone in state["session"].discovered_phones:
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
                    return (not official_phone_url(phone.brand,source.url),not bool(allowed),
                            -int(source.checked_on.replace('-','')), -value if profile.price_preference=='near_budget' and allowed else value)
                for source in sorted(matched, key=price_source_order):
                    if (source.market == "中国大陆"
                        and (official_phone_url(phone.brand, source.url) or source.price_options
                             and re.search(r"中国大陆|国行|人民币|\bRMB\b|国补", source.quote, re.I))):
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
        try:
            update=self.knowledge.remember_sources(result['sources'],state['catalog'])
            if result['reply'].agent_trace is not None:
                result['reply'].agent_trace['local_knowledge_update']=update
        except Exception as error:
            logger.warning("knowledge_update_failed error_type=%s", type(error).__name__)
            if result['reply'].agent_trace is not None:
                result['reply'].agent_trace['local_knowledge_update']={'status':'error','error_type':type(error).__name__}
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
