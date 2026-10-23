import json
from concurrent.futures import ThreadPoolExecutor

import httpx2
import pytest
from fastapi.testclient import TestClient
from langchain_deepseek import ChatDeepSeek

from app import web
from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest
from app.chat_store import ChatStore, SessionConflict, SessionNotFound
from app.knowledge import KnowledgeBase, KnowledgeIndex, KnowledgeChunk
from app.model_api import ModelClient


@pytest.fixture
def assistant(tmp_path):
    class NoPublicSources:
        def collect(self, *args, **kwargs):
            return []
    return SelectionAssistant(agent_enabled=False, store=ChatStore(tmp_path / "sessions.sqlite3"), source_loader=NoPublicSources())


def say(assistant, message, previous=None):
    return assistant.turn(ChatRequest(message=message, session_id=previous.session_id if previous else None))


def last(session):
    return session.messages[-1]


def test_budget_then_use_then_followup_keeps_conditions(assistant):
    first = say(assistant, "预算四千")
    assert first.profile.budget_max == 4000
    assert "主要看重" in last(first).content
    second = say(assistant, "主要拍照，也希望续航好", first)
    assert second.profile.priorities == ["拍照", "续航"]
    assert last(second).cards[0].name == "Xiaomi 17T"
    assert all(card.price <= 4000 for card in last(second).cards)
    assert last(second).sources
    third = say(assistant, "第一款夜景怎么样？", second)
    assert third.profile.budget_max == 4000
    assert third.profile.priorities == ["拍照", "续航"]
    assert third.profile.focus_phone_id == "xiaomi-17t"
    assert {source.phone_id for source in last(third).sources} == {"xiaomi-17t"}
    assert "对应的实测" in last(third).content
    assert last(third).blocks
    assert '夜景' in last(third).content
    fourth = say(assistant, "它的续航多久？", third)
    assert {source.phone_id for source in last(fourth).sources} == {"xiaomi-17t"}


def test_ambiguous_pronoun_does_not_select_first_phone(assistant):
    first = say(assistant, "预算4000，主要拍照和续航")
    second = say(assistant, "它的续航怎么样？", first)
    assert second.profile.focus_phone_id is None
    assert "你指哪款" in last(second).content
    assert not last(second).sources


def test_budget_update_and_exclusion_persist(assistant):
    first = say(assistant, "预算4000，拍照和续航")
    second = say(assistant, "预算提高到5000", first)
    assert second.profile.budget_max == 5000
    assert second.profile.priorities == ["拍照", "续航"]
    third = say(assistant, "不考虑K80了", second)
    assert "redmi-k80" in third.profile.excluded_phones
    assert "redmi-k80" not in third.profile.candidate_ids
    fourth = say(assistant, "还是更看重续航", third)
    assert fourth.profile.priorities == ["续航", "拍照"]
    assert "redmi-k80" not in fourth.profile.candidate_ids
    fifth = say(assistant, "重新考虑K80", fourth)
    assert "redmi-k80" not in fifth.profile.excluded_phones
    assert fifth.profile.focus_phone_id == "redmi-k80"


@pytest.mark.parametrize("message,value", [("预算4k",4000),("预算0.4万",4000),("预算三千",3000),("预算一万二",12000),("预算三千五",3500),("预算一万二千五百",12500)])
def test_common_budget_expressions(assistant, message, value):
    result = say(assistant, message)
    assert result.profile.budget_max == value


def test_battery_capacity_is_not_a_budget_and_unlimited_is_distinct(assistant):
    first = say(assistant, "想了解一加13的6000mAh电池")
    assert first.profile.budget_max is None
    assert last(first).sources
    assert last(first).cards[0].price is None
    second = say(assistant, "预算不限，主要轻便，重新推荐", first)
    assert second.profile.budget_unlimited
    assert last(second).cards[0].id == "iphone-16"


def test_explicit_brand_and_budget_no_match_does_not_relax_conditions(assistant):
    first = say(assistant, "预算4000，主要拍照")
    second = say(assistant, "只考虑苹果", first)
    assert second.profile.preferred_brands == ["Apple"]
    assert not second.profile.candidate_ids
    assert "没有参考价与品牌条件同时符合" in last(second).content
    third = say(assistant, "预算提高到6000", second)
    assert third.profile.candidate_ids == ["iphone-16"]
    fourth = say(assistant, "品牌不限", third)
    assert not fourth.profile.preferred_brands


def test_exclude_brand_then_restore_keeps_budget(assistant):
    first = say(assistant, "预算4000，主要游戏")
    second = say(assistant, "不要小米", first)
    assert not second.profile.candidate_ids
    third = say(assistant, "品牌不限", second)
    assert third.profile.candidate_ids
    assert third.profile.budget_max == 4000


@pytest.mark.parametrize("name", ["K80 Pro", "iPhone16 Pro", "小米15", "一加15"])
def test_exact_models_never_use_base_model_sources(assistant, name):
    result = say(assistant, f"{name}拍照怎么样")
    assert "暂不在已维护资料" not in last(result).content
    assert result.discovered_phones
    assert last(result).cards[0].name != "REDMI K80"
    # A larger bundled corpus can contain the requested Pro model. It must
    # still never reuse the base model's record when the exact model is absent.
    phone = result.discovered_phones[0]
    known = {chunk.phone_name for chunk in assistant.knowledge.index.chunks}
    assert bool(last(result).sources) == (phone.name in known)
    chunks = {chunk.chunk_id: chunk for chunk in assistant.knowledge.index.chunks}
    assert all(chunks[source.id.split(':', 1)[1]].phone_name == phone.name
               for source in last(result).sources)


def test_other_phone_requires_unique_reference(assistant):
    first = say(assistant, "预算4000，比较小米17T和K100Pro，主要续航")
    assert len(first.profile.candidate_ids) == 2
    second = say(assistant, "另一款续航怎么样", first)
    assert "另一款是哪款" in last(second).content
    third = say(assistant, "小米17T续航怎么样", second)
    fourth = say(assistant, "另一款呢", third)
    assert fourth.profile.focus_phone_id == "redmi-k100-pro"


def test_durable_restore_and_session_isolation(assistant):
    first = say(assistant, "预算4000，主要续航")
    reloaded = SelectionAssistant(agent_enabled=False, store=ChatStore(assistant.store.path), source_loader=assistant.source_loader)
    second = say(reloaded, "预算提高到5000", first)
    assert second.profile.priorities == ["续航"]
    assert len(second.messages) == 4
    independent = say(reloaded, "想买手机")
    assert independent.profile.budget_max is None
    assert not independent.profile.priorities
    assert first.session_id != independent.session_id


def test_concurrent_save_cannot_overwrite_another_turn(assistant):
    session = assistant.store.create()
    copies = [assistant.store.load(session.session_id) for _ in range(2)]
    def save(copy):
        try:
            assistant.store.save(copy)
            return "saved"
        except SessionConflict:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, copies)) == ["conflict", "saved"]


def test_no_evidence_no_phone_claims(assistant):
    assistant.knowledge = KnowledgeBase(KnowledgeIndex(chunks=[KnowledgeChunk(
        phone_name="Unknown", title="Other", source_url="https://example.com", text="nothing relevant",
        document_id="doc", chunk_id="chunk")]))
    result = say(assistant, "预算4000，拍照")
    assert "没有相关资料" in " ".join(last(result).details)
    assert not last(result).blocks
    assert not last(result).blocks


def test_retrieval_failure_keeps_profile_and_does_not_invent_answer(assistant):
    class FailedKnowledge:
        def as_tool(self):
            return self
        def invoke(self, *args):
            raise RuntimeError("private failure")
    assistant.knowledge = FailedKnowledge()
    result = say(assistant, "预算4000，主要拍照")
    assert result.profile.budget_max == 4000
    assert result.profile.priorities == ["拍照"]
    assert not last(result).sources
    assert "private failure" not in last(result).model_dump_json()


def test_cancel_priority_and_reselect(assistant):
    first = say(assistant, "预算4000，主要游戏和拍照")
    second = say(assistant, "不玩游戏了", first)
    assert second.profile.priorities == ["拍照"]
    assert last(second).cards[0].id == "xiaomi-17t"


def sdk_model(handler):
    return ModelClient(ChatDeepSeek(model="deepseek-chat", api_key="synthetic-test-key",
        base_url="https://api.deepseek.com", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0, max_tokens=400))


def sdk_response(content):
    return httpx2.Response(200, json={"id":"mock", "object":"chat.completion", "created":0,
        "model":"deepseek-chat", "choices":[{"index":0,"message":{"role":"assistant", "content":json.dumps(content, ensure_ascii=False)},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}})


def test_actual_langchain_sdk_chat_intent_and_grounded_answer(assistant):
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return sdk_response({"quote":"预算4000，拍照", "budget_max":4000,"priorities":["拍照"],"preferred_brands":None})
        payload = json.loads(body["messages"][-1]["content"])
        source = payload["来源"][0]
        return sdk_response({"blocks":[{"phone_id":source["phone_id"],"text":"这款有5X长焦配置，可以先核对你的远景拍摄需求。没有夜景实测资料。","evidence_ids":[source["id"]]}]})
    assistant.model = sdk_model(handler)
    result = say(assistant, "预算4000，拍照")
    assert len(requests) == 2
    assert requests[0]["response_format"] == {"type":"json_object"}
    assert last(result).mode == "AI对话"
    assert "5X长焦" in last(result).blocks[0].text
    assert result.profile.budget_max == 4000


def test_cross_phone_model_citation_rejected_not_displayed(assistant):
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return sdk_response({"quote":"预算4000，拍照", "budget_max":4000, "priorities":["拍照"]})
        source = json.loads(body["messages"][-1]["content"])["来源"][0]
        return sdk_response({"blocks":[{"phone_id":"iphone-16","text":"UNSUPPORTED_9999","evidence_ids":[source["id"]]}]})
    assistant.model = sdk_model(handler)
    result = say(assistant, "预算4000，拍照")
    assert last(result).mode == "资料模式（回答未通过核对）"
    assert "UNSUPPORTED" not in last(result).model_dump_json()


def test_model_understanding_can_fill_budget_alongside_rule_topics(assistant):
    class IntentModel:
        available = True
        def complete_json(self, messages):
            if messages[0]["content"].startswith("提取"):
                return {"quote":"愿意花四千块，主要拍照", "budget_max":4000,"priorities":["拍照"]}
            raise TimeoutError()
    assistant.model = IntentModel()
    result = say(assistant, "我愿意花四千块，主要拍照")
    assert result.profile.budget_max == 4000
    assert last(result).cards


def test_unquoted_model_patch_is_rejected(assistant):
    class BadIntentModel:
        available = True
        def complete_json(self, messages):
            return {"quote":"invented budget", "budget_max":9000,"priorities":["游戏"]}
    assistant.model = BadIntentModel()
    result = say(assistant, "预算4000，主要拍照")
    assert result.profile.budget_max == 4000
    assert result.profile.priorities == ["拍照"]
    assert last(result).mode == "资料模式（模型暂不可用）"


@pytest.mark.parametrize("message,expected_brands", [
    ("先只看Xiaomi 17T", []),
    ("只考虑小米品牌，先看Xiaomi 17T", ["小米"]),
])
def test_model_focus_is_distinct_from_explicit_brand_preference(assistant, message, expected_brands):
    first = say(assistant, "预算5000，主要拍照和续航")
    class BrandModel:
        available = True
        def complete_json(self, messages):
            if messages[0]["content"].startswith("提取"):
                return {"quote":message,"preferred_brands":["小米"], "selected_phones":["Xiaomi 17T"]}
            raise TimeoutError()
    assistant.model = BrandModel()
    result = say(assistant, message, first)
    assert result.profile.preferred_brands == expected_brands
    assert result.profile.budget_max == 5000
    assert result.profile.priorities == ["拍照", "续航"]
    assert result.profile.focus_phone_id == "xiaomi-17t"


def test_model_followup_receives_previous_generated_answer(assistant):
    answer_payloads = []
    answer_text = "有5X长焦配置，但不能据此判断夜景实际效果。"
    def handler(request):
        body = json.loads(request.content)
        if body["messages"][0]["content"].startswith("提取"):
            text = body["messages"][-1]["content"]
            return sdk_response({"quote":text, **({"budget_max":4000, "selected_phones":["Xiaomi 17T"]}
                                if text == "预算4000，先看Xiaomi 17T" else {})})
        payload = json.loads(body["messages"][-1]["content"])
        answer_payloads.append(payload)
        source = payload["来源"][0]
        return sdk_response({"blocks":[{"phone_id":source["phone_id"],"text":answer_text,"evidence_ids":[source["id"]]}]})
    assistant.model = sdk_model(handler)
    first = say(assistant, "预算4000，先看Xiaomi 17T", None)
    second = say(assistant, "你刚说的夜景限制是什么？", first)
    assert last(second).mode == "AI对话"
    assert any(answer_text in m["content"] for m in answer_payloads[1]["历史"] if m["role"] == "assistant")


def test_model_failure_preserves_confirmed_state_and_returns_sources(assistant):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx2.Response(503,json={"error":{"message":"private-details"}})
    assistant.model = sdk_model(handler)
    result = say(assistant, "预算4000，主要拍照和续航")
    assert len(calls) == 1
    assert result.profile.budget_max == 4000
    assert last(result).sources
    assert last(result).mode == "资料模式（模型暂不可用）"
    assert "private-details" not in last(result).model_dump_json()


def test_http_create_send_restore_delete_and_missing_session(assistant, monkeypatch):
    monkeypatch.setattr(web, "get_chat_service", lambda: assistant)
    client = TestClient(web.app)
    assert client.get('/').headers['cache-control'] == 'no-store'
    session = client.post("/api/chat/sessions").json()
    result = client.post("/api/chat", json={"session_id":session["session_id"],"message":"预算4000，主要拍照"})
    assert result.status_code == 200
    restored = client.get(f'/api/chat/sessions/{session["session_id"]}')
    assert restored.json() == result.json()
    assert client.post("/api/chat",json={"message":" ","session_id":session["session_id"]}).status_code == 422
    assert client.get("/api/chat/sessions/../../env").status_code == 404
    assert client.post("/api/chat",json={"message":"hello","session_id":"bad"}).status_code == 422
    assert client.delete(f'/api/chat/sessions/{session["session_id"]}').status_code == 204
    assert client.get(f'/api/chat/sessions/{session["session_id"]}').status_code == 404
    assert client.post("/api/chat",json={"message":"hello","session_id":session["session_id"]}).status_code == 404
    with pytest.raises(SessionNotFound):
        assistant.store.load(session["session_id"])
