import json
from itertools import count

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from app import web
from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest, ChatSource, ChatMessage
from app.chat_store import ChatStore
from app.knowledge import KnowledgeBase, KnowledgeChunk, KnowledgeIndex
from app.tool_loop import AgentPolicy


IDS = count()


def tool(name, **args):
    return {"name": name, "args": args, "id": f"chat-call-{next(IDS)}", "type": "tool_call"}


def done(kind="answer", message="资料已核对"):
    return {"kind": kind, "message": message}


class Model:
    available = True
    last_usage = {"total_tokens": 20}

    def __init__(self, steps, patches=None):
        self.steps = iter(steps)
        self.patches = patches or {}
        self.contexts, self.histories = [], []
        self.intent_calls, self.answer_calls = 0, 0

    def complete_json(self, messages, **kwargs):
        if messages[0]["content"].startswith("提取"):
            self.intent_calls += 1
            text = messages[-1]["content"]
            return {"quote": text, **self.patches.get(text, {})}
        self.answer_calls += 1
        data = json.loads(messages[-1]["content"])
        return {"blocks": [{"phone_id": s["phone_id"], "text": "资料记载：" + s["quote"][:100],
                            "evidence_ids": [s["id"]]} for s in data["来源"]]}

    def invoke_tools(self, messages, schemas, **kwargs):
        self.histories.append(list(messages))
        self.contexts.append(json.loads(messages[1].content))
        step = next(self.steps)
        if isinstance(step, Exception):
            raise step
        return AIMessage(content=json.dumps(step, ensure_ascii=False), tool_calls=[] if isinstance(step, dict) else step)


class Public:
    def __init__(self):
        self.searches, self.fetches = [], []

    def collect(self, *args, **kwargs):
        return []

    def search(self, phone, query, **kwargs):
        self.searches.append((phone.id, query))
        return [ChatSource(id=phone.id + ":web-test", phone_id=phone.id, title=phone.name,
                url="https://example.com/phone", quote=phone.name + "公开搜索中的屏幕资料，正文尚未核对。",
                checked_on="2026-10-21", source_type="snippet", evidence_kind="unknown")]

    def fetch(self, phone, url, question, **kwargs):
        self.fetches.append((phone.id, url))
        return [ChatSource(id=phone.id + ":web-test", phone_id=phone.id, title=phone.name,
                url=url, quote=phone.name + "屏幕支持120Hz，仅为测试夹具。", checked_on="2026-10-21",
                source_type="page", evidence_kind="spec")]


def service(tmp_path, model, **kwargs):
    base = KnowledgeBase(KnowledgeIndex(chunks=[KnowledgeChunk(phone_name="一加 13", title="测试资料",
        source_url="https://example.com/local", text="一加 13电池容量6000mAh，仅为固定测试资料。",
        document_id="test-doc", chunk_id="test-chunk", checked_on="2026-10-21")]))
    return SelectionAssistant(ChatStore(tmp_path / "chat.sqlite3"), model_client=model,
                               knowledge=base, source_loader=kwargs.pop("source_loader", Public()), **kwargs)


def say(assistant, text, previous=None):
    return assistant.turn(ChatRequest(message=text, session_id=previous.session_id if previous else None))


def test_default_home_chat_agent_queries_one_phone_and_preserves_citations(tmp_path):
    text = "一加13的电池容量是多少"
    model = Model([[tool("search_phone_knowledge", phone_name="一加 13", query="电池容量")], done()],
                  {text: {"selected_phones": ["一加13"]}})
    assistant = service(tmp_path, model)
    session = say(assistant, text)
    reply = session.messages[-1]
    assert assistant.agent_enabled
    assert reply.mode == "AI对话" and "6000mAh" in reply.blocks[0].text
    assert reply.agent_trace["engine"] == "langgraph-chat-agent"
    assert reply.agent_trace["metrics"]["model_calls"] == 4
    results = [m for m in model.histories[1] if isinstance(m, ToolMessage)]
    assert len(results) == 1 and results[0].tool_call_id.startswith("chat-call-")
    assert reply.blocks[0].evidence_ids[0] == reply.sources[0].id


def test_long_tool_exchange_compacts_planning_context_without_erasing_evidence(tmp_path):
    text = "想了解一加13"
    model = Model([[tool("search_phone_knowledge", phone_name="一加 13", query="电池")], done()],
                  {text: {"selected_phones": ["一加13"]}})
    assistant = service(tmp_path, model, agent_policy=AgentPolicy(max_total_tokens=30000,max_context_characters=32000))
    assistant.knowledge = KnowledgeBase(KnowledgeIndex(chunks=[KnowledgeChunk(
        phone_name="一加 13", title="长文测试夹具", source_url=f"https://example.com/long-{i}",
        text=("一加13电池资料，仅为上下文压缩测试，不能作为真实配置。" * 26),
        document_id=f"long-{i}", chunk_id=f"long-{i}", checked_on="2026-10-22") for i in range(4)]))
    reply = say(assistant, text).messages[-1]
    assert reply.agent_trace["metrics"]["context_compactions"] >= 1
    assert "已收集的来源摘要" in json.loads(model.histories[1][1].content)
    assert len(reply.sources) == 4 and all(len(s.quote) > 600 for s in reply.sources)
    assert any(a["action"] == "context_compaction" for a in reply.agent_trace["actions"])


def test_budget_followup_passes_remembered_games_to_fetch_and_updates_old_page(tmp_path):
    class ReviewPublic(Public):
        def fetch(self, phone, url, question, **kwargs):
            self.need = question
            return [ChatSource(id=phone.id + ":web-test", phone_id=phone.id, title="实测测试夹具", url=url,
                quote="原神测试30分钟平均60fps；崩坏：星穹铁道测试30分钟平均59fps。屏幕刷新率120Hz。仅为虚构测试夹具。",
                checked_on="2026-10-22", source_type="page", evidence_kind="review")]
    text = "预算改到7000"
    public = ReviewPublic()
    model = Model([[tool("fetch_page", phone_name="一加 13", url="https://example.com/phone")], done()],
                  {text: {"budget_max": 7000}})
    assistant = service(tmp_path, model, source_loader=public)
    initial = assistant.store.create()
    initial.profile.budget_max = 7500
    initial.profile.priorities = ["游戏"]
    initial.profile.games = ["原神", "崩坏：星穹铁道"]
    initial.profile.high_refresh_required = True
    initial.profile.candidate_ids = initial.profile.explicit_candidate_ids = ["oneplus-13"]
    old = ChatSource(id="oneplus-13:web-test", phone_id="oneplus-13", title="旧规格", url="https://example.com/phone",
                     quote="旧摘录只有容量，没有游戏测试。", checked_on="2026-10-21", source_type="page", evidence_kind="spec")
    initial.messages = [ChatMessage(role="assistant", content="旧回答保持原样", sources=[old])]
    initial = assistant.store.save(initial)
    session = say(assistant, text, initial)
    assert all(word in public.need for word in ["原神", "星穹铁道", "高刷"])
    source = next(s for s in session.messages[-1].sources if s.id == old.id)
    assert source.evidence_kind == "review" and "60fps" in source.quote
    assert session.messages[0].sources[0].quote == old.quote


def test_agent_decides_to_ask_budget_without_querying(tmp_path):
    model = Model([done("clarify", "你的预算大概是多少？")])
    session = say(service(tmp_path, model), "想换个手机")
    reply = session.messages[-1]
    assert "预算" in reply.content and not reply.sources
    assert reply.agent_trace["decision_kind"] == "clarify"
    assert reply.agent_trace["metrics"]["tool_calls"] == 0
    assert model.intent_calls == 1 and model.answer_calls == 0


def test_fixed_mode_flag_does_not_enter_tool_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_AGENT_ENABLED", "0")
    text = "想了解一加13"
    model = Model([], {text: {"selected_phones": ["一加13"]}})
    assistant = service(tmp_path, model)
    reply = say(assistant, text).messages[-1]
    assert not assistant.agent_enabled and not model.histories
    assert reply.agent_trace['engine']=='shared-evidence-retrieval' and reply.sources
    assert not any(a.get('name') for a in reply.agent_trace['actions'])


def test_acknowledgement_updates_budget_and_keeps_other_conditions(tmp_path):
    model = Model([done("acknowledge", "预算已调整为5000，继续保留拍照和续航。")],
                  {"预算提高到5000": {"budget_max": 5000}})
    assistant = service(tmp_path, model)
    initial = assistant.store.create()
    initial.profile.budget_max = 4000
    initial.profile.priorities = ["拍照", "续航"]
    initial = assistant.store.save(initial)
    session = say(assistant, "预算提高到5000", initial)
    assert session.profile.budget_max == 5000 and session.profile.priorities == ["拍照", "续航"]
    assert model.contexts[0]["条件"]["budget_max"] == 5000
    assert session.messages[-1].agent_trace["metrics"]["tool_calls"] == 0


def test_followup_reuses_sources_and_restores_trace_from_sqlite(tmp_path):
    text = "一加13的电池容量是多少"
    model = Model([[tool("search_phone_knowledge", phone_name="一加 13", query="电池容量")], done(), done()],
                  {text: {"selected_phones": ["一加13"]}})
    assistant = service(tmp_path, model)
    first = say(assistant, text)
    second = say(assistant, "你刚才查到的容量再说一下", first)
    assert model.contexts[-1]["可复用来源"]
    assert "6000mAh" in model.contexts[-1]["最近对话"][-1]["content"]
    assert second.messages[-1].agent_trace["metrics"]["tool_calls"] == 0
    restored = assistant.store.load(second.session_id)
    assert restored.messages[-1].agent_trace == second.messages[-1].agent_trace
    assert restored.messages[-1].sources[0].id == first.messages[-1].sources[0].id


def test_new_session_does_not_inherit_agent_sources(tmp_path):
    text = "一加13的电池容量是多少"
    model = Model([[tool("search_phone_knowledge", phone_name="一加 13", query="电池容量")], done(),
                   done("clarify", "你希望了解哪款手机？")], {text: {"selected_phones": ["一加13"]}})
    assistant = service(tmp_path, model)
    first = say(assistant, text)
    second = say(assistant, "我想换手机")
    assert first.session_id != second.session_id
    assert not model.contexts[-1]["可复用来源"] and not model.contexts[-1]["最近对话"]
    assert not second.messages[-1].sources


def test_refresh_does_not_treat_previous_sources_as_fresh(tmp_path):
    text = "一加13的电池容量是多少"
    model = Model([[tool("search_phone_knowledge", phone_name="一加 13", query="电池容量")], done(),
                   done("clarify", "需要确认哪个渠道的当前售价？")], {text: {"selected_phones": ["一加13"]}})
    assistant = service(tmp_path, model)
    first = say(assistant, text)
    say(assistant, "刷新最新价格", first)
    assert model.contexts[-1]["可复用来源"]
    assert all(s['evidence_kind']!='price' or s['refresh_status']=='pending'
               for s in model.contexts[-1]['可复用来源'])


def test_chat_agent_rejects_cross_phone_answer_citation(tmp_path):
    class BadAnswer(Model):
        def complete_json(self, messages, **kwargs):
            result = super().complete_json(messages, **kwargs)
            if "blocks" in result:
                result["blocks"][0]["phone_id"] = "redmi-k80"
            return result
    text = "想了解一加13"
    model = BadAnswer([[tool("search_phone_knowledge", phone_name="一加 13", query="电池容量")], done()],
                      {text: {"selected_phones": ["一加13"]}})
    reply = say(service(tmp_path, model), text).messages[-1]
    assert reply.mode == "资料模式（回答未通过核对）"
    assert reply.blocks and all(b.phone_id == "oneplus-13" for b in reply.blocks)


def test_agent_handles_missing_local_then_searches_and_fetches_in_same_chat(tmp_path):
    text = "iPhone17屏幕怎么样"
    model = Model([
        [tool("search_phone_knowledge", phone_name="iPhone 17", query="屏幕")],
        [tool("web_search", phone_name="iPhone 17", query="iPhone 17 屏幕")],
        [tool("fetch_page", phone_name="iPhone 17", url="https://example.com/phone")], done()],
        {text: {"selected_phones": ["iPhone17"]}})
    public = Public()
    session = say(service(tmp_path, model, source_loader=public), text)
    reply = session.messages[-1]
    assert len(public.searches) == len(public.fetches) == 1
    assert reply.agent_trace["stop_reason"] == "model_finished"
    assert reply.agent_trace["metrics"]["model_calls"] == 6
    assert len(reply.sources) == 1 and reply.sources[0].source_type == "page"
    assert session.discovered_phones[0].source_urls


@pytest.mark.parametrize("name,args,status", [
    ("web_search", {"phone_name": "iPhone 17 Pro", "query": "iPhone 17 Pro 屏幕"}, "invalid_scope"),
    ("web_search", {"phone_name": "一加 13", "query": "屏幕"}, "invalid_arguments"),
    ("fetch_page", {"phone_name": "一加 13", "url": "http://127.0.0.1"}, "invalid_url"),
    ("web_search", {"phone_name": "一加 13", "query": 123}, "invalid_arguments"),
])
def test_chat_agent_rejects_invalid_tool_request(tmp_path, name, args, status):
    text = "想了解一加13"
    model = Model([[tool(name, **args)], done()], {text: {"selected_phones": ["一加13"]}})
    public = Public()
    reply = say(service(tmp_path, model, source_loader=public), text).messages[-1]
    event = next(e for e in reply.agent_trace["actions"] if e["action"] == "tool")
    assert event["result"]["status"] == status
    assert not public.searches and not public.fetches


def test_tool_limit_still_keeps_collected_sources(tmp_path):
    text = "想了解一加13"
    model = Model([[tool("search_phone_knowledge", phone_name="一加 13", query="电池容量")]],
                  {text: {"selected_phones": ["一加13"]}})
    reply = say(service(tmp_path, model, agent_policy=AgentPolicy(max_tool_calls=1)), text).messages[-1]
    assert reply.agent_trace["stop_reason"] == "tool_call_limit"
    assert reply.sources and reply.blocks


def test_mid_loop_model_failure_keeps_sources_and_conditions(tmp_path):
    text = "想了解一加13"
    model = Model([[tool("search_phone_knowledge", phone_name="一加 13", query="电池容量")], TimeoutError("private")],
                  {text: {"selected_phones": ["一加13"]}})
    reply = say(service(tmp_path, model), text).messages[-1]
    assert reply.agent_trace["stop_reason"] == "model_error"
    assert reply.sources and reply.mode == "资料模式（模型暂不可用）"
    assert "private" not in reply.model_dump_json()


def test_excluded_phone_is_not_in_next_turn_tool_scope(tmp_path):
    model = Model([done("acknowledge", "已排除K80。")], {"不考虑K80了": {"excluded_phones": ["K80"]}})
    assistant = service(tmp_path, model)
    first = assistant.store.create()
    first.profile.budget_max, first.profile.priorities = 4000, ["续航"]
    first = assistant.store.save(first)
    result = say(assistant, "不考虑K80了", first)
    assert "redmi-k80" in result.profile.excluded_phones
    assert "REDMI K80" not in model.contexts[0]["可查询机型"].values()


def test_home_http_endpoint_returns_agent_trace_and_restores_it(tmp_path, monkeypatch):
    model = Model([done("clarify", "你希望预算控制在多少元？")])
    assistant = service(tmp_path, model)
    monkeypatch.setattr(web, "get_chat_service", lambda: assistant)
    client = TestClient(web.app)
    result = client.post("/api/chat", json={"message": "想换手机"})
    assert result.status_code == 200
    data = result.json()
    trace=data['messages'][-1]['agent_trace']
    assert trace['engine']=='conversation-control' and trace['decision_kind']=='clarify'
    assert trace['metrics']['model_calls']==1 and trace['metrics']['tool_calls']==0
    assert client.get('/api/chat/sessions/' + data["session_id"]).json() == data
