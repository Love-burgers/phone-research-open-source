"""Regressions for the user's price-tier, gaming and stale-candidate report.

Phone names, prices and performance below are synthetic fixtures, not product recommendations.
"""

import json
import socket

import pytest

from langchain_core.messages import AIMessage

from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest, ChatSource, SelectionProfile
from app.chat_store import ChatStore
from app.chat_intent import read_budget
from app.phone_sources import PublicPhoneSources
from app.selection_catalog import load_catalog, shortlist
from app.selection_requirements import card_matches, literal_requirements
from app.phone_sources import extract_phone_excerpt, extract_price_options, phone_page_links
from app.phone_names import register_phone, parse_phone_names


@pytest.fixture(autouse=True)
def public_fixture_dns(monkeypatch):
    # Offline URL validation still runs, with public fixture DNS instead of traffic.
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))])


def test_target_price_parsing_and_old_database_defaults():
    profile = SelectionProfile.model_validate({"budget_max": 7500, "priorities": ["游戏"]})
    assert not profile.high_refresh_required and profile.games == []
    assert read_budget("我想要7000元的手机") == 7000
    assert read_budget("要7000左右的手机") == 7000
    assert read_budget("想了解一加13的6000mAh电池") is None
    literal_requirements(profile, "预算7500推荐一款手机，游戏性能优先。要求高刷")
    assert profile.high_refresh_required and profile.recommendation_count == 1
    literal_requirements(profile, "玩原神和崩坏：星穹铁道")
    literal_requirements(profile, "想要7000元的手机")
    profile.budget_max = 7000
    assert profile.price_preference == "near_budget"
    assert profile.games == ["原神", "崩坏：星穹铁道"]
    selected = shortlist(load_catalog(), profile)
    assert all(5600 <= load_catalog()[pid].price <= 7000 for pid in selected)


def test_card_does_not_confuse_specs_with_game_benchmarks():
    profile = SelectionProfile(priorities=["游戏"], games=["原神", "崩坏：星穹铁道"], high_refresh_required=True)
    source = ChatSource(id="fixture", phone_id="test", title="测试夹具", url="https://example.com",
        quote="屏幕支持144Hz刷新率，芯片是测试芯片；不含游戏实测。", checked_on="2026-10-22", evidence_kind="spec")
    labels, urls = card_matches("test", profile, [source])
    assert "游戏表现：待补实测" in labels and "高刷：来源标注144Hz" in labels
    assert "原神：待查" in labels and urls
    source.quote = "触控采样率120Hz，不是屏幕刷新率。"
    assert "高刷：待核对" in card_matches("test", profile, [source])[0]
    source.source_type = "snippet"
    source.quote = "屏幕支持144Hz刷新率"
    assert "高刷：待核对" in card_matches("test", profile, [source])[0]


def test_comparison_paragraph_about_a_different_subject_is_not_phone_evidence():
    html = '<title>游戏手机推荐</title><main><h2>iQOO 15</h2><p>iQOO 15 Ultra的亲民版，7000mAh电池和高刷屏幕。</p></main>'
    assert extract_phone_excerpt("iQOO 15 Ultra", html, "游戏 屏幕") is None


def test_multi_version_prices_and_short_refresh_values_are_preserved():
    html = '''<title>iQOO 15 Ultra 官方测试夹具</title><main><h1>iQOO 15 Ultra</h1>
      <section><h2>建议零售价</h2><p>16GB+256GB：6200. 00元<br>16GB+512GB：7000.00元<br>16GB+1TB：8000元</p></section>
      <section><p class="attr-item-name">刷新率</p><div><p>最高支持144Hz</p></div></section>
      <section><p class="attr-item-name">电池容量</p><p>6000mAh</p><p class="attr-item-name">充电</p><p>OTG反向充电</p></section></main>'''
    offers, quote = extract_price_options("iQOO 15 Ultra", html)
    assert [(o.variant, o.price) for o in offers] == [("16GB+256GB", 6200), ("16GB+512GB", 7000), ("16GB+1TB", 8000)]
    excerpt = extract_phone_excerpt("iQOO 15 Ultra", html, "预算 价格 高刷")
    assert "刷新率：最高支持144Hz" in excerpt
    assert "电池容量：OTG" not in excerpt and "7000" in quote
    mixed = html.replace("16GB+512GB：7000.00元", "16GB+256GB：7000.00元")
    assert all(o.variant != "16GB+256GB" for o in extract_price_options("iQOO 15 Ultra", mixed)[0])


def test_review_excerpt_keeps_both_requested_games_and_their_conditions():
    html = '''<title>iQOO 15 Ultra 评测</title><main><h1>iQOO 15 Ultra 评测</h1>
      <p>原神测试，极高画质60帧，连续30分钟，平均59.5帧，温度40℃。测试夹具。</p>
      <p>崩坏：星穹铁道测试，非常高画质，连续30分钟，平均59.1帧。测试夹具。</p>
      <p>iQOO 15 Ultra支持高速充电，处理器和相机等配置信息，仅为测试夹具。</p></main>'''
    excerpt = extract_phone_excerpt("iQOO 15 Ultra", html, "原神 崩坏星穹铁道 帧率 发热 实测")
    assert "59.5" in excerpt and "59.1" in excerpt and "30分钟" in excerpt


def test_version_selection_changes_with_target_budget_without_inventing_price(tmp_path):
    from app.chat_schemas import ChatSession
    phone = register_phone(parse_phone_names("iQOO 15 Pro")[0], {})
    session = ChatSession(session_id="a" * 32, discovered_phones=[phone])
    session.profile.budget_max = 7500
    service = SelectionAssistant(ChatStore(tmp_path / "price.sqlite3"))
    state = {"session": session, "catalog": {phone.id: phone.model_copy(deep=True)}}
    source = ChatSource(id="fixture-prices", phone_id=phone.id, title="官方价格测试夹具", url="https://www.iqoo.com/fixture",
        quote="16GB+256GB：6200元；16GB+512GB：6900元；16GB+1TB：8000元。虚构测试数据。",
        checked_on="2026-10-22", source_type="page", evidence_kind="price", market="中国大陆",
        reference_price=6200, price_variant="16GB+256GB", price_options=[
            {"variant": "16GB+256GB", "price": 6200}, {"variant": "16GB+512GB", "price": 6900},
            {"variant": "16GB+1TB", "price": 8000}])
    service._remember_sources(state, [source])
    assert phone.price == 6200
    session.profile.budget_max = 7000
    session.profile.price_preference = "near_budget"
    service._remember_sources(state, [source])
    assert phone.price == 6900 and phone.variant == "16GB+512GB"
    session.profile.budget_max = 6500
    service._remember_sources(state, [source])
    assert phone.price == 6200 and len(phone.price_options) == 3


def test_navigation_exposes_only_same_model_public_links_and_survives_cache(tmp_path):
    phone = register_phone(parse_phone_names("iQOO 15 Pro")[0], {})
    html = '''<title>iQOO 15 Pro 规格测试夹具</title><main><h1>iQOO 15 Pro</h1>
      <p>iQOO 15 Pro屏幕支持144Hz，仅为虚构测试夹具。</p>
      <a href="/1234/123456/article.shtml">评测</a>
      <a href="#section">专业评测</a>
      <a href="https://8.8.4.4/test-review">iQOO 15 Pro游戏评测</a>
      <a href="http://127.0.0.1/private">iQOO 15 Pro评测</a>
      <a href="https://8.8.4.4/wrong">iQOO 15 Pro Max评测</a></main>'''
    url = "https://8.8.8.8/index123456.shtml"
    links = phone_page_links(phone, url, html, "原神 实测")
    assert {x["url"] for x in links} == {"https://8.8.8.8/1234/123456/article.shtml", "https://8.8.4.4/test-review"}
    fetched = []
    loader = PublicPhoneSources(tmp_path / "cache", fetcher=lambda u: fetched.append(u) or html)
    first = loader.fetch_result(phone, url, "原神 实测")
    second = loader.fetch_result(phone, url, "原神 实测")
    assert first['sources'] == second['sources'] and first['candidates'] == second['candidates']
    assert first['lookup_origin'] == 'network' and second['lookup_origin'] == 'cache'
    assert first["candidates"] and len(fetched) == 1
    assert loader.fetch(phone, url, "原神 实测") == first["sources"]


def test_discovery_uses_model_chosen_query_and_directory_without_hidden_retry(tmp_path):
    queries, fetches = [], []
    def search(q, n):
        queries.append(q)
        return [{"title": "7500 CPU测试夹具", "snippet": "与手机无关", "url": "https://8.8.8.8/cpu"}]
    html = '''<div class="rank-list__item"><div class="rank__name"><a href="https://8.8.8.8/a">iQOO 15 Pro</a></div><div class="rank__price">￥6800</div></div>
      <div class="rank-list__item"><div class="rank__name"><a href="https://8.8.8.8/b">iQOO 16 Pro</a></div><div class="rank__price">￥3000</div></div>'''
    loader = PublicPhoneSources(tmp_path / "cache", searcher=search, fetcher=lambda u: fetches.append(u) or html)
    leads = loader.discover("7500元 游戏手机", budget_max=7500)
    assert len(queries) == 1 and not fetches
    assert leads[0]["url"] == "https://8.8.8.8/cpu"
    leads = loader.discover("寻找机型", url=leads[0]["url"])
    assert len(queries) == 1 and len(fetches) == 1
    assert leads[0]["title"] == "iQOO 15 Pro" and leads[0]["status"] == "directory_lead_only"
    assert "尚未核对" in leads[0]["snippet"] and "reference_price" not in leads[0]


class ScriptedModel:
    available = True
    last_usage = {"total_tokens": 50}

    def __init__(self, steps):
        self.steps = iter(steps)
        self.histories = []

    def complete_json(self, messages, **kwargs):
        if messages[0]["content"].startswith("提取"):
            text = messages[-1]["content"]
            requirements = SelectionProfile()
            literal_requirements(requirements, text)
            patch = {"quote": text, "budget_max": read_budget(text), "priorities": ["游戏"]}
            patch.update({k: getattr(requirements, k) for k in requirements.model_fields_set
                          if k in {"price_preference", "recommendation_count", "high_refresh_required", "games"}
                          and getattr(requirements, k) != getattr(SelectionProfile(), k)})
            return patch
        data = json.loads(messages[-1]["content"])
        return {"blocks": [{"phone_id": s["phone_id"], "text": s["quote"][:300], "evidence_ids": [s["id"]]}
                           for s in data["来源"]]}

    def invoke_tools(self, messages, schemas, **kwargs):
        self.histories.append(messages)
        step = next(self.steps)
        if isinstance(step, list):
            return AIMessage(content="", tool_calls=step)
        return AIMessage(content=json.dumps(step, ensure_ascii=False))


def call(name, **args):
    return {"name": name, "args": args, "id": name + str(len(str(args))), "type": "tool_call"}


def test_price_tier_discovery_fetch_selection_and_budget_change(tmp_path):
    urls = {"iQOO 15 Pro": "https://www.iqoo.com/fixture-15", "iQOO 16 Pro": "https://www.iqoo.com/fixture-16"}
    def search(query, limit):
        name = "iQOO 16 Pro" if "7000" in query else "iQOO 15 Pro"
        return [{"title": name + "测试夹具", "snippet": "仅为测试，非真实产品", "url": urls[name]}]
    def fetch(url):
        name = next(n for n, u in urls.items() if u == url)
        price = 6999 if "15" in name else 6499
        return f'<title>{name} 测试夹具</title><main><p>{name} 16GB+512GB 售价{price}元，屏幕刷新率144Hz，骁龙8至尊版。仅为测试夹具。</p><p>{name} 原神和崩坏：星穹铁道实测，30分钟帧率59fps，温度40度。仅为虚构测试夹具。</p></main>'
    steps = []
    for name, query in (("iQOO 15 Pro", "7500 游戏 高刷"), ("iQOO 16 Pro", "7000 游戏 高刷")):
        steps += [[call("discover_phones", query=query)], [call("fetch_page", phone_name=name, url=urls[name])],
                  {"kind": "answer", "candidate_names": [name]}]
    assistant = SelectionAssistant(ChatStore(tmp_path / "sessions.sqlite3"), model_client=ScriptedModel(steps),
        source_loader=PublicPhoneSources(tmp_path / "cache", searcher=search, fetcher=fetch))
    first = assistant.turn(ChatRequest(message="预算7500推荐一款手机，游戏性能优先。要求高刷"))
    assert len(first.messages[-1].cards) == 1
    assert first.messages[-1].cards[0].name == "iQOO 15 Pro"
    assert first.messages[-1].cards[0].price == 6999
    second = assistant.turn(ChatRequest(message="想要7000元的手机", session_id=first.session_id))
    assert second.profile.budget_max == 7000 and second.profile.high_refresh_required
    assert second.profile.candidate_ids != first.profile.candidate_ids
    assert second.messages[-1].cards[0].name == "iQOO 16 Pro"
    assert second.messages[-1].cards[0].price == 6499
    assert second.messages[-1].agent_trace["metrics"]["search_calls"] >= 1
    restored = assistant.store.load(second.session_id)
    assert restored.profile.high_refresh_required and restored.profile.recommendation_count == 1


def test_model_cannot_stop_at_local_specs_when_user_needs_game_review(tmp_path):
    # First premature finish is sent back as a missing-evidence observation.
    model = ScriptedModel([{"kind": "answer"}, [call("web_search", phone_name="REDMI K100 Pro",
                           query="REDMI K100 Pro 原神 实测 帧率")], {"kind": "answer"}])
    loader = PublicPhoneSources(tmp_path / "cache", searcher=lambda *a: [])
    assistant = SelectionAssistant(ChatStore(tmp_path / "sessions.sqlite3"), model_client=model, source_loader=loader)
    session = assistant.turn(ChatRequest(message="预算4000，玩原神，想了解REDMI K100 Pro"))
    # Explicit phone selection is supplied separately since this fake doesn't extract names.
    actions = session.messages[-1].agent_trace["actions"]
    assert any(a.get("validation_feedback") for a in actions)
    assert any(a.get("name") == "web_search" for a in actions)
    assert "没核实到" in session.messages[-1].content


def test_unverified_or_excluded_discovery_does_not_become_recommendation(tmp_path):
    model = ScriptedModel([[call("discover_phones", query="7500 游戏")],
                           {"kind": "answer", "candidate_names": ["iQOO 15 Pro"]}])
    loader = PublicPhoneSources(tmp_path / "cache", searcher=lambda *a: [{"title": "iQOO 15 Pro", "snippet": "测试",
                                                                 "url": "https://www.iqoo.com/fixture"}])
    assistant = SelectionAssistant(ChatStore(tmp_path / "sessions.sqlite3"), model_client=model, source_loader=loader)
    session = assistant.turn(ChatRequest(message="预算7500推荐一款手机，游戏性能优先。要求高刷"))
    assert not session.profile.candidate_ids and not session.messages[-1].blocks
    assert not session.messages[-1].cards
    assert "没核实到" in session.messages[-1].content


def test_old_session_restores_games_without_rewriting_old_answers(tmp_path):
    from app.chat_schemas import ChatMessage
    model = ScriptedModel([{"kind": "answer", "candidate_names": []}])
    loader = PublicPhoneSources(tmp_path / "cache", searcher=lambda *a: [])
    assistant = SelectionAssistant(ChatStore(tmp_path / "sessions.sqlite3"), model_client=model, source_loader=loader)
    session = assistant.store.create()
    session.profile = SelectionProfile.model_validate({"budget_max": 7500, "priorities": ["游戏"]})
    session.messages = [ChatMessage(role="user", content="玩原神和崩坏星穹铁道，要求高刷"),
                        ChatMessage(role="assistant", content="原历史回答，必须保留")]
    # Simulate the legacy on-disk schema, rather than saving modern defaults.
    with assistant.store._connect() as db:
        legacy = session.model_dump()
        legacy["profile"] = {"budget_max": 7500, "priorities": ["游戏"]}
        db.execute("UPDATE sessions SET data=? WHERE id=?", (json.dumps(legacy, ensure_ascii=False), session.session_id))
    result = assistant.turn(ChatRequest(message="想要7000左右的手机", session_id=session.session_id))
    assert result.profile.budget_max == 7000
    assert result.profile.high_refresh_required and result.profile.games == ["原神", "崩坏：星穹铁道"]
    assert result.messages[1].content == "原历史回答，必须保留"


def test_local_only_request_does_not_execute_public_search(tmp_path):
    model = ScriptedModel([[call("discover_phones", query="游戏手机")], {"kind": "answer", "candidate_names": []}])
    loader = PublicPhoneSources(tmp_path / "cache", searcher=lambda *a: pytest.fail("Unexpected external search"))
    assistant = SelectionAssistant(ChatStore(tmp_path / "sessions.sqlite3"), model_client=model, source_loader=loader)
    reply = assistant.turn(ChatRequest(message="预算7500游戏手机，只用本地资料")).messages[-1]
    assert any(a.get("result", {}).get("status") == "local_only" for a in reply.agent_trace["actions"])
    assert "已尝试联网" not in reply.content
