"""Offline acceptance of archived retrieval and complete HTTP session flows."""

import pytest
from fastapi.testclient import TestClient

from app import web
from app.chat import SelectionAssistant
from app.chat_store import ChatStore, SessionConflict
from app.knowledge import KnowledgeBase, KnowledgeIndex, build_index
from app.selection_catalog import DATA_DIR


class NoModel:
    available = False


class NoPublicSources:
    def collect(self, *args, **kwargs):
        return []


@pytest.fixture
def archived_search():
    index = KnowledgeIndex(chunks=[
        *build_index(DATA_DIR / "selection_phones.jsonl").chunks,
        *build_index(DATA_DIR / "selection_reviews.jsonl").chunks,
    ])
    return KnowledgeBase(index).as_tool()


@pytest.mark.parametrize("phone,query,urls", [
    ("Xiaomi 17T", "长焦", {"https://m.mi.com/commodity/detail/24814"}),
    ("REDMI K80", "电池容量", {"https://m.mi.com/commodity/detail/10050084"}),
    ("一加 13", "无线充电", {"https://www.oneplus.com/cn/13/specs"}),
    ("Xiaomi 17", "原神功耗", {
        "https://www.antutu.com/doc/135323",
        "https://www.3elife.net/Art/test/202509/26/104460_0.html",
    }),
    ("iPhone 17", "256GB价格", {"https://www.apple.com.cn/shop/buy-iphone/iphone-17"}),
    ("REDMI K80 Pro", "续航", set()),
], ids=["telephoto", "battery", "charging", "game-reviews", "variant-price", "missing-pro"])
def test_reviewed_archive_retrieval_and_provenance(archived_search, phone, query, urls):
    # These are frozen archive expectations, not a check of today's phone facts.
    hits = archived_search.invoke({"phone_name": phone, "need": query})
    assert {hit["source_url"] for hit in hits} == urls
    for hit in hits:
        assert hit["phone_name"] == phone
        assert hit["document_id"] and hit["chunk_id"]
        assert hit["checked_on"] == "2026-10-11"
        assert hit["text"].strip()
    if phone == "Xiaomi 17":
        assert all(hit["evidence_kind"] == "review" for hit in hits)
        assert all(hit["published_on"] == "2025-09-26" for hit in hits)
    if phone == "iPhone 17":
        assert all(hit["reference_price"] == 6799 and hit["price_variant"] == "256GB" for hit in hits)


@pytest.fixture
def api_service(tmp_path, monkeypatch):
    path = tmp_path / "sessions.sqlite3"

    def create():
        return SelectionAssistant(ChatStore(path), model_client=NoModel(), source_loader=NoPublicSources())

    active = {"service": create()}
    monkeypatch.setattr(web, "get_chat_service", lambda: active["service"])
    return active, create


def assert_scoped_sources(session):
    reply = session["messages"][-1]
    sources = {source["id"]: source for source in reply["sources"]}
    assert sources and reply["blocks"]
    for block in reply["blocks"]:
        assert block["evidence_ids"]
        assert all(eid in sources and sources[eid]["phone_id"] == block["phone_id"]
                   for eid in block["evidence_ids"])


def test_http_multiturn_cold_restore_and_independent_session(api_service):
    active, create = api_service
    with TestClient(web.app) as client:
        first = client.post("/api/chat", json={"message": "预算4000，主要拍照和续航"})
        assert first.status_code == 200
        sid = first.json()["session_id"]
        url = f"/api/chat/sessions/{sid}"
        changed = client.post("/api/chat", json={"session_id": sid, "message": "预算提高到5000，不考虑K80了"})
        assert changed.status_code == 200
        snapshot = changed.json()
        assert snapshot["profile"]["budget_max"] == 5000
        assert snapshot["profile"]["priorities"] == ["拍照", "续航"]
        assert "redmi-k80" not in snapshot["profile"]["candidate_ids"]
        assert_scoped_sources(snapshot)
        independent = client.post("/api/chat/sessions").json()
    # A fresh service and client reload the same database; no old service state.
    active["service"] = create()
    with TestClient(web.app) as client:
        assert client.get(url).json() == snapshot
        other = client.get(f'/api/chat/sessions/{independent["session_id"]}').json()
        assert other["profile"]["budget_max"] is None and not other["messages"]
        followup = client.post("/api/chat", json={"session_id": sid, "message": "第一款夜景怎么样？"})
        assert followup.status_code == 200
        assert followup.json()["profile"]["budget_max"] == 5000
        assert_scoped_sources(followup.json())
        assert client.delete(url).status_code == 204
        assert client.get(url).status_code == 404
        assert client.get(f'/api/chat/sessions/{independent["session_id"]}').json() == other


@pytest.mark.parametrize("error,status", [
    (SessionConflict("internal-conflict-detail"), 409),
    (ValueError("internal-storage-detail"), 503),
], ids=["revision-conflict", "service-unavailable"])
def test_http_failed_turn_preserves_saved_state_and_can_continue(api_service, monkeypatch, error, status):
    active, _ = api_service
    service = active["service"]
    with TestClient(web.app) as client:
        initial = client.post("/api/chat", json={"message": "预算4000，主要拍照"}).json()
        sid = initial["session_id"]
        original_turn = service.turn

        def fail(request):
            raise error

        monkeypatch.setattr(service, "turn", fail)
        failed = client.post("/api/chat", json={"session_id": sid, "message": "预算提高到5000"})
        assert failed.status_code == status
        assert str(error) not in failed.text
        assert client.get(f"/api/chat/sessions/{sid}").json() == initial
        monkeypatch.setattr(service, "turn", original_turn)
        recovered = client.post("/api/chat", json={"session_id": sid, "message": "预算提高到5000"})
        assert recovered.status_code == 200
        assert recovered.json()["profile"]["budget_max"] == 5000
        assert recovered.json()["profile"]["priorities"] == ["拍照"]
        assert recovered.json()["revision"] == initial["revision"] + 1
