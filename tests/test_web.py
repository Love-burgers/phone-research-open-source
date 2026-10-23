from fastapi.testclient import TestClient
from app import web


def test_chat_health_and_home():
    client = TestClient(web.app)
    health=client.get("/api/health").json()
    assert {key:health[key] for key in ('status','model_configured','chat_agent_enabled','knowledge_configured')} == {"status": "ok", "model_configured": False, "chat_agent_enabled": True, "knowledge_configured": False}
    assert health['retrieval_method']=='bm25' and health['knowledge_models']>0 and health['knowledge_chunks']>0
    page = client.get("/")
    assert page.status_code == 200 and "选机助手" in page.text
    assert 'href="/research"' not in page.text


def test_only_chat_api_is_exposed():
    client = TestClient(web.app)
    assert set(client.get("/openapi.json").json()["paths"]) == {"/api/health", "/api/chat", "/api/chat/sessions", "/api/chat/sessions/{session_id}"}
    assert client.get("/research").status_code == 404
    assert client.post("/api/phones/research", json={"need": "续航", "phone_names": ["A", "B"]}).status_code == 404


def test_chat_health_reads_its_own_knowledge_setting(tmp_path, monkeypatch):
    index = tmp_path / "chat-index.json"
    index.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CHAT_KNOWLEDGE_INDEX_PATH", str(index))
    assert TestClient(web.app).get("/api/health").json()["knowledge_configured"] is True
