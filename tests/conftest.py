import socket

import ddgs
import httpx
import pytest


@pytest.fixture(autouse=True)
def disable_real_model_api(monkeypatch, tmp_path):
    monkeypatch.setenv('CHAT_GUARD_PATH',str(tmp_path/'usage.sqlite3'))
    monkeypatch.setenv('CHAT_RATE_LIMIT_PER_MINUTE','0')
    monkeypatch.setenv('CHAT_DAILY_TURN_LIMIT','0')
    monkeypatch.setenv('CHAT_DAILY_MODEL_CALLS','0')
    monkeypatch.setenv('CHAT_DAILY_TOKEN_LIMIT','0')
    monkeypatch.delenv('CHAT_ACCESS_TOKEN',raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("KNOWLEDGE_INDEX_PATH", raising=False)
    monkeypatch.delenv("CHAT_KNOWLEDGE_INDEX_PATH", raising=False)
    monkeypatch.delenv("CHAT_RETRIEVAL_METHOD", raising=False)
    monkeypatch.delenv("CHAT_AGENT_ENABLED", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_LOCAL_CACHE_DIR", raising=False)


@pytest.fixture(autouse=True)
def prevent_external_network(monkeypatch):
    """A missed fake must never send real search/HTTP/DNS requests."""
    def blocked(*args, **kwargs):
        raise AssertionError("External network is disabled in offline tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(ddgs.DDGS, "text", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    # Current OpenAI SDK also uses httpx2. Keep accidental SDK calls offline.
    import httpx2
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", blocked)
