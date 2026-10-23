"""Exercise the actual LangChain provider adapter through a mock SDK transport."""

import json

import httpx2
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
import pytest

from app.model_api import ModelClient, model_base_url


def mock_model(handler):
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return ChatDeepSeek(model="deepseek-chat", api_key="synthetic-test-key", base_url="https://api.deepseek.com",
                        http_client=client, max_retries=0, max_tokens=100)


def test_actual_langchain_request_json_mode_and_usage():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
            "model": "deepseek-chat", "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"answer":"测试"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}})
    client = ModelClient(mock_model(handler))
    assert client.available
    assert client.complete_json([{"role": "user", "content": "输出 JSON"}]) == {"answer": "测试"}
    assert client.last_usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    payload = json.loads(requests[0].content)
    assert payload["response_format"] == {"type": "json_object"}
    assert str(requests[0].url) == "https://api.deepseek.com/chat/completions"
    assert payload["max_tokens"] == 100


def test_provider_error_does_not_retry():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx2.Response(500, json={"error": {"message": "synthetic failure", "type": "server_error"}})
    with pytest.raises(Exception):
        ModelClient(mock_model(handler)).complete_json([{"role": "user", "content": "JSON"}])
    assert len(calls) == 1


def test_old_endpoint_config_still_converts_to_sdk_base():
    assert model_base_url("https://api.deepseek.com/chat/completions/") == "https://api.deepseek.com"
    assert model_base_url("https://example.com/v1") == "https://example.com/v1"


def test_actual_embedding_adapter_sends_text_and_returns_vectors():
    requests = []
    def handler(request):
        requests.append(request)
        payload = json.loads(request.content)
        texts = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
        return httpx2.Response(200, json={"object": "list", "model": "synthetic-embedding-model",
            "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]} for i in range(len(texts))],
            "usage": {"prompt_tokens": 2, "total_tokens": 2}})
    embeddings = OpenAIEmbeddings(model="synthetic-embedding-model", api_key="synthetic-test-key",
        base_url="https://example.com/v1", check_embedding_ctx_length=False, max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    assert embeddings.embed_documents(["续航资料", "相机资料"]) == [[1., 0.], [1., 0.]]
    assert embeddings.embed_query("电量维持时间") == [1., 0.]
    assert len(requests) == 2
    assert json.loads(requests[0].content)["input"] == ["续航资料", "相机资料"]
    assert all(str(request.url) == "https://example.com/v1/embeddings" for request in requests)
