from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import web
from app.chat_store import ChatStore
from app.config import Settings
from app.usage_guard import UsageGuard, QuotaExceeded


def test_api_authentication_covers_read_write_delete_without_touching_service(tmp_path,monkeypatch):
    monkeypatch.setenv('CHAT_ACCESS_TOKEN','only-for-test')
    def fail():
        raise AssertionError('unauthenticated request reached service')
    monkeypatch.setattr(web,'get_chat_service',fail)
    client=TestClient(web.app)
    for method,path in [('post','/api/chat'),('post','/api/chat/sessions'),
                        ('get','/api/chat/sessions/'+'a'*32),('delete','/api/chat/sessions/'+'a'*32)]:
        assert getattr(client,method)(path).status_code==401
    store=ChatStore(tmp_path/'sessions.db')
    monkeypatch.setattr(web,'get_chat_service',lambda:SimpleNamespace(store=store))
    assert client.post('/api/chat/sessions',headers={'Authorization':'bearer only-for-test'}).status_code==200


def test_remote_access_is_denied_until_a_key_is_configured(monkeypatch):
    monkeypatch.setattr(web,'get_chat_service',lambda:pytest.fail('remote request reached service'))
    client=TestClient(web.app,client=('203.0.113.9',54321))
    assert client.post('/api/chat/sessions').status_code==403
    monkeypatch.setenv('CHAT_ACCESS_TOKEN','required')
    assert client.post('/api/chat/sessions').status_code==401


def test_http_rate_limit_runs_before_provider(monkeypatch):
    monkeypatch.setenv('CHAT_RATE_LIMIT_PER_MINUTE','1')
    monkeypatch.setattr(web,'get_chat_service',lambda:pytest.fail('invalid body reached service'))
    client=TestClient(web.app)
    assert client.post('/api/chat',json={}).status_code==422
    response=client.post('/api/chat',json={})
    assert response.status_code==429 and response.headers['retry-after']=='60'


def test_usage_is_atomic_persistent_and_rolls_over_by_hong_kong_date(tmp_path):
    now=[0.0]
    path=tmp_path/'usage.db'
    guard=UsageGuard(path,clock=lambda:now[0])
    settings=Settings(chat_daily_model_calls=3,chat_daily_token_limit=1000)
    def reserve(_):
        try:return guard.reserve(100,settings)
        except QuotaExceeded:return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        tickets=[v for v in pool.map(reserve,range(16)) if v]
    assert len(tickets)==3
    restarted=UsageGuard(path,clock=lambda:now[0])
    with pytest.raises(QuotaExceeded):restarted.reserve(1,settings)
    now[0]+=86400
    assert restarted.reserve(1,settings)


def test_token_reservation_settles_once_and_unknown_usage_is_retained(tmp_path):
    guard=UsageGuard(tmp_path/'usage.db')
    settings=Settings(chat_daily_model_calls=0,chat_daily_token_limit=100)
    ticket=guard.reserve(90,settings)
    guard.settle(ticket,None)
    with pytest.raises(QuotaExceeded):guard.reserve(11,settings)
    guard.settle(ticket,10)
    guard.settle(ticket,1)
    assert guard.reserve(90,settings)
    with pytest.raises(QuotaExceeded):guard.reserve(1,settings)


def test_provider_quota_rejects_before_another_sdk_request(monkeypatch):
    from langchain_core.messages import AIMessage
    from app.model_api import ModelClient
    calls=[]
    class SDK:
        def bind(self,**options):return self
        def invoke(self,messages):
            calls.append(messages)
            return AIMessage(content='{"blocks":[]}',usage_metadata={'input_tokens':2,'output_tokens':1,'total_tokens':3})
    monkeypatch.setenv('CHAT_DAILY_MODEL_CALLS','1')
    client=ModelClient(chat_model=SDK())
    assert client.complete_json([{'role':'user','content':'test'}])=={'blocks':[]}
    with pytest.raises(QuotaExceeded):client.complete_json([{'role':'user','content':'test'}])
    assert len(calls)==1


def test_deadline_timeout_shrinks_and_context_is_restored(monkeypatch):
    from app import execution_context as context
    now=[0.0]
    monkeypatch.setattr(context,'monotonic',lambda:now[0])
    with context.turn_deadline(10):
        assert context.request_timeout(40)==10
        now[0]=8
        assert context.request_timeout(40)==2
        with context.turn_deadline(20):assert context.request_timeout(40)==2
        now[0]=11
        with pytest.raises(context.DeadlineExceeded):context.request_timeout(40)
    assert context.request_timeout(40)==40


def test_remote_embedding_batches_share_provider_call_cap(monkeypatch):
    from app.provider_embeddings import QuotaEmbeddings
    calls=[]
    class Remote:
        def embed_documents(self,texts):calls.append(texts);return [[1.,2.] for _ in texts]
        def embed_query(self,text):calls.append(text);return [1.,2.]
    monkeypatch.setenv('CHAT_DAILY_MODEL_CALLS','2')
    embeddings=QuotaEmbeddings(Remote(),batch_size=2)
    assert len(embeddings.embed_documents(['a','b','c']))==3
    with pytest.raises(QuotaExceeded):embeddings.embed_query('next')
    assert len(calls)==2


def test_concurrent_model_usage_isolated_for_trace_and_quota(monkeypatch):
    from threading import Barrier
    from langchain_core.messages import AIMessage
    from app.model_api import ModelClient
    from app.usage_guard import get_usage_guard
    monkeypatch.setenv('CHAT_DAILY_TOKEN_LIMIT','10000')
    barrier=Barrier(2)
    class SDK:
        def bind(self,**kwargs):return self
        def invoke(self,messages):
            amount=int(messages[-1]['content'])
            return AIMessage(content='{"ok":true}',usage_metadata={
                'input_tokens':amount-1,'output_tokens':1,'total_tokens':amount})
    class Synchronized(ModelClient):
        def _record_usage(self,response):
            super()._record_usage(response)
            barrier.wait(timeout=5)
    client=Synchronized(chat_model=SDK())
    def call(amount):
        client.complete_json([{'role':'user','content':str(amount)}])
        return client.last_usage['total_tokens']
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(call,[3,30]))==[3,30]
    guard=get_usage_guard()
    with guard.connect() as db:
        assert guard.value(db,'tokens',guard.day())==33
