"""Personal provider keys are request-scoped, never a global configuration edit."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from app import web
from app.chat_schemas import ChatSession
from app.config import get_model_api_key
from app.model_credentials import current_model_api_key,use_model_api_key

OWNER='offline-owner-credential'
ALPHA='offline-personal-alpha'
BETA='offline-personal-beta'


def client(**kwargs):
    return TestClient(web.app,base_url=kwargs.pop('base_url','http://127.0.0.1'),client=kwargs.pop('client',('127.0.0.1',12345)),**kwargs)


def fake_service(monkeypatch, action):
    monkeypatch.setattr(web,'get_chat_service',lambda:SimpleNamespace(turn=action))


def reply():return ChatSession(session_id='a'*32)


def test_nested_credentials_restore_even_after_exception(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY',OWNER)
    assert current_model_api_key() is None
    with use_model_api_key(SecretStr(ALPHA)):
        assert get_model_api_key()==ALPHA
        with pytest.raises(RuntimeError),use_model_api_key(SecretStr(BETA)):
            assert get_model_api_key()==BETA
            raise RuntimeError('offline failure')
        assert get_model_api_key()==ALPHA
    assert get_model_api_key()==OWNER and current_model_api_key() is None


def test_chat_request_uses_personal_key_then_restores_environment(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY',OWNER)
    seen=[]
    def turn(request):
        seen.append(get_model_api_key())
        assert ALPHA not in request.model_dump_json()
        return reply()
    fake_service(monkeypatch,turn)
    app=client()
    response=app.post('/api/chat',json={'message':'预算4000'},headers={'X-Model-API-Key':ALPHA})
    assert response.status_code==200 and ALPHA not in response.text and OWNER not in response.text
    assert app.post('/api/chat',json={'message':'预算4000'}).status_code==200
    assert seen==[ALPHA,OWNER] and get_model_api_key()==OWNER


def test_parallel_chat_requests_do_not_share_keys(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY',OWNER)
    barrier=Barrier(2);seen=[]
    def turn(request):
        before=get_model_api_key();barrier.wait(timeout=10)
        seen.append((before,get_model_api_key()))
        return reply()
    fake_service(monkeypatch,turn)
    def send(key):
        with client() as app:return app.post('/api/chat',json={'message':'预算4000'},headers={'X-Model-API-Key':key}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(send,[ALPHA,BETA]))==[200,200]
    assert sorted(seen)==sorted([(ALPHA,ALPHA),(BETA,BETA)])
    assert get_model_api_key()==OWNER


def test_bad_personal_key_failure_does_not_retry_owner_key_or_echo_credentials(monkeypatch,caplog):
    monkeypatch.setenv('MODEL_API_KEY',OWNER)
    seen=[]
    def turn(request):
        seen.append(get_model_api_key())
        raise ValueError('offline provider rejected '+get_model_api_key())
    fake_service(monkeypatch,turn)
    response=client().post('/api/chat',json={'message':'预算4000'},headers={'X-Model-API-Key':ALPHA})
    assert response.status_code==503 and seen==[ALPHA]
    assert ALPHA not in response.text+caplog.text and OWNER not in response.text+caplog.text
    assert current_model_api_key() is None


@pytest.mark.parametrize('key',['short','contains spaces in credential','x'*2049,''])
def test_invalid_personal_key_is_rejected_without_reaching_service(monkeypatch,key):
    fake_service(monkeypatch,lambda r:pytest.fail('invalid key reached provider'))
    response=client().post('/api/chat',json={'message':'预算4000'},headers={'X-Model-API-Key':key})
    assert response.status_code==400
    assert response.json()=={'detail':'API Key格式无效，请检查后重试。'}


def test_personal_key_does_not_bypass_site_access_key(monkeypatch):
    monkeypatch.setenv('CHAT_ACCESS_TOKEN','offline-site-access')
    fake_service(monkeypatch,lambda r:pytest.fail('authentication bypass'))
    assert client().post('/api/chat',json={'message':'预算4000'},headers={'X-Model-API-Key':ALPHA}).status_code==401


def test_remote_key_entry_requires_https_and_existing_access_protection(monkeypatch):
    monkeypatch.setenv('CHAT_ACCESS_TOKEN','offline-site-access')
    seen=[]
    fake_service(monkeypatch,lambda r:seen.append(get_model_api_key()) or reply())
    headers={'Authorization':'Bearer offline-site-access','X-Model-API-Key':ALPHA}
    assert client(base_url='http://phone.example',client=('203.0.113.1',1000)).post('/api/chat',json={'message':'预算4000'},headers=headers).status_code==400
    # Also refuse a plain-HTTP public hostname behind a local reverse proxy.
    assert client(base_url='http://phone.example').post('/api/chat',json={'message':'预算4000'},headers=headers).status_code==400
    assert client(base_url='https://phone.example',client=('203.0.113.1',1000)).post('/api/chat',json={'message':'预算4000'},headers=headers).status_code==200
    assert seen==[ALPHA]


def test_real_model_adapter_receives_only_request_key(monkeypatch):
    from langchain_core.messages import AIMessage
    from app import model_api
    monkeypatch.setenv('MODEL_API_KEY',OWNER)
    monkeypatch.setenv('MODEL_PROVIDER','deepseek')
    keys=[]
    class SDK:
        def __init__(self,**kwargs):keys.append(kwargs['api_key'])
        def bind(self,**kwargs):return self
        def invoke(self,messages):return AIMessage(content='{}',usage_metadata={'input_tokens':1,'output_tokens':1,'total_tokens':2})
    monkeypatch.setattr(model_api,'ChatDeepSeek',SDK)
    with use_model_api_key(SecretStr(ALPHA)):
        assert model_api.ModelClient().complete_json([{'role':'user','content':'offline adapter probe'}])=={}
    assert keys==[ALPHA] and get_model_api_key()==OWNER


def test_health_describes_destination_without_returning_key(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY',OWNER)
    response=client().get('/api/health')
    assert response.status_code==200 and OWNER not in response.text
    assert response.json()['model_name'] and response.json()['model_api_host']


def test_personal_key_is_never_sent_to_plain_http_remote_provider(monkeypatch):
    monkeypatch.setenv('MODEL_API_URL','http://provider.example/chat/completions')
    fake_service(monkeypatch,lambda r:pytest.fail('key sent to insecure provider'))
    response=client().post('/api/chat',json={'message':'预算4000'},headers={'X-Model-API-Key':ALPHA})
    assert response.status_code==400 and ALPHA not in response.text


def test_personal_key_is_absent_from_real_saved_conversation(tmp_path,monkeypatch):
    from app.chat import SelectionAssistant
    from app.chat_store import ChatStore
    class Offline:available=False
    store=ChatStore(tmp_path/'chat.db')
    assistant=SelectionAssistant(store,model_client=Offline(),agent_enabled=False)
    monkeypatch.setattr(web,'get_chat_service',lambda:assistant)
    response=client().post('/api/chat',json={'message':'预算4000，主要拍照','search_mode':'local'},headers={'X-Model-API-Key':ALPHA})
    assert response.status_code==200 and ALPHA not in response.text
    saved=store.load(response.json()['session_id'])
    assert ALPHA not in saved.model_dump_json()
    assert ALPHA.encode() not in store.path.read_bytes()


def test_personal_key_provider_matches_destination_shown_on_homepage(monkeypatch):
    from app import model_api
    monkeypatch.setenv('MODEL_API_URL','https://offline-provider.example/v1/chat/completions')
    monkeypatch.setenv('MODEL_NAME','offline-model')
    monkeypatch.setenv('MODEL_PROVIDER','openai-compatible')
    captured=[]
    monkeypatch.setattr(model_api,'ChatOpenAI',lambda **kwargs:captured.append(kwargs) or object())
    with use_model_api_key(SecretStr(ALPHA)):model_api.ModelClient()._model()
    metadata=client().get('/api/health').json()
    assert captured[0]['base_url']=='https://'+metadata['model_api_host']+'/v1'
    assert captured[0]['model']==metadata['model_name']=='offline-model'
    assert captured[0]['api_key']==ALPHA
