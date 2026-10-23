"""Resource, tool-budget and cache regressions; all requests stay offline."""
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import web
from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest
from app.chat_store import ChatStore, SessionConflict
from app.phone_sources import PublicPhoneSources
from app.selection_catalog import load_catalog
from app.tool_loop import AgentPolicy
from tests.test_chat_agent import Model, Public, tool, done


def test_sqlite_connections_close_after_success_and_conflict(tmp_path, monkeypatch):
    original = sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, 'connect', connect)
    store = ChatStore(tmp_path/'sessions.db')
    session = store.create()
    store.save(store.load(session.session_id))
    with pytest.raises(SessionConflict):
        store.save(session)
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connection.execute('SELECT 1')


def test_concurrent_service_start_builds_one_instance(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep
    from app import chat
    calls=[]
    def create(store):
        calls.append(store)
        sleep(.03)
        return object()
    monkeypatch.setenv('CHAT_STORAGE_PATH',str(tmp_path/'startup.db'))
    monkeypatch.setattr(chat,'SelectionAssistant',create)
    chat._service.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            instances=list(pool.map(lambda _:chat.get_chat_service(),range(8)))
        assert len(calls)==1 and all(instance is instances[0] for instance in instances)
    finally:
        chat._service.cache_clear()


@pytest.mark.parametrize('method,path', [
    ('post','/api/chat/sessions'), ('get','/api/chat/sessions/'+'a'*32),
    ('delete','/api/chat/sessions/'+'a'*32), ('get','/api/health'),
])
def test_storage_failure_is_a_consistent_api_response(monkeypatch, method, path):
    def fail():
        raise sqlite3.OperationalError('internal storage detail')
    monkeypatch.setattr(web,'get_chat_service',fail)
    response = getattr(TestClient(web.app),method)(path)
    assert response.status_code == 503
    assert 'internal storage detail' not in response.text


def test_directory_discovery_uses_fetch_budget_after_search_is_exhausted(tmp_path):
    directory='https://8.8.8.8/directory'
    next_directory='https://8.8.8.8/next'

    class Sources(Public):
        visits=[]
        def discover(self, query, *, url=None, **kwargs):
            self.visits.append(url)
            return [{'title':'一加 13','url':next_directory if url else directory,'snippet':'一加 13目录线索'}]

    source=Sources()
    model=Model([[tool('discover_phones',query='手机')],
        [tool('discover_phones',query='手机',url=directory)],
        [tool('discover_phones',query='手机',url=next_directory)],done()],
        {'预算6000，续航':{'budget_max':6000,'priorities':['续航']}})
    assistant=SelectionAssistant(ChatStore(tmp_path/'sessions.db'),model_client=model,source_loader=source,
        agent_policy=AgentPolicy(max_search_calls=1,max_fetch_calls=1,max_total_tokens=64000,max_context_characters=48000))
    result=assistant.turn(ChatRequest(message='预算6000，续航'))
    trace=result.messages[-1].agent_trace
    tools=[a['result'] for a in trace['actions'] if a.get('name')=='discover_phones']
    assert source.visits==[None,directory]
    assert tools[-1]['status']=='lookup_limit'
    assert trace['metrics']['search_calls']==1 and trace['metrics']['fetch_calls']==1


def test_search_cache_preserves_search_failure_feedback(tmp_path):
    def fail(*args):
        raise TimeoutError()
    loader=PublicPhoneSources(tmp_path,searcher=fail)
    # The curated directory can still return pointers after the search fails.
    from app.phone_names import parse_phone_names,register_phone
    phone=register_phone(parse_phone_names('iQOO 15 Ultra')[0],load_catalog())
    from unittest.mock import patch
    with patch('app.phone_sources.directory_links',return_value=[{
        'title':phone.name+'规格','url':'https://8.8.8.8/spec','snippet':phone.name+'规格'}]):
        first=loader.search(phone,phone.name+' 规格')
        second=loader.search(phone,phone.name+' 规格')
    assert first['search_feedback']==second['search_feedback']
    assert second['search_feedback']['error_type']=='TimeoutError'
    assert second['lookup_origin']=='cache'


def test_bad_cache_is_ignored_and_failed_write_cleans_up(tmp_path,monkeypatch):
    loader=PublicPhoneSources(tmp_path)
    bad=tmp_path/'bad.json'
    bad.write_text('{invalid',encoding='utf-8')
    assert loader._read_cache(bad) is None
    original=Path.replace
    def fail(path,target):
        if path.suffix=='.tmp':raise PermissionError()
        return original(path,target)
    monkeypatch.setattr(Path,'replace',fail)
    loader._write_cache(tmp_path/'new.json',{'sources':[]})
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('body,valid', [
    ('12GB+256GB 4200元',True), ('12GB+256GB 4,200.00元',True),
    ('12GB+256GB 14200元',False), ('12GB+256GB 42000元',False),
    ('12GB+256GB 4200.50元',False), ('16GB+512GB 4200元',False),
])
def test_state_and_display_require_a_complete_price_number(body,valid):
    from app.chat_schemas import ChatSource,SelectionProfile
    from app.selection_catalog import source_price_options
    from app.selection_presentation import quoted_price
    phone=load_catalog()['oneplus-13']
    source=ChatSource(id='offer',phone_id=phone.id,title='Fixture',url='https://www.oneplus.com/cn/price',
        quote=body,checked_on='2026-10-22',evidence_kind='price',
        price_options=[{'variant':'12GB+256GB','price':4200}])
    assert bool(source_price_options(source))==valid
    assert bool(quoted_price(phone,SelectionProfile(budget_max=5000),[source]))==valid
