import json
import pytest

from scripts import evaluate_sealed_chat as evaluation


def cases_at(tmp_path):
    path=tmp_path/'external.json'
    path.write_text(json.dumps([{'id':'external-a','kind':'conversation','turns':[{'message':'a','search_mode':'local'}]}]),encoding='utf-8')
    return path


def test_seal_records_claimed_origin_without_manufacturing_blindness(tmp_path):
    plan=evaluation.seal(cases_at(tmp_path),tmp_path/'sealed',3,'external')
    assert plan['origin_declared']=='external'
    assert plan['blindness']=='not_independently_verified'
    assert plan['human_review_status']=='pending' and plan['max_provider_requests']==3


def test_seal_accepts_36_requests_and_rejects_out_of_bounds_before_creating_output(tmp_path):
    cases=cases_at(tmp_path)
    plan=evaluation.seal(cases,tmp_path/'sealed',36,'developer')
    assert plan['max_provider_requests']==36
    for limit in (0,37):
        output=tmp_path/f'invalid-{limit}'
        with pytest.raises(ValueError,match='between 1 and 36'):
            evaluation.seal(cases,output,limit,'developer')
        assert not output.exists()


def test_changed_case_or_implementation_is_rejected_before_model(tmp_path,monkeypatch):
    cases=cases_at(tmp_path);output=tmp_path/'sealed'
    evaluation.seal(cases,output,2,'developer')
    monkeypatch.setattr(evaluation,'BoundedModel',lambda *a,**k:pytest.fail('provider initialized'))
    cases.write_text('[]',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):evaluation.run(output)
    assert not (output/'run.started').exists()


def test_interrupted_run_cannot_be_paid_again(tmp_path,monkeypatch):
    output=tmp_path/'sealed';evaluation.seal(cases_at(tmp_path),output,2,'developer')
    class Unavailable:
        available=False
        def __init__(self,*a,**k):pass
    monkeypatch.setattr(evaluation,'BoundedModel',Unavailable)
    with pytest.raises(ValueError,match='No configured model'):evaluation.run(output)
    with pytest.raises(FileExistsError):evaluation.run(output)


def test_sealed_run_rejects_changed_runtime_budget_before_provider(tmp_path,monkeypatch):
    output=tmp_path/'sealed'
    evaluation.seal(cases_at(tmp_path),output,2,'developer')
    monkeypatch.setenv('MODEL_MAX_TOKENS','97')
    monkeypatch.setattr(evaluation,'BoundedModel',lambda *a,**k:pytest.fail('provider initialized'))
    with pytest.raises(ValueError,match='changed'):evaluation.run(output)
    assert not (output/'run.started').exists()


def test_vectorization_preserves_source_and_chunk_ids():
    from app.knowledge import KnowledgeIndex,KnowledgeChunk,vectorize_index
    chunk=KnowledgeChunk(phone_name='fixture',title='fixture',source_url='https://example.com',text='fixture',
                         document_id='doc',chunk_id='chunk')
    class Vectors:
        def embed_documents(self,texts):return [[.5,.7] for _ in texts]
    result=vectorize_index(KnowledgeIndex(chunks=[chunk]),embeddings=Vectors(),embedding_signature='test')
    assert result.chunks[0].chunk_id=='chunk' and result.chunks[0].source_url==chunk.source_url
    assert result.method=='vector' and result.chunks[0].vector==[.5,.7]


def test_seed_snapshot_is_frozen_before_provider_calls(tmp_path,monkeypatch):
    from app.chat_schemas import ChatSession
    seed=tmp_path/'seed.json';seed.write_text(ChatSession(session_id='a'*32).model_dump_json(),encoding='utf-8')
    cases=cases_at(tmp_path);data=json.loads(cases.read_text());data[0]['seed_session']=str(seed)
    cases.write_text(json.dumps(data),encoding='utf-8');output=tmp_path/'sealed'
    evaluation.seal(cases,output,2,'developer')
    (output/'external-a-seed.json').write_text('{}',encoding='utf-8')
    monkeypatch.setattr(evaluation,'BoundedModel',lambda *a,**k:pytest.fail('provider initialized'))
    with pytest.raises(ValueError,match='Seed snapshot changed'):evaluation.run(output)
    assert not (output/'run.started').exists()


def test_sealed_conversation_can_reload_persisted_session_between_turns(tmp_path,monkeypatch):
    from app.chat_store import ChatStore
    from app.tool_loop import AgentPolicy
    cases=tmp_path/'reload.json'
    cases.write_text(json.dumps([{'id':'resume','kind':'conversation','reload_between_turns':True,
        'turns':[{'message':'first','search_mode':'local'},{'message':'next','search_mode':'local'}]}]),encoding='utf-8')
    output=tmp_path/'sealed';evaluation.seal(cases,output,2,'developer')
    instances=[];store=ChatStore(tmp_path/'sessions.db')
    class Model:
        available=True
        def __init__(self,*a,limit,**kw):self.limit,self.requests=limit,[]
    class Service:
        def __init__(self,store,**kw):
            self.store,self.agent_policy=store,AgentPolicy();instances.append(self)
        def turn(self,request):
            session=self.store.load(request.session_id) if request.session_id else self.store.create()
            if request.session_id:assert session.profile.budget_max==5432
            session.profile.budget_max=5432
            return self.store.save(session)
    monkeypatch.setattr(evaluation,'BoundedModel',Model)
    monkeypatch.setattr(evaluation,'SelectionAssistant',Service)
    monkeypatch.setattr(evaluation,'ChatStore',lambda path:store)
    monkeypatch.setattr(evaluation.KnowledgeBase,'from_path',lambda *a,**k:None)
    evaluation.run(output)
    rows=json.loads((output/'results.json').read_text(encoding='utf-8'))
    assert len(instances)==2 and rows[0]['turns'][1]['result']['revision']==2
