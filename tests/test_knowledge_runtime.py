"""Offline checks for local candidate discovery, persistence and request ceilings."""
import socket
from app.chat import SelectionAssistant
from app.chat_store import ChatStore
from app.chat_schemas import ChatSource
from app.knowledge import KnowledgeBase,KnowledgeIndex,KnowledgeChunk,save_index,load_index
from app.local_embeddings import LocalEmbeddings
from app.phone_sources import PublicPhoneSources
from app.selection_catalog import CatalogPhone
from app.tool_loop import AgentPolicy
from tests.test_chat_agent import Model,Public,tool,say,done


def test_agent_discovers_multiple_local_models_without_any_public_lookup(tmp_path):
    names=['vivo X100','vivo X200','vivo X300'];chunks=[]
    for i,name in enumerate(names):
        for kind,text in [('spec','相机5000万像素主摄，OIS防抖。电池6000mAh，80W快充。'),
                          ('price',f'12GB+256GB：{3500+i*100}元。拍照续航版本价。')]:
            chunks.append(KnowledgeChunk(phone_name=name,title='虚构夹具',source_url=f'https://www.vivo.com.cn/vivo/{i}/',
                text=text,document_id=f'doc-{i}-{kind}',chunk_id=f'chunk-{i}-{kind}',checked_on='2026-10-22',
                market='中国大陆',evidence_kind=kind,reference_price=3500+i*100 if kind=='price' else None,
                price_variant='12GB+256GB' if kind=='price' else None))
    text='预算4000，拍照续航，只用本地资料'
    model=Model([[tool('search_local_phones',query='拍照 续航')],{'kind':'answer','candidate_names':names}],
        {text:{'budget_max':4000,'priorities':['拍照','续航']}})
    public=Public();assistant=SelectionAssistant(ChatStore(tmp_path/'s.db'),model_client=model,catalog={},
        knowledge=KnowledgeBase(KnowledgeIndex(chunks=chunks)),source_loader=public)
    reply=say(assistant,text).messages[-1]
    assert len(reply.cards)==3 and {c.price for c in reply.cards}=={3500,3600,3700}
    assert not public.searches and not public.fetches
    assert reply.agent_trace['metrics']['model_calls']==4
    assert reply.agent_trace['metrics']['knowledge_calls']==1 and reply.agent_trace['metrics']['network_calls']==0


def test_public_search_budget_applies_to_each_call_in_a_batch(tmp_path):
    text='想了解vivo X100的屏幕'
    model=Model([[tool('web_search',phone_name='vivo X100',query=f'vivo X100 屏幕 {i}') for i in range(4)],done()],
        {text:{'selected_phones':['vivo X100']}})
    public=Public();phone=CatalogPhone(id='fixture',name='vivo X100',aliases=[],brand='vivo',variant='256GB',checked_on='2026-10-22',note='虚构夹具')
    assistant=SelectionAssistant(ChatStore(tmp_path/'s.db'),model_client=model,catalog={phone.id:phone},
        source_loader=public,agent_policy=AgentPolicy(max_search_calls=2,max_tool_calls=8))
    reply=say(assistant,text).messages[-1]
    assert len(public.searches)==reply.agent_trace['metrics']['search_calls']==2
    assert sum(a.get('result',{}).get('status')=='lookup_limit' for a in reply.agent_trace['actions'])==2


def test_same_page_new_focus_reuses_html_and_explicit_refresh_fetches_again(tmp_path,monkeypatch):
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443))])
    calls=[]
    def fetch(url):
        calls.append(url)
        return '<title>vivo X100参数</title><main><h1>vivo X100</h1><p>相机5000万像素主摄，光学防抖。</p><p>电池容量6000mAh，80W快充。</p></main>'
    phone=CatalogPhone(id='fixture',name='vivo X100',aliases=[],brand='vivo',variant='256GB',checked_on='2026-10-22',note='虚构夹具')
    loader=PublicPhoneSources(tmp_path/'cache',fetcher=fetch)
    a=loader.fetch_result(phone,'https://8.8.8.8/spec','相机配置')
    b=loader.fetch_result(phone,'https://8.8.8.8/spec','电池配置')
    assert len(calls)==1 and a['lookup_origin']=='network' and b['lookup_origin']=='cache'
    assert any('6000mAh' in s.quote for s in b['sources'])
    loader.fetch_result(phone,'https://8.8.8.8/spec','电池配置',refresh=True)
    assert len(calls)==2


class FakeLocal(LocalEmbeddings):
    def __init__(self):self.calls=0
    def embed_documents(self,texts):self.calls+=1;return [[1.,0.] for _ in texts]
    def embed_query(self,text):return [1.,0.]


def test_verified_official_body_persists_once_but_preview_and_review_are_not_auto_imported(tmp_path):
    chunk=KnowledgeChunk(phone_name='vivo X100',title='虚构旧资料',source_url='https://www.vivo.com.cn/old',
        text='电池5000mAh',document_id='old',chunk_id='old',vector=[1.,0.])
    index=KnowledgeIndex(method='vector',embedding_signature='offline-fixture',chunks=[chunk])
    path=tmp_path/'index.json';save_index(index,path)
    embedding=FakeLocal();base=KnowledgeBase(index,embedding,search_method='hybrid');base.path=path
    phone=CatalogPhone(id='fixture',name='vivo X100',aliases=[],brand='vivo',variant='256GB',checked_on='2026-10-22',note='虚构夹具')
    src=ChatSource(id='new',phone_id=phone.id,title='虚构参数',url='https://www.vivo.com.cn/spec',
        quote='vivo X100后置5000万像素主摄，电池6000mAh。',checked_on='2026-10-22',source_type='page',evidence_kind='spec',market='中国大陆')
    for rejected in [src.model_copy(update={'source_type':'snippet'}),src.model_copy(update={'evidence_kind':'review'}),src.model_copy(update={'url':'https://example.com/spec'})]:
        assert base.remember_sources([rejected],{phone.id:phone})['added_chunks']==0
    assert base.remember_sources([src],{phone.id:phone})['added_chunks']==1
    assert base.remember_sources([src],{phone.id:phone})['added_chunks']==0 and embedding.calls==1
    assert len(load_index(path).chunks)==2
    assert path.with_suffix('.updates.jsonl').is_file()


def test_final_agent_blocks_are_validated_without_another_model_request(tmp_path):
    text='想了解vivo X100的电池容量'
    phone=CatalogPhone(id='fixture',name='vivo X100',aliases=[],brand='vivo',price=3500,variant='256GB',checked_on='2026-10-22',note='虚构夹具')
    chunk=KnowledgeChunk(phone_name=phone.name,title='虚构参数',source_url='https://example.com/config',text='6000mAh电池，80W快充。',document_id='known',chunk_id='known')
    model=Model([[tool('search_phone_knowledge',phone_name=phone.name,query='电池容量')],
        {'kind':'answer','blocks':[{'phone_id':phone.id,'text':'6000mAh电池，80W快充。','evidence_ids':['fixture:known']}]}],
        {text:{'selected_phones':[phone.name]}})
    assistant=SelectionAssistant(ChatStore(tmp_path/'s.db'),catalog={phone.id:phone},model_client=model,
        knowledge=KnowledgeBase(KnowledgeIndex(chunks=[chunk])),source_loader=Public())
    reply=say(assistant,text).messages[-1]
    assert reply.blocks[0].text=='6000mAh电池，80W快充。'
    assert model.answer_calls==0 and reply.agent_trace['metrics']['model_calls']==3
    assert any(a['action']=='answer_reused_from_decision' for a in reply.agent_trace['actions'])


def test_camera_test_does_not_support_battery_superiority_claim(tmp_path):
    from app.chat_schemas import ChatSession,SelectionProfile
    phone=CatalogPhone(id='fixture',name='vivo X100',aliases=[],brand='vivo',price=3500,variant='256GB',checked_on='2026-10-22',note='虚构夹具')
    source=ChatSource(id='evidence',phone_id=phone.id,title='虚构评测',url='https://example.com/review',
        quote='拍照样张曝光稳定、肤色自然。电池6000mAh，80W快充。',source_type='page',evidence_kind='review',checked_on='2026-10-22')
    class Draft:
        available=True
        def complete_json(self,*args,**kwargs):
            return {'blocks':[{'phone_id':phone.id,'text':'续航表现最强，电池6000mAh。','evidence_ids':[source.id]}]}
    assistant=SelectionAssistant(ChatStore(tmp_path/'s.db'),catalog={phone.id:phone},model_client=Draft())
    state={'session':ChatSession(session_id='a'*32,profile=SelectionProfile(budget_max=4000,priorities=['拍照','续航'],candidate_ids=[phone.id])),
        'message':'预算4000拍照续航推荐','recommendation_turn':True,'catalog':assistant.catalog,'selected':[phone.id],'sources':[source],'acknowledged':[]}
    reply=assistant.answer(state)['reply']
    assert state['answer_failure']['stage']=='experience_evidence'
    assert all('续航表现最强' not in b.text for b in reply.blocks)
