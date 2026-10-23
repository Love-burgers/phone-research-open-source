"""Mixed conditions, alternatives, and explicit network choice; no paid calls."""
import json

from langchain_core.messages import AIMessage, ToolMessage

from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest, SelectionProfile, ChatSession, ChatSource
from app.chat_store import ChatStore
from app.chat_intent import understand_intent
from app.knowledge import KnowledgeBase, KnowledgeIndex, KnowledgeChunk
from app.selection_catalog import CatalogPhone, load_catalog
from app.selection_presentation import configuration_facts
from app.tool_loop import AgentPolicy
from tests.test_chat_agent import Model, Public, tool, done, service


def test_incremental_photo_keeps_main_priority_and_explicit_reorder_is_distinct():
    profile=SelectionProfile(budget_max=6000,priorities=['游戏','续航'])
    text='也想兼顾拍照'
    understand_intent(profile,text,load_catalog(),Model([],{text:{'priorities':['拍照']}}))
    assert profile.priorities==['游戏','续航','拍照'] and profile.budget_max==6000
    text='改成拍照优先，游戏也保留'
    understand_intent(profile,text,load_catalog(),Model([],{text:{'priorities':['拍照'],'priority_action':'reorder'}}))
    assert profile.priorities==['拍照','游戏','续航']
    text='不需要续航优先了'
    understand_intent(profile,text,load_catalog(),Model([],{text:{'remove_priorities':['续航']}}))
    assert profile.priorities==['拍照','游戏']


def test_parameter_identity_and_round_robin_do_not_drop_mixed_dimensions():
    p=CatalogPhone(id='fixture',name='vivo X100',aliases=[],brand='vivo',variant='256GB',checked_on='2026-10-22',note='虚构测试')
    s=ChatSource(id='s',phone_id=p.id,title='虚构参数',url='https://www.vivo.com.cn/spec',checked_on='2026-10-22',
        quote='1200 万像素摄像头、摄像头 1200 万像素。5000万像素主摄，5000万像素长焦。天玑9300，6000mAh，80W快充。',evidence_kind='spec')
    camera=configuration_facts(p,['拍照'],[s],limit=8)
    assert sum('1200' in value for value,_ in camera)==1
    assert any('主摄' in v for v,_ in camera) and any('长焦' in v for v,_ in camera)
    mixed=configuration_facts(p,['拍照','游戏','续航'],[s])
    assert len(mixed)==3 and any('天玑9300' in v for v,_ in mixed) and any('6000mAh' in v for v,_ in mixed)


def test_official_playback_rating_is_labelled_and_camera_role_beats_generic_pixel_count():
    p=CatalogPhone(id='fixture',name='iPhone 16',aliases=[],brand='Apple',variant='256GB',checked_on='2026-10-22',note='虚构夹具')
    s=ChatSource(id='s',phone_id=p.id,title='虚构参数',url='https://support.apple.com/zh-cn/121029',checked_on='2026-10-22',
        quote='1200 万像素摄像头；4800 万像素融合式主摄，1200万像素超广角。A18 芯片。视频播放: 最长可达 22 小时。',evidence_kind='spec')
    facts=configuration_facts(p,['游戏','续航','拍照'],[s])
    assert facts==[('A18','s'),('官方标称视频播放: 最长可达 22 小时','s'),('4800 万像素融合式主摄','s')]
    assert not configuration_facts(p,['续航'],[s.model_copy(update={'url':'https://example.com/review','evidence_kind':'review'})])
    from app.phone_sources import official_phone_url,source_market
    assert official_phone_url('Apple',s.url) and source_market(s.url,'技术规格')=='中国大陆'
    assert not official_phone_url('Apple','https://support.apple.com.evil.example/zh-cn/spec')


def test_three_rounds_find_other_candidates_under_cap_and_keep_all_conditions(tmp_path):
    texts=['预算6000，主要玩游戏，注重续航','换一批','也想兼顾拍照']
    names=[f'vivo X{n}00' for n in range(1,7)]
    chunks=[]
    for n,name in enumerate(names):
        for tag,body in [('camera','5000万像素主摄、1200万像素超广角。'),('game','游戏相关配置：天玑9300。'),('battery','电池6000mAh，80W快充。')]:
            chunks.append(KnowledgeChunk(phone_name=name,title='虚构参数',source_url=f'https://www.vivo.com.cn/{n}/{tag}',text=body,
                document_id=f'{n}-{tag}',chunk_id=f'{n}-{tag}',evidence_kind='spec',checked_on='2026-10-22'))
        chunks.append(KnowledgeChunk(phone_name=name,title='虚构价格',source_url=f'https://www.vivo.com.cn/{n}/price',
            text=f'12GB+256GB {3500+n*100}元',document_id=f'{n}-price',chunk_id=f'{n}-price',evidence_kind='price',
            reference_price=3500+n*100,price_variant='12GB+256GB',checked_on='2026-10-22',market='中国大陆'))
    class Archive(KnowledgeBase):
        queries=[]
        def discover(self,query):
            self.queries.append(query)
            # Cross-model results contain only camera passages. The local tool
            # must enrich each model using the other requested dimensions.
            return [{'text':c.text,**c.model_dump(exclude={'text','vector'})} for c in self.index.chunks if c.chunk_id.endswith('camera')]
    class Planner(Model):
        next_lookup=True
        def invoke_tools(self,messages,schemas,**kwargs):
            self.histories.append(list(messages));self.contexts.append(json.loads(messages[1].content))
            if self.next_lookup and messages[-1].type=='human' and '本轮只覆盖了明显低于预算价位' in messages[-1].content:
                return self.last_decision
            if self.next_lookup:
                self.next_lookup=False
                return AIMessage(content='',tool_calls=[tool('search_local_phones',query='相机配置',limit=5)])
            self.next_lookup=True
            payload=next((json.loads(m.content) for m in reversed(messages) if isinstance(m,ToolMessage)),None)
            chosen=payload['local_candidates'] if payload else self.contexts[-1]['最近工具结果'][-1]['候选核对']
            self.last_decision=AIMessage(content=json.dumps({'kind':'answer','candidate_names':[c['name'] for c in chosen],
                'blocks':[{'phone_id':c['configuration_facts'][0][1].split(':')[0],
                           'text':'、'.join(v for v,_ in c['configuration_facts'])+'，作为配置初筛备选。',
                           'evidence_ids':list(dict.fromkeys(sid for _,sid in c['configuration_facts']))} for c in chosen]}),tool_calls=[])
            return self.last_decision
    model=Planner([],{texts[0]:{'budget_max':6000,'priorities':['游戏','续航']},
                      texts[1]:{'candidate_action':'alternatives','candidate_action_quote':texts[1]},texts[2]:{'priorities':['拍照']}})
    base=Archive(KnowledgeIndex(chunks=chunks));public=Public()
    app=SelectionAssistant(ChatStore(tmp_path/'s.db'),catalog={},knowledge=base,model_client=model,source_loader=public,
        agent_policy=AgentPolicy(max_total_tokens=64000,max_context_characters=48000))
    session=None;results=[]
    for text in texts:
        session=app.turn(ChatRequest(message=text,session_id=session.session_id if session else None,search_mode='auto'))
        results.append(session)
    assert all(len(s.messages[-1].cards)==3 for s in results)
    assert not set(results[0].profile.candidate_ids)&set(results[1].profile.candidate_ids)
    assert results[-1].profile.priorities==['游戏','续航','拍照']
    assert all('天玑9300' in b.text and '6000mAh' in b.text and '主摄' in b.text for b in results[-1].messages[-1].blocks)
    assert all('游戏' in q and '续航' in q for q in base.queries) and '拍照' in base.queries[-1]
    assert not public.searches and not public.fetches
    assert app.store.load(session.session_id).profile.priorities==['游戏','续航','拍照']


def test_local_choice_blocks_public_tool_even_if_model_emits_it(tmp_path):
    text='一加13的电池容量是多少'
    model=Model([[tool('web_search',phone_name='一加 13',query='一加13电池')],done()],{text:{'selected_phones':['一加13']}})
    public=Public();app=service(tmp_path,model,source_loader=public)
    result=app.turn(ChatRequest(message=text,search_mode='local'))
    assert not public.searches
    assert any(a.get('result',{}).get('status')=='local_only' for a in result.messages[-1].agent_trace['actions'])


def test_web_choice_requests_model_selected_public_lookup_and_reuses_local_facts(tmp_path):
    text='一加13的电池容量是多少'
    model=Model([[tool('search_phone_knowledge',phone_name='一加 13',query='电池容量')],done(),
                 [tool('web_search',phone_name='一加 13',query='一加13 官网电池')],done()],{text:{'selected_phones':['一加13']}})
    public=Public();app=service(tmp_path,model,source_loader=public)
    result=app.turn(ChatRequest(message=text,search_mode='web'))
    assert len(public.searches)==1 and result.messages[-1].sources
    assert result.messages[-1].agent_trace['metrics']['model_calls']<=6


def test_local_choice_also_applies_without_agent(tmp_path):
    class NoModel:available=False
    public=Public();public.collections=[]
    def collect(*args,**kwargs):public.collections.append(args);return []
    public.collect=collect
    app=service(tmp_path,NoModel(),source_loader=public,agent_enabled=False)
    app.turn(ChatRequest(message='一加13想看游戏性能',search_mode='local'))
    assert not public.collections


def test_local_recommendation_still_asks_model_to_select_after_tool_limit(tmp_path):
    text='预算6000，主要游戏，也看续航，换一批'
    class Selector(Model):
        def complete_json(self,messages,**kwargs):
            if messages[0]['content'].startswith('工具已停止'):
                data=json.loads(messages[-1]['content'])
                assert data['条件']['priorities']==['游戏','续航']
                pid,phone=next(iter(data['机型'].items()))
                source=next(s for s in data['来源'] if s['phone_id']==pid)
                return {'kind':'answer','candidate_names':[phone['name']],
                        'blocks':[{'phone_id':pid,'text':'骁龙8至尊版、6000mAh电池，作为配置初筛备选。','evidence_ids':[source['id']]}]}
            return super().complete_json(messages,**kwargs)
    model=Selector([[tool('search_local_phones',query='游戏 电池',limit=5)]],
        {text:{'budget_max':6000,'priorities':['游戏','续航'],'candidate_action':'alternatives'}})
    chunk=KnowledgeChunk(phone_name='一加 13',title='虚构参数',source_url='https://www.oneplus.com/cn/13/specs',
        text='骁龙8至尊版，6000mAh电池。',document_id='s',chunk_id='s',evidence_kind='spec',checked_on='2026-10-22')
    price=KnowledgeChunk(phone_name='一加 13',title='虚构版本价',source_url='https://www.oneplus.com/cn/13/specs',
        text='12GB+512GB 5000元。',document_id='p',chunk_id='p',evidence_kind='price',reference_price=5000,
        price_variant='12GB+512GB',market='中国大陆',checked_on='2026-10-22')
    app=SelectionAssistant(ChatStore(tmp_path/'s.db'),catalog={},knowledge=KnowledgeBase(KnowledgeIndex(chunks=[chunk,price])),
        model_client=model,source_loader=Public(),agent_policy=AgentPolicy(max_tool_calls=1))
    result=app.turn(ChatRequest(message=text,search_mode='local'))
    reply=result.messages[-1]
    assert reply.cards and '6000mAh' in reply.blocks[0].text
    assert reply.agent_trace['stop_reason']=='tool_call_limit'
    assert any(a.get('action')=='candidate_selection' and a['status']=='ok' for a in reply.agent_trace['actions'])
    assert reply.agent_trace['metrics']['model_calls']==3 and model.answer_calls==0
