"""Refinement is not re-recommendation; budget coverage is not a hard floor."""
import json

import pytest
from langchain_core.messages import AIMessage,ToolMessage

from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest,ChatMessage,ChatSession,ChatSource,SelectionProfile
from app.chat_store import ChatStore
from app.knowledge import KnowledgeBase,KnowledgeIndex,KnowledgeChunk
from app.selection_catalog import CatalogPhone,budget_position,shortlist
from app.selection_presentation import presentation_copy
from tests.test_chat_agent import Model,Public,tool,done


@pytest.mark.parametrize('text,expected',[
 ('骁龙8至尊版、6000mAh电池。游戏与续航实测待补。','骁龙8至尊版、6000mAh电池。'),
 ('价格5999元在预算内，但游戏与续航实测待补。','价格5999元在预算内。'),
 ('电池6000mAh但暂无续航实测。','电池6000mAh'),
 ('第三代骁龙8，游戏表现待补实测。','第三代骁龙8。'),
 ('骁龙8至尊版。游戏表现仍待核。','骁龙8至尊版。'),
 ('原神最高画质连续30分钟，平均59帧，温度40℃。','原神最高画质连续30分钟，平均59帧，温度40℃。'),
])
def test_gap_notices_leave_facts_and_measured_conditions_intact(text,expected):
    assert presentation_copy(text)[0]==expected
    assert presentation_copy(text,False)[0]==text


def phone(pid,price):
    return CatalogPhone(id=pid,name='vivo X'+{'a':'100','b':'200','c':'300','d':'400'}[pid],aliases=[],brand='vivo',
        price=price,variant='12GB+256GB',checked_on='2026-10-22',topics=['游戏','续航'],note='虚构测试')


def test_budget_tier_is_preferred_but_savings_remain_eligible_and_explicit():
    catalog={pid:phone(pid,p) for pid,p in zip('abcd',[2200,3200,5299,5799])}
    profile=SelectionProfile(budget_max=6000,priorities=['游戏','续航'])
    chosen=shortlist(catalog,profile)
    assert chosen[:2]==['d','c'] and len(chosen)==3
    assert budget_position(2200,profile)==2
    profile.price_preference='save_money'
    assert shortlist(catalog,profile)[:2]==['a','b']


@pytest.mark.parametrize('game',['原神','绝区零'])
@pytest.mark.parametrize('price',[2200,5299])
def test_game_refinement_keeps_current_candidate_and_answers_that_game_without_price(tmp_path,game,price):
    p=phone('c',price)
    source=ChatSource(id='s',phone_id=p.id,title='虚构参数',url='https://www.vivo.com.cn/spec',checked_on='2026-10-22',
        quote='骁龙8至尊版，6000mAh电池，80W快充。',evidence_kind='spec')
    text='主要玩'+game
    model=Model([done()],{text:{'games':[game],'priorities':['游戏']}})
    # Deliberately generic draft should be rejected as off-topic and replaced
    # using the known configuration, without another model call.
    public=Public();store=ChatStore(tmp_path/'s.db')
    app=SelectionAssistant(store,catalog={p.id:p},model_client=model,source_loader=public,
        knowledge=KnowledgeBase(KnowledgeIndex(chunks=[KnowledgeChunk(phone_name=p.name,title='夹具',source_url=source.url,
            text=source.quote,document_id='s',chunk_id='s',evidence_kind='spec')])))
    s=store.create();s.profile=SelectionProfile(budget_max=6000,priorities=['游戏','续航'],candidate_ids=[p.id])
    s.messages=[ChatMessage(role='user',content='预算6000，游戏和续航'),ChatMessage(role='assistant',content='你主要玩什么游戏？',
        cards=[p.card(s.profile,[source])],sources=[source])];s=store.save(s)
    result=app.turn(ChatRequest(message=text,session_id=s.session_id,search_mode='local'))
    reply=result.messages[-1]
    assert result.profile.candidate_ids==[p.id] and result.profile.budget_max==6000
    assert result.profile.priorities==['游戏','续航'] and result.profile.games==[game]
    assert model.contexts[0]['本轮是细节追问'] is True
    assert model.contexts[0]['候选价位覆盖不足'] is False
    body=reply.content+''.join(b.text for b in reply.blocks)
    assert game in body and '骁龙8至尊版' in body
    assert '推荐这' not in body and '5299' not in body and '实测待补' not in body
    assert not public.searches and not public.fetches


def test_price_and_primary_use_coverage_must_be_on_the_same_candidate():
    from app.selection_requirements import discovery_needed
    cheap=phone('a',2200);expensive=phone('d',5799);expensive.topics=['拍照']
    profile=SelectionProfile(budget_max=6000,priorities=['游戏','续航'],candidate_ids=['a','d'])
    assert discovery_needed(profile,{'a':cheap,'d':expensive})
    expensive.topics=['游戏','续航']
    assert not discovery_needed(profile,{'a':cheap,'d':expensive})


def test_separate_parameter_labels_keep_short_chip_model_and_battery_values():
    from app.phone_sources import extract_phone_excerpt
    html='<title>vivo X100 参数</title><main><h1>vivo X100</h1><ul>'
    for label,value in [('CPU型号','天玑 9400'),('电池容量','典型容量6000mAh'),('充电规格','90W有线闪充')]:
        html+=f'<li><p class="attr-item-name">{label}</p><div><div><div><p>{value}</p></div></div></div></li>'
    html+='</ul></main>'
    quote=extract_phone_excerpt('vivo X100',html,'芯片处理器配置 电池容量 充电功率')
    assert 'CPU型号：天玑 9400' in quote and '6000mAh' in quote and '90W有线闪充' in quote


def test_game_tool_keeps_known_chip_when_measurement_query_has_no_hits(tmp_path):
    p=phone('c',5299)
    chunk=KnowledgeChunk(phone_name=p.name,title='虚构官方参数',source_url='https://www.vivo.com.cn/spec',text='天玑9400，6000mAh。',
        document_id='chip',chunk_id='chip',evidence_kind='spec')
    class Archive(KnowledgeBase):
        def as_tool(self):
            class Empty:
                def invoke(self,args):return []
            return Empty()
    text='主要玩原神'
    model=Model([[tool('search_phone_knowledge',phone_name=p.name,query='原神 帧率')],done()],{text:{'games':['原神']}})
    app=SelectionAssistant(ChatStore(tmp_path/'s.db'),catalog={p.id:p},model_client=model,source_loader=Public(),
        knowledge=Archive(KnowledgeIndex(chunks=[chunk])))
    s=app.store.create();s.profile=SelectionProfile(budget_max=6000,priorities=['游戏','续航'],candidate_ids=[p.id])
    s.messages=[ChatMessage(role='user',content='预算6000，主要游戏'),ChatMessage(role='assistant',content='主要玩什么？',cards=[p.card(s.profile)])]
    s=app.store.save(s)
    reply=app.turn(ChatRequest(message=text,session_id=s.session_id,search_mode='local')).messages[-1]
    assert '原神' in reply.content and '天玑9400' in ''.join(b.text for b in reply.blocks)
    assert all('暂作待核对' not in b.text and '实测待补' not in b.text for b in reply.blocks)


def test_local_tool_does_not_fill_up_with_cheap_models_before_budget_tier(tmp_path):
    names=[phone(pid,p) for pid,p in zip('abcd',[2200,3200,5299,5799])]
    chunks=[]
    for p in names:
        chunks.extend([KnowledgeChunk(phone_name=p.name,title='虚构参数',source_url='https://www.vivo.com.cn/'+p.id,
            text='游戏芯片天玑9300，6000mAh电池。',document_id=p.id,chunk_id=p.id,evidence_kind='spec'),
            KnowledgeChunk(phone_name=p.name,title='虚构价格',source_url='https://www.vivo.com.cn/price/'+p.id,
            text=f'12GB+256GB {p.price}元',document_id=p.id+'price',chunk_id=p.id+'price',evidence_kind='price',
            reference_price=p.price,price_variant='12GB+256GB',market='中国大陆')])
    class Archive(KnowledgeBase):
        def discover(self,query):return [{'text':c.text,**c.model_dump(exclude={'text','vector'})} for c in self.index.chunks if c.evidence_kind=='spec']
    text='预算6000，游戏和续航'
    model=Model([[tool('search_local_phones',query='游戏 续航',limit=5)],{'kind':'answer','candidate_names':[names[3].name,names[2].name,names[0].name]}],
        {text:{'budget_max':6000,'priorities':['游戏','续航']}})
    app=SelectionAssistant(ChatStore(tmp_path/'s.db'),catalog={},model_client=model,source_loader=Public(),knowledge=Archive(KnowledgeIndex(chunks=chunks)))
    result=app.turn(ChatRequest(message=text,search_mode='local'))
    returned=next(a['result']['local_candidates'] for a in result.messages[-1].agent_trace['actions'] if a.get('name')=='search_local_phones')
    assert [c['price'] for c in returned][:2]==[5299,5799]
    assert len(returned)==3 and returned[-1]['budget_role']=='省钱备选'
