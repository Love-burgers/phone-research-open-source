"""Cross-layer business invariants; synthetic facts, no network or paid models."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.chat import SelectionAssistant
from app.chat_agent import ChatAgentRun
from app.chat_answer import AnswerComposer
from app.chat_intent import apply_model_patch, resolve_references
from app.chat_schemas import AnswerBlock, ChatMessage, ChatRequest, ChatSession, ChatSource, IntentPatch, SelectionProfile
from app.chat_store import ChatStore, SessionConflict
from app.evidence_registry import finish_refresh, merge_sources, refresh_notices
from app.feature_requirements import requirement_atoms, update_requirements
from app.knowledge import KnowledgeBase, KnowledgeChunk, KnowledgeIndex, load_index, save_index
from app.local_embeddings import LocalEmbeddings
from app.selection_catalog import CatalogPhone
from app.selection_prices import unsupported_prices
from app.selection_requirements import card_matches, check_requirement, literal_requirements, recommendation_gaps
from app.selection_presentation import quoted_price
from app.agent_tools import LocalQueryArguments


class Offline:
    available=False


class Public:
    def __init__(self):self.calls=0
    def collect_result(self,*args,**kwargs):
        self.calls+=1
        return {'sources':[],'actions':[]}


def phone():
    return CatalogPhone(id='oneplus-13',name='一加 13',aliases=['一加13'],brand='一加',price=3000,
        variant='12GB+256GB',price_options=[{'variant':'12GB+256GB','price':3000}],
        checked_on='2026-10-25',topics=['拍照'],note='Synthetic fixture')


def source(quote='一加 13，采用直屏，支持无线充电，机身重量210克，5000万像素主摄，电池6000mAh。',**changes):
    data=dict(id='spec',phone_id='oneplus-13',title='一加 13 合成资料',url='https://8.8.8.8/spec',quote=quote,
        checked_on='2026-10-25',market='中国大陆',evidence_kind='spec')
    data.update(changes)
    return ChatSource(**data)


def price(value=3000,url='https://8.8.8.8/price',**changes):
    return source(f'一加 13，12GB+256GB，{value}元。',id=url,url=url,evidence_kind='price',
        reference_price=value,price_variant='12GB+256GB',**changes)


def assistant(tmp_path,model=None,store=None,knowledge=None):
    sources=[source(),price()]
    chunks=[KnowledgeChunk(phone_name='一加 13',title=s.title,source_url=s.url,text=s.quote,
        chunk_id=s.id,document_id=s.id,checked_on=s.checked_on,evidence_kind=s.evidence_kind,
        market=s.market,reference_price=s.reference_price,price_variant=s.price_variant) for s in sources]
    return SelectionAssistant(store or ChatStore(tmp_path/'chat.sqlite3'),catalog={'oneplus-13':phone()},
        model_client=model or Offline(),knowledge=knowledge or KnowledgeBase(KnowledgeIndex(chunks=chunks)),
        source_loader=Public(),agent_enabled=False)


@pytest.mark.parametrize('mode',['local','auto','web'])
def test_atomic_conditions_survive_edit_cancel_and_cold_restore(tmp_path,mode):
    a=assistant(tmp_path)
    s=a.turn(ChatRequest(message='预算4000，主要拍照，必须直屏，支持无线充电，至少256GB，重量不超过190克，推荐一款。',search_mode=mode))
    assert not s.profile.candidate_ids  # The recorded 210g phone fails the actual gate.
    a=assistant(tmp_path,store=ChatStore(a.store.path))
    s=a.turn(ChatRequest(session_id=s.session_id,search_mode=mode,
        message='重量改为不超过220克，取消无线充电要求，直屏和至少256GB都保留，推荐一款。'))
    assert s.profile.candidate_ids==['oneplus-13']
    assert '必须直屏' in s.profile.requirements and '至少256GB' in s.profile.requirements
    assert not any('190' in r or '无线' in r for r in s.profile.requirements)
    a=assistant(tmp_path,store=ChatStore(a.store.path))
    s=a.turn(ChatRequest(session_id=s.session_id,search_mode=mode,
        message='现在不限制重量了，预算提高到6000，其它条件不变，再推荐两款。'))
    assert s.profile.requirements==['必须直屏','至少256GB']
    assert s.profile.focus_phone_id is None and s.profile.recommendation_count==2
    assert s.messages[-1].blocks and all(b.phone_id in s.profile.candidate_ids for b in s.messages[-1].blocks)
    assert a.store.load(s.session_id)==s


def test_model_paraphrase_and_offline_change_share_atomic_reducer():
    initial=['必须直屏','支持无线充电','至少256GB','重量不超过190克']
    text='重量改为不超过220克，取消无线充电要求，直屏和至少256GB都保留。可以联网补查。'
    p=SelectionProfile(requirements=initial)
    apply_model_patch(p,IntentPatch(quote=text,requirements=['重量不超过220克'],
        remove_requirements=['重量不超过190克','支持无线充电']),text,{})
    q=SelectionProfile(requirements=initial);literal_requirements(q,text)
    assert p.requirements==q.requirements==['必须直屏','至少256GB','重量改为不超过220克']
    assert [a.key for a in requirement_atoms('至少256GB并且重量不超过190克')]==['storage','weight']


@pytest.mark.parametrize('text',['不要联网','有测试条件的资料','保留测试条件','请保留原日期，不要当成最新到手价'])
def test_controls_never_become_product_gates(text):
    p=SelectionProfile(requirements=['必须直屏'])
    apply_model_patch(p,IntentPatch(quote=text,requirements=[text]),text,{})
    assert p.requirements==['必须直屏']


def test_evidence_preferences_persist_separately_without_reselecting_candidates(tmp_path):
    a=assistant(tmp_path);s=a.turn(ChatRequest(message='一加13的无线充电？请保留测试条件和核对日期',search_mode='local'))
    assert not s.profile.requirements and set(s.profile.answer_requirements)=={'测试条件','核对日期'}
    restored=ChatStore(a.store.path).load(s.session_id)
    assert restored.profile.answer_requirements==s.profile.answer_requirements
    s=a.turn(ChatRequest(message='继续说说它的重量',session_id=s.session_id,search_mode='local'))
    assert s.profile.answer_requirements==restored.profile.answer_requirements
    assert any('210克' in b.text for b in s.messages[-1].blocks)


def test_ungrounded_clear_and_compound_removal_cannot_delete_other_properties():
    p=SelectionProfile(requirements=['必须直屏','至少256GB'])
    apply_model_patch(p,IntentPatch(quote='预算6000',budget_max=6000,clear_requirements=True),'预算6000',{})
    assert p.requirements==['必须直屏','至少256GB']
    old='不要曲面屏且重量不超过190克'
    assert update_requirements([old],'取消重量限制',removals=[old])==['不要曲面屏']
    assert update_requirements(['必须直屏','需要卫星通信'],'取消所有要求',clear=True)==[]


def test_unmentioned_redundant_model_fields_do_not_reset_confirmed_preferences():
    p=SelectionProfile(budget_unlimited=True,recommendation_count=2,high_refresh_required=True,
        priorities=['游戏'],games=['原神'],preferred_brands=['一加'])
    apply_model_patch(p,IntentPatch(quote='继续说',budget_max=6000,budget_unlimited=False,recommendation_count=3,
        high_refresh_required=False,clear_priorities=True,clear_games=True,preferred_brands=[]),'继续说',{})
    assert p.budget_max is None and p.budget_unlimited and p.recommendation_count==2
    assert p.high_refresh_required and p.priorities==['游戏'] and p.games==['原神'] and p.preferred_brands==['一加']


@pytest.mark.parametrize('with_model',[False,True])
def test_requirement_limit_never_silently_evicts_an_unmentioned_condition(tmp_path,with_model):
    class Delta:
        available=True
        def complete_json(self,messages):return {'quote':messages[-1]['content'],'budget_max':5000,'requirements':['必须防尘13']}
    a=assistant(tmp_path,model=Delta() if with_model else None);s=a.store.create()
    s.profile=SelectionProfile(budget_max=4000,priorities=['拍照'],requirements=[f'必须防尘{i}' for i in range(12)])
    s=a.store.save(s)
    result=a.turn(ChatRequest(session_id=s.session_id,message='预算改到5000，必须防尘13',search_mode='local'))
    assert result.profile.requirements==s.profile.requirements and result.profile.budget_max==4000
    assert '12项' in result.messages[-1].content and not result.messages[-1].cards


@pytest.mark.parametrize('text',['取消重量并保留直屏','取消重量但直屏不变','取消重量，直屏和至少256GB都保留'])
def test_cancel_and_retain_operations_have_separate_targets(text):
    assert update_requirements(['必须直屏','至少256GB','重量不超过190克'],text)==['必须直屏','至少256GB']


@pytest.mark.parametrize('requirement',['至少256GB并且重量不超过190克','至少256GB和必须直屏和重量不超过190克'])
def test_storage_cannot_bypass_other_hard_gates(requirement):
    p=SelectionProfile(budget_max=4000,requirements=[requirement])
    assert check_requirement('重量不超过190克',[source()]).status=='unsatisfied'
    assert any('190' in gap for gap in recommendation_gaps(phone(),p,[source(),price()]))
    assert check_requirement('需要无线充电和卫星通信',[source()]).status=='unknown'


@pytest.mark.parametrize('spelling',['其它','其他'])
def test_condition_preservation_is_not_a_phone_reference(spelling):
    p=SelectionProfile(candidate_ids=['oneplus-13'],focus_phone_id='oneplus-13')
    assert resolve_references(spelling+'条件不变，再推荐两款',p,[],{'oneplus-13':phone()})==([], '')
    assert resolve_references('它支持无线充电吗？',p,[],{'oneplus-13':phone()})[0]==['oneplus-13']


def answer_state(sources,message='一加13支持无线充电吗？'):
    return {'session':ChatSession(session_id='0'*32,profile=SelectionProfile(candidate_ids=['oneplus-13'],focus_phone_id='oneplus-13')),
        'message':message,'catalog':{'oneplus-13':phone()},'selected':['oneplus-13'],'sources':sources,
        'acknowledged':[],'recommendation_turn':False}


@pytest.mark.parametrize('quote,status',[('不支持无线充电。','unsatisfied'),('支持无线充电。','satisfied'),
    ('无线充电待确认。','unknown'),('支持无线充电；另一记录不支持无线充电。','conflict')])
def test_answer_boolean_matches_the_same_four_state_evidence(tmp_path,quote,status):
    s=source('一加 13，'+quote)
    assert check_requirement('需要无线充电',[s]).status==status
    c=AnswerComposer(assistant(tmp_path),answer_state([s]))
    claim=AnswerBlock(phone_id='oneplus-13',text='支持无线充电。',evidence_ids=[s.id])
    assert c.validate_block(claim,{s.id:s})==(status=='satisfied')


def test_independent_valid_block_survives_bad_sibling_without_citation_relaxation(tmp_path):
    s=source()
    class Draft:
        available=True
        def complete_json(self,messages):return {'blocks':[
            {'phone_id':'oneplus-13','text':'电池6000mAh。','evidence_ids':[s.id]},
            {'phone_id':'unrequested','text':'无引用段落','evidence_ids':[]}]}
    state=answer_state([s],'一加13的电池容量多少？')
    reply=AnswerComposer(assistant(tmp_path),state,Draft()).run()['reply']
    assert len(reply.blocks)==1 and reply.blocks[0].evidence_ids==[s.id]
    assert state['answer_draft']['blocks'][1]['evidence_ids']==[] and state['answer_rejections']


def test_numeric_field_question_does_not_answer_old_usage(tmp_path):
    a=assistant(tmp_path)
    s=a.turn(ChatRequest(message='一加13重量多少？',search_mode='local'))
    assert any('210克' in b.text for b in s.messages[-1].blocks)
    assert not any('5000万' in b.text for b in s.messages[-1].blocks)


def test_scalar_price_conflicts_and_latest_same_url_share_one_fact_view():
    p=SelectionProfile(budget_max=4000)
    first=price(); opposing=price(3500,'https://8.8.8.8/other')
    assert any('冲突' in label for label in card_matches(phone().id,p,[first,opposing])[0])
    assert '来源冲突未解决' in recommendation_gaps(phone(),p,[first,opposing])
    assert quoted_price(phone(),p,[first,opposing]) is None
    newer=price(3500,checked_on='2026-10-26')
    assert not any('冲突' in label for label in card_matches(phone().id,p,[first,newer])[0])
    assert '3500' in quoted_price(phone(),p,[first,newer])[0]


def test_same_source_new_property_snapshot_replaces_only_its_older_observation(tmp_path):
    old=source('一加 13，不支持无线充电，重量210克。',checked_on='2026-10-20')
    new=source('一加 13，支持无线充电，重量180克。',id='new',checked_on='2026-10-25')
    assert check_requirement('需要无线充电',[old,new]).status=='satisfied'
    assert check_requirement('重量不超过190克',[old,new]).status=='satisfied'
    c=AnswerComposer(assistant(tmp_path),answer_state([old,new]))
    assert not c.validate_block(AnswerBlock(phone_id=old.phone_id,text='不支持无线充电。',evidence_ids=[old.id]),{old.id:old,new.id:new})
    opposing=new.model_copy(update={'id':'other','url':'https://8.8.8.8/other','quote':old.quote})
    assert check_requirement('需要无线充电',[new,opposing]).status=='conflict'


def test_new_version_table_supersedes_old_chunks_but_preserves_other_sources():
    old=price(3000,refresh_status='pending',checked_on='2026-10-20')
    other=price(3200,'https://8.8.8.8/other',refresh_status='pending',checked_on='2026-10-20')
    fresh=price(3500,source_type='page',refresh_status='verified')
    state={'sources':{s.id:s for s in [old,other]},'allowed_urls':set(),'actions':[]}
    # A real page uses a different ID from an imported per-version knowledge chunk.
    fresh=fresh.model_copy(update={'id':'new-table'})
    merge_sources(state,phone().id,[fresh],{phone().id:phone()},SelectionProfile(),refresh={'price'},fresh=True)
    turn={'session':ChatSession(session_id='0'*32),'message':'刷新最新价格','search_mode':'web'}
    finish_refresh(turn,state,attempted={phone().id})
    assert state['sources'][old.id].refresh_status=='superseded'
    assert state['sources'][other.id].refresh_status=='failed'
    assert old.refresh_status=='pending' and old.checked_on=='2026-10-20'
    notices=refresh_notices(list(state['sources'].values()))
    assert any('部分' in n for n in notices) and not any('刷新未取得可核验的新记录' in n for n in notices)


def test_budget_and_forward_version_reference_are_not_false_price_claims():
    s=price()
    text='12GB+256GB售价3000元，在4000元以内。此前记录3000元对应12GB+256GB。'
    assert not unsupported_prices(text,[s],4000)
    assert unsupported_prices('16GB+512GB售价3000元。',[s],4000)


def test_repeated_local_synonyms_cannot_scan_the_same_missing_field_forever(tmp_path):
    a=assistant(tmp_path)
    session=a.store.create()
    # Remove weight from this archive to exercise a real missing target.
    a.knowledge.index.chunks=[a.knowledge.index.chunks[1]]
    turn={'session':session,'message':'一加13的重量多少？','search_mode':'auto'}
    turn.update(a.understand(turn));run=ChatAgentRun(a,turn);state=run.initialize({})
    first=run.execute_scoped(state,'search_phone_knowledge',LocalQueryArguments(phone_name='一加 13',query='重量'))
    second=run.execute_scoped(state,'search_phone_knowledge',LocalQueryArguments(phone_name='一加 13',query='机身有多重'))
    assert first['target_gaps'] and second['status']=='duplicate' and second['fact_progress']==0


class Local(LocalEmbeddings):
    def __init__(self):pass
    def embed_documents(self,texts):return [[1.,0.] for t in texts]
    def embed_query(self,text):return [1.,0.]


def test_index_update_serializes_instances_and_refuses_old_session_rollback(tmp_path):
    initial=KnowledgeChunk(phone_name='一加 13',title='合成旧记录',source_url='https://www.oneplus.com/cn/13/specs',
        text='一加 13，12GB+256GB，3000元。',chunk_id='old',document_id='old',checked_on='2026-10-20',
        evidence_kind='price',market='中国大陆',reference_price=3000,price_variant='12GB+256GB',vector=[1.,0.])
    index=KnowledgeIndex(method='vector',embedding_signature='fixture',chunks=[initial]);path=tmp_path/'index.json';save_index(index,path)
    def base():
        b=KnowledgeBase(index,Local());b.path=path;return b
    new=price(3500,initial.source_url,source_type='page',checked_on='2026-10-25',
        price_options=[{'variant':'12GB+256GB','price':3500}])
    old=price(3000,initial.source_url,source_type='page',checked_on='2026-10-20',
        price_options=[{'variant':'12GB+256GB','price':3000}])
    b1,b2=base(),base()
    b1.remember_sources([new],{'oneplus-13':phone()})
    assert b2.remember_sources([old],{'oneplus-13':phone()})['added_chunks']==0
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda b:b.remember_sources([new],{'oneplus-13':phone()}),[base(),base()]))
    assert {c.reference_price for c in load_index(path).chunks if c.evidence_kind=='price'}=={3500}


def test_failed_session_save_has_no_global_knowledge_side_effect(tmp_path):
    a=assistant(tmp_path);calls=[]
    a.knowledge.remember_sources=lambda *args: calls.append(args)
    def fail(*args):raise SessionConflict()
    a.store.save=fail
    with pytest.raises(SessionConflict):a.turn(ChatRequest(message='一加13无线充电？',search_mode='local'))
    assert not calls


# Follow-up failures: keep these synthetic, public and independent of saved runs.
@pytest.mark.parametrize('requirement',[
    '具体存储版本','说明核对日期','展示容量版本','列出支持无线充电的来源',
    '请核对重量','给出测试条件','说明电池容量',
])
def test_reporting_operations_do_not_create_hardware_gates(requirement):
    assert update_requirements(['必须直屏'],requirement,additions=[requirement])==['必须直屏']


@pytest.mark.parametrize('requirement',['必须卫星通信','需要IP68','需要存储版本512GB','重量不超过190克'])
def test_unknown_or_numeric_product_predicates_are_not_reporting_preferences(requirement):
    atoms=requirement_atoms(requirement)
    assert atoms and atoms[0].text==requirement
    assert update_requirements([],requirement)==[requirement]


def test_mixed_product_and_reporting_operations_survive_cold_profile():
    p=SelectionProfile(requirements=['必须直屏'])
    text='需要卫星通信并说明核对日期，列出具体存储版本'
    apply_model_patch(p,IntentPatch(quote=text,requirements=['需要卫星通信','说明核对日期','具体存储版本']),text,{})
    p=SelectionProfile.model_validate_json(p.model_dump_json())
    assert p.requirements==['必须直屏','需要卫星通信']
    assert set(p.answer_requirements)=={'核对日期','存储版本'}


@pytest.mark.parametrize('text,valid',[
    ('历史记录12GB+256GB 3000元，不是最新到手价。',True),
    ('没有查到最新价格，旧记录12GB+256GB 3000元。',True),
    ('12GB+256GB当前价3000元，不是最新到手价。',False),
    ('历史3000元，最新3000元。',False),
    ('当前价格如下：12GB+256GB 3000元。',False),
    ('12GB+256GB 3000元是今天的到手价。',False),
    ('当前价格（2026-10-20核对）12GB+256GB 3000元。',False),
])
def test_temporal_price_role_is_per_assertion_and_negation(text,valid):
    s=price(3000,refresh_status='failed',checked_on='2026-10-20')
    assert bool(unsupported_prices(text,[s],4000)) is not valid


def test_current_price_needs_today_body_and_deal_price_provenance():
    from app.phone_names import checked_today
    old=price(3000,refresh_status='verified',source_type='page',checked_on='2026-01-01')
    fresh=old.model_copy(update={'checked_on':checked_today()})
    assert unsupported_prices('当前价3000元。',[old])
    assert not unsupported_prices('当前页面标价3000元。',[fresh])
    assert unsupported_prices('最新到手价3000元。',[fresh])
    deal=fresh.model_copy(update={'quote':fresh.quote+'今日到手价3000元。'})
    assert not unsupported_prices('最新到手价3000元。',[deal])


@pytest.mark.parametrize('mode',['local','auto','web'])
@pytest.mark.parametrize('with_model',[False,True])
def test_historical_price_fallback_and_valid_draft_share_guard(tmp_path,mode,with_model):
    s=price(3000,refresh_status='failed',checked_on='2026-10-20')
    class Draft:
        available=True
        def complete_json(self,*args):return {'blocks':[{'phone_id':s.phone_id,
            'text':'12GB+256GB历史页面参考价3000元，不是最新到手价。','evidence_ids':[s.id]}]}
    state=answer_state([s],'接着说它的价格和具体存储版本，说明核对日期。');state['search_mode']=mode
    reply=AnswerComposer(assistant(tmp_path),state,Draft() if with_model else Offline()).run()['reply']
    assert reply.blocks and '3000元' in reply.blocks[0].text and '2026-10-20' in reply.blocks[0].text
    assert reply.blocks[0].evidence_ids==[s.id]
    assert any('此前刷新' in d for d in reply.details) and not any('本轮刷新' in d for d in reply.details)


def test_unbacked_price_cannot_be_salvaged_by_history_disclaimer(tmp_path):
    s=price(3000,refresh_status='failed',checked_on='2026-10-20')
    class Draft:
        available=True
        def complete_json(self,*args):return {'blocks':[{'phone_id':s.phone_id,
            'text':'历史记录12GB+256GB 3900元，不是最新价。','evidence_ids':[s.id]}]}
    reply=AnswerComposer(assistant(tmp_path),answer_state([s],'它的价格？'),Draft()).run()['reply']
    assert reply.blocks and all('3900元' not in b.text for b in reply.blocks)
    assert any('3000元' in b.text for b in reply.blocks)


def test_inquiry_can_list_cited_over_budget_versions_without_recommending_them(tmp_path):
    s=price(5000)
    state=answer_state([s],'它的各版本价格多少？');state['session'].profile.budget_max=4000
    c=AnswerComposer(assistant(tmp_path),state)
    assert c.validate_block(AnswerBlock(phone_id=s.phone_id,text='12GB+256GB页面价5000元。',evidence_ids=[s.id]),{s.id:s})
    assert not c.validate_block(AnswerBlock(phone_id=s.phone_id,text='12GB+256GB 5000元，符合预算。',evidence_ids=[s.id]),{s.id:s})


@pytest.mark.parametrize('own,comparison',[(6200,12070),(12070,6200)])
def test_dom_and_old_cache_follow_fact_owner_not_number_size(tmp_path,own,comparison):
    from app.phone_sources import extract_phone_excerpt
    from app.evidence_rules import unsupported_parameters
    from app.selection_presentation import configuration_facts
    html=f'<title>一加13参数</title><main><h1>一加13</h1><li>电池：{own}mAh<span class="text-r">行业最高：<font>{comparison}mAh</font></span></li></main>'
    text=extract_phone_excerpt('一加 13',html,'电池配置')
    assert text and str(own) in text and str(comparison) not in text
    cached=source(f'一加 13 / 电池：{own}mAh 行业最高：{comparison}mAh / CPU：天玑9400+ 行业最高：骁龙8至尊版')
    assert unsupported_parameters(f'电池{comparison}mAh。',[cached])
    assert not unsupported_parameters(f'电池{own}mAh。',[cached])
    assert all(str(comparison) not in v and '至尊版' not in v for v,sid in configuration_facts(phone(),['续航','游戏'],[cached]))
    state={'sources':{},'allowed_urls':set(),'actions':[]}
    merge_sources(state,phone().id,[cached],{phone().id:phone()},SelectionProfile())
    assert str(comparison) not in state['sources'][cached.id].quote
    assert str(comparison) in cached.quote  # Original saved history is immutable.


def test_game_coverage_keeps_temperature_and_power_separate_and_ignores_room_temperature():
    from app.evidence_registry import evidence_gaps
    from app.evidence_rules import game_measurements
    text='原神，极高画质，跑图30分钟，平均帧率60.2fps，室温25℃。'
    turn=answer_state([],'原神的帧率、发热、功耗和耗电怎么样？')
    turn['task']={'kind':'detail','fields':[],'games':['原神']}
    review=source(text,evidence_kind='review')
    assert game_measurements(text,'原神')=={'帧率'}
    gaps=evidence_gaps(turn,phone(),[review])
    assert len(gaps)==3 and all('帧率' not in g for g in gaps)
    measured=review.model_copy(update={'quote':text+'机身最高温度43℃，平均功耗4.2W，耗电15%。'})
    assert game_measurements(measured.quote,'原神')=={'帧率','温度','功耗','耗电'}
    assert evidence_gaps(turn,phone(),[measured])==[]
    assert game_measurements(measured.quote,'王者荣耀')==set()


def test_old_price_refresh_status_is_retained_but_not_displayed_as_this_game_turn(tmp_path):
    old=price(refresh_status='failed',checked_on='2026-10-20')
    turn=answer_state([old],'这款玩原神的帧率和发热怎么样？')
    reply=AnswerComposer(assistant(tmp_path),turn).run()['reply']
    assert all('刷新' not in detail for detail in reply.details)
    assert old.refresh_status=='failed' and old.checked_on=='2026-10-20'


@pytest.mark.parametrize('fetch_limit,expected_fetches',[(0,0),(4,1)])
def test_new_search_leads_get_one_bounded_read_opportunity(fetch_limit,expected_fetches):
    from time import monotonic
    from langchain_core.messages import AIMessage,HumanMessage
    from app.tool_loop import AgentPolicy,build_tool_loop
    class Decisions:
        n=0;last_usage={'total_tokens':10}
        def invoke_tools(self,*args,**kw):
            self.n+=1
            name='web_search' if self.n==1 else 'fetch_page'
            return AIMessage(content='',tool_calls=[{'name':name,'args':{'url':'https://example.com/body'},'id':str(self.n)}])
    fetched=[]
    def execute(state,name,args):
        state['search_calls' if name=='web_search' else 'fetch_calls']+=1
        if name=='fetch_page':fetched.append(args['url'])
        return {'fact_progress':0,'candidates':[{'url':'https://example.com/body','status':'body_unverified'}] if name=='web_search' else []}
    init=lambda _:{'messages':[HumanMessage(content='test')],'actions':[],'nodes':[],'sources':{},'started_at':monotonic(),
        'model_calls':3,'tool_calls_used':4,'search_calls':0,'fetch_calls':0,'charged_tokens':0,'usage':[],'no_progress':2,'stop_reason':''}
    limits=AgentPolicy(max_fetch_calls=fetch_limit,max_total_tokens=128000,max_context_characters=128000)
    result=build_tool_loop(client=Decisions(),limits=limits,schemas=[],initialize=init,execute=execute,finish=lambda state,*a:state).invoke({})
    assert len(fetched)==expected_fetches
    assert result['model_calls']<=limits.max_model_calls and result['tool_calls_used']<=limits.max_tool_calls


def test_ever_changing_links_cannot_reset_stagnation_forever():
    from time import monotonic
    from langchain_core.messages import AIMessage,HumanMessage
    from app.tool_loop import AgentPolicy,build_tool_loop
    class Decisions:
        n=0;last_usage={'total_tokens':10}
        def invoke_tools(self,*args,**kw):
            self.n+=1
            return AIMessage(content='',tool_calls=[{'name':'web_search','args':{},'id':str(self.n)}])
    def execute(state,*args):
        return {'fact_progress':0,'candidates':[{'url':f'https://example.com/{state["tool_calls_used"]}'}]}
    init=lambda _:{'messages':[HumanMessage(content='test')],'actions':[],'nodes':[],'sources':{},'started_at':monotonic(),
        'model_calls':0,'tool_calls_used':0,'search_calls':0,'fetch_calls':0,'charged_tokens':0,'usage':[],'no_progress':1,'stop_reason':''}
    result=build_tool_loop(client=Decisions(),limits=AgentPolicy(max_model_calls=12,max_tool_calls=18,max_no_progress=2,
        max_total_tokens=128000,max_context_characters=128000),schemas=[],initialize=init,execute=execute,finish=lambda state,*a:state).invoke({})
    assert result['stop_reason']=='no_progress' and result['tool_calls_used']==3


@pytest.mark.parametrize('mode',['local','auto','web'])
def test_no_qualified_recommendation_names_the_actual_hard_gap(tmp_path,mode):
    a=assistant(tmp_path)
    s=a.turn(ChatRequest(message='预算4000，主要拍照，必须直屏，重量不超过190克，推荐一款。',search_mode=mode))
    reply=s.messages[-1]
    assert not reply.blocks and not reply.cards and '190克' in reply.content
    assert any('190克' in detail for detail in reply.details)


def test_duplicate_local_lookup_reports_current_coverage_after_web_measurement(tmp_path):
    from app.evidence_registry import evidence_gaps
    a=assistant(tmp_path);session=a.store.create()
    turn={'session':session,'message':'一加13原神的帧率和发热怎么样？','search_mode':'web'}
    turn.update(a.understand(turn));run=ChatAgentRun(a,turn);state=run.initialize({})
    first=run.execute_scoped(state,'search_phone_knowledge',LocalQueryArguments(phone_name='一加 13',query='原神帧率发热'))
    review=source('一加 13，原神极高画质跑图30分钟，平均帧率60.2fps。',id='review',evidence_kind='review')
    state['sources'][review.id]=review
    second=run.execute_scoped(state,'search_phone_knowledge',LocalQueryArguments(phone_name='一加 13',query='游戏帧率温度'))
    assert second['status']=='duplicate' and second['target_gaps']==evidence_gaps(turn,phone(),list(state['sources'].values()))
    assert any('温度' in gap for gap in second['target_gaps'])
    assert not any('帧率' in gap for gap in second['target_gaps'])
    assert any('帧率' in gap for gap in second['local_archive_gaps'])


def test_agent_empty_recommendation_explains_gap_and_actual_stop_reason(tmp_path):
    a=assistant(tmp_path);session=a.store.create()
    turn={'session':session,'message':'预算4000，主要拍照，重量不超过190克，推荐一款','search_mode':'auto'}
    turn.update(a.understand(turn));run=ChatAgentRun(a,turn);state=run.initialize({})
    for s in [source(),price()]:state['sources'][s.id]=s
    run.scope=[phone().id]  # Mimic an admitted discovery tool result.
    state['stop_reason']='no_progress'
    reply=run.finish(state,lambda *a:('model_call_limit',0),lambda *a:{})['report']
    assert not reply.cards and '190克' in reply.content
    assert any('连续查询未增加可核验事实' in d for d in reply.details)


def compared_price():
    return source('一加 13 256GB版本在官网售价5000元。作为对比，vivo X200S（12GB+256GB 2500元，国补2100.15元起）。',
        title='一加 13 价格文章',id='compared',evidence_kind='price',source_type='page',
        reference_price=2500,price_variant='12GB+256GB',price_options=[{'variant':'12GB+256GB','price':2500}])


@pytest.mark.parametrize('target,expected',[
    ('一加 13',[('256GB',5000)]),('vivo X200S',[('12GB+256GB',2500)]),
])
def test_comparison_paragraph_prices_belong_to_their_local_model(target,expected):
    from app.phone_sources import extract_price_options
    s=compared_price()
    html='<title>'+target+'价格</title><main><p>'+s.quote+'</p></main>'
    assert [(o.variant,o.price) for o in extract_price_options(target,html)[0]]==expected


@pytest.mark.parametrize('text,valid',[
    ('一加13 12GB+256GB页面参考价2500元。',False),
    ('正文中2500元是vivo X200S的12GB+256GB价格，不能当作一加13售价；其256GB官网价5000元，超过4000元预算。',True),
    ('正文中的2500元不是一加13价格。',True),
    ('一加13不是5000元。',False),
    ('一加13不是9999元。',False),
    ('一加13官方售价5000元。',False),  # Article is not an official page.
])
def test_false_price_and_legitimate_correction_share_final_validator(tmp_path,text,valid):
    s=compared_price();state=answer_state([s],'一加13的价格是不是混了别的机型？')
    state['session'].profile.budget_max=4000
    c=AnswerComposer(assistant(tmp_path),state)
    assert c.validate_block(AnswerBlock(phone_id=s.phone_id,text=text,evidence_ids=[s.id]),{s.id:s}) is valid


def test_corrupted_cached_offer_cannot_reenter_fallback_or_official_archive(tmp_path):
    from app.selection_prices import source_price_options
    from app.knowledge_ingest import source_documents
    s=compared_price()
    assert source_price_options(s)==[]
    forged=s.model_copy(update={'url':'https://www.oneplus.com/cn/13/specs',
        'quote':'一加 13，12GB+256GB，页面展示价2500元。原文：'+s.quote})
    assert source_documents([forged],{phone().id:phone()})==[]
    reply=AnswerComposer(assistant(tmp_path),answer_state([s],'一加13价格多少？'),Offline()).run()['reply']
    assert not any('2500元' in b.text for b in reply.blocks)
    assert s.reference_price==2500 and s.price_options  # Input evidence remains immutable.


def test_sqlite_restores_current_derived_price_without_rewriting_failed_history(tmp_path):
    from pydantic import HttpUrl
    s=compared_price();p=phone();p.price=2500;p.variant='12GB+256GB';p.price_url=HttpUrl(s.url)
    p.price_options=s.price_options
    store=ChatStore(tmp_path/'restore.db');session=store.create();session.discovered_phones=[p]
    session.messages=[ChatMessage(role='assistant',content='错误旧报价2500元',sources=[s],cards=[p.card(SelectionProfile())])]
    saved=store.save(session);before=saved.messages[0].model_dump()
    restored=store.load(session.session_id)
    assert restored.discovered_phones[0].price is None and not restored.discovered_phones[0].price_options
    assert restored.messages[0].model_dump()==before
    import sqlite3
    with sqlite3.connect(store.path) as db:
        raw=json.loads(db.execute('SELECT data FROM sessions').fetchone()[0])
    assert raw['discovered_phones'][0]['price']==2500  # Projection is lazy, never an unsolicited DB rewrite.


@pytest.mark.parametrize('text,expected',[
    ('7000内',7000),('7000以内',7000),('7000',7000),('七千',7000),
    ('电池7000mAh',None),('重量不超过700克',None),('至少256GB',None),('第7000个',None),
])
def test_contextual_short_budget_accepts_money_without_reinterpreting_specifications(text,expected):
    from app.chat_intent import read_budget
    history=[ChatMessage(role='assistant',content='你希望预算控制在多少元？')]
    p=SelectionProfile()
    if expected:
        apply_model_patch(p,IntentPatch(quote=text,budget_max=expected),text,{},history)
        assert p.budget_max==expected
    assert read_budget(text,history)==expected
    if text=='7000':assert read_budget(text)==None


@pytest.mark.parametrize('mode',['local','auto','web'])
def test_known_clarification_never_executes_search_or_answer_model(tmp_path,mode):
    class Model:
        available=True;calls=[];last_usage={'total_tokens':10}
        def complete_json(self,messages,**kw):
            self.calls.append('intent');text=messages[-1]['content']
            if text=='游戏':return {'quote':text,'priorities':['游戏']}
            if text=='7000内':return {'quote':text,'budget_max':7000}
            return {'quote':text,'preferred_brands':['Apple']}
        def invoke_tools(self,*a,**kw):raise AssertionError('Clarification must precede agent execution')
    a=assistant(tmp_path,model=Model());a.agent_enabled=True
    # Rebuild through the public constructor so the graph has Agent enabled.
    a=SelectionAssistant(a.store,model_client=a.model,catalog=a.catalog,knowledge=a.knowledge,source_loader=a.source_loader,agent_enabled=True)
    first=a.turn(ChatRequest(message='比较喜欢iPhone，有什么性价比高的款式',search_mode=mode))
    second=a.turn(ChatRequest(session_id=first.session_id,message='7000内',search_mode=mode))
    assert second.profile.budget_max==7000 and '用途' not in second.profile.requirements
    assert '看重' in second.messages[-1].content and second.messages[-1].agent_trace['metrics']['tool_calls']==0
    assert first.messages[-1].agent_trace['metrics']['model_calls']==1
    assert len(a.model.calls)==2


@pytest.mark.parametrize('stage',['intent','decision','answer'])
def test_project_quota_stops_the_pipeline_without_smaller_followup_requests(tmp_path,stage):
    from app.usage_guard import QuotaExceeded
    from langchain_core.messages import AIMessage
    class Model:
        available=True;last_usage={'total_tokens':10}
        def __init__(self):self.calls=[]
        def complete_json(self,messages,**kw):
            step='intent' if messages[0]['content'].startswith('提取') else 'answer';self.calls.append(step)
            if stage==step:raise QuotaExceeded('daily_token_limit')
            return {'quote':messages[-1]['content'],'selected_phones':['一加13']}
        def invoke_tools(self,*a,**kw):
            self.calls.append('decision')
            if stage=='decision':raise QuotaExceeded('daily_token_limit')
            return AIMessage(content=json.dumps({'kind':'answer','message':'回答电池容量'}))
    model=Model();base=assistant(tmp_path,model=model)
    a=SelectionAssistant(base.store,model_client=model,catalog=base.catalog,knowledge=base.knowledge,source_loader=base.source_loader,agent_enabled=True)
    prior=a.store.create();prior.messages=[ChatMessage(role='assistant',content='已核对资料',sources=[source(),price()])]
    a.store.save(prior)
    s=a.turn(ChatRequest(session_id=prior.session_id,message='一加13电池容量多少？',search_mode='local'));reply=s.messages[-1]
    assert model.calls=={'intent':['intent'],'decision':['intent','decision'],'answer':['intent','decision','answer']}[stage]
    assert '本项目今日Token额度' in reply.content and '未执行' in reply.content
    assert reply.agent_trace['stop_reason']=='daily_token_limit' and reply.agent_trace['metrics']['tool_calls']==0
    assert reply.mode!='资料模式（模型暂不可用）'


def test_quota_during_final_selection_does_not_retry_answer(tmp_path):
    from app.usage_guard import QuotaExceeded
    class Model:
        available=True;last_usage=None;calls=0
        def complete_json(self,*a,**kw):self.calls+=1;raise QuotaExceeded('daily_model_call_limit')
    a=assistant(tmp_path);turn={'session':a.store.create(),'message':'预算4000主要拍照，推荐一款','search_mode':'auto'}
    turn.update(a.understand(turn));model=Model();a.model=model;run=ChatAgentRun(a,turn);state=run.initialize({})
    run.scope=[phone().id];state['sources']={s.id:s for s in [source(),price()]};state['stop_reason']='tool_call_limit'
    reply=run.finish(state,lambda *a:('',1),lambda *a:{})['report']
    assert model.calls==1 and reply.agent_trace['stop_reason']=='daily_model_call_limit'
    assert '次数已达上限' in reply.content


def test_alternative_memory_uses_last_displayed_batch_but_pronoun_does_not_resurrect_it():
    from app.chat_intent import conversation_candidates
    p=SelectionProfile();history=[ChatMessage(role='assistant',content='第一批',cards=[phone().card(p)]),
        ChatMessage(role='assistant',content='本轮没有合格新候选')]
    assert conversation_candidates(p,history,{phone().id:phone()})==[]
    assert conversation_candidates(p,history,{phone().id:phone()},last_displayed=True)==[phone().id]
    assert resolve_references('它的夜景呢？',p,history,{phone().id:phone()})[1]


def test_series_expansion_is_turn_local_and_explicit_rejection_persists(tmp_path):
    from app.phone_names import parse_phone_names,register_phone
    from app.phone_identity import series_excluded
    a=assistant(tmp_path);s=a.store.create();s.profile=SelectionProfile(budget_max=7000,priorities=['游戏'],preferred_brands=['Apple'])
    for name in ['iPhone 16','iPhone 16 Pro','iPhone 15','iPhone 17']:
        p=register_phone(parse_phone_names(name)[0],{});s.discovered_phones.append(p)
    p=s.discovered_phones[0];s.messages=[ChatMessage(role='assistant',content='旧展示',cards=[CatalogPhone.model_validate(p.model_dump()).card(s.profile)])]
    turn={'session':s,'message':'别的款式有没有不止是16系列','search_mode':'web'};turn.update(a.understand(turn))
    run=ChatAgentRun(a,turn)
    assert not s.profile.excluded_series and turn['scope_series']==['Apple||16']
    assert all(not run.phone_allowed(p) for p in s.discovered_phones[:2])
    assert all(run.phone_allowed(p) for p in s.discovered_phones[2:])
    following={'session':s,'message':'预算改到7500重新推荐','search_mode':'web'};following.update(a.understand(following))
    assert not following['scope_series']
    turn={'session':s,'message':'不要16系列，重新推荐','search_mode':'web'};turn.update(a.understand(turn))
    restored=ChatSession.model_validate_json(s.model_dump_json())
    assert restored.profile.excluded_series==['Apple||16']
    assert series_excluded('iPhone 16 Pro Max',restored.profile.excluded_series)


def test_weak_brand_preference_prioritizes_without_becoming_exclusive(tmp_path):
    from app.selection_catalog import eligible
    from app.chat_intent import rule_intent
    p=SelectionProfile();rule_intent(p,'比较喜欢iPhone',{})
    assert p.preferred_brands==['Apple'] and p.brand_scope=='prefer' and eligible(phone(),p)
    rule_intent(p,'只考虑苹果品牌',{})
    assert p.brand_scope=='only' and not eligible(phone(),p)


def test_repeated_alternatives_keep_the_last_shown_batch_out_after_empty_turns(tmp_path):
    a=assistant(tmp_path);s=a.store.create();s.profile=SelectionProfile(budget_max=4000,priorities=['拍照'])
    s.messages=[ChatMessage(role='assistant',content='旧批次',cards=[phone().card(s.profile)]),
        ChatMessage(role='user',content='换几款'),ChatMessage(role='assistant',content='本轮没有合格新款'),
        ChatMessage(role='user',content='还有别的吗'),ChatMessage(role='assistant',content='本轮仍未核实到新款')]
    turn={'session':s,'message':'有没有别的款式','search_mode':'auto'};turn.update(a.understand(turn))
    assert turn['alternative_ids']==[phone().id] and phone().id not in turn['selected']
    assert turn['conversation_route']['operation']=='alternatives' and not s.profile.excluded_phones


def test_series_scope_operation_is_not_an_unknown_hardware_gate():
    assert update_requirements(['必须直屏'],'不止是16系列',additions=['不止是16系列'])==['必须直屏']
    assert requirement_atoms('必须具备16系列同款防水')


def test_monetary_currency_prefix_and_archival_owner_checks_agree():
    from app.selection_prices import source_price_options
    from app.knowledge_ingest import source_documents
    s=price(source_type='page').model_copy(update={'url':'https://www.oneplus.com/cn/13/specs',
        'quote':'一加13，12GB+256GB版本RMB3000。'})
    assert source_price_options(s) and source_documents([s],{phone().id:phone()})
    overseas=s.model_copy(update={'market':'其他地区'})
    assert unsupported_prices('12GB+256GB 3000元。',[overseas])


def test_denied_anonymous_price_is_not_a_sale_offer():
    from app.selection_prices import source_price_options
    s=price().model_copy(update={'title':'版本价说明','quote':'12GB+256GB版本不是3000元。'})
    assert source_price_options(s)==[]


def test_denied_daily_reservation_preserves_existing_project_ledger(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from app.usage_guard import UsageGuard,QuotaExceeded
    from app.model_api import ModelClient
    from app import usage_guard
    guard=UsageGuard(tmp_path/'limits.db')
    day=guard.day()
    with guard.connect() as db:
        guard.add(db,'tokens',day,583808);guard.add(db,'model_calls',day,117)
    settings=SimpleNamespace(chat_daily_model_calls=200,chat_daily_token_limit=600000)
    monkeypatch.setattr(usage_guard,'get_settings',lambda:settings)
    monkeypatch.setattr(usage_guard,'get_usage_guard',lambda *a:guard)
    class NeverProvider:
        def bind(self,**kw):raise AssertionError('Denied reservation must not reach the provider')
    with pytest.raises(QuotaExceeded,match='daily_token_limit'):
        ModelClient(chat_model=NeverProvider()).complete_json([{'role':'user','content':'中'*6000}],max_output_tokens=1200)
    with guard.connect() as db:
        assert guard.value(db,'tokens',day)==583808 and guard.value(db,'model_calls',day)==117
        assert db.execute('SELECT COUNT(*) FROM reservations').fetchone()[0]==0


def test_legacy_weak_brand_history_restores_preference_and_labels_cross_brand_result(tmp_path):
    import sqlite3
    a=assistant(tmp_path);s=a.store.create()
    legacy={'session_id':s.session_id,'revision':0,'profile':{'budget_max':4000,'priorities':['拍照'],'preferred_brands':['Apple']},
        'messages':[{'role':'user','content':'比较喜欢iPhone，有什么性价比高的款式'},
                    {'role':'assistant','content':'你主要看重哪项用途？'}]}
    with sqlite3.connect(a.store.path) as db:db.execute('UPDATE sessions SET data=? WHERE id=?',(json.dumps(legacy,ensure_ascii=False),s.session_id))
    result=a.turn(ChatRequest(session_id=s.session_id,message='继续推荐，主要拍照',search_mode='local'))
    assert result.profile.brand_scope=='prefer' and result.profile.preferred_brands==['Apple']
    assert result.messages[-1].blocks and '其他品牌备选' in result.messages[-1].content


def test_version_of_another_model_cannot_be_borrowed_for_unversioned_price():
    from app.phone_sources import extract_price_options
    html='<title>vivo X200S价格</title><main><p>一加13 256GB配置作为参考，vivo X200S页面售价2500元。</p></main>'
    assert extract_price_options('vivo X200S',html)[0]==[]


def test_corrected_raw_article_price_can_support_its_budget_difference():
    s=compared_price()
    text='原文记载一加13 256GB官网价5000元，超预算1000元。'
    assert not unsupported_prices(text,[s],4000,phone_name='一加13')


def compute_source():
    return source('6 核中央处理器，具有2个性能核心和4个能效核心 / 5 核图形处理器 / ProMotion自适应刷新率技术，最高可达120Hz。电池6000mAh，5000万像素主摄，支持OIS防抖。')


@pytest.mark.parametrize('quote,expected',[
    ('6 核中央处理器，具有2个性能核心 / 5 核图形处理器 / ProMotion自适应刷新率，最高120Hz',['6 核中央处理器','5 核图形处理器','120Hz']),
    ('CPU采用8核架构，GPU为6核，屏幕刷新率144Hz',['CPU采用8核','GPU为6核','144Hz']),
    ('6-core CPU，5-core GPU，屏幕刷新率120Hz',['6-core CPU','5-core GPU','120Hz']),
])
def test_compute_fact_view_handles_roles_word_order_and_adjacent_display_label(quote,expected):
    from app.selection_presentation import configuration_facts
    facts=configuration_facts(phone(),['游戏'],[source(quote)])
    assert [v for v,sid in facts]==expected


def test_specifications_support_initial_use_qualification_but_not_unknown_hard_gate():
    p=SelectionProfile(budget_max=4000,priorities=['游戏','续航','拍照'],games=['原神'])
    evidence=[compute_source(),price()]
    assert recommendation_gaps(phone(),p,evidence)==[]
    p.requirements=['重量不超过190克']
    assert any('硬要求未确认' in g for g in recommendation_gaps(phone(),p,evidence))
    p.requirements=[];p.budget_max=2500
    assert any('版本价' in g for g in recommendation_gaps(phone(),p,evidence))


@pytest.mark.parametrize('text,valid',[
    ('已核实6核CPU、5核GPU与120Hz屏。配置判断：日常游戏有望流畅，可作为初筛备选；长时高画质的稳定性仍取决于散热与适配。',True),
    ('6核CPU、5核GPU，基于硬件的初步判断是性能够用，但持续高负载仍需核实。',True),
    ('已核实6000mAh电池，配置判断续航用途值得考虑，实际使用时长取决于负载。',True),
    ('配置判断：5000万像素主摄和OIS防抖提供影像基础，夜景可能更值得比较，成片仍受算法与场景影响。',True),
    ('配置判断：原神平均60帧，具体表现可能有波动。',False),
    ('初步判断原神帧率为120，具体表现仍需核实。',False),
    ('初步判断原神机身最高40度，具体情况可能有波动。',False),
    ('初步预计游戏稳定90fps，机身40℃，仅供参考。',False),
    ('有120Hz屏，所以游戏帧率上限不受60Hz限制。',False),
    ('配置判断：续航能用12小时，仍存在不确定性。',False),
    ('8核CPU、5核GPU足够游戏，具体表现未知。',False),
    ('6核CPU和5核GPU，初步判断保证任何游戏都不发热。',False),
    ('性能配置够用。',False),
])
def test_initial_judgment_is_allowed_without_invented_measurements_or_premises(tmp_path,text,valid):
    state=answer_state([compute_source()],'这款的游戏和拍照、续航性能？')
    state['session'].profile.priorities=['游戏','续航','拍照'];state['session'].profile.games=['原神']
    c=AnswerComposer(assistant(tmp_path),state)
    assert c.validate_block(AnswerBlock(phone_id=phone().id,text=text,evidence_ids=['spec']),{'spec':compute_source()}) is valid


@pytest.mark.parametrize('text,valid',[
    ('作者报告原神极高画质跑图30分钟，平均58.8fps，机身最高温度43℃。',True),
    ('原神平均60fps。',False),
    ('王者荣耀平均58.8fps。',False),
    ('原神机身温度25℃。',False),
    ('原神极高画质连续玩4小时稳定58.8fps。',False),
    ('原神原生平均58.8fps。',False),
    ('我们实测原神极高画质跑图30分钟，平均58.8fps。',False),
])
def test_measured_numbers_keep_game_duration_temperature_role_and_rendering_mode(tmp_path,text,valid):
    review=source('一加13原神极高画质，60帧设置，跑图30分钟，实测平均58.8fps，室温25℃，机身最高温度43℃。',id='review',evidence_kind='review')
    state=answer_state([review],'原神帧率发热实测？');state['session'].profile.priorities=['游戏'];state['session'].profile.games=['原神']
    c=AnswerComposer(assistant(tmp_path),state)
    assert c.validate_block(AnswerBlock(phone_id=phone().id,text=text,evidence_ids=['review']),{'review':review}) is valid


def test_declared_media_duration_is_not_gaming_endurance(tmp_path):
    from app.evidence_rules import unsupported_measurements
    s=source('一加13，视频播放最长30小时，支持4K 60帧视频录制。')
    assert not unsupported_measurements('官方标称视频播放最长30小时，支持4K 60帧视频录制。',[s],phone().name)
    assert unsupported_measurements('原神可以连续玩30小时。',[s],phone().name,['原神'])
    compact=source('视频播放最长30小时 音频播放最长80小时 内置锂离子电池 可快速充电30分钟充至50%。')
    assert not unsupported_measurements('官方标称视频播放30小时、官方标称音频播放80小时。',[compact],phone().name)
    assert unsupported_measurements('官方标称视频播放80小时。',[compact],phone().name)


@pytest.mark.parametrize('mode',['local','auto','web'])
@pytest.mark.parametrize('text',['一加13怎么样','一加 13怎么样','ONEPLUS13','这款适合我吗？'])
def test_natural_overview_forms_share_target_and_grounded_fallback(tmp_path,mode,text):
    a=assistant(tmp_path);s=a.store.create();s.profile=SelectionProfile(budget_max=4000,priorities=['游戏'],candidate_ids=[phone().id],focus_phone_id=phone().id)
    turn={'session':s,'message':text,'search_mode':mode};turn.update(a.understand(turn));turn['sources']=[compute_source(),price()]
    assert turn['task']['kind']=='overview' and turn['selected']==[phone().id]
    reply=AnswerComposer(a,turn,Offline()).run()['reply']
    assert reply.blocks and all(b.evidence_ids for b in reply.blocks)
    shown=' '.join(b.text for b in reply.blocks)
    assert '120Hz' in shown and '配置判断' in shown and '初筛备选' in shown
    assert '没有可引用的新信息' not in reply.content


@pytest.mark.parametrize('shape',['missing_ids','single_block','singleton_blocks','bad_semantics'])
def test_bounded_protocol_repair_preserves_draft_and_revalidates_every_claim(tmp_path,shape):
    raw={'phone_id':phone().id,'text':'已核实6核CPU、5核GPU，120Hz屏。配置判断可列入游戏初筛备选，持续表现仍需核实。'}
    if shape=='bad_semantics':raw['text']='已核实6核CPU，120Hz屏，因此原神稳定120fps。'
    envelope=raw if shape=='single_block' else {'blocks':raw if shape=='singleton_blocks' else [raw]}
    class Draft:
        available=True
        def complete_json(self,*a,**kw):return envelope
    state=answer_state([compute_source()],'一加13怎么样');state['session'].profile.priorities=['游戏']
    state['task']={'kind':'overview','fields':[],'topics':['游戏'],'games':[]}
    reply=AnswerComposer(assistant(tmp_path),state,Draft()).run()['reply']
    assert state['answer_draft']==envelope and state['answer_protocol_repairs']
    assert reply.blocks and any('6' in b.text and '配置判断' in b.text for b in reply.blocks)
    assert all('120fps' not in b.text for b in reply.blocks)
    assert all(set(b.evidence_ids)=={'spec'} for b in reply.blocks)
    if shape=='bad_semantics':assert state.get('answer_failure')


def test_explicit_wrong_citation_is_never_rebound_to_available_facts(tmp_path):
    class Draft:
        available=True
        def complete_json(self,*a,**kw):return {'blocks':[{'phone_id':phone().id,'text':'6核CPU。','evidence_ids':['unknown']}]}
    state=answer_state([compute_source()],'一加13怎么样');state['task']={'kind':'overview','fields':[],'topics':['游戏'],'games':[]}
    reply=AnswerComposer(assistant(tmp_path),state,Draft()).run()['reply']
    assert not state.get('answer_protocol_repairs') and state['answer_failure']['stage']=='citation_scope'
    assert reply.blocks and all('unknown' not in b.evidence_ids for b in reply.blocks)


def background_setup(tmp_path):
    p2=phone().model_copy(update={'id':'second','name':'vivo X200S','aliases':['vivoX200S'],'brand':'vivo'})
    background=source('vivo X200S在原神极高画质，跑图30分钟，实测平均60fps，机身温度43℃。',id='background',phone_id=p2.id,evidence_kind='review')
    state=answer_state([compute_source()],'一加13的游戏性能？');state['catalog'][p2.id]=p2
    state['background_sources']=[background];state['session'].profile.priorities=['游戏']
    c=AnswerComposer(assistant(tmp_path),state)
    return c,{s.id:s for s in c.registered_sources}


@pytest.mark.parametrize('text,ids,valid',[
    ('本机6核CPU、5核GPU，配置判断可作初筛，持续表现未知。背景：vivo X200S原神极高画质跑图30分钟，平均60fps，机身温度43℃，不能据此保证本机表现。',['background'],True),
    ('本机原神平均60fps。背景：vivo X200S原神极高画质跑图30分钟，平均60fps，仅作背景参考。',['background'],False),
    ('本机6核CPU。vivo X200S原神平均60fps。',['background'],False),
    ('本机6核CPU。背景：vivo X200S原神极高画质跑图30分钟，平均90fps，仅作背景参考。',['background'],False),
    ('本机6核CPU。背景：vivo X200S原神平均60fps，因此本机也60fps。',['background'],False),
    ('本机6核CPU。背景：vivo X200S原神平均60fps，仅作背景参考。',['unknown'],False),
])
def test_background_citations_preserve_their_object_and_cannot_supply_device_measurements(tmp_path,text,ids,valid):
    c,by_id=background_setup(tmp_path)
    block=AnswerBlock(phone_id=phone().id,text=text,evidence_ids=['spec'],background_ids=ids)
    assert c.validate_block(block,by_id) is valid


def test_other_phone_test_hidden_in_same_source_body_is_not_this_phone_measurement(tmp_path):
    s=source('一加13配置采用6核CPU。背景：vivo X200S原神极高画质，跑图30分钟，平均60fps，机身温度43℃。',evidence_kind='review')
    state=answer_state([s],'原神帧率发热？');state['session'].profile.priorities=['游戏']
    c=AnswerComposer(assistant(tmp_path),state)
    assert not c.validate_block(AnswerBlock(phone_id=phone().id,text='一加13原神平均60fps、温度43℃。',evidence_ids=['spec']),{'spec':s})


def test_agent_can_recommend_from_verified_configuration_without_forced_public_measurement(tmp_path):
    from langchain_core.messages import AIMessage
    data=[compute_source(),price()]
    kb=KnowledgeBase(KnowledgeIndex(chunks=[KnowledgeChunk(phone_name=phone().name,title=s.title,source_url=s.url,
        text=s.quote,chunk_id=s.id,document_id=s.id,checked_on=s.checked_on,evidence_kind=s.evidence_kind,
        market=s.market,reference_price=s.reference_price,price_variant=s.price_variant) for s in data]))
    class Model:
        available=True;last_usage={'total_tokens':10};decisions=0
        def complete_json(self,messages,**kw):
            return {'quote':messages[-1]['content'],'budget_max':3500,'priorities':['游戏'],'recommendation_count':1}
        def invoke_tools(self,*a,**kw):
            self.decisions+=1
            if self.decisions==1:return AIMessage(content='',tool_calls=[{'name':'search_phone_knowledge',
                'args':{'phone_name':'一加 13','query':'游戏配置价格'},'id':'local','type':'tool_call'}])
            return AIMessage(content=json.dumps({'kind':'answer','candidate_names':['一加13'],'blocks':[{
                'phone_id':phone().id,'text':'已核实6核CPU、5核GPU和120Hz屏。配置判断可作为游戏初筛备选，持续高负载仍受散热与适配影响。',
                'evidence_ids':[phone().id+':spec',phone().id+':https://8.8.8.8/price']}]}))
    p=phone();p.topics=['游戏']
    a=SelectionAssistant(ChatStore(tmp_path/'agent.db'),model_client=Model(),catalog={p.id:p},knowledge=kb,source_loader=Public(),agent_enabled=True)
    result=a.turn(ChatRequest(message='预算3500主要玩游戏，推荐一款',search_mode='auto'));reply=result.messages[-1]
    assert reply.cards and reply.cards[0].id==p.id and '配置判断' in reply.blocks[0].text
    assert reply.agent_trace['metrics']['search_calls']==reply.agent_trace['metrics']['fetch_calls']==0
    assert reply.agent_trace['stop_reason']=='model_finished'
