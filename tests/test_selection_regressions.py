"""Control-flow regressions; these do not measure real model accuracy."""
import pytest

from app.chat_intent import understand_intent, IntentResult
from app.chat_schemas import SelectionProfile, ChatMessage, ChatRequest, PriceOption
from app.conversation_policy import route_turn
from app.selection_catalog import select_variant
from tests.test_chat_intent import Model as PatchModel
from tests.test_conversation_choices import assistant_with_choices


@pytest.mark.parametrize('text,paraphrase', [
    ('一般一玩就是两小时', '连续游玩两小时'),
    ('我常拍的是家里的猫', '宠物摄影'),
])
def test_unfaithful_free_text_retains_original_without_losing_valid_patch(text, paraphrase):
    profile = SelectionProfile(budget_max=6000, priorities=['游戏','续航'])
    result = understand_intent(profile, text, {}, PatchModel(dict(quote=text,
        turn_action='refine', response_focus=text, requirements=[paraphrase])))
    assert profile.requirements == [text]
    assert not result.model_failed
    assert result.grounding_repairs == ['requirements_grounded_by_property_and_original_span']
    assert profile.priorities == ['游戏','续航'] and profile.budget_max == 6000


@pytest.mark.parametrize('requirement,expected', [
    ('机身必须不超过180克','revise'), ('至少512GB存储','revise'),
    ('必须支持无线充电','revise'), ('屏幕不低于120Hz','revise'),
    ('一般一玩就是两小时','refine'), ('主要拍孩子的人像','refine'),
])
def test_gate_overrides_semantic_refinement_without_phone_or_game_branches(requirement, expected):
    old = SelectionProfile(budget_max=5000,priorities=['拍照','续航'])
    current = old.model_copy(update={'requirements':[requirement]})
    intent = IntentResult(True, [], False, turn_action='refine')
    assert route_turn(old.model_dump(),current,intent,has_candidates=True,selected=[]).operation == expected


@pytest.mark.parametrize('preference,price', [('save_money',4200),('within_budget',4200),('near_budget',5500)])
def test_version_price_preference_does_not_force_storage_upgrades(preference, price):
    from tests.test_selection_budget import phone
    p = phone('c',4200)
    p.price_options = [PriceOption(variant='12GB+256GB',price=4200),PriceOption(variant='16GB+512GB',price=5500)]
    select_variant(p, SelectionProfile(budget_max=6000,price_preference=preference))
    assert p.price == price


def test_alternatives_only_temporarily_avoid_current_batch(tmp_path):
    text='换一批'
    assistant,_ = assistant_with_choices(tmp_path,text,{'candidate_action':'alternatives'},
        ['vivo X100','vivo X200','vivo X300'])
    session=assistant.store.create()
    session.profile=SelectionProfile(budget_max=5000,priorities=['拍照','续航'],candidate_ids=['fixture-1'])
    session.messages=[ChatMessage(role='assistant',content='第一批',cards=[assistant.catalog['fixture-0'].card(session.profile)]),
        ChatMessage(role='assistant',content='第二批',cards=[assistant.catalog['fixture-1'].card(session.profile)])]
    state=assistant.understand(dict(session=session,message=text,search_mode='local'))
    assert state['alternative_ids']==['fixture-1']
    assert not session.profile.excluded_phones


def test_question_keeps_other_candidates_and_focus_even_if_model_reselects(tmp_path):
    text='第二款的拍照怎么样？'
    names=['vivo X100','vivo X200','vivo X300']
    assistant,model=assistant_with_choices(tmp_path,text,{'turn_action':'question'},names)
    session=assistant.store.create()
    session.profile=SelectionProfile(budget_max=5000,priorities=['拍照','续航'],candidate_ids=list(assistant.catalog))
    session.messages=[ChatMessage(role='assistant',content='当前候选',
        cards=[p.card(session.profile) for p in assistant.catalog.values()])]
    session=assistant.store.save(session)
    result=assistant.turn(ChatRequest(message=text,session_id=session.session_id,search_mode='local'))
    assert result.profile.candidate_ids==list(assistant.catalog)
    assert result.profile.focus_phone_id=='fixture-1'
    assert all(b.phone_id=='fixture-1' for b in result.messages[-1].blocks)
    assert result.messages[-1].blocks


def test_discovery_diversifies_before_document_cutoff():
    from app.knowledge import KnowledgeBase, KnowledgeIndex, KnowledgeChunk
    chunks=[KnowledgeChunk(phone_name='Phone A',title='Fixture',source_url='https://example.com/a',
        text='游戏 芯片 电池 续航 游戏 芯片 电池 续航 '+str(index),
        document_id='a',chunk_id=str(index)) for index in range(80)]
    chunks.append(KnowledgeChunk(phone_name='Phone B',title='Fixture',source_url='https://example.com/b',
        text='游戏配置',document_id='b',chunk_id='b'))
    results=KnowledgeBase(KnowledgeIndex(chunks=chunks)).discover('游戏 芯片 电池 续航')
    assert [item['phone_name'] for item in results[:2]]==['Phone A','Phone B']


@pytest.mark.parametrize('name', ['vivo Y200','OPPO A5 Pro','realme C67','vivo T3'])
def test_numeric_series_names_remain_full_models(name):
    from app.phone_names import parse_phone_names, normalized_name
    found=parse_phone_names(name)
    assert len(found)==1 and normalized_name(found[0].name)==normalized_name(name)


def test_chip_extraction_does_not_take_prefix_of_hardware_model_number():
    from tests.test_selection_budget import phone
    from app.chat_schemas import ChatSource
    from app.selection_presentation import configuration_facts
    p=phone('a',3000)
    source=ChatSource(id='chip',phone_id=p.id,title='Fixture',url='https://example.com/spec',
        quote='A3288型号支持5G，A18芯片，6核中央处理器。',checked_on='2026-10-22',evidence_kind='spec')
    facts=configuration_facts(p,['游戏'],[source])
    assert ('A18','chip') in facts and ('6核中央处理器','chip') in facts
    assert not any('A3288' in text for text,sid in facts)


def test_screen_facts_do_not_promote_touch_sampling_to_refresh_rate():
    from tests.test_selection_budget import phone
    from app.chat_schemas import ChatSource
    from app.selection_presentation import configuration_facts
    p=phone('a',3000)
    source=ChatSource(id='screen',phone_id=p.id,title='Fixture',url='https://example.com/spec',
        quote='触控采样率240Hz。屏幕刷新率120Hz。',checked_on='2026-10-22',evidence_kind='spec')
    assert configuration_facts(p,['屏幕'],[source])==[('120Hz','screen')]


def test_repeated_local_results_do_not_reset_no_progress(tmp_path):
    import json
    from langchain_core.messages import AIMessage
    from tests.test_chat_agent import Model, tool
    assistant,_=assistant_with_choices(tmp_path,'预算5000，拍照和续航',
        {},['vivo X100','vivo X200','vivo X300'])
    assistant.model=Model([[tool('search_local_phones',query=query,limit=3)] for query in ['拍照','相机','镜头','人像']],
        {'预算5000，拍照和续航':{'budget_max':5000,'priorities':['拍照','续航']}})
    assistant.agent_policy=assistant.agent_policy.model_copy(update={'max_total_tokens':64000,'max_context_characters':48000})
    result=assistant.turn(ChatRequest(message='预算5000，拍照和续航',search_mode='local'))
    tools=[a['result'] for a in result.messages[-1].agent_trace['actions']if a.get('name')=='search_local_phones']
    assert tools[0]['added_evidence']>0
    assert tools[1]['sources'] and tools[1]['added_evidence']==0 and tools[1]['added_candidates']==0


def test_followup_suggestion_uses_current_focus_and_remembered_game(tmp_path):
    from app.chat_schemas import ChatSession
    assistant,_=assistant_with_choices(tmp_path,'',{},['vivo X100','vivo X200','vivo X300'])
    profile=SelectionProfile(budget_max=5000,priorities=['游戏'],games=['绝区零'],
        candidate_ids=list(assistant.catalog),focus_phone_id='fixture-1')
    state=dict(session=ChatSession(session_id='a'*32,profile=profile),message='刷新率是多少？',
        sources=[],catalog=assistant.catalog,selected=['fixture-1'],acknowledged=[],recommendation_turn=False)
    assistant.model.available=False
    reply=assistant.answer(state)['reply']
    assert reply.suggestions[0]=='vivo X200玩绝区零的帧率和发热怎么样？'
    assert '原神' not in ''.join(reply.suggestions)


@pytest.mark.parametrize('preference,requirements,expected', [
    ('within_budget',[],4200),('save_money',[],4200),('near_budget',[],5500),
    ('within_budget',['至少512GB存储'],5500),
])
def test_separate_observed_price_records_use_same_version_policy(tmp_path,preference,requirements,expected):
    from app.chat_schemas import ChatSession,ChatSource
    assistant,_=assistant_with_choices(tmp_path,'',{},['vivo X100','vivo X200','vivo X300'])
    p=assistant.catalog['fixture-0'].model_copy(deep=True)
    session=ChatSession(session_id='a'*32,profile=SelectionProfile(budget_max=6000,
        price_preference=preference,requirements=requirements),discovered_phones=[p])
    sources=[ChatSource(id='price-'+str(price),phone_id=p.id,title='Fixture',url='https://www.vivo.com.cn/price',
        quote=f'{variant} {price}元',checked_on='2026-10-22',evidence_kind='price',reference_price=price,
        price_variant=variant,price_options=[PriceOption(variant=variant,price=price)])
        for variant,price in [('16GB+512GB',5500),('12GB+256GB',4200)]]
    assistant._remember_sources({'session':session,'catalog':assistant.catalog},sources)
    assert session.discovered_phones[0].price==expected


def test_special_version_page_is_not_imported_as_basic_model():
    from app.knowledge_ingest import source_documents
    from app.chat_schemas import ChatSource
    from app.retrieval import variant_only_title
    from tests.test_selection_budget import phone
    p=phone('a',3000)
    assert variant_only_title(p.name,p.name+'卫星通信版 参数')
    source=ChatSource(id='special',phone_id=p.id,title=p.name+'卫星通信版 参数',
        url='https://www.vivo.com.cn/satellite/specs',quote='6000mAh电池，支持120Hz屏幕刷新率。',
        checked_on='2026-10-22',source_type='page',evidence_kind='spec')
    assert source_documents([source],{p.id:p})==[]
