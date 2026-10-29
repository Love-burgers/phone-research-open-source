"""Synthetic offline regressions; no phone facts or live model quality claims."""
import pytest

from app.chat import SelectionAssistant
from app.chat_agent import ChatAgentRun, ChatDecision
from app.chat_schemas import ChatMessage, ChatSource, SelectionProfile
from app.chat_store import ChatStore
from app.knowledge import KnowledgeBase, KnowledgeIndex
from app.selection_requirements import recommendation_gaps, check_requirement, literal_requirements
from app.chat_schemas import IntentPatch
from app.chat_intent import apply_model_patch
from app.source_retention import retained_sources
from tests.test_chat_agent import Model, Public


def assistant(tmp_path, patches=None):
    return SelectionAssistant(ChatStore(tmp_path/'sessions.sqlite3'), model_client=Model([], patches),
        source_loader=Public())


def source(pid, sid='spec', quote='骁龙8至尊版，屏幕刷新率120Hz，采用曲面屏，不支持无线充电。', **kw):
    return ChatSource(id=sid, phone_id=pid, title='虚构回归资料', url='https://example.com/'+sid,
        quote=quote, checked_on='2026-10-24', source_type='page', evidence_kind=kw.pop('evidence_kind','spec'), **kw)


def turn(a, s, text):
    state={'session':s, 'message':text, 'search_mode':'local'}
    state.update(a.understand(state))
    return state


def test_detail_focus_does_not_pin_then_recommend_without_redundant_flag(tmp_path):
    a=assistant(tmp_path)
    pid=next(iter(a.catalog)); name=a.catalog[pid].name
    detail=name+'的屏幕怎么样？'; recommend='预算改为6000，给我推荐三款游戏手机'
    a.model.patches={detail:{'selected_phones':[name],'turn_action':'question'},
        recommend:{'budget_max':6000,'recommendation_count':3,'turn_action':'recommend'}}
    s=a.store.create(); s.profile=SelectionProfile(budget_max=5000,priorities=['游戏'],candidate_ids=[pid])
    state=turn(a,s,detail)
    assert state['selected']==[pid] and not s.profile.explicit_candidate_ids
    s=a.store.load(a.store.save(s).session_id)
    state=turn(a,s,recommend)
    assert state['recommendation_turn'] and s.profile.focus_phone_id is None
    assert len(state['selected'])==3 and not s.profile.explicit_candidate_ids


@pytest.mark.parametrize('requirement', ['不要曲面屏','需要无线充电','必须支持无线充电'])
def test_contradicted_non_numeric_gate_cannot_be_selected(tmp_path, requirement):
    a=assistant(tmp_path); pid=next(iter(a.catalog)); phone=a.catalog[pid]
    phone.price=3999
    s=a.store.create(); s.profile=SelectionProfile(budget_max=4500,priorities=['游戏'],requirements=[requirement],candidate_ids=[pid])
    state={'session':s,'message':'推荐游戏手机','catalog':a.catalog,'selected':[pid],'recommendation_turn':True,'search_mode':'local'}
    run=ChatAgentRun(a,state); evidence=[source(pid)]
    assert recommendation_gaps(phone,s.profile,evidence)
    run.select_candidates(ChatDecision(kind='answer',candidate_names=[phone.name]),evidence)
    assert not state['selected']


def test_full_body_registry_accepts_new_price_and_keeps_old_citations(tmp_path):
    a=assistant(tmp_path); pid=next(iter(a.catalog)); s=a.store.create()
    s.profile=SelectionProfile(budget_max=4500,priorities=['游戏'],candidate_ids=[pid])
    old=[source(pid,'old-'+str(i)) for i in range(8)]
    s.messages=[ChatMessage(role='assistant',content='历史回答',sources=old)]
    run=ChatAgentRun(a,{'session':s,'message':'核对价格','catalog':a.catalog,'selected':[pid],'recommendation_turn':True})
    state={'sources':{x.id:x for x in old},'allowed_urls':{(pid,x.url) for x in old},'actions':[]}
    price=source(pid,'new-price','虚构价格：12GB+256GB 售价3999元',evidence_kind='price',
        reference_price=3999,price_variant='12GB+256GB')
    returned,added=run.record_sources(state,pid,[price])
    assert added==1 and returned[0]['id']=='new-price'
    assert len(state['sources'])==8 and 'new-price' in state['sources']
    assert (pid,price.url) in state['allowed_urls']
    restored=a.store.load(a.store.save(s).session_id)
    assert restored.messages[0].sources==old


@pytest.mark.parametrize('requirement,quote,status', [
    ('不要曲面屏','采用OLED直屏','satisfied'),
    ('不要曲面屏','不是曲面屏','satisfied'),
    ('不要曲面屏','不采用曲面屏','satisfied'),
    ('只接受平面屏','采用微曲屏','unsatisfied'),
    ('需要无线快充','无线充电：支持','unknown'),
    ('需要无线充电','支持无线快充','satisfied'),
    ('需要无线充电','支持无线充电，不支持无线快充','satisfied'),
    ('必须支持无线充电','支持50W无线充电','satisfied'),
    ('需要无线充电','不支持50W无线充电','unknown'),
    ('需要50W无线充电','支持30W无线充电','unknown'),
    ('需要无线充电','无线充电功率50W','satisfied'),
    ('需要NFC','支持NFC功能','satisfied'),
    ('不要曲面屏','曲面屏：否','satisfied'),
    ('需要无线充电','无线充电：不支持','unsatisfied'),
    ('需要无线充电','可能支持无线充电','unknown'),
    ('需要无线充电','不是不支持无线充电','unknown'),
    ('需要无线充电','支持反向无线充电','unknown'),
    ('必须卫星通信','支持NFC','unknown'),
    ('需要无线充电和卫星通信','支持无线充电','unknown'),
    ('不要曲面屏且重量不超过200g','采用直屏，重量190g','satisfied'),
    ('需要无线充电','支持无线充电；不支持无线充电','conflict'),
])
def test_literal_gate_states(requirement,quote,status):
    assert check_requirement(requirement,[source('p',quote=quote)]).status==status


def test_conflict_unknown_and_citation_ownership():
    positive=source('p','yes','支持无线充电')
    negative=source('p','no','不支持无线充电')
    result=check_requirement('需要无线充电',[positive,negative])
    assert result.status=='conflict' and set(result.source_ids)=={'yes','no'}
    assert check_requirement('需要无线充电',[positive.model_copy(update={'source_type':'snippet'})]).status=='unknown'
    numeric=check_requirement('重量不超过200g',[source('p','light','重量190g'),source('p','heavy','重量210g')])
    assert numeric.status=='conflict'


def test_feature_requirement_add_modify_cancel_and_restore(tmp_path):
    a=assistant(tmp_path); s=a.store.create()
    literal_requirements(s.profile,'不要曲面屏，需要无线充电')
    assert len(s.profile.requirements)==2
    literal_requirements(s.profile,'改为需要曲面屏')
    assert '不要曲面屏' not in s.profile.requirements and len(s.profile.requirements)==2
    s=a.store.load(a.store.save(s).session_id)
    literal_requirements(s.profile,'取消无线充电要求')
    assert s.profile.requirements==['改为需要曲面屏']
    text='不要曲面屏'
    apply_model_patch(s.profile,IntentPatch(quote=text,requirements=[text]),text,a.catalog)
    assert s.profile.requirements==[text]
    apply_model_patch(s.profile,IntentPatch(quote='取消这个限制',remove_requirements=[text]),'取消这个限制',a.catalog)
    assert not s.profile.requirements


def test_same_turn_contradictory_demands_need_clarification(tmp_path):
    a=assistant(tmp_path); profile=SelectionProfile(budget_max=5000)
    literal_requirements(profile,'不要曲面屏，需要曲面屏')
    gaps=recommendation_gaps(next(iter(a.catalog.values())),profile,[source(next(iter(a.catalog)))])
    assert any('用户硬要求冲突' in gap for gap in gaps)


@pytest.mark.parametrize('price_preference',['within_budget','near_budget'])
def test_real_pinned_comparison_survives_budget_revision_but_recommend_releases(tmp_path,price_preference):
    a=assistant(tmp_path); ids=list(a.catalog)[:2]; names=[a.catalog[pid].name for pid in ids]
    compare='只比较'+names[0]+'和'+names[1]
    budget='预算改为1000'; recommend='按新预算推荐三款'
    a.model.patches={compare:{'selected_phones':names,'turn_action':'question'},
        budget:{'budget_max':1000,'price_preference':price_preference,'turn_action':'revise'},
        recommend:{'turn_action':'recommend','recommendation_count':3}}
    s=a.store.create(); s.profile=SelectionProfile(budget_max=6000,priorities=['游戏'])
    turn(a,s,compare)
    assert s.profile.explicit_candidate_ids==ids
    state=turn(a,s,budget)
    assert state['selected']==ids and not state['recommendation_turn']
    state=turn(a,s,recommend)
    assert not s.profile.explicit_candidate_ids and state['recommendation_turn']


def test_recommendation_with_explicit_references_stays_within_comparison_scope(tmp_path):
    a=assistant(tmp_path); ids=list(a.catalog)[:2]
    text='从这两款中推荐一款'
    a.model.patches={text:{'turn_action':'recommend','recommendation_count':1}}
    s=a.store.create(); s.profile=SelectionProfile(budget_max=6000,priorities=['游戏'],
        candidate_ids=ids,explicit_candidate_ids=ids)
    state=turn(a,s,text)
    assert s.profile.explicit_candidate_ids==ids and state['selected']==ids
    assert not state['recommendation_turn']


def test_source_retention_keeps_price_gate_opposition_and_usage(tmp_path):
    a=assistant(tmp_path); phone=next(iter(a.catalog.values()))
    profile=SelectionProfile(budget_max=4500,priorities=['游戏'],requirements=['需要无线充电'])
    evidence=[source(phone.id,'filler-'+str(i),'与本轮无关的说明') for i in range(8)]
    important=[source(phone.id,'price','12GB+256GB售价3999元',evidence_kind='price',reference_price=3999,price_variant='12GB+256GB'),
        source(phone.id,'gate-yes','支持无线充电'),source(phone.id,'gate-no','不支持无线充电'),
        source(phone.id,'review','原神最高画质测试30分钟平均帧率59fps。',evidence_kind='review')]
    kept=retained_sources([*evidence,*important],phone,profile)
    assert len(kept)==8 and {s.id for s in important}<={s.id for s in kept}
    assert check_requirement('需要无线充电',kept).status=='conflict'


def test_source_refresh_keeps_id_and_older_body_cannot_replace_newer(tmp_path):
    a=assistant(tmp_path); pid=next(iter(a.catalog)); s=a.store.create()
    run=ChatAgentRun(a,{'session':s,'message':'核对屏幕','catalog':a.catalog,'selected':[pid]})
    original=source(pid,'stable','采用曲面屏')
    state={'sources':{'stable':original},'allowed_urls':set(),'actions':[]}
    updated=source(pid,'incoming','采用直屏').model_copy(update={'url':original.url})
    returned,added=run.record_sources(state,pid,[updated])
    assert added==1 and returned[0]['id']=='stable' and state['sources']['stable'].quote=='采用直屏'
    older=original.model_copy(update={'id':'old','checked_on':'2025-01-01'})
    assert run.record_sources(state,pid,[older])[1]==0
    assert state['sources']['stable'].quote=='采用直屏'
    collision=source(pid,'stable','采用曲面屏').model_copy(update={'url':'https://example.com/different'})
    assert run.record_sources(state,pid,[collision])[1]==0
    assert state['sources']['stable'].url==original.url


def test_two_detail_targets_do_not_erase_the_recommendation_list(tmp_path):
    a=assistant(tmp_path); ids=list(a.catalog)[:3]; names=[a.catalog[pid].name for pid in ids[:2]]
    text='、'.join(names)+'的屏幕各有什么特点？'
    a.model.patches={text:{'selected_phones':names,'turn_action':'question'}}
    s=a.store.create(); s.profile=SelectionProfile(budget_max=6000,priorities=['游戏'],candidate_ids=ids)
    state=turn(a,s,text)
    assert state['selected']==ids[:2] and s.profile.candidate_ids==ids
    assert not s.profile.explicit_candidate_ids


def test_compound_gate_is_not_silently_erased_by_a_feature_update():
    p=SelectionProfile(requirements=['不要曲面屏且重量不超过200g'])
    literal_requirements(p,'需要曲面屏')
    assert p.requirements==['重量不超过200g','需要曲面屏']
    literal_requirements(p,'取消曲面屏要求')
    assert p.requirements==['重量不超过200g']


def test_restore_reuses_opposing_gate_evidence_without_turning_conflict_into_match(tmp_path):
    a=assistant(tmp_path); pid=next(iter(a.catalog)); s=a.store.create()
    s.profile=SelectionProfile(budget_max=4500,requirements=['需要无线充电'],candidate_ids=[pid])
    evidence=[source(pid,'filler'+str(i),'无关资料') for i in range(8)]
    evidence+=[source(pid,'yes','支持无线充电'),source(pid,'no','不支持无线充电')]
    s.messages=[ChatMessage(role='assistant',content='历史原文',sources=evidence)]
    s=a.store.load(a.store.save(s).session_id)
    run=ChatAgentRun(a,{'session':s,'message':'这款支持吗？','catalog':a.catalog,'selected':[pid],
        'recommendation_turn':False,'search_mode':'local'})
    initialized=run.initialize({})
    assert len(initialized['sources'])==8
    assert check_requirement('需要无线充电',list(initialized['sources'].values())).status=='conflict'
