"""Offline control-flow regressions; no paid model or public search calls."""
import pytest

from app.chat import SelectionAssistant
from app.chat_intent import IntentResult, apply_model_patch, understand_intent
from app.chat_schemas import ChatMessage, ChatRequest, ChatSource, IntentPatch, SelectionProfile
from app.chat_store import ChatStore
from app.conversation_policy import route_turn
from app.knowledge import KnowledgeBase, KnowledgeIndex, KnowledgeChunk
from tests.test_chat_agent import Model, Public, done, tool
from tests.test_selection_budget import phone


@pytest.mark.parametrize('text,patch,focus,topics', [
    ('主要拍人像', {'requirements': ['拍人像']}, '人像', ['拍照', '续航']),
    ('经常拍远处的风景', {'requirements': ['拍远处的风景']}, '远处的风景', ['拍照', '续航']),
    ('每次连续玩两小时', {'requirements': ['连续玩两小时']}, '两小时', ['游戏', '续航']),
    ('主要玩绝区零', {'games': ['绝区零']}, '绝区零', ['游戏', '续航']),
])
def test_different_use_details_share_route_and_keep_saved_state(tmp_path, text, patch, focus, topics):
    current, other = phone('c', 5299), phone('d', 5799)
    current.topics = other.topics = topics
    source = ChatSource(id='s', phone_id=current.id, title='离线虚构参数',
        url='https://www.vivo.com.cn/spec', checked_on='2026-10-22', evidence_kind='spec',
        quote='天玑9400，6000mAh电池，80W快充，5000万像素主摄，3倍光学变焦。')
    store = ChatStore(tmp_path / 'conversation.db')
    session = store.create()
    session.profile = SelectionProfile(budget_max=6000, priorities=topics, excluded_brands=['华为'],
        candidate_ids=[current.id], requirements=['至少256GB存储'])
    session.messages = [ChatMessage(role='user', content='预算6000'),
        ChatMessage(role='assistant', content='你平时具体怎么使用？', cards=[current.card(session.profile, [source])], sources=[source])]
    session = store.save(session)
    model = Model([done()], {text: {**patch, 'turn_action': 'refine', 'response_focus': focus}})
    public = Public()
    assistant = SelectionAssistant(store, model_client=model, catalog={current.id:current,other.id:other},
        source_loader=public, knowledge=KnowledgeBase(KnowledgeIndex(chunks=[KnowledgeChunk(
            phone_name=current.name,title=source.title,source_url=source.url,text=source.quote,
            document_id='fixture',chunk_id='fixture',evidence_kind='spec')])) )
    result = assistant.turn(ChatRequest(message=text, session_id=session.session_id, search_mode='local'))
    reply = result.messages[-1]
    assert result.profile.candidate_ids == [current.id]  # The more expensive other phone doesn't replace it.
    assert result.profile.budget_max == 6000 and result.profile.priorities == topics
    assert result.profile.excluded_brands == ['华为'] and '至少256GB存储' in result.profile.requirements
    assert all(requirement in result.profile.requirements for requirement in patch.get('requirements', []))
    assert result.profile.games == patch.get('games', [])
    assert model.contexts[0]['本轮对话动作']['operation'] == 'refine'
    assert model.contexts[0]['候选价位覆盖不足'] is False
    body = reply.content + ''.join(block.text for block in reply.blocks)
    assert focus in body and '5299' not in body and '推荐这' not in body
    assert ('5000万' if '拍照' in topics else '天玑9400') in body
    assert not public.searches and not public.fetches
    trace = next(a for a in reply.agent_trace['actions'] if a['action'] == 'conversation_route')
    assert trace['operation'] == 'refine' and trace['focus'] == focus
    # This is a resumed SQLite session; old user/assistant text is not rewritten.
    reloaded = ChatStore(store.path).load(session.session_id)
    assert reloaded.messages[1].content == '你平时具体怎么使用？'
    assert reloaded.profile == result.profile


@pytest.mark.parametrize('changes,action,candidate_action,expected', [
    ({'budget_max':4000}, 'refine', 'retain', 'revise'),
    ({'excluded_brands':['华为']}, 'refine', 'retain', 'revise'),
    ({'priorities':['拍照','游戏','续航']}, 'refine', 'retain', 'revise'),
    ({'high_refresh_required':True}, 'refine', 'retain', 'revise'),
    ({'games':[]}, 'refine', 'retain', 'revise'),
    ({'requirements':[]}, 'refine', 'retain', 'revise'),
    ({}, 'auto', 'alternatives', 'alternatives'),
    ({}, 'recommend', 'retain', 'revise'),
    ({}, 'question', 'retain', 'question'),
    ({'requirements':['连续一小时','至少512GB']}, 'revise', 'retain', 'revise'),
])
def test_semantic_refine_cannot_override_constraint_revision(changes, action, candidate_action, expected):
    old = SelectionProfile(budget_max=6000, priorities=['游戏','续航'],games=['王者荣耀'],requirements=['连续一小时'])
    new = old.model_copy(update=changes)
    intent = IntentResult(bool(changes), [], False, turn_action=action,candidate_action=candidate_action)
    assert route_turn(old.model_dump(), new, intent, has_candidates=True, selected=[]).operation == expected


def test_known_conflict_requires_rechecking_even_for_additive_detail():
    old=SelectionProfile(budget_max=5000,priorities=['轻便'])
    new=old.model_copy(update={'requirements':['重量不超过180克']})
    intent=IntentResult(True,[],False,turn_action='refine')
    assert route_turn(old.model_dump(),new,intent,has_candidates=True,selected=[],incompatible=True).operation=='revise'


def test_focus_is_validated_before_any_profile_mutation():
    profile=SelectionProfile(budget_max=6000)
    with pytest.raises(ValueError,match='回答重点'):
        apply_model_patch(profile,IntentPatch(quote='拍人像',budget_max=5000,response_focus='夜景'),'拍人像',{})
    assert profile.budget_max==6000


def test_semantic_extractor_receives_the_actual_followup_question():
    class Capture(Model):
        def complete_json(self,messages,**kwargs):
            self.intent_messages=messages
            return super().complete_json(messages,**kwargs)
    model=Capture([])
    understand_intent(SelectionProfile(budget_max=5000,priorities=['拍照']), '人像', {}, model,
        [ChatMessage(role='assistant',content='先看这些候选。',followup='你更常拍人像、夜景还是远景？')])
    assert '你更常拍人像、夜景还是远景？' in model.intent_messages[1]['content']


@pytest.mark.parametrize('text,patch,expected', [
    ('预算降到4000', {'budget_max':4000}, 'a'),
    ('改成拍照优先', {'priorities':['拍照'], 'priority_action':'reorder'}, 'd'),
])
def test_actual_turn_reselects_for_core_changes_even_if_model_calls_them_refinements(tmp_path,text,patch,expected):
    phones={p.id:p for p in [phone('a',3200),phone('c',5299),phone('d',5799)]}
    phones['d'].topics=['拍照']
    chunks=[KnowledgeChunk(phone_name=p.name,title='离线参数夹具',source_url='https://www.vivo.com.cn/'+p.id,
        text='天玑9400，6000mAh电池，5000万像素主摄。',document_id=p.id,chunk_id=p.id,evidence_kind='spec') for p in phones.values()]
    model=Model([[tool('search_phone_knowledge',phone_name=phones[expected].name,query='天玑9400 电池 主摄')],done()],
                {text:{**patch,'turn_action':'refine'}})
    store=ChatStore(tmp_path/'selection.db')
    assistant=SelectionAssistant(store,model_client=model,catalog=phones,source_loader=Public(),
        knowledge=KnowledgeBase(KnowledgeIndex(chunks=chunks)))
    session=store.create()
    session.profile=SelectionProfile(budget_max=6000,priorities=['游戏','续航'],candidate_ids=['c'],recommendation_count=1)
    session.messages=[ChatMessage(role='user',content='预算6000，游戏和续航'),ChatMessage(role='assistant',content='先看当前候选。')]
    session=store.save(session)
    result=assistant.turn(ChatRequest(message=text,session_id=session.session_id,search_mode='local'))
    assert result.profile.candidate_ids==[expected]
    assert model.contexts[0]['本轮对话动作']['operation']=='revise'
    assert result.messages[0].content=='预算6000，游戏和续航'
