"""Mechanism regressions; synthetic facts, never real purchasing advice."""
import json
import socket

import pytest

from app.chat import SelectionAssistant
from app.chat_intent import understand_intent
from app.chat_schemas import ChatMessage, ChatRequest, ChatSession, ChatSource, SelectionProfile
from app.chat_store import ChatStore
from app.selection_catalog import CatalogPhone, shortlist
from app.selection_requirements import card_matches, missing_evidence, literal_requirements
from app.phone_sources import PublicPhoneSources
from app.phone_sources import extract_phone_excerpt
from tests.test_chat_agent import Model, Public, tool, done


def test_successful_patch_not_reparsed_and_context_is_passed(tmp_path):
    text = "高刷不用强求，原神不玩了；另外那个重量要求也取消"
    model = Model([], {text: {"high_refresh_required": False, "remove_games": ["原神"],
                                  "remove_requirements": ["重量不超过190克"]}})
    captured = []
    original = model.complete_json
    model.complete_json = lambda messages: captured.append(messages) or original(messages)
    session = ChatSession(session_id="a"*32, profile=SelectionProfile(budget_max=4500,
        high_refresh_required=True, games=["原神", "绝区零"], priorities=["游戏"],
        requirements=["重量不超过190克", "至少512GB"]), messages=[ChatMessage(role="user", content="重量不超过190克")])
    assistant = SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'), model_client=model)
    assistant.understand({"session": session, "message": text})
    assert not session.profile.high_refresh_required
    assert session.profile.games == ["绝区零"] and session.profile.requirements == ["至少512GB"]
    assert session.profile.budget_max == 4500
    assert "重量不超过190克" in captured[0][1]["content"] and captured[0][-1]["content"] == text


def test_noop_with_requirement_words_does_not_become_demand(tmp_path):
    text = "高刷和原神是怎么回事？"
    assistant = SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'), model_client=Model([], {text: {}}))
    session = ChatSession(session_id="a"*32, profile=SelectionProfile(budget_max=3000, priorities=["续航"]))
    assistant.understand({"session": session, "message": text})
    assert not session.profile.high_refresh_required and not session.profile.games
    assert session.profile.priorities == ["续航"]


@pytest.mark.parametrize('initial,patch,expected', [
    ({"games":["原神"]}, {"games":["绝区零"]}, {"games":["原神","绝区零"]}),
    ({"games":["原神"],"priorities":["游戏","拍照"]}, {"remove_priorities":["游戏"]}, {"games":[],"priorities":["拍照"]}),
    ({"price_preference":"near_budget"}, {"budget_max":2800}, {"price_preference":"within_budget","budget_max":2800}),
    ({"requirements":["至少512GB"]}, {"remove_requirements":["至少512GB"],"requirements":["256GB也行"]}, {"requirements":["256GB也行"]}),
])
def test_atomic_incremental_operations(initial, patch, expected):
    profile = SelectionProfile(**initial)
    text = "绝区零，256GB也行，预算2800，取消游戏用途"
    model = Model([], {text: patch})
    result = understand_intent(profile, text, {}, model)
    assert not result.model_failed
    for key, value in expected.items():
        assert getattr(profile,key) == value


def test_cancel_games_scoped_offline():
    p = SelectionProfile(games=["原神","崩坏：星穹铁道"], high_refresh_required=True)
    literal_requirements(p,"不玩原神了，但继续玩星铁，高刷不强求")
    assert p.games == ["崩坏：星穹铁道"] and not p.high_refresh_required


def phone(pid, price, topics, offers=()):
    return CatalogPhone(id=pid, name=pid, aliases=[pid], brand="vivo", price=price,
        price_options=list(offers), topics=topics, variant="测试", checked_on="2026-10-22", note="虚构夹具")


def test_budget_increase_reranks_and_decrease_changes_saved_variant(tmp_path):
    catalog={"a": phone("a",2000,["拍照"]),"b":phone("b",4200,["拍照"],
        [{"variant":"256GB","price":3200},{"variant":"512GB","price":4200}])}
    model=Model([], {"预算4500":{"budget_max":4500},"预算3300":{"budget_max":3300}})
    assistant=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=model,catalog=catalog)
    s=ChatSession(session_id="a"*32,profile=SelectionProfile(budget_max=2200,priorities=["拍照"],candidate_ids=["a"]))
    state=assistant.understand({"session":s,"message":"预算4500"})
    assert s.profile.candidate_ids[0] == "b"
    s.profile.price_preference="near_budget"
    s.profile.budget_max=4500
    assert shortlist(catalog,s.profile)==["b"] and catalog["b"].price==4200
    state=assistant.understand({"session":s,"message":"预算3300"})
    assert s.profile.candidate_ids[0]=="b" and state["catalog"]["b"].price==3200
    assert state["catalog"]["b"].variant=="256GB"


def source(quote,kind='spec',sid='test'):
    return ChatSource(id=sid,phone_id='a',title='虚构资料',url='https://example.com/'+sid,
        quote=quote,evidence_kind=kind,checked_on='2026-10-22',source_type='page')


def test_mixed_use_cards_distinguish_parameters_reviews_absence_and_conflict():
    p=SelectionProfile(priorities=["拍照","续航","轻便"],high_refresh_required=True)
    specs=source('电池6000mAh；机身重量185g；屏幕刷新率120Hz')
    labels,_=card_matches('a',p,[specs])
    assert '拍照：待补实测' in labels and '续航：待补实测' in labels
    assert any('185g' in l and '配置标注' in l for l in labels)
    reviews=source('夜景实拍样张暗部细节保留；续航实测连续视频12小时，剩余电量20%', 'review','review')
    labels,_=card_matches('a',p,[specs,reviews])
    assert any('拍照：实测资料' in l for l in labels) and any('续航：实测资料' in l for l in labels)
    conflict=source('屏幕刷新率60Hz','spec','other')
    assert any('来源冲突' in l for l in card_matches('a',p,[specs,conflict])[0])
    assert missing_evidence(p,['a'],[specs,reviews,conflict])==['a']


def test_game_mention_does_not_borrow_other_game_measurement():
    p=SelectionProfile(games=['原神','绝区零'])
    review=source('原神实测平均59fps，30分钟。绝区零尚未测试。','review')
    labels,_=card_matches('a',p,[review])
    assert '原神：有测试记录' in labels and '绝区零：待查' in labels


def test_source_index_cannot_displace_public_results(tmp_path,monkeypatch):
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443))])
    from app.selection_catalog import load_catalog
    p=load_catalog()['iphone-16']
    loader=PublicPhoneSources(tmp_path/'cache',searcher=lambda *a:[
        {'title':p.name+' 新来源','snippet':p.name+' 正文待核对','url':f'https://example.com/new{i}'} for i in range(4)])
    found=loader.search(p,p.name+' 拍照')
    assert all('/new' in c['url'] for c in found['candidates'][:4])


@pytest.mark.parametrize('fail',[False,True])
def test_agent_photo_gap_searches_and_preserves_empty_or_error(tmp_path,fail):
    text='预算3000，主要夜景拍照'
    model=Model([done(),[tool('web_search',phone_name='a',query='a 夜景 实拍')],done()],
                {text:{'budget_max':3000,'priorities':['拍照']}})
    class Loader(Public):
        def search(self,*a,**k):
            if fail: raise TimeoutError('fixture')
            return []
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=model,
        catalog={'a':phone('a',2800,['拍照'])},source_loader=Loader())
    result=service.turn(ChatRequest(message=text))
    trace=result.messages[-1].agent_trace
    assert any(a.get('validation_feedback') for a in trace['actions'])
    assert any(a.get('result',{}).get('status')==('tool_error' if fail else 'empty') for a in trace['actions'])
    assert result.profile.budget_max==3000
    assert not result.messages[-1].cards


def test_new_session_isolated_and_resume_preserves_requirements(tmp_path):
    store=ChatStore(tmp_path/'s.sqlite3')
    s=store.create();s.profile=SelectionProfile(budget_max=2600,priorities=['续航'],requirements=['至少512GB'])
    store.save(s)
    reopened=ChatStore(store.path).load(s.session_id)
    assert reopened.profile.requirements==['至少512GB']
    assert store.create().profile.requirements==[]


def test_local_only_keeps_available_local_models_even_when_tier_coverage_is_weak(tmp_path):
    text='预算2500，续航优先，只用本地资料'
    model=Model([done('acknowledge','保留条件')],{text:{'budget_max':2500,'priorities':['续航']}})
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=model,
        catalog={'a':phone('a',1600,['屏幕'])},source_loader=Public())
    service.turn(ChatRequest(message=text))
    assert model.contexts[0]['可查询机型']=={'a':'a'}


def test_pronoun_cannot_resurrect_old_rejected_card(tmp_path):
    from app.chat_schemas import PhoneCard
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=Model([]),
        catalog={'a':phone('a',2800,['拍照'])})
    s=ChatSession(session_id='a'*32,profile=SelectionProfile(budget_max=5000,priorities=['拍照']),
        messages=[ChatMessage(role='assistant',content='旧候选',cards=[phone('a',2800,['拍照']).card(SelectionProfile())]),
                  ChatMessage(role='assistant',content='没有找到当前合适候选')])
    state=service.understand({'session':s,'message':'它的夜景呢？'})
    assert state['clarification'] and not s.profile.focus_phone_id


def test_anonymous_tv_promotions_are_not_phone_specifications():
    html='<title>Xiaomi 17T</title><main><h1>Xiaomi 17T</h1><p>性价比爆款-REDMI A 2025</p><p>REDMI 显示器 G34WQ</p></main>'
    assert extract_phone_excerpt('Xiaomi 17T',html,'屏幕 拍照 续航') is None


def test_official_price_with_parenthesized_version_and_photo_review_kind():
    from app.phone_sources import extract_price_options, page_kind
    html='<title>vivo X300 专业影像旗舰</title><p>全网通：vivo X300（12GB+256GB）：4599.00元 vivo X300（16GB+256GB）：4999.00元</p>'
    offers,_=extract_price_options('vivo X300',html)
    assert [(o.variant,o.price) for o in offers]==[('12GB+256GB',4599),('16GB+256GB',4999)]
    assert page_kind('测试手机影像评测','夜景实拍样张：人物肤色偏暖，暗部细节保留')=='review'
    assert page_kind('测试手机续航评测','续航实测连续视频12小时，耗电80%')=='review'


def test_hypothesis_cannot_become_recommendation_without_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443))])
    text='预算4500，拍照优先，排除华为'
    model=Model([[tool('consider_phones',names=['vivo X300','华为 Pura80'])],
                 [tool('web_search',phone_name='vivo X300',query='vivo X300 官方价格')],
                 {'kind':'answer','candidate_names':['vivo X300']}],
                {text:{'budget_max':4500,'priorities':['拍照'],'excluded_brands':['华为']}})
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=model,
        source_loader=PublicPhoneSources(tmp_path/'cache',searcher=lambda *a:[]))
    s=service.turn(ChatRequest(message=text))
    assert [p.name for p in s.discovered_phones]==['vivo X300']
    assert not s.messages[-1].cards and not s.profile.candidate_ids


def test_related_article_links_and_measurement_headings_are_not_test_results():
    html='<title>iQOO 15评测</title><main><h1>iQOO 15</h1><div class="related"><p><a href="/a">iQOO 15原神帧率实测：60帧稳了！</a></p></div><p><a href="/b">iQOO 15散热实测：原神40℃，王者2小时不烫手+FAQ</a></p><h2>iQOO 15 原神60帧实测</h2></main>'
    assert extract_phone_excerpt('iQOO 15',html,'原神 实测') is None
    p=SelectionProfile(games=['原神'])
    labels,_=card_matches('a',p,[source('原神帧率实测：60帧稳了！','review')])
    assert '原神：待查' in labels
    labels,_=card_matches('a',SelectionProfile(priorities=['拍照']),[source('相机实测：5000万哈苏三摄，3倍光学变焦全焦段覆盖','review')])
    assert '拍照：待补实测' in labels


def test_storage_requirement_changes_version_and_filters_insufficient_options():
    p=SelectionProfile(budget_max=4000,priorities=['续航'],requirements=['至少512GB'])
    catalog={'a':phone('a',2900,['续航'],[{'variant':'12GB+256GB','price':2900},{'variant':'12GB+512GB','price':3600}])}
    assert shortlist(catalog,p)==['a'] and catalog['a'].variant=='12GB+512GB'
    p.budget_max=3300
    assert shortlist(catalog,p)==[]


@pytest.mark.parametrize('raw,brand',[('红魔11 Pro','红魔'),('ROG Phone 9 Pro','华硕'),('魅族22','魅族'),('索尼 Xperia 1','索尼')])
def test_additional_brands_are_parsed_as_hypotheses_not_preloaded_answers(raw,brand):
    from app.phone_names import parse_phone_names,register_phone
    mentions=parse_phone_names(raw)
    assert len(mentions)==1 and mentions[0].brand==brand
    registered=register_phone(mentions[0],{})
    assert registered.price is None and not registered.source_urls


def test_brand_followed_by_refresh_measurement_is_not_a_model():
    from app.phone_names import parse_phone_names
    assert not parse_phone_names('荣耀144Hz屏幕')


def test_pro_plus_keeps_distinct_identity_and_cannot_lend_base_model_evidence():
    from app.phone_names import parse_phone_names,register_phone,normalized_name
    from app.retrieval import exact_model_in_text,variant_only_title
    base=register_phone(parse_phone_names('红魔11 Pro')[0],{})
    plus=register_phone(parse_phone_names('红魔11 Pro+')[0],{})
    assert base.id!=plus.id and normalized_name(base.name)!=normalized_name(plus.name)
    assert not exact_model_in_text(base.name,'红魔11 Pro+ 实测')
    assert variant_only_title(base.name,'红魔11 Pro+ 实测')
    assert exact_model_in_text(plus.name,'红魔11 Pro+ 实测')


def test_partial_bad_citation_keeps_valid_answer_blocks(tmp_path):
    class Client:
        available=True
        def complete_json(self,*args,**kwargs):
            return {'blocks':[{'phone_id':'a','text':'机身重量185g','evidence_ids':['test']},
                              {'phone_id':'a','text':'无效引用','evidence_ids':['missing-id']}]}
    assistant=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=Client(),catalog={'a':phone('a',2800,['轻便'])})
    session=ChatSession(session_id='a'*32,profile=SelectionProfile(budget_max=3000,priorities=['轻便'],candidate_ids=['a']))
    state={'session':session,'sources':[source('机身重量185g')],'catalog':assistant.catalog,
           'selected':['a'],'message':'重量多少？','acknowledged':[]}
    reply=assistant.answer(state)['reply']
    assert len(reply.blocks)==1 and reply.blocks[0].text=='机身重量185g'
    assert state['answer_failure']['stage']=='citation_scope' and reply.mode=='AI对话'


@pytest.mark.parametrize('text,stage',[
    ('官网建议价4999元，优先看这款。','price_provenance'),
    ('预算内6499元，剩余预算可以考虑7499元的版本。','budget_scope'),
])
def test_purchase_claims_cannot_upgrade_sources_or_exceed_budget(tmp_path,text,stage):
    class Client:
        available=True
        def complete_json(self,*a,**k):
            return {'blocks':[{'phone_id':'a','text':text,'evidence_ids':['test']}]}
    assistant=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=Client(),catalog={'a':phone('a',6499,['拍照'])})
    session=ChatSession(session_id='a'*32,profile=SelectionProfile(budget_max=7000,priorities=['拍照'],candidate_ids=['a']))
    state={'session':session,'sources':[source('版本价格6499元和7499元，仅为媒体转述，无拍照实测')],
           'catalog':assistant.catalog,'selected':['a'],'message':'预算7000重新推荐','acknowledged':[]}
    from app.chat_answer import AnswerComposer
    from app.chat_schemas import AnswerBlock
    composer=AnswerComposer(assistant,state)
    assert not composer.validate_block(AnswerBlock(phone_id='a',text=text,evidence_ids=['test']),{'test':state['sources'][0]})
    assert state['answer_failure']['stage']==stage


def test_saved_body_review_is_reused_ahead_of_old_search_previews(tmp_path):
    snippets=[source('a 搜索摘要',sid=f's{i}').model_copy(update={'source_type':'snippet'}) for i in range(8)]
    review=source('原神最高画质连续30分钟，平均59fps，机身温度40℃。','review','review')
    text='继续看原神的实测'
    model=Model([done()],{text:{}})
    store=ChatStore(tmp_path/'s.sqlite3')
    assistant=SelectionAssistant(store,model_client=model,catalog={'a':phone('a',2800,['游戏'])},source_loader=Public())
    s=store.create();s.profile=SelectionProfile(budget_max=3000,priorities=['游戏'],games=['原神'],candidate_ids=['a'])
    s.messages=[ChatMessage(role='user',content='预算3000，游戏'),ChatMessage(role='assistant',content='已查到',sources=[*snippets,review])]
    s=store.save(s)
    reply=assistant.turn(ChatRequest(message=text,session_id=s.session_id)).messages[-1]
    assert any(x['id']=='review' for x in model.contexts[0]['可复用来源'])
    assert any(s.id=='review' for s in reply.sources)


def test_official_product_navigation_exposes_exact_model_parameters(tmp_path,monkeypatch):
    from app.phone_names import parse_phone_names,register_phone
    from app.phone_sources import phone_page_links
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443))])
    phone=register_phone(parse_phone_names('vivo X300')[0],{})
    html='<title>vivo X300</title><header><nav><a href="/vivo/param/x300">参数</a><a href="/vivo/param/x300pro">参数</a></nav></header><main><p>vivo X300专业影像旗舰，页面展示配置。</p></main>'
    links=phone_page_links(phone,'https://www.vivo.com.cn/vivo/x300/',html,'价格 参数')
    assert [p['url'] for p in links]==['https://www.vivo.com.cn/vivo/param/x300']


def test_photo_purchase_fallback_does_not_copy_marketing_recommendation(tmp_path):
    class Broken:
        available=False
    assistant=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=Broken(),catalog={'a':phone('a',2800,['拍照'])})
    state={'session':ChatSession(session_id='a'*32,profile=SelectionProfile(budget_max=3000,priorities=['拍照'],candidate_ids=['a'])),
           'sources':[source('作为本次推荐核心机型，这款相机强，闭眼买，第一名。')],
           'catalog':assistant.catalog,'selected':['a'],'message':'预算3000重新推荐','acknowledged':[]}
    reply=assistant.answer(state)['reply']
    assert not reply.blocks and not reply.cards
    assert '无法给出有依据的推荐' in reply.content and '闭眼买' not in reply.content


def test_known_phone_reference_cannot_be_asked_again(tmp_path):
    text='它的夜景怎么样'
    model=Model([{'kind':'clarify','message':'它是哪款？','clarification_target':'phone'},done()],{text:{}})
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=model,catalog={'a':phone('a',2800,['拍照'])},source_loader=Public())
    s=service.store.create();s.profile=SelectionProfile(budget_max=3000,priorities=['拍照'],candidate_ids=['a'],focus_phone_id='a')
    s=service.store.save(s)
    reply=service.turn(ChatRequest(message=text,session_id=s.session_id)).messages[-1]
    assert '它是哪款' not in reply.content
    assert any(e.get('validation_feedback') for e in reply.agent_trace['actions'])


@pytest.mark.parametrize('weight,expected',[(185,'配置符合'),(205,'配置不符合')])
def test_specific_requirement_uses_numeric_evidence_not_keyword(weight,expected):
    p=SelectionProfile(requirements=['重量不超过190克'])
    labels,_=card_matches('a',p,[source(f'机身重量{weight}g')])
    assert any(expected in label for label in labels)
    assert '待逐项核对' in card_matches('a',p,[source('重量轻巧，手感舒适')])[0][0]


def test_model_can_choose_another_search_engine_after_failure(tmp_path,monkeypatch):
    import app.phone_sources as module
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443))])
    queries=[]
    monkeypatch.setattr(module,'search_web',lambda q,n:queries.append(q) or [])
    loader=PublicPhoneSources(tmp_path/'cache',searcher=lambda *a:[])
    text='预算6000，拍照优先'
    model=Model([[tool('discover_phones',query='手机 拍照',backend='google')],
                 [tool('discover_phones',query='手机 拍照',backend='bing')],done()],
                {text:{'budget_max':6000,'priorities':['拍照']}})
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),model_client=model,source_loader=loader)
    result=service.turn(ChatRequest(message=text))
    trace=result.messages[-1].agent_trace
    assert queries==['手机 拍照']
    assert [a['arguments']['backend'] for a in trace['actions'] if a.get('name')=='discover_phones']==['google','bing']
