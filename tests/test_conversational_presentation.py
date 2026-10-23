"""Source-backed provisional recommendations versus focused experience questions."""
import pytest

from app.chat import SelectionAssistant
from app.chat_schemas import ChatSession, ChatSource, SelectionProfile
from app.chat_store import ChatStore
from app.selection_catalog import CatalogPhone
from app.selection_presentation import configuration_block, quoted_price


def fixture(topic="拍照", budget=5000):
    phone = CatalogPhone(id="fixture", name="vivo X100", aliases=[], brand="vivo", price=4299,
        variant="12GB+256GB", checked_on="2026-10-22", note="虚构测试，非购买资料")
    spec = ChatSource(id="config", phone_id=phone.id, title="虚构参数",
        url="https://www.vivo.com.cn/vivo/x100/", source_type="page", evidence_kind="spec",
        quote="虚构参数：5000万像素主摄、1200万像素超广角，OIS光学防抖。电池6000mAh，90W快充。",
        checked_on="2026-10-22", market="中国大陆")
    price = spec.model_copy(update={"id":"price", "quote":"虚构价格：12GB+256GB：4299元。",
        "evidence_kind":"price", "reference_price":4299, "price_variant":"12GB+256GB"})
    profile = SelectionProfile(budget_max=budget, priorities=[topic], candidate_ids=[phone.id])
    return phone, [spec, price], profile


@pytest.mark.parametrize("topic,expected", [("拍照","5000万像素主摄"),("续航","6000mAh")])
def test_preliminary_choices_keep_parameters_without_measured_test(tmp_path,topic,expected):
    phone, sources, profile = fixture(topic)
    class Offline:
        available=False
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),catalog={phone.id:phone},model_client=Offline())
    state={"session":ChatSession(session_id='a'*32,profile=profile),"message":"换一批",
        "recommendation_turn":True,"selected":[phone.id],"catalog":service.catalog,"sources":sources,"acknowledged":[]}
    reply=service.answer(state)['reply']
    assert expected in reply.blocks[0].text and '官网标价12GB+256GB 4299元' in reply.blocks[0].text
    assert '缺对应实测' not in reply.blocks[0].text and '售价待核对' not in reply.blocks[0].text
    assert '备选' in reply.content and '换一批' in reply.followup and '只留第一款' in reply.followup
    assert any('未取得对应实测' in d for d in reply.details)
    assert '配置：' in reply.cards[0].reason


def test_model_can_recommend_camera_configuration_without_being_rejected(tmp_path):
    phone,sources,profile=fixture()
    text='vivo X100：5000万像素主摄、OIS光学防抖，可作为相机配置方向的备选。官网标价12GB+256GB 4299元。'
    class Model:
        available=True
        def complete_json(self,*a,**k):
            return {'blocks':[{'phone_id':phone.id,'text':text,'evidence_ids':['config','price']}]}
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),catalog={phone.id:phone},model_client=Model())
    state={'session':ChatSession(session_id='a'*32,profile=profile),'message':'预算5000拍照推荐',
        'recommendation_turn':True,'selected':[phone.id],'catalog':service.catalog,'sources':sources,'acknowledged':[]}
    reply=service.answer(state)['reply']
    assert reply.mode=='AI对话' and reply.blocks[0].text==text
    assert 'answer_failure' not in state


@pytest.mark.parametrize('draft_kind', ['offline', 'configuration', 'gap_only', 'unsupported_fps'])
def test_gaming_shortlist_keeps_basic_information_without_benchmarks(tmp_path, draft_kind):
    phone,sources,profile=fixture('游戏',6000)
    profile.priorities=['游戏','续航']
    sources[0]=sources[0].model_copy(update={'quote':'骁龙8至尊版，6000mAh电池，100W有线与50W无线充电；1至120Hz屏幕。'})
    texts={'configuration':'骁龙8至尊版、6000mAh电池和50W无线充电，游戏画面与流畅度上有配置基础，可作为初筛备选。',
           'gap_only':'还缺游戏、续航的对应实测，暂作待核对备选。',
           'unsupported_fps':'骁龙8至尊版，游戏稳定60帧，6000mAh电池。'}
    class Model:
        available=draft_kind!='offline'
        def complete_json(self,*a,**k):
            return {'blocks':[{'phone_id':phone.id,'text':texts[draft_kind],'evidence_ids':['config']}]}
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),catalog={phone.id:phone},model_client=Model())
    state={'session':ChatSession(session_id='a'*32,profile=profile),'message':'预算6000，主要玩游戏，注重续航',
        'recommendation_turn':True,'selected':[phone.id],'catalog':service.catalog,'sources':sources,'acknowledged':[]}
    reply=service.answer(state)['reply']
    text=reply.blocks[0].text
    assert '骁龙8至尊版' in text and '6000mAh' in text and '50W无线充电' in text
    assert '缺游戏' not in text and '60帧' not in text
    assert any('未取得对应实测' in d for d in reply.details)
    assert set(reply.blocks[0].evidence_ids)<={'config','price'}
    if draft_kind=='configuration':
        assert reply.mode=='AI对话' and 'answer_failure' not in state
    elif draft_kind=='unsupported_fps':
        assert state['answer_failure']['stage']=='recommendation_evidence'
    elif draft_kind=='gap_only':
        assert state['answer_failure']['stage']=='basic_information'


def test_focused_night_scene_question_explains_test_gap_once_and_keeps_parameters(tmp_path):
    phone,sources,profile=fixture()
    profile.focus_phone_id=phone.id
    class Offline:
        available=False
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),catalog={phone.id:phone},model_client=Offline())
    reply=service.answer({'session':ChatSession(session_id='a'*32,profile=profile),'message':'它的夜景成片怎么样',
        'recommendation_turn':False,'selected':[phone.id],'catalog':service.catalog,'sources':sources,'acknowledged':[]})['reply']
    assert '实测' in reply.content and '5000万像素主摄' in reply.blocks[0].text
    assert '4299' not in reply.blocks[0].text and not reply.followup


def test_price_requires_literal_version_mainland_and_respects_requested_storage():
    phone,sources,profile=fixture()
    # Changing metadata alone cannot create a new version or official regional offer.
    wrong=sources[-1].model_copy(update={'price_variant':'16GB+512GB'})
    assert quoted_price(phone,profile,[wrong]) is None
    foreign=sources[-1].model_copy(update={'market':'其他地区'})
    assert quoted_price(phone,profile,[foreign]) is None
    profile.requirements=['至少512GB']
    assert quoted_price(phone,profile,sources) is None


def test_conflicting_official_prices_are_not_resolved_by_picking_the_cheaper_one():
    phone,sources,profile=fixture()
    other=sources[-1].model_copy(update={'id':'other','quote':'虚构价格：12GB+256GB：4499元。','reference_price':4499})
    assert quoted_price(phone,profile,[sources[-1],other]) is None


def test_search_preview_does_not_become_a_configuration_recommendation():
    phone,sources,profile=fixture()
    assert configuration_block(phone,profile,[s.model_copy(update={'source_type':'snippet'}) for s in sources]) is None


def test_adjacent_spec_cells_do_not_attach_rear_pixel_count_to_front_camera():
    from app.selection_presentation import configuration_facts
    phone,sources,profile=fixture()
    sources[0].quote='后置摄像头：5000万像素 前置摄像头：3200万像素。'
    facts=configuration_facts(phone,profile.priorities,sources)
    assert all('5000万像素 前置' not in value for value,_ in facts)
    assert any('3200万' in value for value,_ in facts)


def test_saved_parameter_only_choices_remain_reusable_after_service_reload(tmp_path):
    from app.chat_schemas import ChatMessage
    from app.selection_requirements import discovery_needed,card_matches
    phone,sources,profile=fixture()
    class Offline:
        available=False
    service=SelectionAssistant(ChatStore(tmp_path/'s.sqlite3'),catalog={},model_client=Offline())
    session=service.store.create();session.profile=profile;session.discovered_phones=[phone]
    session.messages=[ChatMessage(role='assistant',content='历史原答保持',sources=sources)]
    session=service.store.save(session)
    loaded=service.store.load(session.session_id)
    state=service.understand({'session':loaded,'message':'按当前条件重新推荐'})
    assert '拍照' in loaded.discovered_phones[0].topics
    assert not discovery_needed(loaded.profile,state['catalog'])
    assert card_matches(phone.id,profile,sources)[0]==['拍照：待补实测']
    assert loaded.messages[0].content=='历史原答保持'


def test_parameter_page_prefers_actual_camera_cells_over_long_mode_list():
    from app.phone_sources import extract_phone_excerpt
    html='<title>vivo X100规格</title><main><h1>vivo X100</h1>'
    html+='<p>拍照模式：人像、夜景、全景、录像、高像素、文档、扫描</p>'
    html+=''.join('<p>虚构通用参数单元，处理器、充电、电池等信息占位。</p>' for _ in range(8))
    html+='<p>5000万像素蔡司主摄，5000万像素长焦，支持OIS防抖和3倍光学变焦。</p></main>'
    quote=extract_phone_excerpt('vivo X100',html,'相机配置')
    assert '5000万像素蔡司主摄' in quote and quote.index('5000万') < quote.index('拍照模式')


def test_official_directory_is_navigation_only_and_can_expose_product_page(tmp_path,monkeypatch):
    import socket
    from app.phone_sources import PublicPhoneSources
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443))])
    phone,_,_=fixture()
    loader=PublicPhoneSources(tmp_path/'cache',searcher=lambda *a:[{'title':'vivo官方网站',
        'url':'https://www.vivo.com.cn/','snippet':'品牌首页'}],fetcher=lambda url:
        '<title>vivo官方网站</title><nav><a href="/vivo/x100/">X100</a><a href="/vivo/x100pro/">X100 Pro</a></nav>')
    searched=loader.search(phone,'vivo X100 官网')
    assert not searched['sources'] and searched['candidates'][0]['status']=='navigation_only'
    fetched=loader.fetch_result(phone,searched['candidates'][0]['url'],'相机配置')
    assert not fetched['sources']
    assert [c['url'] for c in fetched['candidates']]==['https://www.vivo.com.cn/vivo/x100/']


def test_public_query_preserves_model_chosen_spacing_and_quotes(monkeypatch):
    import socket
    from urllib.parse import parse_qs,urlsplit
    import app.phone_sources as module
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443))])
    observed=[]
    def fetch(url):
        observed.append(parse_qs(urlsplit(url).query)['q'][0])
        return '<li class="b_algo"><h2><a href="https://8.8.8.8/spec">参数</a></h2><p>虚构结果</p></li>'
    monkeypatch.setattr(module,'fetch_html',fetch)
    query='"vivo X100" site:vivo.com.cn 参数'
    assert module.search_phone_web(query) and observed==[query]


def test_compaction_keeps_official_specs_for_each_phone_ahead_of_review_flood():
    from app.chat_agent import pending_page_links
    phone,_,_=fixture()
    second=phone.model_copy(update={'id':'second','name':'vivo X200'})
    actions=[{'action':'tool','name':'web_search','arguments':{'phone_name':phone.name},'result':{
        'candidates':[{'title':'虚构评测','url':f'https://example.com/review/{i}'} for i in range(8)]}}]
    for p in [phone,second]:
        actions.append({'action':'tool','name':'web_search','arguments':{'phone_name':p.name},'result':{
            'candidates':[{'title':'规格参数','url':f'https://www.vivo.com.cn/vivo/param/{p.id}'}]}})
    pending=pending_page_links(actions,{phone.id:phone,second.id:second},True)
    assert len(pending)==3
    assert {p['phone_name'] for p in pending[:2]}=={phone.name,second.name}
    assert all('param' in p['url'] for p in pending[:2])
