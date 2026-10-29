"""Offline cross-path regressions. All product observations here are synthetic."""
import pytest

from app.chat import SelectionAssistant
from app.chat_agent import ChatAgentRun
from app.chat_schemas import ChatMessage, ChatRequest, ChatSource, SelectionProfile
from app.chat_store import ChatStore
from app.phone_sources import PublicPhoneSources, extract_phone_excerpt, page_kind
from app.phone_sources import extract_price_options
from app.retrieval import exact_model_in_text, variant_only_title
from app.selection_catalog import load_catalog
from app.selection_requirements import check_requirement
from tests.test_chat_agent import Model, Public, service, tool, done
from app.knowledge import KnowledgeBase,KnowledgeIndex,KnowledgeChunk
from app.evidence_registry import session_evidence,merge_sources,finish_refresh
from app.chat_answer import AnswerComposer
from app.chat_schemas import AnswerBlock


@pytest.mark.parametrize('suffix',['至尊版','青春版','竞速版','Ultra','Pro','Pro Max','卫星通信版'])
def test_variant_boundary_matches_local_and_web_identity(suffix):
    title='红米 k80 '+suffix+' 游戏体验'
    assert not exact_model_in_text('REDMI K80',title)
    assert variant_only_title('REDMI K80',title)


def short_specs():
    return '''<title>一加13 参数规格</title><main><h1>一加13</h1>
    <dl><dt>无线快充</dt><dd>50W 无线闪充</dd></dl>
    <dl><dt>NFC</dt><dd>支持</dd></dl><dl><dt>屏幕形态</dt><dd>直屏</dd></dl>
    <dl><dt>无线反向充电</dt><dd>支持（10W）</dd></dl></main>'''


def test_short_field_values_survive_web_extraction(tmp_path):
    loader=PublicPhoneSources(tmp_path,fetcher=lambda _:short_specs())
    sources=loader.fetch(load_catalog()['oneplus-13'],'https://8.8.8.8/specs','需要无线充电 NFC 屏幕')
    assert check_requirement('需要无线充电',sources).status=='satisfied'
    assert check_requirement('需要NFC',sources).status=='satisfied'
    assert check_requirement('不要曲面屏',sources).status=='satisfied'


def saved_turn(tmp_path,text,mode='local'):
    a=service(tmp_path,Model([])); phone=a.catalog['oneplus-13']
    review=ChatSource(id='saved-review',phone_id=phone.id,title=phone.name+'虚构游戏体验',
        url='https://example.com/review',quote='虚构资料：一加13原神最高画质连续30分钟，平均59fps，温度40℃。',
        source_type='page',evidence_kind='review',checked_on='2026-10-24')
    s=a.store.create();s.profile=SelectionProfile(budget_max=5000,priorities=['游戏'],games=['原神'],
        candidate_ids=[phone.id],focus_phone_id=phone.id)
    s.messages=[ChatMessage(role='assistant',content='历史回答',sources=[review])]
    state={'session':s,'message':text,'search_mode':mode};state.update(a.understand(state))
    return a,state,review


def test_price_refresh_keeps_saved_configuration_and_review(tmp_path):
    a,turn,review=saved_turn(tmp_path,'刷新一下最新价格','web')
    state=ChatAgentRun(a,turn).initialize({})
    assert review.id in state['sources']
    assert review.id in {s.id for s in turn['session'].messages[0].sources}


def test_fixed_local_path_reuses_the_same_saved_web_body_as_agent(tmp_path):
    a,turn,review=saved_turn(tmp_path,'这款原神的温度怎么样？')
    agent=ChatAgentRun(a,turn);state=agent.initialize({})
    assert review.id in state['sources']
    assert not {'fetch_page','web_search','discover_phones'} & {s['function']['name'] for s in agent.schemas(state)}
    assert review.id in {s.id for s in a.retrieve(turn)['sources']}


@pytest.mark.parametrize('title',['虚构游戏评测','虚构游戏体验','原神到底能不能稳住'])
def test_measured_body_does_not_depend_on_review_word_in_title(title):
    body='虚构测试：原神最高画质，连续30分钟测试，平均帧率59fps，机身最高温度40℃。'
    assert page_kind(title,body)=='review'


def test_empty_web_result_preserves_detail_question(tmp_path):
    text='REDMI K80玩原神会发热吗？'
    model=Model([[tool('web_search',phone_name='REDMI K80',query='REDMI K80 原神 发热 实测')],done()],
        {text:{'selected_phones':['REDMI K80'],'turn_action':'question'}})
    class Empty(Public):
        def search(self,*args,**kwargs):
            return {'sources':[],'candidates':[],'lookup_origin':'network','search_feedback':{'matching_pages':0}}
    a=service(tmp_path,model,source_loader=Empty())
    reply=a.turn(ChatRequest(message=text,search_mode='web')).messages[-1]
    assert '原神' in reply.content
    assert '不足以推荐' not in reply.content and '合适机型' not in reply.content


@pytest.mark.parametrize('wanted,alias',[
    ('一加 13','OnePlus13'),('REDMI K80','红米 k80 标准版'),
    ('iPhone 16 Pro Max','苹果16 pro max'),('小米 17','XIAOMI17'),
])
def test_aliases_use_same_identity_for_search_and_local_retrieval(wanted,alias):
    assert exact_model_in_text(wanted,alias+' 参数')
    kb=KnowledgeBase(KnowledgeIndex(chunks=[KnowledgeChunk(phone_name=alias,title='虚构参数',
        source_url='https://example.com/spec',text='屏幕刷新率120Hz，仅为虚构夹具。',document_id='d',chunk_id='c')]))
    assert kb.as_tool().invoke({'phone_name':wanted,'need':'屏幕刷新率'})


def test_english_prose_is_not_a_pro_suffix():
    from app.phone_names import parse_phone_names
    title='REDMI K80 product review'
    assert exact_model_in_text('REDMI K80',title)
    assert [m.name for m in parse_phone_names(title)]==['REDMI K80']


def test_short_cells_and_prices_cannot_borrow_an_upgrade_section(tmp_path):
    html='''<title>一加13 参数</title><main><h1>一加13</h1><dl><dt>NFC</dt><dd>支持</dd></dl>
    <h2>一加13 Ultra</h2><dl><dt>无线充电</dt><dd>不支持</dd></dl>
    <p>国行12GB+256GB售价3999元，仅为虚构夹具。</p></main>'''
    result=PublicPhoneSources(tmp_path,fetcher=lambda _:html).fetch(load_catalog()['oneplus-13'],
        'https://8.8.8.8/mixed','无线充电 NFC 价格')
    assert check_requirement('需要无线充电',result).status=='unknown'
    assert check_requirement('需要NFC',result).status=='satisfied'
    assert not extract_price_options('一加13',html)[0]


@pytest.mark.parametrize('status',[403,429])
def test_http_access_block_is_distinct_from_no_results(tmp_path,status):
    import httpx
    def blocked(url):
        raise httpx.HTTPStatusError('synthetic block',request=httpx.Request('GET',url),response=httpx.Response(status))
    loader=PublicPhoneSources(tmp_path,fetcher=blocked)
    result=loader.fetch_result(load_catalog()['oneplus-13'],'https://8.8.8.8/blocked','屏幕')
    assert result['status']=='blocked' and result['fetch_feedback']['reason']=='access_blocked'


def test_wrong_variant_pending_link_is_not_reintroduced_from_history():
    from app.chat_agent import pending_page_links
    catalog=load_catalog()
    actions=[{'action':'tool','name':'web_search','arguments':{'phone_name':'REDMI K80'},
              'result':{'candidates':[{'title':'红米K80 至尊版原神体验','url':'https://example.com/ultra'}]}}]
    assert pending_page_links(actions,catalog,False)==[]


def test_fixed_public_adapter_obeys_shared_turn_search_fetch_limits(tmp_path,monkeypatch):
    from app.phone_sources import PublicPhoneSources
    from app.tool_loop import AgentPolicy
    queries=[]
    loader=PublicPhoneSources(tmp_path/'cache',searcher=lambda q,k:queries.append(q) or [])
    class Offline:
        available=False
    a=SelectionAssistant(ChatStore(tmp_path/'sessions.sqlite3'),model_client=Offline(),source_loader=loader,
        agent_enabled=False,agent_policy=AgentPolicy(max_tool_calls=2,max_search_calls=1,max_fetch_calls=1))
    s=a.store.create();ids=list(a.catalog)[:3]
    s.profile=SelectionProfile(candidate_ids=ids,explicit_candidate_ids=ids,priorities=['游戏'],budget_max=5000)
    turn={'session':s,'message':'这几款的原神发热呢？','catalog':a.catalog,'selected':ids,
          'query':'原神 发热','search_mode':'web','recommendation_turn':False}
    a.retrieve(turn)
    assert len(queries)==1


def test_existing_seed_phone_price_rebuilds_from_latest_session_snapshot(tmp_path):
    a,turn,review=saved_turn(tmp_path,'这款呢？')
    old=price_source(review.phone_id,3999,'2026-10-24')
    new=price_source(review.phone_id,4299,'2026-10-24')
    s=turn['session'];s.messages[0].sources.append(old)
    s.messages.append(ChatMessage(role='assistant',content='更新记录',sources=[new,review]))
    state={'session':s,'message':'它的价格呢？','search_mode':'local'}
    state.update(a.understand(state))
    assert state['catalog'][review.phone_id].price==4299
    assert a.catalog[review.phone_id].price!=4299


@pytest.mark.parametrize('body,status',[
    ('<dl><dt>无线充电</dt><dd>不支持</dd></dl>','unsatisfied'),
    ('<table><tr><th>无线充电</th><td>支持</td></tr></table>','satisfied'),
    ('<div><p>无线充电</p><p>待确认</p></div>','unknown'),
    ('<dl><dt>无线反向充电</dt><dd>支持10W</dd></dl>','unknown'),
    ('<p>支持无线充电</p><p>不支持无线充电</p>','conflict'),
])
def test_short_fields_keep_negative_unknown_conflict_and_power_direction(tmp_path,body,status):
    html='<title>一加13 参数</title><main><h1>一加13</h1>'+body+'</main>'
    result=PublicPhoneSources(tmp_path,fetcher=lambda _:html).fetch(load_catalog()['oneplus-13'],
        'https://8.8.8.8/fields','需要无线充电')
    assert check_requirement('需要无线充电',result).status==status


@pytest.mark.parametrize('title,html,reason',[
    ('验证码_某网站','<div id="risk-captcha-app"></div>','captcha'),
    ('Access denied','<p>Forbidden</p>','access_blocked'),
    ('一加13','<script>renderContent()</script>','no_body'),
    ('一加13参数','<h1>一加13</h1><p>本页面只有新闻目录和售后服务入口，没有本轮所需的配置数据。</p>','no_relevant_passage'),
    ('一加13 Pro参数','<h1>一加13 Pro</h1><p>一加13 Pro屏幕刷新率120Hz，仅为虚构夹具。</p>','model_mismatch'),
])
def test_fetch_failures_are_explained_and_cached_without_becoming_evidence(tmp_path,title,html,reason):
    calls=[];phone=load_catalog()['oneplus-13']
    loader=PublicPhoneSources(tmp_path,fetcher=lambda u:calls.append(u) or '<title>'+title+'</title><main>'+html+'</main>')
    first=loader.fetch_result(phone,'https://8.8.8.8/failure','屏幕')
    second=loader.fetch_result(phone,'https://8.8.8.8/failure','屏幕')
    assert first['fetch_feedback']['reason']==second['fetch_feedback']['reason']==reason
    assert not first['sources'] and len(calls)==1
    assert second['lookup_origin']=='cache'


@pytest.mark.parametrize('title,body',[
    ('虚构广告','原神最高画质30分钟平均60fps，官方实验室测算，广告宣传效果。'),
    ('虚构游戏体验','支持120fps模式，屏幕刷新率120Hz。'),
    ('虚构游戏体验','最高画质，连续30分钟，平均60fps。'),
])
def test_marketing_settings_and_missing_workload_are_not_measured_reviews(title,body):
    assert page_kind(title,body)=='spec'


def price_source(pid,price=3999,day='2026-10-20'):
    return ChatSource(id='saved-price',phone_id=pid,title='虚构版本价',url='https://example.com/price',
        quote=f'国行12GB+256GB售价{price}元，仅为虚构夹具。',checked_on=day,source_type='page',
        evidence_kind='price',market='中国大陆',reference_price=price,price_variant='12GB+256GB')


def test_price_failure_preserves_review_and_marks_historical_price_without_mutating_history(tmp_path):
    a,turn,review=saved_turn(tmp_path,'刷新一下最新价格','web')
    old=price_source(review.phone_id);turn['session'].messages[0].sources.append(old)
    run=ChatAgentRun(a,turn);state=run.initialize({})
    assert state['sources'][old.id].refresh_status=='pending'
    finish_refresh(turn,state,attempted=True)
    assert state['sources'][old.id].refresh_status=='failed' and review.id in state['sources']
    assert turn['session'].messages[0].sources[-1].refresh_status=='not_requested'
    # A same-URL current read upgrades the original ID, rather than inventing a
    # conflict with its own historical snapshot. Other URLs still conflict.
    newer=price_source(review.phone_id,4299,'2026-10-24')
    run.record_sources(state,review.phone_id,[newer],fresh=True)
    assert state['sources'][old.id].refresh_status=='verified'
    assert state['sources'][old.id].reference_price==4299
    restored=a.store.load(a.store.save(turn['session']).session_id)
    assert restored.messages[0].sources[-1].reference_price==3999


def test_local_refresh_never_accesses_public_sources_and_keeps_price_date(tmp_path):
    a,turn,review=saved_turn(tmp_path,'刷新最新价格，只用已有资料','web')
    old=price_source(review.phone_id);turn['session'].messages[0].sources.append(old)
    class Forbidden(Public):
        def collect(self,*a,**kw):pytest.fail('public access in local intent')
    a.source_loader=Forbidden()
    result=a.retrieve(turn)
    price=next(s for s in result['sources'] if s.id==old.id)
    assert price.refresh_status=='not_permitted' and price.checked_on=='2026-10-20'
    reply=a.answer({**turn,**result})['reply']
    assert any('未允许公开刷新' in d and '2026-10-20' in d for d in reply.details)
    assert any('历史记录' in b.text and '3999元' in b.text for b in reply.blocks)


def test_stale_price_cannot_be_validated_as_current(tmp_path):
    a,turn,review=saved_turn(tmp_path,'最新价格是多少？','local')
    price=price_source(review.phone_id).model_copy(update={'refresh_status':'failed'})
    composer=AnswerComposer(a,{**turn,'sources':[price],'acknowledged':[]})
    bad=AnswerBlock(phone_id=review.phone_id,text='今天最新价格3999元。',evidence_ids=[price.id])
    assert not composer.validate_block(bad,{price.id:price})


def test_unrelated_saved_game_review_does_not_answer_a_price_question(tmp_path):
    a,turn,review=saved_turn(tmp_path,'最新价格是多少？','local')
    a.model.available=False
    reply=a.answer({**turn,**a.retrieve(turn)})['reply']
    assert not reply.blocks and '价格' in reply.content and '59fps' not in reply.content
    assert review.id in {s.id for s in reply.sources}


def test_wrong_variant_cannot_enter_either_source_registry(tmp_path):
    a,turn,review=saved_turn(tmp_path,'这款的屏幕呢？')
    source=review.model_copy(update={'title':'一加13 Ultra规格','quote':'一加13 Ultra屏幕刷新率120Hz。'})
    registry={'sources':{},'allowed_urls':set(),'actions':[]}
    assert merge_sources(registry,review.phone_id,[source],a.catalog,turn['session'].profile)[1]==0
    assert ChatAgentRun(a,turn).record_sources(registry,review.phone_id,[source])[1]==0


def test_cross_mode_multiturn_reuse_refresh_failure_and_cold_restore(tmp_path):
    first='一加13玩原神会发热吗？';web='继续查原神的帧率和温度';local='它的原神温度呢？';refresh='刷新最新价格'
    patches={first:{'selected_phones':['一加13'],'turn_action':'question','priorities':['游戏'],'games':['原神']},
        web:{'turn_action':'question'},local:{'turn_action':'question'},refresh:{'turn_action':'question'}}
    steps=[[tool('search_phone_knowledge',phone_name='一加 13',query='原神')],done(),
        [tool('web_search',phone_name='一加 13',query='一加13 原神 实测 温度')],
        [tool('fetch_page',phone_name='一加 13',url='https://8.8.8.8/review')],done(),done(),
        [tool('web_search',phone_name='一加 13',query='一加13 最新价格')],done()]
    class Lookup(Public):
        def search(self,phone,query,**kwargs):
            self.searches.append(query)
            return {'sources':[],'candidates':[] if '价格' in query else [{'title':phone.name+'游戏体验','url':'https://8.8.8.8/review'}],
                    'lookup_origin':'network','search_feedback':{'matching_pages':0 if '价格' in query else 1}}
        def fetch(self,phone,url,*args,**kwargs):
            self.fetches.append(url)
            return [ChatSource(id='new-review',phone_id=phone.id,title=phone.name+'游戏体验',url=url,
                quote='虚构资料：一加13原神最高画质连续30分钟测试，平均59fps，温度40℃。',
                checked_on='2026-10-24',source_type='page',evidence_kind='review')]
    lookup=Lookup();model=Model(steps,patches);a=service(tmp_path,model,source_loader=lookup)
    s=a.turn(ChatRequest(message=first,search_mode='local'))
    assert not lookup.searches
    s=a.turn(ChatRequest(message=web,search_mode='auto',session_id=s.session_id))
    assert 'new-review' in {source.id for source in s.messages[-1].sources}
    s=a.turn(ChatRequest(message=local,search_mode='local',session_id=s.session_id))
    assert len(lookup.searches)==len(lookup.fetches)==1
    assert '59fps' in ' '.join(b.text for b in s.messages[-1].blocks)
    s=a.turn(ChatRequest(message=refresh,search_mode='web',session_id=s.session_id))
    assert '价格' in s.messages[-1].content and not s.messages[-1].blocks
    assert 'new-review' in {source.id for source in s.messages[-1].sources}
    class Offline:
        available=False
    class NoNewWeb(Public):
        def collect(self,*args,**kwargs):pytest.fail('cold local restore must not fetch')
    restored=SelectionAssistant(ChatStore(a.store.path),model_client=Offline(),knowledge=a.knowledge,source_loader=NoNewWeb(),agent_enabled=False)
    s=restored.turn(ChatRequest(message=local,search_mode='local',session_id=s.session_id))
    assert '59fps' in ' '.join(b.text for b in s.messages[-1].blocks)
    assert s.profile.priorities==['游戏'] and s.profile.games==['原神']


@pytest.mark.parametrize('mode',['local','auto','web'])
def test_condition_delta_cancel_and_fresh_recommendation_preserve_state_across_modes(tmp_path,mode):
    a,seed,review=saved_turn(tmp_path,'这款呢？',mode)
    class Offline:
        available=False
    a.model=Offline();s=seed['session'];s.profile.explicit_candidate_ids=[review.phone_id]
    for text in ['不要曲面屏','需要NFC','取消曲面屏要求','预算改为6000']:
        state={'session':s,'message':text,'search_mode':mode};state.update(a.understand(state))
        initialized=ChatAgentRun(a,state).initialize({})
        assert review.id in initialized['sources']
    assert s.profile.budget_max==6000 and s.profile.priorities==['游戏'] and s.profile.games==['原神']
    assert s.profile.requirements==['需要NFC'] and s.profile.explicit_candidate_ids==[review.phone_id]
    state={'session':s,'message':'按新预算推荐三款游戏手机','search_mode':mode};state.update(a.understand(state))
    assert state['recommendation_turn'] and not s.profile.explicit_candidate_ids
    assert s.profile.candidate_ids!=[review.phone_id] and s.profile.requirements==['需要NFC']
