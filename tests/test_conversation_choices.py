"""Conversation behavior with synthetic sources, not purchasing fixtures."""
from app.chat import SelectionAssistant
from app.chat_schemas import ChatMessage, ChatSource, SelectionProfile
from app.chat_store import ChatStore
from app.knowledge import KnowledgeBase, KnowledgeIndex, KnowledgeChunk
from app.selection_catalog import CatalogPhone
from tests.test_chat_agent import Model, Public, tool, say
from app.phone_sources import page_kind, extract_phone_excerpt
from app.phone_sources import PublicPhoneSources
from app.selection_requirements import card_matches


def assistant_with_choices(tmp_path, text, patch, names, *, finished=None):
    catalog, chunks = {}, []
    for n, name in enumerate(names):
        pid = f'fixture-{n}'
        catalog[pid] = CatalogPhone(id=pid, name=name, aliases=[name], brand='vivo', price=4200+n*100,
            variant='256GB', topics=['拍照', '续航'], note='虚构夹具', checked_on='2026-10-22')
        quote = '虚构测试夹具：拍照样张对焦准确、肤色自然。续航测试连续视频播放10小时剩余20%。'
        if n == 2:
            quote = '虚构测试夹具：拍照样张细节清晰、曝光稳定。没有续航测试。电池6000mAh。'
        chunks.append(KnowledgeChunk(phone_name=name, title='虚构测试', source_url=f'https://example.com/{n}',
            text=quote, document_id=pid, chunk_id=pid, checked_on='2026-10-22', evidence_kind='review'))
    if patch.get('candidate_action')=='alternatives':patch={**patch,'candidate_action_quote':text}
    model = Model([[tool('search_phone_knowledge', phone_name=name, query='拍照 续航') for name in names],
        {'kind':'answer', 'candidate_names':finished or names}], {text:patch})
    assistant = SelectionAssistant(ChatStore(tmp_path/'session.sqlite3'), catalog=catalog, model_client=model,
        source_loader=Public(), knowledge=KnowledgeBase(KnowledgeIndex(chunks=chunks)))
    return assistant, model


def test_default_three_distinct_choices_keep_partial_alternative_and_one_block_each(tmp_path):
    text = '预算5000，拍照和续航，只用本地资料'
    names = ['vivo X100', 'vivo X200', 'vivo X300']
    assistant, _ = assistant_with_choices(tmp_path, text, {'budget_max':5000,'priorities':['拍照','续航']}, names)
    session = say(assistant, text)
    reply = session.messages[-1]
    assert len(reply.cards) == len(reply.blocks) == session.profile.recommendation_count == 3
    assert len({b.phone_id for b in reply.blocks}) == 3
    assert any('续航：配置' in m and '6000mAh' in m for m in reply.cards[-1].matches)
    assert not any(m.startswith('续航：实测') for m in reply.cards[-1].matches)
    assert '推荐这3款' in reply.content and '换一批' in reply.followup
    assert reply.agent_trace['metrics']['network_calls'] == 0


def test_semantic_alternatives_clear_focus_but_preserve_conditions_and_history(tmp_path):
    # Wording need not match a literal keyword. The model supplies the operation.
    text = '这批都不太对胃口，给点新选择'
    assistant, _ = assistant_with_choices(tmp_path,text,{'candidate_action':'alternatives'},['vivo X100','vivo X200','vivo X300'])
    s = assistant.store.create()
    s.profile = SelectionProfile(budget_max=5000,priorities=['拍照','续航'],excluded_brands=['华为'],
        candidate_ids=['fixture-0'],explicit_candidate_ids=['fixture-0'],focus_phone_id='fixture-0')
    old = ChatMessage(role='assistant',content='历史原文',cards=[assistant.catalog['fixture-0'].card(s.profile)])
    s.messages = [old]
    state = assistant.understand({'session':s,'message':text})
    assert state['alternative_ids'] == ['fixture-0']
    assert not s.profile.focus_phone_id and not s.profile.explicit_candidate_ids
    assert 'fixture-0' not in s.profile.candidate_ids
    assert s.profile.budget_max == 5000 and s.profile.priorities == ['拍照','续航']
    assert s.profile.excluded_brands == ['华为'] and not s.profile.excluded_phones
    assert s.messages[0] == old


def test_alternatives_cannot_return_old_phone_even_if_model_selects_it(tmp_path):
    text = '还有其他备选吗，只用本地资料'
    names = ['vivo X100', 'vivo X200', 'vivo X300']
    assistant, model = assistant_with_choices(tmp_path,text,{'candidate_action':'alternatives'}, names)
    # Only query the new alternatives, then deliberately include the old one.
    model.steps = iter([[tool('search_phone_knowledge',phone_name=n,query='拍照 续航') for n in names[1:]],
                        {'kind':'answer','candidate_names':names}])
    s = assistant.store.create()
    s.profile = SelectionProfile(budget_max=5000,priorities=['拍照','续航'],candidate_ids=['fixture-0'])
    s.messages = [ChatMessage(role='assistant',content='旧结果',cards=[assistant.catalog['fixture-0'].card(s.profile)])]
    s = assistant.store.save(s)
    updated = say(assistant,text,s)
    assert {c.id for c in updated.messages[-1].cards} == {'fixture-1','fixture-2'}
    assert model.contexts[0]['暂避上一批'] == ['vivo X100']
    assert 'fixture-0' not in model.contexts[0]['可查询机型']
    assert not updated.profile.excluded_phones


def test_qualitative_camera_review_is_not_demoted_to_specs_or_marketing_promoted():
    quote = '主摄像头拍摄时对焦准确。不同镜头的曝光和白平衡几乎没有差别，但暗部细节仍然不够清晰。'
    assert page_kind('虚构机型影像测评', quote) == 'review'
    assert page_kind('虚构机型参数', quote) == 'spec'
    assert page_kind('虚构机型评测', '影像效果非常好！5000万像素带来惊艳照片。') == 'spec'
    src = ChatSource(id='review',phone_id='fixture',title='虚构机型测评',url='https://example.com/test',
        quote=quote,checked_on='2026-10-22',source_type='page',evidence_kind='review')
    labels, _ = card_matches('fixture', SelectionProfile(priorities=['拍照']), [src])
    assert any('拍照：实测资料' in label for label in labels)


def test_mixed_need_excerpt_preserves_camera_and_endurance_observations():
    # Six attractive price paragraphs used to crowd out the second use case.
    html = '<title>vivo X100 评测夹具</title><main>'
    html += ''.join(f'<p>vivo X100 测试版本{i}，本次售价{4000+i}元，内存与价格资料仅为虚构。</p>' for i in range(6))
    html += '<p>主摄像头在室内拍照时，对焦准确，曝光稳定，但暗部有噪点。</p>'
    html += '<p>电池续航测试连续播放视频10小时，剩余电量20%，仅为虚构测试。</p></main>'
    excerpt = extract_phone_excerpt('vivo X100',html,'拍照 续航 价格')
    assert '对焦准确' in excerpt and '10小时' in excerpt


def test_empty_parallel_batch_reaches_model_before_no_progress_stop(tmp_path):
    text = '想了解这三款，只用本地资料'
    names = ['vivo X100','vivo X200','vivo X300']
    assistant, model = assistant_with_choices(tmp_path,text,{'selected_phones':[]},names)
    assistant.knowledge = KnowledgeBase(KnowledgeIndex(chunks=[KnowledgeChunk(phone_name='另一部虚构手机',
        title='无关资料',source_url='https://example.com/unrelated',text='无关测试夹具',document_id='other',chunk_id='other')]))
    s = assistant.store.create()
    s.profile = SelectionProfile(budget_max=5000,priorities=['拍照'],candidate_ids=list(assistant.catalog),
        explicit_candidate_ids=list(assistant.catalog))
    s = assistant.store.save(s)
    model.steps = iter([[tool('search_phone_knowledge',phone_name=n,query='拍照') for n in names],
                        {'kind':'answer'}])
    reply = say(assistant,text,s).messages[-1]
    assert len(model.histories) == 2
    assert reply.agent_trace['stop_reason'] == 'model_finished'
    assert reply.agent_trace['metrics']['tool_calls'] == 3


def test_article_body_excludes_related_products_without_recognizable_phone_names():
    html = '''<title>vivo X100 评测</title><body><div class="context-box">
        <p>vivo X100样张中肤色自然，但逆光时高光容易过曝。</p>
        <p>电池续航测试连续使用4小时，电量从95%降至61%。仅为虚构夹具。</p></div>
        <div><p>另一款未命名产品拍照更好，续航测试10小时剩余90%，售价4999元，值得购买。</p></div></body>'''
    excerpt = extract_phone_excerpt('vivo X100',html,'拍照 续航 价格')
    assert '未命名产品' not in excerpt and '95%' in excerpt


def test_explicit_mainland_retail_price_is_recorded_without_claiming_official(tmp_path):
    text='预算5000，拍照，只用本地资料'
    assistant,_=assistant_with_choices(tmp_path,text,{},['vivo X100'])
    phone=assistant.catalog['fixture-0']
    html='<title>vivo X100 价格</title><main><p>vivo X100国行12GB+256GB：4299元。仅为虚构价格夹具。</p></main>'
    loader=PublicPhoneSources(tmp_path/'cache',fetcher=lambda _:html)
    result=loader.fetch_result(phone,'https://8.8.8.8/test','价格')
    prices=[s for s in result['sources'] if s.evidence_kind=='price' and s.price_options]
    assert prices and prices[0].market=='中国大陆'
    s=assistant.store.create();s.discovered_phones=[phone]
    assistant._remember_sources({'catalog':assistant.catalog,'session':s},prices)
    assert s.discovered_phones[0].price==4299
    assert '非官方报价' in s.discovered_phones[0].note


def test_verified_body_evicts_previews_when_source_registry_is_full(tmp_path):
    text='想了解vivo X100'
    assistant,model=assistant_with_choices(tmp_path,text,{'selected_phones':['vivo X100']},['vivo X100'])
    class ManyPreviews(Public):
        def search(self,phone,query,**kwargs):
            return {'sources':[ChatSource(id=query+str(i),phone_id=phone.id,title=phone.name,
                url=f'https://example.com/{query}/{i}',quote='搜索摘要',checked_on='2026-10-22',source_type='snippet') for i in range(4)],
                'candidates':[{'url':'https://example.com/body','title':phone.name}]}
        def fetch(self,phone,url,*args,**kwargs):
            return [ChatSource(id='verified-body',phone_id=phone.id,title=phone.name,url=url,
                quote='虚构夹具：主摄像头曝光稳定、对焦准确。',checked_on='2026-10-22',source_type='page',evidence_kind='review')]
    assistant.source_loader=ManyPreviews()
    model.steps=iter([[tool('web_search',phone_name='vivo X100',query='vivo X100 第一查询')],
        [tool('web_search',phone_name='vivo X100',query='vivo X100 第二查询')],
        [tool('fetch_page',phone_name='vivo X100',url='https://example.com/body')],{'kind':'answer'}])
    reply=say(assistant,text).messages[-1]
    assert len(reply.sources)==8 and any(s.id=='verified-body' for s in reply.sources)
    assert any(b.evidence_ids==['verified-body'] for b in reply.blocks)


def test_local_tool_schema_cannot_masquerade_as_google_search(tmp_path):
    text='想了解这两款，只用本地资料'
    assistant,model=assistant_with_choices(tmp_path,text,{},['vivo X100','vivo X200'])
    assistant.knowledge=KnowledgeBase(KnowledgeIndex(chunks=assistant.knowledge.index.chunks[:1]))
    observed=[]
    invoke=model.invoke_tools
    model.invoke_tools=lambda messages,schemas,**kw: observed.extend(schemas) or invoke(messages,schemas,**kw)
    model.steps=iter([{'kind':'answer'}])
    s=assistant.store.create();s.profile=SelectionProfile(budget_max=5000,priorities=['拍照'],
        candidate_ids=list(assistant.catalog),explicit_candidate_ids=list(assistant.catalog));s=assistant.store.save(s)
    say(assistant,text,s)
    definitions={t['function']['name']:t['function']['parameters']['properties'] for t in observed}
    assert 'backend' not in definitions['search_phone_knowledge']
    assert definitions['search_phone_knowledge']['phone_name']['enum']==['vivo X100']
    assert 'web_search' not in definitions


def test_review_document_does_not_turn_spec_cells_into_camera_or_battery_tests():
    src=ChatSource(id='mixed',phone_id='fixture',title='虚构综合评测',url='https://example.com/test',
        quote='长焦：5000万像素，最近对焦10cm。电池7000mAh，支持30分钟充至90%。长焦拍摄、震感反馈等细节存在妥协。',
        checked_on='2026-10-22',source_type='page',evidence_kind='review')
    labels,_=card_matches('fixture',SelectionProfile(priorities=['拍照','续航']),[src])
    assert labels==['拍照：待补实测','续航：待补实测']


def test_invalid_intent_does_not_silently_disable_agent_or_restore_old_catalog(tmp_path):
    text='还有别的推荐码，只用本地资料'
    names=['vivo X100','vivo X200','vivo X300']
    assistant,model=assistant_with_choices(tmp_path,text,{'candidate_action':'invalid'},names)
    model.steps=iter([[tool('search_phone_knowledge',phone_name=n,query='拍照 续航') for n in names[1:]],
                      {'kind':'answer','candidate_names':names[1:]}])
    s=assistant.store.create();s.profile=SelectionProfile(budget_max=5000,priorities=['拍照','续航'],candidate_ids=['fixture-0'])
    s.messages=[ChatMessage(role='assistant',content='原答',cards=[assistant.catalog['fixture-0'].card(s.profile)])]
    s=assistant.store.save(s)
    reply=say(assistant,text,s).messages[-1]
    assert reply.agent_trace['engine']=='langgraph-chat-agent'
    assert any(a['action']=='intent_fallback' for a in reply.agent_trace['actions'])
    assert {c.id for c in reply.cards}=={'fixture-1','fixture-2'}


def test_photo_battery_fallback_uses_observations_and_exposes_missing_primary_test(tmp_path):
    text='预算5000，拍照和续航'
    assistant,_=assistant_with_choices(tmp_path,text,{},['vivo X100'])
    s=assistant.store.create();s.profile=SelectionProfile(budget_max=5000,priorities=['拍照','续航'],candidate_ids=['fixture-0'])
    src=ChatSource(id='real-shape',phone_id='fixture-0',title='虚构综合评测',url='https://example.com/test',
        quote='相机采用5000万像素。电池续航实测连续视频播放10小时，剩余20%。',checked_on='2026-10-22',
        source_type='page',evidence_kind='review')
    class Offline:
        available=False
    reply=assistant.answer({'session':s,'message':text,'catalog':assistant.catalog,'selected':['fixture-0'],
        'sources':[src],'acknowledged':[]},model_client=Offline())['reply']
    assert '5000万' in reply.blocks[0].text and '10小时' in reply.blocks[0].text
    assert any('未取得对应实测' in d for d in reply.details) and len(reply.blocks)==1
    assert '拍照还缺对应实测' not in reply.blocks[0].text


def test_other_use_only_article_cannot_fill_requested_recommendation_count(tmp_path):
    text='预算5000，拍照和续航，只用本地资料'
    assistant,_=assistant_with_choices(tmp_path,text,{'budget_max':5000,'priorities':['拍照','续航']},
        ['vivo X100','vivo X200','vivo X300'])
    assistant.knowledge.index.chunks[1].text='原神极高画质连续测试30分钟平均60fps，虚构游戏测试夹具。'
    reply=say(assistant,text).messages[-1]
    assert {c.id for c in reply.cards}=={'fixture-0','fixture-2'}
    assert len(reply.cards)==2  # Don't fill the third slot with an unrelated test.


def test_execution_limit_does_not_claim_model_provider_is_unavailable(tmp_path):
    from app.tool_loop import ExecutionLimit
    text='预算5000，拍照'
    assistant,_=assistant_with_choices(tmp_path,text,{},['vivo X100'])
    s=assistant.store.create();s.profile=SelectionProfile(budget_max=5000,priorities=['拍照'],candidate_ids=['fixture-0'])
    src=ChatSource(id='verified',phone_id='fixture-0',title='虚构实测',url='https://example.com/test',
        quote='拍照样张曝光稳定，细节清晰。',checked_on='2026-10-22',source_type='page',evidence_kind='review')
    class Limited:
        available=True
        def complete_json(self,*args,**kwargs):
            raise ExecutionLimit('token_budget')
    reply=assistant.answer({'session':s,'message':text,'catalog':assistant.catalog,'selected':['fixture-0'],
        'sources':[src],'acknowledged':[]},model_client=Limited())['reply']
    assert reply.mode=='资料模式' and reply.blocks
    assert any('执行用量' in d for d in reply.details)
