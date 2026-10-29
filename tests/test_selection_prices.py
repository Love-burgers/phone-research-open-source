import pytest

from app.chat_schemas import ChatSource
from app.knowledge_ingest import source_documents
from app.phone_sources import extract_price_options
from app.selection_catalog import load_catalog
from app.selection_prices import source_price_options
from app.selection_prices import unsupported_prices


@pytest.mark.parametrize('amount,accepted',[
    ('4200元',True),('4200.00元',True),('4200. 00元',True),('4200 . 00元',True),
    ('4200.50元',False),('4200. 50元',False),('14200元',False),('42000元',False),
])
def test_price_state_and_admission_agree_for_decimal_whitespace(amount,accepted):
    phone=load_catalog()['oneplus-13']
    source=ChatSource(id='price',phone_id=phone.id,title='一加13版本价格',
        url='https://www.oneplus.com/cn/13',quote='12GB+256GB：'+amount,
        checked_on='2026-10-23',evidence_kind='price',source_type='page',market='中国大陆',
        price_options=[{'variant':'12GB+256GB','price':4200}])
    assert bool(source_price_options(source))==accepted
    assert bool(source_documents([source],{phone.id:phone}))==accepted


def test_satellite_sibling_price_is_not_a_conflict_for_regular_version():
    html='<title>vivo X200 Pro 官方参数</title><main><p>建议零售价 vivo X200 Pro（16GB+1TB）：6499.00元 vivo X200 Pro 卫星通信版（16GB+1TB）：6799.00元</p></main>'
    offers,_=extract_price_options('vivo X200 Pro',html)
    assert [(o.variant,o.price) for o in offers]==[('16GB+1TB',6499)]


def test_same_regular_version_with_two_prices_stays_ambiguous():
    html='<title>一加13价格</title><main><p>12GB+256GB：4200元；12GB+256GB：3900元</p></main>'
    assert extract_price_options('一加 13',html)[0]==[]


@pytest.mark.parametrize('text,valid',[
    ('版本页面参考价5299元，在5800元预算内。',True),
    ('预算5800元，版本售价4999元。',False),
    ('版本售价5299元，比预算省下501元。',True),
    ('版本售价5299元，比预算省下900元。',False),
    ('版本售价5299.50元。',False),
    ('版本售价5,299.00元。',True),
    ('版本售价5299 . 50元。',False),
    ('12GB+256GB版本售价5299元。',True),
    ('16GB+512GB版本售价5299元。',False),
])
def test_prices_and_derived_savings_need_cited_support(text,valid):
    source=ChatSource(id='price',phone_id='test',title='版本价',url='https://example.com',
        quote='12GB+256GB：5299元',checked_on='2026-10-23',evidence_kind='price',
        price_options=[{'variant':'12GB+256GB','price':5299}])
    assert bool(unsupported_prices(text,[source],5800)) is not valid


@pytest.mark.parametrize('context,expected',[
    ('目前京东活动售价8599元，叠加满减优惠后7579元','中国大陆'),
    ('天猫256GB售价5999元','中国大陆'),
    ('国行256GB售价5999元','中国大陆'),
    ('256GB售价5999元','地区未确认'),
    ('京东国际256GB售价5999元','地区未确认'),
    ('天猫国际256GB售价5999元','地区未确认'),
    ('京东港版256GB售价5999元','地区未确认'),
    ('京东256GB售价5999港元','地区未确认'),
])
def test_market_uses_owned_price_channel_and_currency_not_domain_alone(context,expected):
    from app.phone_sources import source_market
    assert source_market('https://example.com/article','一加13版本报价',context)==expected


@pytest.mark.parametrize('condition',['领券实付','国补后','叠加满减后','以旧换新后','会员专享价'])
def test_conditional_prices_do_not_become_unconditional_budget_offers(condition):
    text=f'一加13 256GB {condition}2599元'
    html='<title>一加13版本报价</title><main><p>'+text+'</p></main>'
    assert extract_price_options('一加 13',html)[0]==[]
    source=ChatSource(id='conditional',phone_id='oneplus-13',title='一加13报价',url='https://example.com',
        quote=text,checked_on='2026-10-25',market='中国大陆',evidence_kind='price',
        price_options=[{'variant':'256GB','price':2599}])
    assert source_price_options(source,'一加 13')==[]


def test_old_promotion_retains_base_price_and_discount_notes_but_not_current_deal():
    from app.phone_sources import PublicPhoneSources
    from app.phone_names import register_phone,parse_phone_names
    from app.chat_schemas import SelectionProfile
    from app.selection_prices import price_lookup_notes, fresh_price_source
    from app.selection_requirements import recommendation_gaps
    phone=register_phone(parse_phone_names('iPhone 15 Pro Max')[0],{})
    html='<title>iPhone 15 Pro Max限时特惠7579元</title><main><p>2025-05-14</p><p>iPhone 15 Pro Max 256GB，目前京东活动售价8599元，叠加满减优惠后实付低至7579元。</p></main>'
    offers,quote=extract_price_options(phone.name,html)
    assert [(o.variant,o.price) for o in offers]==[('256GB',8599)]
    source=ChatSource(id='old',phone_id=phone.id,title='iPhone 15 Pro Max版本价格',url='https://example.com',
        quote=quote,checked_on='2026-10-25',published_on='2025-05-14',source_type='page',
        market='中国大陆',evidence_kind='price',price_options=offers)
    phone.price,phone.variant=8599,'256GB'
    notes='；'.join(price_lookup_notes(phone,[source]))
    assert all(x in notes for x in ['2025-05-14','8599','7579','优惠资格和时效未确认'])
    assert not fresh_price_source(source)
    assert unsupported_prices('目前256GB到手价7579元。',[source],phone_name=phone.name)
    gaps=recommendation_gaps(phone,SelectionProfile(budget_max=8000),[source])
    assert any('超过预算599元' in x for x in gaps)
    from app.selection_requirements import recommendation_notes
    assert recommendation_notes(phone,SelectionProfile(budget_max=8000),[source])
    from app.selection_prices import historical_price_references
    references=historical_price_references({phone.id:phone},[source])
    assert len(references)==1 and references[0].evidence_ids==['old']
    assert all(value in references[0].text for value in ['2025-05-14','京东','256GB 报价8599元','256GB 优惠条件价7579元','满减','当前价格'])
    assert references[0].name==phone.name


def test_market_marker_in_footer_cannot_promote_unscoped_price(tmp_path,monkeypatch):
    from app.phone_sources import PublicPhoneSources
    from app.phone_names import register_phone,parse_phone_names
    monkeypatch.setattr('app.phone_sources.validate_public_url',lambda url:None)
    phone=register_phone(parse_phone_names('iPhone 15 Pro Max')[0],{})
    html='<title>iPhone 15 Pro Max报价</title><main><p>iPhone 15 Pro Max 256GB参考价5999元，屏幕支持120Hz刷新率。</p></main><footer>国行手机报价导航</footer>'
    loader=PublicPhoneSources(tmp_path,fetcher=lambda url:html)
    result=loader.fetch_result(phone,'https://example.com/phone','版本价格')
    assert not any(s.price_options for s in result['sources'])
    assert all(s.market=='地区未确认' for s in result['sources'])


def test_ninth_tool_can_read_another_quote_without_cancelling_conditions(tmp_path):
    from tests.test_chat_agent import Model, tool, service, done, say
    from app.phone_names import checked_today
    catalog=load_catalog()
    phone=catalog['oneplus-13'].model_copy(deep=True)
    phone.price=None;phone.price_options=[]
    class Public:
        def search(self,p,query,**kwargs):
            number=query[-1]
            return [ChatSource(id='lead'+number,phone_id=p.id,title=p.name+'报价',
                quote=p.name+'版本价格线索，正文未核对',url='https://example.com/quote'+number,
                checked_on=checked_today(),source_type='snippet',evidence_kind='unknown')]
        def fetch(self,p,url,need,**kwargs):
            old=url.endswith('1');price=8599 if old else 7599
            return [ChatSource(id='body'+url[-1],phone_id=p.id,title=p.name+'版本价格',url=url,
                quote=f'{p.name} 人民币256GB '+('活动售价' if old else '参考价')+f'{price}元',
                checked_on=checked_today(),published_on='2025-05-14' if old else checked_today(),
                source_type='page',market='中国大陆',evidence_kind='price',
                price_options=[{'variant':'256GB','price':price}])]
    text='预算8000元，续航优先，只考虑一加，推荐一款'
    model=Model([
        [tool('search_local_phones',query='一加 续航',limit=1),*[tool('search_phone_knowledge',phone_name=phone.name,query=q) for q in ['电池','续航','价格']]],
        [tool('web_search',phone_name=phone.name,query=phone.name+' 版本报价'+str(i)) for i in [1,2,3]],
        [tool('fetch_page',phone_name=phone.name,url='https://example.com/quote1')],
        [tool('fetch_page',phone_name=phone.name,url='https://example.com/quote2')],
        {'kind':'answer','candidate_names':[phone.name]},
    ],{text:{'budget_max':8000,'priorities':['续航'],'preferred_brands':['一加'],'recommendation_count':1}})
    assistant=service(tmp_path,model,catalog={phone.id:phone},source_loader=Public())
    result=say(assistant,text);reply=result.messages[-1]
    assert reply.agent_trace['metrics']['tool_calls']==9,(reply.content,reply.agent_trace['stop_reason'],reply.agent_trace['actions'][-1:])
    assert reply.agent_trace['stop_reason']=='model_finished',[(a.get('validation_feedback'),a.get('result',{}).get('candidate_state')) for a in reply.agent_trace['actions'] if a.get('validation_feedback') or a.get('name')=='fetch_page']
    assert any('7599' in block.text for block in reply.blocks)
    assert result.profile.budget_max==8000 and result.profile.priorities==['续航']
    assert result.profile.preferred_brands==['一加'] and result.profile.brand_scope=='only'
    assert reply.price_references==[]


def test_historical_price_references_exclude_snippets_wrong_owners_and_discount_amounts():
    from app.selection_prices import historical_price_references
    phone=load_catalog()['oneplus-13']
    source=ChatSource(id='old',phone_id=phone.id,title=phone.name+'历史报价',url='https://example.com',
        quote='一加13 256GB 京东活动价4999元，立减1000元。对比小米15 256GB 3999元。',
        checked_on='2026-10-25',published_on='2025-05-14',source_type='page',evidence_kind='price',market='中国大陆')
    refs=historical_price_references({phone.id:phone},[source])
    assert len(refs)==1
    assert '4999元' in refs[0].text and '3999元' not in refs[0].text and '1000元' not in refs[0].text
    assert historical_price_references({phone.id:phone},[source.model_copy(update={'source_type':'snippet'})])==[]


def test_agent_can_offer_historical_budget_match_as_explicitly_provisional(tmp_path):
    from tests.test_chat_agent import Model,tool,service,say
    from app.phone_names import checked_today
    phone=load_catalog()['oneplus-13'].model_copy(deep=True)
    phone.price=None;phone.price_options=[]
    class Public:
        def search(self,p,query,**kwargs):
            return [ChatSource(id='lead',phone_id=p.id,title=p.name+'报价',url='https://example.com/old',
                quote=p.name+'价格线索',checked_on=checked_today(),source_type='snippet',evidence_kind='unknown')]
        def fetch(self,p,url,need,**kwargs):
            return [ChatSource(id='old',phone_id=p.id,title=p.name+'历史促销',url=url,
                quote=p.name+' 人民币256GB 京东活动价7599元',checked_on=checked_today(),published_on='2025-05-14',
                source_type='page',market='中国大陆',evidence_kind='price',price_options=[{'variant':'256GB','price':7599}])]
    text='预算8000元，续航优先，只考虑一加，推荐一款'
    model=Model([
        [tool('search_local_phones',query='一加 续航',limit=1),tool('web_search',phone_name=phone.name,query=phone.name+'报价')],
        [tool('fetch_page',phone_name=phone.name,url='https://example.com/old')],
        {'kind':'answer','candidate_names':[phone.name]},
    ],{text:{'budget_max':8000,'priorities':['续航'],'preferred_brands':['一加'],'recommendation_count':1}})
    result=say(service(tmp_path,model,catalog={phone.id:phone},source_loader=Public()),text)
    reply=result.messages[-1]
    assert result.profile.candidate_ids==[phone.id] and reply.blocks
    assert len(reply.price_references)==1 and '7599元' in reply.price_references[0].text
    assert '2025-05-14' in reply.price_references[0].text and '当前价格' in reply.price_references[0].text
    assert '初步备选' in reply.content and '初步备选' in reply.blocks[0].text
    assert '现价' in reply.blocks[0].text
    assert '换品牌' not in reply.followup
    assert any(s.id in reply.price_references[0].evidence_ids for s in reply.sources)


def test_subsidy_cap_is_not_a_phone_price_and_final_decimal_price_is_preserved():
    from app.selection_prices import source_price_facts,historical_price_references
    phone=load_catalog()['redmi-k80']
    quote='REDMI K80 12+512GB，原价2899元，通过国补15%最高补贴500元，叠加优惠后到手价低至2392.33元。'
    source=ChatSource(id='old',phone_id=phone.id,title=phone.name+'促销',url='https://example.com',quote=quote,
        checked_on='2026-10-26',published_on='2025-02-19',source_type='page',market='中国大陆',evidence_kind='price')
    facts=source_price_facts(source,phone.name)
    assert [str(f['price']) for f in facts]==['2899','2392.33']
    references=historical_price_references({phone.id:phone},[source])
    assert '2392.33元' in references[0].text and '优惠条件价500元' not in references[0].text
    assert '12GB+512GB' in references[0].text


@pytest.mark.parametrize('prefix',['最高补贴','立减','节省','运费','定金','礼盒价值'])
def test_non_price_amounts_cannot_become_structured_offers(prefix):
    html=f'<title>一加13价格</title><main><p>一加13 256GB {prefix}1000元</p></main>'
    assert extract_price_options('一加 13',html)[0]==[]


def test_copyright_repeated_title_is_not_a_second_rounded_deal_price():
    html='<title>REDMI K80促销</title><main><p>REDMI K80 12+512GB 原价2899元，通过国补最高补贴500元，实付2392.33元。</p><p>本文属于原创文章，如若转载，请注明来源：REDMI K80 12+512GB到手2392元</p></main>'
    offers,quote=extract_price_options('REDMI K80',html)
    assert [(o.variant,o.price) for o in offers]==[('12GB+512GB',2899)]
    assert '本文属于原创' not in quote
