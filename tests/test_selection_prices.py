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
