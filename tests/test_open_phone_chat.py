from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest, ChatSource
from app.chat_store import ChatStore
from app.knowledge import KnowledgeBase, build_index
from app.selection_catalog import DATA_DIR
from app.phone_names import parse_phone_names, register_phone
from app import phone_sources
from app.phone_sources import PublicPhoneSources, extract_phone_excerpt, source_market, bing_target
import base64
import os
import time
import pytest


class PublicFake:
    def __init__(self, fail=()):
        self.calls, self.fail = [], fail

    def collect(self, phone, need, *, refresh=False):
        self.calls.append((phone.name, need, refresh))
        if phone.name in self.fail:
            raise TimeoutError('private-network-detail')
        return [ChatSource(id=phone.id + ':web-test', phone_id=phone.id, title=phone.name + '规格',
                           url='https://example.com/' + phone.id, quote=phone.name + '资料：屏幕配置；未核对售价。',
                           checked_on='2026-10-11', source_type='page')]


def say(service, message, previous=None):
    return service.turn(ChatRequest(message=message, session_id=previous.session_id if previous else None))


def service(tmp_path, fail=()):
    loader = PublicFake(fail)
    # These registry tests control all evidence rather than using archived reviews.
    knowledge = KnowledgeBase(build_index(DATA_DIR / 'selection_phones.jsonl'))
    return SelectionAssistant(agent_enabled=False, store=ChatStore(tmp_path / 'sessions.sqlite3'), source_loader=loader, knowledge=knowledge), loader


@pytest.mark.parametrize('text,names', [
    ('iPhone17和小米17', ['iPhone 17', 'Xiaomi 17']),
    ('iPhone１７和小米１７ T', ['iPhone 17', 'Xiaomi 17T']),
    ('K80 Pro和REDMI K80', ['REDMI K80 Pro', 'REDMI K80']),
    ('Xiaomi17T、小米17 Pro、iPhone17ProMax', ['Xiaomi 17T', 'Xiaomi 17 Pro', 'iPhone 17 Pro Max']),
    ('华为Mate80、vivo X200 Pro、三星S26', ['华为 Mate80', 'vivo X200 Pro', '三星 S26']),
])
def test_full_names_and_variants(text, names):
    assert [m.name for m in parse_phone_names(text)] == names


def test_original_input_enters_public_lookup_and_preserves_explicit_candidates(tmp_path):
    assistant, loader = service(tmp_path)
    first = say(assistant, 'iPhone17和小米17')
    ids = first.profile.candidate_ids
    assert {phone.name for phone in first.discovered_phones} == {'iPhone 17', 'Xiaomi 17'}
    assert {name for name, _, _ in loader.calls} == {'iPhone 17', 'Xiaomi 17'}
    assert len(first.messages[-1].blocks) == 2
    assert '暂不在' not in first.messages[-1].content
    assert first.profile.explicit_candidate_ids == ids
    assert all(card.price is None and card.price_url is None for card in first.messages[-1].cards)
    second = say(assistant, '预算提高到5000，主要拍照', first)
    assert second.profile.candidate_ids == ids
    assert second.profile.budget_max == 5000
    assert second.profile.priorities == ['拍照']
    assert '不能确认是否符合预算' in ' '.join(second.messages[-1].details)
    third = say(assistant, '第一款夜景怎么样？', second)
    assert third.profile.focus_phone_id == ids[0]
    fourth = say(assistant, '它的续航多久？', third)
    assert {s.phone_id for s in fourth.messages[-1].sources} == {ids[0]}
    assert fourth.profile.priorities == ['拍照']
    fifth = say(assistant, '另一款呢', fourth)
    assert fifth.profile.focus_phone_id == ids[1]
    sixth = say(assistant, '它的续航呢？', fifth)
    assert sixth.profile.priorities == ['拍照']
    seventh = say(assistant, '它的续航呢？预算还是5000，主要用途还是拍照。', sixth)
    assert seventh.profile.priorities == ['拍照']
    eighth = say(assistant, '这两款的续航相比呢？', seventh)
    assert eighth.profile.focus_phone_id is None
    assert {s.phone_id for s in eighth.messages[-1].sources} == set(ids)


def test_restore_aliases_urls_dates_and_isolation(tmp_path):
    assistant, loader = service(tmp_path)
    first = say(assistant, '小米17和iPhone17')
    restored = SelectionAssistant(agent_enabled=False, store=ChatStore(assistant.store.path), source_loader=loader)
    second = say(restored, '先只看Xiaomi17', first)
    assert second.profile.preferred_brands == []
    assert second.discovered_phones[0].source_urls
    assert second.discovered_phones[0].checked_on == '2026-10-11'
    assert len(second.discovered_phones) == 2
    other = say(restored, '预算4000，主要拍照')
    assert not other.discovered_phones
    assert not other.profile.explicit_candidate_ids
    assert all(phone.id not in restored.catalog for phone in first.discovered_phones)


@pytest.mark.parametrize('text', ['K80 Pro', 'iPhone16 Pro', '小米17', '小米17T Pro'])
def test_new_variant_never_borrows_bundled_base_evidence(tmp_path, text):
    assistant, loader = service(tmp_path)
    result = say(assistant, text)
    assert len(loader.calls) == 1
    assert all(s.source_type == 'page' for s in result.messages[-1].sources)
    assert result.messages[-1].cards[0].price is None


def test_single_failure_missing_phone_and_reselection(tmp_path):
    assistant, loader = service(tmp_path, fail=['Xiaomi 17'])
    first = say(assistant, 'iPhone17和小米17')
    reply = first.messages[-1]
    assert len(reply.sources) == 1
    assert len(reply.cards) == 2
    assert '未查到Xiaomi 17' in reply.content
    assert 'private-network-detail' not in reply.model_dump_json()
    second = say(assistant, '预算4000，主要续航，重新推荐', first)
    assert not second.profile.explicit_candidate_ids
    assert all(card.price is not None for card in second.messages[-1].cards)


def test_new_brand_filter_is_distinct_from_name_and_persists(tmp_path):
    assistant, _ = service(tmp_path)
    first = say(assistant, '华为Mate80')
    assert not first.profile.preferred_brands
    second = say(assistant, '只考虑华为品牌，预算4000，主要拍照', first)
    assert second.profile.preferred_brands == ['华为']
    assert second.profile.candidate_ids == first.profile.candidate_ids


def test_refresh_requests_public_sources_for_bundled_phone(tmp_path):
    assistant, loader = service(tmp_path)
    first = say(assistant, 'iPhone16')
    assert not loader.calls
    second = say(assistant, '刷新它的最新资料', first)
    assert loader.calls[-1][2] is True
    assert {'knowledge', 'page'} == {s.source_type for s in second.messages[-1].sources}


def test_limits_are_explained_without_server_failure(tmp_path):
    assistant, _ = service(tmp_path)
    first = say(assistant, '、'.join('小米' + str(n) for n in range(30, 36)))
    assert len(first.profile.candidate_ids) == 5
    assert '最多比较5款' in first.messages[-1].content
    second = say(assistant, '、'.join('小米' + str(n) for n in range(36, 51)), first)
    assert len(second.discovered_phones) == 20
    assert '已记录20款' in second.messages[-1].content


def test_new_focus_after_five_candidates_is_visible_and_persistent(tmp_path):
    assistant, _ = service(tmp_path)
    first = say(assistant, '、'.join('小米' + str(n) for n in range(30, 35)))
    second = say(assistant, '小米35', first)
    assert len(second.profile.explicit_candidate_ids) == 5
    assert second.profile.focus_phone_id in second.profile.candidate_ids
    assert second.messages[-1].cards[0].name == 'Xiaomi 35'
    third = say(assistant, '预算提高到6000', second)
    assert third.profile.candidate_ids == second.profile.candidate_ids
    fourth = say(assistant, '第五款屏幕怎么样？', third)
    assert fourth.profile.focus_phone_id == third.profile.candidate_ids[4]
    fifth = say(assistant, '第六款呢？', fourth)
    assert '没有这一个序号' in fifth.messages[-1].content


@pytest.mark.parametrize('html', [
    '<title>Xiaomi 17</title><nav>Xiaomi 17</nav><main><h1>404 页面不存在</h1></main>',
    '<title>Xiaomi 17 Pro</title><main><p>Xiaomi 17 Pro 电池容量很大，充电支持很多协议。</p></main>',
    '<title>手机规格</title><nav>Xiaomi 17</nav><main><p>5000mAh电池，120Hz屏幕，80W充电，页面参数比较丰富。</p></main>',
    '<title>Xiaomi 17</title><main><p>Xiaomi 17 Pro 支持大容量电池与充电规格，基础款没有资料。</p></main>',
])
def test_wrong_model_navigation_and_404_are_not_body_evidence(html):
    assert extract_phone_excerpt('Xiaomi 17', html, '续航 屏幕') is None


def test_dedicated_page_specs_and_comparison_scope():
    html = '<title>iPhone 17 - 技术规格</title><main><h1>iPhone 17</h1><p>显示屏采用120Hz刷新率，6.3英寸屏幕。</p><p>iPhone 17 Pro 电池容量与重量属于另一款手机。</p></main>'
    quote = extract_phone_excerpt('iPhone 17', html, '屏幕')
    assert '120Hz' in quote
    assert 'Pro' not in quote
    comparison = '<title>Xiaomi 17 和 Xiaomi 17 Pro 比较</title><main><p>Xiaomi 17 屏幕配置为本款说明，具体参数需要核对。</p><p>电池容量为5500mAh，充电80W，未指明型号。</p></main>'
    assert '5500' not in extract_phone_excerpt('Xiaomi 17', comparison, '续航 屏幕')


@pytest.mark.parametrize('url,title,expected', [
    ('https://www.apple.com/iphone-17/specs/', 'iPhone 17', '其他地区'),
    ('https://www.apple.com.cn/iphone-17/specs/', 'iPhone 17', '中国大陆'),
    ('https://www.mi.com/global/product/xiaomi-17/', 'Xiaomi 17', '其他地区'),
    ('https://example.com/review', '小米17 台湾版', '其他地区'),
    ('https://example.com/review', '小米17评测', '地区未确认'),
])
def test_market_variant_filter(url, title, expected):
    assert source_market(url, title) == expected


def test_bing_domain_boundary_and_bad_hit_does_not_hide_good_hit(monkeypatch):
    encoded = 'a1' + base64.urlsafe_b64encode(b'https://example.com/phone').decode().rstrip('=')
    assert bing_target('https://cn.bing.com/ck/a?u=' + encoded) == 'https://example.com/phone'
    malicious = 'https://evilbing.com/ck/a?u=' + encoded
    assert bing_target(malicious) == malicious
    def validate(url):
        if not url.startswith('https://example.com/'):
            raise ValueError()
    monkeypatch.setattr(phone_sources, 'validate_public_url', validate)
    monkeypatch.setattr(phone_sources, 'fetch_html', lambda url: '<li class="b_algo"><h2><a href="javascript:bad">bad</a></h2></li><li class="b_algo"><h2><a href="https://example.com/phone">Xiaomi 17</a></h2><p>Xiaomi 17 屏幕参数</p></li>')
    assert len(phone_sources.search_phone_web('Xiaomi 17')) == 1


def test_cache_refresh_source_types_and_malformed_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_sources, 'validate_public_url', lambda url: None)
    phone = register_phone(parse_phone_names('小米17')[0], {})
    calls = []
    def search(query, limit):
        calls.append(query)
        return [None, {'title':'Xiaomi 17 Pro','url':'https://example.com/pro','snippet':'Xiaomi 17 Pro 电池'},
                {'title':'Xiaomi 17','url':'https://www.mi.com/global/product/xiaomi-17/','snippet':'Xiaomi 17 电池'},
                {'title':'Xiaomi 17','url':'https://example.com/base','snippet':'Xiaomi 17 有屏幕相关配置，尚未核对售价。'}]
    loader = PublicPhoneSources(tmp_path, searcher=search, fetcher=lambda url: '<title>Xiaomi 17</title><main><h1>Xiaomi 17</h1><p>屏幕配置为120Hz，显示参数需要按实际版本核对。</p></main>')
    first = loader.collect(phone, '屏幕')
    assert len(first) == 1 and first[0].source_type == 'page'
    assert len(calls) == 2
    assert loader.collect(phone, '屏幕') == first
    assert len(calls) == 2
    loader.collect(phone, '屏幕', refresh=True)
    assert len(calls) == 4
    loader.fetcher = lambda url: '<title>404</title>'
    assert loader.collect(phone, '屏幕', refresh=True)[0].source_type == 'snippet'
    calls_before = len(calls)
    for path in tmp_path.glob('*.json'):
        os.utime(path, (time.time() - 7 * 3600, time.time() - 7 * 3600))
    loader.collect(phone, '屏幕')
    assert len(calls) == calls_before + 2


def test_no_data_preserves_requested_model_without_fabricating_price(tmp_path):
    assistant, _ = service(tmp_path, fail=['Xiaomi 999'])
    result = say(assistant, '小米999')
    assert result.messages[-1].cards[0].name == 'Xiaomi 999'
    assert not result.messages[-1].blocks
    assert result.messages[-1].mode == '资料模式'
    assert '没有相关资料' in ' '.join(result.messages[-1].details)


@pytest.mark.parametrize('include_citations,mode', [(True, 'AI对话'), (False, '资料模式（回答未通过核对）')])
def test_dynamic_model_mapping_and_mandatory_citations(tmp_path, include_citations, mode):
    import json
    assistant, _ = service(tmp_path)
    class Model:
        available = True
        def complete_json(self, messages):
            if messages[0]['content'].startswith('提取'):
                return {'quote': messages[-1]['content'], 'selected_phones': ['iPhone17', '小米17']}
            payload = json.loads(messages[-1]['content'])
            assert set(payload['机型'].values()) == {'iPhone 17', 'Xiaomi 17'}
            blocks = []
            for phone_id, name in payload['机型'].items():
                assert payload['引用范围'][phone_id]
                block = {'phone_id': phone_id, 'text': name + '：本轮有正文资料，售价未核对。'}
                if include_citations:
                    block['evidence_ids'] = payload['引用范围'][phone_id]
                blocks.append(block)
            return {'blocks': blocks}
    assistant.model = Model()
    result = say(assistant, 'iPhone17和小米17')
    assert result.messages[-1].mode == mode
    assert all(block.evidence_ids for block in result.messages[-1].blocks)
