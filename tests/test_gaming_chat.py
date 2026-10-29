"""Gaming buying turns must use measured evidence and preserve price provenance."""
import json

import pytest

from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest, ChatSource
from app.chat_store import ChatStore
from app.knowledge import KnowledgeBase, KnowledgeIndex, build_index, load_index, save_index
from app.phone_names import parse_phone_names, register_phone
from app.phone_sources import PublicPhoneSources
from app.selection_catalog import DATA_DIR
from app import phone_sources


class NoPublic:
    def collect(self, *args, **kwargs):
        return []


def say(assistant, text, previous=None):
    return assistant.turn(ChatRequest(message=text, session_id=previous.session_id if previous else None))


def test_original_gaming_turn_gives_review_and_budget_direction_then_restores(tmp_path):
    assistant = SelectionAssistant(agent_enabled=False, store=ChatStore(tmp_path / 'sessions.sqlite3'), source_loader=NoPublic())
    first = say(assistant, 'iPhone17和小米17')
    result = say(assistant, '预算6000，主要游戏性能', first)
    reply = result.messages[-1]
    assert result.profile.candidate_ids == first.profile.candidate_ids
    assert result.profile.explicit_candidate_ids == first.profile.explicit_candidate_ids
    assert result.profile.budget_max == 6000 and result.profile.priorities == ['游戏']
    apple, xiaomi = reply.cards
    assert apple.price == 6799 and apple.variant == '256GB' and '超出' in apple.note
    assert apple.price_url == 'https://www.apple.com.cn/shop/buy-iphone/iphone-17'
    assert xiaomi.price is None and xiaomi.price_url is None
    assert any(s.evidence_kind == 'review' and s.published_on == '2025-09-26' for s in reply.sources)
    body = '\n'.join(block.text for block in reply.blocks)
    assert '优先核对' in body and '购买前先确认' in body and '6799元' in body and '超预算799元' in body
    assert '原神' in body and '30分钟' in body and '2025-09-26' in body
    assert '没有对应实测' not in reply.content
    assert '拍照' not in str(reply.suggestions) and '提高到5000' not in str(reply.suggestions)
    assert reply.suggestions[1] == '按预算6000重新推荐游戏手机'
    assert reply.suggestions[0].startswith('Xiaomi 17的游戏')
    restored = assistant.store.load(result.session_id)
    assert restored.messages[-1] == reply
    reloaded = SelectionAssistant(agent_enabled=False, store=assistant.store, source_loader=NoPublic())
    newer = say(reloaded, '这两款原神怎么样？', restored)
    assert newer.messages[-1].cards[0].price == 6799
    assert newer.profile.priorities == ['游戏']
    reselect = say(reloaded, reply.suggestions[1], newer)
    assert not reselect.profile.explicit_candidate_ids
    assert reselect.profile.budget_max == 6000 and reselect.profile.priorities == ['游戏']


def test_local_spec_does_not_block_measured_review_supplement(tmp_path):
    calls = []
    class Reviews:
        def collect(self, phone, need, *, refresh=False):
            calls.append((phone.name, need))
            return [ChatSource(id=phone.id + ':review', phone_id=phone.id, title=phone.name + '游戏实测',
                               url='https://example.com/review', quote=phone.name + ' 2025年游戏测试：原神30分钟平均60帧，样机条件不能保证今天表现。',
                               checked_on='2026-10-11', published_on='2025-01-01', evidence_kind='review', source_type='page')]
    assistant = SelectionAssistant(agent_enabled=False, store=ChatStore(tmp_path / 'sessions.sqlite3'), source_loader=Reviews())
    result = say(assistant, '预算6000，主要游戏性能，先看REDMI K80的原神帧率与发热实测')
    assert len(calls) == 1 and '游戏' in calls[0][1]
    reply = result.messages[-1]
    assert 'review' in {s.evidence_kind for s in reply.sources}
    assert '原神30分钟平均60帧' in reply.blocks[0].text
    assert '没有对应实测' not in reply.content


def test_gaming_model_receives_relevant_evidence_and_answer_stays_concise(tmp_path):
    payloads = []
    class Model:
        available = True
        def complete_json(self, messages):
            if messages[0]['content'].startswith('提取'):
                return {'quote': messages[-1]['content'], 'budget_max': 6000, 'priorities': ['游戏'],
                        'selected_phones': ['iPhone17', '小米17']}
            payload = json.loads(messages[-1]['content'])
            payloads.append(payload)
            assert all(source['evidence_kind'] in {'review', 'price'} for source in payload['来源'])
            blocks = []
            for phone_id, name in payload['机型'].items():
                for number in range(3):
                    blocks.append(dict(phone_id=phone_id, text=f'{name}已引用资料，段{number}',
                                       evidence_ids=payload['引用范围'][phone_id]))
            return {'blocks':blocks}
    class Specs:
        def collect(self, phone, *args, **kwargs):
            return [ChatSource(id=phone.id + ':spec', phone_id=phone.id, title=phone.name,
                               url='https://example.com/spec', quote=phone.name + '芯片配置：不是游戏实测。',
                               checked_on='2026-10-11', source_type='page', evidence_kind='spec')]
    assistant = SelectionAssistant(agent_enabled=False, store=ChatStore(tmp_path / 'sessions.sqlite3'), model_client=Model(), source_loader=Specs())
    result = say(assistant, 'iPhone17和小米17，预算6000，主要游戏性能')
    assert result.messages[-1].mode == 'AI对话' and len(result.messages[-1].blocks) == 2
    assert len(payloads) == 1
    # The full registry is retained; irrelevant spec prose stays out of the model request.
    assert all(':spec' not in source['id'] for source in payloads[0]['来源'])


@pytest.mark.parametrize('source_type,market,url,quote', [
    ('snippet', '中国大陆', 'https://www.apple.com.cn/shop/buy-iphone/iphone-99', 'iPhone 99 256GB 6799元'),
    ('page', '地区未确认', 'https://www.apple.com.cn/shop/buy-iphone/iphone-99', 'iPhone 99 256GB 6799元'),
    ('page', '中国大陆', 'https://apple.com.cn.evil.example/phone', 'iPhone 99 256GB 6799元'),
    ('page', '中国大陆', 'https://www.apple.com.cn/shop/buy-iphone/iphone-99', 'iPhone 99 256GB 16799元'),
])
def test_untrusted_or_mismatched_price_cannot_update_candidate(tmp_path, source_type, market, url, quote):
    class Prices:
        def collect(self, phone, *args, **kwargs):
            return [ChatSource(id=phone.id + ':price', phone_id=phone.id, title=phone.name,
                               url=url, quote=quote, checked_on='2026-10-11', source_type=source_type,
                               market=market, evidence_kind='price', reference_price=6799, price_variant='256GB')]
    assistant = SelectionAssistant(agent_enabled=False, store=ChatStore(tmp_path / 'sessions.sqlite3'), source_loader=Prices())
    result = say(assistant, 'iPhone99预算6000，主要游戏性能')
    assert result.messages[-1].cards[0].price is None
    assert result.discovered_phones[0].price_url is None


def test_gaming_public_lookup_reserves_review_and_price_fetches(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_sources, 'validate_public_url', lambda url: None)
    phone = register_phone(parse_phone_names('小米17')[0], {})
    queries, fetched = [], []
    hits = [dict(title='Xiaomi 17技术规格', url='https://www.mi.com/spec1', snippet='Xiaomi 17芯片游戏配置'),
            dict(title='Xiaomi 17技术规格', url='https://www.mi.com/spec2', snippet='Xiaomi 17芯片游戏配置'),
            dict(title='Xiaomi 17游戏实测', url='https://example.com/review', snippet='Xiaomi 17原神游戏测试'),
            dict(title='Xiaomi 17购买价格', url='https://www.mi.com/shop', snippet='Xiaomi 17售价')]
    def search(query, limit):
        queries.append(query)
        return hits
    def fetch(url):
        fetched.append(url)
        title = next(item['title'] for item in hits if item['url'] == url)
        body = 'Xiaomi 17 原神最高画质30分钟，平均60帧，最高温度40℃。' if url.endswith('review') else 'Xiaomi 17 屏幕120Hz；页面价格为5000元，实际优惠需核对。'
        return f'<title>{title}</title><meta property="article:published_time" content="2025-09-26T10:00:00+08:00"><main><p>{body}</p></main>'
    loader = PublicPhoneSources(tmp_path, searcher=search, fetcher=fetch)
    sources = loader.collect(phone, '预算6000，主要游戏性能')
    assert len(queries) == 3
    assert any('游戏 实测 帧率 发热' in query for query in queries)
    assert any('中国大陆 购买 价格' in query for query in queries)
    assert fetched[:2] == ['https://example.com/review', 'https://www.mi.com/shop']
    assert len(fetched) == 3
    review = next(source for source in sources if source.evidence_kind == 'review')
    assert review.source_type == 'page' and review.published_on == '2025-09-26'
    assert '平均60帧' in review.quote
    assert all(source.reference_price is None for source in sources)  # No guessed price parsing.
    loader.fetcher = lambda url: '<title>404</title>'
    snippets = loader.collect(phone, '预算6000，主要游戏性能', refresh=True)
    assert all(source.evidence_kind == 'unknown' and source.published_on is None for source in snippets)


def test_knowledge_provenance_survives_index_and_langchain_tool(tmp_path):
    index = build_index(DATA_DIR / 'selection_reviews.jsonl')
    path = tmp_path / 'index.json'
    save_index(index, path)
    base = KnowledgeBase(load_index(path))
    hits = base.as_tool().invoke({'phone_name': 'Xiaomi 17', 'need': '游戏 原神 发热'})
    assert hits and all(hit['evidence_kind'] == 'review' for hit in hits)
    assert all(hit['published_on'] == '2025-09-26' and hit['checked_on'] == '2026-10-11' for hit in hits)
    old = index.model_dump()
    for chunk in old['chunks']:
        for field in ('published_on', 'checked_on', 'market', 'evidence_kind', 'reference_price', 'price_variant'):
            chunk.pop(field)
    assert all(chunk.evidence_kind == 'unknown' for chunk in KnowledgeIndex.model_validate(old).chunks)
