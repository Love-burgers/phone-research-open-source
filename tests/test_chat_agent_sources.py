from app.phone_sources import PublicPhoneSources
from app.selection_catalog import load_catalog
import pytest


def test_agent_public_search_does_not_fetch_and_reuses_query_cache(tmp_path):
    calls, fetched = [], []
    def search(query, limit):
        calls.append(query)
        return [{"title": "一加 13规格", "url": "https://8.8.8.8/phone",
                 "snippet": "一加 13屏幕规格和刷新率资料。"}]
    loader = PublicPhoneSources(tmp_path, searcher=search, fetcher=lambda url: fetched.append(url))
    phone = load_catalog()["oneplus-13"]
    first = loader.search(phone, "一加 13 屏幕")
    second = loader.search(phone, "一加 13 屏幕")
    assert first['sources'] == second['sources'] and first['candidates'] == second['candidates']
    assert first['lookup_origin'] == 'network' and second['lookup_origin'] == 'cache'
    assert len(calls) == 1 and not fetched
    assert first["sources"][0].source_type == "snippet"
    loader.search(phone, "一加 13 屏幕", refresh=True)
    assert len(calls) == 2


def test_matching_title_is_fetch_candidate_without_promoting_anonymous_preview(tmp_path):
    loader = PublicPhoneSources(tmp_path, searcher=lambda *args: [{
        "title": "一加 13官方规格", "url": "https://8.8.8.8/spec", "snippet": "查看全部规格参数"}])
    result = loader.search(load_catalog()["oneplus-13"], "一加 13规格")
    assert not result["sources"]
    assert result["candidates"][0]["url"] == "https://8.8.8.8/spec"
    cached = loader.search(load_catalog()["oneplus-13"], "一加 13规格")
    assert cached['candidates'] == result['candidates'] and cached['sources'] == result['sources']
    assert cached['lookup_origin'] == 'cache'


def test_agent_fetch_keeps_same_source_id_and_reuses_body_cache(tmp_path):
    calls = []
    def fetch(url):
        calls.append(url)
        return '<title>一加 13屏幕规格</title><main><p>一加 13屏幕在此测试夹具里支持120Hz刷新率。</p></main>'
    loader = PublicPhoneSources(tmp_path, fetcher=fetch)
    phone = load_catalog()["oneplus-13"]
    first = loader.fetch(phone, "https://8.8.8.8/phone", "屏幕")
    assert loader.fetch(phone, "https://8.8.8.8/phone", "屏幕") == first
    assert len(calls) == 1 and first[0].source_type == "page"
    assert "120Hz" in first[0].quote
    loader.fetch(phone, "https://8.8.8.8/phone", "屏幕", refresh=True)
    assert len(calls) == 2


def test_agent_fetch_rejects_private_url_before_transport(tmp_path):
    loader = PublicPhoneSources(tmp_path, fetcher=lambda url: pytest.fail("private transport executed"))
    with pytest.raises(ValueError):
        loader.fetch(load_catalog()["oneplus-13"], "http://127.0.0.1/private", "屏幕")
