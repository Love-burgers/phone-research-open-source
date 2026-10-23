"""Direct stage contracts; these do not evaluate a real model's understanding."""
import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.chat_agent import ChatAgentRun
from app.chat_answer import AnswerComposer
from app.chat_schemas import AnswerBlock, ChatSource
from app.config import get_settings
from app.knowledge import KnowledgeBase
from app.selection_catalog import DATA_DIR
from tests.test_chat_agent import Model, service


def prepared(tmp_path):
    assistant = service(tmp_path, Model([]))
    session = assistant.store.create()
    session.profile.candidate_ids = ['oneplus-13']
    session.profile.focus_phone_id = 'oneplus-13'
    state = {'session': session, 'message': '一加13电池容量', 'query': '电池容量',
             'search_mode': 'local', 'catalog': assistant.catalog, 'selected': ['oneplus-13'],
             'recommendation_turn': False, 'acknowledged': []}
    return assistant, state


def source(phone_id, **changes):
    return ChatSource(id='evidence', phone_id=phone_id, title='测试正文',
        url='https://example.com/spec', quote='电池容量6000mAh', checked_on='2026-10-22',
        source_type='page', evidence_kind='spec', **changes)


def test_registry_upgrades_preview_and_keeps_reference_id(tmp_path):
    assistant, turn = prepared(tmp_path)
    run = ChatAgentRun(assistant, turn)
    state = run.initialize({})
    pid = turn['selected'][0]
    preview = source(pid).model_copy(update={'id': 'preview', 'source_type': 'snippet'})
    state['sources'] = {preview.id: preview}
    returned, added = run.record_sources(state, pid, [source(pid)])
    assert added == 1 and returned[0]['id'] == 'preview'
    assert set(state['sources']) == {'preview'}
    assert state['sources']['preview'].source_type == 'page'
    assert run.record_sources(state, pid, [source(pid)])[1] == 0


def test_registry_preserves_distinct_local_chunks_from_one_page(tmp_path):
    assistant, turn = prepared(tmp_path)
    run = ChatAgentRun(assistant, turn)
    state = run.initialize({})
    first = source('oneplus-13').model_copy(update={'source_type':'knowledge', 'id':'chunk-a'})
    second = first.model_copy(update={'id':'chunk-b', 'quote':'屏幕120Hz'})
    _, added = run.record_sources(state, 'oneplus-13', [first, second])
    assert added == 2 and set(state['sources']) == {'chunk-a', 'chunk-b'}


def test_registry_rejects_other_phone_and_unknown_url_before_fetch(tmp_path):
    assistant, turn = prepared(tmp_path)
    run = ChatAgentRun(assistant, turn)
    state = run.initialize({})
    pid = turn['selected'][0]
    assert run.record_sources(state, pid, [source('other-phone')]) == ([], 0)
    result = run.execute(state, 'fetch_page', {'phone_name': run.catalog[pid].name, 'url': 'https://example.com/unseen'})
    assert result['status'] == 'local_only'
    assert state['fetch_calls'] == 0


@pytest.mark.parametrize('block_phone,evidence_phone,expected', [
    ('oneplus-13', 'oneplus-13', True), ('oneplus-13', 'other-phone', False),
    ('other-phone', 'oneplus-13', False),
])
def test_answer_validation_checks_both_candidate_and_citation_scope(tmp_path, block_phone, evidence_phone, expected):
    assistant, turn = prepared(tmp_path)
    turn.update(sources=[source('oneplus-13')], acknowledged=[])
    composer = AnswerComposer(assistant, turn)
    block = AnswerBlock(phone_id=block_phone, text='电池容量6000mAh。', evidence_ids=['evidence'])
    assert composer.validate_block(block, {'evidence': source(evidence_phone)}) is expected


def test_typed_settings_redact_secrets_and_read_current_environment(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY', 'secret-test-value')
    monkeypatch.setenv('CHAT_AGENT_ENABLED', 'false')
    assert not get_settings().chat_agent_enabled
    assert 'secret-test-value' not in repr(get_settings())
    monkeypatch.setenv('CHAT_VECTOR_MIN_SCORE', '0.47')
    assert get_settings().chat_vector_min_score == .47
    monkeypatch.setenv('CHAT_VECTOR_MIN_SCORE', '2')
    with pytest.raises(ValidationError):
        get_settings()


def test_cited_source_does_not_authorize_invented_parameter(tmp_path):
    assistant, turn = prepared(tmp_path)
    turn.update(sources=[source('oneplus-13')])
    composer = AnswerComposer(assistant, turn)
    block = AnswerBlock(phone_id='oneplus-13', text='电池容量7000mAh。', evidence_ids=['evidence'])
    assert not composer.validate_block(block, {'evidence': source('oneplus-13')})
    assert turn['answer_failure']['stage'] == 'parameter_evidence'


def test_parameter_check_normalizes_units_and_ignores_search_previews():
    from app.evidence_rules import unsupported_parameters
    body = source('oneplus-13').model_copy(update={'quote':'重量200克，厚度8毫米，容量1TB，电池6000mAh。'})
    assert not unsupported_parameters('200g，8mm，1024GB，6000 mAh。', [body])
    assert unsupported_parameters('6000mAh', [body.model_copy(update={'source_type':'snippet'})])


def test_parameter_check_distinguishes_rejected_requirement_from_claim():
    from app.evidence_rules import unsupported_parameters
    body = source('oneplus-13').model_copy(update={'quote':'存储版本为256GB。'})
    assert not unsupported_parameters('此版本256GB，不满足你至少512GB的要求。', [body], ['至少512GB'])
    assert unsupported_parameters('此版本支持512GB，满足你的要求。', [body], ['至少512GB'])


def test_bundled_corpus_is_portable_and_has_provenance(monkeypatch):
    monkeypatch.delenv('CHAT_KNOWLEDGE_INDEX_PATH', raising=False)
    base = KnowledgeBase.from_path(DATA_DIR/'selection_index.json', allow_embeddings=False)
    assert len({c.phone_name for c in base.index.chunks}) >= 80
    assert all(c.source_url.startswith('https://') and c.checked_on for c in base.index.chunks)
    assert base.embeddings is None and all(c.vector is None for c in base.index.chunks)
    assert base.as_tool().invoke({'phone_name': 'iPhone 16 Pro', 'need': '相机'})


def test_module_imports_are_acyclic_including_deferred_imports():
    directory = Path(__file__).parents[1]/'app'
    modules = {p.stem: p for p in directory.glob('*.py')}
    graph = {name: {n.module for n in ast.walk(ast.parse(p.read_text(encoding='utf-8')))
                   if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module in modules}
             for name, p in modules.items()}
    visited = set()
    def visit(name, active):
        assert name not in active, f'Import cycle: {active} -> {name}'
        if name in visited:
            return
        for child in graph[name]:
            visit(child, active + [name])
        visited.add(name)
    for name in graph:
        visit(name, [])
