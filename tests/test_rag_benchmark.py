from copy import deepcopy
import json
import math

import pytest

from app.knowledge import KnowledgeChunk, KnowledgeIndex
from scripts.evaluate_rag_benchmark import (
    FrozenQueryEmbeddings, aggregate, evaluate, paired_bootstrap, rank_metrics, resolve_cases,
)
from scripts.prepare_rag_benchmark import sections
from scripts.prepare_rag_review import check_answer
from scripts.serve_rag_review import agreement, save_review, validate_submission


def chunk(cid, phone, text, vector):
    return KnowledgeChunk(phone_name=phone, title="测试规格", source_url="https://example.org/spec",
                          text=text, document_id=cid, chunk_id=cid, vector=vector)


def test_metrics_preserve_rank_penalize_missing_and_do_not_inflate_empty_gold():
    result = rank_metrics(["wrong", "a", "b", "c"], ["a", "b"], k=3)
    assert result["recall"] == 1
    assert result["precision"] == 2 / 3
    assert result["mrr"] == .5
    assert result["ndcg"] < 1
    assert rank_metrics([], ["a"])["recall"] == 0
    assert rank_metrics([], [])["recall"] is None
    assert rank_metrics([], [])["empty"] is True
    assert rank_metrics(["a"], [])["empty"] is False


def test_duplicate_chunks_do_not_inflate_recall_and_requested_k_is_precision_denominator():
    result = rank_metrics(["a", "a"], ["a"], k=4)
    assert result["recall"] == result["ndcg"] == 1
    assert result["precision"] == .25


def test_same_frozen_vectors_drive_real_retriever_filter_ablation_and_cross_phone():
    index = KnowledgeIndex(method="vector", embedding_signature="test-only",
        chunks=[chunk("a", "Phone A", "屏幕刷新率", [1., 0.]),
                chunk("b", "Phone A Pro", "屏幕刷新率", [.9, .1]),
                chunk("c", "Phone A", "包装清单", [0., 1.])])
    cases = [{"id": "one", "phone": "Phone A", "question": "屏幕刷新率", "group": "literal", "relevant_chunk_ids": ["a"]},
             {"id": "two", "phone": None, "question": "屏幕刷新率", "group": "cross_phone", "relevant_chunk_ids": ["a", "b"]}]
    runs, summaries = evaluate(index, cases, FrozenQueryEmbeddings({"屏幕刷新率": [1., 0.]}))
    assert len(runs) == 6
    assert all(hit["phone_name"] == "Phone A" for hit in runs["vector-filtered"][0]["hits"])
    assert runs["vector-unfiltered"][0]["wrong_phone_fraction"] > 0
    assert not runs["vector-filtered"][1]["filter_applied"]
    assert runs["vector-filtered"][1]["hits"] == runs["vector-unfiltered"][1]["hits"]
    assert summaries["bm25-filtered"]["overall"]["positive"] == 2


def test_resolve_cases_rejects_misassigned_single_phone_gold():
    index = KnowledgeIndex(chunks=[chunk("a", "Phone A", "text", None)])
    manifest = {"sections": [{"phone": "Phone A", "topic": "screen", "chunk_id": "a"}]}
    case = {"id": "x", "phone": "Phone B", "relevant": [["Phone A", "screen"]]}
    with pytest.raises(ValueError, match="another model"):
        resolve_cases([case], manifest, index)


def test_paired_delta_uses_same_question_ids_and_reports_negative_effect():
    left = [{"id": "x", "relevant_chunk_ids": ["a"], "metrics": {"4": {"recall": 1}}}]
    right = [{"id": "x", "relevant_chunk_ids": ["a"], "metrics": {"4": {"recall": 0}}}]
    assert paired_bootstrap(left, right, samples=100)["delta"] == -1


def test_snapshot_parser_stops_at_section_boundaries_and_excludes_footnote_numbers():
    html = '<h3>屏幕<sup>1</sup></h3><p>120Hz</p><h3>电池</h3><p>5000mAh</p>'
    found = sections(html, "apple")
    assert "5000mAh" not in found[0][1]
    assert "120Hz" in found[0][1]


def packet():
    return {"packet_id": "blind", "cases": [{"id": "Q01", "passages": [{"id": "P1"}, {"id": "P2"}],
                                              "generated_answer": {}}]}


def submission():
    return {"packet_id": "blind", "reviewer": "test-only", "independent_attestation": True,
            "labels": {"P1": "2", "P2": "?"},
            "answers": {"Q01": {"citation_support": "yes", "question_focus": "no", "respects_gaps": "?"}}}


def test_human_review_cannot_be_complete_without_independent_attestation_and_all_items():
    data = submission()
    assert validate_submission(packet(), data)["status"] == "complete"
    data["independent_attestation"] = False
    assert validate_submission(packet(), data)["status"] == "partial"
    del data["labels"]["P1"]
    assert validate_submission(packet(), data)["reviewed_passages"] == 1


@pytest.mark.parametrize("change", [
    {"packet_id": "different"}, {"labels": {"injected": "2"}},
    {"labels": {"P1": "approved"}}, {"reviewer": ""},
    {"answers": {"other": {}}},
])
def test_human_review_rejects_other_packets_unknown_ids_and_bad_grades(change):
    data = submission()
    data.update(change)
    with pytest.raises(ValueError):
        validate_submission(packet(), data)


def test_agreement_excludes_uncertain_and_is_not_automatic_human_accuracy():
    initial = {"a": {"supports_core": True}, "b": {"supports_core": False}, "c": {"supports_core": True}}
    result = agreement(initial, {"a": "2", "b": "1", "c": "?"})
    assert result["compared"] == 2 and result["agreement"] == 1
    assert result["cohen_kappa"] == 1
    assert agreement(initial, {"c": "?"})["agreement"] is None


def test_human_saves_preserve_history_instead_of_rewriting_previous_judgments(tmp_path):
    key = {"initial_labels": {"P1": {"supports_core": True}, "P2": {"supports_core": False}}}
    (tmp_path / "private-ai-key.json").write_text(json.dumps(key), encoding="utf-8")
    save_review(tmp_path, packet(), submission())
    updated = submission()
    updated["labels"]["P1"] = "0"
    save_review(tmp_path, packet(), updated)
    assert len(list((tmp_path / "human-submissions").glob("*.json"))) == 2
    assert json.loads((tmp_path / "human-agreement.json").read_text("utf-8"))["agreement"] == 0


def test_answer_schema_and_citation_validation_is_separate_from_semantics():
    raw = {"blocks": [{"phone_id": "p1", "text": "an unsupported claim", "evidence_ids": ["e2"]}]}
    with pytest.raises(ValueError, match="cross_phone"):
        check_answer(raw, [{"id": "e2", "phone_id": "p2"}])
    assert check_answer(raw, [{"id": "e2", "phone_id": "p1"}]) == raw
