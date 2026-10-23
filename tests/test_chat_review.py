import json

import pytest

from scripts.prepare_chat_review import review_turn
from scripts.serve_rag_review import save_review, validate_chat_submission, validate_submission


def packet():
    return {"packet_id": "chat-only", "review_kind": "home_chat", "cases": [
        {"id": "dialogue", "turns": [{"id": "turn-1"}, {"id": "turn-2"}, {"id": "turn-3"}]}]}


def submission():
    return {"packet_id": "chat-only", "reviewer": "test-only", "independent_attestation": True,
            "turn_reviews": {"turn-2": {"note": "The changed budget is missing."}}}


def test_one_turn_note_completes_a_sample_without_grading_every_turn_or_dimension():
    result = validate_chat_submission(packet(), submission())
    assert result["status"] == "sample_complete"
    assert result["reviewed_turns"] == 1
    assert result["total_turns"] == 3
    assert "turn-1" not in result["turn_reviews"]
    assert "citation_support" not in result["turn_reviews"]["turn-2"]


def test_visiting_a_turn_and_leaving_an_empty_note_is_not_a_judgment():
    data = submission()
    data["turn_reviews"] = {"turn-1": {}, "turn-2": {"note": " "}}
    result = validate_chat_submission(packet(), data)
    assert result["reviewed_turns"] == 0
    assert result["status"] == "partial"


def test_uncertain_human_judgment_remains_uncertain_instead_of_passing():
    data = submission()
    data["turn_reviews"] = {"turn-1": {"overall": "?"}}
    result = validate_chat_submission(packet(), data)
    assert result["turn_reviews"]["turn-1"]["overall"] == "?"
    assert result["status"] == "sample_complete"


@pytest.mark.parametrize("change", [
    {"packet_id": "retrieval-only"}, {"turn_reviews": {"foreign-id": {"overall": "clear"}}},
    {"turn_reviews": {"turn-1": {"overall": "passed"}}}, {"reviewer": None},
    {"turn_reviews": {"turn-1": {"condition_memory": "always"}}},
])
def test_chat_reviews_do_not_accept_foreign_samples_or_invented_scores(change):
    data = submission()
    data.update(change)
    with pytest.raises(ValueError):
        validate_chat_submission(packet(), data)


def test_chat_and_retrieval_judgments_are_saved_separately(tmp_path):
    chat = tmp_path / "chat"
    retrieval = tmp_path / "retrieval"
    chat.mkdir()
    retrieval.mkdir()
    old = retrieval / "human-review.json"
    old.write_text('{"historical": true}', encoding="utf-8")
    result = save_review(chat, packet(), submission())
    assert result["reviewed_turns"] == 1
    summary = json.loads((chat / "human-summary.json").read_text("utf-8"))
    assert summary["review_kind"] == "home_chat"
    assert "accuracy" not in summary and "agreement" not in summary
    assert old.read_text("utf-8") == '{"historical": true}'


def test_export_keeps_actual_reply_and_state_without_rewriting_model_answers():
    session = {"profile": {"budget_max": 3500}, "messages": [
        {"role": "user", "content": "预算改3500"},
        {"role": "assistant", "content": "original failure", "blocks": [], "sources": []}]}
    result = review_turn(session, "t1", {"budget_max": 4000})
    assert result["reply"]["content"] == "original failure"
    assert result["profile_before"]["budget_max"] == 4000
    assert result["profile_after"]["budget_max"] == 3500


def test_optional_retrieval_review_can_finish_one_sample_without_marking_all_complete():
    old_packet = {"packet_id": "old", "cases": [{"id": "q", "passages": [{"id": "a"}, {"id": "b"}]}]}
    result = validate_submission(old_packet, {"packet_id": "old", "reviewer": "test-only",
        "labels": {"a": "?"}, "answers": {}, "finish_sample": True, "independent_attestation": True})
    assert result["status"] == "sample_complete"
    assert result["reviewed_passages"] == 1 and result["total_passages"] == 2
