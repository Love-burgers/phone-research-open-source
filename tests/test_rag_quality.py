"""Check evaluation accounting and bounded generation without real services."""

import json

import pytest

from scripts.evaluate_rag_quality import compare_retrieval, generate_answers, retrieval_metrics, review_answers


def test_retrieval_metrics_count_missing_and_extra_sources():
    result = retrieval_metrics(["right", "wrong"], ["right", "missed"])
    assert result["source_recall_at_4"] == 0.5
    assert result["source_precision_at_4"] == 0.5
    assert not result["exact_source_match"]
    assert retrieval_metrics([], ["missed"])["source_recall_at_4"] == 0
    assert retrieval_metrics([], [])["correct_empty_result"] is True
    assert retrieval_metrics(["wrong"], [])["correct_empty_result"] is False


def question(number, hits=True):
    return {"id": f"case-{number}", "phone": "Demo phone", "question": "配置是什么？",
            "reference": "参考标注不作为模型答案提示", "live_review": True,
            "hits": [{"title": "演示", "source_url": "https://example.com", "text": "演示配置。"}] if hits else []}


class FakeClient:
    last_usage = {"total_tokens": 10}

    def __init__(self, mode="good"):
        self.calls = 0
        self.mode = mode

    def complete_json(self, messages):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        assert "reference" not in payload
        if self.mode == "error":
            raise TimeoutError("do not persist private-error-detail")
        return {"blocks": [{"phone_id": "p1", "text": "演示配置。",
                             "evidence_ids": ["unknown" if self.mode == "wrong-citation" else "source-1"]}]}


def test_generation_limit_and_empty_retrieval_do_not_request_model(tmp_path):
    client = FakeClient()
    result = generate_answers([question(0, False), *[question(i) for i in range(1, 7)]], tmp_path, client, call_limit=4)
    assert client.calls == result["attempts"] == 4
    assert result["answers"][0]["status"] == "no_evidence_no_call"
    assert len(result["answers"]) == 5
    assert (tmp_path / "answers.json").exists()


@pytest.mark.parametrize("mode", ["error", "wrong-citation"])
def test_failed_or_invalid_answers_stop_after_two_without_retry(tmp_path, mode):
    client = FakeClient(mode)
    result = generate_answers([question(i) for i in range(5)], tmp_path, client)
    assert client.calls == result["attempts"] == 2
    assert all(answer["status"] == "generation_or_validation_failed" for answer in result["answers"])
    assert "private-error-detail" not in (tmp_path / "answers.json").read_text(encoding="utf-8")


def test_comparison_refuses_changed_corpus_or_questions():
    old = {"dataset_sha256": "a", "corpus_sha256": {"docs": "b"}}
    with pytest.raises(ValueError):
        compare_retrieval(old, {**old, "dataset_sha256": "new"})
    with pytest.raises(ValueError):
        compare_retrieval(old, {**old, "corpus_sha256": {"docs": "new"}})


def test_review_rejects_changed_answer_unquoted_evidence_or_missing_cases(tmp_path):
    answers = generate_answers([question(1)], tmp_path, FakeClient())["answers"]
    claim = {"block_index": 0, "text": "演示配置。", "status": "supported",
             "evidence": [{"source_id": "source-1", "quote": "演示配置。"}]}
    item = {"case_id": "case-1", "answer_sha256": answers[0]["answer_sha256"], "covered_blocks": [0],
            "claims": [claim], "focus_issues": []}
    review = {"reviewer": "test", "method": "fixture", "cases": [item]}
    assert review_answers(answers, review)["claim_count"] == 1
    with pytest.raises(ValueError):
        review_answers(answers, {**review, "cases": [{**item, "answer_sha256": "changed"}]})
    with pytest.raises(ValueError):
        review_answers(answers, {**review, "cases": []})
    claim["evidence"][0]["quote"] = "来源里没有的内容"
    with pytest.raises(ValueError):
        review_answers(answers, review)
