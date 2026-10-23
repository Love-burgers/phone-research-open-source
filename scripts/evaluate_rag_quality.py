"""Fixed-corpus retrieval metrics and bounded real RAG answer review material."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import sys

from app.chat_prompts import ANSWER_INSTRUCTIONS
from app.chat_schemas import ModelAnswer
from app.knowledge import KnowledgeBase, KnowledgeIndex, build_index
from app.model_api import ModelClient
from app.selection_catalog import DATA_DIR


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def retrieval_metrics(actual_urls, relevant_urls):
    actual, relevant = set(actual_urls), set(relevant_urls)
    overlap = len(actual & relevant)
    return {"source_recall_at_4": overlap / len(relevant) if relevant else None,
            "source_precision_at_4": overlap / len(actual) if actual else None,
            "correct_empty_result": not actual if not relevant else None,
            "exact_source_match": actual == relevant}


def evaluate_retrieval(dataset):
    corpus_paths = [DATA_DIR / name for name in dataset["corpus"]]
    corpus_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in corpus_paths}
    index = KnowledgeIndex(chunks=[chunk for path in corpus_paths for chunk in build_index(path).chunks])
    search = KnowledgeBase(index).as_tool()
    rows = []
    for case in dataset["cases"]:
        hits = search.invoke({"phone_name": case["phone"], "need": case["question"]})
        actual = sorted({hit["source_url"] for hit in hits})
        rows.append({**case, "hits": hits, "actual_urls": actual,
                     **retrieval_metrics(actual, case["relevant_urls"])})
    positive = [row for row in rows if row["relevant_urls"]]
    negative = [row for row in rows if not row["relevant_urls"]]
    # Empty recall is zero for a positive question; undefined precision is zero
    # in positive macro precision, so a missed query cannot inflate the mean.
    return {"schema_version": 1, "dataset_sha256": digest(dataset), "corpus_sha256": corpus_hashes,
            "method": "bm25", "unit": "unique source URLs within up to four chunks per question",
            "question_count": len(rows), "positive_count": len(positive), "negative_count": len(negative),
            "macro_source_recall_at_4": sum(row["source_recall_at_4"] for row in positive) / len(positive),
            "macro_source_precision_at_4": sum(row["source_precision_at_4"] or 0 for row in positive) / len(positive),
            "correct_empty_count": sum(row["correct_empty_result"] is True for row in negative),
            "exact_source_match_count": sum(row["exact_source_match"] for row in rows), "cases": rows,
            "scope": "小规模固定档案来源级评测；不证明片段完整支持回答、当前事实或泛化质量。"}


def answer_messages(case):
    sources = [{"id": f"source-{i}", "phone_id": "p1", "title": hit["title"],
                "url": hit["source_url"], "quote": hit["text"], "source_type": "knowledge",
                **{key: hit[key] for key in ("checked_on", "published_on", "market", "evidence_kind",
                                            "reference_price", "price_variant") if key in hit}}
               for i, hit in enumerate(case["hits"], 1)]
    payload = {"本轮": case["question"], "条件": {"budget_max": case.get("budget"), "priorities": []},
               "机型": {"p1": case["phone"]}, "引用范围": {"p1": [s["id"] for s in sources]},
               "历史": [], "来源": sources}
    return [{"role": "system", "content": ANSWER_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], sources


def check_citations(raw, sources):
    draft = ModelAnswer.model_validate(raw)
    by_id = {source["id"]: source for source in sources}
    for block in draft.blocks:
        if block.phone_id != "p1" or any(eid not in by_id or by_id[eid]["phone_id"] != block.phone_id
                                       for eid in block.evidence_ids):
            raise ValueError("citation_scope_mismatch")
    return draft.model_dump()


def generate_answers(rows, output, client, call_limit=4):
    answers = []
    attempts = failures = 0
    for case in rows:
        if not case.get("live_review"):
            continue
        if attempts >= call_limit or failures >= 2:
            break
        messages, sources = answer_messages(case)
        record = {"case_id": case["id"], "question": case["question"], "reference": case["reference"],
                  "sources": sources, "messages": messages, "answer": None, "raw": None, "usage": None}
        if not sources:
            record["status"] = "no_evidence_no_call"
        else:
            attempts += 1
            try:
                raw = client.complete_json(messages)
                record["raw"] = raw
                record["answer"] = check_citations(raw, sources)
                record["status"] = "generated"
                failures = 0
            except Exception as error:
                record["status"] = "generation_or_validation_failed"
                record["error_type"] = type(error).__name__
                failures += 1
            record["usage"] = client.last_usage
        record["answer_sha256"] = digest(record["raw"])
        answers.append(record)
        (output / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"attempts": attempts, "automatic_retries": 0, "answers": answers}


def compare_retrieval(previous, current):
    if previous["dataset_sha256"] != current["dataset_sha256"] or previous["corpus_sha256"] != current["corpus_sha256"]:
        raise ValueError("比较必须使用相同问题标注与语料")
    return {key: {"before": previous[key], "after": current[key], "delta": current[key] - previous[key]}
            for key in ("macro_source_recall_at_4", "macro_source_precision_at_4", "correct_empty_count", "exact_source_match_count")}


def review_answers(answers, annotations):
    """Validate review provenance; semantic labels are supplied by the reviewer."""
    indexed = {answer["case_id"]: answer for answer in answers}
    reviewed = []
    seen = set()
    for item in annotations["cases"]:
        answer = indexed[item["case_id"]]
        if item["case_id"] in seen or item["answer_sha256"] != answer["answer_sha256"]:
            raise ValueError("重复评审或回答已变化，必须重新评审")
        seen.add(item["case_id"])
        if answer["status"] != "generated":
            raise ValueError("生成或结构校验失败不能记为回答评审通过")
        blocks = answer["answer"]["blocks"]
        if sorted(item["covered_blocks"]) != list(range(len(blocks))):
            raise ValueError("必须覆盖该回答的全部输出块")
        sources = {source["id"]: source for source in answer["sources"]}
        for claim in item["claims"]:
            block = blocks[claim["block_index"]]
            if not claim["text"] or claim["text"] not in block["text"]:
                raise ValueError("评审结论必须对应实际回答原文")
            if claim["status"] not in {"supported", "supported_inference", "unsupported", "contradicted"}:
                raise ValueError("未知语义评审标签")
            if claim["status"].startswith("supported") and not claim["evidence"]:
                raise ValueError("支持判断必须记录依据")
            for evidence in claim["evidence"]:
                eid = evidence["source_id"]
                if eid not in block["evidence_ids"] or eid not in sources:
                    raise ValueError("评审依据必须来自该块实际引用")
                if "field" in evidence:
                    if evidence["field"] not in {"market", "checked_on", "published_on", "reference_price", "price_variant"}:
                        raise ValueError("未知来源元数据字段")
                    if evidence["value"] != sources[eid].get(evidence["field"]):
                        raise ValueError("引用元数据与实际来源不一致")
                elif not evidence["quote"] or evidence["quote"] not in sources[eid]["quote"]:
                    raise ValueError("证据摘录必须存在于实际来源片段")
        if not item["claims"] or {claim["block_index"] for claim in item["claims"]} != set(range(len(blocks))):
            raise ValueError("评审不得跳过输出块中的结论")
        reviewed.append(item)
    if seen != set(indexed):
        raise ValueError("存在尚未评审的回答")
    claims = [claim for item in reviewed for claim in item["claims"]]
    return {"reviewer": annotations["reviewer"], "method": annotations["method"],
            "answer_count": len(reviewed), "claim_count": len(claims),
            "unsupported_claims": sum(claim["status"] in {"unsupported", "contradicted"} for claim in claims),
            "focus_issue_cases": [item["case_id"] for item in reviewed if item["focus_issues"]],
            "cases": reviewed,
            "scope": "脚本核对原文、哈希和来源归属；支持关系来自评审者判断，不自动证明语义正确。"}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="固定档案RAG质量评测，默认仅离线检索")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="显式生成4个小样本，至多4次模型请求，无自动重试")
    parser.add_argument("--compare", type=Path, help="相同问题与语料的旧retrieval.json")
    parser.add_argument("--review-file", type=Path, help="对已保存answers.json的逐条语义评审标注")
    args = parser.parse_args(argv)
    if args.review_file:
        if args.live or args.compare:
            parser.error("评审已保存回答时不重新调用模型或运行检索对照")
        answers_path = args.output_directory / "answers.json"
        review = review_answers(json.loads(answers_path.read_text(encoding="utf-8")),
                                json.loads(args.review_file.read_text(encoding="utf-8")))
        destination = args.output_directory / "answer_review.json"
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
        summary_path = args.output_directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update({"answer_review": "completed", "reviewer": review["reviewer"],
                        "reviewed_answers": review["answer_count"], "reviewed_claims": review["claim_count"],
                        "unsupported_claims": review["unsupported_claims"], "focus_issue_cases": review["focus_issue_cases"]})
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: review[key] for key in ("answer_count", "claim_count", "unsupported_claims", "focus_issue_cases")}, ensure_ascii=False))
        return 0
    client = ModelClient()
    if args.live and not client.available:
        parser.error("真实回答评测需要本项目模型配置")
    dataset = json.loads(files("evaluation").joinpath("rag_quality_cases.json").read_text(encoding="utf-8"))
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = evaluate_retrieval(dataset)
    if args.compare:
        result["comparison"] = compare_retrieval(json.loads(args.compare.read_text(encoding="utf-8")), result)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    (output / "retrieval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    generation = generate_answers(result["cases"], output, client) if args.live else None
    summary = {key: result[key] for key in ("question_count", "positive_count", "negative_count", "macro_source_recall_at_4",
                                          "macro_source_precision_at_4", "correct_empty_count", "exact_source_match_count")}
    summary.update({"live": args.live, "model_requests": generation["attempts"] if generation else 0,
                    "automatic_retries": 0, "answer_review": "pending" if generation else "not_run"})
    if generation:
        usage = [answer["usage"] for answer in generation["answers"]]
        summary["total_tokens"] = sum(item["total_tokens"] for item in usage) if usage and all(item and "total_tokens" in item for item in usage) else None
        summary["cost"] = None
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
