"""Create anonymous human-review material; AI judgments stay in a separate key."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

from app.chat_prompts import ANSWER_INSTRUCTIONS
from app.chat_schemas import ModelAnswer
from app.knowledge import load_index
from app.model_api import ModelClient
from .evaluate_rag_benchmark import sha, write_json


REVIEW_CASES = ("s02", "s13", "s26", "s36", "x01", "x04", "n01", "n05")
ANSWER_CASES = ("s26", "s36", "x01", "x04", "n01", "n05")


def check_answer(raw, sources):
    draft = ModelAnswer.model_validate(raw)
    by_id = {s["id"]: s for s in sources}
    for block in draft.blocks:
        if any(eid not in by_id or by_id[eid]["phone_id"] != block.phone_id for eid in block.evidence_ids):
            raise ValueError("cross_phone_or_unknown_citation")
    return draft.model_dump()


def generate_answers(rows, output, client, limit=6):
    """One request per preselected case, no retries; raw outputs are kept on failure."""
    attempts, failures, answers = 0, 0, []
    for case in rows:
        if case["id"] not in ANSWER_CASES or attempts >= limit or failures >= 2:
            continue
        phones = list(dict.fromkeys(hit["phone_name"] for hit in case["hits"]))
        phone_ids = {name: f"p{i}" for i, name in enumerate(phones, 1)}
        sources = [{"id": f"e{i}", "phone_id": phone_ids[h["phone_name"]], "phone_name": h["phone_name"],
                    "quote": h["text"], "title": h["title"], "url": h["source_url"],
                    "source_type": "knowledge", "market": h["market"], "checked_on": h["checked_on"],
                    "evidence_kind": h["evidence_kind"], "chunk_id": h["chunk_id"]}
                   for i, h in enumerate(case["hits"], 1)]
        payload = {"本轮": case["question"], "条件": {"budget_max": None, "priorities": []},
                   "机型": {v: k for k, v in phone_ids.items()}, "历史": [], "来源": sources,
                   "引用范围": {pid: [s["id"] for s in sources if s["phone_id"] == pid] for pid in phone_ids.values()}}
        messages = [{"role": "system", "content": ANSWER_INSTRUCTIONS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        record = {"case_id": case["id"], "question": case["question"], "sources": sources,
                  "messages": messages, "raw": None, "answer": None, "status": "no_sources"}
        if sources:
            attempts += 1
            started = perf_counter()
            try:
                raw = client.complete_json(messages, max_output_tokens=1200)
                record["raw"] = raw
                record["answer"] = check_answer(raw, sources)
                record["status"] = "schema_and_citation_valid"
                failures = 0
            except Exception as error:
                failures += 1
                record.update(status="generation_or_validation_failed", error_type=type(error).__name__)
            record.update(elapsed_seconds=perf_counter() - started, usage=client.last_usage)
        answers.append(record)
        write_json(output / "answers.json", answers)
    write_json(output / "answer-generation.json", {"attempts": attempts, "automatic_retries": 0,
        "schema_and_citation_valid": sum(a["status"] == "schema_and_citation_valid" for a in answers),
        "semantic_support": "pending independent human judgment", "retrieval_method": "hybrid-filtered",
        "scope": "RAG answer node with frozen retrieval; not a full live web/chat Agent evaluation"})
    return answers


def prepare_packet(results, corpus, output, answers):
    rng = random.Random(20261021)
    runs = {key: json.loads((results / f"{key}.json").read_text(encoding="utf-8")) for key in (
        "bm25-filtered", "vector-filtered", "hybrid-filtered", "bm25-unfiltered", "vector-unfiltered", "hybrid-unfiltered")}
    baseline = {r["id"]: r for r in runs["bm25-filtered"]}
    chunks = {c.chunk_id: c for c in load_index(corpus / "index-bm25.json").chunks}
    answer_map = {a["case_id"]: a for a in answers}
    cases, key = [], {"cases": {}, "initial_labels": {}}
    ordered = list(REVIEW_CASES)
    rng.shuffle(ordered)
    for number, cid in enumerate(ordered, 1):
        case, candidates = baseline[cid], set()
        for rows in runs.values():
            row = next(r for r in rows if r["id"] == cid)
            candidates.update(h["chunk_id"] for h in row["hits"])
        gold = case["relevant_chunk_ids"]
        # Include initial relevant passages and random pooled distractors. No method/rank/label is exposed.
        remaining = sorted(candidates - set(gold))
        rng.shuffle(remaining)
        selected = list(gold) + remaining[:max(0, 6 - len(gold))]
        rng.shuffle(selected)
        rid = f"Q{number:02d}"
        passages = []
        for i, chunk_id in enumerate(selected, 1):
            passage_id, c = f"{rid}-P{i}", chunks[chunk_id]
            passages.append({"id": passage_id, "phone": c.phone_name, "title": c.title,
                             "text": c.text, "url": c.source_url, "checked_on": c.checked_on})
            key["initial_labels"][passage_id] = {"chunk_id": chunk_id, "case_id": cid,
                                                  "supports_core": chunk_id in gold}
        public = {"id": rid, "question": case["question"], "passages": passages}
        key["cases"][rid] = cid
        if cid in answer_map:
            a = answer_map[cid]
            public["generated_answer"] = {"blocks": a["answer"]["blocks"] if a.get("answer") else [],
                                          "status": a["status"], "sources": a["sources"]}
        cases.append(public)
    packet = {"schema_version": 1, "packet_id": hashlib.sha256((sha(results / "freeze.json") + sha(corpus / "corpus.jsonl")).encode()).hexdigest()[:24],
        "created_at": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "instructions": "2=片段足以直接支持问题对该机型所问的核心事实（跨机型题可支持其中一款）；1=主题相关但不足以回答核心事实；0=不相关或型号错误；?=无法判断。不要从规格推断实测。"}
    write_json(output / "blind-packet.json", packet)
    write_json(output / "private-ai-key.json", key)
    return {"questions": len(cases), "passages": sum(len(c["passages"]) for c in cases),
            "answers": sum("generated_answer" in c for c in cases), "human_review": "pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--corpus-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-answers", action="store_true", help="At most six real model calls; no retries")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.results / "hybrid-filtered.json").read_text(encoding="utf-8"))
    answers = []
    if args.live_answers:
        client = ModelClient()
        if not client.available:
            raise ValueError("No configured answer model")
        answers = generate_answers(rows, args.output, client)
    print(json.dumps(prepare_packet(args.results, args.corpus_directory, args.output, answers)))


if __name__ == "__main__":
    main()
