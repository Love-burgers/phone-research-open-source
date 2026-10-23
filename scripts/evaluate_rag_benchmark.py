"""Frozen chunk-level BM25/vector/RRF comparison and phone-filter ablation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from langchain_core.embeddings import Embeddings

from app.knowledge import KnowledgeIndex, PhoneKnowledgeRetriever, load_index, save_index


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
METHODS = ("bm25", "vector", "hybrid")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rank_metrics(actual, relevant, k=4):
    """Binary chunk relevance. Empty gold is evaluated separately, never Recall=1."""
    actual, relevant = list(dict.fromkeys(actual))[:k], set(relevant)
    if not relevant:
        return {"recall": None, "precision": None, "mrr": None, "ndcg": None, "empty": not actual}
    overlap = len(set(actual) & relevant)
    first = next((i for i, cid in enumerate(actual, 1) if cid in relevant), None)
    dcg = sum(1 / math.log2(i + 1) for i, cid in enumerate(actual, 1) if cid in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return {"recall": overlap / len(relevant), "precision": overlap / k,
            "mrr": 1 / first if first else 0.0, "ndcg": dcg / ideal, "empty": None}


class FrozenQueryEmbeddings(Embeddings):
    def __init__(self, mapping):
        self.mapping = mapping

    def embed_documents(self, texts):
        raise AssertionError("Evaluation must reuse the frozen document vectors")

    def embed_query(self, text):
        return self.mapping[text]


def resolve_cases(cases, manifest, index):
    labels = {(s["phone"], s["topic"]): s["chunk_id"] for s in manifest["sections"]}
    chunk_ids = {c.chunk_id for c in index.chunks}
    ids = set()
    resolved = []
    for case in cases:
        if case["id"] in ids:
            raise ValueError("Duplicate question ID")
        ids.add(case["id"])
        relevant = [labels[tuple(pair)] for pair in case["relevant"]]
        if not set(relevant) <= chunk_ids or len(relevant) != len(set(relevant)):
            raise ValueError("Invalid relevance labels")
        if case["phone"] and any(phone != case["phone"] for phone, _ in case["relevant"]):
            raise ValueError("Single-phone gold contains another model")
        resolved.append({**case, "relevant_chunk_ids": relevant})
    return resolved


def aggregate(rows):
    positive, negative = [r for r in rows if r["relevant_chunk_ids"]], [r for r in rows if not r["relevant_chunk_ids"]]
    result = {"questions": len(rows), "positive": len(positive), "negative": len(negative)}
    for k in (1, 3, 4):
        for name in ("recall", "precision", "mrr", "ndcg"):
            result[f"{name}@{k}"] = mean(r["metrics"][str(k)][name] for r in positive) if positive else None
    result["negative_empty_rate"] = mean(r["metrics"]["4"]["empty"] for r in negative) if negative else None
    result["wrong_phone_fraction@4"] = mean(r["wrong_phone_fraction"] for r in rows if r["phone"]) if any(r["phone"] for r in rows) else None
    result["mean_candidate_count"] = mean(r["candidate_count"] for r in rows)
    result["candidates_le_k"] = sum(r["candidate_count"] <= 4 for r in rows)
    result["mean_rank_ms"] = mean(r["rank_ms"] for r in rows)
    return result


def paired_bootstrap(left, right, samples=2000):
    # Descriptive paired resampling of this dataset, not population accuracy inference.
    left = {r["id"]: r for r in left if r["relevant_chunk_ids"]}
    deltas = [r["metrics"]["4"]["recall"] - left[r["id"]]["metrics"]["4"]["recall"]
              for r in right if r["id"] in left]
    if not deltas:
        return None
    rng = random.Random(20261021)
    means = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
    return {"metric": "recall@4", "paired_questions": len(deltas), "delta": mean(deltas),
            "descriptive_bootstrap_95": [means[int(samples * .025)], means[int(samples * .975)]],
            "seed": 20261021, "resamples": samples}


def evaluate(index, cases, embeddings):
    runs, summaries = {}, {}
    for method in METHODS:
        for filtered in (True, False):
            key, rows = f"{method}-{'filtered' if filtered else 'unfiltered'}", []
            for case in cases:
                use_filter = bool(case["phone"]) and filtered
                retriever = PhoneKnowledgeRetriever(index=index, phone_name=case["phone"] or "",
                    filter_phone=use_filter, embeddings=embeddings, search_method=method,
                    k=4, candidate_k=20, min_vector_score=0.35)
                started = perf_counter()
                docs = retriever.invoke(case["question"])
                elapsed = (perf_counter() - started) * 1000
                actual = [d.metadata["chunk_id"] for d in docs]
                wrong = sum(d.metadata["phone_name"] != case["phone"] for d in docs) if case["phone"] else 0
                rows.append({**case, "hits": [{"text": d.page_content, **d.metadata} for d in docs],
                    "metrics": {str(k): rank_metrics(actual, case["relevant_chunk_ids"], k) for k in (1, 3, 4)},
                    "rank_ms": round(elapsed, 4), "filter_applied": use_filter,
                    "candidate_count": sum(c.phone_name == case["phone"] for c in index.chunks) if use_filter else len(index.chunks),
                    "wrong_phone_fraction": wrong / len(docs) if docs else 0})
            runs[key] = rows
            summaries[key] = {"overall": aggregate(rows), "groups": {
                group: aggregate([r for r in rows if r["group"] == group]) for group in sorted({c["group"] for c in cases})}}
    return runs, summaries


def markdown(summary):
    text = ["# 冻结语料RAG对照（AI初始标注，人工复核待完成）", "",
            "| 方法 | 过滤 | Recall@1 | Recall@4 | MRR@4 | nDCG@4 | 单机型误召回占比 | 无资料空召回 |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, value in summary["methods"].items():
        row = value["overall"]
        text.append(f"| {key.split('-')[0]} | {key.split('-')[1]} | " + " | ".join(
            f"{row[n]:.3f}" for n in ("recall@1", "recall@4", "mrr@4", "ndcg@4", "wrong_phone_fraction@4", "negative_empty_rate")) + " |")
    text += ["", "52条有资料题参与排序指标；8条无资料题单独统计。跨机型题始终不做单机型过滤。",
             "这是固定官方规格片段的检索结果，不是回答准确率；无资料仍有召回不代表回答可以据此下结论。",
             "排名计时不含模型下载/编码，包含现有Retriever每次重建存储；不代表端到端服务耗时。", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-directory", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("evaluation/rag_benchmark_cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, default=Path("runs/model-cache"))
    parser.add_argument("--embed-local", action="store_true", help="Explicitly load/download the real local ONNX model")
    parser.add_argument("--vectors-from", type=Path, help="Reuse vectors from a previous frozen run; no model downloads")
    args = parser.parse_args()
    if not args.embed_local and not args.vectors_from:
        parser.error("Select --embed-local or --vectors-from; no model requests by default")
    args.output.mkdir(parents=True, exist_ok=False)
    base = load_index(args.corpus_directory / "index-bm25.json")
    manifest = json.loads((args.corpus_directory / "manifest.json").read_text(encoding="utf-8"))
    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = resolve_cases(dataset["cases"], manifest, base)
    fingerprint = {"corpus_sha256": sha(args.corpus_directory / "corpus.jsonl"),
                   "cases_sha256": sha(args.cases), "index_sha256": sha(args.corpus_directory / "index-bm25.json"),
                   "knowledge_source_sha256": sha(Path("app/knowledge.py")), "k": [1, 3, 4],
                   "vector_threshold": 0.35, "rrf_constant": 60, "candidate_k": 20,
                   "model": MODEL_NAME, "query_prefix": QUERY_PREFIX}
    write_json(args.output / "freeze.json", fingerprint)  # Written BEFORE scoring or embedding.
    if args.vectors_from:
        previous = json.loads((args.vectors_from / "freeze.json").read_text(encoding="utf-8"))
        if previous != fingerprint:
            raise ValueError("Vector reuse requires exactly the same frozen corpus, questions and retrieval implementation")
        index = load_index(args.vectors_from / "index-vector.json")
        encoded = json.loads((args.vectors_from / "query-vectors.json").read_text(encoding="utf-8"))
        model_record = json.loads((args.vectors_from / "embedding.json").read_text(encoding="utf-8"))
    else:
        from fastembed import TextEmbedding
        started = perf_counter()
        model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(args.model_cache), threads=4, providers=["CPUExecutionProvider"])
        docs = list(model.embed([c.text for c in base.chunks], batch_size=16))
        doc_seconds = perf_counter() - started
        started = perf_counter()
        queries = list(model.embed([QUERY_PREFIX + c["question"] for c in cases], batch_size=16))
        query_seconds = perf_counter() - started
        encoded = {c["question"]: v.tolist() for c, v in zip(cases, queries)}
        signature = json.dumps({"model": MODEL_NAME, "runtime": "fastembed-onnx-cpu", "query_prefix": QUERY_PREFIX}, sort_keys=True)
        index = KnowledgeIndex(method="vector", embedding_signature=signature,
            chunks=[c.model_copy(update={"vector": v.tolist()}) for c, v in zip(base.chunks, docs)])
        model_files = {str(p.relative_to(args.model_cache)): sha(p) for p in args.model_cache.rglob("*")
                       if p.is_file() and "bge-small-zh" in str(p) and ".cache" not in p.parts}
        model_record = {"real_pretrained_embeddings": True, "model": MODEL_NAME,
            "runtime": {name: importlib.metadata.version(name) for name in ("fastembed", "onnxruntime", "numpy")},
            "device": "CPU", "threads": 4, "batch_size": 16, "dimension": len(docs[0]),
            "document_encoding_seconds_including_load": doc_seconds, "query_encoding_seconds": query_seconds,
            "model_files_sha256": model_files, "external_embedding_api_requests": 0,
            "model_source": "https://huggingface.co/Qdrant/bge-small-zh-v1.5"}
    save_index(index, args.output / "index-vector.json")
    write_json(args.output / "query-vectors.json", encoded)
    write_json(args.output / "embedding.json", model_record)
    runs, summaries = evaluate(index, cases, FrozenQueryEmbeddings(encoded))
    for key, rows in runs.items():
        write_json(args.output / f"{key}.json", rows)
    summary = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "freeze": fingerprint,
        "phones": manifest["phone_count"], "chunks": len(index.chunks), "questions": len(cases),
        "groups": dict(Counter(c["group"] for c in cases)), "methods": summaries,
        "paired_recall_comparisons": {key: paired_bootstrap(runs["bm25-filtered"], runs[key])
            for key in ("vector-filtered", "hybrid-filtered")},
        "human_review": {"status": "pending", "completed": 0},
        "scope": "Codex initial relevance labels, frozen before scoring; official specs, no user-behavior or held-out production data"}
    write_json(args.output / "summary.json", summary)
    (args.output / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "phones": manifest["phone_count"], "chunks": len(index.chunks), "questions": len(cases)}))


if __name__ == "__main__":
    main()
