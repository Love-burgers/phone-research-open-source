"""Offline retrieval recall and abstention on explicitly fictional documents."""

import argparse
from importlib.resources import files
import json
from pathlib import Path
import sys

from app.knowledge import KnowledgeBase, build_index


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="虚构手机知识库召回与无资料检查；不访问网络")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise FileExistsError("评测输出已存在")
    base = KnowledgeBase(build_index(Path(str(files("app.knowledge_samples").joinpath("phones.jsonl")))))
    search = base.as_tool()
    cases = json.loads(files("evaluation").joinpath("knowledge_cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        hits = search.invoke({"phone_name": case["phone"], "need": case["need"]})
        actual, expected = {item["source_url"] for item in hits}, set(case["expected_urls"])
        recall = len(actual & expected) / len(expected) if expected else None
        precision = len(actual & expected) / len(actual) if actual else None
        passed = actual == expected and all(item["phone_name"] == case["phone"] for item in hits)
        rows.append({**case, "actual_urls": sorted(actual), "recall_at_4": recall, "precision_at_4": precision,
                     "correct_abstention": not hits if not expected else None, "passed": passed})
    result = {"dataset": "fictional_phone_knowledge", "method": "bm25", "cases": rows,
              "passed": all(row["passed"] for row in rows),
              "limitation": "虚构固定语料，只验证检索和机型隔离，不证明真实语义检索或回答质量。"}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(text + "\n")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
