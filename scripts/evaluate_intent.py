"""Evaluate patch acceptance offline, or a bounded set of real model extractions."""

import argparse
from copy import deepcopy
from importlib.resources import files
import json
from pathlib import Path
import tempfile

from app.chat import SelectionAssistant
from app.chat_schemas import ChatSession, SelectionProfile
from app.chat_store import ChatStore
from app.model_api import ModelClient


class SuppliedPatch:
    available = True

    def __init__(self, patch):
        self.patch = patch

    def complete_json(self, messages):
        return deepcopy(self.patch)


class NoPublicSources:
    def collect(self, *args, **kwargs):
        raise AssertionError("Intent evaluation must not retrieve public sources")


def evaluate(cases, assistant, *, live=False, call_limit=5):
    rows, attempts, consecutive_failures = [], 0, 0
    for case in cases:
        if live and (not case.get("live") or attempts >= call_limit or consecutive_failures >= 2):
            continue
        if not live:
            assistant.model = SuppliedPatch(case["patch"])
        session = ChatSession(session_id="0" * 32, profile=SelectionProfile(**case.get("initial", {})))
        attempts += int(live)
        result = assistant.understand({"session": session, "message": case["message"]})
        actual = session.profile.model_dump()
        differences = {key: {"expected": value, "actual": actual[key]}
                       for key, value in case["expected"].items() if actual[key] != value}
        model_failed = result.get("model_failed", False)
        consecutive_failures = consecutive_failures + 1 if model_failed else 0
        rows.append({"id": case["id"], "passed": not differences and not model_failed,
                     "model_failed": model_failed, "differences": differences, "profile": actual})
    return {"mode": "real_model_extraction" if live else "supplied_correct_patch",
            "model_attempts": attempts, "cases": rows, "count": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "scope": "固定样本；离线仅验证接收逻辑，不能作为真实模型理解准确率"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--call-limit", type=int, choices=range(1, 6), default=5)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("输出文件已存在，请使用新文件名")
    model = ModelClient()
    if args.live and not model.available:
        parser.error("未配置模型；未发送请求")
    cases = json.loads(files("evaluation").joinpath("intent_cases.json").read_text(encoding="utf-8"))["cases"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        assistant = SelectionAssistant(ChatStore(Path(directory) / "sessions.sqlite3"),
                                       model_client=model, source_loader=NoPublicSources())
        report = evaluate(cases, assistant, live=args.live, call_limit=args.call_limit)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({key: report[key] for key in ("mode", "count", "passed", "model_attempts")}))
    return int(report["passed"] != report["count"])


if __name__ == "__main__":
    raise SystemExit(main())
