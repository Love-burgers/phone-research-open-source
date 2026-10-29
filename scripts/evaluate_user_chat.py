"""Run user-authored questions through the actual conversation backend; preserve outputs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest
from app.chat_store import ChatStore
from app.model_api import ModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("evaluation/user_chat_cases.json"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="Explicitly allow the real model and public-source tools")
    parser.add_argument("--reload-between-turns", action="store_true", help="Recreate the service to verify persisted-session continuation")
    args = parser.parse_args()
    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    case = next(c for c in dataset["cases"] if c["id"] == args.case)
    if not args.live:
        print(json.dumps({"case": case, "status": "plan_only", "real_calls": 0}, ensure_ascii=False))
        return
    model = ModelClient()
    if not model.available:
        parser.error("No configured conversation model")
    args.output.mkdir(parents=True, exist_ok=False)
    service = SelectionAssistant(ChatStore(args.output / "sessions.sqlite3"), model_client=model)
    record = {"case_id": case["id"], "original_messages": case["messages"],
              "explicit_requirements": case["explicit_requirements"], "turns": [],
              "started_at": datetime.now(timezone.utc).isoformat(), "human_verdict": "pending"}
    sid = None
    for number, message in enumerate(case["messages"], 1):
        if args.reload_between_turns and number > 1:
            service = SelectionAssistant(ChatStore(args.output / "sessions.sqlite3"), model_client=model)
        try:
            session = service.turn(ChatRequest(message=message, session_id=sid))
        except Exception as error:
            record["failure"] = {"turn": number, "input": message, "error_type": type(error).__name__}
            (args.output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        sid = session.session_id
        turn = session.model_dump(mode="json")
        record["turns"].append(turn)
        (args.output / f"turn-{number}.json").write_text(json.dumps(turn, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    record["reload_between_turns"] = args.reload_between_turns
    (args.output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case": case["id"], "turns": len(record["turns"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
