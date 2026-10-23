"""Prepare human review of the actual home conversation backend, separate from research."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest
from app.chat_store import ChatStore
from app.model_api import ModelClient
from .evaluate_rag_benchmark import write_json


CHAT_REVIEW_MESSAGES = (
    "预算4000，主要打游戏和拍照，先看一加13，只用本地资料。",
    "拍照不在意了，主要打游戏，预算降到3500，苹果就算了，其他不变。只用已有资料。",
    "一加13连续玩原神一小时平均帧率、机身温度、耗电多少？没有实测就说不知道，只用本地资料。",
)


class LocalOnlySources:
    """Keep this human-review run bounded and repeatable; no web is silently consulted."""
    def search(self, *_args, **_kwargs):
        return {"sources": [], "candidates": []}

    def fetch(self, *_args, **_kwargs):
        return []

    def collect(self, *_args, **_kwargs):
        return []


class BoundedReviewModel(ModelClient):
    def __init__(self, limit=18):
        super().__init__()
        self.limit, self.attempts = limit, 0

    def charge(self):
        if self.attempts >= self.limit:
            raise ValueError("review_model_request_limit")
        self.attempts += 1

    def complete_json(self, *args, **kwargs):
        self.charge()
        return super().complete_json(*args, **kwargs)

    def invoke_tools(self, *args, **kwargs):
        self.charge()
        return super().invoke_tools(*args, **kwargs)


def review_turn(session, tid, previous_profile=None):
    messages = session["messages"]
    if len(messages) < 2 or messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
        raise ValueError("Expected an actual user/assistant turn")
    return {"id": tid, "user": messages[-2]["content"], "reply": messages[-1],
            "profile_before": previous_profile, "profile_after": session["profile"]}


def prepare(previous_directory, output, live=False):
    output.mkdir(parents=True, exist_ok=False)
    sessions, turns, before = [], [], None
    for number in (1, 2, 3):
        path = previous_directory / f"turn-{number}.json"
        session = json.loads(path.read_text(encoding="utf-8"))
        turns.append(review_turn(session, f"chat-memory-{number}", before))
        sessions.append({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        before = session["profile"]
    cases = [{"id": "conversation-memory", "title": "查资料 → 改预算 → 接着追问", "turns": turns,
              "provenance": "已有真实首页对话，原回答完整保留；实际随包资料，公网工具受控返回空"}]
    attempts = 0
    if live:
        model = BoundedReviewModel()
        if not model.available:
            raise ValueError("No configured chat model")
        service = SelectionAssistant(ChatStore(output / "review-sessions.sqlite3"),
                                     model_client=model, source_loader=LocalOnlySources())
        sid, before, turns = None, None, []
        for number, message in enumerate(CHAT_REVIEW_MESSAGES, 1):
            session = service.turn(ChatRequest(message=message, session_id=sid))
            sid = session.session_id
            raw = session.model_dump(mode="json")
            write_json(output / f"live-turn-{number}.json", raw)
            turns.append(review_turn(raw, f"chat-change-{number}", before))
            before = raw["profile"]
        attempts = model.attempts
        cases.append({"id": "conversation-change", "title": "修改用途与预算 → 排除品牌 → 缺资料时回答", "turns": turns,
                      "provenance": "本次真实调用首页SelectionAssistant.turn；只使用随包资料，未执行公网查询"})
    fingerprint = json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")
    packet = {"schema_version": 2, "review_kind": "home_chat", "packet_id": hashlib.sha256(fingerprint).hexdigest()[:24],
              "created_at": datetime.now(timezone.utc).isoformat(), "cases": cases,
              "instructions": "只需看懂用户这轮要求和助手实际回答。可只评一轮，问题描述可代替打分，不要求全部填写。"}
    write_json(output / "blind-packet.json", packet)
    write_json(output / "preparation.json", {"review_kind": "home_chat", "conversations": len(cases),
        "turns": sum(len(c["turns"]) for c in cases), "new_model_requests": attempts,
        "max_new_requests": 18 if live else 0, "automatic_retries": 0, "reused_records": sessions,
        "human_review_status": "pending", "scope": "Real homepage conversation backend with local archive; no research report endpoint"})
    return {"conversations": len(cases), "turns": sum(len(c["turns"]) for c in cases), "model_requests": attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-directory", type=Path, default=Path("runs/home-agent-20261021"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="At most 18 new model requests; local knowledge only")
    args = parser.parse_args()
    print(json.dumps(prepare(args.previous_directory, args.output, args.live), ensure_ascii=False))


if __name__ == "__main__":
    main()
