"""Preserve a user-reported chat failure and prepare optional before/after review.

Reads saved actual turns only. Does not generate answers or fill human grades.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .prepare_chat_review import review_turn


FEEDBACK = [
    "主回复中的‘当前检索没有相关资料，不能给出配置或购买结论’和每次重复的价格复核说明可以收起；真正影响选择的未知信息仍应说明。",
    "问游戏性能时，候选框应显示游戏表现、高刷等匹配情况，而不是笼统的相关配置。",
    "预算7500却只推荐三四千甚至两千的机型；应主动找该预算价位的选择，不能只用预算作上限过滤旧目录。",
    "已经说玩原神、崩坏星穹铁道，现有资料没有实测时应尝试联网搜索，不要反复问玩什么游戏。",
    "说想要7000左右的手机后，应按新价位重新核对候选，并保留游戏和排除华为等条件。",
]


def prepare(before_store, case_file, case_id, after, output):
    case = next(c for c in json.loads(case_file.read_text(encoding="utf-8"))["cases"] if c["id"] == case_id)
    feedback = FEEDBACK if case_id != "user-001" else [*FEEDBACK[:3], "原题明确要一款，应收敛到一款，不用三款目录候选代替推荐。"]
    with sqlite3.connect(before_store.resolve().as_uri() + "?mode=ro", uri=True) as db:
        sessions = [json.loads(r[0]) for r in db.execute("SELECT data FROM sessions ORDER BY rowid DESC")]
    before = next(s for s in sessions if [m["content"] for m in s["messages"] if m["role"] == "user"][:len(case["messages"])] == case["messages"])
    output.mkdir(parents=True, exist_ok=False)
    write = lambda filename, value: (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write("before-session.json", before)
    turns = []
    for i in range(len(case["messages"])):
        snapshot = {**before, "messages": before["messages"][:2 * (i + 1)]}
        # Historical per-turn profiles were not saved: don't manufacture them.
        turns.append({**review_turn(snapshot, f"{case_id}-before-{i+1}"), "profile_after": None})
    cases = [{"id": case_id + "-before", "title": "修改前：用户报告的真实对话", "turns": turns,
              "provenance": "来自原会话数据库，回答未改写；没有当轮条件快照的地方显示未记录。", "reported_feedback": feedback}]
    record = json.loads((after / "result.json").read_text(encoding="utf-8"))
    if record["original_messages"] != case["messages"]:
        raise ValueError("After-run must use the user's original messages")
    previous, turns = None, []
    for i, snapshot in enumerate(record["turns"], 1):
        turns.append(review_turn(snapshot, f"{case_id}-after-{i}", previous))
        previous = snapshot["profile"]
    cases.append({"id": case_id + "-after", "title": "修改后：同样原话重新运行", "turns": turns,
                  "provenance": "真实调用对话助手和公开工具；未代填人工判断。网络无结果也原样保留。", "reported_feedback": feedback})
    fingerprint = json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()
    packet = {"schema_version": 2, "review_kind": "home_chat", "packet_id": hashlib.sha256(fingerprint).hexdigest()[:24],
              "created_at": datetime.now(timezone.utc).isoformat(), "cases": cases,
              "instructions": "用户报告的问题已经收录，可只评一轮或只写意见。修改前后对照不是盲审；不冒充独立人工评测结果。"}
    write("blind-packet.json", packet)
    write("reported-feedback.json", {"source": "用户本次文字反馈与四张截图", "feedback": feedback,
                                     "human_review_status": "pending", "after_run": str(after)})
    return {"turns": sum(len(c["turns"]) for c in cases), "human_review_status": "pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-store", type=Path, default=Path("runs/chat-sessions.sqlite3"))
    parser.add_argument("--cases", type=Path, default=Path("evaluation/user_chat_cases.json"))
    parser.add_argument("--case", default="user-002")
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.before_store, args.cases, args.case, args.after, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
