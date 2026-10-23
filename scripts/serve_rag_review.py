"""Local-only blind human annotation UI. Never fills in human judgments itself."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import uuid

from .evaluate_rag_benchmark import write_json


ANSWER_FIELDS = ("citation_support", "question_focus", "respects_gaps")
CHAT_FIELDS = ("question_focus", "condition_memory", "citation_support", "respects_gaps")


def validate_chat_submission(packet, data):
    if not isinstance(data, dict) or data.get("packet_id") != packet["packet_id"]:
        raise ValueError("Wrong review packet")
    reviewer = data.get("reviewer", "")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 100:
        raise ValueError("Enter a reviewer alias")
    if not isinstance(data.get("independent_attestation", False), bool):
        raise ValueError("Invalid attestation")
    expected = {t["id"] for case in packet["cases"] for t in case["turns"]}
    reviews = data.get("turn_reviews", {})
    if not isinstance(reviews, dict) or not reviews.keys() <= expected:
        raise ValueError("Unknown conversation turn")
    checked = {}
    for tid, review in reviews.items():
        if not isinstance(review, dict) or not review.keys() <= {*CHAT_FIELDS, "overall", "note"}:
            raise ValueError("Invalid review fields")
        if any(review[k] not in {"yes", "no", "?", "na"} for k in CHAT_FIELDS if k in review):
            raise ValueError("Invalid dimension grade")
        if "overall" in review and review["overall"] not in {"clear", "problem", "?"}:
            raise ValueError("Invalid overall grade")
        note = review.get("note", "")
        if not isinstance(note, str) or len(note) > 1000:
            raise ValueError("Invalid review note")
        # An honest note alone is useful; no forced dozens of radio selections.
        if note.strip() or any(k in review for k in (*CHAT_FIELDS, "overall")):
            checked[tid] = review
    attested = data.get("independent_attestation", False)
    status = "sample_complete" if checked and attested else "partial"
    return {"schema_version": 2, "review_kind": "home_chat", "packet_id": packet["packet_id"],
            "reviewer": reviewer.strip(), "turn_reviews": checked, "independent_attestation": attested,
            "status": status, "reviewed_turns": len(checked), "total_turns": len(expected),
            "submitted_at": datetime.now(timezone.utc).isoformat()}


def validate_submission(packet, data):
    if not isinstance(data, dict) or data.get("packet_id") != packet["packet_id"]:
        raise ValueError("Wrong review packet")
    expected = {p["id"] for case in packet["cases"] for p in case["passages"]}
    expected_answers = {c["id"] for c in packet["cases"] if "generated_answer" in c}
    labels, answers = data.get("labels", {}), data.get("answers", {})
    if not isinstance(labels, dict) or not labels.keys() <= expected:
        raise ValueError("Unknown passage IDs")
    if any(v not in {"0", "1", "2", "?"} for v in labels.values()):
        raise ValueError("Invalid passage grade")
    if not isinstance(answers, dict) or not answers.keys() <= expected_answers:
        raise ValueError("Unknown answer IDs")
    for values in answers.values():
        if not isinstance(values, dict) or not values.keys() <= {*ANSWER_FIELDS, "note"}:
            raise ValueError("Invalid answer dimensions")
        if any(values.get(key, "?") not in {"yes", "no", "?", "na"} for key in ANSWER_FIELDS):
            raise ValueError("Invalid answer grade")
        if not isinstance(values.get("note", ""), str) or len(values.get("note", "")) > 1000:
            raise ValueError("Invalid review note")
    reviewer = data.get("reviewer", "").strip()
    if not reviewer or len(reviewer) > 100:
        raise ValueError("Enter a reviewer name or alias")
    if not isinstance(data.get("independent_attestation", False), bool):
        raise ValueError("Invalid attestation")
    answers_complete = sum(all(k in values for k in ANSWER_FIELDS) for values in answers.values())
    complete = len(labels) == len(expected) and answers_complete == len(expected_answers) and data.get("independent_attestation") is True
    sample_complete = bool(labels or answers) and data.get("finish_sample") is True and data.get("independent_attestation") is True
    return {"schema_version": 1, "packet_id": packet["packet_id"], "reviewer": reviewer,
            "labels": labels, "answers": answers, "independent_attestation": data.get("independent_attestation", False),
            "status": "complete" if complete else "sample_complete" if sample_complete else "partial", "reviewed_passages": len(labels),
            "total_passages": len(expected), "reviewed_answers": answers_complete,
            "total_answers": len(expected_answers), "submitted_at": datetime.now(timezone.utc).isoformat()}


def agreement(initial, labels):
    compared = [(initial[pid]["supports_core"], grade == "2") for pid, grade in labels.items() if grade != "?"]
    if not compared:
        return {"compared": 0, "agreement": None, "cohen_kappa": None}
    n = len(compared)
    observed = sum(a == b for a, b in compared) / n
    p_ai, p_human = sum(a for a, _ in compared) / n, sum(b for _, b in compared) / n
    chance = p_ai * p_human + (1 - p_ai) * (1 - p_human)
    return {"compared": n, "agreement": observed,
            "cohen_kappa": (observed - chance) / (1 - chance) if chance != 1 else None,
            "human_positive": sum(b for _, b in compared), "ai_positive": sum(a for a, _ in compared),
            "scope": "Binary core-support agreement on a pooled subset, not factual accuracy or full-corpus recall"}


def save_review(directory, packet, data):
    if packet.get("review_kind") == "home_chat":
        record = validate_chat_submission(packet, data)
        history = directory / "human-submissions"
        history.mkdir(exist_ok=True)
        write_json(history / f"{uuid.uuid4().hex}.json", record)
        write_json(directory / "human-review.json", record)
        write_json(directory / "human-summary.json", {
            "review_kind": "home_chat", "status": record["status"],
            "reviewed_turns": record["reviewed_turns"], "total_turns": record["total_turns"],
            "scope": "Human-selected conversational sample, not an exhaustive pass rate or RAG label agreement",
            "independent_attestation": record["independent_attestation"], "turn_reviews": record["turn_reviews"]})
        return {k: record[k] for k in ("status", "reviewed_turns", "total_turns")}
    record = validate_submission(packet, data)
    key = json.loads((directory / "private-ai-key.json").read_text(encoding="utf-8"))
    analysis = agreement(key["initial_labels"], record["labels"])
    analysis.update(status=record["status"], reviewer=record["reviewer"],
        independent_attestation=record["independent_attestation"],
        excluded_uncertain=sum(v == "?" for v in record["labels"].values()),
        disagreements=[{"passage_id": pid, **key["initial_labels"][pid], "human_grade": grade}
                       for pid, grade in record["labels"].items() if grade != "?" and (grade == "2") != key["initial_labels"][pid]["supports_core"]],
        answer_judgments=record["answers"])
    # Preserve every save; the latest pointers are convenience files, not an audit-history overwrite.
    history = directory / "human-submissions"
    history.mkdir(exist_ok=True)
    write_json(history / f"{uuid.uuid4().hex}.json", record)
    write_json(directory / "human-review.json", record)
    write_json(directory / "human-agreement.json", analysis)
    return {k: record[k] for k in ("status", "reviewed_passages", "total_passages", "reviewed_answers", "total_answers")}


def make_handler(directory):
    packet = json.loads((directory / "blind-packet.json").read_text(encoding="utf-8"))
    filename = "chat_review.html" if packet.get("review_kind") == "home_chat" else "rag_review.html"
    template = Path(__file__).resolve().parent.parent / "evaluation" / filename

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, content_type="application/json; charset=utf-8", status=200):
            raw = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/":
                return self.reply(template.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/packet":
                return self.reply(packet)
            if self.path == "/saved":
                path = directory / "human-review.json"
                return self.reply(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
            if self.path == "/status":
                path = directory / "human-review.json"
                record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "pending"}
                return self.reply({k: record[k] for k in ("status", "reviewed_passages", "total_passages", "reviewed_answers", "total_answers", "reviewed_turns", "total_turns") if k in record})
            return self.reply({"error": "not_found"}, status=404)

        def do_POST(self):
            if self.path != "/submit":
                return self.reply({"error": "not_found"}, status=404)
            host = self.headers.get("Host", "")
            if host != f"127.0.0.1:{self.server.server_port}" or self.headers.get("Origin") != f"http://{host}":
                return self.reply({"error": "origin_rejected"}, status=403)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 150000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                return self.reply(save_review(directory, packet, data))
            except (ValueError, TypeError, KeyError):
                return self.reply({"error": "invalid_review", "message": "请填写复核者代称，并检查评分。"}, status=400)

        def log_message(self, *_):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8140)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.directory.resolve()))
    print(f"Local human review: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
