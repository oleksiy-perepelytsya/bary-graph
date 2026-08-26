"""Minimal Telegram control-plane bot for цех — read-only observability.

Commands:
  /status  — graph counts by level, source, and author
  /digest  — everything written in the last 24h
  /help    — command list

Truth is read from MongoDB directly; opencode stdout is never trusted.
Long-polls getUpdates, so no webhook or public URL is needed.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27117/?directConnection=true")
DB_NAME = os.environ.get("MONGO_DB", "barygraph_poc")
COLL_NAME = os.environ.get("MONGO_COLLECTION", "barygraph")

API = f"https://api.telegram.org/bot{TOKEN}"


def _coll():
    return MongoClient(MONGO_URI)[DB_NAME][COLL_NAME]


def _fmt_counts(pairs):
    return "\n".join(f"  {k}: {v}" for k, v in pairs)


def status() -> str:
    coll = _coll()
    levels = list(coll.aggregate([
        {"$group": {"_id": {"doc_type": "$doc_type", "level": "$level"}, "n": {"$sum": 1}}},
        {"$sort": {"_id.doc_type": 1, "_id.level": -1}},
    ]))
    sources = Counter()
    authors = Counter()
    for d in coll.aggregate([
        {"$match": {"source": "structural"}},
        {"$group": {"_id": "$author", "n": {"$sum": 1}}},
    ]):
        authors[d["_id"] or "(anonymous)"] = d["n"]
    for d in coll.aggregate([
        {"$match": {"source": "inferred"}},
        {"$group": {"_id": None, "n": {"$sum": 1}}},
    ]):
        sources["pipeline (inferred)"] = d["n"]
    lines = ["levels:"]
    lines += [f"  {s['_id']['doc_type']}/L{s['_id']['level']}: {s['n']}" for s in levels]
    lines.append("sources:")
    lines += [f"  {k}: {v}" for k, v in sources.items()]
    lines.append("structural authors:")
    if authors:
        lines += [f"  {k}: {v}" for k, v in authors.most_common(20)]
        anon = authors.get("(anonymous)", 0)
        if anon:
            lines.append(f"  ⚠ anonymous SMBs present: {anon}")
    else:
        lines.append("  (none yet)")
    return "\n".join(lines)


def digest() -> str:
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    docs = list(_coll().find(
        {"created_at": {"$gte": since}},
        {"level": 1, "source": 1, "author": 1, "label": 1,
         "properties.word": 1, "cm1_id": 1},
    ).sort("created_at", -1).limit(30))
    if not docs:
        return "last 24h: nothing written"
    lines = [f"last 24h: {len(docs)} docs (showing ≤30)"]
    for d in docs:
        who = d.get("author") or d.get("source") or "?"
        what = d.get("label") or d.get("properties", {}).get("word") or ""
        kind = "SMB" if d.get("source") == "structural" else (
            "node" if d.get("label") and not d.get("cm1_id") else "edge/MB")
        lines.append(f"  [{kind}] L{d.get('level', '?')} by {who} {what}".rstrip())
    return "\n".join(lines)


def send(chat_id: str, text: str) -> None:
    with httpx.Client(timeout=30) as c:
        c.post(f"{API}/sendMessage", json={
            "chat_id": chat_id, "text": text[:4000],
            "parse_mode": "", "disable_web_page_preview": True,
        }).raise_for_status()


COMMANDS = {
    "/status": status,
    "/digest": digest,
}


def handle(text: str, chat_id: str) -> None:
    cmd = text.split()[0].split("@")[0].lower()
    fn = COMMANDS.get(cmd)
    reply = fn() if fn else (
        "commands:\n/status — counts by level/source/author\n/digest — last 24h writes\n/help — this text"
    )
    send(chat_id, reply)


def main() -> None:
    offset = 0
    while True:
        try:
            with httpx.Client(timeout=70) as c:
                r = c.get(f"{API}/getUpdates", params={
                    "timeout": 60, "offset": offset,
                    "allowed_updates": '["message"]',
                })
                updates = r.json().get("result", [])
            for u in updates:
                offset = u["update_id"] + 1
                msg = u.get("message", {})
                chat = msg.get("chat", {}).get("id")
                text = msg.get("text", "")
                if CHAT_ID and str(chat) != CHAT_ID:
                    continue
                try:
                    handle(text, str(chat))
                except Exception as e:
                    send(str(chat), f"error: {e}")
        except KeyboardInterrupt:
            raise
        except Exception:
            time.sleep(5)


if __name__ == "__main__":
    main()
