"""SQLite-backed admission and provider quotas, shared across service workers."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
from pathlib import Path
import sqlite3
from time import time
from uuid import uuid4

from .config import get_settings
from .tool_loop import ExecutionLimit


class QuotaExceeded(ExecutionLimit):
    def __init__(self, reason, retry_after=60):
        self.reason, self.retry_after = reason, retry_after
        super().__init__(reason)


class UsageGuard:
    def __init__(self, path, clock=time):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS counters (kind TEXT, key TEXT, value INTEGER NOT NULL, PRIMARY KEY(kind,key))')
            db.execute('CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, day TEXT, tokens INTEGER NOT NULL, settled INTEGER NOT NULL DEFAULT 0)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                yield db
        finally:
            db.close()

    def day(self):
        return datetime.fromtimestamp(self.clock(), timezone(timedelta(hours=8))).date().isoformat()

    @staticmethod
    def value(db, kind, key):
        row = db.execute('SELECT value FROM counters WHERE kind=? AND key=?', (kind,key)).fetchone()
        return row[0] if row else 0

    @staticmethod
    def add(db, kind, key, amount):
        db.execute('INSERT INTO counters VALUES (?,?,?) ON CONFLICT(kind,key) DO UPDATE SET value=value+excluded.value', (kind,key,amount))

    def admit(self, ip, settings):
        day, minute = self.day(), int(self.clock()//60)
        identity = hashlib.sha256(ip.encode()).hexdigest()
        minute_key = f'{minute}:{identity}'
        with self.connect() as db:
            for kind,key,limit,reason in [
                ('minute',minute_key,settings.chat_rate_limit_per_minute,'rate_limit'),
                ('turns',day,settings.chat_daily_turn_limit,'daily_turn_limit'),
            ]:
                if limit and self.value(db,kind,key) >= limit:
                    raise QuotaExceeded(reason)
            self.add(db,'minute',minute_key,1)
            self.add(db,'turns',day,1)
            # Minute buckets are operational counters, never conversation data.
            db.execute("DELETE FROM counters WHERE kind='minute' AND CAST(substr(key,1,instr(key,':')-1) AS INTEGER) < ?", (minute-2,))

    def reserve(self, tokens, settings):
        day, ticket = self.day(), uuid4().hex
        with self.connect() as db:
            if settings.chat_daily_model_calls and self.value(db,'model_calls',day) >= settings.chat_daily_model_calls:
                raise QuotaExceeded('daily_model_call_limit')
            if settings.chat_daily_token_limit and self.value(db,'tokens',day)+tokens > settings.chat_daily_token_limit:
                raise QuotaExceeded('daily_token_limit')
            self.add(db,'model_calls',day,1)
            self.add(db,'tokens',day,tokens)
            db.execute('INSERT INTO reservations(id,day,tokens) VALUES (?,?,?)',(ticket,day,tokens))
        return ticket

    def settle(self, ticket, actual_tokens):
        # Unknown usage keeps its reservation, including provider failures.
        if not isinstance(actual_tokens,int) or isinstance(actual_tokens,bool) or actual_tokens < 0:
            return
        with self.connect() as db:
            row = db.execute('SELECT day,tokens,settled FROM reservations WHERE id=?',(ticket,)).fetchone()
            if row and not row[2]:
                self.add(db,'tokens',row[0],actual_tokens-row[1])
                db.execute('UPDATE reservations SET settled=1 WHERE id=?',(ticket,))


@lru_cache(maxsize=8)
def guard_at(path):
    return UsageGuard(path)


def get_usage_guard(settings=None):
    settings = settings or get_settings()
    return guard_at(settings.chat_guard_path or settings.chat_storage_path+'.limits.sqlite3')


@contextmanager
def provider_quota(messages, max_output_tokens):
    import json
    settings = get_settings()
    # Daily admission uses a conservative UTF-8 reservation. Usage replaces it
    # exactly once; crashes/unknown usage retain the reservation across restarts.
    serializable = [m if isinstance(m,dict) else m.model_dump(mode='json') for m in messages]
    reservation = len(json.dumps(serializable,ensure_ascii=False).encode())+max_output_tokens
    guard = get_usage_guard(settings)
    ticket = guard.reserve(reservation,settings)
    usage = {}
    try:
        yield usage
    finally:
        guard.settle(ticket,usage.get('total_tokens'))
