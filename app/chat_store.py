"""SQLite conversation persistence with optimistic revision checks."""

from pathlib import Path
from contextlib import closing, contextmanager
import sqlite3
import uuid

from .chat_schemas import ChatSession
from .selection_prices import sanitize_session_prices


class SessionNotFound(Exception):
    pass


class SessionConflict(Exception):
    pass


class ChatStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions "
                       "(id TEXT PRIMARY KEY, revision INTEGER NOT NULL, data TEXT NOT NULL)")

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            yield db

    def create(self) -> ChatSession:
        session = ChatSession(session_id=uuid.uuid4().hex)
        with self._connect() as db:
            db.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                       (session.session_id, session.revision, session.model_dump_json()))
        return session

    def load(self, session_id: str) -> ChatSession:
        with self._connect() as db:
            row = db.execute("SELECT data FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise SessionNotFound()
        return sanitize_session_prices(ChatSession.model_validate_json(row[0]))

    def save(self, session: ChatSession) -> ChatSession:
        old_revision = session.revision
        saved = session.model_copy(deep=True)
        saved.revision += 1
        # Keep the last 30 turns; confirmed conditions are stored separately.
        saved.messages = saved.messages[-60:]
        with self._connect() as db:
            result = db.execute("UPDATE sessions SET revision = ?, data = ? WHERE id = ? AND revision = ?",
                                (saved.revision, saved.model_dump_json(), saved.session_id, old_revision))
            if result.rowcount != 1:
                raise SessionConflict()
        return saved

    def delete(self, session_id: str) -> None:
        with self._connect() as db:
            if db.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount != 1:
                raise SessionNotFound()

    def record_knowledge_update(self, session, update):
        """Annotate a committed turn only; never overwrite a newer revision."""
        session.messages[-1].agent_trace['local_knowledge_update']=update
        with self._connect() as db:
            db.execute('UPDATE sessions SET data=? WHERE id=? AND revision=?',
                (session.model_dump_json(),session.session_id,session.revision))
