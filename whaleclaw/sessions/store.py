"""SQLite-backed session and message persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import aiosqlite

from whaleclaw.config.paths import SESSIONS_DIR
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_DB_PATH = SESSIONS_DIR / "sessions.db"
_GROUP_COMPRESSION_MAX_GROUPS = 300
_PRESERVED_METADATA_KEYS = frozenset(
    {
        "group_compression_plan_levels",
        "group_compression_pending",
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    peer_id TEXT NOT NULL,
    model TEXT NOT NULL,
    thinking_level TEXT DEFAULT 'off',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_call_id TEXT,
    tool_name TEXT,
    timestamp TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, timestamp);

CREATE TABLE IF NOT EXISTS context_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    level TEXT NOT NULL,
    content TEXT NOT NULL,
    source_msg_start INTEGER NOT NULL,
    source_msg_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summaries_session
    ON context_summaries(session_id, level);

CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_usage_session
    ON token_usage(session_id);

CREATE TABLE IF NOT EXISTS session_group_compressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    group_idx INTEGER NOT NULL,
    level TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, group_idx, level)
);

CREATE INDEX IF NOT EXISTS idx_group_compress_session
    ON session_group_compressions(session_id, group_idx, level);
"""


class SessionRow:
    """Typed wrapper for a session row."""

    __slots__ = (
        "id",
        "channel",
        "peer_id",
        "model",
        "thinking_level",
        "metadata",
        "created_at",
        "updated_at",
    )

    def __init__(self, row: aiosqlite.Row) -> None:
        self.id: str = row[0]
        self.channel: str = row[1]
        self.peer_id: str = row[2]
        self.model: str = row[3]
        self.thinking_level: str = row[4]
        self.metadata: dict[str, object] = json.loads(row[5]) if row[5] else {}
        self.created_at: str = row[6]
        self.updated_at: str = row[7]


class MessageRow:
    """Typed wrapper for a message row."""

    __slots__ = (
        "id",
        "session_id",
        "role",
        "content",
        "tool_call_id",
        "tool_name",
        "timestamp",
        "metadata",
    )

    def __init__(self, row: aiosqlite.Row) -> None:
        self.id: int = row[0]
        self.session_id: str = row[1]
        self.role: str = row[2]
        self.content: str = row[3]
        self.tool_call_id: str | None = row[4]
        self.tool_name: str | None = row[5]
        self.timestamp: str = row[6]
        self.metadata: dict[str, object] = json.loads(row[7]) if row[7] else {}


class SummaryRow:
    """Typed wrapper for a context_summaries row."""

    __slots__ = (
        "id",
        "session_id",
        "level",
        "content",
        "source_msg_start",
        "source_msg_end",
        "token_count",
        "created_at",
    )

    def __init__(self, row: aiosqlite.Row) -> None:
        self.id: int = row[0]
        self.session_id: str = row[1]
        self.level: str = row[2]
        self.content: str = row[3]
        self.source_msg_start: int = row[4]
        self.source_msg_end: int = row[5]
        self.token_count: int = row[6]
        self.created_at: str = row[7]


class SessionStore:
    """Async SQLite store for sessions and messages."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        """Open the database and ensure schema exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.commit()
        log.debug("session_store.opened", path=str(self._db_path))

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            msg = "SessionStore not opened"
            raise RuntimeError(msg)
        return self._db

    async def save_session(
        self,
        *,
        session_id: str,
        channel: str,
        peer_id: str,
        model: str,
        thinking_level: str = "off",
        metadata: dict[str, object] | None = None,
        created_at: str,
        updated_at: str,
    ) -> None:
        await self._conn.execute(
            """INSERT OR REPLACE INTO sessions
               (id, channel, peer_id, model, thinking_level, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                channel,
                peer_id,
                model,
                thinking_level,
                json.dumps(metadata or {}),
                created_at,
                updated_at,
            ),
        )
        await self._conn.commit()

    async def get_session(self, session_id: str) -> SessionRow | None:
        cursor = await self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        return SessionRow(row) if row else None

    async def get_session_by_peer(self, channel: str, peer_id: str) -> SessionRow | None:
        cursor = await self._conn.execute(
            "SELECT * FROM sessions WHERE channel = ? AND peer_id = ?"
            " ORDER BY updated_at DESC LIMIT 1",
            (channel, peer_id),
        )
        row = await cursor.fetchone()
        return SessionRow(row) if row else None

    async def list_sessions(self) -> list[SessionRow]:
        cursor = await self._conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        return [SessionRow(r) for r in await cursor.fetchall()]

    async def get_message_counts(self) -> dict[str, int]:
        """Return {session_id: message_count} for all sessions."""
        cursor = await self._conn.execute(
            "SELECT session_id, COUNT(*) FROM messages GROUP BY session_id"
        )
        return {row[0]: row[1] for row in await cursor.fetchall()}

    async def delete_session(self, session_id: str) -> None:
        await self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self._conn.commit()

    async def update_session_field(
        self, session_id: str, **fields: object
    ) -> None:
        """Update arbitrary fields on a session row."""
        allowed = {"model", "thinking_level", "metadata", "updated_at"}
        parts: list[str] = []
        values: list[object] = []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k == "metadata" and isinstance(v, dict):
                v = json.dumps(v)
            parts.append(f"{k} = ?")
            values.append(v)
        if not parts:
            return
        values.append(session_id)
        sql = f"UPDATE sessions SET {', '.join(parts)} WHERE id = ?"  # noqa: S608
        await self._conn.execute(sql, values)
        await self._conn.commit()

    async def update_session_metadata(
        self,
        session_id: str,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        """Persist session metadata while preserving store-owned compression state."""
        merged = dict(metadata)
        current = await self.get_session(session_id)
        if current is not None:
            for key in _PRESERVED_METADATA_KEYS:
                if key not in merged and key in current.metadata:
                    merged[key] = current.metadata[key]
        await self.update_session_field(session_id, metadata=merged)
        return merged

    async def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        timestamp: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> int:
        ts = timestamp or datetime.now().isoformat()
        cursor = await self._conn.execute(
            """INSERT INTO messages
               (session_id, role, content, tool_call_id, tool_name, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, role, content, tool_call_id, tool_name, ts, json.dumps(metadata or {})),
        )
        await self._conn.commit()
        return cursor.lastrowid or 0

    async def get_messages(
        self, session_id: str, *, limit: int | None = None
    ) -> list[MessageRow]:
        sql = "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC"
        params: list[object] = [session_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        cursor = await self._conn.execute(sql, params)
        return [MessageRow(r) for r in await cursor.fetchall()]

    async def get_recent_messages(
        self, session_id: str, limit: int = 50
    ) -> list[MessageRow]:
        """Get the most recent N messages (returned in chronological order)."""
        sql = """SELECT * FROM (
                    SELECT * FROM messages WHERE session_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                 ) sub ORDER BY timestamp ASC"""
        cursor = await self._conn.execute(sql, (session_id, limit))
        return [MessageRow(r) for r in await cursor.fetchall()]

    async def delete_messages(self, session_id: str) -> None:
        await self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self._conn.commit()

    async def count_messages(self, session_id: str) -> int:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Context summaries (L0/L1) ──

    async def save_summary(
        self,
        *,
        session_id: str,
        level: str,
        content: str,
        source_msg_start: int,
        source_msg_end: int,
        token_count: int,
    ) -> int:
        """Persist an L0 or L1 summary. Returns the new row id."""
        ts = datetime.now().isoformat()
        cursor = await self._conn.execute(
            """INSERT INTO context_summaries
               (session_id, level, content, source_msg_start, source_msg_end,
                token_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, level, content, source_msg_start, source_msg_end,
             token_count, ts),
        )
        await self._conn.commit()
        return cursor.lastrowid or 0

    async def get_summaries(
        self, session_id: str, *, level: str | None = None,
    ) -> list[SummaryRow]:
        """Get summaries for a session, optionally filtered by level."""
        if level:
            cursor = await self._conn.execute(
                "SELECT * FROM context_summaries"
                " WHERE session_id = ? AND level = ?"
                " ORDER BY source_msg_start ASC",
                (session_id, level),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM context_summaries"
                " WHERE session_id = ?"
                " ORDER BY source_msg_start ASC",
                (session_id,),
            )
        return [SummaryRow(r) for r in await cursor.fetchall()]

    async def get_latest_summary(
        self, session_id: str, level: str,
    ) -> SummaryRow | None:
        """Get the most recent summary of the given level."""
        cursor = await self._conn.execute(
            "SELECT * FROM context_summaries"
            " WHERE session_id = ? AND level = ?"
            " ORDER BY source_msg_end DESC LIMIT 1",
            (session_id, level),
        )
        row = await cursor.fetchone()
        return SummaryRow(row) if row else None

    async def delete_summaries(self, session_id: str) -> None:
        """Delete all summaries for a session."""
        await self._conn.execute(
            "DELETE FROM context_summaries WHERE session_id = ?",
            (session_id,),
        )
        await self._conn.commit()

    # ── Token usage ──

    async def record_token_usage(
        self,
        *,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        ts = datetime.now().isoformat()
        await self._conn.execute(
            """INSERT INTO token_usage
               (session_id, model, input_tokens, output_tokens, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, model, input_tokens, output_tokens, ts),
        )
        await self._conn.commit()

    async def get_session_token_usage(self, session_id: str) -> dict[str, int]:
        """Return {input_tokens, output_tokens} totals for a session."""
        cursor = await self._conn.execute(
            "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0)"
            " FROM token_usage WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row:
            return {"input_tokens": row[0], "output_tokens": row[1]}
        return {"input_tokens": 0, "output_tokens": 0}

    async def get_total_token_usage(self) -> dict[str, int]:
        """Return global {input_tokens, output_tokens} totals."""
        cursor = await self._conn.execute(
            "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0)"
            " FROM token_usage"
        )
        row = await cursor.fetchone()
        if row:
            return {"input_tokens": row[0], "output_tokens": row[1]}
        return {"input_tokens": 0, "output_tokens": 0}

    async def get_token_usage_by_model(self) -> list[dict[str, object]]:
        """Return token usage grouped by model."""
        cursor = await self._conn.execute(
            "SELECT model, SUM(input_tokens), SUM(output_tokens), COUNT(*)"
            " FROM token_usage GROUP BY model ORDER BY SUM(input_tokens)+SUM(output_tokens) DESC"
        )
        return [
            {"model": r[0], "input_tokens": r[1], "output_tokens": r[2], "calls": r[3]}
            for r in await cursor.fetchall()
        ]

    # ── Group compression cache ──

    async def get_group_compression(
        self,
        *,
        session_id: str,
        group_idx: int,
        level: str,
        source_hash: str,
    ) -> str | None:
        cursor = await self._conn.execute(
            "SELECT content, source_hash FROM session_group_compressions"
            " WHERE session_id = ? AND group_idx = ? AND level = ? LIMIT 1",
            (session_id, group_idx, level),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        if row[1] != source_hash:
            return None
        return str(row[0])

    async def upsert_group_compression(
        self,
        *,
        session_id: str,
        group_idx: int,
        level: str,
        source_hash: str,
        content: str,
    ) -> None:
        now = datetime.now().isoformat()
        await self._conn.execute(
            """INSERT INTO session_group_compressions
               (session_id, group_idx, level, source_hash, content, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, group_idx, level)
               DO UPDATE SET source_hash=excluded.source_hash,
                             content=excluded.content,
                             updated_at=excluded.updated_at
            """,
            (session_id, group_idx, level, source_hash, content, now, now),
        )
        # Keep only latest N groups per session; evict oldest group indexes first.
        await self._conn.execute(
            """DELETE FROM session_group_compressions
               WHERE session_id = ?
                 AND group_idx IN (
                     SELECT group_idx FROM (
                         SELECT DISTINCT group_idx
                         FROM session_group_compressions
                         WHERE session_id = ?
                         ORDER BY group_idx DESC
                         LIMIT -1 OFFSET ?
                     )
               )""",
            (session_id, session_id, _GROUP_COMPRESSION_MAX_GROUPS),
        )
        await self._conn.commit()
