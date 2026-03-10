"""Session lifecycle management."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import Message
from whaleclaw.sessions.store import SessionStore
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

MessagePersistHook = Callable[["Session", str, str], Awaitable[None] | None]


class Session(BaseModel):
    """In-memory session representation."""

    id: str
    channel: str
    peer_id: str
    messages: list[Message] = Field(default_factory=list)
    model: str
    thinking_level: str = "off"
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0


class SessionManager:
    """Manages session lifecycle backed by SQLite."""

    def __init__(self, store: SessionStore, config: WhaleclawConfig) -> None:
        self._store = store
        self._config = config
        self._message_persist_hook: MessagePersistHook | None = None

    @property
    def store(self) -> SessionStore:
        return self._store

    def set_message_persist_hook(self, hook: MessagePersistHook | None) -> None:
        self._message_persist_hook = hook

    async def create(self, channel: str, peer_id: str) -> Session:
        """Create a new session."""
        now = datetime.now(UTC)
        session = Session(
            id=uuid4().hex,
            channel=channel,
            peer_id=peer_id,
            model=self._config.agent.model,
            thinking_level=self._config.agent.thinking_level,
            created_at=now,
            updated_at=now,
        )
        await self._store.save_session(
            session_id=session.id,
            channel=session.channel,
            peer_id=session.peer_id,
            model=session.model,
            thinking_level=session.thinking_level,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )
        log.info("session.created", session_id=session.id, channel=channel)
        return session

    async def get(self, session_id: str) -> Session | None:
        """Load a session with its message history."""
        row = await self._store.get_session(session_id)
        if not row:
            return None
        msg_rows = await self._store.get_messages(session_id)
        messages: list[Message] = []
        for m in msg_rows:
            if m.role == "tool":
                messages.append(Message(
                    role="assistant",  # type: ignore[arg-type]
                    content=m.content,
                ))
            else:
                messages.append(Message(
                    role=m.role,  # type: ignore[arg-type]
                    content=m.content,
                ))
        return Session(
            id=row.id,
            channel=row.channel,
            peer_id=row.peer_id,
            messages=messages,
            model=row.model,
            thinking_level=row.thinking_level,
            created_at=datetime.fromisoformat(row.created_at),
            updated_at=datetime.fromisoformat(row.updated_at),
            metadata=row.metadata,  # type: ignore[arg-type]
        )

    async def get_or_create(self, channel: str, peer_id: str) -> Session:
        """Find an existing session for the peer or create a new one."""
        row = await self._store.get_session_by_peer(channel, peer_id)
        if row:
            session = await self.get(row.id)
            if session:
                return session
        return await self.create(channel, peer_id)

    async def add_message(
        self,
        session: Session,
        role: str,
        content: str,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Append a message to the session and persist it."""
        mem_role = "assistant" if role == "tool" else role
        msg = Message(role=mem_role, content=content)  # type: ignore[arg-type]
        session.messages.append(msg)
        session.updated_at = datetime.now(UTC)
        await self._store.add_message(
            session_id=session.id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
        await self._store.update_session_field(
            session.id, updated_at=session.updated_at.isoformat()
        )
        if role == "assistant" and self._message_persist_hook is not None:
            maybe_awaitable = self._message_persist_hook(session, role, content)
            if maybe_awaitable is not None:
                await maybe_awaitable

    async def update_model(self, session: Session, model: str) -> None:
        """Switch the model for a session."""
        session.model = model
        await self._store.update_session_field(session.id, model=model)

    async def update_thinking(self, session: Session, level: str) -> None:
        session.thinking_level = level
        await self._store.update_session_field(session.id, thinking_level=level)

    async def update_metadata(self, session: Session, metadata: dict[str, Any]) -> None:
        """Update session metadata and persist it."""
        merged = await self._store.update_session_metadata(session.id, metadata)
        session.metadata = merged

    async def reset(self, session_id: str) -> Session | None:
        """Clear messages but keep the session."""
        session = await self.get(session_id)
        if not session:
            return None
        await self._store.delete_messages(session_id)
        session.messages.clear()
        session.updated_at = datetime.now(UTC)
        await self._store.update_session_field(
            session_id, updated_at=session.updated_at.isoformat()
        )
        log.info("session.reset", session_id=session_id)
        return session

    async def list_sessions(self) -> list[Session]:
        rows = await self._store.list_sessions()
        counts = await self._store.get_message_counts()
        sessions: list[Session] = []
        for row in rows:
            s = Session(
                id=row.id,
                channel=row.channel,
                peer_id=row.peer_id,
                model=row.model,
                thinking_level=row.thinking_level,
                created_at=datetime.fromisoformat(row.created_at),
                updated_at=datetime.fromisoformat(row.updated_at),
                metadata=row.metadata,  # type: ignore[arg-type]
            )
            s.message_count = counts.get(row.id, 0)
            sessions.append(s)
        return sessions

    async def delete(self, session_id: str) -> None:
        await self._store.delete_session(session_id)
        log.info("session.deleted", session_id=session_id)
