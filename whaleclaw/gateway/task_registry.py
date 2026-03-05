"""Background task registry for multi-Agent execution resilience.

Decouples long-running multi-Agent tasks from the WebSocket connection
lifecycle so that browser refresh / disconnect does not kill the task.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from whaleclaw.gateway.protocol import WSMessage
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

SinkCallback = Callable[[WSMessage], Awaitable[bool]]

_BUFFER_MAX = 500
_RESULT_TTL_SECONDS = 300  # 5 minutes


class TaskStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunningTask:
    """A single background Agent task bound to a session."""

    session_id: str
    task: asyncio.Task[str]
    status: TaskStatus = TaskStatus.RUNNING
    final_result: str = ""
    error: str = ""
    finished_at: float = 0.0
    buffer: deque[WSMessage] = field(default_factory=lambda: deque(maxlen=_BUFFER_MAX))
    _sink: SinkCallback | None = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def emit(self, msg: WSMessage) -> None:
        """Push an event: forward to live sink or buffer for later replay."""
        async with self._lock:
            if self._sink is not None:
                try:
                    await self._sink(msg)
                    return
                except Exception:
                    pass
            self.buffer.append(msg)

    async def attach(self, sink: SinkCallback) -> None:
        """Bind a new WebSocket sink and replay buffered events."""
        async with self._lock:
            replay = list(self.buffer)
            self.buffer.clear()
            self._sink = sink
        for msg in replay:
            try:
                await sink(msg)
            except Exception:
                break

    async def detach(self) -> None:
        """Unbind the current sink (connection lost)."""
        async with self._lock:
            self._sink = None

    @property
    def is_alive(self) -> bool:
        return self.status == TaskStatus.RUNNING


class TaskRegistry:
    """Process-global registry of background multi-Agent tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, RunningTask] = {}
        self._cleanup_handle: asyncio.TimerHandle | None = None

    def has_running(self, session_id: str) -> bool:
        entry = self._tasks.get(session_id)
        return entry is not None and entry.is_alive

    def get(self, session_id: str) -> RunningTask | None:
        return self._tasks.get(session_id)

    def launch(
        self,
        session_id: str,
        coro: Any,
    ) -> RunningTask:
        """Create an asyncio.Task for *coro* and register it.

        If a previous task for the same session exists and is still alive,
        raises RuntimeError (caller should check ``has_running`` first).
        """
        existing = self._tasks.get(session_id)
        if existing is not None and existing.is_alive:
            raise RuntimeError(
                f"session {session_id} already has a running task"
            )

        task = asyncio.create_task(coro, name=f"ma:{session_id[:12]}")
        entry = RunningTask(session_id=session_id, task=task)
        self._tasks[session_id] = entry

        task.add_done_callback(lambda t: self._on_task_done(session_id, t))
        self._schedule_cleanup()

        log.info("task_registry.launched", session_id=session_id)
        return entry

    async def attach(self, session_id: str, sink: SinkCallback) -> bool:
        """Attach a WebSocket sink to an existing task (reconnection)."""
        entry = self._tasks.get(session_id)
        if entry is None:
            return False
        await entry.attach(sink)
        log.info("task_registry.attached", session_id=session_id, alive=entry.is_alive)
        return True

    async def detach(self, session_id: str) -> None:
        """Detach the sink when a WebSocket disconnects."""
        entry = self._tasks.get(session_id)
        if entry is not None:
            await entry.detach()
            log.debug("task_registry.detached", session_id=session_id)

    async def cancel(self, session_id: str) -> bool:
        entry = self._tasks.get(session_id)
        if entry is None or not entry.is_alive:
            return False
        entry.task.cancel()
        entry.status = TaskStatus.CANCELLED
        entry.finished_at = time.monotonic()
        log.info("task_registry.cancelled", session_id=session_id)
        return True

    def _on_task_done(self, session_id: str, task: asyncio.Task[str]) -> None:
        entry = self._tasks.get(session_id)
        if entry is None:
            return
        if task.cancelled():
            entry.status = TaskStatus.CANCELLED
        elif task.exception() is not None:
            entry.status = TaskStatus.FAILED
            entry.error = str(task.exception())
            log.error("task_registry.task_failed", session_id=session_id, error=entry.error)
        else:
            entry.status = TaskStatus.DONE
            entry.final_result = task.result()
        entry.finished_at = time.monotonic()
        log.info(
            "task_registry.task_done",
            session_id=session_id,
            status=entry.status,
        )

    def _schedule_cleanup(self) -> None:
        if self._cleanup_handle is not None:
            return
        loop = asyncio.get_event_loop()
        self._cleanup_handle = loop.call_later(60.0, self._run_cleanup)

    def _run_cleanup(self) -> None:
        self._cleanup_handle = None
        now = time.monotonic()
        expired = [
            sid
            for sid, entry in self._tasks.items()
            if not entry.is_alive and (now - entry.finished_at) > _RESULT_TTL_SECONDS
        ]
        for sid in expired:
            del self._tasks[sid]
            log.debug("task_registry.expired", session_id=sid)
        if self._tasks:
            self._schedule_cleanup()


task_registry = TaskRegistry()
