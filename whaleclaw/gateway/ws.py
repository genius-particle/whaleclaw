"""WebSocket connection handler with session and tool support."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from whaleclaw.agent.commands import ChatCommand
from whaleclaw.agent.loop import run_agent
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.gateway.protocol import (
    MessageType,
    WSMessage,
    make_error,
    make_message,
    make_pong,
    make_status,
    make_stream,
    make_tool_call,
    make_tool_result,
)
from whaleclaw.plugins.evomap.bridge import build_memory_hint_from_hook_data
from whaleclaw.plugins.hooks import HookContext, HookManager, HookPoint
from whaleclaw.providers.base import ImageContent
from whaleclaw.providers.router import ModelRouter
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.tools.base import ToolResult
from whaleclaw.tools.registry import ToolRegistry
from whaleclaw.utils.log import get_logger

if TYPE_CHECKING:
    from whaleclaw.memory.manager import MemoryManager
    from whaleclaw.sessions.group_compressor import SessionGroupCompressor

log = get_logger(__name__)

_active_connections: dict[str, WebSocket] = {}


async def push_to_session(session_id: str, msg: WSMessage) -> bool:
    """Push a message to a connected WebSocket by session ID.

    Returns True if sent successfully, False if session not connected.
    """
    ws = _active_connections.get(session_id)
    if ws is None:
        return False
    return await _safe_send(ws, msg)


async def broadcast_all(msg: WSMessage) -> int:
    """Broadcast a message to ALL active WebSocket connections (deduplicated)."""
    sent = 0
    seen: set[int] = set()
    for ws in list(_active_connections.values()):
        ws_id = id(ws)
        if ws_id in seen:
            continue
        seen.add(ws_id)
        if await _safe_send(ws, msg):
            sent += 1
    return sent


async def _safe_send(ws: WebSocket, msg: WSMessage) -> bool:
    """Send a WS message, returning False if connection is dead."""
    try:
        await ws.send_text(msg.model_dump_json())
        return True
    except Exception:
        return False


async def _resolve_session(
    session_id: str | None,
    current: Session | None,
    manager: SessionManager,
) -> Session:
    """Resolve session: reuse existing, load from DB, or create new."""
    target_id = session_id or (current.id if current else None)
    if target_id:
        if current and current.id == target_id:
            return current
        loaded = await manager.get(target_id)
        if loaded:
            return loaded
    return await manager.create("webchat", "ws-anonymous")


async def websocket_handler(
    websocket: WebSocket,
    config: WhaleclawConfig,
    session_manager: SessionManager,
    registry: ToolRegistry,
    memory_manager: MemoryManager | None = None,
    hook_manager: HookManager | None = None,
    group_compressor: SessionGroupCompressor | None = None,
    compression_ready_fn: Callable[[], bool] | None = None,
) -> None:
    """Handle a single WebSocket connection lifecycle."""
    await websocket.accept()

    conn_key = f"_conn:{uuid4().hex[:12]}"
    _active_connections[conn_key] = websocket

    session: Session | None = None
    router = ModelRouter(config.models)
    from whaleclaw.sessions.compressor import ContextCompressor

    _store = session_manager._store  # noqa: SLF001
    _compressor = ContextCompressor()
    chat_cmd = ChatCommand(
        session_manager,
        config=config,
        router=router,
        compressor=_compressor,
        session_store=_store,
    )
    ws_alive = True
    inbox: asyncio.Queue[WSMessage] = asyncio.Queue()

    log.debug("ws.connected", conn_key=conn_key)

    async def _reader() -> None:
        """Read WS messages and dispatch: ping handled inline, others queued."""
        nonlocal ws_alive
        try:
            while ws_alive:
                raw = await websocket.receive_text()
                log.debug("ws.recv_raw", size=len(raw), preview=raw[:120])
                try:
                    incoming = WSMessage.model_validate_json(raw)
                except Exception:
                    sid = session.id if session else ""
                    log.warning("ws.invalid_message_json", preview=raw[:200], session_id=sid)
                    await _safe_send(websocket, make_error(sid, "无效的消息格式"))
                    continue
                if incoming.type == MessageType.PING:
                    log.debug("ws.ping", session_id=incoming.session_id)
                    await _safe_send(websocket, make_pong())
                    continue

                log.debug(
                    "ws.incoming",
                    type=incoming.type,
                    session_id=incoming.session_id,
                    payload_keys=list(incoming.payload.keys()),
                )

                if incoming.type == MessageType.MESSAGE:
                    content = str(incoming.payload.get("content", ""))
                    log.info(
                        "ws.message_enqueued",
                        session_id=incoming.session_id,
                        content_len=len(content),
                        content_preview=(" ".join(content.split())[:80]),
                    )
                    await inbox.put(incoming)
        except WebSocketDisconnect:
            ws_alive = False
            sid = session.id if session else "(none)"
            log.debug("ws.disconnected", session_id=sid)
        except Exception:
            ws_alive = False
            sid = session.id if session else "(none)"
            log.debug("ws.closed_unexpectedly", session_id=sid)

    async def _processor() -> None:
        """Process queued user messages (run agent, send replies)."""
        nonlocal session
        while ws_alive:
            try:
                incoming = await asyncio.wait_for(inbox.get(), timeout=1.0)
            except TimeoutError:
                continue

            session = await _resolve_session(
                incoming.session_id, session, session_manager,
            )
            _active_connections[session.id] = websocket

            if compression_ready_fn is not None and not compression_ready_fn():
                await _safe_send(
                    websocket,
                    make_status(session.id, "会话压缩中，请稍后再试…"),
                )
                continue

            content = incoming.payload.get("content", "")
            raw_images = incoming.payload.get("images", [])
            log.debug(
                "ws.message_processing",
                session_id=session.id,
                incoming_session_id=incoming.session_id,
                content_len=len(content) if isinstance(content, str) else 0,
                images_count=len(raw_images) if isinstance(raw_images, list) else 0,
            )

            images: list[ImageContent] = []
            if isinstance(raw_images, list):
                for img_data in raw_images:
                    if isinstance(img_data, dict) and img_data.get("data"):
                        images.append(ImageContent(
                            mime=img_data.get("mime", "image/png"),
                            data=img_data["data"],
                        ))

            if not content and not images:
                await _safe_send(websocket, make_error(session.id, "消息内容为空"))
                continue

            if images:
                log.debug("ws.images", count=len(images), session_id=session.id)

            cmd_result = await chat_cmd.handle(content, session) if content else None
            if cmd_result is not None:
                await _safe_send(websocket, make_message(session.id, cmd_result))
                if content.strip().lower() in ("/new", "/reset"):
                    session = await session_manager.get(session.id) or session
                continue

            try:
                await session_manager.add_message(session, "user", content or "(图片)")
            except Exception as exc:
                log.warning("ws.add_message_failed", error=str(exc))
                session = await session_manager.create("webchat", "ws-anonymous")
                await session_manager.add_message(session, "user", content or "(图片)")

            sid = session.id

            async def on_stream(chunk: str, _sid: str = sid) -> None:
                if ws_alive:
                    await _safe_send(websocket, make_stream(_sid, chunk))

            async def on_tool_call(
                name: str,
                arguments: dict[str, Any],
                _sid: str = sid,
            ) -> None:
                if ws_alive:
                    await _safe_send(websocket, make_tool_call(_sid, name, arguments))

            async def on_tool_result_cb(
                name: str,
                result: ToolResult,
                _sid: str = sid,
            ) -> None:
                if ws_alive:
                    await _safe_send(
                        websocket,
                        make_tool_result(_sid, name, result.output, result.success),
                    )

            try:
                extra_memory = ""
                if hook_manager is not None:
                    hook_out = await hook_manager.run(
                        HookPoint.BEFORE_MESSAGE,
                        HookContext(
                            hook=HookPoint.BEFORE_MESSAGE,
                            session_id=session.id,
                            data={
                                "message": content,
                                "channel": "webchat",
                            },
                        ),
                    )
                    if not hook_out.proceed:
                        await _safe_send(websocket, make_message(session.id, "消息被策略阻止。"))
                        continue
                    extra_memory = build_memory_hint_from_hook_data(hook_out.data)
                reply = await run_agent(
                    message=content or "(用户发送了图片)",
                    session_id=session.id,
                    config=config,
                    on_stream=on_stream,
                    session=session,
                    router=router,
                    registry=registry,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result_cb,
                    images=images or None,
                    session_manager=session_manager,
                    session_store=_store,
                    memory_manager=memory_manager,
                    extra_memory=extra_memory,
                    trigger_event_id=incoming.id,
                    trigger_text_preview=content or "(用户发送了图片)",
                    group_compressor=group_compressor,
                )
                await session_manager.add_message(session, "assistant", reply)
                await _safe_send(websocket, make_message(session.id, reply))
            except Exception as exc:
                if hook_manager is not None:
                    with suppress(Exception):
                        await hook_manager.run(
                            HookPoint.ON_ERROR,
                            HookContext(
                                hook=HookPoint.ON_ERROR,
                                session_id=session.id,
                                data={
                                    "error": str(exc),
                                    "message": content,
                                    "channel": "webchat",
                                },
                            ),
                        )
                log.error(
                    "agent.error",
                    error=str(exc),
                    session_id=session.id,
                )
                await _safe_send(
                    websocket, make_error(session.id, f"Agent 处理失败: {exc}"),
                )

    reader_task = asyncio.create_task(_reader())
    processor_task = asyncio.create_task(_processor())

    try:
        done, pending = await asyncio.wait(
            [reader_task, processor_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        ws_alive = False
        for t in pending:
            t.cancel()
        for t in done:
            if t.exception() and not isinstance(t.exception(), asyncio.CancelledError):
                log.error("ws.task_error", error=str(t.exception()))
    except Exception:
        ws_alive = False
        reader_task.cancel()
        processor_task.cancel()
    finally:
        _active_connections.pop(conn_key, None)
        if session:
            _active_connections.pop(session.id, None)
