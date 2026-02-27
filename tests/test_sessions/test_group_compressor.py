"""Tests for session group compressor behavior."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from whaleclaw.providers.base import Message
from whaleclaw.sessions.group_compressor import SessionGroupCompressor, _hash_group
from whaleclaw.sessions.store import SessionStore


def _mk_group(i: int, text: str) -> list[Message]:
    return [
        Message(role="user", content=f"u{i}:{text}"),
        Message(role="assistant", content=f"a{i}:{text}"),
    ]


def _flatten(groups: list[list[Message]]) -> list[Message]:
    out: list[Message] = []
    for g in groups:
        out.extend(g)
    return out


class _NoopRouter:
    async def chat(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("chat should not be called when model_id is empty")


class _SlowRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        await asyncio.sleep(0.15)
        return SimpleNamespace(content="压缩摘要")


async def _mk_store(tmp_path) -> SessionStore:  # noqa: ANN001
    store = SessionStore(db_path=tmp_path / "group_compressor.db")
    await store.open()
    return store


@pytest.mark.asyncio
async def test_window_plan_uses_absolute_group_index(tmp_path) -> None:  # noqa: ANN001
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        groups = [_mk_group(i, "短消息") for i in range(1, 31)]
        plan = compressor._window_plan(_flatten(groups))  # noqa: SLF001
        assert len(plan) == 25
        assert plan[0].group_idx == 6
        assert plan[-1].group_idx == 30
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_build_window_messages_schedules_background_generation(tmp_path) -> None:  # noqa: ANN001
    store = await _mk_store(tmp_path)
    compressor = SessionGroupCompressor(store)
    try:
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s2",
            channel="webchat",
            peer_id="u2",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
        )
        groups = [_mk_group(i, "需要压缩的历史消息 " + ("x" * 120)) for i in range(1, 13)]
        router = _SlowRouter()

        t0 = time.monotonic()
        output = await compressor.build_window_messages(
            session_id="s2",
            messages=_flatten(groups),
            router=router,  # type: ignore[arg-type]
            model_id="compress-model",
        )
        elapsed = time.monotonic() - t0

        assert elapsed < 0.2
        assert output

        plan = compressor._window_plan(_flatten(groups))  # noqa: SLF001
        first = next(item for item in plan if item.level != "L2")
        source_hash = _hash_group(first.group)

        found = False
        for _ in range(20):
            cached = await store.get_group_compression(
                session_id="s2",
                group_idx=first.group_idx,
                level=first.level,
                source_hash=source_hash,
            )
            if cached:
                found = True
                break
            await asyncio.sleep(0.05)

        assert found
        assert router.calls > 0
    finally:
        await compressor.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_recent_over_budget_downgrades_2_and_3_to_l0(tmp_path) -> None:  # noqa: ANN001
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        groups = [
            _mk_group(1, "旧历史"),
            _mk_group(2, "最近第3组 " + ("超长内容 " * 600)),
            _mk_group(3, "最近第2组 " + ("超长内容 " * 600)),
            _mk_group(4, "最近第1组 " + ("超长内容 " * 600)),
        ]
        plan = compressor._window_plan(_flatten(groups))  # noqa: SLF001
        assert [x.level for x in plan][-3:] == ["L0", "L0", "L2"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_window_plan_compresses_22_groups_when_25_groups_present(tmp_path) -> None:  # noqa: ANN001
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        groups = [_mk_group(i, "消息") for i in range(1, 26)]
        plan = compressor._window_plan(_flatten(groups))  # noqa: SLF001
        l2 = sum(1 for x in plan if x.level == "L2")
        l1 = sum(1 for x in plan if x.level == "L1")
        l0 = sum(1 for x in plan if x.level == "L0")
        assert l2 == 3
        assert l1 == 7
        assert l0 == 15
        assert l1 + l0 == 22
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_build_window_messages_outputs_structured_blocks(tmp_path) -> None:  # noqa: ANN001
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s1",
            channel="webchat",
            peer_id="u1",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
        )
        groups = [_mk_group(i, "历史消息") for i in range(1, 6)]
        groups.append([Message(role="user", content="u6:当前轮用户请求")])
        output = await compressor.build_window_messages(
            session_id="s1",
            messages=_flatten(groups),
            router=_NoopRouter(),  # type: ignore[arg-type]
            model_id="",
        )

        text = "\n".join(m.content for m in output)
        assert "【历史摘要" in text
        assert "【最近对话原文" in text
        assert "【当前任务状态】" in text
        assert output[-1].role == "user"
        assert "当前轮用户请求" in output[-1].content
    finally:
        await store.close()
