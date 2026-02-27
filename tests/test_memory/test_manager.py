"""Tests for MemoryManager."""

from __future__ import annotations

import json

import pytest

from whaleclaw.memory.manager import MemoryManager
from whaleclaw.memory.vector import SimpleMemoryStore
from whaleclaw.providers.base import AgentResponse


@pytest.fixture
def store() -> SimpleMemoryStore:
    return SimpleMemoryStore()


@pytest.fixture
def manager(store: SimpleMemoryStore) -> MemoryManager:
    return MemoryManager(store)


@pytest.mark.asyncio
async def test_recall(manager: MemoryManager) -> None:
    await manager.memorize("用户喜欢 Rust 编程语言", source="session-1")
    result = await manager.recall("Rust 喜欢", max_tokens=500)
    assert "Rust" in result


@pytest.mark.asyncio
async def test_memorize(manager: MemoryManager) -> None:
    entry = await manager.memorize("重要信息：会议在明天", source="manual", tags=["meeting"])
    assert entry.id
    assert entry.content == "重要信息：会议在明天"
    assert "meeting" in entry.tags


@pytest.mark.asyncio
async def test_compact(manager: MemoryManager, store: SimpleMemoryStore) -> None:
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "我最喜欢的语言是 Python"},
    ]
    summary = await manager.compact(messages, source="session-x")
    assert isinstance(summary, str)
    assert len(summary) > 0
    recent = await store.list_recent(limit=10)
    assert len(recent) >= 1


@pytest.mark.asyncio
async def test_auto_capture_balanced_style_preference(
    manager: MemoryManager, store: SimpleMemoryStore
) -> None:
    ok = await manager.auto_capture_user_message(
        "回答我问题请简洁明了",
        source="session:s1",
        mode="balanced",
    )
    assert ok is True
    recent = await store.list_recent(limit=10)
    assert any("简洁明了" in e.content for e in recent)


@pytest.mark.asyncio
async def test_auto_capture_dedup_and_cooldown(
    manager: MemoryManager, store: SimpleMemoryStore
) -> None:
    first = await manager.auto_capture_user_message(
        "请记住，我喜欢简洁回答",
        source="session:s1",
        mode="balanced",
        cooldown_seconds=300,
    )
    second = await manager.auto_capture_user_message(
        "请记住，我喜欢简洁回答",
        source="session:s1",
        mode="balanced",
        cooldown_seconds=300,
    )
    assert first is True
    assert second is False
    recent = await store.list_recent(limit=20)
    assert len([e for e in recent if "喜欢简洁回答" in e.content]) == 1


def test_recall_policy_intent_levels(manager: MemoryManager) -> None:
    should, raw = manager.recall_policy("继续上次那个南京两日游PPT")
    assert should is True
    assert raw is True

    should2, raw2 = manager.recall_policy("你好")
    assert should2 is True
    assert raw2 is True


@pytest.mark.asyncio
async def test_auto_capture_batches_before_flushing(
    manager: MemoryManager, store: SimpleMemoryStore
) -> None:
    for text in ("以后称呼我老王", "默认语言用中文", "下次按这个规则回复"):
        ok = await manager.auto_capture_user_message(
            text,
            source="session:s1",
            mode="balanced",
            cooldown_seconds=0,
            max_per_hour=100,
            batch_size=4,
            merge_window_seconds=9999,
        )
        assert ok is True
    assert await store.list_recent(limit=10) == []

    flushed = await manager.flush_capture_buffer(source="session:s1", force=True)
    assert flushed == 3
    recent = await store.list_recent(limit=10)
    assert len(recent) == 3
    assert all("batched" in e.tags for e in recent)


class _FakeRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, model_id: str, messages: list[object], **kwargs: object) -> AgentResponse:  # noqa: ARG002
        self.calls += 1
        payload = {
            "l1": "用户明确要求回答风格简洁明了，优先要点式回复。",
            "style_directive": "回答风格：简洁明了，先结论后细节，优先要点列表。",
            "keep": ["风格偏好稳定"],
            "drop": ["重复描述"],
        }
        return AgentResponse(content=json.dumps(payload, ensure_ascii=False), model=model_id)


class _FakeRuleRouter:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    async def chat(self, model_id: str, messages: list[object], **kwargs: object) -> AgentResponse:  # noqa: ARG002
        return AgentResponse(content=json.dumps(self._payload, ensure_ascii=False), model=model_id)


class _FakeCompressRouter:
    async def chat(self, model_id: str, messages: list[object], **kwargs: object) -> AgentResponse:  # noqa: ARG002
        return AgentResponse(content="压缩后画像", model=model_id)


@pytest.mark.asyncio
async def test_organize_if_needed_generates_profile(
    manager: MemoryManager, store: SimpleMemoryStore
) -> None:
    for txt in (
        "回答要简洁明了",
        "以后请直接给结论",
        "默认使用要点列表",
        "别写太长",
        "语气保持专业",
    ):
        _ = await manager.auto_capture_user_message(
            txt,
            source="session:s1",
            mode="balanced",
            cooldown_seconds=0,
            max_per_hour=100,
        )

    router = _FakeRouter()
    ok = await manager.organize_if_needed(
        router=router,  # type: ignore[arg-type]
        model_id="zhipu/glm-4.7-flash",
        organizer_min_new_entries=3,
        organizer_interval_seconds=0,
    )
    assert ok is True
    assert router.calls == 1
    recent = await store.list_recent(limit=50)
    assert any("memory_profile" in e.tags for e in recent)
    assert any("level:L1" in e.tags for e in recent)
    assert any("style:global" in e.tags for e in recent)
    style = await manager.get_global_style_directive()
    assert "简洁明了" in style


@pytest.mark.asyncio
async def test_recall_includes_latest_l1_profile(
    manager: MemoryManager, store: SimpleMemoryStore
) -> None:
    await store.add(
        "PPT制作规则：封面图与内容页布局不同，内容页图片保持统一尺寸。",
        source="memory_organizer",
        tags=["memory_profile", "level:L1", "curated"],
    )

    recalled = await manager.recall(
        "帮我做PPT",
        max_tokens=500,
        include_profile=True,
        include_raw=False,
    )
    assert "长期记忆画像" in recalled
    assert "PPT制作规则" in recalled


@pytest.mark.asyncio
async def test_build_profile_for_injection_prefers_l1_text_when_within_budget(
    manager: MemoryManager, store: SimpleMemoryStore
) -> None:
    await store.add(
        "用户长期偏好：回答先给结论。",
        source="memory_organizer",
        tags=["memory_profile", "level:L1", "curated"],
    )
    out = await manager.build_profile_for_injection(
        max_tokens=1600,
        router=_FakeCompressRouter(),  # type: ignore[arg-type]
        model_id="zhipu/glm-4.7-flash",
    )
    assert "长期记忆画像" in out
    assert "回答先给结论" in out


@pytest.mark.asyncio
async def test_set_and_clear_global_style_directive(
    manager: MemoryManager, store: SimpleMemoryStore
) -> None:
    changed = await manager.set_global_style_directive(
        "回答风格：简洁明了，先结论后细节。",
        source="test",
    )
    assert changed is True
    assert "简洁明了" in await manager.get_global_style_directive()

    changed2 = await manager.set_global_style_directive(
        "回答风格：简洁明了，先结论后细节。",
        source="test",
    )
    assert changed2 is False

    removed = await manager.clear_global_style_directive()
    assert removed >= 1
    assert await manager.get_global_style_directive() == ""
    recent = await store.list_recent(limit=20)
    assert not any("style:global" in e.tags for e in recent)


@pytest.mark.asyncio
async def test_upsert_profile_from_capture_appends_rule_to_single_layer(
    manager: MemoryManager, store: SimpleMemoryStore
) -> None:
    await store.add(
        "用户偏好：回答简洁。",
        source="memory_organizer",
        tags=["memory_profile", "level:L1", "curated"],
    )
    router = _FakeRuleRouter({"accept": True, "rule": "PPT 内容页图片尺寸统一"})
    ok = await manager.upsert_profile_from_capture(
        "做PPT时内容页图片尺寸统一",
        router=router,  # type: ignore[arg-type]
        model_id="qwen/qwen3.5-plus",
        max_tokens=1600,
    )
    assert ok is True
    recent = await store.list_recent(limit=20)
    l1 = next(
        e for e in recent
        if "memory_profile" in e.tags and "level:L1" in e.tags
    )
    assert "PPT 内容页图片尺寸统一" in l1.content
