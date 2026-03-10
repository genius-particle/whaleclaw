"""Tests for the SessionManager."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.sessions.manager import SessionManager
from whaleclaw.sessions.store import SessionStore


@pytest.fixture()
async def manager(tmp_path):  # noqa: ANN001
    db = tmp_path / "test.db"
    store = SessionStore(db_path=db)
    await store.open()
    mgr = SessionManager(store, WhaleclawConfig())
    yield mgr
    await store.close()


@pytest.mark.asyncio
async def test_create_session(manager: SessionManager) -> None:
    session = await manager.create("webchat", "user1")
    assert session.channel == "webchat"
    assert session.peer_id == "user1"
    assert "anthropic" in session.model


@pytest.mark.asyncio
async def test_get_or_create(manager: SessionManager) -> None:
    s1 = await manager.get_or_create("webchat", "user2")
    s2 = await manager.get_or_create("webchat", "user2")
    assert s1.id == s2.id


@pytest.mark.asyncio
async def test_add_message(manager: SessionManager) -> None:
    session = await manager.create("webchat", "user3")
    await manager.add_message(session, "user", "hello")
    await manager.add_message(session, "assistant", "hi")
    assert len(session.messages) == 2


@pytest.mark.asyncio
async def test_reset(manager: SessionManager) -> None:
    session = await manager.create("webchat", "user4")
    await manager.add_message(session, "user", "hello")
    reset = await manager.reset(session.id)
    assert reset is not None
    assert len(reset.messages) == 0


@pytest.mark.asyncio
async def test_update_model(manager: SessionManager) -> None:
    session = await manager.create("webchat", "user5")
    await manager.update_model(session, "openai/gpt-5.2")
    assert session.model == "openai/gpt-5.2"


@pytest.mark.asyncio
async def test_update_metadata_preserves_group_compression_state(
    manager: SessionManager,
) -> None:
    session = await manager.create("webchat", "user6")
    await manager.store.update_session_field(
        session.id,
        metadata={
            "group_compression_plan_levels": {"14": "L0"},
            "group_compression_pending": [{"group_idx": 14, "level": "L0"}],
        },
    )

    await manager.update_metadata(session, {"foo": "bar"})

    reloaded = await manager.get(session.id)
    assert reloaded is not None
    assert reloaded.metadata["foo"] == "bar"
    assert reloaded.metadata["group_compression_plan_levels"] == {"14": "L0"}
    assert reloaded.metadata["group_compression_pending"] == [
        {"group_idx": 14, "level": "L0"}
    ]
    assert session.metadata == reloaded.metadata


@pytest.mark.asyncio
async def test_add_message_triggers_persist_hook_only_for_assistant(
    manager: SessionManager,
) -> None:
    session = await manager.create("webchat", "user7")
    hook = AsyncMock()
    manager.set_message_persist_hook(hook)

    await manager.add_message(session, "user", "hello")
    hook.assert_not_awaited()

    await manager.add_message(session, "assistant", "hi")
    hook.assert_awaited_once_with(session, "assistant", "hi")
