"""Tests for the SQLite session store."""

from __future__ import annotations

import pytest

from whaleclaw.sessions.store import SessionStore


@pytest.fixture()
async def store(tmp_path):  # noqa: ANN001
    db = tmp_path / "test.db"
    s = SessionStore(db_path=db)
    await s.open()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_save_and_get_session(store: SessionStore) -> None:
    await store.save_session(
        session_id="s1",
        channel="webchat",
        peer_id="user1",
        model="anthropic/claude-sonnet-4-20250514",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    row = await store.get_session("s1")
    assert row is not None
    assert row.id == "s1"
    assert row.channel == "webchat"
    assert row.model == "anthropic/claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_get_nonexistent(store: SessionStore) -> None:
    assert await store.get_session("nope") is None


@pytest.mark.asyncio
async def test_add_and_get_messages(store: SessionStore) -> None:
    await store.save_session(
        session_id="s2",
        channel="webchat",
        peer_id="user1",
        model="test",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    await store.add_message(session_id="s2", role="user", content="hello")
    await store.add_message(session_id="s2", role="assistant", content="hi there")

    msgs = await store.get_messages("s2")
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].content == "hi there"


@pytest.mark.asyncio
async def test_delete_messages(store: SessionStore) -> None:
    await store.save_session(
        session_id="s3",
        channel="webchat",
        peer_id="user1",
        model="test",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    await store.add_message(session_id="s3", role="user", content="msg")
    await store.delete_messages("s3")
    assert await store.count_messages("s3") == 0


@pytest.mark.asyncio
async def test_list_sessions(store: SessionStore) -> None:
    for i in range(3):
        await store.save_session(
            session_id=f"ls{i}",
            channel="webchat",
            peer_id=f"user{i}",
            model="test",
            created_at=f"2026-01-0{i + 1}T00:00:00",
            updated_at=f"2026-01-0{i + 1}T00:00:00",
        )
    rows = await store.list_sessions()
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_group_compression_cache_roundtrip(store: SessionStore) -> None:
    await store.save_session(
        session_id="gc1",
        channel="webchat",
        peer_id="user1",
        model="test",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )

    assert await store.get_group_compression(
        session_id="gc1",
        group_idx=4,
        level="L1",
        source_hash="hash-a",
    ) is None

    await store.upsert_group_compression(
        session_id="gc1",
        group_idx=4,
        level="L1",
        source_hash="hash-a",
        content="压缩内容 A",
    )

    assert await store.get_group_compression(
        session_id="gc1",
        group_idx=4,
        level="L1",
        source_hash="hash-a",
    ) == "压缩内容 A"
    assert await store.get_group_compression(
        session_id="gc1",
        group_idx=4,
        level="L1",
        source_hash="hash-b",
    ) is None


@pytest.mark.asyncio
async def test_group_compression_cache_evicts_old_groups_over_300(store: SessionStore) -> None:
    await store.save_session(
        session_id="gc2",
        channel="webchat",
        peer_id="user2",
        model="test",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )

    for i in range(305):
        await store.upsert_group_compression(
            session_id="gc2",
            group_idx=i + 1,
            level="L1",
            source_hash=f"hash-{i + 1}",
            content=f"压缩内容 {i + 1}",
        )
        await store.upsert_group_compression(
            session_id="gc2",
            group_idx=i + 1,
            level="L0",
            source_hash=f"hash0-{i + 1}",
            content=f"压缩内容0 {i + 1}",
        )

    cursor = await store._conn.execute(  # noqa: SLF001
        "SELECT COUNT(DISTINCT group_idx) FROM session_group_compressions WHERE session_id = ?",
        ("gc2",),
    )
    distinct_row = await cursor.fetchone()
    assert distinct_row is not None
    assert int(distinct_row[0]) == 300

    # Oldest 5 rows should be evicted.
    for i in range(1, 6):
        assert await store.get_group_compression(
            session_id="gc2",
            group_idx=i,
            level="L1",
            source_hash=f"hash-{i}",
        ) is None
        assert await store.get_group_compression(
            session_id="gc2",
            group_idx=i,
            level="L0",
            source_hash=f"hash0-{i}",
        ) is None

    # Latest rows should still exist.
    assert await store.get_group_compression(
        session_id="gc2",
        group_idx=305,
        level="L1",
        source_hash="hash-305",
    ) == "压缩内容 305"
    assert await store.get_group_compression(
        session_id="gc2",
        group_idx=305,
        level="L0",
        source_hash="hash0-305",
    ) == "压缩内容0 305"
