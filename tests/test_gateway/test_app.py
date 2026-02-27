"""Tests for the Gateway FastAPI application."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from whaleclaw.config.schema import ProviderModelEntry, WhaleclawConfig
from whaleclaw.gateway.app import create_app


@pytest.fixture()
def app():  # noqa: ANN201
    config = WhaleclawConfig()
    return create_app(config)


@pytest.mark.asyncio
async def test_status_endpoint(app) -> None:  # noqa: ANN001
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["gateway"]["port"] == 18666
    assert "compression_ready" in data
    assert "compression_running" in data


@pytest.mark.asyncio
async def test_index_returns_html_or_json(app) -> None:  # noqa: ANN001
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_sessions_crud(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """Test session CRUD via REST API (with lifespan)."""
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")

    config = WhaleclawConfig()
    test_app = create_app(config)
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        test_app.router.lifespan_context(test_app),
    ):
        resp = await client.post("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        sid = data["id"]
        assert sid

        resp = await client.get("/api/sessions")
        assert resp.status_code == 200
        assert any(s["id"] == sid for s in resp.json())

        resp = await client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sid

        resp = await client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_login_no_password_mode(app) -> None:  # noqa: ANN001
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/auth/login",
            json={"password": "test"},
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_models_oauth_openai_only_shows_gpt52() -> None:
    config = WhaleclawConfig()
    config.models.openai.auth_mode = "oauth"
    config.models.openai.oauth_access = "token"
    config.models.openai.configured_models = [
        ProviderModelEntry(id="gpt-5.2", name="GPT-5.2", verified=True),
        ProviderModelEntry(id="gpt-5.2-codex", name="GPT-5.2 Codex", verified=True),
    ]

    app = create_app(config)
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()["models"]
    ids = [m["id"] for m in models if m["provider"] == "openai"]
    assert ids == ["openai/gpt-5.2"]


@pytest.mark.asyncio
async def test_memory_style_rest_api(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")
    monkeypatch.setattr(app_mod, "MEMORY_DIR", tmp_path / "memory")

    config = WhaleclawConfig()
    test_app = create_app(config)
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        test_app.router.lifespan_context(test_app),
    ):
        resp = await client.get("/api/memory/style")
        assert resp.status_code == 200
        assert resp.json()["has_style"] is False

        resp = await client.post("/api/memory/style", json={"style_directive": "回答简洁明了"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        resp = await client.get("/api/memory/style")
        assert resp.status_code == 200
        assert "简洁明了" in resp.json()["style_directive"]

        resp = await client.delete("/api/memory/style")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
