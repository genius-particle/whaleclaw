"""FastAPI application factory for the Gateway."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from whaleclaw.agent.loop import create_default_registry
from whaleclaw.config.paths import MEMORY_DIR, WHALECLAW_HOME
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.cron.scheduler import CronAction, CronScheduler
from whaleclaw.cron.store import CronStore
from whaleclaw.gateway.middleware import AuthMiddleware, create_jwt
from whaleclaw.gateway.protocol import make_message
from whaleclaw.gateway.ws import push_to_session, websocket_handler
from whaleclaw.memory.manager import MemoryManager
from whaleclaw.memory.vector import SimpleMemoryStore
from whaleclaw.plugins.hooks import HookManager
from whaleclaw.plugins.loader import PluginLoader
from whaleclaw.plugins.registry import PluginRegistry
from whaleclaw.providers.router import ModelRouter
from whaleclaw.sessions.group_compressor import SessionGroupCompressor
from whaleclaw.sessions.manager import SessionManager
from whaleclaw.sessions.store import SessionStore
from whaleclaw.tools.registry import ToolRegistry
from whaleclaw.utils.log import get_logger
from whaleclaw.version import __version__

log = get_logger(__name__)

_UPLOAD_DIR = WHALECLAW_HOME / "uploads"
_CRON_DB_PATH = WHALECLAW_HOME / "cron.db"
_STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


def create_app(config: WhaleclawConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    store = SessionStore()
    cron_store = CronStore(_CRON_DB_PATH)
    async def _on_cron_fire(job_id: str, action: CronAction) -> None:
        if action.type == "message":
            session_id = action.target
            text = action.payload.get("text", "")
            if not (session_id and text):
                return
            content = f"⏰ **提醒**: {text}"

            sent = await push_to_session(session_id, make_message(session_id, content))

            if not sent:
                from whaleclaw.gateway.ws import broadcast_all
                await broadcast_all(make_message(session_id, content))

            if not sent and feishu_channel is not None:
                mgr = state.get("manager")
                if isinstance(mgr, SessionManager):
                    s = await mgr.get(session_id)
                    if s and s.channel == "feishu" and feishu_channel.client:
                        await feishu_channel.client.send_message(
                            s.peer_id,
                            "text",
                            json.dumps({"text": content}, ensure_ascii=False),
                        )

            mgr = state.get("manager")
            if isinstance(mgr, SessionManager):
                s = await mgr.get(session_id)
                if s:
                    await mgr.add_message(s, "assistant", content)

    cron_scheduler = CronScheduler(on_fire=_on_cron_fire)
    plugin_registry = PluginRegistry()
    hook_manager = HookManager()

    feishu_channel: Any = None

    state: dict[str, object] = {
        "manager": None,
        "registry": None,
        "memory_manager": None,
        "hook_manager": hook_manager,
        "group_compressor": None,
        "compression_ready": True,
        "compression_running": False,
        "compression_sessions_total": 0,
        "compression_sessions_done": 0,
        "compression_groups_total": 0,
        "compression_groups_done": 0,
        "compression_cache_hits": 0,
        "compression_generated": 0,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await store.open()
        await cron_store.open()

        persisted = await cron_store.load_jobs()
        for job in persisted:
            await cron_scheduler.add_job(job)

        manager = SessionManager(store, config)
        state["manager"] = manager
        memory_store = SimpleMemoryStore(persist_dir=MEMORY_DIR)
        memory_manager = MemoryManager(memory_store)
        state["memory_manager"] = memory_manager

        registry = create_default_registry(
            session_manager=manager,
            cron_scheduler=cron_scheduler,
            memory_manager=memory_manager,
            memory_store=memory_store,
        )

        await _load_plugins(config, registry, plugin_registry, hook_manager)
        state["registry"] = registry

        await plugin_registry.start_all()
        await cron_scheduler.start()

        summarizer_model = config.agent.summarizer.model.strip()
        prewarm_task: asyncio.Task[None] | None = None
        if config.agent.summarizer.enabled and summarizer_model:
            group_compressor = SessionGroupCompressor(store)
            state["group_compressor"] = group_compressor
            state["compression_ready"] = False
            state["compression_running"] = True
            state["compression_sessions_total"] = 0
            state["compression_sessions_done"] = 0
            state["compression_groups_total"] = 0
            state["compression_groups_done"] = 0
            state["compression_cache_hits"] = 0
            state["compression_generated"] = 0

            async def _prewarm_all_sessions() -> None:
                router = ModelRouter(config.models)
                try:
                    sessions = await manager.list_sessions()
                    state["compression_sessions_total"] = len(sessions)
                    log.info("compressor.prewarm_all_start", sessions_total=len(sessions))
                    for s in sessions:
                        loaded = await manager.get(s.id)
                        sessions_done = int(state["compression_sessions_done"])
                        if not loaded:
                            state["compression_sessions_done"] = sessions_done + 1
                            continue
                        if not loaded.messages:
                            state["compression_sessions_done"] = sessions_done + 1
                            continue
                        stats = await group_compressor.prewarm_session(
                            session_id=loaded.id,
                            messages=loaded.messages,
                            router=router,
                            model_id=summarizer_model,
                        )
                        groups_total = int(state["compression_groups_total"])
                        groups_done = int(state["compression_groups_done"])
                        cache_hits = int(state["compression_cache_hits"])
                        generated = int(state["compression_generated"])

                        state["compression_groups_total"] = (
                            groups_total + int(stats["total_groups"])
                        )
                        state["compression_groups_done"] = (
                            groups_done + int(stats["processed_groups"])
                        )
                        state["compression_cache_hits"] = cache_hits + int(stats["cache_hits"])
                        state["compression_generated"] = generated + int(stats["generated"])
                        state["compression_sessions_done"] = sessions_done + 1
                        log.info(
                            "compressor.prewarm_session_done",
                            session_id=loaded.id,
                            sessions_done=state["compression_sessions_done"],
                            sessions_total=state["compression_sessions_total"],
                            groups_done=state["compression_groups_done"],
                            groups_total=state["compression_groups_total"],
                            cache_hits=state["compression_cache_hits"],
                            generated=state["compression_generated"],
                        )
                    log.info("compressor.prewarm_done", sessions=len(sessions))
                except Exception as exc:
                    log.warning("compressor.prewarm_failed", error=str(exc))
                finally:
                    state["compression_running"] = False
                    state["compression_ready"] = True

            prewarm_task = asyncio.create_task(
                _prewarm_all_sessions(),
                name="session-group-prewarm",
            )
        else:
            state["compression_ready"] = True
            state["compression_running"] = False

        nonlocal feishu_channel
        feishu_cfg = config.channels.feishu
        if feishu_cfg.app_id and feishu_cfg.app_secret:
            from whaleclaw.channels.feishu import FeishuChannel
            from whaleclaw.channels.feishu.config import FeishuConfig

            feishu_channel = FeishuChannel(FeishuConfig(**feishu_cfg.model_dump()))
            await feishu_channel.start()
            if feishu_channel.bot is not None:
                feishu_channel.bot.bind_agent(
                    config,
                    manager,
                    registry,
                    memory_manager=memory_manager,
                    hook_manager=hook_manager,
                    group_compressor=state["group_compressor"],
                    compression_ready_fn=lambda: bool(state["compression_ready"]),
                )

        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        log.info(
            "gateway.started",
            tools=len(registry.list_tools()),
            plugins=len(plugin_registry.list_plugins()),
        )
        yield

        if feishu_channel is not None:
            await feishu_channel.stop()
        if prewarm_task is not None and not prewarm_task.done():
            prewarm_task.cancel()
        compressor = state["group_compressor"]
        if isinstance(compressor, SessionGroupCompressor):
            await compressor.shutdown()
        await cron_scheduler.stop()
        await plugin_registry.stop_all()
        await cron_store.close()
        await store.close()

    def _mgr() -> SessionManager:
        mgr = state["manager"]
        assert isinstance(mgr, SessionManager), "App not started"
        return mgr

    def _tool_registry() -> ToolRegistry:
        reg = state["registry"]
        assert isinstance(reg, ToolRegistry), "App not started"
        return reg

    def _memory_manager() -> MemoryManager:
        mgr = state["memory_manager"]
        assert isinstance(mgr, MemoryManager), "App not started"
        return mgr

    def _hook_manager() -> HookManager:
        hm = state["hook_manager"]
        assert isinstance(hm, HookManager), "App not started"
        return hm

    app = FastAPI(
        title="WhaleClaw Gateway",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if config.gateway.auth.mode != "none":
        app.add_middleware(
            AuthMiddleware,
            auth_config=config.gateway.auth,
        )

    # ── Health ──────────────────────────────────────────────

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "compression_ready": bool(state["compression_ready"]),
            "compression_running": bool(state["compression_running"]),
            "compression_progress": {
                "sessions_total": int(state["compression_sessions_total"]),
                "sessions_done": int(state["compression_sessions_done"]),
                "groups_total": int(state["compression_groups_total"]),
                "groups_done": int(state["compression_groups_done"]),
                "cache_hits": int(state["compression_cache_hits"]),
                "generated": int(state["compression_generated"]),
            },
            "gateway": {
                "port": config.gateway.port,
                "bind": config.gateway.bind,
            },
            "agent": {"model": config.agent.model},
            "auth_mode": config.gateway.auth.mode,
        }

    # ── Models ───────────────────────────────────────────────

    @app.get("/api/models")
    async def list_models() -> dict[str, object]:
        """Return all verified configured models grouped by provider.

        Each provider reads from ``configured_models`` (persisted via the
        config script).  Models include ``thinking`` level and ``tools``
        support flag so the frontend can display and auto-apply them.
        """
        from whaleclaw.providers.nvidia import NvidiaProvider

        providers_cfg = config.models
        available: list[dict[str, object]] = []

        all_providers = [
            "anthropic", "openai", "deepseek", "qwen", "zhipu",
            "minimax", "moonshot", "google", "nvidia",
        ]

        for pname in all_providers:
            pcfg = getattr(providers_cfg, pname, None)
            if not pcfg:
                continue
            has_auth = bool(pcfg.api_key) or (pcfg.auth_mode == "oauth" and pcfg.oauth_access)
            if not has_auth:
                continue

            if pcfg.configured_models:
                for cm in pcfg.configured_models:
                    if not cm.verified:
                        continue
                    if pname == "openai" and pcfg.auth_mode == "oauth" and cm.id != "gpt-5.2":
                        continue
                    tools: bool = True
                    if pname == "nvidia":
                        tools = NvidiaProvider.model_supports_tools(cm.id)
                    entry: dict[str, object] = {
                        "id": f"{pname}/{cm.id}",
                        "name": cm.name or cm.id,
                        "provider": pname,
                        "tools": tools,
                        "thinking": cm.thinking,
                    }
                    available.append(entry)

        return {
            "default": config.agent.model,
            "thinking_level": config.agent.thinking_level,
            "models": available,
        }

    # ── Auth ────────────────────────────────────────────────

    @app.post("/api/auth/login")
    async def auth_login(body: dict[str, str]) -> JSONResponse:
        if config.gateway.auth.mode != "password":
            return JSONResponse(
                {"error": "当前认证模式不支持密码登录"},
                status_code=400,
            )
        if body.get("password") != config.gateway.auth.password:
            return JSONResponse({"error": "密码错误"}, status_code=401)
        token = create_jwt(config.gateway.auth)
        return JSONResponse({"token": token})

    @app.get("/api/auth/verify")
    async def auth_verify() -> JSONResponse:
        return JSONResponse({"valid": True})

    # ── Sessions REST ───────────────────────────────────────

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict[str, object]]:
        mgr = _mgr()
        sessions = await mgr.list_sessions()
        result = []
        for s in sessions:
            usage = await store.get_session_token_usage(s.id)
            result.append({
                "id": s.id,
                "channel": s.channel,
                "peer_id": s.peer_id,
                "model": s.model,
                "thinking_level": s.thinking_level,
                "message_count": s.message_count or len(s.messages),
                "tokens": usage["input_tokens"] + usage["output_tokens"],
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            })
        return result

    @app.post("/api/sessions")
    async def create_session() -> dict[str, object]:
        mgr = _mgr()
        session = await mgr.create("webchat", "web-user")
        return {
            "id": session.id,
            "model": session.model,
            "created_at": session.created_at.isoformat(),
        }

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> JSONResponse:
        mgr = _mgr()
        session = await mgr.get(session_id)
        if not session:
            return JSONResponse(
                {"error": "会话不存在"}, status_code=404
            )
        return JSONResponse({
            "id": session.id,
            "channel": session.channel,
            "model": session.model,
            "thinking_level": session.thinking_level,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in session.messages
            ],
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        })

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse:
        mgr = _mgr()
        await mgr.delete(session_id)
        return JSONResponse({"ok": True})

    @app.post("/api/sessions/{session_id}/compact")
    async def compact_session(session_id: str) -> JSONResponse:
        return JSONResponse(
            {"message": "上下文压缩功能将在后续版本实现"}
        )

    @app.get("/api/token-usage")
    async def get_total_token_usage() -> JSONResponse:
        total = await store.get_total_token_usage()
        by_model = await store.get_token_usage_by_model()
        return JSONResponse({
            "total": total,
            "by_model": by_model,
        })

    @app.get("/api/sessions/{session_id}/token-usage")
    async def get_session_token_usage(session_id: str) -> JSONResponse:
        usage = await store.get_session_token_usage(session_id)
        return JSONResponse(usage)

    # ── Skills REST ────────────────────────────────────────

    _skill_manager: Any = None

    def _get_skill_manager() -> Any:
        nonlocal _skill_manager
        if _skill_manager is None:
            from whaleclaw.skills.manager import SkillManager
            _skill_manager = SkillManager()
        return _skill_manager

    @app.get("/api/skills")
    async def list_skills() -> list[dict[str, object]]:
        mgr = _get_skill_manager()
        bundled = mgr.discover()
        installed_ids = {s.id for s in mgr.list_installed()}
        result: list[dict[str, object]] = []
        for s in bundled:
            result.append({
                "id": s.id,
                "name": s.name,
                "triggers": s.triggers,
                "trigger_description": s.trigger_description,
                "tools": s.tools,
                "max_tokens": s.max_tokens,
                "source": "user" if s.id in installed_ids else "bundled",
            })
        return result

    @app.post("/api/skills/install")
    async def install_skill(body: dict[str, str]) -> JSONResponse:
        source = body.get("source", "").strip()
        if not source:
            return JSONResponse({"error": "缺少 source 参数"}, status_code=400)
        mgr = _get_skill_manager()
        try:
            skill = mgr.install(source)
            return JSONResponse({
                "ok": True,
                "skill": {"id": skill.id, "name": skill.name},
            })
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/skills/{skill_id}", response_model=None)
    async def get_skill_detail(skill_id: str):
        mgr = _get_skill_manager()
        all_skills = mgr.discover()
        skill = next((s for s in all_skills if s.id == skill_id), None)
        if not skill:
            return JSONResponse({"error": "技能不存在"}, status_code=404)
        installed_ids = {s.id for s in mgr.list_installed()}
        raw_content = skill.source_path.read_text(encoding="utf-8")
        return {
            "id": skill.id,
            "name": skill.name,
            "triggers": skill.triggers,
            "trigger_description": skill.trigger_description,
            "instructions": skill.instructions,
            "tools": skill.tools,
            "examples": skill.examples,
            "max_tokens": skill.max_tokens,
            "source": "user" if skill.id in installed_ids else "bundled",
            "raw_markdown": raw_content,
        }

    @app.delete("/api/skills/{skill_id}")
    async def uninstall_skill(skill_id: str) -> JSONResponse:
        mgr = _get_skill_manager()
        removed = mgr.uninstall(skill_id)
        if not removed:
            return JSONResponse({"error": "技能不存在或为内置技能"}, status_code=404)
        return JSONResponse({"ok": True})

    # ── Tools REST ────────────────────────────────────────

    @app.get("/api/tools")
    async def list_tools() -> list[dict[str, object]]:
        reg = _tool_registry()
        tools = reg.list_tools()
        tool_categories: dict[str, str] = {
            "bash": "system",
            "file_read": "file", "file_write": "file", "file_edit": "file",
            "browser": "browser",
            "sessions_list": "session", "sessions_history": "session", "sessions_send": "session",
            "cron_manage": "automation", "reminder": "automation",
        }
        result: list[dict[str, object]] = []
        for t in tools:
            result.append({
                "name": t.name,
                "description": t.description,
                "category": tool_categories.get(t.name, "other"),
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                        **({"enum": p.enum} if p.enum else {}),
                    }
                    for p in t.parameters
                ],
            })
        return result

    # ── Memory Style REST ─────────────────────────────────

    @app.get("/api/memory/style")
    async def get_memory_style() -> JSONResponse:
        mgr = _memory_manager()
        style = await mgr.get_global_style_directive()
        return JSONResponse({
            "enabled": config.agent.memory.global_style_enabled,
            "style_directive": style,
            "has_style": bool(style.strip()),
        })

    @app.post("/api/memory/style")
    async def set_memory_style(body: dict[str, str]) -> JSONResponse:
        directive = str(body.get("style_directive", "")).strip()
        if not directive:
            return JSONResponse({"error": "style_directive 不能为空"}, status_code=400)
        if len(directive) > 300:
            return JSONResponse({"error": "style_directive 过长（最多 300 字）"}, status_code=400)
        mgr = _memory_manager()
        changed = await mgr.set_global_style_directive(
            directive,
            source="api:webchat",
        )
        return JSONResponse({"ok": True, "changed": changed})

    @app.delete("/api/memory/style")
    async def clear_memory_style() -> JSONResponse:
        mgr = _memory_manager()
        removed = await mgr.clear_global_style_directive()
        return JSONResponse({"ok": True, "removed": removed})

    # ── File upload ─────────────────────────────────────────

    @app.post("/api/upload")
    async def upload_file(file: UploadFile) -> JSONResponse:
        if not file.filename:
            return JSONResponse(
                {"error": "文件名为空"}, status_code=400
            )
        dest = _UPLOAD_DIR / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        return JSONResponse({
            "url": f"/api/files/{file.filename}",
            "filename": file.filename,
            "size": dest.stat().st_size,
        })

    @app.get("/api/files/{filename}", response_model=None)
    async def get_file(filename: str) -> FileResponse | JSONResponse:
        path = _UPLOAD_DIR / filename
        if not path.is_file():
            return JSONResponse(
                {"error": "文件不存在"}, status_code=404
            )
        return FileResponse(path)

    def _resolve_local_path(path: str) -> Path | None:
        """Decode and resolve a local file path, with fuzzy fallback."""
        import re as _re
        decoded = unquote(unquote(path))
        fp = Path(decoded).resolve()
        if fp.is_file():
            return fp
        stem = fp.stem
        hash_match = _re.search(r"_([0-9a-f]{6,8})$", stem)
        if hash_match and fp.parent.is_dir():
            suffix_pattern = hash_match.group(0) + fp.suffix
            for candidate in fp.parent.iterdir():
                if candidate.name.endswith(suffix_pattern) and candidate.is_file():
                    return candidate
        return None

    @app.get("/api/local-file", response_model=None)
    async def get_local_file(
        path: str, download: bool = False,
    ) -> FileResponse | JSONResponse:
        """Serve a local file generated by the Agent."""
        fp = _resolve_local_path(path)
        if not fp:
            log.warning("local-file.not_found", path=path)
            return JSONResponse({"error": "文件不存在"}, status_code=404)
        if download:
            return FileResponse(
                fp, filename=fp.name,
                media_type="application/octet-stream",
            )
        return FileResponse(fp, filename=fp.name)

    @app.get("/api/file-info")
    async def file_info(path: str) -> JSONResponse:
        """Return metadata about a local file (name, size)."""
        fp = _resolve_local_path(path)
        if not fp:
            return JSONResponse({"error": "文件不存在"}, status_code=404)
        size = fp.stat().st_size
        return JSONResponse({
            "name": fp.name,
            "size": size,
            "size_human": (
                f"{size / 1048576:.1f}MB" if size >= 1048576
                else f"{size / 1024:.0f}KB" if size >= 1024
                else f"{size}B"
            ),
            "ext": fp.suffix.lstrip(".").lower(),
        })

    # ── WebSocket ───────────────────────────────────────────

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket_handler(
            websocket,
            config,
            session_manager=_mgr(),
            registry=_tool_registry(),
            memory_manager=_memory_manager(),
            hook_manager=_hook_manager(),
            group_compressor=state["group_compressor"],
            compression_ready_fn=lambda: bool(state["compression_ready"]),
        )

    # ── Static files & SPA fallback ────────────────────────

    if _STATIC_DIR.is_dir():
        if (_STATIC_DIR / "assets").is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(_STATIC_DIR / "assets")),
                name="assets",
            )

        _boot_ts = str(int(time.time()))

        @app.get("/")
        async def index() -> HTMLResponse:
            html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
            v = f"{__version__}.{_boot_ts}"
            html = html.replace("/assets/app.css", f"/assets/app.css?v={v}")
            html = html.replace("/assets/app.js", f"/assets/app.js?v={v}")
            return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    else:

        @app.get("/")
        async def health() -> dict[str, str]:
            return {"status": "ok", "version": __version__}

    return app


async def _load_plugins(
    config: WhaleclawConfig,
    tool_registry: ToolRegistry,
    plugin_registry: PluginRegistry,
    hook_manager: HookManager,
) -> None:
    """Discover, load, and register all plugins."""
    from whaleclaw.plugins.sdk import WhaleclawPluginApi

    loader = PluginLoader()
    metas = loader.discover()
    if not metas:
        return

    for meta in metas:
        try:
            plugin = loader.load(meta.id)

            def _make_config_fn(pid: str) -> object:
                plugins_cfg = getattr(config, "plugins", {})
                if isinstance(plugins_cfg, dict):
                    return plugins_cfg.get(pid, {})
                return {}

            api = WhaleclawPluginApi(
                plugin_id=meta.id,
                get_config_fn=lambda pid, key, default: (
                    _make_config_fn(pid).get(key, default)  # type: ignore[union-attr]
                    if isinstance(_make_config_fn(pid), dict)
                    else default
                ),
                get_secret_fn=lambda pid, key: None,
                channel_register_fn=lambda ch: None,
                tool_register_fn=lambda t: tool_registry.register(t),
                hook_register_fn=lambda h, cb, p: hook_manager.register(h, cb, p),
                command_register_fn=lambda cmd, handler: None,
            )
            plugin.register(api)
            await plugin_registry.register(plugin, meta)

            log.info("plugin.loaded", plugin_id=meta.id, name=meta.name)
        except Exception as exc:
            log.warning(
                "plugin.load_failed",
                plugin_id=meta.id,
                error=str(exc),
            )
