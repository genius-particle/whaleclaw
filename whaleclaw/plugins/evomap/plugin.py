"""EvoMap plugin main entry."""

from __future__ import annotations

import asyncio
import re
from typing import Any, cast

from whaleclaw.plugins.evomap.bounty import BountyManager
from whaleclaw.plugins.evomap.client import A2AClient
from whaleclaw.plugins.evomap.config import EvoMapConfig
from whaleclaw.plugins.evomap.fetcher import AssetFetcher
from whaleclaw.plugins.evomap.identity import EvoMapIdentity
from whaleclaw.plugins.evomap.publisher import AssetPublisher
from whaleclaw.plugins.hooks import HookContext, HookPoint, HookResult
from whaleclaw.plugins.sdk import WhaleclawPlugin, WhaleclawPluginApi
from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult


class EvoMapPublishTool(Tool):
    """发布解决方案到 EvoMap 网络。"""

    def __init__(
        self,
        publisher: AssetPublisher,
        client: A2AClient | None = None,
        identity: EvoMapIdentity | None = None,
    ) -> None:
        self._publisher = publisher
        self._client = client
        self._identity = identity

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="evomap_publish",
            description="将验证过的修复/优化/创新打包为 Gene+Capsule 发布到 EvoMap 网络。",
            parameters=[
                ToolParameter(
                    name="category",
                    type="string",
                    description="repair | optimize | innovate",
                ),
                ToolParameter(
                    name="signals",
                    type="string",
                    description=(
                        "逗号分隔的泛化技术关键词（任务类型+技术栈，不含具体业务内容），"
                        "如 'ppt,python-pptx,slide_generation' 或 'websocket,reconnect,backoff'"
                    ),
                ),
                ToolParameter(
                    name="gene_summary",
                    type="string",
                    description="Gene 摘要",
                ),
                ToolParameter(
                    name="capsule_summary",
                    type="string",
                    description="Capsule 摘要 (至少 20 字)",
                ),
                ToolParameter(
                    name="confidence",
                    type="number",
                    description="置信度 0-1",
                ),
                ToolParameter(
                    name="files_affected",
                    type="integer",
                    description="影响文件数",
                ),
                ToolParameter(
                    name="lines_affected",
                    type="integer",
                    description="影响行数",
                ),
            ],
        )

    async def _reauth(self) -> None:
        if not self._client:
            return
        resp = await self._client.hello()
        payload: Any = resp.get("payload") or {}
        if self._identity and isinstance(payload, dict):
            self._identity.save_hello_response(cast(dict[str, str | int | None], payload))

    async def execute(self, **kwargs: Any) -> ToolResult:
        from whaleclaw.plugins.evomap.models import BlastRadius, Outcome

        signals = [s.strip() for s in str(kwargs.get("signals", "")).split(",") if s.strip()]
        if not signals:
            return ToolResult(success=False, output="", error="信号不能为空")

        async def _do_publish() -> str:
            result = await self._publisher.publish_fix(
                category=kwargs.get("category", "repair"),
                signals=signals,
                gene_summary=str(kwargs.get("gene_summary", "")),
                capsule_summary=str(kwargs.get("capsule_summary", "")),
                confidence=float(kwargs.get("confidence", 0.8)),
                blast_radius=BlastRadius(
                    files=int(kwargs.get("files_affected", 1)),
                    lines=int(kwargs.get("lines_affected", 1)),
                ),
                outcome=Outcome(status="success", score=1.0),
            )
            return str(result)

        try:
            out = await _do_publish()
        except Exception:
            try:
                await self._reauth()
                out = await _do_publish()
            except Exception as e:
                return ToolResult(success=False, output="", error=f"发布失败: {e}")
        return ToolResult(success=True, output=out, error=None)


class EvoMapFetchTool(Tool):
    """从 EvoMap 搜索已验证的解决方案。"""

    def __init__(
        self,
        fetcher: AssetFetcher,
        client: A2AClient | None = None,
        identity: EvoMapIdentity | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._client = client
        self._identity = identity

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="evomap_fetch",
            description="按信号搜索 EvoMap 网络中已有的 Capsule 解决方案。",
            parameters=[
                ToolParameter(
                    name="signals",
                    type="string",
                    description=(
                        "逗号分隔的泛化技术关键词，用于匹配已有方案。"
                        "只提取任务类型和技术栈，去掉具体业务内容。"
                        "例：用户说'做个刘亦菲3页PPT' → 'ppt,python-pptx,slide_generation'；"
                        "'WebSocket 断开重连' → 'websocket,reconnect,exponential_backoff'；"
                        "'修复 SQL 慢查询' → 'sql_performance,n_plus_one,query_optimization'"
                    ),
                ),
            ],
        )

    async def _reauth_and_retry(self, signals: list[str]) -> list[dict[str, Any]]:
        """Re-authenticate via hello then retry fetch (secret may have expired)."""
        if not self._client:
            raise RuntimeError("no client")
        resp = await self._client.hello()
        payload: Any = resp.get("payload") or {}
        if self._identity and isinstance(payload, dict):
            self._identity.save_hello_response(cast(dict[str, str | int | None], payload))
        return await self._fetcher.search_by_signals(signals)

    async def execute(self, **kwargs: Any) -> ToolResult:
        signals = [s.strip() for s in str(kwargs.get("signals", "")).split(",") if s.strip()]
        if not signals:
            return ToolResult(success=False, output="", error="搜索关键词不能为空")
        try:
            assets = await self._fetcher.search_by_signals(signals)
        except Exception:
            try:
                assets = await self._reauth_and_retry(signals)
            except Exception as e:
                return ToolResult(success=False, output="", error=f"搜索失败: {e}")
        out = "\n".join(
            f"- {a.get('asset_id', '')}: {a.get('summary', '')}" for a in assets[:10]
        )
        return ToolResult(success=True, output=out or "未找到匹配方案", error=None)


class EvoMapBountyTool(Tool):
    """查看和认领 EvoMap 赏金任务。"""

    def __init__(
        self,
        bounty: BountyManager,
        client: A2AClient | None = None,
        identity: EvoMapIdentity | None = None,
    ) -> None:
        self._bounty = bounty
        self._client = client
        self._identity = identity

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="evomap_bounty",
            description="列出/认领/完成 EvoMap 赏金任务。",
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="list | claim | complete | my_tasks",
                ),
                ToolParameter(
                    name="task_id",
                    type="string",
                    description="任务 ID (claim/complete 时必填)",
                    required=False,
                ),
                ToolParameter(
                    name="asset_id",
                    type="string",
                    description="完成时提交的 asset_id",
                    required=False,
                ),
            ],
        )

    async def _reauth(self) -> None:
        if not self._client:
            return
        resp = await self._client.hello()
        payload: Any = resp.get("payload") or {}
        if self._identity and isinstance(payload, dict):
            self._identity.save_hello_response(cast(dict[str, str | int | None], payload))

    async def _execute_inner(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "list"))
        if action == "list":
            tasks = await self._bounty.list_tasks()
            out = "\n".join(f"- {t.task_id}: {t.title}" for t in tasks[:20])
            return ToolResult(success=True, output=out or "暂无可用任务", error=None)
        if action == "claim":
            tid = kwargs.get("task_id")
            if not tid:
                return ToolResult(success=False, output="", error="需要 task_id")
            await self._bounty.claim_task(str(tid))
            return ToolResult(success=True, output=f"任务 {tid} 已认领", error=None)
        if action == "complete":
            tid = kwargs.get("task_id")
            aid = kwargs.get("asset_id")
            if not tid or not aid:
                return ToolResult(success=False, output="", error="需要 task_id 和 asset_id")
            await self._bounty.complete_task(str(tid), str(aid))
            return ToolResult(success=True, output=f"任务 {tid} 已提交完成", error=None)
        if action == "my_tasks":
            tasks = await self._bounty.my_tasks()
            out = "\n".join(f"- {t.task_id}: {t.title}" for t in tasks[:20])
            return ToolResult(success=True, output=out or "无已认领任务", error=None)
        return ToolResult(success=False, output="", error=f"未知操作: {action}")

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            return await self._execute_inner(**kwargs)
        except Exception:
            try:
                await self._reauth()
                return await self._execute_inner(**kwargs)
            except Exception as e:
                return ToolResult(success=False, output="", error=f"赏金操作失败: {e}")


class EvoMapPlugin(WhaleclawPlugin):
    """EvoMap 协作进化市场插件。"""

    def __init__(self) -> None:
        self._client: A2AClient | None = None
        self._publisher: AssetPublisher | None = None
        self._fetcher: AssetFetcher | None = None
        self._bounty: BountyManager | None = None
        self._identity: EvoMapIdentity | None = None
        self._cfg: EvoMapConfig | None = None
        self._startup_task: asyncio.Task[None] | None = None

    @property
    def id(self) -> str:
        return "evomap"

    @property
    def name(self) -> str:
        return "EvoMap"

    def _build_components(self, api: WhaleclawPluginApi) -> EvoMapConfig | None:
        cfg = EvoMapConfig(
            enabled=api.get_config("enabled", False),
            hub_url=api.get_config("hub_url", "https://evomap.ai"),
            auto_fetch=api.get_config("auto_fetch", True),
            auto_publish=api.get_config("auto_publish", False),
            sync_interval_hours=api.get_config("sync_interval_hours", 4.0),
            webhook_url=api.get_config("webhook_url"),
            min_confidence_to_publish=api.get_config("min_confidence_to_publish", 0.7),
            auto_search_on_error=api.get_config("auto_search_on_error", True),
        )
        if not cfg.enabled:
            return None
        self._cfg = cfg
        identity = EvoMapIdentity()
        self._identity = identity
        sender_id = identity.get_or_create_sender_id()
        cached_secret = identity.get_node_secret()
        self._client = A2AClient(cfg.hub_url, sender_id, node_secret=cached_secret)
        self._publisher = AssetPublisher(self._client, identity)
        self._fetcher = AssetFetcher(self._client)
        self._bounty = BountyManager(self._client)
        return cfg

    def register(self, api: WhaleclawPluginApi) -> None:
        cfg = self._build_components(api)
        if not cfg:
            return
        assert self._publisher and self._fetcher and self._bounty
        api.register_tool(EvoMapPublishTool(self._publisher, self._client, self._identity))
        api.register_tool(EvoMapFetchTool(self._fetcher, self._client, self._identity))
        api.register_tool(EvoMapBountyTool(self._bounty, self._client, self._identity))
        api.register_hook(HookPoint.BEFORE_MESSAGE, self._on_before_message_suggest)
        api.register_hook(HookPoint.ON_ERROR, self._on_error_search)
        api.register_hook(HookPoint.AFTER_TOOL_CALL, self._on_tool_success_publish)

    async def _on_before_message_suggest(self, ctx: HookContext) -> HookResult:
        if not self._fetcher:
            return HookResult(proceed=True, data=ctx.data)
        text = str(ctx.data.get("message", "")).strip()
        if len(text) < 6:
            return HookResult(proceed=True, data=ctx.data)

        terms = [w.lower() for w in re.findall(r"[\w\u4e00-\u9fff]+", text) if len(w) >= 2]
        if not terms:
            return HookResult(proceed=True, data=ctx.data)
        signals = terms[:8]

        assets = self._fetcher.search_cached_by_signals(signals, limit=4)
        if not assets and self._cfg and self._cfg.auto_search_on_error:
            try:
                assets = await self._fetcher.search_by_signals(signals[:3])
            except Exception:
                assets = []
        if assets:
            ctx.data["evomap_suggestions"] = assets[:4]
        return HookResult(proceed=True, data=ctx.data)

    async def _on_error_search(self, ctx: HookContext) -> HookResult:
        if not self._fetcher:
            return HookResult(proceed=True, data=ctx.data)
        err = ctx.data.get("error") or ctx.data.get("error_type") or ""
        signals = [err] if err else []
        if signals:
            try:
                assets = await self._fetcher.search_by_signals(signals)
                if assets:
                    ctx.data["evomap_suggestions"] = assets[:4]
            except Exception:
                pass
        return HookResult(proceed=True, data=ctx.data)

    async def _on_tool_success_publish(self, ctx: HookContext) -> HookResult:
        return HookResult(proceed=True, data=ctx.data)

    async def on_start(self) -> None:
        if not self._client or not self._fetcher:
            return
        if self._startup_task and not self._startup_task.done():
            return

        async def _warmup() -> None:
            try:
                resp = await self._client.hello()
                payload: Any = resp.get("payload") or {}
                if self._identity and isinstance(payload, dict):
                    self._identity.save_hello_response(cast(dict[str, str | int | None], payload))
                await self._fetcher.fetch_promoted()
            except Exception:
                pass

        self._startup_task = asyncio.create_task(_warmup(), name="evomap-startup-warmup")

    async def on_stop(self) -> None:
        if self._startup_task and not self._startup_task.done():
            self._startup_task.cancel()
            try:
                await self._startup_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.close()


def plugin() -> EvoMapPlugin:
    """Plugin entry point for loader."""
    return EvoMapPlugin()
