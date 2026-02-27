"""Agent main loop — message -> LLM -> tool -> reply (multi-turn).

The loop is provider-agnostic.  Tool invocation follows a single code
path regardless of whether the provider supports native ``tools`` API:

* **Native mode** — tool schemas are passed via ``tools=`` parameter;
  the provider returns structured ``ToolCall`` objects in the response.
* **Fallback mode** — tool descriptions are injected into the system
  prompt; the LLM outputs a JSON block which the loop parses.
"""

import asyncio
import json
import re
from typing import TYPE_CHECKING, cast

from whaleclaw.agent.context import OnToolCall, OnToolResult
from whaleclaw.agent.prompt import PromptAssembler
from whaleclaw.config.schema import (
    AgentConfig,
    ModelsConfig,
    SummarizerConfig,
    WhaleclawConfig,
)
from whaleclaw.providers.base import AgentResponse, ImageContent, Message, ToolCall
from whaleclaw.providers.router import ModelRouter
from whaleclaw.sessions.compressor import ContextCompressor
from whaleclaw.sessions.context_window import RECENT_PROTECTED, ContextWindow
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.sessions.store import SessionStore
from whaleclaw.tools.base import ToolDefinition, ToolResult
from whaleclaw.tools.registry import ToolRegistry
from whaleclaw.types import StreamCallback
from whaleclaw.utils.log import get_logger

if TYPE_CHECKING:
    from whaleclaw.cron.scheduler import CronScheduler
    from whaleclaw.memory.base import MemoryStore
    from whaleclaw.memory.manager import MemoryManager
    from whaleclaw.sessions.group_compressor import SessionGroupCompressor

log = get_logger(__name__)

_assembler = PromptAssembler()
_context_window = ContextWindow()
_compressor = ContextCompressor()
_memory_organizer_tasks: dict[str, asyncio.Task[None]] = {}

_MAX_OUTPUT_TOKENS = 200_000
_EVOMAP_MAX_TOKENS = 1000
_EXTRA_MEMORY_COMPRESS_TIMEOUT_SECONDS = 8

_IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\((/[^)]+)\)")

_TOOL_HINTS: dict[str, str] = {
    "browser": "搜索相关资料",
    "desktop_capture": "点亮并截图桌面",
    "bash": "执行命令",
    "file_write": "生成文件",
    "file_read": "读取文件",
    "file_edit": "编辑文件",
    "memory_search": "检索长期记忆",
    "memory_add": "写入长期记忆",
    "memory_list": "查看长期记忆",
    "skill": "查找技能",
}

_CORE_NATIVE_TOOLS: set[str] = {
    "bash",
    "file_read",
    "file_write",
    "file_edit",
    "browser",
}
_MAX_NATIVE_TOOLS = 8
_TOOL_POLICY_KEYWORDS: dict[tuple[str, ...], set[str]] = {
    ("提醒", "定时", "闹钟", "计划", "cron", "schedule"): {"cron", "reminder"},
    ("记忆", "memory", "回忆", "记住"): {"memory_search", "memory_add", "memory_list"},
    ("技能", "skill", "安装技能"): {"skill"},
    ("会话", "session", "历史消息", "上下文"): {
        "sessions_list",
        "sessions_history",
        "sessions_send",
    },
    ("桌面", "截图", "截屏", "screen", "screenshot"): {"desktop_capture"},
}


def _preview_text(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())


def _score_tool_relevance(user_message: str, tool: ToolDefinition) -> int:
    query = _normalize_for_match(user_message)
    if not query:
        return 0

    score = 0
    corpus_parts = [tool.name, tool.description]
    corpus_parts.extend(p.name for p in tool.parameters)
    corpus_parts.extend(p.description for p in tool.parameters)
    corpus = _normalize_for_match(" ".join(corpus_parts))

    keywords = {w for w in re.findall(r"[\w\u4e00-\u9fff]{2,}", query)}
    for kw in keywords:
        if kw and kw in corpus:
            score += 1

    if (
        ("图片" in query or "photo" in query or "image" in query)
        and tool.name in {"browser", "desktop_capture"}
    ):
        score += 2

    return score


def _select_native_tool_names(registry: ToolRegistry, user_message: str) -> set[str]:
    defs = registry.list_tools()
    available = {d.name for d in defs}
    selected = {name for name in _CORE_NATIVE_TOOLS if name in available}
    query = _normalize_for_match(user_message)

    for keywords, names in _TOOL_POLICY_KEYWORDS.items():
        if any(k in query for k in keywords):
            for name in names:
                if name in available:
                    selected.add(name)

    if len(selected) >= _MAX_NATIVE_TOOLS:
        return selected

    scored: list[tuple[int, str]] = []
    for d in defs:
        if d.name in selected:
            continue
        scored.append((_score_tool_relevance(user_message, d), d.name))

    scored.sort(key=lambda x: (-x[0], x[1]))
    for score, name in scored:
        if score <= 0:
            break
        selected.add(name)
        if len(selected) >= _MAX_NATIVE_TOOLS:
            break

    return selected


def _build_memory_system_message(recalled: str) -> Message:
    """Wrap recalled memory as durable preference/fact context."""
    return Message(
        role="system",
        content=(
            "以下是从长期记忆召回的历史信息，包含用户长期偏好、稳定约束与历史事实。\n"
            "执行规则：\n"
            "1) 若内容属于长期偏好/写作与产出规则，且不与本轮用户要求冲突，请默认执行；\n"
            "2) 若内容属于历史事实且你不确定当前是否仍然有效，可先向用户确认。\n"
            f"{recalled}"
        ),
    )


def _build_global_style_system_message(style_directive: str) -> Message:
    return Message(
        role="system",
        content=(
            "以下是用户长期稳定的全局回复风格偏好，请默认遵守：\n"
            f"{style_directive.strip()}\n"
            "若用户在本轮消息中明确提出不同风格/长度要求，以本轮用户要求为准。"
        ),
    )


def _build_external_memory_system_message(extra_memory: str) -> Message:
    return Message(
        role="system",
        content=(
            "以下是来自协作网络的外部经验候选，仅作为补充参考：\n"
            f"{extra_memory.strip()}\n"
            "若与用户本轮明确要求冲突，以用户本轮要求为准；"
            "若与本地长期记忆冲突，以本地长期记忆为准。"
        ),
    )


def _est_tokens(text: str) -> int:
    return max(0, len(text) // 3)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    char_cap = max_tokens * 3
    if len(text) <= char_cap:
        return text
    return text[:char_cap]


async def _compress_external_memory_with_llm(
    *,
    router: ModelRouter,
    model_id: str,
    text: str,
    max_tokens: int,
) -> str:
    sys_prompt = (
        "你是外部经验压缩器。"
        "请将输入经验压缩到给定 token 上限内，保留可执行做法与约束。"
        "禁止新增事实。输出纯文本。"
    )
    user_prompt = (
        f"目标上限约 {max_tokens} tokens。\n"
        "输入如下：\n"
        f"{text}\n\n"
        "请输出压缩结果。"
    )
    try:
        resp = await router.chat(
            model_id,
            [
                Message(role="system", content=sys_prompt),
                Message(role="user", content=user_prompt),
            ],
        )
    except Exception:
        return ""
    out = resp.content.strip()
    if not out:
        return ""
    if _est_tokens(out) <= max_tokens:
        return out
    return _truncate_to_tokens(out, max_tokens)


def _merge_recall_blocks(profile: str, raw: str) -> str:
    blocks = [x.strip() for x in (profile, raw) if x.strip()]
    return "\n".join(blocks)


def _is_creation_task_message(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return False
    keys = (
        "ppt", "幻灯片", "演示文稿", "文档", "报告", "方案", "写", "生成", "制作", "整理",
        "设计", "润色", "总结", "改写", "脚本", "代码", "html", "页面", "海报", "计划",
        "create", "generate", "draft", "design", "write", "build", "compose",
    )
    return any(k in low for k in keys)


def _schedule_memory_organizer_task(
    session_id: str,
    *,
    memory_manager: "MemoryManager",
    router: ModelRouter,
    model_id: str,
    organizer_min_new_entries: int,
    organizer_interval_seconds: int,
    organizer_max_raw_window: int,
    keep_profile_versions: int,
    max_raw_entries: int,
) -> None:
    running = _memory_organizer_tasks.get(session_id)
    if running is not None and not running.done():
        return

    async def _run() -> None:
        try:
            organized = await memory_manager.organize_if_needed(
                router=router,
                model_id=model_id,
                organizer_min_new_entries=organizer_min_new_entries,
                organizer_interval_seconds=organizer_interval_seconds,
                organizer_max_raw_window=organizer_max_raw_window,
                keep_profile_versions=keep_profile_versions,
                max_raw_entries=max_raw_entries,
            )
            if organized:
                log.info("agent.memory_organized", session_id=session_id)
        except Exception as exc:
            log.debug("agent.memory_organize_failed", error=str(exc), session_id=session_id)

    task = asyncio.create_task(_run(), name=f"memory-organizer:{session_id}")
    _memory_organizer_tasks[session_id] = task

    def _cleanup(_task: asyncio.Task[None]) -> None:
        current = _memory_organizer_tasks.get(session_id)
        if current is _task:
            _memory_organizer_tasks.pop(session_id, None)

    task.add_done_callback(_cleanup)


def _make_plan_hint(tool_names: list[str], user_msg: str) -> str:
    """Generate a brief plan message when LLM jumps straight to tool calls."""
    steps: list[str] = []
    seen: set[str] = set()
    for name in tool_names:
        if name in seen:
            continue
        seen.add(name)
        steps.append(_TOOL_HINTS.get(name, f"调用 {name}"))
    plan = "、".join(steps)
    return f"好的，我来处理。正在{plan}…\n\n"


def _fix_image_paths(text: str, known_paths: list[str] | None = None) -> str:
    """Validate image paths in markdown; fix fabricated paths using known real ones."""
    from pathlib import Path

    unused_real = list(known_paths or [])

    def _replace(m: re.Match[str]) -> str:
        alt, raw_path = m.group(1), m.group(2)
        fp = Path(raw_path)
        if fp.is_file():
            return m.group(0)

        # Priority 1: match from tool-returned real paths (by hash or order)
        for i, real in enumerate(unused_real):
            rp = Path(real)
            if rp.is_file():
                unused_real.pop(i)
                log.info("fix_image_path.known", original=raw_path, found=real)
                return f"![{alt}]({real})"

        # Priority 2: fuzzy match by hash suffix
        stem = fp.stem
        hash_m = re.search(r"_([0-9a-f]{6,8})$", stem)
        if hash_m and fp.parent.is_dir():
            suffix = hash_m.group(0) + fp.suffix
            for candidate in fp.parent.iterdir():
                if candidate.name.endswith(suffix) and candidate.is_file():
                    log.info("fix_image_path.fuzzy", original=raw_path, found=str(candidate))
                    return f"![{alt}]({candidate})"

        # Priority 3: most recent file in same directory
        if fp.parent.is_dir():
            files = sorted(
                (f for f in fp.parent.iterdir() if f.is_file() and f.suffix == fp.suffix),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if files:
                best = files[0]
                log.info("fix_image_path.recent", original=raw_path, found=str(best))
                return f"![{alt}]({best})"

        log.warning("fix_image_path.removed", path=raw_path)
        return f"[图片未找到: {alt}]"

    return _IMG_MD_RE.sub(_replace, text)


def create_default_registry(
    session_manager: SessionManager | None = None,
    cron_scheduler: "CronScheduler | None" = None,
    *,
    memory_manager: "MemoryManager | None" = None,
    memory_store: "MemoryStore | None" = None,
) -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools registered.

    Args:
        session_manager: Optional SessionManager for session tools.
        cron_scheduler: Optional CronScheduler for cron/reminder tools.
    """
    from whaleclaw.tools.bash import BashTool
    from whaleclaw.tools.browser import BrowserTool
    from whaleclaw.tools.desktop_capture import DesktopCaptureTool
    from whaleclaw.tools.file_edit import FileEditTool
    from whaleclaw.tools.file_read import FileReadTool
    from whaleclaw.tools.file_write import FileWriteTool

    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(BrowserTool())
    registry.register(DesktopCaptureTool())

    if session_manager is not None:
        from whaleclaw.tools.sessions import (
            SessionsHistoryTool,
            SessionsListTool,
            SessionsSendTool,
        )

        registry.register(SessionsListTool(session_manager))
        registry.register(SessionsHistoryTool(session_manager))
        registry.register(SessionsSendTool(session_manager))

    if cron_scheduler is not None:
        from whaleclaw.tools.cron_tool import CronManageTool
        from whaleclaw.tools.reminder import ReminderTool

        registry.register(CronManageTool(cron_scheduler))
        registry.register(ReminderTool(cron_scheduler))

    from whaleclaw.skills.manager import SkillManager
    from whaleclaw.tools.skill_tool import SkillManageTool

    registry.register(SkillManageTool(SkillManager()))

    if memory_manager is not None:
        from whaleclaw.tools.memory_tool import MemoryAddTool, MemorySearchTool

        registry.register(MemorySearchTool(memory_manager))
        registry.register(MemoryAddTool(memory_manager))
    if memory_store is not None:
        from whaleclaw.tools.memory_tool import MemoryListTool

        registry.register(MemoryListTool(memory_store))

    return registry


def _parse_fallback_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from LLM text output (fallback mode).

    Looks for JSON objects with ``"tool"`` key, either fenced or bare.
    """
    calls: list[ToolCall] = []

    fenced = re.findall(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    candidates: list[str] = list(fenced)

    if not candidates:
        for match in re.finditer(r"\{[^{}]*\"tool\"[^{}]*\{[^}]*\}[^}]*\}", text):
            candidates.append(match.group(0))
        for match in re.finditer(r"\{[^{}]*\"tool\"[^{}]*\}", text):
            candidates.append(match.group(0))

    for raw in candidates:
        raw = raw.strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        d = cast(dict[str, object], obj)
        raw_name = d.get("tool", "")
        raw_args = d.get("arguments", {})
        if isinstance(raw_name, str) and raw_name and isinstance(raw_args, dict):
            calls.append(ToolCall(
                id=f"fallback_{len(calls)}",
                name=raw_name,
                arguments=raw_args,
            ))

    return calls


def _strip_tool_json(text: str) -> str:
    """Remove tool-call JSON blocks from text for clean display."""
    cleaned = re.sub(
        r'```(?:json)?\s*\n?\s*\{[^`]*"tool"\s*:[^`]*\}\s*\n?\s*```',
        "",
        text,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r'\{\s*"tool"\s*:\s*"[^"]*"[^}]*\}',
        "",
        cleaned,
    )
    return cleaned.strip()


async def _persist_message(
    manager: SessionManager,
    session: Session,
    role: str,
    content: str,
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> None:
    """Persist a message to session store (best-effort, never raises)."""
    try:
        await manager.add_message(
            session, role, content,
            tool_call_id=tool_call_id, tool_name=tool_name,
        )
    except Exception as exc:
        log.debug("agent.persist_failed", error=str(exc))


async def _execute_tool(
    registry: ToolRegistry,
    tc: ToolCall,
    on_tool_call: OnToolCall | None,
    on_tool_result: OnToolResult | None,
) -> tuple[str, ToolResult]:
    """Execute a single tool call and return (tool_call_id, result)."""
    if on_tool_call:
        await on_tool_call(tc.name, tc.arguments)

    import time
    t0 = time.monotonic()

    tool = registry.get(tc.name)
    if tool is None:
        result = ToolResult(
            success=False,
            output="",
            error=f"未知工具: {tc.name}",
        )
    else:
        try:
            result = await tool.execute(**tc.arguments)
        except Exception as exc:
            result = ToolResult(success=False, output="", error=str(exc))

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    log.info(
        "agent.tool_exec",
        tool=tc.name,
        success=result.success,
        elapsed_ms=elapsed_ms,
        args_preview=str(tc.arguments)[:200],
    )

    if on_tool_result:
        await on_tool_result(tc.name, result)

    return tc.id, result


def _format_tool_output(result: ToolResult) -> str:
    """Format ToolResult into a string for LLM consumption."""
    if result.success:
        return result.output or "(empty output)"
    return f"[ERROR] {result.error or 'unknown error'}\n{result.output}".strip()


def _validate_tool_call_args(tc: ToolCall, registry: ToolRegistry) -> str | None:
    """Return validation error text for invalid tool call args, else None."""
    tool = registry.get(tc.name)
    if tool is None:
        return None
    for p in tool.definition.parameters:
        if not p.required:
            continue
        if p.name not in tc.arguments:
            return f"{tc.name}.{p.name} 缺失"
        val = tc.arguments.get(p.name)
        if isinstance(val, str) and not val.strip():
            return f"{tc.name}.{p.name} 为空"
    if tc.name == "browser":
        action = str(tc.arguments.get("action", "")).strip()
        if not action:
            return "browser.action 为空"
        action_reqs: dict[str, tuple[str, ...]] = {
            "navigate": ("url",),
            "click": ("selector",),
            "type": ("selector", "text"),
            "evaluate": ("script",),
            "search_images": ("text",),
        }
        required_fields = action_reqs.get(action, ())
        for field in required_fields:
            value = tc.arguments.get(field)
            if isinstance(value, str):
                if not value.strip():
                    return f"browser.{field} 为空"
            elif value is None:
                return f"browser.{field} 缺失"
    if tc.name == "file_edit":
        old_s = tc.arguments.get("old_string")
        new_s = tc.arguments.get("new_string")
        for field_name, field_val in (("old_string", old_s), ("new_string", new_s)):
            if not isinstance(field_val, str):
                continue
            # If the payload is mostly literal escape sequences (\\n/\\t),
            # model likely attempted a brittle large block replace via JSON string.
            # Force retry with file_write full rewrite to avoid corrupt edits.
            if field_val.count("\\n") >= 3 and "\n" not in field_val:
                return f"file_edit.{field_name} 疑似转义块文本，改用 file_write 重写文件"
    return None


def _is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _first_non_empty_arg(arguments: dict[str, object], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = arguments.get(name)
        if _is_non_empty_str(value):
            return str(value).strip()
    return None


def _looks_like_image_request(text: str) -> bool:
    lower = text.lower()
    keywords = (
        "图", "图片", "照片", "近照", "头像", "搜图", "找图", "壁纸", "海报", "背景", "写真",
        "image", "photo", "picture", "wallpaper", "poster",
    )
    return any(k.lower() in lower for k in keywords)


def _is_garbled_query(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    escaped_noise = t.count("\\n") + t.count("\\t") + t.count("\\r") + t.count("\\x")
    if escaped_noise >= 2:
        return True
    if (
        re.search(r"(?:\\+[nrt]\d*){2,}", t)
        or re.search(r"(?:\n\d*){2,}", t)
        or t.count("\\x") >= 2
    ):
        return True
    return bool(len(t) > 40 and len(set(t)) < 10)


def _repair_tool_call(tc: ToolCall, user_message: str) -> tuple[ToolCall, str | None]:
    """Best-effort deterministic arg repair for common malformed tool calls."""
    args: dict[str, object] = dict(tc.arguments)
    changed = False
    reasons: list[str] = []

    # normalize common aliases
    alias_by_tool: dict[str, dict[str, tuple[str, ...]]] = {
        "bash": {"command": ("cmd", "script", "shell")},
        "browser": {"text": ("query", "keyword", "keywords", "q"), "selector": ("css",)},
        "file_read": {"path": ("file", "file_path")},
        "file_write": {"path": ("file", "file_path"), "content": ("text", "code")},
        "file_edit": {
            "path": ("file", "file_path"),
            "old_string": ("old", "old_text"),
            "new_string": ("new", "new_text"),
        },
    }
    alias_map = alias_by_tool.get(tc.name, {})
    for canonical, aliases in alias_map.items():
        if _is_non_empty_str(args.get(canonical)):
            continue
        value = _first_non_empty_arg(args, aliases)
        if value is None:
            continue
        args[canonical] = value
        changed = True
        reasons.append(f"{canonical}<-alias")

    if tc.name == "browser":
        action = str(args.get("action", "")).strip().lower()
        inferred_action: str | None = None

        if not action:
            if _is_non_empty_str(args.get("url")):
                inferred_action = "navigate"
            elif _is_non_empty_str(args.get("script")):
                inferred_action = "evaluate"
            elif _is_non_empty_str(args.get("selector")) and _is_non_empty_str(args.get("text")):
                inferred_action = "type"
            elif _is_non_empty_str(args.get("selector")):
                if any(k in user_message for k in ("读取", "提取", "文本", "内容", "text")):
                    inferred_action = "get_text"
                else:
                    inferred_action = "click"
            else:
                query = _first_non_empty_arg(args, ("text", "query", "keyword", "keywords", "q"))
                if query is not None and _looks_like_image_request(user_message):
                    args["text"] = query
                    inferred_action = "search_images"

            if inferred_action is not None:
                args["action"] = inferred_action
                action = inferred_action
                changed = True
                reasons.append("action<-inferred")

        if action == "navigate" and not _is_non_empty_str(args.get("url")):
            candidate = _first_non_empty_arg(args, ("text",))
            if candidate and candidate.startswith(("http://", "https://")):
                args["url"] = candidate
                changed = True
                reasons.append("url<-text")
        if action == "search_images":
            q = str(args.get("text", "")).strip()
            if _is_garbled_query(q) and _looks_like_image_request(user_message):
                args["text"] = user_message.strip()
                changed = True
                reasons.append("text<-user_message")

    if not changed:
        return tc, None
    return ToolCall(id=tc.id, name=tc.name, arguments=args), ",".join(reasons)


async def run_agent(
    message: str,
    session_id: str,
    config: WhaleclawConfig,
    on_stream: StreamCallback | None = None,
    *,
    session: Session | None = None,
    router: ModelRouter | None = None,
    registry: ToolRegistry | None = None,
    on_tool_call: OnToolCall | None = None,
    on_tool_result: OnToolResult | None = None,
    images: list[ImageContent] | None = None,
    session_manager: SessionManager | None = None,
    session_store: SessionStore | None = None,
    memory_manager: "MemoryManager | None" = None,
    extra_memory: str = "",
    trigger_event_id: str = "",
    trigger_text_preview: str = "",
    group_compressor: "SessionGroupCompressor | None" = None,
) -> str:
    """Run the Agent loop with tool support and multi-turn context.

    The loop is provider-agnostic:
    1. Check if provider supports native tools API
    2. If yes  -> pass schemas via ``tools=``; parse structured tool_calls
    3. If no   -> inject tool descriptions into system prompt; parse JSON text
    4. Execute tools, append results, loop (until no tool calls or token budget exhausted)
    5. Return final text reply
    """
    agent_cfg = cast(AgentConfig, config.agent)
    models_cfg = cast(ModelsConfig, config.models)
    summarizer_cfg = cast(SummarizerConfig, agent_cfg.summarizer)

    model_id: str = session.model if session else agent_cfg.model
    if router is None:
        router = ModelRouter(models_cfg)
    if registry is None:
        registry = create_default_registry()

    native_tools = router.supports_native_tools(model_id)
    selected_tool_names: set[str] | None = None
    if native_tools:
        selected_tool_names = _select_native_tool_names(registry, message)
        tool_schemas = registry.to_llm_schemas(include_names=selected_tool_names)
        all_tool_names = {d.name for d in registry.list_tools()}
        dropped_names = sorted(all_tool_names - selected_tool_names)
        log.info(
            "agent.tools_selected",
            session_id=session_id,
            selected=sorted(selected_tool_names),
            selected_count=len(selected_tool_names),
            dropped_count=len(dropped_names),
            dropped=dropped_names,
        )
    else:
        tool_schemas = None
    fallback_text = "" if native_tools else registry.to_prompt_fallback()

    system_messages = _assembler.build(
        config, message, tool_fallback_text=fallback_text
    )
    import time as _time

    _t_pre_llm_start = _time.monotonic()
    _memory_stage_ms = 0
    _extra_memory_stage_ms = 0
    _group_compress_stage_ms = 0
    if memory_manager is not None and agent_cfg.memory.enabled:
        _t_memory_start = _time.monotonic()
        memory_cfg = agent_cfg.memory
        try:
            style_directive = (
                await memory_manager.get_global_style_directive()
                if memory_cfg.global_style_enabled
                else ""
            )
            if style_directive:
                system_messages.append(_build_global_style_system_message(style_directive))
                log.info(
                    "agent.memory_style_applied",
                    session_id=session_id,
                    chars=len(style_directive),
                )
        except Exception as exc:
            log.debug("agent.memory_style_load_failed", error=str(exc), session_id=session_id)
        try:
            should_recall, include_raw = memory_manager.recall_policy(message)
            if not should_recall and _is_creation_task_message(message):
                should_recall = True
                include_raw = False
            recalled = ""
            if should_recall:
                profile_block = await memory_manager.build_profile_for_injection(
                    max_tokens=memory_cfg.recall_profile_max_tokens,
                    router=router,
                    model_id=memory_cfg.organizer_model,
                )
                raw_block = ""
                if include_raw:
                    raw_block = await memory_manager.recall(
                        message,
                        max_tokens=memory_cfg.recall_raw_max_tokens,
                        limit=memory_cfg.recall_limit,
                        include_profile=False,
                        include_raw=True,
                    )
                recalled = _merge_recall_blocks(profile_block, raw_block)
            if recalled.strip():
                system_messages.append(_build_memory_system_message(recalled))
                log.info("agent.memory_recalled", session_id=session_id, chars=len(recalled))
        except Exception as exc:
            log.debug("agent.memory_recall_failed", error=str(exc), session_id=session_id)
        _memory_stage_ms = int((_time.monotonic() - _t_memory_start) * 1000)
    if extra_memory.strip():
        _t_extra_start = _time.monotonic()
        normalized_extra = extra_memory.strip()
        compress_model = summarizer_cfg.model.strip()
        can_compress = bool(compress_model)
        should_compress_extra = _est_tokens(normalized_extra) > _EVOMAP_MAX_TOKENS
        if can_compress and should_compress_extra:
            try:
                router.resolve(compress_model)
                compressed = await asyncio.wait_for(
                    _compress_external_memory_with_llm(
                        router=router,
                        model_id=compress_model,
                        text=normalized_extra,
                        max_tokens=_EVOMAP_MAX_TOKENS,
                    ),
                    timeout=_EXTRA_MEMORY_COMPRESS_TIMEOUT_SECONDS,
                )
                if compressed:
                    normalized_extra = compressed
                else:
                    normalized_extra = _truncate_to_tokens(
                        normalized_extra,
                        _EVOMAP_MAX_TOKENS,
                    )
            except Exception:
                normalized_extra = _truncate_to_tokens(
                    normalized_extra,
                    _EVOMAP_MAX_TOKENS,
                )
        else:
            normalized_extra = _truncate_to_tokens(
                normalized_extra,
                _EVOMAP_MAX_TOKENS,
            )
        system_messages.append(_build_external_memory_system_message(normalized_extra))
        _extra_memory_stage_ms = int((_time.monotonic() - _t_extra_start) * 1000)

    conversation: list[Message] = []
    if session:
        conversation = list(session.messages)
    conversation.append(Message(role="user", content=message, images=images))
    conversation_message_count = len(conversation)

    if (
        group_compressor is not None
        and session_store is not None
        and session is not None
        and summarizer_cfg.model.strip()
    ):
        _t_group_start = _time.monotonic()
        try:
            conversation = await group_compressor.build_window_messages(
                session_id=session_id,
                messages=conversation,
                router=router,
                model_id=summarizer_cfg.model.strip(),
            )
        except Exception as exc:
            log.debug("agent.group_compress_failed", session_id=session_id, error=str(exc))
        _group_compress_stage_ms = int((_time.monotonic() - _t_group_start) * 1000)

    _pre_llm_elapsed_ms = int((_time.monotonic() - _t_pre_llm_start) * 1000)
    log.debug(
        "agent.pre_llm_stages",
        session_id=session_id,
        memory_ms=_memory_stage_ms,
        extra_memory_ms=_extra_memory_stage_ms,
        group_compress_ms=_group_compress_stage_ms,
        total_ms=_pre_llm_elapsed_ms,
    )

    model_short: str = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    trigger_preview = trigger_text_preview.strip() or _preview_text(message)

    log.info(
        "agent.run",
        model=model_id,
        session_id=session_id,
        native_tools=native_tools,
        history_messages=len(conversation),
        trigger_event_id=trigger_event_id,
        trigger_preview=trigger_preview,
    )

    final_text_parts: list[str] = []
    real_image_paths: list[str] = []
    total_input = 0
    total_output = 0
    announced_plan = False
    db_summaries = []
    if session_store and summarizer_cfg.enabled:
        try:
            db_summaries = await session_store.get_summaries(session_id)
        except Exception as exc:
            log.debug("agent.summaries_load_failed", error=str(exc))

    import hashlib as _hashlib

    _recent_signatures: list[str] = []
    _loop_detect_window = 3
    invalid_tool_rounds = 0
    empty_reply_rounds = 0
    browser_fail_streak = 0
    blocked_tools: set[str] = set()

    round_idx = -1
    while total_output < _MAX_OUTPUT_TOKENS:
        round_idx += 1
        if db_summaries:
            all_messages = _context_window.trim_with_summaries(
                [*system_messages, *conversation], model_short, db_summaries,
            )
        else:
            all_messages = _context_window.trim(
                [*system_messages, *conversation], model_short,
            )

        _llm_t0 = _time.monotonic()
        response: AgentResponse = await router.chat(
            model_id,
            all_messages,
            tools=tool_schemas or None,
            on_stream=on_stream,
        )
        _llm_ms = int((_time.monotonic() - _llm_t0) * 1000)
        round_input = response.input_tokens
        round_output = response.output_tokens
        total_input += round_input
        total_output += round_output
        log.info(
            "agent.llm_call",
            round=round_idx,
            elapsed_ms=_llm_ms,
            model=model_id,
            round_input_tokens=round_input,
            round_output_tokens=round_output,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            trigger_event_id=trigger_event_id,
            trigger_preview=trigger_preview,
            session_id=session_id,
        )

        tool_calls = response.tool_calls
        if not tool_calls and not native_tools and response.content:
            tool_calls = _parse_fallback_tool_calls(response.content)
        if tool_calls:
            valid_tool_calls = [tc for tc in tool_calls if tc.name.strip()]
            dropped = len(tool_calls) - len(valid_tool_calls)
            if dropped > 0:
                if valid_tool_calls:
                    log.debug(
                        "agent.invalid_tool_calls_dropped",
                        dropped=dropped,
                        kept=len(valid_tool_calls),
                        session_id=session_id,
                    )
                else:
                    log.warning(
                        "agent.invalid_tool_calls_dropped",
                        dropped=dropped,
                        kept=0,
                        session_id=session_id,
                    )
            repaired_calls: list[ToolCall] = []
            for tc in valid_tool_calls:
                repaired, reason = _repair_tool_call(tc, message)
                if reason:
                    log.info(
                        "agent.tool_call_repaired",
                        tool=tc.name,
                        reason=reason,
                        before=str(tc.arguments)[:200],
                        after=str(repaired.arguments)[:200],
                        session_id=session_id,
                    )
                repaired_calls.append(repaired)
            tool_calls = repaired_calls
            blocked_reasons = [
                f"{tc.name} 已熔断，禁止继续调用"
                for tc in tool_calls
                if tc.name in blocked_tools
            ]
            if blocked_reasons:
                invalid_tool_rounds += 1
                log.warning(
                    "agent.blocked_tool_calls_dropped",
                    reasons=blocked_reasons,
                    round=round_idx,
                    session_id=session_id,
                )
                conversation.append(
                    Message(
                        role="user",
                        content=(
                            "[系统提示] 该工具已被熔断："
                            f"{'; '.join(blocked_reasons)}。"
                            "请改用其他工具（例如 bash）完成任务。"
                        ),
                    )
                )
                if invalid_tool_rounds >= 2:
                    final_text_parts.append(
                        "工具调用连续无效，已停止自动重试。请明确参数后重试。"
                    )
                    break
                continue
            invalid_reasons = [
                reason
                for tc in tool_calls
                if (reason := _validate_tool_call_args(tc, registry)) is not None
            ]
            if invalid_reasons:
                invalid_tool_rounds += 1
                log.warning(
                    "agent.invalid_tool_call_args",
                    reasons=invalid_reasons,
                    round=round_idx,
                    session_id=session_id,
                )
                conversation.append(
                    Message(
                        role="user",
                        content=(
                            "[系统提示] 你上一轮的工具调用参数无效："
                            f"{'; '.join(invalid_reasons)}。"
                            "请只重发一个有效的工具调用，必填参数必须完整且非空。"
                        ),
                    )
                )
                if invalid_tool_rounds >= 2:
                    final_text_parts.append(
                        "工具调用参数连续无效，已停止自动重试。请明确参数后重试。"
                    )
                    break
                continue
            invalid_tool_rounds = 0

        content = response.content or ""
        if content:
            empty_reply_rounds = 0
            if tool_calls and not native_tools:
                clean = _strip_tool_json(content)
                if clean:
                    final_text_parts.append(clean)
            else:
                final_text_parts.append(content)

        if not tool_calls and not content.strip():
            empty_reply_rounds += 1
            log.warning(
                "agent.empty_response",
                session_id=session_id,
                round=round_idx,
                model=model_id,
                empty_reply_rounds=empty_reply_rounds,
            )
            if empty_reply_rounds == 1:
                conversation.append(
                    Message(
                        role="user",
                        content=(
                            "[系统提示] 你上一轮没有输出任何内容。"
                            "请直接给出简短回复；如果需求不明确，请先问用户要做什么。"
                        ),
                    )
                )
                continue
            final_text_parts.append(
                "我这边没收到模型有效回复。请再发一次需求，我会继续处理。"
            )
            break

        if not tool_calls:
            break

        if not announced_plan and on_stream:
            announced_plan = True
            has_text = content.strip() if content else ""
            if not has_text:
                tool_names = [tc.name for tc in tool_calls]
                plan = _make_plan_hint(tool_names, message)
                await on_stream(plan)

        log.info(
            "agent.tool_calls",
            round=round_idx,
            count=len(tool_calls),
            tools=[tc.name for tc in tool_calls],
        )

        tool_names_str = ", ".join(tc.name for tc in tool_calls)
        assistant_content = response.content or ""
        assistant_persist = (
            assistant_content
            if assistant_content
            else f"(调用工具: {tool_names_str}) {assistant_content}".strip()
        )
        assistant_msg = Message(
            role="assistant",
            content=assistant_content,
            tool_calls=tool_calls if native_tools else None,
        )
        conversation.append(assistant_msg)

        if session_manager and session:
            await _persist_message(session_manager, session, "assistant", assistant_persist)

        for _tname in ("reminder", "cron"):
            _tool = registry.get(_tname)
            if _tool is not None and hasattr(_tool, "current_session_id"):
                _tool.current_session_id = session_id  # type: ignore[union-attr]

        for tc in tool_calls:
            tc_id, result = await _execute_tool(
                registry, tc, on_tool_call, on_tool_result
            )
            if tc.name == "browser":
                if result.success:
                    browser_fail_streak = 0
                else:
                    browser_fail_streak += 1
                    if browser_fail_streak >= 2 and "browser" not in blocked_tools:
                        blocked_tools.add("browser")
                        log.warning(
                            "agent.tool_circuit_open",
                            tool="browser",
                            fail_streak=browser_fail_streak,
                            session_id=session_id,
                        )
                        conversation.append(
                            Message(
                                role="user",
                                content=(
                                    "[系统降级] browser 工具连续失败，已自动熔断。"
                                    "后续请不要再调用 browser。"
                                    "请改用 bash 工具执行可复现的命令行方案完成任务。"
                                ),
                            )
                        )

            if result.success and result.output:
                for path_match in re.finditer(
                    r"(/[^\s]+\.(?:jpg|jpeg|png|gif|webp))", result.output
                ):
                    real_image_paths.append(path_match.group(1))

            tool_output = _format_tool_output(result)

            if native_tools:
                tool_msg = Message(
                    role="tool",
                    content=tool_output,
                    tool_call_id=tc_id,
                )
            else:
                tool_msg = Message(
                    role="user",
                    content=(
                        f"[工具 {tc.name} 执行结果]\n"
                        f"{tool_output}"
                    ),
                )
            conversation.append(tool_msg)

            if session_manager and session:
                snippet = tool_output[:500] if len(tool_output) > 500 else tool_output
                await _persist_message(
                    session_manager, session, "tool",
                    f"[{tc.name}] {snippet}",
                    tool_call_id=tc_id, tool_name=tc.name,
                )

            log.debug(
                "agent.tool_result",
                tool=tc.name,
                success=result.success,
                output_len=len(result.output),
            )

        sig_parts = []
        for tc in tool_calls:
            arg_str = json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)[:200]
            sig_parts.append(f"{tc.name}:{arg_str}")
        round_sig = _hashlib.md5("|".join(sig_parts).encode()).hexdigest()  # noqa: S324
        _recent_signatures.append(round_sig)

        if len(_recent_signatures) >= _loop_detect_window:
            tail = _recent_signatures[-_loop_detect_window:]
            if len(set(tail)) == 1:
                log.warning(
                    "agent.loop_detected",
                    session_id=session_id,
                    rounds=round_idx + 1,
                    repeated_tool=tool_calls[0].name,
                )
                conversation.append(Message(
                    role="user",
                    content=(
                        "[系统提示] 检测到你在重复执行相同操作且未取得进展。"
                        "请换一种方式解决问题，或直接向用户说明当前遇到的困难。"
                    ),
                ))

        final_text_parts.clear()
    else:
        log.warning(
            "agent.token_budget_exhausted",
            session_id=session_id,
            rounds=round_idx + 1,
            total_output=total_output,
        )

    final_text = "".join(final_text_parts)
    final_text = _fix_image_paths(final_text, real_image_paths)

    # Background: generate L0/L1 summaries for older messages if needed
    if (
        session_store
        and router
        and summarizer_cfg.enabled
        and session
        and group_compressor is None
        and _compressor.should_compress(conversation_message_count)
    ):
        try:
            latest = await session_store.get_latest_summary(session_id, "L0")
            msg_rows = await session_store.get_messages(session_id)

            already_covered = latest.source_msg_end if latest else 0
            uncovered = [r for r in msg_rows if r.id > already_covered]
            protected = min(RECENT_PROTECTED, len(uncovered))
            to_compress = uncovered[:-protected] if protected < len(uncovered) else []

            if len(to_compress) >= 8:
                compress_msgs = [
                    Message(role=r.role if r.role != "tool" else "assistant", content=r.content)
                    for r in to_compress
                ]
                start_id = to_compress[0].id
                end_id = to_compress[-1].id
                _store_ref = session_store
                _router_ref = router
                _model_ref: str = summarizer_cfg.model

                async def _bg_compress() -> None:
                    try:
                        await _compressor.compress_segment(
                            session_id=session_id,
                            messages=compress_msgs,
                            msg_id_start=start_id,
                            msg_id_end=end_id,
                            store=_store_ref,
                            router=_router_ref,
                            model=_model_ref,
                        )
                    except Exception as exc:
                        log.debug("agent.bg_compress_failed", error=str(exc))

                asyncio.create_task(_bg_compress())
        except Exception as exc:
            log.debug("agent.bg_compress_prep_failed", error=str(exc))

    log.info(
        "agent.done",
        model=model_id,
        llm_rounds=max(0, round_idx + 1),
        input_tokens=total_input,
        output_tokens=total_output,
        session_id=session_id,
        trigger_event_id=trigger_event_id,
        trigger_preview=trigger_preview,
    )

    if session_store and total_input + total_output > 0:
        try:
            await session_store.record_token_usage(
                session_id=session_id,
                model=model_id,
                input_tokens=total_input,
                output_tokens=total_output,
            )
        except Exception:
            log.debug("agent.token_usage_save_failed")

    if memory_manager is not None and agent_cfg.memory.enabled:
        memory_cfg = agent_cfg.memory
        captured = False
        organized = False
        organizer_ready = True
        try:
            captured = await memory_manager.auto_capture_user_message(
                message,
                source=f"session:{session_id}",
                mode=memory_cfg.auto_capture_mode,
                cooldown_seconds=memory_cfg.cooldown_seconds,
                max_per_hour=memory_cfg.max_per_hour,
                batch_size=memory_cfg.capture_batch_size,
                merge_window_seconds=memory_cfg.capture_merge_window_seconds,
            )
            if captured:
                log.info("agent.memory_captured", session_id=session_id)
        except Exception as exc:
            log.debug("agent.memory_capture_failed", error=str(exc), session_id=session_id)
        if memory_cfg.organizer_enabled:
            try:
                router.resolve(memory_cfg.organizer_model)
            except Exception:
                organizer_ready = False
            if memory_cfg.organizer_background:
                if organizer_ready:
                    _schedule_memory_organizer_task(
                        session_id,
                        memory_manager=memory_manager,
                        router=router,
                        model_id=memory_cfg.organizer_model,
                        organizer_min_new_entries=memory_cfg.organizer_min_new_entries,
                        organizer_interval_seconds=memory_cfg.organizer_interval_seconds,
                        organizer_max_raw_window=memory_cfg.organizer_max_raw_window,
                        keep_profile_versions=memory_cfg.keep_profile_versions,
                        max_raw_entries=memory_cfg.max_raw_entries,
                    )
            else:
                try:
                    organized = await memory_manager.organize_if_needed(
                        router=router,
                        model_id=memory_cfg.organizer_model,
                        organizer_min_new_entries=memory_cfg.organizer_min_new_entries,
                        organizer_interval_seconds=memory_cfg.organizer_interval_seconds,
                        organizer_max_raw_window=memory_cfg.organizer_max_raw_window,
                        keep_profile_versions=memory_cfg.keep_profile_versions,
                        max_raw_entries=memory_cfg.max_raw_entries,
                    )
                    if organized:
                        log.info("agent.memory_organized", session_id=session_id)
                except Exception as exc:
                    organizer_ready = False
                    log.debug("agent.memory_organize_failed", error=str(exc), session_id=session_id)

        if captured and (not memory_cfg.organizer_enabled or not organizer_ready or not organized):
            try:
                updated = await memory_manager.upsert_profile_from_capture(
                    message,
                    router=router,
                    model_id=model_id,
                    max_tokens=memory_cfg.recall_profile_max_tokens,
                    keep_profile_versions=memory_cfg.keep_profile_versions,
                )
                if updated:
                    log.info("agent.memory_profile_fallback_updated", session_id=session_id)
            except Exception as exc:
                log.debug(
                    "agent.memory_profile_fallback_failed",
                    error=str(exc),
                    session_id=session_id,
                )

    return final_text
