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
import os
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from whaleclaw.agent.context import OnToolCall, OnToolResult
from whaleclaw.agent.prompt import PromptAssembler
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import AgentResponse, ImageContent, Message, ToolCall
from whaleclaw.providers.router import ModelRouter
from whaleclaw.sessions.compressor import ContextCompressor
from whaleclaw.sessions.context_window import RECENT_PROTECTED, ContextWindow
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.sessions.store import SessionStore
from whaleclaw.skills.parser import Skill, SkillParamItem
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

OnRoundResult = Callable[[int, str], Awaitable[None]]

_assembler = PromptAssembler()
_context_window = ContextWindow()
_compressor = ContextCompressor()
_memory_organizer_tasks: dict[str, asyncio.Task[None]] = {}

_MAX_OUTPUT_TOKENS = 200_000
_EVOMAP_MAX_TOKENS = 1000
_EXTRA_MEMORY_COMPRESS_TIMEOUT_SECONDS = 8
_DEFAULT_ASSISTANT_NAME = "WhaleClaw"

_IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\((/[^)]+)\)")
_EVOMAP_LINE_RE = re.compile(r"^\s*-\s*([^:]+):\s*(.+?)\s*$")
_OFFICE_PATH_RE = re.compile(r"(/[^\n\"']+\.(?:pptx|docx|xlsx))", re.IGNORECASE)
_ABS_FILE_PATH_RE = re.compile(r"(/[^\s\"')]+?\.[A-Za-z0-9]{1,8})(?=[\s\"')]|$)")
_NON_DELIVERY_EXTS = {".py", ".sh", ".bash", ".zsh", ".log", ".tmp"}
_VERSION_SUFFIX_RE = re.compile(r"_V\d+$", re.IGNORECASE)
_COORDINATOR_ASK_RE = re.compile(
    r"(?:你要(?:我|什么)|需要你(?:提供|告诉|回复|回答|确认|选择)|"
    r"请(?:告诉|选择|提供|告知)我|"
    r"(?:按|用)(?:下面|以下)(?:模板|格式)(?:回|填|答))",
)
_EVOMAP_CHOICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:选|选择)?\s*([ABCabc])\s*$"),
    re.compile(r"^\s*(?:选|选择)?\s*([123])\s*$"),
)
_ASSISTANT_NAME_RESET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"恢复默认名字"),
    re.compile(r"改回\s*whaleclaw", re.IGNORECASE),
    re.compile(r"还是叫\s*whaleclaw", re.IGNORECASE),
)
_ASSISTANT_NAME_SET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:以后|从现在起|今后|之后|开始)\s*(?:你|机器人|助手)?\s*(?:就)?(?:叫|改叫|改名叫)\s*([^\s，。！？!?、,]{1,24})"
    ),
    re.compile(r"(?:把你|你)\s*(?:改名|改名为|名字改成|名字改为)\s*([^\s，。！？!?、,]{1,24})"),
    re.compile(r"^\s*(?:你|助手|机器人)\s*(?:就)?叫\s*([^\s，。！？!?、,]{1,24})\s*$"),
)
_USE_CMD_RE = re.compile(r"^\s*/use\s+([^\s]+)\s*(.*)$", re.IGNORECASE | re.DOTALL)
_USE_CLEAR_IDS = {"clear", "none", "off", "default", "reset"}
_TASK_DONE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*任务完成\s*$"),
    re.compile(r"^\s*完成了?\s*$"),
    re.compile(r"^\s*结束了?\s*$"),
    re.compile(r"^\s*可以了?\s*$"),
    re.compile(r"^\s*ok\s*$", re.IGNORECASE),
)
_SKILL_SWITCH_CONSENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:同意|可以|确认|允许).{0,8}(?:切换|换).{0,8}(?:技能|skill)?", re.IGNORECASE),
    re.compile(r"(?:切换|换).{0,8}(?:技能|skill).{0,8}(?:吧|可以|行|好的|ok)", re.IGNORECASE),
    re.compile(r"(?:换成|改用|切到).{0,24}(?:技能|skill)", re.IGNORECASE),
)
_SKILL_ACTIVATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:使用|调用|启动|启用|走|用).{0,24}(?:技能|skill)", re.IGNORECASE),
    re.compile(r"(?:技能|skill).{0,16}(?:文生图|图生图|处理|执行|联调)", re.IGNORECASE),
)
_MULTI_AGENT_CONFIRM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(确认|确认开始|开始执行|开始多agent|开始多\s*agent|执行吧|开始吧)\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*/multi\s+go\s*$", re.IGNORECASE),
)
_MULTI_AGENT_CANCEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(取消|取消执行|先别执行|暂停|停止)\s*$", re.IGNORECASE),
    re.compile(r"^\s*/multi\s+cancel\s*$", re.IGNORECASE),
)
_MULTI_AGENT_ROUNDS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:改为|改成|设为|设置为|设置|调整为|改到)\s*(\d{1,2})\s*轮",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(\d{1,2})\s*轮\s*$", re.IGNORECASE),
    re.compile(r"^\s*/multi\s+rounds\s+(\d{1,2})\s*$", re.IGNORECASE),
)
_MULTI_AGENT_DISCUSS_DONE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(需求确认|确认需求|需求已确认|信息齐了|进入执行确认)\s*$"),
    re.compile(r"^\s*/multi\s+ready\s*$", re.IGNORECASE),
)
_MULTI_AGENT_SCENARIO_LABELS: dict[str, str] = {
    "product_design": "产品设计",
    "content_creation": "内容创作",
    "software_development": "软件开发",
    "data_analysis_decision": "数据分析决策",
    "scientific_research": "科研",
    "intelligent_assistant": "智能助理",
    "workflow_automation": "自动化工作流",
}

_TOOL_HINTS: dict[str, str] = {
    "browser": "搜索相关资料",
    "desktop_capture": "点亮并截图桌面",
    "bash": "执行命令",
    "file_write": "生成文件",
    "file_read": "读取文件",
    "file_edit": "编辑文件",
    "ppt_edit": "修改现有PPT",
    "docx_edit": "修改现有Word",
    "xlsx_edit": "修改现有Excel",
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
    ("改ppt", "修改ppt", "第一页", "第一面", "幻灯片", "slide", "pptx"): {"ppt_edit"},
    ("改word", "修改word", "docx", "文档", "段落"): {"docx_edit"},
    ("改excel", "修改excel", "xlsx", "表格", "单元格"): {"xlsx_edit"},
    ("evomap", "evo map", "进化市场", "协作经验", "方案库"): {"evomap_fetch"},
    ("赏金", "bounty", "任务认领", "claim task", "认领任务"): {"evomap_bounty"},
    ("发布经验", "发布到evomap", "publish evomap", "贡献方案"): {"evomap_publish"},
}


def _preview_text(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _sanitize_assistant_name(raw: str) -> str:
    name = raw.strip().strip("\"'“”‘’「」[]()（）")
    if not name:
        return ""
    if any(ch in name for ch in ("?", "？", "!", "！", "吗")):
        return ""
    low = name.lower()
    if low in {"什么", "啥", "name", "名字"}:
        return ""
    if "什么" in name:
        return ""
    if len(name) > 24:
        return ""
    if not re.fullmatch(r"[\w\u4e00-\u9fff·\-.]{1,24}", name):
        return ""
    return name


def _detect_assistant_name_update(message: str) -> tuple[str, str]:
    text = message.strip()
    if not text:
        return ("none", "")
    for p in _ASSISTANT_NAME_RESET_PATTERNS:
        if p.search(text):
            return ("reset", "")
    for p in _ASSISTANT_NAME_SET_PATTERNS:
        m = p.search(text)
        if not m:
            continue
        name = _sanitize_assistant_name(m.group(1))
        if name:
            return ("set", name)
    return ("none", "")


def _normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())


def _parse_use_command(text: str) -> tuple[list[str], str] | None:
    m = _USE_CMD_RE.match(text)
    if not m:
        return None
    token = m.group(1).strip().lower()
    skill_ids = [x.strip() for x in token.split(",") if x.strip()]
    remainder = m.group(2).strip()
    return (skill_ids, remainder)


def _is_task_done_confirmation(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in _TASK_DONE_PATTERNS)


def _build_skill_lock_system_message(skill_ids: list[str]) -> Message:
    joined = ", ".join(skill_ids)
    return Message(
        role="system",
        content=(
            f"当前会话已锁定技能：{joined}。\n"
            "执行时仅允许在这些技能范围内规划与调用，不要偏移到无关方案。\n"
            "若需切换到其它技能，必须先征得用户明确同意。\n"
            "若用户明确回复“任务完成”，再解除该锁定。"
        ),
    )


def _looks_like_skill_activation_message(text: str) -> bool:  # pyright: ignore[reportUnusedFunction]
    t = text.strip()
    if not t:
        return False
    if t.lower().startswith("/use "):
        return True
    return any(p.search(t) for p in _SKILL_ACTIVATION_PATTERNS)


def _is_skill_switch_consent(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in _SKILL_SWITCH_CONSENT_PATTERNS)


def _normalize_skill_ids(skills: list[Skill]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in skills:
        sid = s.id.strip().lower()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _skill_announcement(skill_ids: list[str], previous_skill_ids: list[str]) -> str:
    joined = "、".join(skill_ids)
    if not previous_skill_ids:
        return f"我将使用 {joined} 技能继续完成任务。"
    if previous_skill_ids != skill_ids:
        return f"已按你的要求切换为 {joined} 技能，我继续处理。"
    return f"我会继续使用 {joined} 技能推进当前任务。"


def _skill_token_mentioned(token: str, text: str) -> bool:
    lower = text.lower()
    msg_norm = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", lower)

    base = token.strip().lower().replace("_", "-")
    if not base:
        return False

    candidates: set[str] = {base, base.replace("-", "")}
    parts = [p for p in base.split("-") if p]
    if len(parts) >= 2:
        short = "-".join(parts[:2])
        candidates.add(short)
        candidates.add(short.replace("-", ""))

    for token in candidates:
        if len(token) < 4:
            continue
        if token in lower or token in msg_norm:
            return True
    return False


def _skill_explicitly_mentioned(skill: Skill, text: str) -> bool:  # pyright: ignore[reportUnusedFunction]
    return _skill_token_mentioned(skill.id, text) or _skill_token_mentioned(skill.name, text)


def _extract_ratio_or_size(text: str) -> str:
    ratio = re.search(r"\b(\d{1,2}\s*:\s*\d{1,2})\b", text)
    if ratio:
        return ratio.group(1).replace(" ", "")
    size = re.search(r"\b(\d{3,5}\s*x\s*\d{3,5})\b", text, re.IGNORECASE)
    if size:
        return size.group(1).replace(" ", "").lower()
    return ""


def _extract_value_by_aliases(text: str, aliases: list[str]) -> str:
    for alias in aliases:
        pattern = rf"{re.escape(alias)}\s*(?:是|为|:|：)\s*(.+)$"
        m = re.search(pattern, text, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""


def _has_param_secret_source(param: SkillParamItem) -> bool:
    for env_name in param.env_vars:
        val = os.getenv(env_name, "").strip()
        if val:
            return True
    if param.saved_file:
        p = Path(param.saved_file).expanduser()
        try:
            return p.is_file() and bool(p.read_text(encoding="utf-8").strip())
        except Exception:
            return False
    return False


def _capture_param_value(
    param: SkillParamItem,
    text: str,
    images: list[ImageContent] | None,
    previous: object,
) -> object:
    ptype = param.type.strip().lower()
    aliases = [param.key, *param.aliases]
    if ptype == "images":
        prev_count = int(previous) if isinstance(previous, int) else 0
        return max(prev_count, len(images or []))
    if ptype in {"ratio", "size"}:
        val = _extract_ratio_or_size(text) or _extract_value_by_aliases(text, aliases)
        return val or previous
    if ptype == "api_key":
        if re.search(r"\bsk-[A-Za-z0-9_-]{12,}\b", text):
            return "__present__"
        alias_val = _extract_value_by_aliases(text, aliases)
        if alias_val and "sk-" in alias_val.lower():
            return "__present__"
        if _has_param_secret_source(param):
            return "__present__"
        return previous
    alias_val = _extract_value_by_aliases(text, aliases)
    if alias_val:
        return alias_val
    if ptype == "text":
        t = text.strip()
        if t and len(t) >= 6 and not t.startswith("/use ") and "技能" not in t:
            if any(x in t for x in ("api key", "apikey", "尺寸", "比例")):
                return previous
            return t
    return previous


def _param_satisfied(param: SkillParamItem, value: object) -> bool:
    ptype = param.type.strip().lower()
    if ptype == "images":
        count = int(value) if isinstance(value, int) else 0
        return count >= max(1, int(param.min_count))
    if value is None:
        return False
    return bool(str(value).strip())


def _format_param_status(param: SkillParamItem, value: object) -> str:
    label = param.label or param.key
    ptype = param.type.strip().lower()
    if ptype == "images":
        count = int(value) if isinstance(value, int) else 0
        need = max(1, int(param.min_count))
        return f"{label}：已收到 {count} 张（至少 {need} 张）"
    if _param_satisfied(param, value):
        if ptype == "api_key":
            return f"{label}：已就绪"
        return f"{label}：已收到"
    return f"{label}：未提供"


def _build_skill_param_guard_reply(
    skill_id: str,
    params: list[SkillParamItem],
    state: dict[str, object],
) -> str:
    lines = [f"我将使用 {skill_id} 技能继续完成任务。", "", "我先确认参数（缺啥补啥）："]
    missing_prompts: list[str] = []
    for idx, p in enumerate(params, start=1):
        value = state.get(p.key)
        lines.append(f"{idx}) {_format_param_status(p, value)}")
        if p.required and not _param_satisfied(p, value):
            missing_prompts.append(p.prompt.strip() or (p.label or p.key))
    lines.append("")
    if missing_prompts:
        lines.append("请补充：" + "；".join(missing_prompts) + "。")
    else:
        lines.append("参数已齐，我现在开始执行。")
    return "\n".join(lines)


def _update_guard_state(
    params: list[SkillParamItem],
    state: dict[str, object],
    message: str,
    images: list[ImageContent] | None,
) -> tuple[dict[str, object], bool]:
    new_state = dict(state)
    missing_required = False
    for p in params:
        prev = new_state.get(p.key)
        captured = _capture_param_value(p, message, images, prev)
        new_state[p.key] = captured
        if p.required and not _param_satisfied(p, captured):
            missing_required = True
    return new_state, missing_required


def _guarded_skills(skills: list[Skill]) -> list[Skill]:
    out: list[Skill] = []
    for s in skills:
        if s.param_guard is None or not s.param_guard.enabled or not s.param_guard.params:
            continue
        out.append(s)
    return out


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

    if ("图片" in query or "photo" in query or "image" in query) and tool.name in {
        "browser",
        "desktop_capture",
    }:
        score += 2

    if ("evomap" in query or "evo map" in query) and tool.name.startswith("evomap_"):
        score += 4

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


def _is_evomap_enabled(config: WhaleclawConfig) -> bool:
    plugins_cfg = getattr(config, "plugins", None)
    if not isinstance(plugins_cfg, dict):
        return False
    evomap_cfg_raw: object = plugins_cfg.get("evomap", None)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(evomap_cfg_raw, dict):
        return False
    evomap_cfg: dict[str, object] = evomap_cfg_raw  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
    return bool(evomap_cfg.get("enabled", False))


def _is_evomap_status_question(text: str) -> bool:
    q = text.lower().strip()
    if not q:
        return False
    if "evomap" not in q and "evo map" not in q:
        return False
    status_hints = (
        "开着",
        "开启",
        "启用",
        "打开",
        "关闭",
        "状态",
        "on",
        "off",
        "enabled",
    )
    return any(h in q for h in status_hints)


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
    user_prompt = f"目标上限约 {max_tokens} tokens。\n输入如下：\n{text}\n\n请输出压缩结果。"
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
        "ppt",
        "幻灯片",
        "演示文稿",
        "文档",
        "报告",
        "方案",
        "写",
        "生成",
        "制作",
        "整理",
        "设计",
        "润色",
        "总结",
        "改写",
        "脚本",
        "代码",
        "html",
        "页面",
        "海报",
        "计划",
        "create",
        "generate",
        "draft",
        "design",
        "write",
        "build",
        "compose",
    )
    return any(k in low for k in keys)


async def _llm_judge_task_phase(
    router: "ModelRouter",
    model_id: str,
    *,
    session: Session | None,
    message: str,
) -> str:
    """Use the main model to classify task phase: NEW_TASK or EDITING."""
    system = Message(
        role="system",
        content=(
            "你是任务阶段分类器，只输出一个标签：NEW_TASK 或 EDITING。\n"
            "NEW_TASK=开始一个全新主要任务/新主题/新产物。\n"
            "EDITING=在已有任务上修改/补充/继续/讨论细节。\n"
            "只输出标签，不要解释。"
        ),
    )
    context: list[Message] = [system]
    if session is not None and session.messages:
        recent: list[Message] = []
        for msg in session.messages[-6:]:
            if msg.role not in {"user", "assistant"}:
                continue
            recent.append(
                Message(role=msg.role, content=_preview_text(msg.content or "", limit=400))
            )
        context.extend(recent)
    context.append(
        Message(role="user", content=f"当前用户消息：{_preview_text(message, limit=600)}")
    )
    try:
        resp = await router.chat(model_id, context, tools=None, on_stream=None)
    except Exception:
        return "EDITING"
    raw = (resp.content or "").strip().upper()
    if "NEW_TASK" in raw:
        return "NEW_TASK"
    if "EDITING" in raw:
        return "EDITING"
    return "EDITING"


def _is_tasky_message_for_evomap(text: str) -> bool:
    low = _normalize_for_match(text)
    if not low:
        return False
    keys = (
        "做",
        "制作",
        "生成",
        "写",
        "整理",
        "设计",
        "计划",
        "PPT",
        "ppt",
        "幻灯片",
        "演示文稿",
        "报告",
        "文档",
        "方案",
        "简历",
        "海报",
        "脚本",
        "代码",
        "页面",
        "表格",
        "excel",
        "xlsx",
        "word",
        "docx",
        "evomap",
        "evo map",
        "方案库",
        "协作经验",
    )
    return any(k in low for k in keys)


def _infer_task_kind(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("ppt", "幻灯片", "演示文稿", "slides", "deck")):
        return "ppt"
    if any(k in low for k in ("网页", "网站", "html", "web", "landing page", "前端")):
        return "web"
    if any(k in low for k in ("检索", "汇总", "调研", "信息", "collect", "research", "summarize")):
        return "research"
    return "general"


def _extract_topic_terms(text: str, *, limit: int = 2) -> list[str]:
    low = _normalize_for_match(text)
    stop = {
        "做",
        "制作",
        "生成",
        "创建",
        "一个",
        "关于",
        "给我",
        "帮我",
        "ppt",
        "网页",
        "网站",
        "html",
        "web",
        "方案",
        "检索",
        "汇总",
        "信息",
        "today",
        "todays",
        "today's",
    }
    terms: list[str] = []
    for t in re.findall(r"[\w\u4e00-\u9fff]{2,8}", low):
        if t in stop:
            continue
        if t not in terms:
            terms.append(t)
        if len(terms) >= max(1, limit):
            break
    return terms


def _recommended_evomap_signals(text: str) -> str:  # pyright: ignore[reportUnusedFunction]
    kind = _infer_task_kind(text)
    if kind == "ppt":
        base = [
            "ppt",
            "presentation",
            "slides",
            "storyline",
            "deck structure",
            "visual layout",
            "python-pptx",
        ]
    elif kind == "web":
        base = [
            "web page",
            "html",
            "css",
            "frontend",
            "responsive layout",
            "content structure",
        ]
    elif kind == "research":
        base = [
            "information retrieval",
            "multi-source collection",
            "source validation",
            "structured summary",
            "fact-check",
        ]
    else:
        base = [
            "workflow",
            "execution plan",
            "quality checklist",
        ]
    return ",".join(base + _extract_topic_terms(text, limit=2))


def _extra_memory_has_evomap_hint(extra_memory: str) -> bool:
    text = extra_memory.strip()
    if not text:
        return False
    return "EvoMap 协作经验候选" in text


def _is_no_match_evomap_output(result: ToolResult) -> bool:
    if not result.success:
        return False
    out = (result.output or "").strip()
    if not out:
        return True
    hints = ("未找到匹配方案", "暂无可用任务", "无已认领任务")
    return any(h in out for h in hints)


def _build_evomap_first_system_message() -> Message:
    return Message(
        role="system",
        content=(
            "执行策略：本轮是流程任务，优先复用 EvoMap 成功经验。\n"
            "1) 必须先调用 evomap_fetch 获取经验候选；\n"
            "2) 只有当 evomap_fetch 无命中或失败时，才可调用 browser；\n"
            "3) 若 evomap_fetch 命中，请先按命中方案执行。"
        ),
    )


def _extract_evomap_choice_index(text: str, options_count: int) -> int | None:
    if options_count <= 0:
        return None
    raw = text.strip()
    if not raw:
        return None
    for p in _EVOMAP_CHOICE_PATTERNS:
        m = p.match(raw)
        if not m:
            continue
        token = m.group(1).strip().upper()
        if token in {"A", "B", "C"}:
            idx = ord(token) - ord("A")
        elif token.isdigit():
            idx = int(token) - 1
        else:
            return None
        if 0 <= idx < options_count:
            return idx
    return None


def _parse_evomap_fetch_candidates(output: str) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    items: list[tuple[str, str]] = []
    for ln in lines:
        m = _EVOMAP_LINE_RE.match(ln)
        if not m:
            continue
        aid = m.group(1).strip()
        summary = m.group(2).strip()
        if not aid and not summary:
            continue
        items.append((aid, summary))
    return items


def _pick_top_evomap_candidates(
    user_message: str,
    candidates: list[tuple[str, str]],
    *,
    limit: int = 3,
) -> list[tuple[str, str]]:
    query = _normalize_for_match(user_message)
    terms = {t for t in re.findall(r"[\w\u4e00-\u9fff]{2,}", query)}

    scored: list[tuple[int, tuple[str, str]]] = []
    for item in candidates:
        aid, summary = item
        hay = _normalize_for_match(f"{aid} {summary}")
        score = 0
        for t in terms:
            if t in hay:
                score += 1
        scored.append((score, item))

    scored.sort(key=lambda x: (-x[0], x[1][0]))
    return [item for _score, item in scored[: max(1, limit)]]


def _build_evomap_choice_prompt(candidates: list[tuple[str, str]]) -> str:
    labels = ("A", "B", "C")
    lines = ["EvoMap 命中了多条可用方案，请先选一个我再执行："]
    for idx, item in enumerate(candidates[:3]):
        aid, summary = item
        label = labels[idx]
        lines.append(f"{label}. {aid} — {summary}")
    lines.append("请直接回复：选A / 选B / 选C")
    return "\n".join(lines)


def _is_low_value_bash_probe(tc: ToolCall) -> bool:
    if tc.name != "bash":
        return False
    raw = str(tc.arguments.get("command", "")).strip().lower()
    if not raw:
        return False
    probe_hints = ("ls ", "ls\t", "stat ", "test -f ", "test -e ", "echo ")
    risky_hints = (
        "python ",
        "python3 ",
        "cp ",
        "mv ",
        "rm ",
        "sed ",
        "awk ",
        "perl ",
        "open ",
        "soffice ",
    )
    if any(h in raw for h in risky_hints):
        return False
    return any(h in raw for h in probe_hints)


def _is_office_path_probe_command(command: str) -> bool:
    low = command.strip().lower()
    if not low:
        return False
    probe_tools = (
        "find ",
        "mdfind ",
        "locate ",
        "fd ",
        "ls ",
        "stat ",
        "test -f ",
        "test -e ",
    )
    office_terms = (
        "ppt",
        "pptx",
        "docx",
        "xlsx",
        "word",
        "excel",
        "幻灯片",
        "文档",
        "表格",
        ".pptx",
        ".docx",
        ".xlsx",
    )
    return any(t in low for t in probe_tools) and any(t in low for t in office_terms)


def _looks_like_ppt_generation_script(text: str) -> bool:
    low = text.lower()
    if not low:
        return False
    hints = (
        "from pptx import presentation",
        "pptx import presentation",
        "presentation()",
        ".save(",
        ".pptx",
    )
    return any(h in low for h in hints)


def _looks_like_ppt_generation_command(command: str) -> bool:
    low = command.lower()
    if not low:
        return False
    if "python" not in low and "python3" not in low:
        return False
    return ".pptx" in low or "pptx" in low or "presentation(" in low


def _normalize_bash_command_signature(command: str) -> str:
    """Normalize bash command text for repeated-failure detection."""
    return re.sub(r"\s+", " ", command.strip())


def _capture_latest_pptx(
    metadata: dict[str, object],
    *,
    roots: tuple[Path, ...],
    window_seconds: int = 180,
) -> bool:
    now = time.time()
    candidates: list[Path] = []
    for root in roots:
        try:
            if not root.exists():
                continue
        except Exception:
            continue
        try:
            for p in root.rglob("*.pptx"):
                try:
                    mtime = p.stat().st_mtime
                except Exception:
                    continue
                if now - mtime <= window_seconds:
                    candidates.append(p)
        except Exception:
            continue
    if not candidates:
        return False
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    newest = candidates[0]
    return _remember_office_path(metadata, str(newest))


def _extract_office_paths(text: str) -> list[str]:
    if not text:
        return []
    paths: list[str] = []
    for m in _OFFICE_PATH_RE.finditer(text):
        p = m.group(1).strip()
        if p:
            paths.append(p)
    return paths


def _extract_round_delivery_section(text: str) -> str:
    if not text:
        return ""
    marker = re.search(
        r"(?m)^\s*[1-5][\)\.、:：]\s*本轮可直接交付结果",
        text,
    )
    if marker is None:
        marker = re.search(
            r"(?m)^\s*[1-5][\)\.、:：]\s*本轮可直接使用",
            text,
        )
    if marker is None:
        return text
    tail = text[marker.start() :]
    next_idx = len(tail)
    next_marker = re.search(r"(?m)^\s*[2-9][\)\.、:：]\s*(?!本轮可直接)", tail)
    if next_marker is not None and next_marker.start() > 0:
        next_idx = next_marker.start()
    return tail[:next_idx].strip()


def _extract_delivery_artifact_paths(
    text: str,
    *,
    include_scripts: bool = False,
) -> list[str]:
    section = _extract_round_delivery_section(text)
    if not section:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _ABS_FILE_PATH_RE.finditer(section):
        p = m.group(1).strip()
        if not p:
            continue
        suffix = Path(p).suffix.lower()
        if (not include_scripts) and suffix in _NON_DELIVERY_EXTS:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _with_round_version_suffix(path: str, round_no: int) -> str:
    raw = str(Path(path).expanduser())
    p = Path(raw)
    stem = p.stem
    base_stem = _VERSION_SUFFIX_RE.sub("", stem)
    target_name = f"{base_stem}_V{round_no}{p.suffix}"
    return str(p.with_name(target_name))


def _fix_version_suffix(paths: list[str], round_no: int) -> list[str]:
    """Rename files whose _V suffix doesn't match the current round_no.

    LLMs sometimes use the wrong version number (e.g. _V4 in round 3).
    This renames the file on disk so later rounds don't collide.
    """
    fixed: list[str] = []
    expected = f"_V{round_no}"
    for p in paths:
        fp = Path(p)
        stem = fp.stem
        m = _VERSION_SUFFIX_RE.search(stem)
        if m is not None and m.group(0).upper() != expected.upper():
            correct = _with_round_version_suffix(p, round_no)
            correct_path = Path(correct)
            try:
                if correct_path.exists():
                    correct_path.unlink()
                fp.rename(correct_path)
                fixed.append(str(correct_path))
                continue
            except OSError:
                pass
        fixed.append(p)
    return fixed


def _extract_artifact_baseline(paths: list[str]) -> str:
    """Extract a structured quality baseline from delivered files.

    Returns a compact string like:
      "上轮基线 | 4页 | 每页图片数: P1=1,P2=1,P3=2,P4=1(共5张) | 每页文字数: P1=45,P2=120,P3=98,P4=80"
    """
    for p in paths:
        fp = Path(p).expanduser()
        if not fp.exists():
            continue
        suffix = fp.suffix.lower()
        if suffix == ".pptx":
            return _baseline_from_pptx(fp)
    return ""


def _baseline_from_pptx(fp: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        return ""
    try:
        prs = Presentation(str(fp))
    except Exception:
        return ""
    slide_count = len(prs.slides)
    total_images = 0
    total_chars = 0
    for slide in prs.slides:
        total_images += sum(1 for s in slide.shapes if getattr(s, "shape_type", None) == 13)
        for s in slide.shapes:
            tf = getattr(s, "text_frame", None)
            if tf is not None:
                total_chars += len(str(getattr(tf, "text", "")).strip())
    return f"上轮基线 | {slide_count}页 | {total_images}张图 | {total_chars}字"


def _snapshot_round_artifacts(paths: list[str], round_no: int) -> list[str]:
    snapshots: list[str] = []
    for src in paths:
        src_path = Path(src).expanduser()
        if not src_path.exists() or not src_path.is_file():
            continue
        target = Path(_with_round_version_suffix(str(src_path), round_no)).expanduser()
        try:
            if target.resolve() != src_path.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, target)
            snapshots.append(str(target.resolve()))
        except Exception:
            # Fallback to source path when snapshot copy fails.
            try:
                snapshots.append(str(src_path.resolve()))
            except Exception:
                snapshots.append(str(src_path))
    return snapshots


def _remember_office_path(metadata: dict[str, object], path: str) -> bool:
    p = path.strip()
    if not p:
        return False
    suffix = Path(p).suffix.lower()
    key_map = {
        ".pptx": "last_pptx_path",
        ".docx": "last_docx_path",
        ".xlsx": "last_xlsx_path",
    }
    key = key_map.get(suffix)
    if key is None:
        return False
    if metadata.get(key) == p:
        return False
    metadata[key] = p
    return True


def _is_office_edit_request(text: str) -> bool:
    q = text.lower()
    if not q:
        return False
    edit_hints = ("改", "修改", "调整", "换", "替换", "更换", "删", "删除", "加", "添加", "插入", "第一页", "第二页", "第三页", "单元格", "段落", "封面")
    doc_hints = ("ppt", "pptx", "word", "docx", "excel", "xlsx", "幻灯片", "文档", "表格", "封面", "页")
    return any(h in q for h in edit_hints) and any(h in q for h in doc_hints)


def _is_image_generation_request(text: str) -> bool:
    q = text.lower()
    if not q:
        return False
    generation_hints = (
        "生图",
        "出图",
        "文生图",
        "图生图",
        "图像生成",
        "以图生图",
        "根据这张图生成",
        "image generation",
        "generate image",
        "text-to-image",
        "image-to-image",
        "txt2img",
        "img2img",
    )
    return any(h in q for h in generation_hints)


def _build_image_generation_system_message() -> Message:
    return Message(
        role="system",
        content=(
            "检测到这是生图任务（文生图/图生图）。\n"
            "执行约束：\n"
            "1) 允许最多 2 次轻量探测（如检查环境变量/关键配置）；\n"
            "2) 之后必须立即写最小脚本到 /tmp 并执行一次真实请求；\n"
            "3) 禁止连续使用 ls/stat/test/echo 循环探测；\n"
            "4) 输出必须包含请求命令、HTTP 状态码、返回摘要与图片绝对路径（或明确失败原因）。"
        ),
    )


def _is_complex_office_request(text: str) -> bool:
    q = text.lower()
    if not q:
        return False
    complex_hints = (
        "插图",
        "配图",
        "图片",
        "海报",
        "背景图",
        "图标",
        "视频",
        "音频",
        "音乐",
        "动效",
        "动画",
        "高端",
        "商务风",
        "版式",
        "排版",
        "重排",
        "模板",
        "封面设计",
        "视觉风格",
    )
    return any(h in q for h in complex_hints)


def _mentions_specific_dark_bar_target(text: str) -> bool:
    q = text.lower()
    if not q:
        return False
    target_hints = ("黑色横条", "黑条", "黑色条", "深色横条", "黑色块", "黑底条")
    return any(h in q for h in target_hints)


def _has_any_last_office_path(metadata: dict[str, object]) -> bool:
    for key in ("last_pptx_path", "last_docx_path", "last_xlsx_path"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _is_followup_edit_message(text: str) -> bool:
    q = text.lower()
    if not q:
        return False
    hints = ("改一下", "修改", "调整", "换一", "替换", "更换", "删掉", "删除", "加一", "添加", "插入", "第一页", "第一面", "第二页", "第三页", "单元格", "段落", "封面图", "封面")
    return any(h in q for h in hints)


def _get_default_office_edit_path(
    tool_name: str,
    metadata: dict[str, object],
) -> str | None:
    key_map = {
        "ppt_edit": "last_pptx_path",
        "docx_edit": "last_docx_path",
        "xlsx_edit": "last_xlsx_path",
    }
    key = key_map.get(tool_name)
    if key is None:
        return None
    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    path = value.strip()
    return path or None


def _build_office_edit_hint_system_message(metadata: dict[str, object]) -> Message | None:
    hints: list[str] = []
    for label, key in (
        ("PPT", "last_pptx_path"),
        ("Word", "last_docx_path"),
        ("Excel", "last_xlsx_path"),
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            hints.append(f"- {label}: {value.strip()}")
    if not hints:
        return None
    return Message(
        role="system",
        content=(
            "检测到用户在要求修改已有 Office 文件。\n"
            "若只是文案小改，请优先使用对应编辑工具（ppt_edit/docx_edit/xlsx_edit）；"
            "若涉及插图/音视频/风格升级/复杂排版，"
            "请先输出简短执行计划，"
            "再组合 browser、bash、file_write 与编辑工具执行。\n"
            "图片操作规则：\n"
            "- 换图/替换图片 → action=replace_image（删旧图+在原位置放新图）\n"
            "- 新增图片 → action=add_image\n"
            "- 删除图片 → action=remove_image\n"
            "严禁用 add_image 来「换图」，那样旧图还在，新图只是盖在上面。\n"
            "必须优先修改用户明确点名的对象（页码/元素/文案），"
            "不要改成泛化动作（例如只改整页背景）。\n"
            "不要把复杂请求机械降级为“只能改文字”。\n"
            "并优先使用以下最近文件路径，不要先用 bash 反复探测：\n" + "\n".join(hints)
        ),
    )


def _build_office_path_block_message(metadata: dict[str, object]) -> str:
    hints: list[str] = []
    for label, key in (
        ("PPT", "last_pptx_path"),
        ("Word", "last_docx_path"),
        ("Excel", "last_xlsx_path"),
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            hints.append(f"{label}: {value.strip()}")
    if not hints:
        return "检测到这是 Office 修改请求，请直接使用已有文件路径，不要用 bash 查找。"
    return (
        "检测到已有 Office 文件路径，禁止用 bash 探测/查找。\n"
        "请直接使用以下路径：\n" + "\n".join(f"- {h}" for h in hints)
    )


def _build_complex_office_plan_system_message() -> Message:
    return Message(
        role="system",
        content=(
            "检测到这是复杂 Office 修改任务（可能包含插图/音视频/风格升级/版式调整）。\n"
            "执行要求：\n"
            "1) 先给用户一句话计划（将做哪些步骤）；\n"
            "2) 再按步骤调用工具，不要只调用一次 ppt_edit 就结束；\n"
            "3) 如需媒体素材，优先先获取素材文件路径，再写入文档。"
        ),
    )


def _append_office_system_hints(
    system_messages: list[Message],
    session: Session | None,
    llm_message: str,
) -> None:
    if session is None:
        return
    is_office_request = _is_office_edit_request(llm_message) or (
        _is_followup_edit_message(llm_message) and _has_any_last_office_path(session.metadata)
    )
    if not is_office_request:
        return
    office_hint = _build_office_edit_hint_system_message(session.metadata)
    if office_hint is not None:
        system_messages.append(office_hint)
    if _is_complex_office_request(llm_message):
        system_messages.append(_build_complex_office_plan_system_message())


def _force_include_office_edit_tools(
    selected: set[str],
    *,
    available: set[str],
    session: Session | None,
    llm_message: str,
) -> set[str]:
    """Keep office edit tools available for follow-up edits with known file paths."""
    if session is None:
        return selected
    if not (
        _is_office_edit_request(llm_message)
        or (_is_followup_edit_message(llm_message) and _has_any_last_office_path(session.metadata))
    ):
        return selected

    key_by_tool = {
        "ppt_edit": "last_pptx_path",
        "docx_edit": "last_docx_path",
        "xlsx_edit": "last_xlsx_path",
    }
    expanded = set(selected)
    for tool_name, meta_key in key_by_tool.items():
        if tool_name not in available:
            continue
        path_value = session.metadata.get(meta_key)
        if isinstance(path_value, str) and path_value.strip():
            expanded.add(tool_name)
    return expanded


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
    from whaleclaw.tools.docx_edit import DocxEditTool
    from whaleclaw.tools.file_edit import FileEditTool
    from whaleclaw.tools.file_read import FileReadTool
    from whaleclaw.tools.file_write import FileWriteTool
    from whaleclaw.tools.ppt_edit import PptEditTool
    from whaleclaw.tools.xlsx_edit import XlsxEditTool

    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(PptEditTool())
    registry.register(DocxEditTool())
    registry.register(XlsxEditTool())
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
            calls.append(
                ToolCall(
                    id=f"fallback_{len(calls)}",
                    name=raw_name,
                    arguments=raw_args,  # pyright: ignore[reportUnknownArgumentType]
                )
            )

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
            session,
            role,
            content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
    except Exception as exc:
        log.debug("agent.persist_failed", error=str(exc))


async def _execute_tool(
    registry: ToolRegistry,
    tc: ToolCall,
    *,
    evomap_enabled: bool,
    browser_allowed: bool,
    office_block_bash_probe: bool,
    office_block_message: str,
    office_edit_only: bool,
    office_edit_path: str,
    on_tool_call: OnToolCall | None,
    on_tool_result: OnToolResult | None,
) -> tuple[str, ToolResult]:
    """Execute a single tool call and return (tool_call_id, result)."""
    if on_tool_call:
        await on_tool_call(tc.name, tc.arguments)

    import time

    t0 = time.monotonic()

    if tc.name == "browser" and not browser_allowed:
        result = ToolResult(
            success=False,
            output="",
            error="请先执行 evomap_fetch；若无命中会自动切换到 browser",
        )
    elif tc.name == "file_write" and office_edit_only:
        content = str(tc.arguments.get("content", ""))
        if _looks_like_ppt_generation_script(content):
            result = ToolResult(
                success=False,
                output="",
                error=(
                    "检测到这是修改已有PPT的请求，禁止重新生成新PPT。\n"
                    f"请直接使用 ppt_edit 修改：{office_edit_path}"
                ),
            )
        else:
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
    elif tc.name == "bash" and office_block_bash_probe:
        raw_command = str(tc.arguments.get("command", ""))
        if _is_office_path_probe_command(raw_command):
            result = ToolResult(
                success=False,
                output="",
                error=office_block_message,
            )
        elif office_edit_only and _looks_like_ppt_generation_command(raw_command):
            result = ToolResult(
                success=False,
                output="",
                error=(
                    "检测到这是修改已有PPT的请求，禁止重新生成新PPT。\n"
                    f"请直接使用 ppt_edit 修改：{office_edit_path}"
                ),
            )
        else:
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
            if not result.success and tool is not None:
                missing_target = _can_auto_create_parent_for_failure(result)
                if missing_target:
                    parent = str(Path(missing_target).expanduser().resolve().parent)
                    try:
                        _mkdir_res = await tool.execute(
                            command=f"mkdir -p '{parent}'",
                            timeout=30,
                        )
                        if _mkdir_res.success:
                            result = await tool.execute(**tc.arguments)
                    except Exception:
                        pass
    elif tc.name.startswith("evomap_") and not evomap_enabled:
        result = ToolResult(
            success=False,
            output="",
            error="EvoMap 已关闭，请先在设置中开启",
        )
    else:
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
        if tc.name == "bash" and not result.success and tool is not None:
            missing_target = _can_auto_create_parent_for_failure(result)
            if missing_target:
                parent = str(Path(missing_target).expanduser().resolve().parent)
                try:
                    _mkdir_res = await tool.execute(
                        command=f"mkdir -p '{parent}'",
                        timeout=30,
                    )
                    if _mkdir_res.success:
                        result = await tool.execute(**tc.arguments)
                except Exception:
                    pass

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
    out = f"[ERROR] {result.error or 'unknown error'}\n{result.output}".strip()
    diagnosis = _diagnose_failure_hint(result)
    if diagnosis:
        out += f"\n[DIAGNOSIS] {diagnosis}"
    return out


def _diagnose_failure_hint(result: ToolResult) -> str:
    text = f"{result.error or ''}\n{result.output or ''}"
    if "No such file or directory" in text and "FileNotFoundError" in text:
        return "更可能是目标路径或上级目录不存在，请先创建目录再写文件，不是依赖缺失。"
    if "ModuleNotFoundError" in text:
        return "这是依赖缺失，请安装缺失模块后重试。"
    if "Permission denied" in text:
        return "这是权限问题，请检查目标路径可写权限。"
    return ""


def _extract_missing_target_path(text: str) -> str:
    # Example: FileNotFoundError: [Errno 2] No such file or directory: '/Users/a/output/x.pptx'
    m = re.search(
        r"No such file or directory:\s*['\"](/[^'\"]+)['\"]",
        text,
    )
    if not m:
        return ""
    path = m.group(1).strip()
    if not path:
        return ""
    return path


def _can_auto_create_parent_for_failure(result: ToolResult) -> str:
    text = f"{result.error or ''}\n{result.output or ''}"
    if "FileNotFoundError" not in text or "No such file or directory" not in text:
        return ""
    target = _extract_missing_target_path(text)
    if not target:
        return ""
    suffix = Path(target).suffix.lower()
    if suffix not in {".pptx", ".docx", ".xlsx", ".pdf", ".html", ".md", ".txt", ".py"}:
        return ""
    return target


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
        "图",
        "图片",
        "照片",
        "近照",
        "头像",
        "搜图",
        "找图",
        "壁纸",
        "海报",
        "背景",
        "写真",
        "image",
        "photo",
        "picture",
        "wallpaper",
        "poster",
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


def _multi_agent_cfg(config: WhaleclawConfig) -> dict[str, object]:
    plugins_raw = config.plugins
    plugins: dict[str, object] = plugins_raw if isinstance(plugins_raw, dict) else {}  # pyright: ignore[reportAssignmentType, reportUnnecessaryIsInstance]
    raw_ma = plugins.get("multi_agent", {})
    if not isinstance(raw_ma, dict):
        return {"enabled": False, "mode": "parallel", "max_rounds": 1, "roles": []}

    raw: dict[str, object] = raw_ma  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
    enabled = bool(raw.get("enabled", False))
    mode_raw = str(raw.get("mode", "parallel")).strip().lower()
    mode = mode_raw if mode_raw in {"parallel", "serial"} else "parallel"

    try:
        max_rounds = int(raw.get("max_rounds", 1))  # pyright: ignore[reportArgumentType]
    except Exception:
        max_rounds = 1
    max_rounds = max(1, min(max_rounds, 10))

    roles_raw = raw.get("roles")
    roles: list[dict[str, object]] = []
    if isinstance(roles_raw, list):
        for idx, item in enumerate(roles_raw[:20], start=1):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            if not isinstance(item, dict):
                continue
            role_item: dict[str, object] = item  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
            rid = str(role_item.get("id", f"role_{idx}")).strip().lower()
            rid = "".join(ch for ch in rid if ch.isalnum() or ch in {"_", "-"})
            if not rid:
                rid = f"role_{idx}"
            name = str(role_item.get("name", f"角色{idx}")).strip() or f"角色{idx}"
            model = str(role_item.get("model", "")).strip()
            system_prompt = str(role_item.get("system_prompt", "")).strip()
            roles.append(
                {
                    "id": rid[:64],
                    "name": name[:50],
                    "enabled": bool(role_item.get("enabled", True)),
                    "model": model[:100],
                    "system_prompt": system_prompt[:3000],
                }
            )
    return {
        "enabled": enabled,
        "mode": mode,
        "max_rounds": max_rounds,
        "roles": roles,
}


def _scenario_discuss_focus(scenario: str) -> str:
    if scenario == "product_design":
        return (
            "重点和用户确认：产品目标、目标用户、核心场景、关键流程、约束条件、"
            "以及交付物类型（如 PRD 文档、流程图图片、原型说明、里程碑计划）。"
        )
    if scenario == "content_creation":
        return (
            "重点和用户确认：受众人群、内容主题、语气风格、发布渠道、篇幅限制、"
            "素材来源与交付物类型（文章/脚本/海报文案/配图说明）。"
        )
    if scenario == "software_development":
        return (
            "重点和用户确认：功能目标、技术栈、运行环境、改动范围、验收标准、"
            "交付物类型（代码补丁、命令步骤、测试报告、部署说明）。"
        )
    if scenario == "data_analysis_decision":
        return (
            "重点和用户确认：决策问题、指标口径、数据来源、时间窗口、可信度要求、"
            "交付物类型（分析报告、图表、结论摘要、决策建议表）。"
        )
    if scenario == "scientific_research":
        return (
            "重点和用户确认：研究问题、假设、实验条件、对照设计、评估指标、"
            "交付物类型（研究提纲、实验方案、结果解读、论文结构草稿）。"
        )
    if scenario == "intelligent_assistant":
        return (
            "重点和用户确认：任务目标、时效要求、可调用工具、执行边界、"
            "交付物类型（行动计划、提醒清单、消息草稿、执行结果汇总）。"
        )
    if scenario == "workflow_automation":
        return (
            "重点和用户确认：触发条件、上下游系统、字段映射、失败重试、监控告警、"
            "交付物类型（流程设计文档、自动化脚本、运行手册、告警规则清单）。"
        )
    return "重点和用户确认目标、约束、验收标准与最终交付物类型。"


def _scenario_delivery_focus(scenario: str) -> str:  # pyright: ignore[reportUnusedFunction]
    if scenario == "product_design":
        return (
            "最终答复必须包含：\n"
            "1) 产品方案摘要\n"
            "2) 结构化交付物清单（文档/图片/表格）\n"
            "3) 每个交付物的建议文件名与路径（例如 /tmp/product_prd.md）\n"
            "4) 执行优先级与里程碑\n"
            "5) 风险与备选方案"
        )
    if scenario == "content_creation":
        return (
            "最终答复必须包含：\n"
            "1) 内容策略与目标受众\n"
            "2) 成品文案或脚本草案\n"
            "3) 渠道适配版本（至少 2 个）\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/content_plan.md）\n"
            "5) 发布节奏与复盘指标"
        )
    if scenario == "software_development":
        return (
            "最终答复必须包含：\n"
            "1) 技术方案与实现路径\n"
            "2) 关键代码改动点或命令步骤\n"
            "3) 测试与回归计划\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/impl_plan.md）\n"
            "5) 风险、回滚与上线注意项"
        )
    if scenario == "data_analysis_decision":
        return (
            "最终答复必须包含：\n"
            "1) 数据结论与关键洞察\n"
            "2) 指标口径说明与分析过程摘要\n"
            "3) 决策选项对比与推荐方案\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/analysis_report.md）\n"
            "5) 风险假设与后续验证计划"
        )
    if scenario == "scientific_research":
        return (
            "最终答复必须包含：\n"
            "1) 研究目标与假设\n"
            "2) 方法设计与实验步骤\n"
            "3) 结果解读框架与可信度边界\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/research_plan.md）\n"
            "5) 下一步实验与论文化建议"
        )
    if scenario == "intelligent_assistant":
        return (
            "最终答复必须包含：\n"
            "1) 任务拆解与优先级\n"
            "2) 可执行动作清单\n"
            "3) 关键提醒与时间节点\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/assistant_actions.md）\n"
            "5) 异常处理与后续跟进建议"
        )
    if scenario == "workflow_automation":
        return (
            "最终答复必须包含：\n"
            "1) 自动化流程设计（触发-处理-输出）\n"
            "2) 集成接口与字段映射说明\n"
            "3) 失败重试与告警策略\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/workflow_spec.md）\n"
            "5) 上线运行与运维检查清单"
        )
    return (
        "最终答复必须包含：结论、可执行清单、交付物清单（含建议文件名/路径）、"
        "风险与回滚建议。"
    )


def _resolve_multi_agent_cfg(
    config: WhaleclawConfig,
    session: Session | None,
) -> dict[str, object]:
    """Build effective multi-agent config with optional session overrides."""
    cfg = _multi_agent_cfg(config)
    plugins_raw2 = config.plugins
    plugins2: dict[str, object] = plugins_raw2 if isinstance(plugins_raw2, dict) else {}  # pyright: ignore[reportAssignmentType, reportUnnecessaryIsInstance]
    raw = plugins2.get("multi_agent", {})
    if isinstance(raw, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raw_typed: dict[str, object] = raw  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
        scenario = str(raw_typed.get("scenario", "software_development")).strip()
        cfg["scenario"] = scenario or "software_development"
    else:
        cfg["scenario"] = "software_development"
    if session is None or not isinstance(session.metadata, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return cfg

    metadata = session.metadata
    if isinstance(metadata.get("multi_agent_enabled"), bool):
        cfg["enabled"] = bool(metadata["multi_agent_enabled"])

    mode_raw = str(metadata.get("multi_agent_mode", "")).strip().lower()
    if mode_raw in {"parallel", "serial"}:
        cfg["mode"] = mode_raw

    rounds_raw = metadata.get("multi_agent_max_rounds")
    if isinstance(rounds_raw, int):
        cfg["max_rounds"] = max(1, min(rounds_raw, 10))

    return cfg


def _is_multi_agent_confirm(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in _MULTI_AGENT_CONFIRM_PATTERNS)


def _is_multi_agent_cancel(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in _MULTI_AGENT_CANCEL_PATTERNS)


def _extract_multi_agent_rounds(text: str) -> int | None:
    t = text.strip()
    if not t:
        return None
    for p in _MULTI_AGENT_ROUNDS_PATTERNS:
        m = p.search(t)
        if not m:
            continue
        try:
            value = int(m.group(1))
        except Exception:
            return None
        if 1 <= value <= 10:
            return value
    return None


def _attach_rounds_marker(topic: str, rounds: int) -> str:
    clean = re.sub(r"\[MA_ROUNDS=\d{1,2}\]\s*", "", topic).strip()
    return f"[MA_ROUNDS={rounds}] {clean}".strip()


def _extract_rounds_marker(topic: str) -> tuple[str, int | None]:
    m = re.search(r"\[MA_ROUNDS=(\d{1,2})\]", topic)
    if not m:
        return (topic, None)
    try:
        value = int(m.group(1))
    except Exception:
        value = None
    clean = re.sub(r"\[MA_ROUNDS=\d{1,2}\]\s*", "", topic).strip()
    if value is None or not (1 <= value <= 10):
        return (clean, None)
    return (clean, value)


def _is_multi_agent_discuss_done(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in _MULTI_AGENT_DISCUSS_DONE_PATTERNS)


def _format_multi_agent_preflight_text(  # pyright: ignore[reportUnusedFunction]
    *,
    cfg: dict[str, object],
    topic: str,
) -> str:
    roles = [
        role
        for role in cast(list[dict[str, object]], cfg["roles"])
        if bool(role.get("enabled", True))
    ]
    mode = cast(str, cfg["mode"])
    rounds = cast(int, cfg["max_rounds"])
    mode_cn = "并行" if mode == "parallel" else "串行"
    lines = [
        "已进入多Agent准备阶段（尚未开始执行）。",
        f"- 当前模式: {mode_cn}（{mode}）",
        f"- 计划回合: {rounds}",
        f"- 角色数量: {len(roles)}",
        "",
        "角色分工:",
    ]
    for role in roles:
        name = str(role.get("name", role.get("id", "角色"))).strip() or "角色"
        duty = _multi_agent_system_prompt(role)
        lines.append(f"- {name}: {_compact_role_output(duty, 120)}")
    lines.extend(
        [
            "",
            "主控建议:",
            "- 先确认目标、交付形式、截止时间与约束。",
            "- 若希望更快出结论，可先改为 2 轮；若任务复杂建议 4 轮以上。",
            "",
            f"当前议题: {topic.strip() or '(未提供)'}",
            "",
            "回复以下任一指令继续:",
            "- 回复“确认开始”：按当前配置启动多Agent执行",
            "- 回复“改为N轮”：修改回合后继续等待确认",
            "- 回复“取消”：退出本次多Agent执行",
        ]
    )
    return "\n".join(lines)


async def _persist_session_metadata(
    session: Session | None,
    session_manager: SessionManager | None,
) -> bool:
    if session is None or session_manager is None:
        return False
    try:
        await session_manager.update_metadata(session, session.metadata)
    except Exception:
        return False
    return True


async def _sync_multi_agent_compression_boundary(
    session: Session | None,
    session_manager: SessionManager | None,
    group_compressor: "SessionGroupCompressor | None" = None,
    *,
    ma_enabled: bool,
) -> None:
    """Track MA on/off transition and set compression boundary on MA -> single.

    When MA is enabled, mark current session as MA-active.
    When MA turns off, record a fixed message-index boundary so future
    group-compression only applies to newly produced messages.
    """
    if session is None or not isinstance(session.metadata, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return None
    metadata = session.metadata
    prev_active = bool(metadata.get("multi_agent_active_prev", False))
    changed = False
    if ma_enabled:
        if not prev_active:
            metadata["multi_agent_active_prev"] = True
            changed = True
    else:
        if prev_active:
            metadata["multi_agent_active_prev"] = False
            metadata["compression_resume_message_index"] = len(session.messages)
            changed = True
    if group_compressor is not None:  # session already checked above
        try:
            await group_compressor.set_session_suspended(
                session_id=session.id,
                suspended=ma_enabled,
            )
        except Exception as exc:
            log.debug(
                "agent.multi_agent_compressor_toggle_failed",
                session_id=session.id,
                error=str(exc),
                enabled=ma_enabled,
            )
    if changed:
        await _persist_session_metadata(session, session_manager)
    if session_manager is None:
        return None
    try:
        await asyncio.wait_for(
            session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                session.id,
                metadata=session.metadata,
            ),
            timeout=1.5,
        )
        return None
    except Exception as exc:
        log.warning(
            "agent.multi_agent_metadata_persist_failed",
            session_id=session.id,
            error=str(exc),
        )
        return None


async def _run_multi_agent_controller_discussion(
    *,
    user_message: str,
    pending_topic: str,
    cfg: dict[str, object],
    session_id: str,
    config: WhaleclawConfig,
    router: ModelRouter,
    registry: ToolRegistry,
    extra_memory: str,
    trigger_event_id: str,
    trigger_text_preview: str,
    include_intro: bool,
) -> str:
    roles = [
        role
        for role in cast(list[dict[str, object]], cfg["roles"])
        if bool(role.get("enabled", True))
    ]
    mode = cast(str, cfg["mode"])
    rounds = cast(int, cfg["max_rounds"])
    scenario = str(cfg.get("scenario", "software_development")).strip()
    scenario_label = _MULTI_AGENT_SCENARIO_LABELS.get(scenario, scenario or "自定义")
    discuss_focus = _scenario_discuss_focus(scenario)
    role_lines: list[str] = []
    role_bullets: list[str] = []
    for role in roles:
        name = str(role.get("name", role.get("id", "角色"))).strip() or "角色"
        duty = _multi_agent_system_prompt(role)
        role_lines.append(f"- {name}: {duty}")
        role_bullets.append(f"- {name}")
    role_block = "\n".join(role_lines) if role_lines else "- （暂无角色）"
    role_intro = "\n".join(role_bullets) if role_bullets else "- （暂无角色）"

    discuss_prompt = (
        "你是“多Agent主控协调者”。当前阶段只做需求澄清，不允许启动多角色执行。\n"
        "请先准确理解用户原始表述的核心意图，再结合场景模板提供专业视角。\n"
        "当用户意图与模板默认方向不一致时，以用户意图为准（如用户要“写真PPT”是做展示，不是做策划方案）。\n"
        "请复述你理解的用户目标（忠实于用户原话的核心意图），"
        "再给 2-3 条有价值建议，最后提出最多 3 个澄清问题。\n"
        "如果信息已经足够执行，也不要直接执行，只需提示用户回复“确认需求”后立即启动多Agent执行。\n"
        "语气自然，不要像程序状态页。\n\n"
        f"场景澄清重点（提供专业框架，与用户意图结合使用）：{discuss_focus}\n\n"
        f"当前配置：场景={scenario_label}，模式={mode}，最大回合={rounds}\n"
        f"角色分工：\n{role_block}\n\n"
        f"累计议题：\n{pending_topic}\n\n"
        f"用户本轮消息：\n{user_message}\n"
    )
    output = await run_agent(
        message=discuss_prompt,
        session_id=f"{session_id}::ma::controller",
        config=config,
        on_stream=None,
        session=None,
        router=router,
        registry=registry,
        on_tool_call=None,
        on_tool_result=None,
        images=None,
        session_manager=None,
        session_store=None,
        memory_manager=None,
        extra_memory=extra_memory,
        trigger_event_id=trigger_event_id,
        trigger_text_preview=trigger_text_preview,
        group_compressor=None,
        multi_agent_internal=True,
    )
    output = output.strip()
    intro = (
        "我是主控Agent协调者。\n"
        f"当前场景：{scenario_label}；协作模式：{'并行' if mode == 'parallel' else '串行'}；"
        f"最大回合：{rounds}。\n"
        f"本轮将使用这 4 个角色协作：\n{role_intro}\n"
    )
    intro_block = f"{intro}\n" if include_intro else ""
    if output:
        return (
            f"{intro_block}{output}\n\n"
            "如果你觉得需求已经讲清楚，请回复“确认需求”；"
            "如果要改回合可回复“改为2轮”；不想继续可回复“取消”。\n"
            "你也可以直接指定交付物：例如“交付 PRD 文档 + 流程图图片 + 里程碑表”。"
        )
    return (
        f"{intro_block}"
        "我先和你确认需求：请补充目标、交付形式、截止时间与约束。"
        "信息齐后回复“确认需求”，我会立即启动多Agent执行。"
    )


def _multi_agent_system_prompt(role: dict[str, object]) -> str:
    text = str(role.get("system_prompt", "")).strip()
    if text:
        return text
    return "负责从本角色视角提出可执行方案、风险与下一步建议。"


def _compact_role_output(text: str, max_chars: int = 600) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def _looks_like_bad_coordinator_output(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    hints = (
        "请把下方各角色多轮输出",
        "把各角色输出贴出来",
        "请粘贴各角色",
        "请提供各角色",
        "请先提供角色输出",
        "你提供的各角色多轮输出",
        "把下面这些内容贴出来",
        "你把材料发来",
        "请把材料发来",
        "我会把你提供的各角色",
    )
    if any(h in t for h in hints):
        return True
    ask_user_hints = (
        "请你补充",
        "请补充",
        "请回复",
        "你回复",
        "请回答",
        "你回答",
        "请确认",
        "需要你确认",
        "请发我",
        "你发我",
        "你把",
        "任选其一",
        "任意选一种",
        "为了我马上开工",
        "我就能继续",
        "我才能继续",
        "你要我交付",
        "你只要按",
        "回一条就行",
        "回一条即可",
        "需要你补",
        "你需要补",
        "你想要什么",
        "你想让我",
        "交付物类型：",
    )
    if any(h in t for h in ask_user_hints):
        return True
    if _COORDINATOR_ASK_RE.search(t):
        return True
    if t.startswith("我将使用 ") and " 技能继续完成任务" in t:
        return True
    return "各角色" in t and ("贴出来" in t or "粘贴" in t or "发来" in t)


def _is_transient_multi_agent_error(exc: Exception) -> bool:
    text = str(exc).lower()
    signals = (
        "error 500",
        "server_error",
        "responses api error 500",
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    )
    return any(s in text for s in signals)


def _looks_like_role_stall_output(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    hints = (
        "我将使用",
        "请你补充",
        "请补充",
        "你可以直接",
        "你现在想先做哪一块",
        "你先发",
        "我将以",
        "为了更高效推进",
        "为了马上进入状态",
        "请把要评审的内容发我",
    )
    if any(h in t for h in hints):
        return True
    return "?" in t or "？" in t


def _role_config(config: WhaleclawConfig, role_model: str) -> WhaleclawConfig:
    if not role_model:
        return config
    copied = config.model_copy(deep=True)
    copied.agent.model = role_model
    return copied


def _need_image_output(user_message: str) -> bool:
    t = user_message.lower()
    negative_hints = (
        "不要图片",
        "不需要图片",
        "无需图片",
        "不要配图",
        "不需要配图",
        "无需配图",
        "纯文字",
        "仅文字",
        "只要文字",
        "text only",
    )
    if any(h in t for h in negative_hints):
        return False
    hints = (
        "配图",
        "图片",
        "图文",
        "海报",
        "封面",
        "插图",
        "image",
        "images",
        "poster",
        "cover",
    )
    return any(h in t for h in hints)


def _extract_requested_deliverables(user_message: str) -> list[str]:
    t = user_message.lower()
    out: list[str] = []
    rules: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ("word", ("word", "docx", ".docx"), ("不要word", "不需要word", "不要docx", "不需要docx")),
        ("ppt", ("ppt", "pptx", ".pptx", "幻灯片", "演示文稿"), ("不要ppt", "不需要ppt")),
        ("excel", ("excel", "xlsx", ".xlsx", "表格"), ("不要excel", "不需要excel", "不要表格")),
        ("html", ("html", "网页", "web页面"), ("不要html", "不需要html", "不要网页")),
        ("pdf", ("pdf", ".pdf"), ("不要pdf", "不需要pdf")),
        (
            "script",
            ("脚本", "python脚本", "py脚本", ".py", ".sh", "shell脚本", "bash脚本"),
            ("不要脚本", "不需要脚本"),
        ),
        (
            "image",
            ("图片", "配图", "图文", "海报", "封面", "image", "images"),
            ("不要图片", "不需要图片", "不要配图", "不需要配图"),
        ),
    ]
    for name, positives, negatives in rules:
        if any(n in t for n in negatives):
            continue
        if any(p in t for p in positives):
            out.append(name)
    return out


def _subset_registry(registry: ToolRegistry, allowed_names: set[str]) -> ToolRegistry:
    sub = ToolRegistry()
    for name in sorted(allowed_names):
        tool = registry.get(name)
        if tool is not None:
            sub.register(tool)
    return sub


def _choose_round_tool_lock(
    *,
    registry: ToolRegistry,
    user_message: str,
    shared_context: str,
    round_no: int,
    requested_deliverables: list[str],
) -> list[str]:
    query = f"第{round_no}轮\n{user_message}\n{shared_context}"
    selected = _select_native_tool_names(registry, query)
    for deliverable in requested_deliverables:
        if deliverable == "word":
            selected.update({"file_write", "docx_edit", "bash"})
        elif deliverable == "ppt":
            selected.update({"file_write", "ppt_edit", "bash"})
        elif deliverable == "excel":
            selected.update({"file_write", "xlsx_edit", "bash"})
        elif deliverable == "html":
            selected.update({"file_write", "file_edit"})
        elif deliverable == "pdf":
            selected.update({"file_write", "bash"})
        elif deliverable == "image":
            selected.update({"browser", "file_write"})
    if not selected:
        selected = {"file_read", "file_write"}
    return sorted(selected)


def _clip_text_for_role_view(text: str, max_chars: int = 200) -> str:
    return " ".join(text.split()).strip() or "（无输出）"


def _build_multi_agent_requirement_baseline(
    *,
    message: str,
    scenario: str,
    mode: str,
    max_rounds: int,
    requested_deliverables: list[str],
) -> str:
    scenario_label = _MULTI_AGENT_SCENARIO_LABELS.get(scenario, scenario or "自定义")
    deliverables = ", ".join(requested_deliverables) if requested_deliverables else "未明确指定"
    text = (
        "【冻结需求基线（每轮必须遵守，不得改写目标）】\n"
        f"- 场景: {scenario_label}\n"
        f"- 协作: {mode}\n"
        f"- 轮次: {max_rounds}\n"
        f"- 交付类型: {deliverables}\n"
        f"- 用户确认需求原文:\n{message}\n"
    )
    return _truncate_to_tokens(text, 700)


async def _run_multi_agent_executor(
    *,
    message: str,
    session_id: str,
    config: WhaleclawConfig,
    on_stream: StreamCallback | None,
    router: ModelRouter,
    registry: ToolRegistry,
    images: list[ImageContent] | None,
    extra_memory: str,
    trigger_event_id: str,
    trigger_text_preview: str,
    ma_cfg: dict[str, object],
    on_round_result: OnRoundResult | None = None,
) -> str:
    roles = [
        role
        for role in cast(list[dict[str, object]], ma_cfg["roles"])
        if bool(role.get("enabled", True))
    ]
    if not roles:
        return "多Agent已启用，但没有可用角色。请在“多Agent”页面至少启用一个角色。"

    mode = cast(str, ma_cfg["mode"])
    max_rounds = cast(int, ma_cfg["max_rounds"])
    scenario = str(ma_cfg.get("scenario", "software_development")).strip()
    requested_deliverables = _extract_requested_deliverables(message)
    _visual_file_types = {"ppt", "word", "html", "pdf"}
    include_image_output = (
        "image" in requested_deliverables
        or bool(_visual_file_types & set(requested_deliverables))
        or _need_image_output(message)
    )
    requirement_baseline = _build_multi_agent_requirement_baseline(
        message=message,
        scenario=scenario,
        mode=mode,
        max_rounds=max_rounds,
        requested_deliverables=requested_deliverables,
    )
    if on_stream is not None:
        await on_stream(
            f"[多Agent] 已启用 {len(roles)} 个角色，模式={mode}，最大回合={max_rounds}。"
        )

    history_blocks: list[str] = []
    round_deliveries: list[str] = []
    shared_context = ""
    prev_round_artifact_baseline: str = ""

    async def _run_one_role(
        role: dict[str, object],
        round_no: int,
        context_block: str,
        round_tools: list[str],
        round_registry: ToolRegistry,
        start_delay: float = 0.0,
    ) -> tuple[str, str, str]:
        if start_delay > 0:
            await asyncio.sleep(start_delay)
        role_id = str(role.get("id", "role")).strip() or "role"
        role_name = str(role.get("name", role_id)).strip() or role_id
        prompt = _multi_agent_system_prompt(role)
        role_msg = (
            f"你当前扮演的角色是「{role_name}」。\n"
            f"角色职责：{prompt}\n"
            f"当前是协作第 {round_no}/{max_rounds} 轮。\n"
            f"用户原始议题：{message}\n"
            "请结合角色专长和场景模板发表观点，但方向须服从用户原始议题的核心意图。\n"
        )
        role_msg += f"\n{requirement_baseline}\n"
        if context_block:
            role_msg += f"\n前序协作要点：\n{context_block}\n"
        if round_no > 1 and round_deliveries:
            prev_summary = _extract_round_delivery_section(round_deliveries[-1]) or round_deliveries[-1]
            prev_summary = _truncate_to_tokens(prev_summary, 400)
            role_msg += (
                f"\n上一轮（第 {round_no - 1} 轮）交付内容摘要（仅供参考，本轮须重新制作，不要读取或修改上一轮文件）：\n"
                + prev_summary
                + "\n"
            )
        role_msg += (
            f"\n本轮工具锁定：{', '.join(round_tools)}\n"
            "\n请只输出本角色观点，必须总结为200字以内的精炼观点（严格不超过200字）。\n"
            "要求：给出明确判断和建议，用连贯段落表达，不要分点罗列，不要列步骤。\n"
            "限制：\n"
            "- 不要向用户追问或索取额外材料。\n"
            "- 信息不足时请自行做“最小必要假设”，并在输出中明确标注“假设”。\n"
            "- 禁止出现“我将使用xx技能继续完成任务”等过程话术。\n"
        )

        role_cfg = _role_config(config, str(role.get("model", "")).strip())
        last_error = ""
        for attempt in range(1, 4):
            try:
                output = await run_agent(
                    message=role_msg,
                    session_id=f"{session_id}::ma::{role_id}::r{round_no}",
                    config=role_cfg,
                    on_stream=None,
                    session=None,
                    router=router,
                    registry=round_registry,
                    on_tool_call=None,
                    on_tool_result=None,
                    images=None,
                    session_manager=None,
                    session_store=None,
                    memory_manager=None,
                    extra_memory=extra_memory,
                    trigger_event_id=trigger_event_id,
                    trigger_text_preview=trigger_text_preview,
                    group_compressor=None,
                    multi_agent_internal=True,
                )
                cleaned = output.strip()
                if not _looks_like_role_stall_output(cleaned):
                    return role_id, role_name, cleaned
                rewrite_prompt = (
                    f"你是角色「{role_name}」。下面是一次无效输出，请改写成可直接交付的轮次结果。\n"
                    "要求：\n"
                    "1) 给出本轮结论（必须具体）\n"
                    "2) 给出3-5条可执行行动项（禁止提问）\n"
                    "3) 给出需要其他角色配合的点\n"
                    "4) 若信息不足，写明“假设”但仍要给出可执行版本\n\n"
                    f"无效输出：\n{cleaned}\n"
                )
                rewritten = await run_agent(
                    message=rewrite_prompt,
                    session_id=f"{session_id}::ma::{role_id}::rewrite::{round_no}",
                    config=role_cfg,
                    on_stream=None,
                    session=None,
                    router=router,
                    registry=round_registry,
                    on_tool_call=None,
                    on_tool_result=None,
                    images=None,
                    session_manager=None,
                    session_store=None,
                    memory_manager=None,
                    extra_memory=extra_memory,
                    trigger_event_id=trigger_event_id,
                    trigger_text_preview=trigger_text_preview,
                    group_compressor=None,
                    multi_agent_internal=True,
                )
                rewritten_clean = rewritten.strip()
                if not _looks_like_role_stall_output(rewritten_clean):
                    return role_id, role_name, rewritten_clean
                fallback = "观点：在既有信息下先交付可用版本，假设最小化并保留下一轮优化空间。"
                return role_id, role_name, fallback
            except Exception as exc:
                last_error = str(exc)
                if not _is_transient_multi_agent_error(exc) or attempt >= 3:
                    break
                await asyncio.sleep(0.8 * attempt)
        return role_id, role_name, f"（角色执行失败，已跳过：{last_error or '未知错误'}）"

    aborted_round_no: int | None = None
    for round_no in range(1, max_rounds + 1):
        round_tools = _choose_round_tool_lock(
            registry=registry,
            user_message=message,
            shared_context=shared_context,
            round_no=round_no,
            requested_deliverables=requested_deliverables,
        )
        round_registry = _subset_registry(registry, set(round_tools))
        role_outputs: list[tuple[str, str, str]] = []
        if mode == "parallel":
            role_outputs = await asyncio.gather(
                *[
                    _run_one_role(
                        role,
                        round_no,
                        shared_context,
                        round_tools,
                        round_registry,
                        idx * 0.25,
                    )
                    for idx, role in enumerate(roles)
                ],
                return_exceptions=False,
            )
        else:
            for role in roles:
                role_outputs.append(
                    await _run_one_role(
                        role,
                        round_no,
                        shared_context,
                        round_tools,
                        round_registry,
                    )
                )

        lines: list[str] = [f"### 第 {round_no} 轮角色观点"]
        for _, role_name, out in role_outputs:
            result = _clip_text_for_role_view((out or "（无输出）").strip(), 200)
            lines.append(f"- {role_name}：{result}")
        role_round_block = "\n".join(lines)
        history_blocks.append(role_round_block)

        requested_text = ", ".join(requested_deliverables) if requested_deliverables else "未指定"
        _file_deliverable_types = {"ppt", "word", "excel", "html", "pdf"}
        requires_file_output = bool(
            _file_deliverable_types & set(requested_deliverables)
        )
        prev_round_block = ""
        if round_no > 1 and round_deliveries:
            prev_summary = _extract_round_delivery_section(round_deliveries[-1]) or round_deliveries[-1]
            prev_summary = _truncate_to_tokens(prev_summary, 400)
            prev_round_block = (
                f"上一轮摘要（仅供参考，本轮须独立完成）：\n{prev_summary}\n"
            )
            if prev_round_artifact_baseline:
                prev_round_block += (
                    f"{prev_round_artifact_baseline}\n"
                    "本轮不得少于上轮，配图须重新搜索，布局和文案应有迭代。\n"
                )

        image_instruction = ""
        if include_image_output:
            image_instruction = (
                "配图流程：先列配图清单（按页/模块），再逐项调用 search_images 搜图下载，"
                "不同主体分别搜索，同一张图不要用于多页。\n"
            )

        round_synthesize_message: str = (
            "你是结果协调者。基于角色输出，独立完成本轮交付。\n"
            "【最高优先级】禁止向用户提问、要求补充材料或输出需求澄清模板。"
            "禁止输出执行计划或步骤预告（如「我先搜图再写脚本」「请稍等」），直接调用工具执行。"
            "你必须直接给出完整交付结果，信息不足时自行合理假设并执行。\n"
            "【语言】所有输出（文本、脚本中的文案、文件内容）的语言必须与用户议题的语言一致。\n"
            f"议题：{message}\n"
            f"场景：{_MULTI_AGENT_SCENARIO_LABELS.get(scenario, scenario or '自定义')}\n"
            f"轮次：第 {round_no}/{max_rounds} 轮\n"
            f"{requirement_baseline}\n"
            f"工具锁定：{', '.join(round_tools)}\n"
            f"交付类型：{requested_text}\n"
            f"角色输出：\n{role_round_block}\n\n"
            f"{prev_round_block}"
            f"{image_instruction}"
            "\n"
            f"## 第 {round_no} 轮交付（严格按此格式，每项必写）\n"
            "1. 本轮工具锁定（原样回写）\n"
            "2. 本轮可直接交付结果（必须给出具体内容）\n"
            "3. 配图\n"
            "4. 与上一轮相比的改进点（首轮写「首轮基线」）\n"
            "\n"
            "规则：\n"
            f"- 需要文件时必须调用工具生成并返回绝对路径，文件名必须用 _V{round_no} 后缀（当前是第{round_no}轮，只能用 _V{round_no}）。\n"
            "- 脚本中图片路径硬编码绝对路径，禁止 os.environ。\n"
            "- PPT 插图直接用 slide.shapes.add_picture(path, left, top, width=w, height=h)，"
            "严禁对图片做任何预处理（禁止 PIL resize/crop/thumbnail，禁止 cv2 resize），"
            "系统会自动后处理人脸感知裁剪。\n"
            "- PPT 中文字必须在图片上方（先 add_picture 再 add_textbox），"
            "文本框之间不要位置重叠。\n"
            "- PPT 文本框必须设置 word_wrap=True，"
            "所有文本框的右边界不得超过幻灯片宽度。\n"
            "- PPT 布局：只有封面页可以用全屏大图做背景，内容页必须采用图文混排布局"
            "（如左图右文、上图下文、图占页面1/3~1/2），禁止每页都铺满整张图。\n"
            "- 不允许越做越差。\n"
        )
        round_delivery = ""
        delivery_accepted = False
        last_round_error = ""

        async def _run_round_coordinator(
            pass_no: int,
            base_message: str = round_synthesize_message,
            current_round_no: int = round_no,
            current_registry: ToolRegistry = round_registry,
        ) -> tuple[str, bool, str]:
            msg = base_message
            if pass_no == 2:
                msg += (
                    "\n这是本轮自动重跑（第2次也是最后一次）。"
                    "请严格围绕当前议题与上一轮交付输出最终结果；"
                    "禁止泛化模板与跑题内容。\n"
                )
            try:
                out = await run_agent(
                    message=msg,
                    session_id=f"{session_id}::ma::round::{current_round_no}::p{pass_no}",
                    config=config,
                    on_stream=None,
                    session=None,
                    router=router,
                    registry=current_registry,
                    on_tool_call=None,
                    on_tool_result=None,
                    images=None,
                    session_manager=None,
                    session_store=None,
                    memory_manager=None,
                    extra_memory=extra_memory,
                    trigger_event_id=trigger_event_id,
                    trigger_text_preview=trigger_text_preview,
                    group_compressor=None,
                    multi_agent_internal=True,
                )
                clean = out.strip()
                ok = bool(clean) and not _looks_like_bad_coordinator_output(clean)
                return clean or out, ok, ""
            except Exception as exc:
                return "", False, str(exc)

        def _has_real_file_in_output(text: str) -> bool:
            paths = _extract_delivery_artifact_paths(text, include_scripts=False)
            if any(Path(p).expanduser().exists() for p in paths):
                return True
            script_paths: list[str] = []
            for m in _ABS_FILE_PATH_RE.finditer(text):
                p = m.group(1).strip()
                if not p:
                    continue
                suffix = Path(p).suffix.lower()
                if suffix in _NON_DELIVERY_EXTS:
                    script_paths.append(p)
                    continue
                if Path(p).expanduser().exists():
                    return True
            for sp in script_paths:
                sp_path = Path(sp).expanduser()
                if not sp_path.exists() or sp_path.suffix.lower() != ".py":
                    continue
                try:
                    script_text = sp_path.read_text("utf-8", errors="ignore")[:8000]
                except Exception:
                    continue
                for sm in _ABS_FILE_PATH_RE.finditer(script_text):
                    sp2 = sm.group(1).strip()
                    if not sp2:
                        continue
                    if Path(sp2).suffix.lower() in _NON_DELIVERY_EXTS:
                        continue
                    if Path(sp2).expanduser().exists():
                        return True
            return False

        first_out, first_ok, first_err = await _run_round_coordinator(1)
        file_missing_on_first = False
        if first_ok:
            round_delivery = first_out
            delivery_accepted = True
            if requires_file_output and not _has_real_file_in_output(first_out):
                file_missing_on_first = True
                delivery_accepted = False
                last_round_error = "协调者未产出实际文件（只有文字或脚本，没有最终交付文件）"
                if on_stream is not None:
                    await on_stream(
                        f"[多Agent] 第 {round_no} 轮未检测到交付文件，自动重跑。"
                    )
        if not delivery_accepted and not first_ok:
            last_round_error = first_err
            if on_stream is not None:
                await on_stream(
                    f"[多Agent] 第 {round_no} 轮首次失败，自动重跑本轮一次（止损上限=2次）。"
                )
        if not delivery_accepted:
            retry_msg = round_synthesize_message
            if requires_file_output:
                retry_msg += (
                    "\n重要：上一次尝试未产出实际文件。本次你必须完成以下全部步骤，不可只输出文字就结束：\n"
                    "步骤1) search_images 搜图并下载（按配图清单搜，搜完立刻进入步骤2，不要反复追加搜图）\n"
                    "步骤2) file_write 写完整 Python 生成脚本到 /tmp/（图片路径硬编码为步骤1的绝对路径）\n"
                    "步骤3) bash 执行脚本产出最终文件（.pptx/.docx/.xlsx/.html/.pdf）\n"
                    "步骤4) 在交付文本中写明最终文件的绝对路径\n"
                    "关键：搜图完成后必须立刻写脚本执行，不要停留在搜图步骤。\n"
                )
            second_out, second_ok, second_err = await _run_round_coordinator(
                2, base_message=retry_msg
            )
            if second_ok:
                round_delivery = second_out
                delivery_accepted = True
                if requires_file_output and not _has_real_file_in_output(second_out):
                    delivery_accepted = False
            if not delivery_accepted:
                last_round_error = second_err or last_round_error
                best_text = second_out or (first_out if file_missing_on_first else "")
                if requires_file_output and best_text and not _has_real_file_in_output(best_text):
                    pass
                elif best_text:
                    round_delivery = best_text
                    delivery_accepted = True

        if not delivery_accepted:
            fallback_base = (
                f"## 第 {round_no} 轮交付\n"
                f"1) 本轮工具锁定：{', '.join(round_tools)}\n"
            )
            if round_no > 1 and round_deliveries:
                prev = _extract_round_delivery_section(round_deliveries[-1]) or round_deliveries[-1]
                base = (
                    fallback_base
                    + "2) 本轮可直接交付结果（降级参考上一轮内容）：\n"
                    + prev
                    + "\n"
                    + "3) 结果协调者执行记录：本轮协调输出未达交付标准，"
                    + "已参考上一轮交付内容生成本轮基线。"
                )
                if last_round_error:
                    base += f" 错误摘要：{last_round_error[:180]}"
                base += "\n"
                base += "4) 与上一轮相比的改进点：本轮为稳定性兜底，未引入新改动。\n"
            else:
                base = (
                    fallback_base
                    + "2) 本轮可直接使用的产物：\n"
                    "- 基于角色观点的可执行摘要\n"
                    "- 当前轮次的风险与下一步建议\n"
                    "3) 结果协调者执行记录：未拿到稳定工具回执，已返回可直接使用文本版本。\n"
                    "4) 与上一轮相比的改进点：首轮基线。\n"
                )
            round_delivery = base
            aborted_round_no = round_no

        round_deliveries.append(round_delivery)
        allow_script_paths = "script" in requested_deliverables
        detected_paths = _extract_delivery_artifact_paths(
            round_delivery,
            include_scripts=allow_script_paths,
        )
        existing_paths = [
            str(Path(p).expanduser().resolve())
            for p in detected_paths
            if Path(p).expanduser().exists()
        ]
        existing_paths = _fix_version_suffix(existing_paths, round_no)
        if existing_paths:
            verified_lines = [f"- {p}" for p in existing_paths]
            round_delivery = (
                f"{round_delivery.rstrip()}\n\n"
                "6) 系统校验的实际产物绝对路径（以此为准）\n"
                + "\n".join(verified_lines)
            )
        if existing_paths:
            _snapshot_round_artifacts(existing_paths, round_no)
            prev_round_artifact_baseline = _extract_artifact_baseline(existing_paths)
        context_chunks: list[str] = []
        for idx, delivery in enumerate(round_deliveries, 1):
            section = _extract_round_delivery_section(delivery) or delivery
            context_chunks.append(f"[第{idx}轮最终交付]\n{section.strip()}")
        shared_context = _truncate_to_tokens(
            f"{requirement_baseline}\n\n" + "\n\n".join(context_chunks),
            1400,
        )
        display_delivery = f"{role_round_block}\n\n{round_delivery}"

        if on_round_result is not None:
            await on_round_result(round_no, display_delivery)

        if on_stream is not None:
            await on_stream(
                "[多Agent] "
                f"第 {round_no} 轮交付：\n"
                f"{display_delivery}\n"
            )

        if aborted_round_no == round_no:
            if on_stream is not None:
                await on_stream(
                    f"[多Agent] 第 {round_no} 轮连续两次失败，"
                    "已提前结束后续轮次以避免偏离目标与额外消耗。"
                )
            break

    if aborted_round_no is not None:
        return (
            f"多Agent执行已提前结束，停在第 {aborted_round_no} 轮（该轮两次失败已止损）。"
            f"已产出 {len(round_deliveries)} 轮结果。"
        )
    return f"多Agent执行完成，共 {len(round_deliveries)} 轮。"


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
    on_round_result: OnRoundResult | None = None,
    images: list[ImageContent] | None = None,
    session_manager: SessionManager | None = None,
    session_store: SessionStore | None = None,
    memory_manager: "MemoryManager | None" = None,
    extra_memory: str = "",
    trigger_event_id: str = "",
    trigger_text_preview: str = "",
    group_compressor: "SessionGroupCompressor | None" = None,
    multi_agent_internal: bool = False,
) -> str:
    """Run the Agent loop with tool support and multi-turn context.

    The loop is provider-agnostic:
    1. Check if provider supports native tools API
    2. If yes  -> pass schemas via ``tools=``; parse structured tool_calls
    3. If no   -> inject tool descriptions into system prompt; parse JSON text
    4. Execute tools, append results, loop (until no tool calls or token budget exhausted)
    5. Return final text reply
    """
    agent_cfg = config.agent
    models_cfg = config.models
    summarizer_cfg = agent_cfg.summarizer

    model_id: str = session.model if session else agent_cfg.model
    if router is None:
        router = ModelRouter(models_cfg)
    if registry is None:
        registry = create_default_registry()

    if not multi_agent_internal:
        ma_cfg = _resolve_multi_agent_cfg(config, session)
        await _sync_multi_agent_compression_boundary(
            session,
            session_manager,
            group_compressor,
            ma_enabled=bool(ma_cfg.get("enabled", False)),
        )
        if bool(ma_cfg.get("enabled", False)):
            if session is not None:
                state = str(session.metadata.get("multi_agent_state", "")).strip().lower()
                waiting = state == "confirm" or (state == "" and bool(
                    session.metadata.get("multi_agent_waiting_confirm", False)
                ))
                intro_done = bool(session.metadata.get("multi_agent_intro_done", False))
                pending_topic = str(session.metadata.get("multi_agent_pending_topic", "")).strip()

                if waiting and _is_multi_agent_cancel(message):
                    session.metadata.pop("multi_agent_state", None)
                    session.metadata.pop("multi_agent_waiting_confirm", None)
                    session.metadata.pop("multi_agent_intro_done", None)
                    session.metadata.pop("multi_agent_pending_topic", None)
                    session.metadata.pop("multi_agent_pending_rounds", None)
                    await _persist_session_metadata(session, session_manager)
                    return "已取消本次多Agent执行。你可以继续普通对话，或发新需求后再确认启动。"

                rounds_override = _extract_multi_agent_rounds(message)
                if rounds_override is not None and waiting:
                    session.metadata["multi_agent_pending_rounds"] = rounds_override
                    topic = _attach_rounds_marker(pending_topic or message, rounds_override)
                    session.metadata["multi_agent_pending_topic"] = topic
                    await _persist_session_metadata(session, session_manager)
                    ma_cfg["max_rounds"] = rounds_override
                    return await _run_multi_agent_controller_discussion(
                        user_message=message,
                        pending_topic=topic,
                        cfg=ma_cfg,
                        session_id=session_id,
                        config=config,
                        router=router,
                        registry=registry,
                        extra_memory=extra_memory,
                        trigger_event_id=trigger_event_id,
                        trigger_text_preview=trigger_text_preview,
                        include_intro=not intro_done,
                    )

                if state == "discuss":
                    if _is_multi_agent_cancel(message):
                        session.metadata.pop("multi_agent_state", None)
                        session.metadata.pop("multi_agent_intro_done", None)
                        session.metadata.pop("multi_agent_pending_topic", None)
                        session.metadata.pop("multi_agent_pending_rounds", None)
                        await _persist_session_metadata(session, session_manager)
                        return "已取消本次多Agent讨论。你可以继续普通对话。"

                    topic = pending_topic

                    if _is_multi_agent_discuss_done(message):
                        rounds_raw = session.metadata.get("multi_agent_pending_rounds")
                        if isinstance(rounds_raw, int):
                            ma_cfg["max_rounds"] = max(1, min(rounds_raw, 10))
                        cleaned_topic, marker_rounds = _extract_rounds_marker(topic or "")
                        if not isinstance(rounds_raw, int) and marker_rounds is not None:
                            ma_cfg["max_rounds"] = marker_rounds
                        session.metadata.pop("multi_agent_state", None)
                        session.metadata.pop("multi_agent_waiting_confirm", None)
                        session.metadata.pop("multi_agent_intro_done", None)
                        session.metadata.pop("multi_agent_pending_topic", None)
                        session.metadata.pop("multi_agent_pending_rounds", None)
                        await _persist_session_metadata(session, session_manager)
                        return await _run_multi_agent_executor(
                            message=cleaned_topic or "（请补充你的任务目标）",
                            session_id=session_id,
                            config=config,
                            on_stream=on_stream,
                            router=router,
                            registry=registry,
                            images=images,
                            extra_memory=extra_memory,
                            trigger_event_id=trigger_event_id,
                            trigger_text_preview=trigger_text_preview,
                            ma_cfg=ma_cfg,
                            on_round_result=on_round_result,
                        )

                    if message.strip():
                        if topic:
                            topic = f"{topic}\n补充要求: {message.strip()}".strip()
                        else:
                            topic = message.strip()
                        session.metadata["multi_agent_pending_topic"] = topic
                    if rounds_override is not None:
                        session.metadata["multi_agent_pending_rounds"] = rounds_override
                        session.metadata["multi_agent_pending_topic"] = _attach_rounds_marker(
                            topic or message,
                            rounds_override,
                        )
                        topic = str(session.metadata["multi_agent_pending_topic"])
                        ma_cfg["max_rounds"] = rounds_override

                    include_intro = not bool(session.metadata.get("multi_agent_intro_done", False))
                    if include_intro:
                        session.metadata["multi_agent_intro_done"] = True
                    await _persist_session_metadata(session, session_manager)
                    return await _run_multi_agent_controller_discussion(
                        user_message=message,
                        pending_topic=topic or message,
                        cfg=ma_cfg,
                        session_id=session_id,
                        config=config,
                        router=router,
                        registry=registry,
                        extra_memory=extra_memory,
                        trigger_event_id=trigger_event_id,
                        trigger_text_preview=trigger_text_preview,
                        include_intro=include_intro,
                    )

                if waiting and _is_multi_agent_confirm(message):
                    topic = pending_topic or "（请补充你的任务目标）"
                    rounds_raw = session.metadata.get("multi_agent_pending_rounds")
                    if isinstance(rounds_raw, int):
                        ma_cfg["max_rounds"] = max(1, min(rounds_raw, 10))
                    cleaned_topic, marker_rounds = _extract_rounds_marker(topic or "")
                    if not isinstance(rounds_raw, int) and marker_rounds is not None:
                        ma_cfg["max_rounds"] = marker_rounds
                    session.metadata.pop("multi_agent_state", None)
                    session.metadata.pop("multi_agent_waiting_confirm", None)
                    session.metadata.pop("multi_agent_intro_done", None)
                    session.metadata.pop("multi_agent_pending_topic", None)
                    session.metadata.pop("multi_agent_pending_rounds", None)
                    await _persist_session_metadata(session, session_manager)
                    return await _run_multi_agent_executor(
                        message=cleaned_topic or topic,
                        session_id=session_id,
                        config=config,
                        on_stream=on_stream,
                        router=router,
                        registry=registry,
                        images=images,
                        extra_memory=extra_memory,
                        trigger_event_id=trigger_event_id,
                        trigger_text_preview=trigger_text_preview,
                        ma_cfg=ma_cfg,
                        on_round_result=on_round_result,
                    )

                if waiting and not _is_multi_agent_confirm(message):
                    topic = pending_topic
                    if message.strip() and not _is_multi_agent_cancel(message):
                        topic = f"{topic}\n补充要求: {message.strip()}".strip()
                        session.metadata["multi_agent_pending_topic"] = topic
                        await _persist_session_metadata(session, session_manager)
                    rounds_raw = session.metadata.get("multi_agent_pending_rounds")
                    if isinstance(rounds_raw, int):
                        ma_cfg["max_rounds"] = max(1, min(rounds_raw, 10))
                    else:
                        _, marker_rounds = _extract_rounds_marker(topic or "")
                        if marker_rounds is not None:
                            ma_cfg["max_rounds"] = marker_rounds
                    topic = topic or "（请补充你的任务目标）"
                    return await _run_multi_agent_controller_discussion(
                        user_message=message,
                        pending_topic=topic,
                        cfg=ma_cfg,
                        session_id=session_id,
                        config=config,
                        router=router,
                        registry=registry,
                        extra_memory=extra_memory,
                        trigger_event_id=trigger_event_id,
                        trigger_text_preview=trigger_text_preview,
                        include_intro=False,
                    )

                session.metadata["multi_agent_state"] = "discuss"
                session.metadata["multi_agent_waiting_confirm"] = False
                session.metadata["multi_agent_intro_done"] = True
                session.metadata["multi_agent_pending_topic"] = message.strip() or message
                session.metadata.pop("multi_agent_pending_rounds", None)
                await _persist_session_metadata(session, session_manager)
                return await _run_multi_agent_controller_discussion(
                    user_message=message,
                    pending_topic=message.strip() or message,
                    cfg=ma_cfg,
                    session_id=session_id,
                    config=config,
                    router=router,
                    registry=registry,
                    extra_memory=extra_memory,
                    trigger_event_id=trigger_event_id,
                    trigger_text_preview=trigger_text_preview,
                    include_intro=True,
                )

            return await _run_multi_agent_executor(
                message=message,
                session_id=session_id,
                config=config,
                on_stream=on_stream,
                router=router,
                registry=registry,
                images=images,
                extra_memory=extra_memory,
                trigger_event_id=trigger_event_id,
                trigger_text_preview=trigger_text_preview,
                ma_cfg=ma_cfg,
                on_round_result=on_round_result,
            )

    if _is_evomap_status_question(message):
        enabled = _is_evomap_enabled(config)
        switch_text = "已开启" if enabled else "已关闭"
        return (
            f"当前 EvoMap 开关{switch_text}（本地配置状态）。"
            "如果你要我检查远端服务连通性，我可以再单独做一次连通检测。"
        )

    metadata_dirty = False

    llm_message = message
    locked_skill_ids: list[str] = []
    previous_locked_skill_ids: list[str] = []
    lock_is_explicit = False
    pending_lock_skill_ids: list[str] = []
    lock_waiting_done = False
    skill_announce_pending = False
    routed_skills: list[Skill] = []
    routed_skill_ids: list[str] = []
    if not multi_agent_internal:
        if session is not None:
            raw_locked = session.metadata.get("locked_skill_ids")
            if isinstance(raw_locked, list):
                locked_skill_ids = [
                    str(x).strip().lower()  # pyright: ignore[reportUnknownArgumentType]
                    for x in raw_locked  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(x, str) and str(x).strip()
                ]
                if locked_skill_ids:
                    lock_is_explicit = True
            elif isinstance(session.metadata.get("forced_skill_id"), str):
                legacy_forced = str(session.metadata.get("forced_skill_id", "")).strip().lower()
                if legacy_forced:
                    locked_skill_ids = [legacy_forced]
                    lock_is_explicit = True
                    session.metadata["locked_skill_ids"] = locked_skill_ids
                    session.metadata.pop("forced_skill_id", None)
                    metadata_dirty = True
            raw_waiting = session.metadata.get("skill_lock_waiting_done")
            lock_waiting_done = bool(raw_waiting)
            skill_announce_pending = bool(session.metadata.get("skill_lock_announce_pending"))
        previous_locked_skill_ids = list(locked_skill_ids)

        use_cmd = _parse_use_command(message)
        if use_cmd is not None:
            use_skill_ids, remainder = use_cmd
            if len(use_skill_ids) == 1 and use_skill_ids[0] in _USE_CLEAR_IDS:
                locked_skill_ids = []
                lock_is_explicit = False
                pending_lock_skill_ids = []
                lock_waiting_done = False
                skill_announce_pending = False
                if session is not None:
                    session.metadata.pop("locked_skill_ids", None)
                    session.metadata.pop("skill_lock_waiting_done", None)
                    session.metadata.pop("skill_lock_announce_pending", None)
                    session.metadata.pop("skill_param_state", None)
                    metadata_dirty = True
                if remainder:
                    llm_message = remainder
            else:
                locked_skill_ids = use_skill_ids
                lock_is_explicit = True
                lock_waiting_done = False
                skill_announce_pending = True
                if session is not None:
                    session.metadata["locked_skill_ids"] = locked_skill_ids
                    session.metadata["skill_lock_waiting_done"] = False
                    session.metadata["skill_lock_announce_pending"] = True
                    metadata_dirty = True
                llm_message = remainder or f"使用技能 {', '.join(locked_skill_ids)} 处理当前请求。"
        elif lock_waiting_done and _is_task_done_confirmation(message):
            locked_skill_ids = []
            lock_is_explicit = False
            lock_waiting_done = False
            if session is not None:
                session.metadata.pop("locked_skill_ids", None)
                session.metadata.pop("skill_lock_waiting_done", None)
                session.metadata.pop("skill_lock_announce_pending", None)
                session.metadata.pop("skill_param_state", None)
                if session_manager is not None:
                    await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                        session.id,
                        metadata=session.metadata,
                    )
            return "已确认任务完成，已解除本轮技能锁定。"
        elif lock_waiting_done:
            lock_waiting_done = False
            if session is not None:
                session.metadata["skill_lock_waiting_done"] = False
                metadata_dirty = True

        routed_skills = _assembler.route_skills(llm_message)
        routed_skill_ids = _normalize_skill_ids(routed_skills)

        if not locked_skill_ids and routed_skill_ids:
            pending_lock_skill_ids = routed_skill_ids

        if lock_is_explicit and locked_skill_ids and routed_skill_ids and routed_skill_ids != locked_skill_ids:
            if _is_skill_switch_consent(message) or _looks_like_skill_activation_message(message):
                locked_skill_ids = routed_skill_ids
                lock_waiting_done = False
                skill_announce_pending = True
                if session is not None:
                    session.metadata["locked_skill_ids"] = locked_skill_ids
                    session.metadata["skill_lock_waiting_done"] = False
                    session.metadata["skill_lock_announce_pending"] = True
                    metadata_dirty = True
            else:
                routed_skill_ids = locked_skill_ids

        if lock_is_explicit and locked_skill_ids and not lock_waiting_done and session is not None:
            locked_skills = _assembler.route_skills(llm_message, forced_skill_ids=locked_skill_ids)
            guards = _guarded_skills(locked_skills)
            if guards:
                state_map_raw = session.metadata.get("skill_param_state")
                state_map: dict[str, dict[str, object]] = state_map_raw.copy() if isinstance(state_map_raw, dict) else {}  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
                missing_any = False
                for skill in guards:
                    guard = skill.param_guard
                    if guard is None:
                        continue
                    skill_state_raw = state_map.get(skill.id, {})
                    skill_state: dict[str, object] = (
                        skill_state_raw.copy()  # pyright: ignore[reportUnknownMemberType]
                        if isinstance(skill_state_raw, dict)  # pyright: ignore[reportUnnecessaryIsInstance]
                        else {}
                    )
                    updated, missing = _update_guard_state(
                        guard.params, skill_state, llm_message, images
                    )
                    state_map[skill.id] = updated
                    missing_any = missing_any or missing
                session.metadata["skill_param_state"] = state_map
                metadata_dirty = True
                if missing_any:
                    if session_manager is not None:
                        await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                            session.id,
                            metadata=session.metadata,
                        )
                    blocks = [
                        _build_skill_param_guard_reply(
                            s.id, s.param_guard.params, state_map.get(s.id, {})  # pyright: ignore[reportUnknownArgumentType]
                        )
                        for s in guards
                        if s.param_guard is not None
                    ]
                    return "\n\n".join(blocks)

    if session is not None:
        pending_raw = session.metadata.get("evomap_pending_choices")
        if isinstance(pending_raw, dict):
            options_raw: object = pending_raw.get("options")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            options: list[dict[str, str]] = []
            if isinstance(options_raw, list):
                for item in options_raw:  # pyright: ignore[reportUnknownVariableType]
                    if not isinstance(item, dict):
                        continue
                    opt: dict[str, object] = item  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
                    aid = str(opt.get("asset_id", "")).strip()
                    summary = str(opt.get("summary", "")).strip()
                    if not summary and not aid:
                        continue
                    options.append({"asset_id": aid, "summary": summary})
            choice_idx = _extract_evomap_choice_index(message, len(options))
            if choice_idx is not None and options:
                selected = options[choice_idx]
                selected_hint = (
                    f"【EvoMap 已选方案】\n- {selected['asset_id']}: {selected['summary']}"
                )
                extra_memory = (
                    f"{selected_hint}\n{extra_memory}" if extra_memory.strip() else selected_hint
                )
                origin_message = str(pending_raw.get("origin_message", "")).strip()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                if origin_message:
                    llm_message = (
                        f"{origin_message}\n"
                        f"用户已选择方案：{selected['summary']}。\n"
                        "请按该方案执行。"
                    )
                session.metadata.pop("evomap_pending_choices", None)
                if session_manager is not None:
                    await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                        session.id,
                        metadata=session.metadata,
                    )

    assistant_name = _DEFAULT_ASSISTANT_NAME
    if memory_manager is not None:
        try:
            current_name = await memory_manager.get_assistant_name()
            if current_name:
                assistant_name = current_name
        except Exception as exc:
            log.debug("agent.assistant_name_load_failed", session_id=session_id, error=str(exc))
    name_action, requested_name = _detect_assistant_name_update(message)
    if name_action == "set":
        assistant_name = requested_name
        if memory_manager is not None:
            try:
                changed = await memory_manager.set_assistant_name(
                    requested_name,
                    source=f"session:{session_id}",
                )
                if changed:
                    log.info(
                        "agent.assistant_name_updated",
                        session_id=session_id,
                        assistant_name=requested_name,
                    )
            except Exception as exc:
                log.debug(
                    "agent.assistant_name_save_failed",
                    session_id=session_id,
                    error=str(exc),
                )
    elif name_action == "reset":
        assistant_name = _DEFAULT_ASSISTANT_NAME
        if memory_manager is not None:
            try:
                removed = await memory_manager.clear_assistant_name()
                log.info(
                    "agent.assistant_name_reset",
                    session_id=session_id,
                    removed=removed,
                )
            except Exception as exc:
                log.debug(
                    "agent.assistant_name_reset_failed",
                    session_id=session_id,
                    error=str(exc),
                )

    native_tools = router.supports_native_tools(model_id)
    selected_tool_names: set[str] | None = None
    evomap_enabled = _is_evomap_enabled(config)
    evomap_phase = "editing"
    if session is not None:
        raw_phase = session.metadata.get("evomap_phase")
        if isinstance(raw_phase, str) and raw_phase.strip():
            evomap_phase = raw_phase.strip().lower()
    if _is_tasky_message_for_evomap(llm_message):
        # First-turn task requests are treated as NEW_TASK directly to avoid
        # an extra classifier LLM call before normal planning/execution.
        if session is None or not session.messages:
            phase = "NEW_TASK"
        else:
            phase = await _llm_judge_task_phase(
                router,
                model_id,
                session=session,
                message=llm_message,
            )
    else:
        phase = "EDITING"
    if session is not None and session_manager is not None:
        desired = "start" if phase == "NEW_TASK" else "editing"
        if desired != evomap_phase:
            session.metadata["evomap_phase"] = desired
            await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                session.id,
                metadata=session.metadata,
            )
    evomap_allowed_for_turn = evomap_enabled and phase == "NEW_TASK"
    available_tool_names = {d.name for d in registry.list_tools()}
    evo_first_mode = evomap_allowed_for_turn and "evomap_fetch" in available_tool_names
    evomap_hint_hit = _extra_memory_has_evomap_hint(extra_memory)

    if native_tools:
        selected_tool_names = _select_native_tool_names(registry, llm_message)
        selected_tool_names = _force_include_office_edit_tools(
            selected_tool_names,
            available=available_tool_names,
            session=session,
            llm_message=llm_message,
        )
        if not evomap_allowed_for_turn:
            selected_tool_names = {
                name for name in selected_tool_names if not name.startswith("evomap_")
            }
        elif evo_first_mode:
            selected_tool_names.add("evomap_fetch")
        tool_schemas = registry.to_llm_schemas(include_names=selected_tool_names)
        dropped_names = sorted(available_tool_names - selected_tool_names)
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
    fallback_names: set[str] | None = None
    if not native_tools and not evomap_allowed_for_turn:
        fallback_names = {d.name for d in registry.list_tools() if not d.name.startswith("evomap_")}
    fallback_text = (
        "" if native_tools else registry.to_prompt_fallback(include_names=fallback_names)
    )

    effective_skill_ids = locked_skill_ids or pending_lock_skill_ids or None
    system_messages = _assembler.build(
        config,
        llm_message,
        tool_fallback_text=fallback_text,
        assistant_name=assistant_name,
        forced_skill_ids=effective_skill_ids,
    )
    if lock_is_explicit and locked_skill_ids:
        system_messages.append(_build_skill_lock_system_message(locked_skill_ids))
    _append_office_system_hints(system_messages, session, llm_message)
    image_api_probe_guard_enabled = _is_image_generation_request(llm_message)
    if image_api_probe_guard_enabled:
        system_messages.append(_build_image_generation_system_message())
    if evo_first_mode:
        system_messages.append(_build_evomap_first_system_message())
        if session is not None and session_manager is not None:
            session.metadata["evomap_phase"] = "editing"
            await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                session.id,
                metadata=session.metadata,
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
            should_recall, include_raw = memory_manager.recall_policy(llm_message)
            if not should_recall and _is_creation_task_message(llm_message):
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
                        llm_message,
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
    conversation.append(Message(role="user", content=llm_message, images=images))
    conversation_message_count = len(conversation)

    if (
        group_compressor is not None
        and session_store is not None
        and session is not None
        and summarizer_cfg.model.strip()
    ):
        _t_group_start = _time.monotonic()
        try:
            conversation_for_compress = conversation
            if isinstance(session.metadata, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
                raw_cutoff = session.metadata.get("compression_resume_message_index")
                if isinstance(raw_cutoff, int) and raw_cutoff > 0:
                    cutoff = max(0, min(raw_cutoff, len(conversation) - 1))
                    if cutoff > 0:
                        conversation_for_compress = conversation[cutoff:]
            conversation = await group_compressor.build_window_messages(
                session_id=session_id,
                messages=conversation_for_compress,
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
    trigger_preview = trigger_text_preview.strip() or _preview_text(llm_message)

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
    pending_office_paths: list[str] = []
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
    bash_fail_streak = 0
    same_failed_bash_streak = 0
    last_failed_bash_signature = ""
    blocked_tools: set[str] = set()
    office_block_bash_probe = False
    office_loop_guard_enabled = False
    office_block_message = ""
    office_edit_only = False
    office_edit_path = ""
    if session is not None:
        is_office_request = _is_office_edit_request(llm_message) or (
            _is_followup_edit_message(llm_message) and _has_any_last_office_path(session.metadata)
        )
        office_loop_guard_enabled = is_office_request
        if is_office_request and _has_any_last_office_path(session.metadata):
            office_block_bash_probe = True
            office_block_message = _build_office_path_block_message(session.metadata)
            last_pptx = session.metadata.get("last_pptx_path")
            if isinstance(last_pptx, str) and last_pptx.strip():
                office_edit_only = True
                office_edit_path = last_pptx.strip()
    low_value_bash_probe_streak = 0
    max_tool_rounds = max(1, int(agent_cfg.max_tool_rounds))

    round_idx = -1
    successful_tool_calls = 0
    browser_locked_by_evomap = evo_first_mode and not evomap_hint_hit
    while total_output < _MAX_OUTPUT_TOKENS:
        round_idx += 1
        if round_idx >= max_tool_rounds:
            final_text_parts.append(
                f"工具调用轮次已达上限（{max_tool_rounds} 轮），"
                "为避免长时间卡住已暂停。请让我改用更直接的编辑方式继续。"
            )
            break
        if db_summaries:
            all_messages = _context_window.trim_with_summaries(
                [*system_messages, *conversation],
                model_short,
                db_summaries,
            )
        else:
            all_messages = _context_window.trim(
                [*system_messages, *conversation],
                model_short,
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
                f"{tc.name} 已熔断，禁止继续调用" for tc in tool_calls if tc.name in blocked_tools
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
                            "请改用其他可用工具完成任务。"
                        ),
                    )
                )
                if invalid_tool_rounds >= 2:
                    final_text_parts.append("工具调用连续无效，已停止自动重试。请明确参数后重试。")
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
            final_text_parts.append("我这边没收到模型有效回复。请再发一次需求，我会继续处理。")
            break

        if not tool_calls:
            break

        if not announced_plan and on_stream:
            announced_plan = True
            has_text = content.strip() if content else ""
            if not has_text:
                tool_names = [tc.name for tc in tool_calls]
                plan = _make_plan_hint(tool_names, llm_message)
                await on_stream(plan)

        log.info(
            "agent.tool_calls",
            round=round_idx,
            count=len(tool_calls),
            tools=[tc.name for tc in tool_calls],
        )

        # Pre-flight: run evomap_fetch first; if it fails, silently remove it
        # so the LLM never sees the failed call — it just proceeds normally.
        _evomap_preflight_results: dict[str, tuple[str, ToolResult]] = {}
        _evomap_failed_ids: set[str] = set()
        for _pf_tc in tool_calls:
            if _pf_tc.name != "evomap_fetch":
                continue
            _pf_id, _pf_result = await _execute_tool(
                registry,
                _pf_tc,
                evomap_enabled=evomap_allowed_for_turn,
                browser_allowed=True,
                office_block_bash_probe=False,
                office_block_message="",
                office_edit_only=False,
                office_edit_path="",
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
            )
            if not _pf_result.success:
                log.info(
                    "agent.evomap_fetch_failed_fallback",
                    session_id=session_id,
                    error=_pf_result.error or _pf_result.output,
                )
                _evomap_failed_ids.add(_pf_tc.id)
            else:
                _evomap_preflight_results[_pf_tc.id] = (_pf_id, _pf_result)

        if _evomap_failed_ids:
            tool_calls = [tc for tc in tool_calls if tc.id not in _evomap_failed_ids]

        if not tool_calls:
            break

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
            if assistant_content:
                for office_path in _extract_office_paths(assistant_content):
                    if _remember_office_path(session.metadata, office_path):
                        metadata_dirty = True

        for _tname in ("reminder", "cron"):
            _tool = registry.get(_tname)
            if _tool is not None and hasattr(_tool, "current_session_id"):
                _tool.current_session_id = session_id  # type: ignore[union-attr]

        stop_for_evomap_choice = False
        stop_for_probe_loop = False
        for tc in tool_calls:
            if tc.name == "file_write" and isinstance(tc.arguments.get("content"), str):
                for office_path in _extract_office_paths(str(tc.arguments.get("content", ""))):
                    if office_path not in pending_office_paths:
                        pending_office_paths.append(office_path)
            if session is not None and tc.name in {"ppt_edit", "docx_edit", "xlsx_edit"}:
                has_path = isinstance(tc.arguments.get("path"), str) and bool(
                    str(tc.arguments.get("path", "")).strip()
                )
                if not has_path:
                    default_path = _get_default_office_edit_path(tc.name, session.metadata)
                    if default_path:
                        fixed_args = dict(tc.arguments)
                        fixed_args["path"] = default_path
                        tc = ToolCall(id=tc.id, name=tc.name, arguments=fixed_args)
                        log.info(
                            "agent.office_path_autofill",
                            tool=tc.name,
                            path=default_path,
                            session_id=session_id,
                        )

            if tc.id in _evomap_preflight_results:
                tc_id, result = _evomap_preflight_results[tc.id]
            else:
                tc_id, result = await _execute_tool(
                    registry,
                    tc,
                    evomap_enabled=evomap_allowed_for_turn,
                    browser_allowed=not browser_locked_by_evomap,
                    office_block_bash_probe=office_block_bash_probe,
                    office_block_message=office_block_message,
                    office_edit_only=office_edit_only,
                    office_edit_path=office_edit_path,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                )
            if (
                tc.name == "ppt_edit"
                and result.success
                and _mentions_specific_dark_bar_target(llm_message)
            ):
                action = str(tc.arguments.get("action", "replace_text")).strip().lower()
                if action == "apply_business_style" and "重设深色条 0 处" in (result.output or ""):
                    result = ToolResult(
                        success=False,
                        output=result.output,
                        error="未命中用户指定对象：黑色横条仍未被替换，请继续定向修改该元素",
                    )
                elif action == "set_background":
                    result = ToolResult(
                        success=False,
                        output=result.output,
                        error="用户要求修改黑色横条，仅设置背景不算完成，请继续定向修改该横条",
                    )
            if tc.name == "evomap_fetch" and result.success:
                candidates = _parse_evomap_fetch_candidates(result.output or "")
                if len(candidates) > 3 and session is not None:
                    top3 = _pick_top_evomap_candidates(llm_message, candidates, limit=3)
                    session.metadata["evomap_pending_choices"] = {
                        "origin_message": llm_message,
                        "options": [{"asset_id": aid, "summary": summary} for aid, summary in top3],
                    }
                    if session_manager is not None:
                        await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                            session.id,
                            metadata=session.metadata,
                        )
                    final_text_parts.append(_build_evomap_choice_prompt(top3))
                    stop_for_evomap_choice = True
                    break
                browser_locked_by_evomap = not _is_no_match_evomap_output(result)
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
            if tc.name == "bash":
                if result.success:
                    bash_fail_streak = 0
                    same_failed_bash_streak = 0
                    last_failed_bash_signature = ""
                else:
                    bash_fail_streak += 1
                    failed_sig = _normalize_bash_command_signature(
                        str(tc.arguments.get("command", ""))
                    )
                    if failed_sig and failed_sig == last_failed_bash_signature:
                        same_failed_bash_streak += 1
                    elif failed_sig:
                        same_failed_bash_streak = 1
                        last_failed_bash_signature = failed_sig
                    else:
                        same_failed_bash_streak = 0
                        last_failed_bash_signature = ""
                    if same_failed_bash_streak >= 3 and "bash" not in blocked_tools:
                        blocked_tools.add("bash")
                        log.warning(
                            "agent.tool_circuit_open",
                            tool="bash",
                            fail_streak=bash_fail_streak,
                            same_failed_streak=same_failed_bash_streak,
                            command_signature=last_failed_bash_signature[:200],
                            session_id=session_id,
                        )
                        conversation.append(
                            Message(
                                role="user",
                                content=(
                                    "[系统降级] 同一 bash 命令模板已连续失败 3 次，"
                                    "已自动熔断并切换策略。"
                                    "后续请不要再调用 bash。"
                                    "请改用结构化编辑工具（ppt_edit/docx_edit/xlsx_edit）"
                                    "或文件工具（file_read/file_write/file_edit）继续。"
                                ),
                            )
                        )
            if (
                office_loop_guard_enabled or image_api_probe_guard_enabled
            ) and _is_low_value_bash_probe(tc):
                low_value_bash_probe_streak += 1
            else:
                low_value_bash_probe_streak = 0
            if image_api_probe_guard_enabled and low_value_bash_probe_streak >= 2:
                final_text_parts.append(
                    "检测到连续探测环境但未进入实测，已停止循环以避免卡住。"
                    "下一步我将直接执行最小生图脚本（/tmp）并返回状态码与图片路径。"
                )
                stop_for_probe_loop = True
                break
            if office_loop_guard_enabled and low_value_bash_probe_streak >= 3:
                final_text_parts.append(
                    "检测到连续的文件探测命令（如 ls/stat/test）且无实质修改，"
                    "已停止循环以避免卡住。"
                    "我将改用文档局部编辑工具继续，请直接告诉我要改哪一页/哪一段/哪个单元格。"
                )
                stop_for_probe_loop = True
                break

            if result.success and result.output:
                successful_tool_calls += 1
                for path_match in re.finditer(
                    r"(/[^\s]+\.(?:jpg|jpeg|png|gif|webp))", result.output
                ):
                    real_image_paths.append(path_match.group(1))
            elif result.success:
                successful_tool_calls += 1
                if session is not None:
                    for office_path in _extract_office_paths(result.output):
                        if _remember_office_path(session.metadata, office_path):
                            metadata_dirty = True
            if session is not None and pending_office_paths:
                for office_path in list(pending_office_paths):
                    if Path(office_path).expanduser().exists():
                        if _remember_office_path(session.metadata, office_path):
                            metadata_dirty = True
                        pending_office_paths.remove(office_path)
            if (
                session is not None
                and tc.name == "bash"
                and result.success
                and _capture_latest_pptx(
                    session.metadata,
                    roots=(
                        Path("/tmp"),
                        Path("/private/tmp"),
                        Path.home() / "Downloads",
                        Path.home() / ".whaleclaw" / "workspace",
                    ),
                    window_seconds=240,
                )
            ):
                metadata_dirty = True

            if session is not None and tc.name in {"ppt_edit", "docx_edit", "xlsx_edit"}:
                arg_path = tc.arguments.get("path")
                if (
                    isinstance(arg_path, str)
                    and arg_path.strip()
                    and _remember_office_path(session.metadata, arg_path.strip())
                ):
                    metadata_dirty = True

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
                    content=(f"[工具 {tc.name} 执行结果]\n{tool_output}"),
                )
            conversation.append(tool_msg)

            if session_manager and session:
                snippet = tool_output[:500] if len(tool_output) > 500 else tool_output
                await _persist_message(
                    session_manager,
                    session,
                    "tool",
                    f"[{tc.name}] {snippet}",
                    tool_call_id=tc_id,
                    tool_name=tc.name,
                )

            log.debug(
                "agent.tool_result",
                tool=tc.name,
                success=result.success,
                output_len=len(result.output),
            )

        if stop_for_evomap_choice:
            break
        if stop_for_probe_loop:
            break
        if metadata_dirty and session is not None and session_manager is not None:
            await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                session.id,
                metadata=session.metadata,
            )
            metadata_dirty = False

        sig_parts: list[str] = []
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
                conversation.append(
                    Message(
                        role="user",
                        content=(
                            "[系统提示] 检测到你在重复执行相同操作且未取得进展。"
                            "请换一种方式解决问题，或直接向用户说明当前遇到的困难。"
                        ),
                    )
                )

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

    if lock_is_explicit and locked_skill_ids and skill_announce_pending:
        announce = _skill_announcement(locked_skill_ids, previous_locked_skill_ids)
        final_text = f"{announce}\n\n{final_text}" if final_text else announce
        skill_announce_pending = False
        if session is not None:
            session.metadata["skill_lock_announce_pending"] = False
            metadata_dirty = True

    _lock_confirm_tip = (
        "如果本轮任务已完成，请回复“任务完成”以解除技能锁定；"
        "若需继续修改，请直接继续说需求。"
    )

    # Deferred lock: first run completed with auto-routed skills -> lock them now.
    if (
        not lock_is_explicit
        and pending_lock_skill_ids
        and session is not None
        and successful_tool_calls > 0
    ):
        locked_skill_ids = pending_lock_skill_ids
        lock_is_explicit = True
        session.metadata["locked_skill_ids"] = locked_skill_ids
        session.metadata["skill_lock_waiting_done"] = True
        metadata_dirty = True
        final_text = f"{final_text}\n\n{_lock_confirm_tip}" if final_text else _lock_confirm_tip

    # Explicit lock: require user confirmation to release after successful tool use.
    elif (
        lock_is_explicit
        and locked_skill_ids
        and session is not None
        and successful_tool_calls > 0
        and not lock_waiting_done
    ):
        session.metadata["locked_skill_ids"] = locked_skill_ids
        session.metadata["skill_lock_waiting_done"] = True
        metadata_dirty = True
        final_text = f"{final_text}\n\n{_lock_confirm_tip}" if final_text else _lock_confirm_tip

    if metadata_dirty and session is not None and session_manager is not None:
        await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            session.id,
            metadata=session.metadata,
        )

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
                    Message(role=r.role if r.role != "tool" else "assistant", content=r.content)  # pyright: ignore[reportArgumentType]
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
