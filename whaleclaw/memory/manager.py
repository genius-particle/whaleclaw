"""Memory manager — recall, memorize, compact with token budget."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict
from datetime import UTC, datetime
from math import exp
from typing import TYPE_CHECKING, Any, Literal

from whaleclaw.memory.base import MemoryEntry, MemoryStore
from whaleclaw.memory.summary import ConversationSummarizer
from whaleclaw.providers.base import Message
from whaleclaw.utils.log import get_logger

if TYPE_CHECKING:
    from whaleclaw.providers.router import ModelRouter

log = get_logger(__name__)
_PROFILE_COMPRESS_TIMEOUT_SECONDS = 8


def _est_tokens(text: str) -> int:
    return max(0, len(text) // 3)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    char_cap = max_tokens * 3
    if len(text) <= char_cap:
        return text
    return text[:char_cap]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _has_tag(entry: MemoryEntry, prefix_or_tag: str) -> bool:
    return any(t == prefix_or_tag or t.startswith(prefix_or_tag) for t in entry.tags)


def _is_profile_entry(entry: MemoryEntry) -> bool:
    return _has_tag(entry, "memory_profile")


def _is_raw_entry(entry: MemoryEntry) -> bool:
    return _has_tag(entry, "auto_capture") or _has_tag(entry, "compact")


def _is_style_profile_entry(entry: MemoryEntry) -> bool:
    return _is_profile_entry(entry) and _has_tag(entry, "style:global")


def _split_query_terms(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[\w\u4e00-\u9fff]+", text) if len(w) >= 2}


def _tag_match_ratio(query_terms: set[str], tags: list[str]) -> float:
    if not query_terms or not tags:
        return 0.0
    tag_text = " ".join(tags).lower()
    hit = sum(1 for t in query_terms if t in tag_text)
    return min(1.0, hit / max(1, len(query_terms)))


def _recency_score(created_at: datetime, now: datetime) -> float:
    age_seconds = max(0.0, (now - created_at).total_seconds())
    age_days = age_seconds / 86400.0
    # Half-life ~ 7 days, keep a tiny floor for older long-term memories.
    return max(0.08, exp(-age_days / 7.0))


def _is_low_signal_text(text: str) -> bool:
    t = text.strip()
    if len(t) < 6:
        return True
    if t.startswith(("/", "{", "[")):
        return True
    return sum(ch.isdigit() for ch in t) > len(t) * 0.5


def _extract_json_block(text: str) -> dict[str, Any] | None:
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    candidates = fenced or [text]
    for raw in candidates:
        candidate = raw.strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start:end + 1]
        try:
            import json

            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _matches_capture_signal(
    text: str, mode: Literal["conservative", "balanced", "aggressive"]
) -> bool:
    t = text.strip()
    if not t:
        return False
    low = t.lower()

    if mode == "conservative":
        keys = (
            "记住",
            "请记住",
            "偏好",
            "喜欢",
            "不喜欢",
            "我是",
            "我叫",
            "prefer",
            "i like",
            "i am",
        )
        return any(k in t or k in low for k in keys)

    if mode == "balanced":
        keys = (
            "记住", "记得", "偏好", "风格", "语气", "以后", "下次", "默认",
            "请用", "请按", "请保持", "回答", "回复", "称呼", "叫我",
            "prefer", "preference", "from now on", "call me", "please answer",
        )
        if any(k in t or k in low for k in keys):
            return True
        if re.search(r"(回答|回复).*(简洁|简短|直接|要点|明了)", t):
            return True
        return bool(re.search(r"(以后|下次|默认|从现在起).*(请|用|按|保持)", t))

    # aggressive
    if len(t) < 6:
        return False
    if any(
        x in t or x in low
        for x in ("我是", "我叫", "我在", "我的", "i am", "my ", "i like", "prefer")
    ):
        return True
    return bool(re.search(r"(请|以后|下次|默认|记住|回答|回复)", t))


def _is_force_flush_capture(text: str) -> bool:
    if any(k in text for k in ("请记住", "记住", "务必记住")):
        return True
    return bool(re.search(r"(回答|回复).*(简洁|简短|直接|要点|明了)", text))


def _style_signal_hits(text: str) -> int:
    patterns = (
        r"(回答|回复).*(简洁|简短|直接|要点|明了)",
        r"(语气|风格).*(专业|严谨|友好|口语)",
        r"(先结论|先说结论|结论先行)",
        r"(控制在|不超过).*(字|条|段)",
        r"(默认|以后|下次).*(回答|回复|输出)",
    )
    return sum(1 for p in patterns if re.search(p, text))


def _infer_style_from_l0(l0: str) -> str:
    txt = l0.strip().replace("\n", " ")
    if not txt:
        return ""
    if any(k in txt for k in ("简洁", "简短", "直接", "要点")):
        return "回答风格：简洁明了，先给结论，再给必要细节；优先要点列表。"
    if any(k in txt for k in ("详细", "展开", "全面")):
        return "回答风格：先概览再展开，结构化分点说明，覆盖关键背景与步骤。"
    return ""


class MemoryManager:
    """Orchestrates recall and storage with token budget."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._summarizer = ConversationSummarizer()
        self._pending_by_source: dict[str, list[str]] = defaultdict(list)
        self._pending_since: dict[str, datetime] = {}
        self._pending_last_flush: dict[str, datetime] = {}
        self._profile_compress_cache: dict[str, str] = {}

    def recall_policy(self, query: str) -> tuple[bool, bool]:
        """Return (should_recall, include_raw_detail)."""
        low = query.lower().strip()
        if not low:
            return (False, False)
        # Always-on recall for non-empty user turns:
        # profile + raw retrieval are gated by scoring/budget downstream.
        return (True, True)

    async def recall(
        self,
        query: str,
        max_tokens: int = 500,
        limit: int = 10,
        *,
        include_profile: bool = True,
        include_raw: bool = True,
    ) -> str:
        parts: list[str] = []
        used = 0
        recent = await self._store.list_recent(limit=300)
        if include_profile:
            latest_l1 = next(
                (
                    e
                    for e in recent
                    if _is_profile_entry(e) and any(t == "level:L1" for t in e.tags)
                ),
                None,
            )
            if latest_l1 is not None:
                content = latest_l1.content.strip()
                if content:
                    txt = f"【长期记忆画像】\n{content}"
                    need = _est_tokens(txt)
                    if used + need <= max_tokens:
                        parts.append(txt)
                        used += need

        if include_raw:
            results = await self._store.search(query, limit=limit * 3, min_score=0.2)
            query_terms = _split_query_terms(query)
            now = datetime.now(UTC)
            ranked = sorted(
                results,
                key=lambda r: (
                    0.65 * r.score
                    + 0.25 * _recency_score(r.entry.created_at, now)
                    + 0.10 * _tag_match_ratio(query_terms, r.entry.tags)
                ),
                reverse=True,
            )
            for r in ranked:
                if len(parts) >= limit + (1 if include_profile else 0):
                    break
                content = r.entry.content.strip()
                if _is_low_signal_text(content):
                    continue
                txt = f"- {content}"
                need = _est_tokens(txt)
                if used + need > max_tokens:
                    break
                parts.append(txt)
                used += need
        return "\n".join(parts) if parts else ""

    async def build_profile_for_injection(
        self,
        *,
        max_tokens: int,
        router: ModelRouter | None = None,
        model_id: str = "",
    ) -> str:
        """Build profile injection text from latest L1, compressing by LLM if needed."""
        recent = await self._store.list_recent(limit=300)
        latest_l1 = next(
            (
                e
                for e in recent
                if _is_profile_entry(e) and any(t == "level:L1" for t in e.tags)
            ),
            None,
        )
        blocks: list[str] = []
        if latest_l1 and latest_l1.content.strip():
            blocks.append(f"【长期记忆画像】\n{latest_l1.content.strip()}")
        if not blocks:
            return ""

        profile_text = "\n\n".join(blocks)
        if _est_tokens(profile_text) <= max_tokens:
            return profile_text

        cache_key = self._profile_cache_key(
            profile_text=profile_text,
            model_id=model_id,
            max_tokens=max_tokens,
        )
        cached = self._profile_compress_cache.get(cache_key)
        if cached:
            log.debug("memory.profile_compress_cache_hit", model=model_id)
            return cached

        if router and model_id:
            try:
                compressed = await asyncio.wait_for(
                    self._compress_profile_with_llm(
                        router=router,
                        model_id=model_id,
                        profile_text=profile_text,
                        max_tokens=max_tokens,
                    ),
                    timeout=_PROFILE_COMPRESS_TIMEOUT_SECONDS,
                )
                if compressed:
                    self._profile_cache_set(cache_key, compressed)
                    return compressed
            except TimeoutError:
                log.warning(
                    "memory.profile_compress_timeout",
                    model=model_id,
                    timeout_s=_PROFILE_COMPRESS_TIMEOUT_SECONDS,
                )

        # Fallback: physical truncation.
        return _truncate_to_tokens(profile_text, max_tokens)

    @staticmethod
    def _profile_cache_key(
        *,
        profile_text: str,
        model_id: str,
        max_tokens: int,
    ) -> str:
        digest = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()[:16]
        return f"{model_id}:{max_tokens}:{digest}"

    def _profile_cache_set(self, key: str, value: str) -> None:
        self._profile_compress_cache[key] = value
        if len(self._profile_compress_cache) <= 64:
            return
        oldest_key = next(iter(self._profile_compress_cache))
        self._profile_compress_cache.pop(oldest_key, None)

    async def _compress_profile_with_llm(
        self,
        *,
        router: ModelRouter,
        model_id: str,
        profile_text: str,
        max_tokens: int,
    ) -> str:
        """Compress L1 profile while preserving actionable rules."""
        sys_prompt = (
            "你是记忆注入压缩器。请把输入的长期记忆压缩成更短版本。"
            "保留所有可执行规则、约束、优先级和关键身份信息。"
            "禁止新增事实。输出纯文本，不要 JSON。"
        )
        user_prompt = (
            f"目标上限约 {max_tokens} tokens。\n"
            "输入记忆如下：\n"
            f"{profile_text}\n\n"
            "请输出压缩版。"
        )
        try:
            response = await router.chat(
                model_id,
                [
                    Message(role="system", content=sys_prompt),
                    Message(role="user", content=user_prompt),
                ],
            )
        except Exception:
            return ""
        txt = response.content.strip()
        if not txt:
            return ""
        if _est_tokens(txt) <= max_tokens:
            return txt
        return _truncate_to_tokens(txt, max_tokens)

    async def upsert_profile_from_capture(
        self,
        content: str,
        *,
        router: ModelRouter,
        model_id: str,
        max_tokens: int = 1600,
        keep_profile_versions: int = 3,
    ) -> bool:
        """Fallback profile update: append captured rule into L1 profile."""
        text = content.strip()
        if len(text) < 6:
            return False

        sys_prompt = (
            "你是长期画像规则分类器。"
            "判断新输入是否是可执行的长期规则。"
            "若是，输出 JSON: {accept:boolean, rule:string}。"
            "rule 必须精炼可执行，不超过 80 字。"
        )
        user_prompt = f"输入：{text}"
        try:
            resp = await router.chat(
                model_id,
                [
                    Message(role="system", content=sys_prompt),
                    Message(role="user", content=user_prompt),
                ],
            )
        except Exception:
            return False

        obj = _extract_json_block(resp.content)
        if obj is None:
            return False
        accept = bool(obj.get("accept", False))
        rule = str(obj.get("rule", "")).strip()
        if not accept or not rule:
            return False

        recent = await self._store.list_recent(limit=300)
        latest_l1 = next(
            (
                e
                for e in recent
                if _is_profile_entry(e) and any(t == "level:L1" for t in e.tags)
            ),
            None,
        )
        target = latest_l1
        old_text = target.content.strip() if target is not None else ""
        if old_text and _normalize_text(rule) in _normalize_text(old_text):
            return False
        merged = f"{old_text}\n- {rule}".strip() if old_text else rule
        merged = _truncate_to_tokens(merged, max_tokens)

        await self._store.add(
            merged,
            source="memory_fallback",
            tags=["memory_profile", "level:L1", "curated", "fallback_rule"],
        )
        await self._prune_profiles(keep_profile_versions=max(1, keep_profile_versions))
        return True

    async def memorize(
        self, content: str, source: str, tags: list[str] | None = None
    ) -> MemoryEntry:
        return await self._store.add(content, source, tags or [])

    async def compact(self, messages: list[dict[str, str]], source: str) -> str:
        summary = await self._summarizer.summarize(messages)
        facts = await self._summarizer.extract_facts(messages)
        for fact in facts:
            await self._store.add(fact, source, tags=["compact"])
        return summary

    async def auto_capture_user_message(
        self,
        content: str,
        *,
        source: str,
        mode: Literal["conservative", "balanced", "aggressive"] = "balanced",
        cooldown_seconds: int = 180,
        max_per_hour: int = 12,
        batch_size: int = 3,
        merge_window_seconds: int = 120,
    ) -> bool:
        """Capture user message into long-term memory with deterministic guards."""
        text = content.strip()
        if len(text) < 6 or len(text) > 600:
            return False
        if text.startswith(("/", "{", "[")):
            return False
        if not _matches_capture_signal(text, mode):
            return False

        now = datetime.now(UTC)
        recent = await self._store.list_recent(limit=200)
        pending_norms = {
            _normalize_text(x) for x in self._pending_by_source.get(source, []) if x.strip()
        }

        normalized = _normalize_text(text)
        if any(_normalize_text(e.content) == normalized for e in recent):
            return False
        if normalized in pending_norms:
            return False

        same_source = [e for e in recent if e.source == source]
        if same_source:
            latest = same_source[0]
            elapsed = (now - latest.created_at).total_seconds()
            if elapsed < max(0, cooldown_seconds):
                return False

            recent_hour = [
                e for e in same_source
                if (now - e.created_at).total_seconds() <= 3600
            ]
            if len(recent_hour) >= max(1, max_per_hour):
                return False

        facts = await self._summarizer.extract_facts([{"role": "user", "content": text}])
        candidates = facts or [text]
        pending = 0
        seen_new: set[str] = set()
        for fact in candidates:
            norm = _normalize_text(fact)
            if not norm or norm in seen_new:
                continue
            if any(_normalize_text(e.content) == norm for e in recent):
                continue
            if norm in pending_norms:
                continue
            seen_new.add(norm)
            self._pending_by_source[source].append(fact)
            self._pending_since.setdefault(source, now)
            pending += 1

        if pending <= 0:
            return False

        await self.flush_capture_buffer(
            source=source,
            force=_is_force_flush_capture(text),
            batch_size=batch_size,
            merge_window_seconds=merge_window_seconds,
            mode=mode,
        )
        return True

    async def flush_capture_buffer(  # noqa: PLR0913
        self,
        *,
        source: str | None = None,
        force: bool = False,
        batch_size: int = 3,
        merge_window_seconds: int = 120,
        mode: Literal["conservative", "balanced", "aggressive"] = "balanced",
    ) -> int:
        """Flush pending capture buffer to store. Returns number of written entries."""
        now = datetime.now(UTC)
        sources = [source] if source else list(self._pending_by_source.keys())
        written = 0
        for src in sources:
            items = self._pending_by_source.get(src, [])
            if not items:
                continue
            since = self._pending_since.get(src, now)
            should_flush = force or len(items) >= max(1, batch_size)
            if not should_flush and (now - since).total_seconds() >= max(1, merge_window_seconds):
                should_flush = True
            if not should_flush:
                continue

            dedup: list[str] = []
            seen: set[str] = set()
            for item in items:
                norm = _normalize_text(item)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                dedup.append(item)

            for item in dedup:
                await self._store.add(
                    item,
                    source=src,
                    tags=["auto_capture", f"mode:{mode}", "batched"],
                )
                written += 1
            self._pending_by_source[src].clear()
            self._pending_since.pop(src, None)
            self._pending_last_flush[src] = now
        return written

    async def organize_if_needed(  # noqa: PLR0913
        self,
        *,
        router: ModelRouter,
        model_id: str,
        organizer_min_new_entries: int = 5,
        organizer_interval_seconds: int = 900,
        organizer_max_raw_window: int = 120,
        keep_profile_versions: int = 3,
        max_raw_entries: int = 800,
    ) -> bool:
        """Use LLM to organize raw memory into L1 profile summary."""
        await self.flush_capture_buffer(force=True)
        recent = await self._store.list_recent(limit=max(organizer_max_raw_window + 50, 200))
        now = datetime.now(UTC)
        profiles = [
            e
            for e in recent
            if _is_profile_entry(e) and any(t == "level:L1" for t in e.tags)
        ]
        latest_profile = profiles[0] if profiles else None

        if latest_profile is not None:
            elapsed = (now - latest_profile.created_at).total_seconds()
            if elapsed < max(0, organizer_interval_seconds):
                return False

        raw_entries = [e for e in recent if _is_raw_entry(e)]
        if latest_profile is not None:
            raw_entries = [e for e in raw_entries if e.created_at > latest_profile.created_at]
        if len(raw_entries) < max(1, organizer_min_new_entries):
            return False

        raw_entries = sorted(raw_entries, key=lambda x: x.created_at)[-organizer_max_raw_window:]
        old_l1 = latest_profile.content if latest_profile is not None else ""
        raw_text = "\n".join(f"- {e.content}" for e in raw_entries)

        sys_prompt = (
            "你是记忆整理器。"
            "请把用户原始记忆整理为稳定、去重、可长期引用的内容。"
            "输出必须是 JSON 对象，字段为：l1, style_directive, keep, drop。"
            "其中 l1 最多 800 字，"
            "style_directive 是可选的全局回复风格指令（最多120字，纯行为约束，不含任务事实）。"
            "keep/drop 是字符串数组（用于解释保留/丢弃依据，简短即可）。"
            "要求：输出去重后的统一画像，不要重复表达。"
        )
        user_prompt = (
            "历史 L1 画像（可能为空）：\n"
            f"{old_l1 or '(empty)'}\n\n"
            "新原始记忆候选：\n"
            f"{raw_text}\n\n"
            "请返回 JSON。"
        )

        response = await router.chat(
            model_id,
            [Message(role="system", content=sys_prompt), Message(role="user", content=user_prompt)],
        )
        obj = _extract_json_block(response.content)
        if obj is None:
            log.warning("memory.organizer_invalid_json", model=model_id)
            return False

        l1 = str(obj.get("l1", "")).strip()
        style_directive = str(obj.get("style_directive", "")).strip()
        if not l1:
            log.warning("memory.organizer_missing_fields", model=model_id)
            return False

        await self._store.add(
            l1[:8000],
            source="memory_organizer",
            tags=["memory_profile", "level:L1", "curated"],
        )
        if style_directive:
            style_hits = sum(_style_signal_hits(e.content) for e in raw_entries)
            if style_hits >= 2:
                await self._store.add(
                    style_directive[:300],
                    source="memory_organizer",
                    tags=["memory_profile", "style:global", "curated"],
                )

        await self._prune_profiles(keep_profile_versions=max(1, keep_profile_versions))
        await self._prune_raw(max_raw_entries=max(50, max_raw_entries))
        await self._prune_style_profiles(keep_versions=max(3, keep_profile_versions))
        return True

    async def get_global_style_directive(self) -> str:
        recent = await self._store.list_recent(limit=200)
        style = next((e for e in recent if _is_style_profile_entry(e)), None)
        if style is None:
            return ""
        return style.content.strip()

    async def set_global_style_directive(
        self,
        directive: str,
        *,
        source: str = "manual",
    ) -> bool:
        text = directive.strip()
        if not text:
            return False
        current = await self.get_global_style_directive()
        if current and _normalize_text(current) == _normalize_text(text):
            return False
        await self._store.add(
            text[:300],
            source=source,
            tags=["memory_profile", "style:global", "curated", "manual_override"],
        )
        await self._prune_style_profiles(keep_versions=5)
        return True

    async def clear_global_style_directive(self) -> int:
        entries = await self._store.list_recent(limit=500)
        styles = [e for e in entries if _is_style_profile_entry(e)]
        removed = 0
        for entry in styles:
            ok = await self._store.delete(entry.id)
            if ok:
                removed += 1
        return removed

    async def _prune_profiles(self, keep_profile_versions: int) -> None:
        entries = await self._store.list_recent(limit=1000)
        # Keep only recent L1 profiles; legacy L0 profiles are removed.
        l1_items = [
            e for e in entries
            if _is_profile_entry(e) and any(t == "level:L1" for t in e.tags)
        ]
        for stale in l1_items[keep_profile_versions:]:
            await self._store.delete(stale.id)

        l0_items = [
            e for e in entries
            if _is_profile_entry(e) and any(t == "level:L0" for t in e.tags)
        ]
        for old in l0_items:
            await self._store.delete(old.id)

    async def _prune_raw(self, max_raw_entries: int) -> None:
        entries = await self._store.list_recent(limit=max_raw_entries * 4)
        raws = [e for e in entries if _is_raw_entry(e)]
        for stale in raws[max_raw_entries:]:
            await self._store.delete(stale.id)

    async def _prune_style_profiles(self, keep_versions: int) -> None:
        entries = await self._store.list_recent(limit=500)
        styles = [e for e in entries if _is_style_profile_entry(e)]
        for stale in styles[max(1, keep_versions):]:
            await self._store.delete(stale.id)
