"""Session group-based L2/L1/L0 compression with DB-backed cache."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import TypedDict
from uuid import uuid4

from structlog.contextvars import bind_contextvars, reset_contextvars

from whaleclaw.providers.base import Message
from whaleclaw.providers.router import ModelRouter
from whaleclaw.sessions.store import SessionStore
from whaleclaw.utils.log import get_logger

RECENT_L2_GROUPS = 3
RECENT_RAW_PREV_GROUPS = 2
L1_GROUPS = 7
L0_GROUPS = 15
MAX_WINDOW_GROUPS = 25
CONTENT_BUDGET = 1600
BUILD_CONCURRENCY = 6
log = get_logger(__name__)


class PrewarmStats(TypedDict):
    total_groups: int
    processed_groups: int
    cache_hits: int
    generated: int


def _estimate_tokens(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = len(text) - cjk
    return max(1, int(cjk / 1.5 + latin / 4))


def _group_tokens(group: list[Message]) -> int:
    return sum(_estimate_tokens(m.content) for m in group)


def _flatten(groups: list[list[Message]]) -> list[Message]:
    out: list[Message] = []
    for g in groups:
        out.extend(g)
    return out


def _group_by_turn(messages: list[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    current: list[Message] = []
    for msg in messages:
        if msg.role == "user":
            if current:
                groups.append(current)
            current = [msg]
            continue
        if not current:
            current = [msg]
        else:
            current.append(msg)
    if current:
        groups.append(current)
    return groups


def _group_text(group: list[Message]) -> str:
    lines: list[str] = []
    for msg in group:
        role = "用户" if msg.role == "user" else "助手" if msg.role == "assistant" else msg.role
        lines.append(f"[{role}] {msg.content.strip()}")
    return "\n".join(lines)


def _hash_group(group: list[Message]) -> str:
    raw = "||".join(f"{m.role}:{m.content}" for m in group)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _clip_text(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    char_cap = max_tokens * 3
    if len(text) <= char_cap:
        return text
    return text[:char_cap].rstrip()


def _extract_latest_user_text(group: list[Message]) -> str:
    for msg in reversed(group):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


def _build_history_summary_block(items: list[tuple[int, str]]) -> str:
    if not items:
        return ""
    start = items[0][0]
    end = items[-1][0]
    lines = [f"【历史摘要（第{start}~{end}轮）】"]
    for idx, content in items:
        txt = " ".join(content.split())
        lines.append(f"- 第{idx}轮: {txt}")
    return "\n".join(lines)


def _build_recent_raw_block(items: list[tuple[int, list[Message]]]) -> str:
    if not items:
        return ""
    lines = ["【最近对话原文（最近2~3轮）】"]
    for idx, group in items:
        lines.append(f"第{idx}轮:")
        lines.append(_group_text(group))
    return "\n".join(lines)


def _build_task_status_block(current_group_idx: int, current_group: list[Message]) -> str:
    latest_user = _extract_latest_user_text(current_group)
    status = (
        f"待处理用户请求：{latest_user}"
        if latest_user
        else "待处理：继续基于当前轮上下文生成下一步动作。"
    )
    return "\n".join([
        "【当前任务状态】",
        f"当前轮次: 第{current_group_idx}轮",
        status,
    ])


def _build_done_summary_zh(
    *,
    window_groups: int,
    compressed_groups: int,
    cache_hits: int,
    fallback_used: int,
    scheduled: int,
    elapsed_ms: int,
) -> str:
    return (
        f"窗口{window_groups}组，需压缩{compressed_groups}组；"
        f"缓存命中{cache_hits}组，回退{fallback_used}组，"
        f"后台排队{scheduled}组，耗时{elapsed_ms}ms。"
    )


@dataclass
class _WindowItem:
    group_idx: int
    level: str
    group: list[Message]


@dataclass
class _PendingBuild:
    item: _WindowItem
    source_hash: str
    router: ModelRouter
    model_id: str


class SessionGroupCompressor:
    """LLM semantic compression for session groups with persistent cache."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self._pending_by_session: dict[str, dict[str, _PendingBuild]] = {}
        self._inflight_keys: set[str] = set()
        self._drain_tasks: dict[str, asyncio.Task[None]] = {}

    async def prewarm_session(
        self,
        *,
        session_id: str,
        messages: list[Message],
        router: ModelRouter,
        model_id: str,
    ) -> PrewarmStats:
        if not model_id.strip():
            return {
                "total_groups": 0,
                "processed_groups": 0,
                "cache_hits": 0,
                "generated": 0,
            }
        plan = [x for x in self._window_plan(messages) if x.level != "L2"]
        total = len(plan)
        if total == 0:
            return {
                "total_groups": 0,
                "processed_groups": 0,
                "cache_hits": 0,
                "generated": 0,
            }

        cache_hits = 0
        pending: list[_WindowItem] = []
        for item in plan:
            source_hash = _hash_group(item.group)
            cached = await self._store.get_group_compression(
                session_id=session_id,
                group_idx=item.group_idx,
                level=item.level,
                source_hash=source_hash,
            )
            if cached:
                cache_hits += 1
            else:
                pending.append(item)

        need_generate = len(pending)
        if need_generate == 0:
            log.info(
                "compressor.prewarm_skip",
                session_id=session_id,
                cache_hits=cache_hits,
                total=total,
            )
            return {
                "total_groups": 0,
                "processed_groups": 0,
                "cache_hits": cache_hits,
                "generated": 0,
            }

        log.info(
            "compressor.prewarm_start",
            session_id=session_id,
            total=need_generate,
            cache_hits=cache_hits,
        )
        generated = 0
        for idx, item in enumerate(pending, start=1):
            await self._get_or_build_group(
                session_id=session_id,
                item=item,
                router=router,
                model_id=model_id,
            )
            generated += 1
            log.info(
                "compressor.prewarm_progress",
                session_id=session_id,
                progress=f"{idx}/{need_generate}",
                done=idx,
                total=need_generate,
                level=item.level,
                group_idx=item.group_idx,
            )
        return {
            "total_groups": need_generate,
            "processed_groups": generated,
            "cache_hits": cache_hits,
            "generated": generated,
        }

    async def build_window_messages(
        self,
        *,
        session_id: str,
        messages: list[Message],
        router: ModelRouter,
        model_id: str,
    ) -> list[Message]:
        t0 = time.monotonic()
        groups = _group_by_turn(messages)
        if not groups:
            return messages

        if len(groups) <= MAX_WINDOW_GROUPS and _group_tokens(groups[-1]) > CONTENT_BUDGET:
            return groups[-1]

        plan = self._window_plan(messages)
        rendered: list[list[Message]] = []
        compressed_items = [(idx, item) for idx, item in enumerate(plan) if item.level != "L2"]
        compressed_results: dict[int, tuple[str, bool]] = {}
        compressed_input_tokens_total = sum(
            _group_tokens(item.group) for _, item in compressed_items
        )
        cache_hits = 0
        generated = 0
        scheduled = 0
        fallback_used = 0
        generated_input_tokens = 0
        generated_output_tokens = 0
        compressed_output_tokens = 0
        if compressed_items:
            async def _resolve_cached(
                i: int, item: _WindowItem,
            ) -> tuple[int, _WindowItem, str, str | None]:
                source_hash = _hash_group(item.group)
                cached = await self._store.get_group_compression(
                    session_id=session_id,
                    group_idx=item.group_idx,
                    level=item.level,
                    source_hash=source_hash,
                )
                return i, item, source_hash, cached

            resolved = await asyncio.gather(
                *(_resolve_cached(i, item) for i, item in compressed_items)
            )
            for i, item, source_hash, cached in resolved:
                if cached is not None:
                    compressed_results[i] = (cached, True)
                    cache_hits += 1
                    compressed_output_tokens += _estimate_tokens(cached)
                    continue

                fallback = self._fallback_group_text(item)
                compressed_results[i] = (fallback, False)
                fallback_used += 1
                compressed_output_tokens += _estimate_tokens(fallback)
                if self._schedule_background_group_build(
                    session_id=session_id,
                    item=item,
                    source_hash=source_hash,
                    router=router,
                    model_id=model_id,
                ):
                    scheduled += 1

        history_items: list[tuple[int, str]] = []
        recent_l2_items: list[tuple[int, list[Message]]] = []
        for idx, item in enumerate(plan):
            if item.level == "L2":
                recent_l2_items.append((item.group_idx, item.group))
                continue
            content, _ = compressed_results[idx]
            history_items.append((item.group_idx, content))

        if history_items:
            history_block = _build_history_summary_block(history_items)
            history_block = _clip_text(history_block, 520)
            if history_block:
                rendered.append([Message(role="assistant", content=history_block)])

        if not recent_l2_items:
            # Fallback: keep latest group raw to preserve current-turn semantics.
            recent_l2_items.append((plan[-1].group_idx, plan[-1].group))

        current_group_idx, current_group = recent_l2_items[-1]
        prev_recent = recent_l2_items[:-1]
        prev_recent = prev_recent[-RECENT_RAW_PREV_GROUPS:]
        if prev_recent:
            recent_block = _build_recent_raw_block(prev_recent)
            recent_block = _clip_text(recent_block, 320)
            if recent_block:
                rendered.append([Message(role="assistant", content=recent_block)])

        status_block = _build_task_status_block(current_group_idx, current_group)
        if status_block:
            rendered.append([Message(role="assistant", content=status_block)])

        rendered.append(current_group)

        pre_truncate_tokens = sum(_group_tokens(g) for g in rendered)
        out = self._truncate_group_atomic(rendered)
        final_output_tokens = sum(_group_tokens(g) for g in out)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        recent_l2_tokens = max(0, pre_truncate_tokens - compressed_output_tokens)
        truncated = final_output_tokens < pre_truncate_tokens
        summary_zh = _build_done_summary_zh(
            window_groups=len(plan),
            compressed_groups=len(compressed_items),
            cache_hits=cache_hits,
            fallback_used=fallback_used,
            scheduled=scheduled,
            elapsed_ms=elapsed_ms,
        )
        token_flow_zh = (
            f"token流：压缩前{compressed_input_tokens_total} -> 压缩后{compressed_output_tokens}，"
            f"组装前{pre_truncate_tokens} -> 最终{final_output_tokens}，"
            f"{'发生截断' if truncated else '未截断'}。"
        )
        log.info(
            "group_compressor.build_done",
            session_id=session_id,
            summary_zh=summary_zh,
            token_flow_zh=token_flow_zh,
        )
        log.debug(
            "group_compressor.build_metrics",
            session_id=session_id,
            step1_input_tokens_compressed_groups=compressed_input_tokens_total,
            step2_output_tokens_compressed_groups=compressed_output_tokens,
            step3_input_tokens_recent_l2_groups=recent_l2_tokens,
            step4_input_tokens_before_truncate=pre_truncate_tokens,
            step5_output_tokens_final=final_output_tokens,
            truncated=truncated,
            cache_hits=cache_hits,
            generated=generated,
            scheduled=scheduled,
            fallback_used=fallback_used,
            generated_input_tokens=generated_input_tokens,
            generated_output_tokens=generated_output_tokens,
            window_groups=len(plan),
            compressed_groups=len(compressed_items),
            input_messages=len(messages),
            output_messages=len(_flatten(out)),
            elapsed_ms=elapsed_ms,
        )
        return _flatten(out)

    async def shutdown(self) -> None:
        tasks = [t for t in self._drain_tasks.values() if not t.done()]
        if not tasks:
            self._pending_by_session.clear()
            self._inflight_keys.clear()
            self._drain_tasks.clear()
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._pending_by_session.clear()
        self._inflight_keys.clear()
        self._drain_tasks.clear()

    def _window_plan(self, messages: list[Message]) -> list[_WindowItem]:
        all_groups = _group_by_turn(messages)
        if not all_groups:
            return []
        total_groups = len(all_groups)
        groups = all_groups[-MAX_WINDOW_GROUPS:]
        start_idx = total_groups - len(groups) + 1
        recent_groups = all_groups[-RECENT_L2_GROUPS:]
        recent_over_budget = sum(_group_tokens(g) for g in recent_groups) > CONTENT_BUDGET
        items: list[_WindowItem] = []
        for offset, group in enumerate(groups):
            group_idx = start_idx + offset
            from_tail = total_groups - group_idx + 1
            if recent_over_budget and from_tail == 1:
                level = "L2"
            elif recent_over_budget and from_tail in {2, 3}:
                level = "L0"
            elif from_tail <= RECENT_L2_GROUPS:
                level = "L2"
            elif from_tail <= RECENT_L2_GROUPS + L1_GROUPS:
                level = "L1"
            else:
                level = "L0"
            items.append(_WindowItem(group_idx=group_idx, level=level, group=group))
        return items

    async def _get_or_build_group(
        self,
        *,
        session_id: str,
        item: _WindowItem,
        router: ModelRouter,
        model_id: str,
    ) -> tuple[str, bool]:
        source_hash = _hash_group(item.group)
        source_tokens = _group_tokens(item.group)
        cached = await self._store.get_group_compression(
            session_id=session_id,
            group_idx=item.group_idx,
            level=item.level,
            source_hash=source_hash,
        )
        if cached:
            return cached, True

        req_id = f"gc_{uuid4().hex[:10]}"
        t0 = time.monotonic()
        source_chars = sum(len(m.content) for m in item.group)
        log.debug(
            "group_compressor.group_start",
            request_id=req_id,
            session_id=session_id,
            group_idx=item.group_idx,
            level=item.level,
            source_tokens=source_tokens,
            source_chars=source_chars,
            message_count=len(item.group),
        )
        t_llm_start = time.monotonic()
        tokens = bind_contextvars(
            compress_request_id=req_id,
            compress_session_id=session_id,
            compress_group_idx=item.group_idx,
        )
        try:
            text = await self._compress_group(
                group=item.group,
                level=item.level,
                router=router,
                model_id=model_id,
            )
        finally:
            reset_contextvars(**tokens)
        llm_ms = int((time.monotonic() - t_llm_start) * 1000)
        t_store_start = time.monotonic()
        await self._store.upsert_group_compression(
            session_id=session_id,
            group_idx=item.group_idx,
            level=item.level,
            source_hash=source_hash,
            content=text,
        )
        store_ms = int((time.monotonic() - t_store_start) * 1000)
        log.debug(
            "group_compressor.group_done",
            request_id=req_id,
            session_id=session_id,
            group_idx=item.group_idx,
            level=item.level,
            source_tokens=source_tokens,
            output_tokens=_estimate_tokens(text),
            llm_ms=llm_ms,
            store_ms=store_ms,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
        )
        return text, False

    def _fallback_group_text(self, item: _WindowItem) -> str:
        budget = 220 if item.level == "L1" else 90
        source = _group_text(item.group).strip()
        if not source:
            return ""
        if _estimate_tokens(source) <= budget:
            return source
        return _clip_text(source, budget)

    def _schedule_background_group_build(
        self,
        *,
        session_id: str,
        item: _WindowItem,
        source_hash: str,
        router: ModelRouter,
        model_id: str,
    ) -> bool:
        model = model_id.strip()
        if not model:
            return False
        key = f"{session_id}:{item.group_idx}:{item.level}:{source_hash}"
        if key in self._inflight_keys:
            return False
        queue = self._pending_by_session.setdefault(session_id, {})
        if key in queue:
            return False
        queue[key] = _PendingBuild(
            item=item,
            source_hash=source_hash,
            router=router,
            model_id=model,
        )
        self._ensure_drain_loop(session_id=session_id)
        return True

    def _ensure_drain_loop(self, *, session_id: str) -> None:
        existing = self._drain_tasks.get(session_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._drain_session_queue(session_id=session_id),
            name=f"group-compress-drain-{session_id[:8]}",
        )
        self._drain_tasks[session_id] = task

    async def _drain_session_queue(self, *, session_id: str) -> None:
        try:
            while True:
                queue = self._pending_by_session.get(session_id)
                if not queue:
                    return
                batch: list[tuple[str, _PendingBuild]] = []
                while queue and len(batch) < BUILD_CONCURRENCY:
                    next_key = next(iter(queue))
                    pending = queue.pop(next_key)
                    self._inflight_keys.add(next_key)
                    batch.append((next_key, pending))
                if not batch:
                    return

                log.debug(
                    "group_compressor.bg_batch_start",
                    session_id=session_id,
                    batch_size=len(batch),
                    queued_remaining=len(queue),
                )
                results = await asyncio.gather(
                    *(self._run_background_job(session_id, key, pending) for key, pending in batch),
                    return_exceptions=True,
                )
                succeeded = sum(1 for ok in results if ok is True)
                failed = sum(1 for ok in results if ok is False)
                queue_after = self._pending_by_session.get(session_id, {})
                log.info(
                    "group_compressor.bg_batch_done",
                    session_id=session_id,
                    batch_size=len(batch),
                    succeeded=succeeded,
                    failed=failed,
                    queued_remaining=len(queue_after),
                    summary_zh=(
                        f"后台压缩批次完成：本批{len(batch)}组，"
                        f"成功{succeeded}组，失败{failed}组，"
                        f"剩余排队{len(queue_after)}组。"
                    ),
                )
                queue = self._pending_by_session.get(session_id)
                if queue is not None and not queue:
                    self._pending_by_session.pop(session_id, None)
        finally:
            current = asyncio.current_task()
            tracked = self._drain_tasks.get(session_id)
            if tracked is current:
                self._drain_tasks.pop(session_id, None)

    async def _run_background_job(
        self,
        session_id: str,
        key: str,
        pending: _PendingBuild,
    ) -> bool:
        t0 = time.monotonic()
        try:
            _, cache_hit = await self._get_or_build_group(
                session_id=session_id,
                item=pending.item,
                router=pending.router,
                model_id=pending.model_id,
            )
            log.debug(
                "group_compressor.bg_group_done",
                session_id=session_id,
                group_idx=pending.item.group_idx,
                level=pending.item.level,
                cache_hit=cache_hit,
                elapsed_ms=int((time.monotonic() - t0) * 1000),
            )
            return True
        except Exception as exc:
            log.debug(
                "group_compressor.bg_group_failed",
                session_id=session_id,
                group_idx=pending.item.group_idx,
                level=pending.item.level,
                error=str(exc),
            )
            return False
        finally:
            self._inflight_keys.discard(key)

    async def _compress_group(
        self,
        *,
        group: list[Message],
        level: str,
        router: ModelRouter,
        model_id: str,
    ) -> str:
        if not model_id.strip():
            return _group_text(group)[:900]
        source = _group_text(group)
        budget = 220 if level == "L1" else 90
        char_hint = 330 if level == "L1" else 140
        sys_prompt = (
            "你是会话压缩器。仅可基于输入原文抽取信息，禁止新增事实。"
            f"目标层级={level}。硬约束：输出必须控制在约{budget} tokens以内"
            f"（中文可参考不超过约{char_hint}字）。"
            "输出纯文本，不要 JSON，不要标题，不要解释。"
        )
        user_prompt = (
            f"目标层级: {level}\n"
            f"目标上限: 约{budget} tokens\n"
            f"中文字数参考上限: 约{char_hint} 字\n"
            "请保留任务目标、关键约束、关键结果与路径。\n\n"
            f"原文:\n{source}"
        )
        try:
            resp = await router.chat(
                model_id,
                [
                    Message(role="system", content=sys_prompt),
                    Message(role="user", content=user_prompt),
                ],
            )
            out = resp.content.strip()
            if out:
                # Enforce hard budget guard to prevent expansion on unstable outputs.
                if _estimate_tokens(out) > budget:
                    out = out[: budget * 3]
                return out
        except Exception:
            pass
        return source[: budget * 3]

    def _truncate_group_atomic(self, groups: list[list[Message]]) -> list[list[Message]]:
        if not groups:
            return groups
        total = sum(_group_tokens(g) for g in groups)
        if total <= CONTENT_BUDGET:
            return groups

        latest = groups[-1]
        if _group_tokens(latest) > CONTENT_BUDGET:
            return [latest]

        kept = list(groups)
        while len(kept) > 1 and sum(_group_tokens(g) for g in kept) > CONTENT_BUDGET:
            kept.pop(0)
        return kept
