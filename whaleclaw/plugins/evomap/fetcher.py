"""Asset fetching and local cache from EvoMap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from whaleclaw.config.paths import EVOMAP_DIR
from whaleclaw.plugins.evomap.client import A2AClient

_DEFAULT_CACHE_DIR = EVOMAP_DIR / "assets"


class AssetFetcher:
    """Fetch promoted assets from EvoMap and cache locally."""

    def __init__(self, client: A2AClient, cache_dir: Path | None = None) -> None:
        self._client = client
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, asset_id: str) -> Path:
        safe_id = asset_id.replace(":", "_").replace("/", "_")
        return self._cache_dir / f"{safe_id}.json"

    @staticmethod
    def _extract_assets(resp: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract asset list from hub response (handles both 'results' and 'assets' keys)."""
        payload: Any = resp.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        typed_payload = cast(dict[str, Any], payload)
        items: Any = (
            typed_payload.get("results")
            or typed_payload.get("assets")
            or resp.get("assets")
            or []
        )
        if not isinstance(items, list):
            return []
        return [a for a in items if isinstance(a, dict)]  # type: ignore[misc]

    def _cache_assets(self, assets: list[dict[str, Any]]) -> None:
        for asset in assets:
            aid = asset.get("asset_id") or asset.get("assetId")
            if aid:
                self._cache_path(str(aid)).write_text(
                    json.dumps(asset, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    async def fetch_promoted(self, asset_type: str = "Capsule") -> list[dict[str, Any]]:
        """Fetch latest promoted assets."""
        resp = await self._client.fetch(asset_type=asset_type, include_tasks=False)
        assets = self._extract_assets(resp)
        self._cache_assets(assets)
        return assets

    async def search_by_signals(self, signals: list[str]) -> list[dict[str, Any]]:
        """Search assets by signals (delegates to fetch with signal filter)."""
        resp = await self._client.fetch(
            asset_type="Capsule",
            include_tasks=False,
        )
        assets = self._extract_assets(resp)
        self._cache_assets(assets)
        clean = [s.strip().lower() for s in signals if s.strip()]
        result: list[tuple[int, dict[str, Any]]] = []
        for asset in assets:
            score = self._score_asset_match(asset, clean)
            if score > 0:
                result.append((score, asset))
        result.sort(key=lambda x: x[0], reverse=True)
        return [asset for _score, asset in result]

    def search_cached_by_signals(
        self,
        signals: list[str],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Search local cached assets by signal/summary text without network I/O."""
        clean = [s.strip().lower() for s in signals if s.strip()]
        if not clean:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for path in sorted(self._cache_dir.glob("*.json"), reverse=True):
            try:
                raw: Any = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            asset = cast(dict[str, Any], raw)

            score = self._score_asset_match(asset, clean)
            if score > 0:
                scored.append((score, asset))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [asset for _score, asset in scored[: max(1, limit)]]

    def list_cached_recent(self, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return recent cached assets without signal filtering."""
        results: list[dict[str, Any]] = []
        for path in sorted(self._cache_dir.glob("*.json"), reverse=True):
            try:
                raw: Any = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(raw, dict):
                results.append(raw)  # type: ignore[arg-type]
            if len(results) >= max(1, limit):
                break
        return results

    def get_cached(self, asset_id: str) -> dict[str, Any] | None:
        """Load asset from local cache."""
        path = self._cache_path(asset_id)
        if not path.exists():
            return None
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw  # type: ignore[return-value]
        return None

    @staticmethod
    def _extract_trigger_tokens(asset: dict[str, Any]) -> list[str]:
        """Extract trigger tokens from both array and comma-separated string formats."""
        tokens: list[str] = []
        raw_list: Any = asset.get("trigger", [])
        if isinstance(raw_list, list):
            for item in cast(list[Any], raw_list):
                s = str(item).strip().lower()
                if s:
                    tokens.append(s)
        raw_text: Any = asset.get("trigger_text", "")
        if isinstance(raw_text, str) and raw_text.strip():
            for part in raw_text.split(","):
                s = part.strip().lower()
                if s:
                    tokens.append(s)
        raw_sm: Any = asset.get("signals_match", [])
        if isinstance(raw_sm, list):
            for item in cast(list[Any], raw_sm):
                s = str(item).strip().lower()
                if s:
                    tokens.append(s)
        return tokens

    @staticmethod
    def _build_haystack(asset: dict[str, Any]) -> str:
        """Build searchable text from all relevant fields (top-level + nested payload)."""
        parts: list[str] = []
        for key in ("title", "summary", "description"):
            val: Any = asset.get(key, "")
            if val:
                parts.append(str(val))
        inner: Any = asset.get("payload")
        if isinstance(inner, dict):
            typed_inner = cast(dict[str, Any], inner)
            for ikey in ("summary", "capsule_summary", "description", "id"):
                ival: Any = typed_inner.get(ikey, "")
                if ival:
                    parts.append(str(ival))
        return " ".join(parts).lower()

    @staticmethod
    def _score_asset_match(asset: dict[str, Any], signals: list[str]) -> int:
        if not signals:
            return 0
        trigger = AssetFetcher._extract_trigger_tokens(asset)
        hay = AssetFetcher._build_haystack(asset)
        score = 0
        for sig in signals:
            if sig in trigger:
                score += 3
                continue
            if any(sig in t or t in sig for t in trigger):
                score += 2
                continue
            if sig in hay:
                score += 1
        return score
