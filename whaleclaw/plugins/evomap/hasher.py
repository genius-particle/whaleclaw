"""Content-addressable asset ID computation."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class AssetHasher:
    """Compute content-addressable IDs for GEP-A2A assets."""

    @staticmethod
    def compute_asset_id(asset: dict[str, Any]) -> str:
        """
        Compute asset ID: remove asset_id, canonical JSON, SHA256.
        Returns 'sha256:<hex>'.
        """
        copy = {k: v for k, v in asset.items() if k != "asset_id"}
        canonical = json.dumps(copy, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
