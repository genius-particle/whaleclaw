"""Asset publishing to EvoMap network."""

from __future__ import annotations

import platform
from typing import Any, Literal

from whaleclaw.plugins.evomap.client import A2AClient
from whaleclaw.plugins.evomap.hasher import AssetHasher
from whaleclaw.plugins.evomap.identity import EvoMapIdentity
from whaleclaw.plugins.evomap.models import (
    BlastRadius,
    Capsule,
    EnvFingerprint,
    EvolutionEvent,
    Gene,
    Outcome,
)


class AssetPublisher:
    """Pack and publish Gene+Capsule+EvolutionEvent to EvoMap."""

    def __init__(self, client: A2AClient, identity: EvoMapIdentity) -> None:
        self._client = client
        self._identity = identity

    async def publish_fix(
        self,
        category: Literal["repair", "optimize", "innovate"],
        signals: list[str],
        gene_summary: str,
        capsule_summary: str,
        confidence: float,
        blast_radius: BlastRadius,
        outcome: Outcome,
        mutations_tried: int = 1,
        total_cycles: int = 1,
    ) -> dict[str, object]:
        """
        Build Gene+Capsule+EvolutionEvent bundle, compute asset_ids, publish.
        """
        arch = "arm64" if platform.machine() in ("arm64", "aarch64") else "x64"
        env = EnvFingerprint(platform=platform.system().lower(), arch=arch)

        gene_raw: dict[str, Any] = Gene(
            category=category,
            signals_match=signals,
            summary=gene_summary,
        ).model_dump()
        gene_id = AssetHasher.compute_asset_id(gene_raw)
        gene_raw["asset_id"] = gene_id

        capsule_raw: dict[str, Any] = Capsule(
            trigger=signals,
            gene=gene_id,
            summary=capsule_summary,
            confidence=confidence,
            blast_radius=blast_radius,
            outcome=outcome,
            env_fingerprint=env,
        ).model_dump()
        capsule_id = AssetHasher.compute_asset_id(capsule_raw)
        capsule_raw["asset_id"] = capsule_id

        event_raw: dict[str, Any] = EvolutionEvent(
            intent=category,
            capsule_id=capsule_id,
            genes_used=[gene_id],
            outcome=outcome,
            mutations_tried=mutations_tried,
            total_cycles=total_cycles,
        ).model_dump()
        event_id = AssetHasher.compute_asset_id(event_raw)
        event_raw["asset_id"] = event_id

        return await self._client.publish([gene_raw, capsule_raw, event_raw])
