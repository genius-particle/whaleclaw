"""GEP-A2A protocol HTTP client."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import httpx


class A2AClient:
    """GEP-A2A protocol HTTP client."""

    def __init__(self, hub_url: str, sender_id: str, node_secret: str | None = None) -> None:
        self._hub_url = hub_url.rstrip("/")
        self._sender_id = sender_id
        self._node_secret: str | None = node_secret
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def node_secret(self) -> str | None:
        return self._node_secret

    @node_secret.setter
    def node_secret(self, value: str | None) -> None:
        self._node_secret = value

    def _build_envelope(self, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Build GEP-A2A envelope with required fields."""
        return {
            "protocol": "gep-a2a",
            "protocol_version": "1.0.0",
            "message_type": message_type,
            "message_id": str(uuid.uuid4()),
            "sender_id": self._sender_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": payload,
        }

    def _url(self, endpoint: str) -> str:
        return f"{self._hub_url}/a2a/{endpoint}"

    def _auth_headers(self) -> dict[str, str]:
        if self._node_secret:
            return {"Authorization": f"Bearer {self._node_secret}"}
        return {}

    async def hello(self) -> dict[str, Any]:
        """POST /a2a/hello — register/refresh node.

        Extracts and stores node_secret from response for subsequent requests.
        """
        envelope = self._build_envelope("hello", {})
        resp = await self._client.post(self._url("hello"), json=envelope)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        payload: dict[str, Any] = data.get("payload") or {}
        secret: Any = payload.get("node_secret")
        if secret and isinstance(secret, str):
            self._node_secret = secret
        return data

    async def publish(self, assets: list[dict[str, Any]]) -> dict[str, Any]:
        """POST /a2a/publish — publish Gene+Capsule+EvolutionEvent bundle."""
        envelope = self._build_envelope("publish", {"assets": assets})
        resp = await self._client.post(
            self._url("publish"), json=envelope, headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def fetch(
        self,
        asset_type: str = "Capsule",
        include_tasks: bool = False,
    ) -> dict[str, Any]:
        """POST /a2a/fetch — fetch promoted assets and optional tasks."""
        envelope = self._build_envelope(
            "fetch",
            {"asset_type": asset_type, "include_tasks": include_tasks},
        )
        resp = await self._client.post(
            self._url("fetch"), json=envelope, headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def report(self, target_asset_id: str, report: dict[str, Any]) -> None:
        """POST /a2a/report — submit validation report."""
        envelope = self._build_envelope(
            "report",
            {"target_asset_id": target_asset_id, "report": report},
        )
        resp = await self._client.post(
            self._url("report"), json=envelope, headers=self._auth_headers(),
        )
        resp.raise_for_status()

    async def claim_task(self, task_id: str) -> dict[str, Any]:
        """POST /a2a/claim — claim a bounty task."""
        envelope = self._build_envelope("claim", {"task_id": task_id})
        resp = await self._client.post(
            self._url("claim"), json=envelope, headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def complete_task(self, task_id: str, asset_id: str) -> dict[str, Any]:
        """POST /a2a/complete — submit task completion."""
        envelope = self._build_envelope("complete", {"task_id": task_id, "asset_id": asset_id})
        resp = await self._client.post(
            self._url("complete"), json=envelope, headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def my_tasks(self) -> dict[str, Any]:
        """POST /a2a/my_tasks — get claimed tasks."""
        envelope = self._build_envelope("my_tasks", {})
        resp = await self._client.post(
            self._url("my_tasks"), json=envelope, headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
