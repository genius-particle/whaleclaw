"""Tests for WebSocket image persistence helpers."""

from __future__ import annotations

import base64
from pathlib import Path

from whaleclaw.gateway import ws as ws_mod
from whaleclaw.providers.base import ImageContent


def test_persist_ws_images_saves_files_and_returns_markdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ws_mod, "_WS_UPLOAD_DIR", tmp_path)
    images = [
        ImageContent(
            mime="image/png",
            data=base64.b64encode(b"png-bytes").decode("ascii"),
        ),
        ImageContent(
            mime="image/jpeg",
            data=base64.b64encode(b"jpeg-bytes").decode("ascii"),
        ),
    ]

    saved_paths, markdown = ws_mod._persist_ws_images("ws-test", images)  # noqa: SLF001

    assert len(saved_paths) == 2
    assert Path(saved_paths[0]).read_bytes() == b"png-bytes"
    assert Path(saved_paths[1]).read_bytes() == b"jpeg-bytes"
    assert "![WebChat图片1]" in markdown
    assert "![WebChat图片2]" in markdown
    assert saved_paths[0] in markdown
    assert saved_paths[1] in markdown
