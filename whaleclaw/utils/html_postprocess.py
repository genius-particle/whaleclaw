"""Post-processing for generated HTML files.

Fixes image display issues by ensuring images use object-fit: cover
instead of being stretched to explicit width/height.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_IMG_TAG_RE = re.compile(
    r"<img\b([^>]*)>",
    re.IGNORECASE | re.DOTALL,
)

_WIDTH_HEIGHT_ATTR_RE = re.compile(
    r'\b(width|height)\s*=\s*["\']?\d+[^"\'>\s]*["\']?',
    re.IGNORECASE,
)

_STYLE_ATTR_RE = re.compile(
    r'\bstyle\s*=\s*"([^"]*)"',
    re.IGNORECASE,
)

_OBJECT_FIT_STYLE = "object-fit: cover;"


def fix_html(path: str | Path) -> bool:
    """Fix image stretching in an HTML file in-place.

    For every <img> that has both width and height attributes (or inline
    style), injects ``object-fit: cover`` so the browser crops instead of
    distorting. Returns True if the file was modified.
    """
    path = Path(path)
    if not path.exists() or path.suffix.lower() not in (".html", ".htm"):
        return False

    try:
        text = path.read_text("utf-8", errors="ignore")
    except Exception:
        return False

    if not _IMG_TAG_RE.search(text):
        return False

    new_text = _IMG_TAG_RE.sub(_fix_img_tag, text)

    if new_text == text:
        return False

    try:
        path.write_text(new_text, "utf-8")
    except Exception:
        log.warning("html_postprocess: failed to write %s", path)
        return False

    return True


def _fix_img_tag(match: re.Match[str]) -> str:
    """Add object-fit:cover to an <img> tag."""
    full = match.group(0)
    attrs = match.group(1)

    has_w = bool(re.search(r"\bwidth\b", attrs, re.IGNORECASE))
    has_h = bool(re.search(r"\bheight\b", attrs, re.IGNORECASE))
    if not (has_w and has_h):
        return full

    if "object-fit" in attrs.lower():
        return full

    style_m = _STYLE_ATTR_RE.search(attrs)
    if style_m:
        old_style = style_m.group(1).rstrip("; ")
        new_style = f"{old_style}; {_OBJECT_FIT_STYLE}" if old_style else _OBJECT_FIT_STYLE
        new_attrs = attrs[: style_m.start(1)] + new_style + attrs[style_m.end(1) :]
    else:
        new_attrs = attrs.rstrip() + f' style="{_OBJECT_FIT_STYLE}"'

    return f"<img{new_attrs}>"
