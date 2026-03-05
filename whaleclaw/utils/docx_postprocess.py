"""Post-processing for generated DOCX files.

Fixes images that are stretched/distorted by replacing their blobs
with face-aware cropped versions, similar to pptx_postprocess.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_RATIO_DIFF_THRESHOLD = 0.08


def fix_docx(path: str | Path) -> bool:
    """Fix distorted images in a .docx file in-place.

    Detects images whose pixel aspect ratio doesn't match the display
    aspect ratio, then replaces the image blob with a face-aware
    cropped version. Returns True if the file was modified.
    """
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".docx":
        return False

    try:
        from docx import Document
        from docx.shared import Emu
    except ImportError:
        return False

    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    from whaleclaw.utils.image_crop import detect_face_info, smart_crop_box

    try:
        doc = Document(str(path))
    except Exception:
        log.warning("docx_postprocess: failed to open %s", path)
        return False

    modified = False

    for rel_id, rel in doc.part.rels.items():
        if "image" not in rel.reltype:
            continue

        image_part = rel.target_part
        try:
            blob = image_part.blob
        except Exception:
            continue

        arr = np.frombuffer(blob, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        img_ratio = iw / ih

        display_sizes = _find_display_sizes_for_rel(doc, rel_id)
        if not display_sizes:
            continue

        best_w, best_h = max(display_sizes, key=lambda s: s[0] * s[1])
        if best_w <= 0 or best_h <= 0:
            continue
        box_ratio = best_w / best_h

        if abs(img_ratio - box_ratio) < _RATIO_DIFF_THRESHOLD:
            continue

        fi = detect_face_info(None, cv_img=img)
        x0, y0, x1, y1 = smart_crop_box(iw, ih, best_w, best_h, face_info=fi)

        cropped = img[y0:y1, x0:x1]
        success, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            continue

        try:
            image_part._blob = bytes(buf)  # type: ignore[attr-defined]
            modified = True
        except Exception:
            continue

    if modified:
        try:
            doc.save(str(path))
        except Exception:
            log.warning("docx_postprocess: failed to save %s", path)
            return False

    return modified


def _find_display_sizes_for_rel(
    doc: object,
    rel_id: str,
) -> list[tuple[int, int]]:
    """Find all (width_emu, height_emu) for images referencing a given rId."""
    sizes: list[tuple[int, int]] = []

    ns = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    }

    body = doc.element.body  # type: ignore[attr-defined]

    for blip in body.findall(f".//a:blip[@r:embed='{rel_id}']", ns):
        drawing = blip.getparent()
        while drawing is not None:
            tag = drawing.tag.split("}")[-1] if "}" in drawing.tag else drawing.tag
            if tag in ("inline", "anchor"):
                ext = drawing.find(".//a:ext", ns)
                if ext is not None:
                    cx = int(ext.get("cx", "0"))
                    cy = int(ext.get("cy", "0"))
                    if cx > 0 and cy > 0:
                        sizes.append((cx, cy))
                break
            drawing = drawing.getparent()

    return sizes
