"""core/monitor_geometry.py - which monitor a point is on, and where a point
in a screenshot lands on the real desktop (2026-10-02, S4).

WHY THIS EXISTS
===============
Live: after JARVIS opened a page on the MIDDLE monitor, local vision named
"Image #4 (TOP monitor)" for it in one round and "the MIDDLE monitor" in the
next, and a description click then landed at (-2325, 1165) - on the LEFT
monitor. Three things were loose:

  * the click captured the whole 7680x2880 virtual desktop (four monitors,
    one of them above the others, two empty corners), shrunk to 1568 px, so
    the target was a few pixels wide and any monitor's content could win;
  * the cloud vision call labelled each image "--- LEFT monitor ---" while the
    local call listed "Image #1 = LEFT monitor" once, up front - two schemes,
    and the local model answered with image numbers it had to map back;
  * the point -> screen arithmetic lived inline in find_click_target, with no
    test of a negative-origin layout.

This module holds the arithmetic and the labels; the monolith pins the click
(and a look at "the page") to the monitor JARVIS opened the page on.

Monitors are ``{name: (x, y, w, h)}`` in logical (pyautogui) pixels, the
core.config.MONITORS shape. Pure stdlib; never raises.
"""
from __future__ import annotations

import re
from typing import Optional


def _rects(monitors):
    out = []
    try:
        for name, rect in dict(monitors or {}).items():
            try:
                x, y, w, h = (int(v) for v in tuple(rect)[:4])
            except Exception:
                continue
            if w > 0 and h > 0:
                out.append((str(name), x, y, w, h))
    except Exception:
        return []
    return out


def monitor_at(x, y, monitors) -> Optional[str]:
    """The monitor containing the point (x, y), else None (a gap between
    monitors, or off the desktop). Left/top edges are inside, right/bottom
    edges are the next monitor's."""
    try:
        px, py = float(x), float(y)
    except Exception:
        return None
    for name, mx, my, mw, mh in _rects(monitors):
        if mx <= px < mx + mw and my <= py < my + mh:
            return name
    return None


def monitor_for_rect(left, top, width, height, monitors) -> Optional[str]:
    """The monitor holding the CENTRE of a window rect, else None."""
    try:
        return monitor_at(float(left) + float(width) / 2.0,
                          float(top) + float(height) / 2.0, monitors)
    except Exception:
        return None


def virtual_bounds(monitors, default=(0, 0, 2560, 1440)) -> tuple:
    """(x, y, w, h) of the box around every monitor; ``default`` when there
    are none."""
    rects = _rects(monitors)
    if not rects:
        return tuple(default)
    x0 = min(r[1] for r in rects)
    y0 = min(r[2] for r in rects)
    x1 = max(r[1] + r[3] for r in rects)
    y1 = max(r[2] + r[4] for r in rects)
    return x0, y0, x1 - x0, y1 - y0


def scale_point(px, py, from_size, to_size) -> tuple:
    """(px, py) in an image of ``from_size`` (w, h) -> the same spot in an
    image of ``to_size`` (the downscaled pass-1 shot -> the full-res one)."""
    fw, fh = from_size
    tw, th = to_size
    sx = (tw / fw) if fw else 1.0
    sy = (th / fh) if fh else 1.0
    return int(px * sx), int(py * sy)


def image_point_to_screen(px, py, image_size, region) -> tuple:
    """A point in a screenshot -> absolute logical desktop coordinates.

    ``image_size`` is the (w, h) of the image the point is in; ``region`` is
    the (x, y, w, h) LOGICAL rect that image photographs (one monitor, or the
    virtual desktop). The point is mapped proportionally onto the region and
    the region's origin - negative for a monitor left of or above the
    primary - is added:  abs = origin + p * (logical / image)."""
    iw, ih = image_size
    rx, ry, rw, rh = region
    sx = (rw / iw) if iw else 1.0
    sy = (rh / ih) if ih else 1.0
    return int(rx + px * sx), int(ry + py * sy)


# ── One label scheme for several monitor images (both vision routes) ────────
def monitor_image_labels(names) -> list:
    """["Image 1 = LEFT monitor", ...] in the order the images are sent."""
    return [f"Image {i + 1} = {str(n).upper()} monitor"
            for i, n in enumerate(list(names or ()))]


def multi_monitor_intro(names) -> str:
    """The text that goes in front of a multi-monitor vision question - the
    same on the cloud and the local route."""
    labels = "\n".join(monitor_image_labels(names))
    return (
        f"You are looking at {len(list(names or ()))} monitors at once, one "
        f"image per monitor, in this order:\n{labels}\n"
        "When you answer, name the monitor by its NAME (for example 'the "
        "MIDDLE monitor'), not by its image number. If something is on only "
        "one monitor, name that monitor. If the question doesn't apply to a "
        "monitor, skip it.")


_IMAGE_REF_RE = re.compile(
    r"\bImage\s*#?\s*(\d+)\b(?:\s*\((?:the\s+)?([A-Za-z]+)(?:\s+monitor)?\))?",
    re.IGNORECASE)


def canonical_monitor_refs(answer, names) -> str:
    """``answer`` with every "Image #N" / "Image N (TOP)" reference rewritten
    to the monitor image N really is ("the MIDDLE monitor"), so the follow-up
    round sees one naming. A number outside the list is left as it is.
    Never raises."""
    try:
        order = [str(n).upper() for n in list(names or ())]
        if not order or not answer:
            return str(answer or "")

        def _sub(m):
            i = int(m.group(1)) - 1
            if not 0 <= i < len(order):
                return m.group(0)
            return f"the {order[i]} monitor"
        out = _IMAGE_REF_RE.sub(_sub, str(answer))
        # "the the MIDDLE monitor monitor" from "the Image #2 monitor".
        out = re.sub(r"\bthe\s+the\b", "the", out, flags=re.IGNORECASE)
        out = re.sub(r"\b(monitor)\s+monitor\b", r"\1", out,
                     flags=re.IGNORECASE)
        return out
    except Exception:
        return str(answer or "")


# Monitor words in the owner's own request ("on the left monitor", "the top
# screen", "main display").
_NAMED_MONITOR_RE = re.compile(
    r"\b(left|right|top|middle|center|centre|main|primary|bottom|upper|"
    r"lower)\s+(?:monitor|screen|display)\b", re.IGNORECASE)


def monitor_named_in(text, monitors) -> Optional[str]:
    """The MONITORS key the words name ("the left monitor", "my main
    screen"), else None. "main" / "primary" / "center" = the monitor at the
    origin."""
    try:
        m = _NAMED_MONITOR_RE.search(str(text or ""))
        if not m:
            return None
        word = m.group(1).lower()
        keys = {r[0].lower(): r[0] for r in _rects(monitors)}
        if word in keys:
            return keys[word]
        if word in ("main", "primary", "center", "centre"):
            for name, x, y, _w, _h in _rects(monitors):
                if (x, y) == (0, 0):
                    return name
        if word in ("upper",) and "top" in keys:
            return keys["top"]
    except Exception:
        return None
    return None
