"""core/vision_grounding.py - how an image goes to the local vision model,
and how its answer is read back (2026-10-05).

WHY THIS EXISTS
===============
Three things were loose on the live click path:

  * IMAGE SIZE. Ollama's bundled llama-server runs gemma with ``-ub 1024``
    and an image budget of 70-1120 tokens (server.log: image_min_pixels
    161280, image_max_pixels 2580480). Image tokens are about
    ceil(w/48) * ceil(h/48); an image over one 1024-token ubatch is the
    llama.cpp #21550 / #21461 crash class. fit_for_vlm keeps every image at
    <= VLM_MAX_IMAGE_TOKENS (960): a 2560x1440 monitor goes at 0.75
    (1920x1080, 920 tokens), a window crop usually native.
  * THE ANSWER. _query_vision_for_coords rejected JSON and gemma's native
    ``box_2d``, and because ask_vision prefixes every local answer with
    "[local-vision] " (15 characters) it accepted a bare pair only through
    its <=32-character trailing rule, and took a 0-1000 pair as pixels.
    parse_reply strips any leading "[...]" tag FIRST, accepts fences, JSON,
    box_2d (y first, 0-1000) and bare 4-number lists, and a pixel "X,Y"
    ONLY when the caller asked for pixels.
  * SET-OF-MARK. A numbered plate on each candidate box lets the model
    answer "7" - a number cannot be off by twenty pixels.

Pure Python + Pillow (imported lazily); never raises at the public API.
"""
from __future__ import annotations

import json
import math
import re

__all__ = [
    "VLM_MAX_IMAGE_TOKENS", "est_tokens", "fit_size", "fit_for_vlm",
    "strip_tags", "parse_reply", "box_to_rect", "valid_box",
    "draw_marks", "prompt_mark", "prompt_box2d", "prompt_pixel",
]

VLM_MAX_IMAGE_TOKENS = 960
_PATCH = 48                     # gemma's image patch grid on this build
_MAX_BOX_AREA = 0.60            # a "box" over 60% of the image is no box


def _max_tokens(max_tokens=None) -> int:
    try:
        if max_tokens is None:
            from core import config as _cfg
            max_tokens = getattr(_cfg, "VLM_MAX_IMAGE_TOKENS",
                                 VLM_MAX_IMAGE_TOKENS)
        v = int(max_tokens)
        return v if v >= 16 else VLM_MAX_IMAGE_TOKENS
    except Exception:
        return VLM_MAX_IMAGE_TOKENS


def est_tokens(w, h) -> int:
    """The image tokens gemma spends on a w x h image (ceil per 48 px)."""
    try:
        return int(math.ceil(float(w) / _PATCH) * math.ceil(float(h) / _PATCH))
    except Exception:
        return 0


def fit_size(w, h, max_tokens=None) -> tuple:
    """(new_w, new_h, scale): the largest scale <= 1 whose estimate is
    <= ``max_tokens``. Never raises."""
    try:
        w, h = int(w), int(h)
        cap = _max_tokens(max_tokens)
        if w <= 0 or h <= 0:
            return w, h, 1.0
        if est_tokens(w, h) <= cap:
            return w, h, 1.0
        scale = min(1.0, math.sqrt(cap * _PATCH * _PATCH / float(w * h)))
        for _ in range(400):
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            if est_tokens(nw, nh) <= cap:
                return nw, nh, nw / float(w)
            scale *= 0.99
        return max(1, int(w * scale)), max(1, int(h * scale)), scale
    except Exception:
        return w, h, 1.0


def fit_for_vlm(img, max_tokens=None):
    """(image, scale): ``img`` (a PIL image) resized so its token estimate
    is <= the cap; ``scale`` maps image pixels back (native = px / scale).
    The image is returned unchanged (scale 1.0) when it already fits or on
    any failure."""
    try:
        w, h = img.size
        nw, nh, scale = fit_size(w, h, max_tokens)
        if (nw, nh) == (w, h):
            return img, 1.0
        from PIL import Image
        return img.resize((nw, nh), Image.LANCZOS), scale
    except Exception:
        return img, 1.0


# ── reading the answer ───────────────────────────────────────────────────
# A leading "[local-vision] " / "[intent:x]" tag - a WORD in brackets, never
# a bracketed list of numbers ("[120, 40, 180, 300]" is an answer).
_TAG_RE = re.compile(r"^\s*(?:\[[A-Za-z][\w :.-]{0,39}\]\s*)+")
_FENCE_RE = re.compile(r"```(?:json|JSON|python)?\s*(.*?)```", re.DOTALL)
_NONE_RE = re.compile(r"^\W*(?:none|not[_\s-]?found|no|n/?a|nothing|"
                      r"not\s+visible|cannot\s+find|can'?t\s+find)\W*$",
                      re.IGNORECASE)
_BOX2D_RE = re.compile(
    r"box_?2d[\"']?\s*[:=]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", re.IGNORECASE)
_LIST4_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
_MARK_RE = re.compile(r"^\W*(?:(?:number|mark|box|option|#)\s*)?(\d{1,3})\W*$",
                      re.IGNORECASE)


def strip_tags(text) -> str:
    """``text`` without leading "[local-vision] "-style tags. Never raises."""
    try:
        return _TAG_RE.sub("", str(text or ""), count=1).strip()
    except Exception:
        return ""


def valid_box(box) -> bool:
    """A 0-1000 (y1, x1, y2, x2) box that is a real box: ordered, inside the
    image, smaller than 60% of it."""
    try:
        y1, x1, y2, x2 = (float(v) for v in box)
        if not (0 <= y1 < y2 <= 1000 and 0 <= x1 < x2 <= 1000):
            return False
        return (y2 - y1) * (x2 - x1) < _MAX_BOX_AREA * 1000 * 1000
    except Exception:
        return False


def _boxes_from_json(obj) -> list:
    out = []
    if isinstance(obj, dict):
        b = obj.get("box_2d") or obj.get("box2d") or obj.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out.append(tuple(float(v) for v in b))
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 4 and all(isinstance(v, (int, float)) for v in obj):
            out.append(tuple(float(v) for v in obj))
        else:
            for item in obj:
                out.extend(_boxes_from_json(item))
    return out


def parse_reply(text, fmt: str = "box2d") -> dict:
    """Read a vision answer. ``fmt``: "box2d" (gemma's 0-1000, y-first
    boxes), "mark" (a set-of-mark number) or "pixel" (the legacy "X,Y").

    Returns {"kind": "box", "box": (y1, x1, y2, x2)} / {"kind": "mark",
    "mark": n} / {"kind": "point", "xy": (x, y)} / {"kind": "none"} /
    {"kind": "ambiguous", "boxes": [...]} / {"kind": "invalid", "why": ...}.
    Never raises."""
    try:
        raw = strip_tags(text)
        if not raw:
            return {"kind": "invalid", "why": "empty"}
        fence = _FENCE_RE.search(raw)
        body = fence.group(1).strip() if fence else raw
        if _NONE_RE.match(body) or body.upper().startswith(("NOT_FOUND",
                                                            "NONE")):
            return {"kind": "none"}
        if fmt == "mark":
            m = _MARK_RE.match(body)
            if m:
                return {"kind": "mark", "mark": int(m.group(1))}
            nums = re.findall(r"\b\d{1,3}\b", body)
            if len(set(nums)) == 1 and len(body) <= 40:
                return {"kind": "mark", "mark": int(nums[0])}
            if len(set(nums)) > 1:
                return {"kind": "ambiguous", "marks": [int(n) for n in nums]}
            return {"kind": "invalid", "why": "no mark number"}
        if fmt == "pixel":
            m = re.fullmatch(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\.?", body)
            if not m and len(body) <= 32:
                m = re.search(r"(\d+)\s*,\s*(\d+)\s*\)?\.?\s*$", body)
            if not m:
                return {"kind": "invalid", "why": "no X,Y pair"}
            return {"kind": "point", "xy": (int(m.group(1)), int(m.group(2)))}
        # box2d
        boxes: list = []
        try:
            boxes = _boxes_from_json(json.loads(body))
        except Exception:
            boxes = []
        if not boxes:
            boxes = [tuple(float(v) for v in m.groups())
                     for m in _BOX2D_RE.finditer(body)]
        if not boxes:
            boxes = [tuple(float(v) for v in m.groups())
                     for m in _LIST4_RE.finditer(body)]
        if not boxes:
            return {"kind": "invalid", "why": "no box_2d"}
        uniq = list(dict.fromkeys(boxes))
        if len(uniq) > 1:
            return {"kind": "ambiguous", "boxes": uniq}
        box = uniq[0]
        if not valid_box(box):
            return {"kind": "invalid", "why": f"bad box {box}"}
        return {"kind": "box", "box": box}
    except Exception as e:
        return {"kind": "invalid", "why": type(e).__name__}


def box_to_rect(box, img_w, img_h) -> tuple:
    """A 0-1000 (y1, x1, y2, x2) box -> (x, y, w, h) in image pixels."""
    y1, x1, y2, x2 = (float(v) for v in box)
    x = x1 / 1000.0 * img_w
    y = y1 / 1000.0 * img_h
    return (x, y, (x2 - x1) / 1000.0 * img_w, (y2 - y1) / 1000.0 * img_h)


# ── set-of-mark ──────────────────────────────────────────────────────────
def _font(size: int):
    from PIL import ImageFont
    for name in ("segoeuib.ttf", "arialbd.ttf", "segoeui.ttf", "arial.ttf",
                 "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def draw_marks(img, rects, max_marks: int = 30, digit_px: int = 22):
    """A copy of ``img`` with a numbered plate (1..n, digits >= 20 px) at
    the top-left of each rect (x, y, w, h in image pixels) and a thin
    outline, so the model can answer with a number. Returns (image,
    numbers) where numbers[i] is the mark drawn for rects[i] (None when
    over ``max_marks``). Never raises (returns the image unmarked)."""
    try:
        from PIL import ImageDraw
        out = img.convert("RGB").copy()
        d = ImageDraw.Draw(out)
        font = _font(max(20, int(digit_px)))
        numbers = []
        for i, r in enumerate(rects or ()):
            if i >= max_marks:
                numbers.append(None)
                continue
            x, y, w, h = (float(v) for v in r)
            n = i + 1
            d.rectangle([x, y, x + w, y + h], outline=(255, 0, 255), width=2)
            label = str(n)
            try:
                tb = d.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                tw, th = 12 * len(label), 20
            px, py = max(0, x - 2), max(0, y - th - 8)
            d.rectangle([px, py, px + tw + 8, py + th + 8], fill=(255, 0, 255))
            d.text((px + 4, py + 2), label, fill=(255, 255, 255), font=font)
            numbers.append(n)
        return out, numbers
    except Exception:
        return img, [None for _ in (rects or ())]


def _q(target) -> str:
    return " ".join(str(target or "").split())[:200]


def prompt_mark(target) -> str:
    return (f"The image shows one window with numbered magenta boxes. Which "
            f"number is {_q(target)}? Reply with the number only, or NONE "
            f"if no numbered box is it.")


def prompt_box2d(target) -> str:
    return (f"Find {_q(target)} in this image. Reply with JSON only: "
            '{"box_2d": [y1, x1, y2, x2]} with coordinates normalised to '
            "0-1000, y first. If it is not visible, reply NONE.")


def prompt_pixel(target, w, h) -> str:
    """The legacy pixel prompt (find_click_target's callers keep it)."""
    return (
        f"You are helping a UI automation agent click PRECISELY on a target.\n"
        f"The image is {w}x{h} pixels. Origin (0,0) is the TOP-LEFT.\n"
        f"Target: {target}\n\n"
        f"Reply with ONLY the pixel coordinates of the EXACT VISUAL CENTRE of "
        f"the clickable element (not the centre of its label, the centre of "
        f"the clickable area itself).\n"
        f"Format: X,Y    (e.g. 432,718)\n"
        f"If the element isn't visible, reply: NOT_FOUND")
