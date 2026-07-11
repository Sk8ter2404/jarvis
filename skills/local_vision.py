"""
Local vision skill for JARVIS.

Mirrors the existing Claude-vision actions (`see_screen` /
`click, <description>`) but forces the query through a local VLM served
by Ollama, so they keep working offline / when the cloud is rate-limited
or out of credits. The cloud-vision retry path in bobert_companion already
falls through to the local VLM automatically; these actions are the
*explicit* hook for voice phrases like "use local vision" or
"describe this screen offline."

Actions added:
  local_describe_screen <question>        — capture all monitors (or one,
                                            if 'monitor:NAME|...' prefix)
                                            and answer via the local VLM.
  local_click_target_by_description <d>   — find a UI target with the
                                            local VLM and click it.

Voice phrasing the LLM is expected to map onto these actions:
  "describe the screen offline"           → local_describe_screen
  "use local vision: what's on screen"    → local_describe_screen
  "click X offline / using local vision"  → local_click_target_by_description X
  "find the play button with the local model"
                                          → local_click_target_by_description …

Config (set in bobert_companion.py at module scope):
  LOCAL_VISION_FALLBACK   — master on/off switch (default True)
  LOCAL_VISION_MODEL      — Ollama tag, e.g. 'qwen2.5vl:7b', 'llava:13b'
  LOCAL_LLM_BASE_URL      — Ollama HTTP endpoint (shared with the text LLM)

If Ollama isn't running / the VLM isn't pulled yet, returns a tidy
one-liner that explains the situation rather than crashing — and the
core fallback path will have already kicked off background install/pull.
"""
from __future__ import annotations

import io
import re


def _bobert():
    """Lazily resolve the bobert_companion module so this skill can be
    imported without forcing a circular reference at load time.

    Prefer whichever candidate actually LOOKS like the monolith:
    `__main__` always exists, so the old `get("__main__") or
    get("bobert_companion")` never fell through — correct in the live
    process (where the monolith IS __main__) but in any other host
    (test harness, action sweep, a driver script) it returned the
    host's own module and the unguarded `b.take_screenshot(...)` call
    sites crashed with AttributeError (caught by the 660-action sweep
    2026-07-11)."""
    import sys
    for name in ("__main__", "bobert_companion"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "take_screenshot"):
            return m
    return sys.modules.get("bobert_companion")


def _parse_monitor_prefix(text: str) -> tuple[str | None, str]:
    """Strip an optional 'monitor:NAME|' prefix. Falls back to the parent
    module's parser when available so we honour the exact monitor-naming
    rules the rest of JARVIS uses."""
    b = _bobert()
    if b is not None and hasattr(b, "_parse_monitor_prefix"):
        return b._parse_monitor_prefix(text)
    m = re.match(r"^\s*monitor:([A-Za-z0-9_-]+)\s*\|\s*(.*)$", text or "")
    if m:
        return m.group(1).lower(), m.group(2)
    return None, text or ""


def _take_screenshot(monitor: str | None):
    b = _bobert()
    if b is not None and hasattr(b, "take_screenshot"):
        return b.take_screenshot(monitor=monitor)
    return None


def _take_all_monitor_screenshots() -> dict:
    b = _bobert()
    if b is not None and hasattr(b, "take_all_monitor_screenshots"):
        return b.take_all_monitor_screenshots()
    return {}


def _call_local_vision(question: str, png_images: list, max_tokens: int = 600):
    """Call into bobert_companion._call_local_vision so we share the
    Ollama-alive check, background-pull logic, and `[local-vision]`
    print tag with the core fallback path."""
    b = _bobert()
    if b is None or not hasattr(b, "_call_local_vision"):
        return None
    return b._call_local_vision(question, png_images, max_tokens=max_tokens)


def _missing_local_vision_msg() -> str:
    """Return a TTS-friendly explanation for why local vision isn't
    available right now. The user-visible side of the same checks
    `_call_local_vision()` runs internally."""
    b = _bobert()
    if b is None:
        return "local vision is not available, sir — bobert_companion isn't loaded"
    if not getattr(b, "LOCAL_VISION_FALLBACK", False):
        return ("local vision is disabled in config, sir — "
                "set LOCAL_VISION_FALLBACK = True to enable it")
    model = getattr(b, "LOCAL_VISION_MODEL", "")
    if not model:
        return ("local vision has no model configured, sir — "
                "set LOCAL_VISION_MODEL to something like 'qwen2.5vl:7b'")
    if hasattr(b, "_ollama_alive") and not b._ollama_alive():
        return ("Ollama isn't running, sir — start the Ollama service "
                "or install it via winget install Ollama.Ollama")
    if hasattr(b, "_ollama_has_model") and not b._ollama_has_model(model):
        return (f"the local vision model `{model}` hasn't finished "
                f"downloading yet, sir — a background pull has been queued")
    return f"local vision call to `{model}` failed, sir — check the console for details"


def local_describe_screen(question: str = "") -> str:
    """Capture the screen and answer the question via the local VLM.

    Honours the same 'monitor:NAME|...' prefix the cloud see_screen does.
    If no monitor is specified, captures every monitor and asks the VLM
    about all of them at once (with positional labels in the prompt)."""
    monitor, question = _parse_monitor_prefix(question)
    q = (question or "").strip() or "Describe in detail what is currently on the screen."

    b = _bobert()

    if monitor is None:
        print("  [local-vision] 📸 Capturing all monitors…", flush=True)
        images = _take_all_monitor_screenshots()
        if not images:
            return "could not capture any monitor"
        names = list(images.keys())
        pngs  = [images[n] for n in names]
        labels = "\n".join(
            f"Image #{i+1} = {n.upper()} monitor" for i, n in enumerate(names)
        )
        prompt = (
            f"You are looking at {len(images)} monitors at once. They are "
            f"provided in this order:\n{labels}\n\n"
            f"When answering, name which monitor(s) the relevant content is on. "
            f"If the question doesn't apply to a given monitor, you can skip it.\n\n"
            f"Question: {q}"
        )
        print(f"  [local-vision] 👁  Asking local VLM about {', '.join(names)}…", flush=True)
        text = _call_local_vision(prompt, pngs, max_tokens=900)
        if not text:
            return _missing_local_vision_msg()
        result = f"[local-vision] {text}"
        if b is not None and hasattr(b, "_push_screen_context"):
            try:
                b._push_screen_context(None, q, result, images)
            except Exception:
                pass
        print(f"  [local-vision] ✓ Got answer ({len(result)} chars)", flush=True)
        return result

    print(f"  [local-vision] 📸 Capturing screen ({monitor} monitor)…", flush=True)
    png = _take_screenshot(monitor)
    if png is None:
        return "could not capture screen"
    print("  [local-vision] 👁  Asking local VLM…", flush=True)
    text = _call_local_vision(q, [png])
    if not text:
        return _missing_local_vision_msg()
    result = f"[local-vision] {text}"
    if b is not None and hasattr(b, "_push_screen_context"):
        try:
            b._push_screen_context(monitor, q, result, {monitor: png})
        except Exception:
            pass
    print(f"  [local-vision] ✓ Got answer ({len(result)} chars)", flush=True)
    return result


def _local_query_coords(description: str, png_bytes: bytes,
                        w: int, h: int) -> tuple[int, int] | None:
    """Ask the local VLM for the (x, y) of a UI element. Same prompt
    contract as the Claude version (_query_vision_for_coords)."""
    prompt = (
        f"You are helping a UI automation agent click PRECISELY on a target.\n"
        f"The image is {w}x{h} pixels. Origin (0,0) is the TOP-LEFT.\n"
        f"Target: {description}\n\n"
        f"Reply with ONLY the pixel coordinates of the EXACT VISUAL CENTRE of "
        f"the clickable element (not the centre of its label, the centre of "
        f"the clickable area itself).\n"
        f"Format: X,Y    (e.g. 432,718)\n"
        f"If the element isn't visible, reply: NOT_FOUND"
    )
    answer = _call_local_vision(prompt, [png_bytes], max_tokens=64) or ""
    if "NOT_FOUND" in answer.upper():
        return None
    m = re.search(r"(\d+)\s*,\s*(\d+)", answer)
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    if 0 <= x <= w and 0 <= y <= h:
        return x, y
    return None


def _find_click_target_local(description: str,
                             monitor: str | None) -> tuple[int, int] | None:
    """Two-pass local-VLM equivalent of find_click_target(): low-res for
    rough estimate, then a 500×500 native-res crop for refinement."""
    try:
        from PIL import Image
    except ImportError:
        return None

    b = _bobert()
    if b is None:
        return None

    png = b.take_screenshot(monitor=monitor, max_dim=1568)
    if png is None:
        return None
    img1 = Image.open(io.BytesIO(png))
    w1, h1 = img1.size

    pass1 = _local_query_coords(description, png, w1, h1)
    if pass1 is None:
        return None
    rx1, ry1 = pass1

    full_png = b.take_screenshot(monitor=monitor, max_dim=10000)
    if full_png is None:
        # Pass-2 capture failed — img1 is the DOWNSCALED Pass-1 image
        # (max_dim=1568), so its coords are NOT native screen pixels. Use
        # the TRUE native size of the captured region so the Pass-1 coords
        # scale to native below; treating the downscaled size as full-res
        # produced off-by-hundreds clicks on any >1568px display. Mirrors
        # the identical fix in bobert_companion.find_click_target.
        full_img = None
        nat = getattr(b, "_native_capture_size", None)
        try:
            nw, nh = nat(monitor) if callable(nat) else (0, 0)
        except Exception:
            nw, nh = 0, 0
        full_w = nw or w1
        full_h = nh or h1
    else:
        full_img = Image.open(io.BytesIO(full_png))
        full_w, full_h = full_img.size

    scale_x = full_w / w1
    scale_y = full_h / h1
    cx_full = int(rx1 * scale_x)
    cy_full = int(ry1 * scale_y)

    refined_x, refined_y = cx_full, cy_full

    if full_img is not None and (full_w > w1 or full_h > h1):
        CROP = 500
        left   = max(0, cx_full - CROP // 2)
        top    = max(0, cy_full - CROP // 2)
        right  = min(full_w, cx_full + CROP // 2)
        bottom = min(full_h, cy_full + CROP // 2)
        crop = full_img.crop((left, top, right, bottom))
        cw, ch = crop.size
        crop_buf = io.BytesIO()
        crop.save(crop_buf, format="PNG")
        crop_png = crop_buf.getvalue()

        print(f"  [local-vision] 🔍 Refining position in {cw}x{ch} crop…", flush=True)
        pass2 = _local_query_coords(description, crop_png, cw, ch)
        if pass2 is not None:
            refined_x = left + pass2[0]
            refined_y = top  + pass2[1]

    # Translate full-res image coords → absolute LOGICAL screen coords, the
    # space pyautogui clicks in. refined_x/y are native-pixel offsets from the
    # captured region's top-left; scale them by (logical_extent/native_extent)
    # so a >100%-DPI display doesn't overshoot, then add the region's logical
    # origin (NEGATIVE for monitors left of / above the primary). Mirrors
    # bobert_companion.find_click_target — the old code added the raw native
    # offset with no DPI scale, and for monitor=None added NO origin at all,
    # so on a negative-origin multi-monitor layout clicks could land on the
    # wrong monitor entirely (the "can't click the bookmark bar" bug).
    MONITORS = getattr(b, "MONITORS", {})
    if monitor and monitor in MONITORS:
        mx, my, lw, lh = MONITORS[monitor]
        sx = (lw / full_w) if full_w else 1.0
        sy = (lh / full_h) if full_h else 1.0
        if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
            print(f"  [local-vision] DPI scale: monitor={monitor} "
                  f"native={full_w}x{full_h} logical={lw}x{lh} "
                  f"→ click x({sx:.3f},{sy:.3f})", flush=True)
        return int(mx + refined_x * sx), int(my + refined_y * sy)
    vsb = getattr(b, "_virtual_screen_bounds", None)
    if callable(vsb):
        try:
            vx, vy, vw, vh = vsb()
        except Exception:
            return refined_x, refined_y
        sx = (vw / full_w) if full_w else 1.0
        sy = (vh / full_h) if full_h else 1.0
        if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
            print(f"  [local-vision] DPI scale: virtual "
                  f"native={full_w}x{full_h} logical={vw}x{vh} "
                  f"→ click x({sx:.3f},{sy:.3f})", flush=True)
        return int(vx + refined_x * sx), int(vy + refined_y * sy)
    return refined_x, refined_y


def local_click_target_by_description(description: str) -> str:
    """Locate a UI element with the local VLM and click it. Honours the
    same 'monitor:NAME|...' prefix and self-preservation guard the cloud
    click action uses."""
    monitor, description = _parse_monitor_prefix(description)
    description = (description or "").strip()
    if not description:
        return "describe what I should click, sir"

    b = _bobert()
    if b is None:
        return "local vision is not available, sir — bobert_companion isn't loaded"

    # Reuse the self-preservation check so 'close powershell' is refused
    # whether the cloud or local eye finds it.
    if hasattr(b, "_is_self_close_attempt") and b._is_self_close_attempt(description):
        return (
            f"REFUSED: '{description}' looks like an attempt to close the terminal "
            f"or Python process running me. Closing it would kill my session. "
            f"Ask the user to close it manually if they really want to."
        )

    target = f" on {monitor} monitor" if monitor else ""
    print(f"  [local-vision] 📸 Looking for '{description}'{target}…", flush=True)
    coords = _find_click_target_local(description, monitor=monitor)
    if coords is None:
        msg = _missing_local_vision_msg()
        if not msg.startswith("local vision call to"):
            return msg
        return f"could not find '{description}' on screen via local vision"

    try:
        b.ui_click(coords[0], coords[1])
    except Exception as e:
        # Surface UIFailsafeError's friendly message rather than the traceback.
        return f"found '{description}' at {coords} but click failed: {e}"
    return f"[local-vision] clicked '{description}' at {coords}"


def register(actions: dict):
    actions["local_describe_screen"] = local_describe_screen
    actions["local_click_target_by_description"] = local_click_target_by_description
