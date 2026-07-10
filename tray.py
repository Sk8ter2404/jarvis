#!/usr/bin/env python3
"""
JARVIS system-tray applet — arc-reactor icon + 4 status pips + grouped menu.

Spawned as a subprocess by bobert_companion.py at startup (mirrors the
hud / reticle launcher pattern). Reads hud_state.json sibling to
bobert_companion.py to drive icon state.

Icon layout (redesigned for legibility at Windows' 16/24 px rasterisations —
the old 4-corner-pip design collapsed to indistinct ~4 px dots when the shell
downscaled the 64 px canvas):

  • Base       — assets/jarvis_icon.png (cyan arc-reactor). Matches the HUD's
                 visual identity. Falls back to a procedural reactor disc if
                 the asset is missing.
  • PRIMARY signal = full-icon TINT. The whole reactor is recoloured by the
                 listening state so it reads at any size where tiny pips don't:
                   green = awake · gray = standby/sleep · RED = muted.
  • Speaking   — a bold blue HALO/ring pulses around the reactor while JARVIS
                 is talking (a large glow survives downscaling).
  • Queue      — when the overnight-upgrade queue is non-empty, a LARGE
                 high-contrast badge (dark disc + bright digit) sits in the
                 bottom-right corner so a single digit is readable at 24 px.
                 It is dropped gracefully (too small to read) at 16 px.
  • Bambu H2D  — a small but bold orange print-mark in the top-right corner
                 only while a print is running (secondary signal, not a pip).

Right-click menu — common toggles at top, power-user verbs grouped into
five submenus (Power tools / AI / Memory / Diagnostics / Settings), with
About + Quit at the bottom. Toggle items show a checkmark when active;
items that have no meaning right now (e.g. "Stop Running Pipeline" when
no overnight engine is active) are greyed out via enabled=lambda.

    ● <status lines>          (disabled MenuItems — same info as tooltip)
    ─────
    Pause Listening      [✓ when in standby]
    Mute TTS             [✓ when hud_state.tts_muted]
    Mute Mic             [✓ when hud_state.mic_muted] — drives the red tint
    Ambient Mode         [✓ when hud_state.ambient_mode_active]
    ─────
    Open HUD
    Run Upgrade Now
    Restart JARVIS
    Shut Down JARVIS
    ─────
    Power tools ▶ Stop Pipeline / Backup / Reload Skills / Smoke Test /
                  Pause Daemons / Reset LLM Cache / Open Folder /
                  Open Task Queue / Live Log / Crash Reports / Changelog
    AI ▶          Switch Claude / qwen / llama / other / Debug / Stats / Clear Cache
    Memory ▶      Open memory / Dossier / Recent Facts / Reset / Export / Forget Hour
    Diagnostics ▶ Run / Last / Test Mic / Test TTS / Test Vision / Test Skills / Latency
    Settings ▶    Voice / AI / Privacy / Integrations / Advanced
    ─────
    About JARVIS
    Show Today's Summary
    Queue Task…
    Quit Tray Only

IPC contract (UNCHANGED — bobert_companion.py drainer depends on this):
  • READS  hud_state.json    (state, tts_amplitude, mic_muted, bambu_active, …)
  • READS  jarvis_todo.md    (count of unchecked '- [ ]' lines → queue badge)
  • READS  logs/             (today's session_*.log files → summary dialog)
  • WRITES tray_commands.json — JSON list of {cmd, ts, …} pending commands.
                                bobert_companion.py drains and dispatches.

CLI:
  python tray.py --parent-pid 12345 [--icon-path PATH]
"""
import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import types

# CREATE_NO_WINDOW safety net — the tray runs as pythonw; any console helper
# it spawns without a flag pops a visible ghost window (2026-07-10).
try:
    from core.no_window_subprocess import install as _install_no_window
    _install_no_window()
except Exception:
    pass

try:
    import pystray
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
except Exception as e:  # pragma: no cover - import-time hard-dep guard; tests inject a fake pystray so the real import never fails here
    print(f"[tray] missing dependency: {e}")
    print("[tray] install with:  pip install pystray pillow")
    sys.exit(1)

# tkinter is stdlib but can be absent on stripped-down Python builds
try:
    import tkinter as tk
    from tkinter import simpledialog
    _HAS_TK = True
except Exception:  # pragma: no cover - import-time optional-dep guard (tkinter absent on stripped Python); not reachable once the module has imported
    _HAS_TK = False

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - import-time optional-dep guard (psutil absent); not reachable once the module has imported
    _HAS_PSUTIL = False

PROJECT_DIR        = os.path.dirname(os.path.abspath(__file__))
HUD_STATE_FILE     = os.path.join(PROJECT_DIR, "hud_state.json")
TRAY_COMMANDS_FILE = os.path.join(PROJECT_DIR, "tray_commands.json")
TODO_FILE          = os.path.join(PROJECT_DIR, "jarvis_todo.md")
LOGS_DIR           = os.path.join(PROJECT_DIR, "logs")
HUD_SCRIPT         = os.path.join(PROJECT_DIR, "hud", "jarvis_hud.py")
ASSETS_DIR         = os.path.join(PROJECT_DIR, "assets")
DEFAULT_ICON_PATH  = os.path.join(ASSETS_DIR, "jarvis_icon.png")
DATA_DIR           = os.path.join(PROJECT_DIR, "data")
CHANGELOG_FILE     = os.path.join(PROJECT_DIR, "CHANGELOG.md")
# Two DIFFERENT version sources — keeping them straight is what stops the
# About dialog from disagreeing with GitHub:
#   • RELEASE_VERSION_FILE — the top-level VERSION file: the shareable release
#     string that also backs core/version.py, the git tag and the GitHub
#     release. This is the PRIMARY "Version:" line the user sees.
#   • VERSION_FILE (data/version.json) — the self-upgrade pipeline's INTERNAL
#     counter, bumped a patch every overnight run (e.g. 1.0.17). Shown only as
#     a clearly-labelled "Upgrade build" so it can't be mistaken for the
#     release version.
RELEASE_VERSION_FILE = os.path.join(PROJECT_DIR, "VERSION")
VERSION_FILE       = os.path.join(DATA_DIR, "version.json")
INSTANCES_FILE     = os.path.join(DATA_DIR, "instances.json")
PIPELINE_LOCK_FILE = os.path.join(PROJECT_DIR, "pipeline_lock.json")
OVERNIGHT_FLAG     = os.path.join(PROJECT_DIR, ".overnight_active")
MEMORY_FACTS_FILE  = os.path.join(DATA_DIR, "long_term_memory", "facts.json")
SETTINGS_WINDOW    = os.path.join(PROJECT_DIR, "tools", "settings_window.py")
SHOW_LOG_PS1       = os.path.join(PROJECT_DIR, "_show_log.ps1")

TICK_SECONDS = 0.20   # 5 Hz animation tick — fast enough for the speaking
                      # dot pulse, slow enough to avoid hammering the Windows
                      # shell with icon updates.
SIZE = 64             # tray-icon canvas (Windows scales 16/24/32/40/48 from this)

# ── Signal palette ────────────────────────────────────────────────────────
# The listen colour is now the PRIMARY signal: the whole reactor is tinted
# toward it (see _tint_image), so it reads at any rasterisation size. The
# speaking colour drives a pulsing halo; the queue colour the badge disc;
# the bambu colour a small corner print-mark.
LISTEN_GREEN = (60, 210, 90)      # awake
LISTEN_GRAY  = (140, 140, 150)    # standby / sleep
LISTEN_RED   = (220, 40, 40)      # muted

SPEAK_BLUE   = (60, 150, 255)
SPEAK_DIM    = (25, 45, 85)       # very dim base when not speaking

QUEUE_YELLOW = (235, 200, 30)
QUEUE_DIM    = (55, 50, 18)       # dim when queue is empty

BAMBU_ORANGE = (235, 130, 30)
BAMBU_WHITE  = (220, 220, 220)    # idle

# Tint strength — how strongly the listen colour recolours the reactor.
# Awake/standby get a gentle wash so the arc-reactor identity survives;
# muted is pushed harder so "RED = muted" is unmistakable even at 16 px.
TINT_STRENGTH       = 0.45
TINT_STRENGTH_MUTED = 0.62
# Badge geometry as a fraction of the canvas — deliberately large so a
# single digit survives Windows' downscale to 24 px.
BADGE_FRAC = 0.46

# Backwards-compat — older code paths still reference COLORS["idle"] etc.
# Keeps the module import-safe if anything outside this file pokes at the
# table; the new renderer doesn't read from it.
COLORS = {
    "idle":      ((48, 100, 180),  (90, 180, 255)),
    "listening": ((180, 180, 200), (255, 255, 255)),
    "thinking":  ((180, 120, 0),   (255, 200, 60)),
    "speaking":  ((220, 170, 30),  (255, 220, 80)),
    "standby":   ((60, 40, 100),   (155, 140, 255)),
    "alert":     ((200, 0, 0),     (255, 80, 80)),
    "bambu":     ((220, 110, 0),   (255, 170, 60)),
}


# ─── Base icon (arc-reactor PNG) ─────────────────────────────────────────
# Loaded once at startup, resized to SIZE×SIZE, then re-used as the
# background layer for every animation frame. If the asset is missing or
# the file is corrupt, _base_icon stays None and _render_icon falls back
# to the procedural 4-dot grid — the tray must never crash because of a
# missing icon file (parent watchdog regression risk).
_base_icon: "Image.Image | None" = None
_icon_path: str = DEFAULT_ICON_PATH


def _load_base_icon(path: str) -> None:
    """Try to load the JARVIS arc-reactor PNG. Sets _base_icon on success;
    leaves it None on any error so the renderer falls through to the
    procedural fallback."""
    global _base_icon
    _base_icon = None
    if not path or not os.path.exists(path):
        print(f"[tray] icon asset not found at {path} — using procedural fallback")
        return
    try:
        img = Image.open(path).convert("RGBA")
        # Resize to the tray canvas — LANCZOS keeps the arc-reactor ring crisp
        # even when Windows downscales further for 16/24/32 px rasterisations.
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
        _base_icon = img
        print(f"[tray] base icon loaded from {path}")
    except Exception as e:
        print(f"[tray] failed to load icon {path} ({e}) — using fallback")
        _base_icon = None


def _read_hud_state() -> dict:
    """Best-effort read of hud_state.json. Returns empty dict on any error."""
    try:
        with open(HUD_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _send_command(cmd: str, **kwargs) -> None:
    """Append a command to tray_commands.json using the same atomic
    temp+rename pattern the other JSON inboxes (pending_speech.json etc.)
    use. Bobert drains this on a 0.5s background timer."""
    payload = {"cmd": cmd, "ts": time.time()}
    payload.update(kwargs)
    try:
        existing = []
        if os.path.exists(TRAY_COMMANDS_FILE):
            try:
                with open(TRAY_COMMANDS_FILE, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if raw:
                    decoded, _ = json.JSONDecoder().raw_decode(raw)
                    if isinstance(decoded, list):
                        existing = decoded
            except Exception:
                existing = []
        existing.append(payload)
        fd, tmp = tempfile.mkstemp(dir=PROJECT_DIR, suffix=".tmp", prefix="tray_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(existing, f)
            os.replace(tmp, TRAY_COMMANDS_FILE)
        except Exception:
            try: os.remove(tmp)
            except Exception: pass
            raise
        print(f"[tray] sent command: {cmd}")
    except Exception as e:
        print(f"[tray] command write failed ({cmd}): {e}")


# ─── Icon rendering ──────────────────────────────────────────────────────

def _blend(dark: tuple, bright: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(dark[i] + (bright[i] - dark[i]) * t) for i in range(3))


# Cached fonts keyed by requested size — Pillow's truetype loader is
# surprisingly hot when re-invoked at 5 Hz, so we keep one font per size.
_FONT_CACHE: dict[int, "ImageFont.ImageFont"] = {}


def _get_font(size: int):
    """Pillow font for the queue-count digit. Falls back to bitmap default."""
    font = _FONT_CACHE.get(size)
    if font is not None:
        return font
    try:
        from PIL import ImageFont as _IF
        # Arial ships with Windows; if it isn't found Pillow's loader walks
        # through fallbacks before we land on the bitmap default.
        for name in ("arialbd.ttf", "arial.ttf", "segoeuib.ttf", "segoeui.ttf"):
            try:
                font = _IF.truetype(name, size)
                break
            except Exception:
                continue
        if font is None:
            font = _IF.load_default()
    except Exception:
        font = None
    if font is not None:
        _FONT_CACHE[size] = font
    return font


def _compute_signal_colors(state: str, frame: int, tts_amplitude: float,
                           queue_count: int, muted: bool,
                           bambu_active: bool) -> dict:
    """Resolve every status signal the renderers need into one dict.

    Shared by the arc-reactor renderer and the procedural fallback so both
    show identical colours for a given state. Besides the four base colours
    it also resolves the derived values the redesign keys off:

      • ``listen``       — the tint colour (PRIMARY signal); the whole
                           reactor is recoloured toward it.
      • ``tint_strength``— how hard to push the tint (muted pushes harder so
                           red is unmistakable at 16 px).
      • ``speak``        — pulsing halo colour; ``speak_t`` is the 0..1 pulse
                           level (0 when quiet) so the halo can fade in/out.
      • ``queue``/``queue_count`` — badge disc colour + the integer to draw.
      • ``bambu``/``bambu_active`` — corner print-mark colour + whether to
                           draw it at all.
    """
    raw = str(state or "").lower()

    if muted:
        listen_rgb = LISTEN_RED
        tint_strength = TINT_STRENGTH_MUTED
    elif raw in ("standby", "sleeping", "sleep"):
        listen_rgb = LISTEN_GRAY
        tint_strength = TINT_STRENGTH
    else:
        listen_rgb = LISTEN_GREEN
        tint_strength = TINT_STRENGTH

    is_speaking = (raw == "speaking") or ((tts_amplitude or 0.0) > 0.02)
    if is_speaking:
        # Pulse between 0.55 and 1.0 brightness at ~1.25 Hz (period 4 frames
        # @ 5 Hz tick); ride the TTS amplitude envelope when published.
        t = 0.55 + 0.45 * (math.sin(frame * 2 * math.pi / 4) + 1) / 2
        t = max(t, min(1.0, 0.55 + (tts_amplitude or 0.0) * 0.5))
        speak_rgb = _blend(SPEAK_DIM, SPEAK_BLUE, t)
        speak_t = t
    else:
        speak_rgb = SPEAK_DIM
        speak_t = 0.0

    count = max(0, int(queue_count or 0))
    queue_rgb = QUEUE_YELLOW if count > 0 else QUEUE_DIM

    bambu_rgb = BAMBU_ORANGE if bambu_active else BAMBU_WHITE

    return {
        "listen": listen_rgb,
        "tint_strength": tint_strength,
        "speak":  speak_rgb,
        "speak_t": speak_t,
        "queue":  queue_rgb,
        "bambu":  bambu_rgb,
        "queue_count": count,
        "bambu_active": bambu_active,
    }


def _tint_image(base: "Image.Image", rgb: tuple, strength: float) -> "Image.Image":
    """Recolour ``base`` toward ``rgb`` while preserving its shape + shading.

    The arc-reactor's own luminance is kept (so the ring/disc detail and the
    transparent surround survive); only the hue is washed toward the signal
    colour. This is the PRIMARY status channel — a full-icon tint stays
    legible at 16 px where the old corner pips dissolved into ~4 px mush.
    Falls back to returning a copy on any error so the renderer never raises.
    """
    try:
        strength = max(0.0, min(1.0, strength))
        src = base if base.mode == "RGBA" else base.convert("RGBA")
        r, g, b, a = src.split()
        # Per-pixel luminance of the original drives the brightness of the
        # tinted result, so highlights stay bright and shadows stay dark.
        lum = src.convert("L")
        tinted_rgb = Image.new("RGB", src.size, rgb)
        # Multiply the flat tint by the luminance ramp -> shaded tint.
        shaded = ImageChops.multiply(
            tinted_rgb, Image.merge("RGB", (lum, lum, lum)))
        orig_rgb = Image.merge("RGB", (r, g, b))
        mixed = Image.blend(orig_rgb, shaded, strength)
        mr, mg, mb = mixed.split()
        return Image.merge("RGBA", (mr, mg, mb, a))
    except Exception:
        return base.copy()


def _draw_speaking_halo(img: "Image.Image", speak_t: float, rgb: tuple) -> None:
    """Draw a soft pulsing ring just inside the canvas edge while speaking.

    A large halo (not a tiny dot) is the point — it reads as "JARVIS is
    talking" even after Windows squashes the icon to 16 px. ``speak_t`` is
    the 0..1 pulse level; at 0 we draw nothing. Mutates ``img`` in place.
    """
    if speak_t <= 0.0:
        return
    try:
        alpha = int(90 + 150 * max(0.0, min(1.0, speak_t)))   # 90..240
        ring = Image.new("RGBA", img.size, (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        w = max(2, int(SIZE * 0.09))            # bold stroke
        inset = max(1, int(SIZE * 0.04))
        rd.ellipse([inset, inset, SIZE - 1 - inset, SIZE - 1 - inset],
                   outline=rgb + (alpha,), width=w)
        # Blur so the ring reads as a glow rather than a hard circle, and so
        # it survives downscaling without aliasing into a dotted line.
        ring = ring.filter(ImageFilter.GaussianBlur(max(1, int(SIZE * 0.03))))
        img.alpha_composite(ring)
    except Exception:
        # A halo is pure polish — never let it break the icon.
        pass


def _draw_queue_badge(img: "Image.Image", count: int, queue_rgb: tuple) -> None:
    """Draw a LARGE bottom-right count badge (dark disc + bright digit).

    Sized at ``BADGE_FRAC`` of the canvas with a near-opaque dark disc behind
    a high-contrast digit so a single character is still readable once the
    shell downscales to 24 px (it simply becomes too small to resolve at
    16 px — an acceptable, graceful degradation). No-op when count <= 0.
    Mutates ``img`` in place.
    """
    if count <= 0:
        return
    try:
        d = ImageDraw.Draw(img)
        bd = max(12, int(SIZE * BADGE_FRAC))
        x = SIZE - bd
        y = SIZE - bd
        # Dark disc with a bright rim in the queue colour -> pops off any base.
        d.ellipse([x, y, x + bd, y + bd], fill=(15, 15, 18, 235),
                  outline=queue_rgb + (255,), width=max(2, int(bd * 0.10)))
        text = str(count) if count < 100 else "99+"
        # One digit gets a big glyph; "99+" needs to be smaller to fit.
        frac = 0.66 if len(text) <= 1 else (0.5 if len(text) == 2 else 0.4)
        font = _get_font(max(8, int(bd * frac)))
        if font is not None:
            try:
                bbox = d.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x + (bd - tw) / 2 - bbox[0]
                ty = y + (bd - th) / 2 - bbox[1]
                d.text((tx, ty), text, fill=queue_rgb + (255,), font=font)
            except Exception:
                pass
    except Exception:
        pass


def _draw_bambu_mark(img: "Image.Image") -> None:
    """Draw a small bold orange print-mark in the TOP-RIGHT corner.

    Only called when a Bambu print is active, so its mere presence is the
    signal (secondary to the listen tint). Kept compact but solid + rimmed
    so it doesn't vanish at small sizes. Mutates ``img`` in place.
    """
    try:
        d = ImageDraw.Draw(img)
        m = max(8, int(SIZE * 0.30))
        x1 = SIZE - m
        y0 = 0
        # Down-pointing triangle (a nozzle laying a line) — distinct from the
        # round queue badge so the two corners never read as the same thing.
        d.polygon([(x1, y0), (SIZE - 1, y0), ((x1 + SIZE - 1) / 2, m)],
                  fill=BAMBU_ORANGE + (255,), outline=(20, 20, 20, 255))
    except Exception:
        pass


def _render_icon_with_base(base: "Image.Image", signals: dict) -> Image.Image:
    """Render the arc-reactor icon with the redesigned status overlays.

    Pipeline: tint the whole reactor by the listen state (primary signal) →
    pulse a speaking halo → stamp the large queue badge (if any) → mark a
    Bambu print (if active). At most two overlays are ever bright at once
    (halo + badge), so the icon stays glanceable rather than busy.
    """
    img = _tint_image(base, signals["listen"], signals["tint_strength"])
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    _draw_speaking_halo(img, signals["speak_t"], signals["speak"])
    if signals["bambu_active"]:
        _draw_bambu_mark(img)
    _draw_queue_badge(img, signals["queue_count"], signals["queue"])
    return img


def _render_reactor_disc(rgb: tuple) -> "Image.Image":
    """Procedural stand-in for the arc-reactor PNG, recoloured to ``rgb``.

    Used when the asset can't be loaded. Mirrors the real design: a glowing
    tinted disc (so the listen-state tint still reads) instead of the legacy
    4-dot grid, keeping the fallback visually consistent with the base path.
    """
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = SIZE / 2
    outer = SIZE * 0.46
    # Outer dark ring -> bright tinted ring -> dim core -> bright centre,
    # echoing the reactor's concentric look so the fallback isn't jarring.
    d.ellipse([c - outer, c - outer, c + outer, c + outer],
              fill=(18, 22, 28, 255))
    r2 = SIZE * 0.40
    d.ellipse([c - r2, c - r2, c + r2, c + r2],
              outline=rgb + (255,), width=max(2, int(SIZE * 0.07)))
    r3 = SIZE * 0.24
    d.ellipse([c - r3, c - r3, c + r3, c + r3],
              fill=_blend((10, 12, 16), rgb, 0.35) + (255,))
    r4 = SIZE * 0.12
    d.ellipse([c - r4, c - r4, c + r4, c + r4],
              fill=_blend(rgb, (255, 255, 255), 0.4) + (255,))
    return img


def _render_icon_procedural(signals: dict) -> Image.Image:
    """Fallback renderer used when the arc-reactor PNG can't be loaded.

    Builds a procedural reactor disc tinted by the listen state, then runs
    the SAME overlay stack as the base path (halo / badge / bambu mark) so
    status reads identically whether or not the asset is present.
    """
    img = _render_reactor_disc(signals["listen"])
    _draw_speaking_halo(img, signals["speak_t"], signals["speak"])
    if signals["bambu_active"]:
        _draw_bambu_mark(img)
    _draw_queue_badge(img, signals["queue_count"], signals["queue"])
    return img


def _render_icon(state: str, frame: int, mic_level: float = 0.0,
                 tts_amplitude: float = 0.0, queue_count: int = 0,
                 muted: bool = False, bambu_active: bool = False) -> Image.Image:
    """Render one tray-icon frame.

    Two modes — picks based on whether the arc-reactor base asset loaded:

      • Base loaded    — tint the arc-reactor PNG by listen state and overlay
                         the speaking halo + queue badge + bambu mark.
      • Base missing   — render a procedural tinted reactor disc and run the
                         same overlay stack, so status still reads with no
                         asset present.

    Returns an RGBA PIL Image suitable for assignment to pystray.Icon.icon.
    Never raises — bad inputs degrade to the fallback rather than crash
    the animation loop (parent watchdog regression risk).
    """
    try:
        signals = _compute_signal_colors(
            state, frame, tts_amplitude, queue_count, muted, bambu_active,
        )
    except Exception:
        # Worst-case: synthesise neutral signals so we still render something.
        signals = {
            "listen": LISTEN_GRAY, "tint_strength": TINT_STRENGTH,
            "speak": SPEAK_DIM, "speak_t": 0.0,
            "queue":  QUEUE_DIM,   "bambu": BAMBU_WHITE,
            "queue_count": 0, "bambu_active": False,
        }

    if _base_icon is not None:
        try:
            return _render_icon_with_base(_base_icon, signals)
        except Exception:
            logging.exception("[tray] base-icon composite failed — falling back")
    try:
        return _render_icon_procedural(signals)
    except Exception:
        # Absolute last resort: a flat tinted square so icon assignment never
        # receives a non-image. Keeps the animation loop alive no matter what.
        logging.exception("[tray] procedural render failed — flat fallback")
        return Image.new("RGBA", (SIZE, SIZE),
                         tuple(signals.get("listen", LISTEN_GRAY)) + (255,))


# ─── Parent-process watchdog ─────────────────────────────────────────────

_parent_pid = [0]
_stop_event = threading.Event()


def _parent_alive() -> bool:
    pid = _parent_pid[0]
    if not pid:
        return True
    if _HAS_PSUTIL:
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _classify_state(state_data: dict) -> dict:
    """Map hud_state.json into the 4-dot grid's render inputs.

    Returns a dict with: state, mic_level, tts_amplitude, muted,
    bambu_active. The animator combines this with the cached
    queue_count (read from jarvis_todo.md on a slower cadence).
    """
    return {
        "state":         str(state_data.get("state") or "").lower(),
        "mic_level":     float(state_data.get("mic_level") or 0.0),
        "tts_amplitude": float(state_data.get("tts_amplitude") or 0.0),
        "muted":         bool(state_data.get("mic_muted")
                              or state_data.get("muted")),
        "bambu_active":  bool(state_data.get("bambu_active")),
    }


# Queue count is recomputed on a slower cadence (every ~2s) so a chatty
# editor saving jarvis_todo.md mid-write doesn't make the icon flicker.
_QUEUE_RECHECK_SECONDS = 2.0
_queue_cache = {"count": 0, "at": 0.0}


def _count_pending_tasks() -> int:
    """Count unchecked '- [ ]' lines in jarvis_todo.md. Cheap (<5ms on a
    160-line file) but we still cache for 2 seconds to keep the animation
    loop allocation-free in steady state."""
    now = time.time()
    if (now - _queue_cache["at"]) < _QUEUE_RECHECK_SECONDS:
        return int(_queue_cache["count"])
    count = 0
    try:
        if os.path.exists(TODO_FILE):
            with open(TODO_FILE, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.lstrip()
                    if s.startswith("- [ ]"):
                        count += 1
    except Exception:
        count = int(_queue_cache["count"])   # keep last good value on read error
    _queue_cache["count"] = count
    _queue_cache["at"]    = now
    return count


# ─── Menu callbacks ──────────────────────────────────────────────────────

def _on_open_hud(icon, item):
    _send_command("open_hud")

def _on_open_logs(icon, item):
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
    except Exception:
        pass
    try:
        os.startfile(LOGS_DIR)   # Windows-only; tray spec is Windows anyway
    except Exception as e:
        print(f"[tray] open logs failed: {e}")

def _on_restart(icon, item):
    _send_command("restart")

def _append_queued_task(text: str) -> None:
    """Append a `- [ ]` entry to jarvis_todo.md (mirrors _act_queue_task)."""
    text = (text or "").strip()
    if not text:
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M")
        entry = f"- [ ] **{ts}** [tray] — {text}\n"
        if not os.path.exists(TODO_FILE):
            with open(TODO_FILE, "w", encoding="utf-8") as f:
                f.write(
                    "# JARVIS Task Queue\n\n"
                    "Things the user wants Claude Code to build, fix, "
                    "or investigate later.\nTick items as you complete "
                    "them; archive when the file gets big.\n\n"
                )
        with open(TODO_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        print(f"[tray] queued: {text[:80]}")
    except Exception as e:
        print(f"[tray] queue task failed: {e}")


# ─── Spawned-dialog lifecycle tracking ──────────────────────────────────────
# Modal dialogs (queue-task, dossier, about, summary) run in short-lived Python
# subprocesses. Before v2.0.23 only the tray icon was stopped on shutdown, so a
# dialog left open orphaned its subprocess — it outlived JARVIS. Track every
# live dialog Popen here and reap them on quit / parent-death. 2026-07-08.
_dialog_procs: "list[subprocess.Popen]" = []
_dialog_procs_lock = threading.Lock()


def _terminate_dialog_procs() -> None:
    """Terminate every still-open spawned dialog subprocess so a modal dialog
    left on screen can't outlive JARVIS. Called on tray quit and when the parent
    JARVIS process is first seen gone. Never raises. 2026-07-08."""
    with _dialog_procs_lock:
        procs = list(_dialog_procs)
        _dialog_procs.clear()
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


def _tracked_dialog_run(args, *, capture_output=False, text=False,
                        timeout=None, creationflags=0):
    """``subprocess.run`` work-alike that registers the child in
    ``_dialog_procs`` for the life of the call, so a shutdown mid-dialog can
    reap the orphan. Returns an object exposing ``.stdout`` / ``.returncode``
    like ``subprocess.run``. On timeout the child is killed (mirrors run's
    contract) before the TimeoutExpired propagates. 2026-07-08."""
    pipe = subprocess.PIPE if capture_output else None
    proc = subprocess.Popen(args, stdout=pipe, stderr=pipe, text=text,
                            creationflags=creationflags)
    with _dialog_procs_lock:
        _dialog_procs.append(proc)
    try:
        out, _err = proc.communicate(timeout=timeout)
    except Exception:
        try:
            proc.kill()
            proc.communicate()
        except Exception:
            pass
        raise
    finally:
        with _dialog_procs_lock:
            try:
                _dialog_procs.remove(proc)
            except ValueError:
                pass
    return types.SimpleNamespace(stdout=out, returncode=proc.returncode)


def _run_queue_task_dialog() -> int:
    """Subprocess entry point: run the tkinter input dialog on THIS
    process's main thread, then print the entered text to stdout.

    Spawned by `_on_queue_task` so the GUI never touches a daemon thread
    in the tray process (tkinter on Windows requires main-thread Tcl)."""
    if not _HAS_TK:
        sys.stderr.write("tkinter not available\n")
        return 2
    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        text = simpledialog.askstring(
            "Queue Task — JARVIS",
            "Describe the task to queue for the next overnight upgrade:",
            parent=root,
        )
    finally:
        try: root.destroy()
        except Exception: pass
    if text and text.strip():
        sys.stdout.write(text.strip())
        sys.stdout.flush()
    return 0


def _on_queue_task(icon, item):
    """Show a small input dialog and append the result to jarvis_todo.md.

    The dialog runs in a separate Python subprocess so tkinter executes
    on that subprocess's main thread — calling `tk.Tk()` from a daemon
    thread on Windows can hang or crash the tray. We still wrap the
    subprocess call in a daemon thread so the pystray menu callback
    returns immediately."""

    def _spawn_and_collect():
        try:
            # CREATE_NO_WINDOW so the subprocess doesn't flash a console
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = _tracked_dialog_run(
                [sys.executable, os.path.abspath(__file__),
                 "--queue-task-dialog"],
                capture_output=True, text=True, timeout=600,
                creationflags=creationflags,
            )
        except Exception as e:
            print(f"[tray] queue task dialog subprocess failed: {e}")
            return
        text = (proc.stdout or "").strip()
        if not text:
            return
        _append_queued_task(text)

    threading.Thread(target=_spawn_and_collect, daemon=True).start()


def _on_pause_listening(icon, item):
    """Toggle JARVIS's standby flag. Sends the inverse command based on the
    current state so a single menu entry can act as a true ✓ toggle."""
    if _is_standby():
        _send_command("force_wake")
    else:
        _send_command("enter_standby")

def _on_mute_tts(icon, item):
    """Toggle TTS mute — JARVIS still thinks/acts but stays silent."""
    _send_command("mute_tts_toggle")

def _on_mute_mic(icon, item):
    """Toggle the microphone mute. Bobert's capture loop drops input while
    muted and mirrors the new flag back to hud_state.mic_muted, which also
    drives the icon's red listen tint."""
    _send_command("mic_mute_toggle")

def _on_ambient_mode(icon, item):
    """Toggle ambient mode (continuous-listen background mode)."""
    _send_command("ambient_mode_toggle")

def _on_force_upgrade(icon, item):
    """Spec verb 'force upgrade now' — kick the overnight engine."""
    _send_command("trigger_overnight")

def _on_shutdown_jarvis(icon, item):
    _send_command("shutdown_jarvis")

# ── Power tools submenu callbacks ────────────────────────────────────────

def _on_stop_pipeline(icon, item):
    _send_command("stop_pipeline")

def _on_force_backup(icon, item):
    _send_command("force_backup")

def _on_reload_skills(icon, item):
    _send_command("reload_skills")

def _on_run_smoke_test(icon, item):
    _send_command("run_smoke_test")

def _on_pause_daemons(icon, item):
    _send_command("pause_daemons_toggle")

def _on_reset_llm_cache(icon, item):
    _send_command("reset_llm_cache")

def _on_open_live_log(icon, item):
    threading.Thread(target=_open_live_log_viewer, daemon=True).start()

def _on_open_crashes(icon, item):
    threading.Thread(target=_open_event_viewer_crashes, daemon=True).start()

def _on_open_changelog(icon, item):
    if os.path.exists(CHANGELOG_FILE):
        _open_path(CHANGELOG_FILE, "CHANGELOG.md")
    else:
        print("[tray] CHANGELOG.md not found")

# ── AI submenu callbacks ────────────────────────────────────────────────

def _on_switch_anthropic(icon, item):
    _send_command("switch_llm", backend="anthropic")

def _on_switch_qwen(icon, item):
    _send_command("switch_llm", backend="qwen2.5:14b")

def _on_switch_llama(icon, item):
    _send_command("switch_llm", backend="llama3.1:8b")

def _on_switch_other_llm(icon, item):
    _send_command("switch_llm_picker")

def _on_toggle_debug_mode(icon, item):
    _send_command("debug_mode_toggle")

def _on_show_llm_stats(icon, item):
    _send_command("show_llm_stats")

def _on_clear_llm_cache(icon, item):
    _send_command("clear_llm_cache")

# ── Audio Controls submenu callbacks ────────────────────────────────────
# Each toggle flips one runtime flag in bobert (audio_master / aec / ns /
# agc). Bobert mirrors the new state back to hud_state.json so the
# checkmark in the menu refreshes on the next right-click. The sub-layer
# toggles still apply only when the master "Audio Processing" toggle is
# on — turning the master off bypasses the processor entirely.

def _on_toggle_audio_processing(icon, item):
    _send_command("audio_processing_toggle")

def _on_toggle_echo_cancel(icon, item):
    _send_command("audio_echo_cancel_toggle")

def _on_toggle_noise_suppress(icon, item):
    _send_command("audio_noise_suppress_toggle")

def _on_toggle_agc(icon, item):
    _send_command("audio_agc_toggle")

# ── Memory submenu callbacks ────────────────────────────────────────────

def _on_open_memory_file(icon, item):
    if os.path.exists(MEMORY_FACTS_FILE):
        _open_path(MEMORY_FACTS_FILE, "facts.json")
    else:
        # Fall back to the legacy single-file location, then to the dir.
        legacy = os.path.join(PROJECT_DIR, "memory.json")
        if os.path.exists(legacy):
            _open_path(legacy, "memory.json")
        else:
            _open_path(os.path.dirname(MEMORY_FACTS_FILE), "memory dir")

def _on_show_dossier(icon, item):
    """Show JARVIS's full known-facts dossier on the user (read-only)."""
    def _spawn():
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            _tracked_dialog_run(
                [sys.executable, os.path.abspath(__file__), "--dossier-dialog"],
                timeout=1800,
                creationflags=creationflags,
            )
        except Exception as e:
            print(f"[tray] dossier dialog subprocess failed: {e}")
    threading.Thread(target=_spawn, daemon=True).start()

def _on_recent_facts(icon, item):
    _send_command("show_recent_facts")

def _on_reset_memory(icon, item):
    _send_command("reset_memory")

def _on_export_memory(icon, item):
    _send_command("export_memory")

def _on_forget_last_hour(icon, item):
    _send_command("forget_last_hour")

# ── Diagnostics submenu callbacks ────────────────────────────────────────

def _on_run_diagnostic(icon, item):
    _send_command("run_diagnostic")

def _on_show_last_diagnostic(icon, item):
    _send_command("show_last_diagnostic")

def _on_test_mic(icon, item):
    _send_command("test_mic")

def _on_test_tts(icon, item):
    _send_command("test_tts")

def _on_test_vision(icon, item):
    _send_command("test_vision")

def _on_test_each_skill(icon, item):
    _send_command("test_each_skill")

def _on_latency_benchmark(icon, item):
    _send_command("latency_benchmark")

# ── Settings submenu callbacks ───────────────────────────────────────────

def _on_settings_voice(icon, item):
    threading.Thread(target=_open_settings_window, args=("voice",),
                     daemon=True).start()

def _on_settings_ai(icon, item):
    threading.Thread(target=_open_settings_window, args=("ai",),
                     daemon=True).start()

def _on_settings_privacy(icon, item):
    threading.Thread(target=_open_settings_window, args=("privacy",),
                     daemon=True).start()

def _on_settings_integrations(icon, item):
    threading.Thread(target=_open_settings_window, args=("integrations",),
                     daemon=True).start()

def _on_settings_advanced(icon, item):
    threading.Thread(target=_open_settings_window, args=("advanced",),
                     daemon=True).start()

# ── About dialog ────────────────────────────────────────────────────────

def _read_release_version() -> str:
    """The shareable RELEASE version — single source of truth (top-level
    VERSION file) that also backs core/version.py, the git tag and the GitHub
    release. Kept SEPARATE from the self-upgrade pipeline's CHANGELOG counter
    below so the About dialog's primary 'Version:' line always matches what
    GitHub shows. Returns 'unknown' if the file is missing (defensive only —
    VERSION is tracked, so a real checkout always has it)."""
    try:
        with open(RELEASE_VERSION_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() or "unknown"
    except Exception:
        return "unknown"


def _read_version_and_upgrade() -> tuple[str, str]:
    """Parse CHANGELOG.md for the latest version + timestamp.

    Header format written by the pipeline runner is:
        ## v1.0.6 — 2026-05-28 22:33
    Returns (version, last_upgrade_at). Falls back to data/version.json so
    the dialog still has something to show if CHANGELOG.md got truncated."""
    version = "unknown"
    upgrade_at = "unknown"
    try:
        if os.path.exists(CHANGELOG_FILE):
            with open(CHANGELOG_FILE, "r", encoding="utf-8",
                      errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("## v") and "—" in s:
                        # "## v1.0.6 — 2026-05-28 22:33"
                        try:
                            after_hash = s[3:].strip()       # "v1.0.6 — ..."
                            ver, _, rest = after_hash.partition("—")
                            version = ver.strip()
                            upgrade_at = rest.strip()
                        except Exception:  # pragma: no cover - defensive; str slice/partition/strip on a matched line cannot raise
                            pass
                        break
    except Exception:
        pass
    if version == "unknown":
        try:
            if os.path.exists(VERSION_FILE):
                with open(VERSION_FILE, "r", encoding="utf-8") as f:
                    vj = json.load(f) or {}
                version = "v" + str(vj.get("version") or "?")
                upgrade_at = str(vj.get("last_upgrade_at") or upgrade_at)
        except Exception:
            pass
    return version, upgrade_at


def _read_uptime_seconds() -> float:
    """Uptime is derived from the prod instance's started_at heartbeat
    (data/instances.json) — that's where bobert_companion.py publishes its
    boot time. Falls back to hud_state.json boot_started_at."""
    try:
        if os.path.exists(INSTANCES_FILE):
            with open(INSTANCES_FILE, "r", encoding="utf-8") as f:
                inst = json.load(f) or {}
            for entry in inst.values():
                if not isinstance(entry, dict):
                    continue
                if entry.get("role") == "prod" and entry.get("started_at"):
                    return max(0.0, time.time() - float(entry["started_at"]))
            # No prod entry — take any entry's started_at.
            for entry in inst.values():
                if isinstance(entry, dict) and entry.get("started_at"):
                    return max(0.0, time.time() - float(entry["started_at"]))
    except Exception:
        pass
    try:
        boot = float(_read_hud_state().get("boot_started_at") or 0.0)
        if boot > 0:
            return max(0.0, time.time() - boot)
    except Exception:
        pass
    return 0.0


def _format_uptime(seconds: float) -> str:
    s = int(max(0.0, seconds))
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    mins, _ = divmod(s, 60)
    if days:
        return f"{days}d {hours}h {mins}m"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _about_lines() -> list[str]:
    release = _read_release_version()
    build, upgrade_at = _read_version_and_upgrade()
    uptime = _format_uptime(_read_uptime_seconds())
    lines = [
        "J.A.R.V.I.S.",
        "",
        f"Version:       {release}",      # matches GitHub + the git tag
    ]
    # The self-upgrade pipeline's internal counter (e.g. v1.0.17) + its
    # timestamp — shown only when there's real upgrade history AND it differs
    # from the release version. A fresh clone (no pipeline runs) just shows the
    # release version, never a confusing 'Upgrade build: unknown'.
    if build and build not in ("unknown", release, f"v{release}"):
        lines.append(f"Upgrade build: {build}")
    if upgrade_at and upgrade_at != "unknown":
        lines.append(f"Last upgrade:  {upgrade_at}")
    lines += [
        f"Uptime:        {uptime}",
        "",
        "Personal AI assistant.",
        "Right-click the tray icon for the full menu.",
    ]
    return lines


def _run_about_dialog() -> int:
    if not _HAS_TK:
        sys.stderr.write("tkinter not available\n")
        return 2
    body = "\n".join(_about_lines())
    root = tk.Tk()
    try:
        root.title("About JARVIS")
        root.attributes("-topmost", True)
        root.geometry("420x260")
        try:
            text = tk.Text(root, wrap="word", font=("Consolas", 11),
                           bg="#0d1117", fg="#c9d1d9", padx=14, pady=12)
            text.pack(fill="both", expand=True)
            text.insert("1.0", body)
            text.configure(state="disabled")
        except Exception:
            tk.Label(root, text=body, justify="left").pack(padx=10, pady=10)
        tk.Button(root, text="OK", command=root.destroy,
                  width=10).pack(pady=(0, 10))
        root.mainloop()
    finally:
        try: root.destroy()
        except Exception: pass
    return 0


def _on_about(icon, item):
    def _spawn():
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            _tracked_dialog_run(
                [sys.executable, os.path.abspath(__file__), "--about-dialog"],
                timeout=1800,
                creationflags=creationflags,
            )
        except Exception as e:
            print(f"[tray] about dialog subprocess failed: {e}")
    threading.Thread(target=_spawn, daemon=True).start()


# ── "Show what JARVIS knows about me" dossier dialog ────────────────────

def _dossier_lines() -> list[str]:
    """Read data/long_term_memory/facts.json and lay it out human-readably.
    This is a read-only dossier — the menu item "Reset Memory" handles
    edits via the bobert action so we don't accidentally drift schemas."""
    lines: list[str] = ["What JARVIS knows about you", ""]
    facts_path = MEMORY_FACTS_FILE
    if not os.path.exists(facts_path):
        legacy = os.path.join(PROJECT_DIR, "memory.json")
        if os.path.exists(legacy):
            facts_path = legacy
        else:
            lines.append("(no memory file found yet)")
            return lines
    try:
        with open(facts_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception as e:
        lines.append(f"(could not read memory: {e})")
        return lines

    # facts.json can be either a list of fact dicts or {"facts": [...]}.
    facts: list = []
    if isinstance(data, list):
        facts = data
    elif isinstance(data, dict):
        if isinstance(data.get("facts"), list):
            facts = data["facts"]
        else:
            # Fall through to a generic key:value dump.
            for k, v in data.items():
                lines.append(f"{k}: {v}")
            return lines

    if not facts:
        lines.append("(no facts learned yet)")
        return lines

    lines.append(f"{len(facts)} fact(s) on file.")
    lines.append("")
    # Show the most-recent ~60; the rest would overflow the dialog.
    for entry in facts[-60:]:
        if isinstance(entry, dict):
            text = (entry.get("text")
                    or entry.get("fact")
                    or entry.get("content")
                    or json.dumps(entry))
        else:
            text = str(entry)
        trimmed = text if len(text) <= 160 else text[:157] + "…"
        lines.append(f"  • {trimmed}")
    return lines


def _run_dossier_dialog() -> int:
    if not _HAS_TK:
        sys.stderr.write("tkinter not available\n")
        return 2
    body = "\n".join(_dossier_lines())
    root = tk.Tk()
    try:
        root.title("What JARVIS Knows About Me")
        root.attributes("-topmost", True)
        root.geometry("680x520")
        try:
            frame = tk.Frame(root, bg="#0d1117")
            frame.pack(fill="both", expand=True)
            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side="right", fill="y")
            text = tk.Text(frame, wrap="word", font=("Consolas", 10),
                           bg="#0d1117", fg="#c9d1d9", padx=10, pady=10,
                           yscrollcommand=scrollbar.set)
            text.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=text.yview)
            text.insert("1.0", body)
            text.configure(state="disabled")
        except Exception:
            tk.Label(root, text=body, justify="left").pack(padx=10, pady=10)
        tk.Button(root, text="OK", command=root.destroy,
                  width=10).pack(pady=(0, 10))
        root.mainloop()
    finally:
        try: root.destroy()
        except Exception: pass
    return 0

def _on_open_todo(icon, item):
    """Open jarvis_todo.md in the user's default markdown editor."""
    if not os.path.exists(TODO_FILE):
        # Create a minimal file so os.startfile doesn't error on a fresh
        # install before any task has been queued.
        try:
            with open(TODO_FILE, "w", encoding="utf-8") as f:
                f.write("# JARVIS Task Queue\n\n")
        except Exception as e:
            print(f"[tray] could not create todo: {e}")
            return
    try:
        os.startfile(TODO_FILE)   # Windows-only; tray spec is Windows anyway
    except Exception as e:
        print(f"[tray] open todo failed: {e}")


def _today_summary_lines() -> list[str]:
    """Build the text shown in the 'Show Today's Summary' dialog.

    Reads today's session_YYYY-MM-DD_*.log files and surfaces:
      • session count + total log size
      • pending vs. completed task counts from jarvis_todo.md
      • last few completed tasks (today only) so the user can see what
        was actually finished in this calendar day
    """
    lines: list[str] = []
    today = time.strftime("%Y-%m-%d")
    lines.append(f"J.A.R.V.I.S. — {today}")
    lines.append("")

    # ── Session activity ──
    try:
        if os.path.isdir(LOGS_DIR):
            sessions = [
                f for f in os.listdir(LOGS_DIR)
                if f.startswith(f"session_{today}_") and f.endswith(".log")
            ]
            total_bytes = 0
            for f in sessions:
                try:
                    total_bytes += os.path.getsize(os.path.join(LOGS_DIR, f))
                except Exception:
                    pass
            kb = total_bytes / 1024
            lines.append(f"Sessions today:   {len(sessions)}  ({kb:.1f} KB logged)")
        else:
            lines.append("Sessions today:   (logs/ not found)")
    except Exception as e:
        lines.append(f"Sessions today:   (error: {e})")

    # ── Task queue health ──
    pending = 0
    completed_today = []
    try:
        if os.path.exists(TODO_FILE):
            with open(TODO_FILE, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.lstrip()
                    if s.startswith("- [ ]"):
                        pending += 1
                    elif s.startswith("- [x]") and today in line:
                        # Strip the leading '- [x] ' so the summary reads cleanly.
                        completed_today.append(s[5:].strip())
    except Exception as e:
        lines.append(f"Todo:             (error: {e})")
    else:
        lines.append(f"Pending tasks:    {pending}")
        lines.append(f"Completed today:  {len(completed_today)}")

    if completed_today:
        lines.append("")
        lines.append("Recent completions:")
        # Show the last 5 — newest at the bottom of the file, so reverse.
        for entry in reversed(completed_today[-5:]):
            # Trim each entry to ~140 chars so the dialog doesn't span the
            # whole screen on a long ✓ DONE summary.
            trimmed = entry if len(entry) <= 140 else entry[:137] + "…"
            lines.append(f"  • {trimmed}")

    return lines


def _run_summary_dialog() -> int:
    """Subprocess entry point: show today's summary in a tkinter window.

    Same pattern as the queue-task dialog — runs in its own subprocess so
    tkinter gets the main thread. Read-only dialog with an OK button.
    """
    if not _HAS_TK:
        sys.stderr.write("tkinter not available\n")
        return 2
    lines = _today_summary_lines()
    body = "\n".join(lines) or "(no activity today)"
    root = tk.Tk()
    try:
        root.title("Today's Summary — JARVIS")
        root.attributes("-topmost", True)
        # Sized to fit ~14 lines of monospaced text; user can scroll if longer.
        root.geometry("560x360")
        try:
            text = tk.Text(root, wrap="word", font=("Consolas", 10),
                           bg="#0d1117", fg="#c9d1d9", padx=10, pady=10)
            text.pack(fill="both", expand=True)
            text.insert("1.0", body)
            text.configure(state="disabled")
        except Exception:
            tk.Label(root, text=body, justify="left").pack(padx=10, pady=10)
        tk.Button(root, text="OK", command=root.destroy,
                  width=10).pack(pady=(0, 10))
        root.mainloop()
    finally:
        try: root.destroy()
        except Exception: pass
    return 0


def _on_show_today_summary(icon, item):
    """Spawn the summary dialog in a subprocess (tkinter wants its own
    main thread, same constraint as the queue-task dialog)."""
    def _spawn():
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            _tracked_dialog_run(
                [sys.executable, os.path.abspath(__file__),
                 "--summary-dialog"],
                timeout=1800,
                creationflags=creationflags,
            )
        except Exception as e:
            print(f"[tray] summary dialog subprocess failed: {e}")

    threading.Thread(target=_spawn, daemon=True).start()


def _on_quit(icon, item):
    _stop_event.set()
    _terminate_dialog_procs()   # reap any open modal dialog before we go. 2026-07-08
    try: icon.stop()
    except Exception: pass


# ─── Animation thread ────────────────────────────────────────────────────

def _animate(icon: "pystray.Icon") -> None:
    frame = 0
    while not _stop_event.is_set():
        try:
            if not _parent_alive():
                print("[tray] parent JARVIS process exited — closing tray")
                _terminate_dialog_procs()   # don't orphan an open dialog. 2026-07-08
                try: icon.stop()
                except Exception: pass
                return
            data = _read_hud_state()
            s = _classify_state(data)
            queue_count = _count_pending_tasks()
            try:
                icon.icon = _render_icon(
                    s["state"],
                    frame,
                    mic_level=s["mic_level"],
                    tts_amplitude=s["tts_amplitude"],
                    queue_count=queue_count,
                    muted=s["muted"],
                    bambu_active=s["bambu_active"],
                )
                # Tooltip enumerates the four signals so the user can sanity-
                # check what each dot is showing without having to remember the
                # palette. Kept terse so Windows' 128-char tooltip cap holds.
                raw = s["state"] or "idle"
                if s["muted"]:
                    listen_label = "muted"
                elif raw in ("standby", "sleeping", "sleep"):
                    listen_label = "standby"
                else:
                    listen_label = "awake"
                speak_label = "speaking" if (raw == "speaking"
                                              or s["tts_amplitude"] > 0.02) else "quiet"
                bambu_label = "printing" if s["bambu_active"] else "idle"
                icon.title = (
                    f"J.A.R.V.I.S. — listen:{listen_label} "
                    f"tts:{speak_label} queue:{queue_count} "
                    f"bambu:{bambu_label}"
                )
            except Exception:
                pass
            frame += 1
            _stop_event.wait(TICK_SECONDS)
        except Exception:
            # Never let a single bad iteration kill the animation thread —
            # log and keep spinning so the tray stays responsive.
            logging.exception("[tray] _animate iteration failed")
            _stop_event.wait(TICK_SECONDS)


# ─── Entry point ─────────────────────────────────────────────────────────

def _on_open_project_folder(icon, item):
    """Open the JARVIS project root in Explorer — handy for poking at
    config files (notification_rules.json, hud_config.json, etc.)."""
    try:
        os.startfile(PROJECT_DIR)
    except Exception as e:
        print(f"[tray] open project folder failed: {e}")


# ── Menu status header ───────────────────────────────────────────────────
# Disabled MenuItems at the top of the menu surface the same 4 signals
# the icon shows, but in words — useful for the "what's that pulsing
# blue dot mean again?" moment. Read at menu-open time (pystray invokes
# the lambdas on each right-click, so they always reflect current state).

def _status_text_listen() -> str:
    data = _read_hud_state()
    raw = str(data.get("state") or "").lower()
    if bool(data.get("mic_muted") or data.get("muted")):
        return "● Listening: muted"
    if raw in ("standby", "sleeping", "sleep"):
        return "● Listening: standby"
    return "● Listening: awake"


def _status_text_tts() -> str:
    data = _read_hud_state()
    raw = str(data.get("state") or "").lower()
    amp = float(data.get("tts_amplitude") or 0.0)
    if raw == "speaking" or amp > 0.02:
        return "● TTS: speaking"
    return "● TTS: quiet"


def _status_text_queue() -> str:
    return f"● Queue: {_count_pending_tasks()} task(s)"


def _status_text_bambu() -> str:
    data = _read_hud_state()
    return "● Bambu: printing" if bool(data.get("bambu_active")) else "● Bambu: idle"


# ── Apple Music tray controls ────────────────────────────────────────────
# JARVIS hosts Apple Music controls in ITS tray because the UWP Apple Music
# app has NO system tray of its own. Transport goes through the SAME command
# IPC as every other tray verb (_send_command -> bobert's drainer -> the
# existing media_playpause / media_next / media_prev / open_apple_music
# ACTIONS, which drive OS media keys + an AUMID launch). We do NOT script the
# app's UI from here — that automation is policy-restricted.
#
# The now-playing LABEL is the one thing read in-process: the tray imports the
# lazy audio.apple_music_app bridge and calls now_playing() (window-title
# parse). That bridge never raises and degrades to None when pygetwindow /
# psutil are missing, so the label still renders ("Apple Music: idle/closed").

def _apple_music_app():
    """Late-bound, best-effort handle to the audio.apple_music_app bridge, or
    None. Imported lazily (not at tray-module import) so a stripped install
    without the audio package — or without the bridge's optional deps — still
    builds the tray; the menu items just degrade to a no-op/'unknown'. Cached on
    the module so repeated menu opens don't re-import. Never raises."""
    cached = globals().get("_apple_music_app_mod")
    if cached is not None:
        return cached if cached is not _AM_UNAVAILABLE else None
    mod = sys.modules.get("audio.apple_music_app")
    if mod is None:
        try:
            from audio import apple_music_app as mod  # type: ignore
        except Exception:
            globals()["_apple_music_app_mod"] = _AM_UNAVAILABLE
            return None
    globals()["_apple_music_app_mod"] = mod
    return mod


# Sentinel so a failed import is remembered (and not retried every menu open)
# without colliding with a genuine None "not looked up yet".
_AM_UNAVAILABLE = object()


def _status_text_apple_music() -> str:
    """Dynamic now-playing label for the Apple Music submenu header. Shows the
    track when the app is playing, 'idle' when it's running but quiet, and
    'closed' when it isn't running (or the bridge/deps are unavailable). pystray
    re-evaluates this lambda on every right-click, so it always reflects current
    state. Never raises — any failure degrades to the closed label."""
    # Prefer the OS media session (SMTC): source-agnostic and reliable, and it
    # names the real track from Chrome / Spotify / the Apple Music app / YouTube
    # instead of the window-title bridge's useless "Apple Music: Apple Music".
    # Falls through to the bridge below when nothing is reporting to SMTC.
    try:
        from core.media_now_playing import now_playing_text as _smtc_np
        _np = _smtc_np()
        if _np:
            return f"♪ {_np}"
    except Exception:
        pass
    amapp = _apple_music_app()
    if amapp is None:
        return "Apple Music: unavailable"
    try:
        running = amapp.is_running()
    except Exception:
        running = False
    if not running:
        return "Apple Music: closed"
    try:
        np = amapp.now_playing()
    except Exception:
        np = None
    if np:
        # Keep the menu row from spanning the screen on a long title.
        np = np if len(np) <= 60 else np[:57].rstrip() + "…"
        return f"Apple Music: {np}"
    return "Apple Music: idle"


def _on_apple_music_playpause(icon, item):
    """Play/Pause the Apple Music app via the OS media-key path. Routes through
    the same command IPC as the rest of the tray: bobert's drainer dispatches
    `media_playpause` to ACTIONS['media_playpause'] (-> _media_key_with_focus)."""
    _send_command("media_playpause")


def _on_apple_music_next(icon, item):
    """Next track via the OS media-key path (ACTIONS['media_next'])."""
    _send_command("media_next")


def _on_apple_music_prev(icon, item):
    """Previous track via the OS media-key path (ACTIONS['media_prev'])."""
    _send_command("media_prev")


def _on_open_apple_music(icon, item):
    """Open / focus the Apple Music app. Routes `open_apple_music` to bobert,
    whose ACTIONS['open_apple_music'] launches it via its AUMID (it no-ops with a
    friendly line if already running). Sending the command (rather than
    launching from the tray subprocess directly) keeps a single launch owner and
    matches every other tray verb."""
    _send_command("open_apple_music")


def _is_standby() -> bool:
    data = _read_hud_state()
    raw = str(data.get("state") or "").lower()
    return raw in ("standby", "sleeping", "sleep")


# ── Toggle state helpers ─────────────────────────────────────────────────
# All read hud_state.json. The fields below are written by bobert_companion.py
# when it processes the matching tray command (mute_tts_toggle,
# ambient_mode_toggle, …). Until those backend handlers exist the field will
# simply be absent and the toggle stays unchecked — that's intentional, the
# tray must not assume the backend supports a feature before it lands.

def _is_listen_paused() -> bool:
    return _is_standby()


def _is_tts_muted() -> bool:
    return bool(_read_hud_state().get("tts_muted"))


def _is_mic_muted() -> bool:
    """Mic-mute checkmark source. bobert publishes hud_state.mic_muted when it
    processes the mic_mute_toggle command; absent until then -> unchecked."""
    return bool(_read_hud_state().get("mic_muted"))


def _is_ambient_mode() -> bool:
    return bool(_read_hud_state().get("ambient_mode_active"))


def _is_debug_mode() -> bool:
    return bool(_read_hud_state().get("debug_mode"))


def _is_daemons_paused() -> bool:
    return bool(_read_hud_state().get("daemons_paused"))


def _active_llm_backend() -> str:
    """Returns 'anthropic' / 'qwen' / 'llama' / 'other' or '' if unknown."""
    return str(_read_hud_state().get("llm_backend") or "").lower()


# Audio Controls submenu readers — default to True (on) when the field is
# absent so a fresh hud_state.json (or one written by an older bobert that
# doesn't publish audio_* flags yet) still shows the pipeline as enabled,
# matching the actual default of AUDIO_PROCESSING_ENABLED = True.
def _audio_field(name: str) -> bool:
    data = _read_hud_state()
    if name not in data:
        return True
    return bool(data.get(name))


def _is_audio_processing_enabled() -> bool:
    return _audio_field("audio_processing_enabled")


def _is_echo_cancel_enabled() -> bool:
    return _audio_field("echo_cancel_enabled")


def _is_noise_suppress_enabled() -> bool:
    return _audio_field("noise_suppress_enabled")


def _is_agc_enabled() -> bool:
    return _audio_field("agc_enabled")


def _is_pipeline_running() -> bool:
    """Detect an in-flight overnight/upgrade pipeline. We accept either the
    documented sentinel (pipeline_lock.json) or the existing .overnight_active
    flag bobert already writes today."""
    try:
        return os.path.exists(PIPELINE_LOCK_FILE) or os.path.exists(OVERNIGHT_FLAG)
    except Exception:
        return False


def _open_path(path: str, label: str = "") -> None:
    """Best-effort Windows-shell open. Used by every 'Open X' menu item."""
    try:
        os.startfile(path)
    except Exception as e:
        print(f"[tray] open {label or path} failed: {e}")


def _open_event_viewer_crashes() -> None:
    """Open Windows Event Viewer filtered for python.exe APPCRASH.
    Falls back to the Application log root if the filtered MSC isn't there."""
    try:
        # eventvwr.msc opens to the Application log; user can pivot.
        # os.startfile uses the shell association for .msc (mmc.exe) and is
        # more robust than Popen(..., shell=True), which with a list arg
        # only honours argv[0].
        os.startfile("eventvwr.msc")
    except Exception as e:
        print(f"[tray] open event viewer failed: {e}")


def _open_live_log_viewer() -> None:
    """Spawn _show_log.ps1 in a visible PowerShell window."""
    if not os.path.exists(SHOW_LOG_PS1):
        print(f"[tray] live log viewer not found: {SHOW_LOG_PS1}")
        return
    try:
        # New visible PowerShell window so the user actually sees the log.
        subprocess.Popen(
            ["powershell.exe", "-NoExit",
             "-ExecutionPolicy", "Bypass",
             "-File", SHOW_LOG_PS1],
            close_fds=True,
        )
    except Exception as e:
        print(f"[tray] live log spawn failed: {e}")


def _open_settings_window(tab: str = "") -> None:
    """Open tools/settings_window.py (from tray-overhaul-1.F). If the file
    isn't there yet — that task hasn't shipped — fall through to opening the
    raw user_settings.json so the user can still poke values by hand."""
    if os.path.exists(SETTINGS_WINDOW):
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            args = [sys.executable, SETTINGS_WINDOW]
            if tab:
                args += ["--tab", tab]
            subprocess.Popen(args, creationflags=creationflags, close_fds=True)
            return
        except Exception as e:
            print(f"[tray] settings window spawn failed: {e}")
    # Fallback: open the JSON directly so settings remain user-editable.
    fallback = os.path.join(DATA_DIR, "user_settings.json")
    if os.path.exists(fallback):
        _open_path(fallback, "user_settings.json")
    else:
        print("[tray] settings window not installed and no user_settings.json"
              " — install tools/settings_window.py to enable Settings menu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="JARVIS parent PID — tray exits if this dies")
    parser.add_argument("--icon-path", type=str, default=DEFAULT_ICON_PATH,
                        help="Path to the arc-reactor PNG. Falls back to a "
                             "procedural 4-dot grid if missing.")
    parser.add_argument("--queue-task-dialog", action="store_true",
                        help=argparse.SUPPRESS)  # internal: tk dialog mode
    parser.add_argument("--summary-dialog", action="store_true",
                        help=argparse.SUPPRESS)  # internal: tk summary dialog
    parser.add_argument("--about-dialog", action="store_true",
                        help=argparse.SUPPRESS)  # internal: About JARVIS dialog
    parser.add_argument("--dossier-dialog", action="store_true",
                        help=argparse.SUPPRESS)  # internal: facts dossier dialog
    args = parser.parse_args()
    if args.queue_task_dialog:
        sys.exit(_run_queue_task_dialog())
    if args.summary_dialog:
        sys.exit(_run_summary_dialog())
    if args.about_dialog:
        sys.exit(_run_about_dialog())
    if args.dossier_dialog:
        sys.exit(_run_dossier_dialog())
    _parent_pid[0] = args.parent_pid
    global _icon_path
    _icon_path = args.icon_path
    _load_base_icon(_icon_path)

    # ── Submenu: Power tools ──
    # Stop Running Pipeline is greyed when no upgrade is mid-flight so the
    # menu doesn't promise a no-op. Other items always fire; bobert handles
    # the actual work (or returns a no-op for not-yet-built ones).
    power_menu = pystray.Menu(
        pystray.MenuItem("Stop Running Pipeline",  _on_stop_pipeline,
                         enabled=lambda i: _is_pipeline_running()),
        pystray.MenuItem("Force Backup Now",       _on_force_backup),
        pystray.MenuItem("Reload All Skills",      _on_reload_skills),
        pystray.MenuItem("Run Smoke Test",         _on_run_smoke_test),
        pystray.MenuItem("Pause All Daemons",      _on_pause_daemons,
                         checked=lambda i: _is_daemons_paused()),
        pystray.MenuItem("Reset Local LLM Cache",  _on_reset_llm_cache),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open JARVIS Folder",     _on_open_project_folder),
        pystray.MenuItem("Open Task Queue",        _on_open_todo),
        pystray.MenuItem("Open Live Log Viewer",   _on_open_live_log),
        pystray.MenuItem("Open Crash Reports",     _on_open_crashes),
        pystray.MenuItem("Open Changelog",         _on_open_changelog),
    )

    # ── Submenu: AI ──
    # Checkmarks reflect hud_state.llm_backend so bobert remains the source
    # of truth on which backend is actually serving requests.
    ai_menu = pystray.Menu(
        pystray.MenuItem("Switch to Anthropic Claude",       _on_switch_anthropic,
                         checked=lambda i: _active_llm_backend() == "anthropic"),
        pystray.MenuItem("Switch to Local LLM (qwen2.5:14b)", _on_switch_qwen,
                         checked=lambda i: _active_llm_backend().startswith("qwen")),
        pystray.MenuItem("Switch to Local LLM (llama3.1:8b)", _on_switch_llama,
                         checked=lambda i: _active_llm_backend().startswith("llama")),
        pystray.MenuItem("Switch to Local LLM (other…)",     _on_switch_other_llm),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Debug Mode",                       _on_toggle_debug_mode,
                         checked=lambda i: _is_debug_mode()),
        pystray.MenuItem("Show LLM Call Stats",              _on_show_llm_stats),
        pystray.MenuItem("Clear LLM Cache",                  _on_clear_llm_cache),
    )

    # ── Submenu: Audio Controls ──
    # Mirrors the AI submenu pattern — top-level toggles with checkmarks
    # bound to hud_state fields that bobert mirrors on every command.
    # Sub-layer toggles (echo/NS/AGC) are greyed when the master switch
    # is off so the menu doesn't promise per-layer control that bobert
    # would short-circuit anyway. Settings > Voice/Audio remains the
    # detailed panel for thresholds and device pickers.
    audio_menu = pystray.Menu(
        pystray.MenuItem("Audio Processing",   _on_toggle_audio_processing,
                         checked=lambda i: _is_audio_processing_enabled()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Echo Cancellation",  _on_toggle_echo_cancel,
                         checked=lambda i: _is_echo_cancel_enabled(),
                         enabled=lambda i: _is_audio_processing_enabled()),
        pystray.MenuItem("Noise Suppression",  _on_toggle_noise_suppress,
                         checked=lambda i: _is_noise_suppress_enabled(),
                         enabled=lambda i: _is_audio_processing_enabled()),
        pystray.MenuItem("Gain Normalization", _on_toggle_agc,
                         checked=lambda i: _is_agc_enabled(),
                         enabled=lambda i: _is_audio_processing_enabled()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Voice / Audio Settings…", _on_settings_voice),
    )

    # ── Submenu: Apple Music ──
    # JARVIS hosts these because the UWP Apple Music app has no tray of its own.
    # A dynamic now-playing header (re-read on each menu open) sits above the
    # transport verbs. Transport uses OS media keys (via the command IPC), and
    # "Open Apple Music" launches the app by AUMID — no UI scripting of the app.
    apple_music_menu = pystray.Menu(
        pystray.MenuItem(lambda i: _status_text_apple_music(), None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Play / Pause", _on_apple_music_playpause),
        pystray.MenuItem("Next",         _on_apple_music_next),
        pystray.MenuItem("Previous",     _on_apple_music_prev),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open Apple Music", _on_open_apple_music),
    )

    # ── Submenu: Memory ──
    memory_menu = pystray.Menu(
        pystray.MenuItem("Open Long-Term Memory",            _on_open_memory_file),
        pystray.MenuItem("Show What JARVIS Knows About Me…", _on_show_dossier),
        pystray.MenuItem("Recent Facts Learned (last 24h)",  _on_recent_facts),
        pystray.MenuItem("Reset Memory…",                    _on_reset_memory),
        pystray.MenuItem("Export Memory (JSON/CSV)",         _on_export_memory),
        pystray.MenuItem("Forget Last Hour",                 _on_forget_last_hour),
    )

    # ── Submenu: Diagnostics ──
    diag_menu = pystray.Menu(
        pystray.MenuItem("Run Diagnostic Now",       _on_run_diagnostic),
        pystray.MenuItem("Show Last Diagnostic Run", _on_show_last_diagnostic),
        pystray.MenuItem("Test Mic",                 _on_test_mic),
        pystray.MenuItem("Test TTS",                 _on_test_tts),
        pystray.MenuItem("Test Vision",              _on_test_vision),
        pystray.MenuItem("Test Each Skill",          _on_test_each_skill),
        pystray.MenuItem("Latency Benchmark",        _on_latency_benchmark),
    )

    # ── Submenu: Settings ──
    # Each tab spawns tools/settings_window.py with --tab <name>; that script
    # comes from tray-overhaul-1.F. If it's missing the helper falls back to
    # opening data/user_settings.json so the user can still adjust settings.
    settings_menu = pystray.Menu(
        pystray.MenuItem("Voice / Audio",   _on_settings_voice),
        pystray.MenuItem("AI / Models",     _on_settings_ai),
        pystray.MenuItem("Privacy / Ambient", _on_settings_privacy),
        pystray.MenuItem("Integrations",    _on_settings_integrations),
        pystray.MenuItem("Advanced",        _on_settings_advanced),
    )

    menu = pystray.Menu(
        # ── status header (read-only) ──
        pystray.MenuItem(lambda i: _status_text_listen(), None, enabled=False),
        pystray.MenuItem(lambda i: _status_text_tts(),    None, enabled=False),
        pystray.MenuItem(lambda i: _status_text_queue(),  None, enabled=False),
        pystray.MenuItem(lambda i: _status_text_bambu(),  None, enabled=False),
        pystray.Menu.SEPARATOR,
        # ── frequent toggles ──
        pystray.MenuItem("Pause Listening", _on_pause_listening,
                         checked=lambda i: _is_listen_paused()),
        pystray.MenuItem("Mute TTS",        _on_mute_tts,
                         checked=lambda i: _is_tts_muted()),
        pystray.MenuItem("Mute Mic",        _on_mute_mic,
                         checked=lambda i: _is_mic_muted()),
        pystray.MenuItem("Ambient Mode",    _on_ambient_mode,
                         checked=lambda i: _is_ambient_mode()),
        pystray.Menu.SEPARATOR,
        # ── core lifecycle verbs ──
        pystray.MenuItem("Open HUD",        _on_open_hud),
        pystray.MenuItem("Run Upgrade Now", _on_force_upgrade),
        pystray.MenuItem("Restart JARVIS",  _on_restart),
        pystray.MenuItem("Shut Down JARVIS", _on_shutdown_jarvis),
        pystray.Menu.SEPARATOR,
        # ── power-user submenus ──
        pystray.MenuItem("Power tools", power_menu),
        pystray.MenuItem("AI",          ai_menu),
        pystray.MenuItem("Audio",       audio_menu),
        pystray.MenuItem("Apple Music", apple_music_menu),
        pystray.MenuItem("Memory",      memory_menu),
        pystray.MenuItem("Diagnostics", diag_menu),
        pystray.MenuItem("Settings",    settings_menu),
        pystray.Menu.SEPARATOR,
        # ── bottom: info + queue actions + quit ──
        pystray.MenuItem("About JARVIS",         _on_about),
        pystray.MenuItem("Show Today's Summary", _on_show_today_summary),
        pystray.MenuItem("Queue Task…",          _on_queue_task),
        pystray.MenuItem("Quit Tray Only",       _on_quit),
    )

    # Boot icon: render once with empty state so the tray has something to
    # show before the first animate() tick arrives.
    icon = pystray.Icon(
        "jarvis-tray",
        icon=_render_icon("idle", 0, queue_count=_count_pending_tasks()),
        title="J.A.R.V.I.S.",
        menu=menu,
    )

    anim = threading.Thread(target=_animate, args=(icon,), daemon=True)
    anim.start()

    print(f"[tray] started (parent pid {_parent_pid[0] or 'unknown'})")
    try:
        icon.run()
    finally:
        _stop_event.set()
        print("[tray] exited")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint; never run under unittest
    main()
