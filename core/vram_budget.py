"""Predictive VRAM budget — a "graphics-settings estimator" for JARVIS.

Like a game's video-settings screen that shows how much VRAM your chosen
preset will eat and turns the bar red when it won't fit, this estimates the
TOTAL GPU memory the *currently selected* model + feature settings WILL load,
so the Settings GUI can warn BEFORE you save a combination that over-commits the
card and wedges Ollama (the dreaded 32B-plus-vision "brick").

Calibration (MEASURED on an RTX 3090, 2026-06 — loaded VRAM = model weights +
KV cache at JARVIS's working context, not just the on-disk file):

    gemma4:26b-a4b-it-qat         ~16.0 GB   (16k ctx, measured 15 + margin;
                                              MULTIMODAL — doubles as the
                                              vision model at no extra cost;
                                              ~110 tok/s on the 3090)
    qwen3:30b-a3b-…-q4_K_M        ~21.0 GB   (12k ctx; MoE)
    qwen2.5:32b-instruct-q4_K_M   ~22.0 GB   (12k ctx; wedges if vision co-loads)
    qwen2.5:14b-instruct-q5_K_M   ~13.0 GB   (16k ctx)
    llama3.1:8b-instruct-q5_K_M    ~6.0 GB
    qwen2.5vl:7b  (vision)         ~7.3 GB    (on-demand — loads when seeing)
    large-v3-turbo (Whisper STT)   ~1.5 GB    (always, while listening)
    nomic-embed-text (RAG embed)   ~0.3 GB    (when RAG indexes / queries)

Total card capacity is read once from ``nvidia-smi
--query-gpu=memory.total`` (24576 MiB on the 3090); a ~1.5 GB headroom is held
back so the prediction's "budget" is the usable ceiling, not the raw total.

Design contract (mirrors core/gpu_state.py): stdlib only, import-light, and
TOTAL — every public function degrades gracefully (missing nvidia-smi / ollama,
a non-NVIDIA host, a hand-broken settings dict) and NEVER raises back to the
caller. A diagnostic helper must not be able to crash the Settings window.
"""
from __future__ import annotations

import subprocess
import time
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────
#  Constants — calibrated anchors (megabytes)
# ──────────────────────────────────────────────────────────────────────────
# Loaded VRAM in MB (model weights + KV cache at JARVIS context), keyed by the
# Ollama tag. These are the ground-truth measurements above — the table the
# estimator trusts before it ever falls back to a heuristic. Keys are matched
# case-insensitively and by prefix (so ``qwen2.5:32b-instruct-q4_K_M`` also
# matches a future ``…-q4_K_M-something`` retag).
_GB = 1024  # MB per GB (binary, to line up with nvidia-smi's MiB)

CALIBRATED_VRAM_MB: dict[str, int] = {
    "gemma4:12b":                  9 * _GB,     # 8.4 GB measured @16k ctx
                                                # (2026-07-10 bake-off) + margin;
                                                # multimodal — vision re-uses it
    "gemma4:latest":               4 * _GB,     # E4B ~3.4 GB measured @8k ctx
    "gemma4:26b-a4b-it-qat":       16 * _GB,    # ~15 GB measured @16k ctx (2026-07,
                                                # +~0.4 GB during a vision call —
                                                # 16 keeps a little margin); multimodal
                                                # (BROKEN quant — returns empty; kept
                                                # only so the estimator prices it)
    "qwen3:30b-a3b-instruct-2507-q4_K_M": 21 * _GB,  # ~21 GB (MoE, 12k ctx window)
    "qwen2.5:32b-instruct-q4_K_M": 22 * _GB,    # ~22.0 GB
    "qwen2.5:14b-instruct-q5_K_M": 13 * _GB,    # ~13.0 GB
    "qwen3:14b":                   12 * _GB,    # 11.8 GB measured @16k ctx
    "llama3.1:8b-instruct-q5_K_M": 6 * _GB,     # ~6.0 GB
    "qwen2.5vl:7b":                int(7.3 * _GB),  # ~7.3 GB (vision, on-demand)
    "large-v3-turbo":              int(1.5 * _GB),  # ~1.5 GB (Whisper, always)
    "nomic-embed-text":            int(0.3 * _GB),  # ~0.3 GB (RAG embeddings)
}

# Fixed component sizes referenced by predict_budget (megabytes).
VISION_VRAM_MB = int(7.3 * _GB)     # qwen2.5vl:7b on-demand peak
WHISPER_VRAM_MB = int(1.5 * _GB)    # large-v3-turbo, always while listening
EMBED_VRAM_MB = int(0.3 * _GB)      # nomic-embed-text, RAG only

# The 3090's raw capacity — used only when nvidia-smi can't be read.
DEFAULT_TOTAL_MB = 24576            # 24 GB (24576 MiB), the calibration card
# Held-back headroom so a "fits" prediction leaves room for the desktop / driver
# / fragmentation rather than packing to the literal last megabyte.
HEADROOM_MB = int(1.5 * _GB)       # ~1.5 GB reserve

# Default chat model when settings don't name one (matches core/config.py).
_DEFAULT_CHAT_MODEL = "gemma4:12b"

# KV-cache / context allowance added to a disk-size estimate for an UNKNOWN
# chat tag (the on-disk blob is weights only; the live load also holds the KV
# cache). A flat allowance keeps the heuristic total and dependency-free.
_UNKNOWN_KV_ALLOWANCE_MB = int(1.5 * _GB)
# Disk→VRAM inflation for an unknown tag: quantised weights unpack slightly
# larger in VRAM than the compressed blob on disk.
_DISK_TO_VRAM_FACTOR = 1.15

# Tags that are NOT chat models (vision / embedding) — used by the param-count
# fallback to pick a sane default when a tag is wholly unrecognised.
_VISION_MARKERS = ("vl:", "-vl", "vision", "llava", "moondream", "bakllava")
_EMBED_MARKERS = ("embed", "nomic", "bge-", "minilm")


# ──────────────────────────────────────────────────────────────────────────
#  Hardware probe — total VRAM (cached briefly)
# ──────────────────────────────────────────────────────────────────────────
# Cache the nvidia-smi total so a live-recomputing GUI slider doesn't shell out
# on every keystroke. (total_mb, monotonic_deadline)
_TOTAL_CACHE: list = [None, 0.0]
_TOTAL_CACHE_TTL_S = 30.0

# Cache for `ollama list` disk sizes (parsed name→MB), same brief TTL.
_OLLAMA_CACHE: list = [None, 0.0]
_OLLAMA_CACHE_TTL_S = 30.0


def _run(cmd: list[str], timeout: float = 2.0) -> Optional[str]:
    """Run ``cmd`` and return stdout, or None on ANY failure (binary missing,
    timeout, non-zero exit). Never raises — the estimator must survive a host
    with no nvidia-smi / ollama on PATH."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout or ""


def total_vram_mb(force: bool = False) -> int:
    """Total GPU VRAM in MB via ``nvidia-smi --query-gpu=memory.total``.

    Falls back to ``DEFAULT_TOTAL_MB`` (the 24 GB calibration card) when
    nvidia-smi is absent or unparseable, so the GUI always has a ceiling to draw
    against. Result is cached for a few seconds (``force=True`` bypasses) so a
    live-updating slider never shells out per keystroke. Never raises."""
    now = time.monotonic()
    if not force and _TOTAL_CACHE[0] is not None and now < _TOTAL_CACHE[1]:
        return _TOTAL_CACHE[0]
    total = DEFAULT_TOTAL_MB
    out = _run(["nvidia-smi", "--query-gpu=memory.total",
                "--format=csv,noheader,nounits"])
    if out:
        # Multi-GPU hosts print one line per card; take the first (JARVIS pins
        # GPU 0). Strip a stray "MiB" unit if a driver emits it despite nounits.
        for line in out.splitlines():
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                try:
                    val = int(digits)
                    if val > 0:
                        total = val
                        break
                except ValueError:
                    pass
    _TOTAL_CACHE[0] = total
    _TOTAL_CACHE[1] = now + _TOTAL_CACHE_TTL_S
    return total


def _ollama_disk_sizes() -> dict[str, int]:
    """Map of Ollama tag → on-disk size in MB, parsed from ``ollama list``.

    Used only to ESTIMATE an unknown tag's VRAM (the calibrated table is
    preferred). Returns {} when ollama is absent / unparseable. Cached briefly.
    Never raises."""
    now = time.monotonic()
    if _OLLAMA_CACHE[0] is not None and now < _OLLAMA_CACHE[1]:
        return _OLLAMA_CACHE[0]
    sizes: dict[str, int] = {}
    out = _run(["ollama", "list"])
    if out:
        for line in out.splitlines():
            parts = line.split()
            # Rows look like: NAME  ID  SIZE  UNIT  MODIFIED…  e.g.
            #   qwen2.5:14b-instruct-q5_K_M  7bb3f324cafc  10 GB  7 days ago
            # Skip the header row and anything without a "<num> <unit>" size.
            if len(parts) < 4 or parts[0].upper() == "NAME":
                continue
            name = parts[0]
            mb = _parse_size_to_mb(parts[2], parts[3])
            if mb:
                sizes[name] = mb
    _OLLAMA_CACHE[0] = sizes
    _OLLAMA_CACHE[1] = now + _OLLAMA_CACHE_TTL_S
    return sizes


def _parse_size_to_mb(num: str, unit: str) -> Optional[int]:
    """Convert an ``ollama list`` size like ("10","GB") / ("274","MB") to MB.
    Returns None if it doesn't look like a size."""
    try:
        value = float(num)
    except (TypeError, ValueError):
        return None
    u = (unit or "").strip().upper()
    if u.startswith("G"):
        return int(value * _GB)
    if u.startswith("M"):
        return int(value)
    if u.startswith("T"):
        return int(value * _GB * _GB)
    if u.startswith("K"):
        return int(value / _GB) or 1
    return None


# ──────────────────────────────────────────────────────────────────────────
#  Per-model estimate
# ──────────────────────────────────────────────────────────────────────────
def _calibrated_lookup(model_tag: str) -> Optional[int]:
    """Exact (case-insensitive) or prefix match against CALIBRATED_VRAM_MB."""
    if not model_tag:
        return None
    tag = model_tag.strip().lower()
    for known, mb in CALIBRATED_VRAM_MB.items():
        kl = known.lower()
        if tag == kl or tag.startswith(kl) or kl.startswith(tag):
            return mb
    return None


def _param_count_heuristic(model_tag: str) -> int:
    """Last-ditch VRAM guess from a tag's parameter-count hint (e.g. "14b",
    "7b", "70b") when neither the table nor ollama can help. Assumes a ~q4/q5
    quant (~0.7 GB VRAM per billion params loaded) plus the KV allowance. A tag
    with no parseable size defaults to a mid 7B-class estimate."""
    tag = (model_tag or "").lower()
    billions = 0.0
    # Find a "<number>b" token (the param-count marker in Ollama tags).
    token = ""
    for ch in tag:
        if ch.isdigit() or ch == ".":
            token += ch
        elif ch == "b" and token:
            try:
                billions = float(token)
            except ValueError:
                billions = 0.0
            break
        else:
            token = ""
    if billions <= 0:
        billions = 7.0  # unknown → assume a 7B-class model
    per_b = 0.7 * _GB   # ~0.7 GB VRAM per billion params at q4/q5
    return int(billions * per_b) + _UNKNOWN_KV_ALLOWANCE_MB


def model_vram_estimate(model_tag: str) -> int:
    """Estimated LOADED VRAM (MB) for one Ollama model tag.

    Resolution order, most-trusted first:
      1. the calibrated table (exact / prefix match) — measured ground truth;
      2. the tag's on-disk size from ``ollama list`` × inflation + KV allowance;
      3. a parameter-count heuristic (…"14b"…) when ollama is unavailable too.
    Never raises; an empty/None tag returns 0."""
    if not model_tag:
        return 0
    hit = _calibrated_lookup(model_tag)
    if hit is not None:
        return hit
    # Unknown tag → try the on-disk size (matched exactly, then by prefix).
    sizes = _ollama_disk_sizes()
    disk_mb = sizes.get(model_tag)
    if disk_mb is None:
        tl = model_tag.strip().lower()
        for name, mb in sizes.items():
            nl = name.lower()
            if tl == nl or nl.startswith(tl) or tl.startswith(nl):
                disk_mb = mb
                break
    if disk_mb:
        return int(disk_mb * _DISK_TO_VRAM_FACTOR) + _UNKNOWN_KV_ALLOWANCE_MB
    # Nothing on disk either → param-count heuristic.
    return _param_count_heuristic(model_tag)


# ──────────────────────────────────────────────────────────────────────────
#  Settings → budget prediction
# ──────────────────────────────────────────────────────────────────────────
def _truthy(val) -> bool:
    """Lenient bool read for a settings value that may be a real bool, an int,
    or a string ("true"/"1"/"on"). Mirrors settings_window.coerce_value."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on", "y")
    return False


def _vision_route(settings: dict) -> str:
    """The configured vision route ('auto' | 'local' | 'cloud'), read from a
    MODEL_ROUTING dict OR a flattened ``MODEL_ROUTING::vision`` key (the form the
    Tk routing combobox vars use). Defaults to 'auto'."""
    routing = settings.get("MODEL_ROUTING")
    if isinstance(routing, dict):
        v = routing.get("vision")
        if v:
            return str(v)
    flat = settings.get("MODEL_ROUTING::vision")
    if flat:
        return str(flat)
    return "auto"


def _rag_enabled(settings: dict) -> bool:
    """Whether RAG (embeddings) should count toward the budget. Accepts any of
    the keys the GUI / config might use for the flag."""
    for key in ("RAG_ENABLED", "RAG_AUTOSTART", "PERSONAL_RAG_ENABLED"):
        if key in settings:
            return _truthy(settings.get(key))
    return False


def predict_budget(settings: dict, total_mb: Optional[int] = None) -> dict:
    """Predict the TOTAL VRAM the given settings will load, as a structured
    breakdown the GUI can render into a bar + warning.

    ``settings`` is the live widget/config dict; the keys consulted are:
      LOCAL_LLM_MODEL        — the always-on local chat brain (baseline load)
      MODEL_ROUTING.vision   — 'local'/'auto' co-load the VLM; 'cloud' does not
      LOCAL_VISION_FALLBACK  — also pulls the VLM in as an on-demand fallback
      SCREEN_VISION_ENABLED  — gates whether vision can run at all
      RAG_ENABLED (or RAG_AUTOSTART) — adds the embedding model
      KINECT_ENABLED         — sensor/skeleton (no GPU model; informational)

    Components are summed into ``total_mb`` as a WORST-CASE peak: on-demand
    vision is included because it CAN co-load with the chat model (that co-load
    is exactly the over-commit this estimator exists to flag). The returned
    ``budget_mb`` is the card total minus headroom, and ``over`` is True when the
    predicted peak exceeds it. Never raises; a malformed ``settings`` degrades to
    the chat-model baseline.

    Returns a dict:
      {"components": [{"label","mb","ondemand"}...],
       "total_mb", "budget_mb", "total_card_mb", "headroom_mb",
       "over": bool, "pct": float}
    """
    if not isinstance(settings, dict):
        settings = {}

    components: list[dict] = []

    # 1) Local chat model — always counted (it's the baseline/fallback brain
    #    that stays resident). Default to config's 32B when unset.
    chat_tag = str(settings.get("LOCAL_LLM_MODEL") or _DEFAULT_CHAT_MODEL).strip()
    chat_mb = model_vram_estimate(chat_tag) if chat_tag else 0
    components.append({
        "label": _short_model_label(chat_tag),
        "mb": chat_mb,
        "ondemand": False,
    })

    # 2) Vision VLM — counts when it WOULD run locally: the vision route is
    #    local/auto, OR the local-vision fallback is on. Gated by the screen-
    #    vision master switch (if vision can't run at all, no VRAM for it).
    #    Marked on-demand: Ollama loads it only when JARVIS actually looks, but
    #    it CAN co-load with chat — so it counts toward the worst-case peak.
    screen_vision_on = _truthy(settings.get("SCREEN_VISION_ENABLED", True))
    route = _vision_route(settings)
    vision_fallback = _truthy(settings.get("LOCAL_VISION_FALLBACK", False))
    vision_local = route in ("local", "auto") or vision_fallback
    if screen_vision_on and vision_local:
        # When LOCAL_VISION_MODEL is the SAME tag as the chat model (a
        # multimodal brain like gemma4:26b-a4b), vision re-uses the resident
        # model — zero extra VRAM, and the historical chat+VLM co-load brick
        # can't happen. "off" means local vision never loads at all. A
        # DIFFERENT tag is a real co-load and is estimated like the chat
        # model; when the key is absent (older settings dicts) we keep the
        # legacy flat qwen2.5vl:7b allowance.
        vision_tag = str(settings.get("LOCAL_VISION_MODEL") or "").strip()
        if vision_tag.lower() == "off":
            pass
        elif vision_tag and vision_tag.lower() == chat_tag.lower():
            components.append({
                "label": "vision (shared with chat)",
                "mb": 0,
                "ondemand": True,
            })
        else:
            components.append({
                "label": "vision",
                "mb": model_vram_estimate(vision_tag) if vision_tag else VISION_VRAM_MB,
                "ondemand": True,
            })

    # 3) Whisper STT — always resident while JARVIS is listening.
    components.append({
        "label": "Whisper",
        "mb": WHISPER_VRAM_MB,
        "ondemand": False,
    })

    # 4) RAG embeddings — only when RAG indexing/query is enabled.
    if _rag_enabled(settings):
        components.append({
            "label": "embeddings",
            "mb": EMBED_VRAM_MB,
            "ondemand": True,
        })

    total = sum(c["mb"] for c in components)

    card_total = total_mb if (total_mb and total_mb > 0) else total_vram_mb()
    budget = max(0, card_total - HEADROOM_MB)
    over = total > budget
    pct = (total / budget * 100.0) if budget > 0 else 0.0

    return {
        "components": components,
        "total_mb": total,
        "budget_mb": budget,
        "total_card_mb": card_total,
        "headroom_mb": HEADROOM_MB,
        "over": over,
        "pct": pct,
    }


def _short_model_label(model_tag: str) -> str:
    """A compact label for the chat model in the breakdown line — the param
    size when present ("32B", "14B", "8B"), else a trimmed tag."""
    tag = (model_tag or "").lower()
    token = ""
    for ch in tag:
        if ch.isdigit() or ch == ".":
            token += ch
        elif ch == "b" and token:
            return f"{token}B"
        else:
            token = ""
    # No "<n>b" marker — fall back to the bit before the first ':' or '-'.
    base = (model_tag or "model").split(":")[0].split("-")[0]
    return base or "model"


# ──────────────────────────────────────────────────────────────────────────
#  Text rendering (for tests + any non-GUI surface)
# ──────────────────────────────────────────────────────────────────────────
def _fmt_gb(mb: int) -> str:
    """MB → a short GB string, e.g. 13312 -> '13', 7475 -> '7.3'."""
    gb = mb / _GB
    if abs(gb - round(gb)) < 0.05:
        return f"{int(round(gb))}"
    return f"{gb:.1f}"


def budget_lines(settings: dict, total_mb: Optional[int] = None) -> list[str]:
    """Human-readable budget summary lines (used by tests + any text UI):
    a headline total, the per-component breakdown, and an over-budget warning
    when applicable. Never raises."""
    b = predict_budget(settings, total_mb)
    head = (f"VRAM budget: {_fmt_gb(b['total_mb'])} / "
            f"{_fmt_gb(b['budget_mb'])} GB usable "
            f"({_fmt_gb(b['total_card_mb'])} GB card, "
            f"{_fmt_gb(b['headroom_mb'])} GB reserved) — {b['pct']:.0f}%")
    parts = []
    for c in b["components"]:
        tag = " (on-demand)" if c.get("ondemand") else ""
        parts.append(f"{c['label']} {_fmt_gb(c['mb'])}{tag}")
    lines = [head, "  " + " · ".join(parts)]
    if b["over"]:
        lines.append("  " + over_warning(b))
    return lines


def over_warning(budget: dict) -> str:
    """The red-state warning string for an over-budget selection — names the
    overage and the two cheapest fixes (smaller chat model / route vision to the
    cloud). ``budget`` is a predict_budget() result."""
    need = _fmt_gb(budget["total_mb"])
    have = _fmt_gb(budget["total_card_mb"])
    return (f"⚠ These settings need ~{need} GB but you have {have} — "
            f"drop to a smaller local model (e.g. the 14B) or route vision to "
            f"the cloud.")


def budget_bar(settings: dict, width: int = 24,
               total_mb: Optional[int] = None) -> str:
    """An ASCII budget bar (for tests / a text UI), ``width`` cells wide, filled
    proportionally to pct and capped at full. '#' = used, '-' = free, and a
    trailing '!' marks an over-budget (clipped) prediction. Never raises."""
    if width < 1:
        width = 1
    b = predict_budget(settings, total_mb)
    frac = 0.0
    if b["budget_mb"] > 0:
        frac = b["total_mb"] / b["budget_mb"]
    filled = int(round(min(1.0, frac) * width))
    filled = max(0, min(width, filled))
    bar = "#" * filled + "-" * (width - filled)
    suffix = "!" if b["over"] else ""
    return f"[{bar}]{suffix}"
