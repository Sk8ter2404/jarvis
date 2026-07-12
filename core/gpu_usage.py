"""Live GPU / VRAM usage engine for JARVIS.

A small, pure, *graceful* module that answers one question at a glance: how
much VRAM is each loaded Ollama model using, what's the total against the
card's capacity, and which JARVIS function (chat / vision / ambient) is on the
local GPU vs the cloud.

It is the data layer behind three surfaces:
  • the unified HUD's "GPU/VRAM" panel (skills/gpu_usage.py publishes
    ``gpu_lines`` into hud_state.json; hud/jarvis_unified_hud.py renders it),
  • the ``gpu_usage`` voice action ("vram status", "what's loaded on the gpu"),
  • any other caller that wants a one-shot snapshot.

Data sources (all best-effort — a missing one degrades, never raises):
  • Per-model VRAM   — Ollama's ``GET /api/ps`` JSON (``.models[].name`` +
    ``.models[].size_vram`` bytes + ``.models[].size``), preferred because it's
    exact and machine-readable. Falls back to parsing the ``ollama ps`` table
    (NAME / SIZE / PROCESSOR columns) when the HTTP endpoint is unreachable.
  • Total VRAM + util + temp — ``nvidia-smi --query-gpu=memory.used,memory.free,
    memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits``.
  • Routing            — ``core.config.MODEL_ROUTING`` / ``model_route(fn)``.

Design rules (match core/gpu_state.py + the system_pulse/status_panel skills):
  • stdlib only (urllib, subprocess, json) so importing this never drags a dep
    onto a cloud-only or CI box.
  • Every subprocess uses CREATE_NO_WINDOW on Windows so no console flashes.
  • ``gpu_snapshot()`` is cached ~3 s (monotonic) so the HUD loop and a voice
    call moments apart don't each spawn nvidia-smi + an HTTP round-trip.
  • NOTHING here raises: a snapshot with missing pieces returns a partial dict
    (``total_mb`` / ``models`` may be absent); the formatters tolerate that.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request

try:  # core.config is always present in-tree; the guard keeps a bare import of
    # this module (e.g. `python -c "import core.gpu_usage"`) working even if a
    # future refactor moves the routing helpers.
    from core.config import MODEL_ROUTING, model_route
except Exception:  # pragma: no cover - defensive; core.config is in-tree
    MODEL_ROUTING = {"chat": "auto", "vision": "auto", "ambient": "auto"}

    def model_route(function: str) -> str:  # type: ignore[misc]
        return MODEL_ROUTING.get(function, "auto")


# ─── tunables ────────────────────────────────────────────────────────────
_OLLAMA_PS_URL = "http://127.0.0.1:11434/api/ps"
_HTTP_TIMEOUT  = 1.5     # /api/ps is local; a slow reply means a busy daemon
_SMI_TIMEOUT   = 2.0     # nvidia-smi cold-start can be ~1 s on Windows
_PS_TIMEOUT    = 2.0     # `ollama ps` subprocess fallback
_CACHE_TTL_S   = 3.0     # snapshot freshness window (monotonic seconds)

# Per-function routing keys we surface, in display order. Mirrors
# core.config.MODEL_ROUTING; kept as an explicit tuple so the HUD/voice line
# has a stable order regardless of dict iteration.
_ROUTE_FUNCTIONS = ("chat", "vision", "ambient")

_NO_WINDOW = (subprocess.CREATE_NO_WINDOW
              if sys.platform == "win32" else 0)  # type: ignore[attr-defined]

# Snapshot cache — guarded so the HUD thread and the voice thread share one.
_cache_lock = threading.Lock()
_cache: dict | None = None
_cache_at = 0.0   # monotonic timestamp of the cached snapshot


# ─── source 1: per-model VRAM via Ollama /api/ps (preferred) ─────────────

def _fetch_ollama_ps_json() -> list[dict] | None:
    """GET /api/ps and return ``.models`` (a list of dicts), or None on any
    failure (daemon down, timeout, non-JSON, schema drift). Never raises."""
    try:
        req = urllib.request.Request(_OLLAMA_PS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, list) else None


def _models_from_api(models: list[dict]) -> list[dict]:
    """Normalise /api/ps ``.models`` into our row shape.

    Each row: ``{"name": str, "vram_mb": int, "size_mb": int, "processor": str}``.
    ``size_vram`` is the bytes resident on the GPU; ``size`` is the total model
    footprint (GPU + any CPU spill). ``processor`` is derived from the GPU/total
    ratio so the HUD can flag a model that's partially on the CPU."""
    rows: list[dict] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        name = (m.get("name") or m.get("model") or "").strip()
        if not name:
            continue
        size_total = _coerce_int(m.get("size"))
        size_vram = _coerce_int(m.get("size_vram"))
        # size_vram missing but size present → assume fully resident (the model
        # is in /api/ps, so it IS loaded). Keeps a row from vanishing on schema
        # drift where only `size` is reported.
        if size_vram is None and size_total is not None:
            size_vram = size_total
        rows.append({
            "name": name,
            "vram_mb": _bytes_to_mb(size_vram),
            "size_mb": _bytes_to_mb(size_total),
            "processor": _processor_label(size_vram, size_total),
        })
    return rows


def _processor_label(size_vram: int | None, size_total: int | None) -> str:
    """Human label for where a model runs, mirroring `ollama ps`'s PROCESSOR
    column: "100% GPU" when fully resident, "N%/M% CPU/GPU" on a CPU spill."""
    if not size_total or size_vram is None:
        return ""
    if size_vram >= size_total:
        return "100% GPU"
    gpu_pct = int(round(100.0 * size_vram / size_total))
    gpu_pct = max(0, min(100, gpu_pct))
    cpu_pct = 100 - gpu_pct
    if cpu_pct <= 0:
        return "100% GPU"
    return f"{cpu_pct}%/{gpu_pct}% CPU/GPU"


# ─── source 1 (fallback): parse `ollama ps` table text ───────────────────

def _run_ollama_ps() -> str | None:
    """Capture ``ollama ps`` stdout, or None on any failure. Never raises."""
    try:
        r = subprocess.run(
            ["ollama", "ps"],
            capture_output=True, text=True, timeout=_PS_TIMEOUT,
            creationflags=_NO_WINDOW,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout or ""


def _parse_ollama_ps(text: str) -> list[dict]:
    """Parse the ``ollama ps`` table into our row shape.

    The table is whitespace-aligned with header NAME / ID / SIZE / PROCESSOR /
    CONTEXT / UNTIL. SIZE is two tokens ("13 GB", "323 MB"); PROCESSOR is one
    ("100% GPU") or three ("5%/95% CPU/GPU") tokens. We anchor on the SIZE
    "<num> <unit>" pair (unit in KB/MB/GB/TB) and treat the name as everything
    before it and the processor as the GPU/CPU tokens after it — robust to the
    variable column spacing without depending on fixed offsets."""
    rows: list[dict] = []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        toks = ln.split()
        if not toks or toks[0].upper() == "NAME":
            continue
        # Find the SIZE pair: a numeric token immediately followed by a unit.
        size_idx = None
        for i in range(1, len(toks) - 1):
            if _looks_numeric(toks[i]) and toks[i + 1].upper() in (
                    "B", "KB", "MB", "GB", "TB"):
                size_idx = i
                break
        if size_idx is None:
            continue
        name = toks[0]   # NAME has no spaces (an Ollama tag)
        vram_mb = _size_pair_to_mb(toks[size_idx], toks[size_idx + 1])
        # PROCESSOR tokens sit right after the unit, before CONTEXT (a bare
        # integer). Collect up to 3 tokens that contain "%", "GPU", "CPU", "/".
        proc_toks: list[str] = []
        for t in toks[size_idx + 2: size_idx + 6]:
            if any(c in t for c in "%/") or t.upper() in ("GPU", "CPU"):
                proc_toks.append(t)
            elif proc_toks:
                break
        rows.append({
            "name": name,
            "vram_mb": vram_mb,
            "size_mb": vram_mb,   # `ollama ps` SIZE is the resident size
            "processor": " ".join(proc_toks),
        })
    return rows


def _resident_models() -> list[dict]:
    """Per-model VRAM rows: /api/ps JSON first, `ollama ps` text as fallback."""
    api = _fetch_ollama_ps_json()
    if api is not None:
        return _models_from_api(api)
    text = _run_ollama_ps()
    if text:
        return _parse_ollama_ps(text)
    return []


# ─── source 2: total VRAM / util / temp via nvidia-smi ───────────────────

def _run_nvidia_smi_query() -> str | None:
    """One nvidia-smi CSV query line, or None on failure. Never raises."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.free,memory.total,"
             "utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_SMI_TIMEOUT,
            creationflags=_NO_WINDOW,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout or ""


def _parse_nvidia_smi(text: str) -> dict:
    """Parse the CSV line into total/used/free/util/temp keys.

    With multiple GPUs nvidia-smi prints one line each; we aggregate memory
    (sum) and take the max utilisation/temperature so a single card's headline
    number is meaningful. Missing/garbage fields are simply omitted."""
    used = free = total = 0
    utils: list[int] = []
    temps: list[int] = []
    saw_mem = False
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        u, f, t, ut, tp = parts[:5]
        if _looks_int(u) and _looks_int(f) and _looks_int(t):
            used += int(u)
            free += int(f)
            total += int(t)
            saw_mem = True
        if _looks_int(ut):
            utils.append(int(ut))
        if _looks_int(tp):
            temps.append(int(tp))
    out: dict = {}
    if saw_mem:
        out["used_mb"] = used
        out["free_mb"] = free
        out["total_mb"] = total
    if utils:
        out["util_pct"] = max(utils)
    if temps:
        out["temp_c"] = max(temps)
    return out


# ─── snapshot ────────────────────────────────────────────────────────────

def gpu_snapshot(*, use_cache: bool = True) -> dict:
    """A single best-effort GPU/VRAM snapshot.

    Returns (keys present only when their source answered)::

        {
          "total_mb": int, "used_mb": int, "free_mb": int,   # nvidia-smi
          "util_pct": int, "temp_c": int,                    # nvidia-smi
          "models": [ {"name": str, "vram_mb": int,
                       "size_mb": int, "processor": str}, ... ],  # ollama
          "routing": {"chat": "auto"|"local"|"cloud", ...},
          "ts": <monotonic float>,
        }

    Cached for ~3 s so rapid HUD + voice callers don't each hit nvidia-smi and
    the Ollama HTTP endpoint. Never raises — if every source is missing you get
    ``{"models": [], "routing": {...}, "ts": ...}``."""
    global _cache, _cache_at
    if use_cache:
        with _cache_lock:
            if (_cache is not None
                    and (time.monotonic() - _cache_at) < _CACHE_TTL_S):
                return _cache

    snap: dict = {"models": [], "routing": _routing_snapshot(),
                  "ts": time.monotonic()}
    try:
        snap["models"] = _resident_models()
    except Exception:
        snap["models"] = []
    try:
        smi = _run_nvidia_smi_query()
        if smi:
            snap.update(_parse_nvidia_smi(smi))
    except Exception:
        pass

    with _cache_lock:
        _cache = snap
        _cache_at = time.monotonic()
    return snap


def _routing_snapshot() -> dict:
    """Current per-function route for chat/vision/ambient (graceful)."""
    out: dict = {}
    for fn in _ROUTE_FUNCTIONS:
        try:
            out[fn] = model_route(fn)
        except Exception:
            out[fn] = "auto"
    return out


# ─── formatting: human / HUD-ready lines + a text bar ────────────────────

def usage_lines(snap: dict | None = None) -> list[str]:
    """Human/HUD-ready lines describing VRAM usage and routing.

    Example output::

        ["qwen2.5:14b-instruct-q5_K_M  13.0/24 GB",
         "qwen2.5vl:7b  7.3/24 GB",
         "nomic-embed-text:latest  0.3/24 GB",
         "TOTAL  20.6/24 GB (86%)  util 39%  33C",
         "chat→cloud  vision→cloud  ambient→local"]

    Degrades: with no models the per-model rows are replaced by a single
    "no models resident" line; with no nvidia-smi the TOTAL line is omitted
    (or shows just the model sum). Always returns at least the routing line."""
    if snap is None:
        snap = gpu_snapshot()
    total_mb = snap.get("total_mb")
    total_gb = _mb_to_gb(total_mb) if total_mb else None
    lines: list[str] = []

    models = snap.get("models") or []
    if models:
        for m in models:
            vram_gb = _mb_to_gb(m.get("vram_mb") or 0)
            if total_gb:
                row = f"{m['name']}  {vram_gb:.1f}/{total_gb:.0f} GB"
            else:
                row = f"{m['name']}  {vram_gb:.1f} GB"
            proc = (m.get("processor") or "")
            # Only annotate the unusual case (a CPU spill) — "100% GPU" is the
            # silent norm and would just add noise to every row.
            if proc and proc != "100% GPU":
                row += f"  [{proc}]"
            lines.append(row)
    else:
        lines.append("no models resident on the GPU")

    # TOTAL line — prefer nvidia-smi's used (the whole card incl. non-Ollama
    # allocations); fall back to the sum of model VRAM when smi is absent.
    used_mb = snap.get("used_mb")
    if used_mb is None and models:
        used_mb = sum((m.get("vram_mb") or 0) for m in models)
    if used_mb is not None:
        used_gb = _mb_to_gb(used_mb)
        if total_gb:
            pct = int(round(100.0 * used_mb / total_mb)) if total_mb else 0
            total_line = f"TOTAL  {used_gb:.1f}/{total_gb:.0f} GB ({pct}%)"
        else:
            total_line = f"TOTAL  {used_gb:.1f} GB"
        extra = []
        if snap.get("util_pct") is not None:
            extra.append(f"util {int(snap['util_pct'])}%")
        if snap.get("temp_c") is not None:
            extra.append(f"{int(snap['temp_c'])}C")
        if extra:
            total_line += "  " + "  ".join(extra)
        lines.append(total_line)

    lines.append(routing_line(snap))
    return lines


def routing_line(snap: dict | None = None) -> str:
    """One-line per-function routing summary, e.g.
    ``"chat→cloud  vision→cloud  ambient→local"``."""
    if snap is None:
        snap = gpu_snapshot()
    routing = snap.get("routing") or _routing_snapshot()
    bits = []
    for fn in _ROUTE_FUNCTIONS:
        route = routing.get(fn, "auto")
        bits.append(f"{fn}→{route}")
    return "  ".join(bits)


def usage_bar(width: int = 20, snap: dict | None = None) -> str:
    """A text bar of total VRAM use, e.g. ``"[##########----] 86%"``.

    Width is the number of cells between the brackets. Returns
    ``"[ n/a ]"`` when total VRAM is unknown (no nvidia-smi)."""
    if snap is None:
        snap = gpu_snapshot()
    width = max(4, int(width))
    total_mb = snap.get("total_mb")
    used_mb = snap.get("used_mb")
    if used_mb is None:
        models = snap.get("models") or []
        if models:
            used_mb = sum((m.get("vram_mb") or 0) for m in models)
    if not total_mb or used_mb is None:
        # No capacity reference — render an empty bar tagged n/a rather than a
        # misleading 0%.
        return "[" + "-" * width + "] n/a"
    frac = max(0.0, min(1.0, used_mb / total_mb))
    filled = int(round(frac * width))
    filled = max(0, min(width, filled))
    pct = int(round(frac * 100))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {pct}%"


def usage_summary_text(snap: dict | None = None) -> str:
    """The usage_lines joined with " · " — the compact one-string form the
    voice action and the HUD strip both want."""
    return "  ·  ".join(usage_lines(snap))


# ─── tiny numeric coercion helpers (all total-failure tolerant) ──────────

def _coerce_int(v) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _bytes_to_mb(b: int | None) -> int:
    if not b:
        return 0
    return int(round(b / (1024 * 1024)))


def _mb_to_gb(mb: float | None) -> float:
    if not mb:
        return 0.0
    return mb / 1024.0


def _looks_int(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
    return s.isdigit()


def _looks_numeric(s: str) -> bool:
    """True for '13', '7.3', '323' — an int or simple decimal."""
    s = (s or "").strip()
    if not s:
        return False
    if s.count(".") > 1:
        return False
    return s.replace(".", "", 1).isdigit()


_UNIT_MB = {"B": 1.0 / (1024 * 1024), "KB": 1.0 / 1024,
            "MB": 1.0, "GB": 1024.0, "TB": 1024.0 * 1024.0}


def _size_pair_to_mb(num: str, unit: str) -> int:
    """Convert an `ollama ps` SIZE pair ('13', 'GB') to whole MB."""
    try:
        val = float(num)
    except (TypeError, ValueError):
        return 0
    mult = _UNIT_MB.get((unit or "").upper())
    if mult is None:
        return 0
    return int(round(val * mult))
