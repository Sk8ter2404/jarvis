"""GPU / VRAM usage skill — voice readout + HUD panel feed.

The thin skill layer over ``core.gpu_usage``: it (1) registers the
``gpu_usage`` voice action (with the natural-language aliases the LLM tends to
emit — "vram status", "what's loaded on the gpu", …) and (2) runs a small
background loop that publishes the HUD-ready usage lines into hud_state.json so
the unified HUD can render a "GPU/VRAM" panel without itself spawning
nvidia-smi or hitting the Ollama HTTP endpoint.

Why a separate skill (not folded into system_pulse)? system_pulse's pulse line
is a single-sentence CPU/RAM/temp aggregator; this is a *per-model* VRAM
breakdown ("qwen 14B 13 GB, vision 7 GB, total 20.6/24") plus the per-function
local-vs-cloud routing — a distinct readout that deserves its own action and
its own HUD field. The heavy lifting (subprocess + HTTP + formatting) all lives
in core/gpu_usage.py, which is pure and unit-tested without a GUI or a real GPU.

HUD contract: every ``GPU_HUD_REFRESH_SECONDS`` (default 6 s) we write::

    hud_state.json:
      gpu_lines        : list[str]   # usage_lines() — per-model rows + TOTAL + routing
      gpu_bar          : str         # usage_bar() — "[####----] 86%"
      gpu_updated_at   : float       # wall-clock seconds

via the canonical ``write_hud_state`` writer (shared lock/cache) so we never
clobber pulse_strip / status_panel_strip. The HUD renders ``gpu_lines`` if
present and recent; otherwise it shows "GPU: n/a".
"""
import threading
import time

# core.gpu_usage is stdlib-only and graceful; import defensively so a skill-load
# failure here can never take down the loader (matches the sibling skills).
try:
    from core import gpu_usage
    _HAS_ENGINE = True
except Exception as _exc:   # pragma: no cover - core.gpu_usage is in-tree
    gpu_usage = None        # type: ignore[assignment]
    _HAS_ENGINE = False
    print(f"  [gpu-usage] core.gpu_usage unavailable ({_exc}); "
          f"action + HUD feed disabled")

# ─── cadence ─────────────────────────────────────────────────────────────
# The engine caches snapshots ~3 s, so a 6 s publish never double-hits the GPU
# but still feels live on the HUD. (system_pulse refreshes its strip at 15 s,
# status_panel at 20 s — GPU is the faster-moving of the three.)
GPU_HUD_REFRESH_SECONDS = 6


# ─── HUD publishing ──────────────────────────────────────────────────────

def _publish_hud(lines: list[str], bar: str) -> None:
    """Merge the GPU fields into hud_state.json via the canonical writer.

    Prefer the typed ``services`` facade injected by the loader; fall back to
    the legacy ``skill_utils['write_hud_state']`` dict so this works under the
    isolated test harness too. Silent on every failure — the HUD is optional."""
    writer = None
    svc = globals().get("services")
    if svc is not None and hasattr(svc, "write_hud_state"):
        writer = svc.write_hud_state
    else:
        try:
            utils = skill_utils   # type: ignore[name-defined]
            if isinstance(utils, dict):
                writer = utils.get("write_hud_state")
        except NameError:
            writer = None
    if writer is None:
        return
    try:
        writer(gpu_lines=lines, gpu_bar=bar, gpu_updated_at=time.time())
    except Exception:
        pass


def _hud_publish_loop() -> None:
    while True:
        try:
            snap = gpu_usage.gpu_snapshot()
            lines = gpu_usage.usage_lines(snap)
            bar = gpu_usage.usage_bar(14, snap)
            _publish_hud(lines, bar)
        except Exception as e:
            print(f"  [gpu-usage] hud publish error: {e}")
        time.sleep(GPU_HUD_REFRESH_SECONDS)


# ─── spoken readout ──────────────────────────────────────────────────────

def _speak_size(gb: float) -> str:
    """Render a VRAM figure for speech: '13' / '7.3' / 'under a gig'."""
    if gb <= 0:
        return "nothing"
    if gb < 1.0:
        return "under a gig"
    if abs(gb - round(gb)) < 0.1:
        return f"{round(gb)}"
    return f"{gb:.1f}"


_SIZE_HINTS = (
    # (substring in the ollama tag, spoken short name)
    ("32b", "the 32B"),
    ("14b", "the 14B"),
    ("8b", "the 8B"),
    ("vl", "the vision model"),
    ("embed", "the embedder"),
)


def _speak_model_name(tag: str) -> str:
    """A spoken-friendly short name for an Ollama tag, e.g.
    'qwen2.5:14b-instruct-q5_K_M' → 'qwen 14B'."""
    low = tag.lower()
    base = "qwen" if low.startswith("qwen") else \
           ("llama" if low.startswith("llama") else tag.split(":")[0])
    for needle, spoken in _SIZE_HINTS:
        if needle in low:
            if spoken.startswith("the "):
                # "the vision model" / "the embedder" stand alone.
                if needle in ("vl", "embed"):
                    return spoken
                return f"{base} {spoken.split()[-1]}"
    return base


def _routing_sentence(routing: dict) -> str:
    """Turn the routing dict into a spoken clause grouping local vs cloud, e.g.
    'Chat and vision are on the cloud, ambient on local.'"""
    if not routing:
        return ""
    local = [fn for fn in ("chat", "vision", "ambient")
             if routing.get(fn) == "local"]
    cloud = [fn for fn in ("chat", "vision", "ambient")
             if routing.get(fn) == "cloud"]
    auto = [fn for fn in ("chat", "vision", "ambient")
            if routing.get(fn) == "auto"]

    def _join(names: list[str]) -> str:
        if len(names) == 1:
            return names[0].capitalize()
        if len(names) == 2:
            return f"{names[0].capitalize()} and {names[1]}"
        return ", ".join(names[:-1]).capitalize() + f", and {names[-1]}"

    clauses: list[str] = []
    if cloud:
        verb = "is" if len(cloud) == 1 else "are"
        clauses.append(f"{_join(cloud)} {verb} on the cloud")
    if local:
        if clauses:
            clauses.append(f"{', '.join(f.lower() for f in local)} on local")
        else:
            verb = "is" if len(local) == 1 else "are"
            clauses.append(f"{_join(local)} {verb} on local")
    if auto:
        if clauses:
            clauses.append(f"{', '.join(f.lower() for f in auto)} on auto")
        else:
            verb = "is" if len(auto) == 1 else "are"
            clauses.append(f"{_join(auto)} {verb} on auto")
    if not clauses:
        return ""
    return ", ".join(clauses).strip() + "."


def _build_spoken(snap: dict) -> str:
    """A natural JARVIS-cadence VRAM readout from a snapshot.

    e.g. "qwen 14B is using 13 of 24 gigs, the vision model 7, total 20.6 of
    24, sir. Chat and vision are on the cloud, ambient on local." """
    models = snap.get("models") or []
    total_mb = snap.get("total_mb")
    total_gb = (total_mb / 1024.0) if total_mb else None

    if not models:
        head = "Nothing is resident on the GPU at the moment, sir."
    else:
        parts: list[str] = []
        for i, m in enumerate(models):
            gb = (m.get("vram_mb") or 0) / 1024.0
            name = _speak_model_name(m.get("name", ""))
            if i == 0:
                if total_gb:
                    parts.append(
                        f"{name} is using {_speak_size(gb)} of "
                        f"{round(total_gb)} gigs")
                else:
                    parts.append(f"{name} is using {_speak_size(gb)} gigs")
            else:
                parts.append(f"{name} {_speak_size(gb)}")
        head = ", ".join(parts)
        used_mb = snap.get("used_mb")
        if used_mb is None:
            used_mb = sum((m.get("vram_mb") or 0) for m in models)
        used_gb = used_mb / 1024.0
        if total_gb:
            head += (f", total {used_gb:.1f} of {round(total_gb)}"
                     f"{_util_clause(snap)}, sir.")
        else:
            head += f", total {used_gb:.1f} gigs, sir."

    routing = _routing_sentence(snap.get("routing") or {})
    return f"{head} {routing}".strip()


def _util_clause(snap: dict) -> str:
    """Optional ' at 41 degrees' style tail if a temp is known."""
    temp = snap.get("temp_c")
    if temp is not None:
        return f", GPU at {int(temp)} degrees"
    return ""


# ─── action registration ─────────────────────────────────────────────────

def register(actions):
    def gpu_usage_action(_: str = "") -> str:
        if not _HAS_ENGINE:
            return ("I can't read the GPU right now, sir — the usage engine "
                    "didn't load.")
        try:
            snap = gpu_usage.gpu_snapshot()
            return _build_spoken(snap)
        except Exception as e:
            return f"GPU usage readout failed: {e}"

    # Primary + the natural phrasings the LLM emits for "what's on the GPU".
    actions["gpu_usage"] = gpu_usage_action
    actions["vram_status"] = gpu_usage_action
    actions["show_vram"] = gpu_usage_action
    actions["gpu_status"] = gpu_usage_action
    actions["whats_loaded"] = gpu_usage_action

    if not _HAS_ENGINE:
        print("  [gpu-usage] engine missing — action returns a graceful "
              "message; HUD feed disabled.")
        return

    threading.Thread(target=_hud_publish_loop, daemon=True).start()
    print(
        f"  [gpu-usage] HUD VRAM panel refresh every {GPU_HUD_REFRESH_SECONDS}s; "
        f"actions: gpu_usage / vram_status / show_vram / gpu_status / whats_loaded"
    )
