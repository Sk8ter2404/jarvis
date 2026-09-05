"""Game low-power mode — JARVIS gets out of the way while he plays.

OWNER'S REQUEST (2026-09-04): "I just opened Fortnite. It needs to be able to
go into low power mode when games are running or something like that." Read
alongside his standing instruction ("try to make sure the RAM and GPU is clear
of everything else ... JARVIS is my main priority ... except the crucials"):
JARVIS wins among his TOOLS, but when he is GAMING the GAME wins. He should not
have to ask, and he must never have to undo it by hand.

WHY IT IS URGENT (measured, same evening):
    RAM   45.3 / 47.8 GB used — 2.5 GB free
    3090  23269 / 24576 MiB   — 95 %
    llama-server (the 26B brain) 23.32 GB commit — BIGGER than the game
Fortnite on this box has a lobby memory leak (8 GB -> 29 GB -> E_OUTOFMEMORY ->
RenderThread hang -> power button). At 2.5 GB free he was minutes from another
hard reset.

================================================================================
THE MECHANISM, IN ONE LINE:  THE DOWNSHIFT *IS* THE GUARD.
================================================================================
`_call_local_llm` calls `_get_local_llm_model()` fresh on EVERY turn, and that
returns `_RESOLVED_LOCAL_LLM_MODEL[0]` whenever the cache is warm
(bobert_companion.py:10503). `_call_local_vision` reads the module global
`LOCAL_VISION_MODEL` fresh on every call (bobert_companion.py:11627, :11711).

So repointing those three cells makes EVERY consumer — the owner's own
utterance, teams_nudge's 10-minute vision tick, banter, ambient extract, the
autocorrect embedder — load the SMALL model instead of the big one. Not five
gated call sites: the resolver itself. That single move is why this needs

    * NO edit to bobert_companion.py (four other agents are editing it),
    * NO stop/pause API on the seven daemons that do not have one,
    * NO JARVIS restart (load_skills() auto-discovers a NEW skills/*.py; see
      its own comment at bobert_companion.py:19130 — only EDITS to an existing
      skill need a restart).

Three designs were considered. The two rejected ones failed here:

  "Gate the Ollama request path."  There is no chokepoint. Verified request
  sites in the live tree: bobert_companion.py:11058 `_call_local_llm`, :11393
  `_warm_up_local_llm_async` (posts keep_alive "20m" deliberately), :11610
  `_call_local_vision`, core/orchestrator.py:250, core/rag_indexer.py:108
  (/api/embeddings). Gating two of five and calling it coverage is this repo's
  #1 documented defect wearing a fix's clothing.

  "Unload the brain and reload on demand."  Measured: the orchestrator ran
  `ollama stop` at ~20:35; by 20:36:16 teams_nudge's vision call had cold-loaded
  the whole 15.2 GB back. A one-shot unload does not survive ten minutes here.
  And the reload IS the crash — `cudaMalloc failed: out of memory` is already in
  this session's log twice (19:41:35, 19:43:27).

  "Suppress the LLM and fall back to Claude."  There is no cloud on this box.
  data/user_settings.json pins AI_BACKEND=ollama and MODEL_ROUTING all-local;
  `_claude_reachable()` returns False unconditionally when AI_BACKEND != claude.
  Suppression here does not mean "cloud" — it means MUTE, and it means JARVIS
  saying "my local model isn't responding" when in fact we switched it off.
  This skill NEVER gates the LLM. He keeps a working JARVIS, on a smaller brain.

================================================================================
WHAT THIS SKILL WILL NOT CLAIM
================================================================================
1. IT DOES NOT PIN THE MODEL FOR 24 HOURS. `keep_alive: "20m"` is HARDCODED in
   the chat payload at bobert_companion.py:11148, so any long keep_alive we send
   is reset to 20m by the owner's very next utterance. Instead, `_keep_warm()`
   re-pings the small model ONLY WHEN IT IS ALREADY RESIDENT — a refresh that
   allocates nothing. When it has already evicted we do NOTHING and let the next
   real call lazy-load it (~9 GB, not ~15 GB). Pinned-if-free, lazy-if-not.

2. IT DOES NOT TRUST THE CALIBRATION TABLE. core/vram_budget.py says
   `gemma4:latest` is 4 GB ("E4B ~3.4 GB measured @8k ctx"). Live `ollama list`
   on 2026-09-04 says 9.6 GB on disk — bigger than gemma4:12b's 7.6 GB. That
   row is STALE and "escalating" to it would have made things WORSE. So
   `_pick_game_brain()` verifies against LIVE /api/tags disk sizes and REFUSES
   any candidate that is not measurably smaller than the tag it replaces.

3. IT DOES NOT CLAIM SAVINGS IT DID NOT MEASURE. `_measure()` is called once
   before and once after, by the same code path, and the delta is the only
   number reported. Below GAME_MODE_MIN_VRAM_DELTA_MB the mode reports FAILURE
   — BUT ONLY WHEN THERE WAS SOMETHING TO FREE. This skill delivers TWO
   different savings and only one of them is a delta:

       freed now      = the unload. Measurable: model VRAM before minus after.
       load avoided   = the downshift. NOT measurable as a delta, because the
                        load it prevents never happens.

   Scoring the second one on the first one's yardstick is how a working mode
   reported its own failure: on an EMPTY card (/api/ps returned [] at 21:19 on
   2026-09-04, nothing resident) the unload frees 0 MB, Fortnite's VRAM climbs
   during the 8 s verify window so the card-level delta is 0 or NEGATIVE, and
   the floor then fired — for a downshift that had worked perfectly. See
   `_classify()`: the floor now applies only to VRAM that was THERE TO FREE,
   and "nothing was resident" is itself a measurement, not a failure.

4. IT DOES NOT FIX THE FREEZE. Fortnite's leak (10.04 -> 16.14 -> 17.13 GB in
   ~17 min) is inside the game process and nothing here touches it. This buys
   headroom so the leak takes far longer to reach the wall. Anyone reading a
   successful game-mode session as "the freeze is fixed" has misread it.

5. THE BOOT WARM-UP HOLE IS REAL AND NAMED. `_warm_up_local_llm_async()` runs at
   bobert_companion.py:24903, and `load_skills()` at :24949 — the warm-up fires
   BEFORE skills load, so a JARVIS restart mid-game cold-loads the FULL brain
   once. We cannot pre-empt that from a skill file. What we do: re-downshift and
   unload within one poll (default 5 s). To close it completely the launcher
   environment needs JARVIS_LOCAL_LLM_MODEL=<small tag>, which the resolver
   honours first (bobert_companion.py:10507) — an orchestrator/config action,
   not a code one.

================================================================================
CANNOT GET STUCK — FIVE INDEPENDENT LAYERS
================================================================================
L1 NOTHING PERSISTS. Every change is a process-memory cell. This module writes
   NO files at all. data/user_settings.json still says gemma4:26b-a4b-it-qat, so
   ANY restart — clean exit, crash, or the power button — restores everything
   with zero cleanup code. This is the layer that needs no code to be correct.
   IT IS NOT FREE, THOUGH: it also forbids calling any SKILL ACTION that
   persists. This claim was already false once — `_suspend_luxuries` drove the
   Kinect levers through `gestures_off()`/`air_mouse_off()`, which write
   data/user_settings.json, so a power-button exit left his gesture control
   permanently off. Use `_idle_cfg_flag` (a live core.config flip) for anything
   whose own action persists. A source-shaped test CANNOT see that mistake — the
   write happens in the called skill — so it is pinned behaviourally instead, in
   tests/skills/test_game_mode_no_persist.py, against the real settings writer.
L2 PID WATCHDOG, NOT EVENTS. Exit is decided by polling "is this pid alive".
   A missed event, swallowed exception or dropped callback still recovers.
L3 DEADMAN CEILING (GAME_MODE_MAX_SECONDS, and a hard 2x absolute ceiling).
L4 ENTRY FAILS CLOSED, EXIT FAILS OPEN. Unsure => never a reason to enter, always
   a reason to leave. `_restore()` runs from defaults when state is unreadable.
L5 MANUAL OVERRIDE that also INHIBITS re-entry for that pid, so the watcher can
   never fight the owner (the escape-hatch/guard race that sank a rival design).

Anti-cheat: we enumerate process names (psutil — verified to see the EAC-
protected client) and compare foreground PIDs. `game_mode_learn_this` uses
kernel32 QueryFullProcessImageNameW, NOT psapi GetModuleBaseNameW — measured
live, EAC denies the latter with ERROR_ACCESS_DENIED and it returns '' for
Fortnite, so skills/ambient_listen.py:_focused_proc_name() is blind to any
protected process. We never OpenProcess with VM_READ, never read game memory,
never inject.

Not enabled by default (GAME_MODE_ENABLED=False). The orchestrator enables it.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
import time
import urllib.request

# ── Speak sets ────────────────────────────────────────────────────────────
# Every action here returns a FINISHED, user-facing sentence. Without this the
# answer is computed, logged and dropped — the exact defect skills/audio_devices
# was written to fix.
SPEAK_VERBATIM_ACTIONS = (
    "game_mode_status", "game_mode_on", "game_mode_off", "game_mode_learn_this",
)

# Registering a handler does NOT teach the local brain the name. A name is only
# emittable if it appears in core/prompts.py or in a module-level
# PROMPT_EXAMPLES string, which load_skills collects at
# bobert_companion.py:_collect_skill_prompt_examples. That exact mistake was
# made and caught earlier today.
# TODO(migrate): core/prompts.py is the home for TRACKED skills like this one.
# It is being edited by another agent right now, so these examples live here for
# tonight; move this block into core/prompts.py once that edit lands.
PROMPT_EXAMPLES = (
    "GAME LOW-POWER MODE (skills/game_mode.py — frees VRAM/RAM for a running\n"
    "  game by downshifting the local brain; never mutes JARVIS):\n"
    "  game_mode_status     — report whether game mode is engaged, which game,\n"
    "                         which brain is loaded, and the MEASURED memory\n"
    "                         freed. Fire on 'are you in game mode', 'what\n"
    "                         power mode are you in', 'did you free any memory',\n"
    "                         'which brain are you on', 'low power status'.\n"
    "  game_mode_on         — force game mode on now. Fire on 'game mode',\n"
    "                         'low power mode', 'divert power', 'I'm gaming',\n"
    "                         'get out of the way', 'free up some memory'.\n"
    "  game_mode_off        — leave game mode and restore everything. Fire on\n"
    "                         'normal power', 'full power', 'exit game mode',\n"
    "                         'game mode off', 'come back'.\n"
    "  game_mode_learn_this — treat the CURRENTLY FOCUSED app as a game from\n"
    "                         now on. Fire on 'treat this as a game', 'add this\n"
    "                         to my games', 'this is a game'.\n"
    "    Example: 'divert power' -> [ACTION: game_mode_on]\n"
    "    Example: 'normal power' -> [ACTION: game_mode_off]"
)

_LOG = "  [game-mode]"
_OLLAMA = "http://127.0.0.1:11434"

# Absolute ceiling multiplier on GAME_MODE_MAX_SECONDS. Past this we exit no
# matter what the rest of the state says — L3's last resort.
_ABSOLUTE_CEILING_FACTOR = 2.0


# ── Monolith access ───────────────────────────────────────────────────────
def _bc():
    """The already-loaded monolith, or None. Never imports it fresh — importing
    bobert_companion has side effects (it starts device pumps). Same contract as
    skills/audio_devices.py."""
    mod = sys.modules.get("bobert_companion")
    if mod is not None:
        return mod
    try:                                     # pragma: no cover - standalone use
        return importlib.import_module("bobert_companion")
    except Exception:
        return None


def _cfg(name, default=None):
    """Read a setting, monolith first (it re-exports core.config), then
    core.config directly so the skill works in isolation and under test."""
    bc = _bc()
    if bc is not None:
        sentinel = object()
        val = getattr(bc, name, sentinel)
        if val is not sentinel:
            return val
    try:
        import core.config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _cfg_float(name, default: float) -> float:
    try:
        return float(_cfg(name, default))
    except Exception:
        return default


def _cfg_int(name, default: int) -> int:
    try:
        return int(_cfg(name, default))
    except Exception:
        return default


def _announce(message: str) -> None:
    """Route through the canonical pending_speech writer. Funnelling every skill
    through proactive_announce eliminates the cross-skill read-modify-write race
    an independent writer would reintroduce."""
    try:
        bc = _bc()
        ann = getattr(bc, "proactive_announce", None) if bc else None
        if callable(ann) and ann(message, source="game_mode"):
            return
    except Exception as e:
        print(f"{_LOG} speech-queue write failed ({e}); msg: {message}")
        return
    print(f"{_LOG} {message}")


def _hud(**updates) -> None:
    """Publish through the monolith's canonical hud_state writer (the module
    holds the lock + cached state, so concurrent skill writers cannot clobber
    each other). Never writes data/ directly. Best-effort."""
    try:
        utils = globals().get("skill_utils") or {}
        w = utils.get("write_hud_state")
        if callable(w):
            w(**updates)
    except Exception:
        pass


# ── Measurement (one code path, called before AND after) ──────────────────
def _nvidia_smi_mb() -> "tuple[int | None, int | None]":
    """(used_mb, free_mb) on cuda:0 from the DRIVER, or (None, None).

    nvidia-smi rather than the HUD: hud_state.json's gpu_lines were measured 62
    minutes stale while 14.5 GB was resident, reporting "no models resident on
    the GPU". Verifying through the HUD would read a lie."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=4,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
        )
        if out.returncode != 0:
            return None, None
        line = (out.stdout or "").strip().splitlines()[0]
        used, free = (p.strip() for p in line.split(",")[:2])
        return int(used), int(free)
    except Exception:
        return None, None


def _ollama_json(path: str, timeout_s: float = 3.0):
    """GET a small Ollama JSON endpoint. Never raises; returns None on failure."""
    import json as _json
    try:
        req = urllib.request.Request(f"{_OLLAMA}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return _json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _resident_models() -> "list[dict] | None":
    """[{name, size_vram_mb}] currently loaded, from /api/ps — or None when
    /api/ps could not be read.

    None, NOT []. Those were the same value here, and that ambiguity is a
    memory claim waiting to happen: `_classify()` reads this to decide whether
    there was anything to reclaim, and "nothing was resident" is a MEASUREMENT
    while "I could not tell" is not. Callers that only want something to
    iterate write `_resident_models() or []`, which keeps the old fail-closed
    behaviour exactly."""
    payload = _ollama_json("/api/ps")
    if payload is None:
        return None
    out = []
    for m in (payload.get("models") or []):
        name = (m or {}).get("name") or (m or {}).get("model") or ""
        if not name:
            continue
        try:
            vram_mb = int((m or {}).get("size_vram", 0)) // (1024 * 1024)
        except Exception:
            vram_mb = 0
        out.append({"name": name, "size_vram_mb": vram_mb})
    return out


def _installed_sizes_mb() -> "dict[str, int]":
    """{tag: on-disk MB} from LIVE /api/tags. The antidote to the stale
    CALIBRATED_VRAM_MB row that would have made 'escalation' load MORE."""
    payload = _ollama_json("/api/tags") or {}
    sizes: dict[str, int] = {}
    for m in (payload.get("models") or []):
        name = (m or {}).get("name") or ""
        if not name:
            continue
        try:
            sizes[name] = int((m or {}).get("size", 0)) // (1024 * 1024)
        except Exception:
            continue
    return sizes


def _free_ram_mb() -> "int | None":
    try:
        import psutil
        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:
        return None


def _measure() -> dict:
    """The ONE snapshot function. Before and after must come from here, or the
    delta is not a measurement — it is a story."""
    used, free = _nvidia_smi_mb()
    return {
        "at": time.time(),
        "vram_used_mb": used,
        "vram_free_mb": free,
        "free_ram_mb": _free_ram_mb(),
        # None here means "/api/ps was unreadable", never "nothing resident".
        "resident": _resident_models(),
    }


def _resident_vram_mb(sample: dict) -> "int | None":
    """Total VRAM held by RESIDENT OLLAMA MODELS in one sample, or None when
    /api/ps was unreadable for that sample.

    THIS is the number game mode is answerable for. nvidia-smi's whole-card
    figure is dominated by the GAME — Fortnite went 10.04 -> 16.14 -> 17.13 GB
    in ~17 minutes on this box — so a card-level delta taken across the 8 s
    verify window can read zero, or NEGATIVE, while the unload worked
    perfectly. Same endpoint, same shape, taken before and after: still a
    measurement, just of the right thing."""
    models = sample.get("resident")
    if models is None:
        return None
    total = 0
    for m in (models or []):
        try:
            total += int((m or {}).get("size_vram_mb") or 0)
        except Exception:
            continue
    return total


def _delta(before: dict, after: dict) -> dict:
    """Signed deltas, or None per-field when either sample could not be read.
    A missing reading is NEVER silently treated as zero."""
    def _d(key, invert=False):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            return None
        return (b - a) if invert else (a - b)
    b_models, a_models = _resident_vram_mb(before), _resident_vram_mb(after)
    return {
        # VRAM freed = used went DOWN, so invert. WHOLE CARD, game included.
        "vram_freed_mb": _d("vram_used_mb", invert=True),
        "ram_freed_mb": _d("free_ram_mb"),
        # Attributable to us: model VRAM that actually went away. None when
        # either /api/ps sample was unreadable — never silently zero.
        "model_vram_freed_mb": (None if (b_models is None or a_models is None)
                                else b_models - a_models),
        # How much model VRAM there was TO reclaim at entry. None = unknown;
        # 0 is a MEASUREMENT saying the card held no model at all.
        "reclaimable_mb": b_models,
    }


# ── Model selection: verified against the LIVE box, not a table ───────────
def _tag_base(tag: str) -> str:
    return (tag or "").strip().split(":", 1)[0].lower()


def _pick_game_brain(current_tag: str) -> "tuple[str | None, str]":
    """(tag, reason). The configured GAME_MODE_BRAIN, accepted ONLY when live
    /api/tags proves it is installed AND measurably smaller than `current_tag`.

    This guard exists because core/vram_budget.py's `"gemma4:latest": 4 * _GB`
    row is stale — the installed blob is 9.6 GB, LARGER than gemma4:12b's
    7.6 GB. Trusting a written table over the live box is this project's
    signature bug; refusing to shrink upward is the fix."""
    want = str(_cfg("GAME_MODE_BRAIN", "") or "").strip()
    if not want:
        return None, "GAME_MODE_BRAIN is not set"
    if _tag_base(want) == _tag_base(current_tag) and want == current_tag:
        return None, f"already on {want}"

    sizes = _installed_sizes_mb()
    if not sizes:
        # /api/tags unreadable. Entry FAILS CLOSED: we will not repoint the
        # brain at a tag we could not prove is installed, because a missing tag
        # makes _call_local_llm kick a background PULL and return None — i.e.
        # a silent mute. Refusing to enter is strictly safer.
        return None, "could not read /api/tags — refusing an unverified downshift"

    tag = None
    if want in sizes:
        tag = want
    else:
        base = _tag_base(want)
        tag = next((n for n in sizes if _tag_base(n) == base), None)
    if not tag:
        return None, f"{want} is not installed"

    cur_mb = sizes.get(current_tag)
    if cur_mb is None:
        cur_mb = next((mb for n, mb in sizes.items()
                       if _tag_base(n) == _tag_base(current_tag)), None)
    new_mb = sizes.get(tag, 0)
    if cur_mb is None:
        return None, (f"could not size the current brain {current_tag!r} — "
                      f"refusing a downshift I cannot prove is a downshift")
    if new_mb >= cur_mb:
        return None, (f"{tag} is {new_mb} MB on disk vs {current_tag} at "
                      f"{cur_mb} MB — that is not a downshift, refusing")
    return tag, f"{tag} ({new_mb} MB) < {current_tag} ({cur_mb} MB) on disk"


def _is_multimodal(tag: str) -> bool:
    """Prefer skills/model_picker's live /api/show capability probe; fall back
    to its tag-marker heuristic. Never raises."""
    mp = sys.modules.get("skill_model_picker")
    for fn_name in ("_is_multimodal", "_is_multimodal_tag"):
        fn = getattr(mp, fn_name, None) if mp else None
        if callable(fn):
            try:
                return bool(fn(tag))
            except Exception:
                continue
    t = (tag or "").lower()
    return any(m in t for m in ("gemma4", "vl", "vision", "llava", "minicpm"))


def _unload(tag: str) -> bool:
    """Ask Ollama to drop `tag` now (keep_alive 0). Never kills ollama.exe —
    that orphans its llama-server child — and never starts a second server."""
    import json as _json
    if not tag:
        return False
    try:
        body = _json.dumps({"model": tag, "messages": [], "keep_alive": 0}
                           ).encode("utf-8")
        req = urllib.request.Request(
            f"{_OLLAMA}/api/chat", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        return True
    except Exception as e:
        print(f"{_LOG} unload of {tag} failed: {e}")
        return False


def _keep_warm(tag: str) -> str:
    """Refresh `tag`'s keep_alive — ONLY IF IT IS ALREADY RESIDENT.

    THIS FUNCTION MUST NEVER CAUSE A LOAD. `keep_alive: "20m"` is hardcoded in
    the chat payload (bobert_companion.py:11148), so the owner's next utterance
    resets any long keep_alive we send; without a refresh the small brain
    evicts on a 20-minute cycle and the next daemon tick pays a cold load. But
    a refresh that fires when the model has ALREADY evicted is itself the cold
    load. So: resident -> refresh (allocates nothing); evicted -> do nothing and
    let the next real call lazy-load it. Pinned-if-free, lazy-if-not.

    num_ctx MUST match what every other JARVIS caller sends. Ollama keys a
    runner by (model, options); a mismatched context silently EVICTS and reloads
    — proven live 2026-07-21 at n_ctx=262144, which bricked the card."""
    import json as _json
    if not tag:
        return "no tag"
    # EXACT tag match. A base-name match would read gemma4:26b-a4b-it-qat as
    # "gemma4:12b is resident" — they share the base "gemma4" — and fire a
    # request for a model that is NOT loaded, which is precisely the ~9 GB cold
    # load this function exists to prevent. A bare configured name is tolerated
    # against Ollama's fully-qualified ":latest" form and nothing else.
    def _same(resident_name: str) -> bool:
        if resident_name == tag:
            return True
        if ":" not in tag and resident_name == f"{tag}:latest":
            return True
        if ":" not in resident_name and tag == f"{resident_name}:latest":
            return True
        return False

    if not any(_same(m["name"]) for m in (_resident_models() or [])):
        return "not resident — skipped (a refresh now would BE a cold load)"
    try:
        from core.ollama_opts import local_num_ctx
        num_ctx = local_num_ctx(tag)
    except Exception:
        num_ctx = 16384
    try:
        body = _json.dumps({
            "model": tag,
            "messages": [{"role": "user", "content": "ok"}],
            "stream": False,
            "options": {"num_predict": 1, "num_ctx": num_ctx},
            "keep_alive": "20m",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{_OLLAMA}/api/chat", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return f"refreshed (num_ctx={num_ctx})"
    except Exception as e:
        return f"refresh failed: {e}"


# ── Detection ─────────────────────────────────────────────────────────────
def _game_hints() -> "tuple[str, ...]":
    raw = _cfg("GAME_MODE_PROCESS_HINTS", ()) or ()
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for h in raw:
        if isinstance(h, str) and h.strip():
            out.append(h.strip().lower())
    return tuple(out)


def _find_game_pid() -> "tuple[int | None, str | None]":
    """(pid, exe_basename) of the first running allowlisted game, else (None, None).

    EXACT basename match, case-insensitive — NOT substring. Measured live
    2026-09-04 20:5x, four Fortnite processes were running at once:
        FortniteBootstrapper.exe
        FortniteClient-Win64-Shipping_EAC_EOS.exe
        FortniteLauncher.exe
        FortniteClient-Win64-Shipping.exe        <- the only real game
    A substring test on "fortniteclient-win64-shipping" matches the EAC_EOS
    wrapper too, so exact match is what makes the allowlist mean anything.

    psutil, not psapi: EasyAntiCheat denies GetModuleBaseNameW on the shipping
    client (ERROR_ACCESS_DENIED, returns ''), so the repo's existing helper
    skills/ambient_listen.py:_focused_proc_name() is blind to it and a detector
    reusing that helper would never fire while reporting itself healthy.
    psutil.process_iter(['name']) sees it — verified live."""
    hints = _game_hints()
    if not hints:
        return None, None
    try:
        import psutil
    except Exception:
        return None, None
    try:
        for p in psutil.process_iter(["pid", "name"]):
            try:
                name = (p.info.get("name") or "").strip().lower()
            except Exception:
                continue
            if name and name in hints:
                return int(p.info["pid"]), name
    except Exception:
        return None, None
    return None, None


def _pid_alive(pid: "int | None") -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return False


def _foreground_pid() -> "int | None":
    """PID owning the foreground window, or None.

    GetWindowThreadProcessId only — it needs no process handle at all, so it is
    the least intrusive primitive available and is unaffected by anti-cheat.
    A minimised window is reported as NOT foreground for our purposes."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        hwnd = u32.GetForegroundWindow()
        if not hwnd:
            return None
        if u32.IsIconic(hwnd):
            return None
        pid = wintypes.DWORD(0)
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value) or None
    except Exception:
        return None


def _foreground_exe_basename() -> "str | None":
    """Basename of the foreground process's image, via kernel32
    QueryFullProcessImageNameW.

    Measured against Fortnite (pid 47036):
        OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) -> handle, err=0   OK
        GetModuleBaseNameW (psapi)  -> rc=0, err=5 ACCESS_DENIED, value='' FAIL
        QueryFullProcessImageNameW  -> rc=1, full path                    OK
    Used only by game_mode_learn_this, so enrolment works WHILE the game is
    focused. PROCESS_QUERY_LIMITED_INFORMATION only — never VM_READ."""
    if sys.platform != "win32":
        return None
    pid = _foreground_pid()
    if not pid:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return None
        try:
            size = wintypes.DWORD(32768)
            buf = ctypes.create_unicode_buffer(size.value)
            if not k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return None
            return os.path.basename(buf.value).strip().lower() or None
        finally:
            k32.CloseHandle(h)
    except Exception:
        return None


# ── Watch state + the PURE decision core ─────────────────────────────────
class _State:
    """Volatile, in-process only. NOTHING here is written to disk (L1)."""

    def __init__(self):
        self.active = False
        self.game_pid = None            # int | None — the pid that owns the mode
        self.game_exe = None            # str | None
        self.entered_at = 0.0
        self.last_seen_at = 0.0         # last tick the game pid was alive
        self.fg_pid = None              # candidate pid currently in foreground
        self.fg_since = None            # when it took foreground (continuous)
        self.manual = False             # entered by voice, not by detection
        self.inhibit_pid = None         # pid the owner said "normal power" for
        self.applied = {}               # lever name -> restore payload
        # A brain restore we OWE him but deliberately deferred because a game
        # was still running. It lives here and NOT in `applied`, because
        # `applied` is (a) wiped by _restore_luxuries and (b) rewritten by the
        # next _enter() — and because nothing ever walks `applied` once
        # `active` is False. A debt parked in `applied` is a debt never paid.
        self.pending_brain = None       # prev-brain payload owed back to him
        self.before = None
        self.after = None
        # "ok" | "armed" | "failed" | "unverified" | None. "armed" is
        # engaged-with-nothing-to-reclaim: the unload had no work to
        # do, the downshift still prevents the next big load.
        self.result = None
        self.notes = []                 # honest per-step outcomes


_st = _State()
_lock = threading.RLock()
_stop = threading.Event()
_thread = [None]
# The settle-and-verify worker for the CURRENT entry, when the caller could not
# afford to wait for it (the voice thread — see _enter's defer_verify note).
# Exposed as a cell so a test can join it deterministically instead of sleeping,
# and daemon=True so it can never hold JARVIS open on exit.
_verify_thread = [None]


def decide(now: float, running_pid, foreground_pid, st: _State, cfg: dict):
    """PURE-ish decision core: ('enter' | 'exit' | 'none', reason).

    Mutates only st's TIMING cells so it can be driven from a test with fake
    clocks and no hardware. Every entry/exit rule in this skill lives here and
    nowhere else — there is no second copy to rot (this repo's #1 bug class).

    ENTRY FAILS CLOSED, EXIT FAILS OPEN. Being unsure is never a reason to
    enter and always a reason to leave.
    """
    dwell = cfg.get("dwell", 20.0)
    grace = cfg.get("grace", 45.0)
    max_s = cfg.get("max_seconds", 12 * 3600.0)
    require_fg = cfg.get("require_foreground", True)

    if st.active:
        # ── L3 deadman: absolute ceiling first, unconditional. ──
        if st.entered_at and (now - st.entered_at) >= max_s * _ABSOLUTE_CEILING_FACTOR:
            return "exit", (f"absolute ceiling reached "
                            f"({(now - st.entered_at) / 3600.0:.1f} h)")
        # A manual "game mode on" is owned by the owner, not by a process: it
        # ends only on his word or on the deadman.
        if st.manual:
            if st.entered_at and (now - st.entered_at) >= max_s:
                return "exit", "deadman ceiling reached on a manual hold"
            return "none", "manual hold"
        if running_pid is not None and running_pid == st.game_pid:
            st.last_seen_at = now
            return "none", "game still running"
        # Alt-tab does NOT exit. PROCESS LIFETIME owns the mode; focus only ever
        # confirmed ENTRY. This is the anti-flap property that matters: he tabs
        # out to Discord mid-match constantly.
        if _pid_alive(st.game_pid):
            st.last_seen_at = now
            return "none", "game pid alive (not focused — that is fine)"
        if st.last_seen_at and (now - st.last_seen_at) < grace:
            return "none", (f"game pid gone {now - st.last_seen_at:.0f}s — "
                            f"inside the {grace:.0f}s relaunch grace")
        if st.entered_at and (now - st.entered_at) >= max_s:
            return "exit", "deadman ceiling reached"
        return "exit", "game process exited"

    # ── Not active. ──
    if running_pid is None:
        st.fg_pid = None
        st.fg_since = None
        return "none", "no allowlisted game running"
    if st.inhibit_pid is not None and running_pid == st.inhibit_pid:
        # L5: the owner said "normal power" for THIS game. The watcher must
        # never fight him — that race (guard unloading a brain the owner just
        # asked for) is what sank a rival design.
        return "none", "re-entry inhibited by the owner for this game"
    if not require_fg:
        if st.fg_since is None or st.fg_pid != running_pid:
            st.fg_pid = running_pid
            st.fg_since = now
            return "none", "first sighting — starting dwell"
        if (now - st.fg_since) >= dwell:
            return "enter", f"game running for {now - st.fg_since:.0f}s"
        return "none", "dwelling"

    if foreground_pid != running_pid:
        # Continuous requirement: any break resets the clock, so a game merely
        # LAUNCHING behind his work never trips the mode.
        st.fg_pid = None
        st.fg_since = None
        return "none", "game is running but not in the foreground"
    if st.fg_since is None or st.fg_pid != running_pid:
        st.fg_pid = running_pid
        st.fg_since = now
        return "none", "took foreground — starting dwell"
    if (now - st.fg_since) >= dwell:
        return "enter", f"foreground for {now - st.fg_since:.0f}s"
    return "none", f"dwelling ({now - st.fg_since:.0f}/{dwell:.0f}s)"


# ── Levers (best-effort; each records applied / unavailable HONESTLY) ─────
def _lever(name: str, fn, restore, notes: list, applied: dict, was_on) -> None:
    """Run one suspend step, and record its undo ONLY when we PROVED there was
    something to undo.

    A lever that is not present in this build is recorded as unavailable — never
    counted as applied. That distinction is the whole point: a mode that says it
    stopped something it could not stop is this project's defining defect.

    `was_on` IS REQUIRED AND HAS NO DEFAULT. It must be a callable returning
    True iff the thing was ON/RUNNING before we touched it. There is no
    convenience default because a missing probe is exactly the defect this
    parameter exists to kill (verified live 2026-09-04):

        data/user_settings.json has KINECT_AIR_MOUSE_ENABLED = false. So
        air_mouse_off("") hits `if not _cfg_flag(...): return "The air-mouse is
        already off, sir."` — a NO-OP, no state change, no exception. _lever
        recorded air_mouse_on as the restore anyway. On the next game exit that
        restore ran, and air_mouse_on() calls _set_enabled(True)
        UNCONDITIONALLY: core.config.KINECT_AIR_MOUSE_ENABLED = True AND
        _persist_setting writes it into data/user_settings.json. Leaving game
        mode therefore TURNED ON a global cursor-takeover with synthetic
        LEFTDOWN/RIGHTDOWN injection (skills/kinect_air_mouse.py:2481-2498) that
        was off before, and persisted it across every future restart. The live
        log already shows his hands on keyboard+mouse registering as a two-hand
        grab seven times in six minutes tonight — so that is a stray click in
        his next match, from the mode whose entire job is to protect it.

        Identical shape, same commit: guard_off() returns "I wasn't on watch,
        sir." and the restore would ARM guard mode plus a camera monitor daemon;
        face-tracking and diagnostics would restore to 'on' regardless of prior
        state; the two thread-backed skills would be STARTED by an exit.

    FAILS SAFE IN BOTH DIRECTIONS. If the probe is missing or raises we neither
    touch the lever nor record a restore. Losing one suppression costs him some
    CPU for one session; arming an input injector costs him the match — and
    unlike the CPU, the injector persists to disk."""
    if not callable(fn):
        notes.append(f"{name}: unavailable in this build")
        return
    if not callable(was_on):        # pragma: no cover - programmer error
        notes.append(f"{name}: SKIPPED (no state probe — refusing to record an "
                     f"undo that could TURN IT ON)")
        return
    try:
        prior_on = bool(was_on())
    except Exception as e:
        notes.append(f"{name}: SKIPPED (couldn't read prior state: {e})")
        return
    if not prior_on:
        # Already off. Do NOT call and do NOT record. Every 'off' below
        # early-returns on this same state, so calling changes nothing — but
        # recording would make the EXIT path an ENABLE.
        notes.append(f"{name}: already off — left alone (no restore recorded)")
        return
    try:
        fn()
        applied[name] = restore
        notes.append(f"{name}: stopped")
    except Exception as e:
        notes.append(f"{name}: FAILED ({e})")


def _suppressor_event_probe(owner, attr: str):
    """`was_on` for a threading.Event that acts as a SUPPRESSOR: set == the
    thing is ALREADY suppressed, so on == not set. Both monolith face-tracking
    switches (_face_track_pause, _face_track_camera_off) are this shape.

    Raises when the Event is absent — _lever turns that into a SKIP, which is
    the safe direction: a pause we never applied must never be 'resumed'. It
    also stops us stealing a pause the speech path owns."""
    def _probe():
        ev = getattr(owner, attr, None) if owner is not None else None
        if ev is None or not hasattr(ev, "is_set"):
            raise AttributeError(f"{attr} is not an Event on the monolith")
        return not bool(ev.is_set())
    return _probe


def _module_state_probe(mod, attr: str, kind: str):
    """`was_on` for a skill whose state is a module global: a thread handle
    (kind='thread') or a single-cell list flag (kind='cell', e.g. guard's
    _armed). Read at CALL time, never captured, so a skill that started between
    load and suspend is still seen."""
    def _probe():
        if mod is None:
            raise AttributeError("module not loaded")
        sentinel = object()
        val = getattr(mod, attr, sentinel)
        if val is sentinel:
            raise AttributeError(f"{attr} not present")
        if kind == "thread":
            return val is not None and bool(val.is_alive())
        return bool(val[0])
    return _probe


def _idle_cfg_flag(label: str, flag: str, notes: list, applied: dict,
                   instead_of: str = "") -> None:
    """Idle a live feature by flipping its core.config constant OFF IN PROCESS,
    recording the restore. NEVER calls that feature's own `*_off` ACTION.

    THIS IS L1's LOAD-BEARING RULE, NOT A STYLE CHOICE. The kinect skills'
    spoken actions PERSIST:

        gestures_off  -> _set_gestures_enabled(False) -> _persist_setting
                      -> tools.settings_window.save_settings   (atomic rewrite
                         of data/user_settings.json)
                         skills/kinect_gestures.py:450-478
        air_mouse_off -> _set_enabled(False)          -> _persist_setting
                         skills/kinect_air_mouse.py:3463-3490

    A persisted `false` SURVIVES THE POWER BUTTON, and the power button is the
    documented end state of the exact scenario this skill exists for: Fortnite's
    lobby leak -> E_OUTOFMEMORY -> RenderThread hang -> hard reset. On that path
    `_restore_luxuries` never runs, `core/config.py:_apply_user_settings()`
    reads the persisted `false` at the next boot, and — because
    KINECT_GESTURES_ENABLED SHIPS False, so no default puts it back — gesture
    control is dead, silently and permanently, with nothing to tell him why.
    Calling those actions made this module's own L1 promise ("ANY restart ...
    restores everything with zero cleanup code") false.

    Flipping the live constant is both restart-safe AND sufficient, because
    every gate re-reads core.config FRESH each tick:
        kinect_gestures._gestures_enabled()   -> _cfg_flag (kinect_gestures:124)
        kinect_air_mouse._air_mouse_enabled() -> _cfg_flag (kinect_air_mouse:2297)
        kinect_two_hand                       -> _cfg_flag (kinect_two_hand:571)
    core.config is the ONLY consumer of both flags (verified across the tree —
    the monolith keeps no mirror to gate on), and the air-mouse's disabled
    branch honours pending 'up' edges via `_mouse_button("up", button)`
    (skills/kinect_air_mouse.py:3355-3360), so a flip DURING A DRAG still
    releases a held button — no stuck mouse button mid-match.

    Anyone who "simplifies" this back into a `gestures_off("")` call has
    reintroduced a permanent, silent disabling of his gesture control. There is
    a test for that.
    """
    try:
        import core.config as _c
        sentinel = object()
        prev = getattr(_c, flag, sentinel)
        if prev is sentinel:
            notes.append(f"{label}: {flag} not declared in this build")
            return
        if not bool(prev):
            notes.append(f"{label}: already off")
            return
        setattr(_c, flag, False)
        applied[label] = lambda p=prev: setattr(_c, flag, p)
        note = f"{label}: idled ({flag}=False, was {prev!r})"
        if instead_of:
            note += f" — deliberately NOT {instead_of}, which persists to disk"
        notes.append(note)
    except Exception as e:
        notes.append(f"{label}: FAILED ({e})")


class _HeldEvent(threading.Event):
    """A threading.Event whose clear() is IGNORED until release().

    It exists for exactly one reason, spelled out in _hold_face_detection():
    the monolith's pause/resume pair is a bare set()/clear() on a SHARED,
    un-ref-counted Event, and the main voice loop clears it on every listen
    cycle. Latching that Event is the only way to hold the pause from outside
    bobert_companion.py without editing it.

    `suppressed` counts the clears it swallowed — a MEASURED number, printed on
    release, so "the hold held" is never a thing we report on faith."""

    def __init__(self) -> None:
        super().__init__()
        self.suppressed = 0
        self._released = False

    def clear(self) -> None:
        if not self._released:
            self.suppressed += 1
            return
        super().clear()

    def release(self) -> None:
        """Stop latching; from here it behaves as a plain Event."""
        self._released = True
        super().clear()


def _hold_face_detection(bc, notes: list, applied: dict) -> None:
    """Hold the Haar-cascade pause for the WHOLE session, not for one breath.

    THE DEFECT THIS REPLACES (verified in the live tree, 2026-09-04): a one-shot
    pause_face_tracking() does not survive a single listen cycle. It is a bare
    `_face_track_pause.set()` on a SHARED, un-ref-counted threading.Event
    (bobert_companion.py:7924-7936), and the main voice loop calls
    resume_face_tracking() — a bare `.clear()` — on EVERY iteration, immediately
    before record_speech(timeout=20) (bobert_companion.py:23917, and :23893 on
    the realtime path). So within at most ~20 s of engaging mid-match the
    detection this file measures as the largest CPU item on the box (pythonw at
    2.31 cores continuous, 32 threads carrying 83 % of it) was back for the rest
    of the match — while _st.notes still said "face_tracking_detect: stopped",
    game_mode_status implied the lever was held, and _restore_luxuries later
    logged "restored" after calling a resume that was by then a no-op. A mode
    that says it stopped something it could not stop is exactly what _lever's
    docstring says this module exists NOT to be.

    THE FIX, WITH NO MONOLITH EDIT: every reader resolves `_face_track_pause` as
    a MODULE GLOBAL at call time — the detection loop (bobert_companion.py:5835,
    re-read once per camera frame, gating the `continue` at :6121 and :6158),
    pause_face_tracking (:7933) and resume_face_tracking (:7936). Swapping the
    module attribute for a latched Event therefore takes effect within one frame
    and holds against every caller, the voice loop included.

    RESTORE IS EXACT, NOT OPINIONATED. We put the ORIGINAL Event object back
    with the flag state it had at entry, so _lever's never-switch-on-what-was-off
    guarantee still holds here: if the SPEECH path owned the pause when we
    engaged, it still owns it when we leave, and if detection was running before,
    it is running after. Nothing is inferred.

    FAILS SAFE: if `_face_track_pause` is not an Event we can latch, we do NOT
    pause and we do NOT record an undo — the same SKIP the state probe already
    produces, and for the same reason: a one-shot pause here would be cleared
    inside one listen window while the note claimed a sustained CPU saving.
    """
    ev = getattr(bc, "_face_track_pause", None) if bc is not None else None
    latchable = bc is not None and all(
        callable(getattr(ev, m, None)) for m in ("set", "clear", "is_set"))
    if not latchable:
        notes.append(
            "face_tracking_detect: SKIPPED (_face_track_pause is not a "
            "latchable Event on this build, so the pause could not be HELD — "
            "the voice loop's per-cycle resume_face_tracking() clears a "
            "one-shot pause inside one listen window (~20 s). No CPU saving "
            "is claimed.)")
        return

    try:
        was_set = bool(ev.is_set())
        held = _HeldEvent()
        held.set()      # set BEFORE the swap — never a frame with neither held
        setattr(bc, "_face_track_pause", held)
    except Exception as e:
        notes.append(f"face_tracking_detect: FAILED to latch ({e})")
        return

    def _release(_bc=bc, _orig=ev, _held=held, _was_set=was_set):
        current = getattr(_bc, "_face_track_pause", None)
        if current is _held:
            # Exact restore: hand the ORIGINAL object back in the state it was
            # in when we took it, never in the state we would prefer.
            if _was_set:
                _orig.set()
            else:
                _orig.clear()
            setattr(_bc, "_face_track_pause", _orig)
        _held.release()         # never leave the latch armed, either way
        print(f"{_LOG}   face_tracking_detect: held for the whole session; "
              f"ignored {_held.suppressed} resume(s) from the voice loop")
        if current is not _held:
            raise RuntimeError(
                "_face_track_pause was replaced while game mode held it — the "
                "latch is disarmed but the original Event was NOT put back")

    applied["face_tracking_detect"] = _release
    notes.append(
        "face_tracking_detect: HELD (latched Event — the voice loop's per-cycle "
        f"resume_face_tracking() can no longer clear it; was "
        f"{'already paused' if was_set else 'running'} at entry)")


def _suspend_luxuries(notes: list, applied: dict) -> None:
    """Stop the expensive/harmful background work. All resolved dynamically —
    the seven daemons a rival design promised to silence (teams_nudge,
    screen_watch, banter, chappie_consciousness, personal_rag) have NO stop API
    at all in this tree, and their threads are already running old bytes in the
    live process, so no flag added tonight could reach them. We do not pretend
    otherwise. It does not matter much: the downshift already made every one of
    them cost ~9 GB instead of ~15 GB."""
    bc = _bc()

    # 1. THE CORRECTNESS HAZARDS FIRST — free, instant, and the only item here
    # that can cost him a match. The live log shows "[two-hand] both hands
    # raised — TWO-HAND mode engaged" at 20:30:39 (x4), 20:33:28, 20:34:24
    # (x2), 20:35:55 — his hands on keyboard and mouse read as a grab gesture,
    # and a closed grip registers as a CLICK (it has closed his Chrome tabs
    # before). A stray click mid-match beats every memory number on this list.
    for mod_name, off_name, on_name, flag in (
            ("skill_kinect_gestures", "gestures_off", "gestures_on",
             "KINECT_GESTURES_ENABLED"),
            ("skill_kinect_air_mouse", "air_mouse_off", "air_mouse_on",
             "KINECT_AIR_MOUSE_ENABLED"),
    ):
        name = mod_name.replace("skill_", "")

        # READ THE OWNER'S STATE FIRST. These two are the only levers here whose
        # restore WRITES TO DISK, and their ON action persists
        # UNCONDITIONALLY: kinect_air_mouse.air_mouse_on runs
        # `persisted = _set_enabled(True)` BEFORE the "already on" early return
        # (kinect_air_mouse.py:3538), and kinect_gestures.gestures_on does the
        # same (kinect_gestures.py:526) -> _persist_setting ->
        # tools.settings_window.save_settings -> data/user_settings.json.
        # Meanwhile both OFF actions early-return with no exception and no
        # state when the flag is already clear ("The air-mouse is already off,
        # sir." at :3555). So _lever, which can only see an exception, recorded
        # a restore for a lever that had stopped nothing — and the exit then
        # "restored" a feature the owner had deliberately switched OFF, ON, and
        # persisted it through every future reboot while logging
        # "kinect_air_mouse: restored".
        #
        # Measured 2026-09-04, read-only, from his live data/user_settings.json:
        # KINECT_AIR_MOUSE_ENABLED = False, KINECT_GESTURES_ENABLED = True. So
        # the very first Fortnite exit would have turned the air-mouse on for
        # good — the feature whose closed-fist click has closed his Chrome tabs.
        # The levers that DO capture prior state (kinect_two_hand just below,
        # cv2_threads, hud_camera_preview, kinect_bridge) are the pattern; these
        # two simply skipped it.
        #
        # Gate on the live flag: a feature already off is LEFT ALONE and gets no
        # restore, so game mode can never switch anything on that was not on
        # when he started playing.
        #
        # ...AND THESE TWO ARE NOT DRIVEN THROUGH THEIR ACTIONS AT ALL.
        # off_name/on_name both PERSIST: entering game mode via gestures_off()
        # writes KINECT_GESTURES_ENABLED:false into data/user_settings.json,
        # and that write SURVIVES THE POWER BUTTON — the documented end state of
        # the very scenario this skill exists for (lobby leak -> E_OUTOFMEMORY
        # -> RenderThread hang -> hard reset). _restore_luxuries never runs on
        # that path, so the next boot's _apply_user_settings() reads the
        # persisted false and his gesture control is dead for good, silently,
        # with nothing to tell him why. That also made L1's "this module writes
        # NO files at all" plainly false.
        #
        # _idle_cfg_flag flips the live core.config constant instead: same
        # effect on the running poller (both gates re-read core.config every
        # tick), nothing on disk, and it captures `prev` itself — so a feature
        # already off is left alone with no restore recorded, which is the same
        # never-switch-on-what-was-off guarantee the _lever gate gives the
        # levers below. Full chain and the proof it is sufficient (including the
        # held-mouse-button release) live in _idle_cfg_flag's docstring.
        _idle_cfg_flag(name, flag, notes, applied,
                       instead_of=f"{off_name}()/{on_name}()")

    # kinect_two_hand registers no on/off action. Its poller re-reads
    # KINECT_TWO_HAND_ENABLED from core.config on EVERY tick
    # (skills/kinect_two_hand.py:571 via _cfg_flag, whose docstring says "Read
    # fresh each call so a Settings toggle takes effect with no restart"), so
    # flipping the config constant idles it live — the correct lever, and the
    # reason this works without touching a running thread.
    try:
        import core.config as _c
        prev_th = getattr(_c, "KINECT_TWO_HAND_ENABLED", True)
        _c.KINECT_TWO_HAND_ENABLED = False
        applied["kinect_two_hand"] = (
            lambda p=prev_th: setattr(_c, "KINECT_TWO_HAND_ENABLED", p))
        notes.append("kinect_two_hand: idled (KINECT_TWO_HAND_ENABLED=False)")
    except Exception as e:
        notes.append(f"kinect_two_hand: FAILED ({e})")

    # 2. Face tracking — BOTH FLAGS. The measured trap: camera_off alone does
    # NOT skip detection. pause_face_tracking()'s own docstring says the capture
    # loop keeps grabbing frames and only `paused` "skips the cv2 cascade
    # recognition cost". Setting only the intuitive flag saves nothing while
    # looking like it worked. This is the largest CPU item on the box (pythonw
    # measured at 2.31 cores continuous; 32 threads carrying 83 % of it).
    # NOTE: we claim the Haar-cascade CPU ONLY. We do NOT claim the frame
    # memcpy — the capture loop keeps grabbing under both flags, per that same
    # docstring, so a bandwidth claim here would be unmeasured.
    cam_off = getattr(bc, "set_face_tracking_camera_off", None) if bc else None
    cam_on = getattr(bc, "clear_face_tracking_camera_off", None) if bc else None
    # The DETECT pause cannot go through _lever. pause_face_tracking() is a bare
    # set() on a shared un-ref-counted Event that the main voice loop clears on
    # every listen iteration (bobert_companion.py:23917, :23893), so _lever would
    # record "stopped" for a lever that stops for one breath — and the largest
    # CPU item here would be a claim, not a saving. _hold_face_detection latches
    # it instead, and restores the original Event in its entry state on exit, so
    # the never-switch-on-what-was-off guarantee still holds.
    _hold_face_detection(bc, notes, applied)
    # camera_off IS a plain suppressor Event nobody else clears, so it keeps the
    # state-probed lever: SET means already-suppressed, and probing stops us
    # 'restoring' a camera that was off before he started.
    _lever("face_tracking_camera", cam_off, cam_on, notes, applied,
           _suppressor_event_probe(bc, "_face_track_camera_off"))

    # 3. OpenCV thread cap. There is no setNumThreads call anywhere in the tree
    # (verified), so OpenCV currently takes all 32 logical processors.
    try:
        import cv2
        prev = cv2.getNumThreads()
        cv2.setNumThreads(2)
        applied["cv2_threads"] = lambda p=prev: cv2.setNumThreads(p)
        notes.append(f"cv2_threads: 2 (was {prev})")
    except Exception as e:
        notes.append(f"cv2_threads: unavailable ({e})")

    # 4. HUD camera preview producer. Read LIVE via getattr each frame
    # (bobert_companion.py:3837), so flipping the module global takes effect
    # with no restart and no monolith edit.
    if bc is not None:
        try:
            prev = getattr(bc, "HUD_CAMERA_PREVIEW", True)
            setattr(bc, "HUD_CAMERA_PREVIEW", False)
            applied["hud_camera_preview"] = (
                lambda p=prev: setattr(bc, "HUD_CAMERA_PREVIEW", p))
            notes.append("hud_camera_preview: off")
        except Exception as e:
            notes.append(f"hud_camera_preview: FAILED ({e})")

    # 5. Kinect streams — frees a measured 71 MB of 3090 VRAM plus continuous
    # USB3 bandwidth (blamed for webcams going black under contention).
    try:
        from audio import kinect_bridge as _kb
        stop_pump = getattr(_kb, "stop_body_pump", None)
        set_en = getattr(_kb, "set_enabled", None)
        was_enabled = bool(getattr(_kb, "get_enabled", lambda: False)())

        def _kinect_off():
            if callable(stop_pump):
                stop_pump()
            if callable(set_en):
                set_en(False)

        def _kinect_on(prev=was_enabled):
            if callable(set_en):
                set_en(prev)
            start = getattr(_kb, "start_body_pump", None)
            if prev and callable(start):
                start()

        # _kinect_on already restores `prev`, so this one could not switch the
        # bridge ON — but the probe is still required, and it is honest about
        # the second thing worth stopping: a pump thread still alive after a
        # disable. ON here means "there is something to stop".
        _pump_alive = getattr(_kb, "_pump_is_alive", None)

        def _kinect_was_on(prev=was_enabled, probe=_pump_alive):
            if prev:
                return True
            try:
                return bool(probe()) if callable(probe) else False
            except Exception:
                return False

        _lever("kinect_bridge", _kinect_off, _kinect_on, notes, applied,
               _kinect_was_on)
    except Exception as e:
        notes.append(f"kinect_bridge: unavailable ({e})")

    # 6. Diagnostic daemons — DELIBERATELY NOT PAUSED. THE PAUSE *IS* A DISK
    # WRITE, and this was the last hole in L1 (found 2026-09-04, read-only).
    #
    # pause_diagnostics() is not a process-memory cell. It goes _update_state ->
    # _write_state -> _atomic_write_json(data/diagnostic_daemons.json)
    # (core/diagnostic_daemons.py:1445, :276, :263) and all FOUR daemon loops
    # re-read that FILE every tick (:353, :519, :927, :1355). There is no
    # in-process equivalent, and wrapping _read_state in memory does not make
    # one: each loop's first act is _update_state(... alive_ts ...), which reads
    # THROUGH the wrapper and writes the injected paused:true straight back out.
    #
    # So it is the one lever that cannot satisfy L1, and the failure is exactly
    # the scenario this skill exists for: Fortnite's lobby leak ->
    # E_OUTOFMEMORY -> RenderThread hang -> power button. _restore_luxuries
    # never runs, `paused: true` is still on disk at the next boot, and
    # self-diag, crash-watch, deep-audit and anomaly-watch stay silently paused
    # FOREVER — including the crash-watch that would have caught his next
    # freeze. Permanently trading away the watchdog to buy one game session is
    # the wrong trade, and it is invisible: nothing ever tells him.
    #
    # The saving was never measured, either. The three local daemons are
    # interval pollers holding no model (SELF_DIAG_INTERVAL_S = 300 s), and the
    # "paid deep audit" calls claude-sonnet-5 (:122) — a CLOUD model this box
    # cannot reach at all (AI_BACKEND=ollama, no key; see the header). Zero
    # VRAM, zero resident GB. Rule 3 above applies: no measurement, no claim,
    # and here not even a cost to justify the risk.
    #
    # stop_diagnostic_daemons() is not the way out: it joins four threads at up
    # to 5 s each, and start_diagnostic_daemons() rewrites the state file on the
    # way back up (:1384), so the RESTORE persists too.
    #
    # "Pause diagnostics" is his own spoken action. Game mode does not make that
    # decision for him and then lose the receipt.
    notes.append("diagnostics: left running on purpose — pausing them writes "
                 "data/diagnostic_daemons.json, which outlives the power button")

    # 7. Skills with a REAL stop API. Anything without one is simply absent from
    # this list rather than pretended at.
    # Each carries the module global that says whether it is actually running.
    # guard_mode is the one that matters most: guard_off() returns "I wasn't on
    # watch, sir." without disarming, so the recorded restore would have ARMED
    # guard mode and started a camera monitor daemon on exit — spending CPU and
    # cameras on a watch he never asked for.
    for mod_name, off_name, on_name, label, state_attr, kind in (
            ("skill_ambient_multimodal_extract", "ambient_extract_stop",
             "ambient_extract_start", "ambient_extract", "_thread", "thread"),
            ("skill_guard_mode", "guard_off", "guard_on", "guard_mode",
             "_armed", "cell"),
            ("skill_standby_audio_detect", "stop_background_loop",
             "_start_background_loop", "standby_audio_detect",
             "_loop_thread", "thread"),
    ):
        mod = sys.modules.get(mod_name)
        off = getattr(mod, off_name, None) if mod else None
        on = getattr(mod, on_name, None) if mod else None

        def _call(f):
            if f is None:
                return None
            def _run(_f=f):
                try:
                    _f("")
                except TypeError:
                    _f()
            return _run

        _lever(label, _call(off), _call(on), notes, applied,
               _module_state_probe(mod, state_attr, kind))

    # DELIBERATELY NOT TOUCHED:
    #  * Whisper large-v3-turbo stays on the 1650 SUPER (cuda:1) — it costs the
    #    game ZERO 3090 VRAM and it is what makes JARVIS hear. Never set
    #    STANDBY_WHISPER_PREFER_GPU.
    #  * Kokoro TTS stays (CPU-only, torch-free) — it is the voice.
    #  * skills/ambient_listen is NOT closed. Tearing down PortAudio streams is
    #    the documented 0xc0000374 heap-corruption race; crashing JARVIS
    #    mid-game is strictly worse than two idle streams. If anyone later
    #    "optimises" this into a stream close, that is a regression, not a
    #    cleanup.
    #  * The tray stays — it is the LLM-independent control plane, the one
    #    channel that still works when the brain is wedged.
    #  * The reticle overlay is NOT stopped. The theory that a 7680x2880
    #    always-on-top layered window costs the game exclusive fullscreen is
    #    plausible and untested; shipping an unmeasured guess is the defect
    #    class this file exists to avoid.
    #  * The diagnostic daemons are NOT paused — the pause is a write to
    #    data/diagnostic_daemons.json that four loops read back from disk, so it
    #    survives the power button and would leave the crash-watch dead
    #    forever. Full reasoning at lever 6 above. Pinned by
    #    tests/skills/test_game_mode_no_persistence.py.


def _restore_luxuries(notes: list, applied: dict) -> None:
    """Reverse order, each in its own try/except so one failure never blocks the
    other twelve. Failures are REPORTED, never swallowed into a clean-looking
    restore."""
    for name in reversed(list(applied.keys())):
        fn = applied.get(name)
        if not callable(fn):
            notes.append(f"{name}: nothing to restore")
            continue
        try:
            fn()
            notes.append(f"{name}: restored")
        except Exception as e:
            notes.append(f"{name}: RESTORE FAILED ({e})")
    applied.clear()


# ── Brain downshift / restore ────────────────────────────────────────────
def _current_brain() -> str:
    bc = _bc()
    if bc is not None:
        cache = getattr(bc, "_RESOLVED_LOCAL_LLM_MODEL", None)
        try:
            if cache and cache[0]:
                return str(cache[0])
        except Exception:
            pass
    return str(_cfg("LOCAL_LLM_MODEL", "") or "")


def _repoint_brain(new_tag: str, notes: list) -> dict:
    """Repoint chat + vision at `new_tag`, IN PROCESS ONLY, and return what to
    put back.

    persist=False is load-bearing. skills/model_picker.set_model() ALWAYS
    persists (_persist_setting + _sync_vision_to_chat(persist=True) at
    skills/model_picker.py:545), which would permanently rewrite his brain
    choice in data/user_settings.json — and a crash would then strand him on
    the small model forever. We never call set_model(). Because nothing
    persists, L1 holds: any restart restores everything with no cleanup code.
    """
    bc = _bc()
    prev = {
        "resolved": None,
        "llm": getattr(bc, "LOCAL_LLM_MODEL", None) if bc else None,
        "vision": getattr(bc, "LOCAL_VISION_MODEL", None) if bc else None,
    }
    old_tag = _current_brain()
    cache = getattr(bc, "_RESOLVED_LOCAL_LLM_MODEL", None) if bc else None
    try:
        if cache is not None:
            prev["resolved"] = cache[0]
            cache[0] = new_tag
    except Exception as e:
        notes.append(f"brain: resolver cache not repointed ({e})")
    if bc is not None:
        try:
            setattr(bc, "LOCAL_LLM_MODEL", new_tag)
        except Exception as e:
            notes.append(f"brain: LOCAL_LLM_MODEL not repointed ({e})")
    try:
        import core.config as _c
        prev["cfg_llm"] = getattr(_c, "LOCAL_LLM_MODEL", None)
        _c.LOCAL_LLM_MODEL = new_tag
    except Exception:
        pass

    # Vision must follow in LOCKSTEP or a vision call reloads the tag we are
    # about to unload. Prefer model_picker's shared rule (it correctly refuses
    # to touch a user-PINNED separate VLM); if it declines while vision is
    # still pointing at the tag we are unloading, close the hole explicitly and
    # say so — a silent hole here is a guaranteed 15 GB reload later.
    synced = False
    mp = sys.modules.get("skill_model_picker")
    sync = getattr(mp, "_sync_vision_to_chat", None) if mp else None
    if callable(sync):
        try:
            synced = bool(sync(old_tag, new_tag, persist=False, bc=bc))
        except Exception as e:
            notes.append(f"vision: lockstep sync raised ({e})")
    cur_vision = getattr(bc, "LOCAL_VISION_MODEL", None) if bc else None
    if synced:
        notes.append(f"vision: lockstepped to {new_tag}")
    elif cur_vision and _tag_base(cur_vision) == _tag_base(old_tag):
        if _is_multimodal(new_tag):
            try:
                if bc is not None:
                    setattr(bc, "LOCAL_VISION_MODEL", new_tag)
                import core.config as _c
                prev["cfg_vision"] = getattr(_c, "LOCAL_VISION_MODEL", None)
                _c.LOCAL_VISION_MODEL = new_tag
                notes.append(f"vision: repointed directly to {new_tag} "
                             f"(lockstep helper declined)")
            except Exception as e:
                notes.append(f"vision: direct repoint FAILED ({e})")
        else:
            notes.append(f"vision: LEFT on {cur_vision} — {new_tag} is not "
                         f"multimodal; a vision call will still reload it")
    else:
        notes.append(f"vision: left on {cur_vision} (pinned separately)")
    return prev


def _restore_brain(prev: dict, notes: list) -> None:
    bc = _bc()
    cache = getattr(bc, "_RESOLVED_LOCAL_LLM_MODEL", None) if bc else None
    try:
        if cache is not None and prev.get("resolved"):
            cache[0] = prev["resolved"]
    except Exception as e:
        notes.append(f"brain: resolver restore FAILED ({e})")
    for attr, key in (("LOCAL_LLM_MODEL", "llm"),
                      ("LOCAL_VISION_MODEL", "vision")):
        val = prev.get(key)
        if val and bc is not None:
            try:
                setattr(bc, attr, val)
            except Exception as e:
                notes.append(f"brain: {attr} restore FAILED ({e})")
    try:
        import core.config as _c
        for attr, key in (("LOCAL_LLM_MODEL", "cfg_llm"),
                          ("LOCAL_VISION_MODEL", "cfg_vision")):
            if prev.get(key):
                setattr(_c, attr, prev[key])
    except Exception:
        pass
    notes.append(f"brain: restored to {prev.get('resolved') or prev.get('llm')}")


def _flush_pending_brain(notes: list) -> bool:
    """Pay back a brain restore that was DEFERRED because a game was running.

    THIS IS THE ONLY THING THAT EVER PAYS THAT DEBT. decide() returns "exit"
    only while st.active is True, so the moment _exit() sets active=False with
    the brain still downshifted, nothing in the watcher can reach the restore
    again — it has to be paid from a path that runs while the mode is OFF.
    Callers hold _lock. The debt is cleared FIRST so a broken restore can never
    spin the poll loop forever; the failure is recorded, never swallowed."""
    prev = _st.pending_brain
    if prev is None:
        return False
    _st.pending_brain = None
    try:
        _restore_brain(prev, notes)
    except Exception as e:                    # pragma: no cover - defensive
        notes.append(f"brain: deferred restore raised ({e})")
        _assert_brain_usable(notes)
        return False
    _assert_brain_usable(notes)
    return True


def _assert_brain_usable(notes: list) -> bool:
    """INVARIANT: after ANY transition, partial or crashed, the chat brain must
    still point at a non-empty tag. This is the "a crash mid-transition cannot
    leave JARVIS muted" guarantee, checked rather than assumed."""
    tag = _current_brain()
    if tag and tag.strip() and tag.strip().lower() not in ("off", "none"):
        return True
    notes.append(f"INVARIANT VIOLATED: chat brain is {tag!r} — restoring config "
                 f"default so JARVIS is never muted")
    bc = _bc()
    try:
        import core.config as _c
        fallback = getattr(_c, "_SHIPPED_LOCAL_LLM_MODEL", None) or \
            getattr(_c, "LOCAL_LLM_MODEL", None)
    except Exception:
        fallback = None
    if fallback and bc is not None:
        try:
            cache = getattr(bc, "_RESOLVED_LOCAL_LLM_MODEL", None)
            if cache is not None:
                cache[0] = fallback
            setattr(bc, "LOCAL_LLM_MODEL", fallback)
        except Exception:
            pass
    return False


# ── Enter / exit ──────────────────────────────────────────────────────────
def _enter(pid, exe, manual: bool = False, *, defer_verify: bool = False) -> str:
    """Engage. ORDERING IS LOAD-BEARING AND MUST NOT BE 'TIDIED':

        measure -> DOWNSHIFT -> unload -> suspend -> re-measure

    The downshift comes BEFORE the unload because the downshift IS the guard.
    Unloading first would open a window in which any of ~10 daemon threads
    cold-loads the FULL 15 GB brain into the gap we just made — which is the
    crash we are preventing. If the downshift fails we do NOT unload at all: a
    sawtooth is survivable, a guaranteed 15 GB reload against a 766 MiB-free
    card is not.

    defer_verify=True IS THE VOICE PATH, AND IT IS NOT AN OPTIMISATION.
    Actions are invoked SYNCHRONOUSLY — `res = fn(arg)` inside _runner, called
    from parse_and_run_actions — on the MAIN LOOP thread: the same thread that
    runs record_speech() and ticks _heartbeat(). Every second spent in here is a
    second JARVIS is deaf AND silent, and the settle wait alone is a fixed 8 of
    them (plus a second nvidia-smi + /api/ps sample). So "divert power" in the
    middle of a match answered him with ten-plus seconds of dead air at the
    exact moment he asked for help — and one slow Ollama socket on top of that
    walks toward the 60 s mic-reset watchdog (_MAIN_LOOP_HEARTBEAT_TIMEOUT).
    Nothing about the settle wait needs the caller: it exists ONLY to let an
    unload show up in the driver's numbers before the 'after' sample.

    The levers are still applied INSIDE the lock before this returns either
    way, so "normal power" one second later still finds a mode to leave — the
    deferred half is the measurement, never the mode. The watcher keeps the
    inline path (defer_verify=False): it is already a background thread, and
    both paths run the SAME _settle_and_verify, so there is no second copy of
    the wait to rot."""
    with _lock:
        if _st.active:
            return "Game mode is already engaged, sir."
        notes: list = []
        before = _measure()
        old_tag = _current_brain()

        new_tag, why = _pick_game_brain(old_tag)
        downshifted = False
        prev_brain = {}
        if new_tag:
            prev_brain = _repoint_brain(new_tag, notes)
            downshifted = _current_brain() == new_tag
            notes.append(f"brain: {old_tag} -> {new_tag} ({why})"
                         if downshifted else f"brain: repoint did not stick")
        else:
            notes.append(f"brain: NOT downshifted — {why}")

        # Unload the big model ONLY once the small one is the resolved target.
        # Compare FULL TAGS, never base names: gemma4:26b-a4b-it-qat and
        # gemma4:12b share the base "gemma4", so a base-name comparison here
        # silently skipped the unload on the exact pair we ship — the whole
        # 15 GB saving, lost to a note claiming the downshift "did not take"
        # while it plainly had. Caught by test_enter_downshifts_... on the
        # first run of this file.
        if downshifted and old_tag and old_tag != new_tag:
            notes.append(f"unload {old_tag}: "
                         f"{'sent' if _unload(old_tag) else 'FAILED'}")
        elif not downshifted:
            notes.append("unload: skipped — the downshift did not take, so "
                         "unloading would only schedule a full cold reload")
        else:
            notes.append(f"unload: nothing to do — already on {new_tag}")

        _suspend_luxuries(notes, _st.applied)
        _assert_brain_usable(notes)

        _st.active = True
        _st.game_pid = pid
        _st.game_exe = exe
        _st.manual = manual
        _st.entered_at = entered_at = time.time()
        _st.last_seen_at = _st.entered_at
        _st.inhibit_pid = None
        _st.before = before
        _st.notes = notes
        # Record what we owe him on the STATE. NEVER overwrite an older debt:
        # re-entry after a deferred exit resolves prev_brain to the SMALL tag we
        # are already sitting on, so an unconditional write here replaces the
        # payload pointing at the big brain with one pointing at the small one —
        # destroying the restore outright while looking like bookkeeping. The
        # OLDEST outstanding payload wins; it is the true 'before'.
        if downshifted and _st.pending_brain is None:
            _st.pending_brain = prev_brain

    # Re-measure OUTSIDE the lock — an unload takes a moment to show up.
    if not defer_verify:
        return _settle_and_verify(entered_at)

    # VOICE PATH: hand the wait to a worker and give him a sentence NOW, so the
    # main loop gets straight back to listening. The sentence claims NOTHING
    # about memory — there is nothing measured yet to claim.
    t = threading.Thread(target=_settle_and_verify, args=(entered_at, True),
                         name="game-mode-verify", daemon=True)
    _verify_thread[0] = t
    t.start()
    if downshifted:
        return (f"Diverting power now, sir — I'm on the {new_tag} brain and "
                f"out of your way. I'm still listening; I'll tell you what it "
                f"actually freed in a moment.")
    return (f"Game mode is engaged, sir, but I could not move off the "
            f"{old_tag} brain, so it may free very little. I'll tell you what "
            f"it actually freed in a moment.")


def _settle_and_verify(entered_at: float, announce: bool = False) -> str:
    """Wait out the unload, then take the 'after' sample and report it.

    The wait is why this function exists: nvidia-smi and /api/ps do not show a
    freed allocation the instant the unload POST returns, so a sample taken too
    early under-reports the saving. It is ALSO the reason this half must be
    able to run off the caller's thread — see _enter's defer_verify note.

    `announce` belongs to the deferred path only. He asked out loud and got an
    acknowledgement instead of a number, so he is OWED the measured verdict: it
    goes out through the canonical proactive_announce queue (held, never
    dropped, if he is in focus mode) and is readable any time from
    game_mode_status. _verify already announces when GAME_MODE_ANNOUNCE is set,
    so the gate below is what stops him hearing it twice."""
    time.sleep(float(_cfg_float("GAME_MODE_VERIFY_DELAY_SECONDS", 8.0)))
    with _lock:
        # The session can END while we settle — "normal power" two seconds after
        # "divert power" is exactly what L5 exists to allow. Verifying then
        # would compute a delta for a mode that no longer exists and SPEAK it at
        # him. Matched against the specific entry this worker was spawned for,
        # inside the same RLock _verify takes, so the check and the measurement
        # cannot be split by another thread.
        if not _st.active or _st.entered_at != entered_at:
            print(f"{_LOG} verify skipped — that session ended while the "
                  f"unload was still settling")
            return ""
        msg = _verify()
    # Announce OUTSIDE the lock: proactive_announce writes a file.
    if announce and msg and not bool(_cfg("GAME_MODE_ANNOUNCE", False)):
        _announce(msg)
    return msg


def _on_game_brain() -> bool:
    """Is the resolver ACTUALLY pointed at the configured game brain right now?

    Re-derived live, never remembered, so it cannot go stale the way a boolean
    latched at entry does. FULL TAG comparison, never base names: the pair we
    ship is gemma4:26b-a4b-it-qat and gemma4:12b, which share the base
    "gemma4" — a base-name test would call the big brain a downshift. Only
    Ollama's ":latest" suffix is tolerated, exactly as `_keep_warm` does it. A
    GAME_MODE_BRAIN written as a bare base name that resolved to some other tag
    reads False here, which can cost us an "armed" verdict and can never buy a
    false one."""
    want = str(_cfg("GAME_MODE_BRAIN", "") or "").strip()
    cur = (_current_brain() or "").strip()
    if not want or not cur:
        return False
    if cur == want:
        return True
    if ":" not in want and cur == f"{want}:latest":
        return True
    if ":" not in cur and want == f"{cur}:latest":
        return True
    return False


def _classify(before: dict, after: dict, floor: int,
              brain: str, on_game_brain: bool) -> "tuple[str, str, object]":
    """(result, finished sentence, headline MB) from one before/after PAIR.

    Pure: no I/O, no state, no clock — so `_verify` and `game_mode_status`
    can both run it, the second one on a FRESH sample. That is the point of
    extracting it: `_st.result` used to be written once, eight seconds after
    entry, and then repeated for the rest of the session.

    THE RULE THIS ENCODES: the floor is a test of the UNLOAD, so it may only be
    applied to VRAM that was THERE TO FREE. `reclaimable_mb` is that number,
    measured from /api/ps at entry.

      * 0 MB reclaimable is not a failure. It means the card held no model when
        we engaged, so there was nothing to unload and the mode's whole value
        is the load it PREVENTS. We say exactly that, and claim no saving.
      * A saving is only ever reported from a real delta, and we take the
        LARGER of two honest measurements: the whole card (nvidia-smi) and the
        attributable model total (/api/ps). The card figure alone is not
        enough — the game allocates across our verify window, so a 14 GB
        unload can show up as a NEGATIVE card delta.
      * Unknown is never zero. No GPU reading, or no /api/ps reading, means
        "unverified" — not "failed", because a failure is itself a claim.
    """
    d = _delta(before or {}, after or {})
    gpu = d.get("vram_freed_mb")
    attributable = d.get("model_vram_freed_mb")
    reclaimable = d.get("reclaimable_mb")

    if gpu is None:
        # No card reading at all. /api/ps is Ollama's self-report; without the
        # DRIVER's number there is nothing to check it against, so we score
        # nothing rather than score it on one source. Unchanged behaviour, and
        # pinned by test_unreadable_gpu_claims_nothing.
        return ("unverified",
                "Game mode is engaged, sir, but I could not read the GPU "
                "before and after, so I have no measured saving to report.",
                None)
    freed = max(v for v in (gpu, attributable) if v is not None)

    if freed >= floor:
        ram = d.get("ram_freed_mb")
        ram_s = f" and {ram} MB of RAM" if ram and ram > 0 else ""
        how = ("on the card" if freed == gpu
               else "in resident models, from /api/ps — the card total was "
                    "masked by the game allocating across the same window")
        return ("ok",
                f"Game mode engaged, sir — {freed} MB of VRAM{ram_s} freed, "
                f"measured {how}. I'm on the {brain} brain until you're done.",
                freed)

    if reclaimable is None:
        return ("unverified",
                f"Game mode is engaged, sir, but I could not read which models "
                f"were resident, so I cannot tell you whether there was "
                f"anything to free — I'm claiming no saving. I'm on the "
                f"{brain} brain.", freed)

    if reclaimable >= floor:
        return ("failed",
                f"Game mode engaged but it did NOT free what it should have, "
                f"sir — {reclaimable} MB of model was resident when I engaged "
                f"and only {freed} MB came back, against a {floor} MB floor. "
                f"Treat that as a failure, not a saving.", freed)

    # Nothing of size was resident, so there was nothing to free and the floor
    # cannot be failed. The only question left is whether the downshift — the
    # half that prevents the NEXT load — actually took.
    if not on_game_brain:
        return ("failed",
                f"Game mode is engaged but it changed nothing, sir — there was "
                f"no model resident to free and the downshift did not take, so "
                f"I'm still on {brain}. Treat that as a failure, not a saving.",
                freed)
    had = ("no model was resident" if not reclaimable
           else f"only {reclaimable} MB of model was resident")
    return ("armed",
            f"Game mode is engaged, sir, and there was nothing to reclaim: "
            f"{had} on the GPU when I engaged, so I freed nothing and I won't "
            f"call that a saving. What it buys you is the next load — I'm on "
            f"the {brain} brain, so the next thing that wakes me loads the "
            f"small model instead of the big one. A load avoided, not memory "
            f"freed.", freed)


def _verify() -> str:
    """The 'after' half of the pair, taken by the same _measure() as the
    'before' and scored by `_classify()` — which game_mode_status() also
    calls, so the verdict is a function of the samples rather than a sentence
    remembered from eight seconds after entry."""
    with _lock:
        if not _st.active or not _st.before:
            return "Game mode is not engaged, sir."
        after = _measure()
        _st.after = after
        floor = _cfg_int("GAME_MODE_MIN_VRAM_DELTA_MB", 3000)
        brain = _current_brain()
        result, msg, freed = _classify(_st.before, after, floor,
                                       brain, _on_game_brain())
        _st.result = result
        print(f"{_LOG} {msg}")
        for n in _st.notes:
            print(f"{_LOG}   {n}")
        _hud(game_mode=True, game_mode_app=_st.game_exe,
             game_mode_result=result,
             game_mode_vram_freed_mb=freed,
             game_mode_brain=brain)
        if bool(_cfg("GAME_MODE_ANNOUNCE", False)):
            _announce(msg)
        return msg


def _exit(reason: str = "", inhibit: bool = False) -> str:
    """Disengage and restore. EXIT FAILS OPEN — if state is missing or corrupt
    we restore from defaults and say we did, because being unsure must never be
    a reason to STAY in low-power."""
    with _lock:
        if not _st.active:
            return "Game mode isn't engaged, sir."
        notes: list = []
        before = _measure()

        # Do NOT re-warm the big brain while a game is still running — that is
        # exactly the ~15 GB spike this feature exists to remove. A deadman
        # exit with the game still up restores the luxuries and leaves the
        # small brain in place; the watcher re-enters on the next tick, which
        # is cheap precisely because nothing has to load.
        still_gaming = _find_game_pid()[0] is not None

        _restore_luxuries(notes, _st.applied)
        deferred = False
        if _st.pending_brain is not None and not still_gaming:
            _flush_pending_brain(notes)
        elif _st.pending_brain is not None:
            deferred = True
            notes.append("brain: LEFT downshifted — an allowlisted game is "
                         "still running and re-warming the big brain now would "
                         "be the 15 GB spike we exist to avoid; the restore is "
                         "held and _tick() pays it the moment that game closes")
        _assert_brain_usable(notes)

        if inhibit:
            _st.inhibit_pid = _st.game_pid
        exe = _st.game_exe
        _st.active = False
        _st.manual = False
        _st.fg_pid = None
        _st.fg_since = None
        _st.result = None
        after = _measure()
        _st.notes = notes
        _hud(game_mode=False, game_mode_app=None, game_mode_result=None,
             game_mode_brain=_current_brain())
        d = _delta(before, after)
        print(f"{_LOG} exited ({reason or 'manual'}); "
              f"VRAM delta on restore: {d.get('vram_freed_mb')} MB")
        for n in notes:
            print(f"{_LOG}   {n}")
        failed = [n for n in notes if "FAILED" in n]
        # Say what is TRUE. "Back to normal power ... I'm on <tag> again" over a
        # brain that is still downshifted is this repo's defining defect said
        # out loud — and it is what stopped anyone noticing the debt was never
        # paid. When we hold the brain back, lead with that and name the trade.
        if deferred:
            held = (f"Everything else is back to normal power, sir, but I've "
                    f"stayed on the {_current_brain()} brain — "
                    f"{exe or 'a game'} is still running and loading the big "
                    f"model now is the ~15 GB spike that would take your match "
                    f"down. I'll put it back the moment the game closes.")
            if failed:
                held += (f" {len(failed)} thing(s) didn't come back cleanly: "
                         + "; ".join(failed[:3]) + ".")
            return held
        if failed:
            return ("Back to normal power, sir — though "
                    f"{len(failed)} thing(s) didn't come back cleanly: "
                    + "; ".join(failed[:3]) + ".")
        return f"Back to normal power, sir. I'm on {_current_brain()} again."


# ── Watcher ───────────────────────────────────────────────────────────────
def _tick() -> str:
    """One poll. Extracted so tests drive the real logic without a thread."""
    cfg = {
        "dwell": _cfg_float("GAME_MODE_ENTER_DWELL_SECONDS", 20.0),
        "grace": _cfg_float("GAME_MODE_EXIT_GRACE_SECONDS", 45.0),
        "max_seconds": _cfg_float("GAME_MODE_MAX_SECONDS", 12 * 3600.0),
        "require_foreground": bool(_cfg("GAME_MODE_REQUIRE_FOREGROUND", True)),
    }
    pid, exe = _find_game_pid()
    fg = _foreground_pid()
    with _lock:
        action, reason = decide(time.time(), pid, fg, _st, cfg)
        # Clear a stale inhibit once that game is gone, so the NEXT session of
        # the same game is not silently un-protected forever.
        if _st.inhibit_pid is not None and pid != _st.inhibit_pid:
            _st.inhibit_pid = None
        # Pay any DEFERRED brain restore the moment no allowlisted game is left
        # running. decide() can only ever return "exit" while st.active is True,
        # so once we have exited with the brain still downshifted this is the
        # ONLY path back to his big model short of restarting JARVIS by hand —
        # which is the exact thing he must never have to do.
        owed = []
        if _st.pending_brain is not None and not _st.active and pid is None:
            _flush_pending_brain(owed)
            _st.notes = list(_st.notes) + owed
    for n in owed:
        print(f"{_LOG} {n}")
    if owed:
        _hud(game_mode_brain=_current_brain())
    if action == "enter":
        print(f"{_LOG} entering — {reason} (pid {pid}, {exe})")
        _enter(pid, exe)
    elif action == "exit":
        print(f"{_LOG} exiting — {reason}")
        _exit(reason)
    return action


def _poll_loop() -> None:  # pragma: no cover - non-terminating daemon; every decision it makes is in decide()/_tick(), both unit-tested directly
    last_warm = 0.0
    while not _stop.is_set():
        try:
            _tick()
            with _lock:
                active, tag = _st.active, _current_brain()
            warm_every = _cfg_float("GAME_MODE_KEEP_WARM_SECONDS", 600.0)
            now = time.monotonic()
            if active and warm_every > 0 and (now - last_warm) >= warm_every:
                last_warm = now
                print(f"{_LOG} keep-warm {tag}: {_keep_warm(tag)}")
        except Exception as e:
            print(f"{_LOG} poll error (continuing): {e}")
        _stop.wait(_cfg_float("GAME_MODE_POLL_SECONDS", 5.0))


def start_watcher() -> bool:
    t = _thread[0]
    if t is not None and t.is_alive():
        return False
    _stop.clear()
    t = threading.Thread(target=_poll_loop, name="game-mode-watch", daemon=True)
    _thread[0] = t
    t.start()
    return True


def stop_watcher() -> None:
    _stop.set()
    _thread[0] = None


# ── Actions ───────────────────────────────────────────────────────────────
def game_mode_status(_: str = "") -> str:
    """Report the mode from a FRESH measurement, not from a remembered verdict.

    `_st.result` is the score of one 8-second window taken right after entry.
    Repeating it for the rest of the session is how a mode that was working
    kept announcing its own failure mid-match — and a feature that announces
    its own failure is a feature he switches off. So we re-sample and
    re-`_classify()` through the same code path `_verify` used, then write the
    corrected verdict back so the HUD tile and the next caller agree with what
    was just said."""
    with _lock:
        active = _st.active
        exe = _st.game_exe
        prev = _st.result
        before = _st.before
        entered = _st.entered_at
        manual = _st.manual
    brain = _current_brain()
    if not active:
        pid, running = _find_game_pid()
        watching = "on" if (_thread[0] is not None and _thread[0].is_alive()) else "off"
        if running:
            return (f"Game mode is not engaged, sir — though {running} is "
                    f"running. The watcher is {watching} and I'm on {brain}.")
        return (f"Game mode is not engaged, sir. The watcher is {watching} and "
                f"I'm on the {brain} brain.")
    floor = _cfg_int("GAME_MODE_MIN_VRAM_DELTA_MB", 3000)
    now = _measure()
    result, _msg, freed = _classify(before or {}, now, floor,
                                    brain, _on_game_brain())
    with _lock:
        if _st.active:                       # it can end while we measure
            _st.after = now
            _st.result = result
    if result != prev:
        _hud(game_mode=True, game_mode_app=exe, game_mode_result=result,
             game_mode_vram_freed_mb=freed, game_mode_brain=brain)
    mins = (time.time() - entered) / 60.0 if entered else 0.0
    how = "because you asked" if manual else f"for {exe}"
    if result == "ok" and freed is not None:
        return (f"Game mode is engaged {how}, sir — {mins:.0f} minutes in, "
                f"{freed} MB of VRAM freed by measurement, running on the "
                f"{brain} brain.")
    if result == "failed":
        return (f"Game mode is engaged {how}, sir, but it FAILED to free what "
                f"it should have — only {freed} MB of VRAM. I'm on {brain}.")
    if result == "armed":
        return (f"Game mode is engaged {how}, sir — {mins:.0f} minutes in, on "
                f"the {brain} brain. There was nothing resident to reclaim "
                f"when I engaged, so I have no VRAM saving to claim; what this "
                f"buys you is the load it prevents, not memory already in use.")
    return (f"Game mode is engaged {how}, sir, on the {brain} brain — but I "
            f"have no verified memory saving to report.")


def game_mode_on(_: str = "") -> str:
    """Engage now, and ALSO start the watcher so the mode ends by itself.

    This is the path that works TONIGHT with no restart. core.config's
    _apply_user_settings() runs at IMPORT time, so writing GAME_MODE_ENABLED
    into data/user_settings.json does not reach an already-running JARVIS —
    register() would still see False and never start the watcher. Firing this
    action does both jobs, so the owner never ends up in a mode he has to leave
    by hand (which is the thing he must not have to do).

    A game already running OWNS the mode (manual=False) so it auto-exits when
    that process dies. Only a truly manual engage — no allowlisted game in
    sight — becomes a manual hold, bounded by the deadman.

    THIS RUNS ON THE MAIN VOICE THREAD, so it returns as soon as the levers are
    applied and hands the settle-and-verify wait to a worker
    (defer_verify=True). Blocking here for the fixed 8 s settle plus a second
    GPU sample is JARVIS going deaf and silent mid-match — at the exact moment
    he asked for help. The sentence he gets back therefore claims NOTHING about
    memory; the measured verdict follows as an announcement once it has
    actually been measured, and game_mode_status can be asked for it any time."""
    pid, exe = _find_game_pid()
    with _lock:
        _st.inhibit_pid = None          # an explicit "on" clears a prior "off"
    out = _enter(pid, exe or "manual", manual=(pid is None), defer_verify=True)
    try:
        if start_watcher():
            out += " I'll also watch for the game closing and put everything back."
    except Exception as e:
        out += f" (I could not start the auto-exit watcher: {e})"
    return out


def game_mode_off(_: str = "") -> str:
    """Always available, handled deterministically so it works even if the
    brain is wedged. Also INHIBITS re-entry for the running game, so the
    watcher can never undo what the owner just asked for.

    THE BRAIN IS THE ONE THING THIS CANNOT GIVE HIM BACK IMMEDIATELY while a
    game is still up — repointing at the big model mid-match means his very
    next utterance cold-loads ~15 GB into a card that is already 95 % full,
    which is the crash this whole file exists to prevent. So we hold that one
    change, SAY we are holding it (never "I'm on <big tag> again" over a brain
    that is still small), and leave a debt on _st.pending_brain that _tick()
    pays the moment the game closes. Deferring is only defensible if the
    deferral is actually honoured, so this also makes sure a ticker exists to
    honour it."""
    out = _exit("owner asked for normal power", inhibit=True)
    with _lock:
        owed = _st.pending_brain is not None
    if not owed:
        return out
    if _find_game_pid()[0] is None:
        # No game left to protect, so there is nothing to defer FOR: pay the
        # debt on the spot rather than waiting on a tick that may not be
        # scheduled. This is also his hand-operated way out of a mode that has
        # already exited with the brain still downshifted.
        notes: list = []
        with _lock:
            _flush_pending_brain(notes)
        for n in notes:
            print(f"{_LOG}   {n}")
        _hud(game_mode_brain=_current_brain())
        return f"Back to normal power, sir. I'm on {_current_brain()} again."
    # We just promised to put the big brain back when the game closes, and only
    # _tick() can keep that promise — so make sure a ticker exists. On the
    # manual path the watcher may never have been started, and a promise with
    # no mechanism behind it is the exact failure this file is written against.
    # Re-entry for the running game is already inhibited, so arming the watcher
    # cannot undo what he just asked for.
    try:
        start_watcher()
    except Exception as e:
        out += (f" (I could not arm the watcher that puts the big brain back: "
                f"{e} — say 'full power' again once the game is closed.)")
    return out


def game_mode_learn_this(_: str = "") -> str:
    """Enrol the CURRENTLY FOCUSED app. In-memory only — this module writes no
    files. The exact basename is spoken back so the orchestrator (or the owner)
    can persist it into GAME_MODE_PROCESS_HINTS."""
    exe = _foreground_exe_basename()
    if not exe:
        return ("I couldn't read the focused application, sir — try again with "
                "the game in front.")
    hints = list(_game_hints())
    if exe in hints:
        return f"I already treat {exe} as a game, sir."
    hints.append(exe)
    bc = _bc()
    for target in (bc,):
        if target is not None:
            try:
                setattr(target, "GAME_MODE_PROCESS_HINTS", hints)
            except Exception:
                pass
    try:
        import core.config as _c
        _c.GAME_MODE_PROCESS_HINTS = hints
    except Exception:
        pass
    return (f"I'll treat {exe} as a game from now on, sir — for this session. "
            f"Add it to GAME_MODE_PROCESS_HINTS to make it stick.")


def register(actions):
    actions["game_mode_status"] = game_mode_status
    actions["game_mode_on"] = game_mode_on
    actions["game_mode_off"] = game_mode_off
    actions["game_mode_learn_this"] = game_mode_learn_this
    # Aliases the deterministic dispatcher can match without the LLM — the
    # override has to work when the brain is wedged.
    actions["low_power_mode"] = game_mode_on
    actions["normal_power"] = game_mode_off
    actions["full_power"] = game_mode_off

    enabled = bool(_cfg("GAME_MODE_ENABLED", False))
    hints = _game_hints()
    # Never autostart under staging/test — it polls processes and mutates the
    # live brain pointer. Same gate as skills/audio_autoswitch.
    if enabled and hints and not os.getenv("JARVIS_STAGING"):
        try:
            if start_watcher():
                print(f"{_LOG} watching for {len(hints)} game(s) every "
                      f"{_cfg_float('GAME_MODE_POLL_SECONDS', 5.0):.0f}s "
                      f"— brain downshifts to "
                      f"{_cfg('GAME_MODE_BRAIN', '?')} while one is running")
        except Exception as e:
            print(f"{_LOG} watcher start failed: {e}")
    else:
        print(f"{_LOG} ready (watcher OFF — set GAME_MODE_ENABLED=true). "
              f"Actions: game_mode_status, game_mode_on, game_mode_off, "
              f"game_mode_learn_this.")
