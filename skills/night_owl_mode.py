"""
Night-owl mode for JARVIS.

Auto-engages at 23:00 local time and runs until 06:00 (configurable). While
active:

  • TTS preset is dimmed — gain × 0.85 and rate further slowed ~5pp on top
    of whatever the existing voice_emotion_router / intent_tag selection
    picks. Same words, just quieter and softer past midnight.
  • Holographic overlay is dimmed to ~40% opacity via a `night_owl_dim`
    field in hud_state.json — jarvis_holo.py reads it each tick and applies
    `-alpha 0.4` on the root window.
  • Non-critical proactive nudges are suppressed by monkey-patching each
    skill's `_enqueue_speech` with a sink that drops the message and logs
    it. Suppressed skills: weather_briefing, news_briefing, banter,
    wellness, anticipation_engine, screen_watch. CRITICAL announcers
    (bambu_monitor failure, timer reminders, teams_screener VIP calls,
    bambu_print_announcer failure/runout/AMS) stay loud.
  • A brief addendum is appended to bobert_companion._system_prompt so the
    LLM knows it's late and should keep replies short and quiet.

Auto-disengages at 06:00 OR when the user says "good morning" (the LLM
emits [ACTION: good_morning] in response to a morning greeting while the
mode is engaged — the prompt addendum tells it so).

Actions registered:
  night_owl_on / night_owl_mode  — engage manually.
  night_owl_off / end_night_owl  — disengage manually.
  good_morning                   — disengage with a morning greeting.
  night_owl_status               — report state.

Module-level helper for other skills:
  is_night_owl_active() -> bool
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure the project root is importable so `core.atomic_io` resolves whether
# this module is loaded as `skills.night_owl_mode` or run directly.
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

# ─── Config ──────────────────────────────────────────────────────────────
NIGHT_OWL_START_HOUR   = 23     # auto-engage at/after 23:00 local
NIGHT_OWL_END_HOUR     = 6      # auto-disengage at/after 06:00 local
NIGHT_OWL_OVERLAY_DIM  = 0.4    # 40% opacity per spec
NIGHT_OWL_GAIN_SCALE   = 0.85   # -15% gain per spec
NIGHT_OWL_RATE_DELTA_PP = 5     # subtract 5pp more from whatever rate the
                                # base preset picked (so "+0%" becomes "-5%",
                                # "-12%" becomes "-17%", etc.)
WATCH_INTERVAL_SECONDS = 60.0

# Skills whose _enqueue_speech is suppressed while night-owl mode is on.
# Critical announcers stay live — Bambu failures, timers, VIP calls.
SUPPRESSED_SKILLS = (
    "weather_briefing",
    "news_briefing",
    "banter",
    "wellness",
    "anticipation_engine",
    "screen_watch",
)

NIGHT_OWL_PROMPT_ADDENDUM = (
    "\n\n[Night-owl mode]\n"
    "It is past 23:00. Sir is winding down. Keep replies to ONE short "
    "sentence whenever possible, soft register, no exuberance. If sir "
    "greets you with 'good morning' or similar, emit [ACTION: good_morning] "
    "to release night-owl mode."
)

_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")

# ─── State ───────────────────────────────────────────────────────────────
_mode_lock         = threading.Lock()
_night_owl_active  = [False]
_started_at        = [0.0]
_trigger           = [""]          # "auto" | "manual" | "phrase"

_saved_resolve_preset: list = [None]
_saved_enqueues: dict[str, object] = {}
_saved_system_prompt: list = [None]
_watcher_thread    = [None]
_speech_lock       = threading.Lock()


# ─── Tiny helpers ────────────────────────────────────────────────────────

def _enqueue_speech(message: str, volume_scale: float = 1.0) -> None:
    """Send our own announcements through the main loop's queue. Always
    goes through — we never suppress our own mode-change messages.

    Routes through bobert_companion.proactive_announce() so this skill and
    every other pending_speech.json co-writer (bambu_monitor, screen_watch,
    status_panel, …) share one write path and don't race each other. Falls
    back to a local atomic write only when the parent module isn't loaded
    yet (import-time registration) or the announcer call fails. If a non-
    default volume_scale is requested the canonical helper can't carry it,
    so we drop straight to the atomic-write path to preserve that field."""
    # proactive_announce now carries volume_scale and serialises every writer
    # under one shared lock, so route ALL announcements through it (previously a
    # non-default volume_scale had to bypass to a local write that raced the
    # canonical path). Fall back to the local atomic write only when the parent
    # module isn't importable yet.
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="night_owl", volume_scale=volume_scale)
            return
    except Exception:
        # Fall through to local atomic write — never let a broken parent
        # import silence a mode-change announcement.
        pass

    with _speech_lock:
        data = []
        if os.path.exists(_SPEECH_QUEUE):
            try:
                with open(_SPEECH_QUEUE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        item: dict = {"ts": time.time(), "message": message}
        if abs(volume_scale - 1.0) > 1e-3:
            item["volume_scale"] = volume_scale
        data.append(item)
        try:
            _atomic_write_json(_SPEECH_QUEUE, data)
        except Exception as e:
            print(f"  [night_owl] speech-queue write failed ({e}); message: {message}")


def _in_night_window(now: datetime | None = None) -> bool:
    """True if the current local hour falls in the night-owl window
    (23:00–06:00, wrap-around safe)."""
    h = (now or datetime.now()).hour
    if NIGHT_OWL_START_HOUR <= NIGHT_OWL_END_HOUR:
        return NIGHT_OWL_START_HOUR <= h < NIGHT_OWL_END_HOUR
    # Wrap-around case (the usual one): 23–24 OR 0–6.
    return h >= NIGHT_OWL_START_HOUR or h < NIGHT_OWL_END_HOUR


def is_night_owl_active() -> bool:
    return bool(_night_owl_active[0])


# ─── TTS preset modifier ─────────────────────────────────────────────────

def _adjust_rate_string(rate: str, delta_pp: int) -> str:
    """Take a rate string like '+6%' or '-10%' and slow it further by
    delta_pp percentage points. Returns a normalised string with explicit
    sign — edge-tts insists on the leading +/-."""
    if not isinstance(rate, str):
        rate = "+0%"
    s = rate.strip()
    if not s.endswith("%"):
        return f"{-delta_pp:+d}%"
    try:
        n = int(s[:-1])
    except ValueError:
        return f"{-delta_pp:+d}%"
    return f"{n - delta_pp:+d}%"


def _install_tts_modifier() -> None:
    """Wrap bobert_companion._resolve_tts_preset so every synthesise() call
    returns a preset whose gain is scaled down and rate is slowed further.
    Reversible: the original is cached in _saved_resolve_preset[0]."""
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return
    orig = getattr(bc, "_resolve_tts_preset", None)
    if orig is None:
        return
    if _saved_resolve_preset[0] is not None:
        # Already installed — don't double-wrap.
        return
    _saved_resolve_preset[0] = orig

    def _wrapped(text: str, user_tone):
        name, preset = orig(text, user_tone)
        # Build a fresh dict so we don't mutate the canonical preset.
        try:
            new_preset = dict(preset)
            new_preset["rate"] = _adjust_rate_string(
                str(preset.get("rate", "+0%")), NIGHT_OWL_RATE_DELTA_PP
            )
            new_preset["gain"] = float(preset.get("gain", 1.0)) * NIGHT_OWL_GAIN_SCALE
        except Exception:
            # Anything weird → fall back to the original so we don't mute TTS.
            return name, preset
        return f"{name}_nightowl", new_preset

    try:
        bc._resolve_tts_preset = _wrapped
    except Exception as e:
        print(f"  [night_owl] could not install TTS modifier: {e}")
        _saved_resolve_preset[0] = None


def _restore_tts_modifier() -> None:
    if _saved_resolve_preset[0] is None:
        return
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        _saved_resolve_preset[0] = None
        return
    try:
        bc._resolve_tts_preset = _saved_resolve_preset[0]
    except Exception as e:
        print(f"  [night_owl] could not restore TTS modifier: {e}")
    finally:
        _saved_resolve_preset[0] = None


# ─── Holographic overlay dim ─────────────────────────────────────────────

def _set_overlay_dim(active: bool) -> None:
    """Publish a `night_owl_dim` field to hud_state.json so jarvis_holo.py's
    tick() can apply `-alpha 0.4` on the overlay window. Goes through
    bobert_companion._write_hud_state so we share the canonical state lock
    and don't race with other publishers."""
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return
    writer = getattr(bc, "_write_hud_state", None)
    if not callable(writer):
        return
    try:
        writer(night_owl_dim=(NIGHT_OWL_OVERLAY_DIM if active else 0.0))
    except Exception as e:
        print(f"  [night_owl] could not publish overlay dim: {e}")


# ─── Proactive-nudge suppression ─────────────────────────────────────────

def _install_nudge_suppressors() -> None:
    # If the target's _enqueue_speech is already one of these sinks (an
    # overlapping mode — e.g. dnd_focus_mode — installed it first), don't save
    # it as the "original" and don't double-wrap: leave the existing sink in
    # place and record None so restore knows this skill didn't own the patch.
    # Without this, overlapping focus + night-owl would save a sink as the
    # original and write it back on exit, permanently stranding the target.
    for skill in SUPPRESSED_SKILLS:
        mod = sys.modules.get(f"skill_{skill}")
        if mod is None:
            continue
        original = getattr(mod, "_enqueue_speech", None)
        if not callable(original):
            continue
        if skill in _saved_enqueues:
            continue
        if getattr(original, "_is_nudge_sink", False):
            _saved_enqueues[skill] = None
            continue
        _saved_enqueues[skill] = original

        def _sink(message: str, _skill=skill, *args, **kwargs):
            print(f"  [night_owl] suppressed {_skill} nudge: {str(message)[:120]}")
        _sink._is_nudge_sink = True
        try:
            mod._enqueue_speech = _sink
        except Exception as e:
            print(f"  [night_owl] could not wrap {skill}._enqueue_speech: {e}")
            _saved_enqueues.pop(skill, None)


def _restore_nudge_suppressors() -> None:
    for skill, original in list(_saved_enqueues.items()):
        mod = sys.modules.get(f"skill_{skill}")
        # Only restore a real saved original — never write back None (we
        # didn't own the patch) or a sink (would re-strand the target).
        if (mod is not None
                and original is not None
                and not getattr(original, "_is_nudge_sink", False)):
            try:
                mod._enqueue_speech = original
            except Exception as e:
                print(f"  [night_owl] could not restore {skill}._enqueue_speech: {e}")
        _saved_enqueues.pop(skill, None)


# ─── System-prompt addendum ──────────────────────────────────────────────

def _apply_prompt_addendum() -> None:
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return
    try:
        if _saved_system_prompt[0] is None:
            _saved_system_prompt[0] = bc._system_prompt
        bc._system_prompt = _saved_system_prompt[0] + NIGHT_OWL_PROMPT_ADDENDUM
    except Exception as e:
        print(f"  [night_owl] prompt addendum failed: {e}")


def _restore_prompt_addendum() -> None:
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return
    if _saved_system_prompt[0] is None:
        return
    try:
        bc._system_prompt = _saved_system_prompt[0]
    finally:
        _saved_system_prompt[0] = None


# ─── Enter / exit ────────────────────────────────────────────────────────

def _enter_night_owl(trigger: str = "manual", *, announce: bool = True) -> str:
    """Engage night-owl mode. Idempotent — repeat calls just re-stamp the
    trigger and return a status line without re-announcing."""
    with _mode_lock:
        if _night_owl_active[0]:
            return f"Night-owl mode already engaged, sir."
        _night_owl_active[0] = True
        _started_at[0]       = time.time()
        _trigger[0]          = trigger

    # Install side effects outside the lock — none of them block long, but
    # status queries shouldn't have to wait on them.
    _install_tts_modifier()
    _install_nudge_suppressors()
    _apply_prompt_addendum()
    _set_overlay_dim(True)

    if announce:
        if trigger == "auto":
            msg = (
                "It's past 11, sir. Night-owl mode engaged — I'll keep my "
                "voice down and hold non-essentials until morning."
            )
        else:
            msg = (
                "Night-owl mode engaged, sir. I'll keep the volume low and "
                "hold non-essentials until you say good morning."
            )
        _enqueue_speech(msg)

    print(f"  [night_owl] engaged via {trigger}")
    return "Night-owl mode engaged, sir."


def _exit_night_owl(trigger: str = "manual", *, announce: bool = True) -> str:
    """Disengage night-owl mode. Idempotent."""
    with _mode_lock:
        if not _night_owl_active[0]:
            return "Night-owl mode was not active, sir."
        _night_owl_active[0] = False
        prev_trigger = _trigger[0]
        _trigger[0] = ""
        _started_at[0] = 0.0

    _restore_tts_modifier()
    _restore_nudge_suppressors()
    _restore_prompt_addendum()
    _set_overlay_dim(False)

    if announce:
        if trigger in ("auto_morning", "phrase_good_morning"):
            msg = "Good morning, sir. Night-owl mode released."
        else:
            msg = "Night-owl mode disengaged, sir."
        _enqueue_speech(msg)

    print(f"  [night_owl] disengaged ({trigger}, was triggered by {prev_trigger})")
    return msg if announce else "Night-owl mode disengaged, sir."


# ─── Watcher thread ──────────────────────────────────────────────────────

def _watch_loop():
    """Poll every WATCH_INTERVAL_SECONDS. Auto-engage when the local clock
    enters the night window; auto-disengage when it leaves OR when the
    main loop's _last_wake_date flips (i.e. a fresh wake-word in the
    morning has fired). Crash-resilient — any exception logs and continues."""
    # Small startup delay so the skill loader finishes registering everyone.
    time.sleep(8.0)
    while True:
        try:
            in_window = _in_night_window()
            active = is_night_owl_active()
            if in_window and not active:
                _enter_night_owl(trigger="auto")
            elif not in_window and active and _trigger[0] == "auto":
                # Only auto-disengage automatically-engaged sessions —
                # manual engagements stay until the user says so or the
                # window naturally lapses.
                _exit_night_owl(trigger="auto_morning")
            elif not in_window and active and _trigger[0] != "auto":
                # Manual engagement that the clock has rolled past 06:00:
                # still auto-disengage so the user isn't stuck in dimmed
                # mode all day from a forgotten previous-night command.
                _exit_night_owl(trigger="auto_morning")
            time.sleep(WATCH_INTERVAL_SECONDS)
        except Exception:
            logging.exception("[night_owl] watcher tick failed")
            time.sleep(WATCH_INTERVAL_SECONDS)


# ─── Action handlers ─────────────────────────────────────────────────────

def register(actions):
    def night_owl_on(_: str = "") -> str:
        return _enter_night_owl(trigger="manual")

    def night_owl_off(_: str = "") -> str:
        return _exit_night_owl(trigger="manual")

    def good_morning(_: str = "") -> str:
        # If night-owl mode is active, release it with the morning greeting.
        # Otherwise just speak a normal morning greeting — keeps the action
        # useful as a generic verbal hook.
        if is_night_owl_active():
            return _exit_night_owl(trigger="phrase_good_morning")
        return "Good morning, sir."

    def night_owl_status(_: str = "") -> str:
        with _mode_lock:
            active = _night_owl_active[0]
            started = _started_at[0]
            trig = _trigger[0]
        if not active:
            return "Night-owl mode is not currently engaged, sir."
        elapsed = max(0, int(time.time() - started))
        h, rem = divmod(elapsed, 3600)
        m = rem // 60
        if h > 0:
            since = f"{h}h {m}m"
        else:
            since = f"{m} minutes"
        label = {"auto": "automatically", "manual": "by voice",
                 "phrase": "via greeting"}.get(trig, trig or "")
        return f"Night-owl mode is engaged ({label}), sir — running for about {since}."

    actions["night_owl_on"]     = night_owl_on
    actions["night_owl_mode"]   = night_owl_on   # alias
    actions["enable_night_owl"] = night_owl_on   # alias
    actions["night_owl_off"]    = night_owl_off
    actions["end_night_owl"]    = night_owl_off  # alias
    actions["disable_night_owl"] = night_owl_off # alias
    actions["good_morning"]     = good_morning
    actions["night_owl_status"] = night_owl_status

    t = threading.Thread(target=_watch_loop, daemon=True, name="night_owl_watcher")
    t.start()
    _watcher_thread[0] = t
    print("  [night_owl] night_owl_mode ready — actions: night_owl_on, "
          "night_owl_off, good_morning, night_owl_status (window "
          f"{NIGHT_OWL_START_HOUR:02d}:00–{NIGHT_OWL_END_HOUR:02d}:00)")
