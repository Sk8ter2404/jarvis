"""JARVIS runtime state slots.

Phase 2 of the modularisation refactor (2026-05-29). All mutable
runtime state that used to live as top-level globals in
bobert_companion.py now lives here. The parent module does
`from core.state import *` so existing references like
`bobert_companion._sleep_mode[0]` (used by skills/morning_chain.py,
skills/anticipation_*, skills/wellness.py, skills/screen_watch.py)
keep resolving — wildcard imports rebind every name in `__all__`
into bobert_companion's namespace.

Contract — read this before touching the file:

  • Every slot is a SINGLE-element list (or set / dict / threading
    primitive). Reads use ``_slot[0]``; writes use ``_slot[0] = X``.
    Under CPython's GIL, list __setitem__ on a builtin list is
    atomic, so reader threads observe the new value without locks.

  • The list IDENTITY never changes — only its single element does.
    DO NOT REFACTOR ANY OF THESE TO PLAIN VARIABLES. The change would
    type-check, run cleanly under single-threaded tests, and silently
    break cross-module mutation in production. See the long comment
    block at the original _last_wake_date definition (now moved into
    this file's body) for the full reasoning.

  • Names are underscore-prefixed by long-standing convention. Python's
    wildcard import normally skips _-prefixed names, so this module
    explicitly defines ``__all__`` to opt them back in.

  • Seeds that depend on config (AUDIO_PROCESSING_ENABLED for
    _audio_master_enabled, VAD_DEBUG for _debug_mode) come from
    core/config.py — NOT from bobert_companion.py — so there's no
    circular import.
"""
from __future__ import annotations

import threading
from typing import Optional

from core.config import AUDIO_PROCESSING_ENABLED, VAD_DEBUG


# ─── Sleep / standby / ambient toggles ─────────────────────────────────
# True means JARVIS is dormant and only listening for a wake phrase.
# Toggled by SLEEP_PHRASES / WAKE_PHRASES in the main loop, and forced
# True by _act_start_overnight_upgrade.
_sleep_mode: list[bool] = [False]

# True when sleep_mode was entered specifically as 'standby' (work-mode
# or ambient-music trigger) rather than the normal sleep phrases.
# Affects HUD label ('Standby' vs 'Idle') and resume message.
_standby_mode: list[bool] = [False]

# Voice-triggered shutdown prompt arming. When the user says any of the
# SHUTDOWN_TRIGGER_PHRASES, JARVIS asks "Would you like to start the
# overnight protocol first, sir? Yes or no." and arms this dict for
# SHUTDOWN_PROMPT_TIMEOUT_S seconds. The next utterance is routed
# through _handle_shutdown_prompt() BEFORE the LLM sees it. After the
# timeout (or any unrelated utterance) the flag is cleared and normal
# routing resumes.
_shutdown_prompt_pending: dict = {"armed": False, "expires_at": 0.0}


# ─── Ambient-music detector state ──────────────────────────────────────
# Timestamp (epoch seconds) of the last JARVIS-initiated music/media
# action. Used to suppress the ambient-music standby trigger when JARVIS
# itself is the source of the audio bleeding into the mic.
_jarvis_played_music_at: list[float] = [0.0]

# Rolling counter of music-marker transcriptions (Whisper output like
# '[Music]') that weren't attributable to JARVIS-initiated playback.
# Reset whenever a music-marker hasn't been seen in MUSIC_HITS_WINDOW
# seconds.
_ambient_music_hits:     list[int]   = [0]
_ambient_music_last_hit: list[float] = [0.0]


# ─── Overnight upgrade trigger ─────────────────────────────────────────
# Set this event to make the overnight upgrade thread skip the idle
# wait and run a cycle immediately on the next check.
_overnight_run_now = threading.Event()


# ─── Wake-word history ─────────────────────────────────────────────────
# Used by context_aware_greeting() to vary the response (first-of-day
# → "Good morning, sir.", 3rd in 10min during 1–5am → "Still up, sir?").
# Trimmed to the last 10 minutes on each wake.
_wake_history: list[float] = []

# ISO date string of the most recent wake (or None). Wrapped in a
# 1-element list because skills/morning_chain.py runs a background
# watcher thread that polls bc._last_wake_date[0] every
# WATCH_POLL_SECONDS to detect the day's first wake event. The list
# wrapper enforces a mutable-reference-sharing contract between this
# module (writer) and that thread (reader):
#
#   • Writes use in-place index assignment: _last_wake_date[0] = today.
#     Under CPython's GIL, list __setitem__ on a builtin list is
#     atomic, so the watcher thread observes the new value without
#     locks.
#   • The list *identity* never changes — only its single element does.
#     This matches the same idiom used by _sleep_mode / _standby_mode /
#     _wake_history above, and removes the need for a `global`
#     declaration at every write site (writes happen inside functions
#     like context_aware_greeting() and the tray force_wake handler).
#
# DO NOT REFACTOR THIS TO A PLAIN STRING/Optional[str] VARIABLE. The
# change would type-check, run cleanly under single-threaded tests,
# and silently break the morning chain in production:
#   - Every `_last_wake_date[0] = X` write site would have to become
#     `_last_wake_date = X` inside a function, which without `global`
#     creates a function-local shadow that the watcher never sees.
#   - Even with `global` everywhere, the watcher's
#     `bc._last_wake_date[0]` would need to become
#     `bc._last_wake_date`, and the change would have to happen
#     across every consumer skill in lockstep with this file.
_last_wake_date: list[Optional[str]] = [None]


# ─── Audio-processor runtime toggles ───────────────────────────────────
# Flipped at runtime by the tray's Audio Controls submenu via
# _dispatch_tray_command. Stored in single-element lists so the tray
# handlers and _process_capture_chunk read the same value without
# threading "global" through every callsite. Master flag seeds from
# the core/config value so flipping AUDIO_PROCESSING_ENABLED in
# core/config.py is the canonical knob; tray toggles are runtime
# overrides only.
_audio_master_enabled = [AUDIO_PROCESSING_ENABLED]
_audio_aec_enabled    = [True]
_audio_ns_enabled     = [True]
_audio_agc_enabled    = [True]


# ─── Tray runtime toggles ──────────────────────────────────────────────
# Each toggle handler in _dispatch_tray_command flips its slot and
# mirrors to hud_state.json so the tray's checked=lambda reads the new
# state on the next right-click. Same single-element-list pattern as
# the audio flags above. _debug_mode seeds from VAD_DEBUG so the
# tray's Debug Mode checkmark matches the file's default; the four
# VAD_DEBUG print sites read _debug_mode[0] at runtime so a toggle
# takes effect immediately.
_tts_muted            = [False]
_ambient_mode_active  = [False]
_daemons_paused       = [False]
_debug_mode           = [VAD_DEBUG]


# ─── Wildcard re-export list ───────────────────────────────────────────
# Without this, `from core.state import *` would skip every _-prefixed
# name (Python's default behaviour for underscore names). The whole
# point of this module is that bobert_companion.py rebinds these slots
# via wildcard, so every name MUST be listed here.
__all__ = [
    "_sleep_mode",
    "_standby_mode",
    "_shutdown_prompt_pending",
    "_jarvis_played_music_at",
    "_ambient_music_hits",
    "_ambient_music_last_hit",
    "_overnight_run_now",
    "_wake_history",
    "_last_wake_date",
    "_audio_master_enabled",
    "_audio_aec_enabled",
    "_audio_ns_enabled",
    "_audio_agc_enabled",
    "_tts_muted",
    "_ambient_mode_active",
    "_daemons_paused",
    "_debug_mode",
]
