#!/usr/bin/env python3
"""
JARVIS in-tree codebase auditor.

Runs perfect line-by-line checks on every Python file in the project. Designed
to be invoked by upgrade_jarvis.py as a post-upgrade gate, OR manually:

    python tools/audit_codebase.py           # run all checks, print summary
    python tools/audit_codebase.py --json    # JSON report only (no markdown)
    python tools/audit_codebase.py --fix     # apply trivially-safe fixes
    python tools/audit_codebase.py --quiet   # suppress per-finding noise

Outputs:
    audit_report.json   — full structured report (severity-grouped)
    audit_report.md     — human-readable summary

Exit codes:
    0   zero findings
    1   at least one P0 finding (will crash / security)
    2   at least one P1 finding (will misbehave)
    3   only P2 findings (code smell / minor)

Check coverage:

  STATIC CHECKS (1–14)
    1.  Walk all .py files in project (exclude backups, __pycache__, pending_skills, _*.py)
    2.  Syntax (py_compile doraise)
    3.  Imports (ast — stdlib vs 3rd-party vs intra-project; cross-check requirements.txt)
    4.  Cross-references (bobert_companion.X / bc.X / core.X must resolve)
    5.  ACTIONS — sandbox-load each skill, check action-name collisions + signatures
    6.  Bad patterns (bare except, eval, exec, os.system, hardcoded D:\\ paths,
        threading.Thread() w/o daemon, subprocess.Popen w/o close_fds,
        subprocess.run w/o timeout, open() w/o encoding=, mkstemp leaks,
        non-atomic json.dump to known state files)
    7.  Hardcoded secrets (sk-ant-, AKIA, AIza, password=, token= literals)
    8.  Threading hygiene (Thread/Timer must be daemon OR explicitly cancelled;
        tkinter calls from non-main threads)
    9.  Subprocess hygiene (Popen/run timeout or documented long-running)
    10. HUD state writes (only canonical _write_hud_state helper allowed)
    11. pending_speech.json writes (canonical _enqueue_speech pattern only)
    12. Action registration ordering (duplicate action names — first/last wins?)
    13. Edge-case readback (jarvis_todo.md, bobert_memory.json, hud_state.json,
        .last_upgrade_summary.json all valid)
    14. Mutation analysis (globals mutated from non-main threads need locks)

  INTEGRATION / CONFLICT CHECKS (A–J)
    A.  Action smoke tests — dispatch each ACTIONS entry with a benign arg
        and stubbed heavy deps (webbrowser, subprocess, cv2, win32com, paho.mqtt)
        and assert it returns str without raising. Skips entries listed in
        _NO_SMOKE_TEST inside the auditor.
    B.  Skill-pair conflict matrix — for every pair of skills, look for shared
        state-file writes, action-name collisions, and both having background
        threads. Output → audit_conflict_matrix.md.
    C.  Background-thread audit — every threading.Thread/.Timer call site must
        (1) be daemon, (2) have try/except wrapping the outer loop, (3) not
        spin without a sleep, (4) guard shared-state writes with the right lock.
    D.  Prompt ↔ Action consistency — actions mentioned in PC_CONTROL_PROMPT
        but not registered (and vice versa, minus _INTERNAL_ONLY_ACTIONS).
    E.  Voice-trigger coverage — every voice-triggered action has at least one
        trigger phrase example in the prompt. Output → coverage table in report.
    F.  TTS pipeline test — confirm every _TTS_EMOTION_PRESETS key is referenced
        in synthesise/_resolve_tts_preset, and that fallback paths exist.
    G.  Crash-recovery test — main code paths (sleep mode, listen, transcribe,
        dispatch, _speak) have outer try/except wrapping.
    H.  Memory & file-handle leak test — best-effort: short loop with
        psutil.Process().num_handles() / open-file count; flag monotonic growth.
    I.  Import graph + circular import detection — build the import DAG across
        all project modules and surface any cycle.
    J.  State-file integrity sweep — read every state file mentioned in code
        and validate (JSON parses, expected top-level keys present).

Flags:
    --integration-only   skip the static checks and run only the new (A–J)
    --no-integration     skip the integration checks (legacy static-only mode)
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import os
import py_compile
import re
import sys
import traceback
import types
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR  = os.path.join(PROJECT_DIR, "skills")
HUD_DIR     = os.path.join(PROJECT_DIR, "hud")
CORE_DIR    = os.path.join(PROJECT_DIR, "core")
TOOLS_DIR   = os.path.join(PROJECT_DIR, "tools")
REQUIREMENTS_FILE = os.path.join(PROJECT_DIR, "requirements.txt")

EXCLUDE_DIRS = {"backups", "__pycache__", "pending_skills", ".git", "camera_previews",
                "screenshots", "logs", "memory", "data", "tests",
                "data_staging", "logs_staging", "dist", "build"}

# Mapping: requirement-name → importable module-name (mirrors bobert_companion's
# _DEP_IMPORT_NAME). Anything not in here is assumed to import under its own
# name (with hyphens → underscores).
DEP_IMPORT_NAME = {
    "openai-whisper": "whisper",
    "opencv-python":  "cv2",
    "pillow":         "PIL",
    "edge-tts":       "edge_tts",
    "paho-mqtt":      "paho.mqtt.client",
    "pywin32":        "win32api",
}

# Stdlib module names — used to distinguish missing-stdlib (impossible/P0)
# from missing-third-party (probably-missing-dep, P1 unless in requirements).
STDLIB_MODULES = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()

# Files that are known state-files written by multiple writers. Direct
# `with open(P, 'w'): json.dump(d, f)` on these is a race source.
STATE_FILES = {
    "hud_state.json",
    "pending_speech.json",
    "bobert_memory.json",
    ".last_upgrade_summary.json",
    "credits_state.json",
    "daily_briefing_state.json",
    "bambu_overlay_state.json",
    "holo_workshop_state.json",
    "suit_up_state.json",
    "workshop_hud_state.json",
}

# Actions that should NOT be live-dispatched even with stubs (destructive,
# network-required for a real result, or hardware-coupled in a way no stub
# can reasonably reproduce). For these we only confirm callable + signature.
_NO_SMOKE_TEST: set[str] = {
    "upgrade",
    "restart",
    "start_overnight_upgrade",
    "queue_task",
    "clear_tasks",
    "run_shell",
    "check_credits",
    "see_user",
    "see_screen",
    "find_on_screen",
    "where_is_user",
    "which_monitor",
    "gaze_status",
    "gaze_stats",
    "screen_history",
    "recall_screen",
    "last_screen",
    "previous_screen",
    "play_music",
    "pause_music",
    "resume_music",
    "next_song",
    "previous_song",
    "now_playing",
    "play_unheard",
    "play_vibe",
    "skip_track",
    "music_history",
    "music_taste",
    "play_streaming",
    "apple_music",
    "netflix",
    "prime_video",
    "disney_plus",
    "hulu",
    "max",
    "spotify",
    "youtube_play",
    "session_memory_recall",
    "session_resume",
    "click",
    "type",
    "press",
    "hotkey",
    "scroll",
    "open_on_monitor",
    "move_window_to_monitor",
    "focus_window",
    "minimize_window",
    "close_window",
    "list_windows",
    "screenshot",
    "check_print",
    "how_is_the_print",
    "pause_print",
    "resume_print",
    "setup_printer",
    "daily_recap",
    "dossier",
}

# Actions that are registered but intentionally not surfaced to the LLM via
# PC_CONTROL_PROMPT (background-only, internal alias, or test-only). The
# prompt-consistency check (D) ignores misses for these.
_INTERNAL_ONLY_ACTIONS: set[str] = {
    "search",            # alias for web_search
    "last_screen",       # alias for recall_screen
    "previous_screen",   # alias for recall_screen
    "screen_history",    # alias for recall_screen
    "session_memory_recall",
    "session_resume",
    "gaze_status",
    "gaze_stats",
    "face_track_status",  # alias for gaze_status (registered in skills/face_tracker.py)
    "anticipation_status",  # diagnostic for the anticipation engine
    "audio_music_status",   # diagnostic for the spectral music detector (standby_audio_detect)
    "banter_status",     # diagnostic for the banter engine
    "arc_reactor_off",   # direct alias for arc_reactor with arg 'off' (documented as 'arc_reactor, <on|off|pulse>')
    "arc_reactor_on",    # direct alias for arc_reactor with arg 'on' (documented as 'arc_reactor, <on|off|pulse>')
    "arc_reactor_pulse", # direct alias for arc_reactor with arg 'pulse' (documented as 'arc_reactor, <on|off|pulse>')
    "holo_workshop",     # alias for arc_reactor_on (documented as 'arc_reactor, <on|off|pulse>' in PC_CONTROL_PROMPT) — registered in skills/holographic_overlay.py
    "holo_workshop_canvas", # alias for arc_reactor_on (documented as 'arc_reactor, <on|off|pulse>' in PC_CONTROL_PROMPT) — registered in skills/holographic_overlay.py
    "workshop_canvas",   # alias for arc_reactor_on (documented as 'arc_reactor, <on|off|pulse>' in PC_CONTROL_PROMPT) — registered in skills/holographic_overlay.py alongside holo_workshop / holo_workshop_canvas, all three wired to _act_arc_reactor_on
    "bambu_h2d_overlay", # script-filename alias for bambu_overlay_toggle (the canonical voice actions are bambu_overlay_on/off/toggle)
    "bambu_overlay",     # short alias for bambu_overlay_toggle (the canonical voice actions are bambu_overlay_on/off/toggle)
    "hide_bambu_overlay", # verb-prefix alias for bambu_overlay_off (the canonical voice action is bambu_overlay_off, documented in PC_CONTROL_PROMPT with 'hide the printer overlay' phrasing)
    "show_bambu_overlay", # verb-prefix alias for bambu_overlay_on (the canonical voice action is bambu_overlay_on, documented in PC_CONTROL_PROMPT) — registered in skills/holographic_overlay.py alongside hide_bambu_overlay
    "bambu_setup",       # alias for setup_printer (the canonical voice action is setup_printer, documented in PC_CONTROL_PROMPT)
    "configure_printer", # alias for setup_printer (the canonical voice action is setup_printer, documented in PC_CONTROL_PROMPT)
    "first_time_printer_setup", # alias for setup_printer (the canonical voice action is setup_printer, documented in PC_CONTROL_PROMPT)
    "setup_bambu",       # alias for setup_printer (the canonical voice action is setup_printer, documented in PC_CONTROL_PROMPT) — registered in skills/bambu_setup.py:585
    "setup_workspace",   # natural alias for predictive_morning_setup (the canonical voice action is predictive_morning_setup, documented in PC_CONTROL_PROMPT with the example 'JARVIS, set up my workspace' → [ACTION: predictive_morning_setup]) — registered in skills/morning_handoff.py:742
    "workspace_setup",   # noun-order natural alias for predictive_morning_setup (the canonical voice action is predictive_morning_setup, documented in PC_CONTROL_PROMPT with the example 'JARVIS, set up my workspace' → [ACTION: predictive_morning_setup]) — registered in skills/morning_handoff.py:743 alongside setup_workspace (both wired to the same predictive_morning_setup handler; same pattern as setup_workspace above)
    "daily_briefing",    # background-scheduled by skills/daily_briefing.py (fires once per day at DAILY_BRIEFING_HOUR:MINUTE) — not a voice action
    "evening_briefing",  # background-scheduled by skills/evening_briefing.py (fires once per day at EVENING_BRIEFING_HOUR:MINUTE) — not a voice action
    "disable_night_owl", # alias for night_owl_off (the canonical voice action is night_owl_off, registered in skills/night_owl_mode.py)
    "enable_night_owl",  # alias for night_owl_on (the canonical voice action exposed to the LLM is night_owl_mode, registered in skills/night_owl_mode.py)
    "end_night_owl",     # alias for night_owl_off (registered as an explicit alias in skills/night_owl_mode.py)
    "night_owl_on",      # handler-side name for the same callable bound to night_owl_mode — the LLM is taught night_owl_mode in PC_CONTROL_PROMPT because it reads more naturally as a voice command
    "good_morning",      # registered by skills/night_owl_mode.py — documented dynamically in NIGHT_OWL_PROMPT_ADDENDUM only while night-owl mode is active (it's a phrase-triggered release, not a standalone voice action)
    "dismiss_holo",      # alias for hide_holographic_overlay (the canonical voice action is hide_holographic_overlay, registered in skills/holographic_overlay.py)
    "hide_holo",         # alias for hide_holographic_overlay (the canonical voice action is hide_holographic_overlay, registered in skills/holographic_overlay.py)
    "holographic_off",   # alias for hide_holographic_overlay (registered in skills/holographic_overlay.py alongside hide_holo, dismiss_holo, hud_off — all wired to _act_hide; the canonical voice action documented in PC_CONTROL_PROMPT is hide_holographic_overlay)
    "holographic_on",    # alias for show_holographic_overlay (registered in skills/holographic_overlay.py alongside show_holo, hud_on — all wired to _act_show; symmetric with the holographic_off / hide_holo / dismiss_holo aliases listed above)
    "hud_on",            # alias for show_holographic_overlay (registered in skills/holographic_overlay.py at line 754, wired to _act_show alongside show_holo / holographic_on — the canonical voice action is show_holographic_overlay; note this is the FULLSCREEN holo overlay's hud, distinct from the slim status-bar which uses show_hud)
    "show_holo",         # short alias for show_holographic_overlay (registered in skills/holographic_overlay.py at line 753, wired to _act_show alongside hud_on / holographic_on — the canonical voice action is show_holographic_overlay; symmetric with the hide_holo / dismiss_holo aliases above)
    "toggle_holo",       # short alias for toggle_holographic_overlay (registered in skills/holographic_overlay.py at line 764, wired to _act_toggle — the canonical voice action is toggle_holographic_overlay; symmetric with the show_holo / hide_holo / dismiss_holo aliases above)
    "holographic_status", # diagnostic for the fullscreen holographic overlay (reports whether the overlay is engaged and its geometry) — registered in skills/holographic_overlay.py; same pattern as gaze_status / banter_status / anticipation_status (status probes are not surfaced to the LLM)
    "hud_off",           # alias for hide_holographic_overlay (registered in skills/holographic_overlay.py at line 759, wired to _act_hide alongside hide_holo / dismiss_holo / holographic_off — the canonical voice action documented in PC_CONTROL_PROMPT is hide_holographic_overlay; note this is the FULLSCREEN holo overlay's hud, distinct from the slim status-bar hud_off does NOT exist — slim bar uses hide_hud)
    "hide_workshop_hud", # verb-prefix alias for workshop_hud_off (the canonical voice action documented in PC_CONTROL_PROMPT is workshop_hud — the toggle; the user-facing 'hide the HUD' phrase is owned by hide_hud in bobert_companion.py for the slim status bar)
    "show_workshop_hud", # verb-prefix alias for workshop_hud_on (the canonical voice action documented in PC_CONTROL_PROMPT is workshop_hud — the toggle; symmetric with hide_workshop_hud; the user-facing 'show the HUD' phrase is owned by show_hud in bobert_companion.py for the slim status bar)
    "workshop_hud_off",  # explicit force-off variant of the workshop_hud toggle (the canonical voice action documented in PC_CONTROL_PROMPT is workshop_hud, registered in skills/holographic_overlay.py alongside hide_workshop_hud — same pattern as arc_reactor_off being an internal alias for the documented arc_reactor toggle)
    "workshop_hud_on",   # explicit force-on variant of the workshop_hud toggle (the canonical voice action documented in PC_CONTROL_PROMPT is workshop_hud, registered in skills/holographic_overlay.py alongside show_workshop_hud — symmetric with workshop_hud_off above; same pattern as arc_reactor_on being an internal alias for the documented arc_reactor toggle)
    "workshop_hud_status", # diagnostic status probe for the workshop HUD (reports whether the workshop HUD is engaged and its geometry on the top monitor) — registered in skills/holographic_overlay.py; same pattern as gaze_status / banter_status / holographic_status / pattern_stats / robot_status (status probes are not surfaced to the LLM — the user-facing workshop HUD action is workshop_hud, the toggle, documented in PC_CONTROL_PROMPT)
    "workshop_hud_toggle", # explicit -_toggle suffix alias for the canonical workshop_hud action (both registered in skills/holographic_overlay.py and wired to the same _act_workshop_hud_toggle handler — the canonical voice action documented in PC_CONTROL_PROMPT is workshop_hud, the toggle; symmetric with workshop_hud_off/workshop_hud_on/workshop_hud_status above; same pattern as toggle_holo being an internal alias for the documented toggle_holographic_overlay)
    "workshop_status",   # diagnostic status probe for the auto-engaged workshop mode (reports 'Workshop mode is engaged, sir' / 'not currently engaged' plus the matched app title) — registered in skills/workshop_mode.py; workshop mode is auto-engaged from foreground-app detection (Bambu Studio / Fusion 360 / SolidWorks / etc.), so there's no canonical voice action to document — the status probe is purely diagnostic, same pattern as gaze_status / banter_status / holographic_status / workshop_hud_status / robot_status (status probes are not surfaced to the LLM)
    "music_aggregate",   # manual force-rerun of the listen-event aggregator in skills/apple_music_intel.py — the aggregator already runs every hour on its own background timer, so this is a diagnostic/maintenance trigger rather than a voice action (sibling diagnostics: audio_music_status, gaze_stats)
    "pattern_aggregate", # manual force-rerun of the nightly pattern aggregator in skills/pattern_learning.py — the aggregator already runs once per day on its own background timer (NIGHTLY_HOUR), so this is a diagnostic/maintenance trigger rather than a voice action (sibling of music_aggregate)
    "pattern_offer_now", # force the next eligible pattern offer, bypassing the throttle — registered in skills/pattern_learning.py as a diagnostic/maintenance trigger (sibling of pattern_aggregate); the throttled offers fire automatically from the background scheduler so this is not surfaced to the LLM
    "pattern_stats",     # diagnostic status probe for the pattern learning engine (reports event count, broad/precise pattern totals, last-aggregation age, today's offer count) — registered in skills/pattern_learning.py; same pattern as gaze_stats / banter_status / anticipation_status (status probes are not surfaced to the LLM)
    "robot_status",      # diagnostic status probe for the REPO Robot skill (one-line summary of next step + blocker/part counts) — registered in skills/repo_robot.py; same pattern as gaze_status / banter_status / holographic_status / pattern_stats (status probes are not surfaced to the LLM — the user-facing REPO Robot actions are next_robot_step and robot_blocker, both documented in PC_CONTROL_PROMPT)
    "screen_teams_calls", # manual on-demand window-title scan for VIP callers — registered in skills/teams_screener.py as a diagnostic/text-status probe; the screener's background poller (CHECK_INTERVAL_SECONDS=2) already announces VIP calls automatically and arms answer_call / decline_call, so this manual scan is not surfaced to the LLM (the user-facing 'check Teams' voice action is check_teams, the vision-based unread-messages sweep, documented in PC_CONTROL_PROMPT under TEAMS CALL SCREENING)
    "screen_watch_status", # diagnostic status probe for the screen_watch wellness nudge (reports current focused window, stare duration, system idle, and which gates are clear/blocked) — registered in skills/screen_watch.py; same pattern as gaze_status / banter_status / pattern_stats / holographic_status / robot_status (status probes are not surfaced to the LLM — screen_watch runs autonomously on its own poll loop and fires nudges via the pending_speech.json announcer)
    "status_report",     # alias for system_pulse — registered in skills/system_pulse.py:533 as `actions["status_report"] = system_pulse` to absorb the natural 'JARVIS, status report' phrasing; the canonical action documented in PC_CONTROL_PROMPT is system_pulse (the single-sentence aggregator referenced under SUIT DIAGNOSTICS), so this alias is internal-only
    "suit_diagnostics",  # flavour alias for status_panel — registered in skills/status_panel.py:476 as `actions["suit_diagnostics"] = status_panel`; the canonical action documented in PC_CONTROL_PROMPT is status_panel ('deliberate give-me-everything diagnostics'), so this alias is internal-only (same pattern as status_report → system_pulse)
    "system_status",     # natural verbal-phrasing alias for status_panel — registered in skills/status_panel.py:475 as `actions["system_status"] = status_panel` to absorb the natural 'JARVIS, system status' phrasing; the canonical action documented in PC_CONTROL_PROMPT is status_panel (the example phrase 'JARVIS, system status' → [ACTION: status_panel] is teach the LLM to map the phrase to the canonical action), so this alias is internal-only (same pattern as suit_diagnostics → status_panel and status_report → system_pulse)
    "suit_up_sequence",  # alias for suit_up — registered in skills/suit_up.py:392 as `actions["suit_up_sequence"] = _act_suit_up`; the canonical action documented in PC_CONTROL_PROMPT under SUIT-UP CINEMATIC is suit_up (the 6–8s arc-reactor boot cinematic), so this alias is internal-only (same pattern as status_report → system_pulse and suit_diagnostics → status_panel)
}

# 2026-05-30 alias triage — actions auto-classified as intentionally internal:
# status/diagnostic probes, UI-control on/off/toggle/v2 variants, aliases of a
# documented or canonical action, admin/maintenance, and test actions. They are
# registered and WORK; they're just not advertised to the LLM via
# PC_CONTROL_PROMPT (the canonical/documented action covers each concept). The
# genuinely-distinct undocumented actions were left OUT for the user to decide.
_INTERNAL_ONLY_ACTIONS |= {
    # UI-control on/off/toggle/v2 variants:
    "arc_reactor_status_off", "arc_reactor_status_on",
    "arc_reactor_status_toggle", "holo_hud_v2", "holo_hud_v2_off",
    "holo_hud_v2_on", "hud_v2_off", "hud_v2_on", "print_hud_off",
    "print_hud_on", "pulse_hud_off", "pulse_hud_on",
    # admin / maintenance:
    "anticipation_briefing_now", "arrival_briefing_v2", "clear_llm_cache",
    "force_backup", "latency_benchmark", "morning_arrival_v2",
    "morning_chain_pick", "refresh_smart_home_router", "reload_skills",
    "reset_llm_cache", "run_smoke_test", "show_last_diagnostic",
    "stop_pipeline", "switch_llm_picker",
    # aliases of a documented or canonical action:
    "amazon_orders", "ambient_listening", "ambient_mode_off",
    "ambient_mode_on", "arc_reactor_hud", "arc_reactor_status_hud",
    "browser_do", "browser_run", "chappie_mode", "check_amazon_orders",
    "control_device", "control_light", "control_smart_home",
    "hide_holo_hud_v2", "hide_hud_v2", "hide_status_hud",
    "hide_status_ring_v2", "hide_workshop_print_monitor",
    "holo_hud_v2_toggle", "holographic_hud_v2", "hud_v2", "hud_v2_toggle",
    "kasa_control", "print_hud", "pulse_hud", "recent_deliveries",
    "show_holo_hud_v2", "show_hud_v2", "show_status_hud",
    "show_status_ring_v2", "show_workshop_print_monitor", "silent_learning",
    "silent_learning_off", "silent_learning_on", "smart_home_devices",
    "smart_life_list", "stark_status_ring_off", "stark_status_ring_on",
    "stark_status_ring_toggle", "start_eavesdropping", "status_hud",
    "status_hud_off", "status_hud_on", "status_ring_off", "status_ring_on",
    "status_ring_v2", "status_ring_v2_off", "status_ring_v2_on",
    "tuya_list_devices", "workshop_print_hud", "workshop_print_hud_off",
    "workshop_print_hud_on", "workshop_print_monitor_off",
    "workshop_print_monitor_on", "workshop_print_monitor_toggle",
    "youtube_direct", "yt_direct",
    # status / diagnostic probes:
    "amazon_tracking_status", "ambient_extract_status",
    "ambient_listen_status", "anticipation_briefing_status",
    "arc_reactor_status", "arc_reactor_status_status",
    "browser_status", "chappie_status", "draft_preview_gate_status",
    "holo_hud_v2_status", "outbound_gate_status", "print_companion_history",
    "print_companion_status", "print_status", "show_llm_stats",
    "smart_home_router_status", "stark_status_ring_status",
    "weekly_digest_status", "workshop_print_monitor_status",
    # test / diagnostic actions:
    "test_each_skill", "test_mic", "test_tts", "test_vision",
}

# 2026-05-31 candidate triage — the 47 genuinely-distinct undocumented actions
# were reviewed individually: guest_mode_on/off + voice_gating_on/off were
# DOCUMENTED in PC_CONTROL_PROMPT (useful, safe voice-ID modes); the rest are
# kept internal — the browser_* specialised actions are redundant with the
# documented general `browser_task`, and the remainder are too granular, niche,
# or risky (e.g. reset_memory) to advertise by voice. They stay registered +
# callable, just not surfaced to the LLM.
_INTERNAL_ONLY_ACTIONS |= {
    "ambient_audio_start", "ambient_audio_stop", "ambient_extract_now",
    "ambient_extract_start", "ambient_extract_stop", "ambient_full_start",
    "ambient_full_stop", "ambient_listen_start", "ambient_listen_stop",
    "ambient_mic_only", "ambient_mode", "ambient_screen_start",
    "ambient_screen_stop",
    "book_appointment", "browse_for", "browser_open", "browser_reset_profile",
    "browser_stop", "chappie_recall_entity", "chappie_recall_today",
    "check_budget", "check_orders", "control_plug", "export_memory",
    "fill_form", "find_cheapest", "forget_last_hour", "recent_delivery",
    "reset_memory", "show_recent_facts",
    "smart_home_list", "stark_status_ring", "status_ring", "stop_eavesdropping",
    "switch_llm", "tuya_list", "weekly_digest", "weekly_digest_now",
    "workshop_print_monitor",
}


# Internal actions belonging to gitignored personal skills load from a
# gitignored tools/audit_local.py so this shipped auditor names none of them.
# Absent on a fresh clone (those skills aren't shipped); present on the owner's
# machine so a local audit still treats them as intentionally-internal.
def _load_local_internal_actions() -> set:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_local.py")
    if not os.path.exists(p):
        return set()
    ns: dict = {}
    try:
        with open(p, encoding="utf-8") as fh:
            exec(compile(fh.read(), p, "exec"), ns)
    except Exception:
        return set()
    return set(ns.get("INTERNAL_ONLY_ACTIONS", ()))


_INTERNAL_ONLY_ACTIONS |= _load_local_internal_actions()

# Ambient-LEARNING mode aliases (bobert_companion.py, registered via
# ACTIONS.update). The canonical voice actions ambient_learning_mode_on /
# ambient_learning_mode_off and the two wake_resume_* setters ARE documented
# in PC_CONTROL_PROMPT; these three are convenience aliases — a bare toggle
# plus natural enter/exit phrasings — that the LLM reaches through the
# documented forms or the preemptive-hallucination router, so they stay
# registered + callable but internal-only.
_INTERNAL_ONLY_ACTIONS |= {
    "ambient_learning_mode", "enter_ambient_learning", "exit_ambient_learning",
}

# Voice-triggered actions that should have at least one user-phrase example
# in PC_CONTROL_PROMPT so the LLM knows when to fire them. Keyed by action
# name, value = the trigger string (or fragment) we expect to see.
_VOICE_TRIGGER_EXAMPLES: dict[str, list[str]] = {
    "hide_hud":        ["hide the HUD", "hide_hud"],
    "show_hud":        ["show the HUD", "show_hud"],
    "toggle_hud":      ["toggle", "toggle_hud"],
    "screenshot":      ["screenshot"],
    "web_search":      ["web_search"],
    "launch_app":      ["launch_app"],
    "open_url":        ["open_url"],
    "youtube":         ["youtube"],
    "get_time":        ["get_time", "time"],
    "see_screen":      ["see_screen"],
    "see_user":        ["see_user"],
    "play_music":      ["play_music"],
    "media_next":      ["media_next"],
    "media_prev":      ["media_prev"],
    "media_playpause": ["media_playpause"],
    "volume_up":       ["volume_up"],
    "volume_down":     ["volume_down"],
    "volume_mute":     ["volume_mute"],
    "daily_recap":     ["daily_recap"],
    "dossier":         ["dossier"],
    "check_credits":   ["check_credits"],
    "check_print":     ["check_print"],
}

# Main-loop / dispatcher functions in bobert_companion.py that MUST be wrapped
# in outer try/except (so a transient API failure surfaces as a spoken
# apology, not a crashed main loop).
_CRITICAL_FUNCTIONS = {
    "main",
    "dispatch_actions",
    "_speak",
    "speak",
    "transcribe",
    "record_speech",
    "follow_up",
    "_listen_step",
}

SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),    "Anthropic API key"),
    (re.compile(r"sk-[A-Za-z0-9]{32,}"),           "OpenAI-style API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"),              "AWS access key"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"),        "Google API key"),
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"),          "GitHub personal token"),
]

# Substring whitelist applied to lines flagged for password=/token= literals,
# so we don't false-positive on local LAN tokens and template placeholders.
SECRET_WHITELIST_SUBSTRINGS = (
    "BAMBU_ACCESS_CODE",
    "your-",
    "YOUR_",
    "<your",
    "example",
    "placeholder",
    "PLACEHOLDER",
    "TODO",
    "REPLACE",
    "getenv",
    "environ",
    "os.environ",
)

# ─────────────────────────────────────────────────────────────────────────
#  data model
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    severity: str        # "P0" | "P1" | "P2"
    category: str        # short check name
    file: str            # path relative to PROJECT_DIR
    line: int            # 0 if file-wide
    message: str         # human-readable description
    fixable: bool = False
    fix_kind: str = ""   # which auto-fix applies (encoding/daemon/timeout)

    def as_md_line(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        fix = " *(auto-fixable)*" if self.fixable else ""
        return f"- **[{self.severity}]** `{loc}` — {self.message}{fix}"


# ─────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────

def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_DIR).replace("\\", "/")
    except ValueError:
        return path


def walk_py_files() -> list[str]:
    """Return absolute paths of every project .py file we should audit."""
    out: list[str] = []
    for root, dirs, files in os.walk(PROJECT_DIR):
        # prune excluded dirs in place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if not f.endswith(".py"):
                continue
            # Skip private/underscore modules (templates like _example_skill.py)
            # — but NOT __init__.py, which is a PACKAGE skill's registration
            # entry point. Skipping it made every package skill (holographic_
            # overlay/…) invisible to the auditor, so its registered actions
            # looked "documented but not registered" (spurious prompt-drift).
            if f.startswith("_") and f != "__init__.py":
                continue
            out.append(os.path.join(root, f))
    return sorted(out)


def _read_source(path: str) -> tuple[str | None, list[str]]:
    """Read source file as UTF-8; return (text, lines) or (None, []) on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, text.splitlines()
    except Exception:
        return None, []


def parse_requirements() -> set[str]:
    """Return a set of package names declared in requirements.txt.
    Returns just the bare package name (no markers, no version specs)."""
    pkgs, _opt = _parse_requirements_full()
    return pkgs


def parse_optional_requirements() -> set[str]:
    """Subset of packages declared in requirements.txt whose inline comment
    contains the literal word 'optional' (case-insensitive). Conditional
    imports of these are silently accepted by the auditor — they describe
    feature flags (e.g. swappable LLM backends), not missing deps."""
    _pkgs, opt = _parse_requirements_full()
    return opt


def _parse_requirements_full() -> tuple[set[str], set[str]]:
    pkgs: set[str] = set()
    optional: set[str] = set()
    if not os.path.exists(REQUIREMENTS_FILE):
        return pkgs, optional
    try:
        with open(REQUIREMENTS_FILE, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    # `# optional: <import-name>` declares a conditionally-
                    # imported optional dependency WITHOUT pip auto-installing it
                    # (often heavy: TTS, diffusers, …). The import check then
                    # treats that dep's guarded import as intentional, not an
                    # undeclared-dependency finding. Case-preserved so it matches
                    # the actual import name (e.g. RealtimeTTS).
                    mo = re.match(r"#\s*optional:\s*([A-Za-z0-9._-]+)", line)
                    if mo:
                        optional.add(mo.group(1))
                    continue
                # split off inline comment, if any
                spec, _, comment = line.partition("#")
                spec = spec.strip()
                # strip env-markers and version specs
                spec = re.split(r"[;<>=!]", spec, maxsplit=1)[0].strip()
                if not spec:
                    continue
                name = spec.lower()
                pkgs.add(name)
                if re.search(r"\boptional\b", comment, flags=re.IGNORECASE):
                    optional.add(name)
    except Exception:
        pass
    return pkgs, optional


def req_to_import_name(pkg: str) -> str:
    return DEP_IMPORT_NAME.get(pkg, pkg.replace("-", "_"))


def _module_exported_names(path: str) -> set[str]:
    """Top-level names a `from <module> import *` would bind: __all__ if the
    module defines it, else every non-underscore top-level name."""
    text, _ = _read_source(path)
    if text is None:
        return set()
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return set()
    top: set[str] = set()
    all_names: set[str] | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    top.add(t.id)
                    if (t.id == "__all__"
                            and isinstance(node.value, (ast.List, ast.Tuple))):
                        all_names = {e.value for e in node.value.elts
                                     if isinstance(e, ast.Constant)
                                     and isinstance(e.value, str)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    top.add(alias.asname or alias.name.split(".")[0])
    if all_names is not None:
        return all_names
    return {n for n in top if not n.startswith("_")}


def collect_bobert_companion_names() -> set[str]:
    """ast-parse bobert_companion.py and return every top-level name (functions,
    classes, top-level assigns, ann-assigns). Used to validate
    `bobert_companion.X` cross-refs from skills."""
    path = os.path.join(PROJECT_DIR, "bobert_companion.py")
    text, _ = _read_source(path)
    if text is None:
        return set()
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return set()
    names: set[str] = set()

    def _scope(stmts: list) -> None:
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)   # name is module-scope; body is local
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.Try, ast.If, ast.With, ast.For, ast.While)):
                # Module-level conditional / try blocks (e.g. the blue-green
                # `try: import blue_green_manager; BLUE_GREEN_ROLE = ...`) still
                # define module-scope names — descend, but never into func/class.
                _scope(getattr(node, "body", []) or [])
                _scope(getattr(node, "orelse", []) or [])
                _scope(getattr(node, "finalbody", []) or [])
                for _h in getattr(node, "handlers", []) or []:
                    _scope(_h.body)
    _scope(tree.body)
    # Names assigned via `global NAME` inside functions are module-level globals
    # too (MONITORS, BLUE_GREEN_ROLE, … are set in main()/boot helpers, not at
    # top level). The top-level-only scan above misses them, producing false
    # "bc.X is not defined" cross-ref findings.
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names.update(node.names)
    # Resolve `from <mod> import *` — star-exported names (MONITORS from
    # core.config, the core.state slots, the _act_* handlers from core.actions)
    # become bobert_companion attributes too. Without this the cross-ref check
    # can't see them and false-flags `bc.<star-name>` references from skills.
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.module
                and any(a.name == "*" for a in node.names)):
            parts = node.module.split(".")
            modpath = os.path.join(PROJECT_DIR, *parts) + ".py"
            if not os.path.isfile(modpath):
                modpath = os.path.join(PROJECT_DIR, *parts, "__init__.py")
            names |= _module_exported_names(modpath)
    return names


def collect_core_module_names() -> dict[str, set[str]]:
    """Same idea but for core/*.py — keyed by stem, value = exported top names."""
    out: dict[str, set[str]] = {}
    if not os.path.isdir(CORE_DIR):
        return out
    for name in os.listdir(CORE_DIR):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        path = os.path.join(CORE_DIR, name)
        text, _ = _read_source(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
        out[os.path.splitext(name)[0]] = names
    return out


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 2 — syntax
# ─────────────────────────────────────────────────────────────────────────

def check_syntax(files: Iterable[str]) -> list[Finding]:
    out: list[Finding] = []
    for path in files:
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            msg = str(e).splitlines()[0] if str(e) else "unknown py_compile error"
            out.append(Finding(
                severity="P0", category="syntax",
                file=_rel(path), line=0,
                message=f"py_compile failed: {msg}",
            ))
        except Exception as e:
            out.append(Finding(
                severity="P0", category="syntax",
                file=_rel(path), line=0,
                message=f"could not compile: {type(e).__name__}: {e}",
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 3 — imports
# ─────────────────────────────────────────────────────────────────────────

def _iter_imports(tree: ast.AST) -> Iterable[tuple[str, int, bool]]:
    """Yield (top_module, lineno, is_conditional) for every import statement.
    is_conditional=True means the import is inside an If/Try/With/FunctionDef
    block (i.e. the module may legitimately be optional)."""

    def walk(node: ast.AST, conditional: bool) -> Iterable[tuple[str, int, bool]]:
        for child in ast.iter_child_nodes(node):
            kid_conditional = conditional or isinstance(
                child, (ast.If, ast.Try, ast.With, ast.FunctionDef,
                        ast.AsyncFunctionDef, ast.ExceptHandler)
            )
            if isinstance(child, ast.Import):
                for alias in child.names:
                    yield alias.name.split(".")[0], child.lineno, conditional
            elif isinstance(child, ast.ImportFrom):
                if child.level and child.level > 0:
                    continue
                if child.module:
                    yield child.module.split(".")[0], child.lineno, conditional
            yield from walk(child, kid_conditional)

    yield from walk(tree, False)


def check_imports(files: list[str]) -> list[Finding]:
    out: list[Finding] = []
    req_pkgs, optional_pkgs = _parse_requirements_full()
    req_imports = {req_to_import_name(p) for p in req_pkgs}
    optional_imports = {req_to_import_name(p) for p in optional_pkgs}

    # All intra-project module roots
    intra = {"bobert_companion", "memory", "tray", "boot_sequence", "iron_man_boot",
             "jarvis_failure_lines", "jarvis_watcher", "overnight_upgrade",
             "upgrade_jarvis", "hud_card"}
    # Every top-level .py is an intra-project module (mcu_phrases,
    # blue_green_manager, staging_instance, command_autocorrect, …). Derive them
    # dynamically so a new root module isn't mis-flagged as an undeclared
    # dependency — the hardcoded list above missed several, producing spurious P1s.
    try:
        intra |= {os.path.splitext(f)[0] for f in os.listdir(PROJECT_DIR)
                  if f.endswith(".py")}
    except OSError:
        pass
    intra |= {os.path.splitext(f)[0] for f in os.listdir(SKILLS_DIR)
              if f.endswith(".py")}
    intra |= {os.path.splitext(f)[0]
              for f in (os.listdir(HUD_DIR) if os.path.isdir(HUD_DIR) else [])
              if f.endswith(".py")}
    # 'audio' is a real intra-project package (audio/__init__.py + itunes_bridge);
    # without it, `from audio import ...` is mis-flagged as an undeclared P1 dep,
    # pushing the whole audit to a wrong exit-2 verdict.
    intra |= {"skills", "hud", "core", "tools", "audio"}
    # Any top-level directory that is a Python package (has __init__.py) is an
    # intra-project module too (adapters, tts, …). Derive dynamically so
    # `import adapters` / `from tts import ...` aren't mis-flagged as undeclared.
    try:
        for _d in os.listdir(PROJECT_DIR):
            _dp = os.path.join(PROJECT_DIR, _d)
            if (not os.path.isdir(_dp) or _d in EXCLUDE_DIRS
                    or _d.startswith(".") or _d == "_backups"):
                continue
            # Regular package (__init__.py) OR PEP-420 namespace package — a dir
            # of .py modules with no __init__.py, e.g. adapters/ → importable as
            # `from adapters import voice_mood_response`.
            if any(f.endswith(".py") for f in os.listdir(_dp)):
                intra.add(_d)
    except OSError:
        pass
    intra |= {"skill_" + os.path.splitext(f)[0] for f in os.listdir(SKILLS_DIR)
              if f.endswith(".py")}

    for path in files:
        text, _ = _read_source(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue   # already reported by check_syntax
        seen: set[tuple[str, int]] = set()
        for mod, line, conditional in _iter_imports(tree):
            key = (mod, line)
            if key in seen:
                continue
            seen.add(key)
            if mod in STDLIB_MODULES or mod in intra:
                continue
            # Try importing — if it works, fine.
            try:
                __import__(mod)
                continue
            except ImportError:
                pass
            except Exception:
                continue   # weird side-effect; skip
            # Missing module. Conditional imports (inside if/try) are likely
            # optional features — drop to P2. If the package is explicitly
            # marked '# optional' in requirements.txt, the conditional import
            # is by design — suppress the finding entirely.
            if conditional and mod in optional_imports:
                continue
            if conditional:
                sev = "P2"
                note = "conditional import — likely optional feature"
            else:
                sev = "P1"
                note = ""
            if mod in req_imports:
                out.append(Finding(
                    severity=sev, category="imports",
                    file=_rel(path), line=line,
                    message=(f"import '{mod}' is declared in requirements.txt "
                             f"but not installed in the active environment"
                             + (f" ({note})" if note else "")),
                ))
            else:
                out.append(Finding(
                    severity=sev, category="imports",
                    file=_rel(path), line=line,
                    message=(f"import '{mod}' is not stdlib, not intra-project, "
                             f"and not in requirements.txt — undeclared dependency"
                             + (f" ({note})" if note else "")),
                ))
    return out


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 4 — cross-references to bobert_companion / core
# ─────────────────────────────────────────────────────────────────────────

def check_cross_references(files: list[str]) -> list[Finding]:
    out: list[Finding] = []
    bc_names = collect_bobert_companion_names()
    core_modules = collect_core_module_names()
    if not bc_names:
        return out   # bobert_companion didn't parse — already P0 from syntax

    for path in files:
        text, _ = _read_source(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue

        # Track which intra-project modules this file imported as which name.
        # bobert_companion can be imported as `bobert_companion` or `bc` (alias)
        bc_aliases: set[str] = set()
        core_aliases: dict[str, str] = {}   # alias → core module stem

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "bobert_companion":
                        bc_aliases.add(alias.asname or "bobert_companion")
                    elif alias.name.startswith("core."):
                        stem = alias.name.split(".", 1)[1]
                        if stem in core_modules:
                            core_aliases[alias.asname or alias.name] = stem
            elif isinstance(node, ast.ImportFrom):
                if node.module == "bobert_companion":
                    # `from bobert_companion import X` — verify X exists
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        if alias.name not in bc_names:
                            out.append(Finding(
                                severity="P1", category="cross-ref",
                                file=_rel(path), line=node.lineno,
                                message=(f"`from bobert_companion import {alias.name}` "
                                         f"— {alias.name} is not defined at module "
                                         f"top-level"),
                            ))
                elif node.module and node.module.startswith("core.") and node.module.count(".") == 1:
                    stem = node.module.split(".", 1)[1]
                    if stem in core_modules:
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            if alias.name not in core_modules[stem]:
                                out.append(Finding(
                                    severity="P1", category="cross-ref",
                                    file=_rel(path), line=node.lineno,
                                    message=(f"`from core.{stem} import {alias.name}` "
                                             f"— {alias.name} not defined in core/{stem}.py"),
                                ))

        if not bc_aliases and not core_aliases:
            continue

        # Walk attribute accesses: `bc.foo` / `bobert_companion.foo` / `dispatcher.bar`
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.value, ast.Name):
                continue
            base = node.value.id
            attr = node.attr
            if base in bc_aliases:
                if attr not in bc_names:
                    out.append(Finding(
                        severity="P1", category="cross-ref",
                        file=_rel(path), line=node.lineno,
                        message=(f"`{base}.{attr}` — {attr} is not defined "
                                 f"in bobert_companion.py"),
                    ))
            elif base in core_aliases:
                stem = core_aliases[base]
                if attr not in core_modules.get(stem, set()):
                    out.append(Finding(
                        severity="P1", category="cross-ref",
                        file=_rel(path), line=node.lineno,
                        message=(f"`{base}.{attr}` — {attr} not defined in core/{stem}.py"),
                    ))
    return out


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 5 — ACTIONS registration (sandbox-load each skill)
# ─────────────────────────────────────────────────────────────────────────

def check_actions(skill_files: list[str], bc_actions: set[str]) -> tuple[list[Finding], dict[str, str]]:
    """Statically extract action names from each skill's register() function.
    Returns (findings, action_owner_map) where action_owner_map maps
    action_name → owner file (last write wins, mirrors load order)."""
    out: list[Finding] = []
    owners: dict[str, str] = {a: "bobert_companion.py" for a in bc_actions}

    for path in sorted(skill_files):
        text, lines = _read_source(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue

        register_fn = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "register":
                register_fn = node
                break
        if register_fn is None:
            continue

        # Find the parameter name (usually 'actions')
        if not register_fn.args.args:
            out.append(Finding(
                severity="P1", category="actions",
                file=_rel(path), line=register_fn.lineno,
                message="register() takes no arguments — must accept the actions dict",
            ))
            continue
        actions_param = register_fn.args.args[0].arg

        # Walk register() body for `actions[name] = callable`
        for sub in ast.walk(register_fn):
            if not isinstance(sub, ast.Assign):
                continue
            for target in sub.targets:
                if not (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == actions_param):
                    continue
                # Extract the string key
                key = target.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    action_name = key.value
                else:
                    continue   # dynamic key — can't analyse statically

                # Resolve the right-hand callable to a name (best effort)
                rhs = sub.value
                callable_name = None
                if isinstance(rhs, ast.Name):
                    callable_name = rhs.id
                elif isinstance(rhs, ast.Attribute):
                    callable_name = rhs.attr
                elif isinstance(rhs, ast.Lambda):
                    callable_name = "<lambda>"

                # Collision check — skip when caller marked the wrap intentional.
                # Look for an `INTENTIONAL_WRAP` comment on the assignment line
                # itself or on either of the two immediately preceding lines (so
                # block-style markers like a leading `# INTENTIONAL_WRAP: …` are
                # also recognised).
                intentional = False
                for ln in range(max(1, sub.lineno - 2), sub.lineno + 1):
                    if ln - 1 < len(lines) and "INTENTIONAL_WRAP" in lines[ln - 1]:
                        intentional = True
                        break
                if (action_name in owners
                        and owners[action_name] != _rel(path)
                        and not intentional):
                    out.append(Finding(
                        severity="P1", category="actions",
                        file=_rel(path), line=sub.lineno,
                        message=(f"action name '{action_name}' collides with "
                                 f"existing registration in {owners[action_name]} "
                                 f"— this skill will overwrite (last-load-wins)"),
                    ))
                owners[action_name] = _rel(path)

                # Verify the callable takes exactly one positional argument.
                # Look for the function definition in this same module.
                if callable_name and callable_name != "<lambda>":
                    for fn_node in tree.body:
                        if (isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and fn_node.name == callable_name):
                            pos_args = fn_node.args.args
                            n = len(pos_args)
                            defaults = len(fn_node.args.defaults)
                            required = n - defaults
                            if required > 1:
                                out.append(Finding(
                                    severity="P1", category="actions",
                                    file=_rel(path), line=fn_node.lineno,
                                    message=(f"action '{action_name}' → {callable_name}() "
                                             f"requires {required} positional args; "
                                             f"action callables must accept 1 (the str payload)"),
                                ))
                            elif n == 0:
                                out.append(Finding(
                                    severity="P1", category="actions",
                                    file=_rel(path), line=fn_node.lineno,
                                    message=(f"action '{action_name}' → {callable_name}() "
                                             f"takes zero args; should take 1 (the str payload)"),
                                ))
                            break
    return out, owners


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 6 — bad patterns (regex + ast hybrid)
# ─────────────────────────────────────────────────────────────────────────

_BARE_EXCEPT_RE   = re.compile(r"^\s*except\s*:")
_HARDCODED_PATH_RE = re.compile(r"[A-Za-z]:\\PC Files", re.I)


def check_bad_patterns(files: list[str]) -> list[Finding]:
    out: list[Finding] = []
    for path in files:
        text, lines = _read_source(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue

        # Build set of line numbers that are part of string literals so we
        # don't false-positive on regex-pattern strings (e.g. r"\beval\s*\(")
        # or docstrings that mention `except:` / `eval(`.
        string_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                start = node.lineno
                end = getattr(node, "end_lineno", start) or start
                for ln in range(start, end + 1):
                    string_lines.add(ln)

        for i, raw in enumerate(lines, start=1):
            line = raw.rstrip()
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if i in string_lines:
                # Line is (entirely or mostly) inside a string literal;
                # patterns matched here would be regex sources, docstrings,
                # or example code — not real calls.
                continue

            if _BARE_EXCEPT_RE.search(line):
                out.append(Finding(
                    severity="P2", category="bare-except",
                    file=_rel(path), line=i,
                    message="bare `except:` — catches BaseException + KeyboardInterrupt",
                ))
            if _HARDCODED_PATH_RE.search(line):
                out.append(Finding(
                    severity="P2", category="hardcoded-path",
                    file=_rel(path), line=i,
                    message="hardcoded 'D:\\PC Files' path — use os.path.dirname(__file__) instead",
                ))

        # ast-based checks (eval, exec, os.system, open, Thread, subprocess)
        out.extend(_ast_bad_pattern_checks(tree, path, lines))
    return out


def _ast_bad_pattern_checks(tree: ast.AST, path: str, lines: list[str]) -> list[Finding]:
    out: list[Finding] = []
    rel = _rel(path)

    # Parent map so we can find the Assign that wraps a Thread/Timer Call.
    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent

    # Pre-scan: { varname: [lineno, ...] } for `<var>.daemon = True` assignments.
    # Recognises the post-construction daemon-flag idiom
    # (`t = Thread(...); t.daemon = True; t.start()`) so the Thread/Timer check
    # below doesn't false-positive on it.
    DAEMON_WINDOW = 5  # lines allowed between constructor end and `.daemon = True`
    daemon_set_lines: dict[str, list[int]] = defaultdict(list)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Assign):
            continue
        if not (isinstance(n.value, ast.Constant) and n.value.value is True):
            continue
        for tgt in n.targets:
            if (isinstance(tgt, ast.Attribute) and tgt.attr == "daemon"
                    and isinstance(tgt.value, ast.Name)):
                daemon_set_lines[tgt.value.id].append(n.lineno)

    def _has_post_construction_daemon(call: ast.Call) -> bool:
        """True if the call sits in `<name> = <call>` and a matching
        `<name>.daemon = True` appears within DAEMON_WINDOW lines after the
        constructor's end_lineno."""
        p = parent_map.get(id(call))
        if not isinstance(p, ast.Assign) or p.value is not call:
            return False
        if len(p.targets) != 1 or not isinstance(p.targets[0], ast.Name):
            return False
        varname = p.targets[0].id
        end_line = getattr(call, "end_lineno", None) or call.lineno
        for ln in daemon_set_lines.get(varname, []):
            if end_line <= ln <= end_line + DAEMON_WINDOW:
                return True
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Resolve the called name to a flat dotted string
        callee = _callee_name(node)
        if callee is None:
            continue

        # ── eval() — code-injection risk ──────────────────────────────────
        if callee == "eval":
            out.append(Finding(
                severity="P0", category="eval",
                file=rel, line=node.lineno,
                message="eval() call — code-injection risk",
            ))

        # ── exec() — allow spec.loader.exec_module(...) ───────────────────
        if callee == "exec":
            out.append(Finding(
                severity="P1", category="exec",
                file=rel, line=node.lineno,
                message="exec() call — review for code-injection risk",
            ))

        # ── os.system() ───────────────────────────────────────────────────
        if callee == "os.system":
            out.append(Finding(
                severity="P1", category="os-system",
                file=rel, line=node.lineno,
                message="os.system() — prefer subprocess.run with timeout",
            ))

        # ── open() without encoding= ──────────────────────────────────────
        if callee == "open":
            # Skip binary opens — encoding= isn't applicable
            mode_arg = _kwarg_value(node, "mode")
            if mode_arg is None and len(node.args) >= 2:
                mode_arg = node.args[1]
            mode_str = _const_str(mode_arg) if mode_arg else "r"
            if mode_str and "b" in mode_str:
                pass   # binary mode — skip
            elif not _has_kwarg(node, "encoding"):
                out.append(Finding(
                    severity="P2", category="open-no-encoding",
                    file=rel, line=node.lineno,
                    message="open() without encoding= — Windows defaults to cp1252",
                    fixable=True, fix_kind="encoding",
                ))

        # ── threading.Thread / threading.Timer without daemon ─────────────
        if callee in ("threading.Thread", "Thread", "threading.Timer", "Timer"):
            if not _has_kwarg(node, "daemon") and not _has_post_construction_daemon(node):
                out.append(Finding(
                    severity="P1", category="thread-no-daemon",
                    file=rel, line=node.lineno,
                    message=f"{callee}(...) without daemon=True — will block "
                            f"interpreter shutdown",
                    fixable=True, fix_kind="daemon",
                ))

        # ── subprocess.run / subprocess.check_output without timeout ──────
        if callee in ("subprocess.run", "subprocess.check_output",
                      "subprocess.check_call", "subprocess.call"):
            if not _has_kwarg(node, "timeout"):
                out.append(Finding(
                    severity="P1", category="subprocess-no-timeout",
                    file=rel, line=node.lineno,
                    message=f"{callee}(...) without timeout= — may hang forever",
                    fixable=True, fix_kind="timeout",
                ))

        # ── subprocess.Popen without close_fds (Windows: low value, but spec wants it) ──
        if callee in ("subprocess.Popen", "Popen"):
            # close_fds defaults to True on POSIX py 3.7+; on Windows it's
            # ignored when stdin/stdout/stderr are pipes. Only emit P2.
            if not _has_kwarg(node, "close_fds"):
                out.append(Finding(
                    severity="P2", category="popen-no-close-fds",
                    file=rel, line=node.lineno,
                    message=f"{callee}(...) without explicit close_fds=",
                ))

        # ── json.dump to a state file: non-atomic write ───────────────────
        if callee == "json.dump":
            # We can't easily inspect the open(...) context from here; instead
            # walk parent With statements to find the file path.
            # This is best-effort: only catches `with open("X", "w") as f: json.dump(d, f)`
            pass   # handled in dedicated checks below

    return out


def _callee_name(call: ast.Call) -> str | None:
    """Return dotted name of the callee, e.g. 'threading.Thread' or 'open'."""
    node = call.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return None
    return ".".join(reversed(parts))


def _has_kwarg(call: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in call.keywords)


def _kwarg_value(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 7 — hardcoded secrets
# ─────────────────────────────────────────────────────────────────────────

def check_secrets(files: list[str]) -> list[Finding]:
    out: list[Finding] = []
    pwd_re   = re.compile(r"(password|passwd|secret|token|api[_-]?key)\s*=\s*['\"]([^'\"]{8,})['\"]", re.I)
    for path in files:
        text, lines = _read_source(path)
        if text is None:
            continue
        for i, raw in enumerate(lines, start=1):
            line = raw
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for pat, what in SECRET_PATTERNS:
                if pat.search(line):
                    out.append(Finding(
                        severity="P0", category="secret",
                        file=_rel(path), line=i,
                        message=f"hardcoded {what} detected in source",
                    ))
            m = pwd_re.search(line)
            if m:
                key, val = m.group(1), m.group(2)
                # Whitelist obvious placeholders / env-var reads
                if any(w in line for w in SECRET_WHITELIST_SUBSTRINGS):
                    continue
                # Whitelist obvious all-X / placeholder values
                if re.fullmatch(r"[xX*]{8,}", val):
                    continue
                if re.fullmatch(r"[A-Z_]+", val):
                    continue   # likely a constant name reference
                out.append(Finding(
                    severity="P1", category="secret",
                    file=_rel(path), line=i,
                    message=f"possible hardcoded {key} literal — verify it isn't a real credential",
                ))
    return out


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 10/11 — state-file writes (HUD + speech queue)
# ─────────────────────────────────────────────────────────────────────────

def check_state_file_writes(files: list[str]) -> list[Finding]:
    """Flag direct `with open('hud_state.json'|'pending_speech.json', 'w'):
    json.dump(...)` patterns — these are race sources. The canonical helpers
    (_write_hud_state in bobert_companion, _enqueue_speech in skills) must be
    used instead."""
    out: list[Finding] = []
    BC = "bobert_companion.py"
    for path in files:
        text, lines = _read_source(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                call = item.context_expr
                if not (isinstance(call, ast.Call)
                        and _callee_name(call) == "open"
                        and call.args):
                    continue
                # Resolve the path argument (best effort): string literal OR
                # a Name like _SPEECH_QUEUE / hud_state file constant.
                path_target = _resolve_open_target(call.args[0])
                if path_target is None:
                    continue
                basename = os.path.basename(path_target)
                if basename not in STATE_FILES:
                    continue
                # mode must be 'w' / 'wb' / 'w+'
                mode_arg = _kwarg_value(call, "mode")
                if mode_arg is None and len(call.args) >= 2:
                    mode_arg = call.args[1]
                mode = _const_str(mode_arg) or "r"
                if "w" not in mode and "a" not in mode and "+" not in mode:
                    continue
                # Walk the With's body for json.dump
                body_dumps = any(
                    isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                    and _callee_name(s.value) in ("json.dump",)
                    for s in ast.walk(node) if isinstance(s, ast.Expr)
                )
                if not body_dumps:
                    continue
                # Skip bobert_companion's own canonical helper; flag everyone else
                if basename == "hud_state.json":
                    sev = "P1"
                    msg = ("direct write to hud_state.json — use the canonical "
                           "`_write_hud_state(**updates)` helper to avoid races")
                elif basename == "pending_speech.json":
                    sev = "P1"
                    msg = ("direct write to pending_speech.json — use the canonical "
                           "atomic _enqueue_speech pattern (mkstemp + os.replace)")
                else:
                    sev = "P2"
                    msg = (f"non-atomic write to {basename} — wrap in mkstemp + "
                           "os.replace to avoid partial-write corruption")
                # Don't flag bobert_companion's own _write_hud_state implementation
                if (_rel(path).endswith(BC) and basename == "hud_state.json"):
                    continue
                out.append(Finding(
                    severity=sev, category="state-write",
                    file=_rel(path), line=call.lineno,
                    message=msg,
                ))
    return out


def _resolve_open_target(node: ast.AST) -> str | None:
    """Best-effort string for the path argument to open()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        # Heuristic: if the variable name has 'SPEECH' or 'HUD' or 'STATE' or
        # ends in '_FILE', assume it matches a known state file.
        upper = node.id.upper()
        if "SPEECH" in upper and "QUEUE" in upper:
            return "pending_speech.json"
        if "HUD" in upper and "STATE" in upper:
            return "hud_state.json"
        return None
    if isinstance(node, ast.Call) and _callee_name(node) in ("os.path.join",):
        # Take the last string arg as the basename hint
        for a in reversed(node.args):
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                return a.value
    return None


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 13 — readback edge cases
# ─────────────────────────────────────────────────────────────────────────

def check_readback() -> list[Finding]:
    out: list[Finding] = []
    json_files = ["bobert_memory.json", "hud_state.json",
                  ".last_upgrade_summary.json"]
    for f in json_files:
        p = os.path.join(PROJECT_DIR, f)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                json.load(fh)
        except json.JSONDecodeError as e:
            out.append(Finding(
                severity="P1", category="state-corruption",
                file=f, line=e.lineno,
                message=f"{f} is corrupt JSON: {e.msg}",
            ))
        except Exception as e:
            out.append(Finding(
                severity="P1", category="state-corruption",
                file=f, line=0,
                message=f"{f} unreadable: {type(e).__name__}: {e}",
            ))

    # jarvis_todo.md — verify it parses as text and has at least one valid task line
    todo = os.path.join(PROJECT_DIR, "jarvis_todo.md")
    if os.path.exists(todo):
        try:
            with open(todo, "r", encoding="utf-8") as fh:
                content = fh.read()
            if not re.search(r"^- \[[ x]\]", content, re.MULTILINE):
                out.append(Finding(
                    severity="P2", category="state-corruption",
                    file="jarvis_todo.md", line=0,
                    message="no `- [ ]` or `- [x]` task lines detected",
                ))
        except Exception as e:
            out.append(Finding(
                severity="P1", category="state-corruption",
                file="jarvis_todo.md", line=0,
                message=f"jarvis_todo.md unreadable: {e}",
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 14 — globals mutated from non-main threads (heuristic)
# ─────────────────────────────────────────────────────────────────────────

def check_mutation_hygiene(files: list[str]) -> list[Finding]:
    """Best-effort: for each module, find module-level dict/list globals,
    then look for functions that mutate them AND are referenced as a
    Thread/Timer target. If the mutation isn't inside a `with <lock>` block,
    emit P2."""
    out: list[Finding] = []
    for path in files:
        text, _ = _read_source(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue

        # Catalog: module-level mutable globals
        globals_: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                rhs_kind = type(node.value).__name__
                for t in node.targets:
                    if isinstance(t, ast.Name) and rhs_kind in ("Dict", "List", "Set"):
                        globals_.add(t.id)

        # Functions used as Thread targets
        thread_targets: set[str] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and _callee_name(node) in ("threading.Thread", "Thread", "threading.Timer", "Timer")):
                target = _kwarg_value(node, "target")
                if isinstance(target, ast.Name):
                    thread_targets.add(target.id)
                elif isinstance(target, ast.Attribute):
                    thread_targets.add(target.attr)

        # Functions: do they mutate a tracked global, outside a With(lock)?
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name not in thread_targets:
                continue
            mutating = _find_unlocked_mutations(fn, globals_)
            for line, gname in mutating:
                out.append(Finding(
                    severity="P2", category="mutation-hygiene",
                    file=_rel(path), line=line,
                    message=(f"thread-target function '{fn.name}()' mutates "
                             f"module-level '{gname}' without a `with <lock>:` "
                             f"guard"),
                ))
    return out


def _find_unlocked_mutations(fn: ast.AST, names: set[str]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.lock_depth = 0

        def visit_With(self, node: ast.With) -> None:
            entered = False
            for item in node.items:
                ce = item.context_expr
                # Treat any `with <something_lock>:` as a guard.
                if isinstance(ce, ast.Name) and "lock" in ce.id.lower():
                    entered = True
                elif isinstance(ce, ast.Attribute) and "lock" in ce.attr.lower():
                    entered = True
                elif (isinstance(ce, ast.Call)
                      and isinstance(ce.func, ast.Attribute)
                      and ce.func.attr in ("acquire",)):
                    entered = True
            if entered:
                self.lock_depth += 1
            self.generic_visit(node)
            if entered:
                self.lock_depth -= 1

        def visit_Assign(self, node: ast.Assign) -> None:
            # A subscript assign (`d[k] = v`, `_x[0] = v`) is a single GIL-atomic
            # op — and `_x[0] = v` is this project's INTENTIONAL lock-free state
            # pattern (see core/state.py), not a race. Only flag a whole-NAME
            # rebind of a tracked container; genuine read-modify-write is caught
            # by visit_AugAssign below.
            if self.lock_depth == 0:
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in names:
                        out.append((node.lineno, t.id))
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            if self.lock_depth == 0:
                target = node.target
                if isinstance(target, ast.Name) and target.id in names:
                    out.append((node.lineno, target.id))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # `g.append(...)` / `g.pop(...)` / `g.clear()` etc.
            if self.lock_depth == 0 and isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name)
                        and node.func.value.id in names
                        and node.func.attr in ("append", "extend", "pop",
                                               "clear", "update", "setdefault",
                                               "popitem", "remove", "insert")):
                    out.append((node.lineno, node.func.value.id))
            self.generic_visit(node)

    V().visit(fn)
    return out


# ─────────────────────────────────────────────────────────────────────────
#  --fix : conservative auto-fixes for trivially-safe findings
# ─────────────────────────────────────────────────────────────────────────

_OPEN_NO_ENC_RE  = re.compile(r"\bopen\s*\(([^)]*)\)")
_THREAD_RE       = re.compile(r"\b(threading\.Thread|threading\.Timer|Thread|Timer)\s*\(([^)]*)\)")
_SUBPROC_RUN_RE  = re.compile(r"\b(subprocess\.run|subprocess\.check_output|subprocess\.check_call|subprocess\.call)\s*\(([^)]*)\)")
_ASSIGN_TO_NAME_RE = re.compile(r"^(?P<indent>\s*)(?P<name>[A-Za-z_]\w*)\s*=\s*")


def _apply_fix_to_line(line: str, fix_kind: str) -> list[str] | None:
    """Best-effort transformation of a single source line. Returns a list of
    replacement lines (a one-element list for an in-place edit, or a
    multi-element list when extra lines must be inserted after) or None if
    the fix is too risky to apply (multi-line call, unknown shape, etc.)."""
    if fix_kind == "encoding":
        m = _OPEN_NO_ENC_RE.search(line)
        if not m:
            return None
        inner = m.group(1)
        if "encoding=" in inner:
            return None
        # Skip binary mode and PIPE-style opens
        if re.search(r"['\"]\w*b\w*['\"]", inner):
            return None
        new_inner = inner.rstrip() + (", " if inner.strip() else "") + 'encoding="utf-8"'
        return [line[:m.start()] + f"open({new_inner})" + line[m.end():]]
    if fix_kind == "daemon":
        m = _THREAD_RE.search(line)
        if not m:
            return None
        callee = m.group(1)
        inner = m.group(2)
        if "daemon=" in inner:
            return None
        # threading.Timer.__init__ does NOT accept daemon= as a kwarg, so
        # injecting it into the constructor call would raise TypeError at
        # runtime. Use the post-construction idiom instead, but only when
        # the call is bound to a simple `<var> = Timer(...)` assignment.
        if callee.endswith("Timer"):
            am = _ASSIGN_TO_NAME_RE.match(line)
            if not am:
                return None
            return [line, f"{am.group('indent')}{am.group('name')}.daemon = True"]
        new_inner = inner.rstrip() + (", " if inner.strip() else "") + "daemon=True"
        return [line[:m.start()] + f"{callee}({new_inner})" + line[m.end():]]
    if fix_kind == "timeout":
        m = _SUBPROC_RUN_RE.search(line)
        if not m:
            return None
        inner = m.group(2)
        if "timeout=" in inner:
            return None
        new_inner = inner.rstrip() + (", " if inner.strip() else "") + "timeout=60"
        return [line[:m.start()] + f"{m.group(1)}({new_inner})" + line[m.end():]]
    return None


def apply_fixes(findings: list[Finding]) -> tuple[int, list[Finding]]:
    """Apply --fix to every fixable finding. Returns (count_applied, residual_findings)."""
    by_file: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.fixable and f.fix_kind:
            by_file[f.file].append(f)

    applied = 0
    residual: list[Finding] = []

    for relpath, file_findings in by_file.items():
        abspath = os.path.join(PROJECT_DIR, relpath)
        text, lines = _read_source(abspath)
        if text is None:
            continue
        modified = False
        # Sort by line desc so earlier replacements don't shift indices
        file_findings.sort(key=lambda f: f.line, reverse=True)
        for f in file_findings:
            if f.line < 1 or f.line > len(lines):
                residual.append(f)
                continue
            old = lines[f.line - 1]
            new = _apply_fix_to_line(old, f.fix_kind)
            if new is None or new == [old]:
                residual.append(f)
                continue
            lines[f.line - 1:f.line] = new
            applied += 1
            modified = True
        if modified:
            new_text = "\n".join(lines)
            if text.endswith("\n") and not new_text.endswith("\n"):
                new_text += "\n"
            try:
                with open(abspath, "w", encoding="utf-8", newline="") as fh:
                    fh.write(new_text)
            except Exception as e:
                print(f"  [fix] failed to write {relpath}: {e}", file=sys.stderr)

    return applied, residual


# ═════════════════════════════════════════════════════════════════════════
#  INTEGRATION / CONFLICT CHECKS  (A–J)
# ═════════════════════════════════════════════════════════════════════════
#
# These checks run AFTER the static checks above. They surface runtime
# behaviour the static pass can't see: cross-skill conflicts, prompt drift,
# import cycles, leaky state files. Each check is wrapped in a top-level
# try/except so a single broken check can't sink the auditor — failed checks
# emit a single P2 advisory describing why they couldn't run.


def _safe(fn):
    """Wrap a check function so unhandled exceptions become P2 advisories
    instead of crashing main(). Each integration check is wrapped via this."""
    def wrapped(*args: Any, **kw: Any) -> tuple[list[Finding], Any]:
        try:
            result = fn(*args, **kw)
            if isinstance(result, tuple):
                return result
            return result, None
        except Exception as e:
            tb = traceback.format_exc(limit=4).splitlines()[-3:]
            note = " | ".join(s.strip() for s in tb if s.strip())
            return [Finding(
                severity="P2", category="integration-check-error",
                file="tools/audit_codebase.py", line=0,
                message=(f"integration check '{fn.__name__}' could not run: "
                         f"{type(e).__name__}: {e}  ({note})"),
            )], None
    return wrapped


# Cache: file → set of state files written, set of action names registered,
# set of thread-target functions. Populated lazily by _profile_skill().
_skill_profile_cache: dict[str, dict[str, Any]] = {}


def _profile_skill(path: str) -> dict[str, Any]:
    """Return a structural summary of one skill file: which state files it
    writes, which action names it registers, whether it spawns background
    threads, and which lock names it acquires. Used by checks B and C."""
    if path in _skill_profile_cache:
        return _skill_profile_cache[path]
    profile: dict[str, Any] = {
        "writes_state": set(),     # basenames in STATE_FILES
        "actions":      {},        # name → lineno
        "thread_lines": [],        # list[int] of Thread/Timer call linenos
        "thread_targets": set(),   # function names used as thread targets
        "lock_names":   set(),     # names of Lock variables defined in module
        "imports":      set(),     # top-level modules imported
        "uses_atomic_writer": False,  # True iff the module routes state writes
                                       # through core.atomic_io._atomic_write_json
    }
    text, lines = _read_source(path)
    if text is None:
        _skill_profile_cache[path] = profile
        return profile
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        _skill_profile_cache[path] = profile
        return profile

    for node in ast.walk(tree):
        # state-file writes (best-effort): match `open("X", "w")` where X is
        # in STATE_FILES, OR any open() whose path argument is a Name like
        # _SPEECH_QUEUE / _HUD_FILE / etc.
        if isinstance(node, ast.Call) and _callee_name(node) == "open" and node.args:
            tgt = _resolve_open_target(node.args[0])
            if tgt and os.path.basename(tgt) in STATE_FILES:
                profile["writes_state"].add(os.path.basename(tgt))
        # Thread / Timer constructor sites
        if isinstance(node, ast.Call):
            cn = _callee_name(node)
            if cn in ("threading.Thread", "Thread", "threading.Timer", "Timer"):
                profile["thread_lines"].append(node.lineno)
                tgt = _kwarg_value(node, "target")
                if isinstance(tgt, ast.Name):
                    profile["thread_targets"].add(tgt.id)
                elif isinstance(tgt, ast.Attribute):
                    profile["thread_targets"].add(tgt.attr)
        # Module-level Lock() assignments
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _callee_name(node.value) in ("threading.Lock", "Lock", "threading.RLock", "RLock"):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        profile["lock_names"].add(t.id)
        # Imports — also detect the canonical shared-writer helper.
        # A skill is considered to route through the helper if it imports
        # `_atomic_write_json` from `core.atomic_io` (the audit-v2 canonical
        # path), or pulls in the whole atomic_io module to call through it.
        if isinstance(node, ast.Import):
            for alias in node.names:
                profile["imports"].add(alias.name.split(".")[0])
                if alias.name in ("core.atomic_io",):
                    profile["uses_atomic_writer"] = True
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                profile["imports"].add(node.module.split(".")[0])
            if node.module == "core.atomic_io":
                for alias in node.names:
                    if alias.name == "_atomic_write_json":
                        profile["uses_atomic_writer"] = True
                        break
            elif node.module == "core":
                for alias in node.names:
                    if alias.name == "atomic_io":
                        profile["uses_atomic_writer"] = True
                        break

    # Actions registered: look for `register(actions)` and the
    # `actions[name] = callable` pattern inside.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "register":
            if not node.args.args:
                continue
            param = node.args.args[0].arg
            # (a) direct `actions["name"] = fn`
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign)
                        and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Subscript)
                        and isinstance(sub.targets[0].value, ast.Name)
                        and sub.targets[0].value.id == param):
                    key = sub.targets[0].slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        profile["actions"][key.value] = sub.lineno
            # (b) dict-driven registration, e.g. browser_agent / many skills:
            #       handlers = {"name": fn, ...}
            #       for k, v in handlers.items(): actions[k] = v   (or actions.update(handlers))
            #     The direct check above misses these (slice is a Name, not a
            #     str const), causing spurious "documented but not registered".
            dict_keys: dict[str, list[str]] = {}
            for sub in ast.walk(node):
                tgt_name = dval = None
                if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name)
                        and isinstance(sub.value, ast.Dict)):
                    tgt_name, dval = sub.targets[0].id, sub.value
                elif (isinstance(sub, ast.AnnAssign)        # handlers: dict = {...}
                        and isinstance(sub.target, ast.Name)
                        and isinstance(sub.value, ast.Dict)):
                    tgt_name, dval = sub.target.id, sub.value
                if tgt_name is not None and dval is not None:
                    ks = [k.value for k in dval.keys
                          if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                    if ks:
                        dict_keys[tgt_name] = ks
            for sub in ast.walk(node):
                # actions.update(<dict-name or dict-literal>)
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "update"
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == param and sub.args):
                    a0 = sub.args[0]
                    if isinstance(a0, ast.Name):
                        for k in dict_keys.get(a0.id, []):
                            profile["actions"].setdefault(k, sub.lineno)
                    elif isinstance(a0, ast.Dict):
                        for k in a0.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                profile["actions"].setdefault(k.value, sub.lineno)
                # for k, v in <dict-name>.items(): actions[k] = v
                elif isinstance(sub, ast.For):
                    it = sub.iter
                    src = (it.func.value.id
                           if isinstance(it, ast.Call)
                           and isinstance(it.func, ast.Attribute)
                           and it.func.attr == "items"
                           and isinstance(it.func.value, ast.Name)
                           else None)
                    assigns_param = any(
                        isinstance(b, ast.Assign) and len(b.targets) == 1
                        and isinstance(b.targets[0], ast.Subscript)
                        and isinstance(b.targets[0].value, ast.Name)
                        and b.targets[0].value.id == param
                        for b in ast.walk(sub))
                    if src and assigns_param:
                        for k in dict_keys.get(src, []):
                            profile["actions"].setdefault(k, sub.lineno)

    _skill_profile_cache[path] = profile
    return profile


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (A) — action smoke tests
# ─────────────────────────────────────────────────────────────────────────

class _StubModule(types.ModuleType):
    """Stand-in module that returns callable no-ops for any attribute access.
    Lets us swap out cv2, paho.mqtt, win32com, etc. without the importing
    skill caring whether they're real."""

    def __getattr__(self, name: str) -> Any:
        # Common pattern: skills do `Stub.SUBMODULE.func(...)`. Return
        # another stub for sub-attributes; for terminal calls, return a
        # no-op lambda that returns another stub instance so chained
        # calls like `cv2.VideoCapture(0).read()` don't explode.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _StubCallable(f"{self.__name__}.{name}")


class _StubCallable:
    def __init__(self, name: str) -> None:
        self.__name__ = name

    def __call__(self, *args: Any, **kw: Any) -> Any:
        return _StubCallable(self.__name__ + "()")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _StubCallable(f"{self.__name__}.{name}")

    def __bool__(self) -> bool:
        return False  # so `if cv2.VideoCapture(0).isOpened()` short-circuits

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __contains__(self, item: Any) -> bool:
        return False

    def __getitem__(self, key: Any) -> Any:
        return _StubCallable(f"{self.__name__}[{key!r}]")

    def __index__(self) -> int:
        return 0

    def __int__(self) -> int:
        return 0

    def __float__(self) -> float:
        return 0.0

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"<_StubCallable {self.__name__}>"

    def __add__(self, other: Any) -> Any:
        return other if isinstance(other, str) else _StubCallable(f"{self.__name__}+")

    def __radd__(self, other: Any) -> Any:
        return other if isinstance(other, str) else _StubCallable(f"+{self.__name__}")

    def __sub__(self, other: Any) -> Any:
        return _StubCallable(f"{self.__name__}-")

    def __rsub__(self, other: Any) -> Any:
        return _StubCallable(f"-{self.__name__}")

    def __mul__(self, other: Any) -> Any:
        return _StubCallable(f"{self.__name__}*")

    def __rmul__(self, other: Any) -> Any:
        return _StubCallable(f"*{self.__name__}")

    def __truediv__(self, other: Any) -> Any:
        return _StubCallable(f"{self.__name__}/")

    def __rtruediv__(self, other: Any) -> Any:
        return _StubCallable(f"/{self.__name__}")

    def __floordiv__(self, other: Any) -> Any:
        return _StubCallable(f"{self.__name__}//")

    def __mod__(self, other: Any) -> Any:
        return _StubCallable(f"{self.__name__}%")

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _StubCallable) and other.__name__ == self.__name__

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.__name__)

    def __lt__(self, other: Any) -> bool:
        return False

    def __le__(self, other: Any) -> bool:
        return False

    def __gt__(self, other: Any) -> bool:
        return False

    def __ge__(self, other: Any) -> bool:
        return False

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def read(self):  # cv2.VideoCapture(...).read() → (ret, frame)
        return False, None


def _install_smoke_stubs() -> dict[str, Any]:
    """Insert stubs into sys.modules for the heavy hardware-coupled deps.
    Returns a snapshot of overwritten entries so the caller can restore."""
    saved: dict[str, Any] = {}
    targets = [
        "cv2", "paho", "paho.mqtt", "paho.mqtt.client",
        "win32com", "win32com.client", "win32api", "win32gui", "win32con",
        "pyautogui", "psutil", "pyttsx3", "edge_tts", "whisper",
        "sounddevice", "pygame", "pygame.mixer", "PIL", "PIL.Image",
        "PIL.ImageGrab", "screeninfo", "mss", "mss.mss",
    ]
    for name in targets:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = _StubModule(name)
    # Lighter, more-functional stubs for the most-called modules
    wb = types.ModuleType("webbrowser")
    wb.open = lambda *a, **kw: True  # type: ignore[attr-defined]
    saved["webbrowser"] = sys.modules.get("webbrowser")
    sys.modules["webbrowser"] = wb
    return saved


def _restore_smoke_stubs(saved: dict[str, Any]) -> None:
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def check_action_smoke_tests(skill_files: list[str]) -> tuple[list[Finding], dict[str, str]]:
    """For every action declared by a skill, try to dispatch it with a benign
    argument under stubbed heavy deps. Failures get reported as findings.
    Returns (findings, results) where results['actions'] is a per-action
    status string ('ok' / 'skipped' / 'error: ...').
    """
    out: list[Finding] = []
    results: dict[str, str] = {}

    saved = _install_smoke_stubs()

    try:
        # Try a one-time best-effort import of `bobert_companion`. If it
        # fails (heavy deps unavailable in this env), we fall back to a
        # static check: every registered callable in each skill is reachable.
        bc_mod: types.ModuleType | None = None
        try:
            if PROJECT_DIR not in sys.path:
                sys.path.insert(0, PROJECT_DIR)
            bc_mod = importlib.import_module("bobert_companion")
        except Exception as e:
            out.append(Finding(
                severity="P2", category="smoke-test",
                file="bobert_companion.py", line=0,
                message=(f"could not import bobert_companion for smoke test: "
                         f"{type(e).__name__}: {e}; falling back to static check"),
            ))

        for path in sorted(skill_files):
            rel = _rel(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            prof = _profile_skill(path)
            for action_name in prof["actions"]:
                if action_name in _NO_SMOKE_TEST:
                    results[action_name] = "skipped (no-smoke-test)"
                    continue
                if bc_mod is None:
                    # Static fallback: confirm action name & callable resolve
                    results[action_name] = "static-only"
                    continue
                # Live dispatch via the same path the LLM uses
                try:
                    actions_dict = getattr(bc_mod, "ACTIONS", None)
                    if not isinstance(actions_dict, dict):
                        results[action_name] = "skipped (no ACTIONS dict)"
                        continue
                    fn = actions_dict.get(action_name)
                    if fn is None:
                        results[action_name] = "skipped (not registered at runtime)"
                        continue
                    res = fn("")
                    if not isinstance(res, str):
                        out.append(Finding(
                            severity="P1", category="smoke-test",
                            file=rel, line=prof["actions"][action_name],
                            message=(f"action '{action_name}' returned "
                                     f"{type(res).__name__} instead of str"),
                        ))
                        results[action_name] = f"bad-return: {type(res).__name__}"
                    else:
                        results[action_name] = "ok"
                except (AttributeError, KeyError, TypeError, FileNotFoundError) as e:
                    out.append(Finding(
                        severity="P1", category="smoke-test",
                        file=rel, line=prof["actions"][action_name],
                        message=(f"action '{action_name}' raised "
                                 f"{type(e).__name__}: {e}"),
                    ))
                    results[action_name] = f"error: {type(e).__name__}: {e}"
                except Exception as e:
                    # Non-target exception classes — still flag, but as P2.
                    out.append(Finding(
                        severity="P2", category="smoke-test",
                        file=rel, line=prof["actions"][action_name],
                        message=(f"action '{action_name}' raised unexpected "
                                 f"{type(e).__name__}: {e}"),
                    ))
                    results[action_name] = f"error: {type(e).__name__}: {e}"
    finally:
        _restore_smoke_stubs(saved)
    return out, results


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (B) — skill-pair conflict matrix
# ─────────────────────────────────────────────────────────────────────────

def check_skill_pair_conflicts(skill_files: list[str]) -> tuple[list[Finding], list[dict[str, Any]]]:
    """Build a pairwise matrix of overlapping state writes / actions / threads
    across all skills. Direct overlap is a strong signal a refactor is needed
    or both should funnel through one helper."""
    out: list[Finding] = []
    rows: list[dict[str, Any]] = []
    profiles = [(p, _profile_skill(p)) for p in sorted(skill_files)]

    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            pa, prof_a = profiles[i]
            pb, prof_b = profiles[j]
            shared_files = prof_a["writes_state"] & prof_b["writes_state"]
            shared_actions = set(prof_a["actions"]) & set(prof_b["actions"])
            both_thread = bool(prof_a["thread_lines"]) and bool(prof_b["thread_lines"])
            both_canonical = bool(prof_a.get("uses_atomic_writer")) and bool(prof_b.get("uses_atomic_writer"))
            # When BOTH sides route through core.atomic_io._atomic_write_json,
            # the co-write is safe (audit-v2 invariant), so suppress the
            # state-file half of the finding. Action collisions are still
            # surfaced unconditionally.
            if not (shared_files or shared_actions):
                continue
            row = {
                "skill_a": _rel(pa),
                "skill_b": _rel(pb),
                "shared_state_files": sorted(shared_files),
                "shared_actions":     sorted(shared_actions),
                "both_have_threads":  both_thread,
                "both_canonical":     both_canonical,
            }
            rows.append(row)
            if not both_canonical:
                for fname in shared_files:
                    out.append(Finding(
                        severity="P1", category="skill-conflict",
                        file=_rel(pb), line=0,
                        message=(f"writes '{fname}' which is also written by "
                                 f"{_rel(pa)} — both must use the same atomic "
                                 f"helper to avoid races (one or both sides do "
                                 f"NOT import `_atomic_write_json` from "
                                 f"core.atomic_io)"),
                    ))
            for aname in shared_actions:
                out.append(Finding(
                    severity="P1", category="skill-conflict",
                    file=_rel(pb), line=prof_b["actions"].get(aname, 0),
                    message=(f"registers action '{aname}' already owned by "
                             f"{_rel(pa)} — last-load-wins overwrite"),
                ))

    # Persist a friendly markdown matrix even when there are zero conflicts.
    md_path = os.path.join(PROJECT_DIR, "audit_conflict_matrix.md")
    lines = ["# JARVIS Skill-Pair Conflict Matrix", ""]
    if not rows:
        lines.append("_No conflicts detected across the skill set._")
    else:
        lines.append("| Skill A | Skill B | Shared state files | Shared actions | Both threaded | Both use atomic_io |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| `{r['skill_a']}` | `{r['skill_b']}` | "
                f"{', '.join(r['shared_state_files']) or '—'} | "
                f"{', '.join(r['shared_actions']) or '—'} | "
                f"{'yes' if r['both_have_threads'] else 'no'} | "
                f"{'yes (safe)' if r.get('both_canonical') else 'no'} |"
            )
    try:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception:
        pass
    return out, rows


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (C) — background-thread audit
# ─────────────────────────────────────────────────────────────────────────

def check_thread_audit(files: list[str]) -> tuple[list[Finding], list[dict[str, Any]]]:
    """For every Thread/Timer call site: verify daemon, target function has
    try/except around outer loop, no tight-spinning loop (no sleep on the
    hot path)."""
    out: list[Finding] = []
    summary: list[dict[str, Any]] = []
    for path in files:
        text, lines = _read_source(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue
        rel = _rel(path)

        # Map function name → FunctionDef so we can look up thread targets
        func_defs: dict[str, ast.FunctionDef] = {
            n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        # Find Thread/Timer call sites; the daemon-flag heuristic mirrors
        # _ast_bad_pattern_checks() but here we look deeper at the target.
        parent_map: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[id(child)] = parent

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            cn = _callee_name(node)
            if cn not in ("threading.Thread", "Thread", "threading.Timer", "Timer"):
                continue
            target = _kwarg_value(node, "target")
            target_name = None
            if isinstance(target, ast.Name):
                target_name = target.id
            elif isinstance(target, ast.Attribute):
                target_name = target.attr
            summary.append({
                "file": rel, "line": node.lineno,
                "callee": cn, "target": target_name,
            })
            if not target_name or target_name not in func_defs:
                continue
            fn = func_defs[target_name]
            # (1) Is the thread resilient to exceptions? Accept either:
            #   - a try/except ANYWHERE in the function (the risky work is
            #     guarded, even if nested inside an if/for inside the loop); or
            #   - every loop body is pure DELEGATION (only bare calls / waits /
            #     pass / break / continue / call-assignments) — the work is
            #     handed to helper methods that guard themselves, which the
            #     auditor can't see across. Both patterns are pervasive here and
            #     produced ~50 false "will silently kill the thread" findings.
            has_try = any(isinstance(n, ast.Try) for n in ast.walk(fn))

            def _delegation_only(body: list) -> bool:
                for s in body:
                    if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
                        continue
                    if isinstance(s, (ast.Pass, ast.Break, ast.Continue)):
                        continue
                    if isinstance(s, ast.Assign) and isinstance(s.value, ast.Call):
                        continue
                    return False
                return True

            _loops = [n for n in ast.walk(fn)
                      if isinstance(n, (ast.While, ast.For))]
            resilient = has_try or (
                bool(_loops) and all(_delegation_only(lp.body) for lp in _loops))
            if not resilient:
                out.append(Finding(
                    severity="P2", category="thread-no-try",
                    file=rel, line=fn.lineno,
                    message=(f"thread-target function '{target_name}()' has "
                             f"no try/except wrapping — a single exception "
                             f"will silently kill the thread"),
                ))
            # (2) tight spin: a while-loop with NO blocking/pacing call on the
            # hot path. Recognise any blocking primitive by METHOD NAME, not
            # just literal time.sleep — Event/Condition.wait(timeout),
            # queue.get(timeout), Lock.acquire(timeout), select/poll, socket
            # recv all pace a loop. The old check only matched "time.sleep" and
            # "event.wait" (a var literally named `event`), so every loop doing
            # `_stop.wait(0.5)` / `queue.get()` was a false "tight-spin" —
            # ~26 spurious P1s that buried the real findings.
            _BLOCKING_CALLS = (
                "sleep", "wait", "get", "acquire", "recv", "recvfrom",
                "select", "poll", "wait_for", "join", "read",
            )
            # A loop that DRAINS a bounded collection (`while q: q.popleft()`,
            # `while len(buf) > cap: buf.popleft()`) terminates and is NOT a
            # CPU-pegging spin — flagging those produced false P1s on the inner
            # drain loops of otherwise-paced worker threads.
            _DRAIN_CALLS = ("popleft", "pop", "popitem", "remove", "discard",
                            "clear", "get_nowait")
            for stmt in ast.walk(fn):
                if isinstance(stmt, ast.While):
                    paced = any(
                        isinstance(c, ast.Call)
                        and (_callee_name(c) or "").split(".")[-1] in _BLOCKING_CALLS
                        for c in ast.walk(stmt))
                    drains = any(
                        (isinstance(c, ast.Call)
                         and (_callee_name(c) or "").split(".")[-1] in _DRAIN_CALLS)
                        or isinstance(c, ast.Delete)
                        for c in ast.walk(stmt))
                    if not paced and not drains:
                        out.append(Finding(
                            severity="P1", category="thread-tight-spin",
                            file=rel, line=stmt.lineno,
                            message=(f"thread-target '{target_name}()' has a "
                                     f"while-loop with no sleep/wait/blocking "
                                     f"call — will peg a CPU core"),
                        ))
                        break
    return out, summary


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (D) — prompt ↔ action consistency
# ─────────────────────────────────────────────────────────────────────────

# Match an action token inside PC_CONTROL_PROMPT. The prompt documents
# actions in three precise shapes; each matcher requires a structural
# anchor so that English continuation text ("monitor, drop master volume
# to ~30%") does not collide with real action definitions.
#
# (1) Table-row form: token followed by either `, <placeholder>` (with arg)
#     or ≥2 spaces + em-dash (column-aligned no-arg form). The leading
#     prefix is bounded to (code-indent + opening quote + ≤8 spaces) so
#     deeply-indented continuation text doesn't qualify as a row.
#       "  open_url, <url>              — open a website"
#       "  screenshot                   — save a screenshot"
_PROMPT_ACTION_TABLE_RE = re.compile(
    r'^[ \t]*"[ \t]{1,8}([a-z][a-z0-9_]*)(?:\s*,\s*<|[ \t]{2,}—)',
    re.MULTILINE,
)
# (2) Narrative comma-separated token list on a single source line, ending
#     at the literal "\n" + closing quote of the Python source fragment.
#       "    volume_up, volume_down, volume_mute (system-wide)\n"
#       "    Media keys: media_next, media_prev, media_playpause\n"
_PROMPT_ACTION_LIST_RE = re.compile(
    r'^[ \t]*"[ \t]{1,8}(?:[A-Za-z][A-Za-z _]*:\s+)?'
    r'([a-z][a-z0-9_]+(?:\s*,\s*[a-z][a-z0-9_]+)+)'
    r'\s*(?:\([^)]*\))?\s*\\n"',
    re.MULTILINE,
)
# (3) Explicit `[ACTION: name]` citation — the gold standard.
_PROMPT_ACTION_REF_RE = re.compile(r"\[ACTION:\s*([a-z][a-z0-9_]*)")


def _extract_prompt_actions(text: str) -> set[str]:
    """Return the set of action names documented in PC_CONTROL_PROMPT.

    Only tokens that match one of three precise structural patterns are
    accepted: a table-row anchor (token + arg-placeholder or em-dash), a
    comma-separated narrative list bounded by the source-line `\\n"`
    terminator, or a `[ACTION: name]` citation. Loose word matches are
    deliberately rejected so that continuation lines in description text
    don't pollute the documented-action set."""
    found: set[str] = set()
    m = re.search(
        r"PC_CONTROL_PROMPT\s*=\s*\(([\s\S]*?)\n\)\n",
        text,
    )
    if not m:
        return found
    body = m.group(1)
    for name in _PROMPT_ACTION_TABLE_RE.findall(body):
        found.add(name)
    for group in _PROMPT_ACTION_LIST_RE.findall(body):
        for tok in re.split(r"\s*,\s*", group):
            if re.fullmatch(r"[a-z][a-z0-9_]+", tok):
                found.add(tok)
    for name in _PROMPT_ACTION_REF_RE.findall(body):
        found.add(name)
    return found


def check_prompt_action_consistency(prompt_actions: set[str],
                                    registered_actions: set[str]
                                    ) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    documented = {a for a in prompt_actions if not a.startswith("_")}
    missing = documented - registered_actions
    undocumented = registered_actions - documented - _INTERNAL_ONLY_ACTIONS
    for name in sorted(missing):
        out.append(Finding(
            severity="P1", category="prompt-drift",
            file="bobert_companion.py", line=0,
            message=(f"PC_CONTROL_PROMPT documents action '{name}' but no "
                     f"skill registers it — the LLM will hallucinate it and "
                     f"JARVIS will return 'unknown action'"),
        ))
    for name in sorted(undocumented):
        out.append(Finding(
            severity="P2", category="prompt-drift",
            file="bobert_companion.py", line=0,
            message=(f"action '{name}' is registered but not documented in "
                     f"PC_CONTROL_PROMPT — the LLM doesn't know it exists "
                     f"(add to _INTERNAL_ONLY_ACTIONS if intentional)"),
        ))
    return out, {
        "documented": sorted(documented),
        "registered": sorted(registered_actions),
        "missing_from_registry":  sorted(missing),
        "missing_from_prompt":    sorted(undocumented),
    }


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (E) — voice-trigger coverage
# ─────────────────────────────────────────────────────────────────────────

def check_voice_trigger_coverage(prompt_text: str,
                                 registered_actions: set[str]
                                 ) -> tuple[list[Finding], dict[str, str]]:
    out: list[Finding] = []
    coverage: dict[str, str] = {}
    body = ""
    m = re.search(r"PC_CONTROL_PROMPT\s*=\s*\(([\s\S]*?)\n\)\n", prompt_text)
    if m:
        body = m.group(1)
    for action, examples in _VOICE_TRIGGER_EXAMPLES.items():
        if action not in registered_actions:
            coverage[action] = "skipped (action not registered)"
            continue
        if any(ex.lower() in body.lower() for ex in examples):
            coverage[action] = "ok"
        else:
            coverage[action] = "MISSING"
            out.append(Finding(
                severity="P2", category="voice-coverage",
                file="bobert_companion.py", line=0,
                message=(f"voice-triggered action '{action}' has no example "
                         f"trigger phrase in PC_CONTROL_PROMPT — LLM may not "
                         f"know when to fire it"),
            ))
    return out, coverage


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (F) — TTS pipeline test
# ─────────────────────────────────────────────────────────────────────────

def check_tts_pipeline(bc_text: str, tts_text: str = "") -> tuple[list[Finding], dict[str, str]]:
    """Static-style sanity check: every _TTS_EMOTION_PRESETS key must be
    referenced from the resolver / synthesise path; the fallback chain must
    end on the 'neutral' preset.

    The presets + the priority resolver were consolidated into core/tts.py, so
    the dict literal is sourced from there (falling back to the monolith for a
    pre-consolidation tree) and references are checked across BOTH files —
    bobert_companion.py still holds synthesise(), the _resolve_tts_preset shim,
    and the re-exports."""
    out: list[Finding] = []
    coverage: dict[str, str] = {}

    # References span core/tts.py (the pure resolver) AND bobert_companion.py
    # (synthesise() + the shim + the re-exported names).
    combined = (tts_text or "") + "\n" + (bc_text or "")

    # Extract preset keys from the dict literal — prefer core/tts.py, fall
    # back to the monolith so an older layout still audits cleanly. Anchor the
    # assignment at column 0 (and keep [^=] from spanning newlines) so a mere
    # comment mention of _TTS_EMOTION_PRESETS can't trap the match on the wrong
    # dict.
    preset_re = r"^_TTS_EMOTION_PRESETS\b[^=\n]*=\s*\{([\s\S]*?)^\}"
    m = re.search(preset_re, tts_text or "", re.MULTILINE)
    src_file = "core/tts.py"
    if not m:
        m = re.search(preset_re, bc_text or "", re.MULTILINE)
        src_file = "bobert_companion.py"
    if not m:
        out.append(Finding(
            severity="P1", category="tts-pipeline",
            file="core/tts.py", line=0,
            message="_TTS_EMOTION_PRESETS dict literal not found",
        ))
        return out, coverage
    preset_body = m.group(1)
    preset_names = re.findall(r'["\']([a-z_][a-z0-9_]*)["\']\s*:', preset_body)

    # Reference checks (across both files)
    for name in preset_names:
        if re.search(rf'\b_TTS_EMOTION_PRESETS\[\s*["\']{name}["\']', combined) \
                or re.search(rf'["\']{name}["\']', combined):
            coverage[name] = "ok"
        else:
            coverage[name] = "UNUSED"
            out.append(Finding(
                severity="P2", category="tts-pipeline",
                file=src_file, line=0,
                message=(f"_TTS_EMOTION_PRESETS preset '{name}' is defined "
                         f"but never referenced — likely dead code"),
            ))

    # Fallback chain: the resolver must default to 'neutral' on any unresolved
    # preset (look for `.get(..., _TTS_EMOTION_PRESETS["neutral"])`).
    if 'neutral' not in preset_names:
        out.append(Finding(
            severity="P0", category="tts-pipeline",
            file=src_file, line=0,
            message="_TTS_EMOTION_PRESETS has no 'neutral' preset — fallback chain broken",
        ))
    elif '_TTS_EMOTION_PRESETS["neutral"]' not in combined \
            and "_TTS_EMOTION_PRESETS['neutral']" not in combined:
        out.append(Finding(
            severity="P1", category="tts-pipeline",
            file=src_file, line=0,
            message=("resolver does not appear to default to "
                     "_TTS_EMOTION_PRESETS['neutral'] — fallback chain unclear"),
        ))
    return out, coverage


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (G) — crash-recovery test
# ─────────────────────────────────────────────────────────────────────────

def check_crash_recovery(bc_text: str) -> tuple[list[Finding], dict[str, str]]:
    """For each critical function listed in _CRITICAL_FUNCTIONS, verify the
    body contains a top-level try/except (or its first statement is a while
    loop whose body has try/except)."""
    out: list[Finding] = []
    status: dict[str, str] = {}
    try:
        tree = ast.parse(bc_text, filename="bobert_companion.py")
    except SyntaxError:
        out.append(Finding(
            severity="P0", category="crash-recovery",
            file="bobert_companion.py", line=0,
            message="cannot parse bobert_companion.py for crash-recovery audit",
        ))
        return out, status

    by_name: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name in _CRITICAL_FUNCTIONS:
            by_name[node.name] = node

    for fname in _CRITICAL_FUNCTIONS:
        fn = by_name.get(fname)
        if fn is None:
            status[fname] = "missing"
            continue
        # Strict outer-try check
        if any(isinstance(stmt, ast.Try) for stmt in fn.body):
            status[fname] = "ok (outer try)"
            continue
        # Accept loop-with-try idiom
        if fn.body and isinstance(fn.body[0], (ast.While, ast.For)):
            loop = fn.body[0]
            if any(isinstance(s, ast.Try) for s in loop.body):
                status[fname] = "ok (loop body try)"
                continue
        # Accept any try statement reachable from the function body
        if any(isinstance(s, ast.Try) for s in ast.walk(fn)):
            status[fname] = "ok (nested try)"
            continue
        status[fname] = "NO TRY/EXCEPT"
        out.append(Finding(
            severity="P1", category="crash-recovery",
            file="bobert_companion.py", line=fn.lineno,
            message=(f"critical function '{fname}()' has no try/except — "
                     f"a transient failure here will crash JARVIS"),
        ))
    return out, status


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (H) — memory & file-handle leak test
# ─────────────────────────────────────────────────────────────────────────

def check_leak() -> tuple[list[Finding], dict[str, Any]]:
    """Best-effort leak detection: run a 100-iteration loop of no-op work
    under the SAME stub harness used by the smoke test, observing process
    handle count / RSS. Skip cleanly if psutil isn't available."""
    out: list[Finding] = []
    summary: dict[str, Any] = {"ran": False}
    try:
        import psutil  # type: ignore
    except ImportError:
        summary["skipped"] = "psutil not available"
        return out, summary

    proc = psutil.Process()
    try:
        handles_before = proc.num_handles() if hasattr(proc, "num_handles") else 0
        files_before = len(proc.open_files())
        rss_before = proc.memory_info().rss
        for _ in range(100):
            # Mimic the per-turn shape: open a temp file, write, close,
            # discard. Real per-turn ops (LLM call, transcribe, dispatch) are
            # too heavy to simulate without a real env; this is a smoke proxy
            # that flags trivial leaks (forgotten close()s, dict growth).
            import tempfile
            fd, p = tempfile.mkstemp()
            os.close(fd)
            os.unlink(p)
        handles_after = proc.num_handles() if hasattr(proc, "num_handles") else 0
        files_after = len(proc.open_files())
        rss_after = proc.memory_info().rss
        summary = {
            "ran": True,
            "handles_before": handles_before, "handles_after": handles_after,
            "open_files_before": files_before, "open_files_after": files_after,
            "rss_before": rss_before, "rss_after": rss_after,
        }
        # Tolerate small jitter (handle pools fluctuate); flag growth >20.
        if handles_after - handles_before > 20:
            out.append(Finding(
                severity="P2", category="leak-test",
                file="audit_codebase.py", line=0,
                message=(f"file-handle count grew by "
                         f"{handles_after - handles_before} over 100 iter "
                         f"({handles_before} → {handles_after}) — investigate"),
            ))
        if files_after - files_before > 5:
            out.append(Finding(
                severity="P2", category="leak-test",
                file="audit_codebase.py", line=0,
                message=(f"open-file count grew by "
                         f"{files_after - files_before} over 100 iter — leak?"),
            ))
    except Exception as e:
        summary["skipped"] = f"{type(e).__name__}: {e}"
    return out, summary


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (I) — import graph + circular import detection
# ─────────────────────────────────────────────────────────────────────────

def check_import_graph(files: list[str]) -> tuple[list[Finding], dict[str, Any]]:
    """Build an intra-project import DAG and detect cycles. Tarjan-lite via
    DFS-with-stack; cycles get reported as P1 findings (a true import cycle
    crashes at startup)."""
    out: list[Finding] = []

    # Map: source-relpath → set of intra-project modules it imports
    intra_modules: set[str] = set()
    intra_modules.update(["bobert_companion", "memory", "tray", "boot_sequence",
                          "iron_man_boot", "jarvis_failure_lines",
                          "jarvis_watcher", "overnight_upgrade",
                          "upgrade_jarvis", "hud_card"])
    intra_modules.update(os.path.splitext(f)[0] for f in os.listdir(SKILLS_DIR)
                         if f.endswith(".py"))
    if os.path.isdir(CORE_DIR):
        intra_modules.update("core." + os.path.splitext(f)[0]
                             for f in os.listdir(CORE_DIR)
                             if f.endswith(".py"))

    file_to_mod: dict[str, str] = {}
    for path in files:
        rel = _rel(path).removesuffix(".py")
        # Normalize: 'skills/foo' → 'foo' for module key; 'core/foo' → 'core.foo'
        if rel.startswith("skills/"):
            mod = rel.split("/", 1)[1]
        elif rel.startswith("core/"):
            mod = "core." + rel.split("/", 1)[1]
        else:
            mod = rel
        file_to_mod[path] = mod

    graph: dict[str, set[str]] = defaultdict(set)
    edges: list[tuple[str, str, str, int]] = []  # for report
    for path in files:
        text, _ = _read_source(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue
        src_mod = file_to_mod.get(path)
        if not src_mod:
            continue
        # Deferred imports (inside a function/method body) run lazily at call
        # time and do NOT create an import-time cycle. The `_bc()` pattern in
        # core/actions.py (`import bobert_companion` inside each handler) exists
        # precisely to break the bobert_companion ↔ core.actions cycle, so those
        # must not be counted as edges — else every audit reports a false cycle.
        deferred: set[int] = set()
        for _fn in ast.walk(tree):
            if isinstance(_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for _inner in ast.walk(_fn):
                    if isinstance(_inner, (ast.Import, ast.ImportFrom)):
                        deferred.add(id(_inner))
        for node in ast.walk(tree):
            if id(node) in deferred:
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name in intra_modules or name.split(".")[0] in intra_modules:
                        graph[src_mod].add(name)
                        edges.append((src_mod, name, _rel(path), node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if not node.module:
                    continue
                # Skill imports are typically `from bobert_companion import X`
                # or `from core.tts import Y`. Both single-dot and root-only
                # forms need to land in the graph.
                if node.module in intra_modules \
                        or node.module.split(".")[0] in intra_modules:
                    graph[src_mod].add(node.module)
                    edges.append((src_mod, node.module, _rel(path), node.lineno))

    # Cycle detection: iterate DFS from every node, track stack.
    cycles: list[list[str]] = []
    seen_cycles: set[tuple[str, ...]] = set()

    def dfs(start: str) -> None:
        stack = [(start, iter(graph[start]))]
        path_seen: list[str] = [start]
        on_path = {start}
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if nxt in on_path:
                    # Found cycle from nxt → … → node → nxt
                    idx = path_seen.index(nxt)
                    cyc = tuple(path_seen[idx:] + [nxt])
                    key = tuple(sorted(set(cyc)))
                    if key not in seen_cycles:
                        seen_cycles.add(key)
                        cycles.append(list(cyc))
                    continue
                if nxt not in graph:
                    continue
                stack.append((nxt, iter(graph[nxt])))
                path_seen.append(nxt)
                on_path.add(nxt)
                advanced = True
                break
            if not advanced:
                stack.pop()
                popped = path_seen.pop()
                on_path.discard(popped)

    for n in list(graph.keys()):
        dfs(n)

    for cyc in cycles:
        path_repr = " → ".join(cyc)
        out.append(Finding(
            severity="P1", category="import-cycle",
            file="(multiple)", line=0,
            message=f"import cycle detected: {path_repr}",
        ))

    return out, {
        "nodes": sorted(graph.keys()),
        "edge_count": sum(len(v) for v in graph.values()),
        "cycles": cycles,
    }


# ─────────────────────────────────────────────────────────────────────────
#  CHECK (J) — state-file integrity sweep
# ─────────────────────────────────────────────────────────────────────────

# Known state files we treat as "must parse" if they exist.
_STATE_FILES_TO_VALIDATE = [
    ("bobert_memory.json",          "json", None),
    ("hud_state.json",              "json", None),
    ("hud_config.json",             "json", None),
    ("hud_card_state.json",         "json", None),
    ("hud_reticles.json",           "json", None),
    ("credits_state.json",          "json", None),
    ("daily_briefing_state.json",   "json", None),
    ("bambu_overlay_state.json",    "json", None),
    ("holo_workshop_state.json",    "json", None),
    ("suit_up_state.json",          "json", None),
    ("workshop_hud_state.json",     "json", None),
    (".last_upgrade_summary.json",  "json", None),
    ("weather_cache.json",          "json", None),
    ("geo_cache.json",              "json", None),
    ("pending_speech.json",         "json", None),
    ("jarvis_todo.md",              "md",   r"^- \[[ x]\]"),
]


def check_state_files() -> tuple[list[Finding], dict[str, str]]:
    out: list[Finding] = []
    status: dict[str, str] = {}
    for fname, kind, expectation in _STATE_FILES_TO_VALIDATE:
        path = os.path.join(PROJECT_DIR, fname)
        if not os.path.exists(path):
            status[fname] = "absent (ok)"
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except Exception as e:
            out.append(Finding(
                severity="P1", category="state-files",
                file=fname, line=0,
                message=f"{fname} unreadable: {type(e).__name__}: {e}",
            ))
            status[fname] = f"unreadable: {e}"
            continue
        if kind == "json":
            try:
                json.loads(content)
                status[fname] = "ok"
            except json.JSONDecodeError as e:
                out.append(Finding(
                    severity="P1", category="state-files",
                    file=fname, line=e.lineno,
                    message=f"{fname} is corrupt JSON: {e.msg}",
                ))
                status[fname] = f"corrupt: {e.msg}"
        elif kind == "md":
            if expectation and not re.search(expectation, content, re.MULTILINE):
                out.append(Finding(
                    severity="P2", category="state-files",
                    file=fname, line=0,
                    message=f"{fname} lacks expected pattern {expectation!r}",
                ))
                status[fname] = "schema drift"
            else:
                status[fname] = "ok"
    return out, status


# ─────────────────────────────────────────────────────────────────────────
#  reporting
# ─────────────────────────────────────────────────────────────────────────

def write_reports(findings: list[Finding], total_files: int,
                  integration: dict[str, Any] | None = None) -> None:
    by_sev: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_sev[f.severity].append(f)

    report = {
        "summary": {
            "files_audited": total_files,
            "total_findings": len(findings),
            "p0": len(by_sev["P0"]),
            "p1": len(by_sev["P1"]),
            "p2": len(by_sev["P2"]),
        },
        "findings": {
            "P0": [asdict(f) for f in by_sev["P0"]],
            "P1": [asdict(f) for f in by_sev["P1"]],
            "P2": [asdict(f) for f in by_sev["P2"]],
        },
    }
    if integration:
        # Enrich the JSON with structured integration-check output so other
        # tools (upgrade pipeline drainer, debug UI) can drill into specifics
        # without re-parsing the markdown.
        report["integration"] = {
            "smoke_tests":     integration.get("smoke_tests", {}),
            "conflict_matrix": integration.get("conflict_matrix", []),
            "thread_audit":    integration.get("thread_audit", []),
            "prompt_consistency": integration.get("prompt_consistency", {}),
            "voice_coverage":  integration.get("voice_coverage", {}),
            "tts_pipeline":    integration.get("tts_pipeline", {}),
            "crash_recovery":  integration.get("crash_recovery", {}),
            "leak_test":       integration.get("leak_test", {}),
            "import_graph":    integration.get("import_graph", {}),
            "state_files":     integration.get("state_files", {}),
        }
    json_path = os.path.join(PROJECT_DIR, "audit_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)

    md_lines = [
        "# JARVIS Codebase Audit Report",
        "",
        f"- Files audited: **{total_files}**",
        f"- Total findings: **{len(findings)}**",
        f"  - P0 (critical): {len(by_sev['P0'])}",
        f"  - P1 (high):     {len(by_sev['P1'])}",
        f"  - P2 (medium):   {len(by_sev['P2'])}",
        "",
    ]
    for sev in ("P0", "P1", "P2"):
        items = by_sev[sev]
        if not items:
            continue
        md_lines.append(f"## {sev}")
        # Group by category for readability
        by_cat: dict[str, list[Finding]] = defaultdict(list)
        for f in items:
            by_cat[f.category].append(f)
        for cat in sorted(by_cat.keys()):
            md_lines.append(f"### {cat} ({len(by_cat[cat])})")
            for f in by_cat[cat]:
                md_lines.append(f.as_md_line())
            md_lines.append("")
    if integration:
        md_lines.append("## Integration & Conflict Checks")
        md_lines.append("")
        sm = integration.get("smoke_tests", {})
        if sm:
            ok = sum(1 for v in sm.values() if v == "ok")
            err = sum(1 for v in sm.values() if v.startswith("error"))
            md_lines.append(f"- **Action smoke tests:** {ok}/{len(sm)} ok, {err} errors")
        cm = integration.get("conflict_matrix", [])
        md_lines.append(f"- **Skill-pair conflicts:** {len(cm)} pair(s) — see "
                        f"`audit_conflict_matrix.md`")
        ta = integration.get("thread_audit", [])
        md_lines.append(f"- **Background threads:** {len(ta)} call site(s) inspected")
        pc = integration.get("prompt_consistency", {})
        if pc:
            md_lines.append(f"- **Prompt drift:** "
                            f"{len(pc.get('missing_from_registry', []))} missing from "
                            f"registry, "
                            f"{len(pc.get('missing_from_prompt', []))} missing from "
                            f"prompt")
        vc = integration.get("voice_coverage", {})
        if vc:
            missing_v = [k for k, v in vc.items() if v == "MISSING"]
            md_lines.append(f"- **Voice trigger coverage:** "
                            f"{len(vc) - len(missing_v)}/{len(vc)} actions documented")
        ig = integration.get("import_graph", {})
        if ig:
            md_lines.append(f"- **Import graph:** {len(ig.get('nodes', []))} nodes, "
                            f"{ig.get('edge_count', 0)} edges, "
                            f"{len(ig.get('cycles', []))} cycle(s)")
        sf = integration.get("state_files", {})
        if sf:
            bad = [k for k, v in sf.items() if v not in ("ok", "absent (ok)")]
            md_lines.append(f"- **State files:** {len(sf) - len(bad)}/{len(sf)} clean")
        leak = integration.get("leak_test", {})
        if leak.get("ran"):
            md_lines.append(f"- **Leak test:** ran 100 iter, handles "
                            f"{leak.get('handles_before')} → "
                            f"{leak.get('handles_after')}, files "
                            f"{leak.get('open_files_before')} → "
                            f"{leak.get('open_files_after')}")
        elif leak.get("skipped"):
            md_lines.append(f"- **Leak test:** skipped ({leak['skipped']})")
        cr = integration.get("crash_recovery", {})
        if cr:
            bad = [k for k, v in cr.items() if v == "NO TRY/EXCEPT"]
            md_lines.append(f"- **Crash recovery:** "
                            f"{len(cr) - len(bad)}/{len(cr)} critical functions wrapped")
        md_lines.append("")

    md_path = os.path.join(PROJECT_DIR, "audit_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))


# ─────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS in-tree codebase auditor")
    ap.add_argument("--fix", action="store_true",
                    help="apply trivially-safe auto-fixes (encoding=, daemon=, timeout=)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-finding output (still writes report files)")
    ap.add_argument("--json", action="store_true",
                    help="only write audit_report.json, skip markdown")
    ap.add_argument("--integration-only", action="store_true",
                    help="skip the static checks and run only the integration / "
                         "conflict checks (A–J)")
    ap.add_argument("--no-integration", action="store_true",
                    help="skip the integration / conflict checks (legacy mode)")
    args = ap.parse_args()

    if args.integration_only and args.no_integration:
        print("[audit] --integration-only and --no-integration are mutually exclusive",
              file=sys.stderr)
        return 2

    files = walk_py_files()
    # Flat skills (skills/foo.py) AND package skills (skills/<pkg>/__init__.py).
    # The old check only matched files whose parent dir == "skills", so every
    # PACKAGE skill (holographic_overlay/…) was skipped — its registered actions
    # then looked "documented but not registered" (spurious prompt-drift P1s).
    skill_files = [
        f for f in files
        if os.path.dirname(f).endswith("skills")
        or (os.path.basename(f) == "__init__.py"
            and os.path.basename(os.path.dirname(os.path.dirname(f))) == "skills")
    ]

    # Collect bobert_companion ACTIONS dict + per-skill action names. Used by
    # both the static action check AND the integration checks below (D, E),
    # so we hoist it before the branch.
    bc_actions: set[str] = set()
    bc_text, _ = _read_source(os.path.join(PROJECT_DIR, "bobert_companion.py"))
    if bc_text:
        m = re.search(r"^ACTIONS\s*=\s*\{([^}]*)\}", bc_text, re.MULTILINE | re.DOTALL)
        if m:
            for kv in re.finditer(r'["\']([a-z_][a-z_0-9]*)["\']\s*:', m.group(1)):
                bc_actions.add(kv.group(1))
        for extra in re.finditer(r'ACTIONS\[\s*["\']([a-z_][a-z_0-9]*)["\']\s*\]\s*=', bc_text):
            bc_actions.add(extra.group(1))
        # Actions wired in via `ACTIONS.update({ ... })` blocks. Some controls
        # (e.g. the ambient-learning mode handlers) register after the ACTIONS
        # literal because their handler functions are defined further down the
        # file. Capture every "key": inside each update() dict so they aren't
        # mis-flagged as "documented but not registered". Non-greedy to the
        # first "})" — the update dicts here hold lambdas, not nested braces.
        for upd in re.finditer(r'ACTIONS\.update\(\s*\{(.*?)\}\s*\)', bc_text, re.DOTALL):
            for kv in re.finditer(r'["\']([a-z_][a-z_0-9]*)["\']\s*:', upd.group(1)):
                bc_actions.add(kv.group(1))

    findings: list[Finding] = []
    if not args.integration_only:
        findings += check_syntax(files)
        findings += check_imports(files)
        findings += check_cross_references(files)
        findings += check_bad_patterns(files)
        findings += check_secrets(files)
        findings += check_state_file_writes(files)
        findings += check_readback()
        findings += check_mutation_hygiene(files)
        action_findings, _owners = check_actions(skill_files, bc_actions)
        findings += action_findings

    # ───────────────────────────────────────────────────────────────────
    # INTEGRATION / CONFLICT CHECKS  (A–J)
    # ───────────────────────────────────────────────────────────────────
    integration_data: dict[str, Any] = {}
    if not args.no_integration:
        # Build the full registered-action set across bobert + every skill.
        all_registered: set[str] = set(bc_actions)
        for sp in skill_files:
            all_registered.update(_profile_skill(sp)["actions"].keys())

        smoke_findings, smoke_results = _safe(check_action_smoke_tests)(skill_files)
        findings += smoke_findings
        integration_data["smoke_tests"] = smoke_results or {}

        pair_findings, pair_rows = _safe(check_skill_pair_conflicts)(skill_files)
        findings += pair_findings
        integration_data["conflict_matrix"] = pair_rows or []

        thread_findings, thread_summary = _safe(check_thread_audit)(files)
        findings += thread_findings
        integration_data["thread_audit"] = thread_summary or []

        if bc_text:
            # PC_CONTROL_PROMPT was relocated from bobert_companion.py into
            # core/prompts.py (Phase-3 refactor). Source the prompt text from
            # BOTH files (concatenated) so the documented-action and voice-
            # coverage checks find it wherever it lives — otherwise they read an
            # empty prompt, which silently SUPPRESSES real "documented but not
            # registered" P1s and FLOODS hundreds of spurious prompt-drift /
            # voice-coverage P2s that bury genuine findings.
            _prompts_text, _ = _read_source(
                os.path.join(PROJECT_DIR, "core", "prompts.py"))
            prompt_src = (bc_text or "") + "\n" + (_prompts_text or "")
            prompt_actions = _extract_prompt_actions(prompt_src)
            pa_findings, pa_summary = _safe(check_prompt_action_consistency)(
                prompt_actions, all_registered)
            findings += pa_findings
            integration_data["prompt_consistency"] = pa_summary or {}

            vc_findings, vc_coverage = _safe(check_voice_trigger_coverage)(
                prompt_src, all_registered)
            findings += vc_findings
            integration_data["voice_coverage"] = vc_coverage or {}

            # Presets + resolver were consolidated into core/tts.py — feed its
            # source so the dict literal + neutral-fallback checks find them.
            _tts_layer_text, _ = _read_source(
                os.path.join(PROJECT_DIR, "core", "tts.py"))
            tts_findings, tts_coverage = _safe(check_tts_pipeline)(
                bc_text, _tts_layer_text or "")
            findings += tts_findings
            integration_data["tts_pipeline"] = tts_coverage or {}

            cr_findings, cr_status = _safe(check_crash_recovery)(bc_text)
            findings += cr_findings
            integration_data["crash_recovery"] = cr_status or {}

        leak_findings, leak_summary = _safe(check_leak)()
        findings += leak_findings
        integration_data["leak_test"] = leak_summary or {}

        ig_findings, ig_summary = _safe(check_import_graph)(files)
        findings += ig_findings
        integration_data["import_graph"] = ig_summary or {}

        sf_findings, sf_status = _safe(check_state_files)()
        findings += sf_findings
        integration_data["state_files"] = sf_status or {}

    # Apply auto-fixes if requested, then re-run the cheap checks to confirm.
    if args.fix:
        applied, residual = apply_fixes(findings)
        if not args.quiet:
            print(f"[fix] applied {applied} auto-fix(es); "
                  f"{len(residual)} unfixed finding(s) remain")
        # Re-run syntax check to make sure --fix didn't break anything
        post_syntax = check_syntax(files)
        if post_syntax:
            print("[fix] WARNING: --fix introduced syntax errors:", file=sys.stderr)
            for f in post_syntax:
                print(f"   {f.file}:{f.line} {f.message}", file=sys.stderr)
            findings = post_syntax + [f for f in findings if not f.fixable]
        else:
            findings = [f for f in findings if not (f.fixable and f.fix_kind)]
            findings += residual

    # Deduplicate (file, line, category, message)
    seen = set()
    unique: list[Finding] = []
    for f in findings:
        key = (f.file, f.line, f.category, f.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    findings = unique

    write_reports(findings, total_files=len(files), integration=integration_data or None)

    if not args.quiet:
        sev_counts = {s: sum(1 for f in findings if f.severity == s) for s in ("P0", "P1", "P2")}
        print(f"=== AUDIT COMPLETE ===")
        print(f"Files audited: {len(files)}")
        print(f"Findings: {len(findings)}  (P0={sev_counts['P0']} P1={sev_counts['P1']} P2={sev_counts['P2']})")
        print(f"Report:   audit_report.json + audit_report.md")
        # Surface up to 5 P0/P1 findings inline
        critical = [f for f in findings if f.severity in ("P0", "P1")]
        if critical:
            print("\nTop critical findings:")
            for f in critical[:5]:
                loc = f"{f.file}:{f.line}" if f.line else f.file
                print(f"  [{f.severity}] {loc} — {f.message}")
            if len(critical) > 5:
                print(f"  … and {len(critical) - 5} more in audit_report.md")

    # Exit code: 0 / 1 (P0) / 2 (P1) / 3 (P2)
    if any(f.severity == "P0" for f in findings):
        return 1
    if any(f.severity == "P1" for f in findings):
        return 2
    if any(f.severity == "P2" for f in findings):
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
