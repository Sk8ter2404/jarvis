"""Exact alias table for invented action names (v3 W04, 2026-10-27).

The model sometimes names an action that does not exist: ``read_screen``,
``pause``, ``volume_set``, ``open_app``. The fuzzy corrector
(command_autocorrect.py) then guesses, and some of its guesses are wrong
and silent: play_video -> play_vibe (0.94), toggle_mute -> toggle_hud (0.82),
open_app -> open_url (0.76) - JARVIS-Research/v3_overhaul_20261027/
05_action_vocabulary.md section 4. Most invented names have one obvious real
action, so this table maps them EXACTLY, with no scoring, before the fuzzy
layer runs (bobert_companion.parse_and_run_actions._runner, logged as
``[alias] x -> y``).

Rules for every row (pinned by tests/test_action_aliases.py):
  * the source is NOT a registered action - a real name always wins, and
    _runner only looks here when ACTIONS has no such name;
  * the target IS registered;
  * the target is SAFE: never self-terminating (core.action_risk
    SELF_TERMINATING_ACTIONS), never _autocorrect_protected, never
    CONFIRM-tier, never destructive. The rewritten name still goes through
    every gate after it (self-termination, confirmation, pushback), exactly
    as if the model had named it;
  * nothing in NEVER_ALIAS / NEVER_ALIAS_PREFIXES is ever a source: sends,
    deletes, money, shutdown / exit / restart, close-all, file moves, shell.

Pure data plus two pure helpers; no imports from the monolith. Behind the
``ACTION_ALIASES`` setting (core/config.py); off = the old path, untouched.
"""
from __future__ import annotations

import re

# invented name -> registered canonical name. SAFE rows only. Each target was
# checked against the registration scan of v2.0.231 (tools/registration_scan).
ALIASES: dict[str, str] = {
    # The screen: "read the results" means look at the live screen.
    # (Live 10-19 19:44:48: read_screen -> "Did you mean recall_screen or
    # read_thread?")
    "read_screen": "see_screen",
    "read_the_screen": "see_screen",
    "read_page": "see_screen",
    "read_results": "see_screen",
    "take_screenshot": "screenshot",
    "screenshot_screen": "screenshot",
    "screen_capture": "screenshot",       # fuzzy: screen_history (wrong)
    "capture_screen": "screenshot",
    "describe_screen": "local_describe_screen",
    "look_at_screen": "local_describe_screen",
    # Windows
    "list_windows_all": "list_windows",
    "show_windows": "list_windows",
    "show_window": "focus_window",
    "bring_back_window": "focus_window",
    # Read-only status
    "network_scan": "network_usage",
    "check_network": "network_usage",
    "connection_status": "network_usage",
    "list_devices": "list_smart_home_devices",
    "list_smart_devices": "list_smart_home_devices",
    "smart_home_list_devices": "list_smart_home_devices",
    "check_server": "server_status",
    "server_check": "server_status",
    "homeserver_status": "server_status",
    "check_cpu": "system_pulse",
    "system_info": "system_pulse",
    "check_system_health": "system_pulse",
    # Sound and media
    "volume_set": "set_volume",           # fuzzy: volume_mute vs volume_up
    "set_the_volume": "set_volume",
    "mute": "volume_mute",
    "mute_volume": "volume_mute",
    "toggle_mute": "volume_mute",         # fuzzy: toggle_hud (wrong)
    # "mute the music": the literal ask is mute. volume_mute sets the system
    # mute ON (idempotent; volume_unmute undoes it). Not pause_music (a
    # different job: the old silent autocorrect, now refused by the verb-clash
    # guard, so with no row the turn went unanswered) and not media_playpause
    # (a toggle that would RESUME paused music).
    "mute_music": "volume_mute",
    "pause": "media_playpause",
    "pause_video": "media_playpause",
    "pause_playback": "media_playpause",
    "stop_music": "pause_music",
    "play_video": "youtube_play",         # fuzzy: play_vibe 0.94 (wrong)
    # Browser and apps (browser_open is a REAL action: skills/browser_agent)
    "open_browser": "open_url",
    "open_website": "open_url",
    "web_open": "open_url",
    "open_app": "launch_app",             # fuzzy: open_url (wrong)
    "open_application": "launch_app",
    "start_app": "launch_app",
    "launch": "launch_app",
    # Clicks (_click_alias already covers a described "click")
    "click_element": "click_on_screen",
    "click_screen": "click_on_screen",
    "click_button": "click_on_screen",
}

# Targets that only make sense with an argument: with none the alias does not
# apply and the name takes today's path (an empty youtube_play / open_url /
# launch_app would only fail, or play something unasked).
NEEDS_ARG = frozenset({
    "youtube_play", "open_url", "launch_app", "focus_window",
    "click_on_screen", "set_volume",
})

# Never aliased, whatever a future row says (05_action_vocabulary.md section
# 6, NEVER rows). These keep today's path: blocked + a question, or
# "unknown action".
NEVER_ALIAS = frozenset({
    "send_email", "send", "send_money", "transfer", "transfer_funds",
    "shutdown", "shutdown_pc", "shut_down", "exit", "quit", "power_off",
    "turn_off", "restart", "restart_pc", "reboot", "sleep_pc", "hibernate",
    "close_all_windows", "close_everything", "close_all", "close_all_apps",
    "move_file", "copy_file", "organize_files", "open_file",
    "delete", "delete_folder", "delete_folder_contents", "delete_window",
    "remove_file", "run_shell", "run_code", "type", "lock", "sleep",
})
NEVER_ALIAS_PREFIXES = (
    "send_", "delete_", "remove_", "wipe_", "format_", "erase_", "pay_",
    "buy_", "purchase", "checkout", "transfer", "kill_", "shutdown",
    "shut_", "restart", "reboot", "exit", "quit", "power_off", "turn_off",
    "close_all", "move_", "copy_", "run_", "reset_", "forget_", "uninstall",
)


def is_never_alias(name: str) -> bool:
    """True when ``name`` must never be rewritten by this table."""
    n = (name or "").strip().lower()
    return n in NEVER_ALIAS or n.startswith(NEVER_ALIAS_PREFIXES)


def alias_for(name: str, arg: str, registered) -> "str | None":
    """The canonical action for an invented ``name``, or None.

    None when ``name`` is registered (a real name always wins), has no row,
    is a NEVER name, the target is not registered (a skill that did not
    load), or the target needs an argument and ``arg`` is empty. Never
    raises: any fault is None (today's path)."""
    try:
        n = (name or "").strip().lower()
        if not n or n in registered or is_never_alias(n):
            return None
        target = ALIASES.get(n)
        if not target or target not in registered:
            return None
        if target in NEEDS_ARG and not str(arg or "").strip():
            return None
        return target
    except Exception:
        return None


# ─── Hidden duplicates (v3 overhaul W12, PROMPT_INDEX_CANON) ─────────────
# Registered actions that do (nearly) the same job as a canonical one. Unlike
# ALIASES these are REAL names: they stay in ACTIONS, keep their core.action_risk
# tier, stay in the INFORMATIVE / VERBATIM sets, and still run when the model
# says them - nothing is ever rewritten. They are only left out of the ACTION
# INDEX in the prompt (core.prompt_router.slim_stable_pc_block ``hide``), so
# the model sees one name per job: 8 pause / next names, 11 play / video
# names, 12 smart-home list names, ``list_schedule`` beside ``list_schedules``
# and so on. A name is hidden only while its canonical name is visible in the
# prompt (the head grammar or a listed row); otherwise it stays listed.
# hidden name -> the canonical name the model should say instead.
# A pair is listed only when the canonical name does the hidden name's whole
# job. Reviewed OUT (not near-duplicates): schedule_status (scheduler health,
# not the job list), is_printer_online (no-arg printer check), the toggle_*
# names (flip, not set), morning_arrival / morning_handoff (own builders;
# handoff also opens apps), the stark / arc ring statuses (other HUDs),
# hand_mouse_* (mouse only), last_diagnostic_run (raw JSON) and
# rag_search_quiet (machine block for Claude). The test pins that each
# remaining pair shares a handler or sits on a reasoned allow-list.
# Pinned by tests/test_prompt_index_canon.py: every key and value is
# registered, no value is itself hidden, no key is in NEVER_ALIAS /
# SELF_TERMINATING_ACTIONS / CONFIRM tier, and a hidden name's risk tier
# equals its canonical's or is stricter.
HIDDEN_DUPLICATES: dict[str, str] = {
    # Raw media keys beside the media-session actions (pause the video ->
    # media_prev was the c013 failure).
    "media_next": "next_song",
    "media_prev": "previous_song",
    "media_playpause": "pause_music",
    "music_status": "now_playing",
    # One generic streaming action takes "<service>|<title>".
    "prime_video": "play_streaming",
    "netflix": "play_streaming",
    "disney_plus": "play_streaming",
    "hulu": "play_streaming",
    "max": "play_streaming",
    "spotify": "play_streaming",
    # One smart-home list instead of one per brand.
    "hue_list_devices": "list_smart_home_devices",
    "govee_list_devices": "list_smart_home_devices",
    "lifx_list_devices": "list_smart_home_devices",
    "kasa_list_devices": "list_smart_home_devices",
    "ecobee_list_devices": "list_smart_home_devices",
    "nest_list_devices": "list_smart_home_devices",
    "ring_list_devices": "list_smart_home_devices",
    "magichome_list_devices": "list_smart_home_devices",
    "tuya_list": "list_smart_home_devices",
    # Status and health.
    "check_system": "system_pulse",
    "system_check": "run_diagnostic",
    "diagnostic_daemon_status": "diagnostic_status",
    "last_gate_result": "gate_status",
    "which_mic_is_active": "current_mic",
    "bandwidth_hogs": "network_usage",
    # Schedules: `list_schedule` and `list_schedules` were both listed.
    "list_schedule": "list_schedules",
    "fire_schedule": "run_schedule",
    # Focus mode.
    "focus_mode_on": "focus_mode",
    "focus_mode_off": "end_focus_mode",
    # Wrappers of the same call (make_picture just calls generate_image).
    "make_picture": "generate_image",
}


def hidden_duplicate_of(name: str) -> "str | None":
    """The canonical name a hidden duplicate stands for, or None. Pure data
    lookup: it never rewrites a name (the hidden name still runs as itself)."""
    try:
        return HIDDEN_DUPLICATES.get((name or "").strip().lower())
    except Exception:
        return None


# Short words said as letters, so a label reads the way the owner says it.
_LABEL_WORDS = {
    "hud": "HUD", "url": "URL", "tv": "TV", "pc": "PC", "cpu": "CPU",
    "gpu": "GPU", "obs": "OBS", "dmx": "DMX", "ups": "UPS", "vip": "VIP",
    "ip": "IP", "usb": "USB", "rgb": "RGB", "ai": "AI",
}


def friendly_label(name: str) -> str:
    """An action name as words for speech: ``recall_screen`` -> "recall
    screen", ``toggle_hud`` -> "toggle HUD". No backticks, no underscores.
    The words stay the name's own, so the owner can answer a "did you mean"
    by saying them (_resolve_autocorrect_choice matches the name with its
    underscores as spaces)."""
    words = [w for w in re.split(r"[_\s\-]+", str(name or "").strip()) if w]
    if not words:
        return "that"
    return " ".join(_LABEL_WORDS.get(w.lower(), w.lower()) for w in words)
