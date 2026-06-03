"""
Do-not-disturb focus mode for JARVIS.

Triggered either by voice ("JARVIS, focus mode for 90 minutes") or
automatically when workshop_mode engages (CAD / slicer activity). While
active:

  • Windows Focus Assist is toggled to 'priority only' via the registry
    CloudStore entry (best-effort — falls back to a console warning if the
    write fails, since the rest of the skill is still useful).
  • Microsoft Teams presence is set to DoNotDisturb via the Graph API
    /me/presence/setUserPreferredPresence call. Requires the Graph token
    to have Presence.ReadWrite scope — degrades silently otherwise.
  • Non-critical proactive nudges (banter, wellness, teams_nudge) are
    paused by monkey-patching each skill's _enqueue_speech with a sink
    that drops queued messages while focus mode is active. Critical
    skills (bambu_monitor failure / timer reminders) stay unaffected so
    a print failure can still interrupt.
  • A brief addendum is appended to bobert_companion._system_prompt so
    the LLM knows the user is heads-down and should keep replies even
    terser than normal.

A background expiry thread auto-restores everything when the duration
elapses. The user can also voice 'end focus mode' to cancel early.

Actions registered:
  focus_mode, <duration>      — engage. duration like "90 minutes" or
                                "1 hour 30 minutes"; default 60 min if
                                blank or unparseable.
  end_focus_mode              — cancel early.
  focus_mode_status           — report current state and time remaining.

Module-level helper for other skills:
  is_focus_mode_active() -> bool

Announces on entry:
  "Holding all non-critical interruptions for N minutes, sir. I'll wake
   you for VIPs or emergencies only."
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from core.atomic_io import _atomic_write_json

# ─── Config ──────────────────────────────────────────────────────────────
DEFAULT_DURATION_SECONDS = 60 * 60      # 60 min if user doesn't specify
WORKSHOP_AUTO_DURATION   = 60 * 60      # auto-trigger from workshop_mode
MAX_DURATION_SECONDS     = 8 * 3600     # safety cap
TEAMS_GRAPH_TIMEOUT      = 6.0

# Skills whose _enqueue_speech we suppress while focus mode is active.
# Critical announcements (bambu print failure, timer reminders, audio
# device drops) stay loud — those are the "VIPs or emergencies" of the
# announcement copy.
SUPPRESSED_SKILLS = ("banter", "wellness", "teams_nudge")

FOCUS_PROMPT_ADDENDUM = (
    "\n\n[Focus mode]\n"
    "Sir is in a do-not-disturb focus block. Keep replies to a single "
    "short sentence. Skip pleasantries and speculation. Only volunteer "
    "information if it is genuinely urgent."
)

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")

# ─── State ───────────────────────────────────────────────────────────────
_mode_lock           = threading.Lock()
_focus_active        = [False]
_focus_started_at    = [0.0]
_focus_ends_at       = [0.0]
_focus_trigger       = [""]      # "voice" | "workshop" | "manual"

_saved_enqueues: dict[str, object] = {}     # skill_name → original callable
_saved_system_prompt: list = [None]
_teams_was_set       = [False]
_focus_assist_was_set = [False]
_expiry_thread       = [None]

_speech_lock = threading.Lock()
_workshop_hook_installed = [False]


# ─── Tiny helpers ────────────────────────────────────────────────────────

def _enqueue_speech(message: str) -> None:
    """Speak via the main loop's queue. Used for our OWN announcements,
    which always go through regardless of focus state.

    Routes through bobert_companion.proactive_announce() — the canonical
    writer for pending_speech.json — so we share its read-modify-write
    sequence with bambu_monitor and the briefing skills. Falls back to a
    direct atomic write only if the parent module isn't importable yet
    (e.g. import-time skill registration before bobert_companion loads)."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="focus")
            return
    except Exception:
        pass

    with _speech_lock:
        data = []
        if os.path.exists(_SPEECH_QUEUE):
            try:
                with open(_SPEECH_QUEUE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append({"ts": time.time(), "message": message})
        try:
            _atomic_write_json(_SPEECH_QUEUE, data)
        except Exception as e:
            print(f"  [focus] speech-queue write failed ({e}); message: {message}")


def _parse_duration_to_seconds(text: str) -> int | None:
    """Mini parser for '90 minutes', '1 hour 30 min', '45m', etc."""
    if not text:
        return None
    text = text.strip().lower()
    total = 0
    found = False
    pattern = r"(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)"
    for n, unit in re.findall(pattern, text):
        n = int(n)
        if unit in ("s", "sec", "secs", "second", "seconds"):
            total += n
        elif unit in ("m", "min", "mins", "minute", "minutes"):
            total += n * 60
        elif unit in ("h", "hr", "hrs", "hour", "hours"):
            total += n * 3600
        found = True
    if found:
        return total if total > 0 else None
    bare = re.fullmatch(r"\s*(\d+)\s*", text)
    if bare:
        # Bare number → assume minutes (user said "90", meant 90 min).
        return int(bare.group(1)) * 60
    return None


def _format_minutes(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} seconds"
    m = round(seconds / 60)
    if m < 60:
        return f"{m} minutes" if m != 1 else "1 minute"
    h, rem = divmod(m, 60)
    if rem == 0:
        return f"{h} hours" if h != 1 else "1 hour"
    return f"{h} hour{'s' if h != 1 else ''} {rem} minutes"


# ─── Windows Focus Assist control (best-effort) ──────────────────────────

# These three keys are flipped by the Windows shell when the user toggles
# Focus Assist via the Action Center. They drive whether the OS will pop
# toast notifications. The CloudStore profile path that holds the *mode*
# (off / priority / alarms) is a binary blob and can't be safely poked, so
# we settle for the global toast-suppression key — it's the part that
# materially affects whether interruptions hit the screen.
_FA_KEY = r"HKCU\Software\Microsoft\Windows\CurrentVersion\PushNotifications"
_FA_VAL = "ToastEnabled"


def _set_focus_assist(enable_dnd: bool) -> bool:
    """Try to suppress (or restore) Windows toasts via reg.exe. Returns
    True if the registry write succeeded — caller treats False as a
    soft failure."""
    try:
        # 0 = toasts disabled (DND-ish), 1 = toasts enabled (normal).
        value = "0" if enable_dnd else "1"
        result = subprocess.run(
            ["reg.exe", "add", _FA_KEY, "/v", _FA_VAL, "/t", "REG_DWORD",
             "/d", value, "/f"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            return True
        print(f"  [focus] reg.exe failed (rc={result.returncode}): "
              f"{(result.stderr or '').strip()[:160]}")
    except Exception as e:
        print(f"  [focus] reg.exe error: {e}")
    return False


# ─── Microsoft Teams presence (Graph) ────────────────────────────────────

def _set_teams_presence(state: str) -> bool:
    """Best-effort: PATCH /me/presence/setUserPreferredPresence.

    state ∈ {'DoNotDisturb', 'Available'}. Requires Presence.ReadWrite
    scope on the token. Silent no-op if Graph isn't configured."""
    try:
        mod = importlib.import_module("skills.ms_graph")
    except Exception:
        try:
            mod = sys.modules.get("skill_ms_graph")
        except Exception:  # pragma: no cover - unreachable: dict.get() never raises; defensive only
            mod = None
    if mod is None:
        return False
    try:
        token = mod.get_access_token()
    except Exception:
        token = None
    if not token:
        return False

    body = {
        "sessionId":        "22553428-c9d3-411e-9da2-f9bd2bdc70b9",
        "availability":     state,
        "activity":         state,
        "expirationDuration": "PT1H",
    }
    url = "https://graph.microsoft.com/v1.0/me/presence/setUserPreferredPresence"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TEAMS_GRAPH_TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        body_txt = ""
        try:
            body_txt = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        print(f"  [focus] Teams presence http {e.code}: {e.reason} {body_txt}")
    except Exception as e:
        print(f"  [focus] Teams presence error: {e}")
    return False


# ─── Proactive-nudge suppression ─────────────────────────────────────────

def _install_nudge_suppressors() -> None:
    """For each skill in SUPPRESSED_SKILLS, swap its module-level
    _enqueue_speech with a sink that just logs and drops. Caches the
    original so we can restore exactly what was there.

    If the target's current _enqueue_speech is already one of these sinks
    (another overlapping mode — e.g. night_owl — installed it first), we do
    NOT save it as the "original" and do NOT double-wrap: the existing sink
    keeps suppressing, and we record None so restore knows this skill didn't
    own the patch and must leave it alone. Prevents an overlapping focus +
    night-owl from saving a sink as the original and stranding the target on
    that sink after the second mode exits."""
    for skill in SUPPRESSED_SKILLS:
        mod = sys.modules.get(f"skill_{skill}")
        if mod is None:
            continue
        original = getattr(mod, "_enqueue_speech", None)
        if not callable(original):
            continue
        if skill in _saved_enqueues:
            # Already wrapped by us — don't double-wrap.
            continue
        if getattr(original, "_is_nudge_sink", False):
            # Already suppressed by another mode. Leave its sink in place and
            # record that we don't own it (None) so restore is a no-op for it.
            _saved_enqueues[skill] = None
            continue
        _saved_enqueues[skill] = original

        def _sink(message: str, _skill=skill):
            print(f"  [focus] suppressed {_skill} nudge: {message[:120]}")
        _sink._is_nudge_sink = True
        try:
            mod._enqueue_speech = _sink
        except Exception as e:
            print(f"  [focus] could not wrap {skill}._enqueue_speech: {e}")
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
                print(f"  [focus] could not restore {skill}._enqueue_speech: {e}")
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
        bc._system_prompt = _saved_system_prompt[0] + FOCUS_PROMPT_ADDENDUM
    except Exception as e:
        print(f"  [focus] prompt addendum failed: {e}")


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

def is_focus_mode_active() -> bool:
    """Public helper other skills can call to check whether they should
    suppress their own proactive output. Kept simple and lock-free for
    cheap polling — readers see a consistent True/False even when the
    state mutates."""
    return bool(_focus_active[0])


def _start_expiry_thread() -> None:
    def _wait_and_exit():
        # Sleep in small steps so end_focus_mode can preempt cleanly.
        while True:
            try:
                with _mode_lock:
                    if not _focus_active[0]:
                        return
                    remaining = _focus_ends_at[0] - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 5.0))
            except Exception:
                logging.exception("[focus] expiry thread iteration failed")
                time.sleep(5.0)
        _exit_focus_mode(reason="expired")

    t = threading.Thread(target=_wait_and_exit, daemon=True)
    t.start()
    _expiry_thread[0] = t


def _enter_focus_mode(duration_seconds: int, trigger: str) -> tuple[bool, str]:
    """Engage focus mode. Returns (was_already_active, summary_message)."""
    duration_seconds = max(60, min(int(duration_seconds), MAX_DURATION_SECONDS))
    now = time.time()
    with _mode_lock:
        if _focus_active[0]:
            # Extend rather than restart — feels less abrupt if user
            # repeats the command. The expiry thread reads _focus_ends_at
            # freshly, so this just slides the deadline forward.
            _focus_ends_at[0] = max(_focus_ends_at[0], now + duration_seconds)
            return True, (
                f"Already in focus mode, sir — extended to "
                f"{_format_minutes(int(_focus_ends_at[0] - now))} from now."
            )

        _focus_active[0]     = True
        _focus_started_at[0] = now
        _focus_ends_at[0]    = now + duration_seconds
        _focus_trigger[0]    = trigger

    # Install side effects outside the lock so a slow Graph call doesn't
    # hold up status queries.
    _install_nudge_suppressors()
    _apply_prompt_addendum()
    _focus_assist_was_set[0] = _set_focus_assist(enable_dnd=True)
    _teams_was_set[0]        = _set_teams_presence("DoNotDisturb")
    _start_expiry_thread()

    summary = (
        f"Holding all non-critical interruptions for "
        f"{_format_minutes(duration_seconds)}, sir. I'll wake you for "
        f"VIPs or emergencies only."
    )
    _enqueue_speech(summary)
    print(f"  [focus] engaged via {trigger} for {duration_seconds}s — "
          f"focus_assist={_focus_assist_was_set[0]} teams={_teams_was_set[0]}")
    return False, summary


def _exit_focus_mode(reason: str = "manual") -> str:
    """Cancel focus mode. Idempotent."""
    with _mode_lock:
        if not _focus_active[0]:
            return "Focus mode was not active, sir."
        _focus_active[0] = False
        trigger = _focus_trigger[0]
        _focus_trigger[0] = ""
        _focus_started_at[0] = 0.0
        _focus_ends_at[0]    = 0.0

    _restore_nudge_suppressors()
    _restore_prompt_addendum()
    if _focus_assist_was_set[0]:
        _set_focus_assist(enable_dnd=False)
        _focus_assist_was_set[0] = False
    if _teams_was_set[0]:
        _set_teams_presence("Available")
        _teams_was_set[0] = False

    print(f"  [focus] disengaged ({reason}, was triggered by {trigger})")
    if reason == "expired":
        msg = "Focus mode complete, sir — interruptions restored."
    elif reason == "workshop_exit":
        msg = "Workshop closed, sir — focus mode released."
    else:
        msg = "Focus mode disengaged, sir."
    _enqueue_speech(msg)
    return msg


# ─── Workshop_mode auto-trigger hook ─────────────────────────────────────

def _install_workshop_hook() -> None:
    """Wrap workshop_mode._enter_workshop_mode so that when CAD activity
    flips workshop mode on, focus mode auto-engages too. The wrap also
    catches workshop exit and releases focus mode if it was workshop-
    triggered (so a brief Bambu Studio session doesn't strand the user
    in DND afterwards)."""
    if _workshop_hook_installed[0]:
        return
    mod = sys.modules.get("skill_workshop_mode")
    if mod is None:
        return
    orig_enter = getattr(mod, "_enter_workshop_mode", None)
    orig_exit  = getattr(mod, "_exit_workshop_mode", None)
    if not callable(orig_enter) or not callable(orig_exit):
        return

    def _wrapped_enter(matched_hint, full_title, _orig=orig_enter):
        result = _orig(matched_hint, full_title)
        try:
            with _mode_lock:
                already = _focus_active[0]
            if not already:
                _enter_focus_mode(WORKSHOP_AUTO_DURATION, trigger="workshop")
        except Exception as e:
            print(f"  [focus] workshop auto-trigger failed: {e}")
        return result

    def _wrapped_exit(_orig=orig_exit):
        result = _orig()
        try:
            with _mode_lock:
                active = _focus_active[0]
                trigger = _focus_trigger[0]
            if active and trigger == "workshop":
                _exit_focus_mode(reason="workshop_exit")
        except Exception as e:
            print(f"  [focus] workshop exit hook failed: {e}")
        return result

    try:
        mod._enter_workshop_mode = _wrapped_enter
        mod._exit_workshop_mode  = _wrapped_exit
        _workshop_hook_installed[0] = True
        print("  [focus] workshop_mode auto-trigger hook installed")
    except Exception as e:
        print(f"  [focus] could not install workshop hook: {e}")


def _delayed_workshop_hook() -> None:
    """workshop_mode loads alphabetically after dnd_focus_mode (d < w)?
    No — d < w but the skill loader sorts alphabetically and 'd' comes
    BEFORE 'w', so workshop_mode is imported AFTER us. Defer the hook
    install until after the loader has finished walking the directory."""
    try:
        # 5 s is plenty for load_skills() to finish even on a cold boot.
        time.sleep(5.0)
        _install_workshop_hook()
    except Exception:
        logging.exception("[focus] _delayed_workshop_hook crashed")


# ─── Action handlers ─────────────────────────────────────────────────────

def register(actions):
    def focus_mode(args: str = "") -> str:
        duration = _parse_duration_to_seconds(args) if args else None
        if duration is None:
            duration = DEFAULT_DURATION_SECONDS
        was_active, msg = _enter_focus_mode(duration, trigger="voice")
        return msg

    def end_focus_mode(_: str = "") -> str:
        return _exit_focus_mode(reason="manual")

    def focus_mode_status(_: str = "") -> str:
        with _mode_lock:
            active = _focus_active[0]
            ends_at = _focus_ends_at[0]
            trigger = _focus_trigger[0]
        if not active:
            return "Focus mode is not currently engaged, sir."
        remaining = max(0, int(ends_at - time.time()))
        trig = {"voice": "by voice", "workshop": "by workshop activity",
                "manual": "manually"}.get(trigger, trigger)
        return (f"Focus mode is engaged ({trig}), sir — about "
                f"{_format_minutes(remaining)} remaining.")

    actions["focus_mode"]        = focus_mode
    actions["end_focus_mode"]    = end_focus_mode
    actions["focus_mode_status"] = focus_mode_status

    t = threading.Thread(target=_delayed_workshop_hook, daemon=True)
    t.start()
    print("  [focus] dnd_focus_mode ready — actions: focus_mode, "
          "end_focus_mode, focus_mode_status")
