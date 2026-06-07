"""
Morning handoff skill for JARVIS.

A single seamless morning briefing that chains:
  1. Weather (briefing_sources.get_weather_data)
  2. Calendar + unread mail (ms_graph)
  3. Microsoft Teams unread — called out specifically when a configured VIP
     (JARVIS_VIP_NAME) is the sender
  4. Overnight Bambu print status (bambu_monitor state)
  5. Overnight news headlines (news_briefing.get_news_text)
  6. "Anything else I should know, sir?" sign-off

Modelled on the MCU JARVIS "Good morning, sir, it's 7 AM" opener — one
continuous handoff rather than several separately triggered briefings.

Actions added:
  morning_handoff             — manually build + queue the chained briefing.
                                Returns the briefing text.
  predictive_morning_setup    — restore the typical workspace (Chrome with
                                Apple Music, Teams, optionally Bambu Studio
                                when an overnight print was active), focus
                                the middle monitor, drop master volume to
                                ~30%, then speak a one-line readback. Fires
                                automatically as the first step of the
                                handoff chain on the day's first wake; can
                                also be invoked verbally.

Auto-trigger:
  Driven by skills/morning_chain.py — a single controller polls
  bobert_companion._last_wake_date for the day's first wake event and picks
  ONE of {morning_arrival, morning_handoff, morning_briefing} to dispatch
  based on day-of-week / DEFAULT_MORNING_SKILL / time-of-day. When the chain
  picks handoff, it calls _fire_from_chain() here. Same-day suppression is
  persisted via morning_handoff_state.json so a JARVIS restart doesn't
  re-fire. This skill no longer spawns its own wake-watcher thread.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time

from core.atomic_io import _atomic_write_json

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_STATE_FILE   = os.path.join(_PROJECT_DIR, "morning_handoff_state.json")

# "After sunrise" window — first wake whose local hour falls in [start, end)
# triggers the handoff. 6 AM matches the morning_briefing convention; the
# upper bound stops a late-morning first-wake-of-the-day (e.g. user slept in
# until lunch) from getting an out-of-place "good morning" briefing.
HANDOFF_START_HOUR = 6
HANDOFF_END_HOUR   = 12

# Delay after the wake greeting before the handoff is queued, so JARVIS
# isn't speaking over its own "Good morning, sir." reply.
HANDOFF_DELAY_SECONDS = 6.0

# Background poll interval for the wake-event watcher.
WATCH_POLL_SECONDS = 5.0

# ─── predictive morning setup configuration ──────────────────────────────
# When True, the day's first-wake auto-trigger ALSO runs
# _predictive_morning_setup() before queuing the briefing — opens Chrome with
# the Apple Music tab, opens Teams, opens Bambu Studio iff a print was active
# overnight, focuses the middle monitor, and drops master volume.
PREDICTIVE_SETUP_ENABLED      = True
# Master volume target as a fraction (0.0–1.0). The spec asks for "~30%".
PREDICTIVE_SETUP_VOLUME       = 0.30
# Default URL opened in the Apple Music tab. The web client autoplays the
# user's "For You" if it's already authenticated; otherwise it lands on the
# library — either is acceptable for the morning workspace.
PREDICTIVE_APPLE_MUSIC_URL    = "https://music.apple.com/us/listen-now"
# Candidate names passed to bobert_companion._act_launch_app for Teams. The
# first one that finds a binary on disk is used.
PREDICTIVE_TEAMS_APP_NAMES    = ("microsoft teams", "teams")
# Candidate names for Bambu Studio. Same single-shot-launch contract.
PREDICTIVE_BAMBU_APP_NAMES    = ("bambu studio", "bambustudio")
# Window of time we consider "overnight" when deciding whether to open
# Bambu Studio. A FINISH announced inside this many seconds counts as
# "overnight finished" for the announcement text too.
PREDICTIVE_OVERNIGHT_WINDOW_S = 12 * 3600

_speech_lock = threading.Lock()
_state_lock  = threading.Lock()


# ─── small i/o helpers ───────────────────────────────────────────────────

def _enqueue_speech(message: str) -> None:
    """Route a spoken handoff through bobert_companion's public
    proactive_announce() API when available, falling back to a direct atomic
    write against pending_speech.json if the parent module hasn't loaded yet
    (e.g. unit test, import-time skill registration before bobert_companion
    finishes initialising). Matches the canonical pattern in
    skills/bambu_monitor.py so co-writers of pending_speech.json funnel
    through the same helper and can't race each other."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="handoff")
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
            print(f"  [handoff] speech-queue write failed ({e}); briefing: {message}")


def _load_state() -> dict:
    with _state_lock:
        if not os.path.exists(_STATE_FILE):
            return {}
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}


def _save_state(state: dict) -> None:
    with _state_lock:
        try:
            _atomic_write_json(_STATE_FILE, state)
        except Exception as e:
            print(f"  [handoff] state write failed: {e}")


def _handoff_already_fired_today() -> bool:
    """True if the handoff has already fired today. Mirrors
    skills/morning_briefing._briefing_already_fired_today so the auto-trigger
    can be cheaply suppressed at register time and we don't spawn a watcher
    thread that would do nothing all day. Also re-checked inside
    _fire_handoff so a manual trigger + watcher race can't double-fire."""
    state = _load_state()
    return state.get("last_fired_date") == time.strftime("%Y-%m-%d")


def _import_skill(name: str):
    """Best-effort import of a sibling skill module — relative first, then
    absolute via the skills directory on sys.path. Returns None on failure.

    Resolve the LIVE skill first: load_skills() registers each skill in
    sys.modules as ``skill_<name>``, and that copy holds the running
    poller's populated _state (e.g. bambu_monitor). A bare
    ``import_module(name)`` would load a SECOND, fresh copy whose _state is
    empty, so print/presence info would silently read nothing. Match
    daily_briefing / evening_briefing and prefer the already-registered
    module."""
    live = sys.modules.get(f"skill_{name}")
    if live is not None:
        return live
    try:
        return importlib.import_module(f"skills.{name}")
    except Exception:
        pass
    try:
        skills_dir = os.path.dirname(os.path.abspath(__file__))
        if skills_dir not in sys.path:
            sys.path.insert(0, skills_dir)
        return importlib.import_module(name)
    except Exception:
        return None


# ─── briefing sections ───────────────────────────────────────────────────

def _section_weather() -> str:
    """'64 degrees and overcast in your area.' Empty on failure.

    Temperature is spoken in Fahrenheit (sir's preference): briefing_sources
    stores Celsius, converted here with the canonical c*9/5+32 formula so this
    section stays in lockstep with morning_arrival / morning_briefing — the
    other two skills morning_chain may pick for the same wake event."""
    bs = _import_skill("briefing_sources")
    if not bs:
        return ""
    try:
        data = bs.get_weather_data()
    except Exception as e:
        print(f"  [handoff] weather: {e}")
        return ""
    if not data:
        return ""
    try:
        temp_c = int(data["temp_c"])
    except (KeyError, TypeError, ValueError):
        return ""
    temp_f = int(round(temp_c * 9 / 5 + 32))
    desc = (data.get("desc") or "").strip().lower()
    suffix = " (cached)" if (data.get("source") == "cache" and data.get("stale")) else ""
    if not desc:
        return f"{temp_f} degrees outside{suffix}."
    return f"{temp_f} degrees and {desc} in your area{suffix}."


def _section_calendar() -> str:
    """'Three unread emails; your first meeting is at 9:30 AM with Sam —
    design review.' Empty when Graph isn't configured / no items."""
    ms = _import_skill("ms_graph")
    if not ms:
        return ""

    parts: list[str] = []

    try:
        unread = ms.get_unread_mail_count()
    except Exception:
        unread = None
    if isinstance(unread, int) and unread > 0:
        parts.append("one unread email" if unread == 1
                     else f"{unread} unread emails")

    try:
        meeting = ms.get_first_meeting("today")
    except Exception:
        meeting = None
    if meeting:
        sdt = meeting.get("start")
        if hasattr(sdt, "hour"):
            hour = sdt.hour
            disp_hour = hour % 12 or 12
            tz_suffix = "AM" if hour < 12 else "PM"
            tstr = f"{disp_hour}:{sdt.minute:02d} {tz_suffix}"
            subject   = (meeting.get("subject") or "").strip()
            organizer = (meeting.get("organizer") or "").strip()
            who = ""
            if organizer and organizer.lower() not in (os.getenv("JARVIS_USER_NAME", "").lower(), "me"):
                first = organizer.split("<")[0].strip().split()
                if first and "@" not in first[0]:
                    who = f" with {first[0]}"
            phrase = f"your first meeting is at {tstr}{who}"
            if subject:
                phrase = f"{phrase} — {subject}"
            parts.append(phrase)

    if not parts:
        return ""
    return "From Outlook: " + "; ".join(parts) + "."


def _section_teams_vip() -> str:
    """Single-line callout for Teams unread, with extra emphasis when the
    configured VIP (JARVIS_VIP_NAME) is the visible sender. Empty when
    nothing unread or vision is offline."""
    tn = _import_skill("teams_nudge")
    if not tn:
        return ""
    try:
        has_unread, count, sender = tn._ask_vision_for_teams_state()
    except Exception as e:
        print(f"  [handoff] teams: {e}")
        return ""
    if not has_unread:
        return ""

    sender = (sender or "").strip()
    sender_l = sender.lower()
    vip_name = os.getenv("JARVIS_VIP_NAME", "").strip()
    vip = bool(vip_name) and vip_name.lower() in sender_l

    if vip:
        if count <= 1:
            return f"You have a message on Teams from {sender}, sir."
        return (f"You have {count} unread messages on Teams, sir — "
                f"including one from {sender}.")

    if count == 1:
        if sender:
            return f"One unread message on Teams from {sender}, sir."
        return "One unread message on Teams, sir."
    head = f"{count} unread messages on Teams, sir."
    if sender:
        head = f"{head} The latest is from {sender}."
    return head


def _section_print() -> str:
    """One-line overnight print status. Empty when no printer state or idle."""
    bm = _import_skill("bambu_monitor")
    if not bm:
        return ""
    try:
        with bm._state_lock:
            state       = dict(bm._state)
        gcode_state = (state.get("gcode_state") or "").upper()
        layer       = state.get("layer_num")
        total       = state.get("total_layer")
        remaining   = state.get("mc_remaining")
        fname       = state.get("filename") or ""
        last_update = state.get("last_update", 0.0)
    except Exception as e:
        print(f"  [handoff] print: {e}")
        return ""

    if last_update == 0.0:
        return ""

    # Strip the gcode extension/path noise the way bambu_monitor does.
    try:
        fname = bm._strip_filename(fname)
    except Exception:
        pass

    if gcode_state == "FINISH":
        who = f" of '{fname}'" if fname else ""
        return f"The overnight print{who} has finished, sir — your part is ready."
    if gcode_state == "FAILED":
        return ("I'm afraid the overnight print appears to have failed, sir. "
                "You'll want to take a look at the H2D.")
    if gcode_state == "PAUSE":
        return "The print is currently paused, sir."

    is_active = gcode_state in ("RUNNING", "PRINTING", "PREPARE") or (
        layer and total and gcode_state not in ("IDLE", "", None))
    if not is_active:
        return ""

    parts = []
    parts.append(f"the H2D is printing '{fname}'" if fname
                 else "the H2D is mid-print")
    if layer and total:
        parts.append(f"layer {layer} of {total}")
    try:
        remaining_str = bm._format_minutes(remaining) if remaining else ""
    except Exception:
        remaining_str = ""
    if remaining_str:
        parts.append(f"about {remaining_str} remaining")
    return ("On the workshop side, " + ", ".join(parts) + ", sir.")


def _section_news() -> str:
    """Headlines paragraph. Empty when news is disabled or every feed failed."""
    nb = _import_skill("news_briefing")
    if not nb:
        return ""
    try:
        text = nb.get_news_text()
    except Exception as e:
        print(f"  [handoff] news: {e}")
        return ""
    return (text or "").strip()


# ─── predictive morning setup ────────────────────────────────────────────

def _bobert():
    """Best-effort import of the main module so we can use its action
    primitives. Returns None when the module isn't loaded yet (unit-test
    contexts, early skill registration, etc.)."""
    return sys.modules.get("bobert_companion")


def _set_master_volume(level: float) -> bool:
    """Set the Windows playback master to `level` (0.0–1.0). Uses pycaw,
    which the audio-ducker layer in bobert_companion already pulls in;
    silent no-op on platforms or installs without it."""
    try:
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize  # type: ignore
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
    except Exception:
        return False
    com_inited = False
    try:
        CoInitialize()
        com_inited = True
    except Exception:
        pass
    try:
        devices  = AudioUtilities.GetSpeakers()
        iface    = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        endpoint = cast(iface, POINTER(IAudioEndpointVolume))
        endpoint.SetMasterVolumeLevelScalar(max(0.0, min(1.0, float(level))), None)
        return True
    except Exception:
        return False
    finally:
        if com_inited:
            try:
                CoUninitialize()
            except Exception:
                pass


def _focus_middle_monitor() -> bool:
    """Park the cursor in the centre of the middle monitor so subsequent
    app launches that open relative to the cursor land there. The browser
    Apple Music tab is opened separately via _open_url_new_window and is
    NOT cursor-anchored — but Teams + Bambu Studio launchers honour the
    cursor's monitor, so this is enough for the spec's 'focus middle
    monitor' contract."""
    bc = _bobert()
    if not bc:
        return False
    monitors = getattr(bc, "MONITORS", None)
    if not monitors or "middle" not in monitors:
        return False
    mx, my, mw, mh = monitors["middle"]
    try:
        from ctypes import windll
        windll.user32.SetCursorPos(int(mx + mw // 2), int(my + mh // 2))
        return True
    except Exception:
        return False


def _morning_pattern_apps() -> set[str]:
    """Look at pattern_learning's event log for the morning hour bucket
    (6–12 local). Returns a set of canonical workspace keys the user has
    actually used in the morning (≥3 hits): {'apple_music', 'teams',
    'bambu_studio'}. The caller still launches the spec defaults too —
    pattern hits just confirm we're not opening apps the user never wants
    in the morning."""
    pl = _import_skill("pattern_learning")
    if not pl:
        return set()
    try:
        events = pl._load_events()
    except Exception:
        return set()
    hits: dict[str, int] = {"apple_music": 0, "teams": 0, "bambu_studio": 0}
    for ev in events:
        try:
            hour = int(ev.get("hour", -1))
        except (TypeError, ValueError):
            hour = -1
        if not (6 <= hour < 12):
            continue
        action = (ev.get("action") or "").lower()
        arg    = (ev.get("arg") or "").lower()
        if action in ("apple_music", "play_music", "resume_music"):
            hits["apple_music"] += 1
        elif action == "launch_app":
            if "bambu" in arg:
                hits["bambu_studio"] += 1
            elif "teams" in arg:
                hits["teams"] += 1
        elif action == "focus_window":
            if "bambu" in arg:
                hits["bambu_studio"] += 1
            elif "teams" in arg:
                hits["teams"] += 1
    return {k for k, v in hits.items() if v >= 3}


def _open_chrome_apple_music() -> bool:
    """Spawn a new Chrome window pointed at the Apple Music web client.
    Falls back to the default browser via webbrowser.open."""
    bc = _bobert()
    url = PREDICTIVE_APPLE_MUSIC_URL
    if bc and hasattr(bc, "_open_url_new_window"):
        try:
            if bc._open_url_new_window(url):
                return True
        except Exception:
            pass
    try:
        import webbrowser
        return bool(webbrowser.open(url))
    except Exception:
        return False


def _launch_named_app(candidates) -> bool:
    """Try each name in `candidates` against bobert_companion._act_launch_app.
    Returns True on first success."""
    bc = _bobert()
    if not bc or not hasattr(bc, "_act_launch_app"):
        return False
    for name in candidates:
        try:
            result = bc._act_launch_app(name)
        except Exception:
            continue
        # _act_launch_app returns 'launched X' on success and an error
        # string on failure (typically 'no install found' / 'launch failed').
        result_l = (result or "").lower()
        if result_l.startswith("launched"):
            return True
    return False


def _overnight_print_phrase(now_ts: float) -> tuple[str, bool]:
    """Return (phrase, was_active_overnight). Both fields are set even when
    a finished print is the trigger — `was_active_overnight=True` means the
    caller should ALSO open Bambu Studio (the user's most likely next step
    is to remove the part or inspect the next layer).

    Examples of the returned phrase:
      "your overnight print finished at 4:12 AM — 2 hours under estimate"
      "your overnight print finished at 4:12 AM"
      "the H2D is still printing — layer 47 of 312, about 18 minutes left"
      ""  (no recent print, no active print)
    """
    bm = _import_skill("bambu_monitor")
    if not bm:
        return "", False

    # Recently finished?
    try:
        summary = bm.get_last_print_completion_summary(
            within_seconds=PREDICTIVE_OVERNIGHT_WINDOW_S
        )
    except Exception:
        summary = None
    if summary:
        phrase = f"your overnight print finished at {summary['finish_phrase']}"
        delta = summary.get("delta_minutes")
        if isinstance(delta, int) and abs(delta) >= 15:
            # Only call out a meaningful skew (≥15 min). "2 hours under
            # estimate" / "20 minutes over estimate".
            abs_min = abs(delta)
            if abs_min >= 60:
                hrs   = abs_min // 60
                mins  = abs_min % 60
                if mins == 0:
                    mag = f"{hrs} hour{'s' if hrs != 1 else ''}"
                else:
                    mag = f"{hrs} hour{'s' if hrs != 1 else ''} {mins} minutes"
            else:
                mag = f"{abs_min} minutes"
            tail = "under estimate" if delta > 0 else "over estimate"
            phrase += f" — {mag} {tail}"
        return phrase, True

    # Currently running?
    try:
        with bm._state_lock:
            state = dict(bm._state)
    except Exception:
        return "", False
    gcode_state = (state.get("gcode_state") or "").upper()
    if gcode_state not in ("RUNNING", "PRINTING", "PAUSE", "PREPARE"):
        return "", False
    layer       = state.get("layer_num")
    total       = state.get("total_layer")
    remaining   = state.get("mc_remaining")
    bits = []
    if layer and total:
        bits.append(f"layer {layer} of {total}")
    try:
        rem_str = bm._format_minutes(remaining) if remaining else ""
    except Exception:
        rem_str = ""
    if rem_str:
        bits.append(f"about {rem_str} left")
    if bits:
        return f"the H2D is still printing — {', '.join(bits)}", True
    return "the H2D is still printing", True


def _predictive_morning_setup(now_ts: float | None = None) -> str:
    """Open the user's typical morning workspace, then return a single-line
    JARVIS-style announcement summarising what was set up. The announcement
    is returned (NOT spoken) so callers can prepend it to a larger briefing
    or queue it on its own. Returns '' if predictive setup is disabled."""
    if not PREDICTIVE_SETUP_ENABLED:
        return ""
    if now_ts is None:
        now_ts = time.time()

    # 1. Cursor → middle monitor BEFORE the app launches so launchers that
    # open on the cursor's screen land there. Chrome's --new-window respects
    # the active window's monitor rather than the cursor, so it's separately
    # geared via existing _open_url_new_window plumbing.
    _focus_middle_monitor()

    # 2. Inform the workspace using pattern_learning hits. The spec defaults
    # (Chrome with Apple Music, Teams) always open; Bambu Studio is opened
    # only when a print is/was active overnight OR pattern_learning confirms
    # the user opens it most mornings.
    pattern_apps    = _morning_pattern_apps()
    print_phrase, print_was_active = _overnight_print_phrase(now_ts)
    open_bambu      = print_was_active or ("bambu_studio" in pattern_apps)

    opened: list[str] = []

    # 3. Chrome with Apple Music
    if _open_chrome_apple_music():
        opened.append("Apple Music is queued")
    else:
        print("  [handoff] predictive: Chrome / Apple Music launch failed")

    # 4. Teams (defer briefly so Chrome focuses first)
    time.sleep(0.4)
    if _launch_named_app(PREDICTIVE_TEAMS_APP_NAMES):
        opened.append("Teams is up")
    else:
        print("  [handoff] predictive: Teams launch failed")

    # 5. Bambu Studio when warranted
    if open_bambu:
        time.sleep(0.4)
        if _launch_named_app(PREDICTIVE_BAMBU_APP_NAMES):
            opened.append("Bambu Studio is open")
        else:
            print("  [handoff] predictive: Bambu Studio launch failed")

    # 6. Master volume → ~30%
    vol_ok = _set_master_volume(PREDICTIVE_SETUP_VOLUME)
    if not vol_ok:
        print("  [handoff] predictive: master-volume drop failed (pycaw missing?)")

    # 7. Build the readback line. Honour the spec phrasing.
    head = "Workshop is yours, sir."
    body = ", ".join(opened) if opened else ""
    if print_phrase:
        if body:
            body = f"{body}, and {print_phrase}"
        else:
            body = print_phrase.capitalize()
    closer = ""
    # Suit-style sign-off — only ask "pull up the next layer file?" when a
    # print actually finished overnight. For an in-progress print or no
    # print at all, fall through to a neutral close.
    if print_was_active and "finished" in print_phrase:
        closer = " Shall I pull up the next layer file?"
    elif print_was_active:
        closer = " Shall I keep an eye on the print, sir?"
    if body:
        return f"{head} {body}.{closer}"
    return f"{head}{closer}"


# ─── orchestration ───────────────────────────────────────────────────────

def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th') }"


def _build_handoff(setup_line: str = "") -> str:
    """Compose the chained morning handoff. Each section is independently
    optional — any one that returns '' is silently dropped from the chain.

    `setup_line` is the predictive_morning_setup readback, prepended to the
    briefing when the auto-trigger ran the workspace restore so the user
    hears the spec's 'Workshop is yours, sir...' line as the lead-in."""
    now = time.localtime()
    day = time.strftime("%A", now)
    date_str = _ordinal(now.tm_mday)
    disp_hour = now.tm_hour % 12 or 12
    tz_suffix = "AM" if now.tm_hour < 12 else "PM"
    opener = (f"Good morning, sir. It's {disp_hour}:{now.tm_min:02d} {tz_suffix} "
              f"on {day} the {date_str}.")

    sections = [opener]
    if setup_line:
        sections.append(setup_line)
    for fn in (_section_weather,
               _section_calendar,
               _section_teams_vip,
               _section_print,
               _section_news):
        try:
            piece = (fn() or "").strip()
        except Exception as e:
            print(f"  [handoff] section {fn.__name__} crashed: {e}")
            piece = ""
        if piece:
            sections.append(piece)

    sections.append("Anything else I should know, sir?")

    # [intent:briefing] makes _speak_pending use the measured "briefing" TTS
    # preset. The tag is stripped before audio synthesis.
    return "[intent:briefing] " + " ".join(sections)


def _fire_handoff(reason: str, *, force: bool = False) -> str:
    """Build and enqueue the chained handoff. When ``force`` is False (the
    auto-trigger path), bail with an empty string if the handoff has already
    fired today — defence-in-depth against a TOCTOU race between the
    register-time gate, the watcher's pre-check, and a parallel manual
    invocation. Manual triggers (force=True) always run."""
    if not force and _handoff_already_fired_today():
        print(f"  [handoff] suppressing ({reason}) — already fired today")
        return ""
    setup_line = ""
    if PREDICTIVE_SETUP_ENABLED:
        try:
            setup_line = _predictive_morning_setup() or ""
            if setup_line:
                print(f"  [handoff] predictive setup ran: {setup_line[:100]}")
        except Exception as e:
            print(f"  [handoff] predictive setup crashed: {e}")
            setup_line = ""
    text = _build_handoff(setup_line)
    print(f"  [handoff] queuing ({reason}): {text[:120]}...")
    _enqueue_speech(text)
    state = _load_state()
    state["last_fired_date"] = time.strftime("%Y-%m-%d")
    state["last_fired_ts"]   = time.time()
    state["last_reason"]     = reason
    _save_state(state)
    return text


# ─── chain entry point ───────────────────────────────────────────────────

def _fire_from_chain(reason: str = "morning_chain") -> str:
    """Auto-trigger entry called by skills/morning_chain.py once it has
    decided handoff is today's pick. Preserves the original watcher's
    TOCTOU-safe pattern verbatim: pre-check → delay → re-check → fire.
    Manual triggers ("morning handoff") still bypass via force=True."""
    if _handoff_already_fired_today():
        return ""
    time.sleep(HANDOFF_DELAY_SECONDS)
    if _handoff_already_fired_today():
        return ""
    return _fire_handoff(reason)


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    def morning_handoff(_: str = "") -> str:
        # Manual invocation ALWAYS runs — user explicitly asked for it. force=True
        # bypasses the same-day suppression that protects the auto-trigger path.
        try:
            return _fire_handoff("manual trigger", force=True)
        except Exception as e:
            return f"morning handoff failed: {e}"

    def predictive_morning_setup(_: str = "") -> str:
        """Manually invoke the workspace setup — opens Chrome/Apple Music,
        Teams, optionally Bambu Studio, focuses the middle monitor, sets
        master volume, and returns the readback. The returned string is
        spoken by the action dispatcher's normal LLM-driven reply path —
        we deliberately don't queue it on pending_speech so the LLM can
        embed it inside a richer in-context reply if the user asked while
        already mid-conversation."""
        try:
            line = _predictive_morning_setup()
            return line or "Workshop setup is already in order, sir."
        except Exception as e:
            return f"predictive morning setup failed: {e}"

    actions["morning_handoff"]          = morning_handoff
    actions["predictive_morning_setup"] = predictive_morning_setup
    actions["setup_workspace"]          = predictive_morning_setup  # natural alias
    actions["workspace_setup"]          = predictive_morning_setup  # natural alias

    # Auto-trigger is owned by skills/morning_chain.py — no per-skill watcher.
