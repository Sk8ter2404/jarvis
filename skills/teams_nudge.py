"""
Microsoft Teams unread-message nudge.

Every CHECK_INTERVAL_SECONDS, this skill takes a full-screen capture, asks
Claude vision to look for a Teams unread badge (taskbar icon or open window
title), and if it sees one — and the same alert hasn't been spoken in the
last SNOOZE_SECONDS — queues a JARVIS-style spoken nudge.

Actions added:
  check_teams   — manual on-demand check that returns the verdict as text.

Config knobs are at the top of the file. The first sweep is delayed by
INITIAL_DELAY_SECONDS so the user isn't bombarded the moment JARVIS boots.
"""
import importlib
import json
import logging
import os
import re
import sys
import threading
import time

_log = logging.getLogger(__name__)

# ─── config ───────────────────────────────────────────────────────────────
CHECK_INTERVAL_SECONDS = 600    # 10 minutes between background checks
SNOOZE_SECONDS         = 1800   # don't re-nudge within 30 min of same alert
INITIAL_DELAY_SECONDS  = 180    # wait 3 min after boot before first check
# ─────────────────────────────────────────────────────────────────────────

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")

# Ensure the project root is importable so `core.atomic_io` resolves whether
# this module is loaded as `skills.teams_nudge` or run directly.
if _PROJECT_DIR not in sys.path:  # pragma: no cover - import-time sys.path guard; root already on path under the test harness
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402
from core.draft_confirm import draft_confirm  # noqa: E402

_speech_lock     = threading.Lock()
_last_alert_at   = [0.0]
_last_alert_text = [""]


def _enqueue_speech(message: str) -> None:
    """Gate a spoken nudge through draft_confirm, then enqueue if approved.

    Per the user's standing rule — "reads messages/drafts aloud before
    sending" — every outbound nudge routes through ``draft_confirm`` first.
    The gate reads the draft aloud, asks "shall I send it?", and only
    on an explicit yes does the message land in ``pending_speech.json``
    for the main loop to speak. A denied or timed-out confirmation
    silently drops the nudge (fail-closed) so JARVIS never auto-speaks
    a message the user didn't approve.

    When approved, routes through bobert_companion.proactive_announce()
    so this skill shares one write path with every other pending_speech.json
    co-writer (bambu_monitor, status_panel, night_owl_mode, screen_watch, …)
    and they don't race each other. Falls back to a local atomic write only
    when the parent module isn't loaded yet (import-time registration /
    unit tests) or the announcer call fails — so a broken parent import
    can't silence a Teams nudge.
    """
    if not draft_confirm(message, recipient="you"):
        print(f"  [teams] nudge denied / unconfirmed — dropped: {message}")
        return

    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="teams_nudge")
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
            # Atomic write failed (e.g. read-only network share, full disk,
            # permission denied). Fall back to console so the nudge isn't
            # silently lost — at minimum the user sees it in the log stream.
            print(f"  [teams] speech-queue write failed ({e}); nudge: {message}")


def _import_companion():
    import importlib
    return importlib.import_module("bobert_companion")


def _ask_vision_for_teams_state() -> tuple[bool, int, str]:
    """Capture all monitors and ask Claude whether Teams shows unread badges.

    Returns (has_unread, count_or_zero, raw_answer).
    Falls back to (False, 0, error_string) when vision can't reach a verdict.
    """
    bc = _import_companion()

    try:
        images = bc.take_all_monitor_screenshots()
    except Exception as e:
        return False, 0, f"capture_failed: {e}"
    if not images:
        return False, 0, "no_images"

    question = (
        "Look across all of these monitors for the Microsoft Teams "
        "application. Check both the Windows taskbar (Teams icon may show a "
        "red unread-count badge) and any open Teams window (sidebar "
        "chat/channel rows often show bold names with a count badge).\n\n"
        "Respond with ONE LINE in exactly this format:\n"
        "  UNREAD: N | first sender name if visible, else NONE\n"
        "  NONE\n"
        "Examples:\n"
        "  UNREAD: 3 | Alex Morgan\n"
        "  UNREAD: 1 | NONE\n"
        "  NONE\n"
        "Reply NONE if Teams is not visible at all or has no unread "
        "indicators."
    )

    try:
        answer = bc.ask_vision_multi(question, images).strip()
    except Exception as e:
        return False, 0, f"vision_failed: {e}"

    upper = answer.upper()
    if upper.startswith("NONE") or "UNREAD:" not in upper:
        return False, 0, answer

    m = re.search(
        r"UNREAD:\s*(\d+)\s*(?:\|\s*(.+?))?\s*$",
        answer.splitlines()[0], re.IGNORECASE,
    )
    if not m:
        return False, 0, answer
    count = int(m.group(1))
    sender = (m.group(2) or "").strip()
    return (count > 0), count, sender if sender and sender.upper() != "NONE" else ""


def _build_message(count: int, sender: str) -> str:
    if count == 1:
        if sender:
            return (f"You have an unread message on Teams, sir — from "
                    f"{sender}, if you care to look.")
        return "You have an unread message on Teams, sir."
    if sender:
        return (f"You have {count} unread messages on Teams, sir — "
                f"including one from {sender}, if you care to look.")
    return f"You have {count} unread messages on Teams, sir."


def _check_once() -> tuple[bool, str]:
    has_unread, count, sender = _ask_vision_for_teams_state()
    if not has_unread:
        return False, "no_unread"

    msg = _build_message(count, sender)
    now = time.time()
    # Snooze: don't repeat the exact same alert within SNOOZE_SECONDS.
    if msg == _last_alert_text[0] and (now - _last_alert_at[0]) < SNOOZE_SECONDS:
        return True, "snoozed"
    _last_alert_text[0] = msg
    _last_alert_at[0]   = now
    return True, msg


def _monitor_loop():  # pragma: no cover - non-terminating background daemon (sleeps INITIAL_DELAY then loops forever); its dispatch delegates to _check_once/_enqueue_speech, both unit-tested directly
    time.sleep(INITIAL_DELAY_SECONDS)
    while True:
        try:
            try:
                has_unread, payload = _check_once()
                if has_unread and payload not in ("no_unread", "snoozed"):
                    print(f"  [teams] nudging: {payload}")
                    _enqueue_speech(payload)
                elif has_unread and payload == "snoozed":
                    print(f"  [teams] unread detected — snoozed")
                else:
                    print(f"  [teams] clear")
            except Exception as e:
                print(f"  [teams] check error: {e}")
            time.sleep(CHECK_INTERVAL_SECONDS)
        except Exception:
            _log.exception("teams_nudge _monitor_loop iteration failed")
            time.sleep(CHECK_INTERVAL_SECONDS)


def register(actions):
    def check_teams(_: str = "") -> str:
        has_unread, count, sender = _ask_vision_for_teams_state()
        if not has_unread:
            return "No unread Teams messages visible, sir."
        return _build_message(count, sender)

    actions["check_teams"] = check_teams

    t = threading.Thread(target=_monitor_loop, daemon=True)
    t.start()
    print(
        f"  [teams] background nudger active — every "
        f"{CHECK_INTERVAL_SECONDS}s, snooze {SNOOZE_SECONDS}s"
    )
