"""
Timesheet-hours reminder.

Pay periods run Monday-Sunday, two weeks at a time, and the hours for a period
can only be entered once it has ended. On the Monday a period ends JARVIS says
"Time to put in your hours, sir, for Monday, September 28 to Sunday, October
11." The calendar (core/hours_calendar.py) is anchored on one known period
start, HOURS_REMINDER_ANCHOR, so the same maths answers "is it hours week?" and
"when do I put in my hours".

Actions added:
  hours_status            "is it hours week?" / "when do I put in my hours"
  hours_reminder_fire     say the reminder now (the Monday job runs this; the
                          job speaks only in an hours week, and only once per
                          period, so a restart never repeats it)

Settings (core/config.py): HOURS_REMINDER_ENABLED, HOURS_REMINDER_ANCHOR,
HOURS_REMINDER_TIME.

How it runs: a weekly Monday cron job on core.scheduler (persistent, so it
survives restarts) plus one short catch-up job a couple of minutes after boot
when this week's reminder is due and has not been said (JARVIS was off at the
reminder time). Neither blocks the voice loop; the line goes through the same
speech queue as every other scheduled reminder. With no APScheduler a daemon
thread polls instead. State: data/hours_reminder.json (the last period
reminded), resolved through core.paths.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timedelta

from core import hours_calendar as _cal

SPEAK_VERBATIM_ACTIONS = ("hours_status",)

_CRON_JOB_ID = "hours_reminder_monday"
_CATCHUP_JOB_ID = "hours_reminder_catchup"
_STATE_NAME = "hours_reminder.json"
_CATCHUP_DELAY_S = 120
_POLL_S = 300
_ARM_WAIT_S = 90
# A catch-up reminder is not spoken in the small hours.
_LATEST_HOUR = 21

_lock = threading.Lock()
_poll_thread = None


def _cfg(name: str, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _anchor():
    return _cal.parse_anchor(_cfg("HOURS_REMINDER_ANCHOR", _cal.DEFAULT_ANCHOR))


def _enabled() -> bool:
    return bool(_cfg("HOURS_REMINDER_ENABLED", True))


def _clock() -> tuple:
    return _cal.parse_clock(_cfg("HOURS_REMINDER_TIME", _cal.DEFAULT_TIME))


def _now() -> datetime:
    return datetime.now()


def _state_path() -> str:
    from core import paths
    return paths.data_file(_STATE_NAME)


def _last_reminded() -> str:
    try:
        with open(_state_path(), "r", encoding="utf-8") as f:
            return str(json.load(f).get("last_reminded", ""))
    except Exception:
        return ""


def _mark_reminded(day_iso: str) -> None:
    try:
        from core.atomic_io import _atomic_write_json
        _atomic_write_json(_state_path(), {"last_reminded": day_iso})
    except Exception as e:
        print(f"  [hours] could not save reminder state ({e})")


def _due_now(now: datetime):
    """The hours-day Monday whose reminder is due and unsaid as of ``now``,
    else None: an hours week, at/after the reminder time, before the evening
    cut-off, and not already reminded for this period."""
    monday = _cal.hours_week_monday(now.date(), _anchor())
    if monday is None:
        return None
    h, m = _clock()
    if (now.hour, now.minute) < (h, m) or now.hour >= max(_LATEST_HOUR, h + 1):
        return None
    if _last_reminded() == monday.isoformat():
        return None
    return monday


def _speak(actions: dict, text: str) -> bool:
    """Queue ``text`` through the scheduled-reminder speech path. False when
    nothing could take it (the caller then leaves the period unmarked, so the
    next catch-up or poll tries again)."""
    say = actions.get("say_aloud")
    if callable(say):
        try:
            res = say(text)
            return isinstance(res, str) and res.startswith("Queued")
        except Exception as e:
            print(f"  [hours] say_aloud failed ({e})")
            return False
    for name in ("bobert_companion", "__main__"):
        mod = sys.modules.get(name)
        announcer = getattr(mod, "proactive_announce", None) if mod else None
        if callable(announcer):
            try:
                return bool(announcer(text, source="schedule"))
            except Exception as e:
                print(f"  [hours] announce failed ({e})")
                return False
    print("  [hours] no speech queue available - reminder not spoken")
    return False


def _fire(actions: dict, arg: str = "") -> str:
    explicit = (arg or "").strip().lower() in ("now", "force", "test")
    now = _now()
    with _lock:
        if explicit:
            monday = _cal.hours_week_monday(now.date(), _anchor())
            text = (_cal.reminder_line(monday) if monday is not None
                    else _cal.status_line(now.date(), _anchor()))
            ok = _speak(actions, text)
            return text if ok else "I couldn't queue the hours reminder, sir."
        if not _enabled():
            return ""
        monday = _due_now(now)
        if monday is None:
            return ""
        text = _cal.reminder_line(monday)
        if not _speak(actions, text):
            print("  [hours] reminder due but could not be spoken; will retry")
            return ""
        _mark_reminded(monday.isoformat())
        print(f"  [hours] reminded for the period ending "
              f"{(monday - timedelta(days=1)).isoformat()}")
        return text


def _poll_loop(actions: dict) -> None:
    while True:
        time.sleep(_POLL_S)
        try:
            _fire(actions, "")
        except Exception as e:
            print(f"  [hours] poll failed ({e})")


def _arm_with(scheduler, actions: dict) -> str:
    """Arm the Monday job on a bootstrapped ``scheduler`` (or remove it when
    the setting is off). Returns a one-line status for the boot log."""
    if not _enabled():
        for jid in (_CRON_JOB_ID, _CATCHUP_JOB_ID):
            try:
                scheduler.cancel_job(jid)
            except Exception:
                pass
        return "disabled (HOURS_REMINDER_ENABLED is off)"
    h, m = _clock()
    try:
        scheduler.schedule_cron(action="hours_reminder_fire", arg="",
                                job_id=_CRON_JOB_ID, hour=h, minute=m,
                                day_of_week="mon")
        if _due_now(_now()) is not None:
            scheduler.schedule_once(
                action="hours_reminder_fire", arg="", job_id=_CATCHUP_JOB_ID,
                run_at=_now() + timedelta(seconds=_CATCHUP_DELAY_S))
    except Exception as e:
        return f"could not arm the Monday job ({type(e).__name__}: {e})"
    return f"armed Mondays {h:02d}:{m:02d}"


def _start_polling(actions: dict) -> None:
    global _poll_thread
    if not _enabled():
        return
    if _poll_thread is None or not _poll_thread.is_alive():
        _poll_thread = threading.Thread(
            target=_poll_loop, args=(actions,), name="hours-reminder",
            daemon=True)
        _poll_thread.start()


def _arm_when_ready(actions: dict) -> None:
    """Background: wait (up to _ARM_WAIT_S) for core.scheduler to be
    bootstrapped by skills/schedule_manager - skills load in file order and
    this one loads first - then arm. With no scheduler, or none in time, poll
    instead so the reminder still happens. Never raises, never blocks boot."""
    try:
        from core import scheduler
        if not scheduler.is_available():
            print("  [hours] APScheduler unavailable - polling instead")
            _start_polling(actions)
            return
        deadline = time.time() + _ARM_WAIT_S
        while time.time() < deadline:
            if scheduler.status().get("running"):
                print(f"  [hours] reminder {_arm_with(scheduler, actions)}")
                return
            time.sleep(3)
        print("  [hours] scheduler never came up - polling instead")
        _start_polling(actions)
    except Exception as e:
        print(f"  [hours] could not arm the reminder ({e}) - polling instead")
        _start_polling(actions)


def register(actions: dict) -> None:
    def hours_status(_: str = "") -> str:
        return _cal.status_line(_now().date(), _anchor())

    def hours_reminder_fire(arg: str = "") -> str:
        return _fire(actions, arg)

    actions["hours_status"] = hours_status
    actions["hours_reminder_fire"] = hours_reminder_fire
    threading.Thread(target=_arm_when_ready, args=(actions,),
                     name="hours-reminder-arm", daemon=True).start()
