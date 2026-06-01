"""
REPO Robot companion skill for JARVIS.

Tracks the state of the physical REPO Robot project across three sources:

  1. data/repo_robot_state.json — manually edited authoritative state
     (next step, blockers, parts on order, last firmware flash, recent
     progress notes). The file is hot-reloaded on every action call so the
     user can edit it without restarting JARVIS.
  2. jarvis_todo.md — counts robot-related pending vs. done tasks via a
     keyword scan. Picks up tasks the user has queued for Claude Code that
     mention the robot, the ESP32, OTA, servos, wiring, eyes, etc.
  3. Recent log files (logs/*.log) — scans the most recent log for any
     robot/ESP32-related mentions in the last 24 h as a rough proxy for
     "what has JARVIS been doing on this project lately".

Actions registered:
  robot_status      — one-line summary: next step + part/blocker count.
  robot_blocker     — list current unresolved blockers.
  next_robot_step   — what the user should do next.

Morning-briefing volunteer:
  If the state file shows a part has just arrived (eta on or before today,
  not yet flagged 'arrived') OR a blocker was recently resolved AND there's
  still an open next_step, exposes `get_morning_volunteer_text()` for the
  morning_briefing skill to splice into the briefing. Example:

    "The REPO Robot's SG90 servo arrived yesterday, sir — shall I queue
     the wiring task?"
"""
from __future__ import annotations

import datetime
import json
import os
import re
import time
from typing import Any

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_FILE  = os.path.join(_PROJECT_DIR, "data", "repo_robot_state.json")
_TODO_FILE   = os.path.join(_PROJECT_DIR, "jarvis_todo.md")
_LOGS_DIR    = os.path.join(_PROJECT_DIR, "logs")

# Words that flag a line as robot-related. Kept conservative: 'eye' alone
# matches too much ('I see'), so we use 'eye servo' / 'robot eye' below.
_ROBOT_KEYWORDS = (
    "repo robot", "repo-robot", "reporobot",
    "esp32", "esp-32", "ota",
    "servo", "wiring", "bobert_code", "robot eye",
    "animatronic", "robot arm", "robot head",
)

# Only consider log lines from the last N hours when reporting "recent
# activity". Longer than 24 h would surface stale context in the morning.
_RECENT_HOURS = 24

# A 'recently arrived' part is one whose ETA is in the last 7 days but
# arrived flag hasn't been flipped yet. Anything older we assume the user
# already saw the morning remark for.
_PART_FRESH_DAYS = 7


# ── state loading ────────────────────────────────────────────────────────
def _load_state() -> dict[str, Any]:
    if not os.path.exists(_STATE_FILE):
        return {}
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"  [repo_robot] state read failed: {e}")
        return {}


def _save_state(state: dict[str, Any]) -> bool:
    state["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        tmp = _STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, _STATE_FILE)
        return True
    except Exception as e:
        print(f"  [repo_robot] state write failed: {e}")
        return False


# ── parsing helpers ──────────────────────────────────────────────────────
def _parse_date(s: str | None) -> datetime.date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.date.fromisoformat(s[:10])
    except Exception:
        return None


def _days_ago(d: datetime.date) -> int:
    return (datetime.date.today() - d).days


def _natural_days(d: datetime.date) -> str:
    n = _days_ago(d)
    if n == 0:
        return "today"
    if n == 1:
        return "yesterday"
    if n < 0:
        return f"in {-n} day{'s' if -n != 1 else ''}"
    return f"{n} days ago"


# ── todo/log scanning ────────────────────────────────────────────────────
def _is_robot_line(line: str) -> bool:
    s = line.lower()
    return any(k in s for k in _ROBOT_KEYWORDS)


def _count_todo_tasks() -> tuple[int, int]:
    """Return (pending_robot, done_robot) counts from jarvis_todo.md."""
    if not os.path.exists(_TODO_FILE):
        return 0, 0
    pending = done = 0
    try:
        with open(_TODO_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not _is_robot_line(line):
                    continue
                stripped = line.lstrip()
                if stripped.startswith("- [ ]"):
                    pending += 1
                elif stripped.startswith("- [x]") or stripped.startswith("- [X]"):
                    done += 1
    except Exception as e:
        print(f"  [repo_robot] todo scan failed: {e}")
    return pending, done


def _recent_log_mentions() -> int:
    """Return count of robot-related lines in the most recent log file
    within the last _RECENT_HOURS hours."""
    if not os.path.isdir(_LOGS_DIR):
        return 0
    cutoff = time.time() - (_RECENT_HOURS * 3600)
    count = 0
    try:
        for name in os.listdir(_LOGS_DIR):
            if not name.endswith(".log"):
                continue
            path = os.path.join(_LOGS_DIR, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if _is_robot_line(line):
                            count += 1
            except Exception:
                continue
    except Exception as e:
        print(f"  [repo_robot] log scan failed: {e}")
    return count


# ── derived views over state ─────────────────────────────────────────────
def _open_blockers(state: dict[str, Any]) -> list[dict]:
    raw = state.get("blockers") or []
    return [b for b in raw if isinstance(b, dict) and not b.get("resolved")]


def _just_arrived_parts(state: dict[str, Any]) -> list[dict]:
    """Parts whose ETA has passed but 'arrived' flag isn't set yet, within
    the freshness window. These are what trigger the morning volunteer."""
    today = datetime.date.today()
    result = []
    for part in state.get("parts_on_order") or []:
        if not isinstance(part, dict) or part.get("arrived"):
            continue
        eta = _parse_date(part.get("eta"))
        if not eta:
            continue
        delta = (today - eta).days
        if 0 <= delta <= _PART_FRESH_DAYS:
            result.append(part)
    return result


# ── action handlers ──────────────────────────────────────────────────────
def _act_robot_status(_: str = "") -> str:
    state = _load_state()
    if not state:
        return ("REPO Robot state file is empty, sir. Edit "
                "data/repo_robot_state.json to seed it.")

    next_step = (state.get("next_step") or "").strip()
    blockers = _open_blockers(state)
    parts    = state.get("parts_on_order") or []
    pending, done = _count_todo_tasks()
    log_hits = _recent_log_mentions()

    bits: list[str] = []
    if next_step:
        bits.append(f"next step: {next_step}")
    if blockers:
        bits.append(f"{len(blockers)} blocker{'s' if len(blockers) != 1 else ''}")
    open_parts = [p for p in parts if isinstance(p, dict) and not p.get("arrived")]
    if open_parts:
        bits.append(f"{len(open_parts)} part{'s' if len(open_parts) != 1 else ''} on order")
    if pending:
        bits.append(f"{pending} todo item{'s' if pending != 1 else ''} pending")

    flash = state.get("last_firmware_flash") or {}
    flash_date = _parse_date(flash.get("at"))
    if flash_date:
        bits.append(f"last flash {_natural_days(flash_date)}")

    lead = "REPO Robot, sir: " if bits else "REPO Robot, sir — "
    body = "; ".join(bits) if bits else "no tracked state yet"

    tail = ""
    if log_hits:
        tail = f" Recent log mentions: {log_hits}."
    if done:
        tail += f" {done} robot task{'s' if done != 1 else ''} completed."
    return f"{lead}{body}.{tail}".strip()


def _act_robot_blocker(_: str = "") -> str:
    state = _load_state()
    blockers = _open_blockers(state)
    if not blockers:
        return "No active blockers on the REPO Robot, sir."
    if len(blockers) == 1:
        b = blockers[0]
        text = (b.get("text") or "").strip() or "unspecified"
        since = _parse_date(b.get("since"))
        when = f" — since {_natural_days(since)}" if since else ""
        return f"One blocker, sir: {text}{when}."
    lines = []
    for b in blockers:
        text = (b.get("text") or "").strip() or "unspecified"
        since = _parse_date(b.get("since"))
        when = f" (since {_natural_days(since)})" if since else ""
        lines.append(f"{text}{when}")
    return f"{len(blockers)} blockers, sir: " + "; ".join(lines) + "."


def _act_next_robot_step(_: str = "") -> str:
    state = _load_state()
    next_step = (state.get("next_step") or "").strip()
    if not next_step:
        return ("No next step recorded, sir. Set 'next_step' in "
                "data/repo_robot_state.json.")
    blockers = _open_blockers(state)
    if blockers:
        return (f"Next, sir: {next_step}. Mind the {len(blockers)} open "
                f"blocker{'s' if len(blockers) != 1 else ''} first.")
    return f"Next, sir: {next_step}."


# ── morning briefing volunteer ───────────────────────────────────────────
def get_morning_volunteer_text() -> str:
    """Called by skills/morning_briefing.py. Returns a one-line JARVIS-style
    remark when there's actionable robot progress to flag, otherwise ''.

    Triggers:
      • A part on order whose ETA has just passed and arrived flag isn't set.
      • A blocker resolved in the last 3 days while next_step is still open.
    Never fires twice in the same calendar day (cached in the state file's
    'last_volunteered_on' field)."""
    state = _load_state()
    if not state:
        return ""

    today_str = datetime.date.today().isoformat()
    if state.get("last_volunteered_on") == today_str:
        return ""

    next_step = (state.get("next_step") or "").strip()
    if not next_step:
        return ""

    # Priority 1: a part that just arrived
    arrived = _just_arrived_parts(state)
    if arrived:
        part = arrived[0]
        name = (part.get("part") or "part").strip()
        eta = _parse_date(part.get("eta"))
        when = _natural_days(eta) if eta else "recently"
        remark = (f"The REPO Robot's {name} arrived {when}, sir — "
                  f"shall I queue the next task?")
        state["last_volunteered_on"] = today_str
        _save_state(state)
        return remark

    # Priority 2: a recently resolved blocker
    for b in state.get("blockers") or []:
        if not isinstance(b, dict) or not b.get("resolved"):
            continue
        resolved_at = _parse_date(b.get("resolved_at"))
        if resolved_at and 0 <= _days_ago(resolved_at) <= 3:
            text = (b.get("text") or "previous blocker").strip()
            remark = (f"The REPO Robot blocker — {text} — is clear, sir. "
                      f"Shall I line up the next step?")
            state["last_volunteered_on"] = today_str
            _save_state(state)
            return remark

    return ""


def register(actions):
    actions["robot_status"]    = _act_robot_status
    actions["robot_blocker"]   = _act_robot_blocker
    actions["next_robot_step"] = _act_next_robot_step

    # Ensure the data dir exists so the morning hook can write its
    # last_volunteered_on stamp without races.
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
    except Exception:
        pass

    has_state = os.path.exists(_STATE_FILE)
    print(f"  [repo_robot] actions registered "
          f"({'state file present' if has_state else 'no state file yet'})")
