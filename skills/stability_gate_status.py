"""Voice-action skill: report the most recent stability-gate verdict.

The upgrade pipeline (upgrade_jarvis._stability_gate) appends one JSON line
per gate run to data/stability_gates.jsonl. This skill reads the freshest
line and speaks a one-sentence summary so the user can ask:

    "JARVIS, what was the last stability gate result?"
    "JARVIS, stability gate status"

If the log is missing / empty (gate never run), returns a graceful message.
"""
from __future__ import annotations

import json
import os
import time

_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "stability_gates.jsonl",
)


def _read_last_record() -> dict | None:
    if not os.path.exists(_LOG_PATH):
        return None
    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            last_line = ""
            for line in f:
                if line.strip():
                    last_line = line.strip()
        if not last_line:
            return None
        return json.loads(last_line)
    except (OSError, ValueError):
        return None


def _friendly_age(ts_iso: str) -> str:
    """'14 minutes ago' / 'just now' / 'about 3 hours ago'."""
    try:
        rec_t = time.mktime(time.strptime(ts_iso, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return ts_iso
    delta = max(0, int(time.time() - rec_t))
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{delta // 60} minute(s) ago"
    if delta < 86400:
        return f"about {delta // 3600} hour(s) ago"
    return f"{delta // 86400} day(s) ago"


def last_stability_gate(_: str = "") -> str:
    """Return a one-sentence summary of the most recent stability gate."""
    rec = _read_last_record()
    if rec is None:
        return ("I haven't run a stability gate yet — the upgrade pipeline "
                "fires one after every batch of completed tasks.")
    verdict = str(rec.get("verdict", "?"))
    batch   = rec.get("batch", "?")
    ts      = str(rec.get("ts", ""))
    age     = _friendly_age(ts) if ts else "(unknown time)"

    if verdict == "PASS":
        dur = rec.get("duration_s", "?")
        return (f"Last stability gate: batch {batch} PASSED {age} "
                f"after a {dur}-second smoke test.")
    if verdict == "FAIL":
        detail = (rec.get("smoke_error")
                  or (rec.get("smoke_stdout_tail", "") or "").strip()
                  or "see data/stability_gates.jsonl")
        return (f"Last stability gate: batch {batch} FAILED {age}. "
                f"Symptom: {str(detail)[:200]}. The pipeline auto-reverted "
                f"and queued a regression task.")
    if verdict == "SKIP":
        reason = rec.get("reason", "skipped")
        return f"Last stability gate (batch {batch}) was SKIPPED {age}: {reason}."
    return f"Last stability gate (batch {batch}) verdict: {verdict} {age}."


# Aliases for natural phrasing variations.
stability_gate_status        = last_stability_gate
last_gate_result             = last_stability_gate
last_stability_gate_result   = last_stability_gate


def register(actions: dict) -> None:
    actions["last_stability_gate"]        = last_stability_gate
    actions["last_stability_gate_result"] = last_stability_gate
    actions["last_gate_result"]           = last_stability_gate
    actions["stability_gate_status"]      = stability_gate_status
    actions["gate_status"]                = stability_gate_status
