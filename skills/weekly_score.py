"""Voice-action skill: the weekly success score.

    "JARVIS, how did you do this week?"   -> weekly_score

Counts the last seven days of logs/session_*.log (turns, answers, actions run
and failed, median time to first sound) via tools/weekly_score.py, writes
data/weekly_score.json and speaks a short summary. Local only: no network, no
conversation text is read out or stored.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEAK_VERBATIM_ACTIONS = ("weekly_score", "how_did_you_do", "success_score")


def weekly_score(_: str = "") -> str:
    """Compute, save and return the spoken weekly score."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    try:
        from tools import weekly_score as ws
        res = ws.build()
        try:
            ws.write_json(res)
        except OSError:
            pass
        return ws.summary_text(res)
    except Exception as exc:
        return f"I could not work out the weekly score, sir: {type(exc).__name__}."


def register(actions: dict) -> None:
    actions["weekly_score"] = weekly_score
    actions["how_did_you_do"] = weekly_score
    actions["success_score"] = weekly_score
