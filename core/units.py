"""
Unit conversion helpers for JARVIS's SPOKEN/displayed output.

The owner thinks in US imperial for everyday distances — feet and yards — and
only in metric for 3D printing (millimetres). The LLM already gets that policy
via core/prompts.py, but several code paths build distance strings DIRECTLY and
speak them through proactive_announce / voice-action returns, bypassing the LLM.
Those paths call meters_to_imperial_phrase() so a Kinect proximity reading comes
out as "about 8 feet" rather than "about 2.5 metres". 2026-07-08.
"""
from __future__ import annotations

_FEET_PER_METRE = 3.28084


def meters_to_imperial_phrase(metres) -> str:
    """A natural spoken imperial distance for a metric source value.

    Feet for anything under ~10 ft (a person in the same room), yards beyond
    that. Grammatical singular/plural. Returns "" for a missing/garbage value so
    callers can drop the clause cleanly (matching how they already treat a None
    distance). Examples: 0.6 m -> "2 feet", 2.5 m -> "8 feet", 4.0 m -> "4 yards".
    """
    try:
        m = float(metres)
    except (TypeError, ValueError):
        return ""
    if m <= 0:
        return ""
    feet = m * _FEET_PER_METRE
    if feet < 1.5:
        return "about a foot"
    if feet < 10.0:
        n = int(round(feet))
        return f"{n} foot" if n == 1 else f"{n} feet"
    n = int(round(feet / 3.0))
    return f"{n} yard" if n == 1 else f"{n} yards"
