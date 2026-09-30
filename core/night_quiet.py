"""NIGHT_QUIET_ENABLED: the one switch over every clock-driven night quieting.

With it on (the shipped default) JARVIS changes how he sounds at night purely
because of the clock:

  * core/tts.py            'hushed_late' preset (gain 0.55) from 23:00, and for
                           an hour after the anticipation engine's late_hour
                           nudge (anticipation_state.json).
  * core/tone_detector.py  the 'late_night' time-of-day tone (22:00-04:59),
                           which core/voice_emotion.py turns into the
                           late_night mood: a quieter preset (gain 0.65) and a
                           one-sentence prompt addendum.
  * core/emotion_tracker.py the time-only 'tired' label (22:00-04:59), which
                           maps to the 'concerned' preset (gain 0.92).
  * core/prompts.py        the system prompt's two clock-only rules: "LATE
                           HOUR" (22:00-05:59 = quietly stressed, one
                           sentence) and "daylight hours" as a condition for
                           leaving that register (base_system_prompt()).
  * skills/anticipation_engine.py the late_hour nudge ("We've been at this a
                           while"), and the 23:00-06:59 hold on every other
                           anticipation line once the owner has been silent
                           for 30 minutes (_should_skip_late_night).
  * skills/night_owl_mode.py the automatic 23:00 switch-on (NIGHT_OWL_AUTO is
                           the finer knob for that one).
  * bobert_companion.py    the softer wake greeting (volume 0.85, 22:00-04:59),
                           the "Still up, sir?" wake greeting (01:00-04:59)
                           and the spoken late-night remark (01:00-04:59).

With it off his voice, reply length and proactive speech at 03:00 are what
they are at 15:00. Anything the owner asks for himself still works: "night
owl on", or saying he is tired.

Deliberately NOT gated (not night quieting): time-of-day habits (pattern
offers, the 19:00-03:59 "wind down with something on Netflix?" suggestion,
music-by-time-of-day), the morning briefing mentioning a late previous
session, and the persona's "Notice the hour" trait, which is a remark, not a
quieter or shorter voice.

Read at CALL time from core.config (never captured at import), so tests can
patch core.config.NIGHT_QUIET_ENABLED and a saved setting applies on the next
start like every other knob. Any trouble reading it keeps the old behaviour
(on). Stdlib-only; imported by core/emotion_tracker.py at import time and
lazily by core/tts.py, core/tone_detector.py and core/prompts.py (so the first
two still run as scripts).
"""
from __future__ import annotations


def _knob(name: str) -> bool:
    """A boolean core.config knob that defaults to True; True on any error."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, True))
    except Exception:
        return True


def night_quiet_enabled() -> bool:
    """True when JARVIS may quiet himself at night because of the clock."""
    return _knob("NIGHT_QUIET_ENABLED")


def night_owl_auto_enabled() -> bool:
    """True when night-owl mode may switch itself on at the start of the
    night window. Both NIGHT_QUIET_ENABLED and NIGHT_OWL_AUTO must be on."""
    return night_quiet_enabled() and _knob("NIGHT_OWL_AUTO")
