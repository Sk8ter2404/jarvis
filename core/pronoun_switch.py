"""core/pronoun_switch.py — "turn it off" with nothing for "it" to mean.

Live 2026-10-01: "Jarvis, turn it off" reached the local model with no device
or media in play. The model invented [ACTION: shutdown], and the action-name
corrector mapped that onto shutdown_jarvis - JARVIS shut itself down. The
corrector now never guesses onto a protected action (command_autocorrect
``protected=``); this module keeps the model from being asked to guess at all
when there is no referent, and words the question JARVIS asks instead.

What JARVIS tracks that can give "it" a referent (checked 2026-10-01; there is
NO "last device controlled" record - the smart-home router keeps none):
  * the conversation itself: a previous owner turn moments ago (the prompt
    router then ships that turn's section, core/prompt_router follow-up
    routing, and the model resolves "it" against the history);
  * the action ledger (bobert_companion._action_history: the last executed
    actions, each stamped) - a device or media action just ran;
  * _jarvis_played_music_at - media JARVIS itself started, which may still be
    playing long after the turn that started it;
  * Kinect point-to-control, which resolves "turn THAT off" by where the owner
    points (skills/kinect_pointing via core/smart_home_router).
With any of those, the turn goes to the model as before. With none, JARVIS
asks "Turn what off, sir?" and calls no model.

``switch_state(text)`` recognises a WHOLE-utterance pronoun on/off command;
``clarifying_question(text)`` is the line to speak; ``referent_question(...)``
is the decision. Pure stdlib, no I/O, never raises.
"""
from __future__ import annotations

import re
from typing import Optional

from core.date_math import normalize

# A turn or an action this recent is something "it" can point back at. Same
# horizon as "do that again" (core/actions._REPLAY_MAX_AGE_S), the other
# pronoun reference to a previous action.
REFERENT_WINDOW_S = 120.0
# Media JARVIS started is still "it" while it plausibly plays on.
MEDIA_REFERENT_WINDOW_S = 30 * 60.0

GENERIC_QUESTION = ("I'd rather not guess at that one, sir. "
                    "What would you like me to do?")

_PRONOUN = r"(?:it|that|this|them|those|these)"
_ONE = r"(?:\s+ones?)?"
# On normalize()d text (wake word, "please", "sir", "now", punctuation gone).
# Whole utterance only: "turn off the lamp", "turn it off and on again",
# "turn it off in ten minutes" and "shut it down" never match.
_SWITCH_RE = re.compile(
    r"^(?:(?:can|could|would|will) you (?:please )?)?"
    r"(?P<verb>turn|switch|shut|flip)\s+"
    r"(?:" + _PRONOUN + _ONE + r"\s+(?P<st1>off|on)"
    r"|(?P<st2>off|on)\s+(?:that|this|those|these)" + _ONE + r")$")


def _match(text):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return _SWITCH_RE.match(normalize(text))
    except Exception:
        return None


def switch_state(text) -> Optional[str]:
    """'on' / 'off' when ``text`` is a bare pronoun on/off command ("Jarvis,
    turn it off.", "switch that on", "turn off that one"), else None."""
    m = _match(text)
    if m is None:
        return None
    return m.group("st1") or m.group("st2")


def clarifying_question(text) -> str:
    """The short question for a command JARVIS will not guess at: "Turn what
    off, sir?" (verb and state mirrored) for a pronoun on/off command, else
    GENERIC_QUESTION."""
    m = _match(text)
    if m is None:
        return GENERIC_QUESTION
    state = m.group("st1") or m.group("st2")
    return f"{m.group('verb').capitalize()} what {state}, sir?"


def _recent(age, window: float) -> bool:
    """A real, non-negative age inside ``window``. Anything else (None, junk,
    a negative age from a clock step) is no context: asking is the safe
    side."""
    if isinstance(age, bool) or not isinstance(age, (int, float)):
        return False
    return 0.0 <= float(age) <= window


def referent_question(text, *, prior_turn_age_s=None, last_action_age_s=None,
                      media_age_s=None, pointing_enabled=False
                      ) -> Optional[str]:
    """The question to ask INSTEAD of calling the model, or None to let the
    turn route as before.

    Asks only when ``text`` is a bare pronoun on/off command AND nothing gives
    "it" a referent: point-to-control is off, the previous owner turn and the
    last executed action are older than REFERENT_WINDOW_S (or absent), and
    media JARVIS started is older than MEDIA_REFERENT_WINDOW_S (or absent).
    Ages are seconds; None = none this process."""
    if switch_state(text) is None:
        return None
    if pointing_enabled:
        return None
    if (_recent(prior_turn_age_s, REFERENT_WINDOW_S)
            or _recent(last_action_age_s, REFERENT_WINDOW_S)
            or _recent(media_age_s, MEDIA_REFERENT_WINDOW_S)):
        return None
    return clarifying_question(text)
