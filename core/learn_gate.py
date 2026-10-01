"""core/learn_gate.py — who may teach JARVIS (owner-only learning).

2026-09-30: with no wake word required, JARVIS answers whoever is talking in
the room, and every automated learner treated an ANSWERED turn as the owner's.
Another person's phone call taught him ten "facts about the user" in one
afternoon. With ``LEARN_ONLY_FROM_OWNER`` on, a turn may teach only when it was
clearly the owner's:

  * typed / injected (the web page, the tray, say_to_jarvis) — operator input;
  * led by the wake word ("JARVIS, ..."), or right after a standby wake;
  * the owner's enrolled voiceprint matched (core/voice_id, a speaker with the
    ``memory_write`` permission);
  * a follow-up inside a conversation one of the above opened (``window_s``
    after the last such turn).

A voice that is confidently NOT the owner never teaches, even led by the wake
word or inside the window. Overheard speech (the ambient learners) may teach
only with a matched owner voiceprint, and never opens the window.

Unlike core/followup_window.py (KNOWN RISK there: every admitted utterance
extends its window, so crosstalk can hold it open), only a POSITIVE turn —
typed, wake word, owner voice — opens or extends this window. A turn admitted
by the window alone never does, so it closes ``window_s`` after the owner was
last positively identified, however chatty the room is.

Pure stdlib, no I/O, caller-supplied timestamps — testable on the light CI.
"""
from __future__ import annotations

import math

# Voice verdicts for one utterance.
OWNER = "owner"              # an enrolled speaker who may write memory matched
NOT_OWNER = "not_owner"      # confidently someone else
UNSURE = "unsure"            # between the reject floor and the match threshold
UNAVAILABLE = "unavailable"  # nobody enrolled, no encoder, no / too-short audio

VERDICTS = (OWNER, NOT_OWNER, UNSURE, UNAVAILABLE)


def voice_verdict(name, score, *, enrolled: bool, may_write: bool = True,
                  reject_below: float = 0.60) -> str:
    """Map core.voice_id.identify_speaker's ``(name, score)`` to a verdict.

    ``name`` is set only above voice_id's match threshold. Below it the score
    still says how close the best match was: under ``reject_below`` the voice
    is someone else; between the two it is too close to call. A score of 0 (or
    junk) means no embedding could be computed, which is no evidence at all.
    ``may_write`` False (an enrolled guest without ``memory_write``) makes a
    match NOT_OWNER: that person may talk to JARVIS but not teach him."""
    if not enrolled:
        return UNAVAILABLE
    if name:
        return OWNER if may_write else NOT_OWNER
    try:
        s = float(score)
    except (TypeError, ValueError):
        return UNAVAILABLE
    if not math.isfinite(s) or s <= 0.0:
        return UNAVAILABLE
    try:
        floor = float(reject_below)
    except (TypeError, ValueError):
        floor = 0.60
    return NOT_OWNER if s < floor else UNSURE


class LearnGate:
    """Decides, turn by turn and in order, whether a turn may teach.

    Feed it every turn (and every standby wake) in the order they happened,
    with the time each one happened; the window is computed from those
    timestamps, so a caller that classifies later (a background thread)
    gets the same answers as one that classifies at once."""

    def __init__(self, window_s: float = 90.0) -> None:
        try:
            w = float(window_s)
        except (TypeError, ValueError):
            w = 0.0                      # a malformed setting disables, never widens
        self.window_s = w if math.isfinite(w) and w > 0.0 else 0.0
        self._last_positive = None       # ts of the last positively-owner turn

    def _open(self, ts: float) -> None:
        if self._last_positive is None or ts >= self._last_positive:
            self._last_positive = ts

    def _in_window(self, ts: float) -> bool:
        if not self.window_s or self._last_positive is None:
            return False
        return 0.0 <= ts - self._last_positive <= self.window_s

    def note_wake(self, ts: float) -> None:
        """A standby wake word ("JARVIS" alone): the owner is starting a
        conversation, so the next turns are follow-ups."""
        self._open(float(ts))

    def decide(self, ts: float, *, injected: bool = False, wake: bool = False,
               voice: str = UNAVAILABLE,
               overheard: bool = False) -> tuple[bool, str]:
        """``(may_teach, reason)`` for one turn. The reason is a short phrase
        for the log; it never contains the turn's text."""
        ts = float(ts)
        if injected:
            self._open(ts)
            return True, "typed"
        if voice == NOT_OWNER:
            return False, "not the owner's voice"
        if overheard:
            if voice == OWNER:
                return True, "owner's voice (overheard)"
            return False, "overheard, owner's voice not confirmed"
        if wake:
            self._open(ts)
            return True, "wake word"
        if voice == OWNER:
            self._open(ts)
            return True, "owner's voice"
        if self._in_window(ts):
            return True, "follow-up in the owner's conversation"
        return False, "not addressed by the owner"
