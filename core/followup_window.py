"""core/followup_window.py — the wake-word follow-up window.

Wake-word mode (config ``REQUIRE_WAKE_MODE``) makes every turn begin with
"JARVIS", which is right in a noisy room and exhausting in a real back-and-forth.
This window relaxes it for a short time AFTER the user has addressed JARVIS
directly: follow-ups inside the window need no wake word, and each one that is
admitted extends the window. When the conversation stops, the window lapses and
overheard room crosstalk is gated out again.

Ported 2026-09-28 from a local-only patch on the Dell edge node (where it ran at
45 s). On main it is OFF by default (``FOLLOWUP_WINDOW_S = 0``) so the desktop's
wake-word behaviour is unchanged; an install opts in through
``data/user_settings.json``.

KNOWN RISK — read before raising the window
-------------------------------------------
Every ADMITTED utterance extends the window, and admission happens at the
background-audio gate, before the speech filter or the LLM has judged it. So in
exactly the room that needed wake-word mode, one overheard line inside the window
extends it for the next, and a steady source such as a TV can hold it open
indefinitely. The window is only as strict as the room is quiet. This port keeps
the Dell's semantics unchanged; capping the total extension is a policy choice
left to the owner.

Pure stdlib, no I/O, injectable clock — testable on the light-deps CI runner.
"""
from __future__ import annotations

import time


class FollowupWindow:
    """Tracks whether a no-wake-word follow-up is currently admissible.

    ``window_s <= 0`` disables it entirely: ``admit()`` is then always False and
    the caller's gate behaves exactly as it did before this module existed.
    """

    def __init__(self, window_s: float = 0.0, clock=time.time) -> None:
        try:
            self.window_s = max(0.0, float(window_s))
        except (TypeError, ValueError):
            self.window_s = 0.0          # a malformed setting disables, never widens
        self._clock = clock
        self._until = 0.0

    @property
    def enabled(self) -> bool:
        return self.window_s > 0.0

    def note_addressed(self) -> None:
        """The user just addressed JARVIS by its wake word: (re)open the window."""
        if self.enabled:
            self._until = self._clock() + self.window_s

    def admit(self) -> bool:
        """True if a no-wake-word utterance is admissible right now.

        An admitted utterance EXTENDS the window (see KNOWN RISK above).
        """
        if not self.enabled:
            return False
        now = self._clock()
        if now < self._until:
            self._until = now + self.window_s
            return True
        return False

    def remaining_s(self) -> float:
        """Seconds left in the window (0.0 when closed or disabled)."""
        if not self.enabled:
            return 0.0
        return max(0.0, self._until - self._clock())
