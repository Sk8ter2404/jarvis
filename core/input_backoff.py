"""core/input_backoff.py — back off when the microphone will not open.

WHY THIS MODULE EXISTS
======================
Live 2026-09-29 18:38:33: a USB hub reset removed the desk mic's endpoint, and
after PortAudio re-enumerated there was NO input device at all. Every
record_speech() call then failed at once — first on the cached index
(-9999 'A device ID has been used that is out of range'), then on its retry
with the system default ('Error querying device -1') — and handed None back to
the main loop, which called it again immediately. About 200 failures a second,
each with a two-part chained traceback: 5,859 failures and 11,718 tracebacks in
two 20-second bursts, a 20.7 MB session log. Nothing slowed the loop down; the
only thing that ever ended a burst was the next (rate-limited) PortAudio
re-enumeration finding the mic again.

This module is the pacing half of the fix: an exponential backoff between OPEN
ATTEMPTS (0.5 -> 1 -> 2 -> 5 s, capped), reset by the first successful open,
plus the bookkeeping for quiet logging — one line when an episode starts, a
summary every ``summary_s`` while it lasts, one line when it ends. The monolith
(bobert_companion.record_speech) owns everything with a side effect: the
interruptible wait, the device-return poll, the re-enumeration and the printing.

Pure stdlib. No clock of its own: every method takes ``now`` (the caller's
monotonic seconds), so tests drive it by hand. Thread-safe (one lock), and no
method raises on ordinary input.
"""
from __future__ import annotations

import threading
from typing import Optional

DEFAULT_STEPS_S = (0.5, 1.0, 2.0, 5.0)
DEFAULT_SUMMARY_S = 60.0


class InputOpenBackoff:
    """Backoff between capture-device open attempts.

    An EPISODE starts with the first failed attempt after a success (or boot)
    and ends with the next successful open. Attempt k of an episode (k = 1 for
    the first failure) is followed by a wait of ``steps[min(k, len) - 1]``
    seconds: 0.5, 1, 2, 5, 5, 5, ... with the defaults.
    """

    def __init__(self, steps=DEFAULT_STEPS_S,
                 summary_s: float = DEFAULT_SUMMARY_S) -> None:
        clean = []
        for s in (steps or ()):
            try:
                v = float(s)
            except (TypeError, ValueError):
                continue
            if v > 0:
                clean.append(v)
        self.steps = tuple(clean) or DEFAULT_STEPS_S
        try:
            self.summary_s = max(1.0, float(summary_s))
        except (TypeError, ValueError):
            self.summary_s = DEFAULT_SUMMARY_S
        self._lock = threading.Lock()
        self._reset_locked()

    # ── state ──────────────────────────────────────────────────────────
    def _reset_locked(self) -> None:
        self.fails = 0            # failed attempts in this episode (0 = healthy)
        self.next_at = 0.0        # earliest next attempt (caller's clock)
        self.since = 0.0          # when the episode started
        self.summary_at = 0.0     # when its last line (first or summary) printed
        self.no_device = False    # the latest failure: no input device at all
        self.error = ""           # the latest failure's message

    @property
    def active(self) -> bool:
        """True while an episode is open (at least one failure since the
        last success)."""
        return self.fails > 0

    @property
    def first_step(self) -> float:
        return self.steps[0]

    def delay_after(self, attempts: int) -> float:
        """The wait that follows failed attempt number ``attempts`` (1-based)."""
        i = max(1, int(attempts)) - 1
        return self.steps[min(i, len(self.steps) - 1)]

    def due(self, now: float) -> bool:
        """May an open attempt run at ``now``? Always True outside an episode."""
        with self._lock:
            return self.fails == 0 or now >= self.next_at

    def remaining(self, now: float) -> float:
        """Seconds until the next attempt is due (0.0 when due / healthy)."""
        with self._lock:
            if self.fails == 0:
                return 0.0
            return max(0.0, self.next_at - now)

    # ── transitions ────────────────────────────────────────────────────
    def note_failure(self, now: float, error: str = "",
                     no_device: bool = False) -> dict:
        """Record one failed attempt and schedule the next.

        Returns ``{"log": "first" | "summary" | None, "attempts": n,
        "elapsed": seconds since the episode began, "delay": the wait before
        the next attempt, "no_device": bool}``. ``log`` is "first" for the
        episode's first failure AND when the failure CLASS changes mid-episode
        (no-device <-> an open error on a device that exists: different
        information), "summary" once every ``summary_s`` otherwise, and None
        in between — the caller prints only when it is set."""
        with self._lock:
            first = self.fails == 0
            changed = (not first) and bool(no_device) != self.no_device
            if first:
                self.since = now
                self.summary_at = now
            self.fails += 1
            self.no_device = bool(no_device)
            self.error = str(error or "")[:200]
            delay = self.delay_after(self.fails)
            self.next_at = now + delay
            if first or changed:
                log = "first"
                self.summary_at = now
            elif now - self.summary_at >= self.summary_s:
                log = "summary"
                self.summary_at = now
            else:
                log = None
            return {"log": log, "attempts": self.fails,
                    "elapsed": max(0.0, now - self.since), "delay": delay,
                    "no_device": self.no_device}

    def pull_in(self, now: float, within_s: float) -> None:
        """Bring the next attempt forward to no later than ``now + within_s``
        — but never earlier than ``now + first_step``, so pulling in can never
        turn into a hot loop. Used when a re-enumeration becomes permissible
        before the backoff would next try. No-op outside an episode."""
        with self._lock:
            if self.fails == 0:
                return
            try:
                at = now + max(self.steps[0], float(within_s))
            except (TypeError, ValueError):
                return
            if at < self.next_at:
                self.next_at = at

    def expedite(self) -> None:
        """Make the next attempt due immediately (the device came back)."""
        with self._lock:
            if self.fails:
                self.next_at = 0.0

    def note_success(self, now: float) -> Optional[dict]:
        """A capture stream opened. Ends the episode and resets the backoff.
        Returns ``{"attempts": n, "elapsed": s, "no_device": bool}`` for the
        episode that just ended, or None when there was none."""
        with self._lock:
            if self.fails == 0:
                return None
            info = {"attempts": self.fails,
                    "elapsed": max(0.0, now - self.since),
                    "no_device": self.no_device}
            self._reset_locked()
            return info

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()
