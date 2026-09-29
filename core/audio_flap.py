"""Audio-device announcement governor: flap detection + damping (2026-09-29).

THE INCIDENT THIS EXISTS FOR (live log, 2026-09-29 15:20-15:29, owner away).
The owner's USB desk microphone's Windows audio ENDPOINT kept disappearing and
reappearing (Active <-> NotPresent every ~20-40 s) while its USB device stayed
connected. His wireless headset was powered OFF but its dongle kept its
endpoints Active, so Windows bounced the default recording device between the
two, and every bounce produced spoken sentences from two independent writers:

  * bobert_companion._refresh_devices -> _announce_device_change
      "Switched to your headset, sir." / "Switched to the <desk mic>, sir."
  * audio/audio_switch.AudioAutoSwitch (the headset daemon)
      "Sir, I may not be able to hear you. ..."           (deaf alert)
      "Windows' default microphone is off the powered-off headset now ..."
                                                           (its recovery)

17 announcements in ~10 minutes, to an empty room. The only filter in the
pipeline (_speak_pending's dedupe) catches exact repeats in a short window,
and none of these were exact repeats.

WHAT THIS MODULE DECIDES (and the only thing it decides). Every audio-device
sentence is SUBMITTED here with a kind; device-change EVENTS are NOTED here.
The governor returns the sentences that may be spoken NOW; the caller
(bobert_companion._audio_device_announce / _audio_flap_flush) enqueues them.
It never speaks, never touches a device, and never changes which device is
used -- it is pure bookkeeping, stdlib only, thread-safe, with an injectable
clock so every rule is testable without sleeping.

THE RULES
  1. FLAPPING. A device family (an endpoint's state, or the default flipping
     between the same two devices) that changes >= threshold times inside
     window_s is FLAPPING. The first family to start flapping opens a STORM:
     ONE plain sentence ("Sir, the Desk Mic keeps dropping in and out; I'll
     stop announcing audio device changes until it settles."), then every
     governed sentence is kept quiet (logged as `[audio-flap] quiet ...`)
     until every flapping family has gone settle_s (= 2 x window_s, 10 min at
     the defaults) without a change. That is said ONCE too.
  2. ONE PER GAP. At most one governed sentence per min_gap_s. A sentence
     arriving inside the gap is HELD; a later one REPLACES the held one (the
     newest state is the only one worth saying) and is released when the gap
     opens. An exact repeat of the last sentence inside the gap is dropped.
  3. HEARING ALERTS. A deaf alert ("I may not be able to hear you") is said
     at most once per DEAF_REPEAT_S while nothing else has been said about
     audio devices, and never inside a storm (the storm sentence covers it).
     Its recovery line is only said when the alert itself was actually said:
     a recovery from a fault he never heard about is not news. A deaf alert
     and its own recovery alternating is itself a flap.

KINDS
  switch      a device switch ("Switched to ...", "headset off - audio back to")
  deaf        "I may not be able to hear you ..."
  deaf-clear  that alert's recovery line
  alert       a safety alert with its OWN throttle (the silent-mic warning):
              kept quiet inside a storm, otherwise passed through untouched
  anything else is not governed and is passed straight through.
"""
from __future__ import annotations

import threading
import time
from collections import deque

KIND_SWITCH = "switch"
KIND_DEAF = "deaf"
KIND_DEAF_CLEAR = "deaf-clear"
KIND_ALERT = "alert"
# The kinds rules 1-3 apply to in full.
GOVERNED_KINDS = frozenset({KIND_SWITCH, KIND_DEAF, KIND_DEAF_CLEAR})
# The governor's own sentences (always governed: they are about devices too).
_KIND_FLAP = "flap"
_KIND_SETTLED = "settled"

# Defaults mirror core/config.py (AUDIO_FLAP_WINDOW_S / AUDIO_FLAP_THRESHOLD /
# AUDIO_ANNOUNCE_MIN_GAP_S); tests/test_audio_flap.py pins the two together.
DEFAULT_WINDOW_S = 300.0
DEFAULT_THRESHOLD = 3
DEFAULT_MIN_GAP_S = 60.0
# "I may not be able to hear you" at most once per 10 minutes while nothing
# changes. Not a knob: the owner asked for exactly this bound.
DEAF_REPEAT_S = 600.0

_FLAP_VERB = "keeps dropping in and out"


def _clip(text: str, n: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


class AudioFlapGovernor:
    """Flap detector + announcement rate limiter for audio-device speech.

    Every public method takes the lock, mutates, collects its log lines and
    prints them AFTER releasing it, and never raises into the caller for a
    bad argument (the callers sit on the device-refresh path and the audio
    daemon's poll)."""

    def __init__(self, *, window_s: float = DEFAULT_WINDOW_S,
                 threshold: int = DEFAULT_THRESHOLD,
                 min_gap_s: float = DEFAULT_MIN_GAP_S,
                 deaf_repeat_s: float = DEAF_REPEAT_S,
                 clock=time.monotonic, log=None):
        self._lock = threading.Lock()
        self._clock = clock
        # None = the builtin print, looked up at CALL time (so a caller that
        # redirects print -- the monolith's log tee, a test -- sees the lines).
        self._log = log
        self.window_s = DEFAULT_WINDOW_S
        self.threshold = DEFAULT_THRESHOLD
        self.min_gap_s = DEFAULT_MIN_GAP_S
        self.deaf_repeat_s = float(deaf_repeat_s)
        self.configure(window_s=window_s, threshold=threshold,
                       min_gap_s=min_gap_s)
        self.reset()

    # ── configuration ──────────────────────────────────────────────────────
    def configure(self, *, window_s=None, threshold=None,
                  min_gap_s=None) -> None:
        """Apply knob values, sanitised. A value that cannot be read keeps
        the current one (a typo in user_settings.json must not disable the
        damping). threshold < 2 turns flap detection off (1 would call every
        single change a flap); min_gap_s <= 0 turns the gap off."""
        with self._lock:
            try:
                if window_s is not None:
                    w = float(window_s)
                    if w == w and w > 0:          # finite-ish and positive
                        self.window_s = min(w, 86400.0)
            except (TypeError, ValueError):
                pass
            try:
                if threshold is not None and not isinstance(threshold, bool):
                    self.threshold = int(threshold)
            except (TypeError, ValueError):
                pass
            try:
                if min_gap_s is not None:
                    g = float(min_gap_s)
                    if g == g:
                        self.min_gap_s = max(0.0, min(g, 3600.0))
            except (TypeError, ValueError):
                pass

    @property
    def settle_s(self) -> float:
        """How long a flapping family must go without a change to count as
        settled: twice the detection window (10 minutes at the defaults)."""
        return 2.0 * self.window_s

    def reset(self) -> None:
        """Forget everything (tests; never needed at runtime)."""
        with self._lock:
            self._flips: dict[str, deque] = {}
            self._labels: dict[str, str] = {}
            self._flapping: dict[str, float] = {}
            self._storm: dict | None = None
            self._last_emit_at: float | None = None
            self._last_emit_msg: str | None = None
            self._held: tuple[str, str] | None = None
            self._deaf_last_at: float | None = None
            self._deaf_outstanding = False
            self._changed_since_deaf = False
            self._last_hearing_kind: str | None = None

    # ── queries ────────────────────────────────────────────────────────────
    def storm_active(self) -> bool:
        with self._lock:
            return self._storm is not None or bool(self._flapping)

    def is_flapping(self, key: str) -> bool:
        with self._lock:
            return key in self._flapping

    def held_message(self) -> str | None:
        with self._lock:
            return self._held[0] if self._held else None

    # ── events ─────────────────────────────────────────────────────────────
    def note_flip(self, key: str, label: str = "", *, verb: str = _FLAP_VERB,
                  now: float | None = None) -> list[str]:
        """Record one change of the device family ``key`` (an endpoint going
        Active/NotPresent, or the default moving between the same two
        devices). Returns the sentences to speak now (the storm sentence,
        when this change is the one that makes it a storm and the gap
        allows)."""
        logs: list[str] = []
        with self._lock:
            now = self._now(now)
            out = self._flip_locked(key, label, verb, now, logs)
        self._emit_logs(logs)
        return out

    def submit(self, message: str, kind: str, *,
               now: float | None = None) -> list[str]:
        """Ask to speak ``message``. Returns what may be spoken NOW (possibly
        nothing: held, kept quiet, or dropped -- each logged)."""
        if not message:
            return []
        if kind not in GOVERNED_KINDS and kind != KIND_ALERT:
            return [message]
        logs: list[str] = []
        with self._lock:
            now = self._now(now)
            out: list[str] = []
            if kind == KIND_ALERT:
                if self._storm is not None or self._flapping:
                    self._quiet_locked(message, kind, logs)
                else:
                    out = [message]
            else:
                out = self._submit_locked(message, kind, now, logs)
        self._emit_logs(logs)
        return out

    def flush(self, *, now: float | None = None) -> list[str]:
        """Settle finished storms and release a held sentence whose gap has
        opened. Call it wherever speech is drained."""
        logs: list[str] = []
        with self._lock:
            now = self._now(now)
            out = self._flush_locked(now, logs)
        self._emit_logs(logs)
        return out

    # ── internals (lock held) ──────────────────────────────────────────────
    def _now(self, now: float | None) -> float:
        if now is not None:
            return float(now)
        return float(self._clock())

    def _emit_logs(self, logs: list[str]) -> None:
        sink = self._log if self._log is not None else print
        for line in logs:
            try:
                sink(f"  [audio-flap] {line}")
            except Exception:
                pass

    def _flip_locked(self, key, label, verb, now, logs) -> list[str]:
        if not key:
            return []
        dq = self._flips.setdefault(key, deque())
        dq.append(now)
        cutoff = now - self.window_s
        while dq and dq[0] < cutoff:
            dq.popleft()
        if label:
            self._labels[key] = label
        label = self._labels.get(key) or "an audio device"
        if key in self._flapping:
            return []                  # already counted as flapping
        if self.threshold < 2 or len(dq) < self.threshold:
            return []
        self._flapping[key] = now
        logs.append(f"{label}: {len(dq)} changes in the last "
                    f"{self.window_s:.0f}s - FLAPPING")
        if self._storm is not None:
            logs.append(f"{label} joins the flap storm already announced for "
                        f"{self._storm['label']} - staying quiet")
            return []
        self._storm = {"label": label, "since": now, "quiet": 0}
        logs.append(f"announcing it ONCE, then keeping audio device "
                    f"announcements quiet until it has gone "
                    f"{self.settle_s:.0f}s without a change")
        msg = (f"Sir, {label} {verb}; I'll stop announcing audio device "
               f"changes until it settles.")
        return self._offer_locked(msg, _KIND_FLAP, now, logs)

    def _quiet_locked(self, message, kind, logs) -> None:
        if self._storm is not None:
            self._storm["quiet"] += 1
        logs.append(f"quiet during the flap storm ({kind}): "
                    f"{_clip(message)}")

    def _submit_locked(self, message, kind, now, logs) -> list[str]:
        out: list[str] = []
        # A hearing alert alternating with its own recovery is a flap too --
        # the daemon's deaf / recovered pair tracks the same bouncing device.
        if kind in (KIND_DEAF, KIND_DEAF_CLEAR):
            prev = self._last_hearing_kind
            self._last_hearing_kind = kind
            if prev is not None and prev != kind:
                out += self._flip_locked("hearing", "the microphone",
                                         _FLAP_VERB, now, logs)
        if self._storm is not None or self._flapping:
            self._quiet_locked(message, kind, logs)
            return out
        if kind == KIND_DEAF:
            if (self._deaf_last_at is not None
                    and not self._changed_since_deaf
                    and now - self._deaf_last_at < self.deaf_repeat_s):
                logs.append(f"not repeating a hearing alert "
                            f"{now - self._deaf_last_at:.0f}s after the last "
                            f"one (at most one per {self.deaf_repeat_s:.0f}s "
                            f"while nothing changes): {_clip(message)}")
                return out
        elif kind == KIND_DEAF_CLEAR:
            if not self._deaf_outstanding:
                if self._held is not None and self._held[1] == KIND_DEAF:
                    logs.append("the hearing alert cleared before it was "
                                "spoken - dropping both: "
                                f"{_clip(self._held[0])}")
                    self._held = None
                else:
                    logs.append("not announcing a recovery from a fault that "
                                f"was never announced: {_clip(message)}")
                return out
        return out + self._offer_locked(message, kind, now, logs)

    def _gap_open(self, now: float) -> bool:
        return (self._last_emit_at is None
                or now - self._last_emit_at >= self.min_gap_s)

    def _offer_locked(self, message, kind, now, logs) -> list[str]:
        gap_open = self._gap_open(now)
        if (not gap_open and message == self._last_emit_msg):
            logs.append(f"dropping an exact repeat inside the "
                        f"{self.min_gap_s:.0f}s gap: {_clip(message)}")
            return []
        if self._held is not None:
            if self._held[0] == message:
                return []              # already waiting to be said
            logs.append(f"superseded the held announcement "
                        f"{_clip(self._held[0], 60)!r} with "
                        f"{_clip(message, 60)!r}")
            self._held = None
        if gap_open:
            self._note_emitted(message, kind, now)
            return [message]
        wait = self.min_gap_s - (now - (self._last_emit_at or now))
        logs.append(f"holding for {max(0.0, wait):.0f}s (at most one audio "
                    f"announcement per {self.min_gap_s:.0f}s): "
                    f"{_clip(message)}")
        self._held = (message, kind)
        return []

    def _note_emitted(self, message, kind, now) -> None:
        self._last_emit_at = now
        self._last_emit_msg = message
        if kind == KIND_DEAF:
            self._deaf_last_at = now
            self._deaf_outstanding = True
            self._changed_since_deaf = False
        elif kind == KIND_DEAF_CLEAR:
            self._deaf_outstanding = False
            self._changed_since_deaf = True
        else:
            self._changed_since_deaf = True

    def _flush_locked(self, now, logs) -> list[str]:
        out: list[str] = []
        if self._flapping:
            settle = self.settle_s
            for key in list(self._flapping):
                dq = self._flips.get(key)
                last = dq[-1] if dq else self._flapping[key]
                if now - last >= settle:
                    del self._flapping[key]
                    logs.append(f"{self._labels.get(key) or key}: no change "
                                f"for {settle:.0f}s - no longer flapping")
        if self._storm is not None and not self._flapping:
            storm = self._storm
            self._storm = None
            mins = self.settle_s / 60.0
            span = (f"{mins:.0f} minutes" if mins >= 1.5
                    else f"{self.settle_s:.0f} seconds")
            logs.append(f"flap storm over: {storm['label']} has been stable "
                        f"for {span}; {storm['quiet']} audio announcement(s) "
                        f"were kept quiet during it")
            out += self._offer_locked(
                f"Sir, {storm['label']} has been steady for {span}, so I'm "
                f"announcing audio device changes again.",
                _KIND_SETTLED, now, logs)
        if self._held is not None and self._gap_open(now):
            message, kind = self._held
            self._held = None
            self._note_emitted(message, kind, now)
            logs.append(f"releasing the held announcement: {_clip(message)}")
            out.append(message)
        return out
