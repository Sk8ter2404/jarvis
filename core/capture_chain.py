"""core/capture_chain.py - the capture chain's pure helpers (v3.0.26 overhaul).

WHY THIS EXISTS
---------------
The 11-13..11-17 logs (9 sessions, 9,959 captures, 76 that became owner turns
or wakes) showed the microphone chain working against itself:

  * record_speech started a capture on any 64 ms chunk over VAD_THRESHOLD
    (raw RMS 0.008) while the room's own floor sat at 0.004-0.008, so ~5,800
    noise captures in two days were decoded (Parakeet, then a Whisper "empty"
    rescue that made 6 usable lines out of 1,686);
  * the capture auto-gain (up to 10x) was computed from the RAW peak but
    multiplied the already noise-suppressed, AGC'd clip (up to 8x) and then
    hard-clipped it: 3,725 of 5,773 auto-gain decisions sat at the x10 cap;
  * voice-ID scored that processed clip against a voiceprint enrolled from
    raw audio, and the owner's own wake turns scored a median 0.69 against
    the 0.72 match line.

Everything here is pure (numpy only, no model, no device, no clock of its
own) so the rules are tested directly; bobert_companion wires them in.

  peak_chunk_rms(audio)  the loudest 1,024-sample chunk RMS of a clip: the
                         level the auto-gain normalises (CAPTURE_AUTO_GAIN_
                         BASIS 'clip') - the clip it is applied to, not a
                         different signal's peak.
  soft_limit(x)          a tanh knee above SOFT_LIMIT_KNEE instead of a hard
                         clip at +/-1.
  NoiseFloor             the rolling idle-chunk RMS percentile and the start
                         threshold it gives (VAD_FLOOR_*): never below
                         VAD_THRESHOLD, never above VAD_FLOOR_MAX_RATIO x it.
  CaptureBook            the counters behind the one "[vad-floor]" line every
                         LOG_EVERY_S (captures, noise held under the floor,
                         one-chunk trips, the quietest turn's margin, the
                         auto-gain, the rescues skipped) - what the first live
                         session reads to score the change, since VAD_DEBUG's
                         per-capture "[vad] peak" line is now off by default.

THE v2.0.104 LESSON
-------------------
"Voiced" once meant raw RMS over a gate that later gain would have made
transcribable, and real words were dropped. So the start threshold only rises
with a MEASURED floor (at least MIN_SAMPLES idle chunks), is capped at a small
multiple of the owner's own VAD_THRESHOLD (the quietest of the 76 turns peaked
at 0.0104 against a 0.010 cap), stays at VAD_THRESHOLD around JARVIS's own
playback (the AEC duck note of 2026-05-30), and never applies to an in-turn
capture (a confirmation he is answering). The end-of-speech gate (silence
counting) is never touched: a capture is never cut shorter than before.
"""
from __future__ import annotations

from collections import deque

import numpy as np

CHUNK = 1024                 # record_speech's capture chunk (64 ms at 16 kHz)
SOFT_LIMIT_KNEE = 0.9        # |x| above this is compressed toward 1.0
FLOOR_MODES = ("off", "shadow", "on")
FLOOR_WINDOW = 940           # idle chunks kept: ~60 s of listening
MIN_SAMPLES = 150            # ~10 s of idle chunks before the floor counts
FLOOR_PERCENTILE = 20.0      # robust to his own sub-threshold onsets
RECOMPUTE_EVERY = 16         # chunks between two percentile computations
LOG_EVERY_S = 600.0          # one "[vad-floor]" line per 10 minutes at most


def floor_mode(value, default: str = "off") -> str:
    """VAD_FLOOR_MODE normalised: 'off' | 'shadow' | 'on'; anything else is
    ``default``. Never raises."""
    try:
        v = str(value).strip().lower()
    except Exception:
        return default
    return v if v in FLOOR_MODES else default


def peak_chunk_rms(audio, chunk: int = CHUNK) -> float:
    """The loudest ``chunk``-sample RMS of ``audio`` (the whole clip's RMS
    when it is shorter than one chunk; 0.0 for nothing). This is the same
    measure record_speech's peak_rms uses, taken on the clip itself. Never
    raises (0.0)."""
    try:
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        n = int(a.size)
        if n == 0:
            return 0.0
        c = max(1, int(chunk))
        if n <= c:
            return float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))
        whole = (n // c) * c
        sq = np.square(a[:whole].astype(np.float64)).reshape(-1, c)
        best = float(np.sqrt(sq.mean(axis=1).max()))
        if whole < n:
            tail = a[whole:].astype(np.float64)
            best = max(best, float(np.sqrt(np.mean(tail * tail))))
        return best
    except Exception:
        return 0.0


def soft_limit(x, knee: float = SOFT_LIMIT_KNEE):
    """``x`` (float32 ndarray, changed IN PLACE and returned) with every
    |sample| over ``knee`` compressed by a tanh curve that approaches 1.0:
    samples under the knee are untouched, the curve's slope is 1 at the knee
    (no corner), louder crests stay louder, and the output stays inside
    [-1, 1] (only a crest far past 1.0 rounds to it in float32). A hard clip
    at +/-1 flattened every crest of a x10 boost. Never raises (returns ``x``
    as it is on a failure)."""
    try:
        k = float(knee)
        if not 0.0 < k < 1.0:
            np.clip(x, -1.0, 1.0, out=x)
            return x
        mag = np.abs(x)
        over = mag > k
        if not np.any(over):
            return x
        head = 1.0 - k
        squeezed = k + head * np.tanh((mag[over] - k) / head)
        x[over] = (np.sign(x[over]) * squeezed).astype(x.dtype, copy=False)
        return x
    except Exception:
        return x


class NoiseFloor:
    """The room's idle level and the capture start threshold it gives.

    note_idle(rms) takes the RMS of a chunk heard while no capture is
    recording and JARVIS is not playing (the caller decides both). The floor
    is the FLOOR_PERCENTILE of the last FLOOR_WINDOW of them, recomputed every
    RECOMPUTE_EVERY notes; None until MIN_SAMPLES were heard (no measurement,
    no rise). Not thread-safe: record_speech runs on one thread at a time
    (the mic claim)."""

    def __init__(self, window: int = FLOOR_WINDOW,
                 min_samples: int = MIN_SAMPLES,
                 percentile: float = FLOOR_PERCENTILE,
                 recompute_every: int = RECOMPUTE_EVERY):
        self._hist: "deque[float]" = deque(maxlen=max(1, int(window)))
        self._min = max(1, int(min_samples))
        self._pct = float(percentile)
        self._every = max(1, int(recompute_every))
        self._since = 0
        self._floor = None

    def reset(self) -> None:
        self._hist.clear()
        self._since = 0
        self._floor = None

    def __len__(self) -> int:
        return len(self._hist)

    def note_idle(self, rms) -> None:
        try:
            v = float(rms)
        except Exception:
            return
        if not np.isfinite(v) or v < 0.0:
            return
        self._hist.append(v)
        self._since += 1
        if self._floor is None or self._since >= self._every:
            self._since = 0
            self._floor = (float(np.percentile(np.fromiter(
                self._hist, dtype=np.float64), self._pct))
                if len(self._hist) >= self._min else None)

    def floor(self) -> "float | None":
        """The measured idle floor, or None before MIN_SAMPLES."""
        return self._floor

    def start_threshold(self, base: float, factor: float,
                        max_ratio: float) -> float:
        """max(base, min(base x max_ratio, factor x floor)); ``base`` until
        the floor is measured. Never below ``base`` (the safe minimum: the
        owner's own VAD_THRESHOLD) and never above ``base x max_ratio``.
        Never raises (``base``)."""
        try:
            b = float(base)
            f = self._floor
            if f is None:
                return b
            cap = b * max(1.0, float(max_ratio))
            return max(b, min(cap, float(factor) * float(f)))
        except Exception:
            return float(base)


class CaptureBook:
    """Counters for one "[vad-floor]" summary line per LOG_EVERY_S (numbers
    only, never words). The caller passes ``now`` (monotonic seconds); the
    first call starts the window. Not thread-safe beyond the GIL: every note
    is a small integer / list update."""

    def __init__(self, every_s: float = LOG_EVERY_S):
        self.every_s = float(every_s)
        self._t0 = None
        self._clear()

    def reset(self) -> None:
        """Every count gone and the window not started."""
        self._t0 = None
        self._clear()

    def _clear(self) -> None:
        self.captures = 0
        self.would_hold = 0          # shadow: 'on' would not have started it
        self.would_hold_turns = 0    # ... and it became a turn ("lost")
        self.held = 0                # on: noise episodes held under the floor
        self.single = 0              # one-chunk trips (VAD_START_CONFIRM_CHUNKS)
        self.single_turns = 0
        self.turns = 0
        self.margins: list = []      # turn peak / its start threshold
        self.gains: list = []        # auto-gain factors applied
        self.gain_capped = 0
        self.rescue_skipped = 0      # 'empty' rescues the speech check skipped
        self.rescue_run = 0

    def _start(self, now: float) -> None:
        if self._t0 is None:
            self._t0 = float(now)

    def note_capture(self, meta: dict, now: float) -> None:
        self._start(now)
        self.captures += 1
        if meta.get("would_hold"):
            self.would_hold += 1
        if meta.get("single"):
            self.single += 1

    def note_held(self, now: float) -> None:
        self._start(now)
        self.held += 1

    def note_turn(self, meta: dict, now: float) -> None:
        self._start(now)
        self.turns += 1
        if meta.get("would_hold"):
            self.would_hold_turns += 1
        if meta.get("single"):
            self.single_turns += 1
        try:
            thr = float(meta.get("start_thr") or 0.0)
            if thr > 0:
                self.margins.append(float(meta.get("peak") or 0.0) / thr)
        except Exception:
            pass

    def note_gain(self, gain: float, capped: bool, now: float) -> None:
        self._start(now)
        try:
            self.gains.append(float(gain))
        except Exception:
            return
        if capped:
            self.gain_capped += 1

    def note_rescue(self, skipped: bool, now: float) -> None:
        self._start(now)
        if skipped:
            self.rescue_skipped += 1
        else:
            self.rescue_run += 1

    def line(self, now: float, *, mode: str, floor, start_thr: float,
             base: float, cap: float, confirm: int,
             force: bool = False) -> "str | None":
        """The summary line when LOG_EVERY_S has passed since the window
        started (or ``force``) and anything was counted; the window then
        restarts. None otherwise. Never raises (None)."""
        try:
            if self._t0 is None:
                return None
            span = float(now) - self._t0
            if not force and span < self.every_s:
                return None
            if not (self.captures or self.held or self.gains
                    or self.rescue_skipped or self.rescue_run):
                self._t0 = float(now)
                return None
            mins = max(1, int(round(span / 60.0)))
            fl = "unmeasured" if floor is None else f"{float(floor):.4f}"
            parts = [f"{self.captures} captures, {self.turns} turns"]
            if self.margins:
                parts[-1] += (f" (quietest peak {min(self.margins):.1f}x its "
                              f"start threshold)")
            if mode == "shadow":
                parts.append(f"{self.would_hold} would not have started "
                             f"({self.would_hold_turns} of them turns)")
            elif mode == "on":
                parts.append(f"{self.held} noise episodes held under the "
                             f"floor")
            parts.append(f"{self.single} one-chunk trips "
                         f"({self.single_turns} of them turns"
                         + (", confirm on" if int(confirm) > 1 else "") + ")")
            if self.gains:
                g = sorted(self.gains)
                parts.append(f"auto-gain x{g[len(g) // 2]:.1f} median, "
                             f"{self.gain_capped} of {len(g)} at the cap")
            if self.rescue_skipped or self.rescue_run:
                parts.append(f"empty rescues {self.rescue_run} run, "
                             f"{self.rescue_skipped} skipped (no speech)")
            line = (f"[vad-floor] {mode}: floor {fl} -> start "
                    f"{float(start_thr):.4f} (base {float(base):.4f}, cap "
                    f"{float(cap):.4f}); last {mins} min: "
                    + "; ".join(parts))
            self._clear()
            self._t0 = float(now)
            return line
        except Exception:
            return None
