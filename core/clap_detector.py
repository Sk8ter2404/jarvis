"""core/clap_detector.py — a lightweight double-clap detector.

Feeds on the mono float32 frames the main loop's microphone capture already
fans out (bobert_companion.add_record_tap); it never opens a stream of its own.
Pure numpy, no state outside the instance, never raises out of ``feed``.

WHAT COUNTS AS A DOUBLE CLAP
============================
Two hand claps 0.15-0.7 s apart (onset to onset), with NOTHING ELSE loud
around them: no sound for ``max_interval_s`` before the first clap, nothing
between them but the first clap's own decaying tail, and nothing for
``max_interval_s`` after the second. That isolation rule is what rejects music
(a beat always has a neighbour inside the window), clapping along to a song
(a train of claps), a triple clap, and "...and then clap clap" mid-sentence. It
also means the detector decides ~0.7 s after the second clap: a third clap
inside that window cancels the pair.

Each clap must ALSO look like a clap on its own (``_classify``):

  * LOUD     — its peak sample reaches ``min_peak`` (the sensitivity knob,
               CLAP_TRIGGER_MIN_PEAK) and it towers over the room floor;
  * SHARP    — the 10 ms block energy jumps >= ONSET_RATIO (~15.5 dB) over the
               blocks just before it and peaks within 30 ms;
  * SHORT    — it decays to DECAY_FRAC (-12 dB) of its peak within
               MAX_DECAY_S. Speech syllables, sustained notes and doors
               ring for far longer;
  * NOT A CLICK — its energy is spread over >= MIN_EFFECTIVE_MS. A key click
               or a digital pop is a 1 ms impulse; a clap's burst lasts several;
  * BRIGHT   — spectral centroid >= MIN_CENTROID_HZ. Kicks, door thumps and
               knocks on wood are low;
  * NOISY    — spectral flatness >= MIN_FLATNESS. Voiced speech, a struck
               glass and synth stabs are harmonic.

Two claps of very different loudness (> 4x) are not a pair either.

The room floor is the 20th percentile of the last 2 s of 10 ms block RMS, so a
steady hum (a fan, a fridge, quiet background music) raises the bar instead of
blocking the feature. "Loud" means above both LOUD_RATIO x that floor and
QUIET_FRAC x the clap's own peak, so faint typing far below the claps is not
"something else loud", while talking or a beat is.

Events are dicts: ``{"t_first", "t_second", "t_detect", "interval_s",
"peak_first", "peak_second"}`` with times in STREAM seconds (samples fed since
the last reset / sample rate). The caller maps them to wall time.

``reset()`` must be called when the stream has a GAP (record_speech closed and
reopened, frames dropped): a clap before the gap and one after it would
otherwise look like a pair. A fresh detector also refuses claps until it has
seen ``max_interval_s`` of history before them — no history, no proof that
nothing loud preceded the first clap.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

DEFAULT_SAMPLE_RATE = 16000

# ── tunables (seconds / ratios). Constructor arguments override the timing
# and loudness ones; the rest are the shape of a clap and stay fixed. ──────
BLOCK_S = 0.010                 # energy resolution
MIN_INTERVAL_S = 0.15           # onset-to-onset spacing of the two claps
MAX_INTERVAL_S = 0.70
DEFAULT_MIN_PEAK = 0.12         # peak |sample| of a clap (0..1 full scale)
ONSET_RATIO = 6.0               # peak-block RMS over the pre-onset blocks
LOUD_RATIO = 4.0                # "loud" = this x the room floor ...
LOUD_ABS_MIN = 0.002            # ... and never below this RMS
QUIET_FRAC = 0.30               # ... and >= this x the clap's own peak RMS
DECAY_FRAC = 0.25               # a clap falls to this x its peak RMS ...
MAX_DECAY_S = 0.15              # ... within this long after the peak
PEAK_SEARCH_S = 0.02            # the peak lands this soon after the onset
MIN_EFFECTIVE_MS = 1.5          # energy spread: shorter = a click / a pop
MIN_CENTROID_HZ = 1000.0        # brightness
MIN_FLATNESS = 0.05             # noise-likeness (0 = pure tone, ~0.5 = white)
MAX_LEVEL_RATIO = 4.0           # the two claps' peak RMS within this factor
FLOOR_WINDOW_S = 2.0            # floor = low percentile over this much history
FLOOR_PERCENTILE = 20.0
FLOOR_MIN = 1e-4
_SPEC_N = 1024                  # FFT window for the spectral check
_SPEC_PRE_S = 0.002             # the window starts this far before the onset

_HISTORY_S = 4.0                # block history kept (covers pre+pair+post)
_RAW_KEEP_S = 1.5               # raw samples kept for the spectral check


def _mono(frame) -> "np.ndarray | None":
    """A 1-D float64 copy of ``frame`` (first channel), or None."""
    try:
        a = np.asarray(frame, dtype=np.float64)
    except Exception:
        return None
    if a.ndim == 0:
        return None
    if a.ndim > 1:
        a = a.reshape(a.shape[0], -1)[:, 0]
    if a.size == 0:
        return None
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def spectral_shape(samples, sample_rate: int = DEFAULT_SAMPLE_RATE):
    """``(centroid_hz, flatness)`` of ``samples`` (Hann-windowed rfft).

    Centroid over 100 Hz-Nyquist; flatness (geometric / arithmetic mean of the
    power spectrum) over 300 Hz-7 kHz, so DC, mains hum and the anti-alias
    roll-off do not decide it. ``(0.0, 0.0)`` for anything unusable. Never
    raises."""
    try:
        x = _mono(samples)
        if x is None or x.size < 32:
            return 0.0, 0.0
        x = x - float(np.mean(x))
        n = int(x.size)
        p = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
        f = np.fft.rfftfreq(n, 1.0 / float(sample_rate))
        band = f >= 100.0
        tot = float(np.sum(p[band]))
        if tot <= 1e-20:
            return 0.0, 0.0
        centroid = float(np.sum(f[band] * p[band]) / tot)
        fb = (f >= 300.0) & (f <= min(7000.0, 0.45 * float(sample_rate)))
        pb = p[fb] + 1e-20
        if pb.size < 4:
            return centroid, 0.0
        flat = float(np.exp(np.mean(np.log(pb))) / np.mean(pb))
        return centroid, flat
    except Exception:
        return 0.0, 0.0


def effective_duration_ms(samples, sample_rate: int = DEFAULT_SAMPLE_RATE,
                          frame_ms: float = 0.5) -> float:
    """How long the energy of ``samples`` lasts: total energy over the energy
    of the loudest ``frame_ms`` frame, in ms. A 1 ms click is ~1; a clap's
    burst is several. 0.0 for silence / junk. Never raises."""
    try:
        x = _mono(samples)
        if x is None:
            return 0.0
        fl = max(1, int(round(frame_ms * 1e-3 * float(sample_rate))))
        n = (x.size // fl) * fl
        if n < fl:
            return 0.0
        e = np.sum(x[:n].reshape(-1, fl) ** 2, axis=1)
        top = float(np.max(e))
        if top <= 1e-20:
            return 0.0
        return float(np.sum(e) / top) * frame_ms
    except Exception:
        return 0.0


class ClapDetector:
    """Streaming double-clap detector. Not thread-safe: one feeder thread."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, *,
                 min_peak: float = DEFAULT_MIN_PEAK,
                 min_interval_s: float = MIN_INTERVAL_S,
                 max_interval_s: float = MAX_INTERVAL_S):
        try:
            sr = int(sample_rate)
        except Exception:
            sr = DEFAULT_SAMPLE_RATE
        self.sample_rate = sr if sr >= 4000 else DEFAULT_SAMPLE_RATE
        self.min_peak = float(min_peak)
        self.min_interval_s = float(min_interval_s)
        self.max_interval_s = float(max_interval_s)
        self._blk_n = max(1, int(round(BLOCK_S * self.sample_rate)))
        self._bs = self._blk_n / float(self.sample_rate)   # block seconds
        self._hist_max = int(_HISTORY_S / self._bs)
        self._floor_n = int(FLOOR_WINDOW_S / self._bs)
        self._decay_blocks = int(round(MAX_DECAY_S / self._bs))
        self._peak_blocks = int(round(PEAK_SEARCH_S / self._bs))
        self._win = int(round(self.max_interval_s / self._bs))
        self._spec_pre = int(round(_SPEC_PRE_S * self.sample_rate))
        # The decision horizon for one transient: its decay is known and the
        # spectral window has fully arrived.
        self._horizon = max(self._peak_blocks + self._decay_blocks + 1,
                            int(math.ceil((_SPEC_N + self._spec_pre)
                                          / self._blk_n)) + 1)
        self.stats = {"events": 0, "claps": 0, "pairs": 0, "fired": 0}
        self.last_rejection: "str | None" = None
        self.last_clap_peak = 0.0          # peak of the last accepted clap
        self.last_transient_peak = 0.0     # peak of the last sharp sound
        self.reset()

    # ── public ──────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Forget the stream (call after any gap in the audio)."""
        self._carry = np.zeros(0)
        self._raw = np.zeros(0)
        self._raw_base = 0              # absolute sample index of _raw[0]
        self._n_samples = 0             # samples consumed into blocks
        self._blk = 0                   # absolute index of the NEXT block
        self._rms: list = []
        self._peak: list = []
        self._loud: list = []
        self._hbase = 0                 # absolute block index of _rms[0]
        self._floor = FLOOR_MIN
        self._open: list = []           # transients awaiting a decision
        self._claps: deque = deque(maxlen=8)
        self._pending: list = []        # pairs awaiting the post window
        self._last_open = -10 ** 9

    def feed(self, frame) -> list:
        """Consume one frame; return any double-clap events it completed."""
        try:
            return self._feed(frame)
        except Exception as exc:  # never let a bad frame kill the feeder
            self.last_rejection = f"internal error: {type(exc).__name__}"
            self.reset()
            return []

    # ── internals ───────────────────────────────────────────────────────
    def _feed(self, frame) -> list:
        x = _mono(frame)
        if x is None:
            return []
        # A long frame is consumed in record_speech-sized pieces, so the raw
        # window the spectral check needs is never trimmed before its block
        # is classified (the answer must not depend on how frames are cut).
        step = 4096
        if x.size > step:
            out: list = []
            for k in range(0, x.size, step):
                out.extend(self._feed_piece(x[k:k + step]))
            return out
        return self._feed_piece(x)

    def _feed_piece(self, x) -> list:
        self._raw = np.concatenate([self._raw, x])
        keep = int(_RAW_KEEP_S * self.sample_rate)
        if self._raw.size > keep:
            cut = self._raw.size - keep
            self._raw = self._raw[cut:]
            self._raw_base += cut
        buf = np.concatenate([self._carry, x]) if self._carry.size else x
        nb = buf.size // self._blk_n
        self._carry = buf[nb * self._blk_n:].copy()
        if nb == 0:
            return []
        blocks = buf[:nb * self._blk_n].reshape(nb, self._blk_n)
        rms = np.sqrt(np.mean(blocks * blocks, axis=1))
        peak = np.max(np.abs(blocks), axis=1)
        events: list = []
        for r, p in zip(rms.tolist(), peak.tolist()):
            events.extend(self._block(r, p))
        self._n_samples += nb * self._blk_n
        return events

    def _h(self, idx: int, arr: list):
        """History value at absolute block ``idx`` (None when not kept)."""
        j = idx - self._hbase
        if 0 <= j < len(arr):
            return arr[j]
        return None

    def _block(self, r: float, p: float) -> list:
        i = self._blk
        if i % 5 == 0 or i - self._hbase < 20:
            recent = self._rms[-self._floor_n:]
            if recent:
                self._floor = max(FLOOR_MIN, float(
                    np.percentile(recent, FLOOR_PERCENTILE)))
            else:
                self._floor = max(FLOOR_MIN, r)
        loud = max(LOUD_ABS_MIN, LOUD_RATIO * self._floor)
        prev_r = self._rms[-1] if self._rms else 0.0
        prev_l = self._loud[-1] if self._loud else loud
        self._rms.append(r)
        self._peak.append(p)
        self._loud.append(loud)
        self._blk = i + 1
        if len(self._rms) > 2 * self._hist_max:
            drop = len(self._rms) - self._hist_max
            del self._rms[:drop], self._peak[:drop], self._loud[:drop]
            self._hbase += drop

        # A new transient: a rising crossing of the loud line, or a sharp jump
        # inside sound that is already loud (a second clap on the first's tail).
        if r >= loud and i - self._last_open > self._peak_blocks:
            before = self._rms[-4:-1]
            jump = r >= ONSET_RATIO * max(max(before) if before else 0.0,
                                          self._floor)
            if prev_r < prev_l or jump:
                self._open.append(i)
                self._last_open = i
                self.stats["events"] += 1

        events: list = []
        while self._open and i - self._open[0] >= self._horizon:
            s = self._open.pop(0)
            c = self._classify(s)
            if c is not None:
                events.extend(self._on_clap(c))
        still: list = []
        for pair in self._pending:
            if i >= pair["decide_at"]:
                ev = self._decide(pair)
                if ev is not None:
                    events.append(ev)
            else:
                still.append(pair)
        self._pending = still
        return events

    def _reject(self, why: str):
        self.last_rejection = why
        return None

    def _classify(self, s: int):
        """A clap record for the transient that opened at block ``s``, or None
        (with ``last_rejection`` saying why)."""
        rms = [self._h(k, self._rms) for k in range(s, s + self._horizon)]
        if any(v is None for v in rms):
            return self._reject("history lost")
        win = rms[:self._peak_blocks + 1]
        k_p = int(np.argmax(win))
        p_rms = win[k_p]
        pk = max(self._h(k, self._peak) or 0.0
                 for k in range(s, s + self._peak_blocks + 2))
        self.last_transient_peak = float(pk)
        if pk < self.min_peak:
            return self._reject(
                f"too quiet (peak {pk:.3f} < {self.min_peak:.3f})")
        pre = [v for v in (self._h(k, self._rms) for k in range(s - 4, s - 1))
               if v is not None]
        base = max(max(pre) if pre else 0.0, self._floor)
        if p_rms < ONSET_RATIO * base:
            return self._reject("not sharp (slow rise)")
        d = None
        for k in range(k_p + 1, min(len(rms), k_p + 1 + self._decay_blocks)):
            if rms[k] <= DECAY_FRAC * p_rms:
                d = k
                break
        if d is None:
            return self._reject("too long (no fast decay)")
        a0 = s * self._blk_n - self._spec_pre - self._raw_base
        a1 = a0 + _SPEC_N
        if a0 < 0 or a1 > self._raw.size:
            return self._reject("history lost")
        seg = self._raw[a0:a1]
        eff = effective_duration_ms(seg, self.sample_rate)
        if eff < MIN_EFFECTIVE_MS:
            return self._reject(f"a click, not a clap ({eff:.1f} ms)")
        centroid, flat = spectral_shape(seg, self.sample_rate)
        if centroid < MIN_CENTROID_HZ:
            return self._reject(f"too dull ({centroid:.0f} Hz)")
        if flat < MIN_FLATNESS:
            return self._reject(f"tonal, not noisy (flatness {flat:.3f})")
        self.stats["claps"] += 1
        self.last_clap_peak = float(pk)
        return {"s": s, "p": s + k_p, "d": s + d, "rms": float(p_rms),
                "peak": float(pk)}

    def _quiet(self, lo: int, hi: int, ref_rms: float):
        """True when every block in [lo, hi] is below the loud line and
        QUIET_FRAC x ``ref_rms``; None when part of the range is not kept."""
        for k in range(lo, hi + 1):
            r = self._h(k, self._rms)
            if r is None:
                return None
            if r >= max(self._h(k, self._loud), QUIET_FRAC * ref_rms):
                return False
        return True

    def _on_clap(self, b: dict) -> list:
        prev = self._claps[-1] if self._claps else None
        self._claps.append(b)
        if prev is None:
            return []
        gap_s = (b["s"] - prev["s"]) * self._bs
        if not (self.min_interval_s <= gap_s <= self.max_interval_s):
            return []
        ratio = max(prev["rms"], b["rms"]) / max(min(prev["rms"], b["rms"]),
                                                 1e-12)
        if ratio > MAX_LEVEL_RATIO:
            self._reject("the two claps differ too much in loudness")
            return []
        between = self._quiet(prev["d"] + 1, b["s"] - 2, prev["rms"])
        if not between:
            self._reject("something loud between the claps")
            return []
        self._pending.append({"a": prev, "b": b,
                              "decide_at": b["s"] + self._win + 1})
        self.stats["pairs"] += 1
        return []

    def _decide(self, pair: dict):
        a, b = pair["a"], pair["b"]
        win = self._win
        pre = self._quiet(a["s"] - win, a["s"] - 2, a["rms"])
        if pre is None:
            return self._reject("not enough quiet history before the claps")
        if not pre:
            return self._reject("something loud just before the claps")
        post = self._quiet(b["d"] + 1, b["s"] + win, b["rms"])
        if not post:
            return self._reject("something loud just after the claps")
        # A pair fired: neither clap may start another one.
        self._claps.clear()
        self.stats["fired"] += 1
        bs = self._bs
        return {"t_first": a["s"] * bs, "t_second": b["s"] * bs,
                "t_detect": self._blk * bs,
                "interval_s": (b["s"] - a["s"]) * bs,
                "peak_first": a["peak"], "peak_second": b["peak"]}
