"""core/listen_media.py - hearing the owner over videos in wake-word mode
(2026-10-05). Pure policy and counters, stdlib only.

WHY THIS EXISTS
---------------
Owner, 2026-10-05 ~02:00: "it seems like he's constantly listening,
especially when videos are playing, and he can't hear me - even in wake word
mode." The 10-05 00:21 session logged 284 "wake-word mode - ignoring
non-wake utterance" lines in ~1.5 h. Four causes (research spec
JARVIS-Research/listen_over_media_20261005/SPEC.md):

  1. Captures are full of video. A capture starts on raw mic loudness and
     music never gives 1.4 s of quiet, so captures run to the 30 s cap.
  2. The wake rule is positional. "Jarvis" must be word 1-3 of the WHOLE
     capture; he speaks into one that is already running, so his name lands
     behind the video's words (97.4 % of media captures were refused).
  3. The mic is closed between captures (a stream per capture): 12-20 % of
     media time deaf.
  4. Nothing cancels the video: AUDIO_ECHO_CANCEL only ducks the mic for
     150 ms after JARVIS's OWN speech.

THE STAGES (each behind its own setting; see core/config.py)
-----------------------------------------------------------
  A1  "hay" read as "hey" (core/wake_prefix).
  A2  WAKE_REANCHOR_MODE: a refused line is re-anchored at a later sentence
      (core/wake_prefix.reanchor); 'on' cuts the audio at the name
      (reanchor_cut_s), checks the voice and re-decodes it first.
  A3  WAKE_PREGATE_MODE 'shadow': openWakeWord scores the mic (scores only;
      ships 'off' - loading it is a native change with its own canary).
  B1  MIC_BUS_MODE: one always-open mic stream with a 30 s ring
      (core/mic_bus).
  B2  MEDIA_SEGMENT_S: over media, a capture is cut every 12 s and the next
      one starts 2 s earlier from the ring (segment_active).
  C1  MEDIA_AEC_MODE: echo cancellation against what the PC plays
      (core/audio_processor.MediaEchoCanceller, core/loopback_ref); 'on'
      decides only on the always-open bus (B1) - a stream per capture
      re-anchors it at every open (2026-10-09 review).
  D1  WAKE_PREGATE_MODE 'on': the wake word starts the capture (PregateTrigger).
  D2  WAKE_LOOPBACK_VETO: a wake the video itself said is vetoed (vetoed_by).
  D3  WAKE_DUCK_MODE: media ducked after a confirmed wake.
  D4  owner-voice score of every confirmed cut (shadow, scores only).

Everything logged here is numbers only - never a word that was heard.

THE 2026-10-09 REVIEW (two reviewers, 17 findings) added: a confirmed wake
whose name was already in a capture that became a turn, or heard while
JARVIS spoke, never re-seats a capture; the bus never seeds a capture with
ring audio from his own playback (PLAYBACK_TAIL_S) and never hands it to the
record taps; a segment's overlap is dropped once a turn is taken or he
spoke; the D1 cap applies only over media the canceller is not cancelling;
a wedged bus open is booked, logged and spoken; the headset barge-in runs on
the bus; a stale loopback (LOOPBACK_STALE_S) is no reference; the canceller
settles 0.5 s after each re-anchor and its anchor keeps the drift it
corrected; WAKE_PREGATE_MODE ships 'off'.
"""
from __future__ import annotations

import collections
import math
import threading
import time

MODES3 = ("off", "shadow", "on")
MODES2 = ("off", "on")

# ── constants (spec §4; not owner-facing) ─────────────────────────────────
SEGMENT_OVERLAP_S = 2.0       # B2: the next segment starts this far back
REANCHOR_PAD_S = 0.3          # A2: the cut starts this far before the name
PREGATE_PREROLL_S = 1.5       # D1: the cut starts this far before the hit
PREGATE_CAP_S = 10.0          # D1: a pre-gate capture is at most this long
CONFIRM_S = 2.5               # D1: Parakeet confirms on this much of the cut
PREGATE_RATE_PER_MIN = 12     # D1: more triggers than this a minute ...
PREGATE_BUMP = 0.05           # ... raise the threshold by this ...
PREGATE_BUMP_S = 60.0         # ... for this long
PREGATE_REFRACTORY_S = 1.5    # one trigger per word (core/wake_word COOLDOWN)
VETO_WINDOW_S = 1.0           # D2: a mic hit within this of ...
VETO_SCORE = 0.3              # ... a loopback score at least this is vetoed
DUCK_CAP_S = 8.0              # D3: a wake duck never lasts longer than this
DUCK_TAIL_S = 0.3             # D3: held this long past the end of speech
PLAYBACK_TAIL_S = 0.3         # B1: ring audio this soon after JARVIS's own
                              # playback ended (speaker -> mic + room decay)
                              # never seeds a capture (2026-10-09 review)
LOOPBACK_STALE_S = 1.0        # C1: no loopback packet this long = no
                              # reference (soundcard pads silence, so a
                              # live reader never goes this quiet)
TURN_PEAK_BEFORE_S = 1.0      # A3: a turn's peak score is the max over
TURN_PEAK_AFTER_S = 3.0       #     [capture start - 1 s, + 3 s]
SHADOW_THRESHOLDS = (0.10, 0.15, 0.20, 0.30, 0.50)
THRESHOLD_MIN, THRESHOLD_MAX = 0.10, 0.50     # WAKE_PREGATE_THRESHOLD range
DATA_THRESHOLD_MIN, DATA_THRESHOLD_MAX = 0.10, 0.30   # from A3's week
DEFAULT_THRESHOLD = 0.15
SCORE_HISTORY_S = 30.0        # how long a score track remembers
COUNTER_WINDOW_S = 60.0
AMBIENT_MATCH_S = 10.0        # an ambient wake hit with no main turn this
                              # long after it counts as "main dropped"


# The shipped defaults of every listening-over-media setting - the ONE
# copy the monolith's fallbacks read; core/config.py, the Settings SCHEMA
# and tools/user_settings.example.json carry the same values and
# tests/test_listen_media.py pins all four together (the stale-duplicate
# rule). Owner-facing: the modes and the threshold. Constants (core/config
# only, not in the GUI): MEDIA_SEGMENT_S, MEDIA_AEC_BACKEND.
DEFAULTS = {
    "WAKE_REANCHOR_MODE": "shadow",     # A2: OD-3 turns it 'on'
    "WAKE_PREGATE_MODE": "off",         # A3 / D1: 'shadow' after a canary
    "WAKE_PREGATE_THRESHOLD": 0.15,     # D1; later from A3's week of data
    "MIC_BUS_MODE": "off",              # B1: 'on' after the canary
    "MEDIA_AEC_MODE": "off",            # C1: 'shadow' after OD-2
    "WAKE_LOOPBACK_VETO": "shadow",     # D2 (needs the loopback)
    "WAKE_DUCK_MODE": "off",            # D3: OD-5
    "WAKE_BARGEIN_MODE": "off",         # D5: OD-7
    "MEDIA_SEGMENT_S": 12.0,            # B2 (0 = off)
    "MEDIA_AEC_BACKEND": "nlms",        # C3's 'windows_comms' is OD-4
}
OWNER_FACING = ("WAKE_REANCHOR_MODE", "WAKE_PREGATE_MODE",
                "WAKE_PREGATE_THRESHOLD", "MIC_BUS_MODE", "MEDIA_AEC_MODE",
                "WAKE_LOOPBACK_VETO", "WAKE_DUCK_MODE", "WAKE_BARGEIN_MODE")


def mode3(value, default: str = "off") -> str:
    """'off' | 'shadow' | 'on'; anything else is ``default``. Never
    raises."""
    try:
        v = str(value).strip().lower()
    except Exception:
        return default
    return v if v in MODES3 else default


def mode2(value, default: str = "off") -> str:
    """'off' | 'on' (True / False read as on / off); anything else is
    ``default``. Never raises."""
    if value is True:
        return "on"
    if value is False or value is None:
        return "off" if value is not None else default
    try:
        v = str(value).strip().lower()
    except Exception:
        return default
    return v if v in MODES2 else default


def threshold_setting(value, default: float = DEFAULT_THRESHOLD) -> float:
    """WAKE_PREGATE_THRESHOLD clamped to THRESHOLD_MIN..THRESHOLD_MAX; a
    value that is not a number is ``default``. Never raises."""
    try:
        v = float(value)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return min(THRESHOLD_MAX, max(THRESHOLD_MIN, v))


def threshold_from_peaks(peaks, default: float = DEFAULT_THRESHOLD) -> float:
    """D1's threshold from A3's shadow data: the 10th percentile of the
    owner's accepted turns' peak scores, clamped to 0.10-0.30; ``default``
    with fewer than 10 peaks. Never raises."""
    try:
        vals = sorted(float(p) for p in peaks if math.isfinite(float(p)))
        if len(vals) < 10:
            return default
        k = (len(vals) - 1) * 0.10
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        p10 = vals[lo] + (vals[hi] - vals[lo]) * (k - lo)
        return min(DATA_THRESHOLD_MAX, max(DATA_THRESHOLD_MIN, p10))
    except Exception:
        return default


# ── word timing (A2) ──────────────────────────────────────────────────────
def segment_word_times(segments, text) -> "list | None":
    """Whisper's word timing for the re-anchor: ``[[word_index, start_s],
    ...]`` for the FIRST word of each segment (faster-whisper gives segment
    times, not word times; a sentence the re-anchor picks usually starts a
    segment). ``segments`` are (segment_text, start_s) pairs in order;
    ``text`` the final transcript - the times are dropped (None) when its
    word count differs from the segments' (a replacement changed it).
    Never raises."""
    try:
        out = []
        n = 0
        for seg_text, start in segments:
            words = str(seg_text or "").split()
            if not words:
                continue
            st = float(start)
            if math.isfinite(st):
                out.append([n, round(max(0.0, st), 3)])
            n += len(words)
        if not out or n != len(str(text or "").split()):
            return None
        return out
    except Exception:
        return None


def _time_of_word(word_t, index: int) -> "float | None":
    for i, t in word_t or ():
        if int(i) == int(index):
            return float(t)
    return None


def reanchor_cut_s(r, conf, pad: float = REANCHOR_PAD_S) -> "float | None":
    """Where the re-anchored command's audio starts in the capture (seconds,
    >= 0): the name's time, else the time of the sentence's first word, minus
    ``pad``. None - do not re-anchor - when the transcript carries no timing
    for either word, or its timing was taken for a different word count
    (``conf['n_words']`` != r.words). ``r`` is a core/wake_prefix.Reanchor.
    Never raises."""
    try:
        if r is None or not isinstance(conf, dict):
            return None
        wt = conf.get("word_t")
        if not wt or int(conf.get("n_words", -1)) != int(r.words):
            return None
        t = _time_of_word(wt, r.name_word)
        if t is None:
            t = _time_of_word(wt, r.start_word)
        if t is None:
            return None
        return max(0.0, float(t) - float(pad))
    except Exception:
        return None


# ── C1 / D3: is the PC playing into a headset? ────────────────────────────
_HEADSET_WORDS = ("headset", "headphone", "earphone", "earbud", "hands-free",
                  "handsfree", "airpods", "buds")


def output_is_headset(endpoint_name, headset_hint="") -> bool:
    """True when the render endpoint the PC plays to is a headset: its name
    holds the auto-switch's headset name (AUDIO_AUTOSWITCH_HEADSET) or a
    headset word. No acoustic echo path then:
    the canceller is bypassed and media is not ducked. Never raises."""
    try:
        name = str(endpoint_name or "").lower()
        if not name:
            return False
        hint = str(headset_hint or "").strip().lower()
        if hint and hint in name:
            return True
        return any(w in name for w in _HEADSET_WORDS)
    except Exception:
        return False


# ── B2: overlapping segments while media plays ────────────────────────────
def segment_active(*, bus_on, wake_mode, media, aec_mode,
                   segment_s) -> bool:
    """B2 applies: the mic bus is on (the overlap is read from its ring),
    wake-word mode is on, the PC is playing audio, AEC is not 'on' (with the
    echo cancelled the capture ends on its own) and MEDIA_SEGMENT_S > 0."""
    try:
        return (bool(bus_on) and bool(wake_mode) and bool(media)
                and mode3(aec_mode) != "on" and float(segment_s or 0) > 0)
    except Exception:
        return False


# ── score tracks (A3 / D1 / D2) ───────────────────────────────────────────
class ScoreTrack:
    """openWakeWord scores with their time (monotonic seconds of the frame's
    end), the last SCORE_HISTORY_S of them. Thread-safe."""

    def __init__(self, history_s: float = SCORE_HISTORY_S):
        self._hist = float(history_s)
        self._mu = threading.Lock()
        self._d: "collections.deque[tuple[float, float]]" = collections.deque()

    def add(self, t: float, score: float) -> None:
        try:
            t, score = float(t), float(score)
        except Exception:
            return
        with self._mu:
            self._d.append((t, score))
            cut = t - self._hist
            while self._d and self._d[0][0] < cut:
                self._d.popleft()

    def max_between(self, t0: float, t1: float) -> "float | None":
        """The highest score with t0 <= t <= t1, or None (no frame)."""
        with self._mu:
            vals = [s for t, s in self._d if t0 <= t <= t1]
        return max(vals) if vals else None

    def last_time(self) -> "float | None":
        with self._mu:
            return self._d[-1][0] if self._d else None


def vetoed_by(loop_track: "ScoreTrack | None", t: float,
              window: float = VETO_WINDOW_S,
              score: float = VETO_SCORE) -> bool:
    """D2: True when the loopback (what the PC played) scored the wake word
    at least ``score`` within ``window`` seconds of ``t`` - the video said
    "Jarvis", not the owner. No loopback track = no veto. Never raises."""
    try:
        if loop_track is None:
            return False
        best = loop_track.max_between(float(t) - window, float(t) + window)
        return best is not None and best >= float(score)
    except Exception:
        return False


class PregateTrigger:
    """D1's trigger policy over a stream of (time, score): a trigger when the
    score reaches the threshold, at most one per PREGATE_REFRACTORY_S. More
    than PREGATE_RATE_PER_MIN in a minute raises the threshold by
    PREGATE_BUMP for PREGATE_BUMP_S (the flood is reported once through
    ``flooded``). Also counts, for the shadow minute line, the events a
    stream would have made at each of SHADOW_THRESHOLDS. Thread-safe."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 rate_per_min: int = PREGATE_RATE_PER_MIN):
        self.threshold = threshold_setting(threshold)
        self._rate = int(rate_per_min)
        self._mu = threading.Lock()
        self._last_trigger = None
        self._recent: "collections.deque[float]" = collections.deque()
        self._bump_until = None
        self.flooded = 0
        self._last_event = {thr: None for thr in SHADOW_THRESHOLDS}
        self.events = dict.fromkeys(SHADOW_THRESHOLDS, 0)

    def effective_threshold(self, now: float) -> float:
        with self._mu:
            return self._effective(now)

    def _effective(self, now: float) -> float:
        thr = self.threshold
        if self._bump_until is not None and now < self._bump_until:
            thr += PREGATE_BUMP
        return min(1.0, thr)

    def offer(self, t: float, score: float) -> bool:
        """One frame. True = a trigger (the caller starts a capture)."""
        try:
            t, score = float(t), float(score)
        except Exception:
            return False
        with self._mu:
            for thr in SHADOW_THRESHOLDS:
                last = self._last_event[thr]
                if score >= thr and (last is None
                                     or t - last >= PREGATE_REFRACTORY_S):
                    self._last_event[thr] = t
                    self.events[thr] += 1
            if score < self._effective(t):
                return False
            if (self._last_trigger is not None
                    and t - self._last_trigger < PREGATE_REFRACTORY_S):
                return False
            while self._recent and t - self._recent[0] > 60.0:
                self._recent.popleft()
            if len(self._recent) >= self._rate:
                if self._bump_until is None or t >= self._bump_until:
                    self.flooded += 1
                self._bump_until = t + PREGATE_BUMP_S
                return False
            self._recent.append(t)
            self._last_trigger = t
            return True

    def take_events(self) -> dict:
        """The shadow event counts since the last call (and reset)."""
        with self._mu:
            out = dict(self.events)
            self.events = dict.fromkeys(SHADOW_THRESHOLDS, 0)
            return out


# ── the per-minute listening line (spec §8 canary) ────────────────────────
class MinuteCounter:
    """What the listening lane did in each minute that had media, as ONE log
    line (numbers only). Kinds are free-form counters; ``gauge`` keeps the
    last value of a reading (ERLE). Thread-safe; ``clock`` injectable."""

    ORDER = ("captures", "refused", "segments", "reanchor_would",
             "reanchor_on", "reanchor_failed", "pregate_hits",
             "pregate_confirms", "pregate_drops", "vetoes", "would_veto",
             "ducks", "mic_closed_s", "ambient_main_dropped")

    def __init__(self, window_s: float = COUNTER_WINDOW_S,
                 clock=time.monotonic):
        self._window = float(window_s)
        self._clock = clock
        self._mu = threading.Lock()
        self._start = None
        self._media = False
        self._c: dict = {}
        self._g: dict = {}

    def note(self, kind: str, n: float = 1) -> None:
        try:
            with self._mu:
                if self._start is None:
                    self._start = float(self._clock())
                self._c[kind] = self._c.get(kind, 0) + n
        except Exception:
            pass

    def gauge(self, kind: str, value) -> None:
        try:
            with self._mu:
                self._g[kind] = value
        except Exception:
            pass

    def mark_media(self) -> None:
        try:
            with self._mu:
                if self._start is None:
                    self._start = float(self._clock())
                self._media = True
        except Exception:
            pass

    def snapshot(self) -> dict:
        with self._mu:
            return dict(self._c)

    def tick(self) -> "str | None":
        """The line for a finished window that had media (None otherwise);
        a finished window starts a fresh one either way."""
        try:
            now = float(self._clock())
            with self._mu:
                if self._start is None:
                    self._start = now
                    return None
                if now - self._start < self._window:
                    return None
                c, g, media = self._c, self._g, self._media
                secs = now - self._start
                self._c, self._media, self._start = {}, False, now
                self._g = {}
            if not media:
                return None
            return self.format(c, g, secs)
        except Exception:
            return None

    @classmethod
    def format(cls, c: dict, g: dict, secs: float) -> str:
        def n(k):
            v = c.get(k, 0)
            return f"{v:.1f}" if isinstance(v, float) else str(int(v))
        parts = [f"captures {n('captures')}", f"refused {n('refused')}",
                 f"segments {n('segments')}",
                 f"re-anchor would {n('reanchor_would')} / on "
                 f"{n('reanchor_on')} / failed {n('reanchor_failed')}",
                 f"pre-gate hits {n('pregate_hits')} / confirms "
                 f"{n('pregate_confirms')} / drops {n('pregate_drops')}",
                 f"vetoes {n('vetoes')} (would {n('would_veto')})",
                 f"ducks {n('ducks')}",
                 f"mic closed {n('mic_closed_s')} s",
                 f"ambient heard the name, main dropped "
                 f"{n('ambient_main_dropped')}"]
        erle = g.get("erle_db")
        if erle is not None:
            parts.append(f"aec erle {float(erle):.1f} dB")
        extra = sorted(k for k in c if k not in cls.ORDER)
        for k in extra:
            parts.append(f"{k} {n(k)}")
        return (f"[listen-media] {secs:.0f} s with media - "
                + ", ".join(parts))


class AmbientMatch:
    """"Ambient heard Jarvis, main dropped" (spec §8, target 0): the ambient
    listener's wake hits while JARVIS is awake, matched against the main
    loop's accepted wake-word turns. A hit with no accepted turn within
    AMBIENT_MATCH_S after it is one the main loop lost. Thread-safe."""

    def __init__(self, window_s: float = AMBIENT_MATCH_S):
        self._w = float(window_s)
        self._mu = threading.Lock()
        self._hits: "collections.deque[float]" = collections.deque()
        self._turns: "collections.deque[float]" = collections.deque()

    def ambient_hit(self, t: float) -> None:
        with self._mu:
            self._hits.append(float(t))

    def main_turn(self, t: float) -> None:
        with self._mu:
            self._turns.append(float(t))

    def settle(self, now: float) -> int:
        """How many hits older than the window found no turn (they are
        dropped from the books either way)."""
        lost = 0
        with self._mu:
            # A turn the ambient line ARRIVED after (2.5 s batches + a
            # decode) still matches: allow the window on both sides.
            while self._hits and now - self._hits[0] >= self._w:
                h = self._hits.popleft()
                if not any(h - self._w <= t <= h + self._w
                           for t in self._turns):
                    lost += 1
            while self._turns and now - self._turns[0] > 3 * self._w:
                self._turns.popleft()
        return lost
