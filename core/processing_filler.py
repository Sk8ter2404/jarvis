"""Processing filler — a short butler line while a slow VOICE turn thinks.

Owner request 2026-09-06: when a spoken command takes a while, JARVIS should say
"Just a moment, sir." instead of sitting in silence. OFF by default
(core.config.PROCESSING_FILLER_ENABLED = False).

This module holds ALL the scheduling / cancel / cache logic and is stdlib-only
(no numpy, no sounddevice, no monolith import) so CI can test it with a fake
clock and recorded thread targets. bobert_companion.py supplies the glue:

  * ``play_fn(turn, stage)``  — plays a pre-rendered clip under _SPEAK_LOCK and
    returns 'played' | 'retry' | 'skipped'. It must call ``claim()`` INSIDE the
    speech lock and ``play_done()`` after a successful claim.
  * ``suppressed_fn()``       — arm-time gate: a reason string (or None) when
    the turn must get no filler (disabled, muted, standby, ...). A transient
    background mic capture is NOT an arm-time reason; begin_capture() /
    end_capture() make claim() answer 'busy' while any capture is live.
  * ``delays_fn()``           — returns the raw (first, still) delays.

Captures (get_mic_buffer, record_speech) on the arming thread are turn
activity, like speech; captures on any other thread (the standby-audio loop)
only defer a claim while they are live.

Timeline of one armed turn (one daemon thread named 'processing-filler'):

  stage 1  fires at t0 + first when NOTHING has spoken in the turn yet
           (retrying up to first_retry_s while a capture holds it off).
  stage 2  fires once after ``still`` seconds of turn silence, counted from the
           later of t0 and the last speech mark (see note_speech()).

Design rules (see the review notes in the 2026-09-29 plan):
  * cancel-if-not-started, finish-if-started — the filler is never cut, because
    every cut path (sd.stop off the reaper, _tts_interrupt, a _tts_interrupt_seq
    bump) is a known crash / silenced-answer class;
  * nothing here ever calls time.sleep, joins a thread, or touches audio;
  * the lock order is always SPEECH LOCK -> filler lock, never the reverse.
"""
from __future__ import annotations

import math
import re
import threading
import time

__all__ = [
    "FIRST_LINES", "STILL_LINES", "QUIET_IMPERATIVES", "DEFAULT_FIRST",
    "DEFAULT_STILL", "normalise_line", "sanitize_delays", "is_quiet_command",
    "FillerTurn", "ProcessingFiller", "ClipCache",
]

# ── line banks ───────────────────────────────────────────────────────────────
# Rules (pinned by tests/test_processing_filler.py):
#   * no line contains "jarvis" — request_tts_interrupt's echo gate reads the
#     text currently playing and would refuse a genuine barge-in;
#   * no line may equal (after normalise_line) a core/prompts.py minimal
#     acknowledgement ('Working.' / 'Working on it.' / 'One moment.') or any
#     line of the monolith's _MID_TASK_STATUS_LINES bank, so the owner never
#     hears the same sentence twice in one turn;
#   * each line is at most 7 words, and short enough that its Kokoro render
#     fits ClipCache's max_secs (a longer render is refused, so the line
#     would simply never play).
FIRST_LINES: tuple[str, ...] = (
    "Just a moment, sir.",
    "Allow me a moment, sir.",
    "Looking into it now, sir.",
)
STILL_LINES: tuple[str, ...] = (
    "Still working on it, sir.",
    "Nearly there, sir.",
    "A little longer, sir.",
)

DEFAULT_FIRST = 2.5
DEFAULT_STILL = 12.0
_FIRST_MIN, _FIRST_MAX = 0.5, 60.0
_STILL_MAX = 600.0

# Stop / cancel / quiet commands. A turn made of one of these must NOT get a
# "one moment" — the owner asked for silence. The monolith passes
# core.tone_detector._CLIPPED_IMPERATIVES and only the phrases present in BOTH
# sets count (so the tone detector stays the single owner of the vocabulary);
# affirmatives from that set ("go", "do it", "now") are deliberately excluded.
QUIET_IMPERATIVES: frozenset[str] = frozenset({
    "stop", "wait", "no", "shut up", "be quiet", "quiet", "enough",
    "cancel", "kill it", "abort",
})
_QUIET_MAX_WORDS = 3
_COURTESY_WORDS = frozenset({"jarvis", "sir", "please", "ok", "okay", "just"})


def normalise_line(text: str) -> str:
    """Lowercase, drop ', sir' and punctuation, collapse whitespace. Used to
    compare spoken lines across banks ("One moment, sir." == "One moment.")."""
    s = str(text or "").lower()
    s = re.sub(r"[^a-z' ]+", " ", s)
    words = [w for w in s.split() if w != "sir"]
    return " ".join(words).strip()


def _to_float(value, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f


def sanitize_delays(first, still) -> tuple[float, float | None]:
    """Clamp the user-settable delays. Nothing upstream clamps
    data/user_settings.json values, so this is the one place that does.

    first: not finite or <= 0 -> DEFAULT_FIRST; then clamped to [0.5, 60].
    still: not finite -> DEFAULT_STILL; <= first -> None (stage 2 OFF — the
           documented way to disable it); clamped to <= 600.
    """
    f = _to_float(first, DEFAULT_FIRST)
    if not math.isfinite(f) or f <= 0:
        f = DEFAULT_FIRST
    f = min(max(f, _FIRST_MIN), _FIRST_MAX)
    s = _to_float(still, DEFAULT_STILL)
    if not math.isfinite(s):
        s = DEFAULT_STILL
    if s <= f:
        return float(f), None
    return float(f), float(min(s, _STILL_MAX))


def is_quiet_command(text: str, clipped=None) -> bool:
    """True for a short stop / cancel / quiet utterance ("stop", "shut up
    jarvis", "cancel that"). ``clipped`` is the tone detector's imperative
    vocabulary; only phrases in both it and QUIET_IMPERATIVES count."""
    try:
        if clipped is None:
            vocab = set(QUIET_IMPERATIVES)
        else:
            vocab = {str(p).lower() for p in clipped} & set(QUIET_IMPERATIVES)
        clean = re.sub(r"[^a-z' ]+", " ", str(text or "").lower())
        words = [w for w in clean.split() if w not in _COURTESY_WORDS]
        if not words or len(words) > _QUIET_MAX_WORDS:
            return False
        joined = " " + " ".join(words) + " "
        return any((" " + p + " ") in joined for p in vocab)
    except Exception:
        return False


def _default_wait(evt: threading.Event, timeout: float) -> bool:
    return evt.wait(max(0.0, float(timeout)))


class FillerTurn:
    """State of one armed voice turn. Mutated only under the filler lock."""

    __slots__ = ("t0", "last_mark", "spoke", "fired", "cancel", "first",
                 "still", "owner")

    def __init__(self, t0: float, first: float, still: float | None,
                 owner: int | None = None):
        self.t0 = t0
        # Thread ident of the voice turn that armed this (the dispatch
        # thread). Only a mic capture on THIS thread is turn activity -- a
        # background capture (the standby-audio loop polls get_mic_buffer
        # every few seconds all day) must never restart the silence clock.
        self.owner = owner
        self.last_mark = t0
        self.spoke = False
        self.fired: set[int] = set()
        self.cancel = threading.Event()
        self.first = first
        self.still = still


class ProcessingFiller:
    """Per-turn scheduler. One instance per process (the monolith's
    ``_processing_filler``); tests build their own."""

    def __init__(self, *, play_fn, suppressed_fn, delays_fn,
                 clock=time.monotonic, wait_fn=_default_wait,
                 thread_factory=threading.Thread, retry_s: float = 0.5,
                 max_retry_s: float = 30.0, first_retry_s: float = 3.0):
        self._play_fn = play_fn
        self._suppressed_fn = suppressed_fn
        self._delays_fn = delays_fn
        self._clock = clock
        self._wait = wait_fn
        self._thread_factory = thread_factory
        self._retry_s = float(retry_s)
        self._max_retry_s = float(max_retry_s)
        # Stage 1 retries this long on 'retry' (e.g. a background capture
        # holding the mic at t0 + first) instead of being dropped.
        self._first_retry_s = float(first_retry_s)
        self._lock = threading.Lock()
        self._current: FillerTurn | None = None
        self._closed = False
        self._playing = 0
        # Live mic captures (begin_capture / end_capture), ANY thread. While
        # non-zero claim() answers 'busy'. Counted under _lock so a capture
        # start and a claim are strictly ordered: either the claim came first
        # (begin_capture reports the clip playing and the caller waits it out)
        # or the capture did (the claim is refused).
        self._captures = 0
        self._idle = threading.Event()
        self._idle.set()
        self.last_reason = ""

    # ── lifecycle ───────────────────────────────────────────────────────
    def arm(self) -> FillerTurn | None:
        """Start a turn. Returns None (and creates NO thread) when the filler
        is latched off or ``suppressed_fn()`` gives a reason."""
        if self._closed:
            return None
        reason = self._suppressed_fn()
        if reason:
            self.last_reason = str(reason)
            return None
        first, still = sanitize_delays(*self._delays_fn())
        turn = FillerTurn(self._clock(), first, still,
                          owner=threading.get_ident())
        with self._lock:
            if self._closed:
                return None
            prev = self._current
            if prev is not None:
                prev.cancel.set()
            self._current = turn
        try:
            th = self._thread_factory(target=self._run, args=(turn,),
                                      name="processing-filler", daemon=True)
            th.start()
        except Exception:
            with self._lock:
                if self._current is turn:
                    self._current = None
            turn.cancel.set()
            return None
        return turn

    def disarm(self, turn: FillerTurn | None) -> None:
        """End a turn. Never joins. Safe with None."""
        if turn is None:
            return
        turn.cancel.set()
        with self._lock:
            if self._current is turn:
                self._current = None

    def cancel(self, reason: str = "") -> None:
        """Cancel the current turn's pending stages (a clip already playing
        finishes). Never raises."""
        if self._current is None:
            return
        with self._lock:
            t = self._current
            if t is not None:
                t.cancel.set()
                self.last_reason = str(reason or "cancel")

    def shutdown(self, reason: str = "") -> None:
        """Teardown latch: cancel the current turn and refuse every later
        arm(). Used on restart / shutdown / blue-green handoff."""
        with self._lock:
            self._closed = True
            t = self._current
            self._current = None
            self.last_reason = str(reason or "shutdown")
        if t is not None:
            t.cancel.set()

    def closed(self) -> bool:
        return self._closed

    def armed(self) -> bool:
        return self._current is not None

    # ── speech marks ────────────────────────────────────────────────────
    @staticmethod
    def _mark(t: FillerTurn | None, now, owner_only: bool) -> None:
        """Apply a speech mark to ``t``. Caller holds _lock."""
        if t is None or now is None:
            return
        if owner_only and t.owner != threading.get_ident():
            return
        t.spoke = True
        t.last_mark = now

    def note_speech(self, owner_only: bool = False) -> None:
        """Any real speech (or, with ``owner_only``, a mic capture on the
        turn's own thread) happened: stage 1 is permanently off for this turn
        and stage 2's silence clock restarts. A near no-op when nothing is
        armed (the default-off path): no lock, no clock read."""
        if self._current is None:
            return
        try:
            now = self._clock()
            with self._lock:
                self._mark(self._current, now, owner_only)
        except Exception:
            pass

    # ── mic captures ────────────────────────────────────────────────────
    def begin_capture(self) -> bool:
        """A mic capture is starting (any thread). Every later claim() is
        refused ('busy') until the matching end_capture(); a capture on the
        armed turn's own thread also counts as turn activity. Returns True
        when a clip claimed EARLIER is still playing -- the caller must then
        wait_idle() before opening the mic. Never raises."""
        try:
            now = self._clock() if self._current is not None else None
            with self._lock:
                self._captures += 1
                self._mark(self._current, now, True)
                return self._playing > 0
        except Exception:
            return False

    def end_capture(self) -> None:
        """The capture begun by begin_capture() ended. The turn-activity mark
        and the release happen in ONE critical section, so a poll can never
        see the capture gone while the silence clock still counts from its
        start. Never raises."""
        try:
            now = self._clock() if self._current is not None else None
            with self._lock:
                self._mark(self._current, now, True)
                self._captures = max(0, self._captures - 1)
        except Exception:
            pass

    def capturing(self) -> bool:
        return self._captures > 0

    # ── firing ──────────────────────────────────────────────────────────
    def claim(self, turn: FillerTurn, stage: int) -> str:
        """Decide, atomically, whether ``stage`` may play now. Must be called
        by play_fn while it holds the speech lock. Returns 'ok' | 'not-yet' |
        'busy' | 'gone' ('busy': a mic capture is live, retry later). An 'ok'
        marks the stage fired and the filler as playing — the caller MUST
        call play_done() afterwards."""
        with self._lock:
            if (self._closed or turn is not self._current
                    or turn.cancel.is_set() or stage in turn.fired):
                return "gone"
            if stage == 1 and turn.spoke:
                return "gone"
            if self._captures > 0:
                return "busy"
            if stage != 1:
                if turn.still is None:
                    return "gone"
                silent = self._clock() - max(turn.t0, turn.last_mark)
                if silent < turn.still - 0.05:
                    return "not-yet"
            turn.fired.add(stage)
            self._playing += 1
            self._idle.clear()
            return "ok"

    def play_done(self) -> None:
        with self._lock:
            self._playing = max(0, self._playing - 1)
            if self._playing == 0:
                self._idle.set()

    def playing(self) -> bool:
        return not self._idle.is_set()

    def wait_idle(self, timeout: float = 3.0) -> bool:
        """Bounded pure-Event wait until no filler clip is playing. True when
        idle. Never touches audio."""
        if self._idle.is_set():
            return True
        return bool(self._wait(self._idle, max(0.0, min(float(timeout), 3.0))))

    def _run(self, turn: FillerTurn) -> None:
        if self._wait(turn.cancel, turn.first):
            return
        retried = 0.0
        while not turn.spoke:
            try:
                r = self._play_fn(turn, 1)
            except Exception:
                r = None
            if r != "retry" or retried >= self._first_retry_s:
                break
            retried += self._retry_s
            if self._wait(turn.cancel, self._retry_s):
                return
        if turn.still is None:
            return
        retry_total = 0.0
        while True:
            if turn.cancel.is_set():
                return
            with self._lock:
                base = max(turn.t0, turn.last_mark)
            remaining = base + turn.still - self._clock()
            if remaining > 0:
                if self._wait(turn.cancel, remaining):
                    return
                continue
            try:
                r = self._play_fn(turn, 2)
            except Exception:
                return
            if r == "retry" and retry_total < self._max_retry_s:
                retry_total += self._retry_s
                if self._wait(turn.cancel, self._retry_s):
                    return
                continue
            return


class ClipCache:
    """Pre-rendered filler clips keyed on (key_fn(), text). ``get`` never
    renders; ``warm`` renders missing lines one at a time under ``lock`` (the
    speech lock, so synthesis stays single-threaded) and yields between lines
    so a blocked speaker is never starved."""

    def __init__(self, *, render_fn, lock, key_fn, max_secs: float = 2.5):
        self._render = render_fn
        self._lock = lock
        self._key_fn = key_fn
        # Keep max_secs + playback overhead under the monolith's bounded
        # end-of-turn wait (_FILLER_END_WAIT_S = 3.0) so a clip can never
        # outlive that wait into the next capture.
        self._max_secs = float(max_secs)
        self._mu = threading.Lock()
        self._clips: dict = {}
        # (key, text) renders refused as too long. Deterministic for a given
        # voice, so they are never re-rendered (and logged once) until the
        # voice key changes.
        self._rejected: set = set()
        self._warming = False

    def _prune(self, key) -> None:
        for k in [k for k in self._clips if k[0] != key]:
            del self._clips[k]
        self._rejected = {k for k in self._rejected if k[0] == key}

    def get(self, text: str):
        key = self._key_fn()
        with self._mu:
            v = self._clips.get((key, text))
        if v is None:
            return None
        audio, sr = v
        return audio.copy(), sr

    def available(self, lines) -> list[str]:
        key = self._key_fn()
        with self._mu:
            return [t for t in lines if (key, t) in self._clips]

    def missing(self, lines) -> list[str]:
        key = self._key_fn()
        with self._mu:
            self._prune(key)
            return [t for t in lines if (key, t) not in self._clips
                    and (key, t) not in self._rejected]

    def rejected(self) -> list[str]:
        key = self._key_fn()
        with self._mu:
            return [t for (k, t) in self._rejected if k == key]

    def put(self, text: str, clip, key=None) -> bool:
        """Store a rendered clip if it is usable. Returns True if stored. A
        render longer than max_secs is remembered as rejected (and logged
        once) so warm() never re-renders it for the same voice."""
        try:
            if clip is None:
                return False
            audio, sr = clip
            sr = int(sr)
            n = len(audio)
            if sr <= 0 or n <= 0:
                return False
            secs = n / float(sr)
        except Exception:
            return False
        k = self._key_fn() if key is None else key
        if secs > self._max_secs:
            with self._mu:
                first = (k, text) not in self._rejected
                self._rejected.add((k, text))
            if first:
                print(f"  [filler] clip {text!r} is {secs:.2f}s "
                      f"(> {self._max_secs:.1f}s); not used")
            return False
        with self._mu:
            self._clips[(k, text)] = (audio, sr)
        return True

    def warming(self) -> bool:
        return self._warming

    def warm(self, lines, stop_fn, wait_fn=_default_wait,
             retry_s: float = 0.25, max_tries: int = 40,
             yield_s: float = 0.3) -> int:
        """Render every missing line. Returns the number stored."""
        stored = 0
        pause = threading.Event()   # never set: a pure timed wait
        for text in self.missing(lines):
            if stop_fn():
                break
            tries = 0
            while not self._lock.acquire(blocking=False):
                tries += 1
                if tries >= max_tries or stop_fn():
                    return stored
                wait_fn(pause, retry_s)
            try:
                if stop_fn():
                    return stored
                try:
                    res = self._render(text)
                except Exception:
                    res = None
                # Key read AFTER the render: the first render may be what
                # imports the TTS module the key names.
                key = self._key_fn()
            finally:
                self._lock.release()
            if self.put(text, res, key=key):
                stored += 1
            # Yield AFTER releasing: Python's Lock is not FIFO, so without this
            # a _speak blocked on the lock could wait out the whole warm.
            wait_fn(pause, yield_s)
        return stored

    def warm_async(self, lines, stop_fn, wait_fn=_default_wait,
                   thread_factory=threading.Thread) -> bool:
        """Single-flight background warm on a 'filler-warm' daemon."""
        lines = tuple(lines)
        if not self.missing(lines):
            return False
        with self._mu:
            if self._warming:
                return False
            self._warming = True

        def _target():
            try:
                self.warm(lines, stop_fn, wait_fn)
            except Exception:
                pass
            finally:
                self._warming = False

        try:
            th = thread_factory(target=_target, name="filler-warm", daemon=True)
            th.start()
        except Exception:
            self._warming = False
            return False
        return True
