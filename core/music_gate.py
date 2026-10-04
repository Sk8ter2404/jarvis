"""core/music_gate.py — stop transcribing music, keep hearing the owner
(2026-10-04).

WHY THIS EXISTS
---------------
With music playing and wake-word mode on, the microphone never goes quiet:
record_speech only ends a capture after 1.3 s under the raw RMS gate, music
sits at 0.03-0.07, so every capture runs to MAX_RECORDING_SECS (30 s) and the
next one starts at once. Measured on 2026-10-04 (13:49-15:40 session):

  * every capture: Parakeet decodes it on the CPU (1.2 s for 30 s), Silero
    hears singing as speech, so the "no-wake" rescue re-decodes it with
    Whisper on the 1650 — 427 rescues of 536 Parakeet decodes that session,
    1,381 rescues 10-02..10-04 that made 3 owner turns;
  * the ambient listener batches every 2.5 s of the same frames through
    Whisper large-v3-turbo (its 0.003 RMS gate always passes music): ~15
    decodes a minute, each ~0.9 s of the 1650 plus ~0.9 CPU-s;
  * altogether the 1650 sat ~35 % busy and the listening threads burned ~39
    CPU-s a minute transcribing lyrics nobody acts on.

THE GATE (MUSIC_GATE_MODE 'off' | 'shadow' | 'on'; shipped 'shadow')
-------------------------------------------------------------------
Music mode = the PC is playing audio (its playback meter at or over
MEDIA_VOICE_GATE_PEAK — the media gate's own test — or sustained room music
from the spectral detector) AND a line needs the wake word right now (standby
/ sleep, wake-word mode, or AMBIENT_MUSIC_REFUSE_WAKE). In music mode, 'on':

  1. Ambient listener: no batch is transcribed (no Whisper decode, no
     voice-ID). An ambient line makes no turn, and voice-ID cannot pick the
     owner out of music (see ambient_decision).
  2. Parakeet's rescue ('empty' / 'no-wake') runs only with a hint that the
     owner addressed JARVIS: a "Jarvis"-like word among Parakeet's first
     three, or the capture in the owner's voice (the media gate's own check
     and floor, shared and memoised, so a turn that goes on pays for it
     once). Voice-ID unavailable = rescue (fail open). 'check-failed' always
     rescues. The voice hint is generous on purpose: lyrics passed its 0.45
     floor on about half the batches of 10-04, so it costs savings, never a
     turn; the text hint does most of the filtering.

A capture is NEVER cut short over music (review 2026-10-04). The microphone
stream is opened per capture, so nothing is recorded between the end of one
capture and the next "Recording…": over music that deaf gap measured a median
5-6 s after a 30 s capture whose rescue ran (n=46 / 26, the 13:49 and 13:09
sessions of 10-04) and ~2 s when no rescue ran (n=10 / 6): ~14 % of the
time deaf today (5 s per 35), 6-10 % with 'on' skipping the rescue on lyrics
(it still rescues the ~half its voice hint passes). A 10 s cap would put a
gap after every 10 s instead of every 30 s — 17-26 % of the time deaf — and
split a command at three times as many boundaries ("Jarvis, pause the
music", 10-04 13:21, was spoken ~12-15 s into its capture: past a 10 s cut).
Skipping the rescue is what shortens the gap; a shorter capture needs a
gap-free capture first.

Wake-word detection (Parakeet on every capture, the rescue with a hint) and
owner-voice detection (the media gate and the room-talk check on the main
loop's captures) keep running; only transcription of music stops.

'shadow' changes nothing — not even the main loop's time: the rescue runs as
always and the voice is NOT asked before it; only a rescue that made a line
the wake gates pass is voice-checked afterwards (the same check the media
gate of that turn then reuses from the memo). A rescue without a wake hint
counts as "would skip" — an upper bound, since 'on' still rescues the ones
in the owner's voice — and one whose line passed the wake gates in a voice
that is not the owner's would have been LOST, and is logged. Ambient lines
'on' would not have transcribed are counted, those with the wake word and
those voice-ID named as an enrolled speaker apart. Turn it 'on' once a week
of shadow minutes shows "lost: rescues 0". Every minute that had music logs
one counter line.

Pure policy and counters, stdlib only. The monolith owns the live inputs
(meters, voice-ID, gates) and calls these with plain values.
"""
from __future__ import annotations

import difflib
import re
import threading
import time

MODES = ("off", "shadow", "on")
DEFAULT_MODE = "shadow"
WAKE_WORD = "jarvis"
HINT_WORDS = 3            # the wake word may sit at word 1-3 (core/wake_prefix)
HINT_MIN_RATIO = 0.6      # difflib ratio to "jarvis" that counts as a hint
COUNTER_WINDOW_S = 60.0

# voice verdicts (core/learn_gate's names; repeated as plain strings so this
# module imports nothing)
OWNER, NOT_OWNER, UNSURE, UNAVAILABLE = ("owner", "not_owner", "unsure",
                                         "unavailable")

_WORD_RE = re.compile(r"[a-z']+")


def mode_setting(value) -> str:
    """'off' | 'shadow' | 'on'; anything else is 'off'. Never raises."""
    try:
        v = str(value).strip().lower()
    except Exception:
        return "off"
    return v if v in MODES else "off"


def music_mode(*, playing, standby=False, wake_mode=False,
               music_refuse=False) -> bool:
    """The PC (or the room) is playing music AND only a wake-word line can
    get through right now."""
    return bool(playing) and bool(standby or wake_mode or music_refuse)


def wake_hint(text, n_words: int = HINT_WORDS,
              min_ratio: float = HINT_MIN_RATIO) -> bool:
    """A word that sounds like "Jarvis" among the first ``n_words`` of
    ``text`` (difflib ratio >= ``min_ratio``: "jervis", "travis", "javis",
    "garvis" yes; "the", "jar", "service" no). Parakeet's text failed the
    exact wake rule already; this only asks whether a rescue is worth a
    Whisper decode. Never raises."""
    try:
        words = _WORD_RE.findall(str(text or "").lower())[:max(0, n_words)]
        for w in words:
            w = w.strip("'")
            if len(w) < 4:
                continue
            if difflib.SequenceMatcher(None, w, WAKE_WORD).ratio() >= min_ratio:
                return True
        return False
    except Exception:
        return False


def _gated(mode: str) -> str:
    return "skip" if mode == "on" else "shadow"


def rescue_decision(mode, music, why, text, voice_fn) -> str:
    """Parakeet's rescue for one capture: '' = rescue as today, 'skip' (mode
    'on'), 'shadow' (no wake hint; rescue anyway). ``voice_fn()`` -> a voice
    verdict, asked only in 'on' and only when the text gives no hint.
    'shadow' never asks it here: the rescue runs either way, so the caller
    asks the voice AFTER the rescue, and only for a line that matters
    (shadow_lost). Any error rescues."""
    try:
        mode = mode_setting(mode)
        if mode == "off" or not music or why not in ("empty", "no-wake"):
            return ""
        if wake_hint(text):
            return ""
        if mode == "shadow":
            return "shadow"
        voice = voice_fn() if callable(voice_fn) else UNAVAILABLE
        if voice in (OWNER, UNSURE, UNAVAILABLE):
            return ""
        return "skip"
    except Exception:
        return ""


def shadow_lost(line_passes, voice_fn) -> bool:
    """Shadow, after a rescue 'on' might have skipped: would 'on' have LOST
    its line? Only when the line passes the wake gates (``line_passes``)
    AND the voice is not the owner's ('on' rescues on OWNER, UNSURE and
    UNAVAILABLE). ``voice_fn`` is asked only for such a line. Any error =
    not lost (this only counts; nothing is dropped)."""
    try:
        if not line_passes:
            return False
        voice = voice_fn() if callable(voice_fn) else UNAVAILABLE
        return voice not in (OWNER, UNSURE, UNAVAILABLE)
    except Exception:
        return False


def ambient_decision(mode, music) -> str:
    """One ambient batch: '' = transcribe as today, 'skip' (mode 'on'),
    'shadow' (would skip). Over music no ambient batch is transcribed: an
    ambient line makes no turn (the main loop hears the owner's wake word
    and checks his voice on its own captures), and voice-ID cannot pick the
    owner out of music — on 10-04 (13:53-17:00, 1,917 ambient lines over
    music) half the lyric batches scored >= 0.45 against his voiceprint
    and 5.8 % >= 0.72, while his own commands over media score 0.46-0.52.
    Never raises."""
    try:
        mode = mode_setting(mode)
        if mode == "off" or not music:
            return ""
        return _gated(mode)
    except Exception:
        return ""


class MinuteCounter:
    """What the listening lane did in each minute that had music, as ONE log
    line per such minute (numbers only, never words). Thread-safe; any
    thread may note() and tick(). ``clock`` is injectable."""

    KINDS = ("whisper_turn", "whisper_ambient", "whisper_other", "rescue",
             "retry", "parakeet", "voice_id",
             "skip_ambient", "skip_rescue",
             "would_ambient", "would_rescue",
             "lost_rescue", "lost_ambient", "lost_ambient_wake",
             "lost_ambient_named")

    def __init__(self, window_s: float = COUNTER_WINDOW_S,
                 clock=time.monotonic):
        self._window = float(window_s)
        self._clock = clock
        self._mu = threading.Lock()
        self._start = None
        self._music = False
        self._counts = dict.fromkeys(self.KINDS, 0)

    def note(self, kind: str, n: int = 1) -> None:
        try:
            with self._mu:
                if self._start is None:
                    self._start = float(self._clock())
                if kind in self._counts:
                    self._counts[kind] += int(n)
        except Exception:
            pass

    def mark_music(self) -> None:
        try:
            with self._mu:
                if self._start is None:
                    self._start = float(self._clock())
                self._music = True
        except Exception:
            pass

    def snapshot(self) -> dict:
        with self._mu:
            return dict(self._counts)

    def tick(self, mode: str = "shadow") -> "str | None":
        """The line for a window that is over and had music (None
        otherwise); a finished window starts a fresh one either way."""
        try:
            now = float(self._clock())
            with self._mu:
                if self._start is None:
                    self._start = now
                    return None
                if now - self._start < self._window:
                    return None
                c, music = self._counts, self._music
                secs = now - self._start
                self._counts = dict.fromkeys(self.KINDS, 0)
                self._music = False
                self._start = now
            if not music:
                return None
            return self.format(c, secs, mode)
        except Exception:
            return None

    @staticmethod
    def format(c: dict, secs: float, mode: str) -> str:
        mode = mode_setting(mode)
        whisper = c["whisper_turn"] + c["whisper_ambient"] + c["whisper_other"]
        line = (f"[music-gate] {mode}: {secs:.0f} s with music — whisper "
                f"{whisper} (ambient {c['whisper_ambient']}, turns "
                f"{c['whisper_turn']}, other {c['whisper_other']}; rescues "
                f"{c['rescue']}, no-VAD retries {c['retry']}), parakeet "
                f"{c['parakeet']}, voice-ID {c['voice_id']}")
        if mode == "on":
            line += (f"; skipped: ambient {c['skip_ambient']}, rescues "
                     f"{c['skip_rescue']}")
        elif mode == "shadow":
            # rescues "<=": the voice is asked only after a rescue that made
            # a wake line (rescue_decision / shadow_lost), so this counts
            # every rescue without a wake hint.
            line += (f"; would skip: ambient {c['would_ambient']}, rescues "
                     f"<={c['would_rescue']}; lost: rescues "
                     f"{c['lost_rescue']}, ambient lines "
                     f"{c['lost_ambient']} (wake word in "
                     f"{c['lost_ambient_wake']}, voice-ID named "
                     f"{c['lost_ambient_named']})")
        return line
