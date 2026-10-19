"""core/room_noise.py - "how bad is the background noise?" answered from a
READING (2026-10-19).

Live 10:45: the owner asked "Jarvis, how bad is the background noise?" and
JARVIS said "roughly 42% of the current audio is non-essential interference".
Nothing measured anything: the number was invented and voiced - the twin of
the 2026-10-09 volume answer (core/audio_check.py reads the OUTPUT side;
nothing read the room).

``room_noise`` listens to the room for one second through the shared mic
(bobert_companion.get_mic_buffer, the same tap voice enrolment uses) and reads
whether the PC itself is making sound (core.media_gate.pc_audio_peak, JARVIS's
own processes left out), then says only what it measured. Anything it could not
read is said to be unknown, never guessed.

Layout: ``is_room_noise_question`` recognises the question; ``measure`` does
the I/O through injected callables (never raises); ``describe`` is pure.
"""
from __future__ import annotations

import math
import re

SAMPLE_S = 1.0
# RMS of the one-second sample (float audio, full scale 1.0).
QUIET_RMS = 0.002      # about -54 dBFS: a quiet room on a desk mic
SPEECH_RMS = 0.008     # the VAD threshold: above it, noise reads as speech
PC_PLAYING_PEAK = 0.01  # core.media_gate.audio_playing's threshold

_NOISE_RE = re.compile(
    r"\b(background (noise|sound)s?|noise level|ambient noise|noisy|noise|"
    r"loud (it is )?in here|how loud is (it|the room))\b", re.I)
_ASK_RE = re.compile(
    r"^\s*(hey\s+)?(jarvis[,.!\s]+)?(how|is|are|was|what's|whats|what is|"
    r"can you|could you|do you|does|check|tell me|any|much)\b", re.I)
# Not a question about the room's level: a sound to play, a change to make,
# the headset's noise cancelling, or "what was that noise" (identify a sound).
_NOT_RE = re.compile(
    r"\b(white|brown|pink|rain|ocean|fan)\s+noise\b|noise[- ]?cancel|"
    r"\b(play|reduce|cancel|filter|turn|mute|lower|stop|make|remove|block|"
    r"record)\b|\bwhat\s*(was|is|'s)\s+that\b|\bwhat\s+(made|makes)\b", re.I)


def is_room_noise_question(text) -> bool:
    """True when the owner asks how noisy the room is / whether background
    noise is a problem - a question the room READING answers."""
    t = str(text or "").strip()
    if not t or len(t) > 160 or _NOT_RE.search(t):
        return False
    return bool(_NOISE_RE.search(t) and _ASK_RE.search(t))


def _rms(buf):
    try:
        import numpy as np
        a = np.asarray(buf, dtype=np.float64).reshape(-1)
        if a.size == 0:
            return None
        v = float(np.sqrt(np.mean(np.square(a))))
        return v if v == v else None          # NaN-safe
    except Exception:
        return None


def measure(capture, pc_peak):
    """(room RMS or None, PC playback peak or None). ``capture(seconds)``
    returns mono float audio or None; ``pc_peak()`` returns 0..1 or None.
    Never raises."""
    rms = None
    try:
        buf = capture(SAMPLE_S)
        if buf is not None:
            rms = _rms(buf)
    except Exception:
        rms = None
    try:
        peak = pc_peak()
        peak = None if peak is None else float(peak)
    except Exception:
        peak = None
    return rms, peak


def describe(rms, pc_peak) -> str:
    """The spoken answer, built only from the two readings."""
    if rms is None and pc_peak is None:
        return "I couldn't take a reading of the room just now, sir."
    if rms is None:
        first = "I couldn't read the microphone just now, sir"
    else:
        db = round(20 * math.log10(max(rms, 1e-6)))
        if rms < QUIET_RMS:
            first = f"It's quiet in here, sir, about {db} dB on the mic over the last second"
        elif rms < SPEECH_RMS:
            first = (f"There's a little background noise, sir, about {db} dB, "
                     f"but it's under the level I treat as speech")
        else:
            first = (f"It's noisy, sir, about {db} dB, loud enough to cross my "
                     f"speech threshold, so it can cut into what I hear")
    if pc_peak is None:
        return first + "."
    if pc_peak >= PC_PLAYING_PEAK:
        return first + ", and your PC is playing sound right now."
    return first + ", and nothing's playing on the PC."


def room_noise(capture, pc_peak) -> str:
    """The action: measure, then describe."""
    return describe(*measure(capture, pc_peak))
