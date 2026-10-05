"""core/media_gate.py — PC audio playing + not the owner's voice = not a command.

THE LIVE FAILURE (2026-10-01 22:28:50): an Instagram reel playing on this PC said
"Jarvis, find me a restaurant in Brickell ... and go and build a website for them
now", the desk mic heard it, the leading "Jarvis" passed the wake-word gate, and
JARVIS ran a web search and a see_screen chain on it. The learn gate scored that
capture's voice at 0.43 against the owner's voiceprint; his own wake-word turns
that evening scored 0.63-0.68.

So while another app on the PC is producing sound, a mic turn must not be
confidently someone else's voice:

  * "PC audio playing" = the highest peak meter (IAudioMeterInformation) of the
    default render device's ACTIVE audio sessions, JARVIS's own process and the
    system-sounds session excluded, at or above a threshold. When the meter
    cannot be read (no pycaw / COM, not Windows) the caller may fall back to the
    OS media session (SMTC) "playing" flag.
  * the voice verdict is core/learn_gate's (owner / not_owner / unsure /
    unavailable). Only NOT_OWNER drops: an UNSURE score is let through because
    the owner's own commands over music land there, and a voice-ID that is
    unavailable (nobody enrolled, no resemblyzer, no audio) keeps today's
    behaviour.

REVIEW REPAIR (2026-10-02). The floor rested on "his wake-word turns score
0.63-0.68", but his own commands OVER media scored lower on the same buffer the
gate checks: 21:43:17 "Jarvis plays Skrillex Essentials on YouTube." 0.52,
22:59:47 "Jarvis, what time is it? ..." 0.50 (a 20.7 s capture, mostly the
video), 13:57:08 0.48 - each dropped in silence while audio played. So:

  * the floor is 0.45 (the reel scored 0.43; core/config.py);
  * a short media-control command ("pause", "next song", "turn it down")
    always passes: over his media that is mostly what he says
    (is_media_control);
  * a long capture is scored again on its LEADING speech, where "Jarvis, ..."
    is, before it is called someone else's (leading_speech_window);
  * a dropped "Jarvis, ..." gets one short spoken cue (DROP_CUE) instead of
    silence (bobert_companion._media_gate_drop_cue).

Pure decisions plus one pycaw reader with injectable seams; never raises.
"""
from __future__ import annotations

import re
import time

from core import learn_gate as _lg
from core import wake_prefix as _wake_prefix

# AudioSessionState: 0 inactive, 1 active, 2 expired.
_SESSION_ACTIVE = 1

DROP_LINE = "[media-gate] PC audio playing and not the owner's voice"

# Spoken when a dropped turn was addressed to JARVIS: a statement, never a
# question (in wake-word mode he could not answer it without the wake word).
DROP_CUE = "Sorry, sir, I couldn't tell that was you over the audio."

# ── what he says over his own media ───────────────────────────────────────
# (The wake word is taken off by core/wake_prefix.strip_wake_lead, the one
# wake-word rule.)
_MEDIA_CONTROL_CORE = (
    r"(?:pause|unpause|resume|mute|unmute)"
    r"(?:\s+(?:it|this|that|the\s+(?:music|video|song|playback|sound|audio|tv)))?"
    r"|(?:skip|next|previous|prev)"
    r"(?:\s+(?:it|this|that|one|song|track|video|episode|chapter|"
    r"this\s+(?:one|song|track|video)|the\s+(?:song|track|video)))?"
    r"|(?:go\s+)?back\s+(?:a|one)\s+(?:song|track)"
    r"|louder|quieter|softer"
    r"|(?:volume|sound)\s+(?:up|down)"
    r"|turn\s+(?:it|this|that|the\s+(?:music|volume|video|sound|tv))"
    r"\s+(?:up|down|off)"
    r"|turn\s+(?:up|down)\s+(?:the\s+)?(?:music|volume|video|sound|tv)"
    r"|(?:stop|kill)\s+(?:the\s+)?(?:music|video|song|playback|sound|audio)"
    r"|(?:lower|raise)\s+(?:the\s+)?(?:volume|music|sound)")
_MEDIA_CONTROL_TAIL = (
    r"(?:\s+(?:a\s+(?:bit|little|touch|notch)|please|for\s+me|now|sir|"
    r"right\s+now|again|thanks|thank\s+you|a\s+little\s+bit))*")
_MEDIA_CONTROL_RE = re.compile(
    r"^(?:please\s+)?(?:" + _MEDIA_CONTROL_CORE + r")" + _MEDIA_CONTROL_TAIL
    + r"[\s.!?]*$", re.IGNORECASE)


def is_media_control(text) -> bool:
    """True when ``text`` (wake word allowed in front) is nothing but a
    media control: pause / resume / mute / skip / next / previous / volume
    up or down / louder / quieter / turn it down / stop the music, with a
    "please" or "a bit" at most. Such a turn passes the gate whatever the
    voice: a reel saying "Jarvis, pause" costs nothing. Never raises."""
    try:
        s = _wake_prefix.strip_wake_lead(str(text or "")).replace(",", " ")
        s = " ".join(s.split())
        return bool(s) and bool(_MEDIA_CONTROL_RE.match(s))
    except Exception:
        return False


def leading_speech_window(audio, sample_rate, *, seconds: float = 3.0,
                          min_extra_s: float = 1.0, frame_s: float = 0.02,
                          pre_s: float = 0.15):
    """The first ``seconds`` of speech in a capture, from its onset (the first
    20 ms frame at a quarter of the capture's loud level, less ``pre_s``), or
    None when the capture is not at least ``min_extra_s`` longer than that
    (scoring it again would score the same audio), silent, or unreadable.
    "Jarvis, ..." is at the start of a capture; on a long one the media behind
    him dominates the rest (live 22:59:47: a 20.7 s capture scored 0.50).
    Never raises."""
    try:
        import numpy as np
        sr = int(sample_rate or 0)
        if audio is None or sr <= 0:
            return None
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        n_win = int(float(seconds) * sr)
        if n_win <= 0 or a.size < n_win + int(float(min_extra_s) * sr):
            return None
        hop = max(1, int(float(frame_s) * sr))
        n_frames = a.size // hop
        if n_frames < 2:
            return None
        frames = a[:n_frames * hop].reshape(n_frames, hop)
        rms = np.sqrt((frames * frames).mean(axis=1))
        loud = float(np.percentile(rms, 95))
        if not loud > 1e-4:
            return None
        above = np.nonzero(rms >= 0.25 * loud)[0]
        if above.size == 0:
            return None
        start = max(0, int(above[0]) * hop - int(float(pre_s) * sr))
        start = min(start, a.size - n_win)
        return a[start:start + n_win]
    except Exception:
        return None


def _pycaw_sessions():
    """[(pid, meter)] for the default render device's active sessions, plus a
    cleanup callable. COM is initialised on THIS thread and released by the
    cleanup; the meters must be used before it runs."""
    import comtypes
    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
    inited = False
    try:
        comtypes.CoInitialize()
        inited = True
    except Exception:
        pass
    out = []
    try:
        for s in AudioUtilities.GetAllSessions():
            try:
                if s.State != _SESSION_ACTIVE:
                    continue
                pid = int(s.ProcessId or 0)
                out.append((pid, s._ctl.QueryInterface(IAudioMeterInformation)))
            except Exception:
                continue
    except Exception:
        out = []
        if inited:
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass
        raise

    def _cleanup():
        out.clear()
        if inited:
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass
    return out, _cleanup


def pc_audio_peak(exclude_pids=(), *, samples: int = 4, interval_s: float = 0.025,
                  stop_at: float = 1.0, sessions_fn=None, sleep=time.sleep):
    """Highest peak (0..1) of the active render sessions of OTHER processes, or
    None when it cannot be read. A peak meter reports one device period, so it
    is sampled `samples` times `interval_s` apart (speech has gaps), stopping
    early once `stop_at` is reached. No active session = 0.0 without sampling.
    pid 0 (system sounds) is never counted. Never raises.

    COM ORDER (v2.0.180 review): every meter reference held HERE is dropped
    BEFORE the cleanup runs CoUninitialize - a Release() after this thread's
    last CoUninitialize calls into an apartment that is gone. v2.0.179's music
    gate made this a read every 2 s from the ambient worker as well."""
    cleanup = None
    sessions = meters = m = None
    try:
        sessions, cleanup = (sessions_fn or _pycaw_sessions)()
        skip = {0}
        for p in exclude_pids or ():
            try:
                skip.add(int(p))
            except Exception:
                pass
        meters = [m for pid, m in sessions if pid not in skip]
        if not meters:
            return 0.0
        best = 0.0
        n = max(1, int(samples))
        for i in range(n):
            for m in meters:
                try:
                    v = float(m.GetPeakValue())
                except Exception:
                    continue
                if v == v and v > best:          # NaN-safe
                    best = v
            if best >= stop_at or i == n - 1:
                break
            sleep(max(0.0, float(interval_s)))
        return max(0.0, min(1.0, best))
    except Exception:
        return None
    finally:
        # Our references go first; the cleanup clears the session list and
        # only then uninitialises COM (see COM ORDER above).
        sessions = meters = m = None
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                pass


def audio_playing(peak, smtc_playing=False, *, threshold: float = 0.01) -> bool:
    """True when the PC is producing sound: the meter at/over `threshold`, or -
    only when the meter could not be read (None) - the media session says
    playing. A readable meter at 0 wins over SMTC (paused / muted media)."""
    try:
        if peak is None:
            return bool(smtc_playing)
        return float(peak) >= float(threshold)
    except Exception:
        return False


def decide(playing: bool, voice: str) -> "tuple[bool, str]":
    """(drop, log line) for one mic turn. Drops only a NOT_OWNER voice while the
    PC plays audio; every other case runs, with a line saying why when audio was
    playing ("" when it was not)."""
    if not playing:
        return False, ""
    if voice == _lg.NOT_OWNER:
        return True, DROP_LINE
    if voice == _lg.OWNER:
        return False, "[media-gate] PC audio playing; the owner's voice — allowed"
    if voice == _lg.UNSURE:
        return False, ("[media-gate] PC audio playing; voice too close to call "
                       "— allowed")
    return False, ("[media-gate] PC audio playing; voice-ID unavailable — "
                   "allowed (today's behaviour)")
