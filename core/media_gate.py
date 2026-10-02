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

Pure decisions plus one pycaw reader with injectable seams; never raises.
"""
from __future__ import annotations

import time

from core import learn_gate as _lg

# AudioSessionState: 0 inactive, 1 active, 2 expired.
_SESSION_ACTIVE = 1

DROP_LINE = "[media-gate] PC audio playing and not the owner's voice"


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
    pid 0 (system sounds) is never counted. Never raises."""
    cleanup = None
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
