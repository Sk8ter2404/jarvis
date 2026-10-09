"""core/audio_check.py — "why can't I hear my video?" answered from a READING.

WHY THIS MODULE EXISTS
======================
Live 2026-10-09: the owner asked "Jarvis, why can't I hear my YouTube video?"
and JARVIS said "I'm afraid the volume is currently set to 20%, sir." while
running system_pulse (CPU / GPU / memory / windows). Nothing read the volume;
the number was invented and voiced. No action could read the sound state at
all: set_volume / volume_mute only WRITE it.

``audio_check`` is the read-only check. It reads, and says only what it read:
  * the Windows default output device's master volume and mute state;
  * the device's name (where the sound is going);
  * every app's own audio session: its mixer volume, its mute and whether it
    is sending sound right now (the session's peak meter).
For the app the owner named ("YouTube" -> the browsers, "Spotify" ->
spotify.exe) it reports that app's session first. Nothing is changed: no
unmute, no volume step - the owner decides what to fix.

Anything that could not be read is said to be unknown, never guessed. With
nothing readable at all the answer says so ("I couldn't read the audio
settings just now, sir").

Layout: ``read_state`` does the pycaw / COM I/O (never raises; a field it
could not read stays None). ``describe`` is pure (tests/test_audio_check.py).
``audio_check(arg)`` is the action: ``describe(read_state(), arg)``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

__all__ = ["AppSession", "AudioState", "audio_check", "describe",
           "is_audio_trouble_question", "read_state", "target_apps"]


@dataclass
class AppSession:
    name: str                       # process name, e.g. "chrome.exe"
    volume_pct: Optional[int] = None
    muted: Optional[bool] = None
    sounding: Optional[bool] = None  # peak meter > 0 right now


@dataclass
class AudioState:
    master_pct: Optional[int] = None
    muted: Optional[bool] = None
    device: Optional[str] = None
    sessions: Optional[list] = None   # [AppSession]; None = unreadable
    ducked: bool = False              # JARVIS is lowering other apps now


# ── what the owner is asking about ──────────────────────────────────────────
_BROWSERS = ("chrome", "msedge", "firefox", "brave", "opera", "vivaldi",
             "arc", "iexplore")
# spoken name -> (friendly name, process-name fragments)
_APPS = (
    (r"you\s*tube|netflix|twitch|hulu|disney|prime\s+video|browser|chrome|"
     r"edge|firefox|website|web\s*page|tab", "the browser", _BROWSERS),
    (r"spotify", "Spotify", ("spotify",)),
    (r"apple\s+music|itunes", "Apple Music", ("applemusic", "itunes")),
    (r"discord", "Discord", ("discord",)),
    (r"vlc", "VLC", ("vlc",)),
    (r"teams", "Teams", ("teams",)),
    (r"zoom", "Zoom", ("zoom",)),
    (r"fortnite|game", "the game", ("fortnite",)),
)
_APP_RES = tuple((re.compile(r"\b(?:" + p + r")\b", re.I), label, frags)
                 for p, label, frags in _APPS)


def target_apps(text: str):
    """(label, process fragments) for the app the owner named, or None."""
    for rx, label, frags in _APP_RES:
        if rx.search(text or ""):
            return label, frags
    return None


_HEAR = r"(?:hear|listen\s+to)"
_TROUBLE_RES = tuple(re.compile(p, re.I) for p in (
    # "why can't I hear my YouTube video" / "I can't hear anything"
    r"\b(?:can'?t|cannot|can\s+not|couldn'?t|not\s+able\s+to|unable\s+to)\s+"
    r"(?:i\s+|we\s+|anyone\s+)?(?:still\s+)?" + _HEAR + r"\b",
    # "why is there no sound" / "there's no audio from Chrome"
    r"\bno\s+(?:sound|audio|volume)\b",
    # "is my sound muted" / "is the PC muted" / "why is it muted"
    r"\b(?:is|are|why\s+is|why\s+are)\s+(?:the\s+|my\s+|it\s+|it\b|this\s+|"
    r"that\s+|everything\s+|(?:the\s+|my\s+)?(?:pc|computer|sound|audio|"
    r"speakers?|volume|video|tab|browser)\s+)*(?:still\s+)?muted\b",
    # "why is the sound not working" / "the audio isn't working"
    r"\b(?:sound|audio|speakers?)\s+(?:is\s+not|isn'?t|are\s+not|aren'?t|"
    r"not|stopped)\s+(?:working|playing|coming)\b",
    # "what's the volume at" / "how loud is the volume"
    r"\bwhat'?s?\s+(?:is\s+)?(?:the\s+|my\s+)?(?:system\s+|master\s+)?"
    r"volume\s+(?:at|set\s+to|level|now)\b",
))
_COMMAND_RE = re.compile(
    r"^\s*(?:(?:hey\s+)?jarvis[,\s]+)?(?:please\s+)?(?:turn|set|mute|unmute|"
    r"raise|lower|increase|decrease|put|make)\b", re.I)


def is_audio_trouble_question(text) -> bool:
    """True when the owner asks why he can't hear something / whether the
    sound is muted / what the volume is - a question the audio READING
    answers. A command ("mute it", "turn the volume up") is not one."""
    t = str(text or "").strip()
    if not t or len(t) > 160 or _COMMAND_RE.search(t):
        return False
    return any(rx.search(t) for rx in _TROUBLE_RES)


# ── the reading ─────────────────────────────────────────────────────────────
_SESSION_ACTIVE = 1


def read_state() -> AudioState:
    """Read the default output device and the app sessions (pycaw). Never
    raises; anything unreadable stays None."""
    st = AudioState()
    try:
        import comtypes
        comtypes.CoInitialize()
        inited = True
    except Exception:
        inited = False
    try:
        try:
            from pycaw.pycaw import AudioUtilities
        except Exception:
            return st
        try:
            dev = AudioUtilities.GetSpeakers()
            try:
                st.device = (getattr(dev, "FriendlyName", None) or None)
            except Exception:
                pass
            vol = getattr(dev, "EndpointVolume", None)
            if vol is None:
                from ctypes import POINTER, cast

                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import IAudioEndpointVolume
                iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL,
                                     None)
                vol = cast(iface, POINTER(IAudioEndpointVolume))
            try:
                st.master_pct = int(round(
                    float(vol.GetMasterVolumeLevelScalar()) * 100))
            except Exception:
                pass
            try:
                st.muted = bool(vol.GetMute())
            except Exception:
                pass
        except Exception:
            pass
        try:
            from pycaw.pycaw import IAudioMeterInformation
            out = []
            for s in AudioUtilities.GetAllSessions():
                try:
                    proc = s.Process
                    if proc is None:
                        continue
                    sess = AppSession(name=str(proc.name() or "").lower())
                    sv = s.SimpleAudioVolume
                    if sv is not None:
                        try:
                            sess.volume_pct = int(round(
                                float(sv.GetMasterVolume()) * 100))
                        except Exception:
                            pass
                        try:
                            sess.muted = bool(sv.GetMute())
                        except Exception:
                            pass
                    try:
                        if s.State == _SESSION_ACTIVE:
                            meter = s._ctl.QueryInterface(
                                IAudioMeterInformation)
                            sess.sounding = float(meter.GetPeakValue()) > 0.0
                            meter = None
                        else:
                            sess.sounding = False
                    except Exception:
                        pass
                    out.append(sess)
                except Exception:
                    continue
            st.sessions = out
        except Exception:
            st.sessions = None
        return st
    finally:
        if inited:
            try:
                import comtypes
                comtypes.CoUninitialize()
            except Exception:
                pass


# ── the answer ──────────────────────────────────────────────────────────────
def _app_word(name: str) -> str:
    base = re.sub(r"\.exe$", "", name or "", flags=re.I)
    return {"msedge": "Edge", "chrome": "Chrome", "firefox": "Firefox",
            "brave": "Brave", "opera": "Opera", "vivaldi": "Vivaldi",
            "spotify": "Spotify", "discord": "Discord",
            "applemusic": "Apple Music"}.get(base.lower(), base or "an app")


def describe(state: AudioState, user_text: str = "") -> str:
    """The spoken answer, built ONLY from ``state``. Pure; never raises."""
    try:
        return _describe(state, user_text)
    except Exception:
        return "I couldn't read the audio settings just now, sir."


def _describe(st: AudioState, user_text: str) -> str:
    if st.master_pct is None and st.muted is None and st.sessions is None:
        return ("I couldn't read the audio settings just now, sir, so I "
                "can't tell you why from here.")
    target = target_apps(user_text)
    found = []
    if target and st.sessions:
        found = [s for s in st.sessions
                 if any(f in s.name for f in target[1])]
    causes: list[str] = []
    if st.muted:
        causes.append("Windows sound output is muted")
    elif st.master_pct == 0:
        causes.append("the system volume is at 0 percent")
    for s in found:
        if s.muted:
            causes.append(f"{_app_word(s.name)} is muted in the volume mixer")
        elif s.volume_pct == 0:
            causes.append(f"{_app_word(s.name)}'s own mixer volume is at "
                          f"0 percent")
    facts: list[str] = []
    if st.master_pct is not None and not st.muted:
        facts.append(f"the system volume is at {st.master_pct} percent"
                     + (", not muted" if st.muted is False else ""))
    if st.device:
        facts.append(f"sound is going to {st.device}")
    app_line = ""
    if target:
        label = target[0]
        if st.sessions is None:
            app_line = f"I couldn't read {label}'s own audio."
        elif not found:
            app_line = (f"I don't see {label} playing any audio at the "
                        f"moment, so it may be paused or not started.")
        else:
            live = [s for s in found if s.sounding]
            quiet = [s for s in found if s.sounding is False]
            if live and not any(s.muted for s in live):
                s = live[0]
                lvl = (f" at {s.volume_pct} percent in the mixer"
                       if s.volume_pct is not None else "")
                app_line = f"{_app_word(s.name)} is sending sound{lvl}."
            elif quiet and not live and not causes:
                app_line = (f"{_app_word(quiet[0].name)} isn't sending any "
                            f"sound right now, so the video may be paused "
                            f"or silent.")
    if st.ducked:
        app_line = (app_line + " I lower other apps while I'm speaking, so "
                    "their levels are briefly down.").strip()
    parts: list[str] = []
    if causes:
        parts.append(_cap(_join(causes)) + ", sir - that would explain it.")
        if facts:
            parts.append(_cap(_join(facts)) + ".")
    else:
        if facts:
            parts.append(_cap(_join(facts)) + ", sir.")
        else:
            parts.append("I couldn't read the system volume, sir.")
    if app_line:
        parts.append(app_line)
    if not causes and target and found and any(s.sounding for s in found):
        parts.append("If you still hear nothing, check that "
                     + (st.device or "that device")
                     + " is the one you're listening on.")
    return " ".join(p for p in parts if p).replace("  ", " ")


def _join(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def audio_check(arg: str = "") -> str:
    """The read-only action: what the sound settings say right now."""
    return describe(read_state(), str(arg or ""))
