"""Answer "what microphone are you using?" out loud.

WHY THIS SKILL EXISTS (owner-reported, 2026-09-04): he asked JARVIS "what
microphone are you using right now" and got `[ACTION: system_pulse]` — a CPU /
memory / GPU read-out that never addressed the question. That was not a model
mistake. There was simply NO action bound to the current input device, so the
model picked the nearest plausible one. `bobert_companion.get_current_mic_name()`
has existed all along, but only ever fed the console banner and a prompt field;
bobert_companion.py says so outright next to its force-refresh path: "force=True
caller is get_current_mic_name(), which no voice action ...".

The general lesson, worth keeping: when a capability is missing, the local brain
does NOT say "I can't" — it emits a wrong-but-plausible action. A silent routing
failure like that reads to the owner as the model being stupid, when the real
defect is a gap in the action registry.

Deliberately a skill and not a monolith edit: skills/ is the extension point,
and this needs nothing from the monolith but two public helpers.

Both replies are finished, user-facing sentences with no self-speak, so the
names are declared in SPEAK_VERBATIM_ACTIONS below — otherwise the answer is
computed, logged and dropped, which is the exact defect class this skill fixes.

THE MIC ANSWER NAMES THE LIVE DEVICE, NOT THE SELECTED ONE (2026-09-29, live).
"what microphone are you using" answered with a configured mic that Windows
reported NotPresent at that moment, while the capture was running on another
device. get_current_mic_name() reports what the NEXT open would ask for, out of
PortAudio's frozen device list — which still carries an unplugged mic until a
re-enumeration, and that is deferred while any stream is live. So the answer
now comes from
  * get_live_capture_device() — what record_speech's stream really opened on;
  * get_capture_endpoints()   — Windows' own state for every recording
                                endpoint, so a NotPresent device is never
                                claimed;
  * the configured preference (MICROPHONE_INDEX / PREFERRED_INPUT_DEVICES) —
    "your preferred microphone isn't connected" when Windows has it but not
    as an active endpoint;
  * get_default_capture_name() — Windows' live default, named when the stream
                                 cannot be vouched for.
Each one is optional: a monolith without them degrades to the selected device,
phrased "set to listen on" (which is what it is), and an unreadable device is
reported as unknown, never guessed.
"""
from __future__ import annotations

import importlib
import re
import sys
from typing import Optional

# Names whose return value is already a finished sentence and must be spoken
# verbatim. `load_skills` reads this and merges it into the monolith's
# SPEAK_RESULT_VERBATIM_ACTIONS (see _collect_skill_speak_sets).
SPEAK_VERBATIM_ACTIONS = (
    "current_mic", "what_microphone", "which_microphone", "what_mic",
    "current_speaker", "what_speakers", "which_speakers",
    "audio_devices", "what_audio_devices",
)

_UNKNOWN = "unknown"


def _bc():
    """The running monolith, or None. Never imports it fresh — importing
    bobert_companion has side effects (it starts device pumps), so we only ever
    take the already-loaded module."""
    mod = sys.modules.get("bobert_companion")
    if mod is not None:
        return mod
    try:                                    # pragma: no cover - standalone use
        return importlib.import_module("bobert_companion")
    except Exception:
        return None


def _friendly(raw: Optional[str]) -> Optional[str]:
    """Speakable form of a device name ('Microphone (Blue Snowball )' -> 'the
    Blue Snowball'), falling back to the raw name. Never raises."""
    # A whitespace-only name is as useless as an empty one — speaking
    # "I'm listening on    , sir" is worse than admitting we don't know.
    if not raw or not raw.strip() or raw.strip().lower() == _UNKNOWN:
        return None
    bc = _bc()
    fn = getattr(bc, "_friendly_device_name", None) if bc else None
    if callable(fn):
        try:
            nice = fn(raw)
            # The helper can hand back blank or "unknown" too; re-apply the
            # same rule to its output rather than trusting it.
            if nice and nice.strip() and nice.strip().lower() != _UNKNOWN:
                return nice.strip()
        except Exception:
            pass
    return raw.strip()


def _read(which: str) -> Optional[str]:
    """Current device name for 'mic' or 'speaker', or None if unavailable."""
    bc = _bc()
    if bc is None:
        return None
    getter = getattr(
        bc, "get_current_mic_name" if which == "mic" else "get_current_speaker_name",
        None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


_MIC_UNKNOWN = "I couldn't determine which microphone is active, sir."

# PortAudio's virtual "follow the Windows default" inputs. A stream on one of
# these reads whatever Windows' default recording device is.
_MAPPER_HINTS = ("sound mapper", "primary sound capture")


def _call(bc, name: str):
    """bc.<name>() or None — absent, not callable, or raising all read as None."""
    fn = getattr(bc, name, None) if bc is not None else None
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:
        return None


def _usable_name(raw) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip() \
            or raw.strip().lower() == _UNKNOWN:
        return None
    return raw.strip()


def _norm(name: str) -> str:
    """Case/space-insensitive form with the spacing around parentheses dropped
    (PortAudio's host APIs disagree about 'Microphone (X )' vs '(X)')."""
    s = " ".join((name or "").lower().split())
    return re.sub(r"\s*([()])\s*", r"\1", s)


def _same_endpoint(pa_name: str, ep_name: str) -> bool:
    """Is PortAudio device `pa_name` the Windows endpoint `ep_name`? Equal after
    normalising, or — MME cuts descriptions at 31 characters — a 31-character
    PortAudio name that the endpoint's name starts with."""
    a, b = _norm(pa_name), _norm(ep_name)
    if not a or not b:
        return False
    return a == b or (len(pa_name) >= 31 and b.startswith(a))


def _states(name: str, endpoints, fragment: bool = False) -> list:
    """Windows states (lower-case) of every recording endpoint that IS `name` —
    or, for a PREFERRED_INPUT_DEVICES fragment, whose name contains it."""
    out = []
    for row in endpoints or ():
        try:
            _id, ep_name, state = row
        except (TypeError, ValueError):
            continue
        ep_name = ep_name if isinstance(ep_name, str) else ""
        hit = ((name.lower() in ep_name.lower()) if fragment
               else _same_endpoint(name, ep_name))
        if hit:
            out.append(str(state or "").strip().lower())
    return out


def _present(name: str, endpoints, fragment: bool = False) -> Optional[bool]:
    """True: an ACTIVE Windows endpoint is this device. False: Windows lists it
    only as NotPresent / Unplugged / Disabled, or not at all. None: Windows
    could not be asked — unknown, which is never reported as absent."""
    if endpoints is None or not name:
        return None
    return "active" in _states(name, endpoints, fragment)


def _absence(name: str, endpoints, fragment: bool = False) -> str:
    states = _states(name, endpoints, fragment)
    if states and all(s == "disabled" for s in states):
        return "is disabled"
    return "isn't connected"


def _preferred_mic(bc):
    """(name, is_fragment) of the configured microphone preference, or None
    when JARVIS simply follows the Windows default (the normal setup)."""
    if bc is None:
        return None
    idx = getattr(bc, "MICROPHONE_INDEX", None)
    if isinstance(idx, int) and not isinstance(idx, bool):
        if idx < 0:
            return None
        try:
            info = bc.sd.query_devices(idx)
            name = _usable_name(info.get("name") if isinstance(info, dict)
                                else None)
        except Exception:
            name = None
        return (name, False) if name else None
    prefs = getattr(bc, "PREFERRED_INPUT_DEVICES", None)
    if isinstance(prefs, (list, tuple)):
        for p in prefs:
            if isinstance(p, str) and p.strip():
                return (p.strip(), True)
    return None


def _is_pref(pref, name: str) -> bool:
    if not pref:
        return False
    return (pref[0].lower() in name.lower()) if pref[1] \
        else _same_endpoint(pref[0], name)


def _default_tail(default: Optional[str], not_this: str) -> str:
    if not default or _same_endpoint(not_this, default):
        return ""
    return f" Windows' default microphone is {_friendly(default) or default}."


def _describe_mic():
    """(phrase, sentence) answering "what microphone are you using".

    `phrase` ('listening on X' / 'set to listen on X') is returned only for the
    plain answer, so audio_devices can fold it into one sentence with the
    speakers; every other outcome is a finished sentence of its own. No
    sentence here that is an honest answer carries a failure marker (couldn't
    / can't / didn't …), so it is spoken verbatim; only "unknown" does."""
    bc = _bc()
    if _call(bc, "_mic_input_disabled") is True:
        return None, "My microphone input is switched off, sir."
    endpoints = _call(bc, "get_capture_endpoints")
    if not isinstance(endpoints, (list, tuple)) or not endpoints:
        endpoints = None
    default = _usable_name(_call(bc, "get_default_capture_name"))

    pref = _preferred_mic(bc)
    pref_missing = bool(pref) and _present(pref[0], endpoints, pref[1]) is False
    pref_note = ""
    if pref_missing:
        pref_note = (f"Your preferred microphone, "
                     f"{_friendly(pref[0]) or pref[0]}, "
                     f"{_absence(pref[0], endpoints, pref[1])}, sir")

    live = _call(bc, "get_live_capture_device")
    if isinstance(live, dict):
        name = _usable_name(live.get("name"))
        if name is None:
            return None, _MIC_UNKNOWN
        if any(h in name.lower() for h in _MAPPER_HINTS):
            # The stream follows the Windows default, so THAT is the device.
            if not default:
                return ("listening on the system default microphone",
                        "I'm listening on the system default microphone, sir.")
            name = default
        shown = _friendly(name) or name
        if _present(name, endpoints) is False:
            # Opened on a device Windows no longer reports as connected. Never
            # claim it: say what is known and what is not.
            if pref_missing and _is_pref(pref, name):
                head = f"{pref_note}, yet my capture stream is still opened on it"
            else:
                head = (f"{pref_note}. " if pref_note else "") + (
                    f"My capture stream is still opened on {shown}"
                    f"{'' if pref_note else ', sir'}, but Windows reports it "
                    f"{_absence(name, endpoints)}")
            return None, (f"{head}, so I'm not certain which microphone is "
                          f"actually hearing you.{_default_tail(default, name)}")
        if pref_missing:
            return None, f"{pref_note} — I'm listening on {shown} instead."
        return f"listening on {shown}", f"I'm listening on {shown}, sir."

    # No capture stream has opened yet (or an older monolith that does not
    # publish one): report the SELECTED device — what the next open will ask
    # for — and word it as that, not as what is hearing him.
    raw = _usable_name(_read("mic"))
    if raw is None:
        return None, _MIC_UNKNOWN
    name = re.sub(r"^\[\d+\]\s*", "", raw)
    if name.lower() == "(system default)" or any(
            h in name.lower() for h in _MAPPER_HINTS):
        if default:
            shown = _friendly(default) or default
            return (f"set to listen on Windows' default microphone, {shown}",
                    f"I'm set to listen on Windows' default microphone, "
                    f"{shown}, sir.")
        return ("set to listen on the system default microphone",
                "I'm set to listen on the system default microphone, sir.")
    shown = _friendly(raw) or name
    if _present(name, endpoints) is False:
        return None, (f"I'm set to listen on {shown}, sir, but Windows reports "
                      f"it {_absence(name, endpoints)}."
                      f"{_default_tail(default, name)}")
    if pref_missing:
        return None, f"{pref_note} — I'm set to listen on {shown} instead."
    return f"set to listen on {shown}", f"I'm set to listen on {shown}, sir."


def current_mic(_arg: str = "") -> str:
    return _describe_mic()[1]


def current_speaker(_arg: str = "") -> str:
    name = _friendly(_read("speaker"))
    if not name:
        return "I couldn't determine which speakers are active, sir."
    return f"I'm speaking through {name}, sir."


def audio_devices(_arg: str = "") -> str:
    """Both at once — what a bare 'what audio devices are you using' deserves.
    The microphone half IS current_mic's answer (one source of truth); only the
    plain case is folded into a single sentence with the speakers."""
    mic_phrase, mic_sentence = _describe_mic()
    mic_known = mic_sentence != _MIC_UNKNOWN
    spk = _friendly(_read("speaker"))
    if not mic_known and not spk:
        return "I couldn't determine my audio devices, sir."
    if mic_phrase and spk:
        return f"I'm {mic_phrase} and speaking through {spk}, sir."
    if mic_known and spk:
        return f"{mic_sentence} I'm speaking through {spk}, sir."
    if mic_known:
        return f"{mic_sentence} I couldn't determine my output device, though."
    return (f"I'm speaking through {spk}, sir — though I couldn't determine "
            f"my microphone.")


def register(actions):
    actions["current_mic"] = current_mic
    actions["what_microphone"] = current_mic
    actions["which_microphone"] = current_mic
    actions["what_mic"] = current_mic
    actions["current_speaker"] = current_speaker
    actions["what_speakers"] = current_speaker
    actions["which_speakers"] = current_speaker
    actions["audio_devices"] = audio_devices
    actions["what_audio_devices"] = audio_devices
    print("  [audio_devices] ready — actions: current_mic, what_microphone, "
          "current_speaker, what_speakers, audio_devices.")
