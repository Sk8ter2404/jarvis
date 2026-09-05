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
"""
from __future__ import annotations

import importlib
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


def current_mic(_arg: str = "") -> str:
    name = _friendly(_read("mic"))
    if not name:
        # Honest failure: say we could not determine it, never invent a device.
        return "I couldn't determine which microphone is active, sir."
    return f"I'm listening on {name}, sir."


def current_speaker(_arg: str = "") -> str:
    name = _friendly(_read("speaker"))
    if not name:
        return "I couldn't determine which speakers are active, sir."
    return f"I'm speaking through {name}, sir."


def audio_devices(_arg: str = "") -> str:
    """Both at once — what a bare 'what audio devices are you using' deserves."""
    mic = _friendly(_read("mic"))
    spk = _friendly(_read("speaker"))
    if not mic and not spk:
        return "I couldn't determine my audio devices, sir."
    if mic and spk:
        return f"I'm listening on {mic} and speaking through {spk}, sir."
    if mic:
        return (f"I'm listening on {mic}, sir — though I couldn't determine "
                f"my output device.")
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
