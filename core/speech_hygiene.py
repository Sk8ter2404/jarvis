"""Spoken-text hygiene (v3 W01): pure helpers, no I/O, never raise.

Three small things the owner can hear:

* ``strip_stray_tags`` - a mid-reply ``[intent:x]`` / ``[mood:x]`` / ``[wry]``
  the model wrote after its first sentence. ``_speak`` only parses LEADING
  tags, so the rest went to the synthesiser as words ("intent briefing").
* ``mentions_device`` / ``reply_has_weather_cue`` - the weather-from-memory
  injector read "PETG prints at 235 degrees Celsius" and "nozzle 350 C" as
  the weather and ran weather_briefing on a printer question.
* ``INTENT_ALIASES`` - tag names the model likes that name no preset
  (``concern``, ``observation``); mapped onto an existing preset name.

All of it sits behind SPEECH_HYGIENE in the monolith; this module has no
switch of its own.
"""
from __future__ import annotations

import re

# Only these three forms. [ACTION: ...] and any other bracket text are left
# alone (the executor and the claim layers own those).
_STRAY_TAG_RE = re.compile(
    r"[ \t]*\[\s*(?:intent\s*:\s*[A-Za-z_][A-Za-z0-9_]*"
    r"|mood\s*:\s*[A-Za-z_][A-Za-z0-9_]*"
    r"|wry)\s*\][ \t]*",
    re.IGNORECASE)


def strip_stray_tags(text: str) -> str:
    """``text`` with every ``[intent:x]`` / ``[mood:x]`` / ``[wry]`` removed.
    One space is kept between the words either side; none before closing
    punctuation or at the ends. Text without such a tag is returned as is."""
    try:
        s = str(text or "")
        if "[" not in s or not _STRAY_TAG_RE.search(s):
            return text
        out: list[str] = []
        pos = 0
        for m in _STRAY_TAG_RE.finditer(s):
            before = s[pos:m.start()]
            out.append(before)
            after = s[m.end():]
            joined = "".join(out)
            if (not joined.strip() or not after.strip()
                    or after[0] in ".,;:!?)" or joined[-1:] in "\n("):
                out.append("")
            else:
                out.append(" ")
            pos = m.end()
        out.append(s[pos:])
        return "".join(out)
    except Exception:
        return text


# intent tag name -> a key that already exists in core.tts._INTENT_PRESETS.
INTENT_ALIASES = {
    "concern": "concerned",
    "observation": "briefing",
}


# A device or material the temperature belongs to. Whole words only, so
# "grilling" / "printing"-style weather chatter is judged on its own cues.
_DEVICE_RE = re.compile(
    r"\b(?:nozzle|hot\s?end|hotend|heated\s+bed|(?:print\s+|build\s+)?bed|"
    r"filament|pla|petg|abs|tpu|asa|print(?:s|ed|ing|er|ers)?|h2d|"
    r"oven|grill|fridge|refrigerator|freezer|cpu|gpu)\b",
    re.IGNORECASE)


def mentions_device(text: str) -> bool:
    """True when ``text`` names a printer part, a filament, or an appliance /
    chip whose temperature is not the weather."""
    try:
        return bool(_DEVICE_RE.search(str(text or "")))
    except Exception:
        return False


# A cue that a bare "64 degrees" is the weather.
_WEATHER_CUE_RE = re.compile(
    r"\b(?:outside|out\s+there|today|tonight|tomorrow|right\s+now|currently|"
    r"weather|forecast|sunny|cloudy|clear|overcast|rain(?:y|ing)?|"
    r"snow(?:y|ing)?|windy|foggy|humid|chilly|breezy|freezing|warm|cold|hot|"
    r"mild|feels\s+like|high\s+of|low\s+of)\b",
    re.IGNORECASE)


def reply_has_weather_cue(text: str) -> bool:
    try:
        return bool(_WEATHER_CUE_RE.search(str(text or "")))
    except Exception:
        return False


def bare_unit_needs_cue(reply: str, user_text: str,
                        may_be_weather_request: bool) -> bool:
    """True when a bare "<n> degrees <unit>" must NOT count as the weather:
    the owner's words are known, they were not a weather / temperature ask,
    and the reply has no weather cue. With the owner's words unknown (empty)
    the old behaviour stands."""
    try:
        if not str(user_text or "").strip():
            return False
        if may_be_weather_request:
            return False
        return not reply_has_weather_cue(reply)
    except Exception:
        return False
