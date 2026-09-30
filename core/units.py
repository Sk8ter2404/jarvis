"""
Unit conversion helpers for JARVIS's SPOKEN/displayed output.

The owner thinks in US imperial for everyday distances — feet and yards — and
only in metric for 3D printing (millimetres). The LLM already gets that policy
via core/prompts.py, but several code paths build distance strings DIRECTLY and
speak them through proactive_announce / voice-action returns, bypassing the LLM.
Those paths call meters_to_imperial_phrase() so a Kinect proximity reading comes
out as "about 8 feet" rather than "about 2.5 metres". 2026-07-08.
"""
from __future__ import annotations

_FEET_PER_METRE = 3.28084


def meters_to_imperial_phrase(metres) -> str:
    """A natural spoken imperial distance for a metric source value.

    Feet for anything under ~10 ft (a person in the same room), yards beyond
    that. Grammatical singular/plural. Returns "" for a missing/garbage value so
    callers can drop the clause cleanly (matching how they already treat a None
    distance). Examples: 0.6 m -> "2 feet", 2.5 m -> "8 feet", 4.0 m -> "4 yards".
    """
    try:
        m = float(metres)
    except (TypeError, ValueError):
        return ""
    if m <= 0:
        return ""
    feet = m * _FEET_PER_METRE
    if feet < 1.5:
        return "about a foot"
    if feet < 10.0:
        n = int(round(feet))
        return f"{n} foot" if n == 1 else f"{n} feet"
    n = int(round(feet / 3.0))
    return f"{n} yard" if n == 1 else f"{n} yards"


# ── unit-conversion turns (2026-09-29) ──────────────────────────────────────
# Live v2.0.131: "convert 100 degrees fahrenheit to celsius" got the right
# answer AND ran weather_briefing. The monolith's preemptive hallucination
# layer reads "37.8 degrees Celsius" in a reply as the weather stated from
# memory and injects the action; the prompt router meanwhile loaded SYSTEM
# HEALTH (hardware temperatures) on "degrees"/"celsius". A conversion is
# arithmetic: neither layer should treat its unit words as a weather or
# hardware question. These two helpers let both layers tell the difference.
import re as _re

_TEMP_UNITS = (r"(?:fahrenheit|celsius|centigrade|kelvin)")
_UNITS = (
    r"(?:(?:degrees?\s+)?" + _TEMP_UNITS + r"|degrees?|°\s*[fck]?|"
    r"feet|foot|ft|inch(?:es)?|yards?|yds?|miles?|mi|"
    r"(?:kilo|centi|milli)?(?:met(?:er|re)s?|grams?|lit(?:er|re)s?)|"
    r"km|cm|mm|kg|mg|ml|lbs?|pounds?|ounces?|oz|stones?|tons?|tonnes?|"
    r"gallons?|gal|quarts?|pints?|cups?|tablespoons?|teaspoons?|"
    r"mph|kph|km/h|knots?|acres?|hectares?)")
_NUM = r"-?\d[\d,]*(?:\.\d+)?"

_CONVERSION_REQUEST_RES = (
    # "convert 100 degrees fahrenheit to celsius", "unit conversion for 5 kg"
    _re.compile(r"\bconver(?:t|ts|ting|sion)\b.*(?:\d|\b" + _UNITS + r"\b)"),
    # "what's 30 celsius in fahrenheit", "5 feet to metres", "10 kg in pounds"
    _re.compile(_NUM + r"\s*" + _UNITS + r"\s+(?:in|to|into|as)\s+"
                r"(?:degrees?\s+)?" + _UNITS + r"(?![a-z])"),
    # "how many feet in a mile", "how many grams are in an ounce"
    # (bare "degrees" is left out: "how many degrees is it outside" is weather)
    _re.compile(r"\bhow\s+many\s+(?!degrees?\b)" + _UNITS +
                r"\s+(?:in|are\s+in|make|per)\b"),
    # "how many degrees celsius is 100 fahrenheit"
    _re.compile(r"\bhow\s+many\s+(?:degrees?\s+)?" + _TEMP_UNITS +
                r"\s+(?:is|are|in|make)\b"),
    # "100 fahrenheit in celsius?" without a leading verb, number first
    _re.compile(_NUM + r".*\b(?:in|to|into)\s+(?:degrees?\s+)?" + _TEMP_UNITS
                + r"\b"),
)


# Weather words in the OWNER's utterance. A number plus "in celsius" is the
# weather when the turn is about the weather: "what's the weather in 90210 in
# celsius" (a zip code) or "is it going to hit 90 today in fahrenheit" must
# keep every weather route and the weather-from-memory injection.
_REQUEST_WEATHER_CUE_RE = _re.compile(
    r"\b(?:weather|forecast|outside|out\s+there|today|tonight|tomorrow|"
    r"this\s+(?:morning|afternoon|evening|week|weekend)|"
    r"(?:today'?s|tomorrow'?s|tonight'?s|the)\s+(?:high|low)s?|"
    r"feels\s+like|zip(?:\s*code)?|rain(?:y|ing)?|snow(?:y|ing)?|windy|"
    r"humid(?:ity)?|going\s+to\s+(?:be|hit|get|reach)|will\s+it|is\s+it)\b")
# ...and, for judging a REPLY that states both scales, any temperature
# question at all ("what's the temperature", "how cold is it").
_REQUEST_TEMP_QUESTION_RE = _re.compile(
    r"\b(?:temperature|temp|how\s+(?:hot|cold|warm|chilly|cool))\b")


def is_unit_conversion_request(text) -> bool:
    """True when the owner's utterance asks to convert a quantity between
    units ("convert 100 degrees fahrenheit to celsius", "what's 5 feet in
    metres", "how many cups in a quart"). A weather or hardware question
    that merely names a unit ("what's the weather in celsius", "how hot is
    the GPU in celsius") has no quantity to convert and returns False, and so
    does any utterance with a weather cue ("what's the weather in 90210 in
    celsius", "will it hit 90 today in fahrenheit")."""
    low = " ".join(str(text or "").lower().split())
    if not low or _REQUEST_WEATHER_CUE_RE.search(low):
        return False
    return any(rx.search(low) for rx in _CONVERSION_REQUEST_RES)


def may_be_weather_request(text) -> bool:
    """True when the owner's utterance could be asking about the weather or a
    temperature reading (a weather cue, "temperature", "how cold ..."). The
    monolith trusts a both-scales REPLY as a conversion only when this is
    False, so "what's the temperature" answered from memory with "68°F /
    20°C" still gets the real weather_briefing."""
    low = " ".join(str(text or "").lower().split())
    return bool(_REQUEST_WEATHER_CUE_RE.search(low)
                or _REQUEST_TEMP_QUESTION_RE.search(low))


_F_TEMP_RE = _re.compile(r"\d\s*(?:°\s*f\b|degrees?\s+(?:fahrenheit|f)\b|"
                         r"fahrenheit\b)")
_C_TEMP_RE = _re.compile(r"\d\s*(?:°\s*c\b|degrees?\s+(?:celsius|centigrade|"
                         r"c)\b|celsius\b|centigrade\b|°\s*k\b|kelvin\b)")
_WEATHER_CUE_RE = _re.compile(
    r"\b(?:outside|out\s+there|today|tonight|tomorrow|right\s+now|currently|"
    r"sunny|cloudy|clear|overcast|rain(?:y|ing)?|snow(?:y|ing)?|windy|foggy|"
    r"humid|chilly|breezy|forecast|feels\s+like|weather)\b")


def reply_is_temperature_conversion(reply) -> bool:
    """True when a reply states one temperature in BOTH scales ("100 degrees
    Fahrenheit is about 37.8 degrees Celsius") and names no weather cue — the
    shape of a conversion answer, not of weather stated from memory."""
    low = " ".join(str(reply or "").lower().split())
    if not (_F_TEMP_RE.search(low) and _C_TEMP_RE.search(low)):
        return False
    return not _WEATHER_CUE_RE.search(low)
