"""core/advice_fallback.py — "what should I have for dinner?" gets a suggestion.

WHY THIS MODULE EXISTS
======================
Live v2.0.129-v2.0.134 (2026-09-29, local model): "what should I have for
dinner tonight" was answered, sweep after sweep, with
"[intent:dry_wit] A bold choice, if I may say so, sir — though I'm afraid my
culinary expertise is somewhat limited to data processing." Two faults in one
line: "A bold choice" answers a choice nobody made, and the rest declines the
question. The prompt already says to match the opener to the content and to
deliver a requested recommendation (core/prompts.py, "MATCH THE OPENER TO THE
CONTENT", "DELIVER WHAT WAS ASKED"); the model still does it. This module is
the deterministic safety net, the same shape as core/joke_fallback.py.

THE RULES
=========
``apply(reply, user_text, now=...)`` returns the text to speak INSTEAD of the
reply, or None to keep the reply:

  1. A meal question ("what should I have for dinner", "what should I eat",
     "any lunch ideas") answered WITHOUT a suggestion (no "how about", no
     "try", no dish named) and WITH a stance quip or a decline ("a bold
     choice", "my culinary expertise is limited", "that's up to you") gets a
     bundled suggestion for that meal. The meal is the one named, else the one
     the clock implies.
  2. Any request for a suggestion ("what should I watch tonight?") whose reply
     OPENS with "A bold choice[, if I may say so][, sir]" but then does suggest
     something keeps the suggestion and loses the opener.

Everything else is returned unchanged. "I'm skipping class tomorrow" -> "A bold
choice, sir." is a correct use of the phrase and is never touched: the owner's
turn must be a request for a suggestion. The suggestions are short, generic and
public on purpose: the repo is public and this is a fallback, not a menu.

Pure: stdlib only, no monolith import — tested on the light-deps CI runner
(tests/test_advice_fallback.py).
"""
from __future__ import annotations

import datetime as _dt
import random
import re
from typing import Optional

__all__ = ["MEAL_SUGGESTIONS", "meal_of_request", "is_advice_request",
           "has_suggestion", "looks_like_non_answer", "strip_misfit_opener",
           "early_hold", "apply"]

MEAL_SUGGESTIONS: dict[str, tuple[str, ...]] = {
    "breakfast": (
        "Eggs and toast, sir. Five minutes, and it will carry you to lunch.",
        "Oatmeal with some fruit on top, sir. Warm, cheap and filling.",
        "A breakfast burrito, sir: eggs, cheese and a tortilla. Portable, too.",
        "Yogurt, granola and a banana, sir. No cooking required.",
    ),
    "lunch": (
        "A grilled cheese and a bowl of soup, sir. A classic for a reason.",
        "A wrap, sir: whatever protein is in the fridge, some greens, and "
        "you're done in five minutes.",
        "Leftovers, sir, if there are any. Failing that, a good sandwich is "
        "hard to beat.",
        "A rice bowl, sir: rice, a fried egg and whatever vegetables are "
        "handy.",
    ),
    "dinner": (
        "How about tacos tonight, sir? Quick, cheap and very hard to get "
        "wrong.",
        "A stir-fry, sir: whatever is in the fridge, a hot pan and ten "
        "minutes.",
        "Pasta with garlic, olive oil and whatever vegetables you have, sir. "
        "Fifteen minutes, very little washing up.",
        "Breakfast for dinner, sir: eggs, toast, perhaps bacon. Nobody has "
        "ever regretted it.",
        "A sheet-pan dinner, sir: chicken and vegetables on one tray, forty "
        "minutes in the oven.",
        "Burrito bowls, sir: rice, beans, whatever protein you have, and "
        "salsa.",
    ),
    "snack": (
        "Apple slices and peanut butter, sir. Sweet, salty, and done in a "
        "minute.",
        "A handful of nuts and some cheese, sir. It will hold you over.",
        "Popcorn, sir. Cheap, quick, and it counts as a whole grain.",
    ),
}

_MEAL_WORDS = {
    "breakfast": "breakfast", "brunch": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner", "supper": "dinner", "tonight": "dinner",
    "snack": "snack",
}

_FILLER_RE = re.compile(
    r"^(?:(?:hey|hi|ok(?:ay)?|so|and|well|um+|uh+|jarvis|sir|please|alright|"
    r"quick\s+question|question)[,.!\s]+)+")

# "what should I have/eat/make/cook/order/get/grab ..." (for dinner / tonight
# / at all), "what's for dinner", "any dinner ideas", "suggest something for
# lunch", "I don't know what to eat", "what do you think I should eat".
_MEAL_VERB = r"(?:have|eat|make|cook|order|get|grab|fix)"
_MEAL_REQUEST_RE = re.compile(
    r"\bwhat\s+(?:should|shall|could|do\s+you\s+think)\s+(?:i|we)\s+(?:should\s+)?"
    + _MEAL_VERB + r"\b"
    r"|\bwhat(?:\s+is|'s)\s+for\s+(?:breakfast|brunch|lunch|dinner|supper)\b"
    r"|\b(?:any|some|got\s+any|give\s+me\s+(?:some|an?))\s+"
    r"(?:good\s+|quick\s+|easy\s+)?(?:breakfast|brunch|lunch|dinner|supper|"
    r"snack|meal|food)\s+(?:ideas?|suggestions?|recommendations?)\b"
    r"|\b(?:suggest|recommend)\s+(?:me\s+)?(?:something|a\s+meal|a\s+snack|"
    r"what)\s+(?:(?:to|i\s+should)\s+" + _MEAL_VERB + r"\s+)?"
    r"(?:for\s+)?(?:breakfast|brunch|lunch|dinner|supper|tonight)\b"
    r"|\b(?:don'?t|do\s+not|can'?t)\s+(?:know|decide)\s+what\s+to\s+"
    + _MEAL_VERB + r"\b"
    r"|\bwhat\s+to\s+" + _MEAL_VERB + r"\s+for\s+(?:breakfast|brunch|lunch|"
    r"dinner|supper)\b")
_MEAL_WORD_RE = re.compile(
    r"\b(breakfast|brunch|lunch|dinner|supper|snack|tonight)\b")
# "what should I have" without a meal word is still food when a food verb
# carries it ("eat", "cook"); "have" and "get" alone are too broad ("what
# should I get my dad", "what should I have done") unless a meal word is there.
_FOOD_VERB_RE = re.compile(r"\b(?:eat|cook)\b")

# Requests for a suggestion of any kind (rule 2).
_ADVICE_REQUEST_RE = re.compile(
    r"\bwhat\s+(?:should|shall|could)\s+(?:i|we)\b"
    r"|\bwhat\s+do\s+you\s+(?:think\s+i\s+should|recommend|suggest)\b"
    r"|\b(?:any|some|got\s+any)\s+(?:[\w'-]+\s+){0,2}?"
    r"(?:ideas?|suggestions?|recommendations?)\b"
    r"|^(?:can\s+you\s+|could\s+you\s+)?(?:suggest|recommend)\b"
    r"|\bwhich\s+(?:one\s+)?should\s+i\b"
    r"|\bwhat(?:'s|\s+is)\s+a\s+good\b")

# A reply that suggests something: a suggestion verb, or a dish named.
_SUGGESTION_RE = re.compile(
    r"\b(?:how\s+about|what\s+about|why\s+not|might\s+i\s+suggest|"
    r"may\s+i\s+suggest|i'?d\s+(?:suggest|go\s+(?:with|for))|i\s+suggest|"
    r"i\s+(?:would\s+)?recommend(?!\s+against)|i'?d\s+recommend(?!\s+against)|"
    r"you\s+(?:could|might|can)\s+(?:try|have|make|cook|order|grab|go\s+for|"
    r"watch|read|play|listen)|"
    r"(?:try|consider|go\s+(?:with|for))\s+(?:a|an|some|the|making|"
    r"ordering|cooking|watching|reading|playing)\b|perhaps\s+(?:a|an|some)\b|"
    r"my\s+(?:suggestion|recommendation|pick|vote)\s+(?:is|would\s+be))")
_DISH_RE = re.compile(
    r"\b(?:tacos?|pizza|pasta|spaghetti|stir[\s-]?fry|burgers?|salad|soup|"
    r"sandwich(?:es)?|curry|chili|burritos?|sushi|ramen|noodles|steak|"
    r"chicken|salmon|fish|eggs?|omelett?e|pancakes?|waffles?|oatmeal|"
    r"quesadillas?|wraps?|rice|lasagna|casserole|fajitas?|grilled\s+cheese|"
    r"toast|bacon|yogh?urt|granola|fruit|bananas?|apples?|peanut\s+butter|"
    r"nuts|cheese|popcorn|leftovers)\b")

# Stance quips and declines that mean "no suggestion given".
_NON_ANSWER_RE = re.compile(
    r"\ba\s+bold\s+choice\b"
    r"|\bare\s+you\s+quite\s+sure\b"
    r"|\bi'?ll\s+note\s+that\s+for\s+posterity\b"
    r"|\b(?:culinary|cooking|dining|gastronomic|food)\s+(?:expertise|"
    r"knowledge|skills?|credentials|abilities)\b"
    r"|\b(?:expertise|knowledge)\s+is\s+(?:somewhat\s+|rather\s+)?limited\b"
    r"|\bnot\s+(?:really\s+)?my\s+(?:area|department|forte|field|"
    r"strong\s+suit)\b"
    r"|\b(?:entirely\s+|all\s+|completely\s+)?up\s+to\s+you\b"
    r"|\bi'?ll\s+leave\s+(?:that|the\s+menu|it)\s+(?:to|in)\b"
    r"|\bi\s+(?:don'?t|do\s+not)\s+(?:eat|have\s+(?:a\s+)?(?:taste|stomach|"
    r"palate))\b"
    r"|\b(?:lacking|without|no)\s+(?:a\s+)?(?:taste\s+buds|palate|stomach)\b"
    r"|\bi'?m\s+afraid\s+i\s+(?:can'?t|cannot|couldn'?t)\s+(?:help|say|"
    r"decide|choose|recommend)\b")

# "A bold choice[, if I may say so][, sir][,.;:!—-]" at the very start.
_MISFIT_OPENER_RE = re.compile(
    r"^\s*a\s+bold\s+choice(?:\s*,?\s*if\s+i\s+may(?:\s+say\s+so)?)?"
    r"(?:\s*,?\s*sir)?\s*[,.;:!—–-]*\s*", re.I)
_TAG_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)+")


def _norm(text: str) -> str:
    t = re.sub(r"\[[^\]]*\]", " ", str(text or ""))
    t = t.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", t).strip().lower()


def _request(user_text: str) -> str:
    return _FILLER_RE.sub("", _norm(user_text)).strip()


def _meal_for_hour(hour: int) -> str:
    if 4 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 16:
        return "lunch"
    if 16 <= hour < 22:
        return "dinner"
    return "snack"


def meal_of_request(user_text: str, now: Optional[_dt.datetime] = None
                    ) -> Optional[str]:
    """"breakfast" / "lunch" / "dinner" / "snack" when the owner is asking what
    to eat, else None. A named meal wins; otherwise the hour decides."""
    t = _request(user_text)
    if not t or not _MEAL_REQUEST_RE.search(t):
        return None
    named = _MEAL_WORD_RE.search(t)
    if named:
        return _MEAL_WORDS[named.group(1)]
    if not _FOOD_VERB_RE.search(t):
        return None
    return _meal_for_hour((now or _dt.datetime.now()).hour)


def is_advice_request(user_text: str) -> bool:
    """True when the owner asks for a suggestion or recommendation."""
    t = _request(user_text)
    return bool(t) and (bool(_ADVICE_REQUEST_RE.search(t))
                        or bool(_MEAL_REQUEST_RE.search(t)))


def has_suggestion(reply: str) -> bool:
    t = _norm(reply)
    return bool(_SUGGESTION_RE.search(t)) or bool(_DISH_RE.search(t))


def looks_like_non_answer(reply: str) -> bool:
    return bool(_NON_ANSWER_RE.search(_norm(reply)))


def strip_misfit_opener(reply: str, user_text: str) -> Optional[str]:
    """The reply without a leading "A bold choice, ..." when the owner ASKED
    for a suggestion and the rest of the reply gives one; None otherwise."""
    try:
        if not reply or not is_advice_request(user_text):
            return None
        tags = _TAG_RE.match(reply)
        head = tags.group(0) if tags else ""
        body = reply[len(head):]
        m = _MISFIT_OPENER_RE.match(body)
        if not m:
            return None
        rest = body[m.end():].strip()
        if len(rest.split()) < 3 or not has_suggestion(rest):
            return None
        if looks_like_non_answer(rest):
            return None
        rest = rest[0].upper() + rest[1:]
        return head + rest
    except Exception:
        return None


def early_hold(piece: str, user_text: str) -> bool:
    """True when a streamed opening sentence must NOT be voiced early because
    apply() may replace or trim it once the whole reply is in."""
    try:
        if not is_advice_request(user_text):
            return False
        body = _TAG_RE.sub("", piece or "")
        return bool(_MISFIT_OPENER_RE.match(body)) or looks_like_non_answer(body)
    except Exception:
        return True


_last: list[Optional[str]] = [None]


def _pick(meal: str, rng) -> str:
    options = MEAL_SUGGESTIONS.get(meal) or MEAL_SUGGESTIONS["dinner"]
    pool = [s for s in options if s != _last[0]] or list(options)
    choice = rng.choice(pool)
    _last[0] = choice
    return choice


def apply(reply: str, user_text: str, *, now: Optional[_dt.datetime] = None,
          rng=None) -> Optional[str]:
    """The text to speak INSTEAD of ``reply``, or None to keep it. ``now``
    picks the meal when none is named; ``rng`` is a ``random.Random``-like
    object (tests pass a seeded one). Never raises."""
    try:
        if not reply:
            return None
        meal = meal_of_request(user_text, now)
        if meal and not has_suggestion(reply) and looks_like_non_answer(reply):
            return _pick(meal, rng or random)
        return strip_misfit_opener(reply, user_text)
    except Exception:
        return None
