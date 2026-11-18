"""core/misheard_names.py - a name launch_app could not start, checked against
JARVIS's own features and the installed apps BEFORE any follow-up round
(v3.0.26, 2026-11-17).

Live 11-17 17:12:41 "Jarvis open hollow hood." (his "holo HUD", misheard) ran
[ACTION: launch_app, Hollow Hood]: "could not launch Hollow Hood: not found as
an app on this PC". The follow-up round offered a web search; the next try
(17:12:54 "open hollow hud") got "I'll assume you meant Hollow Knight; shall I
launch that instead?" - a game that is not installed - and that became an
open offer a yes would have run. Two minutes later he shut JARVIS down.

  * feature_for(name)       the JARVIS feature a failed app name means, or
                            None: the hologram's own voice route first
                            (core.dispatcher.holo_route on "open <name>"), then
                            a tolerant sound-alike match against FEATURES.
  * offer_line(name, f)     "I've no app called Hollow Hood, sir. Shall I open
                            the holo HUD instead?" - launch_app returns it as a
                            TERMINAL line (no follow-up round, nothing for it
                            to invent) and the monolith keeps it as the open
                            offer. An offer, never a run: a camera window never
                            opens on a guess.
  * offer_token(offer)      that offer's action token, for the yes.
  * closest_apps(name, installed)  up to two installed app names that look or
                            sound like it, for the failure line, so the
                            follow-up round names only apps that exist.
  * followup_rule(results)  the follow-up round's rule for such a failure.

Bounded on purpose ("never make a short common word match a feature by
accident"):
  * a ONE-word spoken name matches only a one-word feature name of 6+ letters
    ("hologram"); "holo", "Hulu", "Halo", "hud" never match anything;
  * every word must sound like its partner (the same phonetic key of 2+
    letters, which starts with the same letter) or be spelled within
    _SPELL_RATIO of it (5+ letters each);
  * the same number of words, or the same letters run together
    ("work shop" / "workshop");
  * only articles, "app" and punctuation are dropped before comparing;
  * a name that fits two different features fits none.

LAUNCH_NOT_FOUND_SUGGEST (core/config.py, read live) switches all of it off.
Pure stdlib (core.dispatcher is imported lazily), never raises.
"""
from __future__ import annotations

import difflib
import re
from typing import NamedTuple


class Feature(NamedTuple):
    label: str          # how the owner calls it, spoken in the offer
    action: str         # the action that opens it (must be registered)
    token: str          # the [ACTION: ...] token the yes runs
    names: tuple        # spoken names, lower case, one space


# JARVIS's own windows that "open <name>" can mean. Only explicit OPEN actions
# (never a toggle: a yes must not close what is up). Every name is two words,
# or one word of 6+ letters.
FEATURES = (
    Feature("the holo HUD", "holo_mode_on", "[ACTION: holo_mode_on]",
            ("holo hud", "hologram", "hologram hud", "holographic hud",
             "hologram window", "hologram mode", "hologram view")),
    Feature("the hand menu", "hand_menu", "[ACTION: hand_menu]",
            ("hand menu",)),
    Feature("the holographic overlay", "show_holographic_overlay",
            "[ACTION: show_holographic_overlay]",
            ("holographic overlay", "holographic display")),
)

_MIN_ONE_WORD_LEN = 6        # a one-word name: "hologram", never "holo" / "hud"
_MIN_KEY_LEN = 2             # a sound-alike key of one letter says nothing
_SPELL_MIN_LEN = 5           # spelling closeness only for longer words
_SPELL_RATIO = 0.8
APP_CUTOFF = 0.75            # closest_apps: difflib ratio on the whole name
_MAX_NAME_CHARS = 60         # a spoken name longer than this is not a name

_LEAD_WORDS = frozenset({"the", "a", "an", "my", "our"})
_TAIL_WORDS = frozenset({"app", "application", "program"})


def _words(name) -> list:
    """Lower-case words of ``name`` without leading articles or an "app"
    tail; [] for anything that is not a short string."""
    if not isinstance(name, str) or len(name) > _MAX_NAME_CHARS:
        return []
    words = re.findall(r"[a-z0-9]+", name.lower().replace("'", ""))
    while words and words[0] in _LEAD_WORDS:
        words = words[1:]
    while len(words) > 1 and words[-1] in _TAIL_WORDS:
        words = words[:-1]
    return words


def phonetic_key(word) -> str:
    """A rough sound-alike key: the first letter, then the consonants (doubles
    collapsed; h / w / y after the first letter count as vowels). "hollow",
    "holo" and "hulu" -> "hl"; "hood", "hud" and "head" -> "hd"; "knight" ->
    "nt". Only ever compared together with the first-letter and length rules
    in _word_alike. Never raises."""
    try:
        w = re.sub(r"[^a-z]", "", str(word or "").lower())
        if not w:
            return ""
        for a, b in (("ph", "f"), ("ck", "k"), ("gh", ""), ("wh", "w"),
                     ("dg", "j"), ("qu", "kw"), ("q", "k"), ("x", "ks"),
                     ("z", "s")):
            w = w.replace(a, b)
        if w[:2] in ("kn", "gn", "wr"):
            w = w[1:]
        w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
        if not w:
            return ""
        key = w[0] + re.sub(r"[aeiouhwy]", "", w[1:])
        return re.sub(r"(.)\1+", r"\1", key)
    except Exception:
        return ""


def _word_alike(a: str, b: str) -> bool:
    if a == b:
        return True
    ka, kb = phonetic_key(a), phonetic_key(b)
    if ka and ka == kb and len(ka) >= _MIN_KEY_LEN and a[:1] == b[:1]:
        return True
    return (min(len(a), len(b)) >= _SPELL_MIN_LEN
            and difflib.SequenceMatcher(None, a, b).ratio() >= _SPELL_RATIO)


def names_alike(spoken, known) -> bool:
    """True when ``spoken`` sounds like the name ``known`` under the bounds in
    the module docstring. Never raises."""
    try:
        sw, kw = _words(spoken), _words(known)
        if not sw or not kw:
            return False
        if len(kw) == 1 and len("".join(kw)) < _MIN_ONE_WORD_LEN:
            return False
        if len(sw) == 1 and len(kw) == 1:
            return (len(sw[0]) >= _MIN_ONE_WORD_LEN
                    and _word_alike(sw[0], kw[0]))
        if len(sw) == len(kw):
            return all(_word_alike(a, b) for a, b in zip(sw, kw))
        joined_s, joined_k = "".join(sw), "".join(kw)
        return (len(joined_k) >= _MIN_ONE_WORD_LEN
                and (joined_s == joined_k
                     or phonetic_key(joined_s) == phonetic_key(joined_k)
                     and difflib.SequenceMatcher(
                         None, joined_s, joined_k).ratio() >= _SPELL_RATIO))
    except Exception:
        return False


def _registered(feature: Feature, registered) -> bool:
    if registered is None:
        return True
    try:
        return feature.action in registered
    except Exception:
        return False


def feature_for(name, registered=None) -> "Feature | None":
    """The JARVIS feature a launch name that nothing could start means, or
    None. ``registered``: the live action names (a feature whose action is not
    loaded is never offered); None skips that check (tests).

    First the hologram's own voice route on "open <name>" (exact, the same
    words a whole "open the holo HUD" request routes on), then the sound-alike
    match against every FEATURES name. Never raises."""
    try:
        words = _words(name)
        if not words:
            return None
        spoken = " ".join(words)
        try:
            from core.dispatcher import holo_route
            tok = holo_route("open " + spoken)
        except Exception:
            tok = None
        if tok:
            act = re.match(r"\[ACTION:\s*([a-z0-9_]+)", tok)
            for f in FEATURES:
                if act and f.action == act.group(1) and _registered(f, registered):
                    return f
        hits = {f for f in FEATURES
                if _registered(f, registered)
                and any(names_alike(spoken, n) for n in f.names)}
        return next(iter(hits)) if len(hits) == 1 else None
    except Exception:
        return None


def _spoken_name(name) -> str:
    s = " ".join(str(name or "").split()).strip(" .!?,;:'\"[]()")
    return s[:_MAX_NAME_CHARS] or "that"


_OFFER_RE = re.compile(r"^Shall I open (.+?) instead\?$")


def offer_line(name, feature: Feature) -> str:
    """The owner-facing offer for ``feature`` when no app is called ``name``.
    Its last sentence is an offer core.offer_reply.offer_text keeps, and
    offer_token reads back."""
    return (f"I've no app called {_spoken_name(name)}, sir. "
            f"Shall I open {feature.label} instead?")


def offer_token(offer) -> str:
    """The action token of the feature ``offer`` (offer_line's last sentence,
    as the open offer keeps it) offers, else "". Never raises."""
    try:
        m = _OFFER_RE.match(" ".join(str(offer or "").split()))
        if not m:
            return ""
        for f in FEATURES:
            if f.label == m.group(1):
                return f.token
        return ""
    except Exception:
        return ""


def _app_key(s) -> str:
    return " ".join(_words(s))


def closest_apps(name, installed, n: int = 2) -> list:
    """Up to ``n`` of the ``installed`` app names that look like ``name``
    (difflib ratio >= APP_CUTOFF on the whole name, same first letter - so
    "Teams" is never "Steam") or sound like it (names_alike), closest first;
    never the name itself. [] on junk. Never raises."""
    try:
        key = _app_key(name)
        if not key:
            return []
        by_key = {}
        for app in installed or ():
            k = _app_key(app)
            if k and k != key:
                by_key.setdefault(k, str(app))
        scored = []
        for k, app in by_key.items():
            r = difflib.SequenceMatcher(None, key, k).ratio()
            if (r >= APP_CUTOFF and k[:1] == key[:1]) or names_alike(key, k):
                scored.append((-r, app))
        return [app for _r, app in sorted(scored)[:max(0, int(n))]]
    except Exception:
        return []


def closest_apps_note(close) -> str:
    """The failure line's tail for closest_apps' answer: "; the closest
    installed app is A" / "...apps are A and B" / "; no installed app has a
    similar name"."""
    close = [str(c) for c in (close or ()) if str(c).strip()]
    if not close:
        return "; no installed app has a similar name"
    if len(close) == 1:
        return f"; the closest installed app is {close[0]}"
    return f"; the closest installed apps are {' and '.join(close)}"


NOT_FOUND_MARK = "not found as an app on this pc"
FOLLOWUP_RULE = (
    "- A launch that says 'not found as an app on this PC' also says what IS "
    "installed: offer only an app that result names, the website, or a web "
    "search. Never guess another app or game name, and never say you will "
    "assume one.\n")


def suggest_on() -> bool:
    """core.config.LAUNCH_NOT_FOUND_SUGGEST, read live (default True): the
    ONE reader, for core/actions and the monolith's follow-up round alike.
    Never raises."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, "LAUNCH_NOT_FOUND_SUGGEST", True))
    except Exception:
        return True


def followup_rule(action_results, enabled=None) -> str:
    """FOLLOWUP_RULE when one of ``action_results`` ((name, result) pairs) is
    a not-found launch, else "". ``enabled``: None reads suggest_on(). Never
    raises."""
    try:
        if not (suggest_on() if enabled is None else enabled):
            return ""
        for item in action_results or ():
            result = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else ""
            if isinstance(result, str) and NOT_FOUND_MARK in result.lower():
                return FOLLOWUP_RULE
        return ""
    except Exception:
        return ""
