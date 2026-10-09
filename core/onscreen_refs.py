"""core/onscreen_refs.py - what the owner's words say about the screen
(2026-10-05).

WHY THIS EXISTS
===============
Live 00:28:23 "Jarvis, go ahead and click that Mr. Beast video and then go
back into ... wake word mode": "that ... video" is a thing ALREADY ON THE
SCREEN, yet the brain answered youtube_play (a new search, a new window on
another monitor). Nothing in the router, the prompt or the dispatcher knew a
demonstrative pointed at the screen. Then "that's not the right video" and
"I wanted the one that was on screen at the time" went to recall_screen,
which re-asked the vision model and invented "the Kai Cenat video".

These are the pure-text predicates every layer asks the same way:

  * is_onscreen_reference   - the turn points at something on screen (router
                              gate, rewrite guard, scene freeze);
  * onscreen_click_target   - a WHOLE "click that X" request (built-in route);
  * is_ui_correction        - "not that one" / "go back" right after a UI
                              action;
  * is_scene_back_reference - "the one that was on screen (at the time)";
  * pending_choice_answer   - the answer to JARVIS's own "which one?";
  * is_screen_recall_request, screen_memory_command, forget_span,
    claude_note             - recall, watch controls, forget, developer notes.

Pure stdlib; every public function never raises.
"""
from __future__ import annotations

import re

from core.lead_fillers import strip_lead_filler as _strip_lead_filler

__all__ = [
    "clean", "is_onscreen_reference", "onscreen_click_target",
    "referent_phrase", "is_ui_correction", "is_scene_back_reference",
    "pending_choice_answer", "is_screen_recall_request",
    "screen_memory_command", "forget_span", "claude_note", "is_video_like",
    "rewrite_referent", "youtube_play_query",
]

_WAKE_RE = re.compile(r"^\s*(?:(?:hey|ok|okay)[\s,]+)?jarvis\b[\s,.:;!?-]*",
                      re.IGNORECASE)
_TAIL_PUNCT_RE = re.compile(r"[.!?,;:\s]+$")


def clean(text) -> str:
    """The utterance without a leading wake word / polite lead-in and
    without trailing punctuation, whitespace collapsed."""
    try:
        s = _WAKE_RE.sub("", str(text or ""), count=1)
        s = _strip_lead_filler(s)
        s = _WAKE_RE.sub("", s, count=1)
        s = _strip_lead_filler(s)
        return _TAIL_PUNCT_RE.sub("", " ".join(s.split()))
    except Exception:
        return ""


# ── is_onscreen_reference ────────────────────────────────────────────────
# "click" as a verb - not the noun "clicks" ("how many clicks did my post
# get" is not about the screen; review 2026-10-05).
_CLICK_RE = re.compile(r"\bclick(?:ed|ing)?\b", re.IGNORECASE)
_VERBS = r"(?:press|tap|select|pick|choose|hit|open|play|watch|start)"
_NOUNS = (r"(?:video|videos|clip|clips|link|links|thumbnail|thumbnails|result|"
          r"results|button|buttons|tab|tabs|icon|icons|one|ones|thing|option|"
          r"entry|item|page)")
# verb ... within 6 words ... that / this / the one ... (<= 4 words) ... noun;
# "that one"; and "the one" only when a place / description follows ("the
# one on the left", "the one with the dog", "the one you just opened") -
# never "start the one hour timer" or "play the one by Drake".
_DEMONSTRATIVE_RE = re.compile(
    r"\b" + _VERBS + r"\b(?:\s+\S+){0,6}?\s+(?:that|this|the\s+one)"
    r"(?:\s+\S+){0,4}?\s+" + _NOUNS + r"\b"
    r"|\b" + _VERBS + r"\b(?:\s+\S+){0,6}?\s+(?:that|this)\s+one\b"
    r"|\b" + _VERBS + r"\b(?:\s+\S+){0,6}?\s+the\s+one\s+(?:on|at|in|with|"
    r"from|that|that's|thats|which|you|i|playing|showing|there|here|up)\b",
    re.IGNORECASE)
_SCREEN_PLACE_RE = re.compile(
    r"\bon\s+(?:the\s+|my\s+)?(?:screen|page|"
    r"(?:left|right|top|middle|main|center|centre|primary|bottom|upper|"
    r"lower)\s+(?:monitor|screen|display))\b"
    r"|\bi'?m\s+looking\s+at\b|\bi\s+am\s+looking\s+at\b|\bright\s+there\b"
    r"|\byou\s+just\s+opened\b|\bthat\s+was\s+on\s+(?:the\s+)?screen\b"
    r"|\bthe\s+one\s+on\s+(?:the\s+)?screen\b", re.IGNORECASE)
_MUSIC_NOUN_RE = re.compile(
    r"\b(?:song|songs|track|tracks|playlist|playlists|album|albums|artist|"
    r"artists|mix|mixes|music)\b", re.IGNORECASE)
_SCREENISH_RE = re.compile(r"\b(?:screen|page|monitor|display)\b",
                           re.IGNORECASE)
_AGAIN_RE = re.compile(r"\bagain\b", re.IGNORECASE)


def is_onscreen_reference(text) -> bool:
    """True when the owner's words point at something ON THE SCREEN: any
    "click", a select/play/open verb with "that / this / the one" + a thing
    noun, or "on the screen / page / left monitor", "I'm looking at",
    "right there", "you just opened". Never for a song / track / playlist /
    mix unless a screen word is said too, and never with "again" ("play that
    song again" is the music player's)."""
    try:
        s = " ".join(str(text or "").split())
        if not s:
            return False
        if _CLICK_RE.search(s):
            return True
        if _AGAIN_RE.search(s):
            return False
        if _MUSIC_NOUN_RE.search(s) and not _SCREENISH_RE.search(s):
            return False
        if _DEMONSTRATIVE_RE.search(s):
            return True
        return bool(_SCREEN_PLACE_RE.search(s))
    except Exception:
        return False


# ── referent phrase + the whole-utterance click route ────────────────────
_REFERENT_RE = re.compile(
    r"\b(?:that|this|the)\s+((?:\S+\s+){0,6}?)" + _NOUNS + r"\b",
    re.IGNORECASE)
# Words that do not name WHICH thing.
_EMPTY_WORDS = frozenset("""
the a an that this these those one ones it video videos clip link thumbnail
result button tab icon thing option entry item page on screen my your there
here right left top middle bottom monitor
""".split())


def _content_words(phrase) -> list:
    return [w for w in re.findall(r"[a-z0-9$']+", str(phrase or "").lower())
            if w not in _EMPTY_WORDS]


def referent_phrase(text) -> str:
    """The "that MrBeast video" part of an utterance (what to look for), or
    "" when the words name nothing in particular. Never raises."""
    try:
        s = clean(text)
        m = re.search(r"\bclick(?:\s+on)?\s+(.+)$", s, re.IGNORECASE)
        cand = None
        if m:
            cand = _cut_trailing_clause(m.group(1))
        if not cand:
            m2 = _REFERENT_RE.search(s)
            if m2:
                cand = m2.group(0)
        if not cand:
            return ""
        cand = cand.strip(" ,.")
        return cand if _content_words(cand) else ""
    except Exception:
        return ""


_TRAILING_CLAUSE_RE = re.compile(
    r"\s*(?:,|\band\b|\bthen\b|\bafter\s+that\b|\balso\b|\bbut\b|;)", re.IGNORECASE)
# Location qualifiers are not a second command.
_LOCATION_TAIL_RE = re.compile(
    r"\s+(?:on|in)\s+(?:the\s+|my\s+)?(?:(?:left|right|top|middle|main|"
    r"center|centre|primary|bottom|upper|lower)\s+(?:monitor|screen|display)|"
    r"screen|page|you\s?tube|chrome|the\s+browser)\s*$", re.IGNORECASE)
_CLICK_ROUTE_RE = re.compile(
    r"^(?P<v>click|tap|press|select)(?:\s+on)?\s+(?P<t>(?:that|this|the)\s+.+)$",
    re.IGNORECASE)
# What a WHOLE "press / select the X" names that is not a thing on the
# screen (review 2026-10-05: "press the enter key", "press the mute
# button", "select the USB desk mic" all became screen clicks before
# the brain saw them): a keyboard key, an audio / camera device, a mute or
# volume control. "click / tap" is screen-only, so only a word that says
# KEY stops those ("click the return policy link", "click the next arrow"
# are on the page); "press / select" also stops on a bare key name.
_KEY_WORD_RE = re.compile(r"\b(?:keys?|keyboard|space\s*bar|spacebar|"
                          r"hotkey|shortcut)\b", re.IGNORECASE)
_KEY_NAME_RE = re.compile(
    r"\b(?:enter|escape|esc|return|backspace|delete|tab|shift|ctrl|control|"
    r"alt|f\d{1,2}|page\s+(?:up|down)|home|end|(?:up|down|left|right)\s+"
    r"arrow)\b", re.IGNORECASE)
_DEVICE_TARGET_RE = re.compile(
    r"\b(?:mics?|microphones?|headsets?|headphones|earbuds|speakers?|"
    r"webcams?|cameras?|audio\s+devices?|sound\s+devices?|mute|unmute|"
    r"volume)\b|\bas\s+(?:the\s+|my\s+)?(?:output|input|default)\b",
    re.IGNORECASE)
_PLAY_ROUTE_RE = re.compile(
    r"^(?:play|open|watch)\s+(?P<t>(?:that|this)\s+.+?\s+"
    r"(?:video|clip|link|thumbnail))$", re.IGNORECASE)


def _cut_trailing_clause(s) -> str:
    m = _TRAILING_CLAUSE_RE.search(s or "")
    return (s[:m.start()] if m else (s or "")).strip()


def onscreen_click_target(text) -> "str | None":
    """The target of a WHOLE "click that X" request ("click the Save
    button", "play that MrBeast video"), else None: the words must name
    something (not just "that one" / "that video") and carry no second
    command ("... and then go back into wake word mode" is the brain's).
    A trailing location ("on the middle monitor", "on YouTube") is kept out
    of the target, not treated as a second command. Never raises."""
    try:
        s = clean(text)
        if not s or "[" in s or "]" in s:
            return None
        loc = _LOCATION_TAIL_RE.search(s)
        core = s[:loc.start()] if loc else s
        m = _CLICK_ROUTE_RE.match(core) or _PLAY_ROUTE_RE.match(core)
        if not m:
            return None
        target = m.group("t").strip()
        if _TRAILING_CLAUSE_RE.search(target):
            return None
        if not _content_words(target):
            return None
        verb = (m.groupdict().get("v") or "").lower()
        if _KEY_WORD_RE.search(target):
            return None
        if verb in ("press", "select") and (_KEY_NAME_RE.search(target)
                                           or _DEVICE_TARGET_RE.search(target)):
            return None
        if _MUSIC_NOUN_RE.search(target) and not _SCREENISH_RE.search(s):
            return None
        if len(target) > 120:
            return None
        return target
    except Exception:
        return None


_VIDEO_LIKE_RE = re.compile(r"\b(?:video|videos|clip|clips|watch|episode|"
                            r"trailer|stream|vlog|short|shorts)\b",
                            re.IGNORECASE)

# ── the rewrite guard's referent ────────────────────────────────────────
# A request to SEARCH / look something up is never "the thing on screen"
# (review 2026-10-05: "search google for how to click a link in python"
# found an on-screen "Python" link and its search became a click).
_SEARCH_REQUEST_RE = re.compile(
    r"^(?:search|google|bing|look\s+up|look\s+for|find\s+me|find\s+out|"
    r"research)\b|\bsearch\s+(?:google|youtube|bing|the\s+web|online|for)\b"
    r"|\blook\s+(?:it|that|this)\s+up\b", re.IGNORECASE)
_DEMONSTRATIVE_WORD_RE = re.compile(r"\b(?:that|this|these|those|the\s+one)\b",
                                    re.IGNORECASE)


def rewrite_referent(text) -> "str | None":
    """The on-screen referent that may turn the brain's youtube_play /
    open_url / web_search into a click: the owner pointed with "that /
    this / the one" at a thing ("click that MrBeast video and then ...",
    "play that phone review video"), and did not ask to search. None
    otherwise. Never raises."""
    try:
        s = clean(text)
        if not s or _SEARCH_REQUEST_RE.search(s):
            return None
        ref = referent_phrase(s)
        if not ref or not _DEMONSTRATIVE_WORD_RE.search(ref):
            return None
        return ref
    except Exception:
        return None


_YT_PLAY_THAT_RE = re.compile(
    r"^(?:play|watch|put\s+on|pull\s+up)\s+(?:that|this)\s+(?P<q>.+?)\s+"
    r"(?:video|clip)\s+(?:on|from)\s+you\s?tube$", re.IGNORECASE)


def youtube_play_query(text) -> "str | None":
    """"play that MrBeast video on YouTube" -> "MrBeast": the search to run
    when the thing is NOT on screen (the owner named YouTube, so main's
    search-and-play stands). None for anything else. Never raises."""
    try:
        m = _YT_PLAY_THAT_RE.match(clean(text))
        if not m:
            return None
        q = " ".join(m.group("q").split()).strip(" ,.")
        return q if _content_words(q) else None
    except Exception:
        return None


def is_video_like(text) -> bool:
    try:
        return bool(_VIDEO_LIKE_RE.search(str(text or "")))
    except Exception:
        return False


# ── corrections right after a UI action ─────────────────────────────────
_CORRECTION_OTHER_RE = re.compile(
    r"^(?:no[,.!]?\s+)?(?:"
    r"not\s+(?:that|this)(?:\s+one)?|wrong\s+(?:one|video|link|thing|tab|"
    r"button)|"
    r"(?:that'?s|that\s+is|this\s+is|this'?s|it'?s)\s+(?:not\s+the\s+right|"
    r"the\s+wrong|not\s+the\s+one|not\s+it|not\s+what\s+i\s+(?:meant|wanted|"
    r"asked\s+for))(?:\s+(?:one|video|link|thing|tab|button|page))?|"
    r"(?:i\s+(?:meant|wanted|said)\s+)?the\s+other\s+one|"
    r"you\s+(?:clicked|picked|opened|played)\s+the\s+wrong\s+(?:one|video|"
    r"thing|link)"
    r")(?:[\s,]+(?:jarvis|sir|please))*$", re.IGNORECASE)
_CORRECTION_UNDO_RE = re.compile(
    r"^(?:no[,.!]?\s+)?(?:go\s+back|undo\s+(?:that|it|the\s+click)|"
    r"take\s+(?:that|it)\s+back|back\s+(?:up|out))"
    r"(?:\s+(?:a\s+page|one\s+page))?(?:[\s,]+(?:jarvis|sir|please))*$",
    re.IGNORECASE)


def is_ui_correction(text) -> "str | None":
    """"other" for "not that one" / "wrong one" / "that's not the right
    video" / "the other one", "undo" for "go back" / "undo that", else
    None. The caller decides whether a JARVIS UI action is recent enough
    for these words to be about it. Never raises."""
    try:
        s = clean(text)
        if not s:
            return None
        if _CORRECTION_OTHER_RE.match(s):
            return "other"
        if _CORRECTION_UNDO_RE.match(s):
            return "undo"
    except Exception:
        return None
    return None


_SCENE_BACK_RE = re.compile(
    r"\b(?:the\s+one|the\s+video|the\s+link|it)\s+(?:that\s+|which\s+)?"
    r"(?:was|were)\s+(?:on\s+(?:the\s+)?screen|there|showing|up)"
    r"(?:\s+(?:at\s+the\s+time|before|earlier|then|a\s+minute\s+ago))?\b"
    r"|\bthe\s+one\s+(?:i\s+(?:was\s+)?(?:looking\s+at|pointing\s+at|"
    r"meant))\b", re.IGNORECASE)


def is_scene_back_reference(text) -> bool:
    """"I wanted the one that was on screen at the time" - a pointer back at
    what was on screen BEFORE JARVIS's last UI action. Never raises."""
    try:
        return bool(_SCENE_BACK_RE.search(clean(text)))
    except Exception:
        return False


# ── the answer to JARVIS's own "which one?" ─────────────────────────────
_PICK_ORD = {"first": 1, "1st": 1, "one": 1, "1": 1,
             "second": 2, "2nd": 2, "two": 2, "2": 2,
             "third": 3, "3rd": 3, "three": 3, "3": 3,
             "fourth": 4, "4th": 4, "four": 4, "4": 4,
             "last": -1}
_PICK_RE = re.compile(
    r"^(?:(?:the|number|option|no\.?)\s+)?"
    r"(?P<o>first|1st|second|2nd|third|3rd|fourth|4th|last|one|two|three|"
    r"four|1|2|3|4)(?:\s+one)?(?:\s+(?:please|sir|jarvis))*$", re.IGNORECASE)
_PICK_POS_RE = re.compile(
    r"^(?:the\s+)?(?P<w>top|bottom|left|right|upper|lower|leftmost|"
    r"rightmost|middle)(?:\s+one)?(?:\s+(?:please|sir|jarvis))*$",
    re.IGNORECASE)
_YES_RE = re.compile(r"^(?:yes|yeah|yep|yup|sure|please|do\s+it|go\s+ahead|"
                     r"that\s+one|correct|right)(?:[\s,]+(?:please|sir|"
                     r"jarvis|do\s+it))*$", re.IGNORECASE)


def pending_choice_answer(text, options, *, allow_yes: bool = False):
    """The 0-based index of the option ``text`` picks among ``options``
    (dicts with "label" and "rect" [x, y, w, h], in the order JARVIS named
    them), else None. "the second one", "number two", "the top one", "the
    MrBeast one" (a phrase unique to one option), and - only when
    ``allow_yes`` and exactly one option was offered - "yes". Never
    raises."""
    try:
        s = clean(text).lower()
        opts = list(options or ())
        if not s or not opts:
            return None
        m = _PICK_RE.match(s)
        if m:
            n = _PICK_ORD.get(m.group("o").lower())
            if n == -1:
                return len(opts) - 1
            if n and 1 <= n <= len(opts):
                return n - 1
            return None
        m = _PICK_POS_RE.match(s)
        if m and all(o.get("rect") for o in opts):
            w = m.group("w").lower()
            key = {
                "top": lambda o: o["rect"][1], "upper": lambda o: o["rect"][1],
                "bottom": lambda o: -o["rect"][1],
                "lower": lambda o: -o["rect"][1],
                "left": lambda o: o["rect"][0],
                "leftmost": lambda o: o["rect"][0],
                "right": lambda o: -o["rect"][0],
                "rightmost": lambda o: -o["rect"][0],
            }.get(w)
            if key is not None:
                ranked = sorted(range(len(opts)), key=lambda i: key(opts[i]))
                return ranked[0]
            if w == "middle" and len(opts) == 3:
                return sorted(range(3), key=lambda i: opts[i]["rect"][0])[1]
            return None
        if allow_yes and len(opts) == 1 and _YES_RE.match(s):
            return 0
        # A phrase that names exactly one option ("the burger one").
        words = [w for w in re.findall(r"[a-z0-9$']+", s)
                 if w not in _EMPTY_WORDS and w not in (
                     "i", "meant", "mean", "want", "wanted", "please", "sir",
                     "jarvis", "click", "play", "open", "pick", "choose")]
        if not words:
            return None
        hits = []
        for i, o in enumerate(opts):
            label = str(o.get("label") or "").lower()
            ltoks = set(re.findall(r"[a-z0-9$']+", label.replace("mr. ", "mr")))
            sq = re.sub(r"[^a-z0-9]", "", label)
            if all(w in ltoks or (len(w) >= 4 and w in sq) for w in words):
                hits.append(i)
        return hits[0] if len(hits) == 1 else None
    except Exception:
        return None


# ── recall, watch controls, forget ──────────────────────────────────────
_RECALL_RE = re.compile(
    r"\bwhat\s+was\s+(?:that|the)\b.*\b(?:on|in)\s+(?:the\s+|my\s+)?"
    r"(?:\w+\s+)?(?:monitor|screen|page|display)\b"
    r"|\bwhat\s+was\s+(?:that|the)\s+(?:video|page|link|site|article|"
    r"thing|window|tab)\b"
    r"|\bwhat\s+was\s+i\s+(?:looking\s+at|watching|reading|doing)\b"
    r"|\bwhat\s+(?:did|have)\s+(?:i|you)\s+(?:see|seen|watch|watched|look\s+at)\b"
    r"|\b(?:\d+|a|an|few|couple(?:\s+of)?|ten|five|twenty|thirty)\s+"
    r"(?:minutes?|mins?|hours?|seconds?)\s+ago\b.*\b(?:screen|monitor|"
    r"looking|watching|open|opened|page)\b"
    r"|\b(?:screen|monitor|looking|watching)\b.*\b(?:\d+|a|an|few|ten|five|"
    r"twenty|thirty)\s+(?:minutes?|mins?|hours?)\s+ago\b"
    r"|\bthe\s+one\s+that\s+was\s+on\s+(?:the\s+)?screen\b"
    r"|\bearlier\s+on\s+(?:my|the)\s+screen\b", re.IGNORECASE)


def is_screen_recall_request(text) -> bool:
    """"What was that video on the middle monitor", "what was I looking at
    ten minutes ago", "the one that was on screen". Never raises."""
    try:
        return bool(_RECALL_RE.search(" ".join(str(text or "").split())))
    except Exception:
        return False


_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
              "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30,
              "forty": 40, "forty-five": 45, "sixty": 60, "few": 3,
              "couple": 2, "half": 0.5}


def _amount(word) -> "float | None":
    w = str(word or "").lower().strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", w):
        return float(w)
    return _NUM_WORDS.get(w)


_WATCH_PAUSE_RE = re.compile(
    r"^(?:(?:please\s+)?(?:stop|quit|pause)\s+watching(?:\s+(?:my|the)\s+"
    r"screens?)?|(?:don'?t|do\s+not)\s+watch(?:\s+(?:my|the)\s+screens?)?|"
    r"stop\s+(?:looking\s+at|reading)\s+(?:my|the)\s+screens?|"
    r"pause\s+(?:the\s+)?screen\s+(?:memory|watching|watcher))"
    r"(?:\s+for\s+(?:the\s+next\s+)?(?P<n>\d+|a|an|one|two|three|five|ten|"
    r"fifteen|twenty|thirty|half)(?:\s+an?)?\s+(?P<u>minutes?|mins?|hours?))?"
    r"(?:\s+(?:please|sir|jarvis|for\s+now|for\s+a\s+bit|for\s+a\s+while))*$",
    re.IGNORECASE)
_WATCH_THIS_RE = re.compile(
    r"^(?:(?:don'?t|do\s+not)\s+(?:watch|read|record|look\s+at)\s+"
    r"(?:this|that)(?:\s+(?:window|page|tab|one|app))?|"
    r"(?:stop|quit)\s+watching\s+(?:this|that)(?:\s+(?:window|page|tab|app))?|"
    r"(?:this|that)\s+is\s+private|ignore\s+this\s+(?:window|page|tab))"
    r"(?:\s+(?:please|sir|jarvis))*$", re.IGNORECASE)
_WATCH_APP_RE = re.compile(
    r"^(?:(?:don'?t|do\s+not|never)\s+(?:watch|read|record)|"
    r"(?:stop|quit)\s+watching)\s+(?:my\s+|the\s+)?(?P<app>[a-z0-9][\w .+-]{1,40}?)"
    r"(?:\s+(?:app|window|windows))?(?:\s+(?:please|sir|jarvis|ever|again|"
    r"anymore|any\s+more))*$", re.IGNORECASE)
_WATCH_RESUME_RE = re.compile(
    r"^(?:you\s+can\s+(?:watch|look)(?:\s+(?:my|the)\s+screens?)?\s+again|"
    r"(?:start|resume|keep)\s+watching(?:\s+(?:my|the)\s+screens?)?(?:\s+again)?|"
    r"(?:watch|look\s+at)\s+(?:my|the)\s+screens?\s+again|"
    r"resume\s+(?:the\s+)?screen\s+(?:memory|watching|watcher)|"
    r"turn\s+(?:the\s+)?screen\s+memory\s+(?:back\s+)?on)"
    r"(?:\s+(?:please|sir|jarvis|now))*$", re.IGNORECASE)
_WATCH_STATUS_RE = re.compile(
    r"^(?:are\s+you\s+(?:still\s+)?watching(?:\s+(?:my|the)\s+screens?)?|"
    r"is\s+(?:the\s+)?screen\s+memory\s+on|"
    r"(?:what'?s|what\s+is)\s+(?:the\s+)?screen\s+(?:memory|watching)\s+status|"
    r"screen\s+(?:memory|watch(?:ing|er)?)\s+status)"
    r"(?:\s+(?:right\s+now|now|sir|jarvis))*\??$", re.IGNORECASE)
# Words after "don't watch" that are not app names.
_NOT_APPS = frozenset({"this", "that", "it", "me", "my", "the", "screen",
                       "screens", "anything", "everything", "out", "tv",
                       "the screen", "my screen", "the screens", "my screens",
                       "this window", "that window"})


def screen_memory_command(text) -> "dict | None":
    """{"op": "pause", "minutes": N|None} / {"op": "resume"} /
    {"op": "status"} / {"op": "exclude_this"} / {"op": "exclude_app",
    "app": name}, or None. Never raises."""
    try:
        s = clean(text)
        if not s:
            return None
        low = s.lower()
        if _WATCH_THIS_RE.match(low):
            return {"op": "exclude_this"}
        m = _WATCH_PAUSE_RE.match(low)
        if m:
            minutes = None
            if m.group("n"):
                amt = _amount(m.group("n"))
                if amt:
                    minutes = amt * (60.0 if m.group("u").startswith("h")
                                     else 1.0)
            return {"op": "pause", "minutes": minutes}
        if _WATCH_RESUME_RE.match(low):
            return {"op": "resume"}
        if _WATCH_STATUS_RE.match(low):
            return {"op": "status"}
        m = _WATCH_APP_RE.match(low)
        if m:
            app = " ".join(m.group("app").split())
            if app and app not in _NOT_APPS and not app.startswith(
                    ("this ", "that ", "my screen", "the screen")):
                return {"op": "exclude_app", "app": app}
    except Exception:
        return None
    return None


_FORGET_RE = re.compile(
    r"^(?:please\s+)?(?:forget|delete|erase|wipe|clear)\s+"
    r"(?:(?:what|everything|anything)\s+(?:you\s+)?(?:saw|have\s+seen|"
    r"recorded|watched)(?:\s+on\s+(?:my|the)\s+screens?)?\s*)?"
    r"(?:(?:from|in|for|over)\s+)?(?:the\s+)?"
    r"(?:(?:last|past)\s+(?P<n>\d+|a|an|one|two|three|five|ten|fifteen|"
    r"twenty|thirty|half|few|couple)?\s*(?:of\s+)?(?:an?\s+)?"
    r"(?P<u>minutes?|mins?|hours?)|(?P<today>today)|"
    r"(?P<all>everything(?:\s+you\s+(?:saw|have\s+seen|recorded))?"
    r"(?:\s+on\s+(?:my|the)\s+screens?)?|it\s+all|all\s+of\s+it))"
    r"(?:\s+(?:of\s+)?(?:screen|watching|what\s+you\s+saw|on\s+(?:my|the)\s+"
    r"screens?))?(?:\s+(?:please|sir|jarvis))*$", re.IGNORECASE)
_FORGET_SCREEN_WORD_RE = re.compile(
    r"\b(?:saw|seen|screen|screens|watched|recorded|watching)\b",
    re.IGNORECASE)


def forget_span(text) -> "dict | None":
    """{"seconds": N} / {"today": True} / {"all": True} for "forget the last
    hour / ten minutes / today / everything you saw on my screen", else
    None. A bare "forget the last hour" (no screen word) is NOT claimed:
    forget_last_hour (memory) owns it. Never raises."""
    try:
        s = clean(text)
        if not s or not _FORGET_SCREEN_WORD_RE.search(s):
            return None
        m = _FORGET_RE.match(s.lower())
        if not m:
            return None
        if m.group("all"):
            return {"all": True}
        if m.group("today"):
            return {"today": True}
        unit = m.group("u") or ""
        amt = _amount(m.group("n")) if m.group("n") else 1.0
        if not amt:
            return None
        secs = amt * (3600.0 if unit.startswith("h") else 60.0)
        return {"seconds": secs}
    except Exception:
        return None


# ── "tell Claude ..." (developer notes) ─────────────────────────────────
_CLAUDE_NOTE_RE = re.compile(
    r"\b(?:tell|let|remind|ask|have|get|message|text|ping)\s+claude\b"
    r"(?:\s+know)?\s*(?:,\s*)?(?P<rest>.*)$", re.IGNORECASE)
_CLAUDE_NOTE_DIRECT_RE = re.compile(
    r"\b(?:(?:leave|write|make|send|give)\s+(?:claude\s+)?(?:a\s+)?"
    r"(?:note|message)(?:\s+(?:for|to)\s+claude)?|"
    r"(?:note|message)\s+(?:for|to)\s+claude)\b[\s:,]*(?P<rest>.*)$",
    re.IGNORECASE)
_TASK_VERB_RE = re.compile(
    r"\b(?:research|fix|fixes|fixing|look\s+into|work\s+on|improve|change|add|"
    r"make|build|learn|study|investigate|figure\s+out|debug|update|check|"
    r"review|watch|see|test|teach|implement|redo|rewrite|tune|train|"
    r"remember|note|know)\b", re.IGNORECASE)
_ABOUT_JARVIS_RE = re.compile(r"\b(?:you|your|yourself|he|him|his|jarvis)\b",
                              re.IGNORECASE)
_CLAUDE_EXCLUDE_RE = re.compile(
    r"\b(?:credits?|balance|usage|cost|costs|spend|spent|billing|bill)\b"
    r"|\b(?:switch\s+(?:back\s+)?to|use|using|open|launch|start)\s+claude\b"
    r"|\bclaude\s+code\b", re.IGNORECASE)
_ASK_QUESTION_RE = re.compile(
    r"^\s*ask\s+claude\s+(?:what|who|why|how|when|where|which|is|are|does|"
    r"do|can|could|would|should|if|whether)\b", re.IGNORECASE)


# What a note for the DEVELOPER is about: JARVIS himself or his workings
# (review 2026-10-05: "ask Claude to make me a workout plan" and "have Claude
# review my essay" are requests for the cloud brain, not developer notes).
_DEV_TOPIC_RE = re.compile(
    r"\b(?:you|your|yourself|he|him|his|jarvis|bugs?|code|coding|features?|"
    r"screen\s+vision|vision|clicks?|clicking|voice|wake\s+word|dispatcher|"
    r"router|routing|prompts?|release|version|builds?|tests?|crash(?:es|ed)?|"
    r"logs?|settings?|patch|skills?|memory|transcri\w*|microphone|mic|"
    r"camera|kinect|hud|tray|latency|lag|slow|glitch\w*|broken|"
    r"not\s+working|working\s+properly)\b", re.IGNORECASE)


def claude_note(text) -> "str | None":
    """The note for the developer (Claude) in "tell Claude to research the
    screen vision", "let Claude know your clicking is off", "leave Claude a
    note: ...", else None. The note must be about JARVIS or his workings
    (a "tell / let ... know" with a task or about him; "ask / have / get
    Claude to ..." only about him). Not for Claude's credits / cost, "use /
    switch to / open Claude", or "ask Claude <question>?" (a question for
    the cloud brain). Never raises."""
    try:
        s = clean(text)
        if not s or _CLAUDE_EXCLUDE_RE.search(s) or _ASK_QUESTION_RE.match(s):
            return None
        m = _CLAUDE_NOTE_DIRECT_RE.search(s)
        if m:
            rest = m.group("rest").strip(" :,.")
            return rest or None
        m = _CLAUDE_NOTE_RE.search(s)
        if not m:
            return None
        rest = m.group("rest").strip(" :,.")
        rest = re.sub(r"^(?:to|that|about)\s+", "", rest, flags=re.IGNORECASE)
        rest = re.sub(r"^(?:go\s+ahead\s+and\s+|go\s+and\s+|please\s+)+", "",
                      rest, flags=re.IGNORECASE)
        if not rest:
            return None
        verb = (m.group(0).split() or [""])[0].lower()
        if verb in ("ask", "have", "get"):
            # Claude asked to DO something: a developer note only when it
            # is about JARVIS or his workings.
            return rest if _DEV_TOPIC_RE.search(rest) else None
        if _TASK_VERB_RE.search(rest) or _ABOUT_JARVIS_RE.search(rest):
            return rest
    except Exception:
        return None
    return None
