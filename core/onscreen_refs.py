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
    "rewrite_referent", "youtube_play_query", "dev_todo_note",
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


def pending_choice_answer(text, options, *, allow_yes: bool = False,
                          hard_yes: bool = False):
    """The 0-based index of the option ``text`` picks among ``options``
    (dicts with "label" and "rect" [x, y, w, h], in the order JARVIS named
    them), else None. "the second one", "number two", "the top one", "the
    MrBeast one" (a phrase unique to one option), and - only when
    ``allow_yes`` and exactly one option was offered - "yes". ``hard_yes``
    (core.send_guard's hard stop before send / pay / delete ...): ONLY an
    explicit yes ("yes", "yes, send it", "go ahead", "do it") answers it -
    not "please", "right", "that one" or a bare "one". Never raises."""
    try:
        s = clean(text).lower()
        opts = list(options or ())
        if not s or not opts:
            return None
        if hard_yes:
            from core import send_guard as _sg
            return 0 if (allow_yes and len(opts) == 1
                         and _sg.is_yes_answer(s)) else None
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
# Screen COMMENTS (core.screen_remarks): "stop commenting", "start commenting",
# "comment less / more". Whole-utterance only.
_COMMENT_THING = (r"(?:\s+(?:on|about)\s+(?:what\s+i(?:'m|\s+am)?\s+watch(?:ing)?|"
                  r"what\s+i\s+watch|my\s+(?:screens?|videos?|shows?)|"
                  r"the\s+(?:screens?|videos?)|videos?|it|that|things))?")
_COMMENT_TAIL = r"(?:\s+(?:please|sir|jarvis|for\s+now|now|again))*$"
_REMARKS_OFF_RE = re.compile(
    r"^(?:(?:please\s+)?(?:stop|quit|cease)\s+(?:the\s+)?(?:commenting|"
    r"making\s+comments|commentary|remarking|screen\s+(?:comments?|remarks?))"
    + _COMMENT_THING + r"|(?:don'?t|do\s+not)\s+comment"
    r"(?:\s+(?:on|about)\s+(?:what\s+i(?:'m|\s+am)?\s+watch(?:ing)?|"
    r"my\s+(?:screens?|videos?|shows?)|the\s+(?:screens?|videos?)|videos?))?|"
    r"no\s+more\s+(?:screen\s+)?(?:comments?|commentary|remarks?)"
    r"(?:\s+(?:on|about)\s+(?:my\s+(?:screens?|videos?)|what\s+i\s+watch))?|"
    r"turn\s+off\s+(?:the\s+)?screen\s+(?:comments?|remarks?|commentary))"
    + _COMMENT_TAIL, re.IGNORECASE)
_REMARKS_ON_RE = re.compile(
    r"^(?:(?:please\s+)?(?:start|resume|begin|go\s+back\s+to|keep)\s+"
    r"(?:the\s+)?(?:commenting|making\s+comments|commentary|remarking|"
    r"screen\s+(?:comments?|remarks?))" + _COMMENT_THING + r"|"
    r"you\s+can\s+(?:comment|remark)(?:\s+(?:on|about)\s+(?:my\s+(?:screens?|"
    r"videos?)|what\s+i(?:'m|\s+am)?\s+watch(?:ing)?))?(?:\s+again)?|"
    r"comment\s+(?:on|about)\s+what\s+i(?:'m|\s+am)?\s+watch(?:ing)?|"
    r"turn\s+on\s+(?:the\s+)?screen\s+(?:comments?|remarks?|commentary))"
    + _COMMENT_TAIL, re.IGNORECASE)
_REMARKS_RATE_RE = re.compile(
    r"^(?:(?:comment|remark)\s+(?P<d>less|more)(?:\s+often)?|"
    r"(?P<f>fewer|more)\s+(?:screen\s+)?(?:comments|remarks)|"
    r"(?:stop\s+commenting|stop\s+remarking)\s+so\s+(?:much|often))"
    + _COMMENT_TAIL, re.IGNORECASE)
_REMARKS_STATUS_RE = re.compile(
    r"^(?:are\s+you\s+(?:still\s+)?commenting(?:\s+on\s+(?:my|the)\s+"
    r"(?:screens?|videos?))?|screen\s+(?:comments?|remarks?)\s+status)"
    r"(?:\s+(?:right\s+now|now|sir|jarvis))*\??$", re.IGNORECASE)
# Words after "don't watch" that are not app names.
_NOT_APPS = frozenset({"this", "that", "it", "me", "my", "the", "screen",
                       "screens", "anything", "everything", "out", "tv",
                       "the screen", "my screen", "the screens", "my screens",
                       "this window", "that window"})


def screen_memory_command(text) -> "dict | None":
    """{"op": "pause", "minutes": N|None} / {"op": "resume"} /
    {"op": "status"} / {"op": "exclude_this"} / {"op": "exclude_app",
    "app": name} / {"op": "remarks_off"} / {"op": "remarks_on"} /
    {"op": "remarks_less"} / {"op": "remarks_more"} / {"op": "remarks_status"},
    or None. Never raises."""
    try:
        s = clean(text)
        if not s:
            return None
        low = s.lower()
        if _REMARKS_OFF_RE.match(low):
            return {"op": "remarks_off"}
        if _REMARKS_ON_RE.match(low):
            return {"op": "remarks_on"}
        m = _REMARKS_RATE_RE.match(low)
        if m:
            d = (m.group("d") or m.group("f") or "less").lower()
            return {"op": "remarks_more" if d == "more" else "remarks_less"}
        if _REMARKS_STATUS_RE.match(low):
            return {"op": "remarks_status"}
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


# ── "open it" with nothing for "it" to be (v3.0.6) ──────────────────────
# Live 2026-10-28 17:25:16: "Jarvis, can you open it up so I can help you?"
# -> [ACTION: open_app, PowerShell] -> "launched PowerShell". Nothing he said
# and nothing JARVIS had just said named PowerShell (the turn before was the
# server status line); the model filled the bare "it" with an app of its own.
# A WHOLE "open it / that (up)" turn whose app is named neither in his words
# nor in the last few lines of the conversation is a question back.
_OPEN_PRONOUN_RE = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?(?:go\s+ahead\s+"
    r"and\s+|just\s+)?(?:open|launch|start|run|pull|bring|fire)\s+"
    r"(?:it|that|this|them|those)\s*(?:back\s+)?(?:up\s*)?"
    r"(?:(?:,\s*)?(?:so|and)\s+(?:that\s+)?(?:i|we)\s+.*"
    r"|for\s+me|please|now|again|real\s+quick|sir|jarvis|[\s,])*$",
    re.IGNORECASE)
_APP_FILLER = frozenset({"the", "app", "application", "program", "a", "an",
                         "my", "window", "exe"})
OPEN_WHAT_LINE = "Open what, sir?"


_REPLY_TOKEN_RE = re.compile(r"\[\s*(?:ACTION|intent)\s*:[^\]]*\]", re.IGNORECASE)


def strip_reply_tokens(text) -> str:
    """``text`` without its [ACTION: ...] / [intent:...] tokens. Pure; never
    raises."""
    try:
        return re.sub(r"\s+", " ", _REPLY_TOKEN_RE.sub(" ", str(text or ""))
                      ).strip()
    except Exception:
        return ""


def history_through_latest_user(hist) -> list:
    """The entries of ``hist`` (conversation_history) up to and INCLUDING the
    latest user entry; the assistant reply of the turn in progress is left
    out. Live 2026-10-28 review: _call_llm appends the brain's reply to
    conversation_history BEFORE parse_and_run_actions runs the action, so a
    handler that read the last lines of history saw its own
    "[ACTION: open_app, PowerShell]" / "[ACTION: set_timer, ... | check the
    oven]" and took the brain's invention for something the owner had said.
    No user entry at all: the list unchanged. Pure; never raises."""
    try:
        if not isinstance(hist, list):
            return []
        for i in range(len(hist) - 1, -1, -1):
            m = hist[i]
            if isinstance(m, dict) and m.get("role") == "user":
                return list(hist[:i + 1])
        return list(hist)
    except Exception:
        return []


def recent_context(hist, n: int = 5) -> str:
    """The last ``n`` conversation lines through the latest user entry (see
    history_through_latest_user), action / intent tokens stripped. Pure;
    never raises."""
    try:
        kept = history_through_latest_user(hist)[-n:]
        return " ".join(
            strip_reply_tokens(m["content"]) for m in kept
            if isinstance(m, dict) and isinstance(m.get("content"), str))
    except Exception:
        return ""


def _compact(text) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def unresolved_open(said, name, recent="") -> "str | None":
    """``OPEN_WHAT_LINE`` when ``said`` is WHOLLY "open it (up) ..." (a bare
    pronoun for the object) and ``name`` - the app the brain chose - is
    named neither in ``said`` nor in ``recent`` (the last lines of the
    conversation, his and JARVIS's), else None. "Shall I open Spotify?" /
    "yes, open it" keeps Spotify; "open it up so I can help you" after a
    server status line gets "Open what, sir?" instead of PowerShell. Pure;
    never raises."""
    try:
        s = clean(said)
        if not s or not name or not _OPEN_PRONOUN_RE.match(s):
            return None
        words = [w for w in re.findall(r"[a-z0-9]+", str(name).lower())
                 if w not in _APP_FILLER]
        if not words:
            return None
        context = f"{s} {recent or ''}"
        ctx_words = set(re.findall(r"[a-z0-9]+", context.lower()))
        ctx_compact = _compact(context)
        if _compact(" ".join(words)) in ctx_compact:
            return None
        if any(len(w) >= 3 and w in ctx_words for w in words):
            return None
        return OPEN_WHAT_LINE
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
# "...  Pass that over to Claude as well." (live 2026-10-18 12:09: "I want you
# to be able to do all this stuff that I'm saying. Pass that over to Claude as
# well." got "I'll ensure the context is fully integrated" and NOTHING was
# noted). The note is what came BEFORE the hand-off; the note records the
# session-log position, so "that" can be read back in context.
# Live 2026-10-26 20:07: "Jarvis passed it over to Claude. It already knows
# how." (Whisper's "passed" for "pass", and a sentence after it) ran a
# web_search instead: the past-tense verbs and a trailing sentence count too.
# Live 2026-10-28 15:58: "No, just pass over to Clog for me." (Parakeet's
# "Clog" for "Claude", no object pronoun, a "for me" tail) matched nothing:
# the brain sent [ACTION: browser_task, pass over to Clog] and the browser
# agent searched DuckDuckGo for "Clog" for ~47 s. The object is optional now,
# a courtesy tail (for me / please / now / as well / too) may follow, and
# the STT's own mishearings of "Claude" count, but ONLY here, as the last
# word of the hand-off frame. Each was heard for "Claude" in the owner's
# logs (2026-10-02..10-28: clog x1 in this very hand-off, claw x2, clawed x2,
# claud x1). NOT "cloud": no log has Claude heard that way, and "send it /
# upload that / back it up to the cloud" (Nextcloud, iCloud) is his real
# meaning. Not "clogged" either (a real word for a 3D-printer nozzle; its one
# sighting was a window name, not a hand-off). "The nozzle is about to clog"
# / "clear the clog" have no hand-off verb, so they never reach the frame.
_CLAUDE_HEARD_AS = r"(?:claude|clog|claw|clawed|claud)"
_CLAUDE_PASS_RE = re.compile(
    r"\b(?:pass|passed|send|sent|forward|forwarded|hand|handed|give|gave|"
    r"relay|relayed)\s+"
    r"(?:(?P<obj>that|this|it|them|those|these|everything|"
    r"all\s+(?:of\s+)?(?:that|this))\s+)?"
    r"(?:over\s+|along\s+|on\s+)?to\s+" + _CLAUDE_HEARD_AS + r"\b"
    r"(?:[\s,]+(?:as\s+well|too|also|for\s+me|please|now|right\s+now|"
    r"instead|then|sir|jarvis))*"
    r"(?:[\s.!?]*$|[.!?]\s+.*$)", re.IGNORECASE)
_PASS_REFERS_BACK = ("what I asked just before this (the session log at this "
                     "note has the conversation)")
# What may stand before the hand-off and still say nothing of its own: "No,
# just pass over to Claude" is the hand-off of the PREVIOUS request, not a
# note reading "No, just" (2026-10-28).
_PASS_FILLER_RE = re.compile(
    r"^(?:(?:no|nope|nah|yes|yeah|yep|ok|okay|actually|just|instead|well|um|"
    r"uh|so|and|or|then|now|please|sir|jarvis|hey|rather|also|can\s+you|"
    r"could\s+you|would\s+you|will\s+you|go\s+ahead\s+and)[\s,.;:!?-]*)*$",
    re.IGNORECASE)
# "What did you send to Claude?" / "Have you passed it to Claude?" ask about a
# hand-off; they are not one. Review 2026-10-28: once the object went optional,
# "What gets sent to Claude?", "Is it sent to Claude?", "Do you send everything
# to Claude?" and "Can I send to Claude?" became notes reading "What gets" /
# "Is it" / "Do you" / "Can I". A passive word right before the verb, or
# do / should / may + a pronoun, asks too. "Can / could / will / would YOU
# pass it to Claude" stays a request.
_PASS_ASKED_RE = re.compile(
    r"(?:\b(?:did|didn't|have|haven't|has|hasn't|had|was|were|what|why|who|"
    r"how|when|where|do|does|don't|doesn't|should|shall|may|might)\s+"
    r"(?:you|i|we|he|it|jarvis)"
    r"|\b(?:can|could|will|would)\s+(?:i|we|he|it|jarvis)"
    r"|\b(?:is|was|were|are|be|been|being|get|gets|got))"
    r"(?:\s+(?:just|already|ever|really|actually|all|always))?\s*$",
    re.IGNORECASE)
# With no object ("pass over to Claude") the hand-off is claimed only when it
# opens the turn, follows filler, or follows a finished clause or "or (just)".
# Review 2026-10-28: "I'm looking forward to Claude." noted "I'm looking".
_PASS_CLAUSE_END_RE = re.compile(
    r"(?:[.!?]|\b(?:or|and)(?:\s+just)?)[\s,]*$", re.IGNORECASE)
# v3.0.8 (PASS_FUTURE_REFERS_BACK): "I'll hand it to Claude" made a note
# reading "I'll" (known since 3.0.5). A first-person future / "let me" right
# before the hand-off says nothing of its own: it is the hand-off of the
# previous request, or of the task said before it ("find the manual, or I'll
# pass it to Claude").
_PASS_FUTURE_SUBJ_RE = re.compile(
    r"(?:^|(?<=[\s,;:.!?-]))"
    r"(?:i[’']ll|i\s+will|i\s+shall|i[’']m\s+(?:going\s+to|gonna)|"
    r"i\s+am\s+(?:going\s+to|gonna)|let\s+me|let[’']s|let\s+us|we[’']ll|"
    r"we\s+will|we[’']re\s+(?:going\s+to|gonna))"
    r"(?:\s+(?:just|probably|then|now|also|go\s+ahead\s+and|go\s+and))*"
    r"[\s,]*$", re.IGNORECASE)
# The clause before a future hand-off is a task when it ended in "or" / "and"
# ("find the manual, or I'll pass it to Claude"), opens with an imperative
# verb, or holds one of the module's task verbs. "know" / "see" / "note" are
# left out of the last test: "I don't know, I'll hand it to Claude" is an
# aside (review 2026-10-28).
_PASS_LEAD_CONNECT_RE = re.compile(r"\b(?:or|and)(?:\s+just)?[\s,]*$",
                                   re.IGNORECASE)
_PASS_IMPERATIVES = (
    r"find|look|get|put|send|write|call|order|buy|book|schedule|create|open|"
    r"search|set|turn|read|draft|plan|ask|tell|show|give|take|pull|run|"
    r"install|remove|delete|pick|bring|email|text|message|make|fix|add|"
    r"research|improve|change|build|learn|study|investigate|figure|debug|"
    r"update|check|review|watch|test|teach|implement|redo|rewrite|tune|train|"
    r"remember|work|clean|move|sort|file|print|download|upload|save|finish|"
    r"start|cancel|reply|respond|draw|design|compare|summarize|"
    r"translate|list|track|monitor")
_PASS_TASK_START_RE = re.compile(
    r"^(?:(?:please|just|go\s+ahead\s+and|can\s+you|could\s+you|would\s+you|"
    r"will\s+you)[\s,]+)*(?:" + _PASS_IMPERATIVES + r")\s+\w", re.IGNORECASE)
_PASS_TASK_ANY_RE = re.compile(
    r"\b(?:research|fix|fixes|fixing|look\s+into|work\s+on|improve|change|add|"
    r"make|build|learn|study|investigate|figure\s+out|debug|update|check|"
    r"review|watch|test|teach|implement|redo|rewrite|tune|train|remember)\b",
    re.IGNORECASE)


def _pass_future_on() -> bool:
    """PASS_FUTURE_REFERS_BACK (core/config.py), read per call. Never
    raises."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, "PASS_FUTURE_REFERS_BACK", True))
    except Exception:
        return True
# "Put my hours in for me, or tell Claude to DO IT" (live 2026-10-26 20:06):
# the task is the clause before the hand-off.
_CLAUDE_DO_IT_RE = re.compile(
    r"^(?:do|handle|take\s+care\s+of|deal\s+with|sort\s+out)\s+"
    r"(?:it|that|this|them)\b[\s.!?]*$", re.IGNORECASE)
# "and then" / "so then" go together (review 2026-10-28: "Research the bug
# and then pass it to Claude" noted "Research the bug and").
_BEFORE_TELL_RE = re.compile(
    r"[\s,;:-]*(?:\b(?:or|and|so)\s+)?(?:\bthen\s+)?(?:just\s+)?$",
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


# ── Developer to-dos WITHOUT the word "Claude" (v3.0.6) ─────────────────
# Live 2026-10-28 17:26: "We need to give the robot more voice lines." (said
# with the owner's robot's name) had no "Claude" in it, so no route claimed
# it; the brain sent the robot's speak action with "I am ready for more
# dialogue." and the robot said a line nobody asked for. The same words at 17:57 got note_for_claude from
# the brain - a coin toss. A "we need to / we should" plan for JARVIS or the
# robot to GAIN something is work for the developer, never a command to run
# now. Deliberately narrow; everything else stays the brain's:
#   * it must OPEN the turn (after the wake word / lead filler) - "do we need
#     to ...?" / "should we make him ...?" are questions;
#   * the subject is "we" (a plan); "you need to ..." only as "you need to
#     stop saying / repeating / calling me ..." (speech habits, which no
#     action changes - "stop playing music" / "stop commenting" are real
#     commands and never match);
#   * the object is JARVIS or the robot (you / yourself / jarvis / the
#     robot, or him / his only together with a feature). The owner's robot
#     is never named here: _ROBOT_WORD is "robot", "bot", a one-word
#     "<word>bot" device name (a doubled final t from the recogniser too)
#     or a two-word spoken name ("<word> bot", "<word>-bot") - since v3.0.8
#     not a common English word such as chatbot / abbot (see
#     _ROBOT_WORD_STRICT);
#   * and what he wants is a feature (voice lines, jokes, features,
#     smarter, able to ...) - "we need to give him more time", "we should
#     make you a playlist", "we need to turn off the lights" never match.
# "update / upgrade you" is NOT here: an update is a real action.
_DEV_SUBJECT = (r"^(?:(?:so|ok|okay|and|also|oh|hey|alright|right|now|"
                r"jarvis)[\s,]+)*we\s+(?:really\s+|still\s+|definitely\s+|"
                r"probably\s+)?(?:need\s+to|have\s+to|gotta|got\s+to|should|"
                r"ought\s+to|must)\s+")
_ROBOT_WORD_LOOSE = r"(?:[a-z]+(?:\s+|\s*-\s*))?[a-z]*bott?"
# v3.0.8 (DEV_TODO_ROBOT_STRICT): the loose word above also took "we need to
# fix the chatbot", "make the Talbot more reliable" and "the abbot needs more
# lines". Strict: "robot", "bot" on its own, a two-word "<word> bot" /
# "<word>-bot" (not "chat bot"), or a one-word "<word>bot" that is not a
# common English word; never a household "robot vacuum / mower ...". A
# doubled final t from the recogniser still counts.
_BOT_COMMON_WORDS = (r"(?:abbott?|sabot|turbot|jabot|talbott?|cabot|chatbot|"
                     r"chabot)")
_ROBOT_WORD_STRICT = (
    r"(?:(?!chat\b)[a-z]+(?:\s+|\s*-\s*)bott?\b|robot\b|bott?\b"
    r"|(?!" + _BOT_COMMON_WORDS + r"\b)[a-z]+bott?\b)"
    r"(?!\s+(?:vacuums?|mops?|mowers?|lawn\s*mowers?|cleaners?|toys?)\b)")
_ROBOT_WORD = _ROBOT_WORD_STRICT


def _dev_target(robot_word: str) -> str:
    return (r"(?P<tgt>you|yourself|jarvis|(?:the\s+|my\s+|our\s+)?"
            + robot_word + r"|him|his)")


_DEV_TARGET = _dev_target(_ROBOT_WORD)
_DEV_FEATURE = (
    r"(?:voice\s+lines?|lines|dialog(?:ue)?s?|phrases|sayings|catch\s*phrases|"
    r"jokes|sounds?|sound\s+effects|responses|replies|reactions|features?|"
    r"personality|abilit(?:y|ies)|skills?|commands|animations?|moves|"
    r"gestures|expressions|options?|settings?|memory|voices?|vocabulary|"
    r"smarter|faster|quicker|funnier|(?:a\s+)?(?:lot\s+)?better\s+at|"
    r"more\s+(?:natural|human|responsive|reliable|accurate|expressive|"
    r"talkative|useful)|able\s+to|(?:a\s+)?way\s+to|support\s+for)")
def _dev_todo_res(robot_word: str):
    """The (we-need-to, fix, needs) regexes for one robot-word pattern."""
    we = re.compile(
        _DEV_SUBJECT
        + r"(?:give|teach|add|make|program|build|let|improve|fix)\s+"
        + r"(?:\S+\s+){0,3}?" + _dev_target(robot_word)
        + r"\b(?P<rest>.*)$", re.IGNORECASE)
    fix = re.compile(
        _DEV_SUBJECT + r"(?:fix|improve)\s+(?:you|yourself|jarvis|"
        r"(?:the\s+|my\s+|our\s+)?" + robot_word + r")$", re.IGNORECASE)
    needs = re.compile(
        r"^(?:the\s+|my\s+|our\s+)?" + robot_word + r"\s+(?:really\s+)?"
        r"(?:needs|should\s+have|could\s+use)\s+(?P<rest>.*)$",
        re.IGNORECASE)
    return we, fix, needs


_DEV_TODO_WE_RE, _DEV_TODO_FIX_RE, _DEV_TODO_NEEDS_RE = _dev_todo_res(
    _ROBOT_WORD_STRICT)
_DEV_TODO_RES_LOOSE = _dev_todo_res(_ROBOT_WORD_LOOSE)
_DEV_TODO_STOP_RE = re.compile(
    r"^you\s+(?:really\s+)?(?:need\s+to|have\s+to|gotta|got\s+to|should|"
    r"must)\s+stop\s+(?:saying|repeating|calling\s+me)\b\s*\S", re.IGNORECASE)
_DEV_FEATURE_RE = re.compile(r"\b" + _DEV_FEATURE + r"\b", re.IGNORECASE)
# "him" / "his" is also a person ("give him more time to finish his skills
# test", "we need to give him a ride to his skills competition", "let him
# know about the settings", "add him to the group chat with more options"):
# the feature may sit ANYWHERE after "you" / "the robot", but after
# "him" / "his" it must come RIGHT after the target, optionally behind a
# quantity word ("make him smarter", "give him more voice lines").
_DEV_FEATURE_NEXT_RE = re.compile(
    r"^\s*(?:(?:more|new|better|some|extra|different|other|additional|"
    r"another|a\s+few|a\s+lot\s+more|lots\s+of|many\s+more)\s+)?"
    + _DEV_FEATURE + r"\b", re.IGNORECASE)
# A second clause in the same breath is a command he gave too ("you need to
# stop calling me sir and turn on the lights", "we need to make him able to
# see, and also turn on the lights"): the note would swallow it, so the turn
# stays the brain's. "and make him smarter" / "and stop saying X" are more
# developer to-dos and do not count.
_DEV_CMD_VERBS = (r"turn|switch|set|play|open|launch|close|dim|brighten|run|"
                  r"call|send|text|put|check|tell|show|search|look|find|"
                  r"pause|resume|mute|unmute|skip|lock|unlock|shut|read|"
                  r"remind|schedule|start")
_DEV_COMPOUND_RE = re.compile(
    r"(?:,\s*and\b|\band\s+also\b|\bthen\b|\balso\s+(?:" + _DEV_CMD_VERBS
    + r")\b|\band\s+(?:please\s+)?(?:" + _DEV_CMD_VERBS
    + r")\b|[,;.]\s*(?:and\s+)?(?:please\s+)?(?:" + _DEV_CMD_VERBS
    + r")\b)", re.IGNORECASE)


def _dev_todo_res_now():
    """The (we, fix, needs) regexes DEV_TODO_ROBOT_STRICT picks (core/
    config.py, read per call; default strict). Never raises."""
    try:
        from core import config as _cfg
        strict = bool(getattr(_cfg, "DEV_TODO_ROBOT_STRICT", True))
    except Exception:
        strict = True
    if strict:
        return _DEV_TODO_WE_RE, _DEV_TODO_FIX_RE, _DEV_TODO_NEEDS_RE
    return _DEV_TODO_RES_LOOSE


def _dev_todo_on() -> bool:
    """DEV_TODO_NOTES (core/config.py), read per call. Never raises."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, "DEV_TODO_NOTES", True))
    except Exception:
        return True


def dev_todo_note(text) -> "str | None":
    """The note when the WHOLE turn is a developer to-do for JARVIS / the
    robot that never says "Claude" ("We need to give the robot more voice
    lines", "we should make you smarter", "you need to stop saying 'indeed'"),
    else None. The note is his sentence as said (wake word and trailing
    punctuation off). See the block comment for how narrow it is. Never
    raises."""
    try:
        s = clean(text)
        if not s or "?" in s:
            return None
        we_re, fix_re, needs_re = _dev_todo_res_now()
        if _DEV_TODO_STOP_RE.match(s) or fix_re.match(s):
            return None if _DEV_COMPOUND_RE.search(s) else s
        m = we_re.match(s)
        if m:
            rest = m.group("rest")
            if _DEV_COMPOUND_RE.search(s):
                return None
            # The feature must be what he wants for JARVIS / the robot: it
            # follows the target ("give the robot more voice lines",
            # "make him smarter"). After him / his it follows IMMEDIATELY.
            if (m.group("tgt") or "").lower() in ("him", "his"):
                return s if _DEV_FEATURE_NEXT_RE.match(rest) else None
            if _DEV_FEATURE_RE.search(rest):
                return s
            return None
        m = needs_re.match(s)
        if (m and _DEV_FEATURE_RE.search(m.group("rest"))
                and not _DEV_COMPOUND_RE.search(s)):
            return s
    except Exception:
        return None
    return None


def claude_note(text) -> "str | None":
    """The note for the developer: a "Claude" note (_claude_named_note), or,
    with DEV_TODO_NOTES on, a developer to-do that never names him
    (dev_todo_note). Never raises."""
    note = _claude_named_note(text)
    if note is None and _dev_todo_on():
        note = dev_todo_note(text)
    return note


def _claude_named_note(text) -> "str | None":
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
        m = _CLAUDE_PASS_RE.search(s)
        if m and _PASS_ASKED_RE.search(s[:m.start()]):
            return None
        if m:
            before = s[:m.start()].strip(" ,.;:!?")
            before = re.sub(r"^(?:(?:hey|ok|okay)[\s,]+)?jarvis\b[\s,.:;!?-]*",
                            "", before, flags=re.IGNORECASE).strip()
            if _PASS_FILLER_RE.match(before):
                return _PASS_REFERS_BACK
            fm = _PASS_FUTURE_SUBJ_RE.search(before) if _pass_future_on() \
                else None
            if fm:
                lead = before[:fm.start()]
                task = _BEFORE_TELL_RE.sub("", lead + " ")
                task = task.strip(" ,.;:!?")
                if not task or _PASS_FILLER_RE.match(task):
                    return _PASS_REFERS_BACK
                # Only a clause that reads as a task is the note; an aside or
                # condition ("I think", "Wait", "If that fails") is not.
                if (_PASS_LEAD_CONNECT_RE.search(lead)
                        or _PASS_TASK_START_RE.match(task)
                        or _PASS_TASK_ANY_RE.search(task)):
                    return task
                return _PASS_REFERS_BACK
            if not m.group("obj") and not _PASS_CLAUSE_END_RE.search(
                    s[:m.start()]):
                return None
            # "Find the manual, or just pass it to Claude": the note is the
            # task, without its "or just".
            before = _BEFORE_TELL_RE.sub("", before + " ").strip(" ,.;:!?")
            return before or _PASS_REFERS_BACK
        m = _CLAUDE_NOTE_DIRECT_RE.search(s)
        if m:
            rest = m.group("rest").strip(" :,.")
            return rest or None
        m = _CLAUDE_NOTE_RE.search(s)
        if not m:
            return None
        raw = m.group("rest").strip(" :,.")
        # "tell Claude TO <x>" is an instruction for Claude whatever <x> is
        # (2026-10-26: "tell Claude to put my hours in" was no note because
        # hours are not about JARVIS). "tell Claude hi" still is not.
        told_to = bool(re.match(r"to\s+", raw, re.IGNORECASE))
        rest = re.sub(r"^(?:to|that|about)\s+", "", raw, flags=re.IGNORECASE)
        rest = re.sub(r"^(?:go\s+ahead\s+and\s+|go\s+and\s+|please\s+)+", "",
                      rest, flags=re.IGNORECASE)
        if not rest:
            return None
        verb = (m.group(0).split() or [""])[0].lower()
        if verb in ("ask", "have", "get"):
            # Claude asked to DO something: a developer note only when it
            # is about JARVIS or his workings.
            return rest if _DEV_TOPIC_RE.search(rest) else None
        if _CLAUDE_DO_IT_RE.match(rest):
            # "..., or tell Claude to do it": the task came first.
            before = _BEFORE_TELL_RE.sub("", s[:m.start()]).strip(" ,.;:!?")
            before = re.sub(r"^(?:(?:hey|ok|okay)[\s,]+)?jarvis\b[\s,.:;!?-]*",
                            "", before, flags=re.IGNORECASE).strip()
            before = re.sub(r"^(?:go\s+ahead\s+and\s+|please\s+)+", "", before,
                            flags=re.IGNORECASE).strip()
            return before or _PASS_REFERS_BACK
        if (told_to or _TASK_VERB_RE.search(rest)
                or _ABOUT_JARVIS_RE.search(rest)):
            return rest
    except Exception:
        return None
    return None
