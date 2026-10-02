"""core/streaming_search.py - the VERIFIED streaming-service search links.

WHY THIS EXISTS (2026-10-02)
============================
Live: asked to find a show on HBO Max, the brain wrote
``[ACTION: open_url, https://www.hbomax.com/search?q=<show>]``. That page does
not exist (HTTP 404: "Oops! Looks like this link isn't working."), so the turn
went on to read, search and click around an error page. The model guesses a
search URL from the shape of other sites; nothing checked the guess.

This module is the ONE table of how each video service really searches, each
entry checked against the live site, and the helpers that use it:

  * ``search_url`` / ``home_url``  - the verified link (None = no verified
    search: open the home page and SAY so, never guess);
  * ``fix_search_url``             - the open_url / open_on_monitor guard: a
    search-shaped URL on a known service's host that is not the verified
    pattern becomes the verified search for the same words (or the home page);
  * ``streaming_route``            - "find / play <title> on <service>" claimed
    before the brain (core.dispatcher's route style), so a dropped prompt
    section can no longer turn it into a guessed URL;
  * ``wall_kind`` / ``wall_line``  - the sign-in wall: a service page that
    says "Sign In" / "Log in" / "Oops ... isn't working" ends the turn with
    one plain sentence instead of clicks on a page that cannot play.

HOW EACH PATTERN WAS VERIFIED (2026-10-02, logged out, desktop Chrome UA)
========================================================================
A single 200 proves nothing on a site that serves every path, so each check is
DIFFERENTIAL: the search URL against a nonsense path on the same host.

  * HBO Max  play.hbomax.com/search?q=  -> 200 on the player host, while
    /search (no q), /searchzzz, /search/zzz and even /home redirect away to
    www.hbomax.com - the server recognises /search exactly when q is given.
    The guessed www.hbomax.com/search?q= is a 404 ("Oops"). play.max.com
    301-redirects to play.hbomax.com.
  * Netflix  www.netflix.com/search?q=  -> 302 to /login?nextpage=<the same
    search URL>; a nonsense path goes to /NotFound instead.
  * Hulu     www.hulu.com/search?q=     -> 302 to /tv?q= (the logged-out
    landing keeps q); a nonsense path is a 404.
  * Prime Video www.primevideo.com/search/ref=atv_nb_sr?phrase=  -> 200,
    title "Prime Video: Search", the query word on the page 253 times; a
    nonsense path is a 404.
  * YouTube  www.youtube.com/results?search_query=  -> 200 with videoIds.
  * Apple TV tv.apple.com/search?term=  -> 301 to /us/search?term=, 200; the
    query word on the page 63 times for a real show, 10 for a nonsense one
    (Apple renamed "Apple TV+" to "Apple TV" in 2025).
  * Disney+  NO verified pattern: /search, /search?q=, /browse/search and
    /en-us/search?q= are all 404 logged out, exactly like a nonsense path.
    A Disney+ search opens the home page and says so.

Re-check a pattern (and this date) before changing it. Pure stdlib; never
raises.
"""
from __future__ import annotations

import re
import urllib.parse
from typing import NamedTuple, Optional

# The date every pattern below was last checked against the live site.
VERIFIED_CHECKED = "2026-10-02"


class Service(NamedTuple):
    key: str            # the play_streaming / _STREAMING_SERVICES key
    name: str           # what JARVIS calls it out loud
    home: str
    search: Optional[str]   # verified search URL with {q}, or None
    search_host: str    # host the verified pattern lives on ("" = none)
    search_path: str    # its path ("/search"); a prefix match
    search_param: str   # the query parameter carrying the words
    hosts: tuple        # every host (suffix) that belongs to the service
    spoken: tuple       # spoken / typed names, lower-case


# Verified 2026-10-02 - see the module docstring for the evidence per row.
SERVICES: dict = {
    "max": Service(
        "max", "HBO Max", "https://play.hbomax.com",
        "https://play.hbomax.com/search?q={q}",
        "play.hbomax.com", "/search", "q",
        ("hbomax.com", "max.com"),
        ("hbo max", "hbomax", "hbo", "max", "hbo_max")),
    "netflix": Service(
        "netflix", "Netflix", "https://www.netflix.com",
        "https://www.netflix.com/search?q={q}",
        "www.netflix.com", "/search", "q",
        ("netflix.com",),
        ("netflix",)),
    "hulu": Service(
        "hulu", "Hulu", "https://www.hulu.com",
        "https://www.hulu.com/search?q={q}",
        "www.hulu.com", "/search", "q",
        ("hulu.com",),
        ("hulu",)),
    "prime_video": Service(
        "prime_video", "Prime Video", "https://www.primevideo.com",
        "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={q}&ie=UTF8",
        "www.primevideo.com", "/search", "phrase",
        ("primevideo.com",),
        ("amazon prime video", "prime video", "amazon prime", "amazon video",
         "primevideo", "prime_video", "prime", "amazon")),
    "youtube": Service(
        "youtube", "YouTube", "https://www.youtube.com",
        "https://www.youtube.com/results?search_query={q}",
        "www.youtube.com", "/results", "search_query",
        ("youtube.com", "youtu.be"),
        ("youtube", "you tube", "yt")),
    "apple_tv": Service(
        "apple_tv", "Apple TV", "https://tv.apple.com",
        "https://tv.apple.com/search?term={q}",
        "tv.apple.com", "/search", "term",
        ("tv.apple.com",),
        ("apple tv plus", "apple tv+", "apple tv", "appletv", "apple_tv",
         "apple_tv_plus")),
    "disney_plus": Service(
        "disney_plus", "Disney+", "https://www.disneyplus.com",
        None, "", "", "",
        ("disneyplus.com",),
        ("disney plus", "disney+", "disneyplus", "disney_plus", "disney")),
}

# Apple TV's storefront prefix (/us/search): the same verified route.
_STOREFRONT_PATH_RE = re.compile(r"^/[a-z]{2}(?=/)")


def _norm_name(name) -> str:
    s = " ".join(str(name or "").lower().replace("_", " ").split())
    s = re.sub(r"\s*\+", "+", s)          # "disney +" is "disney+"
    s = re.sub(r"^(?:the|my)\s+", "", s)
    s = re.sub(r"\s+(?:app|website|site|service)$", "", s)
    return s.strip(" .,!?")


def canon_service(name) -> Optional[str]:
    """The SERVICES key for a spoken or typed service name ("HBO Max",
    "disney plus", "Prime"), else None. Never raises."""
    try:
        s = _norm_name(name)
        if not s:
            return None
        for key, svc in SERVICES.items():
            if s == key.replace("_", " ") or s in (
                    _norm_name(x) for x in svc.spoken):
                return key
    except Exception:
        return None
    return None


def service(key) -> Optional[Service]:
    """The Service row for ``key`` (or a spoken name), else None."""
    k = key if key in SERVICES else canon_service(key)
    return SERVICES.get(k) if k else None


def service_name(key) -> str:
    svc = service(key)
    return svc.name if svc else str(key or "")


def home_url(key) -> Optional[str]:
    svc = service(key)
    return svc.home if svc else None


def search_url(key, query) -> Optional[str]:
    """The VERIFIED search link for ``query`` on service ``key``; None when
    the service has no verified pattern (open its home page instead) or the
    query is empty. Never raises."""
    try:
        svc = service(key)
        q = " ".join(str(query or "").split())
        if svc is None or not svc.search or not q:
            return None
        return svc.search.format(q=urllib.parse.quote(q, safe=""))
    except Exception:
        return None


def _parse(url):
    s = str(url or "").strip()
    if not s:
        return None
    if not re.match(r"^[a-z][a-z0-9+.-]*://", s, re.IGNORECASE):
        s = "https://" + s
    try:
        return urllib.parse.urlsplit(s)
    except Exception:
        return None


def service_for_url(url) -> Optional[str]:
    """The SERVICES key whose host ``url`` is on, else None. Never raises."""
    try:
        p = _parse(url)
        host = (p.hostname or "").lower() if p else ""
        if not host:
            return None
        for key, svc in SERVICES.items():
            for h in svc.hosts:
                if host == h or host.endswith("." + h):
                    return key
    except Exception:
        return None
    return None


# Query parameters a guessed search link carries its words in.
_QUERY_PARAMS = ("q", "query", "term", "phrase", "search_query", "search",
                 "k", "keyword", "keywords", "searchterm", "text")


def _query_words(p) -> str:
    try:
        params = urllib.parse.parse_qs(p.query, keep_blank_values=False)
    except Exception:
        params = {}
    low = {k.lower(): v for k, v in params.items()}
    for name in _QUERY_PARAMS:
        vals = low.get(name)
        if vals and vals[0].strip():
            return " ".join(vals[0].split())
    # A path-style search: /search/<words>
    m = re.search(r"/search/([^/?#]+)$", p.path or "", re.IGNORECASE)
    if m and not m.group(1).lower().startswith("ref="):
        return " ".join(urllib.parse.unquote_plus(m.group(1)).split())
    return ""


def _looks_like_search(p) -> bool:
    path = (p.path or "").lower()
    if "search" in path or path.rstrip("/").endswith("/results"):
        return True
    return bool(_query_words(p))


def _is_verified_search(p, svc: Service) -> bool:
    if not svc.search:
        return False
    host = (p.hostname or "").lower()
    if host != svc.search_host and host != svc.search_host.replace("www.", "", 1):
        return False
    path = _STOREFRONT_PATH_RE.sub("", p.path or "")
    if not path.lower().startswith(svc.search_path):
        return False
    try:
        params = urllib.parse.parse_qs(p.query)
    except Exception:
        return False
    return bool((params.get(svc.search_param) or [""])[0].strip())


class UrlFix(NamedTuple):
    url: str     # the URL to open (unchanged when there is nothing to fix)
    note: str    # "" when unchanged; else what was swapped, for the result
    service: str  # the service key, or ""


def fix_search_url(url) -> UrlFix:
    """Guard for a URL the brain wrote (open_url / open_on_monitor).

    A search-shaped URL on a known service's host that is NOT that service's
    verified pattern becomes the verified search for the same words - or the
    service's home page when it has no verified search (Disney+), with a note
    saying so. A verified search, a title page, a home page and any other
    site come back unchanged. A bare service name ("HBO Max") becomes its
    home page. Never raises."""
    try:
        raw = str(url or "").strip()
        key = canon_service(raw) if "/" not in raw and "." not in raw else None
        if key:
            svc = SERVICES[key]
            return UrlFix(svc.home, f"opened the {svc.name} home page", key)
        p = _parse(raw)
        key = service_for_url(raw) if p else None
        if not key:
            return UrlFix(raw, "", "")
        svc = SERVICES[key]
        if not _looks_like_search(p) or _is_verified_search(p, svc):
            return UrlFix(raw, "", key)
        words = _query_words(p)
        fixed = search_url(key, words) if words else None
        if fixed:
            return UrlFix(fixed, (
                f"{raw} is not a real {svc.name} link, so I opened the "
                f"verified {svc.name} search for '{words}' instead"), key)
        tail = f" - search for '{words}' there" if words else ""
        return UrlFix(svc.home, (
            f"there is no verified {svc.name} search link, so I opened the "
            f"{svc.name} home page instead of {raw}{tail}"), key)
    except Exception:
        return UrlFix(str(url or ""), "", "")


# ── "find / play <title> on <service>" (claimed before the brain) ────────────
_WAKE_LEAD_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay)[\s,]+)?jarvis\b[\s,.:;!?-]*", re.IGNORECASE)
_LEAD_RE = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\s+|please\s+|go\s+ahead\s+and\s+|"
    r"i\s+(?:need|want)\s+you\s+to\s+|let'?s\s+|now\s+)+", re.IGNORECASE)
# Verbs that mean "find it" (search) and "play it".
_FIND_VERBS = ("find", "search for", "search", "look up", "look for",
               "pull up", "bring up", "locate")
_PLAY_VERBS = ("play", "plays", "put on", "watch", "stream", "start")


def _alt(words) -> str:
    return "|".join(r"\s+".join(re.escape(w) for w in v.split())
                    for v in sorted(words, key=len, reverse=True))


_SPOKEN_TO_KEY = {}
for _k, _svc in SERVICES.items():
    for _w in _svc.spoken:
        if "_" not in _w:
            _SPOKEN_TO_KEY[_w] = _k
_SPOKEN_TO_KEY.pop("yt", None)   # too short to trust in free speech
_ROUTE_RE = re.compile(
    r"^(?P<verb>" + _alt(_FIND_VERBS + _PLAY_VERBS) + r")\s+(?P<title>.+?)\s+"
    r"(?:on|in|from)\s+(?:the\s+)?(?P<svc>" + _alt(_SPOKEN_TO_KEY) + r")"
    r"(?:\s+(?:app|website|site))?(?P<tail>(?:[\s,]+.*)?)$", re.IGNORECASE)
# Words a trailing clause may hold: "and start playing", "and resume it for
# me", "please". Anything else ("... on the left monitor", "and dim the
# lights") is the brain's turn.
_TAIL_WORDS = frozenset({
    "and", "then", "start", "starting", "begin", "resume", "resuming",
    "play", "playing", "it", "that", "the", "show", "episode", "first",
    "next", "watching", "streaming", "for", "me", "please", "now", "go",
    "ahead", "up", "again", "just", "from", "where", "i", "left", "off",
    "right", "away", "sir",
})
_TAIL_PLAY_WORDS = frozenset({"play", "playing", "start", "starting",
                              "begin", "resume", "resuming", "watching",
                              "streaming"})
# Titles that only make sense against earlier context: the brain, which sees
# the conversation, resolves those.
_VAGUE_TITLES = frozenset({
    "it", "that", "this", "them", "those", "these", "one", "something",
    "anything", "whatever", "something good", "a show", "the show",
    "a movie", "the movie", "a film", "the film", "something to watch",
    "that show", "this show", "that movie", "this movie", "the next episode",
    "the episode", "an episode", "the first episode",
})
# Sentences that may stand beside the command: "Try again." / "Okay."
_FILLER_SENTENCE_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all\s+right|right|please|now|so|well|"
    r"actually|sorry|no|yes|yeah|again|try\s+(?:it\s+)?again|"
    r"one\s+more\s+time|jarvis)[\s,]*)+$", re.IGNORECASE)
_ROUTE_MAX_TITLE = 120


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text) if s.strip()]


def streaming_route(utterance, *, allow_play: bool = True,
                    allow_find: bool = True) -> Optional[str]:
    """The action token for a whole "find / play <title> on <service>"
    request, else None. Never raises.

      * a play verb ("play", "watch", "put on") or a play clause after the
        service ("... and start playing") ->
        ``[ACTION: play_streaming, <service>|<title>]``;
      * a find verb alone ("find", "look up", "pull up") ->
        ``[ACTION: streaming_search, <service>|<title>]``.

    Only the SERVICES table's services (music stays with its own routes), never
    "play ... on YouTube" (core.dispatcher.youtube_play_route owns that), and
    nothing with a vague title ("play it on Netflix"), an extra clause ("... on
    the left monitor") or a second command. A leading wake word, a polite
    lead-in and a filler sentence ("Try again.") are allowed."""
    try:
        if not isinstance(utterance, str) or not utterance.strip():
            return None
        text = _WAKE_LEAD_RE.sub("", utterance, count=1)
        hits = []
        for sent in _sentences(text):
            body = _LEAD_RE.sub("", sent.strip(" \t.,!?;:")).strip()
            body = _WAKE_LEAD_RE.sub("", body, count=1).strip(" .,!?;:")
            if not body or _FILLER_SENTENCE_RE.match(body):
                continue
            hits.append(body)
        if len(hits) != 1:
            return None
        m = _ROUTE_RE.match(" ".join(hits[0].split()))
        if not m:
            return None
        verb = " ".join(m.group("verb").lower().split())
        title = " ".join(m.group("title").split()).strip(" ,'\"")
        key = _SPOKEN_TO_KEY.get(" ".join(m.group("svc").lower().split()))
        tail = re.findall(r"[a-z']+", (m.group("tail") or "").lower())
        if (not key or not title or len(title) > _ROUTE_MAX_TITLE
                or title.lower() in _VAGUE_TITLES
                or any(c in title for c in "[]|\r\n")):
            return None
        if any(w not in _TAIL_WORDS for w in tail):
            return None
        play = verb in _PLAY_VERBS or bool(_TAIL_PLAY_WORDS & set(tail))
        if key == "youtube" and play:
            return None
        if play and allow_play:
            return f"[ACTION: play_streaming, {key}|{title}]"
        if not play and allow_find:
            return f"[ACTION: streaming_search, {key}|{title}]"
    except Exception:
        return None
    return None


# ── The sign-in wall (S5) ────────────────────────────────────────────────────
# Free text (a vision answer about the page) that says the page wants a
# sign-in, or is an error page. "Not signed in" is a wall; "already signed
# in" / "signed in as" / "no sign-in prompt" are not.
_SIGNED_IN_RE = re.compile(
    r"\b(?:already\s+(?:signed|logged)\s+in|(?:signed|logged)\s+in\s+as|"
    r"(?:is|are|you're|you\s+are)\s+(?:already\s+)?(?:signed|logged)\s+in\b)",
    re.IGNORECASE)
_NOT_SIGNED_IN_RE = re.compile(
    r"\b(?:not|isn'?t|aren'?t)\s+(?:currently\s+)?(?:signed|logged)\s+in\b|"
    r"\b(?:signed|logged)\s+out\b", re.IGNORECASE)
_SIGN_IN_RE = re.compile(
    r"\b(?:sign[\s-]?in|log[\s-]?in|signin|login)\b", re.IGNORECASE)
_NO_SIGN_IN_RE = re.compile(
    r"\b(?:no|without|not\s+(?:asking|showing|requiring)|doesn'?t\s+"
    r"(?:show|ask|require|need)|isn'?t\s+(?:asking|showing))\s+(?:a\s+|any\s+|"
    r"for\s+(?:a\s+)?|to\s+)?(?:sign[\s-]?in|log[\s-]?in|signin|login)",
    re.IGNORECASE)
_ERROR_PAGE_RE = re.compile(
    r"\boops\b|link\s+(?:isn'?t|is\s+not)\s+working|"
    r"page\s+(?:not\s+found|(?:isn'?t|is\s+not|wasn'?t)\s+(?:available|found))|"
    r"\b404\b|something\s+went\s+wrong|page\s+(?:you\s+requested\s+)?"
    r"(?:could\s*n[o']t|cannot|can'?t)\s+be\s+found", re.IGNORECASE)


def wall_kind(text) -> Optional[str]:
    """"sign_in", "error" or None for a description of a page. A sign-in
    prompt wins over an error (the live HBO Max page showed both, and the
    sign-in is what the owner can fix). Never raises."""
    try:
        t = " ".join(str(text or "").split())
        if not t:
            return None
        if _NOT_SIGNED_IN_RE.search(t):
            return "sign_in"
        if _SIGN_IN_RE.search(t) and not _SIGNED_IN_RE.search(t):
            stripped = _NO_SIGN_IN_RE.sub(" ", t)
            if _SIGN_IN_RE.search(stripped):
                return "sign_in"
        if _ERROR_PAGE_RE.search(t):
            return "error"
    except Exception:
        return None
    return None


def wall_line(key, kind) -> str:
    """The one plain sentence for a sign-in wall / error page on service
    ``key``; "" for no wall."""
    name = service_name(key) or "That service"
    if kind == "sign_in":
        return (f"{name} isn't signed in on this browser, sir - sign in once "
                "and I can take it from there.")
    if kind == "error":
        return (f"{name} showed an error page instead of the show, sir - if "
                "it isn't signed in on this browser, sign in once and I can "
                "take it from there.")
    return ""


def wall_question(key) -> str:
    """The strict vision question that asks whether a service page is a
    sign-in wall or an error page. Its answer goes to ``parse_wall_verdict``."""
    name = service_name(key) or "streaming"
    return (
        f"This is a {name} page in a web browser. Ignore any chat, assistant "
        "or terminal windows. Is the page asking the viewer to SIGN IN or "
        "LOG IN (a Sign In / Log In button or form, or a sign-up / pricing "
        "page with no profile and no search results), or is it an ERROR page "
        "(for example 'Oops', 'this link isn't working', 'page not found')? "
        "Reply with exactly one word first - SIGNIN, ERROR or OK - then a "
        "short reason.")


def parse_wall_verdict(answer) -> Optional[str]:
    """"sign_in" / "error" / "ok" from a ``wall_question`` answer (leading
    [tags] such as "[local-vision]" ignored), else None. Never raises."""
    try:
        text = str(answer or "").strip()
        while text.startswith("[") and "]" in text:
            text = text.split("]", 1)[1].strip()
        first = re.sub(r"[^A-Za-z-]", "", text.split(maxsplit=1)[0]
                       if text else "").upper().replace("-", "")
        return {"SIGNIN": "sign_in", "LOGIN": "sign_in", "ERROR": "error",
                "OK": "ok"}.get(first)
    except Exception:
        return None
