"""core/screen_resolve.py - which on-screen element a spoken target means,
from TEXT alone (2026-10-05).

WHY THIS EXISTS
===============
Live 00:28:23 "click that Mr. Beast video": JARVIS had no way to read the
YouTube grid it had just opened, so the brain searched YouTube instead and
played a different video in a new window on another monitor. Every element
a click could want is already named by Windows (UI Automation) or readable
by OCR; this module turns those names + rectangles into ONE target, an
honest "which one?" with the real options, or nothing.

The rules are the research resolver (screen_vision_20261005, dev set UIA
30/30, OCR 33/33) ported unchanged in scale (floor 0.34, margin 0.15), plus
four general fixes found on the held-out page:

  * cards are grouped by geometry, so "that MrBeast video" resolves to the
    card's TITLE link (or its thumbnail), never the channel link;
  * ordinals ("the third video") count VIDEO cards in reading order; a
    stacked navigation list (the sidebar) is not a card;
  * YouTube's long accessible labels ("Title by Channel 1.2M views 3 days
    ago 10 minutes") are split into title and channel before scoring, so the
    spoken line names the title;
  * light stemming ("landing" ~ "land", "rockets" ~ "rocket") and compound
    containment ("airplane" ~ "plane") for speech-recognition wording.

Candidates are dicts: ``text``, ``rect`` [x, y, w, h], ``type`` (a UIA
control type name, or "ocr"), and any extra keys the caller wants back
(``href``, ``el``, ``hwnd`` ...). Results:

  {"status": "ok", "target": cand, "score": s, "why": ...}
  {"status": "ambiguous", "options": [cand, ...], "score": s}
  {"status": "none"}

Pure stdlib; never raises at the public API.
"""
from __future__ import annotations

import difflib
import functools
import math
import re

__all__ = [
    "FLOOR", "MARGIN", "REWRITE_MIN",
    "toks", "content_tokens", "tok_sim", "split_label", "prepare",
    "group_cards", "video_cards", "card_target", "reading_order",
    "is_ad_card", "ad_members",
    "resolve", "describe_position", "legend", "label_of",
]

# Acceptance on the research prototype's own scale (SPEC 0.2-3).
FLOOR = 0.34
MARGIN = 0.15
# The rewrite guard (an on-screen match overriding the brain's chosen
# youtube_play / open_url / web_search) needs a much stronger match.
REWRITE_MIN = 0.75

STOP = frozenset("""
click clicks clicking on open play watch select press tap hit choose pick go
to the a an that this those these one ones video videos clip clips thing
things button buttons link links please jarvis sir for me it from of in at
with about by screen monitor middle left right top bottom page tab tabs i
want wanted can you could would just there here called named titled item
entry option result results thumbnail icon start and then ahead also mean
meant like is was were be been am are which what whatever thumbnails up
my your our his her their
""".split())

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
       "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
       "last": -1, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
       "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10}
NUMW = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
# A duration / LIVE badge is never the title.
DUR = re.compile(r"^\s*(\S.*?\s)?(\d{1,2}:)?\d{1,2}:\d{2}\s*$|LIVE\s*$")
# "Title by Channel 1.2M views 3 days ago 10 minutes, 5 seconds"
_LONG_LABEL_RE = re.compile(
    r"^(?P<title>.+?)\s+by\s+(?P<channel>.+?)\s+"
    r"(?:[\d.,]+\s*[KMB]?\s+views?|No\s+views|[\d.,]+\s*[KMB]?\s+watching)\b",
    re.IGNORECASE)
_CLICKABLE = frozenset({"Hyperlink", "Button", "MenuItem", "TabItem",
                        "ListItem", "CheckBox", "RadioButton", "TreeItem",
                        "SplitButton", "DataItem", "ComboBox"})
# A card holding more elements than this is a list (a sidebar), not a card.
_CARD_MAX_ELEMENTS = 6


def _norm(s) -> str:
    return str(s or "").lower().replace("’", "'")


def toks(s) -> list:
    """Lower-case word tokens; "Mr. Beast" / "mister beast" -> "mrbeast"."""
    try:
        s = _norm(s)
        s = re.sub(r"\bmr\.?\s+", "mr", s)
        s = re.sub(r"\bmister\s+", "mr", s)
        return re.findall(r"[a-z0-9$]+", s)
    except Exception:
        return []


def squash(s) -> str:
    return re.sub(r"[^a-z0-9]", "", _norm(s).replace("mister", "mr"))


def _stem(t: str) -> str:
    if t.endswith("'s"):
        t = t[:-2]
    for suf, min_len in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if len(t) >= min_len and t.endswith(suf) and not t.endswith("ss"):
            return t[: -len(suf)]
    return t


@functools.lru_cache(maxsize=65536)
def tok_sim(q: str, c: str) -> float:
    """How well spoken token ``q`` matches on-screen token ``c`` (0..1).
    Cached (a page repeats its words, and the IDF pass and the scoring pass
    ask the same pairs), and the edit-distance ratio is only computed when
    the lengths allow 0.8 (review 2026-10-05: 850 ms for 2,400
    candidates, all in difflib)."""
    if q == c:
        return 1.0
    if len(q) >= 4 and len(c) >= 4:
        if _stem(q) == _stem(c):
            return 0.9
        lq, lc = len(q), len(c)
        if 2.0 * min(lq, lc) / (lq + lc) < 0.8:
            r = 0.0                       # ratio() could not reach 0.8
        else:
            r = difflib.SequenceMatcher(None, q, c).ratio()
        # ASR slips: "senat"~"cenat", "veritasiam" - a letter or so, not a
        # missing prefix ("mrbeast" is not "beast", ratio 0.83).
        if r >= 0.8 and (abs(len(q) - len(c)) <= 1 or r >= 0.9):
            return r
    # Compound words: "airplane" ~ "plane" (at least three more letters in
    # the spoken word - "mrbeast" is NOT "beast mode"), and "beast" ~
    # "mrbeast" (the spoken word inside the on-screen one).
    if len(c) >= 5 and c in q and len(q) - len(c) >= 3:
        return 0.8
    if len(q) >= 5 and q in c:
        return 0.8
    return 0.0


def content_tokens(query) -> list:
    """The words of ``query`` that name a thing (stop words, ordinals and
    stray single letters - the "s" of "that's" - dropped, number words as
    digits)."""
    return [NUMW.get(t, t) for t in toks(query)
            if t not in STOP and t not in ORD
            and (len(t) > 1 or t.isdigit())]


# "the one that's playing / on now / already up": a referent to whatever is
# PLAYING, not words to look for in titles.
_PLAYING_RE = re.compile(r"\b(playing|currently|already|on\s+now)\b")
_PLAYING_WORDS = frozenset({"playing", "currently", "already", "now"})


def ordinals(query) -> list:
    return [ORD[t] for t in toks(query) if t in ORD]


def split_label(text) -> tuple:
    """(title, channel) from a long video label, else (text, "")."""
    try:
        s = " ".join(str(text or "").split())
        m = _LONG_LABEL_RE.match(s)
        if m and len(m.group("title")) >= 3:
            return m.group("title").strip(), m.group("channel").strip()
        return s, ""
    except Exception:
        return str(text or ""), ""


def prepare(cands) -> list:
    """Copies of ``cands`` with long video labels split: ``text`` becomes
    the title, ``channel`` the channel, ``raw`` the label as read."""
    out = []
    for c in cands or ():
        try:
            if not isinstance(c, dict) or not c.get("rect"):
                continue
            d = dict(c)
            d["raw"] = str(c.get("text") or "")
            title, channel = split_label(d["raw"])
            d["text"] = title
            if channel:
                d["channel"] = channel
            d["rect"] = [float(v) for v in list(c["rect"])[:4]]
            out.append(d)
        except Exception:
            continue
    return out


def label_of(cand) -> str:
    """The words to SAY for a candidate (its title, not the long label)."""
    try:
        return " ".join(str(cand.get("text") or "").split())
    except Exception:
        return ""


_CARD_GAP = 18
_CARD_COL = 60


def group_cards(cands) -> list:
    """Stack elements that share a column (x overlap) and touch vertically
    (thumbnail / title / channel / meta of one video card). Near-linear:
    candidates arrive top to bottom, so only cards still OPEN (their last
    element ends at most 18 px above) in a nearby column are tried (review
    2026-10-05: every card was tried for every element - 364 ms at 2,400).
    The first matching card in creation order wins, as before."""
    cs = sorted([c for c in cands if c.get("rect")],
                key=lambda c: (c["rect"][1], c["rect"][0]))
    cards: list = []
    cols: dict = {}               # column bucket -> [card index, ...]
    for c in cs:
        x, y, w, h = c["rect"]
        home = None
        b0 = int(x // _CARD_COL)
        near = []
        for bk in (b0 - 1, b0, b0 + 1):
            idxs = cols.get(bk)
            if not idxs:
                continue
            # drop cards that can never take another element
            idxs[:] = [i for i in idxs
                       if y - (cards[i][-1]["rect"][1]
                               + cards[i][-1]["rect"][3]) <= _CARD_GAP]
            near.extend(idxs)
        for i in sorted(near):
            card = cards[i]
            lx, ly, lw, lh = card[-1]["rect"]
            ov = min(x + w, lx + lw) - max(x, lx)
            if (ov > 0.5 * min(w, lw) and -4 <= y - (ly + lh) <= _CARD_GAP
                    and abs(x - card[0]["rect"][0]) < _CARD_COL):
                home = card
                break
        if home is None:
            cards.append([c])
            cols.setdefault(b0, []).append(len(cards) - 1)
        else:
            home.append(c)
    return cards


def _is_titleish(c) -> bool:
    return len(c.get("text") or "") >= 12 and c["rect"][3] <= 60


def _is_thumb(c) -> bool:
    return c["rect"][3] > 120


# An advert is not a video (review 2026-10-05: "the first video" picked a
# "Sponsored" Shop-Now card): a card with a sponsor marker or a sales
# call-to-action as one of its own lines.
_AD_MARK_RE = re.compile(
    r"^\s*(?:sponsored|ad|ads|promoted|advertisement)\b(?:\s*[·•|:-].*)?$",
    re.IGNORECASE)
_AD_CTA_RE = re.compile(
    r"^\s*(?:shop\s+now|buy\s+now|order\s+now|learn\s+more|sign\s+up|"
    r"get\s+offer|visit\s+(?:site|advertiser)|install(?:\s+now)?|"
    r"download(?:\s+now)?|get\s+the\s+app|book\s+now|apply\s+now)\s*$",
    re.IGNORECASE)
_AD_SUFFIX_RE = re.compile(r"\s[-–—|·]\s*(?:shop|buy|order)\s+now\s*$",
                           re.IGNORECASE)


def _line_text(c) -> str:
    return str(c.get("raw") or c.get("text") or "")


def is_ad_card(card) -> bool:
    """A grouped card that is an advert: a sponsor marker ("Sponsored",
    "Ad · shop.example") or a "<title> - Shop Now" line among its own
    lines, or a sales call-to-action ("Shop now", "Install") on a card WITH
    a thumbnail - a plain "Download" / "Learn more" / "Sign up" button under
    a heading is not an advert. Never raises."""
    try:
        if any(_AD_MARK_RE.match(_line_text(c))
               or _AD_SUFFIX_RE.search(_line_text(c)) for c in card):
            return True
        return (any(_AD_CTA_RE.match(_line_text(c)) for c in card)
                and any(_is_thumb(c) for c in card if c.get("rect")))
    except Exception:
        return False


def ad_members(cands) -> set:
    """ids of the candidates that belong to an advert: every element of an
    ad card (a sponsor marker or a sales call-to-action among its stacked
    lines), and - for a LONE "Sponsored" label - the block right under it
    (the same column, within 120 px). Never raises."""
    out = set()
    try:
        lone = []
        for card in group_cards(cands):
            if not is_ad_card(card):
                continue
            if len(card) >= 2:
                out.update(id(c) for c in card)
            elif _AD_MARK_RE.match(str(card[0].get("raw")
                                       or card[0].get("text") or "")):
                lone.append(card[0])
                out.add(id(card[0]))
            else:
                out.add(id(card[0]))
        for m in lone:
            mx, my, mw, mh = m["rect"]
            for c in cands:
                if not c.get("rect") or id(c) in out:
                    continue
                x, y, w, h = c["rect"]
                if (0 < y - my <= 120 and abs(x - mx) < _CARD_COL):
                    out.add(id(c))
    except Exception:
        return out
    return out


def video_cards(cands) -> list:
    """The video cards among ``cands``: 2-6 stacked elements with a title
    line, and a thumbnail or at least three lines; never an advert. A
    stacked navigation list (the sidebar) is not one."""
    cards = group_cards(cands)
    grid = [cd for cd in cards
            if 2 <= len(cd) <= _CARD_MAX_ELEMENTS
            and any(_is_titleish(c) for c in cd)
            and (any(_is_thumb(c) for c in cd) or len(cd) >= 3)
            and not is_ad_card(cd)]
    # Thumbnails known (UIA / OCR saw them): the cards ARE the thumb cards.
    if any(any(_is_thumb(c) for c in cd) for cd in grid):
        grid = [cd for cd in grid if any(_is_thumb(c) for c in cd)]
    return grid


def card_target(card):
    """The element of a card to click: its title (the first long text that
    is not a duration and not the channel line), else the thumbnail."""
    texts = [c for c in card
             if not DUR.match(c.get("text") or "") and c.get("text")]
    if not texts:
        return card[0]
    titles = [c for c in texts if _is_titleish(c)]
    if titles:
        return titles[0]
    thumbs = [c for c in card if _is_thumb(c)]
    return thumbs[0] if thumbs else texts[0]


def _rows(units) -> list:
    """Units grouped into visual rows (tops within 40 px), top to bottom,
    each row left to right."""
    us = sorted(units, key=lambda u: (u[0]["rect"][1], u[0]["rect"][0]))
    rows: list = []
    for u in us:
        top = u[0]["rect"][1]
        if rows and abs(top - rows[-1][0][0]["rect"][1]) <= 40:
            rows[-1].append(u)
        else:
            rows.append([u])
    for r in rows:
        r.sort(key=lambda u: u[0]["rect"][0])
    return rows


def reading_order(units) -> list:
    """``units`` (lists of candidates) in reading order: rows top to
    bottom, each left to right."""
    return [u for row in _rows(units) for u in row]


def _pick_ordinal(units, n):
    ordered = reading_order(units)
    i = n - 1 if n > 0 else len(ordered) - 1
    return ordered[i] if 0 <= i < len(ordered) else None


def resolve(query, cands, playing_title=None, margin: float = MARGIN) -> dict:
    """Resolve spoken ``query`` against on-screen ``cands`` (see module
    docstring). Never raises."""
    try:
        return _resolve(query, prepare(cands), playing_title, margin)
    except Exception:
        return {"status": "none"}


def _resolve(query, cands, playing_title, margin) -> dict:
    ql = _norm(query)
    ords = ordinals(query)
    content = content_tokens(query)
    if (playing_title and re.search(
            r"\b(playing|on screen|currently|already)\b", ql)):
        hits = [c for c in cands if c.get("text")
                and squash(c["text"]) == squash(playing_title)]
        if hits:
            return {"status": "ok", "target": hits[0], "score": 1.0,
                    "why": "playing"}
    if _PLAYING_RE.search(ql):
        # Without a known playing title, "the one that's playing" names
        # nothing on the page - never a title that happens to score
        # (live bench 2026-10-05: it picked a sidebar video).
        content = [t for t in content if t not in _PLAYING_WORDS]
        if not content and not ords:
            return {"status": "none", "why": "playing-unknown"}
    want_video = bool(re.search(r"\b(video|videos|one|clip|watch|play)\b", ql))
    # A video request never lands on an advert (review 2026-10-05: "the
    # first video" picked a "Sponsored" Shop-Now card) - unless he asks for
    # the ad. Any other request ("click Download") sees every element.
    if want_video and not re.search(
            r"\b(?:ad|ads|advert|sponsored|promoted)\b", ql):
        ads = ad_members(cands)
        if ads:
            cands = [c for c in cands if id(c) not in ads]
    grid = video_cards(cands) if want_video else []
    if ords and grid and not content:
        unit = _pick_ordinal(grid, ords[0])
        if unit is not None:
            return {"status": "ok", "target": card_target(unit),
                    "score": 1.0, "why": "ordinal"}
    if not content:
        return {"status": "none"}
    # IDF over the candidate texts (a word on every card says little).
    docs = [set(toks(c.get("text")) + toks(c.get("channel"))) for c in cands]
    n = len(docs) or 1

    def idf(t):
        df = sum(1 for d in docs if any(tok_sim(t, x) for x in d))
        return math.log((n + 1) / (df + 0.5))

    w = {t: max(idf(t), 0.05) for t in content}
    tot = sum(w.values()) or 1.0
    units = grid if grid else [[c] for c in cands]
    scored = []
    for unit in units:
        utoks = [x for c in unit
                 for x in toks(c.get("text")) + toks(c.get("channel"))]
        usq = squash(" ".join(str(c.get("text") or "") + " "
                              + str(c.get("channel") or "") for c in unit))
        # Spoken words that are ONE word on screen ("i show speed" ->
        # "IShowSpeed"): adjacent content words joined, found in the unit.
        # A short word alone inside a longer one is not a match ("test" is
        # not "fastest" - 2026-10-05).
        joined = set()
        for i in range(len(content) - 1):
            j = content[i] + content[i + 1]
            if len(j) >= 6 and j in usq:
                joined |= {i, i + 1}
        s = 0.0
        for i, t in enumerate(content):
            best = max([tok_sim(t, x) for x in utoks]
                       + [1.0 if i in joined else 0.0]
                       + [1.0 if len(t) >= 6 and t in usq else 0.0])
            s += w[t] * best
        s = s / tot
        # Exact-label bonus ("click usage" -> the "Usage" link, not "Usage
        # this month").
        if any(squash(c.get("text")) == squash(" ".join(content))
               for c in unit):
            s += 0.3
        if any(c.get("type") in _CLICKABLE for c in unit):
            s += 0.05
        scored.append((s, unit))
    scored.sort(key=lambda x: -x[0])
    if not scored or scored[0][0] < FLOOR:
        return {"status": "none"}
    top = scored[0][0]
    close = [u for s, u in scored if s >= top - margin and s >= FLOOR]
    pick = (card_target if grid else (lambda u: u[0]))
    if len(close) > 1:
        if ords:
            unit = _pick_ordinal(close, ords[0])
            if unit is not None:
                return {"status": "ok", "target": pick(unit), "score": top,
                        "why": "ordinal among matches"}
        return {"status": "ambiguous", "options": [pick(u) for u in close],
                "score": top}
    others = [pick(u) for s, u in scored[1:4] if s >= FLOOR]
    return {"status": "ok", "target": pick(scored[0][1]), "score": top,
            "why": "text", "others": others}


_ORDINAL_WORDS = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th",
                  "9th", "10th")


def describe_position(cand, cands) -> str:
    """Where ``cand`` sits among ``cands``: "top row, 2nd", "2nd row, 1st",
    or "" when it is not in a grid. Never raises."""
    try:
        grid = video_cards(prepare(cands))
        units = grid or [[c] for c in prepare(cands)]
        rows = _rows(units)
        r0 = [float(v) for v in list(cand.get("rect"))[:4]]
        for ri, row in enumerate(rows):
            for ci, unit in enumerate(row):
                if any([float(v) for v in c["rect"]] == r0 for c in unit):
                    row_name = ("top row" if ri == 0 else
                                f"{_ORDINAL_WORDS[ri] if ri < 10 else ri + 1}"
                                " row")
                    col = (_ORDINAL_WORDS[ci] if ci < 10 else str(ci + 1))
                    return f"{row_name}, {col}" if len(row) > 1 else row_name
    except Exception:
        return ""
    return ""


def legend(options, cands=None, max_options: int = 3) -> str:
    """"'A' (top row, 2nd) or 'B' (2nd row, 1st)" for a question naming
    real options. Never raises."""
    try:
        parts = []
        for o in list(options or ())[:max_options]:
            name = label_of(o)
            if not name:
                continue
            if len(name) > 70:
                name = name[:67].rstrip() + "..."
            pos = describe_position(o, cands) if cands else ""
            parts.append(f"'{name}'" + (f" ({pos})" if pos else ""))
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + " or " + parts[-1]
    except Exception:
        return ""
