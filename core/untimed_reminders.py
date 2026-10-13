"""Reminders that name no time, and the shopping list (2026-10-12).

"Remind me to buy milk" says WHAT but not WHEN. A timer needs a duration, so
the only things that ever happened to such a request were a guess (a model
inventing a duration, a bare number word in the label read as minutes) or a
refusal. Neither is an answer. The rule now:

  * a reminder that names no time and is about BUYING something goes on the
    shopping list;
  * any other reminder that names no time goes on the to-do list;
  * a reminder that names a time ("in 20 minutes", "at 5 pm", "tomorrow",
    "every Monday") is not handled here at all (``match`` returns None) - it
    stays a timer / schedule.

This module is pure parsing plus a small JSON store (data/shopping_list.json).
``match`` is the whole-utterance recogniser the monolith's voice shortcut and
the tests share; the actions live in skills/shopping_list.py.
"""
from __future__ import annotations

import re
import threading
from datetime import datetime
from typing import NamedTuple, Optional

from core.lead_fillers import strip_lead_filler

# The five actions the grammar can route to (skills/shopping_list.py).
ADD_REMINDER = "add_reminder"
ADD_ITEM = "add_shopping_item"
SHOW_LIST = "show_shopping_list"
REMOVE_ITEM = "remove_shopping_item"
CLEAR_LIST = "clear_shopping_list"


class Intent(NamedTuple):
    action: str
    arg: str


# ── time words: a reminder carrying one is a timer's job, not ours ──────────
_NUM = (r"(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
        r"twelve|fifteen|twenty|thirty|forty|forty five|sixty|\d+(?:\.\d+)?|"
        r"half an?|a couple of|a few)")
_UNIT = (r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|"
         r"years?)")
_DAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_DAY_ABBR = r"(?:mon|tues?|wed|thur?s?|fri|sat|sun)"
_CLOCK = r"(?:\d{1,2}(?::\d{2})?\s*(?:[ap]\.?\s?m\.?)?|noon|midnight|midday)"
_TIME_RE = re.compile(
    r"\b(?:"
    rf"(?:in|after|within)\s+{_NUM}\s*{_UNIT}"
    rf"|{_NUM}\s*{_UNIT}\s+from\s+now"
    rf"|(?:at|by|around|until|till)\s+{_CLOCK}\b"
    r"|\d{1,2}:\d{2}"
    r"|\d{1,2}\s*[ap]\.?\s?m\b\.?"
    r"|\d{1,2}\s*o'?clock"
    r"|noon|midnight|midday"
    r"|tomorrow|tonight|today|yesterday"
    r"|this\s+(?:morning|afternoon|evening|weekend|week|month|year)"
    r"|in\s+the\s+(?:morning|afternoon|evening)"
    r"|next\s+(?:week|month|year|weekend|" + _DAY + r")"
    r"|on\s+(?:" + _DAY + r"|" + _DAY_ABBR + r"|the\s+\d{1,2})"
    r"|(?:every|each)\s+(?:day|week|month|year|hour|morning|afternoon|evening|"
    r"night|weekend|weekday|other\s+\w+|\d+\s*" + _UNIT + r"|" + _DAY + r")"
    r"|daily|weekly|monthly|hourly|nightly"
    r"|" + _DAY + r"s?"
    r")\b",
    re.IGNORECASE)

# Vague "not now" words carry no time: the item still goes on a list. They are
# dropped from the stored text ("buy milk later" -> "buy milk").
_VAGUE_RE = re.compile(
    r"\s*,?\s*\b(?:later(?: on)?|sometime|some time|soon|eventually|"
    r"at some point|when i can|when i get a chance|at some stage)\b\s*$",
    re.IGNORECASE)


def has_time_words(text: str) -> bool:
    """True when ``text`` names a time a timer or schedule could use."""
    return bool(_TIME_RE.search(text or ""))


# ── shopping or not ─────────────────────────────────────────────────────────
_BUY_VERB_RE = re.compile(
    r"^(?:buy|purchase|restock|stock up on|order|pick up|pickup|grab|get|"
    r"fetch)\b\s*(.*)$", re.IGNORECASE)
_SURE_BUY = frozenset({"buy", "purchase", "restock", "stock up on", "order"})
# "get / grab / pick up" are also errands ("get the oil changed", "pick up the
# kids"): treat them as shopping only when nothing says otherwise.
_ERRAND_WORDS = frozenset({
    "with", "from", "at", "by", "before", "after", "when", "while", "until",
    "because", "so", "if", "then", "to", "up", "out", "back", "home", "going",
    "ready", "there", "rid", "together", "done", "started", "going", "kids",
    "kid", "dad", "mom", "mum", "wife", "husband", "son", "daughter",
    "brother", "sister", "friend", "friends", "him", "her", "them", "us",
    "dog", "cat", "laundry", "prescription", "package", "parcel", "mail",
    "truck", "car", "haircut", "checkup", "physical", "license", "lunch",
    "dinner", "breakfast",
})
# A shopping line is never a chore: "buy milk and call mom" is two errands.
_NON_ITEM_VERBS = frozenset({
    "call", "text", "email", "send", "tell", "ask", "check", "go", "take",
    "pay", "book", "schedule", "clean", "do", "make", "finish", "fix", "feed",
    "walk", "visit", "meet", "write", "read", "study", "submit",
    "return", "cancel", "renew", "pick", "drop", "turn", "sign", "wash", "mow",
    "cook", "wake",
})


def _split_items(rest: str) -> Optional[list]:
    """The things a shopping line names, or None when it is not a plain list
    of things ("milk, eggs and bread" -> three; "milk and call mom" -> None)."""
    rest = re.sub(r"^(?:some|a few|a couple of)\s+", "", rest.strip(),
                  flags=re.IGNORECASE)
    rest = rest.strip(" ,.-")
    if not rest:
        return None
    parts = [p.strip(" ,.-") for p in
             re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", rest, flags=re.IGNORECASE)]
    parts = [p for p in parts if p]
    if not parts:
        return None
    if len(parts) > 1 and any(len(p.split()) > 4 for p in parts):
        parts = [rest]           # a sentence, not a list: keep it whole
    if len(parts) > 1 and any(p.split()[0].lower() in _NON_ITEM_VERBS
                              for p in parts):
        return None
    return parts


def shopping_items(task: str) -> Optional[list]:
    """The items when ``task`` ("buy milk and eggs") is a purchase, else None."""
    m = _BUY_VERB_RE.match((task or "").strip())
    if not m:
        return None
    verb = re.match(r"^(stock up on|pick up|pickup|\w+)", task.strip(),
                    re.IGNORECASE).group(1).lower()
    rest = m.group(1)
    if not rest.strip():
        return None
    if verb not in _SURE_BUY:
        words = re.findall(r"[a-z']+", rest.lower())
        if any(w in _ERRAND_WORDS for w in words):
            return None
        if words and (words[-1].endswith("ed") or words[-1] in
                      ("done", "fixed", "cut", "set", "made", "ready")):
            return None
    return _split_items(rest)


# ── the grammar ─────────────────────────────────────────────────────────────
_LIST_N = r"(?:shopping|grocery|groceries)"
_MY = r"(?:my |the |our |a )?"
_LIST = rf"{_MY}{_LIST_N} list"

_REMIND_RE = re.compile(
    r"^(?:(?:set|make|add|create|put in|put down|give me)\s+(?:a |an )?"
    r"reminder\s+(?:to|for me to|about|that i (?:need to|have to|should))"
    r"|(?:i (?:need|want|would like)(?: you)? (?:a |to )?(?:reminder|be "
    r"reminded)\s+(?:to|about))"
    r"|(?:remind|reminds) me\s+(?:to|about|of|that i (?:need to|have to|"
    r"should)|i (?:need to|have to|should))"
    r"|(?:do not|dont|don't) let me forget\s+(?:to|about)?"
    r"|make sure i remember\s+(?:to|about)?"
    r")\s+(.+)$",
    re.IGNORECASE)

_ADD_RE = re.compile(
    rf"^(?:add|put|throw|stick|pop)\s+(.+?)\s+(?:to|on|onto|in|into)\s+{_LIST}$",
    re.IGNORECASE)

_SHOW_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    rf"^what(?:'s|s| is| are)(?: currently| now)? (?:on|in) {_LIST}$",
    rf"^(?:what|which) (?:do i|have i got to|all do i) (?:have on|got on) "
    rf"{_LIST}$",
    rf"^(?:read|show|list|give|tell|check|pull up|open)(?: me)?(?: out)?"
    rf"(?: all)?(?: of)? (?:what(?:'s| is) (?:on|in) )?{_LIST}$",
    rf"^(?:read|show|tell) me what(?:'s|s| is) (?:on|in) {_LIST}$",
    rf"^(?:do i have|have i got|is there) (?:anything|something|much|"
    rf"stuff|any items?) (?:on|in) {_LIST}$",
    rf"^(?:what|anything) (?:is |s )?(?:left )?on {_LIST}$",
    r"^what (?:do i|should i) (?:still )?(?:need to )?buy$",
    r"^what (?:do|did) i (?:still )?need to buy$",
    r"^(?:read|show|list) (?:me )?(?:the |my )?(?:shopping|grocery) "
    r"(?:items|things)$",
))

_REMOVE_RE = re.compile(
    rf"^(?:remove|delete|take|cross|scratch|strike|knock|mark|check|tick)\s+"
    rf"(.+?)\s+(?:off|from|out of|off of)\s+{_LIST}$",
    re.IGNORECASE)
_ALL_WORDS = frozenset({"everything", "all", "all of it", "all of them",
                        "all items", "all the items", "all the things",
                        "every item", "everything on it", "it all",
                        "all of the items"})
_CLEAR_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    rf"^(?:clear|empty|wipe|reset|erase|delete|scrap|clean out|dump)"
    rf"(?: out)?(?: all| everything| all of)?(?: on| in| from| off)?"
    rf"(?: the)?(?: entire| whole)? ?{_LIST}$",
    rf"^(?:clear|empty|wipe|erase) (?:everything|all) (?:off|from|on|in) "
    rf"{_LIST}$",
    rf"^start (?:over|fresh|a new one)(?: on| with)? {_LIST}$",
    rf"^start a new {_LIST_N} list$",
))


def _clean(text) -> str:
    """The utterance without lead-in filler, wake word and end punctuation,
    internal spacing collapsed; original case kept (items are shown as said)."""
    if not isinstance(text, str):
        return ""
    t = strip_lead_filler(text.replace("’", "'"))
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[\s.!?;:]+$", "", t)
    t = re.sub(r"(?:,\s*|\s+)(?:please|sir|jarvis|thanks|thank you)$", "", t,
               flags=re.IGNORECASE)
    return t.strip(" ,")


def _task_intent(task: str) -> Optional[Intent]:
    """Route an untimed reminder body ("buy milk", "call the dentist")."""
    task = _VAGUE_RE.sub("", task).strip(" ,.-")
    if not task:
        return None
    items = shopping_items(task)
    if items:
        return Intent(ADD_ITEM, ", ".join(items))
    return Intent(ADD_REMINDER, task)


def match(text) -> Optional[Intent]:
    """The shopping-list / untimed-reminder intent in a WHOLE utterance, or
    None (the model, a timer or a schedule handles it). Never raises."""
    try:
        t = _clean(text)
        if not t or len(t) > 300:
            return None
        for rx in _CLEAR_RES:
            if rx.match(t):
                return Intent(CLEAR_LIST, "")
        for rx in _SHOW_RES:
            if rx.match(t):
                return Intent(SHOW_LIST, "")
        m = _REMOVE_RE.match(t)
        if m:
            what = m.group(1).strip(" ,")
            if what.lower() in _ALL_WORDS:
                return Intent(CLEAR_LIST, "")
            return Intent(REMOVE_ITEM, what)
        m = _ADD_RE.match(t)
        if m:
            parts = _split_items(m.group(1))
            return Intent(ADD_ITEM, ", ".join(parts or [m.group(1).strip()]))
        m = _REMIND_RE.match(t)
        if m:
            body = m.group(1).strip()
            body = re.sub(r"^(?:to|about)\s+", "", body, flags=re.IGNORECASE)
            if has_time_words(body) or not body:
                return None
            return _task_intent(body)
    except Exception:
        return None
    return None


# ── the lists (data/shopping_list.json, data/reminders_list.json) ───────────
# Personal no-time reminders live in their OWN file, never in jarvis_todo.md:
# that file is the developer task queue the overnight upgrade pipeline reads
# as coding tasks.
SHOPPING_LIST = "shopping_list.json"
REMINDERS_LIST = "reminders_list.json"
_FILE_NAME = SHOPPING_LIST
_lock = threading.RLock()


def _path(name: str = SHOPPING_LIST) -> str:
    from core import paths
    return paths.data_file(name)


def load_items(name: str = SHOPPING_LIST) -> list:
    """The list as [{"text": str, "added": iso}], oldest first. A missing or
    unreadable file is an empty list, never an error."""
    import json
    with _lock:
        try:
            with open(_path(name), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
    rows = data.get("items") if isinstance(data, dict) else None
    out = []
    for r in rows or []:
        if isinstance(r, dict) and str(r.get("text", "")).strip():
            out.append({"text": str(r["text"]).strip(),
                        "added": str(r.get("added", ""))})
    return out


def _save(items: list, name: str = SHOPPING_LIST) -> None:
    from core.atomic_io import _atomic_write_json
    _atomic_write_json(_path(name), {"items": items})


def _key(s: str) -> str:
    """Comparison key: lower case, no articles / "some", singular-ish."""
    words = [w for w in re.findall(r"[a-z0-9%']+", (s or "").lower())
             if w not in ("a", "an", "the", "some", "any", "of", "my")]
    out = []
    for w in words:
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith(("ses", "xes", "ches", "shes")):
            w = w[:-2]
        elif len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return " ".join(out)


def add_items(texts: list, name: str = SHOPPING_LIST) -> tuple:
    """Add items. Returns (added, already_there) as lists of the spoken text.
    Raises OSError if the list cannot be saved (the caller says so)."""
    with _lock:
        items = load_items(name)
        have = {_key(i["text"]) for i in items}
        added, dup = [], []
        for raw in texts:
            text = re.sub(r"\s+", " ", str(raw)).strip(" ,.")
            k = _key(text)
            if not text or not k:
                continue
            if k in have:
                dup.append(text)
                continue
            have.add(k)
            added.append(text)
            items.append({"text": text,
                          "added": datetime.now().isoformat(timespec="seconds")})
        if added:
            _save(items, name)
        return added, dup


def remove_matching(what: str, name: str = SHOPPING_LIST) -> tuple:
    """Remove the item ``what`` names. Returns (removed_text | None,
    candidates): ``removed_text`` when exactly one item matched (an exact
    match wins over partial ones), else None with the ambiguous
    ``candidates`` (empty = nothing matched)."""
    with _lock:
        items = load_items(name)
        k = _key(what)
        if not k:
            return None, []
        exact = [i for i in items if _key(i["text"]) == k]
        if len(exact) == 1:
            hit = exact
        else:
            def _words(s):
                return set(s.split())
            kw = _words(k)
            hit = [i for i in items
                   if kw and (kw <= _words(_key(i["text"]))
                              or _words(_key(i["text"])) <= kw)]
        if len(hit) == 1:
            items.remove(hit[0])
            _save(items, name)
            return hit[0]["text"], []
        return None, [i["text"] for i in hit]


def clear_all(name: str = SHOPPING_LIST) -> int:
    """Empty the list, keeping the previous contents in a .bak file beside it
    (so a mis-heard "clear" can be undone by hand). Returns how many items were
    removed."""
    import shutil
    with _lock:
        items = load_items(name)
        if not items:
            return 0
        try:
            shutil.copyfile(_path(name), _path(name) + ".bak")
        except Exception:
            pass
        _save([], name)
        return len(items)


def spoken_join(parts: list) -> str:
    """"a", "a and b", "a, b and c"."""
    parts = [p for p in parts if p]
    if len(parts) <= 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]
