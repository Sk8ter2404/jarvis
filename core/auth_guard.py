"""core/auth_guard.py - JARVIS never signs in for the owner (2026-10-05).

WHY THIS EXISTS
===============
Live 00:14:07 the owner said "pull up that page so I can sign in" - HE signs
in. JARVIS opened the console page, looked at the screen, and then, on its
own, emitted a click on the owner's Google account-chooser entry (his name and
e-mail address) and said "I've selected your account; just one more step to
get through the gate." The click missed; had it landed, JARVIS would have
picked an identity and started a sign-in nobody asked it to make. Nothing in
the action layer knew a sign-in page from any other page.

THE RULE
========
A click, typing, or a submit key (Enter / Tab / Space) is REFUSED when there
is EVIDENCE of a sign-in - and the owner's own words this turn did not ask
for exactly that input:

  * the click's TARGET is itself a sign-in control (``auth_control``): an
    account-chooser entry (a name with an e-mail address), "Sign in" / "Log
    in", "Continue with Google" (an identity provider), "Continue as <name>",
    "Use another account", "Authorize" / "Allow access";
  * the PAGE is a sign-in page (``auth_page``): its address (only while the
    window JARVIS opened it in is in front and has not moved on - the caller
    decides), its window title anchored at the start ("Sign in - Google
    Accounts", "Log in to ...", "Login | ..."), or what a look at the screen
    this turn said ("Choose an account", "a login prompt", "asking for your
    email address", "listing one account");
  * a sign-in POP-UP was seen over an ordinary page (``auth_overlay``: "a
    Google sign-in pop-up", "Sign in with Google", "Continue with Google"):
    a click with no target (coordinates), typing and a submit key are
    refused; a click on a named, ordinary target ("the article headline")
    still goes ahead;
  * this turn already looked for a sign-in control (``find_on_screen`` of the
    account entry) - a coordinate click is then that control;
  * or an earlier input was already refused this turn - the rest of that
    reply's clicks and keys are skipped (a refused click followed by
    "[ACTION: press, enter]" signed in anyway, review 2026-10-05).

"The owner asked" means an imperative clause addressed to JARVIS that starts
with the verb ("click Continue with Google", "Jarvis, press enter", "can you
pick my account") and names the target in that same clause. "So I can sign
in", "I'll choose my account" and anything with "don't click" never count.

The refusal is a TERMINAL line (core.failure_markers), so the follow-up chain
stops on it and it is spoken word for word: the page is up and ready for him
(READY_LINE), or - a sign-in button on an ordinary page - that the click
would sign him in (CONTROL_LINE). Pure stdlib, no I/O, never raises.
"""
from __future__ import annotations

import re
from typing import Iterable

from core.failure_markers import TERMINAL_FAILURE_PREFIX

__all__ = [
    "CONTROL_LINE",
    "READY_LINE",
    "auth_control",
    "auth_overlay",
    "auth_page",
    "click_refusal",
    "input_refusal",
    "is_submit_key",
    "owner_asked_for_click",
    "owner_asked_for_input",
]

READY_LINE = ("The sign-in page is up and ready for you, sir - I'll leave the "
              "signing in to you.")
CONTROL_LINE = ("That click would sign you in or grant access, sir - I'll "
                "leave that to you.")

ACCOUNT_ENTRY = "an account entry"
SIGN_IN_CONTROL = "a sign-in control"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A message row's subject divider ("Prof. X <x@y.edu> - Exam schedule"): an
# inbox row is not an account entry.
_SUBJECT_SEP_RE = re.compile(r"\s[-–—|]\s|:\s")
_FILE_RE = re.compile(r"\.[a-z0-9]{2,5}\s*$", re.IGNORECASE)
_PROVIDERS = (r"(?:google|microsoft|apple|github|facebook|e-?mail|sso|passkey|"
              r"phone|(?:a\s+)?work(?:\s+or\s+school)?|school)")

# A control whose whole job is signing someone in, choosing an identity, or
# granting an app access to it. NOT "password", "consent" or a bare "Allow":
# "Show password" and a cookie banner's "Accept all (consent)" sign nobody in
# (review 2026-10-05); on a real sign-in page the PAGE evidence refuses every
# click anyway.
_AUTH_CONTROL_RE = re.compile(
    r"\b(?:"
    r"sign[\s-]?(?:in|into|on)|signin|log[\s-]?(?:in|into|on)|login|"
    r"continue\s+with\s+" + _PROVIDERS + r"|"
    r"continue\s+as\s+(?!guest\b)\w+|"
    r"(?:choose|select|pick|use|switch)\s+(?:an?\s+|another\s+|your\s+|my\s+|"
    r"the\s+|this\s+|a\s+different\s+)?(?:\w+\s+)?account|"
    r"account\s+(?:chooser|picker|switcher)|"
    r"(?:google|microsoft|apple|github|anthropic)\s+account|"
    r"authori[sz]e|grant\s+access|allow\s+access"
    r")\b", re.IGNORECASE)
_ACCOUNT_WORD_RE = re.compile(
    r"\b(?:accounts?|profile|identity|signed\s+in|sign\s+in)\b",
    re.IGNORECASE)

# Sign-in hosts and paths (an address the page was opened at, or a title that
# carries one).
_AUTH_HOST_RE = re.compile(
    r"(?:^|[/.@\s])(?:accounts\.google\.com|login\.microsoftonline\.com|"
    r"login\.live\.com|account\.live\.com|login\.microsoft\.com|"
    r"appleid\.apple\.com|idmsa\.apple\.com|auth0\.com|okta\.com|"
    r"github\.com/(?:login|session))", re.IGNORECASE)
_AUTH_PATH_RE = re.compile(
    r"https?://[^\s/]+/(?:[^\s?#]*/)?(?:login|log-in|signin|sign-in|sign_in|"
    r"oauth\d?|authorize|sso|session/new|accountchooser|choose-account)"
    r"(?:[/?#]|$)", re.IGNORECASE)
# Window titles. ANCHORED (review 2026-10-05): "How to log in to Fortnite on
# PC - YouTube" and "Login flow redesign.docx - Word" are not sign-in pages.
# The title's FIRST part must be the sign-in phrase itself ("Sign in",
# "Log in to Example", "Login"), or one part a provider's sign-in page name.
_TITLE_SPLIT_RE = re.compile(r"\s+[-–—|·]\s+")
_TITLE_HEAD_RE = re.compile(
    r"^(?:sign[\s-]?in|log[\s-]?in|login|sign[\s-]?on|choose\s+an\s+account)"
    r"(?:\s+(?:to|with)\b.*)?$", re.IGNORECASE)
_TITLE_PART_RE = re.compile(
    r"^(?:google\s+accounts?|microsoft\s+account|(?:sign|log)\s+in\s+to\s+"
    r"your\s+(?:\w+\s+)?account)$", re.IGNORECASE)
# What a look at the screen says about a sign-in PAGE (page-level: every
# click, typing and submit key is refused).
_AUTH_SCREEN_RE = re.compile(
    r"\b(?:"
    r"choose\s+an\s+account|use\s+another\s+account|"
    r"account\s+(?:chooser|picker)|"
    r"listing\s+(?:one|two|three|\d+|an?|your|his|the\s+owner'?s?|several)\s+"
    r"(?:\w+\s+)?accounts?|"
    r"enter\s+(?:your\s+)?(?:password|passcode)|"
    r"(?:sign[\s-]?in|log[\s-]?in|login)\s+(?:page|screen|form|gate|wall|"
    r"prompt|dialog|window|modal)|"
    r"(?:asking|prompting|wants?|requires?|requesting)\s+(?:you\s+|him\s+|"
    r"the\s+(?:user|owner|viewer)\s+)?to\s+(?:sign|log)\s+in|"
    r"(?:requesting|requires?|needs?)\s+(?:a\s+)?(?:login|sign[\s-]?in)|"
    r"(?:sign|log)\s+in\s+to\s+continue|"
    r"(?:asking|prompting)\s+(?:you\s+|him\s+)?for\s+(?:an?\s+|your\s+|his\s+)?"
    r"(?:e-?mail(?:\s+address)?|password|username)"
    r")\b", re.IGNORECASE)
# A sign-in POP-UP over an ordinary page (Google One Tap, "Sign in with
# Google"): a click with no target, typing and a submit key are refused.
_AUTH_OVERLAY_RE = re.compile(
    r"\b(?:"
    r"(?:sign[\s-]?in|log[\s-]?in|login)\s+(?:pop-?up|popup|overlay)|"
    r"(?:sign|log)\s+in\s+with\s+(?:google|microsoft|apple|github)|"
    r"continue\s+with\s+" + _PROVIDERS + r"|"
    r"(?:e-?mail|password)\s+(?:field|box|input)"
    r")\b", re.IGNORECASE)
_SIGNED_IN_RE = re.compile(
    r"\b(?:already\s+(?:signed|logged)\s+in|(?:signed|logged)\s+in\s+as)\b",
    re.IGNORECASE)

# ── What the owner asked for ────────────────────────────────────────────────
_CLAUSE_SPLIT_RE = re.compile(
    r"[,.;:!?]+|\s+(?:and|then|so|but|because|while|after|before|once|or)\s+",
    re.IGNORECASE)
_CLAUSE_LEAD_RE = re.compile(
    r"^(?:(?:hey|ok|okay|now|just|please|jarvis|sir|yes|yeah|go\s+ahead\s+and|"
    r"can\s+you|could\s+you|would\s+you|will\s+you|i\s+need\s+you\s+to|"
    r"i\s+want\s+you\s+to|i'?d\s+like\s+you\s+to)\s+)+", re.IGNORECASE)
# "don't click anything", "do not sign me in", "never touch that".
_NEGATED_INPUT_RE = re.compile(
    r"\b(?:don'?t|do\s+not|never|no\s+need\s+to|without|stop)\s+"
    r"(?:\w+\s+){0,2}?(?:click|clicking|press|tap|hit|select|choose|pick|"
    r"type|typing|enter|sign|log|touch|fill)\b", re.IGNORECASE)
_CLICK_LEAD_RE = re.compile(
    r"^(?:click|clicks|press|tap|hit|select|choose|pick)\b", re.IGNORECASE)
_TYPE_LEAD_RE = re.compile(
    r"^(?:type|enter|write|fill(?:\s+in)?|put|input|paste)\b", re.IGNORECASE)
_KEY_LEAD_RE = re.compile(r"^(?:press|hit|tap|push)\b", re.IGNORECASE)
_LOGIN_WORD_RE = re.compile(
    r"\b(?:e-?mail|address|username|user\s*name|login|password|account)\b",
    re.IGNORECASE)
# Keys that submit or move through a sign-in form.
_SUBMIT_KEYS = {"enter": ("enter", "return"), "return": ("enter", "return"),
                "tab": ("tab",), "space": ("space", "spacebar"),
                "spacebar": ("space", "spacebar")}
# Words that do not name WHICH control ("click on the button for me").
_STOP_WORDS = frozenset("""
    the a an on in at of to for with my your his our this that these those
    button link option entry tile row icon box field page screen window
    please now for me it and or then just go ahead click clicks clicking
    press tap hit select choose pick jarvis sir you can could would
""".split())


def _words(text) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(w) >= 2 and w not in _STOP_WORDS}


def _is_account_entry(d: str) -> bool:
    """True for an account chooser's entry: an e-mail address alone or with
    a short name ("Pat Example (pat@example.com)"), or with an account word.
    An inbox row (a subject after a divider) is not one."""
    if not _EMAIL_RE.search(d):
        return False
    rest = _EMAIL_RE.sub(" ", d)
    if _ACCOUNT_WORD_RE.search(rest):
        return True
    if _SUBJECT_SEP_RE.search(rest):
        return False
    return len(re.findall(r"[A-Za-z][\w'.-]*", rest)) <= 4


def auth_control(description) -> str:
    """Why the click target ``description`` is a sign-in control
    (ACCOUNT_ENTRY, SIGN_IN_CONTROL), or "" when it is not one."""
    try:
        d = " ".join(str(description or "").split())
        if not d:
            return ""
        if _is_account_entry(d):
            return ACCOUNT_ENTRY
        if _FILE_RE.search(d):
            return ""
        if _AUTH_CONTROL_RE.search(d):
            return SIGN_IN_CONTROL
    except Exception:
        return ""
    return ""


def _screen_lines(screen_texts) -> list:
    out = []
    for x in screen_texts or ():
        s = " ".join(str(x or "").split())
        if s and not _SIGNED_IN_RE.search(s):
            out.append(s)
    return out


def auth_page(urls: Iterable = (), titles: Iterable = (),
              screen_texts: Iterable = ()) -> str:
    """Why the page in front is a sign-in page, from the URLs it was opened
    at, the window titles showing it and what a look at the screen this
    turn said about it; "" when none of them says so."""
    try:
        for u in urls or ():
            s = str(u or "")
            if s and (_AUTH_HOST_RE.search(s) or _AUTH_PATH_RE.search(s)):
                return "its address is a sign-in page"
        for t in titles or ():
            s = " ".join(str(t or "").split())
            if not s:
                continue
            parts = [p.strip() for p in _TITLE_SPLIT_RE.split(s) if p.strip()]
            head = parts[0] if parts else ""
            if ((head and _TITLE_HEAD_RE.match(head)
                    and not _FILE_RE.search(head))
                    or any(_TITLE_PART_RE.match(p) for p in parts)
                    or _AUTH_HOST_RE.search(s)):
                return "its title is a sign-in page"
        for s in _screen_lines(screen_texts):
            if _AUTH_SCREEN_RE.search(s):
                return "the screen shows a sign-in page"
    except Exception:
        return ""
    return ""


def auth_overlay(screen_texts: Iterable = ()) -> str:
    """Why a look at the screen this turn saw a sign-in POP-UP over the page
    ("Sign in with Google", "Continue with Google"), or ""."""
    try:
        for s in _screen_lines(screen_texts):
            if _AUTH_OVERLAY_RE.search(s):
                return "a sign-in pop-up is on the screen"
    except Exception:
        return ""
    return ""


def _imperative_clauses(owner_text, lead_re) -> list:
    """The clauses of ``owner_text`` that are a command to JARVIS starting
    with a verb ``lead_re`` matches ("click X", "Jarvis, press enter", "can
    you pick my account"); [] when the owner said not to click / type / sign
    in at all."""
    said = str(owner_text or "")
    if not said.strip() or _NEGATED_INPUT_RE.search(said):
        return []
    out = []
    for c in _CLAUSE_SPLIT_RE.split(said):
        c = " ".join(c.split())
        c = _CLAUSE_LEAD_RE.sub("", c + " ").strip()
        if c and lead_re.match(c):
            out.append(c)
    return out


def owner_asked_for_click(owner_text, description="") -> bool:
    """True when the owner's own words this turn ask for this exact click:
    an imperative clause that starts with a click verb and, in that same
    clause, names the target ("click Continue with Google") - for an account
    entry "my / the account" or its name; for a click with no target
    (coordinates), a sign-in word ("click sign in")."""
    try:
        target = " ".join(str(description or "").split())
        for c in _imperative_clauses(owner_text, _CLICK_LEAD_RE):
            if not target:
                if _AUTH_CONTROL_RE.search(c) or _ACCOUNT_WORD_RE.search(c):
                    return True
                continue
            said = _words(c)
            if _EMAIL_RE.search(target):
                if (_ACCOUNT_WORD_RE.search(c)
                        or said & _words(_EMAIL_RE.sub(" ", target))
                        or any(e.lower() in c.lower()
                               for e in _EMAIL_RE.findall(target))):
                    return True
                continue
            if said & _words(target):
                return True
    except Exception:
        return False
    return False


def owner_asked_for_input(owner_text, kind, value="") -> bool:
    """True when the owner's own words this turn ask for this input: "type
    <it>" / "enter my email" (``kind`` "type") or "press enter" (``kind``
    "press" / "hotkey", ``value`` the key or keys)."""
    try:
        v = str(value or "")
        if kind == "type":
            for c in _imperative_clauses(owner_text, _TYPE_LEAD_RE):
                if (_words(c) & _words(v) or _LOGIN_WORD_RE.search(c)
                        or (v.strip() and v.strip().lower() in c.lower())):
                    return True
            return False
        keys = [k.strip().lower() for k in re.split(r"[+\s,]+", v)
                if k.strip()]
        names = {n for k in keys for n in _SUBMIT_KEYS.get(k, (k,))}
        for c in _imperative_clauses(owner_text, _KEY_LEAD_RE):
            if set(re.findall(r"[a-z]+", c.lower())) & names:
                return True
    except Exception:
        return False
    return False


def is_submit_key(value) -> bool:
    """True when key / hotkey ``value`` ("enter", "ctrl+enter", "tab")
    includes a key that submits or moves through a form."""
    try:
        return any(k.strip().lower() in _SUBMIT_KEYS
                   for k in re.split(r"[+\s,]+", str(value or "")))
    except Exception:
        return False


def _line(page_or_overlay: bool, control: str) -> str:
    if page_or_overlay or control == ACCOUNT_ENTRY or not control:
        return TERMINAL_FAILURE_PREFIX + READY_LINE
    return TERMINAL_FAILURE_PREFIX + CONTROL_LINE


def click_refusal(description="", owner_text="", urls: Iterable = (),
                  titles: Iterable = (), screen_texts: Iterable = (),
                  looked_for: Iterable = (), refused_before=False) -> str:
    """The TERMINAL refusal for a click JARVIS must not make - see the module
    docstring - that the owner did not ask for this turn; "" when the click
    may go ahead. ``looked_for``: what this turn's find_on_screen looked
    for; ``refused_before``: an input was already refused this turn. Never
    raises."""
    try:
        desc = " ".join(str(description or "").split())
        control = auth_control(desc)
        page = auth_page(urls, titles, screen_texts)
        overlay = "" if page else auth_overlay(screen_texts)
        looked = (not desc) and any(auth_control(x) for x in looked_for or ())
        if not (control or page or refused_before
                or (not desc and (overlay or looked))):
            return ""
        if owner_asked_for_click(owner_text, desc):
            return ""
        return _line(bool(page or overlay or looked or refused_before),
                     control)
    except Exception:
        return ""


def input_refusal(kind, value="", owner_text="", urls: Iterable = (),
                  titles: Iterable = (), screen_texts: Iterable = (),
                  refused_before=False) -> str:
    """The TERMINAL refusal for typing (``kind`` "type") or a submit key
    (``kind`` "press" / "hotkey": Enter, Tab, Space) on a sign-in page or
    under a sign-in pop-up - or after an input was refused this turn - that
    the owner did not ask for this turn; "" otherwise. Other keys (volume,
    media, arrows) are never refused. Never raises."""
    try:
        if kind != "type":
            if not is_submit_key(value):
                return ""
        elif not str(value or "").strip():
            return ""
        if not (refused_before or auth_page(urls, titles, screen_texts)
                or auth_overlay(screen_texts)):
            return ""
        if owner_asked_for_input(owner_text, kind, value):
            return ""
        return TERMINAL_FAILURE_PREFIX + READY_LINE
    except Exception:
        return ""
