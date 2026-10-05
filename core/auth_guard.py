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
A description click (or a coordinate click) is REFUSED when

  * its target is itself a sign-in control: an e-mail address (an account
    chooser's entry), "Sign in" / "Log in", "Continue with Google", "Choose
    an account", a password / passkey / verification-code control, an
    "Authorize" / "Allow access" / consent button - ``auth_control``; or
  * the page it would land on is a sign-in page: its URL (accounts.google.com,
    login.microsoftonline.com, ``/login``, ``/signin``, ``/oauth`` ...), its
    window title ("Sign in - Google Accounts", "Log in to ..."), or what a
    look at the screen this turn said about it ("Choose an account", "to
    continue to", "Enter your password", "a login page") - ``auth_page``;

unless the owner's OWN words this turn ask for that exact click: a click verb
("click", "press", "tap", "select", "choose", "pick", "hit") AND the target's
own words ("click Continue with Google"), or "my account" / "the account" for
an account entry - ``owner_asked_for_click``. "So I can sign in" is not a
request for JARVIS to click anything.

The refusal is a TERMINAL line (core.failure_markers), so the follow-up chain
stops on it and it is spoken word for word: the page is up and ready for him.

Deliberately narrow: an ordinary page that merely has a "Sign in" link in its
header is not a sign-in page, and a click on anything else there is not
blocked. Pure stdlib, no I/O, never raises.
"""
from __future__ import annotations

import re
from typing import Iterable

from core.failure_markers import TERMINAL_FAILURE_PREFIX

__all__ = [
    "READY_LINE",
    "auth_control",
    "auth_page",
    "click_refusal",
    "owner_asked_for_click",
]

READY_LINE = ("The sign-in page is up and ready for you, sir - I'll leave the "
              "signing in to you.")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# A control whose whole job is signing someone in, choosing an identity,
# proving it, or granting an app access to it.
_AUTH_CONTROL_RE = re.compile(
    r"\b(?:"
    r"sign[\s-]?in|signin|log[\s-]?in|login|log[\s-]?on|"
    r"continue\s+(?:with|as)\s+\w+|"
    r"(?:choose|select|pick|use)\s+(?:an?\s+|another\s+|your\s+|my\s+|"
    r"the\s+|this\s+)?(?:\w+\s+)?account|"
    r"account\s+(?:chooser|picker|entry|tile|row|switcher)|"
    r"(?:google|microsoft|apple|github|anthropic)\s+account|"
    r"password|passcode|passkey|"
    r"two[\s-]?factor|2fa|mfa|verification\s+code|one[\s-]?time\s+code|"
    r"security\s+key|authori[sz]e|grant\s+access|allow\s+access|consent"
    r")\b", re.IGNORECASE)

# Sign-in hosts and paths (a URL the page was opened at, or a title that
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
# Window titles of sign-in pages ("Sign in - Google Accounts - Google
# Chrome", "Log in to <service>", "Login | <service>").
_AUTH_TITLE_RE = re.compile(
    r"^\s*(?:sign\s*in|log\s*in|login|choose\s+an\s+account)\b|"
    r"\bgoogle\s+accounts\b|"
    r"\b(?:sign|log)\s*in\s+(?:to|with)\b", re.IGNORECASE)
# What a look at the screen says about a sign-in PAGE. Strong, page-level
# phrases only: a "Sign in" link in a page header, or a "Sign in with Google"
# pop-up over an ordinary site, is not one (a click ON such a control is
# still caught by auth_control).
_AUTH_SCREEN_RE = re.compile(
    r"\b(?:"
    r"choose\s+an\s+account|use\s+another\s+account|account\s+chooser|"
    r"to\s+continue\s+to\b|enter\s+(?:your\s+)?password|"
    r"(?:sign[\s-]?in|log[\s-]?in|login)\s+(?:page|screen|form|gate|wall)|"
    r"(?:asking|prompting|wants?|requires?|requesting)\s+(?:you\s+|him\s+|"
    r"the\s+(?:user|owner|viewer)\s+)?to\s+(?:sign|log)\s+in|"
    r"(?:requesting|requires?|needs?)\s+(?:a\s+)?(?:login|sign[\s-]?in)"
    r")\b", re.IGNORECASE)
_SIGNED_IN_RE = re.compile(
    r"\b(?:already\s+(?:signed|logged)\s+in|(?:signed|logged)\s+in\s+as)\b",
    re.IGNORECASE)

_CLICK_VERB_RE = re.compile(
    r"\b(?:click|clicks|clicking|press|tap|hit|select|choose|pick)\b",
    re.IGNORECASE)
_ACCOUNT_WORD_RE = re.compile(
    r"\b(?:account|accounts|e-?mail|profile|identity)\b", re.IGNORECASE)
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


def auth_control(description) -> str:
    """Why the click target ``description`` is a sign-in control ("an
    account entry", "a sign-in control"), or "" when it is not one."""
    try:
        d = str(description or "")
        if not d.strip():
            return ""
        if _EMAIL_RE.search(d):
            return "an account entry"
        if _AUTH_CONTROL_RE.search(d):
            return "a sign-in control"
    except Exception:
        return ""
    return ""


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
            s = str(t or "")
            if s and (_AUTH_TITLE_RE.search(s) or _AUTH_HOST_RE.search(s)):
                return "its title is a sign-in page"
        for x in screen_texts or ():
            s = " ".join(str(x or "").split())
            if not s:
                continue
            if _AUTH_SCREEN_RE.search(s) and not _SIGNED_IN_RE.search(s):
                return "the screen shows a sign-in page"
    except Exception:
        return ""
    return ""


def owner_asked_for_click(owner_text, description="") -> bool:
    """True when the owner's own words this turn ask for this exact click: a
    click verb and the target's own words ("click Continue with Google"), or
    "my / the account" for an account entry. Without a target (a coordinate
    click) a click verb plus a sign-in word is needed ("click sign in")."""
    try:
        said = str(owner_text or "")
        if not _CLICK_VERB_RE.search(said):
            return False
        target = str(description or "")
        if not target.strip():
            return bool(_AUTH_CONTROL_RE.search(said))
        if _words(said) & _words(_EMAIL_RE.sub(" ", target)):
            return True
        return bool(_EMAIL_RE.search(target) and _ACCOUNT_WORD_RE.search(said))
    except Exception:
        return False


def click_refusal(description="", owner_text="", urls: Iterable = (),
                  titles: Iterable = (), screen_texts: Iterable = ()) -> str:
    """The TERMINAL refusal for a click JARVIS must not make (a sign-in
    control, or anything on a sign-in page) that the owner did not ask for
    this turn; "" when the click may go ahead. Never raises."""
    try:
        why = auth_control(description) or auth_page(urls, titles,
                                                      screen_texts)
        if not why or owner_asked_for_click(owner_text, description):
            return ""
        return TERMINAL_FAILURE_PREFIX + READY_LINE
    except Exception:
        return ""
