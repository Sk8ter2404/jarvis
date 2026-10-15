"""core/send_guard.py - a hard stop before send, submit, post, pay, delete
(2026-10-15).

WHY THIS EXISTS
===============
JARVIS's screen control (core.grounded_click: UI Automation first, OCR
fallback, "checks every click") can press any control it can read. Most are
harmless. A few cannot be taken back or reach other people: Send, Submit,
Post, Publish, Pay, Buy, Place order, Transfer, Delete, Accept terms, Sign.
The v2.0.181 sign-in guard (core.auth_guard) already keeps JARVIS off the
owner's sign-in page; this module is its sibling for the OTHER irreversible
controls, and it reuses the confirmation flow that already exists
(core.grounded_click's pending "confirm" question, answered by the owner's
NEXT turn through core.dispatcher.screen_route -> click_on_screen, pick:1).

THE RULE
========
When a click (or Enter in a message box that would send) is about to go to a
control whose purpose is one of the classes below, NOTHING is clicked or
typed. JARVIS says one short honest sentence (``hold_line``) and waits. Only
the owner's explicit spoken yes ("yes", "yes, send it", "go ahead") on his
next turn does it - ONE yes is ONE click, the question is closed once
answered (or when another turn passes), and a second click needs a new yes.
His own words earlier ("click Send") do not waive the stop.

WHAT COUNTS (``classify``)
==========================
The control's accessible name, automation id, role, and the words near it
(the neighbouring controls, the window title) are normalised (lower case, no
punctuation, "&" = "and", a trailing "(Alt+S)" or leading "Click to" /
"Yes," dropped) and judged by a WORD-BOUNDARY vocabulary anchored at the
START of the label - the control's verb, not any word in it:

  send      Send, Send message / email / reply / invite / feedback
  submit    Submit, Submit order / form / application / answers
  post      Post, Post reply / comment / now
  publish   Publish, Publish now / site / changes
  pay       Pay, Pay now / $12.00 / invoice; Complete / Make / Authorize
            payment
  order     Buy, Buy now, Purchase, Place (your) order, Order now, Donate,
            Checkout / Secure checkout / Proceed to checkout
  transfer  Transfer, Transfer funds, Send money, Withdraw, Wire
  delete    Delete, Delete forever / account / all, Erase, Wipe, Discard,
            Uninstall, Remove permanently / forever / all / account, a bare
            "Remove", Empty trash / recycle bin, Format drive, Factory
            reset, Deactivate, Cancel subscription / membership
  terms     Accept, I accept, Agree, I agree, Agree and continue, Sign,
            Sign document / here / now, Sign up, Create account
  confirm   Confirm, Confirm payment / order (a bare "Confirm" is as
            outward-facing as the thing it confirms)
  account   Sign out, Log out, Unsubscribe (kept from the old hold)

An OK / Yes / Continue / Proceed / Done / Finish / Apply / Go button has no
verb of its own: it is held only when the words NEAR it say what it does
("Place your order", "cannot be undone", "permanently delete this file",
"Send this message", "Terms of service").

FALSE-POSITIVE RULE (decided, documented, tested)
=================================================
  * Only the label's FIRST verb counts: "Submittal folder", "Sender",
    "Postal codes", "Post office", "Payment methods", "Purchase history",
    "Order history", "Transfer files", "New post", "Send/Receive" and
    "Unpublish" are not held - and neither is anything that merely CONTAINS
    a trigger word later ("Open", "Next page", "Search", "Back").
  * "Send to back / front / backward / forward" (a drawing program's
    layering menu) and a bare "Send to" (Explorer's submenu) reach nobody:
    NOT held. "Accept changes / suggestion" (a word processor's review pane)
    and "Remove from playlist / filter / queue / cart" are not held either.
  * Buttons, menu items and split buttons are judged on the whole label.
    Anything else - a link, a list item, a tab, OCR text of a paragraph
    line - is a plain navigation or a sentence unless it is SHORT (four
    words or fewer) with no article / pronoun ("Post a job", "Send us a
    message", "Pay your bill" lead somewhere; "Send" or "Delete account"
    alone do not).
  * The sign-in family ("Sign in", "Log in", "Continue with Google") is not
    here: core.auth_guard owns it.

ENTER (``enter_would_send``)
============================
Plain Enter or Ctrl+Enter (Shift/Alt+Enter is a new line) is held when the
window in front is a chat or mail compose (title names Discord, Slack,
WhatsApp, Telegram, Messenger, Skype, Teams, Signal; a "New Message", "Re:",
"Fwd:" or "Message (HTML)" window) or this turn's look at the screen said a
message / compose / reply box is open. Anywhere else Enter is untouched: the
code cannot tell, and a search box must keep working.

Pure stdlib, no I/O, never raises.
"""
from __future__ import annotations

import re
from typing import Iterable

from core.failure_markers import TERMINAL_FAILURE_PREFIX

__all__ = [
    "ACCOUNT", "CONFIRM", "DELETE", "ORDER", "PAY", "POST", "PUBLISH",
    "SEND", "SUBMIT", "TERMS", "TRANSFER",
    "classify", "classify_description", "enter_would_send", "hold_line", "held_text", "is_yes_answer",
    "nearby_texts",
]

SEND, SUBMIT, POST, PUBLISH = "send", "submit", "post", "publish"
PAY, ORDER, TRANSFER = "pay", "order", "transfer"
DELETE, TERMS, CONFIRM, ACCOUNT = "delete", "terms", "confirm", "account"

# kind -> (what the control does, the words that approve it)
_LINES = {
    SEND: ("sends the message", "yes, send it"),
    SUBMIT: ("submits the form", "yes, submit it"),
    POST: ("posts it for others to see", "yes, post it"),
    PUBLISH: ("publishes it", "yes, publish it"),
    PAY: ("makes a payment", "yes, pay it"),
    ORDER: ("places an order", "yes, place it"),
    TRANSFER: ("moves money", "yes, transfer it"),
    DELETE: ("deletes something", "yes, delete it"),
    TERMS: ("accepts terms or signs up", "yes, accept it"),
    CONFIRM: ("confirms something that may not be undoable",
              "yes, confirm it"),
    ACCOUNT: ("signs you out or unsubscribes you", "yes, do it"),
}


def hold_line(kind, action="click") -> str:
    """The one honest sentence JARVIS says instead of acting, as a TERMINAL
    failure (core.failure_markers: spoken word for word, the follow-up chain
    stops). ``action`` "click" or "press"."""
    what, yes = _LINES.get(kind, _LINES[CONFIRM])
    if action == "press":
        return (TERMINAL_FAILURE_PREFIX + f"Enter would send that message, "
                f"sir - say '{_LINES[SEND][1]}' and I'll press it.")
    return (TERMINAL_FAILURE_PREFIX + f"That button {what}, sir - say "
            f"'{yes}' and I'll click it.")


def held_text(text) -> str:
    """``text`` without the terminal prefix (for matching in tests/logs)."""
    s = str(text or "")
    return s[len(TERMINAL_FAILURE_PREFIX):] if s.startswith(
        TERMINAL_FAILURE_PREFIX) else s


# ── normalising ─────────────────────────────────────────────────────────────
_PAREN_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_NONWORD_RE = re.compile(r"[^a-z0-9$/\s]")
_LEAD_RE = re.compile(
    r"^(?:(?:yes|ok|okay|please|now|and|then|just|click|tap|press|here|"
    r"continue\s+to|proceed\s+to|go\s+to|to)\s+)+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _norm(text) -> str:
    s = _PAREN_RE.sub(" ", str(text or ""))
    s = s.replace("&", " and ").replace("_", " ").replace("-", " ")
    s = _NONWORD_RE.sub(" ", s.lower())
    s = " ".join(s.split())
    # a lead-in ("Yes, ", "Click to ", "Proceed to ") is not the verb; a
    # label that is ONLY a lead-in ("Yes", "Go") stays as it is
    return _LEAD_RE.sub("", s).strip() or s


def _id_words(auto_id) -> str:
    """An automation id as words: "btnSendMessage" -> "send message",
    "submit-button" -> "submit". Empty for ids that are only noise."""
    s = str(auto_id or "").strip()
    if not s or len(s) > 60:
        return ""
    s = _CAMEL_RE.sub(" ", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    noise = {"btn", "button", "id", "cmd", "action", "control", "link", "ctl",
             "el", "elem", "element", "ui", "the", "main", "primary"}
    words = [w for w in s.split() if w not in noise and not w.isdigit()]
    return " ".join(words[:4])


# ── the vocabulary ──────────────────────────────────────────────────────────
# Labels that name something else although they start with a trigger verb.
_EXEMPT_RE = re.compile(
    r"^(?:"
    r"send\s+to\s+(?:back|front|bottom|top)\b|send\s+to$|"
    r"send\s+(?:backward|backwards|forward|forwards)\b|"
    r"send\s*/\s*receive|send\s+and\s+receive|"
    r"post\s+(?:office|code|codes|box|card|cards|script|production|"
    r"processing|mortem|merge|install|graduate|doctoral)\b|"
    r"pay\s+(?:attention|stub|stubs|slip|period|rate|grade|scale|history|"
    r"roll|schedule)\b|"
    r"accept\s+(?:changes?|all\s+changes|suggest\w*|edits?|autocomplete|"
    r"revisions?|insertion|deletion)\b|"
    r"remove\s+(?:from|filter|filters|formatting|highlight|link|border|"
    r"borders|shading|background|duplicates|tag|tags|label)\b|"
    r"purchase\s+(?:history|orders?|receipts?|details|options?|info\w*|"
    r"date|order\s+number)\b|"
    r"order\s+(?:history|status|details|number|summary|by|confirmation)\b|"
    r"transfer\s+(?:files?|data|history|list|speed|rate|settings)\b|"
    r"buy\s+(?:a\s+)?gift\s+card\b"
    r")", re.IGNORECASE)

_RULES = (
    # money first, so "send money" is a transfer, not a message
    (TRANSFER, re.compile(
        r"^(?:send\s+(?:money|funds|cash|payment)|transfer\b|wire\b|"
        r"withdraw\b|make\s+(?:a\s+)?transfer|move\s+money)")),
    (SUBMIT, re.compile(r"^submit\b")),
    (PAY, re.compile(
        r"^(?:pay\b|(?:complete|make|authori[sz]e|approve|confirm|"
        r"finish)\s+(?:a\s+|the\s+|your\s+|my\s+|this\s+)?"
        r"(?:payment|transaction)\b)")),
    (ORDER, re.compile(
        r"^(?:buy\b|purchase\b|pre\s?order\b|donate\b|"
        r"place\s+(?:your\s+|my\s+|the\s+|this\s+)?order\b|order\s+now\b|"
        r"order\s+and\s+pay\b|(?:complete|confirm|finish|approve)\s+"
        r"(?:the\s+|your\s+|my\s+|this\s+)?(?:order|purchase|booking|"
        r"reservation|subscription)\b|"
        r"(?:(?:secure|guest|express)\s+)?check\s?out"
        r"(?:\s+(?:now|securely|as\s+guest|with\s+\w+))?$)")),
    (POST, re.compile(r"^(?:post|tweet)\b")),
    (PUBLISH, re.compile(r"^publish\b")),
    (SEND, re.compile(
        r"^send\b|\b(?:and|then|schedule)\s+send\b|"
        r"\b(?:and|then)\s+(?:submit|post|publish)\b")),
    (DELETE, re.compile(
        r"^(?:delete\b|erase\b|wipe\b|discard\b|uninstall\b|deactivate\b|"
        r"permanently\s+(?:delete|remove|erase)\b|"
        r"remove\s+(?:permanently|forever|all|everything|account|profile|"
        r"user|member|device|data|files?)\b|remove$|"
        r"empty\s+(?:the\s+)?(?:trash|recycle\s+bin|bin)\b|"
        r"format\s+(?:disk|drive|volume|partition|usb|sd|device)\b|"
        r"factory\s+reset\b|close\s+account\b|"
        r"cancel\s+(?:my\s+|your\s+|the\s+)?(?:subscription|membership|"
        r"order|account)\b)")),
    (TERMS, re.compile(
        r"^(?:accept\b|i\s+accept\b|agree\b|i\s+agree\b|e\s?sign\b|"
        r"sign\s+(?:document|here|now|and|contract|agreement|form|pdf|it|all|"
        r"up|electronically)\b|sign$|sign\s+up$|create\s+(?:an\s+|your\s+)?"
        r"account\b|join\s+now$)")),
    (CONFIRM, re.compile(r"^confirm\b")),
    (ACCOUNT, re.compile(
        r"^(?:unsubscribe\b|sign\s?out\b|log\s?out\b|logout\b)")),
)

# Roles judged on the whole label (a button's name IS its verb).
_BUTTON_ROLES = frozenset({"button", "splitbutton", "menuitem", "checkbox",
                           "togglebutton", "pushbutton"})
# Words that make a short label a sentence ("Send us a message").
_SENTENCE_WORDS = frozenset(
    "a an us your our the my me you we i for about of what how why when who "
    "is are can that this these those it".split())
_MAX_LOOSE_WORDS = 4

# OK / Yes / Continue ...: no verb of their own.
_GENERIC_RE = re.compile(
    r"^(?:ok|okay|yes|continue|proceed|finish|i\s+understand)$")
_NEARBY = (
    (ORDER, re.compile(
        r"\b(?:place\s+(?:your\s+)?order|complete\s+(?:your\s+)?purchase)"
        r"\b")),
    (PAY, re.compile(
        r"\b(?:pay\s+now|you\s+will\s+be\s+charged|will\s+be\s+charged)\b")),
    (DELETE, re.compile(
        r"\b(?:permanently\s+delete\w*|delete\s+(?:this|these|all|your)\b|"
        r"(?:cannot|can'?t|can\s+not)\s+be\s+(?:undone|recovered|restored)|"
        r"(?:will\s+be|is|are)\s+(?:permanently\s+)?deleted)")),
    (SEND, re.compile(
        r"\b(?:send\s+(?:this|the|your)\s+(?:message|e-?mail|reply|"
        r"invitation)|ready\s+to\s+send)\b")),
    (TERMS, re.compile(
        r"\b(?:by\s+(?:clicking|continuing|signing\s+up|creating)|"
        r"i\s+agree\s+to|accept\s+the\s+terms)\b")),
)


def _kind_of(s: str) -> str:
    if not s or _EXEMPT_RE.match(s):
        return ""
    for kind, rx in _RULES:
        if rx.search(s):
            return kind
    return ""


def _is_short(s: str) -> bool:
    words = s.split()
    return (len(words) <= _MAX_LOOSE_WORDS
            and not any(w in _SENTENCE_WORDS for w in words))


def classify(label="", role="", auto_id="", nearby: Iterable = ()) -> str:
    """The class of irreversible / outward-facing action the control does
    (SEND, SUBMIT, POST, PUBLISH, PAY, ORDER, TRANSFER, DELETE, TERMS,
    CONFIRM, ACCOUNT), or "" for an ordinary control. ``label``: its
    accessible name (or OCR text); ``role``: its UIA control type ("Button",
    "Hyperlink", "ocr" ...); ``auto_id``: its automation id; ``nearby``:
    the names of the neighbouring controls / the window title. See the
    module docstring for the vocabulary and the false-positive rule. Never
    raises."""
    try:
        r = str(role or "").strip().lower()
        strict = (not r) or r in _BUTTON_ROLES
        for source, text in (("name", label), ("id", _id_words(auto_id))):
            s = _norm(text)
            if not s:
                continue
            kind = _kind_of(s)
            if kind and (strict or source == "id" or _is_short(s)):
                return kind
        s = _norm(label)
        if s and _GENERIC_RE.match(s) and (strict or _is_short(s)):
            near = " ".join(_norm_near(x) for x in (nearby or ()) if x)
            for kind, rx in _NEARBY:
                if rx.search(near):
                    return kind
    except Exception:
        return ""
    return ""


_DESC_LEAD_RE = re.compile(
    r"^(?:(?:the|a|an|that|this|my|your|green|blue|red|big|small|top|bottom|"
    r"bottom\s+right|top\s+right)\s+)+")
_DESC_TAIL_RE = re.compile(
    r"\s+(?:button|link|icon|option|key|control|item)$")


def classify_description(description) -> str:
    """The class of a spoken description of a control ("the Send button",
    "that green Pay now link") - for the one path that has only words and
    no control to read. Judged as a button's name. Never raises."""
    try:
        s = _norm(description)
        s = _DESC_TAIL_RE.sub("", _DESC_LEAD_RE.sub("", s)).strip()
        return classify(s, "Button")
    except Exception:
        return ""


def _norm_near(text) -> str:
    return " ".join(re.sub(r"[^a-z0-9$'\s]", " ",
                           str(text or "").lower()).split())


def nearby_texts(target_rect, cands, own_text="", pad=140.0, limit=8) -> list:
    """The names of the candidates (dicts with "text" and "rect") within
    ``pad`` px of ``target_rect`` (x, y, w, h), except the target itself:
    the words around a control, for ``classify``'s ``nearby``. Never
    raises."""
    out = []
    try:
        x, y, w, h = (float(v) for v in target_rect)
        for c in cands or ():
            t = " ".join(str((c or {}).get("text") or "").split())
            r = (c or {}).get("rect")
            if not t or t == own_text or not r or len(r) < 4:
                continue
            cx, cy, cw, ch = (float(v) for v in r[:4])
            if (cx < x + w + pad and cx + cw > x - pad
                    and cy < y + h + pad and cy + ch > y - pad):
                out.append(t)
            if len(out) >= limit:
                break
    except Exception:
        return out
    return out


# ── Enter in a message box ──────────────────────────────────────────────────
_ENTER_KEYS = ("enter", "return")
_COMPOSE_TITLE_RE = re.compile(
    r"\b(?:discord|slack|whatsapp|telegram|messenger|imessage|skype|"
    r"groupme|viber|google\s+chat|hangouts|microsoft\s+teams|"
    r"direct\s+message|new\s+message|new\s+e-?mail|compose\s+(?:mail|"
    r"message|e-?mail)|message\s*\((?:html|plain\s+text|rich\s+text)\))|"
    r"^\s*(?:compose|(?:re|fwd?)\s*:)|(?:^|\s[-–—|]\s)signal\s*$",
    re.IGNORECASE)
_COMPOSE_SCREEN_RE = re.compile(
    r"\b(?:(?:message|compose|reply|chat)\s+(?:box|field|input|window|bar)|"
    r"type\s+a\s+message|write\s+a\s+(?:message|reply)|"
    r"compos(?:e|ing)\s+(?:an?\s+)?(?:e-?mail|message|reply)|"
    r"new\s+(?:e-?mail|message)\s+(?:window|draft))\b", re.IGNORECASE)


def enter_would_send(value, titles: Iterable = (),
                     screen_texts: Iterable = ()) -> str:
    """SEND when key / hotkey ``value`` is a plain Enter or Ctrl+Enter and a
    chat or mail compose is in front (see the module docstring); "" for any
    other key or any other window. Never raises."""
    try:
        keys = [k.strip().lower() for k in re.split(r"[+\s,]+", str(value or ""))
                if k.strip()]
        if not any(k in _ENTER_KEYS for k in keys):
            return ""
        if any(k in ("shift", "alt") for k in keys):
            return ""                       # a new line, not a send
        for t in titles or ():
            if _COMPOSE_TITLE_RE.search(" ".join(str(t or "").split())):
                return SEND
        for s in screen_texts or ():
            if _COMPOSE_SCREEN_RE.search(" ".join(str(s or "").split())):
                return SEND
    except Exception:
        return ""
    return ""


# ── the owner's yes ─────────────────────────────────────────────────────────
_YES_RE = re.compile(
    r"^(?:(?:yes|yeah|yep|yup|sure|ok|okay|affirmative|go\s+ahead)"
    r"(?:[\s,]+(?:please|sir|jarvis))*"
    r"(?:[\s,]+(?:go\s+ahead|do\s+it|(?:send|submit|post|publish|pay|buy|"
    r"purchase|order|place|delete|remove|erase|accept|agree|sign|confirm|"
    r"transfer|click|press|tweet)(?:\s+(?:it|that|this|them|the\s+\w+))?))?"
    r"|do\s+it)(?:[\s,]+(?:please|sir|jarvis|now))*$", re.IGNORECASE)


def is_yes_answer(text) -> bool:
    """True when ``text`` is an explicit spoken yes to a hard stop: "yes",
    "yeah go ahead", "yes, send it", "do it". NOT "please", "right",
    "correct" or "that one" (fine for "which one?", too weak for a payment),
    and nothing longer ("yes, send it to Dana instead" is a new request)."""
    try:
        s = " ".join(re.sub(r"[^\w\s,']", " ", str(text or "").lower()).split())
        return bool(s) and bool(_YES_RE.match(s))
    except Exception:
        return False
