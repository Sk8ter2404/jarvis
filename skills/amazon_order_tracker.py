"""
Amazon order tracking skill for JARVIS.

Polls the owner's mailbox every POLL_INTERVAL_SECONDS, filters for Amazon
shipment notifications, and proactively announces status transitions
(shipped → out for delivery → delivered, plus delay warnings).

MAILBOX ACCESS (2026-10-12)
  1. Gmail, READ-ONLY: its own token, data/gmail_readonly_token.json, with the
     single scope https://www.googleapis.com/auth/gmail.readonly, made once by
         python -m skills.amazon_order_tracker --auth-gmail
     (it needs the OAuth client file gmail_credentials.json at the project
     root, the same one skills/email_triage uses). It can list and read mail;
     it cannot change, send or delete anything. Used first when it exists.
  2. Otherwise whatever skills/email_triage already has: its Gmail service, or
     Microsoft Graph (Outlook). Only fetches are made; nothing is marked read.
  With none of those the tracker does NOT guess: the voice actions answer
  "I need a Google sign-in first, sir" (or say what is wrong with the one that
  exists), the poller idles and re-checks every minute, and signing in later
  needs no restart. Nothing is spoken at boot about it.

FIRST RUN IS SILENT: the first poll only records what is already in the mailbox
(the last week of orders) so JARVIS does not read out three transitions per
old order; announcements start from the next change. A message older than
ANNOUNCE_MAX_AGE_S is recorded but not announced either.

State lives in data/amazon_order_state.json. Writes go through
core.atomic_io._atomic_write_json so concurrent skill reloads can't
corrupt it (same approach as skills/bambu_monitor.py).

Gated by ``AMAZON_TRACKING_ENABLED`` (core/config.py, default True: with no
sign-in it does nothing).

Actions registered:
  check_orders            — roll-call of in-flight orders (refreshes first).
  recent_delivery         — packages delivered in the last 5 days.
  amazon_tracking_status  — diagnostic: backend health + poller state.

Proactive speech (via bobert_companion.proactive_announce()):
  • 'Sir, Amazon order <id> has shipped.'
  • 'Sir, Amazon order <id> is out for delivery.'
  • 'Sir, your Amazon package — order <id> — has been delivered.'
  • 'Sir, Amazon order <id> appears to be delayed.'

When a 'shipped' notice carries a parsable estimated-arrival date in the
subject ('Arriving Mon, Jun 2'), the skill creates a ``time_at`` promise
via skill_utils["make_promise"] so JARVIS reminds the user on the
delivery day even if no follow-up email lands.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
import threading
import time

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json

_log = logging.getLogger(__name__)

# Poll cadence. Amazon emails aren't second-sensitive; 15 minutes keeps us
# well clear of Graph/Gmail rate limits while staying responsive enough
# that out-for-delivery → delivered transitions feel near-real-time.
POLL_INTERVAL_SECONDS  = 15 * 60
# With no usable sign-in the poller checks again this often (a file-exists
# test; no network), so signing in later takes effect without a restart.
NO_BACKEND_RECHECK_SECONDS = 60
INITIAL_DELAY_SECONDS  = 45
LOOKBACK_DAYS          = 7
STATE_MAX_ORDERS       = 200   # drop oldest delivered orders past this cap
PER_BACKEND_FETCH_LIMIT = 25
# A status email older than this is recorded but never spoken (the tracker was
# off, or the mailbox was only just connected).
ANNOUNCE_MAX_AGE_S     = 48 * 3600
# check_orders refreshes from the mailbox first, but never holds the voice
# loop longer than this.
REFRESH_WAIT_S         = 6.0

_STATE_FILE = os.path.join(_PROJECT_DIR, "data", "amazon_order_state.json")

# The tracker's own READ-ONLY Gmail credential (see the module docstring).
GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_READONLY_TOKEN_FILE = os.path.join(_PROJECT_DIR, "data",
                                         "gmail_readonly_token.json")
GMAIL_CLIENT_FILE = os.path.join(_PROJECT_DIR, "gmail_credentials.json")
SIGN_IN_COMMAND = "python -m skills.amazon_order_tracker --auth-gmail"

# Set by a fetcher that reached its mailbox and got an answer (even an empty
# one). _poll_once reads it to tell "nothing new" from "could not look".
_fetch_status = {"ok": False}

_state_lock  = threading.RLock()
_stop_evt    = threading.Event()
_poll_thread: list = [None]

# Higher rank = further along the pipeline. We only announce when a fresh
# email moves an order's rank strictly upward. 'delayed' is treated as a
# side-channel flag rather than a forward step so it can co-fire with any
# in-flight state without resetting the canonical status.
_STATUS_ORDER = {
    "unknown":          0,
    "ordered":          1,
    "shipped":          2,
    "out_for_delivery": 3,
    "delivered":        4,
}

# Subject + snippet keyword → status. Order matters: more-specific phrases
# first so 'out for delivery' doesn't get clobbered by the broader
# 'delivery' check below it. The SUBJECT is classified first and the snippet
# only when the subject says nothing (a "Shipped" mail whose preview reads "will
# be delivered by Friday" is shipped, not delivered).
_STATUS_RULES = [
    (re.compile(r"(?<!will be )(?<!to be )\bdelivered\b",   re.I), "delivered"),
    (re.compile(r"\bout for delivery\b",                    re.I), "out_for_delivery"),
    (re.compile(r"\barriving today\b",                      re.I), "out_for_delivery"),
    (re.compile(r"\b(shipped|on the way|on its way|"
                r"your package has shipped|shipment notification|"
                r"arriving (tomorrow|soon))\b",             re.I), "shipped"),
    (re.compile(r"\b(order (placed|confirmation)|"
                r"thank you for your order|"
                r"your amazon\.com order)\b|^\s*ordered\b", re.I), "ordered"),
]
# Delay markers are independent — they can co-occur with shipped / out for
# delivery without overriding them.
_DELAY_RE = re.compile(r"\bdelay(ed|s)?\b|running\s+late", re.I)

# Amazon order IDs are 3-7-7 dashed digits, e.g. 123-1234567-1234567.
_ORDER_ID_RE = re.compile(r"\b(\d{3}-\d{7}-\d{7})\b")

# Recognised Amazon sender domains. We accept anything with 'amazon' in the
# address or display name as a courtesy fallback (some carriers forward
# Amazon notices through their own domains).
_AMAZON_SENDER_RE = re.compile(
    r"@(?:[a-z0-9.-]*amazon[a-z0-9.-]*)\.[a-z.]+", re.I,
)

# 'Arriving Mon, Jun 2' or 'Arriving Monday, June 2' — pull weekday-prefixed
# month/day so we can schedule a follow-up promise. Year is inferred from
# the email's receivedDateTime (or now if that's missing), with a small
# nudge forward when the parsed month is far in the past.
_ARRIVING_RE = re.compile(
    r"Arriving\s+(?:[A-Za-z]+,?\s+)?"
    r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
    r"(?P<day>\d{1,2})",
    re.I,
)
_MONTH_INDEX = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


# ─── Config / shared helpers ────────────────────────────────────────────

def _read_feature_flag() -> bool:
    """AMAZON_TRACKING_ENABLED, read fresh. The monolith re-exports
    core.config, so its attribute is the same value; it is consulted first so
    a test (or an older build that still declares the flag in the monolith)
    keeps working, then core.config directly."""
    sentinel = object()
    try:
        bc = importlib.import_module("bobert_companion")
        val = getattr(bc, "AMAZON_TRACKING_ENABLED", sentinel)
        if val is not sentinel:
            return bool(val)
    except Exception:
        pass
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, "AMAZON_TRACKING_ENABLED", False))
    except Exception:
        return False


def _email_triage():
    """Lazy-import the email_triage helper module. Returns None if it
    isn't loaded yet (boot-order race) or if its import failed."""
    for name in ("skills.email_triage", "email_triage", "skill_email_triage"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def _proactive_announce(message: str) -> None:
    """Route through bobert_companion.proactive_announce(); print fallback
    so a missing parent module never silently drops an alert."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="amazon")
            return
    except Exception:
        pass
    print(f"  [amazon] (no announcer) {message}")


def _outlook_configured() -> bool:
    et = _email_triage()
    if not et:
        return False
    try:
        return bool(et._outlook_configured())
    except Exception:
        return False


def _gmail_configured() -> bool:
    """A Gmail credential exists: the tracker's own read-only token, or a
    Gmail service skills/email_triage already has. Existence only - it does
    not prove the token still works (see _readonly_problem)."""
    if _readonly_token_present():
        return True
    et = _email_triage()
    if not et:
        return False
    try:
        return bool(et.is_gmail_available())
    except Exception:
        return False


# ─── Gmail, READ-ONLY (the tracker's own credential) ─────────────────────

_ro_cache: list = [None]          # (service, creds) once built
_ro_problem: list = [""]          # why the read-only credential is unusable


def _readonly_token_present() -> bool:
    try:
        return os.path.exists(GMAIL_READONLY_TOKEN_FILE)
    except Exception:
        return False


def _google_libs():
    """(build, Credentials, Request, InstalledAppFlow) or None when the Google
    client libraries are not installed. Imported lazily: the skill must load
    and answer honestly without them."""
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except Exception:
            InstalledAppFlow = None
        return build, Credentials, Request, InstalledAppFlow
    except Exception:
        return None


def _readonly_credentials():
    """The read-only credentials, refreshed when expired, or None with
    _ro_problem saying why. Never raises, never opens a browser."""
    _ro_problem[0] = ""
    if not _readonly_token_present():
        _ro_problem[0] = "no read-only token yet"
        return None
    libs = _google_libs()
    if libs is None:
        _ro_problem[0] = ("the Google client libraries are not installed "
                          "(google-api-python-client, google-auth-oauthlib)")
        return None
    _build, Credentials, Request, _flow = libs
    try:
        creds = Credentials.from_authorized_user_file(
            GMAIL_READONLY_TOKEN_FILE, [GMAIL_READONLY_SCOPE])
    except Exception as e:
        _ro_problem[0] = f"the saved token is unreadable ({type(e).__name__})"
        return None
    try:
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            try:
                _atomic_write_json(GMAIL_READONLY_TOKEN_FILE,
                                   json.loads(creds.to_json()))
            except Exception as e:
                _log.debug("[amazon] token save failed: %s", e)
            return creds
    except Exception as e:
        # A revoked / expired-for-good refresh token lands here.
        _ro_problem[0] = f"Google refused the saved sign-in ({type(e).__name__})"
        return None
    _ro_problem[0] = "the saved sign-in has expired"
    return None


def _readonly_service():
    """A Gmail client on the read-only credential, or None. Cached while the
    credentials stay valid."""
    cached = _ro_cache[0]
    if cached is not None:
        service, creds = cached
        try:
            if creds.valid:
                return service
        except Exception:
            pass
        _ro_cache[0] = None
    creds = _readonly_credentials()
    if creds is None:
        return None
    libs = _google_libs()
    if libs is None:
        return None
    try:
        service = libs[0]("gmail", "v1", credentials=creds,
                          cache_discovery=False)
    except Exception as e:
        _ro_problem[0] = f"could not build the Gmail client ({type(e).__name__})"
        return None
    _ro_cache[0] = (service, creds)
    return service


def authenticate_gmail_readonly() -> bool:
    """One-time sign-in: opens the browser, asks for READ-ONLY Gmail access
    and stores the token at GMAIL_READONLY_TOKEN_FILE. Run from a terminal,
    never by voice."""
    libs = _google_libs()
    if libs is None or libs[3] is None:
        print("[amazon] Install the Google libraries first: "
              "pip install google-api-python-client google-auth-oauthlib")
        return False
    if not os.path.exists(GMAIL_CLIENT_FILE):
        print("[amazon] The OAuth client file gmail_credentials.json was not "
              f"found at {GMAIL_CLIENT_FILE}. Create a Desktop-app OAuth "
              "client in Google Cloud Console (Gmail API enabled), download "
              "its JSON and save it there, then run this again.")
        return False
    flow = libs[3].from_client_secrets_file(GMAIL_CLIENT_FILE,
                                            [GMAIL_READONLY_SCOPE])
    creds = flow.run_local_server(port=0)
    os.makedirs(os.path.dirname(GMAIL_READONLY_TOKEN_FILE), exist_ok=True)
    _atomic_write_json(GMAIL_READONLY_TOKEN_FILE, json.loads(creds.to_json()))
    _ro_cache[0] = None
    print(f"[amazon] Read-only Gmail token saved to {GMAIL_READONLY_TOKEN_FILE}")
    return True


def _backend_problem() -> str:
    """'' when a mailbox backend looks usable, else the SPOKEN reason it is not
    (the honest answer instead of an empty "no orders"). A sign-in that exists
    but is broken is reported as that, not as 'never signed in'. Runs on the
    voice thread, so it never refreshes a token or touches the network: what
    the poller last found (_ro_problem) is the evidence."""
    if _outlook_configured():
        return ""
    if _readonly_token_present():
        et = _email_triage()
        try:
            if et is not None and et.is_gmail_available():
                return ""
        except Exception:
            pass
        if _google_libs() is None:
            return ("I can't read your Amazon mail, sir - the Google client "
                    "libraries are not installed.")
        if _ro_problem[0]:
            return ("Your Google sign-in for Amazon tracking needs doing again, "
                    f"sir - {_ro_problem[0]}. Run {SIGN_IN_COMMAND} once.")
        return ""
    if _gmail_configured():
        return ""
    return ("I need a Google sign-in first, sir. Run "
            f"{SIGN_IN_COMMAND} once (it asks for read-only Gmail access) and "
            "I'll watch for your Amazon orders.")


def backend_problem() -> str:
    """'' when the tracker can read mail, else the reason it cannot - the
    answer it gives when he ASKS, sign-in command and all. A caller that speaks
    unprompted (the morning briefing, skills/daily_briefing/arrival_v2.py)
    says nothing about orders when this is non-empty."""
    return _backend_problem()


# ─── State persistence ──────────────────────────────────────────────────

def _empty_state() -> dict:
    return {"orders": {}, "last_poll_ts": 0.0, "last_error": ""}


def _load_state() -> dict:
    if not os.path.exists(_STATE_FILE):
        return _empty_state()
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            return _empty_state()
        data.setdefault("orders", {})
        data.setdefault("last_poll_ts", 0.0)
        data.setdefault("last_error", "")
        if not isinstance(data["orders"], dict):
            data["orders"] = {}
        return data
    except Exception as e:
        _log.debug("[amazon] state load failed: %s", e)
        s = _empty_state()
        s["last_error"] = str(e)
        return s


def _save_state(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        orders = state.get("orders") or {}
        if len(orders) > STATE_MAX_ORDERS:
            kept = sorted(orders.items(),
                          key=lambda kv: kv[1].get("last_seen_ts", 0),
                          reverse=True)[:STATE_MAX_ORDERS]
            state["orders"] = dict(kept)
        _atomic_write_json(_STATE_FILE, state)
    except Exception as e:
        _log.warning("[amazon] state save failed: %s", e)


# ─── Email parsing ──────────────────────────────────────────────────────

def _parse_iso(s: str) -> float:
    if not s:
        return 0.0
    import datetime as _dt
    iso = s.replace("Z", "+00:00")
    try:
        return _dt.datetime.fromisoformat(iso).timestamp()
    except Exception:
        pass
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).timestamp()
    except Exception:
        return 0.0


def _classify(subject: str, snippet: str) -> str:
    for text in (subject or "", snippet or ""):
        for rx, status in _STATUS_RULES:
            if rx.search(text):
                return status
    return "unknown"


def _is_delayed(subject: str, snippet: str) -> bool:
    return bool(_DELAY_RE.search(f"{subject}\n{snippet}"))


def _extract_order_id(subject: str, snippet: str) -> str | None:
    for src in (subject, snippet):
        m = _ORDER_ID_RE.search(src or "")
        if m:
            return m.group(1)
    return None


# Amazon puts the item in the subject, in quotes: Shipped: "Anker 65W Charger".
# Its mails rarely carry the order number in the subject or the preview, so the
# item is what ties "Shipped" to "Out for delivery" to "Delivered".
_ITEM_RE = re.compile(r'[“"]([^”"]{3,160})[”"]')


def _extract_item(subject: str) -> str:
    m = _ITEM_RE.search(subject or "")
    if not m:
        return ""
    item = " ".join(m.group(1).split())
    if len(item) > 60:
        item = item[:60].rsplit(" ", 1)[0].rstrip(" ,-:") + "..."
    return item


def _order_key(subject: str, snippet: str, msg_id: str) -> str:
    """The key an order is tracked under: its number when a mail carries one,
    else the item named in the subject, else the single message."""
    oid = _extract_order_id(subject, snippet)
    if oid:
        return oid
    item = _extract_item(subject)
    if item:
        return "item:" + item.lower()[:80]
    return f"msg:{msg_id[:24]}"


def _order_label(order_id: str, entry: dict) -> str:
    """How an order is named aloud: its number, else the item in quotes, else a
    generic phrase (never the synthetic key)."""
    if order_id.startswith("item:"):
        item = (entry or {}).get("item") or order_id[5:]
        return "“" + item + "”"
    if order_id.startswith("msg:"):
        return "an Amazon order"
    return order_id


def _extract_estimated_delivery(text: str, ref_ts: float = 0.0) -> float | None:
    if not text:
        return None
    m = _ARRIVING_RE.search(text)
    if not m:
        return None
    try:
        day = int(m.group("day"))
        mon_key = m.group("month")[:3].title()
        month = _MONTH_INDEX.get(mon_key)
        if not month or not (1 <= day <= 31):
            return None
    except Exception:
        return None
    base = time.localtime(ref_ts) if ref_ts else time.localtime()
    year = base.tm_year
    # Roll year forward when the parsed month is well behind 'now' — handles
    # December emails parsed in early January.
    if month < base.tm_mon - 2:
        year += 1
    try:
        return time.mktime((year, month, day, 12, 0, 0, 0, 0, -1))
    except Exception:
        return None


def _is_amazon_message(msg: dict) -> bool:
    addr = (msg.get("from_addr") or "").lower()
    if "amazon" in addr or _AMAZON_SENDER_RE.search(addr):
        return True
    name = (msg.get("from_name") or "").lower()
    if name.startswith("amazon") or "amazon.com" in name:
        return True
    subj = (msg.get("subject") or "").lower()
    # Sender-less / forwarded edge case: subject mentions Amazon plus a
    # shipping-context keyword. We require both to avoid false positives
    # on generic 'Amazon gift card' marketing mail.
    return ("amazon" in subj
            and any(k in subj for k in ("order", "shipped", "delivered",
                                        "delivery", "arriving")))


# ─── Backend fetchers (read-only — never marks messages read) ───────────

def _fetch_outlook() -> list[dict]:
    et = _email_triage()
    if not et:
        return []
    mg = et._ms_graph()
    if not mg or not mg.is_configured():
        return []
    graph_get = getattr(mg, "_graph_get", None)
    shape = getattr(mg, "_shape_outlook_message", None)
    if not callable(graph_get) or not callable(shape):
        return []
    # $search lets us narrow to amazon-sender mail server-side; it can't be
    # combined with $orderby/$filter per Graph docs, so we drop them and
    # sort client-side. Top 25 covers a typical week of shipping mail.
    try:
        body = graph_get("/me/messages", {
            "$search": '"from:amazon"',
            "$top":    str(PER_BACKEND_FETCH_LIMIT),
            "$select": "id,subject,from,bodyPreview,receivedDateTime,isRead",
        })
    except Exception as e:
        _log.debug("[amazon] outlook fetch crashed: %s", e)
        return []
    if not body:
        return []
    _fetch_status["ok"] = True
    cutoff = time.time() - (LOOKBACK_DAYS * 86400)
    out: list[dict] = []
    for raw in body.get("value") or []:
        try:
            shaped = shape(raw)
        except Exception:
            continue
        recv = _parse_iso(shaped.get("received", ""))
        if recv and recv < cutoff:
            continue
        if _is_amazon_message(shaped):
            out.append(shaped)
    return out


def _shape_gmail_message(full: dict) -> dict:
    """Normalise a Gmail messages.get(format=metadata) reply into the shape the
    tracker reads (the same keys skills/email_triage produces), so the
    read-only path needs nothing from email_triage."""
    headers = {}
    for h in ((full.get("payload") or {}).get("headers")) or []:
        try:
            headers[str(h.get("name", "")).lower()] = str(h.get("value", ""))
        except Exception:
            continue
    from email.utils import parseaddr
    name, addr = parseaddr(headers.get("from", ""))
    received = ""
    try:
        ms = int(full.get("internalDate") or 0)
        if ms > 0:
            import datetime as _dt
            received = _dt.datetime.fromtimestamp(
                ms / 1000.0, _dt.timezone.utc).isoformat()
    except Exception:
        received = ""
    return {
        "id":        full.get("id") or "",
        "backend":   "gmail",
        "from_addr": (addr or "").strip(),
        "from_name": (name or "").strip(),
        "subject":   headers.get("subject", "").strip(),
        "snippet":   (full.get("snippet") or "").strip(),
        "received":  received or headers.get("date", ""),
    }


def _gmail_for_fetch():
    """(service, shape, source) for the Gmail fetch, or (None, None, ''):
    the tracker's own READ-ONLY credential first, else the Gmail service
    skills/email_triage already holds."""
    if _readonly_token_present():
        try:
            svc = _readonly_service()
        except Exception as e:
            _log.debug("[amazon] read-only gmail unavailable: %s", e)
            svc = None
        if svc is not None:
            return svc, _shape_gmail_message, "read-only"
    et = _email_triage()
    if not et:
        return None, None, ""
    try:
        if not et.is_gmail_available():
            return None, None, ""
        service = et._gmail_service()
    except Exception as e:
        _log.debug("[amazon] gmail service unavailable: %s", e)
        return None, None, ""
    if service is None:
        return None, None, ""
    shape = getattr(et, "_shape_gmail_message", None)
    return service, (shape if callable(shape) else _shape_gmail_message), "email_triage"


def _fetch_gmail() -> list[dict]:
    service, shape, source = _gmail_for_fetch()
    if service is None:
        return []
    q = (f"from:(amazon.com OR shipment-tracking@amazon.com "
         f"OR auto-confirm@amazon.com) "
         f"newer_than:{LOOKBACK_DAYS}d")
    try:
        resp = (service.users().messages()
                .list(userId="me", q=q,
                      maxResults=PER_BACKEND_FETCH_LIMIT)
                .execute())
    except Exception as e:
        _log.debug("[amazon] gmail list failed: %s", e)
        if source == "read-only":
            _ro_problem[0] = f"Gmail refused the request ({type(e).__name__})"
            _ro_cache[0] = None
        return []
    _fetch_status["ok"] = True
    if source == "read-only":
        _ro_problem[0] = ""
    out: list[dict] = []
    for item in resp.get("messages") or []:
        try:
            full = (service.users().messages()
                    .get(userId="me", id=item["id"],
                         format="metadata",
                         metadataHeaders=["From", "Subject", "Date"])
                    .execute())
        except Exception:
            continue
        try:
            shaped = shape(full)
        except Exception:
            continue
        if _is_amazon_message(shaped):
            out.append(shaped)
    return out


# ─── Promise scheduling ─────────────────────────────────────────────────

def _make_promise(message: str, condition: str, params: dict) -> None:
    """Wire through skill_utils["make_promise"] when available. The
    load_skills() loader injects skill_utils at module level, so a missing
    key just means core.memory isn't loaded — silent no-op."""
    su = globals().get("skill_utils")
    if not isinstance(su, dict):
        return
    fn = su.get("make_promise")
    if not callable(fn):
        return
    try:
        fn(message, condition, params=params, source="amazon")
    except Exception as e:
        _log.debug("[amazon] make_promise failed: %s", e)


# ─── Diff / announce ────────────────────────────────────────────────────

def _humanise_status(status: str) -> str:
    return {
        "ordered":          "ordered",
        "shipped":          "shipped",
        "out_for_delivery": "out for delivery",
        "delayed":          "delayed",
        "delivered":        "delivered",
    }.get(status, status)


def _announce_message(status: str, order_label: str) -> str:
    if status == "delivered":
        return f"Sir, your Amazon package — order {order_label} — has been delivered."
    if status == "out_for_delivery":
        return f"Sir, Amazon order {order_label} is out for delivery."
    if status == "shipped":
        return f"Sir, Amazon order {order_label} has shipped."
    if status == "delayed":
        return f"Sir, Amazon order {order_label} appears to be delayed."
    if status == "ordered":
        # Order-placed mail is usually self-triggered by the user buying
        # something; speaking it back would be noise.
        return ""
    return f"Sir, Amazon order {order_label} is now {_humanise_status(status)}."


def _process_messages(messages: list[dict], state: dict, announce: bool = True,
                      max_age_s: float | None = None) -> dict:
    """Fold fetched mail into the order table, speaking forward progress.

    ``announce`` False records everything and says nothing (the first poll of a
    fresh mailbox connection). ``max_age_s`` makes a mail older than that
    silent too - it is still recorded."""
    orders = state["orders"]
    # Oldest-first so a same-poll progression (e.g. shipped + delivered for
    # the same order) climbs each rank in order rather than skipping.
    messages = sorted(messages, key=lambda m: _parse_iso(m.get("received", "")))

    for msg in messages:
        msg_id = (msg.get("id") or "").strip()
        if not msg_id:
            continue
        subject = msg.get("subject") or ""
        snippet = msg.get("snippet") or ""
        order_id = _order_key(subject, snippet, msg_id)
        status = _classify(subject, snippet)
        delayed = _is_delayed(subject, snippet)
        recv_ts = _parse_iso(msg.get("received", ""))
        speak = bool(announce)
        if (speak and max_age_s is not None and recv_ts
                and time.time() - recv_ts > max_age_s):
            speak = False

        prev = orders.get(order_id) or {
            "first_seen_ts":    time.time(),
            "status":           "unknown",
            "history":          [],
            "last_message_ids": [],
            "subject":          subject,
        }
        already_processed = msg_id in (prev.get("last_message_ids") or [])
        prev.setdefault("last_message_ids", [])
        if msg_id not in prev["last_message_ids"]:
            prev["last_message_ids"].append(msg_id)
            prev["last_message_ids"] = prev["last_message_ids"][-5:]
        prev["last_seen_ts"] = time.time()
        prev["subject"] = subject
        item = _extract_item(subject)
        if item:
            prev["item"] = item

        est = _extract_estimated_delivery(f"{subject}\n{snippet}", ref_ts=recv_ts)
        if est:
            prev["est_delivery_ts"] = est

        prev_rank = _STATUS_ORDER.get(prev.get("status") or "unknown", 0)
        new_rank = _STATUS_ORDER.get(status, 0)
        advanced = (status != "unknown" and new_rank > prev_rank
                    and not already_processed)
        label = _order_label(order_id, prev)

        # Delay side-channel — one announcement per (order, delay flag).
        if delayed and not prev.get("delayed_announced") and not already_processed:
            prev["delayed_announced"] = True
            if speak:
                _proactive_announce(_announce_message("delayed", label))

        if advanced:
            prev["status"] = status
            history = prev.setdefault("history", [])
            history.append({"status": status, "ts": time.time()})
            prev["history"] = history[-10:]
            spoken = _announce_message(status, label)
            if spoken and speak:
                _proactive_announce(spoken)
            # Reminder promise: only on 'shipped' with a future ETA, and
            # only once per order so a re-shipped notice doesn't double up.
            est_ts = prev.get("est_delivery_ts")
            if (status == "shipped" and est_ts
                    and est_ts > time.time()
                    and not prev.get("promise_scheduled")):
                _make_promise(
                    f"Sir, Amazon order {label} is expected to arrive today.",
                    "time_at",
                    {"epoch": float(est_ts)},
                )
                prev["promise_scheduled"] = True

        orders[order_id] = prev

    state["orders"] = orders
    state["last_poll_ts"] = time.time()
    state["last_error"] = ""
    return state


# ─── Poll loop / lifecycle ──────────────────────────────────────────────

def _poll_once() -> int:
    """One pass over the mailbox. Returns how many Amazon mails it saw."""
    _fetch_status["ok"] = False
    messages: list[dict] = []
    try:
        messages.extend(_fetch_outlook())
    except Exception:
        _log.exception("[amazon] outlook fetch crashed")
    try:
        messages.extend(_fetch_gmail())
    except Exception:
        _log.exception("[amazon] gmail fetch crashed")
    reached = bool(_fetch_status["ok"] or messages)

    with _state_lock:
        state = _load_state()
        # The first poll of a mailbox only RECORDS what is already there; a
        # state written before this flag existed (last_poll_ts set) was
        # already live, so it is not silenced.
        seeded = bool(state.get("seeded")) or float(state.get("last_poll_ts") or 0) > 0
        if messages:
            state = _process_messages(messages, state, announce=seeded,
                                      max_age_s=ANNOUNCE_MAX_AGE_S)
            if not seeded:
                print(f"  [amazon] first look at the mailbox: recorded "
                      f"{len(state.get('orders') or {})} order(s) silently")
        elif reached:
            state["last_poll_ts"] = time.time()
        if reached:
            state["seeded"] = True
            state["last_error"] = ""
        else:
            state["last_error"] = (_backend_problem() or
                                   "the mailbox could not be reached")
        _save_state(state)
    return len(messages)


def _any_backend() -> bool:
    return _outlook_configured() or _gmail_configured()


def _poll_loop(stop_evt: threading.Event) -> None:
    if stop_evt.wait(INITIAL_DELAY_SECONDS):
        return
    logged_missing = False
    while not stop_evt.is_set():
        if not _any_backend():
            # Not signed in (yet): idle cheaply and re-check, so a sign-in made
            # while JARVIS runs is picked up. One quiet log line, nothing
            # spoken.
            if not logged_missing:
                logged_missing = True
                print("  [amazon] waiting for a Google sign-in "
                      f"({SIGN_IN_COMMAND}) - checking again every "
                      f"{NO_BACKEND_RECHECK_SECONDS}s")
            if stop_evt.wait(NO_BACKEND_RECHECK_SECONDS):
                return
            continue
        if logged_missing:
            logged_missing = False
            print("  [amazon] a mailbox sign-in is available - tracking now")
        try:
            _poll_once()
        except Exception:
            logging.exception("[amazon] poll iteration crashed")
            try:
                with _state_lock:
                    state = _load_state()
                    state["last_error"] = "poll crashed (see log)"
                    _save_state(state)
            except Exception:
                pass
        if stop_evt.wait(POLL_INTERVAL_SECONDS):
            return


def stop_monitor() -> None:
    """Signal the poller to exit. Idempotent — safe to call from a skill
    reload, atexit, or test teardown."""
    _stop_evt.set()
    t = _poll_thread[0]
    if t is not None and t.is_alive():
        # Give the thread a brief window to break out of its wait() so the
        # next start_monitor() call doesn't see a zombie handle. We don't
        # block long — daemon=True means a hung thread won't keep the
        # interpreter alive.
        t.join(timeout=2.0)
    _poll_thread[0] = None


def start_monitor() -> bool:
    """Start the poller (idempotent). It starts even with no mailbox sign-in:
    it then idles and notices a sign-in made later. False only when the
    feature flag is off."""
    if not _read_feature_flag():
        return False
    if _poll_thread[0] is not None and _poll_thread[0].is_alive():
        return True
    _stop_evt.clear()
    t = threading.Thread(target=_poll_loop, args=(_stop_evt,),
                         daemon=True, name="amazon-order-tracker")
    t.start()
    _poll_thread[0] = t
    if _any_backend():
        print(f"  [amazon] order tracker active - polling every "
              f"{POLL_INTERVAL_SECONDS // 60} minutes")
    else:
        print(f"  [amazon] order tracker idle: {_backend_problem()}")
    return True


def _refresh_now(timeout: float = REFRESH_WAIT_S) -> None:
    """Look at the mailbox now so an answer is fresh, but never hold the voice
    loop longer than `timeout`: the poll runs on a worker thread and the caller
    answers from the state file if it is not done."""
    if not _any_backend():
        return
    t = threading.Thread(target=_poll_once_safely, daemon=True,
                         name="amazon-refresh")
    t.start()
    t.join(timeout=max(0.0, timeout))


def _poll_once_safely() -> None:
    try:
        _poll_once()
    except Exception:
        _log.exception("[amazon] refresh crashed")


# ─── Voice actions ──────────────────────────────────────────────────────

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS   = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fmt_date(ts: float | None) -> str:
    if not ts:
        return ""
    try:
        lt = time.localtime(ts)
        return f"{_WEEKDAYS[lt.tm_wday]}, {_MONTHS[lt.tm_mon - 1]} {lt.tm_mday}"
    except Exception:
        return ""


def _entry_name(oid: str, entry: dict, fallback: str) -> str:
    """How an order is named in a spoken roll-call."""
    if oid.startswith("msg:"):
        return fallback
    return _order_label(oid, entry)


def action_check_orders(_: str = "") -> str:
    problem = _backend_problem()
    if problem and not _load_state().get("orders"):
        # Not signed in (or the sign-in is broken): say so. An empty list here
        # would read as "nothing is coming", which nobody has checked.
        return problem
    if not problem:
        _refresh_now()
    state = _load_state()
    orders = state.get("orders") or {}
    if not orders:
        return ("No active Amazon orders on record, sir — either nothing's "
                "shipping or the email backend hasn't been polled yet.")
    in_flight = [
        (oid, entry) for oid, entry in orders.items()
        if entry.get("status") in ("ordered", "shipped", "out_for_delivery")
    ]
    if not in_flight:
        out = "Nothing currently in transit, sir."
        return (out + " " + problem) if problem else out
    in_flight.sort(key=lambda kv: kv[1].get("last_seen_ts", 0), reverse=True)
    parts = []
    for oid, entry in in_flight[:5]:
        label = _entry_name(oid, entry, "an order")
        status_words = _humanise_status(entry.get("status") or "unknown")
        eta = _fmt_date(entry.get("est_delivery_ts"))
        if entry.get("delayed_announced"):
            status_words += " (delayed)"
        if eta:
            parts.append(f"{label} — {status_words}, arriving {eta}")
        else:
            parts.append(f"{label} — {status_words}")
    out = "Amazon, sir: " + "; ".join(parts) + "."
    return (out + " (That is from the last time I could look: " + problem + ")"
            ) if problem else out


def action_recent_delivery(_: str = "") -> str:
    problem = _backend_problem()
    if problem and not _load_state().get("orders"):
        return problem
    if not problem:
        _refresh_now()
    state = _load_state()
    orders = state.get("orders") or {}
    cutoff = time.time() - (5 * 86400)
    delivered = [
        (oid, entry) for oid, entry in orders.items()
        if entry.get("status") == "delivered"
        and entry.get("last_seen_ts", 0) >= cutoff
    ]
    if not delivered:
        return "No Amazon deliveries in the last five days, sir."
    delivered.sort(key=lambda kv: kv[1].get("last_seen_ts", 0), reverse=True)
    parts = []
    for oid, entry in delivered[:5]:
        label = _entry_name(oid, entry, "an order")
        when = _fmt_date(entry.get("last_seen_ts"))
        parts.append(f"{label} ({when})" if when else label)
    return "Recently delivered, sir: " + "; ".join(parts) + "."


def action_amazon_tracking_status(_: str = "") -> str:
    state = _load_state()
    enabled = _read_feature_flag()
    bits = [f"enabled: {enabled}"]
    bits.append(f"Outlook: {'configured' if _outlook_configured() else 'not configured'}")
    if _readonly_token_present():
        gmail = "read-only token saved"
        if _ro_problem[0]:
            gmail += f" but {_ro_problem[0]}"
    elif _gmail_configured():
        gmail = "configured through email triage"
    else:
        gmail = "not signed in"
    bits.append(f"Gmail: {gmail}")
    last = state.get("last_poll_ts") or 0.0
    if last:
        age_s = max(0, int(time.time() - last))
        bits.append(f"last poll {age_s}s ago")
    else:
        bits.append("last poll: never")
    if state.get("last_error"):
        bits.append(f"last error: {state['last_error']}")
    bits.append(f"orders tracked: {len(state.get('orders') or {})}")
    running = bool(_poll_thread[0] and _poll_thread[0].is_alive())
    bits.append(f"poller: {'running' if running else 'stopped'}")
    out = "Amazon tracker — " + "; ".join(bits) + ", sir."
    problem = _backend_problem()
    return (out + " " + problem) if problem else out


def register(actions):
    actions["check_orders"]           = action_check_orders
    actions["check_amazon_orders"]    = action_check_orders
    actions["amazon_orders"]          = action_check_orders
    actions["recent_delivery"]        = action_recent_delivery
    actions["recent_deliveries"]      = action_recent_delivery
    actions["amazon_tracking_status"] = action_amazon_tracking_status

    if not _read_feature_flag():
        print("  [amazon] AMAZON_TRACKING_ENABLED=False — actions registered, poller off")
        return
    # Enabled: start the poller whether or not a mailbox sign-in exists yet.
    # With none it idles quietly (a log line, never a spoken nag at boot) and
    # the voice actions say "I need a Google sign-in first, sir".
    start_monitor()


if __name__ == "__main__":   # pragma: no cover - interactive one-time sign-in
    import argparse
    ap = argparse.ArgumentParser(
        description="Amazon order tracker: one-time READ-ONLY Gmail sign-in")
    ap.add_argument("--auth-gmail", action="store_true",
                    help="open the browser and save a read-only Gmail token")
    args = ap.parse_args()
    if args.auth_gmail:
        sys.exit(0 if authenticate_gmail_readonly() else 1)
    ap.print_help()
