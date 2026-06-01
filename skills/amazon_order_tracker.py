"""
Amazon order tracking skill for JARVIS.

Polls whichever email backend (Microsoft Graph or Gmail) the user has
already wired up via skills/email_triage.py every POLL_INTERVAL_SECONDS,
filters for Amazon shipment notifications, and proactively announces
status transitions (shipped → out for delivery → delivered, plus delay
warnings).

State lives in data/amazon_order_state.json. Writes go through
core.atomic_io._atomic_write_json so concurrent skill reloads can't
corrupt it (same approach as skills/bambu_monitor.py).

Auth: gated by ``AMAZON_TRACKING_ENABLED`` in bobert_companion.py. No
new credentials — reuses Microsoft Graph and/or Gmail via skills.email_triage.

Actions registered:
  check_orders            — manual roll-call of in-flight orders.
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
INITIAL_DELAY_SECONDS  = 45
LOOKBACK_DAYS          = 7
STATE_MAX_ORDERS       = 200   # drop oldest delivered orders past this cap
PER_BACKEND_FETCH_LIMIT = 25

_STATE_FILE = os.path.join(_PROJECT_DIR, "data", "amazon_order_state.json")

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
# 'delivery' check below it.
_STATUS_RULES = [
    (re.compile(r"\bdelivered\b",                          re.I), "delivered"),
    (re.compile(r"\bout for delivery\b",                    re.I), "out_for_delivery"),
    (re.compile(r"\barriving (today|tomorrow|soon)\b",     re.I), "out_for_delivery"),
    (re.compile(r"\b(shipped|on the way|on its way|"
                r"your package has shipped|shipment notification)\b",
                                                            re.I), "shipped"),
    (re.compile(r"\b(order (placed|confirmation)|"
                r"thank you for your order|"
                r"your amazon\.com order)\b",               re.I), "ordered"),
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
    try:
        bc = importlib.import_module("bobert_companion")
        return bool(getattr(bc, "AMAZON_TRACKING_ENABLED", False))
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
    et = _email_triage()
    if not et:
        return False
    try:
        return bool(et.is_gmail_available())
    except Exception:
        return False


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
    text = f"{subject}\n{snippet}"
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


def _fetch_gmail() -> list[dict]:
    et = _email_triage()
    if not et:
        return []
    try:
        if not et.is_gmail_available():
            return []
        service = et._gmail_service()
    except Exception as e:
        _log.debug("[amazon] gmail service unavailable: %s", e)
        return []
    if service is None:
        return []
    shape = getattr(et, "_shape_gmail_message", None)
    if not callable(shape):
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
        return []
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


def _process_messages(messages: list[dict], state: dict) -> dict:
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
        order_id = _extract_order_id(subject, snippet) or f"msg:{msg_id[:24]}"
        status = _classify(subject, snippet)
        delayed = _is_delayed(subject, snippet)

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

        recv_ts = _parse_iso(msg.get("received", ""))
        est = _extract_estimated_delivery(f"{subject}\n{snippet}", ref_ts=recv_ts)
        if est:
            prev["est_delivery_ts"] = est

        prev_rank = _STATUS_ORDER.get(prev.get("status") or "unknown", 0)
        new_rank = _STATUS_ORDER.get(status, 0)
        advanced = (status != "unknown" and new_rank > prev_rank
                    and not already_processed)

        # Delay side-channel — one announcement per (order, delay flag).
        if delayed and not prev.get("delayed_announced") and not already_processed:
            prev["delayed_announced"] = True
            label = order_id if not order_id.startswith("msg:") else "an Amazon order"
            _proactive_announce(_announce_message("delayed", label))

        if advanced:
            prev["status"] = status
            history = prev.setdefault("history", [])
            history.append({"status": status, "ts": time.time()})
            prev["history"] = history[-10:]
            label = order_id if not order_id.startswith("msg:") else "an Amazon order"
            spoken = _announce_message(status, label)
            if spoken:
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
    messages: list[dict] = []
    try:
        messages.extend(_fetch_outlook())
    except Exception:
        _log.exception("[amazon] outlook fetch crashed")
    try:
        messages.extend(_fetch_gmail())
    except Exception:
        _log.exception("[amazon] gmail fetch crashed")

    with _state_lock:
        state = _load_state()
        if messages:
            state = _process_messages(messages, state)
        else:
            state["last_poll_ts"] = time.time()
        _save_state(state)
    return len(messages)


def _poll_loop(stop_evt: threading.Event) -> None:
    if stop_evt.wait(INITIAL_DELAY_SECONDS):
        return
    while not stop_evt.is_set():
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
    if not _read_feature_flag():
        return False
    if _poll_thread[0] is not None and _poll_thread[0].is_alive():
        return True
    if not (_outlook_configured() or _gmail_configured()):
        print("  [amazon] no email backend configured — poller off. "
              "Run `python -m skills.ms_graph --auth` or "
              "`python -m skills.email_triage --auth-gmail` first.")
        return False
    _stop_evt.clear()
    t = threading.Thread(target=_poll_loop, args=(_stop_evt,),
                         daemon=True, name="amazon-order-tracker")
    t.start()
    _poll_thread[0] = t
    print(f"  [amazon] order tracker active — polling every "
          f"{POLL_INTERVAL_SECONDS // 60} minutes")
    return True


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


def action_check_orders(_: str = "") -> str:
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
        return "Nothing currently in transit, sir."
    in_flight.sort(key=lambda kv: kv[1].get("last_seen_ts", 0), reverse=True)
    parts = []
    for oid, entry in in_flight[:5]:
        label = oid if not oid.startswith("msg:") else "an order"
        status_words = _humanise_status(entry.get("status") or "unknown")
        eta = _fmt_date(entry.get("est_delivery_ts"))
        if entry.get("delayed_announced"):
            status_words += " (delayed)"
        if eta:
            parts.append(f"{label} — {status_words}, arriving {eta}")
        else:
            parts.append(f"{label} — {status_words}")
    return "Amazon, sir: " + "; ".join(parts) + "."


def action_recent_delivery(_: str = "") -> str:
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
        label = oid if not oid.startswith("msg:") else "an order"
        when = _fmt_date(entry.get("last_seen_ts"))
        parts.append(f"{label} ({when})" if when else label)
    return "Recently delivered, sir: " + "; ".join(parts) + "."


def action_amazon_tracking_status(_: str = "") -> str:
    state = _load_state()
    enabled = _read_feature_flag()
    bits = [f"enabled: {enabled}"]
    bits.append(f"Outlook: {'configured' if _outlook_configured() else 'not configured'}")
    bits.append(f"Gmail: {'configured' if _gmail_configured() else 'not configured'}")
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
    return "Amazon tracker — " + "; ".join(bits) + ", sir."


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
    if not (_outlook_configured() or _gmail_configured()):
        # Surface this once so a misconfigured first-time setup doesn't go
        # silent. Skill stays loaded so check_orders / status still report.
        _proactive_announce(
            "Sir, Amazon order tracking is enabled but no email backend "
            "is configured. Authenticate Microsoft Graph or Gmail and "
            "I'll start watching for shipments."
        )
        print("  [amazon] no email backend configured — poller off")
        return
    start_monitor()
