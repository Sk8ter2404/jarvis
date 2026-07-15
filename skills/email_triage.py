"""
skills/email_triage.py — unified Gmail + Outlook inbox triage for JARVIS.

Sits on top of two backends:

  • Outlook / Microsoft 365 — via ``skills/ms_graph.py`` (already wired into
    morning_briefing for unread counts). New write actions added there:
    create_draft_reply / send_draft / archive_message / apply_category /
    mark_as_read.

  • Gmail — via ``google-api-python-client`` + ``google-auth-oauthlib``.
    Lazy-imported, so the skill registers cleanly on machines without the
    deps. OAuth client secret lives at ``gmail_credentials.json`` (project
    root, downloaded from Google Cloud Console → APIs & Services →
    Credentials → OAuth client → Desktop app). Token cached at
    ``data/gmail_token.json``. First-run authorisation::

        python -m skills.email_triage --auth-gmail

    A browser opens, the user grants Gmail modify + compose, and the
    refresh token is cached so subsequent runs are silent.

The skill exposes a JARVIS-voice command surface that doesn't care which
backend a message lives on — list_unread / read_thread / draft_reply /
archive / categorize_inbox / email_briefing — and a Haiku classifier that
triages each unread message into one of {urgent, fyi, newsletter, spam}.
``email_briefing`` is what the morning briefing calls to get a short
spoken summary of the priority inbox.

Pre-drafted replies follow the "one-word confirm" flow described in the
research task:

  1. User: 'JARVIS, draft a reply to that one'  →  action ``draft_reply``
     generates a candidate via Claude (Haiku for speed), creates the
     draft on the backend (so it shows up in Outlook/Gmail Drafts), and
     stashes it in ``data/email_pending_drafts.json`` as the active
     pending draft.
  2. JARVIS reads it back: 'Sir, draft reply ready — "..." — say send,
     scrap, or edit.'
  3. User: 'send'  →  ``confirm_pending_draft`` actually .send()s it.
     User: 'scrap' →  ``scrap_pending_draft`` discards.
     User: 'edit <new text>' →  ``edit_pending_draft <text>`` PATCH-es
     the body on the backend.

All actions return a string (JARVIS voice, ', sir.' suffix where
natural) so they can be dispatched by the existing dispatcher.

ACTIONS REGISTERED
------------------
  list_unread            [N|backend]          inbox roll-call
  unread_email           alias of list_unread
  read_thread            <msg_id|index|"latest"> read aloud + mark read
  read_email             alias of read_thread
  draft_reply            <msg_id|index|"latest"> [instructions]
  pre_draft_reply        alias of draft_reply
  archive_email          <msg_id|index|"latest">
  archive_message        alias of archive_email
  categorize_inbox       run Haiku across unread; apply backend categories
  triage_inbox           alias of categorize_inbox
  email_briefing         priority-mail summary for morning briefing
  confirm_pending_draft  send the active pending draft
  send_draft             alias of confirm_pending_draft
  scrap_pending_draft    discard the active pending draft
  edit_pending_draft     <new body> overwrite + PATCH
  list_pending_drafts    show what's queued
  email_triage_status    health of Gmail + Outlook backends
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

# Project root on sys.path so `skills.email_triage` and direct
# `python -m skills.email_triage` both find core/* and sibling skills.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

try:
    from core.atomic_io import _atomic_write_json
except Exception:  # pragma: no cover — boot-order safety
    import tempfile

    def _atomic_write_json(path, data, *, indent=2):
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise

_log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────
LLM_MODEL                = "claude-haiku-4-5"
LLM_TIMEOUT_SECONDS      = 8.0
# Overall wall-clock budget for the synchronous categorize/briefing loops so a
# large unread set can't freeze the voice thread for minutes (2026-07-14 #16).
INBOX_TRIAGE_BUDGET_S    = 45.0
ENABLE_LLM_TRIAGE        = True
DEFAULT_LIST_LIMIT       = 8
MAX_SPOKEN_BODY_CHARS    = 600   # truncate long bodies before TTS
MAX_DRAFT_BODY_CHARS     = 1200  # cap LLM output so it doesn't ramble
GMAIL_CREDENTIALS_FILE   = os.path.join(_PROJECT_DIR, "gmail_credentials.json")
GMAIL_TOKEN_FILE         = os.path.join(_PROJECT_DIR, "data", "gmail_token.json")
PENDING_DRAFTS_FILE      = os.path.join(_PROJECT_DIR, "data", "email_pending_drafts.json")
INBOX_INDEX_FILE         = os.path.join(_PROJECT_DIR, "data", "email_inbox_index.json")
# Gmail scopes: modify covers read + label + archive (trash/move-to-archive
# is just removing INBOX label). compose covers draft creation + send.
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]

LLM_VERDICTS = ("urgent", "fyi", "newsletter", "spam")
CATEGORY_LABELS = {
    "urgent":     "JARVIS/Urgent",
    "fyi":        "JARVIS/FYI",
    "newsletter": "JARVIS/Newsletter",
    "spam":       "JARVIS/Spam",
}

_state_lock = threading.RLock()
_gmail_service_cache: list = [None]    # holds (service, expires_at) tuple
_gmail_unavailable_reason = [""]


# ─── Backend: Outlook (via ms_graph) ─────────────────────────────────────

def _ms_graph():
    """Late-bound import so a stale ms_graph module doesn't break our boot."""
    try:
        return importlib.import_module("skills.ms_graph")
    except Exception:
        try:
            return importlib.import_module("ms_graph")
        except Exception:
            return None


def _outlook_configured() -> bool:
    mod = _ms_graph()
    return bool(mod and mod.is_configured())


def _outlook_list_unread(top_n: int) -> list[dict]:
    mod = _ms_graph()
    if not mod:
        return []
    try:
        return mod.list_unread_messages(top_n=top_n)
    except Exception as e:
        _log.debug("[email] outlook list failed: %s", e)
        return []


def _outlook_get_thread(msg_id: str) -> dict | None:
    mod = _ms_graph()
    if not mod:
        return None
    try:
        return mod.get_message_thread(msg_id)
    except Exception as e:
        _log.debug("[email] outlook fetch failed: %s", e)
        return None


def _outlook_create_draft(msg_id: str, body: str, reply_all: bool = False) -> str | None:
    mod = _ms_graph()
    if not mod:
        return None
    try:
        draft = mod.create_draft_reply(msg_id, body, reply_all=reply_all)
    except Exception as e:
        _log.debug("[email] outlook draft failed: %s", e)
        return None
    if not draft:
        return None
    return draft.get("id") or None


def _outlook_send_draft(draft_id: str) -> bool:
    mod = _ms_graph()
    if not mod:
        return False
    try:
        return mod.send_draft(draft_id)
    except Exception as e:
        _log.debug("[email] outlook send failed: %s", e)
        return False


def _outlook_update_draft(draft_id: str, body: str) -> bool:
    mod = _ms_graph()
    if not mod:
        return False
    try:
        return mod.update_draft_body(draft_id, body)
    except Exception as e:
        _log.debug("[email] outlook update failed: %s", e)
        return False


def _outlook_archive(msg_id: str) -> bool:
    mod = _ms_graph()
    if not mod:
        return False
    try:
        return mod.archive_message(msg_id)
    except Exception as e:
        _log.debug("[email] outlook archive failed: %s", e)
        return False


def _outlook_apply_category(msg_id: str, category: str) -> bool:
    mod = _ms_graph()
    if not mod:
        return False
    try:
        return mod.apply_category(msg_id, category)
    except Exception as e:
        _log.debug("[email] outlook category failed: %s", e)
        return False


def _outlook_mark_read(msg_id: str) -> bool:
    mod = _ms_graph()
    if not mod:
        return False
    try:
        return mod.mark_as_read(msg_id, True)
    except Exception as e:
        _log.debug("[email] outlook mark_read failed: %s", e)
        return False


# ─── Backend: Gmail ──────────────────────────────────────────────────────
#
# google-api-python-client + google-auth-oauthlib are intentionally
# optional. is_gmail_available() returns False until they're installed AND
# gmail_credentials.json exists AND data/gmail_token.json has been
# populated by --auth-gmail. Each helper degrades silently to [] / None /
# False so the unified actions just skip Gmail when it isn't ready.

def _probe_gmail_deps() -> tuple[object, object, object, object, object]:
    """Return (build, Credentials, InstalledAppFlow, Request, HttpError)
    or (None, ...) if any import fails. Cached so we don't re-import."""
    try:
        from googleapiclient.discovery import build               # type: ignore
        from googleapiclient.errors import HttpError              # type: ignore
        from google.oauth2.credentials import Credentials         # type: ignore
        from google.auth.transport.requests import Request        # type: ignore
        from google_auth_oauthlib.flow import InstalledAppFlow    # type: ignore
        return build, Credentials, InstalledAppFlow, Request, HttpError
    except Exception as e:
        _gmail_unavailable_reason[0] = (
            f"google-api-python-client / google-auth-oauthlib not installed: {e}. "
            "Run `pip install google-api-python-client google-auth-oauthlib` "
            "to enable Gmail triage."
        )
        return None, None, None, None, None


def is_gmail_available() -> bool:
    build, *_ = _probe_gmail_deps()
    if build is None:
        return False
    if not os.path.exists(GMAIL_CREDENTIALS_FILE):
        _gmail_unavailable_reason[0] = (
            f"gmail_credentials.json missing at {GMAIL_CREDENTIALS_FILE}. "
            "Download an OAuth desktop client from Google Cloud Console → "
            "APIs & Services → Credentials and save it there."
        )
        return False
    return True


def _load_gmail_credentials():
    """Return a google.oauth2.credentials.Credentials object, refreshing
    against the cached refresh_token when needed. None if no token is
    cached (caller should run --auth-gmail)."""
    build, Credentials, InstalledAppFlow, Request, HttpError = _probe_gmail_deps()
    if Credentials is None:
        return None
    if not os.path.exists(GMAIL_TOKEN_FILE):
        return None
    try:
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
    except Exception as e:
        _log.debug("[email] gmail token unreadable: %s", e)
        return None
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            _log.warning("[email] gmail refresh failed: %s", e)
            return None
        try:
            _atomic_write_json(GMAIL_TOKEN_FILE, json.loads(creds.to_json()))
        except Exception as e:
            _log.debug("[email] gmail token save failed: %s", e)
        return creds
    return None


def _gmail_service():
    """Lazy build + cache the Gmail API client. Cache is invalidated when
    the credentials object reports invalid so a token rotation doesn't
    keep us using a stale handle."""
    if not is_gmail_available():
        return None
    cached = _gmail_service_cache[0]
    if cached is not None:
        service, creds_obj = cached
        if creds_obj is not None and getattr(creds_obj, "valid", False):
            return service
    build, *_ = _probe_gmail_deps()
    creds = _load_gmail_credentials()
    if creds is None:
        return None
    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as e:
        _log.warning("[email] gmail service build failed: %s", e)
        return None
    _gmail_service_cache[0] = (service, creds)
    return service


def authenticate_gmail() -> bool:
    """Run the InstalledAppFlow against gmail_credentials.json. Browser
    opens, user consents, token cached at data/gmail_token.json."""
    build, Credentials, InstalledAppFlow, Request, HttpError = _probe_gmail_deps()
    if InstalledAppFlow is None:
        print(
            "[email] Gmail OAuth libs missing. "
            "Run `pip install google-api-python-client google-auth-oauthlib` "
            "and retry --auth-gmail."
        )
        return False
    if not os.path.exists(GMAIL_CREDENTIALS_FILE):
        print(
            f"[email] {GMAIL_CREDENTIALS_FILE} not found. "
            "Create an OAuth client at https://console.cloud.google.com/apis/credentials "
            "(type: Desktop app), download the JSON, and save it as "
            "gmail_credentials.json at the project root."
        )
        return False
    flow = InstalledAppFlow.from_client_secrets_file(
        GMAIL_CREDENTIALS_FILE, GMAIL_SCOPES,
    )
    creds = flow.run_local_server(port=0)
    os.makedirs(os.path.dirname(GMAIL_TOKEN_FILE), exist_ok=True)
    _atomic_write_json(GMAIL_TOKEN_FILE, json.loads(creds.to_json()))
    print(f"[email] Gmail token saved to {GMAIL_TOKEN_FILE}")
    return True


def _gmail_header(headers: list[dict], name: str) -> str:
    """Pull a header value by name from a Gmail message['payload']['headers']."""
    if not headers:
        return ""
    lname = name.lower()
    for h in headers:
        if (h.get("name") or "").lower() == lname:
            return (h.get("value") or "").strip()
    return ""


def _gmail_decode_part(part: dict) -> str:
    """Decode a Gmail message body part (base64url) to plain text."""
    import base64
    data = ((part or {}).get("body") or {}).get("data") or ""
    if not data:
        return ""
    try:
        raw = base64.urlsafe_b64decode(data.encode("ascii") + b"==").decode(
            "utf-8", errors="replace"
        )
    except Exception:
        return ""
    return raw


def _gmail_extract_body(payload: dict) -> tuple[str, str]:
    """Walk the MIME tree and return (plain_text, html). Prefers text/plain
    but falls back to stripping text/html when plain isn't present."""
    if not payload:
        return "", ""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def _walk(part: dict):
        mt = (part.get("mimeType") or "").lower()
        if mt.startswith("multipart/"):
            for sub in part.get("parts") or []:
                _walk(sub)
            return
        body = _gmail_decode_part(part)
        if not body:
            return
        if mt == "text/plain":
            plain_parts.append(body)
        elif mt == "text/html":
            html_parts.append(body)

    _walk(payload)
    plain = "\n".join(plain_parts).strip()
    html  = "\n".join(html_parts).strip()
    if not plain and html:
        mod = _ms_graph()
        if mod and hasattr(mod, "_strip_html"):
            plain = mod._strip_html(html)
    return plain, html


def _shape_gmail_message(msg: dict) -> dict:
    """Normalise a Gmail messages.get response into the unified shape."""
    headers = ((msg.get("payload") or {}).get("headers")) or []
    from_full = _gmail_header(headers, "From")
    from_name, from_addr = _split_from(from_full)
    subject = _gmail_header(headers, "Subject")
    received = _gmail_header(headers, "Date")
    label_ids = list(msg.get("labelIds") or [])
    return {
        "backend":    "gmail",
        "id":         msg.get("id") or "",
        "thread_id":  msg.get("threadId") or "",
        "from_name":  from_name,
        "from_addr":  from_addr,
        "subject":    subject,
        "snippet":    (msg.get("snippet") or "").strip(),
        "received":   received,
        "unread":     "UNREAD" in label_ids,
        "categories": [lid for lid in label_ids if lid.startswith("Label_") or lid.startswith("JARVIS")],
        "label_ids":  label_ids,
    }


def _split_from(raw: str) -> tuple[str, str]:
    """'Jane Doe <jane@x.com>' → ('Jane Doe', 'jane@x.com')."""
    if not raw:
        return "", ""
    m = re.match(r'^\s*"?([^"<]*?)"?\s*<\s*([^>]+)\s*>\s*$', raw)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if "@" in raw:
        return "", raw.strip()
    return raw.strip(), ""


def _gmail_list_unread(top_n: int) -> list[dict]:
    service = _gmail_service()
    if service is None:
        return []
    try:
        resp = (service.users().messages()
                .list(userId="me", q="in:inbox is:unread",
                      maxResults=max(1, min(50, top_n)))
                .execute())
    except Exception as e:
        _log.debug("[email] gmail list failed: %s", e)
        return []
    out: list[dict] = []
    for item in resp.get("messages") or []:
        try:
            full = (service.users().messages()
                    .get(userId="me", id=item["id"],
                         format="metadata",
                         metadataHeaders=["From", "Subject", "Date"])
                    .execute())
        except Exception as e:
            _log.debug("[email] gmail get metadata failed: %s", e)
            continue
        out.append(_shape_gmail_message(full))
    return out


def _gmail_get_thread(msg_id: str) -> dict | None:
    service = _gmail_service()
    if service is None:
        return None
    try:
        full = (service.users().messages()
                .get(userId="me", id=msg_id, format="full")
                .execute())
    except Exception as e:
        _log.debug("[email] gmail get failed: %s", e)
        return None
    shaped = _shape_gmail_message(full)
    plain, html = _gmail_extract_body((full or {}).get("payload") or {})
    shaped["body_text"] = plain
    shaped["body_html"] = html
    return shaped


def _gmail_create_draft(msg_id: str, body: str) -> str | None:
    service = _gmail_service()
    if service is None:
        return None
    original = _gmail_get_thread(msg_id)
    if original is None:
        return None
    import base64
    from email.message import EmailMessage
    msg = EmailMessage()
    to_addr = original.get("from_addr") or ""
    if not to_addr:
        return None
    msg["To"] = to_addr
    subject = original.get("subject") or ""
    if subject and not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    msg["Subject"] = subject
    msg.set_content(body or "")
    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    draft_body = {
        "message": {
            "raw": encoded,
            "threadId": original.get("thread_id") or "",
        }
    }
    try:
        resp = (service.users().drafts()
                .create(userId="me", body=draft_body)
                .execute())
    except Exception as e:
        _log.warning("[email] gmail draft create failed: %s", e)
        return None
    return resp.get("id") or None


def _gmail_send_draft(draft_id: str) -> bool:
    service = _gmail_service()
    if service is None:
        return False
    try:
        (service.users().drafts()
            .send(userId="me", body={"id": draft_id})
            .execute())
        return True
    except Exception as e:
        _log.warning("[email] gmail draft send failed: %s", e)
        return False


def _gmail_update_draft(draft_id: str, new_body: str, original_msg_id: str) -> bool:
    """Replace a draft's body. Gmail's update endpoint requires the full raw
    MIME so we reconstruct the message using the original thread's headers."""
    service = _gmail_service()
    if service is None:
        return False
    original = _gmail_get_thread(original_msg_id)
    if original is None:
        return False
    import base64
    from email.message import EmailMessage
    msg = EmailMessage()
    to_addr = original.get("from_addr") or ""
    if not to_addr:
        return False
    msg["To"] = to_addr
    subject = original.get("subject") or ""
    if subject and not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    msg["Subject"] = subject
    msg.set_content(new_body or "")
    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    try:
        (service.users().drafts()
            .update(userId="me", id=draft_id,
                    body={"message": {"raw": encoded,
                                       "threadId": original.get("thread_id") or ""}})
            .execute())
        return True
    except Exception as e:
        _log.warning("[email] gmail draft update failed: %s", e)
        return False


def _gmail_archive(msg_id: str) -> bool:
    """Archive on Gmail = remove the INBOX label (the message stays in All
    Mail / its labels, just not the inbox)."""
    service = _gmail_service()
    if service is None:
        return False
    try:
        (service.users().messages()
            .modify(userId="me", id=msg_id,
                    body={"removeLabelIds": ["INBOX"]})
            .execute())
        return True
    except Exception as e:
        _log.warning("[email] gmail archive failed: %s", e)
        return False


def _gmail_apply_label(msg_id: str, label_name: str) -> bool:
    """Find or create a user label, then attach it to the message."""
    service = _gmail_service()
    if service is None:
        return False
    try:
        labels = (service.users().labels().list(userId="me").execute()
                  .get("labels") or [])
    except Exception as e:
        _log.debug("[email] gmail label list failed: %s", e)
        return False
    label_id = None
    for lab in labels:
        if (lab.get("name") or "").lower() == label_name.lower():
            label_id = lab.get("id")
            break
    if label_id is None:
        try:
            created = (service.users().labels()
                       .create(userId="me",
                               body={"name": label_name,
                                     "labelListVisibility": "labelShow",
                                     "messageListVisibility": "show"})
                       .execute())
            label_id = created.get("id")
        except Exception as e:
            _log.debug("[email] gmail label create failed: %s", e)
            return False
    if not label_id:
        return False
    try:
        (service.users().messages()
            .modify(userId="me", id=msg_id,
                    body={"addLabelIds": [label_id]})
            .execute())
        return True
    except Exception as e:
        _log.debug("[email] gmail label apply failed: %s", e)
        return False


def _gmail_mark_read(msg_id: str) -> bool:
    service = _gmail_service()
    if service is None:
        return False
    try:
        (service.users().messages()
            .modify(userId="me", id=msg_id,
                    body={"removeLabelIds": ["UNREAD"]})
            .execute())
        return True
    except Exception as e:
        _log.debug("[email] gmail mark_read failed: %s", e)
        return False


# ─── Cross-backend operations ────────────────────────────────────────────

def list_unread(top_n: int = DEFAULT_LIST_LIMIT,
                backend: str = "all") -> list[dict]:
    """Aggregate unread mail across both backends. Results sorted by
    received-date descending (best-effort — Gmail returns RFC-2822 date
    strings; Outlook returns ISO-8601). Truncates to ``top_n`` total."""
    top_n = max(1, min(50, int(top_n)))
    out: list[dict] = []
    if backend in ("all", "outlook") and _outlook_configured():
        out.extend(_outlook_list_unread(top_n))
    if backend in ("all", "gmail") and is_gmail_available():
        out.extend(_gmail_list_unread(top_n))
    out.sort(key=lambda m: _parse_received(m.get("received") or ""), reverse=True)
    return out[:top_n]


def _parse_received(s: str) -> float:
    """Cheap sort key — best-effort epoch conversion for both ISO-8601 and
    RFC-2822 date strings. Returns 0 on parse failure (oldest)."""
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


# ─── Inbox index — lets the user say 'read the first one' by number ──────

def _save_inbox_index(messages: list[dict]) -> None:
    """Persist the most recent unread roll-call so subsequent voice
    commands like 'read the second one' / 'archive number three' can
    resolve numeric indices to backend ids."""
    index = []
    for i, m in enumerate(messages, start=1):
        index.append({
            "index":    i,
            "backend":  m.get("backend"),
            "id":       m.get("id"),
            "subject":  m.get("subject", "")[:120],
            "from":     (m.get("from_name") or m.get("from_addr") or "")[:80],
        })
    try:
        os.makedirs(os.path.dirname(INBOX_INDEX_FILE), exist_ok=True)
        _atomic_write_json(INBOX_INDEX_FILE, {
            "captured_at": time.time(),
            "messages":    index,
        })
    except Exception as e:
        _log.debug("[email] inbox index save failed: %s", e)


def _load_inbox_index() -> list[dict]:
    if not os.path.exists(INBOX_INDEX_FILE):
        return []
    try:
        with open(INBOX_INDEX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list((data or {}).get("messages") or [])
    except Exception:
        return []


def _resolve_handle(token: str) -> dict | None:
    """Map a user-provided handle ('1', 'first', '2', 'latest', or a raw
    backend id) onto {backend, id}. Returns None when nothing matches."""
    if not token:
        return None
    tok = token.strip().lower()
    index = _load_inbox_index()
    if index:
        if tok in ("latest", "newest", "first", "1", "one"):
            row = index[0]
            return {"backend": row["backend"], "id": row["id"]}
        # numeric / ordinal lookup
        words = {"second": 2, "third": 3, "fourth": 4, "fifth": 5,
                 "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
                 "tenth": 10}
        n = None
        if tok.isdigit():
            n = int(tok)
        elif tok in words:
            n = words[tok]
        if n is not None and 1 <= n <= len(index):
            row = index[n - 1]
            return {"backend": row["backend"], "id": row["id"]}
    # Fall back to treating the whole token as a raw id. Try both backends.
    return {"backend": "auto", "id": token.strip()}


def _get_thread(handle: dict) -> dict | None:
    backend = handle.get("backend") or "auto"
    mid = handle.get("id") or ""
    if not mid:
        return None
    if backend == "outlook":
        return _outlook_get_thread(mid)
    if backend == "gmail":
        return _gmail_get_thread(mid)
    # auto — try outlook first (more likely the longer id format), then gmail.
    return _outlook_get_thread(mid) or _gmail_get_thread(mid)


def _archive(handle: dict) -> bool:
    backend = handle.get("backend") or "auto"
    mid = handle.get("id") or ""
    if not mid:
        return False
    if backend == "outlook":
        return _outlook_archive(mid)
    if backend == "gmail":
        return _gmail_archive(mid)
    return _outlook_archive(mid) or _gmail_archive(mid)


def _mark_read(handle: dict) -> bool:
    backend = handle.get("backend") or "auto"
    mid = handle.get("id") or ""
    if not mid:
        return False
    if backend == "outlook":
        return _outlook_mark_read(mid)
    if backend == "gmail":
        return _gmail_mark_read(mid)
    return _outlook_mark_read(mid) or _gmail_mark_read(mid)


def _apply_category(handle: dict, verdict: str) -> bool:
    """Apply the JARVIS triage category/label appropriate for ``verdict``.
    Maps onto Outlook category strings or Gmail labels."""
    backend = handle.get("backend") or "auto"
    mid = handle.get("id") or ""
    if not mid or verdict not in LLM_VERDICTS:
        return False
    label = CATEGORY_LABELS[verdict]
    if backend == "outlook":
        return _outlook_apply_category(mid, label)
    if backend == "gmail":
        return _gmail_apply_label(mid, label)
    return _outlook_apply_category(mid, label) or _gmail_apply_label(mid, label)


# ─── Haiku triage classifier ─────────────────────────────────────────────

def _triage_message(msg: dict) -> str | None:
    """Return one of LLM_VERDICTS, or None if the LLM is unavailable."""
    if not ENABLE_LLM_TRIAGE:
        return None
    _claude_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if _claude_ok:
        try:
            import anthropic  # type: ignore
        except Exception:
            _claude_ok = False
    snippet = (msg.get("snippet") or msg.get("body_text") or "")[:600]
    prompt = (
        "You triage inbox email for a personal assistant. Classify the "
        "message into EXACTLY one label (lowercase, no extra text):\n"
        "  urgent     — direct personal message needing a same-day reply, "
        "a meeting / calendar invite for the next 24h, an account alert "
        "the user must act on, a bill / payment due, a security "
        "challenge.\n"
        "  fyi        — informational, useful to know but no action "
        "required (project status, automated digest from a service the "
        "user cares about, ticket updates).\n"
        "  newsletter — bulk email the user opted into (Substack, "
        "company newsletter, weekly digest).\n"
        "  spam       — marketing, cold outreach, generic promotional, "
        "phishing.\n\n"
        f"From: {msg.get('from_name') or msg.get('from_addr') or '(unknown)'}"
        f" <{msg.get('from_addr') or ''}>\n"
        f"Subject: {msg.get('subject') or '(no subject)'}\n"
        f"Preview: {snippet or '(empty)'}\n\n"
        "Answer with only the single label, lowercase."
    )
    _triage_system = ("You are a precise inbox triage classifier. Respond "
                      "with one of: urgent, fyi, newsletter, spam.")

    def _parse_verdict(text: str) -> str | None:
        verdict = re.sub(r"[^a-z]", "", (text.strip().lower().split()[:1] or [""])[0])
        return verdict if verdict in LLM_VERDICTS else None

    if _claude_ok:
        try:
            # max_retries=1: the SDK default is 2, and this runs on the
            # voice/dispatch thread — an un-capped client turns the 8s
            # per-attempt timeout into a ~24-40s stall per message.
            # 2026-07-14 bug-hunt #16.
            client = anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS, max_retries=1)
            resp = client.messages.create(
                model=LLM_MODEL,
                max_tokens=8,
                system=_triage_system,
                messages=[{"role": "user", "content": prompt}],
                timeout=LLM_TIMEOUT_SECONDS,
            )
            text = ""
            for block in getattr(resp, "content", []) or []:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    text += t
            verdict = _parse_verdict(text)
            if verdict:
                return verdict
        except Exception as e:
            _log.debug("[email] triage LLM failed: %s", e)

    # Local-Ollama fallback (Claude capped / unavailable / no API key).
    try:
        bc = sys.modules.get("bobert_companion") or importlib.import_module("bobert_companion")
        local = bc._call_local_llm(_triage_system, [{"role": "user", "content": prompt}], max_tokens=8)
        if local:
            verdict = _parse_verdict(local)
            if verdict:
                return verdict
    except Exception as e:
        _log.debug("[email] triage local fallback failed: %s", e)
    return None


def _generate_draft_reply(thread: dict, user_instructions: str = "") -> str | None:
    """Ask Haiku to draft a concise reply for the given thread. Honours
    optional ``user_instructions`` ('keep it short', 'politely decline',
    'agree to Tuesday at 2pm')."""
    _claude_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if _claude_ok:
        try:
            import anthropic  # type: ignore
        except Exception:
            _claude_ok = False
    body_text = (thread.get("body_text") or thread.get("snippet") or "")[:2000]
    sender = thread.get("from_name") or thread.get("from_addr") or "the sender"
    subject = thread.get("subject") or "(no subject)"
    instr_block = ""
    if user_instructions.strip():
        instr_block = (
            "User instructions for this reply (follow them carefully):\n"
            f"  {user_instructions.strip()}\n\n"
        )
    prompt = (
        "Draft a short, professional reply to the email below. Match the "
        "tone of the original. Be concise — three short paragraphs MAX, "
        "ideally one. Don't repeat the subject or sender's name. Sign off "
        "with the user's first name only. Don't include any "
        "preamble like 'Here is your reply' — output only the body of the "
        f"reply.\n\n{instr_block}"
        f"From: {sender}\n"
        f"Subject: {subject}\n\n"
        f"Original message:\n{body_text}\n"
    )
    _draft_system = ("You are a helpful email drafting assistant. Output "
                     "only the body of a reply email — no headers, no "
                     "subject line, no preamble. Plain text, no markdown.")

    def _finish_draft(text: str) -> str | None:
        text = (text or "").strip()
        if len(text) > MAX_DRAFT_BODY_CHARS:
            text = text[:MAX_DRAFT_BODY_CHARS].rstrip() + "…"
        return text or None

    if _claude_ok:
        try:
            # max_retries=1: the SDK default is 2, and this runs on the
            # voice/dispatch thread — an un-capped client turns the 8s
            # per-attempt timeout into a ~24-40s stall per message.
            # 2026-07-14 bug-hunt #16.
            client = anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS, max_retries=1)
            resp = client.messages.create(
                model=LLM_MODEL,
                max_tokens=600,
                system=_draft_system,
                messages=[{"role": "user", "content": prompt}],
                timeout=LLM_TIMEOUT_SECONDS * 2,
            )
            text = ""
            for block in getattr(resp, "content", []) or []:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    text += t
            result = _finish_draft(text)
            if result:
                return result
        except Exception as e:
            _log.warning("[email] draft generation failed: %s", e)

    # Local-Ollama fallback (Claude capped / unavailable / no API key).
    try:
        bc = sys.modules.get("bobert_companion") or importlib.import_module("bobert_companion")
        local = bc._call_local_llm(_draft_system, [{"role": "user", "content": prompt}], max_tokens=600)
        if local:
            return _finish_draft(local)
    except Exception as e:
        _log.warning("[email] draft local fallback failed: %s", e)
    return None


# ─── Pending draft state ─────────────────────────────────────────────────

def _load_pending() -> dict:
    if not os.path.exists(PENDING_DRAFTS_FILE):
        return {"active": None, "history": []}
    try:
        with open(PENDING_DRAFTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"active": None, "history": []}


def _save_pending(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(PENDING_DRAFTS_FILE), exist_ok=True)
        _atomic_write_json(PENDING_DRAFTS_FILE, state)
    except Exception as e:
        _log.debug("[email] pending save failed: %s", e)


def _set_pending(record: dict | None) -> None:
    with _state_lock:
        state = _load_pending()
        if state.get("active"):
            state.setdefault("history", []).append(state["active"])
            state["history"] = state["history"][-20:]
        state["active"] = record
        _save_pending(state)


def _get_pending() -> dict | None:
    with _state_lock:
        return _load_pending().get("active")


# Public accessor for the draft-preview gate (core/draft_preview_gate.py).
# Exposing this under a non-underscore name keeps the gate from importing
# private helpers and lets future skills surface "what's queued for send?"
# without reaching into module internals.
def get_pending_draft() -> dict | None:
    return _get_pending()


def _clear_pending() -> None:
    _set_pending(None)


# ─── Voice action implementations ────────────────────────────────────────

def _format_msg_line(m: dict, idx: int | None = None) -> str:
    who = m.get("from_name") or m.get("from_addr") or "(unknown)"
    subj = (m.get("subject") or "(no subject)").strip()
    backend = (m.get("backend") or "?").title()
    prefix = f"{idx}. " if idx is not None else ""
    return f"{prefix}{who} — {subj} [{backend}]"


def _format_for_speech(text: str) -> str:
    text = (text or "").strip()
    if len(text) > MAX_SPOKEN_BODY_CHARS:
        text = text[: MAX_SPOKEN_BODY_CHARS - 1].rstrip() + "…"
    return text


def action_list_unread(arg: str = "") -> str:
    """List up to N unread messages across both backends."""
    n = DEFAULT_LIST_LIMIT
    backend = "all"
    if arg.strip():
        for tok in arg.lower().split():
            if tok in ("gmail", "outlook"):
                backend = tok
            elif tok.isdigit():
                n = max(1, min(20, int(tok)))
    messages = list_unread(top_n=n, backend=backend)
    if not messages:
        if not _outlook_configured() and not is_gmail_available():
            return ("No email backends configured, sir. Run "
                    "`python -m skills.ms_graph --auth` for Outlook, or "
                    "`python -m skills.email_triage --auth-gmail` for Gmail.")
        return "Inbox is clear, sir."
    _save_inbox_index(messages)
    header = f"{len(messages)} unread, sir:"
    lines = [_format_msg_line(m, i) for i, m in enumerate(messages, 1)]
    return header + "\n" + "\n".join(lines)


def action_read_thread(arg: str = "") -> str:
    token = (arg or "latest").strip()
    handle = _resolve_handle(token)
    if handle is None:
        return "I need a number or message id to read, sir."
    thread = _get_thread(handle)
    if thread is None:
        return f"Couldn't fetch that one, sir — id {token} not found."
    sender = thread.get("from_name") or thread.get("from_addr") or "(unknown sender)"
    subject = thread.get("subject") or "(no subject)"
    body = thread.get("body_text") or thread.get("snippet") or "(no body)"
    body = _format_for_speech(body)
    backend = (thread.get("backend") or handle.get("backend") or "?").title()
    # Mark read so it stops showing up in subsequent unread lists.
    _mark_read({"backend": thread.get("backend"), "id": thread.get("id")})
    return (
        f"Sir, {backend} message from {sender} — subject: {subject}.\n\n"
        f"{body}"
    )


def action_draft_reply(arg: str = "") -> str:
    """draft_reply <handle> [user instructions...]

    handle: 'latest' / a number / a raw backend id. Optional instructions
    after the handle are passed to Haiku verbatim."""
    parts = (arg or "").strip().split(None, 1)
    token = parts[0] if parts else "latest"
    instructions = parts[1] if len(parts) > 1 else ""
    handle = _resolve_handle(token)
    if handle is None:
        return "Tell me which message to reply to, sir."
    thread = _get_thread(handle)
    if thread is None:
        return f"Couldn't fetch that message, sir — id {token} not found."
    candidate = _generate_draft_reply(thread, instructions)
    if not candidate:
        return ("Draft generation unavailable, sir — set ANTHROPIC_API_KEY "
                "and install the anthropic SDK.")
    backend = thread.get("backend") or handle.get("backend") or "auto"
    # Create the draft on the backend so it also shows up in the user's
    # mail client. The draft id stays in pending state for confirm/scrap.
    draft_id = None
    if backend == "outlook":
        draft_id = _outlook_create_draft(thread.get("id") or "", candidate)
    elif backend == "gmail":
        draft_id = _gmail_create_draft(thread.get("id") or "", candidate)
    else:
        draft_id = (_outlook_create_draft(thread.get("id") or "", candidate)
                    or _gmail_create_draft(thread.get("id") or "", candidate))
    record = {
        "ts":        time.time(),
        "backend":   thread.get("backend"),
        "message_id": thread.get("id"),
        "draft_id":  draft_id,
        "subject":   thread.get("subject"),
        "to":        thread.get("from_addr"),
        "body":      candidate,
    }
    _set_pending(record)
    backend_label = (thread.get("backend") or "?").title()
    persisted = "saved to your Drafts folder" if draft_id else "held locally (backend write failed)"
    return (
        f"Sir, draft reply ready for {backend_label} message to "
        f"{thread.get('from_name') or thread.get('from_addr')} — {persisted}. "
        f"Say 'send', 'scrap', or 'edit <new text>'.\n\n"
        f"DRAFT:\n{candidate}"
    )


def action_confirm_pending_draft(_: str = "") -> str:
    pending = _get_pending()
    if not pending:
        return "No draft waiting, sir."
    backend = pending.get("backend")
    draft_id = pending.get("draft_id")
    if draft_id:
        sent = (_outlook_send_draft(draft_id) if backend == "outlook"
                else _gmail_send_draft(draft_id))
        if sent:
            _clear_pending()
            return (f"Sent the reply to "
                    f"{pending.get('to') or 'the recipient'}, sir.")
        return ("Backend refused to send, sir — the draft is still in your "
                "Drafts folder if you'd like to send it manually.")
    return ("Draft has no backend id, sir — open Drafts and send it manually, "
            "or run draft_reply again so I can recreate it on the server.")


def action_scrap_pending_draft(_: str = "") -> str:
    pending = _get_pending()
    if not pending:
        return "Nothing pending to scrap, sir."
    _clear_pending()
    return ("Scrapped the draft, sir. (The empty draft may still appear in "
            "your Drafts folder — delete it there if it bothers you.)")


def action_edit_pending_draft(arg: str = "") -> str:
    new_body = (arg or "").strip()
    if not new_body:
        return "Tell me the new body text to use, sir."
    pending = _get_pending()
    if not pending:
        return "No draft waiting, sir — call draft_reply first."
    backend = pending.get("backend")
    draft_id = pending.get("draft_id")
    msg_id = pending.get("message_id")
    updated = False
    if draft_id:
        if backend == "outlook":
            updated = _outlook_update_draft(draft_id, new_body)
        elif backend == "gmail":
            updated = _gmail_update_draft(draft_id, new_body, msg_id or "")
    pending["body"] = new_body
    _set_pending(pending)
    if draft_id and not updated:
        return ("Updated the local copy, sir, but the backend rejected the "
                "PATCH — send may still go using the old text.")
    return f"Draft updated, sir. Say 'send' or 'scrap'.\n\nDRAFT:\n{new_body}"


def action_list_pending_drafts(_: str = "") -> str:
    pending = _get_pending()
    if not pending:
        return "No pending drafts, sir."
    body = (pending.get("body") or "").strip()
    preview = body if len(body) < 400 else body[:399] + "…"
    return (f"Pending draft to {pending.get('to') or '(unknown)'} "
            f"[{pending.get('backend')}]:\n{preview}")


def action_archive_email(arg: str = "") -> str:
    token = (arg or "latest").strip()
    handle = _resolve_handle(token)
    if handle is None:
        return "Tell me which message to archive, sir."
    ok = _archive(handle)
    return "Archived, sir." if ok else "Couldn't archive that one, sir."


def action_categorize_inbox(arg: str = "") -> str:
    """Run Haiku across the current unread set, apply backend categories /
    labels per verdict. Optional arg = top_n (default 20). Returns the
    counts so the user can hear 'sir, three urgent, two FYI, the rest
    newsletter or spam'."""
    try:
        n = int(arg.strip()) if arg.strip() else 20
    except Exception:
        n = 20
    n = max(1, min(50, n))
    messages = list_unread(top_n=n)
    if not messages:
        return "Inbox is clear, sir — nothing to categorize."
    _save_inbox_index(messages)
    counts = {v: 0 for v in LLM_VERDICTS}
    processed = 0
    # Wall-clock deadline (2026-07-14 bug-hunt #16). This runs synchronously on
    # the voice/dispatch thread and _triage_message makes a per-message Claude
    # call; up to 50 messages x ~16s each is a many-minute freeze. Stop
    # triaging once the budget is spent and report what we got through.
    _deadline = time.monotonic() + INBOX_TRIAGE_BUDGET_S
    truncated = False
    for m in messages:
        if time.monotonic() >= _deadline:
            truncated = True
            break
        verdict = _triage_message(m)
        if verdict is None:
            continue
        processed += 1
        counts[verdict] = counts.get(verdict, 0) + 1
        _apply_category({"backend": m.get("backend"), "id": m.get("id")}, verdict)
    if processed == 0:
        return ("Triaged nothing, sir — set ANTHROPIC_API_KEY and install "
                "the anthropic SDK to enable the classifier.")
    summary = ", ".join(f"{counts[v]} {v}" for v in LLM_VERDICTS if counts[v])
    tail = (f" (stopped at {processed} to stay responsive — say 'categorize "
            f"inbox' again for the rest)" if truncated else "")
    return f"Triaged {processed} messages, sir — {summary}.{tail}"


def action_email_briefing(_: str = "") -> str:
    """Spoken summary for the morning briefing. Pulls the unread set,
    classifies (best-effort) and speaks the urgent items first, then a
    tail count of FYI / newsletter / spam."""
    messages = list_unread(top_n=15)
    if not messages:
        return "Inbox is clear, sir."
    _save_inbox_index(messages)
    urgent: list[dict] = []
    other_counts = {"fyi": 0, "newsletter": 0, "spam": 0, "unclassified": 0}
    _deadline = time.monotonic() + INBOX_TRIAGE_BUDGET_S   # voice-thread bound (#16)
    for m in messages:
        if time.monotonic() >= _deadline:
            break
        verdict = _triage_message(m)
        if verdict == "urgent":
            urgent.append(m)
        elif verdict in other_counts:
            other_counts[verdict] += 1
        else:
            other_counts["unclassified"] += 1
    if not urgent and not any(other_counts.values()):  # pragma: no cover - unreachable: the loop buckets every non-empty message into urgent or a count, so this only fires on an empty set already handled above
        return ""
    bits: list[str] = []
    if urgent:
        for m in urgent[:3]:
            who = m.get("from_name") or m.get("from_addr") or "an unknown sender"
            subj = (m.get("subject") or "").strip()
            bits.append(f"urgent message from {who}"
                        + (f" — {subj}" if subj else ""))
    tail = []
    if other_counts["fyi"]:
        tail.append(f"{other_counts['fyi']} FYI")
    if other_counts["newsletter"]:
        tail.append(f"{other_counts['newsletter']} newsletter")
    if other_counts["spam"]:
        tail.append(f"{other_counts['spam']} spam")
    if other_counts["unclassified"]:
        tail.append(f"{other_counts['unclassified']} other")
    if tail:
        bits.append("plus " + ", ".join(tail))
    return "; ".join(bits)


def action_email_triage_status(_: str = "") -> str:
    bits = []
    if _outlook_configured():
        bits.append("Outlook: configured")
    else:
        bits.append("Outlook: not configured (run `python -m skills.ms_graph --auth`)")
    if is_gmail_available():
        token_ok = os.path.exists(GMAIL_TOKEN_FILE)
        bits.append("Gmail: token present" if token_ok
                    else "Gmail: creds present, token missing (run `python -m skills.email_triage --auth-gmail`)")
    else:
        bits.append(f"Gmail: unavailable — {_gmail_unavailable_reason[0] or 'deps + credentials missing'}")
    pending = _get_pending()
    if pending:
        bits.append(f"pending draft to {pending.get('to') or '?'}")
    else:
        bits.append("no pending drafts")
    return "Email triage — " + "; ".join(bits) + ", sir."


# ─── Skill registration ──────────────────────────────────────────────────

def register(actions):
    actions["list_unread"]              = action_list_unread
    actions["unread_email"]             = action_list_unread
    actions["unread_emails"]            = action_list_unread
    actions["list_emails"]              = action_list_unread

    actions["read_thread"]              = action_read_thread
    actions["read_email"]               = action_read_thread
    actions["read_message"]             = action_read_thread

    actions["draft_reply"]              = action_draft_reply
    actions["pre_draft_reply"]          = action_draft_reply
    actions["compose_reply"]            = action_draft_reply

    actions["archive_email"]            = action_archive_email
    actions["archive_message"]          = action_archive_email

    actions["categorize_inbox"]         = action_categorize_inbox
    actions["triage_inbox"]             = action_categorize_inbox
    actions["categorise_inbox"]         = action_categorize_inbox  # UK spelling

    actions["email_briefing"]           = action_email_briefing
    actions["inbox_briefing"]           = action_email_briefing

    actions["confirm_pending_draft"]    = action_confirm_pending_draft
    actions["send_draft"]               = action_confirm_pending_draft
    actions["send_pending_draft"]       = action_confirm_pending_draft

    actions["scrap_pending_draft"]      = action_scrap_pending_draft
    actions["discard_draft"]            = action_scrap_pending_draft

    actions["edit_pending_draft"]       = action_edit_pending_draft

    actions["list_pending_drafts"]      = action_list_pending_drafts
    actions["pending_drafts"]           = action_list_pending_drafts

    actions["email_triage_status"]      = action_email_triage_status

    print("  [email_triage] ready — actions: list_unread, read_thread, "
          "draft_reply, archive_email, categorize_inbox, email_briefing, "
          "confirm/scrap/edit_pending_draft, email_triage_status.")


# ─── CLI: gmail auth + smoke test ────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover - CLI smoke/auth entry; exercised by hand, not under unittest
    if "--auth-gmail" in sys.argv:
        ok = authenticate_gmail()
        sys.exit(0 if ok else 1)
    print(action_email_triage_status())
    print()
    print(action_list_unread("5"))
