"""
ms_graph.py — Microsoft Graph client for JARVIS calendar + mail.

Replaces the Outlook COM lookups in hud_card.py / morning_briefing.py that
were failing with 'not connected' whenever Outlook desktop wasn't running.
Microsoft Graph works from anywhere (the user can be signed into web Outlook
or have no Outlook desktop at all) and returns both calendar AND unread
mail counts.

Two ways to authenticate:

  1. MSAL device-code flow (recommended): set MS_GRAPH_CLIENT_ID on
     bobert_companion to your Azure AD public-client app's client id, then
     run:
         python -m skills.ms_graph --auth
     A code + URL is printed; sign in once and the refresh token is cached
     for ~90 days, after which the device flow must be re-run.

  2. Manual token file: write microsoft_graph_token.json with the shape
         {"access_token": "...", "expires_at": <epoch seconds>}
     optionally including "refresh_token" + MS_GRAPH_CLIENT_ID so silent
     refresh works.

All getters return None / [] silently when no token is available — callers
degrade gracefully (no banner, no crash).
"""
from __future__ import annotations

import concurrent.futures
import datetime
import importlib
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

_PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOKEN_FILE     = os.path.join(_PROJECT_DIR, "microsoft_graph_token.json")
_MSAL_CACHE_FILE = os.path.join(_PROJECT_DIR, "ms_graph_msal_cache.json")
_GRAPH_BASE     = "https://graph.microsoft.com/v1.0"
_GRAPH_TIMEOUT  = 8.0
# MSAL's acquire_token_silent() does a network refresh with no timeout of its
# own, so a stalled Microsoft endpoint would block the voice turn forever. Cap
# it on a worker thread; on timeout we fall through to the bounded manual
# refresh-token path in get_access_token().
_MSAL_SILENT_TIMEOUT = 8.0
# Default scopes now include Mail.ReadWrite (archive / categorize) and Mail.Send
# (send drafts) so the email triage skill can act on messages, not just read
# unread counts. Override via MS_GRAPH_SCOPES on bobert_companion if the user
# wants a narrower consent surface.
_DEFAULT_SCOPES = [
    "Calendars.Read",
    "Mail.Read",
    "Mail.ReadWrite",
    "Mail.Send",
    "Chat.Read",
]

_lock = threading.Lock()


# ─── config + small helpers ──────────────────────────────────────────────

def _config(name: str, default):
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return default
    return getattr(bc, name, default)


def _atomic_write_json(path: str, payload: dict) -> None:
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except Exception: pass
        raise


# ─── token encryption at rest (Windows DPAPI) ────────────────────────────
# The access_token + refresh_token used to sit in microsoft_graph_token.json
# as plaintext (2026-05-30 audit). DPAPI (CryptProtectData) encrypts the blob
# bound to THIS Windows user + machine, so the ciphertext is useless if the
# file is copied elsewhere. Mirrors skills/network_deco.py's helpers. Falls
# back to plaintext (with a warning) if pywin32 is unavailable so Graph never
# hard-breaks on machines without win32crypt. On-disk encrypted shape is
# {"dpapi": "<base64>"}; the base64 decrypts to the original token JSON, so
# the dict handed back to callers is byte-for-byte the legacy shape.
import base64 as _base64


def _dpapi_encrypt(plaintext: str) -> str | None:
    """Encrypt with Windows DPAPI; return base64 ciphertext, or None if DPAPI
    isn't available (caller then keeps plaintext as a last resort)."""
    if not plaintext:
        return None
    try:
        import win32crypt
        blob = win32crypt.CryptProtectData(
            plaintext.encode("utf-8"), "jarvis-ms-graph", None, None, None, 0)
        return _base64.b64encode(blob).decode("ascii")
    except Exception:
        return None


def _dpapi_decrypt(b64: str) -> str | None:
    """Decrypt DPAPI base64 ciphertext back to plaintext, or None on failure."""
    if not b64:
        return None
    try:
        import win32crypt
        blob = _base64.b64decode(b64.encode("ascii"))
        _desc, data = win32crypt.CryptUnprotectData(blob, None, None, None, 0)
        return data.decode("utf-8")
    except Exception:
        return None


def _load_token() -> dict | None:
    if not os.path.exists(_TOKEN_FILE):
        return None
    try:
        with open(_TOKEN_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    # Encrypted form: {"dpapi": "<base64>"} → decrypt to the original dict.
    enc = raw.get("dpapi")
    if enc and "access_token" not in raw:
        dec = _dpapi_decrypt(enc)
        if dec:
            try:
                return json.loads(dec)
            except Exception:
                return None
        # Ciphertext present but undecryptable (file copied from another
        # machine/user, or pywin32 missing). Nothing usable here.
        return None
    # Legacy plaintext token dict — use it, but AUTO-MIGRATE to encrypted so
    # the secrets stop living on disk in the clear. Best-effort: a failed
    # migration just leaves the plaintext as-is rather than breaking Graph.
    try:
        _save_token(raw)
    except Exception:
        pass
    return raw


def _save_token(token: dict) -> None:
    try:
        blob = json.dumps(token)
        enc = _dpapi_encrypt(blob)
        with _lock:
            if enc:
                _atomic_write_json(_TOKEN_FILE, {"dpapi": enc})
            else:
                # pywin32 unavailable — keep the legacy plaintext behaviour so
                # Graph still works, but make the downgrade visible.
                print("  [ms_graph] win32crypt unavailable; storing token "
                      "UNENCRYPTED (install pywin32 to encrypt at rest)")
                _atomic_write_json(_TOKEN_FILE, token)
    except Exception as e:
        print(f"  [ms_graph] token save failed: {e}")


# ─── MSAL ────────────────────────────────────────────────────────────────

def _msal_app():
    """Build an MSAL PublicClientApplication or return None if msal isn't
    installed or MS_GRAPH_CLIENT_ID isn't configured."""
    client_id = (_config("MS_GRAPH_CLIENT_ID", "") or "").strip()
    if not client_id:
        return None
    try:
        import msal  # type: ignore
    except Exception:
        return None
    tenant_id = (_config("MS_GRAPH_TENANT_ID", "common") or "common").strip()
    authority = f"https://login.microsoftonline.com/{tenant_id}"

    cache = msal.SerializableTokenCache()
    if os.path.exists(_MSAL_CACHE_FILE):
        try:
            with open(_MSAL_CACHE_FILE, "r", encoding="utf-8") as f:
                cache.deserialize(f.read())
        except Exception:
            pass

    app = msal.PublicClientApplication(
        client_id, authority=authority, token_cache=cache,
    )
    # Stash the cache so _save_msal_cache can persist it after token ops.
    app._jarvis_cache = cache       # type: ignore[attr-defined]
    return app


def _save_msal_cache(app) -> None:
    try:
        cache = getattr(app, "_jarvis_cache", None)
        if cache is None or not cache.has_state_changed:
            return
        with _lock:
            with open(_MSAL_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(cache.serialize())
    except Exception as e:
        print(f"  [ms_graph] msal cache save failed: {e}")


def _get_access_token_msal() -> str | None:
    app = _msal_app()
    if app is None:
        return None
    scopes = list(_config("MS_GRAPH_SCOPES", _DEFAULT_SCOPES) or _DEFAULT_SCOPES)
    try:
        accounts = app.get_accounts()
        if not accounts:
            return None
        # acquire_token_silent may hit the network (refresh) with no timeout of
        # its own; run it on a worker thread and bound the wait so a stalled
        # endpoint can't hang the voice turn. On timeout, return None to fall
        # through to the bounded manual refresh-token path. shutdown(wait=False)
        # so a still-running call doesn't re-block us — the orphaned worker
        # finishes in the background and is reaped by the interpreter.
        _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            _fut = _ex.submit(
                app.acquire_token_silent, scopes, account=accounts[0])
            try:
                result = _fut.result(timeout=_MSAL_SILENT_TIMEOUT)
            except concurrent.futures.TimeoutError:
                print("  [ms_graph] msal silent acquire timed out; "
                      "falling back to manual refresh")
                return None
        finally:
            _ex.shutdown(wait=False)
    except Exception as e:
        print(f"  [ms_graph] msal silent acquire failed: {e}")
        return None
    _save_msal_cache(app)
    if result and "access_token" in result:
        return result["access_token"]
    return None


# ─── manual-token-file refresh (fallback when MSAL isn't installed) ──────

def _refresh_with_refresh_token(token: dict) -> dict | None:
    refresh = token.get("refresh_token")
    client_id = (_config("MS_GRAPH_CLIENT_ID", "") or "").strip()
    if not refresh or not client_id:
        return None
    tenant_id = (_config("MS_GRAPH_TENANT_ID", "common") or "common").strip()
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    scopes = list(_config("MS_GRAPH_SCOPES", _DEFAULT_SCOPES) or _DEFAULT_SCOPES)
    body = urllib.parse.urlencode({
        "client_id":     client_id,
        "grant_type":    "refresh_token",
        "refresh_token": refresh,
        "scope":         " ".join(scopes),
    }).encode("ascii")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_GRAPH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [ms_graph] refresh failed: {e}")
        return None
    if "access_token" not in data:
        return None
    new_token = {
        "access_token":  data["access_token"],
        "expires_at":    time.time() + int(data.get("expires_in", 3600)) - 60,
        "refresh_token": data.get("refresh_token", refresh),
    }
    _save_token(new_token)
    return new_token


def get_access_token() -> str | None:
    """Return a fresh access token (string), or None if no auth is available.
    Tries MSAL first, then a manual token file with refresh."""
    tok = _get_access_token_msal()
    if tok:
        return tok
    token = _load_token()
    if not token:
        return None
    expires_at = float(token.get("expires_at", 0.0) or 0.0)
    if expires_at > time.time() + 30:
        return token.get("access_token") or None
    refreshed = _refresh_with_refresh_token(token)
    if refreshed:
        return refreshed.get("access_token") or None
    return None


# ─── Graph HTTP ──────────────────────────────────────────────────────────

def _graph_get(path: str, params: dict | None = None) -> dict | None:
    access_token = get_access_token()
    if not access_token:
        return None
    url = f"{_GRAPH_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {access_token}",
        "Accept":        "application/json",
        "Prefer":        'outlook.timezone="UTC"',
    })
    try:
        with urllib.request.urlopen(req, timeout=_GRAPH_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            detail = ""
        print(f"  [ms_graph] http {e.code} on {path}: {e.reason} {detail}")
        return None
    except Exception as e:
        print(f"  [ms_graph] {path} failed: {e}")
        return None


def _graph_call(method: str, path: str, body: dict | None = None,
                params: dict | None = None) -> tuple[int, dict | None]:
    """POST / PATCH / DELETE / PUT against Graph. Returns (status, payload-or-None).

    Status 0 indicates 'no token / transport error'. A successful no-content
    response (e.g. DELETE) returns (204, None). Body is JSON-encoded.
    """
    access_token = get_access_token()
    if not access_token:
        return 0, None
    url = f"{_GRAPH_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method.upper())
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Accept",        "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=_GRAPH_TIMEOUT) as resp:
            raw = resp.read()
            if not raw:
                return resp.status, None
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except Exception:
                return resp.status, None
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            detail = ""
        print(f"  [ms_graph] http {e.code} on {method} {path}: {e.reason} {detail}")
        return e.code, None
    except Exception as e:
        print(f"  [ms_graph] {method} {path} failed: {e}")
        return 0, None


# ─── windows + parsing ───────────────────────────────────────────────────

def _meeting_window(when: str) -> tuple[datetime.datetime, datetime.datetime]:
    """Resolve a window keyword to (start, end) naive local datetimes."""
    now = datetime.datetime.now()
    if when == "tomorrow":
        d = (now + datetime.timedelta(days=1)).date()
        return (
            datetime.datetime.combine(d, datetime.time(0, 0)),
            datetime.datetime.combine(d, datetime.time(23, 59, 59)),
        )
    if when == "next_14_days":
        return now, now + datetime.timedelta(days=14)
    # default: today, now → end of day
    return now, now.replace(hour=23, minute=59, second=59, microsecond=0)


def _parse_graph_start(evt: dict) -> datetime.datetime | None:
    """Turn evt['start'] (Graph dateTime + timeZone) into a naive local dt."""
    try:
        raw = evt["start"]["dateTime"]
        tz  = (evt["start"].get("timeZone") or "UTC").strip()
    except (KeyError, TypeError):
        return None
    # Graph sometimes includes fractional seconds. fromisoformat handles them
    # in 3.11+, but we strip defensively for older interpreters.
    if "." in raw:
        raw = raw.split(".")[0]
    try:
        dt = datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if tz.upper() == "UTC":
        return (
            dt.replace(tzinfo=datetime.timezone.utc)
              .astimezone(None)
              .replace(tzinfo=None)
        )
    return dt


# ─── public getters ──────────────────────────────────────────────────────

def get_upcoming_events(top_n: int = 3, when: str = "next_14_days") -> list[dict]:
    """Return up to ``top_n`` upcoming events (sorted by start). Each item is
    {start: naive-local datetime, subject: str, organizer: str}.

    ``when`` is one of 'today', 'tomorrow', 'next_14_days'.
    """
    start_dt, end_dt = _meeting_window(when)
    body = _graph_get("/me/calendarView", {
        "startDateTime": start_dt.isoformat(),
        "endDateTime":   end_dt.isoformat(),
        "$orderby":      "start/dateTime",
        "$top":          str(max(1, int(top_n))),
        "$select":       "subject,start,organizer",
    })
    if not body:
        return []
    out: list[dict] = []
    for evt in (body.get("value") or []):
        s_dt = _parse_graph_start(evt)
        if s_dt is None:
            continue
        organizer = ""
        try:
            organizer = (evt["organizer"]["emailAddress"].get("name") or "").strip()
        except Exception:
            pass
        out.append({
            "start":     s_dt,
            "subject":   (evt.get("subject") or "").strip(),
            "organizer": organizer,
        })
    return out


def get_first_meeting(when: str = "today") -> dict | None:
    events = get_upcoming_events(top_n=1, when=when)
    return events[0] if events else None


def get_unread_mail_count() -> int | None:
    """Inbox unread count from Graph, or None if Graph is unavailable."""
    body = _graph_get("/me/mailFolders/Inbox", None)
    if not body:
        return None
    val = body.get("unreadItemCount")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def is_configured() -> bool:
    """True if either MSAL or a manual token file is set up."""
    return bool(_msal_app()) or bool(_load_token())


def get_teams_unread_count() -> dict | None:
    """Count unread Microsoft Teams chats and identify the most recent sender.

    Returns {"count": int, "top_sender": str} where ``top_sender`` is the
    first-name of whoever sent the newest unread message (best-effort), or
    None when Graph is unavailable. ``count`` is 0 when nothing is unread.

    Implementation: Graph has no direct unread-chat counter, so we list the
    user's chats with each chat's lastMessagePreview + viewpoint and compare
    lastMessagePreview.createdDateTime against viewpoint.lastMessageReadDateTime.
    Requires the ``Chat.Read`` delegated scope; tokens issued before that scope
    was added will 403 here and the briefing degrades by skipping the section.
    """
    body = _graph_get("/me/chats", {
        "$expand": "lastMessagePreview",
        "$top":    "50",
    })
    if not body:
        return None
    unread = 0
    senders: list[tuple[str, str]] = []
    for chat in (body.get("value") or []):
        preview = chat.get("lastMessagePreview") or {}
        last_msg_dt = (preview.get("createdDateTime") or "").strip()
        if not last_msg_dt:
            continue
        viewpoint = chat.get("viewpoint") or {}
        last_read_dt = (viewpoint.get("lastMessageReadDateTime") or "").strip()
        # If the user has read at or after the last message, it's not unread.
        if last_read_dt and last_msg_dt <= last_read_dt:
            continue
        # Skip system / bot events that have no human sender — Graph returns
        # from.user == None for those, and we don't want them inflating the count.
        from_obj = preview.get("from") or {}
        user_obj = from_obj.get("user") or {}
        sender = (user_obj.get("displayName") or "").strip()
        if not sender:
            continue
        unread += 1
        senders.append((last_msg_dt, sender))
    if not senders:
        return {"count": unread, "top_sender": ""}
    senders.sort(reverse=True)
    top_full = senders[0][1]
    top_first = top_full.split()[0] if top_full else ""
    return {"count": unread, "top_sender": top_first}


# ─── Email triage primitives ─────────────────────────────────────────────
#
# These return the unified shape used by skills/email_triage.py:
#   {
#       "backend":   "outlook",
#       "id":        "<graph message id>",
#       "from_name": str,
#       "from_addr": str,
#       "subject":   str,
#       "snippet":   str,         # short body preview
#       "received":  iso str,
#       "unread":    bool,
#       "categories": list[str],
#   }
# The triage skill maps these onto its Haiku classifier and voice actions.

def _shape_outlook_message(m: dict) -> dict:
    """Normalise a Graph /messages JSON object into the unified shape."""
    sender = (m.get("from") or {}).get("emailAddress") or {}
    return {
        "backend":    "outlook",
        "id":         m.get("id") or "",
        "from_name":  (sender.get("name") or "").strip(),
        "from_addr":  (sender.get("address") or "").strip(),
        "subject":    (m.get("subject") or "").strip(),
        "snippet":    (m.get("bodyPreview") or "").strip(),
        "received":   m.get("receivedDateTime") or "",
        "unread":     not bool(m.get("isRead", True)),
        "categories": list(m.get("categories") or []),
    }


def list_unread_messages(top_n: int = 10) -> list[dict]:
    """Return up to ``top_n`` unread Outlook inbox messages in unified shape.

    Returns [] if Graph isn't configured / reachable so callers can chain
    cleanly across backends.
    """
    top_n = max(1, min(50, int(top_n)))
    body = _graph_get("/me/mailFolders/Inbox/messages", {
        "$filter":  "isRead eq false",
        "$orderby": "receivedDateTime desc",
        "$top":     str(top_n),
        "$select":  "id,subject,from,bodyPreview,receivedDateTime,"
                    "isRead,categories",
    })
    if not body:
        return []
    return [_shape_outlook_message(m) for m in (body.get("value") or [])]


def get_message_thread(message_id: str) -> dict | None:
    """Fetch a single message with its full body. Returns the unified shape
    plus a ``body_text`` key (HTML stripped to plaintext for TTS) and
    ``body_html`` (raw HTML if Graph returned that). None on failure."""
    if not message_id:
        return None
    body = _graph_get(f"/me/messages/{urllib.parse.quote(message_id, safe='')}", {
        "$select": "id,subject,from,toRecipients,ccRecipients,bodyPreview,"
                   "body,receivedDateTime,isRead,categories,conversationId",
    })
    if not body:
        return None
    shaped = _shape_outlook_message(body)
    body_obj = body.get("body") or {}
    raw = (body_obj.get("content") or "").strip()
    content_type = (body_obj.get("contentType") or "").lower()
    shaped["body_text"] = _strip_html(raw) if content_type == "html" else raw
    shaped["body_html"] = raw if content_type == "html" else ""
    shaped["conversation_id"] = body.get("conversationId") or ""
    shaped["to"] = [
        (r.get("emailAddress") or {}).get("address", "")
        for r in (body.get("toRecipients") or [])
    ]
    return shaped


def _strip_html(html: str) -> str:
    """Very lightweight HTML → text for read-aloud / Haiku triage. Avoids
    pulling in a real HTML parser dep just for inbox snippets."""
    if not html:
        return ""
    import re as _re
    # Drop <style>/<script> bodies first so their CSS / JS doesn't leak.
    cleaned = _re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    # Convert <br> and </p> to newlines for readable structure.
    cleaned = _re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", cleaned)
    cleaned = _re.sub(r"(?i)</\s*p\s*>", "\n", cleaned)
    cleaned = _re.sub(r"<[^>]+>", " ", cleaned)
    # Collapse whitespace and unescape the handful of common entities.
    cleaned = (cleaned
        .replace("&nbsp;", " ")
        .replace("&amp;",  "&")
        .replace("&lt;",   "<")
        .replace("&gt;",   ">")
        .replace("&quot;", '"')
        .replace("&#39;",  "'")
    )
    cleaned = _re.sub(r"[ \t]+", " ", cleaned)
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def create_draft_reply(message_id: str, reply_body: str,
                       reply_all: bool = False) -> dict | None:
    """Create a draft reply (or replyAll) for ``message_id`` with ``reply_body``
    as the user's text. The Graph endpoint returns the new draft Message
    resource so callers can later .send() it by id. None on failure.

    Body is sent as plain text (contentType=Text) — the user reviews via
    voice or Outlook UI, so we don't need to author HTML.
    """
    if not message_id:
        return None
    op = "createReplyAll" if reply_all else "createReply"
    status, payload = _graph_call(
        "POST",
        f"/me/messages/{urllib.parse.quote(message_id, safe='')}/{op}",
        body={
            "comment": reply_body or "",
        },
    )
    if 200 <= status < 300 and isinstance(payload, dict):
        return payload
    return None


def send_draft(draft_id: str) -> bool:
    """Send a previously-created draft via the Graph /send endpoint.
    Returns True on 2xx."""
    if not draft_id:
        return False
    status, _ = _graph_call(
        "POST",
        f"/me/messages/{urllib.parse.quote(draft_id, safe='')}/send",
    )
    return 200 <= status < 300


def update_draft_body(draft_id: str, new_body: str) -> bool:
    """PATCH a draft's body text. Useful when the user says 'edit it to say
    X instead'."""
    if not draft_id:
        return False
    status, _ = _graph_call(
        "PATCH",
        f"/me/messages/{urllib.parse.quote(draft_id, safe='')}",
        body={
            "body": {"contentType": "Text", "content": new_body or ""},
        },
    )
    return 200 <= status < 300


def archive_message(message_id: str) -> bool:
    """Move a message to the Archive folder. The well-known folder id
    ``archive`` is supported by Graph so we don't need to look it up first."""
    if not message_id:
        return False
    status, _ = _graph_call(
        "POST",
        f"/me/messages/{urllib.parse.quote(message_id, safe='')}/move",
        body={"destinationId": "archive"},
    )
    return 200 <= status < 300


def apply_category(message_id: str, category: str) -> bool:
    """Add an Outlook category (string label) to a message — merges with any
    existing categories so we don't clobber the user's manual labels."""
    if not message_id or not category:
        return False
    # Fetch current categories so we append rather than replace.
    current = _graph_get(
        f"/me/messages/{urllib.parse.quote(message_id, safe='')}",
        {"$select": "categories"},
    )
    cats = list((current or {}).get("categories") or [])
    if category not in cats:
        cats.append(category)
    status, _ = _graph_call(
        "PATCH",
        f"/me/messages/{urllib.parse.quote(message_id, safe='')}",
        body={"categories": cats},
    )
    return 200 <= status < 300


def mark_as_read(message_id: str, read: bool = True) -> bool:
    """Flip the isRead flag — handy after the triage skill reads a message
    aloud so it doesn't keep speaking the same one every briefing."""
    if not message_id:
        return False
    status, _ = _graph_call(
        "PATCH",
        f"/me/messages/{urllib.parse.quote(message_id, safe='')}",
        body={"isRead": bool(read)},
    )
    return 200 <= status < 300


# ─── device-code flow (interactive, one-time) ────────────────────────────

def authenticate_device_flow() -> bool:
    """Run MSAL device-code flow. Prints a URL + code; caches the result so
    later calls work non-interactively. Returns True on success."""
    app = _msal_app()
    if app is None:
        print(
            "[ms_graph] MSAL unavailable or MS_GRAPH_CLIENT_ID not configured. "
            "Install msal (pip install msal) and set MS_GRAPH_CLIENT_ID on "
            "bobert_companion before running --auth."
        )
        return False
    scopes = list(_config("MS_GRAPH_SCOPES", _DEFAULT_SCOPES) or _DEFAULT_SCOPES)
    flow = app.initiate_device_flow(scopes=scopes)
    if "user_code" not in flow:
        print(f"[ms_graph] device flow init failed: {flow}")
        return False
    print(flow["message"])
    sys.stdout.flush()
    result = app.acquire_token_by_device_flow(flow)
    _save_msal_cache(app)
    if "access_token" in result:
        print("[ms_graph] authentication successful.")
        return True
    print(f"[ms_graph] authentication failed: {result.get('error_description', result)}")
    return False


# ─── orchestrator / voice action: calendar summary ───────────────────────
#
# ms_graph was a pure helper module until now (no register()/actions), so the
# orchestrator's `calendar_scanner` sub-agent (skills/sub_agents/calendar_
# scanner.json) was INERT — its allowed_actions (ms_graph_calendar /
# calendar_today / calendar_next) resolved to nothing, so the worker fetched no
# real data and the merger silently dropped the calendar. Registering a single
# string-returning action here under those names wires the sub-agent to the
# real Graph read in get_upcoming_events(). The orchestrator worker calls an
# action as Callable[[str], str]: it passes a window keyword ("today" /
# "tomorrow" / "next_14_days") and summarises the returned string per the spec's
# system prompt. Graceful degradation mirrors the rest of this module — when
# Graph isn't configured we return a friendly one-liner instead of crashing or
# fabricating, so the worker's no-fabrication contract still holds (a non-empty
# but honest "not set up" line, never invented events).

_CAL_WINDOWS = ("today", "tomorrow", "next_14_days")


def _normalise_when(arg: str) -> str:
    """Map a free-text arg from the planner/worker to a known window keyword.
    Defaults to 'today' (the calendar_scanner sub-agent's primary horizon)."""
    a = (arg or "").strip().lower()
    if not a:
        return "today"
    if "tomorrow" in a:
        return "tomorrow"
    if "week" in a or "14" in a or "upcoming" in a or "next" in a:
        return "next_14_days"
    if a in _CAL_WINDOWS:
        return a
    return "today"


def action_calendar_today(arg: str = "") -> str:
    """Concise, chronological summary of calendar events for a window.

    ``arg`` is an optional window keyword ('today' | 'tomorrow' |
    'next_14_days'); anything else defaults to today. Returns a plain string
    suitable for the orchestrator worker to fold into a briefing, or read aloud
    directly. Degrades gracefully: a friendly 'calendar isn't set up' line when
    Graph has no credentials, and a clear 'nothing scheduled' line when the
    window is empty — never raises, never invents events."""
    when = _normalise_when(arg)
    # Distinguish "no creds" (tell the user how to fix) from "configured but
    # empty" (nothing scheduled) so the summary is honest either way.
    try:
        configured = is_configured()
    except Exception:
        configured = False
    if not configured:
        return ("Calendar isn't set up, sir — run `python -m skills.ms_graph "
                "--auth` to connect Microsoft Graph.")
    try:
        events = get_upcoming_events(top_n=10, when=when)
    except Exception:
        # Any unexpected Graph/parse failure degrades to an honest line rather
        # than bubbling an exception up into the worker.
        return "Couldn't reach the calendar just now, sir."
    label = {"today": "today", "tomorrow": "tomorrow",
             "next_14_days": "the next two weeks"}[when]
    if not events:
        return f"Nothing on the calendar for {label}, sir."
    lines = []
    for e in events:
        start = e.get("start")
        hhmm = start.strftime("%H:%M") if hasattr(start, "strftime") else "??:??"
        subject = (e.get("subject") or "").strip() or "(no title)"
        organizer = (e.get("organizer") or "").strip()
        suffix = f" ({organizer})" if organizer else ""
        lines.append(f"{hhmm} {subject}{suffix}")
    header = f"Calendar for {label}: {len(events)} event" \
             + ("s" if len(events) != 1 else "")
    return header + " — " + "; ".join(lines) + "."


def register(actions):
    """Expose the calendar summary under every name the calendar_scanner
    sub-agent spec references, so the orchestrator can dispatch a real,
    registered action. All three names point at the same string-returning
    callable (the worker auto-runs the spec's first registered allowed_action)."""
    actions["ms_graph_calendar"] = action_calendar_today
    actions["calendar_today"]    = action_calendar_today
    actions["calendar_next"]     = action_calendar_today
    print("  [ms_graph] ready — calendar actions: ms_graph_calendar, "
          "calendar_today, calendar_next.")


# ─── manual smoke test / CLI ─────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover - CLI smoke/auth entry; run by hand, not under unittest
    if "--auth" in sys.argv:
        ok = authenticate_device_flow()
        sys.exit(0 if ok else 1)
    print("Configured:", is_configured())
    print("Today's events:")
    for e in get_upcoming_events(top_n=5, when="today"):
        print(" ", e["start"].strftime("%H:%M"), "—", e["subject"])
    print("Unread mail:", get_unread_mail_count())
