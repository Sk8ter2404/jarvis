"""
Phone bridge — push to / pull from the user's mobile phone.

Closes the "JARVIS only works when I'm at my desk" gap. Three pluggable
backends, configured via env vars; the skill registers cleanly even when
none are configured and every action reports the precise setup hint.

  Telegram (two-way, primary, free):
    TELEGRAM_BOT_TOKEN  — get from @BotFather on Telegram
    TELEGRAM_USER_ID    — your own chat id (one integer, or comma-
                          separated list to whitelist multiple devices).
                          Resolve once by messaging the bot and reading
                          the inbound update via Telegram's getUpdates,
                          or open https://t.me/userinfobot.

  ntfy (one-way push, no account required):
    NTFY_TOPIC          — pick a long random string; subscribe to
                          ntfy.sh/<topic> on the phone (free Android/iOS
                          app + web UI). NO account, NO password — the
                          security model is "the topic name IS the
                          secret". Pick something opaque.

  Pushover (one-way push, paid app):
    PUSHOVER_TOKEN      — application token from pushover.net
    PUSHOVER_USER       — user / group key

Two-way semantics (Telegram only):
  Every inbound text from a whitelisted user is run through the JARVIS
  command pipeline at the PC:
    1. core.dispatcher.resolve_and_dispatch  — multi-step chains
    2. core.dispatcher.match_single_intent   — single named actions
    3. Stateless LLM fallback                — fresh anthropic.Anthropic()
                                               call, isolated from the
                                               local conversation_history
                                               so phone chatter never
                                               pollutes the in-person
                                               session
  The reply is returned to the user via Telegram's sendMessage. Slash-
  commands bypass the pipeline:
    /help    — list available verbs + status
    /status  — backend health summary
    /pause   — pause inbound polling
    /resume  — resume inbound polling
    /mode    — show current conversation mode (smart/agent/controlled)

Why raw HTTP instead of the python-telegram-bot package?
  python-telegram-bot v20+ is async-only (asyncio Application). Telegram
  exposes a perfectly stable HTTPS getUpdates / sendMessage REST API and
  `requests` is already a hard dependency, so we get a 200-line bridge
  with no asyncio thread juggling, no dependency to install, and the
  full feature set we need. The python-telegram-bot lib is only useful
  for advanced workflows (inline keyboards, file uploads, webhooks)
  none of which this bridge currently exercises.

Outbound use from other skills:
    from skill_phone_bridge import push_to_phone
    push_to_phone("Print finished, sir.", priority="high", source="bambu")
  Broadcasts to every configured backend in one call. Failures on one
  backend don't block the others. Returns a dict of {backend: success}.

Proactive pings (core/phone_ping.py, 2026-10-02):
  register() attaches this bridge to core.phone_ping, which decides when
  JARVIS texts the owner unprompted (a print finished / failed, a confirmation
  left unanswered while he is away, a guard alert, a robot that needs him, an
  optional daily summary) — rate-limited, quiet hours, focus mode, per-category
  Settings switches, secrets scrubbed. Other skills call
  skill_utils["ping_phone"](category, message) or ping_phone() below. Without
  a backend that can send UNSOLICITED (Telegram needs TELEGRAM_USER_ID too),
  every ping is a no-op and one boot line says so.

Actions registered:
  notify_phone <message>            send via every configured backend
  push_to_phone <message>           alias for notify_phone
  text_my_phone <message>           alias for notify_phone
  phone_bridge_status               backend availability + last activity
  phone_status                      alias for phone_bridge_status
  list_phone_backends               show configured / unconfigured backends
  pause_phone_bridge                stop inbound polling
  resume_phone_bridge               restart inbound polling
  phone_setup_help                  "how do I connect my phone" — BotFather steps
  phone_ping_status                 are proactive pings on, which kinds, limits
  phone_pings_on / phone_pings_off  the master switch (live + saved)
  phone_ping_test                   send one fixed test ping now
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from typing import Any

# Project-root onto sys.path so `core.*` resolves whether this module is
# loaded as `skills.phone_bridge` or run standalone.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# Atomic-write helper — prefer core.atomic_io, fall back to inline
# mkstemp+replace so this skill loads cleanly even if core is mid-import.
try:
    from core.atomic_io import _atomic_write_json
except Exception:  # pragma: no cover

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

# Pre-send confirmation gate — every user-driven push reads the body
# aloud and waits for a 'yes / send / confirm' token before dispatching.
# Imported lazily-safe at module top because core is on sys.path above.
try:
    from core.draft_confirm import draft_confirm
except Exception as _dc_err:
    draft_confirm = None  # type: ignore[assignment]
    _log.warning("[phone] draft_confirm unavailable (%s); pushes will "
                 "fail-closed unless confirm=False is passed explicitly",
                 _dc_err)

# Proactive-ping policy (core/phone_ping.py). Stdlib-only; None only if core
# itself is broken, and then every ping action says so.
try:
    from core import phone_ping as _phone_ping
except Exception as _pp_err:   # pragma: no cover - core is in-tree
    _phone_ping = None  # type: ignore[assignment]
    _log.warning("[phone] core.phone_ping unavailable (%s); no proactive pings",
                 _pp_err)

# Spoken verbatim (the monolith collects this from every skill).
SPEAK_VERBATIM_ACTIONS = ("phone_setup_help", "phone_ping_status",
                          "phone_pings_on", "phone_pings_off",
                          "phone_ping_test")

# ─── Config ──────────────────────────────────────────────────────────────
TELEGRAM_API_BASE      = "https://api.telegram.org"
TELEGRAM_LONGPOLL_SEC  = 30        # seconds — Telegram keeps the conn open for this long
TELEGRAM_RETRY_SLEEP   = 5.0       # back-off after a getUpdates network failure
TELEGRAM_MAX_MSG_LEN   = 4000      # Telegram's hard cap is 4096; leave headroom
TELEGRAM_INIT_DELAY    = 8.0       # let bobert finish loading before opening sockets
TELEGRAM_HTTP_TIMEOUT  = 35.0      # must exceed TELEGRAM_LONGPOLL_SEC

NTFY_DEFAULT_HOST      = "https://ntfy.sh"
NTFY_HTTP_TIMEOUT      = 10.0

PUSHOVER_URL           = "https://api.pushover.net/1/messages.json"
PUSHOVER_HTTP_TIMEOUT  = 10.0

_STATE_FILE            = os.path.join(_PROJECT_DIR, "data", "phone_bridge_state.json")
_LLM_TIMEOUT_SECONDS   = 12.0
_LLM_MAX_TOKENS        = 400

# Stateless LLM call used to answer phone messages that aren't named
# actions. Mirrors bobert_companion.CLAUDE_MODEL by default but can be
# overridden so phone questions don't burn the user's main model budget.
# This literal is only the last-resort fallback (no env override, monolith
# not importable) - kept equal to core.config.CLAUDE_MODEL's default.
_DEFAULT_LLM_MODEL     = "claude-sonnet-5-5"

# Severity → ntfy / pushover headers
_NTFY_PRIORITY_MAP = {
    "low":      "low",
    "normal":   "default",
    "high":     "high",
    "urgent":   "urgent",
}
_PUSHOVER_PRIORITY_MAP = {
    "low":      -1,
    "normal":   0,
    "high":     1,
    "urgent":   2,        # requires retry/expire params, see _send_pushover
}


# ─── Status / state ──────────────────────────────────────────────────────
_state_lock = threading.RLock()

_status: dict[str, Any] = {
    "started_at":       None,
    "last_inbound_at":  None,
    "last_outbound_at": None,
    "last_error":       None,
    "messages_in":      0,
    "messages_out":     0,
    "polling":          False,
}

_persisted: dict[str, Any] = {
    "last_update_id":   0,           # Telegram's monotonic update offset
    "messages_in":      0,
    "messages_out":     0,
}

_pause_flag        = [False]
_polling_thread    = [None]            # type: ignore[var-annotated]


# ─── Persistence ────────────────────────────────────────────────────────

def _load_state() -> None:
    """Restore persisted state from data/phone_bridge_state.json (if any).
    Tolerates partial / corrupt files — falls back to defaults."""
    if not os.path.exists(_STATE_FILE):
        return
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ("last_update_id", "messages_in", "messages_out"):
                if k in data and isinstance(data[k], int):
                    _persisted[k] = data[k]
    except Exception as e:
        _log.warning("[phone] could not load state: %s", e)


def _save_state() -> None:
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        _atomic_write_json(_STATE_FILE, _persisted)
    except Exception as e:
        _log.warning("[phone] state save failed: %s", e)


# ─── Backend availability probes ─────────────────────────────────────────

def _telegram_token() -> str:
    return (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()


def _telegram_whitelist() -> set[int]:
    """Parse TELEGRAM_USER_ID (comma-separated ints) into a set. Empty
    means "no inbound allowed", since unauthenticated bots get scraped
    fast on Telegram."""
    raw = (os.environ.get("TELEGRAM_USER_ID") or "").strip()
    if not raw:
        return set()
    out: set[int] = set()
    for piece in raw.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.add(int(piece))
        except ValueError:
            _log.warning("[phone] non-integer TELEGRAM_USER_ID entry: %r", piece)
    return out


def _ntfy_config() -> tuple[str | None, str]:
    """Returns (topic, host). Topic None → not configured."""
    topic = (os.environ.get("NTFY_TOPIC") or "").strip()
    host  = (os.environ.get("NTFY_HOST")  or NTFY_DEFAULT_HOST).strip().rstrip("/")
    return (topic or None), host


def _pushover_config() -> tuple[str | None, str | None]:
    tok = (os.environ.get("PUSHOVER_TOKEN") or "").strip() or None
    usr = (os.environ.get("PUSHOVER_USER")  or "").strip() or None
    return tok, usr


def _telegram_configured() -> bool:
    return bool(_telegram_token())


def _ntfy_configured() -> bool:
    topic, _host = _ntfy_config()
    return bool(topic)


def _pushover_configured() -> bool:
    tok, usr = _pushover_config()
    return bool(tok and usr)


def any_backend_configured() -> bool:
    return _telegram_configured() or _ntfy_configured() or _pushover_configured()


def can_push_unsolicited() -> bool:
    """True when at least one backend can message the owner WITHOUT him
    writing first: Telegram only with a TELEGRAM_USER_ID to send to (a token
    alone can only reply), ntfy or Pushover whenever set."""
    return ((_telegram_configured() and bool(_telegram_whitelist()))
            or _ntfy_configured() or _pushover_configured())


def configured_backend_names() -> list[str]:
    """The backends a proactive ping would go out on, in send order."""
    names = []
    if _telegram_configured() and _telegram_whitelist():
        names.append("telegram")
    if _ntfy_configured():
        names.append("ntfy")
    if _pushover_configured():
        names.append("pushover")
    return names


# ─── Outbound senders ───────────────────────────────────────────────────

def _split_long(message: str, limit: int) -> list[str]:
    """Split on paragraph then word boundaries so messages above the
    backend's per-message cap arrive as 2+ ordered messages. Used for
    Telegram's 4096-char ceiling."""
    if len(message) <= limit:
        return [message]
    chunks: list[str] = []
    remaining = message
    while len(remaining) > limit:
        # Prefer the last newline before the cap; if none, fall back to
        # the last whitespace; if still none, hard-cut at the cap.
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _send_telegram(message: str, *, chat_id: int | None = None) -> bool:
    """Post `message` to Telegram via sendMessage. Returns True on
    success. If `chat_id` is None, sends to the first whitelisted id."""
    token = _telegram_token()
    if not token:
        return False
    if chat_id is None:
        wl = _telegram_whitelist()
        if not wl:
            _log.debug("[phone] telegram: no whitelist, cannot send unsolicited")
            return False
        chat_id = sorted(wl)[0]
    try:
        import requests  # type: ignore
    except Exception:
        _log.warning("[phone] requests not importable; cannot send Telegram")
        return False

    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    ok_all = True
    for chunk in _split_long(message, TELEGRAM_MAX_MSG_LEN):
        try:
            r = requests.post(url,
                              json={"chat_id": chat_id, "text": chunk,
                                    "disable_web_page_preview": True},
                              timeout=TELEGRAM_HTTP_TIMEOUT / 2)
            if r.status_code != 200:
                _log.warning("[phone] telegram sendMessage HTTP %s: %s",
                             r.status_code, r.text[:200])
                ok_all = False
                with _state_lock:
                    _status["last_error"] = f"telegram {r.status_code}"
        except Exception as e:
            _log.warning("[phone] telegram sendMessage failed: %s", e)
            with _state_lock:
                _status["last_error"] = f"telegram: {e}"
            ok_all = False
    return ok_all


def _send_ntfy(message: str, *, title: str = "", priority: str = "normal") -> bool:
    topic, host = _ntfy_config()
    if not topic:
        return False
    try:
        import requests  # type: ignore
    except Exception:
        return False
    url = f"{host}/{topic}"
    # ntfy uses HTTP headers for metadata; the body is the plain message.
    headers = {}
    if title:
        # ntfy headers are ASCII-only; transliterate non-ASCII rather than
        # bouncing the whole publish.
        headers["Title"] = title.encode("ascii", "replace").decode("ascii")
    ntfy_prio = _NTFY_PRIORITY_MAP.get(priority, "default")
    headers["Priority"] = ntfy_prio
    headers["Tags"]     = "robot"          # 🤖 emoji on the phone
    try:
        r = requests.post(url, data=message.encode("utf-8"),
                          headers=headers,
                          timeout=NTFY_HTTP_TIMEOUT)
        if r.status_code >= 400:
            _log.warning("[phone] ntfy HTTP %s: %s", r.status_code, r.text[:200])
            with _state_lock:
                _status["last_error"] = f"ntfy {r.status_code}"
            return False
        return True
    except Exception as e:
        _log.warning("[phone] ntfy publish failed: %s", e)
        with _state_lock:
            _status["last_error"] = f"ntfy: {e}"
        return False


def _send_pushover(message: str, *, title: str = "", priority: str = "normal") -> bool:
    tok, usr = _pushover_config()
    if not (tok and usr):
        return False
    try:
        import requests  # type: ignore
    except Exception:
        return False
    pushover_prio = _PUSHOVER_PRIORITY_MAP.get(priority, 0)
    params = {
        "token":    tok,
        "user":     usr,
        "message":  message,
        "priority": pushover_prio,
    }
    if title:
        params["title"] = title
    if pushover_prio == 2:
        # Pushover's "emergency" priority requires retry + expire knobs;
        # 30 s retry, 1 h expire is a reasonable urgent default.
        params["retry"]  = 30
        params["expire"] = 3600
    try:
        r = requests.post(PUSHOVER_URL, data=params,
                          timeout=PUSHOVER_HTTP_TIMEOUT)
        if r.status_code != 200:
            _log.warning("[phone] pushover HTTP %s: %s",
                         r.status_code, r.text[:200])
            with _state_lock:
                _status["last_error"] = f"pushover {r.status_code}"
            return False
        return True
    except Exception as e:
        _log.warning("[phone] pushover send failed: %s", e)
        with _state_lock:
            _status["last_error"] = f"pushover: {e}"
        return False


# ─── Public outbound API ────────────────────────────────────────────────

def push_to_phone(message: str,
                  *,
                  priority: str = "normal",
                  source: str = "skill",
                  title: str = "",
                  backends: list[str] | None = None,
                  confirm: bool = True,
                  recipient: str = "your phone") -> dict[str, bool] | None:
    """Broadcast a message to every configured phone backend.

    Returns a dict of {backend_name: success_bool}. Backends that aren't
    configured don't appear in the dict (vs. False, which means
    "configured but the send failed"). A denied / timed-out confirmation
    returns ``None`` — no backend was attempted, the message was dropped
    deliberately, which callers must NOT report as a send failure.

    `priority` is one of: low | normal | high | urgent.
      - ntfy maps it to its 5-level scale.
      - Pushover urgent triggers the retry/expire flow that re-pings
        until the user acknowledges.
      - Telegram has no priority concept (every message arrives equally);
        the field is accepted but ignored on that backend.

    `source` is just a tag used in the console fallback log so it's
    obvious which skill produced the push when no backend is configured.

    `title` is used only by ntfy + Pushover (Telegram has no title field
    on plain text messages).

    `backends` restricts the broadcast to a subset (e.g.
    ['ntfy', 'pushover']) when a skill wants a specific delivery
    pattern. None = use all configured.

    `confirm` (default True) routes the push through
    ``core.draft_confirm`` first — JARVIS reads the body aloud and waits
    for an explicit yes / no token before dispatching. Background alert
    sources (hardware faults, diagnostic pages) that need fire-and-forget
    delivery should pass ``confirm=False`` explicitly; user-driven pushes
    inherit the default so the standing read-aloud-before-sending rule
    applies. Fail-closed: if the gate can't reach a verdict (mic down,
    whisper unavailable, ambiguous reply, no confirmation in the window),
    the push is dropped rather than auto-sent.

    `recipient` is the name the gate uses in the spoken prompt — e.g.
    'I have a draft for <recipient>, sir: <body>. Shall I send it?'
    Defaults to "your phone" because most phone pushes are addressed to
    the user themselves; callers with a specific target (a partner, an
    on-call number) should override.
    """
    if not message or not message.strip():
        return {}

    if confirm:
        if draft_confirm is None:
            # Gate module failed to import — fail closed rather than
            # silently bypassing the confirmation. Caller can retry by
            # passing confirm=False once the underlying issue is fixed.
            print(f"  [phone/{source}] confirmation gate unavailable; "
                  f"dropping message: {message}")
            return None
        if not draft_confirm(message, recipient=recipient):
            print(f"  [phone/{source}] push denied / unconfirmed — "
                  f"dropped: {message}")
            return None

    title = title or "JARVIS"
    requested = set(backends or [])

    results: dict[str, bool] = {}

    def _want(name: str) -> bool:
        return not requested or name in requested

    if _want("telegram") and _telegram_configured():
        results["telegram"] = _send_telegram(message)

    if _want("ntfy") and _ntfy_configured():
        results["ntfy"] = _send_ntfy(message, title=title, priority=priority)

    if _want("pushover") and _pushover_configured():
        results["pushover"] = _send_pushover(message, title=title, priority=priority)

    if not results:
        # No backend fired — log to console so the message isn't lost.
        print(f"  [phone/{source}] (no backend configured) message: {message}")
    else:
        with _state_lock:
            _status["last_outbound_at"] = time.time()
            sent_count = sum(1 for v in results.values() if v)
            _status["messages_out"]   += sent_count
            _persisted["messages_out"] += sent_count
        _save_state()
    return results


# ─── Inbound: stateless LLM fallback ────────────────────────────────────

def _cloud_allowed() -> bool:
    """core.cloud_gate's answer (2026-10-01): may a phone message go to
    Claude? The fallback below used to try Claude whenever ANTHROPIC_API_KEY
    was set, so on a local-only install (AI_BACKEND=ollama / chat routed
    local) the owner's texts still went to the cloud. Unknown = no."""
    try:
        from core.cloud_gate import chat_cloud_allowed
        return chat_cloud_allowed()
    except Exception:
        return False


def _llm_fallback(text: str) -> str | None:
    """Stateless one-shot LLM call. Returns a reply string, or None when
    no LLM is configured / the call fails. Uses a fresh anthropic client
    with NO conversation_history — phone chat never leaks into the
    in-person session.

    Model is overridable via PHONE_BRIDGE_MODEL env var. Falls back to
    bobert_companion.CLAUDE_MODEL, then to _DEFAULT_LLM_MODEL so the
    user only rotates models in one place."""
    _claude_ok = bool(os.environ.get("ANTHROPIC_API_KEY")) and _cloud_allowed()
    if _claude_ok:
        try:
            import anthropic  # type: ignore
        except Exception:
            _claude_ok = False
    model = (os.environ.get("PHONE_BRIDGE_MODEL") or "").strip()
    if not model:
        try:
            bc = importlib.import_module("bobert_companion")
            model = getattr(bc, "CLAUDE_MODEL", "") or _DEFAULT_LLM_MODEL
        except Exception:
            model = _DEFAULT_LLM_MODEL

    sys_prompt = (
        "You are JARVIS, a concise British AI assistant replying to the user "
        "over their phone while they are away from their desk. Constraints:\n"
        "  • Keep replies tight — 1 to 4 short sentences, no bullet lists.\n"
        "  • Address the user as 'sir'.\n"
        "  • You do NOT have access to the user's PC actions in this channel — "
        "if they ask you to control the house, music, printer, etc., say so "
        "briefly and suggest the exact command verb they should send instead "
        "(e.g. 'send `play lo-fi` and I will route it through the dispatcher').\n"
        "  • For general questions, answer directly. For chitchat, respond in "
        "character but stay under three sentences."
    )
    if _claude_ok:
        try:
            from core import llm_client
            client = anthropic.Anthropic()
            # "voice": a reply the owner reads on his phone - effort low + a
            # max_tokens floor on the 5.x thinking models.
            msg = llm_client.create_message(
                client, purpose="voice",
                model=model,
                max_tokens=_LLM_MAX_TOKENS,
                system=sys_prompt,
                messages=[{"role": "user", "content": text}],
                timeout=_LLM_TIMEOUT_SECONDS,
            )
            out = llm_client.response_text(msg).strip()
            if out:
                return out
        except Exception as e:
            _log.warning("[phone] LLM fallback failed: %s", e)

    # Local-Ollama fallback (Claude capped / unavailable / no API key).
    try:
        bc = sys.modules.get("bobert_companion") or importlib.import_module("bobert_companion")
        local = bc._call_local_llm(sys_prompt, [{"role": "user", "content": text}], max_tokens=_LLM_MAX_TOKENS)
        if local:
            return local.strip() or None
    except Exception as e:
        _log.warning("[phone] local LLM fallback failed: %s", e)
    return None


# ─── Inbound: dispatch ──────────────────────────────────────────────────

# Reference to ACTIONS dict, captured at register() time. None until
# register has run.
_actions_ref: list[dict | None] = [None]


def _dispatch_remote(text: str) -> str:
    """Process an inbound text as if it were a typed JARVIS command.

    Resolution order:
      1. chain dispatch (multi-step)        — core.dispatcher.resolve_and_dispatch
      2. single intent match                — core.dispatcher.match_single_intent
      3. stateless LLM fallback             — fresh anthropic call

    Returns the reply text. Always returns a non-empty string so the
    Telegram reply path always has something to send back."""
    text = (text or "").strip()
    if not text:
        return "Empty message, sir."

    actions = _actions_ref[0]
    if actions is None:
        return ("I'm not fully booted yet, sir — phone bridge loaded before the "
                "action registry. Try again in a moment.")

    # Mode toggles (smart/agent/controlled mode) work via phone too.
    try:
        from core.mode_router import maybe_handle_mode_toggle
        toggle = maybe_handle_mode_toggle(text)
        if toggle is not None:
            return toggle
    except Exception:
        pass

    # Multi-step chain.
    try:
        from core.dispatcher import resolve_and_dispatch
        chain = resolve_and_dispatch(text, actions)
        if chain is not None:
            return chain
    except Exception as e:
        _log.warning("[phone] chain dispatch raised: %s", e)

    # Single intent.
    try:
        from core.dispatcher import match_single_intent
        step = match_single_intent(text, actions.keys())
    except Exception as e:
        _log.warning("[phone] single intent match raised: %s", e)
        step = None

    if step is not None:
        fn = actions.get(step.action)
        if fn is not None:
            try:
                rv = fn(step.arg)
            except Exception as e:
                return f"That action failed, sir ({type(e).__name__})."
            if isinstance(rv, str) and rv.strip():
                return rv
            return f"{step.confirmation.capitalize()}, sir."

    # LLM fallback — stateless, isolated history.
    llm_reply = _llm_fallback(text)
    if llm_reply:
        return llm_reply

    return ("I couldn't match that to a command, sir, and the LLM fallback is "
            "unavailable. Send /help for a hint.")


# ─── Inbound: slash-commands ────────────────────────────────────────────

def _help_text() -> str:
    backends_present = []
    if _telegram_configured():
        backends_present.append("telegram")
    if _ntfy_configured():
        backends_present.append("ntfy")
    if _pushover_configured():
        backends_present.append("pushover")
    return (
        "JARVIS phone bridge, sir.\n"
        "Send any direct command (e.g. 'play michael jackson', 'turn off the "
        "office lights', 'how many unread emails'). Multi-step chains work "
        "too: 'play lo-fi and start a 25 minute focus block'.\n\n"
        "Slash commands:\n"
        "  /help    — this message\n"
        "  /status  — backend health summary\n"
        "  /pause   — pause inbound polling (stops this bot listening, so "
        "resume from the desk: 'resume phone bridge')\n"
        "  /resume  — resume inbound polling (desk voice only — /pause "
        "stops me reading Telegram)\n"
        "  /mode    — show current conversation mode\n\n"
        "Backends configured: " + (", ".join(backends_present) or "(none)") + "."
    )


def _handle_slash(text: str) -> str:
    """Process a /-prefixed Telegram command. Returns the reply string."""
    cmd, _, rest = text.partition(" ")
    cmd = cmd.lower().lstrip("/")
    # Telegram appends @botname when the bot is in a group — strip it.
    if "@" in cmd:
        cmd = cmd.split("@", 1)[0]

    if cmd in ("start", "help", "?"):
        return _help_text()

    if cmd == "status":
        return phone_bridge_status("")

    if cmd == "pause":
        # Pausing stops the long-poll worker itself, so a later /resume
        # can never arrive over Telegram — be explicit about that here
        # rather than letting the user discover the bridge is deaf.
        return (pause_phone_bridge("") + " Note: I've stopped reading "
                "Telegram, so /resume won't reach me — say 'resume phone "
                "bridge' at the desk to re-enable.")

    if cmd == "resume":
        return resume_phone_bridge("")

    if cmd == "mode":
        try:
            from core.mode_router import current_mode
            return f"Current conversation mode: {current_mode()}, sir."
        except Exception:
            return "Mode router not loaded, sir."

    # Unrecognised slash command — fall through to the normal dispatcher.
    full = (cmd + " " + rest).strip()
    return _dispatch_remote(full)


# ─── Inbound: Telegram long-poll worker ─────────────────────────────────

def _telegram_polling_loop() -> None:
    """Long-poll Telegram's getUpdates endpoint. Re-entrant — exits
    cleanly if pause_phone_bridge is called, restarts on resume."""
    time.sleep(TELEGRAM_INIT_DELAY)

    token = _telegram_token()
    if not token:
        # Token absent — nothing to do.
        return
    whitelist = _telegram_whitelist()
    if not whitelist:
        # FAIL CLOSED (2026-07-14 bug-hunt). An empty whitelist means "no inbound
        # allowed" (see _telegram_whitelist's docstring) — so we must NOT poll.
        # The old code only warned and fell through to the polling loop, and the
        # per-message guard below (`if whitelist and ...`) short-circuits to
        # False on an empty set, so EVERY message from ANY Telegram user was
        # dispatched to _dispatch_remote — arbitrary JARVIS actions (lights,
        # email, the LLM) with no auth. The whitelist is the ONLY inbound
        # boundary; refuse to start until it's set.
        _log.warning("[phone] TELEGRAM_BOT_TOKEN set but TELEGRAM_USER_ID is "
                     "empty — inbound STAYS OFF until you whitelist a chat.")
        with _state_lock:
            _status["last_error"] = "no TELEGRAM_USER_ID whitelist"
        return

    try:
        import requests  # type: ignore
    except Exception:
        _log.warning("[phone] requests unavailable; inbound disabled")
        with _state_lock:
            _status["last_error"] = "requests not installed"
        return

    with _state_lock:
        _status["polling"]    = True
        _status["started_at"] = time.time()

    while not _pause_flag[0]:
        offset = _persisted["last_update_id"] + 1
        url = f"{TELEGRAM_API_BASE}/bot{token}/getUpdates"
        try:
            r = requests.get(url,
                             params={"offset":  offset,
                                     "timeout": TELEGRAM_LONGPOLL_SEC,
                                     "allowed_updates": json.dumps(["message"])},
                             timeout=TELEGRAM_HTTP_TIMEOUT)
            if r.status_code != 200:
                _log.warning("[phone] getUpdates HTTP %s: %s",
                             r.status_code, r.text[:200])
                with _state_lock:
                    _status["last_error"] = f"getUpdates {r.status_code}"
                time.sleep(TELEGRAM_RETRY_SLEEP)
                continue
            payload = r.json()
        except Exception as e:
            _log.warning("[phone] getUpdates exception: %s", e)
            with _state_lock:
                _status["last_error"] = f"getUpdates: {e}"
            time.sleep(TELEGRAM_RETRY_SLEEP)
            continue

        if not payload.get("ok"):
            _log.warning("[phone] getUpdates not ok: %s",
                         payload.get("description", ""))
            with _state_lock:
                _status["last_error"] = payload.get("description", "telegram error")
            time.sleep(TELEGRAM_RETRY_SLEEP)
            continue

        for upd in payload.get("result", []) or []:
            try:
                _process_update(upd, whitelist)
            except Exception as e:
                _log.warning("[phone] update %s handler crash: %s",
                             upd.get("update_id"), e)
                _log.debug(traceback.format_exc())
            # Persist offset incrementally so a crash mid-batch doesn't
            # cause us to re-process the same updates on restart.
            uid = upd.get("update_id")
            if isinstance(uid, int) and uid > _persisted["last_update_id"]:
                _persisted["last_update_id"] = uid
                _save_state()

    with _state_lock:
        _status["polling"] = False


def _process_update(upd: dict, whitelist: set[int]) -> None:
    """Handle a single Telegram update object. Whitelist enforced here
    so a leaked bot token can't be exploited just by knowing the
    username — the attacker also needs to be one of the whitelisted
    chat ids."""
    msg = upd.get("message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    from_user = msg.get("from") or {}
    user_id = from_user.get("id")
    text = (msg.get("text") or "").strip()
    if not text:
        return
    if not isinstance(chat_id, int):
        return
    # Reject when the whitelist is EMPTY too (2026-07-14 bug-hunt): `if whitelist
    # and ...` let an empty set short-circuit to False and admit everyone. An
    # empty whitelist is "deny all", never "allow all".
    if not whitelist or (user_id not in whitelist and chat_id not in whitelist):
        _log.warning("[phone] rejected message from unauthorised user_id=%s "
                     "chat_id=%s", user_id, chat_id)
        try:
            _send_telegram(
                "Unauthorised, sir. This bot only responds to whitelisted "
                "chat ids.", chat_id=chat_id,
            )
        except Exception:
            pass
        return

    with _state_lock:
        _status["last_inbound_at"] = time.time()
        _status["messages_in"]    += 1
        _persisted["messages_in"] += 1

    print(f"  [phone] inbound from {user_id}: {text[:120]}")

    reply: str
    if text.startswith("/"):
        try:
            reply = _handle_slash(text)
        except Exception as e:
            reply = f"Slash handler crashed, sir: {type(e).__name__}."
            _log.debug(traceback.format_exc())
    else:
        try:
            reply = _dispatch_remote(text)
        except Exception as e:
            reply = f"Dispatch crashed, sir: {type(e).__name__}."
            _log.debug(traceback.format_exc())

    if reply:
        _send_telegram(reply, chat_id=chat_id)


# ─── Thread launcher ────────────────────────────────────────────────────

def _start_polling_thread() -> None:
    """Spawn the long-poll worker if a Telegram token is set. Idempotent
    — re-calls when polling is already alive are no-ops."""
    if not _telegram_configured():
        return
    t = _polling_thread[0]
    if t is not None and t.is_alive():
        return
    th = threading.Thread(target=_telegram_polling_loop,
                          name="phone-bridge-poller",
                          daemon=True)
    th.start()
    _polling_thread[0] = th


# ─── Action handlers ────────────────────────────────────────────────────

def _format_priority_arg(arg: str) -> tuple[str, str]:
    """Strip a leading 'priority=high' / '!high' / '!urgent' shortcut so
    user / skill code can pre-tag a message without packaging it into a
    dict. Returns (message, priority)."""
    arg = arg.strip()
    priority = "normal"
    if arg.startswith("!"):
        rest = arg[1:]
        head, _, tail = rest.partition(" ")
        head_l = head.strip().lower()
        if head_l in _NTFY_PRIORITY_MAP:
            priority = head_l
            arg = tail.strip()
    return arg, priority


def notify_phone(arg: str) -> str:
    """Voice action: send a text message to every configured phone backend.

    Arg shapes:
      'message body here'                — sends at normal priority
      '!urgent message body here'        — leading !<priority> shorthand
      '!high build failed, sir'

    Returns a per-backend success summary in JARVIS voice."""
    message, priority = _format_priority_arg(arg)
    if not message:
        return ("Pass the message to send, sir — e.g. 'notify phone print "
                "finished' or '!urgent build is red'.")
    if not any_backend_configured():
        return ("No phone backends configured, sir. Set TELEGRAM_BOT_TOKEN + "
                "TELEGRAM_USER_ID, or NTFY_TOPIC, or PUSHOVER_TOKEN + "
                "PUSHOVER_USER, then reload.")
    results = push_to_phone(message, priority=priority, source="voice")
    if results is None:
        # The read-back was denied (or never confirmed) — a deliberate
        # drop, not a backend failure, so don't say "failed" here.
        return "Understood, sir — I've dropped that message unsent."
    sent = [b for b, ok in results.items() if ok]
    failed = [b for b, ok in results.items() if not ok]
    if sent and not failed:
        return f"Message sent to {' and '.join(sent)}, sir."
    if sent and failed:
        return (f"Sent via {' and '.join(sent)} but {' and '.join(failed)} "
                f"failed, sir — check phone bridge status.")
    return ("Send failed on every configured backend, sir — check phone "
            "bridge status for the last error.")


def phone_bridge_status(_: str = "") -> str:
    bits: list[str] = []

    if _telegram_configured():
        wl = _telegram_whitelist()
        if wl:
            bits.append(f"telegram (whitelist: {len(wl)})")
        else:
            bits.append("telegram (no whitelist)")
    else:
        bits.append("telegram not configured")

    bits.append("ntfy: " + ("yes" if _ntfy_configured() else "no"))
    bits.append("pushover: " + ("yes" if _pushover_configured() else "no"))

    with _state_lock:
        bits.append("polling" if _status["polling"] else "idle")
        bits.append(f"in: {_status['messages_in']}")
        bits.append(f"out: {_status['messages_out']}")
        if _status["last_inbound_at"]:
            ago = max(0, int(time.time() - _status["last_inbound_at"]))
            bits.append(f"last inbound {ago}s ago")
        if _status["last_outbound_at"]:
            ago = max(0, int(time.time() - _status["last_outbound_at"]))
            bits.append(f"last outbound {ago}s ago")
        if _pause_flag[0]:
            bits.append("paused")
        if _status["last_error"]:
            bits.append(f"last error: {_status['last_error'][:80]}")
    return "Phone bridge — " + "; ".join(bits) + ", sir."


def list_phone_backends(_: str = "") -> str:
    lines = []
    if _telegram_configured():
        wl = sorted(_telegram_whitelist())
        wl_str = ", ".join(str(x) for x in wl) if wl else "(none — no inbound)"
        lines.append(f"telegram: configured, whitelist=[{wl_str}]")
    else:
        lines.append("telegram: NOT configured (set TELEGRAM_BOT_TOKEN + "
                     "TELEGRAM_USER_ID)")
    if _ntfy_configured():
        topic, host = _ntfy_config()
        lines.append(f"ntfy: configured, topic={topic}, host={host}")
    else:
        lines.append("ntfy: NOT configured (set NTFY_TOPIC)")
    if _pushover_configured():
        lines.append("pushover: configured")
    else:
        lines.append("pushover: NOT configured (set PUSHOVER_TOKEN + "
                     "PUSHOVER_USER)")
    return "Phone backends, sir:\n" + "\n".join(lines)


def pause_phone_bridge(_: str = "") -> str:
    _pause_flag[0] = True
    return "Phone bridge polling paused, sir."


def resume_phone_bridge(_: str = "") -> str:
    _pause_flag[0] = False
    _start_polling_thread()
    return "Phone bridge polling resumed, sir."


# ─── Proactive pings (core/phone_ping.py) ───────────────────────────────

def _bridge_send(text: str, *, priority: str = "normal", title: str = "JARVIS",
                 category: str = "ping") -> dict[str, bool] | None:
    """core.phone_ping's sender. Fire-and-forget (confirm=False): the policy
    layer composed the text itself, and a read-back would wait on a mic the
    owner is, by definition, away from."""
    return push_to_phone(text, priority=priority, source=f"ping:{category}",
                         title=title or "JARVIS", confirm=False)


def ping_phone(category: str, message: str, **kwargs) -> str:
    """Public: ask core.phone_ping to text the owner if — and only if — the
    category, his presence, quiet hours, focus mode and the hourly cap allow
    it. Returns the outcome ("queued" when it went out to the sender)."""
    if _phone_ping is None:
        return "unavailable"
    return _phone_ping.ping(category, message, **kwargs)


def _cfg(name: str, default=None):
    """A live core.config value (the Settings window and the voice toggle
    change it at run time)."""
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _persist_setting(key: str, value) -> bool:
    """The Settings writer every voice toggle uses."""
    try:
        from tools import settings_window as sw
    except Exception:
        return False
    try:
        current = sw.load_settings()
        if not isinstance(current, dict):
            current = {}
        current[key] = value
        sw.save_settings(current, changed=(key,))
        return True
    except Exception:
        return False


def _set_pings_enabled(on: bool) -> bool:
    """Flip PHONE_PING_ENABLED live and save it. False when the save failed
    (the flip then lasts until the next restart)."""
    try:
        import core.config as _c
        _c.PHONE_PING_ENABLED = bool(on)
    except Exception:
        pass
    return _persist_setting("PHONE_PING_ENABLED", bool(on))


_SETUP_STEPS = (
    "Here's how to connect your phone, sir. One: in Telegram, message "
    "@BotFather, send /newbot, and choose a name and a username ending in "
    "'bot'. BotFather replies with a token. Two: put that token in the .env "
    "file in my folder as TELEGRAM_BOT_TOKEN. Three: message @userinfobot to "
    "get your numeric user ID, and add it as TELEGRAM_USER_ID. Four: open a "
    "chat with your new bot and press Start, so it's allowed to message you. "
    "Five: restart me, then say 'send a test ping'. If you'd rather not use "
    "Telegram, an ntfy topic in NTFY_TOPIC works for one-way pings."
)

_SETUP_NEEDS_USER_ID = (
    "You're nearly there, sir: the bot token is set, but I need your Telegram "
    "user ID before I can message you. Message @userinfobot on Telegram, put "
    "the number it gives you in the .env file in my folder as "
    "TELEGRAM_USER_ID, open a chat with your bot and press Start, then "
    "restart me and say 'send a test ping'."
)

_TEST_TEXT = "Test ping from JARVIS, sir. Your phone is connected."

_PING_CATEGORY_WORDS = (
    ("print", "prints"), ("confirm", "unanswered confirmations"),
    ("security", "guard alerts"), ("robot", "the robot"),
)


def phone_setup_help(_: str = "") -> str:
    """'How do I connect my phone' — the BotFather steps (or, when a backend
    is already set, how to check it)."""
    names = configured_backend_names()
    if names:
        return (f"Your phone is already connected through "
                f"{' and '.join(names)}, sir. Say 'send a test ping' to check "
                f"it, or 'phone ping status' to see what I'll text you about.")
    if _telegram_configured() and not _telegram_whitelist():
        return _SETUP_NEEDS_USER_ID
    return _SETUP_STEPS


def _join_words(words: list[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return "just now"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes} minutes ago"
    hours = minutes // 60
    return f"{hours} hour{'s' if hours != 1 else ''} ago"


def phone_ping_status(_: str = "") -> str:
    """Are proactive pings on, through what, about what, within what limits."""
    if _phone_ping is None:
        return "Phone pings aren't available in this build, sir."
    st = _phone_ping.status()
    if not st.get("enabled"):
        msg = ("Phone pings are off, sir. Say 'turn on phone pings' and I'll "
               "text you when something needs you.")
        if st.get("configured"):
            msg += _guard_line()
        return msg
    if not st.get("configured"):
        return ("Phone pings are on, sir, but your phone isn't connected yet, "
                "so nothing can go out. Ask me 'how do I connect my phone'.")
    cats = st.get("categories") or {}
    kinds = [label for cat, label in _PING_CATEGORY_WORDS if cats.get(cat)]
    via = _join_words(configured_backend_names()) or "the phone bridge"
    parts = [f"Phone pings are on, sir, through {via}: "
             + (_join_words(kinds) + "." if kinds
                else "every kind is switched off in Settings.")]
    if cats.get("summary"):
        parts.append(f"The daily summary goes out at {st.get('summary_time')}.")
    else:
        parts.append("The daily summary is off.")
    parts.append(f"Quiet hours run {st.get('quiet_start')} to "
                 f"{st.get('quiet_end')}, except guard alerts; at most "
                 f"{st.get('max_per_hour')} an hour.")
    sent = int(st.get("sent_last_hour") or 0) + int(
        st.get("critical_last_hour") or 0)
    line = f"{sent} sent in the last hour."
    last = st.get("last")
    if isinstance(last, dict) and last.get("t"):
        word = dict(_PING_CATEGORY_WORDS).get(last.get("cat"), last.get("cat"))
        line += f" The last one went out {_ago(time.time() - last['t'])}, " \
                f"about {word}."
    parts.append(line)
    held = int(st.get("held") or 0)
    if held:
        parts.append(f"{held} {'is' if held == 1 else 'are'} waiting for the "
                     f"end of quiet hours.")
    if st.get("blocked"):
        parts.append("This is a staging instance, so none go out from here.")
    return " ".join(parts)


def phone_pings_on(_: str = "") -> str:
    """Master switch on (live + saved) and start the watcher."""
    was_on = bool(_cfg("PHONE_PING_ENABLED", True))
    persisted = _set_pings_enabled(True)
    if not can_push_unsolicited():
        msg = ("Phone pings are on, sir, but your phone isn't connected yet. "
               "Ask me 'how do I connect my phone'.")
    else:
        if _phone_ping is not None:
            _phone_ping.start_watcher()
        msg = ("Phone pings are already on, sir." if was_on else
               "Phone pings on, sir. I'll text you when something needs you.")
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


def _guard_line() -> str:
    """What the master switch does NOT cover: guard-mode alerts answer to
    PHONE_PING_SECURITY alone (core/phone_ping._gate)."""
    if bool(_cfg("PHONE_PING_SECURITY", True)):
        return (" Guard-mode alerts still reach your phone; switch those off "
                "under Guard-mode alerts in Settings.")
    return " Guard-mode alerts are switched off in Settings too."


def phone_pings_off(_: str = "") -> str:
    """Master switch off (live + saved): no print, confirmation, robot or
    summary pings. Guard alerts keep their own switch; notify_phone still
    works."""
    was_on = bool(_cfg("PHONE_PING_ENABLED", True))
    persisted = _set_pings_enabled(False)
    if _phone_ping is not None:
        _phone_ping.stop_watcher()
    msg = ("Phone pings off, sir. I won't text you about prints or anything "
           "else unprompted; 'text my phone' still works." if was_on
           else "Phone pings are already off, sir.")
    msg += _guard_line()
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


def phone_ping_test(_: str = "") -> str:
    """Send one fixed test ping right now (it skips the quiet-hours / presence
    gates: he asked for it). Never from a staging instance."""
    if not can_push_unsolicited():
        return ("Your phone isn't connected yet, sir, so there's nowhere to "
                "send it. Ask me 'how do I connect my phone'.")
    blocked = ""
    if _phone_ping is not None:
        try:
            blocked = _phone_ping.get_pinger().blocked() or ""
        except Exception:
            blocked = ""
    if blocked:
        return "Not from the staging instance, sir."
    results = push_to_phone(_TEST_TEXT, priority="normal", source="ping:test",
                            title="JARVIS", confirm=False) or {}
    sent = [b for b, ok in results.items() if ok]
    failed = [b for b, ok in results.items() if not ok]
    if sent and not failed:
        return f"Test ping sent through {' and '.join(sent)}, sir."
    if sent:
        return (f"Test ping sent through {' and '.join(sent)}, but "
                f"{' and '.join(failed)} failed, sir.")
    return ("The test ping failed on every backend, sir. Say 'phone status' "
            "for the last error.")


# ── utterance route: the exact phrasings, before the LLM ─────────────────
_PHONE_NOUN = r"(?:phone|cell ?phone|mobile(?: phone)?|iphone|android(?: phone)?)"
_RLEAD = r"^(?:(?:hey |ok |okay )?jarvis )?(?:please )?(?:can you |could you )?"
_RTAIL = r"(?: (?:please|sir|jarvis|now))*$"
_PINGS = r"(?:the )?phone pings?"
_PHONE_ROUTES = (
    (re.compile(_RLEAD + r"(?:"
                r"(?:how (?:do|can|should|would) i|how to|help me|show me how "
                r"to|walk me through(?: how to)?|tell me how to|explain how "
                r"to|what do i (?:need to )?do to) (?:connect|link|hook up|"
                r"pair|set up|setup|add) (?:my|your|the|a) " + _PHONE_NOUN
                + r"(?: (?:to|with|up to|up with) (?:you|jarvis))?"
                r"|(?:connect|link|hook up|pair) (?:my|the) " + _PHONE_NOUN
                + r" (?:to|with) (?:you|jarvis)"
                r"|(?:how (?:do|can) i |help me |show me how to )?(?:set up|"
                r"setup|configure|connect) (?:the |my )?(?:phone bridge|phone "
                r"pings?|telegram(?: bot)?|phone notifications)"
                r")" + _RTAIL), "[ACTION: phone_setup_help]"),
    (re.compile(_RLEAD + r"(?:(?:turn|switch) off " + _PINGS + r"|(?:turn|"
                r"switch) " + _PINGS + r" off|(?:disable|stop|pause|mute) "
                + _PINGS + r"|phone pings? off|stop pinging (?:me|my phone)|"
                r"(?:don't|do not) ping (?:me|my phone))" + _RTAIL),
     "[ACTION: phone_pings_off]"),
    (re.compile(_RLEAD + r"(?:(?:turn|switch) on " + _PINGS + r"|(?:turn|"
                r"switch) " + _PINGS + r" on|(?:enable|resume|start) "
                + _PINGS + r"|phone pings? on)" + _RTAIL),
     "[ACTION: phone_pings_on]"),
    (re.compile(_RLEAD + r"(?:(?:what is |what are )?(?:the |my )?phone "
                r"pings? (?:status|settings)|(?:are|is) " + _PINGS
                + r" (?:on|off|enabled|working)|what will you ping me about|"
                r"when (?:will|do) you ping (?:me|my phone))" + _RTAIL),
     "[ACTION: phone_ping_status]"),
    (re.compile(_RLEAD + r"(?:(?:send|give) (?:me )?(?:a )?test (?:ping|"
                r"message|notification)(?: to (?:my|the) phone)?|test (?:my "
                r"|the )?phone pings?|(?:send|do) (?:a )?phone ping test|test "
                r"(?:the )?phone bridge)" + _RTAIL), "[ACTION: phone_ping_test]"),
)


def _phone_route(text):
    """Utterance route: the phone-ping token for an exact setup / on / off /
    status / test request, else None. Never raises."""
    try:
        s = str(text or "").lower().replace("-", " ")
        s = s.replace("what's", "what is").replace("whats", "what is")
        s = " ".join(re.sub(r"[^a-z0-9' ]+", " ", s).split())
        if not any(w in s for w in ("phone", "ping", "telegram", "mobile")):
            return None
        for rx, token in _PHONE_ROUTES:
            if rx.match(s):
                return token
        return None
    except Exception:
        return None


def _attach_pings() -> bool:
    """Hand this bridge to core.phone_ping and start its watcher. With no
    backend that can send unsolicited, ONE line says pings are a no-op."""
    if _phone_ping is None:
        return False
    p = _phone_ping.attach_bridge(_bridge_send, can_push_unsolicited)
    if not can_push_unsolicited():
        p.log_unconfigured_once()
        return False
    if not bool(_cfg("PHONE_PING_ENABLED", True)):
        print("  [phone-ping] off (PHONE_PING_ENABLED) — say 'turn on phone "
              "pings' to enable")
        return False
    if _phone_ping.start_watcher():
        print("  [phone-ping] on — pings go out through "
              f"{', '.join(configured_backend_names())}")
        return True
    return False


# ─── Registration ───────────────────────────────────────────────────────

def register(actions):
    _actions_ref[0] = actions
    _load_state()

    actions["notify_phone"]         = notify_phone
    actions["push_to_phone"]        = notify_phone
    actions["text_my_phone"]        = notify_phone
    actions["phone_bridge_status"]  = phone_bridge_status
    actions["phone_status"]         = phone_bridge_status
    actions["list_phone_backends"]  = list_phone_backends
    actions["pause_phone_bridge"]   = pause_phone_bridge
    actions["resume_phone_bridge"]  = resume_phone_bridge
    actions["phone_setup_help"]     = phone_setup_help
    actions["phone_ping_status"]    = phone_ping_status
    actions["phone_pings_on"]       = phone_pings_on
    actions["phone_pings_off"]      = phone_pings_off
    actions["phone_ping_test"]      = phone_ping_test

    try:
        su = globals().get("skill_utils") or {}
        reg = su.get("register_utterance_route") if isinstance(su, dict) else None
        if callable(reg):
            reg(_phone_route, "phone pings")
    except Exception:
        pass

    # Kick off the inbound long-poll only when a Telegram token is set.
    _start_polling_thread()
    _attach_pings()

    configured = []
    if _telegram_configured():
        configured.append("telegram")
    if _ntfy_configured():
        configured.append("ntfy")
    if _pushover_configured():
        configured.append("pushover")
    if configured:
        print(f"  [phone] phone_bridge ready — backends: {', '.join(configured)}")
    else:
        print("  [phone] phone_bridge loaded — no backends configured "
              "(set TELEGRAM_BOT_TOKEN + TELEGRAM_USER_ID, NTFY_TOPIC, or "
              "PUSHOVER_TOKEN + PUSHOVER_USER)")
