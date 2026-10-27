"""Background brain on a second Ollama (v3 W13).

WHY
===
Ollama's one slot on the 3090 holds the voice brain's cached ~12k-token
prompt. Every background job (learn_batch, session_checkpoint, ltm_reflect
...) that POSTs to the SAME server replaces that cache; the owner's next turn
then pays a partial or full prefill (p50 1.76 s warm vs 3.84 s partial vs
6.26 s full; 18 % of turns paid). A CPU-only Ollama on the LAN (Unraid) can
take those jobs so the 3090's slot is left alone.

WHAT THIS MODULE IS
===================
Pure routing + one HTTP helper. The monolith's ``_call_local_llm`` asks
``route()``; when it says "remote" it calls ``chat()``; on any failure it logs
and falls back to the main brain, so no learning is lost.

* ``BACKGROUND_LLM_BASE_URL`` "" (the default) -> ``route()`` says "main" and
  nothing else in this module runs: today's behaviour, byte for byte.
* ONLY callers on ``REMOTE_OK_CALLERS`` ever go remote, and only when the
  call is background. That is an allowlist on purpose: a caller whose output
  is SPOKEN right away (``OWNER_FACING_CALLERS`` documents the known ones) must
  never wait on a CPU box, and a new caller defaults to the main brain until
  somebody decides it can wait.
* The remote host is CPU-only and serves ONE pre-configured model (on the
  owner's Unraid box: ``gemma4-e4b-cpu6``, gemma4:e4b with num_thread 6 baked
  into its Modelfile, context 8192 server-side). The payload therefore carries
  NO ``num_ctx``, ``num_thread``, GPU option or ``keep_alive``: Ollama keys a
  runner on its options, so any per-request difference would reload 6.6 GB on
  a CPU host, and the server's own 24 h keep-alive must win. Prompts are
  budgeted against ``BACKGROUND_LLM_NUM_CTX`` (8192) before sending; one that
  would not fit stays on the main brain.
* Loopback URLs use 127.0.0.1, never ``localhost`` (Windows resolves ::1
  first: a 2 s tax per request).
"""
from __future__ import annotations

import ipaddress
import threading
import time
from urllib.parse import urlsplit

try:
    import requests as _requests
except Exception:  # pragma: no cover - requests is a hard dependency elsewhere
    _requests = None

DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_TIMEOUT_S = 120.0
# The remote server's context window (set server-side, never sent).
DEFAULT_NUM_CTX = 8192
# After a failure, skip the remote host for this long so a dead server costs
# one timeout per window, not one per job.
HEALTH_MEMO_S = 60.0
# A CPU box that is only reached for background jobs: a short connect budget.
CONNECT_TIMEOUT_S = 5.0
# Sampling knobs only: none of these key an Ollama runner, so they never
# force a reload on the remote host.
SAMPLING = {"temperature": 0.4, "top_p": 0.9, "repeat_penalty": 1.05,
            "top_k": 40}

# Background callers whose result is NOT spoken and is not latency-bound.
# Census of every background=True caller that reaches _call_local_llm
# (2026-10-27, v2.0.231):
REMOTE_OK_CALLERS = frozenset({
    "learn_from_turn",       # fact extraction after a turn
    "learn_batch",           # batched fact extraction
    "ambient_media_judge",   # is the ambient audio media? (4-token judge)
    "session_checkpoint",    # running conversation summary
    "ltm_reflect",           # long-term-memory reflection
    "ambient_extractor",     # ambient window-text fact extraction
})
# Background-tagged but OWNER-FACING (spoken or shown to him right away): never
# remote. Listed so the census is explicit and a test can pin it.
OWNER_FACING_CALLERS = frozenset({
    "proactive_remark",         # spoken unprompted
    "screen_remark",            # spoken unprompted
    "boot_changelog_summary",   # spoken at boot
    "chappie",                  # spoken thoughts
    "briefing_headline_summary",  # part of the spoken briefing
    "notification_triage",      # decides what is announced now
})


class RemoteFailure(Exception):
    """The remote host did not give a usable reply (connection, 5xx, timeout,
    empty body). The caller falls back to the main brain."""


class HealthMemo:
    """Remembers a recent failure for ``ttl`` seconds. Thread-safe."""

    def __init__(self, ttl: float = HEALTH_MEMO_S, clock=time.monotonic):
        self.ttl = float(ttl)
        self._clock = clock
        self._lock = threading.Lock()
        self._down_until = 0.0

    def is_down(self) -> bool:
        with self._lock:
            return self._clock() < self._down_until

    def note_failure(self) -> None:
        with self._lock:
            self._down_until = self._clock() + self.ttl

    def note_ok(self) -> None:
        with self._lock:
            self._down_until = 0.0


HEALTH = HealthMemo()

# Calls made inside ``stay_on_main()`` on this thread never go remote. Used
# where the OWNER waits on a background-tagged job: the session summary taken
# at shutdown and at a peer hand-off (the periodic checkpoint still goes
# remote). Thread-local, because _llm_quick runs the call on the caller's
# own thread.
_PIN = threading.local()


class stay_on_main:
    """Context manager: background calls on this thread stay on the main
    brain while it is active. Re-entrant."""

    def __enter__(self):
        _PIN.depth = getattr(_PIN, "depth", 0) + 1
        return self

    def __exit__(self, *_exc):
        _PIN.depth = max(0, getattr(_PIN, "depth", 1) - 1)
        return False


def pinned_to_main() -> bool:
    return getattr(_PIN, "depth", 0) > 0


def clean_base_url(value) -> str:
    """The configured URL without trailing slashes; "" when unset/not text."""
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


# The Tailscale CGNAT range 100.64/10, built from its integer form.
_TAILNET = ipaddress.ip_network((0x64400000, 10))


def url_allowed(base_url) -> bool:
    """True when the background host may receive memory / conversation /
    room text (security review NET-3, 2026-10-27): https to any host, or
    plain http ONLY to loopback, a Tailscale 100.64/10 address or a
    *.ts.net name (WireGuard-encrypted). A LAN or foreign host over plain
    http is refused, so a sniffing or spoofing device on the LAN (or a
    stranger's .52 on a foreign Deco network) never gets that text."""
    url = clean_base_url(base_url)
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    if not host:
        return False
    if parts.scheme == "https":
        return True
    if parts.scheme != "http":
        return False
    if host == "localhost" or host.endswith(".ts.net"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or (ip.version == 4 and ip in _TAILNET)


def clean_model(value) -> str:
    s = value.strip() if isinstance(value, str) else ""
    return s or DEFAULT_MODEL


def clean_timeout(value) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S
    if t != t or t <= 0 or t == float("inf"):
        return DEFAULT_TIMEOUT_S
    return t


def clean_num_ctx(value) -> int:
    """The remote window used for budgeting; DEFAULT_NUM_CTX when unset/bad."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_NUM_CTX
    return n if n >= 1024 else DEFAULT_NUM_CTX


def route(base_url, caller: str, background: bool, health: HealthMemo = None) -> str:
    """"remote" or "main" for one call. "remote" needs ALL of: a configured
    URL, a background call, an allowlisted (not owner-facing) caller, and a
    remote host not memoised as down."""
    if not clean_base_url(base_url) or not url_allowed(base_url):
        return "main"
    if not background or pinned_to_main():
        return "main"
    name = str(caller or "")
    if name in OWNER_FACING_CALLERS or name not in REMOTE_OK_CALLERS:
        return "main"
    if (HEALTH if health is None else health).is_down():
        return "main"
    return "remote"


def build_payload(model: str, system: str, messages: list, max_tokens: int,
                  think=None) -> dict:
    """The /api/chat body for the remote CPU host: the main brain's sampling
    knobs and num_predict ONLY. No num_ctx / num_thread / GPU options and no
    keep_alive (see the module docstring: any of them would reload the model
    or override the server's residency)."""
    options = dict(SAMPLING)
    options["num_predict"] = max_tokens
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + list(messages),
        "stream": False,
        "options": options,
    }
    if think is not None:
        payload["think"] = think
    return payload


_SESSION = []
_SESSION_LOCK = threading.Lock()


def _session():
    """One shared requests.Session with trust_env off: no HTTP(S)_PROXY,
    Windows system proxy or .netrc ever carries the payload (NET-5)."""
    if _requests is None:
        return None
    with _SESSION_LOCK:
        if not _SESSION or _SESSION[0][0] is not _requests:
            sess = _requests.Session()
            sess.trust_env = False
            _SESSION[:] = [(_requests, sess)]
        return _SESSION[0][1]


def chat(base_url: str, model: str, system: str, messages: list,
         max_tokens: int, timeout_s: float, think=None, post=None):
    """POST one /api/chat to the remote host. Returns (text, body_dict).
    Raises RemoteFailure on a connection error, timeout, non-2xx or empty
    reply; the health memo is updated either way."""
    if not url_allowed(base_url):
        raise RemoteFailure("background host refused: plain http off the tailnet")
    if post is not None:
        poster = post
    else:
        sess = _session()
        poster = sess.post if sess is not None else None
    if poster is None:
        raise RemoteFailure("requests unavailable")
    payload = build_payload(model, system, messages, max_tokens, think=think)
    url = clean_base_url(base_url) + "/api/chat"
    try:
        # Never follow a redirect: a 307/308 would re-POST the whole body to
        # wherever Location says (NET-5).
        r = poster(url, json=payload,
                   timeout=(CONNECT_TIMEOUT_S, clean_timeout(timeout_s)),
                   allow_redirects=False)
    except Exception as e:
        HEALTH.note_failure()
        raise RemoteFailure(f"{type(e).__name__}: {e}") from e
    status = getattr(r, "status_code", 200)
    # Any non-2xx (a 3xx, a 404 for a missing model, a 400) arms the memo, so
    # the next jobs go to the main brain instead of re-sending the full
    # payload (NET-5). requests counts a 3xx as .ok, so the status decides.
    if not (200 <= status < 300) or not getattr(r, "ok", True):
        HEALTH.note_failure()
        raise RemoteFailure(f"HTTP {status}")
    try:
        body = r.json()
    except Exception as e:
        HEALTH.note_failure()
        raise RemoteFailure(f"bad JSON: {e}") from e
    text = ((body.get("message") or {}).get("content") or "").strip() \
        if isinstance(body, dict) else ""
    if not text:
        raise RemoteFailure("empty reply")
    HEALTH.note_ok()
    return text, body
