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
* The remote host is CPU-only: the payload carries no ``main_gpu``/GPU
  options. ``num_ctx`` is still pinned through ``core.ollama_opts`` (Ollama
  keys a runner on its options; a tree-wide test enforces the pin).
* Loopback URLs use 127.0.0.1, never ``localhost`` (Windows resolves ::1
  first: a 2 s tax per request).
"""
from __future__ import annotations

import threading
import time

try:
    import requests as _requests
except Exception:  # pragma: no cover - requests is a hard dependency elsewhere
    _requests = None

from core import ollama_opts

DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_TIMEOUT_S = 120.0
# After a failure, skip the remote host for this long so a dead server costs
# one timeout per window, not one per job.
HEALTH_MEMO_S = 60.0
# A CPU box that is only reached for background jobs: a short connect budget.
CONNECT_TIMEOUT_S = 5.0
KEEP_ALIVE = "30m"

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


def clean_base_url(value) -> str:
    """The configured URL without trailing slashes; "" when unset/not text."""
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


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


def route(base_url, caller: str, background: bool, health: HealthMemo = None) -> str:
    """"remote" or "main" for one call. "remote" needs ALL of: a configured
    URL, a background call, an allowlisted (not owner-facing) caller, and a
    remote host not memoised as down."""
    if not clean_base_url(base_url):
        return "main"
    if not background:
        return "main"
    name = str(caller or "")
    if name in OWNER_FACING_CALLERS or name not in REMOTE_OK_CALLERS:
        return "main"
    if (HEALTH if health is None else health).is_down():
        return "main"
    return "remote"


def build_payload(model: str, system: str, messages: list, max_tokens: int,
                  think=None) -> dict:
    """The /api/chat body for the remote CPU host: the same sampling knobs as
    the main brain, num_ctx pinned through core.ollama_opts.chat_options, and
    NO GPU options (``main_gpu`` etc. are meaningless on a CPU-only host)."""
    options = ollama_opts.chat_options(
        model, num_predict=max_tokens, temperature=0.4,
        extra={"top_p": 0.9, "repeat_penalty": 1.05, "top_k": 40})
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + list(messages),
        "stream": False,
        "options": options,
        "keep_alive": KEEP_ALIVE,
    }
    if think is not None:
        payload["think"] = think
    return payload


def chat(base_url: str, model: str, system: str, messages: list,
         max_tokens: int, timeout_s: float, think=None, post=None):
    """POST one /api/chat to the remote host. Returns (text, body_dict).
    Raises RemoteFailure on a connection error, timeout, non-2xx or empty
    reply; the health memo is updated either way."""
    poster = post or (_requests.post if _requests is not None else None)
    if poster is None:
        raise RemoteFailure("requests unavailable")
    payload = build_payload(model, system, messages, max_tokens, think=think)
    url = clean_base_url(base_url) + "/api/chat"
    try:
        r = poster(url, json=payload,
                   timeout=(CONNECT_TIMEOUT_S, clean_timeout(timeout_s)))
    except Exception as e:
        HEALTH.note_failure()
        raise RemoteFailure(f"{type(e).__name__}: {e}") from e
    status = getattr(r, "status_code", 200)
    if not getattr(r, "ok", 200 <= status < 300):
        if status >= 500:
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
