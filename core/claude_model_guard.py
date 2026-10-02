"""Retired / unknown Claude model guard (2026-10-02).

WHY
===
Anthropic retires Claude models on a published schedule (the platform docs'
"Model deprecations" page). A request to a retired model fails, and a model id
that does not exist, or that the key's organization cannot use, gets the very
same answer: HTTP 404, ``error.type == "not_found_error"``, with a message that
starts ``model: <id>``. Checked 2026-10-02: ``claude-haiku-4-5-20251001`` (the
snapshot behind the ``claude-haiku-4-5`` alias the notification sorter, the
email triage and the briefing workers use) is listed Active, deprecated N/A,
tentative retirement "Not sooner than October 15, 2026"; the same page promises
at least 60 days' notice before a publicly released model retires.

Before this guard every Claude caller treated that 404 like any other cloud
hiccup: each call paid the round trip again, each caller logged it its own way
(or not at all), and the browser agent, which has no local path, read the raw
error body aloud. Nothing said, once and plainly, "this model is gone".

WHAT THIS MODULE HOLDS (stdlib only, so the CI-light tier covers it)
=====================================================================
* ``is_model_not_found(err, model=None)`` -- is this exception (or error text)
  Anthropic's "model not found" answer? Reads ``status_code`` / ``body`` the
  way the SDK's APIStatusError carries them, walks ``__cause__`` /
  ``__context__`` (browser-use wraps the SDK error in its own
  ModelProviderError), and falls back to the error text. A 404 alone is not
  enough: the text must name a model, so an unrelated 404 never retires one.
* ``ModelGuard`` / ``GUARD`` -- this session's record of models Anthropic said
  are gone. ``note_not_found`` logs ONE line per model per session;
  ``resolve`` picks the model a call should use: the configured one, else its
  configured successor, else None (the caller's local path answers).
  A model marked gone is tried for real again after ``recheck_after_s`` so a
  one-off answer can never strand it for a whole long session; a repeat 404
  re-marks it silently, a success clears it with one "answering again" line.
* ``RetiredModelError`` -- what core.llm_client raises INSTEAD of a network
  call for a model already known to be gone with no successor configured. A
  RuntimeError, so every caller's existing ``except Exception`` fallback (the
  local brain, raw tool data, an honest line) runs exactly as it did for the
  404 itself.

Log lines carry model ids and caller names only -- never prompt text.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping, Optional

# A model marked gone gets one real request again after this long (seconds).
DEFAULT_RECHECK_AFTER_S = 3600.0

_TAG = "[model-guard]"


class RetiredModelError(RuntimeError):
    """A Claude call skipped because Anthropic already answered not_found for
    this model this session and no successor is configured."""

    def __init__(self, model: str, where: str = ""):
        self.model = model
        self.where = where
        super().__init__(
            f"Claude model {model!r} is retired or unknown to the API "
            f"(Anthropic answered not_found earlier this session); pick a "
            f"current model in Settings or add it to CLAUDE_MODEL_SUCCESSORS")


def _norm(model: Any) -> str:
    """Lower-case, stripped model id; '' for anything unusable."""
    try:
        return str(model or "").strip().lower()
    except Exception:
        return ""


def _error_parts(err: Any) -> tuple[Optional[int], str, str]:
    """(status_code, error.type, text) of ONE exception or string. Never
    raises."""
    if isinstance(err, str):
        return None, "", err
    status = None
    etype = ""
    texts: list[str] = []
    try:
        sc = getattr(err, "status_code", None)
        if sc is None:
            sc = getattr(getattr(err, "response", None), "status_code", None)
        if isinstance(sc, int) and not isinstance(sc, bool):
            status = sc
    except Exception:
        pass
    try:
        body = getattr(err, "body", None)
        if isinstance(body, Mapping):
            inner = body.get("error")
            if not isinstance(inner, Mapping):
                inner = body
            t = inner.get("type")
            if isinstance(t, str):
                etype = t
            m = inner.get("message")
            if isinstance(m, str):
                texts.append(m)
    except Exception:
        pass
    for attr in ("message",):
        try:
            v = getattr(err, attr, None)
            if isinstance(v, str):
                texts.append(v)
        except Exception:
            pass
    try:
        texts.append(str(err))
    except Exception:
        pass
    return status, etype, " ".join(texts)


def _one_is_not_found(err: Any, model: str) -> bool:
    status, etype, text = _error_parts(err)
    low = text.lower()
    says_not_found = (status == 404 or etype == "not_found_error"
                      or "not_found_error" in low)
    if not says_not_found:
        return False
    # The answer must be ABOUT a model: Anthropic's message is "model: <id>".
    # An unrelated 404 (a Files API id, a bad endpoint) never retires one.
    return ("model" in low) or bool(model and model in low)


def is_model_not_found(err: Any, model: Any = None) -> bool:
    """True when ``err`` (an exception, or an error string) is Anthropic's
    answer that a model does not exist / is retired / is not available to
    this key: HTTP 404 or ``not_found_error``, about a model. Follows
    ``__cause__`` / ``__context__`` a few levels. Never raises."""
    m = _norm(model)
    seen = 0
    cur = err
    try:
        while cur is not None and seen < 5:
            if _one_is_not_found(cur, m):
                return True
            if isinstance(cur, str):
                return False
            nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
            cur = nxt if nxt is not cur else None
            seen += 1
    except Exception:
        return False
    return False


def successor_for(model: Any, table: Optional[Mapping] = None) -> str:
    """The configured replacement for ``model`` ('' when none): an exact
    (case-insensitive) key first, then the id without a ``-YYYYMMDD`` snapshot
    suffix, so one entry for ``claude-haiku-4-5`` also covers
    ``claude-haiku-4-5-20251001``. A successor equal to the model is no
    successor."""
    m = _norm(model)
    if not m or not isinstance(table, Mapping):
        return ""
    try:
        lowered = {_norm(k): _norm(v) for k, v in table.items()}
    except Exception:
        return ""
    cands = [m]
    base = m
    if len(base) > 9 and base[-9] == "-" and base[-8:].isdigit():
        cands.append(base[:-9])
    for c in cands:
        s = lowered.get(c, "")
        if s and s != m:
            return s
    return ""


class ModelGuard:
    """Thread-safe, per-session record of Claude models Anthropic answered
    not_found for. ``sink`` receives each log line (print by default);
    ``clock`` is the monotonic clock (injectable for tests)."""

    def __init__(self, sink: Optional[Callable[[str], Any]] = None,
                 clock: Optional[Callable[[], float]] = None,
                 recheck_after_s: float = DEFAULT_RECHECK_AFTER_S):
        self._lock = threading.Lock()
        self._sink = sink
        self._clock = clock or time.monotonic
        self.recheck_after_s = float(recheck_after_s)
        # model -> monotonic stamp of the latest not_found answer
        self._gone: dict[str, float] = {}
        # models already announced this session (ONE line per model)
        self._announced: set[str] = set()

    # ── logging ──────────────────────────────────────────────────────────
    def _emit(self, line: str) -> None:
        try:
            (self._sink or print)(line)
        except Exception:
            pass

    # ── state ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Forget everything (tests; a fresh session)."""
        with self._lock:
            self._gone.clear()
            self._announced.clear()

    def is_retired(self, model: Any) -> bool:
        """Is ``model`` marked gone right now (inside its recheck window)?"""
        m = _norm(model)
        if not m:
            return False
        with self._lock:
            stamp = self._gone.get(m)
            if stamp is None:
                return False
            return (self._clock() - stamp) < self.recheck_after_s

    def retired_models(self) -> list[str]:
        """Every model marked gone this session (recheck window ignored)."""
        with self._lock:
            return sorted(self._gone)

    def note_not_found(self, model: Any, where: str = "",
                       successor: str = "") -> bool:
        """Record Anthropic's not_found answer for ``model``. Logs ONE line the
        first time per model per session (returns True then), silent after.
        ``where`` names the caller; ``successor`` is what calls will use
        instead ('' = the caller's local path)."""
        m = _norm(model)
        if not m:
            return False
        with self._lock:
            self._gone[m] = self._clock()
            first = m not in self._announced
            if first:
                self._announced.add(m)
        if first:
            via = f" (first seen by {where})" if where else ""
            if successor:
                then = (f"Claude calls for it now use {successor} "
                        f"(CLAUDE_MODEL_SUCCESSORS)")
            else:
                then = ("features using it fall back to their local path; "
                        "pick a current model in Settings or add it to "
                        "CLAUDE_MODEL_SUCCESSORS")
            mins = max(1, int(round(self.recheck_after_s / 60.0)))
            self._emit(f"  {_TAG} Anthropic answered not_found for {m}{via} "
                       f"— the model is retired or this key cannot use it; "
                       f"{then}. Re-tried every {mins} min; this is the only "
                       f"line this session.")
        return first

    def note_ok(self, model: Any) -> None:
        """A real call to ``model`` succeeded: clear a gone mark (one line when
        there was one). Cheap no-op for a model never marked."""
        m = _norm(model)
        if not m:
            return
        with self._lock:
            was = self._gone.pop(m, None)
        if was is not None:
            self._emit(f"  {_TAG} {m} is answering again — cleared its "
                       f"not_found mark.")

    def resolve(self, model: Any, table: Optional[Mapping] = None) -> Optional[str]:
        """The model a Claude call should use: ``model`` itself when it is not
        marked gone, else its configured successor (from ``table``) when that
        is not marked gone either, else None -- skip the cloud, the caller's
        local path answers. A blank model resolves to itself (the caller's own
        validation handles it)."""
        m = _norm(model)
        if not m or not self.is_retired(m):
            return model
        succ = successor_for(m, table)
        if succ and not self.is_retired(succ):
            return succ
        return None


# The process-wide guard every Claude caller shares (via core.llm_client).
GUARD = ModelGuard()
