"""Draft-preview confirmation gate.

Middleware that wraps any ``send_*`` action so the user always hears the draft
read back to them and is given an 8-second window to say "yes" / "confirm" /
"send" before the underlying send fires. Goal: stop JARVIS from ever silently
firing off a message — every outgoing draft routes through a TTS read-back +
voice gate, no matter which skill registered the send action.

Wiring: ``bobert_companion.parse_and_run_actions`` calls ``run_with_gate(name,
arg, fn)`` instead of ``fn(arg)`` whenever ``name.startswith("send_")``. For
actions with no pending draft attached (e.g. the LLM emits send_draft when
nothing is queued), the gate is a transparent no-op — the wrapped function
runs immediately and the gate adds no latency or speech.

The gate is intentionally tolerant of missing dependencies: if TTS fails, the
mic isn't available, or Whisper can't be loaded, we abort the send with a
clear status rather than ever auto-sending. "Fail closed" — silence means no.
"""
from __future__ import annotations

import importlib
import logging
import re
import time
from typing import Any, Callable

_log = logging.getLogger(__name__)

# Words that count as "yes, send it". Any one of these appearing in the
# transcribed reply (whole-word match, case-insensitive) clears the gate.
# Kept aligned with core.draft_confirm so the user only learns one set
# of voice tokens — the skills/draft_preview_gate.py coordinator routes
# both gates through the same vocabulary.
_CONFIRM_KEYWORDS = (
    "yes", "yeah", "yep", "yup",
    "confirm", "confirmed",
    "send", "send it", "ship it",
    "do it", "go ahead", "proceed",
    "affirmative", "okay", "ok",
)

# Words that explicitly cancel. Caught separately so a clear "no" aborts
# even if the user kept talking afterwards.
_CANCEL_KEYWORDS = (
    "no", "nope", "cancel", "abort", "stop", "scrap",
    "don't", "do not", "negative", "hold off", "wait",
)

CONFIRM_TIMEOUT_S = 8.0

_PROMPT_LINE = "Shall I send it, sir?"


def _import_companion():
    """Lazy import so the gate works at registration time before
    bobert_companion has finished loading."""
    try:
        return importlib.import_module("bobert_companion")
    except Exception as e:
        _log.debug("[draft_gate] companion import failed: %s", e)
        return None


def _get_pending(action_name: str = "") -> dict | None:
    """Fetch the current pending draft for ``action_name``, if any.

    Routing:
      * ``send_vip_reply`` (the vip_intercept priority-contact flow)
        looks at ``skills.vip_intercept``.
      * Anything else falls back to ``skills.email_triage`` — preserves the
        original draft-gate behaviour for send_draft / send_pending_draft.

    Uses the public ``get_pending_draft`` if exposed, else falls back to the
    private ``_get_pending`` so the gate keeps working even before a skill
    has been updated. Returns None on any error — callers treat that as
    "no draft to preview, pass through."""
    providers: list[str] = []
    if action_name and "vip" in action_name.lower():
        providers.append("skills.vip_intercept")
    providers.append("skills.email_triage")
    for mod_name in providers:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            _log.debug("[draft_gate] %s import failed: %s", mod_name, e)
            continue
        getter = getattr(mod, "get_pending_draft", None) or getattr(mod, "_get_pending", None)
        if getter is None:
            continue
        try:
            pending = getter()
        except Exception as e:
            _log.debug("[draft_gate] %s pending fetch failed: %s", mod_name, e)
            continue
        if pending:
            return pending
    return None


def _speak(text: str) -> None:
    """Route a line through the companion's TTS path so it shares the
    serialised _SPEAK_LOCK with every other speech caller (mid-task timer,
    tray drainer, proactive_announce). Falls back to a console print so the
    gate stays usable in headless / unit-test contexts."""
    bc = _import_companion()
    speaker = getattr(bc, "_speak", None) if bc is not None else None
    if callable(speaker):
        try:
            speaker(text)
            return
        except Exception as e:
            _log.debug("[draft_gate] _speak failed: %s", e)
    print(f"  [draft_gate] (tts unavailable) {text}")


def _capture_and_transcribe(timeout_s: float) -> str:
    """Block up to ``timeout_s`` waiting for the user to speak, then return
    the transcribed text (lowercase, whitespace-normalised). Empty string
    means "no speech within the window" — the caller treats that as
    silence = abort."""
    bc = _import_companion()
    if bc is None:
        return ""
    record = getattr(bc, "record_speech", None)
    transcribe = getattr(bc, "transcribe", None)
    if not callable(record) or not callable(transcribe):
        return ""
    try:
        audio = record(timeout=timeout_s)
    except Exception as e:
        _log.debug("[draft_gate] record_speech failed: %s", e)
        return ""
    if audio is None:
        return ""
    try:
        text, _meta = transcribe(audio)
    except Exception as e:
        _log.debug("[draft_gate] transcribe failed: %s", e)
        return ""
    return (text or "").strip().lower()


def _matches_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Whole-word match against a keyword list. ``do not`` / ``send it`` are
    handled as substrings (they contain a space). Single tokens use \\b so
    'noted' doesn't trigger on 'no'."""
    if not text:
        return False
    for kw in keywords:
        if " " in kw:
            if kw in text:
                return True
        else:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
    return False


def _readback_text(pending: dict) -> str:
    """Compose the spoken draft preview. Pulls subject + recipient so the
    user can tell *which* draft is being confirmed when several are stacked
    up (rare but possible — email_triage keeps history)."""
    to = (pending.get("to") or "").strip()
    subject = (pending.get("subject") or "").strip()
    body = (pending.get("body") or "").strip()
    head_bits = []
    if to:
        head_bits.append(f"to {to}")
    if subject:
        head_bits.append(f"subject {subject}")
    head = ", ".join(head_bits)
    if head:
        return f"Reading the draft {head}, sir. {body}"
    return f"Reading the draft, sir. {body}"


def should_gate(action_name: str) -> bool:
    """True if this action name should route through the preview gate.

    Any action whose name starts with ``send_`` qualifies — that's the
    contract callers rely on, and it intentionally covers send_draft,
    send_pending_draft, plus any future send_* skill register without
    needing to touch this module."""
    return bool(action_name) and action_name.lower().startswith("send_")


def run_with_gate(action_name: str, arg: str,
                  fn: Callable[[str], Any]) -> Any:
    """Middleware entry point. Wraps ``fn(arg)`` with the preview gate.

    Behaviour:
      * If there is no pending draft, the gate is a transparent no-op —
        ``fn(arg)`` runs immediately. (Some send_* shortcuts the LLM might
        emit with nothing queued; we let the wrapped action surface its
        own "no draft" message rather than synthesise one here.)
      * Otherwise: read the draft body aloud, prompt the user, listen for
        up to ``CONFIRM_TIMEOUT_S`` seconds. If a confirmation keyword is
        heard, call ``fn(arg)`` and return its result. If a cancel keyword
        is heard, or the window times out, abort and return a short
        status string explaining what happened (so the dispatcher can
        surface it like any other action result).

    Fail-closed: any internal error during the gate aborts the send. The
    pending draft stays in email_triage, so the user can retry by saying
    'send' again — we never silently fall through to ``fn(arg)`` when the
    gate itself broke."""
    pending = _get_pending(action_name)
    if not pending:
        # Nothing to preview — let the underlying action speak for itself.
        return fn(arg)

    try:
        _speak(_readback_text(pending))
        _speak(_PROMPT_LINE)
    except Exception as e:
        _log.exception("[draft_gate] readback failed: %s", e)
        return ("Draft preview failed before I could read it back, sir — "
                "holding the send. Say 'send' again to retry.")

    started = time.time()
    heard = _capture_and_transcribe(CONFIRM_TIMEOUT_S)
    waited = time.time() - started

    if not heard:
        return ("No confirmation in the window, sir — holding the draft. "
                "Say 'send' to try again.")

    if _matches_any(heard, _CANCEL_KEYWORDS):
        return (f"Holding the draft, sir — heard '{heard}'. "
                "Say 'send' if you change your mind.")

    if _matches_any(heard, _CONFIRM_KEYWORDS):
        _log.debug("[draft_gate] confirmed after %.1fs: %r", waited, heard)
        return fn(arg)

    # Heard *something* but neither confirm nor cancel. Treat as
    # ambiguous = abort (fail closed) rather than guessing.
    return (f"Couldn't tell whether that was a confirmation, sir — heard "
            f"'{heard}'. Holding the draft; say 'send' to try again.")
