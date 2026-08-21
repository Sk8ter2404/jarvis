"""Pre-send confirmation gate for outbound message actions.

A single shared helper that every outbound channel (teams_nudge,
phone_bridge, future send_sms, etc.) routes through before firing. The
gate speaks the draft body aloud, prompts for confirmation, and waits up
to ``CONFIRM_TIMEOUT_S`` seconds for a yes / no token.

This is a sibling of ``core.draft_preview_gate``: that module is a
middleware shim used by the dispatcher to wrap ``send_*`` actions
registered with the action registry. ``draft_confirm`` is the imperative
counterpart for skills that fire outbound messages from background
threads (Teams nudges, phone pushes) and never go through the
dispatcher's send_* path. Both share the same confirm/cancel keyword
sets and the same fail-closed semantics so the user only learns one set
of voice tokens.

Public surface:

    confirmed = draft_confirm(text, recipient)
        → True  iff the user explicitly confirmed within the window
        → False on cancel, timeout, ambiguous reply, or any internal
                error (mic unavailable, TTS down, whisper unloaded)

The bool return is intentional — callers wrap their own send code with
``if not draft_confirm(...): return`` and the abort path stays in their
hands, so each skill can decide what to log / queue / drop when a send
is denied.
"""
from __future__ import annotations

import importlib
import logging
import os
import re
import sys
import threading
import time
from typing import Any

_log = logging.getLogger(__name__)

# Resolve project root so we can persist pending drafts even when this
# module is loaded under different package paths (skills/* imports it as
# `core.draft_confirm`, the dispatcher imports it the same way, and unit
# tests sometimes load it via importlib).
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)  # pragma: no cover - import-time path bootstrap; the project root is already on sys.path under the test runner, so this branch only fires in a bare interpreter and can't be re-triggered post-import

from core.atomic_io import _atomic_write_json  # noqa: E402

# The mute rule and the "a send is being held" publisher live in the sibling
# gate and are IMPORTED, never mirrored — the #1 bug class in this repo is the
# stale duplicate (a rule fixed in one copy while the others rot), and this
# file already hand-mirrors _CONFIRM_KEYWORDS / _CANCEL_KEYWORDS. If the
# sibling is unavailable we claim no cause rather than growing a second copy
# of the rule: the refusal falls back to the generic wording.
try:                                                       # noqa: E402
    from core.draft_preview_gate import (MUTED_REASON, UNVOICED_REASON,
                                         publish_held_send, tts_is_muted)
except Exception as _e:  # pragma: no cover - sibling core module is in-tree
    MUTED_REASON = 'tray "Mute TTS" is on'
    UNVOICED_REASON = "the speech path did not voice it"
    tts_is_muted = lambda: False            # noqa: E731 - no local mute rule
    publish_held_send = lambda *a, **k: None  # noqa: E731
    logging.getLogger(__name__).warning(
        "[draft_confirm] core.draft_preview_gate unavailable (%s) — a held "
        "send will still be refused, but the cause cannot be named", _e)

# Words that count as "yes, send it". Whole-word match,
# case-insensitive. Kept aligned with core.draft_preview_gate so the user
# only learns one vocabulary.
_CONFIRM_KEYWORDS: tuple[str, ...] = (
    "yes", "yeah", "yep", "yup",
    "confirm", "confirmed",
    "send", "send it", "ship it",
    "do it", "go ahead", "proceed",
    "affirmative", "okay", "ok",
)

_CANCEL_KEYWORDS: tuple[str, ...] = (
    "no", "nope", "cancel", "abort", "stop", "scrap",
    "don't", "do not", "negative", "hold off", "wait",
)

CONFIRM_TIMEOUT_S = 8.0

_PENDING_FILE = os.path.join(_PROJECT_DIR, "data", "draft_confirm_pending.json")

# Serialises the prompt / record / transcribe sequence across callers so
# two skills firing nudges at the same moment can't double-prompt the
# user or race on the shared mic input stream. The lock is module-level
# (one per process); the file write inside is its own atomic step via
# core.atomic_io._atomic_write_json.
_gate_lock = threading.Lock()

# Why the last confirmation was refused before it ever prompted, or "".
# Single-element list, mutated in place (the core/state.py convention) so a
# background nudge thread's write is visible to a reader on the voice thread.
_last_hold_reason: list[str] = [""]


def last_hold_reason() -> str:
    """The cause of the most recent unvoiced-read-back refusal, or "".

    Exposed so a diagnostic ("outbound gate status") can answer "why did
    nothing send?" without the owner having to find it in the log — which,
    when the cause is muted TTS, he cannot be told any other way.
    """
    return _last_hold_reason[0]


def _report_unvoiced_hold(body: str, recipient: str) -> None:
    """Announce a confirmation refused because the read-back was not voiced.

    THE POINT (2026-08-20 review, HIGH): failing closed on an unvoiced
    read-back is right, but tray "Mute TTS" is a deliberate owner toggle that
    survives reboots, so with it on this gate refused EVERY outbound draft
    forever — and the refusal itself was inaudible by construction. A refusal
    nobody can perceive is indistinguishable from a bug. So the cause goes to
    the console and the HUD, where a muted owner can actually see it.

    There is deliberately NO owner-override here, unlike
    core.draft_preview_gate.run_with_gate. This module's callers (teams_nudge,
    phone_bridge) fire from BACKGROUND threads with no user turn attached, so
    the last thing he said is not an answer to THIS prompt and must never be
    read as consent for it. Holding a JARVIS-initiated push is the correct
    answer; what changed is that the hold is no longer silent.
    """
    reason = MUTED_REASON if tts_is_muted() else UNVOICED_REASON
    _last_hold_reason[0] = reason
    _log.warning("[draft_confirm] read-back was NOT voiced (%s) — refusing the "
                 "send. Nothing was read aloud, so nothing was confirmed.",
                 reason)
    try:
        publish_held_send(_prompt_line(body, recipient), reason,
                          source="draft_confirm")
    except Exception as e:  # pragma: no cover - reporting is never load-bearing
        _log.debug("[draft_confirm] hold publish failed: %s", e)


def _import_companion():
    """Lazy import so the gate works at registration time before
    bobert_companion has finished loading."""
    try:
        return importlib.import_module("bobert_companion")
    except Exception as e:
        _log.debug("[draft_confirm] companion import failed: %s", e)
        return None


def _speak(text: str) -> bool:
    """Route the prompt through the companion's TTS path so it shares the
    serialised _SPEAK_LOCK with every other speech caller.

    Returns True **iff the line was actually voiced**. Anything else — no
    companion, no callable speaker, a raise, or a companion ``_speak``
    that returned anything other than literal ``True`` — returns False and
    the caller must abort (fail-closed: never auto-send when we couldn't
    even read the draft aloud).

    The strict ``is True`` check below is load-bearing.
    ``bobert_companion._speak`` has four exits and only one of them
    returns True: it returns ``None`` when the tray "Mute TTS" toggle is
    on (a state that PERSISTS ACROSS REBOOTS), ``None`` when nothing
    audible survives tag/markdown stripping, ``None`` on a staging
    instance, and ``False`` when synthesis/playback fails. None of those
    raise. Treating a silent no-op as "the owner heard the draft" would
    open the confirmation window with nobody having been read to, and any
    stray "okay"/"yeah" in the room would confirm the send."""
    bc = _import_companion()
    speaker = getattr(bc, "_speak", None) if bc is not None else None
    if not callable(speaker):
        _log.warning("[draft_confirm] no TTS available; aborting confirmation")
        return False
    try:
        ok = speaker(text)
    except Exception as e:
        _log.warning("[draft_confirm] _speak raised: %s", e)
        return False
    # Strict identity, NOT bool(ok) / truthiness: a MagicMock is truthy, and
    # so is any future non-bool sentinel. Only a literal True means voiced.
    if ok is not True:
        _log.warning("[draft_confirm] readback was NOT voiced "
                     "(muted / nothing audible / staging / playback failed) "
                     "— aborting confirmation")
        return False
    return True


def _capture_and_transcribe(timeout_s: float) -> str | None:
    """Block up to ``timeout_s`` for the user to speak, then return the
    transcribed text (lowercase, whitespace-normalised).

    Returns:
      str  — the heard text (possibly empty if whisper returned nothing)
      None — hard failure (mic unavailable, whisper unloaded, etc.).
             Caller treats this as fail-closed = abort, distinct from an
             empty string which means "silence in the window".
    """
    bc = _import_companion()
    if bc is None:
        return None
    record = getattr(bc, "record_speech", None)
    transcribe = getattr(bc, "transcribe", None)
    if not callable(record) or not callable(transcribe):
        return None
    try:
        audio = record(timeout=timeout_s)
    except Exception as e:
        _log.warning("[draft_confirm] record_speech failed: %s", e)
        return None
    if audio is None:
        return ""   # silence in the window — caller treats as abort
    try:
        text, _meta = transcribe(audio)
    except Exception as e:
        _log.warning("[draft_confirm] transcribe failed: %s", e)
        return None
    return (text or "").strip().lower()


def _matches_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Whole-word keyword match. Multi-word phrases (e.g. 'send it',
    'do not') match as substrings; single tokens use \\b so 'noted'
    doesn't trip on 'no'."""
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


def _write_pending(record: dict | None) -> None:
    """Persist the active draft to disk so a watchdog / restart inspector
    can see what was held when something went wrong. Best-effort — a
    write failure logs but doesn't abort the confirmation."""
    try:
        os.makedirs(os.path.dirname(_PENDING_FILE), exist_ok=True)
        payload: dict[str, Any] = {"active": record} if record else {"active": None}
        _atomic_write_json(_PENDING_FILE, payload)
    except Exception as e:
        _log.debug("[draft_confirm] pending write failed: %s", e)


def _prompt_line(text: str, recipient: str) -> str:
    """Compose the spoken read-back. Falls back to a generic phrasing
    when no recipient name was supplied."""
    rcpt = (recipient or "").strip()
    body = (text or "").strip()
    if rcpt:
        return f"I have a draft for {rcpt}, sir: {body}. Shall I send it?"
    return f"I have a draft, sir: {body}. Shall I send it?"


def draft_confirm(text: str, recipient: str = "") -> bool:
    """Read the draft aloud and wait for an explicit yes / no token.

    Returns True iff the user spoke a confirmation keyword within
    ``CONFIRM_TIMEOUT_S`` seconds. Every other outcome — explicit
    cancellation, silence past the window, ambiguous reply, mic failure,
    whisper unavailable, TTS down — returns False. The caller is
    responsible for deciding what to log / drop / re-queue on a False
    return; this module just gates the send.

    Thread-safe: a process-wide lock serialises the prompt + record +
    transcribe cycle so simultaneous calls from background skills don't
    double-prompt or race the mic stream.
    """
    body = (text or "").strip()
    if not body:
        # Nothing to confirm. Caller passed an empty draft; refuse so
        # the silent no-op doesn't accidentally read as "user said yes".
        return False

    rcpt = (recipient or "").strip()
    prompt = _prompt_line(body, rcpt)

    with _gate_lock:
        _write_pending({
            "ts":        time.time(),
            "to":        rcpt,
            "body":      body,
            "prompt":    prompt,
        })
        try:
            if not _speak(prompt):
                # Fail closed — but SAY SO somewhere he can see it, and record
                # the cause. See _report_unvoiced_hold.
                _report_unvoiced_hold(body, rcpt)
                return False
            _last_hold_reason[0] = ""

            heard = _capture_and_transcribe(CONFIRM_TIMEOUT_S)
            if heard is None:
                # Hard failure (mic / whisper). Fail closed.
                return False
            if not heard:
                # Silence in the window — treat as "no answer = no".
                return False
            if _matches_any(heard, _CANCEL_KEYWORDS):
                return False
            if _matches_any(heard, _CONFIRM_KEYWORDS):
                return True
            # Heard something that wasn't yes-shaped or no-shaped.
            # Ambiguous → fail closed.
            _log.debug("[draft_confirm] ambiguous reply: %r", heard)
            return False
        except Exception:
            # Any unexpected error: fail closed.
            _log.exception("[draft_confirm] gate raised")
            return False
        finally:
            _write_pending(None)
