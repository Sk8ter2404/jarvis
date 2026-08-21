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

MUTED vs BROKEN (2026-08-20 review, HIGH — a regression from this morning's
fail-closed fix). Failing closed on an unvoiced read-back is right, but tray
"Mute TTS" is a DELIBERATE owner choice that is restored across reboots, so
with it on, every send_* draft was held forever, the refusal said "say 'send'
again to retry" — a retry that could never succeed — and the refusal itself was
inaudible by construction. That is a silent, permanent block.

Three things changed, and the safety property is unchanged: JARVIS still never
auto-sends a draft the owner has not heard.
  1. The gate distinguishes MUTED from BROKEN and names the cause and the
     remedy in the refusal instead of asking for a retry that cannot work.
  2. A held send is PUBLISHED — console + HUD state, with the draft body — so
     the hold is visible even though nothing can be voiced.
  3. There is an explicit owner override: if what he actually said asks for an
     unheard send ("send it anyway", "send it without reading it back"), the
     send proceeds and says plainly that it went out unread. That is his own
     deliberate instruction, not JARVIS deciding silence meant yes — the
     ordinary confirm vocabulary ("yes", "okay", "send") still cannot clear an
     unvoiced gate, so a stray word in the room can never send anything.
"""
from __future__ import annotations

import importlib
import logging
import re
import sys
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


def _loaded_companion():
    """The ALREADY-LOADED monolith, or None — never an import.

    ``_import_companion`` genuinely has to import (it is how the gate finds a
    speaker). The two read-only consumers below must NOT: importing
    bobert_companion runs its import-time boot code, which in a process that
    is not JARVIS prints the singleton refusal and drags the whole monolith in
    just to write a HUD key. Same rule skills/self_diagnostic._bc() follows.
    """
    return sys.modules.get("bobert_companion")


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


# Why a read-back was not voiced, in the owner's words. Shared with
# core.draft_confirm so both gates name the same cause the same way.
MUTED_REASON = 'tray "Mute TTS" is on'
UNVOICED_REASON = "the speech path did not voice it"

# Outcome of one attempted read-back line. "voiced" is the only success.
_VOICED = "voiced"
_MUTED = "muted"        # deliberate owner choice (tray "Mute TTS"), not a fault
_UNVOICED = "unvoiced"  # no TTS, synthesis/playback failed, staging, empty text


def tts_is_muted() -> bool:
    """True when the tray "Mute TTS" toggle is on.

    PUBLIC and shared: core.draft_confirm imports this rather than keeping its
    own copy. The #1 bug class in this repo is the stale duplicate — a rule
    fixed in one copy while the others rot — and "is TTS muted?" is exactly the
    kind of one-line rule that grows mirrors.

    Reads ``core.state._tts_muted`` — the canonical cell.
    ``bobert_companion._tts_muted`` is the SAME list object (the monolith
    re-exports core.state via ``from core.state import *``), so this needs no
    monolith import and works before the companion has finished loading; the
    companion is consulted only as a fallback. Any odd shape reads as
    "not muted", which routes the outcome to the generic BROKEN wording — the
    conservative direction, since that message never claims a cause.
    """
    try:
        from core import state as _state_mod
    except Exception:       # pragma: no cover - core/ is in-tree
        _state_mod = None
    # The companion entry is _loaded_companion(), NOT an import: this is a
    # read, and a read must never drag the monolith into a process that has
    # not loaded it.
    for mod in (_state_mod, _loaded_companion()):
        if mod is None:
            continue
        cell = getattr(mod, "_tts_muted", None)
        # Strict shape check, not truthiness: a MagicMock is truthy and
        # subscriptable, and would answer "muted" for every test double.
        if isinstance(cell, list) and cell:
            try:
                return bool(cell[0])
            except Exception:
                continue
    return False


def _speak_outcome(text: str) -> str:
    """Attempt one read-back line and classify the result.

    Delegates the actual voicing to ``_speak`` (kept as the single speaking
    primitive so there is one place that knows what "voiced" means — and so
    tests that patch ``_speak`` still drive the whole gate). The only thing
    added here is WHY a line was not voiced, which is what lets the refusal
    name a cause and a remedy instead of asking for an impossible retry.
    """
    # Strict `is True`, never truthiness: the guard this whole module exists
    # for. A MagicMock is truthy, and so is any future non-bool sentinel.
    if _speak(text) is True:
        return _VOICED
    # Read the flag AFTER the attempt: bobert_companion._speak checks
    # `_tts_muted[0]` itself at call time, so this is the same instant's answer.
    return _MUTED if tts_is_muted() else _UNVOICED


# What the owner actually said, if it EXPLICITLY asks for a send he has not
# heard. Deliberately NOT the ordinary confirm vocabulary: "yes" / "okay" /
# "send it" must never clear an unvoiced gate, or a stray word in the room
# would send a draft nobody heard — the exact hole the fail-closed fix shut.
# Each pattern names the unheard-ness out loud, and _owner_asked_for_unheard_send
# additionally requires the word "send", so a stale utterance cannot be read as
# consent for some later background send.
_UNHEARD_OVERRIDE_PATTERNS: tuple[str, ...] = (
    r"\bsend\s+(?:it\s+|that\s+|them\s+)?anyway\b",
    r"\bsend\s+(?:it\s+)?(?:unheard|unread|blind)\b",
    r"\b(?:without|skip|skipping)\b[^.]{0,24}\bread[\s-]?back\b",
    r"\b(?:without|skip|skipping)\b[^.]{0,24}\breading it (?:back|out|aloud)\b",
    r"\bdon'?t\s+(?:bother\s+)?read(?:ing)?\s+it\b",
    r"\bno\s+read[\s-]?back\b",
    r"\bi\s+(?:already\s+)?know\s+what\s+(?:it|the draft)\s+says\b",
)


# A negated or cancelled send can NEVER be an override, however the rest of the
# sentence reads: "do not send it anyway" contains the literal override phrase
# and means the exact opposite. Deliberately narrower than _CANCEL_KEYWORDS,
# which contains a bare "no" and would veto the legitimate "no readback, send
# it anyway".
_OVERRIDE_VETO_PATTERNS: tuple[str, ...] = (
    r"\b(?:do not|don'?t|never)\s+(?:just\s+|actually\s+|ever\s+)?send\b",
    r"\b(?:cancel|abort|scrap)\b",
)


def _last_user_utterance() -> str:
    """The utterance that produced this turn's actions, or "".

    ``bobert_companion._last_user_text`` is set in ``_call_llm`` immediately
    before the LLM dispatch whose reply carried this ``send_*`` token, so it is
    what he actually said. Strict list-of-str shape check: a MagicMock would
    otherwise stringify into something a pattern could match.
    """
    bc = _loaded_companion()
    cell = getattr(bc, "_last_user_text", None) if bc is not None else None
    if isinstance(cell, list) and cell and isinstance(cell[0], str):
        return cell[0]
    return ""


def _owner_asked_for_unheard_send() -> bool:
    """True only when the owner's own words asked for a send WITHOUT a
    read-back. This is the honest route out of a muted gate: he cannot hear
    the draft, so he says so and takes it explicitly — JARVIS never infers it.
    """
    low = _last_user_utterance().strip().lower()
    if not low or not re.search(r"\bsend\b", low):
        return False
    if any(re.search(p, low) for p in _OVERRIDE_VETO_PATTERNS):
        return False
    return any(re.search(p, low) for p in _UNHEARD_OVERRIDE_PATTERNS)


def publish_held_send(readback: str, reason: str, *,
                      source: str = "draft_gate") -> bool:
    """Make a held send VISIBLE. Best-effort, never raises.

    The refusal string alone is not enough here: when the hold is caused by
    "Mute TTS" the refusal is inaudible by construction, so it has to land
    somewhere he can see — the console log and the HUD state file. Pass
    ``readback=""`` to clear the marker once the send goes through.

    PUBLIC and shared with core.draft_confirm for the same reason
    ``tts_is_muted`` is: one home for the rule, no mirrors to rot.

    Returns True only when the HUD actually received it — the caller's refusal
    says "it is on the HUD", and under the honest-failure contract that
    sentence may not be spoken unless it is true. The console print is
    unconditional, so the caller always has one channel it can promise.
    """
    bc = _loaded_companion()
    if readback:
        print(f"  [{source}] SEND HELD — {reason}. The draft was NOT read "
              f"aloud, so it has NOT been sent.")
        print(f"  [{source}] draft: {readback}")
        print(f"  [{source}] un-mute and say 'send' again, or say "
              f"'send it anyway' to send it unheard.")
    try:
        from core.config import HUD_ENABLED as _hud_on
    except Exception:           # pragma: no cover - core/ is in-tree
        _hud_on = False
    try:
        writer = getattr(bc, "_write_hud_state", None) if bc is not None else None
        if _hud_on and callable(writer):
            # _write_hud_state swallows its own errors and no-ops when the HUD
            # is off, so HUD_ENABLED is checked here rather than inferred from
            # a return value it does not have.
            writer(held_send=readback or "",
                   held_send_reason=reason if readback else "",
                   held_send_source=source if readback else "",
                   held_send_at=time.time() if readback else 0.0,
                   tts_muted=tts_is_muted())
            return True
    except Exception as e:      # pragma: no cover - HUD is never load-bearing
        _log.debug("[%s] HUD publish failed: %s", source, e)
    return False


def _speak(text: str) -> bool:
    """Route a line through the companion's TTS path so it shares the
    serialised _SPEAK_LOCK with every other speech caller (mid-task timer,
    tray drainer, proactive_announce).

    Returns True **iff the line was actually voiced**, so ``run_with_gate``
    can hold the send when the owner never heard the read-back. This module's
    docstring promises "if TTS fails ... we abort the send"; that promise is
    only real if this function reports the failure.

    Every non-voiced outcome returns False:
      * no companion / no callable ``_speak``  → console print, then False.
        A console line is NOT a read-back the owner heard, so a headless
        context is a *refusal*, not a pass. That is the correct reading for
        a send gate.
      * ``speaker(text)`` raised                → False.
      * ``speaker(text)`` returned anything other than literal ``True``
        → False. ``bobert_companion._speak`` returns ``None`` when the tray
        "Mute TTS" toggle is on (a state restored across reboots), ``None``
        when nothing audible survives tag/markdown stripping, ``None`` on a
        staging instance, and ``False`` when synthesis/playback fails — none
        of which raise. Strict ``is True``, never ``bool(...)``: a MagicMock
        is truthy and would re-open this exact hole in the tests."""
    bc = _import_companion()
    speaker = getattr(bc, "_speak", None) if bc is not None else None
    if callable(speaker):
        try:
            ok = speaker(text)
        except Exception as e:
            _log.warning("[draft_gate] _speak raised: %s", e)
            return False
        if ok is not True:
            _log.warning("[draft_gate] readback was NOT voiced "
                         "(muted / nothing audible / staging / playback "
                         "failed) — failing closed")
            return False
        return True
    _log.warning("[draft_gate] no TTS available; a console print is not a "
                 "readback — failing closed")
    print(f"  [draft_gate] (tts unavailable) {text}")
    return False


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
    gate itself broke. See MUTED vs BROKEN in the module docstring for the one
    deliberate, owner-spoken way past an unvoiced read-back."""
    pending = _get_pending(action_name)
    if not pending:
        # Nothing to preview — let the underlying action speak for itself.
        return fn(arg)

    # Both lines must be genuinely VOICED before the confirm window opens.
    # `_speak` returns False (without raising) when TTS is muted, staged, or
    # playback failed — treating that silence as "the owner heard the draft"
    # is what made this gate fail open. Strict `is True` so a truthy test
    # double can't stand in for a real read-back. The try/except stays because
    # _readback_text can still raise (pending.get on a truthy non-dict).
    readback = ""
    try:
        readback = _readback_text(pending)
        outcome = _speak_outcome(readback)
        if outcome == _VOICED:
            outcome = _speak_outcome(_PROMPT_LINE)
    except Exception as e:
        _log.exception("[draft_gate] readback failed: %s", e)
        readback = ""
        outcome = _UNVOICED
    if outcome != _VOICED:
        return _hold_unvoiced(arg, readback, outcome, fn)

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
        publish_held_send("", "")      # nothing is being held any more
        return fn(arg)

    # Heard *something* but neither confirm nor cancel. Treat as
    # ambiguous = abort (fail closed) rather than guessing.
    return (f"Couldn't tell whether that was a confirmation, sir — heard "
            f"'{heard}'. Holding the draft; say 'send' to try again.")


def _hold_unvoiced(arg: str, readback: str, outcome: str,
                   fn: Callable[[str], Any]) -> Any:
    """The read-back was not voiced. Decide what the owner is TOLD, and honour
    an explicit instruction to send unheard.

    The safety property is unchanged and load-bearing: the confirm window is
    never opened, so no spoken word — stray or deliberate — can confirm a draft
    he did not hear. The only way through is his own utterance explicitly
    asking for an unheard send.

    What changed is everything else:
      * the hold is PUBLISHED to console + HUD with the draft body, because a
        refusal delivered only by TTS is inaudible exactly when TTS is the
        thing that failed;
      * the refusal names the cause (muted vs broken) and a remedy that can
        actually work, instead of "say 'send' again to retry" against a
        persistent toggle, which never could;
      * every returned line keeps a FAILURE_MARKER substring ("couldn't"), on
        purpose: send_* is not in SPEAK_RESULT_VERBATIM_ACTIONS, so a
        marker-free result would be classified as a plain success and never
        surfaced to him at all — silence is the failure mode being fixed here.
    """
    muted = (outcome == _MUTED)
    reason = (MUTED_REASON if muted else UNVOICED_REASON)
    _log.warning("[draft_gate] readback not voiced (%s) — holding the send",
                 reason)
    on_hud = publish_held_send(readback, reason)
    where = ("on the HUD and in the console" if on_hud
             else "in the console log")

    # The one honest route out. Requires a composed draft: if _readback_text
    # raised there is nothing we could have shown him either, so there is
    # nothing for him to have consented to.
    if readback and _owner_asked_for_unheard_send():
        _log.warning("[draft_gate] owner explicitly asked for an unheard send "
                     "(%s) — sending without a read-back", reason)
        publish_held_send("", "")
        result = fn(arg)
        note = (f"Sent without reading it back, sir, on your explicit "
                f"instruction — {reason}. ")
        return note + (result if isinstance(result, str) else str(result))

    if muted:
        return (f"TTS is muted, sir, so I couldn't read that draft back to you "
                f"— and I will not send a draft you haven't heard. I've put it "
                f"{where}. Un-mute and say 'send' again, or tell me to send it "
                f"anyway and it goes out unread.")
    return (f"I couldn't read that draft back to you, sir — the speech path "
            f"produced nothing, so I'm holding the send. The draft is "
            f"{where}. Fix the audio and say 'send' again, or tell me to send "
            f"it anyway and it goes out unread.")
