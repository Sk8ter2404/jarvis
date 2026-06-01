"""Draft preview gate — unified outbound message confirmation skill.

This skill is the registration / coordination point for the standing rule
"every outbound message gets read aloud before it sends". The heavy
lifting lives in two core modules:

  core.draft_preview_gate  middleware the dispatcher wraps around every
                           send_* action (email_triage, vip_intercept,
                           any future send_<channel> verb). Bound in
                           bobert_companion.parse_and_run_actions.
  core.draft_confirm       imperative confirm() helper for background /
                           non-dispatcher sends (Teams nudges, phone
                           pushes, future SMS / Slack / Discord bridges).

What this skill adds on top:

  1. ``gate_outbound_message(channel, draft)`` — a single helper any skill
     can call regardless of whether its outbound path uses the send_*
     dispatcher contract. Composes a channel-aware prompt
     ("I have a reply drafted for Sam, sir. Shall I read it before
     sending?"), reads the body via TTS, listens for the user's verdict,
     returns True iff the user explicitly confirmed.

  2. ``draft_preview_gate_status`` action — diagnostics. Reports whether
     the core gates loaded, which channels are currently wired, and any
     last-error from the underlying confirmation infrastructure.

  3. Sleep / standby suppression. If JARVIS is asleep or in standby when
     a gated send is attempted, the gate refuses rather than prompting —
     the user would otherwise be voice-confirming messages while they
     literally slept. Background hardware/diagnostic alerts that need
     fire-and-forget delivery bypass the gate by calling
     ``core.draft_confirm`` directly with their own confirm=False semantics
     (or skipping it entirely), so this suppression only affects pushes
     that *would* have prompted.

Why a skill and not just core? The dispatcher already auto-routes
send_* through ``core.draft_preview_gate``; new channels that follow
that naming convention get gated for free. This skill exists for the
channels that DON'T fit that pattern (phone_bridge.push_to_phone,
teams_nudge.queue_teams_nudge) and for unified diagnostics, plus it
gives us a single place to hang future extensions (SMS, Slack, Discord)
without scattering the standing read-aloud-before-sending rule across
every outbound module.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any

# Project root on sys.path so `core.*` resolves whether this module is
# loaded as `skills.draft_preview_gate` (load_skills mangles it to
# `skill_draft_preview_gate`) or run standalone.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

_log = logging.getLogger(__name__)

# Best-effort imports — the skill registers cleanly even if one of the
# underlying gates is broken, and ``draft_preview_gate_status`` will
# report the broken piece for the user.
try:
    from core import draft_confirm as _draft_confirm_mod
except Exception as _e:  # pragma: no cover - import-time fail
    _draft_confirm_mod = None  # type: ignore[assignment]
    _log.warning("[draft_preview_gate skill] core.draft_confirm unavailable: %s", _e)

try:
    from core import draft_preview_gate as _core_preview_mod
except Exception as _e:  # pragma: no cover - import-time fail
    _core_preview_mod = None  # type: ignore[assignment]
    _log.warning("[draft_preview_gate skill] core.draft_preview_gate unavailable: %s", _e)


# Channel → spoken-name map used in the "I have a reply drafted for X, sir"
# prompt. Channel keys are matched case-insensitively; unknown channels
# fall through to the generic "I have a draft" phrasing.
_CHANNEL_LABELS: dict[str, str] = {
    "teams":    "Teams",
    "email":    "email",
    "sms":      "text",
    "phone":    "your phone",
    "telegram": "Telegram",
    "ntfy":     "ntfy",
    "pushover": "Pushover",
    "slack":    "Slack",
    "discord":  "Discord",
}


def _companion():
    """Lazy import of bobert_companion. The skill loads BEFORE companion
    finishes setting up its module globals, so any sleep/standby check
    has to resolve at call-time, not import-time."""
    try:
        return importlib.import_module("bobert_companion")
    except Exception:
        return None


def _is_asleep_or_standby() -> bool:
    """True if JARVIS is currently asleep or in wake-work standby.

    Returns False on any lookup error — fail-open here means "treat the
    user as awake and prompt them", which is the safer default than
    silently sending. If companion globals aren't loaded yet we also
    return False so an early send during boot still routes through the
    normal confirm path."""
    bc = _companion()
    if bc is None:
        return False
    try:
        sleep_flag    = getattr(bc, "_sleep_mode",   None)
        standby_flag  = getattr(bc, "_standby_mode", None)
        if isinstance(sleep_flag,   list) and sleep_flag   and sleep_flag[0]:
            return True
        if isinstance(standby_flag, list) and standby_flag and standby_flag[0]:
            return True
    except Exception:
        return False
    return False


def _resolve_recipient(channel: str, draft: dict) -> str:
    """Pick the spoken-recipient phrase for the confirmation prompt.

    Order of preference (per channel-aware UX):
      1. explicit ``draft['recipient']``  (caller knows the human name)
      2. ``draft['to']``                  (mirrors email / Teams payloads)
      3. ``_CHANNEL_LABELS[channel]``     (generic, e.g. 'Teams', 'email')
      4. the literal channel string       (catch-all so prompts stay
                                          informative for unknown channels)
    """
    for key in ("recipient", "to"):
        val = (draft.get(key) or "").strip()
        if val:
            return val
    label = _CHANNEL_LABELS.get((channel or "").strip().lower())
    if label:
        return label
    return (channel or "").strip() or "the recipient"


def _draft_body(draft: dict) -> str:
    """Best-effort extraction of the spoken body from common draft shapes.

    Skills already use a mix of {'body': ...}, {'message': ...}, and
    {'text': ...} so we accept all three. Whichever key is set wins;
    if more than one is set, body > message > text (matches the existing
    email_triage / vip_intercept convention so users hear what those
    skills queued)."""
    for key in ("body", "message", "text"):
        val = draft.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def gate_outbound_message(channel: str, draft: dict | str) -> bool:
    """Route an outbound message through the unified preview gate.

    Returns True iff the user explicitly confirmed within the
    ``core.draft_confirm.CONFIRM_TIMEOUT_S`` window. Every other outcome
    (cancel, timeout, ambiguous reply, mic / TTS / whisper failure,
    sleep/standby) returns False so the caller can decide whether to
    queue, drop, or surface a failure of its own.

    Args:
      channel: short channel identifier — 'teams', 'email', 'sms',
               'phone', 'telegram', etc. Used to pick the spoken
               recipient phrase when ``draft`` doesn't carry an explicit
               recipient. Unknown channels fall back to a generic prompt.
      draft:   either a string (the body) or a dict with at least one of
                 'body' / 'message' / 'text' for the spoken body, and
                 optionally 'recipient' / 'to' for the named recipient.

    Use this from skills whose outbound path bypasses the send_* action
    contract — e.g. teams_nudge, phone_bridge, future Slack / Discord
    bridges. Skills that DO use send_* don't need to call this directly;
    the dispatcher already wraps them via core.draft_preview_gate.

    Fail-closed: if the core gate isn't loaded (broken import, missing
    dependency) the function returns False — the caller's send should
    NOT proceed in that case. Callers that absolutely must fire (system
    alerts, hardware faults) should not call gate_outbound_message at
    all; they should call their backend directly with whatever
    confirm=False / bypass parameter that backend exposes.
    """
    if isinstance(draft, str):
        draft_dict: dict[str, Any] = {"body": draft}
    elif isinstance(draft, dict):
        draft_dict = draft
    else:
        return False

    body = _draft_body(draft_dict)
    if not body:
        # No body to read aloud — refuse rather than silently approving.
        return False

    if _is_asleep_or_standby():
        # Regression guard: never voice-prompt while the user is asleep
        # or has explicitly said "go on standby". Otherwise the user
        # could mumble a sleep-talk "yes" and accidentally send.
        _log.info("[draft_preview_gate skill] sleep/standby active — "
                  "refusing %s draft (body=%r)", channel, body[:40])
        return False

    if _draft_confirm_mod is None:
        # Core gate failed to import. Fail closed.
        _log.warning("[draft_preview_gate skill] core.draft_confirm "
                     "unavailable — refusing %s send", channel)
        return False

    recipient = _resolve_recipient(channel, draft_dict)
    try:
        return bool(_draft_confirm_mod.draft_confirm(body, recipient=recipient))
    except Exception as e:
        _log.warning("[draft_preview_gate skill] draft_confirm raised: %s", e)
        return False


def draft_preview_gate_status(_: str = "") -> str:
    """Voice action: report whether the outbound-message gate is healthy.

    Surfaces:
      • whether the two core modules loaded
      • the confirm timeout currently in force
      • whether JARVIS is in a state that would auto-refuse prompts
        (sleep / standby)
    """
    bits: list[str] = []

    if _core_preview_mod is None:
        bits.append("core preview gate: NOT loaded")
    else:
        bits.append("core preview gate: ok")

    if _draft_confirm_mod is None:
        bits.append("core draft confirm: NOT loaded")
    else:
        timeout = getattr(_draft_confirm_mod, "CONFIRM_TIMEOUT_S", "?")
        bits.append(f"core draft confirm: ok, {timeout}s window")

    if _is_asleep_or_standby():
        bits.append("currently asleep / standby — prompts auto-refused")
    else:
        bits.append("awake — prompts will fire normally")

    return "Outbound draft gate — " + "; ".join(bits) + ", sir."


def register(actions: dict) -> None:
    """Wire the skill into the action registry.

    The skill only registers diagnostics: the user-facing gating already
    happens through the dispatcher's core/draft_preview_gate middleware
    (send_* actions) and through core/draft_confirm called imperatively
    by phone_bridge / teams_nudge. ``gate_outbound_message`` is a
    module-level helper, not an action, so skills import it directly:

        from skill_draft_preview_gate import gate_outbound_message
    """
    actions["draft_preview_gate_status"] = draft_preview_gate_status
    actions["outbound_gate_status"]      = draft_preview_gate_status
