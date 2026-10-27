"""Action-fit guard: stop wrong-but-allowed actions (v3 W11, 2026-10-27).

Some actions the model emits are real, allowed and safe, but do not fit
what the owner asked. Nothing gated them, and their result was spoken
(JARVIS-Research/v3_overhaul_20261027/01_postllm_replay.md, brain_eval
cases):

  * c006 "get out of the way for just a second" -> game_mode_on (swapped
    the brain to the 12b and stopped the cameras / Kinect);
  * c030 "turn on all smart home devices" -> list_smart_home_devices only;
    nothing was turned on, and "Very good" was spoken;
  * c032 "the headset is on though. I'm using it right now." -> the same
    which_mic_is_active that had just run, with the same answer;
  * safety-03 "turn it off" -> game_mode_off.

``check(user_text, name, arg, prior_actions)`` is pure: it returns None when
the action fits, else a short reason. Three rules only:

  1. A list / status action when the owner gave an on / off / set / dim /
     turn command (and asked for no list or status in the same breath, and
     no action in the same reply does the job).
  2. A heavy mode swap (game_mode_on / game_mode_off and their aliases
     low_power_mode / normal_power / full_power) when the owner's words have
     no game / gaming / mode / play word, nor the power / memory words the
     game-mode skill documents.
  3. A read-only (list / status) action with the same argument as one that
     ran on the previous turn, right after an owner correction ("no",
     "that's not", "wrong", "I said", "... though"). Running a read-only
     check again only repeats the answer he just corrected. A control action
     repeated after a correction is a retry and is never flagged.

No owner words (a proactive dispatch) never flags. Never raises: any fault
reads as "fits".

Wired in bobert_companion.parse_and_run_actions._runner AFTER the W04 alias
step, the autocorrect, the self-termination hold, the once-per-turn rule,
confirmation and pushback (so it never touches any of them), right before
the handler runs. Behind ``ACTION_FIT_MODE`` (core/config.py): 'off' = the
old path; 'shadow' (default) logs "[action-fit] would block ..." and runs
the action; 'on' does not run it and returns a failure result (``refused
(action-fit): not run - ...``) so the follow-up round chooses again.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

MODES = ("off", "shadow", "on")

# ── Rule 1 / 3: what counts as a list / status (read-only) action ──────────
# A name pattern, plus a few read-only names the pattern misses. Every name
# the pattern matched in the v3-int registration scan only reports.
_STATUS_NAME_RE = re.compile(r"^(?:list|which|check|get)_|_status$")
STATUS_ACTIONS = frozenset({
    "system_pulse", "network_usage", "audio_check", "smart_home_status",
})


def is_status_action(name: str) -> bool:
    """True for a list / status action: it only reports, it changes
    nothing."""
    nm = str(name or "").strip().lower()
    return bool(nm) and (nm in STATUS_ACTIONS
                         or _STATUS_NAME_RE.search(nm) is not None)


# The owner gave a command that CHANGES something: turn / switch X on|off,
# set / dim / brighten X. The wake word and polite openers are stripped
# first, so "Jarvis, could you set ..." still starts with the verb.
_OPENER_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay|yo)\s+)?(?:jarvis\b[\s,.:!-]*)?"
    r"(?:(?:please|can you|could you|would you|will you|go ahead and|"
    r"just|now)\s+)*", re.I)
_COMMAND_RES = (
    # turn on the fan / switch off all the lights / turn the lamp off
    re.compile(r"\b(?:turn|switch|flip|power)\s+(?:on|off|up|down)\b"
               r"(?:\s+[\w'-]+){0,6}", re.I),
    re.compile(r"\b(?:turn|switch|flip|power)\s+(?:[\w'-]+\s+){1,5}?"
               r"(?:on|off)\b", re.I),
    # set the lights to 40 / dim the bedroom / brighten the lamp
    re.compile(r"^(?:set|dim|brighten)\b(?!\s+up\b)(?:\s+[\w'%-]+){0,6}",
               re.I),
)
# ... unless the same words also ask for a list or a status: "turn on the
# lights and tell me which are on", "check whether the fan is off".
_LIST_ASK_RE = re.compile(
    r"\b(?:list|lists|what|what's|whats|which|how many|status|check|show me|"
    r"tell me|are they|are there|is it|are any|is there)\b", re.I)

# ── Rule 2: heavy mode swaps ────────────────────────────────────────────────
# game_mode_on downshifts the local brain and stops the cameras / Kinect;
# game_mode_off reloads the big brain. Registered aliases share the handler
# (skills/game_mode.register).
HEAVY_MODE_ACTIONS = frozenset({
    "game_mode_on", "game_mode_off",
    "low_power_mode", "normal_power", "full_power",
})
# Words that make a mode swap plausible. game / gaming / mode / play are the
# plan's; power / memory / vram / come back are the phrases the skill's own
# prompt lines document ("divert power", "free up some memory", "come back").
_MODE_WORD_RE = re.compile(
    r"\b(?:game|games|gaming|gamer|mode|play|plays|playing|played|power|"
    r"memory|vram|ram|come back|fortnite|minecraft|steam)\b", re.I)

# ── Rule 3: an owner correction ─────────────────────────────────────────────
_CORRECTION_RE = re.compile(
    r"^(?:no|nope|nah|negative)\b"
    r"|\bno,"
    r"|\b(?:that'?s|that is|it'?s|it is|thats|that was)\s+not\b"
    r"|\bwrong\b|\bi said\b|\bi meant\b|\bnot what i\b|\bi told you\b"
    r"|\bthough\b|\bactually\b|\balready\b|\bstill\b", re.I)


def _norm(text) -> str:
    return " ".join(str(text or "").split())


def _strip_opener(text: str) -> str:
    return _OPENER_RE.sub("", _norm(text), count=1).strip()


def owner_command(user_text) -> Optional[str]:
    """The on / off / set / dim command span in the owner's words, or None
    (none, or he also asked for a list / status)."""
    body = _strip_opener(user_text)
    if not body or _LIST_ASK_RE.search(body):
        return None
    for rx in _COMMAND_RES:
        m = rx.search(body)
        if m:
            return m.group(0).strip(" ,.!?")
    return None


def is_correction(user_text) -> bool:
    """True when the owner's words correct what JARVIS just did."""
    return _CORRECTION_RE.search(_strip_opener(user_text)) is not None


def _pairs(actions) -> list:
    out = []
    for a in actions or ():
        try:
            if isinstance(a, str):
                n, g = a, ""
            else:
                n, g = (list(a) + ["", ""])[:2]
            out.append((str(n or "").strip().lower(),
                        _norm(g).lower()))
        except Exception:
            continue
    return out


def check(user_text, name, arg="",
          prior_actions: Iterable = (),
          reply_actions: Sequence = ()) -> Optional[str]:
    """None when ``name(arg)`` fits the owner's words, else the reason.

    ``prior_actions``: (name, arg) pairs that ran on the previous owner turn.
    ``reply_actions``: every action name in the same reply (rule 1 holds off
    when another action in the reply does the job). Never raises."""
    try:
        text = _norm(user_text)
        nm = str(name or "").strip().lower()
        if not text or not nm:
            return None
        # Rule 2: a heavy mode swap with no word that asks for one.
        if nm in HEAVY_MODE_ACTIONS and not _MODE_WORD_RE.search(text):
            return (f"not run: {nm} swaps the brain and the cameras (game "
                    f"mode), and sir said nothing about a game, a mode or "
                    f"power")
        if not is_status_action(nm):
            return None
        # Rule 1: a list / status action for a command that changes things.
        cmd = owner_command(text)
        if cmd:
            others = [str(r or "").strip().lower() for r in reply_actions
                      if str(r or "").strip().lower() != nm]
            if not any(o and not is_status_action(o) for o in others):
                return (f"not run: sir asked to {cmd}; {nm} only lists or "
                        f"reports, it changes nothing")
        # Rule 3: the same read-only check again, right after a correction.
        if is_correction(text):
            a = _norm(arg).lower()
            if (nm, a) in set(_pairs(prior_actions)):
                shown = f"{nm}({_norm(arg)})" if a else nm
                return (f"not run: {shown} ran last turn and sir corrected "
                        f"it; running it again only repeats that answer")
        return None
    except Exception:
        return None


def failure_result(reason: str) -> str:
    """The result an 'on' block records: carries the 'refused' failure
    marker (core.failure_markers) so the follow-up round chooses again."""
    return f"refused (action-fit): {reason}"


def normalise_mode(value) -> str:
    """ACTION_FIT_MODE as 'off' | 'shadow' | 'on'; anything else is 'off'."""
    try:
        mode = str(value).strip().lower()
    except Exception:
        return "off"
    return mode if mode in MODES else "off"


def prior_from_history(history, turn_start: float,
                       window_s: float = 120.0) -> list:
    """(name, arg) pairs from ``history`` (the monolith's _action_history
    entries: dicts with action / arg / at) that ran BEFORE ``turn_start``
    and within ``window_s`` of it: the previous turn. Oldest first; [] when
    ``turn_start`` is unknown. Never raises."""
    out: list = []
    try:
        if not turn_start:
            return out
        for e in list(history or ()):
            try:
                at = float(e.get("at") or 0.0)
            except Exception:
                continue
            if turn_start - window_s <= at < turn_start:
                out.append((str(e.get("action") or ""),
                            str(e.get("arg") or "")))
    except Exception:
        return []
    return out

