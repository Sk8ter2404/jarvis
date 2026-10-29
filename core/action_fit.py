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
the action fits, else a short reason. Six rules only:

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
  4. A face look (recognize_face and its aliases) when the owner's words do
     not ask who someone is, to recognise / see / look at someone, or
     whether anyone is there. Live 2026-10-27 20:38:11, right after several
     desk-robot turns, "He does not sound happy." (about the robot) ran
     recognize_face and "That's you, sir." was spoken. The handler
     (skills/face_id.whoami) applies the same test live, whatever the mode.
  5. The same informative reading (a status action, or a read-only fact
     such as version_info / get_time / weather_briefing) with the same
     argument as one that ran on the previous turn, when the owner's line
     is a REMARK: no question word, no "?", no command or re-check word, no
     first-person plan. Live v3.0.0 20:28:45, "Yay, version 3 is done."
     ran version_info again 17 s after it had answered. "Check again" and
     "what version again?" still run. The monolith's preemptive-injection
     path applies the same remark test (_preempt_recent_reading, 60 s).

  6. The owner's robot made to speak (a device speak action, named
     "<...>bot_speak" by its private skill) when the owner's words ask
     nobody to say, speak, talk, tell, greet ... anything. Live 2026-10-28
     17:26:20, "We need to give the robot more voice lines." (a developer
     to-do, said with the robot's name) ran the robot's speak action with
     "I am ready for more dialogue." and the robot said it. "Make him
     talk", "have the bot say hi" and "tell the robot I aced my exam" fit.

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


# ── Rule 5: the same reading again, for a remark about it ──────────────────
# Live v3.0.0 20:28:45: JARVIS had just answered "what version are you
# running?" (version_info), the owner said "Yay, version 3 is done." and
# version_info ran AGAIN, so he heard the version twice. A celebration or a
# comment about something JARVIS just reported is answered, not re-run: an
# INFORMATIVE action (a status action, or one of these read-only facts) with
# the same argument that ran on the previous turn is skipped when the
# owner's line carries no question and no command word. "Check again",
# "what version again?" and "is it still 3?" all ask, so they still run.
INFO_ACTIONS = frozenset({
    "version_info", "what_version", "when_updated", "get_time",
    "weather_briefing", "weather_forecast", "status_report",
    "check_system", "battery_status", "uptime",
})
# A question: a question word opening the line or a clause ("Yay, what
# version is that" with no "?" from the recogniser), or a trailing "?".
_QUESTION_OPEN_RE = re.compile(
    r"(?:^|[,.;:!]\s*)(?:what|who|whom|whose|how|why|when|where|which|is|are|was|were|am|"
    r"do|does|did|can|could|will|would|should|shall|have|has|had)\b", re.I)
# A command word anywhere: the owner asks for the reading (or anything else).
_COMMAND_WORD_RE = re.compile(
    r"\b(?:check|tell|show|give|run|get|read|say|report|look|see|find|"
    r"open|turn|set|play|update|refresh|repeat|confirm|verify|remind|"
    r"again|please|once more)\b", re.I)
# The owner telling JARVIS about himself or his plans ("I have to leave for
# class now, it starts at eleven-thirty" -> get_time fits): not a remark
# about the reading, so the reading may run.
_SELF_PLAN_RE = re.compile(
    r"\b(?:i|i'?m|i'?ve|i'?ll|i'?d|we|we'?re|let'?s|gotta|gonna|need|"
    r"have to|got to|going to|about to)\b", re.I)


def is_informative_action(name: str) -> bool:
    """True for an action that only reads a fact back (rule 5)."""
    nm = str(name or "").strip().lower()
    return bool(nm) and (nm in INFO_ACTIONS or is_status_action(nm))


def is_remark(user_text) -> bool:
    """True when the owner's words neither ask nor command: a celebration
    or a comment ("Yay, version 3 is done.", "Nice, that's quick."). Any
    question word opening the line, a trailing "?", a re-check word or a
    command word makes it a request instead."""
    body = _strip_opener(user_text)
    if not body:
        return False
    if _asks_fresh_reading(body) or "?" in body:
        return False
    if (_QUESTION_OPEN_RE.search(body) or _COMMAND_WORD_RE.search(body)
            or _SELF_PLAN_RE.search(body)):
        return False
    return True


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
# The plan's words only (no / that's not / wrong / I said), plus what c032
# needs: "though" closing a clause ("the headset is on though."), the owner
# contradicting the answer. Bare still / already / actually are NOT here:
# "is the download still running?" or "actually, what time is it?" ask for
# a fresh reading, they correct nothing.
_CORRECTION_RE = re.compile(
    r"^(?:no|nope|nah|negative)\b"
    r"|\bno,"
    r"|\b(?:that'?s|that is|thats|that was)\s+not\b"
    r"|\bwrong\b|\bi said\b"
    r"|\bthough\s*(?:[.!;,]|$)", re.I)
# Words that ask for a fresh reading: never rule 3, whatever else he said.
_RECHECK_RE = re.compile(
    r"\bagain\b|\bre-?check|\brefresh|\bre-?run\b|\bone more time\b", re.I)


# ── Rule 4: a face look the owner's words do not ask for ────────────────────
# recognize_face is a LIVE CAMERA LOOK (the always-on local guard lists it
# beside "who's here"), so the brain reached for it on a remark about a
# third person's mood: 2026-10-27 "He does not sound happy." -> "That's you,
# sir.". It fits only when the owner asks WHO (who / whom / whose), to
# recognise / identify / see / look at someone or a face, or whether anyone
# is there / in frame. skills/face_id.whoami calls asks_face_look too.
FACE_LOOK_ACTIONS = frozenset({
    "recognize_face", "whoami", "who_am_i", "do_you_recognize_me",
    "whos_at_the_desk",
})
_FACE_LOOK_RE = re.compile(
    r"\bwho(?:m|se|'?s|'?re)?\b|\brecogni[sz]\w*|\bidentif\w*|\bfaces?\b"
    r"|\b(?:see|seeing|sees|look at|looking at|spot|know)\s+(?:me|us|him|"
    r"her|them|anyone|anybody|someone|somebody)\b"
    r"|\b(?:anyone|anybody|someone|somebody)\b"
    r"|\b(?:it'?s|that'?s|is it|is that|is this) me\b"
    r"|\bin (?:the )?frame\b|\bin front of\b|\bcamera\b|\bwebcam\b"
    # "is my brother here?", "is she at the desk?", and a re-ask right after
    # a face answer ("try again", "check again").
    r"|\b(?:is|are)\s+\w+(?:\s+\w+)?\s+(?:here|there|in the room|"
    r"at (?:the|my) desk)\b|\bagain\b|\bre-?check\b", re.I)


# ── Rule 6: the robot made to speak, unasked ────────────────────────────────
# The owner's robot skill lives in a private folder outside the repo, so
# public code never names the device: a robot speak action is recognised by
# its NAME SHAPE, any action ending "bot_speak" (one word or with an
# underscore before "bot", a doubled final t allowed: "deskbot_speak",
# "desk_bot_speak", "robot_speak"). A name pattern, not a registry, because
# the private skill already registers its actions under that shape and needs
# no edit for this rule; the robot's other actions (banter, engage, halt ...)
# never match. It puts words in the robot's mouth, so it fits only a turn
# that asks someone to SAY something: a speech verb or greeting anywhere
# ("make him talk", "have the bot say hi", "tell the robot I aced my exam",
# "let him introduce himself"). Broad on purpose - the rule flags only a
# turn with NO speech word at all, like the live developer to-do "We need to
# give the robot more voice lines." ("voice lines" is a feature, not a
# request to speak).
_ROBOT_SPEAK_ACTION_RE = re.compile(r"^[a-z0-9_]*bott?_speak$", re.I)


def is_robot_speak_action(name) -> bool:
    """True for a device speak action of the owner's robot (rule 6): an
    action name ending "bot_speak". Never raises."""
    try:
        return (_ROBOT_SPEAK_ACTION_RE.match(str(name or "").strip())
                is not None)
    except Exception:
        return False


_SPEECH_ASK_RE = re.compile(
    r"\b(?:say|says|saying|said|speak|speaks|speaking|talk|talks|talking|"
    r"tell|tells|telling|announce|repeat|greet|greeting|introduce|sing|"
    r"shout|yell|whisper|chat|chatter|ask|asks|read|recite|pronounce|"
    r"hello|hi|hey|goodbye|bye|good\s+(?:morning|night|evening|afternoon)|"
    r"thank|thanks|welcome|congratulate|congrats|joke|roast|compliment|"
    r"insult|respond|reply|answer|words?|quote)\b", re.I)


def asks_robot_speech(user_text) -> bool:
    """True when the owner's words ask someone to say something: any speech
    verb or greeting (rule 6). "We need to give the robot more voice lines."
    is not. Never raises."""
    try:
        return _SPEECH_ASK_RE.search(_norm(user_text)) is not None
    except Exception:
        return False


def asks_face_look(user_text) -> bool:
    """True when the owner's words ask for a face look: who someone is, to
    recognise / see / look at a person or face, or whether anyone is there.
    "He does not sound happy." and "she seems upset" are not. Never
    raises."""
    try:
        return _FACE_LOOK_RE.search(_norm(user_text)) is not None
    except Exception:
        return False


def _asks_fresh_reading(text: str) -> bool:
    """True when the owner asks for the reading again: a re-check word, or
    the turn ends on a question ("is it still up?", "is it working now?")."""
    body = text.strip().rstrip(" .!\"'")
    return bool(_RECHECK_RE.search(body)) or body.endswith("?")


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
    """True when the owner's words correct what JARVIS just did (and do not
    ask for a fresh reading: "no, check again" is a re-check)."""
    body = _strip_opener(user_text)
    if _asks_fresh_reading(body):
        return False
    return _CORRECTION_RE.search(body) is not None


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
        # Rule 4: a face look when sir asked nobody's name or face.
        if nm in FACE_LOOK_ACTIONS and not asks_face_look(text):
            return (f"not run: {nm} looks through the webcam to say who is "
                    f"there, and sir asked nothing about who someone is or "
                    f"a face")
        # Rule 6: the robot made to speak when nobody was asked to say
        # anything.
        if is_robot_speak_action(nm) and not asks_robot_speech(text):
            return (f"not run: {nm} makes the desk robot say a line, and "
                    f"sir asked nobody to say anything")
        # Rule 5: the same reading again, for a remark about it.
        if is_informative_action(nm) and is_remark(text):
            a = _norm(arg).lower()
            if (nm, a) in set(_pairs(prior_actions)):
                shown = f"{nm}({_norm(arg)})" if a else nm
                return (f"not run: {shown} ran last turn and sir only "
                        f"remarked on it; running it again only repeats "
                        f"that answer")
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

