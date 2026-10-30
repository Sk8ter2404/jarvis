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
the action fits, else a short reason. Seven rules only:

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

     Since v3.0.9 (ACTION_FIT_ROBOT_SPEAK_ENFORCE) a speech word is not
     enough: the owner's words must make the ROBOT the speaker - a robot
     word (core/robot_words) or a third-person form ("make him ...", "have
     it say", "let him ...", "get him to ...", "tell him to ...", "can he
     say", "he should say"). Live 2026-10-29 18:54:35 and 18:55:16 "Jarvis,
     say <line>." and 18:55:35 "I just want to say it, sir. Or if it's say
     <line>." (the recogniser's "I just want you to say it") each ran the
     robot's speak action, so the robot said it, not JARVIS. "Say X", "can
     you say X" and "I want you to say X" ask JARVIS himself.
  7. The same action and argument again (v3.0.9, ACTION_FIT_ACK_REPEAT_
     ENFORCE) when the owner's WHOLE line is a bare acknowledgement or
     decline ("All right.", "No, not right now.", "okay, thanks") and the
     turn did not accept an open offer. Live 2026-10-29 18:55:26 "All
     right." ran the robot's speak action again with the same line. "Yes",
     "sure", "again" and any line with a request in it are never bare.

``check_rule`` returns (rule number, reason) - (0, None) when the action
fits - so the monolith can enforce rules 6 and 7 while the rest stay in
shadow; ``check`` is the same test returning only the reason.

No owner words (a proactive dispatch) never flags. Never raises: any fault
reads as "fits".

Wired in bobert_companion.parse_and_run_actions._runner AFTER the W04 alias
step, the autocorrect, the self-termination hold, the once-per-turn rule,
confirmation and pushback (so it never touches any of them), right before
the handler runs. Behind ``ACTION_FIT_MODE`` (core/config.py): 'off' = the
old path; 'shadow' (default) logs "[action-fit] would block ..." and runs
the action; 'on' does not run it and returns a failure result (``refused
(action-fit): not run - ...``) so the follow-up round chooses again.
Rules 6 and 7 are enforced as in 'on' even while the mode is 'shadow' when
ACTION_FIT_ROBOT_SPEAK_ENFORCE / ACTION_FIT_ACK_REPEAT_ENFORCE are on (the
v3.0.9 defaults); 'off' still turns every rule off.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

from core.robot_words import ROBOT_WORD_STRICT

MODES = ("off", "shadow", "on")

# The rule numbers check_rule returns (0 = the action fits).
RULE_STATUS_FOR_COMMAND = 1
RULE_MODE_SWAP = 2
RULE_RECHECK_AFTER_CORRECTION = 3
RULE_FACE_LOOK = 4
RULE_REMARK_RERUN = 5
RULE_ROBOT_SPEECH = 6
RULE_ACK_REPEAT = 7

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


# v3.0.9 (ACTION_FIT_ROBOT_SPEAK_ENFORCE): a speech word is not enough -
# "Jarvis, say <line>." asks JARVIS to say it (live 2026-10-29 18:54:35 and
# 18:55:16, and 18:55:35 "I just want to say it, sir. Or if it's say
# <line>.", the recogniser's "I just want you to say it"); each ran the
# robot's speak action and the owner asked "They sent it over to the
# robot?". The robot speaks only when his words make the ROBOT the speaker:
#   * a robot word (core/robot_words.ROBOT_WORD_STRICT, the developer-to-do
#     pattern: "robot", "bot", "<word> bot", a one-word "<word>bot" that is
#     not a common word);
#   * or a third-person speaker form: make / have / let / get him / her /
#     it / them (or "the <name>") ... <speech verb> ("make him talk", "have
#     it say hello", "let him introduce himself", "get him to tell a
#     joke"), "tell him / it to ...", "can he / it say ...", "he / it
#     should say ...".
# "say X", "can you say X", "you say it" and "I want you to say X" name no
# robot and no third person: JARVIS says those words himself.
_ROBOT_NAMED_RE = re.compile(r"\b" + ROBOT_WORD_STRICT, re.I)
_SPEECH_VERB = (r"(?:say|says|speak|talk|tell|announce|repeat|greet|"
                r"introduce|sing|shout|yell|whisper|chat|ask|read|recite|"
                r"pronounce|roast|compliment|insult|respond|reply|answer|"
                r"quote|thank|congratulate|welcome)")
_THIRD_PRON = r"(?:him|her|it|them)"
_THIRD_SUBJ = r"(?:he|she|it|they)"
_NOT_A_NAME = (r"(?:you|me|us|yourself|myself|ourselves|jarvis|sir|a|an|the|"
               r"my|our|your|his|its|their|sure|it|him|her|them|this|that|"
               r"these|those|some|any|everyone|everybody|someone|somebody|"
               r"people|and|or|then|so|to|i|we|fun|time)")
_THIRD_SPEAKER_RES = (
    # make him talk / have it say hello / let him introduce himself / get
    # him to tell a joke / make the <name> talk
    re.compile(r"\b(?:make|makes|making|have|has|having|let|lets|letting|"
               r"get|gets|getting)\s+(?:" + _THIRD_PRON + r"|(?:the|my|our|"
               r"your)\s+[\w'-]+(?:\s+[\w'-]+)?)\s+(?:to\s+)?"
               r"(?:(?:just|also|please|now|quickly)\s+)?" + _SPEECH_VERB
               + r"\b", re.I),
    # tell him to ... / tell it to ...
    re.compile(r"\btell\s+" + _THIRD_PRON + r"\s+to\b", re.I),
    # can he say ... / could it say ... / will she greet ...
    re.compile(r"\b(?:can|could|will|would|should|does|did)\s+"
               + _THIRD_SUBJ + r"\s+(?:(?:just|also|please)\s+)?"
               + _SPEECH_VERB + r"\b", re.I),
    # he should say ... / it needs to say ... / she can greet ...
    re.compile(r"\b" + _THIRD_SUBJ + r"\s+(?:should|can|could|will|would|"
               r"must|might|needs\s+to|has\s+to|ought\s+to|is\s+going\s+to|"
               r"gets\s+to)\s+(?:(?:just|also|please)\s+)?" + _SPEECH_VERB
               + r"\b", re.I),
    # ask him to say hi / I want him to say hello / need it to greet her
    re.compile(r"\b(?:ask|asked|want|wanted|need|tell|told|get|got)\s+"
               + _THIRD_PRON + r"\s+to\b", re.I),
    # let's hear him say something / see it greet her / watch him sing
    re.compile(r"\b(?:hear|see|watch)\s+" + _THIRD_PRON + r"\s+(?:to\s+)?"
               + _SPEECH_VERB + r"\b", re.I),
    # get him talking / keep it chatting / have her singing
    re.compile(r"\b(?:get|got|have|has|keep)\s+" + _THIRD_PRON
               + r"\s+(?:talking|speaking|saying|singing|chatting)\b", re.I),
    # have <name> say hi / make <two-word name> talk: a bare name the
    # recogniser spelled so it matches no robot word. Never "have you /
    # let me / make sure / have a say / have to say / let Jarvis ...":
    # those words name nobody else (_NOT_A_NAME).
    re.compile(r"\b(?:have|make|let)\s+(?!" + _NOT_A_NAME + r"\b)[a-z][\w'-]*"
               r"(?:\s+(?!" + _NOT_A_NAME + r"\b)[a-z][\w'-]*)?\s+"
               r"(?:to\s+)?" + _SPEECH_VERB + r"\b", re.I),
)
# A robot word that only names who is ADDRESSED ("say hi to the robot",
# "say goodnight to the bot") makes JARVIS the speaker, not the robot.
_ADDRESSED_ROBOT_RE = re.compile(
    r"\b(?:to|at|about)\s+(?:(?:the|my|our|your|that|this)\s+)?"
    + ROBOT_WORD_STRICT + r"$", re.I)
# Words the owner quotes are the line to say, not who says it ("say 'I am
# not a robot'"): double or curly quotes anywhere, a single quote only when
# it opens after a space (never the apostrophe in "don't").
_QUOTED_RE = re.compile(r"\"[^\"]*\"|“[^”]*”"
                        r"|(?:^|(?<=\s))'[^']+'(?=[\s.,!?;:]|$)")
# Re-run words: right after the robot spoke, "Try it again." / "do it
# again" / "one more time" ask for the same line again with no speech word
# in them (live 2026-10-29 18:57:07, the owner testing the delay).
_RERUN_RE = re.compile(
    r"\bagain\b|\bone more time\b|\bonce more\b|\bre-?run\b"
    r"|\b(?:try|do)\s+(?:it|that|this)\b|\brepeat\b|\bsame\s+thing\b",
    re.I)


def asks_robot_speaker(user_text) -> bool:
    """True when the owner's words make the ROBOT the speaker: a robot word
    or a third-person speaker form (see the block comment). "Jarvis, say
    hello.", "I want you to say it", "say hi to the robot" and a robot word
    inside the quoted line are not. Never raises."""
    try:
        text = _norm(user_text)
        if not text:
            return False
        named = _QUOTED_RE.sub(" ", text)
        for m in _ROBOT_NAMED_RE.finditer(named):
            if not _ADDRESSED_ROBOT_RE.search(named[:m.end()]):
                return True
        return any(rx.search(text) for rx in _THIRD_SPEAKER_RES)
    except Exception:
        return False


def asks_rerun(user_text) -> bool:
    """True when the owner's words ask for the same thing again ("Try it
    again.", "do it again", "one more time"). Never raises."""
    try:
        return _RERUN_RE.search(_norm(user_text)) is not None
    except Exception:
        return False


# ── Rule 7: the same action again for a bare acknowledgement / decline ─────
# Live 2026-10-29 18:55:26 "All right." ran the robot's speak action again
# with the line it had just said (rule 6 only logged it, in shadow), and
# 18:53:47 "No, not right now." (a decline of JARVIS's offer) ended in the
# previous turn's pause_music twice. An acknowledgement or decline asks for
# nothing: an action whose name and argument ran on the previous turn is not
# run again unless the turn accepted an open offer (the monolith passes
# accepted_offer). The WHOLE line must be acknowledgement / decline words,
# with "Jarvis" / "sir" / "thanks" around them allowed. "Yes", "yeah",
# "sure", "please", "go ahead", "do it", "again" are never bare (they ask),
# and neither is any line with a request in it ("okay, play it").
# core/claim_validator's ack-claim drop (ACK_CLAIM_NO_FOLLOWUP) asks the
# same predicate (the monolith passes it this one).
_BARE_ACK_WORDS = (
    r"all\s*right|alright|ok(?:ay)?|cool|nice|great|good|fine|got\s+it|"
    r"thanks|thank\s+you|no|nope|nah|not\s+now|not\s+right\s+now|"
    r"not\s+yet|maybe\s+later|never\s*mind|forget\s+it|no\s+thanks|"
    r"no\s+thank\s+you|that'?s\s+fine|that\s+is\s+fine|i'?m\s+good|"
    r"i\s+am\s+good|all\s+good|perfect|understood")
_BARE_ACK_EXTRA = r"jarvis|sir|thanks|thank\s+you"
_BARE_ACK_RE = re.compile(
    r"^(?:(?:" + _BARE_ACK_WORDS + r"|" + _BARE_ACK_EXTRA + r")"
    r"(?:[\s,.!;:—–-]+|$))+$", re.I)
_BARE_ACK_WORD_RE = re.compile(r"\b(?:" + _BARE_ACK_WORDS + r")\b", re.I)


def is_bare_ack(user_text) -> bool:
    """True when the owner's WHOLE line is a bare acknowledgement or decline
    ("All right.", "No, not right now.", "okay sir", "thanks, Jarvis"), with
    no request and no question in it. "Yes", "sure", "again" and "okay play
    it" are not. Never raises."""
    try:
        text = _norm(user_text).lower().replace("’", "'")
        text = text.strip(" .!,;:")
        if not text or "?" in text:
            return False
        return (_BARE_ACK_RE.match(text) is not None
                and _BARE_ACK_WORD_RE.search(text) is not None)
    except Exception:
        return False


# While JARVIS's offer was OPEN when the turn began ("Shall I try again?"),
# an affirmative acknowledgement ("All right.", "great", "got it", "that's
# fine") may be the owner accepting it - core.yes_no.classify_reply reads
# only "alright" as a yes, so the recogniser's "All right." was "other" and
# rule 7 blocked the retry he had just agreed to. Then only a DECLINE is
# bare: these words, with the acknowledgement / "Jarvis" / "sir" words
# around them allowed ("okay, no thanks").
_BARE_DECLINE_WORD_RE = re.compile(
    r"\b(?:no|nope|nah|not\s+now|not\s+right\s+now|not\s+yet|maybe\s+later|"
    r"never\s*mind|forget\s+it|no\s+thanks|no\s+thank\s+you|i'?m\s+good|"
    r"i\s+am\s+good)\b", re.I)


def is_bare_decline(user_text) -> bool:
    """True when the owner's WHOLE line is a bare acknowledgement or decline
    (is_bare_ack) with a decline word in it ("No, not right now.", "never
    mind", "okay, no thanks"). "All right.", "great", "got it" are not.
    Never raises."""
    try:
        if not is_bare_ack(user_text):
            return False
        text = _norm(user_text).lower().replace("’", "'")
        return _BARE_DECLINE_WORD_RE.search(text) is not None
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
    when another action in the reply does the job). The reason of
    check_rule with its defaults. Never raises."""
    return check_rule(user_text, name, arg, prior_actions, reply_actions)[1]


def check_rule(user_text, name, arg="",
               prior_actions: Iterable = (),
               reply_actions: Sequence = (), *,
               robot_speaker: bool = True,
               ack_repeat: bool = True,
               accepted_offer: bool = False,
               offer_open: bool = False,
               offer_text: str = "") -> tuple:
    """(rule number, reason) when ``name(arg)`` misfits the owner's words,
    (0, None) when it fits - see the module docstring for the rules and
    check for ``prior_actions`` / ``reply_actions``.

    ``robot_speaker``: rule 6 also needs the robot as the speaker (v3.0.9);
    False = the v3.0.8 rule 6 (any speech word fits). ``ack_repeat``: rule 7
    is on; False = no rule 7. ``accepted_offer``: this turn accepted JARVIS's
    open offer (rule 7 never applies then). ``offer_open``: JARVIS's offer
    was open when the turn began and this turn did not accept it as a clear
    yes - rule 7 then needs a DECLINE (is_bare_decline), since "All right."
    may accept it. ``offer_text``: the offer this turn TOOK (a clear yes, or
    an affirmative acknowledgement while it was open), else "" - with
    ``robot_speaker`` on, an offer that makes the robot the speaker ("Shall
    I have the robot say it?") lets rule 6 fit. Never raises."""
    try:
        text = _norm(user_text)
        nm = str(name or "").strip().lower()
        if not text or not nm:
            return 0, None
        # Rule 7: the same action again for a bare acknowledgement or
        # decline (and no accepted offer; only a decline while an offer
        # was open).
        bare = is_bare_decline(text) if offer_open else is_bare_ack(text)
        if ack_repeat and not accepted_offer and bare:
            a = _norm(arg).lower()
            if (nm, a) in set(_pairs(prior_actions)):
                shown = f"{nm}({_norm(arg)})" if a else nm
                return RULE_ACK_REPEAT, (
                    f"not run: {shown} already ran a moment ago and sir only "
                    f"acknowledged or declined (\"{text}\"); running it "
                    f"again repeats it - answer him briefly instead")
        # Rule 2: a heavy mode swap with no word that asks for one.
        if nm in HEAVY_MODE_ACTIONS and not _MODE_WORD_RE.search(text):
            return RULE_MODE_SWAP, (
                f"not run: {nm} swaps the brain and the cameras (game "
                f"mode), and sir said nothing about a game, a mode or "
                f"power")
        # Rule 4: a face look when sir asked nobody's name or face.
        if nm in FACE_LOOK_ACTIONS and not asks_face_look(text):
            return RULE_FACE_LOOK, (
                f"not run: {nm} looks through the webcam to say who is "
                f"there, and sir asked nothing about who someone is or "
                f"a face")
        # Rule 6: the robot made to speak when nobody was asked to say
        # anything - or (v3.0.9) when sir asked JARVIS himself to say it.
        if is_robot_speak_action(nm):
            # (v3.0.9) "Try it again." right after the robot spoke asks for
            # its line again: it fits, with or without a speech word.
            if (robot_speaker and asks_rerun(text)
                    and any(is_robot_speak_action(p)
                            for p, _g in _pairs(prior_actions))):
                return 0, None
            # (v3.0.9 final review) "Yes." / "All right." to JARVIS's own
            # offer "Shall I have the robot say it?" asks for exactly that:
            # the offer he took is his words this turn. The monolith passes
            # ``offer_text`` only when the turn took the offer.
            ot = _norm(offer_text)
            if (robot_speaker and ot and asks_robot_speech(ot)
                    and asks_robot_speaker(ot)):
                return 0, None
            if not asks_robot_speech(text):
                return RULE_ROBOT_SPEECH, (
                    f"not run: {nm} makes the desk robot say a line, and "
                    f"sir asked nobody to say anything")
            if robot_speaker and not asks_robot_speaker(text):
                return RULE_ROBOT_SPEECH, (
                    f"not run: {nm} makes the desk robot say a line, but "
                    f"sir asked you (JARVIS) to say it and named no robot, "
                    f"so say the words yourself in your reply")
        # Rule 5: the same reading again, for a remark about it.
        if is_informative_action(nm) and is_remark(text):
            a = _norm(arg).lower()
            if (nm, a) in set(_pairs(prior_actions)):
                shown = f"{nm}({_norm(arg)})" if a else nm
                return RULE_REMARK_RERUN, (
                    f"not run: {shown} ran last turn and sir only "
                    f"remarked on it; running it again only repeats "
                    f"that answer")
        if not is_status_action(nm):
            return 0, None
        # Rule 1: a list / status action for a command that changes things.
        cmd = owner_command(text)
        if cmd:
            others = [str(r or "").strip().lower() for r in reply_actions
                      if str(r or "").strip().lower() != nm]
            if not any(o and not is_status_action(o) for o in others):
                return RULE_STATUS_FOR_COMMAND, (
                    f"not run: sir asked to {cmd}; {nm} only lists or "
                    f"reports, it changes nothing")
        # Rule 3: the same read-only check again, right after a correction.
        if is_correction(text):
            a = _norm(arg).lower()
            if (nm, a) in set(_pairs(prior_actions)):
                shown = f"{nm}({_norm(arg)})" if a else nm
                return RULE_RECHECK_AFTER_CORRECTION, (
                    f"not run: {shown} ran last turn and sir corrected "
                    f"it; running it again only repeats that answer")
        return 0, None
    except Exception:
        return 0, None


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

