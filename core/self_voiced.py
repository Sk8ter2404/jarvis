"""core/self_voiced.py - what JARVIS says when a SELF-VOICED action said nothing.

WHY THIS MODULE EXISTS
======================
A self-voiced action (a device dialogue: JARVIS and a talking device in the
room take turns, core/dialogue.py) does ALL of its own talking inside the
action, so the main path speaks nothing else for it: no prose, no quip, no
verbatim result, no follow-up round.

Live 2026-10-05 00:49-00:53 (v2.0.180): five owner turns asked for a chat
with the device. The brain answered "Right away, sir. [ACTION: <chat>, ...]",
the skill refused before saying a word (its result: "... not started:
<reason>."), and JARVIS then said NOTHING: the reply's own "Right away, sir."
was dropped as self-voiced prose, and the refusal was neither news nor a
failure to report. The owner heard a long think, then silence.

The rule since then (bobert_companion._run_self_voiced): a self-voiced action
that returns without having spoken a single line on its own thread did NOT
do its own talking. Its result becomes a TERMINAL failure
(core.failure_markers.TERMINAL_FAILURE_PREFIX) carrying ONE honest line built
here, which every path voices word for word (the main turn, the routed turn,
a confirmed "yes"). The routed path and the brain path therefore say the
same thing. One exception: an owner stop (his wake word, a stop, a tray
interrupt) before the first line - he ended it, so nothing is added.

Review repairs the same day: a stop while the action waits for another chat
cancels it (STOPPED_WHILE_WAITING); an action that raised after it had spoken
did its talking (spoke_then_raised: no failure round that could run it
twice); a deferral-shaped result handed back BY the action is a silent run
like any other (only the dispatcher defers, before an action runs).

The line names a reason only when the result carries one of JARVIS's OWN
dialogue words (bobert_companion._dialogue_ready() and core.dialogue.REASONS).
A skill's private reason ("binding", "unreachable", ...) means nothing to the
owner, so it is left out rather than guessed at; a skill that wants its own
wording returns TERMINAL_FAILURE_PREFIX + its line, which JARVIS speaks as is.

Pure stdlib, no I/O. Never raises.
"""
from __future__ import annotations

import re

# JARVIS's own dialogue words (the reasons _dialogue_ready() refuses with and
# core.dialogue.REASONS a run ends with) -> the clause the owner hears.
REASON_CLAUSES: dict = {
    "active": "another chat is still going",
    "boot_grace": "I've only just started; ask me again in a minute",
    "tts_muted": "my voice is muted",
    "mic_muted": "the microphone is muted",
    "sleep": "I'm in standby",
    "realtime_voice": "the live voice call has the speaker",
    "disabled": "device chats are switched off",
    "device_lost": "the device stopped answering",
    "device_busy": "the device was busy",
    "device_muted": "the device is muted",
    "device_no_answer": "the device didn't answer",
    "device_heap": "the device is short of memory",
    "expired": "it ran out of time",
    "error": "something went wrong on my side",
}

# The owner ended it before the first line: nothing is added.
OWNER_STOP_REASONS: tuple = ("owner_stop", "wake", "interrupted")

# Said when another device dialogue was still running after the bounded wait
# (bobert_companion._self_voiced_wait_ready) and the action was not started.
BUSY_LINE = ("I'm afraid another chat is still going, sir; ask me again "
             "once it's finished.")

# The result when the action was NOT started because an accepted interrupt
# (the owner's stop or wake word, a tray / web STOP, the other chat's own
# stop) landed while it waited for another dialogue (review 2026-10-05: the
# waiting chat used to start the moment the stopped one ended, out of the
# stop's reach). That turn is barged; this reads as an owner stop
# (stopped_by_owner), so nothing is added.
STOPPED_WHILE_WAITING = "Not started: interrupted."


def spoke_then_raised(exc) -> str:
    """The result of a self-voiced action that raised AFTER it had spoken
    (bobert_companion._run_self_voiced): its talking is done. It carries no
    crash marker (core.failure_markers.ACTION_CRASH_MARK) and no terminal
    prefix, so no failure round runs the action again and nothing more is
    said. Never raises."""
    try:
        return (f"Ended after it had spoken; then it raised "
                f"{type(exc).__name__}.")
    except Exception:
        return "Ended after it had spoken; then it raised."

_WORD_RE = re.compile(r"[a-z_]+")
# "Chat not started: x." / "Banter finished: 0 lines, x." -> "chat" / "banter".
_SUBJECT_RE = re.compile(
    r"^\s*([a-z][a-z ]{0,30}?)\s+(?:not started|finished|did not start|"
    r"didn't start|not run)\b", re.I)
_SUBJECT_MAX_WORDS = 3


def reason_of(result) -> str:
    """The LAST of JARVIS's dialogue words in ``result`` (a reason in
    REASON_CLAUSES, an owner stop, or "done"), else "". Never raises."""
    try:
        if not isinstance(result, str):
            return ""
        found = ""
        for w in _WORD_RE.findall(result.lower()):
            if w in REASON_CLAUSES or w in OWNER_STOP_REASONS or w == "done":
                found = w
        return found
    except Exception:
        return ""


def stopped_by_owner(result) -> bool:
    """True when ``result`` says the owner ended the action (see
    OWNER_STOP_REASONS). Never raises."""
    return reason_of(result) in OWNER_STOP_REASONS


def subject_of(result) -> str:
    """What did not start, as the owner hears it: "the chat" for "Chat not
    started: ...", else "that". Never raises."""
    try:
        m = _SUBJECT_RE.match(result if isinstance(result, str) else "")
        if m:
            words = m.group(1).lower().split()
            if 0 < len(words) <= _SUBJECT_MAX_WORDS:
                return "the " + " ".join(words)
    except Exception:
        pass
    return "that"


def silent_line(result) -> str:
    """The one honest line for a self-voiced action that said nothing:
    "I'm afraid the chat didn't start, sir; the microphone is muted." The
    reason clause only for JARVIS's own dialogue words (see the module
    docstring). Never raises."""
    try:
        subject = subject_of(result)
        clause = REASON_CLAUSES.get(reason_of(result), "")
        if clause:
            return f"I'm afraid {subject} didn't start, sir; {clause}."
        return f"I'm afraid {subject} didn't start, sir."
    except Exception:
        return "I'm afraid that didn't start, sir."
