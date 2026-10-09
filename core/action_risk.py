"""core/action_risk.py — the ONE risk classification of action NAMES.

An action name that runs WITHOUT the LLM's judgement needs a guard when it can
stop JARVIS, destroy data, run code, act on the desktop or spend money. Two
paths dispatch such names:

  * the web dashboard's Actions tab (tools/web_interface.py, POST /api/action):
    a name matching any rule below asks for an explicit confirmation first;
  * the voice dispatcher's fuzzy action-name corrector (command_autocorrect,
    called from bobert_companion.parse_and_run_actions): a GUESSED name is
    never mapped onto a name in GUESS_PROTECTED_REASONS' categories. Live
    2026-10-01: for "Jarvis, turn it off" the local model invented
    [ACTION: shutdown] and the corrector routed it to shutdown_jarvis.

A third classification, SELF_TERMINATING_ACTIONS (2026-10-02), is the
monolith's _FIRE_AND_EXIT_ACTIONS: actions that end JARVIS's process at once.
One of those, emitted by the model by its EXACT name, runs only when the
owner's own words ask for it (asked_for_self_termination); the prompt router
never lets a section documenting one ride a follow-up's history.

The rules lived in tools/web_interface.py until 2026-10-01; that module now
imports them from here (its ``_ACTION_CONFIRM_RULES`` / ``action_confirm_reason``
are these objects), so the two paths cannot drift.

Rules are fnmatch patterns on the lower-cased name, each with the reason shown
in the web confirm prompt. Deliberately broad: a spurious prompt costs a click
(or, for a guess, one clarifying question); a missing one can message someone,
wipe memory or shut JARVIS down. The patterns name the known shutdown / code
runner aliases too, for callers that have no live registry to match aliases by
handler (the web index fallback).

Pure stdlib, no I/O, never raises.
"""
from __future__ import annotations

import re
from fnmatch import fnmatchcase

STOPS_JARVIS = "stops or restarts JARVIS or the PC"
SENDS = "sends or says something to someone"
DELETES = "deletes, resets or exports data"
RUNS_CODE = "changes JARVIS's own code or runs code"
DESKTOP = "acts on the desktop or stops a running service"
SPENDS = "spends money"

ACTION_CONFIRM_RULES = (
    (("*shutdown*", "*shut_down*", "*restart*", "*reboot*", "*hibernate*",
      "sleep_pc", "*log_off*", "*logoff*", "*sign_out*", "lock_pc",
      "lock_screen", "*relaunch*", "exit_jarvis", "quit_jarvis",
      "*power_off*", "turn_off_jarvis"),
     STOPS_JARVIS),
    (("send_*", "*_send", "reply_*", "*_reply", "text_*", "*_text_*",
      "email_*", "*_email", "sms_*", "call_*", "answer_call", "decline_call",
      "post_*", "publish_*", "share_*", "notify_*", "message_*", "*_message",
      "announce_*", "speak_*", "say_*"),
     SENDS),
    (("archive_*", "delete_*", "*_delete", "forget_*", "*_forget", "clear_*",
      "wipe_*", "reset_*", "*_reset", "*purge*", "remove_*", "*_remove",
      "erase_*", "empty_*", "drop_*", "scrap_*", "uninstall_*", "unenroll_*",
      "export_memory", "revoke_*"),
     DELETES),
    (("start_overnight_upgrade", "*upgrade*", "*self_update*", "apply_*",
      "install_*", "run_shell", "run_code", "run_python", "python",
      "eval_python", "compute", "execute_*", "*_execute", "*_script",
      "code_*", "pip_*", "git_*", "rollback*", "*_rollback"),
     RUNS_CODE),
    (("type", "type_*", "hotkey", "click", "*_click", "press_*", "kill_*",
      "close_*", "*_close", "stop_pipeline", "web_interface_off", "*_off_all",
      "force_*", "switch_llm", "switch_model", "set_model", "use_model"),
     DESKTOP),
    (("buy_*", "order_*", "pay_*", "purchase_*", "checkout*", "transfer_*"),
     SPENDS),
)

# The categories a GUESSED action name may never land on. SENDS is left out on
# purpose: its patterns are broad enough to catch read-outs (*_email matches
# read_email / unread_email), and the voice path already reads every send_*
# draft back and waits for a yes before it goes out (core/draft_preview_gate),
# so a guessed send cannot reach anyone unheard.
GUESS_PROTECTED_REASONS = frozenset(
    {STOPS_JARVIS, DELETES, RUNS_CODE, DESKTOP, SPENDS})


def _norm(name) -> str:
    return str(name or "").strip().lower()


def action_confirm_reason(name: str) -> str:
    """The confirm-prompt reason for action ``name`` (the FIRST matching
    rule), or '' when it may run on one click."""
    n = _norm(name)
    for patterns, why in ACTION_CONFIRM_RULES:
        if any(fnmatchcase(n, p) for p in patterns):
            return why
    return ""


def confirm_reasons(name: str) -> tuple:
    """Every rule category ``name`` matches, in rule order (a name can be in
    two: archive_email is SENDS and DELETES). () for an unmatched name."""
    n = _norm(name)
    if not n:
        return ()
    return tuple(why for patterns, why in ACTION_CONFIRM_RULES
                 if any(fnmatchcase(n, p) for p in patterns))


def guess_protected(name: str) -> bool:
    """True when a fuzzy-corrected (guessed) action name must never be routed
    onto ``name``: it matches a rule in a GUESS_PROTECTED_REASONS category.
    By NAME only - bobert_companion._autocorrect_protected adds the monolith's
    own sets, CONFIRM_KEYWORDS and same-handler aliases. Never raises."""
    try:
        return any(why in GUESS_PROTECTED_REASONS
                   for why in confirm_reasons(name))
    except Exception:
        return True


# ── Self-terminating actions (review 2026-10-02) ─────────────────────────
# Actions that end JARVIS's own process the moment they run: each handler
# schedules os._exit on a short timer, and none of them is confirmation- or
# pushback-gated. bobert_companion._FIRE_AND_EXIT_ACTIONS is this set, and
# core.prompt_router never lets a section that documents one ride a short
# follow-up's history - the ONE list, so the three cannot drift.
SELF_TERMINATING_ACTIONS = frozenset({
    "start_overnight_upgrade",
    "upgrade",
    "restart",
    # every shutdown alias: one graceful power-down handler
    "shutdown_jarvis",
    "shut_down",
    "exit_jarvis",
    "quit_jarvis",
    "power_off_jarvis",
    "turn_off_jarvis",
})

_SELF_TERMINATION_CLASS = {
    "start_overnight_upgrade": "overnight",
    "upgrade": "upgrade",
    "restart": "restart",
}

# Words after a stop / restart verb that leave it aimed at JARVIS himself:
# "shut down", "shut down now", "restart and upgrade", "reboot yourself".
# Anything else is an object ("restart the router", "shut down Spotify",
# "turn it off") and the verb does not ask for JARVIS to go.
_SELF_AIM = frozenset({
    "", "yourself", "jarvis", "you", "now", "please", "sir", "completely",
    "fully", "entirely", "for", "tonight", "overnight", "already",
    "immediately", "right", "then", "and", "so", "again", "asap", "quickly",
    "first", "too", "when", "after", "once",
})
# The verbs, per class. "turn off" / "switch off" count ONLY with JARVIS as
# the object: a bare "turn it off" is the 2026-10-01 incident itself.
_VERBS = {
    "shutdown": ("shut down", "shutdown", "power off", "power down", "exit",
                 "quit", "go offline"),
    "restart": ("restart", "reboot", "relaunch", "re launch", "reload",
                "start over"),
    "upgrade": ("upgrade",),
}
_SELF_PHRASES = {
    "shutdown": re.compile(
        r"\b(?:turn|switch|shut|power|close|kill)\s+(?:yourself|jarvis)\b"
        r"|\b(?:turn|switch|close|kill)\s+off\s+(?:yourself|jarvis)\b"
        r"|\bgo(?:ing)?\s+offline\b"),
    "restart": re.compile(
        r"\b(?:restart|reboot|relaunch|reload|reset)\s+(?:yourself|jarvis)\b"),
    "upgrade": re.compile(
        r"\b(?:upgrade|update|improve)\s+(?:yourself|jarvis)\b"
        r"|\bself\s+(?:update|upgrade)\b"
        r"|\b(?:apply|install)\s+the\s+(?:changes|update|updates|upgrade)\b"),
    "overnight": re.compile(
        r"\b(?:overnight|good\s*night|night\s+night|nighty\s+night|bed|"
        r"bedtime|sleep|turn(?:ing)?\s+in|call(?:ing)?\s+it\s+a\s+night|"
        r"hit(?:ting)?\s+the\s+(?:hay|sack)|done\s+for\s+the\s+(?:night|day))\b"),
}


def self_termination_class(name) -> str:
    """'shutdown' / 'restart' / 'upgrade' / 'overnight' for a self-terminating
    action, '' for anything else."""
    n = _norm(name)
    if n not in SELF_TERMINATING_ACTIONS:
        return ""
    return _SELF_TERMINATION_CLASS.get(n, "shutdown")


# Words QUOTED in the utterance are not the owner asking (review 2026-10-06):
# 'He said "shut down."' / 'the line was "go offline now"' armed the
# shutdown prompt. Double quotes only - an apostrophe is a contraction.
_QUOTED_RE = re.compile(r"[\"“”][^\"“”]*[\"“”]")
# A negator in the three words before the verb takes that mention out (review
# 2026-10-06): "don't shut down" / "never go offline" / "no, don't turn
# yourself off" / "don't go to bed" all asked for it here, so an armed
# overnight prompt answered "No, don't shut down." powered JARVIS off.
_NEGATORS = frozenset({
    "dont", "not", "never", "wont", "cant", "cannot", "shouldnt", "mustnt",
    "didnt", "doesnt", "wouldnt", "couldnt", "isnt", "wasnt",
})
# Someone else as the verb's subject: "should I shut down?", "it shut down",
# "they shut down at nine" - the owner or a device, not JARVIS.
_OTHER_SUBJECTS = frozenset({"i", "we", "they", "he", "she", "it"})
# "No overnight protocol" said WITH the shutdown (live 2026-10-06): "shut
# down no overnight protocol", "shut down with no overnight protocol", "shut
# down without the protocol" answer the overnight question JARVIS asks first,
# so the verb is still aimed at JARVIS. Only the overnight protocol (or its
# upgrade) is a thing to decline here: "shut down with no warning" / "shut
# down? No." are not.
_OVERNIGHT_THING = (
    r"(?:(?:the|any|an|that|your)\s+)?"
    r"(?:overnight(?:\s+(?:protocol|mode|upgrades?|thing|run))?|"
    r"night\s+protocol|protocol|upgrades?|updates?|upgrading|updating)")
_DECLINE = r"(?:with\s+no|with\s+out|without|no|not)\s+" + _OVERNIGHT_THING
_DECLINE_AFTER_VERB_RE = re.compile(r"^" + _DECLINE + r"\b")
# The WHOLE utterance is a shutdown that declines the overnight protocol, or
# (in reply to the overnight question) the decline alone. Strict on purpose:
# anything more ("shut down the printer with no overnight protocol", "don't
# shut down, no overnight protocol") is not this shape and goes the usual way.
_WAKE = r"(?:(?:hey|ok|okay)\s+)?jarvis\s+"
_LEAD = r"(?:(?:okay|ok|just|please|then|so|go\s+ahead\s+and|and|no|nope)\s+)*"
_SHUTDOWN_CMD = (
    r"(?:shut\s*down|power\s+(?:off|down)|go\s+offline|"
    r"turn\s+(?:yourself\s+off|off\s+jarvis)|switch\s+yourself\s+off)"
    r"(?:\s+(?:yourself|jarvis))?")
_SOFT = (r"(?:\s+(?:now|please|sir|jarvis|then|completely|fully|tonight|"
         r"thanks|thank\s+you|right\s+now|for\s+the\s+night|this\s+time))*")
_DECLINE_FULL = _DECLINE + _SOFT
_CMD_FULL = _SHUTDOWN_CMD + _SOFT
_SHUTDOWN_DECLINING_RE = re.compile(
    r"^(?:" + _WAKE + r")?" + _LEAD + r"(?:"
    + _CMD_FULL + r"\s+(?:and\s+)?" + _DECLINE_FULL
    + r"|" + _DECLINE_FULL + r"\s+(?:(?:and|so|just|then)\s+)*" + _CMD_FULL
    + r")$")
_DECLINE_ONLY_RE = re.compile(
    r"^(?:" + _WAKE + r")?" + _LEAD + _DECLINE_FULL + r"$")


def _clean_words(text) -> str:
    t = _QUOTED_RE.sub(" ", str(text or "").lower())
    t = re.sub(r"[^a-z0-9]+", " ", t.replace("'", "").replace("’", ""))
    return " " + " ".join(t.split()) + " "


def _aimed_elsewhere(low: str, i: int) -> bool:
    """True when the verb starting at index ``i`` of ``low`` (a
    _clean_words string) is negated ("don't shut down") or has someone
    other than JARVIS as its subject ("should I shut down")."""
    before = low[:i].split()
    if any(w in _NEGATORS for w in before[-3:]):
        return True
    return bool(before) and before[-1] in _OTHER_SUBJECTS


def shutdown_declining_overnight(text, *, alone_ok: bool = False) -> bool:
    """True when ``text`` is, as a whole, a shutdown of JARVIS that already
    says no to the overnight protocol (live 2026-10-06): "Jarvis shut down no
    overnight protocol.", "Jarvis shut down with no overnight protocol.",
    "shut down without the protocol", "no overnight protocol, just shut
    down". ``alone_ok`` (the overnight question is open): the decline alone
    counts too - "without the overnight protocol", "with no overnight
    protocol". A negated, quoted or longer sentence never matches. Never
    raises; a fault is False."""
    try:
        low = _clean_words(text).strip()
        if not low:
            return False
        if _SHUTDOWN_DECLINING_RE.match(low):
            return True
        return bool(alone_ok and _DECLINE_ONLY_RE.match(low))
    except Exception:
        return False


def asked_for_self_termination(name, user_text) -> bool:
    """True when the owner's own words ask for what ``name`` does to JARVIS.

    A self-terminating action (SELF_TERMINATING_ACTIONS) emitted by the model
    runs at once, with no confirmation. Review 2026-10-02: an inherited
    section (or the cloud route's full prompt) names shutdown_jarvis exactly,
    so for an ambiguous "Okay, turn it off." the model could emit it and
    JARVIS would go. The dispatcher runs one only when this is True and asks
    first otherwise. Generous on purpose - a needless "say yes" costs one
    word, a missing one ends the session:
      shutdown  "shut down" / "shutdown" / "power off" / "exit" / "quit"
                with no object (or JARVIS as it), "turn yourself off",
                "turn off Jarvis", "go offline";
      restart   "restart" / "reboot" / "relaunch" / "reload" / "start over"
                with no object, "restart yourself";
      upgrade   "upgrade" with no object, "update yourself", "apply the
                changes", "install the update";
      overnight "overnight", "goodnight", "bed", "sleep", "calling it a
                night", ...
    A shutdown may also decline the overnight protocol in the same breath:
    "shut down no overnight protocol", "shut down with no overnight
    protocol", "shut down without the protocol" (live 2026-10-06).
    Never counted (review 2026-10-06): a negated verb ("don't shut down",
    "never go offline"), another subject ("should I shut down?", "it shut
    down"), or words inside double quotes ('He said "shut down."').
    Any other action name is True (not this gate's business). Never raises;
    a fault is False, so JARVIS asks."""
    try:
        cls = self_termination_class(name)
        if not cls:
            return True
        low = _clean_words(user_text)
        phrases = _SELF_PHRASES.get(cls)
        if phrases is not None:
            for m in phrases.finditer(low):
                if not _aimed_elsewhere(low, m.start()):
                    return True
        for verb in _VERBS.get(cls, ()):
            start = 0
            needle = " " + verb + " "
            while True:
                i = low.find(needle, start)
                if i < 0:
                    break
                start = i + 1
                if _aimed_elsewhere(low, i + 1):
                    continue
                rest = low[i + len(needle):]
                nxt = rest.split(" ", 1)[0]
                if nxt in _SELF_AIM:
                    return True
                if cls == "shutdown" and _DECLINE_AFTER_VERB_RE.match(rest):
                    return True
        return False
    except Exception:
        return False


_SELF_TERMINATION_QUESTIONS = {
    "shutdown": ("That would shut me down completely, sir. Say yes if that "
                 "is what you want."),
    "restart": "That would restart me, sir. Say yes to go ahead.",
    "upgrade": ("That would start an upgrade and take me offline for it, "
                "sir. Say yes to go ahead."),
    "overnight": ("That would put me into overnight standby, sir. Say yes "
                  "to go ahead."),
}


def self_termination_question(name) -> str:
    """The question JARVIS asks instead of running ``name`` unasked."""
    return _SELF_TERMINATION_QUESTIONS.get(
        self_termination_class(name) or "shutdown",
        _SELF_TERMINATION_QUESTIONS["shutdown"])
