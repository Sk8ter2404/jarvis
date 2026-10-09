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

from core import yes_no as _yes_no

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


# ── The owner's own shutdown words (live 2026-10-06, review 2026-10-09) ──
# Four tries, zero shutdowns live: "Jarvis shut down no overnight protocol."
# was read as unrelated, "Jarvis shut down with no overnight protocol." was
# held for a yes, and "Jarvis, yes, that is what I want." cancelled the held
# shutdown. The fix accepts those shapes and NOTHING looser: every helper
# below is a whole-utterance grammar, and the gate's own guards only ever
# take a mention OUT (a negated verb, a question, someone else as the
# subject, a quoted line) - a doubtful sentence still asks for a yes.

# A quoted span is someone else's words. It becomes one placeholder word, so
# it still fills its slot (review 2026-10-09): stripping it turned 'shut down
# "Plex"' into a bare "shut down", which asked for JARVIS himself. Double
# quotes only - an apostrophe is a contraction.
_QUOTED_RE = re.compile(r"[\"“”][^\"“”]*[\"“”]")
_QUOTED = " quoted "
# A negator before the verb takes that mention out: "don't shut down",
# "never go offline", "no, don't turn yourself off" (an armed overnight
# prompt answered "No, don't shut down." powered JARVIS off). Three words
# back for restart / upgrade / overnight; ANYWHERE earlier for a shutdown
# (review 2026-10-09: "No, I don't want you to shut down." shut it down).
_NEGATORS = frozenset({
    "dont", "not", "never", "wont", "cant", "cannot", "shouldnt", "mustnt",
    "didnt", "doesnt", "wouldnt", "couldnt", "isnt", "wasnt",
})
# ...and a verb that undoes it, right before it or before its article:
# "cancel shut down", "abort the shutdown", "skip the shutdown" (review
# 2026-10-09: "Cancel the shutdown." with the prompt open shut JARVIS down).
_UNDO_VERBS = frozenset({
    "cancel", "abort", "stop", "halt", "postpone", "delay", "skip",
    "prevent", "avoid",
})
# Someone else as the verb's subject: "should I shut down?", "it shut down",
# "I'll shut down", "it'll shut down" - the owner or a device, not JARVIS.
_OTHER_SUBJECTS = frozenset({
    "i", "we", "they", "he", "she", "it", "ill", "itll", "theyll", "wed",
    "theyd", "itd", "id",
})
# For a SHUTDOWN, the word right before the verb (same clause) must be one
# an order to JARVIS can carry there (review 2026-10-09): "Jarvis, laptop
# shut down", "my laptop shut down", "the factory had to shut down",
# "Windows will shut down" armed the overnight prompt, and a plain "No." to
# it powered JARVIS off. After a clause break (a comma, a full stop) the verb
# opens its own clause and is an order whatever came before.
_ORDER_PREV = frozenset({
    "jarvis", "hey", "hi", "ok", "okay", "so", "alright", "right", "well",
    "now", "then", "and", "um", "uh", "oh", "yes", "yeah", "yep", "yup",
    "sure", "fine", "no", "nope", "nah", "please", "sir", "kindly", "just",
    "you", "lets", "go", "ahead", "also", "finally", "immediately",
    "quickly", "completely", "fully", "full", "complete", "simply",
    "actually", "really", "enough", "done", "thanks", "thank", "goodnight",
    "night", "bye", "goodbye", "already", "again", "safely", "properly",
    "gracefully", "totally", "entirely", "officially", "today", "tonight",
})
# A modal is an order only with JARVIS as its subject ("you can shut down
# now", "Jarvis, will you shut down"); "it will shut down" is a forecast.
_MODALS = frozenset({
    "can", "could", "would", "will", "may", "might", "must", "should",
    "shall",
})
# "to" is an order after "I want you to" / "time to" / "you need to", not
# after "had to" / "going to" / "about to".
_TO_AFTER = frozenset({"you", "time", "want", "need", "like", "ready"})
# "I said shut down" repeats an order; "he said shut down" reports one.
_SAID = frozenset({"said", "say", "told"})
# A question about JARVIS is not an order (review 2026-10-09): "what happens
# if you shut down ...", "why did you go offline last night?", "is it safe to
# shut down ...", "did you shut down ...". "Can / could / would / will you
# shut down?" stay polite orders.
_QUESTION_OPENERS = frozenset({
    "what", "whats", "why", "how", "hows", "when", "whens", "where",
    "wheres", "which", "who", "whos", "whose", "whom", "did", "didnt",
    "does", "doesnt", "is", "isnt", "was", "wasnt", "werent", "are",
    "arent", "am", "if", "whether", "should", "shouldnt", "shall",
})
# ("were" is not one: with the apostrophe gone it is also "we're" - "Jarvis,
# we're done, shut down.")
_QUESTION_AUX = frozenset({
    "do", "dont", "have", "havent", "has", "hasnt", "had", "hadnt",
})
_PRONOUNS = frozenset({"you", "i", "we", "they", "he", "she", "it"})
_OPENER_SKIP = frozenset({
    "jarvis", "hey", "hi", "ok", "okay", "so", "um", "uh", "oh", "well",
    "and", "but", "also", "alright", "sir", "quick", "question",
})
# "shutdown" as a NOUN: "no shutdown", "cancel the shutdown", "the shutdown
# sequence" (review 2026-10-09: "Jarvis, no shutdown." armed the prompt).
_NOUN_DETS = frozenset({
    "no", "the", "a", "an", "any", "this", "that", "your", "my", "our",
    "his", "her", "their", "its", "every",
})
# Declining the overnight protocol ("no overnight", "skip the overnight
# protocol") is not asking for it (review 2026-10-09: the overnight gate
# counted "shut down with no overnight protocol" as asking for overnight).
_OVERNIGHT_DECLINERS = frozenset({
    "no", "without", "skip", "skipping", "forget", "minus",
})

# The whole-utterance grammar of the owner's shutdown words. Words only
# (_clean_words), so punctuation is gone: the sentence and question checks
# run on the raw text first (_one_request).
_DET = r"(?:(?:the|any|an|that|your|this)\s+)?"
# The overnight protocol by name ("shut down without the protocol" may
# leave out the "overnight" - see _decline).
_OVERNIGHT = (r"(?:over\s*night(?:\s+(?:protocols?|mode|upgrades?|thing|run|"
              r"routine|stuff|sequence|process))?|night\s+protocol)")
# Only the overnight protocol is a thing to decline here (review 2026-10-09):
# "updates" / "upgrade" / "updating" were too, so "Jarvis, laptop shut down
# without updating." armed the prompt and "Jarvis, power off, no updates."
# shut JARVIS down at once. "shut down with no warning" is not a decline,
# and "No, the overnight protocol." (a correction toward it) is not either.


def _decline(any_protocol: bool) -> str:
    """The regex of a decline of the overnight protocol. ``any_protocol``:
    the bare word "protocol" counts too, after "without" / "with out" /
    "not" only ("shut down without the protocol"). Everything a
    newer phrasing added - skip / forget / don't run / don't need - names
    the overnight protocol itself, so a film line ("Shut it down, skip the
    protocol!") is never one."""
    thing = _DET + _OVERNIGHT
    basic = (r"(?:with\s+no|no)\s+" + _OVERNIGHT
             + r"|(?:with\s+out|without|not)\s+" + _DET
             + (r"(?:" + _OVERNIGHT + r"|protocols?)" if any_protocol
                else _OVERNIGHT))
    more = (r"(?:minus|skip(?:ping)?|forget(?:\s+about)?|never\s+mind|"
            r"no\s+need\s+for)\s+" + thing
            + r"|without\s+(?:running|doing|starting|bothering\s+with)\s+"
            + thing
            + r"|(?:(?:i|we)\s+)?(?:dont|do\s+not)\s+(?:run|do|start|"
              r"bother\s+with|need|want|worry\s+about)\s+" + thing)
    return (r"(?:(?:but|and)\s+)?(?:" + basic + r"|" + more + r")"
            r"(?:\s+(?:(?:i|we)\s+)?(?:dont|do\s+not)\s+need\s+(?:it|that))?")


_DECLINE = _decline(True)
_DECLINE_OVERNIGHT = _decline(False)
_WAKE = r"(?:(?:(?:hey|ok|okay|um|uh|so|alright)\s+)*jarvis\s+)*"
_LEAD = (r"(?:(?:okay|ok|alright|just|please|then|so|go\s+ahead\s+and|and|"
         r"no|nope|nah|um|uh)\s+)*")
# The shutdown of JARVIS himself...
_SHUTDOWN_SELF = (
    r"(?:shut\s*down|shut\s+yourself\s+down|"
    r"power\s+(?:yourself\s+)?(?:off|down)|go\s+offline|"
    r"turn\s+yourself\s+off|turn\s+off\s+jarvis|switch\s+yourself\s+off)"
    r"(?:\s+(?:yourself|jarvis))?")
# ...and "shut it down" / "turn (it) off", which appear here only WITH a
# decline that names the overnight protocol - that names JARVIS as the
# "it" (alone they stay the 2026-10-01 "turn it off" incident and ask).
_SHUTDOWN_IT = (r"(?:shut\s+it\s+down|turn\s+(?:it\s+)?off|"
                r"switch\s+(?:it\s+)?off)(?:\s+jarvis)?")
_SHUTDOWN_CMD = r"(?:" + _SHUTDOWN_SELF + r"|" + _SHUTDOWN_IT + r")"
_SOFT = (r"(?:\s+(?:now|please|sir|jarvis|then|completely|fully|tonight|"
         r"thanks|thank\s+you|right\s+now|for\s+the\s+night|for\s+tonight|"
         r"this\s+time|immediately|already|asap|right\s+away|"
         r"straight\s+away))*")
_CONNECT = r"(?:\s+(?:and|so|just|then|please|but))*"


def _cmd_and_decline(cmd: str, decline: str) -> str:
    return (r"(?:" + cmd + _SOFT + _CONNECT + r"\s+" + decline + _SOFT
            + r"|" + decline + _SOFT + _CONNECT + r"\s+" + cmd + _SOFT + r")")


_SHUTDOWN_DECLINING_RE = re.compile(
    r"^" + _WAKE + _LEAD + r"(?:"
    + _cmd_and_decline(_SHUTDOWN_SELF, _DECLINE) + r"|"
    + _cmd_and_decline(_SHUTDOWN_IT, _DECLINE_OVERNIGHT) + r")$")
_DECLINE_ONLY_RE = re.compile(r"^" + _WAKE + _LEAD + _DECLINE + _SOFT + r"$")
_NO_SHUTDOWN_NOUN_RE = re.compile(r"\b(?:no|nope|nah)\s+shutdown\b")
# A YES to the overnight question that names it (review 2026-10-09): "Yes,
# run the overnight protocol, then shut down." ran a FULL shutdown (the
# repeated "shut down" read as insisting), and "Yes, overnight protocol." /
# "Do the overnight protocol." were cancelled.
_OVERNIGHT_YES_RE = re.compile(
    r"^" + _WAKE
    + r"(?:(?:yes|yeah|yep|yup|yea|sure|okay|ok|alright|please|lets|just|"
      r"go\s+ahead\s+and|um|uh|and)\s+)*"
    + r"(?:(?:shut\s*down|power\s+(?:off|down)|go\s+offline)\s+)?"
    + r"(?:(?:do|run|start|begin|use|with|go\s+with|kick\s+off|initiate|"
      r"activate|engage|launch)\s+)?"
    + r"(?:the\s+)?over\s*night(?:\s+(?:protocols?|mode|upgrades?|routine|"
      r"run|thing|sequence|process))?"
    + r"(?:\s+(?:first|please|now|sir|jarvis|tonight|thanks|thank\s+you))*"
    + r"(?:(?:\s+(?:and|then|so))*\s+(?:shut\s*down|power\s+(?:off|down)|"
      r"go\s+offline)(?:\s+(?:after|afterwards|after\s+that|please|sir|"
      r"then))*)?$")
# A plain NO to the overnight question (review 2026-10-09): any "no ..." the
# hedge list did not know was a full shutdown - "No, stay on.", "No, keep
# running.", "No, I need you.", "No, my laptop.", "Just shut down the
# printer.". Now only a no that adds nothing but thanks, the protocol it
# declines, or the shutdown itself is a no; anything else cancels. "No
# shutdown" (the noun) is not "No, shut down".
_PLAIN_NO_RE = re.compile(
    r"^(?:(?:no|nope|nah|negative)(?:\s+(?:no|nope|nah))*"
    r"(?:\s+(?:sir|jarvis))*"
    r"(?:\s+over\s*night(?:\s+(?:protocols?|mode))?)?"
    r"(?:\s+(?:thanks|thank\s+you|im\s+good|i\s+am\s+good|im\s+fine|"
    r"i\s+am\s+fine|im\s+ok(?:ay)?|all\s+good|thats\s+(?:fine|ok(?:ay)?|"
    r"alright|all)|skip\s+(?:it|that)|no\s+need|not\s+needed|"
    r"not\s+necessary))?"
    r"(?:(?:\s+(?:just|please|so|then|and|go\s+ahead\s+and))*\s+"
    r"(?:shut\s+down|shut\s+yourself\s+down|power\s+(?:off|down)|"
    r"go\s+offline|turn\s+(?:yourself\s+)?off|switch\s+(?:yourself\s+)?off|"
    r"full\s+shut\s*down|shut\s*down\s+completely|go\s+ahead|do\s+it|"
    r"proceed)"
    r"|(?:\s+(?:just|please|go\s+ahead\s+and))+\s+shutdown)?"
    r"|(?:just\s+)?(?:full\s+shut\s*down|shut\s*down\s+completely)"
    r"|just\s+shut\s*down)" + _SOFT + r"$")
# A YES to "That would shut me down completely, sir. Say yes if that is what
# you want." that restates the shutdown (review 2026-10-09): "Yes, shut down
# with no overnight protocol." was a "no" (the "no" read as a hedge), "Yes, I
# want you to shut down." / "Yes, shut it down." / "Yes, power down." were
# "other" - each answer cancelled the held shutdown.
_YES_WORD = (r"(?:yes|yeah|yep|yup|yea|ya|sure|okay|ok|alright|absolutely|"
             r"definitely|certainly|correct|affirmative|confirm(?:ed)?|"
             r"indeed|do\s+it|go\s+ahead)")
_YES_MORE = r"(?:" + _YES_WORD[3:-1] + r"|please)"
_HELD_LEAD = (r"(?:(?:i\s+(?:(?:really|do|definitely)\s+)*want\s+"
              r"(?:you\s+)?to|i\s+asked\s+you\s+to|i\s+told\s+you\s+to|"
              r"i\s+said|go\s+ahead\s+and|please|just|you\s+can|you\s+may|"
              r"so|and|then)\s+)*")
_HELD_SURE = (r"(?:im\s+sure|i\s+am\s+sure|im\s+certain|i\s+am\s+certain|"
              r"thats\s+(?:what\s+i\s+want|right)|" + _yes_no.AFFIRM_TAIL
              + r")")
_HELD_YES_RE = re.compile(
    r"^" + _WAKE + r"(?:(?:um|uh)\s+)*" + _YES_WORD
    + r"(?:\s+" + _YES_MORE + r")*(?:\s+(?:sir|jarvis))*"
    + r"(?:\s+" + _HELD_SURE + r")?"
    + r"(?:\s+" + _HELD_LEAD + _SHUTDOWN_CMD + _SOFT
    + r"(?:" + _CONNECT + r"\s+" + _DECLINE + _SOFT + r")?)?"
    + _SOFT + r"$")
# ...and the owner's restatement on its own answers that question too ("That
# is what I want." / "That's what I said.") - THIS question only: elsewhere a
# bare restatement is a correction as often as a yes (core.yes_no).
_HELD_TAIL_ONLY_RE = re.compile(
    r"^" + _WAKE + r"(?:" + _yes_no.AFFIRM_TAIL + r")" + _SOFT + r"$")
# Sentences that are only address or an answer word ("Jarvis.", "No.",
# "Okay.") ride along with the one that carries the request.
_LEAD_ONLY = frozenset({
    "jarvis", "hey", "hi", "ok", "okay", "sir", "please", "thanks", "thank",
    "you", "so", "alright", "um", "uh", "oh", "well", "no", "nope", "nah",
    "just",
})


def _clean_words(text) -> str:
    t = _QUOTED_RE.sub(_QUOTED, str(text or "").lower())
    t = re.sub(r"[^a-z0-9]+", " ", t.replace("'", "").replace("’", ""))
    return " " + " ".join(t.split()) + " "


def _clause_words(text):
    """(words, breaks): the words of ``text`` as _clean_words has them, and
    per word whether a clause break (, ; : . ! ? or a dash) comes right
    before it."""
    t = _QUOTED_RE.sub(_QUOTED, str(text or "").lower())
    t = t.replace("'", "").replace("’", "")
    t = re.sub(r"\s-\s|[,;:.!?…—–]+", " | ", t)
    t = re.sub(r"[^a-z0-9|]+", " ", t)
    words: list = []
    breaks: list = []
    brk = False
    for w in t.split():
        if w == "|":
            brk = True
            continue
        words.append(w)
        breaks.append(brk)
        brk = False
    return words, breaks


def _one_request(text):
    """The words of ``text`` (a _clean_words string, stripped) when it is ONE
    spoken request: no question mark, nothing quoted, and one sentence once
    the address-only ones ("Jarvis.", "No.") are set aside - else None.
    Review 2026-10-09: "Shut down with no overnight protocol? No, wait." and
    "Jarvis, shut down. No, overnight protocol." (a correction toward the
    protocol) read as the shutdown-and-decline shape once punctuation went."""
    raw = str(text or "")
    if "?" in raw or _QUOTED_RE.search(raw):
        return None
    sentences = [_clean_words(p).split() for p in re.split(r"[.!;…]+", raw)]
    sentences = [s for s in sentences if s]
    if sum(1 for s in sentences
           if not all(w in _LEAD_ONLY for w in s)) > 1:
        return None
    return " ".join(w for s in sentences for w in s)


def _opens_question(words) -> bool:
    """True when ``words`` (after the wake word and discourse openers) open
    with a question word: "what happens if you shut down", "why did you go
    offline", "do you shut down at night"."""
    i = 0
    while i < len(words) and words[i] in _OPENER_SKIP:
        i += 1
    if i >= len(words):
        return False
    if words[i] in _QUESTION_OPENERS:
        return True
    return (words[i] in _QUESTION_AUX and i + 1 < len(words)
            and words[i + 1] in _PRONOUNS)


def _ordered_here(words, breaks, k) -> bool:
    """True when a SHUTDOWN verb at word ``k`` can be an order to JARVIS
    from what stands right before it (_ORDER_PREV and friends)."""
    if k == 0 or breaks[k]:
        return True
    prev = words[k - 1]
    prev2 = words[k - 2] if k >= 2 and not breaks[k - 1] else ""
    if prev in _ORDER_PREV:
        return True
    if prev in _MODALS:
        return prev2 in ("", "you", "jarvis")
    if prev == "to":
        return prev2 in _TO_AFTER
    if prev in _SAID:
        return prev2 in ("", "i", "ive", "jarvis")
    return False


def _aimed_elsewhere(words, breaks, k, cls) -> bool:
    """True when the verb at word ``k`` does NOT ask JARVIS to go: negated
    ("don't shut down", "no, I don't want you to shut down"), part of a
    question ("why did you go offline"), with someone else as its subject
    ("should I shut down", "my laptop shut down"), the noun ("cancel the
    shutdown"), or a declined overnight protocol ("no overnight")."""
    before = words[:k]
    if cls == "shutdown":
        if any(w in _NEGATORS for w in before):
            return True
        if before and (before[-1] in _UNDO_VERBS or (
                len(before) >= 2 and before[-1] in _NOUN_DETS
                and before[-2] in _UNDO_VERBS)):
            return True
    elif any(w in _NEGATORS for w in before[-3:]):
        return True
    if _opens_question(words):
        return True
    if k > 0 and not breaks[k] and before[-1] in _OTHER_SUBJECTS:
        return True
    if cls == "shutdown":
        if (words[k] == "shutdown" and k > 0 and not breaks[k]
                and before[-1] in _NOUN_DETS):
            return True
        return not _ordered_here(words, breaks, k)
    if cls == "overnight" and before:
        if before[-1] in _OVERNIGHT_DECLINERS:
            return True
        if (len(before) >= 2 and before[-1] in ("the", "any", "your")
                and before[-2] in _OVERNIGHT_DECLINERS):
            return True
    return False


def shutdown_declining_overnight(text, *, alone_ok: bool = False) -> bool:
    """True when ``text`` is, as a whole, ONE request to shut JARVIS down
    that already says no to the overnight protocol (live 2026-10-06):
    "Jarvis shut down no overnight protocol.", "Jarvis shut down with no
    overnight protocol.", "shut down without the protocol", "no overnight
    protocol, just shut down", "Jarvis, shut down, skip the overnight
    protocol", "shut down, I don't need the overnight protocol".
    ``alone_ok`` (the overnight question is open): the decline alone counts
    too - "without the overnight protocol", "skip the overnight protocol",
    "I don't want the overnight protocol". Never: a question, a quoted line,
    two sentences ("Shut down. No, overnight protocol."), a negated or
    device sentence, "no shutdown", or anything that declines something
    else ("without updating", "with no warning"). Never raises; a fault is
    False."""
    try:
        low = _one_request(text)
        if not low or _NO_SHUTDOWN_NOUN_RE.search(low):
            return False
        if _SHUTDOWN_DECLINING_RE.match(low):
            return True
        return bool(alone_ok and _DECLINE_ONLY_RE.match(low))
    except Exception:
        return False


def accepts_overnight(text) -> bool:
    """True when ``text``, in reply to "Would you like to start the overnight
    protocol first?", says yes to the overnight protocol by name: "Yes,
    overnight protocol.", "Do the overnight protocol.", "Yes, run the
    overnight protocol, then shut down.", "Jarvis, shut down with the
    overnight protocol.", "Overnight first, then shut down." One request
    only (no question, nothing quoted, one sentence); "no overnight",
    "without the overnight protocol" never match. Never raises."""
    try:
        low = _one_request(text)
        return bool(low and _OVERNIGHT_YES_RE.match(low))
    except Exception:
        return False


def plain_no_to_overnight(text) -> bool:
    """True when ``text`` (yes_no.normalize'd) answers the overnight question
    with a plain NO and nothing else: "no", "no, no", "no thanks", "nope,
    just shut down", "no overnight protocol", "no, I'm good", "negative",
    "just shut down", "full shutdown". A no that goes on to say anything
    else - "No, stay on.", "No, I need you.", "No, my laptop.", "No, no
    shutdown." - is not, and cancels. Never raises."""
    try:
        low = " ".join(str(text or "").split())
        return bool(low and _PLAIN_NO_RE.match(low))
    except Exception:
        return False


def confirms_held_shutdown(text) -> bool:
    """True when ``text`` answers "That would shut me down completely, sir.
    Say yes if that is what you want." with a yes that restates the shutdown
    - "Yes, shut down with no overnight protocol.", "Yes, I want you to shut
    down.", "Yes, shut it down.", "Yes, power down.", "Yes, that's what I
    want, shut down." - or with the restatement alone ("That is what I
    want."). For THAT question only (the caller checks the held action is a
    shutdown). Never a question or a quoted line; "Yes, shut down the
    printer" / "Yes, shut down later" do not match. Never raises."""
    try:
        raw = str(text or "")
        if "?" in raw or _QUOTED_RE.search(raw):
            return False
        low = _clean_words(raw).strip()
        if not low:
            return False
        return bool(_HELD_YES_RE.match(low) or _HELD_TAIL_ONLY_RE.match(low))
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
    A shutdown that declines the overnight protocol in the same breath is
    asked for too, when the WHOLE utterance is that one request
    (shutdown_declining_overnight): "Jarvis shut down with no overnight
    protocol." (live 2026-10-06).
    Never counted (reviews 2026-10-06 / 2026-10-09): a negated verb ("don't
    shut down", "no, I don't want you to shut down", "cancel the
    shutdown"), a question ("what happens if you shut down ...", "why did
    you go offline?"), someone else as the subject ("should I shut down?",
    "my laptop shut down", "Windows will shut down"), words inside double
    quotes ('He said "shut down."', 'shut down "Plex"'), and a declined
    overnight protocol for the overnight action ("no overnight protocol").
    Any other action name is True (not this gate's business). Never raises;
    a fault is False, so JARVIS asks."""
    try:
        cls = self_termination_class(name)
        if not cls:
            return True
        if cls == "shutdown" and shutdown_declining_overnight(user_text):
            return True
        words, breaks = _clause_words(user_text)
        low = " " + " ".join(words) + " "
        phrases = _SELF_PHRASES.get(cls)
        if phrases is not None:
            for m in phrases.finditer(low):
                k = low[:m.start()].count(" ") - 1
                if not _aimed_elsewhere(words, breaks, k, cls):
                    return True
        for verb in _VERBS.get(cls, ()):
            start = 0
            needle = " " + verb + " "
            while True:
                i = low.find(needle, start)
                if i < 0:
                    break
                start = i + 1
                k = low[:i + 1].count(" ") - 1
                if _aimed_elsewhere(words, breaks, k, cls):
                    continue
                nxt = low[i + len(needle):].split(" ", 1)[0]
                if nxt in _SELF_AIM:
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
