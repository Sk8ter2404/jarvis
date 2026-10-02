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
