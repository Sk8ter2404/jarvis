"""
skills/schedule_manager.py — voice-friendly bridge to ``core.scheduler``.

Lets the user say things like::

    "every morning at 8 a.m. brief me on emails and weather and play lo-fi"
    "every thirty minutes run system pulse"
    "remind me in two hours to take a screenshot"
    "when bambu print finishes proactive_announce sir the print is done"
    "list schedules"
    "cancel schedule cron_abcd1234"

Actions registered
------------------
    schedule_recurring  <spec> | <action> [arg]
    schedule_once       <when> | <action> [arg]
    schedule_when       <condition> | <action> [arg]
    list_schedules
    cancel_schedule     <job_id>
    fire_schedule       <job_id>
    schedule_status

Multi-step chains are expressed by chaining actions with " && " in the
RHS of the pipe — ``schedule_recurring 8am | morning_briefing && weather
&& play_music lo-fi``.  The first segment is the primary action; the
rest are appended to the job's ``chain`` list and dispatched in order at
fire time.

If APScheduler is not installed the skill still registers, but every
action returns a clean install hint instead of crashing.
"""
from __future__ import annotations

import re
from typing import Callable


_INSTALL_HINT = (
    "scheduler unavailable, sir — APScheduler isn't installed. "
    "Run `pip install apscheduler sqlalchemy` and restart."
)

# Set in register() if scheduler.bootstrap() returned False or raised.
# Holds the raw error string so action factories can surface it instead of
# letting `_scheduler()` raise a cryptic "scheduler not bootstrapped".
_bootstrap_error: str | None = None


def _bootstrap_failure_message() -> str:
    err = _bootstrap_error or "unknown bootstrap failure"
    low = err.lower()
    hint = ""
    if "sqlalchemy" in low or "no module named 'sqlalchemy'" in low:
        hint = " Run `pip install sqlalchemy` and restart."
    elif "apscheduler" in low and "modulenotfound" in low.replace(" ", ""):
        hint = " Run `pip install apscheduler` and restart."
    return f"scheduler bootstrap failed, sir — {err}.{hint}"


def _preflight(scheduler) -> str | None:
    """Return an error string if the scheduler isn't usable, else None."""
    if not scheduler.is_available():
        return _INSTALL_HINT
    if _bootstrap_error is not None:
        return _bootstrap_failure_message()
    return None


# ── shared parsing helpers ──────────────────────────────────────────
def _split_pipe(arg: str) -> tuple[str, str]:
    """Split '<lhs> | <rhs>' into (lhs, rhs); '|' may be surrounded by spaces."""
    if "|" not in arg:
        return arg.strip(), ""
    lhs, rhs = arg.split("|", 1)
    return lhs.strip(), rhs.strip()


def _parse_action_chain(rhs: str) -> tuple[str, str, list[dict]]:
    """Parse 'action arg && action2 arg2 && action3'.

    Returns (primary_action, primary_arg, chain_list).  The chain list
    is a list of ``{"action": str, "arg": str}`` dicts.
    """
    parts = [p.strip() for p in re.split(r"\s*&&\s*", rhs) if p.strip()]
    if not parts:
        return "", "", []
    primary = parts[0]
    p_action, p_arg = _split_action_and_arg(primary)
    chain: list[dict] = []
    for p in parts[1:]:
        a, b = _split_action_and_arg(p)
        if a:
            chain.append({"action": a, "arg": b})
    return p_action, p_arg, chain


def _split_action_and_arg(token: str) -> tuple[str, str]:
    """'morning_briefing' → ('morning_briefing', '');
    'play_music lo-fi beats' → ('play_music', 'lo-fi beats')."""
    token = token.strip()
    if not token:
        return "", ""
    parts = token.split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1].strip()


def _format_jobs(jobs: list[dict]) -> str:
    if not jobs:
        return "No scheduled jobs, sir."
    lines = []
    for j in jobs:
        chain = j.get("chain") or []
        head = f"  • {j['id']} — {j['kind']} ({j['trigger']}) → {j['action']}"
        if j.get("arg"):
            head += f" {j['arg']!r}"
        if chain:
            head += f" + {len(chain)} chained step(s)"
        if j.get("next_run"):
            head += f" — next: {j['next_run']}"
        lines.append(head)
    return f"{len(jobs)} schedule(s), sir:\n" + "\n".join(lines)


def _format_conditions(conds: list[dict]) -> str:
    if not conds:
        return ""
    lines = []
    for c in conds:
        chain = c.get("chain") or []
        head = f"  • {c['id']} — when:{c['condition']} → {c['action']}"
        if c.get("arg"):
            head += f" {c['arg']!r}"
        if chain:
            head += f" + {len(chain)} chained step(s)"
        if c.get("one_shot"):
            head += " (one-shot)"
        cv = c.get("current_value")
        if cv is not None:
            head += f" — currently {cv}"
        lines.append(head)
    return f"{len(conds)} conditional trigger(s), sir:\n" + "\n".join(lines)


# ── action factories ────────────────────────────────────────────────
def _make_recurring(scheduler) -> Callable[[str], str]:
    def _act(arg: str = "") -> str:
        pf = _preflight(scheduler)
        if pf:
            return pf
        lhs, rhs = _split_pipe(arg or "")
        if not lhs or not rhs:
            return (
                "Format: schedule_recurring <spec> | <action> [arg] "
                "[&& <action2> [arg2] ...].  Examples of <spec>: '8am', "
                "'8:30 am weekdays', 'every 30 minutes', 'wednesday 9pm'."
            )
        p_action, p_arg, chain = _parse_action_chain(rhs)
        if not p_action:
            return "Format: <spec> | <action> [arg]"
        try:
            jid = _build_recurring_job(scheduler, lhs, p_action, p_arg, chain)
        except ValueError as e:
            return f"Could not parse '{lhs}', sir — {e}"
        except Exception as e:
            return f"Schedule failed, sir — {type(e).__name__}: {e}"
        n_chain = f" + {len(chain)} chained step(s)" if chain else ""
        return f"Recurring schedule '{jid}' armed, sir — {lhs} → {p_action}{n_chain}."
    return _act


def _build_recurring_job(scheduler, spec: str, action: str, arg: str, chain: list[dict]) -> str:
    """Translate a spec string into the right scheduler.schedule_* call."""
    spec = spec.strip()
    low  = spec.lower()

    # "every <N> <unit>" → IntervalTrigger
    if low.startswith("every "):
        body = spec[6:].strip()
        # "every 30 minutes"
        intv = scheduler.parse_every(body)
        if intv:
            return scheduler.schedule_interval(action=action, arg=arg, chain=chain, **intv)
        # "every morning at 8am" / "every monday at 9pm" / "every day at 7"
        return _parse_cron_phrase(scheduler, body, action, arg, chain)

    # "<dow> at <clock>" / "<clock>" / "<dow> <clock>"
    return _parse_cron_phrase(scheduler, spec, action, arg, chain)


def _parse_cron_phrase(scheduler, body: str, action: str, arg: str, chain: list[dict]) -> str:
    """Parse "morning at 8am" / "weekdays 9am" / "8:30 pm" → CronTrigger."""
    body = body.strip()
    low  = body.lower()
    # Strip filler words.
    for filler in ("morning at ", "afternoon at ", "evening at ", "night at ",
                   "morning ", "afternoon ", "evening ", "night ",
                   "day at ", "at "):
        if low.startswith(filler):
            body = body[len(filler):].strip()
            low  = body.lower()
            break

    # Try to split into "<dow> <clock>" or just "<clock>".
    tokens = body.split()
    dow = None
    clock_str = body
    if len(tokens) >= 2:
        # Last 1–2 tokens are the clock, the rest is the dow phrase.
        # Try the last token as clock; if it fails, try the last two.
        if scheduler.parse_clock(tokens[-1]) is not None:
            clock_str = tokens[-1]
            dow_part  = " ".join(tokens[:-1])
            dow = scheduler.parse_dow(dow_part)
            if dow is None and dow_part:
                # The "leading" text wasn't a recognised dow phrase — bail
                # back to treating the whole thing as a clock.
                clock_str = body
        elif scheduler.parse_clock(" ".join(tokens[-2:])) is not None:
            clock_str = " ".join(tokens[-2:])
            dow_part  = " ".join(tokens[:-2])
            dow = scheduler.parse_dow(dow_part)

    clock = scheduler.parse_clock(clock_str)
    if clock is None:
        raise ValueError(
            "expected a clock like '8am' or '8:30 pm', "
            "optionally prefixed with a weekday or 'weekdays/weekends'"
        )
    h, m = clock
    return scheduler.schedule_cron(
        action=action, arg=arg, chain=chain,
        hour=h, minute=m, day_of_week=dow,
    )


def _make_once(scheduler) -> Callable[[str], str]:
    def _act(arg: str = "") -> str:
        pf = _preflight(scheduler)
        if pf:
            return pf
        lhs, rhs = _split_pipe(arg or "")
        if not lhs or not rhs:
            return (
                "Format: schedule_once <when> | <action> [arg].  "
                "<when> can be 'in 30 minutes', 'tomorrow 8am', "
                "'today 3:15 pm', or an ISO datetime."
            )
        p_action, p_arg, chain = _parse_action_chain(rhs)
        when = scheduler.parse_when(lhs)
        if when is None:
            return f"Could not parse when='{lhs}', sir."
        try:
            jid = scheduler.schedule_once(
                action=p_action, arg=p_arg, chain=chain, run_at=when,
            )
        except Exception as e:
            return f"Schedule failed, sir — {type(e).__name__}: {e}"
        return f"One-shot '{jid}' armed for {when.isoformat()}, sir — → {p_action}."
    return _act


def _make_when(scheduler) -> Callable[[str], str]:
    def _act(arg: str = "") -> str:
        pf = _preflight(scheduler)
        if pf:
            return pf
        lhs, rhs = _split_pipe(arg or "")
        if not lhs or not rhs:
            return (
                "Format: schedule_when <condition> | <action> [arg].  "
                "Available conditions: "
                + ", ".join(scheduler.available_conditions())
            )
        p_action, p_arg, chain = _parse_action_chain(rhs)
        # Auto-derive a stable id from condition + primary action so the
        # user can re-issue the same when-clause without piling up
        # duplicate triggers.
        tid = f"when_{lhs.strip().lower()}_{p_action}"
        tid = re.sub(r"[^a-z0-9_]+", "_", tid).strip("_") or "when_trigger"
        try:
            scheduler.schedule_when(
                name=tid, condition=lhs.strip(),
                action=p_action, arg=p_arg, chain=chain,
            )
        except ValueError as e:
            return f"Could not arm trigger, sir — {e}"
        except Exception as e:
            return f"Trigger failed, sir — {type(e).__name__}: {e}"
        return f"Conditional trigger '{tid}' armed, sir — when {lhs} → {p_action}."
    return _act


def _make_list(scheduler) -> Callable[[str], str]:
    def _act(_: str = "") -> str:
        pf = _preflight(scheduler)
        if pf:
            return pf
        jobs  = scheduler.list_jobs()
        conds = scheduler.list_conditions()
        parts = [_format_jobs(jobs)]
        cond_str = _format_conditions(conds)
        if cond_str:
            parts.append(cond_str)
        return "\n".join(parts)
    return _act


def _make_cancel(scheduler) -> Callable[[str], str]:
    def _act(arg: str = "") -> str:
        pf = _preflight(scheduler)
        if pf:
            return pf
        job_id = (arg or "").strip()
        if not job_id:
            return "Format: cancel_schedule <job_id>"
        ok = scheduler.cancel_job(job_id)
        if ok:
            return f"Schedule '{job_id}' cancelled, sir."
        return f"No schedule '{job_id}' found, sir."
    return _act


def _make_fire(scheduler) -> Callable[[str], str]:
    def _act(arg: str = "") -> str:
        pf = _preflight(scheduler)
        if pf:
            return pf
        job_id = (arg or "").strip()
        if not job_id:
            return "Format: fire_schedule <job_id>"
        return scheduler.fire_now(job_id)
    return _act


def _make_status(scheduler) -> Callable[[str], str]:
    def _act(_: str = "") -> str:
        pf = _preflight(scheduler)
        if pf:
            return pf
        s = scheduler.status()
        line = (
            f"Scheduler {'running' if s['running'] else 'stopped'}, sir — "
            f"{s['job_count']} job(s), {s['condition_count']} conditional trigger(s). "
            f"Conditions available: {', '.join(s['registered_conditions'])}."
        )
        if s.get("last_error"):
            line += f" Last error: {s['last_error']}."
        return line
    return _act


# ── skill entry point ───────────────────────────────────────────────
def register(actions: dict) -> None:
    global _bootstrap_error

    try:
        from core import scheduler  # type: ignore
    except Exception as e:
        print(f"  [schedule_manager] core.scheduler unavailable: {e}")
        return

    # Reset on every register() so re-loads can recover after a fix.
    _bootstrap_error = None

    # Bootstrap the scheduler against the live ACTIONS dict.  Note: the
    # dict is shared by reference; future skills that add actions after
    # this skill loads will still be reachable from scheduled jobs.
    if scheduler.is_available():
        try:
            ok = scheduler.bootstrap(actions)
        except Exception as e:
            ok = False
            _bootstrap_error = f"{type(e).__name__}: {e}"
        if not ok:
            # bootstrap() returns False on failure and stashes the reason
            # in its internal state — surface that instead of leaving the
            # user with a cryptic "scheduler not bootstrapped" later.
            if _bootstrap_error is None:
                try:
                    _bootstrap_error = (
                        scheduler.status().get("last_error")
                        or "unknown bootstrap failure"
                    )
                except Exception as e:
                    _bootstrap_error = f"status() raised: {type(e).__name__}: {e}"
            print(f"  [schedule_manager] bootstrap failed: {_bootstrap_error}")
            low = _bootstrap_error.lower()
            if "sqlalchemy" in low:
                print("  [schedule_manager] hint: pip install sqlalchemy")
    else:
        print("  [schedule_manager] APScheduler not installed — actions will "
              "return an install hint until you `pip install apscheduler sqlalchemy`.")

    actions["schedule_recurring"]   = _make_recurring(scheduler)
    actions["schedule_cron"]        = actions["schedule_recurring"]
    actions["schedule_once"]        = _make_once(scheduler)
    actions["schedule_when"]        = _make_when(scheduler)
    actions["when_condition"]       = actions["schedule_when"]
    actions["list_schedules"]       = _make_list(scheduler)
    actions["list_schedule"]        = actions["list_schedules"]
    actions["show_schedules"]       = actions["list_schedules"]
    actions["cancel_schedule"]      = _make_cancel(scheduler)
    actions["remove_schedule"]      = actions["cancel_schedule"]
    actions["fire_schedule"]        = _make_fire(scheduler)
    actions["run_schedule"]         = actions["fire_schedule"]
    actions["schedule_status"]      = _make_status(scheduler)
