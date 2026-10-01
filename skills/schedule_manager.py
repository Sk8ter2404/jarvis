"""
skills/schedule_manager.py — voice-friendly bridge to ``core.scheduler``.

Lets the user say things like::

    "every morning at 8 a.m. brief me on emails and weather and play lo-fi"
    "every thirty minutes run system pulse"
    "remind me in two hours to take a screenshot"
    "when the print finishes tell me the print is done"
          -> schedule_when bambu_print_finished | say_aloud The print is done, sir.
    "list schedules"
    "cancel schedule cron_abcd1234"

Actions registered
------------------
    schedule_recurring  <spec> | <action> [arg]
    schedule_once       <when> | <action> [arg]
    schedule_when       <condition> | <action> [arg]
    say_aloud           <text>   (speak <text>; the step a spoken reminder uses)
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

Arm-time validation (2026-08-20)
--------------------------------
Every step named on the RHS is checked against the live ACTIONS dict via
``core.scheduler.unknown_actions()`` BEFORE the job is created, so an
impossible schedule is refused out loud while the owner is still in the
conversation instead of evaporating silently at 6 a.m.  The check is
deliberately absent from ``core.scheduler``'s own ``schedule_*`` entry
points: ``skills/self_diagnostic.py`` and ``skills/sh_hue.py`` arm jobs
during skill load for actions their own module registers moments later,
and a check there would start rejecting those.  ``list_schedules`` and
``schedule_status`` additionally report any already-armed job whose
action no longer resolves.
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


# ── arm-time action validation ──────────────────────────────────────
# WHY THIS EXISTS (2026-08-20). "every morning at 8 remind me to…" used to
# arm happily against an action name that doesn't exist, speak "Recurring
# schedule armed, sir", and then evaporate at 6am with no log and no notice.
# Rejecting at arm time is the only point where the owner is still in the
# conversation and can be told. The rule lives in ONE place —
# core.scheduler.unknown_actions() — precisely so it can't rot into three
# divergent copies across _make_recurring / _make_once / _make_when.
def _unknown_step_actions(scheduler, action: str, chain: list[dict]) -> list[str]:
    """Names in this job that aren't registered actions. [] means 'arm it'.

    Fails OPEN on any unexpected error (an old/partial core.scheduler, a
    duck-typed double) so validation can never be what stops a legitimate
    schedule — but says so on stdout rather than degrading quietly.
    """
    try:
        return list(scheduler.unknown_actions(action, chain) or [])
    except Exception as e:
        print(f"  [schedule_manager] arm-time action validation unavailable "
              f"({type(e).__name__}: {e}) — arming without it")
        return []


def _name_phrase(names: list[str]) -> tuple[str, str]:
    """("'a' or 'b'", "are") / ("'a'", "is") — these strings are SPOKEN."""
    quoted = [f"'{n}'" for n in names]
    if len(quoted) == 1:
        return quoted[0], "is"
    return " or ".join([", ".join(quoted[:-1]), quoted[-1]]), "are"


def _reject_unknown(scheduler, action: str, chain: list[dict]) -> str | None:
    """Owner-facing refusal string if any step is unregistered, else None."""
    bad = _unknown_step_actions(scheduler, action, chain)
    if not bad:
        return None
    names, _verb = _name_phrase(bad)
    msg = (f"I don't have an action called {names}, sir — nothing armed. "
           f"Say 'list skills' if you'd like the registered names.")
    try:
        hints = scheduler.suggest_actions(bad[0])
    except Exception:
        hints = []
    if hints:
        msg = (f"I don't have an action called {names}, sir — nothing armed. "
               f"Did you mean {' or '.join(hints)}?")
    return msg


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
        # A next_run printed for a job whose action does not exist actively
        # corroborates a false belief, so the verdict rides on the same line.
        missing = j.get("unknown_actions") or []
        if missing:
            _names, _verb = _name_phrase(missing)
            head += (f" — BROKEN: {_names} {_verb} not a registered "
                     f"action, so this will do nothing")
        lines.append(head)
    out = f"{len(jobs)} schedule(s), sir:\n" + "\n".join(lines)
    n_broken = sum(1 for j in jobs if j.get("unknown_actions"))
    if n_broken:
        # Spoken verbatim by list_schedules, so get the grammar right.
        if n_broken == 1:
            out += ("\nHeads up, sir: 1 of those is broken and will do nothing "
                    "when it fires — cancel it and re-arm with a registered "
                    "action.")
        else:
            out += (f"\nHeads up, sir: {n_broken} of those are broken and will do "
                    f"nothing when they fire — cancel them and re-arm with "
                    f"registered actions.")
    return out


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
        missing = c.get("unknown_actions") or []
        if missing:
            _names, _verb = _name_phrase(missing)
            head += (f" — BROKEN: {_names} {_verb} not a registered "
                     f"action, so this will do nothing")
        lines.append(head)
    return f"{len(conds)} conditional trigger(s), sir:\n" + "\n".join(lines)


def _volatile_note(scheduler) -> str:
    """Spoken suffix for an arm reply when the job store is in memory only.

    2026-10-01: without SQLAlchemy, APScheduler silently falls back to its
    in-memory store, so every cron / interval / one-shot job vanishes at the
    next restart (several a day) while the reply said "armed, sir". Say so
    while the owner is still in the conversation. Empty when the store
    persists, when the scheduler can't tell (an older core.scheduler), or on
    any error — this note must never break an arm reply."""
    try:
        fn = getattr(scheduler, "is_persistent", None)
        if callable(fn) and fn() is False:
            return (" Note, sir: SQLAlchemy isn't installed, so this schedule "
                    "is held in memory and will not survive a restart.")
    except Exception:
        pass
    return ""


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
        # Refuse BEFORE _build_recurring_job — a rejected job must never reach
        # the scheduler, or list_schedules would show a ghost with a next_run.
        bad = _reject_unknown(scheduler, p_action, chain)
        if bad:
            return bad
        try:
            jid = _build_recurring_job(scheduler, lhs, p_action, p_arg, chain)
        except ValueError as e:
            return f"Could not parse '{lhs}', sir — {e}"
        except Exception as e:
            return f"Schedule failed, sir — {type(e).__name__}: {e}"
        n_chain = f" + {len(chain)} chained step(s)" if chain else ""
        return (f"Recurring schedule '{jid}' armed, sir — {lhs} → "
                f"{p_action}{n_chain}.{_volatile_note(scheduler)}")
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


# Day phrases that legitimately mean "every day" (parse_dow returns None for
# them, the same None it returns for text it does not understand).
_ANY_DAY = ("daily", "everyday", "every day", "day", "any", "each day")


def _check_dow(dow, dow_part: str) -> None:
    """Refuse day text parse_dow could not read (2026-10-01).

    A None day_of_week means EVERY day to schedule_cron, so an unrecognised
    day phrase must never fall through as None: "monday at 9 pm" used to arm
    a job that fired seven days a week while the reply said it was set as
    asked. Only the every-day words in _ANY_DAY may yield None."""
    if dow is None and dow_part and dow_part.strip().lower() not in _ANY_DAY:
        raise ValueError(
            f"I didn't recognise '{dow_part}' as a weekday, 'weekdays' or "
            f"'weekends'"
        )


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
    # Drop connective words ANYWHERE, not just in front (2026-10-01): the
    # filler strip above only removes a LEADING "at", so "monday at 9 pm" kept
    # it and the day text became "monday at" — unreadable, and armed daily.
    body = " ".join(t for t in body.split() if t.lower() not in ("at", "on", "@"))
    low  = body.lower()

    # Try to split into "<dow> <clock>" or just "<clock>".
    tokens = body.split()
    dow = None
    clock_str = body
    if len(tokens) >= 2:
        # Last 1–2 tokens are the clock, the rest is the dow phrase.
        # Try the last token as clock; if it fails, try the last two. Either
        # way the day text goes through _check_dow, so both branches refuse
        # the same unreadable day phrases (the two-token branch had no check).
        if scheduler.parse_clock(tokens[-1]) is not None:
            clock_str = tokens[-1]
            dow_part  = " ".join(tokens[:-1])
            dow = scheduler.parse_dow(dow_part)
            _check_dow(dow, dow_part)
        elif scheduler.parse_clock(" ".join(tokens[-2:])) is not None:
            clock_str = " ".join(tokens[-2:])
            dow_part  = " ".join(tokens[:-2])
            dow = scheduler.parse_dow(dow_part)
            _check_dow(dow, dow_part)

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
        bad = _reject_unknown(scheduler, p_action, chain)
        if bad:
            return bad
        try:
            jid = scheduler.schedule_once(
                action=p_action, arg=p_arg, chain=chain, run_at=when,
            )
        except Exception as e:
            return f"Schedule failed, sir — {type(e).__name__}: {e}"
        return (f"One-shot '{jid}' armed for {when.isoformat()}, sir — → "
                f"{p_action}.{_volatile_note(scheduler)}")
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
        cond = _normalise_condition(scheduler, lhs)
        # Auto-derive a stable id from condition + primary action so the
        # user can re-issue the same when-clause without piling up
        # duplicate triggers.
        tid = f"when_{cond.lower()}_{p_action}"
        tid = re.sub(r"[^a-z0-9_]+", "_", tid).strip("_") or "when_trigger"
        bad = _reject_unknown(scheduler, p_action, chain)
        if bad:
            return bad
        try:
            scheduler.schedule_when(
                name=tid, condition=cond,
                action=p_action, arg=p_arg, chain=chain,
            )
        except ValueError as e:
            return f"Could not arm trigger, sir — {e}"
        except Exception as e:
            return f"Trigger failed, sir — {type(e).__name__}: {e}"
        return f"Conditional trigger '{tid}' armed, sir — when {lhs} → {p_action}."
    return _act


# Spoken verb forms → the tense the registered condition names use.
_COND_VERB_SUFFIXES = (
    ("_is_finished", "_finished"), ("_is_done", "_finished"),
    ("_has_finished", "_finished"), ("_finishes", "_finished"),
    ("_completes", "_finished"), ("_completed", "_finished"),
    ("_complete", "_finished"), ("_done", "_finished"),
    ("_fails", "_failed"), ("_starts", "_started"), ("_begins", "_started"),
)


def _normalise_condition(scheduler, phrase: str) -> str:
    """Map a spoken condition onto a registered condition name (2026-10-01).

    The prompt taught 'bambu print finishes' / 'when the print finishes', but
    core.scheduler.schedule_when accepts only exact registered names
    (bambu_print_finished, disk_low, ...), so every spoken form was refused.
    Lower-cases, joins words with underscores, drops a leading when/if/the,
    maps the verb tense (finishes → finished) and tries the 'bambu_' prefix.
    Returns the phrase unchanged when nothing matches, so the refusal still
    names what the owner said and lists the real conditions."""
    raw = (phrase or "").strip()
    try:
        known = set(scheduler.available_conditions() or [])
    except Exception:
        known = set()
    if not raw or raw in known:
        return raw
    cond = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    for lead in ("when_", "if_", "once_", "the_"):
        if cond.startswith(lead):
            cond = cond[len(lead):]
    if cond.startswith("the_"):
        cond = cond[len("the_"):]
    for old, new in _COND_VERB_SUFFIXES:
        if cond.endswith(old):
            cond = cond[: -len(old)] + new
            break
    if cond not in known and ("bambu_" + cond) in known:
        cond = "bambu_" + cond
    return cond if cond in known else raw


def _make_say_aloud() -> Callable[[str], str]:
    """`say_aloud <text>` — queue <text> to be spoken (2026-10-01).

    The step a spoken scheduled reminder needs: "every morning at 8 remind me
    to take my vitamins" → schedule_recurring 8am | say_aloud <reminder>.
    No action could speak arbitrary text before — the prompt's own example
    named proactive_announce, a Python function, not an action, so every
    attempt was refused at arm time. Tagged source="schedule" so the standby
    loop speaks it like the owner's timers."""
    def _act(arg: str = "") -> str:
        import sys as _sys
        text = (arg or "").strip()
        if not text:
            return "Format: say_aloud <what to say>"
        announcer = None
        for _name in ("bobert_companion", "__main__"):
            _mod = _sys.modules.get(_name)
            announcer = getattr(_mod, "proactive_announce", None) if _mod else None
            if callable(announcer):
                break
        if not callable(announcer):
            return "I can't reach the speech queue to say that, sir."
        try:
            ok = announcer(text, source="schedule")
        except Exception as e:
            return f"I couldn't queue that line, sir — {type(e).__name__}: {e}"
        return "Queued to speak, sir." if ok else "I couldn't queue that line, sir."
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
        broken = s.get("broken_jobs") or []
        if broken:
            ids = ", ".join(str(b.get("id")) for b in broken)
            line += (f" Warning, sir: {len(broken)} schedule(s) point at actions "
                     f"that are not registered and will do nothing when they "
                     f"fire — {ids}.")
        misses = s.get("unresolved") or []
        if misses:
            line += " " + "; ".join(
                f"job {m.get('job_id') or 'unknown'} already tried to run "
                f"'{m.get('action')}' {m.get('count')} time(s) and found nothing"
                for m in misses) + "."
        if s.get("persistent") is False:
            line += (" Job store is in memory, sir — SQLAlchemy isn't "
                     "installed, so recurring and one-shot schedules vanish "
                     "at the next restart.")
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
    actions["say_aloud"]            = _make_say_aloud()
