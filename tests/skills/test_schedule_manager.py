"""Logic tests for skills/schedule_manager.py.

Two layers:
  1. Pure parsing helpers (_split_pipe, _parse_action_chain,
     _split_action_and_arg, _format_jobs, _format_conditions) — no scheduler.
  2. The action factories (_make_recurring / _make_once / _make_when /
     _make_list / _make_cancel / _make_fire / _make_status) driven against a
     hand-built fake `scheduler` object. This exercises the spec-parsing
     translation (_build_recurring_job / _parse_cron_phrase) and the
     graceful-degradation paths (APScheduler missing / bootstrap failed)
     WITHOUT touching the real core.scheduler or APScheduler.
"""
from __future__ import annotations

import datetime
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class FakeScheduler:
    """Minimal stand-in for core.scheduler — records calls and returns
    deterministic values so the action factories can be unit-tested."""

    def __init__(self, available=True):
        self._available = available
        self.calls = []
        self.jobs = []
        self.conditions = []

    def is_available(self):
        return self._available

    # ── spec parsing primitives the skill calls ──
    def parse_every(self, body):
        # "30 minutes" → {"minutes": 30}; anything else → None.
        parts = body.split()
        if len(parts) == 2 and parts[0].isdigit():
            unit = parts[1].rstrip("s")
            if unit in ("minute", "min"):
                return {"minutes": int(parts[0])}
            if unit in ("hour", "hr"):
                return {"hours": int(parts[0])}
        return None

    def parse_clock(self, s):
        s = s.strip().lower()
        mapping = {"8am": (8, 0), "8:30 pm": (20, 30), "9pm": (21, 0),
                   "7": (7, 0), "8": (8, 0), "30": None, "minutes": None}
        return mapping.get(s)

    def parse_dow(self, s):
        s = (s or "").strip().lower()
        return {"weekdays": "mon-fri", "monday": "mon", "wednesday": "wed"}.get(s)

    def parse_when(self, s):
        if s.strip().lower() == "in 30 minutes":
            return datetime.datetime(2026, 6, 1, 12, 30)
        return None

    def available_conditions(self):
        return ["bambu_print_done", "credits_low"]

    # ── scheduling calls ──
    def schedule_interval(self, **kw):
        self.calls.append(("interval", kw))
        return "cron_interval_1"

    def schedule_cron(self, **kw):
        self.calls.append(("cron", kw))
        return "cron_cron_1"

    def schedule_once(self, **kw):
        self.calls.append(("once", kw))
        return "cron_once_1"

    def schedule_when(self, **kw):
        self.calls.append(("when", kw))
        return "when_1"

    def list_jobs(self):
        return self.jobs

    def list_conditions(self):
        return self.conditions

    def cancel_job(self, job_id):
        return job_id == "cron_known"

    def fire_now(self, job_id):
        return f"Fired {job_id}, sir."

    def status(self):
        return {"running": True, "job_count": 2, "condition_count": 1,
                "registered_conditions": ["bambu_print_done"], "last_error": None}


class ScheduleParsingHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("schedule_manager")

    def test_split_pipe(self):
        self.assertEqual(self.mod._split_pipe("8am | morning_briefing"),
                         ("8am", "morning_briefing"))
        self.assertEqual(self.mod._split_pipe("no pipe here"),
                         ("no pipe here", ""))

    def test_split_action_and_arg(self):
        self.assertEqual(self.mod._split_action_and_arg("morning_briefing"),
                         ("morning_briefing", ""))
        self.assertEqual(self.mod._split_action_and_arg("play_music lo-fi beats"),
                         ("play_music", "lo-fi beats"))

    def test_parse_action_chain(self):
        primary, arg, chain = self.mod._parse_action_chain(
            "morning_briefing && weather && play_music lo-fi")
        self.assertEqual(primary, "morning_briefing")
        self.assertEqual(arg, "")
        self.assertEqual(chain,
                         [{"action": "weather", "arg": ""},
                          {"action": "play_music", "arg": "lo-fi"}])

    def test_parse_action_chain_empty(self):
        self.assertEqual(self.mod._parse_action_chain(""), ("", "", []))

    def test_format_jobs_empty(self):
        self.assertEqual(self.mod._format_jobs([]), "No scheduled jobs, sir.")

    def test_format_jobs_renders_detail(self):
        out = self.mod._format_jobs([{
            "id": "j1", "kind": "cron", "trigger": "cron[h=8]",
            "action": "brief", "arg": "x", "chain": [{"a": 1}],
            "next_run": "2026-06-02 08:00"}])
        self.assertIn("j1", out)
        self.assertIn("brief", out)
        self.assertIn("chained step", out)
        self.assertIn("next", out)

    def test_format_conditions_empty_is_blank(self):
        self.assertEqual(self.mod._format_conditions([]), "")

    def test_format_conditions_renders_oneshot(self):
        out = self.mod._format_conditions([{
            "id": "c1", "condition": "bambu_done", "action": "announce",
            "arg": "", "chain": [], "one_shot": True, "current_value": False}])
        self.assertIn("c1", out)
        self.assertIn("one-shot", out)
        self.assertIn("currently", out)


class ScheduleBuildJobTests(unittest.TestCase):
    """_build_recurring_job / _parse_cron_phrase translation."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("schedule_manager")
        self.sched = FakeScheduler()

    def test_every_interval(self):
        jid = self.mod._build_recurring_job(
            self.sched, "every 30 minutes", "system_pulse", "", [])
        self.assertEqual(jid, "cron_interval_1")
        kind, kw = self.sched.calls[-1]
        self.assertEqual(kind, "interval")
        self.assertEqual(kw["minutes"], 30)

    def test_cron_clock_only(self):
        jid = self.mod._build_recurring_job(
            self.sched, "8am", "morning_briefing", "", [])
        self.assertEqual(jid, "cron_cron_1")
        kind, kw = self.sched.calls[-1]
        self.assertEqual(kind, "cron")
        self.assertEqual((kw["hour"], kw["minute"]), (8, 0))
        self.assertIsNone(kw["day_of_week"])

    def test_cron_dow_plus_clock(self):
        self.mod._build_recurring_job(
            self.sched, "wednesday 9pm", "play_music", "lo-fi", [])
        kind, kw = self.sched.calls[-1]
        self.assertEqual(kind, "cron")
        self.assertEqual((kw["hour"], kw["minute"]), (21, 0))
        self.assertEqual(kw["day_of_week"], "wed")

    def test_cron_strips_filler_every_morning_at(self):
        self.mod._build_recurring_job(
            self.sched, "every morning at 8am", "brief", "", [])
        kind, kw = self.sched.calls[-1]
        self.assertEqual((kw["hour"], kw["minute"]), (8, 0))

    def test_unparseable_clock_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.mod._build_recurring_job(
                self.sched, "whenever I feel like it", "brief", "", [])


class ScheduleActionFactoryTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("schedule_manager")
        self.mod._bootstrap_error = None
        self.sched = FakeScheduler()

    def test_recurring_happy_path(self):
        act = self.mod._make_recurring(self.sched)
        out = act("8am | morning_briefing && weather")
        self.assertIn("armed", out.lower())
        self.assertIn("8am", out)
        self.assertIn("1 chained step", out)

    def test_recurring_format_hint_on_missing_pipe(self):
        act = self.mod._make_recurring(self.sched)
        out = act("8am no pipe")
        self.assertIn("Format:", out)

    def test_recurring_reports_parse_error(self):
        act = self.mod._make_recurring(self.sched)
        out = act("garbage spec | brief")
        self.assertIn("could not parse", out.lower())

    def test_once_happy_path(self):
        act = self.mod._make_once(self.sched)
        out = act("in 30 minutes | take_screenshot")
        self.assertIn("one-shot", out.lower())
        self.assertEqual(self.sched.calls[-1][0], "once")

    def test_once_unparseable_when(self):
        act = self.mod._make_once(self.sched)
        out = act("at some point | brief")
        self.assertIn("could not parse when", out.lower())

    def test_when_happy_path_derives_id(self):
        act = self.mod._make_when(self.sched)
        out = act("bambu_print_done | proactive_announce done")
        self.assertIn("armed", out.lower())
        kind, kw = self.sched.calls[-1]
        self.assertEqual(kind, "when")
        # The auto-derived trigger id is slugified from condition + action.
        self.assertIn("bambu_print_done", kw["name"])

    def test_when_lists_conditions_on_missing_rhs(self):
        act = self.mod._make_when(self.sched)
        out = act("bambu_print_done")
        self.assertIn("bambu_print_done", out)
        self.assertIn("credits_low", out)

    def test_cancel_known_and_unknown(self):
        act = self.mod._make_cancel(self.sched)
        self.assertIn("cancelled", act("cron_known").lower())
        self.assertIn("no schedule", act("cron_missing").lower())

    def test_cancel_requires_id(self):
        act = self.mod._make_cancel(self.sched)
        self.assertIn("Format:", act(""))

    def test_fire_delegates(self):
        act = self.mod._make_fire(self.sched)
        self.assertEqual(act("cron_x"), "Fired cron_x, sir.")

    def test_list_renders_jobs_and_conditions(self):
        self.sched.jobs = [{"id": "j1", "kind": "cron", "trigger": "t",
                            "action": "brief", "arg": "", "chain": []}]
        self.sched.conditions = [{"id": "c1", "condition": "x", "action": "y",
                                  "arg": "", "chain": []}]
        act = self.mod._make_list(self.sched)
        out = act("")
        self.assertIn("j1", out)
        self.assertIn("c1", out)

    def test_status_running(self):
        act = self.mod._make_status(self.sched)
        out = act("")
        self.assertIn("running", out.lower())
        self.assertIn("2 job", out)


class SchedulePreflightTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("schedule_manager")

    def test_preflight_install_hint_when_unavailable(self):
        self.mod._bootstrap_error = None
        out = self.mod._preflight(FakeScheduler(available=False))
        self.assertIsNotNone(out)
        self.assertIn("apscheduler", out.lower())

    def test_preflight_bootstrap_failure_surfaces_error(self):
        self.mod._bootstrap_error = "No module named 'sqlalchemy'"
        out = self.mod._preflight(FakeScheduler(available=True))
        self.assertIn("bootstrap failed", out.lower())
        self.assertIn("sqlalchemy", out.lower())

    def test_preflight_ok_returns_none(self):
        self.mod._bootstrap_error = None
        self.assertIsNone(self.mod._preflight(FakeScheduler(available=True)))

    def test_action_returns_install_hint_when_unavailable(self):
        # An action factory bound to an unavailable scheduler returns the hint
        # instead of attempting to schedule.
        self.mod._bootstrap_error = None
        act = self.mod._make_recurring(FakeScheduler(available=False))
        out = act("8am | brief")
        self.assertIn("apscheduler", out.lower())


class ScheduleHelperEdgeTests(unittest.TestCase):
    """Remaining branches in the pure parsing/formatting helpers."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("schedule_manager")

    def test_bootstrap_failure_message_apscheduler_hint(self):
        # ModuleNotFound for apscheduler → install hint for apscheduler.
        self.mod._bootstrap_error = "ModuleNotFoundError: No module named 'apscheduler'"
        out = self.mod._bootstrap_failure_message()
        self.assertIn("pip install apscheduler", out)

    def test_bootstrap_failure_message_unknown(self):
        self.mod._bootstrap_error = None
        out = self.mod._bootstrap_failure_message()
        self.assertIn("unknown bootstrap failure", out)

    def test_split_action_and_arg_empty_token(self):
        self.assertEqual(self.mod._split_action_and_arg("   "), ("", ""))

    def test_format_conditions_with_arg_and_chain(self):
        out = self.mod._format_conditions([{
            "id": "c2", "condition": "credits_low", "action": "warn",
            "arg": "now", "chain": [{"action": "x", "arg": ""}],
            "one_shot": False, "current_value": None}])
        self.assertIn("'now'", out)         # arg branch (134)
        self.assertIn("chained step", out)  # chain branch (136)
        self.assertNotIn("currently", out)  # current_value None → no note


class ScheduleParseCronPhraseEdgeTests(unittest.TestCase):
    """_parse_cron_phrase dow/clock split corner cases."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("schedule_manager")
        self.sched = FakeScheduler()

    def test_leading_text_not_a_dow_falls_back_to_whole_clock(self):
        # tokens[-1] ('8am') parses as a clock, but the leading 'random words'
        # isn't a dow → clock_str resets to the whole body, which then fails to
        # parse → ValueError (covers 216-219 bail path).
        with self.assertRaises(ValueError):
            self.mod._parse_cron_phrase(
                self.sched, "random words 8am", "brief", "", [])

    def test_two_token_clock_with_dow(self):
        # Last token ('pm') alone isn't a clock; last TWO ('8:30 pm') are, and
        # the remaining 'weekdays' is a dow (covers 220-223).
        self.mod._parse_cron_phrase(
            self.sched, "weekdays 8:30 pm", "brief", "", [])
        kind, kw = self.sched.calls[-1]
        self.assertEqual((kw["hour"], kw["minute"]), (20, 30))
        self.assertEqual(kw["day_of_week"], "mon-fri")


class ScheduleActionErrorBranchTests(unittest.TestCase):
    """Preflight + error branches across every action factory."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("schedule_manager")
        self.mod._bootstrap_error = None

    def _unavailable(self):
        return FakeScheduler(available=False)

    def test_recurring_empty_primary_action(self):
        # rhs is only '&&' separators → no primary action (covers 161).
        act = self.mod._make_recurring(FakeScheduler())
        out = act("8am | &&")
        self.assertIn("Format:", out)

    def test_recurring_generic_exception(self):
        sched = FakeScheduler()
        sched.schedule_cron = lambda **kw: (_ for _ in ()).throw(RuntimeError("db down"))
        act = self.mod._make_recurring(sched)
        out = act("8am | brief")
        self.assertIn("schedule failed", out.lower())
        self.assertIn("RuntimeError", out)

    def test_once_preflight_blocks(self):
        out = self.mod._make_once(self._unavailable())("in 30 minutes | brief")
        self.assertIn("apscheduler", out.lower())

    def test_once_missing_pipe(self):
        out = self.mod._make_once(FakeScheduler())("just text no pipe")
        self.assertIn("Format:", out)

    def test_once_schedule_raises(self):
        sched = FakeScheduler()
        sched.schedule_once = lambda **kw: (_ for _ in ()).throw(OSError("io"))
        out = self.mod._make_once(sched)("in 30 minutes | brief")
        self.assertIn("schedule failed", out.lower())

    def test_when_preflight_blocks(self):
        out = self.mod._make_when(self._unavailable())("credits_low | warn")
        self.assertIn("apscheduler", out.lower())

    def test_when_value_error(self):
        sched = FakeScheduler()
        sched.schedule_when = lambda **kw: (_ for _ in ()).throw(ValueError("bad cond"))
        out = self.mod._make_when(sched)("credits_low | warn")
        self.assertIn("could not arm trigger", out.lower())
        self.assertIn("bad cond", out)

    def test_when_generic_exception(self):
        sched = FakeScheduler()
        sched.schedule_when = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        out = self.mod._make_when(sched)("credits_low | warn")
        self.assertIn("trigger failed", out.lower())

    def test_list_preflight_blocks(self):
        out = self.mod._make_list(self._unavailable())("")
        self.assertIn("apscheduler", out.lower())

    def test_cancel_preflight_blocks(self):
        out = self.mod._make_cancel(self._unavailable())("cron_x")
        self.assertIn("apscheduler", out.lower())

    def test_fire_preflight_blocks(self):
        out = self.mod._make_fire(self._unavailable())("cron_x")
        self.assertIn("apscheduler", out.lower())

    def test_fire_requires_id(self):
        out = self.mod._make_fire(FakeScheduler())("")
        self.assertIn("Format:", out)

    def test_status_preflight_blocks(self):
        out = self.mod._make_status(self._unavailable())("")
        self.assertIn("apscheduler", out.lower())

    def test_status_includes_last_error(self):
        sched = FakeScheduler()
        sched.status = lambda: {
            "running": False, "job_count": 0, "condition_count": 0,
            "registered_conditions": [], "last_error": "boom"}
        out = self.mod._make_status(sched)("")
        self.assertIn("stopped", out.lower())
        self.assertIn("Last error: boom", out)


class _FakeSchedulerModule:
    """A stand-in for the `core.scheduler` MODULE that register() imports."""

    def __init__(self, *, available=True, bootstrap_ok=True,
                 bootstrap_raises=False, last_error=None):
        self._available = available
        self._bootstrap_ok = bootstrap_ok
        self._bootstrap_raises = bootstrap_raises
        self._last_error = last_error
        self.bootstrap_called_with = None

    def is_available(self):
        return self._available

    def bootstrap(self, actions):
        self.bootstrap_called_with = actions
        if self._bootstrap_raises:
            raise RuntimeError("bootstrap kaboom")
        return self._bootstrap_ok

    def status(self):
        return {"last_error": self._last_error}


class ScheduleRegisterTests(unittest.TestCase):
    """register(): import failure + every bootstrap branch, with a fake
    `core.scheduler` module injected into sys.modules."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("schedule_manager")

    def _register_with(self, fake_module):
        import contextlib
        import io
        import core  # the real package; we swap its `scheduler` attribute
        actions = {}
        buf = io.StringIO()
        # register() does `from core import scheduler`, which binds the
        # `scheduler` attribute off the already-imported `core` package — so
        # patch that attribute, not sys.modules.
        with mock.patch.object(core, "scheduler", fake_module, create=True), \
             contextlib.redirect_stdout(buf):
            self.mod.register(actions)
        return actions, buf.getvalue()

    def test_register_scheduler_import_fails(self):
        # `from core import scheduler` raising → prints the diagnostic, returns
        # early, registers no actions. We force the failure by removing the
        # cached `scheduler` attribute off `core` and poisoning the submodule
        # entry in sys.modules so the re-import raises ImportError.
        import contextlib
        import io
        import sys
        import core
        actions = {}
        buf = io.StringIO()
        had_attr = hasattr(core, "scheduler")
        saved_attr = getattr(core, "scheduler", None)
        saved_mod = sys.modules.get("core.scheduler")
        try:
            if had_attr:
                delattr(core, "scheduler")
            sys.modules["core.scheduler"] = None  # makes import raise ImportError
            with contextlib.redirect_stdout(buf):
                self.mod.register(actions)
        finally:
            if saved_mod is not None:
                sys.modules["core.scheduler"] = saved_mod
            else:
                sys.modules.pop("core.scheduler", None)
            if had_attr:
                core.scheduler = saved_attr
        self.assertNotIn("schedule_recurring", actions)
        self.assertIn("unavailable", buf.getvalue())

    def test_register_happy_bootstrap(self):
        fake = _FakeSchedulerModule(available=True, bootstrap_ok=True)
        actions, _ = self._register_with(fake)
        self.assertIn("schedule_recurring", actions)
        self.assertIs(fake.bootstrap_called_with, actions)
        self.assertIsNone(self.mod._bootstrap_error)

    def test_register_bootstrap_raises_records_error(self):
        fake = _FakeSchedulerModule(available=True, bootstrap_raises=True)
        actions, out = self._register_with(fake)
        self.assertIn("schedule_recurring", actions)  # still registers
        self.assertIn("RuntimeError", self.mod._bootstrap_error)

    def test_register_bootstrap_false_pulls_status_error(self):
        fake = _FakeSchedulerModule(
            available=True, bootstrap_ok=False,
            last_error="No module named 'sqlalchemy'")
        actions, out = self._register_with(fake)
        self.assertIn("sqlalchemy", self.mod._bootstrap_error)
        self.assertIn("pip install sqlalchemy", out)  # the sqlalchemy hint

    def test_register_bootstrap_false_status_raises(self):
        fake = _FakeSchedulerModule(available=True, bootstrap_ok=False)
        fake.status = lambda: (_ for _ in ()).throw(RuntimeError("status boom"))
        actions, out = self._register_with(fake)
        self.assertIn("status() raised", self.mod._bootstrap_error)

    def test_register_apscheduler_unavailable(self):
        fake = _FakeSchedulerModule(available=False)
        actions, out = self._register_with(fake)
        self.assertIn("schedule_recurring", actions)
        self.assertIn("APScheduler not installed", out)


if __name__ == "__main__":
    unittest.main()
