"""Logic tests for skills/morning_chain.py.

The chain is the single controller that picks ONE of arrival/handoff/briefing
per wake event. Tests cover the pure selection precedence (config by_weekday →
config default → env var → time-of-day fallback), skill-name normalisation,
the on-disk same-day-fired reads, and the morning_chain_pick debug action.
The wake-watcher daemon is neutered by the harness (threads no-op on start).
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _LoopBreak(BaseException):
    """Break the wake-watcher's `while True` from a stubbed sleep without being
    caught by the loop's `except Exception`."""


class MorningChainTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_chain")

    # ── _normalize_skill (pure) ──────────────────────────────────────────
    def test_normalize_strips_prefix_and_validates(self):
        n = self.mod._normalize_skill
        self.assertEqual(n("arrival"), "arrival")
        self.assertEqual(n("morning_handoff"), "handoff")
        self.assertEqual(n("  BRIEFING "), "briefing")
        self.assertIsNone(n("nonsense"))
        self.assertIsNone(n(None))
        self.assertIsNone(n(123))

    # ── _choose_skill_for_today precedence ───────────────────────────────
    def test_choose_time_of_day_fallback(self):
        # No config file, no env var → falls to time-of-day boundaries.
        with mock.patch.object(self.mod, "_load_chain_config", return_value={}), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEFAULT_MORNING_SKILL", None)
            self.assertEqual(self.mod._choose_skill_for_today(6), "arrival")   # < 8
            self.assertEqual(self.mod._choose_skill_for_today(9), "handoff")   # < 10
            self.assertEqual(self.mod._choose_skill_for_today(11), "briefing")  # else

    def test_choose_config_by_weekday_wins(self):
        today = __import__("time").strftime("%A").lower()
        cfg = {"by_weekday": {today: "morning_arrival"}, "default": "briefing"}
        with mock.patch.object(self.mod, "_load_chain_config", return_value=cfg):
            # by_weekday outranks default AND time-of-day.
            self.assertEqual(self.mod._choose_skill_for_today(11), "arrival")

    def test_choose_config_default_over_env_and_tod(self):
        cfg = {"default": "handoff"}
        with mock.patch.object(self.mod, "_load_chain_config", return_value=cfg), \
             mock.patch.dict(os.environ, {"DEFAULT_MORNING_SKILL": "arrival"}):
            self.assertEqual(self.mod._choose_skill_for_today(6), "handoff")

    def test_choose_env_var_over_tod(self):
        with mock.patch.object(self.mod, "_load_chain_config", return_value={}), \
             mock.patch.dict(os.environ, {"DEFAULT_MORNING_SKILL": "briefing"}):
            # 6am would otherwise be arrival; env var forces briefing.
            self.assertEqual(self.mod._choose_skill_for_today(6), "briefing")

    def test_choose_ignores_garbage_config_values(self):
        cfg = {"by_weekday": {"someday": "bogus"}, "default": "also_bogus"}
        with mock.patch.object(self.mod, "_load_chain_config", return_value=cfg), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEFAULT_MORNING_SKILL", None)
            # All invalid → time-of-day fallback still produces a valid pick.
            self.assertEqual(self.mod._choose_skill_for_today(7), "arrival")

    # ── _load_chain_config graceful degradation ──────────────────────────
    def test_load_config_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._load_chain_config(), {})

    def test_load_config_bad_json(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{not json")):
            # Parse error → {} (and the print is swallowed by capture).
            self.assertEqual(self.mod._load_chain_config(), {})

    def test_load_config_non_dict_returns_empty(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="[1, 2, 3]")):
            self.assertEqual(self.mod._load_chain_config(), {})

    # ── _skill_already_fired_today (on-disk read) ────────────────────────
    def test_already_fired_unknown_skill(self):
        self.assertFalse(self.mod._skill_already_fired_today("does_not_exist"))

    def test_already_fired_json_today(self):
        import time
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open",
                        mock.mock_open(read_data='{"last_fired_date": "%s"}' % today)):
            self.assertTrue(self.mod._skill_already_fired_today("handoff"))

    def test_already_fired_text_format_mismatch(self):
        # briefing uses the raw-text flag file; a stale date is "not today".
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="1999-01-01")):
            self.assertFalse(self.mod._skill_already_fired_today("briefing"))

    def test_already_fired_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertFalse(self.mod._skill_already_fired_today("arrival"))

    # ── morning_chain_pick action ────────────────────────────────────────
    def test_pick_action_reports_choice_and_fired_map(self):
        with mock.patch.object(self.mod, "_choose_skill_for_today", return_value="handoff"), \
             mock.patch.object(self.mod, "_skill_already_fired_today", return_value=False):
            out = self.actions["morning_chain_pick"]("")
        self.assertIn("handoff", out)
        self.assertIn("fired today", out)
        # Mentions all three skills' fired-state in the dict repr.
        self.assertIn("arrival", out)
        self.assertIn("briefing", out)

    def test_already_fired_read_error_is_false(self):
        # open() blowing up mid-read → swallowed, treated as "not fired" (168-9).
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("io error")):
            self.assertFalse(self.mod._skill_already_fired_today("handoff"))

    def test_already_fired_text_format_today(self):
        # The briefing flag is raw-text; today's date string counts as fired.
        import time
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=today + "\n")):
            self.assertTrue(self.mod._skill_already_fired_today("briefing"))

    def test_already_fired_json_non_dict_is_false(self):
        # JSON that isn't a dict (e.g. a list) → not fired.
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="[1,2,3]")):
            self.assertFalse(self.mod._skill_already_fired_today("arrival"))


class MorningChainImportSkillTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_chain")

    def test_import_via_skills_package(self):
        sentinel = types.ModuleType("skills.morning_arrival")
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=sentinel) as imp:
            got = self.mod._import_skill("arrival")
        self.assertIs(got, sentinel)
        imp.assert_called_once_with("skills.morning_arrival")

    def test_import_falls_back_to_bare_name(self):
        sentinel = types.ModuleType("morning_handoff")

        def _imp(name):
            if name == "skills.morning_handoff":
                raise ImportError("no skills pkg here")
            if name == "morning_handoff":
                return sentinel
            raise AssertionError(f"unexpected import {name}")
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=_imp):
            got = self.mod._import_skill("handoff")
        self.assertIs(got, sentinel)

    def test_import_both_paths_fail_returns_none(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("nope")):
            self.assertIsNone(self.mod._import_skill("briefing"))


class MorningChainInvokeSkillTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_chain")

    def test_invoke_returns_false_when_import_none(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertFalse(self.mod._invoke_skill("arrival", "reason"))

    def test_invoke_returns_false_when_no_entry(self):
        mod = types.ModuleType("morning_arrival")   # no _fire_from_chain attr
        with mock.patch.object(self.mod, "_import_skill", return_value=mod):
            self.assertFalse(self.mod._invoke_skill("arrival", "reason"))

    def test_invoke_returns_false_when_entry_not_callable(self):
        mod = types.ModuleType("morning_handoff")
        mod._fire_from_chain = "not a function"
        with mock.patch.object(self.mod, "_import_skill", return_value=mod):
            self.assertFalse(self.mod._invoke_skill("handoff", "reason"))

    def test_invoke_calls_entry_with_reason_and_returns_true(self):
        mod = types.ModuleType("morning_briefing")
        called = {}
        mod._fire_from_chain = lambda reason: called.setdefault("reason", reason)
        with mock.patch.object(self.mod, "_import_skill", return_value=mod):
            self.assertTrue(self.mod._invoke_skill("briefing", "chain pick"))
        self.assertEqual(called["reason"], "chain pick")

    def test_invoke_returns_false_when_entry_raises(self):
        mod = types.ModuleType("morning_arrival")

        def _boom(reason):
            raise RuntimeError("fire failed")
        mod._fire_from_chain = _boom
        with mock.patch.object(self.mod, "_import_skill", return_value=mod):
            self.assertFalse(self.mod._invoke_skill("arrival", "reason"))


class MorningChainWatcherTests(unittest.TestCase):
    """Drive the wake-watcher loop deterministically. bobert_companion is a
    fake module; time/localtime/strftime are pinned; a stubbed time.sleep
    raises _LoopBreak to exit after the iteration(s) under test."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_chain")
        self._saved_bc = sys.modules.get("bobert_companion")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved_bc is not None:
            sys.modules["bobert_companion"] = self._saved_bc
        else:
            sys.modules.pop("bobert_companion", None)

    def _fake_bc(self, wake_date):
        bc = types.ModuleType("bobert_companion")
        bc._last_wake_date = [wake_date]
        return bc

    def test_watcher_disabled_when_bc_import_fails(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")), \
             mock.patch.object(self.mod.time, "sleep") as slept:
            self.mod._watch_for_first_wake()   # returns immediately
        slept.assert_not_called()

    def test_watcher_dispatches_chosen_skill_on_morning_wake(self):
        today = "2026-06-02"
        bc = self._fake_bc(today)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod.time, "strftime", return_value=today), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=types.SimpleNamespace(tm_hour=9)), \
             mock.patch.object(self.mod, "_choose_skill_for_today", return_value="handoff"), \
             mock.patch.object(self.mod, "_skill_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_invoke_skill", return_value=True) as inv, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_LoopBreak):
            with self.assertRaises(_LoopBreak):
                self.mod._watch_for_first_wake()
        inv.assert_called_once()
        self.assertEqual(inv.call_args[0][0], "handoff")

    def test_watcher_idle_when_already_fired(self):
        today = "2026-06-02"
        bc = self._fake_bc(today)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod.time, "strftime", return_value=today), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=types.SimpleNamespace(tm_hour=7)), \
             mock.patch.object(self.mod, "_choose_skill_for_today", return_value="arrival"), \
             mock.patch.object(self.mod, "_skill_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_invoke_skill") as inv, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_LoopBreak):
            with self.assertRaises(_LoopBreak):
                self.mod._watch_for_first_wake()
        inv.assert_not_called()

    def test_watcher_skips_outside_morning_window(self):
        # A wake at 14:00 is outside [6,12) → no dispatch.
        today = "2026-06-02"
        bc = self._fake_bc(today)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod.time, "strftime", return_value=today), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=types.SimpleNamespace(tm_hour=14)), \
             mock.patch.object(self.mod, "_invoke_skill") as inv, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_LoopBreak):
            with self.assertRaises(_LoopBreak):
                self.mod._watch_for_first_wake()
        inv.assert_not_called()

    def test_watcher_no_dispatch_when_no_wake_today(self):
        # _last_wake_date is yesterday → wake_date != today → idle.
        bc = self._fake_bc("2026-06-01")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod.time, "strftime", return_value="2026-06-02"), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=types.SimpleNamespace(tm_hour=9)), \
             mock.patch.object(self.mod, "_invoke_skill") as inv, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_LoopBreak):
            with self.assertRaises(_LoopBreak):
                self.mod._watch_for_first_wake()
        inv.assert_not_called()

    def test_watcher_wake_date_read_error_is_swallowed(self):
        # Reading bc._last_wake_date[0] raising is caught (wake_date stays None);
        # the loop continues to the sleep where we break. No crash.
        bc = types.ModuleType("bobert_companion")
        # A property that raises on [0] access: use an object whose __getitem__
        # raises to mimic a maintainer turning _last_wake_date into a bad type.
        class _Bad:
            def __getitem__(self, i):
                raise TypeError("not subscriptable the way you think")
        bc._last_wake_date = _Bad()
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod.time, "strftime", return_value="2026-06-02"), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=types.SimpleNamespace(tm_hour=9)), \
             mock.patch.object(self.mod, "_invoke_skill") as inv, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_LoopBreak):
            with self.assertRaises(_LoopBreak):
                self.mod._watch_for_first_wake()
        inv.assert_not_called()

    def test_watcher_invoke_failure_leaves_day_undispatched(self):
        # If _invoke_skill returns False, dispatched_for_date is NOT set, so a
        # subsequent tick would retry. We run two ticks: invoke fails on the
        # first, then we break on the second sleep — invoke called twice.
        today = "2026-06-02"
        bc = self._fake_bc(today)
        sleeps = {"n": 0}

        def _sleep(_):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise _LoopBreak
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod.time, "strftime", return_value=today), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=types.SimpleNamespace(tm_hour=9)), \
             mock.patch.object(self.mod, "_choose_skill_for_today", return_value="handoff"), \
             mock.patch.object(self.mod, "_skill_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_invoke_skill", return_value=False) as inv, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep):
            with self.assertRaises(_LoopBreak):
                self.mod._watch_for_first_wake()
        self.assertEqual(inv.call_count, 2)

    def test_watcher_tick_exception_is_caught(self):
        # An unexpected error inside the tick body (here _choose_skill_for_today
        # raising) is caught by the loop's except → printed, then sleep → break.
        today = "2026-06-02"
        bc = self._fake_bc(today)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod.time, "strftime", return_value=today), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=types.SimpleNamespace(tm_hour=9)), \
             mock.patch.object(self.mod, "_choose_skill_for_today",
                               side_effect=RuntimeError("decide boom")), \
             mock.patch.object(self.mod, "_skill_already_fired_today", return_value=False), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_LoopBreak):
            with self.assertRaises(_LoopBreak):
                self.mod._watch_for_first_wake()


class MorningChainRegisterTests(unittest.TestCase):
    def test_register_adds_pick_action(self):
        mod, actions = load_skill_isolated("morning_chain")
        self.assertIn("morning_chain_pick", actions)


if __name__ == "__main__":
    unittest.main()
