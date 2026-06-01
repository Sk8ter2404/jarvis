"""Logic tests for skills/morning_chain.py.

The chain is the single controller that picks ONE of arrival/handoff/briefing
per wake event. Tests cover the pure selection precedence (config by_weekday →
config default → env var → time-of-day fallback), skill-name normalisation,
the on-disk same-day-fired reads, and the morning_chain_pick debug action.
The wake-watcher daemon is neutered by the harness (threads no-op on start).
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


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


if __name__ == "__main__":
    unittest.main()
