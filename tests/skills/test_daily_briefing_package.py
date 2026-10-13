"""The daily briefing is ONE skill (skills/daily_briefing/) made of parts.

Ten skills became one package on 2026-10-12 (the 08:00 daily briefing, the
morning chain, the arrival cold-open, arrival v2, the handoff, the short
morning briefing, the evening briefing, the daily recap and the news
briefing). These tests pin what must NOT change by the move: every action
name, every state file in the project root, and the way the parts find each
other (the instance the loader registered, never a second copy).
"""
from __future__ import annotations

import importlib
import os
import sys
import unittest
from unittest import mock

from tests._skill_harness import (SKILLS_DIR, load_briefing_part,
                                  load_skill_isolated, no_background_threads)

_ROOT = os.path.dirname(SKILLS_DIR)

# Every action the ten skills registered before the merge, by the part that
# registers it now.
_ACTIONS_BY_PART = {
    "daily": {"daily_briefing"},
    "news": {"news_briefing"},
    "recap": {"daily_recap"},
    "evening": {"evening_briefing"},
    "arrival": {"morning_arrival", "arrival_briefing"},
    "arrival_v2": {"morning_arrival_v2", "arrival_briefing_v2"},
    "light": {"morning_briefing"},
    "chain": {"morning_chain_pick"},
    "handoff": {"morning_handoff", "predictive_morning_setup",
                "setup_workspace", "workspace_setup"},
}


class HubTests(unittest.TestCase):
    def test_register_adds_every_action_the_ten_skills_had(self):
        _mod, actions = load_skill_isolated("daily_briefing")
        want = set().union(*_ACTIONS_BY_PART.values())
        self.assertEqual(set(actions), want)
        self.assertEqual(len(want), 14)

    def test_each_part_registers_exactly_its_own_actions(self):
        for part, want in _ACTIONS_BY_PART.items():
            with self.subTest(part=part):
                _mod, actions = load_briefing_part(part)
                self.assertEqual(set(actions), want)

    def test_the_old_skills_are_gone_so_nothing_loads_twice(self):
        for old in ("morning_arrival", "morning_arrival_v2", "morning_briefing",
                    "morning_chain", "morning_handoff", "evening_briefing",
                    "daily_recap", "news_briefing", "daily_briefing"):
            with self.subTest(old=old):
                self.assertFalse(
                    os.path.exists(os.path.join(SKILLS_DIR, old + ".py")))
        self.assertTrue(os.path.isfile(
            os.path.join(SKILLS_DIR, "daily_briefing", "__init__.py")))
        # weather stays its own skill: forecasts are asked for on their own.
        self.assertTrue(os.path.isfile(
            os.path.join(SKILLS_DIR, "weather_briefing.py")))

    def test_a_part_that_fails_does_not_take_the_others_down(self):
        load_skill_isolated("daily_briefing", register=False)
        pkg = sys.modules["skill_daily_briefing"]
        recap = importlib.import_module("skill_daily_briefing.recap")
        actions: dict = {}
        with mock.patch.object(recap, "register",
                               side_effect=RuntimeError("recap broke")), \
                no_background_threads():
            pkg.register(actions)
        self.assertNotIn("daily_recap", actions)
        for survivor in ("daily_briefing", "evening_briefing",
                         "news_briefing", "morning_arrival",
                         "morning_handoff", "morning_chain_pick"):
            self.assertIn(survivor, actions)


class StateStaysWhereItWasTests(unittest.TestCase):
    """The parts moved one folder deeper. A state file that silently moved
    with them would re-fire a briefing already given today after the update."""

    def test_every_part_still_points_at_the_project_root(self):
        for part in ("daily", "news", "recap", "evening", "arrival",
                     "arrival_v2", "light", "chain", "handoff"):
            with self.subTest(part=part):
                mod, _ = load_briefing_part(part, register=False)
                self.assertEqual(os.path.normcase(mod._PROJECT_DIR),
                                 os.path.normcase(_ROOT))

    def test_state_file_names_and_places_are_unchanged(self):
        want = {
            "daily": ("_STATE_FILE", "daily_briefing_state.json"),
            "recap": ("_STATE_FILE", "daily_recap_state.json"),
            "evening": ("_STATE_FILE", "evening_briefing_state.json"),
            "arrival": ("_STATE_FILE", "morning_arrival_state.json"),
            "arrival_v2": ("_STATE_FILE", "morning_arrival_v2_state.json"),
            "handoff": ("_STATE_FILE", "morning_handoff_state.json"),
            "light": ("_BRIEFING_FLAG_FILE", ".morning_briefing_last"),
            "chain": ("_CONFIG_FILE", "morning_chain_config.json"),
        }
        for part, (attr, name) in want.items():
            with self.subTest(part=part):
                mod, _ = load_briefing_part(part, register=False)
                self.assertEqual(os.path.normcase(getattr(mod, attr)),
                                 os.path.normcase(os.path.join(_ROOT, name)))

    def test_the_skills_folder_constant_finds_the_sibling_skills(self):
        # briefing_sources, ms_graph, weather_briefing ... are separate skills
        # the parts fall back to importing off this folder.
        for part in ("daily", "arrival", "arrival_v2", "handoff", "light",
                     "evening"):
            with self.subTest(part=part):
                mod, _ = load_briefing_part(part, register=False)
                self.assertEqual(os.path.normcase(mod._SKILLS_DIR),
                                 os.path.normcase(SKILLS_DIR))
        self.assertTrue(os.path.isfile(
            os.path.join(SKILLS_DIR, "briefing_sources.py")))


class SiblingLookupTests(unittest.TestCase):
    """The parts used to find each other by guessing sys.modules names and, on
    a miss, importing a SECOND copy off the skills path. They now get the
    instance the loader registered."""

    def test_the_chain_dispatches_to_the_registered_parts(self):
        chain, _ = load_briefing_part("chain", register=False)
        for short, part in (("arrival", "arrival"), ("handoff", "handoff"),
                            ("briefing", "light")):
            with self.subTest(pick=short):
                self.assertIs(chain._import_skill(short),
                              sys.modules[f"skill_daily_briefing.{part}"])

    def test_every_chain_pick_has_the_entry_point_the_chain_calls(self):
        chain, _ = load_briefing_part("chain", register=False)
        for short in chain.SKILL_NAMES:
            with self.subTest(pick=short):
                part = chain._import_skill(short)
                self.assertTrue(callable(getattr(
                    part, chain._SKILL_CHAIN_ENTRY, None)))

    def test_news_briefing_resolves_to_the_news_part_everywhere(self):
        for part in ("arrival", "arrival_v2", "handoff"):
            with self.subTest(part=part):
                mod, _ = load_briefing_part(part, register=False)
                self.assertIs(mod._import_skill("news_briefing"),
                              sys.modules["skill_daily_briefing.news"])

    def test_the_short_and_evening_briefings_read_the_real_news_part(self):
        # No injected fake: the lazy import inside _fetch_news must resolve
        # against the package, or the headlines silently vanish.
        for part in ("light", "evening"):
            with self.subTest(part=part):
                mod, _ = load_briefing_part(part, register=False)
                news = importlib.import_module("skill_daily_briefing.news")
                with mock.patch.object(news, "get_news_text",
                                       return_value="Headlines, sir."):
                    self.assertEqual(mod._fetch_news(), "Headlines, sir.")

    def test_the_chain_v2_and_daily_see_each_others_flags(self):
        chain, _ = load_briefing_part("chain", register=False)
        v2 = importlib.import_module("skill_daily_briefing.arrival_v2")
        with mock.patch.object(v2, "_already_fired_today", return_value=True):
            self.assertTrue(chain._arrival_v2_fired_today())
        with mock.patch.object(v2, "_already_fired_today", return_value=False):
            self.assertFalse(chain._arrival_v2_fired_today())
        pkg = sys.modules["skill_daily_briefing"]
        with mock.patch.object(pkg, "owner_heard_briefing_today",
                               return_value=True):
            self.assertTrue(chain._daily_briefing_heard_today())
            self.assertTrue(v2._daily_briefing_heard_today())
        with mock.patch.object(v2, "_already_fired_today", return_value=True):
            self.assertTrue(pkg._other_morning_briefing_fired_today())
        with mock.patch.object(v2, "_already_fired_today", return_value=False), \
                mock.patch.object(chain, "_skill_already_fired_today",
                                  side_effect=lambda n: n == "handoff"):
            self.assertTrue(pkg._other_morning_briefing_fired_today())


if __name__ == "__main__":
    unittest.main()
