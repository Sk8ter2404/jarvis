"""Tests for the skill-isolation harness itself (tests/_skill_harness.py).

Validates that the harness loads representative skills without booting the
monolith, and — critically — that its fake ``skill_utils`` keys stay in sync
with the real loader by AST-parsing ``bobert_companion.py`` (no import needed,
so this runs in the bare CI tier).
"""
from __future__ import annotations

import ast
import os
import unittest

from tests._skill_harness import (
    SKILL_UTILS_KEYS,
    list_skill_names,
    load_skill_isolated,
    make_fake_skill_utils,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MONOLITH = os.path.join(_ROOT, "bobert_companion.py")


class SkillHarnessTests(unittest.TestCase):
    def test_loads_representative_skills(self):
        # A pure-logic skill, a daemon-spawning skill, and a smart-home skill —
        # each must import + register at least one action with no monolith.
        for name, expected in (("timer", "set_timer"), ("banter", None),
                               ("sh_hue", None)):
            with self.subTest(skill=name):
                mod, actions = load_skill_isolated(name)
                self.assertTrue(hasattr(mod, "register"))
                self.assertGreater(len(actions), 0,
                                   f"{name} registered no actions")
                if expected:
                    self.assertIn(expected, actions)

    def test_timer_action_actually_runs(self):
        # Driving a registered handler end-to-end with the fake utils.
        _, actions = load_skill_isolated("timer")
        out = actions["set_timer"]("1 minute | tea")
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip(), "set_timer returned empty")

    def test_fake_skill_utils_has_all_keys_and_is_callable(self):
        u = make_fake_skill_utils()
        self.assertEqual(set(u), set(SKILL_UTILS_KEYS))
        # every value must be callable (skills invoke them)
        for k, v in u.items():
            self.assertTrue(callable(v), f"skill_utils[{k}] not callable")
        # overrides win
        u2 = make_fake_skill_utils(open_url=lambda _u: "stub")
        self.assertEqual(u2["open_url"]("x"), "stub")

    def test_skill_utils_keys_in_sync_with_monolith(self):
        """AST-parse the module-level ``skill_utils = {...}`` in the monolith
        and assert the fake's keys match — a new helper added to the loader
        without updating the harness fails here (light, no import)."""
        with open(_MONOLITH, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        keys = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                if any(isinstance(t, ast.Name) and t.id == "skill_utils"
                       for t in node.targets):
                    keys = [k.value for k in node.value.keys
                            if isinstance(k, ast.Constant)]
        self.assertIsNotNone(keys, "module-level skill_utils dict not found")
        self.assertEqual(set(keys), set(SKILL_UTILS_KEYS),
                         "fake skill_utils keys drifted from the monolith loader")

    def test_list_skill_names_reasonable(self):
        names = list_skill_names()
        self.assertGreater(len(names), 60)
        self.assertNotIn("_example_skill", names)
        self.assertIn("timer", names)


if __name__ == "__main__":
    unittest.main()
