"""NIGHT_OWL_AUTO (v2.0.147, 2026-09-29): the owner asked for night-owl mode to
stop switching itself on at 23:00 (a voice about 15% quieter and a little
slower, one-short-sentence replies, no 'thinking' filler, held non-essential
announcements and a dimmed overlay). The setting gates only the watcher's
AUTOMATIC engage; the manual "night owl on" / "night owl off" actions and the
morning release of a manual engagement are unchanged. The master switch
NIGHT_QUIET_ENABLED (core/night_quiet.py) stops the automatic engage too:
either one off means no auto-engage.

Every test patches both knobs explicitly and the shipped default is read from
the core/config.py SOURCE, so the gitignored data/user_settings.json can never
decide a result.

    python -m unittest tests.skills.test_night_owl_auto
"""
from __future__ import annotations

import ast
import os
import unittest
from unittest import mock

from core import config as cfg
from tests._skill_harness import load_skill_isolated

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _config_literals() -> dict:
    """Top-level constant assignments in core/config.py, from the source text
    (the shipped defaults, before any user_settings.json override)."""
    with open(os.path.join(_ROOT, "core", "config.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    lits = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    lits[tgt.id] = node.value.value
    return lits


class NightOwlAutoSettingTests(unittest.TestCase):

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")
        m = self.mod
        m._opted_out_night[0] = ""
        m._night_owl_active[0] = False
        m._engaged_in_window[0] = False
        m._trigger[0] = ""
        self.enter = mock.patch.object(m, "_enter_night_owl").start()
        self.exit = mock.patch.object(m, "_exit_night_owl").start()
        self.addCleanup(mock.patch.stopall)

    def _tick(self, *, in_window, auto, quiet=True):
        with mock.patch.object(self.mod, "_in_night_window",
                               return_value=in_window), \
             mock.patch.object(cfg, "NIGHT_OWL_AUTO", auto, create=True), \
             mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", quiet, create=True):
            self.mod._watch_tick()

    def test_ships_on(self):
        # The SOURCE default, not the runtime value: core.config applies the
        # gitignored user_settings.json at import, which may say false.
        self.assertIs(_config_literals().get("NIGHT_OWL_AUTO"), True)

    def test_auto_on_engages_at_night(self):
        self._tick(in_window=True, auto=True)
        self.enter.assert_called_once_with(trigger="auto")

    def test_auto_off_never_engages_by_itself(self):
        for _ in range(3):
            self._tick(in_window=True, auto=False)
        self.enter.assert_not_called()

    def test_master_switch_off_never_engages_by_itself(self):
        for _ in range(3):
            self._tick(in_window=True, auto=True, quiet=False)
        self.enter.assert_not_called()

    def test_both_off_never_engages_by_itself(self):
        self._tick(in_window=True, auto=False, quiet=False)
        self.enter.assert_not_called()

    def test_auto_off_still_releases_a_manual_night_in_the_morning(self):
        m = self.mod
        m._night_owl_active[0] = True
        m._trigger[0] = "manual"
        m._engaged_in_window[0] = True
        self._tick(in_window=False, auto=False)
        self.exit.assert_called_once_with(trigger="auto_morning")

    def test_master_switch_off_still_releases_a_manual_night(self):
        m = self.mod
        m._night_owl_active[0] = True
        m._trigger[0] = "manual"
        m._engaged_in_window[0] = True
        self._tick(in_window=False, auto=True, quiet=False)
        self.exit.assert_called_once_with(trigger="auto_morning")

    def test_manual_on_still_works_with_auto_off(self):
        with mock.patch.object(cfg, "NIGHT_OWL_AUTO", False, create=True), \
             mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", True, create=True):
            self.actions["night_owl_on"]("")
        self.enter.assert_called_once_with(trigger="manual")

    def test_manual_on_still_works_with_master_switch_off(self):
        with mock.patch.object(cfg, "NIGHT_OWL_AUTO", True, create=True), \
             mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", False, create=True):
            self.actions["night_owl_on"]("")
        self.enter.assert_called_once_with(trigger="manual")

    def test_unreadable_setting_keeps_the_old_behaviour(self):
        class _Unreadable:
            def __bool__(self):
                raise ValueError("not a bool")
        with mock.patch.object(cfg, "NIGHT_OWL_AUTO", _Unreadable(),
                               create=True), \
             mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", True, create=True):
            self.assertTrue(self.mod._auto_enabled())
        with mock.patch.object(cfg, "NIGHT_OWL_AUTO", True, create=True), \
             mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", _Unreadable(),
                               create=True):
            self.assertTrue(self.mod._auto_enabled())


if __name__ == "__main__":
    unittest.main()
