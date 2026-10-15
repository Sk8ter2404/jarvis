"""hud/holo_workshop_canvas.py: {"mode": "off"} hides the window, a later
{"mode": "on"} shows it again (it used to destroy the window, so "on" had
nothing to bring back - B7). Fakes only: no Tk window is created."""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from unittest import mock

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_PROJECT, "hud", "holo_workshop_canvas.py")


class _FakeRoot:
    def __init__(self):
        self._state = "normal"
        self.destroyed = False
        self.calls = []

    def state(self):
        return self._state

    def withdraw(self):
        self.calls.append("withdraw")
        self._state = "withdrawn"

    def deiconify(self):
        self.calls.append("deiconify")
        self._state = "normal"

    def attributes(self, *a):
        self.calls.append(("attributes",) + a)

    def destroy(self):
        self.destroyed = True


@unittest.skipUnless(importlib.util.find_spec("tkinter"), "tkinter unavailable")
class WorkshopOffOnTests(unittest.TestCase):
    def setUp(self):
        name = "_holo_workshop_toggle_under_test"
        spec = importlib.util.spec_from_file_location(name, _PATH)
        self.mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.mod
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(self.mod)
        cls = self.mod.WorkshopCanvas
        with mock.patch.object(self.mod.tk, "Tk"), \
                mock.patch.object(self.mod.tk, "Canvas"), \
                mock.patch.object(cls, "tick"):
            self.cv = cls(0, 0, 320, 320, 0)
        self.cv.root = _FakeRoot()
        self.cv.canvas = mock.Mock()

    def _tick(self, workshop):
        files = {self.mod.HUD_STATE_FILE: {"state": "Idle"},
                 self.mod.WORKSHOP_FILE: workshop}
        with mock.patch.object(self.mod, "_read_json",
                               side_effect=lambda p: dict(files.get(p, {}))), \
                mock.patch.object(self.mod, "_is_parent_alive", return_value=True):
            self.cv._tick_body()

    def test_off_hides_but_does_not_destroy(self):
        self._tick({"mode": "off"})
        self.assertEqual(self.cv.root._state, "withdrawn")
        self.assertFalse(self.cv.root.destroyed)
        self.assertFalse(self.cv._closing)

    def test_on_after_off_shows_the_same_window_again(self):
        self._tick({"mode": "off"})
        self._tick({"mode": "off"})            # idempotent: one withdraw only
        self.assertEqual(self.cv.root.calls.count("withdraw"), 1)
        self._tick({"mode": "on"})
        self.assertEqual(self.cv.root._state, "normal")
        self.assertIn("deiconify", self.cv.root.calls)
        self.assertFalse(self.cv.root.destroyed)

    def test_force_visible_overrides_off(self):
        self._tick({"mode": "off"})
        self._tick({"mode": "off", "force_visible": True})
        self.assertEqual(self.cv.root._state, "normal")

    def test_visible_window_is_not_touched_every_tick(self):
        self._tick({"mode": "on"})
        self.assertNotIn("deiconify", self.cv.root.calls)
        self.assertNotIn("withdraw", self.cv.root.calls)

    def test_dead_parent_still_closes(self):
        with mock.patch.object(self.mod, "_read_json", return_value={}), \
                mock.patch.object(self.mod, "_is_parent_alive", return_value=False):
            self.cv._tick_body()
        self.assertTrue(self.cv.root.destroyed)


if __name__ == "__main__":
    unittest.main()
