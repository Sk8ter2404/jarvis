"""W05 (v3): open_on_monitor 'secondary' / 'the other one' (live 10-19 19:45).

    python -m unittest tests.test_monitor_resolve_aliases
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import actions
from core import config as cfg

TWO = {"main": (0, 0, 2560, 1440), "side": (2560, 0, 2560, 1440)}
THREE = {"left": (-2560, 0, 2560, 1440), "middle": (0, 0, 2560, 1440),
         "right": (2560, 0, 2560, 1440)}


class ResolveAliasTests(unittest.TestCase):
    def test_two_monitors(self):
        with mock.patch.object(cfg, "MONITORS", TWO):
            for w in ("secondary", "second", "other", "the other one",
                      "the secondary monitor", "Other"):
                self.assertEqual(actions._resolve_monitor(w), "side", w)
            self.assertEqual(actions._resolve_monitor("primary"), "main")
            self.assertEqual(actions._resolve_monitor("side"), "side")

    def test_three_monitors_stay_unknown(self):
        with mock.patch.object(cfg, "MONITORS", THREE):
            self.assertIsNone(actions._resolve_monitor("secondary"))
            self.assertEqual(actions._resolve_monitor("left"), "left")

    def test_setting_off(self):
        with mock.patch.object(cfg, "MONITORS", TWO), \
                mock.patch.object(cfg, "ARG_FIXES", False):
            self.assertIsNone(actions._resolve_monitor("secondary"))


if __name__ == "__main__":
    unittest.main()
