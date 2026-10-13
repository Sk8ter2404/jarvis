"""Every on/off switch the holographic overlay skill reads can really be set.

skills/holographic_overlay reads its auto-launch switches off the monolith with
``getattr(bobert_companion, "NAME", False)``. A name that core/config.py never
defines cannot be switched on from data/user_settings.json (the loader applies
only keys that already exist in core.config), and a name the monolith assigns
again after ``from core.config import *`` ignores user_settings.json too. Three
switches were the first kind and one was the second until 2026-10-12: they were
documented, tested with a fake monolith, and impossible to turn on.
"""
from __future__ import annotations

import ast
import os
import re
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OVERLAY = os.path.join(_ROOT, "skills", "holographic_overlay", "__init__.py")
_SWITCH = re.compile(r"AUTO_LAUNCH$|AUTO_ON_THINK$|AUTO_WHILE_PRINTING$|"
                     r"^HUD_BAMBU_CAMERA$")


def _switches_the_overlay_reads() -> set:
    with open(_OVERLAY, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and _SWITCH.search(node.args[1].value)):
            names.add(node.args[1].value)
    return names


class OverlaySwitchTests(unittest.TestCase):
    def test_the_scan_sees_the_switches(self):
        found = _switches_the_overlay_reads()
        for name in ("HOLOGRAPHIC_OVERLAY_AUTO_LAUNCH",
                     "WORKSHOP_HUD_AUTO_LAUNCH", "HUD_BAMBU_CAMERA",
                     "HOLO_STARK_STATUS_RING_AUTO_LAUNCH"):
            self.assertIn(name, found)
        self.assertGreaterEqual(len(found), 9)

    def test_every_switch_is_defined_in_core_config_off_by_default(self):
        from core import config
        for name in sorted(_switches_the_overlay_reads()):
            with self.subTest(name=name):
                self.assertTrue(hasattr(config, name),
                                f"{name} is read by the overlay but "
                                "core/config.py never defines it, so "
                                "user_settings.json cannot set it")
                self.assertIs(type(getattr(config, name)), bool)
        # The retired surfaces stay opt-in; the camera panel's master is on.
        for name in sorted(_switches_the_overlay_reads()):
            if name != "HUD_BAMBU_CAMERA":
                self.assertFalse(getattr(config, name), name)

    def test_the_monolith_does_not_assign_a_switch_a_second_time(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        for name in sorted(_switches_the_overlay_reads()):
            with self.subTest(name=name):
                self.assertIsNone(
                    re.search(r"^%s\s*=" % re.escape(name), src, re.M),
                    f"bobert_companion.py re-assigns {name} after "
                    "`from core.config import *`, so a user_settings.json "
                    "value is silently overridden")

    def test_a_user_setting_reaches_each_switch(self):
        # What _apply_user_settings does: only an existing bool constant is
        # overridden. Prove the three formerly-missing names qualify.
        from core import config
        for name in ("HOLO_HUD_V2_AUTO_LAUNCH",
                     "HOLO_ARC_REACTOR_STATUS_AUTO_LAUNCH",
                     "HOLO_STARK_STATUS_RING_AUTO_LAUNCH"):
            with self.subTest(name=name):
                self.assertIn(name, vars(config))
                self.assertIsInstance(vars(config)[name], bool)


if __name__ == "__main__":
    unittest.main()
