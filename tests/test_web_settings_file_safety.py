"""The web dashboard's settings file handling (v2.0.146, 2026-09-30).

The Settings window audit found that a user_settings.json it could not parse
(a PowerShell BOM, a trailing comma) was treated as EMPTY and the next Save
wiped every key the panel does not manage. The Settings window now refuses; the
web panel had the same reader and must refuse too, and must accept a BOM.

    python -m unittest tests.test_web_settings_file_safety
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unittest

from tools import settings_window as sw
from tools import web_interface as wi


class WebSettingsFileSafetyTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="websettings_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "user_settings.json")

    def _write(self, text, encoding="utf-8"):
        with open(self.path, "w", encoding=encoding) as f:
            f.write(text)

    def test_a_corrupt_file_is_refused_and_left_untouched(self):
        broken = '{"AI_BACKEND": "ollama", "CAMERAS": [1, 2],}'   # trailing comma
        self._write(broken)
        with self.assertRaises(wi.SettingsWriteError) as cm:
            wi._write_settings({"FAST_PATHS_ENABLED": False}, self.path)
        self.assertIn("can't be read", str(cm.exception))
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), broken)

    def test_a_bom_file_is_read_and_merged(self):
        self._write('{"AI_BACKEND": "ollama", "CAMERAS": ["x"]}', "utf-8-sig")
        self.assertEqual(wi._read_saved_settings(self.path)["AI_BACKEND"], "ollama")
        wi._write_settings({"FAST_PATHS_ENABLED": False}, self.path)
        with open(self.path, encoding="utf-8-sig") as f:
            data = json.load(f)
        self.assertEqual(data["CAMERAS"], ["x"])          # unmanaged key kept
        self.assertIs(data["FAST_PATHS_ENABLED"], False)

    def test_a_missing_file_is_created(self):
        wi._write_settings({"FAST_PATHS_ENABLED": False}, self.path)
        with open(self.path, encoding="utf-8") as f:
            self.assertIs(json.load(f)["FAST_PATHS_ENABLED"], False)

    def test_the_read_only_overlay_still_degrades_to_empty(self):
        self._write("{not json")
        self.assertEqual(wi._read_saved_settings(self.path), {})


class WebSettingsTabTitleTests(unittest.TestCase):
    def test_every_settings_tab_has_a_web_title(self):
        page = wi.DASHBOARD_HTML if hasattr(wi, "DASHBOARD_HTML") else None
        if page is None:
            page = open(wi.__file__, encoding="utf-8").read()
        block = re.search(r"const TAB_TITLES = \{(.*?)\};", page, re.S)
        self.assertIsNotNone(block, "TAB_TITLES not found in the dashboard")
        for tab in sw.TAB_ORDER:
            with self.subTest(tab=tab):
                self.assertRegex(block.group(1), r"\b%s\s*:" % re.escape(tab))


class SkillRoutesRowTests(unittest.TestCase):
    def test_the_route_kill_switch_has_a_settings_row(self):
        self.assertIn("SKILL_ROUTES_ENABLED", sw.SCHEMA if hasattr(sw, "SCHEMA")
                      else sw.default_settings())
        self.assertIs(sw.default_settings()["SKILL_ROUTES_ENABLED"], True)


if __name__ == "__main__":
    unittest.main()
