"""Dashboard accessibility pins (2026-10-13 audit): viewport meta, labelled
groups, keyboard-reachable action names, visible focus, 24 px checkboxes.

    python -B -m unittest tests.test_web_dashboard_a11y_1013
"""
from __future__ import annotations

import re
import unittest

from tools import web_interface as wi

PAGE = wi._DASHBOARD_PAGE


class DashboardA11yTests(unittest.TestCase):
    def test_viewport_meta_and_lang(self):
        self.assertIn('<meta name="viewport" content="width=device-width, initial-scale=1">', PAGE)
        self.assertIn('<html lang="en">', PAGE)
        self.assertNotIn("user-scalable=no", PAGE)

    def test_labelled_regions_have_a_role(self):
        for ident in ("strip", "controls", "actions", "estopDock", "tlList"):
            m = re.search(r'<div[^>]*id="%s"[^>]*>' % ident, PAGE)
            self.assertIsNotNone(m, ident)
            self.assertIn("role=", m.group(0), ident)
            self.assertIn("aria-label=", m.group(0), ident)

    def test_action_name_is_keyboard_operable(self):
        self.assertIn("nm.setAttribute('role','button'); nm.tabIndex=0;", PAGE)
        self.assertIn("nm.addEventListener('keydown'", PAGE)

    def test_focus_is_never_removed_without_an_outline(self):
        self.assertNotIn("outline:none", PAGE)
        self.assertIn(".lrow .nm:focus-visible", PAGE)

    def test_checkbox_target_is_at_least_24px(self):
        self.assertIn("input[type=checkbox] { width:24px; height:24px;", PAGE)

    def test_no_external_resources_or_alert(self):
        self.assertNotRegex(PAGE, r'(src|href)="https?://')
        self.assertNotIn("alert(", PAGE.replace("window.confirm(", ""))


if __name__ == "__main__":
    unittest.main()
