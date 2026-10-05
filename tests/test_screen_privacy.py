"""core/screen_privacy.py - ONE rule for every screen reader (2026-10-05).

    python -m unittest tests.test_screen_privacy
"""
from __future__ import annotations

import os
import re
import unittest
from unittest import mock

from core import config as cfg
from core import screen_privacy as P


class _Base(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(cfg, "SCREENSHOT_PRIVACY_BLOCKLIST",
                              ["1password", "bitwarden", "keepass", "banking"])
        p.start()
        self.addCleanup(p.stop)
        P.clear_exclusions()
        self.addCleanup(P.clear_exclusions)


class TitleReasonTests(_Base):
    def test_owner_blocklist_substring(self):
        self.assertEqual(P.title_reason("My Vault - 1Password"), "1password")
        self.assertEqual(P.title_reason("x", "KeePass.exe"), "keepass")

    def test_a_bank_caught_by_its_url_only(self):
        # A bank site whose window title doesn't say "banking".
        self.assertTrue(P.title_reason("Accounts - Google Chrome", "chrome.exe",
                                       "https://secure.chase.com/web/auth"))
        self.assertTrue(P.title_reason("Home", "chrome.exe",
                                       "https://www.example-bank.com/banking/"))

    def test_defaults_apply_with_an_empty_owner_list(self):
        with mock.patch.object(cfg, "SCREENSHOT_PRIVACY_BLOCKLIST", []):
            self.assertTrue(P.title_reason("Bitwarden"))
            self.assertTrue(P.title_reason("Authenticator app"))

    def test_ordinary_windows(self):
        self.assertIsNone(P.title_reason("Home - YouTube - Google Chrome",
                                         "chrome.exe",
                                         "https://www.youtube.com/"))
        self.assertIsNone(P.title_reason("report.docx - Word", "winword.exe"))

    def test_the_ambient_regexes_keep_working(self):
        from skills import ambient_listen as AL  # noqa: F401 - import check
        self.assertIs(AL._DEFAULT_SCREEN_BLOCKLIST, P.DEFAULT_PATTERNS)
        pats = [re.compile(p) for p in AL._DEFAULT_SCREEN_BLOCKLIST]
        self.assertTrue(AL._is_sensitive_window("Chase Online Banking",
                                                "chrome.exe", pats))


class WindowPrivateTests(_Base):
    def test_a_password_box_makes_a_window_private(self):
        w = {"hwnd": 5, "title": "Example", "process": "chrome.exe"}
        self.assertIsNone(P.window_private(w))
        self.assertTrue(P.window_private(w, has_password=True))
        self.assertTrue(P.window_private(dict(w, has_password=True)))

    def test_a_sign_in_page(self):
        w = {"hwnd": 5, "title": "Sign in - Google Accounts - Google Chrome",
             "url": "https://accounts.google.com/v3/signin"}
        self.assertTrue(P.window_private(w))

    def test_owner_exclusions(self):
        w = {"hwnd": 77, "title": "Chat - Discord", "process": "Discord.exe"}
        self.assertIsNone(P.window_private(w))
        P.exclude_app("discord")
        self.assertEqual(P.window_private(w), "excluded by the owner")
        P.clear_exclusions()
        P.exclude_window(77)
        self.assertEqual(P.excluded(w), "excluded by the owner")

    def test_an_excluded_site_by_host(self):
        P.exclude_window(1, url="https://news.example.org/a")
        self.assertTrue(P.excluded({"hwnd": 2, "url":
                                    "https://news.example.org/b"}))

    def test_never_raises(self):
        self.assertIsNone(P.window_private(None) or None)


class RegionGateTests(_Base):
    WIN = {"hwnd": 1, "rect": (0, 0, 2560, 1400), "title": "Home - YouTube",
           "private": None}
    SECRET = {"hwnd": 2, "rect": (100, 100, 400, 300), "title": "Vault - "
              "1Password", "private": "1password"}

    def test_an_unfocused_private_window_above_the_target_is_masked(self):
        g = P.region_gate((0, 0, 2560, 1400), [self.SECRET, self.WIN],
                          target_hwnd=1)
        self.assertTrue(g.allowed)
        self.assertEqual(len(g.masks), 1)
        mx, my, mw, mh = g.masks[0]
        self.assertLessEqual(mx, 100 - 8 + 0.01)
        self.assertGreaterEqual(mw, 400 + 16 - 0.01)

    def test_a_private_window_below_the_target_is_covered_by_it(self):
        g = P.region_gate((0, 0, 2560, 1400), [self.WIN, self.SECRET],
                          target_hwnd=1)
        self.assertTrue(g.allowed)
        self.assertEqual(g.masks, [])

    def test_a_private_target_is_refused(self):
        g = P.region_gate((100, 100, 400, 300), [self.SECRET], target_hwnd=2)
        self.assertFalse(g.allowed)

    def test_mostly_private_is_refused(self):
        big = dict(self.SECRET, rect=(0, 0, 2400, 1300))
        g = P.region_gate((0, 0, 2560, 1400), [big, self.WIN])
        self.assertFalse(g.allowed)

    def test_whole_desktop_refused_if_any_private_window_is_visible(self):
        g = P.region_gate((0, 0, 7680, 2880), [self.WIN, self.SECRET],
                          whole_desktop=True)
        self.assertFalse(g.allowed)

    def test_masks_are_applied(self):
        from PIL import Image
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        out = P.apply_masks(img, [(10, 10, 20, 20)], origin=(0, 0))
        self.assertEqual(out.getpixel((15, 15)), (0, 0, 0))
        self.assertEqual(out.getpixel((100, 100)), (255, 255, 255))


class ReadsBlockedTests(unittest.TestCase):
    def test_a_test_process_never_reads_the_real_screen(self):
        # tests/__init__.py sets JARVIS_NO_SCREEN_READ=1.
        self.assertTrue(P.reads_blocked())
        from core import screen_scope, uia_host
        self.assertEqual(screen_scope.visible_windows(), [])
        ok, why = uia_host.call(lambda u: 1, timeout_s=0.1)
        self.assertFalse(ok)

    def test_the_switch(self):
        with mock.patch.dict(os.environ, {"JARVIS_NO_SCREEN_READ": "0",
                                          "JARVIS_TEST_MODE": "0"}):
            self.assertFalse(P.reads_blocked())


if __name__ == "__main__":
    unittest.main()
