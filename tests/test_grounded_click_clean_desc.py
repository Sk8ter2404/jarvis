"""W05 (v3): grounded_click._clean_desc strips the wrapper words the brain adds
(live c015 / c027: the long target came back "not found").

    python -m unittest tests.test_grounded_click_clean_desc
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import config as cfg
from core import grounded_click as gc


class CleanDescTests(unittest.TestCase):
    def test_c015_omnitrix(self):
        self.assertEqual(gc._clean_desc("that Omnitrix video for me"),
                         "Omnitrix video")
        self.assertEqual(gc._clean_desc("that Omnitrix video"), "Omnitrix video")

    def test_c027_says_wrapper(self):
        self.assertEqual(
            gc._clean_desc('that video that says, Can I find the rare daddy bear?'),
            "Can I find the rare daddy bear?")
        self.assertEqual(
            gc._clean_desc('the video that says, "Can I find the rare daddy bear?"'),
            "Can I find the rare daddy bear?")

    def test_monitor_words_are_left_for_the_scope_parser(self):
        self.assertEqual(gc._clean_desc("the MrBeast video on the left monitor"),
                         "MrBeast video on the left monitor")

    def test_positional_kept(self):
        self.assertEqual(gc._clean_desc("the first video"), "first video")
        self.assertEqual(gc._clean_desc("the video"), "video")
        self.assertEqual(gc._clean_desc("that"), "that")

    def test_plain_untouched(self):
        self.assertEqual(gc._clean_desc("Subscribe"), "Subscribe")
        self.assertEqual(gc._clean_desc('"Sign in".'), "Sign in")

    def test_setting_off_is_old_behaviour(self):
        with mock.patch.object(cfg, "ARG_FIXES", False):
            self.assertEqual(gc._clean_desc("that Omnitrix video for me"),
                             "that Omnitrix video for me")


if __name__ == "__main__":
    unittest.main()
