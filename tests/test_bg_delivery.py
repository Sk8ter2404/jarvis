"""core/bg_delivery.py -- owner-spoke-during-call guard (v3.0.27,
overhaul finding reliability-07).

A background brain call cannot be cancelled once it is out; v3.0.26 only kept
one from starting while the owner talks. Pinned here:
* a mark taken before the call, an owner turn noted during it -> skip;
* a turn / capture in progress at delivery (``active``) -> skip;
* nothing in between -> deliver;
* BG_SKIP_IF_OWNER_SPOKE off (``enabled=False``) -> never skip;
* a skip logs one ``[bg-skip]`` line with the caller tag only;
* an unreadable mark counts as "he spoke" (never talk over him).

Stdlib only: CI-light tier.
"""
from __future__ import annotations

import os
import re
import unittest

from core import bg_delivery as bgd

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GuardTests(unittest.TestCase):
    def test_a_turn_during_the_call_skips(self):
        m = bgd.mark()
        bgd.note_owner_turn()
        logs = []
        self.assertTrue(bgd.should_skip("proactive_remark", m,
                                        log=logs.append))
        self.assertEqual(len(logs), 1)
        self.assertIn("[bg-skip] proactive_remark:", logs[0])

    def test_no_turn_delivers(self):
        m = bgd.mark()
        logs = []
        self.assertFalse(bgd.should_skip("screen_remark", m,
                                         log=logs.append))
        self.assertEqual(logs, [])

    def test_a_turn_in_progress_skips(self):
        m = bgd.mark()
        self.assertTrue(bgd.should_skip("x", m, active=True))

    def test_flag_off_never_skips(self):
        m = bgd.mark()
        bgd.note_owner_turn()
        logs = []
        self.assertFalse(bgd.should_skip("x", m, active=True, enabled=False,
                                         log=logs.append))
        self.assertEqual(logs, [])

    def test_an_unreadable_mark_counts_as_spoke(self):
        self.assertTrue(bgd.owner_spoke_since("not a number"))

    def test_the_caller_tag_is_cleaned(self):
        logs = []
        m = bgd.mark()
        bgd.note_owner_turn()
        bgd.should_skip("a b\nc" * 20, m, log=logs.append)
        tag = logs[0].split("[bg-skip] ", 1)[1].split(":", 1)[0]
        self.assertLessEqual(len(tag), 40)
        self.assertNotIn(" ", tag)
        self.assertNotIn("\n", tag)

    def test_a_raising_log_still_skips(self):
        def boom(_line):
            raise RuntimeError("x")
        m = bgd.mark()
        bgd.note_owner_turn()
        self.assertTrue(bgd.should_skip("x", m, log=boom))


class ConfigTests(unittest.TestCase):
    def test_default_is_on_and_documented(self):
        with open(os.path.join(_PROJECT, "core", "config.py"),
                  encoding="utf-8") as f:
            src = f.read()
        self.assertRegex(src, r"(?m)^BG_SKIP_IF_OWNER_SPOKE = True$")
        block = src.split("# BG_SKIP_IF_OWNER_SPOKE", 1)[1][:900]
        self.assertIn("v3.0.26 behaviour", block)
        self.assertTrue(re.search(r"Learning records are still", block))


if __name__ == "__main__":
    unittest.main()
