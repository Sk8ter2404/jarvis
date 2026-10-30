"""core.action_fit.action_topic_missing (v3.0.10, ACTION_TOPIC_GATE).

Live 2026-10-29 18:55:45 the misheard short line "Much more percentage, so
I've got" got [ACTION: server_status] and a 10 s server read-out. A read-only
server / printer check runs only when his line, or the turns just before it,
name its subject. Pure; stdlib unittest only.

    python tools/run_tests.py action_topic
"""
from __future__ import annotations

import unittest

from core import action_fit as af

_LIVE = "Much more percentage, so I've got"


class ActionTopicTests(unittest.TestCase):
    def test_live_line_flags_server_status(self):
        why = af.action_topic_missing("server_status", _LIVE)
        self.assertTrue(why)
        self.assertNotIn("percentage", why)      # no transcript words

    def test_real_asks_fit(self):
        for name, text in (
                ("server_status", "is the server up?"),
                ("server_status", "how's the server"),
                ("server_status", "Jarvis, is unraid online"),
                ("server_status", "how is the rebuild going"),
                ("server_status", "any disks down?"),
                ("check_print", "how's the print?"),
                ("check_print", "check the printer"),
                ("check_print", "how much more percentage"),
                ("how_is_the_print", "how long is left?"),
                ("print_details", "is the bambu done yet")):
            with self.subTest(name=name, text=text):
                self.assertEqual(af.action_topic_missing(name, text), "")

    def test_review_asks_fit(self):
        # v3.0.10 review: clearly heard asks the first word lists refused.
        for text in ("is DNS working?", "can you check SSH",
                     "is the web front end up?",
                     "are my docker containers up?", "is Nextcloud up?",
                     "is the swap done yet?", "are the backups ok?",
                     "is the VM running?",
                     "how much space do I have left?"):
            with self.subTest(text=text):
                self.assertEqual(
                    af.action_topic_missing("server_status", text), "")
        for text in ("how much longer?", "how far along is it?",
                     "how's the job going?", "is the plate ready?",
                     "how's the engraving going",
                     "what time will it be ready?", "check on the job"):
            with self.subTest(text=text):
                self.assertEqual(
                    af.action_topic_missing("check_print", text), "")

    def test_print_check_on_the_live_line_fits(self):
        # "percentage" names a print's progress: not blocked.
        self.assertEqual(af.action_topic_missing("check_print", _LIVE), "")

    def test_context_lets_a_follow_up_run(self):
        self.assertEqual(af.action_topic_missing(
            "server_status", "how about now?",
            ["is the server up?", "[ACTION: server_status]"]), "")
        self.assertEqual(af.action_topic_missing(
            "server_status", "and now?", ["server_status"]), "")
        self.assertEqual(af.action_topic_missing(
            "how_is_the_print", "and now?", ["[ACTION: check_print]"]), "")
        self.assertTrue(af.action_topic_missing(
            "server_status", "and now?", ["tell me a joke", "", None]))

    def test_other_actions_and_no_words_never_flag(self):
        for name in ("pause_music", "smart_home_control", "system_pulse",
                     "", None):
            with self.subTest(name=name):
                self.assertEqual(af.action_topic_missing(name, _LIVE), "")
        self.assertEqual(af.action_topic_missing("server_status", ""), "")
        self.assertEqual(af.action_topic_missing("server_status", None), "")

    def test_never_raises(self):
        class Bad:
            def __str__(self):
                raise RuntimeError("boom")
        self.assertEqual(af.action_topic_missing("server_status", Bad()), "")


if __name__ == "__main__":
    unittest.main()
