"""core/action_risk.py — the ONE risk classification of action names.

Two consumers read it: the web Actions tab (a name that runs on one click
without the LLM's judgement must ask first) and the voice dispatcher's fuzzy
action-name corrector (a GUESSED name must never land on a destructive or
self-terminating action — 2026-10-01: an invented [ACTION: shutdown] for
"Jarvis, turn it off" was corrected onto shutdown_jarvis).
"""
from __future__ import annotations

import unittest

from core import action_risk as ar


class GuessProtectedTests(unittest.TestCase):
    def test_destructive_and_self_terminating_names_are_protected(self):
        for name in ("shutdown_jarvis", "shut_down", "exit_jarvis",
                     "quit_jarvis", "power_off_jarvis", "turn_off_jarvis",
                     "restart", "reboot_pc", "upgrade",
                     "start_overnight_upgrade", "reset_memory",
                     "forget_face", "forget_voice", "forget_last_hour",
                     "clear_tasks", "delete_file", "wipe_cache",
                     "close_window", "kill_process", "run_shell",
                     "run_python", "switch_llm", "buy_item"):
            with self.subTest(name=name):
                self.assertTrue(ar.guess_protected(name), name)

    def test_everyday_names_are_not_protected(self):
        # read-outs and reversible everyday actions keep their typo fix-ups
        for name in ("screenshot", "play_music", "pause_music", "get_time",
                     "smart_home_control", "read_email", "unread_email",
                     "see_screen", "volume_up", "set_timer", ""):
            with self.subTest(name=name):
                self.assertFalse(ar.guess_protected(name), name)

    def test_sends_are_left_to_the_draft_read_back_gate(self):
        # The "sends or says" patterns are deliberately broad for the web tab
        # (*_email catches read_email), and a guessed send_* is still read
        # back before it goes out (core/draft_preview_gate) - not protected.
        self.assertEqual(ar.confirm_reasons("read_email"), (ar.SENDS,))
        self.assertFalse(ar.guess_protected("read_email"))

    def test_a_name_in_two_categories_is_protected_by_either(self):
        # archive_email: first match is SENDS (the web tab's prompt reason),
        # but it is also DELETES, which protects it.
        self.assertEqual(ar.action_confirm_reason("archive_email"), ar.SENDS)
        self.assertIn(ar.DELETES, ar.confirm_reasons("archive_email"))
        self.assertTrue(ar.guess_protected("archive_email"))

    def test_case_and_whitespace_are_ignored(self):
        self.assertTrue(ar.guess_protected("  Shutdown_JARVIS "))
        self.assertFalse(ar.guess_protected(None))


class SingleSourceTests(unittest.TestCase):
    def test_the_web_actions_tab_reads_the_same_rules(self):
        from tools import web_interface as wi
        self.assertIs(wi._ACTION_CONFIRM_RULES, ar.ACTION_CONFIRM_RULES)
        self.assertIs(wi.action_confirm_reason, ar.action_confirm_reason)

    def test_every_rule_reason_is_a_known_category(self):
        cats = {ar.STOPS_JARVIS, ar.SENDS, ar.DELETES, ar.RUNS_CODE,
                ar.DESKTOP, ar.SPENDS}
        self.assertEqual({why for _p, why in ar.ACTION_CONFIRM_RULES}, cats)
        self.assertEqual(ar.GUESS_PROTECTED_REASONS, cats - {ar.SENDS})


if __name__ == "__main__":
    unittest.main()
