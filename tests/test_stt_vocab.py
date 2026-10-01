"""Tests for core/stt_vocab.py and its wiring into the open / launch actions.

THE LIVE FAILURE (2026-10-01): "open <the helpdesk app> on my left monitor" came out of
Whisper as "open a cello on my left monitor"; the turn became a YouTube search for a
cello. The owner's hotwords, his correction table and his named site shortcuts fix it.
Generic stand-ins only (example.com), never the owner's real sites.

    python -m unittest tests.test_stt_vocab
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import stt_vocab as sv


class HotwordsTests(unittest.TestCase):
    def test_unset_is_none_so_whisper_is_unchanged(self):
        for v in ("", None, "  ,  ", [], 7):
            with self.subTest(v=v):
                self.assertIsNone(sv.hotwords_arg(v))

    def test_string_and_list_are_cleaned(self):
        self.assertEqual(sv.hotwords_arg(" Zorblat ,  Flemwick,"), "Zorblat, Flemwick")
        self.assertEqual(sv.hotwords_arg(["Zorblat", " Flemwick "]), "Zorblat, Flemwick")

    def test_capped(self):
        self.assertLessEqual(len(sv.hotwords_arg(", ".join(["word"] * 500))), 400)


class WakeWordNeverAHotwordTests(unittest.TestCase):
    def test_wake_words_are_stripped(self):
        self.assertEqual(sv.hotwords_arg("JARVIS, Zorblat, hey  Jarvis, Flemwick, wake up"),
                         "Zorblat, Flemwick")
        self.assertIsNone(sv.hotwords_arg("Jarvis"))

    def test_a_name_containing_jarvis_is_kept(self):
        self.assertEqual(sv.hotwords_arg("Jarvis Rover"), "Jarvis Rover")


class HotwordEchoTests(unittest.TestCase):
    HOT = "JARVIS, Zorblat, Entry, Token, Flemwick, Quonset, 365, Brindle"

    def test_the_live_echo_is_caught(self):
        # Live 2026-10-01 17:32 shape: the list read back, trailing comma and all.
        self.assertTrue(sv.is_hotword_echo(
            "JARVIS, Zorblat, Entry, Token, Flemwick, Quonset, 365, Brindle,", self.HOT))
        self.assertTrue(sv.is_hotword_echo("Zorblat, Entry, Token.", self.HOT))

    def test_real_requests_survive(self):
        for said in ("Jarvis, open Zorblat on my left monitor",
                     "Jarvis open Zorblat and Flemwick please",
                     "what's the token count on Quonset today",
                     "Zorblat", "", "hello"):
            with self.subTest(said=said):
                self.assertFalse(sv.is_hotword_echo(said, self.HOT))

    def test_no_hotwords_means_no_echo(self):
        self.assertFalse(sv.is_hotword_echo("Zorblat, Entry, Token", ""))
        self.assertFalse(sv.is_hotword_echo("Zorblat, Entry, Token", None))

    def test_whole_words_only(self):
        self.assertFalse(sv.is_hotword_echo("tokens entryway zorblatty", self.HOT))


class ReplacementTests(unittest.TestCase):
    MAP = {"a cello": "Zorblat", "zor blat": "Zorblat", "a cello app": "the Zorblat app"}

    def test_the_live_mishearing_is_corrected(self):
        self.assertEqual(sv.apply_replacements("Jarvis, open a cello on my left monitor.", self.MAP),
                         "Jarvis, open Zorblat on my left monitor.")

    def test_whole_words_only_and_case_insensitive(self):
        self.assertEqual(sv.apply_replacements("Open A Cello", self.MAP), "Open Zorblat")
        self.assertEqual(sv.apply_replacements("play a celloist", self.MAP), "play a celloist")
        self.assertEqual(sv.apply_replacements("zor  blat time", self.MAP), "Zorblat time")

    def test_longest_phrase_wins(self):
        self.assertEqual(sv.apply_replacements("open a cello app", self.MAP), "open the Zorblat app")

    def test_empty_or_junk_mapping_is_a_no_op(self):
        for m in ({}, None, [], {"": "x"}, {"a cello": 5}):
            with self.subTest(m=m):
                self.assertEqual(sv.apply_replacements("open a cello", m), "open a cello")


class ShortcutTests(unittest.TestCase):
    S = {"zorblat": "https://zorblat.example.com/tickets", "bad": "javascript:alert(1)"}

    def test_spoken_forms_resolve(self):
        for name in ("Zorblat", "zorblat", "my Zorblat tickets", "the zorblat site",
                     "  Zorblat  "):
            with self.subTest(name=name):
                self.assertEqual(sv.site_shortcut(name, self.S),
                                 "https://zorblat.example.com/tickets")

    def test_no_match_or_unsafe_url_is_none(self):
        self.assertIsNone(sv.site_shortcut("flemwick", self.S))
        self.assertIsNone(sv.site_shortcut("bad", self.S))
        self.assertIsNone(sv.site_shortcut("zorblat", {}))
        self.assertIsNone(sv.site_shortcut(None, self.S))


class ActionWiringTests(unittest.TestCase):
    S = {"zorblat": "https://zorblat.example.com/tickets"}

    def test_launch_app_opens_a_site_shortcut(self):
        from core import actions as A
        with mock.patch("core.config.SITE_SHORTCUTS", self.S, create=True), \
                mock.patch.object(A, "_act_open_url", return_value="opened") as ou:
            out = A._act_launch_app("Zorblat")
        ou.assert_called_once_with("https://zorblat.example.com/tickets")
        self.assertEqual(out, "opened")

    def test_open_url_maps_a_bare_shortcut_name(self):
        from core import actions as A
        with mock.patch("core.config.SITE_SHORTCUTS", self.S, create=True), \
                mock.patch.object(A.webbrowser, "open") as wb, \
                mock.patch.object(A.time, "sleep"):
            A._act_open_url("zorblat")
        wb.assert_called_once_with("https://zorblat.example.com/tickets")

    def test_unset_shortcuts_change_nothing(self):
        from core import actions as A
        with mock.patch("core.config.SITE_SHORTCUTS", {}, create=True), \
                mock.patch.object(A.webbrowser, "open") as wb, \
                mock.patch.object(A.time, "sleep"):
            A._act_open_url("example.com")
        wb.assert_called_once_with("https://example.com")


if __name__ == "__main__":
    unittest.main()
