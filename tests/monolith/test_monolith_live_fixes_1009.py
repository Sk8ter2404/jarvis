"""Live fixes of 2026-10-09 (logs session_2026-10-09_17-49-22 and the
2026-10-05 session): hand tracking routes to the air mouse, a singular
"close that ... window" closes one window, a misheard "you two" is YouTube
only inside an open / close / play command, a website name opens in the
browser, and the boot warm-up gets a longer budget after a cold server start.

    python -m unittest tests.monolith.test_monolith_live_fixes_1009
"""
from __future__ import annotations

import types
import unittest
from unittest import mock

from tests._monolith_harness import load_monolith, requires_monolith


class HandTrackingRouteTests(unittest.TestCase):
    def test_on_off_phrasings(self):
        from core.dispatcher import hand_tracking_route as r
        for t in ("Jarvis, turn on hand tracking.", "enable hand tracking",
                  "hand tracking on", "turn hand tracking on please",
                  "switch on the hand tracking"):
            self.assertEqual(r(t), "[ACTION: air_mouse_on]", t)
        for t in ("turn off hand tracking", "disable hand tracking",
                  "hand tracking off", "stop the hand tracking"):
            self.assertEqual(r(t), "[ACTION: air_mouse_off]", t)

    def test_questions_and_compound_stay_with_the_brain(self):
        from core.dispatcher import hand_tracking_route as r
        for t in ("is hand tracking on?", "turn on hand tracking and play jazz",
                  "how does hand tracking work"):
            self.assertIsNone(r(t), t)


class MishearingTests(unittest.TestCase):
    def test_you_two_is_youtube_in_commands_only(self):
        from core.stt_vocab import fix_command_mishearings as f
        self.assertEqual(f("Jarvis opening you two back up."),
                         "Jarvis opening YouTube back up.")
        self.assertEqual(f("close you tube"), "close YouTube")
        self.assertEqual(f("play jazz on you two"), "play jazz on YouTube")
        self.assertEqual(f("open u2"), "open YouTube")

    def test_everything_else_untouched(self):
        from core.stt_vocab import fix_command_mishearings as f
        for t in ("I'll see you two tomorrow", "you two should talk",
                  "play U2 songs", "tell you two things"):
            self.assertEqual(f(t), t)


class SiteNameTests(unittest.TestCase):
    def test_website_names_become_pages(self):
        from core.actions import _website_name_url as u
        self.assertEqual(u("youtube"), "https://www.youtube.com")
        self.assertEqual(u("HBO Max"), "https://www.max.com")
        self.assertEqual(u("the netflix website"), "https://www.netflix.com")
        self.assertEqual(u("gmail"), "https://mail.google.com")

    def test_apps_are_not_sites(self):
        from core.actions import _website_name_url as u
        for n in ("spotify", "notepad", "claude", ""):
            self.assertIsNone(u(n), n)


class SingularCloseTests(unittest.TestCase):
    def test_singular_vs_plural(self):
        from core.actions import _singular_close_said as s
        self.assertTrue(s("Jarvis, go ahead and close that chrome window."))
        self.assertTrue(s("close that Chrome window and open up a YouTube one"))
        self.assertTrue(s("close this window"))
        self.assertFalse(s("close all the chrome windows"))
        self.assertFalse(s("close those chrome windows"))
        self.assertFalse(s("close chrome"))
        self.assertFalse(s("close that chrome window and all the others"))

    def test_one_window_prefers_focus_then_frontmost(self):
        from core.actions import _one_window
        a = types.SimpleNamespace(_hWnd=1, title="a")
        b = types.SimpleNamespace(_hWnd=2, title="b")
        bc = types.SimpleNamespace(_read_focused_window=lambda: (2, "b", None))
        self.assertEqual(_one_window(bc, [a, b]), [b])
        bc = types.SimpleNamespace(_read_focused_window=lambda: (9, "x", None))
        self.assertEqual(_one_window(bc, [a, b]), [a])

    def test_close_matches_trimmed_only_for_singular(self):
        from core import actions as act
        a = types.SimpleNamespace(_hWnd=1, title="a - Google Chrome")
        b = types.SimpleNamespace(_hWnd=2, title="b - Google Chrome")
        for said, expect in (("close that chrome window", 1),
                             ("close all chrome windows", 2)):
            bc = types.SimpleNamespace(
                _find_windows_by_title=lambda q: [a, b],
                _turn_user_text=lambda said=said: said,
                _read_focused_window=lambda: (1, "a", None))
            with mock.patch.object(act, "_forgot_bulk_close", lambda: None):
                matches, _l, _b = act._close_window_matches(bc, "Chrome")
            self.assertEqual(len(matches), expect, said)


class ReviewFixTests(unittest.TestCase):
    def test_u2_band_requests_untouched(self):
        from core.stt_vocab import fix_command_mishearings as f
        for t in ("start U2 radio", "open the u2 concert playlist",
                  "open u2 on spotify", "go to u2 website"):
            self.assertEqual(f(t), t)
        self.assertEqual(f("open u2"), "open YouTube")
        self.assertEqual(f("open you two"), "open YouTube")

    def test_compound_close_not_singular(self):
        from core.actions import _singular_close_said as g
        self.assertTrue(g("close that chrome window"))
        for t in ("close that chrome window and the youtube window",
                  "close that window or this window",
                  "close that window then close all windows",
                  "close that chrome window, and then after that also close every other window"):
            self.assertFalse(g(t), t)

    def test_hand_tracking_ambiguous_words_left_to_brain(self):
        from core.dispatcher import hand_tracking_route as h
        for t in ("turn up hand tracking", "pause hand tracking",
                  "hand tracking enabled", "hand tracking disabled"):
            self.assertIsNone(h(t), t)
        self.assertEqual(h("turn on hand tracking"), "[ACTION: air_mouse_on]")
        self.assertEqual(h("hand tracking off"), "[ACTION: air_mouse_off]")

    def test_gmail_dot_com_and_twitch_app(self):
        from core.actions import _website_name_url as w
        self.assertEqual(w("gmail.com"), "https://mail.google.com")
        self.assertIsNone(w("twitch"))


@requires_monolith
class WarmupBudgetTests(unittest.TestCase):
    def test_cold_start_gets_longer_budget(self):
        bc = load_monolith()
        warm = bc._warmup_timeout(False)
        cold = bc._warmup_timeout(True)
        self.assertEqual(warm, tuple(bc._LOCAL_GENERATE_TIMEOUT))
        self.assertGreater(cold[1], warm[1])


if __name__ == "__main__":
    unittest.main()
