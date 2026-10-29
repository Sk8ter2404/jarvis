"""core/loop_guard.py (v3 W21): ping-pong, repeat and screen-look budget.

Pure module: no monolith, no network, no clock.
"""
from __future__ import annotations

import unittest

from core import loop_guard
from core.loop_guard import LoopGuard


def _run(calls):
    """Feed ``calls`` in order; the (1-based index, reason) of the first
    verdict, or None."""
    g = LoopGuard()
    for i, (name, arg) in enumerate(calls, 1):
        reason = g.check(name, arg)
        if reason:
            return i, reason
    return None


class PingPongTests(unittest.TestCase):
    def test_the_live_1019_sequence_stops_at_the_second_cycle(self):
        # web_search, see_screen, web_search, see_screen ... (6 calls in 35 s)
        hit = _run([("web_search", "weather in Ann Arbor"),
                    ("see_screen", ""),
                    ("web_search", "weather Ann Arbor today"),
                    ("see_screen", "the browser"),
                    ("web_search", "Ann Arbor weather"),
                    ("see_screen", "")])
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], 4)
        self.assertEqual(hit[1], "ping-pong web_search/see_screen")

    def test_one_cycle_is_not_a_loop(self):
        self.assertIsNone(_run([("web_search", "weather"), ("see_screen", ""),
                                ("web_search", "weather today")]))

    def test_the_liked_songs_retry_chain_is_not_stopped(self):
        # 2026-10-02 review case: click fails, retry works, click Play fails.
        self.assertIsNone(_run([("click", "Liked Songs"),
                                ("see_screen", "music app"),
                                ("click", "Liked Songs tab"),
                                ("click", "Play"),
                                ("see_screen", "music app"),
                                ("click", "the play button")]))

    def test_a_three_step_real_chain_is_not_stopped(self):
        self.assertIsNone(_run([("web_search", "best pizza near me"),
                                ("open_url", "https://example.com/pizza"),
                                ("see_screen", "the page")]))

    def test_alternation_of_unrelated_arguments_is_not_a_loop(self):
        # Two different jobs done twice each, not the same job going round.
        self.assertIsNone(_run([("play_music", "jazz"),
                                ("set_volume", "40"),
                                ("play_music", "classical piano"),
                                ("set_volume", "75")]))

    def test_form_filling_and_page_reading_are_exempt(self):
        self.assertIsNone(_run([("click", "Name"), ("type", "Sam"),
                                ("click", "Email"), ("type", "a@b.c"),
                                ("click", "Name"), ("type", "Sam")]))
        self.assertIsNone(_run([("scroll", "down"), ("see_screen", ""),
                                ("scroll", "down"), ("see_screen", "")]))

    def test_a_step_between_the_lookups_breaks_the_run(self):
        self.assertIsNone(_run([("web_search", "x y"), ("see_screen", ""),
                                ("click", "Next"),
                                ("web_search", "x y z"), ("see_screen", "")]))


class RepeatAndBudgetTests(unittest.TestCase):
    def test_the_same_call_a_third_time_is_a_repeat(self):
        hit = _run([("list_windows", ""), ("get_time", ""),
                    ("list_windows", "  "), ("now_playing", ""),
                    ("list_windows", "")])
        self.assertEqual(hit[0], 5)
        self.assertEqual(hit[1], "repeat list_windows x3")

    def test_the_repeat_ignores_case_and_punctuation(self):
        hit = _run([("web_search", "Pizza, near me!"),
                    ("get_time", ""),
                    ("web_search", "pizza near me"),
                    ("now_playing", ""),
                    ("Web_Search", "PIZZA near me.")])
        self.assertEqual(hit[0], 5)

    def test_the_fourth_screen_look_is_over_budget(self):
        names = ("see_screen", "read_screen", "local_describe_screen",
                 "see_screen")
        hit = _run([(n, f"look {i}") for i, n in enumerate(names)])
        self.assertEqual(hit[0], 4)
        self.assertTrue(hit[1].startswith("poll budget"))

    def test_looks_between_screen_changing_steps_are_not_a_poll_stream(self):
        # Investigation 2026-10-28 (W21 latent bug): look, click, look,
        # click, look, click, look was cut at the fourth look in 'on'.
        chain = [("see_screen", ""), ("click", "Next"), ("see_screen", ""),
                 ("click", "Next"), ("see_screen", ""), ("scroll", "down"),
                 ("see_screen", ""), ("type_text", "abc"), ("see_screen", "")]
        self.assertIsNone(_run(chain))
        # The same as screen_read_refusal says for the same calls.
        for i in range(1, len(chain) + 1):
            self.assertEqual(loop_guard.screen_read_refusal(chain[i - 1][0],
                                                            chain[:i - 1]), "")

    def test_listing_windows_does_not_reset_the_look_count(self):
        hit = _run([("see_screen", "a"), ("list_windows", ""),
                    ("read_screen", "b"), ("list_windows", "x"),
                    ("local_describe_screen", "c"), ("see_screen", "d")])
        self.assertEqual(hit[0], 6)
        self.assertEqual(hit[1], "poll budget see_screen x4")

    def test_three_screen_looks_are_fine(self):
        self.assertIsNone(_run([("see_screen", "a"), ("get_time", ""),
                                ("see_screen", "b"), ("now_playing", ""),
                                ("see_screen", "c")]))

    def test_waiting_or_locating_between_looks_is_still_a_poll_stream(self):
        # Review 2026-10-28: wait / find_on_screen / move_mouse change
        # nothing on the screen, so they must not start a fresh look count
        # (the reset fired on ANY non-look call and hid this loop).
        for neutral in ("wait", "find_on_screen", "move_mouse"):
            with self.subTest(neutral=neutral):
                hit = _run([("see_screen", ""), (neutral, "2")] * 6)
                self.assertEqual(hit, (5, "repeat see_screen x3"))
        hit = _run([("see_screen", "a"), ("wait", "2"), ("see_screen", "b"),
                    ("wait", "2"), ("see_screen", "c"), ("wait", "2"),
                    ("see_screen", "d")])
        self.assertEqual(hit, (7, "poll budget see_screen x4"))

    def test_the_1019_search_and_look_alternation_is_still_caught(self):
        # An unrelated action (web_search with a new query each time) is
        # not a change to the screen being looked at. Live 2026-10-19: the
        # alternation went on for six rounds.
        chain = []
        for q in ("vision quest trailer", "best pizza near me",
                  "weather tomorrow chicago", "news today", "movie times",
                  "lunch ideas"):
            chain += [("web_search", q), ("see_screen", "")]
        hit = _run(chain)
        self.assertEqual(hit, (6, "repeat see_screen x3"))

    def test_only_screen_driving_steps_reset_the_look_count(self):
        for step in ("click", "double_click", "right_click", "type",
                     "press_key", "hotkey", "scroll_down", "drag",
                     "open_url", "open_app", "launch_app",
                     "local_click_target_by_description"):
            with self.subTest(step=step):
                self.assertTrue(loop_guard._drives_screen(step))
        for step in ("wait", "sleep", "find_on_screen", "move_mouse",
                     "mouse_move", "list_windows", "web_search", "get_time",
                     "pause_for"):
            with self.subTest(step=step):
                self.assertFalse(loop_guard._drives_screen(step))

    def test_a_fresh_guard_starts_clean(self):
        g = LoopGuard()
        for _ in range(2):
            g.check("list_windows", "")
        self.assertEqual(len(g), 2)
        self.assertEqual(len(LoopGuard()), 0)


class RobustnessTests(unittest.TestCase):
    def test_odd_input_never_raises_and_judges_nothing(self):
        g = LoopGuard()
        for name, arg in ((None, None), ("", ""), (42, object()),
                          ("_unverified_claim", "x"), (b"see_screen", 3)):
            self.assertIsNone(g.check(name, arg))

    def test_mode_names(self):
        n = loop_guard.normalise_mode
        self.assertEqual([n(v) for v in ("off", "shadow", "on", " ON ")],
                         ["off", "shadow", "on", "on"])
        self.assertEqual([n(v) for v in (None, "", "yes", True, 1)],
                         ["off"] * 5)

    def test_the_module_imports_only_the_stdlib(self):
        import ast
        tree = ast.parse(open(loop_guard.__file__, encoding="utf-8").read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        self.assertLessEqual(mods, {"__future__", "re"})


if __name__ == "__main__":
    unittest.main()
