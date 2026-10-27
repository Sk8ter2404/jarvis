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

    def test_three_screen_looks_are_fine(self):
        self.assertIsNone(_run([("see_screen", "a"), ("get_time", ""),
                                ("see_screen", "b"), ("now_playing", ""),
                                ("see_screen", "c")]))

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
