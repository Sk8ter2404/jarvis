"""Tests for core/prompt_router — dynamic local-prompt slimming (2026-07-15).

The local brain's context is capped at 12-16k tokens but the full system prompt
is ~30k, so it was TRUNCATED. The router keeps the core + only the sections a
turn needs, so the relevant instructions fit uncut. These pin: correct parsing,
relevant-section selection, the always-present core, the drop INDEX, big size
reduction, and never-raises.
"""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import prompts, prompt_router as pr   # noqa: E402

FULL = prompts.PC_CONTROL_PROMPT


class SplitTests(unittest.TestCase):
    def test_parses_core_and_sections(self):
        core, sections = pr.split_pc_control(FULL)
        self.assertTrue(len(core) > 500, "core preamble must be substantial")
        self.assertGreaterEqual(len(sections), 8,
                                "PC_CONTROL should split into many named sections")
        names = [h for h, _ in sections]
        self.assertIn("MUSIC CONTROLS", names)
        self.assertTrue(any("BAMBU" in n for n in names))

    def test_no_headers_returns_whole_as_core(self):
        core, sections = pr.split_pc_control("just some text, no headers here")
        self.assertEqual(sections, [])
        self.assertIn("just some text", core)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        _core, self.sections = pr.split_pc_control(FULL)

    def test_music_query_includes_music_excludes_printer(self):
        inc, drop = pr.select_sections("play some relaxing jazz", self.sections)
        self.assertIn("MUSIC CONTROLS", inc)
        self.assertTrue(any("BAMBU" in d for d in drop),
                        "a music query must NOT load the huge 3D-printer section")

    def test_printer_query_includes_printer(self):
        inc, _drop = pr.select_sections("is my 3d print finished", self.sections)
        self.assertTrue(any("BAMBU" in i for i in inc))

    def test_app_launching_is_always_included(self):
        # even an unrelated query keeps the fundamental app-launch grammar
        inc, _drop = pr.select_sections("tell me a joke", self.sections)
        self.assertIn("MULTI-MONITOR APP LAUNCHING", inc)

    def test_health_query_includes_health(self):
        inc, _drop = pr.select_sections("what's my cpu temperature", self.sections)
        self.assertIn("SYSTEM HEALTH", inc)


class SlimTests(unittest.TestCase):
    def test_slim_is_much_smaller_for_common_turn(self):
        slim = pr.slim_pc_control("what time is it", FULL)
        self.assertLess(len(slim), len(FULL) * 0.55,
                        "a common turn should drop well over 40% of the prompt")

    def test_slim_keeps_core_and_names_dropped_sections(self):
        slim = pr.slim_pc_control("play music", FULL)
        # the drop INDEX advertises what was left out so the model still knows
        self.assertIn("ADDITIONAL CAPABILITIES", slim)
        # the huge printer section is dropped but named in the index
        self.assertNotIn("BAMBU 3D PRINTER (H2D):\n", slim.replace(
            "ADDITIONAL CAPABILITIES", ""))  # body not present
        self.assertIn("BAMBU", slim)  # but named in the index

    def test_printer_turn_actually_loads_printer_body(self):
        slim = pr.slim_pc_control("start the 3d printer", FULL)
        # the section body (not just the index name) must be present
        self.assertIn("BAMBU 3D PRINTER", slim)
        self.assertGreater(len(slim), 20000, "printer body is large and included")

    def test_never_raises_returns_full_on_bad_input(self):
        # a prompt with no sections just comes back whole
        self.assertEqual(pr.slim_pc_control("x", "no sections in here"),
                         "no sections in here")

    def test_slim_fits_local_window_for_common_turns(self):
        # BASE identity (~4.6k tok) + slim PC + ~800 tok rules/phrasebook must
        # clear the 16k local window for a representative spread of turns.
        base = len(prompts.BASE_SYSTEM_PROMPT) // 4
        for q in ("who are you", "open chrome", "set a timer for 10 minutes",
                  "what's my gpu temp", "remind me to call mom"):
            total = base + len(pr.slim_pc_control(q, FULL)) // 4 + 800
            self.assertLess(total, 16000,
                            f"{q!r} slim prompt must fit the local window: {total}")


if __name__ == "__main__":
    unittest.main()
