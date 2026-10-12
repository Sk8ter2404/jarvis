"""PROMPT_SLIM_MODE's pure halves in core/prompt_router.py (2026-10-12).

The 2026-10-12 census found the cache-stable local layout's CAPABILITY INDEX
named sections only, so a router miss left the model with no action name at
all, while the cached head carried every skill's routing examples and the
learned tail every auto-learned topic. Slim mode indexes every documented
action BY NAME in the static head and moves skill examples / topics to the
turns that mention them. These tests pin the router side; the monolith side
(prefix stability, budget, wiring) is tests/monolith/test_monolith_prompt_slim.py.

Light deps only: runs on the CI runner.
"""
from __future__ import annotations

import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import prompt_router as pr  # noqa: E402
from core.action_risk import SELF_TERMINATING_ACTIONS  # noqa: E402
from core.prompts import PC_CONTROL_PROMPT  # noqa: E402


_SKILLS = (
    "\n\nADDITIONAL SKILL ACTIONS (locally-installed skills - same "
    "[ACTION: ...] contract as above):\n"
    "WIDGET POLISHER (polishes the widgets):\n"
    "  polish_widget - fire on 'polish the widget', 'shine my widgets'.\n"
    "    Example: 'polish the widget' -> [ACTION: polish_widget]\n"
    "  widget_status / widget_report - report the widgets.\n"
    "GADGET LAUNCHER:\n"
    "  launch_gadget - fire on 'launch the gadget'.\n"
    "    Example: 'launch the gadget' -> [ACTION: launch_gadget]")


def _index_part(block: str) -> str:
    i = block.find("ACTION INDEX")
    return block[i:] if i >= 0 else ""


class DocumentedActionsTests(unittest.TestCase):
    def test_bracketed_and_known_names_in_order_without_duplicates(self):
        text = ("say alpha_one first, then [ACTION: beta_two, x] and "
                "[ACTION: alpha_one] and not_an_action here")
        self.assertEqual(pr.documented_actions(text, {"alpha_one"}),
                         ["alpha_one", "beta_two"])
        self.assertEqual(pr.documented_actions(text), ["beta_two",
                                                       "alpha_one"])

    def test_never_raises(self):
        self.assertEqual(pr.documented_actions(None), [])


class SlimStableBlockTests(unittest.TestCase):
    def setUp(self):
        self.block = pr.slim_stable_pc_block(PC_CONTROL_PROMPT, _SKILLS)
        self.core, self.sections = pr.split_pc_control(PC_CONTROL_PROMPT)

    def test_byte_identical_on_every_call(self):
        self.assertEqual(self.block,
                         pr.slim_stable_pc_block(PC_CONTROL_PROMPT, _SKILLS))

    def test_core_preamble_and_safety_sections_are_in_full(self):
        self.assertTrue(self.block.startswith(self.core))
        bodies = dict((h.strip(), b) for h, b in self.sections)
        for name in pr._SLIM_HEAD_SECTIONS | pr._ALWAYS:
            self.assertIn(bodies[name], self.block, name)
        self.assertIn("SAFETY:", self.block)

    def test_every_capability_is_still_named(self):
        idx = _index_part(self.block)
        for h, _b in self.sections:
            if h.strip().upper() in pr._ALWAYS | pr._SLIM_HEAD_SECTIONS:
                continue
            self.assertIn("\n" + h.strip(), idx, h)
        self.assertIn("\nWIDGET POLISHER: polish_widget", idx)
        self.assertIn("\nGADGET LAUNCHER: launch_gadget", idx)

    def test_every_bracketed_action_is_nameable(self):
        documented = set(pr.documented_actions(PC_CONTROL_PROMPT + _SKILLS))
        missing = sorted(a for a in documented - set(SELF_TERMINATING_ACTIONS)
                         if not re.search(r"(?<![a-z0-9_])" + re.escape(a)
                                          + r"(?![a-z0-9_])", self.block))
        self.assertEqual(missing, [])

    def test_self_terminating_actions_are_never_indexed(self):
        idx = _index_part(self.block)
        for a in SELF_TERMINATING_ACTIONS:
            self.assertIsNone(
                re.search(r"(?<![a-z0-9_])" + re.escape(a) + r"(?![a-z0-9_])",
                          idx), a)

    def test_known_prose_names_are_indexed_and_aliases_collapse(self):
        known = {"polish_widget", "widget_status", "widget_report",
                 "launch_gadget"}
        block = pr.slim_stable_pc_block(PC_CONTROL_PROMPT, _SKILLS,
                                        known=known)
        self.assertIn("polish_widget, widget_status, widget_report",
                      _index_part(block))
        group = {"widget_status": "h1", "widget_report": "h1"}
        block = pr.slim_stable_pc_block(PC_CONTROL_PROMPT, _SKILLS,
                                        known=known, group=group)
        idx = _index_part(block)
        self.assertIn("widget_status", idx)
        self.assertNotIn("widget_report", idx)

    def test_no_sections_falls_back_to_the_old_block(self):
        self.assertEqual(pr.slim_stable_pc_block("just text"),
                         pr.stable_pc_block("just text"))


class SkillExampleRoutingTests(unittest.TestCase):
    def setUp(self):
        self.parts = pr.split_skill_examples(_SKILLS)

    def test_split(self):
        self.assertEqual([n for n, _t in self.parts],
                         ["WIDGET POLISHER", "GADGET LAUNCHER"])
        self.assertEqual("\n".join(t for _n, t in self.parts),
                         _SKILLS.split("as above):\n", 1)[1])
        self.assertEqual(pr.split_skill_examples(""), [])

    def test_routes_on_the_turns_words_only(self):
        got = pr.select_skill_examples("Jarvis, shine my widgets please",
                                       self.parts)
        self.assertEqual(len(got), 1)
        self.assertIn("polish_widget", got[0])
        self.assertEqual(pr.select_skill_examples("what's the weather",
                                                  self.parts), [])
        both = pr.select_skill_examples("polish the widget and launch the "
                                        "gadget", self.parts)
        self.assertEqual(len(both), 2)

    def test_a_follow_up_inherits_from_history(self):
        hist = [{"role": "user", "content": "launch the gadget"},
                {"role": "assistant", "content": "Done, sir."}]
        got = pr.select_skill_examples("do it again", self.parts,
                                       history=hist)
        self.assertEqual(len(got), 1)
        self.assertIn("launch_gadget", got[0])


    def test_split_on_the_per_skill_boundary_when_given(self):
        # A column-0 continuation line in one skill's text ("NOTE: ...") and
        # an indented first line in the next must not move the boundary: the
        # per-skill texts decide it (review finding 4, 2026-10-12).
        plex = ("PLEX CONTROL:\n  plex_play - fire on 'play the movie'.\n"
                "NOTE: plex_pause needs a title. Fire on 'pause plex'.")
        lamp = "  lamp_on - fire on 'lamp on'."
        block = ("\n\nADDITIONAL SKILL ACTIONS (same contract as above):\n"
                 + plex + "\n" + lamp)
        heur = pr.split_skill_examples(block)
        self.assertEqual([n for n, _t in heur], ["PLEX CONTROL", "NOTE"])
        parts = pr.split_skill_examples(block, [plex, lamp])
        self.assertEqual([t for _n, t in parts], [plex, lamp])
        self.assertEqual(parts[0][0], "PLEX CONTROL")
        self.assertEqual(pr.select_skill_examples("play the movie", parts),
                         [plex])
        # Texts that are not this block's fall back to the heuristic.
        self.assertEqual(pr.split_skill_examples(block, ["other"]), heur)
        self.assertIn("plex_pause",
                      pr.slim_stable_pc_block(PC_CONTROL_PROMPT, block,
                                              known={"plex_pause"},
                                              skill_texts=[plex, lamp]))
        self.assertEqual(pr.section_for_action(PC_CONTROL_PROMPT, "plex_pause",
                                               block, skill_texts=[plex, lamp]),
                         plex)


class LearnedLinesTests(unittest.TestCase):
    def test_relevant_lines(self):
        lines = ["- 2026-10-12 (desk): the rover build",
                 "- 2026-10-12 (desk): printer filament",
                 "- 2026-10-12 (desk): rovers on mars"]
        self.assertEqual(pr.relevant_lines("how's the rover going", lines),
                         [lines[0], lines[2]])
        self.assertEqual(pr.relevant_lines("how's the rover going", lines,
                                           max_n=1), [lines[2]])
        self.assertEqual(pr.relevant_lines("tell me a joke", lines), [])
        self.assertEqual(pr.relevant_lines("", lines), [])

    def test_owner_memory_request(self):
        for t in ("Jarvis, what do you know about me?",
                  "what have you learned about me so far",
                  "How well do you know me",
                  "tell me everything you know about me",
                  "what do you know of me",
                  "what have you got on me",
                  "what do you remember of me",
                  "what else have you learned about me"):
            self.assertTrue(pr.is_owner_memory_request(t), t)
        for t in ("what do you know about printers", "play music", "",
                  "what have you got on the calendar",
                  "tell me everything you know about rovers",
                  "forget everything you know about me"):
            self.assertFalse(pr.is_owner_memory_request(t), t)


class RouterMissTests(unittest.TestCase):
    def test_miss_is_an_action_neither_routed_nor_in_the_head(self):
        stable = pr.slim_stable_pc_block(PC_CONTROL_PROMPT, _SKILLS)
        routed = "\n".join(pr.select_skill_examples("polish the widget",
                                                    pr.split_skill_examples(
                                                        _SKILLS)))
        self.assertEqual(pr.router_miss(["polish_widget"], routed, stable),
                         [])
        self.assertEqual(pr.router_miss(["launch_gadget"], routed, stable),
                         ["launch_gadget"])
        self.assertEqual(pr.router_miss(["made_up_name"], "", stable),
                         ["made_up_name"])

    def test_section_for_action(self):
        body = pr.section_for_action(PC_CONTROL_PROMPT, "launch_gadget",
                                     _SKILLS)
        self.assertTrue(body.startswith("GADGET LAUNCHER"))
        self.assertEqual(pr.section_for_action(PC_CONTROL_PROMPT,
                                               "made_up_name"), "")


if __name__ == "__main__":
    unittest.main()
