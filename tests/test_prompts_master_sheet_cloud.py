"""v3.0.8: the CLOUD prompt's PROJECT STATUS section carries the master-sheet
rule the local brain's never-guess guard got in v3.0.6.

Live 2026-10-28 ~18:00 "how's the master sheet looking?" ran project_status
("no project list"); no action reads that sheet. v3.0.6 taught the local
brain the honest line; the cloud prompt's PROJECT STATUS section - which is
exactly where the model reads about project_status - still did not say it.

Read through the REAL per-turn block the cloud route ships
(core.prompt_router.turn_pc_block over core.prompts.PC_CONTROL_PROMPT).
Fails on v3.0.7 (1561e62): the section had no master-sheet line.

    python -m unittest tests.test_prompts_master_sheet_cloud
"""
from __future__ import annotations

import unittest

from core import prompt_router as PR
from core import prompts as P

_RULE = "The master sheet: no action reads it"
_HONEST = "say \"I can't read the master\n  sheet yet, sir.\""
_NEVER = "Never project_status for it."


class CloudProjectStatusMasterSheetTests(unittest.TestCase):
    def _section(self):
        _core, sections = PR.split_pc_control(P.PC_CONTROL_PROMPT)
        bodies = [b for h, b in sections
                  if h.strip().upper().startswith("PROJECT STATUS")]
        self.assertEqual(len(bodies), 1)
        return bodies[0]

    def test_the_section_says_no_action_reads_the_master_sheet(self):
        body = self._section()
        self.assertIn(_RULE, body)
        self.assertIn(_HONEST, body)
        self.assertIn(_NEVER, body)

    def test_a_project_turn_ships_the_rule_to_the_cloud(self):
        for said in ("what am I working on",
                     "what's the status of my master sheet?",
                     "what are my projects"):
            with self.subTest(said=said):
                block = PR.turn_pc_block(said, P.PC_CONTROL_PROMPT)
                self.assertIn(_RULE, block)
                self.assertIn(_NEVER, block)

    def test_the_rule_is_tiny_and_said_once(self):
        body = self._section()
        self.assertEqual(body.count(_RULE), 1)
        start = body.index(_RULE)
        self.assertLess(len(body[start:].split(_NEVER)[0]), 120)

    def test_project_status_triggers_are_unchanged(self):
        body = self._section()
        for ex in ("'JARVIS, what am I working on?' → [ACTION: project_status]",
                   "'what are my projects' → [ACTION: project_status]"):
            self.assertIn(ex, body)


if __name__ == "__main__":
    unittest.main()
