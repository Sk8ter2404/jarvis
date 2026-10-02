"""'are you ok' / 'run a system check' run the REAL self-check (2026-10-01).

The 09-05 live diagnostic: both answered with a pass or something off-topic.
"run a system check" loaded only SYSTEM HEALTH, whose check_system claimed
'system check' as its own trigger (a CPU / RAM / disk readout), while
SELF DIAGNOSTIC — the home of system_check, the full sweep — had no "system
check" keyword at all. "are you ok" did load SELF DIAGNOSTIC, but nothing
told the model to RUN it, so the local brain answered from its head ("Quite
right, sir. Always operational."). These pin the routing, the prompt text,
and the deterministic classifier the monolith's shortcut uses
(tests/monolith/test_monolith_diag_fixes.py pins the shortcut itself).

Stdlib unittest, CI-safe.

    python -m unittest tests.test_self_check_routing
"""
from __future__ import annotations

import unittest

from core import fast_paths as fp
from core import prompt_router as pr
from core import prompts

FULL = prompts.PC_CONTROL_PROMPT


class SelfCheckRoutingTests(unittest.TestCase):
    def setUp(self):
        _core, self.sections = pr.split_pc_control(FULL)
        self.bodies = dict(self.sections)

    def test_system_check_loads_the_self_diagnostic_section(self):
        for q in ("run a system check", "do a systems check",
                  "run a self check", "are you ok", "are you okay",
                  "are you alright"):
            with self.subTest(q=q):
                inc, _ = pr.select_sections(q, self.sections)
                self.assertIn("SELF DIAGNOSTIC", inc)
                slim = pr.slim_pc_control(q, FULL)
                self.assertIn("system_check", slim)
                self.assertIn("are_you_ok", slim)

    def test_system_check_is_not_claimed_by_the_quick_readout(self):
        body = " ".join(self.bodies["SYSTEM HEALTH"].split())
        at = body.index("check_system")
        triggers = body[at:body.index("Example: [ACTION: check_system]", at)]
        self.assertNotIn("'system check'", triggers)
        self.assertNotIn("'systems check'", triggers)

    def test_the_section_says_run_it_with_arrow_examples(self):
        body = " ".join(self.bodies["SELF DIAGNOSTIC"].split())
        self.assertRegex(body, r"'are you ok' (?:→|->) \[ACTION: are_you_ok\]")
        self.assertRegex(body,
                         r"'run a system check' (?:→|->) \[ACTION: system_check\]")
        self.assertRegex(body.lower(), r"never answer .{0,80}from (?:your|memory)")


class SelfCheckClassifierTests(unittest.TestCase):
    def test_self_check_questions(self):
        for q in ("are you ok", "Are you okay?", "Jarvis, are you alright?",
                  "are you all right jarvis", "you ok?", "are you doing okay",
                  "run a system check", "Run a systems check, please.",
                  "do a full system check", "run a self-check",
                  "run a diagnostic", "run diagnostics", "system check",
                  "perform a self diagnostic", "check yourself",
                  "is everything ok with you"):
            with self.subTest(q=q):
                self.assertTrue(fp.is_self_check_request(q))

    def test_not_self_check_questions(self):
        for q in ("are you ok with that", "are you there", "can you hear me",
                  "check the system", "how's the system", "system status",
                  "what's broken", "are you okay to drive me", "is the mic ok",
                  "run a check on the printer", "diagnostic history",
                  "", None, 3):
            with self.subTest(q=q):
                self.assertFalse(fp.is_self_check_request(q))

    def test_match_never_answers_it_itself(self):
        # The answer needs the live diagnostic, so core.fast_paths.match (pure,
        # no I/O) must not produce one; the monolith runs the action.
        self.assertIsNone(fp.match("are you ok"))


if __name__ == "__main__":
    unittest.main()
