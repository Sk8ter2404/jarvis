"""F8 (v3.0.6): "how's the master sheet looking?" is answered honestly.

THE LIVE TURN (2026-10-28 ~18:00, human-style test): the owner asked about
the master sheet (the server sheet he keeps outside JARVIS). No action
reads it, so the router loaded no section, and the model reached for
the nearest name in the always-on never-guess guard - project_status - which
said there was no project list. The guard now says plainly that no action
reads the master sheet and gives the honest line to speak.

Read through the REAL stable prefix the local turn sends
(_slim_stable_pc: the guard + the slim stable PC block, both
PROMPT_DIRECTIVE_V3 states). Fails on v3.0.5 (62d520b): the guard had no
master-sheet line.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_master_sheet
"""
from __future__ import annotations

from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_HONEST = "say \"I can't read the master\nsheet yet, sir.\""


@requires_monolith
class MasterSheetGuardTests(MonolithGlobalsTestCase):
    def test_the_stable_prefix_says_no_action_reads_the_master_sheet(self):
        bc = self.bc
        for flag in (False, True):
            with self.subTest(PROMPT_DIRECTIVE_V3=flag), \
                    mock.patch.object(bc, "PROMPT_DIRECTIVE_V3", flag,
                                      create=True):
                bc._SLIM_PC_CACHE.clear()
                try:
                    out = bc._slim_stable_pc("", None)
                finally:
                    bc._SLIM_PC_CACHE.clear()
                self.assertIn("The master sheet: no action reads it", out)
                self.assertIn(_HONEST, out)
                self.assertIn("Never project_status for it.", out)

    def test_the_line_sits_outside_the_only_the_token_list(self):
        # "If sir asks any of the above, your reply must contain ONLY the
        # action token" must not cover the master sheet: it has no token.
        guard = self.bc._LOCAL_NEVER_GUESS_GUARD
        self.assertLess(guard.index("If sir asks any of the above"),
                        guard.index("The master sheet: no action reads it"))
