"""move_file through JARVIS's REAL confirmation path (v3 W15, 2026-10-27).

The brain's [ACTION: move_file, ...] goes through parse_and_run_actions ->
_jarvis_pushback -> skills/file_move.confirm_question: the owner hears the
question naming the file and both folders, the action waits on
_pending_confirmation, and only handle_confirmation_response's yes moves the
file. Every file lives in a temp-dir fake profile (tests/skills/test_file_move
.FakeProfile); LLM, speech and HUD are stubbed.

    python -m unittest tests.monolith.test_monolith_file_move
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

from skills import file_move as FM
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base
from tests.skills.test_file_move import FakeProfile

ASK = "take my resume to documents"
REPLY = "[intent:confirmation] Right away, sir. [ACTION: move_file, my resume to documents]"
QUESTION = "Move 'Sample CV Resume.pdf' from Downloads to Documents, sir?"


@requires_monolith
class FileMoveConfirmTests(_Base):
    def setUp(self):
        real_pushback = self.bc._jarvis_pushback
        real_needs = self.bc._needs_confirmation
        super().setUp()
        bc = self.bc
        self._p(bc, "_jarvis_pushback", real_pushback)
        self._p(bc, "_needs_confirmation", real_needs)
        self._p(bc, "_pending_confirmation", [])
        self._p(bc, "_pending_confirmation_at", [0.0])
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(bc, "SPEAK_RESULT_VERBATIM_ACTIONS",
                set(bc.SPEAK_RESULT_VERBATIM_ACTIONS) | set(FM.SPEAK_VERBATIM_ACTIONS))
        p = mock.patch.dict(sys.modules, {"skill_file_move": FM})
        p.start()
        self.addCleanup(p.stop)
        self.fp = FakeProfile(self)
        with contextlib.redirect_stdout(io.StringIO()):
            FM.register(self._actions)
        self.src = self.fp.dl("Sample CV Resume.pdf", data=b"r" * 321)
        self.dest = os.path.join(self.fp.known["documents"], "Sample CV Resume.pdf")

    def _yes(self, words="Jarvis, yes."):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc.handle_confirmation_response(words)

    def test_confirm_tier_membership(self):
        bc = self.bc
        for nm in ("move_file", "undo_last_move"):
            with self.subTest(nm=nm):
                self.assertIn(nm, bc._FILE_MOVE_CONFIRM_ACTIONS)
                self.assertIn(nm, bc._DESTRUCTIVE_REPLAY_ACTIONS)
                self.assertTrue(bc._autocorrect_protected(nm))
        self.assertIn("FILE_MOVE_ENABLED", bc._TURN_FLAG_KEYS)
        self.assertIs(bc.FILE_MOVE_ENABLED, True)

    def test_the_turn_asks_and_moves_nothing_until_yes(self):
        self._dispatch(ASK, REPLY)
        self.assertTrue(os.path.isfile(self.src), "moved before the yes")
        self.assertFalse(os.path.exists(self.dest))
        said = " ".join(self.spoken)
        self.assertIn(QUESTION, said)
        self.assertNotIn("Right away", said)
        self.assertEqual(list(self.bc._pending_confirmation),
                         [("move_file", "my resume to documents")])
        self.spoken.clear()
        self.assertTrue(self._yes())
        self.assertFalse(os.path.exists(self.src))
        self.assertEqual(os.path.getsize(self.dest), 321)
        self.assertEqual(self.spoken,
                         ["Done, sir - 'Sample CV Resume.pdf' is now in Documents."])

    def test_a_no_moves_nothing(self):
        self._dispatch(ASK, REPLY)
        self.assertTrue(self._yes("No."))
        self.assertTrue(os.path.isfile(self.src))
        self.assertFalse(os.path.exists(self.dest))
        self.assertEqual(list(self.bc._pending_confirmation), [])

    def test_something_else_cancels_and_moves_nothing(self):
        self._dispatch(ASK, REPLY)
        self.assertFalse(self._yes("what time is it"))
        self.assertTrue(os.path.isfile(self.src))

    def test_asks_with_pushback_off_too(self):
        self._p(self.bc, "PUSHBACK_ENABLED", False)
        self._dispatch(ASK, REPLY)
        self.assertTrue(os.path.isfile(self.src))
        self.assertIn(QUESTION, " ".join(self.spoken))

    def test_keyword_names_still_get_the_file_question(self):
        # 'pay' / 'transfer' are CONFIRM_KEYWORDS; their generic prompt's yes
        # would have no plan behind it and move nothing.
        self.fp.dl("payslip transfer form.pdf")
        self._dispatch("move my payslip to the desktop",
                       "[ACTION: move_file, payslip transfer form to the desktop]")
        self.assertIn("Move 'payslip transfer form.pdf' from Downloads to Desktop, sir?",
                      " ".join(self.spoken))
        self.assertTrue(self._yes())
        self.assertTrue(os.path.isfile(
            os.path.join(self.fp.known["desktop"], "payslip transfer form.pdf")))

    def test_unresolved_request_says_why_and_moves_nothing(self):
        self._dispatch("move my taxes to documents",
                       "[ACTION: move_file, taxes to documents]")
        self.assertEqual(list(self.bc._pending_confirmation), [])
        self.assertIn("I couldn't find a file called 'taxes' in your folders, sir.",
                      " ".join(self.spoken))
        self.assertTrue(os.path.isfile(self.src))

    def test_undo_goes_through_the_same_yes(self):
        self._dispatch(ASK, REPLY)
        self._yes()
        self.spoken.clear()
        self._dispatch("undo that move", "[ACTION: undo_last_move]")
        self.assertIn("Move 'Sample CV Resume.pdf' back from Documents to Downloads, sir?",
                      " ".join(self.spoken))
        self.assertTrue(os.path.isfile(self.dest), "moved back before the yes")
        self.spoken.clear()
        self.assertTrue(self._yes())
        self.assertTrue(os.path.isfile(self.src))
        self.assertEqual(self.spoken,
                         ["Done, sir - 'Sample CV Resume.pdf' is back in Downloads."])


if __name__ == "__main__":
    unittest.main()
