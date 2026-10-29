"""The "I've noted that for your presentation" false claim (v3.0.7,
"noted-claim").

Live 2026-10-28 19:12 on 3.0.5: a remark not meant for JARVIS, heard inside
the follow-up window, got "Certainly, sir. I've noted that for your
presentation." and no note, memory or list action ran. The reactive claim
check passed it (no note family), so it was spoken.

Through the REAL parse_and_run_actions in an owner turn (the turn ledger
open, as _run_llm_dispatch opens it):
  * with nothing run, the claim is not spoken and the honest line is - no
    synthetic result, so no extra LLM round;
  * other content in the reply is kept;
  * a reply whose own token ran an unrelated action loses the claim and says
    nothing more (the action speaks for itself);
  * a note_for_claude that really ran this turn keeps the claim untouched;
  * NOTE_CLAIM_CHECK False is the old path exactly.
And the REAL _SentenceFlushBuffer holds such a sentence instead of voicing
it early. The rules are unit-tested in tests/test_note_claims.py. Neutral
fixtures only.

    python -m unittest tests.monolith.test_monolith_note_claims
"""
from __future__ import annotations

import contextlib
import io
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_answer_first import _Base
from tests.monolith.test_monolith_claim_validation import _InlineThread

_ROOM_LINE = "I'll get to the next slide after lunch."
_LIVE_REPLY = ("[intent:confirmation] Certainly, sir. I've noted that for "
               "your presentation.")
_HONEST = "I haven't made a note of that, sir."


@requires_monolith
class NoteClaimTests(_Base):
    def _parse_in_turn(self, reply, ran=(), user=_ROOM_LINE):
        bc = self.bc
        prev = bc._begin_turn_grounding(user)
        self.addCleanup(bc._end_turn_grounding, prev)
        for name, result in ran:
            bc._note_turn_action_ran(name, result)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cleaned, results = bc.parse_and_run_actions(reply)
        return cleaned, results, buf.getvalue()

    def test_the_live_reply_speaks_the_honest_line(self):
        cleaned, results, printed = self._parse_in_turn(_LIVE_REPLY)
        self.assertEqual(cleaned, _HONEST)
        self.assertNotIn("noted that", cleaned)
        self.assertEqual(results, [])          # no follow-up round
        self.assertIn("[validation] dropped the claim", printed)

    def test_other_content_is_kept(self):
        reply = ("[intent:confirmation] Certainly, sir. I've noted that for "
                 "your presentation. The next slide covers the budget.")
        cleaned, results, _printed = self._parse_in_turn(reply)
        self.assertNotIn("noted", cleaned)
        self.assertIn("The next slide covers the budget.", cleaned)
        self.assertEqual(results, [])

    def test_an_unrelated_action_does_not_ground_it(self):
        reply = "I've noted that for your presentation, sir. [ACTION: volume_up]"
        cleaned, results, _printed = self._parse_in_turn(reply)
        self.assertEqual([n for n, _r, _i in results], ["volume_up"])
        self.assertEqual(cleaned.strip(), "")

    def test_a_note_that_really_ran_keeps_the_claim(self):
        cleaned, results, printed = self._parse_in_turn(
            _LIVE_REPLY, ran=[("note_for_claude", "Noted for Claude.")])
        self.assertIn("I've noted that for your presentation.", cleaned)
        self.assertEqual(results, [])
        self.assertNotIn("dropped the claim", printed)

    def test_a_bare_noted_ack_is_untouched(self):
        cleaned, _results, printed = self._parse_in_turn(
            "[intent:confirmation] Noted, sir.")
        self.assertIn("Noted, sir.", cleaned)
        self.assertNotIn("dropped the claim", printed)

    def test_the_setting_off_is_the_old_path(self):
        self._p(self.bc, "NOTE_CLAIM_CHECK", False, create=True)
        cleaned, results, printed = self._parse_in_turn(_LIVE_REPLY)
        self.assertIn("I've noted that for your presentation.", cleaned)
        self.assertEqual(results, [])
        self.assertNotIn("dropped the claim", printed)

    def test_a_check_fault_changes_nothing(self):
        self._p(self.bc._claim_validator, "drop_note_claims",
                side_effect=RuntimeError("boom"))
        cleaned, _results, printed = self._parse_in_turn(_LIVE_REPLY)
        self.assertIn("I've noted that for your presentation.", cleaned)
        self.assertIn("note-claim check failed", printed)

    def test_outside_an_owner_turn_nothing_changes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cleaned, _results = self.bc.parse_and_run_actions(_LIVE_REPLY)
        self.assertIn("I've noted that for your presentation.", cleaned)

    def test_a_read_only_action_with_a_shared_word_grounds_nothing(self):
        # The live path: the turn ledger holds show_shopping_list, whose name
        # shares "shopping" and "list" with the claim; nothing was added.
        cleaned, results, printed = self._parse_in_turn(
            "I've added eggs to your shopping list, sir.",
            ran=[("show_shopping_list", "Your list has milk.")],
            user="add eggs to my shopping list")
        self.assertEqual(cleaned, "I haven't added that to any list, sir.")
        self.assertEqual(results, [])
        self.assertIn("dropped the claim", printed)

    def test_the_real_writer_keeps_the_list_claim(self):
        cleaned, _results, printed = self._parse_in_turn(
            "I've added eggs to your shopping list, sir.",
            ran=[("add_shopping_item", "Added eggs.")],
            user="add eggs to my shopping list")
        self.assertIn("I've added eggs to your shopping list", cleaned)
        self.assertNotIn("dropped the claim", printed)

    def test_a_remember_turn_keeps_a_memory_claim(self):
        for reply in ("Saved to memory, sir.",
                      "I've added that to my memory, sir."):
            with self.subTest(reply=reply):
                cleaned, _results, printed = self._parse_in_turn(
                    reply, user="remember that my car is in bay four")
                self.assertIn(reply.rstrip("."), cleaned)
                self.assertNotIn("dropped the claim", printed)

    def test_a_diagnosis_is_not_replaced_by_the_honest_line(self):
        cleaned, _results, printed = self._parse_in_turn(
            "I'm putting that down to a driver glitch, sir.",
            user="why did my screen flicker just now")
        self.assertIn("driver glitch", cleaned)
        self.assertNotIn("dropped the claim", printed)

    def test_a_report_after_noted_the_following_is_kept(self):
        reply = ("I've noted the following, sir: memory is at 82 percent "
                 "and the GPU is busy.")
        cleaned, _results, printed = self._parse_in_turn(
            reply, ran=[("system_pulse", "Memory 82 percent.")],
            user="why is the ram so high")
        self.assertIn("memory is at 82 percent", cleaned)
        self.assertNotIn("dropped the claim", printed)

    def test_only_the_claim_clause_is_dropped_from_a_report(self):
        cleaned, _results, _printed = self._parse_in_turn(
            "I've noted that: memory is at 82 percent and the GPU is busy.",
            ran=[("system_pulse", "Memory 82 percent.")],
            user="why is the ram so high")
        self.assertEqual(cleaned,
                         "Memory is at 82 percent and the GPU is busy.")

    def _flush(self, chunks):
        bc = self.bc
        prev = bc._begin_turn_grounding(_ROOM_LINE)
        self.addCleanup(bc._end_turn_grounding, prev)
        spoken: list[str] = []
        with mock.patch.object(bc, "threading", mock.Mock(
                Thread=_InlineThread)):
            buf = bc._SentenceFlushBuffer(speak_fn=spoken.append)
            for c in chunks:
                buf.feed(c)
        return spoken

    def test_the_stream_flush_holds_the_claim(self):
        chunks = ("I've noted that for your presentation, sir. ",
                  "The next slide covers the budget, as planned. ", "tail")
        self.assertEqual(self._flush(chunks), [])
        # The hold is this check's: with it off the sentence flushes early.
        self._p(self.bc, "NOTE_CLAIM_CHECK", False, create=True)
        self.assertNotEqual(self._flush(chunks), [])
