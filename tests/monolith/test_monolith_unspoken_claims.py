"""An unverified execution claim is never SPOKEN (2026-10-02 live).

Two live turns, same day, after Parakeet became the primary speech-to-text
(it writes the imperative "close" as "closed"):

  * 11:57:29 - the owner asked JARVIS to close every window but one. The
    reply claimed it had "taken the liberty of closing everything else" with
    no [ACTION:] token. The hallucinated-action detector FIRED - and JARVIS
    still spoke the claim; the correction came one LLM round later.
  * 11:59:52 - "close <app>" was answered "Very good, sir. <app> has been
    closed." No action ran, the passive voice escaped the detector, and no
    correction ever followed.

Root cause of the first: parse_and_run_actions appended the synthetic
_unverified_claim result but returned the claim prose unchanged, and
_run_llm_dispatch_body speaks that prose BEFORE the follow-up loop runs. Fix:
when the detector fires the prose is withheld (cleaned = ""), so the turn
goes straight to the follow-up round, which emits the real action or admits
it can't; the synthetic result tells the model its reply was never heard.
The streaming flush (cloud route) refuses to voice a claim sentence early for
the same reason. The second is the passive rule in core/claim_validator.py
(tests/test_claim_validator_passive.py).

Made-up fixtures of the live shapes; every LLM call, action and speech output
is stubbed (no audio, no network, no real windows).

    python -m unittest tests.monolith.test_monolith_unspoken_claims
"""
from __future__ import annotations

import contextlib
import io

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

_PASSIVE = "[intent:confirmation] Very good, sir. Notepad has been closed."
_LIBERTY = ("[intent:confirmation] Very good, sir. I've taken the liberty of "
            "closing everything else so you have some room.")


@requires_monolith
class UnspokenClaimTests(_Base):
    def setUp(self):
        super().setUp()
        self._stub("close_window", "closed: Notepad")

    def _no_claim_spoken(self, *fragments):
        for s in self.spoken:
            for frag in fragments:
                self.assertNotIn(frag, s)

    def test_the_passive_claim_is_not_spoken_and_the_action_runs(self):
        self._dispatch("Jarvis closed notepad.", _PASSIVE,
                       ["[intent:confirmation] Right away, sir. "
                        "[ACTION: close_window, notepad]"])
        self._no_claim_spoken("has been closed")
        self.assertEqual(self._followup_names(), ["_unverified_claim"])
        self.assertEqual(self.calls["close_window"], ["notepad"])
        self.assertEqual(self.spoken,
                         ["[intent:confirmation] Right away, sir."])

    def test_the_liberty_claim_is_not_spoken(self):
        # A phrasing the window-keep route leaves to the model (2026-10-03:
        # "close all windows except X" itself now runs close_all_windows_
        # except without it).
        self._dispatch("Jarvis, clear away every window except the editor.",
                       _LIBERTY,
                       ["[intent:bad_news] I'm afraid I can't pick out the "
                        "windows from here, sir."])
        self._no_claim_spoken("taken the liberty", "closing everything")
        self.assertEqual(self._followup_names(), ["_unverified_claim"])
        self.assertEqual(self.spoken,
                         ["[intent:bad_news] I'm afraid I can't pick out the "
                          "windows from here, sir."])

    def test_a_claim_repeated_in_the_follow_up_ends_in_the_close_out(self):
        bc = self.bc
        self._dispatch("Jarvis closed notepad.", _PASSIVE,
                       ["[intent:confirmation] Done, sir. Notepad is shut."])
        self._no_claim_spoken("has been closed", "Done, sir")
        self.assertEqual(self.calls["close_window"], [])
        self.assertEqual(self.spoken, [bc._CLOSE_OUT_GENERIC])

    def test_parse_withholds_the_prose_and_tells_the_model(self):
        bc = self.bc
        prev = bc._begin_turn_grounding("close everything but the editor")
        self.addCleanup(bc._end_turn_grounding, prev)
        cleaned, results = self._quiet(bc.parse_and_run_actions, _LIBERTY)
        self.assertEqual(cleaned, "")
        self.assertEqual([n for n, _r, _i in results], ["_unverified_claim"])
        warn = results[0][1]
        self.assertIn("hallucinated execution", warn)
        self.assertIn("NOT spoken", warn)

    def test_a_proactive_remark_is_not_silenced(self):
        # Review 2026-10-02: outside a dispatch (the proactive remark) there
        # is no follow-up round to answer instead, so withholding would only
        # silence the remark - the prose is kept as before.
        bc = self.bc
        self.assertIsNone(getattr(bc._turn_grounding, "frame", None))
        remark = "Sir, your print job has been completed."
        cleaned, results = self._quiet(bc.parse_and_run_actions, remark)
        self.assertEqual(cleaned, remark)
        self.assertEqual([n for n, _r, _i in results], ["_unverified_claim"])
        self.assertNotIn("NOT spoken", results[0][1])

    def test_the_proactive_turn_still_speaks_its_remark(self):
        bc = self.bc
        remark = "Sir, your print job has been completed."
        self._p(bc, "generate_proactive_comment", return_value=remark)
        for name in ("pause_face_tracking", "resume_face_tracking",
                     "_thinking_loop", "_proactive_note_spoken"):
            self._p(bc, name)
        n0 = len(bc.conversation_history)
        self.addCleanup(lambda: bc.conversation_history.__delitem__(
            slice(n0, None)))
        self._quiet(bc._do_proactive_turn, {})
        self.assertEqual(self.spoken, [remark])

    def test_a_report_of_an_action_that_ran_is_still_spoken(self):
        bc = self.bc
        prev = bc._begin_turn_grounding("close notepad")
        self.addCleanup(bc._end_turn_grounding, prev)
        bc._note_turn_action_ran("close_window", "closed: Notepad")
        reply = "Notepad has been closed, sir."
        cleaned, results = self._quiet(bc.parse_and_run_actions, reply)
        self.assertEqual(results, [])
        self.assertEqual(cleaned, reply)

    def test_review_false_positives_are_still_spoken(self):
        # Review 2026-10-02: a withheld reply is silence plus a re-prompt to
        # "emit the real action", so these true replies must pass untouched:
        # a report worded in another family than the action that ran, and a
        # passive echo of the owner's thanks or news.
        bc = self.bc
        for user, ran, reply in (
                ("turn off the desk lamp", ("smart_home_control",
                                            "lamp: off"),
                 "The desk lamp has been switched off, sir."),
                ("thanks for closing that", None,
                 "Of course, sir. It has been closed."),
                ("the dentist appointment got moved", None,
                 "Indeed, sir, it has been moved to Thursday.")):
            with self.subTest(user=user):
                prev = bc._begin_turn_grounding(user)
                try:
                    if ran:
                        bc._note_turn_action_ran(*ran)
                    cleaned, results = self._quiet(bc.parse_and_run_actions,
                                                   reply)
                finally:
                    bc._end_turn_grounding(prev)
                self.assertEqual(results, [])
                self.assertEqual(cleaned, reply)

    def test_an_answer_with_no_claim_is_untouched(self):
        reply = "[intent:amused] Notepad dates back to 1983, sir."
        cleaned, results = self._quiet(self.bc.parse_and_run_actions, reply)
        self.assertEqual(results, [])
        self.assertEqual(cleaned, reply)


@requires_monolith
class StreamFlushClaimGateTests(_Base):
    """The cloud route voices the first sentences while the reply streams.
    A claim sentence must not be voiced early either: the reply may carry no
    action token, and early speech cannot be unsaid."""

    def _flush(self, text):
        spoken: list[str] = []
        buf = self.bc._SentenceFlushBuffer(speak_fn=spoken.append)
        with contextlib.redirect_stdout(io.StringIO()):
            buf.feed(text)
            buf.join()
        return spoken

    def test_claim_sentences_are_never_flushed_early(self):
        for text in (
                "Very good, sir. I've closed every other window for you. "
                "Anything else you need?",
                "Very good, sir. The window has been closed. Anything else "
                "you need?",
                "Certainly, sir. I've taken the liberty of closing everything "
                "else. Anything else?"):
            with self.subTest(text=text):
                self.assertEqual(self._flush(text), [])

    def test_a_plain_answer_still_flushes_early(self):
        got = self._flush("The tower opened in 1889, sir. It was the tallest "
                          "structure of its day. More?")
        self.assertEqual(got, ["The tower opened in 1889, sir.",
                               "It was the tallest structure of its day."])
