"""v3.0.8 H1: "what's new" is spoken word for word, with no read-back round.

THE LIVE TURN (2026-10-28 19:47, 3.0.6 canary): "what's new" -> the version
line was voiced, then "[ack-hold] dropped the acknowledgement ... an action in
this reply failed" although read_changelog had NOT failed, and a follow-up
model round re-summarised the summary already written (end 22.7 s). The
summary says fixes as fixes ("no longer fails", "can't ... any more"), and
those words are FAILURE_MARKERS, so the successful result read as a failure:
the verbatim read-out skipped it and the failure follow-up loop re-prompted.

Drives the REAL _run_llm_dispatch_body -> parse_and_run_actions -> the REAL
core.actions._act_read_changelog (a temporary release history, the summariser
stubbed) -> verbatim read-out / follow-up loop, with a recording _speak (the
answer-first harness). Names and text are neutral. No audio, no LLM.

    python -m unittest tests.monolith.test_monolith_v308_whats_new
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_answer_first import _Base

_HISTORY = (
    "# Release history\n\n"
    "## v9.1.2 (2026-01-02)\n\nTimer and search fixes. A timer no longer "
    "invents a label.\n\n"
    "## v9.1.1 (2026-01-01)\n\nOlder entry.\n")
_VERSION = "I'm on version 9.1.2, sir."
# The summariser's answer: fixes said as fixes, full of FAILURE_MARKERS words.
_SUMMARY = ("A timer no longer invents a label you didn't give it, and a "
            "failed search can't leak an action name into speech any more.")
_FOLLOWUP = "Here is a second summary of the same changes, sir."


@requires_monolith
class _WhatsNewBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        # A confirmed voicing (`is True`), as the live _speak returns.
        self._p(bc, "_speak", side_effect=self._say)
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        os.makedirs(os.path.join(self._td.name, "docs"))
        with open(os.path.join(self._td.name, "docs", "RELEASE_HISTORY.md"),
                  "w", encoding="utf-8") as f:
            f.write(_HISTORY)
        self._p(bc, "__file__",
                os.path.join(self._td.name, "bobert_companion.py"))
        self.llm_quick = self._p(bc, "_llm_quick", return_value=_SUMMARY)
        import core.config as cfg
        self._p(cfg, "WHATS_NEW_VERSION_FIRST", True, create=True)
        self._p(bc, "VERBATIM_CONTENT_RESULTS", True, create=True)
        self.followup.side_effect = [_FOLLOWUP, ""]
        # The REAL handler, not a stub.
        from core import actions as A
        self.assertIs(bc.ACTIONS["read_changelog"], A._act_read_changelog)

    def _say(self, t, *a, **k):
        self.spoken.append(t)
        return True


class LiveWhatsNewTests(_WhatsNewBase):
    def test_live_turn_summary_is_spoken_once_with_no_read_back(self):
        printed = self._run("Right away, sir. [ACTION: read_changelog]",
                            text="what's new")
        self.assertEqual(self.spoken[0], _VERSION)
        self.assertIn(_SUMMARY, self.spoken)
        self.assertEqual(self.spoken.count(_SUMMARY), 1)
        self.followup.assert_not_called()
        self.assertNotIn("Reading results", printed)
        # read_changelog did not fail, so nothing says it did.
        self.assertNotIn("[ack-hold]", printed)
        self.llm_quick.assert_called_once()

    def test_every_alias_is_a_content_result(self):
        for alias in ("whats_new", "what_changed", "show_changelog",
                      "recent_changes"):
            with self.subTest(alias=alias):
                self.spoken.clear()
                self.followup.reset_mock()
                self.followup.side_effect = [_FOLLOWUP, ""]
                self._run(f"[ACTION: {alias}]", text="what's new")
                self.assertIn(_SUMMARY, self.spoken)
                self.followup.assert_not_called()

    def test_a_summary_opening_with_a_failure_word_is_still_spoken(self):
        # H1 follow-up: a fix release's summary can open with "Failed ..." or
        # "Error ..."; only the action's own failure wording is a failure.
        for summary in (
                "Failed prints are now announced once, and errors no longer "
                "repeat.",
                "Error messages from the printer are now spoken once.",
                "Could not-reach alerts are now spoken once.",
                "Couldn't-connect notices no longer repeat."):
            with self.subTest(summary=summary):
                self.spoken.clear()
                self.followup.reset_mock()
                self.followup.side_effect = [_FOLLOWUP, ""]
                self.llm_quick.return_value = summary
                printed = self._run("Right away, sir. [ACTION: read_changelog]",
                                    text="what's new")
                self.assertEqual(self.spoken[0], _VERSION)
                self.assertEqual(self.spoken.count(summary), 1)
                self.followup.assert_not_called()
                self.assertNotIn("[ack-hold]", printed)

    def test_a_real_failure_still_gets_the_failure_follow_up(self):
        self.llm_quick.side_effect = RuntimeError("summariser down")
        printed = self._run("Right away, sir. [ACTION: read_changelog]",
                            text="what's new")
        self.followup.assert_called()
        self.assertIn(_FOLLOWUP, self.spoken)
        self.assertIn("[ack-hold] dropped the acknowledgement (3 words): "
                      "read_changelog failed", printed)

    def test_setting_off_is_the_old_word_test(self):
        self._p(self.bc, "VERBATIM_CONTENT_RESULTS", False, create=True)
        self._run("[ACTION: read_changelog]", text="what's new")
        # The old path: the summary reads as a failure and is re-prompted.
        self.followup.assert_called()
        self.assertNotIn(_SUMMARY, self.spoken)


@requires_monolith
class AckHoldReasonTests(_Base):
    """The [ack-hold] line names what really happened (H1 second half)."""

    def setUp(self):
        super().setUp()
        self.result = ["I won't send the rover out while it is charging, sir."]
        self.bc.ACTIONS["rover_explore"] = lambda a="": self.result[0]

    def test_a_refusal_is_logged_as_refused_not_failed(self):
        bc = self.bc
        self._p(bc, "SPEAK_RESULT_VERBATIM_ACTIONS",
                set(bc.SPEAK_RESULT_VERBATIM_ACTIONS) | {"rover_explore"})
        # Answer-first would drop the lead-in first; the ack-hold is the
        # rule under test.
        self._p(bc, "ANSWER_FIRST_ENABLED", False)
        printed = self._run("Right away, sir. [ACTION: rover_explore, 60]",
                            text="send the rover out")
        self.assertIn("[ack-hold] dropped the acknowledgement (3 words): "
                      "rover_explore refused", printed)
        self.assertNotIn("an action in this reply failed", printed)

    def test_a_failure_names_the_action(self):
        self.result[0] = "could not reach the rover"
        self.followup.side_effect = ["The rover is out of reach, sir.", ""]
        printed = self._run("Right away, sir. [ACTION: rover_explore, 60]",
                            text="send the rover out")
        self.assertIn("(3 words): rover_explore failed", printed)

    def test_reason_helper(self):
        r = self.bc._ack_hold_reason
        self.assertEqual(r([("a_x", "could not do it", False)]),
                         "a_x failed")
        self.assertEqual(r([("read_changelog", _SUMMARY, False)]),
                         "an action in this reply failed")
        self.assertEqual(r(None), "an action in this reply failed")


@requires_monolith
class ContentResultHelperTests(_Base):
    def test_content_results_fail_only_on_their_own_failure_words(self):
        f = self.bc._action_result_failed
        with mock.patch.object(self.bc, "VERBATIM_CONTENT_RESULTS", True,
                               create=True):
            self.assertFalse(f(_SUMMARY, "read_changelog"))
            self.assertFalse(f(_SUMMARY, "Recent_Changes"))
            self.assertTrue(f("could not read the changelog: x",
                              "read_changelog"))
            self.assertTrue(f("could not summarise the changelog: x",
                              "whats_new"))
            self.assertFalse(f("Failed prints are now announced once.",
                               "read_changelog"))
            self.assertFalse(f("Error messages are now spoken once.",
                               "read_changelog"))
            self.assertTrue(f("Sorry (action failed; class=X; y)",
                              "read_changelog"))
            # Every other action keeps the word test.
            self.assertTrue(f(_SUMMARY, "web_search"))
            self.assertTrue(f(_SUMMARY))
        with mock.patch.object(self.bc, "VERBATIM_CONTENT_RESULTS", False,
                               create=True):
            self.assertTrue(f(_SUMMARY, "read_changelog"))


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
