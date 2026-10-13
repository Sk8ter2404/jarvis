"""The last open July-audit findings, closed 2026-10-13.

A40  The shutdown prompt's overnight "yes" ignored what the overnight action
     returned: with the engine switched off nothing ran, JARVIS stayed up and
     said nothing. Its reply is now spoken (the success line too).
A82  The local cheatsheet was cached once and never rebuilt, so an action
     registered later was missing. It is now keyed on the action count.
A83  _speak's nothing-left-to-say return cleared the shared _last_* voice
     cells outside _SPEAK_LOCK, stomping on a concurrent speaker's prosody.
     It now leaves them alone, like the mute path.
(A47, the wake listener's Porcupine carry on pause, is in tests/test_wake_word.py.)

    python -m unittest tests.monolith.test_monolith_july_audit_1013
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m


@requires_monolith
class OvernightYesIsAnsweredTests(_Base):
    """A40: the overnight action's own reply is spoken."""

    def setUp(self):
        bc = self.bc
        self.spoken = []
        self._p(bc, "_speak", lambda text, *a, **k: self.spoken.append(text))
        self._p(bc, "_write_hud_state", lambda **k: None)
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self._p(bc, "record_action_error", lambda *a, **k: None)
        self._p(bc._cmd_autocorrect, "_embed_disabled", True)
        self.shutdown = mock.Mock(return_value="Goodbye, sir.")
        self._p(bc, "_act_shutdown_jarvis", self.shutdown)
        self._p(bc, "_shutdown_prompt_pending",
                {"armed": True, "expires_at": 9e18})

    def _yes(self, text, overnight):
        self._p(self.bc, "_act_start_overnight_upgrade", overnight)
        self.assertTrue(self.bc._handle_shutdown_prompt(text))
        overnight.assert_called_once()
        self.shutdown.assert_not_called()

    def test_engine_off_reply_is_spoken(self):
        off = ("The overnight upgrade engine is switched off, sir, so "
               "there's nothing to run - I'm staying awake.")
        self._yes("Yes, run the overnight protocol, then shut down.",
                  mock.Mock(return_value=off))
        self.assertEqual(self.spoken, [off])

    def test_success_reply_is_spoken(self):
        self._yes("Yes, overnight protocol.",
                  mock.Mock(return_value="On it, sir."))
        self.assertEqual(self.spoken, ["On it, sir."])

    def test_a_failing_action_still_answers(self):
        self._yes("Run the overnight protocol.",
                  mock.Mock(side_effect=RuntimeError("boom")))
        self.assertEqual(len(self.spoken), 1)
        self.assertIn("staying awake", self.spoken[0])


@requires_monolith
class CheatsheetFollowsTheActionRegistryTests(_Base):
    """A82: an action registered after the first build is listed."""

    def test_a_late_action_appears_after_a_rebuild(self):
        bc = self.bc
        acts = {"get_time": lambda *_: "noon"}
        self._p(bc, "ACTIONS", acts)
        self._p(bc, "_LOCAL_CHEATSHEET_CACHE", [None])
        first = bc._local_cheatsheet()
        self.assertNotIn("late_added_action_x", first)
        self.assertIs(bc._local_cheatsheet(), first)      # cached
        acts["late_added_action_x"] = lambda *_: "ok"
        self.assertIn("late_added_action_x", bc._local_cheatsheet())


@requires_monolith
class EmptySpeechLeavesTheVoiceCellsAloneTests(_Base):
    """A83: nothing audible -> return without touching the _last_* cells."""

    def test_tag_only_text_does_not_clear_another_speakers_cells(self):
        bc = self.bc
        self._p(bc, "_write_hud_state", lambda **k: None)
        self._p(bc, "_session_start_time", 0.0)
        self._p(bc, "_tts_muted", [False])
        self._p(bc, "_last_intent_override", ["amused"])
        self._p(bc, "_last_wry", [True])
        self._p(bc, "_last_mood", ["dry_amused"])
        synth = self._p(bc, "synthesise", side_effect=AssertionError(
            "nothing audible must never reach synthesis"))
        bc._speak("[intent:amused]   ")
        synth.assert_not_called()
        self.assertEqual(bc._last_intent_override, ["amused"])
        self.assertEqual(bc._last_wry, [True])
        self.assertEqual(bc._last_mood, ["dry_amused"])


if __name__ == "__main__":
    unittest.main()
