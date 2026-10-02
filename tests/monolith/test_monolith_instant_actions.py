"""INSTANT_ACTIONS wired into _run_llm_dispatch_body (2026-10-02).

Volume, media transport, lights on/off and "pause the print" can run with no
LLM call (core/instant_actions.py has the rules; tests/test_instant_actions.py
pins them). Three modes:
  shadow (default)  the brain answers; "[instant] would run X without the
                    brain" is logged and, after the first
                    parse_and_run_actions, whether the brain ran X goes to
                    data/instant_actions.jsonl (time + action names only);
  on                X's token is the reply and runs like an utterance route:
                    the LLM is never called and no follow-up round restates
                    it, but a FAILED action still gets the failure follow-up;
  off               nothing is matched or logged.
The exclusions (questions, polite asks, two commands, confirmation / pushback
/ protected gates, the allowlist, an unregistered action) are pinned here on
the REAL dispatch path: in mode "on" a miss means the brain is asked.

Every LLM call, action and speech output is stubbed; the log goes to a temp
dir. "zebra" is a stand-in transcript word that must never reach the log.

    python -m unittest tests.monolith.test_monolith_instant_actions
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base


@requires_monolith
class _InstantBase(_Base):

    MODE = "shadow"

    def setUp(self):
        super().setUp()
        bc = self.bc
        from core import config as _cfg
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(bc, "INSTANT_ACTIONS_MODE", self.MODE, create=True)
        self._p(bc, "INSTANT_ACTIONS_ALLOW", list(_cfg.INSTANT_ACTIONS_ALLOW),
                create=True)
        self.history: list = []
        self._p(bc, "conversation_history", self.history)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmpdir = tmp.name
        self.log = os.path.join(tmp.name, "instant_actions.jsonl")
        self._real_log_path = bc._instant_actions_log_path
        self._p(bc, "_instant_actions_log_path", return_value=self.log)
        # pause_music's result is restated by a follow-up round on the brain
        # path; pause_print / smart_home_control are spoken verbatim (the
        # latter is a skill's declaration, folded in at load time).
        self._p(bc, "INFORMATIVE_ACTIONS",
                set(bc.INFORMATIVE_ACTIONS) | {"pause_music"})
        self._p(bc, "SPEAK_RESULT_VERBATIM_ACTIONS",
                set(bc.SPEAK_RESULT_VERBATIM_ACTIONS)
                | {"pause_print", "smart_home_control"})
        self._stub("pause_music", "paused the media player, sir")
        self._stub("volume_up", "volume up")
        self._stub("volume_down", "volume down")
        self._stub("next_song", "skipped to the next track, sir")
        self._stub("pause_print", "Pausing the print, sir.")
        self.lights = self._stub("smart_home_control",
                                 "The office light is off, sir.")
        self._actions["control_device"] = self.lights
        self.calls["control_device"] = self.calls["smart_home_control"]
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value="Right away, sir. [ACTION: pause_music]")
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=["It is paused, sir."] + [None] * 8)

    def _run(self, text):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc._run_llm_dispatch(text)
        return buf.getvalue()

    def _rows(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log, encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]


class ShadowModeTests(_InstantBase):

    def test_the_brain_answers_and_agreement_is_logged(self):
        out = self._run("Jarvis, pause the music.")
        self.llm.assert_called_once()
        self.assertEqual(self.calls["pause_music"], [""],
                         "the action must run ONCE, from the brain's reply")
        self.assertIn("[instant] would run pause_music without the brain", out)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual({k: v for k, v in rows[0].items() if k != "ts"},
                         {"mode": "shadow", "action": "pause_music",
                          "brain": ["pause_music"], "agree": True})
        self.assertIsInstance(rows[0]["ts"], float)

    def test_the_turn_is_otherwise_unchanged(self):
        self._run("pause the music")
        # The informative follow-up still restates the result on the brain path.
        self.gfr.assert_called_once()
        self.assertIn("It is paused, sir.", self.spoken)

    def test_a_different_brain_action_is_a_disagreement(self):
        self.llm.return_value = "[ACTION: volume_down]"
        self._run("volume up")
        self.assertEqual(self.calls["volume_up"], [])
        self.assertEqual(self.calls["volume_down"], [""])
        row = self._rows()[0]
        self.assertEqual((row["action"], row["brain"], row["agree"]),
                         ("volume_up", ["volume_down"], False))

    def test_a_brain_reply_with_no_action_is_a_disagreement(self):
        self.llm.return_value = "[intent:wry] A fine track, sir."
        self._run("next song")
        row = self._rows()[0]
        self.assertEqual((row["action"], row["agree"]), ("next_song", False))
        self.assertEqual(row["brain"], [])

    def test_an_alias_of_the_same_handler_agrees(self):
        self.llm.return_value = "[ACTION: control_device, turn off the lights]"
        self._run("turn off the lights")
        row = self._rows()[0]
        self.assertEqual((row["action"], row["brain"], row["agree"]),
                         ("smart_home_control", ["control_device"], True))

    def test_the_owners_words_never_reach_the_log(self):
        self.llm.return_value = ("[ACTION: smart_home_control, turn off the "
                                 "zebra light]")
        self._run("turn off the zebra light")
        self.assertEqual(self.calls["smart_home_control"],
                         ["turn off the zebra light"])
        with open(self.log, encoding="utf-8") as fh:
            raw = fh.read()
        self.assertTrue(raw.strip())
        self.assertNotIn("zebra", raw)
        self.assertNotIn("light", raw)

    def test_a_glance_reply_is_not_scored(self):
        self.bc.maybe_glance_response.return_value = "A settings dialog, sir."
        self._run("pause the music")
        self.llm.assert_not_called()
        self.assertEqual(self._rows(), [])

    def test_a_turn_no_rule_matches_logs_nothing(self):
        out = self._run("what's the weather like")
        self.assertNotIn("[instant]", out)
        self.assertEqual(self._rows(), [])


class OnModeTests(_InstantBase):

    MODE = "on"

    def test_the_action_runs_and_the_llm_is_never_called(self):
        out = self._run("pause the music")
        self.llm.assert_not_called()
        self.gfr.assert_not_called()
        self.assertEqual(self.calls["pause_music"], [""])
        self.assertIn("[instant] pause_music without the brain", out)
        self.assertEqual(self.spoken, ["Paused the media player, sir."])
        row = self._rows()[0]
        self.assertEqual((row["mode"], row["action"], row["ok"]),
                         ("on", "pause_music", True))

    def test_the_route_bookkeeping_is_done(self):
        self._p(self.bc, "_last_user_text", ["what time is it"])
        self._p(self.bc, "_last_stable_sys_prompt", ["stale prompt"])
        self._p(self.bc, "_last_turn_pc_block", ["stale block"])
        self._run("Jarvis, next song")
        self.assertEqual(self.bc._last_user_text[0], "Jarvis, next song")
        self.assertEqual(self.bc._last_stable_sys_prompt[0], "")
        self.assertEqual(self.bc._last_turn_pc_block[0], "")
        self.assertEqual(self.history[-2]["content"], "Jarvis, next song")
        self.assertEqual(self.history[-1]["content"], "[ACTION: next_song]")

    def test_a_terse_result_gets_a_short_acknowledgement(self):
        self._run("volume up")
        self.llm.assert_not_called()
        self.assertEqual(self.calls["volume_up"], [""])
        self.assertEqual(self.spoken, ["Volume up, sir."])

    def test_a_verbatim_result_is_spoken_once(self):
        self._run("pause the print")
        self.llm.assert_not_called()
        self.assertEqual(self.calls["pause_print"], [""])
        self.assertEqual(self.spoken, ["Pausing the print, sir."])

    def test_lights_get_the_command_as_their_argument(self):
        self._run("Turn off the office light, please.")
        self.llm.assert_not_called()
        self.assertEqual(self.calls["smart_home_control"],
                         ["Turn off the office light"])
        self.assertEqual(self.spoken, ["The office light is off, sir."])

    def test_a_failed_action_still_gets_the_failure_follow_up(self):
        self._stub("pause_music",
                   "I couldn't reach the Windows media controls, sir.")
        self.gfr.side_effect = ["The media controls didn't answer, sir."] \
            + [None] * 8
        out = self._run("pause the music")
        self.llm.assert_not_called()
        self.gfr.assert_called_once()
        self.assertIn("did not run cleanly", out)
        self.assertEqual(self._rows()[0]["ok"], False)

    def test_a_skill_route_still_wins(self):
        self._stub("desk_chat", "Chat finished.")
        self.assertTrue(self.bc.register_utterance_route(
            lambda t: "[ACTION: desk_chat, music]" if "music" in t else None,
            "desk device"))
        out = self._run("pause the music")
        self.assertEqual(self.calls["desk_chat"], ["music"])
        self.assertEqual(self.calls["pause_music"], [])
        self.assertNotIn("[instant]", out)

    def test_pc_control_off_means_no_instant_action(self):
        self._p(self.bc, "PC_CONTROL_ENABLED", False)
        out = self._run("pause the music")
        self.llm.assert_called_once()
        self.assertNotIn("[instant]", out)
        self.assertEqual(self.calls["pause_music"], [])


class OnModeExclusionTests(_InstantBase):
    """Mode on: every excluded turn is answered by the brain."""

    MODE = "on"

    def _brain(self, text):
        out = self._run(text)
        self.llm.assert_called_once()
        self.llm.reset_mock()
        self.assertNotIn("[instant]", out, text)

    def test_questions_and_polite_asks(self):
        for text in ("is the music paused", "pause the music?",
                     "can you pause the music", "could you turn off the lights",
                     "would you mind pausing the music"):
            self._brain(text)
        self.assertEqual(self._rows(), [])

    def test_two_commands_pronouns_and_negation(self):
        for text in ("pause the music and turn the volume down",
                     "pause it", "turn it up", "don't pause the music"):
            self._brain(text)

    def test_a_confirmation_gate(self):
        self._p(self.bc, "_needs_confirmation",
                lambda n, a: n == "pause_music")
        self._brain("pause the music")

    def test_a_pushback_gate(self):
        self._p(self.bc, "_jarvis_pushback",
                lambda n, a: ("Are you certain, sir?", "test")
                if n == "volume_up" else None)
        self._brain("volume up")

    def test_a_protected_action(self):
        self._p(self.bc, "_autocorrect_protected",
                lambda n: n == "next_song")
        self._brain("next song")

    def test_a_faulting_gate_refuses(self):
        # (_autocorrect_protected: the one gate parse_and_run_actions does not
        # also call here - _Base turns the autocorrect layer off.)
        def boom(n):
            raise RuntimeError("gate fault")

        self._p(self.bc, "_autocorrect_protected", boom)
        self._brain("volume up")

    def test_the_allowlist(self):
        self._p(self.bc, "INSTANT_ACTIONS_ALLOW", ["volume_up"], create=True)
        self._brain("pause the music")
        out = self._run("volume up")
        self.llm.assert_not_called()
        self.assertIn("[instant] volume_up without the brain", out)

    def test_an_unregistered_action(self):
        self._actions.pop("pause_print", None)
        self._brain("pause the print")


class OffModeTests(_InstantBase):

    MODE = "off"

    def test_nothing_is_matched_or_logged(self):
        out = self._run("pause the music")
        self.llm.assert_called_once()
        self.assertNotIn("[instant]", out)
        self.assertEqual(self.calls["pause_music"], [""])
        self.assertFalse(os.path.exists(self.log))


class DefaultModeTests(_InstantBase):

    def test_an_unknown_mode_value_is_shadow(self):
        self._p(self.bc, "INSTANT_ACTIONS_MODE", "sometimes", create=True)
        self.assertEqual(self.bc._instant_actions_mode(), "shadow")
        out = self._run("pause the music")
        self.llm.assert_called_once()
        self.assertIn("[instant] would run pause_music", out)

    def test_the_log_path_is_the_staging_aware_data_dir(self):
        with mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.tmpdir}):
            self.assertEqual(self._real_log_path(),
                             os.path.join(self.tmpdir, "instant_actions.jsonl"))


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
