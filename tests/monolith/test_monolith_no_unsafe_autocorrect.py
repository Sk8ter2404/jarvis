"""No unsafe autocorrect (2026-10-01).

Live: "Jarvis, turn it off" reached the local model with nothing in play for
"it" to mean. The model invented [ACTION: shutdown], and parse_and_run_actions'
fuzzy action-name corrector silently mapped that onto shutdown_jarvis (0.78
over the 0.75 floor) - JARVIS shut itself down.

Pinned here, against the REAL corrector (embeddings off, so no Ollama call):
  * a guessed name never lands on a protected action (the set is derived from
    _FIRE_AND_EXIT_ACTIONS, _DESTRUCTIVE_REPLAY_ACTIONS, CONFIRM_KEYWORDS and
    core.action_risk, plus any alias bound to the same handler). The guess is
    dropped, the model's prose ("Shutting down, sir.") is not voiced, and
    JARVIS asks what to turn off - with no failure follow-up round in which
    the model could name shutdown_jarvis itself;
  * a "did you mean" pick never runs one either;
  * a bare "turn it off" with no recent turn, action or JARVIS-started media
    asks "Turn what off, sir?" before the LLM is ever called.
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def setUp(self):
        bc = self.bc
        self.spoken = []
        self._p(bc, "_speak", lambda text, *a, **k: self.spoken.append(text))
        self._p(bc, "_write_hud_state", lambda **k: None)
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self._p(bc, "record_action_error", lambda *a, **k: None)
        self._p(bc, "PC_CONTROL_ENABLED", True)
        # The real corrector, lexical only: never touch the live Ollama.
        self._p(bc._cmd_autocorrect, "_embed_disabled", True)
        self.shutdown = mock.Mock(return_value="Goodbye, sir.")
        self.shot = mock.Mock(return_value="screenshot saved")
        self.acts = {
            "shutdown_jarvis": self.shutdown,
            "shut_down": self.shutdown,
            "screenshot": self.shot,
            "get_time": mock.Mock(return_value="noon"),
            "play_music": mock.Mock(return_value="playing"),
        }
        self._p(bc, "ACTIONS", self.acts)

    def _dispatch(self, reply, user_text="Jarvis, turn it off."):
        bc = self.bc
        prev = bc._begin_turn_grounding(user_text)
        try:
            return bc.parse_and_run_actions(reply)
        finally:
            bc._end_turn_grounding(prev)


@requires_monolith
class InventedNameNeverShutsJarvisDownTests(_Base):
    def test_invented_shutdown_is_dropped_and_jarvis_asks(self):
        bc = self.bc
        cleaned, results = self._dispatch(
            "Shutting down, sir. [ACTION: shutdown]")
        self.shutdown.assert_not_called()
        self.assertEqual(cleaned, "Turn what off, sir?")
        self.assertEqual(len(results), 1)
        name, res, info = results[0]
        self.assertEqual(name, "shutdown")
        self.assertTrue(res.startswith("⚠  UNCLEAR:"), res)
        self.assertIn("shutdown_jarvis", res)
        # Not a failure and not informative: the follow-up loop never
        # re-prompts the model, so it cannot name shutdown_jarvis itself.
        self.assertFalse(info)
        self.assertFalse(bc._action_result_failed(res))
        self.assertTrue(res.startswith(bc._ANSWER_FIRST_DEFERRED_PREFIXES))
        self.assertFalse(any("Interpreting" in s for s in self.spoken))

    def test_other_utterances_get_a_generic_question(self):
        cleaned, _results = self._dispatch("[ACTION: shutdown]",
                                           user_text="do the thing")
        self.shutdown.assert_not_called()
        self.assertEqual(cleaned,
                         self.bc._pronoun_switch.GENERIC_QUESTION)

    def test_an_alias_of_a_protected_handler_is_protected(self):
        # A name no classification knows, bound to the shutdown handler.
        self.acts["bye_bye"] = self.shutdown
        self.assertTrue(self.bc._autocorrect_protected("bye_bye"))
        cleaned, results = self._dispatch("[ACTION: bye_by]")
        self.shutdown.assert_not_called()
        self.assertTrue(results[0][1].startswith("⚠  UNCLEAR:"))

    def test_a_benign_typo_still_routes(self):
        cleaned, results = self._dispatch("[ACTION: screen_shot]",
                                          user_text="take a screenshot")
        self.shot.assert_called_once()
        self.assertEqual(results[0][0], "screenshot")

    def test_protected_set_comes_from_the_existing_classifications(self):
        bc = self.bc
        for name in sorted(bc._FIRE_AND_EXIT_ACTIONS
                           | bc._DESTRUCTIVE_REPLAY_ACTIONS):
            with self.subTest(name=name):
                self.assertTrue(bc._autocorrect_protected(name))
        for name in ("delete_note", "forget_face", "switch_llm"):
            with self.subTest(name=name):
                self.assertTrue(bc._autocorrect_protected(name))
        for name in ("screenshot", "get_time", "play_music"):
            with self.subTest(name=name):
                self.assertFalse(bc._autocorrect_protected(name))


@requires_monolith
class DisambigPickNeverShutsJarvisDownTests(_Base):
    def test_a_did_you_mean_pick_of_shutdown_is_refused(self):
        bc = self.bc
        bc._pending_autocorrect_choice.clear()
        bc._pending_autocorrect_choice.append({
            "primary": ("shutdown_jarvis", ""),
            "secondary": ("screenshot", ""),
            "original": "shutdwn",
        })
        self.assertTrue(bc.handle_autocorrect_disambig_response("the first"))
        self.shutdown.assert_not_called()
        self.assertTrue(any("won't run" in s for s in self.spoken),
                        self.spoken)


@requires_monolith
class PronounSwitchShortcutTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "set_state", lambda *a, **k: None)
        self._p(bc, "_prev_owner_turn_at", [0.0])
        self._p(bc, "_jarvis_played_music_at", [0.0])
        bc._action_history.clear()
        import core.config as cfg
        self._p(cfg, "KINECT_POINT_CONTROL_ENABLED", False, create=True)
        self.hist_len = len(bc.conversation_history)

    def test_no_referent_asks_before_the_llm(self):
        bc = self.bc
        self.assertTrue(bc._run_pronoun_switch_shortcut("Jarvis, turn it off."))
        self.assertEqual(self.spoken, ["Turn what off, sir?"])
        self.assertEqual(bc.conversation_history[self.hist_len:], [
            {"role": "user", "content": "Jarvis, turn it off."},
            {"role": "assistant", "content": "Turn what off, sir?"},
        ])

    def test_it_is_wired_into_the_voice_shortcuts(self):
        self.assertTrue(self.bc._run_voice_shortcuts("Jarvis, turn that off."))
        self.assertIn("Turn what off, sir?", self.spoken)

    def test_a_recent_turn_lets_the_llm_resolve_it(self):
        self.bc._prev_owner_turn_at[0] = time.monotonic() - 20.0
        self.assertFalse(self.bc._run_pronoun_switch_shortcut("turn it off"))
        self.assertEqual(self.spoken, [])

    def test_a_recent_action_lets_the_llm_resolve_it(self):
        self.bc._action_history.append({"action": "play_music", "arg": "x",
                                        "result": "ok",
                                        "at": time.time() - 30.0})
        self.assertFalse(self.bc._run_pronoun_switch_shortcut("turn it off"))

    def test_media_jarvis_started_lets_the_llm_resolve_it(self):
        self.bc._jarvis_played_music_at[0] = time.time() - 600.0
        self.assertFalse(self.bc._run_pronoun_switch_shortcut("turn it off"))

    def test_point_to_control_keeps_pronoun_commands(self):
        import core.config as cfg
        with mock.patch.object(cfg, "KINECT_POINT_CONTROL_ENABLED", True,
                               create=True):
            self.assertFalse(
                self.bc._run_pronoun_switch_shortcut("turn that off"))

    def test_fast_paths_kill_switch(self):
        self._p(self.bc, "FAST_PATHS_ENABLED", False, create=True)
        self.assertFalse(self.bc._run_pronoun_switch_shortcut("turn it off"))

    def test_a_named_target_is_never_intercepted(self):
        self.assertFalse(
            self.bc._run_pronoun_switch_shortcut("turn off the lamp"))

    def test_owner_turn_keeps_the_previous_stamp(self):
        bc = self.bc
        self._p(bc, "_last_owner_turn_at", [0.0])
        bc._note_owner_turn()
        first = bc._last_owner_turn_at[0]
        bc._note_owner_turn()
        self.assertEqual(bc._prev_owner_turn_at[0], first)


if __name__ == "__main__":
    unittest.main()
