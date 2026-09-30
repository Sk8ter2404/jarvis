"""Skill utterance routes (v2.0.142, 2026-09-29).

Live: "Jarvis, talk to the <device>." was answered with chat ("I'll see if I
can get him to cooperate, sir.") and the device skill's action never ran;
speech-to-text also spelled the device's name three ways. A skill can now
register a route -- callable(text) -> "[ACTION: name, arg]" | None -- that
claims an exact request BEFORE the LLM. The token becomes the turn's reply, so
the action runs through the normal path and the LLM is never called.

    python -m unittest tests.monolith.test_monolith_utterance_routes
"""
from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base


@requires_monolith
class UtteranceRouteDispatchTests(_Base):

    def setUp(self):
        # _Base stubs the confirmation gate out; the confirmation test needs
        # the REAL one (a keyword inside the owner's free-text topic).
        self._real_needs_confirmation = self.bc._needs_confirmation
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(bc, "SKILL_ROUTES_ENABLED", True, create=True)
        self.history: list = []
        self._p(bc, "conversation_history", self.history)
        self.chat = self._stub("desk_chat", "Chat finished.")
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value="[intent:wry] A riveting chat, sir.")
        self._p(bc, "get_followup_response", side_effect=[None] * 8)

    def _run(self, text):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc._run_llm_dispatch(text)
        return buf.getvalue()

    def _route(self, fn, name="desk device"):
        self.assertTrue(self.bc.register_utterance_route(fn, name))

    def test_a_route_runs_the_action_and_skips_the_llm(self):
        self._route(lambda t: ("[ACTION: desk_chat, pizza]"
                               if "talk to the desk device" in t.lower() else None))
        out = self._run("Jarvis, talk to the desk device about pizza.")
        self.assertEqual(self.calls["desk_chat"], ["pizza"])
        self.llm.assert_not_called()
        self.assertIn("[skill-route] desk device -> desk_chat", out)
        self.assertEqual(self.history[-2]["content"],
                         "Jarvis, talk to the desk device about pizza.")

    def test_a_routed_turn_records_the_owner_words_for_skills(self):
        # A device skill checks _last_user_text before starting (the transcript
        # is the authority). _call_llm sets it; a routed turn skips _call_llm,
        # so without this the skill would judge the PREVIOUS sentence.
        self._p(self.bc, "_last_user_text", ["what time is it"])
        seen = []
        self.bc.ACTIONS["desk_chat"] = (
            lambda arg="": seen.append(self.bc._last_user_text[0]) or "ok")
        self._route(lambda t: "[ACTION: desk_chat, pizza]")
        self._run("talk to the desk device about pizza")
        self.assertEqual(seen, ["talk to the desk device about pizza"])

    def test_no_match_goes_to_the_llm(self):
        self._route(lambda t: None)
        self._run("what's the capital of France")
        self.llm.assert_called_once()
        self.assertEqual(self.calls["desk_chat"], [])

    def test_junk_or_unknown_action_is_ignored(self):
        for token in ("sure thing", "[ACTION: no_such_action, x]",
                      "[ACTION: desk_chat, x] and more", "[ACTION desk_chat]"):
            with self.subTest(token=token):
                self.bc._UTTERANCE_ROUTES[:] = []
                self._route(lambda t, _tok=token: _tok)
                self.llm.reset_mock()
                out = self._run("talk to the desk device")
                self.llm.assert_called_once()
                self.assertIn("invalid route", out)
        self.assertEqual(self.calls["desk_chat"], [])

    def test_a_crashing_route_falls_through(self):
        def boom(_t):
            raise RuntimeError("route bug")
        self._route(boom)
        out = self._run("talk to the desk device")
        self.llm.assert_called_once()
        self.assertIn("[skill-route] desk device failed: RuntimeError", out)

    def test_the_kill_switch_sends_everything_to_the_llm(self):
        self._p(self.bc, "SKILL_ROUTES_ENABLED", False, create=True)
        self._route(lambda t: "[ACTION: desk_chat]")
        self._run("talk to the desk device")
        self.llm.assert_called_once()
        self.assertEqual(self.calls["desk_chat"], [])

    def test_same_name_replaces_a_route(self):
        self._route(lambda t: "[ACTION: no_such_action]")
        self._route(lambda t: "[ACTION: desk_chat, second]")
        self.assertEqual(len(self.bc._UTTERANCE_ROUTES), 1)
        self._run("talk to the desk device")
        self.assertEqual(self.calls["desk_chat"], ["second"])

    def test_non_callable_is_refused(self):
        self.assertFalse(self.bc.register_utterance_route("nope", "x"))

    def test_a_self_voiced_route_speaks_nothing_more(self):
        self._p(self.bc, "SELF_VOICED_ACTIONS", {"desk_chat"})
        self._route(lambda t: "[ACTION: desk_chat, pizza]")
        out = self._run("talk to the desk device about pizza")
        self.assertEqual(self.calls["desk_chat"], ["pizza"])
        self.assertIn("[self-voiced]", out)
        self.assertEqual(self.spoken, [])

    def test_skill_utils_exposes_the_hook(self):
        self.assertIn("register_utterance_route", self.bc.skill_utils)

    def test_pc_control_off_leaves_the_turn_to_the_llm(self):
        # PC control off: parse_and_run_actions runs NO action, so a routed
        # token would be spoken aloud as text ("[ACTION: desk chat, pizza]")
        # and stored as the reply. The LLM keeps the turn instead.
        self._p(self.bc, "PC_CONTROL_ENABLED", False)
        self._route(lambda t: "[ACTION: desk_chat, pizza]")
        out = self._run("Jarvis, talk to the desk device about pizza.")
        self.llm.assert_called_once()
        self.assertEqual(self.calls["desk_chat"], [])
        self.assertNotIn("[skill-route]", out)
        for line in self.spoken + [m["content"] for m in self.history]:
            self.assertNotIn("[ACTION", line)

    def test_a_confirmable_self_voiced_route_asks_aloud_and_yes_runs_it(self):
        # The owner's topic "information" contains the CONFIRM_KEYWORDS
        # substring "format", so the REAL gate defers the self-voiced action.
        # A deferred action has said nothing yet: the "say 'yes'" question
        # must be spoken (it used to be blanked as self-voiced, so JARVIS
        # waited silently and the next sentence cancelled it).
        bc = self.bc
        self._p(bc, "_needs_confirmation", self._real_needs_confirmation)
        self._p(bc, "CONFIRM_KEYWORDS", ["format"])
        self._p(bc, "_pending_confirmation", [])
        self._p(bc, "SELF_VOICED_ACTIONS", {"desk_chat"})
        self._route(lambda t: "[ACTION: desk_chat, information]")
        out = self._run("Jarvis, talk to the desk device about information.")
        self.assertIn("REQUIRES CONFIRMATION: desk_chat(information)", out)
        self.assertNotIn("[self-voiced]", out)
        self.llm.assert_not_called()
        self.assertEqual(self.calls["desk_chat"], [])
        self.assertEqual(bc._pending_confirmation, [("desk_chat", "information")])
        self.assertEqual(len(self.spoken), 1, self.spoken)
        self.assertIn("say 'yes' to proceed", self.spoken[0])
        # ... and the owner's "yes" runs the deferred action.
        self.spoken.clear()
        self.assertTrue(self._quiet(bc.handle_confirmation_response, "yes"))
        self.assertEqual(self.calls["desk_chat"], ["information"])
        self.assertEqual(bc._pending_confirmation, [])
        self.assertNotIn("Cancelled.", self.spoken)

    def test_a_self_voiced_result_that_ran_is_still_silent(self):
        self._p(self.bc, "SELF_VOICED_ACTIONS", {"desk_chat"})
        self.assertTrue(self.bc._all_self_voiced(
            [("desk_chat", "Chat finished.", False)]))
        for prefix in self.bc._ANSWER_FIRST_DEFERRED_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertFalse(self.bc._all_self_voiced(
                    [("desk_chat", prefix + " desk_chat(x)", False)]))

    def test_a_routed_turn_classifies_its_own_mood(self):
        # _call_llm stamps this utterance's tone / voice mood / emotion, which
        # TTS reads for every line the turn speaks. A routed turn skips
        # _call_llm, so without its own classification the action's lines
        # were voiced with the PREVIOUS turn's mood ("I'm exhausted" ->
        # tired / late_night / the 'concerned' preset).
        bc = self.bc
        stale_er = types.SimpleNamespace(label="tired", reason="prior",
                                         addendum="", tts_preset="concerned")
        self._p(bc, "_last_user_tone", ["tired"])
        self._p(bc, "_last_voice_route", [{"mood": "late_night", "addendum": ""}])
        self._p(bc, "_last_emotion", [stale_er])
        fresh_er = types.SimpleNamespace(label=None, reason="", addendum="",
                                         tts_preset=None)
        tone = self._p(bc, "detect_tone", return_value=None)
        mood = self._p(bc, "route_voice_emotion",
                       return_value={"mood": "casual", "addendum": ""})
        tracker = mock.Mock()
        tracker.classify_emotion.return_value = fresh_er
        self._p(bc, "_emotion_tracker", tracker)
        seen = []
        bc.ACTIONS["desk_chat"] = lambda arg="": seen.append(
            (bc._synth_user_tone(), bc._last_emotion[0])) or "ok"
        text = "Jarvis, talk to the desk device about pizza."
        self._route(lambda t: "[ACTION: desk_chat, pizza]")
        self._run(text)
        self.llm.assert_not_called()
        tone.assert_called_once_with(text)
        mood.assert_called_once_with(text)
        tracker.classify_emotion.assert_called_once_with(text)
        self.assertEqual(seen, [(None, fresh_er)])

    def test_call_llm_and_the_route_share_one_mood_classifier(self):
        import inspect
        self.assertIn("_classify_turn_mood(user_text)",
                      inspect.getsource(self.bc._call_llm))
        body = inspect.getsource(self.bc._run_llm_dispatch_body)
        self.assertIn("_classify_turn_mood(text)", body)
        # one copy of the classification, not two
        self.assertNotIn("detect_tone(", body)
        self.assertNotIn("detect_tone(", inspect.getsource(self.bc._call_llm))


if __name__ == "__main__":
    unittest.main()
