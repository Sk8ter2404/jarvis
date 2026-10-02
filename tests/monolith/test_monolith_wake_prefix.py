"""Monolith side of the wake-in-the-first-words fix (2026-10-01).

Live 2026-10-01 20:57, wake-word mode on: "What Jarvis what model are you?"
was dropped by the background-audio gate ("[bg-audio] wake-word mode —
ignoring non-wake utterance") because _text_has_wake_prefix only took
"Jarvis" as the FIRST word. Every wake-prefix test in the monolith now asks
core.wake_prefix:

  * the background-audio gate (_should_refuse_background_audio), and with it
    the follow-up window it opens;
  * the standby wake that carries a command (_standby_wake_carries_command);
  * the learn gate's ``wake=`` signal (_text_has_wake_prefix);
  * the main loop rewrites an admitted filler-led wake to the plain prefix
    form before any handler sees it (_wake_lead_canonical).

The standby music gate is pinned in tests/skills/test_standby_audio_detect.py
and the rule itself in tests/test_wake_prefix.py.

    python -m unittest tests.monolith.test_monolith_wake_prefix
"""
from __future__ import annotations

import contextlib
import inspect
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

LIVE_LINE = "What Jarvis what model are you?"


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        import core.config as _cfg
        self._cfg = _cfg
        saved = (_cfg.REQUIRE_WAKE_MODE, _cfg.AMBIENT_MUSIC_REFUSE_WAKE)
        self.addCleanup(self._restore_cfg, saved)
        self.spoken = []
        self._p(self.bc, "_speak",
                side_effect=lambda t, *a, **k: self.spoken.append(t))
        for name in ("_mic_muted", "_sleep_mode", "_standby_mode"):
            cell = getattr(self.bc, name)
            self.addCleanup(cell.__setitem__, 0, cell[0])
        self.bc._mic_muted[0] = False
        # Wake-word mode on, no music anywhere, no follow-up window.
        from core.followup_window import FollowupWindow
        self._p(self.bc, "_require_wake_runtime", True)
        self._p(self.bc, "_followup_window", FollowupWindow(0))
        self._p(self.bc, "_smtc_media_playing", return_value=False)
        self._p(self.bc, "_audio_music_should_refuse_wake", return_value=False)
        self.bc._standby_greet_admit_until[0] = 0.0
        self.addCleanup(self.bc._standby_greet_admit_until.__setitem__, 0, 0.0)

    def _restore_cfg(self, saved):
        self._cfg.REQUIRE_WAKE_MODE, self._cfg.AMBIENT_MUSIC_REFUSE_WAKE = saved

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _quiet(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()


class WakeModeGateTests(_Base):
    def test_the_live_line_passes_wake_word_mode(self):
        self.assertEqual(self.bc._should_refuse_background_audio(LIVE_LINE),
                         (False, ""))

    def test_a_filler_led_wake_passes(self):
        for text in ("Um, Jarvis, pause the music", "Oh Jarvis turn it down",
                     "Yo Jarvis what's up", "So, Jarvis, what's next",
                     "Alright, Jarvis.", "okay so Jarvis lights off",
                     "uh Jarvis set a timer", "hey Jarvis play jazz",
                     "All right, Jarvis, lights off",
                     "Jarvis, what time is it?"):
            with self.subTest(text=text):
                self.assertEqual(
                    self.bc._should_refuse_background_audio(text), (False, ""))

    def test_a_mid_sentence_mention_is_still_refused(self):
        for text in ("I asked Jarvis yesterday about the weather",
                     "tell Jarvis that dinner is ready",
                     "so um uh Jarvis play music",
                     "okay okay okay Jarvis",
                     "So Jarvis said it would rain",
                     "what time is it"):
            with self.subTest(text=text):
                self.assertEqual(
                    self.bc._should_refuse_background_audio(text),
                    (True, "wake-word mode"))

    def test_a_filler_led_wake_opens_the_follow_up_window(self):
        from core.followup_window import FollowupWindow
        self._p(self.bc, "_followup_window", FollowupWindow(45))
        self.assertEqual(self.bc._should_refuse_background_audio(LIVE_LINE),
                         (False, ""))
        self.assertEqual(
            self.bc._should_refuse_background_audio("and the context window"),
            (False, "follow-up window"))

    def test_the_learn_gate_sees_a_filler_led_wake(self):
        self.assertTrue(self.bc._text_has_wake_prefix(LIVE_LINE))
        self.assertFalse(self.bc._text_has_wake_prefix(
            "I asked Jarvis yesterday"))

    def test_the_monolith_rule_is_the_core_helper(self):
        from core import wake_prefix
        with mock.patch.object(wake_prefix, "has_wake_prefix",
                               return_value=True) as helper:
            self.assertTrue(self.bc._text_has_wake_prefix("anything"))
        helper.assert_called_once_with("anything")


class CanonicalCommandTextTests(_Base):
    def test_the_filler_is_dropped_from_the_command_text(self):
        out, log = self._quiet(self.bc._wake_lead_canonical, LIVE_LINE)
        self.assertEqual(out, "Jarvis what model are you?")
        # The rewritten text takes the legacy first-word path downstream.
        self.assertEqual(self.bc._yes_no.normalize("Um, Jarvis, yes."), "um jarvis yes")
        self.assertEqual(
            self.bc._yes_no.normalize(
                self._quiet(self.bc._wake_lead_canonical, "Um, Jarvis, yes.")[0]),
            "yes")
        self.assertIn("[wake]", log)
        self.assertNotIn("model", log)      # never the words

    def test_other_text_is_untouched_and_silent(self):
        for text in ("Jarvis, what time is it?", "hey Jarvis play jazz",
                     "Alright, Jarvis.", "what time is it",
                     "I asked Jarvis yesterday"):
            with self.subTest(text=text):
                out, log = self._quiet(self.bc._wake_lead_canonical, text)
                self.assertEqual(out, text)
                self.assertEqual(log, "")

    def test_never_raises(self):
        from core import wake_prefix
        with mock.patch.object(wake_prefix, "canonical_wake_text",
                               side_effect=RuntimeError("boom")):
            out, _ = self._quiet(self.bc._wake_lead_canonical, LIVE_LINE)
        self.assertEqual(out, LIVE_LINE)

    def test_the_main_loop_rewrites_after_the_gate_before_any_handler(self):
        src = inspect.getsource(self.bc.main)
        gate = src.index("_bg_gate_for_turn(")
        canon = src.index("text = _wake_lead_canonical(text)")
        self.assertLess(gate, canon)
        for later in ("_noise_verdict(text", 'print(f"  You:    {text}")',
                      "_note_session_owner_utterance(text)",
                      "_handle_sleep_triggers(text)",
                      "_run_voice_shortcuts(text)",
                      "_run_llm_dispatch(text"):
            with self.subTest(handler=later):
                self.assertLess(canon, src.index(later))


class StandbyCarriedCommandTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.tmp = tempfile.mkdtemp(prefix="wake_prefix_mono_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "context_aware_greeting", return_value=("Yes, sir?", 1.0))
        self._p(bc, "OVERNIGHT_FLAG_FILE", os.path.join(self.tmp, "none"))
        self._p(bc, "_learn_gate_note_wake")
        self._p(bc, "_standby_wake_detected", return_value=None)
        self._p(bc, "_audio_music_feed")
        self._p(bc, "_device_speech_ignored", return_value=False)
        self._p(bc, "_dialogue_hold_ignored", return_value=False)
        self._p(bc, "_self_echo_ignored", return_value=False)

    def _spoken(self, transcript):
        bc = self.bc
        self._p(bc, "record_speech",
                return_value=bc.np.zeros(bc.SAMPLE_RATE, dtype="float32"))
        self._p(bc, "_transcribe_capture",
                return_value=(transcript, {"no_speech_prob": 0.01}))
        return self._quiet(bc._handle_sleep_standby, None)[0]

    def test_a_filler_led_wake_carries_its_command(self):
        out = self._spoken("Um, Jarvis, turn off the lights")
        self.assertIsNotNone(out, "the command was dropped behind a greeting")
        self.assertEqual(out[0], "Um, Jarvis, turn off the lights")
        self.assertEqual(self.spoken, [])
        # ...and the carried turn passes the wake-word gate.
        self.assertFalse(self.bc._should_refuse_background_audio(out[0])[0])

    def test_a_filler_led_greeting_only_greets(self):
        self.assertIsNone(self._spoken("Oh Jarvis, wake up"))
        self.assertEqual(self.spoken, ["Yes, sir?"])

    def test_carries_command_rule(self):
        bc = self.bc
        self.assertTrue(bc._standby_wake_carries_command(LIVE_LINE))
        self.assertTrue(bc._standby_wake_carries_command(
            "So Jarvis, pause", typed=True))
        self.assertFalse(bc._standby_wake_carries_command("So Jarvis, pause"))
        self.assertFalse(bc._standby_wake_carries_command(
            "I asked Jarvis to turn off the lights"))
        self.assertFalse(bc._standby_wake_carries_command(
            "Um Jarvis, are you there", typed=True))


if __name__ == "__main__":
    unittest.main()
