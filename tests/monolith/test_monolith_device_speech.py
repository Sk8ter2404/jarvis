"""Monolith wiring for the known-device speech filter
(core/device_speech_filter.py, R3 2026-09-29).

A line a known device in the room speaks must never command JARVIS: the main
loop drops such a turn right after transcription, BEFORE the background/wake
gate, the LLM and any learning; the standby wake path never wakes on it and
never feeds it to the ambient learner. Logs carry the device (source) name
only, never the utterance.

GENERIC fixtures only (a made-up "desk speaker"); the phrase directory is
redirected to a temp dir through JARVIS_DATA_DIR, as core/paths resolves it.

    python -m unittest tests.monolith.test_monolith_device_speech
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_DEVICE_LINE = "Jarvis, desk speaker ready to play."
_FIXTURE = {
    "source": "desk speaker",
    "phrases": [_DEVICE_LINE, "Battery level is getting low.", "Jarvis",
                # Owner vocabulary a device may also say (R3 review).
                "Yes.", "Next track", "Go to sleep.", "Overnight protocol."],
}


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        from core import device_speech_filter as dsf
        self.dsf = dsf
        dsf._reset_cache_for_tests()
        self.addCleanup(dsf._reset_cache_for_tests)
        self.tmp = tempfile.mkdtemp(prefix="dsf_mono_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        os.makedirs(os.path.join(self.tmp, "device_phrases"))
        with open(os.path.join(self.tmp, "device_phrases", "desk.json"), "w",
                  encoding="utf-8") as f:
            json.dump(_FIXTURE, f)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.tmp})
        env.start()
        self.addCleanup(env.stop)
        self._p(self.bc, "DEVICE_SPEECH_FILTER_ENABLED", True)

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


class DeviceSpeechIgnoredTests(_Base):
    def test_device_line_is_ignored_and_logs_the_source_only(self):
        out, log = self._quiet(self.bc._device_speech_ignored,
                               "Jarvis desk speaker ready to play")
        self.assertTrue(out)
        self.assertIn("[device-speech] ignored (desk speaker)", log)
        self.assertNotIn("ready to play", log.lower())

    def test_misheard_device_line_is_ignored(self):
        out, _ = self._quiet(self.bc._device_speech_ignored,
                             "battery level is getting slow")
        self.assertTrue(out)

    def test_owner_command_passes_silently(self):
        out, log = self._quiet(self.bc._device_speech_ignored,
                               "jarvis open my bookmarks")
        self.assertFalse(out)
        self.assertEqual(log, "")

    def test_stop_word_always_passes(self):
        out, _ = self._quiet(self.bc._device_speech_ignored,
                             "stop jarvis desk speaker ready to play")
        self.assertFalse(out)

    def test_bare_wake_phrase_is_never_ignored(self):
        # The fixture device also says a bare "Jarvis"; the owner's wake word
        # must still work.
        out, _ = self._quiet(self.bc._device_speech_ignored, "Jarvis.")
        self.assertFalse(out)

    def test_owner_yes_passes_even_when_a_device_list_has_yes(self):
        # The gate runs BEFORE _handle_shutdown_prompt and
        # handle_confirmation_response: the owner's bare "yes" to a pending
        # confirmation must still arrive. Sleep and shutdown-prompt phrases
        # are protected by the monolith's own constants.
        for text in ("Yes.", "yes", "Next track.", "go to sleep",
                     "Overnight protocol"):
            out, log = self._quiet(self.bc._device_speech_ignored, text)
            self.assertFalse(out, text)
            self.assertEqual(log, "")

    def test_injected_text_is_never_checked(self):
        # Typed / injected input (LAN page, tray, say_to_jarvis) is explicit
        # operator input, not overheard room audio — like _bg_gate_for_turn.
        out, log = self._quiet(self.bc._device_speech_ignored, _DEVICE_LINE,
                               True)
        self.assertFalse(out)
        self.assertEqual(log, "")
        out, _ = self._quiet(self.bc._device_speech_ignored, _DEVICE_LINE,
                             False)
        self.assertTrue(out)

    def test_disabled_flag_turns_the_gate_off(self):
        self._p(self.bc, "DEVICE_SPEECH_FILTER_ENABLED", False)
        out, _ = self._quiet(self.bc._device_speech_ignored, _DEVICE_LINE)
        self.assertFalse(out)

    def test_filter_error_fails_open(self):
        self._p(self.dsf, "match", side_effect=RuntimeError("boom"))
        out, _ = self._quiet(self.bc._device_speech_ignored, _DEVICE_LINE)
        self.assertFalse(out)

    def test_missing_phrase_dir_means_no_filtering(self):
        shutil.rmtree(os.path.join(self.tmp, "device_phrases"))
        out, _ = self._quiet(self.bc._device_speech_ignored, _DEVICE_LINE)
        self.assertFalse(out)

    def test_config_default_is_on(self):
        import core.config as cfg
        src = inspect.getsource(cfg)
        self.assertRegex(src, r"(?m)^DEVICE_SPEECH_FILTER_ENABLED = True\s*$")
        # The comment block DIRECTLY above the assignment says when a change
        # applies (the repo convention), not merely somewhere in the file.
        head = src[:src.index("\nDEVICE_SPEECH_FILTER_ENABLED = True")]
        block = head[head.rindex("\n\n"):].strip().splitlines()
        self.assertTrue(block and all(ln.startswith("#") for ln in block),
                        block)
        self.assertIn("on the next start",
                      " ".join(ln.lstrip("# ") for ln in block))


class StandbyWakePathTests(_Base):
    def setUp(self):
        super().setUp()
        self._speak = self._p(self.bc, "_speak")
        self._p(self.bc, "set_state")
        self._p(self.bc, "_heartbeat")
        self._p(self.bc, "_audio_music_should_refuse_wake", return_value=False)
        self._p(self.bc, "_standby_wake_detected", return_value=None)
        self._p(self.bc, "context_aware_greeting",
                return_value=("Back online, sir.", 1.0))
        self._p(self.bc, "OVERNIGHT_FLAG_FILE",
                os.path.join(self.tmp, "no_such_overnight_flag.json"))
        self.feed = self._p(self.bc, "_ambient_learning_feed")
        self.bc._sleep_mode[0] = True
        self.bc._standby_mode[0] = True
        self.bc._ambient_learning[0] = True

    def _mic(self, text):
        good = self.bc.np.zeros(int(self.bc.SAMPLE_RATE * 1.0), dtype="float32")
        self._p(self.bc, "record_speech", return_value=good)
        self._p(self.bc, "_audio_music_feed")
        self._p(self.bc, "transcribe", return_value=(text, {}))
        self._p(self.bc, "_transcribe_capture", return_value=(text, {}))

    def test_device_line_led_by_the_wake_word_never_wakes(self):
        self._mic(_DEVICE_LINE)
        _, log = self._quiet(self.bc._handle_sleep_standby, None)
        self.assertTrue(self.bc._sleep_mode[0])
        self.assertTrue(self.bc._standby_mode[0])
        self._speak.assert_not_called()
        self.feed.assert_not_called()          # not learned either
        self.assertIn("[device-speech] ignored (desk speaker)", log)
        # The gate never prints the utterance.
        self.assertNotIn("ready to play", log.lower())

    def test_misheard_device_line_from_the_mic_never_wakes(self):
        self._mic("Jarvis the speaker ready to play")
        self._quiet(self.bc._handle_sleep_standby, None)
        self.assertTrue(self.bc._sleep_mode[0])
        self._speak.assert_not_called()
        self.feed.assert_not_called()

    def test_injected_device_line_is_operator_input_and_wakes(self):
        # Injects are typed operator input: the device gate does not apply.
        _, log = self._quiet(self.bc._handle_sleep_standby, _DEVICE_LINE)
        self.assertFalse(self.bc._sleep_mode[0])
        self.assertNotIn("[device-speech]", log)

    def test_owner_wake_still_wakes(self):
        self._quiet(self.bc._handle_sleep_standby, "hey JARVIS wake up please")
        self.assertFalse(self.bc._sleep_mode[0])
        self._speak.assert_called_once_with("Back online, sir.",
                                            volume_scale=1.0)

    def test_bare_wake_word_still_wakes_even_if_a_device_says_it(self):
        self._quiet(self.bc._handle_sleep_standby, "Jarvis")
        self.assertFalse(self.bc._sleep_mode[0])


class MainLoopWiringTests(_Base):
    """Source-level: main() cannot run in a test (see
    test_monolith_turn_timing.WiringTests)."""

    def test_device_gate_runs_first_and_skips_the_whole_turn(self):
        src = inspect.getsource(self.bc.main)
        cap = src.index("text, conf = _cap")
        gate_block = ("if _device_speech_ignored(text, _injected_text is not None):\n"
                      "                    set_state(\"idle\")\n"
                      "                    continue\n")
        # Injects bypass the gate (mirrors _bg_gate_for_turn).
        self.assertEqual(src.count("_device_speech_ignored("), 1)
        gate = src.index(gate_block)
        self.assertLess(cap, gate)
        # Nothing but comments between the transcript and the gate.
        between = src[cap + len("text, conf = _cap"):gate]
        code = [ln for ln in between.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
        self.assertEqual(code, [])
        # ...and it precedes the bg/wake gate, the learners and the LLM.
        for later in ("_handle_ambient_music(text)",
                      "_bg_gate_for_turn(",
                      "target=_ambient_learn_from_gated",
                      "is_valid_speech(text, conf",
                      "pattern_memory.record_voice_command(text)",
                      "_run_voice_shortcuts(text)",
                      "reply = _run_llm_dispatch(text",
                      # prefix: the call also passes the turn's conf now
                      "learn_from_turn(text, reply, memory"):
            self.assertLess(gate, src.index(later), later)

    def test_standby_gate_precedes_the_wake_match_and_learning(self):
        src = inspect.getsource(self.bc._handle_sleep_standby)
        gate = src.index("if _device_speech_ignored(text, injected_text is not None):"
                         "\n        return\n")
        self.assertLess(src.index("_wake_hit = _standby_wake_detected(audio)"),
                        gate)
        self.assertLess(gate, src.index("if _WAKE_RE.search(tl):"))
        self.assertLess(gate, src.index("_ambient_learning_feed(text)"))


if __name__ == "__main__":
    unittest.main()
