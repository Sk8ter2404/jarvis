"""Monolith wiring for the self-echo gate (core/self_echo.py, R9 2026-09-29).

Live 2026-09-29: the tray's force_wake said "At your service, sir." from the
tray-drain thread while the main loop sat inside record_speech. The desk mic
heard it, Whisper transcribed it, and the turn ran as the OWNER — JARVIS
answered himself, four times in two minutes. Nothing in record_speech knew a
line was playing on ANOTHER thread.

These tests reproduce that sequence with the REAL record_speech (its
sounddevice InputStream faked by a feeder thread that pushes frames into the
real callback), the REAL tray dispatch -> _speak -> play_with_lipsync wrapper
(only the device body is faked, and it advances a frozen clock instead of
playing), and the real gate the main loop calls. The self-echo clock is frozen
and moved by hand at every step, so the timing is exact, not wall-clock luck.

Synthetic lines only.

    python -m unittest tests.monolith.test_monolith_self_echo
"""
from __future__ import annotations

import contextlib
import inspect
import io
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

try:
    from core import self_echo as _se_mod
except ImportError:          # a pre-R9 tree: the live test must FAIL there,
    _se_mod = None           # not error out in setUp (see LiveSequenceTests)

_TRAY_LINE = "At your service, sir."
_WAIT = 5.0     # bound on every cross-thread wait (never reached when green)


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.se = _se_mod
        self.clock = [100.0]
        if self.se is not None:
            self.se._reset_for_tests()
            self.addCleanup(self.se._reset_for_tests)
            self._p(self.se, "_clock", lambda: self.clock[0])
        self._p(bc, "SELF_ECHO_FILTER_ENABLED", True, create=True)
        self._p(bc, "SELF_ECHO_WINDOW_S", 20.0, create=True)
        self._p(bc, "SELF_ECHO_TAIL_S", 0.8, create=True)
        # _speak reaches the device (not the staging recorder / mute paths).
        self._p(bc, "_is_staging", lambda: False)
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_session_start_time", time.time() - 3600)
        self._p(bc, "_sentence_tts_plan", lambda text: None)
        self._p(bc, "synthesise",
                side_effect=lambda t: (bc.np.zeros(2400, dtype="float32"),
                                       24000))
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_heartbeat")
        bc._tts_muted[0] = False
        self.tmp = tempfile.mkdtemp(prefix="self_echo_mono_")
        self.addCleanup(lambda: os.path.isdir(self.tmp) and os.rmdir(self.tmp))
        self._p(bc, "OVERNIGHT_FLAG_FILE",
                os.path.join(self.tmp, "no_such_overnight_flag.json"))
        # _capture_utterance drains the pending-speech queue first: without
        # this it CLAIMED the LIVE project-root pending_speech.json (renamed to
        # .consuming and spoke it) - a running JARVIS's queued announcements
        # (found 2026-09-30 by a write audit). An absent temp queue: nothing
        # is drained and nothing is created.
        self._p(bc, "PENDING_SPEECH_PATH",
                os.path.join(self.tmp, "no_such_pending_speech.json"))
        # The device body of play_with_lipsync: a fake 1.6 s playback that
        # holds until the capture has heard it (when a test asks for that).
        self.playing = threading.Event()
        self.heard = threading.Event()
        self.heard.set()                       # default: don't hold
        self.played = []

        def _body(audio, sr):
            self.played.append(self.clock[0])
            self.playing.set()
            self.heard.wait(_WAIT)
            self.clock[0] += 1.6
        # (A pre-R9 tree has no wrapper: fake play_with_lipsync itself, so
        # the live sequence still runs there and shows the echo getting in.)
        self._p(bc, ("_play_with_lipsync_body"
                     if hasattr(bc, "_play_with_lipsync_body")
                     else "play_with_lipsync"), side_effect=_body)

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

    # ── the tray thread ──────────────────────────────────────────────────
    def _tray_thread(self):
        th = threading.Thread(target=self._quiet,
                              args=(self.bc._dispatch_tray_command,
                                    "force_wake", {}),
                              daemon=True, name="tray-drain-test")
        return th

    # ── the REAL record_speech on a fake stream ─────────────────────────
    def _capture(self, feeder, transcript):
        """Run _capture_utterance (-> the real record_speech) with a fake
        InputStream whose start() launches ``feeder(push, wait_vad)`` on its
        own thread. Returns (text, conf) the way the main loop gets it."""
        bc = self.bc
        np = bc.np
        # record_speech's _prof("voiced") runs right AFTER it reads the
        # self-echo clock for the VAD trip, so waiting on it (not on
        # _utterance_in_progress, set just BEFORE that read) means the test
        # can never move the clock under the read — no sleep, no race.
        voiced = threading.Event()
        self._p(bc, "_prof", side_effect=lambda *a, **k: (
            voiced.set() if a and a[0] == "voiced" else None))

        class FakeStream:
            device = 1

            def __init__(self, *a, callback=None, **k):
                self.cb = callback

            def start(self):
                cb = self.cb

                def push(n, amp):
                    frame = np.full((1024, 1), amp, dtype="float32")
                    for _ in range(n):
                        cb(frame, 1024, None, None)

                def wait_vad():
                    # A miss leaves the capture waiting for silence; the
                    # outer assertIsNotNone / window checks then fail.
                    voiced.wait(_WAIT)

                threading.Thread(target=feeder, args=(push, wait_vad),
                                 daemon=True, name="fake-mic").start()

        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "get_input_device", return_value=1)
        self._p(bc, "_safe_close_stream", lambda s: None)
        self._p(bc.sd, "InputStream", FakeStream)
        self._p(bc, "_note_live_capture", lambda *a, **k: None)
        self._p(bc, "_filler_capture_mark", lambda *a, **k: None)
        self._p(bc, "_process_capture_chunk",
                lambda data, sr, skip_ns=False: data)
        self._p(bc, "_spec_stt_should_snapshot", return_value=False)
        self._p(bc, "pause_face_tracking")
        self._p(bc, "resume_face_tracking")
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "_audio_music_feed")
        self._p(bc, "_transcribe_capture",
                return_value=(transcript, {"no_speech_prob": 0.0,
                                           "avg_logprob": -0.1}))
        self._p(bc, "_get_realtime_session", return_value=None)
        self._p(bc, "VAD_THRESHOLD", 0.008)
        bc._utterance_in_progress[0] = False
        cap, _ = self._quiet(bc._capture_utterance, None, {})
        self.assertIsNotNone(cap, "no utterance was captured")
        return cap

    def _gate(self, text, injected=False):
        """What the main loop's self-echo gate says. A tree without the gate
        passes every transcript on to the owner path (False)."""
        gate = getattr(self.bc, "_self_echo_ignored", None)
        if gate is None:
            return (False, "")
        return self._quiet(gate, text, injected)


class LiveSequenceTests(_Base):
    """The 2026-09-29 incident, end to end."""

    def _tray_speaks_mid_capture(self, transcript):
        """The main loop is listening (stream open at 100.0); at 101.0 the
        tray thread says its line (1.6 s); the mic's VAD trips on it at
        101.2; the capture ends at 104.0."""
        self.heard.clear()
        tray = self._tray_thread()

        def feeder(push, wait_vad):
            self.clock[0] = 101.0
            tray.start()
            self.assertTrue(self.playing.wait(_WAIT))
            self.clock[0] = 101.2
            push(6, 0.2)                     # JARVIS's voice in the room
            wait_vad()
            self.heard.set()                 # the line plays out (+1.6 s)
            tray.join(_WAIT)
            self.clock[0] = 104.0
            push(40, 0.0)                    # silence -> VAD break
        text, _ = self._capture(feeder, transcript)
        self.assertFalse(tray.is_alive())
        self.assertEqual(self.played, [101.0])
        return text

    def test_tray_line_heard_mid_capture_is_dropped(self):
        text = self._tray_speaks_mid_capture(_TRAY_LINE)
        dropped, log = self._gate(text)
        self.assertTrue(dropped, "JARVIS's own tray line reached the owner path")
        self.assertIn("[self-echo] ignored", log)
        # Numbers only — never the transcript.
        self.assertNotIn("service", log.lower())

    def test_record_speech_publishes_the_utterance_timing(self):
        self._tray_speaks_mid_capture(_TRAY_LINE)
        # (open, VAD trip, end, clip start = trip - the tripping chunk; the
        # pre-roll ring was empty: the first frame was already voiced).
        win = self.bc._last_capture_window[0]
        self.assertEqual(win[:3], (100.0, 101.2, 104.0))
        self.assertAlmostEqual(win[3], 101.2 - 1024 / self.bc.SAMPLE_RATE,
                               places=6)

    def test_timing_alone_drops_a_misheard_echo(self):
        # Whisper garbles the echo beyond the content match: the timing layer
        # still knows the mic heard it while JARVIS was speaking.
        text = self._tray_speaks_mid_capture("Add a sour vice, sure")
        self.assertIsNone(self.se.match(text))
        dropped, log = self._gate(text)
        self.assertTrue(dropped)
        self.assertIn("heard during playback", log)

    def test_stop_heard_during_the_tray_line_still_passes(self):
        text = self._tray_speaks_mid_capture("stop")
        dropped, log = self._gate(text)
        self.assertFalse(dropped)
        self.assertNotIn("[self-echo]", log)

    def test_wake_word_barge_during_a_line_without_jarvis_passes(self):
        text = self._tray_speaks_mid_capture("Jarvis, what time is it")
        self.assertFalse(self._gate(text)[0])

    def test_wake_word_during_a_line_that_says_jarvis_is_his_echo(self):
        # request_tts_interrupt's echo rule: while his own line says
        # "jarvis", a wake hit is his own voice.
        self.se.playback_end(
            self.se.playback_begin("Say JARVIS when ready.", at=101.0),
            at=102.0)
        self.bc._last_capture_window[0] = (100.0, 101.2, 103.0)
        self.clock[0] = 103.0
        self.assertTrue(self._gate("Jarvis when ready")[0])


class OwnerStillHeardTests(_Base):
    def test_owner_speaking_right_after_the_tray_line_gets_through(self):
        tray = self._tray_thread()

        def feeder(push, wait_vad):
            self.clock[0] = 101.0
            tray.start()
            tray.join(_WAIT)                 # line ends at 102.6
            self.clock[0] = 103.5            # the owner, after the tail
            push(6, 0.2)
            wait_vad()
            self.clock[0] = 105.0
            push(40, 0.0)
        text, _ = self._capture(feeder, "what's on my calendar today")
        self.assertEqual(self.played, [101.0])
        dropped, log = self._gate(text)
        self.assertFalse(dropped)
        self.assertNotIn("[self-echo]", log)

    def test_quick_answer_to_his_own_question_gets_through(self):
        # Main thread: speak, THEN listen. A capture opened after the line
        # ended is never tail-gated, so a fast "yes" is still the owner.
        self._quiet(self.bc._speak, "Shall I start the print, sir?")
        self.assertEqual(self.clock[0], 101.6)
        self.bc._last_capture_window[0] = (101.65, 101.7, 102.4)
        self.clock[0] = 102.4
        self.assertFalse(self._gate("yes")[0])
        self.assertFalse(self._gate("yes start it")[0])

    def _speak_then_owner(self, line, owner_says):
        """The ordinary main-thread turn: JARVIS speaks ``line`` (100.0 to
        101.6), the loop opens the next capture at 101.7 and the owner starts
        talking at 102.2 — the REAL record_speech end to end."""
        self._quiet(self.bc._speak, line)
        self.assertEqual(self.clock[0], 101.6)
        self.clock[0] = 101.7

        def feeder(push, wait_vad):
            self.clock[0] = 102.2
            push(6, 0.2)
            wait_vad()
            self.clock[0] = 103.5
            push(40, 0.0)
        text, _ = self._capture(feeder, owner_says)
        return self._gate(text)

    def test_owner_reissuing_a_command_he_heard_acknowledged_gets_through(self):
        # Reviewer fix: "Turning off the desk lamp, sir." vs the owner's
        # "turn off the desk lamp" is a 0.86 char ratio — the content layer
        # used to drop the owner's retry as an echo.
        dropped, log = self._speak_then_owner(
            "Turning off the desk lamp, sir.", "turn off the desk lamp")
        self.assertFalse(dropped)
        self.assertNotIn("[self-echo]", log)

    def test_owner_answering_with_the_questions_own_words_gets_through(self):
        dropped, log = self._speak_then_owner(
            "Should I lock the front door?", "lock the front door")
        self.assertFalse(dropped)
        self.assertNotIn("[self-echo]", log)

    def test_muted_playback_never_gates_the_owner(self):
        muted = mock.Mock()
        muted.is_muted.return_value = True
        muted.parse_wry_tag.side_effect = lambda t: (False, t)
        self._p(self.bc, "_tts_layer", muted)
        self.bc.play_with_lipsync(self.bc.np.zeros(10, dtype="float32"), 24000)
        self.assertFalse(self.se.playback_live())
        self.assertEqual(len(self.se._playbacks), 0)


class ContentLayerTests(_Base):
    def test_every_spoken_line_is_remembered_from_any_thread(self):
        th = threading.Thread(target=self._quiet,
                              args=(self.bc._speak, "The kettle has boiled, sir."),
                              daemon=True)
        th.start()
        th.join(_WAIT)
        # No capture timing at all (e.g. the realtime path): content only.
        self.bc._last_capture_window[0] = None
        self.clock[0] += 3.0
        dropped, log = self._gate("the kettle has boiled sir")
        self.assertTrue(dropped)
        self.assertIn("[self-echo] ignored (score 1.00", log)
        self.assertNotIn("kettle", log)

    def test_lines_expire(self):
        self._quiet(self.bc._speak, _TRAY_LINE)       # ends at 101.6
        self.bc._last_capture_window[0] = None
        self.clock[0] = 101.6 + 19.5
        self.assertTrue(self._gate(_TRAY_LINE)[0])
        self.clock[0] = 101.6 + 20.5
        self.assertFalse(self._gate(_TRAY_LINE)[0])

    def test_window_knob_is_read_live(self):
        self._quiet(self.bc._speak, _TRAY_LINE)
        self.bc._last_capture_window[0] = None
        self.clock[0] += 6.0
        self._p(self.bc, "SELF_ECHO_WINDOW_S", 5.0)
        self.assertFalse(self._gate(_TRAY_LINE)[0])

    def test_stop_always_passes_the_content_layer(self):
        self._quiet(self.bc._speak, "Stop the music, sir?")
        self.bc._last_capture_window[0] = None
        self.assertFalse(self._gate("stop the music sir")[0])

    def test_bare_wake_word_is_never_an_echo(self):
        self._quiet(self.bc._speak, "Jarvis.")
        self.bc._last_capture_window[0] = None
        self.assertFalse(self._gate("Jarvis")[0])


class BypassAndKnobTests(_Base):
    def _echo_everywhere(self):
        self._quiet(self.bc._speak, _TRAY_LINE)       # 100.0-101.6
        self.bc._last_capture_window[0] = (99.0, 100.2, 102.0)
        self.clock[0] = 102.0

    def test_typed_turns_bypass(self):
        self._echo_everywhere()
        self.assertTrue(self._gate(_TRAY_LINE)[0])     # the mic twin drops
        dropped, log = self._gate(_TRAY_LINE, injected=True)
        self.assertFalse(dropped)
        self.assertNotIn("[self-echo]", log)

    def test_injected_capture_never_carries_mic_timing(self):
        self._echo_everywhere()
        text, _ = self._quiet(self.bc._capture_utterance, _TRAY_LINE, {})[0]
        self.assertEqual(text, _TRAY_LINE)
        self.assertIsNone(self.bc._last_capture_window[0])

    def test_disabled_knob_turns_both_layers_off(self):
        self._echo_everywhere()
        self._p(self.bc, "SELF_ECHO_FILTER_ENABLED", False)
        self.assertFalse(self._gate(_TRAY_LINE)[0])

    def test_gate_fails_open(self):
        self._p(self.se, "capture_overlap",
                side_effect=RuntimeError("boom"))
        self.bc._last_capture_window[0] = (1.0, 2.0, 3.0)
        self.assertFalse(self._gate("anything at all here")[0])

    def test_config_defaults_and_settings_wiring(self):
        import ast
        import json
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        with open(os.path.join(root, "core", "config.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        lits = {}
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and isinstance(node.value,
                                                           ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        lits[tgt.id] = node.value.value
        self.assertIs(lits["SELF_ECHO_FILTER_ENABLED"], True)
        self.assertEqual(lits["SELF_ECHO_WINDOW_S"], 20.0)
        self.assertEqual(lits["SELF_ECHO_TAIL_S"], 0.8)
        head = src[:src.index("\nSELF_ECHO_FILTER_ENABLED = True")]
        block = head[head.rindex("\n\n"):].strip().splitlines()
        self.assertTrue(block and all(ln.startswith("#") for ln in block))
        self.assertIn("on the next start",
                      " ".join(ln.lstrip("# ") for ln in block))
        spec = importlib.util.spec_from_file_location(
            "sw_self_echo", os.path.join(root, "tools", "settings_window.py"))
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        with open(os.path.join(root, "tools", "user_settings.example.json"),
                  encoding="utf-8") as fh:
            example = json.load(fh)
        for key, typ in (("SELF_ECHO_FILTER_ENABLED", "bool"),
                         ("SELF_ECHO_WINDOW_S", "float"),
                         ("SELF_ECHO_TAIL_S", "float")):
            self.assertEqual(sw.SCHEMA[key]["type"], typ)
            self.assertEqual(sw.SCHEMA[key]["default"], lits[key])
            self.assertEqual(example[key], lits[key])
        # The monolith reads the same defaults through `from core.config import *`.
        import core.config as cfg
        self.assertIs(cfg.SELF_ECHO_FILTER_ENABLED, True)


class StandbyPathTests(_Base):
    def test_tray_line_heard_in_standby_is_not_learned(self):
        bc = self.bc
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        bc._ambient_learning[0] = True
        feed = self._p(bc, "_ambient_learning_feed")
        self._p(bc, "_standby_wake_detected", return_value=None)
        self._p(bc, "_device_speech_ignored", return_value=False)
        self._quiet(bc._speak, "Your timer is done, sir.")     # 100.0-101.6

        def _rec(timeout=None):
            bc._last_capture_window[0] = (99.5, 100.4, 102.5)
            return bc.np.zeros(int(bc.SAMPLE_RATE * 1.0), dtype="float32")
        self._p(bc, "record_speech", side_effect=_rec)
        self._p(bc, "_audio_music_feed")
        self._p(bc, "_transcribe_capture",
                return_value=("Your timer is done, sir.", {}))
        self.clock[0] = 102.5
        _, log = self._quiet(bc._handle_sleep_standby, None)
        feed.assert_not_called()
        self.assertIn("[self-echo] ignored", log)
        self.assertTrue(bc._sleep_mode[0])


class MainLoopWiringTests(_Base):
    """Source-level: main() cannot run in a test."""

    def test_gate_runs_right_after_the_device_gate_and_skips_the_turn(self):
        src = inspect.getsource(self.bc.main)
        gate_block = ("if _self_echo_ignored(text, _injected_text is not None):\n"
                      "                    set_state(\"idle\")\n"
                      "                    continue\n")
        self.assertEqual(src.count("_self_echo_ignored("), 1)
        gate = src.index(gate_block)
        self.assertLess(src.index("text, conf = _cap"), gate)
        for later in ("_handle_ambient_music(text)",
                      "_bg_gate_for_turn(",
                      "target=_ambient_learn_from_gated",
                      "is_valid_speech(text, conf",
                      "pattern_memory.record_voice_command(text)",
                      "reply = _run_llm_dispatch(text",
                      "learn_from_turn(text, reply, memory"):
            self.assertLess(gate, src.index(later), later)

    def test_standby_gate_precedes_the_wake_match_and_learning(self):
        src = inspect.getsource(self.bc._handle_sleep_standby)
        gate = src.index("if _self_echo_ignored(text, injected_text is not None):"
                         "\n        return\n")
        self.assertLess(gate, src.index("if _WAKE_RE.search(tl):"))
        self.assertLess(gate, src.index("_ambient_learning_feed(text)"))

    def test_every_voiced_clip_passes_the_registering_wrapper(self):
        # _speak, the sentence player and the filler all call the public
        # play_with_lipsync; only the wrapper may call the device body.
        src = inspect.getsource(self.bc)
        self.assertEqual(src.count("_play_with_lipsync_body(audio, sr)"), 1)
        wrapper = inspect.getsource(self.bc.play_with_lipsync)
        self.assertIn("_play_with_lipsync_body(audio, sr)", wrapper)
        self.assertIn("playback_end", wrapper)


if __name__ == "__main__":
    unittest.main()
