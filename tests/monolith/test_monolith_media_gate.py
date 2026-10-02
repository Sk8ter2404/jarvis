"""Monolith wiring for the media gate (core/media_gate.py).

THE LIVE FAILURE (2026-10-01 22:28:50): an Instagram reel playing on this PC said
"Jarvis, find me a restaurant in Brickell, Miami, that doesn't already have a
website, and go and build a website for them now", the desk mic heard it, the
leading "Jarvis" passed the wake-word gate (_bg_gate_for_turn), and JARVIS ran a
web search + see_screen chain on it. The learn gate scored that capture's voice at
0.43 against the owner's voiceprint; his own wake-word turns scored 0.63-0.68.

Pinned here:
  * with the PC playing audio, a mic turn in a voice confidently not the owner's
    (under MEDIA_VOICE_GATE_REJECT_BELOW) is dropped with the one-line log;
  * the owner (matched, or too close to call) still runs over his own music;
  * no PC audio -> no voice check at all (no cost); voice-ID unavailable ->
    allowed and logged; typed / injected turns, a stop word, guest mode and
    the kill switch all pass; any error fails open;
  * the meter is read on a short-lived thread started at the end of each mic
    capture (parallel with Whisper), never stacks threads behind a hung read;
  * main() runs the gate after the speech filters and before the turn is
    printed, learned or dispatched; both capture paths start the probe.

Fakes only: no audio device, voiceprint or COM is touched.

    python -m unittest tests.monolith.test_monolith_media_gate
"""
from __future__ import annotations

import inspect
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import (MonolithGlobalsTestCase, load_monolith,
                                     requires_monolith)

try:
    import numpy as np
except Exception:  # pragma: no cover - light CI
    np = None

REEL = ("Jarvis, find me a restaurant in Brickell, Miami, that doesn't already "
        "have a website, and go and build a website for them now, go.")


class _FakeVoiceId:
    """core.voice_id stand-in: one enrolled owner, a fixed (name, score).
    ``short``: the (name, score) for a clip of at most 4 s (the leading
    speech window), when it should differ from the whole capture's.
    ``may_write``: what can(name, "memory_write") answers."""

    def __init__(self, name, score, enrolled=True, available=True,
                 short=None, may_write=True):
        self.name, self.score = name, score
        self.enrolled, self.available = enrolled, available
        self.short, self.may_write = short, may_write
        self.calls = 0
        self.lengths = []

    def list_enrolled(self):
        return ["owner"] if self.enrolled else []

    def is_available(self):
        return self.available

    def identify_speaker(self, audio, sr):
        self.calls += 1
        n = len(audio) if audio is not None else 0
        self.lengths.append(n)
        if self.short is not None and sr and n <= 4 * sr:
            return self.short
        return self.name, self.score

    def can(self, name, cap):
        return self.may_write


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        bc = self.bc
        self.audio = np.zeros(16000 * 3, dtype=np.float32)
        self._saved = (bc._last_capture_audio, bc._last_capture_sr)
        bc._last_capture_audio = self.audio
        bc._last_capture_sr = 16000
        self.addCleanup(self._restore)
        # MEDIA_VOICE_GATE_REJECT_BELOW is the SHIPPED default (2026-10-02:
        # a test pinned at 0.55 hid that the owner's own turns score 0.48-0.52).
        for p in (mock.patch.object(bc, "MEDIA_VOICE_GATE_ENABLED", True, create=True),
                  mock.patch.object(bc, "MEDIA_VOICE_GATE_PEAK", 0.01, create=True),
                  mock.patch.object(bc, "GUEST_MODE_ENABLED", False),
                  mock.patch.object(bc, "_smtc_media_playing", return_value=False)):
            p.start()
            self.addCleanup(p.stop)

    def _restore(self):
        self.bc._last_capture_audio, self.bc._last_capture_sr = self._saved

    def _gate(self, text, *, peak=0.4, vid=None, injected=False):
        bc = self.bc
        vid = vid if vid is not None else _FakeVoiceId(None, 0.43)
        lines = []
        # Patch the REAL module's functions: `import core.voice_id` resolves the
        # package attribute once another test has imported it, so a sys.modules
        # stand-in would be bypassed in a full run.
        import core.voice_id as real_vid
        with mock.patch.object(bc, "_media_probe_result", return_value=peak,
                               create=True), \
                mock.patch.object(real_vid, "list_enrolled", side_effect=vid.list_enrolled), \
                mock.patch.object(real_vid, "is_available", side_effect=vid.is_available), \
                mock.patch.object(real_vid, "identify_speaker",
                                  side_effect=lambda a, sr: vid.identify_speaker(a, sr)), \
                mock.patch.object(real_vid, "can", side_effect=vid.can), \
                mock.patch("builtins.print",
                           side_effect=lambda *a, **k: lines.append(" ".join(map(str, a)))):
            drop = bc._media_voice_gate(text, injected)
        return drop, lines, vid


class ReelReplayTests(_Base):
    def test_the_bg_gate_alone_let_the_reel_through(self):
        # Why the new gate exists: the wake prefix is the one-command path.
        with mock.patch.object(self.bc, "_require_wake_runtime", True):
            self.assertEqual(self.bc._bg_gate_for_turn(REEL, False), (False, ""))

    def test_the_live_reel_is_dropped_with_the_one_line_log(self):
        drop, lines, vid = self._gate(REEL)
        self.assertTrue(drop)
        self.assertEqual(vid.calls, 1)
        hit = [ln for ln in lines if "[media-gate]" in ln]
        self.assertEqual(len(hit), 1, lines)
        self.assertIn("[media-gate] PC audio playing and not the owner's voice", hit[0])
        self.assertNotIn("restaurant", hit[0])          # numbers, never the words

    def test_the_owner_still_runs_over_his_music(self):
        for name, score in ((None, 0.63), (None, 0.68), ("owner", 0.86)):
            with self.subTest(score=score):
                drop, lines, _vid = self._gate("Jarvis, turn it down a bit",
                                               vid=_FakeVoiceId(name, score))
                self.assertFalse(drop)
                self.assertTrue(any("allowed" in ln for ln in lines), lines)

    def test_meter_unreadable_falls_back_to_the_media_session(self):
        with mock.patch.object(self.bc, "_smtc_media_playing", return_value=True):
            drop, _lines, _vid = self._gate(REEL, peak=None)
        self.assertTrue(drop)


class OwnerOverMediaTests(_Base):
    """Review repair (2026-10-02). The 0.55 floor rested on "his wake-word
    turns score 0.63-0.68", but his own commands over media scored lower on
    the same buffer the gate checks: 21:43:17 "Jarvis plays Skrillex
    Essentials on YouTube." 0.52 (music detected just before), 22:59:47
    "Jarvis, what time is it? ..." 0.50 (a 20.7 s capture, mostly video),
    13:57:08 0.48. With PC audio playing those were dropped in silence; the
    live reel scored 0.43."""

    def test_his_live_turns_over_media_run(self):
        for text, score in (("Jarvis plays Skrillex Essentials on YouTube.", 0.52),
                            ("Jarvis, what time is it?", 0.50),
                            ("Jarvis, you did not program me to say that?", 0.48)):
            with self.subTest(score=score):
                drop, lines, _v = self._gate(text, vid=_FakeVoiceId(None, score))
                self.assertFalse(drop, lines)

    def test_media_controls_pass_whatever_the_voice(self):
        for text in ("Jarvis, pause.", "Jarvis, next song",
                     "Jarvis, turn it down", "Jarvis, volume down"):
            with self.subTest(text=text):
                drop, lines, vid = self._gate(text, vid=_FakeVoiceId(None, 0.30))
                self.assertFalse(drop)
                self.assertEqual(vid.calls, 0)
                self.assertTrue(any("media control" in ln for ln in lines), lines)

    def test_the_leading_speech_rescues_a_long_capture(self):
        # 22:59:47: his "Jarvis, what time is it?" at the start, the video
        # under the rest of a 20.7 s capture.
        bc = self.bc
        audio = np.zeros(16000 * 20, dtype=np.float32)
        audio[16000:16000 * 3] = 0.3
        audio[16000 * 4:] = 0.05
        bc._last_capture_audio = audio
        vid = _FakeVoiceId(None, 0.38, short=(None, 0.61))
        drop, lines, _v = self._gate("Jarvis, what time is it? This team is "
                                     "going to be building the data center",
                                     vid=vid)
        self.assertFalse(drop, lines)
        self.assertEqual(vid.calls, 2)
        self.assertLessEqual(min(vid.lengths), 16000 * 4)
        self.assertTrue(any("leading speech" in ln for ln in lines), lines)

    def test_a_long_reel_capture_stays_dropped(self):
        bc = self.bc
        audio = np.zeros(16000 * 20, dtype=np.float32)
        audio[16000:] = 0.2
        bc._last_capture_audio = audio
        drop, _lines, vid = self._gate(
            REEL, vid=_FakeVoiceId(None, 0.40, short=(None, 0.41)))
        self.assertTrue(drop)
        self.assertEqual(vid.calls, 2)

    def test_an_enrolled_guest_is_a_person_not_a_reel(self):
        # A guest without memory_write is NOT_OWNER to the learn gate, which
        # dropped every command of theirs while audio played.
        drop, lines, _v = self._gate("Jarvis, what is the weather tomorrow",
                                     vid=_FakeVoiceId("guest", 0.81,
                                                      may_write=False))
        self.assertFalse(drop, lines)


class DropCueTests(_Base):
    """Review repair (2026-10-02): a dropped turn that was addressed to
    JARVIS ("Jarvis, ...") gets a short spoken cue instead of silence, so
    the owner knows to say it again (or pause the video) - at most once a
    minute, never for a line that was not led by the wake word."""

    def setUp(self):
        super().setUp()
        self.said = []
        p = mock.patch.object(self.bc, "_speak",
                              side_effect=lambda t, *a, **k: self.said.append(t))
        p.start()
        self.addCleanup(p.stop)
        cell = getattr(self.bc, "_media_drop_cue_at", None)
        if isinstance(cell, list) and cell:
            cell[0] = 0.0

    def test_a_wake_led_drop_is_answered_once(self):
        bc = self.bc
        with mock.patch("builtins.print"):
            bc._media_gate_drop_cue(REEL)
            bc._media_gate_drop_cue(REEL)
        self.assertEqual(len(self.said), 1, self.said)
        self.assertIn("sir", self.said[0])
        self.assertNotIn("restaurant", self.said[0])

    def test_a_line_without_the_wake_word_gets_no_cue(self):
        with mock.patch("builtins.print"):
            self.bc._media_gate_drop_cue("find me a restaurant in Brickell")
        self.assertEqual(self.said, [])

    def test_the_cue_never_raises(self):
        with mock.patch.object(self.bc, "_speak",
                               side_effect=RuntimeError("tts")), \
                mock.patch("builtins.print"):
            self.bc._media_gate_drop_cue(REEL)


class PassThroughTests(_Base):
    def test_no_pc_audio_means_no_voice_check(self):
        drop, lines, vid = self._gate(REEL, peak=0.0)
        self.assertFalse(drop)
        self.assertEqual(vid.calls, 0)
        self.assertEqual([ln for ln in lines if "[media-gate]" in ln], [])

    def test_voice_id_unavailable_is_allowed_but_logged(self):
        for vid in (_FakeVoiceId(None, 0.0, enrolled=False),
                    _FakeVoiceId(None, 0.0, available=False)):
            with self.subTest(enrolled=vid.enrolled, available=vid.available):
                drop, lines, _v = self._gate(REEL, vid=vid)
                self.assertFalse(drop)
                self.assertTrue(any("voice-ID unavailable" in ln for ln in lines), lines)

    def test_typed_turns_are_never_checked(self):
        drop, _lines, vid = self._gate(REEL, injected=True)
        self.assertFalse(drop)
        self.assertEqual(vid.calls, 0)

    def test_a_stop_word_always_passes(self):
        drop, _lines, vid = self._gate("Jarvis, stop")
        self.assertFalse(drop)
        self.assertEqual(vid.calls, 0)

    def test_guest_mode_passes(self):
        with mock.patch.object(self.bc, "GUEST_MODE_ENABLED", True):
            drop, _lines, vid = self._gate(REEL)
        self.assertFalse(drop)
        self.assertEqual(vid.calls, 0)

    def test_kill_switch(self):
        with mock.patch.object(self.bc, "MEDIA_VOICE_GATE_ENABLED", False):
            drop, _lines, vid = self._gate(REEL)
        self.assertFalse(drop)
        self.assertEqual(vid.calls, 0)

    def test_no_capture_audio_passes(self):
        self.bc._last_capture_audio = None
        drop, _lines, vid = self._gate(REEL)
        self.assertFalse(drop)

    def test_an_error_fails_open(self):
        bc = self.bc
        with mock.patch.object(bc, "_media_probe_result", side_effect=RuntimeError("x"),
                               create=True):
            self.assertFalse(bc._media_voice_gate(REEL, False))
        vid = _FakeVoiceId(None, 0.43)
        vid.identify_speaker = mock.Mock(side_effect=RuntimeError("encoder"))
        drop, _lines, _v = self._gate(REEL, vid=vid)
        self.assertFalse(drop)


class ProbeTests(_Base):
    def test_probe_runs_off_thread_and_is_matched_to_its_capture(self):
        bc = self.bc
        seen = []

        def _read():
            seen.append(threading.current_thread().name)
            return 0.37
        with mock.patch.object(bc, "_media_probe_read", side_effect=_read, create=True):
            bc._media_probe_start(self.audio)
            self.assertAlmostEqual(bc._media_probe_result(self.audio), 0.37)
        self.assertEqual(len(seen), 1)
        self.assertNotEqual(seen[0], threading.current_thread().name)

    def test_a_hung_read_is_bounded_and_never_stacks_threads(self):
        bc = self.bc
        release = threading.Event()
        starts = []

        def _hang():
            starts.append(1)
            release.wait(5.0)
            return 0.9
        self.addCleanup(release.set)
        with mock.patch.object(bc, "_media_probe_read", side_effect=_hang, create=True):
            bc._media_probe_start(self.audio)
            t0 = time.monotonic()
            self.assertIsNone(bc._media_probe_result(self.audio, wait_s=0.05))
            self.assertLess(time.monotonic() - t0, 1.0)
            other = np.zeros(16000, dtype=np.float32)
            bc._media_probe_start(other)                 # previous read still hung
            self.assertIsNone(bc._media_probe_result(other, wait_s=0.05))
        self.assertEqual(len(starts), 1)

    def test_a_capture_without_a_probe_gets_one(self):
        bc = self.bc
        with mock.patch.object(bc, "_media_probe_read", return_value=0.2, create=True):
            other = np.ones(16000, dtype=np.float32)
            self.assertAlmostEqual(bc._media_probe_result(other), 0.2)

    def test_disabled_gate_never_reads_the_meter(self):
        bc = self.bc
        with mock.patch.object(bc, "MEDIA_VOICE_GATE_ENABLED", False), \
                mock.patch.object(bc, "_media_probe_read", create=True) as rd:
            bc._media_probe_start(self.audio)
            time.sleep(0.05)
        rd.assert_not_called()


class WiringTests(_Base):
    """Source-level: main() cannot run in a test."""

    def test_main_loop_runs_the_gate_before_the_turn_is_used(self):
        src = inspect.getsource(self.bc.main)
        call = "if _media_voice_gate(text, _injected_text is not None):"
        self.assertEqual(src.count("_media_voice_gate("), 1)
        at = src.index(call)
        self.assertLess(src.index("is_valid_speech(text, conf"), at)
        self.assertLess(src.index("_bg_gate_for_turn("), at)
        for later in ('print(f"  You:    {text}")',
                      "pattern_memory.record_voice_command(text)",
                      "reply = _run_llm_dispatch(text",
                      "learn_from_turn(text, reply, memory"):
            self.assertLess(at, src.index(later), later)
        tail = src[at:at + 300]
        self.assertIn("continue", tail)
        self.assertIn("_media_gate_drop_cue(text)", tail)
        self.assertLess(tail.index("_media_gate_drop_cue(text)"),
                        tail.index("continue"))

    def test_both_capture_paths_start_the_probe(self):
        for fn in (self.bc._capture_utterance, self.bc._handle_sleep_standby):
            with self.subTest(fn=fn.__name__):
                src = inspect.getsource(fn)
                self.assertIn("_media_probe_start(audio)", src)
                self.assertLess(src.index("_last_capture_audio = audio"),
                                src.index("_media_probe_start(audio)"))

    def test_config_defaults(self):
        import core.config as cfg
        self.assertIs(cfg.MEDIA_VOICE_GATE_ENABLED, True)
        self.assertIsInstance(cfg.MEDIA_VOICE_GATE_PEAK, float)
        self.assertIsInstance(cfg.MEDIA_VOICE_GATE_REJECT_BELOW, float)
        # His own live turns over media scored 0.48-0.52 (13:57:08, 22:59:47,
        # 21:43:17); the live reel 0.43.
        self.assertLess(cfg.MEDIA_VOICE_GATE_REJECT_BELOW, 0.48)
        self.assertGreater(cfg.MEDIA_VOICE_GATE_REJECT_BELOW, 0.43)

    def test_the_settings_page_has_the_switch_and_the_floor(self):
        import importlib
        sw = importlib.import_module("tools.settings_window")
        import core.config as cfg
        row = sw.SCHEMA["MEDIA_VOICE_GATE_ENABLED"]
        self.assertEqual(row["type"], "bool")
        self.assertIs(row["default"], cfg.MEDIA_VOICE_GATE_ENABLED)
        floor = sw.SCHEMA["MEDIA_VOICE_GATE_REJECT_BELOW"]
        self.assertEqual(floor["type"], "float")
        self.assertEqual(floor["default"], cfg.MEDIA_VOICE_GATE_REJECT_BELOW)


if __name__ == "__main__":
    unittest.main()
