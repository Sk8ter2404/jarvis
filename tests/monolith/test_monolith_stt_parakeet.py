"""Monolith wiring for speed plan R6 — Parakeet on the CPU for the owner's
captures (core/stt_parakeet.py; the logic itself is covered CI-light by
tests/test_stt_parakeet.py).

Pins, on the REAL _transcribe_capture:
  * both flags off (the default): exactly ``return transcribe(audio)`` —
    the same result object, nothing Parakeet touched, no stt_engine note,
    no boot warmer;
  * STT_ENGINE='parakeet': the owner's decode never waits on _stt_lock —
    it completes while a fake ambient Whisper decode holds that lock
    (Event-based; routing the decode through _stt_lock turns this red);
  * Parakeet never lands in _stt;
  * replacements, the rescue (the real _text_has_wake_prefix and wake-word
    mode), the exception latch and the turn line's stt_engine;
  * STT_SHADOW='parakeet': Whisper's result stands and the capture is
    offered to the shadow worker; the shadow judge runs the real gates.

No real model, no real Whisper, no real audio.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_stt_parakeet
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _Res:
    def __init__(self, text, logprobs):
        self.text = text
        self.logprobs = logprobs


class _FakeEngine:
    """Stands in for onnx-asr's timestamped adapter."""

    def __init__(self, text="Jarvis, what time is it?",
                 logprobs=(-0.01, -0.02, -0.03)):
        self.calls = 0
        self.res = _Res(text, list(logprobs))

    def recognize(self, a, sample_rate=None):
        self.calls += 1
        return self.res


W_RES = ("whisper words", {"no_speech_prob": 0.0, "avg_logprob": -0.2})


class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("JARVIS_STT_ENGINE", None)
        # Parakeet's session state is process-wide: start and end clean.
        latch, prim, shadow = (bc._parakeet_latch, bc._parakeet_primary,
                               bc._parakeet_shadow)
        saved = (latch.failed, prim.decodes, prim.rescues, prim._hot_logged,
                 prim._rescue_logged_at, prim._rescue_quiet)

        def _restore():
            (latch.failed, prim.decodes, prim.rescues, prim._hot_logged,
             prim._rescue_logged_at, prim._rescue_quiet) = saved
        latch.failed = ""
        prim.decodes = prim.rescues = prim._rescue_quiet = 0
        prim._rescue_logged_at = None
        self.addCleanup(_restore)
        self._p(bc, "STT_ENGINE", "whisper")
        self._p(bc, "STT_SHADOW", "")
        self._p(bc, "TURN_TAIL_PROBE", False)
        self._p(bc, "_SPECULATIVE_STT", False)
        self._p(bc, "_stt_alt", None)
        self._p(bc, "_require_wake_runtime", False)
        self._p(bc, "_standby_mode", [False])
        self._p(bc, "_sleep_mode", [False])
        self._p(bc, "STT_HOTWORDS", "")
        self._p(bc, "STT_REPLACEMENTS", {})
        self._p(bc, "STT_REPLACEMENTS_PARAKEET", {})
        self._p(bc, "_last_capture_preroll", [None])
        # The live wake gates' state, isolated from this PC (R6 review: the
        # rescue asks every one of them): no media session, no room-music
        # skill, no post-dialogue hold, no greeting admit, no follow-up.
        from core.followup_window import FollowupWindow
        self._p(bc, "_smtc_media_playing", return_value=False)
        self._set_module("skill_standby_audio_detect", None)
        self._p(bc, "_turn_hold_until", [0.0])
        self._p(bc, "_standby_greet_admit_until", [0.0])
        self._p(bc, "_followup_window", FollowupWindow(0))
        self.notes = []
        self._p(bc, "_tt_note_stat",
                side_effect=lambda n, v: self.notes.append((n, v)))
        self.audio = bc.np.zeros(bc.SAMPLE_RATE, dtype=bc.np.float32)
        self.shadow = shadow

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _set_module(self, name, mod):
        """sys.modules[name] = mod for this test (None = not loaded, as
        the monolith's sys.modules.get lookups see it); only that key is
        restored."""
        had, old = name in sys.modules, sys.modules.get(name)
        sys.modules[name] = mod

        def _restore():
            if had:
                sys.modules[name] = old
            else:
                sys.modules.pop(name, None)
        self.addCleanup(_restore)

    def _engine_notes(self):
        return [v for n, v in self.notes if n == "stt_engine"]


# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class R6FlagsOffTests(_Base):
    def test_both_flags_ship_off(self):
        import core.config as cfg
        self.assertEqual((cfg.STT_ENGINE, cfg.STT_SHADOW), ("whisper", ""))
        self.assertIsNone(self.bc._stt_r6_route())

    def test_flags_off_is_exactly_transcribe(self):
        bc = self.bc
        res = ("hi there", {"no_speech_prob": 0.0, "avg_logprob": -0.1})
        tr = self._p(bc, "transcribe", return_value=res)
        load = self._p(bc._stt_parakeet, "load",
                       side_effect=AssertionError("Parakeet loaded"))
        run = self._p(bc._parakeet_primary, "run")
        offer = self._p(bc._parakeet_shadow, "offer")
        self.assertIs(bc._transcribe_capture(self.audio), res)
        tr.assert_called_once()
        self.assertIs(tr.call_args.args[0], self.audio)
        for m in (load, run, offer):
            m.assert_not_called()
        self.assertEqual(self._engine_notes(), [])
        self.assertIsNone(bc._stt_alt)

    def test_flags_off_registers_no_boot_warmer(self):
        bc = self.bc
        names = [n for n, _ in bc._boot_warmers]
        if bc._stt_r6_route() is None:
            self.assertNotIn("parakeet", names)
        src = inspect.getsource(bc)
        self.assertIn('if _stt_r6_route() is not None:\n'
                      '    _register_boot_warmer("parakeet", _warm_parakeet)',
                      src)

    def test_route(self):
        bc = self.bc
        self._p(bc, "STT_SHADOW", "parakeet")
        self.assertEqual(bc._stt_r6_route(), "shadow")
        self._p(bc, "STT_ENGINE", "parakeet")
        self.assertEqual(bc._stt_r6_route(), "primary")
        self._p(bc, "STT_ENGINE", "whisper")
        self._p(bc, "STT_SHADOW", "")
        os.environ["JARVIS_STT_ENGINE"] = "parakeet"
        self.assertEqual(bc._stt_r6_route(), "primary")


# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class R6PrimaryTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "STT_ENGINE", "parakeet")

    def test_owner_decode_never_waits_on_stt_lock(self):
        """A fake ambient Whisper decode holds _stt_lock for the whole test;
        the owner's capture must still be transcribed. Event-based: if the
        decode were routed through _stt_lock, `done` would never be set."""
        bc = self.bc
        eng = _FakeEngine()
        self._p(bc, "_stt_alt", eng)
        # Were Whisper used after all, the REAL transcribe() would block on
        # _stt_lock too (its inner decode is the fake below).
        self._p(bc, "_transcribe_impl", return_value=W_RES)
        held, release, done = (threading.Event(), threading.Event(),
                               threading.Event())
        box = {}

        def ambient():
            with bc._stt_lock:
                held.set()
                release.wait(10.0)

        def owner():
            try:
                box["res"] = bc._transcribe_capture(self.audio)
            except BaseException as e:      # pragma: no cover - reported
                box["err"] = e
            finally:
                done.set()

        amb = threading.Thread(target=ambient, daemon=True)
        own = threading.Thread(target=owner, daemon=True)
        amb.start()
        self.assertTrue(held.wait(5.0))
        try:
            own.start()
            finished = done.wait(5.0)
        finally:
            release.set()
            amb.join(5.0)
            own.join(5.0)
        self.assertTrue(finished, "the owner's decode waited on _stt_lock")
        self.assertNotIn("err", box)
        self.assertEqual(box["res"][0], "Jarvis, what time is it?")
        self.assertEqual(box["res"][1]["engine"], "parakeet")
        self.assertEqual(eng.calls, 1)
        self.assertEqual(self._engine_notes(), ["parakeet"])

    def test_speculative_stt_stands_aside(self):
        """JARVIS_SPECULATIVE_STT=1 with Parakeet primary: no speculative
        Whisper snapshot is started, and one already standing is not
        joined — it would pre-empt Parakeet with Whisper's text (R6 review,
        finding 6)."""
        bc = self.bc
        self._p(bc, "_SPECULATIVE_STT", True)
        self.assertFalse(bc._spec_stt_should_snapshot(7, 7, 20, 6))
        self._p(bc, "STT_ENGINE", "whisper")
        self.assertTrue(bc._spec_stt_should_snapshot(7, 7, 20, 6))
        self._p(bc, "STT_ENGINE", "parakeet")
        done = threading.Thread(target=lambda: None)
        done.start()
        done.join()
        spec = dict(bc._spec_stt, thread=done, chunks=12,
                    result=("speculative words", {}))
        self._p(bc, "_spec_stt", spec)
        self._p(bc, "_stt_alt", _FakeEngine(text="turn the lights off"))
        self.assertEqual(bc._transcribe_capture(self.audio)[0],
                         "turn the lights off")
        self.assertEqual(self._engine_notes(), ["parakeet"])

    def test_the_turn_flags_line_shows_the_engine_in_effect(self):
        # JARVIS_STT_ENGINE overrides the config value: the boot line (which
        # turn_latency_report --split flag=STT_ENGINE reads) says so.
        bc = self.bc
        self._p(bc, "STT_ENGINE", "whisper")
        os.environ["JARVIS_STT_ENGINE"] = "parakeet"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._log_turn_flags()
        self.assertIn(" STT_ENGINE=parakeet", out.getvalue())
        os.environ.pop("JARVIS_STT_ENGINE")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._log_turn_flags()
        self.assertIn(" STT_ENGINE=whisper", out.getvalue())

    def test_parakeet_never_lands_in_stt(self):
        bc = self.bc
        sentinel = object()
        self._p(bc, "_stt", sentinel)
        eng = _FakeEngine()
        load = self._p(bc._stt_parakeet, "load", return_value=eng)
        bc._warm_parakeet()
        bc._transcribe_capture(self.audio)
        self.assertIs(bc._stt, sentinel)
        self.assertIs(bc._stt_alt, eng)
        load.assert_called_once_with(bc.PARAKEET_MODEL_DIR,
                                     bc.PARAKEET_THREADS)
        # No R6 function rebinds _stt or so much as names _stt_lock.
        for fn in (bc._parakeet_engine, bc._parakeet_decode,
                   bc._warm_parakeet, bc._transcribe_capture_r6,
                   bc._transcribe_capture):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Global):
                    self.assertNotIn("_stt", node.names, fn.__name__)
                if isinstance(node, ast.Name):
                    self.assertNotEqual(node.id, "_stt_lock", fn.__name__)

    def test_replacements_then_parakeets_own(self):
        bc = self.bc
        self._p(bc, "_stt_alt", _FakeEngine(text="open a cello please"))
        self._p(bc, "STT_REPLACEMENTS", {"a cello": "Accelo"})
        self._p(bc, "STT_REPLACEMENTS_PARAKEET", {"Accelo please": "Accelo"})
        text, _conf = bc._transcribe_capture(self.audio)
        self.assertEqual(text, "open Accelo")

    def test_rescue_on_a_lost_wake_word_in_wake_mode(self):
        bc = self.bc
        self._p(bc, "_stt_alt", _FakeEngine(text="Travis, what time is it?"))
        self._p(bc, "_require_wake_runtime", True)
        # record_speech's stash for this clip: a full 12-chunk pre-roll.
        bc._last_capture_preroll[0] = (len(self.audio), 12 * 1024)
        head = self._p(bc._tail_vad, "speech_in_head", return_value=True)
        tr = self._p(bc, "transcribe", return_value=W_RES)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = bc._transcribe_capture(self.audio)
        self.assertIs(res, W_RES)
        tr.assert_called_once()
        head.assert_called_once()
        self.assertEqual(head.call_args.args[1], 0.8)
        self.assertEqual(head.call_args.kwargs["start"], 12 * 1024)
        self.assertEqual(self._engine_notes(), ["parakeet-rescued"])
        self.assertIn("whisper rescue (no-wake", out.getvalue())
        self.assertNotIn("Travis", out.getvalue())
        self.assertEqual(bc._parakeet_primary.rescues, 1)

    def test_rescue_on_a_lost_wake_word_in_standby(self):
        # Wake-word mode off, but standby wakes on the wake word alone: a
        # misheard one would lose the wake just the same.
        bc = self.bc
        self._p(bc, "_stt_alt", _FakeEngine(text="Travis, are you there?"))
        self._p(bc, "_standby_mode", [True])
        self._p(bc._tail_vad, "speech_in_head", return_value=True)
        tr = self._p(bc, "transcribe", return_value=W_RES)
        self.assertIs(bc._transcribe_capture(self.audio), W_RES)
        tr.assert_called_once()
        self.assertEqual(self._engine_notes(), ["parakeet-rescued"])
        self._p(bc, "_standby_mode", [False])
        self._p(bc, "_sleep_mode", [True])
        self.assertTrue(bc._parakeet_wake_lost("Travis, are you there?"))
        self._p(bc, "_sleep_mode", [False])
        self.assertFalse(bc._parakeet_wake_lost("Travis, are you there?"))

    def test_a_wake_led_transcript_is_not_rescued(self):
        bc = self.bc
        self._p(bc, "_stt_alt", _FakeEngine(text="Hey Jarvis, lights"))
        self._p(bc, "_require_wake_runtime", True)
        head = self._p(bc._tail_vad, "speech_in_head", return_value=True)
        tr = self._p(bc, "transcribe", return_value=W_RES)
        self.assertEqual(bc._transcribe_capture(self.audio)[0],
                         "Hey Jarvis, lights")
        tr.assert_not_called()
        head.assert_not_called()

    def test_rescue_on_empty_text(self):
        bc = self.bc
        self._p(bc, "_stt_alt", _FakeEngine(text="", logprobs=()))
        tr = self._p(bc, "transcribe", return_value=W_RES)
        self.assertIs(bc._transcribe_capture(self.audio), W_RES)
        tr.assert_called_once()
        self.assertEqual(self._engine_notes(), ["parakeet-rescued"])

    def test_wake_mode_off_keeps_parakeets_text(self):
        bc = self.bc
        self._p(bc, "_stt_alt", _FakeEngine(text="turn the lights off"))
        tr = self._p(bc, "transcribe", return_value=W_RES)
        self.assertEqual(bc._transcribe_capture(self.audio)[0],
                         "turn the lights off")
        tr.assert_not_called()

    def test_a_load_failure_latches_off_and_whisper_decodes(self):
        bc = self.bc
        load = self._p(bc._stt_parakeet, "load",
                       side_effect=FileNotFoundError("no model"))
        tr = self._p(bc, "transcribe", return_value=W_RES)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIs(bc._transcribe_capture(self.audio), W_RES)
            self.assertIs(bc._transcribe_capture(self.audio), W_RES)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(tr.call_count, 2)
        self.assertIn("FileNotFoundError", bc._parakeet_latch.failed)
        self.assertEqual(out.getvalue().count("parakeet off for this session"),
                         1)
        self.assertEqual(self._engine_notes(), ["whisper-fallback", "whisper"])

    def test_the_boot_warmer_decodes_a_second_of_silence(self):
        bc = self.bc
        seen = []

        class _Eng(_FakeEngine):
            def recognize(self, a, sample_rate=None):
                seen.append((len(a), str(a.dtype), float(abs(a).max())))
                return _Res("", [])
        self._p(bc._stt_parakeet, "load", return_value=_Eng())
        bc._warm_parakeet()
        self.assertEqual(seen, [(bc.SAMPLE_RATE, "float32", 0.0)])

    def test_the_owners_turn_never_waits_on_the_boot_load(self):
        """The boot warmer holds _parakeet_lock while onnxruntime loads the
        model (seconds; a hung load would be forever). The owner's capture
        must not wait behind it: Whisper decodes this one ('whisper-loading',
        no latch), and the next capture after the load is Parakeet's (R6
        review, finding 3). Event-based."""
        bc = self.bc
        eng = _FakeEngine(text="Jarvis, lights")
        started, release, done = (threading.Event(), threading.Event(),
                                  threading.Event())

        def slow_load(model_dir, threads):
            started.set()
            release.wait(10.0)
            return eng
        self._p(bc._stt_parakeet, "load", side_effect=slow_load)
        tr = self._p(bc, "transcribe", return_value=W_RES)
        box = {}

        def owner():
            try:
                box["res"] = bc._transcribe_capture(self.audio)
            finally:
                done.set()
        warm = threading.Thread(target=bc._warm_parakeet, daemon=True)
        own = threading.Thread(target=owner, daemon=True)
        warm.start()
        self.assertTrue(started.wait(5.0))
        try:
            own.start()
            finished = done.wait(5.0)
        finally:
            release.set()
            warm.join(5.0)
            own.join(5.0)
        self.assertTrue(finished, "the owner's turn waited on the boot load")
        self.assertIs(box["res"], W_RES)
        tr.assert_called_once()
        self.assertEqual(self._engine_notes(), ["whisper-loading"])
        self.assertEqual(bc._parakeet_latch.failed, "")
        self.assertEqual(bc._parakeet_primary.decodes, 0)
        # Loaded: the next capture is Parakeet's.
        self.assertEqual(bc._transcribe_capture(self.audio)[0],
                         "Jarvis, lights")
        self.assertEqual(self._engine_notes(), ["whisper-loading",
                                                "parakeet"])

    def test_the_boot_warmer_also_warms_the_rescue_detector(self):
        # The rescue's head check runs Silero on the turn thread; with
        # TURN_TAIL_PROBE off nothing else would load it before the first
        # wake-word capture (R6 review, finding 9).
        bc = self.bc
        self._p(bc._stt_parakeet, "load", return_value=_FakeEngine())
        warm = self._p(bc._tail_vad, "warm", return_value=True)
        bc._warm_parakeet()
        warm.assert_called_once_with()
        # A detector that will not load costs Parakeet nothing: no latch,
        # no raise (the head check then answers None and the rescue runs).
        warm.side_effect = RuntimeError("no silero")
        bc._warm_parakeet()
        self.assertEqual(bc._parakeet_latch.failed, "")

    def test_a_failing_boot_warmer_latches_quietly_and_raises(self):
        bc = self.bc
        self._p(bc._stt_parakeet, "load", side_effect=RuntimeError("nope"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(RuntimeError):
                bc._warm_parakeet()
        self.assertIn("nope", bc._parakeet_latch.failed)
        self.assertEqual(out.getvalue(), "")   # the [warm] line says it


class _OnsetSession:
    """A Silero stand-in with the real detector's onset lag: a window's speech
    probability climbs 0.2 for each loud window in a row, so speech crosses
    SPEECH_THRESHOLD (0.5) on its third loud window. The real detector scored
    a wake word's first speech window 0.04-0.49 on 15 clips shaped the way
    record_speech returns them (R6 review, 2026-10-02)."""

    def __init__(self):
        self.batches = []

    def run(self, _names, feeds):
        import numpy as np
        from core import endpointing as ep
        x = feeds["input"]
        self.batches.append(x.shape[0])
        loud = np.abs(x[:, ep.CONTEXT:]).mean(axis=1) > 0.05
        probs, k = [], 0
        for hit in loud:
            k = k + 1 if hit else 0
            probs.append(min(1.0, 0.2 * k))
        return (np.asarray(probs, dtype=np.float32).reshape(-1, 1),
                feeds["h"], feeds["c"])


# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class R6RescueHeadTests(_Base):
    """The rescue's "speech in the first 0.8 s" is asked from where the
    owner's speech starts — the chunk that tripped record_speech — not from
    the clip's first sample: record_speech puts PRE_BUFFER (12) chunks of
    below-threshold pre-roll (0.768 s) in front of it (R6 review, finding
    1). Through the REAL record_speech and the REAL SileroVad (a fake ORT
    session with the onset lag above)."""

    def setUp(self):
        super().setUp()
        self._p(self.bc, "STT_ENGINE", "parakeet")
        self.lim = int(self.bc.SILENCE_SECS * self.bc.SAMPLE_RATE / 1024)

    def _record(self, n_quiet, n_voiced):
        bc = self.bc
        np = bc.np
        lim = self.lim

        class FakeStream:
            device = 1
            latency = 0.0

            def __init__(self, *a, callback=None, **k):
                self.cb = callback

            def start(self):
                for amp, n in ((0.001, n_quiet), (0.2, n_voiced),
                               (0.0, lim)):
                    frame = np.full((1024, 1), amp, dtype="float32")
                    for _ in range(n):
                        self.cb(frame, 1024, None, None)

        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "_mic_muted", [False])
        self._p(bc, "_capture_holds_mic", return_value=False)
        self._p(bc, "_input_backoff_wait", return_value=False)
        self._p(bc, "get_input_device", return_value=1)
        self._p(bc, "_safe_close_stream", lambda s: None)
        self._p(bc.sd, "InputStream", FakeStream)
        self._p(bc, "_note_live_capture", lambda *a, **k: None)
        self._p(bc, "_filler_capture_mark", lambda *a, **k: None)
        self._p(bc, "_fanout_record_frame", lambda *a, **k: None)
        self._p(bc, "_process_capture_chunk",
                lambda data, sr, skip_ns=False: data)
        self._p(bc, "_spec_stt_should_snapshot", return_value=False)
        self._p(bc, "pause_face_tracking")
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "_utterance_in_progress", [False])
        self._p(bc, "VAD_THRESHOLD", 0.008)
        with contextlib.redirect_stdout(io.StringIO()):
            audio = bc.record_speech(timeout=3)
        self.assertIsNotNone(audio, "no utterance was captured")
        return audio

    def test_record_speech_publishes_its_pre_roll(self):
        bc = self.bc
        audio = self._record(20, 15)
        self.assertEqual(len(audio), (12 + 15 + self.lim) * 1024)
        self.assertEqual(bc._last_capture_preroll[0], (len(audio), 12 * 1024))
        # A capture that started with less than a full ring: what it had.
        audio = self._record(5, 15)
        self.assertEqual(bc._last_capture_preroll[0], (len(audio), 5 * 1024))

    def test_a_lost_wake_word_after_a_full_pre_roll_is_rescued(self):
        bc = self.bc
        audio = self._record(20, 15)
        sess = _OnsetSession()
        self._p(bc, "_tail_vad", bc._endpointing.SileroVad(
            model_path="silero.onnx", session_factory=lambda p: sess))
        self._p(bc, "_stt_alt", _FakeEngine(text="Travis, what time is it?"))
        self._p(bc, "_require_wake_runtime", True)
        tr = self._p(bc, "transcribe", return_value=W_RES)
        with contextlib.redirect_stdout(io.StringIO()):
            res = bc._transcribe_capture(audio)
        self.assertIs(res, W_RES, "the head check scored the pre-roll")
        tr.assert_called_once()
        self.assertEqual(self._engine_notes(), ["parakeet-rescued"])
        self.assertEqual(sess.batches, [25])    # 0.8 s = 25 windows

    def test_an_unknown_pre_roll_rescues(self):
        # A clip record_speech did not just return (its length does not
        # match the stash): the head cannot be placed, so — like an
        # unusable detector — the rescue runs.
        bc = self.bc
        self._p(bc, "_last_capture_preroll", [(123, 0)])
        head = self._p(bc._tail_vad, "speech_in_head")
        self._p(bc, "_stt_alt", _FakeEngine(text="Travis, what time is it?"))
        self._p(bc, "_require_wake_runtime", True)
        tr = self._p(bc, "transcribe", return_value=W_RES)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(bc._transcribe_capture(self.audio), W_RES)
        tr.assert_called_once()
        head.assert_not_called()


def _load_music_skill():
    """The REAL skills/standby_audio_detect, under the name the monolith
    looks it up by (sys.modules['skill_standby_audio_detect'])."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location(
        "skill_standby_audio_detect",
        os.path.join(root, "skills", "standby_audio_detect.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class R6RescueGateTests(_Base):
    """The rescue fires whenever the LIVE wake gates would drop Parakeet's
    text for want of a wake word — not only in wake-word mode and standby
    (R6 review, finding 2 / finding 10): media playing or sustained room
    music with AMBIENT_MUSIC_REFUSE_WAKE, the post-dialogue hold; and not
    when the live gates would let the text through: the follow-up window,
    the standby greeting reply, a standby wake word anywhere in the line.
    The state is peeked, never consumed."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        import core.config as cfg
        from core.followup_window import FollowupWindow
        self._p(bc, "STT_ENGINE", "parakeet")
        bc._last_capture_preroll[0] = (len(self.audio), 0)
        self.head = self._p(bc._tail_vad, "speech_in_head",
                            return_value=True)
        self.tr = self._p(bc, "transcribe", return_value=W_RES)
        self.fw = FollowupWindow(30.0)
        self._p(bc, "_followup_window", self.fw)
        self.smtc = bc._smtc_media_playing        # _Base: False
        self._p(cfg, "AMBIENT_MUSIC_REFUSE_WAKE", True)
        self.skill = _load_music_skill()
        self.room = self._p(self.skill, "is_music_currently_playing",
                            return_value=False)
        self._set_module("skill_standby_audio_detect", self.skill)

    def _run(self, text):
        bc = self.bc
        self._p(bc, "_stt_alt", _FakeEngine(text=text))
        self.tr.reset_mock()
        with contextlib.redirect_stdout(io.StringIO()):
            res = bc._transcribe_capture(self.audio)
        return res is W_RES

    def test_media_playing_with_wake_mode_off_rescues(self):
        self.smtc.return_value = True
        self.assertTrue(self._run("Travis, pause the music"))
        self.tr.assert_called_once()
        # ...and the live gate agrees the line would have been refused.
        self.assertEqual(self.bc._should_refuse_background_audio(
            "Travis, pause the music"), (True, "media playing"))

    def test_sustained_room_music_with_wake_mode_off_rescues(self):
        self.room.return_value = True
        self.assertTrue(self._run("Travis, pause the music"))

    def test_the_music_switch_off_keeps_parakeets_text(self):
        import core.config as cfg
        self._p(cfg, "AMBIENT_MUSIC_REFUSE_WAKE", False)
        self.smtc.return_value = True
        self.room.return_value = True
        self.assertFalse(self._run("Travis, pause the music"))

    def test_the_post_dialogue_hold_rescues(self):
        bc = self.bc
        bc._turn_hold_until[0] = bc.time.monotonic() + 30.0
        self.assertTrue(self._run("Travis, stop"))

    def test_a_wake_led_line_never_asks_smtc(self):
        self.smtc.side_effect = AssertionError("SMTC read for a wake line")
        self.assertFalse(self._run("Jarvis, pause the music"))
        self.smtc.assert_not_called()
        self.head.assert_not_called()

    def test_an_open_follow_up_window_is_peeked_not_extended(self):
        bc = self.bc
        self._p(bc, "_require_wake_runtime", True)
        self.fw.note_addressed()
        until = self.fw._until
        self.assertFalse(self._run("and the kitchen too"))
        self.assertEqual(self.fw._until, until)   # admit() not called

    def test_an_armed_greeting_reply_is_peeked_not_taken(self):
        bc = self.bc
        self._p(bc, "_require_wake_runtime", True)
        bc._standby_greet_admit_until[0] = bc.time.time() + 8.0
        self.assertFalse(self._run("what time is it"))
        self.assertGreater(bc._standby_greet_admit_until[0], 0.0)

    def test_standby_wakes_on_the_wake_word_anywhere(self):
        # Standby wakes on the wake word anywhere in the line, so a
        # mid-sentence "Jarvis" from Parakeet would wake: keep it (the old
        # prefix rule rescued it and could lose the wake to Whisper).
        bc = self.bc
        self._p(bc, "_standby_mode", [True])
        self._p(bc, "_sleep_mode", [True])
        self.assertFalse(bc._text_has_wake_prefix("is that you there Jarvis"))
        self.assertFalse(self._run("is that you there Jarvis"))
        self.assertTrue(self._run("is that you there Travis"))

    def test_standby_over_room_music_needs_the_wake_prefix(self):
        # The standby lyric guard: over sustained room music a mid-line
        # wake word is refused, so it is rescued.
        bc = self.bc
        self._p(bc, "_standby_mode", [True])
        self._p(bc, "_sleep_mode", [True])
        self.room.return_value = True
        self.assertTrue(bc._audio_music_should_refuse_wake(
            "is that you there Jarvis"))
        self.assertTrue(self._run("is that you there Jarvis"))
        self.assertFalse(self._run("Jarvis, are you there"))

    def test_the_live_gate_and_the_verdict_share_one_rule(self):
        # One copy: the live background gate and the rescue's verdict both
        # ask _bg_refuse_rule (a re-implemented verdict stays green here).
        bc = self.bc
        rule = self._p(bc, "_bg_refuse_rule",
                       return_value=(True, "sentinel"))
        self.assertEqual(bc._should_refuse_background_audio("x"),
                         (True, "sentinel"))
        self.assertEqual(bc._wake_gate_verdict("x", bc._wake_gate_state()),
                         "sentinel")
        self.assertEqual(rule.call_count, 2)
        heard = self._p(bc, "_wake_word_heard", return_value=False)
        bc._turn_hold_until[0] = bc.time.monotonic() + 30.0
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(bc._dialogue_hold_ignored("Jarvis, stop"))
        self.assertEqual(bc._wake_gate_verdict("Jarvis, stop",
                                               bc._wake_gate_state()),
                         "dialogue hold")
        self.assertEqual(heard.call_count, 2)

    def test_the_verdict_matches_the_live_gate_in_every_state(self):
        import itertools
        bc = self.bc
        texts = ("Jarvis, pause the music", "pause the music",
                 "um Jarvis what time is it", "I told Jarvis yesterday", "")
        for wm, music, media, room, fw, greet in itertools.product(
                (False, True), repeat=6):
            import core.config as cfg
            self._p(cfg, "AMBIENT_MUSIC_REFUSE_WAKE", music)
            self._p(bc, "_require_wake_runtime", wm)
            self.smtc.return_value = media
            self.room.return_value = room
            for text in texts:
                self.fw._until = 0.0
                if fw:
                    self.fw.note_addressed()
                bc._standby_greet_admit_until[0] = (
                    bc.time.time() + 8.0 if greet else 0.0)
                why = bc._wake_gate_verdict(text, bc._wake_gate_state())
                snap = bc._wake_gate_state(resolve_media=True)
                why_snap = bc._wake_gate_verdict(text, snap)
                live = bc._should_refuse_background_audio(text)
                case = (wm, music, media, room, fw, greet, text)
                self.assertEqual(bool(why), live[0], case)
                if live[0]:
                    self.assertEqual(why, live[1], case)
                self.assertEqual(why_snap, why, case)

    def test_the_standby_lyric_guard_is_the_skills_rule(self):
        bc = self.bc
        self._p(bc, "_standby_mode", [True])
        self._p(bc, "_sleep_mode", [True])
        for room in (False, True):
            self.room.return_value = room
            for text in ("Jarvis", "Jarvis, lights", "hey there Jarvis",
                         "um Jarvis lights", "jar visit", "nothing here"):
                why = bc._wake_gate_verdict(text, bc._wake_gate_state())
                heard = bc._wake_word_heard(text)
                want = ("standby" if not heard else
                        "standby music" if self.skill.should_refuse_wake(text)
                        else "")
                self.assertEqual(why, want, (room, text))


# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class R6ShadowTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "STT_SHADOW", "parakeet")

    def test_whisper_result_stands_and_the_capture_is_offered(self):
        bc = self.bc
        tr = self._p(bc, "transcribe", return_value=W_RES)
        offer = self._p(bc._parakeet_shadow, "offer", return_value=True)
        run = self._p(bc._parakeet_primary, "run")
        self._p(bc, "_last_recording_peak", 0.031)
        self.assertIs(bc._transcribe_capture(self.audio), W_RES)
        tr.assert_called_once()
        run.assert_not_called()
        offer.assert_called_once()
        a, text, conf, ms, peak, ctx = offer.call_args.args
        self.assertIs(a, self.audio)
        self.assertEqual((text, conf, peak), (W_RES[0], W_RES[1], 0.031))
        self.assertIsInstance(ms, int)
        self.assertEqual(set(ctx), {"owner_idle_s", "since_jarvis_s",
                                    "jarvis_asked", "prompt_pending",
                                    "wake_mode", "standby", "noise_filter"})
        self.assertEqual(self._engine_notes(), [])

    def test_a_shadow_fault_never_reaches_the_turn(self):
        bc = self.bc
        self._p(bc, "transcribe", return_value=W_RES)
        self._p(bc._parakeet_shadow, "offer", side_effect=RuntimeError("x"))
        self.assertIs(bc._transcribe_capture(self.audio), W_RES)

    def test_busy_follows_the_turn_and_the_utterance(self):
        bc = self.bc
        self._p(bc, "_turn_in_progress", [False])
        self._p(bc, "_utterance_in_progress", [False])
        self.assertFalse(bc._parakeet_shadow_busy())
        bc._turn_in_progress[0] = True
        self.assertTrue(bc._parakeet_shadow_busy())
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = True
        self.assertTrue(bc._parakeet_shadow_busy())

    def test_the_judge_runs_the_real_gates(self):
        bc = self.bc
        good = {"no_speech_prob": 0.0, "avg_logprob": -0.2}
        ctx = {"wake_mode": True, "owner_idle_s": 5.0, "since_jarvis_s": None,
               "jarvis_asked": False, "prompt_pending": False,
               "noise_filter": True}
        j = bc._parakeet_shadow_judge("Jarvis, what time is it?", good,
                                      0.05, ctx)
        self.assertTrue(j["wake_prefix"])
        self.assertTrue(j["accepted"])
        j = bc._parakeet_shadow_judge("Travis, what time is it?", good,
                                      0.05, ctx)
        self.assertTrue(j["wake_refused"])
        self.assertFalse(j["accepted"])
        j = bc._parakeet_shadow_judge("Thank you.", good, 0.009,
                                      dict(ctx, wake_mode=False,
                                           owner_idle_s=None))
        self.assertEqual(j["noise_verdict"], "noise")
        self.assertFalse(j["accepted"])
        j = bc._parakeet_shadow_judge("", {"no_speech_prob": 1.0,
                                           "avg_logprob": -10.0},
                                      0.05, dict(ctx, wake_mode=False))
        self.assertFalse(j["valid"])
        self.assertEqual(j["filter_reason"], "empty")

    def test_rows_go_to_the_data_dir_file(self):
        bc = self.bc
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.environ["JARVIS_DATA_DIR"] = d
        self.assertTrue(bc._parakeet_ab_write({"a": 1}))
        with open(os.path.join(d, "stt_ab.jsonl"), encoding="utf-8") as f:
            self.assertEqual([json.loads(x) for x in f], [{"a": 1}])

    def test_the_shadow_decodes_under_parakeet_lock_end_to_end(self):
        bc = self.bc
        eng = _FakeEngine(text="Jarvis what time is it")
        self._p(bc, "_stt_alt", eng)
        self._p(bc, "_turn_in_progress", [False])
        self._p(bc, "_utterance_in_progress", [False])
        rows = []
        self._p(bc, "_parakeet_ab_write",
                side_effect=lambda r: rows.append(r) or True)
        sh = bc._stt_parakeet.Shadow(
            lambda a: bc._parakeet_decode(a), bc._parakeet_shadow_busy,
            bc._parakeet_shadow_judge, lambda r: bc._parakeet_ab_write(r),
            latch=bc._stt_parakeet.Latch(), log=lambda s: None)
        self.assertTrue(sh.offer(self.audio, "Jarvis, what time is it?",
                                 W_RES[1], 1400, 0.05,
                                 bc._parakeet_shadow_ctx(), start=False))
        with bc._stt_lock:          # an ambient decode cannot stall it
            row = sh.process(sh._q.get_nowait())
        self.assertEqual(rows, [row])
        self.assertTrue(row["same_words"])
        self.assertEqual(row["parakeet"]["text"], "Jarvis what time is it")
        self.assertEqual(eng.calls, 1)


if __name__ == "__main__":
    unittest.main()
