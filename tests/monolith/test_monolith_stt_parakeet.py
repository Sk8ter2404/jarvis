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
        saved = (latch.failed, prim.decodes, prim.rescues, prim._hot_logged)

        def _restore():
            (latch.failed, prim.decodes, prim.rescues,
             prim._hot_logged) = saved
        latch.failed = ""
        prim.decodes = prim.rescues = 0
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
        head = self._p(bc._tail_vad, "speech_in_head", return_value=True)
        tr = self._p(bc, "transcribe", return_value=W_RES)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = bc._transcribe_capture(self.audio)
        self.assertIs(res, W_RES)
        tr.assert_called_once()
        head.assert_called_once()
        self.assertEqual(head.call_args.args[1], 0.8)
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
        self.assertTrue(bc._parakeet_wake_mode())
        self._p(bc, "_sleep_mode", [False])
        self.assertFalse(bc._parakeet_wake_mode())

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

    def test_a_failing_boot_warmer_latches_quietly_and_raises(self):
        bc = self.bc
        self._p(bc._stt_parakeet, "load", side_effect=RuntimeError("nope"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(RuntimeError):
                bc._warm_parakeet()
        self.assertIn("nope", bc._parakeet_latch.failed)
        self.assertEqual(out.getvalue(), "")   # the [warm] line says it


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
