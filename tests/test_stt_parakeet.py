"""core/stt_parakeet.py — Parakeet TDT on the CPU for the owner's captures
(speed plan R6, 2026-10-02).

Light tier: fakes for onnx_asr / onnxruntime / the engine (CI installs
neither), so these pin the logic only:
  * the confidence map (monotone, never above 0; empty -> (1.0, -10.0));
  * nothing heavy is imported at module load, or with both flags off;
  * load() asks onnx-asr for exactly the int8, CPU-only, tuned model;
  * a Parakeet exception latches it off and Whisper decodes;
  * the rescue fires on empty text and on a lost wake word;
  * the shadow worker waits out a turn (bounded), never logs a transcript,
    and its queue is bounded.

The heavy tier at the bottom loads the REAL model. It runs only when
JARVIS_TEST_PARAKEET=1 and the model folder and onnx_asr exist: it costs
~2 s of load and several CPU-seconds, so it is an explicit, idle-checked
local run, never part of the default suite. JARVIS_TEST_PARAKEET_REF=<wav>
adds the owner's consented reference clip (any rate; resampled to 16 kHz;
only its length is printed, never its words).

Run: python tools/run_tests.py test_stt_parakeet
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from core import stt_parakeet as sp

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _prefix(text):
    w = (text or "").strip().lower().split()
    return bool(w) and w[0].strip(",.!?") == "jarvis"


class _Clock:
    """A fake monotonic clock; wait(s) advances it instead of sleeping."""

    def __init__(self, t=100.0):
        self.t = t
        self.waits = []

    def __call__(self):
        return self.t

    def wait(self, s):
        self.waits.append(s)
        self.t += s
        return False


# ── settings ──────────────────────────────────────────────────────────────
class SettingsTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop(sp.ENV_ENGINE, None)

    def test_engine_defaults_to_whisper(self):
        for v in (None, "", "whisper", "WHISPER ", "faster-whisper", 3):
            self.assertEqual(sp.engine_setting(v), "whisper", v)
        self.assertEqual(sp.engine_setting(" Parakeet "), "parakeet")

    def test_env_override_wins_when_it_names_an_engine(self):
        os.environ[sp.ENV_ENGINE] = "parakeet"
        self.assertEqual(sp.engine_setting("whisper"), "parakeet")
        os.environ[sp.ENV_ENGINE] = "whisper"
        self.assertEqual(sp.engine_setting("parakeet"), "whisper")
        os.environ[sp.ENV_ENGINE] = "nonsense"
        self.assertEqual(sp.engine_setting("parakeet"), "parakeet")

    def test_shadow_setting(self):
        self.assertEqual(sp.shadow_setting(""), "")
        self.assertEqual(sp.shadow_setting(None), "")
        self.assertEqual(sp.shadow_setting("parakeet"), "parakeet")
        self.assertEqual(sp.shadow_setting("whisper"), "")

    def test_config_ships_both_flags_off(self):
        from core import config
        self.assertEqual(config.STT_ENGINE, "whisper")
        self.assertEqual(config.STT_SHADOW, "")
        self.assertEqual(config.PARAKEET_THREADS, 8)
        self.assertEqual(config.STT_REPLACEMENTS_PARAKEET, {})
        self.assertEqual(sp.anchors_or_default(config.PARAKEET_CONF_ANCHORS),
                         sp.DEFAULT_CONF_ANCHORS)
        self.assertTrue(config.PARAKEET_MODEL_DIR.endswith(
            "parakeet-tdt-0.6b-v2-onnx"))


# ── map_conf ──────────────────────────────────────────────────────────────
class MapConfTests(unittest.TestCase):
    def test_empty_gives_whispers_no_speech_shape(self):
        for lp in ((), [], None, [float("nan")], ["x"]):
            c = sp.map_conf(lp)
            self.assertEqual((c["no_speech_prob"], c["avg_logprob"]),
                             (1.0, -10.0), lp)
            self.assertEqual(c["n_tok"], 0)

    def test_monotone_and_never_above_zero(self):
        prev = None
        for i in range(0, 4001):
            mean = -4.0 + i * 0.001
            c = sp.map_conf([mean, mean])
            self.assertLessEqual(c["avg_logprob"], 0.0)
            self.assertEqual(c["no_speech_prob"], 0.0)
            if prev is not None:
                self.assertGreaterEqual(c["avg_logprob"], prev, mean)
            prev = c["avg_logprob"]

    def test_custom_anchors_are_monotone_and_capped_at_zero_too(self):
        anchors = [[-2.0, -5.0], [-0.1, -0.2], [0.0, 0.0]]
        prev = -99.0
        for i in range(0, 301):
            v = sp.map_conf([-3.0 + i * 0.01], anchors)["avg_logprob"]
            self.assertLessEqual(v, 0.0)
            self.assertGreaterEqual(v, prev)
            prev = v

    def test_the_anchors_are_hit_exactly(self):
        for x, y in sp.DEFAULT_CONF_ANCHORS:
            self.assertAlmostEqual(sp.map_conf([x])["avg_logprob"], y)
        # Clamped at both ends.
        lo = sp.DEFAULT_CONF_ANCHORS[0][1]
        self.assertAlmostEqual(sp.map_conf([-50.0])["avg_logprob"], lo)

    def test_clean_speech_clears_the_gate_and_mush_does_not(self):
        from core import speech_filter as sf
        clean = sp.map_conf([-0.001, -0.01, -0.05])          # calibration
        mush = sp.map_conf([-0.9, -1.2, -0.4])
        self.assertGreater(clean["avg_logprob"], sf.WHISPER_MIN_AVG_LOGPROB)
        self.assertLess(mush["avg_logprob"], sf.WHISPER_MIN_AVG_LOGPROB)

    def test_raw_numbers_are_kept(self):
        c = sp.map_conf([-0.5, -0.1, float("inf"), 0.2])
        self.assertEqual(c["n_tok"], 3)            # inf skipped, 0.2 -> 0
        self.assertAlmostEqual(c["tok_lp_mean"], -0.2)
        self.assertAlmostEqual(c["tok_lp_min"], -0.5)

    def test_bad_anchors_fall_back_to_the_defaults(self):
        D = sp.DEFAULT_CONF_ANCHORS
        for bad in ([], [[0, 0]], [[-1, -1], [-1, -0.5]],      # dup x
                    [[-1, -0.5], [0, -1]],                      # y falls
                    [[-1, -1], [0.5, 0]],                       # x > 0
                    [[-1, -1], [0, 0.3]],                       # y > 0
                    [[-1, "x"], [0, 0]], "nope", None, 5):
            self.assertEqual(sp.anchors_or_default(bad), D, bad)
        self.assertEqual(sp.anchors_or_default([[0, -0.1], [-1, -2]]),
                         ((-1.0, -2.0), (0.0, -0.1)))


# ── session options / load / transcribe ───────────────────────────────────
class _FakeSO:
    def __init__(self):
        self.entries = {}
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.execution_mode = None

    def add_session_config_entry(self, k, v):
        self.entries[k] = v


def _fake_rt():
    return types.SimpleNamespace(
        SessionOptions=_FakeSO,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="SEQ"))


class SessionOptionsTests(unittest.TestCase):
    def test_copied_from_kokoros_tuned_session(self):
        so = sp.session_options(8, rt=_fake_rt())
        self.assertEqual(so.intra_op_num_threads, min(8, os.cpu_count() or 8))
        self.assertEqual(so.inter_op_num_threads, 1)
        self.assertEqual(so.execution_mode, "SEQ")
        self.assertEqual(so.entries, {"session.intra_op.allow_spinning": "0",
                                      "session.inter_op.allow_spinning": "0"})

    def test_thread_count_is_clamped(self):
        with mock.patch.object(sp.os, "cpu_count", return_value=4):
            self.assertEqual(sp.session_options(64, rt=_fake_rt())
                             .intra_op_num_threads, 4)
            self.assertEqual(sp.session_options("junk", rt=_fake_rt())
                             .intra_op_num_threads, 4)
        self.assertEqual(sp.session_options(0, rt=_fake_rt())
                         .intra_op_num_threads,
                         min(8, os.cpu_count() or 8))


class LoadTests(unittest.TestCase):
    def test_missing_package_raises_without_importing_it(self):
        with mock.patch.object(sp.importlib.util, "find_spec",
                               return_value=None), \
                mock.patch.dict(sys.modules):
            sys.modules.pop("onnx_asr", None)
            with self.assertRaises(RuntimeError):
                sp.load(tempfile.gettempdir())
            self.assertNotIn("onnx_asr", sys.modules)

    def test_missing_model_dir_raises(self):
        with mock.patch.object(sp, "package_available", return_value=True):
            with self.assertRaises(FileNotFoundError):
                sp.load(os.path.join(tempfile.gettempdir(), "no-such-model"))

    def test_asks_onnx_asr_for_the_int8_cpu_only_tuned_model(self):
        calls = []
        stamped = object()

        class _Model:
            def with_timestamps(self):
                calls.append("with_timestamps")
                return stamped

        def load_model(*a, **k):
            calls.append((a, k))
            return _Model()

        fake_asr = types.SimpleNamespace(load_model=load_model)
        with mock.patch.object(sp, "package_available", return_value=True), \
                mock.patch.dict(sys.modules, {"onnx_asr": fake_asr,
                                              "onnxruntime": _fake_rt()}):
            d = tempfile.mkdtemp()
            try:
                got = sp.load(d, threads=6)
            finally:
                os.rmdir(d)
        self.assertIs(got, stamped)
        (args, kw), last = calls
        self.assertEqual(last, "with_timestamps")
        self.assertEqual(args, ("nemo-conformer-tdt", d))
        self.assertEqual(kw["quantization"], "int8")
        self.assertEqual(kw["providers"], ["CPUExecutionProvider"])
        so = kw["sess_options"]
        self.assertEqual(so.intra_op_num_threads, min(6, os.cpu_count() or 6))
        self.assertEqual(so.inter_op_num_threads, 1)
        self.assertEqual(so.entries["session.intra_op.allow_spinning"], "0")


class _Res:
    def __init__(self, text, logprobs):
        self.text = text
        self.logprobs = logprobs


class _Engine:
    def __init__(self, text=" Jarvis,  what   time is it? ",
                 logprobs=(-0.01, -0.02)):
        self.seen = []
        self._res = _Res(text, list(logprobs))

    def recognize(self, a, sample_rate=None):
        self.seen.append((a, sample_rate))
        return self._res


class TranscribeTests(unittest.TestCase):
    def test_casts_to_contiguous_float32_and_shapes_the_result(self):
        eng = _Engine()
        audio = np.linspace(-0.1, 0.1, 32000, dtype=np.float64)[::-1]
        text, conf = sp.transcribe(eng, audio)
        a, sr = eng.seen[0]
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(a.flags["C_CONTIGUOUS"])
        self.assertEqual(sr, 16000)
        self.assertEqual(text, "Jarvis, what time is it?")
        self.assertEqual(conf["engine"], "parakeet")
        self.assertEqual(conf["speech_s"], 2.0)
        self.assertIsInstance(conf["stt_ms"], int)
        self.assertEqual(conf["no_speech_prob"], 0.0)
        self.assertEqual(conf["n_tok"], 2)

    def test_empty_text_is_no_speech_even_with_tokens(self):
        text, conf = sp.transcribe(_Engine(text="  ", logprobs=(-0.1,)),
                                   np.zeros(16000, np.float32))
        self.assertEqual(text, "")
        self.assertEqual((conf["no_speech_prob"], conf["avg_logprob"]),
                         (1.0, -10.0))

    def test_a_too_short_clip_is_not_decoded(self):
        eng = _Engine()
        text, conf = sp.transcribe(eng, np.zeros(10, np.float32))
        self.assertEqual((text, eng.seen), ("", []))
        self.assertEqual(conf["avg_logprob"], -10.0)

    def test_column_audio_is_flattened_and_stereo_refused(self):
        eng = _Engine()
        sp.transcribe(eng, np.zeros((16000, 1), np.float32))
        self.assertEqual(eng.seen[0][0].shape, (16000,))
        with self.assertRaises(ValueError):
            sp.transcribe(eng, np.zeros((16000, 2), np.float32))


# ── the rescue ────────────────────────────────────────────────────────────
class RescueReasonTests(unittest.TestCase):
    def _r(self, text, wake=True, head=True):
        calls = []

        def head_speech():
            calls.append(1)
            if isinstance(head, Exception):
                raise head
            return head
        why = sp.rescue_reason(text,
                               wake_lost=lambda t: wake and not _prefix(t),
                               head_speech=head_speech)
        return why, len(calls)

    def test_empty_text_is_rescued_whatever_the_mode(self):
        self.assertEqual(self._r("", wake=False), ("empty", 0))
        self.assertEqual(self._r("   ", wake=True), ("empty", 0))
        self.assertEqual(self._r(None, wake=False), ("empty", 0))

    def test_missing_wake_prefix_in_wake_mode_with_speech_up_front(self):
        self.assertEqual(self._r("Travis, what time is it?"), ("no-wake", 1))

    def test_unknown_head_rescues(self):
        self.assertEqual(self._r("Travis, lights", head=None), ("no-wake", 1))

    def test_no_rescue_cases_and_the_head_check_runs_only_when_needed(self):
        self.assertEqual(self._r("Jarvis, lights"), ("", 0))
        self.assertEqual(self._r("turn the lights off", wake=False), ("", 0))
        self.assertEqual(self._r("he said something", head=False),
                         ("", 1))

    def test_wake_lost_is_asked_about_the_text(self):
        seen = []
        why = sp.rescue_reason("Travis, lights",
                               wake_lost=lambda t: seen.append(t) or True,
                               head_speech=lambda: True)
        self.assertEqual((why, seen), ("no-wake", ["Travis, lights"]))

    def test_a_failing_check_rescues(self):
        self.assertEqual(self._r("Travis", head=RuntimeError("x")),
                         ("check-failed", 1))


# ── the primary path ──────────────────────────────────────────────────────
class _Whisper:
    def __init__(self, text="Jarvis, what time is it?"):
        self.calls = []
        self.res = (text, {"no_speech_prob": 0.0, "avg_logprob": -0.2})

    def __call__(self, audio):
        self.calls.append(audio)
        return self.res


def _primary(decode, whisper=None, wake=False, head=True, post=None,
             hot=False):
    log, notes = [], []
    w = whisper or _Whisper()
    p = sp.Primary(
        decode, w, latch=sp.Latch(log=log.append),
        post_text=post,
        rescue=lambda t, a: sp.rescue_reason(
            t, wake_lost=lambda x: wake and not _prefix(x),
            head_speech=lambda: head),
        note=notes.append, hotwords=lambda: hot, log=log.append)
    return p, w, log, notes


class PrimaryTests(unittest.TestCase):
    AUDIO = np.zeros(16000, np.float32)

    def test_parakeets_transcript_is_used(self):
        conf = {"no_speech_prob": 0.0, "avg_logprob": -0.3}
        p, w, log, notes = _primary(lambda a: ("Jarvis, lights on", conf))
        self.assertEqual(p.run(self.AUDIO), ("Jarvis, lights on", conf))
        self.assertEqual((w.calls, notes, log), ([], ["parakeet"], []))
        self.assertEqual((p.decodes, p.rescues), (1, 0))

    def test_an_exception_latches_off_and_falls_back_to_transcribe(self):
        n = []

        def boom(a):
            n.append(1)
            raise RuntimeError("onnx went away")
        p, w, log, notes = _primary(boom)
        self.assertEqual(p.run(self.AUDIO), w.res)
        self.assertIs(w.calls[0], self.AUDIO)
        self.assertIn("RuntimeError: onnx went away", p.latch.failed)
        self.assertEqual(len([ln for ln in log if "parakeet off" in ln]), 1)
        # Latched: Parakeet is never asked again, Whisper decodes.
        self.assertEqual(p.run(self.AUDIO), w.res)
        self.assertEqual((len(n), len(w.calls)), (1, 2))
        self.assertEqual(notes, ["whisper-fallback", "whisper"])
        self.assertEqual(len([ln for ln in log if "parakeet off" in ln]), 1)

    def test_a_replacement_failure_also_falls_back(self):
        def post(t):
            raise ValueError("bad map")
        p, w, _log, notes = _primary(lambda a: ("x y", {}), post=post)
        self.assertEqual(p.run(self.AUDIO), w.res)
        self.assertEqual(notes, ["whisper-fallback"])

    def test_the_rescue_fires_on_empty_text(self):
        p, w, log, notes = _primary(lambda a: ("", {"avg_logprob": -10.0}))
        self.assertEqual(p.run(self.AUDIO), w.res)
        self.assertEqual((p.decodes, p.rescues), (1, 1))
        self.assertEqual(notes, ["parakeet-rescued"])
        self.assertTrue(any("rescue (empty; 1 of 1" in ln for ln in log))

    def test_the_rescue_fires_on_a_lost_wake_word(self):
        p, w, log, notes = _primary(
            lambda a: ("Travis, what time is it?", {}), wake=True, head=True)
        self.assertEqual(p.run(self.AUDIO), w.res)
        self.assertEqual(notes, ["parakeet-rescued"])
        self.assertTrue(any("rescue (no-wake" in ln for ln in log))
        # ...and never prints either transcript.
        self.assertFalse(any("Travis" in ln or "time is it" in ln
                             for ln in log))

    def test_no_rescue_when_the_head_is_silent(self):
        p, w, _log, notes = _primary(
            lambda a: ("he said lights", {}), wake=True, head=False)
        self.assertEqual(p.run(self.AUDIO)[0], "he said lights")
        self.assertEqual((w.calls, notes), ([], ["parakeet"]))

    def test_replacements_run_before_the_rescue(self):
        p, w, _log, _notes = _primary(
            lambda a: ("Travis, lights", {}), wake=True, head=True,
            post=lambda t: t.replace("Travis", "Jarvis"))
        self.assertEqual(p.run(self.AUDIO)[0], "Jarvis, lights")
        self.assertEqual(w.calls, [])

    def test_not_ready_falls_back_without_latching(self):
        # The model is still loading (the boot warmer holds its lock): this
        # capture is Whisper's, nothing latches, the next one tries again.
        calls = []

        def dec(a):
            calls.append(1)
            if len(calls) == 1:
                raise sp.NotReady("loading")
            return ("Jarvis, lights", {})
        p, w, log, notes = _primary(dec)
        self.assertEqual(p.run(self.AUDIO), w.res)
        self.assertEqual(p.latch.failed, "")
        self.assertEqual(p.run(self.AUDIO)[0], "Jarvis, lights")
        self.assertEqual(notes, ["whisper-loading", "parakeet"])
        self.assertEqual((p.decodes, len(w.calls), log), (1, 1, []))

    def test_the_rescue_line_is_rate_limited(self):
        # One line, then at most one per RESCUE_LOG_EVERY_S carrying how
        # many were not logged — every rescue is still counted.
        clock = _Clock(t=1000.0)
        log, notes = [], []
        p = sp.Primary(lambda a: ("", {}), _Whisper(), latch=sp.Latch(),
                       note=notes.append, log=log.append, clock=clock)
        for _ in range(4):
            p.run(self.AUDIO)
        self.assertEqual(len(log), 1)
        self.assertIn("rescue (empty; 1 of 1 parakeet decodes)", log[0])
        clock.t += sp.RESCUE_LOG_EVERY_S
        p.run(self.AUDIO)
        self.assertEqual(len(log), 2)
        self.assertIn("rescue (empty; 5 of 5 parakeet decodes; 3 more not "
                      "logged)", log[1])
        self.assertEqual(p.rescues, 5)
        self.assertEqual(notes, ["parakeet-rescued"] * 5)

    def test_hotwords_ignored_is_logged_once(self):
        p, _w, log, _n = _primary(lambda a: ("Jarvis, x y", {}), hot=True)
        p.run(self.AUDIO)
        p.run(self.AUDIO)
        self.assertEqual(len([ln for ln in log if "STT_HOTWORDS" in ln]), 1)
        p2, _w, log2, _n = _primary(lambda a: ("Jarvis, x y", {}), hot=False)
        p2.run(self.AUDIO)
        self.assertEqual(log2, [])


class PrimaryMusicGateTests(unittest.TestCase):
    """Primary's music-gate hooks (core/music_gate.py, 2026-10-04): 'skip'
    keeps Parakeet's text, 'shadow' rescues and then tells shadow() what
    Whisper made of THIS capture, and a gate that raises RESCUES — a slower
    turn, never a lost one (review 2026-10-04: no test pinned that)."""
    AUDIO = np.zeros(16000, np.float32)

    def _p(self, gate, shadow=None):
        notes, log = [], []
        w = _Whisper()
        p = sp.Primary(lambda a: ("la la la", {}), w, latch=sp.Latch(),
                       rescue=lambda t, a: "no-wake", note=notes.append,
                       log=log.append, gate=gate, shadow=shadow)
        return p, w, notes

    def test_a_raising_gate_rescues(self):
        def boom(text, audio, why):
            raise RuntimeError("meter")
        p, w, notes = self._p(boom)
        self.assertEqual(p.run(self.AUDIO), w.res)
        self.assertEqual((len(w.calls), p.gated), (1, 0))
        self.assertEqual(notes, ["parakeet-rescued"])

    def test_skip_keeps_parakeets_text(self):
        p, w, notes = self._p(lambda t, a, why: "skip")
        self.assertEqual(p.run(self.AUDIO)[0], "la la la")
        self.assertEqual((len(w.calls), p.gated, p.rescues), (0, 1, 0))
        self.assertEqual(notes, ["parakeet-gated"])

    def test_shadow_rescues_and_hands_over_the_capture(self):
        seen = []
        p, w, _notes = self._p(lambda t, a, why: "shadow",
                               shadow=lambda why, res, audio:
                               seen.append((why, res, audio)))
        self.assertEqual(p.run(self.AUDIO), w.res)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][:2], ("no-wake", w.res))
        self.assertIs(seen[0][2], self.AUDIO)

    def test_a_raising_shadow_still_returns_the_rescue(self):
        def boom(why, res, audio):
            raise RuntimeError("count")
        p, w, _notes = self._p(lambda t, a, why: "shadow", shadow=boom)
        self.assertEqual(p.run(self.AUDIO), w.res)


# ── the shadow worker ─────────────────────────────────────────────────────
def _judge(text, conf, peak, ctx):
    return {"gates_passed": bool(text), "wake_prefix": _prefix(text)}


def _shadow(busy=lambda: False, decode=None, maxsize=4, clock=None,
            **kw):
    clock = clock or _Clock()
    log, rows, decoded = [], [], []

    def dec(a):
        decoded.append(a)
        return ("jarvis turn the lights off",
                {"no_speech_prob": 0.0, "avg_logprob": -0.3, "stt_ms": 150,
                 "lock_wait_ms": 4, "tok_lp_mean": -0.02, "tok_lp_min": -0.2,
                 "n_tok": 9})
    sh = sp.Shadow(decode or dec, busy, _judge,
                   lambda r: rows.append(r) or True,
                   latch=sp.Latch(log=log.append), maxsize=maxsize,
                   wait_s=30.0, poll_s=0.25, log=log.append, clock=clock,
                   wait=clock.wait, wall=lambda: 1_000_000.0, **kw)
    return sh, log, rows, decoded, clock


W_CONF = {"no_speech_prob": 0.0, "avg_logprob": -0.25}


class ShadowTests(unittest.TestCase):
    AUDIO = np.full(32000, 0.01, np.float32)

    def _offer(self, sh, text="Jarvis, turn the lights off"):
        return sh.offer(self.AUDIO, text, W_CONF, 1500, 0.02,
                        {"wake_mode": True}, start=False)

    def _take(self, sh):
        return sh._q.get_nowait()

    def test_skips_while_a_turn_is_in_progress_bounded(self):
        sh, log, rows, decoded, clock = _shadow(busy=lambda: True)
        self.assertTrue(self._offer(sh))
        self.assertIsNone(sh.process(self._take(sh)))
        self.assertEqual(decoded, [])
        self.assertEqual([r["dropped"] for r in rows], ["busy"])
        self.assertEqual(sh.dropped_busy, 1)
        # It waited (no sleeping: the fake clock) and gave up at 30 s.
        self.assertGreaterEqual(clock.t - 100.0, 30.0)
        self.assertLess(clock.t - 100.0, 30.0 + 0.25 + 1e-9)
        self.assertTrue(any("dropped" in ln for ln in log))

    def test_runs_once_the_turn_ends(self):
        state = {"n": 0}

        def busy():
            state["n"] += 1
            return state["n"] <= 3          # busy for three polls
        sh, log, rows, decoded, clock = _shadow(busy=busy)
        self._offer(sh)
        row = sh.process(self._take(sh))
        self.assertEqual(len(decoded), 1)
        self.assertEqual(rows, [row])
        self.assertEqual(row["shadow_wait_ms"], 750)
        self.assertEqual(row["whisper"]["text"], "Jarvis, turn the lights off")
        self.assertEqual(row["parakeet"]["text"], "jarvis turn the lights off")
        self.assertTrue(row["same_words"])
        self.assertEqual(row["whisper"]["stt_ms"], 1500)
        self.assertEqual(row["parakeet"]["stt_ms"], 150)
        self.assertEqual(row["parakeet"]["n_tok"], 9)
        self.assertTrue(row["wake_mode"])
        self.assertFalse(row["standby"])
        self.assertEqual(row["speech_s"], 2.0)
        self.assertTrue(row["parakeet"]["gates_passed"])
        self.assertTrue(any("gates whisper=1 parakeet=1" in ln
                            for ln in log))
        json.dumps(row)                        # one JSON line

    def test_the_row_records_the_would_be_rescue_and_primary_cost(self):
        # Primary mode would have rescued (Whisper decodes again): the row
        # says so, and what primary would have cost — Parakeet, plus
        # Whisper's decode on a rescue (R6 review, finding 4).
        seen = []

        def rescue(text, audio, ctx):
            seen.append((text, len(audio), ctx))
            return "no-wake"
        sh, _log, rows, _d, _c = _shadow(rescue=rescue)
        sh.offer(self.AUDIO, "Travis, lights", W_CONF, 1200, 0.02,
                 {"wake_mode": True}, start=False, wait_ms=300)
        row = sh.process(self._take(sh))
        self.assertEqual(seen, [("jarvis turn the lights off", 32000,
                                 {"wake_mode": True})])
        self.assertEqual(row["rescue"], "no-wake")
        self.assertEqual(row["primary_ms"], 150 + 4 + 1200)
        sh, _log, rows, _d, _c = _shadow(rescue=lambda t, a, c: "")
        self._offer(sh)
        row = sh.process(self._take(sh))
        self.assertEqual((row["rescue"], row["primary_ms"]), ("", 154))
        # No rescue callable: unknown, not "no rescue".
        sh, _log, rows, _d, _c = _shadow()
        self._offer(sh)
        self.assertIsNone(sh.process(self._take(sh))["rescue"])

    def test_whispers_lock_wait_is_kept_apart_from_its_decode(self):
        # Whisper's wall time included the wait for _stt_lock behind an
        # ambient decode; Parakeet's was pure decode (R6 review, finding 7).
        sh, _log, rows, _d, _c = _shadow()
        sh.offer(self.AUDIO, "x", W_CONF, 1200, 0.02, {}, start=False,
                 wait_ms=300)
        row = sh.process(self._take(sh))
        self.assertEqual((row["whisper"]["stt_ms"],
                          row["whisper"]["lock_wait_ms"]), (1200, 300))
        self.assertEqual((row["parakeet"]["stt_ms"],
                          row["parakeet"]["lock_wait_ms"]), (150, 4))

    def test_the_live_outcome_is_recorded(self):
        seen = []

        def live(t, ctx):
            seen.append((t, ctx))
            return {"you": True}
        sh, _log, rows, _d, clock = _shadow(live=live)
        self._offer(sh)
        row = sh.process(self._take(sh))
        self.assertEqual(row["live"], {"you": True})
        self.assertEqual(seen, [(100.0, {"wake_mode": True})])

    def test_dropped_captures_leave_a_numbers_only_row(self):
        # The A/B file must show what it is missing: long turns that ran
        # past the wait, and captures a full queue refused (R6 review,
        # finding 8). Numbers only — no text, no audio.
        sh, log, rows, _d, clock = _shadow(busy=lambda: True, maxsize=1)
        self.assertTrue(self._offer(sh))
        self.assertFalse(self._offer(sh))          # full
        self.assertIsNone(sh.process(self._take(sh)))
        self.assertEqual([r.get("dropped") for r in rows], ["full", "busy"])
        for r in rows:
            self.assertEqual(r["speech_s"], 2.0)
            self.assertTrue(r["wake_mode"])
            self.assertEqual(set(r), {"ts", "dropped", "speech_s",
                                      "wake_mode", "standby"})
        self.assertEqual((sh.dropped_full, sh.dropped_busy), (1, 1))

    def test_nothing_is_recorded_while_the_mic_is_muted(self):
        # The owner said "don't listen": no decode, no row — not even a
        # numbers-only one (R6 review, finding 2 of the second review).
        sh, log, rows, decoded, _c = _shadow(muted=lambda: True)
        self._offer(sh)
        self.assertIsNone(sh.process(self._take(sh)))
        self.assertEqual((decoded, rows), ([], []))
        self.assertEqual(sh.dropped_muted, 1)

    def test_a_mute_during_the_decode_writes_nothing(self):
        state = {"n": 0}

        def muted():
            state["n"] += 1
            return state["n"] > 1           # muted once the decode ran
        sh, _log, rows, decoded, _c = _shadow(muted=muted)
        self._offer(sh)
        self.assertIsNone(sh.process(self._take(sh)))
        self.assertEqual((len(decoded), rows), (1, []))

    def test_words_are_kept_only_for_a_line_the_gates_would_pass(self):
        # A line no gate would pass is as likely someone else's as the
        # owner's: its row keeps the numbers and the verdicts, never the
        # words (R6 review: standby / room lines went to the file verbatim).
        def judge(text, conf, peak, ctx):
            return {"gates_passed": ctx.get("pass") == text}
        for ctx, live, kept in (
                ({}, None, False),
                ({"pass": "Jarvis, turn the lights off"}, None, True),
                ({"pass": "jarvis turn the lights off"}, None, True),
                ({}, {"you": True}, True),
                ({}, {"you": False, "woke": True}, True),
                ({}, {"you": False}, False)):
            sh, _log, rows, _d, _c = _shadow(live=lambda t, c, v=live: v)
            sh._judge = judge
            sh.offer(self.AUDIO, "Jarvis, turn the lights off", W_CONF,
                     1500, 0.02, ctx, start=False)
            row = sh.process(self._take(sh))
            case = (ctx, live)
            self.assertEqual(rows, [row], case)
            if kept:
                self.assertEqual(row["words"], "kept", case)
                self.assertEqual(row["whisper"]["text"],
                                 "Jarvis, turn the lights off", case)
            else:
                self.assertEqual(row["words"], "dropped", case)
                self.assertEqual((row["whisper"]["text"],
                                  row["parakeet"]["text"]), ("", ""), case)
                self.assertEqual((row["whisper"]["chars"],
                                  row["parakeet"]["chars"]), (27, 26), case)
                self.assertNotIn("lights", json.dumps(row), case)

    def test_never_prints_either_transcript(self):
        sh, log, rows, _d, _c = _shadow()
        out = io.StringIO()
        with redirect_stdout(out):
            self._offer(sh, text="Jarvis, open the secret project folder")
            sh.process(self._take(sh))
        text = "\n".join(log) + out.getvalue()
        for word in ("secret", "project", "lights", "jarvis", "Jarvis"):
            self.assertNotIn(word, text)
        self.assertEqual(len(rows), 1)
        self.assertTrue(any("[stt-shadow]" in ln for ln in log))

    def test_the_queue_is_bounded_and_never_blocks(self):
        sh, _log, _rows, _d, _c = _shadow(maxsize=2)
        self.assertTrue(self._offer(sh))
        self.assertTrue(self._offer(sh))
        self.assertFalse(self._offer(sh))
        self.assertEqual((sh.pending(), sh.dropped_full, sh.offered),
                         (2, 1, 2))

    def test_the_audio_is_copied(self):
        sh, _log, _rows, decoded, _c = _shadow()
        a = np.full(16000, 0.5, np.float32)
        sh.offer(a, "x", {}, 1, 0.1, {}, start=False)
        a[:] = 0.0
        sh.process(self._take(sh))
        self.assertTrue(np.all(decoded[0] == 0.5))

    def test_a_decode_failure_latches_and_stops_the_offers(self):
        def boom(a):
            raise RuntimeError("no model")
        sh, log, rows, _d, _c = _shadow(decode=boom)
        self._offer(sh)
        self.assertIsNone(sh.process(self._take(sh)))
        self.assertTrue(sh.latch.failed)
        self.assertFalse(self._offer(sh))
        self.assertEqual(rows, [])
        self.assertEqual(len([ln for ln in log if "parakeet off" in ln]), 1)

    def test_the_daemon_processes_offers_in_the_background(self):
        done = threading.Event()
        rows = []

        def write(r):
            rows.append(r)
            done.set()
            return True
        sh = sp.Shadow(lambda a: ("hello there", {"stt_ms": 5}),
                       lambda: False, _judge, write, latch=sp.Latch(),
                       log=lambda s: None)
        self.addCleanup(sh.stop)
        self.assertTrue(sh.offer(self.AUDIO, "hello there", W_CONF, 9, 0.02,
                                 {}))
        self.assertTrue(done.wait(10.0))
        self.assertEqual(rows[0]["parakeet"]["text"], "hello there")
        self.assertEqual(sh._thread.name, "stt-shadow")
        self.assertTrue(sh._thread.daemon)


class AppendJsonlTests(unittest.TestCase):
    def test_appends_lines_and_respects_the_cap(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "sub", "stt_ab.jsonl")
        try:
            self.assertTrue(sp.append_jsonl(path, {"a": 1}))
            self.assertTrue(sp.append_jsonl(path, {"b": "é"}))
            with open(path, encoding="utf-8") as f:
                lines = [json.loads(x) for x in f]
            self.assertEqual(lines, [{"a": 1}, {"b": "é"}])
            self.assertFalse(sp.append_jsonl(path, {"c": 3}, max_bytes=1))
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


# ── import hygiene ────────────────────────────────────────────────────────
class ImportHygieneTests(unittest.TestCase):
    def test_import_and_flags_off_pull_in_nothing_heavy(self):
        # A FRESH interpreter (this one may hold numpy / onnxruntime from
        # other tests): importing the module and resolving both flags at
        # their defaults must import none of onnx_asr, onnxruntime, numpy.
        code = ("import sys, os; sys.path.insert(0, sys.argv[1]); "
                "os.environ.pop('JARVIS_STT_ENGINE', None); "
                "import core.stt_parakeet as sp; "
                "from core import config; "
                "assert sp.engine_setting(config.STT_ENGINE) == 'whisper'; "
                "assert sp.shadow_setting(config.STT_SHADOW) == ''; "
                "sp.map_conf([-0.1]); sp.rescue_reason('', wake_lost=None, "
                "head_speech=None); "
                "print(sorted(m for m in ('onnx_asr', 'onnxruntime', "
                "'numpy') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code, _ROOT],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]",
                         out.stdout)


# ── heavy / local tier: the real model ───────────────────────────────────
def _real_model_dir():
    try:
        from core import config
        d = config.PARAKEET_MODEL_DIR
    except Exception:
        return None
    if os.environ.get("JARVIS_TEST_PARAKEET", "").strip() != "1":
        return None
    if not sp.package_available() or not os.path.isdir(d):
        return None
    return d


def _reference_wav():
    """The owner's consented enrollment clip, named explicitly by
    JARVIS_TEST_PARAKEET_REF for the heavy run (never resolved from a data
    directory here). Read only; never copied, never committed; its
    transcript is never printed."""
    cand = os.environ.get("JARVIS_TEST_PARAKEET_REF", "").strip()
    return cand if cand and os.path.isfile(cand) else None


def _read_16k(path):
    import wave
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != sp.SAMPLE_RATE:                   # band-limited (FFT) resample
        n = int(round(len(a) * sp.SAMPLE_RATE / float(sr)))
        spec = np.fft.rfft(a)
        keep = min(len(spec), n // 2 + 1)
        out = np.zeros(n // 2 + 1, dtype=spec.dtype)
        out[:keep] = spec[:keep]
        a = (np.fft.irfft(out, n) * (n / float(len(a)))).astype(np.float32)
    return a


@unittest.skipUnless(_real_model_dir(),
                     "JARVIS_TEST_PARAKEET=1 + onnx_asr + the model folder: "
                     "heavy local tier only")
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import time
        t0 = time.perf_counter()
        cls.eng = sp.load(_real_model_dir(), threads=8)
        cls.load_ms = (time.perf_counter() - t0) * 1000.0

    def test_cpu_only(self):
        # Every onnxruntime session inside runs on the CPU provider alone.
        seen = []
        for obj in vars(self.eng.asr).values():
            if hasattr(obj, "get_providers"):
                seen.append(obj.get_providers())
        self.assertTrue(seen)
        for provs in seen:
            self.assertEqual(provs, ["CPUExecutionProvider"])

    def test_warm_decode_of_one_second_of_silence_is_under_a_second(self):
        text, conf = sp.transcribe(self.eng, np.zeros(16000, np.float32))
        text, conf = sp.transcribe(self.eng, np.zeros(16000, np.float32))
        self.assertEqual(text, "")
        self.assertLess(conf["stt_ms"], 1000)
        print(f"\n  [parakeet] load {self.load_ms:.0f} ms, warm 1 s silence "
              f"{conf['stt_ms']} ms")

    def test_reference_clip_gives_text(self):
        ref = _reference_wav()
        if not ref:
            self.skipTest("no consented reference clip on this box")
        text, conf = sp.transcribe(self.eng, _read_16k(ref))
        self.assertTrue(text.strip())
        self.assertGreater(conf["n_tok"], 0)
        self.assertLessEqual(conf["avg_logprob"], 0.0)
        # Numbers only: the words are the owner's.
        print(f"\n  [parakeet] reference clip: {len(text)} chars, "
              f"{conf['n_tok']} tokens, {conf['stt_ms']} ms")


if __name__ == "__main__":
    unittest.main()
