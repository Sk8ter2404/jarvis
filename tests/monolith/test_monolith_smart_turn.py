"""Smart Turn end of turn wired into the REAL record_speech (speed plan R7,
2026-10-02 — the SHADOW release).

core/endpointing.py's models and EotDecider are covered CI-light by
tests/test_endpointing.py. This file drives the real capture loop through
tests/test_speculative_stt.py's fake mic stream and its _run_capture — reused,
never re-implemented (the speculative-STT "green by re-implementation" trap)
— with the real SileroStream / SmartTurn classes over FAKE onnxruntime
sessions (test_endpointing's _StreamSession / _TurnSession: p 0.9 for a loud
window, a scripted Smart Turn p). No model file, no real model run.

Pinned here:
  * shadow returns byte-identical audio to off, on a turn it WOULD have cut;
  * one numeric [eot-shadow] line per turn, and eot / st_p / st_n on the
    [turn-timing] line;
  * Silero and Smart Turn hear what Whisper hears: the clip from its first
    pre-roll sample, x the capture's auto-gain;
  * a mid-sentence pause after a would-be end reports resumed=1;
  * mode 'off' never imports a model runtime; no capture loads a model (only
    the boot warmer does);
  * a fault in the decider, Silero or Smart Turn never breaks a capture and
    turns Smart Turn off for the session with one line;
  * in-turn captures (no smart_endpoint; off the main loop's thread) never
    build a decider, and only the two main-loop listens pass it;
  * mode 'on' (not shipped) ends a complete turn early, with eot=st.

Run (full-deps tier):
    python -m unittest tests.monolith.test_monolith_smart_turn
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import os
import re
import threading
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
# Modules, not names: discovery must not collect their TestCases again here.
from tests import test_endpointing as _ept
from tests import test_speculative_stt as _spec

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_CHUNK = _spec._CHUNK                 # 1,024 samples = 64 ms
_PRE = _spec._PRE_BUFFER              # 12 pre-roll chunks join at the trip
_MS = 64
_LOUD = 0.1     # RMS: loud for the fake Silero at ANY gain (|mean| 0.09)
# A complete turn: room tone (fills the pre-roll), 30 voiced chunks, then
# silence well past the 21-chunk break.
_TURN = [0.001] * 20 + [_LOUD] * 30 + [0.0] * 25
# The same words with a mid-sentence pause (8 silent chunks) before 20 more.
_PAUSED = [0.001] * 20 + [_LOUD] * 30 + [0.0] * 8 + [_LOUD] * 20 + [0.0] * 25
# Smart Turn first runs at the 4th silent chunk (SMART_TURN_MIN_SILENCE_S).
_FIRE_MS = (_PRE + 30 + 4) * _MS                       # 2,944
_TURN_MS = (_PRE + 30 + 21) * _MS                      # 4,032
_PAUSED_MS = (_PRE + 30 + 8 + 20 + 21) * _MS           # 5,824
_SHADOW_LINE = re.compile(r"\[eot-shadow\] fire_ms=(?:\d+|-) "
                          r"p=(?:[01]\.\d{3}|-) resumed=[01] actual_ms=\d+")


class _RecTurn(_ept.ep.SmartTurn):
    """SmartTurn that keeps a copy of every clip it is asked about."""

    def __init__(self, heard, *a, **k):
        super().__init__(*a, **k)
        self.heard = heard

    def predict(self, audio, sample_rate=16000):
        import numpy as np
        self.heard.append(np.array(audio, dtype=np.float32, copy=True))
        return super().predict(audio, sample_rate)


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    MODE = "shadow"
    P = 0.99

    def setUp(self):
        super().setUp()
        import numpy as np
        from core import turn_timing as tt
        self.np, self.tt = np, tt
        bc = self.bc
        bc._spec_stt_reset()
        self.addCleanup(bc._spec_stt_reset)
        ep = bc._endpointing
        self.vsess = _ept._StreamSession()
        self.tsess = _ept._TurnSession(ps=self.P)
        self.loads = {"silero": 0, "turn": 0}
        self.silero_load_error = None
        self.heard_by_turn = []
        self._p(bc, "_eot_stream", ep.SileroStream(
            model_path="fake.onnx", session_factory=self._silero_factory))
        self._p(bc, "_eot_turn", _RecTurn(
            self.heard_by_turn, "fake.onnx",
            session_factory=self._turn_factory,
            features=lambda x: np.zeros((80, 800), np.float32),
            clock=self.tsess.clock))
        self._p(bc, "_eot_state", {"ready": False, "off": "", "logged": False})
        self._p(bc, "SMART_TURN_MODE", self.MODE)
        self.timing = tt.TurnTiming(print_fn=lambda line: None)
        self._p(bc, "_turn_timing", self.timing)
        # Every EotDecider a capture builds.
        self.made = []
        made = self.made

        class Spy(ep.EotDecider):
            def __init__(s, *a, **k):
                super().__init__(*a, **k)
                made.append(s)

        self.Spy = Spy
        self._p(ep, "EotDecider", Spy)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _silero_factory(self, _path):
        self.loads["silero"] += 1
        if self.silero_load_error is not None:
            raise self.silero_load_error
        return self.vsess

    def _turn_factory(self, _path):
        self.loads["turn"] += 1
        return self.tsess

    def _warm(self):
        """The real boot warmer, over the fake sessions."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._warm_smart_turn()
        self.assertTrue(self.bc._eot_state["ready"])
        del self.vsess.feeds[:]           # the warm window is no turn's

    def _capture(self, rms_seq, smart=True, **extra):
        """(audio, log) of ONE real record_speech capture of `rms_seq`, as
        the main loop's listen calls it (smart_endpoint=`smart`)."""
        bc = self.bc
        real = bc.record_speech

        def rec(timeout=None, **kw):
            return real(timeout, smart_endpoint=smart, **extra, **kw)

        out = io.StringIO()
        with mock.patch.object(bc, "record_speech", rec), \
                contextlib.redirect_stdout(out):
            audio, _decodes = _spec.SpeculativeRealCaptureLoopTests \
                ._run_capture(self, rms_seq)
        return audio, out.getvalue()

    def _off_audio(self, rms_seq):
        """The same capture with no Smart Turn at all (mode 'off')."""
        with mock.patch.object(self.bc, "SMART_TURN_MODE", "off"):
            audio, log = self._capture(rms_seq)
        self.assertNotIn("[eot", log)
        return audio

    @staticmethod
    def _shadow_lines(log):
        return [ln.strip() for ln in log.splitlines() if "[eot-shadow]" in ln]

    @staticmethod
    def _eot_lines(log):
        return [ln.strip() for ln in log.splitlines() if "[eot]" in ln]

    def _turn_fields(self, since):
        """The [turn-timing] fields of the turn that begins at this capture's
        VAD break, the way _capture_utterance begins it."""
        self.timing.begin_voice(since)
        return self.tt.parse_line(self.timing.emit())


class ShadowCaptureTests(_Base):
    def test_shadow_returns_audio_identical_to_off(self):
        # Mutation-proven (2026-10-02): a shadow mode that breaks where Smart
        # Turn fires (_EotCapture.feed returning True on the fire) turns this
        # red — the shadow audio is cut 1,088 ms short.
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)
        self.assertEqual(self.made, [])
        audio, log = self._capture(_TURN)
        self.assertEqual(len(self.made), 1)
        # It WOULD have cut this turn, so "identical" is not vacuous.
        self.assertEqual(self.made[0].record().fire_ms, _FIRE_MS)
        self.assertEqual(len(audio), len(off))
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(audio.dtype, off.dtype)

    def test_one_numeric_shadow_line_and_the_turn_stats(self):
        self._warm()
        since = self.timing.now()
        audio, log = self._capture(_TURN)
        lines = self._shadow_lines(log)
        self.assertEqual(lines, [f"[eot-shadow] fire_ms={_FIRE_MS} p=0.990 "
                                 f"resumed=0 actual_ms={_TURN_MS}"])
        # actual_ms is the returned clip, pre-roll included.
        self.assertEqual(len(audio) * 1000 // 16000, _TURN_MS)
        d = self._turn_fields(since)
        self.assertEqual((d["eot"], d["st_p"], d["st_n"]), ("rms", "0.99", "1"))
        self.assertEqual(d["vad_break"], "0")
        self.assertEqual(self._eot_lines(log), [])

    def test_the_shadow_line_never_carries_what_was_said(self):
        # The whole line is numbers in a fixed shape: no field can hold text.
        # Whatever the turn did (fired, never fired, Smart Turn failed), the
        # line is a full match of that shape.
        self._warm()
        logs = [self._capture(_TURN)[1], self._capture(_PAUSED)[1]]
        self.tsess.ps = [0.2]
        logs.append(self._capture(_TURN)[1])
        self.tsess.fail = RuntimeError("the owner said: open the pod bay")
        logs.append(self._capture(_TURN)[1])
        lines = [ln for log in logs for ln in self._shadow_lines(log)]
        self.assertEqual(len(lines), 4)
        for ln in lines:
            self.assertRegex(ln, r"^" + _SHADOW_LINE.pattern + r"$")
        self.assertNotIn("pod bay", "".join(lines))

    def test_a_pause_after_a_would_be_end_is_resumed_and_kept(self):
        np = self.np
        self._warm()
        off = self._off_audio(_PAUSED)
        audio, log = self._capture(_PAUSED)
        self.assertTrue(np.array_equal(audio, off))   # the words after it
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms={_FIRE_MS} p=0.990 resumed=1 "
            f"actual_ms={_PAUSED_MS}"])

    def test_below_the_threshold_it_keeps_checking_and_never_fires(self):
        self._warm()
        self.tsess.ps = [0.5]
        since = self.timing.now()
        _audio, log = self._capture(_TURN)
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=0.500 resumed=0 actual_ms={_TURN_MS}"])
        d = self._turn_fields(since)
        # Checks at silent chunks 4, 9, 14 and 19 (five apart).
        self.assertEqual((d["eot"], d["st_p"], d["st_n"]), ("rms", "0.5", "4"))

    def test_silero_and_smart_turn_hear_what_whisper_hears(self):
        # Whisper gets apply_capture_auto_gain(clip, peak). Silero must get
        # the same samples from the clip's FIRST pre-roll sample on, and Smart
        # Turn the clip so far — not the raw, un-gained capture.
        np = self.np
        from core import config as cfg
        for name, val in (("CAPTURE_AUTO_GAIN_ENABLED", True),
                          ("CAPTURE_AUTO_GAIN_TARGET_PEAK", 0.25),
                          ("CAPTURE_AUTO_GAIN_MAX", 10.0),
                          ("CAPTURE_AUTO_GAIN_NOISE_FLOOR", 0.005)):
            self._p(cfg, name, val)
        self._warm()
        audio, _log = self._capture(_TURN)
        whisper, gain = self.bc.apply_capture_auto_gain(
            audio, self.bc._last_recording_peak)
        self.assertGreater(gain, 2.0)          # a real gain, not x1
        silero = np.concatenate(
            [f["input"][:, _ept.ep.CONTEXT:].reshape(-1)
             for f in self.vsess.feeds])
        # Silero is fed up to the would-be end (shadow stops listening there).
        self.assertEqual(len(silero), _FIRE_MS * 16)
        self.assertTrue(np.array_equal(silero, whisper[:len(silero)]))
        self.assertEqual(len(self.heard_by_turn), 1)
        self.assertTrue(np.array_equal(self.heard_by_turn[0],
                                       whisper[:_FIRE_MS * 16]))

    def test_a_capture_cut_at_the_utterance_ceiling_is_eot_max(self):
        self._warm()
        self._p(self.bc, "MAX_RECORDING_SECS", -1.0)
        since = self.timing.now()
        audio, log = self._capture(_TURN)
        n = (_PRE + 1) * _MS                  # pre-roll + the tripping chunk
        self.assertEqual(len(audio) * 1000 // 16000, n)
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={n}"])
        self.assertEqual(self._turn_fields(since)["eot"], "max")

    def test_a_capture_with_no_speech_prints_nothing(self):
        # A listen that times out is no turn: no line, no stats.
        self._warm()
        bc = self.bc
        real = bc.record_speech
        out = io.StringIO()
        with mock.patch.object(
                bc, "record_speech",
                lambda timeout=None, **kw: real(0.0, smart_endpoint=True)), \
                contextlib.redirect_stdout(out):
            audio, _ = _spec.SpeculativeRealCaptureLoopTests._run_capture(
                self, [0.001] * 5, tail_silence=0)
        self.assertIsNone(audio)
        self.assertEqual(self._shadow_lines(out.getvalue()), [])


class ModeOffTests(_Base):
    MODE = "off"

    @contextlib.contextmanager
    def _import_spy(self):
        """Every import STATEMENT of onnxruntime / faster_whisper (even one
        already in sys.modules) while the block runs."""
        tried = []
        real_import = builtins.__import__

        def spy(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name.split(".")[0] in ("onnxruntime",
                                                     "faster_whisper"):
                tried.append(name.split(".")[0])
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.object(builtins, "__import__", spy):
            yield tried

    def test_mode_off_never_imports_a_model_runtime(self):
        bc = self.bc
        ep = bc._endpointing
        np = self.np
        # The REAL loaders (default sessions) over a model path that does not
        # exist, and marked warm: in 'off', the mode alone must keep them
        # untouched.
        self._p(bc, "_eot_stream", ep.SileroStream(model_path="x.onnx"))
        self._p(bc, "_eot_turn", ep.SmartTurn("x.onnx"))
        bc._eot_state["ready"] = True
        with self._import_spy() as tried:
            off, log = self._capture(_TURN)
        self.assertEqual((tried, self.made), ([], []))
        self.assertNotIn("[eot", log)
        self.assertEqual(bc._eot_stream.failed, "")
        # The spy is live: the same capture in shadow does try to load
        # Silero (onnxruntime) — and, Silero being unavailable, the capture
        # is unchanged, ends nothing early and turns Smart Turn off once.
        bc.SMART_TURN_MODE = "shadow"
        with self._import_spy() as tried:
            audio, log = self._capture(_TURN)
        self.assertIn("onnxruntime", tried)
        self.assertNotIn("faster_whisper", tried)   # Smart Turn never asked
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={_TURN_MS}"])
        self.assertEqual(len(self._eot_lines(log)), 1)
        self.assertIn("smart turn off for this session (silero: load failed",
                      log)
        _a, log2 = self._capture(_TURN)
        self.assertEqual(len(self.made), 1)            # none for this one
        self.assertNotIn("[eot", log2)

    def test_mode_off_registers_no_boot_warmer(self):
        bc = self.bc
        reg = []
        self._p(bc, "_boot_warmers", reg)
        for mode, want in (("off", []), ("true", []), ("", []),
                           (" Shadow ", [("smart-turn", bc._warm_smart_turn)]),
                           ("on", [("smart-turn", bc._warm_smart_turn)])):
            del reg[:]
            with mock.patch.object(bc, "SMART_TURN_MODE", mode):
                bc._eot_register_warmer()
            self.assertEqual(reg, want, mode)


@requires_monolith
class ShippedRegistrationTests(MonolithGlobalsTestCase):
    def test_the_import_registered_the_warmer_iff_mode_is_not_off(self):
        bc = self.bc
        entry = ("smart-turn", bc._warm_smart_turn)
        if bc._eot_mode() == "off":
            self.assertNotIn(entry, bc._boot_warmers)
        else:
            self.assertIn(entry, bc._boot_warmers)
        self.assertLessEqual(
            sum(1 for n, _ in bc._boot_warmers if n == "smart-turn"), 1)


class FaultTests(_Base):
    def test_a_decider_fault_never_breaks_a_capture_and_latches_off(self):
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)

        class Boom(self.Spy):
            def update(s, chunk, silence_n):
                if s._chunks >= 20:
                    raise RuntimeError("decider bug")
                return super().update(chunk, silence_n)

        self._p(self.bc._endpointing, "EotDecider", Boom)
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(self._shadow_lines(log), [])
        self.assertEqual(self._eot_lines(log), [
            "[eot] smart turn off for this session (capture hook failed: "
            "RuntimeError: decider bug)"])
        built = len(self.made)
        audio2, log2 = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio2, off))
        self.assertEqual(len(self.made), built)       # no decider any more
        self.assertNotIn("[eot", log2)

    def test_a_decider_that_cannot_be_built_never_breaks_a_capture(self):
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)

        def boom(*a, **k):
            raise TypeError("bad knob")

        self._p(self.bc._endpointing, "EotDecider", boom)
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertIn("smart turn off for this session (capture hook "
                      "failed: TypeError: bad knob)", log)

    def test_silero_failing_mid_session_changes_nothing_and_latches(self):
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)
        self.vsess.fail = RuntimeError("ort")
        since = self.timing.now()
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(self.heard_by_turn, [])      # never asked
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={_TURN_MS}"])
        self.assertEqual(self._eot_lines(log), [
            "[eot] smart turn off for this session (silero: run failed: "
            "RuntimeError: ort)"])
        d = self._turn_fields(since)
        self.assertEqual((d["eot"], d["st_p"], d["st_n"]), ("rms", "-", "0"))
        _a, log2 = self._capture(_TURN)
        self.assertEqual(len(self.made), 1)
        self.assertNotIn("[eot", log2)

    def test_smart_turn_failing_mid_session_changes_nothing_and_latches(self):
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)
        self.tsess.fail = RuntimeError("ort")
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={_TURN_MS}"])
        self.assertEqual(self._eot_lines(log), [
            "[eot] smart turn off for this session (smart turn: run failed: "
            "RuntimeError: ort)"])
        _a, log2 = self._capture(_TURN)
        self.assertEqual(len(self.made), 1)
        self.assertNotIn("[eot", log2)


class WarmerTests(_Base):
    def _run_warmers(self):
        bc = self.bc
        self._p(bc, "_boot_warmers", [])
        self._p(bc, "_boot_warmers_started", [False])
        bc._eot_register_warmer()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            th = bc._run_boot_warmers()
            th.join(10)
        return [ln.strip() for ln in out.getvalue().splitlines()]

    def test_no_capture_loads_a_model_before_the_warmer(self):
        _audio, log = self._capture(_TURN)
        self.assertEqual((self.made, self.loads), ([], {"silero": 0,
                                                        "turn": 0}))
        self.assertEqual(log.count("[eot"), 0)
        lines = self._run_warmers()
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"^\[warm\] smart-turn ok \(\d+ ms\)$")
        self.assertEqual(self.loads, {"silero": 1, "turn": 1})
        self._capture(_TURN)
        self.assertEqual(len(self.made), 1)
        self.assertEqual(self.loads, {"silero": 1, "turn": 1})

    def test_a_silero_load_failure_is_its_one_line_and_skips_smart_turn(self):
        self.silero_load_error = OSError("gone")
        lines = self._run_warmers()
        self.assertEqual(lines, ["[warm] smart-turn failed (RuntimeError: "
                                 "silero: load failed: OSError: gone)"])
        self.assertEqual(self.loads, {"silero": 1, "turn": 0})
        st = self.bc._eot_state
        self.assertFalse(st["ready"])
        self.assertIn("silero: load failed", st["off"])
        _audio, log = self._capture(_TURN)
        self.assertEqual(self.made, [])
        self.assertNotIn("[eot", log)                 # the warm line was it

    def test_a_slow_smart_turn_warm_latches_off(self):
        self.tsess.costs = [0.05, 0.2]
        lines = self._run_warmers()
        self.assertEqual(lines, ["[warm] smart-turn failed (RuntimeError: "
                                 "smart turn: too slow: warm call 200 ms > "
                                 "150 ms)"])
        self.assertFalse(self.bc._eot_state["ready"])
        self._capture(_TURN)
        self.assertEqual(self.made, [])


class InTurnCaptureTests(_Base):
    def test_a_capture_without_smart_endpoint_never_builds_a_decider(self):
        # bambu_setup's spoken access code, draft confirmations: a plain
        # record_speech(timeout=...).
        self._warm()
        audio, log = self._capture(_TURN, smart=False)
        self.assertEqual(self.made, [])
        self.assertEqual(self.vsess.feeds, [])
        self.assertNotIn("[eot", log)
        self.assertEqual(len(audio) * 1000 // 16000, _TURN_MS)

    def test_an_off_thread_capture_never_builds_a_decider(self):
        # A dashboard-run action asking the owner something captures off the
        # main loop's thread (_record_speech_offthread), even if a caller
        # passed smart_endpoint.
        self._warm()
        bc = self.bc
        real = bc.record_speech
        reentry = []
        box = {}

        def rec(timeout=None, **kw):
            if threading.current_thread() is threading.main_thread():
                th = threading.Thread(
                    target=lambda: box.setdefault(
                        "audio", real(timeout, smart_endpoint=True)),
                    name="action-capture")
                th.start()
                th.join(30)
                return box.get("audio")
            reentry.append(dict(kw))     # _record_speech_offthread's call
            return real(timeout, **kw)

        out = io.StringIO()
        with mock.patch.object(bc, "record_speech", rec), \
                contextlib.redirect_stdout(out):
            audio, _ = _spec.SpeculativeRealCaptureLoopTests._run_capture(
                self, _TURN)
        self.assertEqual(reentry, [{"_offthread_claimed": True}])
        self.assertEqual(len(audio) * 1000 // 16000, _TURN_MS)
        self.assertEqual(self.made, [])
        self.assertNotIn("[eot", out.getvalue())
        # And the claimed re-entry ignores the flag itself.
        audio, log = self._capture(_TURN, _offthread_claimed=True)
        self.assertEqual(self.made, [])
        self.assertNotIn("[eot", log)

    def test_only_the_two_main_loop_listens_pass_smart_endpoint(self):
        path = os.path.join(_ROOT, "bobert_companion.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        sites = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and any(k.arg == "smart_endpoint"
                                for k in node.keywords)):
                    callee = getattr(node.func, "id", None) or getattr(
                        node.func, "attr", None)
                    kw = [k for k in node.keywords
                          if k.arg == "smart_endpoint"][0]
                    sites.append((fn.name, callee,
                                  getattr(kw.value, "value", None)))
        # Both listens sit directly in their (top-level) functions.
        self.assertEqual(sorted(set(sites)), [
            ("_capture_utterance", "record_speech", True),
            ("_handle_sleep_standby", "record_speech", True)])
        for sub in ("core", "skills"):
            for dirpath, _dirs, files in os.walk(os.path.join(_ROOT, sub)):
                for name in files:
                    if not name.endswith(".py"):
                        continue
                    with open(os.path.join(dirpath, name),
                              encoding="utf-8", errors="replace") as fh:
                        self.assertNotIn("smart_endpoint", fh.read(), name)


class OnModeTests(_Base):
    MODE = "on"

    def test_on_ends_a_complete_turn_at_the_first_check(self):
        self._warm()
        since = self.timing.now()
        audio, log = self._capture(_TURN)
        self.assertEqual(len(audio) * 1000 // 16000, _FIRE_MS)
        self.assertEqual(self._shadow_lines(log), [])
        d = self._turn_fields(since)
        self.assertEqual((d["eot"], d["st_p"], d["st_n"]), ("st", "0.99", "1"))
        self.assertEqual(d["vad_break"], "0")         # t0 is this break

    def test_on_below_the_threshold_is_the_fixed_wait(self):
        self._warm()
        self.tsess.ps = [0.5]
        since = self.timing.now()
        audio, _log = self._capture(_TURN)
        self.assertEqual(len(audio) * 1000 // 16000, _TURN_MS)
        self.assertEqual(self._turn_fields(since)["eot"], "rms")


if __name__ == "__main__":
    unittest.main()
