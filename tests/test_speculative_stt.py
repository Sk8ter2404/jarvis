"""Tests for speculative transcription (2026-09-06 latency work).

WHAT THIS PROTECTS
------------------
record_speech() must count out SILENCE_SECS (1.4 s = 21 chunks) of dead air
after the owner's last voiced frame before anything else may start. Whisper
then costs another ~1.4 s. Those were strictly sequential and need not be: once
~0.45 s of silence has passed, snapshot the buffer and decode it in the
background while the loop keeps counting.

The whole thing is safe for exactly ONE reason: a snapshot is accepted only
when NO voiced frame arrived after it, so the audio it decoded differs from the
final buffer by trailing SILENCE alone. That rule is what these tests pin. If
``_spec_stt_note_voiced`` ever stops being called from the voiced branch of the
capture loop, a mid-sentence pause would ship a TRUNCATED transcript — the
owner says "set a timer for ten … minutes" and JARVIS acts on "set a timer for
ten". That is the failure mode; the frame-sequence test below is written to
reproduce it.

MODELS vs THE LOOP (2026-09-06 review of these tests, the-invariant lens)
------------------------------------------------------------------------
Every class in this file except SpeculativeRealCaptureLoopTests drives a
hand-written RE-IMPLEMENTATION of the capture loop (_drive, _drive_rms) rather
than record_speech itself. Those are useful, fast unit tests of the decision
helpers, but on their own they validate the author's MODEL of the loop, and the
model is provably not the loop:

  * the models have no pre_ring, so every chunk count they assert runs exactly
    PRE_BUFFER (12) below the real loop's -- the "37" pinned below is 49 in
    record_speech;
  * the models never call _spec_stt_start, so nothing in them can observe WHAT
    AUDIO the worker actually decodes -- only a chunk count. The worker gains
    its snapshot and (since the 2026-09-06 cancellation fix) decodes through
    _transcribe_impl under _stt_lock; a model that stands in for it with
    ``_spec_stt["chunks"] = n`` cannot notice any of that changing;
  * _drive's frames are bare bools, so sub-VAD_THRESHOLD-but-decodable speech
    -- the ENTIRE shipped defect class -- is unrepresentable in it.

SpeculativeRealCaptureLoopTests at the end of this file therefore runs the real
record_speech against a synthetic mic stream and inspects the real float32
buffers. Keep the models, but do not let them be the only coverage again.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import load_monolith, requires_monolith  # noqa: E402


def _silence(n: int = 16000):
    """A dummy snapshot buffer; these tests never decode it for real."""
    import numpy as np
    return np.zeros(n, dtype=np.float32)


@requires_monolith
class SpeculativeSnapshotRuleTests(unittest.TestCase):
    """Drive synthetic frame sequences through the two decision helpers the
    capture loop calls, and assert which snapshot (if any) survives."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self.bc._spec_stt_reset()
        self.addCleanup(self.bc._spec_stt_reset)
        # The SHIPPED default is off (2026-09-06 adversarial review: the
        # snapshot gate is VAD_THRESHOLD, which on this mic sits above most
        # of the speech, so accepted snapshots truncated real utterances).
        # These tests cover the state machine itself, not the rollout
        # decision, so force the flag on and restore whatever ships.
        _shipped = self.bc._SPECULATIVE_STT
        self.bc._SPECULATIVE_STT = True
        self.addCleanup(setattr, self.bc, "_SPECULATIVE_STT", _shipped)

    def _drive(self, frames, spec_at=7, silence_lim=21, min_chunks=6):
        """frames: iterable of bools (True = above VAD_THRESHOLD).
        Returns the snapshot chunk-count that would be accepted, or -1.

        MODEL ONLY -- this is NOT record_speech. It seeds no pre_ring (so its
        counts run PRE_BUFFER=12 below the loop's), it stands in for
        _spec_stt_start instead of calling it (so it never sees the real
        snapshot audio), and its bool frames cannot express speech that sits
        below VAD_THRESHOLD. SpeculativeRealCaptureLoopTests runs the loop."""
        bc = self.bc
        n_chunks = 0
        silence_n = 0
        recording = False
        for voiced in frames:
            if voiced:
                recording = True
                n_chunks += 1
                silence_n = 0
                bc._spec_stt_note_voiced()
            elif recording:
                n_chunks += 1
                silence_n += 1
                if bc._spec_stt_should_snapshot(silence_n, spec_at, n_chunks,
                                                min_chunks):
                    # stand in for _spec_stt_start without touching Whisper
                    bc._spec_stt["chunks"] = n_chunks
                    bc._spec_stt["thread"] = None
                if silence_n >= silence_lim:
                    break
        return bc._spec_stt["chunks"]

    def test_clean_utterance_snapshots_at_the_trip_point(self):
        accepted = self._drive([True] * 30 + [False] * 21)
        self.assertEqual(accepted, 37,
                         "a clean utterance must snapshot exactly at the trip "
                         "point (30 voiced + 7 silent chunks)")

    def test_a_resumed_utterance_discards_the_stale_snapshot(self):
        """THE regression this file exists for: pause, snapshot, resume."""
        accepted = self._drive([True] * 30 + [False] * 8
                               + [True] * 10 + [False] * 21)
        self.assertGreater(accepted, 48,
                           "the snapshot taken before the owner resumed must "
                           "be discarded — accepting it would truncate the "
                           "utterance mid-sentence")
        self.assertEqual(accepted, 55,
                         "the surviving snapshot must be the one taken in the "
                         "SECOND silence run (30+8+10 voiced/silent chunks, "
                         "then 7 more)")

    def test_utterance_that_never_pauses_long_enough_has_no_snapshot(self):
        # Stops recording via silence_lim without ever reaching spec_at?
        # Impossible by construction (spec_at < silence_lim), so instead:
        # too little audio to be worth a decode.
        accepted = self._drive([True] * 2 + [False] * 21, min_chunks=20)
        self.assertEqual(accepted, -1,
                         "a sub-minimum capture must not spend a decode")

    def test_trip_point_is_clamped_below_the_real_end_of_speech(self):
        """A snapshot at or past silence_lim buys nothing, and one at 0 would
        fire on every chunk of a still-live utterance."""
        bc = self.bc
        SR, CHUNK = 16000, 1024
        for silence_secs, spec_secs in ((1.4, 0.45), (0.3, 0.45), (1.4, 5.0),
                                        (1.4, 0.0)):
            silence_lim = int(silence_secs * SR / CHUNK)
            spec_at = max(1, min(silence_lim - 1,
                                 int(spec_secs * SR / CHUNK)))
            self.assertGreaterEqual(spec_at, 1)
            self.assertLess(spec_at, max(silence_lim, 2))
        self.assertLess(bc._SPEC_STT_SILENCE_SECS, bc.SILENCE_SECS,
                        "the shipped trip point must sit inside the shipped "
                        "hangover or the feature is a no-op")

    def test_only_one_decode_in_flight(self):
        bc = self.bc

        class _Alive:
            def is_alive(self):
                return True

        bc._spec_stt["thread"] = _Alive()
        bc._spec_stt["chunks"] = -1
        self.assertFalse(
            bc._spec_stt_should_snapshot(7, 7, 100, 6),
            "a second speculative decode must not queue behind the first — "
            "they serialise on _stt_lock and would delay the real one")


@requires_monolith
class SpeculativeCollectionTests(unittest.TestCase):
    """_transcribe_capture must use the speculative result when and ONLY when
    the capture produced a valid one."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self.bc._spec_stt_reset()
        self.addCleanup(self.bc._spec_stt_reset)
        # The SHIPPED default is off (2026-09-06 adversarial review: the
        # snapshot gate is VAD_THRESHOLD, which on this mic sits above most
        # of the speech, so accepted snapshots truncated real utterances).
        # These tests cover the state machine itself, not the rollout
        # decision, so force the flag on and restore whatever ships.
        _shipped = self.bc._SPECULATIVE_STT
        self.bc._SPECULATIVE_STT = True
        self.addCleanup(setattr, self.bc, "_SPECULATIVE_STT", _shipped)

    class _Done:
        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    def test_uses_the_speculative_result_when_valid(self):
        bc = self.bc
        bc._spec_stt["thread"] = self._Done()
        bc._spec_stt["chunks"] = 42
        bc._spec_stt["result"] = ("speculative text", {"avg_logprob": -0.1})
        calls = []
        real = bc.transcribe
        bc.transcribe = lambda a: (calls.append(a) or ("real text", {}))
        try:
            text, _conf = bc._transcribe_capture(object())
        finally:
            bc.transcribe = real
        self.assertEqual(text, "speculative text")
        self.assertEqual(calls, [], "Whisper must not run twice")

    def test_falls_back_when_the_snapshot_was_invalidated(self):
        bc = self.bc
        bc._spec_stt["thread"] = self._Done()
        bc._spec_stt["chunks"] = -1          # owner resumed speaking
        bc._spec_stt["result"] = ("stale truncated text", {})
        real = bc.transcribe
        bc.transcribe = lambda a: ("real text", {})
        try:
            text, _conf = bc._transcribe_capture(object())
        finally:
            bc.transcribe = real
        self.assertEqual(text, "real text",
                         "an invalidated snapshot must NEVER be spoken — that "
                         "is a truncated utterance")

    def test_falls_back_when_the_worker_failed(self):
        bc = self.bc
        bc._spec_stt["thread"] = self._Done()
        bc._spec_stt["chunks"] = 42
        bc._spec_stt["result"] = None        # worker raised
        real = bc.transcribe
        bc.transcribe = lambda a: ("real text", {})
        try:
            text, _conf = bc._transcribe_capture(object())
        finally:
            bc.transcribe = real
        self.assertEqual(text, "real text")

    def test_reset_orphans_a_late_worker(self):
        """A worker from the PREVIOUS capture must not be able to write its
        result into this one."""
        bc = self.bc
        token = bc._spec_stt["token"]
        bc._spec_stt_reset()
        self.assertNotEqual(bc._spec_stt["token"], token)
        self.assertEqual(bc._spec_stt["chunks"], -1)
        self.assertIsNone(bc._spec_stt["result"])
        self.assertIsNone(bc._spec_stt["thread"])

    def test_disabled_flag_bypasses_everything(self):
        bc = self.bc
        bc._spec_stt["thread"] = self._Done()
        bc._spec_stt["chunks"] = 42
        bc._spec_stt["result"] = ("speculative text", {})
        real_flag = bc._SPECULATIVE_STT
        real = bc.transcribe
        bc._SPECULATIVE_STT = False
        bc.transcribe = lambda a: ("real text", {})
        try:
            self.assertFalse(bc._spec_stt_should_snapshot(7, 7, 100, 6))
            text, _conf = bc._transcribe_capture(object())
        finally:
            bc._SPECULATIVE_STT = real_flag
            bc.transcribe = real
        self.assertEqual(text, "real text")


@requires_monolith
class SpeculativeGateSensitivityTests(unittest.TestCase):
    """THE REVIEW THAT WAS SKIPPED (2026-09-06).

    The two classes above pin the state machine: a snapshot dies the instant a
    frame above VAD_THRESHOLD follows it. That rule is enforced correctly, and
    it is still the WRONG rule.

    VAD_THRESHOLD (0.008) is an ENDPOINT gate. Whisper reads the buffer AFTER
    _process_capture_chunk's AGC and apply_capture_auto_gain, which multiplies
    a quiet capture by up to CAPTURE_AUTO_GAIN_MAX (10x) toward
    CAPTURE_AUTO_GAIN_TARGET_PEAK (0.25) and treats anything above
    CAPTURE_AUTO_GAIN_NOISE_FLOOR (0.005) as signal worth amplifying. So the
    decoder is far more sensitive than the gate that guards the snapshot.

    On this machine that gap is not theoretical. 459 real `[vad] peak RMS=`
    lines in logs/ put the WHOLE-UTTERANCE peak at p25 0.0083, p50 0.0087,
    p75 0.0096 -- the median utterance peaks 9 % above the gate, so nearly
    every chunk of a normal sentence sits BELOW it. Those chunks take the
    `else` branch of the capture loop: they are appended to `chunks`, they
    increment `silence_n`, and they never call _spec_stt_note_voiced. The
    invariant therefore HOLDS while the transcript is truncated -- the
    accepted snapshot differs from the final buffer by trailing SPEECH.

    These tests reproduce that with the real module constants, and pin the
    default-off decision that is the only thing preventing it in production.
    """

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self.bc._spec_stt_reset()
        self.addCleanup(self.bc._spec_stt_reset)

    # A median-level "turn off the lights in the kitchen": 3 chunks clear the
    # 0.008 gate, the rest of the WORDS sit under it (raw RMS 0.003-0.0072,
    # i.e. around and above the 0.005 floor the auto-gain still amplifies and
    # Whisper still decodes), then real room silence. (rms, carries_speech).
    _MEDIAN_CAPTURE = (
        [(0.0087, True), (0.0084, True), (0.0082, True)]         # "turn off"
        + [(0.0072, True), (0.0065, True), (0.0058, True),
           (0.0051, True), (0.0047, True), (0.0039, True),
           (0.0035, True)]                                       # "the lights"
        + [(0.0038, True), (0.0042, True), (0.0049, True),
           (0.0044, True), (0.0036, True), (0.0031, True),
           (0.0033, True), (0.0040, True), (0.0037, True)]       # "in the kitchen"
        + [(0.0004, False)] * 24                                 # room silence
    )

    def _drive_rms(self, frames, spec_at=7, silence_lim=21, min_chunks=6):
        """Replay `frames` through the SAME branch structure record_speech
        uses, calling the real helpers. Returns
        (accepted_snapshot_chunks, chunks_in_final_buffer,
         buffer_position_of_last_speech_chunk)."""
        bc = self.bc
        vad = bc.VAD_THRESHOLD
        n_chunks = 0
        silence_n = 0
        recording = False
        last_speech_at = 0
        for rms, is_speech in frames:
            if rms > vad:
                recording = True
                n_chunks += 1
                silence_n = 0
                bc._spec_stt_note_voiced()
            elif recording:
                n_chunks += 1
                silence_n += 1
                if bc._spec_stt_should_snapshot(silence_n, spec_at, n_chunks,
                                                min_chunks):
                    # stand in for _spec_stt_start without touching Whisper
                    bc._spec_stt["chunks"] = n_chunks
                    bc._spec_stt["thread"] = None
            else:
                continue  # pre-ring / not recording yet
            if is_speech:
                last_speech_at = n_chunks
            if silence_n >= silence_lim:
                break
        return bc._spec_stt["chunks"], n_chunks, last_speech_at

    def test_the_gate_is_coarser_than_the_decoder(self):
        """The premise of the defect, straight from the real constants: a
        frame can be inaudible to the snapshot gate and loud to Whisper."""
        bc = self.bc
        from core import config as _cfg
        floor = float(getattr(_cfg, "CAPTURE_AUTO_GAIN_NOISE_FLOOR", 0.005))
        max_gain = float(getattr(_cfg, "CAPTURE_AUTO_GAIN_MAX", 10.0))
        # There is a whole band of real speech the gate calls silence.
        self.assertLess(floor, bc.VAD_THRESHOLD,
                        "auto-gain amplifies below VAD_THRESHOLD, so the gate "
                        "cannot be used as a speech detector")
        # And the gain stage makes that band plainly audible to the decoder.
        self.assertGreaterEqual(max_gain * 0.0035, bc.VAD_THRESHOLD * 2,
                                "a 0.0035 chunk is amplified to well above "
                                "the gate before Whisper ever sees it")

    def test_median_capture_snapshot_truncates_real_speech(self):
        """THE DEFECT. Force the state machine on and replay the median
        capture: the invariant holds (no frame above VAD_THRESHOLD follows the
        snapshot, so it is ACCEPTED) and the accepted audio is still a
        truncated prefix that drops real words."""
        bc = self.bc
        shipped = bc._SPECULATIVE_STT
        bc._SPECULATIVE_STT = True
        self.addCleanup(setattr, bc, "_SPECULATIVE_STT", shipped)

        accepted, total, last_speech = self._drive_rms(self._MEDIAN_CAPTURE)

        # The invariant the feature shipped on is NOT violated ...
        self.assertGreaterEqual(
            accepted, 0,
            "the snapshot survived: no frame above VAD_THRESHOLD followed it")
        # ... and the transcript is truncated anyway.
        self.assertLess(
            accepted, last_speech,
            f"accepted snapshot is {accepted} chunks but speech runs to chunk "
            f"{last_speech} of {total} - the snapshot drops real words, which "
            "is exactly what the VAD_THRESHOLD invariant fails to catch")
        # Concretely: over half a second of words thrown away.
        dropped_secs = (last_speech - accepted) * 1024 / bc.SAMPLE_RATE
        self.assertGreater(dropped_secs, 0.5,
                           f"only {dropped_secs:.2f}s dropped - the fixture "
                           "no longer reproduces the defect")

    def test_shipped_default_takes_no_snapshot_at_all(self):
        """THE FIX. With what actually ships, the median capture produces no
        snapshot, so _transcribe_capture decodes the full buffer exactly as it
        did before the latency work.

        If someone re-enables speculative STT by default while the snapshot
        gate is still VAD_THRESHOLD, this fails."""
        accepted, _total, _last = self._drive_rms(self._MEDIAN_CAPTURE)
        self.assertEqual(
            accepted, -1,
            "a snapshot was taken on a median-level capture with the shipped "
            "settings - speculative STT must stay off until the snapshot gate "
            "is at least as sensitive as the decoder (the auto-gain noise "
            "floor or a fraction of peak_rms, not VAD_THRESHOLD)")

    def test_env_default_is_off(self):
        """The rollout decision itself, pinned at the source of truth."""
        bc = self.bc
        if os.environ.get("JARVIS_SPECULATIVE_STT") is not None:
            self.skipTest("JARVIS_SPECULATIVE_STT is set in this environment")
        self.assertFalse(
            bc._SPECULATIVE_STT,
            "JARVIS_SPECULATIVE_STT must default to off: the snapshot gate "
            "(VAD_THRESHOLD 0.008) sits above most of a normal utterance on "
            "this mic, so accepted snapshots truncate real commands silently")


@requires_monolith
class SpeculativeCancellationTests(unittest.TestCase):
    """An invalidated snapshot must not go on to burn a Whisper pass.

    THE REGRESSION (2026-09-06 review, the-invariant lens). The owner pauses
    ~0.5 s mid-sentence, a snapshot fires, then he resumes.
    ``_spec_stt_note_voiced()`` sets ``chunks = -1`` so the truncated prefix is
    correctly never spoken — but nothing ever told the WORKER. It ran on,
    holding ``_stt_lock`` across a full decode, while ``_transcribe_capture``
    (which skips its join entirely once ``chunks < 0`` and goes straight to
    ``return transcribe(audio)``) queued the REAL decode behind it. Net effect
    on exactly the mid-sentence-pause turn: one wasted Whisper pass, and up to
    that pass's whole duration (~1.4 s p50 on this machine) ADDED to the turn.
    A latency optimisation that made one class of turn slower.

    WHY THE NON-BLOCKING-ACQUIRE FIX DOES NOT COVER THIS. That fix makes the
    worker fold when the model is ALREADY busy. Here the model is free, the
    worker wins the lock, and only THEN does the owner resume — so these tests
    deliberately leave ``_stt_lock`` unheld. A green run here means the worker
    re-checked the cancellation generation on the far side of the lock, not
    that it happened to fold.

    WHAT THIS CANNOT COVER: a resume that lands after the decode is already
    inside faster-whisper. Nothing can interrupt that; the turn pays for it.
    That residual is a standing reason _SPECULATIVE_STT ships off.
    """

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        bc = self.bc
        bc._spec_stt_reset()
        self.addCleanup(bc._spec_stt_reset)
        _shipped = bc._SPECULATIVE_STT
        bc._SPECULATIVE_STT = True
        self.addCleanup(setattr, bc, "_SPECULATIVE_STT", _shipped)
        # Fake the DECODE, keep the REAL lock: the worker and transcribe() both
        # funnel through _transcribe_impl, so lock contention stays genuine
        # while no Whisper model is ever loaded.
        self.decodes = []
        # Open by default; a test that needs a decode to STALL clears it, which
        # is how "the worker stopped" is told apart from "the worker finished".
        self.decode_gate = threading.Event()
        self.decode_gate.set()
        self.addCleanup(self.decode_gate.set)
        _real_impl = bc._transcribe_impl
        self.addCleanup(setattr, bc, "_transcribe_impl", _real_impl)
        bc._transcribe_impl = self._fake_impl
        # A barrier INSIDE the worker, before it touches the lock. This is what
        # makes the race deterministic: the test opens the resume window at a
        # known point instead of sleeping and hoping.
        self.in_worker = threading.Event()
        self.release = threading.Event()
        _real_gain = bc.apply_capture_auto_gain
        self.addCleanup(setattr, bc, "apply_capture_auto_gain", _real_gain)
        bc.apply_capture_auto_gain = self._fake_gain
        self.addCleanup(self.release.set)

    def _fake_impl(self, audio):
        self.decodes.append(len(audio))
        self.decode_gate.wait(10)
        return (f"decode of {len(audio)}", {"avg_logprob": -0.1})

    def _fake_gain(self, audio, peak):
        self.in_worker.set()
        self.release.wait(10)
        return audio, 1.0

    def _snapshot(self, n=16000):
        import numpy as np
        return np.zeros(n, dtype=np.float32)

    def _start_and_pause_worker(self):
        """Launch the speculative worker and hold it just before the lock."""
        self.bc._spec_stt_start(self._snapshot(), 0.02, 30)
        self.assertTrue(self.in_worker.wait(10), "worker never started")
        self.assertFalse(self.bc._stt_lock.acquire(blocking=False)
                         and self.bc._stt_lock.release(),
                         "sanity: the model must be FREE here, so a green "
                         "result cannot come from the busy-fold path")
        return self.bc._spec_stt["thread"]

    def test_invalidated_snapshot_does_not_burn_a_decode(self):
        bc = self.bc
        t = self._start_and_pause_worker()
        bc._spec_stt_note_voiced()          # <- the owner resumed speaking
        self.release.set()
        t.join(10)
        self.assertFalse(t.is_alive())
        self.assertEqual(
            self.decodes, [],
            "the worker decoded a snapshot the owner had already invalidated: "
            "that pass can never be used, and while it runs it holds "
            "_stt_lock, so the real end-of-capture decode queues behind it")

    def test_real_decode_is_not_charged_a_second_pass(self):
        """One turn, one Whisper pass — on the FULL buffer, not the prefix."""
        bc = self.bc
        t = self._start_and_pause_worker()
        bc._spec_stt_note_voiced()
        self.release.set()
        t.join(10)
        text, _conf = bc._transcribe_capture(self._snapshot(32000))
        self.assertEqual(text, "decode of 32000",
                         "the spoken transcript must come from the full "
                         "buffer, never the invalidated prefix")
        self.assertEqual(self.decodes, [32000],
                         "a resumed utterance must cost ONE decode, not two")

    def test_reset_cancels_the_previous_captures_worker(self):
        """A worker orphaned by the NEXT capture must not decode either — its
        result is already unusable (the token moved), so every second it holds
        _stt_lock is stolen from the new capture."""
        bc = self.bc
        t = self._start_and_pause_worker()
        bc._spec_stt_reset()
        self.release.set()
        t.join(10)
        self.assertEqual(self.decodes, [])

    def test_cancelled_worker_frees_the_next_snapshot(self):
        """The feature's own docstring promises 'the snapshot is discarded and
        a fresh one is taken at the next pause'. That is only true if the
        cancelled worker actually stops: _spec_stt_should_snapshot refuses to
        fire while it is alive, so a wasted decode silently disables
        speculation for the rest of the turn."""
        bc = self.bc
        # A decode that STARTS here never finishes, so the thread staying alive
        # is proof it committed to the model rather than standing down.
        self.decode_gate.clear()
        t = self._start_and_pause_worker()
        bc._spec_stt_note_voiced()
        self.release.set()
        t.join(3)
        self.assertFalse(t.is_alive(),
                         "a cancelled worker must stand down, not commit to a "
                         "decode nobody can use")
        self.assertTrue(
            bc._spec_stt_should_snapshot(7, 7, 100, 6),
            "after the stale worker was cancelled, the second pause must be "
            "able to snapshot again")


# ── the real capture loop ────────────────────────────────────────────────────
# record_speech's own constants, restated so a change on either side shows up
# as a failing test rather than as a silently-diverging copy.
_CHUNK = 1024
_PRE_BUFFER = 12
_SPEC_AT = 7          # int(_SPEC_STT_SILENCE_SECS 0.45 * 16000 / 1024)


def _mic_frame(rms):
    """One _CHUNK-long float32 mic frame whose RMS is exactly `rms`.

    A 220 Hz sine of amplitude rms*sqrt(2) has RMS == rms, so a test can place
    a frame precisely above or below VAD_THRESHOLD (0.008) or the auto-gain
    noise floor (0.005) and mean it. (numpy is imported lazily: it is a
    monolith heavy dep and absent on the light-deps CI runner, where every
    class here is @requires_monolith-skipped.)"""
    import numpy as np
    if rms <= 0:
        return np.zeros(_CHUNK, dtype=np.float32)
    t = np.arange(_CHUNK, dtype=np.float32) / 16000.0
    return (np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
            * np.float32(rms * np.sqrt(2.0)))


class _FakeMicStream:
    """Stands in for sd.InputStream.

    Delivers a fixed frame list into record_speech's OWN audio callback the
    moment the loop start()s it, so the loop drains an already-full queue: no
    real device, no sleeps, no timing race, identical every run."""

    def __init__(self, frames, callback, **_kw):
        self._frames = frames
        self._cb = callback

    def start(self):
        for f in self._frames:
            self._cb(f.reshape(-1, 1), len(f), None, None)

    def stop(self):
        pass

    def close(self):
        pass


@requires_monolith
class SpeculativeRealCaptureLoopTests(unittest.TestCase):
    """Drives the REAL record_speech, not a model of it.

    WHY THIS CLASS EXISTS. Everything above re-implements the capture loop, so
    nothing above can observe the audio the speculative worker actually
    decodes, the pre_ring's contribution to the snapshot, the auto-gain the
    worker applies, or the placement of _spec_stt_note_voiced inside the real
    branch structure. All four are load-bearing for the feature's safety
    claim, and all four were unpinned while the suite was green.

    WHAT IS STUBBED, AND WHY IT IS STILL THE REAL LOOP: only the hardware and
    UI edges — the mic stream, device resolution, stream teardown, face
    tracking, the state/HUD calls, the silent-mic reporter, and
    _transcribe_impl (so no Whisper model is ever loaded). The VAD branch, the
    pre-ring, the silence counter, the snapshot decision, _spec_stt_start, the
    cancellation generation and apply_capture_auto_gain all run for real.
    """

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self.bc._spec_stt_reset()
        self.addCleanup(self.bc._spec_stt_reset)

    def _force_on(self):
        shipped = self.bc._SPECULATIVE_STT
        self.addCleanup(setattr, self.bc, "_SPECULATIVE_STT", shipped)
        self.bc._SPECULATIVE_STT = True

    def _run_capture(self, rms_seq, tail_silence=40):
        """Feed `rms_seq` (per-chunk RMS floats) through record_speech.

        Returns (final_audio, decodes) where `decodes` holds the float32
        buffers the speculative worker actually handed to the decoder, in
        order."""
        import numpy as np
        bc = self.bc
        frames = [_mic_frame(r) for r in rms_seq] + [_mic_frame(0.0)] * tail_silence

        class _SDShim:
            PortAudioError = bc.sd.PortAudioError

            @staticmethod
            def InputStream(**kw):
                return _FakeMicStream(frames, **kw)

        decodes = []
        lock = threading.Lock()

        def _fake_impl(audio):
            with lock:
                decodes.append(np.asarray(audio).copy())
            return ("stub transcript", {"avg_logprob": -0.1})

        for name, val in (
            ("_mic_input_disabled", lambda: False),   # staging has no mic
            ("sd", _SDShim),
            ("get_input_device", lambda: None),
            ("_safe_close_stream", lambda _s: None),
            ("pause_face_tracking", lambda *a, **k: None),
            ("set_state", lambda *a, **k: None),
            ("_report_silent_mic", lambda *a, **k: None),
            ("_transcribe_impl", _fake_impl),
        ):
            self.addCleanup(setattr, bc, name, getattr(bc, name))
            setattr(bc, name, val)
        # Pass-through _process_capture_chunk: noisereduce's spectral gating
        # costs 1-2 s of CPU per chunk and would make this class take minutes.
        # The snapshot is a slice of the SAME `chunks` list the final buffer is
        # built from, so the prefix relations asserted below hold either way.
        _prev = bc._audio_master_enabled[0]
        self.addCleanup(bc._audio_master_enabled.__setitem__, 0, _prev)
        bc._audio_master_enabled[0] = False
        # record_speech writes real module globals on its way through, and this
        # class does NOT inherit MonolithGlobalsTestCase. Put them back, or a
        # later test reads a peak/latch this capture left behind and passes (or
        # fails) for a reason that has nothing to do with its own code —
        # tests/monolith/test_monolith_sec7.py asserts on _last_recording_peak.
        self.addCleanup(setattr, bc, "_last_recording_peak",
                        bc._last_recording_peak)
        for _cell in ("_last_mic_hud_write", "_silent_mic_warned",
                      "_silent_mic_warned_at", "_silent_mic_warned_device"):
            _c = getattr(bc, _cell, None)
            if isinstance(_c, list) and _c:
                self.addCleanup(_c.__setitem__, 0, _c[0])

        audio = bc.record_speech(timeout=None)
        t = bc._spec_stt.get("thread")
        if t is not None:
            t.join(timeout=20)
            self.assertFalse(t.is_alive(),
                             "the speculative worker never finished")
        return audio, decodes

    @staticmethod
    def _n_chunks(buf):
        return 0 if buf is None else len(buf) // _CHUNK

    # ── the defect the models could not express ──────────────────────────────
    def test_an_accepted_snapshot_can_drop_real_speech(self):
        """THE defect, reproduced through the real loop.

        The invariant ("no frame above VAD_THRESHOLD followed the snapshot") is
        SATISFIED here — the snapshot is accepted — and it still discards ~0.6 s
        of audio well above the auto-gain noise floor, i.e. speech the full
        buffer decodes and the snapshot never sees. The safety claim is that
        the two differ by trailing SILENCE; this shows they do not.

        This is why the feature ships OFF. Re-enabling it without moving the
        snapshot gate below VAD_THRESHOLD fails this test."""
        import numpy as np
        bc = self.bc
        self._force_on()
        # "turn off the lights | in the kitchen": a normal sentence whose tail
        # is quieter than its head and falls under the 0.008 endpoint gate.
        audio, decodes = self._run_capture(
            [0.0010] * 20       # room tone (fills the pre-ring)
            + [0.0300] * 14     # loud head, clearly above VAD_THRESHOLD
            + [0.0062] * 16     # REAL SPEECH, under the gate (p25-p50 here)
            + [0.0005] * 25)    # actual silence
        self.assertTrue(decodes, "the speculative worker never ran")
        snap = decodes[0]
        self.assertGreaterEqual(
            bc._spec_stt["chunks"], 0,
            "this capture must ACCEPT its snapshot — the whole point is that "
            "the invariant holds and the transcript is truncated anyway")
        # The worker gains its snapshot, so compare lengths, not samples.
        dropped = audio[self._n_chunks(snap) * _CHUNK:]
        floor = 0.005   # core.config CAPTURE_AUTO_GAIN_NOISE_FLOOR
        lost = sum(
            1 for i in range(len(dropped) // _CHUNK)
            if float(np.sqrt(np.mean(
                dropped[i * _CHUNK:(i + 1) * _CHUNK] ** 2))) > floor)
        self.assertGreaterEqual(
            lost, 5,
            "an ACCEPTED snapshot dropped %d chunks of above-noise-floor "
            "audio — it differs from the final buffer by SPEECH, not by "
            "trailing silence, so the feature's safety argument does not hold"
            % lost)

    def test_shipped_default_never_starts_a_speculative_decode(self):
        """The rollout guard, through the loop: with the SHIPPED flag (not a
        forced one) a normal capture must spend no GPU and leave no worker."""
        bc = self.bc
        audio, decodes = self._run_capture(
            [0.001] * 20 + [0.050] * 30 + [0.0] * 25)
        self.assertFalse(
            bc._SPECULATIVE_STT,
            "speculative STT is shipping ON — if that is deliberate, "
            "test_an_accepted_snapshot_can_drop_real_speech must first be "
            "made to pass for the right reason")
        self.assertEqual(decodes, [], "the disabled feature still decoded")
        self.assertIsNone(bc._spec_stt["thread"])
        self.assertEqual(bc._spec_stt["chunks"], -1)
        self.assertGreater(self._n_chunks(audio), 0)

    def test_the_real_loop_counts_the_pre_ring(self):
        """Pins the count the MODELS above get wrong: record_speech seeds
        `chunks` with up to PRE_BUFFER pre-roll frames at the VAD trip, so a
        30-voiced-chunk utterance snapshots at 49 chunks, not the 37
        test_clean_utterance_snapshots_at_the_trip_point reports."""
        bc = self.bc
        self._force_on()
        _audio, decodes = self._run_capture(
            [0.001] * 20 + [0.050] * 30 + [0.0] * 25)
        self.assertEqual(bc._spec_stt["chunks"], _PRE_BUFFER + 30 + _SPEC_AT)
        self.assertEqual(self._n_chunks(decodes[0]),
                         _PRE_BUFFER + 30 + _SPEC_AT)

    def test_worker_decodes_exactly_the_gained_prefix_of_the_capture(self):
        """_transcribe_capture's docstring promises the two paths "feed
        faster-whisper identically-scaled samples". That is only true because
        any frame loud enough to raise peak_rms is above VAD_THRESHOLD and so
        invalidates the snapshot — peak_rms is frozen once an accepted snapshot
        exists. Pin it end to end: the worker's buffer must be byte-identical
        to the prefix of the auto-gained final buffer. If the gain stages ever
        drift apart, the "same transcript" claim is void."""
        import numpy as np
        bc = self.bc
        self._force_on()
        audio, decodes = self._run_capture(
            [0.001] * 20 + [0.050] * 30 + [0.0] * 25)
        self.assertEqual(len(decodes), 1)
        snap = decodes[0]
        gained, _g = bc.apply_capture_auto_gain(audio, bc._last_recording_peak)
        self.assertLess(len(snap), len(gained))
        self.assertTrue(
            np.array_equal(snap, gained[:len(snap)]),
            "the speculative worker decoded audio that is not the identically "
            "gained prefix of the capture")

    def test_a_resumed_utterance_discards_the_stale_snapshot(self):
        """Pause, snapshot, resume — through the real loop. Whatever the
        worker did, the snapshot standing at the end must be the one taken
        AFTER the owner stopped for good, or JARVIS acts on "set a timer for
        ten" instead of "...ten minutes"."""
        bc = self.bc
        self._force_on()
        audio, decodes = self._run_capture(
            [0.001] * 20 + [0.050] * 30 + [0.0] * 8
            + [0.050] * 10 + [0.0] * 25)
        accepted = bc._spec_stt["chunks"]
        self.assertEqual(accepted, _PRE_BUFFER + 30 + 8 + 10 + _SPEC_AT,
                         "the accepted snapshot must be the SECOND one")
        self.assertLess(accepted, self._n_chunks(audio))
        # The stale prefix must never be what stands at the end.
        self.assertNotEqual(accepted, _PRE_BUFFER + 30 + _SPEC_AT,
                            "the snapshot taken before the owner resumed was "
                            "accepted — that is a truncated utterance")
        if decodes:
            self.assertEqual(self._n_chunks(decodes[-1]), accepted)

    def test_shipped_min_chunk_gate_does_not_stop_a_short_utterance(self):
        """Honesty pin. test_utterance_that_never_pauses_long_enough_has_no_
        snapshot asserts "a sub-minimum capture must not spend a decode", but
        only under a SYNTHETIC min_chunks=20. With the SHIPPED gate (6 chunks)
        the pre-ring alone clears it, so a 64 ms blip does spend a full
        speculative decode. Recorded here so that model claim is never read as
        a statement about production."""
        bc = self.bc
        self._force_on()
        _audio, decodes = self._run_capture(
            [0.001] * 20 + [0.050] * 1 + [0.0] * 25)
        self.assertEqual(len(decodes), 1)
        self.assertEqual(bc._spec_stt["chunks"], _PRE_BUFFER + 1 + _SPEC_AT)

    def test_note_voiced_really_is_wired_into_the_loops_voiced_branch(self):
        """The models call _spec_stt_note_voiced themselves, so they stay
        green even if the real loop stops calling it — the exact regression the
        file's own docstring says it exists to catch. Prove it from the loop:
        with the helper neutered, a resumed utterance keeps the STALE
        snapshot."""
        bc = self.bc
        self._force_on()
        self.addCleanup(setattr, bc, "_spec_stt_note_voiced",
                        bc._spec_stt_note_voiced)
        bc._spec_stt_note_voiced = lambda: None      # simulate the regression
        _audio, _decodes = self._run_capture(
            [0.001] * 20 + [0.050] * 30 + [0.0] * 8
            + [0.050] * 10 + [0.0] * 25)
        self.assertEqual(
            bc._spec_stt["chunks"], _PRE_BUFFER + 30 + _SPEC_AT,
            "with note_voiced neutered the loop must strand the stale "
            "snapshot; if it does not, the call is no longer what protects "
            "the invariant and this suite is testing nothing")


@requires_monolith
class SpeculativeLockContentionTests(unittest.TestCase):
    """The 2026-09-06 review defect: the join cap bounded NOTHING.

    transcribe() holds _stt_lock for its whole call, and the speculative worker
    calls transcribe(). So a worker that is still running is INSIDE the lock the
    voice thread's own fallback needs. Giving up on the join therefore could not
    free the voice thread — it queued a SECOND decode behind the very one it had
    just abandoned, and the turn paid both. Measured on these helpers with a
    saturated-GPU decode cost of 12 s: 12,000 ms with the feature off, 23,700 ms
    with the shipped code, 11,701 ms after the fix.

    These tests use the REAL threads and the REAL _stt_lock. The tests above
    stub bobert_companion.transcribe outright, which is exactly why the whole
    suite stayed green while this was live.
    """

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        bc = self.bc
        bc._spec_stt_reset()
        self.addCleanup(bc._spec_stt_reset)
        _shipped = bc._SPECULATIVE_STT
        bc._SPECULATIVE_STT = True
        self.addCleanup(setattr, bc, "_SPECULATIVE_STT", _shipped)
        # Short caps so the test is fast; the behaviour under test is the same.
        for name, value in (("_SPEC_STT_WAIT", 0.2),
                            ("_SPEC_STT_HARD_WAIT", 10.0)):
            self.addCleanup(setattr, bc, name, getattr(bc, name))
            setattr(bc, name, value)

    def test_slow_decode_is_awaited_not_duplicated(self):
        """A worker still running when the cap expires must be WAITED for, not
        abandoned in favour of a second decode queued behind it."""
        bc = self.bc
        finished = threading.Event()

        def worker():
            with bc._stt_lock:                 # what transcribe() would hold
                time.sleep(1.0)
                bc._spec_stt["result"] = ("speculative text", {})
            finished.set()

        t = threading.Thread(target=worker, daemon=True, name="spec-stt")
        bc._spec_stt["chunks"] = 100
        bc._spec_stt["thread"] = t
        t.start()
        self.addCleanup(finished.wait, 10)

        calls = []
        real = bc.transcribe
        bc.transcribe = lambda a: (calls.append(a) or ("real text", {}))
        self.addCleanup(setattr, bc, "transcribe", real)

        text, _conf = bc._transcribe_capture(object())

        self.assertEqual(
            text, "speculative text",
            "the decode already in flight must be collected — abandoning it "
            "cannot free the voice thread, it only adds a second decode")
        self.assertEqual(
            calls, [],
            "a redundant decode was queued behind the worker we gave up on; "
            "that is the defect — the turn pays for BOTH")

    def test_worker_folds_instead_of_queueing_on_a_busy_model(self):
        """A speculative decode that cannot have the model immediately must
        fold. Queueing puts it AHEAD of the voice thread's real transcription,
        so an utterance the owner resumed (snapshot invalidated, result useless)
        would still wait out a full throwaway decode."""
        bc = self.bc
        impl_calls = []
        real_impl = bc._transcribe_impl
        real_gain = bc.apply_capture_auto_gain
        bc._transcribe_impl = lambda a: (impl_calls.append(a)
                                         or ("speculative text", {}))
        bc.apply_capture_auto_gain = lambda s, p: (s, 1.0)
        self.addCleanup(setattr, bc, "_transcribe_impl", real_impl)
        self.addCleanup(setattr, bc, "apply_capture_auto_gain", real_gain)

        # Someone else (an ambient decode) already owns the model.
        held = threading.Event()
        release = threading.Event()

        def holder():
            with bc._stt_lock:
                held.set()
                release.wait(10)

        h = threading.Thread(target=holder, daemon=True)
        h.start()
        self.addCleanup(h.join, 10)
        self.addCleanup(release.set)
        self.assertTrue(held.wait(5), "holder never took the lock")

        bc._spec_stt_start(_silence(), 0.01, 100)
        t = bc._spec_stt["thread"]
        t.join(timeout=5)

        self.assertFalse(
            t.is_alive(),
            "the speculative worker QUEUED on _stt_lock — it must fold when "
            "the model is busy, or it delays the real transcription behind it")
        self.assertEqual(impl_calls, [], "a folded bet must not decode")
        self.assertIsNone(bc._spec_stt["result"])

    def test_folded_worker_falls_back_normally(self):
        """After a fold, the capture path must transcribe as if the feature
        were off — same text, no extra wait."""
        bc = self.bc
        bc._spec_stt["chunks"] = 100
        bc._spec_stt["thread"] = self._dead_thread()
        bc._spec_stt["result"] = None            # folded: never decoded
        real = bc.transcribe
        bc.transcribe = lambda a: ("real text", {})
        self.addCleanup(setattr, bc, "transcribe", real)
        text, _conf = bc._transcribe_capture(object())
        self.assertEqual(text, "real text")

    @staticmethod
    def _dead_thread():
        t = threading.Thread(target=lambda: None, name="spec-stt")
        t.start()
        t.join()
        return t


if __name__ == "__main__":
    unittest.main()
