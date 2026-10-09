"""Echo cancellation of what the PC plays (MEDIA_AEC_MODE, 2026-10-05):
core/audio_processor.MediaEchoCanceller and its helpers.

Everything is SYNTHETIC and seeded: a music-like reference (band-limited
noise + a gated tone) goes through a known room FIR, a clock drift (a
windowed-sinc resampler, as two USB crystals do) and a bulk delay to make
the "mic"; no device is opened. numpy only (light tier).

The spec's light test: ERLE >= 15 dB within 5 s at 20 ppm drift, and a
near-end burst (the owner talking over the video) must not drop it below
15 dB.

    python -m unittest tests.test_media_aec
"""
from __future__ import annotations

import unittest

import numpy as np

from core import audio_processor as ap

SR = 16000


def _resample_ppm(x, ppm, taps=32):
    """x evaluated at n * (1 + ppm*1e-6) (Kaiser-windowed sinc)."""
    x = np.asarray(x, np.float64)
    if not ppm:
        return x
    n = len(x)
    xp = np.concatenate([np.zeros(taps), x, np.zeros(taps + 2)])
    k = np.arange(-taps // 2 + 1, taps // 2 + 1)
    kw = np.kaiser(taps + 2, 8.0)[1:-1][None, :]
    pos = np.arange(n, dtype=np.float64) * (1 + ppm * 1e-6)
    i = np.floor(pos).astype(np.int64)
    f = pos - i
    w = np.sinc((k[None, :] - f[:, None]) * 0.95) * 0.95 * kw
    idx = np.clip(i[:, None] + k[None, :] + taps, 0, len(xp) - 1)
    return np.sum(xp[idx] * w, axis=1)


def _bandpass(x, lo_hz=80.0, hi_hz=6000.0):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    X[(f < lo_hz) | (f > hi_hz)] = 0.0
    return np.fft.irfft(X, len(x))


def make_scene(seconds=8.0, ppm=20.0, delay_ms=60.0, seed=7,
               near=None, gain=0.2):
    """(reference, mic, echo, near) as float64 arrays."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ref = _bandpass(0.05 * rng.standard_normal(n))
    ref += 0.03 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    h = np.zeros(int(0.12 * SR))
    h[0] = 1.0
    h[1:] = (rng.standard_normal(len(h) - 1) * 0.3
             * np.exp(-np.arange(1, len(h)) / (0.02 * SR)))
    e = _resample_ppm(ref, ppm)
    e = np.convolve(e, h)[:n]
    d = int(delay_ms * 1e-3 * SR)
    echo = np.concatenate([np.zeros(d), e[:n - d]]) * gain
    nearsig = np.zeros(n)
    if near is not None:
        a, b = int(near[0] * SR), int(near[1] * SR)
        seg = rng.standard_normal(b - a)
        nearsig[a:b] = (0.03 * _bandpass(seg, 200.0, 3500.0)
                        / max(1e-9, np.std(_bandpass(seg, 200.0, 3500.0)))
                        * np.hanning(b - a))
    mic = echo + nearsig + 0.0004 * rng.standard_normal(n)
    return ref, mic, echo, nearsig


class FakeRef:
    """The loopback ring, holding the whole synthetic reference."""

    def __init__(self, x):
        self.x = np.asarray(x, np.float32)
        self.n_written = len(self.x)
        self.gap_seq = 0

    def index_at(self, t):
        return t * SR

    def read(self, start, n):
        if start < 0 or start + n > len(self.x):
            return None
        return self.x[start:start + n]


def run(aec, mic, chunk=1024):
    lin_out, sup_out = [], []
    for i in range(0, len(mic) - chunk + 1, chunk):
        lin, sup = aec.process(mic[i:i + chunk], t=(i + chunk) / SR)
        lin_out.append(lin)
        sup_out.append(sup)
    return np.concatenate(lin_out), np.concatenate(sup_out)


def erle_db(echo, residual):
    return 10 * np.log10(np.mean(echo ** 2)
                         / max(np.mean(residual ** 2), 1e-20))


class ConvergenceTests(unittest.TestCase):

    def test_reaches_15_db_within_5_s_at_20_ppm(self):
        ref, mic, echo, near = make_scene(seconds=6.0, ppm=20.0)
        aec = ap.MediaEchoCanceller(FakeRef(ref))
        lin, _sup = run(aec, mic)
        a, b = int(4.5 * SR), int(5.0 * SR)
        self.assertGreaterEqual(erle_db(echo[a:b], lin[a:b] - near[a:b]), 15.0)
        st = aec.status()
        self.assertGreater(st["drift_updates"], 0)
        self.assertAlmostEqual(st["drift_ppm"], 20.0, delta=4.0)

    def test_owner_talking_over_the_video_keeps_it_above_15_db(self):
        ref, mic, echo, near = make_scene(seconds=9.0, ppm=20.0,
                                          near=(6.0, 7.5))
        aec = ap.MediaEchoCanceller(FakeRef(ref))
        lin, _sup = run(aec, mic)
        for a, b in ((6.0, 7.5), (7.5, 9.0)):
            with self.subTest(window=(a, b)):
                s0, s1 = int(a * SR), int(b * SR) - 1024
                self.assertGreaterEqual(
                    erle_db(echo[s0:s1], lin[s0:s1] - near[s0:s1]), 15.0)
        # ... and his voice is still there (linear output, for STT).
        s0, s1 = int(6.3 * SR), int(7.2 * SR)
        kept = np.corrcoef(lin[s0:s1], near[s0:s1])[0, 1]
        self.assertGreater(kept, 0.95)

    def test_the_suppressed_copy_quiets_the_capture_start_signal(self):
        # Video only: the frames a capture would start on (> VAD 0.008).
        ref, mic, echo, near = make_scene(seconds=8.0, ppm=0.0, gain=0.4)
        aec = ap.MediaEchoCanceller(FakeRef(ref))
        _lin, sup = run(aec, mic)
        tail = slice(int(5.0 * SR), len(sup))
        f_raw = mic[tail][: (len(sup[tail]) // 1024) * 1024].reshape(-1, 1024)
        f_sup = sup[tail][: (len(sup[tail]) // 1024) * 1024].reshape(-1, 1024)
        above_raw = np.mean(np.sqrt(np.mean(f_raw ** 2, axis=1)) > 0.008)
        above_sup = np.mean(np.sqrt(np.mean(f_sup ** 2, axis=1)) > 0.008)
        self.assertGreater(above_raw, 0.9)
        self.assertLess(above_sup, 0.05)

    def test_drift_is_mandatory(self):
        # The same scene with the drift estimate frozen at 0 cannot get
        # there: the reason the estimator exists.
        ref, mic, echo, near = make_scene(seconds=6.0, ppm=20.0)
        aec = ap.MediaEchoCanceller(FakeRef(ref))
        aec._update_drift = lambda: None
        lin, _sup = run(aec, mic)
        a, b = int(4.5 * SR), int(5.0 * SR)
        self.assertLess(erle_db(echo[a:b], lin[a:b]), 12.0)


class AlignmentTests(unittest.TestCase):

    def test_gcc_finds_the_delay(self):
        rng = np.random.default_rng(1)
        x = _bandpass(rng.standard_normal(SR))
        d = np.concatenate([np.zeros(300), x[:-300]])
        lag, peak = ap.gcc_phat_lag(x, d, max_lag=2000, max_lead=500)
        self.assertAlmostEqual(lag, 300.0, delta=0.2)
        self.assertGreater(peak, 0.5)
        lead, _ = ap.gcc_phat_lag(d, x, max_lag=2000, max_lead=500)
        self.assertAlmostEqual(lead, -300.0, delta=0.2)

    def test_a_far_delay_is_re_aligned(self):
        # 250 ms: past the margin + 50 ms, so the reference is moved.
        ref, mic, echo, near = make_scene(seconds=6.0, ppm=0.0,
                                          delay_ms=250.0)
        aec = ap.MediaEchoCanceller(FakeRef(ref))
        lin, _sup = run(aec, mic)
        st = aec.status()
        self.assertGreaterEqual(st["realigns"], 1)
        self.assertAlmostEqual(st["delay_ms"], 250.0, delta=5.0)
        a, b = int(5.0 * SR), int(5.9 * SR)
        self.assertGreaterEqual(erle_db(echo[a:b], lin[a:b]), 15.0)

    def test_filter_shift_moves_the_taps(self):
        f = ap._PBFDAF()
        h = np.zeros(f.P * f.B)
        h[700] = 1.0
        g = np.zeros((f.P, 2 * f.B))
        g[:, :f.B] = h.reshape(f.P, f.B)
        f.Wf[:] = np.fft.rfft(g, axis=1)
        f.Wb[:] = f.Wf
        f.shift(100)
        self.assertEqual(int(np.argmax(np.abs(f.taps()))), 600)
        f.shift(-50)
        self.assertEqual(int(np.argmax(np.abs(f.taps()))), 650)


class GuardTests(unittest.TestCase):

    def test_no_reference_passes_the_mic_through(self):
        aec = ap.MediaEchoCanceller(None)
        x = np.random.default_rng(2).standard_normal(1024).astype(np.float32)
        lin, sup = aec.process(x, t=1.0)
        np.testing.assert_array_equal(lin, x)
        np.testing.assert_array_equal(sup, x)

    def test_a_silent_reference_keeps_the_weights(self):
        ref, mic, echo, near = make_scene(seconds=5.0, ppm=0.0)
        aec = ap.MediaEchoCanceller(FakeRef(ref))
        run(aec, mic)
        w = aec.f.Wf.copy()
        quiet = FakeRef(np.zeros(len(ref)))
        aec.ref = quiet
        aec.new_session()
        silence = np.random.default_rng(3).standard_normal(4 * 1024) * 1e-4
        run(aec, silence)
        np.testing.assert_allclose(aec.f.Wf, w)

    def test_divergence_resets_the_filter(self):
        aec = ap.MediaEchoCanceller(None)
        f = aec.f
        f.Wf[:] = 50.0        # a wildly wrong output filter
        d = np.zeros(256)
        e = np.ones(256)      # output far louder than input
        for _ in range(int(ap.MEDIA_AEC_DIVERGE_S * SR / 256) + 1):
            aec._guard(d + 0.001, e, True)
        self.assertEqual(aec.stats["resets"], 1)
        self.assertFalse(np.any(f.Wf))

    def test_a_diverged_filter_is_reset_through_process(self):
        # Both filters garbage (the fg/bg pair cannot heal itself): the
        # output is far louder than the mic, and process() must reset them
        # within ~1 s and then cancel again.
        ref, mic, echo, near = make_scene(seconds=8.0, ppm=0.0)
        aec = ap.MediaEchoCanceller(FakeRef(ref))
        run(aec, mic[:3 * SR])
        rng = np.random.default_rng(4)
        junk = (rng.standard_normal(aec.f.Wf.shape)
                + 1j * rng.standard_normal(aec.f.Wf.shape)) * 5.0
        aec.f.Wf[:] = junk
        aec.f.Wb[:] = junk
        lin, _sup = run(aec, mic[3 * SR:])
        self.assertGreaterEqual(aec.stats["resets"], 1)
        tail = slice(len(lin) - SR, len(lin))
        self.assertLess(float(np.mean(lin[tail] ** 2)),
                        float(np.mean(mic[3 * SR:][tail] ** 2)))

    def test_never_raises(self):
        class Boom:
            n_written = 10 ** 9
            gap_seq = 0

            def index_at(self, t):
                return 1000.0

            def read(self, start, n):
                raise RuntimeError("boom")

        aec = ap.MediaEchoCanceller(Boom())
        x = np.ones(1024, np.float32)
        lin, sup = aec.process(x, t=1.0)
        self.assertEqual(lin.shape, x.shape)
        self.assertIn("last_error", aec.stats)

    def test_odd_chunk_sizes_keep_the_length(self):
        ref, mic, echo, near = make_scene(seconds=2.0, ppm=0.0)
        aec = ap.MediaEchoCanceller(FakeRef(ref))
        pos = 0
        for n in (1000, 300, 2048, 17, 5000):
            lin, sup = aec.process(mic[pos:pos + n], t=(pos + n) / SR)
            self.assertEqual(len(lin), n)
            self.assertEqual(len(sup), n)
            pos += n


if __name__ == "__main__":
    unittest.main()
