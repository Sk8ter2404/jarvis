"""Tests for core/clap_detector.py — the double-clap transient detector.

Everything here runs on SYNTHETIC waveforms built with numpy: no microphone,
no audio files, no sounddevice. Each generator models the property the
detector has to tell apart from a hand clap:

  * a clap        — a bright, noise-like burst: ~1 ms attack, ~7 ms direct
                    decay plus a quiet room tail;
  * speech        — harmonic (voiced) syllables of 120-250 ms with plosive
                    bursts and "s" fricatives, the bright/sharp parts of speech;
  * music beats   — kick (low, long), snare (bright, clap-like!) and hats on a
                    120 BPM grid over a sustained pad;
  * keyboard      — 1-2 ms clicks, bright and sharp but far too SHORT, in
                    typing bursts and as a lone pair;
  * a door        — a low thump with a long decay, then the latch click.

The contract (the feature spec): two sharp claps 0.15-0.7 s apart, with
nothing else loud around them, fire ONCE; nothing else ever fires.

stdlib unittest + numpy (numpy is a light-tier CI dependency).
"""
from __future__ import annotations

import unittest

import numpy as np

from core import clap_detector as cd

SR = 16000


# ─── synthetic sound generators ─────────────────────────────────────────────

def _band_noise(n, rng, lo, hi, sr=SR):
    """Unit-peak noise band-limited to [lo, hi] Hz with soft edges."""
    x = rng.normal(0.0, 1.0, n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    rise = np.clip((f - 0.7 * lo) / max(0.3 * lo, 1.0), 0.0, 1.0)
    fall = np.clip((1.3 * hi - f) / max(0.3 * hi, 1.0), 0.0, 1.0)
    y = np.fft.irfft(spec * rise * fall, n)
    return y / (np.max(np.abs(y)) + 1e-12)


def clap(rng, amp=0.5, sr=SR):
    """A hand clap: band-limited noise, fast attack, short decay + room tail."""
    n = int(0.15 * sr)
    t = np.arange(n) / sr
    body = _band_noise(n, rng, 600.0, 6000.0, sr)
    env = np.minimum(1.0, t / 0.0007) * (np.exp(-t / 0.007)
                                         + 0.08 * np.exp(-t / 0.045))
    y = body * env
    return amp * y / np.max(np.abs(y))


def key_click(rng, amp=0.3, sr=SR):
    """A keyboard key: a ~1 ms bright click and a smaller bottom-out click."""
    n = int(0.02 * sr)
    t = np.arange(n) / sr
    body = _band_noise(n, rng, 2000.0, 7000.0, sr)
    env = np.exp(-t / 0.0004) + 0.3 * np.exp(-np.maximum(t - 0.004, 0) / 0.0003) \
        * (t >= 0.004)
    y = body * env
    return amp * y / np.max(np.abs(y))


def vowel(rng, dur, amp, sr=SR):
    """A voiced syllable: harmonic series with formant weighting, vibrato,
    15 ms attack and 40 ms release."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = rng.uniform(100.0, 180.0) * (1.0 + 0.04 * np.sin(2 * np.pi * 4.0 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = np.zeros(n)
    base = float(f0[0])
    for h in range(1, 30):
        fh = h * base
        if fh > 7000:
            break
        w = sum(np.exp(-((fh - fc) / bw) ** 2)
                for fc, bw in ((700.0, 200.0), (1200.0, 250.0), (2500.0, 400.0)))
        y += (w + 0.02) * np.sin(h * phase)
    env = np.minimum(1.0, t / 0.015) * np.minimum(1.0, (dur - t) / 0.04)
    y = y * np.clip(env, 0.0, 1.0)
    return amp * y / (np.max(np.abs(y)) + 1e-12)


def speech_like(rng, dur=6.0, amp=0.3, sr=SR):
    """Running speech: syllables with plosive onsets and 's' fricatives."""
    out = np.zeros(int(dur * sr))
    pos = int(0.05 * sr)
    while pos < len(out) - int(0.4 * sr):
        kind = rng.integers(0, 3)
        if kind == 0:                       # plosive burst, then the vowel
            b = _band_noise(int(0.008 * sr), rng, 1500.0, 7000.0, sr) * 0.7 * amp
            out[pos:pos + len(b)] += b
            pos += len(b) + int(0.006 * sr)
        elif kind == 1:                     # fricative "s"
            ln = int(rng.uniform(0.06, 0.11) * sr)
            tt = np.arange(ln) / sr
            s = _band_noise(ln, rng, 3500.0, 7500.0, sr) * 0.35 * amp
            s *= np.minimum(1.0, tt / 0.01) * np.minimum(1.0, (tt[-1] - tt) / 0.01 + 1e-9)
            out[pos:pos + ln] += s
            pos += ln
        v = vowel(rng, rng.uniform(0.12, 0.25), amp * rng.uniform(0.6, 1.0), sr)
        out[pos:pos + len(v)] += v[:len(out) - pos]
        pos += len(v) + int(rng.uniform(0.03, 0.12) * sr)
    return out


def kick(rng, amp=0.6, sr=SR):
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    f = 50.0 + 70.0 * np.exp(-t / 0.03)
    y = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.12)
    y *= np.minimum(1.0, t / 0.002)
    return amp * y / np.max(np.abs(y))


def snare(rng, amp=0.45, sr=SR):
    """A snare: bright noise (clap-like) plus a body tone."""
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    y = (_band_noise(n, rng, 1000.0, 7000.0, sr) * np.exp(-t / 0.03)
         + 0.5 * np.sin(2 * np.pi * 185.0 * t) * np.exp(-t / 0.05))
    y *= np.minimum(1.0, t / 0.001)
    return amp * y / np.max(np.abs(y))


def hat(rng, amp=0.12, sr=SR):
    n = int(0.05 * sr)
    t = np.arange(n) / sr
    y = _band_noise(n, rng, 5000.0, 7800.0, sr) * np.exp(-t / 0.008)
    return amp * y / np.max(np.abs(y))


def thud(rng, amp=0.6, sr=SR):
    """A book dropped on the desk / a fist on the table: a short LOW body with
    a little broadband slap on top. Sharp, short and noisy — but dull."""
    n = int(0.2 * sr)
    t = np.arange(n) / sr
    y = (np.sin(2 * np.pi * 90.0 * t) * np.exp(-t / 0.03)
         + 0.12 * _band_noise(n, rng, 300.0, 7000.0, sr) * np.exp(-t / 0.006))
    y *= np.minimum(1.0, t / 0.001)
    return amp * y / np.max(np.abs(y))


def tink(rng, amp=0.5, sr=SR):
    """A spoon tapped on a glass: sharp, short, bright — but TONAL."""
    n = int(0.2 * sr)
    t = np.arange(n) / sr
    y = (np.sin(2 * np.pi * 2630.0 * t) + 0.6 * np.sin(2 * np.pi * 4410.0 * t)
         + 0.3 * np.sin(2 * np.pi * 6150.0 * t)) * np.exp(-t / 0.015)
    y *= np.minimum(1.0, t / 0.0005)
    return amp * y / np.max(np.abs(y))


def hiss(rng, amp=0.5, sr=SR):
    """A sharp-onset bright hiss that rings on (a spray can, a kettle's
    first burst): noise-like and bright — but far too LONG."""
    n = int(0.6 * sr)
    t = np.arange(n) / sr
    y = _band_noise(n, rng, 800.0, 7000.0, sr) * np.exp(-t / 0.2)
    y *= np.minimum(1.0, t / 0.001)
    return amp * y / np.max(np.abs(y))


def door(rng, amp=0.6, sr=SR):
    """A door shutting: a low thump with a long decay, latch click 0.35 s on."""
    n = int(0.9 * sr)
    t = np.arange(n) / sr
    y = (np.sin(2 * np.pi * 70.0 * t) + 0.5 * np.sin(2 * np.pi * 140.0 * t)
         + 0.2 * _band_noise(n, rng, 80.0, 400.0, sr)) * np.exp(-t / 0.12)
    y *= np.minimum(1.0, t / 0.003)
    y = amp * y / np.max(np.abs(y))
    c = key_click(rng, amp=0.35 * amp, sr=sr)
    at = int(0.35 * sr)
    y[at:at + len(c)] += c
    return y


class _Scene:
    """A mono timeline at SR with a quiet room-noise floor; sounds are mixed
    in at absolute times."""

    def __init__(self, seconds, rng, floor=0.0008):
        self.rng = rng
        self.x = rng.normal(0.0, floor, int(seconds * SR))

    def add(self, at_s, sound):
        i = int(at_s * SR)
        j = min(len(self.x), i + len(sound))
        self.x[i:j] += sound[:j - i]
        return self

    def claps(self, times, amp=0.5):
        for t in times:
            self.add(t, clap(self.rng, amp))
        return self

    def samples(self):
        return np.clip(self.x, -1.0, 1.0).astype(np.float32)


def run(samples, chunk=1024, **kw):
    """Feed ``samples`` in record_speech-sized chunks; return every event."""
    det = cd.ClapDetector(SR, **kw)
    events = []
    for i in range(0, len(samples), chunk):
        events.extend(det.feed(samples[i:i + chunk]))
    return events, det


# ─── the spec ───────────────────────────────────────────────────────────────

class DoubleClapFiresTests(unittest.TestCase):
    def test_a_double_clap_fires_exactly_once(self):
        rng = np.random.default_rng(1)
        x = _Scene(4.0, rng).claps([1.5, 1.8]).samples()
        events, _ = run(x)
        self.assertEqual(len(events), 1, events)
        ev = events[0]
        self.assertAlmostEqual(ev["interval_s"], 0.30, delta=0.03)
        self.assertAlmostEqual(ev["t_first"], 1.5, delta=0.03)
        self.assertAlmostEqual(ev["t_second"], 1.8, delta=0.03)
        # Decided only after the post-quiet window (a third clap would cancel).
        self.assertGreaterEqual(ev["t_detect"], ev["t_second"] + 0.6)

    def test_the_whole_spec_interval_range_fires(self):
        for gap in (0.18, 0.3, 0.45, 0.65):
            with self.subTest(gap=gap):
                rng = np.random.default_rng(int(gap * 1000))
                x = _Scene(4.0, rng).claps([1.2, 1.2 + gap]).samples()
                events, _ = run(x)
                self.assertEqual(len(events), 1, f"gap {gap}: {events}")

    def test_claps_over_a_steady_quiet_hum_still_fire(self):
        # A fridge / PC-fan hum is not "something loud": the floor adapts.
        rng = np.random.default_rng(2)
        sc = _Scene(4.0, rng)
        t = np.arange(len(sc.x)) / SR
        sc.x += 0.01 * np.sin(2 * np.pi * 120.0 * t)
        sc.claps([2.0, 2.35])
        events, _ = run(sc.samples())
        self.assertEqual(len(events), 1, events)

    def test_chunk_size_does_not_change_the_answer(self):
        rng = np.random.default_rng(3)
        x = _Scene(4.0, rng).claps([1.5, 1.85]).samples()
        for chunk in (160, 333, 1024, 4096, len(x)):
            with self.subTest(chunk=chunk):
                events, _ = run(x, chunk=chunk)
                self.assertEqual(len(events), 1, f"chunk {chunk}: {events}")

    def test_two_separate_double_claps_fire_twice(self):
        rng = np.random.default_rng(4)
        x = _Scene(7.0, rng).claps([1.0, 1.3, 4.5, 4.8]).samples()
        events, _ = run(x)
        self.assertEqual(len(events), 2, events)


class NotADoubleClapTests(unittest.TestCase):
    def test_a_single_clap_never_fires(self):
        rng = np.random.default_rng(10)
        events, det = run(_Scene(4.0, rng).claps([1.5]).samples())
        self.assertEqual(events, [])
        # ...but it WAS heard as a clap (the detector is not simply deaf).
        self.assertEqual(det.stats["claps"], 1)

    def test_a_triple_clap_never_fires(self):
        rng = np.random.default_rng(11)
        x = _Scene(4.0, rng).claps([1.2, 1.5, 1.8]).samples()
        events, det = run(x)
        self.assertEqual(events, [])
        self.assertEqual(det.stats["claps"], 3)

    def test_too_fast_and_too_slow_pairs_never_fire(self):
        for gap in (0.08, 0.9, 1.4):
            with self.subTest(gap=gap):
                rng = np.random.default_rng(int(gap * 100) + 20)
                x = _Scene(5.0, rng).claps([1.5, 1.5 + gap]).samples()
                events, _ = run(x)
                self.assertEqual(events, [], f"gap {gap}")

    def test_quiet_claps_are_below_the_sensitivity_knob(self):
        rng = np.random.default_rng(12)
        x = _Scene(4.0, rng).claps([1.5, 1.8], amp=0.08).samples()
        events, det = run(x)                    # default min_peak
        self.assertEqual(events, [])
        self.assertIn("quiet", det.last_rejection or "")
        events, _ = run(x, min_peak=0.05)       # a more sensitive setting
        self.assertEqual(len(events), 1, "the knob must reach quiet claps")

    def test_a_clap_right_after_talking_never_fires(self):
        # "...and then clap clap": speech inside the pre-quiet window.
        rng = np.random.default_rng(13)
        sc = _Scene(5.0, rng)
        sc.add(0.2, speech_like(rng, dur=1.6))
        sc.claps([2.0, 2.3])
        events, _ = run(sc.samples())
        self.assertEqual(events, [])

    def test_talking_right_after_the_claps_cancels_them(self):
        rng = np.random.default_rng(14)
        sc = _Scene(5.0, rng)
        sc.claps([1.0, 1.3])
        sc.add(1.55, speech_like(rng, dur=1.5))
        events, _ = run(sc.samples())
        self.assertEqual(events, [])

    def test_talking_well_before_the_claps_does_not_block_them(self):
        rng = np.random.default_rng(15)
        sc = _Scene(6.0, rng)
        sc.add(0.1, speech_like(rng, dur=1.6))
        sc.claps([3.5, 3.8])
        events, _ = run(sc.samples())
        self.assertEqual(len(events), 1, events)


class FalseTriggerTests(unittest.TestCase):
    """Sounds that are not claps: none of them may ever fire."""

    def test_running_speech_never_fires(self):
        for seed in range(30, 36):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                sc = _Scene(8.0, rng)
                sc.add(0.3, speech_like(rng, dur=7.5))
                events, _ = run(sc.samples())
                self.assertEqual(events, [])

    def test_loud_speech_never_fires(self):
        rng = np.random.default_rng(37)
        sc = _Scene(8.0, rng)
        sc.add(0.3, speech_like(rng, dur=7.5, amp=0.9))
        events, _ = run(sc.samples())
        self.assertEqual(events, [])

    def _beat_track(self, rng, seconds=10.0, bpm=120.0, snare_fn=None):
        sc = _Scene(seconds, rng)
        t = np.arange(len(sc.x)) / SR
        sc.x += 0.03 * (np.sin(2 * np.pi * 110.0 * t)
                        + 0.6 * np.sin(2 * np.pi * 164.8 * t)
                        + 0.4 * np.sin(2 * np.pi * 220.0 * t))
        beat = 60.0 / bpm
        n = int(seconds / beat)
        for b in range(n):
            at = 0.2 + b * beat
            if b % 2 == 0:
                sc.add(at, kick(rng))
            else:
                sc.add(at, (snare_fn or snare)(rng))
            sc.add(at, hat(rng))
            sc.add(at + beat / 2, hat(rng))
        return sc

    def test_a_120_bpm_beat_never_fires(self):
        rng = np.random.default_rng(40)
        events, _ = run(self._beat_track(rng).samples())
        self.assertEqual(events, [])

    def test_a_beat_with_real_handclaps_on_two_and_four_never_fires(self):
        rng = np.random.default_rng(41)
        sc = self._beat_track(rng, snare_fn=lambda r: clap(r, 0.5))
        events, _ = run(sc.samples())
        self.assertEqual(events, [])

    def test_a_train_of_claps_never_fires(self):
        # Clapping along to music, every 0.5 s: never two "and nothing else".
        rng = np.random.default_rng(42)
        sc = _Scene(8.0, rng).claps([1.0 + 0.5 * k for k in range(12)])
        events, det = run(sc.samples())
        self.assertEqual(events, [])
        self.assertGreaterEqual(det.stats["claps"], 10)

    def test_a_snare_only_groove_with_gaps_never_fires(self):
        # Snare on a 0.4 s grid in bars of four with a gap: every pair has a
        # neighbour inside the isolation window.
        rng = np.random.default_rng(43)
        sc = _Scene(9.0, rng)
        for bar in range(4):
            for k in range(4):
                sc.add(0.5 + bar * 2.0 + k * 0.4, snare(rng))
        events, _ = run(sc.samples())
        self.assertEqual(events, [])

    def test_typing_never_fires(self):
        for seed in range(50, 54):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                sc = _Scene(6.0, rng)
                at = 0.5
                while at < 5.5:
                    sc.add(at, key_click(rng, amp=rng.uniform(0.05, 0.45)))
                    at += rng.uniform(0.07, 0.25)
                events, _ = run(sc.samples())
                self.assertEqual(events, [])

    def test_two_lone_loud_keystrokes_never_fire(self):
        # Two isolated, LOUD key clicks at a clap-like spacing: too short to be
        # a clap no matter how loud.
        rng = np.random.default_rng(55)
        sc = _Scene(4.0, rng)
        sc.add(1.5, key_click(rng, amp=0.6)).add(1.8, key_click(rng, amp=0.6))
        events, det = run(sc.samples())
        self.assertEqual(events, [])
        self.assertEqual(det.stats["claps"], 0)

    def test_a_door_never_fires(self):
        for seed in range(60, 63):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                events, _ = run(_Scene(4.0, rng).add(1.0, door(rng)).samples())
                self.assertEqual(events, [])

    # Each of the next tests is the ONLY scene in this file that one rule of
    # the clap shape rejects on its own (everything else about the sound is
    # clap-like and it is isolated), so deleting that rule turns it red.
    def test_two_knocks_on_the_desk_never_fire(self):        # brightness
        rng = np.random.default_rng(65)
        sc = _Scene(4.0, rng).add(1.5, thud(rng)).add(1.85, thud(rng))
        events, det = run(sc.samples())
        self.assertEqual(events, [])
        self.assertEqual(det.stats["claps"], 0)
        self.assertIn("dull", det.last_rejection or "")

    def test_two_taps_on_a_glass_never_fire(self):           # flatness
        rng = np.random.default_rng(66)
        sc = _Scene(4.0, rng).add(1.5, tink(rng)).add(1.85, tink(rng))
        events, det = run(sc.samples())
        self.assertEqual(events, [])
        self.assertEqual(det.stats["claps"], 0)
        self.assertIn("tonal", det.last_rejection or "")

    def test_a_ringing_hiss_is_not_a_clap(self):             # decay
        rng = np.random.default_rng(67)
        _events, det = run(_Scene(4.0, rng).add(1.5, hiss(rng)).samples())
        self.assertEqual(det.stats["claps"], 0)
        self.assertIn("too long", det.last_rejection or "")

    def test_something_loud_between_the_claps_cancels_them(self):  # between
        rng = np.random.default_rng(68)
        sc = _Scene(4.0, rng).claps([1.5, 2.15])
        sc.add(1.75, vowel(rng, 0.15, 0.25))      # "uh" between the claps
        events, det = run(sc.samples())
        self.assertEqual(events, [])
        self.assertEqual(det.stats["claps"], 2)

    def test_a_loud_clap_and_a_faint_one_never_fire(self):   # level match
        rng = np.random.default_rng(69)
        sc = _Scene(4.0, rng)
        sc.add(1.5, clap(rng, 0.9)).add(1.8, clap(rng, 0.13))
        events, det = run(sc.samples())
        self.assertEqual(events, [])
        self.assertEqual(det.stats["claps"], 2)

    def test_claps_that_barely_clear_a_loud_background_do_not_fire(self):
        # sharpness: a vacuum / loud fan ~13 dB under the claps. Quiet enough
        # for the isolation rule, but the claps do not stand clear of it.
        rng = np.random.default_rng(75)
        sc = _Scene(4.0, rng, floor=0.033).claps([1.5, 1.8])
        events, det = run(sc.samples())
        self.assertEqual(events, [])
        self.assertIn("not sharp", det.last_rejection or "")

    def test_claps_well_clear_of_a_background_still_fire(self):
        rng = np.random.default_rng(75)
        sc = _Scene(4.0, rng, floor=0.006).claps([1.5, 1.8])
        events, _ = run(sc.samples())
        self.assertEqual(len(events), 1, events)

    def test_a_digital_pop_never_fires(self):
        # A buffer glitch: single-sample spikes 0.3 s apart.
        rng = np.random.default_rng(64)
        sc = _Scene(4.0, rng)
        sc.x[int(1.5 * SR)] = 0.9
        sc.x[int(1.8 * SR)] = -0.9
        events, det = run(sc.samples())
        self.assertEqual(events, [])
        self.assertEqual(det.stats["claps"], 0)


class StreamHandlingTests(unittest.TestCase):
    def test_reset_forgets_the_first_clap(self):
        rng = np.random.default_rng(70)
        x = _Scene(4.0, rng).claps([1.5, 1.8]).samples()
        det = cd.ClapDetector(SR)
        cut = int(1.65 * SR)
        events = det.feed(x[:cut])
        det.reset()
        events += det.feed(x[cut:])
        self.assertEqual(events, [], "a pair must never span a reset")

    def test_claps_too_soon_after_a_reset_are_not_trusted(self):
        # No history = no proof of "nothing loud before the first clap".
        rng = np.random.default_rng(71)
        x = _Scene(2.0, rng).claps([0.1, 0.4]).samples()
        events, _ = run(x)
        self.assertEqual(events, [])

    def test_empty_and_odd_frames_never_raise(self):
        det = cd.ClapDetector(SR)
        self.assertEqual(det.feed(np.zeros(0, dtype=np.float32)), [])
        self.assertEqual(det.feed(np.zeros((7, 1), dtype=np.float32)), [])
        self.assertEqual(det.feed(None), [])
        self.assertEqual(det.feed(np.full(1024, np.nan, dtype=np.float32)), [])

    def test_stereo_frames_use_the_first_channel(self):
        rng = np.random.default_rng(72)
        x = _Scene(4.0, rng).claps([1.5, 1.8]).samples()
        st = np.stack([x, np.zeros_like(x)], axis=1)
        events, _ = run(st)
        self.assertEqual(len(events), 1)

    def test_other_sample_rates_work(self):
        # 48 kHz device: resample the scene by repeating samples x3 (crude but
        # spectrally fine below 8 kHz for this purpose).
        rng = np.random.default_rng(73)
        x = _Scene(4.0, rng).claps([1.5, 1.8]).samples()
        x48 = np.repeat(x, 3)
        det = cd.ClapDetector(48000)
        events = []
        for i in range(0, len(x48), 3072):
            events.extend(det.feed(x48[i:i + 3072]))
        self.assertEqual(len(events), 1, events)

    def test_processing_is_cheap(self):
        # The detector runs on its own thread, but it must stay negligible:
        # well under 1 ms of CPU per 64 ms chunk on any dev box.
        import time
        rng = np.random.default_rng(74)
        sc = _Scene(10.0, rng)
        sc.add(0.3, speech_like(rng, dur=9.0))
        x = sc.samples()
        det = cd.ClapDetector(SR)
        t0 = time.perf_counter()
        n = 0
        for i in range(0, len(x), 1024):
            det.feed(x[i:i + 1024])
            n += 1
        per_chunk_ms = (time.perf_counter() - t0) * 1000.0 / n
        self.assertLess(per_chunk_ms, 2.0, f"{per_chunk_ms:.3f} ms per chunk")


class FeatureHelperTests(unittest.TestCase):
    def test_flatness_separates_noise_from_tones(self):
        rng = np.random.default_rng(80)
        noise = rng.normal(0, 0.1, 1024)
        t = np.arange(1024) / SR
        tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
        c_noise, f_noise = cd.spectral_shape(noise, SR)
        c_tone, f_tone = cd.spectral_shape(tone, SR)
        self.assertGreater(f_noise, 0.3)
        self.assertLess(f_tone, 0.05)

    def test_centroid_tracks_brightness(self):
        t = np.arange(1024) / SR
        low, _ = cd.spectral_shape(np.sin(2 * np.pi * 300.0 * t), SR)
        high, _ = cd.spectral_shape(np.sin(2 * np.pi * 4000.0 * t), SR)
        self.assertAlmostEqual(low, 300.0, delta=80.0)
        self.assertAlmostEqual(high, 4000.0, delta=150.0)

    def test_effective_duration_separates_clicks_from_claps(self):
        rng = np.random.default_rng(81)
        clk = np.concatenate([key_click(rng, 0.5), np.zeros(800)])
        clp = clap(rng, 0.5)[:1120]
        self.assertLess(cd.effective_duration_ms(clk, SR), 1.5)
        self.assertGreater(cd.effective_duration_ms(clp, SR), 2.0)

    def test_helpers_never_raise_on_junk(self):
        for junk in (np.zeros(0), np.zeros(10), None):
            cd.spectral_shape(junk, SR)
            cd.effective_duration_ms(junk, SR)


if __name__ == "__main__":
    unittest.main()
