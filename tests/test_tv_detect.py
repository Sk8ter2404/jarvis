"""Tests for the camera-based "is the TV on?" detector.

Three layers, matching the feature's three pieces:

  1. PURE MATH + STORE (audio/tv_detect.py) — luma brightness, frame-to-frame
     temporal variance, the per-frame qualify rule, the rolling hysteresis
     TVDecider, region cropping + the uncalibrated whole-frame fallback, and the
     TVRegionStore round-trip on a TMP path. numpy only (NO cv2), so these run on
     the CI runner too.

  2. SKILL WIRING (skills/tv_detect.py) — is_tv_on() is False when the flag is
     off; with the flag on and a FAKE monolith feeding BRIGHT + HIGH-VARIANCE
     frames it goes True (→ ambient suppression), while DARK/STATIC frames keep
     it False; calibrate writes the region store; the on/off toggle persists.
     Loaded in isolation via the shared skill harness (no monolith boot), with
     the region store + settings writer redirected to TMP paths.

  3. AMBIENT OR-SIGNAL (bobert_companion._ambient_media_is_playing) — with a
     fake skill_tv_detect whose is_tv_on() returns True, the ambient
     media-playing gate returns True (suppress). @requires_monolith → local full
     tier only (skips on CI, where cv2 is absent and the monolith can't import),
     mirroring the existing _ambient_media_is_playing tests.

The region store always uses a throwaway path (JARVIS_TV_REGION_PATH → tempfile)
so the real data/tv_region.json is never read or written; the settings writer is
redirected via JARVIS_SETTINGS_PATH so data/user_settings.json is never touched.

stdlib unittest + mock. Skips cleanly if numpy is unavailable.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy as np
except Exception:   # pragma: no cover - numpy is present locally + on CI
    np = None

from audio import tv_detect as td
from tests._skill_harness import load_skill_isolated


# ─── synthetic frame builders (BGR uint8, like the face-tracker cache) ──────
def _bright_noisy_frame(h=64, w=64, base=200, noise=60, seed=None):
    """A BRIGHT frame with strong per-pixel noise — stands in for a lit, moving
    TV panel. Each call (or each seed) differs so consecutive frames have a large
    temporal delta (the flicker signal)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(max(0, base - noise), min(255, base + noise) + 1,
                       size=(h, w, 3), dtype=np.int16)
    return img.clip(0, 255).astype(np.uint8)


def _dark_static_frame(h=64, w=64, value=10):
    """A DARK, perfectly static frame — an off TV / a dim wall. Identical every
    call so the temporal delta between two of them is 0."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _bright_static_frame(h=64, w=64, value=220):
    """A BRIGHT but perfectly STATIC frame — a lamp or a sunlit wall. Bright
    enough to clear the brightness gate but with zero motion, so it must NOT read
    as a TV (this is the case the temporal-variance gate exists to reject)."""
    return np.full((h, w, 3), value, dtype=np.uint8)


@unittest.skipUnless(np is not None, "numpy required for tv_detect math")
class PureStatsTests(unittest.TestCase):
    def test_brightness_bright_vs_dark(self):
        self.assertGreater(td.frame_brightness(_bright_noisy_frame(seed=1)),
                           td.BRIGHTNESS_ON_MIN)
        self.assertLess(td.frame_brightness(_dark_static_frame()),
                        td.BRIGHTNESS_ON_MIN)

    def test_brightness_none_on_bad_frame(self):
        self.assertIsNone(td.frame_brightness(None))
        self.assertIsNone(td.frame_brightness("not a frame"))

    def test_temporal_delta_moving_vs_static(self):
        a = _bright_noisy_frame(seed=1)
        b = _bright_noisy_frame(seed=2)
        # Two different noisy frames → large mean abs delta.
        self.assertGreater(td.frame_temporal_delta(a, b), td.TEMPORAL_DELTA_ON_MIN)
        # A frame vs itself → zero motion.
        self.assertEqual(td.frame_temporal_delta(a, a), 0.0)
        # Two identical static frames → zero motion.
        s = _bright_static_frame()
        self.assertEqual(td.frame_temporal_delta(s, s.copy()), 0.0)

    def test_temporal_delta_none_on_shape_mismatch(self):
        a = _bright_noisy_frame(h=64, w=64, seed=1)
        b = _bright_noisy_frame(h=48, w=48, seed=2)
        self.assertIsNone(td.frame_temporal_delta(a, b))

    def test_frame_qualifies_requires_both(self):
        # Bright AND moving → qualifies.
        self.assertTrue(td.frame_qualifies(200.0, 30.0))
        # Bright but static → no.
        self.assertFalse(td.frame_qualifies(200.0, 0.0))
        # Moving but dark → no.
        self.assertFalse(td.frame_qualifies(5.0, 30.0))
        # Missing stat → no.
        self.assertFalse(td.frame_qualifies(None, 30.0))
        self.assertFalse(td.frame_qualifies(200.0, None))


@unittest.skipUnless(np is not None, "numpy required for tv_detect math")
class DeciderTests(unittest.TestCase):
    def test_sustained_bright_moving_turns_on(self):
        """Bright + high-variance readings, sustained → TV ON → suppress."""
        d = td.TVDecider()
        on = False
        for i in range(td.ON_QUALIFYING_FRAMES + 1):
            on = d.observe(200.0, 30.0, ts=float(i))
        self.assertTrue(on)
        self.assertTrue(d.is_on(now=float(td.ON_QUALIFYING_FRAMES + 1)))

    def test_dark_static_never_turns_on(self):
        d = td.TVDecider()
        for i in range(10):
            d.observe(8.0, 0.0, ts=float(i))   # dark + static
        self.assertFalse(d.is_on(now=10.0))

    def test_bright_but_static_never_turns_on(self):
        """A bright, motionless scene (lamp / sunlit wall) must NOT read as a TV
        — temporal variance is the discriminator."""
        d = td.TVDecider()
        for i in range(10):
            d.observe(230.0, 0.5, ts=float(i))  # very bright, no motion
        self.assertFalse(d.is_on(now=10.0))

    def test_single_qualifying_frame_does_not_flip(self):
        """One bright+moving frame amid quiet frames must not trip the verdict
        (hysteresis / majority requirement)."""
        d = td.TVDecider()
        d.observe(8.0, 0.0, ts=0.0)
        d.observe(200.0, 30.0, ts=1.0)   # lone qualifying frame
        d.observe(8.0, 0.0, ts=2.0)
        self.assertFalse(d.is_on(now=2.0))

    def test_turns_off_after_quiet(self):
        """Once ON, sustained quiet frames switch it back OFF."""
        d = td.TVDecider()
        for i in range(td.ON_QUALIFYING_FRAMES + 1):
            d.observe(200.0, 30.0, ts=float(i))
        self.assertTrue(d.is_on())
        t = float(td.ON_QUALIFYING_FRAMES + 1)
        for _ in range(td.DECISION_WINDOW + 1):
            d.observe(8.0, 0.0, ts=t)
            t += 1.0
        self.assertFalse(d.is_on(now=t))

    def test_stale_readings_decay_to_off(self):
        """A frozen camera (no new frames) must not keep the verdict latched —
        aging the window past READING_MAX_AGE_S drops it to OFF."""
        d = td.TVDecider()
        for i in range(td.ON_QUALIFYING_FRAMES + 1):
            d.observe(200.0, 30.0, ts=float(i))
        self.assertTrue(d.is_on())
        # No new observations; jump the clock far past the max age.
        future = float(td.ON_QUALIFYING_FRAMES + 1) + td.READING_MAX_AGE_S + 100
        self.assertFalse(d.is_on(now=future))

    def test_missing_stats_are_non_qualifying(self):
        d = td.TVDecider()
        for i in range(td.DECISION_WINDOW):
            d.observe(None, None, ts=float(i))   # no readings (camera gap)
        self.assertFalse(d.is_on(now=float(td.DECISION_WINDOW)))


@unittest.skipUnless(np is not None, "numpy required for tv_detect math")
class RegionTests(unittest.TestCase):
    def test_normalize_region_clamps_and_rejects(self):
        r = td.normalize_region(0.1, 0.2, 0.5, 0.5)
        self.assertAlmostEqual(r["x"], 0.1)
        self.assertAlmostEqual(r["w"], 0.5)
        # Far edge clamped into the frame.
        r2 = td.normalize_region(0.8, 0.8, 0.5, 0.5)
        self.assertAlmostEqual(r2["x"] + r2["w"], 1.0)
        # Degenerate → None.
        self.assertIsNone(td.normalize_region(0.0, 0.0, 0.0, 0.5))
        self.assertIsNone(td.normalize_region(0.0, 0.0, -0.5, 0.5))

    def test_crop_region_selects_subframe(self):
        # Left half dark, right half bright; crop the right half → bright.
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, 20:, :] = 240
        right = {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}
        self.assertGreater(td.frame_brightness(img, right), td.BRIGHTNESS_ON_MIN)
        left = {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}
        self.assertLess(td.frame_brightness(img, left), td.BRIGHTNESS_ON_MIN)

    def test_uncalibrated_whole_frame_fallback(self):
        """A None/empty/degenerate region uses the WHOLE frame — the documented
        uncalibrated behaviour. A whole-frame bright+moving pair still detects."""
        a = _bright_noisy_frame(seed=1)
        b = _bright_noisy_frame(seed=2)
        # region=None → whole frame.
        self.assertGreater(td.frame_brightness(a, None), td.BRIGHTNESS_ON_MIN)
        self.assertGreater(td.frame_temporal_delta(a, b, None),
                           td.TEMPORAL_DELTA_ON_MIN)
        # An empty dict and a degenerate region also fall back to whole-frame
        # (don't raise / don't zero out).
        self.assertIsNotNone(td.frame_brightness(a, {}))
        self.assertIsNotNone(td.frame_brightness(a, {"x": 0, "y": 0, "w": 0, "h": 0}))


@unittest.skipUnless(np is not None, "numpy required for tv_detect math")
class StoreTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="jarvis_tvregion_")
        self.path = os.path.join(self._dir, "tv_region.json")
        self.addCleanup(lambda: shutil.rmtree(self._dir, ignore_errors=True))

    def test_round_trip(self):
        store = td.TVRegionStore(path=self.path)
        self.assertFalse(store.is_calibrated())
        self.assertIsNone(store.get_region())
        self.assertTrue(store.put_region(0.1, 0.2, 0.5, 0.6))
        self.assertTrue(store.is_calibrated())
        r = store.get_region()
        self.assertAlmostEqual(r["x"], 0.1)
        self.assertAlmostEqual(r["y"], 0.2)
        # A fresh store reading the same file sees it (persisted).
        self.assertTrue(td.TVRegionStore(path=self.path).is_calibrated())

    def test_clear(self):
        store = td.TVRegionStore(path=self.path)
        store.put_region(0.0, 0.0, 1.0, 1.0)
        self.assertTrue(store.is_calibrated())
        self.assertTrue(store.clear())
        self.assertFalse(store.is_calibrated())

    def test_put_degenerate_region_rejected(self):
        store = td.TVRegionStore(path=self.path)
        self.assertFalse(store.put_region(0.0, 0.0, 0.0, 0.0))
        self.assertFalse(store.is_calibrated())

    def test_corrupt_file_reads_as_uncalibrated(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        store = td.TVRegionStore(path=self.path)
        self.assertFalse(store.is_calibrated())   # no raise
        self.assertIsNone(store.get_region())

    def test_atomic_write_leaves_no_tmp(self):
        store = td.TVRegionStore(path=self.path)
        store.put_region(0.0, 0.0, 1.0, 1.0)
        leftovers = [n for n in os.listdir(self._dir) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])


# ─── SKILL WIRING ───────────────────────────────────────────────────────────
@unittest.skipUnless(np is not None, "numpy required for tv_detect skill")
class _SkillBase(unittest.TestCase):
    def setUp(self):
        # Redirect the region store + settings writer to throwaway paths.
        self._dir = tempfile.mkdtemp(prefix="jarvis_tvskill_")
        self.addCleanup(lambda: shutil.rmtree(self._dir, ignore_errors=True))
        self._region_path = os.path.join(self._dir, "tv_region.json")
        env = mock.patch.dict(os.environ, {
            "JARVIS_TV_REGION_PATH": self._region_path,
            "JARVIS_SETTINGS_PATH": os.path.join(self._dir, "settings.json"),
        })
        env.start()
        self.addCleanup(env.stop)

    def _load(self):
        mod, _actions = load_skill_isolated("tv_detect", register=False)
        # Reset the module-level decider/poll state between tests (the singleton
        # would otherwise carry readings across tests in the same process).
        if getattr(mod, "_decider", None) is not None:
            mod._decider.reset()
        mod._decider = None
        mod._prev_frame[0] = None
        mod._last_poll[0] = 0.0
        self._mod = mod
        return mod

    def _patch_flag(self, value):
        from core import config as cfg
        p = mock.patch.object(cfg, "TV_DETECT_ENABLED", value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def _fake_bc(self, frames_by_index):
        """A stand-in monolith module exposing the camera-frame cache the skill
        reads. `frames_by_index` is {index: frame}. The skill's _bc() prefers the
        live __main__ (which, under the test runner, is NOT the monolith), so we
        patch the skill's _bc to return this fake directly — exactly how the
        kinect_pointing skill tests pin _is_staging/_bc."""
        import threading
        m = types.ModuleType("bobert_companion")
        m._camera_state_lock = threading.Lock()
        m._camera_latest_frame = dict(frames_by_index)
        m._camera_last_frame_at = {i: 1e18 for i in frames_by_index}  # very fresh
        m._camera_last_seen = {i: 1e18 for i in frames_by_index}
        self._mod_bc = m
        p = mock.patch.object(self._mod, "_bc", lambda: m)
        p.start()
        self.addCleanup(p.stop)
        return m

    def _set_frame(self, bc, index, frame):
        with bc._camera_state_lock:
            bc._camera_latest_frame[index] = frame
            bc._camera_last_frame_at[index] = 1e18


class IsTvOnTests(_SkillBase):
    def test_false_when_disabled(self):
        mod = self._load()
        self._patch_flag(False)
        self._fake_bc({0: _bright_noisy_frame(seed=1)})
        self.assertFalse(mod.is_tv_on())

    def test_true_on_sustained_bright_moving(self):
        """Flag on + a fake camera feeding fresh BRIGHT+NOISY frames → is_tv_on()
        goes True after enough polls (→ ambient suppression)."""
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_noisy_frame(seed=0)})
        # Drive several polls with a DIFFERENT noisy frame each time (high
        # temporal variance) — advance the skill's monotonic clock past the
        # poll interval each call so the rate-limiter lets every sample through.
        clock = [1000.0]
        with mock.patch.object(mod.time, "monotonic", lambda: clock[0]):
            result = False
            for i in range(td.ON_QUALIFYING_FRAMES + 2):
                self._set_frame(bc, 0, _bright_noisy_frame(seed=i + 1))
                clock[0] += mod.POLL_MIN_INTERVAL_S + 0.01
                result = mod.is_tv_on()
        self.assertTrue(result)

    def test_false_on_dark_static(self):
        """Flag on but the camera shows a DARK, STATIC scene → stays False."""
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _dark_static_frame()})
        clock = [2000.0]
        with mock.patch.object(mod.time, "monotonic", lambda: clock[0]):
            result = True
            for _ in range(td.DECISION_WINDOW + 2):
                self._set_frame(bc, 0, _dark_static_frame())
                clock[0] += mod.POLL_MIN_INTERVAL_S + 0.01
                result = mod.is_tv_on()
        self.assertFalse(result)

    def test_false_on_bright_static(self):
        """A BRIGHT but motionless view (lamp/wall) → stays False (variance gate)."""
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_static_frame()})
        clock = [3000.0]
        with mock.patch.object(mod.time, "monotonic", lambda: clock[0]):
            result = True
            for _ in range(td.DECISION_WINDOW + 2):
                self._set_frame(bc, 0, _bright_static_frame())
                clock[0] += mod.POLL_MIN_INTERVAL_S + 0.01
                result = mod.is_tv_on()
        self.assertFalse(result)

    def test_false_when_no_camera(self):
        mod = self._load()
        self._patch_flag(True)
        self._fake_bc({})           # no cached frames
        self.assertFalse(mod.is_tv_on())

    def test_uncalibrated_uses_whole_frame(self):
        """No calibration stored → the detector watches the whole frame and still
        detects a bright+moving feed."""
        mod = self._load()
        self._patch_flag(True)
        # Confirm uncalibrated.
        self.assertFalse(td.TVRegionStore(path=self._region_path).is_calibrated())
        bc = self._fake_bc({0: _bright_noisy_frame(seed=0)})
        clock = [4000.0]
        with mock.patch.object(mod.time, "monotonic", lambda: clock[0]):
            result = False
            for i in range(td.ON_QUALIFYING_FRAMES + 2):
                self._set_frame(bc, 0, _bright_noisy_frame(seed=i + 1))
                clock[0] += mod.POLL_MIN_INTERVAL_S + 0.01
                result = mod.is_tv_on()
        self.assertTrue(result)


class CalibrateTests(_SkillBase):
    def test_calibrate_writes_region(self):
        mod = self._load()
        self._patch_flag(True)
        self._fake_bc({0: _bright_noisy_frame(seed=1)})
        out = mod.calibrate_tv_region("")
        self.assertIn("calibrated", out.lower())
        self.assertTrue(td.TVRegionStore(path=self._region_path).is_calibrated())

    def test_calibrate_honest_when_no_camera(self):
        mod = self._load()
        self._patch_flag(True)
        self._fake_bc({})
        # No sleeping in tests.
        with mock.patch.object(mod.time, "sleep", lambda *_a, **_k: None):
            out = mod.calibrate_tv_region("")
        self.assertIn("can't see a camera frame", out.lower())
        self.assertFalse(td.TVRegionStore(path=self._region_path).is_calibrated())

    def test_calibrate_honest_when_flag_off(self):
        mod = self._load()
        self._patch_flag(False)
        out = mod.calibrate_tv_region("")
        self.assertIn("off", out.lower())


class StatusToggleTests(_SkillBase):
    def _patch_settings_writer(self, initial=None):
        from tools import settings_window as sw
        saved = dict(initial or {})
        p1 = mock.patch.object(sw, "load_settings", lambda *a, **k: dict(saved))
        p2 = mock.patch.object(sw, "save_settings",
                               lambda d, *a, **k: saved.update(d))
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        return saved

    def test_status_off(self):
        mod = self._load()
        self._patch_flag(False)
        self.assertIn("off", mod.tv_detect_status("").lower())

    def test_on_persists_flag(self):
        mod = self._load()
        self._patch_flag(False)
        saved = self._patch_settings_writer()
        out = mod.tv_detect_on("")
        self.assertIn("on", out.lower())
        self.assertTrue(saved.get("TV_DETECT_ENABLED"))
        from core import config as cfg
        self.assertTrue(cfg.TV_DETECT_ENABLED)

    def test_off_persists_flag(self):
        mod = self._load()
        self._patch_flag(True)
        saved = self._patch_settings_writer({"TV_DETECT_ENABLED": True})
        out = mod.tv_detect_off("")
        self.assertIn("off", out.lower())
        self.assertFalse(saved.get("TV_DETECT_ENABLED"))

    def test_status_on_sees_tv(self):
        """With the flag on, a fresh fake camera frame, and the decider primed
        ON, status reports that it sees a TV."""
        mod = self._load()
        self._patch_flag(True)
        self._fake_bc({0: _bright_noisy_frame(seed=1)})
        # Prime the decider straight to ON so status reflects 'seeing a TV'
        # without driving the whole poll sequence again.
        d = mod._get_decider()
        for i in range(td.ON_QUALIFYING_FRAMES + 1):
            d.observe(200.0, 30.0, ts=float(i))
        # Freeze is_tv_on() True so the status line is deterministic regardless
        # of the live poll's timing.
        with mock.patch.object(mod, "is_tv_on", return_value=True):
            out = mod.tv_detect_status("")
        self.assertIn("see a lit, moving screen", out.lower())


class RegisterTests(_SkillBase):
    def test_register_exposes_actions(self):
        _mod, actions = load_skill_isolated("tv_detect", register=True)
        for name in ("calibrate_tv_region", "tv_detect_status",
                     "tv_detect_on", "tv_detect_off"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))


# ─── AMBIENT OR-SIGNAL (monolith seam; local full tier only) ────────────────
from tests._monolith_harness import load_monolith, requires_monolith  # noqa: E402


@requires_monolith
class AmbientOrSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_camera_tv_on_makes_media_playing_true(self):
        """A loaded skill_tv_detect whose is_tv_on() is True must make
        _ambient_media_is_playing() return True (suppress), independent of the
        audio gates."""
        bc = self.bc
        fake = types.ModuleType("skill_tv_detect")
        fake.is_tv_on = lambda: True
        with mock.patch.object(bc, "_smtc_media_playing", return_value=False), \
             mock.patch.dict(sys.modules, {"skill_tv_detect": fake,
                                           "skill_standby_audio_detect": None}):
            # Remove the audio detector so ONLY the camera signal can fire.
            sys.modules.pop("skill_standby_audio_detect", None)
            self.assertTrue(bc._ambient_media_is_playing())

    def test_camera_tv_off_does_not_force_true(self):
        """When the camera detector says no TV (and the audio gates are quiet),
        _ambient_media_is_playing() stays False — the camera signal only ADDS a
        veto, it never blocks the False path."""
        bc = self.bc
        fake = types.ModuleType("skill_tv_detect")
        fake.is_tv_on = lambda: False
        with mock.patch.object(bc, "_smtc_media_playing", return_value=False), \
             mock.patch.dict(sys.modules, {"skill_tv_detect": fake}):
            sys.modules.pop("skill_standby_audio_detect", None)
            self.assertFalse(bc._ambient_media_is_playing())

    def test_missing_tv_skill_is_safe(self):
        """No skill_tv_detect loaded → the probe is a no-op (no raise), gate
        falls through to its existing audio-only behaviour."""
        bc = self.bc
        with mock.patch.object(bc, "_smtc_media_playing", return_value=False):
            sys.modules.pop("skill_tv_detect", None)
            sys.modules.pop("skill_standby_audio_detect", None)
            self.assertFalse(bc._ambient_media_is_playing())


if __name__ == "__main__":
    unittest.main()
