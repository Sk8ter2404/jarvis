"""Unit tests for core/voice_clone.py — the local Chatterbox voice-clone
wrapper.

Everything heavy (chatterbox, torch, CUDA) is MOCKED or its import failure is
simulated, so this whole module runs on a headless / no-GPU / no-chatterbox CI
box (the model is never imported). We exercise the pure surface — the profile
registry, the consent gate, the selection logic, and the fallback decision —
plus the two heavy seams (is_available / synthesize) with the engine faked.

stdlib unittest + unittest.mock only (no pytest). Profiles live in a per-test
tempdir (PROFILES_DIR is patched) so nothing touches the real
data/voice_profiles/.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from core import voice_clone as vc


def _write_profile(root: str, name: str, *, consent, source, with_wav=True):
    """Create data/voice_profiles/<name>/ under `root` with a meta.json (and
    optionally a reference.wav). Returns the profile dir."""
    pdir = os.path.join(root, name)
    os.makedirs(pdir, exist_ok=True)
    if with_wav:
        with open(os.path.join(pdir, "reference.wav"), "wb") as f:
            f.write(b"RIFF....WAVEfmt ")   # dummy bytes; never decoded in tests
    meta = {"name": name, "created_at": "2026-07-07T00:00:00",
            "source": source}
    if consent is not None:
        meta["consent"] = consent
    with open(os.path.join(pdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return pdir


class ProfileRegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        # Point the module at our tempdir for the duration of the test.
        self._patch = mock.patch.object(vc, "PROFILES_DIR", self.root)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_load_profile_reads_meta_and_folds_in_paths(self):
        _write_profile(self.root, "me", consent=True, source="owner")
        meta = vc.load_profile("me")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["name"], "me")
        self.assertEqual(meta["source"], "owner")
        self.assertTrue(meta["reference_wav"].endswith("reference.wav"))
        self.assertTrue(os.path.isfile(meta["reference_wav"]))

    def test_load_profile_missing_returns_none(self):
        self.assertIsNone(vc.load_profile("nope"))

    def test_load_profile_bad_name_returns_none(self):
        self.assertIsNone(vc.load_profile(""))
        self.assertIsNone(vc.load_profile(None))  # type: ignore[arg-type]

    def test_load_profile_corrupt_meta_returns_none(self):
        pdir = os.path.join(self.root, "broken")
        os.makedirs(pdir)
        with open(os.path.join(pdir, "meta.json"), "w", encoding="utf-8") as f:
            f.write("{ not json")
        self.assertIsNone(vc.load_profile("broken"))

    def test_list_profiles_sorted_and_skips_dotdirs(self):
        _write_profile(self.root, "bravo", consent=True, source="owner")
        _write_profile(self.root, "alpha", consent=True, source="character")
        os.makedirs(os.path.join(self.root, "_hidden"))   # skipped
        names = [m["name"] for m in vc.list_profiles()]
        self.assertEqual(names, ["alpha", "bravo"])

    def test_list_profiles_missing_dir_returns_empty(self):
        with mock.patch.object(vc, "PROFILES_DIR", os.path.join(self.root, "gone")):
            self.assertEqual(vc.list_profiles(), [])


class ConsentGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self._patch = mock.patch.object(vc, "PROFILES_DIR", self.root)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_usable_owner_profile_passes(self):
        _write_profile(self.root, "me", consent=True, source="owner")
        self.assertTrue(vc.profile_is_usable(vc.load_profile("me")))

    def test_usable_character_profile_passes(self):
        _write_profile(self.root, "jarvis", consent=True, source="character")
        self.assertTrue(vc.profile_is_usable(vc.load_profile("jarvis")))

    def test_no_consent_flag_is_refused(self):
        _write_profile(self.root, "sneaky", consent=None, source="owner")
        self.assertFalse(vc.profile_is_usable(vc.load_profile("sneaky")))

    def test_consent_false_is_refused(self):
        _write_profile(self.root, "nope", consent=False, source="owner")
        self.assertFalse(vc.profile_is_usable(vc.load_profile("nope")))

    def test_truthy_but_not_true_consent_is_refused(self):
        # Guard the literal-True check: "yes"/1 must NOT count as consent.
        _write_profile(self.root, "truthy", consent="yes", source="owner")
        self.assertFalse(vc.profile_is_usable(vc.load_profile("truthy")))

    def test_disallowed_source_is_refused(self):
        # A celebrity / real-person source is out of scope even WITH consent.
        _write_profile(self.root, "celeb", consent=True, source="real_actor")
        self.assertFalse(vc.profile_is_usable(vc.load_profile("celeb")))

    def test_missing_reference_wav_is_refused(self):
        _write_profile(self.root, "nowav", consent=True, source="owner",
                       with_wav=False)
        self.assertFalse(vc.profile_is_usable(vc.load_profile("nowav")))

    def test_non_dict_is_refused(self):
        self.assertFalse(vc.profile_is_usable(None))
        self.assertFalse(vc.profile_is_usable("not a dict"))  # type: ignore[arg-type]


class ResolveActiveProfileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self._patch = mock.patch.object(vc, "PROFILES_DIR", self.root)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._tmp.cleanup)
        _write_profile(self.root, "me", consent=True, source="owner")
        _write_profile(self.root, "unconsented", consent=None, source="owner")

    def test_disabled_returns_none(self):
        self.assertIsNone(vc.resolve_active_profile(False, "me"))

    def test_no_profile_name_returns_none(self):
        self.assertIsNone(vc.resolve_active_profile(True, ""))

    def test_unknown_profile_returns_none(self):
        self.assertIsNone(vc.resolve_active_profile(True, "ghost"))

    def test_unconsented_profile_returns_none(self):
        self.assertIsNone(vc.resolve_active_profile(True, "unconsented"))

    def test_enabled_consented_returns_meta(self):
        meta = vc.resolve_active_profile(True, "me")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["name"], "me")


class IsAvailableTests(unittest.TestCase):
    """is_available() must fail CLOSED for every missing precondition."""

    def _usable_profile(self):
        return {"name": "me", "source": "owner", "consent": True,
                "reference_wav": __file__}   # any existing file passes the wav check

    def test_false_when_disabled(self):
        with mock.patch.object(vc, "_cfg_enabled", return_value=False):
            self.assertFalse(vc.is_available())

    def test_false_when_no_usable_profile(self):
        with mock.patch.object(vc, "_cfg_enabled", return_value=True), \
             mock.patch.object(vc, "resolve_active_profile", return_value=None):
            self.assertFalse(vc.is_available())

    def test_false_when_package_missing(self):
        # The key CI scenario: chatterbox not importable → is_available False.
        with mock.patch.object(vc, "_cfg_enabled", return_value=True), \
             mock.patch.object(vc, "resolve_active_profile",
                               return_value=self._usable_profile()), \
             mock.patch.object(vc, "_chatterbox_importable", return_value=False), \
             mock.patch.object(vc, "_cuda_available", return_value=True):
            self.assertFalse(vc.is_available())

    def test_false_when_no_cuda(self):
        with mock.patch.object(vc, "_cfg_enabled", return_value=True), \
             mock.patch.object(vc, "resolve_active_profile",
                               return_value=self._usable_profile()), \
             mock.patch.object(vc, "_chatterbox_importable", return_value=True), \
             mock.patch.object(vc, "_cuda_available", return_value=False):
            self.assertFalse(vc.is_available())

    def test_true_when_all_present(self):
        with mock.patch.object(vc, "_cfg_enabled", return_value=True), \
             mock.patch.object(vc, "resolve_active_profile",
                               return_value=self._usable_profile()), \
             mock.patch.object(vc, "_chatterbox_importable", return_value=True), \
             mock.patch.object(vc, "_cuda_available", return_value=True):
            self.assertTrue(vc.is_available())

    def test_exception_fails_closed(self):
        with mock.patch.object(vc, "_cfg_enabled", side_effect=RuntimeError("boom")):
            self.assertFalse(vc.is_available())


class ChatterboxImportableTests(unittest.TestCase):
    def test_false_when_find_spec_none(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(vc._chatterbox_importable())

    def test_true_when_find_spec_present(self):
        with mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertTrue(vc._chatterbox_importable())

    def test_false_when_find_spec_raises(self):
        with mock.patch("importlib.util.find_spec", side_effect=ImportError):
            self.assertFalse(vc._chatterbox_importable())


class CudaAvailableTests(unittest.TestCase):
    def test_false_when_torch_absent(self):
        # Simulate `import torch` failing.
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertFalse(vc._cuda_available())


class SynthesizeFallbackTests(unittest.TestCase):
    """synthesize() must return None (→ caller falls back) on every failure
    path, and only return a waveform when the engine renders one."""

    def _usable_profile(self):
        return {"name": "me", "source": "owner", "consent": True,
                "reference_wav": __file__}

    def test_empty_text_returns_none(self):
        self.assertIsNone(vc.synthesize("   ", self._usable_profile()))

    def test_unusable_profile_returns_none(self):
        bad = {"name": "x", "source": "owner", "consent": False,
               "reference_wav": __file__}
        self.assertIsNone(vc.synthesize("hello", bad))

    def test_package_missing_returns_none(self):
        with mock.patch.object(vc, "_chatterbox_importable", return_value=False), \
             mock.patch.object(vc, "_cuda_available", return_value=True):
            self.assertIsNone(vc.synthesize("hello", self._usable_profile()))

    def test_no_cuda_returns_none(self):
        with mock.patch.object(vc, "_chatterbox_importable", return_value=True), \
             mock.patch.object(vc, "_cuda_available", return_value=False):
            self.assertIsNone(vc.synthesize("hello", self._usable_profile()))

    def test_engine_load_failure_returns_none(self):
        with mock.patch.object(vc, "_chatterbox_importable", return_value=True), \
             mock.patch.object(vc, "_cuda_available", return_value=True), \
             mock.patch.object(vc, "_load_engine", side_effect=RuntimeError("no model")):
            self.assertIsNone(vc.synthesize("hello", self._usable_profile()))

    @unittest.skipIf(vc.np is None, "numpy required for the render-path assertion")
    def test_successful_render_returns_waveform(self):
        import numpy as np
        fake_model = mock.MagicMock()
        fake_model.sr = 22050
        fake_model.generate.return_value = np.zeros(2048, dtype=np.float32)
        with mock.patch.object(vc, "_chatterbox_importable", return_value=True), \
             mock.patch.object(vc, "_cuda_available", return_value=True), \
             mock.patch.object(vc, "_load_engine", return_value=fake_model):
            out = vc.synthesize("hello sir", self._usable_profile())
        self.assertIsNotNone(out)
        audio, sr = out
        self.assertEqual(sr, 22050)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        fake_model.generate.assert_called_once()

    @unittest.skipIf(vc.np is None, "numpy required")
    def test_render_exception_returns_none(self):
        fake_model = mock.MagicMock()
        fake_model.generate.side_effect = RuntimeError("cuda oom")
        with mock.patch.object(vc, "_chatterbox_importable", return_value=True), \
             mock.patch.object(vc, "_cuda_available", return_value=True), \
             mock.patch.object(vc, "_load_engine", return_value=fake_model):
            self.assertIsNone(vc.synthesize("hello", self._usable_profile()))

    @unittest.skipIf(vc.np is None, "numpy required")
    def test_2d_output_is_squeezed_to_mono(self):
        import numpy as np
        fake_model = mock.MagicMock()
        fake_model.sr = 24000
        fake_model.generate.return_value = np.zeros((1, 1000), dtype=np.float32)
        with mock.patch.object(vc, "_chatterbox_importable", return_value=True), \
             mock.patch.object(vc, "_cuda_available", return_value=True), \
             mock.patch.object(vc, "_load_engine", return_value=fake_model):
            out = vc.synthesize("hi", self._usable_profile())
        self.assertIsNotNone(out)
        audio, sr = out
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.shape[0], 1000)


class EngineCacheTests(unittest.TestCase):
    def test_reset_engine_cache_clears(self):
        vc._engine_cache[0] = object()
        vc._engine_key[0] = ("chatterbox", "me")
        vc._reset_engine_cache()
        self.assertIsNone(vc._engine_cache[0])
        self.assertIsNone(vc._engine_key[0])

    def test_load_engine_caches_by_key(self):
        # Fake the heavy import so _load_engine runs without chatterbox.
        fake_model = mock.MagicMock()
        fake_cls = mock.MagicMock()
        fake_cls.from_pretrained.return_value = fake_model
        fake_mod = mock.MagicMock()
        fake_mod.ChatterboxTTS = fake_cls
        self.addCleanup(vc._reset_engine_cache)
        with mock.patch.dict(sys.modules, {"chatterbox": mock.MagicMock(),
                                           "chatterbox.tts": fake_mod}), \
             mock.patch.object(vc, "_cuda_available", return_value=False):
            m1 = vc._load_engine({"name": "me"})
            m2 = vc._load_engine({"name": "me"})
        self.assertIs(m1, fake_model)
        self.assertIs(m2, fake_model)
        # Cached: from_pretrained only called once for the same key.
        fake_cls.from_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
