"""Tests for core/tts_render_cache — the Kokoro render cache (speed plan R4).

Pins: the key moves with speed / voice / language / text / model and voices
files (and the text is never stored); the byte cap holds with least-recently-
used eviction; get() and put() copy (callers scale audio in place); a corrupt
persisted entry is ignored; 'off' never consults the cache, 'shadow' serves
nothing, 'on' serves a repeat without rendering; prefill_openers stops when
told and runs one at a time. Stdlib + numpy only (CI-light tier).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from core import config
from core import kokoro_tts as k
from core import tts_render_cache as rc


def _tmpdir(test) -> str:
    d = tempfile.mkdtemp(prefix="jarvis_tts_cache_test_")
    test.addCleanup(shutil.rmtree, d, True)
    return d


def _model_files(test):
    """A fake model + voices file pair; returns their paths."""
    d = _tmpdir(test)
    model, voices = os.path.join(d, "kokoro.onnx"), os.path.join(d, "voices.bin")
    with open(model, "wb") as f:
        f.write(b"m" * 2048)
    with open(voices, "wb") as f:
        f.write(b"v" * 1024)
    return model, voices


def _patch(test, *patchers):
    for p in patchers:
        p.start()
        test.addCleanup(p.stop)


def _key(i: int) -> str:
    return f"{i:064x}"


class FlagTests(unittest.TestCase):
    def test_flags_ship_with_todays_behaviour(self):
        self.assertIs(config.KOKORO_PERSISTENT_PHONEMIZER, False)
        self.assertEqual(config.KOKORO_RENDER_CACHE, "off")
        self.assertEqual(config.KOKORO_RENDER_CACHE_MB, 64)
        self.assertIs(config.KOKORO_RENDER_CACHE_PERSIST, False)

    def test_unknown_mode_is_off(self):
        for raw, want in (("ON", "on"), (" shadow ", "shadow"), ("off", "off"),
                          ("yes", "off"), ("", "off"), (None, "off")):
            with mock.patch.object(config, "KOKORO_RENDER_CACHE", raw):
                self.assertEqual(rc.mode(), want, raw)


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.model, self.voices = _model_files(self)

    def key(self, text="Of course, sir.", speed=1.0, voice="bm_george",
            lang="en-gb"):
        return rc.make_key(text, speed, voice, lang, self.model, self.voices)

    def test_key_is_a_hash_and_stable(self):
        a = self.key()
        self.assertRegex(a, r"^[0-9a-f]{64}$")
        self.assertEqual(a, self.key())
        self.assertNotIn("course", a)

    def test_key_changes_with_speed_voice_lang_and_text(self):
        base = self.key()
        self.assertNotEqual(base, self.key(speed=1.1))
        self.assertNotEqual(base, self.key(voice="bf_emma"))
        self.assertNotEqual(base, self.key(lang="en-us"))
        self.assertNotEqual(base, self.key(text="Of course, sir!"))
        # speed is rounded to 3 dp
        self.assertEqual(base, self.key(speed=1.0004))

    def test_key_changes_with_model_and_voices_files(self):
        base = self.key()
        st = os.stat(self.model)
        os.utime(self.model, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        after_model_mtime = self.key()
        self.assertNotEqual(base, after_model_mtime)
        st = os.stat(self.voices)
        os.utime(self.voices, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        after_voices_mtime = self.key()
        self.assertNotEqual(after_model_mtime, after_voices_mtime)
        st = os.stat(self.model)
        with open(self.model, "ab") as f:          # same mtime, new size
            f.write(b"x")
        os.utime(self.model, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertNotEqual(after_voices_mtime, self.key())

    def test_no_key_without_the_model(self):
        self.assertIsNone(rc.make_key("hi", 1.0, "v", "en-gb",
                                      self.model + ".missing", self.voices))


class MemoryCacheTests(unittest.TestCase):
    def setUp(self):
        _patch(self, mock.patch.object(config, "KOKORO_RENDER_CACHE_PERSIST",
                                       False))
        self.cache = rc.RenderCache(async_writes=False)

    def test_byte_cap_holds_with_lru_eviction(self):
        cap = 1024 * 1024
        _patch(self, mock.patch.object(config, "KOKORO_RENDER_CACHE_MB", 1))
        a = np.zeros(100_000, dtype=np.float32)        # 400 kB each
        self.assertTrue(self.cache.put(_key(1), a))
        self.assertTrue(self.cache.put(_key(2), a))
        self.assertIsNotNone(self.cache.get(_key(1)))  # 1 is now most recent
        self.assertTrue(self.cache.put(_key(3), a))    # 1.2 MB > cap: evict 2
        self.assertIsNotNone(self.cache.get(_key(1)))
        self.assertIsNone(self.cache.get(_key(2)))
        self.assertIsNotNone(self.cache.get(_key(3)))
        self.assertLessEqual(self.cache.stats()["bytes"], cap)
        self.assertEqual(self.cache.stats()["entries"], 2)
        # one entry bigger than the whole cap is never stored
        self.assertFalse(self.cache.put(_key(4), np.zeros(300_000, np.float32)))
        self.assertIsNone(self.cache.get(_key(4)))
        self.assertLessEqual(self.cache.stats()["bytes"], cap)

    def test_stores_float32_mono(self):
        self.cache.put(_key(1), np.ones((1, 50), dtype=np.float64))
        got = self.cache.get(_key(1))
        self.assertEqual(got.dtype, np.float32)
        self.assertEqual(got.shape, (50,))

    def test_mutating_a_returned_array_does_not_change_the_cache(self):
        self.cache.put(_key(1), np.full(10, 0.5, dtype=np.float32))
        got = self.cache.get(_key(1))
        got *= 0.0                                   # a caller scaling in place
        self.assertTrue(np.all(self.cache.get(_key(1)) == 0.5))

    def test_mutating_the_stored_array_does_not_change_the_cache(self):
        src = np.full(10, 0.5, dtype=np.float32)
        self.cache.put(_key(1), src)
        src *= 0.0
        self.assertTrue(np.all(self.cache.get(_key(1)) == 0.5))

    def test_bad_keys_are_refused(self):
        a = np.ones(4, dtype=np.float32)
        for bad in (None, "", "../../evil", "A" * 64, _key(1) + "0"):
            self.assertFalse(self.cache.put(bad, a), bad)
            self.assertIsNone(self.cache.get(bad), bad)
            self.assertFalse(self.cache.contains(bad), bad)
        self.assertFalse(self.cache.put(_key(1), np.zeros(0, np.float32)))


class PersistTests(unittest.TestCase):
    def setUp(self):
        self.data = _tmpdir(self)
        _patch(self,
               mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.data}),
               mock.patch.object(config, "KOKORO_RENDER_CACHE_PERSIST", True))
        self.dir = os.path.join(self.data, rc.SUBDIR)

    def test_round_trip_through_npy_files(self):
        rc.RenderCache(async_writes=False).put(
            _key(7), np.full(32, 0.25, dtype=np.float32))
        self.assertEqual(os.listdir(self.dir), [_key(7) + ".npy"])
        fresh = rc.RenderCache(async_writes=False)     # a restart
        self.assertTrue(fresh.contains(_key(7)))
        got = fresh.get(_key(7))
        self.assertTrue(np.all(got == 0.25))
        self.assertEqual(got.dtype, np.float32)

    def test_corrupt_or_foreign_entries_are_ignored(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, _key(1) + ".npy"), "wb") as f:
            f.write(b"\x93NUMPY garbage, not an array")
        np.save(os.path.join(self.dir, _key(2) + ".npy"),
                np.ones(8, dtype=np.float64))          # wrong dtype
        np.save(os.path.join(self.dir, _key(3) + ".npy"),
                np.ones((2, 8), dtype=np.float32))     # not mono
        np.save(os.path.join(self.dir, _key(4) + ".npy"),
                np.array([{"a": 1}], dtype=object), allow_pickle=True)
        cache = rc.RenderCache(async_writes=False)
        for i in (1, 2, 3, 4):
            self.assertIsNone(cache.get(_key(i)), i)
            self.assertFalse(cache.contains(_key(i)), i)
        # a fresh render replaces the corrupt file
        cache.put(_key(1), np.full(8, 0.5, dtype=np.float32))
        self.assertTrue(np.all(rc.RenderCache().get(_key(1)) == 0.5))

    def test_disk_is_trimmed_to_the_cap_oldest_first(self):
        _patch(self, mock.patch.object(config, "KOKORO_RENDER_CACHE_MB", 1))
        cache = rc.RenderCache(async_writes=False)
        a = np.zeros(100_000, dtype=np.float32)
        for i in (1, 2, 3):
            cache.put(_key(i), a)
            p = os.path.join(self.dir, _key(i) + ".npy")
            os.utime(p, ns=(i * 10**9, i * 10**9))     # strictly ordered
        names = sorted(os.listdir(self.dir))
        self.assertEqual(names, [_key(2) + ".npy", _key(3) + ".npy"])

    def test_nothing_written_when_persist_is_off(self):
        with mock.patch.object(config, "KOKORO_RENDER_CACHE_PERSIST", False):
            rc.RenderCache(async_writes=False).put(
                _key(1), np.ones(8, dtype=np.float32))
        self.assertFalse(os.path.exists(self.dir))


class SynthesizeCacheTests(unittest.TestCase):
    """The lookup inside core/kokoro_tts.synthesize()."""

    def setUp(self):
        model, voices = _model_files(self)
        self.eng = mock.Mock()
        self.eng.create.side_effect = lambda *a, **kw: (
            np.full(240, 0.25, dtype=np.float32), 24000)
        _patch(self,
               mock.patch.object(k, "_MODEL", model),
               mock.patch.object(k, "_VOICES", voices),
               mock.patch.object(k, "is_available", return_value=True),
               mock.patch.object(k, "_engine", return_value=self.eng),
               mock.patch.object(config, "KOKORO_RENDER_CACHE_PERSIST", False))
        rc.CACHE.clear()
        self.addCleanup(rc.CACHE.clear)

    def test_off_never_consults_the_cache(self):
        spies = {name: mock.Mock(side_effect=AssertionError(name))
                 for name in ("lookup", "get", "put", "contains")}
        with mock.patch.object(config, "KOKORO_RENDER_CACHE", "off"), \
             mock.patch.object(rc, "make_key",
                               side_effect=AssertionError("make_key")), \
             mock.patch.multiple(rc.CACHE, **spies):
            for _ in range(2):
                audio, sr = k.synthesize("Of course, sir.")
                self.assertTrue(np.all(audio == 0.25))
            self.assertFalse(k.fill_cache("Of course, sir."))
        self.assertEqual(self.eng.create.call_count, 2)
        for name, spy in spies.items():
            spy.assert_not_called()

    def test_shadow_serves_nothing(self):
        with mock.patch.object(config, "KOKORO_RENDER_CACHE", "shadow"):
            key = rc.make_key("Of course, sir.", 1.0, k._VOICE, k._LANG,
                              k._MODEL, k._VOICES)
            rc.CACHE.put(key, np.full(240, 0.9, dtype=np.float32))
            for _ in range(2):
                audio, sr = k.synthesize("Of course, sir.")
                self.assertTrue(np.all(audio == 0.25), "a render, not the cache")
        self.assertEqual(self.eng.create.call_count, 2)
        st = rc.CACHE.stats()
        self.assertEqual((st["hits"], st["misses"]), (2, 0))

    def test_shadow_counts_misses_and_fills(self):
        with mock.patch.object(config, "KOKORO_RENDER_CACHE", "shadow"):
            k.synthesize("Right away, sir.")
            k.synthesize("Right away, sir.")
        self.assertEqual(self.eng.create.call_count, 2)
        st = rc.CACHE.stats()
        self.assertEqual((st["hits"], st["misses"], st["entries"]), (1, 1, 1))

    def test_on_serves_a_repeat_without_rendering(self):
        with mock.patch.object(config, "KOKORO_RENDER_CACHE", "on"):
            first, _ = k.synthesize("Of course, sir.")
            first *= 0.0                       # the caller scales in place
            second, sr = k.synthesize("Of course, sir.")
            second *= 0.0
            third, _ = k.synthesize("Of course, sir.")
            other, _ = k.synthesize("Of course, sir.", speed=1.1)
        self.assertEqual(sr, 24000)
        self.assertTrue(np.all(third == 0.25))
        self.assertTrue(np.all(other == 0.25))
        self.assertEqual(self.eng.create.call_count, 2)   # 1.0 once, 1.1 once

    def test_a_failed_render_is_not_cached(self):
        self.eng.create.side_effect = RuntimeError("onnx boom")
        with mock.patch.object(config, "KOKORO_RENDER_CACHE", "on"):
            self.assertIsNone(k.synthesize("Of course, sir."))
        self.assertEqual(rc.CACHE.stats()["entries"], 0)


class PrefillTests(unittest.TestCase):
    def setUp(self):
        _patch(self, mock.patch.object(config, "KOKORO_RENDER_CACHE", "on"))

    def test_renders_every_opener_into_the_cache_once(self):
        model, voices = _model_files(self)
        eng = mock.Mock()
        eng.create.side_effect = lambda *a, **kw: (
            np.full(24, 0.25, dtype=np.float32), 24000)
        _patch(self,
               mock.patch.object(k, "_MODEL", model),
               mock.patch.object(k, "_VOICES", voices),
               mock.patch.object(k, "is_available", return_value=True),
               mock.patch.object(k, "_engine", return_value=eng),
               mock.patch.object(config, "KOKORO_RENDER_CACHE_PERSIST", False))
        rc.CACHE.clear()
        self.addCleanup(rc.CACHE.clear)
        self.assertGreaterEqual(len(rc.OPENERS), 20)
        self.assertEqual(rc.prefill_openers(threading.Event()), len(rc.OPENERS))
        self.assertEqual(eng.create.call_count, len(rc.OPENERS))
        self.assertEqual(rc.prefill_openers(None), 0)      # all cached already
        self.assertEqual(eng.create.call_count, len(rc.OPENERS))
        self.assertEqual(rc.CACHE.stats()["hits"] + rc.CACHE.stats()["misses"],
                         0, "prefill counts no lookups")
        self.assertIsNotNone(k.synthesize(rc.OPENERS[0]))   # served, not rendered
        self.assertEqual(eng.create.call_count, len(rc.OPENERS))

    def test_stops_when_told(self):
        stop = threading.Event()
        seen = []

        def fill(line, speed=1.0):
            seen.append(line)
            if len(seen) == 3:
                stop.set()
            return True

        with mock.patch.object(k, "fill_cache", side_effect=fill):
            self.assertEqual(rc.prefill_openers(stop), 3)
            seen.clear()
            self.assertEqual(rc.prefill_openers(lambda: len(seen) >= 2), 2)
            seen.clear()
            stop.set()
            self.assertEqual(rc.prefill_openers(stop), 0)
        self.assertEqual(seen, [])

    def test_single_flight_and_off(self):
        fill = mock.Mock(return_value=True)
        with mock.patch.object(k, "fill_cache", fill):
            with rc._PREFILL_LOCK:                     # one already running
                self.assertEqual(rc.prefill_openers(None), 0)
            with mock.patch.object(config, "KOKORO_RENDER_CACHE", "off"):
                self.assertEqual(rc.prefill_openers(None), 0)
        fill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
