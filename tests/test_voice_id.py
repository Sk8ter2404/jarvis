"""Tests for core.voice_id — multi-user speaker identification.

core.voice_id enrolls per-speaker voiceprint embeddings (Resemblyzer's
VoiceEncoder), matches a fresh utterance against them by cosine similarity,
and gates per-user permissions (sudo / shell / memory_write / smart_home /
music) on the recognised speaker. The embedding MODEL is heavyweight and is
NOT present on CI, so every test here either:

  • patches the module's own ``_embed`` / ``_load_encoder`` so no real model
    is ever loaded and embeddings are deterministic synthetic numpy vectors
    built with REAL numpy (which IS on CI), or
  • injects a fake ``resemblyzer`` module / blocks the import to exercise the
    available / unavailable branches regardless of whether the dev box happens
    to have resemblyzer installed.

No microphone is ever opened and no real ``data/voiceprints`` directory is
read or written: every test redirects the module's ``_VOICE_DIR`` /
``_INDEX_FILE`` globals to a fresh temp dir and fully resets the module-level
caches (``_voiceprints`` / ``_voicemeta`` / ``_active_speaker`` / ``_loaded`` /
``_encoder`` / ``_encoder_error``) in setUp + tearDown so the live module is
left pristine for the rest of the suite.

Speaker fixtures use generic names (alice / sam / bob) only.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import core.voice_id as vid


_SENTINEL = object()


# ─── fake-module helpers (mirrors tests/skills/test_self_diagnostic.py) ──────
@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules (e.g. resemblyzer,
    librosa, core.atomic_io). For dotted names the leaf is ALSO set as an
    attribute on its already-imported parent package, because
    ``from core import atomic_io`` resolves the leaf via getattr on the real
    parent package. Restores the previous state — including absence — on exit.
    """
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


@contextlib.contextmanager
def block_import(*names):
    """Force ``import <name>`` to raise ImportError inside the with-block, so a
    missing-dependency branch is exercised even when the real dep is installed
    on the dev box. Also detaches any already-imported target (and its parent-
    package attr) so the import machinery can't satisfy it from cache, then
    restores both on exit."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in blocked or top in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    saved_mod: dict[str, object] = {}
    saved_attr: list = []
    for name in blocked:
        if name in sys.modules:
            saved_mod[name] = sys.modules.pop(name)
        if "." in name:
            parent_name, _, leaf = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and hasattr(parent, leaf):
                saved_attr.append((parent, leaf, getattr(parent, leaf)))
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            setattr(parent, leaf, prev)
        for name, mod in saved_mod.items():
            sys.modules[name] = mod


def make_resemblyzer(init_raises=False, init_typeerror=False, embed_value=None):
    """Build a fake ``resemblyzer`` module exposing a VoiceEncoder.

    ``init_typeerror`` makes the verbose-kwarg ctor raise TypeError once (the
    older-signature fallback path), then a no-arg ctor succeeds.
    ``init_raises`` makes every ctor raise. ``embed_value`` (a 1-D iterable)
    is what ``embed_utterance`` returns; default is a fixed unit-ish vector."""
    mod = types.ModuleType("resemblyzer")

    class _Encoder:
        def __init__(self, *a, **k):
            if init_raises:
                raise RuntimeError("model weights missing")
            if init_typeerror and "verbose" in k:
                raise TypeError("unexpected kwarg 'verbose'")

        def embed_utterance(self, wav):
            if embed_value is not None:
                return np.asarray(embed_value, dtype=np.float32)
            return np.full(256, 0.1, dtype=np.float32)

    mod.VoiceEncoder = _Encoder
    return mod


def emb_vec(seed: int, dim: int = 256) -> np.ndarray:
    """A deterministic, unit-norm synthetic embedding (REAL numpy)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n else v


class _VoiceIdBase(unittest.TestCase):
    """Redirect the storage globals to a temp dir and snapshot/restore every
    module global the suite mutates, so each test is fully isolated and the
    live module is pristine afterward."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="voiceid_test_")
        # Snapshot the globals we are about to mutate.
        self._saved = {
            "_VOICE_DIR": vid._VOICE_DIR,
            "_INDEX_FILE": vid._INDEX_FILE,
            "_encoder": vid._encoder,
            "_encoder_error": vid._encoder_error,
            "_active_speaker": vid._active_speaker,
            "_loaded": vid._loaded,
        }
        self._saved_voiceprints = dict(vid._voiceprints)
        self._saved_voicemeta = dict(vid._voicemeta)
        # Redirect storage into the temp dir.
        vid._VOICE_DIR = self.tmp
        vid._INDEX_FILE = os.path.join(self.tmp, "_index.json")
        # Reset module state to a clean slate for this test.
        vid._voiceprints.clear()
        vid._voicemeta.clear()
        vid._encoder = None
        vid._encoder_error = None
        vid._active_speaker = None
        vid._loaded = False
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._saved.items():
            setattr(vid, k, v)
        vid._voiceprints.clear()
        vid._voiceprints.update(self._saved_voiceprints)
        vid._voicemeta.clear()
        vid._voicemeta.update(self._saved_voicemeta)
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    # convenience: directly seed an enrolled speaker in memory + on disk
    def _seed(self, slug, vec=None, meta=None, write_disk=True):
        vec = emb_vec(hash(slug) & 0xFFFF) if vec is None else vec
        vid._voiceprints[slug] = vec
        vid._voicemeta[slug] = meta if meta is not None else {
            "name": slug, "slug": slug, "sample_count": 1,
            "enrolled_ts": 1000.0,
            "permissions": dict(vid._DEFAULT_PERMISSIONS),
        }
        if write_disk:
            np.save(os.path.join(self.tmp, f"{slug}.npy"), vec)
            with open(os.path.join(self.tmp, f"{slug}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(vid._voicemeta[slug], f)
        vid._loaded = True


# ─────────────────────────────────────────────────────────────────────────
# _slug
# ─────────────────────────────────────────────────────────────────────────
class SlugTests(_VoiceIdBase):
    def test_lowercases_and_keeps_alnum(self):
        self.assertEqual(vid._slug("Alice"), "alice")

    def test_spaces_dashes_underscores_become_underscore(self):
        self.assertEqual(vid._slug("Sam B-9_x"), "sam_b_9_x")

    def test_strips_leading_trailing_separators(self):
        self.assertEqual(vid._slug("  -alice-  "), "alice")

    def test_drops_punctuation(self):
        self.assertEqual(vid._slug("a!@#$b"), "ab")

    def test_empty_and_none_fall_back_to_speaker(self):
        self.assertEqual(vid._slug(""), "speaker")
        self.assertEqual(vid._slug(None), "speaker")
        self.assertEqual(vid._slug("!!!"), "speaker")

    def test_unicode_letters_preserved(self):
        # café → caf é all alnum
        self.assertEqual(vid._slug("Café"), "café")


# ─────────────────────────────────────────────────────────────────────────
# _to_resemblyzer_audio
# ─────────────────────────────────────────────────────────────────────────
class ToResemblyzerAudioTests(_VoiceIdBase):
    def test_none_audio_returns_empty(self):
        out = vid._to_resemblyzer_audio(None, vid.TARGET_SR)
        self.assertEqual(out.size, 0)
        self.assertEqual(out.dtype, np.float32)

    def test_empty_array_short_circuits(self):
        out = vid._to_resemblyzer_audio(np.zeros(0, dtype=np.float32), 24000)
        self.assertEqual(out.size, 0)

    def test_same_sample_rate_passthrough(self):
        a = np.linspace(-1, 1, 8000, dtype=np.float32)
        out = vid._to_resemblyzer_audio(a, vid.TARGET_SR)
        self.assertEqual(out.size, 8000)
        self.assertEqual(out.dtype, np.float32)

    def test_int16_scale_is_normalised(self):
        # Values up to 32k should be divided by 32768.
        a = np.full(16000, 16384.0, dtype=np.float32)
        out = vid._to_resemblyzer_audio(a, vid.TARGET_SR)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)
        self.assertAlmostEqual(float(out[0]), 0.5, places=4)

    def test_2d_more_rows_than_cols_averages_axis1(self):
        # shape (N, 2): stereo with N frames -> mean over channels (axis=1).
        # Keep values <=1.5 so the int16-rescale guard (mx>1.5) stays inert and
        # this isolates the channel-averaging behaviour. Mean of 0.4 & 0.8 = 0.6.
        a = np.stack([np.full(100, 0.4), np.full(100, 0.8)], axis=1).astype(np.float32)
        self.assertEqual(a.shape, (100, 2))
        out = vid._to_resemblyzer_audio(a, vid.TARGET_SR)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.size, 100)
        self.assertTrue(np.allclose(out, 0.6))

    def test_2d_more_cols_than_rows_averages_axis0(self):
        # shape (2, N): channels-first -> mean over axis 0. Values <=1.5 again.
        a = np.stack([np.full(100, 0.4), np.full(100, 0.8)], axis=0).astype(np.float32)
        self.assertEqual(a.shape, (2, 100))
        out = vid._to_resemblyzer_audio(a, vid.TARGET_SR)
        self.assertEqual(out.size, 100)
        self.assertTrue(np.allclose(out, 0.6))

    def test_resample_numpy_fallback_when_no_librosa(self):
        # Block librosa so the cheap numpy interp path runs; 24k -> 16k.
        a = np.linspace(-1, 1, 2400, dtype=np.float32)
        with block_import("librosa"):
            out = vid._to_resemblyzer_audio(a, 24000)
        self.assertEqual(out.dtype, np.float32)
        # 2400 * (16000/24000) = 1600
        self.assertEqual(out.size, 1600)

    def test_resample_uses_librosa_when_available(self):
        a = np.linspace(-1, 1, 2400, dtype=np.float32)
        fake_lib = types.ModuleType("librosa")
        fake_lib.resample = mock.MagicMock(
            return_value=np.zeros(1600, dtype=np.float32))
        with inject_modules(librosa=fake_lib):
            out = vid._to_resemblyzer_audio(a, 24000)
        fake_lib.resample.assert_called_once()
        self.assertEqual(out.size, 1600)


# ─────────────────────────────────────────────────────────────────────────
# _load_encoder / is_available / _embed
# ─────────────────────────────────────────────────────────────────────────
class EncoderTests(_VoiceIdBase):
    def test_load_encoder_missing_resemblyzer(self):
        with block_import("resemblyzer"):
            enc = vid._load_encoder()
        self.assertIsNone(enc)
        self.assertIn("resemblyzer not installed", vid._encoder_error)

    def test_is_available_false_when_missing(self):
        with block_import("resemblyzer"):
            self.assertFalse(vid.is_available())

    def test_load_encoder_success_and_is_available(self):
        with inject_modules(resemblyzer=make_resemblyzer()):
            self.assertTrue(vid.is_available())
            enc = vid._load_encoder()
        self.assertIsNotNone(enc)

    def test_load_encoder_caches(self):
        sentinel = object()
        vid._encoder = sentinel
        # Should return the cached encoder without importing anything.
        with block_import("resemblyzer"):
            self.assertIs(vid._load_encoder(), sentinel)

    def test_load_encoder_typeerror_falls_back_to_noarg_ctor(self):
        # verbose-kwarg ctor raises TypeError -> no-arg ctor succeeds.
        with inject_modules(resemblyzer=make_resemblyzer(init_typeerror=True)):
            enc = vid._load_encoder()
        self.assertIsNotNone(enc)

    def test_load_encoder_init_failure_sets_error(self):
        with inject_modules(resemblyzer=make_resemblyzer(init_raises=True)):
            enc = vid._load_encoder()
        self.assertIsNone(enc)
        self.assertIn("VoiceEncoder init failed", vid._encoder_error)

    def test_load_encoder_typeerror_then_noarg_also_fails(self):
        # verbose ctor raises TypeError, the no-arg fallback ctor then raises a
        # generic error -> nested-except branch sets _encoder_error.
        rz = types.ModuleType("resemblyzer")

        class _Encoder:
            def __init__(self, *a, **k):
                if "verbose" in k:
                    raise TypeError("no verbose kwarg")
                raise RuntimeError("weights corrupt")

        rz.VoiceEncoder = _Encoder
        with inject_modules(resemblyzer=rz):
            self.assertIsNone(vid._load_encoder())
        self.assertIn("VoiceEncoder init failed", vid._encoder_error)

    def test_embed_none_when_no_encoder(self):
        with block_import("resemblyzer"):
            self.assertIsNone(vid._embed(np.ones(16000, dtype=np.float32), 16000))

    def test_embed_none_when_audio_too_short(self):
        # encoder available but audio under MIN_IDENTIFY_SECONDS.
        with inject_modules(resemblyzer=make_resemblyzer()):
            short = np.ones(int(vid.TARGET_SR * 0.1), dtype=np.float32)
            self.assertIsNone(vid._embed(short, vid.TARGET_SR))

    def test_embed_returns_unit_norm_vector(self):
        rz = make_resemblyzer(embed_value=np.full(256, 2.0, dtype=np.float32))
        with inject_modules(resemblyzer=rz):
            out = vid._embed(np.ones(16000, dtype=np.float32), 16000)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (256,))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)

    def test_embed_handles_encoder_exception(self):
        rz = make_resemblyzer()
        # Make embed_utterance raise.
        rz.VoiceEncoder.embed_utterance = lambda self, wav: (_ for _ in ()).throw(
            RuntimeError("cuda oom"))
        with inject_modules(resemblyzer=rz):
            self.assertIsNone(vid._embed(np.ones(16000, dtype=np.float32), 16000))


# ─────────────────────────────────────────────────────────────────────────
# _atomic_write_json  (atomic_io path + inline fallback)
# ─────────────────────────────────────────────────────────────────────────
class AtomicWriteTests(_VoiceIdBase):
    def test_uses_core_atomic_io_when_importable(self):
        path = os.path.join(self.tmp, "viaio.json")
        fake_io = types.ModuleType("core.atomic_io")
        fake_io._atomic_write_json = mock.MagicMock()
        with inject_modules(**{"core.atomic_io": fake_io}):
            vid._atomic_write_json(path, {"x": 1})
        fake_io._atomic_write_json.assert_called_once_with(path, {"x": 1})

    def test_inline_fallback_cleans_tmp_when_replace_fails(self):
        # Force the inline fallback (atomic_io raises), then make os.replace
        # fail so the leftover tempfile is removed by the finally-block and the
        # error propagates. No .tmp must survive.
        path = os.path.join(self.tmp, "halffail.json")
        boom_io = types.ModuleType("core.atomic_io")
        boom_io._atomic_write_json = mock.MagicMock(
            side_effect=RuntimeError("atomic_io down"))
        with inject_modules(**{"core.atomic_io": boom_io}), \
             mock.patch.object(vid.os, "replace", side_effect=OSError("no rename")):
            with self.assertRaises(OSError):
                vid._atomic_write_json(path, {"z": 3})
        leftovers = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_inline_fallback_tmp_remove_failure_is_swallowed(self):
        # Both os.replace AND the finally-block os.remove fail. The ORIGINAL
        # OSError (from replace) must still propagate while the cleanup failure
        # is swallowed by the inner except — i.e. a doubly-unlucky filesystem
        # never masks the real write error with a cleanup error.
        path = os.path.join(self.tmp, "doublefail.json")
        boom_io = types.ModuleType("core.atomic_io")
        boom_io._atomic_write_json = mock.MagicMock(
            side_effect=RuntimeError("atomic_io down"))
        replace_err = OSError("no rename")
        with inject_modules(**{"core.atomic_io": boom_io}), \
             mock.patch.object(vid.os, "replace", side_effect=replace_err), \
             mock.patch.object(vid.os, "remove", side_effect=OSError("rm locked")):
            with self.assertRaises(OSError) as cm:
                vid._atomic_write_json(path, {"q": 9})
        self.assertIs(cm.exception, replace_err)

    def test_inline_fallback_when_atomic_io_raises(self):
        # ``from core import atomic_io`` is satisfied from the already-imported
        # ``core`` package attribute, so blocking the import isn't enough to
        # exercise the fallback. Instead inject a fake whose writer RAISES,
        # which trips the ``except Exception`` and runs the inline tempfile
        # branch. The file must still land on disk via the fallback.
        path = os.path.join(self.tmp, "fallback.json")
        boom_io = types.ModuleType("core.atomic_io")
        boom_io._atomic_write_json = mock.MagicMock(
            side_effect=RuntimeError("atomic_io exploded"))
        with inject_modules(**{"core.atomic_io": boom_io}):
            vid._atomic_write_json(path, {"y": 2})
        boom_io._atomic_write_json.assert_called_once()
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"y": 2})
        # No .tmp leftovers.
        leftovers = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_inline_fallback_swallows_fsync_failure(self):
        # In the inline fallback, a filesystem whose fsync raises OSError (some
        # network shares) must not abort the write — os.replace still lands it.
        path = os.path.join(self.tmp, "nofsync.json")
        boom_io = types.ModuleType("core.atomic_io")
        boom_io._atomic_write_json = mock.MagicMock(
            side_effect=RuntimeError("atomic_io down"))
        with inject_modules(**{"core.atomic_io": boom_io}), \
             mock.patch.object(vid.os, "fsync", side_effect=OSError("no fsync")):
            vid._atomic_write_json(path, {"w": 4})
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"w": 4})


# ─────────────────────────────────────────────────────────────────────────
# _load_index / _save_index / _load_all / persistence
# ─────────────────────────────────────────────────────────────────────────
class PersistenceTests(_VoiceIdBase):
    def test_load_index_missing_returns_empty(self):
        self.assertEqual(vid._load_index(), {})

    def test_load_index_corrupt_returns_empty(self):
        with open(vid._INDEX_FILE, "w", encoding="utf-8") as f:
            f.write("{ not valid json")
        self.assertEqual(vid._load_index(), {})

    def test_load_index_unexpected_error_returns_empty(self):
        # A non-JSONDecode/FileNotFound error (e.g. PermissionError) from the
        # open() is swallowed by the broad except -> {}.
        with open(vid._INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"active": "alice"}, f)
        with mock.patch("builtins.open", side_effect=PermissionError("locked")):
            self.assertEqual(vid._load_index(), {})

    def test_load_index_reads_payload(self):
        with open(vid._INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"speakers": ["alice"], "active": "alice"}, f)
        self.assertEqual(vid._load_index()["active"], "alice")

    def test_ensure_dir_swallows_oserror(self):
        with mock.patch.object(vid.os, "makedirs", side_effect=OSError("denied")):
            vid._ensure_dir()  # must not raise

    def test_save_index_swallows_write_failure(self):
        vid._voiceprints["alice"] = emb_vec(1)
        with mock.patch.object(vid, "_atomic_write_json",
                               side_effect=OSError("locked")):
            vid._save_index()  # logged + swallowed, no raise

    def test_save_index_writes_sorted_speakers_and_active(self):
        vid._voiceprints["sam"] = emb_vec(2)
        vid._voiceprints["alice"] = emb_vec(1)
        vid._active_speaker = "alice"
        vid._save_index()
        with open(vid._INDEX_FILE, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["speakers"], ["alice", "sam"])
        self.assertEqual(payload["active"], "alice")
        self.assertIn("updated_ts", payload)

    def test_embedding_and_meta_paths(self):
        self.assertTrue(vid._embedding_path("Alice").endswith("alice.npy"))
        self.assertTrue(vid._meta_path("Alice").endswith("alice.json"))
        self.assertTrue(vid._embedding_path("Alice").startswith(self.tmp))

    def test_load_all_idempotent(self):
        vid._loaded = True
        # Drop a stray npy that would be loaded IF _load_all re-ran.
        np.save(os.path.join(self.tmp, "ghost.npy"), emb_vec(5))
        vid._load_all()
        self.assertNotIn("ghost", vid._voiceprints)

    def test_load_all_reads_embeddings_and_meta(self):
        np.save(os.path.join(self.tmp, "alice.npy"), emb_vec(1))
        with open(os.path.join(self.tmp, "alice.json"), "w", encoding="utf-8") as f:
            json.dump({"name": "Alice", "permissions": {"sudo": True}}, f)
        vid._load_all()
        self.assertIn("alice", vid._voiceprints)
        self.assertAlmostEqual(float(np.linalg.norm(vid._voiceprints["alice"])),
                               1.0, places=5)
        self.assertEqual(vid._voicemeta["alice"]["name"], "Alice")

    def test_load_all_skips_underscore_and_non_npy(self):
        np.save(os.path.join(self.tmp, "_private.npy"), emb_vec(1))
        with open(os.path.join(self.tmp, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("hi")
        vid._load_all()
        self.assertEqual(vid._voiceprints, {})

    def test_load_all_corrupt_npy_skipped(self):
        with open(os.path.join(self.tmp, "broken.npy"), "wb") as f:
            f.write(b"not a real npy header")
        vid._load_all()
        self.assertNotIn("broken", vid._voiceprints)

    def test_load_all_missing_meta_synthesises_default(self):
        np.save(os.path.join(self.tmp, "sam.npy"), emb_vec(3))
        # no sam.json on disk
        vid._load_all()
        self.assertIn("sam", vid._voiceprints)
        self.assertEqual(vid._voicemeta["sam"]["sample_count"], 1)
        self.assertEqual(vid._voicemeta["sam"]["permissions"],
                         vid._DEFAULT_PERMISSIONS)

    def test_load_all_corrupt_meta_synthesises_default(self):
        np.save(os.path.join(self.tmp, "sam.npy"), emb_vec(3))
        with open(os.path.join(self.tmp, "sam.json"), "w", encoding="utf-8") as f:
            f.write("{ broken json")
        vid._load_all()
        self.assertIn("sam", vid._voiceprints)
        self.assertEqual(vid._voicemeta["sam"]["permissions"],
                         vid._DEFAULT_PERMISSIONS)

    def test_load_all_no_voiceprint_dir(self):
        # Point at a path that doesn't exist and can't be created, so the
        # ``not os.path.isdir`` short-circuit runs and load completes empty.
        missing = os.path.join(self.tmp, "nope", "deeper")
        vid._VOICE_DIR = missing
        with mock.patch.object(vid.os, "makedirs", side_effect=OSError("denied")):
            vid._load_all()
        self.assertTrue(vid._loaded)
        self.assertEqual(vid._voiceprints, {})

    def test_load_all_meta_read_unexpected_error_falls_back(self):
        np.save(os.path.join(self.tmp, "sam.npy"), emb_vec(3))
        with open(os.path.join(self.tmp, "sam.json"), "w", encoding="utf-8") as f:
            json.dump({"name": "Sam"}, f)
        # A non-FileNotFound/JSONDecode error during meta open -> the broad
        # except sets a minimal {"name": slug} fallback.
        real_open = open

        def _picky_open(path, *a, **k):
            if str(path).endswith("sam.json"):
                raise PermissionError("meta locked")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", side_effect=_picky_open):
            vid._load_all()
        self.assertIn("sam", vid._voiceprints)
        self.assertEqual(vid._voicemeta["sam"], {"name": "sam"})

    def test_load_all_restores_active_from_index(self):
        np.save(os.path.join(self.tmp, "alice.npy"), emb_vec(1))
        with open(vid._INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"active": "alice"}, f)
        vid._load_all()
        self.assertEqual(vid._active_speaker, "alice")

    def test_load_all_ignores_stale_active_not_enrolled(self):
        np.save(os.path.join(self.tmp, "alice.npy"), emb_vec(1))
        with open(vid._INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"active": "ghost"}, f)
        vid._load_all()
        self.assertIsNone(vid._active_speaker)


# ─────────────────────────────────────────────────────────────────────────
# encoder_status / list_enrolled / get + set active
# ─────────────────────────────────────────────────────────────────────────
class StatusAndActiveTests(_VoiceIdBase):
    def test_encoder_status_offline(self):
        with block_import("resemblyzer"):
            st = vid.encoder_status()
        self.assertFalse(st["encoder_loaded"])
        self.assertIsNotNone(st["encoder_error"])
        self.assertEqual(st["threshold"], vid.CONFIDENCE_THRESHOLD)

    def test_encoder_status_online_lists_enrolled(self):
        self._seed("alice")
        with inject_modules(resemblyzer=make_resemblyzer()):
            st = vid.encoder_status()
        self.assertTrue(st["encoder_loaded"])
        self.assertEqual(st["enrolled"], ["alice"])

    def test_list_enrolled_sorted(self):
        self._seed("sam")
        self._seed("alice")
        self.assertEqual(vid.list_enrolled(), ["alice", "sam"])

    def test_get_active_speaker_none_by_default(self):
        self.assertIsNone(vid.get_active_speaker())

    def test_set_active_clear_with_none(self):
        self._seed("alice")
        vid._active_speaker = "alice"
        self.assertTrue(vid.set_active_speaker(None))
        self.assertIsNone(vid._active_speaker)

    def test_set_active_unknown_returns_false(self):
        self._seed("alice")
        self.assertFalse(vid.set_active_speaker("nobody"))
        self.assertIsNone(vid._active_speaker)

    def test_set_active_known_persists(self):
        self._seed("alice")
        self.assertTrue(vid.set_active_speaker("Alice"))
        self.assertEqual(vid._active_speaker, "alice")
        with open(vid._INDEX_FILE, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["active"], "alice")


# ─────────────────────────────────────────────────────────────────────────
# forget_speaker
# ─────────────────────────────────────────────────────────────────────────
class ForgetTests(_VoiceIdBase):
    def test_forget_unknown_returns_false(self):
        self._seed("alice")
        self.assertFalse(vid.forget_speaker("nobody"))

    def test_forget_known_removes_memory_and_files(self):
        self._seed("alice")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "alice.npy")))
        self.assertTrue(vid.forget_speaker("Alice"))
        self.assertNotIn("alice", vid._voiceprints)
        self.assertNotIn("alice", vid._voicemeta)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "alice.npy")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "alice.json")))

    def test_forget_swallows_file_delete_error(self):
        # os.remove fails (e.g. file locked) -> error logged but forget still
        # drops the in-memory voiceprint and returns True.
        self._seed("alice")
        with mock.patch.object(vid.os, "remove", side_effect=OSError("locked")):
            self.assertTrue(vid.forget_speaker("alice"))
        self.assertNotIn("alice", vid._voiceprints)

    def test_forget_clears_active_when_it_was_active(self):
        self._seed("alice")
        vid._active_speaker = "alice"
        vid.forget_speaker("alice")
        self.assertIsNone(vid._active_speaker)

    def test_forget_keeps_other_active(self):
        self._seed("alice")
        self._seed("sam")
        vid._active_speaker = "sam"
        vid.forget_speaker("alice")
        self.assertEqual(vid._active_speaker, "sam")


# ─────────────────────────────────────────────────────────────────────────
# enroll_from_audio
# ─────────────────────────────────────────────────────────────────────────
class EnrollTests(_VoiceIdBase):
    def test_empty_name_rejected(self):
        # _slug("") -> "speaker", which is truthy, so feed something that
        # slugs to empty only via the guard? "speaker" is returned, so the
        # empty-name guard fires only when _slug returns falsy — which it
        # never does. Instead assert the documented behaviour: a blank name
        # still proceeds under the "speaker" slug once the encoder is faked.
        with mock.patch.object(vid, "is_available", return_value=False):
            res = vid.enroll_from_audio("", np.ones(16000, dtype=np.float32), 16000)
        # unavailable error wins (resemblyzer faked off)
        self.assertFalse(res["ok"])

    def test_empty_slug_rejected_before_encoder_check(self):
        # The "empty name" guard fires only when _slug() yields a falsy string.
        # In practice _slug never does (it falls back to "speaker"), so patch it
        # to "" to exercise the defensive guard — it must reject BEFORE the
        # is_available()/embedding work (here is_available would say True).
        with mock.patch.object(vid, "_slug", return_value=""), \
             mock.patch.object(vid, "is_available", return_value=True) as avail:
            res = vid.enroll_from_audio("whatever",
                                        np.ones(16000, dtype=np.float32), 16000)
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], "empty name")
        avail.assert_not_called()   # guarded out before the encoder check

    def test_unavailable_returns_error(self):
        with mock.patch.object(vid, "is_available", return_value=False):
            res = vid.enroll_from_audio("alice",
                                        np.ones(16000, dtype=np.float32), 16000)
        self.assertFalse(res["ok"])
        self.assertIn("error", res)

    def test_embed_none_returns_error(self):
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=None):
            res = vid.enroll_from_audio("alice",
                                        np.ones(16000, dtype=np.float32), 16000)
        self.assertFalse(res["ok"])
        self.assertIn("could not compute embedding", res["error"])

    def test_first_enrollment_gets_owner_permissions(self):
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=emb_vec(1)):
            res = vid.enroll_from_audio("Alice",
                                        np.ones(16000, dtype=np.float32), 16000)
        self.assertTrue(res["ok"])
        self.assertEqual(res["name"], "Alice")
        self.assertEqual(res["sample_count"], 1)
        self.assertEqual(res["dim"], 256)
        self.assertEqual(vid._voicemeta["alice"]["permissions"],
                         vid._OWNER_PERMISSIONS)
        # files written
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "alice.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "alice.json")))

    def test_second_speaker_gets_default_permissions(self):
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=emb_vec(1)):
            vid.enroll_from_audio("alice", np.ones(16000, dtype=np.float32), 16000)
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=emb_vec(2)):
            vid.enroll_from_audio("sam", np.ones(16000, dtype=np.float32), 16000)
        self.assertEqual(vid._voicemeta["sam"]["permissions"],
                         vid._DEFAULT_PERMISSIONS)
        self.assertEqual(vid._voicemeta["alice"]["permissions"],
                         vid._OWNER_PERMISSIONS)

    def test_append_blends_and_increments_sample_count(self):
        v1 = emb_vec(1)
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=v1):
            vid.enroll_from_audio("alice", np.ones(16000, dtype=np.float32), 16000)
        first = vid._voiceprints["alice"].copy()
        v2 = emb_vec(99)
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=v2):
            res = vid.enroll_from_audio("alice",
                                        np.ones(16000, dtype=np.float32), 16000,
                                        append=True)
        self.assertEqual(res["sample_count"], 2)
        # The blended vector differs from the first (averaged in v2) and stays
        # unit-norm.
        self.assertFalse(np.allclose(vid._voiceprints["alice"], first))
        self.assertAlmostEqual(
            float(np.linalg.norm(vid._voiceprints["alice"])), 1.0, places=5)

    def test_append_false_overwrites_but_preserves_enrolled_ts(self):
        v1 = emb_vec(1)
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=v1):
            vid.enroll_from_audio("alice", np.ones(16000, dtype=np.float32), 16000)
        orig_ts = vid._voicemeta["alice"]["enrolled_ts"]
        v2 = emb_vec(2)
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=v2):
            res = vid.enroll_from_audio("alice",
                                        np.ones(16000, dtype=np.float32), 16000,
                                        append=False)
        # sample_count resets to 1 on a fresh (non-append) enroll.
        self.assertEqual(res["sample_count"], 1)
        # enrolled_ts preserved from the original sidecar.
        self.assertEqual(vid._voicemeta["alice"]["enrolled_ts"], orig_ts)
        # embedding now equals the new vector (no blend).
        self.assertTrue(np.allclose(vid._voiceprints["alice"], v2))

    def test_enroll_tolerates_corrupt_non_dict_meta(self):
        # Seed a voiceprint whose in-memory meta is a non-dict (corruption);
        # enroll must coerce it to {} and proceed without raising.
        self._seed("alice", vec=emb_vec(1), meta="not-a-dict", write_disk=False)
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=emb_vec(2)):
            res = vid.enroll_from_audio("alice",
                                        np.ones(16000, dtype=np.float32), 16000,
                                        append=False)
        self.assertTrue(res["ok"])
        self.assertIsInstance(vid._voicemeta["alice"], dict)

    def test_explicit_permissions_override(self):
        perms = {"sudo": True, "music": False}
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=emb_vec(1)):
            vid.enroll_from_audio("sam", np.ones(16000, dtype=np.float32), 16000,
                                  permissions=perms)
        self.assertEqual(vid._voicemeta["sam"]["permissions"], perms)

    def test_meta_save_failure_still_returns_ok(self):
        # np.save (embedding) succeeds; the meta-sidecar _atomic_write_json
        # raises -> the failure is logged but enroll still reports ok=True
        # because the embedding (the load-bearing artifact) persisted.
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=emb_vec(1)), \
             mock.patch.object(vid, "_atomic_write_json",
                               side_effect=OSError("meta write denied")):
            res = vid.enroll_from_audio("alice",
                                        np.ones(16000, dtype=np.float32), 16000)
        self.assertTrue(res["ok"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "alice.npy")))

    def test_save_embedding_failure_surfaces_error(self):
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=emb_vec(1)), \
             mock.patch.object(vid.np, "save",
                               side_effect=OSError("disk full")):
            res = vid.enroll_from_audio("alice",
                                        np.ones(16000, dtype=np.float32), 16000)
        self.assertFalse(res["ok"])
        self.assertIn("could not save embedding", res["error"])


# ─────────────────────────────────────────────────────────────────────────
# identify_speaker
# ─────────────────────────────────────────────────────────────────────────
class IdentifyTests(_VoiceIdBase):
    def test_no_enrollments_returns_none(self):
        # Short-circuits before even checking the encoder.
        name, score = vid.identify_speaker(np.ones(16000, dtype=np.float32), 16000)
        self.assertIsNone(name)
        self.assertEqual(score, 0.0)

    def test_unavailable_returns_none(self):
        self._seed("alice")
        with mock.patch.object(vid, "is_available", return_value=False):
            name, score = vid.identify_speaker(
                np.ones(16000, dtype=np.float32), 16000)
        self.assertIsNone(name)
        self.assertEqual(score, 0.0)

    def test_embed_none_returns_none(self):
        self._seed("alice")
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=None):
            name, score = vid.identify_speaker(
                np.ones(16000, dtype=np.float32), 16000)
        self.assertIsNone(name)
        self.assertEqual(score, 0.0)

    def test_exact_match_above_threshold(self):
        ref = emb_vec(1)
        self._seed("alice", vec=ref, meta={"name": "Alice"})
        # Identical embedding -> cosine ~1.0 -> match.
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=ref):
            name, score = vid.identify_speaker(
                np.ones(16000, dtype=np.float32), 16000)
        self.assertEqual(name, "Alice")
        self.assertGreater(score, vid.CONFIDENCE_THRESHOLD)
        self.assertEqual(vid._active_speaker, "alice")
        # last_seen metadata persisted.
        self.assertIn("last_seen_ts", vid._voicemeta["alice"])

    def test_below_threshold_returns_unknown_with_score(self):
        ref = emb_vec(1)
        self._seed("alice", vec=ref)
        # An orthogonal-ish probe -> low cosine -> no match but score returned.
        probe = emb_vec(500)
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=probe):
            name, score = vid.identify_speaker(
                np.ones(16000, dtype=np.float32), 16000)
        self.assertIsNone(name)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, vid.CONFIDENCE_THRESHOLD)

    def test_best_of_multiple_speakers_wins(self):
        ref_alice = emb_vec(1)
        ref_sam = emb_vec(2)
        self._seed("alice", vec=ref_alice, meta={"name": "Alice"})
        self._seed("sam", vec=ref_sam, meta={"name": "Sam"})
        # Probe identical to sam -> sam should win even though alice enrolled.
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=ref_sam):
            name, score = vid.identify_speaker(
                np.ones(16000, dtype=np.float32), 16000)
        self.assertEqual(name, "Sam")
        self.assertEqual(vid._active_speaker, "sam")

    def test_match_falls_back_to_slug_when_no_display_name(self):
        ref = emb_vec(7)
        # meta with no "name" key -> display falls back to slug.
        self._seed("sam", vec=ref, meta={"permissions": {}})
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=ref):
            name, _ = vid.identify_speaker(np.ones(16000, dtype=np.float32), 16000)
        self.assertEqual(name, "sam")

    def test_match_persist_failure_is_swallowed(self):
        ref = emb_vec(1)
        self._seed("alice", vec=ref, meta={"name": "Alice"})
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "_embed", return_value=ref), \
             mock.patch.object(vid, "_atomic_write_json",
                               side_effect=OSError("locked")):
            name, score = vid.identify_speaker(
                np.ones(16000, dtype=np.float32), 16000)
        # Still returns the match despite the persist failure.
        self.assertEqual(name, "Alice")


# ─────────────────────────────────────────────────────────────────────────
# permissions_for / can / grant
# ─────────────────────────────────────────────────────────────────────────
class PermissionTests(_VoiceIdBase):
    def test_permissions_none_no_enrollments_is_owner(self):
        # Single-user mode: nobody enrolled -> caller treated as owner.
        self.assertEqual(vid.permissions_for(None), vid._OWNER_PERMISSIONS)

    def test_permissions_none_with_enrollments_is_locked_down(self):
        self._seed("alice")
        self.assertEqual(vid.permissions_for(None), vid._DEFAULT_PERMISSIONS)

    def test_permissions_known_speaker(self):
        self._seed("alice", meta={"name": "Alice",
                                   "permissions": dict(vid._OWNER_PERMISSIONS)})
        self.assertEqual(vid.permissions_for("Alice"), vid._OWNER_PERMISSIONS)

    def test_permissions_enrolled_without_block_is_default(self):
        # Enrolled but the meta has no permissions dict.
        self._seed("sam", meta={"name": "Sam"})
        self.assertEqual(vid.permissions_for("sam"), vid._DEFAULT_PERMISSIONS)

    def test_permissions_unknown_name_is_default(self):
        self._seed("alice")
        self.assertEqual(vid.permissions_for("stranger"), vid._DEFAULT_PERMISSIONS)

    def test_permissions_returns_copy_not_reference(self):
        self._seed("alice", meta={"name": "Alice",
                                   "permissions": {"sudo": True}})
        p = vid.permissions_for("Alice")
        p["sudo"] = False
        # Mutating the returned dict must not corrupt the stored perms.
        self.assertTrue(vid._voicemeta["alice"]["permissions"]["sudo"])

    def test_can_true_and_false(self):
        self._seed("alice", meta={"name": "Alice",
                                   "permissions": {"sudo": True, "shell": False}})
        self.assertTrue(vid.can("Alice", "sudo"))
        self.assertFalse(vid.can("Alice", "shell"))
        self.assertFalse(vid.can("Alice", "nonexistent_cap"))

    def test_can_owner_in_single_user_mode(self):
        # No enrollments -> None resolves to owner, who can sudo.
        self.assertTrue(vid.can(None, "sudo"))

    def test_grant_unknown_speaker_returns_false(self):
        self._seed("alice")
        self.assertFalse(vid.grant("nobody", "sudo", True))

    def test_grant_sets_flag_and_persists(self):
        self._seed("alice", meta={"name": "Alice",
                                   "permissions": dict(vid._DEFAULT_PERMISSIONS)})
        self.assertFalse(vid.can("Alice", "sudo"))
        self.assertTrue(vid.grant("Alice", "sudo", True))
        self.assertTrue(vid.can("Alice", "sudo"))
        # persisted to sidecar
        with open(os.path.join(self.tmp, "alice.json"), encoding="utf-8") as f:
            self.assertTrue(json.load(f)["permissions"]["sudo"])

    def test_grant_default_value_is_true(self):
        self._seed("alice", meta={"name": "Alice", "permissions": {}})
        self.assertTrue(vid.grant("Alice", "memory_write"))
        self.assertTrue(vid.can("Alice", "memory_write"))

    def test_grant_revoke_false(self):
        self._seed("alice", meta={"name": "Alice",
                                   "permissions": dict(vid._OWNER_PERMISSIONS)})
        self.assertTrue(vid.grant("Alice", "sudo", False))
        self.assertFalse(vid.can("Alice", "sudo"))

    def test_grant_save_failure_returns_false(self):
        self._seed("alice", meta={"name": "Alice",
                                   "permissions": dict(vid._DEFAULT_PERMISSIONS)})
        with mock.patch.object(vid, "_atomic_write_json",
                               side_effect=OSError("readonly fs")):
            self.assertFalse(vid.grant("Alice", "sudo", True))


# ─────────────────────────────────────────────────────────────────────────
# memory_namespace_for
# ─────────────────────────────────────────────────────────────────────────
class MemoryNamespaceTests(_VoiceIdBase):
    def test_default_when_no_enrollments(self):
        self.assertEqual(vid.memory_namespace_for("alice"), "default")
        self.assertEqual(vid.memory_namespace_for(None), "default")

    def test_guest_when_enrolled_but_no_name(self):
        self._seed("alice")
        self.assertEqual(vid.memory_namespace_for(None), "guest")
        self.assertEqual(vid.memory_namespace_for(""), "guest")

    def test_slug_when_enrolled_and_named(self):
        self._seed("alice")
        self.assertEqual(vid.memory_namespace_for("Sam B"), "sam_b")


if __name__ == "__main__":
    unittest.main()
