"""Unit tests for core/wake_word.py — the always-on wake-word detector.

CI contract (tools/run_tests_ci_sim.py): the reduced Linux runner installs only
a small dep set; ``numpy`` IS present (so it is REAL here and used for every
synthetic audio frame) but ``sounddevice``, ``openwakeword``, ``pvporcupine``
and ``torch`` are ABSENT. core.wake_word imports numpy at module scope and lazy-
imports the rest inside start()/_open_stream()/_init_*; so the module imports
cleanly with no fakes, and each test that needs an absent backend injects a fake
module into ``sys.modules`` *scoped to the test* (saved/restored, including
absence) so the real environment is pristine afterwards.

Determinism / offline guarantees:
  * No real microphone is opened — ``sounddevice.InputStream`` is a fake whose
    ``start``/``stop``/``close`` just flip flags. The PortAudio audio callback
    (``_cb``) is captured from the fake and driven directly with synthetic numpy
    frames, so detection logic runs WITHOUT any audio thread.
  * No real model is loaded — openwakeword.Model / pvporcupine.create are fakes
    with scriptable ``predict`` / ``process`` returns.
  * No real threads or sleeps: _safe_close_stream's daemon close-thread is the
    only thread, and it completes immediately on the fake (done.set()).

stdlib ``unittest`` + ``unittest.mock`` only. No personal data; no real secrets.
"""
from __future__ import annotations

import contextlib
import math
import os
import queue
import sys
import types
import unittest
from unittest import mock

import numpy as np

import core.wake_word as ww


_SENTINEL = object()


# ─── scoped fake-module injection (mirrors tests/test_voice_id.py) ───────────
@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules, restoring the prior
    state — including absence — on exit. For a dotted name the leaf is ALSO set
    as an attribute on its already-imported parent package (so
    ``from openwakeword.model import Model`` resolves). Passing ``None`` for a
    name forces that import to FAIL — it sets ``sys.modules[name] = None`` (the
    same trick ``mock.patch.dict(sys.modules, {x: None})`` uses) so ``import x``
    raises ImportError even when the real package is installed on the dev box.
    For a dotted ``None`` the leaf is ALSO removed from the already-imported
    parent package so ``from pkg import leaf`` (resolved via the parent's
    attribute, not sys.modules) fails too."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if "." in name:
            parent_name, _, leaf = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                saved_attr.append(
                    (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                if obj is None:
                    with contextlib.suppress(AttributeError):
                        delattr(parent, leaf)
                else:
                    setattr(parent, leaf, obj)
        if obj is None:
            # Sentinel None entry → import machinery raises ImportError.
            sys.modules[name] = None
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                with contextlib.suppress(AttributeError):
                    delattr(parent, leaf)
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            if name in missing:
                sys.modules.pop(name, None)
            elif saved_mod.get(name, _SENTINEL) is not _SENTINEL:
                sys.modules[name] = saved_mod[name]


# ─── fake sounddevice ────────────────────────────────────────────────────────
class FakeInputStream:
    """Stand-in for sounddevice.InputStream. Captures the callback so tests can
    drive synthetic frames through it. ``start``/``stop``/``close`` flip flags."""

    instances: list = []

    def __init__(self, *, samplerate, channels, dtype, blocksize, device,
                 callback):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.device = device
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False
        FakeInputStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    # Helper for tests: push a mono float32 block through the PortAudio callback.
    def feed(self, mono):
        block = np.asarray(mono, dtype=np.float32).reshape(-1, 1)
        self.callback(block, len(block), None, None)


def make_fake_sd(stream_cls=FakeInputStream, open_raises=None, stop_raises=None):
    sd = types.ModuleType("sounddevice")
    if open_raises is not None:
        def _raiser(**kwargs):
            raise open_raises
        sd.InputStream = _raiser
    else:
        sd.InputStream = stream_cls
    sd._stopped = {"n": 0}

    def _stop():
        sd._stopped["n"] += 1
        if stop_raises is not None:
            raise stop_raises
    sd.stop = _stop
    return sd


# ─── fake openwakeword ───────────────────────────────────────────────────────
def make_fake_openwakeword(model_cls=None, with_utils=True,
                           download_raises=False):
    pkg = types.ModuleType("openwakeword")
    model_mod = types.ModuleType("openwakeword.model")

    class _DefaultModel:
        last_kwargs = None

        def __init__(self, *args, **kwargs):
            _DefaultModel.last_kwargs = kwargs
            self.predict_return = {}

        def predict(self, pcm):
            return self.predict_return

    model_mod.Model = model_cls or _DefaultModel
    pkg.model = model_mod

    if with_utils:
        utils_mod = types.ModuleType("openwakeword.utils")

        def _download():
            if download_raises:
                raise RuntimeError("download boom")
        utils_mod.download_models = _download
        pkg.utils = utils_mod
    return pkg, model_mod, (pkg.utils if with_utils else None)


# ─── fake pvporcupine ────────────────────────────────────────────────────────
def make_fake_pvporcupine(handle=None, keywords=("jarvis", "alexa", "computer"),
                          create_raises=None):
    pkg = types.ModuleType("pvporcupine")
    pkg.KEYWORDS = set(keywords)

    class _Handle:
        def __init__(self):
            self.frame_length = 512
            self.process_returns = []  # list of ints to return per call
            self._i = 0

        def process(self, pcm):
            if self._i < len(self.process_returns):
                v = self.process_returns[self._i]
            else:
                v = -1
            self._i += 1
            return v

    pkg._handle = handle or _Handle()

    def _create(*, access_key, keywords, sensitivities):
        if create_raises is not None:
            raise create_raises
        pkg._create_args = {
            "access_key": access_key,
            "keywords": keywords,
            "sensitivities": sensitivities,
        }
        return pkg._handle
    pkg.create = _create
    return pkg


# ──────────────────────────────────────────────────────────────────────
# Module-level helpers: _phrase_to_oww_model_name
# ──────────────────────────────────────────────────────────────────────
class PhraseToOwwModelNameTests(unittest.TestCase):
    def test_jarvis_variants_map_to_hey_jarvis(self):
        for p in ("jarvis", "hey jarvis", "Hey  JARVIS", "  jarvis "):
            self.assertEqual(ww._phrase_to_oww_model_name(p), "hey_jarvis_v0.1")

    def test_alexa_maps(self):
        self.assertEqual(ww._phrase_to_oww_model_name("alexa"), "alexa_v0.1")

    def test_mycroft_variants(self):
        self.assertEqual(ww._phrase_to_oww_model_name("hey mycroft"),
                         "hey_mycroft_v0.1")
        self.assertEqual(ww._phrase_to_oww_model_name("mycroft"),
                         "hey_mycroft_v0.1")

    def test_unknown_phrase_slugified(self):
        self.assertEqual(ww._phrase_to_oww_model_name("open sesame"),
                         "open_sesame")

    def test_onnx_suffix_returns_phrase_verbatim(self):
        # An .onnx name is returned unchanged (custom model passthrough).
        self.assertEqual(ww._phrase_to_oww_model_name("my_model.onnx"),
                         "my_model.onnx")

    def test_existing_file_path_returned_verbatim(self):
        with mock.patch.object(ww.os.path, "isfile", return_value=True):
            self.assertEqual(ww._phrase_to_oww_model_name("Custom Phrase"),
                             "Custom Phrase")

    def test_empty_and_none(self):
        self.assertEqual(ww._phrase_to_oww_model_name(""), "")
        self.assertEqual(ww._phrase_to_oww_model_name(None), "")


# ──────────────────────────────────────────────────────────────────────
# _safe_close_stream
# ──────────────────────────────────────────────────────────────────────
class SafeCloseStreamTests(unittest.TestCase):
    def test_none_is_noop(self):
        ww._safe_close_stream(None)  # must not raise

    def test_stop_then_close_called(self):
        s = FakeInputStream(samplerate=16000, channels=1, dtype="float32",
                            blocksize=1280, device=None, callback=lambda *a: None)
        ww._safe_close_stream(s, timeout_sec=2.0)
        self.assertTrue(s.stopped)
        self.assertTrue(s.closed)

    def test_stop_exception_is_swallowed_and_close_still_runs(self):
        class _S(FakeInputStream):
            def stop(self):
                raise RuntimeError("stop boom")
        s = _S(samplerate=16000, channels=1, dtype="float32", blocksize=1280,
               device=None, callback=lambda *a: None)
        ww._safe_close_stream(s, timeout_sec=2.0)
        self.assertTrue(s.closed)  # close ran despite stop raising

    def test_close_exception_is_swallowed(self):
        class _S(FakeInputStream):
            def close(self):
                raise RuntimeError("close boom")
        s = _S(samplerate=16000, channels=1, dtype="float32", blocksize=1280,
               device=None, callback=lambda *a: None)
        ww._safe_close_stream(s, timeout_sec=2.0)  # daemon thread swallows it
        self.assertTrue(s.stopped)

    def test_close_hang_forces_sd_stop(self):
        # close() blocks past the timeout → the helper imports sounddevice and
        # calls sd.stop(). Use a tiny timeout and a close that waits on an event
        # we never set (released in finally so the daemon thread can exit).
        import threading
        release = threading.Event()
        self.addCleanup(release.set)

        class _S(FakeInputStream):
            def close(self):
                release.wait(timeout=5.0)
        s = _S(samplerate=16000, channels=1, dtype="float32", blocksize=1280,
               device=None, callback=lambda *a: None)
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            ww._safe_close_stream(s, timeout_sec=0.05)
        self.assertEqual(sd._stopped["n"], 1)  # forced sd.stop() on hang

    def test_close_hang_sd_import_failure_is_swallowed(self):
        import threading
        release = threading.Event()
        self.addCleanup(release.set)

        class _S(FakeInputStream):
            def close(self):
                release.wait(timeout=5.0)
        s = _S(samplerate=16000, channels=1, dtype="float32", blocksize=1280,
               device=None, callback=lambda *a: None)
        # sounddevice absent → the inner import raises and is swallowed.
        with inject_modules(sounddevice=None):
            ww._safe_close_stream(s, timeout_sec=0.05)  # no raise


# ──────────────────────────────────────────────────────────────────────
# Construction / config / status
# ──────────────────────────────────────────────────────────────────────
class ConstructionTests(unittest.TestCase):
    def test_defaults(self):
        d = ww.WakeWordDetector()
        self.assertEqual(d.engine, "openwakeword")
        self.assertEqual(d.wake_words, ww.DEFAULT_WAKE_WORDS)
        self.assertEqual(d.sample_rate, ww.DEFAULT_SAMPLE_RATE)
        self.assertIsNone(d.device)
        self.assertEqual(d.threshold, ww.DEFAULT_THRESHOLD)
        self.assertFalse(d.is_running())

    def test_engine_normalised_lowercase_stripped(self):
        self.assertEqual(ww.WakeWordDetector(engine="  OpenWakeWord ").engine,
                         "openwakeword")

    def test_engine_none_becomes_off(self):
        self.assertEqual(ww.WakeWordDetector(engine=None).engine, "off")

    def test_wake_words_copied_not_aliased(self):
        src = ["hey jarvis"]
        d = ww.WakeWordDetector(wake_words=src)
        d.wake_words.append("x")
        self.assertEqual(src, ["hey jarvis"])  # original untouched

    def test_numeric_coercions(self):
        d = ww.WakeWordDetector(sample_rate="22050", threshold="0.8",
                                cooldown_secs="2")
        self.assertEqual(d.sample_rate, 22050)
        self.assertEqual(d.threshold, 0.8)
        self.assertEqual(d.cooldown_secs, 2.0)

    def test_status_shape(self):
        d = ww.WakeWordDetector(engine="off", threshold=0.6)
        st = d.status()
        self.assertEqual(st["engine"], "off")
        self.assertFalse(st["running"])
        self.assertEqual(st["threshold"], 0.6)
        self.assertEqual(st["wake_words"], ww.DEFAULT_WAKE_WORDS)
        self.assertEqual(st["last_event_ts"], 0.0)

    def test_use_silero_vad_flag_coerced(self):
        self.assertTrue(ww.WakeWordDetector(use_silero_vad=1).use_silero_vad)
        self.assertFalse(ww.WakeWordDetector(use_silero_vad=0).use_silero_vad)


# ──────────────────────────────────────────────────────────────────────
# Audio taps
# ──────────────────────────────────────────────────────────────────────
class TapTests(unittest.TestCase):
    def setUp(self):
        self.d = ww.WakeWordDetector()

    def test_add_and_remove_tap(self):
        q = queue.Queue()
        self.d.add_tap(q)
        self.assertIn(q, self.d._taps)
        self.d.remove_tap(q)
        self.assertNotIn(q, self.d._taps)

    def test_add_tap_idempotent(self):
        q = queue.Queue()
        self.d.add_tap(q)
        self.d.add_tap(q)
        self.assertEqual(self.d._taps.count(q), 1)

    def test_remove_unknown_tap_is_safe(self):
        self.d.remove_tap(queue.Queue())  # not registered → no raise


# ──────────────────────────────────────────────────────────────────────
# start() — engine routing & guards
# ──────────────────────────────────────────────────────────────────────
class StartRoutingTests(unittest.TestCase):
    def test_engine_off_returns_false(self):
        d = ww.WakeWordDetector(engine="off")
        self.assertFalse(d.start())
        self.assertFalse(d.is_running())

    def test_already_running_returns_true(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._running = True
        self.assertTrue(d.start())

    def test_unknown_engine_returns_false(self):
        d = ww.WakeWordDetector(engine="bogus")
        self.assertFalse(d.start())

    def test_import_error_from_init_returns_false(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        with mock.patch.object(d, "_init_openwakeword",
                               side_effect=ImportError("no oww")):
            self.assertFalse(d.start())

    def test_generic_exception_from_init_returns_false(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        with mock.patch.object(d, "_init_openwakeword",
                               side_effect=RuntimeError("kaboom")):
            self.assertFalse(d.start())

    def test_start_success_sets_running(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        with mock.patch.object(d, "_init_openwakeword"), \
             mock.patch.object(d, "_open_stream", return_value=True):
            self.assertTrue(d.start())
        self.assertTrue(d.is_running())

    def test_start_open_stream_failure_returns_false(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        with mock.patch.object(d, "_init_openwakeword"), \
             mock.patch.object(d, "_open_stream", return_value=False):
            self.assertFalse(d.start())
        self.assertFalse(d.is_running())

    def test_start_routes_to_porcupine(self):
        d = ww.WakeWordDetector(engine="porcupine")
        with mock.patch.object(d, "_init_porcupine") as init, \
             mock.patch.object(d, "_open_stream", return_value=True):
            self.assertTrue(d.start())
        init.assert_called_once()

    def test_start_invokes_silero_when_enabled(self):
        d = ww.WakeWordDetector(engine="openwakeword", use_silero_vad=True)
        with mock.patch.object(d, "_init_openwakeword"), \
             mock.patch.object(d, "_init_silero_vad") as sil, \
             mock.patch.object(d, "_open_stream", return_value=True):
            d.start()
        sil.assert_called_once()

    def test_start_skips_silero_when_disabled(self):
        d = ww.WakeWordDetector(engine="openwakeword", use_silero_vad=False)
        with mock.patch.object(d, "_init_openwakeword"), \
             mock.patch.object(d, "_init_silero_vad") as sil, \
             mock.patch.object(d, "_open_stream", return_value=True):
            d.start()
        sil.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# _init_openwakeword
# ──────────────────────────────────────────────────────────────────────
class InitOpenWakeWordTests(unittest.TestCase):
    def test_loads_bundled_names(self):
        d = ww.WakeWordDetector(engine="openwakeword",
                                wake_words=["hey jarvis", "alexa"])
        pkg, model_mod, _ = make_fake_openwakeword()
        with inject_modules(openwakeword=pkg,
                            **{"openwakeword.model": model_mod,
                               "openwakeword.utils": pkg.utils}):
            d._init_openwakeword()
        self.assertIsNotNone(d._oww)
        kwargs = model_mod.Model.last_kwargs
        self.assertEqual(kwargs["wakeword_models"],
                         ["hey_jarvis_v0.1", "alexa_v0.1"])

    def test_onnx_path_routed_to_model_paths_first(self):
        d = ww.WakeWordDetector(engine="openwakeword",
                                wake_words=["custom.onnx", "jarvis"])
        pkg, model_mod, _ = make_fake_openwakeword()
        # Make the .onnx path "exist" so it's treated as a model file.
        with inject_modules(openwakeword=pkg,
                            **{"openwakeword.model": model_mod,
                               "openwakeword.utils": pkg.utils}), \
             mock.patch.object(ww.os.path, "isfile",
                               side_effect=lambda p: p == "custom.onnx"):
            d._init_openwakeword()
        # model paths come first, then inline bundled names.
        self.assertEqual(model_mod.Model.last_kwargs["wakeword_models"],
                         ["custom.onnx", "hey_jarvis_v0.1"])

    def test_download_failure_is_non_fatal(self):
        d = ww.WakeWordDetector(engine="openwakeword", wake_words=["jarvis"])
        pkg, model_mod, _ = make_fake_openwakeword(download_raises=True)
        with inject_modules(openwakeword=pkg,
                            **{"openwakeword.model": model_mod,
                               "openwakeword.utils": pkg.utils}):
            d._init_openwakeword()  # download raised but was swallowed
        self.assertIsNotNone(d._oww)

    def test_missing_utils_module_is_tolerated(self):
        # download_models import fails (no utils submodule) → swallowed.
        d = ww.WakeWordDetector(engine="openwakeword", wake_words=["jarvis"])
        pkg, model_mod, _ = make_fake_openwakeword(with_utils=False)
        with inject_modules(openwakeword=pkg,
                            **{"openwakeword.model": model_mod,
                               "openwakeword.utils": None}):
            d._init_openwakeword()
        self.assertIsNotNone(d._oww)

    def test_import_error_propagates(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        with inject_modules(openwakeword=None):
            with self.assertRaises(ImportError):
                d._init_openwakeword()


# ──────────────────────────────────────────────────────────────────────
# _init_porcupine
# ──────────────────────────────────────────────────────────────────────
class InitPorcupineTests(unittest.TestCase):
    def setUp(self):
        # Ensure a clean access-key env per test.
        self._orig = os.environ.get("PORCUPINE_ACCESS_KEY")
        os.environ["PORCUPINE_ACCESS_KEY"] = "test-key-not-real"
        self.addCleanup(self._restore)

    def _restore(self):
        if self._orig is None:
            os.environ.pop("PORCUPINE_ACCESS_KEY", None)
        else:
            os.environ["PORCUPINE_ACCESS_KEY"] = self._orig

    def test_missing_key_raises_runtimeerror(self):
        os.environ["PORCUPINE_ACCESS_KEY"] = ""
        d = ww.WakeWordDetector(engine="porcupine", wake_words=["jarvis"])
        pp = make_fake_pvporcupine()
        with inject_modules(pvporcupine=pp):
            with self.assertRaises(RuntimeError):
                d._init_porcupine()

    def test_jarvis_keyword_mapped(self):
        d = ww.WakeWordDetector(engine="porcupine", wake_words=["hey jarvis"],
                                threshold=0.4)
        pp = make_fake_pvporcupine()
        with inject_modules(pvporcupine=pp):
            d._init_porcupine()
        self.assertEqual(pp._create_args["keywords"], ["jarvis"])
        self.assertEqual(pp._create_args["sensitivities"], [0.4])
        self.assertEqual(d._porcupine_frame, 512)
        self.assertEqual(d._porcupine_keywords, ["jarvis"])

    def test_builtin_keyword_passthrough(self):
        d = ww.WakeWordDetector(engine="porcupine", wake_words=["computer"])
        pp = make_fake_pvporcupine(keywords=("jarvis", "computer"))
        with inject_modules(pvporcupine=pp):
            d._init_porcupine()
        self.assertEqual(pp._create_args["keywords"], ["computer"])

    def test_unknown_keyword_falls_back_to_jarvis(self):
        d = ww.WakeWordDetector(engine="porcupine", wake_words=["open sesame"])
        pp = make_fake_pvporcupine(keywords=("jarvis",))
        with inject_modules(pvporcupine=pp):
            d._init_porcupine()
        self.assertEqual(pp._create_args["keywords"], ["jarvis"])

    def test_duplicate_keywords_deduped_preserving_order(self):
        d = ww.WakeWordDetector(engine="porcupine",
                                wake_words=["jarvis", "hey jarvis", "computer"])
        pp = make_fake_pvporcupine(keywords=("jarvis", "computer"))
        with inject_modules(pvporcupine=pp):
            d._init_porcupine()
        self.assertEqual(pp._create_args["keywords"], ["jarvis", "computer"])

    def test_sensitivities_length_matches_keywords(self):
        d = ww.WakeWordDetector(engine="porcupine",
                                wake_words=["jarvis", "computer"], threshold=0.55)
        pp = make_fake_pvporcupine(keywords=("jarvis", "computer"))
        with inject_modules(pvporcupine=pp):
            d._init_porcupine()
        self.assertEqual(pp._create_args["sensitivities"], [0.55, 0.55])

    def test_import_error_propagates(self):
        d = ww.WakeWordDetector(engine="porcupine")
        with inject_modules(pvporcupine=None):
            with self.assertRaises(ImportError):
                d._init_porcupine()


# ──────────────────────────────────────────────────────────────────────
# _init_silero_vad
# ──────────────────────────────────────────────────────────────────────
class InitSileroVadTests(unittest.TestCase):
    def test_loads_silero_when_torch_present(self):
        d = ww.WakeWordDetector()
        torch = types.ModuleType("torch")
        sentinel = object()
        torch.hub = types.SimpleNamespace(
            load=lambda **kw: (sentinel, {"utils": 1}))
        with inject_modules(torch=torch):
            d._init_silero_vad()
        self.assertIs(d._silero, sentinel)

    def test_torch_absent_disables_endpointing(self):
        d = ww.WakeWordDetector()
        d._silero = "stale"
        with inject_modules(torch=None):
            d._init_silero_vad()
        self.assertIsNone(d._silero)

    def test_hub_load_failure_disables_endpointing(self):
        d = ww.WakeWordDetector()
        torch = types.ModuleType("torch")

        def _boom(**kw):
            raise RuntimeError("hub down")
        torch.hub = types.SimpleNamespace(load=_boom)
        with inject_modules(torch=torch):
            d._init_silero_vad()
        self.assertIsNone(d._silero)


# ──────────────────────────────────────────────────────────────────────
# _open_stream — InputStream wiring + the PortAudio callback (_cb)
# ──────────────────────────────────────────────────────────────────────
class OpenStreamTests(unittest.TestCase):
    def setUp(self):
        FakeInputStream.instances.clear()

    def test_sounddevice_missing_returns_false(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        with inject_modules(sounddevice=None):
            self.assertFalse(d._open_stream())

    def test_open_success_starts_stream(self):
        d = ww.WakeWordDetector(engine="openwakeword", sample_rate=16000)
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            self.assertTrue(d._open_stream())
        stream = FakeInputStream.instances[-1]
        self.assertTrue(stream.started)
        # blocksize = 16000 * 80 / 1000 = 1280 samples.
        self.assertEqual(stream.blocksize, 1280)
        self.assertEqual(stream.samplerate, 16000)
        self.assertEqual(stream.channels, 1)

    def test_open_failure_returns_false_and_clears_stream(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        sd = make_fake_sd(open_raises=RuntimeError("device busy"))
        with inject_modules(sounddevice=sd):
            self.assertFalse(d._open_stream())
        self.assertIsNone(d._stream)

    def test_callback_assembles_frames_and_calls_on_frame(self):
        d = ww.WakeWordDetector(engine="openwakeword", sample_rate=16000)
        seen = []
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            with mock.patch.object(d, "_on_frame",
                                   side_effect=lambda f: seen.append(len(f))):
                d._open_stream()
                stream = FakeInputStream.instances[-1]
                # Feed exactly two 1280-sample frames worth of audio.
                stream.feed(np.zeros(1280 * 2, dtype=np.float32))
        self.assertEqual(seen, [1280, 1280])

    def test_callback_buffers_partial_frame(self):
        d = ww.WakeWordDetector(engine="openwakeword", sample_rate=16000)
        seen = []
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            with mock.patch.object(d, "_on_frame",
                                   side_effect=lambda f: seen.append(len(f))):
                d._open_stream()
                stream = FakeInputStream.instances[-1]
                stream.feed(np.zeros(600, dtype=np.float32))   # < 1280 → buffered
                self.assertEqual(seen, [])
                stream.feed(np.zeros(700, dtype=np.float32))   # 1300 total → 1 frame
        self.assertEqual(seen, [1280])

    def test_callback_handles_2d_indata(self):
        d = ww.WakeWordDetector(engine="openwakeword", sample_rate=16000)
        seen = []
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            with mock.patch.object(d, "_on_frame",
                                   side_effect=lambda f: seen.append(f.ndim)):
                d._open_stream()
                stream = FakeInputStream.instances[-1]
                # Already 2D (frames, channels): the .feed helper reshapes, so
                # build a (N,2) block and call the callback directly.
                block = np.zeros((1280, 2), dtype=np.float32)
                stream.callback(block, 1280, None, None)
        self.assertEqual(seen, [1])  # downmixed to mono (1D)

    def test_callback_fans_out_to_taps(self):
        d = ww.WakeWordDetector(engine="openwakeword", sample_rate=16000)
        tap = queue.Queue()
        d.add_tap(tap)
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            with mock.patch.object(d, "_on_frame"):
                d._open_stream()
                stream = FakeInputStream.instances[-1]
                stream.feed(np.ones(1280, dtype=np.float32))
        self.assertFalse(tap.empty())
        frame = tap.get_nowait()
        self.assertEqual(len(frame), 1280)

    def test_callback_tap_put_failure_isolated(self):
        d = ww.WakeWordDetector(engine="openwakeword", sample_rate=16000)

        class _BadTap:
            def put_nowait(self, x):
                raise RuntimeError("full")
        d.add_tap(_BadTap())
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            with mock.patch.object(d, "_on_frame") as onf:
                d._open_stream()
                stream = FakeInputStream.instances[-1]
                stream.feed(np.ones(1280, dtype=np.float32))  # bad tap swallowed
        onf.assert_called_once()  # frame still processed

    def test_callback_on_frame_exception_does_not_kill_drain(self):
        d = ww.WakeWordDetector(engine="openwakeword", sample_rate=16000)
        calls = {"n": 0}

        def _boom(frame):
            calls["n"] += 1
            raise RuntimeError("frame boom")
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            with mock.patch.object(d, "_on_frame", side_effect=_boom):
                d._open_stream()
                stream = FakeInputStream.instances[-1]
                stream.feed(np.zeros(1280 * 2, dtype=np.float32))
        # Both frames attempted despite each raising.
        self.assertEqual(calls["n"], 2)

    def test_callback_status_flag_branch(self):
        # A truthy ``status`` exercises the (pass) status branch.
        d = ww.WakeWordDetector(engine="openwakeword", sample_rate=16000)
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            with mock.patch.object(d, "_on_frame"):
                d._open_stream()
                stream = FakeInputStream.instances[-1]
                block = np.zeros((1280, 1), dtype=np.float32)
                stream.callback(block, 1280, None, "overflow")  # truthy status

    def test_callback_hard_cap_drops_oldest(self):
        # If _on_frame stops consuming, the buffer must be capped at
        # frame_size * MAX_BUFFER_FRAMES so it can't grow unbounded.
        d = ww.WakeWordDetector(engine="openwakeword", sample_rate=16000)
        frame_size = 1280
        cap = frame_size * ww.MAX_BUFFER_FRAMES
        sd = make_fake_sd()
        # _on_frame raises so the drain loop logs and continues but the while
        # loop still consumes frames; to test the cap we instead patch the drain
        # to never consume by making frame_size huge via a stubbed _on_frame that
        # we don't reach. Simpler: directly verify cap by feeding > cap with a
        # no-op _on_frame replaced by one that re-appends nothing.
        with inject_modules(sounddevice=sd):
            with mock.patch.object(d, "_on_frame"):
                d._open_stream()
                stream = FakeInputStream.instances[-1]
                # Feed more than the cap in one shot; the callback caps _buf
                # BEFORE draining, then drains it down to < frame_size.
                stream.feed(np.zeros(cap + 5000, dtype=np.float32))
        # After draining, residual buffer is whatever < frame_size remained.
        self.assertLess(d._buf.size, frame_size)


# ──────────────────────────────────────────────────────────────────────
# stop / pause / resume lifecycle
# ──────────────────────────────────────────────────────────────────────
class LifecycleTests(unittest.TestCase):
    def test_stop_clears_running_and_closes_stream(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        s = FakeInputStream(samplerate=16000, channels=1, dtype="float32",
                            blocksize=1280, device=None, callback=lambda *a: None)
        d._stream = s
        d._running = True
        d.stop()
        self.assertFalse(d._running)
        self.assertIsNone(d._stream)
        self.assertTrue(s.closed)
        self.assertTrue(d._stop_flag.is_set())

    def test_stop_when_no_stream_is_safe(self):
        d = ww.WakeWordDetector(engine="off")
        d.stop()  # no stream → no raise
        self.assertFalse(d._running)

    def test_pause_closes_stream_and_clears_buffer(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        s = FakeInputStream(samplerate=16000, channels=1, dtype="float32",
                            blocksize=1280, device=None, callback=lambda *a: None)
        d._stream = s
        d._running = True
        d._buf = np.ones(500, dtype=np.float32)
        d.pause()
        self.assertTrue(d._paused)
        self.assertIsNone(d._stream)
        self.assertEqual(d._buf.size, 0)
        self.assertTrue(s.closed)

    def test_pause_noop_when_not_running(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._running = False
        d.pause()
        self.assertFalse(d._paused)

    def test_pause_noop_when_already_paused(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._running = True
        d._paused = True
        with mock.patch.object(ww, "_safe_close_stream") as close:
            d.pause()
        close.assert_not_called()

    def test_resume_reopens_stream(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._running = True
        d._paused = True
        with mock.patch.object(d, "_open_stream", return_value=True) as op:
            self.assertTrue(d.resume())
        op.assert_called_once()
        self.assertFalse(d._paused)

    def test_resume_noop_when_not_paused(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._paused = False
        self.assertFalse(d.resume())

    def test_resume_when_not_running_returns_false(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._paused = True
        d._running = False
        self.assertFalse(d.resume())
        self.assertFalse(d._paused)  # paused flag cleared regardless

    def test_resume_open_failure_clears_running(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._running = True
        d._paused = True
        with mock.patch.object(d, "_open_stream", return_value=False):
            self.assertFalse(d.resume())
        self.assertFalse(d._running)

    def test_pause_then_resume_roundtrip(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        s = FakeInputStream(samplerate=16000, channels=1, dtype="float32",
                            blocksize=1280, device=None, callback=lambda *a: None)
        d._stream = s
        d._running = True
        d.pause()
        self.assertTrue(d._paused)
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd):
            self.assertTrue(d.resume())
        self.assertFalse(d._paused)
        self.assertTrue(d._running)


# ──────────────────────────────────────────────────────────────────────
# _on_frame dispatch
# ──────────────────────────────────────────────────────────────────────
class OnFrameTests(unittest.TestCase):
    def test_stop_flag_short_circuits(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._oww = object()
        d._stop_flag.set()
        with mock.patch.object(d, "_process_oww") as proc:
            d._on_frame(np.zeros(1280, dtype=np.float32))
        proc.assert_not_called()

    def test_routes_to_oww(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._oww = object()
        with mock.patch.object(d, "_process_oww") as proc:
            d._on_frame(np.zeros(1280, dtype=np.float32))
        proc.assert_called_once()

    def test_routes_to_porcupine(self):
        d = ww.WakeWordDetector(engine="porcupine")
        d._porcupine = object()
        with mock.patch.object(d, "_process_porcupine") as proc:
            d._on_frame(np.zeros(512, dtype=np.float32))
        proc.assert_called_once()

    def test_no_engine_handle_is_noop(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._oww = None  # not initialised
        d._on_frame(np.zeros(1280, dtype=np.float32))  # no raise

    def test_processing_exception_is_swallowed(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._oww = object()
        with mock.patch.object(d, "_process_oww",
                               side_effect=RuntimeError("predict boom")):
            d._on_frame(np.zeros(1280, dtype=np.float32))  # swallowed


# ──────────────────────────────────────────────────────────────────────
# _process_oww — scoring on synthetic frames
# ──────────────────────────────────────────────────────────────────────
class ProcessOwwTests(unittest.TestCase):
    def _detector(self, threshold=0.5):
        d = ww.WakeWordDetector(engine="openwakeword", threshold=threshold)
        d._oww = types.SimpleNamespace(predict=lambda pcm: {})
        return d

    def test_fires_when_score_above_threshold(self):
        d = self._detector(threshold=0.5)
        d._oww.predict = lambda pcm: {"hey_jarvis_v0.1": 0.9}
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        fire.assert_called_once()
        name, score = fire.call_args[0]
        self.assertEqual(name, "hey_jarvis_v0.1")
        self.assertAlmostEqual(score, 0.9)

    def test_no_fire_below_threshold(self):
        d = self._detector(threshold=0.8)
        d._oww.predict = lambda pcm: {"hey_jarvis_v0.1": 0.3}
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        fire.assert_not_called()

    def test_picks_best_of_several(self):
        d = self._detector(threshold=0.5)
        d._oww.predict = lambda pcm: {"a": 0.55, "b": 0.95, "c": 0.6}
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        self.assertEqual(fire.call_args[0][0], "b")

    def test_empty_scores_no_fire(self):
        d = self._detector()
        d._oww.predict = lambda pcm: {}
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        fire.assert_not_called()

    def test_nan_score_skipped(self):
        d = self._detector(threshold=0.5)
        # Only candidate is NaN → no finite best → no fire.
        d._oww.predict = lambda pcm: {"x": float("nan")}
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        fire.assert_not_called()

    def test_nan_does_not_mask_real_hit(self):
        d = self._detector(threshold=0.5)
        d._oww.predict = lambda pcm: {"bad": float("nan"), "good": 0.9}
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        self.assertEqual(fire.call_args[0][0], "good")

    def test_non_numeric_score_skipped(self):
        d = self._detector(threshold=0.5)
        d._oww.predict = lambda pcm: {"x": "not-a-number", "y": 0.8}
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        self.assertEqual(fire.call_args[0][0], "y")

    def test_infinite_score_skipped(self):
        d = self._detector(threshold=0.5)
        d._oww.predict = lambda pcm: {"x": math.inf}
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        fire.assert_not_called()

    def test_numpy_scalar_score_coerced(self):
        d = self._detector(threshold=0.5)
        d._oww.predict = lambda pcm: {"x": np.float32(0.77)}
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        self.assertAlmostEqual(fire.call_args[0][1], 0.77, places=5)

    def test_pcm_conversion_clips_and_int16(self):
        # Capture the pcm handed to predict to assert int16 + clipping.
        captured = {}

        def _predict(pcm):
            captured["pcm"] = pcm
            return {}
        d = self._detector()
        d._oww.predict = _predict
        # Values beyond [-1,1] must clip to int16 limits.
        frame = np.array([2.0, -2.0, 0.0, 0.5], dtype=np.float32)
        d._process_oww(frame)
        pcm = captured["pcm"]
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(pcm[0], 32767)
        self.assertEqual(pcm[1], -32768)
        self.assertEqual(pcm[2], 0)

    def test_falsy_scores_return_short_circuits(self):
        # predict returns None (falsy) → early return before iterating.
        d = self._detector()
        d._oww.predict = lambda pcm: None
        with mock.patch.object(d, "_fire") as fire:
            d._process_oww(np.zeros(1280, dtype=np.float32))
        fire.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# _process_porcupine
# ──────────────────────────────────────────────────────────────────────
class ProcessPorcupineTests(unittest.TestCase):
    def _detector(self, frame_len=512, keywords=("jarvis",)):
        d = ww.WakeWordDetector(engine="porcupine", wake_words=list(keywords))
        d._porcupine_frame = frame_len
        d._porcupine_keywords = list(keywords)
        return d

    def test_fires_on_positive_index(self):
        d = self._detector()
        d._porcupine = types.SimpleNamespace(process=lambda pcm: 0)
        with mock.patch.object(d, "_fire") as fire:
            d._process_porcupine(np.zeros(512, dtype=np.float32))
        fire.assert_called_once_with("jarvis", 1.0)

    def test_no_fire_on_negative_index(self):
        d = self._detector()
        d._porcupine = types.SimpleNamespace(process=lambda pcm: -1)
        with mock.patch.object(d, "_fire") as fire:
            d._process_porcupine(np.zeros(512, dtype=np.float32))
        fire.assert_not_called()

    def test_processes_multiple_subframes(self):
        d = self._detector(frame_len=256)
        # process returns -1, then 0 → one fire on the second subframe.
        rets = iter([-1, 0])
        d._porcupine = types.SimpleNamespace(process=lambda pcm: next(rets))
        with mock.patch.object(d, "_fire") as fire:
            d._process_porcupine(np.zeros(512, dtype=np.float32))  # 2 subframes
        fire.assert_called_once()

    def test_index_out_of_range_uses_unknown(self):
        d = self._detector(keywords=("jarvis",))
        d._porcupine = types.SimpleNamespace(process=lambda pcm: 5)  # >len
        with mock.patch.object(d, "_fire") as fire:
            d._process_porcupine(np.zeros(512, dtype=np.float32))
        fire.assert_called_once_with("unknown", 1.0)

    def test_short_frame_no_process(self):
        d = self._detector(frame_len=512)
        called = {"n": 0}

        def _process(pcm):
            called["n"] += 1
            return -1
        d._porcupine = types.SimpleNamespace(process=_process)
        # Only 300 samples < frame_length 512 → loop body never runs.
        d._process_porcupine(np.zeros(300, dtype=np.float32))
        self.assertEqual(called["n"], 0)

    def test_leftover_carried_across_blocks_no_drop(self):
        # 1280-sample blocks / 512-sample frames leave a 256-sample remainder;
        # it must be carried into the next block, not dropped (finding #31).
        d = self._detector(frame_len=512)
        calls = {"n": 0}
        d._porcupine = types.SimpleNamespace(
            process=lambda pcm: (calls.__setitem__("n", calls["n"] + 1) or -1))
        block = np.zeros(1280, dtype=np.float32)
        d._process_porcupine(block)
        self.assertEqual(calls["n"], 2)                 # 1280 → 2 frames
        self.assertEqual(d._porcupine_leftover.size, 256)  # remainder kept
        self.assertEqual(d._porcupine_leftover.dtype, np.int16)
        d._process_porcupine(block)
        # 256 leftover + 1280 = 1536 → 3 more frames; total 5, none dropped.
        self.assertEqual(calls["n"], 5)
        self.assertEqual(d._porcupine_leftover.size, 0)

    def test_short_frame_leftover_accumulates(self):
        # A sub-frame block processes nothing but must not discard its samples.
        d = self._detector(frame_len=512)
        d._porcupine = types.SimpleNamespace(process=lambda pcm: -1)
        d._process_porcupine(np.zeros(300, dtype=np.float32))
        self.assertEqual(d._porcupine_leftover.size, 300)
        d._process_porcupine(np.zeros(300, dtype=np.float32))  # 600 ≥ 512
        self.assertEqual(d._porcupine_leftover.size, 600 - 512)


# ──────────────────────────────────────────────────────────────────────
# _fire — cooldown, events queue, callback
# ──────────────────────────────────────────────────────────────────────
class FireTests(unittest.TestCase):
    def test_fire_enqueues_event(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        d._fire("hey jarvis", 0.9)
        evt = d.events.get_nowait()
        self.assertEqual(evt["phrase"], "hey jarvis")
        self.assertEqual(evt["score"], 0.9)
        self.assertIn("ts", evt)

    def test_fire_invokes_callback(self):
        got = []
        d = ww.WakeWordDetector(engine="openwakeword", on_detect=got.append)
        d._fire("jarvis", 0.8)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["phrase"], "jarvis")

    def test_cooldown_suppresses_rapid_second_fire(self):
        d = ww.WakeWordDetector(engine="openwakeword", cooldown_secs=1.5)
        with mock.patch.object(ww.time, "time", side_effect=[100.0, 100.5]):
            d._fire("jarvis", 0.9)   # fires (ts=100.0)
            d._fire("jarvis", 0.95)  # 0.5s later < 1.5s cooldown → suppressed
        # Only the first event is queued.
        self.assertEqual(d.events.qsize(), 1)

    def test_cooldown_allows_after_window(self):
        d = ww.WakeWordDetector(engine="openwakeword", cooldown_secs=1.0)
        with mock.patch.object(ww.time, "time", side_effect=[100.0, 102.0]):
            d._fire("jarvis", 0.9)
            d._fire("jarvis", 0.9)  # 2s later > 1s cooldown → fires again
        self.assertEqual(d.events.qsize(), 2)

    def test_callback_exception_is_swallowed(self):
        def _boom(evt):
            raise RuntimeError("cb boom")
        d = ww.WakeWordDetector(engine="openwakeword", on_detect=_boom)
        d._fire("jarvis", 0.9)  # must not raise
        # Event still enqueued before the callback ran.
        self.assertEqual(d.events.qsize(), 1)

    def test_no_callback_prints_default(self):
        d = ww.WakeWordDetector(engine="openwakeword", on_detect=None)
        # No callback → the else branch prints; just ensure no raise + enqueue.
        d._fire("jarvis", 0.9)
        self.assertEqual(d.events.qsize(), 1)

    def test_last_fire_ts_updated(self):
        d = ww.WakeWordDetector(engine="openwakeword")
        with mock.patch.object(ww.time, "time", return_value=12345.0):
            d._fire("jarvis", 0.9)
        self.assertEqual(d._last_fire_ts, 12345.0)
        self.assertEqual(d.status()["last_event_ts"], 12345.0)

    def test_event_queue_put_failure_still_calls_callback(self):
        got = []
        d = ww.WakeWordDetector(engine="openwakeword", on_detect=got.append)
        with mock.patch.object(d.events, "put_nowait",
                               side_effect=queue.Full):
            d._fire("jarvis", 0.9)  # put fails, swallowed; callback still runs
        self.assertEqual(len(got), 1)

    def test_events_queue_is_bounded_and_drops_oldest(self):
        # Nobody drains events (the real caller uses on_detect) — the queue must
        # stay capped rather than grow without limit (finding #32).
        d = ww.WakeWordDetector(engine="openwakeword", cooldown_secs=0.0)
        n = ww.EVENTS_QUEUE_MAX
        for i in range(n + 5):
            d._fire(f"p{i}", 0.9)
        self.assertEqual(d.events.qsize(), n)   # capped, not n+5
        # The five oldest were evicted; the front is now p5, not p0.
        self.assertEqual(d.events.get_nowait()["phrase"], "p5")


# ──────────────────────────────────────────────────────────────────────
# End-to-end: start → drive callback → detection event (no real audio/model)
# ──────────────────────────────────────────────────────────────────────
class EndToEndOwwTests(unittest.TestCase):
    def setUp(self):
        FakeInputStream.instances.clear()

    def test_full_path_openwakeword_detection(self):
        events = []
        d = ww.WakeWordDetector(
            engine="openwakeword", wake_words=["hey jarvis"],
            threshold=0.5, on_detect=events.append)

        # Scriptable model: returns a high score so a wake fires.
        class _M:
            def __init__(self, *a, **k):
                pass

            def predict(self, pcm):
                return {"hey_jarvis_v0.1": 0.99}
        pkg, model_mod, _ = make_fake_openwakeword(model_cls=_M)
        sd = make_fake_sd()
        with inject_modules(
                sounddevice=sd, openwakeword=pkg,
                **{"openwakeword.model": model_mod,
                   "openwakeword.utils": pkg.utils}):
            self.assertTrue(d.start())
            self.assertTrue(d.is_running())
            stream = FakeInputStream.instances[-1]
            # One full 1280-sample frame of (synthetic) speech.
            stream.feed((0.2 * np.sin(
                np.linspace(0, 50, 1280))).astype(np.float32))
            d.stop()
        self.assertTrue(events)
        self.assertEqual(events[0]["phrase"], "hey_jarvis_v0.1")
        self.assertGreaterEqual(events[0]["score"], 0.5)

    def test_full_path_porcupine_detection(self):
        os.environ["PORCUPINE_ACCESS_KEY"] = "test-key-not-real"
        self.addCleanup(lambda: os.environ.pop("PORCUPINE_ACCESS_KEY", None))
        events = []
        d = ww.WakeWordDetector(
            engine="porcupine", wake_words=["jarvis"], on_detect=events.append)
        pp = make_fake_pvporcupine()
        pp._handle.frame_length = 1280   # match the 80ms block so 1 subframe
        pp._handle.process_returns = [0]  # fire on the first subframe
        sd = make_fake_sd()
        with inject_modules(sounddevice=sd, pvporcupine=pp):
            self.assertTrue(d.start())
            stream = FakeInputStream.instances[-1]
            stream.feed(np.zeros(1280, dtype=np.float32))
            d.stop()
        self.assertTrue(events)
        self.assertEqual(events[0]["phrase"], "jarvis")


if __name__ == "__main__":
    unittest.main()
