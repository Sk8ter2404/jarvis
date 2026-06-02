"""Logic tests for skills/enroll_voice.py.

enroll_voice is a thin voice layer over core.voice_id (Resemblyzer). Its logic
is almost entirely graceful-degradation + delegation, so tests focus on:
  • the missing-core / Resemblyzer-unavailable / no-one-enrolled messages,
  • _default_user resolution from bobert / env / fallback,
  • the enroll / identify / list / forget / set-active happy + error paths
    with core.voice_id and the mic-capture mocked (no sounddevice, no model),
  • the full mic-capture state machine in _record_seconds (shared-buffer tap,
    sd.rec happy path, sd.rec→InputStream callback fallback, open-failure and
    teardown branches) against an injected fake `sounddevice` — no real audio
    device is ever opened,
  • the _voice_id / _bobert / _input_device / _say helper plumbing,
  • register() wiring every alias to the right handler.

mod._voice_id is patched per-test to return a controllable stub, and
mod._record_seconds is patched (except in the dedicated capture tests) so no
real audio device is opened.

ISOLATION: every fake module lives only inside an `inject_modules` /
`block_import` with-block that saves+restores sys.modules (and any parent-
package attribute) on exit — there are NO module-level sys.modules writes, so
real numpy / core.voice_id stay intact for the rest of the suite.
"""
from __future__ import annotations

import contextlib
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated


# ─── isolation helpers (fakes live ONLY inside these with-blocks) ─────────
_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules. For dotted names
    (``core.voice_id``) the leaf is ALSO set as an attribute on its already-
    imported parent package, because ``from core import voice_id`` resolves the
    leaf via ``getattr(parent, leaf)`` when the parent is a real package.
    Restores the previous state — including absence — on exit so the next test
    sees exactly the real environment again. Pass ``name=None`` to force a
    module to look absent inside the block."""
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
    """Force ``import <name>`` to raise ImportError inside the block, detaching
    any already-imported target (and its parent-package attr) first so the
    block is robust to cross-test pollution. Restores both on exit."""
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
        for name, m in saved_mod.items():
            sys.modules[name] = m


def _fake_vid(available=True, enrolled=None, **over):
    vid = mock.MagicMock()
    vid.is_available.return_value = available
    vid.list_enrolled.return_value = enrolled if enrolled is not None else []
    vid.CONFIDENCE_THRESHOLD = over.get("threshold", 0.75)
    return vid


_FAKE_AUDIO = np.zeros(16000, dtype=np.float32)


class EnrollVoiceDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("enroll_voice")

    def test_enroll_missing_core(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=None):
            self.assertIn("Voice ID core is missing", self.actions["enroll_voice"](""))

    def test_enroll_resemblyzer_unavailable(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=_fake_vid(available=False)):
            out = self.actions["enroll_voice"]("")
        self.assertIn("Resemblyzer isn't installed", out)

    def test_whos_talking_no_enrollments(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=_fake_vid(enrolled=[])):
            out = self.actions["whos_talking"]("")
        self.assertIn("single-user mode", out.lower())

    def test_list_enrolled_none(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=_fake_vid(enrolled=[])):
            out = self.actions["list_enrolled_voices"]("")
        self.assertIn("No voiceprints enrolled", out)

    def test_forget_requires_name(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=_fake_vid()):
            self.assertIn("whose voiceprint to forget", self.actions["forget_voice"](""))

    # ── _default_user resolution ─────────────────────────────────────────
    def test_default_user_from_bobert(self):
        bc = mock.MagicMock()
        bc.VOICE_ID_DEFAULT_USER = "Alice"
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._default_user(), "Alice")

    def test_default_user_from_env_when_no_bobert(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {"VOICE_ID_DEFAULT_USER": "Bob"}, clear=False):
            self.assertEqual(self.mod._default_user(), "Bob")

    def test_default_user_fallback(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod._default_user(), "user")


class EnrollVoiceActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("enroll_voice")

    # ── enroll_voice ─────────────────────────────────────────────────────
    def test_enroll_capture_failure(self):
        vid = _fake_vid()
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=None), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["enroll_voice"]("Alice")
        self.assertIn("couldn't capture the mic", out)

    def test_enroll_happy_first_sample(self):
        vid = _fake_vid()
        vid.enroll_from_audio.return_value = {"ok": True, "name": "Alice", "sample_count": 1}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["enroll_voice"]("Alice")
        self.assertIn("Voiceprint saved for Alice", out)
        # First sample → no "sample N" suffix.
        self.assertNotIn("sample 1", out)

    def test_enroll_reports_additional_samples(self):
        vid = _fake_vid()
        vid.enroll_from_audio.return_value = {"ok": True, "name": "Alice", "sample_count": 3}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["enroll_voice"]("Alice")
        self.assertIn("sample 3", out)

    def test_enroll_failure_surfaces_error(self):
        vid = _fake_vid()
        vid.enroll_from_audio.return_value = {"ok": False, "error": "embedding too short"}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["enroll_voice"]("Alice")
        self.assertIn("Enrollment failed", out)
        self.assertIn("embedding too short", out)

    def test_enroll_defaults_name_when_blank(self):
        vid = _fake_vid()
        vid.enroll_from_audio.return_value = {"ok": True, "name": "user", "sample_count": 1}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_default_user", return_value="user"), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO), \
             mock.patch.object(self.mod, "_say"):
            self.actions["enroll_voice"]("")
        # The resolved default name is passed to enroll_from_audio.
        self.assertEqual(vid.enroll_from_audio.call_args[0][0], "user")

    # ── whos_talking ─────────────────────────────────────────────────────
    def test_whos_talking_match(self):
        vid = _fake_vid(enrolled=["Alice"])
        vid.identify_speaker.return_value = ("Alice", 0.91)
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO):
            out = self.actions["whos_talking"]("")
        self.assertIn("sounds like Alice", out)
        self.assertIn("0.91", out)

    def test_whos_talking_no_match(self):
        vid = _fake_vid(enrolled=["Alice"], threshold=0.75)
        vid.identify_speaker.return_value = (None, 0.40)
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO):
            out = self.actions["whos_talking"]("")
        self.assertIn("doesn't match anyone", out)
        self.assertIn("0.40", out)

    # ── list / forget / set-active ───────────────────────────────────────
    def test_list_enrolled_with_active(self):
        vid = _fake_vid(enrolled=["Alice", "Bob"])
        vid.get_active_speaker.return_value = "Alice"
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["list_enrolled_voices"]("")
        self.assertIn("Alice, Bob", out)
        self.assertIn("Active speaker: Alice", out)

    def test_forget_known(self):
        vid = _fake_vid()
        vid.forget_speaker.return_value = True
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["forget_voice"]("Bob")
        self.assertIn("Forgotten Bob's voiceprint", out)

    def test_forget_unknown(self):
        vid = _fake_vid()
        vid.forget_speaker.return_value = False
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["forget_voice"]("Nobody")
        self.assertIn("don't have a voiceprint enrolled for Nobody", out)

    def test_set_active_clears_on_empty(self):
        vid = _fake_vid()
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["set_active_speaker"]("")
        self.assertIn("Active speaker cleared", out)
        vid.set_active_speaker.assert_called_once_with(None)

    def test_set_active_known(self):
        vid = _fake_vid()
        vid.set_active_speaker.return_value = True
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["set_active_speaker"]("Alice")
        self.assertIn("Active speaker set to Alice", out)

    # ── voice_id_status ──────────────────────────────────────────────────
    def test_status_offline(self):
        vid = _fake_vid()
        vid.encoder_status.return_value = {"encoder_loaded": False,
                                           "encoder_error": "model file missing"}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["voice_id_status"]("")
        self.assertIn("Voice ID is offline", out)
        self.assertIn("model file missing", out)

    def test_status_online(self):
        vid = _fake_vid()
        vid.encoder_status.return_value = {
            "encoder_loaded": True, "enrolled": ["Alice"],
            "active_speaker": "Alice", "threshold": 0.75}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["voice_id_status"]("")
        self.assertIn("Voice ID is online", out)
        self.assertIn("Alice", out)
        self.assertIn("0.75", out)


# ─────────────────────────────────────────────────────────────────────────
#  Helper plumbing: _voice_id, _bobert, _input_device, _default_user, _say
# ─────────────────────────────────────────────────────────────────────────
class HelperPlumbingTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("enroll_voice")

    # ── _voice_id ────────────────────────────────────────────────────────
    def test_voice_id_imports_real_core_module(self):
        # core.voice_id is import-safe (numpy only at module load), so the
        # real import path resolves and returns the module object.
        vid = self.mod._voice_id()
        self.assertIsNotNone(vid)
        self.assertTrue(hasattr(vid, "is_available"))

    def test_voice_id_returns_none_when_import_blocked(self):
        # Force `from core import voice_id` to raise so the except branch
        # (print + return None) is exercised. Two things can satisfy that
        # import: the cached submodule in sys.modules, AND the `voice_id`
        # attribute on the already-imported real `core` package. We neutralise
        # BOTH — `sys.modules[name] = None` makes the submodule import raise
        # ("None in sys.modules"), and detaching the parent attr stops
        # `getattr(core, "voice_id")` from short-circuiting to the real module.
        # Both are restored in finally so the rest of the suite is unaffected.
        import core as _core_pkg
        saved_mod = sys.modules.get("core.voice_id", _SENTINEL)
        saved_attr = getattr(_core_pkg, "voice_id", _SENTINEL)
        sys.modules["core.voice_id"] = None  # type: ignore[assignment]
        if saved_attr is not _SENTINEL:
            try:
                delattr(_core_pkg, "voice_id")
            except AttributeError:
                pass
        try:
            self.assertIsNone(self.mod._voice_id())
        finally:
            if saved_mod is _SENTINEL:
                sys.modules.pop("core.voice_id", None)
            else:
                sys.modules["core.voice_id"] = saved_mod
            if saved_attr is not _SENTINEL:
                setattr(_core_pkg, "voice_id", saved_attr)

    # ── _bobert ──────────────────────────────────────────────────────────
    def test_bobert_none_when_not_loaded(self):
        with inject_modules(bobert_companion=None, __main__=None):
            self.assertIsNone(self.mod._bobert())

    def test_bobert_prefers_loaded_companion(self):
        fake = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=fake):
            self.assertIs(self.mod._bobert(), fake)

    # ── _input_device ────────────────────────────────────────────────────
    def test_input_device_from_bobert(self):
        bc = types.SimpleNamespace(get_input_device=lambda: 4)
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._input_device(), 4)

    def test_input_device_getter_raises_returns_none(self):
        def _boom():
            raise RuntimeError("no device")
        bc = types.SimpleNamespace(get_input_device=_boom)
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIsNone(self.mod._input_device())

    def test_input_device_no_bobert_returns_none(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertIsNone(self.mod._input_device())

    def test_input_device_getter_not_callable(self):
        # bc present but get_input_device is not callable -> the `if callable`
        # guard is False and the function returns None (75->80 branch).
        bc = types.SimpleNamespace(get_input_device="not a function")
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIsNone(self.mod._input_device())

    # ── _default_user (USER_NAME fallback branch) ────────────────────────
    def test_default_user_falls_back_to_user_name_attr(self):
        # VOICE_ID_DEFAULT_USER absent → the `or getattr(USER_NAME)` arm wins.
        bc = types.SimpleNamespace(USER_NAME="Sam")
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._default_user(), "Sam")

    def test_default_user_blank_attr_falls_through_to_env(self):
        bc = types.SimpleNamespace(VOICE_ID_DEFAULT_USER="   ")
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.dict(os.environ, {"JARVIS_USER": "Alice"}, clear=True):
            self.assertEqual(self.mod._default_user(), "Alice")

    # ── _say ─────────────────────────────────────────────────────────────
    def test_say_uses_bobert_say(self):
        spoken = []
        bc = types.SimpleNamespace(say=lambda t: spoken.append(t))
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.mod._say("hello sir")
        self.assertEqual(spoken, ["hello sir"])

    def test_say_falls_back_to_synthesise(self):
        spoken = []
        # No `say`; only `synthesise` present (getattr chain second arm).
        bc = types.SimpleNamespace(synthesise=lambda t: spoken.append(t))
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.mod._say("via synth")
        self.assertEqual(spoken, ["via synth"])

    def test_say_swallows_speak_error_and_prints(self):
        def _boom(_):
            raise RuntimeError("audio dead")
        bc = types.SimpleNamespace(say=_boom)
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            # Must not raise; falls through to the print() path.
            self.mod._say("resilient")

    def test_say_prints_when_no_bobert(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            self.mod._say("printed only")   # no exception == covered

    def test_say_prints_when_speaker_not_callable(self):
        # bc present but neither say nor synthesise is callable -> the
        # `if callable(say)` guard is False, so _say falls to print (229->235).
        bc = types.SimpleNamespace(say=None, synthesise="nope")
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.mod._say("fallback print")


# ─────────────────────────────────────────────────────────────────────────
#  _record_seconds capture state-machine — against a FAKE sounddevice.
#  No real PortAudio stream is ever opened; every branch is driven by the
#  injected fake's behaviour.
# ─────────────────────────────────────────────────────────────────────────
def _make_sd(*, rec_result=_SENTINEL, rec_raises=False,
             stream_factory=_SENTINEL, stream_raises=False):
    """Build a fake `sounddevice` module.

    rec_result    : array returned by sd.rec()+sd.wait() (default: a 1-D
                    non-empty float32 buffer). Pass None to simulate an empty
                    capture that forces the InputStream fallback.
    rec_raises    : sd.rec() raises -> exercises the rec→InputStream fallback.
    stream_factory: callable(**kwargs) -> fake InputStream (for the fallback).
    stream_raises : sd.InputStream(...) construction raises (open-failed path).
    """
    sd = types.ModuleType("sounddevice")

    def _rec(n, **k):
        if rec_raises:
            raise RuntimeError("sd.rec exploded")
        if rec_result is _SENTINEL:
            return np.ones(n, dtype=np.float32)
        return rec_result

    sd.rec = _rec
    sd.wait = lambda: None
    sd.stop = lambda: None

    if stream_raises:
        def _InputStream(**k):
            raise RuntimeError("device in use")
        sd.InputStream = _InputStream
    elif stream_factory is not _SENTINEL:
        sd.InputStream = stream_factory
    return sd


class _FakeStream:
    """Minimal sd.InputStream stand-in. On start() it pumps `n_pushes` frames
    of `value` through the callback, then reports time elapsed so the capture
    while-loop exits quickly. close()/stop() record that they ran."""
    def __init__(self, *, callback, value=0.2, n_pushes=3, frames=512):
        self._cb = callback
        self._value = value
        self._n = n_pushes
        self._frames = frames
        self.stopped = False
        self.closed = False

    def start(self):
        for _ in range(self._n):
            block = np.full(self._frames, self._value, dtype=np.float32)
            self._cb(block, self._frames, None, None)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class RecordSecondsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("enroll_voice")
        # Keep the InputStream poll-loop short & non-blocking. The capture loop
        # is `while time()-start < seconds: q.get(timeout=0.2)`. We freeze a
        # clock that jumps past `seconds` after the queued frames drain.

    # ── shared mic-buffer tap (preferred path) ───────────────────────────
    def test_uses_bobert_mic_buffer_when_available(self):
        buf = np.ones(8000, dtype=np.float32)
        bc = types.SimpleNamespace(get_mic_buffer=lambda secs, sr: buf)
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.mod._record_seconds(0.5)
        self.assertIsNotNone(out)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.size, 8000)

    def test_mic_buffer_empty_falls_through_to_sd_rec(self):
        # get_mic_buffer returns an empty array (size 0) -> skip tap, use sd.rec.
        bc = types.SimpleNamespace(
            get_mic_buffer=lambda secs, sr: np.zeros(0, dtype=np.float32),
            get_input_device=lambda: None)
        sd = _make_sd()   # default rec returns a non-empty buffer
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd):
            out = self.mod._record_seconds(0.25)
        self.assertIsNotNone(out)
        self.assertGreater(out.size, 0)

    def test_mic_buffer_getter_raises_falls_through(self):
        def _boom(secs, sr):
            raise RuntimeError("buffer unavailable")
        bc = types.SimpleNamespace(get_mic_buffer=_boom,
                                   get_input_device=lambda: None)
        sd = _make_sd()
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd):
            out = self.mod._record_seconds(0.25)
        self.assertIsNotNone(out)

    # ── sounddevice missing entirely ─────────────────────────────────────
    def test_sounddevice_missing_returns_none(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             block_import("sounddevice"):
            self.assertIsNone(self.mod._record_seconds(0.25))

    # ── sd.rec happy path ────────────────────────────────────────────────
    def test_sd_rec_happy_path_1d(self):
        sd = _make_sd(rec_result=np.ones(4000, dtype=np.float32))
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             inject_modules(sounddevice=sd):
            out = self.mod._record_seconds(0.25)
        self.assertEqual(out.size, 4000)

    def test_sd_rec_happy_path_2d_takes_first_channel(self):
        stereo = np.ones((3000, 2), dtype=np.float32)
        sd = _make_sd(rec_result=stereo)
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             inject_modules(sounddevice=sd):
            out = self.mod._record_seconds(0.25)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.size, 3000)

    # ── sd.rec fails -> InputStream callback fallback succeeds ────────────
    def test_rec_fails_then_inputstream_callback_succeeds(self):
        created = {}

        def _factory(**k):
            s = _FakeStream(callback=k["callback"], value=0.3, n_pushes=4)
            created["stream"] = s
            return s

        sd = _make_sd(rec_raises=True, stream_factory=_factory)
        # bc has no _safe_close_stream -> exercises the inline daemon-close
        # teardown (stream.stop + daemon close + Event.wait).
        bc = types.SimpleNamespace(get_input_device=lambda: None)
        # Freeze time so the capture loop runs one drain pass then exits.
        clock = iter([1000.0, 1000.0, 1000.05, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(clock)):
            out = self.mod._record_seconds(0.25)
        self.assertIsNotNone(out)
        self.assertGreater(out.size, 0)
        self.assertTrue(created["stream"].stopped)

    # ── InputStream open itself fails -> None ────────────────────────────
    def test_rec_fails_and_inputstream_open_fails(self):
        sd = _make_sd(rec_raises=True, stream_raises=True)
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             inject_modules(sounddevice=sd):
            self.assertIsNone(self.mod._record_seconds(0.25))

    # ── InputStream path with bc._safe_close_stream teardown branch ──────
    def test_inputstream_uses_bc_safe_close_stream(self):
        closed = {}

        def _factory(**k):
            return _FakeStream(callback=k["callback"], value=0.25, n_pushes=3)

        sd = _make_sd(rec_raises=True, stream_factory=_factory)
        bc = types.SimpleNamespace(
            get_input_device=lambda: None,
            _safe_close_stream=lambda s: closed.setdefault("via", "bc"))
        clock = iter([500.0, 500.0, 500.02, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(clock)):
            out = self.mod._record_seconds(0.25)
        self.assertIsNotNone(out)
        self.assertEqual(closed.get("via"), "bc")

    # ── InputStream.start() raises mid-capture -> None (181-183) ─────────
    def test_inputstream_start_raises_returns_none(self):
        class _BoomStream:
            def __init__(self, **k):
                self.stopped = False
                self.closed = False

            def start(self):
                raise RuntimeError("stream start blew up")

            def stop(self):
                self.stopped = True

            def close(self):
                self.closed = True

        sd = _make_sd(rec_raises=True, stream_factory=lambda **k: _BoomStream(**k))
        bc = types.SimpleNamespace(get_input_device=lambda: None,
                                   _safe_close_stream=lambda s: None)
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd):
            out = self.mod._record_seconds(0.25)
        self.assertIsNone(out)

    # ── inline teardown: stop() AND close() raise (197-198, 203-204) ─────
    def test_inline_teardown_swallows_stop_and_close_errors(self):
        class _GrumpyStream:
            def __init__(self, **k):
                self._cb = k["callback"]

            def start(self):
                # Push one frame so chunks is non-empty (return value valid).
                self._cb(np.full(256, 0.2, dtype=np.float32), 256, None, None)

            def stop(self):
                raise RuntimeError("stop failed")

            def close(self):
                raise RuntimeError("close failed")

        sd = _make_sd(rec_raises=True,
                      stream_factory=lambda **k: _GrumpyStream(**k))
        # No _safe_close_stream -> inline daemon-close path runs; both stop()
        # and close() raise and must be swallowed (finally never propagates).
        bc = types.SimpleNamespace(get_input_device=lambda: None)
        clock = iter([10.0, 10.0, 10.02, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(clock)):
            out = self.mod._record_seconds(0.25)
        self.assertIsNotNone(out)
        self.assertGreater(out.size, 0)

    # ── inline teardown escape hatch: daemon close hangs (213-217) ───────
    def test_inline_teardown_escape_hatch_on_hung_close(self):
        import threading as _threading
        stopped_globally = {"called": False}

        class _SlowStream:
            def __init__(self, **k):
                self._cb = k["callback"]

            def start(self):
                self._cb(np.full(256, 0.2, dtype=np.float32), 256, None, None)

            def stop(self):
                pass

            def close(self):
                pass

        sd = _make_sd(rec_raises=True,
                      stream_factory=lambda **k: _SlowStream(**k))
        sd.stop = lambda: stopped_globally.__setitem__("called", True)
        bc = types.SimpleNamespace(get_input_device=lambda: None)
        clock = iter([20.0, 20.0, 20.02, 9999.0, 9999.0, 9999.0])
        # Force the post-daemon wait() to report a timeout so the global
        # sd.stop() escape hatch fires — without actually waiting 2 s.
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd), \
             mock.patch.object(_threading.Event, "wait", return_value=False), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(clock)):
            out = self.mod._record_seconds(0.25)
        self.assertIsNotNone(out)
        self.assertTrue(stopped_globally["called"])

    # ── bc._safe_close_stream raises -> swallowed (188-189) ──────────────
    def test_safe_close_stream_error_is_swallowed(self):
        def _factory(**k):
            return _FakeStream(callback=k["callback"], value=0.2, n_pushes=2)

        def _boom_close(_s):
            raise RuntimeError("safe-close blew up")

        sd = _make_sd(rec_raises=True, stream_factory=_factory)
        bc = types.SimpleNamespace(get_input_device=lambda: None,
                                   _safe_close_stream=_boom_close)
        clock = iter([30.0, 30.0, 30.02, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(clock)):
            out = self.mod._record_seconds(0.25)
        self.assertIsNotNone(out)   # teardown error must not break the return

    # ── escape-hatch global sd.stop() itself raises (216-217) ────────────
    def test_escape_hatch_global_stop_error_swallowed(self):
        import threading as _threading

        def _factory(**k):
            return _FakeStream(callback=k["callback"], value=0.2, n_pushes=2)

        sd = _make_sd(rec_raises=True, stream_factory=_factory)
        def _boom_stop():
            raise RuntimeError("global stop failed")
        sd.stop = _boom_stop
        bc = types.SimpleNamespace(get_input_device=lambda: None)
        clock = iter([40.0, 40.0, 40.02, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd), \
             mock.patch.object(_threading.Event, "wait", return_value=False), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(clock)):
            out = self.mod._record_seconds(0.25)
        self.assertIsNotNone(out)

    # ── sd.rec returns an empty (size-0) buffer -> InputStream fallback ──
    def test_sd_rec_empty_buffer_falls_to_inputstream(self):
        def _factory(**k):
            return _FakeStream(callback=k["callback"], value=0.2, n_pushes=2)

        # rec returns a size-0 array: the `rec.size > 0` guard is False, so the
        # code drops past sd.rec into the InputStream fallback (138->145).
        sd = _make_sd(rec_result=np.zeros(0, dtype=np.float32),
                      stream_factory=_factory)
        bc = types.SimpleNamespace(get_input_device=lambda: None,
                                   _safe_close_stream=lambda s: None)
        clock = iter([50.0, 50.0, 50.02, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(clock)):
            out = self.mod._record_seconds(0.25)
        self.assertIsNotNone(out)
        self.assertGreater(out.size, 0)

    # ── InputStream yields no frames -> returns None ─────────────────────
    def test_inputstream_no_frames_returns_none(self):
        def _factory(**k):
            # n_pushes=0 -> callback never fires -> chunks stays empty.
            return _FakeStream(callback=k["callback"], n_pushes=0)

        sd = _make_sd(rec_raises=True, stream_factory=_factory)
        bc = types.SimpleNamespace(
            get_input_device=lambda: None,
            _safe_close_stream=lambda s: None)
        # Clock jumps straight past the window so the empty loop exits at once.
        clock = iter([0.0, 0.0, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(clock)):
            out = self.mod._record_seconds(0.25)
        self.assertIsNone(out)


# ─────────────────────────────────────────────────────────────────────────
#  Remaining action branches + registration wiring
# ─────────────────────────────────────────────────────────────────────────
class RemainingActionBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("enroll_voice")

    def test_whos_talking_missing_core(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=None):
            self.assertIn("Voice ID core is missing",
                          self.actions["whos_talking"](""))

    def test_whos_talking_resemblyzer_unavailable(self):
        vid = _fake_vid(available=False, enrolled=["Alice"])
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["whos_talking"]("")
        self.assertIn("Resemblyzer is unavailable", out)

    def test_whos_talking_capture_failure(self):
        vid = _fake_vid(enrolled=["Alice"])
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=None):
            out = self.actions["whos_talking"]("")
        self.assertIn("couldn't capture the mic", out)

    def test_list_enrolled_missing_core(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=None):
            self.assertIn("Voice ID core is missing",
                          self.actions["list_enrolled_voices"](""))

    def test_list_enrolled_no_active_speaker(self):
        vid = _fake_vid(enrolled=["Alice", "Bob"])
        vid.get_active_speaker.return_value = None
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["list_enrolled_voices"]("")
        self.assertIn("Alice, Bob", out)
        self.assertNotIn("Active speaker", out)

    def test_forget_missing_core(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=None):
            self.assertIn("Voice ID core is missing",
                          self.actions["forget_voice"]("Bob"))

    def test_set_active_missing_core(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=None):
            self.assertIn("Voice ID core is missing",
                          self.actions["set_active_speaker"]("Alice"))

    def test_set_active_unknown_speaker(self):
        vid = _fake_vid()
        vid.set_active_speaker.return_value = False
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["set_active_speaker"]("Ghost")
        self.assertIn("don't have a voiceprint enrolled for Ghost", out)

    def test_voice_id_status_missing_core(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=None):
            self.assertIn("Voice ID core is missing",
                          self.actions["voice_id_status"](""))

    def test_status_online_no_enrollments_and_no_active(self):
        vid = _fake_vid()
        vid.encoder_status.return_value = {
            "encoder_loaded": True, "enrolled": [],
            "active_speaker": None, "threshold": 0.72}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["voice_id_status"]("")
        self.assertIn("(none)", out)
        self.assertIn("none", out.lower())

    # ── register() wiring ────────────────────────────────────────────────
    def test_register_wires_every_alias(self):
        actions: dict = {}
        self.mod.register(actions)
        # Aliases that should resolve to the same enrollment handler.
        self.assertIs(actions["enroll_voice"], actions["learn_my_voice"])
        for who in ("whos_talking", "who_is_talking", "identify_speaker"):
            self.assertIs(actions[who], actions["whos_talking"])
        self.assertIs(actions["list_enrolled_voices"], actions["enrolled_voices"])
        self.assertIn("forget_voice", actions)
        self.assertIn("set_active_speaker", actions)
        self.assertIn("voice_id_status", actions)


if __name__ == "__main__":
    unittest.main()
