"""Logic tests for skills/ambient_listen.py.

ambient_listen is the passive-transcription daemon (mic + system-audio +
screen). It's heavy (sounddevice / whisper / mss), so tests target the
deterministic, mockable helpers and the action surface — never a real stream:
  • wake-phrase regex compilation with word boundaries ('jar visit' ≠ jarvis),
  • screen blocklist compilation + sensitive-window matching (1Password,
    banking, auth screens),
  • the daily vision-budget tracker (remaining / charge / 14-day prune),
  • the average-hash + hamming distance helpers,
  • the rolling-buffer time-window + hard-cap trim,
  • wake-listener-active gating + the start/stop/status actions.

Disk paths (state file, budget, jsonl) are redirected to a temp dir. The
worker threads are neutered by the harness (Thread.start no-op), so the
start actions never open a device.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated


# ─── isolation helpers ───────────────────────────────────────────────────
_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules and restore the prior
    state — including absence — on exit. Mirrors the save/restore contract used
    by tests/skills/test_self_diagnostic.py so injected fakes (sounddevice, PIL,
    ctypes, …) never persist process-wide and the real modules are restored for
    every other test. Pass dotted keys via ``**{"a.b": obj}``; the leaf is also
    set as an attribute on an already-imported parent package."""
    saved_mod: dict = {}
    missing: set = set()
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


class _FakeStream:
    """Minimal sd.InputStream stand-in. Captures the callback so a test can
    push synthetic audio frames into the worker, and records start/stop/close
    so teardown paths can be asserted. ``start_raises`` simulates a device that
    opens but fails to start."""
    def __init__(self, *, start_raises=False, on_start=None, **kw):
        self.kw = kw
        self.callback = kw.get("callback")
        self.started = False
        self.stopped = False
        self.closed = False
        self._start_raises = start_raises
        self._on_start = on_start

    def feed(self, indata, frames=None):
        """Push one synthetic audio block through the captured callback,
        exactly as PortAudio would from its own thread. ``frames`` mirrors
        PortAudio's frame count; defaults to len(indata) but can be passed
        explicitly for stand-in blocks that don't support len()."""
        if self.callback is not None:
            if frames is None:
                try:
                    frames = len(indata)
                except TypeError:
                    frames = 0
            self.callback(indata, frames, None, None)

    def start(self):
        if self._start_raises:
            raise RuntimeError("device start boom")
        self.started = True
        if self._on_start is not None:
            self._on_start(self)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    # context-manager form is intentionally NOT used by the module, but provide
    # it so an accidental `with` wouldn't explode a test.
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.close()


def _make_sd(*, devices=None, hostapis=None, stream=None,
             open_raises=None, wasapi=True):
    """Build a fake ``sounddevice`` module. ``stream`` is returned from
    InputStream(); ``open_raises`` makes InputStream() raise instead."""
    sd = types.ModuleType("sounddevice")
    holder = {"stream": stream}

    def _input_stream(**kw):
        if open_raises is not None:
            raise open_raises
        st = holder["stream"]
        if st is None:
            st = _FakeStream(**kw)
            holder["stream"] = st
        else:
            st.kw = kw
            st.callback = kw.get("callback")
        return st

    sd.InputStream = _input_stream
    sd.stop = mock.MagicMock(name="sd.stop")
    sd.query_hostapis = mock.MagicMock(
        return_value=hostapis if hostapis is not None else
        ([{"name": "Windows WASAPI", "default_output_device": 3}] if wasapi
         else [{"name": "MME", "default_output_device": 0}]))

    def _query_devices(idx=None):
        devs = devices if devices is not None else []
        if idx is None:
            return devs
        return devs[idx]
    sd.query_devices = _query_devices
    if wasapi:
        sd.WasapiSettings = lambda **kw: ("wasapi", kw)
    sd._stream_holder = holder
    return sd


class _ScriptedEvent:
    """A threading.Event stand-in whose ``wait()`` returns scripted booleans so
    a worker's ``while not stop_evt.is_set()`` loop runs a fixed number of
    iterations on the *calling* thread, then exits — no real threads, no sleep.

    ``wait()`` also runs an optional ``on_wait`` callback each call, letting a
    test inject audio frames into the worker's captured callback at the exact
    point the loop blocks (so the next drain sees them)."""
    def __init__(self, wait_returns, on_wait=None):
        self._returns = list(wait_returns)
        self._set = False
        self._on_wait = on_wait
        self.wait_calls = 0

    def set(self):
        self._set = True

    def clear(self):
        self._set = False

    def is_set(self):
        return self._set

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._on_wait is not None:
            try:
                self._on_wait(self.wait_calls)
            except Exception:
                pass
        if self._returns:
            val = self._returns.pop(0)
        else:
            val = True  # fail-safe: always terminate
        if val:
            self._set = True
        return val


class AmbientListenHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")

    # ── _compile_wake_pattern ────────────────────────────────────────────
    def test_wake_pattern_word_boundary(self):
        bc = mock.MagicMock()
        bc.WAKE_PHRASES = {"jarvis", "hey jarvis"}
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            pat = self.mod._compile_wake_pattern()
        self.assertTrue(pat.search("ok jarvis are you there"))
        self.assertTrue(pat.search("hey jarvis"))
        # Word-boundary guard: 'jar visit' / 'jarvisible' must NOT match.
        self.assertIsNone(pat.search("jar visit the store"))
        self.assertIsNone(pat.search("jarvisible spectrum"))

    def test_wake_pattern_defaults_when_empty(self):
        bc = mock.MagicMock()
        bc.WAKE_PHRASES = set()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            pat = self.mod._compile_wake_pattern()
        self.assertTrue(pat.search("jarvis"))

    # ── blocklist / sensitive window ─────────────────────────────────────
    def test_sensitive_window_matches_password_managers(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            bl = self.mod._compile_blocklist()
        self.assertTrue(self.mod._is_sensitive_window("1Password — Vault", "1password.exe", bl))
        self.assertTrue(self.mod._is_sensitive_window("Chase Online Banking", "chrome.exe", bl))
        self.assertTrue(self.mod._is_sensitive_window("Bitwarden", "bitwarden.exe", bl))

    def test_sensitive_window_matches_auth_screens(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            bl = self.mod._compile_blocklist()
        self.assertTrue(self.mod._is_sensitive_window("Authenticator app", "", bl))
        self.assertTrue(self.mod._is_sensitive_window("Enter your SSN", "", bl))

    def test_non_sensitive_window_passes(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            bl = self.mod._compile_blocklist()
        self.assertFalse(self.mod._is_sensitive_window("report.docx - Word", "winword.exe", bl))

    def test_blocklist_tolerates_bad_extra_regex(self):
        bc = mock.MagicMock()
        bc.AMBIENT_SCREEN_BLOCKLIST = ("(",)   # invalid regex → skipped, not raised
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            bl = self.mod._compile_blocklist()
        # The defaults still compiled even though the extra one was bad.
        self.assertTrue(any(p.search("1password") for p in bl))

    # ── _set_ambient_stream_active refcount (0xc0000374 guard) ───────────
    def test_ambient_stream_refcount_survives_concurrent_owners(self):
        # The mic worker and the loopback worker can be live at once; a shared
        # boolean let one's teardown clear the PortAudio-reinit guard while the
        # other's InputStream was still live (0xc0000374). The refcount must stay
        # > 0 until BOTH release, and never go negative on a stray release.
        bc = types.SimpleNamespace(_ambient_stream_active=[0],
                                   _mic_lock=threading.Lock())
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._set_ambient_stream_active(True)    # loopback claims
            self.mod._set_ambient_stream_active(True)    # mic claims
            self.assertEqual(bc._ambient_stream_active[0], 2)
            self.mod._set_ambient_stream_active(False)   # one exits
            self.assertEqual(bc._ambient_stream_active[0], 1)   # guard STILL active
            self.mod._set_ambient_stream_active(False)   # the other exits
            self.assertEqual(bc._ambient_stream_active[0], 0)
            self.mod._set_ambient_stream_active(False)   # stray release
            self.assertEqual(bc._ambient_stream_active[0], 0)   # never negative

    def test_ambient_claim_prefers_host_teardown_gate(self):
        # 2026-08-14: when the host exposes the _pa_gate teardown-gate API,
        # claims/releases must route through bc._pa_claim_owner /
        # bc._pa_release_owner (refcount semantics) so the claim is atomic
        # with the monolith's reinit latch — and a refused claim must surface
        # as False so the worker aborts instead of opening a stream into a
        # native teardown.
        calls = []

        def _claim(cell, refcount=False, timeout=1.0, deny_if=None):
            calls.append(("claim", refcount, timeout))
            cell[0] = int(cell[0]) + 1
            return True

        def _release(cell, refcount=False):
            calls.append(("release", refcount))
            cell[0] = max(0, int(cell[0]) - 1)

        bc = types.SimpleNamespace(_ambient_stream_active=[0],
                                   _mic_lock=threading.Lock(),
                                   _pa_claim_owner=_claim,
                                   _pa_release_owner=_release)
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertTrue(self.mod._set_ambient_stream_active(True))
            self.assertEqual(bc._ambient_stream_active[0], 1)
            self.assertTrue(self.mod._set_ambient_stream_active(False))
            self.assertEqual(bc._ambient_stream_active[0], 0)
        self.assertEqual(calls, [("claim", True, 2.0), ("release", True)])

    def test_ambient_claim_refused_by_gate_returns_false(self):
        bc = types.SimpleNamespace(_ambient_stream_active=[0],
                                   _mic_lock=threading.Lock(),
                                   _pa_claim_owner=lambda *a, **k: False,
                                   _pa_release_owner=lambda *a, **k: None)
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertFalse(self.mod._set_ambient_stream_active(True))
        self.assertEqual(bc._ambient_stream_active[0], 0)

    # ── _hamming / _phash64 ──────────────────────────────────────────────
    def test_hamming_distance(self):
        self.assertEqual(self.mod._hamming(0b1010, 0b1010), 0)
        self.assertEqual(self.mod._hamming(0b1111, 0b1010), 2)

    def test_phash_identical_images_zero_distance(self):
        # Two solid-grey images hash identically → hamming 0.
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, format="PNG")
        png = buf.getvalue()
        h1 = self.mod._phash64(png)
        h2 = self.mod._phash64(png)
        self.assertIsNotNone(h1)
        self.assertEqual(self.mod._hamming(h1, h2), 0)

    def test_phash_returns_none_on_garbage(self):
        self.assertIsNone(self.mod._phash64(b"not a png"))


class AmbientListenBudgetTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self.tmp = tempfile.mkdtemp(prefix="ambient_budget_")
        self.addCleanup(self._cleanup)
        self.mod._DATA_DIR = self.tmp
        self.mod._BUDGET_PATH = os.path.join(self.tmp, "budget.json")

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def test_budget_remaining_starts_at_cap(self):
        bc = mock.MagicMock()
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertAlmostEqual(self.mod._vision_budget_remaining(), 1.0)

    def test_budget_charge_reduces_remaining(self):
        bc = mock.MagicMock()
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._vision_budget_charge(0.30)
            self.assertAlmostEqual(self.mod._vision_budget_remaining(), 0.70)

    def test_budget_never_negative(self):
        bc = mock.MagicMock()
        bc.AMBIENT_VISION_BUDGET_USD = 0.50
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._vision_budget_charge(5.0)   # massive overspend
            self.assertEqual(self.mod._vision_budget_remaining(), 0.0)

    def test_budget_prunes_to_14_days(self):
        # Pre-seed 20 days of history; a charge should prune to the last 14.
        old = {f"2026-05-{d:02d}": 0.01 for d in range(1, 21)}
        with mock.patch.object(self.mod, "_save_budget") as save, \
             mock.patch.object(self.mod, "_load_budget", return_value=dict(old)):
            self.mod._vision_budget_charge(0.01)
        saved = save.call_args[0][0]
        self.assertLessEqual(len(saved), 14)


class AmbientListenStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self.mod._buffer.clear()

    # ── _trim_buffer ─────────────────────────────────────────────────────
    def test_trim_buffer_drops_old_entries(self):
        bc = mock.MagicMock()
        bc.AMBIENT_LISTEN_BUFFER_MINUTES = 10
        now = time.time()
        self.mod._buffer.append({"ts": now - 9999, "text": "old"})   # >10 min
        self.mod._buffer.append({"ts": now - 60, "text": "recent"})
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._trim_buffer(now)
        texts = [e["text"] for e in self.mod._buffer]
        self.assertEqual(texts, ["recent"])

    def test_trim_buffer_enforces_hard_cap(self):
        bc = mock.MagicMock()
        bc.AMBIENT_LISTEN_BUFFER_MINUTES = 100000   # window won't drop anything
        now = time.time()
        for i in range(self.mod._HARD_ENTRY_CAP + 50):
            self.mod._buffer.append({"ts": now, "text": str(i)})
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._trim_buffer(now)
        self.assertEqual(len(self.mod._buffer), self.mod._HARD_ENTRY_CAP)

    # ── _maybe_nudge_wake ────────────────────────────────────────────────
    def _make_bc(self, sleep=False, standby=False, tts=False):
        bc = mock.MagicMock()
        bc._sleep_mode = [sleep]
        bc._standby_mode = [standby]
        bc._tts_playback_active = [tts]
        return bc

    def test_maybe_nudge_announces_in_standby(self):
        import re
        bc = self._make_bc(standby=True)
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        self.mod._tts_last_active[0] = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("hey jarvis you there")
        bc.proactive_announce.assert_called_once()
        self.assertIn("heard my name", bc.proactive_announce.call_args[0][0])

    def test_maybe_nudge_announces_in_sleep(self):
        import re
        bc = self._make_bc(sleep=True)
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        self.mod._tts_last_active[0] = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("jarvis wake up")
        bc.proactive_announce.assert_called_once()
        self.assertIn("listening", bc.proactive_announce.call_args[0][0].lower())

    def test_maybe_nudge_silent_when_awake(self):
        # Awake (no sleep, no standby): the main loop already handles the
        # utterance — the nudge must NOT announce "I heard my name" on top.
        import re
        bc = self._make_bc()
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        self.mod._tts_last_active[0] = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("jarvis open my bookmarks")
        bc.proactive_announce.assert_not_called()

    def test_maybe_nudge_silent_during_own_tts(self):
        # JARVIS's own voice echoing into the mic must not trip the nudge.
        import re
        bc = self._make_bc(standby=True, tts=True)
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        self.mod._tts_last_active[0] = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("jarvis online at your service")
        bc.proactive_announce.assert_not_called()

    def test_maybe_nudge_silent_just_after_tts(self):
        # Echo cooldown: playback ended a moment ago, the batched echo
        # transcript lands late — still suppressed.
        import re
        bc = self._make_bc(standby=True)
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        self.mod._tts_last_active[0] = time.time() - 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("jarvis standing by")
        bc.proactive_announce.assert_not_called()

    def test_maybe_nudge_debounces(self):
        import re
        bc = self._make_bc(standby=True)
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = time.time()   # just fired → debounce
        self.mod._tts_last_active[0] = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("jarvis again")
        bc.proactive_announce.assert_not_called()

    # ── _wake_listener_active ────────────────────────────────────────────
    def test_wake_listener_active_true(self):
        import sys
        wl = mock.MagicMock()
        wl._detector.is_running.return_value = True
        with mock.patch.dict(sys.modules, {"skill_wake_listener": wl}):
            self.assertTrue(self.mod._wake_listener_active())

    def test_wake_listener_inactive_when_module_absent(self):
        import sys
        sys.modules.pop("skill_wake_listener", None)
        self.assertFalse(self.mod._wake_listener_active())


class AmbientListenActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        # Ensure no daemon is recorded as alive between tests.
        self.mod._thread = None
        self.mod._audio_thread = None
        self.mod._screen_thread = None

    def test_listen_stop_when_not_running(self):
        self.assertIn("not running", self.actions["ambient_listen_stop"](""))

    def test_listen_start_refused_when_wake_listener_owns_mic(self):
        with mock.patch.object(self.mod, "_wake_listener_active", return_value=True):
            out = self.actions["ambient_listen_start"]("")
        self.assertIn("stop the wake-word listener first", out)

    def test_audio_start_windows_only_guard(self):
        # On non-Windows the action refuses immediately. On Windows it would
        # try to start (thread neutered); assert whichever branch this host hits.
        import sys
        out = self.actions["ambient_audio_start"]("")
        if sys.platform != "win32":
            self.assertIn("Windows-only", out)
        else:
            self.assertIn("System-audio capture", out)

    def test_screen_stop_when_not_running(self):
        self.assertIn("not running", self.actions["ambient_screen_stop"](""))

    def test_status_action_renders(self):
        out = self.actions["ambient_listen_status"]("")
        # Status should mention the mic listening state in some form.
        self.assertTrue(len(out) > 0)
        self.assertIn("sir", out.lower())


# ─────────────────────────────────────────────────────────────────────────
# Fake bobert_companion + a temp-dir base for the disk-touching paths.
# ─────────────────────────────────────────────────────────────────────────
class _FakeBobert:
    """A configurable bobert_companion stand-in. Only the attributes the
    workers actually read are set; everything else is absent so the module's
    ``getattr(b, ..., default)`` fallbacks are exercised."""
    def __init__(self, **attrs):
        # Sensible transcribe: returns scripted (text, conf) tuples in order.
        self.SAMPLE_RATE = 16000
        for k, v in attrs.items():
            setattr(self, k, v)


def _good_conf():
    return {"no_speech_prob": 0.01, "avg_logprob": -0.2}


class _TmpDirMixin:
    """Redirect every on-disk path the module writes to into a private temp
    dir so a test never touches the real project files, and clean up after."""
    def _redirect_paths(self):
        self.tmp = tempfile.mkdtemp(prefix="ambient_t_")
        self.addCleanup(self._rm_tmp)
        self.mod._DATA_DIR = self.tmp
        self.mod._STATE_PATH = os.path.join(self.tmp, "state.json")
        self.mod._AUDIO_JSONL = os.path.join(self.tmp, "audio.jsonl")
        self.mod._SCREEN_JSONL = os.path.join(self.tmp, "screen.jsonl")
        self.mod._BUDGET_PATH = os.path.join(self.tmp, "budget.json")
        # _persist_state writes its tmp file into _PROJECT_DIR; point it at tmp.
        self.mod._PROJECT_DIR = self.tmp

    def _rm_tmp(self):
        for root, _dirs, files in os.walk(self.tmp):
            for fn in files:
                try:
                    os.unlink(os.path.join(root, fn))
                except OSError:
                    pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────
# Mic worker loop (_worker_loop)
# ─────────────────────────────────────────────────────────────────────────
class MicWorkerLoopTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._buffer.clear()
        self.mod._last_error = None

    def _run_worker(self, bc, *, feed_block, wait_returns, sd=None,
                    extra_modules=None):
        """Drive _worker_loop synchronously: inject a fake sounddevice whose
        stream feeds ``feed_block`` on start, and a scripted stop event so the
        loop runs once and exits."""
        if sd is None:
            sd = _make_sd()

        def _on_start(stream):
            stream.feed(feed_block)

        stream = _FakeStream(on_start=_on_start)
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent(wait_returns)
        mods = {"sounddevice": sd}
        if extra_modules:
            mods.update(extra_modules)
        with inject_modules(**mods), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title",
                               return_value="Notepad"):
            self.mod._worker_loop()
        return stream

    def test_worker_transcribes_and_buffers(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("hello world", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.is_ambient_music = mock.MagicMock(return_value=False)
        # 3 s of loud audio at 16 kHz clears the 2.5 s batch + 0.003 RMS gate.
        block = (np.ones(16000 * 3, dtype=np.float32) * 0.2)
        self.mod._wake_pattern = None
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        self.assertEqual(len(self.mod._buffer), 1)
        self.assertEqual(self.mod._buffer[0]["text"], "hello world")
        # Mirror line went to the audio jsonl with source='mic'.
        with open(self.mod._AUDIO_JSONL, encoding="utf-8") as f:
            line = json.loads(f.readline())
        self.assertEqual(line["source"], "mic")
        self.assertEqual(line["window"], "Notepad")

    def test_worker_rms_gate_skips_silence(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("should not run", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        block = np.zeros(16000 * 3, dtype=np.float32)  # silent → RMS 0
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        bc.transcribe.assert_not_called()
        self.assertEqual(len(self.mod._buffer), 0)

    def test_worker_drops_invalid_speech(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("garble", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(False, "halluc"))
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        self.assertEqual(len(self.mod._buffer), 0)

    def test_worker_drops_ambient_music(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("la la la", _good_conf()))
        bc.is_ambient_music = mock.MagicMock(return_value=True)
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        self.assertEqual(len(self.mod._buffer), 0)

    def test_worker_empty_text_skipped(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        self.assertEqual(len(self.mod._buffer), 0)

    def test_worker_transcribe_exception_sets_error(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(side_effect=RuntimeError("whisper boom"))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        self.assertIn("transcribe failed", self.mod._last_error)
        self.assertEqual(len(self.mod._buffer), 0)

    def test_worker_fires_wake_nudge(self):
        import re
        bc = _FakeBobert()
        # Standby: the one state where the tap-fed nudge is the wake path.
        bc._sleep_mode = [False]
        bc._standby_mode = [True]
        bc.proactive_announce = mock.MagicMock()
        bc.transcribe = mock.MagicMock(return_value=("hey jarvis", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        self.mod._tts_last_active[0] = 0.0
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        bc.proactive_announce.assert_called_once()

    def test_worker_pause_clears_queue(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        self.mod._paused[0] = True
        self.addCleanup(lambda: self.mod._paused.__setitem__(0, False))
        # First wait() (paused branch) returns True → exit before any transcribe.
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        bc.transcribe.assert_not_called()

    def test_worker_drops_batches_while_the_mic_is_muted(self):
        # 2026-10-01: the tray's Mute Mic drops every batch like the
        # daemon-pause guard — nothing heard while muted is transcribed or
        # written to the transcript log (it used to be both).
        bc = _FakeBobert(_mic_muted=[True])
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        self.mod._prev_mic_batch[0] = "half a line"
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        bc.transcribe.assert_not_called()
        self.assertEqual(len(self.mod._buffer), 0)
        self.assertFalse(os.path.exists(self.mod._AUDIO_JSONL))
        self.assertIsNone(self.mod._prev_mic_batch[0])

    def test_worker_sounddevice_import_failure(self):
        bc = _FakeBobert()
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=None), \
             mock.patch.dict(sys.modules, {}), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt):
            # Block the import so the deferred `import sounddevice` raises.
            real_import = __import__

            def _fake(name, *a, **k):
                if name == "sounddevice":
                    raise ImportError("no sounddevice")
                return real_import(name, *a, **k)
            with mock.patch("builtins.__import__", side_effect=_fake):
                self.mod._worker_loop()
        self.assertIn("sounddevice import failed", self.mod._last_error)

    def test_worker_bobert_absent(self):
        sd = _make_sd()
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=None), \
             mock.patch.object(self.mod, "_stop_evt", evt):
            self.mod._worker_loop()
        self.assertIn("bobert_companion not loaded", self.mod._last_error)

    def test_worker_transcribe_not_callable(self):
        bc = _FakeBobert()
        bc.transcribe = "not callable"
        sd = _make_sd()
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt):
            self.mod._worker_loop()
        self.assertIn("not callable", self.mod._last_error)

    def test_worker_mic_disabled_guard(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc._mic_input_disabled = lambda: True
        sd = _make_sd()
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt):
            self.mod._worker_loop()
        self.assertIn("mic disabled", self.mod._last_error)

    def test_worker_inputstream_open_fails_plain(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        sd = _make_sd(open_raises=RuntimeError("PortAudio nope"))
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_wake_listener_active", return_value=False), \
             mock.patch.object(self.mod, "_stop_evt", evt):
            self.mod._worker_loop()
        self.assertIn("InputStream open failed", self.mod._last_error)

    def test_worker_inputstream_open_fails_wake_listener(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        sd = _make_sd(open_raises=RuntimeError("locked"))
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_wake_listener_active", return_value=True), \
             mock.patch.object(self.mod, "_stop_evt", evt):
            self.mod._worker_loop()
        self.assertIn("mic locked by wake-word listener", self.mod._last_error)

    def test_worker_stream_start_fails(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        sd = _make_sd()
        sd._stream_holder["stream"] = _FakeStream(start_raises=True)
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod, "_safe_close_stream") as close:
            self.mod._worker_loop()
        self.assertIn("InputStream.start failed", self.mod._last_error)
        close.assert_called_once()

    def test_worker_multichannel_callback_path(self):
        # 2-D indata exercises the indata[:, 0] branch of the mic callback.
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("stereo", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        self.mod._wake_pattern = None
        block = (np.ones((16000 * 3, 2), dtype=np.float32) * 0.2)
        self._run_worker(bc, feed_block=block, wait_returns=[True])
        self.assertEqual(len(self.mod._buffer), 1)


# ─────────────────────────────────────────────────────────────────────────
# Mic worker — record_speech TAP-SHARING path (symptom-1 regression guard).
#
# When the host exposes add_record_tap / remove_record_tap, the worker MUST
# share the main loop's mic via the tap instead of opening a competing
# sd.InputStream. A second stream on the same WASAPI device starves
# record_speech and makes JARVIS deaf to the wake word "JARVIS" — the
# user-reported "go ambient → won't wake" bug. These tests assert the worker
# registers a tap, NEVER opens an InputStream, transcribes the tapped frames,
# and detaches the tap on exit.
# ─────────────────────────────────────────────────────────────────────────
class MicTapSharingTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._buffer.clear()
        self.mod._last_error = None

    def _tap_bobert(self):
        """A _FakeBobert that implements the record_speech tap API. Captures the
        registered queue so the test can push frames the host 'captured'."""
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("tapped speech", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.is_ambient_music = mock.MagicMock(return_value=False)
        bc._tap_queues = []
        bc.add_record_tap = lambda q: (bc._tap_queues.append(q) or True)
        bc.remove_record_tap = mock.MagicMock(name="remove_record_tap")
        return bc

    def test_tap_path_never_opens_inputstream(self):
        bc = self._tap_bobert()
        self.mod._wake_pattern = None
        # An sd whose InputStream BLOWS UP if ever called — proves the worker
        # took the tap path and never tried to open a competing stream.
        sd = _make_sd(open_raises=AssertionError(
            "tap path must not open an InputStream"))

        # Feed 3 s of loud audio through the tap right after it's registered, so
        # the real drain thread transfers it into the batch loop. A scripted
        # stop event with short real sleeps lets the daemon drain thread run.
        block = (np.ones(16000 * 3, dtype=np.float32) * 0.2)

        def _on_wait(n):
            if n == 1:
                # Tap is registered by now; hand the host-captured frame over.
                for q in bc._tap_queues:
                    q.put(block)
            time.sleep(0.03)   # yield so the drain thread moves frames across

        evt = _ScriptedEvent([False, False, False, True], on_wait=_on_wait)
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title",
                               return_value="Notepad"):
            self.mod._worker_loop()

        # The tap was registered exactly once.
        self.assertEqual(len(bc._tap_queues), 1)
        # The tapped audio was transcribed + buffered (learning is alive).
        self.assertGreaterEqual(len(self.mod._buffer), 1)
        self.assertEqual(self.mod._buffer[-1]["text"], "tapped speech")
        # No open error — InputStream was never touched.
        self.assertIsNone(self.mod._last_error)

    def test_tap_detached_on_exit(self):
        bc = self._tap_bobert()
        self.mod._wake_pattern = None
        sd = _make_sd()  # present but should remain unused on the tap path
        evt = _ScriptedEvent([True])   # exit immediately
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title",
                               return_value="Notepad"):
            self.mod._worker_loop()
        # On the way out the worker detaches its tap from record_speech.
        self.assertEqual(len(bc._tap_queues), 1)
        bc.remove_record_tap.assert_called_once_with(bc._tap_queues[0])

    def _fake_ap(self):
        """core.audio_processor stand-in: ``shared`` is the get_processor()
        singleton the MAIN mic path runs every chunk through; ``private`` is
        what AudioProcessor() builds. Both pass audio through unchanged."""
        ap = types.ModuleType("core.audio_processor")
        shared = mock.MagicMock(name="shared_singleton")
        shared.process.side_effect = lambda a, **kw: a
        private = mock.MagicMock(name="private_processor")
        private.process.side_effect = lambda a, **kw: a
        private.sample_rate = 16000
        ap.get_processor = mock.MagicMock(return_value=shared)
        ap.AudioProcessor = mock.MagicMock(return_value=private)
        return ap, shared, private

    def test_tap_batches_never_touch_the_shared_processor(self):
        # Audit P1-8 part 2. The tapped frames are record_speech's own mic
        # frames, which the main path has ALREADY run through the shared
        # get_processor() singleton (and counted in the mic-health stats).
        # Running each 2.5 s batch through it again fed the same audio into
        # its AGC running-RMS / flatness / noise state twice, skewing the gain
        # the main mic path depends on. The worker must use its own processor
        # and record no mic stats for frames the host already measured.
        bc = self._tap_bobert()
        bc._audio_master_enabled = [True]
        self.mod._wake_pattern = None
        sd = _make_sd(open_raises=AssertionError("no stream on the tap path"))
        ap, shared, private = self._fake_ap()
        block = (np.ones(16000 * 3, dtype=np.float32) * 0.2)

        def _on_wait(n):
            if n == 1:
                for q in bc._tap_queues:
                    q.put(block)
            time.sleep(0.03)

        evt = _ScriptedEvent([False, False, False, True], on_wait=_on_wait)
        with inject_modules(sounddevice=sd, **{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title",
                               return_value="Notepad"):
            self.mod._worker_loop()

        self.assertEqual(self.mod._buffer[-1]["text"], "tapped speech")
        shared.process.assert_not_called()
        private.process.assert_called()
        self.assertFalse(private.process.call_args.kwargs["record_mic_stats"])

    def test_own_stream_batches_use_the_private_processor_and_count(self):
        # Fallback path (host without the tap API): still never the shared
        # singleton, but these frames are measured nowhere else, so they DO
        # feed the dead-mic stats (real mic, never loopback).
        bc = _FakeBobert()
        bc._audio_master_enabled = [True]
        bc.transcribe = mock.MagicMock(return_value=("fallback speech", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.is_ambient_music = mock.MagicMock(return_value=False)
        self.mod._wake_pattern = None
        block = (np.ones(16000 * 3, dtype=np.float32) * 0.2)
        stream = _FakeStream(on_start=lambda s: s.feed(block))
        sd = _make_sd(stream=stream)
        ap, shared, private = self._fake_ap()
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd, **{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title",
                               return_value="Notepad"):
            self.mod._worker_loop()

        self.assertEqual(self.mod._buffer[0]["text"], "fallback speech")
        shared.process.assert_not_called()
        self.assertTrue(private.process.call_args.kwargs["record_mic_stats"])

    def test_falls_back_to_stream_when_no_tap_api(self):
        # Host WITHOUT the tap API (older monolith) → the worker must still
        # function via its own dedicated InputStream (the fallback path).
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("fallback speech", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.is_ambient_music = mock.MagicMock(return_value=False)
        # No add_record_tap / remove_record_tap on bc.
        self.mod._wake_pattern = None
        block = (np.ones(16000 * 3, dtype=np.float32) * 0.2)

        def _on_start(stream):
            stream.feed(block)
        stream = _FakeStream(on_start=_on_start)
        sd = _make_sd(stream=stream)
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title",
                               return_value="Notepad"):
            self.mod._worker_loop()
        # Dedicated stream was opened + started (fallback path engaged).
        self.assertTrue(stream.started)
        self.assertEqual(len(self.mod._buffer), 1)
        self.assertEqual(self.mod._buffer[0]["text"], "fallback speech")


# ─────────────────────────────────────────────────────────────────────────
# System-audio (WASAPI loopback) worker (_audio_worker_loop)
# ─────────────────────────────────────────────────────────────────────────
class AudioWorkerLoopTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._audio_entries_total = 0
        self.mod._audio_last_error = None

    def _loopback_sd(self, *, native_sr=16000, stream=None):
        devs = [
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 0,
             "default_samplerate": 48000},
            {"name": "Mic", "hostapi": 0, "max_input_channels": 1,
             "default_samplerate": 16000},
            {"name": "Speakers (loopback)", "hostapi": 0,
             "max_input_channels": 2, "default_samplerate": native_sr},
        ]
        hostapis = [{"name": "Windows WASAPI", "default_output_device": 0}]
        sd = _make_sd(devices=devs, hostapis=hostapis, wasapi=True)
        if stream is not None:
            sd._stream_holder["stream"] = stream
        return sd

    def _run(self, bc, *, sd, feed_block, wait_returns):
        def _on_start(stream):
            stream.feed(feed_block)
        stream = _FakeStream(on_start=_on_start)
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent(wait_returns)
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title", return_value="YouTube"), \
             mock.patch.object(self.mod, "_focused_proc_name", return_value="chrome.exe"):
            self.mod._audio_worker_loop()
        return stream

    def test_audio_worker_happy_path_same_rate(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("podcast line", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.is_ambient_music = mock.MagicMock(return_value=False)
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        sd = self._loopback_sd(native_sr=16000)
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2  # >5 s, loud
        self._run(bc, sd=sd, feed_block=block, wait_returns=[True])
        self.assertEqual(self.mod._audio_entries_total, 1)
        with open(self.mod._AUDIO_JSONL, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["source"], "system_audio")
        self.assertEqual(entry["proc"], "chrome.exe")

    def test_audio_worker_resamples_48k(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("resampled", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        sd = self._loopback_sd(native_sr=48000)
        # 6 s at 48 kHz, 2-channel → mean-down + linear resample to 16 kHz.
        block = np.ones((48000 * 6, 2), dtype=np.float32) * 0.2
        self._run(bc, sd=sd, feed_block=block, wait_returns=[True])
        self.assertEqual(self.mod._audio_entries_total, 1)
        # transcribe should have received ~16 kHz * 6 s samples.
        got = bc.transcribe.call_args[0][0]
        self.assertAlmostEqual(len(got) / 16000.0, 6.0, delta=0.2)

    def test_audio_worker_rms_gate(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        sd = self._loopback_sd(native_sr=16000)
        block = np.zeros(16000 * 6, dtype=np.float32)
        self._run(bc, sd=sd, feed_block=block, wait_returns=[True])
        bc.transcribe.assert_not_called()
        self.assertEqual(self.mod._audio_entries_total, 0)

    def test_audio_worker_invalid_speech_dropped(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("noise", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(False, "no"))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        sd = self._loopback_sd(native_sr=16000)
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        self._run(bc, sd=sd, feed_block=block, wait_returns=[True])
        self.assertEqual(self.mod._audio_entries_total, 0)

    def test_audio_worker_transcribe_exception(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(side_effect=RuntimeError("boom"))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        sd = self._loopback_sd(native_sr=16000)
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        self._run(bc, sd=sd, feed_block=block, wait_returns=[True])
        self.assertIn("transcribe failed", self.mod._audio_last_error)

    def test_audio_worker_pause_branch(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        sd = self._loopback_sd(native_sr=16000)
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        self.mod._paused[0] = True
        self.addCleanup(lambda: self.mod._paused.__setitem__(0, False))
        self._run(bc, sd=sd, feed_block=block, wait_returns=[True])
        bc.transcribe.assert_not_called()

    def test_audio_worker_no_loopback_device(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        sd = _make_sd(wasapi=False)  # no WASAPI host → finder returns None
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt):
            self.mod._audio_worker_loop()
        self.assertIn("no WASAPI loopback device", self.mod._audio_last_error)

    def test_audio_worker_import_failure(self):
        bc = _FakeBobert()
        evt = _ScriptedEvent([True])
        real_import = __import__

        def _fake(name, *a, **k):
            if name == "sounddevice":
                raise ImportError("nope")
            return real_import(name, *a, **k)
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt), \
             mock.patch("builtins.__import__", side_effect=_fake):
            self.mod._audio_worker_loop()
        self.assertIn("sounddevice import failed", self.mod._audio_last_error)

    def test_audio_worker_bobert_absent(self):
        sd = self._loopback_sd()
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=None), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt):
            self.mod._audio_worker_loop()
        self.assertIn("bobert_companion not loaded", self.mod._audio_last_error)

    def test_audio_worker_transcribe_not_callable(self):
        bc = _FakeBobert()
        bc.transcribe = None
        sd = self._loopback_sd()
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt):
            self.mod._audio_worker_loop()
        self.assertIn("not callable", self.mod._audio_last_error)

    def test_audio_worker_open_fails(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        sd = self._loopback_sd()
        # Make InputStream raise.
        def _raise(**kw):
            raise RuntimeError("loopback open boom")
        sd.InputStream = _raise
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt):
            self.mod._audio_worker_loop()
        self.assertIn("loopback open failed", self.mod._audio_last_error)

    def test_audio_worker_start_fails(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        sd = self._loopback_sd(stream=_FakeStream(start_raises=True))
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt), \
             mock.patch.object(self.mod, "_safe_close_stream") as close:
            self.mod._audio_worker_loop()
        self.assertIn("loopback start failed", self.mod._audio_last_error)
        close.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────
# _find_loopback_device
# ─────────────────────────────────────────────────────────────────────────
class FindLoopbackDeviceTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")

    def test_finds_explicit_loopback_entry(self):
        devs = [
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 0},
            {"name": "Speakers (loopback)", "hostapi": 0, "max_input_channels": 2},
        ]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])
        self.assertEqual(self.mod._find_loopback_device(sd), 1)

    def test_falls_back_to_name_match(self):
        devs = [
            {"name": "Realtek Speakers", "hostapi": 0, "max_input_channels": 0},
            {"name": "Realtek Speakers", "hostapi": 0, "max_input_channels": 2},
        ]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])
        self.assertEqual(self.mod._find_loopback_device(sd), 1)

    def test_returns_none_without_wasapi(self):
        sd = _make_sd(hostapis=[{"name": "MME", "default_output_device": 0}],
                      devices=[])
        self.assertIsNone(self.mod._find_loopback_device(sd))

    def test_returns_none_on_query_exception(self):
        sd = types.ModuleType("sounddevice")
        sd.query_hostapis = mock.MagicMock(side_effect=RuntimeError("x"))
        self.assertIsNone(self.mod._find_loopback_device(sd))

    def test_returns_none_when_no_match(self):
        devs = [{"name": "Webcam Mic", "hostapi": 0, "max_input_channels": 1}]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": -1}])
        self.assertIsNone(self.mod._find_loopback_device(sd))


# ─────────────────────────────────────────────────────────────────────────
# VLM summariser (_summarize_screen_via_vlm)
# ─────────────────────────────────────────────────────────────────────────
class VlmSummariseTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")
        self._redirect_paths()

    def test_returns_none_when_bobert_absent(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            self.assertIsNone(self.mod._summarize_screen_via_vlm(b"png"))

    def test_local_vlm_clean_json(self):
        bc = _FakeBobert()
        payload = '{"summary":"VS Code open","entities":["main.py"],"sensitive":false,"sensitive_reason":""}'
        bc._call_local_vision = mock.MagicMock(return_value=payload)
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._summarize_screen_via_vlm(b"png")
        self.assertEqual(out["summary"], "VS Code open")
        self.assertEqual(out["entities"], ["main.py"])
        self.assertFalse(out["sensitive"])
        bc._call_local_vision.assert_called_once()

    def test_local_vlm_json_with_codefence(self):
        bc = _FakeBobert()
        bc._call_local_vision = mock.MagicMock(
            return_value='[local-vision] ```{"summary":"docs","entities":[],"sensitive":true,"sensitive_reason":"pw"}```')
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._summarize_screen_via_vlm(b"png")
        self.assertTrue(out["sensitive"])
        self.assertEqual(out["sensitive_reason"], "pw")

    def test_local_vlm_non_json_text(self):
        bc = _FakeBobert()
        bc._call_local_vision = mock.MagicMock(return_value="just prose, no braces")
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._summarize_screen_via_vlm(b"png")
        self.assertEqual(out["summary"], "just prose, no braces")
        self.assertTrue(out["raw"])

    def test_local_vlm_malformed_json_falls_to_raw(self):
        bc = _FakeBobert()
        bc._call_local_vision = mock.MagicMock(return_value="{not: valid json,,}")
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._summarize_screen_via_vlm(b"png")
        self.assertTrue(out["raw"])

    def test_local_vlm_exception_then_cloud(self):
        bc = _FakeBobert()
        bc._call_local_vision = mock.MagicMock(side_effect=RuntimeError("vlm down"))
        bc.ask_vision = mock.MagicMock(
            return_value='{"summary":"cloud said","entities":[],"sensitive":false}')
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._summarize_screen_via_vlm(b"png")
        self.assertEqual(out["summary"], "cloud said")
        bc.ask_vision.assert_called_once()

    def test_cloud_skipped_when_budget_exhausted(self):
        bc = _FakeBobert()
        bc._call_local_vision = mock.MagicMock(return_value="")  # local gives nothing
        bc.ask_vision = mock.MagicMock(return_value='{"summary":"x"}')
        bc.AMBIENT_VISION_BUDGET_USD = 0.0  # no budget → cloud refused
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._summarize_screen_via_vlm(b"png")
        self.assertIsNone(out)
        bc.ask_vision.assert_not_called()

    def test_cloud_exception_returns_none(self):
        bc = _FakeBobert()
        bc._call_local_vision = mock.MagicMock(return_value="")
        bc.ask_vision = mock.MagicMock(side_effect=RuntimeError("cloud boom"))
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._summarize_screen_via_vlm(b"png")
        self.assertIsNone(out)

    def test_no_vlm_helpers_returns_none(self):
        bc = _FakeBobert()  # neither _call_local_vision nor ask_vision present
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._summarize_screen_via_vlm(b"png")
        self.assertIsNone(out)


# ─────────────────────────────────────────────────────────────────────────
# Screen-snapshot worker (_screen_worker_loop)
# ─────────────────────────────────────────────────────────────────────────
class ScreenWorkerLoopTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._screen_entries_total = 0
        self.mod._screen_skipped_total = 0
        self.mod._screen_blocked_total = 0
        self.mod._screen_last_phash = None
        self.mod._screen_last_error = None

    def _png(self, shade):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (shade, shade, shade)).save(buf, format="PNG")
        return buf.getvalue()

    def _run(self, bc, *, wait_returns, title="Notepad", proc="notepad.exe",
             summarize=_SENTINEL):
        evt = _ScriptedEvent(wait_returns)
        ctx = [
            inject_modules(),  # placeholder to keep a uniform 'with' list
            mock.patch.object(self.mod, "_get_bobert", return_value=bc),
            mock.patch.object(self.mod, "_screen_stop_evt", evt),
            mock.patch.object(self.mod, "_focused_window_title", return_value=title),
            mock.patch.object(self.mod, "_focused_proc_name", return_value=proc),
        ]
        if summarize is not _SENTINEL:
            ctx.append(mock.patch.object(self.mod, "_summarize_screen_via_vlm",
                                         return_value=summarize))
        with contextlib.ExitStack() as stack:
            for c in ctx:
                stack.enter_context(c)
            self.mod._screen_worker_loop()
        return evt

    def test_screen_logs_entry(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"mon1": self._png(80)})
        self._run(bc, wait_returns=[True],
                  summarize={"summary": "code", "entities": ["x"],
                             "sensitive": False, "sensitive_reason": ""})
        self.assertEqual(self.mod._screen_entries_total, 1)
        with open(self.mod._SCREEN_JSONL, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["source"], "screen")
        self.assertEqual(entry["summary"], "code")

    def test_screen_redacts_sensitive(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"mon1": self._png(80)})
        self._run(bc, wait_returns=[True],
                  summarize={"summary": "secret stuff", "entities": ["pw"],
                             "sensitive": True, "sensitive_reason": "password"})
        with open(self.mod._SCREEN_JSONL, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        self.assertIn("redacted", entry["summary"])
        self.assertEqual(entry["entities"], [])
        self.assertEqual(entry["sensitive_reason"], "password")

    def test_screen_blocked_window(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"mon1": self._png(80)})
        self._run(bc, wait_returns=[True], title="1Password Vault",
                  proc="1password.exe")
        self.assertEqual(self.mod._screen_blocked_total, 1)
        self.assertEqual(self.mod._screen_entries_total, 0)

    def test_screen_budget_exhausted_skips(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 0.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"mon1": self._png(80)})
        self._run(bc, wait_returns=[True])
        self.assertEqual(self.mod._screen_skipped_total, 1)
        bc.take_all_monitor_screenshots.assert_not_called()

    def test_screen_dedupes_identical_frame(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"mon1": self._png(80)})
        # Pre-seed the prior pHash to the same image's hash → hamming 0 → skip.
        self.mod._screen_last_phash = self.mod._phash64(self._png(80))
        self._run(bc, wait_returns=[True])
        self.assertEqual(self.mod._screen_skipped_total, 1)
        self.assertEqual(self.mod._screen_entries_total, 0)

    def test_screen_no_screenshot_bytes_skips(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(return_value={})
        self._run(bc, wait_returns=[True])
        self.assertEqual(self.mod._screen_skipped_total, 1)

    def test_screen_capture_exception(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            side_effect=RuntimeError("grab failed"))
        self._run(bc, wait_returns=[True])
        self.assertIn("screenshot capture failed", self.mod._screen_last_error)

    def test_screen_summarize_none_skips(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"mon1": self._png(80)})
        self._run(bc, wait_returns=[True], summarize=None)
        self.assertEqual(self.mod._screen_skipped_total, 1)
        self.assertEqual(self.mod._screen_entries_total, 0)

    def test_screen_pause_branch(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"mon1": self._png(80)})
        self.mod._paused[0] = True
        self.addCleanup(lambda: self.mod._paused.__setitem__(0, False))
        self._run(bc, wait_returns=[True])
        bc.take_all_monitor_screenshots.assert_not_called()

    def test_screen_single_monitor_fallback(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        # No take_all; only take_screenshot(None).
        bc.take_screenshot = mock.MagicMock(return_value=self._png(80))
        self._run(bc, wait_returns=[True],
                  summarize={"summary": "one", "entities": [],
                             "sensitive": False, "sensitive_reason": ""})
        self.assertEqual(self.mod._screen_entries_total, 1)

    def test_screen_summary_runs_as_tagged_background_work(self):
        # 2026-09-29: the periodic VLM summary must not evict the local
        # brain's warm conversation prefix mid-chat, so it runs as tagged
        # background work (core.local_traffic's gate then defers its local
        # POST, bounded) and the daemon's stop event ends that wait.
        from core import local_traffic as lt
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"mon1": self._png(80)})
        seen = []

        def _summ(png):
            job = lt.current_job()
            seen.append((job.tag, job.cancel()) if job else None)
            return {"summary": "code", "entities": [], "sensitive": False,
                    "sensitive_reason": ""}
        with mock.patch.object(self.mod, "_summarize_screen_via_vlm",
                               side_effect=_summ):
            self._run(bc, wait_returns=[True])
        self.assertEqual(seen, [("ambient-screen", False)])
        self.assertIsNone(lt.current_job())
        self.assertEqual(self.mod._screen_entries_total, 1)

    def test_screen_bobert_absent(self):
        evt = _ScriptedEvent([True])
        with mock.patch.object(self.mod, "_get_bobert", return_value=None), \
             mock.patch.object(self.mod, "_screen_stop_evt", evt):
            self.mod._screen_worker_loop()
        self.assertIn("bobert_companion not loaded", self.mod._screen_last_error)

    def test_screen_no_helper_available(self):
        bc = _FakeBobert()  # neither take_all nor take_one
        evt = _ScriptedEvent([True])
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_screen_stop_evt", evt):
            self.mod._screen_worker_loop()
        self.assertIn("no screenshot helper", self.mod._screen_last_error)


# ─────────────────────────────────────────────────────────────────────────
# Persistence + jsonl helpers
# ─────────────────────────────────────────────────────────────────────────
class PersistenceTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._buffer.clear()

    def test_persist_state_writes_file(self):
        bc = _FakeBobert()
        bc.AMBIENT_LISTEN_BUFFER_MINUTES = 10
        self.mod._buffer.append({"ts": time.time(), "text": "hi"})
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._persist_state()
        with open(self.mod._STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["entries"][0]["text"], "hi")
        self.assertIn("audio_daemon", data)
        self.assertIn("screen_daemon", data)

    def test_persist_state_swallows_write_error(self):
        # Point _PROJECT_DIR at a path that can't be created → mkstemp fails,
        # but _persist_state must not raise.
        self.mod._PROJECT_DIR = os.path.join(self.tmp, "does", "not", "exist")
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            self.mod._persist_state()  # should print + return, not raise

    def test_append_jsonl_writes_line(self):
        path = os.path.join(self.tmp, "out.jsonl")
        self.mod._append_jsonl(path, {"a": 1})
        self.mod._append_jsonl(path, {"b": 2})
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[1]), {"b": 2})

    def test_append_jsonl_handles_bad_path(self):
        # A directory path can't be opened for append → caught, no raise.
        self.mod._append_jsonl(self.tmp, {"a": 1})

    def test_rotate_jsonl_noop_when_small(self):
        path = os.path.join(self.tmp, "small.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"x":1}\n')
        self.mod._rotate_jsonl_if_needed(path, cap=10)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 1)

    def test_rotate_jsonl_trims_to_cap(self):
        path = os.path.join(self.tmp, "big.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(40):
                f.write(json.dumps({"i": i}) + "\n")
        # cap=10 → triggers when > 15 lines; keeps last 10.
        self.mod._rotate_jsonl_if_needed(path, cap=10)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 10)
        self.assertEqual(json.loads(lines[-1])["i"], 39)

    def test_rotate_jsonl_missing_file_noop(self):
        self.mod._rotate_jsonl_if_needed(os.path.join(self.tmp, "nope.jsonl"))


# ─────────────────────────────────────────────────────────────────────────
# Budget save/load round-trips + edge cases
# ─────────────────────────────────────────────────────────────────────────
class BudgetIoTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")
        self._redirect_paths()

    def test_load_budget_missing_returns_empty(self):
        self.assertEqual(self.mod._load_budget(), {})

    def test_save_then_load_roundtrip(self):
        self.mod._save_budget({"2026-06-01": 0.25})
        self.assertEqual(self.mod._load_budget(), {"2026-06-01": 0.25})

    def test_load_budget_non_dict_returns_empty(self):
        with open(self.mod._BUDGET_PATH, "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]")
        self.assertEqual(self.mod._load_budget(), {})

    def test_load_budget_corrupt_returns_empty(self):
        with open(self.mod._BUDGET_PATH, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(self.mod._load_budget(), {})

    def test_budget_today_key_format(self):
        k = self.mod._budget_today_key()
        self.assertRegex(k, r"^\d{4}-\d{2}-\d{2}$")


# ─────────────────────────────────────────────────────────────────────────
# focused-window ctypes helpers (Windows) + non-Windows fallback
# ─────────────────────────────────────────────────────────────────────────
class FocusedWindowTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")

    def test_title_non_windows_returns_empty(self):
        with mock.patch.object(self.mod.sys, "platform", "linux"):
            self.assertEqual(self.mod._focused_window_title(), "")

    def test_proc_non_windows_returns_empty(self):
        with mock.patch.object(self.mod.sys, "platform", "linux"):
            self.assertEqual(self.mod._focused_proc_name(), "")

    def test_title_windows_reads_buffer(self):
        if sys.platform != "win32":
            self.skipTest("ctypes.windll only exists on Windows")
        user32 = mock.MagicMock()
        user32.GetForegroundWindow.return_value = 1234
        user32.GetWindowTextLengthW.return_value = 5

        def _get_text(hwnd, buf, n):
            buf.value = "Hello"
            return 5
        user32.GetWindowTextW.side_effect = _get_text
        with mock.patch("ctypes.windll") as windll:
            windll.user32 = user32
            self.assertEqual(self.mod._focused_window_title(), "Hello")

    def test_title_windows_no_hwnd(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        user32 = mock.MagicMock()
        user32.GetForegroundWindow.return_value = 0
        with mock.patch("ctypes.windll") as windll:
            windll.user32 = user32
            self.assertEqual(self.mod._focused_window_title(), "")

    def test_title_windows_exception_returns_empty(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        user32 = mock.MagicMock()
        user32.GetForegroundWindow.side_effect = RuntimeError("boom")
        with mock.patch("ctypes.windll") as windll:
            windll.user32 = user32
            self.assertEqual(self.mod._focused_window_title(), "")


# ─────────────────────────────────────────────────────────────────────────
# audio-processing + speaker-id + voice-id passthroughs
# ─────────────────────────────────────────────────────────────────────────
class AudioProcAndSpeakerTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")
        # Reset the function-attr cache on _voice_id_module between tests.
        if hasattr(self.mod._voice_id_module, "_mod"):
            del self.mod._voice_id_module._mod

    def test_apply_audio_processing_master_disabled_passthrough(self):
        bc = _FakeBobert()
        bc._audio_master_enabled = [False]
        arr = np.ones(10, dtype=np.float32)
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._apply_audio_processing(arr, 16000)
        self.assertIs(out, arr)

    def test_apply_audio_processing_processor_failure_passthrough(self):
        # get_processor() raising must fall through to the raw batch. (The real
        # core.audio_processor is already importable on the dev box, so we make
        # the processor itself blow up rather than block the import.)
        bc = _FakeBobert()
        bc._audio_master_enabled = [True]
        arr = np.ones(10, dtype=np.float32)
        ap = types.ModuleType("core.audio_processor")
        ap.get_processor = mock.MagicMock(side_effect=RuntimeError("proc down"))
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             inject_modules(**{"core.audio_processor": ap}):
            out = self.mod._apply_audio_processing(arr, 16000)
        self.assertIs(out, arr)

    def test_apply_audio_processing_runs_processor(self):
        bc = _FakeBobert()
        bc._audio_master_enabled = [True]
        arr = np.ones(10, dtype=np.float32)
        processed = np.zeros(10, dtype=np.float32)
        ap = types.ModuleType("core.audio_processor")
        proc = mock.MagicMock()
        proc.process.return_value = processed
        ap.get_processor = mock.MagicMock(return_value=proc)
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             inject_modules(**{"core.audio_processor": ap}):
            out = self.mod._apply_audio_processing(arr, 16000)
        self.assertIs(out, processed)

    def test_identify_speaker_no_module(self):
        with mock.patch.object(self.mod, "_voice_id_module", return_value=None):
            sid, score = self.mod._identify_speaker_safe(np.ones(10), 16000)
        self.assertIsNone(sid)
        self.assertEqual(score, 0.0)

    def test_identify_speaker_none_enrolled(self):
        vid = mock.MagicMock()
        vid.list_enrolled.return_value = []
        with mock.patch.object(self.mod, "_voice_id_module", return_value=vid):
            sid, score = self.mod._identify_speaker_safe(np.ones(10), 16000)
        self.assertIsNone(sid)
        vid.identify_speaker.assert_not_called()

    def test_identify_speaker_returns_id(self):
        vid = mock.MagicMock()
        vid.list_enrolled.return_value = ["alice"]
        vid.identify_speaker.return_value = ("alice", 0.91)
        with mock.patch.object(self.mod, "_voice_id_module", return_value=vid):
            sid, score = self.mod._identify_speaker_safe(np.ones(10), 16000)
        self.assertEqual(sid, "alice")
        self.assertAlmostEqual(score, 0.91)

    def test_identify_speaker_swallows_exception(self):
        vid = mock.MagicMock()
        vid.list_enrolled.side_effect = RuntimeError("resemblyzer boom")
        with mock.patch.object(self.mod, "_voice_id_module", return_value=vid):
            sid, score = self.mod._identify_speaker_safe(np.ones(10), 16000)
        self.assertIsNone(sid)
        self.assertEqual(score, 0.0)

    def test_voice_id_module_caches_none_on_import_failure(self):
        real_import = __import__

        def _fake(name, *a, **k):
            if name == "core.voice_id" or name == "core":
                raise ImportError("no voice_id")
            return real_import(name, *a, **k)
        with mock.patch("builtins.__import__", side_effect=_fake):
            out = self.mod._voice_id_module()
        self.assertIsNone(out)
        # Cached: a second call returns the cached None without re-import.
        self.assertIsNone(self.mod._voice_id_module())

    def test_voice_id_module_returns_cached(self):
        sentinel = object()
        self.mod._voice_id_module._mod = sentinel
        self.assertIs(self.mod._voice_id_module(), sentinel)


# ─────────────────────────────────────────────────────────────────────────
# _safe_close_stream
# ─────────────────────────────────────────────────────────────────────────
class SafeCloseStreamTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")

    def test_none_is_noop(self):
        self.mod._safe_close_stream(None)  # must not raise

    def test_uses_bobert_helper_when_present(self):
        bc = _FakeBobert()
        bc._safe_close_stream = mock.MagicMock()
        stream = _FakeStream()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._safe_close_stream(stream)
        bc._safe_close_stream.assert_called_once_with(stream)

    def test_bobert_helper_exception_swallowed(self):
        bc = _FakeBobert()
        bc._safe_close_stream = mock.MagicMock(side_effect=RuntimeError("x"))
        stream = _FakeStream()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._safe_close_stream(stream)  # swallowed

    def test_fallback_daemon_close(self):
        # No bobert helper → falls back to stop() + daemon close().
        stream = _FakeStream()
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            self.mod._safe_close_stream(stream)
        self.assertTrue(stream.stopped)
        # close() runs on a daemon thread; give the wait() loop a beat by
        # asserting it eventually closed (done.wait has 2 s budget).
        self.assertTrue(stream.closed)


# ─────────────────────────────────────────────────────────────────────────
# Action handlers: start/stop/status/full/mic-only.
# Threads are neutered by the harness, so these drive the pure orchestration.
# ─────────────────────────────────────────────────────────────────────────
class ActionOrchestrationTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._thread = None
        self.mod._audio_thread = None
        self.mod._screen_thread = None
        self.mod._buffer.clear()
        # Never really sleep in the start actions.
        p = mock.patch.object(self.mod.time, "sleep", lambda *_a, **_k: None)
        p.start()
        self.addCleanup(p.stop)

    def _dead_thread(self):
        t = mock.MagicMock()
        t.is_alive.return_value = False
        return t

    def _live_thread(self):
        t = mock.MagicMock()
        t.is_alive.return_value = True
        return t

    def _alive_then_dead_thread(self):
        """is_alive() True on the guard check, False after join() — i.e. a
        thread that exists and is running, then stops cleanly when asked."""
        t = mock.MagicMock()
        t.is_alive.side_effect = [True, False, False, False]
        return t

    # ── listen start ────────────────────────────────────────────────────
    def test_listen_start_already_active(self):
        self.mod._thread = self._live_thread()
        out = self.actions["ambient_listen_start"]("")
        self.assertIn("already active", out)

    def test_listen_start_success_path(self):
        bc = _FakeBobert()
        bc.AMBIENT_LISTEN_BUFFER_MINUTES = 7
        # Thread.start is neutered (harness), so the thread never goes alive;
        # but with no _last_error the success branch returns the engaged msg.
        with mock.patch.object(self.mod, "_wake_listener_active", return_value=False), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            # Make the just-created thread report alive so the error branch is skipped.
            with mock.patch.object(threading.Thread, "is_alive", lambda self: True):
                out = self.actions["ambient_listen_start"]("")
        self.assertIn("Ambient listening engaged", out)
        self.assertIn("7-minute", out)

    def test_listen_start_reports_worker_error(self):
        # The action clears _last_error then starts the worker thread. Simulate
        # the worker dying instantly by having (the patched) Thread.start set
        # the module-level _last_error, with is_alive() False afterwards.
        def _start_sets_error(_self):
            self.mod._last_error = "InputStream open failed: boom"
        with mock.patch.object(self.mod, "_wake_listener_active", return_value=False), \
             mock.patch.object(threading.Thread, "start", _start_sets_error), \
             mock.patch.object(threading.Thread, "is_alive", lambda self: False):
            out = self.actions["ambient_listen_start"]("")
        self.assertIn("failed to start", out)
        self.assertIn("boom", out)

    # ── listen stop ─────────────────────────────────────────────────────
    def test_listen_stop_joins_and_reports(self):
        t = self._alive_then_dead_thread()
        self.mod._thread = t
        self.mod._started_at = time.time() - 12
        self.mod._buffer.append({"ts": time.time(), "text": "x"})
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            out = self.actions["ambient_listen_stop"]("")
        self.assertIn("disengaged", out)
        self.assertIn("1 entries", out)
        t.join.assert_called_once()

    def test_listen_stop_unclean(self):
        t = self._live_thread()  # stays alive after join → unclean
        self.mod._thread = t
        self.mod._started_at = time.time()
        out = self.actions["ambient_listen_stop"]("")
        self.assertIn("did not stop cleanly", out)

    # ── audio start/stop ────────────────────────────────────────────────
    def test_audio_start_non_windows(self):
        with mock.patch.object(self.mod.sys, "platform", "linux"):
            out = self.actions["ambient_audio_start"]("")
        self.assertIn("Windows-only", out)

    def test_audio_start_already_active(self):
        with mock.patch.object(self.mod.sys, "platform", "win32"):
            self.mod._audio_thread = self._live_thread()
            out = self.actions["ambient_audio_start"]("")
        self.assertIn("already active", out)

    def test_audio_start_success(self):
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(threading.Thread, "is_alive", lambda self: True):
            out = self.actions["ambient_audio_start"]("")
        self.assertIn("System-audio capture engaged", out)

    def test_audio_start_reports_error(self):
        def _start_sets_error(_self):
            self.mod._audio_last_error = "no WASAPI loopback device"
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(threading.Thread, "start", _start_sets_error), \
             mock.patch.object(threading.Thread, "is_alive", lambda self: False):
            out = self.actions["ambient_audio_start"]("")
        self.assertIn("failed to start", out)

    def test_audio_stop_not_running(self):
        out = self.actions["ambient_audio_stop"]("")
        self.assertIn("not running", out)

    def test_audio_stop_reports(self):
        t = self._alive_then_dead_thread()
        self.mod._audio_thread = t
        self.mod._audio_started_at = time.time() - 5
        self.mod._audio_entries_total = 3
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            out = self.actions["ambient_audio_stop"]("")
        self.assertIn("disengaged", out)
        self.assertIn("3 entries", out)

    def test_audio_stop_unclean(self):
        self.mod._audio_thread = self._live_thread()
        self.mod._audio_started_at = time.time()
        out = self.actions["ambient_audio_stop"]("")
        self.assertIn("did not stop cleanly", out)

    # ── screen start/stop ───────────────────────────────────────────────
    # 2026-10-05: ambient_screen_start is SCREEN MEMORY (core.screen_memory,
    # text only, no AI calls); the old vision-model loop runs only with
    # AMBIENT_SCREEN_VLM_ENABLED.
    def test_screen_start_already_active(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_VLM_ENABLED = True
        self.mod._screen_thread = self._live_thread()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.actions["ambient_screen_start"]("")
        self.assertIn("already active", out)

    def test_screen_start_success(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_VLM_ENABLED = True
        bc.AMBIENT_SCREEN_INTERVAL_S = 45.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.actions["ambient_screen_start"]("")
        self.assertIn("Screen watcher engaged", out)
        self.assertIn("45s", out)

    def test_screen_start_is_screen_memory_by_default(self):
        from core import screen_memory as SW
        bc = _FakeBobert()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
                mock.patch.object(SW, "is_running", return_value=False), \
                mock.patch.object(SW, "start", return_value=True) as st:
            out = self.actions["ambient_screen_start"]("")
        st.assert_called_once_with()
        self.assertIn("text only, no AI calls", out)
        self.assertIsNone(self.mod._screen_thread)

    def test_screen_stop_reports(self):
        t = self._alive_then_dead_thread()
        self.mod._screen_thread = t
        self.mod._screen_started_at = time.time() - 9
        self.mod._screen_entries_total = 2
        self.mod._screen_skipped_total = 4
        self.mod._screen_blocked_total = 1
        from core import screen_memory as SW
        with mock.patch.object(self.mod, "_get_bobert", return_value=None), \
                mock.patch.object(SW, "is_running", return_value=False):
            out = self.actions["ambient_screen_stop"]("")
        self.assertIn("2 snapshots", out)
        self.assertIn("4 skipped", out)
        self.assertIn("1 blocked", out)

    def test_screen_stop_unclean(self):
        self.mod._screen_thread = self._live_thread()
        self.mod._screen_started_at = time.time()
        out = self.actions["ambient_screen_stop"]("")
        self.assertIn("did not stop cleanly", out)

    # ── full + mic-only composites ──────────────────────────────────────
    def test_full_start_composes(self):
        with mock.patch.object(self.mod, "ambient_listen_start", return_value="A, x"), \
             mock.patch.object(self.mod, "ambient_audio_start", return_value="B, y"), \
             mock.patch.object(self.mod, "ambient_screen_start", return_value="C"):
            out = self.actions["ambient_full_start"]("")
        self.assertIn("Full ambient mode engaged", out)
        self.assertIn("A", out)
        self.assertIn("C", out)

    def test_full_stop_composes(self):
        with mock.patch.object(self.mod, "ambient_listen_stop", return_value="a"), \
             mock.patch.object(self.mod, "ambient_audio_stop", return_value="b"), \
             mock.patch.object(self.mod, "ambient_screen_stop", return_value="c"):
            out = self.actions["ambient_full_stop"]("")
        self.assertIn("disengaged", out)
        self.assertIn("a | b | c", out)

    def test_mic_only_starts_mic_when_off(self):
        with mock.patch.object(self.mod, "ambient_listen_start", return_value="started") as start, \
             mock.patch.object(self.mod, "ambient_audio_stop", return_value="audio off"), \
             mock.patch.object(self.mod, "ambient_screen_stop", return_value="screen off"):
            out = self.actions["ambient_mic_only"]("")
        start.assert_called_once()
        self.assertIn("Mic-only", out)

    def test_mic_only_skips_start_when_mic_already_on(self):
        self.mod._thread = self._live_thread()
        with mock.patch.object(self.mod, "ambient_listen_start") as start, \
             mock.patch.object(self.mod, "ambient_audio_stop", return_value="audio off"), \
             mock.patch.object(self.mod, "ambient_screen_stop", return_value="screen off"):
            out = self.actions["ambient_mic_only"]("")
        start.assert_not_called()
        self.assertIn("Mic-only", out)

    # ── status variants ─────────────────────────────────────────────────
    def test_status_all_off_with_errors(self):
        self.mod._last_error = "mic boom"
        self.mod._audio_last_error = "audio boom"
        self.mod._screen_last_error = "screen boom"
        bc = _FakeBobert()
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.actions["ambient_listen_status"]("")
        self.assertIn("mic OFF", out)
        self.assertIn("mic boom", out)
        self.assertIn("audio boom", out)
        self.assertIn("screen boom", out)

    def test_status_mic_on_with_buffer_and_stale_heartbeat(self):
        self.mod._thread = self._live_thread()
        self.mod._started_at = time.time() - 100
        self.mod._heartbeat = time.time() - 999  # stale
        long_text = "x" * 90
        self.mod._buffer.append({"ts": time.time(), "text": long_text})
        bc = _FakeBobert()
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.actions["ambient_listen_status"]("")
        self.assertIn("mic ON", out)
        self.assertIn("heartbeat stale", out)
        # Long text is truncated with an ellipsis.
        self.assertIn("…", out)

    def test_status_audio_and_screen_on(self):
        self.mod._audio_thread = self._live_thread()
        self.mod._screen_thread = self._live_thread()
        self.mod._audio_entries_total = 5
        self.mod._screen_entries_total = 2
        bc = _FakeBobert()
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.actions["ambient_listen_status"]("")
        self.assertIn("audio ON", out)
        self.assertIn("screen ON", out)


# ─────────────────────────────────────────────────────────────────────────
# set_paused + register autostart
# ─────────────────────────────────────────────────────────────────────────
class MiscTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")

    def test_set_paused_toggles_flag(self):
        self.mod.set_paused(True)
        self.assertTrue(self.mod._paused[0])
        self.mod.set_paused(False)
        self.assertFalse(self.mod._paused[0])
        self.addCleanup(lambda: self.mod._paused.__setitem__(0, False))

    def test_register_populates_all_actions(self):
        acts: dict = {}
        self.mod.register(acts)
        for name in ("ambient_listen_start", "ambient_listen_stop",
                     "ambient_listen_status", "ambient_audio_start",
                     "ambient_audio_stop", "ambient_screen_start",
                     "ambient_screen_stop", "ambient_full_start",
                     "ambient_full_stop", "ambient_mic_only"):
            self.assertIn(name, acts)

    def test_register_autostart_thread_spawns_when_enabled(self):
        bc = _FakeBobert()
        bc.AMBIENT_LISTEN_ENABLED = True
        acts: dict = {}
        # Thread.start is neutered by the harness only inside load_skill_isolated;
        # here we patch it ourselves to confirm a thread is constructed.
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(threading.Thread, "start", lambda self: None), \
             mock.patch.object(self.mod.threading, "Thread",
                               wraps=self.mod.threading.Thread) as TH:
            self.mod.register(acts)
        # At least one Thread was constructed (the autostart watcher).
        self.assertTrue(TH.called)

    def test_register_autostart_runs_starts(self):
        # Exercise the _bg_autostart inner function body directly by capturing
        # the target passed to Thread, then invoking it with sleep + starts stubbed.
        bc = _FakeBobert()
        bc.AMBIENT_LISTEN_ENABLED = True
        bc.AMBIENT_AUDIO_ENABLED = True
        bc.AMBIENT_SCREEN_ENABLED = True
        captured = {}

        class _CapThread:
            def __init__(self, target=None, **kw):
                captured["target"] = target

            def start(self):
                pass
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod.threading, "Thread", _CapThread), \
             mock.patch.object(self.mod.time, "sleep", lambda *_a: None), \
             mock.patch.object(self.mod, "ambient_listen_start") as a1, \
             mock.patch.object(self.mod, "ambient_audio_start") as a2, \
             mock.patch.object(self.mod, "ambient_screen_start") as a3:
            self.mod.register({})
            self.assertIn("target", captured)
            captured["target"]()  # run the autostart body
        a1.assert_called_once()
        a2.assert_called_once()
        a3.assert_called_once()

    def _run_autostart(self, bc):
        captured = {}

        class _CapThread:
            def __init__(self, target=None, **kw):
                captured["target"] = target

            def start(self):
                pass
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod.threading, "Thread", _CapThread), \
             mock.patch.object(self.mod.time, "sleep", lambda *_a: None), \
             mock.patch.object(self.mod, "ambient_listen_start") as start, \
             mock.patch.object(self.mod, "ambient_audio_start"), \
             contextlib.redirect_stdout(io.StringIO()):
            self.mod.register({})
            captured["target"]()
        return start

    def test_register_autostart_respects_an_explicit_ambient_off(self):
        """REGRESSION (2026-10-01). A tray / voice "ambient mode off" did not
        survive a restart: this autostart put room transcription back on. The
        OFF is now saved as AMBIENT_LISTEN_ENABLED=False
        (core.actions._act_ambient_mode_set), the one key this autostart reads
        for the mic, so the room mic stays closed even while another ambient
        source (system audio here) still starts the autostart thread."""
        bc = _FakeBobert(AMBIENT_LISTEN_ENABLED=False, AMBIENT_AUDIO_ENABLED=True)
        self._run_autostart(bc).assert_not_called()
        bc = _FakeBobert(AMBIENT_LISTEN_ENABLED=True)
        self._run_autostart(bc).assert_called_once()


# ─────────────────────────────────────────────────────────────────────────
# Residual edge-path coverage (guards, exception swallows, loop-continue arms).
# ─────────────────────────────────────────────────────────────────────────
class ResidualEdgeTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self._redirect_paths()

    # ── _compile_wake_pattern: blank phrase skipped ─────────────────────
    def test_wake_pattern_skips_blank_phrase(self):
        bc = _FakeBobert()
        bc.WAKE_PHRASES = {"", "   ", "jarvis"}  # blanks skipped via `continue`
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            pat = self.mod._compile_wake_pattern()
        self.assertTrue(pat.search("jarvis"))

    # ── _maybe_nudge_wake guards ────────────────────────────────────────
    def test_nudge_no_pattern_returns(self):
        self.mod._wake_pattern = None
        # Must simply return without error.
        self.mod._maybe_nudge_wake("jarvis")

    def test_nudge_empty_text_returns(self):
        import re
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._maybe_nudge_wake("")

    def test_nudge_no_match_returns(self):
        import re
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=mock.MagicMock()) as g:
            self.mod._maybe_nudge_wake("nothing here")
        g.return_value.proactive_announce.assert_not_called()

    def test_nudge_bobert_none_returns(self):
        import re
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            self.mod._maybe_nudge_wake("jarvis")  # no crash, just returns

    def test_nudge_sleep_mode_message(self):
        import re
        bc = _FakeBobert()
        bc._sleep_mode = [True]
        bc.proactive_announce = mock.MagicMock()
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("jarvis")
        self.assertIn("listening", bc.proactive_announce.call_args[0][0].lower())

    def test_nudge_announce_exception_swallowed(self):
        import re
        bc = _FakeBobert()
        bc._sleep_mode = [False]
        bc.proactive_announce = mock.MagicMock(side_effect=RuntimeError("boom"))
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("jarvis")  # exception swallowed

    def test_nudge_announce_not_callable(self):
        import re
        bc = _FakeBobert()
        bc._sleep_mode = [False]
        bc.proactive_announce = "not callable"
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("jarvis")

    # ── _wake_listener_active branches ──────────────────────────────────
    def test_wake_listener_detector_none(self):
        wl = mock.MagicMock()
        wl._detector = None
        with mock.patch.dict(sys.modules, {"skill_wake_listener": wl}):
            self.assertFalse(self.mod._wake_listener_active())

    def test_wake_listener_is_running_raises(self):
        wl = mock.MagicMock()
        wl._detector.is_running.side_effect = RuntimeError("boom")
        with mock.patch.dict(sys.modules, {"skill_wake_listener": wl}):
            self.assertFalse(self.mod._wake_listener_active())

    # ── _ensure_data_dir exception swallow ──────────────────────────────
    def test_ensure_data_dir_swallows(self):
        with mock.patch.object(self.mod.os, "makedirs",
                               side_effect=OSError("nope")):
            self.mod._ensure_data_dir()  # must not raise

    # ── _persist_state tmp-unlink on dump error ─────────────────────────
    def test_persist_state_unlinks_tmp_on_dump_error(self):
        # json.dump raising mid-write → the tmp file is unlinked, no raise.
        with mock.patch.object(self.mod, "_get_bobert", return_value=None), \
             mock.patch.object(self.mod.json, "dump",
                               side_effect=RuntimeError("disk full")):
            self.mod._persist_state()
        # No .tmp files left behind in the project dir.
        leftover = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        self.assertEqual(leftover, [])

    # ── _save_budget exception swallow ──────────────────────────────────
    def test_save_budget_swallows_error(self):
        with mock.patch.object(self.mod.json, "dump",
                               side_effect=RuntimeError("boom")):
            self.mod._save_budget({"k": 1.0})  # must not raise

    # ── _rotate_jsonl write-error path ──────────────────────────────────
    def test_rotate_jsonl_write_error_swallowed(self):
        path = os.path.join(self.tmp, "big.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(40):
                f.write(json.dumps({"i": i}) + "\n")
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=RuntimeError("rename boom")):
            self.mod._rotate_jsonl_if_needed(path, cap=10)  # swallowed

    # ── _safe_close_stream fallback: stop()/close() raise ───────────────
    def test_safe_close_fallback_stop_and_close_raise(self):
        stream = mock.MagicMock()
        stream.stop.side_effect = RuntimeError("stop boom")
        stream.close.side_effect = RuntimeError("close boom")
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            self.mod._safe_close_stream(stream)  # both swallowed, no raise

    def test_safe_close_fallback_timeout_never_calls_sd_stop(self):
        # H-3 (2026-08-20): done.wait() returns False (close hangs) → the
        # daemon is ABANDONED. The old escape hatch called the process-global
        # sd.stop() under a comment claiming it would "force-stop every stream
        # globally"; sounddevice 0.5.5 stops+closes only `_last_callback`,
        # which nothing but sd.play()/sd.rec()/sd.playrec() publishes. It could
        # never free this explicitly constructed InputStream, and its one
        # reachable effect was closing the host's live TTS play stream
        # (double Pa_CloseStream -> 0xc0000374).
        stream = mock.MagicMock()
        sd = _make_sd()
        fake_evt = mock.MagicMock()
        fake_evt.wait.return_value = False  # simulate the daemon close hanging
        with mock.patch.object(self.mod, "_get_bobert", return_value=None), \
             mock.patch.object(self.mod.threading, "Event", return_value=fake_evt), \
             mock.patch.object(self.mod.threading, "Thread"), \
             inject_modules(sounddevice=sd):
            self.mod._safe_close_stream(stream)   # must not raise
        sd.stop.assert_not_called()

    # ── _focused_window_title length<=0 branch (Windows) ────────────────
    def test_focused_title_zero_length(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        user32 = mock.MagicMock()
        user32.GetForegroundWindow.return_value = 99
        user32.GetWindowTextLengthW.return_value = 0  # → returns ""
        with mock.patch("ctypes.windll") as windll:
            windll.user32 = user32
            self.assertEqual(self.mod._focused_window_title(), "")

    # ── _focused_proc_name Windows happy path ───────────────────────────
    def test_focused_proc_name_windows(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        user32 = mock.MagicMock()
        user32.GetForegroundWindow.return_value = 77

        def _pid(_hwnd, pid_ref):
            pid_ref._obj.value = 4242
            return 1
        user32.GetWindowThreadProcessId.side_effect = _pid
        kernel32 = mock.MagicMock()
        kernel32.OpenProcess.return_value = 555

        def _image(h, flags, buf, size_ref):
            buf.value = "C:\\Program Files\\Google\\Chrome\\Application\\Chrome.exe"
            return 1
        kernel32.QueryFullProcessImageNameW.side_effect = _image
        psapi = mock.MagicMock()
        with mock.patch("ctypes.windll") as windll:
            windll.user32 = user32
            windll.kernel32 = kernel32
            windll.psapi = psapi
            out = self.mod._focused_proc_name()
        self.assertEqual(out, "chrome.exe")
        self.assertEqual(kernel32.OpenProcess.call_args[0], (0x1000, False, 4242))
        kernel32.CloseHandle.assert_called_once_with(555)
        # psapi.GetModuleBaseNameW is ACCESS_DENIED on a limited handle: never used.
        psapi.GetModuleBaseNameW.assert_not_called()

    def test_focused_proc_name_query_fails_returns_empty(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        user32 = mock.MagicMock()
        user32.GetForegroundWindow.return_value = 77

        def _pid(_hwnd, pid_ref):
            pid_ref._obj.value = 4242
            return 1
        user32.GetWindowThreadProcessId.side_effect = _pid
        kernel32 = mock.MagicMock()
        kernel32.OpenProcess.return_value = 555
        kernel32.QueryFullProcessImageNameW.return_value = 0
        with mock.patch("ctypes.windll") as windll:
            windll.user32 = user32
            windll.kernel32 = kernel32
            self.assertEqual(self.mod._focused_proc_name(), "")
        kernel32.CloseHandle.assert_called_once_with(555)

    def test_proc_image_basename_resolves_a_live_process(self):
        # LIVE, no mocks: the old psapi lookup returned '' for every process
        # (146/146 ambient log entries had proc:""), so this fails on it.
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        name = self.mod._proc_image_basename(os.getpid())
        self.assertTrue(name.endswith(".exe"), name)
        self.assertEqual(name, name.lower())
        self.assertIn("python", name)

    def test_proc_image_basename_rejects_bad_pids(self):
        self.assertEqual(self.mod._proc_image_basename(0), "")
        self.assertEqual(self.mod._proc_image_basename(None), "")
        with mock.patch.object(self.mod.sys, "platform", "linux"):
            self.assertEqual(self.mod._proc_image_basename(os.getpid()), "")

    def test_no_getmodulebasename_call_remains(self):
        src = open(self.mod.__file__, encoding="utf-8").read()
        self.assertNotIn("GetModuleBaseNameW(", src)

    def test_focused_proc_name_no_hwnd(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        user32 = mock.MagicMock()
        user32.GetForegroundWindow.return_value = 0
        with mock.patch("ctypes.windll") as windll:
            windll.user32 = user32
            self.assertEqual(self.mod._focused_proc_name(), "")

    def test_focused_proc_name_openprocess_fails(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        user32 = mock.MagicMock()
        user32.GetForegroundWindow.return_value = 77

        def _pid(_hwnd, pid_ref):
            pid_ref._obj.value = 4242
            return 1
        user32.GetWindowThreadProcessId.side_effect = _pid
        kernel32 = mock.MagicMock()
        kernel32.OpenProcess.return_value = 0  # → returns ""
        with mock.patch("ctypes.windll") as windll:
            windll.user32 = user32
            windll.kernel32 = kernel32
            self.assertEqual(self.mod._focused_proc_name(), "")
        kernel32.OpenProcess.assert_called_once()     # reached, not a vacuous pass
        kernel32.QueryFullProcessImageNameW.assert_not_called()

    # ── register autostart exception swallow ────────────────────────────
    def test_register_autostart_exception_swallowed(self):
        bc = _FakeBobert()
        bc.AMBIENT_LISTEN_ENABLED = True
        captured = {}

        class _CapThread:
            def __init__(self, target=None, **kw):
                captured["target"] = target

            def start(self):
                pass
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod.threading, "Thread", _CapThread), \
             mock.patch.object(self.mod.time, "sleep",
                               side_effect=RuntimeError("boom in autostart")):
            self.mod.register({})
            # Running the body must swallow the sleep exception.
            captured["target"]()


# ─────────────────────────────────────────────────────────────────────────
# Screen worker: second-iteration `continue` arms (wait() returns False once).
# ─────────────────────────────────────────────────────────────────────────
class ScreenWorkerContinueTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._screen_entries_total = 0
        self.mod._screen_skipped_total = 0
        self.mod._screen_blocked_total = 0
        self.mod._screen_last_phash = None
        self.mod._screen_last_error = None

    def _png(self, shade):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (shade, shade, shade)).save(buf, format="PNG")
        return buf.getvalue()

    def _run(self, bc, *, wait_returns, title="Notepad", proc="notepad.exe",
             summarize=_SENTINEL):
        evt = _ScriptedEvent(wait_returns)
        ctx = [
            mock.patch.object(self.mod, "_get_bobert", return_value=bc),
            mock.patch.object(self.mod, "_screen_stop_evt", evt),
            mock.patch.object(self.mod, "_focused_window_title", return_value=title),
            mock.patch.object(self.mod, "_focused_proc_name", return_value=proc),
        ]
        if summarize is not _SENTINEL:
            ctx.append(mock.patch.object(self.mod, "_summarize_screen_via_vlm",
                                         return_value=summarize))
        with contextlib.ExitStack() as stack:
            for c in ctx:
                stack.enter_context(c)
            self.mod._screen_worker_loop()

    def test_blocked_then_exit(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"m": self._png(80)})
        # wait() → False (continue, second loop), then True (exit).
        self._run(bc, wait_returns=[False, True], title="Bitwarden",
                  proc="bitwarden.exe")
        self.assertEqual(self.mod._screen_blocked_total, 2)

    def test_budget_exhausted_then_exit(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 0.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"m": self._png(80)})
        self._run(bc, wait_returns=[False, True])
        self.assertEqual(self.mod._screen_skipped_total, 2)

    def test_capture_exception_then_exit(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            side_effect=RuntimeError("grab boom"))
        self._run(bc, wait_returns=[False, True])
        self.assertIn("screenshot capture failed", self.mod._screen_last_error)

    def test_no_bytes_then_exit(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(return_value={})
        self._run(bc, wait_returns=[False, True])
        self.assertEqual(self.mod._screen_skipped_total, 2)

    def test_dedupe_then_exit(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"m": self._png(80)})
        self.mod._screen_last_phash = self.mod._phash64(self._png(80))
        self._run(bc, wait_returns=[False, True])
        self.assertEqual(self.mod._screen_skipped_total, 2)

    def test_summarize_none_then_exit(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"m": self._png(80)})
        self._run(bc, wait_returns=[False, True], summarize=None)
        self.assertGreaterEqual(self.mod._screen_skipped_total, 1)

    def test_pause_then_exit(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        bc.take_all_monitor_screenshots = mock.MagicMock(
            return_value={"m": self._png(80)})
        self.mod._paused[0] = True
        self.addCleanup(lambda: self.mod._paused.__setitem__(0, False))
        self._run(bc, wait_returns=[False, True])
        bc.take_all_monitor_screenshots.assert_not_called()

    def _patterned_png(self, seed):
        """A non-uniform image whose average-hash differs from another seed —
        solid colours all hash identically, so we need real structure to make
        two frames survive the dedupe check."""
        from PIL import Image
        import random
        rng = random.Random(seed)
        im = Image.new("L", (16, 16))
        im.putdata([rng.randint(0, 255) for _ in range(16 * 16)])
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    def test_logged_entry_then_exit(self):
        bc = _FakeBobert()
        bc.AMBIENT_SCREEN_INTERVAL_S = 60.0
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        # Two structurally-different frames so dedupe doesn't fire → both log.
        shots = iter([{"m": self._patterned_png(1)},
                      {"m": self._patterned_png(99)}])
        bc.take_all_monitor_screenshots = mock.MagicMock(
            side_effect=lambda: next(shots))
        self._run(bc, wait_returns=[False, True],
                  summarize={"summary": "x", "entities": [], "sensitive": False,
                             "sensitive_reason": ""})
        self.assertEqual(self.mod._screen_entries_total, 2)


# ─────────────────────────────────────────────────────────────────────────
# Worker loops: multi-iteration arms (sub-batch wait → continue).
# ─────────────────────────────────────────────────────────────────────────
class WorkerContinueTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._buffer.clear()
        self.mod._last_error = None
        self.mod._audio_last_error = None

    def test_mic_sub_batch_waits_then_feeds(self):
        # First drain has too few samples → wait(0.1) returns False (continue);
        # on that wait we feed a full batch; second drain transcribes; then the
        # trailing sub-batch wait returns True → exit.
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("late audio", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        self.mod._wake_pattern = None
        sd = _make_sd()
        big = np.ones(16000 * 3, dtype=np.float32) * 0.2

        def _on_start(stream):
            pass  # feed nothing initially → first drain is empty/sub-batch
        stream = _FakeStream(on_start=_on_start)
        sd._stream_holder["stream"] = stream

        def _on_wait(n):
            if n == 1:
                stream.feed(big)  # arrive during the first sub-batch wait
        evt = _ScriptedEvent([False, True], on_wait=_on_wait)
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title", return_value="X"):
            self.mod._worker_loop()
        self.assertEqual(len(self.mod._buffer), 1)

    def test_audio_sub_batch_waits_then_feeds(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("late sys audio", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        devs = [
            {"name": "Speakers (loopback)", "hostapi": 0,
             "max_input_channels": 1, "default_samplerate": 16000},
        ]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])
        big = np.ones(16000 * 6, dtype=np.float32) * 0.2
        stream = _FakeStream()
        sd._stream_holder["stream"] = stream

        def _on_wait(n):
            if n == 1:
                stream.feed(big)
        evt = _ScriptedEvent([False, True], on_wait=_on_wait)
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title", return_value="Y"), \
             mock.patch.object(self.mod, "_focused_proc_name", return_value="z.exe"):
            self.mod._audio_worker_loop()
        self.assertEqual(self.mod._audio_entries_total, 1)


# ─────────────────────────────────────────────────────────────────────────
# Deep callback / fallback / crash arms in the worker loops + finder.
# ─────────────────────────────────────────────────────────────────────────
class DeepWorkerArmTests(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._buffer.clear()
        self.mod._last_error = None
        self.mod._audio_last_error = None
        self.mod._audio_entries_total = 0

    # ── mic: get_input_device() raises → device falls back to None ──────
    def test_mic_get_input_device_raises(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("hi", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.get_input_device = mock.MagicMock(side_effect=RuntimeError("dev boom"))
        self.mod._wake_pattern = None
        sd = _make_sd()
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        stream = _FakeStream(on_start=lambda s: s.feed(block))
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title", return_value="X"):
            self.mod._worker_loop()
        self.assertEqual(len(self.mod._buffer), 1)
        # device passed to InputStream was the None fallback.
        self.assertIsNone(stream.kw.get("device"))

    # ── mic callback: bad indata triggers the inner except (no append) ──
    def test_mic_callback_bad_indata_swallowed(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        self.mod._wake_pattern = None
        sd = _make_sd()

        class _BadBlock:
            ndim = 1
            def copy(self):
                raise ValueError("cannot copy")
        stream = _FakeStream(on_start=lambda s: s.feed(_BadBlock()))
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt):
            self.mod._worker_loop()
        # Bad block never queued → nothing transcribed.
        bc.transcribe.assert_not_called()

    # ── mic callback: stop already set → callback returns immediately ───
    def test_mic_callback_returns_when_stopped(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        self.mod._wake_pattern = None
        sd = _make_sd()
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2

        # Set the stop event BEFORE start feeds → the callback's is_set() guard
        # fires (line 645) and the frame is dropped.
        evt = _ScriptedEvent([True])

        def _on_start(s):
            evt.set()
            s.feed(block)
        stream = _FakeStream(on_start=_on_start)
        sd._stream_holder["stream"] = stream
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt):
            self.mod._worker_loop()
        bc.transcribe.assert_not_called()

    # ── mic worker outer crash arm (np.concatenate blows up) ────────────
    def test_mic_worker_outer_crash(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        self.mod._wake_pattern = None
        sd = _make_sd()
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        stream = _FakeStream(on_start=lambda s: s.feed(block))
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt), \
             mock.patch.object(self.mod.np, "concatenate",
                               side_effect=RuntimeError("concat boom")):
            self.mod._worker_loop()
        self.assertIn("worker crashed", self.mod._last_error)

    # ── loopback finder: wrong-hostapi + no-input devices are skipped ───
    def test_finder_skips_wrong_hostapi_and_no_input(self):
        devs = [
            {"name": "MME thing", "hostapi": 5, "max_input_channels": 2},   # wrong host
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 0},    # no input
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 2},    # match by name
        ]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 1}])
        self.assertEqual(self.mod._find_loopback_device(sd), 2)

    # ── audio worker: query_devices(idx) raises → sr/channel fallback ───
    def test_audio_worker_devinfo_query_raises(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("sys", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        devs = [{"name": "Speakers (loopback)", "hostapi": 0,
                 "max_input_channels": 1, "default_samplerate": 16000}]
        # default_output_device=-1 so the finder matches the explicit
        # "(loopback)" entry WITHOUT calling query_devices(int); only the
        # worker's own query_devices(device_idx) call then raises → exercises
        # the dev_sr / in_channels fallback (lines 875-877).
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": -1}])
        real_qd = sd.query_devices

        def _qd(idx=None):
            if idx is not None:
                raise RuntimeError("devinfo boom")
            return real_qd(None)
        sd.query_devices = _qd
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        stream = _FakeStream(on_start=lambda s: s.feed(block))
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([True])
        # Regression (was a latent crash): when sd.query_devices(idx) raises, the
        # except at L875-877 sets dev_sr/in_channels and dev_info now defaults to
        # {} (source fix), so the info banner stays safe instead of dereferencing
        # an unbound dev_info and silently killing the audio daemon with an
        # UnboundLocalError. The except body is exercised on the way through; the
        # worker must run to completion rather than raise.
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title", return_value="Y"), \
             mock.patch.object(self.mod, "_focused_proc_name", return_value="z.exe"):
            self.mod._audio_worker_loop()   # previously raised UnboundLocalError

    # ── audio worker: WasapiSettings raises → extra_settings None ───────
    def test_audio_worker_wasapi_settings_raises(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("sys", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        devs = [{"name": "Speakers (loopback)", "hostapi": 0,
                 "max_input_channels": 1, "default_samplerate": 16000}]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])
        def _ws(**kw):
            raise RuntimeError("wasapi settings boom")
        sd.WasapiSettings = _ws
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        stream = _FakeStream(on_start=lambda s: s.feed(block))
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt), \
             mock.patch.object(self.mod, "_focused_window_title", return_value="Y"), \
             mock.patch.object(self.mod, "_focused_proc_name", return_value="z.exe"):
            self.mod._audio_worker_loop()
        self.assertEqual(self.mod._audio_entries_total, 1)
        # extra_settings forwarded as None when WasapiSettings is unavailable.
        self.assertIsNone(stream.kw.get("extra_settings"))

    # ── audio callback: stereo-mean exception swallowed ─────────────────
    def test_audio_callback_bad_indata_swallowed(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        devs = [{"name": "Speakers (loopback)", "hostapi": 0,
                 "max_input_channels": 1, "default_samplerate": 16000}]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])

        class _BadBlock:
            ndim = 1
            def astype(self, *a, **k):
                raise ValueError("astype boom")
        stream = _FakeStream(on_start=lambda s: s.feed(_BadBlock()))
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt):
            self.mod._audio_worker_loop()
        bc.transcribe.assert_not_called()

    # ── audio callback: stop already set → callback returns early (L893) ─
    def test_audio_callback_returns_when_stopped(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        devs = [{"name": "Speakers (loopback)", "hostapi": 0,
                 "max_input_channels": 1, "default_samplerate": 16000}]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        evt = _ScriptedEvent([True])

        def _on_start(s):
            evt.set()       # stop set BEFORE the frame arrives → guard fires
            s.feed(block)
        stream = _FakeStream(on_start=_on_start)
        sd._stream_holder["stream"] = stream
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt):
            self.mod._audio_worker_loop()
        bc.transcribe.assert_not_called()

    # ── audio worker: ambient-music + empty-text continues ──────────────
    def test_audio_worker_ambient_music_continue(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("la la", _good_conf()))
        bc.is_ambient_music = mock.MagicMock(return_value=True)
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        devs = [{"name": "Speakers (loopback)", "hostapi": 0,
                 "max_input_channels": 1, "default_samplerate": 16000}]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        stream = _FakeStream(on_start=lambda s: s.feed(block))
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt):
            self.mod._audio_worker_loop()
        self.assertEqual(self.mod._audio_entries_total, 0)

    def test_audio_worker_empty_text_continue(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("", _good_conf()))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        devs = [{"name": "Speakers (loopback)", "hostapi": 0,
                 "max_input_channels": 1, "default_samplerate": 16000}]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        stream = _FakeStream(on_start=lambda s: s.feed(block))
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt):
            self.mod._audio_worker_loop()
        self.assertEqual(self.mod._audio_entries_total, 0)

    # ── audio worker outer crash arm ────────────────────────────────────
    def test_audio_worker_outer_crash(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        devs = [{"name": "Speakers (loopback)", "hostapi": 0,
                 "max_input_channels": 1, "default_samplerate": 16000}]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        stream = _FakeStream(on_start=lambda s: s.feed(block))
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt), \
             mock.patch.object(self.mod.np, "concatenate",
                               side_effect=RuntimeError("concat boom")):
            self.mod._audio_worker_loop()
        self.assertIn("audio worker crashed", self.mod._audio_last_error)

    # ── _focused_proc_name outer exception → '' ─────────────────────────
    def test_focused_proc_name_exception(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only")
        user32 = mock.MagicMock()
        user32.GetForegroundWindow.side_effect = RuntimeError("boom")
        with mock.patch("ctypes.windll") as windll:
            windll.user32 = user32
            self.assertEqual(self.mod._focused_proc_name(), "")

    # ── mic pause loop: wait False → continue, then exit (L706) ─────────
    def test_mic_pause_continue_then_exit(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        sd = _make_sd()
        stream = _FakeStream()
        sd._stream_holder["stream"] = stream
        self.mod._paused[0] = True
        self.addCleanup(lambda: self.mod._paused.__setitem__(0, False))
        evt = _ScriptedEvent([False, True])  # pause-wait False (continue) then True
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_stop_evt", evt):
            self.mod._worker_loop()
        bc.transcribe.assert_not_called()
        self.assertGreaterEqual(evt.wait_calls, 2)

    # ── audio pause loop: wait False → continue, then exit (L951) ───────
    def test_audio_pause_continue_then_exit(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        devs = [{"name": "Speakers (loopback)", "hostapi": 0,
                 "max_input_channels": 1, "default_samplerate": 16000}]
        sd = _make_sd(devices=devs,
                      hostapis=[{"name": "Windows WASAPI", "default_output_device": 0}])
        stream = _FakeStream()
        sd._stream_holder["stream"] = stream
        self.mod._paused[0] = True
        self.addCleanup(lambda: self.mod._paused.__setitem__(0, False))
        evt = _ScriptedEvent([False, True])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_audio_stop_evt", evt):
            self.mod._audio_worker_loop()
        bc.transcribe.assert_not_called()
        self.assertGreaterEqual(evt.wait_calls, 2)

    # NOTE: the `if n_out <= 0: continue` guard at L981 is effectively
    # unreachable — to reach the resample block the batch must already hold
    # >= dev_sr*batch_secs samples, so n_out ≈ sample_rate*batch_secs >= 5 > 0
    # for any valid sample_rate. Left uncovered by design (no test can drive a
    # zero-length resample without bypassing the batch gate).

    # ── _persist_state double-fault: replace + unlink both raise (L344) ─
    def test_persist_state_unlink_also_raises(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None), \
             mock.patch.object(self.mod.os, "replace",
                               side_effect=RuntimeError("replace boom")), \
             mock.patch.object(self.mod.os, "unlink",
                               side_effect=RuntimeError("unlink boom")):
            self.mod._persist_state()  # both swallowed, no raise

    # ── _rotate_jsonl double-fault: replace + unlink both raise (L425) ──
    def test_rotate_jsonl_unlink_also_raises(self):
        path = os.path.join(self.tmp, "big.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(40):
                f.write(json.dumps({"i": i}) + "\n")
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=RuntimeError("replace boom")), \
             mock.patch.object(self.mod.os, "unlink",
                               side_effect=RuntimeError("unlink boom")):
            self.mod._rotate_jsonl_if_needed(path, cap=10)  # both swallowed

    # ── H-3: the hung-close path is inert even when sounddevice is a
    # landmine — nothing on the module-global API may be touched ────────
    def test_safe_close_hung_path_touches_no_global_sd_api(self):
        stream = mock.MagicMock()
        sd = _make_sd()
        sd.stop = mock.MagicMock(side_effect=RuntimeError("sd.stop boom"))
        sd.wait = mock.MagicMock(side_effect=RuntimeError("sd.wait boom"))
        fake_evt = mock.MagicMock()
        fake_evt.wait.return_value = False
        with mock.patch.object(self.mod, "_get_bobert", return_value=None), \
             mock.patch.object(self.mod.threading, "Event", return_value=fake_evt), \
             mock.patch.object(self.mod.threading, "Thread"), \
             inject_modules(sounddevice=sd):
            self.mod._safe_close_stream(stream)  # must not raise
        sd.stop.assert_not_called()
        sd.wait.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# Known-device speech (core/device_speech_filter.py, R3 2026-09-29)
# ─────────────────────────────────────────────────────────────────────────
class DeviceSpeechTests(_TmpDirMixin, unittest.TestCase):
    """A line a known device speaks can never wake JARVIS through the ambient
    nudge and is never buffered for the learners. GENERIC fixture only (a
    made-up "desk speaker"); the phrase dir is redirected via JARVIS_DATA_DIR."""

    _LINE = "Jarvis, desk speaker ready to play."

    def setUp(self):
        import shutil
        from core import device_speech_filter as dsf
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._buffer.clear()
        self.mod._last_error = None
        dsf._reset_cache_for_tests()
        self.addCleanup(dsf._reset_cache_for_tests)
        self.pdir = tempfile.mkdtemp(prefix="ambient_dsf_")
        self.addCleanup(shutil.rmtree, self.pdir, True)
        os.makedirs(os.path.join(self.pdir, "device_phrases"))
        with open(os.path.join(self.pdir, "device_phrases", "desk.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"source": "desk speaker",
                       "phrases": [self._LINE, "Jarvis"]}, f)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.pdir})
        env.start()
        self.addCleanup(env.stop)

    def _bc(self, **extra):
        bc = mock.MagicMock()
        bc._sleep_mode = [False]
        bc._standby_mode = [True]
        bc._tts_playback_active = [False]
        bc.WAKE_PHRASES = {"jarvis", "hey jarvis"}
        bc.DEVICE_SPEECH_FILTER_ENABLED = True
        for k, v in extra.items():
            setattr(bc, k, v)
        return bc

    def _nudge(self, bc, text):
        import re
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        self.mod._tts_last_active[0] = 0.0
        buf = io.StringIO()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             contextlib.redirect_stdout(buf):
            self.mod._maybe_nudge_wake(text)
        return buf.getvalue()

    def test_device_line_never_nudges_a_wake(self):
        bc = self._bc()
        log = self._nudge(bc, self._LINE)
        bc.proactive_announce.assert_not_called()
        self.assertIn("device speech ignored (desk speaker)", log)
        self.assertNotIn("ready to play", log.lower())

    def test_misheard_device_line_never_nudges(self):
        bc = self._bc()
        self._nudge(bc, "jarvis the speaker ready to play")
        bc.proactive_announce.assert_not_called()

    def test_owner_wake_still_nudges(self):
        bc = self._bc()
        self._nudge(bc, "hey jarvis you there")
        bc.proactive_announce.assert_called_once()

    def test_bare_wake_word_still_nudges_even_if_a_device_says_it(self):
        bc = self._bc()
        self._nudge(bc, "Jarvis.")
        bc.proactive_announce.assert_called_once()

    def test_stop_word_line_still_nudges(self):
        bc = self._bc()
        self._nudge(bc, "stop jarvis desk speaker ready to play")
        bc.proactive_announce.assert_called_once()

    def test_disabled_flag_restores_the_old_nudge(self):
        bc = self._bc(DEVICE_SPEECH_FILTER_ENABLED=False)
        self._nudge(bc, self._LINE)
        bc.proactive_announce.assert_called_once()

    def _source(self, bc, fn, *a, **k):
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)

    def _write_phrases(self, phrases):
        from core import device_speech_filter as dsf
        with open(os.path.join(self.pdir, "device_phrases", "desk.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"source": "desk speaker", "phrases": phrases}, f)
        dsf._reset_cache_for_tests()

    def test_owner_vocabulary_is_never_a_device_line(self):
        # A device list holding generic owner words ("yes", a sleep phrase)
        # never swallows the owner's own reply, exactly like the monolith.
        self._write_phrases([self._LINE, "Yes.", "Next track", "Go to sleep."])
        bc = self._bc(SLEEP_PHRASES={"go to sleep"})
        for text in ("Yes.", "next track", "Go to sleep."):
            self.assertIsNone(
                self._source(bc, self.mod._device_speech_source, text), text)
        self.assertEqual(
            self._source(bc, self.mod._device_speech_source, self._LINE),
            "desk speaker")

    def test_line_split_across_two_batches_is_caught_on_the_join(self):
        # The mic worker cuts fixed 2.5 s batches, so a device line can
        # straddle a boundary: neither fragment matches alone.
        self.mod._prev_mic_batch[:] = [None, 0.0]
        self.addCleanup(self.mod._prev_mic_batch.__setitem__,
                        slice(None), [None, 0.0])
        bc = self._bc()
        src = self.mod._device_speech_batch_source
        self.assertIsNone(self._source(bc, src, "Jarvis, desk", now=100.0))
        self.assertEqual(
            self._source(bc, src, "speaker ready to play.", now=102.6),
            "desk speaker")
        # Batches far apart (or split by silence) are never joined.
        self.assertIsNone(self._source(bc, src, "Jarvis, desk", now=200.0))
        self.assertIsNone(
            self._source(bc, src, "speaker ready to play.", now=220.0))
        # An owner sentence after a whole device line is not swallowed.
        self.assertEqual(self._source(bc, src, self._LINE, now=300.0),
                         "desk speaker")
        self.assertIsNone(
            self._source(bc, src, "what time is it", now=302.5))

    def test_worker_silence_clears_the_previous_batch(self):
        bc = _FakeBobert(WAKE_PHRASES={"jarvis"},
                         DEVICE_SPEECH_FILTER_ENABLED=True)
        bc.transcribe = mock.MagicMock(return_value=("x", _good_conf()))
        self.mod._prev_mic_batch[:] = ["Jarvis, desk", time.time()]
        self.addCleanup(self.mod._prev_mic_batch.__setitem__,
                        slice(None), [None, 0.0])
        block = np.zeros(16000 * 3, dtype=np.float32)
        MicWorkerLoopTests._run_worker(self, bc, feed_block=block,
                                       wait_returns=[True])
        self.assertIsNone(self.mod._prev_mic_batch[0])

    def test_worker_never_buffers_or_nudges_a_device_line(self):
        import re
        bc = _FakeBobert(WAKE_PHRASES={"jarvis"},
                         DEVICE_SPEECH_FILTER_ENABLED=True)
        bc._sleep_mode = [False]
        bc._standby_mode = [True]
        bc.proactive_announce = mock.MagicMock()
        bc.transcribe = mock.MagicMock(return_value=(self._LINE, _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        self.mod._tts_last_active[0] = 0.0
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            MicWorkerLoopTests._run_worker(self, bc, feed_block=block,
                                           wait_returns=[True])
        self.assertEqual(len(self.mod._buffer), 0)
        self.assertFalse(os.path.exists(self.mod._AUDIO_JSONL))
        bc.proactive_announce.assert_not_called()
        self.assertIn("device speech ignored (desk speaker)", buf.getvalue())

class DialogueHoldTests(_TmpDirMixin, unittest.TestCase):
    """While a device dialogue runs (core/device_speech_filter
    dialogue_active), both ambient workers drop their batches BEFORE
    transcription: nothing is buffered, learned or a wake nudge."""

    def setUp(self):
        from core import device_speech_filter as dsf
        self.dsf = dsf
        dsf._reset_cache_for_tests()
        self.addCleanup(dsf._reset_cache_for_tests)
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._buffer.clear()
        self.mod._last_error = None
        self.mod._audio_entries_total = 0
        self.mod._audio_last_error = None

    def _bc(self):
        bc = _FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=("hello world",
                                                     _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.is_ambient_music = mock.MagicMock(return_value=False)
        bc.AMBIENT_AUDIO_CHUNK_DURATION_SECONDS = 5.0
        return bc

    def test_helper_follows_the_dialogue_bracket(self):
        self.assertFalse(self.mod._dialogue_holds_batches())
        tok = self.dsf.begin_dialogue("desk device")
        self.assertTrue(self.mod._dialogue_holds_batches())
        self.dsf.end_dialogue(tok, tail_s=0.0)
        self.assertFalse(self.mod._dialogue_holds_batches())

    def test_mic_batch_dropped_during_dialogue(self):
        self.dsf.begin_dialogue("desk device")
        bc = self._bc()
        self.mod._wake_pattern = None
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        MicWorkerLoopTests._run_worker(self, bc, feed_block=block,
                                       wait_returns=[True])
        bc.transcribe.assert_not_called()
        self.assertEqual(len(self.mod._buffer), 0)

    def test_mic_batch_kept_without_dialogue(self):
        bc = self._bc()
        self.mod._wake_pattern = None
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        MicWorkerLoopTests._run_worker(self, bc, feed_block=block,
                                       wait_returns=[True])
        bc.transcribe.assert_called_once()
        self.assertEqual(len(self.mod._buffer), 1)

    def test_loopback_batch_dropped_during_dialogue(self):
        self.dsf.begin_dialogue("desk device")
        bc = self._bc()
        sd = AudioWorkerLoopTests._loopback_sd(self, native_sr=16000)
        block = np.ones(16000 * 6, dtype=np.float32) * 0.2
        AudioWorkerLoopTests._run(self, bc, sd=sd, feed_block=block,
                                  wait_returns=[True])
        bc.transcribe.assert_not_called()
        self.assertEqual(self.mod._audio_entries_total, 0)


# ─────────────────────────────────────────────────────────────────────────
# AMBIENT_STT_YIELD (speed plan R2): ambient Whisper defers to the owner's
# turn instead of competing with it for the shared STT lock.
# ─────────────────────────────────────────────────────────────────────────
class SttYieldTests(_TmpDirMixin, unittest.TestCase):
    """With the flag on, a batch that is ready while the owner is mid-
    utterance (_utterance_in_progress) is parked instead of transcribed;
    parked batches are transcribed oldest-first once the flag clears. Each
    test drives a worker synchronously over scripted steps: batch number n is
    a constant block of amplitude n/100, transcribed as "line n"."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._buffer.clear()
        self.mod._last_error = None
        self.mod._audio_entries_total = 0
        self.mod._audio_last_error = None
        self.mod._wake_pattern = None
        self.mod._prev_mic_batch[:] = [None, 0.0]
        self.addCleanup(self.mod._prev_mic_batch.__setitem__,
                        slice(None), [None, 0.0])
        self.clock = [0.0]       # the yield's monotonic clock
        self.calls_at = []       # transcribe() calls so far, per step

    def _bc(self, flag=True):
        bc = _FakeBobert(_utterance_in_progress=[False],
                         _audio_master_enabled=[False],   # raw audio in
                         AMBIENT_AUDIO_CHUNK_DURATION_SECONDS=5.0)
        if flag is not None:
            bc.AMBIENT_STT_YIELD = flag
        bc.transcribe = mock.MagicMock(
            side_effect=lambda a: (f"line {self._n(a)}", _good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.is_ambient_music = mock.MagicMock(return_value=False)
        return bc

    @staticmethod
    def _n(audio):
        return int(round(float(audio[0]) * 100))

    def _heard(self, bc):
        """Batch numbers transcribe() was given, in call order."""
        return [self._n(c.args[0]) for c in bc.transcribe.call_args_list]

    def _drive(self, bc, steps, *, loopback=False, extra=()):
        """Run the mic (or loopback) worker once over ``steps``. Step k is
        (busy, n, t): the owner flag and the yield clock are set, then batch
        n is captured (None = a tick with no new audio). Returns the log."""
        samples = 16000 * (6 if loopback else 3)

        def _apply(k):
            if k >= len(steps):
                return
            busy, n, t = steps[k]
            self.calls_at.append(bc.transcribe.call_count)
            bc._utterance_in_progress[0] = busy
            self.clock[0] = t
            if n is not None:
                stream.feed(np.full(samples, n / 100.0, dtype=np.float32))

        stream = _FakeStream(on_start=lambda s: _apply(0))
        sd = (AudioWorkerLoopTests._loopback_sd(self, native_sr=16000)
              if loopback else _make_sd())
        sd._stream_holder["stream"] = stream
        evt = _ScriptedEvent([False] * (len(steps) - 1) + [True],
                             on_wait=_apply)
        log = io.StringIO()
        with contextlib.ExitStack() as st:
            st.enter_context(inject_modules(sounddevice=sd))
            st.enter_context(mock.patch.object(self.mod, "_get_bobert",
                                               return_value=bc))
            st.enter_context(mock.patch.object(
                self.mod, "_audio_stop_evt" if loopback else "_stop_evt", evt))
            st.enter_context(mock.patch.object(
                self.mod, "_yield_mono", side_effect=lambda: self.clock[0],
                create=True))
            # The focused window names the clock, so a parked batch shows
            # whether it kept its capture-time attribution.
            st.enter_context(mock.patch.object(
                self.mod, "_focused_window_title",
                side_effect=lambda: f"win@{self.clock[0]:g}"))
            st.enter_context(mock.patch.object(
                self.mod, "_focused_proc_name", return_value="player.exe"))
            for cm in extra:
                st.enter_context(cm)
            st.enter_context(contextlib.redirect_stdout(log))
            (self.mod._audio_worker_loop if loopback
             else self.mod._worker_loop)()
        return log.getvalue()

    def _jsonl(self):
        if not os.path.exists(self.mod._AUDIO_JSONL):
            return []
        with open(self.mod._AUDIO_JSONL, encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]

    # ── parking ──────────────────────────────────────────────────────────
    def test_mic_batch_is_not_transcribed_while_the_owner_speaks(self):
        bc = self._bc()
        log = self._drive(bc, [(True, 11, 0.0)])
        bc.transcribe.assert_not_called()
        self.assertEqual(len(self.mod._buffer), 0)
        self.assertEqual(self._jsonl(), [])
        # One count line (counts only, never a transcript).
        self.assertEqual(log.count("yielded to the owner's turn"), 1)
        self.assertIn("1 batch(es) deferred, 0 dropped", log)
        self.assertIsNone(self.mod._last_error)

    def test_loopback_batch_is_not_transcribed_while_the_owner_speaks(self):
        bc = self._bc()
        self._drive(bc, [(True, 11, 0.0)], loopback=True)
        bc.transcribe.assert_not_called()
        self.assertEqual(self.mod._audio_entries_total, 0)
        self.assertEqual(self._jsonl(), [])

    # ── release: order, capture time, attribution ───────────────────────
    def test_mic_parked_batches_go_first_in_order_after_release(self):
        bc = self._bc()
        spy = mock.patch.object(self.mod, "_device_speech_batch_source",
                                wraps=self.mod._device_speech_batch_source)
        with spy as dev:
            self._drive(bc, [(True, 11, 0.0), (True, 12, 3.0),
                             (False, 13, 6.0)])
        self.assertEqual(self.calls_at, [0, 0, 0])   # nothing while busy
        self.assertEqual(self._heard(bc), [11, 12, 13])
        texts = [e["text"] for e in self.mod._buffer]
        self.assertEqual(texts, ["line 11", "line 12", "line 13"])
        # Parked batches join split lines (and are stamped) by capture time;
        # the live batch makes exactly the call it always made.
        c11, c12, c13 = dev.call_args_list
        self.assertEqual(c11.args, ("line 11",))
        self.assertEqual(c11.kwargs["now"], self.mod._buffer[0]["ts"])
        self.assertEqual(c12.kwargs["now"], self.mod._buffer[1]["ts"])
        self.assertLessEqual(self.mod._buffer[0]["ts"],
                             self.mod._buffer[1]["ts"])
        self.assertEqual(c13, mock.call("line 13"))
        self.assertEqual([m["window"] for m in self._jsonl()],
                         ["win@0", "win@3", "win@6"])

    def test_mic_parked_batch_is_kept_on_an_idle_tick(self):
        # No new audio after the owner's turn: the parked batch is still
        # transcribed and kept, it does not wait for the next loud batch.
        bc = self._bc()
        self._drive(bc, [(True, 11, 0.0), (False, None, 4.0)])
        self.assertEqual(self.calls_at, [0, 0])   # parked, then released
        self.assertEqual(self._heard(bc), [11])
        self.assertEqual([e["text"] for e in self.mod._buffer], ["line 11"])
        self.assertEqual(self._jsonl()[0]["source"], "mic")

    def test_loopback_parked_batch_logged_in_order_after_release(self):
        bc = self._bc()
        self._drive(bc, [(True, 11, 0.0), (False, 12, 7.0)], loopback=True)
        self.assertEqual(self.calls_at, [0, 0])
        self.assertEqual(self._heard(bc), [11, 12])
        rows = self._jsonl()
        self.assertEqual([r["text"] for r in rows], ["line 11", "line 12"])
        self.assertEqual([r["window"] for r in rows], ["win@0", "win@7"])
        self.assertTrue(all(r["source"] == "system_audio" for r in rows))
        self.assertEqual(self.mod._audio_entries_total, 2)

    # ── bounds ───────────────────────────────────────────────────────────
    def test_parked_batches_are_bounded_and_overflow_is_counted(self):
        bc = self._bc()
        bc._tt_note_stat = mock.MagicMock()
        steps = [(True, n, 0.0) for n in range(11, 19)] + [(False, None, 1.0)]
        with mock.patch.object(self.mod, "_AMBIENT_YIELD_LOG_EVERY", 1):
            log = self._drive(bc, steps)
        self.assertEqual(self.mod.AMBIENT_YIELD_MAX_BATCHES, 6)
        # 8 parked into 6 slots: the two oldest were dropped.
        self.assertEqual(self._heard(bc), [13, 14, 15, 16, 17, 18])
        self.assertIn("8 batch(es) deferred, 2 dropped", log)
        self.assertEqual(log.count("yielded to the owner's turn"), 8)
        self.assertNotIn("line 1", log)   # counts only, never transcripts
        self.assertEqual(bc._tt_note_stat.call_args_list,
                         [mock.call("amb_deferred", 1)] * 8)

    def test_default_logs_one_count_line_per_ten_deferrals(self):
        bc = self._bc()
        log = self._drive(bc, [(True, n, 0.0) for n in range(11, 23)])
        self.assertEqual(log.count("yielded to the owner's turn"), 2)
        self.assertIn("11 batch(es) deferred, 5 dropped", log)

    def test_parked_batch_never_nudges_a_wake(self):
        import re
        bc = self._bc()
        bc._sleep_mode = [False]
        bc._standby_mode = [True]          # the one state that nudges
        bc.proactive_announce = mock.MagicMock()
        bc.transcribe.side_effect = (
            lambda a: (f"hey jarvis, line {self._n(a)}", _good_conf()))
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        self.mod._tts_last_active[0] = 0.0
        self._drive(bc, [(True, 11, 0.0), (False, None, 4.0)])
        self.assertEqual([e["text"] for e in self.mod._buffer],
                         ["hey jarvis, line 11"])
        bc.proactive_announce.assert_not_called()
        # The same words in a live batch still nudge.
        self._drive(bc, [(False, 12, 5.0)])
        bc.proactive_announce.assert_called_once()

    def test_stale_owner_flag_never_silences_ambient(self):
        bc = self._bc()
        cap = self.mod.AMBIENT_YIELD_MAX_DEFER_S
        self.assertEqual(cap, 15)
        log = self._drive(bc, [(True, 11, 0.0), (True, 12, 10.0),
                               (True, 13, cap + 1.0), (True, 14, cap + 4.0)])
        # Parked while the flag was fresh; past the cap everything is
        # transcribed, oldest first, although the flag never cleared.
        self.assertEqual(self.calls_at, [0, 0, 0, 3])
        self.assertEqual(self._heard(bc), [11, 12, 13, 14])
        self.assertEqual(log.count("transcribing anyway"), 1)

    # ── flag off ─────────────────────────────────────────────────────────
    def _call_sequence(self, flag):
        """Every host + helper call one flag-off run makes, in order, with
        arrays reduced to their batch number and dicts to their text."""
        bc = self._bc(flag=flag)
        bc._tt_note_stat = mock.MagicMock()
        rec = mock.MagicMock()
        for name in ("transcribe", "is_ambient_music", "is_valid_speech"):
            rec.attach_mock(getattr(bc, name), name)
        spies = []
        for name in ("_dialogue_holds_batches", "_apply_audio_processing",
                     "_device_speech_batch_source", "_identify_speaker_safe",
                     "_persist_state", "_append_jsonl", "_maybe_nudge_wake"):
            p = mock.patch.object(self.mod, name,
                                  wraps=getattr(self.mod, name))
            spies.append(p)
        # Flag off builds no yield and never reads its clock (create=True
        # lets this same test run against the pre-flag code).
        never = (mock.patch.object(self.mod, "_SttYield", create=True,
                                   side_effect=AssertionError("built")),
                 mock.patch.object(self.mod, "_yield_mono", create=True,
                                   side_effect=AssertionError("clock")))

        def _norm(v):
            if isinstance(v, np.ndarray):
                return ("batch", self._n(v))
            if isinstance(v, dict):
                return ("row", v.get("text"))
            return v

        with contextlib.ExitStack() as st:
            mocks = [st.enter_context(p) for p in spies]
            for p, m in zip(spies, mocks):
                rec.attach_mock(m, p.attribute)
            # The owner is speaking throughout.
            self._drive(bc, [(True, 11, 0.0), (True, 12, 30.0)], extra=never)
        self.assertIsNone(self.mod._last_error)
        bc._tt_note_stat.assert_not_called()
        seq = [(c[0], tuple(_norm(a) for a in c[1]),
                tuple(sorted((k, _norm(v)) for k, v in c[2].items())))
               for c in rec.mock_calls]
        return seq, [e["text"] for e in self.mod._buffer]

    def test_flag_off_makes_exactly_todays_calls(self):
        absent, kept_absent = self._call_sequence(flag=None)
        self.mod._buffer.clear()
        os.unlink(self.mod._AUDIO_JSONL)
        off, kept_off = self._call_sequence(flag=False)
        self.assertEqual(off, absent)
        self.assertEqual(kept_off, kept_absent)
        self.assertEqual(kept_off, ["line 11", "line 12"])
        per_batch = ["_dialogue_holds_batches", "_apply_audio_processing",
                     "transcribe", "_device_speech_batch_source",
                     "is_ambient_music", "is_valid_speech",
                     "_identify_speaker_safe", "_persist_state",
                     "_append_jsonl", "_maybe_nudge_wake"]
        self.assertEqual([c[0] for c in off], per_batch * 2)
        self.assertEqual(off[3], ("_device_speech_batch_source",
                                  ("line 11",), ()))


if __name__ == "__main__":
    unittest.main()
