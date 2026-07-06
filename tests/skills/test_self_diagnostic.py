"""Logic tests for skills/self_diagnostic.py.

self_diagnostic probes every JARVIS subsystem and auto-queues repair tasks.
The probes themselves touch hardware/network, so tests target the
deterministic, mockable LOGIC around them:
  • the canonical _result shape,
  • the PnP camera diagnosis state-machine (absent / problem / ok / unknown),
  • the recent-problem voice-mood flag window,
  • _suggested_files_for / _suggested_files_for_action source-file hints,
  • _traceback_excerpt tail extraction,
  • _last_successful_ts history walk,
  • _collect_action_error_groups grouping + threshold,
  • _summarise / diagnostic_status / diagnostic_history / whats_broken
    rendering (history + todo paths redirected to a temp dir).

A fake bobert_companion is injected where helpers consult it. The scheduler /
boot-sweep thread in register() is neutered by the harness. No probe ever runs
against real hardware.
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

from tests._skill_harness import load_skill_isolated


# ─── fake-module helpers ─────────────────────────────────────────────────
_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules (e.g. cv2, torch,
    sounddevice). For dotted names (``core.scheduler``, ``paho.mqtt.client``)
    the leaf is ALSO set as an attribute on its already-imported parent
    package, because ``from core import scheduler`` / ``import a.b.c`` resolve
    the leaf via ``getattr(parent, leaf)`` when the parent is a real package.
    Restores the previous state — including absence — on exit so probes that
    do deferred imports see exactly the fake we provide and tests stay
    isolated. Keys may be passed via ``**{"a.b": obj}``.
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
    """Force ``import <name>`` to raise ImportError inside the with-block, so
    a probe's missing-dependency branch is exercised even when the real dep
    is installed on the dev box.

    For an ALREADY-IMPORTED dotted target (e.g. ``core.scheduler`` after some
    other test imported the real module) patching ``__import__`` is not enough:
    ``from core import scheduler`` is satisfied by the import system straight
    from ``sys.modules`` / as an attribute of the already-loaded ``core``
    package via ``getattr`` — the blocked ``__import__`` is never consulted, so
    the block silently fails and the probe sees the real (possibly leaked-and-
    bootstrapped) module. So we ALSO detach each blocked name from
    ``sys.modules`` and remove the leaf attribute from its parent package for
    the duration of the block, then restore both — forcing the import to
    re-run and hit the raising ``__import__``. This makes the block robust to
    cross-test pollution (a real ``core.scheduler`` left in ``sys.modules``)."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in blocked or top in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    # Detach already-imported blocked modules (and their parent-package attrs)
    # so the import machinery can't satisfy the import from cache.
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


def _fake_np():
    """A numpy stand-in covering only what the mic/STT probes use:
    sqrt/mean/square/sin/linspace/pi/array squeeze + float32."""
    np = types.ModuleType("numpy")
    np.float32 = "float32"
    np.pi = 3.141592653589793

    class _Arr(list):
        """A tiny ndarray stand-in: supports squeeze/astype and the scalar
        arithmetic (0.1 * arr, arr * 2pi, ...) the STT sine-gen performs."""
        def squeeze(self):
            return self

        def astype(self, _dtype):
            return self

        def _binop(self, other):
            if isinstance(other, list):
                return _Arr([a + 0 for a in self])
            return _Arr([v * 0 for v in self])

        __mul__ = _binop
        __rmul__ = _binop
        __add__ = _binop
        __radd__ = _binop

    def _wrap(seq):
        return _Arr(seq)

    np.sqrt = lambda x: (x ** 0.5) if not isinstance(x, list) else _wrap([v ** 0.5 for v in x])
    np.mean = lambda a: (sum(a) / len(a)) if len(a) else 0.0
    np.square = lambda a: _wrap([v * v for v in a]) if isinstance(a, list) else a * a
    np.sin = lambda a: _wrap([0.0 for _ in a]) if isinstance(a, list) else 0.0
    np.linspace = lambda lo, hi, n, dtype=None: _wrap([0.0] * n)
    np.array = lambda seq, dtype=None: _wrap(list(seq))
    return np


class SelfDiagPureTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("self_diagnostic")

    # ── _result ──────────────────────────────────────────────────────────
    def test_result_ok_clears_error(self):
        r = self.mod._result(True, 12.34)
        self.assertTrue(r["ok"])
        self.assertEqual(r["latency_ms"], 12.3)   # rounded to 1 dp
        self.assertIsNone(r["error"])

    def test_result_failure_defaults_error(self):
        r = self.mod._result(False, 5.0)
        self.assertEqual(r["error"], "unknown error")
        self.assertFalse(r["ok"])

    # ── _camera_pnp_diagnosis ────────────────────────────────────────────
    def test_pnp_diagnosis_none_is_unknown(self):
        d = self.mod._camera_pnp_diagnosis(None)
        self.assertEqual(d["failure_mode"], "unknown")
        self.assertFalse(d["hardware_present"])

    def test_pnp_diagnosis_absent(self):
        d = self.mod._camera_pnp_diagnosis([])
        self.assertEqual(d["failure_mode"], "absent")
        self.assertFalse(d["hardware_present"])

    def test_pnp_diagnosis_absent_when_not_present(self):
        devices = [{"name": "Cam", "status": "OK", "problem": 0, "present": False}]
        d = self.mod._camera_pnp_diagnosis(devices)
        self.assertEqual(d["failure_mode"], "absent")

    def test_pnp_diagnosis_problem_device(self):
        devices = [{"name": "Logi Cam", "status": "Error", "problem": 43,
                    "present": True}]
        d = self.mod._camera_pnp_diagnosis(devices)
        self.assertEqual(d["failure_mode"], "problem")
        self.assertTrue(d["has_problem_device"])
        self.assertIn("Logi Cam", d["summary"])
        self.assertIn("43", d["summary"])

    def test_pnp_diagnosis_healthy(self):
        devices = [{"name": "Cam", "status": "OK", "problem": 0, "present": True}]
        d = self.mod._camera_pnp_diagnosis(devices)
        self.assertEqual(d["failure_mode"], "ok")
        self.assertEqual(d["healthy_devices"], 1)

    def test_pnp_diagnosis_healthy_wins_over_problem_when_mixed(self):
        devices = [
            {"name": "Good", "status": "OK", "problem": 0, "present": True},
            {"name": "Bad", "status": "Error", "problem": 22, "present": True},
        ]
        d = self.mod._camera_pnp_diagnosis(devices)
        self.assertEqual(d["failure_mode"], "ok")     # at least one healthy
        self.assertTrue(d["has_problem_device"])
        self.assertEqual(d["healthy_devices"], 1)

    # ── recent-problem flag ──────────────────────────────────────────────
    def test_recent_problem_flag_window(self):
        self.mod._recent_problem_at[0] = 0.0
        self.assertFalse(self.mod.get_recent_problem_flag(now=1000.0))
        self.mod._mark_recent_problem(now=1000.0)
        self.assertTrue(self.mod.get_recent_problem_flag(now=1000.0 + 60))
        # Beyond the window → flag clears.
        self.assertFalse(self.mod.get_recent_problem_flag(
            now=1000.0 + self.mod._RECENT_PROBLEM_WINDOW_SEC + 1))

    # ── _suggested_files_for ─────────────────────────────────────────────
    def test_suggested_files_known_and_unknown(self):
        self.assertIn("face_tracker", self.mod._suggested_files_for("webcam"))
        self.assertIn("bambu_monitor", self.mod._suggested_files_for("bambu"))
        self.assertEqual(self.mod._suggested_files_for("nonexistent"), "(no suggestion)")

    # ── _suggested_files_for_action ──────────────────────────────────────
    def test_suggested_files_for_action_maps_skill_module(self):
        bc = mock.MagicMock()
        fn = mock.MagicMock()
        fn.__module__ = "skill_timer"
        bc.ACTIONS = {"set_timer": fn}
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            out = self.mod._suggested_files_for_action("set_timer")
        self.assertEqual(out, "skills/timer.py")

    def test_suggested_files_for_action_core_module(self):
        bc = mock.MagicMock()
        fn = mock.MagicMock()
        fn.__module__ = "core.tts"
        bc.ACTIONS = {"speak": fn}
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._suggested_files_for_action("speak"), "core/tts.py")

    def test_suggested_files_for_action_no_bc(self):
        with mock.patch.object(self.mod, "_bc", return_value=None):
            self.assertIn("bobert_companion.py",
                          self.mod._suggested_files_for_action("anything"))

    # ── _traceback_excerpt ───────────────────────────────────────────────
    def test_traceback_excerpt_keeps_tail(self):
        tb = "line1\n\nline2\nline3\nline4\nline5\nline6"
        out = self.mod._traceback_excerpt(tb, max_lines=3)
        self.assertEqual(out.splitlines(), ["line4", "line5", "line6"])

    def test_traceback_excerpt_empty(self):
        self.assertEqual(self.mod._traceback_excerpt(""), "")

    # ── _last_successful_ts ──────────────────────────────────────────────
    def test_last_successful_ts_finds_most_recent_ok(self):
        history = [
            {"iso": "2026-05-01T00:00:00", "probes": {"webcam": {"ok": True}}},
            {"iso": "2026-05-02T00:00:00", "probes": {"webcam": {"ok": False}}},
        ]
        # Walks backward; the most recent OK is the 05-01 run.
        self.assertEqual(self.mod._last_successful_ts(history, "webcam"),
                         "2026-05-01T00:00:00")

    def test_last_successful_ts_none_when_never_ok(self):
        history = [{"iso": "2026-05-02T00:00:00", "probes": {"stt": {"ok": False}}}]
        self.assertIsNone(self.mod._last_successful_ts(history, "stt"))

    # ── _collect_action_error_groups ─────────────────────────────────────
    def test_collect_action_errors_groups_and_thresholds(self):
        # 3 same-class errors for one action (>= group count 3) and 1 for
        # another (below threshold).
        errors = (
            [{"action": "play_music", "exc_class": "KeyError", "exc_msg": "no key",
              "traceback": "tb", "ts": float(i)} for i in range(3)]
            + [{"action": "see_screen", "exc_class": "TimeoutError",
                "exc_msg": "slow", "traceback": "tb2", "ts": 9.0}]
        )
        bc = mock.MagicMock()
        bc.get_recent_action_errors.return_value = errors
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            groups = self.mod._collect_action_error_groups()
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g["action"], "play_music")
        self.assertEqual(g["exc_class"], "KeyError")
        self.assertEqual(g["count"], 3)

    def test_collect_action_errors_empty_when_no_getter(self):
        bc = mock.MagicMock(spec=[])   # no get_recent_action_errors attr
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._collect_action_error_groups(), [])


class SelfDiagSummaryTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("self_diagnostic")

    def _run(self, failed, sev=None):
        sev = sev or {}
        return {
            "ts": 0.0, "iso": "2026-05-30T00:00:00", "duration_ms": 1234.0,
            "probes": {c: {"severity": sev.get(c, self.mod.SEVERITY_MED),
                           "error": f"{c} down"} for c in failed},
            "failed": failed,
            "severity_failed": {c: sev.get(c, self.mod.SEVERITY_MED) for c in failed},
        }

    def test_summarise_all_nominal(self):
        run = self._run([])
        out = self.mod._summarise(run, [])
        self.assertIn("All systems nominal", out)

    def test_summarise_single_issue(self):
        run = self._run(["microphone"], {"microphone": self.mod.SEVERITY_HIGH})
        out = self.mod._summarise(run, ["microphone"])
        self.assertIn("one issue — microphone", out)
        self.assertIn("1 high", out.lower())
        self.assertIn("1 repair task", out)

    def test_summarise_many_issues_truncates(self):
        comps = ["webcam", "stt", "tts", "gpu", "ram", "disk"]
        run = self._run(comps)
        out = self.mod._summarise(run, [])
        self.assertIn("6 issues", out)
        self.assertIn("and 3 more", out)   # lists first 3 + "and N more"

    def test_diagnostic_status_no_run(self):
        self.mod._state["last_run"] = None
        self.assertIn("No diagnostic has run yet", self.actions["diagnostic_status"](""))

    def test_diagnostic_status_reports_failures(self):
        run = self._run(["microphone"])
        run["ts"] = self.mod._now() - 30
        self.mod._state["last_run"] = run
        out = self.actions["diagnostic_status"]("")
        self.assertIn("microphone", out)
        self.assertIn("seconds ago", out)

    def test_diagnostic_status_all_nominal_age(self):
        run = self._run([])
        run["ts"] = self.mod._now() - 5
        self.mod._state["last_run"] = run
        out = self.actions["diagnostic_status"]("")
        self.assertIn("All systems nominal", out)


class SelfDiagFileTests(unittest.TestCase):
    """Tests touching jarvis_todo.md / history — redirect those paths to a
    temp dir so the real project files aren't read or written."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("self_diagnostic")
        self.tmp = tempfile.mkdtemp(prefix="selfdiag_test_")
        self.addCleanup(self._cleanup)
        self.todo = os.path.join(self.tmp, "jarvis_todo.md")
        self.history = os.path.join(self.tmp, "self_diagnostic.json")
        self.mod._TODO_PATH = self.todo
        self.mod._HISTORY_PATH = self.history

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

    # ── whats_broken ─────────────────────────────────────────────────────
    def test_whats_broken_no_todo_file(self):
        # _TODO_PATH points at a not-yet-created temp file.
        self.assertIn("can't find jarvis_todo.md", self.actions["whats_broken"](""))

    def test_whats_broken_clean_queue(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("# todo\n- [ ] **2026-05-30** something unrelated\n")
        self.assertIn("queue is clean", self.actions["whats_broken"](""))

    def test_whats_broken_lists_open_tasks(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: microphone reports x.\n")
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: stt reports y.\n")
        out = self.actions["whats_broken"]("")
        self.assertIn("2 open repair tasks", out)
        self.assertIn("microphone", out)
        self.assertIn("stt", out)

    def test_whats_broken_single_task_with_date(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-29** [self-diag] - Fix: webcam reports z.\n")
        out = self.actions["whats_broken"]("")
        self.assertIn("One open repair task", out)
        self.assertIn("webcam", out)
        self.assertIn("2026-05-29", out)

    def test_whats_broken_deduplicates_repeated_component(self):
        # The same component queued twice (different dates) must collapse to a
        # single entry via the seen-set dedup (covers the skip-`continue`).
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-29** [self-diag] - Fix: microphone reports a.\n")
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: microphone reports b.\n")
        out = self.actions["whats_broken"]("")
        # One unique component → "One open repair task", mentioned once.
        self.assertIn("One open repair task", out)
        self.assertEqual(out.lower().count("microphone"), 1)

    def test_whats_broken_includes_self_heal_tasks(self):
        # 2026-07-06 audit tail: whats_broken only matched [self-diag] and
        # silently missed the [self-heal] pipeline tasks its own sweep queues.
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: microphone reports x.\n")
            f.write("- [ ] **2026-05-30** [self-heal] - Fix: VAD has not tripped in 40m.\n")
            f.write("- [ ] **2026-05-30** [self-heal] - Fix: camera 0 hit a face_tracker error.\n")
        out = self.actions["whats_broken"]("")
        self.assertIn("3 open repair tasks", out)
        self.assertIn("microphone", out)
        self.assertIn("VAD", out)
        self.assertIn("camera", out)

    # ── _open_selfdiag_components dedupe ──────────────────────────────────
    def test_open_components_parses_todo(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: disk reports full.\n")
            f.write("- [x] **2026-05-29** [self-diag] - Fix: ram reports done.\n")  # closed
        comps = self.mod._open_selfdiag_components()
        self.assertIn("disk", comps)
        self.assertNotIn("ram", comps)   # checked box → not open

    # ── diagnostic_history ───────────────────────────────────────────────
    def test_diagnostic_history_empty(self):
        self.assertIn("No diagnostic history", self.actions["diagnostic_history"](""))

    def test_diagnostic_history_lists_runs(self):
        import json
        runs = [
            {"iso": "2026-05-30T01:00:00", "failed": []},
            {"iso": "2026-05-30T02:00:00", "failed": ["stt", "tts"]},
        ]
        with open(self.history, "w", encoding="utf-8") as f:
            json.dump(runs, f)
        out = self.actions["diagnostic_history"]("5")
        self.assertIn("all nominal", out)
        self.assertIn("2 issue(s)", out)
        self.assertIn("stt", out)

    # ── _queue_repair_task ───────────────────────────────────────────────
    def test_queue_repair_task_skips_low_severity(self):
        run = {"probes": {"claude_api": {"severity": self.mod.SEVERITY_LOW,
                                         "error": "capped"}}}
        self.assertFalse(self.mod._queue_repair_task("claude_api", run, []))
        # No file written for a LOW failure.
        self.assertFalse(os.path.exists(self.todo))

    def test_queue_repair_task_appends_for_med(self):
        run = {"probes": {"webcam": {"severity": self.mod.SEVERITY_MED,
                                     "error": "no frame", "latency_ms": 40,
                                     "details": {}}}}
        ok = self.mod._queue_repair_task("webcam", run, [])
        self.assertTrue(ok)
        with open(self.todo, encoding="utf-8") as f:
            body = f.read()
        self.assertIn("[self-diag] - Fix: webcam", body)
        self.assertIn("face_tracker", body)   # suggested file hint

    def test_queue_repair_task_dedupes_open_component(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: webcam reports x.\n")
        run = {"probes": {"webcam": {"severity": self.mod.SEVERITY_MED,
                                     "error": "again", "latency_ms": 1, "details": {}}}}
        # Already open → not queued again.
        self.assertFalse(self.mod._queue_repair_task("webcam", run, []))

    def test_queue_repair_task_unserialisable_details(self):
        # Details with a circular reference make json.dumps raise even with
        # default=str → the task body falls back to "(details unavailable)"
        # but is still queued.
        circ: dict = {}
        circ["self"] = circ
        run = {"probes": {"webcam": {"severity": self.mod.SEVERITY_MED,
                                     "error": "loop", "latency_ms": 2,
                                     "details": circ}}}
        ok = self.mod._queue_repair_task("webcam", run, [])
        self.assertTrue(ok)
        with open(self.todo, encoding="utf-8") as f:
            body = f.read()
        self.assertIn("details unavailable", body)


# ─────────────────────────────────────────────────────────────────────────
# Fakes for hardware/network modules the probes import lazily.
# ─────────────────────────────────────────────────────────────────────────
class _FakeFrame(list):
    """Stands in for a numpy image frame: .mean(), .size, .shape."""
    def __init__(self, seq, shape=(2, 2, 3)):
        super().__init__(seq)
        self.shape = shape

    @property
    def size(self):
        return len(self)

    def mean(self):
        return (sum(self) / len(self)) if len(self) else 0.0


class _FakeCap:
    """Fake cv2.VideoCapture. ``frames`` is a list of (ok, frame) tuples
    returned by successive read() calls; the last is repeated when exhausted."""
    def __init__(self, opened=True, frames=None):
        self._opened = opened
        self._frames = frames or [(True, _FakeFrame([200, 200, 200, 200]))]
        self._i = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._i < len(self._frames):
            r = self._frames[self._i]
            self._i += 1
            return r
        return self._frames[-1]

    def release(self):
        self.released = True


def make_cv2(open_map=None, frames=None, cascade_empty=False, raise_cascade=False):
    """Build a fake cv2 module. ``open_map`` maps camera index -> opened bool;
    default opens index 0. ``frames`` overrides the frame sequence."""
    cv2 = types.ModuleType("cv2")
    cv2.CAP_DSHOW = 700
    open_map = open_map if open_map is not None else {0: True}

    def _VideoCapture(idx, *a, **k):
        opened = open_map.get(idx, False)
        return _FakeCap(opened=opened, frames=frames)

    cv2.VideoCapture = _VideoCapture

    class _Cascade:
        def __init__(self, path):
            if raise_cascade:
                raise RuntimeError("cascade boom")
            self._path = path

        def empty(self):
            return cascade_empty

    cv2.CascadeClassifier = _Cascade
    data = types.SimpleNamespace(haarcascades="C:/cv2/data/")
    cv2.data = data
    return cv2


def make_sounddevice(devices, default_input=0, rec_rms=None, rec_raises=False):
    """Fake sounddevice. ``devices`` is a list of dicts with
    max_input_channels/name. ``rec_rms`` lets rec()->wait() yield a chosen RMS."""
    sd = types.ModuleType("sounddevice")
    sd.query_devices = lambda: devices
    sd.default = types.SimpleNamespace(device=(default_input, 1))

    def _rec(n, **k):
        if rec_raises:
            raise RuntimeError("PortAudio exploded")
        # Magnitude chosen so RMS computed by _fake_np equals rec_rms.
        val = 0.0 if rec_rms is None else rec_rms
        from tests.skills.test_self_diagnostic import _FakeFrame  # local _Arr-like
        arr = _FakeFrame([val, val], shape=(2,))
        return arr

    sd.rec = _rec
    sd.wait = lambda: None
    return sd


class _ProbeTestBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("self_diagnostic")
        # Make sure no stale bobert_companion leaks in from another test.
        self._saved_bc = sys.modules.get("bobert_companion")
        sys.modules.pop("bobert_companion", None)
        self.addCleanup(self._restore_bc)
        # Freeze time so latency math is deterministic and never sleeps. Use a
        # realistic epoch so cooldown math (now - 8h, now - 6h) stays positive
        # and a never-queued signature (ts 0.0) is correctly < cutoff.
        self._t = [1_700_000_000.0]
        mock.patch.object(self.mod, "_now", lambda: self._t[0]).start()
        mock.patch.object(self.mod.time, "sleep", lambda *_a, **_k: None).start()
        # ``subprocess.CREATE_NO_WINDOW`` is Windows-only. The PnP/hardware-count
        # probes reference it in a ``creationflags=(... if sys.platform=="win32"
        # else 0)`` kwarg that Python evaluates *before* calling the (mocked)
        # subprocess. On the Linux CI the attribute is absent, so even with
        # ``sys.platform`` forced to "win32" the expression raises AttributeError,
        # which the probe swallows and returns None/[] — making the win32-branch
        # assertions fail. Materialise the flag (no-op on Windows, where it already
        # exists with this value) so the Windows code path runs and is covered on
        # any host. stopall() restores/removes it after each test.
        mock.patch.object(self.mod.subprocess, "CREATE_NO_WINDOW",
                          0x08000000, create=True).start()
        self.addCleanup(mock.patch.stopall)

    def _restore_bc(self):
        if self._saved_bc is not None:
            sys.modules["bobert_companion"] = self._saved_bc
        else:
            sys.modules.pop("bobert_companion", None)


# ─── Probe 1: webcam ─────────────────────────────────────────────────────
class WebcamProbeTests(_ProbeTestBase):
    def test_opencv_missing(self):
        with block_import("cv2"):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertIn("opencv not importable", r["error"])

    def test_happy_path(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)),
                               (True, _FakeFrame([200] * 4))])
        with inject_modules(cv2=cv2):
            r = self.mod._probe_webcam()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["cascade"], "loaded")
        self.assertEqual(r["details"]["index"], 0)

    def test_cascade_empty_fails(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)),
                               (True, _FakeFrame([200] * 4))],
                       cascade_empty=True)
        with inject_modules(cv2=cv2):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertIn("cascade failed to load", r["error"])

    def test_cascade_raises(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)),
                               (True, _FakeFrame([200] * 4))],
                       raise_cascade=True)
        with inject_modules(cv2=cv2):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertIn("cascade failed", r["error"])

    def test_no_camera_absent_hardware(self):
        cv2 = make_cv2(open_map={})  # no index opens
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_windows_camera_pnp_devices", return_value=[]), \
             mock.patch.object(self.mod, "_maybe_announce_once") as ann:
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)
        self.assertEqual(r["details"]["failure_mode"], "hardware_absent")
        ann.assert_called_once()

    def test_no_camera_locked_by_other_app(self):
        cv2 = make_cv2(open_map={})
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_windows_camera_pnp_devices", return_value=None), \
             mock.patch.object(self.mod, "_windows_camera_hardware_count", return_value=1), \
             mock.patch.object(self.mod, "_camera_lock_suspects", return_value=["teams.exe"]):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)
        self.assertIn("teams.exe", r["error"])
        self.assertEqual(r["details"]["failure_mode"], "locked_by_other_app")

    def test_no_camera_pnp_problem(self):
        cv2 = make_cv2(open_map={})
        devs = [{"name": "Cam", "status": "Error", "problem": 43, "present": True}]
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_windows_camera_pnp_devices", return_value=devs), \
             mock.patch.object(self.mod, "_camera_lock_suspects", return_value=[]), \
             mock.patch.object(self.mod, "_maybe_announce_once"):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "pnp_device_problem")

    def test_no_camera_open_failed_generic(self):
        cv2 = make_cv2(open_map={})
        # PnP unavailable, no suspects, failure_mode ends up unknown -> generic.
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_windows_camera_pnp_devices", return_value=None), \
             mock.patch.object(self.mod, "_windows_camera_hardware_count", return_value=None), \
             mock.patch.object(self.mod, "_camera_lock_suspects", return_value=[]):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "open_failed")
        # generic open_failed keeps the default (non-LOW) severity
        self.assertIsNone(r["severity"])

    def test_read_no_frame_then_wake_recovers(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)),  # warmup
                               (False, None)])                 # real read fails
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_attempt_camera_wake",
                               return_value=(True, "woke up")):
            r = self.mod._probe_webcam()
        self.assertTrue(r["ok"])
        self.assertTrue(r["details"]["wake_recovered"])

    def test_read_no_frame_wake_fails_locked(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)), (False, None)])
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_attempt_camera_wake",
                               return_value=(False, "no recover")), \
             mock.patch.object(self.mod, "_windows_camera_pnp_devices", return_value=None), \
             mock.patch.object(self.mod, "_camera_lock_suspects", return_value=["zoom.exe"]):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "locked_by_other_app")
        self.assertIn("zoom.exe", r["error"])

    def test_read_no_frame_wake_fails_hardware_unplugged(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)), (False, None)])
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_attempt_camera_wake",
                               return_value=(False, "no recover")), \
             mock.patch.object(self.mod, "_windows_camera_pnp_devices", return_value=[]), \
             mock.patch.object(self.mod, "_camera_lock_suspects", return_value=[]), \
             mock.patch.object(self.mod, "_maybe_announce_once"):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "hardware_unplugged")

    def test_read_no_frame_wake_fails_pnp_problem(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)), (False, None)])
        devs = [{"name": "Cam", "status": "Error", "problem": 22, "present": True}]
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_attempt_camera_wake",
                               return_value=(False, "no recover")), \
             mock.patch.object(self.mod, "_windows_camera_pnp_devices", return_value=devs), \
             mock.patch.object(self.mod, "_camera_lock_suspects", return_value=[]), \
             mock.patch.object(self.mod, "_maybe_announce_once"):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "pnp_device_problem")

    def test_read_no_frame_wake_fails_unresponsive(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)), (False, None)])
        devs = [{"name": "Cam", "status": "OK", "problem": 0, "present": True}]
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_attempt_camera_wake",
                               return_value=(False, "no recover")), \
             mock.patch.object(self.mod, "_windows_camera_pnp_devices", return_value=devs), \
             mock.patch.object(self.mod, "_camera_lock_suspects", return_value=[]):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "unresponsive_after_wake")

    def test_persistent_black_frame(self):
        # Warmup ok, then all reads are black (mean < 1).
        black = _FakeFrame([0, 0, 0, 0])
        cv2 = make_cv2(frames=[(True, black), (True, black)])
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_maybe_announce_once"):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "persistent_black_frame")
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)

    def test_black_frame_then_recovers_on_retry(self):
        # First real frame black, a retry frame is bright -> sensor warmed up.
        cv2 = make_cv2(frames=[(True, _FakeFrame([0] * 4)),    # warmup
                               (True, _FakeFrame([0] * 4)),    # first real (black)
                               (True, _FakeFrame([200] * 4))])  # retry bright
        with inject_modules(cv2=cv2):
            r = self.mod._probe_webcam()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["cascade"], "loaded")


# ─── webcam helper functions ─────────────────────────────────────────────
class CameraHelperTests(_ProbeTestBase):
    def test_camera_lock_suspects_uses_bc_finder(self):
        bc = types.SimpleNamespace(
            find_camera_locking_processes=lambda: ["obs64.exe"])
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._camera_lock_suspects(), ["obs64.exe"])

    def test_camera_lock_suspects_bc_finder_raises_falls_back(self):
        def _boom():
            raise RuntimeError("nope")
        bc = types.SimpleNamespace(find_camera_locking_processes=_boom)
        fake_proc = mock.MagicMock()
        fake_proc.info = {"name": "Teams.exe"}
        psutil = types.ModuleType("psutil")
        psutil.process_iter = lambda attrs=None: [fake_proc]
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(psutil=psutil):
            out = self.mod._camera_lock_suspects()
        self.assertIn("Teams.exe", out)

    def test_camera_lock_suspects_no_psutil(self):
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             block_import("psutil"):
            self.assertEqual(self.mod._camera_lock_suspects(), [])

    def test_windows_camera_hardware_count_non_windows(self):
        with mock.patch.object(self.mod.sys, "platform", "linux"):
            self.assertIsNone(self.mod._windows_camera_hardware_count())

    def test_windows_camera_hardware_count_parses_int(self):
        proc = types.SimpleNamespace(returncode=0, stdout="2\n", stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertEqual(self.mod._windows_camera_hardware_count(), 2)

    def test_windows_camera_hardware_count_empty_is_zero(self):
        proc = types.SimpleNamespace(returncode=0, stdout="  \n", stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertEqual(self.mod._windows_camera_hardware_count(), 0)

    def test_windows_camera_hardware_count_bad_rc(self):
        proc = types.SimpleNamespace(returncode=1, stdout="", stderr="err")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertIsNone(self.mod._windows_camera_hardware_count())

    def test_windows_camera_hardware_count_raises(self):
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("no powershell")):
            self.assertIsNone(self.mod._windows_camera_hardware_count())

    def test_windows_camera_pnp_devices_non_windows(self):
        with mock.patch.object(self.mod.sys, "platform", "linux"):
            self.assertIsNone(self.mod._windows_camera_pnp_devices())

    def test_windows_camera_pnp_devices_parses_list(self):
        payload = json.dumps([
            {"FriendlyName": "Logi", "Status": "OK", "Class": "Camera",
             "Problem": 0, "Present": True}])
        proc = types.SimpleNamespace(returncode=0, stdout=payload, stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            devs = self.mod._windows_camera_pnp_devices()
        self.assertEqual(len(devs), 1)
        self.assertEqual(devs[0]["name"], "Logi")
        self.assertTrue(devs[0]["present"])

    def test_windows_camera_pnp_devices_single_object(self):
        payload = json.dumps({"FriendlyName": "Solo", "Status": "OK",
                              "Class": "Camera", "Problem": 0, "Present": True})
        proc = types.SimpleNamespace(returncode=0, stdout=payload, stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            devs = self.mod._windows_camera_pnp_devices()
        self.assertEqual(len(devs), 1)
        self.assertEqual(devs[0]["name"], "Solo")

    def test_windows_camera_pnp_devices_empty_stdout(self):
        proc = types.SimpleNamespace(returncode=0, stdout="  ", stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertEqual(self.mod._windows_camera_pnp_devices(), [])

    def test_windows_camera_pnp_devices_bad_json(self):
        proc = types.SimpleNamespace(returncode=0, stdout="{not json", stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertIsNone(self.mod._windows_camera_pnp_devices())

    def test_windows_camera_pnp_devices_bad_rc(self):
        proc = types.SimpleNamespace(returncode=2, stdout="", stderr="x")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertIsNone(self.mod._windows_camera_pnp_devices())

    def test_windows_camera_pnp_devices_subprocess_raises(self):
        # subprocess.run itself raising (PowerShell missing / timeout) → the
        # outer except returns None (covers the top-level guard).
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("powershell gone")):
            self.assertIsNone(self.mod._windows_camera_pnp_devices())

    def test_attempt_camera_wake_no_cv2(self):
        with block_import("cv2"):
            ok, note = self.mod._attempt_camera_wake(0)
        self.assertFalse(ok)
        self.assertIn("opencv unavailable", note)

    def test_attempt_camera_wake_success(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)),
                               (True, _FakeFrame([200] * 4))])
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_bc", return_value=None):
            ok, note = self.mod._attempt_camera_wake(0)
        self.assertTrue(ok)
        self.assertIn("wake succeeded", note)

    def test_attempt_camera_wake_reopen_refused(self):
        cv2 = make_cv2(open_map={0: False})
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_bc", return_value=None):
            ok, note = self.mod._attempt_camera_wake(0)
        self.assertFalse(ok)
        self.assertIn("refused open", note)

    def test_attempt_camera_wake_no_frame(self):
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)), (False, None)])
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_bc", return_value=None):
            ok, note = self.mod._attempt_camera_wake(0)
        self.assertFalse(ok)
        self.assertIn("no frame", note)

    def test_attempt_camera_wake_uses_io_lock(self):
        import threading as _thr
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)),
                               (True, _FakeFrame([200] * 4))])
        lock = _thr.Lock()
        bc = types.SimpleNamespace(_camera_io_lock=lock)
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            ok, _note = self.mod._attempt_camera_wake(0)
        self.assertTrue(ok)
        self.assertFalse(lock.locked())   # released afterward


# ─── Probe 2: microphone ─────────────────────────────────────────────────
class MicrophoneProbeTests(_ProbeTestBase):
    def test_sounddevice_missing(self):
        with block_import("sounddevice"):
            r = self.mod._probe_microphone()
        self.assertFalse(r["ok"])
        self.assertIn("sounddevice not importable", r["error"])

    def test_query_devices_raises(self):
        sd = types.ModuleType("sounddevice")
        sd.query_devices = mock.MagicMock(side_effect=RuntimeError("boom"))
        sd.default = types.SimpleNamespace(device=(0, 1))
        with inject_modules(sounddevice=sd):
            r = self.mod._probe_microphone()
        self.assertFalse(r["ok"])
        self.assertIn("query_devices failed", r["error"])

    def test_no_input_devices(self):
        sd = make_sounddevice([{"name": "Speakers", "max_input_channels": 0}])
        with inject_modules(sounddevice=sd):
            r = self.mod._probe_microphone()
        self.assertFalse(r["ok"])
        self.assertIn("no input devices enumerated", r["error"])

    def test_skips_live_capture_when_awake(self):
        sd = make_sounddevice([{"name": "Mic", "max_input_channels": 2}])
        bc = types.SimpleNamespace(_sleep_mode=[False],
                                   _mic_input_disabled=lambda: False,
                                   get_input_device=lambda: 0)
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            r = self.mod._probe_microphone()
        self.assertTrue(r["ok"])
        self.assertIn("awake", r["details"]["live_capture_skipped"])

    def test_skips_live_capture_when_mic_disabled(self):
        sd = make_sounddevice([{"name": "Mic", "max_input_channels": 2}])
        bc = types.SimpleNamespace(_sleep_mode=[True],
                                   _mic_input_disabled=lambda: True,
                                   get_input_device=lambda: 0)
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            r = self.mod._probe_microphone()
        self.assertTrue(r["ok"])
        self.assertIn("hard-disabled", r["details"]["live_capture_skipped"])

    def test_numpy_missing_when_capturing(self):
        sd = make_sounddevice([{"name": "Mic", "max_input_channels": 2}])
        with inject_modules(sounddevice=sd), \
             mock.patch.object(self.mod, "_bc", return_value=None), \
             block_import("numpy"):
            r = self.mod._probe_microphone()
        self.assertFalse(r["ok"])
        self.assertIn("numpy not importable", r["error"])

    def test_active_mic_has_signal(self):
        sd = make_sounddevice([{"name": "Mic", "max_input_channels": 2}],
                              rec_rms=0.5)
        with inject_modules(sounddevice=sd, numpy=_fake_np()), \
             mock.patch.object(self.mod, "_bc", return_value=None):
            r = self.mod._probe_microphone()
        self.assertTrue(r["ok"])
        self.assertGreaterEqual(r["details"]["rms"], self.mod.MIC_RMS_FLOOR)

    def test_all_inputs_silent_hardware_present(self):
        sd = make_sounddevice([{"name": "Mic", "max_input_channels": 2}],
                              rec_rms=0.0)
        with inject_modules(sounddevice=sd, numpy=_fake_np()), \
             mock.patch.object(self.mod, "_bc", return_value=None), \
             mock.patch.object(self.mod, "_windows_microphone_hardware_count",
                               return_value=1):
            r = self.mod._probe_microphone()
        self.assertFalse(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)
        self.assertIn("silent", r["error"])

    def test_alternate_mic_has_signal(self):
        devices = [
            {"name": "Active Mic", "max_input_channels": 2},   # idx 0 (active)
            {"name": "Backup Mic", "max_input_channels": 2},   # idx 1
        ]
        sd = make_sounddevice(devices, default_input=0)
        # Active silent, alternate loud: vary rec by device index.
        def _rec(n, device=None, **k):
            from tests.skills.test_self_diagnostic import _FakeFrame
            val = 0.5 if device == 1 else 0.0
            return _FakeFrame([val, val], shape=(2,))
        sd.rec = _rec
        with inject_modules(sounddevice=sd, numpy=_fake_np()), \
             mock.patch.object(self.mod, "_bc", return_value=None), \
             mock.patch.object(self.mod, "_windows_microphone_hardware_count",
                               return_value=2):
            r = self.mod._probe_microphone()
        self.assertFalse(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)
        self.assertIn("alternate", r["error"])

    def test_no_mic_hardware_pnp_zero(self):
        sd = make_sounddevice([{"name": "Mic", "max_input_channels": 2}],
                              rec_rms=0.0)
        with inject_modules(sounddevice=sd, numpy=_fake_np()), \
             mock.patch.object(self.mod, "_bc", return_value=None), \
             mock.patch.object(self.mod, "_windows_microphone_hardware_count",
                               return_value=0):
            r = self.mod._probe_microphone()
        self.assertFalse(r["ok"])
        self.assertIn("no microphone hardware detected", r["error"])

    def test_active_capture_raises_then_silent(self):
        sd = make_sounddevice([{"name": "Mic", "max_input_channels": 2}],
                              rec_raises=True)
        with inject_modules(sounddevice=sd, numpy=_fake_np()), \
             mock.patch.object(self.mod, "_bc", return_value=None), \
             mock.patch.object(self.mod, "_windows_microphone_hardware_count",
                               return_value=None):
            r = self.mod._probe_microphone()
        self.assertFalse(r["ok"])
        self.assertIn("active_capture_error", r["details"])


# ─── microphone helpers ──────────────────────────────────────────────────
class MicHelperTests(_ProbeTestBase):
    def test_active_mic_index_from_bc(self):
        sd = types.SimpleNamespace(default=types.SimpleNamespace(device=(3, 1)))
        bc = types.SimpleNamespace(get_input_device=lambda: 7)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._jarvis_active_mic_index(sd), 7)

    def test_active_mic_index_bc_raises_falls_back(self):
        sd = types.SimpleNamespace(default=types.SimpleNamespace(device=(3, 1)))
        def _boom():
            raise RuntimeError("x")
        bc = types.SimpleNamespace(get_input_device=_boom)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._jarvis_active_mic_index(sd), 3)

    def test_active_mic_index_no_bc_uses_default(self):
        sd = types.SimpleNamespace(default=types.SimpleNamespace(device=(2, 1)))
        with mock.patch.object(self.mod, "_bc", return_value=None):
            self.assertEqual(self.mod._jarvis_active_mic_index(sd), 2)

    def test_active_mic_index_all_unavailable(self):
        sd = types.SimpleNamespace(default=types.SimpleNamespace(device=(-1, 1)))
        with mock.patch.object(self.mod, "_bc", return_value=None):
            self.assertIsNone(self.mod._jarvis_active_mic_index(sd))

    def test_windows_microphone_hardware_count_non_windows(self):
        with mock.patch.object(self.mod.sys, "platform", "linux"):
            self.assertIsNone(self.mod._windows_microphone_hardware_count())

    def test_windows_microphone_hardware_count_parses(self):
        proc = types.SimpleNamespace(returncode=0, stdout="3\n", stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertEqual(self.mod._windows_microphone_hardware_count(), 3)

    def test_windows_microphone_hardware_count_raises(self):
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("no ps")):
            self.assertIsNone(self.mod._windows_microphone_hardware_count())


# ─── Probe 3: TTS ────────────────────────────────────────────────────────
class TtsProbeTests(_ProbeTestBase):
    def _requests(self, status=200, raises=False):
        req = types.ModuleType("requests")
        if raises:
            req.get = mock.MagicMock(side_effect=RuntimeError("offline"))
        else:
            req.get = mock.MagicMock(return_value=types.SimpleNamespace(status_code=status))
        return req

    def _pyttsx3(self, raises=False):
        mod = types.ModuleType("pyttsx3")
        if raises:
            mod.init = mock.MagicMock(side_effect=RuntimeError("no sapi"))
        else:
            eng = mock.MagicMock()
            eng.getProperty.return_value = ["v1"]
            mod.init = mock.MagicMock(return_value=eng)
        return mod

    def test_both_backends_ok(self):
        with inject_modules(requests=self._requests(), pyttsx3=self._pyttsx3()):
            r = self.mod._probe_tts()
        self.assertTrue(r["ok"])
        self.assertIsNone(r["severity"])   # both healthy → no degradation
        self.assertTrue(r["details"]["edge_ok"])
        self.assertTrue(r["details"]["pyttsx_ok"])

    def test_edge_down_pyttsx_ok_is_degraded(self):
        with inject_modules(requests=self._requests(raises=True),
                            pyttsx3=self._pyttsx3()):
            r = self.mod._probe_tts()
        self.assertTrue(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)
        self.assertIn("edge_error", r["details"])

    def test_pyttsx_down_edge_ok(self):
        with inject_modules(requests=self._requests(),
                            pyttsx3=self._pyttsx3(raises=True)):
            r = self.mod._probe_tts()
        self.assertTrue(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)
        self.assertIn("pyttsx_error", r["details"])

    def test_both_backends_fail(self):
        with inject_modules(requests=self._requests(raises=True),
                            pyttsx3=self._pyttsx3(raises=True)):
            r = self.mod._probe_tts()
        self.assertFalse(r["ok"])
        self.assertIn("both TTS backends failed", r["error"])

    def test_pyttsx_engine_stop_failure_is_swallowed(self):
        # eng.stop() raising during teardown must not fail the probe — the
        # except around stop() swallows it and pyttsx still counts as OK.
        mod = types.ModuleType("pyttsx3")
        eng = mock.MagicMock()
        eng.getProperty.return_value = ["v1"]
        eng.stop.side_effect = RuntimeError("stop boom")
        mod.init = mock.MagicMock(return_value=eng)
        with inject_modules(requests=self._requests(), pyttsx3=mod):
            r = self.mod._probe_tts()
        self.assertTrue(r["ok"])
        self.assertTrue(r["details"]["pyttsx_ok"])


# ─── Probe 4: STT ────────────────────────────────────────────────────────
class SttProbeTests(_ProbeTestBase):
    def _whisper(self, load_raises=False, transcribe_raises=None):
        mod = types.ModuleType("whisper")
        model = mock.MagicMock()
        type(model).__name__ = "Whisper"
        if transcribe_raises:
            model.transcribe = mock.MagicMock(side_effect=transcribe_raises)
        else:
            model.transcribe = mock.MagicMock(return_value={"text": "hello"})
        if load_raises:
            mod.load_model = mock.MagicMock(side_effect=RuntimeError("no weights"))
        else:
            mod.load_model = mock.MagicMock(return_value=model)
        return mod, model

    def test_whisper_not_importable(self):
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             block_import("whisper"):
            r = self.mod._probe_stt()
        self.assertFalse(r["ok"])
        self.assertIn("whisper not importable", r["error"])

    def test_load_model_fails(self):
        wmod, _ = self._whisper(load_raises=True)
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(whisper=wmod, numpy=_fake_np()):
            r = self.mod._probe_stt()
        self.assertFalse(r["ok"])
        self.assertIn("load_model('tiny') failed", r["error"])

    def test_happy_path_openai_whisper(self):
        wmod, _ = self._whisper()
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(whisper=wmod, numpy=_fake_np()):
            r = self.mod._probe_stt()
        self.assertTrue(r["ok"])
        self.assertIn("model_loaded", r["details"])

    def test_uses_cached_model(self):
        cached = mock.MagicMock()
        type(cached).__name__ = "Whisper"
        cached.transcribe = mock.MagicMock(return_value={"text": "hi"})
        bc = types.SimpleNamespace(_stt=cached, _stt_model_name="base",
                                   _stt_device="cuda")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(numpy=_fake_np()):
            r = self.mod._probe_stt()
        self.assertTrue(r["ok"])
        self.assertIn("cached from main loop", r["details"]["model_loaded"])

    def test_faster_whisper_model_path(self):
        wmod = types.ModuleType("whisper")
        model = mock.MagicMock()
        type(model).__name__ = "WhisperModel"
        seg = types.SimpleNamespace(text="hi")
        model.transcribe = mock.MagicMock(return_value=(iter([seg]), {}))
        wmod.load_model = mock.MagicMock(return_value=model)
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(whisper=wmod, numpy=_fake_np()):
            r = self.mod._probe_stt()
        self.assertTrue(r["ok"])

    def test_transcribe_generic_failure(self):
        wmod, _ = self._whisper(transcribe_raises=ValueError("bad audio"))
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(whisper=wmod, numpy=_fake_np()):
            r = self.mod._probe_stt()
        self.assertFalse(r["ok"])
        self.assertIn("transcribe failed", r["error"])
        self.assertIsNone(r["severity"])

    def test_transcribe_cuda_dll_error_cpu_fallback_ok(self):
        err = RuntimeError("cublas64_12.dll is not found or cannot be loaded")
        wmod, _ = self._whisper(transcribe_raises=err)
        # faster_whisper CPU fallback succeeds.
        fw = types.ModuleType("faster_whisper")
        cpu_model = mock.MagicMock()
        type(cpu_model).__name__ = "WhisperModel"
        cpu_model.transcribe = mock.MagicMock(return_value=(iter([]), {}))
        fw.WhisperModel = mock.MagicMock(return_value=cpu_model)
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(whisper=wmod, numpy=_fake_np(), faster_whisper=fw), \
             mock.patch.object(self.mod, "_maybe_announce_once") as ann:
            r = self.mod._probe_stt()
        self.assertFalse(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)
        self.assertEqual(r["details"]["failure_mode"], "cuda_dll_missing")
        self.assertTrue(r["details"]["cpu_fallback_ok"])
        ann.assert_called_once()

    def test_transcribe_cuda_dll_error_cpu_fallback_fails(self):
        err = RuntimeError("could not load library cudnn64_9.dll")
        wmod, _ = self._whisper(transcribe_raises=err)
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(whisper=wmod, numpy=_fake_np()), \
             block_import("faster_whisper"), \
             mock.patch.object(self.mod, "_maybe_announce_once") as ann:
            r = self.mod._probe_stt()
        self.assertFalse(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)
        self.assertFalse(r["details"]["cpu_fallback_ok"])
        ann.assert_called_once()


class SttHelperTests(_ProbeTestBase):
    def test_is_cuda_dll_error_true(self):
        self.assertTrue(self.mod._is_stt_cuda_dll_error(
            RuntimeError("cudart64_12.dll is not found or cannot be loaded")))

    def test_is_cuda_dll_error_false(self):
        self.assertFalse(self.mod._is_stt_cuda_dll_error(ValueError("bad shape")))

    def test_cuda_remediation_default(self):
        with mock.patch.object(self.mod, "_bc", return_value=None):
            note = self.mod._stt_cuda_remediation_note()
        self.assertIn("nvidia-cublas-cu12", note)

    def test_cuda_remediation_from_bc(self):
        bc = types.SimpleNamespace(_cuda_dll_remediation_note=lambda: "custom note")
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._stt_cuda_remediation_note(), "custom note")

    def test_cuda_remediation_bc_raises(self):
        def _boom():
            raise RuntimeError("x")
        bc = types.SimpleNamespace(_cuda_dll_remediation_note=_boom)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertIn("nvidia-cublas-cu12", self.mod._stt_cuda_remediation_note())


# ─── Probe 5: Claude API ─────────────────────────────────────────────────
class ClaudeApiProbeTests(_ProbeTestBase):
    def setUp(self):
        super().setUp()
        self._env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _anthropic(self, raises=None):
        mod = types.ModuleType("anthropic")
        client = mock.MagicMock()
        if raises is not None:
            client.messages.create = mock.MagicMock(side_effect=raises)
        else:
            client.messages.create = mock.MagicMock(return_value=mock.MagicMock())
        mod.Anthropic = mock.MagicMock(return_value=client)
        return mod

    def test_no_api_key_skips(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            r = self.mod._probe_claude_api()
        self.assertTrue(r["ok"])
        self.assertIn("skipped", r["details"])

    def test_anthropic_not_importable(self):
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             block_import("anthropic"):
            r = self.mod._probe_claude_api()
        self.assertFalse(r["ok"])
        self.assertIn("anthropic SDK not importable", r["error"])

    def test_ping_ok(self):
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(anthropic=self._anthropic()):
            r = self.mod._probe_claude_api()
        self.assertTrue(r["ok"])
        self.assertIn("model", r["details"])

    def test_ping_timeout(self):
        exc = type("APITimeoutError", (Exception,), {})("slow")
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(anthropic=self._anthropic(raises=exc)):
            r = self.mod._probe_claude_api()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "network_timeout")

    def test_ping_network_unreachable(self):
        exc = type("APIConnectionError", (Exception,), {})("down")
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(anthropic=self._anthropic(raises=exc)):
            r = self.mod._probe_claude_api()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "network_unreachable")

    def test_ping_api_error(self):
        exc = type("AuthenticationError", (Exception,), {})("bad key")
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(anthropic=self._anthropic(raises=exc)):
            r = self.mod._probe_claude_api()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "api_error")

    def test_model_from_bc(self):
        bc = types.SimpleNamespace(CLAUDE_MODEL="claude-custom")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(anthropic=self._anthropic()):
            r = self.mod._probe_claude_api()
        self.assertEqual(r["details"]["model"], "claude-custom")


# ─── Probe 6: internet ───────────────────────────────────────────────────
class InternetProbeTests(_ProbeTestBase):
    def test_both_ok(self):
        proc = types.SimpleNamespace(returncode=0, stdout="reply", stderr="")
        with mock.patch.object(self.mod.socket, "gethostbyname", return_value="1.2.3.4"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            r = self.mod._probe_internet()
        self.assertTrue(r["ok"])
        self.assertTrue(r["details"]["dns_ok"])
        self.assertTrue(r["details"]["ping_ok"])

    def test_dns_only_is_degraded(self):
        proc = types.SimpleNamespace(returncode=1, stdout="timeout", stderr="")
        with mock.patch.object(self.mod.socket, "gethostbyname", return_value="1.2.3.4"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            r = self.mod._probe_internet()
        self.assertTrue(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_LOW)

    def test_both_fail(self):
        with mock.patch.object(self.mod.socket, "gethostbyname",
                               side_effect=OSError("no dns")), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("no ping")):
            r = self.mod._probe_internet()
        self.assertFalse(r["ok"])
        self.assertIn("DNS and ICMP both failed", r["error"])

    def test_ping_only_non_windows_cmd(self):
        proc = types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with mock.patch.object(self.mod.sys, "platform", "linux"), \
             mock.patch.object(self.mod.socket, "gethostbyname",
                               side_effect=OSError("no dns")), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc) as run:
            r = self.mod._probe_internet()
        self.assertTrue(r["ok"])   # ping alone is still functional/degraded
        # non-windows uses -c flag
        self.assertIn("-c", run.call_args[0][0])


# ─── Probe 7: HUD subprocesses ───────────────────────────────────────────
class HudProbeTests(_ProbeTestBase):
    def test_no_bc_skips(self):
        with mock.patch.object(self.mod, "_bc", return_value=None):
            r = self.mod._probe_hud_subprocesses()
        self.assertTrue(r["ok"])
        self.assertIn("skipped", r["details"])

    def test_all_alive(self):
        alive_proc = mock.MagicMock()
        alive_proc.poll.return_value = None
        alive_proc.pid = 1234
        bc = types.SimpleNamespace(_hud_process=alive_proc,
                                   _reticle_process=alive_proc,
                                   _tray_process=alive_proc)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            r = self.mod._probe_hud_subprocesses()
        self.assertTrue(r["ok"])

    def test_dead_process(self):
        dead = mock.MagicMock()
        dead.poll.return_value = 1
        bc = types.SimpleNamespace(_hud_process=dead, _reticle_process=None,
                                   _tray_process=None)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            r = self.mod._probe_hud_subprocesses()
        self.assertFalse(r["ok"])
        self.assertIn("jarvis_hud", r["error"])

    def test_poll_raises(self):
        bad = mock.MagicMock()
        bad.poll.side_effect = RuntimeError("poll boom")
        bc = types.SimpleNamespace(_hud_process=bad, _reticle_process=None,
                                   _tray_process=None)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            r = self.mod._probe_hud_subprocesses()
        self.assertFalse(r["ok"])

    def test_workshop_hud_alive_and_dead(self):
        bc = types.SimpleNamespace(_hud_process=None, _reticle_process=None,
                                   _tray_process=None)
        overlay = types.ModuleType("skill_holographic_overlay")
        overlay._workshop_hud_is_alive = lambda: True
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(skill_holographic_overlay=overlay):
            r = self.mod._probe_hud_subprocesses()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["workshop_hud"], "alive")

        overlay._workshop_hud_is_alive = mock.MagicMock(side_effect=RuntimeError("x"))
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(skill_holographic_overlay=overlay):
            r2 = self.mod._probe_hud_subprocesses()
        self.assertFalse(r2["ok"])

    def test_workshop_hud_not_spawned(self):
        bc = types.SimpleNamespace(_hud_process=None, _reticle_process=None,
                                   _tray_process=None)
        overlay = types.ModuleType("skill_holographic_overlay")
        overlay._workshop_hud_is_alive = lambda: False
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(skill_holographic_overlay=overlay):
            r = self.mod._probe_hud_subprocesses()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["workshop_hud"], "not-spawned")


# ─── Probe 8: state files ────────────────────────────────────────────────
class StateFilesProbeTests(_ProbeTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="selfdiag_state_")
        self.addCleanup(self._cleanup)
        mock.patch.object(self.mod, "_PROJECT_DIR", self.tmp).start()

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

    def _write(self, name, content, age_s=120):
        p = os.path.join(self.tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        old = self.mod._now() - age_s   # _now frozen at 1000.0
        os.utime(p, (old, old))
        return p

    def test_all_valid(self):
        self._write("good.json", '{"a": 1}')
        r = self.mod._probe_state_files()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["parsed"], 1)

    def test_corrupt_file(self):
        self._write("bad.json", "{not json")
        r = self.mod._probe_state_files()
        self.assertFalse(r["ok"])
        self.assertIn("bad.json", r["error"])

    def test_recent_file_skipped(self):
        # Modified 5s ago (< 30s cooloff) → skipped, not parsed.
        self._write("fresh.json", "{bad", age_s=5)
        r = self.mod._probe_state_files()
        self.assertTrue(r["ok"])
        self.assertIn("fresh.json", r["details"]["skipped_recent"])

    def test_pending_speech_ignored(self):
        self._write("pending_speech.json", "{bad", age_s=120)
        r = self.mod._probe_state_files()
        self.assertTrue(r["ok"])

    def test_listdir_raises(self):
        with mock.patch.object(self.mod.os, "listdir",
                               side_effect=OSError("denied")):
            r = self.mod._probe_state_files()
        self.assertFalse(r["ok"])
        self.assertIn("could not list project root", r["error"])

    def test_getmtime_failure_falls_back_to_zero(self):
        # If os.path.getmtime raises, mtime defaults to 0.0 → the file is NOT
        # treated as recent (now - 0 >= 30s) so it's still parsed normally.
        self._write("good.json", '{"a": 1}', age_s=120)
        with mock.patch.object(self.mod.os.path, "getmtime",
                               side_effect=OSError("stat denied")):
            r = self.mod._probe_state_files()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["parsed"], 1)
        self.assertNotIn("good.json", r["details"].get("skipped_recent", []))


# ─── Probe 9: Bambu ──────────────────────────────────────────────────────
class BambuProbeTests(_ProbeTestBase):
    def setUp(self):
        super().setUp()
        # Shield every Bambu test from the REAL skills.bambu_monitor, whose
        # is_printer_offline() imports bobert_companion → sys.exit(0). A benign
        # fake (offline=False) lets the probe fall through to the MQTT path;
        # individual tests can still override it.
        bm = types.ModuleType("skills.bambu_monitor")
        bm.is_printer_offline = lambda: False
        self._bm_cm = inject_modules(**{"skills.bambu_monitor": bm})
        self._bm_cm.__enter__()
        self.addCleanup(lambda: self._bm_cm.__exit__(None, None, None))

    def test_no_bc_skips(self):
        with mock.patch.object(self.mod, "_bc", return_value=None):
            r = self.mod._probe_bambu()
        self.assertTrue(r["ok"])
        self.assertIn("skipped", r["details"])

    def test_not_configured_skips(self):
        bc = types.SimpleNamespace(BAMBU_PRINTER_IP="", BAMBU_ACCESS_CODE="",
                                   BAMBU_SERIAL="")
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            r = self.mod._probe_bambu()
        self.assertTrue(r["ok"])
        self.assertIn("not configured", r["details"]["skipped"])

    def test_printer_offline_skips(self):
        bc = types.SimpleNamespace(BAMBU_PRINTER_IP="1.2.3.4",
                                   BAMBU_ACCESS_CODE="code", BAMBU_SERIAL="ser")
        bm = types.ModuleType("skills.bambu_monitor")
        bm.is_printer_offline = lambda: True
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(**{"skills.bambu_monitor": bm}):
            r = self.mod._probe_bambu()
        self.assertTrue(r["ok"])
        self.assertIn("offline", r["details"]["skipped"])

    def test_paho_missing(self):
        bc = types.SimpleNamespace(BAMBU_PRINTER_IP="1.2.3.4",
                                   BAMBU_ACCESS_CODE="code", BAMBU_SERIAL="ser")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             block_import("paho"):
            r = self.mod._probe_bambu()
        self.assertFalse(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_MED)
        self.assertIn("paho-mqtt not installed", r["error"])

    def _mqtt(self, connect_rc=0, fire=True, connect_raises=False,
              cleanup_raises=False):
        """Return a 3-level fake paho package tree (paho / paho.mqtt /
        paho.mqtt.client) so ``import paho.mqtt.client`` resolves entirely to
        fakes — the real installed ``paho`` lacks the ``mqtt`` submodule."""
        client = types.ModuleType("paho.mqtt.client")
        client.MQTTv311 = 4

        class _Client:
            def __init__(self, *a, **k):
                self.on_connect = None

            def username_pw_set(self, *a, **k):
                pass

            def tls_set_context(self, *a, **k):
                pass

            def tls_insecure_set(self, *a, **k):
                pass

            def connect_async(self, *a, **k):
                if connect_raises:
                    raise RuntimeError("connect boom")

            def loop_start(self):
                if fire and self.on_connect:
                    self.on_connect(self, None, None, connect_rc)

            def loop_stop(self):
                if cleanup_raises:
                    raise RuntimeError("loop_stop boom")

            def disconnect(self):
                pass

        client.Client = _Client
        paho = types.ModuleType("paho")
        paho_mqtt = types.ModuleType("paho.mqtt")
        paho.mqtt = paho_mqtt
        paho_mqtt.client = client
        return {"paho": paho, "paho.mqtt": paho_mqtt, "paho.mqtt.client": client}

    def test_connect_success(self):
        bc = types.SimpleNamespace(BAMBU_PRINTER_IP="1.2.3.4",
                                   BAMBU_ACCESS_CODE="code", BAMBU_SERIAL="ser")
        # event.wait returns immediately because on_connect set the event.
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(**self._mqtt(connect_rc=0)), \
             mock.patch.object(self.mod.threading.Event, "wait", lambda self, timeout=None: True):
            r = self.mod._probe_bambu()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["ip"], "1.2.3.4")

    def test_connect_bad_rc(self):
        bc = types.SimpleNamespace(BAMBU_PRINTER_IP="1.2.3.4",
                                   BAMBU_ACCESS_CODE="code", BAMBU_SERIAL="ser")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(**self._mqtt(connect_rc=4)), \
             mock.patch.object(self.mod.threading.Event, "wait", lambda self, timeout=None: True):
            r = self.mod._probe_bambu()
        self.assertFalse(r["ok"])
        self.assertIn("rc=4", r["error"])

    def test_connect_timeout(self):
        bc = types.SimpleNamespace(BAMBU_PRINTER_IP="1.2.3.4",
                                   BAMBU_ACCESS_CODE="code", BAMBU_SERIAL="ser")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(**self._mqtt(fire=False)), \
             mock.patch.object(self.mod.threading.Event, "wait", lambda self, timeout=None: False):
            r = self.mod._probe_bambu()
        self.assertFalse(r["ok"])
        self.assertIn("timed out", r["error"])

    def test_connect_raises(self):
        bc = types.SimpleNamespace(BAMBU_PRINTER_IP="1.2.3.4",
                                   BAMBU_ACCESS_CODE="code", BAMBU_SERIAL="ser")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(**self._mqtt(connect_raises=True)), \
             mock.patch.object(self.mod.threading.Event, "wait", lambda self, timeout=None: False):
            r = self.mod._probe_bambu()
        self.assertFalse(r["ok"])
        self.assertIn("raised", r["error"])

    def test_cleanup_failure_is_swallowed(self):
        # A successful connect where loop_stop()/disconnect() raise during
        # teardown must not flip the result — the cleanup except swallows it.
        bc = types.SimpleNamespace(BAMBU_PRINTER_IP="1.2.3.4",
                                   BAMBU_ACCESS_CODE="code", BAMBU_SERIAL="ser")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(**self._mqtt(connect_rc=0, cleanup_raises=True)), \
             mock.patch.object(self.mod.threading.Event, "wait",
                               lambda self, timeout=None: True):
            r = self.mod._probe_bambu()
        self.assertTrue(r["ok"])


# ─── Probe 10: media playback ────────────────────────────────────────────
class MediaPlaybackProbeTests(_ProbeTestBase):
    def test_chrome_found(self):
        with mock.patch.object(self.mod.os.path, "exists",
                               lambda p: "chrome.exe" in p):
            r = self.mod._probe_media_playback()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["chrome"], "found")

    def test_apple_music_via_process(self):
        proc = mock.MagicMock()
        proc.info = {"name": "AppleMusic.exe"}
        psutil = types.ModuleType("psutil")
        psutil.process_iter = lambda attrs=None: [proc]
        with mock.patch.object(self.mod.os.path, "exists", lambda p: False), \
             mock.patch("glob.glob", return_value=[]), \
             inject_modules(psutil=psutil):
            r = self.mod._probe_media_playback()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["apple_music"], "found")

    def test_apple_music_via_glob(self):
        with mock.patch.object(self.mod.os.path, "exists", lambda p: False), \
             mock.patch("glob.glob", return_value=["C:/WindowsApps/AppleMusic"]):
            r = self.mod._probe_media_playback()
        self.assertTrue(r["ok"])

    def test_nothing_found(self):
        with mock.patch.object(self.mod.os.path, "exists", lambda p: False), \
             mock.patch("glob.glob", return_value=[]), \
             block_import("psutil"):
            r = self.mod._probe_media_playback()
        self.assertFalse(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_MED)
        self.assertIn("no playback target", r["error"])


# ─── Probe 11: skill imports ─────────────────────────────────────────────
class SkillImportsProbeTests(_ProbeTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="selfdiag_skills_")
        self.addCleanup(self._cleanup)
        self.skills = os.path.join(self.tmp, "skills")
        os.makedirs(self.skills)
        mock.patch.object(self.mod, "_PROJECT_DIR", self.tmp).start()

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_skill(self, name, src):
        with open(os.path.join(self.skills, name), "w", encoding="utf-8") as f:
            f.write(src)

    def test_skills_dir_missing(self):
        import shutil
        shutil.rmtree(self.skills)
        r = self.mod._probe_skill_imports()
        self.assertFalse(r["ok"])
        self.assertIn("skills directory missing", r["error"])

    def test_all_compile(self):
        self._write_skill("alpha.py", "x = 1\n")
        # uncached module → compiled fresh
        sys.modules.pop("skill_alpha", None)
        r = self.mod._probe_skill_imports()
        self.assertTrue(r["ok"])
        self.assertGreaterEqual(r["details"]["checked"], 1)

    def test_syntax_error_flagged(self):
        self._write_skill("broken.py", "def f(:\n")
        sys.modules.pop("skill_broken", None)
        r = self.mod._probe_skill_imports()
        self.assertFalse(r["ok"])
        self.assertIn("broken", r["error"])

    def test_cached_module_trusted(self):
        self._write_skill("cached_skill.py", "def f(:\n")  # would fail to compile
        # but it's already in sys.modules → trusted, skipped
        sys.modules["skill_cached_skill"] = types.ModuleType("skill_cached_skill")
        self.addCleanup(lambda: sys.modules.pop("skill_cached_skill", None))
        r = self.mod._probe_skill_imports()
        self.assertTrue(r["ok"])

    def test_underscore_skills_ignored(self):
        self._write_skill("_private.py", "def f(:\n")  # bad but ignored
        r = self.mod._probe_skill_imports()
        self.assertTrue(r["ok"])

    def test_listdir_raises(self):
        with mock.patch.object(self.mod.os, "listdir",
                               side_effect=OSError("denied")):
            r = self.mod._probe_skill_imports()
        self.assertFalse(r["ok"])
        self.assertIn("could not list skills dir", r["error"])

    def test_non_syntax_compile_error_flagged(self):
        # When compile() raises something OTHER than SyntaxError (e.g. a
        # ValueError / RecursionError from a pathological source), the generic
        # except branch records it. We force a non-SyntaxError from compile()
        # for our skill file so that branch is exercised deterministically.
        self._write_skill("weird.py", "x = 1\n")
        sys.modules.pop("skill_weird", None)
        self.addCleanup(lambda: sys.modules.pop("skill_weird", None))
        real_compile = compile

        def fake_compile(src, path, mode, *a, **k):
            if path.endswith("weird.py"):
                raise ValueError("pathological source")
            return real_compile(src, path, mode, *a, **k)

        with mock.patch("builtins.compile", side_effect=fake_compile):
            r = self.mod._probe_skill_imports()
        self.assertFalse(r["ok"])
        self.assertIn("weird", r["error"])
        # The recorded failure carries the ValueError class name, not SyntaxError.
        fail = next(f for f in r["details"]["failures"] if f["skill"] == "weird")
        self.assertIn("ValueError", fail["error"])


# ─── Probe 12: GPU ───────────────────────────────────────────────────────
class GpuProbeTests(_ProbeTestBase):
    def _torch(self, cuda_avail=True, cuda_raises=False):
        torch = types.ModuleType("torch")
        cuda = types.SimpleNamespace()
        if cuda_raises:
            cuda.is_available = mock.MagicMock(side_effect=RuntimeError("cuda boom"))
        else:
            cuda.is_available = lambda: cuda_avail
        cuda.get_device_name = lambda i: "RTX 4090"
        cuda.get_device_properties = lambda i: types.SimpleNamespace(
            total_memory=24 * 1024**3)
        torch.cuda = cuda
        return torch

    def test_torch_missing_auto_ok(self):
        bc = types.SimpleNamespace(WHISPER_DEVICE="auto")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             block_import("torch"):
            r = self.mod._probe_gpu()
        self.assertTrue(r["ok"])
        self.assertIn("skipped", r["details"])

    def test_torch_missing_cuda_required_fails(self):
        bc = types.SimpleNamespace(WHISPER_DEVICE="cuda")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             block_import("torch"):
            r = self.mod._probe_gpu()
        self.assertFalse(r["ok"])
        self.assertIn("torch not importable", r["error"])

    def test_cuda_available_reports_device(self):
        bc = types.SimpleNamespace(WHISPER_DEVICE="cuda")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(torch=self._torch(cuda_avail=True)):
            r = self.mod._probe_gpu()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["device_name"], "RTX 4090")
        self.assertEqual(r["details"]["vram_total_mb"], 24 * 1024)

    def test_cuda_required_but_unavailable(self):
        bc = types.SimpleNamespace(WHISPER_DEVICE="cuda")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(torch=self._torch(cuda_avail=False)):
            r = self.mod._probe_gpu()
        self.assertFalse(r["ok"])
        self.assertIn("is_available() is False", r["error"])

    def test_cuda_is_available_raises(self):
        bc = types.SimpleNamespace(WHISPER_DEVICE="auto")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(torch=self._torch(cuda_raises=True)):
            r = self.mod._probe_gpu()
        self.assertTrue(r["ok"])   # auto mode tolerates no CUDA
        self.assertIn("cuda_error", r["details"])

    def test_cpu_device_skips_cuda(self):
        bc = types.SimpleNamespace(WHISPER_DEVICE="cpu")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(torch=self._torch(cuda_avail=False)):
            r = self.mod._probe_gpu()
        self.assertTrue(r["ok"])

    def test_device_name_query_failure_is_swallowed(self):
        # CUDA is available but get_device_name/get_device_properties raise
        # (driver hiccup) — the probe still passes; the name/vram fields are
        # simply omitted (covers the inner except-pass).
        bc = types.SimpleNamespace(WHISPER_DEVICE="cuda")
        torch = self._torch(cuda_avail=True)
        torch.cuda.get_device_name = mock.MagicMock(
            side_effect=RuntimeError("nvml gone"))
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(torch=torch):
            r = self.mod._probe_gpu()
        self.assertTrue(r["ok"])
        self.assertNotIn("device_name", r["details"])


# ─── Probe 13: disk ──────────────────────────────────────────────────────
class DiskProbeTests(_ProbeTestBase):
    def test_plenty_free(self):
        usage = (500 * 1024**3, 100 * 1024**3, 400 * 1024**3)
        with mock.patch("shutil.disk_usage", return_value=usage):
            r = self.mod._probe_disk()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["free_gb"], 400.0)

    def test_low_free_fails(self):
        usage = (500 * 1024**3, 499 * 1024**3, 100 * 1024**2)  # ~100MB free
        with mock.patch("shutil.disk_usage", return_value=usage):
            r = self.mod._probe_disk()
        self.assertFalse(r["ok"])
        self.assertIn("GB free", r["error"])

    def test_disk_usage_raises(self):
        with mock.patch("shutil.disk_usage", side_effect=OSError("no path")):
            r = self.mod._probe_disk()
        self.assertFalse(r["ok"])
        self.assertIn("disk_usage failed", r["error"])


# ─── Probe 14: RAM ───────────────────────────────────────────────────────
class RamProbeTests(_ProbeTestBase):
    def _psutil(self, percent=50.0, vm_raises=False):
        psutil = types.ModuleType("psutil")
        if vm_raises:
            psutil.virtual_memory = mock.MagicMock(side_effect=RuntimeError("vm boom"))
        else:
            vm = types.SimpleNamespace(percent=percent, used=8 * 1024**3,
                                       total=16 * 1024**3)
            psutil.virtual_memory = lambda: vm
        return psutil

    def test_psutil_missing(self):
        with block_import("psutil"):
            r = self.mod._probe_ram()
        self.assertFalse(r["ok"])
        self.assertEqual(r["severity"], self.mod.SEVERITY_MED)
        self.assertIn("psutil not importable", r["error"])

    def test_ram_ok(self):
        with inject_modules(psutil=self._psutil(percent=40.0)):
            r = self.mod._probe_ram()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["percent"], 40.0)

    def test_ram_saturated(self):
        with inject_modules(psutil=self._psutil(percent=95.0)):
            r = self.mod._probe_ram()
        self.assertFalse(r["ok"])
        self.assertIn("95%", r["error"])

    def test_virtual_memory_raises(self):
        with inject_modules(psutil=self._psutil(vm_raises=True)):
            r = self.mod._probe_ram()
        self.assertFalse(r["ok"])
        self.assertIn("virtual_memory failed", r["error"])


# ─── Probe 15: optional skills ───────────────────────────────────────────
class OptionalSkillsProbeTests(_ProbeTestBase):
    def test_neither_loaded(self):
        for n in ("skill_alexa", "skill_alexa_voice", "skill_network_deco"):
            sys.modules.pop(n, None)
        r = self.mod._probe_optional_skills()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["alexa"], "skill not loaded")
        self.assertEqual(r["details"]["deco"], "skill not loaded")

    def test_alexa_probe_hook(self):
        alexa = types.ModuleType("skill_alexa")
        alexa.diagnostic_probe = lambda: "alexa ok"
        deco = types.ModuleType("skill_network_deco")
        deco.diagnostic_probe = mock.MagicMock(side_effect=RuntimeError("deco boom"))
        with inject_modules(skill_alexa=alexa, skill_network_deco=deco):
            r = self.mod._probe_optional_skills()
        self.assertEqual(r["details"]["alexa"], "alexa ok")
        self.assertIn("probe-raised", r["details"]["deco"])

    def test_loaded_no_hook(self):
        alexa = types.ModuleType("skill_alexa")
        deco = types.ModuleType("skill_network_deco")
        with inject_modules(skill_alexa=alexa, skill_network_deco=deco):
            r = self.mod._probe_optional_skills()
        self.assertEqual(r["details"]["alexa"], "loaded, no probe hook")
        self.assertEqual(r["details"]["deco"], "loaded, no probe hook")

    def test_alexa_probe_hook_raises(self):
        # The ALEXA hook itself raising is caught and reported (covers the
        # alexa-specific except branch).
        alexa = types.ModuleType("skill_alexa")
        alexa.diagnostic_probe = mock.MagicMock(
            side_effect=RuntimeError("alexa boom"))
        with inject_modules(skill_alexa=alexa):
            r = self.mod._probe_optional_skills()
        self.assertIn("probe-raised", r["details"]["alexa"])
        self.assertIn("alexa boom", r["details"]["alexa"])


# ─── _run_with_timeout ───────────────────────────────────────────────────
class RunWithTimeoutTests(_ProbeTestBase):
    def test_success(self):
        r = self.mod._run_with_timeout(lambda: {"ok": True, "v": 1}, 5.0, name="x")
        self.assertTrue(r["ok"])

    def test_probe_raises(self):
        def _boom():
            raise ValueError("kaboom")
        r = self.mod._run_with_timeout(_boom, 5.0, name="x")
        self.assertFalse(r["ok"])
        self.assertIn("ValueError: kaboom", r["error"])

    def test_non_dict_result(self):
        r = self.mod._run_with_timeout(lambda: "not a dict", 5.0, name="x")
        self.assertFalse(r["ok"])
        self.assertIn("non-dict result", r["error"])

    def test_timeout(self):
        # Patch the thread used inside _run_with_timeout so it reports alive.
        class _StuckThread:
            def __init__(self, *a, **k):
                pass

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return True

        with mock.patch.object(self.mod.threading, "Thread", _StuckThread):
            r = self.mod._run_with_timeout(lambda: {"ok": True}, 0.01, name="x")
        self.assertFalse(r["ok"])
        self.assertIn("timed out", r["error"])


# ─── _run_all_probes / run_diagnostic orchestration ──────────────────────
class SweepOrchestrationTests(_ProbeTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="selfdiag_sweep_")
        self.addCleanup(self._cleanup)
        self.todo = os.path.join(self.tmp, "jarvis_todo.md")
        self.history = os.path.join(self.tmp, "self_diagnostic.json")
        self.autoq = os.path.join(self.tmp, "autoqueue.json")
        mock.patch.object(self.mod, "_TODO_PATH", self.todo).start()
        mock.patch.object(self.mod, "_HISTORY_PATH", self.history).start()
        mock.patch.object(self.mod, "_AUTOQUEUE_PATH", self.autoq).start()
        # Run probes inline (no real threads) so the sweep is deterministic.
        mock.patch.object(self.mod, "_run_with_timeout",
                          lambda fn, t, name: fn()).start()
        # Reset shared module state between tests.
        self.mod._state["last_run"] = None
        self.mod._announced_failure_state.clear()

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch_probes(self, mapping):
        """Replace PROBES with deterministic fakes returning given results."""
        fake = {name: (lambda res=res: res) for name, res in mapping.items()}
        return mock.patch.object(self.mod, "PROBES", fake)

    def test_run_all_probes_all_pass(self):
        results = {n: self.mod._result(True, 1.0) for n in self.mod.PROBES}
        with self._patch_probes(results):
            run = self.mod._run_all_probes()
        self.assertEqual(run["failed"], [])
        self.assertEqual(len(run["probes"]), len(results))

    def test_run_all_probes_fills_default_severity(self):
        results = {"microphone": self.mod._result(False, 1.0, error="dead")}
        with self._patch_probes(results):
            run = self.mod._run_all_probes()
        self.assertEqual(run["severity_failed"]["microphone"], self.mod.SEVERITY_HIGH)

    def test_run_all_probes_respects_result_severity(self):
        results = {"webcam": self.mod._result(False, 1.0, error="x",
                                              severity=self.mod.SEVERITY_LOW)}
        with self._patch_probes(results):
            run = self.mod._run_all_probes()
        self.assertEqual(run["severity_failed"]["webcam"], self.mod.SEVERITY_LOW)

    def test_run_diagnostic_all_nominal(self):
        results = {n: self.mod._result(True, 1.0) for n in self.mod.PROBES}
        with self._patch_probes(results), \
             mock.patch.object(self.mod, "_run_autoqueue_pass", return_value=[]):
            out = self.mod.run_diagnostic("")
        self.assertIn("All systems nominal", out)
        self.assertEqual(self.mod._state["runs_completed"],
                         self.mod._state["runs_completed"])  # incremented
        self.assertTrue(os.path.exists(self.history))

    def test_run_diagnostic_queues_med_failure(self):
        results = {n: self.mod._result(True, 1.0) for n in self.mod.PROBES}
        results["webcam"] = self.mod._result(False, 1.0, error="no frame",
                                             severity=self.mod.SEVERITY_MED)
        with self._patch_probes(results), \
             mock.patch.object(self.mod, "_run_autoqueue_pass", return_value=[]), \
             mock.patch.object(self.mod, "_announce_failures"):
            out = self.mod.run_diagnostic("")
        self.assertIn("webcam", out)
        with open(self.todo, encoding="utf-8") as f:
            self.assertIn("[self-diag] - Fix: webcam", f.read())

    def test_run_diagnostic_high_failure_announces(self):
        results = {n: self.mod._result(True, 1.0) for n in self.mod.PROBES}
        results["microphone"] = self.mod._result(False, 1.0, error="dead",
                                                 severity=self.mod.SEVERITY_HIGH)
        with self._patch_probes(results), \
             mock.patch.object(self.mod, "_run_autoqueue_pass", return_value=[]), \
             mock.patch.object(self.mod, "_proactive_announce") as ann, \
             mock.patch.object(self.mod, "_push_phone") as push:
            self.mod.run_diagnostic("")
        ann.assert_called_once()
        push.assert_called_once()

    def test_run_diagnostic_low_claude_api_logs_info(self):
        results = {n: self.mod._result(True, 1.0) for n in self.mod.PROBES}
        results["claude_api"] = self.mod._result(False, 1.0, error="capped",
                                                 severity=self.mod.SEVERITY_LOW)
        with self._patch_probes(results), \
             mock.patch.object(self.mod, "_run_autoqueue_pass", return_value=[]):
            out = self.mod.run_diagnostic("")
        # LOW claude_api is not queued.
        self.assertFalse(os.path.exists(self.todo))
        self.assertIn("claude api", out.lower())

    def test_run_diagnostic_single_flight(self):
        # Acquire the lock so run_diagnostic sees an in-flight sweep.
        self.assertTrue(self.mod._run_lock.acquire(blocking=False))
        try:
            out = self.mod.run_diagnostic("")
        finally:
            self.mod._run_lock.release()
        self.assertIn("already in flight", out)

    def test_run_diagnostic_autoqueue_appends(self):
        results = {n: self.mod._result(True, 1.0) for n in self.mod.PROBES}
        with self._patch_probes(results), \
             mock.patch.object(self.mod, "_run_autoqueue_pass",
                               return_value=["vad_stall"]):
            out = self.mod.run_diagnostic("")
        # autoqueued signatures are folded into the queued count summary path
        self.assertIsInstance(out, str)


# ─── history persistence ─────────────────────────────────────────────────
class HistoryPersistenceTests(_ProbeTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="selfdiag_hist_")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.history = os.path.join(self.tmp, "self_diagnostic.json")
        mock.patch.object(self.mod, "_HISTORY_PATH", self.history).start()

    def test_load_history_missing(self):
        self.assertEqual(self.mod._load_history(), [])

    def test_load_history_list(self):
        with open(self.history, "w", encoding="utf-8") as f:
            json.dump([{"iso": "x"}], f)
        self.assertEqual(self.mod._load_history(), [{"iso": "x"}])

    def test_load_history_dict_with_runs(self):
        with open(self.history, "w", encoding="utf-8") as f:
            json.dump({"runs": [{"iso": "y"}]}, f)
        self.assertEqual(self.mod._load_history(), [{"iso": "y"}])

    def test_load_history_corrupt(self):
        with open(self.history, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(self.mod._load_history(), [])

    def test_save_history_trims(self):
        runs = [{"i": i} for i in range(self.mod.MAX_HISTORY_RUNS + 50)]
        self.mod._save_history(runs)
        loaded = self.mod._load_history()
        self.assertEqual(len(loaded), self.mod.MAX_HISTORY_RUNS)
        self.assertEqual(loaded[-1]["i"], self.mod.MAX_HISTORY_RUNS + 49)  # newest kept

    def test_save_history_write_failure_swallowed(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            # should not raise
            self.mod._save_history([{"i": 1}])


# ─── announcements + phone + recent-problem ──────────────────────────────
class AnnouncementTests(_ProbeTestBase):
    def setUp(self):
        super().setUp()
        self.mod._announced_failure_state.clear()
        self.mod._announce_cooldown.clear()

    def test_announce_failures_no_high(self):
        run = {"severity_failed": {"webcam": self.mod.SEVERITY_MED},
               "probes": {"webcam": {"error": "x"}}}
        with mock.patch.object(self.mod, "_proactive_announce") as ann:
            self.mod._announce_failures(run)
        ann.assert_not_called()

    def test_announce_failures_single_high(self):
        run = {"severity_failed": {"microphone": self.mod.SEVERITY_HIGH},
               "probes": {"microphone": {"error": "dead"}}}
        with mock.patch.object(self.mod, "_proactive_announce") as ann, \
             mock.patch.object(self.mod, "_push_phone") as push:
            self.mod._announce_failures(run)
        self.assertIn("the microphone", ann.call_args[0][0])
        push.assert_called_once()

    def test_announce_failures_multiple_high(self):
        run = {"severity_failed": {"microphone": self.mod.SEVERITY_HIGH,
                                   "disk": self.mod.SEVERITY_HIGH},
               "probes": {"microphone": {"error": "a"}, "disk": {"error": "b"}}}
        with mock.patch.object(self.mod, "_proactive_announce") as ann, \
             mock.patch.object(self.mod, "_push_phone"):
            self.mod._announce_failures(run)
        self.assertIn("multiple core systems", ann.call_args[0][0])

    def test_announce_failures_dedupes_same_signature(self):
        run = {"severity_failed": {"microphone": self.mod.SEVERITY_HIGH},
               "probes": {"microphone": {"error": "dead"}}}
        with mock.patch.object(self.mod, "_proactive_announce") as ann, \
             mock.patch.object(self.mod, "_push_phone"):
            self.mod._announce_failures(run)
            self.mod._announce_failures(run)  # same signature → silent 2nd time
        self.assertEqual(ann.call_count, 1)

    def test_announce_failures_marks_recent_problem(self):
        run = {"severity_failed": {"disk": self.mod.SEVERITY_HIGH},
               "probes": {"disk": {"error": "full"}}}
        with mock.patch.object(self.mod, "_proactive_announce"), \
             mock.patch.object(self.mod, "_push_phone"):
            self.mod._announce_failures(run)
        self.assertTrue(self.mod.get_recent_problem_flag())

    def test_proactive_announce_no_bc(self):
        with mock.patch.object(self.mod, "_bc", return_value=None):
            self.mod._proactive_announce("hi")  # no raise

    def test_proactive_announce_with_mood(self):
        bc = mock.MagicMock()
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.mod._proactive_announce("hi", mood="concerned_soft")
        bc.proactive_announce.assert_called_once()

    def test_proactive_announce_mood_typeerror_fallback(self):
        bc = mock.MagicMock()
        bc.proactive_announce.side_effect = [TypeError("no mood kwarg"), None]
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.mod._proactive_announce("hi", mood="concerned_soft")
        self.assertEqual(bc.proactive_announce.call_count, 2)

    def test_proactive_announce_swallows_exception(self):
        bc = mock.MagicMock()
        bc.proactive_announce.side_effect = RuntimeError("boom")
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.mod._proactive_announce("hi")  # no raise

    def test_proactive_announce_mood_fallback_also_fails(self):
        # mood given → first call raises TypeError (no mood kwarg) → the
        # no-mood fallback ALSO raises → that inner failure is logged and
        # swallowed (covers the nested except).
        bc = mock.MagicMock()
        bc.proactive_announce.side_effect = [TypeError("no mood kwarg"),
                                             RuntimeError("still broken")]
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.mod._proactive_announce("hi", mood="concerned_soft")  # no raise
        self.assertEqual(bc.proactive_announce.call_count, 2)

    def test_push_phone_no_module(self):
        sys.modules.pop("skill_phone_bridge", None)
        self.mod._push_phone("hi")  # no raise

    def test_push_phone_no_callable(self):
        mod = types.ModuleType("skill_phone_bridge")
        with inject_modules(skill_phone_bridge=mod):
            self.mod._push_phone("hi")  # push_to_phone missing → no raise

    def test_push_phone_calls_through(self):
        mod = types.ModuleType("skill_phone_bridge")
        mod.push_to_phone = mock.MagicMock()
        with inject_modules(skill_phone_bridge=mod):
            self.mod._push_phone("hi", priority="urgent")
        mod.push_to_phone.assert_called_once()

    def test_push_phone_swallows_exception(self):
        mod = types.ModuleType("skill_phone_bridge")
        mod.push_to_phone = mock.MagicMock(side_effect=RuntimeError("boom"))
        with inject_modules(skill_phone_bridge=mod):
            self.mod._push_phone("hi")  # no raise

    def test_maybe_announce_once_cooldown(self):
        with mock.patch.object(self.mod, "_proactive_announce") as ann:
            self.mod._maybe_announce_once("k", "msg")
            self.mod._maybe_announce_once("k", "msg")  # within cooldown → 1 call
        self.assertEqual(ann.call_count, 1)


# ─── _bc resolver ────────────────────────────────────────────────────────
class BcResolverTests(_ProbeTestBase):
    def test_bc_returns_module_when_present(self):
        fake = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=fake):
            self.assertIs(self.mod._bc(), fake)

    def test_bc_none_when_absent(self):
        sys.modules.pop("bobert_companion", None)
        self.assertIsNone(self.mod._bc())


# ─── autoqueue: state + signals + formatting ─────────────────────────────
class AutoqueueTests(_ProbeTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="selfdiag_aq_")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.todo = os.path.join(self.tmp, "jarvis_todo.md")
        self.autoq = os.path.join(self.tmp, "autoqueue.json")
        mock.patch.object(self.mod, "_TODO_PATH", self.todo).start()
        mock.patch.object(self.mod, "_AUTOQUEUE_PATH", self.autoq).start()

    def test_load_autoqueue_state_missing(self):
        self.assertEqual(self.mod._load_autoqueue_state(), {})

    def test_load_autoqueue_state_valid(self):
        with open(self.autoq, "w", encoding="utf-8") as f:
            json.dump({"sig": {"last_queued_ts": 1.0}}, f)
        self.assertEqual(self.mod._load_autoqueue_state(),
                         {"sig": {"last_queued_ts": 1.0}})

    def test_load_autoqueue_state_corrupt(self):
        with open(self.autoq, "w", encoding="utf-8") as f:
            f.write("{bad")
        self.assertEqual(self.mod._load_autoqueue_state(), {})

    def test_save_autoqueue_state_swallows(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("nope")):
            self.mod._save_autoqueue_state({"a": 1})  # no raise

    def test_session_log_tail_no_bc(self):
        with mock.patch.object(self.mod, "_bc", return_value=None):
            self.assertEqual(self.mod._session_log_tail(), [])

    def test_session_log_tail_reads_file(self):
        logp = os.path.join(self.tmp, "session.log")
        with open(logp, "w", encoding="utf-8") as f:
            f.write("\n".join(f"line{i}" for i in range(50)))
        bc = types.SimpleNamespace(get_session_log_path=lambda: logp)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            tail = self.mod._session_log_tail(n_lines=5)
        self.assertEqual(len(tail), 5)
        self.assertEqual(tail[-1], "line49")

    def test_session_log_tail_large_file_seeks(self):
        logp = os.path.join(self.tmp, "big.log")
        with open(logp, "w", encoding="utf-8") as f:
            f.write("\n".join(f"line{i}" for i in range(5000)))
        bc = types.SimpleNamespace(get_session_log_path=lambda: logp)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            tail = self.mod._session_log_tail(n_lines=3)
        self.assertEqual(len(tail), 3)

    def test_session_log_tail_path_fn_raises(self):
        def _boom():
            raise RuntimeError("x")
        bc = types.SimpleNamespace(get_session_log_path=_boom)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._session_log_tail(), [])

    def test_suggested_files_for_action_unknown_module(self):
        bc = mock.MagicMock()
        fn = mock.MagicMock()
        fn.__module__ = "weird_module"
        bc.ACTIONS = {"act": fn}
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._suggested_files_for_action("act"),
                             "weird_module (module)")

    def test_suggested_files_for_action_no_actions_dict(self):
        bc = types.SimpleNamespace(ACTIONS=None)
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertIn("dispatcher",
                          self.mod._suggested_files_for_action("act"))

    def test_suggested_files_for_action_no_module(self):
        bc = mock.MagicMock()
        fn = mock.MagicMock()
        fn.__module__ = None
        bc.ACTIONS = {"act": fn}
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertIn("dispatcher",
                          self.mod._suggested_files_for_action("act"))

    def test_collect_action_error_groups_getter_raises(self):
        bc = mock.MagicMock()
        bc.get_recent_action_errors.side_effect = RuntimeError("boom")
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._collect_action_error_groups(), [])

    def test_collect_action_error_groups_below_threshold(self):
        errs = [{"action": "a", "exc_class": "E", "exc_msg": "m",
                 "traceback": "tb", "ts": 1.0}]
        bc = mock.MagicMock()
        bc.get_recent_action_errors.return_value = errs
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._collect_action_error_groups(), [])

    def test_collect_action_error_groups_updates_last_ts(self):
        errs = [
            {"action": "a", "exc_class": "E", "exc_msg": "old",
             "traceback": "t1", "ts": 5.0},
            {"action": "a", "exc_class": "E", "exc_msg": "new",
             "traceback": "t2", "ts": 9.0},
            {"action": "a", "exc_class": "E", "exc_msg": "mid",
             "traceback": "t3", "ts": 1.0},
        ]
        bc = mock.MagicMock()
        bc.get_recent_action_errors.return_value = errs
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            groups = self.mod._collect_action_error_groups()
        self.assertEqual(groups[0]["count"], 3)
        self.assertEqual(groups[0]["exc_msg"], "new")    # latest ts wins
        self.assertEqual(groups[0]["first_ts"], 1.0)

    def test_collect_vad_stall_no_audio_processor(self):
        # Poison the submodule so `from core import audio_processor` raises
        # ImportError → the import-guard except returns None.
        import core
        saved_mod = sys.modules.get("core.audio_processor", _SENTINEL)
        had_attr = hasattr(core, "audio_processor")
        saved_attr = getattr(core, "audio_processor", None)
        sys.modules["core.audio_processor"] = None
        if had_attr:
            delattr(core, "audio_processor")
        try:
            self.assertIsNone(self.mod._collect_vad_stall_signal())
        finally:
            if saved_mod is _SENTINEL:
                sys.modules.pop("core.audio_processor", None)
            else:
                sys.modules["core.audio_processor"] = saved_mod
            if had_attr:
                core.audio_processor = saved_attr

    def test_collect_vad_stall_sleep_flag_unsubscriptable(self):
        # _sleep_mode present but not indexable → the guard treats JARVIS as
        # sleeping (fail-safe) and returns None (covers the sleep-flag except).
        ap = types.ModuleType("core.audio_processor")
        ap.get_vad_state = lambda: {}
        bc = types.SimpleNamespace(_sleep_mode=42)  # int → 42[0] raises TypeError
        with inject_modules(**{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertIsNone(self.mod._collect_vad_stall_signal())

    def test_collect_vad_stall_no_bc(self):
        ap = types.ModuleType("core.audio_processor")
        ap.get_vad_state = lambda: {}
        with inject_modules(**{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_bc", return_value=None):
            self.assertIsNone(self.mod._collect_vad_stall_signal())

    def test_collect_vad_stall_sleeping(self):
        ap = types.ModuleType("core.audio_processor")
        ap.get_vad_state = lambda: {}
        bc = types.SimpleNamespace(_sleep_mode=[True])
        with inject_modules(**{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertIsNone(self.mod._collect_vad_stall_signal())

    def test_collect_vad_stall_detected(self):
        now = self.mod._now()
        ap = types.ModuleType("core.audio_processor")
        ap.get_vad_state = lambda: {
            "last_vad_poll_ts": now - 1.0,         # fresh poll
            "last_vad_active_ts": now - 200.0,     # no trip for 200s
            "vad_session_start": now - 300.0,      # long session
            "total_vad_trips": 4,
        }
        bc = types.SimpleNamespace(_sleep_mode=[False])
        with inject_modules(**{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            sig = self.mod._collect_vad_stall_signal()
        self.assertIsNotNone(sig)
        self.assertEqual(sig["signature"], "vad_stall")

    def test_collect_vad_stall_poll_stale(self):
        now = self.mod._now()
        ap = types.ModuleType("core.audio_processor")
        ap.get_vad_state = lambda: {
            "last_vad_poll_ts": now - 100.0,   # stale poll → covered elsewhere
            "last_vad_active_ts": now - 200.0,
            "vad_session_start": now - 300.0,
        }
        bc = types.SimpleNamespace(_sleep_mode=[False])
        with inject_modules(**{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertIsNone(self.mod._collect_vad_stall_signal())

    def test_collect_vad_stall_recent_trip(self):
        now = self.mod._now()
        ap = types.ModuleType("core.audio_processor")
        ap.get_vad_state = lambda: {
            "last_vad_poll_ts": now - 1.0,
            "last_vad_active_ts": now - 5.0,   # tripped recently → no stall
            "vad_session_start": now - 300.0,
        }
        bc = types.SimpleNamespace(_sleep_mode=[False])
        with inject_modules(**{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertIsNone(self.mod._collect_vad_stall_signal())

    def test_collect_vad_stall_session_too_young(self):
        now = self.mod._now()
        ap = types.ModuleType("core.audio_processor")
        ap.get_vad_state = lambda: {
            "last_vad_poll_ts": now - 1.0,
            "last_vad_active_ts": now - 200.0,
            "vad_session_start": now - 5.0,   # session younger than stall window
        }
        bc = types.SimpleNamespace(_sleep_mode=[False])
        with inject_modules(**{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertIsNone(self.mod._collect_vad_stall_signal())

    def test_collect_vad_stall_get_state_raises(self):
        ap = types.ModuleType("core.audio_processor")
        ap.get_vad_state = mock.MagicMock(side_effect=RuntimeError("x"))
        bc = types.SimpleNamespace(_sleep_mode=[False])
        with inject_modules(**{"core.audio_processor": ap}), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertIsNone(self.mod._collect_vad_stall_signal())

    def test_collect_face_failure_no_module(self):
        sys.modules.pop("skill_face_tracker", None)
        self.assertEqual(self.mod._collect_face_failure_signals(), [])

    def test_collect_face_failure_no_fn(self):
        mod = types.ModuleType("skill_face_tracker")
        with inject_modules(skill_face_tracker=mod):
            self.assertEqual(self.mod._collect_face_failure_signals(), [])

    def test_collect_face_failure_signals(self):
        mod = types.ModuleType("skill_face_tracker")
        mod.get_read_failure_spike_signals = lambda threshold=0: [
            {"cam_index": 1, "consecutive_fails": 7, "max_consecutive_fails": 9,
             "last_error": "read False", "seconds_since_last_ok": 12.0}]
        with inject_modules(skill_face_tracker=mod):
            out = self.mod._collect_face_failure_signals()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["signature"], "face_read_fail::cam1")

    def test_collect_face_failure_fn_raises(self):
        mod = types.ModuleType("skill_face_tracker")
        mod.get_read_failure_spike_signals = mock.MagicMock(
            side_effect=RuntimeError("boom"))
        with inject_modules(skill_face_tracker=mod):
            self.assertEqual(self.mod._collect_face_failure_signals(), [])

    def test_collect_face_failure_skips_malformed_signal(self):
        # A signal missing 'cam_index' makes the f-string subscript raise
        # KeyError; that entry is skipped while a well-formed one survives
        # (covers the per-signal except-continue).
        mod = types.ModuleType("skill_face_tracker")
        mod.get_read_failure_spike_signals = lambda threshold=0: [
            {"consecutive_fails": 7},  # no cam_index → KeyError on build
            {"cam_index": 2, "consecutive_fails": 8, "max_consecutive_fails": 8,
             "last_error": "x", "seconds_since_last_ok": 5.0},
        ]
        with inject_modules(skill_face_tracker=mod):
            out = self.mod._collect_face_failure_signals()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["signature"], "face_read_fail::cam2")

    # ── formatters ───────────────────────────────────────────────────────
    def test_format_action_error_task(self):
        group = {"action": "play_music", "exc_class": "KeyError", "count": 3,
                 "exc_msg": "no key\nsecond line", "traceback": "a\nb\nc\nd\ne\nf"}
        with mock.patch.object(self.mod, "_suggested_files_for_action",
                               return_value="skills/music.py"):
            out = self.mod._format_action_error_task(group, ["log1", "log2"])
        self.assertIn("[self-heal]", out)
        self.assertIn("play_music", out)
        self.assertIn("skills/music.py", out)
        self.assertIn("log1", out)

    def test_format_action_error_task_no_log(self):
        group = {"action": "a", "exc_class": "E", "count": 5, "exc_msg": "",
                 "traceback": ""}
        with mock.patch.object(self.mod, "_suggested_files_for_action",
                               return_value="x.py"):
            out = self.mod._format_action_error_task(group, [])
        self.assertIn("session log unavailable", out)

    def test_format_vad_stall_task(self):
        sig = {"seconds_since_active": 75.0}
        out = self.mod._format_vad_stall_task(sig, ["L1"])
        self.assertIn("[self-heal]", out)
        self.assertIn("75s", out)
        self.assertIn("L1", out)

    def test_format_vad_stall_task_unknown_secs(self):
        out = self.mod._format_vad_stall_task({"seconds_since_active": None}, [])
        self.assertIn("unknown", out)

    def test_format_face_fail_task(self):
        sig = {"cam_index": 0, "consecutive_fails": 6, "max_consecutive_fails": 8,
               "last_error": "read False"}
        out = self.mod._format_face_fail_task(sig, ["L1"])
        self.assertIn("[self-heal]", out)
        self.assertIn("camera 0", out)

    def test_append_autoqueue_line_writes(self):
        ok = self.mod._append_autoqueue_line("- [ ] task")
        self.assertTrue(ok)
        with open(self.todo, encoding="utf-8") as f:
            self.assertIn("task", f.read())

    def test_append_autoqueue_line_write_failure(self):
        with mock.patch("builtins.open", side_effect=OSError("ro fs")):
            self.assertFalse(self.mod._append_autoqueue_line("x"))

    # ── _run_autoqueue_pass ──────────────────────────────────────────────
    def test_run_autoqueue_pass_appends_and_dedupes(self):
        group = {"signature": "action_error::a::E", "action": "a",
                 "exc_class": "E", "count": 3, "exc_msg": "m", "traceback": "tb"}
        with mock.patch.object(self.mod, "_collect_action_error_groups",
                               return_value=[group]), \
             mock.patch.object(self.mod, "_collect_vad_stall_signal", return_value=None), \
             mock.patch.object(self.mod, "_collect_face_failure_signals", return_value=[]), \
             mock.patch.object(self.mod, "_session_log_tail", return_value=[]):
            appended = self.mod._run_autoqueue_pass()
            self.assertEqual(appended, ["action_error::a::E"])
            # second pass within cooldown → no re-append
            appended2 = self.mod._run_autoqueue_pass()
        self.assertEqual(appended2, [])

    def test_run_autoqueue_pass_all_signal_types(self):
        group = {"signature": "action_error::a::E", "action": "a",
                 "exc_class": "E", "count": 3, "exc_msg": "m", "traceback": "tb"}
        vad = {"signature": "vad_stall", "seconds_since_active": 80.0}
        face = {"signature": "face_read_fail::cam0", "cam_index": 0,
                "consecutive_fails": 6}
        with mock.patch.object(self.mod, "_collect_action_error_groups",
                               return_value=[group]), \
             mock.patch.object(self.mod, "_collect_vad_stall_signal", return_value=vad), \
             mock.patch.object(self.mod, "_collect_face_failure_signals",
                               return_value=[face]), \
             mock.patch.object(self.mod, "_session_log_tail", return_value=["L"]):
            appended = self.mod._run_autoqueue_pass()
        self.assertEqual(set(appended),
                         {"action_error::a::E", "vad_stall", "face_read_fail::cam0"})

    def test_run_autoqueue_pass_swallows_exception(self):
        with mock.patch.object(self.mod, "_load_autoqueue_state",
                               side_effect=RuntimeError("boom")):
            self.assertEqual(self.mod._run_autoqueue_pass(), [])

    def test_run_autoqueue_pass_skips_face_signal_within_cooldown(self):
        # A face signal whose signature was queued moments ago must be skipped
        # by the cooldown guard (the face-loop `continue`), so nothing new is
        # appended even though the signal is still present.
        face = {"signature": "face_read_fail::cam0", "cam_index": 0,
                "consecutive_fails": 6}
        fresh_state = {"face_read_fail::cam0": {"last_queued_ts": self.mod._now(),
                                                "kind": "face_read_fail"}}
        with mock.patch.object(self.mod, "_collect_action_error_groups",
                               return_value=[]), \
             mock.patch.object(self.mod, "_collect_vad_stall_signal",
                               return_value=None), \
             mock.patch.object(self.mod, "_collect_face_failure_signals",
                               return_value=[face]), \
             mock.patch.object(self.mod, "_load_autoqueue_state",
                               return_value=fresh_state), \
             mock.patch.object(self.mod, "_session_log_tail", return_value=[]):
            appended = self.mod._run_autoqueue_pass()
        self.assertEqual(appended, [])


# ─── _queue_repair_task extra paths ──────────────────────────────────────
class QueueRepairExtraTests(_ProbeTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="selfdiag_qr_")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.todo = os.path.join(self.tmp, "jarvis_todo.md")
        mock.patch.object(self.mod, "_TODO_PATH", self.todo).start()

    def test_queue_includes_details_blob(self):
        run = {"probes": {"gpu": {"severity": self.mod.SEVERITY_MED,
                                  "error": "cuda gone", "latency_ms": 12,
                                  "details": {"cuda_available": False}}}}
        ok = self.mod._queue_repair_task("gpu", run, [])
        self.assertTrue(ok)
        with open(self.todo, encoding="utf-8") as f:
            body = f.read()
        self.assertIn("cuda_available", body)

    def test_queue_write_failure_returns_false(self):
        run = {"probes": {"gpu": {"severity": self.mod.SEVERITY_MED,
                                  "error": "x", "latency_ms": 1, "details": {}}}}
        with mock.patch("builtins.open", side_effect=OSError("ro")):
            self.assertFalse(self.mod._queue_repair_task("gpu", run, []))

    def test_queue_uses_last_successful_ts(self):
        history = [{"iso": "2026-05-01T00:00:00",
                    "probes": {"disk": {"ok": True}}}]
        run = {"probes": {"disk": {"severity": self.mod.SEVERITY_HIGH,
                                   "error": "full", "latency_ms": 3,
                                   "details": {}}}}
        self.mod._queue_repair_task("disk", run, history)
        with open(self.todo, encoding="utf-8") as f:
            self.assertIn("2026-05-01T00:00:00", f.read())


# ─── summary / status edge paths ─────────────────────────────────────────
class SummaryEdgeTests(_ProbeTestBase):
    def _run(self, failed, sev=None):
        sev = sev or {}
        return {
            "ts": 0.0, "iso": "2026-05-30T00:00:00", "duration_ms": 1500.0,
            "probes": {c: {"severity": sev.get(c, self.mod.SEVERITY_MED),
                           "error": f"{c} down"} for c in failed},
            "failed": failed,
            "severity_failed": {c: sev.get(c, self.mod.SEVERITY_MED) for c in failed},
        }

    def test_summarise_four_issues_inline(self):
        comps = ["webcam", "stt", "tts", "gpu"]
        out = self.mod._summarise(self._run(comps), [])
        self.assertIn("4 issues", out)
        self.assertNotIn("more", out)   # 4 listed inline, no truncation

    def test_summarise_no_queued_omits_repair_line(self):
        out = self.mod._summarise(self._run(["gpu"]), [])
        self.assertNotIn("repair task", out)

    def test_summarise_plural_repair_tasks(self):
        out = self.mod._summarise(self._run(["gpu", "tts"]), ["gpu", "tts"])
        self.assertIn("2 repair tasks", out)

    def test_diagnostic_status_minutes_age(self):
        run = self._run([])
        run["ts"] = self.mod._now() - 600   # 10 min
        self.mod._state["last_run"] = run
        out = self.actions["diagnostic_status"]("")
        self.assertIn("minutes ago", out)

    def test_diagnostic_status_hours_age(self):
        run = self._run([])
        run["ts"] = self.mod._now() - 7200   # 2 h
        self.mod._state["last_run"] = run
        out = self.actions["diagnostic_status"]("")
        self.assertIn("hours ago", out)

    def test_last_diagnostic_run_empty(self):
        self.mod._state["last_run"] = None
        self.assertEqual(self.actions["last_diagnostic_run"](""), "{}")

    def test_last_diagnostic_run_serialises(self):
        self.mod._state["last_run"] = self._run(["gpu"])
        out = self.actions["last_diagnostic_run"]("")
        self.assertIn("gpu", out)
        self.assertIn("severity", out)

    def test_diagnostic_history_clamps_arg(self):
        # arg far above max clamps to 25; junk arg falls back to default.
        with mock.patch.object(self.mod, "_load_history", return_value=[]):
            self.assertIn("No diagnostic history",
                          self.actions["diagnostic_history"]("999"))
            self.assertIn("No diagnostic history",
                          self.actions["diagnostic_history"]("notanumber"))

    def test_whats_broken_scan_error(self):
        # _TODO_PATH exists but open() raises mid-scan.
        with open(self.mod._TODO_PATH if os.path.isabs(str(self.mod._TODO_PATH))
                  else os.devnull, "a"):
            pass


# ─── scheduling + register ───────────────────────────────────────────────
class SchedulingTests(_ProbeTestBase):
    def test_schedule_recurring_uses_apscheduler(self):
        sched = types.ModuleType("core.scheduler")
        sched.is_available = lambda: True
        sched.schedule_interval = mock.MagicMock()
        with inject_modules(**{"core.scheduler": sched}):
            self.assertTrue(self.mod._schedule_recurring_sweep())
        sched.schedule_interval.assert_called_once()

    def test_schedule_recurring_unavailable(self):
        sched = types.ModuleType("core.scheduler")
        sched.is_available = lambda: False
        with inject_modules(**{"core.scheduler": sched}):
            self.assertFalse(self.mod._schedule_recurring_sweep())

    def test_schedule_recurring_schedule_raises(self):
        sched = types.ModuleType("core.scheduler")
        sched.is_available = lambda: True
        sched.schedule_interval = mock.MagicMock(side_effect=RuntimeError("not ready"))
        with inject_modules(**{"core.scheduler": sched}):
            self.assertFalse(self.mod._schedule_recurring_sweep())

    def test_schedule_recurring_import_fails(self):
        with block_import("core.scheduler"):
            self.assertFalse(self.mod._schedule_recurring_sweep())

    def test_register_wires_actions_apscheduler_path(self):
        actions: dict = {}
        sched = types.ModuleType("core.scheduler")
        sched.is_available = lambda: True
        sched.schedule_interval = mock.MagicMock()
        with inject_modules(**{"core.scheduler": sched}), \
             mock.patch.object(self.mod.threading, "Thread") as Thr:
            self.mod.register(actions)
        for name in ("run_diagnostic", "system_check", "are_you_ok",
                     "self_diagnostic", "diagnostic_status", "whats_broken",
                     "what_is_broken", "diagnostic_history", "last_diagnostic_run"):
            self.assertIn(name, actions)
        Thr.assert_called()   # boot sweep thread spawned

    def test_register_timer_fallback_path(self):
        actions: dict = {}
        with mock.patch.object(self.mod, "_schedule_recurring_sweep",
                               return_value=False), \
             mock.patch.object(self.mod, "_spawn_timer_thread") as spawn:
            self.mod.register(actions)
        spawn.assert_called_once()
        self.assertIn("run_diagnostic", actions)

    def test_spawn_timer_thread(self):
        with mock.patch.object(self.mod.threading, "Thread") as Thr:
            self.mod._spawn_timer_thread()
        Thr.assert_called_once()
        Thr.return_value.start.assert_called_once()

    def test_timer_based_sweep_loop_runs_then_breaks(self):
        # Boot sweep + one interval sweep, then break the infinite loop by
        # raising from the 2nd sleep. run_diagnostic is a no-op spy.
        calls = {"sleep": 0}

        def _sleep(_secs):
            calls["sleep"] += 1
            if calls["sleep"] >= 2:
                raise KeyboardInterrupt("stop loop")

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "run_diagnostic") as rd:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._timer_based_sweep_loop()
        rd.assert_called()   # boot sweep fired

    def test_timer_based_sweep_loop_boot_sweep_raises(self):
        # Boot sweep raises → logged, then 1st interval sleep raises to exit.
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=[None, KeyboardInterrupt()]), \
             mock.patch.object(self.mod, "run_diagnostic",
                               side_effect=RuntimeError("boom")):
            with self.assertRaises(KeyboardInterrupt):
                self.mod._timer_based_sweep_loop()

    def test_register_boot_sweep_closure_runs(self):
        # Capture the boot-sweep target passed to Thread (apscheduler path) and
        # invoke it directly to cover the closure body.
        actions: dict = {}
        sched = types.ModuleType("core.scheduler")
        sched.is_available = lambda: True
        sched.schedule_interval = mock.MagicMock()
        captured = {}

        def _fake_thread(target=None, name=None, daemon=None):
            captured["target"] = target
            return mock.MagicMock()

        with inject_modules(**{"core.scheduler": sched}), \
             mock.patch.object(self.mod.threading, "Thread", _fake_thread), \
             mock.patch.object(self.mod.time, "sleep", lambda *_a: None), \
             mock.patch.object(self.mod, "run_diagnostic") as rd:
            self.mod.register(actions)
            self.assertIn("target", captured)
            captured["target"]()          # run the boot-sweep closure
        rd.assert_called_once_with("")

    def test_register_boot_sweep_closure_swallows(self):
        actions: dict = {}
        sched = types.ModuleType("core.scheduler")
        sched.is_available = lambda: True
        sched.schedule_interval = mock.MagicMock()
        captured = {}

        def _fake_thread(target=None, name=None, daemon=None):
            captured["target"] = target
            return mock.MagicMock()

        with inject_modules(**{"core.scheduler": sched}), \
             mock.patch.object(self.mod.threading, "Thread", _fake_thread), \
             mock.patch.object(self.mod.time, "sleep", lambda *_a: None), \
             mock.patch.object(self.mod, "run_diagnostic",
                               side_effect=RuntimeError("boom")):
            self.mod.register(actions)
            captured["target"]()          # closure swallows the exception


# ─── misc exception-handler coverage ─────────────────────────────────────
class MiscExceptionTests(_ProbeTestBase):
    def test_camera_lock_suspects_proc_iter_raises(self):
        psutil = types.ModuleType("psutil")
        psutil.process_iter = mock.MagicMock(side_effect=RuntimeError("iter boom"))
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(psutil=psutil):
            self.assertEqual(self.mod._camera_lock_suspects(), [])

    def test_camera_lock_suspects_proc_info_raises(self):
        bad_proc = mock.MagicMock()
        type(bad_proc).info = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("no info")))
        psutil = types.ModuleType("psutil")
        psutil.process_iter = lambda attrs=None: [bad_proc]
        with mock.patch.object(self.mod, "_bc", return_value=None), \
             inject_modules(psutil=psutil):
            self.assertEqual(self.mod._camera_lock_suspects(), [])

    def test_bc_finder_returns_non_list_falls_through(self):
        bc = types.SimpleNamespace(find_camera_locking_processes=lambda: "nope")
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             block_import("psutil"):
            self.assertEqual(self.mod._camera_lock_suspects(), [])

    def test_open_selfdiag_components_read_error(self):
        # _TODO_PATH exists but open() raises → returns empty set.
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._open_selfdiag_components(), set())

    def test_bambu_monitor_query_raises_falls_through(self):
        # is_printer_offline() raises → caught, falls through to the paho path,
        # which we make missing so we get a clean error result.
        bc = types.SimpleNamespace(BAMBU_PRINTER_IP="1.2.3.4",
                                   BAMBU_ACCESS_CODE="c", BAMBU_SERIAL="s")
        bm = types.ModuleType("skills.bambu_monitor")
        bm.is_printer_offline = mock.MagicMock(side_effect=RuntimeError("query boom"))
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             inject_modules(**{"skills.bambu_monitor": bm}), \
             block_import("paho"):
            r = self.mod._probe_bambu()
        self.assertFalse(r["ok"])
        self.assertIn("paho-mqtt not installed", r["error"])

    def test_whats_broken_scan_raises(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            out = self.mod.whats_broken("")
        self.assertIn("couldn't scan", out)

    def test_last_diagnostic_run_unserialisable(self):
        # default=str normally stringifies anything; force json.dumps to raise.
        self.mod._state["last_run"] = {"x": 1}
        with mock.patch.object(self.mod.json, "dumps",
                               side_effect=RuntimeError("nope")):
            out = self.mod.last_diagnostic_run("")
        self.assertIn("couldn't serialise", out)

    def test_session_log_tail_read_raises(self):
        logp = os.path.join(tempfile.gettempdir(), "diag_missing_xyz.log")
        bc = types.SimpleNamespace(get_session_log_path=lambda: logp)
        with mock.patch.object(self.mod, "_bc", return_value=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os.path, "getsize", return_value=10), \
             mock.patch("builtins.open", side_effect=OSError("io")):
            self.assertEqual(self.mod._session_log_tail(), [])

    def test_run_diagnostic_autoqueue_raises_swallowed(self):
        tmp = tempfile.mkdtemp(prefix="selfdiag_aqr_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        with mock.patch.object(self.mod, "_TODO_PATH",
                               os.path.join(tmp, "todo.md")), \
             mock.patch.object(self.mod, "_HISTORY_PATH",
                               os.path.join(tmp, "hist.json")), \
             mock.patch.object(self.mod, "_run_with_timeout",
                               lambda fn, t, name: fn()), \
             mock.patch.object(self.mod, "PROBES",
                               {"disk": lambda: self.mod._result(True, 1.0)}), \
             mock.patch.object(self.mod, "_run_autoqueue_pass",
                               side_effect=RuntimeError("aq boom")):
            out = self.mod.run_diagnostic("")   # must not raise
        self.assertIn("nominal", out.lower())


# ─── deeper probe-internal exception branches ────────────────────────────
class ProbeInternalBranchTests(_ProbeTestBase):
    def test_webcam_black_retry_read_returns_false(self):
        # warmup ok, first real read black, retry read returns (False, None),
        # then a black frame again → still persistent black.
        black = _FakeFrame([0, 0, 0, 0])
        frames = [(True, black), (True, black), (False, None), (True, black),
                  (True, black)]
        cv2 = make_cv2(frames=frames)
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_maybe_announce_once"):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "persistent_black_frame")
        # one of the retry means is 0.0 from the failed read
        self.assertIn(0.0, r["details"]["frame_retry_means"])

    def test_wake_warmup_read_raises_swallowed(self):
        # First warmup read() raises (swallowed), second read yields a frame.
        class _Cap:
            def __init__(self):
                self._n = 0

            def isOpened(self):
                return True

            def read(self):
                self._n += 1
                if self._n == 1:
                    raise RuntimeError("warmup read boom")
                return True, _FakeFrame([200] * 4)

            def release(self):
                pass

        cv2 = types.ModuleType("cv2")
        cv2.CAP_DSHOW = 700
        cv2.VideoCapture = lambda *a, **k: _Cap()
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_bc", return_value=None):
            ok, _note = self.mod._attempt_camera_wake(0)
        self.assertTrue(ok)

    def test_wake_times_out(self):
        # A thread that never finishes → join times out → wake reports timeout.
        class _StuckThread:
            def __init__(self, *a, **k):
                pass

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return True

        cv2 = make_cv2()
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_bc", return_value=None), \
             mock.patch.object(self.mod.threading, "Thread", _StuckThread):
            ok, note = self.mod._attempt_camera_wake(0, timeout_s=0.01)
        self.assertFalse(ok)
        self.assertIn("timed out", note)

    def test_webcam_videocapture_construction_raises(self):
        # VideoCapture(idx) raising for every index → no cap → open_failed.
        cv2 = types.ModuleType("cv2")
        cv2.CAP_DSHOW = 700
        cv2.VideoCapture = mock.MagicMock(side_effect=RuntimeError("ctor boom"))
        cv2.data = types.SimpleNamespace(haarcascades="C:/cv2/")
        cv2.CascadeClassifier = lambda p: types.SimpleNamespace(empty=lambda: False)
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_windows_camera_pnp_devices", return_value=None), \
             mock.patch.object(self.mod, "_windows_camera_hardware_count", return_value=None), \
             mock.patch.object(self.mod, "_camera_lock_suspects", return_value=[]):
            r = self.mod._probe_webcam()
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["failure_mode"], "open_failed")

    def test_schedule_recurring_is_available_raises(self):
        sched = types.ModuleType("core.scheduler")
        sched.is_available = mock.MagicMock(side_effect=RuntimeError("boom"))
        with inject_modules(**{"core.scheduler": sched}):
            self.assertFalse(self.mod._schedule_recurring_sweep())

    def test_wake_io_lock_release_raises_is_swallowed(self):
        # A lock whose release() raises must not propagate out of the wake.
        cv2 = make_cv2(frames=[(True, _FakeFrame([200] * 4)),
                               (True, _FakeFrame([200] * 4))])

        class _BadLock:
            def acquire(self, *a, **k):
                return True

            def release(self, *a, **k):
                raise RuntimeError("release boom")

        bc = types.SimpleNamespace(_camera_io_lock=_BadLock())
        with inject_modules(cv2=cv2), \
             mock.patch.object(self.mod, "_bc", return_value=bc):
            ok, _note = self.mod._attempt_camera_wake(0)
        self.assertTrue(ok)   # frame produced despite release() raising

    def test_pnp_devices_json_not_list_or_dict(self):
        # ConvertTo-Json yielded a bare scalar → neither dict nor list → None.
        proc = types.SimpleNamespace(returncode=0, stdout="42", stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertIsNone(self.mod._windows_camera_pnp_devices())

    def test_pnp_devices_skips_non_dict_entries(self):
        payload = json.dumps([{"FriendlyName": "Cam", "Status": "OK",
                               "Problem": 0, "Present": True}, "junk", 7])
        proc = types.SimpleNamespace(returncode=0, stdout=payload, stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            devs = self.mod._windows_camera_pnp_devices()
        self.assertEqual(len(devs), 1)   # only the dict entry survived

    def test_media_glob_raises_swallowed(self):
        with mock.patch.object(self.mod.os.path, "exists", lambda p: False), \
             mock.patch("glob.glob", side_effect=RuntimeError("glob boom")), \
             block_import("psutil"):
            r = self.mod._probe_media_playback()
        self.assertFalse(r["ok"])   # nothing found, glob error swallowed

    def test_media_proc_info_raises_swallowed(self):
        bad = mock.MagicMock()
        type(bad).info = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("no info")))
        psutil = types.ModuleType("psutil")
        psutil.process_iter = lambda attrs=None: [bad]
        with mock.patch.object(self.mod.os.path, "exists", lambda p: False), \
             mock.patch("glob.glob", return_value=[]), \
             inject_modules(psutil=psutil):
            r = self.mod._probe_media_playback()
        self.assertFalse(r["ok"])   # proc.info error swallowed, nothing found

    def test_collect_action_error_groups_skips_bad_entry(self):
        class _BadEntry:
            def get(self, *a, **k):
                raise RuntimeError("bad entry")
        good = [{"action": "a", "exc_class": "E", "exc_msg": "m",
                 "traceback": "t", "ts": float(i)} for i in range(3)]
        errs = good + [_BadEntry()]
        bc = mock.MagicMock()
        bc.get_recent_action_errors.return_value = errs
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            groups = self.mod._collect_action_error_groups()
        self.assertEqual(len(groups), 1)   # bad entry skipped, good group kept

    def test_timer_loop_full_interval_iteration(self):
        # Let boot sweep + interval sleep + interval sweep all run, then break
        # on the 2nd interval sleep so the loop body (run_diagnostic) executes.
        seq = [None, None, KeyboardInterrupt()]   # boot, interval#1, interval#2
        with mock.patch.object(self.mod.time, "sleep", side_effect=seq), \
             mock.patch.object(self.mod, "run_diagnostic") as rd:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._timer_based_sweep_loop()
        self.assertEqual(rd.call_count, 2)   # boot + one interval sweep

    def test_timer_loop_interval_sweep_raises_swallowed(self):
        # The interval sweep raising is logged and the loop continues until the
        # next sleep aborts it.
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=[None, None, KeyboardInterrupt()]), \
             mock.patch.object(self.mod, "run_diagnostic",
                               side_effect=[None, RuntimeError("sweep boom")]):
            with self.assertRaises(KeyboardInterrupt):
                self.mod._timer_based_sweep_loop()

    def test_state_files_skips_non_json_and_non_file(self):
        tmp = tempfile.mkdtemp(prefix="selfdiag_sf2_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        # a non-.json file, and a DIRECTORY named like a .json (isfile False)
        with open(os.path.join(tmp, "notes.txt"), "w") as f:
            f.write("x")
        os.makedirs(os.path.join(tmp, "weird.json"))
        good = os.path.join(tmp, "ok.json")
        with open(good, "w", encoding="utf-8") as f:
            f.write('{"a": 1}')
        old = self.mod._now() - 120
        os.utime(good, (old, old))
        with mock.patch.object(self.mod, "_PROJECT_DIR", tmp):
            r = self.mod._probe_state_files()
        self.assertTrue(r["ok"])
        self.assertEqual(r["details"]["parsed"], 1)   # only ok.json parsed

    def test_mic_pnp_count_bad_rc_and_empty(self):
        bad = types.SimpleNamespace(returncode=1, stdout="", stderr="e")
        empty = types.SimpleNamespace(returncode=0, stdout="  ", stderr="")
        with mock.patch.object(self.mod.sys, "platform", "win32"):
            with mock.patch.object(self.mod.subprocess, "run", return_value=bad):
                self.assertIsNone(self.mod._windows_microphone_hardware_count())
            with mock.patch.object(self.mod.subprocess, "run", return_value=empty):
                self.assertEqual(self.mod._windows_microphone_hardware_count(), 0)

    def test_suggested_files_for_action_bobert_module(self):
        bc = mock.MagicMock()
        fn = mock.MagicMock()
        fn.__module__ = "bobert_companion"
        bc.ACTIONS = {"speak": fn}
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._suggested_files_for_action("speak"),
                             "bobert_companion.py")

    def test_mic_skips_virtual_and_dup_alternates(self):
        devices = [
            {"name": "Active Mic", "max_input_channels": 2},   # 0 active/silent
            {"name": "Stereo Mix", "max_input_channels": 2},   # 1 virtual → skip
            {"name": "Backup Mic", "max_input_channels": 2},   # 2 alt
            {"name": "Backup Mic", "max_input_channels": 2},   # 3 dup name → skip
            {"name": "", "max_input_channels": 2},             # 4 empty → skip
        ]
        sd = make_sounddevice(devices, default_input=0, rec_rms=0.0)
        with inject_modules(sounddevice=sd, numpy=_fake_np()), \
             mock.patch.object(self.mod, "_bc", return_value=None), \
             mock.patch.object(self.mod, "_windows_microphone_hardware_count",
                               return_value=2):
            r = self.mod._probe_microphone()
        self.assertFalse(r["ok"])
        tried_names = [a["name"] for a in r["details"].get("alternates_tried", [])]
        # Only the single unique physical "Backup Mic" alternate was probed.
        self.assertEqual(tried_names.count("Backup Mic"), 1)
        self.assertNotIn("Stereo Mix", tried_names)


if __name__ == "__main__":
    unittest.main()
