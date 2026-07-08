"""Logic tests for skills/anticipation_engine.py.

Targets the pure decision logic the proactive scheduler is built from:
duration/clock formatting, productivity-window detection + app-name shortening,
in-call suppression, the late-night gate, dwell tracking, and the individual
trigger pickers (_try_long_dwell / _try_late_hour_active). Also covers the
anticipation_status action and the speech-queue writer.

The second half drives the harder paths deterministically: config read with a
fake bobert_companion, persistent-state load/save (round-trip + corruption +
save-failure), the window/face/speech environment probes via injected fake
modules, the pattern-offer bridge, the whole _scheduler_loop across every gate
and trigger (time/random/sleep all frozen, sleep raising a sentinel to stop the
otherwise-infinite loop after one pass), the rich _format_status branches, and
register() (enabled / disabled / duplicate-thread).

The background scheduler thread is neutered by the harness; we never let it run.
bobert_companion / face_tracker are absent (sys.modules lookups return None) for
the permissive-default tests, and injected as fakes (removed in tearDown via a
save/restore context manager) where a present-module path is exercised. No real
network/LLM/thread/sleep; real numpy is untouched.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _struct(hour, minute):
    return time.struct_time((2026, 6, 1, hour, minute, 0, 0, 152, -1))


# ─── fake-module injection (mirrors tests/skills/test_self_diagnostic.py) ──
_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules and, for dotted
    names, set the leaf as an attribute on its already-imported parent package.
    Restores the previous state — including absence — on exit so the skill sees
    exactly the fake we provide and tests stay isolated. Pass ``name=None`` to
    force-remove a module for the duration of the block."""
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
    """Force a bare ``import <name>`` to raise ImportError inside the block,
    even when the real dependency is installed on the dev box. Both
    ``_all_window_titles`` and ``_focused_window_title`` use a bare
    ``import pygetwindow`` (not importlib), so a None sys.modules entry alone is
    fragile across re-imports — we detach any already-imported target AND patch
    ``__import__`` to raise, restoring both afterwards."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in blocked or top in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    saved_mod: dict[str, object] = {}
    for name in blocked:
        if name in sys.modules:
            saved_mod[name] = sys.modules.pop(name)
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        for name, mod in saved_mod.items():
            sys.modules[name] = mod


def _fake_gw(all_titles=None, active_title=None, raise_all=False,
             raise_active=False):
    """Build a fake ``pygetwindow`` module. ``all_titles`` feeds
    getAllWindows(); ``active_title`` feeds getActiveWindow()."""
    gw = types.ModuleType("pygetwindow")

    class _Win:
        def __init__(self, title):
            self.title = title

    def _get_all():
        if raise_all:
            raise RuntimeError("enum boom")
        return [_Win(t) for t in (all_titles or [])]

    def _get_active():
        if raise_active:
            raise RuntimeError("active boom")
        return _Win(active_title) if active_title is not None else None

    gw.getAllWindows = _get_all
    gw.getActiveWindow = _get_active
    return gw


class _StopLoop(Exception):
    """Sentinel raised from a patched time.sleep to break _scheduler_loop after
    a fixed number of poll iterations."""


def _sleep_after(n):
    """Return a fake time.sleep that no-ops the first `n` calls then raises
    _StopLoop — lets a single scheduler pass complete then exits the loop."""
    box = {"n": n}

    def _fake(_seconds):
        if box["n"] <= 0:
            raise _StopLoop()
        box["n"] -= 1
    return _fake


def _benign_bc():
    """A minimal fake ``bobert_companion`` exposing only the config knobs the
    engine reads, with defaults that keep the engine ENABLED."""
    bc = types.ModuleType("bobert_companion")
    bc.ANTICIPATION_ENABLED = True
    bc.ANTICIPATION_COOLDOWN_MINUTES = 20
    return bc


class _EngineTestBase(unittest.TestCase):
    """Shared setup that loads the skill in isolation WITH a benign fake
    ``bobert_companion`` pre-installed.

    Critical: ``register()`` (run by the harness at load) and ``_read_config``
    both do ``importlib.import_module("bobert_companion")``. With the real
    monolith absent from sys.modules that import executes ``bobert_companion.py``
    for real — which runs an early-boot singleton lock that calls
    ``sys.exit(0)`` when another JARVIS instance is live, aborting the whole test
    run. Pre-seeding a fake (and any other module the loader/gates consult) makes
    import_module return the fake instead, and we restore sys.modules — including
    prior absence — in tearDown so nothing leaks between tests.
    """
    #: subclasses may override to seed extra fake modules during load
    _EXTRA_MODULES: dict = {}

    def setUp(self):
        self._seeded = {}
        seed = {"bobert_companion": _benign_bc()}
        seed.update(self._EXTRA_MODULES)
        for name, obj in seed.items():
            self._seeded[name] = sys.modules.get(name, _SENTINEL)
            sys.modules[name] = obj
        self.addCleanup(self._restore_seeded)
        self.mod, self.actions = load_skill_isolated("anticipation_engine")
        # Several helpers print on their failure paths (state save, pattern
        # offer, register, speech-queue). Tests assert on return values / side
        # effects, so swallow direct-call stdout to keep the runner output clean.
        _out = contextlib.redirect_stdout(io.StringIO())
        _out.__enter__()
        self.addCleanup(_out.__exit__, None, None, None)

    def _restore_seeded(self):
        for name, prev in self._seeded.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class AnticipationHelperTests(_EngineTestBase):
    # ── formatting ───────────────────────────────────────────────────────
    def test_format_hours_minutes(self):
        f = self.mod._format_hours_minutes
        self.assertEqual(f(0), "0 minutes")
        self.assertEqual(f(600), "10 minutes")
        self.assertEqual(f(3600), "1 hour")
        self.assertEqual(f(7200), "2 hours")
        self.assertEqual(f(7800), "2 hours and 10 minutes")

    def test_format_clock(self):
        self.assertEqual(self.mod._format_clock(_struct(9, 5)), "9:05 AM")
        self.assertEqual(self.mod._format_clock(_struct(0, 0)), "12:00 AM")
        self.assertEqual(self.mod._format_clock(_struct(13, 30)), "1:30 PM")
        self.assertEqual(self.mod._format_clock(_struct(12, 0)), "12:00 PM")

    # ── window helpers ───────────────────────────────────────────────────
    def test_is_productivity_window(self):
        self.assertTrue(self.mod._is_productivity_window("untitled - Blender"))
        self.assertTrue(self.mod._is_productivity_window(
            "part.f3d - Autodesk Fusion 360"))
        self.assertFalse(self.mod._is_productivity_window("Solitaire"))
        self.assertFalse(self.mod._is_productivity_window(""))

    def test_shorten_app_name_known_hint(self):
        # The longest matching hint wins: "visual studio code" → title-cased.
        self.assertEqual(
            self.mod._shorten_app_name("project - Visual Studio Code"),
            "Visual Studio Code")
        # OpenSCAD is in the acronym set, so it stays styled.
        self.assertEqual(
            self.mod._shorten_app_name("model.scad - OpenSCAD"), "OpenSCAD")
        # The "vscode" hint maps to the "VS Code" acronym form.
        self.assertEqual(
            self.mod._shorten_app_name("foo - vscode"), "VS Code")

    def test_shorten_app_name_separator_fallback(self):
        # No known hint → take the tail after the last separator.
        self.assertEqual(
            self.mod._shorten_app_name("Inbox - SomeMailApp"), "SomeMailApp")

    def test_shorten_app_name_empty(self):
        self.assertEqual(self.mod._shorten_app_name(""), "")

    def test_shorten_app_name_emdash_separator(self):
        # The em-dash separator branch (first in the tuple).
        self.assertEqual(
            self.mod._shorten_app_name("Doc — WeirdEditor"), "WeirdEditor")

    def test_shorten_app_name_no_separator_truncates(self):
        # No hint and no separator → first 40 chars of the raw title.
        long = "x" * 60
        self.assertEqual(self.mod._shorten_app_name(long), "x" * 40)

    def test_shorten_app_name_long_tail_falls_through(self):
        # A separator exists but the tail is > 40 chars → ignored, raw[:40].
        title = "head - " + ("y" * 50)
        self.assertEqual(self.mod._shorten_app_name(title), title[:40])

    def test_title_case_app_acronyms(self):
        self.assertEqual(self.mod._title_case_app("vscode"), "VS Code")
        self.assertEqual(self.mod._title_case_app("freecad"), "FreeCAD")
        self.assertEqual(self.mod._title_case_app("blender"), "Blender")
        self.assertEqual(self.mod._title_case_app("intellij"), "IntelliJ")
        self.assertEqual(self.mod._title_case_app("openscad"), "OpenSCAD")
        self.assertEqual(self.mod._title_case_app("ableton"), "Ableton")

    def test_call_window_hints_present(self):
        # The shared call-hint list must include the major platforms.
        joined = " ".join(self.mod.CALL_WINDOW_HINTS)
        self.assertIn("zoom meeting", joined)
        self.assertIn("discord call", joined)

    # ── in-call detection ────────────────────────────────────────────────
    def test_is_in_call_matches_title(self):
        with mock.patch.object(self.mod, "_all_window_titles",
                               return_value=["Weekly sync | Microsoft Teams Meeting"]):
            self.assertTrue(self.mod._is_in_call())

    def test_is_in_call_false_when_no_meeting(self):
        with mock.patch.object(self.mod, "_all_window_titles",
                               return_value=["Inbox - Outlook", "Notepad"]):
            self.assertFalse(self.mod._is_in_call())

    def test_is_in_call_false_when_no_windows(self):
        with mock.patch.object(self.mod, "_all_window_titles", return_value=[]):
            self.assertFalse(self.mod._is_in_call())

    # ── absent-dependency defaults ───────────────────────────────────────
    def test_user_at_desk_none_without_tracker(self):
        # No face_tracker module loaded → None (permissive, not "away").
        with inject_modules(skill_face_tracker=None):
            self.assertIsNone(self.mod._user_at_desk())

    def test_sleep_or_standby_false_without_bc(self):
        # With bobert_companion absent, the gate reads False (permissive).
        with inject_modules(bobert_companion=None):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_last_speech_age_none_without_bc(self):
        # No bobert_companion → no last_speech_time → None.
        with inject_modules(bobert_companion=None):
            self.assertIsNone(self.mod._last_speech_age_seconds())


# ─── config + persistent state ───────────────────────────────────────────
class AnticipationConfigStateTests(_EngineTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="anticip_eng_")
        self.addCleanup(self._cleanup)
        self.state_file = os.path.join(self.tmp, "anticipation_state.json")
        # Redirect persistence at the module level so the real project file is
        # never read or written.
        self.mod._STATE_FILE = self.state_file
        self.mod._PROJECT_DIR = self.tmp

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

    # ── _read_config with a present bobert_companion ─────────────────────
    # NOTE: we do NOT test the bc-absent path by popping bobert_companion from
    # sys.modules — _read_config calls importlib.import_module("bobert_companion")
    # which, with the real module absent, executes the monolith for real and its
    # early-boot singleton lock calls sys.exit(0), aborting the run. The
    # defaults-on-failure branch is covered below by making import_module raise.
    def test_read_config_reads_bc_knobs(self):
        bc = types.ModuleType("bobert_companion")
        bc.ANTICIPATION_ENABLED = False
        bc.ANTICIPATION_COOLDOWN_MINUTES = 45
        with inject_modules(bobert_companion=bc):
            cfg = self.mod._read_config()
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["cooldown"], 45)

    def test_read_config_import_failure_uses_defaults(self):
        # importlib.import_module raising is the bc=None branch (lines 156-157).
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            cfg = self.mod._read_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["cooldown"], 20)

    # ── _load_state / _save_state round-trip + edges ─────────────────────
    def test_load_state_missing_file(self):
        self.assertEqual(self.mod._load_state(), {})

    def test_save_then_load_round_trip(self):
        self.mod._save_state({"last_proactive_at": 123.0, "last_trigger": "pattern"})
        self.assertTrue(os.path.exists(self.state_file))
        loaded = self.mod._load_state()
        self.assertEqual(loaded["last_trigger"], "pattern")
        self.assertEqual(loaded["last_proactive_at"], 123.0)

    def test_load_state_corrupt_json(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write("{ not valid json")
        self.assertEqual(self.mod._load_state(), {})

    def test_load_state_non_dict_payload(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(self.mod._load_state(), {})

    def test_save_state_swallows_mkstemp_failure(self):
        # mkstemp raising drives the outer except (lines 187-188) — no raise.
        with mock.patch.object(self.mod.tempfile, "mkstemp",
                               side_effect=OSError("disk full")):
            self.mod._save_state({"x": 1})   # must not raise
        # Nothing got written.
        self.assertFalse(os.path.exists(self.state_file))

    def test_save_state_replace_failure_unlinks_tmp(self):
        # mkstemp succeeds but os.replace fails → inner except unlinks the temp
        # file and re-raises into the outer except (lines 183-186). The temp
        # file must not be left behind in the (redirected) project dir.
        before = set(os.listdir(self.tmp))
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("rename denied")):
            self.mod._save_state({"y": 2})   # must not raise
        after = set(os.listdir(self.tmp))
        self.assertEqual(before, after)       # tmp cleaned up
        self.assertFalse(os.path.exists(self.state_file))

    def test_save_state_replace_and_unlink_both_fail(self):
        # Both os.replace AND the cleanup os.unlink raise → the innermost
        # ``except: pass`` (line 185) runs, then re-raise into the outer except.
        # Must still not propagate out of _save_state.
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("rename denied")), \
             mock.patch.object(self.mod.os, "unlink",
                               side_effect=OSError("unlink denied")):
            self.mod._save_state({"z": 3})   # must not raise
        self.assertFalse(os.path.exists(self.state_file))


# ─── speech queue ─────────────────────────────────────────────────────────
class AnticipationDwellTriggerTests(_EngineTestBase):
    def setUp(self):
        super().setUp()
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = ""
            self.mod._dwell_state["started_at"] = 0.0
            self.mod._dwell_state["last_seen"] = 0.0
        # Reset dwell globals after the test too, so a leaked record can't
        # affect another class's _current_dwell_seconds().
        self.addCleanup(self._reset_dwell)

    def _reset_dwell(self):
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = ""
            self.mod._dwell_state["started_at"] = 0.0
            self.mod._dwell_state["last_seen"] = 0.0

    def test_update_dwell_starts_and_continues(self):
        self.mod._update_dwell("model.scad - OpenSCAD")
        win, dwell = self.mod._current_dwell_seconds()
        self.assertEqual(win, "model.scad - OpenSCAD")
        self.assertGreaterEqual(dwell, 0.0)
        # Same window again keeps started_at (dwell keeps accruing).
        start_before = self.mod._dwell_state["started_at"]
        self.mod._update_dwell("model.scad - OpenSCAD")
        self.assertEqual(self.mod._dwell_state["started_at"], start_before)

    def test_update_dwell_resets_on_window_change(self):
        self.mod._update_dwell("A - Blender")
        first_start = self.mod._dwell_state["started_at"]
        self.mod._update_dwell("B - VS Code")
        self.assertEqual(self.mod._dwell_state["window"], "B - VS Code")
        # New window → started_at refreshed (>= previous).
        self.assertGreaterEqual(self.mod._dwell_state["started_at"], first_start)

    def test_current_dwell_seconds_empty_when_unset(self):
        win, dwell = self.mod._current_dwell_seconds()
        self.assertEqual(win, "")
        self.assertEqual(dwell, 0.0)

    def test_try_long_dwell_requires_threshold(self):
        # Productivity window but only a few seconds of dwell → no line.
        self.mod._update_dwell("model.scad - OpenSCAD")
        line, key = self.mod._try_long_dwell({})
        self.assertEqual(line, "")
        self.assertEqual(key, "")

    def test_try_long_dwell_fires_after_long_session(self):
        # Force a 3-hour dwell on a productivity window.
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = "model.scad - OpenSCAD"
            self.mod._dwell_state["started_at"] = time.time() - 3 * 3600
            self.mod._dwell_state["last_seen"] = time.time()
        line, key = self.mod._try_long_dwell({})
        self.assertIn("OpenSCAD", line)
        self.assertTrue(key.startswith("dwell:"))

    def test_try_long_dwell_respects_repeat_gap(self):
        app = "OpenSCAD"
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = "model.scad - OpenSCAD"
            self.mod._dwell_state["started_at"] = time.time() - 3 * 3600
            self.mod._dwell_state["last_seen"] = time.time()
        # Remarked on this app moments ago → suppressed by LONG_DWELL_REPEAT_GAP.
        state = {"last_dwell_remark_at": {app: time.time() - 60}}
        line, key = self.mod._try_long_dwell(state)
        self.assertEqual(line, "")

    def test_try_long_dwell_fires_after_repeat_gap_expires(self):
        # Same app, but the last remark was long enough ago → fires again.
        app = "OpenSCAD"
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = "model.scad - OpenSCAD"
            self.mod._dwell_state["started_at"] = time.time() - 3 * 3600
            self.mod._dwell_state["last_seen"] = time.time()
        state = {"last_dwell_remark_at": {app: time.time() - (3 * 3600)}}
        line, key = self.mod._try_long_dwell(state)
        self.assertIn("OpenSCAD", line)
        self.assertEqual(key, "dwell:OpenSCAD")

    def test_try_long_dwell_ignores_non_productivity(self):
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = "Solitaire"
            self.mod._dwell_state["started_at"] = time.time() - 3 * 3600
            self.mod._dwell_state["last_seen"] = time.time()
        line, key = self.mod._try_long_dwell({})
        self.assertEqual(line, "")

    def test_try_long_dwell_empty_window(self):
        # No focused window at all → immediate empty return.
        line, key = self.mod._try_long_dwell({})
        self.assertEqual((line, key), ("", ""))


class AnticipationLateHourTests(_EngineTestBase):
    def test_late_hour_active_fires_when_recently_spoke(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(23, 30)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=120.0):
            line = self.mod._try_late_hour_active()
        self.assertIn("stretch", line.lower())
        self.assertIn("11:30 PM", line)

    def test_late_hour_active_fires_early_morning(self):
        # The other side of the wrap-around window (before 7am).
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2, 15)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=60.0):
            line = self.mod._try_late_hour_active()
        self.assertIn("2:15 AM", line)

    def test_late_hour_active_silent_when_idle(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2, 0)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=99999.0):
            self.assertEqual(self.mod._try_late_hour_active(), "")

    def test_late_hour_active_silent_when_age_none(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(23, 30)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=None):
            self.assertEqual(self.mod._try_late_hour_active(), "")

    def test_late_hour_active_silent_during_day(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(14, 0)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=60.0):
            self.assertEqual(self.mod._try_late_hour_active(), "")

    def test_should_skip_late_night_no_activity_skips(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(3, 0)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=None):
            self.assertTrue(self.mod._should_skip_late_night())

    def test_should_skip_late_night_idle_skips(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(4, 0)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=99999.0):
            self.assertTrue(self.mod._should_skip_late_night())

    def test_should_skip_late_night_false_during_day(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(12, 0)):
            self.assertFalse(self.mod._should_skip_late_night())

    def test_should_skip_late_night_recent_speech_does_not_skip(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(23, 30)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=60.0):
            self.assertFalse(self.mod._should_skip_late_night())


# ─── environment probes via injected fake modules ─────────────────────────
class AnticipationEnvironmentTests(_EngineTestBase):
    # ── _all_window_titles ───────────────────────────────────────────────
    def test_all_window_titles_collects_nonblank(self):
        gw = _fake_gw(all_titles=["  ", "Notepad", "", "Outlook"])
        with inject_modules(pygetwindow=gw):
            out = self.mod._all_window_titles()
        self.assertEqual(out, ["Notepad", "Outlook"])

    def test_all_window_titles_no_pygetwindow(self):
        with block_import("pygetwindow"):
            self.assertEqual(self.mod._all_window_titles(), [])

    def test_all_window_titles_enum_raises(self):
        gw = _fake_gw(raise_all=True)
        with inject_modules(pygetwindow=gw):
            self.assertEqual(self.mod._all_window_titles(), [])

    # ── _focused_window_title ────────────────────────────────────────────
    def test_focused_window_title_returns_active(self):
        gw = _fake_gw(active_title="  part.f3d - Autodesk Fusion 360  ")
        with inject_modules(pygetwindow=gw):
            self.assertEqual(self.mod._focused_window_title(),
                             "part.f3d - Autodesk Fusion 360")

    def test_focused_window_title_none_active(self):
        gw = _fake_gw(active_title=None)
        with inject_modules(pygetwindow=gw):
            self.assertEqual(self.mod._focused_window_title(), "")

    def test_focused_window_title_raises(self):
        gw = _fake_gw(raise_active=True)
        with inject_modules(pygetwindow=gw):
            self.assertEqual(self.mod._focused_window_title(), "")

    def test_focused_window_title_no_pygetwindow(self):
        with block_import("pygetwindow"):
            self.assertEqual(self.mod._focused_window_title(), "")

    # ── _is_in_call real path (loop body line 228) ───────────────────────
    def test_is_in_call_real_window_hit(self):
        gw = _fake_gw(all_titles=["Zoom Meeting", "Notepad"])
        with inject_modules(pygetwindow=gw):
            self.assertTrue(self.mod._is_in_call())

    # ── _is_sleep_or_standby with a present bc ───────────────────────────
    def test_sleep_mode_true(self):
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [True]
        bc._standby_mode = [False]
        with inject_modules(bobert_companion=bc):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_standby_mode_true(self):
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [False]
        bc._standby_mode = [True]
        with inject_modules(bobert_companion=bc):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_sleep_standby_both_false(self):
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [False]
        bc._standby_mode = [False]
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_sleep_standby_attr_missing_returns_false(self):
        # bc present but without the mode lists → getattr raises → except False.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._is_sleep_or_standby())

    # ── _user_at_desk via fake face_tracker ──────────────────────────────
    def _face_mod(self, snap=None, raises=False, no_snap_func=False):
        mod = types.ModuleType("skill_face_tracker")
        if no_snap_func:
            return mod
        if raises:
            mod._snapshot_state = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        else:
            mod._snapshot_state = lambda: snap
        return mod

    def test_user_at_desk_present_monitor(self):
        face = self._face_mod({"last_sample_at": 111.0, "current_monitor": "left"})
        with inject_modules(skill_face_tracker=face):
            self.assertTrue(self.mod._user_at_desk())

    def test_user_at_desk_away(self):
        face = self._face_mod({"last_sample_at": 111.0, "current_monitor": "away"})
        with inject_modules(skill_face_tracker=face):
            self.assertFalse(self.mod._user_at_desk())

    def test_user_at_desk_unknown_monitor(self):
        face = self._face_mod({"last_sample_at": 111.0, "current_monitor": "elsewhere"})
        with inject_modules(skill_face_tracker=face):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_no_sample(self):
        face = self._face_mod({"last_sample_at": 0, "current_monitor": "left"})
        with inject_modules(skill_face_tracker=face):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_snapshot_raises(self):
        face = self._face_mod(raises=True)
        with inject_modules(skill_face_tracker=face):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_no_snapshot_func(self):
        face = self._face_mod(no_snap_func=True)
        with inject_modules(skill_face_tracker=face):
            self.assertIsNone(self.mod._user_at_desk())

    # ── _last_speech_age_seconds with a present bc ───────────────────────
    def test_last_speech_age_computes(self):
        bc = types.ModuleType("bobert_companion")
        bc.last_speech_time = time.time() - 50.0
        with inject_modules(bobert_companion=bc):
            age = self.mod._last_speech_age_seconds()
        self.assertGreaterEqual(age, 49.0)
        self.assertLessEqual(age, 60.0)

    def test_last_speech_age_non_numeric(self):
        bc = types.ModuleType("bobert_companion")
        bc.last_speech_time = "not-a-number"
        with inject_modules(bobert_companion=bc):
            self.assertIsNone(self.mod._last_speech_age_seconds())

    # ── _try_pattern_offer bridge ────────────────────────────────────────
    def test_try_pattern_offer_returns_line(self):
        pm = types.ModuleType("memory")
        pm.maybe_pattern_offer = lambda: "Shall I start your Monday routine, sir?"
        with inject_modules(memory=pm):
            self.assertIn("Monday", self.mod._try_pattern_offer())

    def test_try_pattern_offer_uses_real_memory_module_name(self):
        # Regression: the engine used to import the non-existent module
        # "pattern_memory" (only ever a local alias in bobert_companion),
        # so the pattern trigger could never fire. It must import the real
        # top-level "memory" module — and NOT need "pattern_memory".
        pm = types.ModuleType("memory")
        pm.maybe_pattern_offer = lambda: "Time for the Tuesday sweep, sir?"
        with inject_modules(memory=pm, pattern_memory=None):
            self.assertEqual(self.mod._try_pattern_offer(),
                             "Time for the Tuesday sweep, sir?")

    def test_try_pattern_offer_empty_when_none(self):
        pm = types.ModuleType("memory")
        pm.maybe_pattern_offer = lambda: None
        with inject_modules(memory=pm):
            self.assertEqual(self.mod._try_pattern_offer(), "")

    def test_try_pattern_offer_import_failure(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no memory module")):
            self.assertEqual(self.mod._try_pattern_offer(), "")

    def test_try_pattern_offer_callable_raises(self):
        pm = types.ModuleType("memory")
        def _boom():
            raise RuntimeError("offer exploded")
        pm.maybe_pattern_offer = _boom
        with inject_modules(memory=pm):
            self.assertEqual(self.mod._try_pattern_offer(), "")


# ─── _format_status / action + speech queue ───────────────────────────────
class AnticipationStatusAndQueueTests(_EngineTestBase):
    def setUp(self):
        super().setUp()
        self.addCleanup(self._reset_dwell)

    def _reset_dwell(self):
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = ""
            self.mod._dwell_state["started_at"] = 0.0
            self.mod._dwell_state["last_seen"] = 0.0

    def test_status_no_fires_yet(self):
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            out = self.actions["anticipation_status"]("")
        self.assertIn("no fires yet", out.lower())
        self.assertTrue(out.startswith("Anticipation engine"))

    def test_status_reports_last_fire(self):
        state = {"last_proactive_at": time.time() - 300}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["anticipation_status"]("")
        self.assertIn("last fire", out.lower())
        self.assertIn("until next eligible", out.lower())

    def test_status_recent_fire_seconds_phrasing(self):
        # A fire < 60s ago renders "N seconds ago" (not the h/m formatter).
        state = {"last_proactive_at": time.time() - 30}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["anticipation_status"]("")
        self.assertIn("seconds ago", out)

    def test_status_cooldown_already_elapsed(self):
        # Last fire well beyond the cooldown → no "until next eligible" line
        # (the remaining>0 branch is False — covers 534->538).
        state = {"last_proactive_at": time.time() - 10 * 3600}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["anticipation_status"]("")
        self.assertIn("last fire", out.lower())
        self.assertNotIn("until next eligible", out.lower())

    def test_status_disabled_in_config(self):
        bc = types.ModuleType("bobert_companion")
        bc.ANTICIPATION_ENABLED = False
        bc.ANTICIPATION_COOLDOWN_MINUTES = 20
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_load_state", return_value={}):
            out = self.actions["anticipation_status"]("")
        self.assertIn("disabled in config", out)

    def test_status_includes_dwell_call_and_sleep(self):
        # Long dwell on a productivity window + in-call + sleep all surfaced.
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = "model.scad - OpenSCAD"
            self.mod._dwell_state["started_at"] = time.time() - 3600
            self.mod._dwell_state["last_seen"] = time.time()
        with mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_in_call", return_value=True), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=True):
            out = self.actions["anticipation_status"]("")
        self.assertIn("focused window: OpenSCAD", out)
        self.assertIn("in a call", out.lower())
        self.assertIn("sleep/standby active", out)

    def test_status_action_swallows_exception(self):
        # The action wrapper catches and reports formatter errors.
        with mock.patch.object(self.mod, "_format_status",
                               side_effect=RuntimeError("kaboom")):
            out = self.actions["anticipation_status"]("")
        self.assertIn("anticipation status failed", out)
        self.assertIn("kaboom", out)

    def test_enqueue_speech_writes_to_temp_queue(self):
        fd, qp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with mock.patch.object(self.mod, "_SPEECH_QUEUE", qp):
                self.mod._enqueue_speech("Sir, a brief stretch would not go amiss.")
            with open(qp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertIn("stretch", data[0]["message"])
        finally:
            os.unlink(qp)

    def test_enqueue_speech_appends_to_existing(self):
        fd, qp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with open(qp, "w", encoding="utf-8") as f:
                json.dump([{"ts": 1.0, "message": "earlier"}], f)
            with mock.patch.object(self.mod, "_SPEECH_QUEUE", qp):
                self.mod._enqueue_speech("later line")
            with open(qp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["message"], "earlier")
            self.assertEqual(data[1]["message"], "later line")
        finally:
            os.unlink(qp)

    def test_enqueue_speech_corrupt_existing_resets(self):
        fd, qp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with open(qp, "w", encoding="utf-8") as f:
                f.write("{ corrupt")
            with mock.patch.object(self.mod, "_SPEECH_QUEUE", qp):
                self.mod._enqueue_speech("fresh start")
            with open(qp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["message"], "fresh start")
        finally:
            os.unlink(qp)

    def test_enqueue_speech_write_failure_is_swallowed(self):
        # _atomic_write_json raising must be caught (lines 147-148), no raise.
        with mock.patch.object(self.mod, "_SPEECH_QUEUE", "X:/nope/queue.json"), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only fs")):
            self.mod._enqueue_speech("dropped line")   # must not raise

    def test_enqueue_speech_routes_through_proactive_announce(self):
        # #12: when bobert_companion exposes proactive_announce, the line must go
        # through that serialized / focus-gated writer and NOT a bare local queue
        # write — so it can't race the other co-writers or leak past focus/DND.
        calls = []
        bc = types.ModuleType("bobert_companion")

        def _announce(message, source="skill", **kw):
            calls.append((message, source))
            return True
        bc.proactive_announce = _announce
        fd, qp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(qp)   # no file present, so any local fallback write would create it
        try:
            with inject_modules(bobert_companion=bc), \
                 mock.patch.object(self.mod, "_SPEECH_QUEUE", qp):
                self.mod._enqueue_speech("Shall I begin, sir?")
            self.assertEqual(calls, [("Shall I begin, sir?", "anticipation")])
            # Announce handled it → no local fallback file written.
            self.assertFalse(os.path.exists(qp))
        finally:
            if os.path.exists(qp):
                os.unlink(qp)

    def test_enqueue_speech_falls_back_when_announce_returns_falsy(self):
        # #12: a falsy proactive_announce (e.g. transiently unavailable) must NOT
        # be treated as handled — the line falls back to the local atomic write so
        # it is never silently lost.
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = lambda message, source="skill", **kw: False
        fd, qp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with inject_modules(bobert_companion=bc), \
                 mock.patch.object(self.mod, "_SPEECH_QUEUE", qp):
                self.mod._enqueue_speech("fallback line")
            with open(qp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["message"], "fallback line")
        finally:
            os.unlink(qp)


# ─── full scheduler-loop drive ────────────────────────────────────────────
class AnticipationSchedulerLoopTests(_EngineTestBase):
    """Drive _scheduler_loop for a single poll iteration. time.sleep is patched
    to no-op the INITIAL_DELAY + first poll-end sleep then raise _StopLoop, so
    the otherwise-infinite while loop exits deterministically. random and the
    environment gates are all stubbed; nothing real runs."""

    def setUp(self):
        super().setUp()
        self.addCleanup(self._reset_dwell)
        # The inner-exception path logs via logging.exception — swallow it so the
        # deliberate failure tests don't spam the runner. stdout is already
        # redirected by the base setUp.
        mock.patch.object(self.mod.logging, "exception", lambda *a, **k: None).start()
        self.addCleanup(mock.patch.stopall)

    def _reset_dwell(self):
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = ""
            self.mod._dwell_state["started_at"] = 0.0
            self.mod._dwell_state["last_seen"] = 0.0

    @contextlib.contextmanager
    def _loop_env(self, *, enabled=True, in_call=False, sleep=False,
                  at_desk=None, skip_night=False, sleeps_before_stop=2):
        """Patch every gate the loop consults. ``sleeps_before_stop`` counts the
        INITIAL_DELAY sleep + intermediate sleeps before _StopLoop fires."""
        cfg = {"enabled": enabled, "cooldown": 20}
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=_sleep_after(sleeps_before_stop)), \
             mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_focused_window_title", return_value=""), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=sleep), \
             mock.patch.object(self.mod, "_is_in_call", return_value=in_call), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=at_desk), \
             mock.patch.object(self.mod, "_should_skip_late_night", return_value=skip_night):
            yield

    def test_loop_disabled_skips_everything(self):
        enqueue = mock.MagicMock()
        with self._loop_env(enabled=False), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_gate_in_call(self):
        enqueue = mock.MagicMock()
        with self._loop_env(in_call=True), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_gate_sleep(self):
        enqueue = mock.MagicMock()
        with self._loop_env(sleep=True), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_gate_away(self):
        enqueue = mock.MagicMock()
        with self._loop_env(at_desk=False), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_gate_late_night(self):
        enqueue = mock.MagicMock()
        with self._loop_env(skip_night=True), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_cooldown_blocks_fire(self):
        enqueue = mock.MagicMock()
        recent = {"last_proactive_at": time.time() - 60}   # within 20-min cooldown
        with self._loop_env(), \
             mock.patch.object(self.mod, "_load_state", return_value=recent), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_no_trigger_no_fire(self):
        enqueue = mock.MagicMock()
        with self._loop_env(), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_try_pattern_offer", return_value=""), \
             mock.patch.object(self.mod, "_try_long_dwell", return_value=("", "")), \
             mock.patch.object(self.mod, "_try_late_hour_active", return_value=""), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_pattern_offer_fires_and_persists(self):
        # Pattern offers bypass the probability gate and persist state.
        enqueue = mock.MagicMock()
        saved = {}
        with self._loop_env(), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_try_pattern_offer",
                               return_value="Your Monday routine, sir?"), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue), \
             mock.patch.object(self.mod, "_save_state",
                               side_effect=lambda s: saved.update(s)):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_called_once()
        self.assertIn("Monday", enqueue.call_args[0][0])
        self.assertEqual(saved["last_trigger"], "pattern")
        self.assertIn("last_proactive_at", saved)

    def test_loop_long_dwell_fires_when_probability_passes(self):
        # Non-pattern trigger: random < FIRE_PROBABILITY → fires; records the
        # per-app dwell-remark timestamp.
        enqueue = mock.MagicMock()
        saved = {}
        with self._loop_env(), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_try_pattern_offer", return_value=""), \
             mock.patch.object(self.mod, "_try_long_dwell",
                               return_value=("In Blender 3h, sir?", "dwell:Blender")), \
             mock.patch.object(self.mod.random, "random", return_value=0.0), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue), \
             mock.patch.object(self.mod, "_save_state",
                               side_effect=lambda s: saved.update(s)):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_called_once()
        self.assertEqual(saved["last_trigger"], "long_dwell")
        self.assertIn("Blender", saved["last_dwell_remark_at"])

    def test_loop_long_dwell_suppressed_by_probability(self):
        # random > FIRE_PROBABILITY → silent this tick even with a match.
        enqueue = mock.MagicMock()
        with self._loop_env(), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_try_pattern_offer", return_value=""), \
             mock.patch.object(self.mod, "_try_long_dwell",
                               return_value=("dwell line", "dwell:Blender")), \
             mock.patch.object(self.mod.random, "random", return_value=0.99), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_late_hour_trigger_path(self):
        # No pattern, no dwell, but late-hour line present → fires on roll pass.
        enqueue = mock.MagicMock()
        saved = {}
        with self._loop_env(), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_try_pattern_offer", return_value=""), \
             mock.patch.object(self.mod, "_try_long_dwell", return_value=("", "")), \
             mock.patch.object(self.mod, "_try_late_hour_active",
                               return_value="It's late, sir. Stretch."), \
             mock.patch.object(self.mod.random, "random", return_value=0.0), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue), \
             mock.patch.object(self.mod, "_save_state",
                               side_effect=lambda s: saved.update(s)):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_called_once()
        self.assertEqual(saved["last_trigger"], "late_hour")

    def test_loop_inner_exception_is_logged_and_continues(self):
        # An exception mid-poll is caught by the loop's try/except, which logs,
        # sleeps, and `continue`s back to the top. We allow the INITIAL_DELAY
        # sleep + the in-except sleep (so `continue` on line 515 runs), then the
        # in-except sleep on the SECOND failing pass raises _StopLoop.
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=_sleep_after(2)), \
             mock.patch.object(self.mod, "_read_config",
                               side_effect=RuntimeError("cfg blew up")):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()


# ─── register() ───────────────────────────────────────────────────────────
class AnticipationRegisterTests(_EngineTestBase):
    def test_register_disabled_no_thread(self):
        actions = {}
        bc = types.ModuleType("bobert_companion")
        bc.ANTICIPATION_ENABLED = False
        bc.ANTICIPATION_COOLDOWN_MINUTES = 20
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)
        self.assertIn("anticipation_status", actions)
        Thread.assert_not_called()

    def test_register_enabled_starts_thread(self):
        actions = {}
        bc = types.ModuleType("bobert_companion")
        bc.ANTICIPATION_ENABLED = True
        bc.ANTICIPATION_COOLDOWN_MINUTES = 20
        made = mock.MagicMock()
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.threading, "Thread", return_value=made) as Thread, \
             mock.patch.object(self.mod.threading, "enumerate", return_value=[]):
            self.mod.register(actions)
        Thread.assert_called_once()
        made.start.assert_called_once()
        self.assertIn("anticipation_status", actions)

    def test_register_skips_when_loop_already_running(self):
        actions = {}
        bc = types.ModuleType("bobert_companion")
        bc.ANTICIPATION_ENABLED = True
        bc.ANTICIPATION_COOLDOWN_MINUTES = 20
        existing = types.SimpleNamespace(
            name="anticipation-scheduler", is_alive=lambda: True)
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.threading, "enumerate",
                               return_value=[existing]), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)
        Thread.assert_not_called()

    def test_registered_action_is_callable(self):
        actions = {}
        bc = types.ModuleType("bobert_companion")
        bc.ANTICIPATION_ENABLED = False
        bc.ANTICIPATION_COOLDOWN_MINUTES = 20
        with inject_modules(bobert_companion=bc):
            self.mod.register(actions)
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            out = actions["anticipation_status"]("ignored-arg")
        self.assertTrue(out.startswith("Anticipation engine"))


class AnticipationImportGuardTests(unittest.TestCase):
    def test_path_bootstrap_inserts_project_root(self):
        mod, _ = load_skill_isolated("anticipation_engine")
        path = mod.__file__
        proj = os.path.dirname(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("anticipation_reexec", path)
        m = importlib.util.module_from_spec(spec)
        m.skill_utils = {}
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path
                           if os.path.abspath(p) != os.path.abspath(proj)]
            spec.loader.exec_module(m)
            self.assertIn(m._PROJECT_DIR, sys.path)
        finally:
            sys.path[:] = saved


if __name__ == "__main__":
    unittest.main()
