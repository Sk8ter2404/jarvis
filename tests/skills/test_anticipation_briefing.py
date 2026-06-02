"""Logic tests for skills/anticipation_briefing.py.

Covers config clamping + knob reads, the weekday bucket, the forward-only
minute delta, the spoken-line composer across every action branch (music /
Teams / morning / evening / generic offer, with and without lead), prediction
selection (precise-with-lead before broad-now, confidence floor, tolerance-band
"happening now", bucket/window mismatches — all driven off a frozen clock), the
throttle-aware _next_eligible pick + _mark_fired, persistent state
load/save/prune (round-trip + corruption + write-failure), the snapshot reader
(injected pattern_learning, file fallback, error paths), the hard gates
(sleep/standby, in-call window scan, face-tracker at-desk), the speech enqueue
(proactive_announce path + direct-file fallback + failure), the full scheduler
loop across every gate, both registered actions, register()'s health-check, and
the __main__ smoke block.

Isolation: a benign fake bobert_companion is seeded into sys.modules before the
skill loads (so register()/_read_config don't import the real ~14K-line monolith
— whose early-boot singleton lock calls sys.exit(0)) and removed in tearDown via
a save/restore. The scheduler thread is neutered by the harness. Everything is
offline + deterministic: datetime/clock frozen where wall-clock-sensitive, no
real network/LLM/threads/sleep, generic (non-personal) event fixtures.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


# ─── fake-module injection ────────────────────────────────────────────────
@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules; pass ``name=None`` to
    force-remove one for the block. Restores prior state (including absence) on
    exit so tests stay isolated."""
    saved: dict[str, object] = {}
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


@contextlib.contextmanager
def block_import(*names):
    """Force a bare ``import <name>`` to raise ImportError inside the block even
    when the real dependency is installed (``_all_window_titles`` uses a bare
    ``import pygetwindow``). Detaches any already-imported target and patches
    ``__import__``; restores both on exit."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        if name.split(".")[0] in blocked or name in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    saved = {n: sys.modules.pop(n) for n in blocked if n in sys.modules}
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        sys.modules.update(saved)


def _benign_bc(**attrs):
    """A minimal fake ``bobert_companion``. By default exposes nothing, so the
    skill's getattr(..., default) config path is taken; pass attrs to pin knobs
    or a ``proactive_announce``."""
    bc = types.ModuleType("bobert_companion")
    for k, v in attrs.items():
        setattr(bc, k, v)
    return bc


def _struct(hour, minute, wday=0):
    """A time.struct_time for a fixed local time. wday 0 = Monday (weekday)."""
    # tm_yday is not consulted by the skill; 152 is fine for June 1.
    return time.struct_time((2026, 6, 1, hour, minute, 0, wday, 152, -1))


class _BriefingTestBase(unittest.TestCase):
    """Load the skill in isolation WITH a benign fake bobert_companion seeded.

    register() (run at load by the harness) calls _read_config →
    importlib.import_module("bobert_companion"); with the real monolith absent
    that would execute it for real and its singleton lock calls sys.exit(0),
    killing the run. Seeding a fake makes import_module return it. stdout is
    redirected after load because several helpers print on failure paths."""
    _BC_ATTRS: dict = {}

    def setUp(self):
        self._seeded = {}
        for name, obj in {"bobert_companion": _benign_bc(**self._BC_ATTRS)}.items():
            self._seeded[name] = sys.modules.get(name, _SENTINEL)
            sys.modules[name] = obj
        self.addCleanup(self._restore_seeded)
        self.mod, self.actions = load_skill_isolated("anticipation_briefing")
        _out = contextlib.redirect_stdout(io.StringIO())
        _out.__enter__()
        self.addCleanup(_out.__exit__, None, None, None)

    def _restore_seeded(self):
        for name, prev in self._seeded.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev

    def _cfg(self, **over):
        base = {"enabled": True, "poll_min": 5, "lead_min": 15, "conf_floor": 0.5}
        base.update(over)
        return base


class AnticipationBriefingTests(_BriefingTestBase):
    # ── _clamp (pure) ────────────────────────────────────────────────────
    def test_clamp(self):
        c = self.mod._clamp
        self.assertEqual(c(5, 1, 60), 5)
        self.assertEqual(c(0, 1, 60), 1)       # below floor
        self.assertEqual(c(99, 1, 60), 60)     # above ceiling
        self.assertEqual(c("nan", 1, 60), 1)   # uncastable → floor
        self.assertEqual(c(0.3, 0.0, 1.0), 0.3)  # float passthrough

    # ── _bucket_for_weekday / _minutes_until (pure) ──────────────────────
    def test_bucket_for_weekday(self):
        self.assertEqual(self.mod._bucket_for_weekday(0), "weekday")
        self.assertEqual(self.mod._bucket_for_weekday(4), "weekday")
        self.assertEqual(self.mod._bucket_for_weekday(5), "weekend")
        self.assertEqual(self.mod._bucket_for_weekday(6), "weekend")

    def test_minutes_until_forward_only(self):
        self.assertEqual(self.mod._minutes_until(600, 540), 60)   # 60 min ahead
        self.assertEqual(self.mod._minutes_until(540, 540), 0)    # now
        self.assertEqual(self.mod._minutes_until(500, 540), 10_000)  # past → sentinel

    # ── _titlecase (pure) ────────────────────────────────────────────────
    def test_titlecase(self):
        self.assertEqual(self.mod._titlecase("sam industries"), "Sam Industries")
        # small words stay lowercase unless first.
        self.assertEqual(self.mod._titlecase("king of the hill"), "King of the Hill")
        self.assertEqual(self.mod._titlecase(""), "")

    # ── _compose_briefing_line branches ──────────────────────────────────
    def test_compose_music_with_arg(self):
        line = self.mod._compose_briefing_line(
            {"action": "play_music", "common_arg": "the beatles"})
        self.assertIn("queue your usual", line.lower())
        self.assertIn("The Beatles", line)

    def test_compose_music_no_arg(self):
        line = self.mod._compose_briefing_line({"action": "resume_music", "common_arg": ""})
        self.assertIn("usual playlist", line.lower())

    def test_compose_music_other_aliases(self):
        for act in ("youtube_play", "spotify", "apple_music"):
            line = self.mod._compose_briefing_line({"action": act, "common_arg": "queen"})
            self.assertIn("Queen", line)

    def test_compose_teams_with_name_and_lead(self):
        line = self.mod._compose_briefing_line(
            {"action": "check_teams", "common_arg": "sam", "__lead_minutes": 12})
        self.assertIn("Sam sync in 12 minutes", line)
        self.assertIn("last conversation", line.lower())

    def test_compose_teams_with_name_singular_minute(self):
        line = self.mod._compose_briefing_line(
            {"action": "check_teams", "common_arg": "sam", "__lead_minutes": 1})
        self.assertIn("Sam sync in 1 minute,", line)   # no trailing 's'

    def test_compose_teams_name_no_lead(self):
        # arg present but lead 0 → the "is upon us" phrasing (line 382-387).
        line = self.mod._compose_briefing_line(
            {"action": "check_teams", "common_arg": "alex", "__lead_minutes": 0})
        self.assertIn("Your Alex sync is upon us", line)

    def test_compose_teams_lead_no_name(self):
        # no arg but lead > 0 → generic Teams lead phrasing (line 388-392).
        line = self.mod._compose_briefing_line(
            {"action": "check_teams", "__lead_minutes": 8})
        self.assertIn("Teams sync in 8 minutes", line)
        self.assertIn("most recent thread", line.lower())

    def test_compose_teams_no_arg_no_lead(self):
        line = self.mod._compose_briefing_line({"action": "check_teams"})
        self.assertIn("check Teams about now", line)

    def test_compose_morning_briefing_with_lead(self):
        line = self.mod._compose_briefing_line(
            {"action": "morning_briefing", "__lead_minutes": 5})
        self.assertIn("Morning briefing", line)
        self.assertIn("5 minute", line)

    def test_compose_morning_briefing_no_lead(self):
        line = self.mod._compose_briefing_line({"action": "morning_briefing"})
        self.assertEqual(line, "Shall I deliver the morning briefing, sir?")

    def test_compose_evening_briefing_with_lead(self):
        line = self.mod._compose_briefing_line(
            {"action": "evening_briefing", "__lead_minutes": 1})
        self.assertIn("Evening briefing", line)
        self.assertIn("1 minute,", line)   # singular

    def test_compose_evening_briefing_no_lead(self):
        line = self.mod._compose_briefing_line({"action": "evening_briefing"})
        self.assertEqual(line, "Shall I deliver the evening briefing, sir?")

    def test_compose_generic_offer_with_lead(self):
        line = self.mod._compose_briefing_line(
            {"action": "something_else", "offer": "Shall I open your inbox, sir?",
             "__lead_minutes": 3})
        self.assertIn("In about 3 minutes", line)
        self.assertIn("Shall I open your inbox", line)

    def test_compose_generic_offer_no_lead(self):
        # offer present, no lead → bare offer returned (line 418).
        line = self.mod._compose_briefing_line(
            {"action": "x", "offer": "Shall I dim the lights, sir?"})
        self.assertEqual(line, "Shall I dim the lights, sir?")

    def test_compose_empty_when_nothing(self):
        self.assertEqual(self.mod._compose_briefing_line({"action": "x"}), "")

    # ── _prune_state ─────────────────────────────────────────────────────
    def test_prune_state_drops_old(self):
        old_day = time.strftime("%Y-%m-%d", time.localtime(time.time() - 200 * 86400))
        today = time.strftime("%Y-%m-%d", time.localtime())
        state = {"stale": old_day, "fresh": today}
        self.mod._prune_state(state)
        self.assertNotIn("stale", state)
        self.assertIn("fresh", state)

    def test_prune_state_ignores_non_string_values(self):
        # A non-str value must be left untouched (the isinstance guard).
        state = {"weird": 12345, "fresh": time.strftime("%Y-%m-%d", time.localtime())}
        self.mod._prune_state(state)
        self.assertIn("weird", state)
        self.assertIn("fresh", state)

    # ── hard gates ───────────────────────────────────────────────────────
    def test_gate_in_call_detects_window_hint(self):
        with mock.patch.object(self.mod, "_all_window_titles",
                               return_value=["Project | Microsoft Teams Meeting"]):
            self.assertTrue(self.mod._is_in_call())

    def test_gate_not_in_call(self):
        with mock.patch.object(self.mod, "_all_window_titles", return_value=["Notepad"]):
            self.assertFalse(self.mod._is_in_call())

    def test_gate_in_call_no_windows(self):
        with mock.patch.object(self.mod, "_all_window_titles", return_value=[]):
            self.assertFalse(self.mod._is_in_call())


# ─── config knob reads ────────────────────────────────────────────────────
class BriefingConfigTests(_BriefingTestBase):
    def test_read_config_reads_and_clamps_knobs(self):
        bc = _benign_bc(
            ANTICIPATION_BRIEFING_ENABLED=True,
            ANTICIPATION_BRIEFING_POLL_MINUTES=999,     # clamps to 60
            ANTICIPATION_BRIEFING_LEAD_MINUTES=0,       # clamps to 1
            ANTICIPATION_BRIEFING_CONFIDENCE_MIN=2.0,   # clamps to 1.0
        )
        with inject_modules(bobert_companion=bc):
            cfg = self.mod._read_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["poll_min"], 60)
        self.assertEqual(cfg["lead_min"], 1)
        self.assertEqual(cfg["conf_floor"], 1.0)

    def test_read_config_disabled_knob(self):
        bc = _benign_bc(ANTICIPATION_BRIEFING_ENABLED=False)
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._read_config()["enabled"])

    def test_read_config_defaults_when_import_fails(self):
        # import_module raising is the bc=None branch (lines 101-102) → defaults.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            cfg = self.mod._read_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["poll_min"], self.mod.DEFAULT_POLL_MINUTES)
        self.assertEqual(cfg["lead_min"], self.mod.DEFAULT_LEAD_MINUTES)
        self.assertEqual(cfg["conf_floor"], self.mod.DEFAULT_CONFIDENCE)


# ─── persistent state load/save/prune-on-save ─────────────────────────────
class BriefingStateTests(_BriefingTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="anticip_brief_")
        self.addCleanup(self._cleanup)
        self.state_file = os.path.join(self.tmp, "anticipation_briefing_state.json")
        self.mod._STATE_FILE = self.state_file
        self.mod._DATA_DIR = self.tmp

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

    def test_ensure_data_dir_swallows_failure(self):
        with mock.patch.object(self.mod.os, "makedirs",
                               side_effect=OSError("denied")):
            self.mod._ensure_data_dir()   # must not raise

    def test_load_state_missing(self):
        self.assertEqual(self.mod._load_state(), {})

    def test_save_then_load_round_trip(self):
        self.mod._save_state({"key:a": "2026-05-30"})
        self.assertTrue(os.path.exists(self.state_file))
        self.assertEqual(self.mod._load_state(), {"key:a": "2026-05-30"})

    def test_load_state_corrupt(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write("}{ not json")
        self.assertEqual(self.mod._load_state(), {})

    def test_load_state_non_dict(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump([1, 2], f)
        self.assertEqual(self.mod._load_state(), {})

    def test_save_state_swallows_mkstemp_failure(self):
        with mock.patch.object(self.mod.tempfile, "mkstemp",
                               side_effect=OSError("disk full")):
            self.mod._save_state({"x": "y"})   # must not raise
        self.assertFalse(os.path.exists(self.state_file))

    def test_save_state_replace_failure_unlinks_tmp(self):
        before = set(os.listdir(self.tmp))
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("rename denied")):
            self.mod._save_state({"x": "y"})   # must not raise
        self.assertEqual(before, set(os.listdir(self.tmp)))  # tmp cleaned up
        self.assertFalse(os.path.exists(self.state_file))

    def test_save_state_replace_and_unlink_both_fail(self):
        # innermost ``except Exception: pass`` around os.unlink (lines 145-146).
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("rename denied")), \
             mock.patch.object(self.mod.os, "unlink",
                               side_effect=OSError("unlink denied")):
            self.mod._save_state({"x": "y"})   # must not raise

    def test_prune_state_swallows_bad_input(self):
        # time.strftime raising drives the prune-level except (lines 163-164).
        with mock.patch.object(self.mod.time, "strftime",
                               side_effect=ValueError("bad fmt")):
            self.mod._prune_state({"a": "2026-01-01"})   # must not raise

    def test_mark_fired_writes_today_for_key(self):
        with mock.patch.object(self.mod.time, "strftime", return_value="2026-06-01"):
            self.mod._mark_fired({"key": "teams:0900"})
        self.assertEqual(self.mod._load_state().get("teams:0900"), "2026-06-01")

    def test_mark_fired_no_key_is_noop(self):
        self.mod._mark_fired({"key": ""})
        self.assertEqual(self.mod._load_state(), {})


# ─── snapshot reader ──────────────────────────────────────────────────────
class BriefingSnapshotTests(_BriefingTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="anticip_snap_")
        self.addCleanup(self._cleanup)
        self.mod._DATA_DIR = self.tmp
        self.snap_path = os.path.join(self.tmp, "usage_patterns_aggregated.json")

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

    def test_snapshot_from_injected_pattern_learning(self):
        pl = types.ModuleType("skill_pattern_learning")
        pl._load_aggregated = lambda: {"broad": [{"key": "b"}], "precise": []}
        with inject_modules(skill_pattern_learning=pl):
            snap = self.mod._load_snapshot()
        self.assertEqual(snap["broad"], [{"key": "b"}])

    def test_snapshot_pl_returns_non_dict_yields_empty(self):
        # When pattern_learning's _load_aggregated returns a non-dict, the
        # reader returns {} directly (it does NOT fall through to the file).
        pl = types.ModuleType("skill_pattern_learning")
        pl._load_aggregated = lambda: ["not", "a", "dict"]
        # A valid file exists, but it is NOT consulted on this path.
        with open(self.snap_path, "w", encoding="utf-8") as f:
            json.dump({"broad": [], "precise": [{"key": "p"}]}, f)
        with inject_modules(skill_pattern_learning=pl):
            snap = self.mod._load_snapshot()
        self.assertEqual(snap, {})

    def test_snapshot_pl_raises_then_file_fallback(self):
        pl = types.ModuleType("skill_pattern_learning")
        def _boom():
            raise RuntimeError("agg exploded")
        pl._load_aggregated = _boom
        with open(self.snap_path, "w", encoding="utf-8") as f:
            json.dump({"broad": [{"key": "fb"}], "precise": []}, f)
        with inject_modules(skill_pattern_learning=pl):
            snap = self.mod._load_snapshot()
        self.assertEqual(snap["broad"], [{"key": "fb"}])

    def test_snapshot_no_pl_no_file(self):
        # No pattern_learning module + no file → {}.
        with inject_modules(skill_pattern_learning=None), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError):
            self.assertEqual(self.mod._load_snapshot(), {})

    def test_snapshot_file_fallback_direct(self):
        with open(self.snap_path, "w", encoding="utf-8") as f:
            json.dump({"broad": [], "precise": [], "generated_at": 1.0}, f)
        with inject_modules(skill_pattern_learning=None), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError):
            snap = self.mod._load_snapshot()
        self.assertIn("generated_at", snap)

    def test_snapshot_file_corrupt(self):
        with open(self.snap_path, "w", encoding="utf-8") as f:
            f.write("{ corrupt")
        with inject_modules(skill_pattern_learning=None), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError):
            self.assertEqual(self.mod._load_snapshot(), {})

    def test_snapshot_file_non_dict(self):
        with open(self.snap_path, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        with inject_modules(skill_pattern_learning=None), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError):
            self.assertEqual(self.mod._load_snapshot(), {})


# ─── environment gate probes via injected fakes ───────────────────────────
class BriefingGateTests(_BriefingTestBase):
    # ── _is_sleep_or_standby ─────────────────────────────────────────────
    def test_sleep_mode_true(self):
        bc = _benign_bc()
        bc._sleep_mode = [True]
        bc._standby_mode = [False]
        with inject_modules(bobert_companion=bc):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_standby_mode_true(self):
        bc = _benign_bc()
        bc._sleep_mode = [False]
        bc._standby_mode = [True]
        with inject_modules(bobert_companion=bc):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_sleep_standby_false(self):
        bc = _benign_bc()
        bc._sleep_mode = [False]
        bc._standby_mode = [False]
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_sleep_standby_none_when_absent(self):
        # Uses sys.modules.get (not import) → popping to None is safe here.
        with inject_modules(bobert_companion=None):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_sleep_standby_attr_missing(self):
        bc = _benign_bc()    # no mode lists → getattr raises → except False
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._is_sleep_or_standby())

    # ── _all_window_titles / _is_in_call real path ───────────────────────
    def test_all_window_titles_collects(self):
        gw = types.ModuleType("pygetwindow")
        gw.getAllWindows = lambda: [
            types.SimpleNamespace(title="  "),
            types.SimpleNamespace(title="Outlook"),
            types.SimpleNamespace(title="Spotify"),
        ]
        with inject_modules(pygetwindow=gw):
            self.assertEqual(self.mod._all_window_titles(), ["Outlook", "Spotify"])

    def test_all_window_titles_enum_raises(self):
        gw = types.ModuleType("pygetwindow")
        def _boom():
            raise RuntimeError("enum boom")
        gw.getAllWindows = _boom
        with inject_modules(pygetwindow=gw):
            self.assertEqual(self.mod._all_window_titles(), [])

    def test_all_window_titles_no_pygetwindow(self):
        with block_import("pygetwindow"):
            self.assertEqual(self.mod._all_window_titles(), [])

    def test_is_in_call_real_window_hit(self):
        gw = types.ModuleType("pygetwindow")
        gw.getAllWindows = lambda: [types.SimpleNamespace(title="Zoom Meeting")]
        with inject_modules(pygetwindow=gw):
            self.assertTrue(self.mod._is_in_call())

    # ── _user_at_desk via fake face_tracker ──────────────────────────────
    def _face(self, snap=None, raises=False, no_func=False):
        mod = types.ModuleType("skill_face_tracker")
        if no_func:
            return mod
        if raises:
            mod._snapshot_state = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        else:
            mod._snapshot_state = lambda: snap
        return mod

    def test_user_at_desk_present(self):
        face = self._face({"last_sample_at": 1.0, "current_monitor": "middle_or_top"})
        with inject_modules(skill_face_tracker=face):
            self.assertTrue(self.mod._user_at_desk())

    def test_user_at_desk_away(self):
        face = self._face({"last_sample_at": 1.0, "current_monitor": "away"})
        with inject_modules(skill_face_tracker=face):
            self.assertFalse(self.mod._user_at_desk())

    def test_user_at_desk_unknown_monitor(self):
        face = self._face({"last_sample_at": 1.0, "current_monitor": "?"})
        with inject_modules(skill_face_tracker=face):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_no_sample(self):
        face = self._face({"last_sample_at": 0, "current_monitor": "left"})
        with inject_modules(skill_face_tracker=face):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_snapshot_raises(self):
        face = self._face(raises=True)
        with inject_modules(skill_face_tracker=face):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_no_func(self):
        face = self._face(no_func=True)
        with inject_modules(skill_face_tracker=face):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_no_module(self):
        with inject_modules(skill_face_tracker=None):
            self.assertIsNone(self.mod._user_at_desk())


# ─── _select_predictions (frozen clock) ───────────────────────────────────
class BriefingSelectTests(_BriefingTestBase):
    @contextlib.contextmanager
    def _at(self, hour, minute, wday=0):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(hour, minute, wday)):
            yield

    def test_select_precise_within_lead(self):
        with self._at(9, 0):       # 09:00 = minute 540
            snap = {"precise": [
                {"key": "p1", "action": "check_teams", "ratio": 0.9,
                 "center_minute": 550, "tolerance_min": 5, "common_arg": "sam"}],
                "broad": []}
            preds = self.mod._select_predictions(snap, self._cfg(lead_min=15))
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["key"], "p1")
        self.assertEqual(preds[0]["__lead_minutes"], 10)

    def test_select_precise_within_tolerance_band_now(self):
        # center already passed (sentinel delta) but within tolerance → "now".
        with self._at(9, 2):       # minute 542
            snap = {"precise": [
                {"key": "now", "action": "check_teams", "ratio": 0.9,
                 "center_minute": 540, "tolerance_min": 5}],  # |542-540|<=5
                "broad": []}
            preds = self.mod._select_predictions(snap, self._cfg())
        self.assertEqual(preds[0]["key"], "now")
        self.assertEqual(preds[0]["__lead_minutes"], 0)

    def test_select_precise_skipped_outside_lead_and_band(self):
        with self._at(9, 0):
            snap = {"precise": [
                {"key": "later", "action": "check_teams", "ratio": 0.9,
                 "center_minute": 700, "tolerance_min": 2}],   # 160 min away
                "broad": []}
            self.assertEqual(self.mod._select_predictions(snap, self._cfg()), [])

    def test_select_precise_center_not_int_skipped(self):
        with self._at(9, 0):
            snap = {"precise": [
                {"key": "bad", "action": "check_teams", "ratio": 0.9,
                 "center_minute": "not-int", "tolerance_min": 2}],
                "broad": []}
            self.assertEqual(self.mod._select_predictions(snap, self._cfg()), [])

    def test_select_filters_below_confidence(self):
        with self._at(9, 0):
            snap = {"precise": [
                {"key": "low", "action": "check_teams", "ratio": 0.2,
                 "center_minute": 545, "tolerance_min": 5}],
                "broad": []}
            self.assertEqual(
                self.mod._select_predictions(snap, self._cfg(conf_floor=0.5)), [])

    def test_select_broad_within_window(self):
        with self._at(10, 0, wday=0):     # Monday, 10:00
            snap = {"precise": [],
                    "broad": [{"key": "b", "action": "play_music", "ratio": 0.9,
                               "bucket": "weekday", "hour_window": [9, 11]}]}
            preds = self.mod._select_predictions(snap, self._cfg())
        self.assertEqual(preds[0]["key"], "b")
        self.assertEqual(preds[0]["__lead_minutes"], 0)

    def test_select_broad_bucket_mismatch(self):
        with self._at(10, 0, wday=0):     # weekday, but pattern is weekend
            snap = {"precise": [],
                    "broad": [{"key": "b", "ratio": 0.9, "bucket": "weekend",
                               "hour_window": [9, 11], "action": "play_music"}]}
            self.assertEqual(self.mod._select_predictions(snap, self._cfg()), [])

    def test_select_broad_below_confidence(self):
        with self._at(10, 0, wday=0):
            snap = {"precise": [],
                    "broad": [{"key": "b", "ratio": 0.1, "bucket": "weekday",
                               "hour_window": [9, 11], "action": "play_music"}]}
            self.assertEqual(self.mod._select_predictions(snap, self._cfg()), [])

    def test_select_broad_bad_window_len(self):
        with self._at(10, 0, wday=0):
            snap = {"precise": [],
                    "broad": [{"key": "b", "ratio": 0.9, "bucket": "weekday",
                               "hour_window": [9], "action": "play_music"}]}
            self.assertEqual(self.mod._select_predictions(snap, self._cfg()), [])

    def test_select_broad_window_non_numeric(self):
        with self._at(10, 0, wday=0):
            snap = {"precise": [],
                    "broad": [{"key": "b", "ratio": 0.9, "bucket": "weekday",
                               "hour_window": ["x", "y"], "action": "play_music"}]}
            self.assertEqual(self.mod._select_predictions(snap, self._cfg()), [])

    def test_select_broad_hour_outside_window(self):
        with self._at(14, 0, wday=0):     # 14:00 outside 9-11
            snap = {"precise": [],
                    "broad": [{"key": "b", "ratio": 0.9, "bucket": "weekday",
                               "hour_window": [9, 11], "action": "play_music"}]}
            self.assertEqual(self.mod._select_predictions(snap, self._cfg()), [])

    def test_select_precise_before_broad_and_broad_ratio_order(self):
        with self._at(10, 0, wday=0):
            snap = {
                "precise": [{"key": "pr", "action": "check_teams", "ratio": 0.9,
                             "center_minute": 605, "tolerance_min": 2}],  # 5 min
                "broad": [
                    {"key": "b_lo", "action": "play_music", "ratio": 0.60,
                     "bucket": "weekday", "hour_window": [9, 11]},
                    {"key": "b_hi", "action": "play_music", "ratio": 0.95,
                     "bucket": "weekday", "hour_window": [9, 11]},
                ]}
            preds = self.mod._select_predictions(snap, self._cfg())
        keys = [p["key"] for p in preds]
        self.assertEqual(keys[0], "pr")          # precise first
        self.assertEqual(keys[1], "b_hi")        # higher-ratio broad before lower
        self.assertEqual(keys[2], "b_lo")

    def test_select_empty_snapshot(self):
        self.assertEqual(self.mod._select_predictions({}, self._cfg()), [])


# ─── _next_eligible throttle ──────────────────────────────────────────────
class BriefingNextEligibleTests(_BriefingTestBase):
    def test_next_eligible_skips_throttled_key(self):
        today = time.strftime("%Y-%m-%d", time.localtime())
        pred = {"key": "k1", "action": "check_teams", "common_arg": "sam",
                "__lead_minutes": 10}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot",
                               return_value={"precise": [], "broad": []}), \
             mock.patch.object(self.mod, "_select_predictions", return_value=[pred]), \
             mock.patch.object(self.mod, "_load_state", return_value={"k1": today}):
            line, p = self.mod._next_eligible(bypass_throttle=False)
        self.assertEqual(line, "")
        self.assertEqual(p, {})

    def test_next_eligible_bypass_returns_line(self):
        pred = {"key": "k1", "action": "check_teams", "common_arg": "sam",
                "__lead_minutes": 10}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot",
                               return_value={"precise": [], "broad": []}), \
             mock.patch.object(self.mod, "_select_predictions", return_value=[pred]):
            line, p = self.mod._next_eligible(bypass_throttle=True)
        self.assertIn("Sam sync", line)
        self.assertEqual(p["key"], "k1")

    def test_next_eligible_skips_keyless_and_empty_line(self):
        # First candidate has no key (skipped), second composes an empty line
        # (skipped), third is a valid music pred → returned.
        preds = [
            {"key": "", "action": "check_teams", "common_arg": "x"},
            {"key": "k2", "action": "unknown_no_offer"},        # composes ""
            {"key": "k3", "action": "play_music", "common_arg": "queen"},
        ]
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot",
                               return_value={"precise": [], "broad": []}), \
             mock.patch.object(self.mod, "_select_predictions", return_value=preds), \
             mock.patch.object(self.mod, "_load_state", return_value={}):
            line, p = self.mod._next_eligible(bypass_throttle=False)
        self.assertIn("Queen", line)
        self.assertEqual(p["key"], "k3")

    def test_next_eligible_no_snapshot(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot", return_value={}):
            self.assertEqual(self.mod._next_eligible(bypass_throttle=True), ("", {}))

    def test_next_eligible_no_candidates(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot",
                               return_value={"precise": [], "broad": []}), \
             mock.patch.object(self.mod, "_select_predictions", return_value=[]):
            self.assertEqual(self.mod._next_eligible(bypass_throttle=True), ("", {}))


# ─── _enqueue_speech ──────────────────────────────────────────────────────
class BriefingEnqueueTests(_BriefingTestBase):
    def test_enqueue_via_proactive_announce(self):
        calls = {}
        def _announce(msg, source=None):
            calls["msg"] = msg
            calls["source"] = source
            return True
        bc = _benign_bc(proactive_announce=_announce)
        with inject_modules(bobert_companion=bc):
            ok = self.mod._enqueue_speech("Sam sync in 5 minutes, sir.")
        self.assertTrue(ok)
        self.assertEqual(calls["source"], "anticipation_briefing")
        self.assertIn("Sam sync", calls["msg"])

    def test_enqueue_announce_returns_false(self):
        bc = _benign_bc(proactive_announce=lambda msg, source=None: False)
        # announce returns falsy → _enqueue_speech returns False (no fallback;
        # the announcer was callable so the except path isn't taken).
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._enqueue_speech("x"))

    def test_enqueue_fallback_file_write(self):
        # bc present but announcer missing/not-callable → direct file fallback.
        tmp = tempfile.mkdtemp(prefix="anticip_q_")
        self.addCleanup(lambda: self._rmtree(tmp))
        bc = _benign_bc()   # no proactive_announce attribute
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_PROJECT_DIR", tmp):
            ok = self.mod._enqueue_speech("queue me")
        self.assertTrue(ok)
        with open(os.path.join(tmp, "pending_speech.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[-1]["message"], "queue me")

    def test_enqueue_fallback_appends_existing(self):
        tmp = tempfile.mkdtemp(prefix="anticip_q_")
        self.addCleanup(lambda: self._rmtree(tmp))
        qp = os.path.join(tmp, "pending_speech.json")
        with open(qp, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        bc = _benign_bc()
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_PROJECT_DIR", tmp):
            self.mod._enqueue_speech("new")
        with open(qp, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["old", "new"])

    def test_enqueue_fallback_corrupt_existing_resets(self):
        tmp = tempfile.mkdtemp(prefix="anticip_q_")
        self.addCleanup(lambda: self._rmtree(tmp))
        qp = os.path.join(tmp, "pending_speech.json")
        with open(qp, "w", encoding="utf-8") as f:
            f.write("{ corrupt")
        bc = _benign_bc()
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_PROJECT_DIR", tmp):
            self.mod._enqueue_speech("fresh")
        with open(qp, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["fresh"])

    def test_enqueue_fallback_when_import_fails(self):
        # import_module raising → fall through to direct file write.
        tmp = tempfile.mkdtemp(prefix="anticip_q_")
        self.addCleanup(lambda: self._rmtree(tmp))
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")), \
             mock.patch.object(self.mod, "_PROJECT_DIR", tmp):
            ok = self.mod._enqueue_speech("via fallback")
        self.assertTrue(ok)
        with open(os.path.join(tmp, "pending_speech.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)[-1]["message"], "via fallback")

    def test_enqueue_total_failure_returns_false(self):
        # No announcer AND the fallback file write fails (mkstemp raises) → False.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")), \
             mock.patch.object(self.mod, "_PROJECT_DIR", "X:/nonexistent"), \
             mock.patch.object(self.mod.tempfile, "mkstemp",
                               side_effect=OSError("nope")):
            self.assertFalse(self.mod._enqueue_speech("doomed"))

    def test_enqueue_fallback_replace_failure_unlinks(self):
        # mkstemp ok but os.replace fails → inner except unlinks tmp + re-raise
        # → outer except returns False. No tmp left behind.
        tmp = tempfile.mkdtemp(prefix="anticip_q_")
        self.addCleanup(lambda: self._rmtree(tmp))
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")), \
             mock.patch.object(self.mod, "_PROJECT_DIR", tmp), \
             mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("rename denied")):
            self.assertFalse(self.mod._enqueue_speech("x"))
        # only files (if any) are the dir itself — tmp file cleaned up.
        self.assertEqual([f for f in os.listdir(tmp) if f.endswith(".tmp")], [])

    def test_enqueue_fallback_replace_and_unlink_both_fail(self):
        # os.replace fails AND the cleanup os.unlink fails → innermost
        # ``except Exception: pass`` (line 453) runs, then re-raise → outer
        # except returns False. Must not propagate.
        tmp = tempfile.mkdtemp(prefix="anticip_q_")
        self.addCleanup(lambda: self._rmtree(tmp))
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")), \
             mock.patch.object(self.mod, "_PROJECT_DIR", tmp), \
             mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("rename denied")), \
             mock.patch.object(self.mod.os, "unlink",
                               side_effect=OSError("unlink denied")):
            self.assertFalse(self.mod._enqueue_speech("x"))

    @staticmethod
    def _rmtree(path):
        for fn in os.listdir(path):
            try:
                os.unlink(os.path.join(path, fn))
            except OSError:
                pass
        try:
            os.rmdir(path)
        except OSError:
            pass


# ─── registered actions ───────────────────────────────────────────────────
class BriefingActionTests(_BriefingTestBase):
    def test_action_now_suppressed_sleep(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=True):
            out = self.actions["anticipation_briefing_now"]("")
        self.assertIn("Suppressed", out)
        self.assertIn("sleep", out.lower())

    def test_action_now_suppressed_in_call(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=True):
            out = self.actions["anticipation_briefing_now"]("")
        self.assertIn("Suppressed", out)
        self.assertIn("call", out.lower())

    def test_action_now_suppressed_away(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=False):
            out = self.actions["anticipation_briefing_now"]("")
        self.assertIn("Suppressed", out)
        self.assertIn("desk", out.lower())

    def test_action_now_no_match(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("", {})):
            out = self.actions["anticipation_briefing_now"]("")
        self.assertIn("No prediction matches", out)

    def test_action_now_fires_and_marks(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("Sam sync in 5 minutes, sir.", {"key": "k"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=True), \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.actions["anticipation_briefing_now"]("")
        self.assertIn("Sam sync", out)
        mark.assert_called_once()

    def test_action_now_enqueue_fails(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("a line", {"key": "k"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=False):
            out = self.actions["anticipation_briefing_now"]("")
        self.assertIn("Could not enqueue", out)
        self.assertIn("a line", out)

    def test_action_status_reports_counts(self):
        snap = {"broad": [{}, {}], "precise": [{}]}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot", return_value=snap), \
             mock.patch.object(self.mod, "_select_predictions", return_value=[]), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True):
            out = self.actions["anticipation_briefing_status"]("")
        self.assertIn("2 broad, 1 precise", out)
        self.assertIn("eligible right now", out)

    def test_action_status_disabled(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=self._cfg(enabled=False)), \
             mock.patch.object(self.mod, "_load_snapshot", return_value={}), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True):
            out = self.actions["anticipation_briefing_status"]("")
        self.assertIn("disabled in config", out)

    def test_action_status_all_suppression_flags(self):
        snap = {"broad": [{}], "precise": []}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot", return_value=snap), \
             mock.patch.object(self.mod, "_select_predictions", return_value=[{}]), \
             mock.patch.object(self.mod, "_load_state",
                               return_value={"k": time.strftime("%Y-%m-%d", time.localtime())}), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=True), \
             mock.patch.object(self.mod, "_is_in_call", return_value=True), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=False):
            out = self.actions["anticipation_briefing_status"]("")
        self.assertIn("sleep/standby active", out)
        self.assertIn("in a call", out.lower())
        self.assertIn("user away", out.lower())
        self.assertIn("1 briefing surfaced today", out)


# ─── scheduler loop ───────────────────────────────────────────────────────
class _StopLoop(Exception):
    pass


def _sleep_after(n):
    box = {"n": n}

    def _fake(_seconds):
        if box["n"] <= 0:
            raise _StopLoop()
        box["n"] -= 1
    return _fake


class BriefingSchedulerLoopTests(_BriefingTestBase):
    def setUp(self):
        super().setUp()
        mock.patch.object(self.mod.logging, "exception", lambda *a, **k: None).start()
        self.addCleanup(mock.patch.stopall)

    @contextlib.contextmanager
    def _loop_env(self, *, enabled=True, sleep=False, in_call=False,
                  at_desk=None, sleeps_before_stop=1):
        # The loop sleeps once at INITIAL_DELAY, then once per poll iteration.
        # sleeps_before_stop=1 → exactly ONE poll runs (initial sleep no-ops,
        # the poll-end/gate sleep raises _StopLoop).
        cfg = self._cfg(enabled=enabled)
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=_sleep_after(sleeps_before_stop)), \
             mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=sleep), \
             mock.patch.object(self.mod, "_is_in_call", return_value=in_call), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=at_desk):
            yield

    def test_loop_disabled(self):
        # sleeps_before_stop=2 lets the gate's sleep no-op once so the `continue`
        # line runs, then _StopLoop fires on the next pass's sleep.
        eq = mock.MagicMock()
        with self._loop_env(enabled=False, sleeps_before_stop=2), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("x", {})), \
             mock.patch.object(self.mod, "_enqueue_speech", eq):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        eq.assert_not_called()

    def test_loop_gate_sleep(self):
        eq = mock.MagicMock()
        with self._loop_env(sleep=True, sleeps_before_stop=2), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("x", {})), \
             mock.patch.object(self.mod, "_enqueue_speech", eq):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        eq.assert_not_called()

    def test_loop_gate_in_call(self):
        eq = mock.MagicMock()
        with self._loop_env(in_call=True, sleeps_before_stop=2), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("x", {})), \
             mock.patch.object(self.mod, "_enqueue_speech", eq):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        eq.assert_not_called()

    def test_loop_gate_away(self):
        eq = mock.MagicMock()
        with self._loop_env(at_desk=False, sleeps_before_stop=2), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("x", {})), \
             mock.patch.object(self.mod, "_enqueue_speech", eq):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        eq.assert_not_called()

    def test_loop_no_line_no_fire(self):
        eq = mock.MagicMock()
        with self._loop_env(), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("", {})), \
             mock.patch.object(self.mod, "_enqueue_speech", eq):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        eq.assert_not_called()

    def test_loop_fires_and_marks(self):
        with self._loop_env(), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("Time for music, sir.", {"key": "m"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=True) as eq, \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        eq.assert_called_once()
        mark.assert_called_once()

    def test_loop_fires_but_enqueue_fails_no_mark(self):
        with self._loop_env(), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("line", {"key": "m"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=False), \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        mark.assert_not_called()

    def test_loop_inner_exception_logged_and_swallowed(self):
        # _next_eligible raising is caught by the loop's try/except, which logs
        # via logging.exception and then hits the end-of-loop sleep (which raises
        # _StopLoop). The fire path is never reached.
        with self._loop_env(), \
             mock.patch.object(self.mod, "_next_eligible",
                               side_effect=RuntimeError("pick blew up")), \
             mock.patch.object(self.mod, "_enqueue_speech") as eq:
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        eq.assert_not_called()


# ─── register() ───────────────────────────────────────────────────────────
class BriefingRegisterTests(_BriefingTestBase):
    def _register_with(self, bc, **patches):
        actions = {}
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.threading, "Thread") as Thread, \
             contextlib.ExitStack() as stack:
            for target, kw in patches.items():
                stack.enter_context(mock.patch.object(self.mod, target, **kw))
            self.mod.register(actions)
        return actions, Thread

    def test_register_disabled_no_thread(self):
        bc = _benign_bc(ANTICIPATION_BRIEFING_ENABLED=False)
        actions, Thread = self._register_with(bc)
        self.assertIn("anticipation_briefing_now", actions)
        self.assertIn("anticipation_briefing_status", actions)
        Thread.assert_not_called()

    def test_register_enabled_starts_thread_with_snapshot(self):
        bc = _benign_bc(ANTICIPATION_BRIEFING_ENABLED=True,
                        proactive_announce=lambda *a, **k: True)
        snap = {"broad": [{"k": 1}], "precise": [{"k": 2}],
                "generated_at": time.time()}
        made = mock.MagicMock()
        actions = {}
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_load_snapshot", return_value=snap), \
             mock.patch.object(self.mod.threading, "Thread", return_value=made) as Thread:
            self.mod.register(actions)
        Thread.assert_called_once()
        made.start.assert_called_once()
        self.assertIn("anticipation_briefing_now", actions)

    def test_register_enabled_empty_snapshot_warns(self):
        # Empty snapshot path (the WARN branch) + thread still starts.
        bc = _benign_bc(ANTICIPATION_BRIEFING_ENABLED=True,
                        proactive_announce=lambda *a, **k: True)
        made = mock.MagicMock()
        actions = {}
        buf = io.StringIO()
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_load_snapshot", return_value={}), \
             mock.patch.object(self.mod.threading, "Thread", return_value=made), \
             contextlib.redirect_stdout(buf):
            self.mod.register(actions)
        made.start.assert_called_once()
        self.assertIn("no pattern_learning snapshot", buf.getvalue())

    def test_register_enabled_announcer_not_callable_warns(self):
        # bc present but proactive_announce missing → the announcer-WARN branch.
        bc = _benign_bc(ANTICIPATION_BRIEFING_ENABLED=True)   # no announce attr
        made = mock.MagicMock()
        actions = {}
        buf = io.StringIO()
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_load_snapshot",
                               return_value={"broad": [], "precise": []}), \
             mock.patch.object(self.mod.threading, "Thread", return_value=made), \
             contextlib.redirect_stdout(buf):
            self.mod.register(actions)
        self.assertIn("proactive_announce", buf.getvalue())
        made.start.assert_called_once()

    def test_register_enabled_bc_import_fails_warns(self):
        # The announcer-wiring import raising drives the except WARN (621-623).
        bc = _benign_bc(ANTICIPATION_BRIEFING_ENABLED=True)
        made = mock.MagicMock()
        actions = {}
        buf = io.StringIO()
        # _read_config consults bc via import_module; we let the FIRST call (in
        # register's config read) succeed by seeding bc, but force the LATER
        # announcer-check import to raise. Simplest: patch import_module to raise
        # and have _read_config use the seeded module via a separate patch.
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_read_config",
                               return_value=self._cfg(enabled=True)), \
             mock.patch.object(self.mod, "_load_snapshot",
                               return_value={"broad": [], "precise": []}), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")), \
             mock.patch.object(self.mod.threading, "Thread", return_value=made), \
             contextlib.redirect_stdout(buf):
            self.mod.register(actions)
        self.assertIn("bobert_companion import failed", buf.getvalue())
        made.start.assert_called_once()

    def test_registered_actions_callable(self):
        bc = _benign_bc(ANTICIPATION_BRIEFING_ENABLED=False)
        actions, _ = self._register_with(bc)
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot", return_value={}), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True):
            out = actions["anticipation_briefing_status"]("ignored")
        self.assertTrue(out.startswith("Anticipation briefing"))


# ─── __main__ smoke block ─────────────────────────────────────────────────
class BriefingMainBlockTests(unittest.TestCase):
    def _run_main(self, extra_modules):
        # Execute the module as __main__ with fakes seeded and threads neutered,
        # so the ``if __name__ == "__main__"`` smoke block is exercised without
        # booting the monolith or starting a real thread. Returns captured stdout.
        import threading
        saved = {}
        for name, obj in extra_modules.items():
            saved[name] = sys.modules.get(name, _SENTINEL)
            sys.modules[name] = obj
        buf = io.StringIO()
        try:
            with mock.patch.object(threading.Thread, "start", lambda self: None), \
                 contextlib.redirect_stdout(buf):
                runpy.run_path(
                    os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "skills", "anticipation_briefing.py"),
                    run_name="__main__")
        finally:
            for name, prev in saved.items():
                if prev is _SENTINEL:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev
        return buf.getvalue()

    def test_main_block_runs_offline_empty_snapshot(self):
        bc = _benign_bc(ANTICIPATION_BRIEFING_ENABLED=True)
        pl = types.ModuleType("skill_pattern_learning")
        pl._load_aggregated = lambda: {}     # no candidates
        out = self._run_main({"bobert_companion": bc, "skill_pattern_learning": pl})
        self.assertIn("candidates now: 0", out)

    def test_main_block_with_eligible_candidate(self):
        # Build a broad pattern eligible at the real current clock so the
        # ``for c in cands[:5]`` print loop (line 646) and _next_eligible run.
        now = time.localtime()
        bucket = "weekday" if now.tm_wday <= 4 else "weekend"
        snapshot = {
            "broad": [{
                "key": "music:now", "action": "play_music", "ratio": 0.99,
                "bucket": bucket, "hour_window": [now.tm_hour, now.tm_hour + 1],
                "common_arg": "test artist",
            }],
            "precise": [],
            "generated_at": time.time(),
        }
        bc = _benign_bc(ANTICIPATION_BRIEFING_ENABLED=True)
        pl = types.ModuleType("skill_pattern_learning")
        pl._load_aggregated = lambda: snapshot
        out = self._run_main({"bobert_companion": bc, "skill_pattern_learning": pl})
        self.assertIn("candidates now: 1", out)
        self.assertIn("Test Artist", out)   # composed line printed in the loop


if __name__ == "__main__":
    unittest.main()
