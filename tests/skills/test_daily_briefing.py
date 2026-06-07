"""Logic tests for skills/daily_briefing.py.

Covers the briefing-text assembly (time phrase + weather/meeting/bambu extras,
"nothing remarkable" empty path), the weather/meeting formatters that sit on
top of briefing_sources, the cross-skill Bambu/face-tracker reads, and the
manual daily_briefing action. The scheduler thread is neutered by the harness;
the speech enqueue is mocked so no pending_speech.json is written.
"""
from __future__ import annotations

import contextlib
import datetime
import json
import sys
import types
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules, restoring the prior
    state (including absence) on exit; dotted leaves are also set on the parent
    package. Mirrors tests/skills/test_self_diagnostic.py's isolation contract."""
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
                with contextlib.suppress(AttributeError):
                    delattr(parent, leaf)
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


class _LoopBreak(BaseException):
    """Sentinel raised to break _scheduler_loop's `while True` after one pass.
    Derives from BaseException so the loop's own `except Exception` (which logs
    and continues) does NOT swallow it."""


class DailyBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_briefing")

    # ── _format_time_phrase (pure) ───────────────────────────────────────
    def test_format_time_phrase(self):
        st = time.struct_time((2026, 6, 1, 13, 5, 0, 0, 152, -1))
        self.assertEqual(self.mod._format_time_phrase(st), "1:05 PM")
        st2 = time.struct_time((2026, 6, 1, 0, 0, 0, 0, 152, -1))
        self.assertEqual(self.mod._format_time_phrase(st2), "12:00 AM")

    # ── _build_briefing assembly ─────────────────────────────────────────
    def test_build_briefing_with_all_extras(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value="57 degrees and clear"), \
             mock.patch.object(self.mod, "_first_meeting_today",
                               return_value="your first meeting today is at 9:30 AM"), \
             mock.patch.object(self.mod, "_bambu_status", return_value="the H2D is mid-print"):
            out = self.mod._build_briefing()
        self.assertIn("Good morning, sir", out)
        self.assertIn("57 degrees and clear", out)
        self.assertIn("9:30 AM", out)
        self.assertIn("mid-print", out)
        self.assertTrue(out.endswith("."))

    def test_build_briefing_nothing_remarkable(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value=""), \
             mock.patch.object(self.mod, "_first_meeting_today", return_value=""), \
             mock.patch.object(self.mod, "_bambu_status", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("nothing remarkable to report", out)

    # ── _fetch_weather (over briefing_sources; store Celsius, speak F) ────
    def test_fetch_weather_phrase(self):
        bs = mock.MagicMock()
        # 14 C stored → 57 F spoken (14*9/5+32 = 57.2 → 57), agreeing with
        # morning_briefing / weather_briefing rather than voicing raw Celsius.
        bs.get_weather_data.return_value = {"temp_c": 14, "desc": "Overcast", "source": "wttr"}
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            self.assertEqual(self.mod._fetch_weather(),
                             "outside temperature is 57 degrees and overcast")

    def test_fetch_weather_cached_suffix(self):
        bs = mock.MagicMock()
        # 9 C stored → 48 F spoken (9*9/5+32 = 48.2 → 48).
        bs.get_weather_data.return_value = {"temp_c": 9, "desc": "", "source": "cache", "stale": True}
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            out = self.mod._fetch_weather()
        self.assertIn("48 degrees", out)
        self.assertIn("(cached)", out)

    def test_fetch_weather_degrades_when_sources_missing(self):
        with mock.patch.object(self.mod, "_briefing_sources", return_value=None):
            self.assertEqual(self.mod._fetch_weather(), "")

    def test_fetch_weather_bad_temp(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"desc": "rain"}  # no temp_c
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            self.assertEqual(self.mod._fetch_weather(), "")

    # ── _first_meeting_today formatting ──────────────────────────────────
    def test_first_meeting_with_organizer_and_subject(self):
        bs = mock.MagicMock()
        bs.get_first_meeting_data.return_value = {
            "start": datetime.datetime(2026, 6, 1, 9, 30),
            "organizer": "Sam Industries <sam@x.com>",
            "subject": "Design review",
        }
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            out = self.mod._first_meeting_today()
        self.assertIn("9:30 AM", out)
        self.assertIn("Sam Industries", out)
        self.assertIn("Design review", out)

    def test_first_meeting_none(self):
        bs = mock.MagicMock()
        bs.get_first_meeting_data.return_value = None
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            self.assertEqual(self.mod._first_meeting_today(), "")

    # ── _bambu_status (cross-skill read) ─────────────────────────────────
    def test_bambu_status_finished_recently(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": time.time() - 600, "gcode_state": "FINISH",
                       "filename": "bracket.3mf"}
        fake._strip_filename = lambda s: "bracket"
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("finished printing", out)
        self.assertIn("bracket", out)

    def test_bambu_status_running_with_layers(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": time.time(), "gcode_state": "RUNNING",
                       "filename": "part.gcode", "layer_num": 10, "total_layer": 100,
                       "mc_remaining": 90}
        fake._strip_filename = lambda s: "part"
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("mid-print", out)
        self.assertIn("layer 10 of 100", out)
        self.assertIn("1 hour and 30 minutes remaining", out)

    def test_bambu_status_not_loaded(self):
        import sys
        # Ensure the monitor module is absent.
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_bambu_monitor", None)
            self.assertEqual(self.mod._bambu_status(), "")

    def test_bambu_status_no_data(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": 0.0}
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_status(), "")

    # ── _user_at_desk (cross-skill read) ─────────────────────────────────
    def test_user_at_desk_present(self):
        fake = mock.MagicMock()
        fake._snapshot_state.return_value = {"last_sample_at": time.time(),
                                             "current_monitor": "middle_or_top"}
        import sys
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIs(self.mod._user_at_desk(), True)

    def test_user_at_desk_away(self):
        fake = mock.MagicMock()
        fake._snapshot_state.return_value = {"last_sample_at": time.time(),
                                             "current_monitor": "away"}
        import sys
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIs(self.mod._user_at_desk(), False)

    def test_user_at_desk_unknown_when_not_loaded(self):
        import sys
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_face_tracker", None)
            self.assertIsNone(self.mod._user_at_desk())

    # ── state load/save ──────────────────────────────────────────────────
    def test_load_last_fired_missing(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._load_last_fired_date(), "")

    def test_load_last_fired_reads_value(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open",
                        mock.mock_open(read_data='{"last_fired_date": "2026-06-01"}')):
            self.assertEqual(self.mod._load_last_fired_date(), "2026-06-01")

    # ── daily_briefing action ────────────────────────────────────────────
    def test_action_returns_and_enqueues(self):
        mod, actions = load_skill_isolated("daily_briefing")
        with mock.patch.object(mod, "_build_briefing", return_value="Good morning, sir. test."), \
             mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_save_last_fired_date") as save:
            out = actions["daily_briefing"]("")
        self.assertEqual(out, "Good morning, sir. test.")
        enq.assert_called_once_with("Good morning, sir. test.")
        save.assert_called_once()

    def test_action_handles_exception(self):
        mod, actions = load_skill_isolated("daily_briefing")
        with mock.patch.object(mod, "_build_briefing", side_effect=RuntimeError("boom")):
            out = actions["daily_briefing"]("")
        self.assertIn("failed", out.lower())


class DailyEnqueueTests(unittest.TestCase):
    """_enqueue_speech — announcer path + atomic-write fallback variants."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_briefing")

    def test_enqueue_uses_bc_announcer(self):
        bc = types.ModuleType("bobert_companion")
        calls = []
        bc.proactive_announce = lambda msg, source=None: calls.append((msg, source))
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("Good morning, sir.")
        self.assertEqual(calls, [("Good morning, sir.", "daily")])

    def test_enqueue_falls_back_to_atomic_write(self):
        bc = types.ModuleType("bobert_companion")   # no proactive_announce
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("hello sir")
        wr.assert_called_once()
        path, data = wr.call_args[0][0], wr.call_args[0][1]
        self.assertEqual(path, self.mod._SPEECH_QUEUE)
        self.assertEqual(data[-1]["message"], "hello sir")

    def test_enqueue_appends_to_existing_queue(self):
        bc = types.ModuleType("bobert_companion")
        existing = json.dumps([{"ts": 1.0, "message": "old"}])
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=existing)), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("new")
        data = wr.call_args[0][1]
        self.assertEqual([d["message"] for d in data], ["old", "new"])

    def test_enqueue_corrupt_queue_resets_to_list(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{garbage")), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("fresh")
        data = wr.call_args[0][1]
        self.assertEqual([d["message"] for d in data], ["fresh"])

    def test_enqueue_announcer_raises_falls_through(self):
        bc = types.ModuleType("bobert_companion")

        def _boom(*_a, **_k):
            raise RuntimeError("announcer broke")
        bc.proactive_announce = _boom
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("still spoken")
        wr.assert_called_once()

    def test_enqueue_atomic_write_failure_swallowed(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only share")):
            self.mod._enqueue_speech("resilient")   # must not raise


class DailyConfigStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_briefing")

    def test_read_config_from_bc(self):
        bc = types.ModuleType("bobert_companion")
        bc.DAILY_BRIEFING_ENABLED = False
        bc.DAILY_BRIEFING_HOUR = 7
        bc.DAILY_BRIEFING_MINUTE = 45
        bc.DAILY_BRIEFING_WAIT_MINUTES = 15
        with inject_modules(bobert_companion=bc):
            cfg = self.mod._read_config()
        self.assertEqual(cfg, {"enabled": False, "hour": 7, "minute": 45, "wait_min": 15})

    def test_read_config_defaults_when_bc_import_fails(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            cfg = self.mod._read_config()
        self.assertEqual(cfg, {"enabled": True, "hour": 8, "minute": 0, "wait_min": 30})

    def test_load_last_fired_corrupt_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{bad json")):
            self.assertEqual(self.mod._load_last_fired_date(), "")

    def test_load_last_fired_empty_payload(self):
        # File holds JSON null → (json.load() or {}).get(...) → "".
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="null")):
            self.assertEqual(self.mod._load_last_fired_date(), "")

    def test_save_last_fired_writes_atomically(self):
        # Redirect both the temp-write dir (_PROJECT_DIR) and the destination
        # (_STATE_FILE) into a throwaway temp dir so nothing in the project is
        # touched, then read back what landed on disk.
        import tempfile
        import os as _os
        tmpdir = tempfile.mkdtemp(prefix="daily_state_")
        state_file = _os.path.join(tmpdir, "daily_briefing_state.json")

        def _cleanup():
            with contextlib.suppress(OSError):
                _os.unlink(state_file)
            with contextlib.suppress(OSError):
                _os.rmdir(tmpdir)
        self.addCleanup(_cleanup)
        with mock.patch.object(self.mod, "_PROJECT_DIR", tmpdir), \
             mock.patch.object(self.mod, "_STATE_FILE", state_file):
            self.mod._save_last_fired_date("2026-06-02")
            with open(state_file, encoding="utf-8") as f:
                payload = json.load(f)
        self.assertEqual(payload, {"last_fired_date": "2026-06-02"})

    def test_save_last_fired_swallows_error(self):
        with mock.patch.object(self.mod.tempfile, "mkstemp",
                               side_effect=OSError("no space")):
            # Must not raise — logged to console.
            self.mod._save_last_fired_date("2026-06-02")

    def test_save_last_fired_replace_failure_unlinks_tmp(self):
        # mkstemp succeeds and the body writes, but os.replace fails → the inner
        # cleanup unlinks the temp file and the error is swallowed by the outer
        # handler. We assert the temp file was removed.
        import tempfile
        import os as _os
        tmpdir = tempfile.mkdtemp(prefix="daily_state2_")

        def _rmdir():
            with contextlib.suppress(OSError):
                _os.rmdir(tmpdir)
        self.addCleanup(_rmdir)
        unlinked = {}
        real_unlink = self.mod.os.unlink

        def _unlink(p):
            unlinked["path"] = p
            real_unlink(p)
        with mock.patch.object(self.mod, "_PROJECT_DIR", tmpdir), \
             mock.patch.object(self.mod, "_STATE_FILE",
                               _os.path.join(tmpdir, "s.json")), \
             mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("rename failed")), \
             mock.patch.object(self.mod.os, "unlink", side_effect=_unlink):
            self.mod._save_last_fired_date("2026-06-02")
        # The temp file was cleaned up and no longer exists.
        self.assertIn("path", unlinked)
        self.assertFalse(_os.path.exists(unlinked["path"]))


class DailyBriefingSourcesTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_briefing")

    def test_briefing_sources_relative_import(self):
        fake = types.ModuleType("briefing_sources")
        # The function does `from . import briefing_sources`; that resolves the
        # leaf as an attribute of the skill's package. Our skill is loaded as a
        # flat module (skill_daily_briefing) with no package, so the relative
        # import fails and it falls back to `import briefing_sources`.
        with inject_modules(briefing_sources=fake):
            out = self.mod._briefing_sources()
        self.assertIs(out, fake)

    def test_briefing_sources_unavailable_returns_none(self):
        real_import = __import__

        def _imp(name, *a, **k):
            if name == "briefing_sources" or name.endswith("briefing_sources"):
                raise ImportError("missing")
            return real_import(name, *a, **k)
        with inject_modules(briefing_sources=None), \
             mock.patch("builtins.__import__", side_effect=_imp):
            self.assertIsNone(self.mod._briefing_sources())


class DailyWeatherMeetingEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_briefing")

    def test_fetch_weather_no_desc(self):
        # 20 C stored → 68 F spoken (20*9/5+32 = 68).
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"temp_c": 20, "desc": "", "source": "wttr"}
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            self.assertEqual(self.mod._fetch_weather(),
                             "outside temperature is 68 degrees")

    def test_fetch_weather_no_data(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = None
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            self.assertEqual(self.mod._fetch_weather(), "")

    def test_first_meeting_sources_missing(self):
        with mock.patch.object(self.mod, "_briefing_sources", return_value=None):
            self.assertEqual(self.mod._first_meeting_today(), "")

    def test_first_meeting_bad_start_type(self):
        bs = mock.MagicMock()
        bs.get_first_meeting_data.return_value = {"start": "not a datetime"}
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            self.assertEqual(self.mod._first_meeting_today(), "")

    def test_first_meeting_pm_no_organizer_no_subject(self):
        bs = mock.MagicMock()
        bs.get_first_meeting_data.return_value = {
            "start": datetime.datetime(2026, 6, 2, 14, 5), "organizer": "", "subject": ""}
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            out = self.mod._first_meeting_today()
        self.assertEqual(out, "your first meeting today is at 2:05 PM")

    def test_first_meeting_organizer_is_self_omitted(self):
        bs = mock.MagicMock()
        bs.get_first_meeting_data.return_value = {
            "start": datetime.datetime(2026, 6, 2, 9, 0),
            "organizer": "me", "subject": "Standup"}
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            out = self.mod._first_meeting_today()
        self.assertNotIn("with", out)
        self.assertIn("Standup", out)

    def test_first_meeting_organizer_email_only_omitted(self):
        # Organizer is a bare email (name part contains '@') → no "with" clause.
        bs = mock.MagicMock()
        bs.get_first_meeting_data.return_value = {
            "start": datetime.datetime(2026, 6, 2, 10, 30),
            "organizer": "boss@corp.com", "subject": "Sync"}
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            out = self.mod._first_meeting_today()
        self.assertNotIn("with", out)


class DailyBambuBranchTests(unittest.TestCase):
    """Remaining _bambu_status branches."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_briefing")

    def _monitor(self, state, *, lock=None, strip=lambda s: "file", strip_raises=False):
        fake = types.ModuleType("skill_bambu_monitor")
        fake._state_lock = lock
        fake._state = state
        if strip_raises:
            def _boom(_s):
                raise RuntimeError("strip failed")
            fake._strip_filename = _boom
        else:
            fake._strip_filename = strip
        return fake

    def test_bambu_uses_state_lock(self):
        lock = __import__("threading").Lock()
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "p.gcode", "layer_num": 1, "total_layer": 2},
            lock=lock, strip=lambda s: "p")
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        self.assertIn("mid-print", out)
        self.assertFalse(lock.locked())   # released after read

    def test_bambu_state_is_none(self):
        fake = self._monitor(None)
        with inject_modules(skill_bambu_monitor=fake):
            self.assertEqual(self.mod._bambu_status(), "")

    def test_bambu_read_exception_returns_empty(self):
        # _state is non-None but not dict-able → dict(raw_state) raises inside
        # the try/except, which returns "".
        fake = types.ModuleType("skill_bambu_monitor")
        fake._state = 12345          # dict(12345) → TypeError
        fake._state_lock = None
        with inject_modules(skill_bambu_monitor=fake):
            self.assertEqual(self.mod._bambu_status(), "")

    def test_bambu_finish_without_filename(self):
        fake = self._monitor(
            {"last_update": time.time() - 600, "gcode_state": "FINISH",
             "filename": ""}, strip=lambda s: "")
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        self.assertIn("finished its overnight print", out)

    def test_bambu_finish_old_no_timestamp(self):
        # Finished >12h ago → no "at HH:MM" suffix.
        fake = self._monitor(
            {"last_update": time.time() - 13 * 3600, "gcode_state": "FINISH",
             "filename": "x.3mf"}, strip=lambda s: "x")
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        self.assertIn("finished printing 'x'", out)
        self.assertNotIn(" at ", out)

    def test_bambu_failed_state(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "FAILED"})
        with inject_modules(skill_bambu_monitor=fake):
            self.assertIn("print failure", self.mod._bambu_status())

    def test_bambu_running_minutes_only(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "p", "layer_num": 5, "total_layer": 50, "mc_remaining": 45},
            strip=lambda s: "p")
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        self.assertIn("about 45 minutes remaining", out)

    def test_bambu_running_exact_hours(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "p", "layer_num": 5, "total_layer": 50, "mc_remaining": 120},
            strip=lambda s: "p")
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        self.assertIn("about 2 hours remaining", out)

    def test_bambu_running_one_hour_singular(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "p", "layer_num": 5, "total_layer": 50, "mc_remaining": 60},
            strip=lambda s: "p")
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        self.assertIn("about 1 hour remaining", out)

    def test_bambu_running_no_filename_no_layers_no_remaining(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "PREPARE"},
            strip=lambda s: "")
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        self.assertIn("has a print in progress", out)

    def test_bambu_running_bad_remaining_value(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "p", "mc_remaining": "soon"}, strip=lambda s: "p")
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        self.assertIn("mid-print", out)
        self.assertNotIn("remaining", out)   # unparseable → omitted

    def test_bambu_running_zero_remaining_omitted(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "PAUSE",
             "filename": "p", "mc_remaining": 0}, strip=lambda s: "p")
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        self.assertNotIn("remaining", out)

    def test_bambu_strip_filename_error_swallowed(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "p", "layer_num": 1, "total_layer": 2}, strip_raises=True)
        with inject_modules(skill_bambu_monitor=fake):
            out = self.mod._bambu_status()
        # _strip_filename raised → fname stays "" → generic phrasing.
        self.assertIn("has a print in progress", out)

    def test_bambu_idle_state_returns_empty(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "IDLE"})
        with inject_modules(skill_bambu_monitor=fake):
            self.assertEqual(self.mod._bambu_status(), "")


class DailyUserAtDeskBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_briefing")

    def test_no_snapshot_function(self):
        fake = types.ModuleType("skill_face_tracker")   # no _snapshot_state
        if hasattr(fake, "_snapshot_state"):
            del fake._snapshot_state
        with inject_modules(skill_face_tracker=fake):
            self.assertIsNone(self.mod._user_at_desk())

    def test_snapshot_raises_returns_none(self):
        fake = types.ModuleType("skill_face_tracker")
        fake._snapshot_state = mock.MagicMock(side_effect=RuntimeError("boom"))
        with inject_modules(skill_face_tracker=fake):
            self.assertIsNone(self.mod._user_at_desk())

    def test_no_sample_yet_returns_none(self):
        fake = types.ModuleType("skill_face_tracker")
        fake._snapshot_state = lambda: {"last_sample_at": 0}
        with inject_modules(skill_face_tracker=fake):
            self.assertIsNone(self.mod._user_at_desk())

    def test_unknown_monitor_returns_none(self):
        fake = types.ModuleType("skill_face_tracker")
        fake._snapshot_state = lambda: {"last_sample_at": time.time(),
                                        "current_monitor": "somewhere_else"}
        with inject_modules(skill_face_tracker=fake):
            self.assertIsNone(self.mod._user_at_desk())

    def test_left_monitor_is_present(self):
        fake = types.ModuleType("skill_face_tracker")
        fake._snapshot_state = lambda: {"last_sample_at": time.time(),
                                        "current_monitor": "left"}
        with inject_modules(skill_face_tracker=fake):
            self.assertIs(self.mod._user_at_desk(), True)


class DailySchedulerTests(unittest.TestCase):
    """_wait_for_presence + _fire_briefing + _scheduler_loop gating, mirroring
    the proven single-iteration driver from test_evening_briefing.py."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_briefing")

    # ── _wait_for_presence ────────────────────────────────────────────────
    def test_wait_for_presence_none_returns_false_fast(self):
        with mock.patch.object(self.mod, "_user_at_desk", return_value=None), \
             mock.patch.object(self.mod.time, "sleep") as slp:
            self.assertFalse(self.mod._wait_for_presence(120))
        slp.assert_not_called()

    def test_wait_for_presence_detects_user(self):
        with mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod.time, "sleep") as slp:
            self.assertTrue(self.mod._wait_for_presence(120))
        slp.assert_not_called()

    def test_wait_for_presence_times_out(self):
        reads = iter([False, False, False])
        with mock.patch.object(self.mod, "_user_at_desk",
                               side_effect=lambda: next(reads, False)), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=[1000.0, 1000.0, 2000.0]), \
             mock.patch.object(self.mod.time, "sleep"):
            self.assertFalse(self.mod._wait_for_presence(30))

    def test_wait_for_presence_appears_after_one_poll(self):
        reads = iter([False, False, True])
        with mock.patch.object(self.mod, "_user_at_desk",
                               side_effect=lambda: next(reads)), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=[1000.0, 1001.0, 1002.0]), \
             mock.patch.object(self.mod.time, "sleep") as slp:
            self.assertTrue(self.mod._wait_for_presence(30))
        slp.assert_called_once()

    # ── _fire_briefing ────────────────────────────────────────────────────
    def test_fire_briefing_pipeline(self):
        with mock.patch.object(self.mod, "_build_briefing",
                               return_value="Good morning, sir."), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_save_last_fired_date") as save:
            out = self.mod._fire_briefing("user-present")
        self.assertEqual(out, "Good morning, sir.")
        enq.assert_called_once_with("Good morning, sir.")
        save.assert_called_once()

    # ── _scheduler_loop single iteration ──────────────────────────────────
    def _run_one_iteration(self, **patches):
        """Run _scheduler_loop through exactly one body pass. The initial
        time.sleep(INITIAL_DELAY) is a no-op; the loop is broken at the TOP of
        the second pass by having _read_config raise _LoopBreak on call #2, so
        the whole first body (including trailing sleeps/continues) executes."""
        provided_cfg = patches.pop("_read_config", None)
        if provided_cfg is None:
            provided_cfg = mock.MagicMock(return_value={
                "enabled": True, "hour": 8, "minute": 0, "wait_min": 30})
        calls = {"n": 0}

        def _cfg_then_break(*a, **k):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise _LoopBreak
            return provided_cfg(*a, **k)

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(self.mod.time, "sleep", lambda *_a: None))
        stack.enter_context(mock.patch.object(self.mod, "_read_config", _cfg_then_break))
        for name, val in patches.items():
            if name == "now":
                continue
            stack.enter_context(mock.patch.object(self.mod, name, val))
        if "now" in patches:
            fake_dt = mock.MagicMock()
            fake_dt.now.return_value = patches["now"]
            stack.enter_context(mock.patch.object(self.mod.datetime, "datetime", fake_dt))
        with stack:
            with self.assertRaises(_LoopBreak):
                self.mod._scheduler_loop()

    def test_loop_disabled_skips(self):
        fire = mock.MagicMock()
        self._run_one_iteration(
            _read_config=mock.MagicMock(return_value={"enabled": False, "hour": 8,
                                                      "minute": 0, "wait_min": 30}),
            _fire_briefing=fire)
        fire.assert_not_called()

    def test_loop_already_fired_today_skips(self):
        today = datetime.date.today().isoformat()
        now = datetime.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        fire = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _load_last_fired_date=mock.MagicMock(return_value=today),
            _fire_briefing=fire)
        fire.assert_not_called()

    def test_loop_before_scheduled_skips(self):
        now = datetime.datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
        fire = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 8,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _fire_briefing=fire)
        fire.assert_not_called()

    def test_loop_past_catchup_marks_done(self):
        # Scheduled 08:00, now 11:00 → 180 min late (> 120) → mark done, no fire.
        now = datetime.datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)
        fire = mock.MagicMock()
        save = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 8,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _save_last_fired_date=save,
            _fire_briefing=fire)
        fire.assert_not_called()
        save.assert_called_once_with(now.date().isoformat())

    def test_loop_fires_when_user_present(self):
        now = datetime.datetime.now().replace(hour=8, minute=5, second=0, microsecond=0)
        fire = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 8,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _wait_for_presence=mock.MagicMock(return_value=True),
            _fire_briefing=fire)
        fire.assert_called_once_with("user-present")

    def test_loop_fires_timed_out_when_no_presence(self):
        now = datetime.datetime.now().replace(hour=8, minute=5, second=0, microsecond=0)
        fire = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 8,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _wait_for_presence=mock.MagicMock(return_value=False),
            _fire_briefing=fire)
        fire.assert_called_once_with("timed-out")

    def test_loop_wait_zero_fires_immediately(self):
        now = datetime.datetime.now().replace(hour=8, minute=5, second=0, microsecond=0)
        fire = mock.MagicMock()
        wait = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 8,
                                                      "minute": 0, "wait_min": 0}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _wait_for_presence=wait,
            _fire_briefing=fire)
        wait.assert_not_called()
        fire.assert_called_once_with("timed-out")

    def test_loop_swallows_body_exception(self):
        # An exception inside the body is logged, not propagated; the loop then
        # reaches the trailing sleep and the next _read_config raises _LoopBreak.
        self._run_one_iteration(
            _read_config=mock.MagicMock(side_effect=RuntimeError("config boom")))


class DailyRegisterTests(unittest.TestCase):
    def test_register_enabled_starts_thread(self):
        mod, _ = load_skill_isolated("daily_briefing", register=False)
        actions: dict = {}
        with mock.patch.object(mod, "_read_config",
                               return_value={"enabled": True, "hour": 8,
                                             "minute": 0, "wait_min": 30}), \
             mock.patch.object(mod.threading, "Thread") as Thread:
            mod.register(actions)
        self.assertIn("daily_briefing", actions)
        Thread.assert_called_once()
        self.assertTrue(Thread.call_args.kwargs.get("daemon"))
        Thread.return_value.start.assert_called_once()

    def test_register_disabled_skips_thread(self):
        mod, _ = load_skill_isolated("daily_briefing", register=False)
        actions: dict = {}
        with mock.patch.object(mod, "_read_config",
                               return_value={"enabled": False, "hour": 8,
                                             "minute": 0, "wait_min": 30}), \
             mock.patch.object(mod.threading, "Thread") as Thread:
            mod.register(actions)
        self.assertIn("daily_briefing", actions)   # action still registered
        Thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
