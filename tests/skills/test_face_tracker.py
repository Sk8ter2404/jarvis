"""Logic tests for skills/face_tracker.py.

face_tracker builds a higher-level gaze layer over bobert_companion's raw
per-camera last-seen timestamps. Tests cover the deterministic core:
  • side classification from CAMERAS look_x presets + freshness window,
  • monitor-name mapping (incl. the ambiguous "both sides" → middle/top case),
  • dwell-stat accumulation in _commit_state,
  • duration / monitor phrasing helpers,
  • the read-failure spike detector consumed by self_diagnostic,
  • the gaze_status / gaze_stats / which_monitor-wrapper actions.

A fake bobert_companion is injected into sys.modules so _classify_sides and
the wrappers resolve their CAMERAS / MONITORS / camera state without the
monolith. The poller thread is neutered by the harness (Thread.start no-op).
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_bc(cameras=None, monitors=None, last_seen=None):
    bc = mock.MagicMock()
    bc.CAMERAS = cameras if cameras is not None else [
        {"index": 0, "look_x": 0.2},   # left camera
        {"index": 1, "look_x": 0.8},   # right camera
    ]
    bc.MONITORS = monitors if monitors is not None else {
        "left": (0, 0, 1920, 1080), "right": (1920, 0, 1920, 1080),
        "middle": (3840, 0, 1920, 1080), "top": (1920, -1080, 1920, 1080),
    }
    bc._camera_last_seen = last_seen if last_seen is not None else {}
    bc._camera_state_lock = threading.Lock()
    return bc


class FaceTrackerGeometryTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("face_tracker")
        # Reset module-global state so each test is independent.
        self.mod._pending_monitor.clear()
        self.mod._dwell_total.clear()
        self.mod._dwell_longest.clear()
        self.mod._face_visible_total[0] = 0.0
        self.mod._state.update({
            "current_monitor": None, "current_sides": None, "last_sample_at": 0.0,
            "monitor_since": 0.0, "face_visible": False, "last_face_at": 0.0,
            "first_face_at": 0.0,
        })

    # ── _classify_sides ──────────────────────────────────────────────────
    def test_classify_sides_fresh_left_only(self):
        now = time.time()
        bc = _fake_bc(last_seen={0: now, 1: now - 100})  # right is stale
        sides, side_map = self.mod._classify_sides(bc)
        self.assertEqual(sides, frozenset({"left"}))
        self.assertEqual(side_map, {0: "left", 1: "right"})

    def test_classify_sides_both_fresh(self):
        now = time.time()
        bc = _fake_bc(last_seen={0: now, 1: now})
        sides, _ = self.mod._classify_sides(bc)
        self.assertEqual(sides, frozenset({"left", "right"}))

    def test_classify_sides_none_when_all_stale(self):
        old = time.time() - 999
        bc = _fake_bc(last_seen={0: old, 1: old})
        sides, _ = self.mod._classify_sides(bc)
        self.assertEqual(sides, frozenset())

    # ── _monitor_name_from_sides ─────────────────────────────────────────
    def test_monitor_name_left(self):
        bc = _fake_bc()
        self.assertEqual(self.mod._monitor_name_from_sides(bc, frozenset({"left"})), "left")

    def test_monitor_name_away_when_empty(self):
        bc = _fake_bc()
        self.assertEqual(self.mod._monitor_name_from_sides(bc, frozenset()), "away")

    def test_monitor_name_both_sides_is_middle_or_top(self):
        bc = _fake_bc()  # has both 'middle' and 'top'
        self.assertEqual(
            self.mod._monitor_name_from_sides(bc, frozenset({"left", "right"})),
            "middle_or_top")

    def test_monitor_name_both_sides_only_top(self):
        bc = _fake_bc(monitors={"left": (0, 0, 1, 1), "right": (1, 0, 1, 1),
                                "top": (0, -1, 1, 1)})
        self.assertEqual(
            self.mod._monitor_name_from_sides(bc, frozenset({"left", "right"})), "top")

    def test_monitor_name_both_sides_defaults_to_middle(self):
        bc = _fake_bc(monitors={"left": (0, 0, 1, 1), "right": (1, 0, 1, 1)})
        self.assertEqual(
            self.mod._monitor_name_from_sides(bc, frozenset({"left", "right"})), "middle")

    # ── _commit_state dwell math ─────────────────────────────────────────
    def test_commit_state_accumulates_dwell_on_change(self):
        # Start on left at t=100, switch to right at t=130 → 30s dwell on left.
        self.mod._commit_state("left", frozenset({"left"}), 100.0)
        self.mod._commit_state("right", frozenset({"right"}), 130.0)
        self.assertAlmostEqual(self.mod._dwell_total["left"], 30.0)
        self.assertAlmostEqual(self.mod._dwell_longest["left"], 30.0)
        self.assertEqual(self.mod._state["current_monitor"], "right")

    def test_commit_state_away_not_counted_in_dwell(self):
        self.mod._commit_state("away", frozenset(), 100.0)
        self.mod._commit_state("left", frozenset({"left"}), 150.0)
        # The 'away' run shouldn't appear in dwell totals.
        self.assertNotIn("away", self.mod._dwell_total)

    def test_commit_state_same_monitor_no_double_count(self):
        self.mod._commit_state("left", frozenset({"left"}), 100.0)
        self.mod._commit_state("left", frozenset({"left"}), 130.0)  # same → just touch
        self.assertNotIn("left", self.mod._dwell_total)  # run not closed yet

    # ── _format_seconds ──────────────────────────────────────────────────
    def test_format_seconds_units(self):
        f = self.mod._format_seconds
        self.assertEqual(f(1), "1 second")
        self.assertEqual(f(45), "45 seconds")
        self.assertEqual(f(90), "1 minute")
        self.assertEqual(f(3660), "1 hour 1 minute")

    # ── _monitor_phrase ──────────────────────────────────────────────────
    def test_monitor_phrase_variants(self):
        p = self.mod._monitor_phrase
        self.assertEqual(p(None), "not yet established")
        self.assertIn("not visible", p("away"))
        self.assertIn("middle or top", p("middle_or_top"))
        self.assertEqual(p("left"), "the left monitor")


class FaceTrackerSpikeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("face_tracker")

    def test_spike_detected_for_live_consecutive_fails(self):
        info = {0: {"consecutive_fails": 7, "max_consecutive_fails": 7,
                    "last_error": "grab() returned None", "last_ok_at": time.time()}}
        with mock.patch.object(self.mod, "get_consecutive_read_failures",
                               return_value=info):
            signals = self.mod.get_read_failure_spike_signals(threshold=5)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["cam_index"], 0)
        self.assertEqual(signals[0]["consecutive_fails"], 7)
        self.assertIn("grab()", signals[0]["last_error"])

    def test_spike_historic_unrecovered(self):
        # consec below threshold but peak high AND last_ok long ago → historic spike.
        info = {1: {"consecutive_fails": 1, "max_consecutive_fails": 9,
                    "last_error": "timeout", "last_ok_at": time.time() - 600}}
        with mock.patch.object(self.mod, "get_consecutive_read_failures",
                               return_value=info):
            signals = self.mod.get_read_failure_spike_signals(threshold=5)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["max_consecutive_fails"], 9)

    def test_no_spike_when_healthy(self):
        info = {0: {"consecutive_fails": 0, "max_consecutive_fails": 1,
                    "last_error": None, "last_ok_at": time.time()}}
        with mock.patch.object(self.mod, "get_consecutive_read_failures",
                               return_value=info):
            self.assertEqual(self.mod.get_read_failure_spike_signals(threshold=5), [])

    def test_spike_empty_when_bc_absent(self):
        with mock.patch.object(self.mod, "get_consecutive_read_failures", return_value={}):
            self.assertEqual(self.mod.get_read_failure_spike_signals(), [])


class FaceTrackerActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("face_tracker")
        self.mod._dwell_total.clear()
        self.mod._face_visible_total[0] = 0.0
        self.mod._state.update({
            "current_monitor": None, "current_sides": None, "last_sample_at": 0.0,
            "monitor_since": 0.0, "face_visible": False, "last_face_at": 0.0,
            "first_face_at": 0.0,
        })

    def test_gaze_status_warming_up(self):
        self.assertIn("warming up", self.actions["gaze_status"](""))

    def test_gaze_status_reports_monitor_and_dwell(self):
        now = time.time()
        self.mod._state.update({"last_sample_at": now, "current_monitor": "right",
                                "monitor_since": now - 120})
        out = self.actions["gaze_status"]("")
        self.assertIn("right monitor", out)
        self.assertIn("2 minutes", out)

    def test_gaze_status_away_reports_last_seen(self):
        now = time.time()
        self.mod._state.update({"last_sample_at": now, "current_monitor": "away",
                                "last_face_at": now - 30})
        out = self.actions["gaze_status"]("")
        self.assertIn("not currently in view", out)

    def test_gaze_stats_no_history(self):
        self.assertIn("no gaze history", self.actions["gaze_stats"]("").lower())

    def test_gaze_stats_ranks_monitors(self):
        self.mod._dwell_total.update({"left": 300.0, "right": 60.0})
        out = self.actions["gaze_stats"]("")
        self.assertIn("left monitor", out)
        # left (5 min) should be named as the most-watched.
        self.assertIn("Most of your attention", out)

    def test_face_track_status_is_gaze_status(self):
        # Alias returns the same warming-up message.
        self.assertEqual(self.actions["face_track_status"](""), self.actions["gaze_status"](""))

    # ── which_monitor fast-path wrapper ──────────────────────────────────
    def test_which_monitor_wrapper_fast_path(self):
        bc = _fake_bc()
        called = {"n": 0}

        def original(arg=""):
            called["n"] += 1
            return "ORIGINAL"

        wrapped = self.mod._build_which_monitor_wrapper(original)
        now = time.time()
        self.mod._state.update({"last_sample_at": now, "current_monitor": "left"})
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            out = wrapped("")
        self.assertIn("LEFT monitor", out)
        self.assertEqual(called["n"], 0)  # fast-path skipped the original

    def test_which_monitor_wrapper_delegates_when_ambiguous(self):
        def original(arg=""):
            return "DELEGATED"

        wrapped = self.mod._build_which_monitor_wrapper(original)
        now = time.time()
        self.mod._state.update({"last_sample_at": now,
                                "current_monitor": "middle_or_top"})
        self.assertEqual(wrapped(""), "DELEGATED")

    def test_which_monitor_wrapper_delegates_when_stale(self):
        def original(arg=""):
            return "DELEGATED"

        wrapped = self.mod._build_which_monitor_wrapper(original)
        # last_sample_at far in the past → not fresh → delegate.
        self.mod._state.update({"last_sample_at": time.time() - 999,
                                "current_monitor": "left"})
        self.assertEqual(wrapped(""), "DELEGATED")

    def test_see_user_wrapper_appends_gaze_note(self):
        def original(arg=""):
            return "I see you at your desk."

        wrapped = self.mod._build_see_user_wrapper(original)
        self.mod._state.update({"current_monitor": "right"})
        out = wrapped("")
        self.assertIn("I see you at your desk.", out)
        self.assertIn("[gaze: currently looking at the right monitor]", out)

    def test_gaze_status_monitor_none_after_sample(self):
        # A sample landed but no monitor committed yet → "haven't established".
        self.mod._state.update({"last_sample_at": time.time(),
                                "current_monitor": None})
        self.assertIn("haven't established", self.actions["gaze_status"](""))

    def test_gaze_status_away_never_seen(self):
        self.mod._state.update({"last_sample_at": time.time(),
                                "current_monitor": "away", "last_face_at": 0.0})
        self.assertIn("haven't seen you at all", self.actions["gaze_status"](""))

    def test_gaze_stats_includes_current_run_and_face_time(self):
        now = time.time()
        self.mod._dwell_total.update({"left": 60.0})
        # A live run on 'right' should be folded into totals, and the
        # face-visible total surfaces the "in view for roughly..." clause.
        self.mod._state.update({"current_monitor": "right",
                                "monitor_since": now - 120})
        self.mod._face_visible_total[0] = 300.0
        out = self.actions["gaze_stats"]("")
        self.assertIn("in view for roughly", out)
        self.assertIn("right", out)

    def test_which_monitor_fast_path_away(self):
        wrapped = self.mod._build_which_monitor_wrapper(lambda a="": "ORIG")
        self.mod._state.update({"last_sample_at": time.time(),
                                "current_monitor": "away"})
        self.assertIn("not visible", wrapped(""))

    def test_which_monitor_fast_path_import_failure_drops_suffix(self):
        # bc import raising → monitors={} → no "(left)" suffix, still fast-path.
        wrapped = self.mod._build_which_monitor_wrapper(lambda a="": "ORIG")
        self.mod._state.update({"last_sample_at": time.time(),
                                "current_monitor": "left"})
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            out = wrapped("")
        self.assertIn("LEFT monitor", out)
        self.assertNotIn("(left)", out)  # suffix dropped without MONITORS

    def test_see_user_wrapper_away_note(self):
        wrapped = self.mod._build_see_user_wrapper(lambda a="": "base")
        self.mod._state.update({"current_monitor": "away"})
        out = wrapped("")
        self.assertIn("[gaze: user not currently in view]", out)

    def test_see_user_wrapper_no_note_when_unknown(self):
        wrapped = self.mod._build_see_user_wrapper(lambda a="": "base")
        self.mod._state.update({"current_monitor": None})
        self.assertEqual(wrapped(""), "base")


class FaceTrackerClassifyNoLockTests(unittest.TestCase):
    """_classify_sides else-branch when bobert exposes no state lock."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("face_tracker")

    def test_classify_without_lock(self):
        now = time.time()
        bc = _fake_bc(last_seen={0: now, 1: now - 999})
        bc._camera_state_lock = None  # exercise the lock-less path (94-98)
        sides, _ = self.mod._classify_sides(bc)
        self.assertEqual(sides, frozenset({"left"}))

    def test_monitor_name_right_only(self):
        bc = _fake_bc()
        self.assertEqual(
            self.mod._monitor_name_from_sides(bc, frozenset({"right"})), "right")


class FaceTrackerPollOnceTests(unittest.TestCase):
    """_poll_once — the per-tick poller body (hysteresis + state/dwell/face
    accounting), driven directly with a fake bobert_companion."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("face_tracker")
        self.mod._pending_monitor.clear()
        self.mod._dwell_total.clear()
        self.mod._dwell_longest.clear()
        self.mod._face_visible_total[0] = 0.0
        self.mod._state.update({
            "current_monitor": None, "current_sides": None, "last_sample_at": 0.0,
            "monitor_since": 0.0, "face_visible": False, "last_face_at": 0.0,
            "first_face_at": 0.0,
        })

    def test_unstable_first_read_does_not_commit(self):
        # One read with HYSTERESIS_SAMPLES=2 is not yet stable → no commit, but
        # current_sides + last_sample_at are touched.
        now = time.time()
        bc = _fake_bc(last_seen={0: now})  # left only
        self.mod._poll_once(bc)
        self.assertIsNone(self.mod._state["current_monitor"])  # not committed
        self.assertEqual(self.mod._state["current_sides"], frozenset({"left"}))
        self.assertTrue(self.mod._state["face_visible"])
        self.assertTrue(self.mod._state["first_face_at"])

    def test_two_identical_reads_commit_monitor(self):
        now = time.time()
        bc = _fake_bc(last_seen={0: now})  # left only, fresh
        self.mod._poll_once(bc)
        self.mod._poll_once(bc)  # second identical read → stable → commit
        self.assertEqual(self.mod._state["current_monitor"], "left")

    def test_face_visible_total_accumulates_across_ticks(self):
        now = time.time()
        bc = _fake_bc(last_seen={0: now, 1: now})  # both fresh → visible
        # First tick arms first_face_at + last_sample_at.
        self.mod._poll_once(bc)
        # Backdate last_sample_at so the next tick credits ~5s of visible time.
        self.mod._state["last_sample_at"] = time.time() - 5
        self.mod._poll_once(bc)
        self.assertGreaterEqual(self.mod._face_visible_total[0], 4.0)

    def test_poll_once_drops_to_away_when_face_lost(self):
        old = time.time() - 999
        bc = _fake_bc(last_seen={0: old, 1: old})  # nothing fresh
        self.mod._poll_once(bc)
        self.mod._poll_once(bc)  # 'away' held two ticks → stable
        self.assertEqual(self.mod._state["current_monitor"], "away")
        self.assertFalse(self.mod._state["face_visible"])

    def test_pending_buffer_trimmed_to_hysteresis_window(self):
        # Polling more than HYSTERESIS_SAMPLES times must trim the buffer back
        # to the window length (covers the del-slice at line 158).
        now = time.time()
        bc = _fake_bc(last_seen={0: now})  # left only, fresh each tick
        for _ in range(self.mod.HYSTERESIS_SAMPLES + 3):
            self.mod._poll_once(bc)
        self.assertEqual(len(self.mod._pending_monitor),
                         self.mod.HYSTERESIS_SAMPLES)


class FaceTrackerFailureSummaryTests(unittest.TestCase):
    """get_consecutive_read_failures — bc import + delegation paths."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("face_tracker")

    def test_returns_empty_when_bc_import_fails(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.assertEqual(self.mod.get_consecutive_read_failures(), {})

    def test_returns_empty_when_summary_fn_missing(self):
        bc = mock.MagicMock(spec=[])  # no get_camera_failure_summary attr
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod.get_consecutive_read_failures(), {})

    def test_delegates_to_summary_fn(self):
        bc = mock.MagicMock()
        bc.get_camera_failure_summary.return_value = {0: {"consecutive_fails": 3}}
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            out = self.mod.get_consecutive_read_failures()
        self.assertEqual(out, {0: {"consecutive_fails": 3}})

    def test_summary_fn_raising_returns_empty(self):
        bc = mock.MagicMock()
        bc.get_camera_failure_summary.side_effect = RuntimeError("boom")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod.get_consecutive_read_failures(), {})

    def test_spike_skips_unparseable_counts(self):
        # A record whose counts can't be int()'d is skipped (covers 271-272).
        info = {0: {"consecutive_fails": object(), "max_consecutive_fails": 9,
                    "last_ok_at": time.time() - 600}}
        with mock.patch.object(self.mod, "get_consecutive_read_failures",
                               return_value=info):
            self.assertEqual(self.mod.get_read_failure_spike_signals(threshold=5), [])


class FaceTrackerRegisterTests(unittest.TestCase):
    """register(): wrapper installation + duplicate-poller guard."""

    def test_register_wraps_existing_actions_and_starts_poller(self):
        import contextlib
        import io
        mod, _ = load_skill_isolated("face_tracker", register=False)
        sentinel_wm = lambda a="": "WM"
        sentinel_su = lambda a="": "SU"
        actions = {"which_monitor": sentinel_wm, "see_user": sentinel_su}
        buf = io.StringIO()
        with mock.patch.object(threading.Thread, "start", lambda self: None), \
             contextlib.redirect_stdout(buf):
            mod.register(actions)
        # Both got wrapped (identity changed) and the new actions registered.
        self.assertIsNot(actions["which_monitor"], sentinel_wm)
        self.assertIsNot(actions["see_user"], sentinel_su)
        self.assertIn("gaze_status", actions)
        self.assertIn("gaze poller active", buf.getvalue())

    def test_register_skips_duplicate_poller(self):
        import contextlib
        import io
        mod, _ = load_skill_isolated("face_tracker", register=False)
        # Simulate an already-running poller of the same OS-thread name.
        live = mock.MagicMock()
        live.name = "face-tracker-skill"
        live.is_alive.return_value = True
        buf = io.StringIO()
        with mock.patch.object(threading, "enumerate", return_value=[live]), \
             mock.patch.object(threading.Thread, "start", lambda self: None), \
             contextlib.redirect_stdout(buf):
            mod.register({})
        self.assertIn("already running", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
