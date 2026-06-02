"""Logic tests for skills/screen_watch.py.

screen_watch fires a gentle stretch nudge after a long single-window idle
session. Tests cover the deterministic gate logic and the poll state machine:
  • _title_is_ignored (HUD / lock screen / game substrings),
  • _fmt_minutes formatting,
  • _is_sleeping_or_standby + _user_is_away gates (reading injected modules),
  • _poll_once: identity-change resets the stare timer; the nudge fires only
    when stare + idle thresholds are met AND no gate blocks; cooldown
    suppresses a repeat for the same window-identity,
  • the screen_watch_status action's gate readout.

mss / pygetwindow / Win32 idle and the speech enqueue are all mocked — no
real screenshots, no window queries, no pending_speech.json writes.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


_SENTINEL = object()


def _inject(name, obj):
    """mock.patch.dict-friendly single-module injector returning a ctx mgr that
    also restores absence. Mirrors the isolation contract: install in the with,
    restore (including prior absence) on exit."""
    return mock.patch.dict(sys.modules, {name: obj})


class ScreenWatchHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")
        self._reset()

    def _reset(self):
        self.mod._current_identity[0] = None
        self.mod._stare_started_at[0] = 0.0
        self.mod._last_nudge_at[0] = 0.0
        self.mod._last_nudged_id[0] = None

    # ── _title_is_ignored ────────────────────────────────────────────────
    def test_title_ignored_hud_and_lockscreen(self):
        self.assertTrue(self.mod._title_is_ignored("J.A.R.V.I.S HUD"))
        self.assertTrue(self.mod._title_is_ignored("Windows Default Lock Screen"))
        self.assertTrue(self.mod._title_is_ignored("Program Manager"))

    def test_title_not_ignored_for_normal_window(self):
        self.assertFalse(self.mod._title_is_ignored("report.docx - Word"))

    # ── _fmt_minutes ─────────────────────────────────────────────────────
    def test_fmt_minutes(self):
        f = self.mod._fmt_minutes
        self.assertEqual(f(30), "30s")
        self.assertEqual(f(90), "1m 30s")
        self.assertEqual(f(3700), "1h 1m")

    # ── gate helpers ─────────────────────────────────────────────────────
    def test_is_sleeping_reads_bobert_flags(self):
        bc = mock.MagicMock()
        bc._sleep_mode = [True]
        bc._standby_mode = [False]
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertTrue(self.mod._is_sleeping_or_standby())

    def test_is_sleeping_false_when_bc_absent(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("bobert_companion", None)
            self.assertFalse(self.mod._is_sleeping_or_standby())

    def test_user_is_away_true_only_when_tracker_says_away(self):
        ft = mock.MagicMock()
        ft._snapshot_state.return_value = {"last_sample_at": time.time(),
                                           "current_monitor": "away"}
        with mock.patch.dict(sys.modules, {"skill_face_tracker": ft}):
            self.assertTrue(self.mod._user_is_away())

    def test_user_is_away_false_when_looking_at_monitor(self):
        ft = mock.MagicMock()
        ft._snapshot_state.return_value = {"last_sample_at": time.time(),
                                           "current_monitor": "left"}
        with mock.patch.dict(sys.modules, {"skill_face_tracker": ft}):
            self.assertFalse(self.mod._user_is_away())

    def test_user_is_away_false_when_tracker_unestablished(self):
        # last_sample_at == 0 → tracker hasn't established gaze → don't suppress.
        ft = mock.MagicMock()
        ft._snapshot_state.return_value = {"last_sample_at": 0.0, "current_monitor": None}
        with mock.patch.dict(sys.modules, {"skill_face_tracker": ft}):
            self.assertFalse(self.mod._user_is_away())

    def test_user_is_away_false_when_face_tracker_not_loaded(self):
        sys.modules.pop("skill_face_tracker", None)
        self.assertFalse(self.mod._user_is_away())


class ScreenWatchPollTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")
        self.mod._current_identity[0] = None
        self.mod._stare_started_at[0] = 0.0
        self.mod._last_nudge_at[0] = 0.0
        self.mod._last_nudged_id[0] = None
        self.threshold = self.mod.STARE_THRESHOLD_SECONDS

    def _patch_focus(self, title="report.docx - Word", h="hash123"):
        """Patch the window + hash helpers so _poll_once sees a stable window."""
        return (
            mock.patch.object(self.mod, "_get_focused_window",
                              return_value=(title, (0, 0, 800, 600))),
            mock.patch.object(self.mod, "_hash_window_thumbnail", return_value=h),
        )

    def test_poll_first_sight_sets_stare_timer(self):
        p1, p2 = self._patch_focus()
        with p1, p2:
            self.mod._poll_once()
        self.assertEqual(self.mod._current_identity[0], ("report.docx - Word", "hash123"))
        self.assertGreater(self.mod._stare_started_at[0], 0.0)

    def test_poll_no_window_clears_identity(self):
        self.mod._current_identity[0] = ("old", "h")
        self.mod._stare_started_at[0] = 100.0
        with mock.patch.object(self.mod, "_get_focused_window", return_value=(None, None)):
            self.mod._poll_once()
        self.assertIsNone(self.mod._current_identity[0])
        self.assertEqual(self.mod._stare_started_at[0], 0.0)

    def test_poll_ignored_title_clears_identity(self):
        with mock.patch.object(self.mod, "_get_focused_window",
                               return_value=("JARVIS HUD", (0, 0, 800, 600))):
            self.mod._poll_once()
        self.assertIsNone(self.mod._current_identity[0])

    def test_poll_fires_nudge_when_all_gates_clear(self):
        # Pre-seed an established stare older than the threshold.
        ident = ("report.docx - Word", "hash123")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        p1, p2 = self._patch_focus()
        with p1, p2, \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds",
                               return_value=self.threshold + 60), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_called_once()
        self.assertIn("stretch", enq.call_args[0][0].lower())

    def test_poll_suppressed_when_idle_below_threshold(self):
        # User IS actively using the window (idle low) → no nudge.
        ident = ("report.docx - Word", "hash123")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        p1, p2 = self._patch_focus()
        with p1, p2, \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds", return_value=5.0), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    def test_poll_suppressed_when_sleeping(self):
        ident = ("report.docx - Word", "hash123")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        p1, p2 = self._patch_focus()
        with p1, p2, \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=True), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    def test_poll_cooldown_suppresses_repeat(self):
        ident = ("report.docx - Word", "hash123")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        # Same identity nudged 1 minute ago → within the 1h cooldown.
        self.mod._last_nudged_id[0] = ident
        self.mod._last_nudge_at[0] = time.time() - 60
        p1, p2 = self._patch_focus()
        with p1, p2, \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds",
                               return_value=self.threshold + 60), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    # ── screen_watch_status action ───────────────────────────────────────
    def test_status_no_window(self):
        self.mod._current_identity[0] = None
        self.assertIn("haven't established", self.actions["screen_watch_status"](""))

    def test_status_lists_open_gates(self):
        self.mod._current_identity[0] = ("report.docx - Word", "h")
        self.mod._stare_started_at[0] = time.time() - 120  # only 2 min stare
        with mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds", return_value=10.0):
            out = self.actions["screen_watch_status"]("")
        self.assertIn("report.docx - Word", out)
        self.assertIn("stare only", out)   # 2m < 25m threshold
        self.assertIn("idle only", out)

    def test_status_all_gates_clear(self):
        # Stare + idle both above threshold, awake, not away → "all gates clear".
        self.mod._current_identity[0] = ("paper.pdf", "h")
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 120)
        with mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds",
                               return_value=self.threshold + 120):
            out = self.actions["screen_watch_status"]("")
        self.assertIn("all gates clear", out)

    def test_status_reports_sleep_and_away_gates(self):
        # Stare/idle long enough that only the sleep + away gates show up.
        self.mod._current_identity[0] = ("game.exe", "h")
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 120)
        with mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=True), \
             mock.patch.object(self.mod, "_user_is_away", return_value=True), \
             mock.patch.object(self.mod, "_get_system_idle_seconds",
                               return_value=self.threshold + 120):
            out = self.actions["screen_watch_status"]("")
        self.assertIn("sleep mode", out)
        self.assertIn("user away", out)

    def test_status_started_zero_stare(self):
        # _stare_started_at == 0 → stare computed as 0.0 (the `else 0.0` branch).
        self.mod._current_identity[0] = ("x", "h")
        self.mod._stare_started_at[0] = 0.0
        with mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds", return_value=1.0):
            out = self.actions["screen_watch_status"]("")
        self.assertIn("0s", out)


# ── _poll_once additional branches ───────────────────────────────────────
class PollOnceBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")
        self.mod._current_identity[0] = None
        self.mod._stare_started_at[0] = 0.0
        self.mod._last_nudge_at[0] = 0.0
        self.mod._last_nudged_id[0] = None
        self.threshold = self.mod.STARE_THRESHOLD_SECONDS

    def test_poll_hash_none_leaves_state_untouched(self):
        # Capture (hash) fails → state is left exactly as it was, no nudge.
        prev = ("paper.pdf", "oldhash")
        self.mod._current_identity[0] = prev
        self.mod._stare_started_at[0] = 123.0
        with mock.patch.object(self.mod, "_get_focused_window",
                               return_value=("paper.pdf", (0, 0, 800, 600))), \
             mock.patch.object(self.mod, "_hash_window_thumbnail", return_value=None):
            self.mod._poll_once()
        self.assertEqual(self.mod._current_identity[0], prev)
        self.assertEqual(self.mod._stare_started_at[0], 123.0)

    def test_poll_stare_below_threshold_no_nudge(self):
        # Same identity but stare just under threshold → returns before gates.
        ident = ("doc", "h")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold - 30)
        with mock.patch.object(self.mod, "_get_focused_window",
                               return_value=("doc", (0, 0, 800, 600))), \
             mock.patch.object(self.mod, "_hash_window_thumbnail", return_value="h"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    def test_poll_suppressed_when_user_away(self):
        ident = ("doc", "h")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        with mock.patch.object(self.mod, "_get_focused_window",
                               return_value=("doc", (0, 0, 800, 600))), \
             mock.patch.object(self.mod, "_hash_window_thumbnail", return_value="h"), \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=True), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    def test_poll_fires_after_cooldown_for_different_identity(self):
        # A DIFFERENT identity than the last nudged one fires even though the
        # last nudge was recent (cooldown keys on identity).
        ident = ("newdoc", "h2")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        self.mod._last_nudged_id[0] = ("olddoc", "h1")
        self.mod._last_nudge_at[0] = time.time() - 5   # recent, but other id
        with mock.patch.object(self.mod, "_get_focused_window",
                               return_value=("newdoc", (0, 0, 800, 600))), \
             mock.patch.object(self.mod, "_hash_window_thumbnail", return_value="h2"), \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds",
                               return_value=self.threshold + 60), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_called_once()
        self.assertEqual(self.mod._last_nudged_id[0], ident)


# ── gate helper edge / exception paths ───────────────────────────────────
class GateEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")

    def test_is_sleeping_swallows_bad_flag_shape(self):
        # _sleep_mode present but not indexable → except → False.
        bc = types.SimpleNamespace(_sleep_mode=None, _standby_mode=None)
        with _inject("bobert_companion", bc):
            self.assertFalse(self.mod._is_sleeping_or_standby())

    def test_is_sleeping_standby_true(self):
        bc = types.SimpleNamespace(_sleep_mode=[False], _standby_mode=[True])
        with _inject("bobert_companion", bc):
            self.assertTrue(self.mod._is_sleeping_or_standby())

    def test_user_away_no_snapshot_func(self):
        # Module present but without _snapshot_state → False.
        ft = types.SimpleNamespace()
        with _inject("skill_face_tracker", ft):
            self.assertFalse(self.mod._user_is_away())

    def test_user_away_snapshot_raises(self):
        ft = types.SimpleNamespace(
            _snapshot_state=mock.MagicMock(side_effect=RuntimeError("boom")))
        with _inject("skill_face_tracker", ft):
            self.assertFalse(self.mod._user_is_away())


# ── _enqueue_speech (announcer path + atomic-write fallback) ──────────────
class EnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")
        self.tmp = tempfile.mkdtemp(prefix="screenwatch_test_")
        self.queue = os.path.join(self.tmp, "pending_speech.json")
        self._saved_queue = self.mod._SPEECH_QUEUE
        self.mod._SPEECH_QUEUE = self.queue
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        self.mod._SPEECH_QUEUE = self._saved_queue
        for n in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, n))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def test_routes_through_proactive_announce(self):
        # When bobert_companion exposes proactive_announce, the skill uses it and
        # never touches the file queue.
        announcer = mock.MagicMock()
        bc = types.SimpleNamespace(proactive_announce=announcer)
        with _inject("bobert_companion", bc):
            self.mod._enqueue_speech("stretch please")
        announcer.assert_called_once_with("stretch please", source="screen_watch")
        self.assertFalse(os.path.exists(self.queue))

    def test_falls_back_to_atomic_write_when_no_announcer(self):
        # bobert present but no proactive_announce → local atomic write to the
        # (redirected) queue file. New file starts empty → one entry appended.
        bc = types.SimpleNamespace()   # no proactive_announce attr
        with _inject("bobert_companion", bc):
            self.mod._enqueue_speech("hello there")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["message"], "hello there")
        self.assertIn("ts", data[0])

    def test_fallback_appends_to_existing_queue(self):
        with open(self.queue, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        bc = types.SimpleNamespace()
        with _inject("bobert_companion", bc):
            self.mod._enqueue_speech("new")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["old", "new"])

    def test_fallback_recovers_from_corrupt_queue(self):
        with open(self.queue, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        bc = types.SimpleNamespace()
        with _inject("bobert_companion", bc):
            self.mod._enqueue_speech("fresh")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        # Corrupt content discarded → only the new message remains.
        self.assertEqual([d["message"] for d in data], ["fresh"])

    def test_announcer_raises_falls_through_to_write(self):
        # proactive_announce raises → caught → falls through to file write.
        bc = types.SimpleNamespace(
            proactive_announce=mock.MagicMock(side_effect=RuntimeError("x")))
        with _inject("bobert_companion", bc):
            self.mod._enqueue_speech("after-error")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[0]["message"], "after-error")

    def test_import_failure_falls_through_to_write(self):
        # importlib.import_module raising (no parent module) → fallback write.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")):
            self.mod._enqueue_speech("imp-fail")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[0]["message"], "imp-fail")

    def test_atomic_write_failure_is_swallowed(self):
        # The atomic write itself fails (read-only share simulation) → the error
        # is caught and printed; _enqueue_speech does not raise.
        bc = types.SimpleNamespace()
        with _inject("bobert_companion", bc), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only fs")):
            self.mod._enqueue_speech("doomed")   # must not raise


# ── _get_system_idle_seconds (Win32 ctypes) ──────────────────────────────
class IdleSecondsTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")

    def _fake_ctypes(self, last_input_ms, tick_ms, get_ok=True):
        """Patch ctypes.windll so GetLastInputInfo/GetTickCount are deterministic.
        The function imports names FROM ctypes inside the body, so we patch the
        real ctypes module's attributes (windll, plus the primitives it uses)."""
        import ctypes as _ct

        user32 = mock.MagicMock()
        user32.GetLastInputInfo.return_value = 1 if get_ok else 0
        kernel32 = mock.MagicMock()
        kernel32.GetTickCount.return_value = tick_ms
        windll = types.SimpleNamespace(user32=user32, kernel32=kernel32)

        # The Structure subclass sets info.dwTime; emulate by having byref read it
        # back. Simplest: make GetLastInputInfo populate the struct's dwTime.
        def _glii(ref):
            ref._obj.dwTime = last_input_ms
            return 1 if get_ok else 0
        user32.GetLastInputInfo.side_effect = _glii

        patches = [
            mock.patch.object(_ct, "windll", windll, create=True),
        ]
        return patches

    def test_idle_seconds_computed(self):
        # tick=10000ms, last input=4000ms → 6000ms idle → 6.0s.
        patches = self._fake_ctypes(last_input_ms=4000, tick_ms=10000)
        with patches[0]:
            out = self.mod._get_system_idle_seconds()
        self.assertAlmostEqual(out, 6.0, places=3)

    def test_idle_seconds_zero_when_call_fails(self):
        # GetLastInputInfo returns 0 → function returns 0.0.
        patches = self._fake_ctypes(last_input_ms=0, tick_ms=10000, get_ok=False)
        with patches[0]:
            self.assertEqual(self.mod._get_system_idle_seconds(), 0.0)

    def test_idle_seconds_negative_clamped(self):
        # Tick wrapped (tick < last input) → millis < 0 → clamped to 0.0.
        patches = self._fake_ctypes(last_input_ms=10000, tick_ms=5000)
        with patches[0]:
            self.assertEqual(self.mod._get_system_idle_seconds(), 0.0)

    def test_idle_seconds_swallows_exception(self):
        # windll present but raises on attribute access (mirrors the AttributeError
        # a non-Windows host hits when `from ctypes import ... windll` resolves but
        # user32 is absent) → the broad except returns 0.0.
        import ctypes as _ct

        class _Boom:
            @property
            def user32(self):
                raise OSError("no user32")
        with mock.patch.object(_ct, "windll", _Boom(), create=True):
            self.assertEqual(self.mod._get_system_idle_seconds(), 0.0)


# ── _get_focused_window (pygetwindow) ────────────────────────────────────
class FocusedWindowTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")

    def _gw(self, win):
        gw = types.ModuleType("pygetwindow")
        gw.getActiveWindow = mock.MagicMock(return_value=win)
        return gw

    def test_pygetwindow_missing_returns_none(self):
        with _inject("pygetwindow", None):
            self.assertEqual(self.mod._get_focused_window(), (None, None))

    def test_get_active_window_raises(self):
        gw = types.ModuleType("pygetwindow")
        gw.getActiveWindow = mock.MagicMock(side_effect=RuntimeError("no x11"))
        with _inject("pygetwindow", gw):
            self.assertEqual(self.mod._get_focused_window(), (None, None))

    def test_no_active_window(self):
        with _inject("pygetwindow", self._gw(None)):
            self.assertEqual(self.mod._get_focused_window(), (None, None))

    def test_blank_title_rejected(self):
        win = types.SimpleNamespace(title="   ", left=0, top=0, width=800, height=600)
        with _inject("pygetwindow", self._gw(win)):
            self.assertEqual(self.mod._get_focused_window(), (None, None))

    def test_geometry_attr_error(self):
        # Accessing .left raises → (None, None).
        class _Win:
            title = "App"
            @property
            def left(self):
                raise AttributeError("no left")
        with _inject("pygetwindow", self._gw(_Win())):
            self.assertEqual(self.mod._get_focused_window(), (None, None))

    def test_degenerate_small_window_rejected(self):
        win = types.SimpleNamespace(title="Tiny", left=0, top=0, width=10, height=10)
        with _inject("pygetwindow", self._gw(win)):
            self.assertEqual(self.mod._get_focused_window(), (None, None))

    def test_offscreen_minimized_rejected(self):
        # Minimized windows report left/top ≈ -32000 → rejected.
        win = types.SimpleNamespace(title="Min", left=-32000, top=-32000,
                                    width=800, height=600)
        with _inject("pygetwindow", self._gw(win)):
            self.assertEqual(self.mod._get_focused_window(), (None, None))

    def test_valid_window_returns_title_and_bbox(self):
        win = types.SimpleNamespace(title="  report.docx - Word  ",
                                    left=100, top=50, width=1280, height=720)
        with _inject("pygetwindow", self._gw(win)):
            title, bbox = self.mod._get_focused_window()
        self.assertEqual(title, "report.docx - Word")
        self.assertEqual(bbox, (100, 50, 1280, 720))


# ── _hash_window_thumbnail (mss + PIL) ───────────────────────────────────
class _FakeSct:
    def __init__(self, size=(800, 600), rgb=b"\x00" * (800 * 600 * 3)):
        self._size = size
        self._rgb = rgb

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def grab(self, region):
        return types.SimpleNamespace(size=self._size, rgb=self._rgb)


class HashThumbnailTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")

    def _mss(self, sct):
        mss = types.ModuleType("mss")
        mss.mss = mock.MagicMock(return_value=sct)
        return mss

    def _pil(self, img):
        pil = types.ModuleType("PIL")
        image_mod = types.ModuleType("PIL.Image")
        image_mod.frombytes = mock.MagicMock(return_value=img)

        class _Resampling:
            LANCZOS = 1
        image_mod.Resampling = _Resampling
        pil.Image = image_mod
        return pil, image_mod

    def test_hash_missing_deps_returns_none(self):
        with _inject("mss", None):
            self.assertIsNone(self.mod._hash_window_thumbnail((0, 0, 800, 600)))

    def test_hash_capture_raises_returns_none(self):
        mss = types.ModuleType("mss")
        mss.mss = mock.MagicMock(side_effect=RuntimeError("no display"))
        pil, _img = self._pil(mock.MagicMock())
        with _inject("mss", mss), \
             mock.patch.dict(sys.modules, {"PIL": pil, "PIL.Image": pil.Image}):
            self.assertIsNone(self.mod._hash_window_thumbnail((0, 0, 800, 600)))

    def test_hash_returns_stable_digest(self):
        # A full PIL chain: frombytes → convert('L') → resize → tobytes.
        thumb = mock.MagicMock()
        thumb.tobytes.return_value = b"THUMBNAILBYTES"
        gray = mock.MagicMock()
        gray.resize.return_value = thumb
        img = mock.MagicMock()
        img.convert.return_value = gray

        sct = _FakeSct()
        pil, image_mod = self._pil(img)
        with _inject("mss", self._mss(sct)), \
             mock.patch.dict(sys.modules, {"PIL": pil, "PIL.Image": image_mod}):
            h1 = self.mod._hash_window_thumbnail((0, 0, 800, 600))
            h2 = self.mod._hash_window_thumbnail((0, 0, 800, 600))
        self.assertIsInstance(h1, str)
        self.assertEqual(h1, h2)            # deterministic
        self.assertEqual(len(h1), 32)       # md5 hex digest
        img.convert.assert_called_with("L")
        # Downsample to THUMB_SIZE × THUMB_SIZE using the LANCZOS resample.
        gray.resize.assert_called_with(
            (self.mod.THUMB_SIZE, self.mod.THUMB_SIZE), 1)


# ── _poll_loop (initial settle + iterate, error resilience) ──────────────
# The loop body wraps everything in `except Exception`, so a sentinel used to
# break out must derive from BaseException (like KeyboardInterrupt) to escape
# the handler instead of being swallowed into an infinite retry.
class _LoopStop(BaseException):
    pass


class PollLoopTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")

    def test_poll_loop_runs_then_stops(self):
        # Drive exactly one iteration: sleep #1 settles (no-op), _poll_once runs,
        # sleep #2 (the post-poll sleep, inside the try) raises _LoopStop, which
        # escapes the `except Exception` and ends the loop.
        sleeps = {"n": 0}

        def _sleep(_s):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise _LoopStop()

        with mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_poll_once") as poll:
            with self.assertRaises(_LoopStop):
                self.mod._poll_loop()
        poll.assert_called_once()

    def test_poll_loop_swallows_poll_error_and_keeps_going(self):
        # _poll_once raises a normal Exception → caught + logged; the loop then
        # hits the error-path sleep, which we make raise _LoopStop to break out.
        # Sequence: settle sleep #1 (no-op) → _poll_once raises → logging.exception
        # → error-path sleep #2 raises _LoopStop.
        sleeps = {"n": 0}

        def _sleep(_s):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise _LoopStop()

        with mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_poll_once",
                               side_effect=RuntimeError("poll boom")), \
             mock.patch.object(self.mod.logging, "exception") as logexc:
            with self.assertRaises(_LoopStop):
                self.mod._poll_loop()
        logexc.assert_called()   # the poll error was logged


# ── register() spawns the watcher thread ─────────────────────────────────
class RegisterTests(unittest.TestCase):
    def test_register_wires_action_and_starts_thread(self):
        # load_skill_isolated neuters Thread.start, so register() builds the
        # daemon thread without it actually running. Assert the action is wired.
        mod, actions = load_skill_isolated("screen_watch")
        self.assertIn("screen_watch_status", actions)
        self.assertTrue(callable(actions["screen_watch_status"]))

    def test_register_thread_target_is_poll_loop(self):
        # Verify register constructs a daemon thread targeting _poll_loop without
        # starting it (Thread.start patched to a no-op).
        created = {}
        import threading as _thr
        orig_init = _thr.Thread.__init__

        def _init(self, *a, **k):
            created["target"] = k.get("target")
            created["daemon"] = k.get("daemon")
            orig_init(self, *a, **k)

        mod, _ = load_skill_isolated("screen_watch", register=False)
        with mock.patch.object(_thr.Thread, "__init__", _init), \
             mock.patch.object(_thr.Thread, "start", lambda self: None):
            actions = {}
            mod.register(actions)
        self.assertIs(created.get("target"), mod._poll_loop)
        self.assertTrue(created.get("daemon"))


if __name__ == "__main__":
    unittest.main()
