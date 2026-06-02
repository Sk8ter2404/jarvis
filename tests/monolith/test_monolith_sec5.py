"""Unit tests for bobert_companion.py, section 5 (source lines ~8882-10902).

This range of the monolith is a mix of:
  * Pure / near-pure helpers (regex parsers, classifiers, formatters).
  * `maybe_*` fast-path dispatchers that gate on module state and bail to
    None early under most conditions.
  * The session-resume greeting composer + the shutdown-prompt state machine.
  * Window / media helpers that lazily `import pygetwindow` / `win32gui` /
    `PIL` / `mss` inside the function body.
  * The big ACTIONS dispatch dict and skill-loader plumbing.

Most genuine `_act_*` handlers in this band were moved to core/actions.py
(the source has "Phase 4x refactor: ... moved to core/actions.py." stubs),
so they are out of scope here; we test the functions that still live in the
monolith between those stubs.

Strategy: import the monolith ONCE via the harness (cached), then patch the
specific module globals each function reads via mock.patch.object(bc, ...).
All external I/O (LLM, vision, pyautogui, win32, filesystem, threads/timers)
is mocked. Lazily-imported modules (pygetwindow/win32gui/PIL/mss) are faked by
injecting a throwaway module into sys.modules for the duration of one test and
removing it in a finally/addCleanup — the harness module object itself is never
swapped.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_sec5
On the light-deps CI runner these all skip via @requires_monolith.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


# ── Small fakes for the lazily-imported Win32 / window libraries ────────────


class _FakeWin:
    """Minimal stand-in for a pygetwindow Window object."""

    def __init__(self, title="", left=0, top=0, width=200, height=100):
        self.title = title
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.activated = False
        self.minimized = False
        self.restored = False
        self._activate_exc = None

    def activate(self):
        if self._activate_exc is not None:
            raise self._activate_exc
        self.activated = True

    def minimize(self):
        self.minimized = True

    def restore(self):
        self.restored = True


def _fake_pygetwindow(all_windows=None, active=None):
    """Build a fake `pygetwindow` module exposing getAllWindows/getActiveWindow."""
    mod = types.ModuleType("pygetwindow")
    mod.getAllWindows = lambda: list(all_windows or [])
    mod.getActiveWindow = lambda: active
    return mod


class _InjectModule:
    """Context manager / cleanup helper that temporarily installs a fake module
    in sys.modules under `name`, restoring the prior entry on exit."""

    def __init__(self, name, module):
        self.name = name
        self.module = module
        self._had = False
        self._prev = None

    def __enter__(self):
        self._had = self.name in sys.modules
        self._prev = sys.modules.get(self.name)
        sys.modules[self.name] = self.module
        return self.module

    def __exit__(self, *exc):
        if self._had:
            sys.modules[self.name] = self._prev
        else:
            sys.modules.pop(self.name, None)
        return False


@requires_monolith
class SectionFiveBase(MonolithGlobalsTestCase):
    # setUpClass (loads the cached monolith) + per-test deep-restore of the
    # mutated bobert_companion globals are inherited from
    # MonolithGlobalsTestCase.
    pass


# ════════════════════════════════════════════════════════════════════════════
#  _build_session_resume / maybe_session_resume_greeting
# ════════════════════════════════════════════════════════════════════════════
class SessionResumeTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        # Neutralise every external the composer touches by default.
        self._patches = [
            mock.patch.object(bc, "_last_session_end_ts", return_value=0.0),
            mock.patch.object(bc, "_last_n_user_commands", return_value=[]),
            mock.patch.object(bc, "_last_queued_task_line", return_value=""),
            mock.patch.object(bc.pattern_memory, "get_session_summaries",
                              return_value=[]),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        # Restore the once-per-process latch after each test.
        orig_latch = list(bc._session_resume_done)
        self.addCleanup(lambda: bc._session_resume_done.__setitem__(
            slice(None), orig_latch))

    def test_no_force_outside_window_returns_empty(self):
        # last_ts 0 -> age inf -> not in window -> ("", details) when not forced.
        text, details = self.bc._build_session_resume(force=False)
        self.assertEqual(text, "")
        self.assertFalse(details["in_window"])
        self.assertEqual(details["last_session_ts"], 0.0)

    def test_force_no_prior_session(self):
        text, details = self.bc._build_session_resume(force=True)
        self.assertIn("no prior session", text)
        self.assertIn("fresh start", text)

    def test_force_stale_session_with_work_reports_hours(self):
        bc = self.bc
        import time as _t
        recent_ts = _t.time() - 25 * 3600  # 25h ago -> "hours" branch (<48)
        with mock.patch.object(bc, "_last_session_end_ts", return_value=recent_ts), \
             mock.patch.object(bc, "_last_queued_task_line",
                               return_value="- [ ] **2026-01-01 09:00** [task] — wire the foobar"):
            text, details = bc._build_session_resume(force=True)
        self.assertIn("rather a while ago", text)
        self.assertIn("hours", text)
        self.assertIn("foobar", text)
        self.assertAlmostEqual(details["age_seconds"], 25 * 3600, delta=120)

    def test_force_very_stale_session_reports_days(self):
        bc = self.bc
        import time as _t
        old_ts = _t.time() - 5 * 86400  # 5 days
        with mock.patch.object(bc, "_last_session_end_ts", return_value=old_ts):
            text, _ = bc._build_session_resume(force=True)
        self.assertIn("days", text)
        self.assertIn("gone cold", text)  # no concrete work -> cold-thread line

    def test_warm_restart_with_task_uses_at_your_service(self):
        bc = self.bc
        import time as _t
        warm_ts = _t.time() - 60  # 1 minute ago -> inside 18h window
        with mock.patch.object(bc, "_last_session_end_ts", return_value=warm_ts), \
             mock.patch.object(bc, "_last_queued_task_line",
                               return_value="- [ ] **2026-01-01 09:00** [task] — refactor the parser"):
            text, details = bc._build_session_resume(force=False)
        self.assertTrue(details["in_window"])
        self.assertIn("Welcome back", text)
        self.assertIn("At your service", text)
        self.assertIn("parser", text)

    def test_warm_restart_no_work_uses_im_afraid(self):
        bc = self.bc
        import time as _t
        warm_ts = _t.time() - 120
        with mock.patch.object(bc, "_last_session_end_ts", return_value=warm_ts):
            text, details = bc._build_session_resume(force=False)
        self.assertTrue(details["in_window"])
        self.assertIn("Welcome back", text)
        self.assertIn("I'm afraid", text)

    def test_warm_restart_falls_back_to_last_command(self):
        bc = self.bc
        import time as _t
        warm_ts = _t.time() - 90
        with mock.patch.object(bc, "_last_session_end_ts", return_value=warm_ts), \
             mock.patch.object(bc, "_last_n_user_commands",
                               return_value=["Open the garage door."]):
            text, details = bc._build_session_resume(force=False)
        # No task line, but a recent command -> work derived from it (lowercased).
        self.assertEqual(details["work"], "open the garage door")
        self.assertIn("open the garage door", text)

    def test_warm_restart_uses_session_summary_when_no_command(self):
        bc = self.bc
        import time as _t
        warm_ts = _t.time() - 90
        with mock.patch.object(bc, "_last_session_end_ts", return_value=warm_ts), \
             mock.patch.object(bc.pattern_memory, "get_session_summaries",
                               return_value=[{"summary": "Debugged the audio pipeline. Other stuff."}]):
            text, details = bc._build_session_resume(force=False)
        self.assertEqual(details["work"], "Debugged the audio pipeline")
        self.assertEqual(details["last_summary"],
                         "Debugged the audio pipeline. Other stuff.")

    def test_summary_lookup_exception_is_swallowed(self):
        bc = self.bc
        import time as _t
        warm_ts = _t.time() - 90
        with mock.patch.object(bc, "_last_session_end_ts", return_value=warm_ts), \
             mock.patch.object(bc.pattern_memory, "get_session_summaries",
                               side_effect=RuntimeError("db down")):
            text, details = bc._build_session_resume(force=False)
        # Should not raise; last_summary stays empty.
        self.assertEqual(details["last_summary"], "")
        self.assertIn("Welcome back", text)

    def test_maybe_greeting_latches_once(self):
        bc = self.bc
        import time as _t
        warm_ts = _t.time() - 60
        bc._session_resume_done[0] = False
        with mock.patch.object(bc, "_last_session_end_ts", return_value=warm_ts):
            first = bc.maybe_session_resume_greeting()
            self.assertTrue(first)              # non-empty greeting
            self.assertTrue(bc._session_resume_done[0])
            second = bc.maybe_session_resume_greeting()
        self.assertEqual(second, "")           # latched -> empty second time

    def test_maybe_greeting_empty_outside_window_does_not_latch(self):
        bc = self.bc
        bc._session_resume_done[0] = False
        # Default patches -> outside window -> "" -> latch stays False.
        out = bc.maybe_session_resume_greeting()
        self.assertEqual(out, "")
        self.assertFalse(bc._session_resume_done[0])


# ════════════════════════════════════════════════════════════════════════════
#  Shutdown prompt state machine
# ════════════════════════════════════════════════════════════════════════════
class ShutdownPromptTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        # Snapshot + restore the shared pending dict.
        self._orig_pending = dict(bc._shutdown_prompt_pending)
        self.addCleanup(lambda: (bc._shutdown_prompt_pending.clear(),
                                 bc._shutdown_prompt_pending.update(self._orig_pending)))
        # Silence speech + stub the terminal dispatch targets.
        self.speak = mock.patch.object(bc, "_speak").start()
        self.addCleanup(mock.patch.stopall)
        self.shutdown = mock.patch.object(bc, "_act_shutdown_jarvis").start()
        self.overnight = mock.patch.object(bc, "_act_start_overnight_upgrade").start()

    # ---- _check_and_arm_shutdown_prompt ----
    def test_arm_on_trigger_phrase(self):
        bc = self.bc
        consumed = bc._check_and_arm_shutdown_prompt("JARVIS, shut down")
        self.assertTrue(consumed)
        self.assertTrue(bc._shutdown_prompt_pending["armed"])
        self.speak.assert_called_once()
        self.assertIn("overnight protocol", self.speak.call_args[0][0].lower())

    def test_no_arm_on_empty(self):
        self.assertFalse(self.bc._check_and_arm_shutdown_prompt(""))

    def test_no_arm_on_long_utterance(self):
        # >6 words -> ignored even though it contains "shut down".
        bc = self.bc
        long_txt = "well if I were to say shut down it should not fire at all"
        self.assertFalse(bc._check_and_arm_shutdown_prompt(long_txt))
        self.assertFalse(bc._shutdown_prompt_pending.get("armed"))

    def test_no_arm_without_trigger_phrase(self):
        self.assertFalse(self.bc._check_and_arm_shutdown_prompt("what time is it"))

    # ---- _handle_shutdown_prompt ----
    def test_handle_returns_false_when_not_armed(self):
        bc = self.bc
        bc._shutdown_prompt_pending["armed"] = False
        self.assertFalse(bc._handle_shutdown_prompt("yes"))

    def test_handle_expired_clears_and_returns_false(self):
        bc = self.bc
        import time as _t
        bc._shutdown_prompt_pending["armed"] = True
        bc._shutdown_prompt_pending["expires_at"] = _t.time() - 5  # already expired
        self.assertFalse(bc._handle_shutdown_prompt("yes"))
        self.assertFalse(bc._shutdown_prompt_pending["armed"])
        self.shutdown.assert_not_called()
        self.overnight.assert_not_called()

    def _arm(self):
        import time as _t
        self.bc._shutdown_prompt_pending["armed"] = True
        self.bc._shutdown_prompt_pending["expires_at"] = _t.time() + 100

    def test_handle_yes_fires_overnight(self):
        self._arm()
        self.assertTrue(self.bc._handle_shutdown_prompt("yes"))
        self.overnight.assert_called_once()
        self.shutdown.assert_not_called()
        self.assertFalse(self.bc._shutdown_prompt_pending["armed"])

    def test_handle_yes_prefix_phrase_fires_overnight(self):
        self._arm()
        self.assertTrue(self.bc._handle_shutdown_prompt("go ahead and do it"))
        self.overnight.assert_called_once()

    def test_handle_no_fires_full_shutdown(self):
        self._arm()
        self.assertTrue(self.bc._handle_shutdown_prompt("no"))
        self.shutdown.assert_called_once()
        self.overnight.assert_not_called()

    def test_handle_no_overnight_hits_no_not_yes(self):
        # "no overnight" contains YES's "overnight" substring; NO is checked
        # first so it must route to full shutdown, not overnight.
        self._arm()
        self.assertTrue(self.bc._handle_shutdown_prompt("no overnight"))
        self.shutdown.assert_called_once()
        self.overnight.assert_not_called()

    def test_handle_reinforced_shutdown_phrase(self):
        self._arm()
        self.assertTrue(self.bc._handle_shutdown_prompt("shut down jarvis"))
        self.shutdown.assert_called_once()
        self.overnight.assert_not_called()

    def test_handle_unrelated_cancels_and_returns_false(self):
        self._arm()
        result = self.bc._handle_shutdown_prompt("what's the weather")
        self.assertFalse(result)            # falls through to normal routing
        self.shutdown.assert_not_called()
        self.overnight.assert_not_called()
        self.speak.assert_called_with("Shutdown cancelled.")
        self.assertFalse(self.bc._shutdown_prompt_pending["armed"])

    def test_handle_dispatch_exception_still_consumes(self):
        # Even if the terminal action raises, the branch returns True.
        self._arm()
        self.overnight.side_effect = RuntimeError("boom")
        self.assertTrue(self.bc._handle_shutdown_prompt("yes"))


# ════════════════════════════════════════════════════════════════════════════
#  Window discovery / focus helpers (lazy pygetwindow import)
# ════════════════════════════════════════════════════════════════════════════
class WindowHelperTests(SectionFiveBase):
    def test_find_windows_by_title_filters_case_insensitive(self):
        bc = self.bc
        wins = [_FakeWin("Spotify Premium"), _FakeWin("Notepad"),
                _FakeWin(""), _FakeWin("My SPOTIFY tab")]
        with _InjectModule("pygetwindow", _fake_pygetwindow(all_windows=wins)):
            out = bc._find_windows_by_title("spotify")
        titles = [w.title for w in out]
        self.assertEqual(titles, ["Spotify Premium", "My SPOTIFY tab"])

    def test_find_windows_by_title_importerror_returns_empty(self):
        bc = self.bc
        # Force the inner `import pygetwindow` to raise ImportError.
        with _InjectModule("pygetwindow", None):
            with mock.patch.dict(sys.modules, {"pygetwindow": None}):
                out = bc._find_windows_by_title("anything")
        self.assertEqual(out, [])

    def test_find_music_window_priority_order(self):
        bc = self.bc
        # Spotify ranks above youtube in MUSIC_WINDOW_HINTS, even though the
        # youtube window appears first in the list.
        wins = [_FakeWin("Some YouTube video"), _FakeWin("Spotify")]
        with _InjectModule("pygetwindow", _fake_pygetwindow(all_windows=wins)):
            win = bc._find_music_window()
        self.assertIsNotNone(win)
        self.assertEqual(win.title, "Spotify")

    def test_find_music_window_none_when_no_hint(self):
        bc = self.bc
        wins = [_FakeWin("Notepad"), _FakeWin("Calculator")]
        with _InjectModule("pygetwindow", _fake_pygetwindow(all_windows=wins)):
            self.assertIsNone(bc._find_music_window())

    def test_focus_music_window_activate_success(self):
        bc = self.bc
        win = _FakeWin("Spotify")
        with mock.patch.object(bc, "_find_music_window", return_value=win):
            title = bc._focus_music_window()
        self.assertEqual(title, "Spotify")
        self.assertTrue(win.activated)

    def test_focus_music_window_none_when_no_candidate(self):
        bc = self.bc
        with mock.patch.object(bc, "_find_music_window", return_value=None):
            self.assertIsNone(bc._focus_music_window())

    def test_focus_music_window_benign_win32_error_treated_as_success(self):
        bc = self.bc
        win = _FakeWin("Spotify")
        win._activate_exc = Exception("Error code from Windows: 0 - "
                                      "The operation completed successfully.")
        with mock.patch.object(bc, "_find_music_window", return_value=win):
            title = bc._focus_music_window()
        self.assertEqual(title, "Spotify")

    def test_focus_music_window_falls_back_to_minimize_restore(self):
        bc = self.bc
        win = _FakeWin("Spotify")
        win._activate_exc = Exception("some genuine failure")
        with mock.patch.object(bc, "_find_music_window", return_value=win):
            title = bc._focus_music_window()
        self.assertEqual(title, "Spotify")
        self.assertTrue(win.minimized)
        self.assertTrue(win.restored)

    def test_flash_window_reticle_publishes_center(self):
        bc = self.bc
        win = _FakeWin("My App — extra", left=100, top=50, width=200, height=100)
        with mock.patch.object(bc, "_publish_reticle") as pub:
            bc._flash_window_reticle(win, label="focus")
        pub.assert_called_once()
        cx, cy, lbl = pub.call_args[0]
        self.assertEqual((cx, cy), (200, 100))  # center: 100+200/2, 50+100/2
        self.assertIn("My App", lbl)

    def test_flash_window_reticle_swallows_geometry_errors(self):
        bc = self.bc

        class Bad:
            title = "x"

            def __getattr__(self, name):
                raise RuntimeError("no geometry")

        with mock.patch.object(bc, "_publish_reticle") as pub:
            bc._flash_window_reticle(Bad())  # must not raise
        pub.assert_not_called()

    def test_active_window_is_terminal_true(self):
        bc = self.bc
        active = _FakeWin("Windows PowerShell")
        with _InjectModule("pygetwindow", _fake_pygetwindow(active=active)):
            self.assertTrue(bc._active_window_is_terminal())

    def test_active_window_is_terminal_false_for_browser(self):
        bc = self.bc
        active = _FakeWin("Gmail - Google Chrome")
        with _InjectModule("pygetwindow", _fake_pygetwindow(active=active)):
            self.assertFalse(bc._active_window_is_terminal())

    def test_active_window_is_terminal_none_active(self):
        bc = self.bc
        with _InjectModule("pygetwindow", _fake_pygetwindow(active=None)):
            self.assertFalse(bc._active_window_is_terminal())


# ════════════════════════════════════════════════════════════════════════════
#  Pure parsers / classifiers
# ════════════════════════════════════════════════════════════════════════════
class PureHelperTests(SectionFiveBase):
    # ---- _parse_monitor_prefix ----
    def test_parse_monitor_prefix_match(self):
        mon, rest = self.bc._parse_monitor_prefix("monitor:LEFT | open notepad")
        self.assertEqual(mon, "left")
        self.assertEqual(rest, "open notepad")

    def test_parse_monitor_prefix_no_match(self):
        mon, rest = self.bc._parse_monitor_prefix("just some text")
        self.assertIsNone(mon)
        self.assertEqual(rest, "just some text")

    # ---- _format_screen_age ----
    def test_format_screen_age_buckets(self):
        f = self.bc._format_screen_age
        self.assertEqual(f(2), "just now")
        self.assertEqual(f(30), "30 seconds ago")
        self.assertEqual(f(60), "1 minute ago")
        self.assertEqual(f(125), "2 minutes ago")
        self.assertEqual(f(3600), "1 hour ago")
        self.assertEqual(f(7200), "2 hours ago")

    # ---- _is_glance_ambiguous_question ----
    def test_glance_ambiguous_positive(self):
        f = self.bc._is_glance_ambiguous_question
        for s in ["what?", "huh", "wait, what", "should I worry?",
                  "explain this", "What Is This?"]:
            self.assertTrue(f(s), s)

    def test_glance_ambiguous_negative(self):
        f = self.bc._is_glance_ambiguous_question
        self.assertFalse(f(""))
        self.assertFalse(f("what is the capital of France and why"))  # >5 words
        self.assertFalse(f("please open my email client now"))

    def test_glance_ambiguous_trailing_question_mark(self):
        # pattern stored without '?' but utterance has one
        self.assertTrue(self.bc._is_glance_ambiguous_question("explain?"))

    # ---- _is_self_close_attempt ----
    def test_self_close_attempt_positive(self):
        f = self.bc._is_self_close_attempt
        self.assertTrue(f("close button on Windows PowerShell"))
        self.assertTrue(f("quit the python process"))
        self.assertTrue(f("kill the terminal"))

    def test_self_close_attempt_negative(self):
        f = self.bc._is_self_close_attempt
        self.assertFalse(f("close the Chrome window"))   # forbidden verb, safe target
        self.assertFalse(f("open powershell"))           # forbidden target, no close verb

    # ---- _normalize_key ----
    def test_normalize_key_aliases(self):
        f = self.bc._normalize_key
        self.assertEqual(f("Super"), "win")
        self.assertEqual(f("CMD"), "win")
        self.assertEqual(f("control"), "ctrl")
        self.assertEqual(f("Return"), "enter")
        self.assertEqual(f("escape"), "esc")
        self.assertEqual(f("  Option "), "alt")
        self.assertEqual(f("a"), "a")          # passthrough, lowercased

    # ---- _looks_like_shell_command ----
    def test_looks_like_shell_command_positive(self):
        f = self.bc._looks_like_shell_command
        self.assertTrue(f("Get-ChildItem"))
        self.assertTrue(f("git status"))
        self.assertTrue(f("  python script.py"))
        self.assertTrue(f("$env:PATH"))
        self.assertTrue(f("& 'C:\\tool.exe'"))

    def test_looks_like_shell_command_negative(self):
        f = self.bc._looks_like_shell_command
        self.assertFalse(f(""))
        self.assertFalse(f("hello there friend"))
        self.assertFalse(f("the quick brown fox"))

    # ---- _substitute_monitor_in_arg ----
    def test_substitute_monitor_open_on_monitor(self):
        out = self.bc._substitute_monitor_in_arg(
            "open_on_monitor", "left | https://example.com", "right")
        self.assertEqual(out, "right | https://example.com")

    def test_substitute_monitor_move_window(self):
        out = self.bc._substitute_monitor_in_arg(
            "move_window_to_monitor", "Notepad | left", "right")
        self.assertEqual(out, "Notepad | right")

    def test_substitute_monitor_unknown_action_unchanged(self):
        out = self.bc._substitute_monitor_in_arg(
            "some_other_action", "a | b", "right")
        self.assertEqual(out, "a | b")

    def test_substitute_monitor_no_pipe_unchanged(self):
        out = self.bc._substitute_monitor_in_arg(
            "open_on_monitor", "no-pipe-here", "right")
        self.assertEqual(out, "no-pipe-here")

    def test_substitute_monitor_empty_monitor_unchanged(self):
        out = self.bc._substitute_monitor_in_arg(
            "open_on_monitor", "left | x", "")
        self.assertEqual(out, "left | x")


# ════════════════════════════════════════════════════════════════════════════
#  Screen-context cache
# ════════════════════════════════════════════════════════════════════════════
class ScreenCacheTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        # Save + clear the shared cache; restore exactly after.
        self._orig = list(bc._screen_cache)
        bc._screen_cache.clear()
        self.addCleanup(lambda: (bc._screen_cache.clear(),
                                 bc._screen_cache.extend(self._orig)))

    def test_push_and_recent_roundtrip(self):
        bc = self.bc
        bc._push_screen_context("left", "what is this", "an editor",
                                {"left": b"PNGBYTES"})
        recent = bc._recent_screen_contexts()
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["question"], "what is this")
        self.assertEqual(recent[0]["answer"], "an editor")
        self.assertEqual(recent[0]["monitor"], "left")
        self.assertEqual(recent[0]["images"]["left"], b"PNGBYTES")

    def test_push_caps_at_max_entries(self):
        bc = self.bc
        for i in range(bc.SCREEN_CACHE_MAX_ENTRIES + 4):
            bc._push_screen_context(None, f"q{i}", f"a{i}", {})
        self.assertEqual(len(bc._screen_cache), bc.SCREEN_CACHE_MAX_ENTRIES)
        # Oldest evicted; newest retained.
        answers = [e["answer"] for e in bc._screen_cache]
        self.assertIn(f"a{bc.SCREEN_CACHE_MAX_ENTRIES + 3}", answers)
        self.assertNotIn("a0", answers)

    def test_recent_returns_newest_first(self):
        bc = self.bc
        bc._push_screen_context(None, "first", "a1", {})
        bc._push_screen_context(None, "second", "a2", {})
        recent = bc._recent_screen_contexts()
        self.assertEqual([e["question"] for e in recent], ["second", "first"])

    def test_recent_filters_stale_entries(self):
        bc = self.bc
        import time as _t
        bc._push_screen_context(None, "old", "a", {})
        # Backdate the entry well beyond the requested max_age.
        bc._screen_cache[0]["ts"] = _t.time() - 9999
        self.assertEqual(bc._recent_screen_contexts(max_age=60), [])


# ════════════════════════════════════════════════════════════════════════════
#  Focused-window read + glance change detection
# ════════════════════════════════════════════════════════════════════════════
class FocusTrackerTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        self._orig_state = dict(bc._focused_window_state)
        self.addCleanup(lambda: (bc._focused_window_state.clear(),
                                 bc._focused_window_state.update(self._orig_state)))

    def _fake_win32gui(self, hwnd=123, title="App", rect=(10, 20, 410, 320)):
        mod = types.ModuleType("win32gui")
        mod.GetForegroundWindow = lambda: hwnd
        mod.GetWindowText = lambda h: title
        mod.GetWindowRect = lambda h: rect
        return mod

    def test_read_focused_window_success(self):
        bc = self.bc
        with _InjectModule("win32gui", self._fake_win32gui()):
            hwnd, title, rect = bc._read_focused_window()
        self.assertEqual(hwnd, 123)
        self.assertEqual(title, "App")
        # rect returned as (left, top, width, height): (10,20, 410-10, 320-20)
        self.assertEqual(rect, (10, 20, 400, 300))

    def test_read_focused_window_no_foreground(self):
        bc = self.bc
        mod = self._fake_win32gui(hwnd=0)
        with _InjectModule("win32gui", mod):
            self.assertEqual(bc._read_focused_window(), (None, "", None))

    def test_read_focused_window_degenerate_rect_is_none(self):
        bc = self.bc
        # right==left, bot==top -> width/height <=1 -> rect None but hwnd/title set.
        mod = self._fake_win32gui(rect=(5, 5, 5, 5))
        with _InjectModule("win32gui", mod):
            hwnd, title, rect = bc._read_focused_window()
        self.assertEqual(hwnd, 123)
        self.assertIsNone(rect)

    def test_read_focused_window_import_failure(self):
        bc = self.bc
        with mock.patch.dict(sys.modules, {"win32gui": None}):
            self.assertEqual(bc._read_focused_window(), (None, "", None))

    def test_focus_changed_recently_live_read_detects_switch(self):
        bc = self.bc
        bc._focused_window_state["hwnd"] = 111   # previous
        with mock.patch.object(bc, "_read_focused_window",
                               return_value=(222, "New", (0, 0, 100, 100))):
            self.assertTrue(bc._focus_changed_recently())
        # State updated to the new window.
        self.assertEqual(bc._focused_window_state["hwnd"], 222)

    def test_focus_changed_recently_recent_timestamp(self):
        bc = self.bc
        import time as _t
        bc._focused_window_state["hwnd"] = 333
        bc._focused_window_state["changed_at"] = _t.monotonic()  # just now
        # Live read returns the SAME hwnd -> no live-switch, rely on timestamp.
        with mock.patch.object(bc, "_read_focused_window",
                               return_value=(333, "Same", None)):
            self.assertTrue(bc._focus_changed_recently())

    def test_focus_changed_recently_stale_timestamp_false(self):
        bc = self.bc
        bc._focused_window_state["hwnd"] = 333
        bc._focused_window_state["changed_at"] = float("-inf")
        with mock.patch.object(bc, "_read_focused_window",
                               return_value=(333, "Same", None)):
            self.assertFalse(bc._focus_changed_recently())


# ════════════════════════════════════════════════════════════════════════════
#  _capture_focused_window_png  (early-return paths only — no real capture)
# ════════════════════════════════════════════════════════════════════════════
class CaptureFocusedWindowTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        self._orig_state = dict(bc._focused_window_state)
        self.addCleanup(lambda: (bc._focused_window_state.clear(),
                                 bc._focused_window_state.update(self._orig_state)))

    def test_returns_none_when_no_rect_anywhere(self):
        bc = self.bc
        bc._focused_window_state["rect"] = None
        with mock.patch.object(bc, "_read_focused_window",
                               return_value=(None, "", None)):
            self.assertIsNone(bc._capture_focused_window_png())

    def test_returns_none_for_tiny_window(self):
        bc = self.bc
        bc._focused_window_state["rect"] = (0, 0, 10, 10)  # < 50px each dim
        self.assertIsNone(bc._capture_focused_window_png())

    def test_returns_none_when_pillow_missing(self):
        bc = self.bc
        bc._focused_window_state["rect"] = (0, 0, 800, 600)
        # Make `from PIL import Image` fail.
        with mock.patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            self.assertIsNone(bc._capture_focused_window_png())


# ════════════════════════════════════════════════════════════════════════════
#  maybe_glance_response  (gating / early-return paths)
# ════════════════════════════════════════════════════════════════════════════
class MaybeGlanceResponseTests(SectionFiveBase):
    def test_returns_none_for_non_ambiguous(self):
        bc = self.bc
        with mock.patch.object(bc, "_is_glance_ambiguous_question",
                               return_value=False):
            self.assertIsNone(bc.maybe_glance_response("open my email"))

    def test_returns_none_when_focus_not_changed(self):
        bc = self.bc
        with mock.patch.object(bc, "_is_glance_ambiguous_question",
                               return_value=True), \
             mock.patch.object(bc, "_focus_changed_recently",
                               return_value=False):
            self.assertIsNone(bc.maybe_glance_response("what?"))

    def test_returns_none_when_backend_not_claude(self):
        bc = self.bc
        with mock.patch.object(bc, "_is_glance_ambiguous_question",
                               return_value=True), \
             mock.patch.object(bc, "_focus_changed_recently",
                               return_value=True), \
             mock.patch.object(bc, "AI_BACKEND", "ollama"):
            self.assertIsNone(bc.maybe_glance_response("what?"))

    def test_returns_none_when_capture_fails(self):
        bc = self.bc
        with mock.patch.object(bc, "_is_glance_ambiguous_question",
                               return_value=True), \
             mock.patch.object(bc, "_focus_changed_recently",
                               return_value=True), \
             mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.object(bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(bc, "_capture_focused_window_png",
                               return_value=None):
            self.assertIsNone(bc.maybe_glance_response("what?"))

    def test_full_path_appends_history_and_caches(self):
        bc = self.bc
        orig_hist = list(bc.conversation_history)
        self.addCleanup(lambda: (bc.conversation_history.clear(),
                                 bc.conversation_history.extend(orig_hist)))
        anim_thread = mock.MagicMock()
        anim_thread.is_alive.return_value = False
        with mock.patch.object(bc, "_is_glance_ambiguous_question",
                               return_value=True), \
             mock.patch.object(bc, "_focus_changed_recently",
                               return_value=True), \
             mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.object(bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(bc, "_capture_focused_window_png",
                               return_value=b"PNG"), \
             mock.patch.object(bc, "pause_face_tracking"), \
             mock.patch.object(bc, "set_state"), \
             mock.patch.object(bc, "_thinking_loop"), \
             mock.patch.object(bc, "_trim_conversation_history"), \
             mock.patch.object(bc, "_push_screen_context") as push, \
             mock.patch.object(bc, "ask_vision",
                               return_value="I'm afraid that's an error dialog, sir."), \
             mock.patch.object(bc.threading, "Thread") as Thread:
            # The worker thread must actually run ask_vision (the function calls
            # _vis_worker on a Thread); emulate by invoking target synchronously.
            def _make_thread(target=None, args=(), daemon=None, **kw):
                if target is not None:
                    target(*args)
                return anim_thread
            Thread.side_effect = _make_thread
            out = bc.maybe_glance_response("what?")
        self.assertIsNotNone(out)
        self.assertIn("I'm afraid", out)
        # user + assistant turns appended
        self.assertEqual(bc.conversation_history[-2]["role"], "user")
        self.assertEqual(bc.conversation_history[-1]["role"], "assistant")
        self.assertEqual(bc.conversation_history[-1]["content"], out)
        push.assert_called_once()

    def test_full_path_vision_parenthetical_returns_none(self):
        # ask_vision returning a "(...)" sentinel means failure -> None.
        bc = self.bc
        anim_thread = mock.MagicMock()
        anim_thread.is_alive.return_value = False
        with mock.patch.object(bc, "_is_glance_ambiguous_question",
                               return_value=True), \
             mock.patch.object(bc, "_focus_changed_recently",
                               return_value=True), \
             mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.object(bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(bc, "_capture_focused_window_png",
                               return_value=b"PNG"), \
             mock.patch.object(bc, "pause_face_tracking"), \
             mock.patch.object(bc, "set_state"), \
             mock.patch.object(bc, "_thinking_loop"), \
             mock.patch.object(bc, "ask_vision",
                               return_value="(vision unavailable)"), \
             mock.patch.object(bc.threading, "Thread") as Thread:
            def _make_thread(target=None, args=(), daemon=None, **kw):
                if target is not None:
                    target(*args)
                return anim_thread
            Thread.side_effect = _make_thread
            self.assertIsNone(bc.maybe_glance_response("what?"))


# ════════════════════════════════════════════════════════════════════════════
#  _media_key_with_focus
# ════════════════════════════════════════════════════════════════════════════
class MediaKeyWithFocusTests(SectionFiveBase):
    def test_pyautogui_unavailable(self):
        bc = self.bc
        with mock.patch.object(bc, "_get_pyautogui", return_value=None):
            out = bc._media_key_with_focus("playpause", "play button", "Play/Pause")
        self.assertEqual(out, "pyautogui unavailable")

    def test_focused_window_press(self):
        bc = self.bc
        pag = mock.MagicMock()
        with mock.patch.object(bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(bc, "_focus_music_window", return_value="Spotify"), \
             mock.patch.object(bc, "_ui_safe") as ui_safe, \
             mock.patch.object(bc.time, "sleep"):
            out = bc._media_key_with_focus("nexttrack", "next button", "Next")
        self.assertIn("Next", out)
        self.assertIn("Spotify", out)
        ui_safe.assert_called_once()

    def test_no_music_window_global_key_then_vision_click(self):
        bc = self.bc
        pag = mock.MagicMock()
        with mock.patch.object(bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(bc, "_focus_music_window", return_value=None), \
             mock.patch.object(bc, "_ui_safe"), \
             mock.patch.object(bc, "find_click_target", return_value=(50, 60)), \
             mock.patch.object(bc, "ui_click") as click:
            out = bc._media_key_with_focus("playpause", "play button", "Play/Pause")
        click.assert_called_once_with(50, 60)
        self.assertIn("clicked on-screen button", out)

    def test_no_music_window_no_vision_target(self):
        bc = self.bc
        pag = mock.MagicMock()
        with mock.patch.object(bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(bc, "_focus_music_window", return_value=None), \
             mock.patch.object(bc, "_ui_safe"), \
             mock.patch.object(bc, "find_click_target", return_value=None):
            out = bc._media_key_with_focus("playpause", "play button", "Play/Pause")
        self.assertIn("key sent globally", out)

    def test_focused_press_ui_failsafe_returns_message(self):
        bc = self.bc
        pag = mock.MagicMock()
        err = bc.UIFailsafeError("failsafe tripped")
        with mock.patch.object(bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(bc, "_focus_music_window", return_value="Spotify"), \
             mock.patch.object(bc, "_ui_safe", side_effect=err), \
             mock.patch.object(bc.time, "sleep"):
            out = bc._media_key_with_focus("nexttrack", "next", "Next")
        self.assertIn("failsafe tripped", out)


# ════════════════════════════════════════════════════════════════════════════
#  maybe_replay_last_action
# ════════════════════════════════════════════════════════════════════════════
class MaybeReplayTests(SectionFiveBase):
    def test_no_match_returns_none(self):
        bc = self.bc
        with mock.patch.object(bc, "_act_replay_last_action") as act:
            self.assertIsNone(bc.maybe_replay_last_action("open notepad please"))
        act.assert_not_called()

    def test_empty_returns_none(self):
        self.assertIsNone(self.bc.maybe_replay_last_action("   "))

    def test_basic_replay_phrase_dispatches(self):
        bc = self.bc
        with mock.patch.object(bc, "_act_replay_last_action",
                               return_value="Replayed.") as act:
            out = bc.maybe_replay_last_action("do that again")
        self.assertEqual(out, "Replayed.")
        act.assert_called_once_with("")     # no monitor captured

    def test_replay_with_monitor_capture(self):
        bc = self.bc
        with mock.patch.object(bc, "_act_replay_last_action",
                               return_value="ok") as act:
            bc.maybe_replay_last_action("run the last action on the left monitor")
        act.assert_called_once_with("left")

    def test_replay_with_bare_digit_monitor(self):
        bc = self.bc
        with mock.patch.object(bc, "_act_replay_last_action",
                               return_value="ok") as act:
            bc.maybe_replay_last_action("repeat that on monitor 2")
        act.assert_called_once_with("2")


# ════════════════════════════════════════════════════════════════════════════
#  Tray plumbing: _tray_async / _selfdiag_module / _probe_via_selfdiag
# ════════════════════════════════════════════════════════════════════════════
class TrayPlumbingTests(SectionFiveBase):
    def test_tray_async_runs_fn_in_thread(self):
        bc = self.bc
        import threading as _thr
        done = _thr.Event()
        result_holder = []

        def _fn():
            result_holder.append("ran")
            done.set()
            return "status line"

        bc._tray_async("unit-test", _fn)
        self.assertTrue(done.wait(timeout=5.0))
        self.assertEqual(result_holder, ["ran"])

    def test_tray_async_swallows_exceptions(self):
        bc = self.bc
        import threading as _thr
        started = _thr.Event()

        def _boom():
            started.set()
            raise RuntimeError("nope")

        # Should not raise in the caller; the daemon catches it.
        bc._tray_async("boom", _boom)
        self.assertTrue(started.wait(timeout=5.0))

    def test_selfdiag_module_present_and_absent(self):
        bc = self.bc
        sentinel = types.ModuleType("skill_self_diagnostic")
        with mock.patch.dict(sys.modules,
                             {"skill_self_diagnostic": sentinel}):
            self.assertIs(bc._selfdiag_module(), sentinel)
        # Absent
        saved = sys.modules.pop("skill_self_diagnostic", None)
        try:
            self.assertIsNone(bc._selfdiag_module())
        finally:
            if saved is not None:
                sys.modules["skill_self_diagnostic"] = saved

    def test_probe_via_selfdiag_unavailable(self):
        bc = self.bc
        with mock.patch.object(bc, "_selfdiag_module", return_value=None):
            out = bc._probe_via_selfdiag("mic", "_probe_mic")
        self.assertIn("unavailable", out)

    def test_probe_via_selfdiag_missing_attr(self):
        bc = self.bc
        sd = types.ModuleType("skill_self_diagnostic")  # no _probe_mic attr
        with mock.patch.object(bc, "_selfdiag_module", return_value=sd):
            out = bc._probe_via_selfdiag("mic", "_probe_mic")
        self.assertIn("unavailable", out)

    def test_probe_via_selfdiag_spawns_async_and_reports_running(self):
        bc = self.bc
        sd = types.ModuleType("skill_self_diagnostic")
        sd._probe_mic = lambda: {"ok": True, "latency_ms": 12.0}
        sd._run_with_timeout = lambda fn, budget, name=None: fn()
        sd.PER_PROBE_TIMEOUT_S = 5.0
        captured = {}

        def _fake_async(name, fn):
            # Run synchronously so we can assert on the probe body result.
            captured["name"] = name
            captured["result"] = fn()

        with mock.patch.object(bc, "_selfdiag_module", return_value=sd), \
             mock.patch.object(bc, "_tray_async", side_effect=_fake_async):
            out = bc._probe_via_selfdiag("mic", "_probe_mic")
        self.assertEqual(out, "mic probe running")
        self.assertIn("mic: OK", captured["result"])
        self.assertIn("12", captured["result"])


# ════════════════════════════════════════════════════════════════════════════
#  load_skills (disabled fast-path) + _audio_music_feed bridge
# ════════════════════════════════════════════════════════════════════════════
class SkillLoaderTests(SectionFiveBase):
    def test_load_skills_noop_when_disabled(self):
        bc = self.bc
        with mock.patch.object(bc, "SKILLS_ENABLED", False), \
             mock.patch.object(bc.os, "makedirs") as mkd:
            bc.load_skills()
        mkd.assert_not_called()   # returned before touching the filesystem

    def test_audio_music_feed_noop_without_skill(self):
        bc = self.bc
        saved = sys.modules.pop("skill_standby_audio_detect", None)
        try:
            # Must simply return without error when the skill isn't loaded.
            self.assertIsNone(bc._audio_music_feed(b"audio", 16000))
        finally:
            if saved is not None:
                sys.modules["skill_standby_audio_detect"] = saved

    def test_audio_music_feed_forwards_to_skill(self):
        bc = self.bc
        skill = types.ModuleType("skill_standby_audio_detect")
        calls = []
        skill.feed_audio = lambda audio, sr: calls.append((audio, sr))
        with mock.patch.dict(sys.modules,
                             {"skill_standby_audio_detect": skill}):
            bc._audio_music_feed(b"chunk", 16000)
        self.assertEqual(calls, [(b"chunk", 16000)])

    def test_audio_music_feed_swallows_skill_exception(self):
        bc = self.bc
        skill = types.ModuleType("skill_standby_audio_detect")

        def _raise(audio, sr):
            raise RuntimeError("detector blew up")

        skill.feed_audio = _raise
        with mock.patch.dict(sys.modules,
                             {"skill_standby_audio_detect": skill}):
            # Must not propagate.
            self.assertIsNone(bc._audio_music_feed(b"chunk", 16000))


# ════════════════════════════════════════════════════════════════════════════
#  Focus-tracker background loop (single bounded iteration, no real thread)
# ════════════════════════════════════════════════════════════════════════════
class FocusTrackerLoopTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        self._orig_state = dict(bc._focused_window_state)
        self._was_set = bc._focus_tracker_stop.is_set()
        self.addCleanup(self._restore)

    def _restore(self):
        bc = self.bc
        bc._focused_window_state.clear()
        bc._focused_window_state.update(self._orig_state)
        if self._was_set:
            bc._focus_tracker_stop.set()
        else:
            bc._focus_tracker_stop.clear()

    def _one_shot_stop(self):
        """A stop-event stand-in: is_set() stays False so the while-guard lets
        the body run, but wait(timeout) returns True so the loop breaks after
        exactly one iteration (and never actually sleeps)."""
        ev = mock.MagicMock()
        ev.is_set.return_value = False
        ev.wait.return_value = True
        return ev

    def test_loop_runs_one_iteration_then_stops_on_event(self):
        bc = self.bc
        bc._focused_window_state["hwnd"] = 1  # previous != new -> changed branch
        with mock.patch.object(bc, "_focus_tracker_stop", self._one_shot_stop()), \
             mock.patch.object(bc, "_read_focused_window",
                               return_value=(999, "Tracked", (1, 2, 300, 400))):
            bc._focus_tracker_loop()   # returns promptly after one iteration
        self.assertEqual(bc._focused_window_state["hwnd"], 999)
        self.assertEqual(bc._focused_window_state["title"], "Tracked")
        self.assertEqual(bc._focused_window_state["rect"], (1, 2, 300, 400))

    def test_loop_swallows_read_exception(self):
        bc = self.bc
        with mock.patch.object(bc, "_focus_tracker_stop", self._one_shot_stop()), \
             mock.patch.object(bc, "_read_focused_window",
                               side_effect=RuntimeError("win32 gone")):
            bc._focus_tracker_loop()   # must not raise

    def test_start_focus_tracker_spawns_daemon(self):
        bc = self.bc
        fake_thread = mock.MagicMock()
        with mock.patch.object(bc.threading, "Thread",
                               return_value=fake_thread) as Thread:
            bc._start_focus_tracker()
        Thread.assert_called_once()
        self.assertTrue(Thread.call_args.kwargs.get("daemon"))
        fake_thread.start.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════
#  load_skills happy path against a throwaway skills dir
# ════════════════════════════════════════════════════════════════════════════
class LoadSkillsHappyPathTests(SectionFiveBase):
    def test_load_skills_imports_and_registers(self):
        import os
        import tempfile

        bc = self.bc
        tmp = tempfile.mkdtemp(prefix="jarvis_skills_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp,
                                                            ignore_errors=True))
        # A minimal flat skill that registers one action.
        skill_src = (
            "def _act_unit_demo(arg=''):\n"
            "    return 'demo:' + str(arg)\n"
            "def register(actions):\n"
            "    actions['unit_demo_action'] = _act_unit_demo\n"
        )
        with open(os.path.join(tmp, "unit_demo.py"), "w", encoding="utf-8") as fh:
            fh.write(skill_src)

        actions = dict(bc.ACTIONS)        # work on a copy; never mutate the real dict
        orig_loaded = set(bc._loaded_skill_names)
        self.addCleanup(lambda: (bc._loaded_skill_names.clear(),
                                 bc._loaded_skill_names.update(orig_loaded)))
        saved_mod = sys.modules.pop("skill_unit_demo", None)
        self.addCleanup(lambda: sys.modules.pop("skill_unit_demo", None))
        if saved_mod is not None:
            self.addCleanup(lambda: sys.modules.__setitem__("skill_unit_demo",
                                                            saved_mod))

        with mock.patch.object(bc, "SKILLS_ENABLED", True), \
             mock.patch.object(bc, "SKILLS_DIR", tmp), \
             mock.patch.object(bc, "ACTIONS", actions), \
             mock.patch.object(bc, "_loaded_skill_names", set()):
            bc.load_skills()

        self.assertIn("unit_demo_action", actions)
        self.assertEqual(actions["unit_demo_action"]("x"), "demo:x")
        self.assertIn("skill_unit_demo", sys.modules)

    def test_load_skills_handles_broken_skill_without_raising(self):
        import os
        import tempfile

        bc = self.bc
        tmp = tempfile.mkdtemp(prefix="jarvis_skills_bad_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp,
                                                            ignore_errors=True))
        with open(os.path.join(tmp, "broken.py"), "w", encoding="utf-8") as fh:
            fh.write("import this_module_does_not_exist_zzz\n")
        self.addCleanup(lambda: sys.modules.pop("skill_broken", None))

        with mock.patch.object(bc, "SKILLS_ENABLED", True), \
             mock.patch.object(bc, "SKILLS_DIR", tmp), \
             mock.patch.object(bc, "ACTIONS", dict(bc.ACTIONS)), \
             mock.patch.object(bc, "_loaded_skill_names", set()):
            bc.load_skills()  # must swallow the import error and continue


# ════════════════════════════════════════════════════════════════════════════
#  ACTIONS dispatch dict wiring
# ════════════════════════════════════════════════════════════════════════════
class ActionsDictTests(SectionFiveBase):
    def test_actions_is_dict_of_callables(self):
        bc = self.bc
        self.assertIsInstance(bc.ACTIONS, dict)
        self.assertTrue(bc.ACTIONS)
        for name, fn in bc.ACTIONS.items():
            self.assertTrue(callable(fn), f"{name} -> not callable")

    def test_known_action_names_registered(self):
        bc = self.bc
        for name in ("open_url", "see_screen", "shutdown_jarvis", "run_shell",
                     "replay_last_action", "switch_llm", "reset_memory",
                     "ambient_mode", "version_info", "test_mic"):
            self.assertIn(name, bc.ACTIONS)

    def test_shutdown_aliases_share_one_handler(self):
        bc = self.bc
        target = bc.ACTIONS["shutdown_jarvis"]
        for alias in ("shut_down", "exit_jarvis", "quit_jarvis",
                      "power_off_jarvis", "turn_off_jarvis"):
            self.assertIs(bc.ACTIONS[alias], target)

    def test_reset_llm_cache_aliases_clear_llm_cache(self):
        bc = self.bc
        self.assertIs(bc.ACTIONS["reset_llm_cache"], bc.ACTIONS["clear_llm_cache"])

    def test_ambient_on_off_lambdas_callable(self):
        bc = self.bc
        # The lambda wrappers accept an optional arg and delegate to
        # _act_ambient_mode_set; verify they fire with the right boolean.
        with mock.patch.object(bc, "_act_ambient_mode_set",
                               return_value="set") as setter:
            bc.ACTIONS["ambient_mode_on"]("")
            bc.ACTIONS["ambient_mode_off"]("")
        setter.assert_has_calls([mock.call(True), mock.call(False)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
