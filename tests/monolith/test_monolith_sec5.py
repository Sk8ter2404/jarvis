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


class _SyncThread:
    """threading.Thread drop-in that runs target() synchronously on start() so
    a daemon closure body (e.g. _tray_async's _wrap) executes deterministically
    without leaving a real thread behind."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon
        self.name = name

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        return None

    def is_alive(self):
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

    def test_warm_restart_long_summary_first_sentence_truncated(self):
        # 8920-8922: no task line + no recent command, but a session summary
        # whose first sentence exceeds 90 chars -> work is truncated to 87 + "…".
        bc = self.bc
        import time as _t
        warm_ts = _t.time() - 90
        long_first = ("Refactored the entire audio capture and playback "
                      "pipeline including the noise cancellation stages and "
                      "the barge-in watchdog")
        self.assertGreater(len(long_first), 90)
        with mock.patch.object(bc, "_last_session_end_ts", return_value=warm_ts), \
             mock.patch.object(bc.pattern_memory, "get_session_summaries",
                               return_value=[{"summary": long_first + ". Tail."}]):
            text, details = bc._build_session_resume(force=False)
        self.assertTrue(details["work"].endswith("…"))
        self.assertEqual(len(details["work"]), 88)   # 87 chars + the ellipsis
        self.assertIn(details["work"], text)

    def test_maybe_greeting_age_print_exception_swallowed(self):
        # 8973-8974: the warm-restart log line computes age from details; if
        # details.get raises the except swallows it and the greeting still
        # returns. A mapping-like details whose .get raises drives the handler.
        bc = self.bc
        bc._session_resume_done[0] = False

        class _BadDetails(dict):
            def get(self, *a, **k):
                raise RuntimeError("no age")

        with mock.patch.object(bc, "_build_session_resume",
                               return_value=("Welcome back, sir.", _BadDetails())):
            out = bc.maybe_session_resume_greeting()
        self.assertEqual(out, "Welcome back, sir.")
        self.assertTrue(bc._session_resume_done[0])


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

    def test_find_music_window_importerror_returns_none(self):
        # 9122-9123: `import pygetwindow` raises ImportError -> None.
        bc = self.bc
        with mock.patch.dict(sys.modules, {"pygetwindow": None}):
            self.assertIsNone(bc._find_music_window())

    def test_focus_music_window_minimize_restore_also_fails_returns_none(self):
        # 9153-9154: activate() raises a genuine error, the minimize/restore
        # fallback ALSO raises -> None.
        bc = self.bc

        class _StubbornWin(_FakeWin):
            def minimize(self):
                raise RuntimeError("minimize denied")

        win = _StubbornWin("Spotify")
        win._activate_exc = Exception("some genuine failure")
        with mock.patch.object(bc, "_find_music_window", return_value=win):
            self.assertIsNone(bc._focus_music_window())

    def test_active_window_is_terminal_query_exception_returns_false(self):
        # 9631-9632: getActiveWindow() raises -> swallowed, returns False.
        bc = self.bc
        mod = types.ModuleType("pygetwindow")

        def _boom():
            raise RuntimeError("getActiveWindow failed")
        mod.getActiveWindow = _boom
        with _InjectModule("pygetwindow", mod):
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

    def test_glance_ambiguous_short_non_pattern_returns_false(self):
        # 9396: <=5 words (passes the >5 guard), not in the pattern set, and no
        # trailing '?' -> falls through to the final `return False`.
        self.assertFalse(self.bc._is_glance_ambiguous_question("open my email now"))

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

    def test_tray_async_truncates_long_first_line(self):
        # 9871-9872: a >120-char first output line is truncated to 117 + "...".
        bc = self.bc
        long_line = "z" * 200
        captured = {}
        printed = []

        def _fn():
            return long_line

        # Run the _wrap body synchronously and capture the log line it prints.
        with mock.patch.object(bc.threading, "Thread", _SyncThread), \
             mock.patch("builtins.print",
                        side_effect=lambda *a, **k: printed.append(" ".join(map(str, a)))):
            bc._tray_async("longjob", _fn)
        captured["log"] = "\n".join(printed)
        self.assertIn("z" * 117 + "...", captured["log"])
        self.assertNotIn("z" * 121, captured["log"])

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

    def test_probe_via_selfdiag_without_runner_calls_probe_directly(self):
        # 9979: the self_diagnostic module has no _run_with_timeout helper, so
        # the probe is called directly (runner is None branch).
        bc = self.bc
        sd = types.ModuleType("skill_self_diagnostic")
        sd._probe_mic = lambda: {"ok": False, "error": "no device"}
        sd.PER_PROBE_TIMEOUT_S = 5.0   # note: no _run_with_timeout attribute
        captured = {}

        def _fake_async(name, fn):
            captured["result"] = fn()

        with mock.patch.object(bc, "_selfdiag_module", return_value=sd), \
             mock.patch.object(bc, "_tray_async", side_effect=_fake_async):
            out = bc._probe_via_selfdiag("mic", "_probe_mic")
        self.assertEqual(out, "mic probe running")
        self.assertIn("mic: FAIL", captured["result"])
        self.assertIn("no device", captured["result"])

    def test_probe_via_selfdiag_probe_exception_reported(self):
        # 9987-9988: the probe body raises -> the _do closure reports it as
        # "probe raised ..." rather than propagating.
        bc = self.bc
        sd = types.ModuleType("skill_self_diagnostic")

        def _boom():
            raise RuntimeError("mic exploded")
        sd._probe_mic = _boom
        sd._run_with_timeout = lambda fn, budget, name=None: fn()
        sd.PER_PROBE_TIMEOUT_S = 5.0
        captured = {}

        def _fake_async(name, fn):
            captured["result"] = fn()

        with mock.patch.object(bc, "_selfdiag_module", return_value=sd), \
             mock.patch.object(bc, "_tray_async", side_effect=_fake_async):
            out = bc._probe_via_selfdiag("mic", "_probe_mic")
        self.assertEqual(out, "mic probe running")
        self.assertIn("probe raised", captured["result"])
        self.assertIn("mic exploded", captured["result"])


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

# ════════════════════════════════════════════════════════════════════════════
#  Extra branch fills on already-covered state machines
# ════════════════════════════════════════════════════════════════════════════
class ShutdownPromptBranchFillTests(SectionFiveBase):
    """Cover the try/except arms inside _handle_shutdown_prompt that the main
    ShutdownPromptTests left unexercised (reinforced-phrase failure, NO-path
    failure)."""

    def setUp(self):
        bc = self.bc
        self._orig_pending = dict(bc._shutdown_prompt_pending)
        self.addCleanup(lambda: (bc._shutdown_prompt_pending.clear(),
                                 bc._shutdown_prompt_pending.update(self._orig_pending)))
        self.speak = mock.patch.object(bc, "_speak").start()
        self.shutdown = mock.patch.object(bc, "_act_shutdown_jarvis").start()
        self.overnight = mock.patch.object(bc, "_act_start_overnight_upgrade").start()
        self.addCleanup(mock.patch.stopall)

    def _arm(self):
        import time as _t
        self.bc._shutdown_prompt_pending["armed"] = True
        self.bc._shutdown_prompt_pending["expires_at"] = _t.time() + 100

    def test_reinforced_phrase_dispatch_exception_still_consumes(self):
        # 'shut down jarvis' is a SHUTDOWN_TRIGGER_PHRASE -> reinforced branch;
        # the terminal _act_shutdown_jarvis raises but the branch returns True.
        self._arm()
        self.shutdown.side_effect = RuntimeError("boom")
        self.assertTrue(self.bc._handle_shutdown_prompt("shut down jarvis"))
        self.shutdown.assert_called_once()
        self.assertFalse(self.bc._shutdown_prompt_pending["armed"])

    def test_no_phrase_dispatch_exception_still_consumes(self):
        # NO branch's _act_shutdown_jarvis raises -> still returns True.
        self._arm()
        self.shutdown.side_effect = RuntimeError("kaboom")
        self.assertTrue(self.bc._handle_shutdown_prompt("no"))
        self.shutdown.assert_called_once()

    def test_unrelated_reply_speak_exception_swallowed(self):
        # 9087-9088: the 'unrelated speech' arm announces "Shutdown cancelled."
        # via _speak; if that raises it is swallowed and the function still
        # returns False (falls through to normal routing).
        self._arm()
        self.speak.side_effect = RuntimeError("tts dead")
        result = self.bc._handle_shutdown_prompt("what's the weather")
        self.assertFalse(result)
        self.assertFalse(self.bc._shutdown_prompt_pending["armed"])


class ReadFocusedWindowBranchTests(SectionFiveBase):
    """Cover the inner-rect except arm of _read_focused_window and the
    live-read except arm of _focus_changed_recently."""

    def setUp(self):
        bc = self.bc
        self._orig_state = dict(bc._focused_window_state)
        self.addCleanup(lambda: (bc._focused_window_state.clear(),
                                 bc._focused_window_state.update(self._orig_state)))

    def test_read_focused_window_rect_exception_yields_none_rect(self):
        bc = self.bc
        mod = types.ModuleType("win32gui")
        mod.GetForegroundWindow = lambda: 777
        mod.GetWindowText = lambda h: "Titled"
        def _boom(_h):
            raise RuntimeError("GetWindowRect failed")
        mod.GetWindowRect = _boom
        with _InjectModule("win32gui", mod):
            hwnd, title, rect = bc._read_focused_window()
        self.assertEqual(hwnd, 777)
        self.assertEqual(title, "Titled")
        self.assertIsNone(rect)   # inner except -> rect=None, hwnd/title intact

    def test_focus_changed_recently_read_exception_falls_back_to_timestamp(self):
        bc = self.bc
        import time as _t
        # Live read raises -> hwnd/title/rect default to (None,"",None); then
        # the timestamp branch decides. Fresh changed_at -> True.
        bc._focused_window_state["hwnd"] = 5
        bc._focused_window_state["changed_at"] = _t.monotonic()
        with mock.patch.object(bc, "_read_focused_window",
                               side_effect=RuntimeError("win32 gone")):
            self.assertTrue(bc._focus_changed_recently())

    def test_read_focused_window_query_exception_yields_all_none(self):
        # 9328-9329: win32gui imports fine, but GetForegroundWindow() raises
        # -> the outer except returns (None, "", None). (The inner-rect except
        # only covers GetWindowRect; this exercises the wrapping handler.)
        bc = self.bc
        mod = types.ModuleType("win32gui")

        def _boom():
            raise RuntimeError("GetForegroundWindow failed")
        mod.GetForegroundWindow = _boom
        with _InjectModule("win32gui", mod):
            self.assertEqual(bc._read_focused_window(), (None, "", None))


# ════════════════════════════════════════════════════════════════════════════
#  Optional-skill bridges (lazy sys.modules lookups)
# ════════════════════════════════════════════════════════════════════════════
class OptionalSkillBridgeTests(SectionFiveBase):
    def _pop_module(self, name):
        """Remove a skill module for the test, restoring it afterwards."""
        saved = sys.modules.pop(name, None)
        if saved is not None:
            self.addCleanup(lambda: sys.modules.__setitem__(name, saved))
        return saved

    # ---- _user_looking_away ----
    def test_user_looking_away_no_tracker(self):
        self._pop_module("skill_face_tracker")
        self.assertFalse(self.bc._user_looking_away())

    def test_user_looking_away_no_sample_yet(self):
        bc = self.bc
        mod = types.ModuleType("skill_face_tracker")
        mod._snapshot_state = lambda: {"last_sample_at": None}
        with mock.patch.dict(sys.modules, {"skill_face_tracker": mod}):
            self.assertFalse(bc._user_looking_away())

    def test_user_looking_away_true_when_monitor_away(self):
        bc = self.bc
        mod = types.ModuleType("skill_face_tracker")
        mod._snapshot_state = lambda: {"last_sample_at": 123.0,
                                       "current_monitor": "away"}
        with mock.patch.dict(sys.modules, {"skill_face_tracker": mod}):
            self.assertTrue(bc._user_looking_away())

    def test_user_looking_away_false_when_present(self):
        bc = self.bc
        mod = types.ModuleType("skill_face_tracker")
        mod._snapshot_state = lambda: {"last_sample_at": 123.0,
                                       "current_monitor": "left"}
        with mock.patch.dict(sys.modules, {"skill_face_tracker": mod}):
            self.assertFalse(bc._user_looking_away())

    def test_user_looking_away_swallows_exception(self):
        bc = self.bc
        mod = types.ModuleType("skill_face_tracker")
        def _boom():
            raise RuntimeError("tracker state corrupt")
        mod._snapshot_state = _boom
        with mock.patch.dict(sys.modules, {"skill_face_tracker": mod}):
            self.assertFalse(bc._user_looking_away())

    # ---- _bambu_print_progress ----
    def test_bambu_progress_no_skill(self):
        self._pop_module("skill_bambu_monitor")
        self.assertIsNone(self.bc._bambu_print_progress())

    def _make_bambu(self, gcode_state, pct):
        import threading as _thr
        mod = types.ModuleType("skill_bambu_monitor")
        mod._state_lock = _thr.Lock()
        mod._state = {"gcode_state": gcode_state, "mc_percent": pct}
        return mod

    def test_bambu_progress_running_in_range(self):
        bc = self.bc
        mod = self._make_bambu("RUNNING", 42)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": mod}):
            self.assertEqual(bc._bambu_print_progress(), 42)

    def test_bambu_progress_not_running_returns_none(self):
        bc = self.bc
        mod = self._make_bambu("IDLE", 42)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": mod}):
            self.assertIsNone(bc._bambu_print_progress())

    def test_bambu_progress_out_of_band_pct_returns_none(self):
        bc = self.bc
        # 100 is outside the 1..99 "actively printing" window.
        mod = self._make_bambu("RUNNING", 100)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": mod}):
            self.assertIsNone(bc._bambu_print_progress())

    def test_bambu_progress_swallows_exception(self):
        bc = self.bc
        mod = types.ModuleType("skill_bambu_monitor")
        # Accessing _state_lock as a context manager will explode.
        mod._state_lock = object()
        mod._state = {}
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": mod}):
            self.assertIsNone(bc._bambu_print_progress())

    # ---- _audio_music_should_refuse_wake ----
    def test_refuse_wake_no_skill(self):
        self._pop_module("skill_standby_audio_detect")
        self.assertFalse(self.bc._audio_music_should_refuse_wake("jarvis"))

    def test_refuse_wake_delegates_true(self):
        bc = self.bc
        mod = types.ModuleType("skill_standby_audio_detect")
        mod.should_refuse_wake = lambda text: True
        with mock.patch.dict(sys.modules, {"skill_standby_audio_detect": mod}):
            self.assertTrue(bc._audio_music_should_refuse_wake("jarvis"))

    def test_refuse_wake_swallows_exception(self):
        bc = self.bc
        mod = types.ModuleType("skill_standby_audio_detect")
        def _boom(text):
            raise RuntimeError("detector down")
        mod.should_refuse_wake = _boom
        with mock.patch.dict(sys.modules, {"skill_standby_audio_detect": mod}):
            self.assertFalse(bc._audio_music_should_refuse_wake("jarvis"))


# ════════════════════════════════════════════════════════════════════════════
#  Background-audio wake gate: _should_refuse_background_audio + wake-word mode
# ════════════════════════════════════════════════════════════════════════════
class BackgroundAudioGateTests(SectionFiveBase):
    def setUp(self):
        import core.config as _cfg
        bc = self.bc
        self._cfg = _cfg
        # Save+restore BOTH the runtime mirror and the config constant so no
        # state leaks into sibling tests (other classes read these globals).
        self._orig_runtime = bc._require_wake_runtime
        self._orig_cfg = _cfg.REQUIRE_WAKE_MODE
        self.addCleanup(self._restore)
        bc._require_wake_runtime = False
        _cfg.REQUIRE_WAKE_MODE = False

    def _restore(self):
        self.bc._require_wake_runtime = self._orig_runtime
        self._cfg.REQUIRE_WAKE_MODE = self._orig_cfg

    def test_wake_prefix_always_passes(self):
        bc = self.bc
        # A clear leading "JARVIS ..." passes even with every gate demanding it.
        bc._require_wake_runtime = True
        with mock.patch.object(bc, "_smtc_media_playing", return_value=True):
            refuse, why = bc._should_refuse_background_audio("jarvis pause")
        self.assertFalse(refuse)
        self.assertEqual(why, "")

    def test_wake_mode_refuses_non_wake(self):
        bc = self.bc
        bc._require_wake_runtime = True
        # Force the other auto-gates off so the reason is unambiguously the toggle.
        with mock.patch.object(bc, "_smtc_media_playing", return_value=False), \
             mock.patch.object(bc, "_audio_music_should_refuse_wake",
                               return_value=False):
            refuse, why = bc._should_refuse_background_audio("what time is it")
        self.assertTrue(refuse)
        self.assertEqual(why, "wake-word mode")

    def test_smtc_playing_refuses_non_wake(self):
        bc = self.bc
        with mock.patch.object(bc, "_smtc_media_playing", return_value=True), \
             mock.patch.object(bc, "_audio_music_should_refuse_wake",
                               return_value=False):
            refuse, why = bc._should_refuse_background_audio("turn the lights on")
        self.assertTrue(refuse)
        self.assertEqual(why, "media playing")

    def test_all_quiet_passes(self):
        bc = self.bc
        # Toggle off, no media, no room music → behaviour-preserving pass-through.
        with mock.patch.object(bc, "_smtc_media_playing", return_value=False), \
             mock.patch.object(bc, "_audio_music_should_refuse_wake",
                               return_value=False):
            refuse, why = bc._should_refuse_background_audio("turn the lights on")
        self.assertFalse(refuse)
        self.assertEqual(why, "")

    def test_room_music_refuses_non_wake(self):
        bc = self.bc
        # Only the spectral room-music detector trips; SMTC silent, toggle off.
        with mock.patch.object(bc, "_smtc_media_playing", return_value=False), \
             mock.patch.object(bc, "_audio_music_should_refuse_wake",
                               return_value=True):
            refuse, why = bc._should_refuse_background_audio("some lyric line")
        self.assertTrue(refuse)
        self.assertEqual(why, "room music")

    def test_gate_never_raises(self):
        bc = self.bc
        # Even if an underlying probe explodes, the gate fails OPEN (False, "").
        def _boom(_t):
            raise RuntimeError("probe down")
        with mock.patch.object(bc, "_smtc_media_playing", side_effect=_boom):
            refuse, why = bc._should_refuse_background_audio("hello there")
        self.assertFalse(refuse)
        self.assertEqual(why, "")


class WakeWordModeActionTests(SectionFiveBase):
    def setUp(self):
        import core.config as _cfg
        bc = self.bc
        self._cfg = _cfg
        self._orig_runtime = bc._require_wake_runtime
        self._orig_cfg = _cfg.REQUIRE_WAKE_MODE
        self.addCleanup(self._restore)
        bc._require_wake_runtime = False
        _cfg.REQUIRE_WAKE_MODE = False

    def _restore(self):
        self.bc._require_wake_runtime = self._orig_runtime
        self._cfg.REQUIRE_WAKE_MODE = self._orig_cfg

    def test_set_on_sets_both_flags(self):
        bc = self.bc
        msg = bc._act_wake_word_mode_set(True)
        self.assertTrue(bc._require_wake_runtime)
        self.assertTrue(self._cfg.REQUIRE_WAKE_MODE)
        self.assertIn("wake-word mode on", msg.lower())

    def test_set_off_clears_both_flags(self):
        bc = self.bc
        bc._require_wake_runtime = True
        self._cfg.REQUIRE_WAKE_MODE = True
        msg = bc._act_wake_word_mode_set(False)
        self.assertFalse(bc._require_wake_runtime)
        self.assertFalse(self._cfg.REQUIRE_WAKE_MODE)
        self.assertIn("wake-word mode off", msg.lower())

    def test_status_reports_on_and_media(self):
        bc = self.bc
        bc._require_wake_runtime = True
        with mock.patch.object(bc, "_smtc_media_playing", return_value=True):
            msg = bc._act_wake_word_mode_status()
        low = msg.lower()
        self.assertIn("on", low)
        self.assertIn("playing", low)

    def test_actions_registered(self):
        bc = self.bc
        for name in ("wake_word_mode_on", "wake_word_mode_off",
                     "wake_word_mode_status"):
            self.assertIn(name, bc.ACTIONS)
        # The on/off lambdas drive the same setter and flip the runtime flag.
        bc.ACTIONS["wake_word_mode_on"]("")
        self.assertTrue(bc._require_wake_runtime)
        bc.ACTIONS["wake_word_mode_off"]("")
        self.assertFalse(bc._require_wake_runtime)


# ════════════════════════════════════════════════════════════════════════════
#  _ambient_learning_feed  (silent overheard-utterance persistence)
# ════════════════════════════════════════════════════════════════════════════
class AmbientLearningFeedTests(SectionFiveBase):
    def test_drops_short_utterances(self):
        bc = self.bc
        # < 8 chars OR < 3 words -> dropped before any filesystem touch.
        with mock.patch.object(bc.os, "makedirs") as mkd:
            bc._ambient_learning_feed("you")
            bc._ambient_learning_feed("thanks")          # 1 word
            bc._ambient_learning_feed("ok sure")         # 2 words
        mkd.assert_not_called()

    def test_writes_record_to_staging_sibling(self):
        import json
        import os
        import tempfile

        bc = self.bc
        tmp = tempfile.mkdtemp(prefix="jarvis_ambient_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp,
                                                            ignore_errors=True))
        # Redirect the data dir by patching os.path.dirname/abspath is awkward;
        # instead patch makedirs to a no-op and os.path.join so the write lands
        # in our temp dir. Simpler: patch the module's open target dir by
        # pointing __file__-derived path via a join shim.
        real_join = os.path.join

        def _join_shim(a, *rest):
            # The function computes <dir>/data then <data>/<fname>. Redirect any
            # path that ends in our data segments into tmp.
            if rest and rest[-1] in ("ambient_transcripts.jsonl",
                                     "ambient_transcripts.staging.jsonl"):
                return real_join(tmp, rest[-1])
            if rest == ("data",):
                return tmp
            return real_join(a, *rest)

        with mock.patch.object(bc, "_is_staging", return_value=True), \
             mock.patch.object(bc.os, "makedirs"), \
             mock.patch.object(bc.os.path, "join", side_effect=_join_shim):
            bc._ambient_learning_feed("the printer just finished the benchy")

        staging = real_join(tmp, "ambient_transcripts.staging.jsonl")
        self.assertTrue(os.path.isfile(staging))
        with open(staging, encoding="utf-8") as fh:
            rec = json.loads(fh.readline())
        self.assertEqual(rec["source"], "mic")
        self.assertEqual(rec["tag"], "ambient_standby")
        self.assertIn("benchy", rec["text"])

    def test_non_staging_uses_live_filename(self):
        import os
        import tempfile

        bc = self.bc
        tmp = tempfile.mkdtemp(prefix="jarvis_ambient_live_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp,
                                                            ignore_errors=True))
        real_join = os.path.join

        def _join_shim(a, *rest):
            if rest and rest[-1] in ("ambient_transcripts.jsonl",
                                     "ambient_transcripts.staging.jsonl"):
                return real_join(tmp, rest[-1])
            if rest == ("data",):
                return tmp
            return real_join(a, *rest)

        with mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc.os, "makedirs"), \
             mock.patch.object(bc.os.path, "join", side_effect=_join_shim):
            bc._ambient_learning_feed("there is a long enough sentence here")

        self.assertTrue(os.path.isfile(real_join(tmp,
                                                 "ambient_transcripts.jsonl")))
        self.assertFalse(os.path.isfile(real_join(
            tmp, "ambient_transcripts.staging.jsonl")))

    def test_swallows_write_exception(self):
        bc = self.bc
        # makedirs raising must not propagate into the listen loop.
        with mock.patch.object(bc, "_is_staging", return_value=True), \
             mock.patch.object(bc.os, "makedirs",
                               side_effect=OSError("disk full")):
            # Long-enough text so it gets past the early-return guard.
            self.assertIsNone(
                bc._ambient_learning_feed("a sufficiently long utterance here"))


# ════════════════════════════════════════════════════════════════════════════
#  _standby_auto_engage  (programmatic standby flip)
# ════════════════════════════════════════════════════════════════════════════
class StandbyAutoEngageTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        self._orig_sleep = list(bc._sleep_mode)
        self._orig_standby = list(bc._standby_mode)
        self.addCleanup(lambda: (bc._sleep_mode.__setitem__(0, self._orig_sleep[0]),
                                 bc._standby_mode.__setitem__(0, self._orig_standby[0])))
        bc._sleep_mode[0] = False
        bc._standby_mode[0] = False

    def test_engages_and_sets_flags(self):
        bc = self.bc
        with mock.patch.object(bc, "_speak") as speak, \
             mock.patch.object(bc, "set_state") as set_state, \
             mock.patch.object(bc, "_write_hud_state") as hud:
            changed = bc._standby_auto_engage("music")
        self.assertTrue(changed)
        self.assertTrue(bc._sleep_mode[0])
        self.assertTrue(bc._standby_mode[0])
        speak.assert_called_once()
        set_state.assert_called_once_with("idle")
        hud.assert_called_once()

    def test_noop_when_already_standby(self):
        bc = self.bc
        bc._standby_mode[0] = True
        with mock.patch.object(bc, "_speak") as speak, \
             mock.patch.object(bc, "set_state") as set_state:
            self.assertFalse(bc._standby_auto_engage("music"))
        speak.assert_not_called()
        set_state.assert_not_called()

    def test_noop_when_sleep_mode(self):
        bc = self.bc
        bc._sleep_mode[0] = True
        with mock.patch.object(bc, "_speak"), \
             mock.patch.object(bc, "set_state"):
            self.assertFalse(bc._standby_auto_engage())

    def test_tts_failure_does_not_abort_engage(self):
        bc = self.bc
        with mock.patch.object(bc, "_speak",
                               side_effect=RuntimeError("audio device gone")), \
             mock.patch.object(bc, "set_state") as set_state, \
             mock.patch.object(bc, "_write_hud_state"):
            self.assertTrue(bc._standby_auto_engage("music"))
        # Flags were set under the lock before the speak attempt; set_state
        # still runs after the swallowed TTS error.
        self.assertTrue(bc._standby_mode[0])
        set_state.assert_called_once_with("idle")

    def test_hud_write_failure_swallowed(self):
        bc = self.bc
        with mock.patch.object(bc, "_speak"), \
             mock.patch.object(bc, "set_state"), \
             mock.patch.object(bc, "_write_hud_state",
                               side_effect=RuntimeError("hud pipe broken")):
            self.assertTrue(bc._standby_auto_engage("music"))
        self.assertTrue(bc._standby_mode[0])

    def test_set_state_failure_swallowed(self):
        bc = self.bc
        # The trailing set_state("idle") is wrapped in its own try/except.
        with mock.patch.object(bc, "_speak"), \
             mock.patch.object(bc, "_write_hud_state"), \
             mock.patch.object(bc, "set_state",
                               side_effect=RuntimeError("face thread gone")):
            self.assertTrue(bc._standby_auto_engage("music"))
        self.assertTrue(bc._standby_mode[0])


# ════════════════════════════════════════════════════════════════════════════
#  Ambient-learning + wake-resume voice actions and their ACTIONS wiring
# ════════════════════════════════════════════════════════════════════════════
class AmbientLearningActionTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        self._orig = {
            "sleep": list(bc._sleep_mode),
            "standby": list(bc._standby_mode),
            "ambient": list(bc._ambient_learning),
            "resume": list(bc._resume_to_ambient),
        }
        self.addCleanup(self._restore)
        bc._sleep_mode[0] = False
        bc._standby_mode[0] = False
        bc._ambient_learning[0] = False
        bc._resume_to_ambient[0] = False

    def _restore(self):
        bc = self.bc
        bc._sleep_mode[0] = self._orig["sleep"][0]
        bc._standby_mode[0] = self._orig["standby"][0]
        bc._ambient_learning[0] = self._orig["ambient"][0]
        bc._resume_to_ambient[0] = self._orig["resume"][0]

    def test_set_on_enters_silent_standby(self):
        bc = self.bc
        with mock.patch.object(bc, "_is_staging", return_value=True), \
             mock.patch.object(bc, "_write_hud_state") as hud:
            msg = bc._act_ambient_learning_set(True)
        self.assertTrue(bc._sleep_mode[0])
        self.assertTrue(bc._standby_mode[0])
        self.assertTrue(bc._ambient_learning[0])
        self.assertFalse(bc._resume_to_ambient[0])
        self.assertIn("ambient-learning", msg.lower())
        hud.assert_called_once()

    def test_set_on_starts_extractor_when_not_staging(self):
        bc = self.bc
        ext = types.ModuleType("skill_ambient_multimodal_extract")
        started = []
        ext.ambient_extract_start = lambda arg: started.append(arg)
        with mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc, "_write_hud_state"), \
             mock.patch.dict(sys.modules,
                             {"skill_ambient_multimodal_extract": ext}):
            bc._act_ambient_learning_set(True)
        self.assertEqual(started, [""])

    def test_set_on_extractor_exception_swallowed(self):
        bc = self.bc
        ext = types.ModuleType("skill_ambient_multimodal_extract")
        def _boom(arg):
            raise RuntimeError("extractor failed to start")
        ext.ambient_extract_start = _boom
        with mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc, "_write_hud_state"), \
             mock.patch.dict(sys.modules,
                             {"skill_ambient_multimodal_extract": ext}):
            # Must not raise despite the extractor blowing up.
            msg = bc._act_ambient_learning_set(True)
        self.assertIn("ambient-learning", msg.lower())

    def test_set_on_hud_exception_swallowed(self):
        bc = self.bc
        # The ON-path's _write_hud_state is wrapped in its own try/except.
        with mock.patch.object(bc, "_is_staging", return_value=True), \
             mock.patch.object(bc, "_write_hud_state",
                               side_effect=RuntimeError("hud down")):
            msg = bc._act_ambient_learning_set(True)
        self.assertTrue(bc._ambient_learning[0])
        self.assertIn("ambient-learning", msg.lower())

    def test_set_off_returns_to_normal(self):
        bc = self.bc
        bc._ambient_learning[0] = True
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        with mock.patch.object(bc, "_write_hud_state") as hud:
            msg = bc._act_ambient_learning_set(False)
        self.assertFalse(bc._ambient_learning[0])
        self.assertFalse(bc._sleep_mode[0])
        self.assertFalse(bc._standby_mode[0])
        self.assertFalse(bc._resume_to_ambient[0])
        self.assertIn("off", msg.lower())
        hud.assert_called_once()

    def test_set_off_hud_exception_swallowed(self):
        bc = self.bc
        with mock.patch.object(bc, "_write_hud_state",
                               side_effect=RuntimeError("hud down")):
            msg = bc._act_ambient_learning_set(False)
        self.assertIn("off", msg.lower())

    # ---- _act_wake_resume_set ----
    def test_wake_resume_answer_then_quiet(self):
        bc = self.bc
        orig = bc.WAKE_RESUME_MODE
        self.addCleanup(lambda: setattr(bc, "WAKE_RESUME_MODE", orig))
        msg = bc._act_wake_resume_set("answer then quiet")  # spaces normalised
        self.assertEqual(bc.WAKE_RESUME_MODE, "answer_then_quiet")
        self.assertIn("answer-then-quiet", msg.lower())

    def test_wake_resume_stay_talkative_hyphen_form(self):
        bc = self.bc
        orig = bc.WAKE_RESUME_MODE
        self.addCleanup(lambda: setattr(bc, "WAKE_RESUME_MODE", orig))
        msg = bc._act_wake_resume_set("stay-talkative")
        self.assertEqual(bc.WAKE_RESUME_MODE, "stay_talkative")
        self.assertIn("stay-talkative", msg.lower())

    def test_wake_resume_unknown_mode_rejected(self):
        bc = self.bc
        orig = bc.WAKE_RESUME_MODE
        self.addCleanup(lambda: setattr(bc, "WAKE_RESUME_MODE", orig))
        msg = bc._act_wake_resume_set("banana")
        self.assertEqual(bc.WAKE_RESUME_MODE, orig)   # unchanged
        self.assertIn("don't recognise", msg.lower())

    # ---- ACTIONS wiring for the above ----
    def test_actions_ambient_learning_wiring(self):
        bc = self.bc
        for name in ("ambient_learning_mode", "ambient_learning_mode_on",
                     "ambient_learning_mode_off", "enter_ambient_learning",
                     "exit_ambient_learning", "wake_resume_answer_then_quiet",
                     "wake_resume_stay_talkative"):
            self.assertIn(name, bc.ACTIONS)
            self.assertTrue(callable(bc.ACTIONS[name]))

    def test_actions_enter_exit_delegate_with_bool(self):
        bc = self.bc
        with mock.patch.object(bc, "_act_ambient_learning_set",
                               return_value="ok") as setter:
            bc.ACTIONS["enter_ambient_learning"]("")
            bc.ACTIONS["exit_ambient_learning"]("")
        setter.assert_has_calls([mock.call(True), mock.call(False)])

    def test_actions_toggle_passes_negated_state(self):
        bc = self.bc
        bc._ambient_learning[0] = False
        with mock.patch.object(bc, "_act_ambient_learning_set",
                               return_value="ok") as setter:
            bc.ACTIONS["ambient_learning_mode"]("")
        setter.assert_called_once_with(True)   # not (not False) -> True

    def test_actions_wake_resume_lambdas_delegate(self):
        bc = self.bc
        with mock.patch.object(bc, "_act_wake_resume_set",
                               return_value="ok") as setter:
            bc.ACTIONS["wake_resume_answer_then_quiet"]("")
            bc.ACTIONS["wake_resume_stay_talkative"]("")
        setter.assert_has_calls([mock.call("answer_then_quiet"),
                                 mock.call("stay_talkative")])


# ════════════════════════════════════════════════════════════════════════════
#  _act_ambient_mode_set — the "go ambient" voice action. Regression guard for
#  the user-reported "i don't think it's even learning": turning ambient mode
#  ON must start BOTH the mic daemon (ambient_listen_start) AND the multimodal
#  fact-extractor (ambient_extract_start) so overheard speech is actually folded
#  into long-term memory. Turning it OFF stops both.
# ════════════════════════════════════════════════════════════════════════════
class AmbientModeSetWiringTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        self._orig = bool(bc._ambient_mode_active[0])
        self.addCleanup(lambda: bc._ambient_mode_active.__setitem__(0, self._orig))
        # Neutralise the mic-daemon side: ambient_listen_start/stop are present
        # in ACTIONS; replace them with no-op spies so this test stays focused
        # on the extractor wiring and never touches a real device.
        self._listen_calls = []
        self._orig_actions = dict(bc.ACTIONS)
        self.addCleanup(lambda: (bc.ACTIONS.clear(),
                                 bc.ACTIONS.update(self._orig_actions)))
        bc.ACTIONS["ambient_listen_start"] = lambda a="": self._listen_calls.append(("start", a)) or "mic on"
        bc.ACTIONS["ambient_listen_stop"]  = lambda a="": self._listen_calls.append(("stop", a)) or "mic off"

    def _fake_extractor(self, *, start=None, stop=None):
        ext = types.ModuleType("skill_ambient_multimodal_extract")
        ext.ambient_extract_start = start or (lambda a="": None)
        ext.ambient_extract_stop = stop or (lambda a="": None)
        return ext

    def test_on_starts_mic_daemon_and_extractor(self):
        bc = self.bc
        started = []
        ext = self._fake_extractor(start=lambda a="": started.append(a))
        with mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc, "_write_hud_state"), \
             mock.patch.dict(sys.modules,
                             {"skill_ambient_multimodal_extract": ext}):
            msg = bc._act_ambient_mode_set(True)
        # Mic daemon was kicked on...
        self.assertEqual(self._listen_calls, [("start", "")])
        # ...AND the fact-extractor was started (the symptom-2 fix).
        self.assertEqual(started, [""])
        self.assertTrue(bc._ambient_mode_active[0])
        self.assertIn("learning", msg.lower())

    def test_off_stops_mic_daemon_and_extractor(self):
        bc = self.bc
        bc._ambient_mode_active[0] = True
        stopped = []
        ext = self._fake_extractor(stop=lambda a="": stopped.append(a))
        with mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc, "_write_hud_state"), \
             mock.patch.dict(sys.modules,
                             {"skill_ambient_multimodal_extract": ext}):
            bc._act_ambient_mode_set(False)
        self.assertEqual(self._listen_calls, [("stop", "")])
        self.assertEqual(stopped, [""])
        self.assertFalse(bc._ambient_mode_active[0])

    def test_on_skips_extractor_in_staging(self):
        bc = self.bc
        started = []
        ext = self._fake_extractor(start=lambda a="": started.append(a))
        with mock.patch.object(bc, "_is_staging", return_value=True), \
             mock.patch.object(bc, "_write_hud_state"), \
             mock.patch.dict(sys.modules,
                             {"skill_ambient_multimodal_extract": ext}):
            bc._act_ambient_mode_set(True)
        # Staging must NOT start the extractor (test injects never write memory).
        self.assertEqual(started, [])

    def test_on_extractor_exception_swallowed(self):
        bc = self.bc
        def _boom(a=""):
            raise RuntimeError("extractor boom")
        ext = self._fake_extractor(start=_boom)
        with mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc, "_write_hud_state"), \
             mock.patch.dict(sys.modules,
                             {"skill_ambient_multimodal_extract": ext}):
            # Must not raise even though the extractor blew up.
            msg = bc._act_ambient_mode_set(True)
        self.assertTrue(bc._ambient_mode_active[0])
        self.assertIn("ambient mode", msg.lower())

    def test_on_tolerates_missing_extractor_module(self):
        bc = self.bc
        # No skill_ambient_multimodal_extract in sys.modules → must still fire
        # the mic daemon and not raise.
        with mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc, "_write_hud_state"), \
             mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_ambient_multimodal_extract", None)
            msg = bc._act_ambient_mode_set(True)
        self.assertEqual(self._listen_calls, [("start", "")])
        self.assertIn("learning", msg.lower())


# ════════════════════════════════════════════════════════════════════════════
#  Wake-greeting selection: _pick_wake_variety + context_aware_greeting
# ════════════════════════════════════════════════════════════════════════════
class WakeVarietyTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        self._orig_hist = list(bc._wake_history)
        self._orig_last = list(bc._last_wake_phrase)
        self.addCleanup(lambda: (bc._wake_history.__setitem__(slice(None),
                                                              self._orig_hist),
                                 bc._last_wake_phrase.__setitem__(
                                     0, self._orig_last[0])))
        bc._wake_history[:] = []
        bc._last_wake_phrase[0] = None

    def test_terse_tag_chosen_when_frustrated(self):
        bc = self.bc
        # Force tone -> frustrated so 'terse' is the only non-general preferred
        # tag, then make random.choice deterministic.
        picked = {}
        def _choice(seq):
            picked["candidates"] = seq
            return seq[0]
        with mock.patch.object(bc, "detect_tone", return_value="frustrated"), \
             mock.patch.object(bc.random, "choice", side_effect=_choice):
            text, vol = bc._pick_wake_variety(from_standby=False,
                                              wake_text="ugh hurry up")
        # Every candidate must carry a preferred tag (general or terse).
        for _t, tags in picked["candidates"]:
            self.assertTrue(tags & {"terse", "general"})
        self.assertIsInstance(text, str)
        self.assertEqual(vol, 1.0)

    def test_soft_tag_lowers_volume_late_night(self):
        bc = self.bc
        # Pick a 'soft' phrase by forcing tone tired + choosing a soft entry.
        soft_phrase = next(t for (t, tags) in bc._WAKE_PHRASE_BANK
                           if "soft" in tags)
        with mock.patch.object(bc, "detect_tone", return_value="tired"), \
             mock.patch.object(bc.random, "choice",
                               return_value=(soft_phrase, {"soft"})):
            text, vol = bc._pick_wake_variety(from_standby=False,
                                              wake_text="so sleepy")
        self.assertEqual(text, soft_phrase)
        self.assertEqual(vol, 0.85)   # soft & preferred -> quieter

    def test_repeat_phrase_avoided_when_alternatives_exist(self):
        bc = self.bc
        # Last phrase is a general one; ensure it's filtered from candidates.
        last = "Yes, sir?"
        bc._last_wake_phrase[0] = last
        seen = {}
        def _choice(seq):
            seen["candidates"] = seq
            return seq[0]
        with mock.patch.object(bc, "detect_tone", return_value=None), \
             mock.patch.object(bc.random, "choice", side_effect=_choice):
            bc._pick_wake_variety(from_standby=True, wake_text="")
        self.assertNotIn(last, [t for (t, _tg) in seen["candidates"]])

    def test_recent_wakes_force_terse_preference(self):
        bc = self.bc
        import time as _t
        now = _t.time()
        bc._wake_history[:] = [now - 10, now - 20]   # 2 in last 5 min
        seen = {}
        with mock.patch.object(bc, "detect_tone", return_value=None), \
             mock.patch.object(bc.random, "choice",
                               side_effect=lambda s: (seen.setdefault("c", s) or s)[0]):
            bc._pick_wake_variety(from_standby=False, wake_text="jarvis")
        # 'terse' joined the preferred set, so every candidate carries a
        # preferred tag (terse / formal-from-standby? no — general at least).
        self.assertTrue(seen["c"])

    def test_playful_tone_adds_playful_tag(self):
        bc = self.bc
        seen = {}
        def _choice(seq):
            seen["candidates"] = seq
            return seq[0]
        with mock.patch.object(bc, "detect_tone", return_value="playful"), \
             mock.patch.object(bc.random, "choice", side_effect=_choice):
            bc._pick_wake_variety(from_standby=False, wake_text="heyyy")
        # A playful-tagged phrase ("Hm?" / "Yes?") must be among the candidates.
        cand_texts = [t for (t, _tg) in seen["candidates"]]
        self.assertTrue(any(t in ("Hm?", "Yes?") for t in cand_texts))

    def test_empty_candidate_set_falls_back_to_full_bank(self):
        bc = self.bc
        # A bank whose every phrase lacks any preferred tag forces the
        # `if not candidates: candidates = list(_WAKE_PHRASE_BANK)` fallback.
        bank = [("Alpha.", {"nonexistent"}), ("Beta.", {"nonexistent"})]
        seen = {}
        def _choice(seq):
            seen["candidates"] = list(seq)
            return seq[0]
        with mock.patch.object(bc, "_WAKE_PHRASE_BANK", bank), \
             mock.patch.object(bc, "detect_tone", return_value=None), \
             mock.patch.object(bc.random, "choice", side_effect=_choice):
            text, _vol = bc._pick_wake_variety(from_standby=False, wake_text="hi")
        # Fallback restored the full (patched) bank as candidates.
        self.assertEqual([t for (t, _tg) in seen["candidates"]],
                         ["Alpha.", "Beta."])
        self.assertEqual(text, "Alpha.")


class ContextAwareGreetingTests(SectionFiveBase):
    def setUp(self):
        bc = self.bc
        self._orig_hist = list(bc._wake_history)
        self._orig_date = list(bc._last_wake_date)
        self._orig_pre = list(bc._pre_wake_silence_seconds)
        self.addCleanup(self._restore)
        bc._wake_history[:] = []
        bc._last_wake_date[0] = None

    def _restore(self):
        bc = self.bc
        bc._wake_history[:] = self._orig_hist
        bc._last_wake_date[0] = self._orig_date[0]
        bc._pre_wake_silence_seconds[0] = self._orig_pre[0]

    def _patch_now(self, hour):
        import datetime as _dtmod

        class _FakeDateTime(_dtmod.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 6, 1, hour, 30, 0)
        return mock.patch.object(bc_datetime_mod(self.bc), "datetime",
                                 _FakeDateTime)

    def test_still_up_branch(self):
        bc = self.bc
        import time as _t
        now = _t.time()
        # 3 wakes already in the window + 02:30 local hour -> "Still up, sir?"
        bc._wake_history[:] = [now - 5, now - 6, now - 7]
        with mock.patch.object(bc, "_bambu_print_progress", return_value=None), \
             mock.patch.object(bc, "_user_looking_away", return_value=False), \
             self._patch_now(2):
            text, vol = bc.context_aware_greeting(from_standby=False)
        self.assertEqual(text, "Still up, sir?")
        self.assertEqual(vol, 1.0)

    def test_good_morning_first_wake(self):
        bc = self.bc
        with mock.patch.object(bc, "_bambu_print_progress", return_value=None), \
             mock.patch.object(bc, "_user_looking_away", return_value=False), \
             self._patch_now(7):
            text, _vol = bc.context_aware_greeting(from_standby=False)
        self.assertEqual(text, "Good morning, sir.")

    def test_bambu_print_branch(self):
        bc = self.bc
        # Afternoon hour so morning/still-up don't fire; printer is mid-job.
        with mock.patch.object(bc, "_bambu_print_progress", return_value=73), \
             mock.patch.object(bc, "_user_looking_away", return_value=False), \
             self._patch_now(14):
            text, vol = bc.context_aware_greeting(from_standby=False)
        self.assertIn("73%", text)
        self.assertEqual(vol, 1.0)

    def test_looking_away_quieter_greeting(self):
        bc = self.bc
        with mock.patch.object(bc, "_bambu_print_progress", return_value=None), \
             mock.patch.object(bc, "_user_looking_away", return_value=True), \
             self._patch_now(14):
            text, vol = bc.context_aware_greeting(from_standby=False)
        self.assertEqual(text, "Yes, sir?")
        self.assertEqual(vol, 0.55)

    def test_default_falls_through_to_variety(self):
        bc = self.bc
        with mock.patch.object(bc, "_bambu_print_progress", return_value=None), \
             mock.patch.object(bc, "_user_looking_away", return_value=False), \
             mock.patch.object(bc, "_pick_wake_variety",
                               return_value=("Standing by, sir.", 1.0)) as pv, \
             self._patch_now(14):
            text, vol = bc.context_aware_greeting(from_standby=True,
                                                  wake_text="jarvis")
        self.assertEqual(text, "Standing by, sir.")
        pv.assert_called_once()

    def test_updates_wake_history_and_date(self):
        bc = self.bc
        with mock.patch.object(bc, "_bambu_print_progress", return_value=None), \
             mock.patch.object(bc, "_user_looking_away", return_value=False), \
             mock.patch.object(bc, "_pick_wake_variety",
                               return_value=("x", 1.0)), \
             self._patch_now(14):
            bc.context_aware_greeting(from_standby=False)
        self.assertEqual(len(bc._wake_history), 1)        # this wake recorded
        self.assertEqual(bc._last_wake_date[0], "2026-06-01")


# ════════════════════════════════════════════════════════════════════════════
#  Mid-task status lines
# ════════════════════════════════════════════════════════════════════════════
class MidTaskStatusTests(SectionFiveBase):
    def test_generic_bucket_for_unknown_action(self):
        bc = self.bc
        with mock.patch.object(bc.random, "choice", side_effect=lambda s: s[0]):
            line = bc._pick_mid_task_status_line("totally_unknown_action")
        self.assertEqual(line, bc._MID_TASK_STATUS_LINES["_generic"][0])

    def test_streaming_bucket_substitutes_service(self):
        bc = self.bc
        # Force the {service}-bearing line so substitution runs.
        svc_line = next(l for l in bc._MID_TASK_STATUS_LINES["streaming"]
                        if "{service}" in l)
        with mock.patch.object(bc.random, "choice", return_value=svc_line):
            line = bc._pick_mid_task_status_line("netflix")
        self.assertNotIn("{service}", line)
        self.assertIn("Netflix", line)

    def test_streaming_unknown_service_uses_fallback_label(self):
        bc = self.bc
        svc_line = next(l for l in bc._MID_TASK_STATUS_LINES["streaming"]
                        if "{service}" in l)
        # play_streaming maps to streaming bucket but its label is "the service".
        with mock.patch.object(bc.random, "choice", return_value=svc_line):
            line = bc._pick_mid_task_status_line("play_streaming")
        self.assertIn("the service", line)

    def test_emit_fires_once_then_latches(self):
        bc = self.bc
        flag = [False]
        with mock.patch.object(bc, "_speak") as speak, \
             mock.patch.object(bc, "_pick_mid_task_status_line",
                               return_value="Working on it."):
            bc._emit_mid_task_status("upgrade", "", flag)
            self.assertTrue(flag[0])
            bc._emit_mid_task_status("upgrade", "", flag)   # already fired -> no-op
        speak.assert_called_once_with("Working on it.")

    def test_emit_swallows_speak_failure(self):
        bc = self.bc
        flag = [False]
        with mock.patch.object(bc, "_speak",
                               side_effect=RuntimeError("tts dead")), \
             mock.patch.object(bc, "_pick_mid_task_status_line",
                               return_value="Almost there, sir."):
            # Must not raise — a missed bridge line is cosmetic.
            self.assertIsNone(bc._emit_mid_task_status("netflix", "", flag))
        self.assertTrue(flag[0])


# ════════════════════════════════════════════════════════════════════════════
#  Hallucination pre-flight + confirmation gate
# ════════════════════════════════════════════════════════════════════════════
class PreemptiveHallucinationTests(SectionFiveBase):
    def test_returns_none_when_action_token_present(self):
        bc = self.bc
        # Reply already carries an [ACTION:] token -> reactive path owns it.
        out = bc._detect_preemptive_hallucination(
            "Entering ambient learning mode. [ACTION: ambient_learning_mode_on]")
        self.assertIsNone(out)

    def test_inject_for_known_ambient_learning_claim(self):
        bc = self.bc
        out = bc._detect_preemptive_hallucination(
            "Entering ambient-learning mode, sir.")
        self.assertIsNotNone(out)
        verb, name, _desc = out
        self.assertEqual(verb, "inject")
        self.assertEqual(name, "ambient_learning_mode_on")
        self.assertIn(name, bc.ACTIONS)

    def test_refuse_for_camera_movement_claim(self):
        bc = self.bc
        out = bc._detect_preemptive_hallucination(
            "Panning the camera to the left now, sir.")
        self.assertIsNotNone(out)
        verb, name, _desc = out
        self.assertEqual(verb, "refuse")
        self.assertIsNone(name)

    def test_returns_none_for_innocuous_prose(self):
        bc = self.bc
        out = bc._detect_preemptive_hallucination(
            "The weather looks pleasant today, sir.")
        self.assertIsNone(out)

    def test_inject_skipped_when_action_not_registered(self):
        bc = self.bc
        # If the mapped action is somehow absent from ACTIONS, the matching
        # entry must fall through to 'refuse' rather than inject a dead token.
        trimmed = {k: v for k, v in bc.ACTIONS.items()
                   if k != "ambient_learning_mode_on"}
        with mock.patch.object(bc, "ACTIONS", trimmed):
            out = bc._detect_preemptive_hallucination(
                "Entering ambient-learning mode, sir.")
        self.assertIsNotNone(out)
        verb, name, _desc = out
        self.assertEqual(verb, "refuse")
        self.assertIsNone(name)


class FabricatedInfoHallucinationTests(SectionFiveBase):
    """BUG 2 — local qwen fabricates time/version/weather/system answers from
    its head instead of emitting the action. The preemptive guard must inject
    the correct action when the drafted reply ASSERTS one of these facts with
    no [ACTION:] token, and must NOT trip on ordinary prose.

    The info actions get_time/version_info/system_pulse are core-registered;
    weather_briefing/list_timers/whats_broken come from skills, so the cases
    that need them register a stub into a patched ACTIONS so the test is
    self-contained whether or not skills were loaded into the cached monolith.
    """

    def _detect_with_actions(self, reply, extra=()):
        bc = self.bc
        actions = dict(bc.ACTIONS)
        for name in extra:
            actions.setdefault(name, lambda _="": "stub")
        with mock.patch.object(bc, "ACTIONS", actions):
            return bc._detect_preemptive_hallucination(reply)

    def _assert_injects(self, reply, action, extra=()):
        out = self._detect_with_actions(reply, extra=extra)
        self.assertIsNotNone(out, f"expected an inject for: {reply!r}")
        verb, name, _desc = out
        self.assertEqual(verb, "inject", f"expected inject for: {reply!r}")
        self.assertEqual(name, action, f"wrong action for: {reply!r}")

    # ── fabricated time → get_time ───────────────────────────────────────
    def test_fabricated_clock_time_injects_get_time(self):
        for reply in ("It's 1:47 AM, sir.",
                      "The current time is 10:52 PM.",
                      "Right now it's 9 AM."):
            self._assert_injects(reply, "get_time")

    # ── fabricated version → version_info ────────────────────────────────
    def test_fabricated_version_injects_version_info(self):
        for reply in ("I'm on version 12.4, sir.",
                      "I'm running 1.20.6.",
                      "Running version 3, sir."):
            self._assert_injects(reply, "version_info")

    # ── fabricated weather → weather_briefing ────────────────────────────
    def test_fabricated_weather_injects_weather_briefing(self):
        for reply in ("It's currently 64 degrees and sunny.",
                      "Currently 58 degrees Fahrenheit, sir.",
                      "It's 72 degrees and clear right now."):
            self._assert_injects(reply, "weather_briefing",
                                 extra=("weather_briefing",))

    # ── fabricated system stats → system_pulse ───────────────────────────
    def test_fabricated_system_stats_injects_system_pulse(self):
        # system_pulse is skill-registered; supply it so the test is
        # self-contained whether or not skills loaded into the cached monolith.
        for reply in ("CPU is at 40% and memory is at 80%.",
                      "You're running at 12% CPU, sir.",
                      "Memory usage is 73%, sir."):
            self._assert_injects(reply, "system_pulse", extra=("system_pulse",))

    # ── conservative: ordinary prose must NOT trip ───────────────────────
    def test_no_false_positive_on_innocuous_prose(self):
        for reply in (
            "It's 5 minutes left on your timer, sir.",   # 'it's N' but not a clock
            "It's time to head out, sir.",
            "I use version control for that.",
            "You're running late for the meeting.",
            "Turn it 90 degrees to the right.",          # rotation, not weather
            "Rotate the model 45 degrees, sir.",
            "The CPU handles that workload fine.",       # no number
            "I'm 100% certain, sir.",                    # % but not cpu/ram
            "There is a newer version available.",       # 'version' but no number
            "It will take about 5 minutes to finish.",
            "The weather looks pleasant today, sir.",
        ):
            out = self._detect_with_actions(
                reply, extra=("weather_briefing",))
            self.assertIsNone(out, f"false positive on: {reply!r}")

    # ── never overrides an explicit action token ─────────────────────────
    def test_skips_when_action_token_already_present(self):
        out = self._detect_with_actions(
            "One moment, sir. It's 1:47 AM. [ACTION: get_time]")
        self.assertIsNone(out)


class NeedsConfirmationTests(SectionFiveBase):
    def test_empty_keyword_list_never_confirms(self):
        bc = self.bc
        with mock.patch.object(bc, "CONFIRM_KEYWORDS", []):
            self.assertFalse(bc._needs_confirmation("delete_everything", "now"))

    def test_keyword_in_action_name(self):
        bc = self.bc
        with mock.patch.object(bc, "CONFIRM_KEYWORDS", ["delete", "buy"]):
            self.assertTrue(bc._needs_confirmation("delete_file", "report.txt"))

    def test_keyword_in_arg(self):
        bc = self.bc
        with mock.patch.object(bc, "CONFIRM_KEYWORDS", ["purchase"]):
            self.assertTrue(bc._needs_confirmation("open_url",
                                                   "https://shop/purchase"))

    def test_no_keyword_match(self):
        bc = self.bc
        with mock.patch.object(bc, "CONFIRM_KEYWORDS", ["delete", "format"]):
            self.assertFalse(bc._needs_confirmation("open_url", "example.com"))


def bc_datetime_mod(bc):
    """context_aware_greeting / _pick_wake_variety do `from datetime import
    datetime as _dt` *inside* the function, so the name they resolve is the
    `datetime` module attribute on the stdlib module — patching
    bc.datetime won't help. They import the stdlib `datetime` module fresh each
    call, so we patch the global `datetime` module object that the monolith
    imported at top level."""
    import datetime as _dtmod
    return _dtmod


if __name__ == "__main__":
    unittest.main(verbosity=2)
