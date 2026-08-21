"""tools/browser_guard — the block on REAL browser launches during a test run.

THE INCIDENT (2026-08-20, 11:38-11:41)
======================================
Two full-CI runs pushed 12 then 20 new Chrome renderers per minute into the
owner's DEFAULT profile and spammed dozens of live tabs into his window. No
selenium, no chromedriver: the tests reached the browser INDIRECTLY, through
production code that calls ``webbrowser.open()`` (core/actions.py:73, the Apple
Music launch path in the monolith, skills/youtube_search.py, tray.py's
``os.startfile`` menu items). Per-call-site patches cannot fix that class of
leak; one process-wide interception can.

These tests pin that interception WITHOUT ever launching anything:

* The behaviour tests drive ``_install_into`` with FAKE module objects — the
  documented mock seam. That keeps them deterministic AND means this file can
  never accidentally un-guard the very suite it is running inside.
* A handful of tests do touch the real ``webbrowser`` / ``subprocess`` modules,
  because the load-bearing claim ("a blocked launch really does not launch") is
  only worth anything on the real ones. They are still launch-free: the guard
  returns a stand-in instead of spawning.
* The last class is the STALE-DUPLICATE guard, this repo's signature bug class.
  It reads the SOURCE of tests/__init__.py and the three tools/ runners and
  fails if any of them stops installing the guard — a future refactor must not
  be able to silently drop the wiring.

Every test that records an attempt saves and restores the process-wide ledger,
so running this file never erases the offender list the atexit summary prints
for the rest of the run.
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import subprocess
import sys
import types
import unittest
import webbrowser
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import browser_guard  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.join(_PROJECT_ROOT, "tools")
_TESTS_INIT = os.path.join(_PROJECT_ROOT, "tests", "__init__.py")

# Every tools/ module that discovers the unittest suite must install the guard,
# exactly as it must apply the memory ceiling (see tests/test_mem_guard.py).
_GUARDED_RUNNERS = ("run_tests.py", "run_tests_ci_sim.py", "run_coverage.py")

# Discovery runners that deliberately do NOT install it, with a reason.
# DELIBERATELY EMPTY — the hatch exists so a FUTURE runner cannot be added
# silently, not to bless a gap.
_KNOWN_UNGUARDED: set[str] = set()

# Calls that import test modules or spawn a child. The guard must precede them.
_TEST_TOUCHING_CALLS = {"discover", "loadTestsFromName", "loadTestsFromModule",
                        "_run_ci_gates", "_run_suite", "run", "Popen"}

_DISCOVERY_MARKERS = ("unittest.TestLoader", "unittest.main(")

_CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
_URL = "https://example.com/guard-probe"


def _source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _fake_modules():
    """Stand-ins for webbrowser / subprocess / os that RECORD what the guard
    would otherwise have let through. ``calls`` collects every delegation, so a
    test can prove the real function was (or was not) reached."""
    calls: list[tuple] = []

    wb = types.ModuleType("fake_webbrowser")

    def _wb_open(url, new=0, autoraise=True):
        calls.append(("open", url, new, autoraise))
        return True

    def _wb_open_new(url):
        calls.append(("open_new", url))
        return True

    def _wb_open_new_tab(url):
        calls.append(("open_new_tab", url))
        return True

    def _wb_get(using=None):
        calls.append(("get", using))
        return "REAL-CONTROLLER"

    def _wb_register(name, klass, instance=None, *, preferred=False):
        calls.append(("register", name, preferred))

    wb.open = _wb_open
    wb.open_new = _wb_open_new
    wb.open_new_tab = _wb_open_new_tab
    wb.get = _wb_get
    wb.register = _wb_register

    sp = types.ModuleType("fake_subprocess")
    sp.CompletedProcess = subprocess.CompletedProcess

    def _sp_popen(*args, **kwargs):
        calls.append(("Popen", args, kwargs))
        return "REAL-POPEN"

    def _sp_run(*args, **kwargs):
        calls.append(("run", args, kwargs))
        return "REAL-RUN"

    def _sp_call(*args, **kwargs):
        calls.append(("call", args, kwargs))
        return 99

    sp.Popen = _sp_popen
    sp.run = _sp_run
    sp.call = _sp_call

    osmod = types.ModuleType("fake_os")

    def _startfile(path, *a, **k):
        calls.append(("startfile", path, a, k))

    osmod.startfile = _startfile
    return wb, sp, osmod, calls


class _GuardTestCase(unittest.TestCase):
    """Base class that protects the PROCESS-WIDE ledger.

    ``browser_guard.reset()`` is global. Clearing it naively would wipe the
    attempts recorded by every earlier test in the run and hide them from the
    atexit summary — the exact opposite of what this module is for. So: snapshot
    on entry, restore on exit."""

    def setUp(self):
        self._saved_ledger = browser_guard.blocked_attempts()
        browser_guard.reset()

    def tearDown(self):
        browser_guard.reset()
        browser_guard._blocked.extend(self._saved_ledger)

    def _install_fakes(self):
        wb, sp, osmod, calls = _fake_modules()
        undos = browser_guard._install_into(wb, sp, osmod)
        for undo in undos:
            self.addCleanup(undo)
        return wb, sp, osmod, calls

    def _only(self):
        attempts = browser_guard.blocked_attempts()
        self.assertEqual(len(attempts), 1, f"expected 1 record, got {attempts}")
        return attempts[0]


@contextlib.contextmanager
def _unlatched():
    """Drop the process-wide latch so install() itself can be exercised, then
    RE-ARM. The finally block is not optional: leaving the latch dropped would
    let every later test in the run open real browser windows."""
    saved = browser_guard.blocked_attempts()
    try:
        browser_guard._reset_for_tests()
        yield
    finally:
        browser_guard._reset_for_tests()
        browser_guard.install(quiet=True)
        browser_guard._blocked.extend(saved)


class WebbrowserInterceptionTests(_GuardTestCase):
    """Every webbrowser entry point that can reach a real window is neutered."""

    def test_open_is_neutered_and_recorded(self):
        wb, _, _, calls = self._install_fakes()
        self.assertTrue(wb.open(_URL))          # mimics a successful launch
        self.assertEqual(calls, [])             # ...but nothing was launched
        att = self._only()
        self.assertEqual(att.api, "webbrowser.open")
        self.assertEqual(att.target, _URL)

    def test_open_new_and_open_new_tab_are_neutered(self):
        wb, _, _, calls = self._install_fakes()
        self.assertTrue(wb.open_new(_URL))
        self.assertTrue(wb.open_new_tab(_URL))
        self.assertEqual(calls, [])
        self.assertEqual([a.api for a in browser_guard.blocked_attempts()],
                         ["webbrowser.open_new", "webbrowser.open_new_tab"])

    def test_get_returns_a_controller_that_cannot_launch(self):
        """The monolith's preferred path is webbrowser.get("chrome").open(url) —
        stubbing only ``open`` would leave that wide open."""
        wb, _, _, calls = self._install_fakes()
        browser = wb.get("chrome")
        self.assertIsInstance(browser, browser_guard._BlockedBrowser)
        self.assertEqual(calls, [])             # the real get() never ran
        self.assertEqual(browser_guard.blocked_attempts(), ())  # get isn't a launch
        self.assertTrue(browser.open(_URL))
        self.assertTrue(browser.open_new(_URL))
        self.assertTrue(browser.open_new_tab(_URL))
        self.assertEqual([a.api for a in browser_guard.blocked_attempts()],
                         ["webbrowser.get().open", "webbrowser.get().open_new",
                          "webbrowser.get().open_new_tab"])

    def test_register_is_a_recorded_no_op(self):
        """register() exists to make a launcher REACHABLE. Recording it without
        registering keeps that door shut."""
        wb, _, _, calls = self._install_fakes()
        wb.register("evil", None, object(), preferred=True)
        self.assertEqual(calls, [])
        self.assertEqual(self._only().api, "webbrowser.register")


class StartfileInterceptionTests(_GuardTestCase):
    """os.startfile is the Windows shell-launch primitive — of an https:// URL
    it opens the default browser, of a folder it throws an Explorer window onto
    the owner's desktop. Both are unwanted mid-suite."""

    def test_startfile_is_neutered_and_recorded(self):
        _, _, osmod, calls = self._install_fakes()
        self.assertIsNone(osmod.startfile(r"C:\JARVIS\logs"))
        self.assertEqual(calls, [])
        att = self._only()
        self.assertEqual(att.api, "os.startfile")
        self.assertIn("logs", att.target)

    def test_missing_startfile_is_skipped_not_created(self):
        """POSIX has no os.startfile. The guard must neither raise nor CREATE
        the attribute — inventing it would make every
        ``mock.patch.object(mod.os, "startfile", create=True)`` test on CI start
        asserting against a Windows-only API that isn't really there."""
        wb, sp, _, calls = _fake_modules()
        posix_os = types.ModuleType("posix_os")          # no startfile, like POSIX
        for undo in browser_guard._install_into(wb, sp, posix_os):
            self.addCleanup(undo)
        self.assertFalse(hasattr(posix_os, "startfile"),
                         "the guard invented os.startfile on a POSIX-shaped os")
        self.assertTrue(wb.open(_URL))       # the rest still installed fine
        self.assertEqual(calls, [])
        self.assertEqual(self._only().api, "webbrowser.open")

    def test_install_into_bare_modules_never_raises(self):
        """A module with none of the target attributes must be a silent no-op,
        not an exception that takes the whole run down."""
        empty = types.ModuleType("empty")
        self.assertEqual(browser_guard._install_into(empty, empty, empty), [])
        self.assertEqual(browser_guard._install_into(None, None, None), [])


class SubprocessInterceptionTests(_GuardTestCase):
    """Refuse a browser launch; leave every other command completely alone."""

    def test_browser_popen_is_refused(self):
        _, sp, _, calls = self._install_fakes()
        proc = sp.Popen([_CHROME_EXE, "--new-window", _URL], close_fds=True)
        self.assertIsInstance(proc, browser_guard._BlockedPopen)
        self.assertEqual(calls, [])
        self.assertEqual(self._only().api, "subprocess.Popen")
        # Shaped like a process that started and exited cleanly, because that is
        # what callers saw before the guard existed.
        self.assertEqual(proc.poll(), 0)
        self.assertEqual(proc.wait(), 0)
        self.assertEqual(proc.communicate(), (None, None))
        self.assertIsNone(proc.kill())
        with proc as ctx:                      # `with subprocess.Popen(...)`
            self.assertIs(ctx, proc)

    def test_browser_run_and_call_are_refused(self):
        _, sp, _, calls = self._install_fakes()
        done = sp.run([_CHROME_EXE, _URL])
        self.assertIsInstance(done, subprocess.CompletedProcess)
        self.assertEqual(done.returncode, 0)
        self.assertIsNone(done.stdout)
        self.assertEqual(sp.call(["firefox", _URL]), 0)
        self.assertEqual(calls, [])
        self.assertEqual([a.api for a in browser_guard.blocked_attempts()],
                         ["subprocess.run", "subprocess.call"])

    def test_refused_run_matches_the_requested_capture_flavour(self):
        _, sp, _, _ = self._install_fakes()
        self.assertEqual(sp.run([_CHROME_EXE, _URL], capture_output=True,
                                text=True).stdout, "")
        self.assertEqual(sp.run([_CHROME_EXE, _URL],
                                capture_output=True).stdout, b"")

    def test_non_browser_subprocess_passes_through_untouched(self):
        """The surgical half: ~200 legitimate subprocess call sites in this repo
        must not be able to notice the wrapper — same args, same kwargs, same
        return value, and NOTHING recorded."""
        _, sp, _, calls = self._install_fakes()
        self.assertEqual(sp.Popen(["git", "status"], cwd="/x", shell=False),
                         "REAL-POPEN")
        self.assertEqual(sp.run(["ffmpeg", "-i", "a.wav"], timeout=5), "REAL-RUN")
        self.assertEqual(sp.call("dir C:\\JARVIS", shell=True), 99)
        self.assertEqual(browser_guard.blocked_attempts(), ())
        self.assertEqual(calls, [
            ("Popen", (["git", "status"],), {"cwd": "/x", "shell": False}),
            ("run", (["ffmpeg", "-i", "a.wav"],), {"timeout": 5}),
            ("call", ("dir C:\\JARVIS",), {"shell": True}),
        ])

    def test_process_management_tools_naming_a_browser_pass_through(self):
        """READ-ONLY process tools count or filter; they cannot open a window
        and cannot close one, so naming a browser is harmless.

        The KILL verbs used to be exempt here too, on a justification that does
        not exist in the tree — see ``KillVerbTests``, which now owns them."""
        _, sp, _, calls = self._install_fakes()
        for cmd in (["tasklist", "/FI", "IMAGENAME eq msedge.exe"],
                    ["pgrep", "-f", "firefox"],
                    "wmic process where name='chrome.exe' get ProcessId",
                    ["reg", "query", r"HKCU\Software\Google\Chrome"]):
            with self.subTest(cmd=cmd):
                sp.Popen(cmd)
        self.assertEqual(browser_guard.blocked_attempts(), ())
        self.assertEqual(len(calls), 4)

    def test_a_browser_named_inside_a_data_argument_is_not_a_launch(self):
        """Only a token whose BASENAME is a browser binary counts, so a string
        that merely contains the word doesn't trip the guard."""
        _, sp, _, calls = self._install_fakes()
        sp.Popen([sys.executable, "-c", "print('chrome')"])
        sp.Popen(["curl", "https://chrome.google.com/webstore"])
        sp.Popen(["mpv", "opera_aria.mp3"])
        self.assertEqual(browser_guard.blocked_attempts(), ())
        self.assertEqual(len(calls), 3)

    def test_quoted_shell_command_line_is_tokenised(self):
        """shell=True hands over ONE string; the quote-aware split has to find
        the browser inside `"C:\\Program Files\\...\\chrome.exe" --new-window`."""
        _, sp, _, calls = self._install_fakes()
        sp.Popen(f'"{_CHROME_EXE}" --new-window {_URL}', shell=True)
        sp.Popen(f'cmd /c start chrome "{_URL}"', shell=True)
        self.assertEqual(calls, [])
        self.assertEqual(len(browser_guard.blocked_attempts()), 2)

    def test_posix_browser_names_are_covered(self):
        """The same guard has to hold on the ubuntu-latest CI runner."""
        _, sp, _, calls = self._install_fakes()
        for cmd in (["google-chrome-stable", _URL], ["/usr/bin/firefox", _URL],
                    ["chromium-browser", _URL], ["/opt/brave.com/brave/brave", _URL],
                    ["microsoft-edge", _URL]):
            with self.subTest(cmd=cmd[0]):
                sp.Popen(cmd)
        self.assertEqual(calls, [])
        self.assertEqual(len(browser_guard.blocked_attempts()), 5)

    def test_command_passed_as_the_args_keyword_is_still_seen(self):
        _, sp, _, calls = self._install_fakes()
        sp.Popen(args=[_CHROME_EXE, _URL])
        self.assertEqual(calls, [])
        self.assertEqual(len(browser_guard.blocked_attempts()), 1)


class MockPatchCompositionTests(_GuardTestCase):
    """LOAD-BEARING. Dozens of existing tests patch these exact names. The guard
    replaces a module attribute; mock.patch replaces it a second time on top and
    restores it to OUR stub — that composition has to work in both directions or
    the guard breaks the suite it is protecting."""

    def test_patch_object_wins_while_active_and_restores_to_the_stub(self):
        wb, _, _, calls = self._install_fakes()
        with mock.patch.object(wb, "open", return_value="MOCK") as m:
            self.assertEqual(wb.open(_URL), "MOCK")
            m.assert_called_once_with(_URL)
        # The test's own mock saw the call; the guard did not double-record it.
        self.assertEqual(browser_guard.blocked_attempts(), ())
        # And the guard is back in place, not the real function.
        self.assertTrue(getattr(wb.open, "_jarvis_browser_guard", False))
        self.assertTrue(wb.open(_URL))
        self.assertEqual(calls, [])
        self.assertEqual(len(browser_guard.blocked_attempts()), 1)

    def test_patch_by_string_target_on_the_real_module_composes(self):
        """`mock.patch("webbrowser.open", ...)` — the form tests/test_bug_reporter.py
        uses. It resolves the attribute at enter and writes it back at exit, so
        the guard has to survive the round trip on the LIVE module."""
        browser_guard.install(quiet=True)   # idempotent; no-op under the suite
        before = webbrowser.open
        self.assertTrue(getattr(before, "_jarvis_browser_guard", False),
                        "the live webbrowser.open is not the guard's stub")
        with mock.patch("webbrowser.open", return_value=True) as m:
            self.assertIsNot(webbrowser.open, before)
            self.assertTrue(webbrowser.open(_URL))
            m.assert_called_once_with(_URL)
        self.assertIs(webbrowser.open, before)
        self.assertEqual(browser_guard.blocked_attempts(), ())

    def test_patch_object_with_create_true_on_os_startfile_composes(self):
        """tests/test_tray.py and tests/test_upgrade_jarvis.py patch
        ``os.startfile`` with create=True (for POSIX CI). That must still land
        on, and restore, whatever the guard left there."""
        browser_guard.install(quiet=True)
        before = getattr(os, "startfile", None)
        with mock.patch.object(os, "startfile", create=True) as sf:
            os.startfile(r"C:\x\y.png")
            sf.assert_called_once_with(r"C:\x\y.png")
        self.assertIs(getattr(os, "startfile", None), before)
        self.assertEqual(browser_guard.blocked_attempts(), ())


class LiveModuleTests(_GuardTestCase):
    """The claims that are only worth anything on the REAL modules. Still
    launch-free — the guard hands back a stand-in instead of spawning."""

    def test_the_live_webbrowser_module_is_guarded(self):
        browser_guard.install(quiet=True)
        self.assertTrue(browser_guard.is_installed())
        for name in ("open", "open_new", "open_new_tab", "get", "register"):
            with self.subTest(attr=name):
                self.assertTrue(
                    getattr(getattr(webbrowser, name), "_jarvis_browser_guard", False),
                    f"webbrowser.{name} is NOT guarded")
        self.assertTrue(webbrowser.open(_URL))
        self.assertEqual(self._only().api, "webbrowser.open")

    def test_the_live_subprocess_module_refuses_a_browser(self):
        browser_guard.install(quiet=True)
        proc = subprocess.Popen([_CHROME_EXE, "--new-window", _URL])
        self.assertIsInstance(proc, browser_guard._BlockedPopen)
        self.assertEqual(self._only().api, "subprocess.Popen")

    def test_check_output_and_check_call_are_covered_transitively(self):
        """They are not wrapped: they resolve ``run`` / ``call`` as subprocess
        module globals at call time, so they land on the wrappers for free.
        Pinning it here means a future 'let's wrap them too' cannot double-block
        and a future stdlib change cannot silently open a hole."""
        browser_guard.install(quiet=True)
        self.assertEqual(subprocess.check_output([_CHROME_EXE, _URL]), b"")
        self.assertEqual(subprocess.check_call([_CHROME_EXE, _URL]), 0)
        self.assertEqual([a.api for a in browser_guard.blocked_attempts()],
                         ["subprocess.run", "subprocess.call"])

    def test_popen_stays_a_subclassable_class(self):
        """REGRESSION, MEASURED 2026-08-20: the first cut replaced
        subprocess.Popen with a FUNCTION, and ``import unittest.mock`` promptly
        died — mock imports asyncio, and asyncio/windows_utils.py line 132 is
        ``class Popen(subprocess.Popen)``. A function base class is a TypeError
        that takes every test module with it. It must stay a class, it must stay
        subclassable, and isinstance() must keep working."""
        browser_guard.install(quiet=True)
        self.assertIsInstance(subprocess.Popen, type)

        class Derived(subprocess.Popen):     # what the stdlib does
            pass

        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        try:
            self.assertIsInstance(proc, subprocess.Popen)
        finally:
            proc.wait(timeout=120)
        self.assertTrue(issubclass(Derived, subprocess.Popen))
        self.assertEqual(browser_guard.blocked_attempts(), ())

    def test_a_real_non_browser_subprocess_still_runs(self):
        """End-to-end pass-through on the live module: a genuine child process
        must still be spawned, with its output intact."""
        browser_guard.install(quiet=True)
        done = subprocess.run([sys.executable, "-c", "print('guard-ok')"],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0)
        self.assertEqual(done.stdout.strip(), "guard-ok")
        self.assertEqual(browser_guard.blocked_attempts(), ())


class RecordingAndSummaryTests(_GuardTestCase):
    """The atexit summary is how the NEXT agent finds the offenders, so it has
    to name the test method, not just a file."""

    def test_the_record_names_this_exact_test_method(self):
        wb, _, _, _ = self._install_fakes()
        wb.open(_URL)
        att = self._only()
        cls = type(self)
        self.assertEqual(att.test_id,
                         f"{cls.__module__}.{cls.__qualname__}.{self._testMethodName}")
        self.assertIn("test_browser_guard.py:", att.call_site)
        self.assertIn("test_the_record_names_this_exact_test_method", att.call_site)

    def test_the_call_site_is_the_production_frame_not_the_test(self):
        """The leak is production code called BY a test. The call site must
        point at the production line so it can be fixed."""
        wb, _, _, _ = self._install_fakes()

        def _act_open_url(url):          # stands in for core/actions.py:73
            wb.open(url)

        _act_open_url(_URL)
        att = self._only()
        self.assertIn("_act_open_url()", att.call_site)
        self.assertIn(self._testMethodName, att.test_id)

    def test_summary_counts_launches_and_distinct_tests(self):
        wb, _, _, _ = self._install_fakes()
        wb.open(_URL)
        wb.open(_URL)
        text = browser_guard.summary()
        self.assertIn("blocked 2 real browser launch(es) from 1 test(s)", text)
        self.assertIn("webbrowser.open()", text)
        self.assertIn(self._testMethodName, text)
        self.assertTrue(all(line.startswith("[browser-guard]")
                            for line in text.splitlines()))

    def test_summary_is_empty_when_nothing_was_blocked(self):
        """A clean run must stay silent — a "blocked 0" line every time would
        train everyone to ignore the one that matters."""
        self.assertEqual(browser_guard.summary(), "")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            browser_guard._print_summary_at_exit()
        self.assertEqual(buf.getvalue(), "")

    def test_atexit_summary_prints_when_something_was_blocked(self):
        wb, _, _, _ = self._install_fakes()
        wb.open(_URL)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            browser_guard._print_summary_at_exit()
        self.assertIn("blocked 1 real browser launch(es)", buf.getvalue())

    def test_reset_clears_and_blocked_attempts_is_a_snapshot(self):
        wb, _, _, _ = self._install_fakes()
        wb.open(_URL)
        snapshot = browser_guard.blocked_attempts()
        self.assertIsInstance(snapshot, tuple)
        browser_guard.reset()
        self.assertEqual(browser_guard.blocked_attempts(), ())
        self.assertEqual(len(snapshot), 1)      # the caller's copy is unaffected

    def test_recording_survives_an_unprintable_target(self):
        class Boom:
            def __str__(self):
                raise RuntimeError("nope")

        wb, _, _, _ = self._install_fakes()
        self.assertTrue(wb.open(Boom()))        # must not raise
        self.assertEqual(self._only().target, "<unprintable>")


class InstallContractTests(_GuardTestCase):
    """One banner, idempotent, and an escape hatch that is loud about itself.
    Each of these drops the process-wide latch and RE-ARMS in a finally."""

    def test_install_prints_exactly_one_banner_and_latches(self):
        with _unlatched():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertTrue(browser_guard.install())
            lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 1, buf.getvalue())
            self.assertTrue(lines[0].startswith("[browser-guard] armed"))
            again = io.StringIO()
            with contextlib.redirect_stdout(again):
                self.assertTrue(browser_guard.install())
            self.assertEqual(again.getvalue(), "",
                             "the second install() re-printed / re-wrapped")

    def test_env_escape_hatch_disables_the_guard_loudly(self):
        with _unlatched():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                armed = browser_guard.install(
                    env={browser_guard.ALLOW_ENV_VAR: "1"})
            self.assertFalse(armed)
            self.assertFalse(browser_guard.is_installed())
            self.assertIn("DISABLED", buf.getvalue())
            self.assertIn(browser_guard.ALLOW_ENV_VAR, buf.getvalue())
            # ...and it really did not touch the live module.
            self.assertFalse(getattr(webbrowser.open, "_jarvis_browser_guard",
                                     False))

    def test_escape_hatch_word_parsing(self):
        for raw in ("1", "true", "TRUE", "yes", "on", "enabled", " 1 "):
            with self.subTest(raw=raw):
                self.assertTrue(
                    browser_guard._allowed({browser_guard.ALLOW_ENV_VAR: raw}))
        for raw in ("", "0", "no", "off", "false", "  ", "maybe"):
            with self.subTest(raw=raw):
                self.assertFalse(
                    browser_guard._allowed({browser_guard.ALLOW_ENV_VAR: raw}))
        self.assertFalse(browser_guard._allowed({}))

    def test_record_only_install_announces_that_launches_get_through(self):
        """The middle setting, for a human confirming a suspected offender. It
        must report itself as NOT protecting the run."""
        with _unlatched():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertFalse(browser_guard.install(record_only=True))
            self.assertIn("RECORD-ONLY", buf.getvalue())
            self.assertFalse(browser_guard.is_installed())

    def test_record_only_lets_the_launch_through_and_still_records(self):
        """Exercised on the FAKE modules with the flag flipped directly, so the
        live interpreter is never left in record-only mode for even an instant."""
        wb, sp, osmod, calls = self._install_fakes()
        with mock.patch.object(browser_guard, "_record_only", True):
            self.assertTrue(wb.open(_URL))
            sp.Popen([_CHROME_EXE, _URL])
            osmod.startfile(r"C:\x\y.png")
        self.assertEqual([c[0] for c in calls], ["open", "Popen", "startfile"])
        self.assertEqual([a.api for a in browser_guard.blocked_attempts()],
                         ["webbrowser.open", "subprocess.Popen", "os.startfile"])

    def test_no_lazy_module_is_imported_at_module_scope(self):
        """webbrowser / subprocess / atexit are imported INSIDE install(), so
        the guard binds whatever is in sys.modules at arming time — which is what
        lets tools/audit_codebase.py swap sys.modules["webbrowser"] in and out
        without the guard latching onto a stale module object."""
        tree = ast.parse(_source(os.path.join(_TOOLS_DIR, "browser_guard.py")))
        banned = {"webbrowser", "subprocess", "atexit"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name, banned,
                                     f"{alias.name} imported at module scope")
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module or "", banned,
                                 f"from {node.module} at module scope")


class ArmedStateTests(_GuardTestCase):
    """"Latched" is not "armed".

    On 2026-08-20 the SIBLING guard (tests/live_data_guard.py) parked its
    ``subprocess.Popen`` interception on whatever the module attribute named at
    install time. This module rebinds that attribute — and
    ``_unlatched()`` above un-wraps and re-wraps it mid-run — so the live-data
    guard spent ~12,000 tests reporting itself installed while a spawn of a real
    ``bobert_companion.py`` went straight through. Same hole shape here, so the
    same question has to be answerable: not "did install() run?" but "is the
    interception still THERE?"."""

    def _guard_on(self):
        if not browser_guard.is_installed():
            self.skipTest("guard disabled via " + browser_guard.ALLOW_ENV_VAR)

    def test_nothing_has_displaced_a_stub_in_this_run(self):
        self._guard_on()
        self.assertEqual(browser_guard.unarmed_targets(), [],
                         "these browser-guard stubs have been replaced by "
                         "something else in this run — real browser launches "
                         "are getting through")
        self.assertTrue(browser_guard.is_armed())

    def test_is_armed_is_stricter_than_is_installed(self):
        """The distinction has to be real, not two names for one flag."""
        self._guard_on()
        saved = webbrowser.open
        depth = len(browser_guard._restore)
        try:
            webbrowser.open = lambda *a, **k: True
            self.assertTrue(browser_guard.is_installed())
            self.assertFalse(browser_guard.is_armed())
            self.assertIn("webbrowser.open", browser_guard.unarmed_targets())
        finally:
            del browser_guard._restore[depth:]
            webbrowser.open = saved

    def test_install_repairs_a_displaced_stub(self):
        """install() latches, but the latch must not turn a later call into a
        no-op while a stub is missing — that is how a guard stays silently
        off."""
        self._guard_on()
        saved = webbrowser.open
        depth = len(browser_guard._restore)
        try:
            webbrowser.open = lambda *a, **k: True
            browser_guard.install(quiet=True)
            self.assertNotIn("webbrowser.open", browser_guard.unarmed_targets())
            self.assertTrue(getattr(webbrowser.open, "_jarvis_browser_guard",
                                    False))
        finally:
            del browser_guard._restore[depth:]
            webbrowser.open = saved

    def test_repair_never_double_wraps_an_intact_stub(self):
        """_install_into skips a target that still carries the marker, so the
        runners calling install() next to tests/__init__.py cannot stack two
        wrappers on one attribute."""
        self._guard_on()
        before = webbrowser.open
        browser_guard.install(quiet=True)
        browser_guard.install(quiet=True)
        self.assertIs(webbrowser.open, before)



class ShellMediatedLaunchTests(_GuardTestCase):
    """A browser hidden INSIDE one argv element.

    ``_command_tokens`` returned a list's elements verbatim, so the quote-aware
    split only ever ran on a bare string. Every real-world shell-mediated launch
    — ``powershell -Command "Start-Process chrome ..."``, ``cmd /c start chrome
    <url>``, ``bash -c "google-chrome <url>"`` — puts the browser name inside
    ONE element, where the guard could not see it. The docstring claimed
    ``cmd /c start chrome <url>`` was caught; it was, but only in the fully
    tokenised list form nobody writes."""

    def test_powershell_command_string_is_split(self):
        _, sp, _, calls = self._install_fakes()
        sp.Popen(["powershell", "-Command",
                  f"Start-Process chrome '{_URL}'"])
        self.assertEqual(calls, [])
        self.assertEqual(len(browser_guard.blocked_attempts()), 1)

    def test_cmd_slash_c_start_chrome_is_split(self):
        _, sp, _, calls = self._install_fakes()
        sp.Popen(["cmd", "/c", f"start chrome {_URL}"])
        self.assertEqual(calls, [])
        self.assertEqual(len(browser_guard.blocked_attempts()), 1)

    def test_bash_dash_c_command_line_is_split(self):
        _, sp, _, calls = self._install_fakes()
        sp.Popen(["bash", "-c", f"google-chrome {_URL}"])
        self.assertEqual(calls, [])
        self.assertEqual(len(browser_guard.blocked_attempts()), 1)

    def test_url_shaped_shell_launches_are_blocked(self):
        """``start <url>`` / ``explorer <url>`` / ``rundll32 url.dll,...`` hand
        the URL to the DEFAULT browser without ever naming one, so the binary
        scan cannot see them. They were listed as 'deliberately not
        intercepted' — but a URL argument makes the intent unambiguous."""
        _, sp, _, calls = self._install_fakes()
        for cmd in (["cmd", "/c", "start", _URL],
                    ["explorer.exe", _URL],
                    ["rundll32", "url.dll,FileProtocolHandler", _URL],
                    ["xdg-open", _URL]):
            with self.subTest(cmd=cmd):
                sp.Popen(cmd)
        self.assertEqual(calls, [])
        self.assertEqual(len(browser_guard.blocked_attempts()), 4)

    def test_a_url_without_a_launcher_still_passes(self):
        """The anti-false-positive half: a URL is only a launch when something
        that OPENS it is on the command line."""
        _, sp, _, calls = self._install_fakes()
        sp.Popen(["curl", "-fsSL", _URL])
        sp.Popen(["git", "clone", "https://github.com/x/y"])
        self.assertEqual(browser_guard.blocked_attempts(), ())
        self.assertEqual(len(calls), 2)

    def test_a_folder_shell_launch_still_passes(self):
        """``explorer <folder>`` is not a browser launch. (os.startfile of a
        folder is separately blocked; this is about the subprocess route.)"""
        _, sp, _, calls = self._install_fakes()
        sp.Popen(["explorer.exe", r"C:\JARVIS\logs"])
        sp.Popen(["cmd", "/c", "start", r"C:\JARVIS"])
        self.assertEqual(browser_guard.blocked_attempts(), ())
        self.assertEqual(len(calls), 2)

    def test_prose_arguments_are_not_split(self):
        """Only an element that FOLLOWS a shell command-line switch is
        re-tokenised. tools/multi_agent_pipeline.py hands the claude CLI
        multi-line prompts; shredding those into words would block any prompt
        that happens to say 'chrome'."""
        _, sp, _, calls = self._install_fakes()
        sp.Popen(["claude", "--print", "--",
                  "Investigate why chrome opens during the test run."])
        self.assertEqual(browser_guard.blocked_attempts(), ())
        self.assertEqual(len(calls), 1)


class KillVerbTests(_GuardTestCase):
    """``_PROCESS_TOOLS`` exempted the ENTIRE command line whenever argv[0] was
    one of eighteen process tools — including the kill verbs — with the stated
    justification that ``taskkill /IM chrome.exe`` is "how the monolith reuses a
    single media window".

    That justification does not exist in the tree. The only ``taskkill /IM`` in
    the repo is ``bobert_companion._reap_wedged_ollama`` and its images are
    ``ollama.exe`` / ``llama-server.exe`` / ``ollama app.exe``; window reuse is
    done with ``pygetwindow``'s ``w.close()``
    (``_close_browser_windows_matching``). So the exemption bought nothing and
    cost an unbounded hole in a guard whose whole point is that a test run must
    not be able to take over the box: one ``taskkill /IM chrome.exe /F`` closes
    every Chrome window the owner has open — strictly worse than the tab spam
    this guard was written for."""

    def test_killing_a_browser_by_image_name_is_refused(self):
        _, sp, _, calls = self._install_fakes()
        for cmd in (["taskkill", "/F", "/IM", "chrome.exe"],
                    ["pkill", "firefox"],
                    ["killall", "google-chrome"]):
            with self.subTest(cmd=cmd):
                sp.Popen(cmd)
        self.assertEqual(calls, [], "a real browser kill was executed")
        self.assertEqual(len(browser_guard.blocked_attempts()), 3)

    def test_killing_a_non_browser_by_image_name_still_works(self):
        """The monolith's ollama reaper and skills/code_executor's PID kill are
        the real users of taskkill, and neither may be touched."""
        _, sp, _, calls = self._install_fakes()
        sp.Popen(["taskkill", "/IM", "ollama.exe", "/F", "/T"])
        sp.Popen(["taskkill", "/F", "/T", "/PID", "1234"])
        self.assertEqual(browser_guard.blocked_attempts(), ())
        self.assertEqual(len(calls), 2)

    def test_read_only_process_tools_naming_a_browser_still_pass_through(self):
        """These enumerate or filter; they cannot open OR close a window, so
        exempting the whole command line costs nothing."""
        _, sp, _, calls = self._install_fakes()
        for cmd in (["tasklist", "/FI", "IMAGENAME eq msedge.exe"],
                    ["pgrep", "chrome"],
                    "wmic process where name='chrome.exe' get ProcessId",
                    ["where", "chrome.exe"]):
            with self.subTest(cmd=cmd):
                sp.Popen(cmd)
        self.assertEqual(browser_guard.blocked_attempts(), ())
        self.assertEqual(len(calls), 4)

    def test_no_kill_verb_is_in_the_unconditional_exemption(self):
        """STALE-DUPLICATE GUARD, structural rather than textual: the exemption
        list is what actually decides, and a kill verb landing back in it would
        silently reopen the hole no matter what the comment says."""
        overlap = sorted(browser_guard._PROCESS_TOOLS
                         & browser_guard._KILL_TOOLS)
        self.assertEqual(overlap, [],
                         "these kill verbs exempt the WHOLE command line from "
                         f"the browser scan again: {overlap}")
        for verb in ("taskkill", "taskkill.exe", "pkill", "killall", "kill"):
            with self.subTest(verb=verb):
                self.assertNotIn(verb, browser_guard._PROCESS_TOOLS)
                self.assertIn(verb, browser_guard._KILL_TOOLS)


class CrossGuardTests(_GuardTestCase):
    """``_reset_for_tests()`` restores ``subprocess.Popen`` to the STOCK class,
    which throws away anything another guard parked on the subclass it replaces.

    On 2026-08-20 that silently uninstalled ``core/no_window_subprocess`` — the
    net that exists because ~38 ghost Windows Terminal windows appeared in 30
    minutes — for the rest of every run, while ``no_window_subprocess.install()``
    went on returning True because it latched on ``_ORIG_INIT[0]``. Same shape
    as the live-data defect already fixed that day, aimed at a THIRD net.
    tests/test_live_data_guard.py has the sibling assertion for the live-data
    guard; this is the one for the no-window net."""

    def test_unlatching_leaves_the_no_window_net_installed(self):
        from core import no_window_subprocess as nw
        if os.name != "nt":
            self.skipTest("the no-window net is Windows-only")
        # Importing bobert_companion.py is what arms this net in a real run;
        # install() is idempotent and repair-only, so calling it here reproduces
        # that state without dragging the monolith in.
        nw.install()
        self.assertTrue(nw.is_armed(),
                        "the no-window net would not arm at all")
        with _unlatched():
            browser_guard.install(quiet=True)
            self.assertTrue(
                nw.is_armed(),
                "browser_guard._reset_for_tests() + install() discarded "
                "core/no_window_subprocess — from here on a bare "
                "subprocess.run() in production code under test spawns a "
                "VISIBLE console window on the owner's desktop, and "
                "no_window_subprocess.install() still reports itself installed")
        self.assertTrue(nw.is_armed())

    def test_unlatching_leaves_the_live_data_guard_armed(self):
        from tests import live_data_guard
        if not live_data_guard._INSTALLED:
            self.skipTest("live-data guard not installed")
        with _unlatched():
            browser_guard.install(quiet=True)
            self.assertEqual(live_data_guard.unarmed_hooks(), [])
        self.assertEqual(live_data_guard.unarmed_hooks(), [])

    def test_the_undo_never_clobbers_a_sibling_that_wrapped_us(self):
        """``_reset_for_tests()`` must not be able to DELETE another guard.

        Whichever guard is armed first ends up underneath, and until 2026-08-20
        our undo callable did a blind ``setattr(module, name, original)``: when a
        sibling had wrapped our stub, that put OUR pre-guard snapshot back and
        threw the sibling's hook away. Under the tools/ runners (which armed the
        browser guard with no live-data guard in the process yet) that snapshot
        was the RAW ``os.startfile``, so the first ``_unlatched()`` in this
        module left the rest of the CI run with NO interception on the shell
        route to a live JARVIS. The ordering is fixed in every runner now; this
        makes the property hold regardless of who installs first.

        Fake modules, so the running suite's own hooks are never touched."""
        wb, sp, osmod, _calls = _fake_modules()
        raw = osmod.startfile
        undos = browser_guard._install_into(wb, sp, osmod)
        our_stub = osmod.startfile
        self.assertIsNot(our_stub, raw, "the guard did not wrap startfile")

        # A sibling guard arriving AFTER us, wrapping our stub (this is exactly
        # what tests/live_data_guard.py does to os.startfile).
        def _sibling(path, *a, **k):
            return our_stub(path, *a, **k)
        _sibling.__wrapped__ = our_stub
        osmod.startfile = _sibling

        for undo in undos:
            undo()
        self.assertIs(osmod.startfile, _sibling,
                      "browser_guard's undo discarded a sibling guard's hook — "
                      "the run is now unprotected on os.startfile and nothing "
                      "reports it")
        # ...and with nobody on top, the undo still restores exactly as before:
        # the identity check must not turn reset into a no-op.
        osmod.startfile = raw
        undos = browser_guard._install_into(wb, sp, osmod)
        self.assertIsNot(osmod.startfile, raw, "re-install did not re-wrap")
        for undo in undos:
            undo()
        self.assertIs(osmod.startfile, raw,
                      "the undo stopped restoring even when nothing else had "
                      "touched the attribute")

    def test_a_sibling_guard_wrapping_our_stub_is_not_read_as_displacement(self):
        """``unarmed_targets()`` must look THROUGH a wrapper another guard put
        on top of our stub, or two correctly-cooperating guards read as one
        broken one."""
        saved = webbrowser.open
        try:
            def _sibling(url, *a, **k):
                return saved(url, *a, **k)
            _sibling.__wrapped__ = saved
            webbrowser.open = _sibling
            self.assertNotIn("webbrowser.open", browser_guard.unarmed_targets())
        finally:
            webbrowser.open = saved

    def test_unarmed_targets_is_not_fooled_by_a_magicmock(self):
        saved = webbrowser.open
        try:
            webbrowser.open = mock.MagicMock()
            self.assertIn("webbrowser.open", browser_guard.unarmed_targets())
        finally:
            webbrowser.open = saved


class WiringTests(unittest.TestCase):
    """STALE-DUPLICATE GUARD — this repo's #1 bug class is a rule that quietly
    stops being applied in one of its copies. The guard is worthless if the
    wiring is dropped, so these read the SOURCE of every place that installs it.

    Deliberately NOT a _GuardTestCase: these touch no ledger and no modules."""

    def test_tests_package_installs_the_guard(self):
        """THE chokepoint. tests/__init__.py is what covers `python -m unittest
        tests.foo`, the capped harness and an IDE — paths no runner sees."""
        src = _source(_TESTS_INIT)
        self.assertIn("browser_guard", src,
                      "tests/__init__.py no longer references the browser guard")
        tree = ast.parse(src)
        found = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and _call_name(node) == "install"
                 and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Name)
                 and node.func.value.id.lstrip("_").startswith("browser_guard")]
        self.assertTrue(found,
                        "tests/__init__.py must call browser_guard.install() — "
                        "without it, `python -m unittest tests.foo` and the "
                        "capped harness run with NO browser guard at all")

    def test_tests_package_install_is_not_hidden_inside_a_function(self):
        """It has to run on IMPORT of the package. A call parked in a helper
        that nothing invokes would pass the test above and guard nothing."""
        tree = ast.parse(_source(_TESTS_INIT))
        for node in tree.body:
            self.assertNotIsInstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                "tests/__init__.py must stay a flat module body so the guard "
                "install runs at import time")

    def test_tests_package_install_cannot_break_collection(self):
        """An exception from the guard must never stop the suite from being
        collected — an unguarded run still beats no run at all."""
        tree = ast.parse(_source(_TESTS_INIT))
        wrapped = any(
            isinstance(node, ast.Try)
            and any(isinstance(inner, ast.Call) and _call_name(inner) == "install"
                    for inner in ast.walk(node))
            for node in tree.body)
        self.assertTrue(wrapped,
                        "the browser_guard.install() call in tests/__init__.py "
                        "must sit inside a try/except")

    def _install_and_first_test_touch(self, filename):
        tree = ast.parse(_source(os.path.join(_TOOLS_DIR, filename)))
        main = next((n for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
        self.assertIsNotNone(main, f"{filename}: no module-level main()")
        guard_line = touch_line = None
        for node in ast.walk(main):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            # ast.walk is not source-ordered — take the minimum line, not the
            # first one seen.
            if name == "install":
                if guard_line is None or node.lineno < guard_line:
                    guard_line = node.lineno
            elif name in _TEST_TOUCHING_CALLS:
                if touch_line is None or node.lineno < touch_line:
                    touch_line = node.lineno
        return guard_line, touch_line

    def test_every_runner_imports_the_guard(self):
        for filename in _GUARDED_RUNNERS:
            with self.subTest(runner=filename):
                self.assertIn("from tools import browser_guard",
                              _source(os.path.join(_TOOLS_DIR, filename)),
                              f"{filename} no longer imports the browser guard")

    def test_every_runner_installs_before_touching_tests(self):
        for filename in _GUARDED_RUNNERS:
            with self.subTest(runner=filename):
                guard_line, touch_line = self._install_and_first_test_touch(filename)
                self.assertIsNotNone(
                    guard_line, f"{filename}: main() never calls install()")
                self.assertIsNotNone(
                    touch_line, f"{filename}: main() no longer runs any tests?")
                self.assertLess(
                    guard_line, touch_line,
                    f"{filename}: the browser guard is installed at line "
                    f"{guard_line}, AFTER tests/subprocesses start at line "
                    f"{touch_line} — a leak in an early import would get through")

    @staticmethod
    def _discovery_runners() -> list[str]:
        found = []
        for name in sorted(os.listdir(_TOOLS_DIR)):
            if not name.endswith(".py"):
                continue
            src = _source(os.path.join(_TOOLS_DIR, name))
            if any(marker in src for marker in _DISCOVERY_MARKERS):
                found.append(name)
        return found

    def test_no_new_unguarded_test_runner_appears(self):
        offenders = [name for name in self._discovery_runners()
                     if name not in _KNOWN_UNGUARDED
                     and "browser_guard" not in _source(os.path.join(_TOOLS_DIR, name))]
        self.assertEqual(
            offenders, [],
            "these tools/ modules run the unittest suite with NO browser guard "
            "— call tools.browser_guard.install() beside apply_memory_ceiling() "
            f"in main(): {offenders}")

    def test_guarded_list_covers_every_discovery_runner(self):
        """The positive half: the ordering checks above only look at the files
        NAMED in _GUARDED_RUNNERS, so one silently dropped from that tuple would
        stop being checked at all."""
        accounted = set(_GUARDED_RUNNERS) | _KNOWN_UNGUARDED
        found = set(self._discovery_runners())
        self.assertEqual(sorted(found - accounted), [],
                         "discovery runners no test checks for the browser "
                         f"guard: {sorted(found - accounted)}")
        self.assertEqual(sorted(accounted - found), [],
                         "listed runners that no longer discover the suite "
                         f"(stale entries): {sorted(accounted - found)}")

    def test_known_unguarded_list_stays_honest(self):
        for name in sorted(_KNOWN_UNGUARDED):
            path = os.path.join(_TOOLS_DIR, name)
            with self.subTest(module=name):
                self.assertTrue(os.path.exists(path),
                                f"{name} is gone — drop it from _KNOWN_UNGUARDED")
                self.assertNotIn("browser_guard", _source(path),
                                 f"{name} now installs the guard — remove it "
                                 "from _KNOWN_UNGUARDED so the scan covers it")

    def test_guard_and_memory_ceiling_are_wired_in_the_same_place(self):
        """Both guards protect the box from a test run; they are applied
        together at the top of every main(). Keeping them adjacent is what stops
        one of them being lost in a refactor of the other."""
        for filename in _GUARDED_RUNNERS:
            with self.subTest(runner=filename):
                src = _source(os.path.join(_TOOLS_DIR, filename))
                mem = src.index("apply_memory_ceiling()")
                brw = src.index("browser_guard.install()")
                self.assertLess(mem, brw)
                self.assertLess(
                    src.count("\n", mem, brw), 12,
                    f"{filename}: the browser guard drifted away from the "
                    "memory ceiling — keep them in one block")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    unittest.main()
