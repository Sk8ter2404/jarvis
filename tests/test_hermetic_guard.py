"""tools/hermetic_guard - the refusal of a TEST RUN reaching the network, the
owner's keyboard / mouse / windows, and live-hardware probes.

THE FINDINGS (2026-09-30)
=========================
With a live JARVIS on the owner's PC, a whole-suite audit found tests that
reached the real world through production code: the live Ollama (/api/ps,
/api/tags) from the dashboard and preflight suites, itunes.apple.com from two
Apple Music tests, the real nvidia-smi / `ollama ps`, a real
SetForegroundWindow, and a HUD overlay window put on the desktop by a skill's
register(). All green - for the wrong reason.

These tests pin the guard WITHOUT reaching anything:

* the verdicts are pure functions, tested directly;
* the hook is exercised with SYNTHETIC audit events (``sys.audit`` runs the
  hooks and nothing else - no socket, no process, no user32 call);
* the few tests that make a REAL call (a connect, a lookup, a spawn, a
  SetCursorPos) run only while the guard is proven armed, so the refusal
  happens before anything leaves the process - they can never reach the thing
  they are about;
* the regression class re-runs every test the audit caught, in-process, and
  requires that it now reaches nothing (the fixes in the tests themselves);
* the wiring class reads the SOURCE of tests/__init__.py and the runners
  (this repo's #1 bug class is a rule that stops being applied in one copy).

Every test that records a refusal restores the process-wide ledger, so this
file never pollutes the offender list the atexit summary prints.
"""
from __future__ import annotations

import ast
import contextlib
import http.server
import os
import socket
import subprocess
import sys
import threading
import unittest
import urllib.request
from unittest import mock

from tools import browser_guard, hermetic_guard as hg

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.join(_PROJECT_ROOT, "tools")
_TESTS_INIT = os.path.join(_PROJECT_ROOT, "tests", "__init__.py")
_GUARDED_RUNNERS = ("run_tests.py", "run_tests_ci_sim.py", "run_coverage.py")
_TEST_NET = "192.0.2.1"          # RFC 5737 documentation address


def _source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


@contextlib.contextmanager
def _ledger_restored():
    """Keep this file's own (deliberate) refusals out of the run's report."""
    with hg._lock:
        saved = list(hg._refusals)
    try:
        yield
    finally:
        with hg._lock:
            hg._refusals[:] = saved


class _Armed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hg.install(quiet=True)          # idempotent; tests/__init__ armed it

    def require_armed(self, guard):
        if guard in hg.unarmed_guards():
            self.skipTest(f"the {guard} guard is not armed in this run - a "
                          f"REAL call here would reach the real thing")


# ─── pure verdicts ─────────────────────────────────────────────────────────

class LocalHostTests(unittest.TestCase):

    def test_loopback_in_every_spelling_is_local(self):
        for host in (None, "", "localhost", "LOCALHOST", "127.0.0.1",
                     "127.9.8.7", "::1", "[::1]", "::ffff:127.0.0.1",
                     "0.0.0.0", "::", "app.localhost", b"127.0.0.1"):
            with self.subTest(host=host):
                self.assertTrue(hg.is_local_host(host))

    def test_everything_else_is_not(self):
        for host in ("172.16.0.10", "10.0.0.1", _TEST_NET, "8.8.8.8",
                     "2001:db8::1", "itunes.apple.com", "example.com",
                     "<broadcast>", "fe80::1%eth0"):
            with self.subTest(host=host):
                self.assertFalse(hg.is_local_host(host))


class NetworkVerdictTests(unittest.TestCase):

    def v(self, event, *args):
        return hg.network_verdict(event, args)

    def test_non_loopback_connect_is_refused(self):
        self.assertIn("not this machine",
                      self.v("socket.connect", None, (_TEST_NET, 443)))
        self.assertIn("not this machine",
                      self.v("socket.connect", None, ("2001:db8::1", 80, 0, 0)))

    def test_live_local_services_are_refused_even_on_loopback(self):
        for port, name in hg.LIVE_SERVICE_PORTS.items():
            with self.subTest(port=port):
                why = self.v("socket.connect", None, ("127.0.0.1", port))
                self.assertIn(name, why)
                self.assertIn(name, self.v("socket.connect", None,
                                           ("::1", port, 0, 0)))
        self.assertIn(11434, hg.LIVE_SERVICE_PORTS)     # Ollama
        self.assertIn(8766, hg.LIVE_SERVICE_PORTS)      # the web UI

    def test_test_local_servers_on_other_loopback_ports_pass(self):
        for port in (0, 1024, 18766, 51234, 65535):
            with self.subTest(port=port):
                self.assertIsNone(self.v("socket.connect", None,
                                         ("127.0.0.1", port)))
        self.assertIsNone(self.v("socket.connect", None, "/tmp/x.sock"))

    def test_binding_a_live_port_is_refused_any_other_bind_passes(self):
        self.assertIn("live port", self.v("socket.bind", None, ("0.0.0.0", 8766)))
        self.assertIn("live port", self.v("socket.bind", None, ("127.0.0.1", 11434)))
        self.assertIsNone(self.v("socket.bind", None, ("127.0.0.1", 0)))
        self.assertIsNone(self.v("socket.bind", None, ("0.0.0.0", 0)))

    def test_lookups_that_leave_the_box_are_refused(self):
        self.assertIn("leaves this machine",
                      self.v("socket.getaddrinfo", "itunes.apple.com", 443, 0, 0, 0))
        self.assertIn("leaves this machine",
                      self.v("socket.gethostbyname", "example.com"))
        self.assertIn("leaves this machine",
                      self.v("socket.gethostbyaddr", "8.8.8.8"))
        for host in ("localhost", "127.0.0.1", "10.1.2.3", None, ""):
            with self.subTest(host=host):
                self.assertIsNone(self.v("socket.getaddrinfo", host, 80, 0, 0, 0))
        # HTTPServer.server_bind resolves its own loopback name
        self.assertIsNone(self.v("socket.gethostbyaddr", "127.0.0.1"))

    def test_datagrams_to_the_lan_are_refused(self):
        self.assertIsNotNone(self.v("socket.sendto", None, ("255.255.255.255", 9999)))
        self.assertIsNotNone(self.v("socket.sendmsg", None, ("239.255.255.250", 1900)))
        self.assertIsNone(self.v("socket.sendto", None, ("127.0.0.1", 9999)))


class ProbeVerdictTests(unittest.TestCase):

    def test_named_probes_are_refused_in_every_spelling(self):
        for cmd in ("nvidia-smi --query-gpu=utilization.gpu",
                    r"C:\WINDOWS\system32\nvidia-smi.EXE -L",
                    ["nvidia-smi", "-L"], "ollama ps", ["ollama", "stop", "m"],
                    "powercfg /getactivescheme", "shutdown /s /t 0",
                    ["ping", "-n", "1", "1.1.1.1"], ["claude", "--print", "hi"]):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(hg.probe_verdict(None, cmd))

    def test_through_a_shell_too(self):
        self.assertIn("through cmd", hg.probe_verdict(None, 'cmd /c "nvidia-smi -L"'))
        self.assertIsNotNone(hg.probe_verdict(
            None, ["powershell", "-Command", "Get-PnpDevice -Class Camera"]))
        self.assertIsNotNone(hg.probe_verdict(
            None, 'powershell.exe -NoProfile -Command "Get-CimInstance '
                  'Win32_VideoController | Select Name"'))
        self.assertIsNotNone(hg.probe_verdict(
            None, ["bash", "-c", "ollama list"]))

    def test_the_executable_argument_counts(self):
        self.assertIsNotNone(hg.probe_verdict(r"C:\x\nvidia-smi.exe", "anything"))

    def test_hud_overlays_are_refused(self):
        self.assertIn("HUD overlay", hg.probe_verdict(
            None, r'C:\Py\python.exe C:\JARVIS\hud\workshop_hud.py --x 1'))
        self.assertIn("HUD overlay", hg.probe_verdict(
            None, [sys.executable, "-m", "hud.jarvis_reticle"]))
        self.assertIsNone(hg.probe_verdict(
            None, [sys.executable, os.path.join("core", "tts.py")]))

    def test_desktop_file_launchers_are_refused(self):
        # Each opens a file in its app on the owner's desktop. Found
        # 2026-09-30: ReadChangelogTests ran `xdg-open CHANGELOG.md` under
        # ci-sim's Linux simulation (absent here, so it failed quietly).
        for exe, cmd in (
                (None, ["xdg-open", "/tmp/CHANGELOG.md"]),
                (None, "xdg-open /tmp/CHANGELOG.md"),
                (None, ["open", "picture.png"]),
                (None, ["explorer.exe", r"shell:AppsFolder\Some.App!App"]),
                (r"C:\Windows\explorer.exe", r"/select,C:\x\y.txt"),
                (None, ["kde-open5", "x.pdf"]), (None, ["wslview", "x.pdf"])):
            with self.subTest(cmd=cmd):
                why = hg.probe_verdict(exe, cmd)
                self.assertIsNotNone(why)
                self.assertIn("desktop file launcher", why)

    def test_shell_launchers_are_refused_in_command_position(self):
        for cmd in (["cmd", "/c", "start", "", r"C:\x\CHANGELOG.md"],
                    # the string Windows audits for Popen(..., shell=True)
                    r'C:\WINDOWS\system32\cmd.exe /c "start "" C:\x\a.md"',
                    ["powershell", "-NoProfile", "-Command",
                     r"Start-Process 'C:\x\a.md'"],
                    ["powershell", "-Command", r"Get-Date; Invoke-Item C:\x"],
                    'powershell -Command "ii ."',
                    ["bash", "-c", "xdg-open ./a.md"]):
            with self.subTest(cmd=cmd):
                why = hg.probe_verdict(None, cmd)
                self.assertIsNotNone(why)
                self.assertIn("desktop file launcher", why)

    def test_launcher_words_as_arguments_pass(self):
        # "start" / "open" / "explorer" as an ARGUMENT is not a launch.
        for cmd in (["cmd", "/c", "echo", "start"],
                    "git log --format=open",
                    ["powershell", "-Command",
                     "Get-Service | Where-Object Status -eq 'open'"],
                    'tasklist /FI "IMAGENAME eq explorer.exe"',
                    ["notes-open-later.txt"]):
            with self.subTest(cmd=cmd):
                self.assertIsNone(hg.probe_verdict(None, cmd))

    def test_ordinary_commands_pass(self):
        for cmd in ("git ls-files -z", [sys.executable, "-c", "print(1)"],
                    "python -m pyflakes tests", "tasklist /FI \"IMAGENAME eq x.exe\"",
                    ["powershell", "-Command", "Get-Date"],
                    ["claude-helper-notes.txt"], "where nvidia-smi"):
            with self.subTest(cmd=cmd):
                self.assertIsNone(hg.probe_verdict(None, cmd))


class InputVerdictTests(unittest.TestCase):

    def test_injection_cursor_and_focus_are_refused(self):
        for fn in ("SendInput", "keybd_event", "mouse_event", "SetCursorPos",
                   "SetForegroundWindow", "BringWindowToTop"):
            with self.subTest(fn=fn):
                self.assertIsNotNone(hg.input_verdict(fn, (0, 0)))

    def test_window_messages_that_type_click_or_close(self):
        for msg in (0x0010, 0x0100, 0x0102, 0x0201, 0x0112, 0x0319):
            with self.subTest(msg=hex(msg)):
                self.assertIsNotNone(hg.input_verdict("PostMessageW", (1, msg, 0, 0)))
                self.assertIsNotNone(hg.input_verdict("SendMessage", (1, msg, 0, 0)))
        # a broadcast WM_SETTINGCHANGE / a WM_GETTEXT read are not input
        self.assertIsNone(hg.input_verdict("SendMessageTimeoutW",
                                           (0xFFFF, 0x001A, 0, 0, 2, 5000, None)))
        self.assertIsNone(hg.input_verdict("SendMessageW", (1, 0x000D, 0, 0)))

    def test_read_only_calls_pass(self):
        self.assertIsNone(hg.input_verdict("GetCursorPos", ()))


# ─── the hook, through synthetic audit events (no I/O) ─────────────────────

class HookTests(_Armed):

    def test_a_refusal_raises_the_right_type_and_is_recorded(self):
        self.require_armed("network")
        with _ledger_restored():
            hg.reset()
            with self.assertRaises(hg.NetworkGuardError) as cm:
                sys.audit("socket.connect", None, ("127.0.0.1", 11434))
            self.assertIsInstance(cm.exception, ConnectionRefusedError)
            self.assertIn("[network-guard] REFUSED", str(cm.exception))
            self.assertIn("in " + self.id(), str(cm.exception))   # names the test
            self.assertIn("Ollama", str(cm.exception))
            (r,) = hg.refusals("network")
            self.assertEqual(r.test_id, self.id())
            self.assertIn("test_hermetic_guard.py", r.call_site)
            self.assertEqual(r.target, "127.0.0.1:11434")

    def test_a_refused_lookup_is_a_gaierror(self):
        self.require_armed("network")
        with _ledger_restored(), self.assertRaises(socket.gaierror):
            sys.audit("socket.getaddrinfo", "itunes.apple.com", 443, 0, 0, 0)

    def test_a_refused_probe_is_file_not_found(self):
        self.require_armed("probe")
        with _ledger_restored(), self.assertRaises(FileNotFoundError) as cm:
            sys.audit("subprocess.Popen", None, "nvidia-smi -L", None, None)
        self.assertIsInstance(cm.exception, hg.ProbeGuardError)

    def test_os_startfile_is_refused_before_it_launches(self):
        # CPython raises both events BEFORE ShellExecute, so a refusal opens
        # nothing. Synthetic here: a regressed guard must not open a file.
        self.require_armed("probe")
        target = os.path.join(_PROJECT_ROOT, "CHANGELOG.md")
        for event, args in (("os.startfile", (target, "open")),
                            ("os.startfile/2",
                             (target, "open", "", None, 1))):
            with self.subTest(event=event), _ledger_restored():
                hg.reset()
                with self.assertRaises(hg.ProbeGuardError) as cm:
                    sys.audit(event, *args)
                self.assertIsInstance(cm.exception, FileNotFoundError)
                self.assertIn("desktop file launcher", str(cm.exception))
                (r,) = hg.refusals("probe")
                self.assertEqual((r.api, r.target), (event, target))
                self.assertEqual(r.test_id, self.id())

    def test_a_launcher_opt_in_lets_it_through(self):
        self.require_armed("probe")
        with _ledger_restored():
            hg.reset()
            with hg.allow("probe", reason="a test that opens a file"):
                sys.audit("os.startfile", "x.md", "open")
                sys.audit("subprocess.Popen", None, ["xdg-open", "x.md"],
                          None, None)
            self.assertEqual(hg.refusals(), ())
            with self.assertRaises(hg.ProbeGuardError):
                sys.audit("subprocess.Popen", None, ["xdg-open", "x.md"],
                          None, None)

    def test_allowed_events_pass_straight_through(self):
        with _ledger_restored():
            hg.reset()
            sys.audit("socket.connect", None, ("127.0.0.1", 51234))
            sys.audit("subprocess.Popen", None, "git status", None, None)
            sys.audit("socket.getaddrinfo", "localhost", 80, 0, 0, 0)
            self.assertEqual(hg.refusals(), ())

    def test_opt_in_lets_one_guard_through_on_every_thread(self):
        self.require_armed("network")
        with _ledger_restored():
            hg.reset()
            with hg.allow("network", reason="a test that needs a socket"):
                sys.audit("socket.connect", None, (_TEST_NET, 80))
                errors = []

                def other_thread():
                    try:
                        sys.audit("socket.connect", None, (_TEST_NET, 81))
                    except Exception as e:  # noqa: BLE001
                        errors.append(e)
                t = threading.Thread(target=other_thread)
                t.start()
                t.join(5)
                self.assertEqual(errors, [])
                self.assertIn("network", hg.unarmed_guards())
                # only the named guard is lifted
                with self.assertRaises(hg.ProbeGuardError):
                    sys.audit("subprocess.Popen", None, "nvidia-smi", None, None)
            with self.assertRaises(hg.NetworkGuardError):
                sys.audit("socket.connect", None, (_TEST_NET, 80))
            self.assertNotIn("network", hg.unarmed_guards())

    def test_allow_is_also_a_decorator_and_rejects_unknown_names(self):
        calls = []

        @hg.allow("probe")
        def run():
            sys.audit("subprocess.Popen", None, "nvidia-smi", None, None)
            calls.append(1)
        with _ledger_restored():
            run()
        self.assertEqual(calls, [1])
        with self.assertRaises(ValueError):
            with hg.allow("camera"):
                pass

    def test_a_background_thread_is_attributed_to_the_running_test(self):
        self.require_armed("network")
        with _ledger_restored():
            hg.reset()
            done = threading.Event()

            def worker():
                try:
                    sys.audit("socket.connect", None, (_TEST_NET, 443))
                except hg.NetworkGuardError:
                    pass
                done.set()
            threading.Thread(target=worker, name="leaked-worker").start()
            self.assertTrue(done.wait(5))
            (r,) = hg.refusals("network")
            self.assertEqual(r.test_id, self.id() + " (during)")

    def test_input_events_by_function_pointer(self):
        if not hg._input_fn_ptrs:
            self.skipTest("no user32 on this host")
        self.require_armed("input")
        ptr = {n: p for p, n in hg._input_fn_ptrs.items()}["SetCursorPos"]
        with _ledger_restored(), self.assertRaises(hg.InputGuardError):
            sys.audit("ctypes.call_function", ptr, (10, 10))
        post = {n: p for p, n in hg._message_fn_ptrs.items()}.get("PostMessageW")
        if post:
            with _ledger_restored(), self.assertRaises(hg.InputGuardError):
                sys.audit("ctypes.call_function", post, (1, 0x0010, 0, 0))
            sys.audit("ctypes.call_function", post, (1, 0x001A, 0, 0))  # passes

    def test_the_summary_names_the_test_the_target_and_the_site(self):
        self.require_armed("probe")
        with _ledger_restored():
            hg.reset()
            with self.assertRaises(hg.ProbeGuardError):
                sys.audit("subprocess.Popen", None, "ollama ps", None, None)
            text = hg.summary()
        self.assertIn("[probe-guard] refused 1 real probe reach(es) from 1 test(s)", text)
        self.assertIn(self.id(), text)
        self.assertIn("ollama ps", text)
        self.assertIn("test_hermetic_guard.py", text)
        self.assertIn("hermetic_guard.allow('probe')", text)

    def test_a_clean_run_prints_nothing(self):
        with _ledger_restored():
            hg.reset()
            self.assertEqual(hg.summary(), "")


class ArmedByBehaviourTests(_Armed):

    def test_every_guard_answers_armed_by_refusing_a_probe(self):
        self.assertEqual(hg.unarmed_guards(), [])
        self.assertTrue(hg.is_armed())

    def test_the_probe_never_reaches_the_ledger(self):
        with _ledger_restored():
            hg.reset()
            hg.unarmed_guards()
            self.assertEqual(hg.refusals(), ())

    def test_install_is_idempotent_and_prints_once(self):
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertTrue(hg.install())
            self.assertTrue(hg.install())
        self.assertEqual(buf.getvalue(), "")
        self.assertIn("[hermetic-guard] armed", hg.banner())

    def test_escape_hatches_are_read_from_the_environment(self):
        for guard, var in hg.ENV_ESCAPES.items():
            with self.subTest(guard=guard):
                self.assertTrue(hg._env_allows(guard, {var: "1"}))
                self.assertFalse(hg._env_allows(guard, {var: "0"}))
                self.assertFalse(hg._env_allows(guard, {}))

    def test_pywin32_input_functions_are_wrapped(self):
        if sys.platform != "win32":
            self.skipTest("pywin32 is Windows-only")
        try:
            import win32api  # noqa: F401
        except ImportError:
            self.skipTest("pywin32 not installed")
        self.assertEqual(hg.unwrapped_pywin32(), [])
        from unittest import mock
        import win32api as w
        with mock.patch.object(w, "SetCursorPos"):
            self.assertIn("win32api.SetCursorPos", hg.unwrapped_pywin32())
        self.assertEqual(hg.unwrapped_pywin32(), [],
                         "mock.patch must restore the guard's stub")


# ─── real calls, refused before anything leaves the process ────────────────

class RealCallTests(_Armed):

    def test_a_real_connect_to_the_internet_is_refused(self):
        self.require_armed("network")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with _ledger_restored(), self.assertRaises(ConnectionRefusedError):
                s.connect((_TEST_NET, 80))
        finally:
            s.close()

    def test_a_real_request_to_the_live_ollama_is_refused(self):
        self.require_armed("network")
        with _ledger_restored(), self.assertRaises(OSError) as cm:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        self.assertIn("[network-guard]", str(cm.exception))

    def test_a_real_lookup_is_refused(self):
        self.require_armed("network")
        with _ledger_restored(), self.assertRaises(socket.gaierror):
            socket.getaddrinfo("itunes.apple.com", 443)

    def test_binding_the_live_web_ui_port_is_refused(self):
        self.require_armed("network")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with _ledger_restored(), self.assertRaises(OSError):
                s.bind(("127.0.0.1", 8766))
        finally:
            s.close()

    def test_a_test_local_server_on_an_ephemeral_port_still_works(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = b"local ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            port = srv.server_address[1]
            with _ledger_restored():
                hg.reset()
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                    self.assertEqual(r.read(), b"local ok")
                with urllib.request.urlopen(f"http://localhost:{port}/", timeout=5) as r:
                    self.assertEqual(r.read(), b"local ok")
                self.assertEqual(hg.refusals(), ())
        finally:
            srv.shutdown()
            srv.server_close()

    def test_a_real_nvidia_smi_launch_is_refused(self):
        self.require_armed("probe")
        with _ledger_restored(), self.assertRaises(FileNotFoundError):
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10)

    def test_a_real_xdg_open_launch_is_refused_by_the_guard(self):
        # Safe if the guard regresses: xdg-open does not exist on the Windows
        # box (a plain FileNotFoundError, which fails the assertion below).
        self.require_armed("probe")
        with _ledger_restored(), self.assertRaises(hg.ProbeGuardError):
            subprocess.Popen(["xdg-open", os.path.join(_PROJECT_ROOT,
                                                       "CHANGELOG.md")],
                             close_fds=True)

    def test_the_changelog_test_launches_nothing_under_ci_sim(self):
        # The offender itself, re-run the way ci-sim runs it (sys.platform
        # "linux" -> the xdg-open branch): no refusal may be recorded.
        test_id = ("tests.test_actions_sec3.ReadChangelogTests."
                   "test_long_entry_opens_file")
        with _ledger_restored(), \
                mock.patch.object(sys, "platform", "linux"):
            h0 = len(hg.refusals())
            result = unittest.TestResult()
            unittest.TestLoader().loadTestsFromName(test_id).run(result)
            hits = hg.refusals()[h0:]
        self.assertEqual([tb.splitlines()[-1] for _t, tb in
                          result.errors + result.failures], [])
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(hits, ())

    def test_an_ordinary_subprocess_still_runs(self):
        out = subprocess.run([sys.executable, "-c", "print('hermetic-ok')"],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.stdout.strip(), "hermetic-ok")

    def test_a_real_setcursorpos_is_refused_and_the_cursor_stays(self):
        if sys.platform != "win32" or not hg._input_fn_ptrs:
            self.skipTest("no user32 on this host")
        self.require_armed("input")
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32")
        before = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(before))
        with _ledger_restored(), self.assertRaises(hg.InputGuardError):
            user32.SetCursorPos(before.x + 37, before.y + 11)
        after = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(after))
        # (the owner may move his own mouse meanwhile; ours never lands)
        self.assertNotEqual((after.x, after.y), (before.x + 37, before.y + 11))


# ─── every test the audit caught now reaches nothing ───────────────────────

# (test id, what it used to reach) - the audit of 2026-09-30. Each is re-run
# in-process; it must pass AND leave no refusal / browser block behind.
_FIXED_OFFENDERS = (
    ("tests.monolith.test_monolith_sec4.StreamingAutoPlayBranchTests."
     "test_vision_click_failsafe_returns_message", "a real browser"),
    ("tests.monolith.test_monolith_sec4.StreamingAutoPlayBranchTests."
     "test_keyboard_path_delegates_to_play_and_verify",
     "itunes.apple.com + a real browser"),
    ("tests.monolith.test_monolith_sec4.AppleMusicAutoPlayNoVisionTests."
     "test_no_ui_automation_degrades_clearly", "itunes.apple.com"),
    ("tests.monolith.test_monolith_sec4.AppleMusicPlaylistSidebarTests."
     "test_sidebar_playlists_click_failsafe", "a real browser"),
    ("tests.monolith.test_monolith_sec4.StreamingApplyPlayStrategyTests."
     "test_space_strategy", "SetForegroundWindow"),
    ("tests.monolith.test_monolith_sec7.StartupPreflightTests."
     "test_subcheck_exceptions_are_swallowed", "the live Ollama /api/tags"),
    ("tests.skills.test_system_pulse.PulseGatherTests."
     "test_gather_assembles_all_fields_with_battery", "nvidia-smi"),
    ("tests.skills.test_system_pulse.PulseGatherTests."
     "test_gather_omits_battery_on_desktop", "nvidia-smi"),
    ("tests.test_web_interface.AirMouseStatusTests."
     "test_build_status_includes_air_mouse_when_loaded",
     "the live Ollama /api/ps, ollama ps, nvidia-smi"),
    ("tests.test_web_interface.StatusEndpointTests."
     "test_status_carries_uptime_field", "the live Ollama /api/ps"),
    ("tests.test_web_interface.ControlPanelEndpointTests."
     "test_system_returns_expected_keys", "the live Ollama /api/ps"),
    ("tests.test_web_interface_audit_fixes.CpuPercentTests."
     "test_cpu_is_measured_across_request_threads", "the live Ollama /api/ps"),
    ("tests.test_web_interface.DashboardTokenSerializationTests."
     "test_hostile_token_round_trips_and_authenticates",
     "the live Ollama /api/ps, ollama ps, nvidia-smi"),
    ("tests.test_web_interface.SystemInfoHelperTests.test_shape_is_stable",
     "nvidia-smi and the live Ollama (when its caches are cold)"),
    ("tests.test_web_interface.BuildStatusGracefulTests."
     "test_status_with_all_sources_absent",
     "nvidia-smi and the live Ollama (when its caches are cold)"),
    ("tests.test_skills_smoke.SkillSmokeTests.test_smoke_holographic_overlay",
     "a HUD overlay window"),
    ("tests.skills.test_air_control.TestFailsafe."
     "test_pyautogui_missing_is_survivable",
     "a real left-button-up at the owner's cursor"),
)


@contextlib.contextmanager
def _cold_live_caches():
    """Run with the dashboard's GPU caches COLD (restored after): a probe
    result another test cached a moment ago (2-3 s TTLs) would otherwise hide
    an offender that still reaches the live Ollama / GPU."""
    saved = []
    try:
        from tools import web_interface as wi
        saved.append((wi._nvidia_smi_cache, dict(wi._nvidia_smi_cache)))
        wi._nvidia_smi_cache["ts"] = 0.0
    except Exception:  # noqa: BLE001 - not importable here: nothing cached
        pass
    try:
        from core import gpu_usage
        g_saved = (gpu_usage._cache, gpu_usage._cache_at)
        gpu_usage._cache = None
    except Exception:  # noqa: BLE001
        gpu_usage = None
    try:
        yield
    finally:
        for cell, value in saved:
            cell.clear()
            cell.update(value)
        if gpu_usage is not None:
            gpu_usage._cache, gpu_usage._cache_at = g_saved


class FixedOffendersStayHermeticTests(_Armed):
    """Re-run each test the audit caught; it must reach nothing now."""

    def test_each_fixed_offender_now_reaches_nothing(self):
        loader = unittest.TestLoader()
        for test_id, reached in _FIXED_OFFENDERS:
            with self.subTest(test=test_id):
                with _ledger_restored(), _cold_live_caches():
                    h0 = len(hg.refusals())
                    b0 = len(browser_guard.blocked_attempts())
                    result = unittest.TestResult()
                    loader.loadTestsFromName(test_id).run(result)
                    short = test_id.rsplit(".", 1)[-1]
                    hits = [r for r in hg.refusals()[h0:] if short in r.test_id]
                    blocks = [b for b in browser_guard.blocked_attempts()[b0:]
                              if short in b.test_id]
                    del browser_guard._blocked[b0:]
                self.assertEqual(
                    [(t.id(), tb.splitlines()[-1]) for t, tb in
                     result.errors + result.failures], [], test_id)
                self.assertEqual(hits, [],
                                 f"{test_id} still reaches {reached}")
                self.assertEqual(blocks, [],
                                 f"{test_id} still reaches a real browser")


# ─── wiring (source) ───────────────────────────────────────────────────────

def _call_name(node) -> str:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


class WiringTests(unittest.TestCase):

    def test_tests_package_installs_the_guard_at_import_in_a_try(self):
        tree = ast.parse(_source(_TESTS_INIT))
        found = False
        for node in tree.body:
            self.assertNotIsInstance(node, (ast.FunctionDef, ast.ClassDef))
            if isinstance(node, ast.Try):
                for inner in ast.walk(node):
                    if (isinstance(inner, ast.Call) and _call_name(inner) == "install"
                            and isinstance(inner.func, ast.Attribute)
                            and "hermetic_guard" in ast.unparse(inner.func.value)):
                        found = True
        self.assertTrue(found, "tests/__init__.py must call "
                               "hermetic_guard.install() at import, in a try")

    def test_it_is_armed_after_the_browser_guard(self):
        src = _source(_TESTS_INIT)
        self.assertLess(src.index("_browser_guard.install()"),
                        src.index("_hermetic_guard.install()"))

    def test_every_runner_installs_it_beside_the_browser_guard(self):
        for name in _GUARDED_RUNNERS:
            with self.subTest(runner=name):
                src = _source(os.path.join(_TOOLS_DIR, name))
                self.assertIn("from tools import hermetic_guard", src)
                b = src.index("browser_guard.install()")
                h = src.index("hermetic_guard.install()")
                self.assertLess(b, h)
                self.assertLess(src.count("\n", b, h), 6,
                                f"{name}: keep the hermetic guard in the "
                                f"browser guard's block")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
