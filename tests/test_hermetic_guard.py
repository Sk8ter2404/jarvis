"""tools/hermetic_guard - the refusal of a TEST RUN reaching the network, the
owner's keyboard / mouse / windows, live-hardware probes, and the owner's
screen.

THE FINDINGS (2026-09-30)
=========================
With a live JARVIS on the owner's PC, a whole-suite audit found tests that
reached the real world through production code: the live Ollama (/api/ps,
/api/tags) from the dashboard and preflight suites, itunes.apple.com from two
Apple Music tests, the real nvidia-smi / `ollama ps`, a real
SetForegroundWindow, and a HUD overlay window put on the desktop by a skill's
register(). All green - for the wrong reason. A tripwire run on 2026-10-02
added the screen: a monolith test photographed the whole desktop twice.

These tests pin the guard WITHOUT reaching anything:

* the verdicts are pure functions, tested directly;
* the hook is exercised with SYNTHETIC audit events (``sys.audit`` runs the
  hooks and nothing else - no socket, no process, no user32 call);
* the few tests that make a REAL call (a connect, a lookup, a spawn, a
  SetCursorPos, a screen BitBlt, an ImageGrab / mss grab) run only while the
  guard is proven armed, so the refusal happens before anything leaves the
  process - they can never reach the thing they are about (the capture ones
  also spy on every pixel source beneath the entry point);
* the regression class re-runs every test the audit caught, in-process, and
  requires that it now reaches nothing (the fixes in the tests themselves);
* the monolith class re-runs every test that injects ``bobert_companion=None``
  under a sys.meta_path trap, and requires that none of them imports the REAL
  bobert_companion.py from disk (2026-10-02);
* the wiring class reads the SOURCE of tests/__init__.py and the runners
  (this repo's #1 bug class is a rule that stops being applied in one copy).

Every test that records a refusal restores the process-wide ledger, so this
file never pollutes the offender list the atexit summary prints.
"""
from __future__ import annotations

import ast
import contextlib
import http.server
import importlib
import inspect
import os
import socket
import subprocess
import sys
import threading
import traceback
import types
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


_SRCCOPY = 0x00CC0020
_WHITENESS = 0x00FF0062


class ScreenVerdictTests(unittest.TestCase):

    def test_a_blit_from_a_screen_dc_is_refused(self):
        for fn in ("BitBlt", "StretchBlt"):
            with self.subTest(fn=fn):
                args = (10, 0, 0, 1, 1, 77, 0, 0, _SRCCOPY)
                self.assertIsNotNone(hg.screen_verdict(
                    fn, args, screen_dc=lambda dc: dc == 77))
                # the default answer for an unknown DC is "the screen"
                self.assertIsNotNone(hg.screen_verdict(fn, args))

    def test_ordinary_drawing_passes(self):
        # a memory-DC source, and a source-less raster op (a fill)
        self.assertIsNone(hg.screen_verdict(
            "BitBlt", (10, 0, 0, 1, 1, 11, 0, 0, _SRCCOPY),
            screen_dc=lambda dc: False))
        self.assertIsNone(hg.screen_verdict(
            "BitBlt", (10, 0, 0, 1, 1, None, 0, 0, _WHITENESS),
            screen_dc=hg._screen_dc))
        self.assertFalse(hg._screen_dc(0))

    def test_printwindow_is_refused_only_on_another_process_window(self):
        self.assertIsNotNone(hg.screen_verdict(
            "PrintWindow", (5, 10, 2), foreign=lambda h: True))
        self.assertIsNone(hg.screen_verdict(
            "PrintWindow", (5, 10, 2), foreign=lambda h: False))

    def test_other_functions_pass(self):
        self.assertIsNone(hg.screen_verdict("GetPixel", (10, 0, 0)))


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


# ─── un-audited real-world effects (COM / WinRT), 2026-10-01 ──────────────
# volume_mute / volume_unmute / set_volume (pycaw) and pause / resume / next /
# previous (the Windows media session) replaced media KEYS, which the input
# guard refused, with COM / WinRT calls no audit event covers: an unpinned
# test would really mute the owner's speakers or pause his media.

class EffectTargetsTests(_Armed):

    def test_the_two_effects_are_targets(self):
        self.assertIn(("core.media_now_playing", None, "_default_transport"),
                      hg._EFFECT_TARGETS)
        self.assertIn(("pycaw.utils", "AudioUtilities", "GetSpeakers"),
                      hg._EFFECT_TARGETS)

    def test_the_real_media_transport_is_refused_before_it_runs(self):
        self.require_armed("input")
        import core.media_now_playing as mnp
        self.assertTrue(hg._marked(mnp._default_transport))
        with _ledger_restored(), \
                mock.patch.object(mnp, "_transport_async",
                                  side_effect=AssertionError("reached WinRT")):
            with self.assertRaises(hg.InputGuardError):
                mnp._default_transport("pause", ())
            # transport() turns the refusal into an honest "failed".
            self.assertEqual(mnp.transport("next", runner=mnp._default_transport),
                             ("failed", None))
        self.assertEqual(hg.unwrapped_effects(), [])

    def test_a_patched_target_reports_unarmed_and_is_restored(self):
        import core.media_now_playing as mnp
        with mock.patch.object(mnp, "_default_transport"):
            self.assertIn("core.media_now_playing._default_transport",
                          hg.unwrapped_effects())
        self.assertEqual(hg.unwrapped_effects(), [],
                         "mock.patch must restore the guard's stub")

    def test_a_module_imported_later_is_wrapped_and_a_fake_never_is(self):
        import importlib
        import tempfile
        import types
        tmp = tempfile.mkdtemp()
        modname = "_hg_effect_probe_mod"
        with open(os.path.join(tmp, modname + ".py"), "w", encoding="utf-8") as fh:
            fh.write("class Util:\n"
                     "    @staticmethod\n"
                     "    def Speakers():\n"
                     "        return 'real'\n")
        targets = hg._EFFECT_TARGETS + ((modname, "Util", "Speakers"),)
        sys.path.insert(0, tmp)
        try:
            with mock.patch.object(hg, "_EFFECT_TARGETS", targets), \
                    mock.patch.object(hg, "_EFFECT_MODULES",
                                      frozenset(t[0] for t in targets)):
                fake = types.ModuleType(modname)
                fake.Util = type("Util", (), {"Speakers": staticmethod(lambda: "fake")})
                with mock.patch.dict(sys.modules, {modname: fake}):
                    mod = importlib.import_module(modname)
                    self.assertIs(mod, fake)
                    self.assertFalse(hg._marked(mod.Util.Speakers))
                sys.modules.pop(modname, None)
                mod = importlib.import_module(modname)
                self.assertTrue(hg._marked(mod.Util.Speakers),
                                "the real module was not wrapped on import")
                with _ledger_restored(), hg.allow("input"):
                    self.assertEqual(mod.Util.Speakers(), "real")
                if "input" not in hg.unarmed_guards():
                    with _ledger_restored(), self.assertRaises(hg.InputGuardError):
                        mod.Util.Speakers()
        finally:
            sys.path.remove(tmp)
            sys.modules.pop(modname, None)
            hg._effect_modules.pop(modname, None)

    def test_real_pycaw_get_speakers_is_wrapped_in_a_fresh_process(self):
        # Proven in a child process: importing the real pycaw here would
        # CoInitialize this thread. The child never reaches COM - the stub
        # refuses first.
        import importlib.util
        if sys.platform != "win32" or importlib.util.find_spec("pycaw") is None:
            self.skipTest("pycaw is not installed here")
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from tools import hermetic_guard as hg\n"
            "hg.install(quiet=True)\n"
            "from pycaw.pycaw import AudioUtilities\n"
            "assert hg._marked(AudioUtilities.GetSpeakers), 'not wrapped'\n"
            "try:\n"
            "    AudioUtilities.GetSpeakers()\n"
            "except hg.InputGuardError:\n"
            "    hg.reset(); print('REFUSED')\n"
        ) % _PROJECT_ROOT
        env = {k: v for k, v in os.environ.items()
               if k not in hg.ENV_ESCAPES.values()}
        out = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        self.assertIn("REFUSED", out.stdout)


# ─── the screen, 2026-10-02 ────────────────────────────────────────────────
# A tripwire run of the monolith tier caught test_space_fires_when_enter_did_
# not_start photographing the owner's whole desktop twice per run (mss, then
# the PIL.ImageGrab fallback). Each test below that calls a REAL capture entry
# point first spies on every pixel source beneath it, so a broken guard fails
# the test instead of photographing anything.

def _need(module: str) -> None:
    import importlib.util
    if importlib.util.find_spec(module.split(".")[0]) is None:
        raise unittest.SkipTest(f"{module} is not installed here")


@contextlib.contextmanager
def _imagegrab_pixel_spies():
    """Every way PIL.ImageGrab.grab reads pixels, replaced by a spy: the C
    grabs (Windows / X11) and the subprocess it shells out to (macOS
    screencapture, the Linux gnome-screenshot fallback). Yields the spies."""
    from PIL import Image, ImageGrab
    with contextlib.ExitStack() as stack:
        spies = [stack.enter_context(mock.patch.object(
                     Image.core, name,
                     side_effect=AssertionError("reached the pixels")))
                 for name in ("grabscreen_win32", "grabscreen_x11")
                 if hasattr(Image.core, name)]
        spies.append(stack.enter_context(
            mock.patch.object(ImageGrab, "subprocess")))
        yield spies


class ScreenGuardTests(_Armed):

    def test_the_capture_entry_points_are_targets(self):
        self.assertIn("screen", hg.GUARDS)
        self.assertIn(("mss.base", "MSS", "grab"), hg._SCREEN_TARGETS)
        self.assertIn(("PIL.ImageGrab", None, "grab"), hg._SCREEN_TARGETS)

    def test_a_real_imagegrab_grab_is_refused_before_any_pixel_is_read(self):
        _need("PIL")
        self.require_armed("screen")
        from PIL import ImageGrab
        self.assertTrue(hg._marked(ImageGrab.grab))
        with _ledger_restored(), _imagegrab_pixel_spies() as spies:
            with self.assertRaises(hg.ScreenGuardError):
                ImageGrab.grab(all_screens=True)
            last = hg.refusals("screen")[-1]
            self.assertEqual(last.api, "PIL.ImageGrab.grab")
            self.assertIn("test_a_real_imagegrab_grab_is_refused", last.test_id)
        for spy in spies:
            self.assertEqual(spy.mock_calls, [])

    def test_a_real_mss_grab_is_refused_before_any_pixel_is_read(self):
        # The REAL MSS.grab, on a fake instance whose backend is a spy: no
        # MSS() is ever made, so nothing below grab() exists to reach.
        _need("mss")
        self.require_armed("screen")
        import mss.base
        self.assertTrue(hg._marked(mss.base.MSS.grab))
        fake = mock.MagicMock()
        fake._impl.grab.side_effect = AssertionError("reached the pixels")
        fake._grab_impl.side_effect = AssertionError("reached the pixels")
        with _ledger_restored():
            with self.assertRaises(hg.ScreenGuardError):
                mss.base.MSS.grab(fake, {"left": 0, "top": 0,
                                         "width": 1, "height": 1})
            self.assertEqual(hg.refusals("screen")[-1].api,
                             "mss.base.MSS.grab")
        fake._impl.grab.assert_not_called()
        fake._grab_impl.assert_not_called()

    def test_a_patched_capture_entry_point_reports_unarmed_and_is_restored(self):
        _need("PIL")
        from PIL import ImageGrab
        with mock.patch.object(ImageGrab, "grab"):
            self.assertIn("PIL.ImageGrab.grab", hg.unwrapped_effects("screen"))
            self.assertIn("screen", hg.unarmed_guards())
            self.assertNotIn("PIL.ImageGrab.grab",
                             hg.unwrapped_effects("input"))
        self.assertEqual(hg.unwrapped_effects("screen"), [],
                         "mock.patch must restore the guard's stub")
        self.assertNotIn("screen", hg.unarmed_guards())

    def test_an_opt_in_lets_a_capture_through(self):
        _need("PIL")
        from PIL import ImageGrab
        with _ledger_restored(), _imagegrab_pixel_spies(), \
                hg.allow("screen", reason="the spy stands in for the screen"):
            with self.assertRaises(AssertionError):   # it reached the spy
                ImageGrab.grab()
            self.assertEqual(hg.refusals("screen"), ())

    def test_a_synthetic_screen_blit_is_refused_and_a_fill_passes(self):
        blit = [p for p, fn in hg._screen_fn_ptrs.items() if fn[1] == "BitBlt"]
        if not blit:
            self.skipTest("no gdi32 on this host")
        self.require_armed("screen")
        with _ledger_restored():
            with self.assertRaises(hg.ScreenGuardError):
                sys.audit("ctypes.call_function", blit[0],
                          (0, 0, 0, 1, 1, 1, 0, 0, _SRCCOPY))
            self.assertEqual(hg.refusals("screen")[-1].api, "gdi32.BitBlt")
            sys.audit("ctypes.call_function", blit[0],
                      (0, 0, 0, 1, 1, None, 0, 0, _WHITENESS))

    def test_a_real_screen_blit_is_refused_and_ordinary_drawing_passes(self):
        if sys.platform != "win32" or not hg._screen_fn_ptrs:
            self.skipTest("no gdi32 on this host")
        self.require_armed("screen")
        import ctypes
        from ctypes import wintypes
        user32, gdi32 = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
        user32.GetDC.argtypes = [wintypes.HWND]
        user32.GetDC.restype = wintypes.HDC
        user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        user32.GetDesktopWindow.restype = wintypes.HWND
        user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC,
                                       wintypes.UINT]
        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int,
                                                 ctypes.c_int]
        gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
        gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi32.SelectObject.restype = wintypes.HGDIOBJ
        gdi32.BitBlt.argtypes = [wintypes.HDC] + [ctypes.c_int] * 4 + [
            wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
        gdi32.GetPixel.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
        gdi32.GetPixel.restype = wintypes.DWORD
        gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        screen = user32.GetDC(None)          # a handle; reads no pixels
        dcs, bitmaps = [], []
        try:
            for _ in range(2):
                dc = gdi32.CreateCompatibleDC(screen)
                bmp = gdi32.CreateCompatibleBitmap(screen, 1, 1)
                gdi32.SelectObject(dc, bmp)
                dcs.append(dc)
                bitmaps.append(bmp)
            mem, other = dcs
            # Ordinary drawing passes: a source-less fill, then a
            # memory-to-memory copy of it.
            self.assertTrue(gdi32.BitBlt(mem, 0, 0, 1, 1, None, 0, 0,
                                         _WHITENESS))
            self.assertTrue(gdi32.BitBlt(other, 0, 0, 1, 1, mem, 0, 0,
                                         _SRCCOPY))
            self.assertEqual(gdi32.GetPixel(other, 0, 0), 0xFFFFFF)
            with _ledger_restored():
                with self.assertRaises(hg.ScreenGuardError):
                    gdi32.BitBlt(mem, 0, 0, 1, 1, screen, 0, 0, _SRCCOPY)
                with self.assertRaises(hg.ScreenGuardError):
                    user32.PrintWindow(user32.GetDesktopWindow(), mem, 0)
                self.assertEqual([r.api for r in hg.refusals("screen")[-2:]],
                                 ["gdi32.BitBlt", "user32.PrintWindow"])
        finally:
            for dc in dcs:
                gdi32.DeleteDC(dc)
            for bmp in bitmaps:
                gdi32.DeleteObject(bmp)
            user32.ReleaseDC(None, screen)

    def test_the_tests_package_arms_it_in_a_fresh_process(self):
        # End to end through the chokepoint: a child that only imports the
        # tests package must refuse a real capture, and say so at exit. Every
        # pixel source is a spy in the child too.
        _need("PIL")
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import contextlib\n"
            "import tests\n"
            "from tools import hermetic_guard as hg\n"
            "from tests.test_hermetic_guard import _imagegrab_pixel_spies\n"
            "assert 'screen' not in hg.unarmed_guards(), hg.unarmed_guards()\n"
            "from PIL import ImageGrab\n"
            "assert hg._marked(ImageGrab.grab), 'ImageGrab.grab not wrapped'\n"
            "with _imagegrab_pixel_spies() as spies:\n"
            "    try:\n"
            "        ImageGrab.grab()\n"
            "    except hg.ScreenGuardError:\n"
            "        print('REFUSED')\n"
            "assert all(not s.mock_calls for s in spies), 'reached the pixels'\n"
            "print('SUMMARY ' + hg.summary().splitlines()[0])\n"
            "hg.reset()\n"
        ) % _PROJECT_ROOT
        env = {k: v for k, v in os.environ.items()
               if k not in hg.ENV_ESCAPES.values()}
        out = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                             cwd=_PROJECT_ROOT, capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        self.assertIn("REFUSED", out.stdout)
        self.assertIn("SUMMARY [screen-guard] refused 1 real screen reach",
                      out.stdout)


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
    # (2026-10-02, the tripwire run of the monolith tier)
    ("tests.monolith.test_monolith_sec4.AppleMusicAutoPlayNoVisionTests."
     "test_space_fires_when_enter_did_not_start",
     "the real screen - two whole-desktop captures (mss, PIL.ImageGrab)"),
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


# ─── a "monolith absent" test never imports the real monolith, 2026-10-02 ──
# ~14 test files simulate "the monolith lookup fails" with
# ``inject_modules(bobert_companion=None)``. Most of their helpers did that
# with ``sys.modules.pop(name)``. A LOOKUP (sys.modules.get) then misses, as
# intended. An IMPORT (importlib.import_module / ``import bobert_companion``)
# does not: it searches sys.path and finds the REAL 43k-line
# bobert_companion.py. On the light-deps Linux CI that import fails, so the
# test passed by accident. On the owner's PC it imports and touches real
# hardware: test_suit_up got his real headset name instead of "system
# default". Those helpers now pin the import system's absent sentinel
# (``sys.modules[name] = None``), so the import raises ModuleNotFoundError.
#
# The trap goes first on sys.meta_path. Any attempt to FIND bobert_companion
# on disk while it is armed is recorded and refused, so the real module never
# executes even if a test regresses. Every test in tests/ that injects
# ``bobert_companion=None`` is found from the SOURCE (a new site is covered
# the day it is written) and re-run under the trap; none may make an attempt
# from its test method. On CI, where the monolith cannot import anyway, the
# attempt is still seen: the trap refuses it before the import would have
# failed on a missing dep.

_MONOLITH = "bobert_companion"
_TESTS_DIR = os.path.join(_PROJECT_ROOT, "tests")
_ABSENT = object()

# The sites that imported the real monolith before the fix (all four expected
# a fallback and got whatever the owner's PC answered).
_MONOLITH_ABSENT_FIXED = (
    "tests.skills.test_suit_up.SuitUpBuilderTests."
    "test_resolve_speaker_blank_explicit_falls_back",
    "tests.skills.test_suit_up.SuitUpBuilderTests."
    "test_resolve_speaker_whitespace_explicit_falls_back",
    "tests.skills.test_dossier.DossierRegisterTests."
    "test_register_wires_all_aliases",
    "tests.skills.test_weekly_digest_briefing.WeeklyDigestBriefingTests."
    "test_read_config_no_bc_uses_defaults",
)


class _MonolithImportTrap:
    """A sys.meta_path finder that finds nothing. For bobert_companion it
    records where the import came from and refuses it, so a regressed test
    can never execute the real monolith.

    With ``body_code`` set (a test method's code object), only an attempt made
    while that method is on the stack lands in ``attempts``; one made by a
    fixture (setUp -> load_skill_isolated -> a skill's register()) is refused
    all the same but lands in ``fixture_attempts``. Those are a different
    leak from the one this class is about, so they are not judged here."""

    def __init__(self):
        self.attempts: list[str] = []
        self.fixture_attempts: list[str] = []
        self.body_code = None

    def _in_body(self) -> bool:
        if self.body_code is None:
            return True
        frame = sys._getframe(2)
        while frame is not None:
            if frame.f_code is self.body_code:
                return True
            frame = frame.f_back
        return False

    def find_spec(self, name, path=None, target=None):
        if name != _MONOLITH:
            return None
        site = "?"
        for frame in reversed(traceback.extract_stack()[:-1]):
            fname = frame.filename
            if "importlib" in fname or fname.startswith("<frozen"):
                continue
            site = f"{os.path.basename(fname)}:{frame.lineno}"
            break
        (self.attempts if self._in_body() else self.fixture_attempts).append(site)
        raise ModuleNotFoundError(
            f"[monolith-trap] refused to import the real {name} from disk "
            f"(from {site})", name=name)

    @contextlib.contextmanager
    def armed(self):
        sys.meta_path.insert(0, self)
        try:
            yield self
        finally:
            with contextlib.suppress(ValueError):
                sys.meta_path.remove(self)


def _injects_monolith_none(func) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (kw.arg == _MONOLITH and isinstance(kw.value, ast.Constant)
                        and kw.value.value is None):
                    return True
    return False


def _monolith_absent_tests() -> list[str]:
    """The id of every test method under tests/ that passes
    ``bobert_companion=None`` to a call (the inject_modules helpers)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(_TESTS_DIR):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fname in sorted(filenames):
            if not (fname.startswith("test_") and fname.endswith(".py")):
                continue
            path = os.path.join(dirpath, fname)
            src = _source(path)
            if _MONOLITH not in src:
                continue
            module = os.path.relpath(path, _PROJECT_ROOT)[:-3].replace(os.sep, ".")
            for cls in ast.parse(src, path).body:
                if not isinstance(cls, ast.ClassDef):
                    continue
                for func in cls.body:
                    if (isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and func.name.startswith("test")
                            and _injects_monolith_none(func)):
                        found.append(f"{module}.{cls.name}.{func.name}")
    return found


def _test_cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _test_cases(item)
        else:
            yield item


def _method_code(case):
    """The code object of a test case's own method, under any decorators
    (``mock.patch`` keeps ``__wrapped__``); None when it can't be found."""
    func = getattr(type(case), getattr(case, "_testMethodName", ""), None)
    try:
        func = inspect.unwrap(func) if func is not None else None
    except ValueError:
        return None
    return getattr(func, "__code__", None)


@contextlib.contextmanager
def _monolith_popped():
    saved = sys.modules.pop(_MONOLITH, _ABSENT)
    try:
        yield
    finally:
        if saved is _ABSENT:
            sys.modules.pop(_MONOLITH, None)
        else:
            sys.modules[_MONOLITH] = saved


class MonolithAbsentTestsNeverImportItTests(unittest.TestCase):

    def test_the_trap_sees_a_pop_then_import_but_never_the_sentinel(self):
        # The old helper semantics (pop, then the code under test imports)
        # must be caught; the fixed semantics (the None sentinel) must not
        # even reach a finder. Nothing here can execute the real monolith:
        # the trap refuses the only two imports that reach a finder.
        trap = _MonolithImportTrap()
        with _monolith_popped(), trap.armed():
            with self.assertRaises(ImportError):
                importlib.import_module(_MONOLITH)
            with self.assertRaises(ImportError):
                __import__(_MONOLITH)
            # (the recorded site is diagnostic only: under ci_sim the first
            # frame outside importlib is its import_module shim, not this file)
            self.assertEqual(len(trap.attempts), 2)
            sys.modules[_MONOLITH] = None
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(_MONOLITH)
            self.assertEqual(len(trap.attempts), 2)
        self.assertNotIn(trap, sys.meta_path)

    def test_the_trap_tells_the_test_body_from_a_fixture(self):
        def body():
            importlib.import_module(_MONOLITH)

        def fixture():
            importlib.import_module(_MONOLITH)
        trap = _MonolithImportTrap()
        trap.body_code = body.__code__
        with _monolith_popped(), trap.armed():
            for fn in (fixture, body):
                with self.assertRaises(ImportError):
                    fn()
        self.assertEqual(len(trap.fixture_attempts), 1)
        self.assertEqual(len(trap.attempts), 1)

    def test_discovery_finds_the_sites_and_the_fixed_ones(self):
        found = _monolith_absent_tests()
        self.assertGreaterEqual(len(found), 20, found)
        self.assertEqual(len(found), len(set(found)))
        for test_id in _MONOLITH_ABSENT_FIXED:
            self.assertIn(test_id, found)

    def test_no_monolith_absent_test_imports_the_real_monolith(self):
        loader = unittest.TestLoader()
        trap = _MonolithImportTrap()
        for test_id in _monolith_absent_tests():
            with self.subTest(test=test_id):
                suite = loader.loadTestsFromName(test_id)
                cases = list(_test_cases(suite))
                self.assertEqual(len(cases), 1, test_id)
                trap.body_code = _method_code(cases[0])
                self.assertIsNotNone(trap.body_code, test_id)
                result = unittest.TestResult()
                with _ledger_restored(), trap.armed():
                    n0 = len(trap.attempts)
                    suite.run(result)
                    sites = trap.attempts[n0:]
                self.assertEqual(
                    [(t.id(), tb.splitlines()[-1]) for t, tb in
                     result.errors + result.failures], [], test_id)
                self.assertEqual(
                    sites, [],
                    f"{test_id} imported the REAL bobert_companion from disk "
                    f"(from {sites}). Make its helper pin the absent sentinel "
                    f"(sys.modules[name] = None) or inject a stub module.")


# ─── a staging monolith left loaded by the monolith suite, 2026-10-02 ──────
# tests/_monolith_harness.load_monolith() imports the REAL monolith with
# JARVIS_STAGING=1 and leaves both behind for the rest of the run. That is
# deliberate: several monolith gates read the env var at call time, and the
# monolith tests rely on it being set suite-wide. The module that stays in
# sys.modules carries the staging posture fixed at its import: BLUE_GREEN_ROLE
# "staging", MICROPHONE_INDEX -1. Skill code that looks the monolith up then
# finds that staging instance. On the owner's PC, seven skills/web_interface
# tests answered "Not while I'm in staging" and four audio_switch tests read
# "index -1". CI never loads the monolith, so CI never saw it. Restoring the
# env var cannot help, because the role was fixed when the module imported.
# Both files now pin the absent sentinel for their whole module. This re-runs
# them with a staging stand-in planted the way the monolith suite leaves it,
# and requires the same outcome as with nothing planted. The import trap is
# armed for both runs, so neither run can load the real monolith even if a
# pin regresses.

_STAGING_ORDER_SENSITIVE = ("tests.skills.test_web_interface",
                            "tests.skills.test_audio_switch")


def _staging_monolith_standin():
    """What the harness-loaded monolith looks like to a skill afterwards."""
    bc = types.ModuleType(_MONOLITH)
    bc.BLUE_GREEN_ROLE = "staging"
    bc._is_staging = lambda: True
    bc.MICROPHONE_INDEX = -1
    bc.PREFERRED_INPUT_DEVICES = []
    return bc


def _module_outcome(module_name):
    result = unittest.TestResult()
    unittest.TestLoader().loadTestsFromName(module_name).run(result)
    return (result.testsRun,
            sorted(t.id() for t, _tb in result.errors + result.failures))


class StagingMonolithLeftLoadedTests(unittest.TestCase):

    def test_outcomes_do_not_depend_on_a_loaded_staging_monolith(self):
        for name in _STAGING_ORDER_SENSITIVE:
            with self.subTest(module=name):
                trap = _MonolithImportTrap()
                with _ledger_restored(), _monolith_popped(), trap.armed():
                    clean = _module_outcome(name)
                with _ledger_restored(), _monolith_popped(), trap.armed(), \
                        mock.patch.dict(os.environ, {"JARVIS_STAGING": "1"}):
                    sys.modules[_MONOLITH] = _staging_monolith_standin()
                    planted = _module_outcome(name)
                self.assertGreater(clean[0], 0, name)
                self.assertEqual(
                    planted, clean,
                    f"{name}: these tests pass or fail depending on whether "
                    f"the monolith suite ran first. Pin the absent sentinel "
                    f"for the module (sys.modules['bobert_companion'] = None "
                    f"in setUpModule).")
                self.assertEqual(trap.attempts, [], name)


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
