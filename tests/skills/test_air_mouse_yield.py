"""B057 - skills/_air_mouse_yield: the air-mouse auto-yield LL hook.

The WH_MOUSE_LL / WH_KEYBOARD_LL hook had never once installed on 64-bit
Python. GetModuleHandleW ran with ctypes' default c_int restype, which truncated
the 64-bit HMODULE, so SetWindowsHookExW got a bogus hMod and returned NULL on
every boot (47/47 session logs). The watcher always fell back to
GetLastInputInfo polling, which cannot tell injected input from the owner's.
CallNextHookEx (untyped, a 64-bit lParam) and the HOOKPROC (c_long LRESULT) were
the next two failures waiting behind the first.

These tests never install a real hook. SetWindowsHookExW, GetMessageW and
UnhookWindowsHookEx are fakes. Only read-only calls run for real:
GetModuleHandleW, GetModuleFileNameW, and CallNextHookEx made outside any hook
chain, which returns 0.

  * _hook_thread hands SetWindowsHookExW the FULL module handle, through private
    typed WinDLL handles, and never touches the process-wide ctypes.windll;
  * both hooks or neither: a partial install is undone, the warning carries the
    Win32 error code, and the pump never runs;
  * the pump ending removes the hooks;
  * the hook procs stamp real vs injected input, chain CallNextHookEx, and
    never raise;
  * the health check: a live hook stays authoritative, a blind one (removed by
    LowLevelHooksTimeout) falls back to polling, and the next callback heals it;
  * GetTickCount arithmetic survives 24.8 days of uptime (signed c_int) and the
    49.7-day wrap.

stdlib unittest + mock. The Windows-only parts skip on the Linux CI runner.
"""
from __future__ import annotations

import ctypes
import importlib
import io
import sys
import time
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

try:
    from ctypes import wintypes
except Exception:   # Linux CI / CI-sim: wintypes is Windows-only there
    wintypes = None

_HAS_WINTYPES = wintypes is not None
_WIN = (sys.platform.startswith("win") and hasattr(ctypes, "WinDLL")
        and _HAS_WINTYPES)

_MOUSE_FLAGS_OFFSET = 12   # MSLLHOOKSTRUCT: POINT pt (8) + DWORD mouseData (4)
_KBD_FLAGS_OFFSET = 8      # KBDLLHOOKSTRUCT: DWORD vkCode (4) + DWORD scanCode (4)


def _yield_mod():
    return importlib.import_module("skills._air_mouse_yield")


def _reset(y):
    """Clear the shared watcher state so tests can't bleed into each other (or
    into the other yield tests in the suite)."""
    y.note_real_input_for_test(float("-inf"))
    with y._install_lock:
        y._installed = False
        y._hook_ok = False
    with y._lock:
        y._last_self_action = float("-inf")
        y._last_self_action_wall = float("-inf")
        y._last_hook_event = float("-inf")
    y._warned[0] = False
    if hasattr(y, "_stale_warned"):
        y._stale_warned[0] = False
    y._mouse_cb = None
    y._kbd_cb = None


class _FakeFn:
    """Stands in for a ctypes foreign function: takes argtypes/restype like
    the real one and records each call with the declarations in force then."""

    def __init__(self, impl):
        self.argtypes = None
        self.restype = ctypes.c_int
        self.impl = impl
        self.calls = []

    def __call__(self, *args):
        self.calls.append((args, self.argtypes, self.restype))
        return self.impl(*args)


class _FakeUser32:
    """user32 with the hook-install calls faked, so no hook is ever installed.
    hook_results: one (handle, last_error) per SetWindowsHookExW call."""

    def __init__(self, y, hook_results):
        results = list(hook_results)
        self.hook_ok_while_pumping = []

        def _set_hook(id_hook, proc, h_mod, thread_id):
            handle, err = results.pop(0)
            ctypes.set_last_error(err)
            return handle

        def _get_message(*_args):
            self.hook_ok_while_pumping.append(y._hook_ok)
            return 0                      # WM_QUIT: the pump exits at once

        self.SetWindowsHookExW = _FakeFn(_set_hook)
        self.GetMessageW = _FakeFn(_get_message)
        self.TranslateMessage = _FakeFn(lambda *_a: 0)
        self.DispatchMessageW = _FakeFn(lambda *_a: 0)
        self.UnhookWindowsHookEx = _FakeFn(lambda _h: 1)
        self.CallNextHookEx = _FakeFn(lambda *_a: 0)


class _WindllSpy:
    """Replaces the process-wide ctypes.windll and records every library looked
    up on it. It serves working objects, so code that still uses windll runs to
    completion and fails on what it did, not on the spy."""

    def __init__(self, libs):
        self._libs = libs
        self.touched = []

    def __getattr__(self, name):
        self.touched.append(name)
        return self._libs[name]


def _typed_kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    k.GetModuleHandleW.restype = wintypes.HMODULE
    k.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR,
                                     wintypes.DWORD]
    k.GetModuleFileNameW.restype = wintypes.DWORD
    return k


@unittest.skipUnless(_WIN, "Windows ctypes (WinDLL / wintypes) only")
class HookInstallTests(unittest.TestCase):
    """_hook_thread run synchronously against a fake user32. The REAL kernel32
    is used, so whatever restype the code declares on GetModuleHandleW decides
    whether the handle reaches SetWindowsHookExW whole or truncated."""

    def setUp(self):
        self.y = _yield_mod()
        _reset(self.y)
        self.addCleanup(_reset, self.y)

    def _run_hook_thread(self, fake_user32):
        real_windll_cls = ctypes.WinDLL

        def _win_dll(name, *args, **kwargs):
            if name == "user32":
                return fake_user32
            return real_windll_cls(name, *args, **kwargs)

        # What the old code got from ctypes.windll: a fresh, UNTYPED kernel32
        # (fresh, so another test's declarations can't mask the default c_int).
        spy = _WindllSpy({"user32": fake_user32,
                          "kernel32": real_windll_cls("kernel32")})
        out = io.StringIO()
        with mock.patch.object(ctypes, "WinDLL", _win_dll), \
                mock.patch.object(ctypes, "windll", spy), redirect_stdout(out):
            self.y._hook_thread()
        return spy, out.getvalue()

    def test_full_module_handle_and_typed_signatures_reach_setwindowshookex(self):
        fake = _FakeUser32(self.y, [(0x1111, 0), (0x2222, 0)])
        spy, _out = self._run_hook_thread(fake)
        expected_hmod = _typed_kernel32().GetModuleHandleW(None)

        calls = fake.SetWindowsHookExW.calls
        self.assertEqual([c[0][0] for c in calls], [14, 13])   # MOUSE_LL, KEYBOARD_LL
        for args, argtypes, restype in calls:
            # The headline bug: the 64-bit HMODULE arrived truncated to 32 bits.
            self.assertEqual(args[2], expected_hmod)
            self.assertEqual(args[3], 0)                         # all threads
            self.assertIsNotNone(argtypes, "SetWindowsHookExW called untyped")
            self.assertIs(argtypes[0], ctypes.c_int)
            self.assertIs(argtypes[1], type(args[1]))            # the HOOKPROC type
            self.assertIs(argtypes[2], wintypes.HINSTANCE)
            self.assertIs(argtypes[3], wintypes.DWORD)
            self.assertIs(restype, wintypes.HHOOK)
            # HOOKPROC returns a pointer-sized LRESULT, not a 32-bit c_long.
            self.assertIs(type(args[1])._restype_, ctypes.c_ssize_t)
        self.assertEqual(fake.CallNextHookEx.argtypes,
                         [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM,
                          wintypes.LPARAM])
        self.assertIs(fake.CallNextHookEx.restype, ctypes.c_ssize_t)
        self.assertEqual(spy.touched, [],
                         "hook path used (and typed) the process-wide ctypes.windll")
        # Live while pumping; once the pump ends both hooks are removed and the
        # polling fallback takes over (a hook with no pump stalls all input).
        self.assertEqual(fake.hook_ok_while_pumping, [True])
        self.assertEqual(sorted(c[0][0] for c in fake.UnhookWindowsHookEx.calls),
                         [0x1111, 0x2222])
        self.assertFalse(self.y._hook_ok)

    def test_partial_install_is_undone_and_warns_with_the_error_code(self):
        fake = _FakeUser32(self.y, [(0x1111, 0), (0, 1428)])   # keyboard fails
        _spy, out = self._run_hook_thread(fake)
        self.assertIn("SetWindowsHookEx returned NULL", out)
        self.assertIn("keyboard err 1428", out)
        self.assertEqual(fake.GetMessageW.calls, [], "pumped a half-installed hook")
        self.assertEqual([c[0][0] for c in fake.UnhookWindowsHookEx.calls], [0x1111])
        self.assertFalse(self.y._hook_ok)


@unittest.skipUnless(_WIN, "Windows ctypes (WinDLL / wintypes) only")
class PrivateTypedApiTests(unittest.TestCase):
    """_win32_hook_api against the real DLLs: read-only calls, no hook."""

    _NAMES = (("user32", "SetWindowsHookExW"), ("user32", "CallNextHookEx"),
              ("user32", "UnhookWindowsHookEx"), ("user32", "GetMessageW"),
              ("user32", "DispatchMessageW"), ("kernel32", "GetModuleHandleW"))

    def _shared_decls(self):
        out = {}
        for lib, fn in self._NAMES:
            f = getattr(getattr(ctypes.windll, lib), fn)
            out[(lib, fn)] = (f.argtypes, f.restype)
        return out

    def test_types_are_64_bit_correct_and_private(self):
        y = _yield_mod()
        before = self._shared_decls()
        api = y._win32_hook_api()

        self.assertIs(api.LRESULT, ctypes.c_ssize_t)
        self.assertIs(api.HOOKPROC._restype_, ctypes.c_ssize_t)
        # The FULL module handle: GetModuleFileNameW resolves it. The old
        # truncated one failed with ERROR_MOD_NOT_FOUND (126).
        h_mod = api.kernel32.GetModuleHandleW(None)
        buf = ctypes.create_unicode_buffer(1024)
        self.assertGreater(_typed_kernel32().GetModuleFileNameW(h_mod, buf, 1024), 0)
        # An lParam pointer above 4 GB: untyped, this raised "int too long to
        # convert" on every hooked event. Outside a hook chain it returns 0.
        self.assertEqual(api.user32.CallNextHookEx(None, -1, 0, 0x7FFF_DEAD_0000), 0)
        # Nothing was declared on the shared, process-wide function objects.
        self.assertIsNot(api.user32, ctypes.windll.user32)
        self.assertEqual(self._shared_decls(), before)


@unittest.skipUnless(_HAS_WINTYPES, "ctypes.wintypes is Windows-only")
class HookProcTests(unittest.TestCase):
    """The LowLevel*Proc bodies, fed a hand-built event (flags at the
    documented Win32 offsets) and a fake CallNextHookEx."""

    def setUp(self):
        self.y = _yield_mod()
        _reset(self.y)
        self.addCleanup(_reset, self.y)
        self._bufs = []

    def _procs(self, call_next_impl=lambda *_a: 0):
        call_next = _FakeFn(call_next_impl)
        api = types.SimpleNamespace(
            ctypes=ctypes, wintypes=wintypes,
            user32=types.SimpleNamespace(CallNextHookEx=call_next))
        mouse_proc, kbd_proc = self.y._build_hook_procs(api)
        return mouse_proc, kbd_proc, call_next

    def _event(self, flags_offset, flags):
        buf = (ctypes.c_ubyte * 64)()
        ctypes.c_uint32.from_buffer(buf, flags_offset).value = flags
        self._bufs.append(buf)                          # keep it alive
        return ctypes.addressof(buf)

    def test_real_mouse_event_is_real_input_and_chains(self):
        mouse_proc, _kbd, call_next = self._procs(lambda *_a: 0)
        lparam = self._event(_MOUSE_FLAGS_OFFSET, 0x0)
        self.assertEqual(mouse_proc(0, 0x0200, lparam), 0)
        self.assertTrue(self.y.real_input_recent(1.5))
        self.assertEqual(call_next.calls[0][0], (None, 0, 0x0200, lparam))

    def test_injected_mouse_event_proves_alive_but_is_not_real(self):
        mouse_proc, _kbd, _cn = self._procs()
        mouse_proc(0, 0x0201, self._event(_MOUSE_FLAGS_OFFSET, 0x1))  # LLMHF_INJECTED
        self.assertFalse(self.y.real_input_recent(1.5))
        self.assertGreater(self.y._last_hook_event, time.monotonic() - 5.0)

    def test_keyboard_injected_bit_is_0x10(self):
        _mouse, kbd_proc, _cn = self._procs()
        kbd_proc(0, 0x0100, self._event(_KBD_FLAGS_OFFSET, 0x10))       # LLKHF_INJECTED
        self.assertFalse(self.y.real_input_recent(1.5))
        kbd_proc(0, 0x0100, self._event(_KBD_FLAGS_OFFSET, 0x0))
        self.assertTrue(self.y.real_input_recent(1.5))

    def test_returns_the_next_hooks_result(self):
        mouse_proc, kbd_proc, _cn = self._procs(lambda *_a: 7)
        self.assertEqual(mouse_proc(0, 0x0200, self._event(_MOUSE_FLAGS_OFFSET, 0)), 7)
        self.assertEqual(kbd_proc(0, 0x0100, self._event(_KBD_FLAGS_OFFSET, 0)), 7)

    def test_never_raises_when_callnexthookex_does(self):
        def _boom(*_a):
            raise ctypes.ArgumentError("argument 4: int too long to convert")
        mouse_proc, kbd_proc, _cn = self._procs(_boom)
        # 0 = let the input through; a raise would print a traceback per event.
        self.assertEqual(mouse_proc(0, 0x0200, self._event(_MOUSE_FLAGS_OFFSET, 0)), 0)
        self.assertEqual(kbd_proc(0, 0x0100, self._event(_KBD_FLAGS_OFFSET, 0)), 0)
        self.assertTrue(self.y.real_input_recent(1.5))   # still recorded first

    def test_negative_ncode_is_passed_on_untouched(self):
        mouse_proc, _kbd, call_next = self._procs()
        lparam = self._event(_MOUSE_FLAGS_OFFSET, 0x0)
        mouse_proc(-1, 0x0200, lparam)
        self.assertFalse(self.y.real_input_recent(1.5))
        self.assertEqual(self.y._last_hook_event, float("-inf"))
        self.assertEqual(call_next.calls[0][0], (None, -1, 0x0200, lparam))


class TickArithmeticTests(unittest.TestCase):
    """_tick_age_s: GetTickCount comes back as a signed c_int and wraps."""

    def test_modular_tick_age(self):
        y = _yield_mod()
        self.assertEqual(y._tick_age_s(10_000, 9_000), 1.0)
        # 24.8+ days of uptime: GetTickCount reads negative through c_int.
        t = 2**31 + 5_000
        self.assertEqual(y._tick_age_s(t - 2**32, t - 60_000), 60.0)
        # 49.7-day wrap: the newer tick is numerically smaller.
        self.assertEqual(y._tick_age_s(3_000, 2**32 - 2_000), 5.0)
        # The last-input tick a hair AHEAD of the read (race) is 'now'.
        self.assertEqual(y._tick_age_s(1_000, 1_005), 0.0)


class HealthCheckDecisionTests(unittest.TestCase):
    """The health-check decision with the OS read stubbed out, so it also runs
    on the Linux CI runner (no wintypes, no windll)."""

    def setUp(self):
        self.y = _yield_mod()
        _reset(self.y)
        self.addCleanup(_reset, self.y)
        with self.y._install_lock:
            self.y._installed = True
            self.y._hook_ok = True

    def _os_age(self, age_s):
        return mock.patch.object(self.y, "_os_last_input_age_s",
                                 return_value=age_s)

    def test_live_hook_is_trusted_and_blind_hook_polls(self):
        with self.y._lock:
            self.y._last_hook_event = time.monotonic()
        with self._os_age(0.05):
            self.assertFalse(self.y.real_input_recent(1.5))      # hook saw it
        with self.y._lock:
            self.y._last_hook_event = time.monotonic() - 30.0
        with self._os_age(0.05), redirect_stdout(io.StringIO()):
            self.assertTrue(self.y.real_input_recent(1.5))       # hook missed it

    def test_blind_hook_still_discounts_the_air_mouse_own_action(self):
        with self.y._lock:
            self.y._last_hook_event = time.monotonic() - 30.0
        self.y.mark_self_action()       # our own click, just now
        with self._os_age(0.0), redirect_stdout(io.StringIO()):
            self.assertFalse(self.y.real_input_recent(1.5))

    def test_os_read_unavailable_means_no_fallback(self):
        with self.y._lock:
            self.y._last_hook_event = time.monotonic() - 30.0
        with self._os_age(None):
            self.assertFalse(self.y.real_input_recent(1.5))


class _FakeInputWindll:
    """ctypes.windll with just GetLastInputInfo + GetTickCount, so the OS
    'last input' age is scripted."""

    def __init__(self, tick_now, last_input_tick):
        def _get_last_input_info(ref):
            ref._obj.dwTime = last_input_tick
            return 1
        self.user32 = types.SimpleNamespace(GetLastInputInfo=_get_last_input_info)
        self.kernel32 = types.SimpleNamespace(GetTickCount=lambda: tick_now)


def _os_input(age_ms, base=50_000_000):
    """A fake windll whose OS input happened `age_ms` ago."""
    return _FakeInputWindll(base, base - age_ms)


@unittest.skipUnless(_HAS_WINTYPES, "ctypes.wintypes is Windows-only")
class FallbackAndHealthCheckTests(unittest.TestCase):

    def setUp(self):
        self.y = _yield_mod()
        _reset(self.y)
        self.addCleanup(_reset, self.y)

    def _state(self, hook_ok, last_hook_event):
        with self.y._install_lock:
            self.y._installed = True
            self.y._hook_ok = hook_ok
        with self.y._lock:
            self.y._last_hook_event = last_hook_event

    def _with_os(self, fake):
        return mock.patch.object(ctypes, "windll", fake, create=True)

    def test_fallback_after_24_8_days_uptime_does_not_report_input_forever(self):
        self._state(hook_ok=False, last_hook_event=float("-inf"))
        t = 2**31 + 5_000                         # uptime past 24.8 days
        with self._with_os(_FakeInputWindll(t - 2**32, t - 60_000)):
            self.assertFalse(self.y.real_input_recent(1.5))
            self.assertAlmostEqual(self.y.seconds_since_real_input(), 60.0, delta=1.0)

    def test_fallback_across_the_49_7_day_wrap(self):
        self._state(hook_ok=False, last_hook_event=float("-inf"))
        with self._with_os(_FakeInputWindll(3_000, 2**32 - 2_000)):     # 5 s ago
            self.assertFalse(self.y.real_input_recent(1.5))
            self.assertAlmostEqual(self.y.seconds_since_real_input(), 5.0, delta=1.0)

    def test_live_hook_stays_authoritative_over_injected_os_input(self):
        # The hook just saw that input and it was INJECTED (AFK-Helper, a
        # SendInput): the OS counter can't tell, the hook can. No yield.
        self._state(hook_ok=True, last_hook_event=time.monotonic())
        out = io.StringIO()
        with self._with_os(_os_input(50)), redirect_stdout(out):
            self.assertFalse(self.y.real_input_recent(1.5))
        self.assertEqual(out.getvalue(), "")

    def test_input_from_before_the_hook_installed_is_not_stale(self):
        self._state(hook_ok=True, last_hook_event=time.monotonic())
        with self._with_os(_os_input(10_000)):          # 10 s ago
            self.assertFalse(self.y.real_input_recent(1.5))

    def test_blind_hook_falls_back_to_polling_then_heals(self):
        # Windows silently removed the hook (LowLevelHooksTimeout): its last
        # callback was 30 s ago, but the OS saw input 50 ms ago.
        self._state(hook_ok=True, last_hook_event=time.monotonic() - 30.0)
        out = io.StringIO()
        with redirect_stdout(out):
            with self._with_os(_os_input(50)):
                self.assertTrue(self.y.real_input_recent(1.5))   # still yields
            # Fresh unreported input may just be a callback waiting for the
            # GIL: no log line yet. Unreported for 2 s proves the hook blind.
            self.assertEqual(out.getvalue(), "")
            with self._with_os(_os_input(2_000)):
                self.assertTrue(self.y.real_input_recent(3.0))
                self.y.real_input_recent(3.0)
        self.assertEqual(out.getvalue().count("LL hook missed OS input"), 1)
        # The hook reports again (an injected event): authoritative once more.
        self.y._note_hook_event(False)
        with self._with_os(_os_input(50)):
            self.assertFalse(self.y.real_input_recent(1.5))


if __name__ == "__main__":
    unittest.main()
