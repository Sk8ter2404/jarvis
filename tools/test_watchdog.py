#!/usr/bin/env python3
"""Hard TIME ceiling for a JARVIS test run — a hung test must NAME ITSELF.

THE INCIDENT (2026-09-04)
=========================
``python tools/run_tests_ci_sim.py`` was run with a 900-second outer timeout and
came back EXIT 124: it never finished. All three CI gates had already printed
OK, so the stall was inside the test run itself — and the run produced NOTHING
that said where. unittest emits its dots as it goes, but a hang leaves no
summary line, no traceback and no test id, so the only thing an exit-124 tells
you is "somewhere in 15,000 tests". ci.yml's ``timeout-minutes: 20`` is the same
shape one level up: it kills the job and tells you nothing either.

The reason nothing fired is that the ONLY bound the runners had was
tools/mem_guard.py, and that is a MEMORY ceiling — a hang is not an allocation.
This module is the TIME half of that pair, and it is deliberately built to the
same contract:

  * applied once, early, in every runner's main();
  * NEVER raises and never fails a run it cannot protect;
  * emits exactly ONE line, so any log makes it self-evident whether the run
    was time-bounded and by what;
  * idempotent — the second and later ``arm()`` calls are no-ops.

WHAT IT ACTUALLY DOES
---------------------
A daemon thread watches a single cell holding (phase label, phase start). Two
budgets run against it:

  * PER-PHASE (``JARVIS_TEST_TIMEOUT_S``, default 120 s) — one test, or the gap
    between two tests (tearDown, tearDownModule, and the import of the next test
    module all live in that gap, and any of them can hang), or the initial
    discovery/import sweep before the first test starts.
  * WHOLE-RUN (``JARVIS_TEST_TOTAL_TIMEOUT_S``, default 900 s) — the backstop
    for death by a thousand slow tests, which no per-test budget can see.

When a budget blows, the watchdog does the one thing the exit-124 could not: it
prints the phase label and elapsed time, then dumps the Python stack of EVERY
thread via faulthandler — so the hang names both the test and the exact line it
is parked on. It dumps a SECOND time after a short grace, because two identical
dumps prove "wedged" while two different ones prove "pathologically slow", and
those want different fixes. Then it hard-exits.

WHY A PROCESS KILL AND NOT AN EXCEPTION
---------------------------------------
The main thread is, by definition, blocked. A blocked thread is usually blocked
inside a C call (``Event.wait()``, ``Thread.join()``, ``socket.recv()``,
``lock.acquire()``) and CPython cannot interrupt that from outside: an async
exception injected with PyThreadState_SetAsyncExc is only noticed when the
target returns to the bytecode loop, which is exactly what is not happening.
``sys.exit`` from this thread would only end this thread. Killing the process is
the only move that actually works, so the value has to be extracted BEFORE the
kill — hence dump-then-exit rather than exit-and-hope.

WHY ``TerminateProcess`` AND NOT ``os._exit`` (2026-10-13)
----------------------------------------------------------
On Windows ``os._exit`` is CRT ``_exit`` -> ``ExitProcess``, which takes the
loader lock and runs every loaded DLL's DLL_PROCESS_DETACH before the process
may die. The heavy tier loads CTranslate2/CUDA, WinRT and COM, and any of them
can wedge that walk. The v2.0.191 release gate (started 2026-10-13 09:15:57)
tripped this watchdog, printed both dumps and "exiting 3" at 09:43:39 — and the
python.exe was still alive ~26 h later: flat CPU, 5,520 MB private, the parent
wrapper blocked on it the whole time. A ``Stop-Process`` by PID (i.e.
``TerminateProcess(h, -1)``; the wrapper logged EXIT=4294967295) ended it at
once. So the watchdog now exits the same way — ``TerminateProcess`` on its own
pseudo-handle, which skips DLL notification entirely — and keeps ``os._exit``
only as the fallback (non-Windows, or a refused kill). The monolith's
``_hard_exit`` has used this primitive since 2026-07-12 for the same reason;
it cannot be reused here because this module must import on the bare CI
runner, and its call-time ``sys.platform`` check would read the ci-sim fake.

CALIBRATION (measured on this box, 2026-09-04)
----------------------------------------------
A full ``tools/run_tests_ci_sim.py`` run: 15,361 tests in 136.4 s, and the
slowest single test in the whole suite was 5.00 s
(tests.skills.test_web_interface ... test_stop_survives_wedged_shutdown, which
is itself a deliberate 5 s timebox assertion). So the 120 s per-phase default is
24x the worst real test, and the 900 s whole-run default is ~6x the whole
measured run and comfortably inside ci.yml's 20-minute job timeout — which is
the point: this watchdog must always win the race against the opaque CI kill,
so the failure arrives as a stack dump instead of as a dead job.

``JARVIS_TEST_TIMEOUT_S=0`` (also ``off`` / ``none`` / ``unlimited``) is the
documented escape hatch, and so is ``JARVIS_TEST_TOTAL_TIMEOUT_S=0``. Anything
unparseable or negative falls back to the default — fail *closed*, protected.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import unittest

# Env overrides, and the defaults (see the module docstring for the numbers
# behind them).
PER_PHASE_ENV = "JARVIS_TEST_TIMEOUT_S"
TOTAL_ENV = "JARVIS_TEST_TOTAL_TIMEOUT_S"
DEFAULT_PER_PHASE_S = 120.0
DEFAULT_TOTAL_S = 900.0

# How long after the first dump the second one is taken, and the exit code the
# process dies with. 3 is distinct from unittest's 1 (test failure) and from the
# 124 a shell `timeout` produces, so a log tells you which bound tripped.
DUMP_GRACE_S = 5.0
EXIT_CODE = 3

# Values meaning "explicitly disabled".
_OFF = {"0", "off", "none", "unlimited", "no", "false"}

# How often the watchdog thread re-checks. Small enough that the report is
# prompt, large enough to cost nothing across a 15,000-test run.
_POLL_S = 0.5

_LABEL_DISCOVERY = "test discovery (importing test modules)"

# ── module state ────────────────────────────────────────────────────────────
# _cell holds (phase label, phase start monotonic). One list, mutated in place,
# so the watchdog thread never needs the lock to read a consistent-enough view.
_cell: list = [_LABEL_DISCOVERY, 0.0]
_state: dict = {"armed": False, "thread": None, "stop": None,
                "per_phase": 0.0, "total": 0.0, "run_start": 0.0}
_lock = threading.Lock()

# The clock and the grace sleep are bound ONCE, at import, so nothing the suite
# patches can reach them. tests/_monolith_harness.shift_monotonic moves
# time.monotonic 30 days ahead for a test, and this thread used to read the
# patched global: the 2026-10-14 gate log has "phase exceeded 120s (2.592e+06
# elapsed)" on a test that had just started. That one stood down at the
# recheck; a shifted test still running 5 s later would have been KILLED, and
# a patched time.sleep would have shrunk the grace window to nothing.
_monotonic = time.monotonic
_sleep = time.sleep


def _bind_kernel32():
    """kernel32 with TerminateProcess declared, or None off Windows.

    Called once, at import — NOT when the watchdog fires. tools/
    run_tests_ci_sim.py deletes ``ctypes.WinDLL``/``ctypes.windll`` and sets
    ``sys.platform = "linux"`` after it has armed this watchdog, so a lookup
    made at fire time would miss on the very Windows box it has to kill. A
    private WinDLL (the tools/mem_guard.py idiom), not the shared
    ``ctypes.windll.kernel32``: declaring the prototypes there would change
    them for every other caller, and a test that patches ``ctypes.windll``
    could not reach the kill either way."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # HANDLE, not the default c_int: the pseudo-handle is pointer-sized.
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetCurrentProcess.argtypes = []
        k32.TerminateProcess.restype = wintypes.BOOL
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        return k32
    except Exception:
        return None


_k32 = _bind_kernel32()
# The fallback, bound here so _hard_exit owns the only reference to it.
_os_exit = os._exit


def _hard_exit(code: int) -> None:
    """End the process unconditionally. TerminateProcess first — it never
    returns on success — then os._exit for a host without it or a kill the
    kernel refused. Never flushes a Python stream on the way out: a wedged
    main thread can be holding that stream's buffer lock, and waiting for it
    would turn the kill back into a hang."""
    k32 = _k32
    if k32 is not None:
        try:
            k32.TerminateProcess(k32.GetCurrentProcess(), code)
        except Exception:
            pass
    _os_exit(code)


def _exit_via() -> str:
    """For the 'exiting' line, so a log says which kill was attempted."""
    return "TerminateProcess" if _k32 is not None else "os._exit"


# Indirection so the unit tests can prove the fire path without killing the
# process running them. Never reassigned outside tests.
_exit = _hard_exit


def _fmt(value: float) -> str:
    """120.0 -> '120', 0.5 -> '0.5' (log lines should read like a human wrote
    them)."""
    return ("%g" % value)


def _budget(env_var: str, default: float) -> float:
    """Resolve one budget from the environment. Returns 0.0 for an explicit
    disable. Garbage or a negative number falls back to the default — the
    failure mode of a typo must be PROTECTED, not unbounded."""
    raw = (os.environ.get(env_var) or "").strip()
    if not raw:
        return default
    if raw.lower() in _OFF:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return value


def _stderr_fd() -> int:
    """A file DESCRIPTOR to report on, never a file object.

    Tests routinely patch ``builtins.print`` and replace ``sys.stderr`` with a
    StringIO; faulthandler needs a real fd and a mocked stream has none. Going
    straight to the descriptor means the dump survives whatever the test under
    the hang happens to have monkeypatched."""
    for stream in (sys.__stderr__, sys.stderr):
        try:
            fd = stream.fileno()
        except Exception:
            continue
        if isinstance(fd, int) and fd >= 0:
            return fd
    return 2


def _write(fd: int, text: str) -> None:
    """Best-effort raw write. Never raises — this runs on the failure path."""
    try:
        os.write(fd, text.encode("utf-8", "replace"))
    except Exception:
        pass


def _dump(fd: int) -> None:
    """All-thread Python stacks. Never raises."""
    try:
        faulthandler.dump_traceback(file=fd, all_threads=True)
    except Exception:
        pass


def _expired(now: float, *, label: str, started: float, per_phase: float,
             total: float, run_start: float) -> str | None:
    """PURE decision function: the reason this run must be killed, or None.

    Takes everything as arguments rather than reading the module globals so it
    can be tested directly and in isolation — a watchdog whose trigger logic is
    only exercised by actually hanging a process is a watchdog nobody ever
    checks, and a test that had to poke the LIVE cell to exercise it would be
    telling the real watchdog thread to kill the run it is running in."""
    if per_phase and started and (now - started) > per_phase:
        return ("phase exceeded %ss (%s elapsed): %s"
                % (_fmt(per_phase), _fmt(round(now - started, 1)), label))
    if total and run_start and (now - run_start) > total:
        return ("whole run exceeded %ss (%s elapsed); current phase: %s"
                % (_fmt(total), _fmt(round(now - run_start, 1)), label))
    return None


def _fire(reason: str, recheck=None) -> bool:
    """Report everything we know, then kill the process — UNLESS the run frees
    itself during the grace window.

    Returns True if it killed (so: normally never returns at all), False if it
    stood down. The stand-down is not politeness, it is correctness: the grace
    window exists to tell WEDGED from merely SLOW, and a watchdog that kills a
    run which finished during that window would turn a green suite into a red
    one. ``recheck`` re-answers "is this still the same stuck phase?" after the
    wait; only a yes is fatal."""
    fd = _stderr_fd()
    _write(fd, "\n\n" + "=" * 78 + "\n"
               "[test-watchdog] TIME CEILING BLOWN — %s\n" % reason +
               "[test-watchdog] all-thread stacks follow; the MainThread frame "
               "is where the run is stuck.\n" + "=" * 78 + "\n")
    _dump(fd)
    _write(fd, "\n[test-watchdog] second dump in %ss — identical stacks mean "
               "WEDGED, different stacks mean pathologically SLOW.\n"
               % _fmt(DUMP_GRACE_S))
    _sleep(DUMP_GRACE_S)
    if recheck is not None:
        try:
            still_stuck = bool(recheck())
        except Exception:
            still_stuck = True          # fail closed: an unanswerable recheck kills
        if not still_stuck:
            _write(fd, "\n[test-watchdog] the run moved on during the grace "
                       "window — SLOW, not wedged. NOT killing it; the stack "
                       "above is still worth reading.\n")
            return False
    _dump(fd)
    # The message is already in the OS (os.write, no Python buffer), so the
    # kill cannot lose it. "exiting N" stays a prefix for anything grepping it.
    _write(fd, "\n[test-watchdog] exiting %d via %s\n" % (EXIT_CODE, _exit_via()))
    _exit(EXIT_CODE)
    return True


def _blown(stop: threading.Event) -> str | None:
    """Read the live cell and ask _expired. Returns None — never a reason — for
    a watchdog that is not the current one, so a thread whose state was swapped
    out from under it can never kill a run it no longer belongs to."""
    if _state.get("stop") is not stop:
        return None
    return _expired(_monotonic(), label=_cell[0], started=_cell[1],
                    per_phase=_state["per_phase"], total=_state["total"],
                    run_start=_state["run_start"])


def _watch(stop: threading.Event) -> None:
    """The daemon body. Never raises out — an exception here would silently
    remove the only time bound the run has."""
    # This loop ends ONLY on its own stop event (set by disarm). A thread that
    # is no longer the current watchdog keeps looping harmlessly — _blown()
    # refuses to give it a reason — rather than exiting, because exiting on a
    # momentary state swap would silently leave the rest of the run with no
    # time ceiling at all.
    while not stop.wait(_POLL_S):
        try:
            reason = _blown(stop)
        except Exception:
            continue
        if not reason:
            continue
        # A per-phase blow-up self-clears when the test finishes (the next phase
        # resets the clock), which is exactly how the recheck tells SLOW from
        # WEDGED. A whole-run blow-up does not self-clear, and should not.
        if _fire(reason, lambda: _blown(stop) is not None):
            return
        # Stood down: keep watching — the next phase gets a fresh budget, and a
        # phase that really is wedged will trip us again.


def note(label: str) -> None:
    """Start a new phase. Cheap enough to call around every single test."""
    _cell[0] = label
    _cell[1] = _monotonic()


def arm(*, per_phase_s: float | None = None,
        total_s: float | None = None) -> bool:
    """Start the watchdog. Idempotent; returns True iff a bound is now active.

    ``per_phase_s`` / ``total_s`` let a runner that is legitimately slower (a
    coverage run instruments every line) raise its own ceiling without touching
    the defaults every other runner gets."""
    with _lock:
        if _state["armed"]:
            return bool(_state["per_phase"] or _state["total"])
        _state["armed"] = True
        per_phase = _budget(PER_PHASE_ENV, DEFAULT_PER_PHASE_S
                            if per_phase_s is None else per_phase_s)
        total = _budget(TOTAL_ENV,
                        DEFAULT_TOTAL_S if total_s is None else total_s)
        _state["per_phase"] = per_phase
        _state["total"] = total
        _state["run_start"] = _monotonic()
        note(_LABEL_DISCOVERY)
        if not (per_phase or total):
            print("[test-watchdog] DISABLED — this run has no time ceiling",
                  flush=True)
            return False
        stop = threading.Event()
        thread = threading.Thread(target=_watch, args=(stop,),
                                  name="test-watchdog", daemon=True)
        _state["stop"] = stop
        _state["thread"] = thread
        thread.start()
        print("[test-watchdog] ceiling %ss per test / %ss per run"
              % (_fmt(per_phase) if per_phase else "off",
                 _fmt(total) if total else "off"), flush=True)
        return True


def disarm() -> None:
    """Stop the watchdog. Called once the suite is done so nothing can fire
    during the summary. Safe to call when never armed."""
    with _lock:
        stop = _state.get("stop")
        if stop is not None:
            stop.set()
        _state["armed"] = False
        _state["stop"] = None
        _state["thread"] = None
        _state["per_phase"] = 0.0
        _state["total"] = 0.0
        _state["run_start"] = 0.0


class WatchdogTextTestResult(unittest.TextTestResult):
    """The result class that tells the watchdog which test is running.

    Pass it to a runner as ``resultclass=``. The gap AFTER a test is a phase in
    its own right on purpose: tearDown, tearDownModule and the import of the
    next module all happen there, and every one of them can hang — the
    2026-07-12 ci_sim freeze was a tearDown, not a test body."""

    def startTest(self, test) -> None:
        try:
            note("test %s" % test.id())
        except Exception:
            note("test <unidentifiable>")
        super().startTest(test)

    def stopTest(self, test) -> None:
        super().stopTest(test)
        try:
            note("after test %s (teardown / next module import)" % test.id())
        except Exception:
            note("after a test (teardown / next module import)")
