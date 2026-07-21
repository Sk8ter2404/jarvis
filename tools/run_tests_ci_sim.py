#!/usr/bin/env python3
"""Run the unit suite the way the LINUX GitHub Actions runner sees it: heavy /
optional pip deps BLOCKED *and* the host faithfully made to look like Linux.

The CI workflow runs on ``ubuntu-latest`` and installs only a small dep set
(see .github/workflows/ci.yml). Tests written on a full Windows dev box can
silently assume

  * an optional dependency (torch, noisereduce, browser_use, chromadb, …) is
    importable, or
  * the host is Windows — ``sys.platform == "win32"``, a Windows-only module
    (winreg / win32* / pythoncom / ctypes.wintypes) imports, ``ctypes.windll``
    exists, or ``subprocess.CREATE_NO_WINDOW`` is defined.

Either way they pass locally but FAIL on CI. This runner reproduces BOTH classes
of failure locally:

  1. Dependency absence — blocked imports raise ModuleNotFoundError and
     ``importlib.util.find_spec`` returns None.
  2. Linux platform — ``sys.platform`` is "linux", the Windows-only modules
     above look absent, ``ctypes.windll`` is gone (but ``ctypes`` itself still
     imports — Linux has ctypes, just no windll), and
     ``subprocess.CREATE_NO_WINDOW`` is removed.

    python tools/run_tests_ci_sim.py

Exit 0 = CI-clean (0 failed / 0 errored; skips are fine).

Calibration: a CLEAN tree must give 0 failed / 0 errored here — only genuinely
Windows-assuming tests should fail. ``tzlocal`` / ``apscheduler`` have real
Linux code paths but pick a Windows backend at import time on this box; they are
imported (with their win32 backend) *before* the platform is flipped so their
already-resolved local-timezone lookup keeps working under the sim.
"""
from __future__ import annotations

import builtins
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# Heavy / optional pip packages the CI runner does NOT install (everything
# beyond .github/workflows/ci.yml's light set + stdlib + their transitive deps).
_BLOCK = {
    # ML / audio / vision
    "torch", "cv2", "sounddevice", "soundfile", "scipy", "noisereduce",
    "chatterbox",   # optional local voice-clone engine (core/voice_clone.py)
    "chromadb", "sentence_transformers", "faster_whisper", "whisper",
    "pyaudio", "pyttsx3", "edge_tts", "webrtc_audio_processing", "mediapipe",
    "dlib", "face_recognition",
    # browser / web automation
    "browser_use", "yarl", "langchain", "langchain_anthropic", "playwright",
    "selenium", "aiohttp", "alexapy",
    # docs / fs / integrations
    "pypdf", "docx", "watchdog", "tplinkrouterc6u", "mcp", "ollama", "phue",
    "kasa", "tinytuya", "govee", "ring_doorbell", "googleapiclient",
    "pyautogui", "mss", "pygetwindow", "win32crypt", "win32com",
}

# Packages that ARE installed on the Linux runner but whose import SELECTS A GUI
# / DISPLAY BACKEND at import time and therefore EXPLODES on a headless host
# (no $DISPLAY). On the real CI runner ``import pystray`` raises
# ``Xlib.error.DisplayNameError: Bad display name ""`` because its default Linux
# backend is X11. On this Windows box it imports fine (win32 backend), so a
# module that does ``import pystray`` at top level would pass locally yet fail
# the headless CI collection. Treat these like a headless-Linux import failure:
# make a fresh ``import`` raise. (A test that needs the symbols must inject its
# own fake into ``sys.modules`` *before* importing the module under test — which
# satisfies the ``key in sys.modules`` bypass below, exactly like the win-only
# handling — so a CLEAN tree still reports 0 failed / 0 errored.)
_HEADLESS = {
    "pystray",
}

# Windows-only stdlib / pywin32 modules that do NOT exist on the Linux runner.
# Importing any of these on CI raises ModuleNotFoundError, so we make them look
# absent here too. ``ctypes`` is deliberately NOT in this set — Linux ships
# ctypes; only its ``windll`` attribute is Windows-only (handled separately).
_WIN_ONLY = {
    "winreg", "winsound", "win32api", "win32con", "win32gui", "win32com",
    "win32crypt", "win32file", "win32event", "win32process", "win32security",
    "win32clipboard", "pywintypes", "pythoncom", "ctypes.wintypes",
}

# CI-installed deps with genuine Linux support that nonetheless do
# platform-specific work AT IMPORT TIME (pick a per-OS backend, or probe
# ``os.uname`` / ``resource`` that only exist off-Windows). Importing them
# fresh *after* the platform flip would wrongly fail on this Windows box, even
# though they import fine on the real Linux runner. Pre-importing them (and
# their Windows backend) BEFORE the flip freezes the working result, so the
# flip can't break them — this is what keeps the sim free of false positives.
# Each is best-effort: one that isn't installed here is simply skipped (it would
# legitimately be absent → its tests skip in both this sim and on CI).
_PREIMPORT = (
    # numpy probes os.uname() at import (AttributeError on Windows post-flip).
    "numpy",
    # psutil imports its per-OS backend at import; _pslinux needs ``resource``.
    "psutil",
    # NB: pystray is deliberately NOT pre-imported. On the real Linux runner it
    # selects an X11 backend and dies headless, so we treat it as unimportable
    # (see ``_HEADLESS``) rather than freezing the working Windows backend here.
    # Pillow has per-platform bits; harmless to lock in early.
    "PIL", "PIL.Image",
    # tzlocal/apscheduler resolve the local timezone via a Windows backend.
    "tzlocal", "tzlocal.win32",
    "apscheduler", "apscheduler.schedulers.background",
    "apscheduler.triggers.cron", "apscheduler.triggers.interval",
)


def _blocked(name: str) -> bool:
    head = name.split(".")[0]
    if head in _BLOCK:
        return True
    # Headless-display backends (pystray): importable on Windows, but a fresh
    # import dies on the headless Linux runner. Block both the bare head and any
    # submodule so ``import pystray`` and ``import pystray._base`` both fail
    # unless a fake was pre-seeded into sys.modules (see _shim_should_block).
    if head in _HEADLESS:
        return True
    # Match win-only modules both as a bare head (winreg) and dotted
    # (ctypes.wintypes) — but never block plain ``ctypes``.
    if name in _WIN_ONLY or head in _WIN_ONLY:
        return True
    return False


def _shim_should_block(name: str) -> bool:
    """A blocked import is only let through when the module is genuinely already
    loaded. For a dotted win-only submodule (``ctypes.wintypes``) the package
    head (``ctypes``) is always loaded, so keying the bypass on the head would
    leak the submodule through. Key it on the *most specific* name instead: a
    win-only submodule stays blocked even though its package is importable."""
    if not _blocked(name):
        return False
    key = name if (name in _WIN_ONLY) else name.split(".")[0]
    return key not in sys.modules


def _preimport_linux_safe() -> None:
    """Import the modules that need their Windows backend resolved before the
    platform flip. Best-effort: a missing one (e.g. on a real Linux box) is
    simply skipped."""
    for name in _PREIMPORT:
        try:
            importlib.import_module(name)
        except Exception:
            pass


def _make_windows_absent() -> None:
    """Flip the running interpreter to look like the Linux CI host."""
    # 1) Platform string — drives sys.platform checks and
    #    skipUnless(sys.platform.startswith("win")) decorators.
    sys.platform = "linux"

    # 2) subprocess.CREATE_NO_WINDOW is a Windows-only constant; on Linux the
    #    attribute is absent, so code that references it unguarded (or inside a
    #    win32 branch a test forces on) raises AttributeError like it does on CI.
    #    NB: we remove ONLY this constant. STARTUPINFO / STARTF_USESHOWWINDOW /
    #    CREATE_NEW_PROCESS_GROUP are also Windows-only, but CPython's own
    #    subprocess implementation references STARTUPINFO internally when
    #    spawning on this (really-Windows) host, so deleting them would break
    #    legitimate Popen calls — a false positive. No JARVIS code path needs
    #    them gone to surface a real Linux failure.
    import subprocess
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        del subprocess.CREATE_NO_WINDOW

    # 3) ctypes stays importable (Linux has it) but loses its Windows-only DLL
    #    loaders, so ``ctypes.windll.user32...`` fails as on CI. Tests that need
    #    windll patch it in with create=True. We touch ONLY the DLL-loader
    #    attributes; the rest of ctypes (used by other libs mid-run) is intact.
    import ctypes
    for _attr in ("windll", "oledll", "WinDLL", "OleDLL"):
        if hasattr(ctypes, _attr):
            try:
                delattr(ctypes, _attr)
            except (AttributeError, TypeError):
                pass

    # 4) Evict any already-imported win-only modules so a re-import goes through
    #    the blocking shim below.
    for mod in list(sys.modules):
        if mod in _WIN_ONLY or mod.split(".")[0] in _WIN_ONLY:
            del sys.modules[mod]


# The non-test CI gate steps from .github/workflows/ci.yml. Run as clean
# subprocesses (exactly how CI runs each step) BEFORE the in-process platform
# flip, so a lint / syntax / PII regression is caught locally instead of on the
# runner. Keep this list in lockstep with ci.yml.
_CI_GATES = (
    ("compileall", ["-m", "compileall", "-q",
                    "core", "skills", "tools", "tests", "adapters", "hud"]),
    ("pyflakes tests", ["-m", "pyflakes", "tests"]),
    ("check_no_pii", ["tools/check_no_pii.py"]),
)


def _run_ci_gates(root: str) -> bool:
    """Run CI's non-test gate steps the way the runner does. Returns True iff all
    pass; prints the tail of any failing step's output for diagnosis."""
    all_ok = True
    print("--- CI gate steps (mirror of ci.yml non-test steps) ---")
    for label, argv in _CI_GATES:
        try:
            proc = subprocess.run([sys.executable, *argv], cwd=root,
                                  capture_output=True, text=True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[ci-sim gate] {label}: could not run ({exc})")
            all_ok = False
            continue
        if proc.returncode == 0:
            print(f"[ci-sim gate] {label}: OK")
        else:
            all_ok = False
            print(f"[ci-sim gate] {label}: FAIL (exit {proc.returncode})")
            tail = ((proc.stdout or "") + (proc.stderr or "")).splitlines()
            for line in tail[-15:]:
                print(f"    {line}")
    print("--- end CI gate steps ---")
    return all_ok


def _redirect_settings_to_throwaway(root: str) -> None:
    """Point the WHOLE suite's settings reads/writes at a throwaway file so a
    test exercising the real ``tools.settings_window.save_settings`` can NEVER
    clobber the owner's live ``data/user_settings.json``.

    Sets ``JARVIS_SETTINGS_PATH`` (honoured by settings_window.settings_path())
    to a file in a fresh temp dir BEFORE any test is imported. Seeds it with a
    copy of the real file when present (so load_settings sees realistic data),
    else load_settings just returns defaults. Respects an existing override."""
    if (os.environ.get("JARVIS_SETTINGS_PATH") or "").strip():
        return
    throwaway_dir = tempfile.mkdtemp(prefix="jarvis_test_settings_")
    throwaway = os.path.join(throwaway_dir, "test_user_settings.json")
    real = os.path.join(root, "data", "user_settings.json")
    if os.path.exists(real):
        try:
            shutil.copyfile(real, throwaway)
        except OSError:
            pass
    os.environ["JARVIS_SETTINGS_PATH"] = throwaway


def _redirect_data_dir_to_throwaway() -> None:
    """Point core.paths.data_dir() at a throwaway directory so ANY test that
    forgets to redirect a module's file-path globals still cannot write the
    owner's live ``data/``.

    Companion to _redirect_settings_to_throwaway — same incident class: a
    targeted run of tests/skills/test_sh_ecobee.py once overwrote the LIVE
    ``data/sh_ecobee_tokens.json`` with fake fixture tokens because an
    ineffective builtins.open mock let the real atomic write through
    (2026-07-21). Sets ``JARVIS_DATA_DIR`` (highest precedence in
    core.paths.data_dir()) BEFORE any test is imported; respects an
    externally-set override (does nothing if already set)."""
    if (os.environ.get("JARVIS_DATA_DIR") or "").strip():
        return
    os.environ["JARVIS_DATA_DIR"] = tempfile.mkdtemp(prefix="jarvis_test_data_")


def _stop_lingering_daemons() -> None:
    """Best-effort: stop any opt-in background daemon a test may have left alive
    so it can't outlive the suite. Currently just the apple-music keep-alive
    watchdog (a non-terminating loop). Never raises; a missing module is a
    no-op."""
    try:
        mod = sys.modules.get("audio.apple_music_keeper")
        if mod is not None and hasattr(mod, "stop_keeper"):
            mod.stop_keeper(timeout=5.0)
    except Exception:
        pass


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Redirect settings I/O to a throwaway copy BEFORE discovery/import (and
    # before the gate subprocesses, which inherit this env), so no test or gate
    # can touch the real data/user_settings.json.
    _redirect_settings_to_throwaway(root)
    # Same guard for the whole data/ directory: a forgotten per-test path
    # redirect must land in a throwaway, never the live runtime state.
    _redirect_data_dir_to_throwaway()
    # Mirror CI's non-test gates (syntax sweep, lint, PII) up front, before the
    # platform flip — these run in a normal environment on CI, so a clean
    # subprocess here is faithful and catches regressions the test run can't.
    gates_ok = _run_ci_gates(root)

    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec
    real_import_module = importlib.import_module

    def _imp(name, *a, **k):
        if _shim_should_block(name):
            raise ModuleNotFoundError(f"[ci-sim] {name!r} is not on the CI runner",
                                      name=name)
        # FROMLIST FORM (2026-07-14 audit, #36). `from ctypes import wintypes`
        # calls __import__("ctypes", ..., ["wintypes"]) — name is the importable
        # PACKAGE ("ctypes"), so the head check above passes and the win-only
        # submodule `ctypes.wintypes` leaked straight through, defeating the very
        # thing _WIN_ONLY lists it for.
        #
        # Scope this NARROWLY to _WIN_ONLY dotted entries only. A fromlist item is
        # usually an ATTRIBUTE (a class/function), not a submodule — `from
        # mcp.client import ClientSession` requests the class ClientSession, and
        # `mcp.client.ClientSession` is not a module. Keying off the broad
        # _blocked() (head-in-_BLOCK) would wrongly reject every attribute import
        # from a faked-but-present package, which is exactly how an over-eager
        # first cut of this fix broke ~50 import-fallback tests. Only a name that
        # is ITSELF a listed win-only submodule (ctypes.wintypes) qualifies.
        fromlist = a[2] if len(a) >= 3 else k.get("fromlist")
        if fromlist:
            for f in fromlist:
                if not isinstance(f, str) or f == "*":
                    continue
                dotted = f"{name}.{f}"
                if dotted in _WIN_ONLY and dotted not in sys.modules:
                    raise ModuleNotFoundError(
                        f"[ci-sim] {dotted!r} is not on the CI runner",
                        name=dotted)
        return real_import(name, *a, **k)

    def _find_spec(name, package=None):
        if _shim_should_block(name):
            # Real importlib.find_spec imports a dotted name's PARENT to locate
            # the submodule: if that parent package is itself absent it RAISES
            # ModuleNotFoundError (it does NOT return None). Mirror that for a
            # dotted name whose head is blocked (e.g. find_spec(
            # "googleapiclient.discovery") with googleapiclient absent) so
            # dotted-probe bugs surface HERE instead of on the real runner. A
            # bare absent top-level module → None; a submodule whose parent DOES
            # import (e.g. ctypes.wintypes, ctypes present) → None.
            head = name.split(".")[0]
            if "." in name and _blocked(head):
                raise ModuleNotFoundError(
                    f"[ci-sim] parent {head!r} is not on the CI runner",
                    name=name)
            return None
        return real_find_spec(name, package)

    def _import_module(name, package=None):
        if _shim_should_block(name):
            raise ModuleNotFoundError(f"[ci-sim] {name!r} is not on the CI runner",
                                      name=name)
        return real_import_module(name, package)

    # Lock in the Linux-safe modules (resolves their win backend) BEFORE we flip
    # the platform or install the shims — otherwise re-importing them post-flip
    # would take an unconfigured Linux path and warn/UTC-default.
    _preimport_linux_safe()

    # Make the host look like Linux: platform string, absent win-only modules,
    # no ctypes.windll, no subprocess.CREATE_NO_WINDOW.
    _make_windows_absent()

    # Evict any already-imported blocked modules, then install the shims.
    for mod in list(sys.modules):
        if _blocked(mod):
            del sys.modules[mod]
    builtins.__import__ = _imp
    importlib.util.find_spec = _find_spec
    importlib.import_module = _import_module

    if root not in sys.path:
        sys.path.insert(0, root)
    suite = unittest.TestLoader().discover(
        os.path.join(root, "tests"), pattern="test_*.py", top_level_dir=root)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    # Belt-and-suspenders: reap any lingering opt-in daemon a test left running
    # (e.g. the apple-music keep-alive watchdog) so it can't bleed CPU into the
    # next invocation. The real guard is per-test cleanup in the test modules;
    # this is a harmless final sweep that never fails the run.
    _stop_lingering_daemons()
    gate_note = "" if gates_ok else "  [+ CI GATE FAILURE above]"
    print(f"=== CI-SIM: {res.testsRun} run, {len(res.failures)} failed, "
          f"{len(res.errors)} errored, {len(res.skipped)} skipped{gate_note} ===")
    return 0 if (res.wasSuccessful() and gates_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
