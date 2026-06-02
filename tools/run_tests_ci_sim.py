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
import sys
import unittest

# Heavy / optional pip packages the CI runner does NOT install (everything
# beyond .github/workflows/ci.yml's light set + stdlib + their transitive deps).
_BLOCK = {
    # ML / audio / vision
    "torch", "cv2", "sounddevice", "soundfile", "scipy", "noisereduce",
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


def main() -> int:
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec
    real_import_module = importlib.import_module

    def _imp(name, *a, **k):
        if _shim_should_block(name):
            raise ModuleNotFoundError(f"[ci-sim] {name!r} is not on the CI runner",
                                      name=name)
        return real_import(name, *a, **k)

    def _find_spec(name, package=None):
        if _shim_should_block(name):
            return None  # faithful: an absent module's find_spec returns None
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

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    suite = unittest.TestLoader().discover(
        os.path.join(root, "tests"), pattern="test_*.py", top_level_dir=root)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    print(f"=== CI-SIM: {res.testsRun} run, {len(res.failures)} failed, "
          f"{len(res.errors)} errored, {len(res.skipped)} skipped ===")
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
