#!/usr/bin/env python3
"""Run the unit suite with the heavy / optional pip deps BLOCKED, mirroring the
light-deps GitHub Actions runner.

The CI workflow installs only a small dep set (see .github/workflows/ci.yml).
Tests written on a full dev box can silently assume an optional dependency
(torch, noisereduce, browser_use, chromadb, ...) is importable — they pass
locally but FAIL on CI. This makes those deps look ABSENT (import raises
ModuleNotFoundError; importlib.util.find_spec returns None) so the failures
surface here instead of on the runner.

    python tools/run_tests_ci_sim.py

Exit 0 = CI-clean (0 failed / 0 errored; skips are fine).

Scope: this simulates dependency ABSENCE only — NOT Linux itself. Windows-only
modules (winreg, win32*, ctypes.windll) are deliberately NOT blocked here
because well-behaved libs take a non-Windows code path on Linux; OS-specific
tests must self-guard with ``skipUnless(sys.platform.startswith("win"))``.
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


def _blocked(name: str) -> bool:
    return name.split(".")[0] in _BLOCK


def main() -> int:
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec
    real_import_module = importlib.import_module

    def _imp(name, *a, **k):
        if _blocked(name) and name.split(".")[0] not in sys.modules:
            raise ModuleNotFoundError(f"[ci-sim] {name!r} is not on the CI runner",
                                      name=name)
        return real_import(name, *a, **k)

    def _find_spec(name, package=None):
        if _blocked(name) and name.split(".")[0] not in sys.modules:
            return None  # faithful: an absent module's find_spec returns None
        return real_find_spec(name, package)

    def _import_module(name, package=None):
        if _blocked(name) and name.split(".")[0] not in sys.modules:
            raise ModuleNotFoundError(f"[ci-sim] {name!r} is not on the CI runner",
                                      name=name)
        return real_import_module(name, package)

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
