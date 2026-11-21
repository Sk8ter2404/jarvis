"""core/fw_import.py - import faster_whisper ONE thread at a time.

WHY THIS EXISTS (the 11-20 boot, v3.0.26)
-----------------------------------------
Two boot threads imported faster_whisper at the same moment: the background
Whisper load (``from faster_whisper import WhisperModel``, parent package
first) and the Smart Turn warmer (core/endpointing.whisper_features:
``from faster_whisper.feature_extractor import FeatureExtractor``, the
SUBMODULE first). CPython takes the submodule's import lock and then imports
the parent while holding it, so the two threads took the same two per-module
locks in opposite orders. Python detected the cycle and raised
``_DeadlockError`` in the Whisper thread, which read it as a broken install,
fell back to openai-whisper on the 4 GB listen card, ran it out of memory and
left about 3.3 GB stuck there: Whisper ran on the CPU 'small' model all
session.

WHAT THIS MODULE DOES
---------------------
``import_faster_whisper(submodule)`` imports the parent package FIRST and
then (optionally) one submodule, under one process-wide lock, so no two
faster_whisper importers ever overlap. A ``_DeadlockError`` that still gets
through (an importer outside this lock, or a shared dependency imported by
another boot thread at the same time) is retried once after a short wait: a
failed import leaves the module out of sys.modules and the other importer has
finished by then, so the second attempt is a plain import.

The lock is held only around the import itself. Nothing that runs inside it
waits on another JARVIS lock, so it can never take part in a lock cycle; a
caller may hold its own lock (the monolith holds _stt_lock) while it waits
here.

CUDA ordering: faster_whisper imports ctranslate2, so the helper calls
core.cuda_preinit.before_ctranslate2() first (idempotent). Callers keep their
own call too, as before.

Stdlib only.
"""
from __future__ import annotations

import sys
import threading
import time

_lock = threading.RLock()
_DEADLOCK_RETRY_WAIT_S = 0.5


def is_import_deadlock(exc) -> bool:
    """True for importlib's ``_DeadlockError`` (a private class, so it is
    matched by name). Never raises."""
    try:
        return type(exc).__name__ == "_DeadlockError"
    except Exception:          # pragma: no cover - defensive
        return False


def import_faster_whisper(submodule: str = "", retries: int = 1):
    """Import ``faster_whisper`` (and then ``faster_whisper.<submodule>``
    when given) under the shared lock; return the last module imported.
    Retries ``retries`` times on an import deadlock. Any other error
    (ImportError when the package is missing included) propagates
    unchanged."""
    # faster_whisper imports ctranslate2: the CUDA driver starts first
    # (core/cuda_preinit.py, v2.0.180). Idempotent; a no-op once the boot or
    # the caller did it.
    try:
        from core import cuda_preinit as _cuda_preinit
        _cuda_preinit.before_ctranslate2()
    except Exception:          # pragma: no cover - it never raises
        pass
    with _lock:
        attempt = 0
        while True:
            try:
                # Plain import statements (builtins.__import__), so the
                # tests' import spies and stub finders see every attempt.
                import faster_whisper as mod
                if submodule:
                    __import__("faster_whisper." + submodule)
                    sub = sys.modules.get("faster_whisper." + submodule)
                    mod = sub if sub is not None else getattr(mod, submodule)
                return mod
            except Exception as e:
                if is_import_deadlock(e) and attempt < retries:
                    attempt += 1
                    print(f"  [whisper] faster_whisper import hit an import "
                          f"deadlock ({e}); retrying once")
                    time.sleep(_DEADLOCK_RETRY_WAIT_S)
                    continue
                raise
