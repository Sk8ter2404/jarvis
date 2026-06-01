"""GPU-state snapshot helper for Ollama model loads.

Logs a one-shot `nvidia-smi` snapshot the first time JARVIS calls into a
given Ollama-served model (qwen2.5:14b, qwen2.5vl, nomic-embed-text, …)
so VRAM allocation can be confirmed in the session log without manual
intervention. Each model is logged at most once per process. The helper
is intentionally tolerant: a missing `nvidia-smi` binary or non-NVIDIA
GPU degrades to a single console warning and is never raised back to
the caller — model-load paths must not block on diagnostics.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_DIR, "logs")

_lock = threading.Lock()
_OLLAMA_MODELS_LOGGED: set[str] = set()
_NVIDIA_SMI_MISSING_WARNED = [False]

# Cap on gpu_snapshots.log before we rotate. Snapshots are appended on every
# first-seen model load and the file is otherwise never pruned, so without a
# cap it grows unbounded across restarts. At ~512 KB we rotate to a single
# .1 backup (overwriting any previous one), keeping at most ~1 MB on disk.
_LOG_MAX_BYTES = 512 * 1024


def _rotate_log_if_large(log_path: str) -> None:
    """Rotate `log_path` to `<path>.1` if it exceeds _LOG_MAX_BYTES.

    Uses os.replace so the swap is atomic and overwrites any existing .1 in a
    single step (robust on Windows, where rename-onto-existing otherwise
    raises). Never raises — a rotation failure must not block logging."""
    try:
        if os.path.getsize(log_path) <= _LOG_MAX_BYTES:
            return
    except OSError:
        # File doesn't exist yet (nothing to rotate) or stat failed — either
        # way, leave the append path to handle it.
        return
    try:
        os.replace(log_path, log_path + ".1")
    except OSError as e:
        print(f"  [gpu] could not rotate gpu_snapshots.log: {e}")


def _run_nvidia_smi() -> str | None:
    """Capture an nvidia-smi snapshot. Returns stdout text or None on
    any failure (binary missing, non-NVIDIA host, timeout)."""
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout or ""


def log_gpu_state(model_name: str) -> None:
    """Log a one-shot nvidia-smi snapshot the first time `model_name`
    is seen. Subsequent calls for the same model are no-ops. Thread-safe.
    Never raises — the worst case is a single console warning.

    Output: a `[gpu]` block on stdout summarising VRAM, and the same
    snapshot appended to logs/gpu_snapshots.log with a timestamp.
    """
    if not model_name:
        return
    with _lock:
        if model_name in _OLLAMA_MODELS_LOGGED:
            return
        _OLLAMA_MODELS_LOGGED.add(model_name)

    try:
        snapshot = _run_nvidia_smi()
        if snapshot is None:
            if not _NVIDIA_SMI_MISSING_WARNED[0]:
                _NVIDIA_SMI_MISSING_WARNED[0] = True
                print("  [gpu] nvidia-smi unavailable — skipping VRAM snapshots "
                      "(non-NVIDIA host or driver missing).")
            return

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [gpu] nvidia-smi snapshot for `{model_name}` ({ts}):")
        # Trim to the lines that matter: header MIB summary + process list.
        # Full output is preserved in the log file regardless.
        for line in snapshot.splitlines():
            if ("MiB" in line or "Processes" in line or "ollama" in line.lower()
                    or "==" in line or "GPU" in line and "Fan" in line):
                print(f"    {line}")

        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            log_path = os.path.join(_LOG_DIR, "gpu_snapshots.log")
            _rotate_log_if_large(log_path)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== {ts}  model={model_name} =====\n")
                f.write(snapshot)
                if not snapshot.endswith("\n"):
                    f.write("\n")
        except OSError as e:
            print(f"  [gpu] could not write gpu_snapshots.log: {e}")
    except Exception as e:
        # Last-ditch guard: GPU logging must never propagate.
        print(f"  [gpu] log_gpu_state failed for `{model_name}`: {e}")
