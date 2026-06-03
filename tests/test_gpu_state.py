"""Logic tests for core.gpu_state — the one-shot nvidia-smi snapshot logger.

subprocess is mocked end-to-end so no real nvidia-smi runs, and _LOG_DIR is
redirected at a tempdir so nothing writes into the real logs/. The contract
under test: each model is snapshotted at most once per process, the helper
never raises (missing binary, non-zero exit, write error all degrade to a
console line), the file is appended with a timestamped header, and the >512 KB
rotation swaps to a single .1 backup.

stdlib unittest + unittest.mock only.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from unittest import mock

from core import gpu_state


_SAMPLE_SMI = (
    "Mon Jun  1 10:00:00 2026\n"
    "+-----------------------------------------------------------------------+\n"
    "| NVIDIA-SMI 555.00   Driver Version: 555.00   CUDA Version: 12.5        |\n"
    "|   0  NVIDIA GeForce RTX 3090   |   10240MiB / 24576MiB |     30%       |\n"
    "| Processes:                                                            |\n"
    "|    0   N/A  N/A     1234      C   ...ollama.exe          10000MiB     |\n"
    "+-----------------------------------------------------------------------+\n"
)


class _GpuTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        p = mock.patch.object(gpu_state, "_LOG_DIR", self._tmp.name)
        p.start()
        self.addCleanup(p.stop)
        # Reset the per-process dedup + warn state so tests are independent.
        with gpu_state._lock:
            gpu_state._OLLAMA_MODELS_LOGGED.clear()
        gpu_state._NVIDIA_SMI_MISSING_WARNED[0] = False
        self.addCleanup(self._reset)
        self.log_path = os.path.join(self._tmp.name, "gpu_snapshots.log")

    def _reset(self):
        with gpu_state._lock:
            gpu_state._OLLAMA_MODELS_LOGGED.clear()
        gpu_state._NVIDIA_SMI_MISSING_WARNED[0] = False

    def _read_log(self):
        with open(self.log_path, "r", encoding="utf-8") as f:
            return f.read()


def _run_ok(stdout=_SAMPLE_SMI):
    return mock.MagicMock(returncode=0, stdout=stdout)


class RunNvidiaSmiTests(_GpuTestBase):
    def test_returns_stdout_on_success(self):
        with mock.patch.object(gpu_state.subprocess, "run", return_value=_run_ok()):
            self.assertEqual(gpu_state._run_nvidia_smi(), _SAMPLE_SMI)

    def test_none_on_missing_binary(self):
        with mock.patch.object(gpu_state.subprocess, "run",
                               side_effect=FileNotFoundError()):
            self.assertIsNone(gpu_state._run_nvidia_smi())

    def test_none_on_timeout(self):
        with mock.patch.object(gpu_state.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("nvidia-smi", 2)):
            self.assertIsNone(gpu_state._run_nvidia_smi())

    def test_none_on_nonzero_exit(self):
        with mock.patch.object(gpu_state.subprocess, "run",
                               return_value=mock.MagicMock(returncode=1, stdout="x")):
            self.assertIsNone(gpu_state._run_nvidia_smi())

    def test_none_on_unexpected_exception(self):
        # A non-OSError surprise from subprocess.run (e.g. a ValueError on a
        # malformed argv) is caught by the catch-all and degrades to None.
        with mock.patch.object(gpu_state.subprocess, "run",
                               side_effect=ValueError("weird")):
            self.assertIsNone(gpu_state._run_nvidia_smi())


class LogGpuStateTests(_GpuTestBase):
    def test_writes_snapshot_with_header(self):
        with mock.patch.object(gpu_state, "_run_nvidia_smi", return_value=_SAMPLE_SMI):
            gpu_state.log_gpu_state("qwen2.5:14b")
        body = self._read_log()
        self.assertIn("model=qwen2.5:14b", body)
        self.assertIn("RTX 3090", body)
        self.assertIn("ollama.exe", body)

    def test_only_logged_once_per_model(self):
        run = mock.MagicMock(return_value=_SAMPLE_SMI)
        with mock.patch.object(gpu_state, "_run_nvidia_smi", run):
            gpu_state.log_gpu_state("qwen2.5:14b")
            gpu_state.log_gpu_state("qwen2.5:14b")   # second call is a no-op
        self.assertEqual(run.call_count, 1)

    def test_distinct_models_each_logged(self):
        run = mock.MagicMock(return_value=_SAMPLE_SMI)
        with mock.patch.object(gpu_state, "_run_nvidia_smi", run):
            gpu_state.log_gpu_state("qwen2.5:14b")
            gpu_state.log_gpu_state("nomic-embed-text")
        self.assertEqual(run.call_count, 2)

    def test_empty_model_name_is_noop(self):
        run = mock.MagicMock(return_value=_SAMPLE_SMI)
        with mock.patch.object(gpu_state, "_run_nvidia_smi", run):
            gpu_state.log_gpu_state("")
        run.assert_not_called()
        self.assertFalse(os.path.exists(self.log_path))

    def test_missing_smi_warns_once_no_file(self):
        # nvidia-smi unavailable → no log file, warn flag latches.
        with mock.patch.object(gpu_state, "_run_nvidia_smi", return_value=None):
            gpu_state.log_gpu_state("m1")
            gpu_state.log_gpu_state("m2")
        self.assertTrue(gpu_state._NVIDIA_SMI_MISSING_WARNED[0])
        self.assertFalse(os.path.exists(self.log_path))

    def test_never_raises_on_run_exception(self):
        # A surprise exception inside the snapshot path must be swallowed.
        with mock.patch.object(gpu_state, "_run_nvidia_smi",
                               side_effect=RuntimeError("boom")):
            try:
                gpu_state.log_gpu_state("m1")
            except Exception as e:           # pragma: no cover - the assert is the point
                self.fail(f"log_gpu_state raised: {e}")

    def test_model_still_marked_when_smi_missing(self):
        # Dedup happens before the snapshot, so a missing-smi model is not
        # retried on the next call (avoids hammering a non-NVIDIA host).
        run = mock.MagicMock(return_value=None)
        with mock.patch.object(gpu_state, "_run_nvidia_smi", run):
            gpu_state.log_gpu_state("m1")
            gpu_state.log_gpu_state("m1")
        self.assertEqual(run.call_count, 1)


class RotationTests(_GpuTestBase):
    def test_rotates_when_over_cap(self):
        # Pre-seed an oversized log; the next snapshot should rotate it to .1.
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("OLD-CONTENT\n")
            f.write("x" * (gpu_state._LOG_MAX_BYTES + 1))
        with mock.patch.object(gpu_state, "_run_nvidia_smi", return_value=_SAMPLE_SMI):
            gpu_state.log_gpu_state("qwen2.5:14b")
        backup = self.log_path + ".1"
        self.assertTrue(os.path.exists(backup))
        with open(backup, "r", encoding="utf-8") as f:
            self.assertIn("OLD-CONTENT", f.read())
        # New file holds the fresh snapshot, not the old content.
        self.assertIn("model=qwen2.5:14b", self._read_log())

    def test_no_rotation_under_cap(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("small\n")
        with mock.patch.object(gpu_state, "_run_nvidia_smi", return_value=_SAMPLE_SMI):
            gpu_state.log_gpu_state("qwen2.5:14b")
        self.assertFalse(os.path.exists(self.log_path + ".1"))

    def test_rotate_missing_file_is_noop(self):
        # No file yet → rotation helper returns quietly, no exception.
        gpu_state._rotate_log_if_large(self.log_path)
        self.assertFalse(os.path.exists(self.log_path + ".1"))

    def test_rotate_replace_failure_is_swallowed(self):
        # An oversized file that can't be renamed (os.replace raises, e.g. a
        # locked .1 backup on Windows) degrades to a console line, not a raise.
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("x" * (gpu_state._LOG_MAX_BYTES + 1))
        with mock.patch.object(gpu_state.os, "replace",
                               side_effect=OSError("locked")):
            gpu_state._rotate_log_if_large(self.log_path)   # must not raise
        # Original file is left in place; no backup was produced.
        self.assertTrue(os.path.exists(self.log_path))
        self.assertFalse(os.path.exists(self.log_path + ".1"))


class WriteFailureTests(_GpuTestBase):
    def test_write_error_is_swallowed(self):
        # The snapshot ran fine but the log write fails (OSError) — log_gpu_state
        # still returns without raising; the failure is just printed.
        with mock.patch.object(gpu_state, "_run_nvidia_smi", return_value=_SAMPLE_SMI), \
             mock.patch.object(gpu_state, "open", create=True,
                               side_effect=OSError("disk full")):
            gpu_state.log_gpu_state("qwen2.5:14b")   # must not raise
        self.assertFalse(os.path.exists(self.log_path))

    def test_snapshot_without_trailing_newline_gets_one(self):
        # When the captured snapshot doesn't end in '\n', the writer appends one
        # so the next header starts on its own line.
        snap = _SAMPLE_SMI.rstrip("\n")   # strip the trailing newline
        with mock.patch.object(gpu_state, "_run_nvidia_smi", return_value=snap):
            gpu_state.log_gpu_state("qwen2.5:14b")
        self.assertTrue(self._read_log().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
