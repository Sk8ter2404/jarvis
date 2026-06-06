"""Headless load + staleness-logic tests for the live camera preview in
``hud/jarvis_unified_hud.py`` (the auto-launched unified HUD).

WHY THIS EXISTS
  The HUD shows a small picture-in-picture mirror of what JARVIS sees. The main
  process (``bobert_companion.py``) writes ONE overwriting, downscaled JPEG of
  the primary face-tracking frame to ``data/.hud_camera_preview.jpg`` while the
  camera is on + tracking, and removes it when the camera is off / tracking is
  paused. The HUD widget loads that file and, crucially, treats a file older
  than ``CAMERA_PREVIEW_STALE_S`` as "camera off" — that staleness gate is the
  belt-and-braces guard that stops a missed delete on the writer side from
  leaving a frozen frame on screen.

  Like the other reactor HUDs, this module imports PyQt6 and subclasses QWidget.
  PyQt6 is NOT on the CI runner (ubuntu-latest, light dep set), and even where it
  is a renderer test must never spin a real Qt loop. The freshness gate under
  test (``_camera_preview_fresh_at``) touches no Qt at all — only ``os`` / ``time``
  / the module's ``CAMERA_PREVIEW_FILE`` + ``CAMERA_PREVIEW_STALE_S`` globals —
  so we load the source with PyQt6 genuinely ABSENT (its own ``except ImportError``
  stub path makes the Qt names harmless placeholders and ``_HAS_PYQT6`` False) and
  call the helper directly. No widget is built and no display is touched, so the
  suite runs identically on the headless runner.

  This mirrors ``tests/skills/test_hud_gpu_util.py`` (same load-with-PyQt6-blocked
  harness); the value here is (a) proving the camera-preview additions still take
  the clean headless degrade path, and (b) pinning the privacy-relevant staleness
  contract so a frozen-frame regression is caught in CI.

ISOLATION
  • PyQt6 imports are blocked for the duration of the load (a fake ``__import__``
    raises ImportError for the ``PyQt6`` package), restored on cleanup. Any
    previously-cached PyQt6 submodules are hidden during the load and restored
    after, so a dev box that *has* PyQt6 still exercises the headless path.
  • The source is loaded from its file path under a synthetic module name and
    dropped on cleanup → pristine module globals.
  • ``CAMERA_PREVIEW_FILE`` is redirected to a per-test temp file, so no real
    project file is read or written.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hud",
)


def _load_hud_no_pyqt(testcase, filename, mod_name):
    """Load a HUD source from hud/<filename> with PyQt6 import blocked, so the
    module takes its graceful-degrade path (``_HAS_PYQT6`` False, Qt names
    stubbed). Restores sys.modules + the real importer on cleanup."""
    path = os.path.join(_HUD_DIR, filename)
    real_import = __import__

    def _imp(name, *a, **k):
        if name.split(".")[0] == "PyQt6":
            raise ImportError(f"[test] PyQt6 blocked: {name}")
        return real_import(name, *a, **k)

    hidden = {n: sys.modules.pop(n)
              for n in list(sys.modules) if n.split(".")[0] == "PyQt6"}
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module

    def restore():
        sys.modules.pop(mod_name, None)
        sys.modules.update(hidden)

    testcase.addCleanup(restore)
    with mock.patch("builtins.__import__", side_effect=_imp):
        spec.loader.exec_module(module)
    testcase.assertFalse(module._HAS_PYQT6,
                         "PyQt6 should be blocked → headless degrade path")
    return module


class UnifiedHudCameraPreviewTests(unittest.TestCase):
    MOD_NAME = "_ju_hud_campreview_under_test"

    def setUp(self):
        self.mod = _load_hud_no_pyqt(self, "jarvis_unified_hud.py", self.MOD_NAME)
        self.tmp = tempfile.mkdtemp(prefix="hud_cam_preview_test_")
        self.addCleanup(self._cleanup_tmp)
        self.preview = os.path.join(self.tmp, ".hud_camera_preview.jpg")
        # Redirect the module's preview-file path at the per-test temp file so no
        # real project file is touched.
        self.mod.CAMERA_PREVIEW_FILE = self.preview

    def _cleanup_tmp(self):
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def _write_preview_with_mtime(self, age_s: float, now: float):
        """Create the preview file and stamp its mtime to (now - age_s)."""
        with open(self.preview, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0not-a-real-jpeg-but-bytes-on-disk")
        os.utime(self.preview, (now - age_s, now - age_s))

    # ── the privacy-relevant staleness contract ─────────────────────────────
    def test_missing_file_is_not_fresh(self):
        # No preview file at all → camera off → placeholder (not fresh).
        self.assertFalse(self.preview and os.path.exists(self.preview))
        self.assertFalse(self.mod._camera_preview_fresh_at(10_000.0))

    def test_just_written_file_is_fresh(self):
        now = 10_000.0
        self._write_preview_with_mtime(age_s=0.0, now=now)
        self.assertTrue(self.mod._camera_preview_fresh_at(now))

    def test_recent_file_within_window_is_fresh(self):
        now = 10_000.0
        # Half the stale window old → still fresh.
        self._write_preview_with_mtime(
            age_s=self.mod.CAMERA_PREVIEW_STALE_S / 2.0, now=now)
        self.assertTrue(self.mod._camera_preview_fresh_at(now))

    def test_old_file_beyond_window_is_stale(self):
        now = 10_000.0
        # Comfortably past the window (writer stopped / a missed delete) → the
        # HUD must treat it as off, not show a frozen frame.
        self._write_preview_with_mtime(
            age_s=self.mod.CAMERA_PREVIEW_STALE_S + 5.0, now=now)
        self.assertFalse(self.mod._camera_preview_fresh_at(now))

    def test_exactly_at_window_boundary_is_fresh(self):
        now = 10_000.0
        # Boundary is inclusive (``<=``): a file exactly STALE_S old still counts.
        self._write_preview_with_mtime(
            age_s=self.mod.CAMERA_PREVIEW_STALE_S, now=now)
        self.assertTrue(self.mod._camera_preview_fresh_at(now))

    def test_stat_error_is_not_fresh(self):
        # An OSError out of getmtime (e.g. file vanished mid-check) → not fresh,
        # never an exception bubbling into the paint loop.
        with mock.patch.object(self.mod.os.path, "getmtime",
                               side_effect=OSError("boom")):
            self.assertFalse(self.mod._camera_preview_fresh_at(10_000.0))

    # ── config / constant sanity ─────────────────────────────────────────────
    def test_preview_path_points_into_data_dir(self):
        # The shipped default must live under data/ (gitignored) and be the
        # single dotfile the writer overwrites — never a growing folder.
        default = os.path.join(self.mod.PROJECT_DIR, "data",
                               ".hud_camera_preview.jpg")
        # Reload a pristine module to read the un-redirected default.
        fresh = _load_hud_no_pyqt(self, "jarvis_unified_hud.py",
                                  self.MOD_NAME + "_b")
        self.assertEqual(fresh.CAMERA_PREVIEW_FILE, default)
        self.assertTrue(os.path.basename(fresh.CAMERA_PREVIEW_FILE).startswith("."),
                        "preview file should be a hidden dotfile")

    def test_stale_window_is_short(self):
        # The freshness window must be small (a couple seconds) so an off camera
        # falls back to the placeholder quickly — not a long-lived frozen frame.
        self.assertGreater(self.mod.CAMERA_PREVIEW_STALE_S, 0.0)
        self.assertLessEqual(self.mod.CAMERA_PREVIEW_STALE_S, 5.0)


if __name__ == "__main__":
    unittest.main()
