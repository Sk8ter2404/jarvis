"""Saved-position file name for ``hud/workshop_hud.py``.

WHY THIS EXISTS
  The workshop HUD's on/off CONTROL file is ``<project>/workshop_hud_state.json``
  (the holographic_overlay skill writes ``{"mode": "off"}`` there to retire the
  widget). Its saved drag position used the SAME name, one folder down
  (``data/workshop_hud_state.json``). Two different files with one name is an
  invitation to read or clear the wrong one (GUI_REVIEW B8), so the position
  file is now ``data/workshop_hud_geometry.json``. A position saved under the
  old name is still restored until the first save under the new one.

ISOLATION
  PyQt6 is blocked for the load (the module's ``except ImportError`` stub path
  keeps every Qt name harmless), so this runs on the headless CI runner. The
  two persistence methods are called unbound on a tiny stand-in ``self``; no
  QWidget is ever built. Every geometry path is redirected into a temp dir, so
  no real project file is read or written.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud",
)


def _load_workshop_hud(testcase):
    path = os.path.join(_HUD_DIR, "workshop_hud.py")
    mod_name = "_workshop_hud_geometry_under_test"
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
    testcase.assertFalse(module._HAS_PYQT6)
    return module


class WorkshopHudGeometryFileTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_workshop_hud(self)
        self.real_geom_name = os.path.basename(self.mod.GEOM_STATE_FILE)
        self.tmp = tempfile.mkdtemp(prefix="workshop_hud_geom_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # Same file names as production, inside a throwaway data dir.
        for attr, value in (
                ("GEOM_STATE_DIR", self.tmp),
                ("GEOM_STATE_FILE", os.path.join(self.tmp, self.real_geom_name)),
                ("LEGACY_GEOM_STATE_FILE",
                 os.path.join(self.tmp, "workshop_hud_state.json"))):
            p = mock.patch.object(self.mod, attr, value, create=True)
            p.start()
            self.addCleanup(p.stop)
        self.win = types.SimpleNamespace(x=lambda: 120, y=lambda: -1400)

    def _write(self, name, data):
        with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_position_file_has_its_own_name(self):
        self.assertEqual(self.real_geom_name, "workshop_hud_geometry.json")
        self.assertNotEqual(self.real_geom_name,
                            os.path.basename(self.mod.CONTROL_FILE))

    def test_save_writes_the_geometry_file_only(self):
        self.mod.WorkshopHudWindow._save_geometry(self.win)
        self.assertEqual(sorted(os.listdir(self.tmp)),
                         ["workshop_hud_geometry.json"])
        with open(os.path.join(self.tmp, "workshop_hud_geometry.json"),
                  encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"x": 120, "y": -1400})

    def test_save_then_load_round_trips(self):
        self.mod.WorkshopHudWindow._save_geometry(self.win)
        self.assertEqual(
            self.mod.WorkshopHudWindow._load_persisted_geometry(self.win),
            (120, -1400))

    def test_position_saved_under_the_old_name_is_still_restored(self):
        self._write("workshop_hud_state.json", {"x": 10, "y": -20})
        self.assertEqual(
            self.mod.WorkshopHudWindow._load_persisted_geometry(self.win),
            (10, -20))

    def test_new_file_wins_over_the_old_one(self):
        self._write("workshop_hud_state.json", {"x": 10, "y": -20})
        self._write("workshop_hud_geometry.json", {"x": 30, "y": -40})
        self.assertEqual(
            self.mod.WorkshopHudWindow._load_persisted_geometry(self.win),
            (30, -40))

    def test_no_saved_position(self):
        self.assertEqual(
            self.mod.WorkshopHudWindow._load_persisted_geometry(self.win),
            (None, None))


if __name__ == "__main__":
    unittest.main()
