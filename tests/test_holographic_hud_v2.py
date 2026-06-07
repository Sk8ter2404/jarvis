"""Crash-guard + dismiss-path tests for ``hud/holographic_hud_v2.py``.

WHY THIS EXISTS
  The holographic v2 reactor is a frameless, click-through PyQt6 overlay: the
  user cannot click it to dismiss it, and an unhandled exception inside the
  QTimer slot can abort the Qt event loop and leave a dead fullscreen layer on
  screen. The overnight self-upgrade added three safety nets that previously
  shipped with NO test:

    1. ``_control_says_off()`` — a *separate* control file (never the shared
       hud_state.json) whose ``{"mode":"off"}`` retires the overlay, so a skill
       can dismiss the click-through window without racing the main process's
       continuous rewrites of the canonical snapshot.
    2. ``ORPHAN_MAX_LIFETIME_S`` — when launched with ``--parent-pid 0`` (no real
       parent to track) the overlay self-exits after the cap so it can never
       become an unkillable fullscreen layer.
    3. Guarded float parses in ``refresh_data`` — a non-numeric ``tts_amplitude``
       / ``mic_level`` in the shared state file degrades to the last-known value
       instead of raising out of the timer slot.

  ``refresh_data`` returning False is the single close signal the owning window
  acts on, so these tests assert the close DECISION directly: they never build a
  QWidget and never spin a Qt loop.

ISOLATION
  PyQt6 is blocked for the load (the module's own ``except ImportError`` stub
  path makes the Qt names harmless and ``_HAS_PYQT6`` False), so the suite runs
  identically on the headless Linux CI runner. ``refresh_data`` is called as an
  unbound method against a tiny ``SimpleNamespace`` stand-in carrying only the
  fields the close-decision branch reads. ``HUD_STATE_FILE`` / ``CONTROL_FILE``
  are redirected to a per-test temp dir so no real project file is touched.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud",
)


def _load_hud_no_pyqt(testcase, filename, mod_name):
    """Load a HUD source with PyQt6 import blocked so the module takes its
    graceful-degrade path (``_HAS_PYQT6`` False, Qt names stubbed). Restores
    sys.modules + the real importer on cleanup."""
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


class _HoloV2Base(unittest.TestCase):
    def setUp(self):
        self.mod = _load_hud_no_pyqt(self, "holographic_hud_v2.py",
                                     "_holo_v2_under_test")
        self.tmp = tempfile.mkdtemp(prefix="holo_v2_test_")
        self.addCleanup(self._cleanup_tmp)
        self.hud_state = os.path.join(self.tmp, "hud_state.json")
        self.control = os.path.join(self.tmp, "holographic_hud_v2_state.json")
        self.mod.HUD_STATE_FILE = self.hud_state
        self.mod.CONTROL_FILE = self.control

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

    def _write(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _scene(self, *, parent_pid=4321, started_at=10_000.0,
               tts_amp=0.0, mic_level=0.0):
        """A stand-in carrying only the fields refresh_data's close-decision and
        guarded-parse branches touch — no QGraphicsScene is constructed."""
        return types.SimpleNamespace(
            parent_pid=parent_pid,
            _started_at=started_at,
            tts_amp=tts_amp,
            mic_level=mic_level,
            state="idle",
            active_action="",
            recent_action="",
            intent_tag="",
            last_spoken="",
            transcripts=None,
            cpu_pct=0.0,
            ram_pct=0.0,
            frame=0,
            update=lambda: None,
        )

    def _refresh(self, scene, *, now=10_000.0):
        """Call refresh_data as an unbound method with the parent forced alive
        and psutil absent so only the file-driven close decision is exercised."""
        with mock.patch.object(self.mod, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.mod, "_HAS_PSUTIL", False), \
                mock.patch.object(self.mod.time, "time", return_value=now):
            return self.mod.ArcReactorScene.refresh_data(scene)


class ControlSaysOffTests(_HoloV2Base):
    def test_off_reads_dedicated_control_file_not_hud_state(self):
        # mode=off written to hud_state.json must NOT dismiss — the control file
        # is a separate file so a skill never races the main snapshot writer.
        self._write(self.hud_state, {"mode": "off"})
        self.assertFalse(self.mod._control_says_off())

    def test_off_in_control_file_is_seen(self):
        self._write(self.control, {"mode": "off"})
        self.assertTrue(self.mod._control_says_off())

    def test_off_is_case_insensitive(self):
        self._write(self.control, {"mode": "OFF"})
        self.assertTrue(self.mod._control_says_off())

    def test_missing_control_file_is_not_off(self):
        self.assertFalse(self.mod._control_says_off())

    def test_other_mode_is_not_off(self):
        self._write(self.control, {"mode": "on"})
        self.assertFalse(self.mod._control_says_off())


class RefreshCloseDecisionTests(_HoloV2Base):
    def test_parent_dead_closes(self):
        s = self._scene()
        with mock.patch.object(self.mod, "_is_parent_alive", return_value=False):
            self.assertFalse(self.mod.ArcReactorScene.refresh_data(s))

    def test_control_off_closes(self):
        s = self._scene()
        self._write(self.control, {"mode": "off"})
        self.assertFalse(self._refresh(s))

    def test_alive_and_no_control_keeps_running(self):
        s = self._scene()
        self._write(self.hud_state, {"state": "Listening"})
        self.assertTrue(self._refresh(s))

    def test_orphan_under_cap_keeps_running(self):
        # parent_pid<=0 (no real parent) but still inside the lifetime cap.
        s = self._scene(parent_pid=0, started_at=10_000.0)
        within = 10_000.0 + self.mod.ORPHAN_MAX_LIFETIME_S - 1.0
        self.assertTrue(self._refresh(s, now=within))

    def test_orphan_past_cap_closes(self):
        # parent_pid<=0 and past the cap → self-exit so a parentless overlay
        # can't become an unkillable fullscreen layer.
        s = self._scene(parent_pid=0, started_at=10_000.0)
        past = 10_000.0 + self.mod.ORPHAN_MAX_LIFETIME_S + 1.0
        self.assertFalse(self._refresh(s, now=past))

    def test_real_parent_never_hits_orphan_cap(self):
        # A supervised HUD (real positive pid) stays up well past the cap.
        s = self._scene(parent_pid=4321, started_at=0.0)
        way_past = self.mod.ORPHAN_MAX_LIFETIME_S * 10.0
        self.assertTrue(self._refresh(s, now=way_past))

    def test_orphan_cap_is_thirty_minutes(self):
        self.assertEqual(self.mod.ORPHAN_MAX_LIFETIME_S, 1800.0)


class RefreshGuardedFloatTests(_HoloV2Base):
    def test_non_numeric_amp_does_not_raise_and_keeps_last_value(self):
        # A bad tts_amplitude must not raise out of the timer slot; the channel
        # degrades to its last-known value instead of crashing the overlay.
        s = self._scene(tts_amp=0.42)
        self._write(self.hud_state, {"tts_amplitude": "loud", "mic_level": 0.5})
        self.assertTrue(self._refresh(s))
        self.assertEqual(s.tts_amp, 0.42)     # unchanged — bad value swallowed
        self.assertEqual(s.mic_level, 0.5)    # good value still applied

    def test_non_numeric_mic_keeps_last_value(self):
        # A truthy non-numeric value (``or 0.0`` doesn't replace it) makes
        # float() raise → the except keeps the last-known channel value.
        s = self._scene(mic_level=0.30)
        self._write(self.hud_state, {"tts_amplitude": 0.8, "mic_level": "quiet"})
        self.assertTrue(self._refresh(s))
        self.assertEqual(s.tts_amp, 0.8)
        self.assertEqual(s.mic_level, 0.30)   # bad value swallowed, kept

    def test_valid_floats_are_applied(self):
        s = self._scene()
        self._write(self.hud_state, {"tts_amplitude": 0.6, "mic_level": 0.9})
        self.assertTrue(self._refresh(s))
        self.assertEqual(s.tts_amp, 0.6)
        self.assertEqual(s.mic_level, 0.9)


if __name__ == "__main__":
    unittest.main()
