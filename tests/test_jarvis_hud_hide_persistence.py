"""Hide / show persistence tests for ``hud/jarvis_hud.py``.

WHY THIS EXISTS
  The user-driven "Hide HUD" menu item used to persist its choice by doing a
  read-modify-write of the SHARED ``hud_state.json``: read the snapshot, set
  ``visible=False``, write it back. That snapshot is rewritten continuously by
  the main bobert_companion process, so the menu hide both (a) raced the main
  writer (torn write of the canonical state) and (b) was instantly clobbered on
  the next tick — the hide was unsafe AND ineffective.

  The fix gives this HUD its OWN control file (``jarvis_hud_control.json``),
  written via the shared atomic writer, read each tick with precedence. An
  explicit JARVIS show (``visible`` rising False→True in hud_state.json) clears
  the user-hide too, honouring the menu's "JARVIS will re-show it on request"
  promise. These three behaviours had no test; this adds one.

  hide              → control file ``{"hidden": true}``; hud_state.json NOT
                      touched (the bug was writing the hide there).
  show persists     → the control file survives and is read back each tick.
  JARVIS-show clears→ ``visible`` False→True clears ``_user_hidden`` and
                      rewrites the control file to ``{"hidden": false}``.

ISOLATION
  jarvis_hud is a tkinter overlay; constructing a Tk root needs a display, which
  the headless Linux CI runner does not have. So no Tk root is ever built: the
  stand-in ``self`` is a real ``HUD`` instance created via ``HUD.__new__(HUD)``
  (``__init__`` skipped → no Tk), with ``root`` / ``canvas`` set to MagicMocks
  that absorb every Tk call. The real ``_hide_via_menu`` / ``_tick_body`` and
  their genuine geometry + draw helpers all run; the production visibility/clear
  branch under test therefore executes for real, and the draw calls land on the
  mock canvas. ``STATE_FILE`` and ``CONTROL_FILE`` are redirected to a per-test
  temp dir, so no real project file is read or written.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud",
)

# Instance attributes the post-visibility render reads. __init__ is skipped (no
# Tk), so the show-clears path — which proceeds into the real render against a
# mock canvas — needs these scalar animation accumulators pre-seeded.
_RENDER_ATTRS = dict(
    last_cpu=0.0, last_ram=0.0, last_mic=0.0, last_amp=0.0,
    _phase=0.0, _halo_phase=0.0,
    _action_reveal_frame=0, _action_at_start=0.0,
    _focused_window="", role="prod",
)


def _load_jarvis_hud(testcase):
    """Load hud/jarvis_hud.py under a synthetic module name with STATE_FILE /
    CONTROL_FILE redirected to a fresh temp dir. tkinter/pygetwindow are NOT
    blocked — the module imports them at top level (stdlib tk imports fine on
    the runner) but we never construct a Tk root."""
    path = os.path.join(_HUD_DIR, "jarvis_hud.py")
    mod_name = "_jarvis_hud_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    testcase.addCleanup(lambda: sys.modules.pop(mod_name, None))
    spec.loader.exec_module(module)
    return module


class _HudHideBase(unittest.TestCase):
    def setUp(self):
        self.mod = _load_jarvis_hud(self)
        self.tmp = tempfile.mkdtemp(prefix="jarvis_hud_hide_test_")
        self.addCleanup(self._cleanup_tmp)
        self.state_file = os.path.join(self.tmp, "hud_state.json")
        self.control_file = os.path.join(self.tmp, "jarvis_hud_control.json")
        self.mod.STATE_FILE = self.state_file
        self.mod.CONTROL_FILE = self.control_file

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

    def _write_state(self, data):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _fake_self(self, **over):
        # Real HUD instance WITHOUT __init__ → no Tk root is created, but every
        # real method (geometry/draw helpers, _hide_via_menu, _tick_body)
        # resolves normally. root/canvas are mocks that absorb Tk calls.
        s = self.mod.HUD.__new__(self.mod.HUD)
        s.parent_pid = 4321
        s._hidden = False
        s._user_hidden = False
        s._prev_state_visible = None
        s._closing = False
        s.frame = 0
        s.root = mock.MagicMock()
        s.canvas = mock.MagicMock()
        for k, v in _RENDER_ATTRS.items():
            setattr(s, k, v)
        for k, v in over.items():
            setattr(s, k, v)
        return s

    def _tick(self, scene):
        """Run the real _tick_body once with the parent forced alive + psutil
        absent. Returns the body's delay value (ms) or None."""
        with mock.patch.object(self.mod, "_is_parent_alive",
                               return_value=True), \
                mock.patch.object(self.mod, "_HAS_PSUTIL", False), \
                mock.patch.object(self.mod, "_HAS_GW", False):
            return self.mod.HUD._tick_body(scene)


class HideViaMenuTests(_HudHideBase):
    def test_hide_writes_control_file_not_state_file(self):
        s = self._fake_self()
        self.mod.HUD._hide_via_menu(s)
        # The hide is persisted in the HUD's OWN control file…
        self.assertEqual(self._read_json(self.control_file), {"hidden": True})
        # …and the shared snapshot is NOT touched (that was the racy bug).
        self.assertFalse(os.path.exists(self.state_file))

    def test_hide_sets_flags_and_withdraws(self):
        s = self._fake_self()
        self.mod.HUD._hide_via_menu(s)
        self.assertTrue(s._user_hidden)
        self.assertTrue(s._hidden)
        s.root.withdraw.assert_called_once()

    def test_hide_does_not_clobber_existing_state_fields(self):
        # Even with a populated hud_state.json present, the hide must leave it
        # byte-for-byte alone (no read-modify-write of the canonical snapshot).
        self._write_state({"state": "Listening", "visible": True,
                           "mic_level": 0.5})
        before = self._read_json(self.state_file)
        s = self._fake_self()
        self.mod.HUD._hide_via_menu(s)
        self.assertEqual(self._read_json(self.state_file), before)


class HidePersistsAcrossTickTests(_HudHideBase):
    def test_user_hide_survives_a_tick(self):
        # control file says hidden; state says visible. The user-hide wins and
        # the HUD stays withdrawn (slow 500 ms cadence) — i.e. it persists.
        self.mod._write_hud_control({"hidden": True})
        self._write_state({"visible": True, "state": "Idle"})
        s = self._fake_self()
        delay = self._tick(s)
        self.assertTrue(s._user_hidden)
        self.assertTrue(s._hidden)
        self.assertEqual(delay, 500)        # hidden slow-tick path
        s.root.withdraw.assert_called_once()

    def test_control_file_read_each_tick(self):
        # No menu hide this session, but the control file already holds a hide
        # (e.g. from a prior run / another process) → picked up on the tick.
        self.mod._write_hud_control({"hidden": True})
        self._write_state({"visible": True})
        s = self._fake_self(_prev_state_visible=True)  # no False→True edge
        self._tick(s)
        self.assertTrue(s._user_hidden)


class JarvisShowClearsUserHideTests(_HudHideBase):
    def test_explicit_show_clears_user_hide_and_rewrites_control(self):
        # Arrange: user hid the HUD (control file hidden) AND the previous tick
        # saw visible=False, so this tick is the False→True show edge.
        self.mod._write_hud_control({"hidden": True})
        self._write_state({"visible": True, "state": "Idle"})
        s = self._fake_self(_hidden=True, _user_hidden=True,
                            _prev_state_visible=False)
        self._tick(s)
        # The explicit JARVIS show clears the user-hide…
        self.assertFalse(s._user_hidden)
        # …and the control file is rewritten to the un-hidden state.
        self.assertEqual(self._read_json(self.control_file), {"hidden": False})
        # …and the window is restored.
        s.root.deiconify.assert_called_once()

    def test_show_without_prior_hide_edge_does_not_clear(self):
        # visible=True but the previous tick was ALSO visible (no False→True
        # edge): a steady-state visible tick must NOT clear a user-hide.
        self.mod._write_hud_control({"hidden": True})
        self._write_state({"visible": True})
        s = self._fake_self(_hidden=True, _user_hidden=True,
                            _prev_state_visible=True)
        self._tick(s)
        self.assertTrue(s._user_hidden)             # still hidden
        self.assertEqual(self._read_json(self.control_file), {"hidden": True})

    def test_state_hide_then_show_without_user_hide_round_trips(self):
        # JARVIS hide_hud (visible=False) withdraws; a later show (visible=True)
        # deiconifies — the plain main-script visibility channel still works and
        # leaves the (empty) control file alone.
        self._write_state({"visible": False})
        s = self._fake_self(_prev_state_visible=True)
        d1 = self._tick(s)
        self.assertTrue(s._hidden)
        self.assertEqual(d1, 500)
        s.root.withdraw.assert_called_once()

        self._write_state({"visible": True})
        d2 = self._tick(s)
        self.assertFalse(s._hidden)
        self.assertNotEqual(d2, 500)        # back to the normal render cadence
        s.root.deiconify.assert_called_once()
        # No user-hide was ever set → control file stays absent.
        self.assertFalse(os.path.exists(self.control_file))


if __name__ == "__main__":
    unittest.main()
