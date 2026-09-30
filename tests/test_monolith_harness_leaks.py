"""tests/_monolith_harness.py - the leak defences added 2026-09-30.

Light tier: nothing here imports the monolith (the harness helpers work on any
module object), so CI runs it.

Found in the monolith tier, which CI skips:
  * all seven test_monolith_voice_clone tests failed in the full tier and
    passed alone: test_monolith_sec3 left bobert_companion._resolve_tts_preset
    as a MagicMock (a patch.dict of the module dict stopped after a sibling
    patch.object restores a snapshot that CONTAINS that sibling's mock), and
    every later synthesise() spoke through it with gain 2.0;
  * three module globals the harness did not restore made later tests pass
    or fail on execution order (_pa_defer_logged, _last_recording_peak) or
    will once the face-track loop writes them (_camera_last_seen).

    python -m unittest tests.test_monolith_harness_leaks
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import types
import unittest
from unittest import mock

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests import _monolith_harness as h  # noqa: E402


def _real():
    return "real"


class HealLeakedMocksTests(unittest.TestCase):
    def _module(self):
        m = types.ModuleType("fake_bobert_companion")
        m.fn = _real
        m.flag = [False]
        m.class_level = mock.Mock(name="installed-before-the-test")
        return m

    def _heal(self, m, before):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            healed = h._heal_leaked_mocks(m, before, "tests.x.Y.test_z")
        return healed, buf.getvalue()

    def test_a_function_left_as_a_mock_is_put_back_and_named(self):
        m = self._module()
        before = dict(vars(m))
        m.fn = mock.MagicMock(return_value=("amused", {"gain": 2.0}))
        healed, err = self._heal(m, before)
        self.assertIs(m.fn, _real)
        self.assertEqual(healed, ["fn"])
        self.assertIn("tests.x.Y.test_z", err)
        self.assertIn("fn", err)

    def test_a_mock_the_test_created_is_removed(self):
        m = self._module()
        before = dict(vars(m))
        m.brand_new = mock.Mock()
        healed, _ = self._heal(m, before)
        self.assertFalse(hasattr(m, "brand_new"))
        self.assertEqual(healed, ["brand_new"])

    def test_a_mock_present_when_the_test_started_is_left_alone(self):
        # A class-level patch (setUpClass) is not the test's leak.
        m = self._module()
        installed = m.class_level
        before = dict(vars(m))
        healed, err = self._heal(m, before)
        self.assertIs(m.class_level, installed)
        self.assertEqual(healed, [])
        self.assertEqual(err, "")

    def test_non_mock_changes_are_not_touched(self):
        # Scalars and containers are the tracked-names restore's business.
        m = self._module()
        before = dict(vars(m))
        m.flag[0] = True
        m.other = 42
        healed, _ = self._heal(m, before)
        self.assertEqual(healed, [])
        self.assertEqual(m.flag, [True])
        self.assertEqual(m.other, 42)

    def test_the_patch_order_leak_itself_is_healed(self):
        # The exact 2026-09-30 shape: patch.object, then patch.dict of the
        # module dict, both stopped in START order.
        m = self._module()
        before = dict(vars(m))
        patches = [mock.patch.object(m, "fn", return_value="mocked"),
                   mock.patch.dict(m.__dict__, {"flag": [True]})]
        for p in patches:
            p.start()
        for p in patches:
            p.stop()
        self.assertIsInstance(m.fn, mock.MagicMock,
                              "precondition: the leak reproduces")
        healed, _ = self._heal(m, before)
        self.assertIs(m.fn, _real)
        self.assertEqual(healed, ["fn"])

    def test_never_raises(self):
        class _Hostile:
            def __getattribute__(self, name):
                raise RuntimeError("no")
        self.assertEqual(h._heal_leaked_mocks(_Hostile(), {}, "t"), [])


class RestoreNamesTests(unittest.TestCase):
    def test_the_order_dependent_globals_are_tracked(self):
        for name in ("_pa_defer_logged", "_last_recording_peak",
                     "_camera_last_seen"):
            self.assertIn(name, h._MONOLITH_RESTORE_NAMES)

    def test_every_tracked_name_exists_in_the_monolith_source(self):
        # A tracked name the monolith no longer defines is silently skipped
        # at snapshot time - i.e. a restore that quietly stopped happening.
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        for name in ("_pa_defer_logged", "_last_recording_peak",
                     "_camera_last_seen"):
            self.assertIn(f"\n{name}", src, name)


if __name__ == "__main__":
    unittest.main()
