"""core.screen_text presses (invoke / back_button / close_tab) - 2026-10-05.

The live UIA bench (tools/vision_bench/live_uia_bench.py: our own Chrome on
a hidden desktop, synthetic pages) measured Chrome answering a UIA
InvokePattern.Invoke on a link, and on its Back button, only after ~2.0 s
- past the 0.8 s wait - while LegacyIAccessible DoDefaultAction returned in
~1 ms. The first build treated the timed-out press as "not pressed" and
then sent Alt+Left / Ctrl+W on top of it: a DOUBLE Back, or a second
closed tab. And comtypes hands back a missing pattern as a NULL pointer,
not None, which the "is None" checks let through.

Fakes only: a fake UIA object injected into core.uia_host and a fake
comtypes.gen.UIAutomationClient module - no COM.

    python -m unittest tests.test_screen_text_press
"""
from __future__ import annotations

import sys
import time
import types
import unittest
from unittest import mock

from core import screen_text as T
from core import uia_host


class _Null:
    """What comtypes returns for a missing pattern / element."""

    def __bool__(self):
        return False

    def QueryInterface(self, iface):
        raise ValueError("NULL COM pointer access")


class _Pattern:
    def __init__(self, log, kind, default_action="Jump", delay=0.0):
        self.log, self.kind = log, kind
        self.CurrentDefaultAction = default_action
        self.delay = delay

    def QueryInterface(self, iface):
        return self

    def DoDefaultAction(self):
        time.sleep(self.delay)
        self.log.append("legacy")

    def Invoke(self):
        time.sleep(self.delay)
        self.log.append("invoke")


class _El:
    def __init__(self, log, legacy=True, invoke=True, default_action="Jump",
                 delay=0.0):
        self.pats = {}
        if legacy:
            self.pats[T.PAT_LEGACY] = _Pattern(log, "legacy", default_action,
                                               delay)
        if invoke:
            self.pats[T.PAT_INVOKE] = _Pattern(log, "invoke", delay=delay)

    def GetCurrentPattern(self, pid):
        return self.pats.get(pid, _Null())


class _Root:
    def __init__(self, found):
        self.found = found

    def FindFirst(self, scope, cond):
        return self.found


class _Uia:
    def __init__(self, root):
        self.root = root

    def ElementFromHandle(self, hwnd):
        return self.root

    def CreateAndCondition(self, a, b):
        return ("and", a, b)

    def CreatePropertyCondition(self, pid, value):
        return (pid, value)


class _Base(unittest.TestCase):
    def setUp(self):
        self.log = []
        gen = types.ModuleType("comtypes.gen.UIAutomationClient")
        gen.IUIAutomationInvokePattern = object()
        gen.IUIAutomationLegacyIAccessiblePattern = object()
        pkg = types.ModuleType("comtypes.gen")
        pkg.UIAutomationClient = gen
        mods = mock.patch.dict(sys.modules, {
            "comtypes.gen": pkg, "comtypes.gen.UIAutomationClient": gen})
        mods.start()
        self.addCleanup(mods.stop)
        self.root = _Root(_Null())
        uia_host.set_backend(lambda: _Uia(self.root))
        self.addCleanup(uia_host.set_backend, None)

    def el(self, fake):
        T._snaps[-1] = [fake]
        self.addCleanup(T._snaps.pop, -1, None)
        return T.El(name="A video", ctype="Hyperlink", rect=(0, 0, 10, 10),
                    invokable=True, ref=(-1, 0))


class InvokeTests(_Base):
    def test_the_default_action_is_used_first(self):
        self.assertIs(T.invoke(self.el(_El(self.log))), True)
        self.assertEqual(self.log, ["legacy"])

    def test_no_default_action_falls_back_to_invoke(self):
        self.assertIs(T.invoke(self.el(_El(self.log, default_action=""))),
                      True)
        self.assertEqual(self.log, ["invoke"])

    def test_a_null_legacy_pattern_is_not_dereferenced(self):
        self.assertIs(T.invoke(self.el(_El(self.log, legacy=False))), True)
        self.assertEqual(self.log, ["invoke"])

    def test_nothing_pressable_is_false(self):
        self.assertIs(T.invoke(self.el(_El(self.log, legacy=False,
                                           invoke=False))), False)
        self.assertEqual(self.log, [])

    def test_a_press_that_outlives_the_wait_is_unconfirmed_not_failed(self):
        r = T.invoke(self.el(_El(self.log, delay=0.4)), timeout_s=0.05)
        self.assertIsNone(r)
        time.sleep(0.6)                     # it lands after the wait
        self.assertEqual(self.log, ["legacy"])


class BackAndCloseTabTests(_Base):
    def test_back_uses_the_default_action(self):
        self.root.found = _El(self.log)
        self.assertIs(T.back_button(101), True)
        self.assertEqual(self.log, ["legacy"])

    def test_no_back_button_is_false(self):
        self.assertIs(T.back_button(101), False)        # NULL from FindFirst

    def test_slow_back_is_unconfirmed(self):
        self.root.found = _El(self.log, delay=0.4)
        self.assertIsNone(T.back_button(101, timeout_s=0.05))
        time.sleep(0.6)

    def test_close_tab_null_tab_is_false(self):
        self.assertIs(T.close_tab(101, "A tab"), False)


class NilTests(unittest.TestCase):
    def test_nil(self):
        self.assertTrue(T._nil(None))
        self.assertTrue(T._nil(_Null()))
        self.assertFalse(T._nil(object()))


if __name__ == "__main__":
    unittest.main()
