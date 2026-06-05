"""Unit tests for audio/audio_switch.py — the render-device filter and the
headset power-transition state machine. No real COM / no device mutation: the
device functions are mocked, so this runs on the CI light tier too."""
import unittest
from unittest import mock

from audio import audio_switch as A


class FindActiveTests(unittest.TestCase):
    def test_render_only_skips_capture_and_matches_fragment(self):
        rows = [
            # capture (mic) endpoint — id prefix {0.0.1. — must be SKIPPED for output
            ("{0.0.1.00000000}.{cap}", "Headset Microphone (3- CORSAIR VOID ELITE)", "Active"),
            # render (earphone) endpoint — id prefix {0.0.0. — the match we want
            ("{0.0.0.00000000}.{ren}", "Headset Earphone (3- CORSAIR VOID ELITE)", "Active"),
            ("{0.0.0.00000000}.{spk}", "Speakers (Realtek USB2.0 Audio)", "Active"),
        ]
        with mock.patch.object(A, "list_render", return_value=rows):
            got = A.find_active("corsair void elite")
        self.assertEqual(got[0], "{0.0.0.00000000}.{ren}")
        self.assertIn("Earphone", got[1])

    def test_inactive_headset_not_found(self):
        rows = [("{0.0.0.00000000}.{ren}", "Headset Earphone (CORSAIR VOID ELITE)", "NotPresent")]
        with mock.patch.object(A, "list_render", return_value=rows):
            self.assertIsNone(A.find_active("corsair void elite"))

    def test_empty_fragment_is_none(self):
        self.assertIsNone(A.find_active(""))


class SetDefaultGuardTests(unittest.TestCase):
    def test_empty_id_returns_false(self):
        self.assertFalse(A.set_default_render(""))


class TickTransitionTests(unittest.TestCase):
    def _sw(self):
        return A.AudioAutoSwitch("VOID ELITE", "Realtek", poll_s=1.0, announce=lambda m: None)

    def test_off_to_on_switches_to_headset_and_remembers_prior(self):
        sw = self._sw()
        with mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="SPK"), \
             mock.patch.object(A, "set_default_render", return_value=True) as setd:
            label = sw.tick(was_on=False)
        self.assertEqual(label, "to_headset")
        setd.assert_called_once_with("HS")
        self.assertEqual(sw._prior_default, "SPK")

    def test_on_to_off_restores_remembered_prior(self):
        sw = self._sw()
        sw._prior_default = "SPK"
        with mock.patch.object(A, "find_active", return_value=None), \
             mock.patch.object(A, "set_default_render", return_value=True) as setd:
            label = sw.tick(was_on=True)
        self.assertEqual(label, "away")
        setd.assert_called_once_with("SPK")
        self.assertIsNone(sw._prior_default)

    def test_on_to_off_uses_fallback_when_no_prior(self):
        sw = self._sw()
        sw._prior_default = None

        def fa(frag, render_only=True):
            return None if "void" in frag.lower() else ("SPK", "Speakers")

        with mock.patch.object(A, "find_active", side_effect=fa), \
             mock.patch.object(A, "set_default_render", return_value=True) as setd:
            label = sw.tick(was_on=True)
        self.assertEqual(label, "away")
        setd.assert_called_once_with("SPK")

    def test_already_default_is_noop(self):
        sw = self._sw()
        with mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="HS"), \
             mock.patch.object(A, "set_default_render") as setd:
            label = sw.tick(was_on=True)          # on->on, headset already default
        self.assertIsNone(label)
        setd.assert_not_called()


class StatusAndBatteryTests(unittest.TestCase):
    def _sw(self, **kw):
        return A.AudioAutoSwitch("VOID ELITE", "Realtek", announce=kw.get("announce", lambda m: None))

    def test_status_includes_battery_when_on(self):
        sw = self._sw()
        with mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(sw, "battery_pct", return_value=72.0):
            s = sw.status()
        self.assertIn("ON", s)
        self.assertIn("72% battery", s)

    def test_status_off_omits_battery(self):
        sw = self._sw()
        with mock.patch.object(A, "find_active", return_value=None), \
             mock.patch.object(sw, "battery_pct", return_value=None):
            s = sw.status()
        self.assertIn("off", s.lower())
        self.assertNotIn("battery", s.lower())

    def test_low_battery_warns_once_then_rearms_after_recharge(self):
        msgs = []
        sw = self._sw(announce=msgs.append)
        with mock.patch.object(sw, "battery_pct", return_value=10.0):
            sw._check_low_battery()
            sw._check_low_battery()                 # still low -> only ONE warning
        self.assertEqual(len(msgs), 1)
        self.assertIn("low", msgs[0].lower())
        with mock.patch.object(sw, "battery_pct", return_value=80.0):
            sw._check_low_battery()                 # recharged -> re-arm
        with mock.patch.object(sw, "battery_pct", return_value=8.0):
            sw._check_low_battery()                 # low again -> warns again
        self.assertEqual(len(msgs), 2)

    def test_low_battery_no_hwinfo_is_silent(self):
        msgs = []
        sw = self._sw(announce=msgs.append)
        with mock.patch.object(sw, "battery_pct", return_value=None):
            sw._check_low_battery()
        self.assertEqual(msgs, [])


if __name__ == "__main__":
    unittest.main()
