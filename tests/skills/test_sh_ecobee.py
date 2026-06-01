"""Logic tests for skills/sh_ecobee.py (Ecobee thermostat controller).

Wraps the optional `pyecobee` library behind an interactive PIN-based OAuth
flow. Coverage:
  * is_available gating on lib presence AND a configured api_key,
  * graceful degradation (no service) for list/get/set,
  * the two-step PIN flow guards (no lib, no key, no pending auth token),
  * set_state's on/off → auto/off mode mapping + tenths-of-a-degree scaling,
  * get_state's tenths→degrees translation.
pyecobee and the network are always mocked.
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class EcobeeAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_ecobee")

    def test_unavailable_without_lib(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_unavailable_without_api_key(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            self.assertFalse(self.mod.is_available())

    def test_available_with_lib_and_key(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=object()), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}):
            self.assertTrue(self.mod.is_available())


class EcobeeDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_ecobee")

    def test_list_devices_empty_without_service(self):
        with mock.patch.object(self.mod, "_get_service", return_value=None):
            self.assertEqual(self.mod.list_devices(), [])

    def test_get_state_errors_without_service(self):
        with mock.patch.object(self.mod, "_get_service", return_value=None):
            res = self.mod.get_state({"name": "Main"})
        self.assertIn("not initialized", res["error"])

    def test_set_state_errors_without_service_mentions_authorize(self):
        with mock.patch.object(self.mod, "_get_service", return_value=None):
            res = self.mod.set_state({"name": "Main"}, temperature=70)
        self.assertIn("not initialized", res["error"])
        self.assertIn("ecobee_authorize", res["error"])

    def test_list_action_counts(self):
        with mock.patch.object(self.mod, "list_devices",
                               return_value=[{"name": "Main"}]):
            out = self.actions["ecobee_list_devices"]("")
        self.assertIn("1 Ecobee thermostat", out)


class EcobeePinFlowGuardTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_ecobee")

    def test_request_pin_without_lib(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=None):
            out = self.actions["ecobee_request_pin"]("")
        self.assertIn("not installed", out)

    def test_request_pin_without_api_key(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            out = self.actions["ecobee_request_pin"]("")
        self.assertIn("API key", out)
        self.assertIn("ecobee.com/developers", out)

    def test_complete_setup_without_pending_token(self):
        # Lib + key present, but no authorization_token cached yet.
        with mock.patch.object(self.mod, "_pyecobee", return_value=object()), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens", return_value={}):
            out = self.actions["ecobee_complete_setup"]("")
        self.assertIn("No pending authorization", out)

    def test_request_pin_happy_returns_pin(self):
        """A successful authorize() returns the PIN and persists the auth token."""
        fake_resp = mock.Mock(ecobee_pin="ABCD")
        fake_svc = mock.Mock()
        fake_svc.authorize.return_value = fake_resp
        fake_pyecobee = mock.Mock()
        fake_pyecobee.EcobeeService.return_value = fake_svc
        # ecobee_request_pin prints setup steps containing a non-ASCII arrow
        # (U+2192). On a cp1252 Windows console that raises UnicodeEncodeError,
        # so capture stdout to a StringIO (no encoding step) exactly as the
        # skill harness does for load-time prints — the action's RETURN value
        # is what we assert on.
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake_pyecobee), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_save_tokens") as save, \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.actions["ecobee_request_pin"]("")
        self.assertIn("ABCD", out)
        save.assert_called_once()  # authorization_token persisted

    def test_authorize_alias_explains_two_steps(self):
        out = self.actions["ecobee_authorize"]("")
        self.assertIn("request", out.lower())
        self.assertIn("complete setup", out.lower())


class EcobeeStateScalingTests(unittest.TestCase):
    """Ecobee temps are tenths of a degree F. get_state divides by 10; set_state
    multiplies the requested target by 10. We use a fake service + thermostat."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_ecobee")

    def test_get_state_divides_tenths(self):
        runtime = mock.Mock(actual_temperature=712, desired_cool=750,
                            desired_heat=680)
        settings = mock.Mock(hvac_mode="heat")
        thermo = mock.Mock(runtime=runtime, settings=settings)
        svc = object()
        with mock.patch.object(self.mod, "_get_service", return_value=svc), \
             mock.patch.object(self.mod, "_match_thermostat", return_value=thermo):
            st = self.mod.get_state({"name": "Main"})
        self.assertEqual(st["actual_f"], 71.2)
        self.assertEqual(st["cool_set"], 75.0)
        self.assertEqual(st["heat_set"], 68.0)
        self.assertEqual(st["mode"], "heat")

    def test_set_state_off_maps_mode_off(self):
        svc = mock.Mock()
        thermo = mock.Mock(identifier="abc123")
        fake_pyecobee = mock.Mock()
        # Selection/SelectionType just need to be constructable.
        fake_pyecobee.Selection.return_value = object()
        fake_pyecobee.SelectionType.THERMOSTATS.value = "thermostats"
        with mock.patch.object(self.mod, "_get_service", return_value=svc), \
             mock.patch.object(self.mod, "_pyecobee", return_value=fake_pyecobee), \
             mock.patch.object(self.mod, "_match_thermostat", return_value=thermo):
            res = self.mod.set_state({"name": "Main"}, on=False)
        self.assertEqual(res["applied"]["mode"], "off")
        svc.set_hvac_mode.assert_called_once()

    def test_set_state_temperature_uses_tenths(self):
        svc = mock.Mock()
        thermo = mock.Mock(identifier="abc123")
        fake_pyecobee = mock.Mock()
        fake_pyecobee.Selection.return_value = object()
        fake_pyecobee.SelectionType.THERMOSTATS.value = "thermostats"
        with mock.patch.object(self.mod, "_get_service", return_value=svc), \
             mock.patch.object(self.mod, "_pyecobee", return_value=fake_pyecobee), \
             mock.patch.object(self.mod, "_match_thermostat", return_value=thermo):
            res = self.mod.set_state({"name": "Main"}, temperature=72)
        self.assertEqual(res["applied"]["temperature"], 72)
        # set_hold called with both setpoints at 720 tenths.
        _args, kwargs = svc.set_hold.call_args
        self.assertEqual(kwargs.get("cool_hold_temp"), 720)
        self.assertEqual(kwargs.get("heat_hold_temp"), 720)


if __name__ == "__main__":
    unittest.main()
