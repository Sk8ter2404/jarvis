"""Logic tests for skills/sh_ring.py (Ring camera/doorbell controller).

Wraps the optional `ring_doorbell` library. Coverage:
  * is_available gating on lib presence AND a cached token,
  * graceful degradation (no client) for list/get/set,
  * set_state's capability-gated apply (lights/siren), and the
    "no requested controls landed" path,
  * the ring_authorize action's argument parsing + CLI-hint fallback,
  * device enumeration / name matching against fake device objects.
The library and the network are always mocked.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _FakeRingDevice:
    """Fake stickup-cam / floodlight exposing `lights` and `siren`."""
    def __init__(self, name="Front Door", did="dev1", has_lights=True,
                 has_siren=True):
        self.name = name
        self.id = did
        self.battery_life = 88
        self.connection_status = "online"
        if has_lights:
            self.lights = "off"
        if has_siren:
            self.siren = "off"


class RingAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_ring")

    def test_unavailable_without_lib(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_unavailable_without_token(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_token", return_value={}):
            self.assertFalse(self.mod.is_available())

    def test_available_with_lib_and_token(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_token",
                               return_value={"refresh_token": "x"}):
            self.assertTrue(self.mod.is_available())


class RingDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_ring")

    def test_list_devices_empty_without_client(self):
        with mock.patch.object(self.mod, "_get_ring", return_value=None):
            self.assertEqual(self.mod.list_devices(), [])

    def test_get_state_errors_without_client(self):
        with mock.patch.object(self.mod, "_get_ring", return_value=None):
            res = self.mod.get_state({"name": "Front Door"})
        self.assertIn("not authorized", res["error"])

    def test_set_state_errors_without_client_mentions_authorize(self):
        with mock.patch.object(self.mod, "_get_ring", return_value=None):
            res = self.mod.set_state({"name": "Front Door"}, on=True)
        self.assertIn("not authorized", res["error"])
        self.assertIn("ring_authorize", res["error"])

    def test_list_action_counts(self):
        with mock.patch.object(self.mod, "list_devices",
                               return_value=[{"name": "A"}, {"name": "B"}]):
            out = self.actions["ring_list_devices"]("")
        self.assertIn("2 Ring device", out)


class RingAuthorizeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_ring")

    def test_authorize_without_lib(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=None):
            out = self.actions["ring_authorize"]("")
        self.assertIn("not installed", out)

    def test_authorize_no_creds_returns_cli_hint(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            out = self.actions["ring_authorize"]("")
        self.assertIn("interactive terminal", out)

    def test_authorize_inline_creds_calls_fetch(self):
        fetch = mock.Mock(return_value="Ring authorized, sir — token persisted.")
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_do_fetch_token", fetch):
            out = self.actions["ring_authorize"]("me@x.com|pw")
        fetch.assert_called_once_with("me@x.com", "pw", "")
        self.assertIn("authorized", out)

    def test_authorize_2fa_required_message(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_do_fetch_token",
                               return_value="2FA_REQUIRED: need code"):
            out = self.actions["ring_authorize"]("me@x.com|pw")
        self.assertIn("2FA code", out)


class RingSetStateApplyTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_ring")

    def test_lights_on_applied(self):
        dev = _FakeRingDevice()
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Front Door"}, on=True)
        self.assertEqual(res["applied"]["lights"], True)
        self.assertEqual(dev.lights, "on")

    def test_siren_on_applied(self):
        dev = _FakeRingDevice()
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Front Door"}, siren=True)
        self.assertEqual(res["applied"]["siren"], True)
        self.assertEqual(dev.siren, "on")

    def test_no_supported_controls_returns_informative_error(self):
        # A device with neither lights nor siren → nothing lands.
        dev = _FakeRingDevice(has_lights=False, has_siren=False)
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Chime"}, on=True, siren=True)
        self.assertIn("doesn't expose", res["error"])
        self.assertIn("on", res["requested"])

    def test_device_not_found(self):
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=None):
            res = self.mod.set_state({"name": "Ghost"}, on=True)
        self.assertIn("not found", res["error"])


class RingEnumerationTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_ring")

    def test_enumerate_flattens_device_dict(self):
        d1 = _FakeRingDevice(name="Front Door", did="a")
        d2 = _FakeRingDevice(name="Backyard", did="b")
        fake_ring = mock.Mock()
        fake_ring.devices.return_value = {"doorbots": [d1], "stickup_cams": [d2]}
        out = self.mod._enumerate(fake_ring)
        self.assertEqual(set(out.keys()), {"a", "b"})

    def test_match_by_name_case_insensitive(self):
        d1 = _FakeRingDevice(name="Front Door", did="a")
        fake_ring = mock.Mock()
        fake_ring.devices.return_value = {"doorbots": [d1]}
        found = self.mod._match(fake_ring, {"name": "front door"})
        self.assertIs(found, d1)


if __name__ == "__main__":
    unittest.main()
