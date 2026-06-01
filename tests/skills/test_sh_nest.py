"""Logic tests for skills/sh_nest.py (Nest SDM thermostat/camera controller).

Wraps the optional `google-nest-sdm` library behind a non-trivial OAuth setup.
The dominant risk is graceful degradation: with the lib absent OR the OAuth
config incomplete, every public path returns a clean error/empty result and
never raises. We also test:
  * is_available gating on both lib presence AND all four config keys,
  * get_state's Celsius→Fahrenheit translation and trait extraction,
  * set_state's on/off → HEATCOOL/OFF mode mapping,
  * the authorize action's missing-config guard.
The asyncio client is always mocked; no event loop / network is used.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class NestAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_nest")

    def test_unavailable_without_lib(self):
        with mock.patch.object(self.mod, "_sdm", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_unavailable_with_lib_but_no_config(self):
        with mock.patch.object(self.mod, "_sdm", return_value=("a", "b", "c")), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            self.assertFalse(self.mod.is_available())

    def test_available_with_lib_and_full_config(self):
        full = {"project_id": "p", "client_id": "c",
                "client_secret": "s", "refresh_token": "r"}
        with mock.patch.object(self.mod, "_sdm", return_value=("a", "b", "c")), \
             mock.patch.object(self.mod, "_read_config", return_value=full):
            self.assertTrue(self.mod.is_available())


class NestDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_nest")

    def test_list_devices_empty_without_client(self):
        with mock.patch.object(self.mod, "_sdm", return_value=None), \
             mock.patch.object(self.mod, "_get_client", return_value=None):
            self.assertEqual(self.mod.list_devices(), [])

    def test_get_state_errors_without_client(self):
        with mock.patch.object(self.mod, "_get_client", return_value=None):
            res = self.mod.get_state({"name": "Hallway"})
        self.assertIn("not initialized", res["error"])

    def test_set_state_errors_without_client_mentions_authorize(self):
        with mock.patch.object(self.mod, "_get_client", return_value=None):
            res = self.mod.set_state({"name": "Hallway"}, temperature=70)
        self.assertIn("not initialized", res["error"])
        self.assertIn("sh_nest_authorize", res["error"])

    def test_nest_list_action_counts_devices(self):
        with mock.patch.object(self.mod, "list_devices",
                               return_value=[{"name": "A"}, {"name": "B"}]):
            out = self.actions["nest_list_devices"]("")
        self.assertIn("2 Nest device", out)

    def test_authorize_requires_config(self):
        with mock.patch.object(self.mod, "_read_config", return_value={}):
            out = self.actions["nest_authorize"]("")
        self.assertIn("project_id", out)
        self.assertIn("client_id", out)


class NestGetStateTranslationTests(unittest.TestCase):
    """get_state pulls SDM traits and converts ambient °C → °F. We stub the
    client + _run_async so a canned trait payload flows through."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_nest")

    def test_celsius_to_fahrenheit_and_mode(self):
        fake_client = ("client", "project", "session")
        traits = {
            "sdm.devices.traits.Temperature": {"ambientTemperatureCelsius": 20.0},
            "sdm.devices.traits.ThermostatMode": {"mode": "HEAT"},
            "sdm.devices.traits.ThermostatTemperatureSetpoint":
                {"heatCelsius": 21.0, "coolCelsius": 25.0},
        }
        # Run the coroutine for real (against a fake client) so we don't leave
        # an un-awaited coroutine — _go() just calls client.request("get", nid).
        class _Client:
            async def request(self, method, path, **kw):
                return {"traits": traits}

        fake_client = (_Client(), "project", "session")

        def _run(coro):
            import asyncio
            return asyncio.run(coro)

        with mock.patch.object(self.mod, "_get_client", return_value=fake_client), \
             mock.patch.object(self.mod, "_device_id", return_value="enterprises/x/devices/y"), \
             mock.patch.object(self.mod, "_run_async", side_effect=_run):
            st = self.mod.get_state({"name": "Hallway"})
        self.assertEqual(st["actual_c"], 20.0)
        self.assertEqual(st["actual_f"], 68.0)          # 20°C == 68°F
        self.assertEqual(st["mode"], "HEAT")
        self.assertEqual(st["heat_set_c"], 21.0)

    def test_get_state_device_not_found(self):
        fake_client = ("client", "project", "session")
        with mock.patch.object(self.mod, "_get_client", return_value=fake_client), \
             mock.patch.object(self.mod, "_device_id", return_value=None):
            res = self.mod.get_state({"name": "Ghost"})
        self.assertIn("not found", res["error"])


class NestSetStateModeMappingTests(unittest.TestCase):
    """set_state maps on/off → HEATCOOL/OFF and issues SetMode/SetHeat commands.
    We capture the coroutines via a _run_async stub that just runs them with a
    fake client whose .request records calls."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_nest")

    def _run_with_recorder(self, **set_kwargs):
        recorded = []

        class _Client:
            async def request(self, method, path, **kw):
                recorded.append((method, path, kw))
                return {}

        fake = (_Client(), "project", "session")

        def _run(coro):
            import asyncio
            return asyncio.run(coro)

        with mock.patch.object(self.mod, "_get_client", return_value=fake), \
             mock.patch.object(self.mod, "_device_id",
                               return_value="enterprises/x/devices/y"), \
             mock.patch.object(self.mod, "_run_async", side_effect=_run):
            res = self.mod.set_state({"name": "Hallway"}, **set_kwargs)
        return res, recorded

    def test_on_maps_to_heatcool(self):
        res, recorded = self._run_with_recorder(on=True)
        self.assertEqual(res["applied"]["mode"], "HEATCOOL")
        # A SetMode command was issued with mode HEATCOOL.
        bodies = [kw.get("json", {}) for _m, _p, kw in recorded]
        self.assertTrue(any(b.get("params", {}).get("mode") == "HEATCOOL"
                            for b in bodies))

    def test_off_maps_to_off(self):
        res, _recorded = self._run_with_recorder(on=False)
        self.assertEqual(res["applied"]["mode"], "OFF")

    def test_temperature_recorded_in_fahrenheit_field(self):
        res, recorded = self._run_with_recorder(temperature=70)
        self.assertEqual(res["applied"]["temperature"], 70)
        # The SetHeat command converts to Celsius in the payload.
        heat_bodies = [kw.get("json", {}) for _m, _p, kw in recorded
                       if "SetHeat" in (kw.get("json", {}) or {}).get("command", "")]
        self.assertTrue(heat_bodies)
        c = heat_bodies[0]["params"]["heatCelsius"]
        self.assertAlmostEqual(c, (70 - 32) * 5 / 9, places=4)


if __name__ == "__main__":
    unittest.main()
