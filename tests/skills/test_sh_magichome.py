"""Logic tests for skills/sh_magichome.py (Magic Home / Flux LED, LAN only).

flux_led is replaced by a fake module (``_flux`` patched): no socket is ever
opened. Covers discovery parsing, on/off / brightness / colour calls, honest
failure (status read disagrees -> error), the router brand mapping, and the
router falling back to Alexa when the direct call errors.
"""
from __future__ import annotations

import types
import unittest
from unittest import mock

from core import smart_home_router as router
from tests._skill_harness import load_skill_isolated


class FakeBulb:
    """Stands in for flux_led.WifiLedBulb. ``stuck`` ignores every command."""
    instances: list = []

    def __init__(self, ip, timeout=5, stuck=False):
        self.ip = ip
        self.is_on = False
        self.rgb_unscaled = (255, 255, 255)
        self.brightness = 255
        self.calls: list = []
        self.stuck = stuck
        self.closed = False
        FakeBulb.instances.append(self)

    def update_state(self):
        self.calls.append(("update",))

    def turnOn(self):
        self.calls.append(("on",))
        if not self.stuck:
            self.is_on = True

    def turnOff(self):
        self.calls.append(("off",))
        if not self.stuck:
            self.is_on = False

    def setRgb(self, r, g, b, persist=True, brightness=None):
        self.calls.append(("rgb", r, g, b, brightness))
        if not self.stuck:
            self.rgb_unscaled = (r, g, b)
            if brightness is not None:
                self.brightness = brightness

    def close(self):
        self.closed = True


class FakeScanner:
    def __init__(self, results):
        self._results = results

    def scan(self, timeout=10, address=None):
        return self._results


def _fake_flux(results=(), stuck=False, scan_raises=False):
    mod = types.ModuleType("flux_led")
    if scan_raises:
        def _boom():
            raise OSError("no socket")
        mod.BulbScanner = _boom
    else:
        mod.BulbScanner = lambda: FakeScanner(list(results))
    mod.WifiLedBulb = lambda ip, timeout=5: FakeBulb(ip, timeout, stuck=stuck)
    return mod


def _disc(ip, ident="AABBCCDDEEFF", model="AK001-ZJ2147"):
    return types.SimpleNamespace(ipaddr=ip, id=ident, model=model)


DEV = {"name": "Magic Home strip", "brand": "Magic Home",
       "controller_skill": "sh_magichome", "lan_ip": "192.0.2.77"}


class MagicHomeBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_magichome")
        FakeBulb.instances = []
        p = mock.patch.object(self.mod, "_SETTLE_SEC", 0)
        p.start()
        self.addCleanup(p.stop)
        e = mock.patch.dict("os.environ", {"MAGICHOME_ENABLED": "1"})
        e.start()
        self.addCleanup(e.stop)

    def use(self, **kw):
        p = mock.patch.object(self.mod, "_flux", return_value=_fake_flux(**kw))
        p.start()
        self.addCleanup(p.stop)


class DiscoveryTests(MagicHomeBase):
    def test_parse_and_dedupe(self):
        self.use(results=[_disc("192.0.2.77"), _disc("192.0.2.77"),
                          _disc("192.0.2.78", ident="112233445566")])
        devs = self.mod.list_devices()
        self.assertEqual([d["lan_ip"] for d in devs], ["192.0.2.77", "192.0.2.78"])
        self.assertEqual(devs[0]["brand"], "Magic Home")
        self.assertEqual(devs[0]["type"], "light")
        self.assertIn("color", devs[0]["capabilities"])
        self.assertEqual(devs[1]["name"], "Magic Home 445566")

    def test_dict_results_accepted(self):
        self.use(results=[{"ipaddr": "192.0.2.5", "id": "X1", "model": "m"}])
        self.assertEqual(self.mod.list_devices()[0]["lan_ip"], "192.0.2.5")

    def test_scan_failure_never_raises(self):
        self.use(scan_raises=True)
        self.assertEqual(self.mod.list_devices(), [])

    def test_not_installed(self):
        with mock.patch.object(self.mod, "_flux", return_value=None):
            self.assertEqual(self.mod.list_devices(), [])
            self.assertFalse(self.mod.is_available())
            self.assertIn("error", self.mod.set_state(DEV, on=True))

    def test_disabled_knob(self):
        self.use(results=[_disc("192.0.2.77")])
        with mock.patch.dict("os.environ", {"MAGICHOME_ENABLED": "0"}):
            self.assertEqual(self.mod.list_devices(), [])
            self.assertIn("error", self.mod.set_state(DEV, on=True))

    def test_register(self):
        self.assertIn("magichome_list_devices", self.actions)


class ControlTests(MagicHomeBase):
    def test_on(self):
        self.use()
        r = self.mod.set_state(DEV, on=True)
        self.assertTrue(r["ok"])
        b = FakeBulb.instances[0]
        self.assertIn(("on",), b.calls)
        self.assertEqual(b.ip, "192.0.2.77")
        self.assertTrue(b.closed)

    def test_off(self):
        self.use()
        r = self.mod.set_state(DEV, on=False)
        self.assertTrue(r["ok"])
        self.assertFalse(r["applied"]["on"])
        self.assertIn(("off",), FakeBulb.instances[0].calls)

    def test_brightness(self):
        self.use()
        r = self.mod.set_state(DEV, brightness=40)
        self.assertTrue(r["ok"])
        self.assertEqual(r["applied"]["brightness"], 40)
        rgb_calls = [c for c in FakeBulb.instances[0].calls if c[0] == "rgb"]
        self.assertEqual(rgb_calls[0][4], round(40 * 255 / 100))

    def test_brightness_zero_is_off(self):
        self.use()
        r = self.mod.set_state(DEV, brightness=0)
        self.assertTrue(r["ok"])
        self.assertIn(("off",), FakeBulb.instances[0].calls)

    def test_colour(self):
        self.use()
        r = self.mod.set_state(DEV, color=(255, 0, 0), color_name="red")
        self.assertTrue(r["ok"])
        self.assertEqual(r["applied"]["color"], [255, 0, 0])
        self.assertIn(("rgb", 255, 0, 0, None), FakeBulb.instances[0].calls)

    def test_colour_and_brightness(self):
        self.use()
        r = self.mod.set_state(DEV, color=(0, 60, 255), brightness=50)
        self.assertTrue(r["ok"])
        self.assertEqual(r["applied"]["brightness"], 50)

    def test_status_read(self):
        self.use()
        s = self.mod.get_state(DEV)
        self.assertEqual(s["rgb"], [255, 255, 255])
        self.assertEqual(s["brightness"], 100)

    def test_nothing_to_do_and_unsupported(self):
        self.use()
        self.assertIn("error", self.mod.set_state(DEV))
        self.assertIn("error", self.mod.set_state(DEV, color_temperature=2700))

    def test_name_match_without_ip(self):
        self.use(results=[_disc("192.0.2.9", ident="AABBCCDDEEFF")])
        dev = {"name": "Magic Home DDEEFF", "brand": "Magic Home"}
        self.assertTrue(self.mod.set_state(dev, on=True)["ok"])
        self.assertEqual(FakeBulb.instances[-1].ip, "192.0.2.9")
        self.assertIn("error", self.mod.set_state({"name": "nope"}, on=True))


class HonestFailureTests(MagicHomeBase):
    def test_unacknowledged_on_is_error(self):
        self.use(stuck=True)
        r = self.mod.set_state(DEV, on=True)
        self.assertIn("error", r)
        self.assertNotIn("ok", r)

    def test_unacknowledged_colour_is_error(self):
        self.use(stuck=True)
        r = self.mod.set_state(DEV, color=(255, 0, 0))
        self.assertIn("error", r)

    def test_connect_failure_is_error(self):
        fl = _fake_flux()

        def _boom(ip, timeout=5):
            raise OSError("refused")
        fl.WifiLedBulb = _boom
        with mock.patch.object(self.mod, "_flux", return_value=fl):
            self.assertIn("error", self.mod.set_state(DEV, on=True))
            self.assertIn("error", self.mod.get_state(DEV))


class RouterIntegrationTests(unittest.TestCase):
    def test_brand_mapping(self):
        for brand in ("Magic Home", "MagicHome", "Flux LED"):
            self.assertEqual(router._controller_for(brand), "sh_magichome")

    def test_discover_brand_mapping_and_source(self):
        from skills import smart_home_discover as d
        self.assertIn("sh_magichome", d._LAN_SOURCES)
        self.assertEqual(d._BRAND_TO_SKILL["magic home"], "sh_magichome")

    def test_failure_falls_back_to_alexa(self):
        err = {"error": "Magic Home did not turn on (status read still off)"}
        with mock.patch.object(router, "_call_skill", return_value=err) as cs, \
             mock.patch.object(router, "_alexa_set_state",
                               return_value={"ok": True}):
            out = router._dispatch_one(DEV, {"verb": "on"})
        self.assertEqual(cs.call_args[0][0], "sh_magichome")
        self.assertEqual(out["path"], "alexa-after-sh_magichome-fail")
        self.assertEqual(out["direct_error"], err["error"])

    def test_direct_success_path(self):
        with mock.patch.object(router, "_call_skill",
                               return_value={"ok": True}):
            out = router._dispatch_one(DEV, {"verb": "on"})
        self.assertEqual(out["path"], "direct/sh_magichome")


if __name__ == "__main__":
    unittest.main()
