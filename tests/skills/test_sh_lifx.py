"""Logic tests for skills/sh_lifx.py (LIFX LAN controller).

Thin wrapper over the optional `lifxlan` library. `lifxlan` is NOT on the CI
runner; the skill resolves it lazily via `_lifxlan()`, which we either patch
to return a hand-rolled fake module or inject into ``sys.modules`` (removed in
tearDown) so the real network-touching ``LifxLAN().get_lights()`` is never
called. No UDP broadcast, thread, or sleep ever runs for real — the discovery
worker is driven with a fake ``lifxlan`` whose ``get_lights()`` returns
instantly, and the timeout branch is exercised by patching the helper to a
canned result rather than blocking a real thread.

Coverage:
  * graceful degradation when lifxlan is absent / nothing discovered,
  * `_lifxlan` import success (fake) and failure (blocked) paths,
  * `_discover_bulbs` worker happy-path, worker-exception, and timeout,
  * `_refresh` cache-hit short-circuit, real discovery building by_name/by_mac,
    per-bulb attribute-error skip, and the timeout-keeps-cache branch,
  * `_bulb_for` MAC-first then name resolution and the miss,
  * the pure `_rgb_to_hsbk` colour conversion (0..65535 channel scaling),
  * list_devices / get_state / set_state against a fake bulb (every kwarg
    branch incl. colour, brightness-with-get_color-failure, colour-temperature,
    and the partial-failure path), with `_refresh`/`_bulb_for` stubbed.

The module-level `_state` cache dict is reset in tearDown so discovery results
never leak between tests.
"""
from __future__ import annotations

import contextlib
import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install/remove fake top-level modules in ``sys.modules`` for
    the duration of a block, restoring prior state — including absence — on
    exit. ``obj=None`` forces ``import <name>`` to miss (module removed). Only
    flat (non-dotted) names are needed here (``lifxlan``)."""
    saved: dict[str, object] = {}
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class _FakeBulb:
    def __init__(self, label="Kitchen", mac="d0:73:d5:00:00:01", power=0,
                 color=(0, 0, 32768, 3500), supports_color=True,
                 get_color_raises=False):
        self._label = label
        self._mac = mac
        self._power = power
        self._color = color
        self._supports = supports_color
        self._get_color_raises = get_color_raises
        self.set_color_calls = []
        self.set_power_calls = []

    def get_label(self): return self._label
    def get_mac_addr(self): return self._mac
    def get_power(self): return self._power

    def get_color(self):
        if self._get_color_raises:
            raise RuntimeError("get_color boom")
        return self._color

    def supports_color(self): return self._supports

    def set_power(self, v): self.set_power_calls.append(v)
    def set_color(self, hsbk): self.set_color_calls.append(hsbk)


def make_fake_lifxlan(*, get_lights=None, lan_raises=False):
    """Build a fake `lifxlan` module exposing ``LifxLAN()`` with a
    ``get_lights()`` that returns ``get_lights`` (a list) instantly. When
    ``lan_raises`` the LifxLAN ctor raises so the discover worker's except-arm
    runs."""
    mod = types.ModuleType("lifxlan")

    class _LifxLAN:
        def __init__(self, *a, **k):
            if lan_raises:
                raise RuntimeError("LAN init boom")

        def get_lights(self):
            return get_lights if get_lights is not None else []

    mod.LifxLAN = _LifxLAN
    return mod


class _LifxBase(unittest.TestCase):
    """Reset the module-level discovery cache between tests so a populated
    ``_state`` never leaks into another test's _refresh()."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_lifx")
        self.addCleanup(self._reset_state)

    def _reset_state(self):
        self.mod._state["by_name"] = {}
        self.mod._state["by_mac"] = {}
        self.mod._state["fetched_at"] = 0.0


# ─── graceful degradation / dependency resolution ────────────────────────
class LifxDegradationTests(_LifxBase):
    def test_is_available_false_without_lifxlan(self):
        with mock.patch.object(self.mod, "_lifxlan", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_is_available_true_with_lifxlan(self):
        with mock.patch.object(self.mod, "_lifxlan",
                               return_value=make_fake_lifxlan()):
            self.assertTrue(self.mod.is_available())

    def test_lifxlan_import_succeeds_when_present(self):
        # Inject a fake module so the lazy `import lifxlan` resolves it without
        # requiring the real (CI-absent) package.
        fake = make_fake_lifxlan()
        with inject_modules(lifxlan=fake):
            self.assertIs(self.mod._lifxlan(), fake)

    def test_lifxlan_import_returns_none_when_absent(self):
        # Force `import lifxlan` to raise so the except-arm returns None.
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "lifxlan" or name.split(".")[0] == "lifxlan":
                raise ImportError("blocked lifxlan")
            return real_import(name, *a, **k)

        with inject_modules(lifxlan=None), \
             mock.patch("builtins.__import__", side_effect=_blocked):
            self.assertIsNone(self.mod._lifxlan())

    def test_refresh_returns_empty_without_lib(self):
        with mock.patch.object(self.mod, "_lifxlan", return_value=None):
            self.assertEqual(self.mod._refresh(force=True), {})

    def test_list_devices_empty_when_none_discovered(self):
        with mock.patch.object(self.mod, "_refresh", return_value={}):
            self.assertEqual(self.mod.list_devices(), [])

    def test_lifx_list_informative_when_empty(self):
        with mock.patch.object(self.mod, "list_devices", return_value=[]):
            out = self.actions["lifx_list"]("")
        self.assertIn("No LIFX bulbs", out)
        self.assertIn("56700", out)  # mentions the UDP port hint

    def test_get_state_bulb_not_found(self):
        with mock.patch.object(self.mod, "_bulb_for", return_value=None):
            res = self.mod.get_state({"name": "Ghost"})
        self.assertIn("not found", res["error"])

    def test_set_state_bulb_not_found(self):
        with mock.patch.object(self.mod, "_bulb_for", return_value=None):
            res = self.mod.set_state({"name": "Ghost"}, on=True)
        self.assertIn("not found", res["error"])


# ─── _discover_bulbs (threaded discovery with hard timeout) ──────────────
class LifxDiscoverBulbsTests(_LifxBase):
    def test_discover_returns_bulb_list(self):
        bulb = _FakeBulb()
        fake = make_fake_lifxlan(get_lights=[bulb])
        out = self.mod._discover_bulbs(fake, timeout=2.0)
        self.assertEqual(out, [bulb])

    def test_discover_worker_exception_yields_empty_list(self):
        # LifxLAN() ctor raises inside the worker → result set to [] (not None),
        # so the caller treats it as "discovered nothing" rather than a timeout.
        fake = make_fake_lifxlan(lan_raises=True)
        out = self.mod._discover_bulbs(fake, timeout=2.0)
        self.assertEqual(out, [])

    def test_discover_timeout_returns_none(self):
        # Worker still alive when join() returns → None signals "keep cache".
        # We patch Thread so the worker never actually runs and is_alive() is
        # forced True — no real blocking thread is spawned.
        class _StuckThread:
            def __init__(self, *a, **k):
                pass

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return True

        with mock.patch.object(self.mod.threading, "Thread", _StuckThread):
            out = self.mod._discover_bulbs(make_fake_lifxlan(), timeout=0.01)
        self.assertIsNone(out)


# ─── _refresh (cache + map building) ─────────────────────────────────────
class LifxRefreshTests(_LifxBase):
    def test_refresh_uses_cache_within_ttl(self):
        # Pre-populate cache and stamp it "fresh"; _refresh must short-circuit
        # and NOT invoke discovery at all.
        bulb = _FakeBulb(label="Cached")
        self.mod._state["by_name"] = {"cached": bulb}
        self.mod._state["fetched_at"] = self.mod.time.monotonic()
        with mock.patch.object(self.mod, "_discover_bulbs") as disc:
            out = self.mod._refresh()
        disc.assert_not_called()
        self.assertEqual(out, {"cached": bulb})

    def test_refresh_builds_name_and_mac_maps(self):
        b1 = _FakeBulb(label="Kitchen", mac="D0:73:D5:00:00:01")
        b2 = _FakeBulb(label="Bedroom", mac="d0:73:d5:00:00:02")
        with mock.patch.object(self.mod, "_lifxlan",
                               return_value=make_fake_lifxlan()), \
             mock.patch.object(self.mod, "_discover_bulbs",
                               return_value=[b1, b2]):
            out = self.mod._refresh(force=True)
        self.assertEqual(set(out), {"kitchen", "bedroom"})
        # MAC keys are lower-cased.
        self.assertIn("d0:73:d5:00:00:01", self.mod._state["by_mac"])
        self.assertIn("d0:73:d5:00:00:02", self.mod._state["by_mac"])
        # fetched_at advanced off the 0.0 default.
        self.assertGreater(self.mod._state["fetched_at"], 0.0)

    def test_refresh_skips_bulb_whose_attrs_raise(self):
        good = _FakeBulb(label="Good", mac="d0:73:d5:00:00:03")

        class _BadBulb:
            def get_label(self):
                raise RuntimeError("label boom")

            def get_mac_addr(self):
                return "x"
        with mock.patch.object(self.mod, "_lifxlan",
                               return_value=make_fake_lifxlan()), \
             mock.patch.object(self.mod, "_discover_bulbs",
                               return_value=[good, _BadBulb()]):
            out = self.mod._refresh(force=True)
        # Only the good bulb made it into the map; the bad one was skipped.
        self.assertEqual(set(out), {"good"})

    def test_refresh_unlabelled_bulb_only_indexed_by_mac(self):
        # A bulb with an empty label is indexed by MAC but not by name.
        b = _FakeBulb(label="  ", mac="d0:73:d5:00:00:09")
        with mock.patch.object(self.mod, "_lifxlan",
                               return_value=make_fake_lifxlan()), \
             mock.patch.object(self.mod, "_discover_bulbs", return_value=[b]):
            out = self.mod._refresh(force=True)
        self.assertEqual(out, {})  # nothing by name
        self.assertIn("d0:73:d5:00:00:09", self.mod._state["by_mac"])

    def test_refresh_labelled_bulb_without_mac_only_indexed_by_name(self):
        # A labelled bulb whose MAC is blank exercises the `if mac:` false edge
        # (line 96 → back to the loop) and is indexed by name only. A second
        # bulb follows so the loop genuinely iterates past the blank-MAC one.
        no_mac = _FakeBulb(label="NoMac", mac="")
        with_mac = _FakeBulb(label="HasMac", mac="d0:73:d5:00:00:0a")
        with mock.patch.object(self.mod, "_lifxlan",
                               return_value=make_fake_lifxlan()), \
             mock.patch.object(self.mod, "_discover_bulbs",
                               return_value=[no_mac, with_mac]):
            out = self.mod._refresh(force=True)
        self.assertEqual(set(out), {"nomac", "hasmac"})
        # The blank-MAC bulb contributed no by_mac entry.
        self.assertEqual(set(self.mod._state["by_mac"]), {"d0:73:d5:00:00:0a"})

    def test_refresh_timeout_keeps_cached_bulbs(self):
        # _discover_bulbs returns None (timeout) → _refresh keeps the existing
        # cache rather than wiping it.
        bulb = _FakeBulb(label="Stale")
        self.mod._state["by_name"] = {"stale": bulb}
        with mock.patch.object(self.mod, "_lifxlan",
                               return_value=make_fake_lifxlan()), \
             mock.patch.object(self.mod, "_discover_bulbs", return_value=None):
            out = self.mod._refresh(force=True)
        self.assertEqual(out, {"stale": bulb})


# ─── _bulb_for (name / MAC resolution) ───────────────────────────────────
class LifxBulbForTests(_LifxBase):
    def test_bulb_for_matches_by_mac_first(self):
        by_mac_bulb = _FakeBulb(label="ByMac")
        by_name_bulb = _FakeBulb(label="ByName")
        self.mod._state["by_mac"] = {"d0:73:d5:00:00:aa": by_mac_bulb}
        self.mod._state["by_name"] = {"kitchen": by_name_bulb}
        with mock.patch.object(self.mod, "_refresh"):
            got = self.mod._bulb_for({"name": "Kitchen",
                                      "lan_mac": "D0:73:D5:00:00:AA"})
        self.assertIs(got, by_mac_bulb)   # MAC wins over name

    def test_bulb_for_matches_by_name_when_no_mac(self):
        bulb = _FakeBulb(label="Kitchen")
        self.mod._state["by_name"] = {"kitchen": bulb}
        with mock.patch.object(self.mod, "_refresh"):
            got = self.mod._bulb_for({"name": "Kitchen"})
        self.assertIs(got, bulb)

    def test_bulb_for_returns_none_when_unknown(self):
        with mock.patch.object(self.mod, "_refresh"):
            self.assertIsNone(self.mod._bulb_for({"name": "Ghost"}))


# ─── pure colour maths ───────────────────────────────────────────────────
class LifxRgbToHsbkTests(_LifxBase):
    def test_red_full_saturation_and_brightness(self):
        h, s, b, k = self.mod._rgb_to_hsbk((255, 0, 0), kelvin=3500)
        self.assertEqual(h, 0)          # red hue 0°
        self.assertEqual(s, 65535)      # fully saturated
        self.assertEqual(b, 65535)      # full brightness
        self.assertEqual(k, 3500)       # kelvin passed through

    def test_green_hue_is_third_of_scale(self):
        h, _s, _b, _k = self.mod._rgb_to_hsbk((0, 255, 0))
        # 120° / 360° * 65535 ≈ 21845.
        self.assertAlmostEqual(h, 21845, delta=2)

    def test_blue_hue_is_two_thirds_of_scale(self):
        h, _s, _b, _k = self.mod._rgb_to_hsbk((0, 0, 255))
        # 240° / 360° * 65535 ≈ 43690.
        self.assertAlmostEqual(h, 43690, delta=2)

    def test_white_zero_saturation(self):
        h, s, b, _k = self.mod._rgb_to_hsbk((255, 255, 255))
        self.assertEqual(s, 0)
        self.assertEqual(b, 65535)

    def test_black_is_all_zero(self):
        # mx == 0 → saturation 0, value 0, hue 0.
        h, s, b, _k = self.mod._rgb_to_hsbk((0, 0, 0))
        self.assertEqual((h, s, b), (0, 0, 0))


# ─── list_devices / get_state ────────────────────────────────────────────
class LifxListAndStateTests(_LifxBase):
    def test_list_devices_shapes_record(self):
        bulb = _FakeBulb(label="Kitchen")
        with mock.patch.object(self.mod, "_refresh",
                               return_value={"kitchen": bulb}):
            devs = self.mod.list_devices()
        self.assertEqual(len(devs), 1)
        d = devs[0]
        self.assertEqual(d["name"], "Kitchen")
        self.assertEqual(d["brand"], "LIFX")
        self.assertIn("color", d["capabilities"])  # supports_color True
        self.assertEqual(d["lan_mac"], "d0:73:d5:00:00:01")

    def test_list_devices_without_color_capability(self):
        bulb = _FakeBulb(label="WhiteOnly", supports_color=False)
        with mock.patch.object(self.mod, "_refresh",
                               return_value={"whiteonly": bulb}):
            devs = self.mod.list_devices()
        self.assertNotIn("color", devs[0]["capabilities"])
        self.assertIn("on_off", devs[0]["capabilities"])

    def test_list_devices_caps_probe_exception_falls_back(self):
        # supports_color() raising falls into the except-arm → minimal caps.
        class _BadCaps(_FakeBulb):
            def supports_color(self):
                raise RuntimeError("caps boom")
        bulb = _BadCaps(label="Weird")
        with mock.patch.object(self.mod, "_refresh",
                               return_value={"weird": bulb}):
            devs = self.mod.list_devices()
        self.assertEqual(devs[0]["capabilities"], ["dim", "on_off"])

    def test_get_state_translates_power_and_brightness(self):
        # power on, brightness mid (32768/65535 ≈ 50%), kelvin 3500.
        bulb = _FakeBulb(power=65535, color=(0, 0, 32768, 3500))
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            st = self.mod.get_state({"name": "Kitchen"})
        self.assertTrue(st["on"])
        self.assertEqual(st["brightness"], 50)
        self.assertEqual(st["color_temperature_k"], 3500)

    def test_get_state_read_failure_returns_error(self):
        bulb = _FakeBulb(get_color_raises=True)
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            st = self.mod.get_state({"name": "Kitchen"})
        self.assertIn("state read failed", st["error"])


# ─── set_state branches ──────────────────────────────────────────────────
class LifxSetStateTests(_LifxBase):
    def test_set_power_off(self):
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, on=False)
        self.assertEqual(res["applied"]["on"], False)
        self.assertEqual(bulb.set_power_calls, ["off"])

    def test_set_power_on(self):
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, on=True)
        self.assertEqual(res["applied"]["on"], True)
        self.assertEqual(bulb.set_power_calls, ["on"])

    def test_set_brightness_scales_and_powers_on(self):
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, brightness=50)
        self.assertEqual(res["applied"]["brightness"], 50)
        # set_color called with brightness ≈ 32767 (50% of 65535).
        self.assertTrue(bulb.set_color_calls)
        _h, _s, bri, _k = bulb.set_color_calls[-1]
        self.assertAlmostEqual(bri, 32767, delta=2)
        # >0% brightness also powers the bulb on.
        self.assertIn("on", bulb.set_power_calls)

    def test_set_brightness_zero_does_not_power_on(self):
        # 0% brightness → set_color happens but NO implicit power-on.
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, brightness=0)
        self.assertEqual(res["applied"]["brightness"], 0)
        self.assertNotIn("on", res["applied"])
        self.assertEqual(bulb.set_power_calls, [])

    def test_set_brightness_clamps_above_100(self):
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, brightness=250)
        self.assertEqual(res["applied"]["brightness"], 100)   # clamped
        _h, _s, bri, _k = bulb.set_color_calls[-1]
        self.assertEqual(bri, 65535)

    def test_set_brightness_when_get_color_fails_uses_defaults(self):
        # get_color() raising falls back to h,s,k = 0,0,3500 but still applies.
        bulb = _FakeBulb(get_color_raises=True)
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, brightness=40)
        self.assertEqual(res["applied"]["brightness"], 40)
        h, s, _bri, k = bulb.set_color_calls[-1]
        self.assertEqual((h, s, k), (0, 0, 3500))

    def test_set_color_converts_to_hsbk(self):
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, color=(255, 0, 0))
        self.assertEqual(res["applied"]["color"], [255, 0, 0])
        # Red → full saturation/brightness HSBK applied.
        h, s, b, _k = bulb.set_color_calls[-1]
        self.assertEqual((h, s, b), (0, 65535, 65535))

    def test_set_color_uses_supplied_color_temperature_as_kelvin(self):
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            self.mod.set_state({"name": "Kitchen"}, color=(255, 0, 0),
                               color_temperature=5000)
        _h, _s, _b, k = bulb.set_color_calls[-1]
        self.assertEqual(k, 5000)

    def test_color_takes_precedence_over_brightness(self):
        # When both color and brightness are supplied, the color branch wins
        # (elif), so brightness is NOT separately applied.
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, color=(0, 255, 0),
                                     brightness=10)
        self.assertIn("color", res["applied"])
        self.assertNotIn("brightness", res["applied"])

    def test_set_color_temperature_only(self):
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, color_temperature=4200)
        self.assertEqual(res["applied"]["color_temperature_k"], 4200)
        # Saturation forced to 0 (white) and kelvin applied.
        _h, s, _b, k = bulb.set_color_calls[-1]
        self.assertEqual(s, 0)
        self.assertEqual(k, 4200)

    def test_set_color_temperature_when_get_color_fails_uses_defaults(self):
        bulb = _FakeBulb(get_color_raises=True)
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, color_temperature=6000)
        self.assertEqual(res["applied"]["color_temperature_k"], 6000)
        h, s, br, k = bulb.set_color_calls[-1]
        # Defaults h=0, br=32768 from the except-arm; saturation forced 0.
        self.assertEqual((h, s, br, k), (0, 0, 32768, 6000))

    def test_set_state_failure_returns_partial(self):
        # set_power raising mid-apply → error dict carrying the partial applied.
        class _PowerBoom(_FakeBulb):
            def set_power(self, v):
                raise RuntimeError("power nope")
        bulb = _PowerBoom()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, on=True, brightness=50)
        self.assertIn("set_state failed", res["error"])
        self.assertIn("partial", res)

    def test_set_state_noop_when_no_kwargs(self):
        # No actionable kwargs → ok with an empty applied dict, no bulb calls.
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"})
        self.assertTrue(res["ok"])
        self.assertEqual(res["applied"], {})
        self.assertEqual(bulb.set_color_calls, [])
        self.assertEqual(bulb.set_power_calls, [])


# ─── lifx_list action (populated paths) ──────────────────────────────────
class LifxListActionTests(_LifxBase):
    def test_lifx_list_names_bulbs(self):
        devs = [{"name": "Kitchen"}, {"name": "Bedroom"}]
        with mock.patch.object(self.mod, "list_devices", return_value=devs):
            out = self.actions["lifx_list"]("")
        self.assertIn("2 LIFX bulb(s)", out)
        self.assertIn("Kitchen", out)
        self.assertIn("Bedroom", out)
        self.assertNotIn("(+more)", out)

    def test_lifx_list_truncates_beyond_ten(self):
        devs = [{"name": f"Bulb{i}"} for i in range(13)]
        with mock.patch.object(self.mod, "list_devices", return_value=devs):
            out = self.actions["lifx_list"]("")
        self.assertIn("13 LIFX bulb(s)", out)
        self.assertIn("(+more)", out)
        # Only the first 10 names are listed inline.
        self.assertIn("Bulb0", out)
        self.assertNotIn("Bulb10", out)

    def test_register_wires_both_aliases(self):
        actions: dict = {}
        self.mod.register(actions)
        self.assertIn("lifx_list", actions)
        self.assertIn("lifx_list_devices", actions)
        self.assertIs(actions["lifx_list"], actions["lifx_list_devices"])


if __name__ == "__main__":
    unittest.main()
