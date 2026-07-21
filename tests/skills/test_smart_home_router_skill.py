"""Logic tests for skills/smart_home_router_skill.py.

The skill file itself is a thin shim that forwards register() to
`core.smart_home_router` (where the real dispatch logic lives). So this file
covers BOTH:

  1. The shim — successful delegation registers the router's actions, and a
     failing core import degrades silently (no raise).
  2. The router logic the shim exposes — utterance parsing, value extraction,
     device matching, action→kwargs translation, and end-to-end
     smart_home_control with dispatch stubbed so nothing touches a real device
     or the catalog on disk.

All catalog access and per-device dispatch are mocked: no disk reads/writes,
no brand-skill imports, no network.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from core import smart_home_router as router
from tests._skill_harness import load_skill_isolated


# A small synthetic catalog injected wherever the router would read one.
_CATALOG = {
    "device_count": 3,
    "echo_count": 1,
    "group_count": 1,
    "devices": [
        {"name": "Office Light", "brand": "LIFX", "type": "light",
         "alexa_room": "Office", "alexa_groups": [], "controller_skill": "sh_lifx",
         "capabilities": ["on_off", "dim", "color"]},
        {"name": "Kitchen Light", "brand": "Philips Hue", "type": "light",
         "alexa_room": "Kitchen", "alexa_groups": [], "controller_skill": "sh_hue",
         "capabilities": ["on_off", "dim"]},
        {"name": "Hallway", "brand": "Nest", "type": "thermostat",
         "alexa_room": "Hall", "alexa_groups": [], "controller_skill": "sh_nest",
         "capabilities": ["thermostat"]},
    ],
}


# ── 1. the shim ──────────────────────────────────────────────────────
class RouterShimTests(unittest.TestCase):
    def test_shim_forwards_register_to_core(self):
        # Loading the shim calls core.smart_home_router.register(actions). We
        # stub warm_up (which reads the catalog) so loading is side-effect free,
        # then assert the router's actions landed in the dict.
        with mock.patch.object(router, "warm_up", return_value=None):
            _mod, actions = load_skill_isolated("smart_home_router_skill")
        for name in ("smart_home_control", "smart_home_devices",
                     "smart_home_router_status", "refresh_smart_home_router"):
            self.assertIn(name, actions)
        self.assertTrue(callable(actions["smart_home_control"]))

    def test_shim_degrades_when_core_import_fails(self):
        # The `from core import smart_home_router` inside register() raises
        # (e.g. partial install / corrupt catalog at import time) -> the shim's
        # outer try/except swallows it, prints a hint, and returns without
        # registering anything.
        mod, _ = load_skill_isolated("smart_home_router_skill", register=False)
        real_import = __import__

        def _imp(name, *a, **k):
            if name == "core" and a and len(a) >= 3 and "smart_home_router" in (a[2] or []):
                raise ImportError("corrupt catalog")
            if name == "core.smart_home_router":
                raise ImportError("corrupt catalog")
            return real_import(name, *a, **k)

        actions: dict = {}
        with mock.patch("builtins.__import__", side_effect=_imp):
            mod.register(actions)
        self.assertEqual(actions, {})  # nothing registered, no raise

    def test_shim_degrades_when_core_register_fails(self):
        # The shim wraps core.smart_home_router.register() in try/except so a
        # failure there (e.g. corrupt catalog) can't take down skill loading.
        # Force register() to raise and assert the shim swallows it: no
        # exception propagates and nothing lands in the actions dict.
        actions: dict = {}
        with mock.patch.object(router, "register",
                               side_effect=RuntimeError("simulated register failure")):
            # load_skill_isolated calls the shim's register(), which calls the
            # (now-raising) core register() inside its own try/except.
            _mod, actions = load_skill_isolated(
                "smart_home_router_skill", actions=actions)
        self.assertEqual(actions, {})  # degraded cleanly, no raise


# ── 2a. utterance classification (pure) ──────────────────────────────
class RouterClassifyTests(unittest.TestCase):
    def test_turn_off_named_device(self):
        a = router._classify_action("turn off the office light")
        self.assertEqual(a["verb"], "off")
        self.assertEqual(a["descriptor"], "the office light")

    def test_turn_on_trailing_form(self):
        a = router._classify_action("kitchen light on")
        self.assertEqual(a["verb"], "on")
        self.assertEqual(a["descriptor"], "kitchen light")

    def test_lock_before_on_reduction(self):
        # 'lock the front door' must NOT collapse to an on/off verb.
        a = router._classify_action("lock the front door")
        self.assertEqual(a["verb"], "lock")
        self.assertEqual(a["descriptor"], "the front door")

    def test_set_to_temperature(self):
        a = router._classify_action("set bedroom to 65")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["temperature"], 65)
        self.assertEqual(a["descriptor"], "bedroom")

    def test_set_to_color(self):
        a = router._classify_action("set the bedroom to blue")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["color"][0], "blue")

    def test_dim_with_percent_word(self):
        a = router._classify_action("dim the kitchen lights to 30 percent")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["brightness"], 30)
        self.assertEqual(a["descriptor"], "the kitchen lights")

    def test_set_to_percent_symbol(self):
        # The reported bug: 'set X to N%' dropped brightness entirely because
        # the '%' extractor branch was dead, so this path fell through with no
        # brightness set. Must now carry the value and the device descriptor.
        a = router._classify_action("set the office to 75%")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["brightness"], 75)
        self.assertEqual(a["descriptor"], "the office")

    def test_dim_to_percent_symbol_not_defaulted(self):
        # 'dim X to N%' previously fell back to the 30% default (which masked
        # the bug when N==30). A non-30 value proves the '%' is really parsed.
        a = router._classify_action("dim the kitchen lights to 50%")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["brightness"], 50)
        self.assertEqual(a["descriptor"], "the kitchen lights")

    def test_dim_default_brightness_when_unspecified(self):
        a = router._classify_action("dim the office")
        self.assertEqual(a["brightness"], 30)  # 'dim' defaults to 30%

    def test_unparseable_has_no_verb(self):
        a = router._classify_action("what time is it")
        self.assertIsNone(a["verb"])


# ── 2b. value extractors (pure) ──────────────────────────────────────
class RouterExtractorTests(unittest.TestCase):
    def test_percent_word_forms(self):
        self.assertEqual(router._extract_percent("30 percent"), 30)
        self.assertEqual(router._extract_percent("thirty percent"), 30)

    def test_percent_symbol_forms(self):
        # Regression guard: the '%' symbol branch must match, not just the
        # word 'percent'. A trailing \b after '%' never matches (both sides
        # are non-word chars), which silently killed this branch and dropped
        # brightness from phrasings like 'set the office to 75%'.
        self.assertEqual(router._extract_percent("30%"), 30)
        self.assertEqual(router._extract_percent("set to 30% now"), 30)
        self.assertEqual(router._extract_percent("75 %"), 75)   # space before %
        self.assertEqual(router._extract_percent("50%."), 50)   # trailing punctuation

    def test_percent_clamped(self):
        self.assertEqual(router._extract_percent("250 percent"), 100)
        self.assertEqual(router._extract_percent("250%"), 100)   # symbol form clamps too

    def test_temperature_range_gated(self):
        self.assertEqual(router._extract_temperature("set to 68 degrees"), 68)
        # 200 is outside the 40..110 thermostat band → ignored.
        self.assertIsNone(router._extract_temperature("to 200 degrees"))

    def test_named_color(self):
        self.assertEqual(router._extract_color("make it red")[1], (255, 0, 0))
        self.assertIsNone(router._extract_color("make it sparkly"))

    def test_color_temperature_kelvin_clamped(self):
        self.assertEqual(router._extract_color_temperature("warm 2700K"), 2700)
        self.assertEqual(router._extract_color_temperature("8000 kelvin"), 6500)
        self.assertIsNone(router._extract_color_temperature("no kelvin here"))

    def test_number_word_parsing(self):
        self.assertEqual(router._parse_number("fifty"), 50)
        self.assertEqual(router._parse_number("72"), 72)
        self.assertIsNone(router._parse_number("banana"))


# ── 2c. action → kwargs translation ──────────────────────────────────
class RouterActionToKwargsTests(unittest.TestCase):
    def test_on(self):
        self.assertEqual(router._action_to_kwargs({"verb": "on"}), {"on": True})

    def test_off(self):
        self.assertEqual(router._action_to_kwargs({"verb": "off"}), {"on": False})

    def test_lock_unlock(self):
        self.assertEqual(router._action_to_kwargs({"verb": "lock"}),
                         {"locked": True})
        self.assertEqual(router._action_to_kwargs({"verb": "unlock"}),
                         {"locked": False})

    def test_brightness_implies_on(self):
        kw = router._action_to_kwargs({"verb": "set", "brightness": 40})
        self.assertEqual(kw["brightness"], 40)
        self.assertTrue(kw["on"])

    def test_zero_brightness_implies_off(self):
        kw = router._action_to_kwargs({"verb": "set", "brightness": 0})
        self.assertEqual(kw["brightness"], 0)
        self.assertFalse(kw["on"])

    def test_color_carries_name_and_rgb(self):
        kw = router._action_to_kwargs({"verb": "set", "color": ("blue", (0, 60, 255))})
        self.assertEqual(kw["color"], (0, 60, 255))
        self.assertEqual(kw["color_name"], "blue")


# ── 2d. device matching ──────────────────────────────────────────────
class RouterMatchingTests(unittest.TestCase):
    def test_match_score_name_substring_bonus(self):
        dev = _CATALOG["devices"][0]  # Office Light
        self.assertGreater(router._match_score("office light", dev), 1.0)

    def test_match_score_zero_for_unrelated(self):
        dev = _CATALOG["devices"][0]
        self.assertEqual(router._match_score("garage door", dev), 0.0)

    def test_resolve_picks_correct_device(self):
        got = router._resolve_devices("the office light", _CATALOG)
        self.assertEqual([d["name"] for d in got], ["Office Light"])

    def test_resolve_type_hint_selects_thermostat(self):
        got = router._resolve_devices("hallway", _CATALOG, want_type="thermostat")
        self.assertEqual([d["name"] for d in got], ["Hallway"])

    def test_resolve_no_match_returns_empty(self):
        self.assertEqual(router._resolve_devices("garage", _CATALOG), [])


# ── 2e. end-to-end smart_home_control (dispatch stubbed) ─────────────
class RouterControlEndToEndTests(unittest.TestCase):
    def setUp(self):
        # Inject the synthetic catalog so _ensure_catalog never reads disk.
        self._patch_cat = mock.patch.object(router, "_ensure_catalog",
                                            return_value=_CATALOG)
        self._patch_cat.start()
        self.addCleanup(self._patch_cat.stop)

    def test_empty_utterance_prompts(self):
        out = router.smart_home_control("")
        self.assertIn("something to do", out.lower())

    def test_unparseable_utterance_message(self):
        out = router.smart_home_control("what time is it")
        self.assertIn("couldn't parse", out.lower())

    def test_turn_off_dispatches_and_summarizes(self):
        disp = mock.Mock(return_value={"ok": True, "device": "Office Light"})
        with mock.patch.object(router, "_dispatch_one", disp):
            out = router.smart_home_control("turn off the office light")
        # The matched device was dispatched with on=False.
        dev_arg, action_arg = disp.call_args[0]
        self.assertEqual(dev_arg["name"], "Office Light")
        self.assertEqual(action_arg["verb"], "off")
        self.assertIn("Off, sir", out)
        self.assertIn("Office Light", out)

    def test_set_temperature_routes_to_thermostat(self):
        disp = mock.Mock(return_value={"ok": True, "device": "Hallway"})
        with mock.patch.object(router, "_dispatch_one", disp):
            out = router.smart_home_control("set hallway to 68")
        dev_arg, _action = disp.call_args[0]
        self.assertEqual(dev_arg["name"], "Hallway")     # thermostat, not a lamp
        self.assertIn("68", out)

    def test_no_catalog_message(self):
        # The live-LAN fallback is stubbed to "nothing on the LAN either" so
        # this deterministically asserts the terminal wizard refusal (the
        # unstubbed fallback would attempt real sh_kasa LAN discovery).
        with mock.patch.object(router, "_ensure_catalog", return_value=None), \
             mock.patch.object(router, "_live_lan_fallback",
                               return_value=None):
            out = router.smart_home_control("turn off the office light")
        self.assertIn("No smart-home catalog", out)

    def test_descriptor_with_no_device_match(self):
        # Fallback stubbed to a LAN miss (see test_no_catalog_message).
        with mock.patch.object(router, "_live_lan_fallback",
                               return_value=None):
            out = router.smart_home_control("turn off the garage door")
        self.assertIn("don't see anything", out.lower())

    def test_failure_summary_surfaces_device_error(self):
        disp = mock.Mock(return_value={"error": "bridge not connected",
                                       "device": "Office Light"})
        with mock.patch.object(router, "_dispatch_one", disp):
            out = router.smart_home_control("turn off the office light")
        self.assertIn("didn't work", out.lower())
        self.assertIn("bridge not connected", out)


# ── 2g. live-LAN fallback + the load-order clobber regression ────────
class RouterLiveLanFallbackTests(unittest.TestCase):
    """2026-07-21 audit: sh_kasa's catalog-free LAN handler registers under the
    same action names as the router, loses the key race (sorted skill order:
    'sh_kasa' < 'smart_home_router_skill'), and the router then dead-ended on
    the empty Alexa-seeded catalog — every LAN plug answered 'No smart-home
    catalog yet'. The router must fall through to the live-LAN path before
    refusing."""

    def _fake_kasa(self, reply="Done, sir — entry light on.", record=None):
        mod = types.ModuleType("skill_sh_kasa")

        def smart_home_control(utterance=""):
            if record is not None:
                record.append(utterance)
            return reply
        mod.smart_home_control = smart_home_control
        return mod

    def test_no_catalog_falls_through_to_live_lan(self):
        record: list = []
        fake = self._fake_kasa(record=record)
        with mock.patch.object(router, "_ensure_catalog", return_value=None), \
             mock.patch.dict(sys.modules, {"skill_sh_kasa": fake}):
            out = router.smart_home_control("turn on the entry light")
        self.assertEqual(out, "Done, sir — entry light on.")
        self.assertEqual(record, ["turn on the entry light"])

    def test_no_catalog_wizard_message_when_lan_also_misses(self):
        # sh_kasa's own no-device refusal → the wizard hint still speaks.
        fake = self._fake_kasa(
            reply="I don't see any controllable smart devices on the "
                  "network yet, sir.")
        with mock.patch.object(router, "_ensure_catalog", return_value=None), \
             mock.patch.dict(sys.modules, {"skill_sh_kasa": fake}):
            out = router.smart_home_control("turn on the entry light")
        self.assertIn("No smart-home catalog", out)

    def test_populated_catalog_no_match_falls_through_to_lan(self):
        # On the LAN but not in the catalog → still reachable.
        fake = self._fake_kasa(reply="Done, sir — garage plug off.")
        with mock.patch.object(router, "_ensure_catalog",
                               return_value=_CATALOG), \
             mock.patch.dict(sys.modules, {"skill_sh_kasa": fake}):
            out = router.smart_home_control("turn off the garage plug")
        self.assertEqual(out, "Done, sir — garage plug off.")

    def test_load_order_clobber_lan_stays_reachable(self):
        # Reproduce the monolith loader's exact sequence for this
        # stale-duplicate bug class: sh_kasa registers its LAN handler first,
        # smart_home_router_skill (sorted later) re-registers the same names
        # and wins. With an EMPTY catalog and one plug on the LAN, the
        # surviving callables must still reach the plug — whatever wins the
        # key race in a future re-shuffle, LAN plugs can never be stranded
        # behind the 'No smart-home catalog' refusal again.
        actions: dict = {}
        kasa_mod, _ = load_skill_isolated("sh_kasa", actions=actions)
        with mock.patch.object(router, "warm_up", return_value=None):
            load_skill_isolated("smart_home_router_skill", actions=actions)
        devs = [{"name": "Entry Light", "lan_ip": "10.0.0.5"}]
        set_state = mock.Mock(return_value={"ok": True})
        with mock.patch.object(router, "_ensure_catalog", return_value=None), \
             mock.patch.object(kasa_mod, "list_devices", return_value=devs), \
             mock.patch.object(kasa_mod, "_tuya_mod", return_value=None), \
             mock.patch.object(kasa_mod, "set_state", set_state):
            out_ctl = actions["smart_home_control"]("turn on entry light")
            out_dev = actions["control_device"]("turn off entry light")
        self.assertNotIn("No smart-home catalog", out_ctl)
        self.assertNotIn("No smart-home catalog", out_dev)
        self.assertIn("entry light on", out_ctl.lower())
        self.assertIn("entry light off", out_dev.lower())
        self.assertEqual(set_state.call_count, 2)


# ── 2f. brand → controller resolution ────────────────────────────────
class RouterBrandResolutionTests(unittest.TestCase):
    def test_controller_for_known(self):
        self.assertEqual(router._controller_for("Philips Hue"), "sh_hue")
        self.assertEqual(router._controller_for("tp-link Kasa"), "sh_kasa")

    def test_controller_for_unknown(self):
        self.assertIsNone(router._controller_for("Wyze"))


if __name__ == "__main__":
    unittest.main()
