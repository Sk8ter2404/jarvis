"""Light-tier regression tests for audit cluster C5 (action honesty and smart
home), July-4 audit leftovers rechecked 2026-10-02.

A44 - "ambient mode on" announced "Chappie is listening quietly and learning"
      even when the mic daemon REFUSED to start (it refuses by returning a line,
      not by raising), and it had already saved AMBIENT_LISTEN_ENABLED=True, so
      the refused start persisted across reboots with no daemon running.
A50 - scenes never activated, and the whole Alexa fallback was dead: it called
      AlexaAPI.set_appliance_state, which the installed alexapy (1.29.22) does
      not have. The call that exists is the static set_light_state(login,
      entity_id, power_on=..., brightness=...), a PUT to /api/phoenix/state
      whose JSON reply says per entity whether Amazon accepted it.

Hermetic: no real device, network, mic or Alexa call. Every settings write goes
to a temp file; alexapy and the discover skill's async runner are faked.

    python -B -m unittest tests.test_audit_c5_honesty
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import core.actions as A
import core.config as cfg
from core import smart_home_router as router
from core.failure_markers import FAILURE_MARKERS

# The skill's own refusal lines (skills/ambient_listen.py ambient_listen_start).
# RealSkillRefusalTests below keeps the first one honest against the real skill.
_EXCLUSIVE_MIC = ("Ambient mode requires an exclusive mic connection — "
                  "stop the wake-word listener first, sir.")
_WORKER_DIED = "Ambient mode failed to start, sir: PortAudio device busy."
_ENGAGED = ("Ambient listening engaged, sir. I'll keep a 10-minute rolling "
            "transcript and stay silent unless I hear my name.")


def _has_failure_marker(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in FAILURE_MARKERS)


_ABSENT = object()


def _swap_modules(case: unittest.TestCase, mods: dict) -> None:
    """Install ``mods`` into sys.modules for one test and put back exactly
    those keys afterwards (absence included). Not mock.patch.dict: that also
    drops every module FIRST imported during the test, and numpy refuses a
    second load in one process."""
    for name, mod in mods.items():
        prior = sys.modules.get(name, _ABSENT)
        sys.modules[name] = mod
        if prior is _ABSENT:
            case.addCleanup(sys.modules.pop, name, None)
        else:
            case.addCleanup(sys.modules.__setitem__, name, prior)


# ──────────────────────────────────────────────────────────────────────────
#  A44 - a refused ambient start is reported, rolled back and never saved
# ──────────────────────────────────────────────────────────────────────────

class _AmbientCase(unittest.TestCase):
    """A Mock bobert_companion, a temp user_settings.json and a fake fact
    extractor. Staging is OFF so the setter really saves (to the temp file)."""

    def setUp(self):
        d = tempfile.mkdtemp(prefix="c5_ambient_")
        self.addCleanup(shutil.rmtree, d, True)
        self.path = os.path.join(d, "user_settings.json")
        env = mock.patch.dict(os.environ, {"JARVIS_SETTINGS_PATH": self.path})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(setattr, cfg, "AMBIENT_LISTEN_ENABLED",
                        cfg.AMBIENT_LISTEN_ENABLED)
        cfg.AMBIENT_LISTEN_ENABLED = False
        self.bc = mock.Mock()
        self.bc._ambient_mode_active = [False]
        self.bc.AMBIENT_LISTEN_ENABLED = False
        self.bc._is_staging = lambda: False
        self.hud = []
        self.bc._write_hud_state.side_effect = lambda **k: self.hud.append(k)
        p = mock.patch.object(A, "_bc", return_value=self.bc)
        p.start()
        self.addCleanup(p.stop)
        self.ext = types.ModuleType("skill_ambient_multimodal_extract")
        self.ext.ambient_extract_start = mock.Mock(return_value="")
        self.ext.ambient_extract_stop = mock.Mock(return_value="")
        _swap_modules(self, {"skill_ambient_multimodal_extract": self.ext})

    def write(self, doc):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(doc, f)

    def saved(self):
        """The saved AMBIENT_LISTEN_ENABLED, or None when nothing was saved."""
        if not os.path.exists(self.path):
            return None
        with open(self.path, encoding="utf-8") as f:
            return json.load(f).get("AMBIENT_LISTEN_ENABLED")

    def start_returns(self, line):
        self.bc.ACTIONS = {"ambient_listen_start": mock.Mock(return_value=line),
                           "ambient_listen_stop": mock.Mock(return_value="")}


class AmbientRefusedStartTests(_AmbientCase):

    def _assert_refused_and_rolled_back(self, out, why):
        self.assertNotIn("listening quietly", out)
        self.assertNotIn("Ambient mode active", out)
        self.assertTrue(_has_failure_marker(out),
                        f"the follow-up loop must see a failure: {out!r}")
        self.assertIn(why, out, "the daemon's own reason must be passed on")
        self.assertIs(self.bc._ambient_mode_active[0], False)
        self.assertEqual(self.hud[-1], {"ambient_mode_active": False})
        self.assertIs(self.bc.AMBIENT_LISTEN_ENABLED, False)
        self.assertIs(cfg.AMBIENT_LISTEN_ENABLED, False)
        self.assertIsNot(self.saved(), True,
                         "a refused start must not be saved ON for next boot")
        self.ext.ambient_extract_start.assert_not_called()

    def test_wake_listener_owns_the_mic(self):
        self.start_returns(_EXCLUSIVE_MIC)
        out = A._act_ambient_mode_set(True)
        self._assert_refused_and_rolled_back(out, "exclusive mic connection")

    def test_worker_died_on_start(self):
        self.start_returns(_WORKER_DIED)
        out = A._act_ambient_mode_set(True)
        self._assert_refused_and_rolled_back(out, "PortAudio device busy")

    def test_start_that_raises_is_rolled_back_too(self):
        self.bc.ACTIONS = {"ambient_listen_start":
                           mock.Mock(side_effect=RuntimeError("mic gone"))}
        out = A._act_ambient_mode_set(True)
        self.assertIn("ambient daemon refused", out)   # the tray logs this prefix
        self._assert_refused_and_rolled_back(out, "mic gone")

    def test_a_refusal_leaves_the_owners_saved_choice_alone(self):
        # Saved ON from an earlier session; the daemon refuses now. The setter
        # writes nothing, so the file and the live flags keep what they held.
        self.write({"AMBIENT_LISTEN_ENABLED": True, "OTHER_KEY": "keep"})
        self.bc.AMBIENT_LISTEN_ENABLED = True
        cfg.AMBIENT_LISTEN_ENABLED = True
        self.start_returns(_EXCLUSIVE_MIC)
        A._act_ambient_mode_set(True)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f),
                             {"AMBIENT_LISTEN_ENABLED": True, "OTHER_KEY": "keep"})
        self.assertIs(self.bc.AMBIENT_LISTEN_ENABLED, True)
        self.assertIs(cfg.AMBIENT_LISTEN_ENABLED, True)
        self.assertIs(self.bc._ambient_mode_active[0], False)


class AmbientAcceptedStartTests(_AmbientCase):
    """The success paths keep working: an accepted start is saved, live, and
    starts the fact extractor; an OFF is still saved."""

    def test_engaged_is_saved_and_learns(self):
        self.start_returns(_ENGAGED)
        out = A._act_ambient_mode_set(True)
        self.assertIn("listening quietly and learning", out)
        self.assertIs(self.saved(), True)
        self.assertIs(self.bc.AMBIENT_LISTEN_ENABLED, True)
        self.assertIs(cfg.AMBIENT_LISTEN_ENABLED, True)
        self.assertIs(self.bc._ambient_mode_active[0], True)
        self.ext.ambient_extract_start.assert_called_once_with("")

    def test_already_active_is_not_a_refusal(self):
        self.start_returns("Ambient listening is already active, sir.")
        out = A._act_ambient_mode_set(True)
        self.assertIn("listening quietly and learning", out)
        self.assertIs(self.saved(), True)

    def test_off_is_still_saved(self):
        self.write({"AMBIENT_LISTEN_ENABLED": True})
        self.bc._ambient_mode_active = [True]
        self.start_returns("")
        out = A._act_ambient_mode_set(False)
        self.assertIn("standing down", out)
        self.assertIs(self.saved(), False)


class RealSkillRefusalTests(_AmbientCase):
    """Drive the REAL skills/ambient_listen.ambient_listen_start, so a reworded
    refusal in the skill cannot quietly turn back into a false success."""

    def test_real_exclusive_mic_refusal_is_reported(self):
        from tests._skill_harness import load_skill_isolated
        # The loader registers sys.modules["skill_ambient_listen"]; the swap
        # puts back whatever was there once the test ends.
        _swap_modules(self, {"skill_ambient_listen": None})
        mod, _ = load_skill_isolated("ambient_listen", register=False)
        with mock.patch.object(mod, "_wake_listener_active", return_value=True):
            self.bc.ACTIONS = {"ambient_listen_start": mod.ambient_listen_start}
            out = A._act_ambient_mode_set(True)
        self.assertIsNone(mod._thread, "the refusal must not start a worker")
        self.assertNotIn("listening quietly", out)
        self.assertTrue(_has_failure_marker(out), out)
        self.assertIs(self.bc._ambient_mode_active[0], False)
        self.assertIsNot(self.saved(), True)


# ──────────────────────────────────────────────────────────────────────────
#  A50 - the Alexa fallback calls what alexapy 1.29.22 really has, and a
#        scene is claimed "running" only when Amazon says SUCCESS
# ──────────────────────────────────────────────────────────────────────────

def _drain(coro):
    """Run a coroutine that never really awaits; return its value."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("coroutine did not complete synchronously")


def _phoenix_ok(entity_id):
    return {"controlResponses": [{"entityId": entity_id, "entityType": "ENTITY",
                                  "code": "SUCCESS"}],
            "errors": []}


class _FakeAlexa:
    """Stand-in for alexapy 1.29.22's AlexaAPI control surface: ONLY the static
    set_light_state that the real class has (no set_appliance_state). Records
    every call; replies like /api/phoenix/state."""

    def __init__(self, reply=None, signature_of=None):
        self.calls = []
        self._reply = reply
        self._sig = signature_of
        fake = self

        async def set_light_state(*args, **kwargs):
            if fake._sig is not None:
                fake._sig.bind(*args, **kwargs)   # TypeError on a wrong shape
            fake.calls.append((args, kwargs))
            if fake._reply is not None:
                return fake._reply
            return _phoenix_ok(args[1] if len(args) > 1 else kwargs.get("entity_id"))

        self.api = types.SimpleNamespace(set_light_state=set_light_state)

    def modules(self):
        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = self.api
        disc = types.ModuleType("skills.smart_home_discover")
        disc._run_async = lambda coro, timeout=None: _drain(coro)
        return {"alexapy": alexapy, "skills.smart_home_discover": disc}

    def power_on_arg(self, call):
        args, kwargs = call
        if "power_on" in kwargs:
            return kwargs["power_on"]
        return args[2] if len(args) > 2 else True


class _RouterCase(unittest.TestCase):
    def setUp(self):
        self.login = object()
        p = mock.patch.object(router, "_alexa_login", return_value=self.login)
        p.start()
        self.addCleanup(p.stop)
        self.alexa = _FakeAlexa()

    def use(self, alexa):
        self.alexa = alexa
        _swap_modules(self, alexa.modules())


class AlexaFallbackRealApiTests(_RouterCase):

    def _dev(self, eid="ent-1"):
        return {"name": "Office Lamp", "alexa_entity_id": eid}

    def test_on_goes_through_set_light_state(self):
        self.use(_FakeAlexa())
        out = router._alexa_set_state(self._dev(), {"on": True})
        self.assertNotIn("error", out)
        self.assertTrue(out.get("ok"))
        self.assertEqual(len(self.alexa.calls), 1)
        args, _kw = self.alexa.calls[0]
        self.assertIs(args[0], self.login)
        self.assertEqual(args[1], "ent-1")
        self.assertIs(self.alexa.power_on_arg(self.alexa.calls[0]), True)

    def test_off_sends_power_off(self):
        self.use(_FakeAlexa())
        out = router._alexa_set_state(self._dev(), {"on": False})
        self.assertTrue(out.get("ok"))
        self.assertIs(self.alexa.power_on_arg(self.alexa.calls[0]), False)

    def test_brightness_is_sent_not_dropped(self):
        # "Set to 50%" is spoken on success, so the level must really be sent.
        self.use(_FakeAlexa())
        out = router._alexa_set_state(self._dev(), {"on": True, "brightness": 50})
        self.assertTrue(out.get("ok"))
        self.assertEqual(self.alexa.calls[0][1].get("brightness"), 50)

    def test_amazon_refusal_is_an_error_not_a_success(self):
        reply = {"controlResponses": [],
                 "errors": [{"entity": {"entityId": "ent-1"},
                             "code": "ENDPOINT_UNREACHABLE",
                             "message": "device offline"}]}
        self.use(_FakeAlexa(reply=reply))
        out = router._alexa_set_state(self._dev(), {"on": True})
        self.assertIn("error", out)
        self.assertIn("ENDPOINT_UNREACHABLE", out["error"])

    def test_a_non_success_code_is_an_error(self):
        reply = {"controlResponses": [{"entityId": "ent-1", "code": "FAILURE"}],
                 "errors": []}
        self.use(_FakeAlexa(reply=reply))
        out = router._alexa_set_state(self._dev(), {"on": True})
        self.assertIn("error", out)
        self.assertIn("FAILURE", out["error"])

    def test_an_empty_reply_is_an_error(self):
        self.use(_FakeAlexa(reply={}))
        out = router._alexa_set_state(self._dev(), {"on": True})
        self.assertIn("error", out)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("alexapy") is not None,
    "alexapy not installed (the light CI runner) - contract runs locally")
class InstalledAlexapyContractTests(_RouterCase):
    """Bind the router's call against the INSTALLED alexapy's real signature,
    so a fake shaped by the test's author cannot hide drift (the old fakes
    offered a set_appliance_state that no installed alexapy had)."""

    def test_call_binds_to_the_real_set_light_state(self):
        import alexapy
        real = getattr(alexapy.AlexaAPI, "set_light_state", None)
        self.assertTrue(callable(real), "installed alexapy has no set_light_state")
        self.use(_FakeAlexa(signature_of=inspect.signature(real)))
        out = router._alexa_set_state(
            {"name": "Office Lamp", "alexa_entity_id": "ent-1"},
            {"on": True, "brightness": 40})
        self.assertTrue(out.get("ok"), out)
        self.assertEqual(len(self.alexa.calls), 1)


class SceneActivationTests(_RouterCase):
    """End to end through smart_home_control with a catalog holding an Alexa
    scene entity and a light that ties it on name score."""

    def setUp(self):
        super().setUp()
        self.catalog = {"devices": [
            # Listed FIRST and scores the same as the scene for "movie night".
            {"name": "Night", "alexa_room": "Movie Room", "type": "light",
             "brand": "Philips Hue", "controller_skill": "sh_hue",
             "alexa_entity_id": "light-1"},
            {"name": "Movie Night", "alexa_room": "", "type": "scene",
             "brand": "Philips Hue", "controller_skill": "sh_hue",
             "alexa_entity_id": "scene-1"},
        ]}
        p = mock.patch.object(router, "_ensure_catalog",
                              return_value=self.catalog)
        p.start()
        self.addCleanup(p.stop)
        self.skill = mock.patch.object(
            router, "_call_skill",
            return_value={"error": "bulb 'Movie Night' not found on bridge"})
        self.call_skill = self.skill.start()
        self.addCleanup(self.skill.stop)

    def test_run_the_scene_activates_the_scene_entity(self):
        self.use(_FakeAlexa())
        out = router.smart_home_control("run the movie night scene")
        self.assertTrue(out.startswith("Scene running"), out)
        self.assertIn("Movie Night", out)
        self.assertEqual([c[0][1] for c in self.alexa.calls], ["scene-1"],
                         "only the scene entity may be driven, never the light")
        self.assertIs(self.alexa.power_on_arg(self.alexa.calls[0]), True)
        # No brand skill knows scenes: one would at best fail, at worst switch
        # a same-named bulb and report the scene as running.
        self.call_skill.assert_not_called()

    def test_scene_amazon_refused_is_not_reported_running(self):
        reply = {"controlResponses": [],
                 "errors": [{"code": "INVALID_ACTION", "message": "no scene"}]}
        self.use(_FakeAlexa(reply=reply))
        out = router.smart_home_control("run the movie night scene")
        self.assertNotIn("Scene running", out)
        self.assertTrue(_has_failure_marker(out), out)
        self.assertIn("INVALID_ACTION", out)

    def test_scene_with_no_usable_alexapy_call_says_so(self):
        alexa = _FakeAlexa()
        alexa.api = types.SimpleNamespace()     # an alexapy with no control call
        self.use(alexa)
        out = router.smart_home_control("run the movie night scene")
        self.assertNotIn("Scene running", out)
        self.assertTrue(_has_failure_marker(out), out)


if __name__ == "__main__":
    unittest.main()
