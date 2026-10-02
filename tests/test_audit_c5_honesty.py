"""Light-tier regression tests for audit cluster C5 (action honesty and smart
home), July-4 audit leftovers rechecked 2026-10-02.

A44 - "ambient mode on" announced "Chappie is listening quietly and learning"
      even when the mic daemon REFUSED to start (it refuses by returning a line,
      not by raising), and it had already saved AMBIENT_LISTEN_ENABLED=True, so
      the refused start persisted across reboots with no daemon running.

Hermetic: no real device, network or mic. Every settings write goes to a temp
file.

    python -B -m unittest tests.test_audit_c5_honesty
"""
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
