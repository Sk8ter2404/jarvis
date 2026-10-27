"""command_autocorrect verb-clash guard (v3 W04, 2026-10-27).

The fuzzy corrector's silent route ran the wrong action when it fired
(offline replay, JARVIS-Research/v3_overhaul_20261027/05_action_vocabulary.md
section 4, embeddings off as live): play_video -> play_vibe 0.94,
toggle_mute -> toggle_hud 0.82, send_email -> read_email 0.82,
quit -> quiet_mode 0.80, open_app -> open_url 0.76.

Pinned, against the real scorer over the scanned registry (embeddings off):
  * with verb_clash=True those five are no longer "silent" (they are "none",
    with the refused pick in ``clash``), so the dispatcher takes the
    existing unknown-action path;
  * legitimate typo routes still route: see_scren -> see_screen,
    launch_ap -> launch_app, list_windows_all -> list_windows, ...;
  * verb_clash=False (the default, and ACTION_ALIASES off) returns exactly
    the old result;
  * "blocked" and "ambiguous" are untouched by the guard.

    python -m unittest tests.test_command_autocorrect_clash
"""
from __future__ import annotations

import os
import sys
import unittest

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import command_autocorrect as ca  # noqa: E402
from tests.test_action_aliases import _registry  # noqa: E402

# The five silent mis-routes and the wrong pick each made.
MISROUTES = {
    "play_video": "play_vibe",
    "toggle_mute": "toggle_hud",
    "send_email": "read_email",
    "quit": "quiet_mode",
    "open_app": "open_url",
}

# Invented name -> the right action the silent route must keep finding.
GOOD_ROUTES = {
    "see_scren": "see_screen",
    "launch_ap": "launch_app",
    "open_ur": "open_url",
    "list_windows_all": "list_windows",
    "describe_screen": "local_describe_screen",
    "take_screenshot": "screenshot",
    "set_volum": "set_volume",
    "volume_mut": "volume_mute",
    "youtube_plya": "youtube_play",
    "set_timer_for": "set_timer",
    "media_play_pause": "media_playpause",
}


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._embed_was = ca._embed_disabled
        ca._embed_disabled = True        # lexical only, never Ollama
        cls.reg = sorted(_registry())

    @classmethod
    def tearDownClass(cls):
        ca._embed_disabled = cls._embed_was

    def choice(self, name, **kw):
        return ca.autocorrect_command_choice(name, self.reg, 0.75, 0.10,
                                             use_embeddings=False, **kw)


class VerbClashTests(_Base):
    def test_old_path_still_misroutes(self):
        # The evidence the guard exists for - and proof the default path is
        # unchanged.
        for name, wrong in MISROUTES.items():
            with self.subTest(name=name):
                c = self.choice(name)
                self.assertEqual(c["status"], "silent")
                self.assertEqual(c["primary"][0], wrong)
                self.assertNotIn("clash", c)

    def test_misroutes_are_no_longer_silent(self):
        for name, wrong in MISROUTES.items():
            with self.subTest(name=name):
                c = self.choice(name, verb_clash=True)
                self.assertEqual(c["status"], "none")
                self.assertEqual(c["clash"][0], wrong)
                self.assertEqual(c["primary"][0], wrong)

    def test_good_routes_still_route(self):
        for name, right in GOOD_ROUTES.items():
            with self.subTest(name=name):
                c = self.choice(name, verb_clash=True)
                self.assertEqual(c["status"], "silent", c)
                self.assertEqual(c["primary"][0], right)

    def test_off_is_identical_to_the_old_call(self):
        for name in list(MISROUTES) + list(GOOD_ROUTES) + [
                "read_screen", "volume_set", "shutdown", "close_all_windows",
                "move_file", "pause"]:
            with self.subTest(name=name):
                self.assertEqual(
                    self.choice(name, verb_clash=False),
                    ca.autocorrect_command_choice(name, self.reg, 0.75, 0.10,
                                                  use_embeddings=False))

    def test_blocked_and_ambiguous_untouched(self):
        protected = {"shutdown_jarvis", "close_all_windows_except"}
        for name in ("shutdown", "close_all_windows"):
            with self.subTest(name=name):
                c = self.choice(name, verb_clash=True, protected=protected)
                self.assertEqual(c["status"], "blocked")
        for name in ("read_screen", "volume_set"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.choice(name, verb_clash=True),
                    self.choice(name))


class TurnCheckerAgreesTests(_Base):
    """core.turn_checker scores a made-up name the way _runner does: with the
    guard on, send_email (no action ran) is a made-up action, not resolved."""

    def test_made_up_with_guard_on(self):
        from core import turn_checker as tc
        args = ("email bob for me", "On it, sir. [ACTION: send_email, bob]",
                ["send_email"], [], self.reg)
        self.assertEqual(tc.check_turn(*args).kind, tc.OK)
        self.assertEqual(tc.check_turn(*args, verb_clash=True).kind,
                         tc.MADE_UP_ACTION)

    def test_good_typo_still_resolved_with_guard_on(self):
        from core import turn_checker as tc
        v = tc.check_turn("look at my screen", "[ACTION: see_scren]",
                          ["see_scren"], [], self.reg, verb_clash=True)
        self.assertNotEqual(v.kind, tc.MADE_UP_ACTION)


class VerbClashPredicateTests(unittest.TestCase):
    def test_different_classes_clash(self):
        self.assertTrue(ca._verb_clash("send_email", "read_email"))
        self.assertTrue(ca._verb_clash("quit", "quiet_mode"))
        self.assertTrue(ca._verb_clash("play_thing", "open_thing"))

    def test_same_verb_other_object_clashes(self):
        self.assertTrue(ca._verb_clash("play_video", "play_vibe"))
        self.assertTrue(ca._verb_clash("toggle_mute", "toggle_hud"))
        self.assertTrue(ca._verb_clash("open_app", "open_url"))

    def test_synonyms_and_typos_do_not_clash(self):
        self.assertFalse(ca._verb_clash("show_windows", "list_windows"))
        self.assertFalse(ca._verb_clash("open_app", "launch_app"))
        self.assertFalse(ca._verb_clash("see_scren", "see_screen"))
        self.assertFalse(ca._verb_clash("launch_ap", "launch_app"))
        self.assertFalse(ca._verb_clash("list_windows_all", "list_windows"))

    def test_non_verb_lead_never_clashes(self):
        self.assertFalse(ca._verb_clash("network_scan", "network_usage"))
        self.assertFalse(ca._verb_clash("ambient_learning_mode",
                                        "ambient_listening"))
        self.assertFalse(ca._verb_clash("", "see_screen"))

    def test_bare_verb_never_clashes(self):
        self.assertFalse(ca._verb_clash("play", "play_music"))
        self.assertFalse(ca._verb_clash("toggle_hud", "toggle"))

    def test_filler_verbs_are_skipped(self):
        # "start"/"stop"/"enable" are filler: _polarity_clash owns them.
        self.assertFalse(ca._verb_clash("start_app", "app_launcher"))


if __name__ == "__main__":
    unittest.main()
