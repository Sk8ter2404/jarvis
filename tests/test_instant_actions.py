"""core/instant_actions.py — the no-brain rules for volume, media transport,
lights on/off and printer pause, plus the shadow log they write.

Precision beats recall: every positive case is a plain, whole command; every
exclusion the module documents (questions, polite asks, two commands, a
pronoun, a negation, an unregistered / disallowed / gated action) has a test.
Registries here are SYNTHETIC name sets; "zebra" is a stand-in transcript word
that must never reach the log.

Run: python tools/run_tests.py test_instant_actions
"""
from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import unittest
from unittest import mock

from core import config as cfg
from core import instant_actions as ia

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALLOW = list(cfg.INSTANT_ACTIONS_ALLOW)
REGISTRY = set(ALLOW) | {"play_music", "set_timer", "control_device",
                         "media_playpause", "media_next", "screenshot",
                         "resume_print", "focus_mode"}


def _m(text, registry=None, allow=None, blocked=None):
    return ia.match(text, REGISTRY if registry is None else registry,
                    allow=ALLOW if allow is None else allow, blocked=blocked)


class PositiveMatchTests(unittest.TestCase):

    def _assert(self, text, action, arg=""):
        hit = _m(text)
        self.assertIsNotNone(hit, text)
        self.assertEqual((hit.action, hit.arg), (action, arg), text)

    def test_volume(self):
        for text, action in (("volume up", "volume_up"),
                             ("louder", "volume_up"),
                             ("turn the volume down", "volume_down"),
                             ("quieter", "volume_down"),
                             ("mute", "volume_mute"),
                             ("mute the sound", "volume_mute"),
                             ("unmute", "volume_unmute")):
            self._assert(text, action)

    def test_media_transport(self):
        for text, action in (("pause the music", "pause_music"),
                             ("stop the music", "pause_music"),
                             ("resume the music", "resume_music"),
                             ("unpause", "resume_music"),
                             ("next song", "next_song"),
                             ("skip this song", "next_song"),
                             ("previous track", "previous_song"),
                             ("go back a song", "previous_song")):
            self._assert(text, action)

    def test_lights_carry_the_command_as_the_router_argument(self):
        for text, arg in (("turn off the lights", "turn off the lights"),
                          ("Turn off the office light.",
                           "Turn off the office light"),
                          ("turn the kitchen lights on",
                           "turn the kitchen lights on"),
                          ("switch off the desk lamp",
                           "switch off the desk lamp"),
                          ("turn on all the lights", "turn on all the lights"),
                          ("lights off", "lights off")):
            self._assert(text, "smart_home_control", arg)

    def test_printer_pause(self):
        for text in ("pause the print", "pause printing", "pause the printer",
                     "pause my print"):
            self._assert(text, "pause_print")

    def test_wake_word_please_and_end_punctuation_are_stripped(self):
        for text in ("Jarvis, pause the music.", "hey jarvis pause the music",
                     "please pause the music", "pause the music, please",
                     "pause the music jarvis!"):
            self._assert(text, "pause_music")

    def test_token_is_the_brains_own_shape(self):
        self.assertEqual(_m("next song").token, "[ACTION: next_song]")
        self.assertEqual(_m("lights on").token,
                         "[ACTION: smart_home_control, lights on]")

    def test_acknowledgements_are_short_sentences(self):
        self.assertEqual(_m("volume up").ack, "Volume up, sir.")
        self.assertEqual(_m("lights off").ack, "Lights off, sir.")
        self.assertEqual(_m("pause the print").ack, "Pausing the print, sir.")


class ExclusionTests(unittest.TestCase):

    def _none(self, *texts, **kw):
        for text in texts:
            self.assertIsNone(_m(text, **kw), text)

    def test_questions(self):
        self._none("is the music paused", "what's the volume",
                   "are the lights on", "pause the music?", "lights off?")

    def test_polite_asks(self):
        self._none("can you pause the music", "could you turn off the lights",
                   "Jarvis, can you mute", "would you mind pausing the music",
                   "will you turn the volume down", "please can you pause",
                   "is it possible to pause the print")

    def test_two_commands_in_one_sentence(self):
        self._none("pause the music and turn the volume down",
                   "pause the music then skip", "turn off the lights and the fan",
                   "mute, then pause the print", "pause the music; next song",
                   "turn off the lights or the lamp")

    def test_pronouns_and_negation(self):
        self._none("pause it", "mute it", "turn it up", "turn that off",
                   "turn them off", "don't pause the music",
                   "do not turn off the lights", "never mute")

    def test_bare_words_that_are_not_clear_commands(self):
        self._none("next", "skip", "continue", "go back", "previous", "silence")

    def test_anything_else_is_the_brains(self):
        self._none("play some jazz", "set a 5 minute timer", "dim the lights",
                   "turn the lights on to 50 percent",
                   "turn off the lights in 10 minutes", "turn off",
                   "turn up the volume a bit more for me now please jarvis ok",
                   "resume the print", "take a screenshot", "")

    def test_non_string_and_overlong_input(self):
        self.assertIsNone(ia.match(None, REGISTRY, allow=ALLOW))
        self.assertIsNone(ia.match(42, REGISTRY, allow=ALLOW))
        self.assertIsNone(_m("turn off the " + "very " * 30 + "lights"))

    def test_an_unregistered_action_never_matches(self):
        reg = REGISTRY - {"pause_print", "smart_home_control"}
        self._none("pause the print", "turn off the lights", registry=reg)

    def test_the_allowlist_narrows(self):
        self.assertIsNone(_m("pause the music", allow=["volume_up"]))
        self.assertIsNotNone(_m("volume up", allow=["volume_up"]))
        self.assertIsNone(_m("volume up", allow=[]))
        self.assertIsNone(_m("volume up", allow="volume_up"))

    def test_the_allowlist_can_never_widen(self):
        allow = ALLOW + ["shutdown_jarvis", "play_music", "set_timer"]
        self.assertEqual(ia.normalize_allow(allow), ia.RULE_ACTIONS)
        self.assertIsNone(_m("play some jazz", allow=allow))

    def test_media_key_toggles_are_not_fallbacks(self):
        # Without pause_music the dispatcher would fall back to the
        # media_playpause TOGGLE ("pause" can START a paused player).
        reg = REGISTRY - {"pause_music"}
        self.assertIsNone(_m("pause the music", registry=reg))

    def test_the_callers_gate_can_refuse(self):
        seen = []

        def blocked(name, arg):
            seen.append((name, arg))
            return True

        self.assertIsNone(_m("lights off", blocked=blocked))
        self.assertEqual(seen, [("smart_home_control", "lights off")])

    def test_a_raising_gate_refuses(self):
        def boom(name, arg):
            raise RuntimeError("gate fault")

        self.assertIsNone(_m("volume up", blocked=boom))

    def test_an_action_risk_confirm_rule_refuses(self):
        with mock.patch.object(ia._action_risk, "action_confirm_reason",
                               return_value="spends money"):
            self.assertIsNone(_m("volume up"))


class ModeTests(unittest.TestCase):

    def test_normalize_mode(self):
        self.assertEqual(ia.normalize_mode("ON "), "on")
        self.assertEqual(ia.normalize_mode("off"), "off")
        self.assertEqual(ia.normalize_mode("shadow"), "shadow")
        for junk in (None, "", "yes", 1, True):
            self.assertEqual(ia.normalize_mode(junk), "shadow")


class SpokenLineTests(unittest.TestCase):

    def test_a_result_that_is_already_a_sentence_is_spoken(self):
        hit = _m("pause the music")
        self.assertEqual(ia.spoken_line(hit, "paused Spotify, sir"),
                         "Paused Spotify, sir.")
        nothing = ("Nothing seems to be playing, sir — open Apple Music and "
                   "I'll take it from there.")
        self.assertEqual(ia.spoken_line(hit, nothing), nothing)

    def test_a_terse_result_gets_the_acknowledgement(self):
        self.assertEqual(ia.spoken_line(_m("volume up"), "volume up"),
                         "Volume up, sir.")
        self.assertEqual(ia.spoken_line(_m("volume up"), None),
                         "Volume up, sir.")

    def test_missing_media_keys_are_said_not_hidden(self):
        self.assertEqual(
            ia.spoken_line(_m("volume up"), "pyautogui unavailable"),
            "Media keys unavailable, sir.")


class BrainRanTests(unittest.TestCase):

    def test_only_registered_actions_that_ran(self):
        results = [
            ("pause_music", "paused it, sir", True),
            ("pause_music", "paused it, sir", True),
            ("_unverified_claim", "warning", True),
            ("water_plants", "unknown action: water_plants", False),
            ("set_timer", "⚠  REQUIRES CONFIRMATION: set_timer() — say 'yes'",
             False),
            ("screenshot", "⚠  PUSHBACK: Are you certain?", False),
            ("next_song", "skipped, sir", True),
        ]
        self.assertEqual(ia.brain_ran(results, REGISTRY),
                         ["pause_music", "next_song"])

    def test_junk_is_skipped(self):
        self.assertEqual(ia.brain_ran([None, (), ("volume_up",)], REGISTRY),
                         [])
        self.assertEqual(ia.brain_ran(None, REGISTRY), [])

    def test_agrees_exactly_or_by_alias(self):
        self.assertTrue(ia.agrees("pause_music", ["pause_music"]))
        self.assertFalse(ia.agrees("pause_music", ["media_playpause"]))
        self.assertFalse(ia.agrees("pause_music", []))
        alias = {frozenset({"smart_home_control", "control_device"})}
        same = lambda a, b: frozenset({a, b}) in alias  # noqa: E731
        self.assertTrue(ia.agrees("smart_home_control", ["control_device"],
                                  same_handler=same))

        def boom(a, b):
            raise KeyError(a)

        self.assertFalse(ia.agrees("volume_up", ["volume_down"],
                                   same_handler=boom))


class LogTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "data", ia.LOG_NAME)

    def test_rows_round_trip(self):
        self.assertTrue(ia.append_row(self.path, ia.shadow_row(
            "pause_music", ["pause_music"], True, ts=100.0)))
        self.assertTrue(ia.append_row(self.path, ia.on_row(
            "volume_up", False, ts=200.0)))
        self.assertEqual(ia.read_rows(self.path), [
            {"ts": 100.0, "mode": "shadow", "action": "pause_music",
             "brain": ["pause_music"], "agree": True},
            {"ts": 200.0, "mode": "on", "action": "volume_up", "ok": False},
        ])

    def test_rows_hold_names_only(self):
        # A "name" carrying words is dropped, never written.
        row = ia.shadow_row("pause_music",
                            ["turn off the zebra light", "pause_music"], False)
        self.assertEqual(row["brain"], ["pause_music"])
        self.assertEqual(set(row), {"ts", "mode", "action", "brain", "agree"})
        ia.append_row(self.path, row)
        with open(self.path, encoding="utf-8") as fh:
            self.assertNotIn("zebra", fh.read())

    def test_rotation_keeps_one_generation_and_reads_both(self):
        ia.append_row(self.path, ia.on_row("volume_up", True, ts=1.0))
        ia.append_row(self.path, ia.on_row("volume_down", True, ts=2.0),
                      max_bytes=1)
        self.assertTrue(os.path.isfile(self.path + ".1"))
        self.assertEqual([r["ts"] for r in ia.read_rows(self.path)],
                         [1.0, 2.0])

    def test_torn_and_malformed_lines_are_skipped(self):
        os.makedirs(os.path.dirname(self.path))
        good = json.dumps(ia.on_row("next_song", True, ts=5.0))
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json\n\n" + good + "\n"
                     + json.dumps({"ts": "x", "mode": "on", "action": "a",
                                   "ok": True}) + "\n"
                     + json.dumps({"ts": 1, "mode": "maybe", "action": "a"})
                     + "\n" + json.dumps([1, 2]) + "\n")
        self.assertEqual(ia.read_rows(self.path),
                         [{"ts": 5.0, "mode": "on", "action": "next_song",
                           "ok": True}])

    def test_missing_log_reads_empty_and_unwritable_path_is_false(self):
        self.assertEqual(ia.read_rows(self.path), [])
        blocker = os.path.join(self.tmp.name, "file")
        with open(blocker, "w", encoding="utf-8") as fh:
            fh.write("x")
        self.assertFalse(ia.append_row(os.path.join(blocker, "x.jsonl"),
                                       ia.on_row("volume_up", True)))


class WiringTests(unittest.TestCase):
    """The setting is wired like FAST_PATHS_ENABLED (config literal, Settings
    row, example json), its names are real actions, and its log stays out of
    git."""

    def test_config_defaults(self):
        with open(os.path.join(_ROOT, "core", "config.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        lits = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id.startswith("INSTANT_ACTIONS_")}
        self.assertEqual(lits["INSTANT_ACTIONS_MODE"], "shadow")
        self.assertEqual(set(lits["INSTANT_ACTIONS_ALLOW"]), ia.RULE_ACTIONS)

    def test_every_allowed_name_is_a_registered_action(self):
        with open(os.path.join(_ROOT, "docs", "ACTION_INDEX.md"),
                  encoding="utf-8") as fh:
            indexed = set(re.findall(r"^\| `([a-z0-9_]+)` \|", fh.read(),
                                     re.MULTILINE))
        self.assertGreater(len(indexed), 100)
        self.assertEqual(sorted(set(cfg.INSTANT_ACTIONS_ALLOW) - indexed), [])

    def test_settings_row_and_example_json(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "jarvis_settings_window_instant",
            os.path.join(_ROOT, "tools", "settings_window.py"))
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        row = sw.SCHEMA["INSTANT_ACTIONS_MODE"]
        self.assertEqual(row["default"], "shadow")
        self.assertEqual(list(row["choices"]), list(ia.MODES))
        self.assertIn("INSTANT_ACTIONS_MODE", sw.persisted_keys())
        with open(os.path.join(_ROOT, "tools", "user_settings.example.json"),
                  encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["INSTANT_ACTIONS_MODE"], "shadow")

    def test_the_log_is_git_ignored(self):
        with open(os.path.join(_ROOT, ".gitignore"), encoding="utf-8") as fh:
            lines = {ln.strip() for ln in fh}
        self.assertTrue({"data/*", "data/"} & lines)
        self.assertIn("data_staging/", lines)


if __name__ == "__main__":
    unittest.main()
