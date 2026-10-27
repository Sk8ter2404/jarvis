"""core/action_fit.py - the action-fit guard (v3 W11).

Pinned, stdlib only (light tier):
  * the live misfits are flagged: c006 "get out of the way" -> game_mode_on,
    c030 "turn on all smart home devices" -> list_smart_home_devices, c032
    the same which_mic_is_active right after "the headset is on though",
    safety-03 "turn it off" -> game_mode_off (and c038 "any resources I can
    free up" -> game_mode_on);
  * ZERO false flags on every action that fits the 140 brain_ab harness
    items (tests/_action_fit_harness_items.json: every action the expectation
    accepts plus every action a model ran on a run scored correct);
  * ZERO false flags on the 224 live owner voice turns that ran an action
    (tests/_action_fit_live_turns.json, from the session logs), except the
    three turns the owner called wrong, which ARE flagged;
  * each rule's edges: a lookup beside the action that does the job, a list
    asked for in the same breath, a control action retried after a
    correction, the game-mode skill's own documented phrases, no owner words;
  * the mode normaliser, the failure result's marker and the history window.

The monolith wiring (_runner: shadow logs only, on refuses without running
the handler, off is the old path) is in
tests/monolith/test_monolith_action_fit.py.

    python -m unittest tests.test_action_fit
"""
from __future__ import annotations

import json
import os
import sys
import unittest

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import action_fit as af  # noqa: E402
from core.failure_markers import FAILURE_MARKERS  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str) -> dict:
    with open(os.path.join(_HERE, name), encoding="utf-8") as fh:
        return json.load(fh)


class LiveMisfitsTests(unittest.TestCase):
    def test_c006_get_out_of_the_way_is_not_game_mode(self):
        r = af.check("Jarvis, get out of the way for just a second.",
                     "game_mode_on", "", [], ["game_mode_on"])
        self.assertIsNotNone(r)
        self.assertIn("game_mode_on", r)

    def test_c030_turn_on_all_is_not_a_list(self):
        r = af.check("Jarvis, turn on all smart home devices.",
                     "list_smart_home_devices", "",
                     [("smart_home_control", "turn on the fan")],
                     ["list_smart_home_devices"])
        self.assertIsNotNone(r)
        self.assertIn("turn on all smart home devices", r)
        self.assertIn("list_smart_home_devices", r)

    def test_c032_same_check_after_a_correction(self):
        r = af.check("Jarvis, the headset is on though. I'm using it right "
                     "now.", "which_mic_is_active", "",
                     [("which_mic_is_active", "")], ["which_mic_is_active"])
        self.assertIsNotNone(r)
        self.assertIn("which_mic_is_active", r)

    def test_safety03_turn_it_off_is_not_game_mode_off(self):
        self.assertIsNotNone(af.check("Jarvis, turn it off.",
                                      "game_mode_off"))

    def test_aliases_of_game_mode_are_heavy_too(self):
        for nm in ("low_power_mode", "normal_power", "full_power"):
            self.assertIsNotNone(af.check("Jarvis, get out of the way.", nm),
                                 nm)


class HarnessItemsTests(unittest.TestCase):
    """Every action that fits a harness item passes; the known wrong ones
    are flagged."""

    @classmethod
    def setUpClass(cls):
        cls.items = _load("_action_fit_harness_items.json")["items"]

    def test_fixture_is_the_whole_harness(self):
        self.assertGreaterEqual(len(self.items), 134)
        self.assertGreater(sum(len(i["fit"]) for i in self.items), 200)

    def test_zero_false_flags_on_fitting_actions(self):
        flagged = []
        for it in self.items:
            for nm in it["fit"]:
                for others in ((nm,), tuple(it["fit"])):
                    r = af.check(it["utterance"], nm, "", it["prior"],
                                 others)
                    if r:
                        flagged.append((it["id"], nm, r))
        self.assertEqual(flagged, [])

    def test_known_wrong_actions_are_flagged(self):
        wrong = [(it, nm) for it in self.items for nm in it["wrong"]]
        self.assertEqual(len(wrong), 5)
        for it, nm in wrong:
            prior_args = {n: a for n, a in it["prior"]}
            self.assertIsNotNone(
                af.check(it["utterance"], nm, prior_args.get(nm, ""),
                         it["prior"], [nm]), it["id"])


class LiveTurnsTests(unittest.TestCase):
    """The owner's own voice turns: only the three he called wrong flag."""

    @classmethod
    def setUpClass(cls):
        cls.turns = _load("_action_fit_live_turns.json")["turns"]

    def test_fixture_size(self):
        self.assertGreaterEqual(len(self.turns), 85)

    def test_only_the_known_wrong_turns_flag(self):
        hits = {}
        for t in self.turns:
            names = [n for n, _ in t["actions"]]
            for nm, arg in t["actions"]:
                if af.check(t["utterance"], nm, arg, t["prior"], names):
                    hits.setdefault(t["id"], []).append(nm)
        expected = {t["id"] for t in self.turns if t["wrong"]}
        self.assertEqual(len(expected), 3)
        self.assertEqual(set(hits), expected)
        by_id = {t["id"]: t["wrong"] for t in self.turns if t["wrong"]}
        self.assertEqual(sorted(by_id.values()), ["c006", "c030", "c032"])


class RuleEdgeTests(unittest.TestCase):
    def test_no_owner_words_never_flags(self):
        self.assertIsNone(af.check("", "game_mode_on"))
        self.assertIsNone(af.check(None, "list_smart_home_devices"))

    def test_game_mode_phrases_pass(self):
        for text in ("Jarvis, game mode on.", "Jarvis, I'm gaming.",
                     "Jarvis, I'm about to play Fortnite.",
                     "Jarvis, divert power.", "Jarvis, normal power.",
                     "Jarvis, free up some memory.", "Jarvis, exit game mode.",
                     "Jarvis, I'm done gaming.", "Jarvis, come back."):
            for nm in ("game_mode_on", "game_mode_off"):
                self.assertIsNone(af.check(text, nm), (text, nm))

    def test_light_mode_swaps_are_not_heavy(self):
        self.assertIsNone(af.check("Jarvis, get out of the way.",
                                   "focus_mode_on"))

    def test_lookup_beside_the_control_action_passes(self):
        self.assertIsNone(af.check(
            "Jarvis, turn on all smart home devices.",
            "list_smart_home_devices", "", [],
            ["list_smart_home_devices", "smart_home_control"]))

    def test_list_asked_for_in_the_same_breath_passes(self):
        for text in ("Jarvis, turn on the lights and tell me which are on.",
                     "Jarvis, check whether the fan is off.",
                     "Jarvis, what devices can you turn on?"):
            self.assertIsNone(
                af.check(text, "list_smart_home_devices"), text)

    def test_status_for_a_question_passes(self):
        self.assertIsNone(af.check("Jarvis, which mic are you using?",
                                   "which_mic_is_active"))

    def test_set_and_dim_commands(self):
        self.assertIsNotNone(af.check("Jarvis, dim the bedroom lights.",
                                      "list_smart_home_devices"))
        self.assertIsNotNone(af.check(
            "Jarvis, could you set the lamp to red?",
            "list_smart_home_devices"))
        # "set up" is not a set command
        self.assertIsNone(af.check("Jarvis, set up my schedule.",
                                   "list_schedule"))

    def test_control_action_is_never_rule_one(self):
        self.assertIsNone(af.check("Jarvis, turn on the fan.",
                                   "smart_home_control", "turn on the fan"))

    def test_control_retry_after_correction_passes(self):
        self.assertIsNone(af.check(
            "Jarvis, no, I said turn on the fan.", "smart_home_control",
            "turn on the fan", [("smart_home_control", "turn on the fan")]))

    def test_repeat_needs_the_same_argument(self):
        self.assertIsNone(af.check(
            "Jarvis, no, the other one.", "list_schedule", "tomorrow",
            [("list_schedule", "today")]))
        self.assertIsNotNone(af.check(
            "Jarvis, no, that's wrong.", "list_schedule", "Today",
            [("list_schedule", "today")]))

    def test_repeat_needs_a_correction(self):
        self.assertIsNone(af.check(
            "Jarvis, which mic is active?", "which_mic_is_active", "",
            [("which_mic_is_active", "")]))

    def test_fresh_reading_asks_are_not_corrections(self):
        # Review W11 #1: polling questions and explicit re-checks want a
        # fresh reading; the same action with the same argument must pass.
        cases = (
            ("No, check again.", "which_mic_is_active", ""),
            ("I plugged it in already, check again", "which_mic_is_active",
             ""),
            ("Is the download still running?", "check_download_status", ""),
            ("Is the server still up?", "check_server_status", ""),
            ("I already restarted it, is it working now?",
             "check_server_status", ""),
            ("What's the CPU at now? Still high?", "system_pulse", ""),
            ("Actually, what time is it?", "get_time", ""),
            ("Refresh it, I already fixed it", "get_weather", ""),
            ("No, re-check the mic.", "which_mic_is_active", ""),
            ("That's not right, run it one more time.", "system_pulse", ""),
        )
        for text, name, arg in cases:
            for said in (text, "Jarvis, " + text):
                with self.subTest(said=said):
                    self.assertFalse(af.is_correction(said))
                    self.assertIsNone(af.check(said, name, arg,
                                               [(name, arg)], [name]))

    def test_bare_still_already_actually_are_not_corrections(self):
        for said in ("It's still loading.", "I already plugged it in.",
                     "Actually the headset is on.", "Though it's late, ok.",
                     "It's not a big deal."):
            with self.subTest(said=said):
                self.assertFalse(af.is_correction(said))

    def test_plan_correction_words_still_flag(self):
        for said in ("No, the headset.", "Nope.", "That's not it.",
                     "Wrong mic.", "I said the headset.",
                     "The headset is on though.", "It's on though, I'm "
                     "using it right now."):
            with self.subTest(said=said):
                self.assertTrue(af.is_correction("Jarvis, " + said))
                self.assertIsNotNone(af.check(
                    "Jarvis, " + said, "which_mic_is_active", "",
                    [("which_mic_is_active", "")], ["which_mic_is_active"]))

    def test_repeat_needs_the_previous_turn(self):
        self.assertIsNone(af.check(
            "Jarvis, no, the headset.", "which_mic_is_active", "", []))

    def test_bad_input_never_raises(self):
        self.assertIsNone(af.check(object(), object(), object(),
                                   [object(), (1,)], [None]))


class HelperTests(unittest.TestCase):
    def test_normalise_mode(self):
        self.assertEqual(af.normalise_mode("ON"), "on")
        self.assertEqual(af.normalise_mode(" shadow "), "shadow")
        self.assertEqual(af.normalise_mode("off"), "off")
        self.assertEqual(af.normalise_mode("yes"), "off")
        self.assertEqual(af.normalise_mode(None), "off")

    def test_failure_result_is_a_failure(self):
        res = af.failure_result("not run: x").lower()
        self.assertTrue(any(m in res for m in FAILURE_MARKERS))
        self.assertTrue(res.startswith("refused (action-fit): "))

    def test_prior_from_history_window(self):
        hist = [{"action": "old", "arg": "", "at": 100.0},
                {"action": "which_mic_is_active", "arg": "", "at": 950.0},
                {"action": "this_turn", "arg": "", "at": 1001.0}]
        self.assertEqual(af.prior_from_history(hist, 1000.0, 120.0),
                         [("which_mic_is_active", "")])
        self.assertEqual(af.prior_from_history(hist, 0.0), [])
        self.assertEqual(af.prior_from_history([{"at": "x"}], 10.0), [])

    def test_is_status_action(self):
        for nm in ("list_smart_home_devices", "which_mic_is_active",
                   "game_mode_status", "check_print", "system_pulse"):
            self.assertTrue(af.is_status_action(nm), nm)
        for nm in ("smart_home_control", "game_mode_on", "use_headset_mic",
                   "see_screen", ""):
            self.assertFalse(af.is_status_action(nm), nm)


if __name__ == "__main__":
    unittest.main()
