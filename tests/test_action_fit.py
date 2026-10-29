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


class FaceLookTests(unittest.TestCase):
    """Rule 4 (live 2026-10-27 20:38:11): "He does not sound happy." about
    the desk robot ran recognize_face and "That's you, sir." was spoken."""

    _REMARKS = ("He does not sound happy.", "she seems upset",
                "Jarvis, he does not sound happy.", "That sounds great.",
                "He is not happy with me", "is he okay?", "he looks tired")
    _ASKS = ("who is that?", "do you recognize me?", "who's at my desk?",
             "Jarvis, who am I?", "Jarvis, can you see me?",
             "is anyone there?", "do you know who's in front of the computer?",
             "do you recognise my face?", "is my brother here?",
             "is she at the desk?", "am I on camera?", "is that me?",
             "try again", "who’s at the desk?")

    def test_remarks_flag_every_face_alias(self):
        for text in self._REMARKS:
            self.assertFalse(af.asks_face_look(text), text)
            for nm in sorted(af.FACE_LOOK_ACTIONS):
                self.assertIsNotNone(af.check(text, nm), (text, nm))

    def test_face_asks_pass(self):
        for text in self._ASKS:
            self.assertTrue(af.asks_face_look(text), text)
            for nm in sorted(af.FACE_LOOK_ACTIONS):
                self.assertIsNone(af.check(text, nm), (text, nm))

    def test_other_actions_are_not_rule_four(self):
        self.assertIsNone(af.check("He does not sound happy.", "get_time"))
        self.assertIsNone(af.check("He does not sound happy.",
                                   "face_id_status"))

    def test_no_owner_words_never_flags(self):
        self.assertIsNone(af.check("", "recognize_face"))
        self.assertFalse(af.asks_face_look(None))


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


class RemarkRerunTests(unittest.TestCase):
    """Rule 5 (v3.0.2). Live v3.0.0 20:28:45: "what version are you
    running?" -> version_info, then "Yay, version 3 is done." ran
    version_info AGAIN. A remark about the reading JARVIS just gave is
    answered, not re-run; an ask for the reading still runs."""
    PRIOR = [("version_info", "")]

    def test_the_live_remark_is_flagged(self):
        r = af.check("Yay, version 3 is done.", "version_info", "",
                     self.PRIOR, ["version_info"])
        self.assertIsNotNone(r)
        self.assertIn("remarked", r)

    def test_remarks_about_other_readings(self):
        for text, nm in (
                ("Nice, that was quick.", "get_time"),
                ("Oh wow, warm today.", "weather_briefing"),
                ("Sweet, all green then.", "system_pulse"),
        ):
            self.assertIsNotNone(af.check(text, nm, "", [(nm, "")], [nm]),
                                 text)

    def test_asks_for_the_reading_again_still_run(self):
        for text in ("Check again.", "What version again?",
                     "Jarvis, what version are you running?",
                     "Is it still version 3?", "Tell me the version.",
                     "Version 3 is done, right?", "Run it again please.",
                     "Jarvis, read me the version."):
            self.assertIsNone(af.check(text, "version_info", "",
                                       self.PRIOR, ["version_info"]), text)

    def test_first_person_plans_are_not_remarks(self):
        # Live turn 2026-10-06 10:41:28: "I have to leave for class now, it
        # starts at eleven-thirty." ran get_time after get_time, rightly.
        text = ("Jarvis, I have to leave for class now, it starts at "
                "eleven-thirty.")
        self.assertFalse(af.is_remark(text))
        self.assertIsNone(af.check(text, "get_time", "", [("get_time", "")],
                                   ["get_time"]))

    def test_needs_the_same_reading_last_turn(self):
        self.assertIsNone(af.check("Yay, version 3 is done.", "version_info",
                                   "", [], ["version_info"]))
        self.assertIsNone(af.check("Yay, version 3 is done.", "version_info",
                                   "", [("get_time", "")], ["version_info"]))
        self.assertIsNone(af.check("Nice.", "system_pulse", "cpu",
                                   [("system_pulse", "")], ["system_pulse"]))

    def test_control_actions_are_never_rule_five(self):
        self.assertIsNone(af.check("Yay, lights are on.", "lights_on", "",
                                   [("lights_on", "")], ["lights_on"]))

    def test_is_remark_edges(self):
        self.assertTrue(af.is_remark("Yay, version 3 is done."))
        self.assertTrue(af.is_remark("Jarvis, splendid."))
        self.assertFalse(af.is_remark(""))
        self.assertFalse(af.is_remark("what's the time?"))
        self.assertFalse(af.is_remark("refresh that"))
        self.assertFalse(af.is_remark("I love it"))
        # v3.0.2 review: a question word opening a later clause, and
        # "once more", ask (the recogniser often drops the "?")
        self.assertFalse(af.is_remark("Yay, what version is that"))
        self.assertFalse(af.is_remark("Nice. How long has it been up"))
        self.assertFalse(af.is_remark("Cool, once more"))
        self.assertTrue(af.is_informative_action("version_info"))
        self.assertTrue(af.is_informative_action("list_timers"))
        self.assertFalse(af.is_informative_action("lights_on"))


class RobotSpeechTests(unittest.TestCase):
    """Rule 6 (v3.0.6 B7; live 2026-10-28 17:26:20): "We need to give the
    deskbot more voice lines." ran deskbot_speak ("I am ready for more
    dialogue.") and the robot said it."""

    # The live words and action named the owner's robot; "deskbot" is a
    # made-up stand-in of the same shape (public code never names it).
    LIVE = "We need to give the deskbot more voice lines."
    _UNASKED = (LIVE, "Jarvis, the deskbot needs a new battery",
                "he does not sound happy", "the robot is on the desk",
                "we should make him smarter")
    # The live fixture's two deskbot_speak turns, plus the shapes the
    # skill is for.
    _ASKS = ("Jarvis, make him talk.", "Jarvis, make the deskbut talk.",
             "have the deskbot say hi", "tell the robot I aced my exam",
             "let him introduce himself", "make him say good morning",
             "deskbot, greet my roommate", "get him to tell a joke",
             "have it read this out", "make the desk bot roast me")

    def test_the_live_turn_flags(self):
        reason = af.check(self.LIVE, "deskbot_speak", "I am ready for more "
                          "dialogue.")
        self.assertIsNotNone(reason)
        self.assertIn("deskbot_speak", reason)

    def test_unasked_speech_flags(self):
        for text in self._UNASKED:
            with self.subTest(text=text):
                self.assertFalse(af.asks_robot_speech(text), text)
                self.assertIsNotNone(af.check(text, "deskbot_speak", "hi"))

    def test_asked_speech_fits(self):
        for text in self._ASKS:
            with self.subTest(text=text):
                self.assertTrue(af.asks_robot_speech(text), text)
                self.assertIsNone(af.check(text, "deskbot_speak", "hi"))

    def test_other_robot_actions_are_not_rule_six(self):
        self.assertIsNone(af.check(self.LIVE, "deskbot_banter"))
        self.assertIsNone(af.check(self.LIVE, "deskbot_engage"))
        self.assertIsNone(af.check(self.LIVE, "say_hello"))
        self.assertIsNone(af.check(self.LIVE, "tts_speak"))

    def test_any_bot_speak_action_is_recognised_by_name(self):
        for nm in ("deskbot_speak", "DeskBot_Speak", "desk_bot_speak",
                   "robot_speak", "deskbott_speak"):
            with self.subTest(nm=nm):
                self.assertTrue(af.is_robot_speak_action(nm), nm)
                self.assertIsNotNone(af.check(self.LIVE, nm, "hi"))
        for nm in ("deskbot_banter", "speak", "tts_speak", "bot_speaker",
                   "", None):
            with self.subTest(nm=nm):
                self.assertFalse(af.is_robot_speak_action(nm), nm)

    def test_no_owner_words_never_flags(self):
        self.assertIsNone(af.check("", "deskbot_speak", "hello"))
        self.assertFalse(af.asks_robot_speech(None))


if __name__ == "__main__":
    unittest.main()
