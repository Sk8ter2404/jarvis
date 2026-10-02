"""Tests for core/turn_checker.py — the decision half of the local -> Claude
escalation tier: is this turn a said-but-did-nothing, a command with no
action, or a made-up action, and is it worth one retry on Claude?

A table of realistic, generic turns (no LLM, a fake registry). Precision is
the point: every negative must come back "ok", and per kind every turn the
checker flags must be that kind. The question-guard mutation test proves the
"asks the owner something" guard is what keeps the questions-back rows green.

    python tools/run_tests.py test_turn_checker
"""
from __future__ import annotations

import inspect
import os
import re
import unittest
from collections import namedtuple
from unittest import mock

import command_autocorrect
from core import tts_render_cache
from core import turn_checker as tc

REGISTERED = frozenset({
    "set_timer", "list_timers", "cancel_timer", "play_music", "pause_music",
    "resume_music", "next_song", "previous_song", "volume_up", "volume_down",
    "volume_mute", "volume_unmute", "screenshot", "see_screen", "show_tasks",
    "focus_mode", "get_time", "weather_briefing", "web_search", "open_url",
    "youtube", "launch_app", "close_app", "smart_home_control", "send_email",
    "system_pulse", "restart_jarvis", "media_playpause",
})

Row = namedtuple("Row", "user reply emitted ran kind asked")


def T(kind, user, reply, emitted=(), ran=(), asked=None):
    return Row(user, reply, tuple(emitted), tuple(ran), kind, asked)


def _check(row, registered=REGISTERED):
    return tc.check_turn(row.user, row.reply, row.emitted, row.ran,
                         registered, asked_question=row.asked)


SAID, CMD, MADE, OK = (tc.SAID_NO_ACTION, tc.COMMAND_NO_ACTION,
                       tc.MADE_UP_ACTION, tc.OK)

# ── turns that failed ────────────────────────────────────────────────────────
POSITIVES = (
    # said but did nothing: no token, nothing ran, the reply claims it acted
    T(SAID, "set a five minute timer",
      "I've set a timer for five minutes, sir."),
    T(SAID, "turn off the lights", "Done, sir. The lights are off."),
    T(SAID, "open youtube", "Opening YouTube now, sir."),
    T(SAID, "play something relaxing",
      "Playing a relaxing playlist for you now, sir."),
    T(SAID, "restart yourself", "Restarting now, sir."),
    T(SAID, "send the email", "Sent, sir."),
    T(SAID, "close the browser", "I've closed the browser, sir."),
    T(SAID, "turn the volume down", "I've turned the volume down, sir."),
    T(SAID, "take a screenshot", "Taking a screenshot now, sir."),
    T(SAID, "remind me in ten minutes to check the oven",
      "I'll remind you in ten minutes, sir."),
    T(SAID, "search the web for the best hiking boots",
      "I've searched the web for hiking boots, sir."),
    T(SAID, "mute the sound", "Muted, sir."),
    T(SAID, "dim the lights", "Very good, sir, turning the lights down."),
    T(SAID, "launch notepad", "Launching Notepad, sir."),
    T(SAID, "skip this song", "Skipped, sir."),
    T(SAID, "move this window to the left monitor",
      "I've moved it to your left monitor, sir."),
    T(SAID, "pause the music", "On it, sir."),
    # a clear command, no token, nothing ran, no claim and no question
    T(CMD, "take a screenshot", "Very good, sir."),
    T(CMD, "play some jazz",
      "Excellent choice, sir. Jazz is a wonderful way to unwind."),
    T(CMD, "pause the music", "Of course, sir."),
    T(CMD, "set a 10 minute timer", "Certainly, sir. Ten minutes it is."),
    T(CMD, "turn the volume up", "Louder it is, sir."),
    T(CMD, "mute", "As you wish, sir."),
    T(CMD, "next song", "A fine pick, sir."),
    T(CMD, "resume the music", "Very well, sir."),
    T(CMD, "show my tasks", "Your tasks, sir."),
    T(CMD, "start a focus session", "Focus is a fine idea, sir."),
    T(CMD, "remind me in 20 minutes to stretch",
      "Stretching is good for you, sir."),
    T(CMD, "JARVIS, could you please pause the music?", "Certainly, sir."),
    T(CMD, "previous track", "Ah, a classic, sir."),
    T(CMD, "unmute", "Welcome back to sound, sir."),
    T(CMD, "pause the music and take a screenshot", "Right, sir."),
    T(CMD, "volume down please", "Quieter it is, sir."),
    T(CMD, "grab a screenshot", "Say cheese, sir."),
    T(CMD, "play the latest album by a jazz trio",
      "An excellent choice for a quiet evening, sir."),
    # a token naming no registered action, autocorrect leaves it unknown
    T(MADE, "set an alarm for seven",
      "[ACTION: set_alarm_clock, 7am] Alarm set, sir.", ["set_alarm_clock"]),
    T(MADE, "order a pizza",
      "[ACTION: order_pizza, large] It is on its way, sir.", ["order_pizza"]),
    T(MADE, "book a table for dinner",
      "[ACTION: reserve_restaurant_table, tonight]",
      ["reserve_restaurant_table"]),
    T(MADE, "water the plants", "[ACTION: water_plants] Watering them now.",
      ["water_plants"]),
    T(MADE, "start the car", "[ACTION: remote_start_vehicle]",
      ["remote_start_vehicle"]),
    T(MADE, "feed the fish", "[ACTION: feed_aquarium]", ["feed_aquarium"]),
    T(MADE, "lock the front door",
      "[ACTION: lock_door_deadbolt] Locked, sir.", ["lock_door_deadbolt"]),
    T(MADE, "translate this into French",
      "[ACTION: translate_text, french]", ["translate_text"]),
    # partial: get_time ran, the second token is made up (low confidence)
    T(MADE, "what's the time, and order more coffee",
      "[ACTION: get_time] [ACTION: order_coffee_beans]",
      ["get_time", "order_coffee_beans"], ["get_time"]),
)

# ── turns that are fine ──────────────────────────────────────────────────────
NEGATIVES = (
    # chit-chat
    T(OK, "how are you today", "Very well, sir, thank you for asking."),
    T(OK, "good morning", "Good morning, sir. I trust you slept well."),
    T(OK, "thanks jarvis", "You're welcome, sir."),
    T(OK, "you're hilarious", "I do try, sir."),
    T(OK, "I'm heading out for a walk", "Enjoy the fresh air, sir."),
    T(OK, "that was a long day", "It certainly sounded like one, sir."),
    T(OK, "good night", "Good night, sir. Sleep well."),
    T(OK, "never mind", "As you wish, sir."),
    T(OK, "stop", "Of course, sir."),
    T(OK, "thanks for setting that timer", "My pleasure, sir."),
    T(OK, "I just turned off the lights myself",
      "Very good, sir. Saves me the trouble."),
    T(OK, "don't play anything yet", "Understood, sir. I'll wait."),
    # jokes
    T(OK, "tell me a joke", "Why don't scientists trust atoms? Because "
      "they make up everything, sir."),
    T(OK, "tell me another one", "I told my computer I needed a break, and "
      "it said it would go to sleep."),
    T(OK, "make me laugh", "A skeleton walks into a bar and orders a drink "
      "and a mop, sir."),
    # answers
    T(OK, "what's the boiling point of water",
      "100 degrees Celsius at sea level, sir."),
    T(OK, "how far away is the moon", "About 384,000 kilometres, sir, and "
      "it is moving away by about 3.8 centimetres a year."),
    T(OK, "what did I just ask you",
      "On it, sir. You asked about the boiling point of water."),
    T(OK, "how many legs does a spider have", "Eight, sir."),
    T(OK, "what's 12 times 7", "84, sir."),
    T(OK, "tell me something interesting",
      "If I may say so, sir... octopuses have three hearts."),
    T(OK, "is the music playing", "It is, sir, the jazz playlist is on."),
    # reports on an earlier turn
    T(OK, "did you send the email", "I sent it, sir."),
    T(OK, "are the lights off", "They are, sir. I turned them off earlier."),
    T(OK, "turn off the lights", "I've already turned them off, sir."),
    # questions back and clarifications
    T(OK, "play some music",
      "Certainly, sir. Any particular artist or genre?"),
    T(OK, "set a timer", "For how long, sir?"),
    T(OK, "turn on the lights", "Which room, sir?"),
    T(OK, "open it", "Open which one, sir, the browser or the file?"),
    T(OK, "remind me later",
      "Of course, sir. When would you like the reminder?"),
    T(OK, "take a screenshot", "Of the whole screen or just this window, sir?"),
    T(OK, "pause the music", "Shall I pause it, sir?"),
    T(OK, "play some jazz", "I'll queue some jazz if you'd like, sir."),
    # confirmation requests
    T(OK, "delete the old downloads", "That one needs your confirmation, "
      "sir. Say 'yes' to proceed, or 'no' to cancel."),
    T(OK, "pause the music", "Say 'yes' and I'll pause it, sir."),
    T(OK, "shut down the computer",
      "Are you sure, sir? That will close everything."),
    T(OK, "restart yourself", "That one needs your confirmation, sir.",
      ["restart_jarvis"]),
    # declines and "I can't"
    T(OK, "turn off the lights", "I'm afraid I can't reach the lights, sir; "
      "no smart devices are connected."),
    T(OK, "order a pizza", "I can't place orders, sir."),
    T(OK, "fly the drone", "That's beyond my abilities at present, sir."),
    T(OK, "open the garage",
      "I don't have a way to control the garage, sir."),
    T(OK, "unlock the front door",
      "I'd rather not, sir. Door locks are off limits."),
    T(OK, "play some jazz",
      "I'm unable to reach the music library right now, sir."),
    # informative actions that ran
    T(OK, "what time is it", "[ACTION: get_time] It is 3:15 PM, sir.",
      ["get_time"], ["get_time"]),
    T(OK, "what's the weather like",
      "[ACTION: weather_briefing] Sunny and 22 degrees, sir.",
      ["weather_briefing"], ["weather_briefing"]),
    T(OK, "what's on my screen",
      "[ACTION: see_screen] You have a spreadsheet open, sir.",
      ["see_screen"], ["see_screen"]),
    T(OK, "how's the system doing", "[ACTION: system_pulse]",
      ["system_pulse"], ["system_pulse"]),
    T(OK, "what timers do I have", "[ACTION: list_timers]",
      ["list_timers"], ["list_timers"]),
    # actions that ran, claims grounded
    T(OK, "set a five minute timer",
      "[ACTION: set_timer, 5 minutes] Timer set, sir.",
      ["set_timer"], ["set_timer"]),
    T(OK, "turn off the lights",
      "[ACTION: smart_home_control, lights off] Done, sir.",
      ["smart_home_control"], ["smart_home_control"]),
    T(OK, "open youtube", "Opening YouTube now, sir. [ACTION: youtube]",
      ["youtube"], ["youtube"]),
    # partial runs
    T(OK, "turn off the lights and play some jazz",
      "Done, sir. Lights off and jazz on the way.",
      ["smart_home_control"], ["smart_home_control"]),
    T(OK, "pause the music and take a screenshot",
      "[ACTION: pause_music] [ACTION: screenshot] Paused and captured, sir.",
      ["pause_music", "screenshot"], ["pause_music"]),
    T(OK, "send the email and set a timer",
      "[ACTION: send_email] [ACTION: set_timer, 5 minutes]",
      ["send_email", "set_timer"], []),
    # autocorrect resolved the name (it ran under its real name, or would)
    T(OK, "set a five minute timer", "[ACTION: set_timr, 5 minutes]",
      ["set_timr"], ["set_timer"]),
    T(OK, "take a screenshot", "[ACTION: take_screenshot]",
      ["take_screenshot"], ["screenshot"]),
    T(OK, "set a timer for five minutes", "[ACTION: set_timers, 5 minutes]",
      ["set_timers"], []),
    # the runtime itself asked "did you mean X or Y?"
    T(OK, "water the plants", "[ACTION: water_plants]", ["water_plants"],
      asked=True),
    # not clear commands on their own
    T(OK, "continue", "Certainly, sir. The third act opens in a storm."),
    T(OK, "next", "The next item on the list is the grocery run, sir."),
    T(OK, "go back", "Very well, sir, to the first point then."),
    T(OK, "play a game with me",
      "Of course, sir! Let's play twenty questions. Think of an animal."),
    T(OK, "play devil's advocate for a second",
      "Very well, sir. One could argue the opposite."),
    T(OK, "can you play music", "I can, sir. The music library is all set up."),
)


class TableTests(unittest.TestCase):
    def test_table_is_big_and_covers_every_kind(self):
        self.assertGreaterEqual(len(POSITIVES) + len(NEGATIVES), 60)
        self.assertGreaterEqual(len(NEGATIVES), 30)
        for kind in (SAID, CMD, MADE):
            with self.subTest(kind=kind):
                self.assertGreaterEqual(
                    sum(r.kind == kind for r in POSITIVES), 8)

    def test_every_negative_is_ok(self):
        for row in NEGATIVES:
            with self.subTest(user=row.user, reply=row.reply):
                v = _check(row)
                self.assertEqual(v.kind, OK, v.reason)
                self.assertEqual(v.confidence, 0.0)

    def test_every_positive_is_its_kind(self):
        for row in POSITIVES:
            with self.subTest(user=row.user, reply=row.reply):
                v = _check(row)
                self.assertEqual(v.kind, row.kind, v.reason)
                self.assertGreater(v.confidence, 0.0)

    def test_per_kind_precision_is_one(self):
        rows = POSITIVES + NEGATIVES
        for kind in (SAID, CMD, MADE):
            flagged = [r for r in rows if _check(r).kind == kind]
            with self.subTest(kind=kind):
                self.assertTrue(flagged)
                right = sum(r.kind == kind for r in flagged)
                self.assertEqual(right / len(flagged), 1.0)

    def test_full_failures_escalate_and_partial_made_up_does_not(self):
        for row in POSITIVES:
            v = _check(row)
            go = tc.should_escalate(v, cloud_allowed=True,
                                    already_escalated=False,
                                    needs_confirmation=False)
            with self.subTest(user=row.user):
                # Only the partial made-up row ran something already.
                self.assertEqual(go, not row.ran)

    def test_no_negative_escalates(self):
        for row in NEGATIVES:
            with self.subTest(user=row.user):
                self.assertFalse(tc.should_escalate(
                    _check(row), cloud_allowed=True,
                    already_escalated=False, needs_confirmation=False))


class GuardTests(unittest.TestCase):
    def test_question_guard_is_load_bearing(self):
        # Mutation: with the "asks the owner something" guard gone, the
        # questions-back rows are flagged again.
        with mock.patch.object(tc._claim_validator, "asks_owner",
                               return_value=False):
            red = [r for r in NEGATIVES if _check(r).kind != OK]
        self.assertTrue(red)
        self.assertIn("take a screenshot", {r.user for r in red})

    def test_explicit_asked_question_overrides_the_reply(self):
        row = T(CMD, "take a screenshot", "Very good, sir.")
        self.assertEqual(_check(row).kind, CMD)
        self.assertEqual(_check(row._replace(asked=True)).kind, OK)
        self.assertEqual(
            _check(row._replace(reply="Very good, sir?", asked=False)).kind,
            CMD)

    def test_empty_registry_skips_registry_checks(self):
        made = POSITIVES[[r.kind for r in POSITIVES].index(MADE)]
        self.assertEqual(_check(made, registered=()).kind, OK)
        cmd = T(CMD, "take a screenshot", "Very good, sir.")
        self.assertEqual(_check(cmd, registered=()).kind, OK)
        # said_no_action needs no registry.
        said = T(SAID, "open youtube", "Opening YouTube now, sir.")
        self.assertEqual(_check(said, registered=()).kind, SAID)

    def test_synthetic_and_unregistered_results_never_count_as_ran(self):
        # parse_and_run_actions' synthetic results and an "unknown action"
        # result are not actions that ran.
        for ran in (["_unverified_claim"], ["_preemptive_hallucinated_claim"],
                    ["teleport_now"]):
            with self.subTest(ran=ran):
                v = tc.check_turn("restart yourself", "Restarting now, sir.",
                                  [], ran, REGISTERED)
                self.assertEqual(v.kind, SAID)

    def test_names_are_case_insensitive(self):
        v = tc.check_turn("what time is it", "[ACTION: Get_Time]",
                          ["Get_Time"], ["GET_TIME"], REGISTERED)
        self.assertEqual(v.kind, OK)

    def test_registry_may_be_the_actions_dict(self):
        actions = {name: (lambda _a="": "ok") for name in REGISTERED}
        v = tc.check_turn("take a screenshot", "Very good, sir.", [], [],
                          actions)
        self.assertEqual(v.kind, CMD)

    def test_autocorrect_fault_is_not_a_made_up_action(self):
        with mock.patch.object(tc._autocorrect, "autocorrect_command_choice",
                               side_effect=RuntimeError("boom")):
            v = tc.check_turn("water the plants", "[ACTION: water_plants]",
                              ["water_plants"], [], REGISTERED)
        self.assertEqual(v.kind, OK)

    def test_autocorrect_is_scored_lexically_never_over_the_network(self):
        with mock.patch.object(tc._autocorrect, "autocorrect_command_choice",
                               wraps=tc._autocorrect.autocorrect_command_choice
                               ) as spy:
            tc.check_turn("water the plants", "[ACTION: water_plants]",
                          ["water_plants"], [], REGISTERED)
        self.assertTrue(spy.called)
        for call in spy.call_args_list:
            self.assertIs(call.kwargs.get("use_embeddings"), False)

    def test_reason_never_carries_the_turn_text(self):
        for row in (T(SAID, "open zebra quartz", "Opening zebra quartz now."),
                    T(CMD, "play zebra quartz", "Very good, sir."),
                    T(MADE, "zebra quartz", "[ACTION: zebra_quartz] Zebra.",
                      ["zebra_quartz"]),
                    T(OK, "zebra quartz", "Quartz, sir?")):
            v = _check(row)
            with self.subTest(kind=row.kind):
                self.assertEqual(v.kind, row.kind)
                self.assertNotIn("zebra", v.reason.lower())
                self.assertNotIn("quartz", v.reason.lower())


class ShouldEscalateTests(unittest.TestCase):
    FAIL = tc.Verdict(SAID, "r", 0.9)

    def _go(self, verdict=FAIL, **kw):
        args = dict(cloud_allowed=True, already_escalated=False,
                    needs_confirmation=False)
        args.update(kw)
        return tc.should_escalate(verdict, **args)

    def test_a_confident_failure_escalates(self):
        self.assertTrue(self._go())

    def test_never_twice_per_turn(self):
        self.assertFalse(self._go(already_escalated=True))

    def test_never_when_cloud_is_disallowed(self):
        self.assertFalse(self._go(cloud_allowed=False))

    def test_never_for_an_action_that_needs_confirmation(self):
        self.assertFalse(self._go(needs_confirmation=True))

    def test_only_at_or_above_the_threshold(self):
        bar = tc.ESCALATE_MIN_CONFIDENCE
        self.assertTrue(self._go(tc.Verdict(CMD, "r", bar)))
        self.assertFalse(self._go(tc.Verdict(CMD, "r", bar - 0.01)))

    def test_ok_never_escalates(self):
        self.assertFalse(self._go(tc.Verdict(OK, "r", 1.0)))
        self.assertFalse(self._go(None))


class ConstantTests(unittest.TestCase):
    def test_one_moment_line_is_short_and_prerendered(self):
        self.assertLessEqual(len(tc.ONE_MOMENT_LINE.split()), 4)
        self.assertIn(tc.ONE_MOMENT_LINE, tts_render_cache.OPENERS)

    def test_kinds(self):
        self.assertEqual(tc.KINDS, ("ok", "said_no_action",
                                    "command_no_action", "made_up_action"))

    def test_autocorrect_defaults_match_the_dispatcher(self):
        # The checker scores with autocorrect_command_choice's defaults; the
        # monolith's dispatcher passes its own. They must agree, or a name
        # the runtime resolves would read as made up.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "bobert_companion.py"),
                  encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        params = inspect.signature(
            command_autocorrect.autocorrect_command_choice).parameters
        for const, param in (("_AUTOCORRECT_THRESHOLD", "threshold"),
                             ("_AUTOCORRECT_AMBIG_GAP", "ambiguity_gap")):
            m = re.search(rf"^{const}\s*=\s*([0-9.]+)", src, re.M)
            with self.subTest(const=const):
                self.assertIsNotNone(m)
                self.assertEqual(float(m.group(1)), params[param].default)


if __name__ == "__main__":
    unittest.main()
