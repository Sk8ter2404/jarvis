"""v3 W01 monolith wiring: spoken-text hygiene.

  * weather_briefing is not injected for a printer / appliance / CPU
    temperature, and a bare "64 degrees" needs a weather cue (replay web-04,
    p-temp-print, p-temp-print2);
  * an unearned "A bold choice" opener is dropped from the final text (action
    and no-action replies) and held back from the early sentence flush;
  * smart_home_discover's result is read back (INFORMATIVE_ACTIONS);
  * SPEECH_HYGIENE off restores the old behaviour.

    python -m unittest tests.monolith.test_monolith_speech_hygiene
"""
from __future__ import annotations

import os
import types
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_advice_fallback import _InlineThread
from tests.monolith.test_monolith_claim_validation import _Base

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

PETG = ("[intent:briefing] Around 235 degrees Celsius for the nozzle, sir, "
        "and 80 for the bed.")
H2D = "[intent:briefing] Up to 350 degrees C, sir."
WEB04 = ("[intent:briefing] Certainly, sir. For PLA, I'd recommend a "
         "temperature between 110 and 140 degrees Fahrenheit to avoid "
         "melting the filament itself.")


@requires_monolith
class WeatherInjectHygieneTests(_Base):
    def det(self, reply, user_text=""):
        self._stub("weather_briefing", "stub")
        return self.bc._detect_preemptive_hallucination(reply,
                                                        user_text=user_text)

    def test_printer_temperatures_do_not_inject(self):
        cases = (
            (PETG, "Jarvis, what temperature should I print PETG at?"),
            (H2D, "Jarvis, how hot does the H2D nozzle get?"),
            (WEB04, "Jarvis, read up on drying temperatures for PLA filament"),
            ("PETG prints at 235 degrees Celsius.", ""),
            ("Set the nozzle to 350 °C.", ""),
            ("Preheat the oven to 400 degrees Fahrenheit.", ""),
            ("The CPU is sitting around 60 degrees Celsius.", ""),
        )
        for reply, user in cases:
            with self.subTest(reply=reply):
                self.assertIsNone(self.det(reply, user))

    def test_a_device_in_the_owner_words_alone_is_enough(self):
        self.assertIsNone(self.det("About 60 degrees Celsius, sir.",
                                   "how hot is my GPU right now"))

    def test_real_weather_still_injects(self):
        for reply in ("It's 64 degrees Fahrenheit and sunny.",
                      "It's currently 64 degrees and sunny.",
                      "Currently 58 degrees Fahrenheit, sir.",
                      "It's 72 and sunny, sir.",
                      "It's 64 degrees Fahrenheit outside, sir."):
            with self.subTest(reply=reply):
                out = self.det(reply, "")
                self.assertEqual(out[:2], ("inject", "weather_briefing"))
                out = self.det(reply, "what's it like outside")
                self.assertEqual(out[:2], ("inject", "weather_briefing"))

    def test_explicit_weather_ask_with_a_device_word_still_injects(self):
        sunny = "It's 72 degrees Fahrenheit and sunny, sir."
        for user in ("what's the weather, is it good to fire up the grill?",
                     "what's the weather before I go to bed",
                     "what's the temperature outside, will my print be ok?",
                     "what's the forecast, should I print outside"):
            with self.subTest(user=user):
                out = self.det(sunny, user)
                self.assertEqual(out[:2], ("inject", "weather_briefing"))
        # a bare "bed" is not a device; a reply with a weather cue keeps it
        out = self.det("It's 72 degrees Fahrenheit, sir.",
                       "how warm is it before bed")
        self.assertEqual(out[:2], ("inject", "weather_briefing"))
        out = self.det("72 degrees and sunny, good for the grill, sir.", "")
        self.assertEqual(out[:2], ("inject", "weather_briefing"))

    def test_print_bed_is_still_a_device(self):
        self.assertIsNone(self.det("About 60 degrees Celsius, sir.",
                                   "how hot is the heated bed"))
        self.assertIsNone(self.det("About 60 degrees Celsius, sir.",
                                   "how hot is the print bed"))

    def test_bare_unit_needs_a_cue_or_a_weather_ask(self):
        bare = "That comes to 37.8 degrees Celsius, sir."
        # owner words unknown: the old behaviour stands
        self.assertEqual(self.det(bare, "")[:2], ("inject", "weather_briefing"))
        # a turn that was not about the weather or a temperature
        self.assertIsNone(self.det(bare, "how much does the soup weigh"))
        # a temperature question keeps the injection
        self.assertEqual(
            self.det(bare, "what's the temperature")[:2],
            ("inject", "weather_briefing"))
        # a cue in the reply keeps it
        self.assertEqual(
            self.det("About 64 degrees Fahrenheit and cloudy, sir.",
                     "how much does the soup weigh")[:2],
            ("inject", "weather_briefing"))

    def test_other_claim_categories_are_untouched(self):
        self._stub("get_time", "stub")
        out = self.bc._detect_preemptive_hallucination(
            "It's 1:47 AM, sir.",
            user_text="what temperature should I print PETG at")
        self.assertEqual(out[:2], ("inject", "get_time"))

    def test_setting_off_is_the_old_behaviour(self):
        self._stub("weather_briefing", "stub")
        with mock.patch.object(self.bc, "SPEECH_HYGIENE", False):
            for reply, user in (
                    (PETG, "what temperature should I print PETG at"),
                    (H2D, "how hot does the H2D nozzle get")):
                out = self.bc._detect_preemptive_hallucination(
                    reply, user_text=user)
                self.assertEqual(out[:2], ("inject", "weather_briefing"))

    def test_dispatch_does_not_run_weather_for_a_print_temperature(self):
        self._stub("weather_briefing", "Currently 64 degrees and clear, sir.")
        self._dispatch("Jarvis, what temperature should I print PETG at?", PETG)
        self.assertEqual(self.calls["weather_briefing"], [])
        self.assertEqual(self.spoken, [PETG])


BOLD = "[intent:dry_wit] A bold choice, if I may say so, sir. {}"


@requires_monolith
class BoldChoiceDispatchTests(_Base):
    def test_unearned_opener_is_dropped_from_a_plain_reply(self):
        self._dispatch("Jarvis, honestly, does pineapple belong on pizza?",
                       BOLD.format("Most Italians would object."))
        self.assertEqual(self.spoken,
                         ["[intent:dry_wit] Most Italians would object."])

    def test_a_yes_does_not_earn_it(self):
        self._dispatch("yes", BOLD.format("I shall prepare the curtains."))
        self.assertEqual(self.spoken,
                         ["[intent:dry_wit] I shall prepare the curtains."])

    def test_a_stated_choice_keeps_it(self):
        reply = BOLD.format("Proceed with care.")
        self._dispatch("I'm going to delete the migrations", reply)
        self.assertEqual(self.spoken, [reply])

    def test_a_bare_opener_is_never_turned_into_silence(self):
        reply = "[intent:dry_wit] A bold choice, if I may say so, sir."
        self._dispatch("yes", reply)
        self.assertEqual(self.spoken, [reply])

    def test_action_reply_loses_the_opener_too(self):
        self._stub("open_on_monitor", "ok")
        self._dispatch("turn on a video on YouTube on the left monitor",
                       "[intent:amused] A bold choice, sir. "
                       "[ACTION: open_on_monitor, left | youtube]")
        self.assertEqual(self.calls["open_on_monitor"], ["left | youtube"])
        self.assertEqual(self.spoken, [])

    def test_action_reply_with_prose_after_the_opener(self):
        self._stub("open_on_monitor", "ok")
        self._dispatch("turn on a video on the left monitor",
                       "[intent:amused] A bold choice, sir. Opening it now. "
                       "[ACTION: open_on_monitor, left | youtube]")
        self.assertEqual(len(self.spoken), 1)
        self.assertNotIn("bold choice", self.spoken[0].lower())
        self.assertIn("Opening it now", self.spoken[0])

    def test_setting_off_is_the_old_behaviour(self):
        reply = BOLD.format("Most Italians would object.")
        with mock.patch.object(self.bc, "SPEECH_HYGIENE", False):
            self._dispatch("does pineapple belong on pizza?", reply)
        self.assertEqual(self.spoken, [reply])


@requires_monolith
class BoldChoiceFlushHoldTests(_Base):
    def _feed(self, user_text, chunks):
        bc = self.bc
        spoken: list[str] = []
        prev = bc._begin_turn_grounding(user_text)
        try:
            with mock.patch.object(bc, "threading",
                                   types.SimpleNamespace(Thread=_InlineThread)):
                buf = bc._SentenceFlushBuffer(speak_fn=spoken.append)
                for c in chunks:
                    buf.feed(c)
        finally:
            bc._end_turn_grounding(prev)
        return spoken

    def test_an_unearned_opener_is_not_voiced_early(self):
        self.assertEqual(
            self._feed("does pineapple belong on pizza",
                       ["A bold choice, sir. ", "Most would object. ", "tail"]),
            [])

    def test_an_earned_opener_still_streams(self):
        spoken = self._feed("I'm going to paint the garage orange",
                            ["A bold choice, sir. ", "I approve. ", "tail"])
        self.assertEqual(spoken[0], "A bold choice, sir.")

    def test_setting_off_streams_it(self):
        with mock.patch.object(self.bc, "SPEECH_HYGIENE", False):
            spoken = self._feed("does pineapple belong on pizza",
                                ["A bold choice, sir. ", "Hm. ", "tail"])
        self.assertEqual(spoken[0], "A bold choice, sir.")


if __name__ == "__main__":
    unittest.main()
