"""weather_briefing must answer for the day that was ASKED ABOUT.

Live, v2.0.115 (2026-09-29, typed turn):
    "what's the weather going to be like tomorrow"
      -> [ACTION: weather_briefing]
      -> "Currently 87 degrees and clear, sir."
Those were current conditions. Tomorrow's forecast was never given — the
handler ignored its argument and only knew how to describe NOW.

These tests pin the fix:
  * the day is read from the action argument, and — when the model passed none
    — from the owner's own words for THIS turn (core.owner_turn), so a bare
    token on a "tomorrow" question still answers for tomorrow;
  * tomorrow / tonight / this weekend / a weekday get that day's high, low,
    conditions and precipitation chance from Open-Meteo's DAILY endpoint;
  * a failed or out-of-range forecast says so, and never falls back to today's
    conditions (the answer to a different question);
  * an old utterance outside a turn is never reused (a scheduled weather call
    must not inherit last night's "tomorrow").

No network: urllib.request.urlopen is replaced with a canned-JSON opener and
the location resolver is mocked. The clock is frozen (CI runs in UTC). The
real monolith is never imported — a fake ``bobert_companion`` module stands in.
"""
from __future__ import annotations

import json
import types
import unittest
from datetime import date, datetime, timedelta
from unittest import mock

from tests._skill_harness import load_skill_isolated
from tests.skills.test_weather_briefing import (
    _fake_urlopen_returning,
    _frozen_now,
    _hour,
    inject_modules,
)

# Tuesday 2026-09-29, 14:00 local — frozen for every test here.
_NOW = datetime(2026, 9, 29, 14, 0, 0)
_TODAY = _NOW.date()


def _fake_companion(utterance=None, in_turn=True):
    """A stand-in monolith carrying only the two cells core.owner_turn reads.
    The weather skill's _config() also resolves against it (attrs absent ->
    defaults), so the real ~30K-line monolith is never imported."""
    bc = types.ModuleType("bobert_companion")
    bc._last_user_text = [utterance]
    bc._turn_in_progress = [bool(in_turn)]
    return bc


def _daily_payload(start: date = _TODAY, n: int = 8):
    """Open-Meteo daily JSON: n days from `start`, Celsius, distinct per day so
    a test can tell WHICH day was read. Day i: max 20+i C, min 10+i C,
    precip 10*i %, weather code cycling clear / partly cloudy / rain."""
    codes = [0, 2, 61]
    days = [start + timedelta(days=i) for i in range(n)]
    return json.dumps({"daily": {
        "time": [d.isoformat() for d in days],
        "temperature_2m_max": [20.0 + i for i in range(n)],
        "temperature_2m_min": [10.0 + i for i in range(n)],
        "precipitation_probability_max": [10 * i for i in range(n)],
        "weather_code": [codes[i % 3] for i in range(n)],
    }})


def _f(c):
    return int(round(c * 9 / 5 + 32))


class _Base(unittest.TestCase):
    """Loads the skill with a fake monolith and keeps that fake installed for
    the whole test (the skill resolves it lazily at call time)."""

    def _load(self, utterance=None, in_turn=True):
        cm = inject_modules(bobert_companion=_fake_companion(utterance, in_turn))
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        mod, actions = load_skill_isolated("weather_briefing")
        return mod, actions

    def _run(self, mod, actions, arg="", payload=None, opener=None):
        """Call the action with the network mocked and the clock frozen."""
        opener = opener or _fake_urlopen_returning(
            payload if payload is not None else _daily_payload())
        with _frozen_now(mod, _NOW), \
                mock.patch.object(mod, "_resolve_location",
                                  return_value=(40.0, -75.0)), \
                mock.patch.object(mod.urllib.request, "urlopen", opener), \
                mock.patch.object(mod, "_current_conditions_line",
                                  return_value="Currently 87 degrees and clear, sir.") as cur, \
                mock.patch.object(mod, "get_umbrella_alert", return_value=""), \
                mock.patch.object(mod, "get_two_hour_alert", return_value=""):
            out = actions["weather_briefing"](arg)
        return out, cur


# ─────────────────────────────────────────────────────────────────────────
#  The live defect, end to end
# ─────────────────────────────────────────────────────────────────────────
class TomorrowQuestionTests(_Base):
    def test_bare_token_on_a_tomorrow_question_answers_for_tomorrow(self):
        """The exact 2026-09-29 turn: no argument, the day only in his words."""
        mod, actions = self._load(
            "what's the weather going to be like tomorrow")
        out, cur = self._run(mod, actions, arg="")
        # Tomorrow = day index 1: max 21 C, min 11 C, 10 %, partly cloudy.
        self.assertEqual(
            out,
            f"Tomorrow, sir: partly cloudy, a high of {_f(21)} and a low of "
            f"{_f(11)}, with a 10% chance of rain.")
        self.assertNotIn("Currently", out,
                         "a tomorrow question must not be answered with NOW")
        cur.assert_not_called()

    def test_argument_names_the_day(self):
        mod, actions = self._load(utterance=None)
        out, cur = self._run(mod, actions, arg="tomorrow")
        self.assertTrue(out.startswith("Tomorrow, sir:"), out)
        self.assertIn(f"a high of {_f(21)}", out)
        cur.assert_not_called()

    def test_weather_forecast_alias_behaves_the_same(self):
        mod, actions = self._load("will it rain tomorrow")
        with _frozen_now(mod, _NOW), \
                mock.patch.object(mod, "_resolve_location",
                                  return_value=(1.0, 2.0)), \
                mock.patch.object(mod.urllib.request, "urlopen",
                                  _fake_urlopen_returning(_daily_payload())):
            out = actions["weather_forecast"]("")
        self.assertTrue(out.startswith("Tomorrow, sir:"), out)

    def test_argument_beats_the_utterance(self):
        """A two-day question split into two tokens must get both answers, so
        an explicit day in the argument wins over the owner's words."""
        mod, actions = self._load("what's the weather today and tomorrow")
        out, cur = self._run(mod, actions, arg="today")
        self.assertEqual(out, "Currently 87 degrees and clear, sir.")
        cur.assert_called_once()

    def test_no_day_anywhere_keeps_current_conditions(self):
        mod, actions = self._load("what's the weather like")
        out, cur = self._run(mod, actions, arg="")
        self.assertEqual(out, "Currently 87 degrees and clear, sir.")

    def test_old_utterance_outside_a_turn_is_not_reused(self):
        """_last_user_text is never cleared. A weather call made OUTSIDE an
        owner turn (a schedule, a proactive job) must not inherit it."""
        mod, actions = self._load("what's the weather tomorrow", in_turn=False)
        out, cur = self._run(mod, actions, arg="")
        self.assertEqual(out, "Currently 87 degrees and clear, sir.")

    def test_failed_forecast_is_honest_and_never_falls_back_to_now(self):
        mod, actions = self._load("what's the weather tomorrow")
        out, cur = self._run(
            mod, actions,
            opener=_fake_urlopen_returning("", raises=OSError("offline")))
        self.assertEqual(out, "I couldn't reach the forecast for tomorrow, sir.")
        cur.assert_not_called()

    def test_day_missing_from_the_forecast_is_honest(self):
        # The provider answered, but not for the day asked about.
        mod, actions = self._load("what about friday")
        out, _cur = self._run(mod, actions,
                              payload=_daily_payload(_TODAY, n=2))
        self.assertEqual(out, "I couldn't reach the forecast for Friday, sir.")


class OtherDaysTests(_Base):
    def test_this_weekend_gives_saturday_and_sunday(self):
        mod, actions = self._load("what's the weather this weekend")
        out, _cur = self._run(mod, actions)
        # Tuesday + 4 = Saturday (index 4: 24/14 C, 40 %, partly cloudy... code
        # cycle index 4 -> 2 = partly cloudy), Sunday index 5 (code 61, rain).
        self.assertTrue(out.startswith("This weekend, sir — Saturday, "), out)
        self.assertIn(f"Saturday, partly cloudy, a high of {_f(24)} and a low "
                      f"of {_f(14)}, with a 40% chance of rain", out)
        self.assertIn(f"Sunday, light rain, a high of {_f(25)} and a low of "
                      f"{_f(15)}, with a 50% chance of rain", out)

    def test_named_weekday(self):
        mod, actions = self._load("will I need a jacket on Friday")
        out, _cur = self._run(mod, actions)
        # Tuesday -> Friday = +3 (index 3: 23/13 C, 30 %, code 0 clear).
        self.assertEqual(
            out, f"Friday, sir: clear, a high of {_f(23)} and a low of "
                 f"{_f(13)}, with a 30% chance of rain.")

    def test_tonight_uses_the_hourly_outlook(self):
        mod, actions = self._load("is it going to rain tonight")
        evening = datetime(2026, 9, 29, 18, 0, 0)
        hours = []
        for i, (temp, prob, cat, desc) in enumerate([
                (20.0, 10, "cloudy", "partly cloudy"),
                (17.0, 60, "rain", "light rain"),
                (15.0, 30, "cloudy", "partly cloudy")]):
            h = _hour(evening + timedelta(hours=i), prob=prob, temp=temp)
            h["category"], h["desc"] = cat, desc
            hours.append(h)
        # An afternoon hour (before 18:00) must not count toward tonight.
        early = _hour(datetime(2026, 9, 29, 15, 0, 0), prob=99, temp=30.0)
        early["category"], early["desc"] = "thunderstorm", "thunderstorms"
        with _frozen_now(mod, _NOW), \
                mock.patch.object(mod, "_fetch_hourly_forecast",
                                  return_value=[early] + hours):
            out = actions["weather_briefing"]("")
        self.assertEqual(
            out, f"Tonight, sir: light rain, down to a low of {_f(15)}, "
                 f"with a 60% chance of rain.")

    def test_tonight_without_data_is_honest(self):
        mod, actions = self._load("what's it like tonight")
        with _frozen_now(mod, _NOW), \
                mock.patch.object(mod, "_fetch_hourly_forecast", return_value=[]):
            out = actions["weather_briefing"]("tonight")
        self.assertEqual(out, "I couldn't reach tonight's forecast, sir.")


# ─────────────────────────────────────────────────────────────────────────
#  The pieces
# ─────────────────────────────────────────────────────────────────────────
class ResolveWhenTests(unittest.TestCase):
    def setUp(self):
        with inject_modules(bobert_companion=_fake_companion()):
            self.mod, _ = load_skill_isolated("weather_briefing")
        self.r = lambda t: self.mod._resolve_when(t, _TODAY)   # a Tuesday

    def test_no_day_reference(self):
        self.assertIsNone(self.r(""))
        self.assertIsNone(self.r("what's the weather like"))
        self.assertIsNone(self.r("weather"))

    def test_today_now(self):
        for t in ("today", "right now", "what's it like at the moment"):
            self.assertEqual(self.r(t), ("now", None), t)

    def test_tomorrow_beats_tonight(self):
        self.assertEqual(self.r("tomorrow night"),
                         ("days", ("tomorrow", [_TODAY + timedelta(days=1)])))

    def test_day_after_tomorrow(self):
        self.assertEqual(self.r("the day after tomorrow"),
                         ("days", ("Thursday", [_TODAY + timedelta(days=2)])))

    def test_tonight(self):
        for t in ("tonight", "this evening", "overnight"):
            self.assertEqual(self.r(t), ("tonight", None), t)

    def test_weekend_from_a_weekday_saturday_and_sunday(self):
        sat = date(2026, 10, 3)
        self.assertEqual(self.r("this weekend"),
                         ("days", ("this weekend", [sat, sat + timedelta(days=1)])))

    def test_weekend_on_saturday_and_sunday(self):
        sat, sun = date(2026, 10, 3), date(2026, 10, 4)
        self.assertEqual(self.mod._resolve_when("weekend", sat),
                         ("days", ("this weekend", [sat, sun])))
        self.assertEqual(self.mod._resolve_when("weekend", sun),
                         ("days", ("this weekend", [sun])))

    def test_weekdays(self):
        self.assertEqual(self.r("friday"),
                         ("days", ("Friday", [date(2026, 10, 2)])))
        self.assertEqual(self.r("monday"),
                         ("days", ("Monday", [date(2026, 10, 5)])))
        # Today's own name is today, unless it says 'next'.
        self.assertEqual(self.r("tuesday"), ("now", None))
        self.assertEqual(self.r("next tuesday"),
                         ("days", ("Tuesday", [date(2026, 10, 6)])))


class DailyFetchTests(unittest.TestCase):
    def setUp(self):
        with inject_modules(bobert_companion=_fake_companion()):
            self.mod, _ = load_skill_isolated("weather_briefing")

    def test_requests_the_daily_endpoint_for_a_week_ahead(self):
        seen = {}
        inner = _fake_urlopen_returning(_daily_payload())

        def opener(req, timeout=None):
            seen["url"] = req.full_url
            return inner(req, timeout)

        with mock.patch.object(self.mod, "_resolve_location",
                               return_value=(40.0, -75.0)), \
                mock.patch.object(self.mod.urllib.request, "urlopen", opener):
            out = self.mod._fetch_daily_forecast()
        self.assertIn("api.open-meteo.com", seen["url"])
        self.assertIn("daily=", seen["url"])
        self.assertIn("forecast_days=8", seen["url"])
        self.assertEqual(len(out), 8)
        day = out[_TODAY + timedelta(days=2)]
        self.assertEqual((day["hi_c"], day["lo_c"], day["precip_prob"]),
                         (22.0, 12.0, 20))
        self.assertEqual((day["desc"], day["category"]), ("light rain", "rain"))

    def test_missing_probability_stays_unknown_not_zero(self):
        payload = json.dumps({"daily": {
            "time": ["2026-09-30"], "temperature_2m_max": [21.0],
            "temperature_2m_min": [11.0],
            "precipitation_probability_max": [None], "weather_code": [3]}})
        with mock.patch.object(self.mod, "_resolve_location",
                               return_value=(1.0, 2.0)), \
                mock.patch.object(self.mod.urllib.request, "urlopen",
                                  _fake_urlopen_returning(payload)):
            out = self.mod._fetch_daily_forecast()
        day = out[date(2026, 9, 30)]
        self.assertIsNone(day["precip_prob"])
        self.assertEqual(self.mod._day_phrase(day),
                         f"overcast, a high of {_f(21)} and a low of {_f(11)}")

    def test_no_location_or_garbage_is_empty(self):
        with mock.patch.object(self.mod, "_resolve_location", return_value=None):
            self.assertEqual(self.mod._fetch_daily_forecast(), {})
        for payload in ("[]", json.dumps({"daily": "nope"}),
                        json.dumps({"daily": {"time": ["not-a-date"]}})):
            with mock.patch.object(self.mod, "_resolve_location",
                                   return_value=(1.0, 2.0)), \
                    mock.patch.object(self.mod.urllib.request, "urlopen",
                                      _fake_urlopen_returning(payload)):
                self.assertEqual(self.mod._fetch_daily_forecast(), {}, payload)


class PromptRoutingTests(unittest.TestCase):
    """The model has to be SHOWN that the day goes in the argument, on the
    default local (slimmed) path."""

    def test_future_day_example_reaches_the_local_model(self):
        from core import prompt_router, prompts
        self.assertIn("'will it rain tomorrow' → [ACTION: weather_briefing, "
                      "tomorrow]", prompts.PC_CONTROL_PROMPT)
        for utt in ("will it rain tomorrow",
                    "what's the weather going to be like tomorrow",
                    "what's the forecast for the weekend"):
            slim = prompt_router.slim_pc_control(utt, prompts.PC_CONTROL_PROMPT)
            self.assertIn("[ACTION: weather_briefing, tomorrow]", slim, utt)


if __name__ == "__main__":
    unittest.main()
