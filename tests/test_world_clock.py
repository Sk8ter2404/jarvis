"""core/world_clock.py: "what time is it in <place>" and the reply guard
(v2.0.148).

Live evidence this pins (v2.0.140, Tue 2026-09-29 22:17 US Central):
"what time is it in London" ran get_time (the LOCAL clock) and the LLM said
"It is 10:17 PM in London, sir." London was at 4:17 AM on Wednesday.

Every test passes an explicit, zone-aware ``now``: nothing depends on the
host's time zone (CI runs in UTC). Generic places only. Stdlib unittest,
CI-safe.

    python -m unittest tests.test_world_clock
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

from core import world_clock as wc

try:
    from zoneinfo import ZoneInfo
    CENTRAL = ZoneInfo("America/Chicago")
    ZoneInfo("Europe/London")
    _HAVE_TZ = True
except Exception:   # pragma: no cover - no zone data on this host
    CENTRAL = None
    _HAVE_TZ = False

_needs_tz = unittest.skipUnless(_HAVE_TZ, "no IANA time zone data")


def _central(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=CENTRAL)


# The live-evidence moment.
NOW = _central(2026, 9, 29, 22, 17) if _HAVE_TZ else None

# (question, exact spoken reply) at NOW.
_TABLE = (
    # the live bug
    ("what time is it in London",
     "It's 4:17 AM in London, sir. That's Wednesday there."),
    # every frame
    ("what's the time in London",
     "It's 4:17 AM in London, sir. That's Wednesday there."),
    ("time in London",
     "It's 4:17 AM in London, sir. That's Wednesday there."),
    ("What time is it in London right now?",
     "It's 4:17 AM in London, sir. That's Wednesday there."),
    ("Jarvis, what's the current time in Tokyo?",
     "It's 12:17 PM in Tokyo, sir. That's Wednesday there."),
    ("can you tell me the time in Paris",
     "It's 5:17 AM in Paris, sir. That's Wednesday there."),
    ("do you know what time it is in New York",
     "It's 11:17 PM in New York, sir."),
    ("local time in Sydney",
     "It's 1:17 PM in Sydney, sir. That's Wednesday there."),
    ("what time is it over in Honolulu",
     "It's 5:17 PM in Honolulu, sir."),
    ("what time would it be in Dubai",
     "It's 7:17 AM in Dubai, sir. That's Wednesday there."),
    ("what time is it London time",
     "It's 4:17 AM in London, sir. That's Wednesday there."),
    # countries, "city, region", accents, abbreviations
    ("what time is it in the UK",
     "It's 4:17 AM in the UK, sir. That's Wednesday there."),
    ("what time is it in India",
     "It's 8:47 AM in India, sir. That's Wednesday there."),
    ("what time is it in Sydney, Australia",
     "It's 1:17 PM in Sydney, sir. That's Wednesday there."),
    ("what time is it in Paris, France",
     "It's 5:17 AM in Paris, sir. That's Wednesday there."),
    ("what time is it in São Paulo",
     "It's 12:17 AM in Sao Paulo, sir. That's Wednesday there."),
    ("what time is it in LA", "It's 8:17 PM in Los Angeles, sir."),
    ("what time is it in DC", "It's 11:17 PM in Washington, D.C., sir."),
    # US states and the zones
    ("what time is it in Arizona", "It's 8:17 PM in Arizona, sir."),
    ("what time is it in Alaska", "It's 7:17 PM in Alaska, sir."),
    ("what time is it in Texas", "It's 10:17 PM in Texas, sir."),
    ("what time is it in Chicago", "It's 10:17 PM in Chicago, sir."),
    ("what time is it Eastern", "It's 11:17 PM Eastern time, sir."),
    ("what time is it in Pacific time", "It's 8:17 PM Pacific time, sir."),
    ("what's the time in Mountain time", "It's 9:17 PM Mountain time, sir."),
    ("what time is it in UTC",
     "It's 3:17 AM UTC, sir. That's Wednesday on UTC."),
    ("what time is it in GMT",
     "It's 3:17 AM GMT, sir. That's Wednesday on GMT."),
    # a 45-minute offset
    ("what time is it in Kathmandu",
     "It's 9:02 AM in Kathmandu, sir. That's Wednesday there."),
)

# Unknown / ambiguous places, other questions and commands: None, so the
# normal turn handles them.
_NEGATIVE = (
    "what time is it", "what time is it now", "what's the time",
    "what time is it there", "what time is it in Narnia",
    "what time is it in Australia",       # several zones
    "what time is it in the US", "what time is it in Canada",
    "what time is it in Tennessee",       # split near evenly
    "what time is it in Georgia",         # the state or the country?
    "what time is it in Washington",      # the state or D.C.?
    "what time is it in Paris, Texas",    # the city is not in the map
    "what time is it in London in 3 hours",
    "what time is it in London and Paris",
    "what's the time zone in London",
    "what time does the store close in London",
    "what's the weather in London",
    "set an alarm for 7 in London",
    "remind me at 5 pm to call London",
    "what day is it in Tokyo",
    "", "   ",
)


@_needs_tz
class WorldClockAnswerTests(unittest.TestCase):
    def test_table(self):
        for text, reply in _TABLE:
            with self.subTest(text=text):
                got = wc.answer(text, NOW)
                self.assertIsNotNone(got)
                self.assertEqual(got.kind, "world-clock")
                self.assertEqual(got.reply, reply)

    def test_live_bug_is_not_the_local_time(self):
        got = wc.answer("what time is it in London", NOW)
        self.assertNotIn("10:17 PM", got.reply)

    def test_negative(self):
        for text in _NEGATIVE:
            with self.subTest(text=text):
                self.assertIsNone(wc.answer(text, NOW))

    def test_bad_inputs(self):
        for text, now in ((None, NOW), (42, NOW),
                          ("what time is it in London", None),
                          ("what time is it in London", "22:17"),
                          ("what time is it in London " + "x " * 80, NOW)):
            with self.subTest(text=text, now=now):
                self.assertIsNone(wc.answer(text, now))

    def test_earlier_date_there_says_still(self):
        tokyo_morning = dt.datetime(2026, 9, 30, 8, 0,
                                    tzinfo=ZoneInfo("Asia/Tokyo"))
        self.assertEqual(
            wc.answer("what time is it in Los Angeles", tokyo_morning).reply,
            "It's 4:00 PM in Los Angeles, sir. That's still Tuesday there.")

    def test_daylight_saving_is_applied(self):
        # Phoenix keeps MST all year; Central shifts. Noon Central is 11 AM
        # in Phoenix in January and 10 AM in July.
        self.assertEqual(
            wc.answer("time in Phoenix", _central(2026, 1, 15, 12, 0)).reply,
            "It's 11:00 AM in Phoenix, sir.")
        self.assertEqual(
            wc.answer("time in Phoenix", _central(2026, 7, 15, 12, 0)).reply,
            "It's 10:00 AM in Phoenix, sir.")

    def test_a_naive_now_is_this_machines_local_time(self):
        naive = dt.datetime(2026, 9, 29, 22, 17)
        self.assertEqual(wc.answer("time in London", naive),
                         wc.answer("time in London", naive.astimezone()))

    def test_every_mapped_zone_exists(self):
        for key, place in wc._INDEX.items():
            with self.subTest(key=key):
                self.assertIsNotNone(wc._zoneinfo(place.zone), place.zone)
                self.assertTrue(wc.time_at(place, NOW) is not None)


class WorldClockWithoutZoneDataTests(unittest.TestCase):
    """Windows without the tzdata package, or a broken zone file: None,
    never a raise, never the local time."""

    def _now(self):
        return dt.datetime(2026, 9, 29, 22, 17,
                           tzinfo=dt.timezone(dt.timedelta(hours=-5)))

    def test_no_zoneinfo_module(self):
        with mock.patch.object(wc, "_ZoneInfo", None):
            self.assertIsNone(wc.answer("what time is it in London",
                                        self._now()))
            self.assertIsNone(wc.check_time_claim(
                "It is 10:17 PM in London, sir.", self._now()))

    def test_zone_lookup_raises(self):
        def boom(_name):
            raise KeyError("no time zone found")
        with mock.patch.object(wc, "_ZoneInfo", boom):
            self.assertIsNone(wc.answer("what time is it in London",
                                        self._now()))
            self.assertIsNone(wc.check_time_claim(
                "It is 10:17 PM in London, sir.", self._now()))


class ResolvePlaceTests(unittest.TestCase):
    def test_names(self):
        for name, label in (("London", "London"), ("the UK", "the UK"),
                            ("São Paulo", "Sao Paulo"),
                            ("new york city", "New York"),
                            ("Paris, France", "Paris"),
                            ("Sydney Australia", "Sydney"),
                            ("Chicago, Illinois", "Chicago"),
                            ("St. Louis", "St. Louis"),
                            ("pacific time", "Pacific time")):
            with self.subTest(name=name):
                self.assertEqual(wc.resolve_place(name).label, label)

    def test_unknown_or_ambiguous(self):
        for name in ("Narnia", "Australia", "the US", "Georgia",
                     "Washington", "Paris Texas", "Tennessee", "", None, 42):
            with self.subTest(name=name):
                self.assertIsNone(wc.resolve_place(name))


@_needs_tz
class TimeClaimGuardTests(unittest.TestCase):
    """The reply guard: a clock time stated for a known place is checked."""

    LIVE = "It is 10:17 PM in London, sir."
    TRUE_LONDON = "It's 4:17 AM in London, sir. That's Wednesday there."

    def test_the_live_reply_is_corrected(self):
        got = wc.check_time_claim(self.LIVE, NOW)
        self.assertTrue(got.corrected)
        self.assertEqual(got.reply, self.TRUE_LONDON)
        self.assertEqual(got.places, ("London",))

    def test_every_claim_shape_is_checked(self):
        for reply in ("[intent:confirmation] It is 10:17 PM in London, sir.",
                      "In London, it's 10:17 PM, sir.",
                      "The current time in London is 10:17 PM, sir.",
                      "It's 10:17 PM London time.",
                      "It is 10:17 PM on Tuesday, September 29, 2026 in "
                      "London, sir.",
                      "It's currently 10:17 p.m. in London.",
                      "It's 22:17 in London.",
                      "It's 10:17 in London."):
            with self.subTest(reply=reply):
                got = wc.check_time_claim(reply, NOW)
                self.assertTrue(got.corrected)
                self.assertTrue(got.reply.endswith(self.TRUE_LONDON))
                self.assertTrue(wc.has_time_claim(reply))

    def test_leading_tags_are_kept_but_never_an_action_token(self):
        got = wc.check_time_claim(
            "[intent:confirmation] It is 10:17 PM in London, sir.", NOW)
        self.assertEqual(got.reply,
                         "[intent:confirmation] " + self.TRUE_LONDON)
        got = wc.check_time_claim(
            "[ACTION: get_time] It is 10:17 PM in London, sir.", NOW)
        self.assertEqual(got.reply, self.TRUE_LONDON)

    def test_a_right_time_is_left_alone(self):
        for reply in ("It's 4:17 AM in London, sir.",
                      "It's 4:19 a.m. in London.",        # within 3 minutes
                      "It's about 4 AM in London.",       # no minutes: 45
                      "It's 4:17 in London.",             # no AM/PM
                      "It's 10:17 PM here in Chicago, sir.",
                      "It's 11:17 PM Eastern time, sir.",
                      "It's 10:17 PM here, while in London it's 4:17 AM."):
            with self.subTest(reply=reply):
                got = wc.check_time_claim(reply, NOW)
                self.assertIsNotNone(got)
                self.assertFalse(got.corrected)
                self.assertEqual(got.reply, reply)

    def test_two_places_one_wrong(self):
        got = wc.check_time_claim(
            "In London, it's 4:17 AM, and in Tokyo, it's 10:17 PM.", NOW)
        self.assertTrue(got.corrected)
        self.assertEqual(
            got.reply,
            "It's 4:17 AM in London, sir. That's Wednesday there. It's "
            "12:17 PM in Tokyo, sir. That's Wednesday there.")

    def test_not_a_claim_about_a_known_place(self):
        for reply in ("It is 2:53 PM on Tuesday, September 29, 2026, sir.",
                      "It's 10:17 PM, and the flight lands in London at "
                      "6 AM.",
                      "It's 30 degrees in London.",
                      "It's 10:17 PM in Narnia.",
                      "It's 10 PM in the evening, sir.",
                      "London is lovely at this time of year.",
                      "", None, 42):
            with self.subTest(reply=reply):
                self.assertIsNone(wc.check_time_claim(reply, NOW))
                self.assertFalse(wc.has_time_claim(reply))

    def test_never_raises(self):
        self.assertIsNone(wc.check_time_claim(self.LIVE, None))
        self.assertIsNone(wc.check_time_claim(self.LIVE, "now"))


@_needs_tz
class GuardReviewRegressionTests(unittest.TestCase):
    """The v2.0.148 review's reply-guard findings. Each one had the guard
    REPLACE a correct (or not-about-now) LLM reply with an unrelated "It's
    4:17 AM in London" line. The guard now only rewrites a statement of the
    time right now at a place it knows for certain; otherwise it leaves the
    reply exactly as written."""

    def _untouched(self, reply, question=None):
        got = (wc.check_time_claim(reply, NOW) if question is None
               else wc.check_time_claim(reply, NOW, question=question))
        self.assertTrue(got is None or not got.corrected, got)
        if got is not None:
            self.assertEqual(got.reply, reply)
        return got

    def test_wc1_conversions_plans_and_event_times_are_not_now(self):
        for reply in (
                "When it's 9 AM here, it's 3 PM in London, sir.",
                "If you call at 8 PM, it'll be 2 AM in London, sir.",
                "When you land in London, it'll be 6 AM, sir.",
                "The local time in Tokyo will be 3 PM when you land, sir.",
                "It is 3 PM in London on Saturday when the match kicks off, "
                "sir.",
                "Your call with the Boston office? It's 3 PM Eastern time, "
                "sir.",
                "It's 9 AM Pacific time when the stream starts, sir.",
                # no conditional word, but another day
                "It is 3 PM in London on Saturday, sir.",
                "I'll remind you when it's 9 AM in London, sir."):
            with self.subTest(reply=reply):
                self.assertIsNone(self._untouched(reply))
                self.assertFalse(wc.has_time_claim(reply))

    def test_wc1_a_conversion_question_is_never_checked(self):
        # A present-tense line in answer to a conversion question is the
        # conversion, not the time now.
        for question in ("if it's 9 AM here what time is it in London",
                         "what time is 3 pm eastern in london",
                         "remind me when it's 9 am in london",
                         "what time is my call in London tomorrow",
                         "what time will it be in London in 3 hours"):
            with self.subTest(question=question):
                self.assertIsNone(self._untouched(
                    "It's 3 PM in London, sir.", question=question))
        # The live question is still checked.
        got = wc.check_time_claim(TimeClaimGuardTests.LIVE, NOW,
                                  question="what time is it in London")
        self.assertTrue(got.corrected)
        self.assertFalse(wc.question_is_about_another_moment(
            "what's it like in London now"))

    def test_wc2_a_longer_unknown_place_is_not_its_first_word(self):
        for reply in ("It's 6:17 AM in Eastern Europe, sir.",
                      "It's 5:17 AM in Central Europe, sir.",
                      "It's 9:17 PM in Central America, sir.",
                      "It's 8:17 AM in Central Asia, sir.",
                      "It's 1:17 PM in eastern Australia, sir.",
                      "It's 8:17 PM in Mountain View, sir.",
                      "It's 10:17 PM in the Rio Grande Valley, sir.",
                      "In Central Europe, it's 5:17 AM, sir."):
            with self.subTest(reply=reply):
                self.assertIsNone(self._untouched(reply))
        # Filler after a known place still keeps it.
        got = wc.check_time_claim("It is 10:17 PM in London right now, sir.",
                                  NOW)
        self.assertTrue(got.corrected)

    def test_wc3_a_same_named_town_is_not_the_mapped_city(self):
        for reply in ("It's 10:17 PM in Athens, Georgia, sir.",
                      "It's 10:17 PM in Athens Georgia, sir.",
                      "It's 10:17 PM in Paris, Texas, sir.",
                      "It's 11:17 PM in London, Ontario, sir.",
                      "It's 8:17 PM in Moscow, Idaho, sir.",
                      "It's 11:17 PM in Dublin, Ohio, sir.",
                      "It's 11:17 PM in London, Canada, sir."):
            with self.subTest(reply=reply):
                self.assertIsNone(self._untouched(reply))
        # A qualifier that agrees keeps the claim (and a wrong time there is
        # still corrected).
        self.assertFalse(wc.check_time_claim(
            "It's 5:17 AM in Paris, France, sir.", NOW).corrected)
        self.assertTrue(wc.check_time_claim(
            "It's 10:17 PM in Paris, France, sir.", NOW).corrected)
        self.assertFalse(wc.check_time_claim(
            "It's 1:17 PM in Sydney, Australia, sir.", NOW).corrected)

    def test_wc4_a_claim_never_crosses_a_sentence_or_another_time(self):
        for reply in ("It's 10:17 PM. In London, it's 4:17 AM.",
                      "It's 10:17 PM. In the UK they're asleep.",
                      "It's 10:17 PM. In London the match starts at 3 PM "
                      "tomorrow.",
                      "It's 10 PM here, 4 AM in London, sir.",
                      "It is 10:17 p.m. In London it is 4:17 a.m."):
            with self.subTest(reply=reply):
                self._untouched(reply)
        # "in St. Louis" / "a.m. in" keep their dots inside one sentence.
        self.assertFalse(wc.check_time_claim(
            "It's 10:17 PM in St. Louis, sir.", NOW).corrected)
        self.assertTrue(wc.check_time_claim(
            "It's currently 10:17 p.m. in London.", NOW).corrected)

    def test_wc5_masked_blanks_only_the_place_claims(self):
        # Changed deliberately (second review, WC-1 residual): this used a
        # WRONG local time ("10:02 PM here"). A second clock time in the
        # sentence that is not the time here now now marks a conversion, so
        # that reply has no place claim at all (None: the caller's local-time
        # check scans all of it, and get_time is still injected — pinned in
        # tests/monolith/test_monolith_claim_validation.py). The masking is
        # pinned with the true local time beside the London clause.
        reply = "It's 10:17 PM here, sir, and in London it's about 4 AM."
        got = wc.check_time_claim(reply, NOW)
        self.assertFalse(got.corrected)
        self.assertEqual(len(got.masked), len(reply))
        self.assertIn("It's 10:17 PM here", got.masked)
        self.assertNotIn("London", got.masked)
        self.assertNotIn("4 AM", got.masked)
        self.assertIsNone(wc.check_time_claim(
            "It's 10:02 PM here, sir, and in London it's about 4 AM.", NOW))
        # A correction leaves nothing of its own true lines to scan.
        got = wc.check_time_claim(
            "[intent:confirmation] It is 10:17 PM in London, sir.", NOW)
        self.assertEqual(got.masked, "[intent:confirmation]")


@_needs_tz
class GuardSecondReviewTests(unittest.TestCase):
    """The second review of the guard: each case below still replaced a
    correct LLM answer (or let the live bug through) after the first fixes."""

    def _untouched(self, reply, question=None):
        got = wc.check_time_claim(reply, NOW, question=question)
        self.assertTrue(got is None or not got.corrected, got)
        if got is not None:
            self.assertEqual(got.reply, reply)

    def test_wc3_an_abbreviated_or_unknown_qualifier_is_another_town(self):
        for reply in ("It's 11:17 PM in Athens, GA, sir.",
                      "It's 11:17 PM in London, KY, sir.",
                      "It's 10:17 PM in Moscow, ID, sir.",
                      "It's 11:17 PM in Manchester, NH, sir.",
                      "It's 11:17 PM in London, ON, sir.",
                      "It's 10:17 PM in Paris, TX, sir.",
                      "It's 11:17 PM in Rome, GA, sir.",
                      "It's 11:17 PM in Dublin, OH, sir.",
                      "It's 11:17 PM in Athens, Ga., sir.",
                      "It's 10:17 PM in Paris, Tenn., sir.",
                      "It's 11:17 PM in London, Ont., sir.",
                      "It's 1:17 PM in Sydney, New South Wales, sir."):
            with self.subTest(reply=reply):
                self._untouched(reply,
                                question="what time is it in Athens Georgia")
                self._untouched(reply)
        # A qualifier that names the same place keeps the check, and filler
        # after the comma is not a qualifier.
        self.assertTrue(wc.check_time_claim(
            "It's 4:17 AM in Chicago, IL, sir.", NOW).corrected)
        self.assertFalse(wc.check_time_claim(
            "It's 10:17 PM in Chicago, IL, sir.", NOW).corrected)
        self.assertTrue(wc.check_time_claim(
            "It's 10:17 PM in London, UK, sir.", NOW).corrected)
        for reply in ("It's 10:17 PM in London, sir.",
                      "It's 10:17 PM in London, Wednesday morning, sir.",
                      "It's 10:17 PM in London, which is late, sir."):
            with self.subTest(reply=reply):
                self.assertTrue(wc.check_time_claim(reply, NOW).corrected)

    def test_wc1_a_second_clock_time_that_is_not_now_is_a_conversion(self):
        for reply, question in (
                ("Tokyo is 14 hours ahead of you, sir, so at 9 AM here it's "
                 "11 PM in Tokyo.",
                 "what is the time difference between here and Tokyo"),
                ("London is six hours ahead, sir. At 12 PM here, it's 6 PM "
                 "in London.", "how far ahead is London"),
                ("Six hours, sir: at 3 PM here it's 9 PM in London.",
                 "how many hours ahead is London"),
                ("London is six hours ahead of you, sir: at 9 AM here, it's "
                 "3 PM in London.",
                 "what's the time difference between here and London"),
                ("Tokyo is 14 hours ahead, sir, so at 8 AM your time it's "
                 "10 PM in Tokyo.", "how far ahead is Tokyo"),
                ("Six hours, sir. At noon here, it's 6 PM in London.",
                 "how many hours ahead is London"),
                ("By then it's 3 PM in London, sir.", None)):
            with self.subTest(reply=reply):
                self._untouched(reply, question=question)
                self._untouched(reply)      # the reply alone says so too
        # The time here NOW beside it is the two-place answer, still checked:
        # the live bug in its "here and in London" form.
        got = wc.check_time_claim(
            "It's 10:17 PM here, and in London it's 10:17 PM.", NOW,
            question="what time is it here and in London")
        self.assertTrue(got.corrected)
        self.assertEqual(got.reply, TimeClaimGuardTests.TRUE_LONDON)
        # Two known places are two claims, not a conversion.
        self.assertTrue(wc.check_time_claim(
            "In London, it's 4:17 AM, and in Tokyo, it's 10:17 PM.",
            NOW).corrected)

    def test_wc1_a_time_now_question_with_a_reason_is_still_checked(self):
        # The first fix's question blacklist ("call", "game", "next", ...)
        # switched the guard off for these, and the live bug went out as is.
        for question, reply in (
                ("what time is it in London? I want to call my mom",
                 TimeClaimGuardTests.LIVE),
                ("is it too late to call London", TimeClaimGuardTests.LIVE),
                ("what time is it in Tokyo right now, is it too late to call",
                 "It is 10:17 PM in Tokyo, sir."),
                ("what time is it in London, did the game start",
                 TimeClaimGuardTests.LIVE)):
            with self.subTest(question=question):
                self.assertFalse(wc.question_is_about_another_moment(question))
                self.assertTrue(wc.check_time_claim(
                    reply, NOW, question=question).corrected)
        for question in ("what is the time difference between here and Tokyo",
                         "how far ahead is London",
                         "how many hours behind is Los Angeles",
                         "what time does the match start in London",
                         "what time is it in London when I land",
                         "when does the market open in Tokyo"):
            with self.subTest(question=question):
                self.assertTrue(wc.question_is_about_another_moment(question))

    def test_wc2_the_time_suffix_never_shortens_a_longer_place(self):
        for reply, question in (
                ("It's 1:17 PM New South Wales time, sir.",
                 "what time is it in New South Wales"),
                ("It's 11:17 PM New England time, sir.", None)):
            with self.subTest(reply=reply):
                self._untouched(reply, question=question)
        # A meridiem or filler in front is still dropped.
        self.assertFalse(wc.check_time_claim(
            "It's 11:17 PM Eastern time, sir.", NOW).corrected)
        self.assertTrue(wc.check_time_claim(
            "It's 4:17 PM local London time, sir.", NOW).corrected)

    def test_wc4_a_connective_or_the_local_clock_ends_the_claim(self):
        for reply in ("It's 10:17 PM here, whereas in London, it's 4:17 AM.",
                      "It's 10:17 PM here, meanwhile in London, it's 4:17 AM.",
                      "It's 10:17 PM here, though in London, it's 4:17 AM.",
                      "It's 10:17 PM for you, over in London, it's 4:17 AM.",
                      "It's 10:17 PM, in London, it's 4:17 AM."):
            with self.subTest(reply=reply):
                self._untouched(reply,
                                question="what time is it here and in London")
        # The London clause itself is still checked.
        self.assertTrue(wc.check_time_claim(
            "It's 10:17 PM here, whereas in London, it's 10:17 PM.",
            NOW).corrected)
        # "here in <place>" names the place.
        self.assertFalse(wc.check_time_claim(
            "It's 10:17 PM here in Chicago, sir.", NOW).corrected)


class ContainerMembershipTests(unittest.TestCase):
    """Review WC-6: "<city> <multi-zone country>" returned the city without
    checking it is IN that country, so the fast path spoke London UK's time
    for London, Ontario."""

    def test_a_city_outside_the_country_is_none(self):
        for name in ("London Canada", "London, Canada", "Moscow USA",
                     "Paris United States", "Sydney Canada", "Perth Russia",
                     "Tokyo Brazil", "London the US"):
            with self.subTest(name=name):
                self.assertIsNone(wc.resolve_place(name))

    def test_a_city_inside_the_country_resolves(self):
        for name, label in (("Toronto Canada", "Toronto"),
                            ("Sydney Australia", "Sydney"),
                            ("Chicago USA", "Chicago"),
                            ("Honolulu, the United States", "Honolulu"),
                            ("Moscow Russia", "Moscow"),
                            ("Sao Paulo Brazil", "Sao Paulo"),
                            ("Memphis Tennessee", "Memphis")):
            with self.subTest(name=name):
                self.assertEqual(wc.resolve_place(name).label, label)

    @_needs_tz
    def test_the_fast_path_defers_to_the_llm(self):
        for q in ("what time is it in London Canada",
                  "what time is it in London, Canada",
                  "what time is it in Moscow USA",
                  "what time is it in Paris United States",
                  "what time is it in Sydney Canada",
                  "what time is it in Perth Russia"):
            with self.subTest(q=q):
                self.assertIsNone(wc.answer(q, NOW))
        self.assertEqual(wc.answer("what time is it in Toronto Canada",
                                   NOW).reply,
                         "It's 11:17 PM in Toronto, sir.")

    def test_every_container_zone_exists(self):
        if not _HAVE_TZ:
            self.skipTest("no IANA time zone data")
        for name, zones in wc._CONTAINERS.items():
            for zone in zones:
                with self.subTest(container=name, zone=zone):
                    self.assertIsNotNone(wc._zoneinfo(zone), zone)


if __name__ == "__main__":
    unittest.main()
