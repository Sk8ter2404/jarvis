"""Facts read off the screen, the screen-read cap and the "out yet" route
(v3.0.1, live 2026-10-27 20:40-20:41).

Live, three turns in a row:

  * 20:40:37 "Jarvis, is it time to watch Vision Quest yet?" ran see_screen
    ("is Vision Quest on any of the streaming services?") - his screen can't
    know a release; that is a web search (core/prompts.py, BASIC web_search).
  * 20:41:16 after web_search + see_screen (742 chars of the Google page):
    "I've found them, sir - episodes five and six were released yesterday.
    It seems they dropped earlier than expected." Checked against the vision
    trace: the page DID say it ("episodes 5 and 6 drop earlier than expected
    13 hours ago", "Episodes 5 And 6 Release Date ... 1 day ago"), but no
    check looked - find_ungrounded_reading knows sensor numbers only.
    core.claim_validator.find_ungrounded_screen_fact now does: the live
    reply is kept against the live page, dropped against a page that says
    nothing of the kind.
  * 20:41:26-36 "Enough and awesome." (misheard) -> web_search, then
    see_screen, read_screen, read_screen on the same page. The per-turn
    capture cap never counted text reads; core.loop_guard.screen_read_refusal
    refuses the third read.

    python -m unittest tests.test_screen_fact_grounding
"""
from __future__ import annotations

import unittest

from core import claim_validator as cv
from core import loop_guard as lg

# The live page text (vision trace vt-20261027-204113-c7a8a4), with the
# account links cut.
LIVE_PAGE = (
    "[left] Chrome — 'Vision Quest episodes 5 and 6 release date - "
    "Google Search' (google.com) | headings: Search Results / Thinking / Top "
    "stories / Marvel's VisionQuest Episodes 5 & 6 Recap and Review | videos: "
    "\"The Tab About this result Here’s the full VisionQuest release "
    "schedule, as episodes 5 and 6 drop earlier than expected 13 hours ago\" "
    "| links: Share, Go to Google Home, Clear, AI Mode, All, News, Videos, "
    "Forbes About this result ‘VisionQuest’ Episodes 5 And 6 "
    "Release Date And Preview: Ultron Takes The Wheel 1 day ago, About this "
    "result")
# The first turn's read (vt-20261027-204039-7fae13): windows, no release.
WINDOWS_ONLY = ("Windows on screen:\n[left] 'Agent Office' [Chrome]\n"
                "[right] 'project.3mf - BambuStudio'")
LIVE_USER = ("Jarvis, do some research and see if it's out yet. The new "
             "episode five and six.")
LIVE_REPLY = ("[intent:briefing] I've found them, sir — episodes five "
              "and six were released yesterday. It seems they dropped "
              "earlier than expected.")


class LiveReplyTests(unittest.TestCase):
    def test_live_reply_is_grounded_by_the_live_page(self):
        self.assertIsNone(cv.find_ungrounded_screen_fact(
            LIVE_REPLY, grounding=[LIVE_PAGE], user_text=LIVE_USER))
        kept, bad = cv.drop_ungrounded_screen_facts(
            LIVE_REPLY, grounding=[LIVE_PAGE], user_text=LIVE_USER)
        self.assertEqual((kept, bad), (LIVE_REPLY, []))

    def test_live_reply_after_a_page_without_it_is_dropped(self):
        kept, bad = cv.drop_ungrounded_screen_facts(
            LIVE_REPLY, grounding=[WINDOWS_ONLY],
            user_text="Jarvis, is it time to watch Vision Quest yet?")
        self.assertEqual(kept, "")
        self.assertEqual(bad, ["released", "dropped"])


class RuleTests(unittest.TestCase):
    def _bad(self, text, page=LIVE_PAGE, user=LIVE_USER):
        return cv.find_ungrounded_screen_fact(text, grounding=[page],
                                              user_text=user)

    def test_relative_day_the_page_does_not_give(self):
        self.assertEqual(self._bad("Episodes five and six came out last "
                                   "week, sir."), "last week")
        self.assertEqual(self._bad("They release tomorrow, sir."),
                         "tomorrow")

    def test_today_is_grounded_by_hours_ago(self):
        self.assertIsNone(self._bad("Episodes five and six dropped today, "
                                    "sir."))

    def test_number_the_page_does_not_show(self):
        self.assertEqual(self._bad("Episode seven was released yesterday, "
                                   "sir."), "seven")

    def test_weekday_and_month(self):
        self.assertEqual(self._bad("Episode six comes out on Friday, sir."),
                         "friday")
        self.assertIsNone(cv.find_ungrounded_screen_fact(
            "It premiered on October 2, 2026, sir.",
            grounding=["Premiere date: Oct 2, 2026"]))
        self.assertEqual(cv.find_ungrounded_screen_fact(
            "It premiered on October 3, 2026, sir.",
            grounding=["Premiere date: Oct 2, 2026"]), "3")

    def test_hedges_questions_and_plain_reports_are_left_alone(self):
        for text in (
                "I can't tell whether it's out yet from this page, sir.",
                "Shall I read the results to you?",
                "If it's out, I can open Disney Plus for you, sir.",
                "You may want to check Disney Plus, sir.",
                "You have 'Agent Office' and two BambuStudio files open, sir.",
                "It seems to be mostly synonyms for 'awesome', sir."):
            with self.subTest(text=text):
                self.assertIsNone(cv.find_ungrounded_screen_fact(
                    text, grounding=[WINDOWS_ONLY]))

    def test_never_raises(self):
        self.assertIsNone(cv.find_ungrounded_screen_fact(None))
        self.assertEqual(cv.drop_ungrounded_screen_facts(""), ("", []))


class ScreenReadCapTests(unittest.TestCase):
    # The live third turn's calls, in order.
    LIVE_CALLS = [("web_search", "enough and awesome"), ("see_screen", ""),
                  ("see_screen", "left")]

    def test_live_third_read_is_refused(self):
        out = lg.screen_read_refusal("read_screen", self.LIVE_CALLS)
        self.assertTrue(out.startswith("refused:"))
        self.assertIn("2 times", out)

    def test_two_reads_are_allowed(self):
        self.assertEqual(lg.screen_read_refusal(
            "see_screen", self.LIVE_CALLS[:2]), "")
        self.assertEqual(lg.screen_read_refusal("see_screen", []), "")

    def test_a_step_between_reads_starts_a_fresh_count(self):
        # Review of 3.0.1: a chain that checks its work after each step, or
        # reads a long page a scroll at a time, must keep its reads.
        calls = [("see_screen", ""), ("click_on_screen", "Settings"),
                 ("see_screen", ""), ("scroll", "down"), ("see_screen", ""),
                 ("press", "tab"), ("read_screen", "left")]
        self.assertEqual(lg.screen_read_refusal("see_screen", calls), "")
        # Listing windows changes nothing: the ping-pong still counts.
        pingpong = [("see_screen", ""), ("list_windows", ""),
                    ("see_screen", ""), ("list_windows", "")]
        self.assertTrue(lg.screen_read_refusal("see_screen", pingpong))

    def test_other_actions_are_never_refused(self):
        self.assertEqual(lg.screen_read_refusal("click", self.LIVE_CALLS),
                         "")
        self.assertEqual(lg.screen_read_refusal(
            "web_search", self.LIVE_CALLS), "")

    def test_alias_counts_through_the_predicate(self):
        calls = [("look_screen", ""), ("look_screen", "")]
        self.assertEqual(lg.screen_read_refusal("look_screen", calls), "")
        self.assertTrue(lg.screen_read_refusal(
            "look_screen", calls, is_read=lambda n: n == "look_screen"))

    def test_fault_lets_the_read_run(self):
        def boom(_n):
            raise RuntimeError("x")
        self.assertEqual(lg.screen_read_refusal(
            "look_screen", [("look_screen", "")] * 3, is_read=boom), "")


class OutYetRouteTests(unittest.TestCase):
    def test_release_question_goes_to_web_search_in_the_core(self):
        from core import prompts
        from core import prompt_router as pr
        core, sections = pr.split_pc_control(prompts.PC_CONTROL_PROMPT)
        self.assertIn("out yet / time to watch", core)
        self.assertIn("release date", core)
        # "What's on my screen" keeps see_screen: its section is untouched.
        vision = " ".join(b for h, b in sections if h == "SCREEN VISION")
        self.assertIn("see_screen, <question>", vision)
        self.assertIn("ask a question about what's on screen", vision)


if __name__ == "__main__":
    unittest.main()
