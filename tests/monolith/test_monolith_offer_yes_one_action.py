"""v3.0.9 lane B2: a yes to a search offer runs ONE action.

Live 2026-10-28 21:12: "Jarvis Open Makers World on main monitor." failed to
launch a program, and the follow-up round offered "Shall I search for it on
the web instead?". At 21:12:56 the owner said "Yes, go ahead."; the brain
ran web_search (Google opened) and the follow-up round then wrote
[ACTION: browse_for, Makers World] - a paid browser agent on top of the
search: two actions for one yes.

The fix (OFFER_YES_ONE_ACTION, bobert_companion._agent_on_search_refusal):
once a web_search / open_url ran this turn, browse_for / browser_task (and
their aliases) are refused unless his words, or the offer he accepted, ask
for reading, summarising, comparing, research or buying. These drive the
REAL _run_llm_dispatch -> _call_llm -> parse_and_run_actions with stub
actions (the offer-yes harness of test_monolith_offer_yes). Paraphrased
fixtures; no LLM, no browser.

    python -m unittest tests.monolith.test_monolith_offer_yes_one_action
"""
from __future__ import annotations

import unittest

from tests.monolith.test_monolith_offer_yes import _OfferBase

OPEN_REPLY = ("[intent:confirmation] Right away, sir. "
              "[ACTION: open_on_monitor, main | Example Site]")
OFFER = "Shall I search for it on the web instead?"
OFFER_REPLY = ("[intent:bad_news] I couldn't find an application by that "
               "name, sir. " + OFFER)
SEARCH = "[intent:confirmation] On it, sir. [ACTION: web_search, Example Site]"
AGENT = "[intent:confirmation] On it, sir. [ACTION: browse_for, Example Site]"


class _Base(_OfferBase):
    def setUp(self):
        super().setUp()
        self._stub("open_on_monitor",
                   "could not launch Example Site: not found as an app on "
                   "this PC ([WinError 2])")
        self._stub("web_search",
                   "opened Google search for 'Example Site' (still loading)")
        self._stub("browse_for", "The top three results for Example Site "
                                 "are ...")

    def _flag(self, value):
        self._p(self.bc, "OFFER_YES_ONE_ACTION", value, create=True)

    def _live_shape(self, yes="Yes, go ahead."):
        """Turn 1: the failed open and the follow-up offer. Turn 2: the yes;
        the brain searches and the follow-up round adds the browser agent
        (what the live brain did)."""
        def brain(prompt):
            if "OFFER ACCEPTED" in prompt and OFFER in prompt:
                return SEARCH
            if len(self.prompts) == 1:
                return OPEN_REPLY
            return "[intent:amused] Standing by, sir."
        self.brain = brain
        self.followups = [OFFER_REPLY]
        self._turn("Jarvis open Example Site on main monitor.")
        self.followups = [AGENT]
        return self._turn(yes)


class OneActionTests(_Base):
    def test_the_live_shape_runs_only_the_search(self):
        self._flag(True)
        printed = self._live_shape()
        self.assertIn("[offer-yes] sir said yes to", printed)
        self.assertEqual(self.calls["web_search"], ["Example Site"])
        self.assertEqual(self.calls["browse_for"], [])
        self.assertIn("refused: a web search already ran this turn", printed)

    def test_without_the_flag_both_run(self):
        # Mutation guard: the old path runs the browser agent as well, so the
        # test above is green for the right reason.
        self._flag(False)
        self._live_shape()
        self.assertEqual(self.calls["web_search"], ["Example Site"])
        self.assertEqual(self.calls["browse_for"], ["Example Site"])

    def test_an_alias_of_the_agent_is_refused_too(self):
        self._flag(True)
        bc = self.bc
        self._actions["browse_web_for"] = self._actions["browse_for"]
        self._actions["google_it"] = self._actions["web_search"]
        ran = frozenset({"google_it"})
        # Outside a turn nothing is refused (the proactive path).
        self.assertEqual(bc._agent_on_search_refusal("browse_web_for", ran,
                                                     set()), "")
        self._p(bc._turn_grounding, "frame",
                {"user_text": "Yes, go ahead.", "ran": set()}, create=True)
        self.assertTrue(bc._agent_on_search_refusal(
            "browse_web_for", ran, set()).startswith("refused:"))


class AgentAskedTests(_Base):
    def test_his_words_asking_for_a_summary_let_the_agent_run(self):
        self._flag(True)
        self.brain = lambda prompt: SEARCH
        self.followups = [AGENT]
        self._turn("Jarvis, search for Example Site and summarise the top "
                   "results.")
        self.assertEqual(self.calls["web_search"], ["Example Site"])
        self.assertEqual(self.calls["browse_for"], ["Example Site"])

    def test_a_plain_agent_request_with_no_search_runs(self):
        self._flag(True)
        self.brain = lambda prompt: AGENT
        self._turn("Jarvis, look up Example Site.")
        self.assertEqual(self.calls["browse_for"], ["Example Site"])
        self.assertEqual(self.calls["web_search"], [])

    def test_the_refusal_rule_reads_the_accepted_offer_too(self):
        bc = self.bc
        self._flag(True)
        frame = {"user_text": "Yes, go ahead.", "ran": set(),
                 "accepted_offer": "Shall I read the top results for you?"}
        self._p(bc._turn_grounding, "frame", frame, create=True)
        self.assertEqual(bc._agent_on_search_refusal(
            "browse_for", frozenset({"web_search"}), set()), "")
        frame["accepted_offer"] = OFFER
        self.assertTrue(bc._agent_on_search_refusal(
            "browse_for", frozenset({"web_search"}), set()).startswith(
                "refused:"))
        # ... in the same reply as the search, too.
        self.assertTrue(bc._agent_on_search_refusal(
            "browse_for", frozenset(), {("web_search", "example site")}
        ).startswith("refused:"))
        # Other actions are never touched.
        self.assertEqual(bc._agent_on_search_refusal(
            "see_screen", frozenset({"web_search"}), set()), "")

    def test_the_browser_agents_own_trigger_phrases_let_it_run(self):
        # Review 10-29: core/prompts.py names "use the browser to <X>",
        # "have your browser <X>" and "autopilot the browser to <X>" as the
        # browser_task asks; an open_url / web_search first must not refuse
        # them.
        bc = self.bc
        self._flag(True)
        for said in ("Jarvis, open Example Shop and use the browser to add "
                     "earbuds to my cart.",
                     "Jarvis, have your browser check the weather.",
                     "Jarvis, autopilot the browser to sign me up.",
                     "Jarvis, open Example Site and find me a boat model.",
                     "Jarvis, search Example Forum and go through the top "
                     "posts."):
            self._p(bc._turn_grounding, "frame",
                    {"user_text": said, "ran": set()}, create=True)
            self.assertEqual(bc._agent_on_search_refusal(
                "browser_task", frozenset({"web_search"}), set()), "", said)

    def test_open_url_and_use_the_browser_in_one_reply_both_run(self):
        self._flag(True)
        self._stub("open_url", "opened https://example.com")
        self._stub("browser_task", "added the earbuds to the cart")
        self.brain = lambda prompt: (
            "[intent:confirmation] On it, sir. [ACTION: open_url, "
            "example.com] [ACTION: browser_task, add earbuds to my cart]")
        self._turn("Jarvis, open Example Shop and use the browser to add "
                   "earbuds to my cart.")
        self.assertEqual(self.calls["open_url"], ["example.com"])
        self.assertEqual(self.calls["browser_task"],
                         ["add earbuds to my cart"])

    def test_an_open_url_alone_is_named_as_a_page_not_a_search(self):
        bc = self.bc
        self._flag(True)
        self._p(bc._turn_grounding, "frame",
                {"user_text": "Yes, go ahead.", "ran": set()}, create=True)
        res = bc._agent_on_search_refusal(
            "browse_for", frozenset({"open_url"}), set())
        self.assertTrue(res.startswith("refused: that page already opened"),
                        res)
        self.assertNotIn("web search", res)
        res = bc._agent_on_search_refusal(
            "browse_for", frozenset(), {("open_url", "example.com")})
        self.assertTrue(res.startswith("refused: that page already opened"),
                        res)


if __name__ == "__main__":
    unittest.main()
