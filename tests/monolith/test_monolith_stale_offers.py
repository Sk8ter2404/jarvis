"""Stale offers expire when their chain ends (NEW #13, 2026-10-01).

THE LIVE TURNS: the 20:59-21:00 "find better models" chain ended with a
web_search whose page nobody read. 35 minutes later, "how's my print
doing?" (21:36:12) got two replies that both ended with that chain's offer:

  21:36:14  "… Also, I've opened those search results in your browser;
            shall I read them to you via screen vision?"
  21:36:33  "[intent:bad_news] I'm afraid the printer isn't responding to my
            queries, sir; it seems I can't reach it at the moment. Also, I've
            opened those search results in your browser—shall I use screen
            vision to read them to you?"

ROOT CAUSE (verified in code): _local_chat_prompt's unread-search guard scans
the last six history messages for a web_search with no see_screen after it
and, with no notion of turns or time, told the model to "Acknowledge the
search was opened in the browser and offer to read the results via
see_screen" - 35 minutes and a whole new subject later. Nothing ever closed
the offer, and the adjacent-fact rule ("Also, sir…") gave it a slot.

Pinned here:
  * the guard keeps its "offer to read" wording only while the search's own
    turn is still running; for a search from an earlier turn it only forbids
    inventing the results, and after OFFER_TTL_S (10 min) it is gone;
  * offers made during a turn are recorded when its chain ends; a later
    turn's "Also, …" / "Incidentally, …" aside that repeats one is not
    spoken (the rest of the reply is); offers of the current turn and new
    offers are untouched.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_stale_offers
"""
from __future__ import annotations

import contextlib
import io
import time

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_answer_first import _Base

_QUERY = "best local LLM benchmarks 2026 speed vs intelligence"
_SEARCH_OK = (f"opened Google search for '{_QUERY}' — use see_screen to read "
              f"the results")
_HISTORY = [
    {"role": "user", "content": "find me better models for my hardware"},
    {"role": "assistant",
     "content": f"On it, sir. [ACTION: web_search, {_QUERY}]"},
    {"role": "user", "content": "Jarvis, how's my print doing?"},
]
_OPEN_OFFER = "offer to read the results via see_screen"

# Verbatim from the 2026-10-01 session log.
_R_205928 = (
    "[intent:confirmation] Right away, sir. [ACTION: see_screen, "
    "screenshot_20261001_204557.png] I've opened the search results in your "
    "browser; shall I read the findings to you, sir?")
_R_213633 = (
    "[intent:bad_news] I'm afraid the printer isn't responding to my "
    "queries, sir; it seems I can't reach it at the moment. Also, I've "
    "opened those search results in your browser—shall I use screen vision "
    "to read them to you?")
_PRINT = ("I don't have a fresh status from the printer yet, sir. Either it "
          "isn't reachable or the monitor hasn't connected.")


@requires_monolith
class _OfferBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "MISSION_NARRATION_ENABLED", False)
        # create=True: on a tree without the fix these tests then fail on
        # what is SAID, not on a missing attribute.
        self._p(bc, "_last_web_search_at", [0.0], create=True)
        self._p(bc, "_offer_ledger", [], create=True)
        bc.ACTIONS["web_search"] = lambda q="": _SEARCH_OK
        bc.ACTIONS["see_screen"] = (
            lambda a="": "A Google results page about local LLMs.")
        bc.ACTIONS["how_is_the_print"] = lambda a="": _PRINT

    def _guard_text(self):
        sys_prompt, msgs = self.bc._local_chat_prompt("BASE", list(_HISTORY))
        return sys_prompt + "\n".join(str(m.get("content", "")) for m in msgs)

    def _dispatch(self, text, first, followups):
        bc = self.bc

        def fake_llm(user_text):
            bc.conversation_history.append({"role": "user",
                                            "content": user_text})
            bc.conversation_history.append({"role": "assistant",
                                            "content": first})
            return first

        self._p(bc, "get_response_with_animation", side_effect=fake_llm)
        self.followup.side_effect = list(followups) + [""] * 4
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bc._run_llm_dispatch(text)
        return buf.getvalue()


class SearchGuardTests(_OfferBase):
    def test_this_turns_search_keeps_the_offer(self):
        bc = self.bc
        prev = bc._begin_turn_grounding("find me better models")
        self.addCleanup(bc._end_turn_grounding, prev)
        bc._note_turn_action_ran("web_search", _SEARCH_OK)
        bc._last_web_search_at[0] = time.time()
        self.assertIn(_OPEN_OFFER, self._guard_text())

    def test_an_earlier_turns_search_only_forbids_inventing_results(self):
        self.bc._last_web_search_at[0] = time.time() - 120
        text = self._guard_text()
        self.assertIn("Do NOT fabricate", text)
        self.assertNotIn(_OPEN_OFFER, text)

    def test_the_live_35_minute_old_search_gets_no_guard(self):
        self.bc._last_web_search_at[0] = time.time() - 35 * 60
        text = self._guard_text()
        self.assertNotIn("IMPORTANT: a web search", text)
        self.assertNotIn(_OPEN_OFFER, text)
        self.assertNotIn("Do NOT fabricate", text)

    def test_a_search_from_before_a_restart_gets_no_guard(self):
        # History restored from disk, nothing searched this process.
        self.assertNotIn("Do NOT fabricate", self._guard_text())

    def test_a_search_in_a_dispatch_is_stamped(self):
        self._dispatch("find me better models",
                       f"[ACTION: web_search, {_QUERY}]", [])
        self.assertGreater(self.bc._last_web_search_at[0], time.time() - 60)


class StaleOfferTests(_OfferBase):
    def _the_2100_chain(self):
        self._dispatch(
            "find me better models for my hardware",
            f"[intent:confirmation] On it, sir. [ACTION: web_search, {_QUERY}]",
            [_R_205928, "Very good, sir."])
        self.spoken.clear()

    def test_live_213633_the_old_offer_is_not_spoken(self):
        self._the_2100_chain()
        printed = self._dispatch("Jarvis, how's my print doing?",
                                 "[intent:briefing] [ACTION: how_is_the_print]",
                                 [_R_213633])
        said = " ".join(self.spoken)
        self.assertIn("isn't responding to my queries", said)
        self.assertNotIn("screen vision", said)
        self.assertNotIn("search results", said)
        self.assertIn("[stale-offer]", printed)

    def test_an_incidentally_aside_is_caught_too(self):
        self._the_2100_chain()
        self._dispatch("what's the capital of France",
                       "Paris, sir. Incidentally, shall I read those search "
                       "results in your browser to you?", [])
        self.assertEqual(self.spoken, ["Paris, sir."])

    def test_an_offer_of_the_current_turn_is_spoken(self):
        offer = ("I've opened the search results in your browser; shall I "
                 "read the findings to you, sir?")
        self._dispatch("search for better models",
                       f"[ACTION: web_search, {_QUERY}]", [offer])
        self.assertIn(offer, self.spoken)

    def test_a_new_offer_is_spoken(self):
        self._the_2100_chain()
        reply = ("The print is at 84 percent, sir. Also, shall I dim the "
                 "lights for the evening?")
        self._dispatch("how's my print doing", reply, [])
        self.assertEqual(self.spoken, [reply])

    def test_a_direct_question_about_the_results_is_not_an_aside(self):
        self._the_2100_chain()
        reply = ("Those search results are still open in your browser, sir; "
                 "shall I read them to you?")
        self._dispatch("what about those search results", reply, [])
        self.assertEqual(self.spoken, [reply])

    def test_the_ledger_is_bounded_and_ages_out(self):
        bc = self.bc
        for i in range(40):
            bc._record_turn_offers([f"Shall I open window number {i} for you?"])
        self.assertLessEqual(len(bc._offer_ledger), bc._OFFER_LEDGER_MAX)
        bc._offer_ledger[:] = [dict(e, at=e["at"] - 3 * 3600)
                               for e in bc._offer_ledger]
        bc._record_turn_offers(["Shall I read the news to you?"])
        self.assertEqual(len(bc._offer_ledger), 1)
