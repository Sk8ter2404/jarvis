"""Chain guards for a multi-round action turn (NEW #6, 2026-10-01).

THE LIVE TURN (2026-10-01 20:59:08 - 21:00:50, "look for better or faster
models that fit the hardware"): web_search ran at 20:59:13, then four rounds
of see_screen. At 21:00:36 the continuation enforcer reported "missing:
web_search (search the web)" - the reply said "the search results suggest
GGUF …", a NOUN phrase, and the search had already run 80 s earlier - so the
next round re-ran the identical web_search (a second Google tab at 21:00:41)
and said "I'm running the numbers now, sir; I'll have those results for you
in a moment." Then the loop guard stopped the chain and nothing more was
said: 102.9 s, no answer.

Four fixes, each pinned here:

  * _detect_dropped_steps counts an action that already ran THIS TURN (the
    grounding ledger) as done, not only one emitted in the same reply;
  * its web_search intent needs a VERB form ("search for X", "google it",
    "look it up"), never the noun "search results";
  * a second web_search / open_url in a later round of the same turn - or the
    same one twice in one reply - is refused before it opens another tab
    (two different sites in ONE reply still both open);
  * a chain the guards stop while the owner has heard nothing but
    acknowledgements and promises ends with an honest close-out line.

Drives the REAL _run_llm_dispatch (grounding ledger and all) with stub actions
and a recording _speak (the answer-first harness). No audio, no LLM, no
browser.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_chain_guards
"""
from __future__ import annotations

import contextlib
import io
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_answer_first import _Base

_USER = ("Jarvis, for now, let's just stay on what we have, but I definitely "
         "want to see if there's any better options, like see if there's any "
         "better, faster, stronger, but requires the same hardware, you know.")
_QUERY = "best local LLM benchmarks 2026 speed vs intelligence"
_SEARCH_OK = (f"opened Google search for '{_QUERY}' — use see_screen to read "
              f"the results")

# The live replies, verbatim from the 2026-10-01 session log.
_R_205910 = (
    f"[intent:confirmation] On it, sir. [ACTION: web_search, {_QUERY}] "
    "[intent:briefing] I'll run the numbers on current quantizations and "
    "model architectures to find a more efficient configuration for your "
    "hardware, sir.")
_R_210024 = (
    "[intent:confirmation] On it. [ACTION: see_screen, "
    "screenshot_20261001_204535.png] I'm afraid that file doesn't exist, "
    "sir; I'll have to try a fresh capture. [ACTION: screenshot]"
    "[intent:confirmation] One moment. [ACTION: see_screen, "
    "screenshot_20261001_204602.png][intent:briefing] I've got it now, sir; "
    "the search results suggest looking into GGUF quantization levels to "
    "squeeze more intelligence out of your existing VRAM.")
_R_210038 = (
    f"[intent:confirmation] On it, sir. [ACTION: web_search, {_QUERY}] "
    "[intent:briefing] I'm running the numbers now, sir; I'll have those "
    "results for you in a moment.")
_BUDGET = ("see_screen budget for this intent is exhausted (3/3 captures "
           "used). Use recall_screen to query the cached visual state from "
           "the last capture.")


@requires_monolith
class _ChainBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "MISSION_NARRATION_ENABLED", False)
        self.searches: list = []
        self.opened: list = []

        def _search(q=""):
            self.searches.append(q)
            return f"opened Google search for '{q}' — use see_screen to read the results"

        def _open(u=""):
            self.opened.append(u)
            return f"opened https://{u} — use see_screen to read what loaded"

        bc.ACTIONS["web_search"] = _search
        bc.ACTIONS["open_url"] = _open
        bc.ACTIONS["see_screen"] = lambda a="": _BUDGET
        bc.ACTIONS["screenshot"] = (
            lambda a="": "screenshot saved to C:\\shots\\shot.png")

    def _dispatch(self, first, followups, text=_USER):
        """The real _run_llm_dispatch (ledger open) with a canned first reply
        and canned follow-up rounds. Returns the printed output."""
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

    def _in_turn(self, ran=()):
        """Open a grounding ledger as _run_llm_dispatch does, with ``ran``
        (name, result) pairs already recorded; closed on cleanup."""
        bc = self.bc
        prev = bc._begin_turn_grounding(_USER)
        self.addCleanup(bc._end_turn_grounding, prev)
        for name, result in ran:
            bc._note_turn_action_ran(name, result)


class DroppedStepTests(_ChainBase):
    def test_an_action_that_ran_earlier_this_turn_is_not_dropped(self):
        self._in_turn(ran=[("web_search", _SEARCH_OK)])
        dropped = self.bc._detect_dropped_steps(
            "I'll search for faster quantizations next.", {"see_screen"})
        self.assertEqual(dropped, [])

    def test_the_noun_search_results_is_not_a_promise(self):
        dropped = self.bc._detect_dropped_steps(_R_210024, {"see_screen",
                                                            "screenshot"})
        self.assertNotIn("web_search", [a for a, _d in dropped])

    def test_verb_forms_still_count(self):
        for prose in ("I'll search for faster models, sir.",
                      "Let me search the web for that.",
                      "I'll google it for you.",
                      "Then I'll look it up.",
                      "I'll run a quick search on it."):
            with self.subTest(prose=prose):
                dropped = self.bc._detect_dropped_steps(prose, {"see_screen"})
                self.assertIn("web_search", [a for a, _d in dropped])

    def test_live_210024_round_adds_no_dropped_step(self):
        self._in_turn(ran=[("web_search", _SEARCH_OK)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _cleaned, results = self.bc.parse_and_run_actions(_R_210024)
        self.assertNotIn("_dropped_step", [n for n, _r, _i in results])
        self.assertNotIn("[continuation_enforcer]", buf.getvalue())


class RepeatSearchTests(_ChainBase):
    def test_a_second_web_search_in_a_later_round_is_refused(self):
        self._in_turn(ran=[("web_search", _SEARCH_OK)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _cleaned, results = self.bc.parse_and_run_actions(_R_210038)
        self.assertEqual(self.searches, [], "no second Google tab")
        name, result, _info = results[0]
        self.assertEqual(name, "web_search")
        self.assertTrue(self.bc._action_result_failed(result), result)

    def test_a_different_query_in_a_later_round_is_refused_too(self):
        self._in_turn(ran=[("web_search", _SEARCH_OK)])
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc.parse_and_run_actions(
                "[ACTION: web_search, fastest 24GB models]")
        self.assertEqual(self.searches, [])

    def test_the_same_search_twice_in_one_reply_runs_once(self):
        self._in_turn()
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc.parse_and_run_actions(
                f"[ACTION: web_search, {_QUERY}] [ACTION: web_search, {_QUERY}]")
        self.assertEqual(self.searches, [_QUERY])

    def test_two_sites_in_one_reply_both_open(self):
        self._in_turn()
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc.parse_and_run_actions(
                "Opening both, sir. [ACTION: open_url, github.com] "
                "[ACTION: open_url, mail.google.com]")
        self.assertEqual(self.opened, ["github.com", "mail.google.com"])

    def test_outside_a_turn_nothing_is_refused(self):
        # No ledger (the proactive path): the old behaviour, both run.
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc.parse_and_run_actions(f"[ACTION: web_search, {_QUERY}]")
            self.bc.parse_and_run_actions(f"[ACTION: web_search, {_QUERY}]")
        self.assertEqual(self.searches, [_QUERY, _QUERY])

    def test_the_live_chain_opens_one_tab(self):
        self._dispatch(_R_205910, [
            "[intent:confirmation] Right away, sir. [ACTION: see_screen, "
            "read the search results]",
            _R_210024,
            _R_210038,
        ])
        self.assertEqual(self.searches, [_QUERY])


class CloseOutTests(_ChainBase):
    def test_the_live_chain_ends_with_an_honest_close_out(self):
        printed = self._dispatch(_R_205910, [
            "[intent:confirmation] Right away, sir. [ACTION: see_screen, "
            "read the search results]",
            _R_210024,
            _R_210038,
        ])
        self.assertTrue(self.spoken, "something must be said at the end")
        last = self.spoken[-1]
        self.assertIn("I'm afraid", last)
        self.assertIn("browser", last)
        self.assertIn("[close-out]", printed)
        self.assertEqual(self.bc.conversation_history[-1],
                         {"role": "assistant", "content": last})
        for s in self.spoken:
            self.assertNotIn("in a moment", s, "no promise left hanging")

    def test_a_chain_that_answered_gets_no_close_out(self):
        answer = ("The page lists Qwen3 thirty-two B at four-bit as the "
                  "fastest fit for twenty-four gigabytes, sir.")
        self.bc.ACTIONS["see_screen"] = (
            lambda a="": "A Google results page about local LLM benchmarks.")
        printed = self._dispatch(_R_205910, [
            "[ACTION: see_screen, read the search results]",
            f"{answer} [ACTION: see_screen, read the search results]",
            "[ACTION: see_screen, read the search results]",
        ])
        self.assertIn(answer, self.spoken)
        self.assertNotIn("[close-out]", printed)

    def test_a_chain_that_finishes_gets_no_close_out(self):
        self.bc.ACTIONS["see_screen"] = (
            lambda a="": "A Google results page about local LLM benchmarks.")
        printed = self._dispatch(_R_205910, [
            "[ACTION: see_screen, read the search results]",
            "Qwen3 at four-bit is your best fit, sir.",
        ])
        self.assertEqual(self.spoken[-1],
                         "Qwen3 at four-bit is your best fit, sir.")
        self.assertNotIn("[close-out]", printed)

    def test_the_depth_cap_with_nothing_said_gets_a_close_out(self):
        bc = self.bc
        seen = iter(range(100))
        bc.ACTIONS["see_screen"] = (
            lambda a="": f"A loading spinner, frame {next(seen)}.")
        with mock.patch("core.mode_router.followup_loop_depth",
                        return_value=2):
            printed = self._dispatch(
                "On it, sir. [ACTION: see_screen, is it loaded]",
                ["On it. [ACTION: see_screen, is it loaded]",
                 "Right away. [ACTION: see_screen, is it loaded]"],
                text="tell me when the page has loaded")
        self.assertIn("[close-out] the chain stopped (depth cap)", printed)
        self.assertEqual(self.spoken[-1], bc._CLOSE_OUT_GENERIC)

    def test_a_reported_success_then_a_guard_stop_gets_no_close_out(self):
        self.bc.ACTIONS["play_music"] = (
            lambda a="": "Playing Thriller by Michael Jackson.")
        printed = self._dispatch(
            "[ACTION: play_music, thriller]",
            ["Playing it now, sir. [ACTION: play_music, thriller]"],
            text="play thriller")
        self.assertIn("[follow-up] informative result(s) repeating", printed)
        self.assertNotIn("[close-out]", printed)
        self.assertEqual(self.spoken, ["Playing it now, sir."])

    def test_a_single_action_turn_is_untouched(self):
        printed = self._dispatch("[ACTION: volume_up] Louder, sir.", [])
        self.assertEqual(self.spoken, ["Louder, sir."])
        self.assertNotIn("[close-out]", printed)

    def test_a_chain_that_ends_on_a_promise_gets_a_close_out(self):
        # Review repair (2026-10-02): the guards are not the only way a chain
        # ends on a promise. The follow-up after the search says "I'm running
        # the numbers now, sir; I'll have those results for you in a moment."
        # with no token, the loop simply finishes - and nothing follows.
        printed = self._dispatch(_R_205910, [
            "[intent:briefing] I'm running the numbers now, sir; I'll have "
            "those results for you in a moment.",
        ])
        self.assertIn("[close-out]", printed)
        self.assertIn("I'm afraid", self.spoken[-1])
        self.assertIn("browser", self.spoken[-1])

    def test_a_follow_up_that_ends_on_an_answer_gets_no_close_out(self):
        printed = self._dispatch(_R_205910, [
            "Qwen3 at four-bit looks like your best fit, sir.",
        ])
        self.assertNotIn("[close-out]", printed)
        self.assertEqual(self.spoken[-1],
                         "Qwen3 at four-bit looks like your best fit, sir.")

    # v3.0.6 (CHAIN_PROMISE_CHECK): the "ended on a promise" label is true.
    def _cut_of(self, first, followups):
        """Dispatch and return (printed, the chain cut note_chain recorded)."""
        trace = self._p(self.bc, "_turn_trace_call")
        printed = self._dispatch(first, followups)
        cuts = [c.args[3] for c in trace.call_args_list
                if c.args and c.args[0] == "note_chain"]
        self.assertEqual(len(cuts), 1)
        return printed, cuts[0]

    def test_an_answer_ending_is_not_labelled_a_promise(self):
        # Live 10-27 20:40:51 / 10-28 17:27:40: a plain spoken answer after a
        # screen read was filed as "ended on a promise" (the label asked only
        # "did the last round run no action?").
        printed, cut = self._cut_of(_R_205910, [
            "Qwen3 at four-bit looks like your best fit, sir.",
        ])
        self.assertNotIn("[close-out]", printed)
        self.assertNotEqual(cut, "ended on a promise")
        self.assertEqual(cut, "")

    def test_a_promise_after_substance_gets_a_close_out(self):
        # The first reply said something real, the follow-up only promised -
        # and nothing followed. Before, the earlier substance excused it.
        first = (f"The search is open, sir. [ACTION: web_search, {_QUERY}]")
        self.assertTrue(self.bc._says_something("The search is open, sir."))
        printed, cut = self._cut_of(first, [
            "[intent:briefing] One moment, sir. I'll have those results "
            "shortly.",
        ])
        self.assertEqual(cut, "ended on a promise")
        self.assertIn("[close-out]", printed)
        self.assertTrue(self.spoken[-1].startswith("I'm afraid"))
        with mock.patch.object(self.bc, "CHAIN_PROMISE_CHECK", False,
                               create=True):
            self.spoken.clear()
            printed = self._dispatch(first, [
                "[intent:briefing] One moment, sir. I'll have those results "
                "shortly.",
            ])
        self.assertNotIn("[close-out]", printed, "off = the old rule")

    # Live 2026-10-28 review: a courtesy after a chain that ANSWERED was filed
    # as "ended on a promise" (is_progress_only is True for "Very good, sir.")
    # and drew "I'm afraid I couldn't get a proper answer out of that".
    _ANSWERED = ("The top result recommends Qwen3 at four-bit for your card, "
                 "sir. [ACTION: open_url, huggingface.co/Qwen]")

    def test_a_courtesy_after_a_real_answer_gets_no_close_out(self):
        first = f"[ACTION: web_search, {_QUERY}]"
        for last in ("Very good, sir.", "Indeed, sir.", "Enjoy, sir.",
                     "Understood, sir.", "I'll leave it open for you, sir.",
                     "I'll keep an eye out for newer benchmarks, sir.",
                     "I'll remind you in five minutes, sir."):
            with self.subTest(last=last):
                self.spoken.clear()
                printed = self._dispatch(first, [self._ANSWERED, last])
                self.assertNotIn("[close-out]", printed)
                self.assertEqual(self.spoken[0], "The top result recommends "
                                 "Qwen3 at four-bit for your card, sir.")
                self.assertNotIn("I'm afraid", " ".join(self.spoken))

    def test_a_real_promise_after_an_answer_still_gets_a_close_out(self):
        first = f"[ACTION: web_search, {_QUERY}]"
        printed = self._dispatch(first, [
            self._ANSWERED, "One moment, sir. I'll have those results "
            "shortly."])
        self.assertIn("[close-out]", printed)
        self.assertTrue(self.spoken[-1].startswith("I'm afraid"))

    def test_a_courtesy_after_nothing_real_keeps_the_old_close_out(self):
        # Nothing of substance was said all turn: the courtesy is not an
        # answer, so the honest line still speaks (the pre-v3.0.6 rule).
        printed, cut = self._cut_of(f"On it, sir. [ACTION: web_search, "
                                    f"{_QUERY}]", ["Very good, sir."])
        self.assertEqual(cut, "ended on a courtesy")
        self.assertIn("[close-out]", printed)
        with mock.patch.object(self.bc, "CHAIN_PROMISE_CHECK", False,
                               create=True):
            self.spoken.clear()
            printed = self._dispatch(f"[ACTION: web_search, {_QUERY}]",
                                     [self._ANSWERED, "Very good, sir."])
        self.assertNotIn("[close-out]", printed, "off = the old rule: "
                         "earlier substance excuses it")

    def test_ended_on_promise_needs_a_result_owed(self):
        yes = ("[intent:briefing] One moment, sir. I'll have those results "
               "shortly.",
               "I'm running the numbers now, sir; I'll have those results "
               "for you in a moment.",
               "Let me take a look, sir.")
        no = ("Very good, sir.", "Indeed, sir.", "Of course, sir.", "",
              "Understood, sir.", "Enjoy, sir.", "Will do, sir.",
              "I'll leave it open for you, sir.",
              "I'll remind you in five minutes, sir.",
              "Qwen3 at four-bit looks like your best fit, sir.",
              "I've opened the results; shall I read them to you?")
        for t in yes:
            with self.subTest(yes=t):
                self.assertTrue(self.bc._chain_ended_on_promise(t))
        for t in no:
            with self.subTest(no=t):
                self.assertFalse(self.bc._chain_ended_on_promise(t))

    def test_an_offer_after_an_action_is_not_a_promise(self):
        # Live 10-28 15:57:59: "... I've opened the search results ... shall
        # I read them to you?" - an offer is substance, not a dropped promise.
        printed, cut = self._cut_of(_R_205910, [
            "I've opened the results; shall I read them to you?",
        ])
        self.assertNotIn("[close-out]", printed)
        self.assertEqual(cut, "")

    def test_a_follow_up_whose_action_succeeds_gets_no_close_out(self):
        # "Opening the top result" + an open_url that ran is not a promise
        # left hanging: the action did what the line said.
        printed = self._dispatch(_R_205910, [
            "Opening the top result for you now, sir. [ACTION: open_url, "
            "example.com]",
        ])
        self.assertNotIn("[close-out]", printed)

    def test_a_barged_turn_says_nothing_more(self):
        bc = self.bc

        def _barge(t, *a, **k):
            self.spoken.append(t)
            bc._tts_interrupt_seq[0] += 1

        self._p(bc, "_speak", side_effect=_barge)
        printed = self._dispatch(_R_205910, [
            "[intent:confirmation] Right away, sir. [ACTION: see_screen, "
            "read the search results]",
            _R_210024,
            _R_210038,
        ])
        self.assertNotIn("[close-out]", printed)


class ProgressOnlyTests(_ChainBase):
    def test_acks_and_promises_are_progress_only(self):
        from core import claim_validator as cv
        for text in ("", "On it, sir.", "[intent:confirmation] Right away.",
                     "I'll have those results for you in a moment.",
                     "On it, sir. I'm running the numbers now, sir; I'll have "
                     "those results for you in a moment.",
                     "One moment, sir."):
            with self.subTest(text=text):
                self.assertTrue(cv.is_progress_only(text))

    def test_an_answer_is_not_progress_only(self):
        from core import claim_validator as cv
        for text in ("Qwen3 at four-bit is your best fit, sir.",
                     "I'm afraid the printer isn't reachable, sir.",
                     "On it, sir. The page lists three models.",
                     "Done, sir.",
                     "Playing it now, sir.",
                     "The printer is offline, I'll keep trying.",
                     "The print is at 84%, I'll check again shortly.",
                     "Shall I read them to you?"):
            with self.subTest(text=text):
                self.assertFalse(cv.is_progress_only(text))
