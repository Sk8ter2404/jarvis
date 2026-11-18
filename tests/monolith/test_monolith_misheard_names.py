"""Misheard names through the REAL turn (v3.0.26, ovh-b; live 2026-11-17).

17:12:41 "Jarvis open hollow hood." (his "holo HUD") ran [ACTION: launch_app,
Hollow Hood], which failed; the follow-up round offered a web search, and on
the next try "I'll assume you meant Hollow Knight; shall I launch that
instead?" became an open offer. These drive the REAL _run_llm_dispatch ->
_utterance_route_reply / _call_llm -> parse_and_run_actions -> the REAL
core.actions._act_launch_app, with a fake brain, stub hologram actions and
os.startfile always failing (nothing is launched, no window opens):

  * the live line routes to holo_mode_on without the brain;
  * a launch_app "Hollow Hood" the brain still writes gets the terminal offer:
    no follow-up round, the offer is the open one, and "yes" opens the
    hologram through the route (one call, no brain);
  * a plain not-found game keeps its follow-up round, whose results message
    now carries the no-guessing rule;
  * LAUNCH_NOT_FOUND_SUGGEST off is the 3.0.25 turn.

    python tools/run_tests.py monolith.test_monolith_misheard_names
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_offer_yes import _OfferBase

LAUNCH_HH = "[intent:confirmation] Certainly, sir. [ACTION: launch_app, Hollow Hood]"
LAUNCH_GAME = ("[intent:confirmation] Certainly, sir. "
               "[ACTION: launch_app, Example Quest Game]")
OFFER_LINE = ("I've no app called Hollow Hood, sir. "
              "Shall I open the holo HUD instead?")


@requires_monolith
class _MisheardBase(_OfferBase):
    def setUp(self):
        super().setUp()
        import core.actions as A
        import core.config as cfg
        self.A, self.cfg = A, cfg
        self._p(A.os, "startfile", create=True,
                side_effect=FileNotFoundError(2, "The system cannot find the "
                                                 "file specified"))
        self.popen = self._p(A.subprocess, "Popen")
        self._p(A.shutil, "which", return_value=None)
        self._p(A, "_start_apps", return_value=(("Example Racer", "id.racer"),))
        self._p(A, "_website_name_url", return_value=None)
        self._p(A, "_site_shortcut_url", return_value=None)
        self._p(A.sys, "platform", "win32")
        self._stub("holo_mode_on", "Holo mode up, sir.")
        self._stub("holo_mode_off", "Holo mode closed, sir.")
        self._stub("hand_menu", "Hand menu up, sir.")
        self.assertIn("launch_app", self._actions)   # the REAL one


class LiveLineRouteTests(_MisheardBase):
    def test_the_live_line_opens_the_hologram_without_the_brain(self):
        self.brain = lambda _p: self.fail("the brain was asked")
        printed = self._turn("Jarvis open hollow hood.")
        self.assertEqual(self.calls["holo_mode_on"], [""])
        self.assertIn("[route] hologram -> holo_mode_on", printed)
        self.assertEqual(self.prompts, [])

    def test_the_live_close_closes_it(self):
        self.brain = lambda _p: self.fail("the brain was asked")
        self._turn("Jarvis close holographic hug.")
        self.assertEqual(self.calls["holo_mode_off"], [""])


class FeatureOfferTests(_MisheardBase):
    def test_launch_app_hollow_hood_offers_and_yes_opens_it(self):
        replies = iter([LAUNCH_HH])
        self.brain = lambda _p: next(replies)
        printed = self._turn("Jarvis, could you open the hollow hood app thing?")
        # Offered, not run; no follow-up round to invent a name.
        self.assertEqual(self.calls["holo_mode_on"], [])
        self.gfr.assert_not_called()
        self.assertIn(OFFER_LINE, " ".join(self.spoken))
        self.assertIn("[launch-feature] no app called 'Hollow Hood'", printed)
        self.assertIn("[follow-up] terminal failure reported - stopping", printed)
        self.assertIn("[offer-yes] open offer: 'Shall I open the holo HUD "
                      "instead?'", printed)
        self.popen.assert_not_called()
        # Yes: the route opens it - once, without the brain.
        self.brain = lambda _p: self.fail("the brain was asked")
        printed = self._turn("Jarvis, yes.")
        self.assertIn("[route] yes to the feature offer -> holo_mode_on", printed)
        self.assertEqual(self.calls["holo_mode_on"], [""])
        self.assertEqual(len(self.prompts), 1)

    def test_no_to_the_offer_opens_nothing(self):
        replies = iter([LAUNCH_HH, "[intent:confirmation] Very good, sir."])
        self.brain = lambda _p: next(replies)
        self._turn("Jarvis, could you open the hollow hood app thing?")
        self._turn("Jarvis, no.")
        self.assertEqual(self.calls["holo_mode_on"], [])

    def test_without_the_offer_note_the_yes_answers_nothing(self):
        # Mutation guard for the terminal-line offer note: with the chain's
        # open offer cleared, the same yes routes nowhere.
        replies = iter([LAUNCH_HH, "[intent:amused] Standing by, sir."])
        self.brain = lambda _p: next(replies)
        self._turn("Jarvis, could you open the hollow hood app thing?")
        self.bc._open_offer.clear()
        printed = self._turn("Jarvis, yes.")
        self.assertNotIn("[route] yes to the feature offer", printed)
        self.assertEqual(self.calls["holo_mode_on"], [])


class NotFoundFollowupTests(_MisheardBase):
    def test_a_missing_game_keeps_its_round_with_the_rule(self):
        replies = iter([LAUNCH_GAME])
        self.brain = lambda _p: next(replies)
        self.followups = ["[intent:bad_news] It isn't installed, sir."]
        self._turn("Jarvis, could you start example quest game for me?")
        self.assertEqual(self.gfr.call_count, 1)
        info = self.gfr.call_args.args[0]
        self.assertEqual([n for n, _r in info], ["launch_app"])
        result = info[0][1]
        self.assertIn("not found as an app on this PC", result)
        self.assertTrue(result.endswith("; no installed app has a similar name"),
                        result)
        extra = self.bc._followup_extra(info)
        self.assertIn("Never guess another app or game name", extra)
        # ... and only for such a failure.
        self.assertNotIn("Never guess another app",
                         self.bc._followup_extra([("launch_app",
                                                   "launched Example Racer")]))

    def test_the_setting_off_is_the_3025_turn(self):
        self._p(self.cfg, "LAUNCH_NOT_FOUND_SUGGEST", False)
        replies = iter([LAUNCH_HH])
        self.brain = lambda _p: next(replies)
        self.followups = ["[intent:bad_news] I can't find it, sir."]
        printed = self._turn("Jarvis, could you open the hollow hood app thing?")
        self.assertNotIn("[launch-feature]", printed)
        self.assertEqual(self.gfr.call_count, 1)
        result = self.gfr.call_args.args[0][0][1]
        self.assertEqual(result, "could not launch Hollow Hood: not found as an "
                                 "app on this PC ([Errno 2] The system cannot "
                                 "find the file specified)")
        self.assertNotIn("Never guess another app",
                         self.bc._followup_extra(self.gfr.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
