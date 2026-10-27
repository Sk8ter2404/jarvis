"""A proactive remark runs nothing, so it never claims it did (v3 W06).

Live: "I've taken the liberty of pulling up your schedule for the day" (10-18
07:37:48, 10-19 08:39:12) and "I've taken the liberty of checking the local
weather" (10-20 09:45:39) were voiced with no action behind them; on 10-20 the
claim validator flagged the weather one and it was spoken anyway.

The REAL _do_proactive_turn and parse_and_run_actions run; the LLM, the voice
and the camera are faked. Synthetic lines only.

    python -m unittest tests.monolith.test_monolith_proactive_claims
"""
from __future__ import annotations

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_proactive_presence import _Base

WEATHER = ("[intent:observation] I've taken the liberty of checking the "
           "local weather, sir, and it is clear all afternoon.")
SCHEDULE = ("[intent:observation] I've taken the liberty of pulling up your "
            "schedule for the day, sir.")
PASSIVE = ("[intent:observation] Your print job has been completed, sir.")


@requires_monolith
class ProactiveClaimTests(_Base):
    def _turn(self, line, setting=True):
        bc = self.bc
        self._p(bc, "pause_face_tracking")
        self._p(bc, "resume_face_tracking")
        self._p(bc, "set_state")
        self._p(bc, "_thinking_loop")
        self._p(bc, "_reset_see_screen_budget")
        self._p(bc, "_apply_quip_layer", side_effect=lambda s, r: s)
        self._p(bc, "generate_proactive_comment", return_value=line)
        self._p(bc, "PROACTIVE_NO_ACTION_CLAIMS", setting, create=True)
        self._keep_history()
        speak = self._p(bc, "_speak")
        before = list(bc._proactive_remarks_at)
        bc._proactive_last_attempt_at[0] = 0.0
        _, log = self._quiet(bc._do_proactive_turn, {})
        return speak, log, before

    def test_first_person_claims_are_not_spoken(self):
        for line in (WEATHER, SCHEDULE):
            speak, log, _ = self._turn(line)
            self.assertEqual(speak.call_count, 0, line)
            self.assertIn("[proactive] dropped unverified first-person "
                          "remark", log)

    def test_dropped_remark_still_advances_pacing(self):
        speak, _, _ = self._turn(WEATHER)
        self.assertEqual(speak.call_count, 0)
        self.assertGreater(self.bc._proactive_last_attempt_at[0], 0.0)

    def test_passive_observation_is_still_spoken(self):
        speak, log, _ = self._turn(PASSIVE)
        self.assertEqual(speak.call_count, 1)
        self.assertNotIn("dropped unverified first-person", log)

    def test_setting_off_keeps_the_old_behaviour(self):
        speak, log, _ = self._turn(WEATHER, setting=False)
        self.assertEqual(speak.call_count, 1)
        self.assertNotIn("dropped unverified first-person", log)

    def test_prompt_forbids_claims_only_when_on(self):
        bc = self.bc
        self._p(bc, "_system_prompt", "BASE")
        seen = []
        self._p(bc, "_llm_quick",
                side_effect=lambda **k: seen.append(k["system"]) or "Hm.")
        for flag in (True, False):
            self._p(bc, "PROACTIVE_NO_ACTION_CLAIMS", flag, create=True)
            self._quiet(bc.generate_proactive_comment)
        self.assertIn("You have run nothing", seen[0])
        self.assertNotIn("You have run nothing", seen[1])

    def test_first_person_matcher(self):
        f = self.bc._proactive_first_person_claim
        mk = lambda p: [("_unverified_claim", f"reply claims '{p}' but x", True)]
        self.assertTrue(f(mk("I've taken the liberty")))
        self.assertTrue(f(mk("taken the liberty of pulling up")))
        self.assertTrue(f(mk("I checked the weather")))
        self.assertFalse(f(mk("has been completed")))
        self.assertFalse(f([]))

    def test_observations_with_ive_are_kept(self):
        f = self.bc._proactive_first_person_claim
        self.assertFalse(f([], "I've noticed you've been focused, sir."))
        self.assertTrue(f([], "I've taken the liberty of pulling up your "
                             "schedule."))
        self.assertTrue(f([], "I checked the weather, sir."))

    def test_words_starting_with_i_are_not_claims(self):
        f = self.bc._proactive_first_person_claim
        self.assertFalse(f([], "Iran's news is heavy today."))
        self.assertFalse(f([], "Irun is a town in Spain."))
        self.assertFalse(f([], "Iset and Iopened are not words."))
        self.assertTrue(f([], "I've pulled up your schedule."))
        self.assertTrue(f([], "I pulled up your schedule."))
        self.assertTrue(f([], "I have checked the weather."))
        self.assertTrue(f([], "Ive checked the weather."))


@requires_monolith
class ProactiveDropRunsNothingTests(_Base):
    """Security review PROMPT-1 (2026-10-27): a dropped remark runs nothing
    and leaves no question armed. The drop used to come AFTER
    parse_and_run_actions, so a held confirm ("buy") or a grounded-click
    "shall I press it?" stayed armed but was never spoken, and the owner's
    next unrelated "yes" would have pressed Buy."""

    BUY = ("I've taken the liberty of finishing that order, sir. "
           "[ACTION: click_on_screen, Buy now]")
    _turn = ProactiveClaimTests._turn

    def setUp(self):
        super().setUp()
        import core.grounded_click as gc
        self.gc = gc
        self._p(self.bc, "_pending_confirmation", [])
        self._p(self.bc, "_pending_confirmation_at", [0.0])
        gc.clear_pending()
        self.addCleanup(gc.clear_pending)

    def test_a_text_claim_never_reaches_the_dispatcher(self):
        bc = self.bc
        spy = self._p(bc, "parse_and_run_actions", return_value=("", []))
        speak, log, _ = self._turn(self.BUY)
        spy.assert_not_called()
        self.assertEqual(speak.call_count, 0)
        self.assertEqual(bc._pending_confirmation, [])
        self.assertIn("dropped unverified first-person remark", log)

    def test_a_results_drop_disarms_what_the_remark_armed(self):
        import time as _time
        bc, gc = self.bc, self.gc
        bc._pending_confirmation.append(("older", "kept"))
        bc._pending_confirmation_at[0] = 5.0

        def fake_dispatch(text, *a, **k):
            bc._pending_confirmation.append(("click_on_screen", "Buy now"))
            bc._pending_confirmation_at[0] = 9.0
            with gc._lock:
                gc._state["pending"] = {"ts": _time.time(), "kind": "confirm",
                                        "allow_yes": True, "options": []}
            return ("Shall I press 'Buy now', sir?",
                    [("_unverified_claim",
                      "reply claims 'I'll go ahead' but no action ran")])
        self._p(bc, "parse_and_run_actions", side_effect=fake_dispatch)
        speak, log, _ = self._turn("I'll go ahead and finish that order, sir. "
                                   "[ACTION: click_on_screen, Buy now]")
        self.assertEqual(speak.call_count, 0)
        self.assertIn("dropped unverified first-person remark", log)
        self.assertEqual(bc._pending_confirmation, [("older", "kept")])
        self.assertIsNone(gc.pending_choice())

    def test_a_results_drop_with_nothing_older_resets_the_clock(self):
        bc = self.bc

        def fake_dispatch(text, *a, **k):
            bc._pending_confirmation.append(("click_on_screen", "Buy now"))
            bc._pending_confirmation_at[0] = 9.0
            return ("Shall I?", [("_unverified_claim",
                                  "reply claims 'I'll go ahead' but nothing ran")])
        self._p(bc, "parse_and_run_actions", side_effect=fake_dispatch)
        self._turn("I'll go ahead, sir. [ACTION: click_on_screen, Buy now]")
        self.assertEqual(bc._pending_confirmation, [])
        self.assertEqual(bc._pending_confirmation_at[0], 0.0)
