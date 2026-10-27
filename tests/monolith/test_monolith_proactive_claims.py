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
