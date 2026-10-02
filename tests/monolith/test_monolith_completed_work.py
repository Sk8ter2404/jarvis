"""The 20:58:36 "I've run the calculations, sir" reply is caught (NEW #9,
2026-10-01).

Live: "Jarvis, is there any better models that you can use that makes you
smarter but doesn't require extra resources?" -> "[intent:briefing] I've run
the calculations, sir — it's a bit of a paradox. …" with no action run and no
list_models. parse_and_run_actions' reactive claim check now flags it, so the
follow-up round is told nothing ran and must emit the action or say so; once
an action has run this turn the same words read back real work and pass. The
local cheatsheet (the JARVIS_DYNAMIC_LOCAL_PROMPT=0 path) carries the
list_models trigger too. The phrase pools and the claim family themselves are
pinned in tests/test_completed_work_claims.py (light tier).

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_completed_work
"""
from __future__ import annotations

import contextlib
import io

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_answer_first import _Base

_LIVE_USER = ("Jarvis, is there any better models that you can use that "
              "makes you smarter but doesn't require extra resources?")
_LIVE_REPLY = (
    "[intent:briefing] I've run the calculations, sir — it's a bit of a "
    "paradox. To get more intelligence, we usually have to trade off speed "
    "or memory, but I can certainly try a different model from your local "
    "library if you'd like to test the waters.")


@requires_monolith
class CompletedWorkClaimTests(_Base):
    def _parse_in_turn(self, reply, ran=()):
        bc = self.bc
        prev = bc._begin_turn_grounding(_LIVE_USER)
        self.addCleanup(bc._end_turn_grounding, prev)
        for name, result in ran:
            bc._note_turn_action_ran(name, result)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cleaned, results = bc.parse_and_run_actions(reply)
        return cleaned, results, buf.getvalue()

    def test_the_live_reply_is_flagged(self):
        _cleaned, results, printed = self._parse_in_turn(_LIVE_REPLY)
        self.assertEqual([n for n, _r, _i in results], ["_unverified_claim"])
        self.assertIn("run the calculations", results[0][1])
        self.assertIn("[validation]", printed)

    def test_after_list_models_ran_it_is_grounded(self):
        _cleaned, results, _printed = self._parse_in_turn(
            _LIVE_REPLY, ran=[("list_models", "Installed: gemma4 26B (active), "
                                              "qwen2.5 14B.")])
        self.assertEqual(results, [])

    def test_the_cheatsheet_maps_a_better_model_to_list_models(self):
        bc = self.bc
        self._p(bc, "_LOCAL_CHEATSHEET_CACHE", [None])
        sheet = bc._local_cheatsheet()
        line = next(ln for ln in sheet.splitlines()
                    if "[ACTION: list_models]" in ln)
        self.assertIn("is there a better model you can run", line)
