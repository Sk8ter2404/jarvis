"""What game mode is allowed to CALL a failure — skills/game_mode.py:_classify.

THE BUG THESE PIN (measured on the owner's box, 2026-09-04 21:19, mid-match)

    GET http://127.0.0.1:11434/api/ps  ->  {"models": []}

Nothing was resident. So `_enter`'s unload had no work to do and freed 0 MB,
and the 8-second verify window straddled Fortnite ALLOCATING (10.04 -> 16.14 ->
17.13 GB in ~17 minutes that evening), which makes the whole-card delta zero or
NEGATIVE. `_verify` scored that against GAME_MODE_MIN_VRAM_DELTA_MB=3000 and
announced:

    "Game mode engaged but it did NOT free what it should have, sir - only
     0 MB of VRAM against a 3000 MB floor. Treat that as a failure, not a
     saving."

for a downshift that had worked perfectly: the next brain load was ~7.6 GB
instead of ~15 GB. Worse, `_st.result` was written once and never recomputed,
so every later `game_mode_status` repeated the failure for the rest of the
session. A feature that announces its own failure mid-match is a feature he
switches off — which would cost him the headroom the mode exists to buy.

The measurement conflated two different quantities:

    freed now     the UNLOAD. A real before/after delta.
    load avoided  the DOWNSHIFT. Not a delta at all — the load never happens.

So these tests pin, in both directions:
  * nothing resident + downshift took        -> "armed", never "failed"   (1)
  * nothing resident + downshift REFUSED     -> still "failed"            (2)
  * unreadable /api/ps                       -> "unverified", not "armed" (3)
  * a real unload hidden by a NEGATIVE card delta -> "ok", measured       (4)
  * big model still resident, nothing freed  -> still "failed"            (5)
  * the verdict is RE-DERIVED per report, never repeated from entry       (6)

Nothing here touches a real process, GPU, model or the disk. It reuses the
harness in tests/skills/test_game_mode.py (stdlib unittest, no pytest) rather
than copying it — a second copy of a stub set is how one of two copies rots.
"""
from __future__ import annotations

import unittest

from tests.skills.test_game_mode import _Base, BIG, SMALL, GAME

FLOOR = 3000            # GAME_MODE_MIN_VRAM_DELTA_MB in _Base.SETTINGS


def _seq(values):
    """A sampler with the harness's own semantics: pop until one is left, then
    repeat it. Lets a test drive before / after / and-then-later samples."""
    box = list(values)

    def _next():
        return box[0] if len(box) == 1 else box.pop(0)
    return _next


class TestNothingToFreeIsNotAFailure(_Base):
    """(1) Tonight's actual machine state: an empty card."""

    def _empty_card(self, gpu_samples):
        mod, actions = self._load()
        mod._nvidia_smi_mb = _seq(gpu_samples)
        # /api/ps is EMPTY and readable — a measurement, not an outage.
        mod._resident_models = lambda: []
        return mod, actions

    def test_engage_with_nothing_resident_does_not_report_failure(self):
        mod, _ = self._empty_card([(18000, 6000), (18000, 6000)])
        out = mod._enter(47036, GAME)
        self.assertEqual(mod._st.result, "armed", out)
        self.assertNotIn("did NOT free", out)
        self.assertNotIn("failure", out.lower())
        # It must still refuse to claim a saving it did not make.
        self.assertIn("nothing to reclaim", out)
        self.assertIn("load avoided", out.lower())
        # And the downshift it IS claiming has to be true.
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL)

    def test_a_negative_card_delta_is_not_a_failure_either(self):
        """The game allocated 900 MB during the 8 s verify window, so the card
        delta is NEGATIVE. Nothing was resident to free, so that is not a
        failure of ours — it is Fortnite's leak, which this mode never claimed
        to fix."""
        mod, _ = self._empty_card([(18000, 6000), (18900, 5100)])
        out = mod._enter(47036, GAME)
        self.assertEqual(mod._st.result, "armed", out)
        self.assertNotIn("did NOT free", out)

    def test_status_does_not_repeat_a_failure_for_the_whole_session(self):
        mod, actions = self._empty_card([(18000, 6000), (18900, 5100)])
        mod._enter(47036, GAME)
        status = actions["game_mode_status"]("")
        self.assertNotIn("failed", status.lower())
        self.assertNotIn("freed by measurement", status)
        self.assertIn("nothing resident to reclaim", status)
        self.assertIn(SMALL, status)

    def test_nothing_resident_and_no_downshift_is_still_a_failure(self):
        """(2) The other direction. If there was nothing to free AND we could
        not move him onto a smaller brain, the mode achieved literally nothing
        and must say so — 'armed' here would be the original defect inverted."""
        mod, _ = self._load(settings={"GAME_MODE_BRAIN": "nonexistent:70b"})
        mod._nvidia_smi_mb = _seq([(18000, 6000), (18000, 6000)])
        mod._resident_models = lambda: []
        out = mod._enter(47036, GAME)
        self.assertEqual(mod._st.result, "failed", out)
        self.assertIn("changed nothing", out)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], BIG)


class TestUnknownIsNeverZero(_Base):
    """(3) [] used to mean both 'nothing resident' and '/api/ps was down'."""

    def test_unreadable_ps_claims_nothing_in_either_direction(self):
        mod, actions = self._load()
        mod._nvidia_smi_mb = _seq([(18000, 6000), (18000, 6000)])
        mod._resident_models = lambda: None          # the endpoint is down
        out = mod._enter(47036, GAME)
        self.assertEqual(mod._st.result, "unverified", out)
        # Must not claim a saving...
        self.assertIn("no saving", out)
        # ...and must not claim the card was empty, which it never established.
        self.assertNotIn("nothing to reclaim", out)
        self.assertNotIn("did NOT free", out)
        self.assertIn("no verified memory saving",
                      actions["game_mode_status"](""))

    def test_the_probe_reports_unreadable_as_none_not_empty(self):
        """The distinction has to exist at the source, or _classify cannot make
        it. This is the one place the two cases are actually told apart, so it
        runs against a PRISTINE copy of the skill: the shared harness stubs
        _resident_models itself, and a test of a stub proves nothing."""
        self._load()                      # pins the fake monolith in sys.modules
        from tests._skill_harness import load_skill_isolated
        mod, _ = load_skill_isolated("game_mode")
        mod._nvidia_smi_mb = lambda: (18000, 6000)     # no real GPU read
        mod._free_ram_mb = lambda: 9000

        mod._ollama_json = lambda path, timeout_s=3.0: None      # endpoint down
        self.assertIsNone(mod._resident_models())
        self.assertIsNone(mod._resident_vram_mb(mod._measure()))

        mod._ollama_json = lambda path, timeout_s=3.0: {"models": []}  # empty
        self.assertEqual(mod._resident_models(), [])
        self.assertEqual(mod._resident_vram_mb(mod._measure()), 0)


class TestARealUnloadIsStillMeasured(_Base):
    """(4)+(5) The floor must keep catching the failure it was written for."""

    def test_a_real_unload_hidden_by_the_game_is_still_reported(self):
        """The 14.5 GB unload landed, but the game grabbed 15 GB in the same
        8 seconds, so the CARD delta reads -500 MB. The attributable /api/ps
        delta is the honest one and it is what gets reported."""
        mod, _ = self._load()
        mod._nvidia_smi_mb = _seq([(23269, 1307), (23769, 807)])
        samples = _seq([[{"name": BIG, "size_vram_mb": 14524}], []])
        mod._resident_models = samples
        out = mod._enter(47036, GAME)
        self.assertEqual(mod._st.result, "ok", out)
        self.assertIn("14524 MB of VRAM", out)
        self.assertIn("in resident models", out)  # names WHICH measurement
        self.assertEqual(self.unloaded, [BIG])

    def test_big_model_still_resident_is_still_a_failure(self):
        """(5) The regression guard on the guard: when there WAS 14.5 GB to
        free and it is all still sitting there, this must still be a failure."""
        mod, _ = self._load()
        mod._nvidia_smi_mb = _seq([(23269, 1307), (23000, 1576)])
        mod._resident_models = lambda: [{"name": BIG, "size_vram_mb": 14524}]
        out = mod._enter(47036, GAME)
        self.assertEqual(mod._st.result, "failed", out)
        self.assertIn("did NOT free", out)
        self.assertIn("14524 MB of model was resident", out)


class TestTheVerdictIsRederived(_Base):
    """(6) _st.result was the score of ONE 8-second window, repeated forever."""

    def test_a_transient_failure_does_not_outlive_the_condition(self):
        mod, actions = self._load()
        # t0 before: 14.5 GB of brain resident. t+8s: the unload has not shown
        # up yet -> a correct FAILURE verdict at that instant. Later, when the
        # owner asks, it is gone -> the report must follow the machine.
        mod._nvidia_smi_mb = _seq([(23269, 1307), (23300, 1276), (8745, 15831)])
        mod._resident_models = _seq([
            [{"name": BIG, "size_vram_mb": 14524}],
            [{"name": BIG, "size_vram_mb": 14524}],
            [],
        ])
        out = mod._enter(47036, GAME)
        self.assertEqual(mod._st.result, "failed", out)

        status = actions["game_mode_status"]("")
        self.assertEqual(mod._st.result, "ok", status)
        self.assertIn("14524 MB of VRAM freed by measurement", status)
        self.assertNotIn("FAILED", status)

    def test_status_reclassifies_through_the_same_code_path_as_verify(self):
        """Same _measure(), same _classify() — a second scoring rule would be
        two half-truths that disagree, which is worse than one stale one."""
        mod, actions = self._load()
        mod._nvidia_smi_mb = _seq([(18000, 6000), (18000, 6000)])
        mod._resident_models = lambda: []
        calls = []
        real = mod._classify
        mod._classify = lambda *a, **k: (calls.append(a[:1]), real(*a, **k))[1]
        mod._enter(47036, GAME)
        self.assertEqual(len(calls), 1)
        actions["game_mode_status"]("")
        self.assertEqual(len(calls), 2, "status did not re-classify")


if __name__ == "__main__":       # pragma: no cover
    unittest.main(verbosity=2)
