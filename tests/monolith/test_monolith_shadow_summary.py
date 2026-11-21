"""The [shadow-summary] line (n20 shadow F5, 2026-11-21; SHADOW_SUMMARY).

The shadow features log only when they WOULD act, so "never consulted" and
"consulted, no disagreement" read the same. Pinned here: one numbers-only
line for the features whose mode is not 'off', owner turns counted at the
"You:" flush, real hooks feeding the counters, the flag off = no line.

Run (full-deps tier):
    python -m unittest tests.monolith.test_monolith_shadow_summary
"""
from __future__ import annotations

import contextlib
import io
import re
import threading
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

# Labels and digits only: never a word the owner said.
_NUMBERS_ONLY = re.compile(r"^\[shadow-summary\] [a-z0-9 ]+: [a-z0-9 ;\-]+$")


@requires_monolith
class ShadowSummaryTests(MonolithGlobalsTestCase):

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_shadow_tally", {})
        self._p(bc, "_eot_counts", {"checks": 7, "slow": 1, "max_ms": 80,
                                    "grace_skip": 3, "captures": 2,
                                    "would_fire": 1})
        self._p(bc, "_eot_state", {"ready": True, "off": "", "logged": False})
        self._p(bc, "SHADOW_SUMMARY", True)
        self._p(bc, "SMART_TURN_MODE", "shadow")
        for fn in ("_action_fit_mode", "_loop_guard_mode", "_head_hold_mode",
                   "_learn_prefilter_mode"):
            self._p(bc, fn, lambda: "shadow")
        self._p(bc, "_instant_actions_mode", lambda: "off")

    def _p(self, *a, **k):
        patcher = mock.patch.object(*a, **k)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def test_line_is_numbers_and_labels_only(self):
        bc = self.bc
        bc._shadow_count("action-fit", "checked", 4)
        bc._shadow_count("action-fit", "would")
        line = bc._shadow_summary_line("30 min")
        self.assertRegex(line, _NUMBERS_ONLY)
        self.assertIn("turns 0", line)          # 0 turns reads as 0 turns
        self.assertIn("smart-turn checks 7 slow 1 grace-skip 3 captures 2 "
                      "would-fire 1 latched 0", line)
        self.assertIn("action-fit checked 4 would-block 1", line)
        self.assertIn("loop-guard chains 0 would-stop 0", line)
        self.assertIn("head-hold plans 0 held 0", line)
        self.assertIn("learn-prefilter judged 0 would-skip 0", line)
        self.assertNotIn("instant", line)       # mode 'off'

    def test_off_features_are_left_out_and_latch_shows(self):
        bc = self.bc
        self._p(bc, "_action_fit_mode", lambda: "off")
        self._p(bc, "_instant_actions_mode", lambda: "shadow")
        bc._eot_state["off"] = "too slow"
        line = bc._shadow_summary_line()
        self.assertNotIn("action-fit", line)
        self.assertIn("latched 1", line)
        self.assertIn("instant matched 0 would-run 0", line)
        self._p(bc, "SMART_TURN_MODE", "off")
        self.assertNotIn("smart-turn", bc._shadow_summary_line())

    def test_flag_off_prints_nothing(self):
        bc = self.bc
        self._p(bc, "SHADOW_SUMMARY", False)
        self.assertIsNone(bc._shadow_summary_line())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._shadow_summary_emit()
        self.assertEqual(out.getvalue(), "")
        self.assertIsNone(bc._shadow_summary_start())

    def test_owner_turn_flush_counts_a_turn(self):
        bc = self.bc
        self._p(bc, "_eot_shadow_pending", [None])
        with contextlib.redirect_stdout(io.StringIO()):
            bc._eot_shadow_flush()
            bc._eot_shadow_flush()
        self.assertIn("turns 2", bc._shadow_summary_line())

    def test_the_real_ambient_prefilter_hook_counts(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(bc._ambient_owner_fragment_skip("ok"))  # shadow
        self.assertEqual(bc._shadow_get("learn-prefilter", "judged"), 1)
        self.assertEqual(bc._shadow_get("learn-prefilter", "would"), 1)

    def test_periodic_loop_emits_until_stopped(self):
        bc = self.bc
        stop = threading.Event()
        seen = []
        calls = {"n": 0}

        def fake_wait(_s):
            calls["n"] += 1
            return calls["n"] > 2

        stop.wait = fake_wait
        self._p(bc, "_shadow_summary_emit", lambda label: seen.append(label))
        bc._shadow_summary_loop(1800.0, stop)
        self.assertEqual(seen, ["30 min", "30 min"])


if __name__ == "__main__":
    unittest.main()
