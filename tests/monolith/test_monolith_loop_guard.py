"""LOOP_GUARD_MODE in the follow-up loop (v3 W21, core/loop_guard.py).

A chain that bounces between two look-ups (list_windows / see_screen, every
result a little different, so none of the older repeat guards fires) is
logged in 'shadow' (the default), cut in 'on' with the usual close-out line,
and untouched in 'off'.
"""
from __future__ import annotations

import contextlib
import io

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

_BOUNCE = [
    "[ACTION: see_screen, the desktop]",
    "[ACTION: list_windows]",
    "[ACTION: see_screen, the taskbar]",
    "[ACTION: list_windows]",
    "[ACTION: see_screen, the desktop again]",
    "[ACTION: list_windows]",
]


@requires_monolith
class LoopGuardChainTests(_Base):
    def setUp(self):
        super().setUp()
        self._stub("see_screen", "Screen one.", "Screen two.", "Screen three.",
                   "Screen four.")
        self._stub("list_windows", "Windows: A.", "Windows: A, B.",
                   "Windows: A, B, C.", "Windows: A, B, C, D.")

    def _run(self, mode, followups=_BOUNCE):
        bc = self.bc
        self._p(bc, "LOOP_GUARD_MODE", mode)
        self._p(bc, "get_response_with_animation",
                return_value="[ACTION: list_windows]")
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=list(followups) + [None] * 8)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bc._run_llm_dispatch("what is on my screen")
        return buf.getvalue()

    def test_shadow_logs_once_and_changes_nothing(self):
        printed = self._run("shadow")
        self.assertEqual(printed.count("[loop-guard] would stop"), 1)
        self.assertIn("(ping-pong list_windows/see_screen)", printed)
        self.assertNotIn("[loop-guard] stopping", printed)
        self.assertNotIn("[close-out]", printed)
        # the chain ran exactly as before: every bounce was made
        self.assertEqual(len(self.calls["see_screen"]), 3)
        self.assertEqual(len(self.calls["list_windows"]), 4)

    def test_on_stops_the_chain_and_says_so(self):
        bc = self.bc
        printed = self._run("on")
        self.assertIn("[loop-guard] stopping the chain "
                      "(ping-pong list_windows/see_screen)", printed)
        self.assertIn("[close-out] the chain stopped (ping-pong)", printed)
        self.assertEqual(len(self.calls["see_screen"]), 2)
        self.assertEqual(len(self.calls["list_windows"]), 2)
        self.assertEqual(self.gfr.call_count, 3)
        self.assertEqual(self.spoken.count(bc._CLOSE_OUT_GENERIC), 1)
        self.assertEqual(self.spoken[-1], bc._CLOSE_OUT_GENERIC)

    def test_off_is_silent_and_unchanged(self):
        printed = self._run("off")
        self.assertNotIn("[loop-guard]", printed)
        self.assertEqual(len(self.calls["see_screen"]), 3)
        self.assertEqual(len(self.calls["list_windows"]), 4)

    def test_an_unknown_mode_reads_as_off(self):
        printed = self._run("maybe")
        self.assertNotIn("[loop-guard]", printed)
        self.assertEqual(self.bc._loop_guard_mode(), "off")

    def test_a_chain_of_distinct_steps_is_not_judged(self):
        printed = self._run("on", ["[ACTION: see_screen, the taskbar]",
                                   "Everything looks fine, sir."])
        self.assertNotIn("[loop-guard]", printed)
        self.assertNotIn("[close-out]", printed)
        self.assertEqual(self.spoken[-1], "Everything looks fine, sir.")

    def test_form_steps_alternating_are_not_a_loop(self):
        self._stub("type", "typed one", "typed two", "typed three")
        bc = self.bc
        self._p(bc, "LOOP_GUARD_MODE", "on")
        self._p(bc, "get_response_with_animation",
                return_value="[ACTION: see_screen, form]")
        self._p(bc, "get_followup_response",
                side_effect=["[ACTION: click, Name]",
                             "[ACTION: type, one]",
                             "[ACTION: click, Email]",
                             "[ACTION: type, two]",
                             "Your form is filled in, sir."] + [None] * 4)
        self._stub("click", "clicked: Name", "clicked: Email")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bc._run_llm_dispatch("fill in the form")
        self.assertNotIn("[loop-guard]", buf.getvalue())

    def test_close_out_line_speaks_after_substance_for_the_loop_cut(self):
        bc = self.bc
        self.assertEqual(
            bc._chain_close_out_line(cut="ping-pong", rounds=2,
                                     spoke_substance=True, barged=False),
            bc._CLOSE_OUT_GENERIC)
        self.assertEqual(
            bc._chain_close_out_line(cut="ping-pong", rounds=2,
                                     spoke_substance=True, barged=True), "")

    def test_the_setting_is_declared_and_logged_per_boot(self):
        from core import config
        self.assertEqual(config.LOOP_GUARD_MODE, "shadow")
        self.assertIn("LOOP_GUARD_MODE", self.bc._TURN_FLAG_KEYS)

    def test_outside_a_turn_nothing_is_judged(self):
        from core.loop_guard import LoopGuard
        g = LoopGuard()
        self.assertEqual(self.bc._loop_guard_feed(g, 0), (None, 0))
        self.assertEqual(len(g), 0)
