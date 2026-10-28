"""Monolith wiring for the remark-after-a-reading guard (v3.0.2).

THE LIVE INCIDENT (v3.0.0 session log, 2026-10-27 20:28:45). JARVIS had just
answered "what version are you running?" with version_info (20:28:32). The
owner said "Yay, version 3 is done." The brain answered without an action
("Splendid news, sir! I'm on version 3.0.0, last updated just now."), the
preemptive-hallucination injector matched 'state the version from memory'
and appended [ACTION: version_info], so the version was read out AGAIN.

Fix: _preempt_recent_reading. A reading in the same grounding group that ran
within _PREEMPT_REMARK_WINDOW_S grounds the claim when the owner's line is a
remark (core.action_fit.is_remark). An ask ("check again", "what version
again?") is never a remark, so the injector still runs the action.

Synthetic context only; the transcript is the one generic live line.

    python -m unittest tests.monolith.test_monolith_remark_rerun
"""
from __future__ import annotations

import time
import unittest

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

LIVE = "Yay, version 3 is done."
REPLY = ("[intent:proud] Splendid news, sir! I'm on version 3.0.0, last "
         "updated just now.")


@requires_monolith
class RecentReadingTests(MonolithGlobalsTestCase):
    def _hist(self, action, age_s, now):
        return [{"action": action, "arg": "", "result": "ok",
                 "at": now - age_s}]

    def test_the_live_remark_is_grounded_by_the_recent_reading(self):
        now = time.time()
        out = self.bc._preempt_recent_reading(
            "version_info", LIVE, now=now,
            history=self._hist("version_info", 17.0, now))
        self.assertIsNotNone(out)
        self.assertEqual(out[0], "version_info")
        self.assertAlmostEqual(out[1], 17.0, places=3)

    def test_a_grounding_group_member_counts(self):
        now = time.time()
        out = self.bc._preempt_recent_reading(
            "version_info", LIVE, now=now,
            history=self._hist("what_version", 5.0, now))
        self.assertEqual(out[0], "what_version")

    def test_asks_are_never_grounded_this_way(self):
        now = time.time()
        hist = self._hist("version_info", 10.0, now)
        for text in ("Check again.", "What version again?",
                     "Jarvis, what version are you running?",
                     "Is it still version 3?", "I need the version again"):
            self.assertIsNone(self.bc._preempt_recent_reading(
                "version_info", text, now=now, history=hist), text)

    def test_window_and_group_bounds(self):
        now = time.time()
        bc = self.bc
        self.assertIsNone(bc._preempt_recent_reading(
            "version_info", LIVE, now=now,
            history=self._hist("version_info",
                               bc._PREEMPT_REMARK_WINDOW_S + 1.0, now)))
        self.assertIsNone(bc._preempt_recent_reading(
            "version_info", LIVE, now=now,
            history=self._hist("get_time", 5.0, now)))
        self.assertIsNone(bc._preempt_recent_reading(
            "version_info", LIVE, now=now, history=[]))
        # a reading from the FUTURE (clock skew) never grounds anything
        self.assertIsNone(bc._preempt_recent_reading(
            "version_info", LIVE, now=now,
            history=self._hist("version_info", -5.0, now)))

    def test_a_forecast_follow_up_is_never_grounded(self):
        # v3.0.2 review: "Cool. And tomorrow." is remark-shaped but asks a
        # different forecast; an invented forecast must still be injected.
        now = time.time()
        bc = self.bc
        for name in ("weather_briefing", "weather_forecast"):
            self.assertIsNone(bc._preempt_recent_reading(
                name, "Cool. And tomorrow.", now=now,
                history=self._hist("weather_briefing", 10.0, now)), name)

    def test_bad_input_never_raises(self):
        bc = self.bc
        self.assertIsNone(bc._preempt_recent_reading("", LIVE))
        self.assertIsNone(bc._preempt_recent_reading("version_info", ""))
        self.assertIsNone(bc._preempt_recent_reading(
            "version_info", LIVE, now=time.time(),
            history=[{"action": None, "at": "x"}, "junk"]))

    def test_the_injector_still_fires_on_a_first_statement(self):
        # The detector itself is untouched: with nothing grounding it, the
        # live reply still maps to a version_info injection.
        out = self.bc._detect_preemptive_hallucination(REPLY, user_text=LIVE)
        self.assertIsNotNone(out)
        self.assertEqual(out[0], "inject")
        self.assertEqual(out[1], "version_info")

    def test_grounded_callback_skips_the_injection(self):
        # With is_grounded answering True for version_info (what the live
        # wrapper does after _preempt_recent_reading finds the reading), no
        # injection is made for the live reply.
        out = self.bc._detect_preemptive_hallucination(
            REPLY, is_grounded=lambda n: n == "version_info", user_text=LIVE)
        self.assertIsNone(out)

    def test_wiring_text(self):
        import inspect
        src = inspect.getsource(self.bc)
        i = src.index("def _preempt_is_grounded(action_name: str) -> bool:")
        body = src[i:i + 1200]
        self.assertIn("_preempt_recent_reading(action_name, _turn_user_text())",
                      body)
        self.assertIn("only remarked on it", body)


if __name__ == "__main__":
    unittest.main()
