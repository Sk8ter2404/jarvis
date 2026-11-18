"""The 2026-11-17 overhaul of JARVIS's unprompted lines, through the REAL
monolith paths (JARVIS-Research/review_20261117_overhaul_audit.json, the
proactive cluster).

  * proactive-10: an idle remark repeated after a restart ("...the first
    computer mouse was made of wood." at 09:28 and 12:30 on 11-16, the repeat
    ring lived in memory only), opened with "Good morning, sir." /
    "Certainly, sir.", and four persona-copy drops in a row each cost a
    brain call five minutes after the last.
  * proactive-08: coming back at night (11-16 23:49:34-23:50:29) meant the
    evening briefing, a recap naming a 6 PM forecast and two false mic
    alerts "plus 3 more" (two "Yes, listening, sir."), the daily recap
    repeating the briefing's count and asking a question nothing answers -
    and the budget's deferral was spoken the same second.

Every brain call, voice and clock is faked; files live in temp dirs.
Synthetic remarks only.

    python tools/run_tests.py monolith.test_monolith_proactive_lines_1117
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests.monolith.test_monolith_presence_hold import _Base as _QueueBase
from tests.monolith.test_monolith_proactive_presence import _GateBase

FACT = "Interestingly, the first computer mouse was made of wood."


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _quiet(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()


# ── proactive-10: openers, repeats across a restart ───────────────────────
class RemarkAcrossRestartTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = os.path.join(d.name, "proactive_recent.json")
        self._p(bc, "_proactive_recent_path", return_value=self.path)
        self._p(bc, "_system_prompt", "SYNTHETIC BASE PROMPT")
        self.llm = self._p(bc, "_llm_quick")
        # _do_proactive_turn's voice / animation / action parser.
        for name in ("pause_face_tracking", "resume_face_tracking",
                     "set_state", "_thinking_loop", "_reset_see_screen_budget"):
            self._p(bc, name)
        self._p(bc, "parse_and_run_actions", side_effect=lambda t: (t, []))
        self._p(bc, "_apply_quip_layer", side_effect=lambda s, r: s)
        n = len(bc.conversation_history)
        self.addCleanup(lambda: bc.conversation_history.__delitem__(
            slice(n, None)))
        self.speak = self._p(bc, "_speak")

    def _turn(self, reply):
        self.llm.return_value = reply
        return self._quiet(self.bc._do_proactive_turn, {})[1]

    def _restart(self):
        """What a new process has: an empty in-memory ring, then the boot
        restore (main() calls _proactive_recent_restore)."""
        self.bc._proactive_recent[:] = []
        self.bc._proactive_saved[:] = []
        self.bc._proactive_remarks_at[:] = []
        return self._quiet(self.bc._proactive_recent_restore)

    def test_live_greeting_is_cut_off_the_fact(self):
        log = self._turn(f"[intent:briefing] Good morning, sir. {FACT}")
        self.speak.assert_called_once()
        said = self.speak.call_args.args[0]
        self.assertNotIn("Good morning", said)
        self.assertIn(FACT, said)
        self.assertIn("[proactive] trimmed the opener 'Good morning, sir.'", log)

    def test_live_repeat_after_a_restart_is_not_spoken(self):
        self._turn(f"Good morning, sir. {FACT}")
        self.assertEqual(self.speak.call_count, 1)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual([r["text"] for r in json.load(f)], [FACT])
        (kept, log) = self._restart()
        self.assertEqual(kept, 1)
        self.assertIn("remembered 1 recent remark", log)
        log = self._turn(f"Certainly, sir. {FACT}")
        self.assertEqual(self.speak.call_count, 1, "repeated after a restart")
        self.assertIn("[proactive] dropped (repeats a recent remark)", log)
        # ...and the model is shown what the earlier run said.
        self.assertIn(FACT, self.llm.call_args.kwargs["user"])

    def test_an_acknowledgement_alone_is_dropped(self):
        log = self._turn("Certainly, sir.")
        self.speak.assert_not_called()
        self.assertIn("[proactive] dropped (only 'Certainly, sir.')", log)
        self.assertEqual(self.bc._proactive_drop_streak[0], 1)

    def test_a_week_old_remark_may_come_round_again(self):
        old = time.time() - self.bc.PROACTIVE_RECENT_KEEP_S - 60
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([{"t": old, "text": FACT},
                       {"t": "bad", "text": "x"}, "junk"], f)
        kept, _ = self._restart()
        self.assertEqual(kept, 0)
        self._turn(FACT)
        self.assertEqual(self.speak.call_count, 1)

    def test_a_corrupt_file_keeps_nothing_and_never_raises(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        kept, log = self._restart()
        self.assertEqual(kept, 0)
        self.assertIn("could not read the recent remarks", log)

    def test_the_saved_ring_is_bounded(self):
        n = int(self.bc.PROACTIVE_RECENT_KEEP_N)
        for i in range(n + 4):
            self._turn(f"Remark number {i} about topic {i * 7919}, sir.")
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)), n)

    def test_keep_zero_is_memory_only(self):
        self._p(self.bc, "PROACTIVE_RECENT_KEEP_S", 0)
        self._turn(FACT)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self._restart()[0], 0)

    def test_the_boot_restores_the_ring(self):
        import inspect
        src = inspect.getsource(self.bc.main)
        self.assertIn("_proactive_recent_restore()", src)


class DropStreakBackoffTests(_GateBase):
    """Live 11-16 11:25:56, 11:33:10, 12:19:47, 12:25:35: "[proactive]
    dropped (copies the persona example 'One moment, sir.')", each a
    ~12k-token brain call five minutes after the last."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "PROACTIVE_OWNER_VOICE_WINDOW_S", 100_000)
        self._owner_spoke(ago=60.0)
        self._p(bc, "_system_prompt", "SYNTHETIC BASE PROMPT")
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self._p(bc, "_proactive_recent_path",
                return_value=os.path.join(d.name, "proactive_recent.json"))
        self.llm = self._p(bc, "_llm_quick", return_value="One moment, sir.")
        self.speak = self._fake_turn_io()

    def _attempt(self):
        return self._quiet(self.bc._do_proactive_turn, {})[1]

    def _later(self, s):
        self.clock[0] += s
        self._silence(1000)

    def test_two_copies_in_a_row_wait_an_hour(self):
        self.assertIn("copies the persona example", self._attempt())
        self._later(301)
        self.assertTrue(self._check()[0], "one drop keeps the 5 min gap")
        log = self._attempt()
        self.assertIn("2 in a row, next try in 60 min", log)
        self._later(301)
        fire, log = self._check()
        self.assertFalse(fire, "the brain was asked again 5 min later")
        self.assertIn("gap 3600 s", log)
        self._later(3300)
        self.assertTrue(self._check()[0])
        self.speak.assert_not_called()

    def test_a_spoken_remark_ends_the_streak(self):
        self._attempt()
        self._later(301)
        self._attempt()
        self.llm.return_value = FACT
        self._later(3601)
        self._attempt()
        self.speak.assert_called_once()
        self.assertEqual(self.bc._proactive_drop_streak[0], 0)

    def test_streak_zero_is_the_old_pacing(self):
        self._p(self.bc, "PROACTIVE_DROP_STREAK", 0)
        self._attempt()
        self._later(301)
        self._attempt()
        self._later(301)
        self.assertTrue(self._check()[0])


# ── proactive-08: the night return ─────────────────────────────────────────
H = 3600.0
FORECAST = ("Sir, the temperature drops about 10 degrees by 6 PM - you may "
            "want a jacket.")
MIC_SILENT = ("Sir, the desk mic has been completely silent for 30 seconds - "
              "I can hear nothing at all from it.")
MIC_DOWN = "Sir, the microphone appears to be down. I'll queue a fix."
WAKE_ACK = "Yes, listening, sir."
PULSE = "GPU pinned at 100 percent, sir. CPU 34 percent, memory 97 percent."
CREDITS = "Heads up, sir - the credit balance is down to 7.10 dollars."
EVENING = ("[intent:briefing] Good evening, sir. 21 voice interactions "
           "logged today.")


class NightReturnTests(_QueueBase):
    def setUp(self):
        super().setUp()
        self._owner_spoke(30)                # "Insane Jarvis." at 23:49:05
        self._cell("_presence_away_seen", True)

    def test_live_night_return_says_only_what_is_still_true(self):
        self._write([self._e(FORECAST, "weather", 5.8 * H)]
                    + [self._e(MIC_SILENT, "audio", 3 * H) for _ in range(4)]
                    + [self._e(MIC_DOWN, "self_diagnostic", 3 * H),
                       self._e(WAKE_ACK, "ambient-listen", 2 * H),
                       self._e(WAKE_ACK, "ambient-listen", 2 * H),
                       self._e(PULSE, "pulse", 2 * H),
                       self._e(CREDITS, "disk_budget_watchdog", 1 * H),
                       self._e(EVENING, "evening", 1.8 * H)])
        _, log = self._drain()
        said = " | ".join(self.spoke)
        self.assertEqual(self.spoke[0], EVENING)          # the greeting first
        recap = self.spoke[1]
        self.assertTrue(recap.startswith("While you were away, sir: "), recap)
        self.assertIn("GPU pinned at 100 percent", recap)
        self.assertIn("7.10 dollars", recap)
        for stale in ("temperature", "silent", "appears to be down",
                      "listening", "more"):
            self.assertNotIn(stale, said)
        self.assertEqual(len(self.spoke), 2)
        self.assertIn("too old to say (weather; weather:", log)
        self.assertIn("too old to say (audio; mic_health:", log)
        self.assertIn("too old to say (ambient-listen; wake_ack:", log)

    def test_a_long_recap_offers_the_rest_and_records_them(self):
        self._keep_history()
        # His turn came first (an assistant line cannot lead the history).
        self.bc.conversation_history.append({"role": "user",
                                             "content": "Jarvis, I'm back."})
        self._write([self._e(f"Notice {n} came in, sir.", f"src{n}", 2 * H)
                     for n in range(5)])
        self._drain()
        self.assertEqual(len(self.spoke), 1)
        self.assertTrue(self.spoke[0].endswith(
            "And two more - shall I list them?"), self.spoke[0])
        last = self.bc.conversation_history[-1]
        self.assertEqual(last["role"], "assistant")
        self.assertIn("Notice 3 came in", last["content"])
        self.assertIn("Notice 4 came in", last["content"])

    def _keep_history(self):
        bc = self.bc
        n = len(bc.conversation_history)
        self.addCleanup(lambda: bc.conversation_history.__delitem__(
            slice(n, None)))


class DeferGapTests(_QueueBase):
    """Live 23:50:09 "drain budget (25s) reached - deferring 2" and the
    recap was spoken in the same second."""

    def setUp(self):
        super().setUp()
        self._owner_spoke(30)
        self._p(self.bc, "_PENDING_DRAIN_BUDGET_S", 0.0)   # one line a pass
        self.bc._last_owner_turn_at[0] = self.mono[0] - 30

    def _three(self):
        self._write([self._e(f"Notice {n}, sir.", f"src{n}") for n in "abc"])

    def test_the_deferred_tail_waits_a_moment(self):
        self._three()
        _, log = self._drain()
        self.assertEqual(self.spoke, ["Notice a, sir."])
        self.assertIn("deferring 2 announcement(s)", log)
        _, log = self._drain()                      # the same second
        self.assertEqual(self.spoke, ["Notice a, sir."])
        self.assertIn("holding queued speech: after the drain budget (20 s "
                      "left, or until he speaks)", log)
        self.mono[0] += self.bc.PENDING_DEFER_GAP_S + 1
        self._drain()
        self.assertEqual(self.spoke, ["Notice a, sir.", "Notice b, sir."])

    def test_his_turn_ends_the_gap(self):
        self._three()
        self._drain()
        self.mono[0] += 3
        self.bc._last_owner_turn_at[0] = self.mono[0]   # he spoke
        self._drain()
        self.assertEqual(self.spoke, ["Notice a, sir.", "Notice b, sir."])

    def test_his_timer_still_speaks_in_the_gap(self):
        self._three()
        self._drain()
        q = self._read()
        q.append(self._e("Reminder, sir — tea", "timer"))
        self._write(q)
        self._drain()
        self.assertEqual(self.spoke, ["Notice a, sir.", "Reminder, sir — tea"])

    def test_gap_zero_is_the_old_drain(self):
        self._p(self.bc, "PENDING_DEFER_GAP_S", 0)
        self._three()
        self._drain()
        self._drain()
        self.assertEqual(self.spoke, ["Notice a, sir.", "Notice b, sir."])


if __name__ == "__main__":
    unittest.main()
