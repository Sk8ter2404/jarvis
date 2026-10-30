"""The speech-queue drain and a queued print-finish line (FINISH_LINE_KEEP,
v3.0.10).

Live 2026-10-28 22:26:44 (paraphrased): "[pending] dropped stale
announcement: Print complete, sir - '<print>' is finished ..." and two
seconds later the companion's "Shall I queue a cooldown timer for you, sir?"
was spoken alone. The drain asked `import skills.bambu_monitor`, a second,
never-started copy of the skill (the live one is
sys.modules["skill_bambu_monitor"]) whose state is empty, so every finish
line failed the check; the offer carried no link to it.

Drives the REAL _speak_pending with the voice faked (the presence-hold test
base) and the REAL bambu_monitor as the live skill. Synthetic lines only.

    python tools/run_tests.py monolith_finish_line_keep
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_presence_hold import _Base

HEAD = ("Print complete, sir — 'cube' is finished. "
        "Shall I notify you when the bed has cooled?")
TAIL = "Shall I queue a cooldown timer for you, sir?"


@requires_monolith
class FinishLineDrainTests(_Base):
    def setUp(self):
        super().setUp()
        # He is at the desk: no presence hold.
        self.input_age[0] = 5.0
        self._owner_spoke(10)
        from tests.skills.test_bambu_monitor import _load_bambu
        self.bm, _ = _load_bambu()
        p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": self.bm})
        p.start()
        self.addCleanup(p.stop)
        self.bm._finish_line_passed_at[0] = 0.0
        from core import config
        self.config = config

    def flag(self, value):
        self._p(self.config, "FINISH_LINE_KEEP", value, create=True)

    def printer(self, **kw):
        with self.bm._state_lock:
            self.bm._state.update(kw)

    def queue_both(self):
        self._write([
            dict(self._e(HEAD, "bambu"),
                 still_valid="bambu_finish:cube|job=7"),
            dict(self._e(TAIL, "print_companion"),
                 still_valid="bambu_finish_offer:"),
        ])

    def test_the_live_finish_and_its_offer_are_both_spoken(self):
        self.flag(True)
        # The printer still says FINISH but now names the file differently.
        self.printer(gcode_state="FINISH", filename="plate_1.gcode",
                     task_id="7")
        self.queue_both()
        _, log = self._drain()
        self.assertEqual(self.spoke, [HEAD, TAIL])
        self.assertNotIn("dropped stale", log)

    def test_a_new_job_drops_the_head_and_the_offer_with_it(self):
        self.flag(True)
        self.printer(gcode_state="RUNNING", filename="next.3mf", task_id="8")
        self.queue_both()
        _, log = self._drain()
        self.assertEqual(self.spoke, [], "the offer was spoken on its own")
        self.assertIn("finish line stale (a new job is under way)", log)
        self.assertIn("finish offer dropped", log)

    def test_no_live_skill_keeps_the_line(self):
        self.flag(True)
        self._write([dict(self._e(HEAD, "bambu"),
                          still_valid="bambu_finish:cube")])
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": None}):
            self._drain()
        self.assertEqual(self.spoke, [HEAD])

    def test_flag_off_asks_the_second_copy_and_drops_it(self):
        # The old path exactly: the empty second copy says "not FINISH".
        self.flag(False)
        self.printer(gcode_state="FINISH", filename="cube.3mf")
        self._write([dict(self._e(HEAD, "bambu"),
                          still_valid="bambu_finish:cube")])
        _, log = self._drain()
        self.assertEqual(self.spoke, [])
        self.assertIn("dropped stale announcement", log)


if __name__ == "__main__":
    unittest.main()
