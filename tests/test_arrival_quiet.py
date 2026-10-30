"""core/arrival_quiet.py: the quiet period after he arrives (v3.0.10).

Live 2026-10-29 07:21-07:23: the "while you were away" recap, the morning
briefing, an anticipation pattern offer and a stretch nudge inside 90 s. The
arrival briefing stamps note_arrival(); the stretch nudge and the offers wait
ARRIVAL_QUIET_MINUTES (core/config.py, read live; 0 = off).

    python tools/run_tests.py arrival_quiet
"""
from __future__ import annotations

import unittest
from unittest import mock

import core.arrival_quiet as aq
import core.config as cfg


class ArrivalQuietTests(unittest.TestCase):
    def setUp(self):
        for target, name, value in ((aq, "_arrived_at", [0.0]),
                                    (cfg, "ARRIVAL_QUIET_MINUTES", 10)):
            p = mock.patch.object(target, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)

    def test_no_arrival_no_quiet(self):
        self.assertFalse(aq.arrival_quiet_active(now=5_000.0))

    def test_quiet_for_the_minutes_after_an_arrival(self):
        aq.note_arrival(now=1_000.0)
        self.assertTrue(aq.arrival_quiet_active(now=1_045.0))     # +45 s
        self.assertTrue(aq.arrival_quiet_active(now=1_000.0 + 599))
        self.assertFalse(aq.arrival_quiet_active(now=1_000.0 + 600))

    def test_zero_minutes_is_off(self):
        cfg.ARRIVAL_QUIET_MINUTES = 0
        aq.note_arrival(now=1_000.0)
        self.assertFalse(aq.arrival_quiet_active(now=1_001.0))

    def test_a_bad_setting_is_off(self):
        cfg.ARRIVAL_QUIET_MINUTES = "soon"
        aq.note_arrival(now=1_000.0)
        self.assertFalse(aq.arrival_quiet_active(now=1_001.0))


if __name__ == "__main__":
    unittest.main()
