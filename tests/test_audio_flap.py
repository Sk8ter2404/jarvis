"""core/audio_flap.py — the audio-device announcement governor (2026-09-29).

Live 2026-09-29 15:20-15:29, owner away: a desk mic's Windows endpoint went
Active <-> NotPresent every ~20-40 s, Windows bounced the default recording
device between it and a powered-off headset's still-Active endpoint, and
JARVIS spoke 17 audio-device sentences in ~10 minutes. These tests pin the
governor's rules on a hand-driven clock (no sleeping, no devices):

  * a family that changes AUDIO_FLAP_THRESHOLD times inside AUDIO_FLAP_WINDOW_S
    is flapping: ONE plain sentence, then quiet until it has been steady for
    twice the window, which is said once;
  * at most one governed sentence per AUDIO_ANNOUNCE_MIN_GAP_S; a later one
    REPLACES a held one;
  * a hearing alert at most once per 10 minutes while nothing changes, never
    inside a storm, and its recovery only if the alert itself was said.

Device names are SYNTHETIC ("Desk Mic", "Wireless Headset").
"""
from __future__ import annotations

import ast
import os
import unittest

from core import audio_flap as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DESK = "endpoint:{0.0.1.00000000}.{desk}"
SWITCH_HS = "Switched to Wireless Headset, sir."
SWITCH_DESK = "Switched to the Desk Mic, sir."
DEAF = ("Sir, I may not be able to hear you. The 'Wireless Headset' headset "
        "measures powered off.")
CLEAR = ("Windows' default microphone is off the powered-off headset now, "
         "sir. I still cannot prove it is picking up sound until I hear you.")


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _gov(**kw):
    clock = _Clock()
    logs: list[str] = []
    g = F.AudioFlapGovernor(clock=clock, log=logs.append, **kw)
    return g, clock, logs


class FlapDetectionTests(unittest.TestCase):
    def test_two_changes_are_not_a_flap_the_third_is(self):
        g, c, logs = _gov()
        self.assertEqual(g.note_flip(DESK, "the Desk Mic"), [])
        c.t += 25
        self.assertEqual(g.note_flip(DESK, "the Desk Mic"), [])
        self.assertFalse(g.storm_active())
        c.t += 25
        out = g.note_flip(DESK, "the Desk Mic")
        self.assertEqual(len(out), 1, out)
        self.assertIn("the Desk Mic keeps dropping in and out", out[0])
        self.assertIn("stop announcing", out[0])
        self.assertTrue(g.storm_active())
        self.assertTrue(any("[audio-flap]" in ln and "FLAPPING" in ln
                            for ln in logs), logs)

    def test_changes_outside_the_window_do_not_count(self):
        g, c, _ = _gov(window_s=300.0)
        for _ in range(4):
            g.note_flip(DESK, "the Desk Mic")
            c.t += 200            # never 3 inside any 300 s window
        self.assertFalse(g.storm_active())

    def test_threshold_below_two_turns_detection_off(self):
        g, c, _ = _gov(threshold=0)
        for _ in range(10):
            self.assertEqual(g.note_flip(DESK, "the Desk Mic"), [])
            c.t += 1
        self.assertFalse(g.storm_active())

    def test_only_one_sentence_per_storm_even_across_families(self):
        g, c, logs = _gov()
        spoken = []
        for i in range(6):
            spoken += g.note_flip(DESK, "the Desk Mic")
            spoken += g.note_flip("default-in:a|b", "the default microphone",
                                  verb="keeps switching back and forth")
            c.t += 20
        self.assertEqual(len(spoken), 1, spoken)
        self.assertIn("Desk Mic", spoken[0])
        self.assertTrue(any("joins the flap storm" in ln for ln in logs), logs)

    def test_everything_governed_is_quiet_inside_a_storm(self):
        g, c, logs = _gov()
        for _ in range(3):
            g.note_flip(DESK, "the Desk Mic")
            c.t += 10
        c.t += 120                                     # gap long open
        for kind, msg in ((F.KIND_SWITCH, SWITCH_HS), (F.KIND_DEAF, DEAF),
                          (F.KIND_DEAF_CLEAR, CLEAR),
                          (F.KIND_ALERT, "Sir, your mic has been silent.")):
            self.assertEqual(g.submit(msg, kind), [], kind)
        self.assertTrue(any("quiet during the flap storm" in ln
                            for ln in logs), logs)
        # A kind the governor does not own is never swallowed.
        self.assertEqual(g.submit("headset battery is low", "other"),
                         ["headset battery is low"])

    def test_the_storm_ends_once_after_twice_the_window_of_calm(self):
        g, c, logs = _gov(window_s=300.0)
        for _ in range(3):
            g.note_flip(DESK, "the Desk Mic")
            c.t += 10
        last = c.t - 10
        c.t = last + 590               # 10 s short of 600 s of calm
        self.assertEqual(g.flush(), [])
        self.assertTrue(g.storm_active())
        c.t += 20
        out = g.flush()
        self.assertEqual(len(out), 1, out)
        self.assertIn("the Desk Mic has been steady for 10 minutes", out[0])
        self.assertFalse(g.storm_active())
        c.t += 100
        self.assertEqual(g.flush(), [], "said once, not every flush")
        self.assertTrue(any("stable for 10 minutes" in ln for ln in logs),
                        logs)
        # ...and device changes are announced again afterwards.
        self.assertEqual(g.submit(SWITCH_HS, F.KIND_SWITCH), [SWITCH_HS])

    def test_a_change_during_the_calm_restarts_the_settle_clock(self):
        g, c, _ = _gov(window_s=300.0)
        for _ in range(3):
            g.note_flip(DESK, "the Desk Mic")
            c.t += 10
        c.t += 500
        g.note_flip(DESK, "the Desk Mic")       # it dropped out again
        c.t += 500
        self.assertEqual(g.flush(), [])
        self.assertTrue(g.storm_active())
        c.t += 101
        self.assertEqual(len(g.flush()), 1)

    def test_hearing_alert_alternating_with_its_recovery_is_a_flap(self):
        g, c, logs = _gov()
        spoken = []
        for msg, kind in ((DEAF, F.KIND_DEAF), (CLEAR, F.KIND_DEAF_CLEAR),
                          (DEAF, F.KIND_DEAF), (CLEAR, F.KIND_DEAF_CLEAR)):
            spoken += g.submit(msg, kind)
            c.t += 70
        self.assertTrue(g.storm_active())
        self.assertTrue(any("the microphone keeps dropping in and out" in m
                            for m in spoken), spoken)


class OneSentencePerGapTests(unittest.TestCase):
    def test_second_sentence_inside_the_gap_is_held_then_released(self):
        g, c, _ = _gov(min_gap_s=60.0)
        self.assertEqual(g.submit(SWITCH_HS, F.KIND_SWITCH), [SWITCH_HS])
        c.t += 20
        self.assertEqual(g.submit(SWITCH_DESK, F.KIND_SWITCH), [])
        self.assertEqual(g.held_message(), SWITCH_DESK)
        c.t += 30
        self.assertEqual(g.flush(), [], "the gap is not open yet")
        c.t += 11
        self.assertEqual(g.flush(), [SWITCH_DESK])
        self.assertIsNone(g.held_message())

    def test_a_later_sentence_replaces_the_held_one(self):
        g, c, logs = _gov(min_gap_s=60.0)
        g.submit(SWITCH_HS, F.KIND_SWITCH)
        c.t += 10
        g.submit(SWITCH_DESK, F.KIND_SWITCH)
        c.t += 10
        g.submit("Switched to your speakers, sir.", F.KIND_SWITCH)
        c.t += 45
        self.assertEqual(g.flush(), ["Switched to your speakers, sir."])
        self.assertTrue(any("superseded" in ln for ln in logs), logs)

    def test_an_exact_repeat_inside_the_gap_is_dropped(self):
        # One press moving mic AND speakers enqueues the identical line twice.
        g, c, _ = _gov(min_gap_s=60.0)
        g.submit(SWITCH_HS, F.KIND_SWITCH)
        self.assertEqual(g.submit(SWITCH_HS, F.KIND_SWITCH), [])
        self.assertIsNone(g.held_message())
        c.t += 61
        self.assertEqual(g.flush(), [])

    def test_gap_zero_turns_the_limit_off(self):
        g, c, _ = _gov(min_gap_s=0.0)
        self.assertEqual(g.submit(SWITCH_HS, F.KIND_SWITCH), [SWITCH_HS])
        self.assertEqual(g.submit(SWITCH_DESK, F.KIND_SWITCH), [SWITCH_DESK])

    def test_the_storm_sentence_replaces_a_held_switch(self):
        g, c, _ = _gov(min_gap_s=60.0)
        g.submit(SWITCH_HS, F.KIND_SWITCH)
        c.t += 5
        g.submit(SWITCH_DESK, F.KIND_SWITCH)            # held
        for _ in range(3):
            g.note_flip(DESK, "the Desk Mic")
            c.t += 5
        c.t += 60
        out = g.flush()
        self.assertEqual(len(out), 1, out)
        self.assertIn("keeps dropping in and out", out[0])


class HearingAlertTests(unittest.TestCase):
    def test_at_most_once_per_ten_minutes_while_nothing_changes(self):
        g, c, logs = _gov()
        self.assertEqual(g.submit(DEAF, F.KIND_DEAF), [DEAF])
        c.t += 300                  # the daemon's own first repeat
        self.assertEqual(g.submit("Still no change, sir. " + DEAF,
                                  F.KIND_DEAF), [])
        c.t += 301                  # 601 s after the first
        self.assertEqual(len(g.submit("Still no change, sir. " + DEAF,
                                      F.KIND_DEAF)), 1)
        self.assertTrue(any("not repeating a hearing alert" in ln
                            for ln in logs), logs)

    def test_a_real_change_in_between_allows_it_again(self):
        g, c, _ = _gov()
        g.submit(DEAF, F.KIND_DEAF)
        c.t += 90
        self.assertEqual(g.submit(SWITCH_DESK, F.KIND_SWITCH), [SWITCH_DESK])
        c.t += 90
        self.assertEqual(g.submit(DEAF, F.KIND_DEAF), [DEAF])

    def test_recovery_of_an_unannounced_fault_is_not_news(self):
        g, _c, logs = _gov()
        self.assertEqual(g.submit(CLEAR, F.KIND_DEAF_CLEAR), [])
        self.assertTrue(any("never announced" in ln for ln in logs), logs)

    def test_a_held_alert_and_its_recovery_cancel_out(self):
        g, c, _ = _gov(min_gap_s=60.0)
        g.submit(SWITCH_HS, F.KIND_SWITCH)
        c.t += 3
        self.assertEqual(g.submit(DEAF, F.KIND_DEAF), [])      # held
        c.t += 20
        self.assertEqual(g.submit(CLEAR, F.KIND_DEAF_CLEAR), [])
        self.assertIsNone(g.held_message())
        c.t += 60
        self.assertEqual(g.flush(), [])

    def test_a_spoken_alert_gets_its_recovery(self):
        g, c, _ = _gov()
        g.submit(DEAF, F.KIND_DEAF)
        c.t += 61
        self.assertEqual(g.submit(CLEAR, F.KIND_DEAF_CLEAR), [CLEAR])


class IncidentReplayTests(unittest.TestCase):
    """The 2026-09-29 15:20-15:29 sequence, submitted the way the two writers
    submit it (monolith switch lines + daemon deaf / recovery lines + the
    endpoint flips the monolith notes). 17 sentences reached the speaker
    live; the governor must let through a handful at most, including exactly
    one plain flapping sentence."""

    def test_ten_minutes_of_flapping_is_a_handful_of_sentences(self):
        g, c, _ = _gov()
        spoken: list[str] = []
        present = True
        for cycle in range(20):                 # ~20 edges in ~10 minutes
            present = not present
            spoken += g.note_flip(DESK, "the Desk Mic")
            if present:
                spoken += g.submit(SWITCH_DESK, F.KIND_SWITCH)
                c.t += 2
                spoken += g.submit(CLEAR, F.KIND_DEAF_CLEAR)
            else:
                spoken += g.submit(SWITCH_HS, F.KIND_SWITCH)
                c.t += 3
                spoken += g.submit(DEAF, F.KIND_DEAF)
            c.t += 25
            spoken += g.flush()
        self.assertLessEqual(len(spoken), 3, spoken)
        flap = [m for m in spoken if "keeps dropping in and out" in m]
        self.assertEqual(len(flap), 1, spoken)
        self.assertIn("the Desk Mic", flap[0])


class ConfigureTests(unittest.TestCase):
    def test_bad_values_keep_the_current_setting(self):
        g, _c, _ = _gov()
        g.configure(window_s="nonsense", threshold="x", min_gap_s=None)
        self.assertEqual((g.window_s, g.threshold, g.min_gap_s),
                         (300.0, 3, 60.0))
        g.configure(window_s=-5, min_gap_s=-1)
        self.assertEqual(g.window_s, 300.0)
        self.assertEqual(g.min_gap_s, 0.0)
        g.configure(window_s=120, threshold=4.0, min_gap_s=30)
        self.assertEqual((g.window_s, g.threshold, g.min_gap_s),
                         (120.0, 4, 30.0))
        self.assertEqual(g.settle_s, 240.0)

    def test_reset_forgets_a_storm(self):
        g, c, _ = _gov()
        for _ in range(3):
            g.note_flip(DESK, "the Desk Mic")
        g.reset()
        self.assertFalse(g.storm_active())
        self.assertEqual(g.submit(SWITCH_HS, F.KIND_SWITCH), [SWITCH_HS])


class ModuleHygieneTests(unittest.TestCase):
    def test_stdlib_only(self):
        with open(F.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        self.assertLessEqual(mods, {"__future__", "threading", "time",
                                    "collections"})

    def test_defaults_match_config(self):
        # Read the SHIPPED literals statically: importing core.config would
        # overlay this box's data/user_settings.json.
        with open(os.path.join(_ROOT, "core", "config.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        lits = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value,
                                                           ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        lits[tgt.id] = node.value.value
        self.assertEqual(lits["AUDIO_FLAP_WINDOW_S"], F.DEFAULT_WINDOW_S)
        self.assertEqual(lits["AUDIO_FLAP_THRESHOLD"], F.DEFAULT_THRESHOLD)
        self.assertEqual(lits["AUDIO_ANNOUNCE_MIN_GAP_S"], F.DEFAULT_MIN_GAP_S)
        self.assertIsInstance(lits["AUDIO_FLAP_WINDOW_S"], float)
        self.assertIsInstance(lits["AUDIO_ANNOUNCE_MIN_GAP_S"], float)
        self.assertIsInstance(lits["AUDIO_REPICK_STABLE_S"], float)
        self.assertIs(type(lits["AUDIO_FLAP_THRESHOLD"]), int)


if __name__ == "__main__":
    unittest.main()
