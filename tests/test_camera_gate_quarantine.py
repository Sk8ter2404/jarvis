"""core/camera_gate.py - post-cool-down PROBATION and CULPRIT QUARANTINE.

WHAT THE 2026-09-29 LIVE LOG SHOWED (real device names redacted). The first
USB-storm trip at 18:38:30 was right. Its cool-down ended at 18:48:34. At
18:48:52.8 the face tracker opened the left webcam; the hub reset at 53.6 and
again at 56.1; at 18:48:58 the right webcam "opened but produced no frame ...
gone from the device list"; the left webcam went dead after 60 failed reads
and was reopened at 18:49:25.55 - and the hub reset 0.04 s later. NO second
trip happened. A read-only USB test the same day showed the left webcam's
plain stream starts reset the hub 8 of 10 times, the right one's 0 of 10.

Every test here runs the gate on a frozen clock, at the SHIPPED defaults
unless it says otherwise. Device keys, labels and app names are synthetic
(public repo).
"""
from __future__ import annotations

import inspect
import unittest

from core import camera_gate as cg

LEFT = "name:synth-left"
RIGHT = "name:synth-right"
AUDIO = "audio:output"
_LABELS = {LEFT: "the left webcam", RIGHT: "the right webcam"}


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s

    def at(self, hms: str) -> None:
        """Jump to a wall-clock time of the live session ("18:48:52.8")."""
        h, m, s = hms.split(":")
        t = int(h) * 3600 + int(m) * 60 + float(s)
        assert t >= self.t, f"the clock only moves forward ({hms})"
        self.t = t


def _gate(clock, present=None, **kw):
    logs: list = []
    spoken: list = []
    kw.setdefault("log", logs.append)
    kw.setdefault("announce", spoken.append)
    # Only when the gate takes one: on a tree WITHOUT this feature the
    # replay must still run and fail on what the gate DOES, not on a
    # TypeError (that is how the fix is proven against the old code).
    if "labeler" in inspect.signature(cg.CameraGate).parameters:
        kw.setdefault("labeler", lambda key: _LABELS.get(key, ""))
    if present is not None:
        kw.setdefault("presence", lambda key: present.get(key, True))
    g = cg.CameraGate(clock=clock, **kw)
    return g, logs, spoken


def _open_ok(g, clk, key, comp="face-track", took=0.8):
    d = g.begin(key, comp)
    assert d.allowed, (key, d)
    clk.advance(took)
    g.end(key, comp, True)


def _trip(g, clk):
    """A plain two-camera storm (both devices streamed first)."""
    g.note_drop(LEFT, "face-track")
    g.note_drop(RIGHT, "face-track")
    assert g.storm_active()


def _end_cool_down(g, clk):
    clk.advance(g.snapshot()["storm_remaining_s"] + 0.5)
    assert not g.storm_active()


def _streaming_pair(g, clk):
    _open_ok(g, clk, LEFT)
    clk.advance(5.0)
    _open_ok(g, clk, RIGHT)
    clk.advance(120.0)
    for k in (LEFT, RIGHT):
        g.note_frame(k)


class ProbationTests(unittest.TestCase):
    """After a cool-down, ONE drop is a storm again."""

    def test_one_drop_right_after_a_cool_down_re_trips_it_doubled(self):
        clk = _Clock()
        g, logs, spoken = _gate(clk)
        _streaming_pair(g, clk)
        _trip(g, clk)
        _end_cool_down(g, clk)
        clk.advance(18.0)
        _open_ok(g, clk, LEFT)
        clk.advance(2.0)
        self.assertTrue(g.note_drop(LEFT, "face-track"),
                        "a single drop 20 s after the cool-down did not trip")
        self.assertEqual(round(g.snapshot()["storm_remaining_s"]), 1200)
        trip2 = [ln for ln in logs if "[usb-storm]" in ln and "trip #2" in ln]
        self.assertEqual(len(trip2), 1, logs)
        self.assertIn("probation", trip2[0])
        self.assertIn("doubled", trip2[0])
        self.assertEqual(len(spoken), 1, "a repeat inside the chain is not "
                                         "spoken again")

    def test_after_the_probation_one_drop_is_one_drop_again(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _streaming_pair(g, clk)
        _trip(g, clk)
        _end_cool_down(g, clk)
        clk.advance(cg.STORM_PROBATION_S + 1.0)
        self.assertFalse(g.note_drop(LEFT, "face-track"))
        self.assertFalse(g.storm_active())

    def test_an_audio_endpoint_vanishing_counts_during_probation(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _streaming_pair(g, clk)
        _trip(g, clk)
        _end_cool_down(g, clk)
        clk.advance(30.0)
        self.assertTrue(g.note_drop(AUDIO, "audio", cg.KIND_AUDIO))

    def test_a_failed_open_of_a_vanished_camera_counts_during_probation(self):
        clk = _Clock()
        present = {}
        g, _l, _s = _gate(clk, present)
        _streaming_pair(g, clk)
        _trip(g, clk)
        _end_cool_down(g, clk)
        clk.advance(20.0)
        d = g.begin(RIGHT, "face-track")
        self.assertTrue(d.allowed)
        present[RIGHT] = False                 # the hub reset under the open
        clk.advance(2.0)
        g.end(RIGHT, "face-track", False)
        self.assertTrue(g.storm_active(),
                        "a camera gone from the device list is a drop")

    def test_a_failed_open_of_a_camera_still_listed_does_not_count(self):
        clk = _Clock()
        present = {}
        g, _l, _s = _gate(clk, present)
        _streaming_pair(g, clk)
        _trip(g, clk)
        _end_cool_down(g, clk)
        clk.advance(20.0)
        self.assertTrue(g.begin(RIGHT, "face-track").allowed)
        g.end(RIGHT, "face-track", False)
        self.assertFalse(g.storm_active())

    def test_a_camera_never_seen_streaming_that_is_unlisted_is_not_a_drop(self):
        clk = _Clock()
        present = {RIGHT: False}
        g, _l, _s = _gate(clk, present)
        _open_ok(g, clk, LEFT)
        clk.advance(10.0)
        g.note_drop(LEFT, "face-track")
        clk.advance(2.0)
        self.assertTrue(g.begin(RIGHT, "cam-probe").allowed)
        g.end(RIGHT, "cam-probe", False, escalate=False)
        self.assertFalse(g.storm_active(),
                         "an unplugged camera at boot tripped the breaker")

    def test_a_reopen_inside_a_live_chain_opens_a_sixty_second_window(self):
        clk = _Clock()
        g, logs, _s = _gate(clk)
        _streaming_pair(g, clk)
        _trip(g, clk)
        _end_cool_down(g, clk)
        clk.advance(cg.STORM_PROBATION_S + 60.0)      # probation over
        self.assertFalse(g.note_drop(LEFT, "face-track"))
        _open_ok(g, clk, LEFT)                         # its recovery reopen
        clk.advance(30.0)
        self.assertTrue(g.note_drop(LEFT, "face-track"),
                        "a drop 30 s after a reopen in a live chain")
        self.assertTrue(any("was reopened" in ln for ln in logs))

    def test_no_reopen_window_outside_a_storm_chain(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _streaming_pair(g, clk)
        g.note_drop(LEFT, "face-track")
        _open_ok(g, clk, LEFT)
        clk.advance(5.0)
        self.assertFalse(g.note_drop(LEFT, "face-track"),
                         "one flaky camera, no storm in the last hour")

    def test_zero_disables_the_probation(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, probation_s=0.0)
        _streaming_pair(g, clk)
        _trip(g, clk)
        _end_cool_down(g, clk)
        clk.advance(20.0)
        self.assertFalse(g.note_drop(LEFT, "face-track"))

    def test_the_probation_is_the_owner_knob(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, probation_s=600.0)
        _streaming_pair(g, clk)
        _trip(g, clk)
        _end_cool_down(g, clk)
        clk.advance(300.0)
        _open_ok(g, clk, LEFT)
        clk.advance(100.0)
        self.assertTrue(g.note_drop(LEFT, "face-track"))
        self.assertIn("probation 600s", g.snapshot()["storm_reason"])


class MinorDropTests(unittest.TestCase):
    """ONE failed read (a side tile's) is not a read-failure BURST and not a
    bus event: on probation it must not stop every camera for 20 minutes by
    itself - unless the camera has also left the device list."""

    def _on_probation(self, present=None):
        clk = _Clock()
        g, logs, spoken = _gate(clk, present)
        _streaming_pair(g, clk)
        _trip(g, clk)
        _end_cool_down(g, clk)
        clk.advance(20.0)
        _open_ok(g, clk, RIGHT, comp="side-tile")
        clk.advance(30.0)
        return clk, g

    def test_one_failed_read_of_a_listed_camera_does_not_re_trip(self):
        clk, g = self._on_probation()
        self.assertFalse(g.note_drop(RIGHT, "side-tile", minor=True))
        self.assertFalse(g.storm_active())

    def test_one_failed_read_of_a_camera_that_left_the_list_re_trips(self):
        present = {}
        clk, g = self._on_probation(present)
        present[RIGHT] = False
        self.assertTrue(g.note_drop(RIGHT, "side-tile", minor=True))

    def test_a_minor_drop_still_counts_toward_the_two_camera_rule(self):
        clk, g = self._on_probation()
        clk.advance(cg.STORM_PROBATION_S)            # probation over
        _open_ok(g, clk, LEFT)                       # its recovery reopen...
        clk.advance(cg.REOPEN_PROBATION_S + 1.0)     # ...and its window over
        self.assertFalse(g.note_drop(RIGHT, "side-tile", minor=True))
        clk.advance(3.0)
        self.assertTrue(g.note_drop(LEFT, "face-track"),
                        "a single read failure + a burst on another camera "
                        "within 10 s is still a storm")

    def test_a_full_burst_after_a_minor_one_on_the_same_stream_counts(self):
        clk, g = self._on_probation()
        self.assertFalse(g.note_drop(RIGHT, "side-tile", minor=True))
        clk.advance(2.0)
        self.assertTrue(g.note_drop(RIGHT, "face-track",
                                    cause="read-failure burst"))


class ReadBurstTests(unittest.TestCase):
    """Read-failure bursts are counted once per STREAM, so two cameras
    bursting within 10 s is a storm even when one of them was reopened after
    an earlier drop and never delivered a healthy frame since."""

    def test_a_reopened_camera_that_fails_again_is_a_new_drop(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _streaming_pair(g, clk)
        g.note_drop(LEFT, "face-track")
        clk.advance(1.0)
        _open_ok(g, clk, LEFT)                # recovered - no frame yet
        clk.advance(300.0)
        g.note_drop(LEFT, "face-track")
        clk.advance(4.0)
        self.assertTrue(g.note_drop(RIGHT, "face-track"),
                        "bursts on two cameras 4 s apart did not trip")

    def test_one_stream_is_still_one_drop(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _streaming_pair(g, clk)
        for _ in range(3):
            g.note_drop(LEFT, "face-track")
            clk.advance(1.0)
        self.assertFalse(g.storm_active())


class _LiveReplay:
    """The 2026-09-29 sequence, step by step, on a gate at the defaults."""

    def setUp(self):
        self.clk = _Clock(0.0)
        self.present = {}
        self.g, self.logs, self.spoken = _gate(self.clk, self.present)

    def _before_the_first_trip(self):
        clk, g = self.clk, self.g
        clk.at("18:00:00")
        _open_ok(g, clk, LEFT)
        clk.at("18:00:03")
        _open_ok(g, clk, RIGHT)
        clk.at("18:10:00")
        for k in (LEFT, RIGHT):
            g.note_frame(k)
        clk.at("18:38:30")
        g.note_drop(LEFT, "face-track")
        g.note_drop(AUDIO, "audio", cg.KIND_AUDIO)
        assert g.storm_active(), "trip #1 is the part that already worked"
        clk.at("18:38:34")
        g.note_drop(RIGHT, "face-track")      # during the cool-down
        clk.at("18:48:34")
        assert not g.storm_active()           # "cool-down over after 10 min"

    def _the_first_reopen_after_the_cool_down(self):
        clk, g = self.clk, self.g
        clk.at("18:48:52.0")
        self.assertTrue(g.begin(LEFT, "face-track").allowed)
        clk.at("18:48:52.8")
        g.end(LEFT, "face-track", True)       # "opened <webcam A>"
        # hub reset at 18:48:53.6 and 18:48:56.1 - the gate cannot see them
        clk.at("18:48:55.8")
        self.assertTrue(g.begin(RIGHT, "face-track").allowed)
        self.present[RIGHT] = False
        clk.at("18:48:58")
        g.end(RIGHT, "face-track", False)     # "... gone from the device list"


class LiveSequenceProbationTests(_LiveReplay, unittest.TestCase):

    def test_the_first_post_cool_down_drop_re_trips_it(self):
        self._before_the_first_trip()
        self._the_first_reopen_after_the_cool_down()
        self.assertTrue(
            self.g.storm_active(),
            "the right webcam vanished 23.9 s after the cool-down and the "
            "breaker did not trip again - the 2026-09-29 gap")
        snap = self.g.snapshot()
        self.assertEqual(snap["storm_trips"], 2)
        self.assertEqual(round(snap["storm_remaining_s"]), 1200)

    def test_the_left_webcam_is_not_reopened_into_the_resetting_hub(self):
        self._before_the_first_trip()
        self._the_first_reopen_after_the_cool_down()
        self.clk.at("18:49:25.55")
        d = self.g.begin(LEFT, "face-track")
        self.assertFalse(d.allowed, "reopened at 18:49:25.55; the hub reset "
                                    "0.04 s later")
        self.assertEqual(d.reason, "usb-storm")


class LiveSequenceQuarantineTests(_LiveReplay, unittest.TestCase):

    def _second_culprit_event(self):
        clk, g = self.clk, self.g
        self._before_the_first_trip()
        self._the_first_reopen_after_the_cool_down()
        clk.at("18:49:00")
        g.note_drop(LEFT, "face-track", onset=clk.t - 6.4,
                    cause="read-failure burst")
        clk.at("18:49:30")
        self.present[RIGHT] = True            # back on the bus
        clk.at("19:08:58.5")
        self.assertFalse(g.storm_active())    # the doubled cool-down is over
        clk.at("19:09:00.0")
        self.assertTrue(g.begin(LEFT, "face-track").allowed)
        clk.at("19:09:00.8")
        g.end(LEFT, "face-track", True)
        # 8 of 10 of its stream starts reset the hub: this one does too.
        clk.at("19:09:02.0")
        g.note_drop(AUDIO, "audio", cg.KIND_AUDIO, onset=clk.t - 0.8)

    def test_the_second_culprit_event_quarantines_the_left_webcam(self):
        self._second_culprit_event()
        g = self.g
        self.assertTrue(g.storm_active(), "trip #3 (probation)")
        self.assertTrue(g.quarantined(LEFT))
        self.assertFalse(g.quarantined(RIGHT),
                         "a device whose own open FAILED is a victim, never "
                         "the culprit")
        snap = g.snapshot()
        self.assertEqual(snap["quarantined"][LEFT]["label"], "the left webcam")
        self.assertEqual(snap["devices"][LEFT]["culprit_strikes"], 2)
        self.assertEqual(snap["devices"][RIGHT]["culprit_strikes"], 0)
        self.assertEqual(
            self.spoken[-1],
            "Sir, the left webcam keeps knocking the USB hub offline whenever "
            "it starts, so I've stopped using it until it's moved to another "
            "port.")
        q_lines = [ln for ln in self.logs if "[camera-quarantine]" in ln]
        self.assertEqual(len(q_lines), 1, self.logs)
        self.assertIn(LEFT, q_lines[0])
        self.assertIn("rest of this session", q_lines[0])
        culprit = [ln for ln in self.logs if "[camera-culprit]" in ln]
        self.assertEqual(len(culprit), 2, culprit)
        self.assertTrue(all(LEFT in ln for ln in culprit), culprit)

    def test_the_left_webcam_stays_off_and_the_right_one_keeps_working(self):
        self._second_culprit_event()
        clk, g = self.clk, self.g
        clk.advance(g.snapshot()["storm_remaining_s"] + 1.0)
        self.assertFalse(g.storm_active())
        d = g.begin(LEFT, "face-track")
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "quarantined")
        self.assertIn(d.reason, cg.HOLD_REASONS)
        for comp in ("side-tile", "cam-probe", "self-diag"):
            self.assertEqual(g.begin(LEFT, comp).reason, "quarantined", comp)
        # The right webcam settles back in and streams normally.
        for _ in range(20):
            d = g.begin(RIGHT, "face-track")
            if d.allowed:
                break
            clk.advance(max(1.0, min(d.wait_s, 5.0)))
        self.assertTrue(d.allowed, d)
        clk.advance(0.8)
        g.end(RIGHT, "face-track", True)
        for _ in range(60):
            clk.advance(5.0)
            g.note_frame(RIGHT)
        self.assertFalse(g.storm_active())
        self.assertFalse(g.quarantined(RIGHT))
        # ...and the left one is still off hours later: rest of the session.
        clk.advance(6 * 3600.0)
        self.assertEqual(g.begin(LEFT, "face-track").reason, "quarantined")
        # Said once, in total: the first storm, then the quarantine.
        self.assertEqual(len(self.spoken), 2, self.spoken)


class CulpritRuleTests(unittest.TestCase):

    def test_one_culprit_event_is_not_a_quarantine(self):
        clk = _Clock()
        g, logs, spoken = _gate(clk)
        _streaming_pair(g, clk)
        clk.advance(600.0)
        g.note_drop(LEFT, "face-track")
        _open_ok(g, clk, LEFT)
        clk.advance(0.5)
        g.note_drop(RIGHT, "face-track")
        g.note_drop(LEFT, "face-track")                # trips
        self.assertTrue(g.storm_active())
        self.assertEqual(g.snapshot()["devices"][LEFT]["culprit_strikes"], 1)
        self.assertFalse(g.quarantined(LEFT))
        self.assertEqual(len(spoken), 1)

    def test_strikes_older_than_an_hour_are_forgotten(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, storm_cooldown_s=60.0)
        _streaming_pair(g, clk)
        for gap in (0.0, cg.CULPRIT_MEMORY_S + 10.0):
            clk.advance(gap)
            g.note_drop(LEFT, "face-track")
            clk.advance(1.0)
            _open_ok(g, clk, LEFT)
            clk.advance(0.5)
            g.note_drop(RIGHT, "face-track")
            g.note_drop(LEFT, "face-track")
            self.assertTrue(g.storm_active())
            clk.advance(g.snapshot()["storm_remaining_s"] + 1.0)
            g.storm_active()
            g.note_frame(RIGHT)
        self.assertFalse(g.quarantined(LEFT))
        self.assertEqual(g.snapshot()["devices"][LEFT]["culprit_strikes"], 1)

    def test_the_latest_starter_is_blamed_not_an_earlier_one(self):
        clk = _Clock()
        g, logs, _s = _gate(clk)
        _open_ok(g, clk, LEFT)
        clk.advance(3.0)
        _open_ok(g, clk, RIGHT)
        clk.advance(0.5)
        g.note_drop(LEFT, "face-track")
        g.note_drop(RIGHT, "face-track")
        snap = g.snapshot()["devices"]
        self.assertEqual(snap[RIGHT]["culprit_strikes"], 1)
        self.assertEqual(snap[LEFT]["culprit_strikes"], 0)

    def test_an_event_outside_the_window_strikes_nobody(self):
        clk = _Clock()
        g, logs, _s = _gate(clk)
        _open_ok(g, clk, LEFT)
        clk.advance(cg.CULPRIT_WINDOW_S + 1.0)
        g.note_drop(LEFT, "face-track")
        g.note_drop(RIGHT, "face-track")
        self.assertTrue(g.storm_active())
        self.assertEqual(g.snapshot()["devices"][LEFT]["culprit_strikes"], 0)
        self.assertFalse(any("[camera-culprit]" in ln for ln in logs))

    def test_the_onset_not_the_report_time_is_what_is_measured(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _open_ok(g, clk, LEFT)
        clk.advance(9.0)                 # the burst took 9 s to reach #25...
        g.note_drop(LEFT, "face-track", onset=clk.t - 8.2)
        g.note_drop(RIGHT, "face-track", onset=clk.t - 8.0)
        self.assertEqual(g.snapshot()["devices"][LEFT]["culprit_strikes"], 1,
                         "...but it began 0.8 s after the stream started")

    def test_a_camera_vanishing_is_a_hub_event_even_without_a_trip(self):
        clk = _Clock()
        present = {}
        g, logs, _s = _gate(clk, present)
        _open_ok(g, clk, RIGHT)
        clk.advance(60.0)
        _open_ok(g, clk, LEFT)
        clk.advance(1.0)
        present[RIGHT] = False
        g.note_drop(RIGHT, "face-track")
        self.assertFalse(g.storm_active())
        self.assertEqual(g.snapshot()["devices"][LEFT]["culprit_strikes"], 1)

    def test_one_stream_start_earns_at_most_one_strike(self):
        clk = _Clock()
        present = {}
        g, _l, _s = _gate(clk, present)
        _open_ok(g, clk, RIGHT)
        clk.advance(60.0)
        _open_ok(g, clk, LEFT)
        clk.advance(1.0)
        present[RIGHT] = False
        g.note_drop(RIGHT, "face-track")          # a vanish: strike
        g.note_drop(AUDIO, "audio", cg.KIND_AUDIO)   # ...then the trip
        self.assertTrue(g.storm_active())
        self.assertEqual(g.snapshot()["devices"][LEFT]["culprit_strikes"], 1)
        self.assertFalse(g.quarantined(LEFT))

    def test_the_threshold_and_window_are_owner_knobs_and_zero_is_off(self):
        for kw in ({"culprit_threshold": 0}, {"culprit_window_s": 0.0}):
            clk = _Clock()
            g, logs, _s = _gate(clk, **kw)
            _open_ok(g, clk, LEFT)
            clk.advance(0.5)
            g.note_drop(LEFT, "face-track")
            g.note_drop(RIGHT, "face-track")
            self.assertTrue(g.storm_active())
            self.assertEqual(
                g.snapshot()["devices"][LEFT]["culprit_strikes"], 0, kw)
        clk = _Clock()
        g, _l, spoken = _gate(clk, culprit_threshold=1)
        _open_ok(g, clk, LEFT)
        clk.advance(0.5)
        g.note_drop(LEFT, "face-track")
        g.note_drop(RIGHT, "face-track")
        self.assertTrue(g.quarantined(LEFT), "threshold 1 = the first event")
        self.assertIn("the left webcam", spoken[-1])

    def test_lift_quarantine_lets_it_open_and_starts_from_zero(self):
        clk = _Clock()
        g, logs, _s = _gate(clk, culprit_threshold=1, storm_cooldown_s=60.0)
        _open_ok(g, clk, LEFT)
        clk.advance(0.5)
        g.note_drop(LEFT, "face-track")
        g.note_drop(RIGHT, "face-track")
        self.assertTrue(g.quarantined(LEFT))
        self.assertEqual(g.quarantined(), {LEFT: "the left webcam"})
        self.assertFalse(g.lift_quarantine(RIGHT))
        self.assertTrue(g.lift_quarantine(LEFT))
        self.assertFalse(g.quarantined(LEFT))
        self.assertEqual(g.snapshot()["devices"][LEFT]["culprit_strikes"], 0)
        self.assertTrue(any("lifted by the owner" in ln for ln in logs))
        clk.advance(61.0)
        clk.advance(cg.STORM_PROBATION_S + 1.0)
        self.assertTrue(g.begin(LEFT, "face-track").allowed)

    def test_a_raising_labeler_falls_back_to_the_key(self):
        clk = _Clock()

        def _boom(_key):
            raise RuntimeError("synthetic")
        g, _l, spoken = _gate(clk, culprit_threshold=1, labeler=_boom)
        _open_ok(g, clk, LEFT)
        clk.advance(0.5)
        g.note_drop(LEFT, "face-track")
        g.note_drop(RIGHT, "face-track")
        self.assertTrue(g.quarantined(LEFT))
        self.assertIn("the synth-left camera", spoken[-1])

    def test_garbage_knobs_fall_back_to_the_shipped_defaults(self):
        g = cg.CameraGate(probation_s="x", culprit_window_s=float("nan"),
                          culprit_threshold="y")
        self.assertEqual(g.probation_s, cg.STORM_PROBATION_S)
        self.assertEqual(g.culprit_window_s, cg.CULPRIT_WINDOW_S)
        self.assertEqual(g.culprit_threshold, cg.CULPRIT_THRESHOLD)

    def test_nothing_is_persisted_reset_forgets_the_quarantine(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, culprit_threshold=1)
        _open_ok(g, clk, LEFT)
        clk.advance(0.5)
        g.note_drop(LEFT, "face-track")
        g.note_drop(RIGHT, "face-track")
        self.assertTrue(g.quarantined(LEFT))
        g.reset()
        self.assertFalse(g.quarantined(LEFT))
        self.assertEqual(g.quarantined(), {})


if __name__ == "__main__":
    unittest.main()
