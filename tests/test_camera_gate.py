"""core/camera_gate.py - the ONE gate in front of every camera and Kinect open.

Pure bookkeeping on an injected clock: no camera, no cv2, no Windows. Every
rule the owner asked for on 2026-09-29 (after his chained USB hubs reset about
once a minute while JARVIS kept reopening cameras) is pinned here at its
PRODUCTION default, with time frozen and advanced by hand.

Device keys and app names below are synthetic on purpose (public repo).
"""
from __future__ import annotations

import unittest

from core import camera_gate as cg


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def _gate(clock, **kw):
    """A gate at the SHIPPED defaults unless a test says otherwise."""
    logs: list = []
    spoken: list = []
    kw.setdefault("log", logs.append)
    kw.setdefault("announce", spoken.append)
    g = cg.CameraGate(clock=clock, **kw)
    return g, logs, spoken


def _fail(g, key, comp="face-track", **kw):
    d = g.begin(key, comp)
    assert d.allowed, d
    g.end(key, comp, False, **kw)


class BackoffLadderTests(unittest.TestCase):
    """Per camera: 30 -> 60 -> 120 -> 300 -> 600 s after failed opens."""

    def test_the_ladder_is_30_60_120_300_600_then_capped(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        waits = []
        for _ in range(7):
            _fail(g, "name:synth-a")
            wait, reason = g.retry_in("name:synth-a", "face-track")
            self.assertEqual(reason, "backoff")
            waits.append(round(wait))
            clk.advance(wait)
        self.assertEqual(waits, [30, 60, 120, 300, 600, 600, 600])

    def test_the_cap_is_the_owner_knob(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, max_backoff_s=120.0)
        waits = []
        for _ in range(5):
            _fail(g, "name:synth-a")
            wait, _r = g.retry_in("name:synth-a", "face-track")
            waits.append(round(wait))
            clk.advance(wait)
        self.assertEqual(waits, [30, 60, 120, 120, 120])

    def test_a_higher_cap_keeps_doubling_past_the_table(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, max_backoff_s=2400.0)
        waits = []
        for _ in range(7):
            _fail(g, "name:synth-a")
            wait, _r = g.retry_in("name:synth-a", "face-track")
            waits.append(round(wait))
            clk.advance(wait)
        self.assertEqual(waits, [30, 60, 120, 300, 600, 1200, 2400])

    def test_every_component_sees_the_backoff_not_only_the_one_that_failed(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _fail(g, "name:synth-a", "face-track")
        for comp in ("side-tile", "cam-probe", "self-diag"):
            d = g.begin("name:synth-a", comp)
            self.assertFalse(d.allowed, comp)
            self.assertEqual(d.reason, "backoff")

    def test_a_non_escalating_failure_arms_no_backoff(self):
        # The boot probes report with escalate=False: their own bounded retry
        # pass must not be cancelled by a 30 s ladder.
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _fail(g, "name:synth-a", "cam-probe", escalate=False)
        self.assertTrue(g.begin("name:synth-a", "cam-probe").allowed)

    def test_the_ladder_resets_only_after_sixty_seconds_of_healthy_frames(self):
        clk = _Clock()
        g, logs, _s = _gate(clk)
        key = "name:synth-a"
        _fail(g, key)
        _fail_at = clk.t
        clk.advance(30.0)
        self.assertTrue(g.begin(key, "face-track").allowed)
        g.end(key, "face-track", True)
        # 59 s of frames: NOT enough.
        for _ in range(60):
            g.note_frame(key)
            clk.advance(59.0 / 60.0)
        _fail(g, key)
        wait, _r = g.retry_in(key, "face-track")
        self.assertEqual(round(wait), 60, "59 s of frames reset the ladder")
        # Now a full 60 s run of frames resets it.
        clk.advance(wait)
        self.assertTrue(g.begin(key, "face-track").allowed)
        g.end(key, "face-track", True)
        for _ in range(61):
            g.note_frame(key)
            clk.advance(1.0)
        self.assertTrue(any("reopen backoff reset" in ln for ln in logs))
        _fail(g, key)
        wait, _r = g.retry_in(key, "face-track")
        self.assertEqual(round(wait), 30, "the ladder did not restart at 30 s")
        self.assertGreater(clk.t, _fail_at)

    def test_a_failure_breaks_the_healthy_run(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        key = "name:synth-a"
        _fail(g, key)
        clk.advance(30.0)
        g.begin(key, "face-track")
        g.end(key, "face-track", True)
        for _ in range(40):
            g.note_frame(key)
            clk.advance(1.0)
        g.note_drop(key, "face-track")          # a drop mid-run ...
        clk.advance(1.0)
        for _ in range(40):                     # ... so 80 s total is NOT 60 s straight
            g.note_frame(key)
            clk.advance(1.0)
        self.assertGreater(g.snapshot()["devices"][key]["level"], 0)

    def test_zero_disables_the_ladder(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, max_backoff_s=0.0)
        _fail(g, "name:synth-a")
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)


class RecoveryTests(unittest.TestCase):
    """Read-failure RECOVERIES back off too, not only failed opens."""

    def test_a_recovery_that_works_still_spends_a_rung(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        key = "name:synth-a"
        g.begin(key, "face-track")
        g.end(key, "face-track", True)
        clk.advance(120.0)
        g.note_drop(key, "face-track")
        d = g.begin(key, "face-track")
        self.assertTrue(d.allowed, "the FIRST recovery must be immediate")
        g.end(key, "face-track", True)
        clk.advance(5.0)
        g.note_drop(key, "face-track")
        d = g.begin(key, "face-track")
        self.assertFalse(d.allowed, "a second recovery 5 s later was allowed")
        self.assertEqual(d.reason, "backoff")
        self.assertAlmostEqual(d.wait_s, 25.0, places=3)

    def test_a_drop_is_counted_once_per_episode(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        g.begin("name:synth-a", "face-track")
        g.end("name:synth-a", "face-track", True)
        for _ in range(50):
            g.note_drop("name:synth-a", "face-track")
        # One camera, one episode: the breaker needs two DISTINCT cameras.
        self.assertFalse(g.storm_active())


class LockedDeviceTests(unittest.TestCase):
    """A camera another app holds is NOT retried on a timer."""

    def test_no_open_while_the_locking_app_runs(self):
        clk = _Clock()
        running = [["SynthMeet.exe"]]
        g, _l, _s = _gate(clk, lockers=lambda: list(running[0]))
        _fail(g, "name:synth-a", lockers=["SynthMeet.exe"])
        for _ in range(100):                    # 500 s of asking every 5 s
            clk.advance(5.0)
            d = g.begin("name:synth-a", "face-track")
            self.assertFalse(d.allowed)
            self.assertEqual(d.reason, "locked")
            self.assertLessEqual(d.wait_s, cg.LOCKED_POLL_S)

    def test_retried_the_moment_the_locking_app_is_gone(self):
        clk = _Clock()
        running = [["SynthMeet.exe"]]
        g, logs, _s = _gate(clk, lockers=lambda: list(running[0]))
        _fail(g, "name:synth-a", lockers=["SynthMeet.exe"])
        clk.advance(40.0)
        running[0] = ["SynthChat.exe"]          # a DIFFERENT app still runs
        d = g.begin("name:synth-a", "face-track")
        self.assertTrue(d.allowed, d)
        self.assertTrue(any("is no longer using a webcam" in ln for ln in logs))

    def test_at_most_every_ten_minutes_when_the_app_never_leaves(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, lockers=lambda: ["SynthMeet.exe"])
        _fail(g, "name:synth-a", lockers=["SynthMeet.exe"])
        clk.advance(cg.LOCKED_RETRY_S - 1.0)
        self.assertFalse(g.begin("name:synth-a", "face-track").allowed)
        clk.advance(2.0)
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)

    def test_without_a_cheap_check_it_waits_for_the_cap(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, lockers=None)
        _fail(g, "name:synth-a", lockers=["SynthMeet.exe"])
        clk.advance(300.0)
        self.assertEqual(g.begin("name:synth-a", "face-track").reason, "locked")
        clk.advance(cg.LOCKED_RETRY_S)
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)

    def test_a_raising_locker_poll_is_not_fatal(self):
        clk = _Clock()

        def _boom():
            raise RuntimeError("psutil fell over")
        g, _l, _s = _gate(clk, lockers=_boom)
        _fail(g, "name:synth-a", lockers=["SynthMeet.exe"])
        clk.advance(10.0)
        d = g.begin("name:synth-a", "face-track")
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "locked")

    def test_frames_clear_the_lock(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, lockers=lambda: ["SynthMeet.exe"])
        _fail(g, "name:synth-a", lockers=["SynthMeet.exe"])
        g.note_frame("name:synth-a")
        self.assertEqual(g.locked_by("name:synth-a"), [])


class OneDeviceTwoComponentsTests(unittest.TestCase):
    """Never two JARVIS components on one device at once or back-to-back."""

    def test_min_gap_between_different_components(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        self.assertTrue(g.begin("name:synth-a", "cam-probe").allowed)
        g.end("name:synth-a", "cam-probe", True)
        clk.advance(3.0)
        d = g.begin("name:synth-a", "face-track")
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "min-gap")
        self.assertAlmostEqual(d.wait_s, 7.0, places=3)
        clk.advance(7.0)
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)

    def test_a_component_is_not_gapped_against_itself(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        g.begin("name:synth-a", "cam-probe")
        g.end("name:synth-a", "cam-probe", False, escalate=False)
        clk.advance(1.0)
        self.assertTrue(g.begin("name:synth-a", "cam-probe").allowed)

    def test_at_once_is_refused_as_in_flight(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, min_gap_s=0.0)
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)
        d = g.begin("name:synth-a", "side-tile")
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "in-flight")
        self.assertIn(d.reason, cg.TIMING_REASONS)

    def test_an_abandoned_reservation_goes_stale(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, min_gap_s=0.0)
        g.begin("name:synth-a", "face-track")             # never ended
        clk.advance(cg.IN_FLIGHT_STALE_S + 1.0)
        self.assertTrue(g.begin("name:synth-a", "side-tile").allowed)

    def test_cancel_drops_a_reservation_without_counting_anything(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, min_gap_s=0.0)
        g.begin("name:synth-a", "cam-probe")
        g.cancel("name:synth-a", "cam-probe")
        self.assertTrue(g.begin("name:synth-a", "side-tile").allowed)
        self.assertEqual(g.snapshot()["devices"]["name:synth-a"]["fails"], 0)

    def test_a_streaming_device_is_held_from_everyone_else(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        g.hold("kinect", "kinect-bridge")
        clk.advance(3600.0)
        d = g.begin("kinect", "self-diag")
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "held")
        self.assertIn(d.reason, cg.HOLD_REASONS)
        g.unhold("kinect", "kinect-bridge")
        self.assertTrue(g.begin("kinect", "self-diag").allowed)

    def test_zero_disables_the_gap(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, min_gap_s=0.0)
        g.begin("name:synth-a", "cam-probe")
        g.end("name:synth-a", "cam-probe", True)
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)


class BootStaggerTests(unittest.TestCase):
    """Don't open all the cameras and the Kinect in the same second."""

    def test_different_devices_are_spaced(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        opened = []
        for key in ("kinect", "name:synth-a", "name:synth-b"):
            while True:
                d = g.begin(key, "face-track")
                if d.allowed:
                    opened.append((key, clk.t))
                    g.end(key, "face-track", True)
                    break
                self.assertEqual(d.reason, "stagger")
                clk.advance(d.wait_s)
        gaps = [b[1] - a[1] for a, b in zip(opened, opened[1:])]
        self.assertEqual([round(x, 3) for x in gaps],
                         [cg.OPEN_STAGGER_S, cg.OPEN_STAGGER_S])

    def test_bare_index_sweeps_are_not_staggered(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        for i in range(12):
            self.assertTrue(g.begin(f"dshow:{i}", "cam-probe").allowed, i)
            g.end(f"dshow:{i}", "cam-probe", False, escalate=False, count=False)

    def test_the_same_device_is_not_staggered_against_itself(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        g.begin("name:synth-a", "face-track")
        g.end("name:synth-a", "face-track", True)
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)


class UsbStormBreakerTests(unittest.TestCase):
    """>=2 cameras (or a camera and an audio device) dropping within ~10 s,
    or >=3 open failures across devices within 60 s, is a USB bus event."""

    def test_two_cameras_dropping_together_trip_it(self):
        clk = _Clock()
        g, logs, spoken = _gate(clk)
        self.assertFalse(g.note_drop("name:synth-a", "face-track"))
        clk.advance(4.0)
        self.assertTrue(g.note_drop("kinect", "kinect-bridge"))
        self.assertTrue(g.storm_active())
        storm_lines = [ln for ln in logs if "[usb-storm]" in ln]
        self.assertEqual(len(storm_lines), 1, logs)
        self.assertIn("2 cameras dropped within 10s", storm_lines[0])
        self.assertEqual(
            spoken,
            ["Sir, the USB bus looks unstable, so I'm leaving the cameras "
             "alone for ten minutes."])

    def test_drops_further_apart_are_not_a_storm(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        g.note_drop("name:synth-a", "face-track")
        clk.advance(11.0)
        g.note_drop("name:synth-b", "face-track")
        self.assertFalse(g.storm_active())

    def test_a_camera_and_an_audio_device_trip_it(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        g.note_drop("audio:input", "audio", cg.KIND_AUDIO)
        clk.advance(2.0)
        g.note_drop("name:synth-a", "face-track")
        self.assertTrue(g.storm_active())

    def test_audio_alone_never_trips_it(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        g.note_drop("audio:input", "audio", cg.KIND_AUDIO)
        g.note_drop("audio:output", "audio", cg.KIND_AUDIO)
        self.assertFalse(g.storm_active())

    def test_three_open_failures_across_two_devices_trip_it(self):
        clk = _Clock()
        g, logs, _s = _gate(clk, min_gap_s=0.0, stagger_s=0.0)
        _fail(g, "name:synth-a", escalate=False)
        clk.advance(20.0)
        _fail(g, "name:synth-b", escalate=False)
        self.assertFalse(g.storm_active())
        clk.advance(20.0)
        _fail(g, "name:synth-a", escalate=False)
        self.assertTrue(g.storm_active())
        self.assertTrue(any("3 camera open failures across 2 devices"
                            in ln for ln in logs))

    def test_three_failures_of_one_device_are_that_device_not_the_bus(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, min_gap_s=0.0, stagger_s=0.0)
        for _ in range(5):
            _fail(g, "name:synth-a", escalate=False)
            clk.advance(5.0)
        self.assertFalse(g.storm_active())

    def test_failures_older_than_a_minute_do_not_count(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, min_gap_s=0.0, stagger_s=0.0)
        _fail(g, "name:synth-a", escalate=False)
        clk.advance(61.0)
        _fail(g, "name:synth-b", escalate=False)
        _fail(g, "name:synth-a", escalate=False)
        self.assertFalse(g.storm_active())

    def test_the_cool_down_stops_every_camera_and_the_kinect(self):
        clk = _Clock()
        g, logs, _s = _gate(clk)
        g.note_drop("name:synth-a", "face-track")
        g.note_drop("name:synth-b", "face-track")
        for key, comp in (("name:synth-a", "face-track"),
                          ("name:synth-c", "side-tile"),
                          ("kinect", "kinect-bridge"),
                          ("dshow:5", "cam-probe"),
                          ("msmf:synth", "self-diag")):
            d = g.begin(key, comp)
            self.assertFalse(d.allowed, key)
            self.assertEqual(d.reason, "usb-storm")
            self.assertAlmostEqual(d.wait_s, 600.0, places=3)
        clk.advance(599.0)
        self.assertFalse(g.begin("kinect", "kinect-bridge").allowed)
        clk.advance(2.0)
        self.assertTrue(g.begin("kinect", "kinect-bridge").allowed)
        ends = [ln for ln in logs if "cool-down over" in ln]
        self.assertEqual(len(ends), 1)
        # ...and it is still said only once however often it is asked.
        g.begin("name:synth-a", "face-track")
        self.assertEqual(len([ln for ln in logs if "cool-down over" in ln]), 1)

    def test_a_repeat_within_the_hour_doubles_and_is_not_spoken_again(self):
        clk = _Clock()
        g, logs, spoken = _gate(clk)
        cools = []
        for _ in range(5):
            g.note_drop("name:synth-a", "face-track")
            g.note_drop("name:synth-b", "face-track")
            snap = g.snapshot()
            cools.append(round(snap["storm_remaining_s"]))
            clk.advance(snap["storm_remaining_s"] + 1.0)
            g.storm_active()                     # let the end register
            # both devices streamed again, then drop again 5 min later
            for key in ("name:synth-a", "name:synth-b"):
                g.note_frame(key)
            clk.advance(300.0)
        self.assertEqual(cools, [600, 1200, 2400, 3600, 3600])
        self.assertEqual(len(spoken), 1, spoken)
        self.assertEqual(len([ln for ln in logs if "doubled" in ln]), 4)

    def test_after_a_quiet_hour_it_starts_again_at_ten_minutes(self):
        clk = _Clock()
        g, _l, spoken = _gate(clk)
        g.note_drop("name:synth-a", "face-track")
        g.note_drop("name:synth-b", "face-track")
        clk.advance(601.0)
        g.storm_active()
        g.note_frame("name:synth-a")
        g.note_frame("name:synth-b")
        clk.advance(cg.STORM_COOLDOWN_MAX_S + 1.0)
        g.note_drop("name:synth-a", "face-track")
        g.note_drop("name:synth-b", "face-track")
        self.assertEqual(round(g.snapshot()["storm_remaining_s"]), 600)
        self.assertEqual(len(spoken), 2, "a new chain is announced again")

    def test_the_cool_down_is_the_owner_knob_and_zero_disables_it(self):
        clk = _Clock()
        g, _l, spoken = _gate(clk, storm_cooldown_s=1200.0)
        g.note_drop("name:synth-a", "face-track")
        g.note_drop("name:synth-b", "face-track")
        self.assertEqual(round(g.snapshot()["storm_remaining_s"]), 1200)
        self.assertIn("twenty minutes", spoken[0])
        g2, _l2, _s2 = _gate(clk, storm_cooldown_s=0.0)
        g2.note_drop("name:synth-a", "face-track")
        g2.note_drop("name:synth-b", "face-track")
        self.assertFalse(g2.storm_active())

    def test_events_from_before_a_cool_down_cannot_re_trip_it(self):
        clk = _Clock()
        g, _l, _s = _gate(clk, min_gap_s=0.0, stagger_s=0.0)
        _fail(g, "name:synth-a", escalate=False)
        _fail(g, "name:synth-b", escalate=False)
        _fail(g, "name:synth-a", escalate=False)      # trips
        clk.advance(601.0)
        self.assertFalse(g.storm_active())
        _fail(g, "name:synth-b", escalate=False)      # one new failure only
        self.assertFalse(g.storm_active())


class AbsentDeviceTests(unittest.TestCase):
    """A device that VANISHED is not retried on a timer: presence is polled,
    and after it returns it must stay listed ABSENT_SETTLE_S first."""

    def _gate(self, clk, present):
        return _gate(clk, presence=lambda key: present[0])

    def test_a_drop_of_a_vanished_device_marks_it_absent(self):
        clk = _Clock()
        present = [False]
        g, logs, _s = self._gate(clk, present)
        g.note_drop("name:synth-a", "face-track")
        d = g.begin("name:synth-a", "face-track")
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "absent")
        self.assertIn(d.reason, cg.HOLD_REASONS)
        self.assertTrue(any("gone from the device list" in ln for ln in logs))

    def test_no_timer_retry_however_long_it_is_gone(self):
        clk = _Clock()
        present = [False]
        g, _l, _s = self._gate(clk, present)
        _fail(g, "name:synth-a", escalate=False)       # failed while gone
        for _ in range(200):                             # 10 min of asking
            clk.advance(cg.ABSENT_POLL_S)
            self.assertEqual(g.begin("name:synth-a", "face-track").reason,
                             "absent")

    def test_back_on_the_bus_it_settles_before_the_open(self):
        clk = _Clock()
        present = [False]
        g, logs, _s = self._gate(clk, present)
        g.note_drop("name:synth-a", "face-track")
        clk.advance(40.0)
        present[0] = True
        d = g.begin("name:synth-a", "face-track")
        self.assertEqual(d.reason, "settling")
        self.assertAlmostEqual(d.wait_s, cg.ABSENT_SETTLE_S, places=3)
        self.assertTrue(any("back on the device list" in ln for ln in logs))
        clk.advance(cg.ABSENT_SETTLE_S - 1.0)
        self.assertEqual(g.begin("name:synth-a", "face-track").reason,
                         "settling")
        clk.advance(1.01)
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)

    def test_leaving_again_mid_settle_restarts_the_settle(self):
        clk = _Clock()
        present = [False]
        g, _l, _s = self._gate(clk, present)
        g.note_drop("name:synth-a", "face-track")
        present[0] = True
        g.begin("name:synth-a", "face-track")          # arrival stamped
        clk.advance(10.0)
        present[0] = False                              # flaps off again
        self.assertEqual(g.begin("name:synth-a", "face-track").reason, "absent")
        present[0] = True
        clk.advance(1.0)
        d = g.begin("name:synth-a", "face-track")
        self.assertEqual(d.reason, "settling")
        self.assertAlmostEqual(d.wait_s, cg.ABSENT_SETTLE_S, places=3)

    def test_a_present_device_that_fails_is_not_absent(self):
        clk = _Clock()
        present = [True]
        g, _l, _s = self._gate(clk, present)
        _fail(g, "name:synth-a")
        self.assertEqual(g.begin("name:synth-a", "face-track").reason,
                         "backoff")

    def test_unreadable_presence_falls_back_to_the_ten_minute_cap(self):
        clk = _Clock()
        state = [False]
        g, _l, _s = _gate(clk, presence=lambda key: state[0])
        g.note_drop("name:synth-a", "face-track")       # marked absent
        state[0] = None                                   # can no longer tell
        clk.advance(60.0)
        self.assertEqual(g.begin("name:synth-a", "face-track").reason, "absent")
        clk.advance(cg.ABSENT_UNKNOWN_RETRY_S)
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)

    def test_without_a_presence_check_nothing_is_ever_absent(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        g.note_drop("name:synth-a", "face-track")
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)


class WedgedOpenTests(unittest.TestCase):
    """Never start a new open while another is stuck inside the driver."""

    def test_a_wedge_holds_every_device_until_it_returns(self):
        clk = _Clock()
        g, logs, _s = _gate(clk)
        tok = g.note_wedged("name:synth-a", "face-track")
        self.assertTrue(g.wedged())
        for key in ("name:synth-a", "name:synth-b", "kinect", "dshow:3"):
            d = g.begin(key, "cam-probe")
            self.assertFalse(d.allowed, key)
            self.assertEqual(d.reason, "wedged")
        g.note_unwedged(tok)
        self.assertFalse(g.wedged())
        self.assertTrue(g.begin("name:synth-b", "cam-probe").allowed)
        self.assertTrue(any("stuck inside the camera driver" in ln
                            for ln in logs))
        self.assertTrue(any("the stuck open returned" in ln for ln in logs))

    def test_other_devices_are_released_after_the_cap_the_stuck_one_is_not(self):
        clk = _Clock()
        g, logs, _s = _gate(clk)
        g.note_wedged("name:synth-a", "face-track")
        clk.advance(cg.WEDGE_HOLD_OTHERS_MAX_S + 1.0)
        self.assertTrue(g.begin("name:synth-b", "face-track").allowed)
        self.assertEqual(g.begin("name:synth-a", "face-track").reason, "wedged")
        self.assertEqual(sum("releasing every OTHER device" in ln
                             for ln in logs), 1)

    def test_two_wedges_both_have_to_return(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        t1 = g.note_wedged("name:synth-a", "face-track")
        t2 = g.note_wedged("name:synth-b", "cam-probe")
        g.note_unwedged(t1)
        self.assertTrue(g.wedged())
        g.note_unwedged(t2)
        self.assertFalse(g.wedged())


class NeverRaisesTests(unittest.TestCase):

    def test_raising_log_and_announce_callbacks_are_contained(self):
        clk = _Clock()

        def _boom(_msg):
            raise RuntimeError("stdout is gone")
        g = cg.CameraGate(clock=clk, log=_boom, announce=_boom)
        g.note_drop("name:synth-a", "face-track")
        self.assertTrue(g.note_drop("name:synth-b", "face-track"))
        self.assertFalse(g.begin("name:synth-a", "face-track").allowed)

    def test_garbage_knobs_fall_back_to_the_shipped_defaults(self):
        g = cg.CameraGate(min_gap_s="nope", max_backoff_s=float("nan"),
                          storm_cooldown_s=-5)
        self.assertEqual(g.min_gap_s, 10.0)
        self.assertEqual(g.max_backoff_s, 600.0)
        self.assertEqual(g.storm_cooldown_s, 600.0)

    def test_a_broken_clock_fails_open(self):
        def _clock():
            raise RuntimeError("clock")
        g = cg.CameraGate(clock=_clock)
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)
        g.end("name:synth-a", "face-track", False)
        g.note_frame("name:synth-a")
        self.assertFalse(g.note_drop("name:synth-a", "face-track"))
        self.assertIsInstance(g.snapshot(), dict)


class SmallPiecesTests(unittest.TestCase):

    def test_minutes_phrase(self):
        self.assertEqual(cg.minutes_phrase(600), "ten minutes")
        self.assertEqual(cg.minutes_phrase(1200), "twenty minutes")
        self.assertEqual(cg.minutes_phrase(3600), "an hour")
        self.assertEqual(cg.minutes_phrase(7200), "2 hours")
        self.assertEqual(cg.minutes_phrase(420), "7 minutes")
        self.assertEqual(cg.minutes_phrase("x"), "a while")

    def test_recent_success_is_per_component_and_windowed(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        g.begin("name:synth-a", "cam-probe")
        g.end("name:synth-a", "cam-probe", True)
        clk.advance(5.0)
        self.assertAlmostEqual(g.recent_success("name:synth-a", "cam-probe", 120.0), 5.0)
        self.assertIsNone(g.recent_success("name:synth-a", "face-track", 120.0))
        clk.advance(200.0)
        self.assertIsNone(g.recent_success("name:synth-a", "cam-probe", 120.0))

    def test_snapshot_is_plain_data(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _fail(g, "name:synth-a")
        g.begin("name:synth-a", "side-tile")
        snap = g.snapshot()
        self.assertEqual(snap["devices"]["name:synth-a"]["level"], 1)
        self.assertEqual(snap["refusals"], {"name:synth-a|backoff": 1})
        self.assertFalse(snap["storm_active"])

    def test_reset_forgets_everything(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        _fail(g, "name:synth-a")
        g.note_drop("name:synth-a", "face-track")
        g.note_drop("name:synth-b", "face-track")
        g.reset()
        self.assertTrue(g.begin("name:synth-a", "face-track").allowed)
        self.assertFalse(g.storm_active())


if __name__ == "__main__":
    unittest.main()
