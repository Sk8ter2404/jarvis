"""core/camera_gate.py - DIES ON OPEN (R11).

WHAT THE 2026-09-29 LIVE LOG SHOWED (19:39-19:59, v2.0.134). The Kinect was
"surprise removed" from USB - sensor, camera and mic array together (Windows
Kernel-PnP event 1010) - at 19:39:55, 19:40:12, 19:40:41, 19:40:57, 19:41:42,
19:43:44, 19:48:46 and 19:58:47, and the bridge's stale-stream check logged
"body AND color streams stale > 4s - resetting runtime" about 4 s after each:
19:39:59.6, 19:40:16.4, 19:40:45.4, 19:41:46.9, 19:43:48.8, 19:48:50.4,
19:58:51.9. The gaps are the gate's backoff ladder (30/60/120/300/600 s):
every drop came the moment the gate let the bridge REOPEN the sensor. Nothing
else on the bus dropped, so the storm breaker and the culprit rule never
fired, and the ladder would have reopened it every 10 minutes for the rest of
the session - a USB re-enumeration, and an audio device-list change, each time.

Every test runs the real gate on a frozen clock at the SHIPPED defaults unless
it says otherwise. The drop reports are the live log's; the open times are
inferred (the bridge logs no line for a first-attempt open). Keys other than
the bridge's own "kinect" are synthetic (public repo).

Written to run - and FAIL on what the gate does, not on a TypeError or a
missing name - against a gate without this feature: that is how the fix is
proven against the old code.
"""
from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest

from core import camera_gate as cg

KINECT = "kinect"
BRIDGE = "kinect-bridge"
LEFT = "name:synth-left"
RIGHT = "name:synth-right"
_LABELS = {KINECT: "the Kinect", LEFT: "the left webcam",
           RIGHT: "the right webcam"}

# The shipped numbers, read through getattr so the old gate runs these tests.
WINDOW = getattr(cg, "DIES_ON_OPEN_WINDOW_S", 15.0)
COUNT = getattr(cg, "DIES_ON_OPEN_COUNT", 3)
RETRY = getattr(cg, "DIES_ON_OPEN_RETRY_S", 1800.0)
RETRY_MAX = getattr(cg, "DIES_ON_OPEN_RETRY_MAX_S", 3600.0)
LADDER_CAP = 600.0

_SAID = ("The Kinect drops off USB the moment it starts streaming, sir. That "
         "is usually its power supply. I'll only retry it every thirty "
         "minutes.")


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s

    def at(self, hms: str) -> None:
        """Jump to a wall-clock time of the live session ("19:40:45.437")."""
        self.t = _hms(hms, not_before=self.t)


def _hms(hms: str, not_before: float = 0.0) -> float:
    h, m, s = hms.split(":")
    t = int(h) * 3600 + int(m) * 60 + float(s)
    assert t >= not_before, f"the clock only moves forward ({hms})"
    return t


def _gate(clock, **kw):
    """A gate at the shipped defaults. Knobs the gate does not take are
    dropped, so the old gate builds too."""
    logs: list = []
    spoken: list = []
    params = inspect.signature(cg.CameraGate).parameters
    kw = {k: v for k, v in kw.items() if k in params}
    kw.setdefault("log", logs.append)
    kw.setdefault("announce", spoken.append)
    kw.setdefault("labeler", lambda key: _LABELS.get(key, ""))
    return cg.CameraGate(clock=clock, **kw), logs, spoken


def _slow(g, key) -> bool:
    fn = getattr(g, "dies_on_open", None)
    return bool(fn(key)) if callable(fn) else False


def _dev(g, key) -> dict:
    return g.snapshot()["devices"].get(key, {})


class _Bridge:
    """What audio/kinect_bridge.py does with the gate, on the test's clock:
    begin() before an open, end() after it, hold() while streaming,
    note_frame() while frames arrive, and unhold() + note_drop() when both
    streams have been stale for 4 s. Asks again when the gate says to, but at
    least once a minute (the bridge's re-ask cap)."""

    def __init__(self, g, clk, key=KINECT, comp=BRIDGE):
        self.g, self.clk, self.key, self.comp = g, clk, key, comp
        self.opens: list = []

    def open_when_allowed(self, until: float = float("inf"),
                          took: float = 0.3) -> bool:
        for _ in range(20000):                 # never an endless test
            if self.clk.t >= until:
                break
            d = self.g.begin(self.key, self.comp)
            if d.allowed:
                self.clk.advance(took)
                self.g.end(self.key, self.comp, True)
                self.g.hold(self.key, self.comp)
                self.opens.append(self.clk.t)
                return True
            self.clk.advance(min(max(0.5, d.wait_s), 60.0,
                                 max(0.5, until - self.clk.t)))
        return False

    def stream(self, seconds: float, every: float = 1.0) -> None:
        end = self.clk.t + seconds
        while self.clk.t < end:
            self.clk.advance(min(every, end - self.clk.t))
            self.g.note_frame(self.key)

    def stale_reset(self) -> None:
        self.g.unhold(self.key, self.comp)
        self.g.note_drop(self.key, self.comp)

    def dies_on_open(self, until: float = float("inf")) -> bool:
        """Opens when allowed, one frame, gone ~1 s later, stale-reset 4 s
        after that - the 2026-09-29 pattern."""
        if not self.open_when_allowed(until):
            return False
        self.clk.advance(0.5)
        self.g.note_frame(self.key)
        self.clk.advance(3.5)
        self.stale_reset()
        return True


class LiveReplayTests(unittest.TestCase):
    """The 2026-09-29 19:39-19:59 sequence, step by step."""

    def setUp(self):
        self.clk = _Clock(0.0)
        self.g, self.logs, self.spoken = _gate(self.clk)
        self.b = _Bridge(self.g, self.clk)

    def _first_three(self):
        clk, g, b = self.clk, self.g, self.b
        # Boot: the bridge opens the sensor; it streams for under a second.
        clk.at("19:39:53.7")
        self.assertTrue(b.open_when_allowed(until=clk.t + 1.0))
        clk.at("19:39:54.5")
        g.note_frame(KINECT)
        clk.at("19:39:59.605")                 # live: gone from USB 19:39:55
        b.stale_reset()
        # The first recovery is immediate; its stream verify waits out the
        # re-enumeration and succeeds.
        clk.at("19:39:59.7")
        self.assertTrue(g.begin(KINECT, BRIDGE).allowed)
        clk.at("19:40:11.4")
        g.end(KINECT, BRIDGE, True)
        g.hold(KINECT, BRIDGE)
        clk.at("19:40:11.9")
        g.note_frame(KINECT)
        clk.at("19:40:16.447")                 # live: gone from USB 19:40:12
        b.stale_reset()
        # The ladder's 30 s rung is what let the live 19:40:41 reopen through.
        clk.at("19:40:41.3")
        self.assertEqual(g.begin(KINECT, BRIDGE).reason, "backoff")
        clk.at("19:40:41.5")
        self.assertTrue(b.open_when_allowed(until=clk.t + 1.0))
        clk.at("19:40:42.0")
        g.note_frame(KINECT)
        clk.at("19:40:45.437")                 # live: gone from USB 19:40:41
        b.stale_reset()

    def test_after_three_dies_on_open_the_retry_gap_grows_past_ten_minutes(self):
        self._first_three()
        wait, reason = self.g.retry_in(KINECT, BRIDGE)
        self.assertEqual(reason, "backoff")
        self.assertGreater(
            wait, LADDER_CAP,
            f"the sensor would be reopened {wait:.0f}s after its third "
            f"die-on-open - the 19:41:42 / 19:43:44 / 19:48:46 / 19:58:47 "
            f"re-enumerations again")
        self.assertAlmostEqual(wait, RETRY, delta=1.0)
        d = self.g.check(KINECT, BRIDGE)
        self.assertIn("each died within 15s", d.detail)
        self.assertIn(d.reason, cg.HOLD_REASONS)

    def test_the_rest_of_the_session_reopens_it_twice_an_hour_at_most(self):
        self._first_three()
        third = self.b.opens[-1]
        while self.b.dies_on_open(until=_hms("21:40:00")):
            pass
        later = self.b.opens[self.b.opens.index(third) + 1:]
        # The old ladder reopened it at 19:41:41, 19:43:4x, 19:48:4x,
        # 19:58:4x and every 10 minutes after that: 13 by 21:40.
        self.assertLessEqual(len(later), 2, [round(t) for t in later])
        self.assertTrue(later, "a device on the slow retry must still be "
                               "retried - this is not a quarantine")
        self.assertGreaterEqual(later[0], _hms("20:10:45.4"),
                                "reopened before the 30 min slow retry ran out")
        gaps = [b - a for a, b in zip([third] + later, later)]
        self.assertTrue(all(g > LADDER_CAP for g in gaps), gaps)
        self.assertAlmostEqual(gaps[0], RETRY + 4.0, delta=2.0)
        if len(gaps) > 1:
            self.assertAlmostEqual(gaps[1], RETRY_MAX + 4.0, delta=2.0,
                                   msg="the next one doubles to the max")
        snap = self.g.snapshot()
        self.assertEqual(snap["devices"][KINECT]["dies_on_open"],
                         COUNT + len(later))
        self.assertEqual(snap["devices"][KINECT]["slow_retry_s"], RETRY_MAX)

    def test_the_owner_is_told_once_plainly(self):
        self._first_three()
        self.assertEqual(self.spoken, [_SAID])
        before = len(self.b.opens)
        while self.b.dies_on_open(until=_hms("23:00:00")):
            pass
        later = len(self.b.opens) - before
        self.assertEqual(later, 3)
        self.assertEqual(self.spoken, [_SAID], "said again on a later raise")
        raised = [ln for ln in self.logs if "opens in a row" in ln]
        self.assertEqual(len(raised), 1 + later, "one log line per raise")
        self.assertIn("kinect (the Kinect)", raised[0])
        self.assertIn("every 30 min instead of every 10 min", raised[0])
        self.assertIn("every 60 min", raised[-1])
        self.assertIn("'use the Kinect again'", raised[0])
        snap = self.g.snapshot()
        self.assertEqual(snap["dies_on_open"],
                         {KINECT: {"label": "the Kinect",
                                   "count": snap["devices"][KINECT][
                                       "dies_on_open"],
                                   "retry_s": RETRY_MAX}})

    def test_it_is_not_a_storm_and_not_a_quarantine(self):
        """R8 / R8b are about the SHARED bus; one device dying on its own
        open trips neither."""
        self._first_three()
        while self.b.dies_on_open(until=_hms("21:40:00")):
            pass
        snap = self.g.snapshot()
        self.assertEqual(snap["storm_trips"], 0)
        self.assertEqual(snap["quarantined"], {})
        self.assertEqual(snap["devices"][KINECT]["culprit_strikes"], 0)
        self.assertTrue(_slow(self.g, KINECT))
        self.assertFalse(self.g.quarantined(KINECT))


class CountRuleTests(unittest.TestCase):

    def setUp(self):
        self.clk = _Clock()
        self.g, self.logs, self.spoken = _gate(self.clk)
        self.b = _Bridge(self.g, self.clk)

    def test_a_normal_long_stream_resets_the_count(self):
        b, g = self.b, self.g
        for _ in range(COUNT - 1):
            self.assertTrue(b.dies_on_open())
        self.assertEqual(_dev(g, KINECT).get("dies_on_open"), COUNT - 1)
        # A reopen that streams normally for five minutes, then drops.
        self.assertTrue(b.open_when_allowed())
        b.stream(300.0)
        b.stale_reset()
        self.assertEqual(_dev(g, KINECT).get("dies_on_open"), 0)
        # So it takes a FULL run again to reach the slow retry.
        for _ in range(COUNT - 1):
            self.assertTrue(b.dies_on_open())
        self.assertFalse(_slow(g, KINECT))
        self.assertLessEqual(g.retry_in(KINECT, BRIDGE)[0], LADDER_CAP)
        self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT), "a full run did not slow it down")
        self.assertAlmostEqual(g.retry_in(KINECT, BRIDGE)[0], RETRY, delta=1.0)

    def test_a_stream_that_proves_itself_clears_the_slow_retry(self):
        b, g = self.b, self.g
        for _ in range(COUNT):
            self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT))
        self.assertTrue(b.open_when_allowed())       # after the 30 min
        b.stream(WINDOW + 5.0)                        # the power was fixed
        self.assertFalse(_slow(g, KINECT))
        self.assertEqual(_dev(g, KINECT)["dies_on_open"], 0)
        self.assertEqual(g.snapshot()["dies_on_open"], {})
        self.assertEqual(
            sum("no longer dies on open" in ln for ln in self.logs), 1)
        # A later single die-on-open is back on the normal ladder.
        b.stale_reset()
        self.assertTrue(b.dies_on_open())
        self.assertFalse(_slow(g, KINECT))
        self.assertLessEqual(g.retry_in(KINECT, BRIDGE)[0], LADDER_CAP)

    def test_a_device_that_streams_five_minutes_then_drops_is_not_counted(self):
        b, g = self.b, self.g
        waits = []
        for _ in range(8):
            self.assertTrue(b.open_when_allowed())
            b.stream(300.0)
            b.stale_reset()
            waits.append(g.retry_in(KINECT, BRIDGE)[0])
        self.assertEqual(_dev(g, KINECT).get("dies_on_open", 0), 0)
        self.assertFalse(_slow(g, KINECT))
        self.assertTrue(all(w <= LADDER_CAP for w in waits), waits)
        self.assertFalse(any("opens in a row" in ln for ln in self.logs))
        self.assertEqual(self.spoken, [])

    def test_without_frame_reports_the_drop_onset_still_tells_them_apart(self):
        """A caller that never reports frames: five minutes between the open
        and the drop is a stream that lived, not one that died on arrival."""
        g = self.g
        for _ in range(COUNT + 2):
            self.assertTrue(self.b.open_when_allowed())
            self.clk.advance(300.0)
            self.b.stale_reset()
        self.assertFalse(_slow(g, KINECT))

    def test_a_frame_between_a_drop_and_the_next_open_does_not_clear_it(self):
        b, g = self.b, self.g
        for _ in range(COUNT - 1):
            self.assertTrue(b.dies_on_open())
            self.clk.advance(WINDOW + 5.0)
            g.note_frame(KINECT)            # a stray, late frame: no stream
        self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT))

    def test_the_onset_not_the_report_is_measured_and_any_camera_counts(self):
        """A webcam's read-failure burst takes ~20 s to be reported, but it
        began a second after the stream started."""
        g = self.g
        for _ in range(COUNT):
            for _i in range(100):
                d = g.begin(LEFT, "face-track")
                if d.allowed:
                    break
                self.clk.advance(min(d.wait_s, 60.0))
            self.assertTrue(d.allowed, d)
            self.clk.advance(0.8)
            g.end(LEFT, "face-track", True)
            ok_at = self.clk.t
            self.clk.advance(20.0)
            g.note_drop(LEFT, "face-track", onset=ok_at + 1.0,
                        cause="read-failure burst")
        self.assertTrue(_slow(g, LEFT))
        self.assertEqual(
            self.spoken,
            ["The left webcam drops off USB the moment it starts streaming, "
             "sir. That is usually its power supply. I'll only retry it every "
             "thirty minutes."])

    def test_one_failed_read_is_not_a_dead_stream(self):
        g = self.g
        for _ in range(COUNT + 1):
            for _i in range(100):
                d = g.begin(RIGHT, "side-tile")
                if d.allowed:
                    break
                self.clk.advance(min(d.wait_s, 60.0))
            g.end(RIGHT, "side-tile", True)
            self.clk.advance(1.0)
            g.note_drop(RIGHT, "side-tile", minor=True)
        self.assertEqual(_dev(g, RIGHT).get("dies_on_open", 0), 0)
        self.assertFalse(_slow(g, RIGHT))

    def test_a_drop_with_no_successful_open_is_not_counted(self):
        g = self.g
        for _ in range(COUNT + 1):
            g.note_drop(LEFT, "face-track")
            self.clk.advance(30.0)
            g.note_frame(LEFT)
        self.assertFalse(_slow(g, LEFT))

    def test_a_second_device_gets_its_own_notice_once(self):
        g = self.g
        for _ in range(COUNT):
            self.assertTrue(self.b.dies_on_open())
        # A minute apart: two devices dropping within 10 s would be a USB
        # storm (R8), which is not what this test is about.
        self.clk.advance(60.0)
        other = _Bridge(g, self.clk, key=LEFT, comp="face-track")
        for _ in range(COUNT):
            self.assertTrue(other.dies_on_open())
        self.assertEqual(len(self.spoken), 2, self.spoken)
        self.assertTrue(self.spoken[1].startswith("The left webcam drops off"))
        self.assertEqual(sorted(g.dies_on_open()), sorted([KINECT, LEFT]))


class OwnerLiftTests(unittest.TestCase):
    """'use the Kinect again' (camera_unquarantine -> lift_quarantine)."""

    def setUp(self):
        self.clk = _Clock()
        self.g, self.logs, self.spoken = _gate(self.clk)
        self.b = _Bridge(self.g, self.clk)
        for _ in range(COUNT):
            self.assertTrue(self.b.dies_on_open())

    def test_the_lift_clears_it_and_allows_an_open_now(self):
        g = self.g
        self.assertTrue(_slow(g, KINECT))
        self.assertEqual(g.begin(KINECT, BRIDGE).reason, "backoff")
        self.assertTrue(g.lift_quarantine(KINECT))
        self.assertFalse(_slow(g, KINECT))
        # One short of the verdict (2026-10-01): the next death re-arms it.
        self.assertEqual(_dev(g, KINECT)["dies_on_open"], COUNT - 1)
        self.assertTrue(g.begin(KINECT, BRIDGE).allowed,
                        "the owner said to use it again; the 30 min hold "
                        "still stood")
        self.assertTrue(any("slow dies-on-open retry is cleared" in ln
                            for ln in self.logs), self.logs)
        self.assertFalse(g.lift_quarantine(KINECT), "nothing left to lift")

    def test_after_a_lift_one_more_death_re_arms_it_and_is_not_said_again(self):
        # B083 (2026-10-01): the lift reply promises "if it still drops off,
        # I'll go back to retrying it every thirty minutes" - but a zeroed run
        # took THREE more deaths (now, +2 min, +5 min on the ladder) first.
        g, b = self.g, self.b
        self.assertTrue(g.lift_quarantine(KINECT))
        self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT),
                        "one death after the lift did not re-arm the retry")
        self.assertEqual(_dev(g, KINECT)["slow_retry_s"], RETRY,
                         "back to the BASE retry, as the reply says")
        self.assertEqual(self.spoken, [_SAID], "once per session")

    def test_a_lift_of_a_partial_run_still_starts_over(self):
        clk = _Clock()
        g, _logs, _spoken = _gate(clk)
        b = _Bridge(g, clk)
        self.assertTrue(b.dies_on_open())             # 1 of COUNT, not slow
        self.assertFalse(_slow(g, KINECT))
        g.lift_quarantine(KINECT)
        self.assertEqual(_dev(g, KINECT)["dies_on_open"], 0)

    def test_the_lift_leaves_a_culprit_quarantine_rule_intact(self):
        """lift_quarantine still lifts a quarantine exactly as before."""
        clk = _Clock()
        g, _l, _s = _gate(clk, culprit_threshold=1, storm_cooldown_s=60.0)
        g.begin(LEFT, "face-track")
        clk.advance(0.8)
        g.end(LEFT, "face-track", True)
        clk.advance(0.5)
        g.note_drop(LEFT, "face-track")
        g.note_drop(RIGHT, "face-track")
        self.assertTrue(g.quarantined(LEFT))
        self.assertFalse(_slow(g, LEFT))
        self.assertTrue(g.lift_quarantine(LEFT))
        self.assertFalse(g.quarantined(LEFT))


class RememberedAcrossRestartsTests(unittest.TestCase):
    """B083 (2026-10-01): the verdict lived in memory, so every restart cost 3
    more USB drops of the Kinect and its mic array in ~45 s, and the same
    spoken warning - on every one of 14 restarts in a day. Merged with
    live-logs B056, whose doo_state_path is the ONE persistence (see
    RestartTests): a restart inside the hold reopens nothing, after it ONE
    drop re-arms the retry. Each "process" is a fresh gate on the same file;
    the owner's lift forgets the saved run (B083's lift keeps it one death
    short in memory)."""

    def setUp(self):
        import shutil
        self.dir = tempfile.mkdtemp(prefix="jarvis_doo_persist_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "camera_gate_doo.json")
        self.clk = _Clock(1000.0)

    def _process(self):
        g, logs, spoken = _gate(self.clk, doo_state_path=self.path)
        return g, _Bridge(g, self.clk), logs, spoken

    def _saved(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)["devices"]

    def test_a_restart_needs_one_drop_not_three(self):
        g, b, _l, spoken = self._process()
        for _ in range(COUNT):
            self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT))
        self.assertEqual(spoken, [_SAID])
        g2, b2, _l2, spoken2 = self._process()
        self.assertEqual(g2.begin(KINECT, BRIDGE).reason, "backoff",
                         "a restart inside the hold reopened the sensor")
        self.assertTrue(b2.dies_on_open())     # waits out the hold, then dies
        self.assertEqual(len(b2.opens), 1, "the restart relearned the verdict")
        self.assertTrue(_slow(g2, KINECT))
        self.assertEqual(spoken2, [], "the same warning again")

    def test_a_lift_forgets_it_and_one_more_death_re_arms_it(self):
        g, b, _l, _s = self._process()
        for _ in range(COUNT):
            b.dies_on_open()
        self.assertIn(KINECT, self._saved())
        self.assertTrue(g.lift_quarantine(KINECT))
        self.assertNotIn(KINECT, self._saved())
        self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT))
        self.assertIn(KINECT, self._saved())


class KnobTests(unittest.TestCase):

    def test_the_shipped_numbers(self):
        self.assertEqual(getattr(cg, "DIES_ON_OPEN_WINDOW_S", None), 15.0)
        self.assertEqual(getattr(cg, "DIES_ON_OPEN_COUNT", None), 3)
        self.assertEqual(getattr(cg, "DIES_ON_OPEN_RETRY_S", None), 1800.0)
        self.assertEqual(getattr(cg, "DIES_ON_OPEN_RETRY_MAX_S", None), 3600.0)

    def test_zero_disables_it(self):
        clk = _Clock()
        g, logs, spoken = _gate(clk, dies_on_open_retry_s=0.0)
        b = _Bridge(g, clk)
        for _ in range(COUNT + 3):
            self.assertTrue(b.dies_on_open())
        self.assertFalse(_slow(g, KINECT))
        self.assertLessEqual(g.retry_in(KINECT, BRIDGE)[0], LADDER_CAP)
        self.assertEqual(spoken, [])

    def test_the_retry_is_the_owner_knob_and_a_higher_one_raises_the_max(self):
        clk = _Clock()
        g, _l, spoken = _gate(clk, dies_on_open_retry_s=7200.0)
        b = _Bridge(g, clk)
        for _ in range(COUNT):
            self.assertTrue(b.dies_on_open())
        self.assertAlmostEqual(g.retry_in(KINECT, BRIDGE)[0], 7200.0, delta=1.0)
        self.assertTrue(b.dies_on_open())
        self.assertAlmostEqual(g.retry_in(KINECT, BRIDGE)[0], 7200.0, delta=1.0)
        self.assertIn("every 2 hours", spoken[0])

    def test_garbage_knobs_fall_back_to_the_shipped_defaults(self):
        params = inspect.signature(cg.CameraGate).parameters
        self.assertIn("dies_on_open_retry_s", params)
        g = cg.CameraGate(dies_on_open_retry_s="x",
                          dies_on_open_window_s=float("nan"),
                          dies_on_open_count="y")
        self.assertEqual(g.dies_on_open_retry_s, RETRY)
        self.assertEqual(g.dies_on_open_window_s, WINDOW)
        self.assertEqual(g.dies_on_open_count, COUNT)

    def test_reset_forgets_it(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        b = _Bridge(g, clk)
        for _ in range(COUNT):
            self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT))
        g.reset()
        self.assertFalse(_slow(g, KINECT))
        self.assertTrue(g.begin(KINECT, BRIDGE).allowed)


class RestartTests(unittest.TestCase):
    """2026-10-01: the run survives a restart (doo_state_path). Live: every
    one of the 14 restarts on 2026-09-30 reopened the Kinect three more times
    (three USB re-enumerations, an audio device-list change) and SPOKE the
    three-sentence warning again - 09:50, 10:13, 10:47, 11:00, ... 15:46."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "camera_gate_doo.json")
        self.clk = _Clock(1000.0)

    def _new_process(self, **kw):
        g, logs, spoken = _gate(self.clk, doo_state_path=self.path, **kw)
        return g, logs, spoken, _Bridge(g, self.clk)

    def _trip(self):
        g, _logs, spoken, b = self._new_process()
        for _ in range(COUNT):
            self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT))
        self.assertEqual(spoken, [_SAID])
        return g

    def test_a_restart_inside_the_hold_does_not_reopen_it(self):
        self._trip()
        self.clk.advance(20 * 60)                  # a deploy 20 min later
        g, logs, spoken, b = self._new_process()
        self.assertFalse(b.open_when_allowed(until=self.clk.t + 5 * 60),
                         "the restart reopened a sensor still on its hold")
        self.assertEqual(b.opens, [])
        self.assertEqual(g.begin(KINECT, BRIDGE).reason, "backoff")
        self.assertTrue(_slow(g, KINECT))
        self.assertTrue(any("restored" in ln for ln in logs), logs)
        self.assertEqual(spoken, [])

    def test_after_the_hold_one_more_death_re_arms_it_and_is_not_said(self):
        self._trip()
        self.clk.advance(20 * 60)
        g, _logs, spoken, b = self._new_process()
        self.assertTrue(b.dies_on_open())          # waits out the hold first
        self.assertEqual(len(b.opens), 1, "one open, not a fresh run of three")
        self.assertTrue(_slow(g, KINECT))
        self.assertAlmostEqual(g.retry_in(KINECT, BRIDGE)[0], RETRY_MAX,
                               delta=1.0)
        self.assertEqual(spoken, [], "said again after a restart")

    def test_the_warning_may_be_said_again_after_twelve_hours(self):
        self._trip()
        self.clk.advance(13 * 3600)
        g, _logs, spoken, b = self._new_process()
        self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT))
        self.assertEqual(len(spoken), 1)

    def test_a_normal_stream_clears_the_saved_run(self):
        self._trip()
        self.clk.advance(2 * 3600)
        g, _logs, _spoken, b = self._new_process()
        self.assertTrue(b.open_when_allowed())
        b.stream(60.0)
        self.assertFalse(_slow(g, KINECT))
        g2, _l, _s, _b = self._new_process()
        self.assertFalse(_slow(g2, KINECT))
        self.assertTrue(g2.begin(KINECT, BRIDGE).allowed)

    def test_the_owners_lift_clears_the_saved_run(self):
        g = self._trip()
        self.assertTrue(g.lift_quarantine(KINECT))
        g2, _l, _s, _b = self._new_process()
        self.assertFalse(_slow(g2, KINECT))
        self.assertTrue(g2.begin(KINECT, BRIDGE).allowed)

    def test_a_saved_hold_is_capped_against_a_clock_jump(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "devices": {KINECT: {
                "count": COUNT, "retry_s": RETRY,
                "until": self.clk.t + 10 * 86400, "said_at": 0.0}}}, f)
        g, _l, _s, _b = self._new_process()
        self.assertTrue(_slow(g, KINECT))
        self.assertLessEqual(g.retry_in(KINECT, BRIDGE)[0], RETRY_MAX + 1.0)

    def test_no_path_writes_nothing_and_a_garbage_file_is_ignored(self):
        g, _l, _s = _gate(self.clk)
        b = _Bridge(g, self.clk)
        for _ in range(COUNT):
            self.assertTrue(b.dies_on_open())
        self.assertEqual(os.listdir(self.tmp), [])
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("not json")
        g2, _l2, _s2, _b2 = self._new_process()
        self.assertTrue(g2.begin(KINECT, BRIDGE).allowed)


class StormSemanticsTests(unittest.TestCase):
    """The drop that counts toward dies-on-open is still a drop to R8."""

    def test_a_die_on_open_during_probation_still_re_trips_the_breaker(self):
        clk = _Clock()
        g, _l, _s = _gate(clk)
        b = _Bridge(g, clk)
        self.assertTrue(b.open_when_allowed())
        b.stream(120.0)
        g.note_drop(LEFT, "face-track")
        b.stale_reset()                            # two devices: a storm
        self.assertTrue(g.storm_active())
        clk.advance(g.snapshot()["storm_remaining_s"] + 1.0)
        self.assertFalse(g.storm_active())
        self.assertTrue(b.open_when_allowed())
        clk.advance(2.0)
        b.g.unhold(KINECT, BRIDGE)
        self.assertTrue(g.note_drop(KINECT, BRIDGE),
                        "a drop on probation must still re-trip it")
        self.assertEqual(_dev(g, KINECT).get("dies_on_open"), 1)


if __name__ == "__main__":
    unittest.main()
