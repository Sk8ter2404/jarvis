"""The side-tile AMPLIFIER is counted, and the count is what makes a
sick-camera soak checkable (2026-09-06).

WHY THIS FILE EXISTS. On 2026-09-06 a soak reported:

    "Under the full amplifier condition (camera permanently unavailable,
     producer retrying opens forever) mfksproxy was 2 -> 2, +0.000/min over
     17.2 min."

None of that happened. At 04:31:14 the preflight had already logged "camera
index 0: failed to open in 2.0s - marking bad" and "dropped 1 bad camera
index/es; 1 remain", and ``_preflight_cameras`` removes a bad index from
``CAMERAS`` for the whole session. ``_kinect_preview_webcam_names()`` is
DERIVED from ``CAMERAS`` - deliberately, so a camera swap is edited in one
place - so the dropped camera's slot no longer has a name to resolve, and
``_read_side_tile_webcams`` takes its ``idx is None`` branch: no enumeration,
no open, no read, and therefore no ``_invalidate_side_tile_indices()``. The run
measured an idle one-camera JARVIS and reported it as the amplifier.

The independent evidence agreed and still did not stop the claim: the tile
sampler recorded the right tile changing 1 sample out of 1108 (0.1%), median
age 806 s, max 1503 s over the whole 23.4-minute window. Nothing IN the process
could contradict it, because nothing in the process counted. That is the defect
these tests close: after this, ``get_side_tile_gate_stats()['invalidations']``
is the precondition every sick-camera soak has to show, and the console line
``[kinect-preview] side-tile AMPLIFIER active: ...`` is its external form in
the session log.

WHAT IS PINNED HERE
  1. The counters count the three distinct events, and only those.
  2. THE VACUITY GUARD: the exact 04:31 configuration - camera dropped from
     CAMERAS - produces ZERO invalidations however long the tile loop runs,
     while the same loop with the camera present produces one per failed read.
     A soak run in the first shape cannot measure the amplifier, and this test
     is the machine-checked form of that sentence.
  3. The report line prints only when the amplifier actually fired, quotes a
     rate over its OWN window, and never raises.

Like tests/monolith/test_monolith_dshow_enum_leak.py, this file asserts CALL
and EVENT COUNTS, never thread or handle counts: those are facts about the
machine, and a test that asserts them is flaky by construction.
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


def _fp(devices=(("usb#vid_1&pid_1", b"\x01" * 8),),
        software=("OBS Virtual Camera",)):
    """A fingerprint shaped like the real one: ``((pnp...), (software...))``."""
    return (tuple(devices), tuple(software))


class _SickCap:
    """Opens fine, delivers ONE frame, then every read fails.

    The one proven frame is load-bearing, not a softening. Since 2026-09-05
    ``_open_tile_capture`` opens through ``_camera_open(..., require_frame=...)``
    because a Media Foundation open of a camera another process holds reports
    ``isOpened()`` True and then delivers nothing. A fixture that never reads is
    therefore refused AT OPEN, which reaches ``cap is None`` -> placeholder and
    never touches ``_invalidate_side_tile_indices()`` at all. Only a camera that
    opens and THEN stops delivering is the amplifier."""

    def __init__(self):
        self.releases = 0
        self._first = True

    def isOpened(self):
        return True

    def set(self, *_a):
        return True

    def get(self, *_a):
        return 0.0

    def read(self, *_a):
        if self._first:
            self._first = False
            import numpy as _np
            return True, _np.zeros((8, 8, 3), dtype=_np.uint8)
        return False, None

    def release(self):
        self.releases += 1


class _Cv2Shim:
    """Stands in for the module-level ``cv2`` inside the open path and records
    every VideoCapture()."""

    CAP_DSHOW = 700
    CAP_MSMF = 1400
    CAP_ANY = 0
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, log, cap_factory):
        self.log = log
        self._cap_factory = cap_factory

    def VideoCapture(self, idx, backend=None):
        self.log.append((idx, backend))
        return self._cap_factory()


@requires_monolith
class SideTileAmplifierCounterTests(MonolithGlobalsTestCase):
    """The counters, and the vacuity guard that is the whole point of them."""

    CAM = {"index": 0, "label": "Left webcam", "name": "left cam",
           "primary": True, "look_x": 0.5, "look_y": 0.5}

    def setUp(self):
        bc = self.bc
        # The face-track fast path is checked FIRST in _read_side_tile_webcams,
        # and a real frame left there by another monolith test - stamped with
        # real wall-clock time, against this class's frozen 1000.0 clock - reads
        # as infinitely fresh and makes every test here measure zero opens.
        # Green for exactly the reason this file exists. Snapshot and clear.
        self._saved_frames = dict(bc._camera_latest_frame)
        self._saved_frame_at = dict(bc._camera_last_frame_at)
        bc._camera_latest_frame.clear()
        bc._camera_last_frame_at.clear()
        self._reset_tiles()
        bc.CAMERAS[:] = [dict(self.CAM)]
        bc._kinect_preview_webcam_idx.clear()
        bc._kinect_preview_webcam_resolved[0] = False
        bc._kinect_preview_webcam_resolved_at[0] = 0.0
        bc._kinect_preview_webcam_fingerprint[0] = None
        bc._kinect_preview_webcam_enumerated_at[0] = 0.0
        self._reset_counts()

    def tearDown(self):
        bc = self.bc
        self._reset_tiles()
        bc._camera_latest_frame.clear()
        bc._camera_latest_frame.update(self._saved_frames)
        bc._camera_last_frame_at.clear()
        bc._camera_last_frame_at.update(self._saved_frame_at)

    def _reset_tiles(self):
        bc = self.bc
        for slot in ("left", "right"):
            bc._kinect_tile_caps[slot] = None
            bc._kinect_tile_frames[slot] = None
            bc._kinect_tile_last_read[slot] = 0.0

    def _reset_counts(self):
        bc = self.bc
        for k in bc._side_tile_gate_counts:
            bc._side_tile_gate_counts[k] = 0
        bc._side_tile_gate_window[:] = [0.0, 0, 0, 0, 0]

    def _drive(self, ticks, devices=("Left Cam",), t0=1000.0, step=0.25,
               keep_fast_path_fresh=False):
        """Run `ticks` real composite ticks and return (enumerations, opens).

        `step` is _KINECT_PREVIEW_TILE_READ_INTERVAL, the only throttle the code
        imposes on this path.

        `keep_fast_path_fresh` re-stamps the face-track loop's cached frame each
        tick. Without it the fast path only serves the first
        _KINECT_PREVIEW_TILE_REUSE_MAX_AGE (2.0 s = 8 ticks) and then decides
        the producer has stalled - which is correct behaviour and exactly what
        the healthy-path test must NOT accidentally measure."""
        bc = self.bc
        enums, opens = [], []
        t = [t0]

        def _enumerate():
            enums.append(1)
            return list(devices)

        shim = _Cv2Shim(opens, _SickCap)
        buf = io.StringIO()
        with mock.patch.object(bc, "_enumerate_dshow_input_devices", _enumerate), \
             mock.patch.object(bc, "_video_device_fingerprint",
                               return_value=_fp()), \
             mock.patch.object(bc, "cv2", shim), \
             mock.patch.object(bc.time, "time", side_effect=lambda: t[0]), \
             contextlib.redirect_stdout(buf):
            for _ in range(ticks):
                if keep_fast_path_fresh:
                    bc._camera_last_frame_at[self.CAM["index"]] = t[0]
                bc._read_side_tile_webcams(t[0])
                t[0] += step
        self._out = buf.getvalue()
        return len(enums), len(opens)

    # ── the counters ────────────────────────────────────────────────────────

    def test_every_failed_side_tile_read_counts_one_invalidation(self):
        """The amplifier's own definition: one forced re-resolve per failed
        read. 40 ticks of a camera that opens, gives one frame and then fails
        every read is 40 reopen cycles, so 40 invalidations."""
        _enums, opens = self._drive(40)
        self.assertEqual(opens, 40, "precondition: 40 reopen cycles")
        self.assertEqual(
            self.bc.get_side_tile_gate_stats()["invalidations"], 40,
            "the amplifier fired %d times for 40 failed reads"
            % self.bc.get_side_tile_gate_stats()["invalidations"])

    def test_the_gate_absorbs_them_and_says_how_many(self):
        """What the fix actually buys, in the counters rather than in prose:
        of 40 forced re-resolves exactly ONE reaches the leaky enumeration (the
        cold start, before there is a fingerprint to compare against) and the
        other 39 are absorbed by the free device fingerprint."""
        enums, _opens = self._drive(40)
        s = self.bc.get_side_tile_gate_stats()
        self.assertEqual(enums, 1, "the gate stopped working")
        self.assertEqual(s["enumerations"], 1)
        self.assertEqual(s["suppressed"], 39)
        self.assertEqual(s["invalidations"], s["suppressed"] + s["enumerations"],
                         "every invalidation must land in exactly one bucket, "
                         "otherwise the accounting hides a path: %r" % (s,))

    def test_leak_damage_is_derived_from_enumerations_not_invalidations(self):
        """The damage estimate must quote what LEAKED, not what fired. Counting
        invalidations as leaks would restate the pre-fix rate forever and make
        the gate look broken; counting nothing would restate the false claim
        this file exists for."""
        self._drive(40)
        s = self.bc.get_side_tile_gate_stats()
        self.assertEqual(s["leaked_threads"], s["enumerations"])
        self.assertEqual(s["leaked_handles"],
                         s["enumerations"] * self.bc._DSHOW_ENUM_LEAK_HANDLES)
        self.assertLess(s["leaked_threads"], s["invalidations"],
                        "the gate absorbed nothing: %r" % (s,))

    def test_an_enumeration_that_failed_at_the_import_is_not_counted_as_a_leak(self):
        """``_enumerate_dshow_input_devices()`` returning None means pygrabber
        was absent and the call died at the IMPORT, before
        CreateClassEnumerator - so nothing leaked. Counting it would inflate the
        exact number this accounting exists to keep honest."""
        bc = self.bc
        t = [1000.0]
        with mock.patch.object(bc, "_enumerate_dshow_input_devices",
                               return_value=None), \
             mock.patch.object(bc, "_video_device_fingerprint",
                               return_value=_fp()), \
             mock.patch.object(bc.time, "time", side_effect=lambda: t[0]):
            bc._resolve_webcam_indices_by_name()
        self.assertEqual(bc.get_side_tile_gate_stats()["enumerations"], 0)

    def test_a_healthy_tile_loop_fires_the_amplifier_zero_times(self):
        """The common case must stay at ZERO of everything, or the counter is
        noise and the report line becomes something an operator learns to
        ignore.

        BOTH slots are populated here on purpose. With only one camera
        configured the OTHER slot has no name to resolve, and
        _read_side_tile_webcams resolves the map before it can discover that -
        so a one-camera rig pays one cold-start enumeration no matter how
        healthy it is. That is real (it is what the dropped-camera session
        pays), but it is not the healthy path, and measuring it here would hide
        the thing this test is for."""
        bc = self.bc
        import numpy as np
        bc.CAMERAS[:] = [dict(self.CAM),
                         {"index": 1, "label": "Right webcam",
                          "name": "right cam", "primary": False,
                          "look_x": 0.5, "look_y": 0.5}]
        for idx in (0, 1):
            bc._camera_latest_frame[idx] = np.zeros((8, 8, 3), dtype=np.uint8)
            bc._camera_last_frame_at[idx] = 1000.0
        t = [1000.0]
        enums, opens = [], []

        def _enumerate():
            enums.append(1)
            return ["Left Cam", "Right Cam"]

        shim = _Cv2Shim(opens, _SickCap)
        with mock.patch.object(bc, "_enumerate_dshow_input_devices", _enumerate), \
             mock.patch.object(bc, "_video_device_fingerprint",
                               return_value=_fp()), \
             mock.patch.object(bc, "cv2", shim), \
             mock.patch.object(bc.time, "time", side_effect=lambda: t[0]), \
             contextlib.redirect_stdout(io.StringIO()):
            for _ in range(40):
                bc._camera_last_frame_at[0] = t[0]
                bc._camera_last_frame_at[1] = t[0]
                bc._read_side_tile_webcams(t[0])
                t[0] += 0.25
        s = bc.get_side_tile_gate_stats()
        self.assertEqual(len(opens), 0, "the fast path stopped serving the tiles")
        self.assertEqual(len(enums), 0,
                         "a healthy pair of tiles must not enumerate DirectShow "
                         "AT ALL - _read_side_tile_webcams resolves lazily "
                         "precisely so the fast path costs nothing")
        self.assertEqual(s["invalidations"], 0)
        self.assertEqual(s["suppressed"], 0)

    # ── THE VACUITY GUARD ───────────────────────────────────────────────────

    def test_a_camera_dropped_from_CAMERAS_cannot_fire_the_amplifier(self):
        """THE 2026-09-06 04:31 CONFIGURATION, pinned.

        ``_preflight_cameras`` drops a camera that failed to open out of
        ``CAMERAS`` for the whole session. ``_kinect_preview_webcam_names()``
        derives the side-tile name map from ``CAMERAS``, so that slot stops
        resolving and ``_read_side_tile_webcams`` takes its ``idx is None``
        branch: no enumeration, no open, no read, no invalidation - forever,
        however sick the device on the bus actually is.

        A soak run in this shape measures an idle JARVIS. It cannot measure the
        amplifier, and that is not a matter of degree: it is exactly zero."""
        bc = self.bc
        bc.CAMERAS[:] = []                      # ← the drop, verbatim
        enums, opens = self._drive(200)
        s = bc.get_side_tile_gate_stats()
        self.assertEqual(opens, 0, "a camera not in CAMERAS was opened")
        # ONE cold-start enumeration, not zero, and it is asserted rather than
        # glossed: _read_side_tile_webcams resolves the name->index map BEFORE
        # it discovers the slot has no name, so a session with a dropped camera
        # still pays exactly one leaked thread + 103 handles at the first tick
        # after the fast path declines. What it never pays again is a second
        # one, because the amplifier that would force it cannot fire.
        self.assertEqual(enums, 1,
                         "expected exactly the cold-start enumeration, got %d"
                         % enums)
        self.assertEqual(
            s["invalidations"], 0,
            "the amplifier is supposed to be UNREACHABLE with the camera "
            "dropped - if this ever becomes non-zero the 2026-09-06 soak's "
            "reasoning changes and the vacuity argument has to be redone")
        # And the resolves that DO happen are paced by the 10 s TTL, not by the
        # 4 Hz tick — which is the whole difference between an idle session and
        # the amplifier. 200 ticks x 0.25 s = 50 s = 5 TTL windows, so 5
        # resolves reach the gate (1 cold-start enumeration + 4 suppressed) out
        # of 200 opportunities. That ratio, not the raw count, is what
        # distinguishes the two regimes.
        ttl_windows = int((200 * 0.25) / bc._WEBCAM_IDX_TTL_SEC)
        self.assertEqual(s["suppressed"] + s["enumerations"], ttl_windows,
                         "resolves are no longer TTL-paced: %r" % (s,))

    def test_the_same_loop_with_the_camera_present_does_fire_it(self):
        """The other half of the guard - without this, the test above would
        also pass if the tile loop had simply stopped working."""
        _enums, opens = self._drive(200)
        self.assertEqual(opens, 200)
        self.assertGreater(self.bc.get_side_tile_gate_stats()["invalidations"], 0)

    # ── the console line, which is the external observable ──────────────────

    def test_the_report_prints_only_after_the_amplifier_actually_fired(self):
        """Silence is the healthy state. A line that printed on a quiet gate
        would be the same lie in the other direction."""
        bc = self.bc
        self.assertFalse(bc._report_side_tile_amplifier(1000.0),
                         "printed before any window was open")
        # Window opened at 1000.0 with zero invalidations; an hour later, still
        # nothing to say.
        self.assertFalse(bc._report_side_tile_amplifier(4600.0))

    def test_the_report_quotes_the_rate_over_its_own_window(self):
        """A burst two hours ago must not be reported as current load: the
        window slides while the amplifier is quiet, so the rate is always over
        the window it is printed for."""
        bc = self.bc
        buf = io.StringIO()
        bc._report_side_tile_amplifier(1000.0)          # open the window
        bc._side_tile_gate_counts["invalidations"] = 240
        bc._side_tile_gate_counts["suppressed"] = 239
        bc._side_tile_gate_counts["enumerations"] = 1
        with contextlib.redirect_stdout(buf):
            printed = bc._report_side_tile_amplifier(1060.0)   # +60 s
        self.assertTrue(printed)
        out = buf.getvalue()
        self.assertIn("AMPLIFIER active", out)
        self.assertIn("240", out)                       # the firings
        self.assertIn("240.0/min", out)                 # 240 in 1.0 min
        self.assertIn("103", out)                       # handles per leak
        # ...and the NEXT window starts empty: no double-counting.
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            self.assertFalse(bc._report_side_tile_amplifier(4660.0))
        self.assertEqual(buf2.getvalue(), "")

    def test_the_first_report_comes_quickly_then_settles(self):
        """A 25-minute soak must not have to wait 5 minutes for its first
        evidence, and a 5-hour session must not be spammed."""
        bc = self.bc
        self.assertLess(bc._SIDE_TILE_GATE_FIRST_REPORT_S,
                        bc._SIDE_TILE_GATE_REPORT_GAP_S)
        bc._report_side_tile_amplifier(1000.0)
        bc._side_tile_gate_counts["invalidations"] = 10
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(bc._report_side_tile_amplifier(
                1000.0 + bc._SIDE_TILE_GATE_FIRST_REPORT_S - 1.0))
            self.assertTrue(bc._report_side_tile_amplifier(
                1000.0 + bc._SIDE_TILE_GATE_FIRST_REPORT_S))
            # second window now needs the LONG gap
            bc._side_tile_gate_counts["invalidations"] = 20
            self.assertFalse(bc._report_side_tile_amplifier(
                1000.0 + bc._SIDE_TILE_GATE_FIRST_REPORT_S
                + bc._SIDE_TILE_GATE_REPORT_GAP_S - 1.0))
            self.assertTrue(bc._report_side_tile_amplifier(
                1000.0 + bc._SIDE_TILE_GATE_FIRST_REPORT_S
                + bc._SIDE_TILE_GATE_REPORT_GAP_S))

    def test_a_clock_that_goes_backwards_reopens_the_window(self):
        """Wall-clock time can step backwards (NTP, sleep/resume). A negative
        window must not produce a negative rate or a division blow-up - it
        re-bases instead."""
        bc = self.bc
        bc._report_side_tile_amplifier(5000.0)
        bc._side_tile_gate_counts["invalidations"] = 5
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(bc._report_side_tile_amplifier(1000.0))
        self.assertEqual(bc._side_tile_gate_window[0], 1000.0)

    def test_the_report_never_raises(self):
        """It runs from the HUD preview compositor's path, which has no error
        handling of its own."""
        bc = self.bc
        for junk in (None, "nope", object(), float("nan")):
            try:
                bc._report_side_tile_amplifier(junk)
            except Exception as exc:      # pragma: no cover - the assertion
                self.fail("raised on %r: %s" % (junk, exc))

    def test_stats_snapshot_is_a_copy(self):
        """A caller (the soak reader, the dashboard) must not be able to zero
        the live counters by mutating what it was handed."""
        bc = self.bc
        s = bc.get_side_tile_gate_stats()
        s["invalidations"] = 999999
        self.assertEqual(bc._side_tile_gate_counts["invalidations"], 0)


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
