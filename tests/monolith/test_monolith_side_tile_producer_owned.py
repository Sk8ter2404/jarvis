"""The side tile must NEVER open a second in-process handle on a camera the
face-track producer is holding (2026-09-06).

WHY THIS FILE EXISTS
────────────────────
A report claimed that *every* producer read failure across three JARVIS
sessions - ten of them - was JARVIS's own 30-minute self-diagnostic sweep, and
recommended leaving the right webcam enabled on the strength of that. JARVIS's
own record contradicts it. ``data/self_diagnostic.json`` timestamps the sweeps
at 23:21:05, 23:50:05, 00:20:05, 00:50:05, 01:20:05, 02:14:28, 02:24:01,
02:53:01, 03:23:01, 03:53:01, 04:23:01, 04:32:56, 04:57:04, 05:26:04 - and
EIGHT right-camera read failures (2026-09-05 22:38:43, 22:41:10, 23:20:21;
2026-09-06 02:13:45, 02:23:17, 04:56:21, 06:01:36, 06:12:37) have NO sweep
before them. The nearest sweep in each case is 37-44 s LATER. What each one
DOES sit behind, by 7-13 s, is ``[kinect-preview] webcam tiles resolved by
name`` - and that is 8 boots out of 8: EVERY boot in the 2026-09-05 22:00 ->
2026-09-06 07:50 window in which the producer actually held the right camera
and the tile had a right slot. No exceptions, in either direction.

That line is printed by ``_resolve_webcam_indices_by_name()``, which
``_read_side_tile_webcams`` only reaches when its fast path declined - i.e.
immediately before it opens its OWN handle on a camera the producer already
holds.

WHAT THAT COSTS, MEASURED (clean room, JARVIS stopped, one process, the real
``_camera_open`` shapes: producer 1280x720 require_frame=2.5, tile 640x480
require_frame=1.0; 3 cycles per camera, 12 s per phase)::

                    producer reads BEFORE     DURING            AFTER (released)
    USB 2.0 Camera  208 ok /   0 fail    1 ok / 210 fail     0 ok / 239 fail
    eMeet C960      211 ok /   0 fail    1 ok / 211 fail     0 ok / 239 fail

The second MSMF open is GRANTED - it satisfies ``require_frame``, so the
prove-a-frame guard does not catch it - takes the stream, and RELEASING it does
not give the stream back. Only a producer release+reopen recovers, which is
exactly the ``read failure #25 … woke via release+reopen`` pair in the logs.
And it is not a sick-camera effect: the healthy eMeet died identically.
``core/camera_backend.py``'s "both backends leave the HOLDER undisturbed" was
measured CROSS-PROCESS; in-process is the opposite.

WHY IT ONLY EVER HIT THE RIGHT TILE, and why that asymmetry is pinned below:
``_read_side_tile_webcams`` runs from inside the PRIMARY camera's branch of the
producer loop, right after the primary's frame is cached - so the primary slot
is always 0 s fresh and always takes the fast path. The NON-primary camera is
read AFTER the composite in the same iteration, so its cache is one whole loop
period old at composite time, and on the first iteration it does not exist at
all.

WHAT IS PINNED HERE
  1. A slot whose camera the producer HOLDS is never opened, however stale the
     cached frame is - zero ``cv2.VideoCapture`` calls, zero enumerations.
  2. The producer-loop ORDERING that made this fire once per boot on the
     non-primary tile and never on the primary one.
  3. A slot the producer has LET GO of (``entry["cap"] is None``) is still
     opened - the gate is about the handle, not about configuration, or a
     camera the producer gave up on could never be shown again.
  4. The tile still shows the producer's cached frame while the producer owns
     the camera, and falls back to the placeholder past
     ``_TILE_PRODUCER_OWNED_MAX_AGE`` rather than displaying a frozen picture.
  5. The decline is COUNTED and SAID. Through those eight boots the open itself
     was invisible: ``_camera_open`` keyed its backend note by ``(index,
     name)``, and the tile's name comes from ``_kinect_preview_webcam_names()``,
     which is derived from ``CAMERAS`` - the same ``(index, name)`` the producer
     had already opened with, so the second open printed nothing at all. That is
     how eight boots of read failures came to be blamed on a sweep that had not
     run yet. The note key has since gained the call-site ``label``, so such an
     open would now announce itself - but a visible duplicate open is still a
     duplicate open, and what this file pins is that it does not happen.

VERIFIED LIVE, not only here. Session ``2026-09-06_07-55-26`` is the first boot
after this gate landed in which the producer held BOTH cameras - the exact
configuration of the eight above. Counting from the moment the producer opened
the right camera:

    pre-fix, 8 boots  right read failures at  7-13 s  AND  59-63 s   (8 of 8)
    07-55-26           right read failures    none, in 120 s and beyond
                       and instead:  "right tile: declined 1 own-handle open(s)"
                       at +0 s, with no "webcam tiles resolved by name" at all

The 7-13 s failure is this gate's. The 59-63 s one is the 30-minute
self-diagnostic sweep - real, but a DIFFERENT opener - and it is closed by the
ownership gate in ``skills/self_diagnostic.py`` (see
``tests/skills/test_self_diagnostic_producer_ownership.py``). A sweep did run in
that session, at 07:56:41, and cost 23.2 ms and zero read failures against the
~3,300 ms device open it used to do. Both openers, same root hazard, two fixes.

Like its sibling ``test_monolith_side_tile_amplifier_counters.py``, this file
asserts CALL and EVENT COUNTS, never thread or handle counts: those are facts
about the machine, and a test that asserts them is flaky by construction.
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


class _StreamingCap:
    """A capture that opens and keeps delivering — the producer's handle."""

    def __init__(self):
        self.releases = 0

    def isOpened(self):
        return True

    def set(self, *_a):
        return True

    def get(self, *_a):
        return 0.0

    def read(self, *_a):
        import numpy as _np
        return True, _np.zeros((8, 8, 3), dtype=_np.uint8)

    def release(self):
        self.releases += 1


class _Cv2Shim:
    """Stands in for the module-level ``cv2`` and records every VideoCapture()."""

    CAP_DSHOW = 700
    CAP_MSMF = 1400
    CAP_ANY = 0
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, log):
        self.log = log

    def VideoCapture(self, idx, backend=None):
        self.log.append((idx, backend))
        return _StreamingCap()


LEFT = {"index": 0, "label": "Left webcam (left monitor)", "name": "left cam",
        "primary": True, "look_x": 0.5, "look_y": 0.5}
RIGHT = {"index": 1, "label": "Right webcam (top of right monitor)",
         "name": "right cam", "primary": False, "look_x": 0.85, "look_y": 0.5}


@requires_monolith
class SideTileProducerOwnedTests(MonolithGlobalsTestCase):

    def setUp(self):
        bc = self.bc
        # The fast path is checked FIRST, and a real frame left behind by
        # another monolith test - stamped with real wall-clock time, against
        # this class's frozen 1000.0 clock - reads as infinitely fresh and makes
        # every test here measure zero opens. Green for exactly the reason this
        # file exists. Snapshot and clear.
        self._saved_frames = dict(bc._camera_latest_frame)
        self._saved_frame_at = dict(bc._camera_last_frame_at)
        bc._camera_latest_frame.clear()
        bc._camera_last_frame_at.clear()
        for slot in ("left", "right"):
            bc._kinect_tile_caps[slot] = None
            bc._kinect_tile_frames[slot] = None
            bc._kinect_tile_last_read[slot] = 0.0
            bc._tile_open_declined_note[slot][:] = [0.0, 0]
        bc.CAMERAS[:] = [dict(LEFT), dict(RIGHT)]
        bc._kinect_preview_webcam_idx.clear()
        bc._kinect_preview_webcam_resolved[0] = False
        bc._kinect_preview_webcam_resolved_at[0] = 0.0
        bc._kinect_preview_webcam_fingerprint[0] = None
        bc._kinect_preview_webcam_enumerated_at[0] = 0.0
        for k in bc._side_tile_gate_counts:
            bc._side_tile_gate_counts[k] = 0
        # Zero the window IN PLACE and AT ITS CURRENT LENGTH. Writing a literal
        # list here pins this test to however many slots the reporter happened
        # to have the day it was written, and a shorter one makes the reporter
        # IndexError the moment somebody adds a bucket — which is a test failing
        # for a reason that has nothing to do with the code it is testing.
        bc._side_tile_gate_window[:] = [0.0] + [0] * (
            len(bc._side_tile_gate_window) - 1)
        bc._face_track_caps[0] = None

    def tearDown(self):
        bc = self.bc
        for slot in ("left", "right"):
            bc._kinect_tile_caps[slot] = None
            bc._kinect_tile_frames[slot] = None
            bc._kinect_tile_last_read[slot] = 0.0
        bc._camera_latest_frame.clear()
        bc._camera_latest_frame.update(self._saved_frames)
        bc._camera_last_frame_at.clear()
        bc._camera_last_frame_at.update(self._saved_frame_at)
        bc._face_track_caps[0] = None

    # ── helpers ─────────────────────────────────────────────────────────────

    def _hold(self, *cams):
        """Make the producer HOLD open captures for ``cams`` (CAMERAS dicts)."""
        self.bc._face_track_caps[0] = [
            {"cam": dict(c), "cap": _StreamingCap(), "fails": 0} for c in cams]

    def _drive(self, ticks, t0=1000.0, step=0.25, fresh=(), devices=None):
        """Run `ticks` real composite ticks; return (enumerations, opens).

        `fresh` is the set of camera indices whose producer cache is re-stamped
        every tick (i.e. the slots the fast path may legitimately serve).
        `step` is _KINECT_PREVIEW_TILE_READ_INTERVAL, the only throttle here.
        """
        bc = self.bc
        import numpy as np
        if devices is None:
            devices = ["Left Cam", "Right Cam"]
        enums, opens = [], []
        t = [t0]

        def _enumerate():
            enums.append(1)
            return list(devices)

        shim = _Cv2Shim(opens)
        buf = io.StringIO()
        with mock.patch.object(bc, "_enumerate_dshow_input_devices", _enumerate), \
             mock.patch.object(bc, "_video_device_fingerprint",
                               return_value=_fp()), \
             mock.patch.object(bc, "_dshow_name_to_index", return_value=None), \
             mock.patch.object(bc, "cv2", shim), \
             mock.patch.object(bc.time, "time", side_effect=lambda: t[0]), \
             contextlib.redirect_stdout(buf):
            for _ in range(ticks):
                for idx in fresh:
                    bc._camera_latest_frame[idx] = np.zeros((8, 8, 3),
                                                            dtype=np.uint8)
                    bc._camera_last_frame_at[idx] = t[0]
                self._last = bc._read_side_tile_webcams(t[0])
                t[0] += step
        self._out = buf.getvalue()
        return len(enums), len(opens)

    # ── 1. the rule itself ──────────────────────────────────────────────────

    def test_a_held_camera_is_never_opened_a_second_time(self):
        """THE DEFECT, pinned. The producer holds BOTH cameras and has never
        published a frame for the right one — the first-iteration state, which
        is when this fired on every boot. Before the gate this took a duplicate
        MSMF handle and starved the producer 98.6-99.5%.

        Zero opens AND zero enumerations: the enumeration is what printed
        ``webcam tiles resolved by name``, the line that sits 7-13 s in front of
        every unexplained read failure in the logs."""
        self._hold(LEFT, RIGHT)
        enums, opens = self._drive(80, fresh=(0,))     # right cache never set
        self.assertEqual(opens, 0,
                         "the tile opened a second handle on a camera the "
                         "producer holds — %r" % (opens,))
        self.assertEqual(enums, 0,
                         "the tile enumerated DirectShow for a held camera; "
                         "that is the 'tiles resolved by name' line that "
                         "precedes every one of the six boot failures")
        self.assertEqual(self.bc.get_side_tile_gate_stats()["producer_owned"],
                         80, "the declines were not counted")

    def test_a_stale_producer_cache_does_not_reopen_a_held_camera(self):
        """The fast path declines past _KINECT_PREVIEW_TILE_REUSE_MAX_AGE (2 s),
        and THAT decline is what used to reach the opener. A held camera with a
        cache older than the window must still never be opened."""
        import numpy as np
        bc = self.bc
        self._hold(LEFT, RIGHT)
        bc._camera_latest_frame[1] = np.zeros((8, 8, 3), dtype=np.uint8)
        bc._camera_last_frame_at[1] = 900.0          # 100 s stale at t0=1000
        enums, opens = self._drive(40, fresh=(0,))
        self.assertEqual((enums, opens), (0, 0),
                         "a stale cache on a HELD camera reopened it")

    # ── 2. the producer-loop ordering that made it fire once per boot ────────

    def test_the_non_primary_slot_is_the_one_that_would_have_reopened(self):
        """The asymmetry, pinned, because it is the whole reason this looked
        like a right-camera hardware fault.

        _read_side_tile_webcams is called from inside the PRIMARY camera's
        branch, immediately after the primary's frame is cached, and the
        non-primary camera is read AFTER it in the same iteration. So the
        primary slot is 0 s fresh (fast path, no gate needed) and the
        non-primary slot's cache is one loop period old. With the gate REMOVED
        that is a duplicate open of the non-primary camera and nothing else —
        which is exactly what the logs show: six right-tile failures, zero left."""
        bc = self.bc
        self._hold(LEFT, RIGHT)
        # The DirectShow index the tile resolver hands _open_tile_capture is
        # the position in the device-name list, so 'Left Cam'=0, 'Right Cam'=1.
        tried: list = []
        real = bc._open_tile_capture

        def _spy(idx, name=None):
            tried.append(idx)
            return real(idx, name=name)

        with mock.patch.object(bc, "_producer_holds_side", return_value=False), \
             mock.patch.object(bc, "_open_tile_capture", _spy):
            _enums, opens = self._drive(8, fresh=(0,))
        self.assertGreater(opens, 0,
                           "precondition failed: without the gate this path "
                           "must reopen, or the test proves nothing")
        self.assertEqual(set(tried), {1},
                         "the primary slot took the fast path and the "
                         "NON-primary slot is the only one that reopens; got "
                         "%r" % (sorted(set(tried)),))
        # With the gate back, the same drive opens nothing at all.
        for slot in ("left", "right"):
            bc._kinect_tile_caps[slot] = None
            bc._kinect_tile_frames[slot] = None
            bc._kinect_tile_last_read[slot] = 0.0
        _enums2, opens2 = self._drive(8, fresh=(0,))
        self.assertEqual(opens2, 0,
                         "the gate did not stop the non-primary reopen")

    # ── 3. the gate is about the HANDLE, not the configuration ──────────────

    def test_a_camera_the_producer_let_go_of_is_still_opened(self):
        """``entry["cap"] is None`` is every state in which the producer has
        released a camera (open backoff, dead after N fails, mid-reopen,
        shutdown). In those states the tile's own handle collides with nothing
        and is the ONLY way the slot can show anything — so the gate must not
        fire. A gate keyed on configuration instead of on the handle would
        blank that tile forever."""
        bc = self.bc
        bc._face_track_caps[0] = [
            {"cam": dict(LEFT), "cap": _StreamingCap(), "fails": 0},
            {"cam": dict(RIGHT), "cap": None, "fails": 99},   # ← let go
        ]
        _enums, opens = self._drive(4, fresh=(0,))
        self.assertGreater(opens, 0,
                           "a RELEASED camera was treated as held; that tile "
                           "can now never recover")
        self.assertEqual(bc.get_side_tile_gate_stats()["producer_owned"], 0)

    def test_no_producer_at_all_still_opens(self):
        """``_face_track_caps[0]`` is None before the producer thread starts and
        after it exits. Nothing is held, so nothing is declined."""
        self.bc._face_track_caps[0] = None
        _enums, opens = self._drive(4, fresh=(0,))
        self.assertGreater(opens, 0, "the gate fired with no producer running")

    def test_a_kinect_backed_entry_does_not_count_as_holding_a_webcam(self):
        """A Kinect-typed entry's handle is the SENSOR, not this webcam, and it
        collides with nothing on the USB bus. Counting it as 'held' would blank
        a real webcam tile that has no producer at all — the same
        both-tiles-mirror-the-Kinect bug class _face_track_frame_for_slot
        already guards."""
        bc = self.bc
        kinect_right = dict(RIGHT)
        kinect_right["type"] = "kinect"
        bc._face_track_caps[0] = [
            {"cam": dict(LEFT), "cap": _StreamingCap(), "fails": 0},
            {"cam": kinect_right, "cap": _StreamingCap(), "fails": 0},
        ]
        self.assertFalse(bc._producer_holds_side("right"))
        self.assertTrue(bc._producer_holds_side("left"))

    # ── 3b. the SHARED primitive both openers must ask ──────────────────────

    def test_held_indices_is_the_one_answer_for_every_opener(self):
        """``get_face_track_held_indices`` exists because two openers needed
        this predicate within hours of each other — the side tile and the
        self-diagnostic webcam probe — and a third copy of the reasoning is
        this project's signature STALE DUPLICATE. Pin the contract so a caller
        can rely on it instead of re-deriving it.

        Held = a recorded, NON-None cv2 handle on a NON-Kinect entry."""
        bc = self.bc
        kinect_typed = {"index": 5, "label": "Kinect", "type": "kinect",
                        "name": "kinect", "primary": False, "look_x": 0.5}
        unnamed = {"index": 6, "label": "Unnamed", "primary": False,
                   "look_x": 0.5}          # KINECT_AS_CAMERA hijacks this one
        bc._face_track_caps[0] = [
            {"cam": dict(LEFT), "cap": _StreamingCap()},     # held      -> 0
            {"cam": dict(RIGHT), "cap": None},               # let go    -> -
            {"cam": kinect_typed, "cap": _StreamingCap()},   # sensor    -> -
            {"cam": unnamed, "cap": _StreamingCap()},        # sensor?   -> ?
        ]
        with mock.patch.object(bc, "KINECT_AS_CAMERA", True):
            self.assertEqual(bc.get_face_track_held_indices(), {0},
                             "an UNNAMED entry on a KINECT_AS_CAMERA rig is the "
                             "sensor, not a webcam — the narrower type-only "
                             "test would report a cv2 index that does not exist")
        with mock.patch.object(bc, "KINECT_AS_CAMERA", False):
            self.assertEqual(bc.get_face_track_held_indices(), {0, 6})

    def test_held_indices_is_empty_with_no_producer(self):
        self.bc._face_track_caps[0] = None
        self.assertEqual(self.bc.get_face_track_held_indices(), set())
        self.bc._face_track_caps[0] = []
        self.assertEqual(self.bc.get_face_track_held_indices(), set())

    # ── 4. what the tile actually shows while the producer owns the camera ──

    def test_the_tile_shows_the_producers_cached_frame_while_it_is_owned(self):
        """Declining the open must not blank a working tile. Between
        _KINECT_PREVIEW_TILE_REUSE_MAX_AGE and _TILE_PRODUCER_OWNED_MAX_AGE the
        producer's cache is the only frame that legitimately exists, and it is
        what the slot gets."""
        import numpy as np
        bc = self.bc
        self._hold(LEFT, RIGHT)
        bc._camera_latest_frame[1] = np.full((8, 8, 3), 7, dtype=np.uint8)
        bc._camera_last_frame_at[1] = 995.0          # 5 s: stale for the fast
        _enums, opens = self._drive(1, fresh=(0,))   # path, inside the ceiling
        self.assertEqual(opens, 0)
        self.assertIsNotNone(self._last["right"],
                             "a held camera's tile went dark although the "
                             "producer had a usable cached frame")
        self.assertEqual(int(self._last["right"][0][0][0]), 7)

    def test_past_the_ceiling_the_tile_goes_to_the_placeholder_not_a_frozen_frame(self):
        """A frozen thumbnail displayed as live is the dishonest failure mode
        this project keeps paying for. Past _TILE_PRODUCER_OWNED_MAX_AGE the
        slot yields None and the compositor draws its 'off' panel."""
        import numpy as np
        bc = self.bc
        self._hold(LEFT, RIGHT)
        bc._camera_latest_frame[1] = np.full((8, 8, 3), 7, dtype=np.uint8)
        bc._camera_last_frame_at[1] = 1000.0 - (bc._TILE_PRODUCER_OWNED_MAX_AGE
                                                + 1.0)
        _enums, opens = self._drive(1, fresh=(0,))
        self.assertEqual(opens, 0, "the ceiling must never reopen the camera")
        self.assertIsNone(self._last["right"],
                          "a frame older than the ceiling was served as live")

    def test_the_ceiling_clears_a_full_producer_recovery(self):
        """Sizing, asserted rather than asserted-in-prose: the ceiling has to be
        longer than the producer's own wake path (25 failed reads AND >= 2.0 s
        of silence, then a release+reopen whose open budget is
        _LOOP_OPEN_FRAME_BUDGET_S) or a normal recovery blinks the tile."""
        bc = self.bc
        self.assertGreater(bc._TILE_PRODUCER_OWNED_MAX_AGE,
                           bc._LOOP_OPEN_FRAME_BUDGET_S + 2.0,
                           "the ceiling is shorter than a normal producer wake")
        self.assertGreater(bc._TILE_PRODUCER_OWNED_MAX_AGE,
                           bc._KINECT_PREVIEW_TILE_REUSE_MAX_AGE)
        self.assertLess(bc._TILE_PRODUCER_OWNED_MAX_AGE, 60.0,
                        "a dead producer must reach the placeholder inside a "
                        "minute, not show a still picture indefinitely")

    # ── 5. it is counted, and it is said ────────────────────────────────────

    def test_the_decline_is_said_once_and_then_throttled(self):
        """The duplicate open was INVISIBLE — _camera_open dedupes its backend
        note by (index, name) and the tile reuses the producer's (index, name),
        so it printed nothing. One line per composite would be noise, so the
        note is throttled per slot; but it must be said at least once or this
        stays undiagnosable."""
        self._hold(LEFT, RIGHT)
        self._drive(80, fresh=(0,))       # 80 ticks x 0.25 s = 20 s
        lines = [ln for ln in self._out.splitlines()
                 if "declined" in ln and "own-handle" in ln]
        self.assertEqual(len(lines), 1,
                         "expected exactly one throttled decline line in 20 s, "
                         "got %d: %r" % (len(lines), lines))
        self.assertIn("right tile", lines[0])
        self.assertIn("producer holds this camera", lines[0])

    def test_a_healthy_pair_of_tiles_says_nothing_and_counts_nothing(self):
        """The steady state must stay silent and at zero, or the counter is
        noise and the line becomes something an operator learns to ignore."""
        self._hold(LEFT, RIGHT)
        enums, opens = self._drive(40, fresh=(0, 1))
        s = self.bc.get_side_tile_gate_stats()
        self.assertEqual((enums, opens), (0, 0))
        self.assertEqual(s["producer_owned"], 0)
        self.assertEqual(s["invalidations"], 0)
        self.assertNotIn("declined", self._out)

    def test_the_gate_removes_the_amplifier_for_a_held_camera(self):
        """The two gates compose. The amplifier is one forced re-resolve per
        FAILED side-tile read; with no tile read happening at all for a held
        camera, there is nothing to fail. ``producer_owned > 0`` with
        ``invalidations == 0`` is the healthy post-fix shape."""
        self._hold(LEFT, RIGHT)
        self._drive(80, fresh=(0,))
        s = self.bc.get_side_tile_gate_stats()
        self.assertGreater(s["producer_owned"], 0)
        self.assertEqual(s["invalidations"], 0)
        self.assertEqual(s["enumerations"], 0)


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
