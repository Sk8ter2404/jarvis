"""The camera-OPEN half of the DirectShow-enumeration leak gate (2026-09-05).

WHY THIS FILE EXISTS SEPARATELY from test_monolith_dshow_enum_leak.py, which
covers the side-tile resolver.

The first pass of the leak fix gated ``_resolve_webcam_indices_by_name()`` — the
per-frame tile path — and reported the leak closed. It was not.
``_dshow_name_to_index()`` was a SECOND caller of the same leaky enumeration,
with no fingerprint, no TTL and no floor, and it is the one the CAMERA OPEN path
uses: ``_open_capture()`` calls it unconditionally, and the boot rescue
``_camera_rescued_by_name()`` calls it too.

Measured on this rig before the fix, 40 iterations per figure:

    _enumerate_dshow_input_devices()             +1.00 thr  +103.00 hnd   39 ms
    _dshow_name_to_index(<present name>)         +1.02 thr  +103.08 hnd   48 ms
    _dshow_name_to_index(<absent name>)          +1.00 thr  +103.00 hnd   43 ms   <-- !
    cv2.VideoCapture(<empty index>, CAP_DSHOW)   +1.00 thr  +103.00 hnd   42 ms

The third line is the one that makes this a leak rather than a cost. The
enumeration happens BEFORE the name comparison, so a camera that is unplugged or
sick pays the full price on every attempt to find it — and the face-track loop's
recovery branch attempts exactly that every ``CAMERA_REOPEN_BACKOFF_SEC`` (2.0 s)
for as long as the camera's handle is None. That branch scores no quarantine
strike, so nothing bounds it: ~30 ungated enumerations per MINUTE, five times the
6.0/min the first pass had just removed, in the exact condition (a sick camera)
that exhausted v2.0.100 at 14,507 threads.

The contract pinned below is the same one the resolver's tests pin, applied to
the open path: **the leaky enumeration happens only when the video-device set
could actually have changed, and it still happens whenever it could.** Skipping
it after a real move would open the WRONG camera and succeed, which is quieter
and worse than the leak.

Deliberately NOT asserted here: thread or handle counts. Those are facts about
the machine, not about the code, and a test that asserts them is flaky by
construction. What is asserted is CALL COUNT into the leaky enumerator, which is
the thing this code actually controls.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


# Shaped like the real fingerprint: ((pnp...), (software...)), each pnp entry
# (instance-path, arrival-stamp). There is NO removal stamp: Windows clears
# DEVPKEY_Device_LastRemovalDate when the device comes back and the probe
# enumerates DIGCF_PRESENT, so it could never read one (measured: 0 of 304
# present devnodes carry one, 161 non-present ones do). A non-empty, non-zero
# 8-byte arrival stamp is what earns the LONG self-heal floor
# (_fingerprint_sees_rejoin).
def _fp(devices=(("usb#vid_1&pid_1", b"\x01" * 8),),
        software=("OBS Virtual Camera",)):
    return (tuple(devices), tuple(software))


_NAMES = ("USB 2.0 Camera", "Kinect V2 Video Sensor", "HD Webcam eMeet C960")

# The two numbers that make the storm a storm, taken from the shipped code
# rather than restated, so a change to either shows up here.
_STORM_MINUTES = 20.0


@requires_monolith
class OpenPathEnumerationGateTests(MonolithGlobalsTestCase):
    """How often does _dshow_name_to_index() reach the leaky enumerator?"""

    def setUp(self):
        self.bc._dshow_open_devices_cache[:] = [None, None, 0.0]
        self.calls = []

    def _enumerate(self, names=_NAMES):
        def _fake():
            self.calls.append(1)
            return list(names)
        return _fake

    def _storm(self, substr, fingerprint, names=_NAMES, spacing=None,
               minutes=_STORM_MINUTES, clock_start=1000.0):
        """Drive a REOPEN STORM: one _dshow_name_to_index() every `spacing`
        seconds for `minutes` minutes, which is what the face-track recovery
        branch does to a camera whose handle is None.

        `fingerprint` is one constant value or a LIST giving the value per
        ATTEMPT (the last entry repeats). Per-attempt and not per-call on
        purpose: the gate samples the probe TWICE on an attempt that
        re-enumerates (once to compare, once to record what it enumerated
        against), so a per-call generator would silently desynchronise and the
        test would end up measuring its own fixture.
        """
        spacing = (self.bc.CAMERA_REOPEN_BACKOFF_SEC if spacing is None
                   else spacing)
        attempts = int(round(minutes * 60.0 / spacing))
        t = [clock_start]
        schedule = (list(fingerprint) if isinstance(fingerprint, list)
                    else [fingerprint])
        i = [0]

        def _fp_now():
            return schedule[min(i[0], len(schedule) - 1)]

        got = []
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               self._enumerate(names)), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               side_effect=_fp_now), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: t[0]):
            for k in range(attempts):
                i[0] = k
                got.append(self.bc._dshow_name_to_index(substr))
                t[0] += spacing
        return len(self.calls), attempts, got

    # -- the defect itself ---------------------------------------------------
    def test_reopen_storm_on_a_present_camera_enumerates_once(self):
        """THE REGRESSION TEST. 20 minutes of the recovery branch's 2 s reopen
        cadence = 600 attempts. Pre-fix that was 600 leaked OS threads and
        61,800 leaked handles; the gate must pay for exactly one."""
        n, attempts, got = self._storm("emeet c960", _fp())
        self.assertEqual(attempts, 600, "fixture drifted: %d attempts" % attempts)
        self.assertEqual(n, 1, "the reopen storm ran the leaky enumeration %d "
                               "times; pre-fix it ran %d" % (n, attempts))
        # ...and every one of those attempts still got the RIGHT index, not a
        # cheap wrong answer.
        self.assertEqual(set(got), {2})

    def test_reopen_storm_on_an_ABSENT_camera_enumerates_once(self):
        """THE ACTUAL FAILURE SCENARIO, and the one the pre-fix code was worst
        at. The camera is gone, so the name never matches — and because the
        enumeration happens BEFORE the comparison, the pre-fix code paid the
        full +1 thread / +103 handles on every single attempt to find a device
        that was not there. The bus is not moving while a camera is merely
        absent, so after the one enumeration that observes it leaving, the gate
        must serve every later attempt for free."""
        n, attempts, got = self._storm("camera that was unplugged", _fp())
        self.assertEqual(n, 1, "an ABSENT camera cost %d leaky enumerations "
                               "over %d reopen attempts" % (n, attempts))
        self.assertEqual(set(got), {None}, "an absent name must not resolve")

    def test_gate_disabled_reproduces_the_original_defect(self):
        """THE COUNTERFACTUAL, so the two tests above cannot pass for the wrong
        reason (e.g. a fixture that never advances the clock). Driving both
        self-heal floors to 0 makes the floor fire on every attempt, which is
        exactly the pre-fix behaviour: enumerate every time, unconditionally."""
        with mock.patch.object(self.bc, "_WEBCAM_IDX_SELFHEAL_STAMPED_SEC", 0.0), \
             mock.patch.object(self.bc, "_WEBCAM_IDX_SELFHEAL_SEC", 0.0):
            n, attempts, _ = self._storm("emeet c960", _fp())
        self.assertEqual(n, attempts)

    def test_empty_substring_never_enumerates(self):
        n, _, got = self._storm("   ", _fp(), minutes=1.0)
        self.assertEqual(n, 0)
        self.assertEqual(set(got), {None})

    # -- the other half: it must STILL re-enumerate when it matters ----------
    def test_device_added_forces_re_enumeration(self):
        added = _fp((("usb#vid_1&pid_1", b"\x01" * 8),
                     ("usb#vid_2&pid_2", b"\x09" * 8)))
        n, _, _ = self._storm("emeet c960", [_fp(), _fp(), added], minutes=0.1)
        self.assertEqual(n, 2)

    def test_device_removed_forces_re_enumeration(self):
        two = _fp((("a", b"\x01" * 8), ("b", b"\x02" * 8)))
        one = _fp((("a", b"\x01" * 8),))
        n, _, _ = self._storm("emeet c960", [two, two, one], minutes=0.1)
        self.assertEqual(n, 2)

    def test_virtual_camera_start_stop_forces_re_enumeration(self):
        """A software DirectShow filter has no PnP arrival stamp, so only the
        software half of the fingerprint can catch it — and starting OBS DOES
        reshuffle the indices behind it."""
        off = _fp(software=())
        on = _fp(software=("OBS Virtual Camera",))
        n, _, _ = self._storm("emeet c960", [off, off, on], minutes=0.1)
        self.assertEqual(n, 2)

    def test_leave_and_rejoin_with_identical_device_set_forces_re_enumeration(self):
        """THE ONE A DEVICE-SET FINGERPRINT CANNOT SEE — and the case that
        matters MOST on the open path, because this is precisely the shape of
        "the camera came back": it dropped off the bus and rejoined, DirectShow
        renumbered underneath, and an index served from the cache would now open
        a DIFFERENT camera and succeed. Only the advancing arrival stamp makes
        it visible after the fact."""
        before = _fp((("usb#vid_1&pid_1", b"\x01" * 8),))
        after = _fp((("usb#vid_1&pid_1", b"\x77" * 8, b"\x66" * 8),))
        self.assertEqual([d[0] for d in before[0]], [d[0] for d in after[0]],
                         "precondition: the device SET must be identical")
        n, _, _ = self._storm("emeet c960", [before, before, after], minutes=0.1)
        self.assertEqual(n, 2, "a camera that left and rejoined was served from "
                               "the cache — the opener would use a stale index")

    def test_unreadable_fingerprint_degrades_to_enumerating_every_time(self):
        """A probe that cannot tell must never be read as 'nothing changed'.
        Degrade to the old, leaky, CORRECT behaviour rather than pin an index."""
        n, attempts, _ = self._storm("emeet c960", None, minutes=0.1)
        self.assertEqual(n, attempts)

    def test_failed_enumeration_is_not_pinned_by_the_gate(self):
        """When the enumeration itself fails there is no name list, so the
        fingerprint sampled alongside it describes nothing. Keeping either would
        let ONE COM hiccup suppress every retry for a whole self-heal period —
        an hour of opening cameras by a stale index."""
        t = [1000.0]
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               return_value=None), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               return_value=_fp()), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: t[0]):
            self.assertIsNone(self.bc._dshow_name_to_index("emeet c960"))
        self.assertIsNone(self.bc._dshow_open_devices_cache[0])
        self.assertIsNone(self.bc._dshow_open_devices_cache[1])
        # ...so the very next attempt really does try again.
        n, _, got = self._storm("emeet c960", _fp(), minutes=0.1,
                                clock_start=t[0] + 1.0)
        self.assertEqual(n, 1)
        self.assertEqual(set(got), {2})

    # -- the self-heal floors ------------------------------------------------
    def test_stamped_fingerprint_earns_the_long_floor(self):
        minutes = self.bc._WEBCAM_IDX_SELFHEAL_SEC * 2 / 60.0
        n, _, _ = self._storm("emeet c960", _fp(), minutes=minutes)
        self.assertEqual(n, 1, "the short floor fired despite a stamped "
                               "fingerprint that makes it unnecessary")

    def test_unstamped_fingerprint_keeps_the_short_floor(self):
        unstamped = _fp((("usb#vid_1&pid_1", b"", b""),))
        minutes = self.bc._WEBCAM_IDX_SELFHEAL_SEC * 2 / 60.0
        n, _, _ = self._storm("emeet c960", unstamped, minutes=minutes)
        self.assertGreaterEqual(n, 2, "the short self-heal floor never fired for "
                                      "a fingerprint that cannot see a rejoin")

    def test_long_floor_does_eventually_fire(self):
        minutes = self.bc._WEBCAM_IDX_SELFHEAL_STAMPED_SEC * 2 / 60.0
        n, _, _ = self._storm("emeet c960", _fp(), minutes=minutes)
        self.assertEqual(n, 2, "the hourly safety-net enumeration never fired")


@requires_monolith
class OpenPathGateIsolationTests(MonolithGlobalsTestCase):
    """The open path and the tile path hold SEPARATE cache slots. That is a
    deliberate cost (each pays at most one enumeration per floor) bought to
    avoid a shared-mutable-state coupling between the HUD compositor thread and
    the face-track producer thread. What must NOT happen is one path quietly
    corrupting the other's state."""

    def setUp(self):
        self.bc._dshow_open_devices_cache[:] = [None, None, 0.0]
        self.bc._kinect_preview_webcam_idx.clear()
        self.bc._kinect_preview_webcam_resolved[0] = False
        self.bc._kinect_preview_webcam_resolved_at[0] = 0.0
        self.bc._kinect_preview_webcam_fingerprint[0] = None
        self.bc._kinect_preview_webcam_enumerated_at[0] = 0.0

    def test_open_path_does_not_touch_the_side_tile_resolver_cells(self):
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               return_value=list(_NAMES)), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               return_value=_fp()):
            self.bc._dshow_name_to_index("emeet c960")
        self.assertIsNone(self.bc._kinect_preview_webcam_fingerprint[0])
        self.assertEqual(self.bc._kinect_preview_webcam_enumerated_at[0], 0.0)
        self.assertFalse(self.bc._kinect_preview_webcam_resolved[0])
        self.assertEqual(dict(self.bc._kinect_preview_webcam_idx), {})

    def test_side_tile_resolver_does_not_touch_the_open_path_cell(self):
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               return_value=list(_NAMES)), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               return_value=_fp()), \
             mock.patch.object(self.bc, "_kinect_preview_webcam_names",
                               return_value={"left": "emeet", "right": "usb 2.0"}):
            self.bc._resolve_webcam_indices_by_name()
        self.assertEqual(self.bc._dshow_open_devices_cache, [None, None, 0.0])


@requires_monolith
class GatedEnumeratorContractTests(MonolithGlobalsTestCase):
    """_dshow_input_devices_gated() itself: the ONE copy of the gate rule."""

    def setUp(self):
        self.cache = [None, None, 0.0]
        self.calls = []

    def _enumerate(self, names=_NAMES):
        def _fake():
            self.calls.append(list(self.cache))     # cache AS SEEN mid-call
            return list(names)
        return _fake

    def test_fingerprint_is_never_published_alongside_stale_names(self):
        """THE WRITE-ORDER PROPERTY, which is what lets this run lock-free on
        two threads. The enumeration takes ~40 ms; a second thread inside the
        gate during that window must never see a NEW fingerprint paired with the
        OLD name list, because it would then serve a stale index for a bus that
        had just moved — open the wrong camera, and succeed.

        The fake enumerator below records the cache exactly as a concurrent
        reader would see it, from inside the slow call."""
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               self._enumerate()), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               return_value=_fp()):
            self.bc._dshow_input_devices_gated(self.cache, now=1000.0)
            # Second pass, with a cache that is now warm: the mid-call snapshot
            # is the interesting one.
            self.bc._dshow_input_devices_gated(
                self.cache, now=1000.0 + self.bc._WEBCAM_IDX_SELFHEAL_STAMPED_SEC * 2)
        self.assertEqual(len(self.calls), 2)
        for seen_names, seen_fp, _seen_at in self.calls:
            self.assertIsNone(seen_fp,
                              "a fingerprint was visible while the enumeration "
                              "it belongs to was still outstanding")

    def test_returns_a_copy_so_a_caller_cannot_corrupt_the_cache(self):
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               return_value=list(_NAMES)), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               return_value=_fp()):
            first = self.bc._dshow_input_devices_gated(self.cache, now=1000.0)
            first.append("A CAMERA THAT DOES NOT EXIST")
            second = self.bc._dshow_input_devices_gated(self.cache, now=1001.0)
        self.assertEqual(second, list(_NAMES))

    def test_junk_cache_degrades_to_the_raw_enumeration_and_never_raises(self):
        """The gate must never be the reason a camera cannot be opened."""
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               return_value=list(_NAMES)), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               return_value=_fp()):
            for junk in ([], [None], "nope", 0):
                self.assertEqual(self.bc._dshow_input_devices_gated(junk),
                                 list(_NAMES), junk)


@requires_monolith
class BootRescueGoesThroughTheGateTests(MonolithGlobalsTestCase):
    """_camera_rescued_by_name() is the THIRD caller of _dshow_name_to_index().
    The boot probe runs it once per configured camera that failed its static
    probe, so on a rig where both cameras are missing it used to pay the leak
    twice before the first frame."""

    def setUp(self):
        self.bc._dshow_open_devices_cache[:] = [None, None, 0.0]
        self.calls = []

    def test_two_rescues_share_one_enumeration(self):
        def _fake():
            self.calls.append(1)
            return list(_NAMES)

        cams = [{"index": 2, "label": "L", "name": "emeet c960"},
                {"index": 0, "label": "R", "name": "usb 2.0 camera"}]
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices", _fake), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               return_value=_fp()), \
             mock.patch.object(self.bc, "KINECT_AS_CAMERA", False), \
             mock.patch.object(self.bc, "_probe_camera_index", return_value=True):
            for cam in cams:
                self.bc._camera_rescued_by_name(cam, cam["index"])
        self.assertEqual(len(self.calls), 1,
                         "the boot rescue enumerated once per camera (%d) "
                         "instead of once for the bus" % len(self.calls))


@requires_monolith
class NoUngatedCallerMayBeAddedTests(MonolithGlobalsTestCase):
    """THE TEST THAT WOULD HAVE CAUGHT THIS DEFECT.

    The first pass of the leak fix was correct about the tile path and simply
    did not notice a second caller. No behavioural test could catch that — the
    second caller was not wrong, it was ABSENT from the review. This one reads
    the source and asserts the shape directly: ``_enumerate_dshow_input_devices``
    is the leaky primitive, and nothing may call it except the two functions
    that gate it.

    Adding a legitimate new consumer is meant to fail here. The fix is to route
    it through ``_dshow_input_devices_gated()`` and, only if it genuinely gates
    the call itself, to add it to the allowlist below WITH the reason."""

    # NOT DUPLICATED HERE: the whole-module allowlist scan ("which functions may
    # call _enumerate_dshow_input_devices at all") lives in
    # tests/monolith/test_monolith_dshow_reopen_leak.py ::
    # NoUngatedEnumeratorCallSitesTests, which pins the caller set with
    # assertEqual and is therefore strictly stronger than the version this class
    # originally carried. Two AST scans asserting the same rule is the
    # stale-duplicate shape this codebase pays for most, so there is one.
    # If that file is ever dropped, restore the scan HERE rather than leaving the
    # rule unpinned -- it is the only kind of test that can catch "a call site
    # the gate did not know about", which is exactly what this defect was.
    # What stays below is the narrower claim only this file is about: the camera
    # OPEN path specifically.

    def test_the_open_path_reaches_the_primitive_only_through_the_gate(self):
        """...and specifically: _dshow_name_to_index, the camera-open path's
        resolver, must call the GATED function and not the raw one."""
        import inspect
        body = inspect.getsource(self.bc._dshow_name_to_index)
        self.assertIn("_dshow_input_devices_gated(", body)
        self.assertNotIn("_enumerate_dshow_input_devices(", body)


if __name__ == "__main__":      # pragma: no cover - manual runner
    unittest.main()
