"""The DirectShow video-enumeration leak gate (2026-09-05).

WHAT BROKE, so the tests below read as consequences rather than trivia.

``ICreateDevEnum::CreateClassEnumerator(CLSID_VideoInputDeviceCategory)`` leaks
+1 OS thread and +103 handles PER CALL, permanently, inside mfksproxy.dll. It is
an OS-level defect, not a missing Release: a raw-ctypes repro that Releases both
interfaces and calls CoUninitialize leaks identically, a fresh apartment leaks
identically, and nothing is ever reclaimed. JARVIS could therefore only ever fix
its USAGE, never the call.

``_resolve_webcam_indices_by_name()`` used to make that call every
``_WEBCAM_IDX_TTL_SEC`` (10 s) = 6 leaked threads and 618 leaked handles per
MINUTE, plus one more per failed side-tile read via
``_invalidate_side_tile_indices()``. v2.0.100 reached 14,507 threads and
1,419,190 handles in 3.5 h and then stopped logging from EVERY subsystem - the
process was thread-exhausted.

SCOPE, MEASURED - because the headline number does not describe the incident.
The two soaks behind "5.952 -> 0.110 mfksproxy threads/min, a 54x reduction"
BOTH ran a healthy single 'emeet c960'. That healthy steady state was never
what killed v2.0.100: 14,507 threads in ~3.5 h is ~70 threads/min, and 6/min is
under 9% of it. The other ~91% arrived through the SICK-camera path, which no
soak had ever exercised.

Per-call cost on this rig 2026-09-05, counting only threads whose Win32 start
address lies inside mfksproxy.dll (n=10, 1 s settle either side):

    _video_device_fingerprint()      (the replacement probe)  +0.00 thr / +0 hnd
    _enumerate_dshow_input_devices() (what the gate removes)  +1.00 thr / +115 hnd
    _open_tile_capture() + release   (what it does NOT)       +5.00 thr / +536 hnd
        of which  cv2.VideoCapture(idx, CAP_DSHOW) open+release  +3.00 / +309
        and       the 640x480 set(WIDTH)+set(HEIGHT) re-negotiation  +2.00 / +206
        (an OUT-OF-RANGE index costs only +1.00 / +103 - there is no device to
         negotiate a mode with, so the reopen is expensive BECAUSE the camera
         exists, which is exactly the sick-camera case)

SOAKED, not extrapolated. 25 min against a REAL in-range DirectShow index with
only read() forced to fail, sampled every 30 s: over the first 19.5 min / 271
retry cycles it leaked 4.86 mfksproxy threads and 501 handles per cycle at 13.9
cycles/min = 67.6 threads/min - indistinguishable from the ~70/min the incident
ran at, WITH this gate in place. (Past ~1,390 leaked threads the measured rate
fell to ~22/min. That is the process running out of room, not recovering; the
mechanism was not established, so nothing should read the tail of that curve as
a plateau.)

COUNTERFACTUAL, measured the only way that survives this device. Two separate
soak processes cannot be compared: the webcam's own DirectShow cost drifts under
hundreds of open/release cycles (within one run it moved between 1.25 and 5.00
threads per reopen), so the second process inherits a different machine. Running
both arms INTERLEAVED in one process - 8 blocks x 20 cycles, alternating the
real gate against the pre-fix behaviour (a None fingerprint, the same
counterfactual test_gate_disabled_reproduces_the_original_defect uses) - gives:

    GATED     80 cycles,  4 enumerations, 3.05 thr/cycle, 317.5 hnd/cycle
    UNGATED   80 cycles, 80 enumerations, 4.01 thr/cycle, 412.1 hnd/cycle
    mean paired difference 0.96 thr/cycle - i.e. exactly the ONE enumeration

So the gate removes ~24% of the sick-camera leak (5.7%-45.3% per block pair,
tracking whatever the reopen happens to cost at the time). It removes ONE
enumeration per retry cycle, reliably; the reopen it leaves behind is the
dominant term and no fingerprint can reach it.

The healthy-path 54x is real and so is this gate; neither is a statement about
a sick camera. What bounds the sick camera is the camera QUARANTINE, not this
gate, and SickCameraReopenIsNotGatedTests below pins that division so the claim
cannot quietly drift back.

So the contract these tests pin is: **the leaky enumeration happens only when
the video-device set could actually have changed**, and it still happens
whenever it could. Both halves matter. Skipping it when a device really moved
means the tiles read the WRONG camera and every read still SUCCEEDS, so nothing
reports it - a quieter bug than the leak it replaced.

Deliberately NOT tested here: thread or handle counts. Those are true of the
machine, not of the code, and a test that asserts them is flaky by construction.
What is asserted instead is CALL COUNT into the leaky enumerator, which is the
thing the code actually controls, plus release-on-every-path for the handle the
replacement probe opens.
"""
from __future__ import annotations

import contextlib
import ctypes
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


def _fp(devices=(("usb#vid_1&pid_1", b"\x01" * 8),),
        software=("OBS Virtual Camera",)):
    """A fingerprint shaped like the real one: ``((pnp...), (software...))``
    where each pnp entry is ``(instance-path, arrival-stamp)``.

    THERE IS NO REMOVAL STAMP, and that is the fix rather than an omission.
    The probe used to carry DEVPKEY_Device_LastRemovalDate as a third field
    and it was ALWAYS b'': Windows clears that property when the device comes
    back, and the probe enumerates DIGCF_PRESENT, so it can never observe one.
    Measured on the dev box: of 304 present devnodes carrying an arrival stamp,
    ZERO carried a removal stamp, while 161 non-present ones did. A field that
    is constant by construction is not a change detector -- it just made the
    mechanism read as though it had two legs when it has one."""
    return (tuple(devices), tuple(software))


@requires_monolith
class FingerprintSeesRejoinTests(MonolithGlobalsTestCase):
    """_fingerprint_sees_rejoin decides WHICH self-heal floor applies, so a
    wrong True buys an hour of stale indices on a fingerprint that cannot in
    fact detect a rejoin."""

    def test_true_when_every_pnp_device_carries_an_arrival_stamp(self):
        self.assertTrue(self.bc._fingerprint_sees_rejoin(
            _fp((("a", b"\x01" * 8), ("b", b"\x02" * 8)))))

    def test_false_when_any_device_has_no_arrival_stamp(self):
        # One unreadable stamp is enough: that device's rejoin is invisible.
        self.assertFalse(self.bc._fingerprint_sees_rejoin(
            _fp((("a", b"\x01" * 8), ("b", b"")))))

    def test_false_for_empty_pnp_half(self):
        # THE VACUOUS-all() TRAP. A SetupAPI probe that silently enumerates
        # nothing returns an empty tuple, not None, and `all(())` is True. If
        # that earned the long floor, a totally broken probe would be rewarded
        # with the WEAKEST self-heal instead of the strongest.
        self.assertFalse(self.bc._fingerprint_sees_rejoin(_fp((), ("OBS",))))

    def test_false_for_an_all_zero_filetime_stamp(self):
        """PRESENCE IS NOT EVIDENCE OF MOVEMENT -- the defect this class
        exists for, in miniature. An 8-byte all-zero FILETIME means 'the
        property exists and was never set', and it is TRUTHY bytes, so a bare
        ``all(entry[1] for ...)`` hands out the LONG (hourly) self-heal floor
        on a value guaranteed never to change. The long floor is only
        defensible because the real stamp was cycled and watched moving; a
        stamp that cannot move must never buy it."""
        self.assertFalse(self.bc._fingerprint_sees_rejoin(
            _fp((("a", b"\x00" * 8),))))
        # ...and one dead stamp poisons an otherwise-good set.
        self.assertFalse(self.bc._fingerprint_sees_rejoin(
            _fp((("a", b"\x01" * 8), ("b", b"\x00" * 8)))))

    def test_false_for_a_stamp_that_is_not_filetime_width(self):
        """A FILETIME is exactly 8 bytes. Any other width means we did not
        read what we think we read, so it has to DOWNGRADE to the short floor
        rather than be trusted."""
        for bad in (b"\x01", b"\x01" * 4, b"\x01" * 7, b"\x01" * 9,
                    b"\x01" * 16):
            self.assertFalse(
                self.bc._fingerprint_sees_rejoin(_fp((("a", bad),))), bad)

    def test_legacy_three_field_entry_still_reads_the_arrival_stamp(self):
        """The removal field was dropped from the probe, but the arrival
        stamp is still element 1 and entries are otherwise opaque. A stray
        3-field entry must not silently flip the floor decision."""
        self.assertTrue(self.bc._fingerprint_sees_rejoin(
            _fp((("a", b"\x01" * 8, b""),))))

    def test_usable_arrival_stamp_never_raises(self):
        """It runs on the preview thread via _fingerprint_sees_rejoin, which
        has no error handling of its own."""
        for junk in (None, 0, 1.5, object(), [1] * 8, b"", bytearray(8)):
            try:
                self.bc._usable_arrival_stamp(junk)
            except Exception as exc:      # pragma: no cover - the assertion
                self.fail("raised on %r: %s" % (junk, exc))

    def test_false_for_none_and_for_junk(self):
        # Called with whatever the fingerprint cell happens to hold, including
        # its import-time None. Must never raise.
        for junk in (None, (), 0, "nope", object(), ((None,), ())):
            self.assertFalse(self.bc._fingerprint_sees_rejoin(junk), junk)


@requires_monolith
class ResolverEnumerationGateTests(MonolithGlobalsTestCase):
    """The hot path: how often does _resolve_webcam_indices_by_name() reach the
    leaky enumerator?  Every test here counts calls, never threads."""

    def setUp(self):
        self.bc._kinect_preview_webcam_idx.clear()
        self.bc._kinect_preview_webcam_resolved[0] = False
        self.bc._kinect_preview_webcam_resolved_at[0] = 0.0
        self.bc._kinect_preview_webcam_fingerprint[0] = None
        self.bc._kinect_preview_webcam_enumerated_at[0] = 0.0
        self.calls = []

    def _enumerate(self, names=("Left Cam", "Right Cam")):
        """A stand-in for the LEAKY DirectShow enumeration that records calls."""
        def _fake():
            self.calls.append(1)
            return list(names)
        return _fake

    def _run_ticks(self, n, fingerprint, names=("Left Cam", "Right Cam"),
                   clock_start=1000.0, tick=None):
        """Drive `n` TTL expiries and return (leaky enumerations, final map).

        `fingerprint` is either one constant fingerprint or a LIST giving the
        value for each tick (the last entry repeats). It is deliberately NOT a
        pop-per-call generator: the resolver samples the probe TWICE on a tick
        that re-enumerates (once to compare, once to record what it resolved
        against), so a per-call sequence silently desynchronises from ticks and
        the test starts measuring its own fixture."""
        tick = tick if tick is not None else self.bc._WEBCAM_IDX_TTL_SEC
        t = [clock_start]
        schedule = list(fingerprint) if isinstance(fingerprint, list) else [fingerprint]
        i = [0]

        def _fp_now():
            return schedule[min(i[0], len(schedule) - 1)]

        got = {}
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               self._enumerate(names)), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               side_effect=_fp_now), \
             mock.patch.object(self.bc, "_kinect_preview_webcam_names",
                               return_value={"left": "left cam", "right": "right cam"}), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: t[0]):
            for k in range(n):
                i[0] = k
                got = self.bc._resolve_webcam_indices_by_name()
                t[0] += tick
        return len(self.calls), got

    # -- the defect itself ---------------------------------------------------
    def test_unchanged_devices_enumerate_exactly_once_over_20_minutes(self):
        """THE REGRESSION TEST. 120 TTL ticks = 20 minutes of production. The
        pre-fix code enumerated on every one of them (120 leaked threads,
        12,360 leaked handles); the gate must do it once, at the start."""
        n, got = self._run_ticks(120, _fp())
        self.assertEqual(n, 1, "leaky enumeration ran %d times for an unchanged "
                               "device set; the pre-fix code ran it 120x" % n)
        # ...and the map it hands back is still the RIGHT one, not a stale {}.
        self.assertEqual(got, {"left": 0, "right": 1})

    def test_gate_disabled_reproduces_the_original_defect(self):
        """THE COUNTERFACTUAL, so the test above cannot pass for the wrong
        reason. Driving both self-heal floors to 0 makes the floor fire on every
        tick, which is exactly the pre-fix behaviour: re-enumerate on every TTL
        expiry regardless of whether anything moved. The same 120 ticks must
        then produce 120 leaky enumerations - i.e. this fixture really is
        sensitive to the defect, and the 1 above is the fix, not the fixture."""
        with mock.patch.object(self.bc, "_WEBCAM_IDX_SELFHEAL_STAMPED_SEC", 0.0), \
             mock.patch.object(self.bc, "_WEBCAM_IDX_SELFHEAL_SEC", 0.0):
            n, _ = self._run_ticks(120, _fp())
        self.assertEqual(n, 120)

    def test_failed_read_invalidation_does_not_re_enumerate_unchanged_devices(self):
        """ONE SIXTH OF THE AMPLIFIER, and this test only ever proved that
        sixth. _invalidate_side_tile_indices() fires on EVERY failed side-tile
        read and clears the resolved flag, bypassing the TTL entirely - so a
        sick camera re-enumerated per retry instead of per 10 s. A failed read
        only justifies re-resolving if the devices MOVED; an unchanged
        fingerprint proves they did not, and the gate below removes exactly that
        re-enumeration.

        HONEST LIMIT. This docstring used to say "~89% of that session's leak",
        a number nobody measured. The re-enumeration it removes is 1 of the 6
        mfksproxy threads a sick-camera retry cycle leaks; the other 5 are spent
        inside _open_tile_capture()'s own cv2.VideoCapture(idx, CAP_DSHOW) - a
        call this fixture never makes and no fingerprint gate can reach (module
        docstring carries the per-call measurements). This test drives the
        RESOLVER directly, so what it proves is exactly "the resolver stops
        re-enumerating on a failed read", and nothing at all about what a sick
        camera costs end to end. SickCameraReopenIsNotGatedTests measures that."""
        t = [1000.0]
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               self._enumerate()), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               return_value=_fp()), \
             mock.patch.object(self.bc, "_kinect_preview_webcam_names",
                               return_value={"left": "left cam", "right": "right cam"}), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: t[0]):
            self.bc._resolve_webcam_indices_by_name()          # initial resolve
            for _ in range(200):                               # 200 failed reads
                self.bc._invalidate_side_tile_indices()
                self.bc._resolve_webcam_indices_by_name()
                t[0] += 0.25                                   # the tile read rate
        self.assertEqual(len(self.calls), 1,
                         "a wedged camera re-enumerated %d times" % len(self.calls))

    # -- the other half: it must STILL re-enumerate when it matters ----------
    def test_device_added_forces_re_enumeration(self):
        added = _fp((("usb#vid_1&pid_1", b"\x01" * 8),
                     ("usb#vid_2&pid_2", b"\x09" * 8)))
        n, _ = self._run_ticks(3, [_fp(), _fp(), added])
        self.assertEqual(n, 2)

    def test_device_removed_forces_re_enumeration(self):
        two = _fp((("a", b"\x01" * 8), ("b", b"\x02" * 8)))
        one = _fp((("a", b"\x01" * 8),))
        n, _ = self._run_ticks(3, [two, two, one])
        self.assertEqual(n, 2)

    def test_virtual_camera_start_stop_forces_re_enumeration(self):
        """OBS Virtual Camera is a software DirectShow filter, not a PnP device
        - no arrival stamp exists for it - so the software half of the
        fingerprint is what has to catch it."""
        off = _fp(software=())
        on = _fp(software=("OBS Virtual Camera",))
        n, _ = self._run_ticks(3, [off, off, on])
        self.assertEqual(n, 2)

    def test_leave_and_rejoin_with_identical_device_set_forces_re_enumeration(self):
        """THE ONE A DEVICE-SET FINGERPRINT CANNOT SEE, and the reason the PnP
        half carries DEVPKEY_Device_LastArrivalDate.

        A camera that drops off and comes back between two 10 s samples leaves
        the set byte-identical while DirectShow renumbers underneath. Reads then
        SUCCEED from the wrong camera, so the failed-read backstop never fires
        and nothing anywhere reports it - strictly worse than the leak. Only the
        advancing arrival stamp makes it visible after the fact."""
        before = _fp((("usb#vid_1&pid_1", b"\x01" * 8),))
        after = _fp((("usb#vid_1&pid_1", b"\x77" * 8),))    # arrival re-stamped
        self.assertEqual([d[0] for d in before[0]], [d[0] for d in after[0]],
                         "precondition: the device SET must be identical")
        n, _ = self._run_ticks(3, [before, before, after])
        self.assertEqual(n, 2, "a leave+rejoin with an unchanged device set was "
                               "not detected - the tiles would read the wrong "
                               "camera indefinitely")

    def test_unreadable_fingerprint_degrades_to_re_enumerating(self):
        """A probe that cannot tell must NOT be read as 'nothing changed'. None
        has to fall through to the real enumeration - the old, leaky, correct
        behaviour - rather than pinning a possibly-stale index."""
        n, _ = self._run_ticks(5, None)
        self.assertEqual(n, 5)

    # -- the self-heal floors ------------------------------------------------
    def test_stamped_fingerprint_earns_the_long_floor(self):
        """With stamps, every reshuffle mechanism is detected directly, so the
        unconditional re-enumeration is a safety net and runs hourly."""
        ticks = int(self.bc._WEBCAM_IDX_SELFHEAL_SEC * 2 / self.bc._WEBCAM_IDX_TTL_SEC)
        n, _ = self._run_ticks(ticks, _fp())      # spans 2x the SHORT floor
        self.assertEqual(n, 1, "the short floor fired despite a stamped "
                               "fingerprint that makes it unnecessary")

    def test_unstamped_fingerprint_keeps_the_short_floor(self):
        """Without stamps the leave+rejoin hole is open, so the short floor is
        the only thing that ever corrects it and MUST still fire."""
        unstamped = _fp((("usb#vid_1&pid_1", b""),))
        ticks = int(self.bc._WEBCAM_IDX_SELFHEAL_SEC * 2 / self.bc._WEBCAM_IDX_TTL_SEC)
        n, _ = self._run_ticks(ticks, unstamped)
        self.assertGreaterEqual(n, 2, "the short self-heal floor never fired for "
                                      "a fingerprint that cannot see a rejoin")

    def test_long_floor_does_eventually_fire(self):
        ticks = int(self.bc._WEBCAM_IDX_SELFHEAL_STAMPED_SEC * 2
                    / self.bc._WEBCAM_IDX_TTL_SEC)
        n, _ = self._run_ticks(ticks, _fp())
        self.assertEqual(n, 2, "the hourly safety-net enumeration never fired")

    def test_failed_enumeration_is_not_pinned_by_the_gate(self):
        """When the enumeration itself fails there is no index map, so the
        fingerprint recorded alongside it describes nothing. Keeping it would
        let one COM hiccup suppress every retry for a whole self-heal period."""
        t = [1000.0]
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               return_value=None), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               return_value=_fp()), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: t[0]):
            self.bc._resolve_webcam_indices_by_name()
        self.assertIsNone(self.bc._kinect_preview_webcam_fingerprint[0])
        # ...so the next tick really does try again.
        n, _ = self._run_ticks(1, _fp(), clock_start=t[0] + 11.0)
        self.assertEqual(n, 1)


@requires_monolith
class AmplifierUnderLoadTests(MonolithGlobalsTestCase):
    """THE AMPLIFIER AT THE RATE THE CODE ACTUALLY ALLOWS, and what a failed-open
    gate does to it.

    WHY THIS CLASS EXISTS. The 2026-09-06 sick-camera soak was reported as "the
    sick camera does NOT reopen the amplifier — 96.4 min, both cameras, sick one
    live". It did not test that. _KINECT_PREVIEW_TILE_READ_INTERVAL is 0.25 s, so
    the right tile got ~24,000 reads in that window; the gap watcher recorded 3
    right-tile gaps over 1 s in 23,465 writes and the tile's own byte count kept
    moving (99 distinct sizes over 460 samples), i.e. the camera was delivering
    frames, not placeholders. HOW OFTEN THE AMPLIFIER ACTUALLY FIRED IS NOT
    KNOWABLE FROM THAT RUN — nothing counted invalidations (the counters landed
    later the same day) and the side-tile failure path logs nothing, so the run
    is silent on the only quantity the claim was about. What it does show is a
    camera that healed. Cut to matched process-age windows the sick
    leg (0.021 mfksproxy thr/min) and the healthy leg (0.019) are
    indistinguishable, and both mfksproxy steps in each leg land 15-31 s after a
    run_diagnostic sweep on its own 30-minute scheduler, in BOTH legs. Nothing in
    that soak was a statement about the amplifier.

    SOAKED PROPERLY 2026-09-06, real COM, real leaky enumeration, threads counted
    by Win32 start address inside mfksproxy.dll:

        35.0 min  gate healthy,  amplifier at 4 Hz   8,320 firings
                        1 enumeration total, +1 mfksproxy thread, +237 handles,
                        and the real probe answered on all 8,320 calls
         3.0 min  gate degraded, amplifier quiet     6.00 enum/min   +664 hnd/min
         3.0 min  gate degraded, amplifier at 4 Hz 208.13 enum/min +19,259 hnd/min
         8.5 min  same, floor removed (confirm)    200.21 enum/min +18,340 hnd/min
        25.0 min  gate degraded, amplifier at 4 Hz,  5.96 enum/min    +566 hnd/min
                        with the floor (5,916 firings)

    The two degraded amplifier rows are what nobody had measured, and they are
    why the module's "+6 OS threads and +618 handles per MINUTE" description of
    the degraded state was ~31x optimistic: at those rates the v2.0.100 death
    numbers (14,507 threads / 1,419,190 handles) arrive in 74-82 minutes rather
    than ~40 hours. The 8.5 min run stopped at a 1,500-thread safety cap with
    the rate still flat, so the number is a rate, not an extrapolation.

    These tests count CALLS into the leaky enumerator, like the rest of the file
    — never threads, which are a property of the machine."""

    # 20 minutes of the tile compositor's 4 Hz read throttle. Long enough that a
    # per-tick leak and a per-TTL leak differ by two orders of magnitude, which
    # is the whole point: at 60 s they differ by 6 vs 240 and a slow fixture
    # could blur that.
    _SOAK_MINUTES = 20.0

    def setUp(self):
        self.bc._kinect_preview_webcam_idx.clear()
        self.bc._kinect_preview_webcam_resolved[0] = False
        self.bc._kinect_preview_webcam_resolved_at[0] = 0.0
        self.bc._kinect_preview_webcam_fingerprint[0] = None
        self.bc._kinect_preview_webcam_enumerated_at[0] = 0.0
        self.calls = []

    def _amplify(self, fingerprint, minutes=None, tick=None,
                 names=("Left Cam", "Right Cam"), clock_start=1000.0):
        """Fire _invalidate_side_tile_indices() + the resolver every `tick`
        seconds for `minutes` minutes — exactly what a camera that opens and
        then fails every read does to this path — and return
        (leaky enumerations, firings).

        `fingerprint` is one constant value or a LIST giving the value per
        FIRING (the last entry repeats), for the same reason the other fixtures
        in this file do it that way: a re-enumerating tick samples the probe
        twice, so a per-CALL generator desynchronises from ticks and the test
        starts measuring itself."""
        tick = self.bc._KINECT_PREVIEW_TILE_READ_INTERVAL if tick is None else tick
        minutes = self._SOAK_MINUTES if minutes is None else minutes
        fires = int(round(minutes * 60.0 / tick))
        t = [clock_start]
        schedule = (list(fingerprint) if isinstance(fingerprint, list)
                    else [fingerprint])
        i = [0]

        def _fp_now():
            return schedule[min(i[0], len(schedule) - 1)]

        def _enumerate():
            self.calls.append(1)
            return list(names)

        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               _enumerate), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               side_effect=_fp_now), \
             mock.patch.object(self.bc, "_kinect_preview_webcam_names",
                               return_value={"left": "left cam",
                                             "right": "right cam"}), \
             mock.patch.object(self.bc, "_report_video_fingerprint_gate",
                               return_value=False), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: t[0]):
            for k in range(fires):
                i[0] = k
                self.bc._invalidate_side_tile_indices()
                self.bc._resolve_webcam_indices_by_name()
                t[0] += tick
        return len(self.calls), fires

    def _ttl_bound(self, minutes=None):
        """The most enumerations the TTL floor can permit over `minutes`."""
        minutes = self._SOAK_MINUTES if minutes is None else minutes
        return int(minutes * 60.0 / self.bc._WEBCAM_IDX_TTL_SEC) + 1

    def test_the_fixture_really_does_fire_the_amplifier_every_tick(self):
        """FIRST, PROVE THE SOAK IS A SOAK. This is the check the 2026-09-06
        sick-camera run could not make: 3 amplifier firings in 100 minutes were
        reported as evidence about a mechanism that fires 240 times a minute
        when it fires at all. Assert the firing COUNT before asserting anything
        about what it cost."""
        _n, fires = self._amplify(_fp(), minutes=1.0)
        self.assertEqual(fires, 240,
                         "the fixture fired the amplifier %d times in a minute; "
                         "_KINECT_PREVIEW_TILE_READ_INTERVAL says 240" % fires)

    def test_twenty_minutes_of_the_amplifier_costs_one_enumeration(self):
        """THE HEALTHY-GATE CONTRACT, at load. 4,800 forced re-resolves — a
        camera that opens and fails every read for 20 minutes — must reach the
        leaky enumerator exactly once, because an unchanged fingerprint proves
        the indices cannot have reshuffled and re-enumeration could not fix a
        sick device anyway. Measured against the real COM probes on this rig the
        same shape gave 1 enumeration and +1 mfksproxy thread over 8,300
        firings."""
        n, fires = self._amplify(_fp())
        self.assertEqual(fires, 4800, "fixture drifted: %d firings" % fires)
        self.assertEqual(n, 1, "%d leaky enumerations for %d forced re-resolves "
                               "on an unchanged bus" % (n, fires))

    def test_a_failed_open_gate_does_not_let_the_amplifier_multiply_the_leak(self):
        """THE DEFECT THIS CLASS WAS WRITTEN FOR.

        When _video_device_fingerprint() returns None the resolver deliberately
        degrades to re-enumerating — the right SAFETY choice, and the module's
        alarm block describes the cost as "+6 OS threads and +618 handles per
        MINUTE". That description silently assumed the TTL was pacing the calls.
        It is not: _invalidate_side_tile_indices() clears the resolved flag, so
        the TTL early-return never fires and the degraded path ran at the TILE
        rate. Measured on this rig at 208 enumerations/min against 6.00/min with
        the amplifier quiet — 31x, and ~75 minutes to the v2.0.100 death numbers
        instead of ~40 hours.

        A forced invalidation may still bypass the TTL, but only once per TTL
        window. Over 20 minutes that is at most 121 enumerations, not 4,800."""
        n, fires = self._amplify(None)
        bound = self._ttl_bound()
        self.assertLessEqual(
            n, bound,
            "a failed-open gate under the amplifier made %d leaky enumerations "
            "in %.0f min (%d firings). The TTL permits at most %d; anything "
            "near %d means the forced path is bypassing the rate floor and the "
            "process is ~75 min from thread exhaustion."
            % (n, self._SOAK_MINUTES, fires, bound, fires))
        # ...and it must still be re-enumerating, not pinned: a probe that
        # cannot tell is never evidence that nothing moved.
        self.assertGreaterEqual(
            n, bound - 2,
            "the degraded gate made only %d enumerations in %.0f min — it has "
            "stopped self-healing, which is the wrong-camera bug the whole "
            "resolver exists to prevent" % (n, self._SOAK_MINUTES))

    def test_the_floor_is_what_bounds_it_not_the_fixture(self):
        """THE COUNTERFACTUAL, so the test above cannot pass for the wrong
        reason (a fixture whose clock never advances would also report a small
        number). Drive _WEBCAM_IDX_TTL_SEC to 0 and the SAME fixture must go
        back to one leaky enumeration per firing — i.e. it really is sensitive
        to the defect, and the bound above is the floor, not the harness."""
        with mock.patch.object(self.bc, "_WEBCAM_IDX_TTL_SEC", 0.0):
            n, fires = self._amplify(None, minutes=1.0)
        self.assertEqual(n, fires,
                         "with the floor removed the amplifier produced %d "
                         "enumerations for %d firings; the fixture is not "
                         "exercising the defect" % (n, fires))

    def test_the_first_failed_read_after_a_quiet_window_still_re_resolves(self):
        """THE OTHER HALF OF THE CONTRACT. The floor must not turn "a failed
        read re-resolves by name" into "a failed read is ignored". With the gate
        degraded and no enumeration for longer than the TTL, the very next
        forced invalidation has to reach the enumerator immediately — that is
        the 2026-07-14 behaviour the invalidation exists for."""
        n, _ = self._amplify(None, minutes=1.0)
        before = n
        # A quiet window longer than the TTL, then ONE failed read.
        n2, _ = self._amplify(
            None, minutes=self.bc._KINECT_PREVIEW_TILE_READ_INTERVAL / 60.0,
            clock_start=1000.0 + 60.0 + self.bc._WEBCAM_IDX_TTL_SEC * 3)
        self.assertEqual(n2 - before, 1,
                         "the first failed read after a quiet window did not "
                         "re-resolve (%d new enumerations)" % (n2 - before))

    def test_the_floor_never_delays_a_real_bus_change(self):
        """THE FLOOR APPLIES ONLY TO 'COULD NOT TELL'. A fingerprint that
        CHANGES is a real device joining, leaving or re-arriving, and DirectShow
        indices really can have reshuffled — so that must still re-enumerate on
        the very next tick even though the previous enumeration was
        milliseconds ago and the amplifier is hammering the path.

        This is the failure that would be quieter than the leak: the tiles would
        keep reading, from the wrong camera."""
        moved = _fp((("usb#vid_1&pid_1", b"\x77" * 8),))
        # 8 firings = 2 s, far inside the TTL. Change the bus on firing 4.
        schedule = [_fp()] * 4 + [moved] * 4
        n, fires = self._amplify(schedule, minutes=8 * 0.25 / 60.0)
        self.assertEqual(fires, 8, "fixture drifted: %d firings" % fires)
        self.assertEqual(n, 2, "a bus change inside the TTL window was not "
                               "re-enumerated (%d enumerations)" % n)

    def test_a_quiet_amplifier_is_not_evidence_about_a_loud_one(self):
        """THE REPORTING DEFECT, pinned so it cannot be made again. Three
        firings and 4,800 firings must be distinguishable from the numbers a
        soak collects; if the accounting cannot tell them apart, a run in which
        the camera healed reads exactly like a run in which the gate held.

        get_side_tile_gate_stats()['invalidations'] is that witness."""
        self.bc._side_tile_gate_counts["invalidations"] = 0
        self.bc._side_tile_gate_counts["enumerations"] = 0
        self._amplify(_fp(), minutes=1.0)
        loud = self.bc.get_side_tile_gate_stats()["invalidations"]
        self.assertEqual(loud, 240,
                         "the amplifier fired 240 times and the accounting saw "
                         "%d — a soak using these numbers could not tell a gate "
                         "that held from a camera that healed" % loud)


@requires_monolith
class SetupApiProbeReleaseTests(MonolithGlobalsTestCase):
    """The replacement probe opens a real OS handle (SetupDiGetClassDevs). The
    whole point of this change is not leaking, so it must destroy that handle on
    EVERY path - including the ones where the enumeration blows up mid-loop."""

    class _FakeFn:
        def __init__(self, impl):
            self.impl = impl
            self.calls = 0
            self.restype = None
            self.argtypes = None

        def __call__(self, *a):
            self.calls += 1
            return self.impl(*a)

    def _fake_windll(self, enum_impl):
        """A stand-in ctypes.windll where SetupDiEnumDeviceInterfaces behaves as
        `enum_impl`. Returns (fake, destroy_fn) so the test can assert on the
        destroy call count."""
        destroy = self._FakeFn(lambda *a: 1)
        setupapi = mock.Mock()
        setupapi.SetupDiGetClassDevsW = self._FakeFn(lambda *a: 0x1234)
        setupapi.SetupDiEnumDeviceInterfaces = self._FakeFn(enum_impl)
        setupapi.SetupDiGetDeviceInterfaceDetailW = self._FakeFn(lambda *a: 0)
        setupapi.SetupDiDestroyDeviceInfoList = destroy
        fake = mock.Mock()
        fake.setupapi = setupapi
        fake.cfgmgr32 = mock.Mock()
        fake.ole32 = mock.Mock()
        fake.ole32.CLSIDFromString = self._FakeFn(lambda *a: 0)
        return fake, destroy

    def test_handle_is_destroyed_when_enumeration_raises(self):
        def _boom(*_a):
            raise OSError("device tree yanked mid-enumeration")
        fake, destroy = self._fake_windll(_boom)
        with mock.patch.object(ctypes, "windll", fake):
            self.assertIsNone(self.bc._setupapi_camera_instances())
        self.assertGreaterEqual(destroy.calls, 1,
                                "SetupDiGetClassDevs handle leaked on the "
                                "exception path")

    def test_handle_is_destroyed_on_the_normal_path(self):
        fake, destroy = self._fake_windll(lambda *_a: 0)     # zero devices
        with mock.patch.object(ctypes, "windll", fake):
            self.bc._setupapi_camera_instances()
        # Two categories are probed, so two opens and two destroys.
        self.assertEqual(destroy.calls, 2)

    def test_unreadable_interface_detail_cannot_spin_forever(self):
        """SetupDiGetDeviceInterfaceDetailW returning 0 makes the loop `continue`.
        If the index were bumped after that instead of before it, this would
        hang the preview thread rather than fail a test."""
        state = {"n": 0}

        def _enum(*_a):
            state["n"] += 1
            return 1 if state["n"] <= 3 else 0       # 3 interfaces, then stop
        fake, _destroy = self._fake_windll(_enum)
        with mock.patch.object(ctypes, "windll", fake):
            got = self.bc._setupapi_camera_instances()
        self.assertEqual(got, ())                    # all details unreadable
        self.assertLessEqual(state["n"], 12, "enumeration did not terminate")


@requires_monolith
class SetupApiProbeShapeTests(MonolithGlobalsTestCase):
    """What the probe actually hands the gate, checked against the real OS.

    SHAPE ONLY -- no device counts and no stamp VALUES, because those are
    facts about the machine rather than about the code, and it skips outright
    when the host shows no PnP cameras."""

    def test_entries_carry_only_fields_that_can_actually_change(self):
        """THE REGRESSION GUARD for the removal stamp. An entry is
        ``(path, arrival-stamp)`` and nothing else. The third field this
        replaced was DEVPKEY_Device_LastRemovalDate, which is CLEARED when a
        device returns -- so a DIGCF_PRESENT probe reads b'' for it every
        time, on every device, forever. Re-adding it would put a
        provably-constant element back into a change detector and make the
        mechanism look twice as strong as it is."""
        got = self.bc._setupapi_camera_instances()
        if not got:
            self.skipTest("no PnP video-capture devices visible on this host")
        for entry in got:
            self.assertEqual(len(entry), 2, entry)
            self.assertIsInstance(entry[0], str)
            self.assertIsInstance(entry[1], bytes)
            # b'' is the documented downgrade signal; otherwise it must be a
            # stamp that could actually advance.
            if entry[1]:
                self.assertTrue(
                    self.bc._usable_arrival_stamp(entry[1]),
                    "probe emitted a stamp that can never move: %r" % (entry[1],))

    def test_probe_output_is_stable_when_the_bus_is_quiet(self):
        """Two back-to-back reads with no bus activity must be byte-identical,
        or the gate would re-enumerate (and leak) on every single tick. This
        is the property the whole fix rests on and it costs nothing to pin.

        It is NOT evidence that the stamp tracks re-arrivals -- that needs the
        stamp to be cycled and watched, which is written up in
        _setupapi_camera_instances. This only pins the quiet case."""
        first = self.bc._setupapi_camera_instances()
        if not first:
            self.skipTest("no PnP video-capture devices visible on this host")
        self.assertEqual(first, self.bc._setupapi_camera_instances())


@requires_monolith
class ProbesNeverRaiseTests(MonolithGlobalsTestCase):
    """Everything on this path is called from the preview thread, which has no
    error handling of its own - every probe returns None instead of raising."""

    def test_setupapi_probe_returns_none_when_the_api_is_missing(self):
        broken = mock.Mock()
        type(broken).setupapi = mock.PropertyMock(
            side_effect=AttributeError("no setupapi"))
        with mock.patch.object(ctypes, "windll", broken):
            self.assertIsNone(self.bc._setupapi_camera_instances())

    def test_software_camera_probe_returns_none_when_pygrabber_is_absent(self):
        import builtins
        real_import = builtins.__import__

        def _boom(name, *a, **k):
            if name.startswith("pygrabber"):
                raise ImportError("no pygrabber")
            return real_import(name, *a, **k)
        with mock.patch.object(builtins, "__import__", _boom):
            self.assertIsNone(self.bc._dshow_software_camera_names())

    def test_fingerprint_is_none_when_either_half_fails(self):
        with mock.patch.object(self.bc, "_setupapi_camera_instances",
                               return_value=None):
            self.assertIsNone(self.bc._video_device_fingerprint())
        with mock.patch.object(self.bc, "_dshow_software_camera_names",
                               return_value=None):
            self.assertIsNone(self.bc._video_device_fingerprint())


@requires_monolith
class GateFailOpenAlarmTests(MonolithGlobalsTestCase):
    """The gate degrading must not be SILENT.

    When ``_video_device_fingerprint()`` cannot answer, the resolver falls back
    to re-enumerating on the bare 10 s TTL - deliberately, because pinning an
    index nobody can prove is still correct is a worse bug (see
    ``test_unreadable_fingerprint_degrades_to_re_enumerating``). But that
    fallback IS the pre-fix leak, and until 2026-09-05 it happened without a
    single byte of output. Measured on this rig, same process, 30 TTL ticks
    each: real pygrabber -> 0 enumerations / +0 threads; pygrabber with
    ``SystemDeviceEnum`` hidden -> 30 enumerations / +30 threads = 6.00/min,
    the full pre-fix rate, and 0 bytes to stdout, stderr or the log. A reverted
    fix was therefore indistinguishable from a working one until the process
    died ~40 h later.

    These tests pin the ALARM, not the fallback: when it speaks, when it stays
    quiet, and that it never changes what the resolver actually does."""

    _NAMES = ("Left Cam", "Right Cam")

    def setUp(self):
        self.bc._kinect_preview_webcam_idx.clear()
        self.bc._kinect_preview_webcam_resolved[0] = False
        self.bc._kinect_preview_webcam_resolved_at[0] = 0.0
        self.bc._kinect_preview_webcam_fingerprint[0] = None
        self.bc._kinect_preview_webcam_enumerated_at[0] = 0.0
        self.bc._webcam_fingerprint_degraded[0] = False
        self.bc._webcam_fingerprint_degraded_since[0] = 0.0
        self.bc._webcam_fingerprint_warned_at[0] = 0.0
        self.bc._webcam_fingerprint_leaky_enums[0] = 0
        self.calls = []

    def _drive(self, n, fingerprint, names=_NAMES, clock_start=1000.0,
               pnp=(("a", b"\x01" * 8),), soft=("OBS Virtual Camera",)):
        """Run `n` TTL expiries; return (leaky enumerations, captured stdout).

        `fingerprint` is one value or a per-tick list (last entry repeats), and
        `names` is what the LEAKY enumerator returns - None models pygrabber
        being absent, which fails at the import and so leaks nothing. `pnp` /
        `soft` are what the two free probes answer when the alarm re-probes them
        to attribute the failure; both are mocked so the message never depends
        on the cameras plugged into the machine running the suite."""
        t = [clock_start]
        schedule = list(fingerprint) if isinstance(fingerprint, list) else [fingerprint]
        i = [0]

        def _enumerate():
            self.calls.append(1)
            return None if names is None else list(names)

        buf = io.StringIO()
        with mock.patch.object(self.bc, "_enumerate_dshow_input_devices",
                               side_effect=_enumerate), \
             mock.patch.object(self.bc, "_video_device_fingerprint",
                               side_effect=lambda: schedule[min(i[0], len(schedule) - 1)]), \
             mock.patch.object(self.bc, "_setupapi_camera_instances",
                               return_value=pnp), \
             mock.patch.object(self.bc, "_dshow_software_camera_names",
                               return_value=soft), \
             mock.patch.object(self.bc, "_kinect_preview_webcam_names",
                               return_value={"left": "left cam", "right": "right cam"}), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: t[0]), \
             contextlib.redirect_stdout(buf):
            for k in range(n):
                i[0] = k
                self.bc._resolve_webcam_indices_by_name()
                t[0] += self.bc._WEBCAM_IDX_TTL_SEC
        return len(self.calls), buf.getvalue()

    # -- the defect ----------------------------------------------------------
    def test_fail_open_is_announced_on_the_very_first_leaky_enumeration(self):
        """THE REGRESSION TEST. Before the fix this produced 5 leaked
        enumerations and an empty string."""
        n, out = self._drive(5, None)
        self.assertEqual(n, 5, "precondition: the gate must still be failing "
                               "open - this test is about the noise it makes")
        self.assertIn("FAILED OPEN", out)
        self.assertIn("[kinect-preview] WARNING", out)

    def test_the_alarm_quantifies_the_leak_it_is_reporting(self):
        """A warning that does not say what it costs gets ignored. The rate is
        derived from the live TTL, not hardcoded, so it stays true if the TTL
        moves."""
        _, out = self._drive(1, None)
        per_min = 60.0 / self.bc._WEBCAM_IDX_TTL_SEC
        self.assertIn("%.0f OS threads" % per_min, out)
        self.assertIn("%.0f handles" % (per_min * 103), out)

    def test_the_alarm_names_the_half_that_failed(self):
        """Without attribution the operator has a leak and no lead. The
        software half is the fragile one: it needs a strictly narrower
        pygrabber surface than the leaky call it guards, and no fallback."""
        _, out = self._drive(1, None, soft=None)
        self.assertIn("_dshow_software_camera_names", out)
        self.assertNotIn("_setupapi_camera_instances failed", out)
        self.setUp()
        _, out = self._drive(1, None, pnp=None)
        self.assertIn("_setupapi_camera_instances", out)

    def test_a_healthy_gate_says_nothing_about_leaking(self):
        """The alarm must not cry wolf on the normal path, or it trains the
        reader to ignore it."""
        _, out = self._drive(120, _fp())
        self.assertNotIn("FAILED OPEN", out)
        self.assertNotIn("WARNING", out)

    def test_absent_pygrabber_is_not_misreported_as_a_leak(self):
        """`names is None` means the enumeration failed at the IMPORT and never
        reached CreateClassEnumerator, so nothing leaked. The fingerprint is
        equally unreadable there, so an alarm keyed only on the fingerprint
        would fire a FALSE leak report on every machine without pygrabber."""
        n, out = self._drive(5, None, names=None)
        self.assertEqual(n, 5)
        self.assertNotIn("FAILED OPEN", out)

    # -- throttling ----------------------------------------------------------
    def test_the_alarm_is_throttled_but_does_not_fall_silent(self):
        """Once per tick would be 6 lines a minute forever - unreadable, and it
        would bury everything else. Once ever would leave a 40-hour degradation
        with one line at the top of a huge log. So: immediately, then every
        WEBCAM_FINGERPRINT_REWARN_SECONDS."""
        ticks = int(self.bc.WEBCAM_FINGERPRINT_REWARN_SECONDS * 2
                    / self.bc._WEBCAM_IDX_TTL_SEC)       # 2x the re-warn window
        _, out = self._drive(ticks, None)
        self.assertEqual(out.count("FAILED OPEN"), 2,
                         "expected one warning per re-warn window over two "
                         "windows, got:\n" + out)

    def test_the_running_cost_is_carried_in_the_repeat_warning(self):
        """The second line has to say how bad it has got, or a reader who joins
        late cannot tell a one-minute blip from a twenty-hour bleed."""
        ticks = int(self.bc.WEBCAM_FINGERPRINT_REWARN_SECONDS
                    / self.bc._WEBCAM_IDX_TTL_SEC) + 1
        _, out = self._drive(ticks, None)
        self.assertIn("%d leaked enumeration(s) so far" % ticks, out)

    # -- recovery ------------------------------------------------------------
    def test_recovery_is_announced_with_the_size_of_the_window(self):
        """A start with no end reads as 'still broken' forever. The recovery
        line closes the window and states the damage, which is the number that
        decides whether a restart is needed."""
        n, out = self._drive(7, [None] * 6 + [_fp()])
        self.assertIn("READABLE again", out)
        self.assertIn("6 leaky enumeration(s)", out)
        self.assertIn("618 handles", out)                        # 6 x 103
        self.assertEqual(n, 7, "the recovering tick still had to enumerate - "
                               "it had no stored fingerprint to compare to")

    def test_recovery_is_announced_once_not_on_every_healthy_tick(self):
        _, out = self._drive(200, [None, None] + [_fp()])
        self.assertEqual(out.count("READABLE again"), 1)

    def test_a_gate_that_was_never_degraded_never_announces_a_recovery(self):
        _, out = self._drive(3, _fp())
        self.assertNotIn("READABLE again", out)

    def test_a_second_degradation_after_a_recovery_warns_again(self):
        """The throttle stamp must not outlive the episode it throttled, or a
        gate that fails, recovers and fails again goes quiet for fifteen
        minutes at exactly the moment it matters."""
        _, out = self._drive(4, [None, _fp(), None, None])
        self.assertEqual(out.count("FAILED OPEN"), 2)

    # -- it must never make things worse -------------------------------------
    def test_the_alarm_never_raises(self):
        """It runs inside the HUD compositor's per-frame path. A probe that
        explodes while we are already degraded must not take the preview with
        it."""
        buf = io.StringIO()
        with mock.patch.object(self.bc, "_setupapi_camera_instances",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(self.bc, "_dshow_software_camera_names",
                               side_effect=RuntimeError("boom")), \
             contextlib.redirect_stdout(buf):
            self.assertTrue(self.bc._report_video_fingerprint_gate(
                None, ["Left Cam"], 1000.0))
        self.assertIn("BOTH halves failed", buf.getvalue())
        # ...and junk arguments are survivable too (`now` is not a number).
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(self.bc._report_video_fingerprint_gate(
                None, None, object()))

    def test_the_alarm_does_not_change_what_the_resolver_returns(self):
        """Pure observation. The degraded path still resolves the map from the
        live enumeration, exactly as it did before the alarm existed."""
        self._drive(1, None)
        self.assertEqual(self.bc._kinect_preview_webcam_idx,
                         {"left": 0, "right": 1})


@requires_monolith
class SickCameraReopenIsNotGatedTests(MonolithGlobalsTestCase):
    """WHAT THIS CHANGE DOES **NOT** FIX - pinned, because the report said it
    did.

    v2.0.100 died with a SICK camera, and in that regime almost none of the leak
    comes from _resolve_webcam_indices_by_name(). The side-tile loop opens a
    handle, the read fails, the handle is released, the name->index memo is
    invalidated and the next composite tick opens again. The expensive call in
    that cycle is cv2.VideoCapture(idx, CAP_DSHOW) inside _open_tile_capture(),
    which no fingerprint can gate: OpenCV's DirectShow backend enumerates the
    video-input category itself to resolve index N, and enumerates it twice more
    re-negotiating the 640x480 mode.

    Measured on this rig 2026-09-05 (threads whose Win32 start address is inside
    mfksproxy.dll, per call, 1 s settle either side):

        _enumerate_dshow_input_devices()                   +1.00 thr / +115 hnd
        _open_tile_capture() + _release_capture_guarded()  +5.00 thr / +536 hnd

    An interleaved A/B in one process (8 blocks x 20 cycles, real gate vs the
    pre-fix behaviour, so device drift lands on both arms) measured 3.05
    threads/cycle gated against 4.01 ungated - a mean paired difference of 0.96,
    which is the one enumeration and nothing else. The gate removes ~24% of the
    sick-camera leak. A 25-minute single-arm soak of the gated tree measured
    4.86 threads/cycle at 13.9 cycles/min = 67.6 mfksproxy threads/min over its
    first 19.5 minutes - the same rate as the incident itself.

    What DOES bound the sick camera is the camera quarantine, and the division
    of labour is worth stating precisely, because the review that produced this
    class got it half wrong in the other direction: the tile path scores no
    strike of its own (test 2), but the face-track loop it runs on scores
    strikes for the SAME index and does bench it (test 3) - unless the name
    resolved to a different index than the one the loop is benching (test 4),
    which is the whole reason resolve-by-name exists.

    Like the rest of this file, these tests count CALLS, never threads."""

    class _SickCap:
        """Opens fine, delivers ONE frame, then every read fails - the
        v2.0.100 symptom. A camera that will not OPEN never reaches the
        invalidate/reopen path at all (_read_side_tile_webcams `continue`s on
        a None capture), so an out-of-range index is the wrong stand-in for a
        sick one.

        THE FIRST FRAME WAS ADDED 2026-09-05 and is not a softening of the
        fixture — it makes it match the device. The owner's sick webcam does
        not fail every read; it delivers, then drops one every ~10 s to ~9 min.
        It also matters now that the openers prove a frame before handing a
        handle back (a Media Foundation open of a camera another process holds
        reports isOpened() True and then delivers nothing, so "opened" stopped
        being evidence). A cap that never reads at all is that OTHER shape, and
        under the current code it is correctly refused at open — which would
        make this class silently measure zero reopens instead of the churn it
        exists to pin."""

        def __init__(self):
            self.releases = 0
            self._first = True

        def isOpened(self):
            return True

        def set(self, *_a):
            return True

        def read(self, *_a):
            if self._first:
                self._first = False
                import numpy as _np
                return True, _np.zeros((8, 8, 3), dtype=_np.uint8)
            return False, None

        def release(self):
            self.releases += 1

    class _Cv2Shim:
        """Stands in for the module-level `cv2` inside _open_tile_capture and
        records every VideoCapture() - the ungated leaky call."""

        CAP_DSHOW = 700
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_BUFFERSIZE = 38

        def __init__(self, log, cap_factory):
            self.log = log
            self._cap_factory = cap_factory

        def VideoCapture(self, idx, backend=None):
            self.log.append((idx, backend))
            return self._cap_factory()

    def setUp(self):
        bc = self.bc
        # NONE of the state below is in the harness's restore set, so reset it
        # here AND in tearDown.
        #
        # _camera_latest_frame / _camera_last_frame_at are the load-bearing
        # pair: they are the face-track loop's FAST PATH, and
        # test_monolith_camera_quarantine really does leave real frames in them
        # stamped with real wall-clock time. Against this class's frozen 1000.0
        # clock `now - seen_at` is then hugely NEGATIVE, the frame reads as
        # fresh, and every test here quietly measures zero opens - green for the
        # exact reason the module docstring warns about. Snapshot and clear.
        self._saved_frames = dict(bc._camera_latest_frame)
        self._saved_frame_at = dict(bc._camera_last_frame_at)
        bc._camera_latest_frame.clear()
        bc._camera_last_frame_at.clear()
        # A capture left in _kinect_tile_caps would make the next test REUSE a
        # handle instead of opening one - the same failure with a different
        # cause.
        self._reset_tiles()
        bc.CAMERAS[:] = [{"index": 0, "label": "Left webcam", "name": "left cam",
                          "primary": True, "look_x": 0.5, "look_y": 0.5}]
        bc._kinect_preview_webcam_idx.clear()
        bc._kinect_preview_webcam_resolved[0] = False
        bc._kinect_preview_webcam_resolved_at[0] = 0.0
        bc._kinect_preview_webcam_fingerprint[0] = None
        bc._kinect_preview_webcam_enumerated_at[0] = 0.0

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

    def _drive(self, ticks, devices=("Left Cam",), t0=1000.0, step=0.25):
        """Run `ticks` composite ticks against a sick camera and return
        (leaky enumerations, ungated DirectShow reopens).

        `step` is _KINECT_PREVIEW_TILE_READ_INTERVAL, the only throttle the code
        imposes on this path (4 Hz per slot). What was actually MEASURED on this
        rig is slower - 13.9 cycles/min - because a real DirectShow open on the
        sick device takes 1-2 s of its own. 4 Hz is the ceiling the code allows,
        not a rate anyone has seen."""
        bc = self.bc
        enums, opens = [], []
        t = [t0]

        def _enumerate():
            enums.append(1)
            return list(devices)

        shim = self._Cv2Shim(opens, self._SickCap)
        with mock.patch.object(bc, "_enumerate_dshow_input_devices", _enumerate), \
             mock.patch.object(bc, "_video_device_fingerprint", return_value=_fp()), \
             mock.patch.object(bc, "cv2", shim), \
             mock.patch.object(bc.time, "time", side_effect=lambda: t[0]):
            for _ in range(ticks):
                bc._read_side_tile_webcams(t[0])
                t[0] += step
        return len(enums), len(opens)

    def test_the_gate_removes_the_enumeration_but_not_the_reopen(self):
        """THE HEADLINE CORRECTION. 40 ticks of a sick camera: the fingerprint
        gate does its job (one enumeration, not forty) and the reopen storm is
        completely untouched (forty). One enumeration removed, forty reopens
        left - which is why the interleaved A/B above measures a ~24% reduction
        in the regime that actually killed v2.0.100, not 54x."""
        enums, opens = self._drive(40)
        self.assertEqual(enums, 1,
                         "the gate stopped working: %d enumerations" % enums)
        self.assertEqual(opens, 40,
                         "this test is no longer measuring the ungated reopen "
                         "(%d opens for 40 failed reads). If the reopen really "
                         "did get gated, that is good news - but the module "
                         "docstring's arithmetic and the fix report's headline "
                         "both have to be rewritten with it." % opens)

    def test_a_failed_read_scores_no_quarantine_strike(self):
        """The side-tile path cannot bench a camera on its own. The three
        _camera_note_sick_cycle() call sites are the bounded-open TIMEOUT and
        two inside _face_tracking_thread_body(); a read that merely returns
        False reaches none of them, so nothing in _read_side_tile_webcams
        escalates however long the storm runs."""
        bc = self.bc
        with mock.patch.object(bc, "_camera_note_sick_cycle") as strike:
            _enums, opens = self._drive(50)
        self.assertEqual(opens, 50)
        self.assertEqual(strike.call_count, 0,
                         "the tile path now scores its own strikes - good, but "
                         "the docstring above says it does not")

    def test_a_benched_index_is_what_actually_stops_the_storm(self):
        """The backstop is real, and it is the only one: the face-track loop the
        tile path runs on scores strikes for the same index (25 consecutive read
        failures plus 2 s of silence per strike, _CAMERA_QUARANTINE_STRIKES of
        them), and once benched _read_side_tile_webcams makes no DirectShow call
        at all. Pinned so a future change that stops benching this index cannot
        silently reopen the storm."""
        bc = self.bc
        for _ in range(bc._CAMERA_QUARANTINE_STRIKES):
            bc._camera_note_sick_cycle(0, "Left webcam", "soft wake", 1000.0)
        self.assertTrue(bc._camera_is_quarantined(0, 1000.0),
                        "precondition: the loop's strikes must bench index 0")
        enums, opens = self._drive(50)
        self.assertEqual(opens, 0, "a benched camera was reopened %d times" % opens)
        # ...but not for free: _read_side_tile_webcams resolves the name->index
        # map BEFORE it checks the bench, so a benched slot still pays the ONE
        # cold-start enumeration. Asserted rather than glossed over - it is 1
        # leaked thread per benching, not 0, and the gate is the only reason it
        # does not become 1 per tick.
        self.assertEqual(enums, 1,
                         "expected exactly the cold-start enumeration, got %d" % enums)

    def test_a_bench_misses_a_tile_whose_name_resolved_elsewhere(self):
        """THE HOLE IN THE BACKSTOP, and the reason it is a backstop and not a
        fix. The loop benches cam['index']; the tile opens whatever the NAME
        resolved to. A bus reshuffle makes those differ - which is the entire
        reason _resolve_webcam_indices_by_name() exists - and the bench then
        protects an index nobody is opening while the storm continues on the one
        that moved."""
        bc = self.bc
        for _ in range(bc._CAMERA_QUARANTINE_STRIKES):
            bc._camera_note_sick_cycle(0, "Left webcam", "soft wake", 1000.0)
        self.assertTrue(bc._camera_is_quarantined(0, 1000.0))
        # The device is now second on the bus, so the name resolves to index 1.
        enums, opens = self._drive(20, devices=("Other Cam", "Left Cam"))
        self.assertEqual(enums, 1)
        self.assertEqual(opens, 20,
                         "the reshuffled tile was benched after all - if that is "
                         "a real fix rather than an accident of this fixture, "
                         "say so here")


if __name__ == "__main__":
    unittest.main()
