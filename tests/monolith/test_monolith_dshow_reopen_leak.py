"""The DirectShow-enumeration leak, from the CAMERA REOPEN LOOP's side.

Scope note, so this file does not rot into a duplicate. The gate MECHANICS --
what ``_dshow_input_devices_gated`` does with a fingerprint, the floors, cache
isolation, the copy-on-return contract -- are covered in
``test_monolith_dshow_open_path_gate.py`` (open path) and
``test_monolith_dshow_enum_leak.py`` (side-tile path). Do not re-assert them
here. What lives here is the part neither of those can see:

  1. NOTHING NEW MAY CALL THE LEAKY PRIMITIVE UNGATED. The defect was never a
     wrong line; it was a call site the gate did not know about. The first pass
     gated ``_resolve_webcam_indices_by_name`` and left ``_dshow_name_to_index``
     -- on the camera (re)open path -- calling
     ``_enumerate_dshow_input_devices()`` directly. Every gate-behaviour test in
     both other files passed the whole time. Only a structural check catches the
     third one.

  2. THE REOPEN LOOP IS UNBOUNDED, which is the entire reason (1) matters on
     this path rather than being a slow-path nicety.

  3. GATING MUST NOT COST THE OPEN-BY-NAME GUARANTEE -- an index shuffle has to
     still move the camera, or the fix trades a loud leak for a silent
     wrong-camera read.

Measured on this rig 2026-09-05, and the numbers the rest of this file quotes:
``_dshow_name_to_index`` cost +1.00 OS thread / +103.1 handles per call for a
PRESENT name and +1.05 / +103.2 for an ABSENT one (n=20 each, 3 s settle) --
the enumeration runs before the name comparison, so a camera that is gone pays
full price. An absent CAP_DSHOW index fails in ~35 ms here, not the 20-30 s
quoted elsewhere in the monolith, so nothing throttles the retry loop.

No test here asserts a thread or handle count: those are facts about the
machine and would be flaky by construction. The live before/after measurement
belongs in a soak.
"""
from __future__ import annotations

import ast
import io
import os
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


def _fp(devices=(("usb#vid_1&pid_1&mi_00", b"\x01" * 8),),
        software=("OBS Virtual Camera",)):
    """A fingerprint shaped like the real one: ((pnp...), (software...)), each
    pnp entry ``(instance-path, arrival-stamp)``.

    NO removal stamp. Windows clears DEVPKEY_Device_LastRemovalDate when the
    device returns and the probe enumerates DIGCF_PRESENT, so it could never
    read one -- it was always b'' and is no longer collected."""
    return (tuple(devices), tuple(software))


# The device list as this rig actually reports it (verified live 2026-09-05 via
# pygrabber): list position IS the cv2 CAP_DSHOW index.
_RIG_NAMES = ["USB 2.0 Camera", "Kinect V2 Video Sensor",
              "HD Webcam eMeet C960", "OBS Virtual Camera"]
# The same devices after a USB re-enumeration moved the eMeet to index 0.
_RIG_NAMES_SHUFFLED = ["HD Webcam eMeet C960", "USB 2.0 Camera",
                       "Kinect V2 Video Sensor", "OBS Virtual Camera"]


@requires_monolith
class NoUngatedEnumeratorCallSitesTests(MonolithGlobalsTestCase):
    """(1) Pin the call sites of the leaky primitive, structurally."""

    #: The only functions allowed to call ``_enumerate_dshow_input_devices``
    #: directly. Everything else must go through ``_dshow_input_devices_gated``.
    _ALLOWED = {
        # The gate itself.
        "_dshow_input_devices_gated",
        # The side-tile resolver carries its own inline copy of the gate RULE:
        # it has to stamp _kinect_preview_webcam_fingerprint and hand
        # _report_video_fingerprint_gate the real enumeration result, neither of
        # which the shared helper does. It is gated -- just not through the
        # shared helper. If it is ever routed through
        # _dshow_input_devices_gated, delete this entry. If a NEW name appears
        # here, that is this file's defect coming back.
        "_resolve_webcam_indices_by_name",
    }

    def test_every_direct_caller_of_the_leaky_enumerator_is_gated(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "bobert_companion.py")
        with io.open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), path)
        callers = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and sub.func.id == "_enumerate_dshow_input_devices"):
                    callers.add(node.name)
        self.assertTrue(callers,
                        "found no callers at all — the AST scan broke, not the "
                        "code")
        self.assertEqual(
            callers, self._ALLOWED,
            "the set of DIRECT callers of the leaky DirectShow enumeration "
            "changed. Every call leaks +1 OS thread and +103 handles "
            "permanently, and nothing reclaims them short of a restart, so a "
            "new direct caller must take its own [names, fingerprint, "
            "enumerated_at] cache through _dshow_input_devices_gated() — or, if "
            "it genuinely is a one-shot (a CLI listing, a boot probe that runs "
            "once), be added here deliberately. Unexpected: "
            f"{sorted(callers - self._ALLOWED)}; missing: "
            f"{sorted(self._ALLOWED - callers)}")


@requires_monolith
class FailedReopenNeverRequarantinesTests(MonolithGlobalsTestCase):
    """(2) WHY an ungated call on this path was unbounded rather than rare.

    These tests do not assert a bug. They PIN the behaviour the leak arithmetic
    depends on: a cleanly-failing reopen scores no quarantine strike, so the
    bench never re-arms and the retry loop runs at ``CAMERA_REOPEN_BACKOFF_SEC``
    forever. All three ``_camera_note_sick_cycle`` call sites need either a
    provably-wedged open (whose sibling branch explicitly prints "No strike is
    scored against this camera") or read failures on an ALREADY-OPEN handle;
    the recovery branch's ``_open_capture() -> None`` path calls none of them,
    and strikes decay after ``_CAMERA_QUARANTINE_WINDOW_S``.

    If a later change makes a failed reopen score a strike, that is probably an
    improvement -- but the retry rate then collapses from ~29/min to roughly one
    attempt per bench, and every "per minute" number in these files stops being
    true. Whoever makes that change should have to update them on purpose
    rather than discover it in a soak.
    """

    IDX = 4242

    def test_a_cleanly_failed_reopen_scores_no_quarantine_strike(self):
        bc = self.bc
        t = [1000.0]
        entry = {"next_reopen_at": 0.0, "contention_logged": False}
        with mock.patch.object(bc.time, "time", side_effect=lambda: t[0]), \
             mock.patch.object(bc, "find_camera_locking_processes",
                               return_value=[]):
            for _ in range(50):
                # This IS the entire failure handling of the recovery branch:
                # _open_capture returned None, so re-arm the backoff and move on.
                bc._schedule_camera_reopen(entry, "ghost", self.IDX, t[0])
                t[0] += bc.CAMERA_REOPEN_BACKOFF_SEC
            q = bc.get_camera_quarantine().get(self.IDX, {})
        self.assertFalse(q.get("quarantined", False))
        self.assertEqual(q.get("quarantine_strikes", 0), 0,
                         "a failed reopen now scores a strike — the reopen rate "
                         "these files' numbers assume is no longer right")

    def test_an_expired_bench_never_re_arms_while_reopens_keep_failing(self):
        """The full sequence: reads fail on an open handle -> 3 strikes -> the
        camera is benched -> the bench expires -> every reopen from then on
        fails cleanly, and the bench never comes back."""
        bc = self.bc
        t = [1000.0]
        with mock.patch.object(bc.time, "time", side_effect=lambda: t[0]), \
             mock.patch.object(bc, "find_camera_locking_processes",
                               return_value=[]):
            for _ in range(bc._CAMERA_QUARANTINE_STRIKES):
                bc._camera_note_sick_cycle(self.IDX, "ghost",
                                           "read failures", t[0])
                t[0] += 1.0
            benched = bc.get_camera_quarantine()[self.IDX]
            self.assertTrue(benched["quarantined"], "precondition: benched")

            t[0] = benched["quarantine_until"] + 0.1
            entry = {"next_reopen_at": 0.0, "contention_logged": False}
            attempts = 0
            rebenched = 0
            end = t[0] + 30 * 60.0
            while t[0] < end:
                if bc._camera_is_quarantined(self.IDX, t[0]):
                    rebenched += 1
                    t[0] += 1.0
                    continue
                if t[0] < entry["next_reopen_at"]:
                    t[0] += 0.1
                    continue
                bc._schedule_camera_reopen(entry, "ghost", self.IDX, t[0])
                attempts += 1
                t[0] += 0.1
        self.assertEqual(rebenched, 0,
                         "the bench re-armed — good, but the leak arithmetic in "
                         "these files assumes it does not; update it")
        # 30 virtual minutes at the 2.0 s spacing (plus 0.1 s of loop each).
        self.assertGreater(attempts, 800,
                           f"only {attempts} reopen attempts in 30 minutes")


@requires_monolith
class GatingDoesNotCostTheShuffleGuaranteeTests(MonolithGlobalsTestCase):
    """(3) The did-the-fix-break-something half.

    ``_dshow_name_to_index`` exists so a USB re-enumeration cannot leave the
    face tracker pointed at the WRONG camera, and its docstring used to promise
    a fresh enumeration on every call. A cache could quietly destroy that, and
    the failure would be silent: a wrong-camera read SUCCEEDS, so no backstop
    anywhere fires. The sibling files assert that a changed fingerprint forces a
    re-enumeration; this asserts the thing that actually matters end to end --
    the returned INDEX moves with the device."""

    def setUp(self):
        self.bc._dshow_open_devices_cache[:] = [None, None, 0.0]

    def test_an_index_shuffle_moves_the_index_the_opener_gets(self):
        bc = self.bc
        # Same three cameras, but the bus moved: a device joined, which is what
        # reshuffles DirectShow order in the first place.
        moved = _fp((("usb#vid_1&pid_1&mi_00", b"\x01" * 8),
                     ("usb#vid_9&pid_9&mi_00", b"\x09" * 8)))
        t = [1000.0]
        names = [list(_RIG_NAMES)]
        fps = [_fp()]
        calls = []

        def _enum():
            calls.append(1)
            return list(names[0])

        seen = []
        with mock.patch.object(bc, "_enumerate_dshow_input_devices", _enum), \
             mock.patch.object(bc, "_video_device_fingerprint",
                               side_effect=lambda: fps[0]), \
             mock.patch.object(bc.time, "time", side_effect=lambda: t[0]):
            seen.append(bc._dshow_name_to_index("emeet c960"))
            t[0] += bc.CAMERA_REOPEN_BACKOFF_SEC
            seen.append(bc._dshow_name_to_index("emeet c960"))   # served cached
            names[0] = list(_RIG_NAMES_SHUFFLED)                 # bus moves
            fps[0] = moved
            t[0] += bc.CAMERA_REOPEN_BACKOFF_SEC
            seen.append(bc._dshow_name_to_index("emeet c960"))
        self.assertEqual(seen, [2, 2, 0],
                         "the opener kept handing out the pre-shuffle index — "
                         "it would open the wrong camera, and the read would "
                         "succeed")
        self.assertEqual(len(calls), 2,
                         "the shuffle cost %d enumerations, not 1 before + 1 "
                         "after" % len(calls))


if __name__ == "__main__":
    unittest.main()
