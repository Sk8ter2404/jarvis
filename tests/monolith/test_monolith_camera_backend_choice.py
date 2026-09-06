"""Every live camera open in the monolith goes through _camera_open(), and
_camera_open() never hands a raw DirectShow index to Media Foundation.

────────────────────────────────────────────────────────────────────────────
WHY
────────────────────────────────────────────────────────────────────────────
v2.0.101 gated the DirectShow *enumeration* leak. The *open* leaks too, and no
gate can help there because the face-track loop has to open the camera.
Measured on the owner's rig 2026-09-05, 25 open/read/release cycles per figure,
each in a fresh process, 1280x720 requested, 30 reads per cycle:

    eMeet C960   CAP_DSHOW  +4.58 OS threads  +479 handles  per cycle
    eMeet C960   CAP_MSMF   -0.21 OS threads    +0.97       per cycle
    USB 2.0 Cam  CAP_DSHOW  +4.55 OS threads  +479 handles  per cycle
    USB 2.0 Cam  CAP_MSMF   -0.18 OS threads    +1.06       per cycle

A dead index costs +103 handles under DirectShow and +0 under Media
Foundation, so the 12-index probe sweep paid it too.

────────────────────────────────────────────────────────────────────────────
THE PART THAT IS ACTUALLY DANGEROUS
────────────────────────────────────────────────────────────────────────────
The backends enumerate DIFFERENT lists, of different lengths (verified on this
rig with both enumerators and confirmed by device-unique resolutions):

    DirectShow  0=USB 2.0 Camera  1=Kinect  2=eMeet C960  3=OBS Virtual Camera
    MediaFdn.   0=Kinect          1=USB 2.0 Camera  2=eMeet C960

CAMERAS on disk holds DIRECTSHOW indices. A `CAP_DSHOW -> CAP_MSMF` edit with
the index left alone repoints index 0 from the USB webcam to the Kinect AND
SUCCEEDS. Nothing raises, nothing logs, the face tracker just watches the wrong
camera. That is the regression these tests exist to make impossible.

DELIBERATELY NOT ASSERTED: thread counts, handle counts, this machine's device
list, or how fast anything is. Those are facts about a rig. What is asserted is
the CALL — which opener each site reaches for, which index and backend the
resolver picks, and that a frame is proven where a frame matters. Same rule
tests/monolith/test_monolith_dshow_open_path_gate.py follows.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


DSHOW_NAMES = ["USB 2.0 Camera", "Kinect V2 Video Sensor",
               "HD Webcam eMeet C960", "OBS Virtual Camera"]
MSMF_NAMES = ("Kinect V2 Video Sensor", "USB 2.0 Camera",
              "HD Webcam eMeet C960")


def _executable_source(fn) -> str:
    """``fn``'s source with comments AND string literals removed.

    Docstrings are stripped deliberately: several of these functions still
    DESCRIBE the old DirectShow behaviour in prose, and that history is worth
    keeping. A naive substring search over raw source cannot tell a description
    from a call. What must never come back is the CALL.

    WHITESPACE IS STRIPPED TOO, and that is not cosmetic. The first version of
    this helper joined tokens with a space, which turned every
    ``cv2.VideoCapture(`` into ``cv2 . VideoCapture (`` — so the assertions
    below could never have failed. A test that cannot fail is worse than no
    test, and this file exists precisely to catch a silent reintroduction."""
    import inspect
    import io
    import tokenize
    src = inspect.getsource(fn)
    skip = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
            tokenize.INDENT, tokenize.DEDENT, tokenize.FSTRING_START,
            tokenize.FSTRING_MIDDLE, tokenize.FSTRING_END}
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in skip:
                continue
            out.append(tok.string.strip())
    except tokenize.TokenError:      # pragma: no cover - defensive
        return src
    return "".join(out)


class _Cap:
    def __init__(self, opened=True, frames=10 ** 9):
        self._opened = opened
        self._frames = frames
        self.released = 0

    def isOpened(self):
        return self._opened

    def set(self, *a):
        return True

    def read(self):
        if self._frames <= 0:
            return False, None
        self._frames -= 1
        import numpy as np
        return True, np.zeros((4, 4, 3), dtype="uint8")

    def release(self):
        self.released += 1


@requires_monolith
class CameraOpenRoutingTests(MonolithGlobalsTestCase):
    """_camera_open(): the one place backend + index are decided."""

    def _patch_backend(self, msmf_names=MSMF_NAMES, backend="msmf"):
        bc = self.bc
        return (
            mock.patch.object(bc._camera_backend, "msmf_device_names",
                              side_effect=lambda *a, **k: msmf_names),
            mock.patch.object(bc._camera_backend, "configured_backend",
                              return_value=backend),
        )

    def test_named_camera_is_translated_and_opened_on_msmf(self):
        bc = self.bc
        p1, p2 = self._patch_backend()
        with p1, p2, \
                mock.patch.object(bc._camera_backend, "open_camera") as oc, \
                mock.patch.object(bc, "_dshow_input_devices_gated") as gated:
            oc.return_value = _Cap()
            bc._camera_backend_notes.clear()
            bc._camera_open(0, name="usb 2.0 camera", width=1280, height=720)
        # DirectShow index 0 is the USB webcam; Media Foundation calls it 1.
        self.assertEqual(oc.call_args.args[0], 1)
        self.assertEqual(oc.call_args.kwargs["backend"], "msmf")
        # And the leaky enumeration was never needed at all.
        gated.assert_not_called()

    def test_unnamed_index_pays_for_the_gated_enumeration_only_as_fallback(self):
        bc = self.bc
        p1, p2 = self._patch_backend()
        with p1, p2, \
                mock.patch.object(bc._camera_backend, "open_camera") as oc, \
                mock.patch.object(bc, "_dshow_input_devices_gated",
                                  return_value=list(DSHOW_NAMES)) as gated:
            oc.return_value = _Cap()
            bc._camera_backend_notes.clear()
            bc._camera_open(0)
        self.assertEqual(gated.call_count, 1)
        self.assertEqual(oc.call_args.args[0], 1)
        self.assertEqual(oc.call_args.kwargs["backend"], "msmf")

    def test_directshow_only_device_stays_on_directshow_at_its_own_index(self):
        bc = self.bc
        p1, p2 = self._patch_backend()
        with p1, p2, \
                mock.patch.object(bc._camera_backend, "open_camera") as oc, \
                mock.patch.object(bc, "_dshow_input_devices_gated",
                                  return_value=list(DSHOW_NAMES)):
            oc.return_value = _Cap()
            bc._camera_backend_notes.clear()
            bc._camera_open(3)          # OBS Virtual Camera: no MF presence
        self.assertEqual(oc.call_args.args[0], 3)
        self.assertEqual(oc.call_args.kwargs["backend"], "dshow")

    def test_the_backend_decision_reaches_the_session_log(self):
        """The decision must be SAID, and by a thread allowed to say it.

        Two constraints that a first attempt got wrong in both directions.
        Printing it from _camera_open() put an unabandonable claim on the
        throwaway open worker; routing it to logging.info() made it vanish
        entirely, because the daemon Tees stdout/stderr into
        logs/session_*.log and installs NO logging handler — a live session log
        grepped for logging-formatted lines on 2026-09-05 contained zero. So
        _camera_open() queues and a joiner prints."""
        bc = self.bc
        p1, p2 = self._patch_backend()
        with p1, p2, \
                mock.patch.object(bc._camera_backend, "open_camera",
                                  return_value=_Cap()):
            bc._camera_backend_notes.clear()
            bc._camera_backend_pending.clear()
            bc._camera_open(0, name="usb 2.0 camera", label="right webcam")
            # Nothing said yet: the worker only queued it.
            queued = list(bc._camera_backend_pending)
            self.assertEqual(len(queued), 1, queued)
            self.assertIn("MSMF", queued[0])
            self.assertIn("index 1", queued[0])

            printed = []
            with mock.patch("builtins.print", printed.append):
                bc._drain_camera_backend_notes()
            self.assertEqual(len(printed), 1, printed)
            self.assertIn("MSMF", printed[0])
            self.assertEqual(len(bc._camera_backend_pending), 0)

    def test_each_call_site_announces_its_own_open_of_the_same_camera(self):
        """Two DIFFERENT openers of one camera must both be named.

        The dedupe key used to be (index, name), and `name` is the CAMERA, not
        the caller — so the first call site to open a webcam was the only one
        this process ever announced, for its whole life. Every later opener of
        that same device (the side tile compositor, the probe sweep, a
        snapshot) hashed to the same key, found the same (index, backend), and
        said nothing.

        What that hid, measured 2026-09-06 over the 8 consecutive restarts that
        opened the right webcam (session logs 2026-09-05_22-37-30 through
        2026-09-06_06-12-11): every restart logs TWO right-camera read failures
        inside 90 s — at camera-open +7..13 s and +59..63 s — and only the
        second has a scheduler line behind it (the self-diag boot sweep, queued
        8-10 s BEFORE the camera opens and firing 60 s later). The boot log
        carried exactly ONE "[camera] ...: opening with MSMF" line for that
        device, so the +9 s failure had no opener named anywhere and read as
        the camera itself dying. It is not a rate this test asserts — it is
        that the log can still answer "who else opened this camera"."""
        bc = self.bc
        p1, p2 = self._patch_backend()
        with p1, p2, \
                mock.patch.object(bc._camera_backend, "open_camera",
                                  return_value=_Cap()):
            bc._camera_backend_notes.clear()
            bc._camera_backend_pending.clear()
            # The face-track producer and the side-tile compositor open the
            # SAME physical camera: same DirectShow index, same configured
            # name. Only the label — the call site — differs.
            bc._camera_open(0, name="usb 2.0 camera", label="usb 2.0 camera")
            bc._camera_open(0, name="usb 2.0 camera",
                            label="side tile index 0")
            queued = list(bc._camera_backend_pending)
        self.assertEqual(len(queued), 2, queued)
        self.assertTrue(any("side tile index 0" in q for q in queued),
                        f"the second call site went unannounced: {queued}")

    def test_one_call_site_reopening_forever_still_narrates_once(self):
        """The dedupe still has to hold: a per-frame tile open must not
        re-narrate the same translation on every frame."""
        bc = self.bc
        p1, p2 = self._patch_backend()
        with p1, p2, \
                mock.patch.object(bc._camera_backend, "open_camera",
                                  return_value=_Cap()):
            bc._camera_backend_notes.clear()
            bc._camera_backend_pending.clear()
            for _ in range(25):
                bc._camera_open(0, name="usb 2.0 camera",
                                label="side tile index 0")
            queued = list(bc._camera_backend_pending)
        self.assertEqual(len(queued), 1, queued)

    def test_the_notes_key_is_bounded_by_call_sites_not_by_calls(self):
        """Adding the label to the key must not turn the memo into a leak.

        Every label a call site passes is either a configured camera's
        name/label or an f-string over the INDEX alone, so repeated opens from
        a fixed set of sites converge on a fixed set of keys."""
        bc = self.bc
        p1, p2 = self._patch_backend()
        labels = ("usb 2.0 camera", "side tile index 0", "probe index 0")
        with p1, p2, \
                mock.patch.object(bc._camera_backend, "open_camera",
                                  return_value=_Cap()):
            bc._camera_backend_notes.clear()
            bc._camera_backend_pending.clear()
            for _ in range(200):
                for lab in labels:
                    bc._camera_open(0, name="usb 2.0 camera", label=lab)
            n_keys = len(bc._camera_backend_notes)
            queued = list(bc._camera_backend_pending)
        self.assertEqual(n_keys, len(labels), bc._camera_backend_notes)
        self.assertEqual(len(queued), len(labels), queued)

    def test_the_open_worker_never_prints_the_decision_itself(self):
        bc = self.bc
        p1, p2 = self._patch_backend()
        printed = []
        with p1, p2, \
                mock.patch.object(bc._camera_backend, "open_camera",
                                  return_value=_Cap()), \
                mock.patch("builtins.print", printed.append):
            bc._camera_backend_notes.clear()
            bc._camera_backend_pending.clear()
            bc._camera_open(0, name="usb 2.0 camera")
        self.assertEqual(printed, [],
                         "the throwaway open worker narrated itself")

    def test_every_bounded_joiner_drains_the_queue(self):
        """A queue nobody drains is the same silence, one indirection later."""
        bc = self.bc
        for fn in (bc._open_capture_bounded, bc._probe_camera_index,
                   bc.list_cameras):
            self.assertIn("_drain_camera_backend_notes",
                          _executable_source(fn),
                          f"{fn.__name__} joins a camera open without ever "
                          f"delivering what the worker queued")

    def test_core_module_absent_degrades_to_the_old_behaviour(self):
        """If core/camera_backend.py could not be imported at all, a rig with
        no working face tracker is a worse outcome than a leaky one. Degrade to
        exactly what this code did before the module existed: a raw DirectShow
        open at the configured index, untranslated."""
        bc = self.bc
        cap = _Cap()
        with mock.patch.object(bc, "_camera_backend", None), \
                mock.patch.object(bc.cv2, "VideoCapture",
                                  return_value=cap) as vc:
            got = bc._camera_open(2, name="emeet c960", width=1280, height=720)
        self.assertIs(got, cap)
        self.assertEqual(vc.call_args.args, (2, bc.cv2.CAP_DSHOW))

    def test_media_foundation_absent_falls_back_without_translating(self):
        bc = self.bc
        p1, p2 = self._patch_backend(msmf_names=None)
        with p1, p2, \
                mock.patch.object(bc._camera_backend, "open_camera") as oc, \
                mock.patch.object(bc, "_dshow_input_devices_gated") as gated:
            oc.return_value = _Cap()
            bc._camera_backend_notes.clear()
            bc._camera_open(0, name="usb 2.0 camera")
        self.assertEqual(oc.call_args.args[0], 0)
        self.assertEqual(oc.call_args.kwargs["backend"], "dshow")
        # No point paying for the leaky list when the target backend is gone.
        gated.assert_not_called()


@requires_monolith
class CallSiteTests(MonolithGlobalsTestCase):
    """Each live open site reaches for _camera_open, not cv2 directly."""

    def test_tile_open_uses_the_shared_opener_and_proves_a_frame(self):
        bc = self.bc
        seen = {}

        def fake_open(idx, **kw):
            seen.update(kw)
            seen["idx"] = idx
            return _Cap()

        with mock.patch.object(bc, "_camera_open", fake_open), \
                mock.patch.object(bc, "_camera_is_quarantined",
                                  return_value=False), \
                mock.patch.object(bc, "_open_capture_bounded",
                                  side_effect=lambda idx, opener, **k: opener()):
            cap = bc._open_tile_capture(0, name="usb 2.0 camera")
        self.assertIsNotNone(cap)
        self.assertEqual(seen["idx"], 0)
        self.assertEqual(seen["name"], "usb 2.0 camera")
        self.assertEqual((seen["width"], seen["height"]), (640, 480))
        # A tile that holds a frameless handle fails every read forever and
        # feeds _invalidate_side_tile_indices — the amplifier that killed
        # v2.0.100. Proving a frame at open is what stops that.
        self.assertGreater(seen["require_frame"], 0)

    def test_tile_open_still_refuses_a_quarantined_index(self):
        bc = self.bc
        with mock.patch.object(bc, "_camera_is_quarantined",
                               return_value=True), \
                mock.patch.object(bc, "_camera_open") as oc:
            self.assertIsNone(bc._open_tile_capture(0))
        oc.assert_not_called()

    def test_probe_camera_index_uses_the_shared_opener(self):
        bc = self.bc
        calls = []

        def fake_open(idx, **kw):
            calls.append((idx, kw))
            return _Cap(frames=1)

        with mock.patch.object(bc, "_camera_open", fake_open):
            self.assertTrue(bc._probe_camera_index(2, timeout_sec=1.0))
        self.assertEqual(calls[0][0], 2)
        # The probe owns its own read-until-deadline budget; a second nested
        # frame budget would only make the timing harder to reason about.
        self.assertFalse(calls[0][1].get("require_frame"))

    def test_probe_camera_index_reports_a_dead_index_as_false(self):
        bc = self.bc
        with mock.patch.object(bc, "_camera_open", return_value=None):
            self.assertFalse(bc._probe_camera_index(9, timeout_sec=0.3))

    def test_probe_camera_index_is_false_when_no_frame_ever_arrives(self):
        # The MSMF busy-device shape: a handle that opens and delivers nothing
        # must not be reported as a working camera.
        bc = self.bc
        with mock.patch.object(bc, "_camera_open",
                               return_value=_Cap(frames=0)):
            self.assertFalse(bc._probe_camera_index(2, timeout_sec=0.4))

    def test_no_live_directshow_constant_remains_in_the_open_paths(self):
        """The four openers must contain no cv2.CAP_DSHOW of their own.

        Reading the source is the point: a future edit that reintroduces a raw
        `cv2.VideoCapture(idx, cv2.CAP_DSHOW)` at one of these sites would pass
        every behavioural test above (they mock _camera_open) while quietly
        restoring +4.6 OS threads per open."""
        bc = self.bc
        for fn in (bc._open_tile_capture, bc._probe_camera_index,
                   bc.list_cameras, bc._face_tracking_thread_body):
            code = _executable_source(fn)
            self.assertNotIn("CAP_DSHOW", code,
                             f"{fn.__name__} still names CAP_DSHOW in code")
            self.assertNotIn("cv2.VideoCapture(", code,
                             f"{fn.__name__} still opens cv2 directly")

    def test_face_track_producer_open_is_msmf_and_frame_proven(self):
        """The producer's opener is a closure inside
        _face_tracking_thread_BODY — _face_tracking_thread is only the
        supervisor around it — so reach it through the source rather than by
        starting the thread."""
        bc = self.bc
        code = _executable_source(bc._face_tracking_thread_body)
        self.assertNotIn("cv2.VideoCapture(", code)
        self.assertNotIn("CAP_DSHOW", code)
        self.assertIn("_camera_open(", code)
        self.assertIn("_LOOP_OPEN_FRAME_BUDGET_S", code)

    def test_frame_budgets_fit_inside_the_open_bound(self):
        """_CAMERA_OPEN_TIMEOUT_S bounds the WHOLE tile open — ctor, a possible
        retry, set(W), set(H), and only then the frame wait. If the frame
        budget eats the margin, the bound fires on a HEALTHY camera and the
        tile goes dark for a reason that has nothing to do with the device.

        Sized from a live measurement, not arithmetic: six consecutive
        _open_tile_capture() calls on the owner's USB 2.0 webcam at 640x480
        over MSMF took 2.005-2.271 s end to end, of which ~2.0 s was ctor +
        set(W) + set(H) before a frame was ever waited for. So the budget must
        leave at least 2.3 s of the bound unspent."""
        bc = self.bc
        self.assertLess(bc._TILE_OPEN_FRAME_BUDGET_S,
                        bc._CAMERA_OPEN_TIMEOUT_S - 2.3)
        self.assertLess(bc._LOOP_OPEN_FRAME_BUDGET_S,
                        bc._CAMERA_LOOP_OPEN_TIMEOUT_S)


if __name__ == "__main__":
    unittest.main()
