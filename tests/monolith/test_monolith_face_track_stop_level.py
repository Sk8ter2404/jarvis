"""The face-track producer's normal shutdown is not an ERROR (NEW #19, 2026-10-02).

THE LIVE EVIDENCE (session_2026-10-01_17-33-02.log and _19-43-10.log): every
clean tray restart wrote

    ERROR [face-track] STOPPED - ... Reason: stop event set (normal shutdown).

`_face_track_note_stopped` called `logging.error` whenever `exc is None`, so a
deliberate stop (shutdown, or the "camera off" action, which sets the same
stop event) landed in the error stream beside real faults and polluted every
error/crash triage pass.

  * stop event set, no exception     -> INFO (still printed, still logged).
  * loop RETURNED without the event  -> ERROR (a defect, unchanged).
  * an exception escaped             -> ERROR with the traceback (unchanged),
                                        even if the stop event was also set.

The tests drive the REAL supervisor (`_face_tracking_thread`) with only the
producer body, the camera release and the watchdog faked.

    python -m unittest tests.monolith.test_monolith_face_track_stop_level
"""
from __future__ import annotations

import contextlib
import io
import logging
import unittest
from unittest import mock

from tests._monolith_harness import (MonolithGlobalsTestCase, load_monolith,
                                     requires_monolith)


@requires_monolith
class FaceTrackStopLevelTests(MonolithGlobalsTestCase):

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self.bc._face_track_stop.clear()

    def tearDown(self):
        self.bc._face_track_stop.clear()

    def _run_supervisor(self, body):
        """Run the real supervisor around `body`; return every log record at
        DEBUG and above that it emitted."""
        bc = self.bc
        logger = logging.getLogger()
        with mock.patch.object(bc, "_face_tracking_thread_body", body), \
             mock.patch.object(bc, "_face_track_release_all"), \
             mock.patch.object(bc, "_face_track_watchdog"), \
             mock.patch.object(bc, "_hud_camera_preview_remove"), \
             mock.patch.object(bc, "_hud_percam_preview_remove_all"), \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertLogs(logger, level=logging.DEBUG) as cm:
            # assertLogs needs at least one record; this marker guarantees it
            # without depending on the code under test.
            logger.debug("test-marker")
            try:
                bc._face_tracking_thread()
            except BaseException:
                pass
        return [r for r in cm.records if "face-track] STOPPED" in r.getMessage()]

    def test_normal_shutdown_is_info_not_error(self):
        def _body():
            self.bc._face_track_stop.set()      # the shutdown path's signal
        recs = self._run_supervisor(_body)
        self.assertEqual(len(recs), 1, "the STOPPED line must still be logged")
        self.assertIn("normal shutdown", recs[0].getMessage())
        self.assertEqual(
            recs[0].levelno, logging.INFO,
            "a deliberate stop (stop event set, no exception) was logged at "
            f"{recs[0].levelname} - it pollutes error/crash triage")

    def test_return_without_the_stop_event_is_still_an_error(self):
        recs = self._run_supervisor(lambda: None)
        self.assertEqual(len(recs), 1)
        self.assertIn("RETURNED without the stop event", recs[0].getMessage())
        self.assertEqual(recs[0].levelno, logging.ERROR,
                         "an unexpected stop must stay an ERROR")

    def test_an_exception_is_still_an_error_even_after_stop(self):
        def _body():
            self.bc._face_track_stop.set()
            raise RuntimeError("synthetic producer failure")
        recs = self._run_supervisor(_body)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].levelno, logging.ERROR)
        self.assertIsNotNone(recs[0].exc_info,
                             "the traceback must ride the ERROR record")

    def test_helper_never_raises_and_prints_both_ways(self):
        bc = self.bc
        for stop in (True, False):
            if stop:
                bc._face_track_stop.set()
            else:
                bc._face_track_stop.clear()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertLogs(level=logging.INFO):
                bc._face_track_note_stopped("synthetic reason")
            self.assertIn("[face-track] STOPPED", buf.getvalue())


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
