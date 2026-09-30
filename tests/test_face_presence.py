"""core/face_presence — sustained, qualified, fresh-frame face presence.

Live evidence (session_2026-09-30_09-48-41.log): proactive remarks at 09:56:22
and 09:59:54 with the owner away from home. should_be_proactive's camera gate
("a face within 60 s") was open because the face-track loop stamped
last_face_seen on ANY single-frame Haar hit — relaxed and profile passes
included, a 40 px minimum on a 1280x720 frame, and no check that the frame was
new (the Kinect shim re-serves the same buffer when its sensor stalls).

Light tier: numpy only for synthetic frames (skipped without it). No camera.

    python -m unittest tests.test_face_presence
"""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import face_presence as fp  # noqa: E402

try:
    import numpy as np
except Exception:   # pragma: no cover - light runner without numpy
    np = None

STRICT_BIG = {"pass": "frontal", "w_frac": 0.20, "h_frac": 0.35}


def _frame(value):
    return np.full((720, 1280, 3), value % 256, dtype=np.uint8)


class QualifiesTests(unittest.TestCase):
    def test_only_a_big_confident_pass_hit_qualifies(self):
        self.assertTrue(fp.qualifies(STRICT_BIG))
        # the minNeighbors=4 profile passes count (a turned head at the desk)
        self.assertTrue(fp.qualifies({"pass": "profile", "w_frac": 0.2}))
        self.assertTrue(fp.qualifies({"pass": "profile_mirror", "w_frac": 0.2}))
        for info in ({"pass": "frontal_relaxed", "w_frac": 0.2},
                     # 40 px on 1280: the old floor, a face across the room
                     {"pass": "frontal", "w_frac": 40 / 1280},
                     {"pass": "profile", "w_frac": 40 / 1280},
                     None, {}, "frontal"):
            with self.subTest(info=info):
                self.assertFalse(fp.qualifies(info))


class SustainedFaceTests(unittest.TestCase):
    def _feed(self, tracker, times, *, fresh=True, qualified=True):
        return [tracker.observe(t, fresh=fresh, qualified=qualified)
                for t in times]

    def test_a_single_frame_hit_never_confirms(self):
        # The old gate: one hit = "the owner is at his desk" for 60 s.
        t = fp.SustainedFace()
        self.assertEqual(self._feed(t, [100.0]), [False])

    def test_a_re_served_frame_never_confirms(self):
        t = fp.SustainedFace()
        self.assertEqual(self._feed(t, [100.0]), [False])
        # The same buffer served again and again: ignored, never a hit.
        self.assertFalse(any(self._feed(
            t, [100.0 + i * 0.5 for i in range(1, 60)], fresh=False)))
        # ...and within one window it adds neither a hit nor a frame.
        t2 = fp.SustainedFace()
        t2.observe(100.0, fresh=True, qualified=True)
        self._feed(t2, [100.0 + i * 0.5 for i in range(1, 10)], fresh=False)
        self.assertEqual(t2.counts(), (1, 1))

    def test_intermittent_hits_do_not_confirm(self):
        # A face-like blob caught on 1 frame in 4 (a TV across the room).
        t = fp.SustainedFace()
        out = [t.observe(100.0 + i * 0.5, fresh=True, qualified=(i % 4 == 0))
               for i in range(40)]
        self.assertFalse(any(out))

    def test_a_burst_shorter_than_the_span_does_not_confirm(self):
        t = fp.SustainedFace()
        self.assertFalse(any(self._feed(t, [100.0 + i * 0.1
                                            for i in range(10)])))

    def test_a_sustained_face_confirms(self):
        t = fp.SustainedFace()
        out = self._feed(t, [100.0 + i * 0.5 for i in range(8)])
        self.assertTrue(out[-1])
        self.assertFalse(any(out[:4]), "confirmed before MIN_HITS hits")

    def test_it_expires_with_the_window(self):
        t = fp.SustainedFace()
        self._feed(t, [100.0 + i * 0.5 for i in range(8)])
        # long gap: the window forgets, one new hit is not enough again
        self.assertFalse(t.observe(200.0, fresh=True, qualified=True))


@unittest.skipIf(np is None, "numpy not installed")
class FingerprintTests(unittest.TestCase):
    def test_identical_content_same_print_new_content_different(self):
        a = _frame(10)
        self.assertEqual(fp.frame_fingerprint(a), fp.frame_fingerprint(a.copy()))
        b = a.copy()
        b[360, 640] = 200            # one sampled pixel changed
        self.assertNotEqual(fp.frame_fingerprint(a), fp.frame_fingerprint(b))

    def test_unprintable_frames_give_none(self):
        self.assertIsNone(fp.frame_fingerprint(None))
        self.assertIsNone(fp.frame_fingerprint(object()))


if __name__ == "__main__":
    unittest.main()
