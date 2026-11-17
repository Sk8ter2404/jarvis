"""hud/holo/gestures.js grab stability (2026-11-17) - static pins for CI.

Live 11-17 16:03-16:11 (desktop, holo on the Kinect, dusk): 38 fist_grab with
no release between, 9 fist_release on the other side than the grab, pulled
parts let go after 0.15 s (about one frame at the Kinect's then ~8 fps). The
page now keeps a grab / pull through a short loss, re-finds it by palm
position (not the Left/Right label) and ignores another hand's far release.

The behaviour itself is covered by hud/holo/selftest.js (run in a browser:
/holo?selftest=1); CI has no JS engine, so this file pins the numbers, the
wiring and that the self tests exist. No browser, no server, no camera.
"""
import os
import re
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools import handtrack_gestures as hg  # noqa: E402

_HOLO = os.path.join(_ROOT, "hud", "holo")


def _read(name):
    with open(os.path.join(_HOLO, name), encoding="utf-8") as f:
        return f.read()


def _const(js, name):
    m = re.search(rf"\b{name}:\s*([0-9.]+)", js)
    return None if m is None else float(m.group(1))


class GrabStabilityPins(unittest.TestCase):
    def setUp(self):
        self.js = _read("gestures.js")
        self.code = re.sub(r"//[^\n]*", "", self.js)

    def test_the_timings(self):
        self.assertEqual(_const(self.js, "PINCH_RELEASE_S"), 0.35)
        self.assertEqual(_const(self.js, "HAND_LOST_S"), 0.4)
        self.assertEqual(_const(self.js, "REASSOC_MAX"), 0.2)

    def test_the_tracker_never_ends_a_grab_before_the_page_would(self):
        # the engine sends its own fist_release when it drops a lost track
        self.assertGreaterEqual(hg.TRACK_DROP_S, _const(self.js, "HAND_LOST_S"))

    def test_grab_and_pull_find_their_hand_by_position(self):
        self.assertIn('this._holderHand(g, "fist")', self.code)
        self.assertIn('this._holderHand(pl, "pinch")', self.code)
        self.assertNotIn("this.hands.get(this.grab.side)", self.code)
        self.assertNotIn("this.hands.get(pl.side)", self.code)
        # a lost hand runs the lost timer, a present-but-open one the release timer
        self.assertEqual(self.code.count("(h ? T.PINCH_RELEASE_S : T.HAND_LOST_S)"), 2)

    def test_a_far_release_by_another_fist_is_ignored(self):
        i = self.code.index('case "fist_release"')
        body = self.code[i:i + 400]
        self.assertIn("T.REASSOC_MAX", body)
        self.assertIn("this._endGrab()", body)

    def test_still_pure(self):
        for banned in ("document.", "window.", "fetch(", "setTimeout", "WebSocket",
                       "THREE", "import "):
            self.assertNotIn(banned, self.code, banned)

    def test_the_self_tests_cover_it(self):
        st = _read("selftest.js")
        for name in ("a hand missing for a moment keeps the grab",
                     "a Left/Right label flip in place keeps holding",
                     "the nearest fist holds it, not the one with the old label",
                     "an open hand still lets go after the release time",
                     "a pulled part survives a label flip and a short loss",
                     "two pinching hands scale even when both carry the same label"):
            self.assertIn(name, st, name)


if __name__ == "__main__":
    unittest.main()
