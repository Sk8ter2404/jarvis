"""The face detector's idle cadence (2026-10-28).

WHY. Measured live with nobody in the room, 3.5 of the process's 4.1 idle
cores were the Haar cascade passes in _detect_face (OpenCV's parallel
workers + the face-track thread), ~1 CPU-second per 1280x720 frame on every
frame. _face_detect_due backs the detector off to FACE_TRACK_IDLE_HZ once no
camera has seen a face for FACE_TRACK_IDLE_AFTER_S; a hit restores
every-frame detection.

CI SAFETY: the helpers are lifted out of the monolith's SOURCE by AST and
exec'd with stub config constants, so this runs on the light-deps runner
with no cv2 / numpy / sounddevice. The source-order test reads the loop
text only.
"""
from __future__ import annotations

import ast
import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

_MONOLITH = os.path.join(_PROJECT, "bobert_companion.py")
_HELPERS = ("_face_detect_due", "_face_detect_mark", "_face_detect_policy_line")


_CACHE: dict = {}


def _source() -> str:
    if "src" not in _CACHE:
        with open(_MONOLITH, encoding="utf-8") as fh:
            _CACHE["src"] = fh.read()
    return _CACHE["src"]


def _tree() -> ast.Module:
    if "tree" not in _CACHE:
        _CACHE["tree"] = ast.parse(_source())
    return _CACHE["tree"]


def _lift_helpers(after_s: float, hz: float) -> dict:
    """exec the three helpers from the monolith's source with stub config."""
    tree = _tree()
    wanted = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in _HELPERS]
    assert len(wanted) == len(_HELPERS), [n.name for n in wanted]
    mod = ast.Module(body=wanted, type_ignores=[])
    ns = {"FACE_TRACK_IDLE_AFTER_S": after_s, "FACE_TRACK_IDLE_HZ": hz,
          "_face_detect_idle": {"seen": None, "ran": {}}}
    exec(compile(mod, _MONOLITH, "exec"), ns)
    return ns


class DueTests(unittest.TestCase):
    def setUp(self):
        self.ns = _lift_helpers(15.0, 1.0)
        self.due = self.ns["_face_detect_due"]
        self.mark = self.ns["_face_detect_mark"]

    def test_every_frame_until_idle_after_elapses(self):
        st = {"seen": None, "ran": 0.0}
        for t in (100.0, 100.05, 100.1, 114.9):
            self.assertTrue(self.due(t, st), t)
            self.mark(t, False, st)
        self.assertEqual(st["seen"], 100.0)   # the clock started at the first call

    def test_idle_rate_once_nobody_seen(self):
        st = {"seen": 100.0, "ran": 114.95}
        self.assertFalse(self.due(115.0, st))    # idle, ran 50 ms ago
        self.assertTrue(self.due(115.95, st))    # 1/hz elapsed
        self.mark(115.95, False, st)
        self.assertFalse(self.due(116.0, st))
        self.assertFalse(self.due(116.9, st))
        self.assertTrue(self.due(116.95, st))

    def test_hit_restores_every_frame(self):
        st = {"seen": 100.0, "ran": 114.95}
        self.assertTrue(self.due(116.0, st))
        self.mark(116.0, True, st)
        self.assertEqual(st["seen"], 116.0)
        for t in (116.05, 116.1, 130.9):
            self.assertTrue(self.due(t, st), t)
        self.mark(130.9, False, st)
        self.assertFalse(self.due(131.0, st))

    def test_zero_disables_backoff(self):
        for after, hz in ((0.0, 1.0), (15.0, 0.0), (-1.0, 2.0)):
            st = {"seen": 0.0, "ran": 9999.0}
            self.assertTrue(self.due(9999.01, st, after, hz), (after, hz))

    def test_bad_config_means_run(self):
        st = {"seen": 0.0, "ran": 9999.0}
        self.assertTrue(self.due(9999.01, st, "nope", 1.0))

    def test_module_state_is_the_default(self):
        self.ns["_face_detect_idle"].update({"seen": 100.0, "ran": 114.95})
        self.assertFalse(self.due(115.0))
        self.mark(115.95, True)
        self.assertEqual(self.ns["_face_detect_idle"]["seen"], 115.95)

    def test_idle_rate_is_the_configured_hz(self):
        st = {"seen": 0.0, "ran": 1000.0}
        self.assertFalse(self.due(1000.4, st, 15.0, 2.0))
        self.assertTrue(self.due(1000.5, st, 15.0, 2.0))

    def test_each_camera_keeps_its_own_idle_slot(self):
        # Two cameras in one loop pass at the same now_loop: with a shared
        # "ran" the first camera's mark starved the second for ever.
        st = {"seen": 100.0, "ran": {}}
        now = 200.0
        for _ in range(5):
            for cam in (0, 2):
                self.assertTrue(self.due(now, st, key=cam), (now, cam))
                self.mark(now, False, st, key=cam)
            self.assertFalse(self.due(now + 0.05, st, key=0))
            self.assertFalse(self.due(now + 0.05, st, key=2))
            now += 1.0
        self.assertEqual(set(st["ran"]), {0, 2})

    def test_a_hit_on_any_camera_restores_every_camera(self):
        st = {"seen": 100.0, "ran": {0: 200.0, 2: 200.0}}
        self.assertFalse(self.due(200.1, st, key=0))
        self.mark(200.1, True, st, key=2)
        self.assertTrue(self.due(200.15, st, key=0))
        self.assertTrue(self.due(200.15, st, key=2))

    def test_bad_ran_value_means_run(self):
        st = {"seen": 0.0, "ran": {0: "x"}}
        self.assertTrue(self.due(9999.0, st, key=0))

    def test_policy_line(self):
        line = self.ns["_face_detect_policy_line"]()
        self.assertTrue(line.startswith("[cpu] face-detect cadence:"), line)
        self.assertIn("1 Hz per camera after 15 s", line)
        off = _lift_helpers(0.0, 1.0)["_face_detect_policy_line"]()
        self.assertIn("no idle back-off", off)


class HandBusyTests(unittest.TestCase):
    """2026-11-16 (the Dell, his "hand tracking ... it's spotty"): with his
    face in view the every-frame detector held 6.2 of 12 threads and starved
    the hand tracker. busy_hz caps the detector while a hand is in use."""

    def setUp(self):
        self.ns = _lift_helpers(15.0, 1.0)
        self.due = self.ns["_face_detect_due"]
        self.mark = self.ns["_face_detect_mark"]

    def test_busy_caps_the_rate_while_a_face_is_seen(self):
        st = {"seen": 100.0, "ran": {0: 100.0}}          # face just seen
        self.assertTrue(self.due(100.05, st, key=0))     # no cap: every frame
        self.assertFalse(self.due(100.05, st, key=0, busy_hz=2.0))
        self.assertFalse(self.due(100.45, st, key=0, busy_hz=2.0))
        self.assertTrue(self.due(100.5, st, key=0, busy_hz=2.0))

    def test_the_slower_cap_wins_when_both_apply(self):
        st = {"seen": 0.0, "ran": {0: 1000.0}}           # idle (1 Hz)
        self.assertFalse(self.due(1000.6, st, key=0, busy_hz=2.0))
        self.assertTrue(self.due(1001.0, st, key=0, busy_hz=2.0))
        st = {"seen": 0.0, "ran": {0: 1000.0}}
        self.assertFalse(self.due(1001.5, st, key=0, busy_hz=0.5))
        self.assertTrue(self.due(1002.0, st, key=0, busy_hz=0.5))

    def test_busy_still_caps_with_the_idle_backoff_off(self):
        st = {"seen": 0.0, "ran": {0: 50.0}}
        self.assertTrue(self.due(50.1, st, 0.0, 1.0, key=0))
        self.assertFalse(self.due(50.1, st, 0.0, 1.0, key=0, busy_hz=1.0))
        self.assertTrue(self.due(51.0, st, 0.0, 1.0, key=0, busy_hz=1.0))

    def test_zero_or_bad_busy_means_no_cap(self):
        st = {"seen": 100.0, "ran": {0: 100.0}}
        for busy in (0.0, None, -3.0):
            self.assertTrue(self.due(100.01, st, key=0, busy_hz=busy), busy)
        self.assertTrue(self.due(100.01, st, key=0, busy_hz="x"))

    def _busy_fn(self, hz, window, module):
        tree = _tree()
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_hand_busy_face_hz"]
        self.assertEqual(len(fn), 1)
        fake_sys = type("S", (), {"modules": {} if module is None
                                  else {"skill_hand_tracking": module}})
        ns = {"HAND_TRACK_FACE_DETECT_HZ": hz, "HAND_TRACK_FACE_BUSY_S": window,
              "sys": fake_sys}
        exec(compile(ast.Module(body=fn, type_ignores=[]), _MONOLITH, "exec"), ns)
        return ns["_hand_busy_face_hz"]

    def test_busy_hz_follows_the_skill(self):
        seen = {"v": True, "window": None}

        def hands_seen_within(window):
            seen["window"] = window
            return seen["v"]
        mod = type("M", (), {"hands_seen_within": staticmethod(hands_seen_within)})
        f = self._busy_fn(1.0, 3.0, mod)
        self.assertEqual(f(), 1.0)
        self.assertEqual(seen["window"], 3.0)
        seen["v"] = False
        self.assertEqual(f(), 0.0)

    def test_busy_hz_is_zero_without_the_skill_or_when_off(self):
        self.assertEqual(self._busy_fn(1.0, 3.0, None)(), 0.0)
        mod = type("M", (), {"hands_seen_within": staticmethod(lambda w: True)})
        self.assertEqual(self._busy_fn(0.0, 3.0, mod)(), 0.0)
        self.assertEqual(self._busy_fn(1.0, 0.0, mod)(), 0.0)
        self.assertEqual(self._busy_fn("x", 3.0, mod)(), 0.0)

        def boom(w):
            raise RuntimeError("no")
        bad = type("M", (), {"hands_seen_within": staticmethod(boom)})
        self.assertEqual(self._busy_fn(1.0, 3.0, bad)(), 0.0)

    def test_config_defaults(self):
        from core import config
        self.assertGreater(config.HAND_TRACK_FACE_DETECT_HZ, 0)
        self.assertLessEqual(config.HAND_TRACK_FACE_DETECT_HZ, 5.0)
        self.assertGreater(config.HAND_TRACK_FACE_BUSY_S, 0)
        from core import peer_settings
        self.assertFalse(peer_settings.is_shared("HAND_TRACK_FACE_DETECT_HZ"))
        self.assertFalse(peer_settings.is_shared("HAND_TRACK_FACE_BUSY_S"))


class SourceOrderTests(unittest.TestCase):
    """The gate sits in the tracking loop right before the detector, the
    detector's run is marked, and the policy prints where the thread starts."""

    def test_gate_precedes_detect_in_the_loop(self):
        src = _source()
        gate = src.index('if not _face_detect_due(now_loop, key=cam["index"],\n'
                         '                                        busy_hz=_hand_busy_face_hz()):')
        detect = src.index("face = _detect_face(frame)", gate)
        self.assertLess(gate, detect)
        self.assertLess(detect - gate, 1200, "gate and detect drifted apart")
        between = src[gate:detect]
        self.assertIn("continue", between)
        mark = src.index('_face_detect_mark(now_loop, bool(face), key=cam["index"])', detect)
        self.assertLess(mark - detect, 200)

    def test_only_one_detect_call_site_is_gated(self):
        src = _source()
        calls = [m.start() for m in re.finditer(r"= _detect_face\(frame\)", src)]
        self.assertEqual(len(calls), 1, "a second _detect_face call site needs the gate too")

    def test_boot_line_in_thread_body(self):
        src = _source()
        body = src.index("def _face_tracking_thread_body():")
        line = src.index("_face_detect_policy_line()", body)
        self.assertLess(line - body, 2500)

    def test_config_defaults_exist(self):
        from core import config
        self.assertGreater(config.FACE_TRACK_IDLE_AFTER_S, 0)
        self.assertGreater(config.FACE_TRACK_IDLE_HZ, 0)
        self.assertLessEqual(config.FACE_TRACK_IDLE_HZ, 5.0)


if __name__ == "__main__":
    unittest.main()
