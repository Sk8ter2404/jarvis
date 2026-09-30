"""Post-cool-down probation + culprit quarantine, wired into the monolith
(v2.0.132).

THE LIVE SEQUENCE THIS FILE REPLAYS (2026-09-29, real names redacted). The
first USB-storm trip was right. Its cool-down ended; the face tracker opened
the left webcam; the hub reset 0.8 s later; the right webcam's open found its
device gone from the list; the left webcam went dead after 60 failed reads -
and NO second trip happened. The left webcam was reopened 21 s later and the
hub reset 0.04 s after that. A read-only USB test the same day showed the left
webcam's stream starts reset the hub 8 of 10 times, the right one's 0 of 10.

The producer tests drive the REAL producer loop (_face_tracking_thread_body)
with fake captures, a fake device list and a frozen clock, exactly like
tests/monolith/test_monolith_camera_storm.py. Nothing opens a camera. Device
names and labels are synthetic.
"""
from __future__ import annotations

import ast
import json
import os
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_camera_storm import (
    _CAM_A, _CAM_B, _Cap, _FakeBackend, _StormBase)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Hub:
    """One shared USB hub. Every stream START of the left camera after its
    first resets it 0.8 s later (the measured 8-of-10 culprit); while it is
    re-enumerating nothing is on the device list and nothing can open."""

    def __init__(self, clock, first_reset_at: float):
        self.clock = clock
        self.down: list = [(first_reset_at, first_reset_at + 3.0)]
        self.opens: list = []

    def is_down(self, t: float) -> bool:
        return any(s <= t < e for s, e in self.down)

    def names(self):
        if self.is_down(self.clock.t):
            return ()
        return ("SynthCam One", "SynthCam Two")

    def open(self, idx):
        t = self.clock.t
        self.opens.append((t, idx))
        if self.is_down(t):
            return None
        if idx == _CAM_A["index"] and sum(
                1 for _t, i in self.opens if i == idx) > 1:
            self.down.append((t + 0.8, t + 3.8))

        def _alive(now, _since=t):
            return not any(_since <= s <= now for s, _e in self.down)
        return _Cap(self.clock, alive=_alive)


@requires_monolith
class LiveSequenceThroughTheProducerTests(_StormBase):

    def _run(self):
        t0 = self.clock.t
        self.hub = _Hub(self.clock, t0 + 20.0)
        backend = _FakeBackend(self.clock, names=self.hub.names)
        out = self._producer([_CAM_A, _CAM_B], iterations=2600,
                             opener=self.hub.open, backend=backend,
                             step=self._held_step)
        trips = [ln for ln in out.splitlines()
                 if "[usb-storm]" in ln and "treating it as a USB bus event"
                 in ln]
        return t0, out, trips

    def _post_cool_down_opens_of_a(self, trips_t):
        return [t for t, i in self.hub.opens
                if i == _CAM_A["index"] and t > trips_t]

    def test_the_first_post_cool_down_reset_re_trips_the_breaker(self):
        t0, out, trips = self._run()
        self.assertGreaterEqual(len(trips), 2, out[-4000:])
        first_trip_t = self.spoken[0][0]
        a_reopen = min(t for t, i in self.hub.opens
                       if i == _CAM_A["index"] and t >= first_trip_t + 600.0)
        # The left webcam's reopen reset the hub; within seconds the breaker
        # must be open again - the old gate let the cameras keep reopening.
        second = [ln for ln in out.splitlines()
                  if "[usb-storm]" in ln and "trip #2" in ln
                  and "treating it" in ln]
        self.assertEqual(len(second), 1, out[-4000:])
        self.assertIn("doubled", second[0])
        self.assertIn("probation", second[0])
        reopens_soon = [t for t, i in self.hub.opens
                        if i == _CAM_A["index"] and a_reopen < t < a_reopen + 600.0]
        self.assertEqual(reopens_soon, [],
                         "the left webcam was reopened into the resetting hub "
                         "(18:49:25.55 on 2026-09-29)")

    def test_the_culprit_is_quarantined_and_its_sibling_keeps_working(self):
        t0, out, trips = self._run()
        q_lines = [ln for ln in out.splitlines() if "[camera-quarantine]" in ln]
        self.assertEqual(len(q_lines), 1, out[-4000:])
        self.assertIn("name:synthcam one", q_lines[0])
        said = [m for _t, m in self.spoken]
        self.assertIn(
            "Sir, the left webcam keeps knocking the USB hub offline whenever "
            "it starts, so I've stopped using it until it's moved to another "
            "port.", said)
        self.assertEqual(len(said), 2, said)       # the storm, the quarantine
        q_at = next(t for t, m in self.spoken if "keeps knocking" in m)
        self.assertEqual([t for t, i in self.hub.opens
                          if i == _CAM_A["index"] and t > q_at], [],
                         "a quarantined camera was opened again")
        b_after = [t for t, i in self.hub.opens
                   if i == _CAM_B["index"] and t > q_at]
        self.assertTrue(b_after, "the right webcam never came back")
        last_b = self.bc._camera_last_frame_at.get(_CAM_B["index"], 0.0)
        self.assertGreater(last_b, max(b_after),
                           "the right webcam came back but never streamed")
        # The producer's "dead after 60 failed reads" line must not promise a
        # reopen the gate will never allow (it used to print the gate's 600 s
        # re-ask interval as "will reopen in 600.0s").
        lines = out.splitlines()
        q_idx = next(i for i, ln in enumerate(lines)
                     if "[camera-quarantine]" in ln)
        dead_a = [ln for ln in lines[q_idx:]
                  if "Synth left" in ln and "dead after" in ln]
        self.assertTrue(dead_a, out[-4000:])
        for ln in dead_a:
            self.assertNotIn("will reopen", ln)
            self.assertIn("quarantined", ln)
        status = self.gate.snapshot()
        self.assertEqual(list(status["quarantined"]), ["name:synthcam one"])
        self.assertEqual(status["quarantined"]["name:synthcam one"]["label"],
                         "the left webcam")


@requires_monolith
class DeadBurstIsReportedPerStreamTests(_StormBase):
    """The producer used to report a read-failure drop once per 'episode
    since the last healthy frame', so a camera reopened into a resetting hub
    that never delivered a frame never reported again."""

    def test_a_reopened_stream_that_dies_again_is_reported_again(self):
        opens: list = []

        def _open(idx):
            opens.append(self.clock.t)
            return _Cap(self.clock, alive=lambda now: False)   # never a frame
        drops: list = []
        real = self.gate.note_drop

        def _spy(key, component="", kind="camera", now=None, **kw):
            drops.append((self.clock.t, key, kw.get("cause", "")))
            return real(key, component, kind, now, **kw)
        # Every open "succeeds" (the fake backend is bypassed), every stream
        # dies at once: each NEW stream's burst must reach the gate.
        with mock.patch.object(self.gate, "note_drop", side_effect=_spy):
            self._producer([_CAM_A], iterations=400, opener=_open,
                           step=self._held_step)
        self.assertGreaterEqual(len(opens), 3, opens)
        cam_drops = [d for d in drops if d[1] == "name:synthcam one"]
        self.assertGreaterEqual(
            len(cam_drops), len(opens) - 1,
            f"{len(opens)} streams died but only {len(cam_drops)} drops "
            f"reached the camera gate")
        self.assertTrue(all(d[2] == "read-failure burst" for d in cam_drops))


@requires_monolith
class QuarantineWiringTests(_StormBase):

    def test_the_friendly_label_comes_from_cameras(self):
        bc = self.bc
        cams = [dict(_CAM_A), dict(_CAM_B), {"index": 7, "type": "kinect"}]
        with mock.patch.object(bc, "CAMERAS", cams):
            self.assertEqual(
                bc._camera_gate_friendly_label("name:synthcam one"),
                "the left webcam")
            self.assertEqual(
                bc._camera_gate_friendly_label("name:synthcam two"),
                "the right webcam")
            self.assertEqual(bc._camera_gate_friendly_label("kinect"),
                             "the Kinect")
            self.assertEqual(bc._camera_gate_friendly_label("dshow:9"), "")

    def _quarantine_left(self):
        g = self.gate
        g.begin("name:synthcam one", "face-track")
        g.end("name:synthcam one", "face-track", True)
        self.clock.advance(0.5)
        g.note_drop("name:synthcam one", "face-track")
        g.note_drop("name:synthcam two", "face-track")

    def test_lift_by_voice_puts_the_camera_back(self):
        bc = self.bc
        g = bc._make_camera_gate(clock=self.clock.time)
        g.culprit_threshold = 1
        self.gate = g
        cams = [dict(_CAM_A), dict(_CAM_B)]
        with mock.patch.object(bc, "_camera_gate", g, create=True), \
             mock.patch.object(bc, "CAMERAS", cams), \
             mock.patch.object(bc, "_camera_backend",
                               _FakeBackend(self.clock)), \
             mock.patch.object(bc, "proactive_announce",
                               side_effect=lambda m, *a, **k:
                               self.spoken.append(m) or True), \
             mock.patch("builtins.print"):
            self._quarantine_left()
            self.assertTrue(g.quarantined("name:synthcam one"))
            self.assertIn("the left webcam keeps knocking", self.spoken[-1])
            from skills import camera_system
            with mock.patch.object(camera_system, "_bc", return_value=bc):
                status = camera_system.camera_status("")
                self.assertIn("the left webcam is switched off", status)
                self.assertEqual(camera_system.camera_unquarantine("right"),
                                 "The left webcam is the only camera switched "
                                 "off, sir - say 'use the left webcam again' "
                                 "to put it back.")
                self.assertTrue(g.quarantined("name:synthcam one"))
                said = camera_system.camera_unquarantine("left")
                self.assertIn("the left webcam is back in use", said)
                self.assertFalse(g.quarantined("name:synthcam one"))
                self.assertEqual(camera_system.camera_unquarantine(""),
                                 "No camera is switched off at the moment, "
                                 "sir - there is nothing to put back.")
        self.assertIn("camera_unquarantine", bc.SPEAK_RESULT_VERBATIM_ACTIONS)

    def test_status_shows_the_quarantine(self):
        bc = self.bc
        g = bc._make_camera_gate(clock=self.clock.time)
        g.culprit_threshold = 1
        with mock.patch.object(bc, "_camera_gate", g, create=True), \
             mock.patch.object(bc, "CAMERAS", [dict(_CAM_A), dict(_CAM_B)]), \
             mock.patch.object(bc, "_camera_backend",
                               _FakeBackend(self.clock)), \
             mock.patch.object(bc, "proactive_announce", return_value=True), \
             mock.patch("builtins.print"):
            self.gate = g
            self._quarantine_left()
            st = bc.get_camera_gate_status()
        self.assertEqual(st["quarantined"]["name:synthcam one"]["label"],
                         "the left webcam")
        self.assertTrue(st["devices"]["name:synthcam one"]["quarantined"])


@requires_monolith
class SideTileDropIsMinorTests(unittest.TestCase):
    """Structural: the side tile reports a drop on ONE failed read. On
    post-cool-down probation that must not re-trip the breaker by itself, so
    its report is flagged minor=True (the gate still counts it toward the
    two-camera rule). Walks the AST, not the text."""

    def test_the_side_tile_drop_is_reported_as_minor(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        found = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_camera_gate_note_drop"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "side-tile"):
                kw = {k.arg: k.value for k in node.keywords}
                found.append(isinstance(kw.get("minor"), ast.Constant)
                             and kw["minor"].value is True)
        self.assertTrue(found, "no side-tile drop report found")
        self.assertTrue(all(found), "a side-tile drop is not flagged minor")


@requires_monolith
class ProbationKnobTests(_StormBase):
    """CAMERA_STORM_PROBATION_S, CAMERA_CULPRIT_WINDOW_S,
    CAMERA_CULPRIT_THRESHOLD: shipped in core/config.py, mirrored in the
    Settings schema and the template, and READ by the gate the monolith
    builds."""

    _KNOBS = {"CAMERA_STORM_PROBATION_S": (180.0, float, "float"),
              "CAMERA_CULPRIT_WINDOW_S": (5.0, float, "float"),
              "CAMERA_CULPRIT_THRESHOLD": (2, int, "int")}

    def test_config_ships_them_with_the_right_types(self):
        with open(os.path.join(_ROOT, "core", "config.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        lits = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value,
                                                           ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        lits[tgt.id] = node.value.value
        for k, (v, typ, _t) in self._KNOBS.items():
            self.assertIs(type(lits.get(k)), typ, k)
            self.assertEqual(lits[k], v, k)

    def test_settings_window_and_template_mirror_them(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "jarvis_settings_window_quarantine",
            os.path.join(_ROOT, "tools", "settings_window.py"))
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        with open(os.path.join(_ROOT, "tools", "user_settings.example.json"),
                  encoding="utf-8") as fh:
            example = json.load(fh)
        for k, (v, typ, t) in self._KNOBS.items():
            self.assertEqual(sw.SCHEMA[k]["type"], t, k)
            self.assertEqual(sw.SCHEMA[k]["default"], v, k)
            self.assertEqual(example[k], v, k)
            self.assertIs(type(example[k]), typ, k)

    def test_the_gate_the_monolith_builds_reads_them(self):
        bc = self.bc
        with mock.patch.object(bc, "CAMERA_STORM_PROBATION_S", 240.0), \
             mock.patch.object(bc, "CAMERA_CULPRIT_WINDOW_S", 7.5), \
             mock.patch.object(bc, "CAMERA_CULPRIT_THRESHOLD", 3):
            g = bc._make_camera_gate()
        self.assertEqual(g.probation_s, 240.0)
        self.assertEqual(g.culprit_window_s, 7.5)
        self.assertEqual(g.culprit_threshold, 3)
        self.assertEqual(bc._camera_gate.probation_s,
                         bc.CAMERA_STORM_PROBATION_S)
        self.assertEqual(bc._camera_gate.culprit_threshold,
                         bc.CAMERA_CULPRIT_THRESHOLD)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
