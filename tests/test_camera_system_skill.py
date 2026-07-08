"""Tests for skills/camera_system — the UNIFIED multi-camera awareness layer.

Loads the skill in isolation (no monolith boot) via the shared skill harness,
then patches its three seams — _bc() (the monolith's shared frame caches),
_kinect_bridge() (the Kinect), and the config flag reader — with fakes so NO
hardware and NO real vision call is touched. Covers:

  * camera_status   — 2 webcams live + Kinect present; a stale/missing webcam
                      reported dark; Kinect absent → only webcams; nothing.
  * situational_awareness / where_am_i — fuses gaze + Kinect presence; the
                      webcam-only, Kinect-only, and none paths; names sources.
  * look_around     — captures from every available source, calls ask_vision
                      per frame (mocked), synthesizes; single- and zero-camera
                      paths; never raises when a frame is missing.

A guard test asserts the skill mutates no monolith/global state.

stdlib unittest + mock only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated, make_fake_skill_utils


# ─── fakes ────────────────────────────────────────────────────────────────

class _FakeFrame:
    """A stand-in for a cv2/numpy BGR frame: only needs .copy() (which the
    skill calls under the state lock). Carries a tag so an encoder mock can
    produce distinguishable PNG bytes."""
    def __init__(self, tag: str):
        self.tag = tag

    def copy(self):
        return self


def _fake_monolith(*, frame_ages=None, faces=None, frames=None, errors=None):
    """Build a stand-in bobert_companion module exposing the shared camera
    caches the skill reads.

      frame_ages: {index: seconds_ago}   → _camera_last_frame_at
      faces:      {index: seconds_ago}   → _camera_last_seen
      frames:     {index: _FakeFrame}    → _camera_latest_frame
      errors:     {index: str}           → _camera_last_read_error
    """
    now = time.time()
    bc = types.ModuleType("bobert_companion")
    bc._camera_state_lock = threading.Lock()
    bc._camera_last_frame_at = {i: now - ago for i, ago in (frame_ages or {}).items()}
    bc._camera_last_seen = {i: now - ago for i, ago in (faces or {}).items()}
    bc._camera_latest_frame = dict(frames or {})
    bc._camera_last_read_error = dict(errors or {})
    return bc


def _fake_bridge(*, enabled=True, available=(True, ""), presence=None,
                 color_png=b"\x89PNG-kinect"):
    """A stand-in audio.kinect_bridge module (only the accessors the skill uses)."""
    m = types.ModuleType("audio.kinect_bridge")
    m.get_enabled = lambda: enabled
    m.available = lambda: available
    m.get_presence = lambda: (presence if presence is not None
                              else {"present": False, "count": 0,
                                    "nearest_m": None, "facing": None, "ts": 0.0})
    m.get_color_png = lambda: color_png
    return m


_CAMERAS_2 = [
    {"index": 1, "label": "Left webcam", "primary": False, "look_x": 0.15, "look_y": 0.5},
    {"index": 0, "label": "Right webcam", "primary": True, "look_x": 0.85, "look_y": 0.5},
]


class CameraSystemBase(unittest.TestCase):
    """Loads the skill fresh per test and patches its seams to the supplied
    fakes. Returns (module, actions)."""

    def _load(self, *, bc=None, bridge=None, cameras=None, kinect_enabled=True,
              ai_backend="claude", vision_route="auto", utils=None,
              gaze_snapshot=None):
        cameras = _CAMERAS_2 if cameras is None else cameras
        mod, actions = load_skill_isolated("camera_system", utils=utils,
                                           register=True)

        # Seam 1: the monolith (shared frame caches). Patch _bc directly so the
        # harness's __main__ shadow can't leak in.
        mod._bc = lambda: bc

        # Seam 2: the Kinect bridge.
        mod._kinect_bridge = lambda: bridge

        # Seam 3: config. The skill reads config three ways — `from core.config
        # import CAMERAS`, `from core.config import model_route`, and
        # `from core import config; getattr(config, FLAG)`. Swapping the
        # sys.modules entry is NOT reliable (a `from core import config` binds the
        # package attribute, and the CI-sim runner resolves it to the real module)
        # — so we patch the skill's OWN seams directly, which is import-resolution
        # independent. _cfg_flag / _cfg_flag_cloud_backend / CAMERAS / model_route
        # all funnel through these.
        flags = {"KINECT_ENABLED": kinect_enabled}
        mod._cfg_flag = lambda name, default=False: bool(flags.get(name, default))
        mod._cfg_flag_cloud_backend = lambda: (ai_backend == "claude")

        # CAMERAS + model_route still come from core.config via a local import;
        # patch those attributes on the REAL module (it's stdlib-only, present on
        # CI), restoring them after the test.
        import core.config as _real_cfg
        _p_cams = mock.patch.object(_real_cfg, "CAMERAS", cameras)
        _p_route = mock.patch.object(_real_cfg, "model_route",
                                     lambda fn: vision_route)
        _p_cams.start()
        _p_route.start()
        self.addCleanup(_p_cams.stop)
        self.addCleanup(_p_route.stop)

        # Optional: pin the gaze reading (otherwise it derives from timestamps).
        if gaze_snapshot is not None:
            mod._gaze_snapshot = lambda: gaze_snapshot

        # cv2 isn't needed: stub the encoder so webcam frames "encode".
        mod._encode_bgr_to_png = lambda fr: (b"PNG:" + fr.tag.encode()
                                             if isinstance(fr, _FakeFrame) else None)
        return mod, actions


# ─────────────────────────────────────────────────────────────────────────
# registration
# ─────────────────────────────────────────────────────────────────────────
class RegistrationTests(CameraSystemBase):
    def test_registers_all_actions(self):
        _mod, actions = self._load(bc=None, bridge=None)
        for name in ("camera_status", "situational_awareness", "where_am_i",
                     "look_around"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))

    def test_situational_awareness_aliases_where_am_i(self):
        mod, actions = self._load(bc=None, bridge=None)
        self.assertIs(actions["situational_awareness"], actions["where_am_i"])


# ─────────────────────────────────────────────────────────────────────────
# camera_status
# ─────────────────────────────────────────────────────────────────────────
class CameraStatusTests(CameraSystemBase):
    def test_two_webcams_live_plus_kinect_present_reports_three(self):
        bc = _fake_monolith(frame_ages={0: 0.2, 1: 0.5}, faces={0: 0.2})
        bridge = _fake_bridge(presence={"present": True, "count": 1,
                                        "nearest_m": 0.7, "facing": True,
                                        "ts": 0.0})
        _mod, actions = self._load(bc=bc, bridge=bridge)
        out = actions["camera_status"]("")
        self.assertIn("Three cameras", out)
        self.assertIn("live", out.lower())
        self.assertIn("Kinect", out)
        self.assertIn("one person", out)
        # Both webcam sides named.
        self.assertIn("left", out.lower())
        self.assertIn("right", out.lower())

    def test_stale_webcam_reported_dark(self):
        # idx0 fresh (live), idx1 stale (30s old → dark).
        bc = _fake_monolith(frame_ages={0: 0.2, 1: 30.0},
                            errors={1: "read returned no frame"})
        bridge = _fake_bridge(available=(False, "no sensor"))
        _mod, actions = self._load(bc=bc, bridge=bridge)
        out = actions["camera_status"]("")
        self.assertIn("live", out.lower())
        self.assertIn("dark", out.lower())

    def test_missing_webcam_frame_reported_dark(self):
        # idx0 has a frame; idx1 never delivered one at all → no frame_at entry.
        bc = _fake_monolith(frame_ages={0: 0.2})
        bridge = _fake_bridge(available=(False, "no sensor"))
        _mod, actions = self._load(bc=bc, bridge=bridge)
        out = actions["camera_status"]("")
        self.assertIn("dark", out.lower())

    def test_kinect_enabled_but_unavailable_reported_dark(self):
        # Kinect is opted-in (configured) but the sensor won't open. It still
        # counts toward the headline (the user HAS three cameras) but is named
        # honestly as dark, with the reason.
        bc = _fake_monolith(frame_ages={0: 0.2, 1: 0.3}, faces={0: 0.2})
        bridge = _fake_bridge(available=(False, "could not open Kinect sensor"))
        _mod, actions = self._load(bc=bc, bridge=bridge)
        out = actions["camera_status"]("")
        self.assertIn("Three cameras", out)        # all three are configured
        self.assertIn("Kinect", out)
        self.assertIn("dark", out.lower())         # but the Kinect is dark
        self.assertIn("could not open Kinect sensor", out)   # with the reason
        # The two webcams are still reported live.
        self.assertIn("live", out.lower())

    def test_kinect_disabled_not_counted_in_headline(self):
        bc = _fake_monolith(frame_ages={0: 0.2, 1: 0.3})
        _mod, actions = self._load(bc=bc, bridge=None, kinect_enabled=False)
        out = actions["camera_status"]("")
        self.assertIn("Two cameras", out)
        self.assertIn("off", out.lower())          # Kinect off
        self.assertIn("privacy", out.lower())

    def test_nothing_available_is_honest(self):
        # No monolith at all and Kinect disabled → can't reach the camera system.
        _mod, actions = self._load(bc=None, bridge=None, kinect_enabled=False)
        out = actions["camera_status"]("")
        self.assertIn("can't reach", out.lower())


# ─────────────────────────────────────────────────────────────────────────
# situational_awareness / where_am_i  (the FUSION)
# ─────────────────────────────────────────────────────────────────────────
class SituationalAwarenessTests(CameraSystemBase):
    def test_fuses_gaze_and_kinect_presence(self):
        bc = _fake_monolith(frame_ages={0: 0.2, 1: 0.3}, faces={0: 0.2})
        bridge = _fake_bridge(presence={"present": True, "count": 1,
                                        "nearest_m": 0.7, "facing": True,
                                        "ts": 0.0})
        mod, _actions = self._load(
            bc=bc, bridge=bridge,
            gaze_snapshot={"monitor": "right", "dwell_s": 12.0,
                           "face_visible": True, "source": "face_tracker"})
        s = mod.situational_awareness()
        self.assertEqual(s, {
            "present": True,
            "people": 1,
            "distance_m": 0.7,
            "facing_monitor": "right",
            "gaze": "right",
            "sources": {"webcams": True, "kinect": True, "gaze": "face_tracker"},
        })

    def test_where_am_i_one_liner_fuses_both(self):
        bc = _fake_monolith(frame_ages={0: 0.2, 1: 0.3}, faces={0: 0.2})
        bridge = _fake_bridge(presence={"present": True, "count": 1,
                                        "nearest_m": 0.7, "facing": True,
                                        "ts": 0.0})
        _mod, actions = self._load(
            bc=bc, bridge=bridge,
            gaze_snapshot={"monitor": "right", "dwell_s": 12.0,
                           "face_visible": True, "source": "face_tracker"})
        out = actions["where_am_i"]("")
        self.assertIn("desk", out.lower())
        # 0.7 m is spoken in imperial now (feet), not metres.
        self.assertIn("2 feet", out)
        self.assertIn("right monitor", out.lower())
        self.assertIn("alone", out.lower())
        # Names the sources it used.
        self.assertIn("webcam", out.lower())
        self.assertIn("kinect", out.lower())

    def test_webcam_only_path(self):
        # Kinect disabled — fusion falls back to gaze alone.
        bc = _fake_monolith(frame_ages={0: 0.2, 1: 0.3}, faces={0: 0.2})
        mod, actions = self._load(
            bc=bc, bridge=None, kinect_enabled=False,
            gaze_snapshot={"monitor": "left", "dwell_s": 5.0,
                           "face_visible": True, "source": "face_tracker"})
        s = mod.situational_awareness()
        self.assertTrue(s["present"])
        self.assertEqual(s["people"], 1)            # inferred lone user
        self.assertIsNone(s["distance_m"])          # no Kinect = no range
        self.assertEqual(s["facing_monitor"], "left")
        self.assertFalse(s["sources"]["kinect"])
        self.assertTrue(s["sources"]["webcams"])
        out = actions["where_am_i"]("")
        self.assertIn("left monitor", out.lower())
        self.assertIn("webcam", out.lower())
        self.assertNotIn("kinect", out.lower())     # didn't use it

    def test_kinect_only_path(self):
        # No webcams loaded (no monolith) but the Kinect sees a body.
        bridge = _fake_bridge(presence={"present": True, "count": 2,
                                        "nearest_m": 1.8, "facing": False,
                                        "ts": 0.0})
        mod, actions = self._load(bc=None, bridge=bridge)
        s = mod.situational_awareness()
        self.assertTrue(s["present"])
        self.assertEqual(s["people"], 2)
        self.assertEqual(s["distance_m"], 1.8)
        self.assertFalse(s["sources"]["webcams"])
        self.assertTrue(s["sources"]["kinect"])
        out = actions["where_am_i"]("")
        self.assertIn("kinect", out.lower())
        # Two people → reports company.
        self.assertIn("other", out.lower())

    def test_none_path_no_signal(self):
        mod, actions = self._load(bc=None, bridge=None, kinect_enabled=False)
        s = mod.situational_awareness()
        self.assertFalse(s["present"])
        self.assertEqual(s["people"], 0)
        self.assertFalse(s["sources"]["webcams"])
        self.assertFalse(s["sources"]["kinect"])
        out = actions["where_am_i"]("")
        self.assertIn("no camera signal", out.lower())

    def test_not_present_says_who_looked(self):
        # Webcams live but no face; Kinect up but empty room.
        bc = _fake_monolith(frame_ages={0: 0.2, 1: 0.3})   # frames but no faces
        bridge = _fake_bridge(presence={"present": False, "count": 0,
                                        "nearest_m": None, "facing": None,
                                        "ts": 0.0})
        _mod, actions = self._load(
            bc=bc, bridge=bridge,
            gaze_snapshot={"monitor": "away", "dwell_s": None,
                           "face_visible": False, "source": "face_tracker"})
        out = actions["where_am_i"]("")
        self.assertIn("don't see you", out.lower())
        self.assertIn("webcam", out.lower())
        self.assertIn("kinect", out.lower())


# ─────────────────────────────────────────────────────────────────────────
# look_around  (unified multi-camera vision sweep)
# ─────────────────────────────────────────────────────────────────────────
class LookAroundTests(CameraSystemBase):
    def _utils_capturing(self, answers):
        """make_fake_skill_utils whose ask_vision pops from `answers` per call
        and records the PNGs it received in `seen`."""
        seen = {"pngs": [], "questions": []}
        answers = list(answers)

        def _ask(question, png):
            seen["questions"].append(question)
            seen["pngs"].append(png)
            return answers.pop(0) if answers else ""
        utils = make_fake_skill_utils(ask_vision=mock.MagicMock(side_effect=_ask))
        return utils, seen

    def test_captures_all_sources_and_synthesizes(self):
        bc = _fake_monolith(
            frame_ages={0: 0.2, 1: 0.3},
            frames={0: _FakeFrame("R"), 1: _FakeFrame("L")})
        bridge = _fake_bridge(color_png=b"\x89PNG-KINECT")
        utils, seen = self._utils_capturing([
            "[local-vision] a person at the keyboard",
            "[local-vision] a tidy left monitor",
            "[local-vision] the rest of the room is empty",
        ])
        _mod, actions = self._load(bc=bc, bridge=bridge, utils=utils,
                                   vision_route="local")
        out = actions["look_around"]("")
        # 3 frames captured: 2 webcams + Kinect.
        self.assertEqual(len(seen["pngs"]), 3)
        self.assertIn(b"\x89PNG-KINECT", seen["pngs"])
        # Synthesized paragraph names each vantage.
        self.assertIn("left monitor webcam", out.lower())
        self.assertIn("right monitor webcam", out.lower())
        self.assertIn("kinect", out.lower())
        # Local route → cost note says free.
        self.assertIn("no cost", out.lower())

    def test_local_route_is_cost_conscious_note(self):
        bc = _fake_monolith(frame_ages={0: 0.2},
                            frames={0: _FakeFrame("R")})
        utils, _seen = self._utils_capturing(["[local-vision] you at the desk"])
        _mod, actions = self._load(bc=bc, bridge=None, kinect_enabled=False,
                                   utils=utils, vision_route="local")
        out = actions["look_around"]("")
        self.assertIn("no cost", out.lower())

    def test_cloud_route_notes_cloud_use(self):
        bc = _fake_monolith(frame_ages={0: 0.2},
                            frames={0: _FakeFrame("R")})
        # Cloud answer has NO [local-vision] prefix → flagged as cloud.
        utils, _seen = self._utils_capturing(["A person at a desk."])
        _mod, actions = self._load(bc=bc, bridge=None, kinect_enabled=False,
                                   utils=utils, vision_route="cloud")
        out = actions["look_around"]("")
        self.assertIn("cloud", out.lower())

    def test_single_camera_path(self):
        bc = _fake_monolith(frame_ages={0: 0.2},
                            frames={0: _FakeFrame("R")})
        utils, seen = self._utils_capturing(["[local-vision] just you, sir"])
        _mod, actions = self._load(bc=bc, bridge=None, kinect_enabled=False,
                                   utils=utils, vision_route="local")
        out = actions["look_around"]("")
        self.assertEqual(len(seen["pngs"]), 1)
        self.assertIn("1 camera", out.lower())
        self.assertIn("right monitor webcam", out.lower())

    def test_zero_cameras_is_honest(self):
        # No monolith, Kinect disabled → no cameras at all.
        utils, seen = self._utils_capturing([])
        _mod, actions = self._load(bc=None, bridge=None, kinect_enabled=False,
                                   utils=utils)
        out = actions["look_around"]("")
        self.assertIn("no cameras", out.lower())
        self.assertEqual(len(seen["pngs"]), 0)      # never called vision

    def test_missing_frame_never_raises(self):
        # Frame caches empty but the webcams are "configured" → cameras-up-but-
        # no-frame branch, no exception.
        bc = _fake_monolith()   # no frames at all
        bridge = _fake_bridge(available=(True, ""))
        utils, _seen = self._utils_capturing([])
        _mod, actions = self._load(bc=bc, bridge=bridge, utils=utils)
        out = actions["look_around"]("")   # must not raise
        self.assertIsInstance(out, str)

    def test_kinect_frame_missing_still_describes_webcams(self):
        # Kinect available but hands back no PNG → skipped; webcams still work.
        bc = _fake_monolith(frame_ages={0: 0.2},
                            frames={0: _FakeFrame("R")})
        bridge = _fake_bridge(color_png=None)   # no Kinect frame this tick
        utils, seen = self._utils_capturing(["[local-vision] you at the desk"])
        _mod, actions = self._load(bc=bc, bridge=bridge, utils=utils,
                                   vision_route="local")
        out = actions["look_around"]("")
        self.assertEqual(len(seen["pngs"]), 1)      # only the webcam frame
        self.assertIn("right monitor webcam", out.lower())

    def test_vision_unavailable_is_honest(self):
        bc = _fake_monolith(frame_ages={0: 0.2},
                            frames={0: _FakeFrame("R")})
        # No ask_vision in skill_utils, and no monolith.ask_vision either.
        utils = make_fake_skill_utils()
        utils.pop("ask_vision")
        _mod, actions = self._load(bc=bc, bridge=None, kinect_enabled=False,
                                   utils=utils)
        out = actions["look_around"]("")
        self.assertIn("vision isn't wired up", out.lower())


# ─────────────────────────────────────────────────────────────────────────
# face-ID identity enrichment of the "who's here" line (soft hook)
# ─────────────────────────────────────────────────────────────────────────
class IdentityEnrichmentTests(CameraSystemBase):
    """When FACE_ID_ENABLED is on and a face is recognised, where_am_i names the
    identity alongside the Kinect count; when off/unsure, the line is unchanged."""

    def _present_kinect(self, count=1):
        return _fake_bridge(presence={"present": True, "count": count,
                                      "nearest_m": 0.7, "facing": True,
                                      "ts": 0.0})

    def _load_present(self, *, count=1):
        bc = _fake_monolith(frame_ages={0: 0.2, 1: 0.3}, faces={0: 0.2})
        mod, actions = self._load(
            bc=bc, bridge=self._present_kinect(count),
            gaze_snapshot={"monitor": "right", "dwell_s": 12.0,
                           "face_visible": True, "source": "face_tracker"})
        return mod, actions

    # ---- the pure clause builder -----------------------------------------
    def test_company_clause_off_uses_kinect_count_alone(self):
        mod, _actions = self._load_present()
        # identity off → original phrasing
        self.assertEqual(
            mod._company_clause(1, True, {"on": False}), "alone")
        self.assertEqual(
            mod._company_clause(3, True, {"on": False}), "with 2 other people")
        # no kinect, no identity → empty (line unchanged)
        self.assertEqual(mod._company_clause(0, False, {"on": False}), "")

    def test_company_clause_owner_alone(self):
        mod, _actions = self._load_present()
        clause = mod._company_clause(
            1, True, {"on": True, "owner": True, "others": [], "unknown": 0})
        self.assertEqual(clause, "alone")

    def test_company_clause_owner_plus_unknown(self):
        mod, _actions = self._load_present()
        # Kinect counts 2 bodies, face-ID named only the owner → 1 unrecognised.
        clause = mod._company_clause(
            2, True, {"on": True, "owner": True, "others": [], "unknown": 1})
        self.assertEqual(clause, "with one person I don't recognise")

    def test_company_clause_named_other(self):
        mod, _actions = self._load_present()
        clause = mod._company_clause(
            2, True, {"on": True, "owner": True, "others": ["Dana"],
                      "unknown": 0})
        self.assertEqual(clause, "with Dana")

    def test_company_clause_named_plus_unknown(self):
        mod, _actions = self._load_present()
        clause = mod._company_clause(
            3, True, {"on": True, "owner": True, "others": ["Dana"],
                      "unknown": 1})
        self.assertIn("Dana", clause)
        self.assertIn("don't recognise", clause)

    # ---- end-to-end where_am_i with identity patched ---------------------
    def test_where_am_i_names_unrecognised_company(self):
        mod, actions = self._load_present(count=2)
        mod._identity_read = lambda: {"on": True, "owner": True,
                                      "others": [], "unknown": 1}
        out = actions["where_am_i"]("")
        self.assertIn("desk", out.lower())
        self.assertIn("don't recognise", out.lower())

    def test_where_am_i_identity_off_unchanged(self):
        # With identity OFF, the line must read exactly as the legacy behaviour
        # ("alone") — the enrichment is invisible.
        mod, actions = self._load_present(count=1)
        mod._identity_read = lambda: {"on": False, "owner": False,
                                      "others": [], "unknown": 0}
        out = actions["where_am_i"]("")
        self.assertIn("alone", out.lower())
        self.assertNotIn("recognise", out.lower())

    def test_identity_read_returns_off_when_flag_disabled(self):
        # The real _identity_read short-circuits to {"on": False} when the flag
        # is off — no engine import, no frame grab.
        mod, _actions = self._load(bc=None, bridge=None, kinect_enabled=False)
        # FACE_ID_ENABLED is not in the patched flags → reads False.
        self.assertEqual(mod._identity_read(), {"on": False, "owner": False,
                                                "others": [], "unknown": 0})


# ─────────────────────────────────────────────────────────────────────────
# no-leak guard
# ─────────────────────────────────────────────────────────────────────────
class NoLeakTests(CameraSystemBase):
    def test_actions_do_not_mutate_shared_state(self):
        bc = _fake_monolith(
            frame_ages={0: 0.2, 1: 0.3}, faces={0: 0.2},
            frames={0: _FakeFrame("R"), 1: _FakeFrame("L")})
        bridge = _fake_bridge(presence={"present": True, "count": 1,
                                        "nearest_m": 0.7, "facing": True,
                                        "ts": 0.0})
        seen = {"pngs": []}

        def _ask(q, p):
            seen["pngs"].append(p)
            return "[local-vision] ok"
        utils = make_fake_skill_utils(ask_vision=mock.MagicMock(side_effect=_ask))
        _mod, actions = self._load(bc=bc, bridge=bridge, utils=utils,
                                   vision_route="local")

        # Snapshot the shared caches before.
        before_frame_at = dict(bc._camera_last_frame_at)
        before_seen = dict(bc._camera_last_seen)
        before_frames = dict(bc._camera_latest_frame)
        before_errors = dict(bc._camera_last_read_error)

        actions["camera_status"]("")
        actions["where_am_i"]("")
        actions["look_around"]("")

        self.assertEqual(bc._camera_last_frame_at, before_frame_at)
        self.assertEqual(bc._camera_last_seen, before_seen)
        self.assertEqual(bc._camera_latest_frame, before_frames)
        self.assertEqual(bc._camera_last_read_error, before_errors)


if __name__ == "__main__":
    unittest.main()
