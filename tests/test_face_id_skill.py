"""Tests for skills/face_id — the face-recognition VOICE skill.

Loads the skill in isolation via the shared skill harness (no monolith boot),
then patches its seams — the engine (audio.face_id), the config flag reader, the
staging gate, and the monolith's shared webcam frame cache — with fakes so NO
hardware, NO model download, and NO real data/ are touched. Covers:

  * gating: every action refuses HONESTLY when FACE_ID_ENABLED is off, when the
    engine reports models unavailable, in staging, and when the webcam is dark.
  * enroll_face: grabs webcam frames, enrolls as the owner, honest success vs
    "couldn't get a clear look".
  * whoami: names the owner ("that's you"), an unrecognised stranger, mixed
    company, and "no face right now".
  * face_id_status / list_enrolled_faces / forget_face honest reporting.

stdlib unittest + mock only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── fakes ──────────────────────────────────────────────────────────────────

class _Frame:
    def __init__(self, tag="f"):
        self.tag = tag

    def copy(self):
        return self


def _fake_monolith(frames=None):
    """A stand-in bobert_companion exposing the shared webcam frame cache the
    skill reads (index -> frame). `frames` maps camera index to a _Frame."""
    bc = types.ModuleType("bobert_companion")
    bc._camera_state_lock = threading.Lock()
    bc._camera_latest_frame = dict(frames or {})
    bc._is_staging = lambda: False
    return bc


def _fake_engine(*, available=(True, ""), recognize=None, enroll_n=1,
                 enrolled=None, enroll_unknown_result=None):
    """A stand-in audio.face_id engine module (only what the skill calls).

    `enroll_n` drives the owner path (enroll()). The guest path (enroll_unknown)
    returns `enroll_unknown_result` verbatim when given, else a success dict
    derived from enroll_n so the legacy guest tests keep passing."""
    m = types.ModuleType("audio.face_id")
    m.is_available = lambda: available
    m.recognize = lambda frame: (recognize if recognize is not None else [])
    m._enroll_calls = []

    def _enroll(name, frames):
        m._enroll_calls.append((name, list(frames)))
        return enroll_n
    m.enroll = _enroll

    # Guest enrollment: capture only the nearest UNKNOWN face. Records its calls
    # separately and returns the rich {"added", "saw_face", "saw_unknown"} dict.
    m._enroll_unknown_calls = []
    _default_unknown = {"added": enroll_n,
                        "saw_face": True,
                        "saw_unknown": enroll_n > 0}

    def _enroll_unknown(name, frames):
        m._enroll_unknown_calls.append((name, list(frames)))
        return (enroll_unknown_result if enroll_unknown_result is not None
                else dict(_default_unknown))
    m.enroll_unknown = _enroll_unknown
    m.list_enrolled = lambda: (enrolled if enrolled is not None else [])
    m._forget_calls = []

    def _forget(name):
        m._forget_calls.append(name)
        return bool(enrolled)  # pretend success only if someone is enrolled
    m.forget = _forget
    return m


_CAMERAS = [
    {"index": 1, "label": "Left webcam", "primary": False, "look_x": 0.15},
    {"index": 0, "label": "Right webcam", "primary": True, "look_x": 0.85},
]


class FaceIdSkillBase(unittest.TestCase):
    def _load(self, *, bc=None, engine=None, enabled=True, staging=False,
              user_name="", cameras=None):
        cameras = _CAMERAS if cameras is None else cameras
        mod, actions = load_skill_isolated("face_id", register=True)

        # Seam: monolith + engine.
        mod._bc = lambda: bc
        mod._engine = lambda: engine

        # Seam: config flag + staging + owner name (patch the skill's own
        # helpers so import-resolution of core.config never leaks the real one).
        flags = {"FACE_ID_ENABLED": enabled}
        mod._cfg_flag = lambda name, default=False: bool(flags.get(name, default))
        mod._is_staging = lambda: staging
        mod._owner_name = lambda: (user_name.strip() or "owner")

        # CAMERAS comes from core.config via a local import — patch the real
        # (stdlib-only, CI-present) module attribute and restore after.
        import core.config as _real_cfg
        p = mock.patch.object(_real_cfg, "CAMERAS", cameras)
        p.start()
        self.addCleanup(p.stop)
        return mod, actions


# ─── registration ───────────────────────────────────────────────────────────
class RegistrationTests(FaceIdSkillBase):
    def test_registers_all_actions(self):
        _mod, actions = self._load(bc=None, engine=None)
        for name in ("enroll_face", "learn_my_face", "remember_my_face",
                     "learn_guest", "remember_their_face",
                     "remember_this_person", "learn_their_face",
                     "whoami", "who_am_i", "recognize_face",
                     "do_you_recognize_me", "whos_at_the_desk",
                     "face_id_status", "forget_face", "list_enrolled_faces"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))

    def test_aliases_share_handler(self):
        _mod, actions = self._load(bc=None, engine=None)
        self.assertIs(actions["whoami"], actions["recognize_face"])
        self.assertIs(actions["enroll_face"], actions["learn_my_face"])


# ─── gating: OFF / staging / models / camera ────────────────────────────────
class GatingTests(FaceIdSkillBase):
    def test_enroll_refuses_when_disabled(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine()
        _mod, actions = self._load(bc=bc, engine=eng, enabled=False)
        out = actions["enroll_face"]("")
        self.assertIn("off", out.lower())
        # No enrollment attempted.
        self.assertEqual(eng._enroll_calls, [])

    def test_whoami_refuses_in_staging(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine()
        _mod, actions = self._load(bc=bc, engine=eng, staging=True)
        out = actions["whoami"]("")
        self.assertIn("staging", out.lower())

    def test_enroll_refuses_when_models_unavailable(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine(available=(False, "models not downloaded"))
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["enroll_face"]("")
        self.assertIn("isn't ready", out.lower())
        self.assertIn("models not downloaded", out)
        self.assertEqual(eng._enroll_calls, [])

    def test_enroll_refuses_when_camera_dark(self):
        bc = _fake_monolith(frames={})  # no frame for the primary index
        eng = _fake_engine()
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["enroll_face"]("")
        self.assertIn("webcam", out.lower())
        self.assertEqual(eng._enroll_calls, [])

    def test_whoami_no_frame(self):
        bc = _fake_monolith(frames={})
        eng = _fake_engine()
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["whoami"]("")
        self.assertIn("can't see", out.lower())


# ─── enroll_face ────────────────────────────────────────────────────────────
class EnrollFaceTests(FaceIdSkillBase):
    def test_enroll_success_owner(self):
        bc = _fake_monolith(frames={0: _Frame("p")})
        eng = _fake_engine(enroll_n=4)
        _mod, actions = self._load(bc=bc, engine=eng, user_name="")
        out = actions["enroll_face"]("")
        self.assertIn("recognise you", out.lower())
        # The engine was asked to enroll the owner with >=1 frame.
        self.assertEqual(len(eng._enroll_calls), 1)
        name, frames = eng._enroll_calls[0]
        self.assertEqual(name, "owner")
        self.assertGreaterEqual(len(frames), 1)

    def test_enroll_success_named_user(self):
        bc = _fake_monolith(frames={0: _Frame("p")})
        eng = _fake_engine(enroll_n=3)
        _mod, actions = self._load(bc=bc, engine=eng, user_name="Tony")
        out = actions["enroll_face"]("")
        self.assertEqual(eng._enroll_calls[0][0], "Tony")
        self.assertIn("Tony", out)

    def test_enroll_no_clear_face(self):
        bc = _fake_monolith(frames={0: _Frame("p")})
        eng = _fake_engine(enroll_n=0)  # engine found no usable face
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["enroll_face"]("")
        self.assertIn("clear look", out.lower())


# ─── learn_guest (enroll a visible stranger under a spoken name) ─────────────
class LearnGuestTests(FaceIdSkillBase):
    def test_learn_guest_success(self):
        bc = _fake_monolith(frames={0: _Frame("p")})
        eng = _fake_engine(enroll_n=4)
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["learn_guest"]("Sam")
        self.assertIn("Sam", out)
        self.assertIn("recognise", out.lower())
        # The engine enrolled under the GUEST name via the UNKNOWN-only path
        # (never the plain enroll(), which would grab the largest/owner face).
        self.assertEqual(eng._enroll_calls, [])
        self.assertEqual(len(eng._enroll_unknown_calls), 1)
        name, frames = eng._enroll_unknown_calls[0]
        self.assertEqual(name, "Sam")
        self.assertGreaterEqual(len(frames), 1)

    def test_learn_guest_strips_lead_in_phrase(self):
        bc = _fake_monolith(frames={0: _Frame("p")})
        eng = _fake_engine(enroll_n=2)
        _mod, actions = self._load(bc=bc, engine=eng)
        # The router may pass the whole spoken tail through.
        actions["remember_their_face"]("this is sam")
        self.assertEqual(eng._enroll_unknown_calls[0][0], "Sam")  # cleaned+cased

    def test_learn_guest_requires_a_name(self):
        bc = _fake_monolith(frames={0: _Frame("p")})
        eng = _fake_engine(enroll_n=3)
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["learn_guest"]("")
        self.assertIn("name", out.lower())
        self.assertEqual(eng._enroll_unknown_calls, [])    # nothing enrolled

    def test_learn_guest_rejects_placeholder_name(self):
        bc = _fake_monolith(frames={0: _Frame("p")})
        eng = _fake_engine(enroll_n=3)
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["learn_guest"]("guest")
        self.assertIn("real name", out.lower())
        self.assertEqual(eng._enroll_unknown_calls, [])

    def test_learn_guest_refuses_when_disabled(self):
        bc = _fake_monolith(frames={0: _Frame("p")})
        eng = _fake_engine()
        _mod, actions = self._load(bc=bc, engine=eng, enabled=False)
        out = actions["learn_guest"]("Sam")
        self.assertIn("off", out.lower())
        self.assertEqual(eng._enroll_unknown_calls, [])

    def test_learn_guest_refuses_when_camera_dark(self):
        bc = _fake_monolith(frames={})                     # no primary frame
        eng = _fake_engine()
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["learn_guest"]("Sam")
        self.assertIn("webcam", out.lower())
        self.assertEqual(eng._enroll_unknown_calls, [])

    def test_learn_guest_no_clear_face(self):
        bc = _fake_monolith(frames={0: _Frame("p")})
        # No face at all in view → couldn't get a clear look.
        eng = _fake_engine(enroll_unknown_result={
            "added": 0, "saw_face": False, "saw_unknown": False})
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["learn_guest"]("Sam")
        self.assertIn("clear look", out.lower())
        self.assertIn("Sam", out)

    def test_learn_guest_all_visible_already_known(self):
        """The owner is closest, but a face IS visible and it's already
        recognised → JARVIS must NOT enrol the owner under the guest name. It
        reports that everyone visible is already known rather than corrupting
        recognition (the P2 'owner-closest gets stored as guest' bug)."""
        bc = _fake_monolith(frames={0: _Frame("p")})
        eng = _fake_engine(enroll_unknown_result={
            "added": 0, "saw_face": True, "saw_unknown": False})
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["learn_guest"]("Sam")
        self.assertIn("already", out.lower())
        self.assertIn("recognise", out.lower())
        self.assertNotIn("clear look", out.lower())


# ─── whoami ─────────────────────────────────────────────────────────────────
class WhoamiTests(FaceIdSkillBase):
    def test_owner_is_you(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine(recognize=[{"name": "owner", "score": 0.8,
                                       "bbox": [0, 0, 80, 80]}])
        _mod, actions = self._load(bc=bc, engine=eng, user_name="")
        out = actions["whoami"]("")
        self.assertEqual(out, "That's you, sir.")

    def test_owner_by_configured_name_is_you(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine(recognize=[{"name": "Tony", "score": 0.8,
                                       "bbox": [0, 0, 80, 80]}])
        _mod, actions = self._load(bc=bc, engine=eng, user_name="Tony")
        out = actions["whoami"]("")
        self.assertEqual(out, "That's you, sir.")

    def test_single_unknown(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine(recognize=[{"name": "unknown", "score": 0.1,
                                       "bbox": [0, 0, 80, 80]}])
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["whoami"]("")
        self.assertIn("don't recognise", out.lower())

    def test_multiple_unknown(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine(recognize=[
            {"name": "unknown", "score": 0.1, "bbox": [0, 0, 80, 80]},
            {"name": "unknown", "score": 0.0, "bbox": [90, 0, 80, 80]}])
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["whoami"]("")
        self.assertIn("2 people", out)

    def test_owner_plus_unknown(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine(recognize=[
            {"name": "owner", "score": 0.8, "bbox": [0, 0, 80, 80]},
            {"name": "unknown", "score": 0.1, "bbox": [90, 0, 80, 80]}])
        _mod, actions = self._load(bc=bc, engine=eng, user_name="")
        out = actions["whoami"]("")
        self.assertIn("you", out.lower())
        self.assertIn("recognise", out.lower())

    def test_no_face(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine(recognize=[])  # detector found nothing
        _mod, actions = self._load(bc=bc, engine=eng)
        out = actions["whoami"]("")
        self.assertIn("don't see a face", out.lower())


# ─── whoami GROUNDING: live frame is the source of truth, never memory ───────
# Regression guard for the "ducked under the desk" bug (caught live): the owner
# is enrolled/known, but when he is OUT OF FRAME whoami must report what the
# CAMERA sees on the CURRENT frame ("no one"), not his remembered name. These
# pin the contract the prompt now routes to: present -> name, empty -> no one,
# read FRESH every call.
class WhoamiGroundingTests(FaceIdSkillBase):
    def test_present_reports_name_empty_reports_no_one(self):
        """The recognizer drives the answer: a face -> 'you', nothing -> 'no
        one' — never the enrolled identity when the frame is empty."""
        bc = _fake_monolith(frames={0: _Frame()})

        # present -> the owner is named
        eng_present = _fake_engine(recognize=[{"name": "Tony", "score": 0.8,
                                               "bbox": [0, 0, 80, 80]}])
        _mod, actions = self._load(bc=bc, engine=eng_present, user_name="Tony")
        self.assertEqual(actions["whoami"](""), "That's you, sir.")

        # empty -> "no one", NOT "Tony" (even though Tony is the known owner)
        eng_empty = _fake_engine(recognize=[])  # camera sees nobody right now
        _mod, actions = self._load(bc=bc, engine=eng_empty, user_name="Tony")
        out = actions["whoami"]("")
        self.assertIn("don't see a face", out.lower())
        self.assertNotIn("tony", out.lower())  # never leak the remembered name

    def test_reads_a_fresh_frame_and_recognizes_it_every_call(self):
        """whoami must grab a NEW frame and re-run recognition each call — no
        cached verdict. We flip the live frame from owner-present to empty and
        the answer must follow the current frame, not the prior one."""
        bc = _fake_monolith(frames={0: _Frame("present")})
        # The engine answers based on which frame it is handed RIGHT NOW, so a
        # stale/cached result would be detectable as a wrong answer.
        eng = _fake_engine()
        grabbed = []

        def _recognize(frame):
            grabbed.append(frame)
            # Owner is visible only while the live frame is the "present" one.
            if getattr(frame, "tag", None) == "present":
                return [{"name": "Tony", "score": 0.8, "bbox": [0, 0, 80, 80]}]
            return []
        eng.recognize = _recognize
        _mod, actions = self._load(bc=bc, engine=eng, user_name="Tony")

        # 1st call: owner in frame -> "you".
        self.assertEqual(actions["whoami"](""), "That's you, sir.")

        # Owner ducks out: the shared frame cache now holds an empty frame.
        bc._camera_latest_frame[0] = _Frame("empty")
        out = actions["whoami"]("")
        self.assertIn("don't see a face", out.lower())
        self.assertNotIn("tony", out.lower())

        # Proof it re-read: recognize() was called once PER whoami, on the
        # frame live at that moment (no cached verdict, no stale frame).
        self.assertEqual(len(grabbed), 2)
        self.assertEqual(getattr(grabbed[0], "tag", None), "present")
        self.assertEqual(getattr(grabbed[1], "tag", None), "empty")


# ─── status / list / forget ─────────────────────────────────────────────────
class StatusListForgetTests(FaceIdSkillBase):
    def test_status_on_with_enrolled(self):
        bc = _fake_monolith(frames={0: _Frame()})
        eng = _fake_engine(available=(True, ""),
                           enrolled=[{"name": "owner", "count": 5}])
        _mod, actions = self._load(bc=bc, engine=eng, enabled=True)
        out = actions["face_id_status"]("")
        self.assertIn("on", out.lower())
        self.assertIn("owner", out)
        self.assertIn("loaded", out.lower())
        self.assertIn("live", out.lower())  # webcam frame present

    def test_status_off_no_enrolled(self):
        bc = _fake_monolith(frames={})
        eng = _fake_engine(available=(False, "models not downloaded"),
                           enrolled=[])
        _mod, actions = self._load(bc=bc, engine=eng, enabled=False)
        out = actions["face_id_status"]("")
        self.assertIn("off", out.lower())
        self.assertIn("no one is enrolled", out.lower())
        self.assertIn("dark", out.lower())

    def test_status_staging(self):
        _mod, actions = self._load(bc=None, engine=_fake_engine(), staging=True)
        out = actions["face_id_status"]("")
        self.assertIn("staging", out.lower())

    def test_list_enrolled_names(self):
        eng = _fake_engine(enrolled=[{"name": "owner", "count": 3},
                                     {"name": "Tony", "count": 1}])
        _mod, actions = self._load(bc=None, engine=eng)
        out = actions["list_enrolled_faces"]("")
        self.assertIn("owner", out)
        self.assertIn("Tony", out)

    def test_list_enrolled_empty(self):
        eng = _fake_engine(enrolled=[])
        _mod, actions = self._load(bc=None, engine=eng)
        out = actions["list_enrolled_faces"]("")
        self.assertIn("No faces", out)

    def test_forget_existing(self):
        eng = _fake_engine(enrolled=[{"name": "Tony", "count": 1}])
        _mod, actions = self._load(bc=None, engine=eng)
        out = actions["forget_face"]("Tony")
        self.assertIn("Forgotten", out)
        self.assertIn("Tony", eng._forget_calls)

    def test_forget_missing(self):
        eng = _fake_engine(enrolled=[])  # forget() returns False
        _mod, actions = self._load(bc=None, engine=eng)
        out = actions["forget_face"]("ghost")
        self.assertIn("don't have", out.lower())


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
