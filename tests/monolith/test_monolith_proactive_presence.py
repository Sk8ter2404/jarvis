"""Monolith wiring for the proactive-remark gates (2026-09-30).

THE LIVE EVIDENCE (grep "[proactive]"):
  session_2026-09-29_22-06-02.log — "You seem rather determined this evening,
    sir." spoken at 22:43:17, 22:49:21, 23:17:02, 23:36:11, 00:02:42, 07:54:17,
    08:07:45 and 08:14:14: a phrasebook example copied verbatim, eight times.
  session_2026-09-30_09-48-41.log — the owner away from home and silent all
    session ("[noise] ignored (owner silent this session, ...)"), yet remarks
    at 09:56:22 and, 212 s later, 09:59:54 — the latter the same evening line
    at ten in the morning. The camera gate was open because any single-frame
    Haar hit stamped last_face_seen.

A1  generate_proactive_comment drops a remark whose time of day the clock
    contradicts or that copies a persona example, and tells the model the
    local time.
A2  _do_proactive_turn never speaks a repeat / near-repeat of a recent remark.
A3  should_be_proactive needs the owner to have SPOKEN recently AND a
    sustained face; the face-track loop stamps last_face_seen only for a
    sustained, qualified detection on fresh frames.
A4  an unanswered remark backs off; two buy silence until the owner speaks.

Every test drives the REAL function (the LLM, the voice and the camera are
faked; no device is opened). On a tree without the gates the reproductions
fail on their assertions, not on a missing name. Synthetic lines only.

    python -m unittest tests.monolith.test_monolith_proactive_presence
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

LIVE_LINE = "You seem rather determined this evening, sir."
ORIGINAL = "I've noticed you've been quite focused for a while, sir."


def _local(h, m, s):
    """A local timestamp on 2026-09-30 (timezone-independent)."""
    return time.mktime((2026, 9, 30, h, m, s, 0, 0, -1))


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _quiet(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()

    def _keep_history(self):
        bc = self.bc
        n = len(bc.conversation_history)
        self.addCleanup(lambda: bc.conversation_history.__delitem__(
            slice(n, None)))

    def _fake_turn_io(self):
        """_do_proactive_turn with its voice / animation / action parser
        faked. Returns the _speak mock."""
        bc = self.bc
        self._p(bc, "pause_face_tracking")
        self._p(bc, "resume_face_tracking")
        self._p(bc, "set_state")
        self._p(bc, "_thinking_loop")
        self._p(bc, "_reset_see_screen_budget")
        self._p(bc, "parse_and_run_actions", side_effect=lambda t: (t, []))
        self._p(bc, "_apply_quip_layer", side_effect=lambda s, r: s)
        self._keep_history()
        return self._p(bc, "_speak")


# ── A1: the text of a remark ───────────────────────────────────────────────
class RemarkTextTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "_system_prompt", "SYNTHETIC BASE PROMPT")

    def _generate(self, at):
        fn = self.bc.generate_proactive_comment
        if "now" in inspect.signature(fn).parameters:
            return self._quiet(fn, now=at)
        return self._quiet(fn)

    def test_the_live_line_at_ten_in_the_morning_is_not_spoken(self):
        self._p(self.bc, "_llm_quick", return_value=LIVE_LINE)
        out, log = self._generate(_local(9, 59, 54))
        self.assertEqual(out, "", "the evening line was returned at 09:59")
        self.assertIn("[proactive] dropped (says 'this evening' at 09:00", log)

    def test_the_live_line_is_not_spoken_in_the_evening_either(self):
        # 22:43:17: the hour fits, but it is a phrasebook line copied whole.
        self._p(self.bc, "_llm_quick", return_value=LIVE_LINE)
        out, log = self._generate(_local(22, 43, 17))
        self.assertEqual(out, "")
        self.assertIn("[proactive] dropped (copies the persona example", log)

    def test_a_tagged_near_copy_is_not_spoken(self):
        self._p(self.bc, "_llm_quick", return_value=(
            "[intent:observation] You seem rather determined this morning, "
            "sir."))
        out, _ = self._generate(_local(9, 30, 0))
        self.assertEqual(out, "")

    def test_an_original_remark_is_spoken_and_the_prompt_has_the_clock(self):
        llm = self._p(self.bc, "_llm_quick", return_value=ORIGINAL + "\nx")
        out, log = self._generate(_local(9, 59, 54))
        self.assertEqual(out, ORIGINAL)
        self.assertNotIn("dropped", log)
        user = llm.call_args.kwargs["user"]
        self.assertIn("Local time: 9:59 AM, morning", user)
        self._generate(_local(22, 43, 17))
        self.assertIn("Local time: 10:43 PM, night",
                      llm.call_args.kwargs["user"])
        # The clock rides in the USER message: the system part is identical
        # at both hours, so the local model's cached prefix survives.
        systems = {c.kwargs["system"] for c in llm.call_args_list}
        self.assertEqual(len(systems), 1)
        self.assertNotIn("a light remark about the hour", systems.pop())


# ── A2: no repeats ─────────────────────────────────────────────────────────
class RepeatTests(_Base):
    def test_the_same_remark_is_spoken_once(self):
        bc = self.bc
        speak = self._fake_turn_io()
        llm = self._p(bc, "_llm_quick", return_value=ORIGINAL)
        self._quiet(bc._do_proactive_turn, {})
        _, log = self._quiet(bc._do_proactive_turn, {})
        self.assertEqual(speak.call_count, 1, "a remark was repeated")
        self.assertIn("[proactive] dropped (repeats a recent remark)", log)
        # a near-repeat is a repeat too
        llm.return_value = "You've been quite focused for a while, sir."
        self._quiet(bc._do_proactive_turn, {})
        self.assertEqual(speak.call_count, 1)
        # ...and the model is told what it said recently
        self.assertIn(ORIGINAL, llm.call_args.kwargs["user"])
        # a different remark still gets through
        llm.return_value = "Shall I queue something calmer to work to, sir?"
        self._quiet(bc._do_proactive_turn, {})
        self.assertEqual(speak.call_count, 2)

    def test_the_ring_is_bounded(self):
        bc = self.bc
        speak = self._fake_turn_io()
        llm = self._p(bc, "_llm_quick")
        ring = int(getattr(bc, "PROACTIVE_RECENT_RING", 8))
        for i in range(ring + 3):
            llm.return_value = f"Remark number {i} about topic {i * 7919}."
            self._quiet(bc._do_proactive_turn, {})
        self.assertEqual(speak.call_count, ring + 3)
        self.assertEqual(len(bc._proactive_recent), ring)


# ── A3 + A4: should_be_proactive ───────────────────────────────────────────
class _GateBase(_Base):
    """Every otherwise-firing condition of the live 09:56 / 09:59 remarks:
    enabled, long silence, a fresh last_face_seen, the dice at 0."""

    T0 = 500_000.0

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "PROACTIVE_ENABLED", True)
        self._p(bc, "PROACTIVE_MIN_SILENCE", 180)
        self._p(bc, "PROACTIVE_MAX_SILENCE", 900)
        self._p(bc, "PROACTIVE_REQUIRE_FACE", True)
        self._p(bc, "_voice_mood_response", None)
        self._p(bc.random, "random", return_value=0.0)
        bc._focus_mode[0] = False
        self.clock = [self.T0]
        self._p(bc, "_proactive_mono", lambda: self.clock[0], create=True)
        self._p(bc, "_last_owner_voice_at", [0.0], create=True)
        bc._last_owner_turn_at[0] = 0.0
        self._silence(1000)
        bc.last_face_seen = time.time()

    def _silence(self, s):
        self.bc.last_speech_time = time.time() - s

    def _owner_spoke(self, ago=0.0):
        at = self.clock[0] - ago
        self.bc._last_owner_voice_at[0] = at
        self.bc._last_owner_turn_at[0] = at

    def _check(self):
        return self._quiet(self.bc.should_be_proactive)

    def _remark(self, text):
        """The REAL _do_proactive_turn at the current fake time."""
        bc = self.bc
        speak = self._fake_turn_io()
        self._p(bc, "generate_proactive_comment", return_value=text)
        self._quiet(bc._do_proactive_turn, {})
        self._silence(0)
        return speak


class PresenceGateTests(_GateBase):
    def test_live_owner_silent_all_session_is_not_addressed(self):
        # 09:56:22: the owner away, not one word this session, face "seen".
        fire, log = self._check()
        self.assertFalse(fire, "a remark to an owner who never spoke")
        self.assertIn("[proactive] holding (owner has not spoken this "
                      "session)", log)

    def test_the_hold_is_logged_once(self):
        _, first = self._check()
        _, second = self._check()
        self.assertIn("[proactive] holding", first)
        self.assertNotIn("[proactive] holding", second)

    def test_owner_who_spoke_long_ago_is_not_addressed(self):
        self._owner_spoke(ago=float(getattr(
            self.bc, "PROACTIVE_OWNER_VOICE_WINDOW_S", 1200)) + 1.0)
        fire, log = self._check()
        self.assertFalse(fire)
        self.assertIn("owner last spoke", log)

    def test_owner_who_just_went_quiet_at_the_desk_is(self):
        self._owner_spoke(ago=300.0)
        self.assertTrue(self._check()[0])

    def test_a_stale_face_still_blocks(self):
        self._owner_spoke(ago=300.0)
        self.bc.last_face_seen = time.time() - 61
        fire, log = self._check()
        self.assertFalse(fire)
        self.assertIn("no sustained face", log)

    def test_only_a_mic_turn_stamps_the_owner_voice(self):
        src = inspect.getsource(self.bc.main)
        at = src.index("_note_owner_turn()")
        self.assertIn("if _injected_text is None:\n"
                      "                    _note_owner_voice()",
                      src[at:at + 300])
        note = getattr(self.bc, "_note_owner_voice", None)
        self.assertIsNotNone(note, "no owner-voice stamp")
        self.assertIn("_last_owner_voice_at[0] = time.monotonic()",
                      inspect.getsource(note))


class PacingTests(_GateBase):
    def setUp(self):
        super().setUp()
        self._owner_spoke(ago=60.0)

    def test_live_follow_up_212_s_later_is_held(self):
        # 09:56:22 remark, nobody answers; 09:59:54 is 212 s later.
        self.assertTrue(self._check()[0])
        self._remark(ORIGINAL)
        self.clock[0] += 212
        self._silence(212)
        fire, log = self._check()
        self.assertFalse(fire, "an ignored remark was followed 212 s later")
        self.assertIn("[proactive] holding (", log)

    def test_one_more_after_the_cooldown_then_quiet_until_he_speaks(self):
        bc = self.bc
        self._p(bc, "PROACTIVE_OWNER_VOICE_WINDOW_S", 100_000)
        self._remark(ORIGINAL)
        self.clock[0] += 601
        self._silence(1000)
        self.assertTrue(self._check()[0], "the cooldown never ended")
        self._remark("Shall I queue something calmer to work to, sir?")
        self.clock[0] += 5000
        self._silence(5000)
        fire, log = self._check()
        self.assertFalse(fire, "a third remark with two unanswered")
        self.assertIn("2 unanswered remarks", log)
        # He speaks: the streak is over.
        self._owner_spoke()
        self.clock[0] += 1
        self.assertTrue(self._check()[0])

    def test_a_dropped_remark_waits_before_asking_the_model_again(self):
        self._remark("")        # the text gate dropped it
        self.clock[0] += 100
        self._silence(1000)
        fire, log = self._check()
        self.assertFalse(fire)
        self.assertIn("last attempt", log)
        self.clock[0] += 201
        self.assertTrue(self._check()[0])


# ── A3: the face-track loop ────────────────────────────────────────────────
class _StubKinectBridge:
    """Never opens a sensor (see tests/monolith/test_monolith_sec2.py)."""

    class _Cap:
        _open_error = "kinect stubbed out for tests"

        def isOpened(self):   # noqa: N802
            return False

        def read(self):
            return False, None

        def get(self, _p):
            return 0.0

        def set(self, *_a, **_k):
            return False

        def release(self):
            pass

    def KinectCapture(self):   # noqa: N802
        return self._Cap()


class FaceLoopPresenceTests(_Base):
    """The REAL face-track producer, one camera, cv2 faked, the Haar cascade
    faked to 'find' a face. No device is opened (same firewall as
    FaceTrackingThreadTests in test_monolith_sec2)."""

    CAM = {"index": 0, "label": "X", "name": "Test Webcam 0",
           "primary": True, "look_x": 0.5, "look_y": 0.5}
    BIG = [[400, 150, 300, 300]]     # 300 px of 1280: someone at the desk
    SMALL = [[600, 300, 60, 60]]     # 60 px of 1280: a face across the room

    def setUp(self):
        super().setUp()
        import numpy as np
        bc = self.bc
        self.np = np
        self._p(bc, "KINECT_AS_CAMERA", False)
        self._p(bc, "_kinect_bridge", _StubKinectBridge())
        self._p(bc, "_dshow_name_to_index", side_effect=lambda _n: 0)
        for ev in (bc._face_track_stop, bc._face_track_pause,
                   bc._face_track_camera_off):
            ev.clear()
            self.addCleanup(ev.clear)
        bc.last_face_seen = 0.0
        # Loose thresholds, so the ONLY reason not to confirm is the one
        # under test (the loop runs in milliseconds here, not seconds).
        mod = getattr(bc, "_face_presence", None)
        if mod is not None:
            self._p(mod, "MIN_SPAN_S", 0.0)
            self._p(mod, "MIN_HITS", 3)
            self._p(mod, "WINDOW_S", 60.0)

    def _run(self, frames, reads, boxes=None, relaxed_only=False):
        bc, np = self.bc, self.np
        empty = np.empty((0, 4))
        box = np.array(self.BIG if boxes is None else boxes)

        def _detect(_gray, **kw):
            if relaxed_only:
                return box if kw.get("minNeighbors") == 3 else empty
            return box

        cascade = mock.Mock()
        cascade.detectMultiScale.side_effect = _detect
        it = iter(frames)
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 1280
        cap.read.side_effect = lambda: (True, next(it))
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        cv2.error = bc.cv2.error
        stop, n = bc._face_track_stop, [0]

        def _counted(*_a, **_k):
            n[0] += 1
            if n[0] >= reads:
                stop.set()

        box_out: dict = {}

        def _runner():
            try:
                bc._face_tracking_thread()
            except BaseException as exc:   # noqa: BLE001
                box_out["exc"] = exc

        with mock.patch.object(bc, "cv2", cv2), \
                mock.patch.object(bc, "CAMERAS", [dict(self.CAM)]), \
                mock.patch.object(bc, "_face_cascade", cascade), \
                mock.patch.object(bc, "_profile_cascade", None), \
                mock.patch.object(bc, "_note_camera_read_attempt",
                                  side_effect=_counted), \
                mock.patch.object(bc, "_hud_camera_preview_enabled",
                                  return_value=False), \
                mock.patch.object(bc, "find_camera_locking_processes",
                                  return_value=[]), \
                mock.patch.object(bc, "send"), \
                mock.patch.object(bc.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join(8.0)
            if t.is_alive():
                stop.set()
                t.join(10.0)
                self.fail("face-track loop did not stop")
        if "exc" in box_out:
            raise box_out["exc"]
        self.assertGreaterEqual(n[0], reads, "the loop never read")

    def _fresh(self, count):
        a = self.np.zeros((720, 1280, 3), dtype=self.np.uint8)
        b = a.copy()
        b[::8, ::8] = 9
        return [a if i % 2 else b for i in range(count + 4)]

    def test_one_frame_with_a_face_is_not_presence(self):
        self._run(self._fresh(1), reads=1)
        self.assertEqual(self.bc.last_face_seen, 0.0,
                         "a single-frame hit stamped last_face_seen")

    def test_a_re_served_frame_is_not_presence(self):
        same = self.np.zeros((720, 1280, 3), dtype=self.np.uint8)
        self._run([same] * 20, reads=12)
        self.assertEqual(self.bc.last_face_seen, 0.0,
                         "a re-served frame kept last_face_seen fresh")

    def test_a_relaxed_pass_hit_is_not_presence(self):
        self._run(self._fresh(12), reads=12, relaxed_only=True)
        self.assertEqual(self.bc.last_face_seen, 0.0)

    def test_a_face_across_the_room_is_not_presence(self):
        self._run(self._fresh(12), reads=12, boxes=self.SMALL)
        self.assertEqual(self.bc.last_face_seen, 0.0)

    def test_a_sustained_face_on_fresh_frames_is_presence(self):
        before = time.time()
        self._run(self._fresh(12), reads=12)
        self.assertGreaterEqual(self.bc.last_face_seen, before,
                                "a sustained face was never confirmed")


class PresenceWiringTests(_Base):
    """Source-level: _face_presence_note is the ONLY writer of
    last_face_seen, and the loop feeds it every frame it detects on."""

    def test_only_the_presence_note_writes_last_face_seen(self):
        with open(self.bc.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        writers = set()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, (ast.Assign, ast.AugAssign)):
                    targets = (node.targets if isinstance(node, ast.Assign)
                               else [node.target])
                    if any(isinstance(t, ast.Name) and t.id == "last_face_seen"
                           for t in targets):
                        writers.add(fn.name)
        self.assertEqual(writers, {"_face_presence_note"})

    def test_the_loop_feeds_every_detected_frame(self):
        src = inspect.getsource(self.bc._face_tracking_thread_body)
        det = src.index("face = _detect_face(frame)")
        note = src.index('_face_presence_note(cam["index"], frame, face, '
                         'now_loop)')
        self.assertLess(det, note)
        self.assertLess(note, src.index("if not face:", det))


if __name__ == "__main__":
    unittest.main()
