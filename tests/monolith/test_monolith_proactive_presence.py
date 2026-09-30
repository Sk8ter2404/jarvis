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
        bc._camera_last_seen.clear()
        self.addCleanup(bc._camera_last_seen.clear)
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

    # ── the PER-CAMERA stamp (_camera_last_seen) obeys the same rule ───────
    # It feeds the face_tracker skill, and through it the wellness nudge, the
    # briefings' presence wait and morning_arrival_v2. Until 2026-09-30 the
    # loop stamped it on EVERY single-frame hit, so everything built on it
    # kept the pre-v2.0.142 false presence.
    def test_one_frame_hit_does_not_stamp_the_camera(self):
        self._run(self._fresh(1), reads=1)
        self.assertNotIn(0, self.bc._camera_last_seen,
                         "a single-frame hit stamped _camera_last_seen")

    def test_re_served_frames_do_not_stamp_the_camera(self):
        same = self.np.zeros((720, 1280, 3), dtype=self.np.uint8)
        self._run([same] * 20, reads=12)
        self.assertNotIn(0, self.bc._camera_last_seen)

    def test_relaxed_pass_hits_do_not_stamp_the_camera(self):
        self._run(self._fresh(12), reads=12, relaxed_only=True)
        self.assertNotIn(0, self.bc._camera_last_seen)

    def test_a_face_across_the_room_does_not_stamp_the_camera(self):
        self._run(self._fresh(12), reads=12, boxes=self.SMALL)
        self.assertNotIn(0, self.bc._camera_last_seen)

    def test_sustained_face_stamps_the_camera_that_saw_it(self):
        before = time.time()
        self._run(self._fresh(12), reads=12)
        self.assertGreaterEqual(self.bc._camera_last_seen.get(0, 0.0), before)
        self.assertEqual(set(self.bc._camera_last_seen), {0})


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

    @staticmethod
    def _camera_last_seen_writers(tree):
        """Functions that write an item of a ``_camera_last_seen`` dict
        (``x[i] = ..``, ``bc._camera_last_seen[i] = ..``, del, augmented, or
        an update/setdefault/pop/clear-style mutator call)."""
        def _is_cls(node):
            return ((isinstance(node, ast.Name)
                     and node.id == "_camera_last_seen")
                    or (isinstance(node, ast.Attribute)
                        and node.attr == "_camera_last_seen"))
        writers = set()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    targets = [node.target]
                elif isinstance(node, ast.Delete):
                    targets = node.targets
                if any(isinstance(t, ast.Subscript) and _is_cls(t.value)
                       for t in targets):
                    writers.add(fn.name)
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and _is_cls(node.func.value)
                        and node.func.attr in ("update", "setdefault", "pop",
                                               "popitem", "__setitem__")):
                    writers.add(fn.name)
        return writers

    def test_only_the_presence_note_stamps_camera_last_seen(self):
        # The monolith: exactly one writer, the sustained-presence verdict.
        with open(self.bc.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        self.assertEqual(self._camera_last_seen_writers(tree),
                         {"_face_presence_note"})
        # ...and no production module writes it behind the monolith's back
        # (the skills and core/actions only READ it).
        import os
        root = os.path.dirname(self.bc.__file__)
        for sub in ("core", "skills"):
            d = os.path.join(root, sub)
            for fname in sorted(os.listdir(d)):
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(d, fname)
                try:
                    with open(path, encoding="utf-8") as fh:
                        sub_tree = ast.parse(fh.read())
                except (OSError, SyntaxError, UnicodeDecodeError):
                    continue
                self.assertEqual(self._camera_last_seen_writers(sub_tree),
                                 set(), f"{sub}/{fname} writes "
                                        f"_camera_last_seen")

    def test_the_writer_scan_is_not_blind(self):
        # A ratchet that cannot see a write passes forever. Feed it the
        # pre-2026-09-30 loop shape.
        tree = ast.parse(
            "def loop():\n"
            "    with _camera_state_lock:\n"
            "        _camera_last_seen[cam['index']] = now_loop\n"
            "def skill(bc):\n"
            "    bc._camera_last_seen.update({0: 1.0})\n")
        self.assertEqual(self._camera_last_seen_writers(tree),
                         {"loop", "skill"})


class PresenceConsumersTests(_Base):
    """The chain the per-camera stamp feeds, end to end with the REAL
    monolith, face_tracker, wellness, daily_briefing and morning_arrival_v2:
    frames -> _face_presence_note -> _camera_last_seen -> face_tracker's
    poller -> the three presence readers. Frames are synthetic arrays and the
    detector's verdict is set directly (no camera, no cascade); the Kinect,
    face-ID and keyboard-idle reads are faked. A single-frame hit - the
    shape of the 04:02 "90 minutes" nudge and the 06:00 arrival briefing
    with nobody at the desk - must make NONE of the readers see the owner;
    a sustained face must make all of them see him."""

    CAM = {"index": 0, "label": "Left webcam", "name": "Test Webcam 0",
           "primary": True, "look_x": 0.5, "look_y": 0.5}

    def setUp(self):
        super().setUp()
        import sys
        import numpy as np
        from tests._skill_harness import load_skill_isolated
        bc = self.bc
        self.np = np
        self._p(bc, "CAMERAS", [dict(self.CAM)])
        self._p(bc, "MONITORS", {"left": (0, 0, 1920, 1080),
                                 "middle": (1920, 0, 1920, 1080)})
        bc._camera_last_seen.clear()
        self.addCleanup(bc._camera_last_seen.clear)
        bc.last_face_seen = 0.0
        saved = {k: sys.modules.get(k) for k in (
            "skill_face_tracker", "skill_wellness", "skill_daily_briefing",
            "skill_morning_arrival_v2")}

        def _restore_modules():
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.addCleanup(_restore_modules)
        self.ft, _ = load_skill_isolated("face_tracker")
        self.well, _ = load_skill_isolated("wellness")
        self.brief, _ = load_skill_isolated("daily_briefing")
        self.arrive, _ = load_skill_isolated("morning_arrival_v2")
        ft = self.ft
        self._p(ft, "_read_kinect_presence", return_value=None)
        self._p(ft, "_kinect_gaze_monitor", return_value=None)
        self._p(ft, "_apply_greet_new_people")
        # Keyboard / mouse and workshop mode must not stand in for a face.
        self._p(self.well, "_recent_input", return_value=False)
        self._p(self.well, "_workshop_mode_active", return_value=False)
        self.arrive._presence_first_seen_at[0] = 0.0
        self.addCleanup(self.arrive._presence_first_seen_at.__setitem__, 0,
                        0.0)

    def _frames(self, n):
        """n consecutive frames that are each NEW content (fresh)."""
        out = []
        for i in range(n):
            f = self.np.zeros((72, 128, 3), dtype=self.np.uint8)
            f[::8, ::8] = i + 1
            out.append(f)
        return out

    def _feed(self, times):
        """One qualified fresh detection per timestamp, through the real
        _face_presence_note (the face-track producer's only presence call)."""
        bc = self.bc
        confirmed = []
        for t, frame in zip(times, self._frames(len(times))):
            bc._face_detect_last[0] = {"pass": "frontal", "w_frac": 0.25,
                                       "h_frac": 0.4}
            confirmed.append(bc._face_presence_note(0, frame, (0.5, 0.5), t))
        return confirmed

    def _readers(self):
        """Poll face_tracker (twice: its hysteresis) and ask every reader."""
        for _ in range(self.ft.HYSTERESIS_SAMPLES):
            self.ft._poll_once(self.bc)
        # The wellness poller's own tick: a presence reading starts (or
        # extends) the 90-minute focus block.
        self.well._poll_once()
        self.arrive._sustained_presence_seconds()
        return {
            "face_visible": self.ft._snapshot_state()["face_visible"],
            "wellness_present": self.well._user_present(),
            "focus_block_started": self.well._block_started_at[0] > 0.0,
            "briefing_at_desk": self.brief._user_at_desk(),
            "arrival_armed": self.arrive._presence_first_seen_at[0] > 0.0,
        }

    def test_a_single_frame_hit_is_nobody_at_the_desk(self):
        self.assertEqual(self._feed([time.time()]), [False])
        self.assertNotIn(0, self.bc._camera_last_seen)
        seen = self._readers()
        self.assertFalse(seen["face_visible"], seen)
        self.assertFalse(seen["wellness_present"], seen)
        self.assertFalse(seen["focus_block_started"],
                         "a one-frame blip started a 90-minute focus block")
        self.assertIsNot(seen["briefing_at_desk"], True, seen)
        self.assertFalse(seen["arrival_armed"],
                         "a one-frame blip armed the morning arrival")

    def test_scattered_blips_are_nobody_at_the_desk(self):
        # One hit every 4 s for a minute: each would have held the old
        # 3 s FACE_FRESH window open on its own.
        now = time.time()
        self._feed([now - 60 + 4 * i for i in range(15)])
        self.assertNotIn(0, self.bc._camera_last_seen)
        seen = self._readers()
        self.assertFalse(seen["focus_block_started"], seen)
        self.assertFalse(seen["arrival_armed"], seen)

    def test_a_sustained_face_is_the_owner_at_the_desk(self):
        now = time.time()
        confirmed = self._feed([now - 2.5 + 0.5 * i for i in range(6)])
        self.assertTrue(confirmed[-1], confirmed)
        self.assertGreaterEqual(self.bc._camera_last_seen.get(0, 0.0),
                                now - 0.01)
        seen = self._readers()
        self.assertTrue(seen["face_visible"], seen)
        self.assertTrue(seen["wellness_present"], seen)
        self.assertTrue(seen["focus_block_started"], seen)
        self.assertTrue(seen["briefing_at_desk"], seen)
        self.assertTrue(seen["arrival_armed"], seen)
        self.assertEqual(self.ft._snapshot_state()["current_monitor"], "left")


if __name__ == "__main__":
    unittest.main()
