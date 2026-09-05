"""Leaving game mode must never TURN ON something that was OFF before it engaged.

THE DEFECT THESE PIN (verified live on the owner's box, 2026-09-04)
-------------------------------------------------------------------
`_lever()` recorded a restore whenever the suspend call did not RAISE. But
several of the "off" calls are no-ops that early-return when the thing is
already off:

    kinect_air_mouse.air_mouse_off()  -> "The air-mouse is already off, sir."
    guard_mode.guard_off()            -> "I wasn't on watch, sir."

while their partner "on" calls change state UNCONDITIONALLY:

    air_mouse_on() runs _set_enabled(True) BEFORE its "already on" return, and
    _set_enabled -> _persist_setting -> tools.settings_window.save_settings ->
    data/user_settings.json.

His live data/user_settings.json has KINECT_AIR_MOUSE_ENABLED = false. So the
first Fortnite exit would have switched ON a global cursor takeover that
injects synthetic LEFTDOWN/RIGHTDOWN on a closed fist, AND written that to
disk so it survived every future restart. The same evening's log shows his
hands on keyboard+mouse registering as a two-hand grab seven times in six
minutes. That is a stray click in his next match, delivered by the mode whose
whole purpose is to protect that match. Same shape for guard mode (exit ARMS a
camera monitor daemon), face tracking, diagnostics, and two idle daemons.

WHY THE ORIGINAL SUITE COULD NOT SEE IT
----------------------------------------
There, none of those skill modules are in sys.modules, so every lever takes the
`not callable(fn)` branch and logs "unavailable in this build". A lever that is
never exercised cannot mis-restore. These tests do the one thing that was
missing: they PUT the modules there, as fakes that reproduce the real
early-return / unconditional-persist asymmetry line for line.

WHAT THEY ASSERT, AND WHY IT IS PHRASED THAT WAY
-------------------------------------------------
The guarantee, not the plumbing. Which helper idles which feature is being
actively reworked (_lever, _idle_cfg_flag and _hold_face_detection all carry a
share of it), and a test written against today's call graph would pass while
the guarantee rotted. So the core assertion is a full before/after SNAPSHOT of
everything a lever can reach: enter, leave, and the world must be exactly as it
was — with nothing switched on, and nothing written to disk.

Nothing here touches a real process, camera, GPU, model, or the disk. The fake
skill modules are the only things any lever can reach. stdlib unittest only.
"""
from __future__ import annotations

import ast
import os
import sys
import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "skills", "game_mode.py")

# Every core.config flag a lever is allowed to touch. Saved and restored around
# each test so this file can never leave the process's config mutated.
_CFG_FLAGS = ("KINECT_AIR_MOUSE_ENABLED", "KINECT_GESTURES_ENABLED",
              "KINECT_TWO_HAND_ENABLED")


class _FakeThread:
    """Just enough of a thread handle for a `_thread` / `_loop_thread` probe."""

    def __init__(self, alive: bool = True):
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def _make_flag_skill(mod_name: str, flag: str, on_name: str, off_name: str,
                     flags: dict, disk: list):
    """A kinect_air_mouse / kinect_gestures stand-in.

    `flags` mirrors what the skill believes; `disk` records every persist —
    i.e. every write that would reach data/user_settings.json. The two
    behaviours that MATTER are copied exactly:

      * the OFF action early-returns with no state change and NO EXCEPTION when
        the flag is already clear   (kinect_air_mouse.py:3553-3554)
      * the ON action calls _set_enabled(True) UNCONDITIONALLY, ahead of its own
        "already on" return, and that persists  (kinect_air_mouse.py:3532-3539)
    """
    m = types.ModuleType(mod_name)

    def _cfg_flag(name, default=False):
        return bool(flags.get(name, default))

    def _set_enabled(on):
        flags[flag] = bool(on)
        disk.append((flag, bool(on)))
        return True

    def _on(_=""):
        already = _cfg_flag(flag)
        _set_enabled(True)                       # UNCONDITIONAL — the trap
        return "already on, sir." if already else "on, sir."

    def _off(_=""):
        if not _cfg_flag(flag):
            return "already off, sir."           # no-op, no exception
        _set_enabled(False)
        return "off, sir."

    m._cfg_flag = _cfg_flag
    m._set_enabled = _set_enabled
    setattr(m, on_name, _on)
    setattr(m, off_name, _off)
    return m


def _make_guard(armed: bool):
    """guard_mode stand-in: guard_off() on a disarmed guard returns a sentence
    and changes nothing; guard_on() ARMS and starts a camera monitor daemon."""
    m = types.ModuleType("skill_guard_mode")
    m._armed = [bool(armed)]
    m.daemons_started = []

    def guard_on(_=""):
        if m._armed[0]:
            return "I'm already standing watch, sir."
        m._armed[0] = True
        m.daemons_started.append("guard-mode-monitor")
        return "Standing watch, sir."

    def guard_off(_=""):
        if not m._armed[0]:
            return "I wasn't on watch, sir."     # no-op, no exception
        m._armed[0] = False
        return "Standing down, sir."

    m.guard_on, m.guard_off = guard_on, guard_off
    return m


def _make_thread_skill(mod_name: str, attr: str, start_name: str,
                       stop_name: str, running: bool):
    """ambient_multimodal_extract / standby_audio_detect stand-in."""
    m = types.ModuleType(mod_name)
    setattr(m, attr, _FakeThread() if running else None)
    m.starts = []
    m.stops = []

    def _start(_=""):
        m.starts.append(1)
        setattr(m, attr, _FakeThread())
        return "started"

    def _stop(_=""):
        m.stops.append(1)
        setattr(m, attr, None)
        return "stopped"

    setattr(m, start_name, _start)
    setattr(m, stop_name, _stop)
    return m


def _fake_bc(*, face_paused: bool, camera_off: bool):
    bc = types.ModuleType("bobert_companion")
    bc._RESOLVED_LOCAL_LLM_MODEL = ["gemma4:26b-a4b-it-qat"]
    bc.LOCAL_LLM_MODEL = "gemma4:26b-a4b-it-qat"
    bc.LOCAL_VISION_MODEL = "gemma4:26b-a4b-it-qat"
    bc.HUD_CAMERA_PREVIEW = True
    # The REAL switches are module-global threading.Events on the monolith
    # (bobert_companion.py:3784 and :3791); SET means "already suppressed".
    bc._face_track_pause = threading.Event()
    bc._face_track_camera_off = threading.Event()
    if face_paused:
        bc._face_track_pause.set()
    if camera_off:
        bc._face_track_camera_off.set()
    bc.pause_face_tracking = mock.MagicMock(
        side_effect=lambda: bc._face_track_pause.set())
    bc.resume_face_tracking = mock.MagicMock(
        side_effect=lambda: bc._face_track_pause.clear())
    bc.set_face_tracking_camera_off = mock.MagicMock(
        side_effect=lambda: bc._face_track_camera_off.set())
    bc.clear_face_tracking_camera_off = mock.MagicMock(
        side_effect=lambda: bc._face_track_camera_off.clear())
    return bc


class _Base(unittest.TestCase):
    """Builds a world in which EVERY luxury lever is reachable and reversible,
    then runs the real _suspend_luxuries / _restore_luxuries pair — the exact
    two calls _enter and _exit make against _st.applied."""

    def _world(self, *, air_mouse=False, gestures=True, guard=False,
               ambient=False, standby=False, kinect=False,
               face_paused=False, camera_off=False, diag_paused=False):
        self.flags = {"KINECT_AIR_MOUSE_ENABLED": air_mouse,
                      "KINECT_GESTURES_ENABLED": gestures}
        self.disk: list = []      # every write that would hit user_settings.json

        self.bc = _fake_bc(face_paused=face_paused, camera_off=camera_off)
        p = mock.patch.dict(sys.modules, {"bobert_companion": self.bc})
        p.start()
        self.addCleanup(p.stop)

        mod, _actions = load_skill_isolated("game_mode")
        self.mod = mod

        # The skill idles features by flipping REAL core.config constants
        # in-process (never the disk). Put every one of them back regardless of
        # how the test ends, so this file cannot leak state into another.
        import core.config as _c
        self.cfg = _c
        for k in _CFG_FLAGS:
            self.addCleanup(setattr, _c, k, getattr(_c, k, False))
        _c.KINECT_TWO_HAND_ENABLED = True
        for k, v in self.flags.items():
            setattr(_c, k, v)

        self.air = _make_flag_skill(
            "skill_kinect_air_mouse", "KINECT_AIR_MOUSE_ENABLED",
            "air_mouse_on", "air_mouse_off", self.flags, self.disk)
        self.gest = _make_flag_skill(
            "skill_kinect_gestures", "KINECT_GESTURES_ENABLED",
            "gestures_on", "gestures_off", self.flags, self.disk)
        self.guard = _make_guard(guard)
        self.ambient = _make_thread_skill(
            "skill_ambient_multimodal_extract", "_thread",
            "ambient_extract_start", "ambient_extract_stop", ambient)
        self.standby = _make_thread_skill(
            "skill_standby_audio_detect", "_loop_thread",
            "_start_background_loop", "stop_background_loop", standby)

        # audio.kinect_bridge — faked so nothing here can open the sensor. He is
        # gaming; no test in this file may touch a real device.
        kb = types.ModuleType("audio.kinect_bridge")
        kb._enabled = [bool(kinect)]
        kb.pump_running = [bool(kinect)]
        kb.get_enabled = lambda: bool(kb._enabled[0])
        kb.set_enabled = lambda on: kb._enabled.__setitem__(0, bool(on))
        kb.stop_body_pump = lambda: kb.pump_running.__setitem__(0, False)
        kb.start_body_pump = lambda: kb.pump_running.__setitem__(0, True)
        kb._pump_is_alive = lambda: bool(kb.pump_running[0])
        self.kb = kb
        fake_audio = types.ModuleType("audio")
        fake_audio.kinect_bridge = kb

        # core.diagnostic_daemons — faked so pause_diagnostics can never write
        # the real state file under data/.
        dd = types.ModuleType("core.diagnostic_daemons")
        dd.state = {"paused": bool(diag_paused)}
        dd._read_state = lambda: dict(dd.state)
        dd.pause_diagnostics = lambda: dd.state.update({"paused": True})
        dd.resume_diagnostics = lambda: dd.state.update({"paused": False})
        self.dd = dd

        fake_cv2 = types.ModuleType("cv2")     # never import real OpenCV here
        fake_cv2.threads = [32]
        fake_cv2.getNumThreads = lambda: fake_cv2.threads[0]
        fake_cv2.setNumThreads = lambda n: fake_cv2.threads.__setitem__(0, n)
        self.cv2 = fake_cv2

        import core as _core_pkg
        mods = mock.patch.dict(sys.modules, {
            "skill_kinect_air_mouse": self.air,
            "skill_kinect_gestures": self.gest,
            "skill_guard_mode": self.guard,
            "skill_ambient_multimodal_extract": self.ambient,
            "skill_standby_audio_detect": self.standby,
            "audio": fake_audio,
            "audio.kinect_bridge": kb,
            "core.diagnostic_daemons": dd,
            "cv2": fake_cv2,
        })
        mods.start()
        self.addCleanup(mods.stop)
        attr = mock.patch.object(_core_pkg, "diagnostic_daemons", dd, create=True)
        attr.start()
        self.addCleanup(attr.stop)

        self.applied: dict = {}
        self.enter_notes: list = []
        self.exit_notes: list = []
        return mod

    # ── the two calls _enter / _exit make ────────────────────────────────
    def _suspend(self):
        self.mod._suspend_luxuries(self.enter_notes, self.applied)
        return self.applied

    def _restore(self):
        self.mod._restore_luxuries(self.exit_notes, self.applied)

    def _cycle(self):
        self._suspend()
        self._restore()

    def _side_effects(self) -> dict:
        """Things that were STARTED, as opposed to state that was restored.
        Deliberately kept OUT of _snapshot: restarting a daemon that really was
        running is correct and must bump these, so folding them into the
        equality check would forbid the restore from doing its job. In the
        already-off scenarios they must all stay empty, and that is asserted
        there explicitly."""
        return {
            "guard_daemons": list(self.guard.daemons_started),
            "ambient_starts": len(self.ambient.starts),
            "standby_starts": len(self.standby.starts),
        }

    def _snapshot(self) -> dict:
        """The observable STATE every lever in _suspend_luxuries can reach. The
        identity of _face_track_pause is included deliberately: the monolith's
        readers resolve it as a module global, so handing back a DIFFERENT
        Event would strand the real one."""
        return {
            "KINECT_AIR_MOUSE_ENABLED":
                bool(getattr(self.cfg, "KINECT_AIR_MOUSE_ENABLED", False)),
            "KINECT_GESTURES_ENABLED":
                bool(getattr(self.cfg, "KINECT_GESTURES_ENABLED", False)),
            "KINECT_TWO_HAND_ENABLED":
                bool(getattr(self.cfg, "KINECT_TWO_HAND_ENABLED", False)),
            "guard_armed": bool(self.guard._armed[0]),
            "ambient_running": self.ambient._thread is not None,
            "standby_running": self.standby._loop_thread is not None,
            "kinect_enabled": self.kb.get_enabled(),
            "kinect_pump": bool(self.kb.pump_running[0]),
            "diag_paused": bool(self.dd.state["paused"]),
            "face_pause_event": self.bc._face_track_pause,
            "face_paused": self.bc._face_track_pause.is_set(),
            "camera_off": self.bc._face_track_camera_off.is_set(),
            "hud_preview": self.bc.HUD_CAMERA_PREVIEW,
            "cv2_threads": self.cv2.threads[0],
            "settings_writes": list(self.disk),
        }


# ══════════════════════════════════════════════════════════════════════════
#  THE REGRESSION — an off feature must come back off
# ══════════════════════════════════════════════════════════════════════════
class TestNeverArmsWhatWasOff(_Base):
    def test_a_full_cycle_over_a_quiet_machine_changes_nothing(self):
        """THE assertion. With every luxury already off or dormant — which is
        his real machine tonight — engaging and leaving game mode must leave the
        world bit-for-bit as it found it, and must write nothing.

        Before the fix this failed on six keys at once: the air-mouse and guard
        mode switched ON, two dormant daemons STARTED, diagnostics RESUMED, the
        camera turned back on, and KINECT_AIR_MOUSE_ENABLED written to disk."""
        self._world(air_mouse=False, gestures=False, guard=False, ambient=False,
                    standby=False, kinect=False, face_paused=True,
                    camera_off=True, diag_paused=True)
        before = self._snapshot()
        self._cycle()
        self.assertEqual(self._snapshot(), before,
                         "leaving game mode changed a machine it found quiet")
        self.assertEqual(self._side_effects(),
                         {"guard_daemons": [], "ambient_starts": 0,
                          "standby_starts": 0},
                         "leaving game mode STARTED something that was dormant")

    def test_air_mouse_off_stays_off_and_never_reaches_the_disk(self):
        """THE bug, on its own. KINECT_AIR_MOUSE_ENABLED is false in his live
        settings, so air_mouse_off() is a no-op — and the old _lever recorded
        air_mouse_on as the undo anyway. Leaving game mode then armed a
        synthetic-click injector and PERSISTED it through every restart."""
        self._world(air_mouse=False)
        self._suspend()
        self.assertFalse(getattr(self.cfg, "KINECT_AIR_MOUSE_ENABLED"),
                         "entering game mode switched the air-mouse ON")
        self._restore()
        self.assertFalse(
            getattr(self.cfg, "KINECT_AIR_MOUSE_ENABLED"),
            "leaving game mode switched the air-mouse ON — a closed fist is now "
            "a mouse click in his next match")
        self.assertFalse(self.flags["KINECT_AIR_MOUSE_ENABLED"])
        self.assertEqual(
            [w for w in self.disk if w[0] == "KINECT_AIR_MOUSE_ENABLED"], [],
            "game mode wrote KINECT_AIR_MOUSE_ENABLED to user_settings.json — "
            "that survives the power button and every future restart")

    def test_guard_mode_is_not_armed_by_leaving_a_game(self):
        """guard_off() on a disarmed guard returns 'I wasn't on watch, sir.' and
        changes nothing — so the recorded restore would ARM it and start a
        camera monitor daemon."""
        self._world(guard=False)
        self._cycle()
        self.assertNotIn("guard_mode", self.applied)
        self.assertFalse(self.guard._armed[0], "leaving game mode armed guard")
        self.assertEqual(self.guard.daemons_started, [],
                         "leaving game mode started a camera monitor daemon")

    def test_dormant_daemons_are_not_started_by_leaving_a_game(self):
        self._world(ambient=False, standby=False)
        self._cycle()
        self.assertNotIn("ambient_extract", self.applied)
        self.assertNotIn("standby_audio_detect", self.applied)
        self.assertEqual(self.ambient.starts, [], "exit started the extractor")
        self.assertEqual(self.standby.starts, [], "exit started the audio loop")

    def test_a_camera_already_off_is_not_switched_back_on(self):
        self._world(camera_off=True)
        self._cycle()
        self.assertNotIn("face_tracking_camera", self.applied)
        self.bc.clear_face_tracking_camera_off.assert_not_called()
        self.assertTrue(self.bc._face_track_camera_off.is_set())

    def test_a_pause_owned_by_the_speech_path_is_given_back_untouched(self):
        """_face_track_pause is also set while JARVIS talks. Whatever game mode
        does with it, the exit must hand back the SAME Event object in the state
        it was in at entry — never un-pause on someone else's behalf."""
        self._world(face_paused=True)
        original = self.bc._face_track_pause
        self._cycle()
        self.assertIs(self.bc._face_track_pause, original,
                      "the monolith's _face_track_pause was replaced and never "
                      "put back — its readers resolve it as a module global")
        self.assertTrue(original.is_set(),
                        "leaving game mode cleared a pause the speech path owned")
        self.bc.resume_face_tracking.assert_not_called()

    def test_paused_diagnostics_are_not_resumed(self):
        """resume_diagnostics would restart the daemons — including the
        crash-watch that would catch his next freeze — that he had paused
        himself. Game mode now declines to touch diagnostics at all (the pause
        is a disk write); either way, leaving a game must never resume them."""
        self._world(diag_paused=True)
        self._cycle()
        self.assertNotIn("diagnostics", self.applied)
        self.assertTrue(self.dd.state["paused"],
                        "leaving game mode resumed diagnostics he had paused")

    def test_a_disabled_kinect_bridge_is_not_re_enabled(self):
        self._world(kinect=False)
        self._cycle()
        self.assertFalse(self.kb.get_enabled())
        self.assertFalse(self.kb.pump_running[0],
                         "leaving game mode started the Kinect body pump")

    def test_no_lever_ever_writes_the_settings_file(self):
        """L1: any restart must restore everything with zero cleanup code. A
        persisted flag survives the power button — the documented end state of
        the exact crash this skill exists to prevent — so nothing in the
        suspend/restore pair may reach data/user_settings.json, in EITHER
        prior state."""
        for on in (False, True):
            with self.subTest(features_on=on):
                self.setUp()
                self._world(air_mouse=on, gestures=on, guard=on, ambient=on,
                            standby=on, kinect=on, diag_paused=not on)
                self._cycle()
                self.assertEqual(self.disk, [],
                                 f"wrote to user_settings.json: {self.disk}")
                self.doCleanups()


# ══════════════════════════════════════════════════════════════════════════
#  THE MODE MUST STILL DO ITS JOB — the fix cannot be "suspend nothing"
# ══════════════════════════════════════════════════════════════════════════
class TestStillSuspendsWhatWasOn(_Base):
    def test_gestures_that_were_on_are_idled_and_given_back(self):
        """KINECT_GESTURES_ENABLED is TRUE in his live settings, so this is the
        lever that must still fire — and must come back. A fix that simply
        stopped suspending things would pass every test above and protect
        nothing."""
        self._world(gestures=True)
        self._suspend()
        self.assertFalse(getattr(self.cfg, "KINECT_GESTURES_ENABLED"),
                         "gesture control was NOT idled — his hands on the "
                         "keyboard still register as gestures mid-match")
        self._restore()
        self.assertTrue(getattr(self.cfg, "KINECT_GESTURES_ENABLED"),
                        "gesture control was not given back after the game")

    def test_everything_that_was_on_is_suspended_then_restored_exactly(self):
        self._world(air_mouse=True, gestures=True, guard=True, ambient=True,
                    standby=True, kinect=True, face_paused=False,
                    camera_off=False, diag_paused=False)
        before = self._snapshot()
        self._suspend()

        # Mid-session: every one of them is actually suppressed.
        self.assertFalse(getattr(self.cfg, "KINECT_AIR_MOUSE_ENABLED"))
        self.assertFalse(getattr(self.cfg, "KINECT_GESTURES_ENABLED"))
        self.assertFalse(getattr(self.cfg, "KINECT_TWO_HAND_ENABLED"))
        self.assertFalse(self.guard._armed[0], "guard mode was not disarmed")
        self.assertIsNone(self.ambient._thread, "extractor was not stopped")
        self.assertFalse(self.kb.get_enabled(), "kinect bridge was not disabled")
        self.assertFalse(self.kb.pump_running[0], "body pump was not stopped")
        self.assertTrue(self.bc._face_track_camera_off.is_set())
        self.assertTrue(self.bc._face_track_pause.is_set())
        self.assertFalse(self.bc.HUD_CAMERA_PREVIEW)
        self.assertEqual(self.cv2.threads[0], 2)
        # Diagnostics are the deliberate exception: pause_diagnostics() is a
        # WRITE to data/diagnostic_daemons.json, which outlives the power button
        # that ends the very crash this skill exists for. They are left running
        # on purpose, so there is nothing to undo either.
        self.assertFalse(self.dd.state["paused"],
                         "game mode paused diagnostics — that pause is a disk "
                         "write and would survive a hard reset")
        self.assertNotIn("diagnostics", self.applied)
        for key in ("guard_mode", "ambient_extract", "standby_audio_detect",
                    "face_tracking_camera", "kinect_bridge"):
            self.assertIn(key, self.applied,
                          f"{key} was ON but recorded no way back")

        self._restore()
        self.assertEqual(self._snapshot(), before,
                         "the world did not come back exactly as it was")
        self.assertEqual(self.disk, [], "the restore wrote to user_settings.json")


# ══════════════════════════════════════════════════════════════════════════
#  THE RATCHET — the trap cannot come back through a NEW lever
# ══════════════════════════════════════════════════════════════════════════
class TestLeverContract(_Base):
    def test_lever_refuses_to_record_an_undo_without_a_probe(self):
        mod = self._world()
        notes, applied = [], {}
        mod._lever("mystery", lambda: None, lambda: None, notes, applied, None)
        self.assertEqual(applied, {},
                         "recorded an undo with no proof there was anything to "
                         "undo — such an undo can only be an ENABLE")
        self.assertTrue(any("SKIPPED" in n for n in notes), notes)

    def test_a_probe_that_raises_skips_the_lever_entirely(self):
        """Fail safe in BOTH directions: unsure must not touch it, and must not
        arm it on the way out."""
        mod = self._world()
        notes, applied, calls = [], {}, []

        def _boom():
            raise RuntimeError("state unreadable")

        mod._lever("mystery", lambda: calls.append("off"), lambda: None,
                   notes, applied, _boom)
        self.assertEqual(applied, {})
        self.assertEqual(calls, [],
                         "touched a lever whose prior state it could not read")
        self.assertTrue(any("SKIPPED" in n for n in notes), notes)

    def test_a_lever_that_was_off_is_not_called_and_records_nothing(self):
        mod = self._world()
        notes, applied, calls = [], {}, []
        mod._lever("mystery", lambda: calls.append("off"), lambda: None,
                   notes, applied, lambda: False)
        self.assertEqual(applied, {})
        self.assertEqual(calls, [])

    def test_a_lever_that_was_on_is_stopped_and_recorded(self):
        mod = self._world()
        notes, applied, calls = [], {}, []
        undo = lambda: None
        mod._lever("mystery", lambda: calls.append("off"), undo,
                   notes, applied, lambda: True)
        self.assertIs(applied.get("mystery"), undo)
        self.assertEqual(calls, ["off"])
        self.assertIn("mystery: stopped", notes)

    def test_an_absent_lever_is_still_reported_as_unavailable(self):
        """The pre-existing honesty contract must survive the new parameter."""
        mod = self._world()
        notes, applied = [], {}
        mod._lever("gone", None, lambda: None, notes, applied, lambda: True)
        self.assertEqual(applied, {})
        self.assertIn("gone: unavailable in this build", notes)

    def test_every_lever_call_site_passes_a_state_probe(self):
        """Source-level ratchet. The defect was call sites missing this
        argument; a fix that only patched the two noisy ones would rot exactly
        the way this repo's #1 bug class does. Any NEW _lever call added without
        a probe fails here."""
        with open(_SRC, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        sites = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_lever"]
        self.assertGreaterEqual(len(sites), 1, "no _lever call sites found")
        missing = [n.lineno for n in sites
                   if len(n.args) < 6
                   and not any(k.arg == "was_on" for k in n.keywords)]
        self.assertEqual(missing, [],
                         f"_lever call(s) with no was_on probe at line(s) "
                         f"{missing} — their restore can TURN SOMETHING ON")

    def test_the_probe_parameter_is_required(self):
        """No default. A default would silently restore the old behaviour for
        every future lever, which is how this defect shipped."""
        with open(_SRC, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_lever")
        self.assertIn("was_on", [a.arg for a in fn.args.args])
        self.assertEqual(
            len(fn.args.defaults), 0,
            "_lever grew a default argument — was_on must stay mandatory")

    def test_enter_and_exit_share_one_applied_dict(self):
        """These tests drive _suspend_luxuries/_restore_luxuries directly, so
        pin that those really are the calls _enter and _exit make, against the
        same dict. Otherwise this whole file could pass while the live path
        used something else."""
        with open(_SRC, "r", encoding="utf-8") as f:
            body = f.read()
        self.assertIn("_suspend_luxuries(notes, _st.applied)", body)
        self.assertIn("_restore_luxuries(notes, _st.applied)", body)


if __name__ == "__main__":
    unittest.main()
