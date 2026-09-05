"""L1 — "NOTHING PERSISTS" — pinned by BEHAVIOUR, not by parsing one file.

WHY THIS FILE EXISTS
────────────────────
skills/game_mode.py's header makes L1 the layer "that needs no code to be
correct":

    L1 NOTHING PERSISTS. Every change is a process-memory cell. This module
       writes NO files at all. ... ANY restart — clean exit, crash, or the
       power button — restores everything with zero cleanup code.

tests/skills/test_game_mode.py::test_a_crashed_jarvis_needs_no_cleanup_code
tried to pin that by AST-parsing game_mode.py for `open`/`write_text`/`mkdir`.
It passed while L1 was FALSE, because a module can write a file without
containing a single write call — it only has to CALL SOMETHING THAT DOES.
That is this repo's #1 documented defect (a guard that reports success it
never verified) wearing a test's clothing.

THE DEFECT IT MISSED (found 2026-09-04, read-only, on the live tree)
    _suspend_luxuries' first levers called skills/kinect_gestures.gestures_off
    and skills/kinect_air_mouse.air_mouse_off. Both are documented
    "live + persisted": each routes through that skill's `_persist_setting`
    into tools.settings_window.save_settings, which rewrites
    data/user_settings.json. The diagnostics lever called
    core.diagnostic_daemons.pause_diagnostics, which writes
    data/diagnostic_daemons.json.

    His live data/user_settings.json said KINECT_GESTURES_ENABLED = true. So
    engaging game mode wrote `false` there. Fortnite then hits the documented
    lobby-leak wall (E_OUTOFMEMORY -> RenderThread hang) and he holds the power
    button — which is the MODAL outcome of this feature's own scenario, not an
    edge case. _restore_luxuries never runs; core/config.py:_apply_user_settings()
    reads that `false` back at the next import; KINECT_GESTURES_ENABLED ships
    False so no default undoes it; game_mode's _st is empty in the fresh
    process so nothing restores it. Gesture control is dead, permanently and
    silently, until he independently thinks to say "gestures on". Same for the
    air-mouse, and diagnostic_daemons.json's paused:true keeps the self-diag,
    crash-watch and deep-audit daemons paused across every future restart —
    including the crash-watch that would have caught the next freeze.

WHAT THESE TESTS DO DIFFERENTLY
    They stand a real-shaped double in front of every persisting API game mode
    can reach, point the settings writer at a TEMP file, run the real
    _suspend_luxuries, and then assert on the FILESYSTEM. "No bytes changed on
    disk" is the same statement L1 makes, measured the same way twice.
    test_the_doubles_are_not_strawmen then proves, from the real source, that
    those APIs genuinely do persist — so the doubles cannot quietly become a
    test of nothing.

Nothing here touches a real process, GPU, model, camera, Kinect or the real
data/ directory. stdlib unittest + unittest.mock only (pytest is not installed).
SIBLING: tests/skills/test_game_mode_l1_kinect_levers.py pins the SAME L1
claim from the other side — by recording which persisting actions get
called, and it carries the write-up of the original defect (the Kinect
levers persisted through their spoken actions). If you change L1
behaviour, change BOTH.

"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BIG = "gemma4:26b-a4b-it-qat"
SMALL = "gemma4:12b"

# Every module-level name game_mode can reach that COMMITS STATE TO DISK.
# A call to any of these from a suspend lever breaks L1.
PERSISTING_APIS = (
    "kinect_gestures.gestures_off",
    "kinect_gestures.gestures_on",
    "kinect_air_mouse.air_mouse_off",
    "kinect_air_mouse.air_mouse_on",
    "diagnostic_daemons.pause_diagnostics",
    "diagnostic_daemons.resume_diagnostics",
)


class _Recorder:
    """Collects every persisting call the doubles receive, and — because a
    mock-only assertion is itself an unverified claim — actually writes the
    file the real API would write, so the test can assert on real bytes."""

    def __init__(self, settings_path: str, daemon_path: str):
        self.calls: list[str] = []
        self.settings_path = settings_path
        self.daemon_path = daemon_path

    def _persist_setting(self, key, value):
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                cur = json.load(f)
        except Exception:
            cur = {}
        cur[key] = value
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2)

    def _write_daemon_state(self, paused):
        try:
            with open(self.daemon_path, "r", encoding="utf-8") as f:
                cur = json.load(f)
        except Exception:
            cur = {}
        cur["paused"] = paused
        with open(self.daemon_path, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2)


def _fake_bc():
    """A stand-in monolith exposing exactly the cells game_mode repoints."""
    bc = types.ModuleType("bobert_companion")
    bc._RESOLVED_LOCAL_LLM_MODEL = [BIG]
    bc.LOCAL_LLM_MODEL = BIG
    bc.LOCAL_VISION_MODEL = BIG
    bc.HUD_CAMERA_PREVIEW = True
    bc.proactive_announce = mock.MagicMock(return_value=True)
    bc.pause_face_tracking = mock.MagicMock()
    bc.resume_face_tracking = mock.MagicMock()
    bc.set_face_tracking_camera_off = mock.MagicMock()
    bc.clear_face_tracking_camera_off = mock.MagicMock()
    return bc


class _Base(unittest.TestCase):
    """Loads game_mode with:
      * a fake monolith,
      * REAL core.config (the thing the in-process levers are supposed to flip)
        with the two kinect gates snapshotted and restored,
      * doubles for every persisting skill/module, faithful to the real ones:
        the gate predicate is `_cfg_flag` reading live core.config, the `*_off`
        action early-returns when already off, and the `*_on` action persists
        UNCONDITIONALLY — all three exactly as the real modules do,
      * fakes for cv2 and audio.kinect_bridge so no camera, Kinect or ~300 MB
        OpenCV import happens in a test (he is gaming; this suite allocates
        nothing).
    """

    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="jarvis-gamemode-l1-")
        self.addCleanup(self._rmtree, tmp)
        self.settings_path = os.path.join(tmp, "user_settings.json")
        self.daemon_path = os.path.join(tmp, "diagnostic_daemons.json")
        # Seed both files with the owner's LIVE values, read read-only from his
        # box on 2026-09-04: gestures ON (so the lever has something to do and
        # a `false` write would be a real, lasting loss), air-mouse OFF.
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump({"KINECT_GESTURES_ENABLED": True,
                       "KINECT_AIR_MOUSE_ENABLED": True}, f, indent=2)
        with open(self.daemon_path, "w", encoding="utf-8") as f:
            json.dump({"paused": False}, f, indent=2)
        self.rec = _Recorder(self.settings_path, self.daemon_path)

        os.environ["JARVIS_SETTINGS_PATH"] = self.settings_path
        self.addCleanup(os.environ.pop, "JARVIS_SETTINGS_PATH", None)

    @staticmethod
    def _rmtree(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    # ── the doubles ──────────────────────────────────────────────────────
    def _kinect_double(self, mod_name, off_name, on_name, flag):
        """Mirrors skills/kinect_gestures.py and skills/kinect_air_mouse.py:
        `_cfg_flag` reads live core.config; `*_off` early-returns when the flag
        is already clear; both actions persist through _persist_setting."""
        rec, cfg = self.rec, self._cfg_module()
        mod = types.ModuleType(mod_name)

        def _cfg_flag(name, default=False):
            return bool(getattr(cfg, name, default))

        def _off(_=""):
            if not _cfg_flag(flag):
                return "already off, sir."
            setattr(cfg, flag, False)
            rec.calls.append(f"{mod_name}.{off_name}")
            rec._persist_setting(flag, False)      # the real skill's path
            return "off, sir."

        def _on(_=""):
            setattr(cfg, flag, True)
            rec.calls.append(f"{mod_name}.{on_name}")
            rec._persist_setting(flag, True)       # persists UNCONDITIONALLY
            return "on, sir."

        mod._cfg_flag = _cfg_flag
        setattr(mod, off_name, _off)
        setattr(mod, on_name, _on)
        return mod

    def _daemons_double(self):
        """Mirrors core/diagnostic_daemons.py. `_read_state` MUST be present:
        game_mode probes prior state through it, and a probe that raises makes
        _lever SKIP the lever — which is how the first draft of this file gave
        a false pass on the diagnostics half while the write was still there.
        A double that is missing the API under test is not a test."""
        rec, path = self.rec, self.daemon_path
        mod = types.ModuleType("core.diagnostic_daemons")

        def _read_state():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"paused": False}

        def _pause():
            rec.calls.append("diagnostic_daemons.pause_diagnostics")
            rec._write_daemon_state(True)
            return "Diagnostics paused, sir."

        def _resume():
            rec.calls.append("diagnostic_daemons.resume_diagnostics")
            rec._write_daemon_state(False)
            return "Diagnostics resumed, sir."

        mod._read_state = _read_state
        mod.diagnostic_daemon_status = lambda: _read_state()
        mod.pause_diagnostics = _pause
        mod.resume_diagnostics = _resume
        return mod

    @staticmethod
    def _cfg_module():
        import core.config as _c
        return _c

    def _load(self):
        cfg = self._cfg_module()
        # Snapshot the REAL config gates and put them back, whatever happens.
        for flag in ("KINECT_GESTURES_ENABLED", "KINECT_AIR_MOUSE_ENABLED",
                     "KINECT_TWO_HAND_ENABLED", "LOCAL_LLM_MODEL"):
            self.addCleanup(setattr, cfg, flag, getattr(cfg, flag, None))
        # Both gates ON, so both levers have real work to do. (Anything the
        # lever skips because it was already off cannot prove anything.)
        cfg.KINECT_GESTURES_ENABLED = True
        cfg.KINECT_AIR_MOUSE_ENABLED = True

        gestures = self._kinect_double(
            "skill_kinect_gestures", "gestures_off", "gestures_on",
            "KINECT_GESTURES_ENABLED")
        air = self._kinect_double(
            "skill_kinect_air_mouse", "air_mouse_off", "air_mouse_on",
            "KINECT_AIR_MOUSE_ENABLED")
        daemons = self._daemons_double()

        # cv2 / audio.kinect_bridge: fakes, so the suite opens no camera, no
        # Kinect, and never pulls OpenCV into this process.
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.getNumThreads = lambda: 32
        fake_cv2.setNumThreads = lambda n: None
        fake_audio = types.ModuleType("audio")
        fake_kb = types.ModuleType("audio.kinect_bridge")
        fake_kb.get_enabled = lambda: False
        fake_kb.set_enabled = lambda v: None
        fake_kb.stop_body_pump = lambda: None
        fake_kb.start_body_pump = lambda: None
        fake_audio.kinect_bridge = fake_kb

        self.bc = _fake_bc()
        patch = mock.patch.dict(sys.modules, {
            "bobert_companion": self.bc,
            "skill_kinect_gestures": gestures,
            "skill_kinect_air_mouse": air,
            "core.diagnostic_daemons": daemons,
            "cv2": fake_cv2,
            "audio": fake_audio,
            "audio.kinect_bridge": fake_kb,
        })
        patch.start()
        self.addCleanup(patch.stop)

        # `from core import diagnostic_daemons` resolves the ATTRIBUTE on the
        # already-imported package first, so patching sys.modules alone would
        # not redirect it. Set both, restore both.
        import core as _core
        had = hasattr(_core, "diagnostic_daemons")
        prev = getattr(_core, "diagnostic_daemons", None)
        _core.diagnostic_daemons = daemons

        def _put_back():
            if had:
                _core.diagnostic_daemons = prev
            else:
                try:
                    delattr(_core, "diagnostic_daemons")
                except Exception:
                    pass
        self.addCleanup(_put_back)

        mod, actions = load_skill_isolated("game_mode")
        return mod

    # ── evidence helpers ─────────────────────────────────────────────────
    def _disk(self):
        """The exact bytes of both state files — the only honest measure of
        'nothing persisted'."""
        out = {}
        for p in (self.settings_path, self.daemon_path):
            with open(p, "rb") as f:
                out[os.path.basename(p)] = f.read()
        return out


# ══════════════════════════════════════════════════════════════════════════
#  L1, MEASURED
# ══════════════════════════════════════════════════════════════════════════
class TestSuspendPersistsNothing(_Base):

    def test_engaging_game_mode_changes_no_bytes_on_disk(self):
        """THE REGRESSION TEST. Run the real suspend path with every persisting
        API reachable and armed, then diff the files. A single differing byte
        is a setting that outlives the power button with nothing to undo it."""
        mod = self._load()
        before = self._disk()
        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        after = self._disk()
        self.assertEqual(
            before, after,
            "game mode wrote to disk during suspend — L1 is false and a crash "
            f"strands the setting. calls={self.rec.calls} notes={notes}")

    def test_it_calls_no_persisting_action(self):
        """The same claim from the other side: name every API that commits
        state and assert none of them was reached."""
        mod = self._load()
        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        self.assertEqual(
            self.rec.calls, [],
            "game_mode called a documented 'live + persisted' action. Each of "
            f"{PERSISTING_APIS} rewrites a file under data/. calls="
            f"{self.rec.calls}")

    def test_a_power_button_exit_leaves_nothing_behind(self):
        """The modal outcome of this feature's own scenario: enter, then the
        process dies with no restore. Simulated by never calling
        _restore_luxuries. Disk must be exactly as it was found."""
        mod = self._load()
        before = self._disk()
        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        del applied                      # the power button: no restore, ever
        self.assertEqual(before, self._disk())
        # And the fresh-process view — what core/config.py would read back.
        with open(self.settings_path, "r", encoding="utf-8") as f:
            reloaded = json.load(f)
        self.assertIs(reloaded.get("KINECT_GESTURES_ENABLED"), True,
                      "a crash left gesture control disabled in his saved "
                      "settings; nothing in the codebase ever turns it back on")
        with open(self.daemon_path, "r", encoding="utf-8") as f:
            self.assertIs(json.load(f).get("paused"), False,
                          "a crash left the diagnostic daemons paused forever "
                          "— including the crash-watch that would catch the "
                          "next freeze")

    def test_the_suppression_still_actually_happens(self):
        """The fix must not be 'delete the lever'. The live gates both pollers
        read fresh every tick (kinect_gestures._gestures_enabled,
        kinect_air_mouse._air_mouse_enabled -> _cfg_flag -> core.config) must
        be OFF once game mode is engaged, or the stray mid-match click this
        lever exists to prevent is back."""
        mod = self._load()
        cfg = self._cfg_module()
        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        self.assertFalse(getattr(cfg, "KINECT_GESTURES_ENABLED"),
                         f"gestures still armed mid-match. notes={notes}")
        self.assertFalse(getattr(cfg, "KINECT_AIR_MOUSE_ENABLED"),
                         f"air-mouse still armed mid-match. notes={notes}")

    def test_the_kinect_levers_are_reached_not_skipped(self):
        """A lever that never ran also writes nothing — for the happiest of
        wrong reasons. The two gates whose suppression is a CORRECTNESS matter
        (a stray grab-click mid-match) must show as applied, not skipped or
        unavailable, or the tests above are proving nothing about them."""
        mod = self._load()
        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        for lever in ("kinect_gestures", "kinect_air_mouse"):
            hit = [n for n in notes if n.startswith(lever + ":")]
            self.assertTrue(hit, f"{lever} lever never ran: {notes}")
            for bad in ("unavailable", "SKIPPED", "FAILED", "already off"):
                self.assertNotIn(bad, hit[0],
                                 f"{lever} did not actually engage: {hit[0]}")
            self.assertIn(lever, applied,
                          f"{lever} recorded no restore: {notes}")

    def test_the_harness_would_catch_a_write(self):
        """Teeth check. Every assertion above is 'nothing changed', which is
        also what a broken harness reports. Call the doubles directly and prove
        that a real persisting call DOES move both needles — so a regression
        cannot hide behind a double that quietly stopped working."""
        self._load()
        before = self._disk()
        sys.modules["skill_kinect_gestures"].gestures_off("")
        sys.modules["skill_kinect_air_mouse"].air_mouse_off("")
        sys.modules["core.diagnostic_daemons"].pause_diagnostics()
        self.assertEqual(len(self.rec.calls), 3, self.rec.calls)
        self.assertNotEqual(before, self._disk(),
                            "the doubles do not write — every 'nothing "
                            "persisted' assertion in this file is vacuous")

    def test_restore_returns_the_owners_value_not_true(self):
        """A clean exit must put back what HE had, not a hardcoded True. The
        real *_on actions enable unconditionally, so a restore built from them
        turns on a cursor-takeover that was off — and persists it."""
        mod = self._load()
        cfg = self._cfg_module()
        cfg.KINECT_AIR_MOUSE_ENABLED = False      # his live value tonight
        before = self._disk()
        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        mod._restore_luxuries(notes, applied)
        self.assertIs(getattr(cfg, "KINECT_GESTURES_ENABLED"), True)
        self.assertFalse(getattr(cfg, "KINECT_AIR_MOUSE_ENABLED"),
                         "leaving game mode ENABLED the air-mouse he had off")
        self.assertEqual(before, self._disk(),
                         f"the restore path wrote to disk. calls={self.rec.calls}")


# ══════════════════════════════════════════════════════════════════════════
#  THE DOUBLES ARE NOT STRAWMEN
# ══════════════════════════════════════════════════════════════════════════
class TestPersistenceContract(unittest.TestCase):
    """The tests above are only meaningful if the real APIs really do write to
    disk. Prove it from the real source — by AST, so nothing is imported and no
    device is opened. If one of these chains is ever broken, this fails and
    tells us the guard above has gone slack, instead of it silently passing."""

    @staticmethod
    def _calls_in(path, funcname):
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == funcname:
                return {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
                        for c in ast.walk(node) if isinstance(c, ast.Call)}
        raise AssertionError(f"{funcname} no longer exists in {path}")

    def test_gestures_off_reaches_the_settings_writer(self):
        p = os.path.join(_ROOT, "skills", "kinect_gestures.py")
        self.assertIn("_set_gestures_enabled", self._calls_in(p, "gestures_off"))
        self.assertIn("_persist_setting", self._calls_in(p, "_set_gestures_enabled"))
        self.assertIn("save_settings", self._calls_in(p, "_persist_setting"))

    def test_air_mouse_off_reaches_the_settings_writer(self):
        p = os.path.join(_ROOT, "skills", "kinect_air_mouse.py")
        self.assertIn("_set_enabled", self._calls_in(p, "air_mouse_off"))
        self.assertIn("_persist_setting", self._calls_in(p, "_set_enabled"))
        self.assertIn("save_settings", self._calls_in(p, "_persist_setting"))

    def test_pause_diagnostics_reaches_the_state_writer(self):
        p = os.path.join(_ROOT, "core", "diagnostic_daemons.py")
        self.assertIn("_update_state", self._calls_in(p, "pause_diagnostics"))
        self.assertIn("_write_state", self._calls_in(p, "_update_state"))
        self.assertIn("_atomic_write_json", self._calls_in(p, "_write_state"))

    def test_game_mode_calls_no_persisting_api_by_name(self):
        """Belt to the behavioural braces. The original AST guard looked for
        `open`/`write_text`/`mkdir` — the syntax of writing a file — and so
        could not see a module that writes by CALLING SOMETHING THAT WRITES.
        This one looks for the callee names instead, which is the shape the
        defect actually had. Comments are invisible to the AST, so the long
        explanations in game_mode.py naming these functions do not trip it."""
        p = os.path.join(_ROOT, "skills", "game_mode.py")
        with open(p, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        banned = {"gestures_off", "gestures_on", "air_mouse_off", "air_mouse_on",
                  "pause_diagnostics", "resume_diagnostics", "save_settings",
                  "_persist_setting", "set_model"}
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                nm = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if nm in banned:
                    hits.append(f"{nm}() at line {node.lineno}")
        self.assertEqual(hits, [],
                         "game_mode calls an API that commits state to disk — "
                         "L1 says it writes NO files at all: " + "; ".join(hits))

    def test_the_daemon_loops_read_paused_from_disk(self):
        """Why the diagnostics lever cannot be made process-memory-only: all
        four loops re-read the FILE each tick, so pausing them is inherently a
        disk write that outlives the process."""
        p = os.path.join(_ROOT, "core", "diagnostic_daemons.py")
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertGreaterEqual(
            src.count('if state.get("paused")'), 4,
            "the four daemon loops no longer gate on the persisted flag — "
            "re-check whether game_mode may pause them")


if __name__ == "__main__":     # pragma: no cover
    unittest.main(verbosity=2)
