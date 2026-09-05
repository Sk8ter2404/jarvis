"""skills/game_mode.py must persist NOTHING — the write that survives the power button.

WHY THIS FILE EXISTS (the defect it was written against, 2026-09-04)

game_mode's docstring sells L1 as the layer that needs no code to be correct:

    "L1 NOTHING PERSISTS. Every change is a process-memory cell. This module
     writes NO files at all. ... so ANY restart — clean exit, crash, or the
     power button — restores everything with zero cleanup code."

That was FALSE. `_suspend_luxuries` drove the two Kinect levers through the
skills' own spoken actions, and those actions persist:

    gestures_off("")  -> kinect_gestures._set_gestures_enabled(False)
                      -> _persist_setting  (kinect_gestures.py:450-478)
                      -> tools.settings_window.save_settings
                      -> ATOMIC REWRITE of data/user_settings.json
    air_mouse_off("") -> kinect_air_mouse._set_enabled(False)
                      -> the same chain      (kinect_air_mouse.py:3463-3490)

Now run the exact scenario the feature exists for. Fortnite's lobby leak
reaches E_OUTOFMEMORY, the RenderThread hangs, he holds the power button (it has
already happened once on this box). `_restore_luxuries` never runs. At the next
boot `core/config.py:_apply_user_settings()` reads the persisted `false`, and
because KINECT_GESTURES_ENABLED *ships* False there is no default to put it
back — gesture control is dead, silently and permanently, with nothing to tell
him why. Measured that evening, read-only, from his live settings file:
KINECT_GESTURES_ENABLED = true. It was one Fortnite session away from false.

WHY THE EXISTING PIN DID NOT CATCH IT
tests/skills/test_game_mode.py:test_a_crashed_jarvis_needs_no_cleanup_code
ast.walk()s ONLY skills/game_mode.py looking for literal open/write_text/
makedirs/mkdir calls. The write happened inside a DIFFERENT module reached
through `getattr(mod, "gestures_off")("")`, so that test is structurally blind
to it and reported the file clean. A source-shaped test cannot see a write made
through a called skill.

SO THESE TESTS ARE BEHAVIOURAL, AND THEY USE THE REAL WRITER.
The real skills/kinect_gestures.py is loaded and left in sys.modules exactly
where game_mode looks for it, and JARVIS_SETTINGS_PATH redirects
tools.settings_window at a throwaway file (it resolves the path at CALL time, so
the redirect works even though the module was imported earlier). Then we assert
the file is BYTE-IDENTICAL afterwards. That catches the reported defect and any
future lever that starts persisting — dynamically dispatched or not — which is
the blind spot the AST pin left open.

MEMORY DISCIPLINE: he is gaming while these run, and the box was measured at
575 MB free / 98.8 % used. cv2 is stubbed (the real import is 100+ MB of RSS)
and every Ollama/nvidia-smi primitive is replaced, so nothing here allocates a
model, touches the GPU, or reaches the live Ollama server.

stdlib unittest + unittest.mock only (pytest is not installed).
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
GAME = "fortniteclient-win64-shipping.exe"
BIG = "gemma4:26b-a4b-it-qat"
SMALL = "gemma4:12b"
LIVE_SIZES_MB = {BIG: 15360, SMALL: 7782}


def _fake_cv2():
    """Stand-in for cv2. game_mode's cv2_threads lever does `import cv2`; the
    real one costs 100+ MB of RSS, and a suite whose entire subject is keeping
    RAM free for his game must not be the thing that allocates it."""
    m = types.ModuleType("cv2")
    m._n = [8]
    m.getNumThreads = lambda: m._n[0]
    m.setNumThreads = lambda n: m._n.__setitem__(0, n)
    return m


def _fake_bc():
    """A stand-in monolith exposing the cells game_mode repoints, plus the
    GAME_MODE_* settings (_cfg reads the monolith first, so pinning them here
    exercises the real lookup path)."""
    bc = types.ModuleType("bobert_companion")
    bc._RESOLVED_LOCAL_LLM_MODEL = [BIG]
    bc.LOCAL_LLM_MODEL = BIG
    bc.LOCAL_VISION_MODEL = BIG
    bc.HUD_CAMERA_PREVIEW = True
    bc.proactive_announce = mock.MagicMock(return_value=True)
    for k, v in {
        "GAME_MODE_ENABLED": False,
        "GAME_MODE_PROCESS_HINTS": [GAME],
        "GAME_MODE_BRAIN": SMALL,
        "GAME_MODE_VERIFY_DELAY_SECONDS": 0.0,
        "GAME_MODE_MIN_VRAM_DELTA_MB": 3000,
        "GAME_MODE_ANNOUNCE": False,
    }.items():
        setattr(bc, k, v)
    return bc


class _Base(unittest.TestCase):
    """Redirects the settings writer at a temp file, pins a fake monolith, and
    stubs every outward-facing primitive."""

    def _temp_settings(self, **seed):
        """A throwaway user_settings.json, seeded to match his live values."""
        fd, path = tempfile.mkstemp(prefix="jarvis_settings_", suffix=".json")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(seed, f)
        self.addCleanup(
            lambda: os.path.exists(path) and os.remove(path))
        # settings_path() resolves this at CALL time, so the redirect applies
        # even though tools.settings_window may already be imported.
        env = mock.patch.dict(os.environ, {"JARVIS_SETTINGS_PATH": path})
        env.start()
        self.addCleanup(env.stop)
        self.settings_path = path
        return path

    def _raw(self):
        with open(self.settings_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_real_kinect_gestures(self):
        """The REAL skill, in the exact sys.modules slot game_mode looks up.
        Its top-level imports are stdlib only, and the harness neuters
        Thread.start, so register() spawns no poller."""
        mod, _ = load_skill_isolated("kinect_gestures")
        self.addCleanup(lambda: sys.modules.pop("skill_kinect_gestures", None))
        return mod

    def _load_game_mode(self):
        self.bc = _fake_bc()
        imp = mock.patch.dict(sys.modules, {"bobert_companion": self.bc})
        imp.start()
        self.addCleanup(imp.stop)

        cv2p = mock.patch.dict(sys.modules, {"cv2": _fake_cv2()})
        cv2p.start()
        self.addCleanup(cv2p.stop)

        mod, actions = load_skill_isolated("game_mode")
        self.addCleanup(lambda: sys.modules.pop("skill_game_mode", None))

        # Outward-facing primitives — never the live GPU or Ollama server.
        self.samples = [(23269, 1307), (8100, 16476)]
        mod._nvidia_smi_mb = lambda: (self.samples[0] if len(self.samples) == 1
                                      else self.samples.pop(0))
        mod._free_ram_mb = lambda: 9000
        mod._installed_sizes_mb = lambda: dict(LIVE_SIZES_MB)
        mod._resident_models = lambda: [{"name": BIG, "size_vram_mb": 14524}]
        self.unloaded = []
        mod._unload = lambda tag: (self.unloaded.append(tag), True)[1]
        mod._pid_alive = lambda pid: False
        self.mod, self.actions = mod, actions
        return mod, actions

    def _cfg_flag(self, name, value):
        """Pin a flag on the LIVE core.config — which is what both the kinect
        pollers and game_mode read — and restore it afterwards."""
        import core.config as cfg
        p = mock.patch.object(cfg, name, value, create=True)
        p.start()
        self.addCleanup(p.stop)
        return cfg


# ══════════════════════════════════════════════════════════════════════════
#  THE DEFECT: a write that outlives the power button
# ══════════════════════════════════════════════════════════════════════════
class TestNothingReachesDisk(_Base):

    def test_suspending_never_persists_gestures_off(self):
        """THE REGRESSION TEST. Enter the suspend path with the REAL
        kinect_gestures module in place and his real value (true) on disk, then
        let the power button fall — no restore. The file must be untouched, so
        the next boot still reads true and his gesture control still works."""
        self._temp_settings(KINECT_GESTURES_ENABLED=True)
        before = self._raw()
        self._load_real_kinect_gestures()
        cfg = self._cfg_flag("KINECT_GESTURES_ENABLED", True)
        mod, _ = self._load_game_mode()

        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)

        # The lever must still WORK — the live poller is idled...
        self.assertFalse(
            cfg.KINECT_GESTURES_ENABLED,
            "game mode did not idle gesture control — a stray grab-click "
            f"can still cost him a match. notes={notes}")
        # ...and it did it WITHOUT touching the disk. Simulate the hard reset by
        # simply never calling _restore_luxuries.
        self.assertEqual(
            self._raw(), before,
            "game_mode wrote to user_settings.json. That write survives the "
            "power button, and _restore_luxuries never runs on that path — so "
            "the next boot reads the persisted value and gesture control is "
            "off for good, silently.")
        self.assertTrue(
            json.loads(self._raw())["KINECT_GESTURES_ENABLED"],
            "his persisted gesture setting was flipped to false")

    def test_full_enter_writes_no_bytes_at_all(self):
        """Generalised: the WHOLE enter path — every lever, not just the two
        Kinect ones — must leave the settings file byte-identical. This is the
        blind spot the AST pin left open, because it catches a persisting lever
        reached through ANY dynamic dispatch, not just a literal write call."""
        self._temp_settings(KINECT_GESTURES_ENABLED=True,
                            LOCAL_LLM_MODEL=BIG)
        before = self._raw()
        self._load_real_kinect_gestures()
        self._cfg_flag("KINECT_GESTURES_ENABLED", True)
        mod, _ = self._load_game_mode()

        out = mod._enter(47036, GAME)

        self.assertTrue(mod._st.active, f"game mode did not engage: {out}")
        self.assertEqual(self.unloaded, [BIG], "the big brain was not unloaded")
        self.assertEqual(
            self._raw(), before,
            f"_enter() persisted something. notes={mod._st.notes}")

    def test_restore_puts_the_live_flag_back_without_persisting(self):
        """The CLEAN exit must also stay off disk: the restore is an in-process
        flip, not gestures_on() (which persists unconditionally, before its own
        'already on' early return)."""
        self._temp_settings(KINECT_GESTURES_ENABLED=True)
        before = self._raw()
        self._load_real_kinect_gestures()
        cfg = self._cfg_flag("KINECT_GESTURES_ENABLED", True)
        mod, _ = self._load_game_mode()

        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        self.assertFalse(cfg.KINECT_GESTURES_ENABLED)

        mod._restore_luxuries(notes, applied)
        self.assertTrue(
            cfg.KINECT_GESTURES_ENABLED,
            f"gesture control was not restored on a clean exit. notes={notes}")
        self.assertEqual(self._raw(), before,
                         "the restore path wrote to user_settings.json")

    def test_a_feature_he_left_off_is_left_alone_and_never_switched_on(self):
        """He had KINECT_AIR_MOUSE_ENABLED = false. Game mode must not record a
        restore for something it never stopped, or the exit 'restores' a feature
        he deliberately switched OFF — turning on the closed-fist click that has
        closed his Chrome tabs before."""
        self._temp_settings(KINECT_AIR_MOUSE_ENABLED=False)
        cfg = self._cfg_flag("KINECT_AIR_MOUSE_ENABLED", False)
        mod, _ = self._load_game_mode()

        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        self.assertNotIn("kinect_air_mouse", applied,
                         "recorded a restore for a lever that stopped nothing")

        mod._restore_luxuries(notes, applied)
        self.assertFalse(
            cfg.KINECT_AIR_MOUSE_ENABLED,
            "game mode switched the air-mouse ON — he had it off")


# ══════════════════════════════════════════════════════════════════════════
#  THE RATCHET — and a check that the hazard it guards is still real
# ══════════════════════════════════════════════════════════════════════════
class TestPersistingActionsAreNotCalled(_Base):

    def _src(self, rel):
        with open(os.path.join(_ROOT, rel), "r", encoding="utf-8") as f:
            return f.read()

    def test_the_kinect_off_actions_really_do_persist(self):
        """Verify the PREMISE rather than trusting my memory of it: prove by AST
        that gestures_off still reaches _persist_setting. If someone makes those
        actions non-persisting, this fails and tells the next reader that the
        ratchet below may be relaxed — instead of leaving a rule nobody can
        re-derive."""
        tree = ast.parse(self._src(os.path.join("skills", "kinect_gestures.py")))
        funcs = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)}

        def calls(name):
            return {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
                    for c in ast.walk(funcs[name]) if isinstance(c, ast.Call)}

        self.assertIn("_set_gestures_enabled", calls("gestures_off"))
        self.assertIn("_persist_setting", calls("_set_gestures_enabled"))
        self.assertIn("save_settings", calls("_persist_setting"))

    def test_game_mode_never_calls_a_persisting_skill_action(self):
        """Ratchet against reintroduction in the literal form. The behavioural
        tests above cover the dynamic form (`getattr(mod, name)("")`), which is
        how the defect actually shipped — this one just makes the direct call
        impossible to add without a red test."""
        tree = ast.parse(self._src(os.path.join("skills", "game_mode.py")))
        forbidden = {"gestures_on", "gestures_off",
                     "air_mouse_on", "air_mouse_off",
                     "_persist_setting", "save_settings", "set_model"}
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (getattr(node.func, "id", None)
                        or getattr(node.func, "attr", None))
                if name in forbidden:
                    called.add(name)
        self.assertEqual(
            called, set(),
            f"game_mode calls persisting action(s) {sorted(called)} — those "
            f"write data/user_settings.json, which survives the power button "
            f"and permanently disables the feature it was only meant to pause.")


if __name__ == "__main__":
    unittest.main()
