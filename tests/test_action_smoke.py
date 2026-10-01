"""tools/action_smoke.py runs its sweep in a hermetic sandbox (2026-10-01).

The 09-05 live diagnostic: running the action sweep made the LIVE JARVIS
speak fake alerts. The sweep only set JARVIS_STAGING=1 (with `setdefault`, so
an inherited "0" won) and a settings redirect, while the pending-speech queue
the live loop speaks from — and the inject / tray inboxes, jarvis_todo.md,
every root *_state.json — are bound to their module's __file__ and honour no
redirect. The sweep now copies the code into a temp dir and runs there with
every redirect forced, and refuses to start when any runtime path would
point at the real tree. These pin the pure pieces; the end-to-end run is
tests/monolith/test_monolith_action_smoke_sandbox.py.

Stdlib unittest, CI-safe (the monolith is never imported).

    python -m unittest tests.test_action_smoke
"""
from __future__ import annotations

import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

from tools import action_smoke as smoke


class HermeticProblemsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="smoke_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.real = os.path.join(self.tmp, "real")
        self.box = os.path.join(self.tmp, "box", "tree")
        os.makedirs(self.real)
        os.makedirs(self.box)

    def test_paths_inside_the_sandbox_are_hermetic(self):
        self.assertEqual(smoke.hermetic_problems(self.real, self.box, {
            "data dir": os.path.join(self.box, "data"),
            "speech queue": os.path.join(self.box, "pending_speech.json"),
        }), [])

    def test_the_live_speech_queue_is_refused(self):
        # Exactly the old layout: the sweep's monolith wrote the queue the
        # live loop drains.
        got = smoke.hermetic_problems(self.real, self.box, {
            "speech queue": os.path.join(self.real, "pending_speech.json")})
        self.assertEqual(len(got), 1)
        self.assertIn("inside the real tree", got[0])

    def test_the_live_data_dir_is_refused(self):
        got = smoke.hermetic_problems(self.real, self.box, {
            "data dir": os.path.join(self.real, "data")})
        self.assertIn("inside the real tree", got[0])

    def test_a_path_outside_both_is_refused(self):
        got = smoke.hermetic_problems(self.real, self.box, {
            "lock dir": os.path.join(self.tmp, "elsewhere")})
        self.assertIn("outside the sandbox", got[0])

    def test_unresolved_and_overlapping_are_refused(self):
        self.assertIn("unresolved",
                      smoke.hermetic_problems(self.real, self.box,
                                              {"x": ""})[0])
        inner = os.path.join(self.real, "sub")
        got = smoke.hermetic_problems(self.real, inner, {})
        self.assertIn("overlaps the real tree", got[0])


class SandboxEnvTests(unittest.TestCase):
    def test_every_redirect_is_forced_into_the_copy(self):
        real, tree = os.path.abspath("realroot"), os.path.abspath("box/tree")
        inherited = {"JARVIS_STAGING": "0",
                     "JARVIS_DATA_DIR": os.path.join(real, "data"),
                     "JARVIS_SETTINGS_PATH": os.path.join(real, "data", "u.json"),
                     "JARVIS_ALLOW_LIVE_DATA": "1", "KEEP_ME": "yes"}
        env = smoke.sandbox_env(real, tree, inherited)
        self.assertEqual(env["JARVIS_STAGING"], "1")
        self.assertEqual(env["MUTE_TTS"], "1")
        self.assertEqual(env["JARVIS_DATA_DIR"], os.path.join(tree, "data"))
        self.assertEqual(env["JARVIS_SETTINGS_PATH"],
                         os.path.join(tree, "data", "user_settings.json"))
        self.assertEqual(env["JARVIS_LOCK_DIR"], os.path.join(tree, "locks"))
        self.assertEqual(env["JARVIS_GUARD_LIVE_ROOT"], real)
        self.assertNotIn("JARVIS_ALLOW_LIVE_DATA", env)
        self.assertEqual(env["KEEP_ME"], "yes")
        self.assertEqual(inherited["JARVIS_STAGING"], "0")   # not mutated


class BuildSandboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="smoke_build_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.real = os.path.join(self.tmp, "real")
        for rel, body in (("bobert_companion.py", "x = 1\n"),
                          ("VERSION", "9.9.9\n"),
                          ("core/a.py", "y = 2\n"),
                          ("core/english_words.txt", "word\n"),
                          ("skills/personal_skill.py", "z = 3\n"),
                          ("pending_speech.json", "[{\"message\": \"live\"}]"),
                          ("hud_state.json", "{}"),
                          ("jarvis_todo.md", "live todo\n"),
                          ("data/user_settings.json", "{\"SECRET\": 1}"),
                          ("logs/session.log", "live log\n")):
            p = os.path.join(self.real, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(body)

    def _build(self, **kw):
        # Not a git repo: the walk fallback is what runs.
        with mock.patch.object(smoke.subprocess, "run",
                               side_effect=OSError("no git")):
            tree = smoke.build_sandbox(self.real, base_dir=self.tmp, **kw)
        self.addCleanup(shutil.rmtree, os.path.dirname(tree), True)
        return tree

    def test_code_is_copied_and_live_state_is_not(self):
        tree = self._build()
        for rel in ("bobert_companion.py", "VERSION", "core/a.py",
                    "core/english_words.txt", "skills/personal_skill.py"):
            self.assertTrue(os.path.isfile(os.path.join(tree, rel)), rel)
        for rel in ("pending_speech.json", "hud_state.json", "jarvis_todo.md",
                    "data/user_settings.json", "logs/session.log"):
            self.assertFalse(os.path.exists(os.path.join(tree, rel)), rel)
        self.assertEqual(os.listdir(os.path.join(tree, "data")), [])

    def test_the_copy_is_outside_the_real_tree(self):
        tree = self._build()
        self.assertEqual(smoke.hermetic_problems(self.real, tree, {
            "code copy": tree}), [])

    def test_settings_are_only_copied_on_request(self):
        src = os.path.join(self.tmp, "seed.json")
        with open(src, "w", encoding="utf-8") as f:
            f.write("{\"A\": 1}")
        tree = self._build(settings=src)
        with open(os.path.join(tree, "data", "user_settings.json"),
                  encoding="utf-8") as f:
            self.assertEqual(f.read(), "{\"A\": 1}")


class ChildRefusalTests(unittest.TestCase):
    def test_the_child_refuses_to_sweep_the_real_tree(self):
        # Run in-tree (no parent), or told the real tree is its own: refuse
        # before importing anything.
        for env in ({}, {"JARVIS_SMOKE_REAL_ROOT": smoke._HERE_ROOT}):
            with self.subTest(env=env):
                with mock.patch.dict(os.environ, env, clear=False):
                    if not env:
                        os.environ.pop("JARVIS_SMOKE_REAL_ROOT", None)
                    with mock.patch("builtins.print"):
                        self.assertEqual(smoke._child([]), 2)

    def test_the_preflight_refuses_a_live_queue(self):
        tmp = tempfile.mkdtemp(prefix="smoke_pf_")
        self.addCleanup(shutil.rmtree, tmp, True)
        real, tree = os.path.join(tmp, "real"), os.path.join(tmp, "box")
        os.makedirs(real)
        os.makedirs(tree)
        announce = mock.Mock(return_value=True)
        fake = types.SimpleNamespace(
            __file__=os.path.join(tree, "bobert_companion.py"),
            _singleton_lock_dir=lambda: os.path.join(tree, "locks"),
            PENDING_SPEECH_PATH=os.path.join(real, "pending_speech.json"),
            INJECTED_COMMANDS_PATH=os.path.join(tree, "injected_commands.json"),
            TRAY_COMMANDS_FILE=os.path.join(tree, "tray_commands.json"),
            proactive_announce=announce)
        with mock.patch.dict(os.environ, {
                "JARVIS_DATA_DIR": os.path.join(tree, "data"),
                "JARVIS_SETTINGS_PATH": os.path.join(tree, "data", "u.json")}):
            problems = smoke._child_preflight(fake, real, tree)
        self.assertTrue(any("speech queue" in p for p in problems), problems)
        announce.assert_not_called()      # nothing ran


class GuardRootOverrideTests(unittest.TestCase):
    """tests/live_data_guard.py protects the REAL tree when the sweep runs in
    a copy (JARVIS_GUARD_LIVE_ROOT), and ignores a value that is not a JARVIS
    tree."""

    def _root_with(self, value):
        from tests import live_data_guard as g
        with mock.patch.dict(os.environ, {"JARVIS_GUARD_LIVE_ROOT": value}):
            return g._protected_root()

    def test_override_must_name_a_jarvis_tree(self):
        from tests import live_data_guard as g
        own = os.path.dirname(os.path.dirname(os.path.abspath(g.__file__)))
        tmp = tempfile.mkdtemp(prefix="guard_root_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.assertEqual(self._root_with(tmp), own)          # not a tree
        open(os.path.join(tmp, "bobert_companion.py"), "w").close()
        self.assertEqual(self._root_with(tmp), os.path.abspath(tmp))
        self.assertEqual(self._root_with(""), own)

    def test_the_module_still_imports_and_names_its_root(self):
        from tests import live_data_guard as g
        self.assertEqual(g.LIVE_DATA_DIR, os.path.join(g.PROJECT_ROOT, "data"))


if __name__ == "__main__":
    unittest.main()
