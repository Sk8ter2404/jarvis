"""Logic tests for skills/voice_clone.py — the voice-facing control surface
over core.voice_clone (Chatterbox).

CI-safe: no chatterbox, no torch, no GPU, no monolith. core.voice_clone is
patched with fakes so the actions' branching (registered? off by default?
refuse unknown/un-consented? honest status?) is exercised without the model.
The skill's best-effort persistence to data/user_settings.json is redirected
into a tempdir so the real file is never touched.

stdlib unittest + unittest.mock only (no pytest).
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

import core.config as cfg


class VoiceCloneSkillTests(unittest.TestCase):
    def setUp(self):
        # Load the skill fresh; capture the actions it registers.
        self.mod, self.actions = load_skill_isolated("voice_clone")

        # Redirect the skill's user_settings persistence into a tempdir so we
        # never write the real data/user_settings.json.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_persist = self.mod._persist
        self.mod._persist = lambda *a, **k: None   # no-op persistence in tests
        self.addCleanup(lambda: setattr(self.mod, "_persist", self._orig_persist))

        # Snapshot + restore the two config knobs the skill mutates so tests
        # don't leak state into each other or the wider suite.
        self._orig_enabled = getattr(cfg, "VOICE_CLONE_ENABLED", False)
        self._orig_profile = getattr(cfg, "VOICE_CLONE_PROFILE", "")
        self.addCleanup(lambda: setattr(cfg, "VOICE_CLONE_ENABLED", self._orig_enabled))
        self.addCleanup(lambda: setattr(cfg, "VOICE_CLONE_PROFILE", self._orig_profile))
        cfg.VOICE_CLONE_ENABLED = False
        cfg.VOICE_CLONE_PROFILE = ""

    def _fake_vc(self, **overrides):
        """A fake core.voice_clone module. Defaults: no profiles, engine not
        available. Override any of list_profiles / load_profile /
        profile_is_usable / is_available."""
        fake = mock.MagicMock()
        fake.list_profiles.return_value = overrides.get("list_profiles", [])
        fake.load_profile.side_effect = overrides.get(
            "load_profile", lambda name: None)
        fake.profile_is_usable.side_effect = overrides.get(
            "profile_is_usable", lambda meta: bool(meta) and meta.get("consent") is True)
        fake.is_available.return_value = overrides.get("is_available", False)
        return fake

    # ── registration ────────────────────────────────────────────────────────
    def test_actions_registered(self):
        for name in ("list_voice_profiles", "set_voice_profile",
                     "voice_clone_status", "disable_voice_clone"):
            self.assertIn(name, self.actions, f"{name} must be registered")
            self.assertTrue(callable(self.actions[name]))

    def test_alias_actions_registered(self):
        for name in ("use_voice_profile", "switch_voice_profile",
                     "stop_voice_clone", "voice_clone_off"):
            self.assertIn(name, self.actions)

    # ── off by default ────────────────────────────────────────────────────────
    def test_status_off_by_default(self):
        with mock.patch.object(self.mod, "_voice_clone", return_value=self._fake_vc()):
            reply = self.actions["voice_clone_status"]("")
        self.assertIn("off", reply.lower())
        self.assertIn("normal voice", reply.lower())

    # ── list ──────────────────────────────────────────────────────────────────
    def test_list_empty(self):
        with mock.patch.object(self.mod, "_voice_clone", return_value=self._fake_vc()):
            reply = self.actions["list_voice_profiles"]("")
        self.assertIn("no voice-clone profiles", reply.lower())

    def test_list_reports_profiles_and_flags_unconsented(self):
        profiles = [
            {"name": "me", "source": "owner", "consent": True},
            {"name": "sketchy", "source": "owner", "consent": False},
        ]
        fake = self._fake_vc(list_profiles=profiles)
        with mock.patch.object(self.mod, "_voice_clone", return_value=fake):
            reply = self.actions["list_voice_profiles"]("")
        self.assertIn("me", reply)
        self.assertIn("sketchy", reply)
        self.assertIn("not consented", reply.lower())

    # ── set: refusals ────────────────────────────────────────────────────────
    def test_set_unknown_profile_refused_gracefully(self):
        fake = self._fake_vc(load_profile=lambda name: None)
        with mock.patch.object(self.mod, "_voice_clone", return_value=fake):
            reply = self.actions["set_voice_profile"]("ghost")
        self.assertIn("don't have a voice profile", reply.lower())
        # Must NOT have enabled cloning on a bogus name.
        self.assertFalse(cfg.VOICE_CLONE_ENABLED)

    def test_set_unconsented_profile_refused(self):
        meta = {"name": "sketchy", "source": "owner", "consent": False}
        fake = self._fake_vc(
            load_profile=lambda name: meta,
            profile_is_usable=lambda m: False)
        with mock.patch.object(self.mod, "_voice_clone", return_value=fake):
            reply = self.actions["set_voice_profile"]("sketchy")
        self.assertIn("won't use", reply.lower())
        self.assertIn("consent", reply.lower())
        self.assertFalse(cfg.VOICE_CLONE_ENABLED)

    def test_set_empty_name_prompts(self):
        with mock.patch.object(self.mod, "_voice_clone", return_value=self._fake_vc()):
            reply = self.actions["set_voice_profile"]("")
        self.assertIn("which voice profile", reply.lower())

    # ── set: success ─────────────────────────────────────────────────────────
    def test_set_consented_profile_enables_and_selects(self):
        meta = {"name": "me", "source": "owner", "consent": True}
        fake = self._fake_vc(
            load_profile=lambda name: meta,
            profile_is_usable=lambda m: True,
            is_available=True)
        with mock.patch.object(self.mod, "_voice_clone", return_value=fake):
            reply = self.actions["set_voice_profile"]("me")
        self.assertIn("switched to", reply.lower())
        self.assertIn("me", reply)
        self.assertTrue(cfg.VOICE_CLONE_ENABLED)
        self.assertEqual(cfg.VOICE_CLONE_PROFILE, "me")

    def test_set_consented_profile_but_engine_not_ready_is_honest(self):
        meta = {"name": "me", "source": "owner", "consent": True}
        fake = self._fake_vc(
            load_profile=lambda name: meta,
            profile_is_usable=lambda m: True,
            is_available=False)     # selected + enabled but chatterbox/CUDA absent
        with mock.patch.object(self.mod, "_voice_clone", return_value=fake):
            reply = self.actions["set_voice_profile"]("me")
        self.assertIn("normal voice", reply.lower())
        # Still selected/enabled so it activates once the engine is installed.
        self.assertTrue(cfg.VOICE_CLONE_ENABLED)
        self.assertEqual(cfg.VOICE_CLONE_PROFILE, "me")

    # ── disable ──────────────────────────────────────────────────────────────
    def test_disable_turns_off(self):
        cfg.VOICE_CLONE_ENABLED = True
        cfg.VOICE_CLONE_PROFILE = "me"
        reply = self.actions["disable_voice_clone"]("")
        self.assertIn("off", reply.lower())
        self.assertFalse(cfg.VOICE_CLONE_ENABLED)

    # ── status: on ───────────────────────────────────────────────────────────
    def test_status_on_with_ready_engine(self):
        cfg.VOICE_CLONE_ENABLED = True
        cfg.VOICE_CLONE_PROFILE = "jarvis"
        fake = self._fake_vc(is_available=True)
        with mock.patch.object(self.mod, "_voice_clone", return_value=fake):
            reply = self.actions["voice_clone_status"]("")
        self.assertIn("on", reply.lower())
        self.assertIn("jarvis", reply)

    def test_status_on_but_engine_unavailable_is_honest(self):
        cfg.VOICE_CLONE_ENABLED = True
        cfg.VOICE_CLONE_PROFILE = "jarvis"
        fake = self._fake_vc(is_available=False)
        with mock.patch.object(self.mod, "_voice_clone", return_value=fake):
            reply = self.actions["voice_clone_status"]("")
        self.assertIn("falling back", reply.lower())

    def test_status_on_but_no_profile(self):
        cfg.VOICE_CLONE_ENABLED = True
        cfg.VOICE_CLONE_PROFILE = ""
        with mock.patch.object(self.mod, "_voice_clone", return_value=self._fake_vc()):
            reply = self.actions["voice_clone_status"]("")
        self.assertIn("no profile", reply.lower())


class PersistenceTests(unittest.TestCase):
    """_persist routes through the shared Settings writer
    (settings_window.load/save_settings) since 2026-07-11 — the old hand-built
    data/user_settings.json path was the ONE writer that ignored the
    JARVIS_SETTINGS_PATH / staging redirects, which is how a staging action
    sweep flipped VOICE_CLONE_ENABLED in the LIVE prod file. Verify the
    expected keys land at settings_path() and that failures never raise."""

    def test_persist_writes_expected_keys(self):
        mod, _ = load_skill_isolated("voice_clone")

        # Redirect the shared writer with its own env override — exactly the
        # isolation contract _persist must now honour.
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "user_settings.json")
            with mock.patch.dict(os.environ,
                                 {"JARVIS_SETTINGS_PATH": target}):
                mod._persist(enabled=True, profile="me")
            self.assertTrue(os.path.isfile(target),
                            "persist should have written the settings file")
            import json as _json
            with open(target, "r", encoding="utf-8") as f:
                written = _json.load(f)

        self.assertTrue(written.get("VOICE_CLONE_ENABLED"))
        self.assertEqual(written.get("VOICE_CLONE_PROFILE"), "me")

    def test_persist_never_raises_on_write_failure(self):
        mod, _ = load_skill_isolated("voice_clone")
        from tools import settings_window as sw
        with mock.patch.object(sw, "save_settings",
                               side_effect=OSError("read-only")):
            # Must swallow and return quietly (persistence is best-effort).
            mod._persist(enabled=False)


if __name__ == "__main__":
    unittest.main()
