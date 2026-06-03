"""Monolith-tier test: the skill loader injects a working ``mod.services``.

M2 Phase 1 wired a typed :class:`core.services.JarvisServices` facade alongside
the legacy ``skill_utils`` dict. ``bobert_companion.load_skills`` now does, per
skill module::

    mod.skill_utils = skill_utils          # unchanged (legacy seam)
    mod.services    = _jarvis_services      # NEW (typed seam), additive

This module proves that end-to-end against the *real* monolith:

  * ``mod.services`` is injected and is a ``JarvisServices``.
  * Its methods reach the actual ``skill_utils`` lambdas — calling
    ``services.write_hud_state(...)`` invokes the monolith's canonical
    ``_write_hud_state`` writer (asserted via a patched recorder), and
    ``services.open_url(...)`` reaches the same lambda the dict exposes.
  * The injection is ADDITIVE: ``mod.skill_utils`` is still injected, unchanged,
    and is still the exact same dict object the monolith built — so every
    existing ``skill_utils["..."](...)`` call keeps working.

LOCAL full-tier only: the monolith top-level-imports heavy deps absent on the
light CI runner, so ``MonolithGlobalsTestCase`` (which is ``@requires_monolith``)
skips the whole class there. Uses the same temp-``SKILLS_DIR`` fixture approach
as ``tests/monolith/test_monolith_sec7.py``'s ``LoadSkillsTests`` and isolates
``ACTIONS`` / ``_loaded_skill_names`` / new ``skill_*`` modules so the live
monolith state is left pristine.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

# A fixture skill that records, at register() time, the objects the loader
# injected into its module namespace. We read them back off the loaded module.
_FIXTURE_SKILL = (
    "captured = {}\n"
    "def register(actions):\n"
    "    # `skill_utils` and `services` are injected as module globals by the\n"
    "    # loader BEFORE exec, so they're visible here.\n"
    "    captured['skill_utils'] = globals().get('skill_utils')\n"
    "    captured['services'] = globals().get('services')\n"
    "    actions['svc_probe'] = lambda a='': 'ok'\n"
)


@requires_monolith
class ServicesInjectionTests(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._dir = self._tmp.name
        # Point the loader at our temp dir and isolate the mutable load state so
        # the real monolith's ACTIONS / loaded-name latch aren't polluted.
        self._patch(bc, "SKILLS_DIR", self._dir)
        self._patch(bc, "SKILLS_ENABLED", True)
        self._patch(bc, "_loaded_skill_names", set())
        self._patch(bc, "ACTIONS", dict(bc.ACTIONS))
        self._mods_before = set(bc.sys.modules)
        self.addCleanup(self._drop_new_skill_modules)

    def _patch(self, *args, **kwargs):
        p = mock.patch.object(*args, **kwargs)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def _drop_new_skill_modules(self):
        for name in set(self.bc.sys.modules) - self._mods_before:
            if name.startswith("skill_"):
                self.bc.sys.modules.pop(name, None)

    def _write_fixture(self, name="svc_fixture"):
        path = os.path.join(self._dir, f"{name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_FIXTURE_SKILL)
        return name

    def _load_fixture_module(self):
        name = self._write_fixture()
        self.bc.load_skills()
        mod = self.bc.sys.modules.get(f"skill_{name}")
        self.assertIsNotNone(mod, "fixture skill failed to load")
        return mod

    # ── the additive injection itself ─────────────────────────────────────
    def test_services_injected_and_is_jarvis_services(self):
        from core.services import JarvisServices
        mod = self._load_fixture_module()
        self.assertTrue(hasattr(mod, "services"),
                        "loader did not inject mod.services")
        self.assertIsInstance(mod.services, JarvisServices)

    def test_skill_utils_still_injected_unchanged(self):
        # ADDITIVITY: the legacy dict is still injected AND is the very same
        # object the monolith built (identity), so existing skills are untouched.
        mod = self._load_fixture_module()
        self.assertTrue(hasattr(mod, "skill_utils"))
        self.assertIsInstance(mod.skill_utils, dict)
        self.assertIs(mod.skill_utils, self.bc.skill_utils)
        # Spot-check the legacy access path a skill uses today still resolves.
        self.assertTrue(callable(mod.skill_utils.get("write_hud_state")))

    def test_services_wraps_the_same_skill_utils_dict(self):
        # The injected facade must be backed by the monolith's live dict, not a
        # copy — single source of truth.
        mod = self._load_fixture_module()
        self.assertIs(mod.services._utils, self.bc.skill_utils)

    # ── methods actually reach the backing lambdas ────────────────────────
    def test_write_hud_state_reaches_monolith_writer(self):
        # services.write_hud_state -> skill_utils["write_hud_state"] lambda ->
        # bc._write_hud_state(**kw). Patch the writer and assert it's hit with
        # the kwargs we passed. (The lambda resolves _write_hud_state at call
        # time, so patching the module global is observed.)
        mod = self._load_fixture_module()
        with mock.patch.object(self.bc, "_write_hud_state") as writer:
            mod.services.write_hud_state(probe_strip="X", probe_ts=123)
        writer.assert_called_once_with(probe_strip="X", probe_ts=123)

    def test_open_url_reaches_monolith_action(self):
        # services.open_url -> skill_utils["open_url"] lambda -> _act_open_url.
        mod = self._load_fixture_module()
        with mock.patch.object(self.bc, "_act_open_url",
                               return_value="opened") as act:
            result = mod.services.open_url("https://example.test")
        act.assert_called_once_with("https://example.test")
        self.assertEqual(result, "opened")

    def test_dict_and_services_paths_hit_the_same_writer(self):
        # The whole point of additivity: the old dict call and the new typed
        # call funnel into the identical monolith writer.
        mod = self._load_fixture_module()
        with mock.patch.object(self.bc, "_write_hud_state") as writer:
            mod.skill_utils["write_hud_state"](via="dict")
            mod.services.write_hud_state(via="services")
        self.assertEqual(writer.call_count, 2)
        self.assertEqual(writer.call_args_list[0].kwargs, {"via": "dict"})
        self.assertEqual(writer.call_args_list[1].kwargs, {"via": "services"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
