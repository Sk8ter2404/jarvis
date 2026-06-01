"""Smoke test for EVERY skill in skills/.

Generates one test per skill that loads it in isolation (fake skill_utils, no
monolith) and runs its register(). The contract verified:

  * the module imports and execs cleanly;
  * register() runs without raising (skills must degrade gracefully — return
    strings, never crash, when an optional dep / credential / data file is
    absent);
  * a missing THIRD-PARTY dependency skips the test (expected in a reduced CI
    env), but a missing INTRA-PROJECT module (core.*, skills.*, …) FAILS — that
    is a real broken import, not an environment gap.

This is the structural "every skill is exercised" guarantee. Per-skill logic
assertions live in tests/skills/test_<skill>.py.
"""
from __future__ import annotations

import unittest

from tests._skill_harness import list_skill_names, load_skill_isolated

# Modules that, if their import fails, indicate a REAL bug (not an env gap).
_INTRA_PREFIXES = {"core", "skills", "bobert_companion", "tools", "adapters",
                   "hud", "tests"}


def _is_intra(module_name: str | None) -> bool:
    return bool(module_name) and module_name.split(".")[0] in _INTRA_PREFIXES


class SkillSmokeTests(unittest.TestCase):
    """One generated test_smoke_<skill> method per skill (added below)."""


def _make_smoke(name: str):
    def test(self: unittest.TestCase) -> None:
        try:
            _mod, actions = load_skill_isolated(name)
        except (ImportError, ModuleNotFoundError) as exc:
            dep = getattr(exc, "name", None)
            if _is_intra(dep):
                raise  # broken intra-project import — real bug
            self.skipTest(f"third-party dep not installed: {dep}")
            return
        self.assertIsInstance(actions, dict)
    test.__name__ = f"test_smoke_{name}"
    test.__doc__ = f"skill {name!r} loads + registers in isolation"
    return test


for _name in list_skill_names():
    setattr(SkillSmokeTests, f"test_smoke_{_name}", _make_smoke(_name))


class SkillActionSurfaceTests(unittest.TestCase):
    """Aggregate guards so a mass-registration regression can't pass silently."""

    def test_action_surface_is_populated(self):
        loaded = 0
        skipped = 0
        total_actions = 0
        crashed: list[str] = []
        for name in list_skill_names():
            try:
                _mod, actions = load_skill_isolated(name, skip_if_missing_dep=False)
            except (ImportError, ModuleNotFoundError) as exc:
                if _is_intra(getattr(exc, "name", None)):
                    crashed.append(f"{name} (intra import: {exc})")
                else:
                    skipped += 1
                continue
            except Exception as exc:  # surfaced per-skill; don't double-fail here
                crashed.append(f"{name} ({type(exc).__name__})")
                continue
            loaded += 1
            total_actions += len(actions)
        # In this full-deps environment most skills load; in CI more skip. Keep
        # the thresholds low enough to hold in a reduced env but high enough to
        # catch a collapse of the registration surface.
        self.assertGreater(loaded, 25,
                           f"only {loaded} skills loaded (skipped {skipped}); "
                           f"crashed={crashed}")
        self.assertGreater(total_actions, 80,
                           f"action surface too small: {total_actions}")


if __name__ == "__main__":
    unittest.main()
