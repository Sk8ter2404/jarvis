"""Test harness for loading JARVIS skills in ISOLATION.

Replicates the injection contract of ``bobert_companion.load_skills()`` — a
fake ``skill_utils`` dict plus a fresh actions dict — so any skill can be
imported and its ``register()`` called WITHOUT booting the ~14K-line monolith.
Stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.

    from tests._skill_harness import load_skill_isolated, make_fake_skill_utils
    mod, actions = load_skill_isolated("timer")
    assert "set_timer" in actions
    out = actions["set_timer"]("5 minutes | tea")

The real loader (bobert_companion.py ~10159-10266) does, per skill:
  spec = importlib.util.spec_from_file_location("skill_<name>", path, ...)
  mod  = importlib.util.module_from_spec(spec)
  mod.skill_utils = skill_utils          # inject BEFORE exec
  sys.modules["skill_<name>"] = mod      # register BEFORE exec
  spec.loader.exec_module(mod)
  if hasattr(mod, "register"): mod.register(ACTIONS)
This module mirrors that exactly, with the live helpers replaced by mocks.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import threading
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = os.path.join(_PROJECT_ROOT, "skills")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# The exact keys bobert_companion.skill_utils injects (loader ~line 10159).
# tests/test_skill_harness.py asserts this stays in sync with the monolith by
# AST-parsing the real dict — so adding a key there without updating here fails.
SKILL_UTILS_KEYS = (
    "ask_vision", "take_screenshot", "find_click_target", "click", "type_text",
    "press_key", "hotkey", "scroll", "sleep", "launch_app", "open_url",
    "write_hud_state", "make_promise", "register_promise_condition",
    "fulfil_promise",
)

# Top-level module names whose import failure indicates a REAL bug (a broken
# intra-project import), NOT a missing optional third-party dependency.
_INTRA_PREFIXES = {"core", "skills", "bobert_companion", "tools", "adapters",
                   "hud", "tests"}


def make_fake_skill_utils(**overrides):
    """Fake ``skill_utils``: every key a MagicMock, with sensible default
    return values for the ones whose result handlers commonly consume. Pass
    keyword overrides to pin specific behaviour for a test."""
    utils = {k: mock.MagicMock(name=f"skill_utils[{k}]") for k in SKILL_UTILS_KEYS}
    utils["ask_vision"].return_value = ""
    utils["take_screenshot"].return_value = None
    utils["find_click_target"].return_value = None
    utils["make_promise"].return_value = None
    utils["register_promise_condition"].return_value = None
    utils["fulfil_promise"].return_value = False
    utils["sleep"].return_value = None  # never actually sleep in a test
    utils.update(overrides)
    return utils


def skill_path(name: str):
    """Resolve a skill name to (path, submodule_search_locations). A package
    ``skills/<name>/__init__.py`` is preferred over a flat ``skills/<name>.py``
    — matching the loader's precedence."""
    pkg_init = os.path.join(SKILLS_DIR, name, "__init__.py")
    if os.path.isfile(pkg_init):
        return pkg_init, [os.path.join(SKILLS_DIR, name)]
    flat = os.path.join(SKILLS_DIR, f"{name}.py")
    if os.path.isfile(flat):
        return flat, None
    raise FileNotFoundError(f"no skill named {name!r} under {SKILLS_DIR}")


@contextlib.contextmanager
def no_background_threads():
    """Neuter ``Thread.start`` (and therefore ``Timer.start``) so the daemon
    threads ~20 skills spawn in ``register()`` don't actually run during a test
    — they'd open audio/camera devices, hit the network, or linger and pollute
    other tests. The thread objects are still constructed; only ``start`` is a
    no-op, so ``register()`` completes normally."""
    with mock.patch.object(threading.Thread, "start", lambda self: None):
        yield


def load_skill_isolated(name, *, utils=None, actions=None, register=True,
                        neuter_threads=True, capture_output=True,
                        skip_if_missing_dep=True):
    """Import ``skills/<name>`` in isolation with a fake ``skill_utils``,
    optionally calling ``register(actions)``. Returns ``(module, actions)``.

    Mirrors ``load_skills()``: injects ``mod.skill_utils`` and registers the
    module as ``sys.modules['skill_<name>']`` BEFORE exec, so package
    sub-imports and cross-skill ``sys.modules.get('skill_x')`` lookups resolve.
    Each call re-execs a fresh module (test isolation), unlike the idempotent
    live loader.
    """
    path, search_locs = skill_path(name)
    utils = make_fake_skill_utils() if utils is None else utils
    actions = {} if actions is None else actions

    spec = importlib.util.spec_from_file_location(
        f"skill_{name}", path, submodule_search_locations=search_locs)
    if not spec or not spec.loader:
        raise ImportError(f"could not build import spec for skill {name!r}")
    mod = importlib.util.module_from_spec(spec)
    mod.skill_utils = utils
    sys.modules[f"skill_{name}"] = mod

    threads_cm = no_background_threads() if neuter_threads else contextlib.nullcontext()
    # Capture stdout so a skill's register()/import-time prints don't pollute
    # test output — and don't crash on a cp1252 console when a skill prints a
    # non-ASCII char (e.g. the latent UnicodeEncodeError some skills hit). Tests
    # assert on return values, not prints.
    out_cm = contextlib.redirect_stdout(io.StringIO()) if capture_output \
        else contextlib.nullcontext()
    try:
        with threads_cm, out_cm:
            spec.loader.exec_module(mod)
            if register and hasattr(mod, "register"):
                mod.register(actions)
    except (ImportError, ModuleNotFoundError) as exc:
        # A missing THIRD-PARTY dep (torch, cv2, …) skips the test on a reduced
        # CI runner; a missing INTRA-PROJECT module is a real broken import and
        # propagates. Local dev has every dep, so this never skips there.
        dep = (getattr(exc, "name", None) or "").split(".")[0]
        if skip_if_missing_dep and dep and dep not in _INTRA_PREFIXES:
            import unittest as _unittest
            raise _unittest.SkipTest(
                f"skill {name!r}: third-party dep not installed: {dep}") from exc
        raise
    return mod, actions


def list_skill_names():
    """Every loadable skill name — flat ``.py`` modules and package dirs with
    an ``__init__.py`` — excluding ``_``-prefixed and ``__pycache__`` (matches
    the live loader's discovery)."""
    names: set[str] = set()
    for entry in sorted(os.listdir(SKILLS_DIR)):
        full = os.path.join(SKILLS_DIR, entry)
        if entry.startswith("_") or entry == "__pycache__":
            continue
        if os.path.isdir(full):
            if os.path.isfile(os.path.join(full, "__init__.py")):
                names.add(entry)
        elif entry.endswith(".py"):
            names.add(entry[:-3])
    return sorted(names)
