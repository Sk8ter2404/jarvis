"""Headless tests for the boot power-up animation hook in the unified HUD
(``hud/jarvis_unified_hud.py``).

WHY THIS EXISTS
  iron_man_boot.py, boot_sequence.py and skills/suit_up.py all publish
  ``boot_phase="powering"`` (+ ``boot_started_at`` / ``boot_duration``) via
  ``_write_hud_state`` and then HOLD the process for MIN_VISIBLE_SECONDS so
  the HUD's power-up rings can fill. For months the ONLY consumer of those
  keys was hud/jarvis_hud.py — a HUD that nothing launches: the launcher
  (``bobert_companion._launch_hud``) spawns hud/jarvis_unified_hud.py, which
  ignored every boot key. Net effect: three producers paid a multi-second
  boot-time sleep to animate a window that did not exist (the classic
  stale-duplicate failure — the rule lived on in one copy while the launched
  copy never got it).

  The fix ports the render: jarvis_unified_hud now reads the boot keys in
  ``_refresh`` and hands ``paintEvent`` to ``_draw_boot_overlay`` while the
  pure gate ``_boot_overlay_active`` holds — including the +0.5 s self-clear
  from jarvis_hud.py so a crashed producer cannot strand the overlay.

  Guarded two ways:
    • a SOURCE-SCANNING INVARIANT that discovers which HUD file the launcher
      actually spawns and which boot keys the producers actually write, then
      requires the launched file to mention every one of those keys — so a
      future HUD repoint that forgets the boot render fails the suite instead
      of silently reintroducing the invisible animation;
    • behavioral tests of ``_refresh`` + the ``_boot_overlay_active`` gate,
      run with PyQt6 genuinely blocked (the module's own ImportError stub
      path) so no display is needed.

ISOLATION
  The module is loaded from its file path under a synthetic name with PyQt6
  imports blocked (mirrors tests/skills/test_hud_geometry_clamp.py). All file
  reads in the behavioral tests are stubbed; nothing under the project tree
  is written.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import time
import unittest
from unittest import mock


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_HUD_DIR = os.path.join(_PROJECT_DIR, "hud")

# The known boot-hook producers (all publish boot_phase="powering"). Listed
# explicitly only as a SANITY FLOOR for the tree scan below — the scan itself
# discovers the real set, so a new producer is picked up automatically.
_KNOWN_PRODUCERS = {"iron_man_boot.py", "boot_sequence.py", "suit_up.py"}

# Directories that must not be crawled for producers (venvs, caches, data).
_SKIP_DIRS = {"__pycache__", "node_modules", "data", "data_staging", "logs",
              "models", "worktrees"}


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _function_body(source: str, func_name: str) -> str:
    """Return the source region of top-level ``def <func_name>`` — from its
    def line up to the next non-indented line (next top-level statement or
    column-0 comment). Text-level on purpose: the scanned monolith must not
    be imported (its singleton lock sys.exit()s in a foreign process)."""
    m = re.search(rf"^def {re.escape(func_name)}\(", source, re.M)
    if not m:
        raise AssertionError(f"def {func_name}( not found")
    start = m.start()
    nxt = re.compile(r"^\S", re.M).search(source, m.end())
    return source[start:nxt.start()] if nxt else source[start:]


def _launched_hud_filename() -> str:
    """The HUD script bobert_companion._launch_hud actually spawns —
    extracted from its os.path.join(..., "hud", "<name>.py") argument."""
    body = _function_body(
        _read_text(os.path.join(_PROJECT_DIR, "bobert_companion.py")),
        "_launch_hud")
    m = re.search(r'"hud",\s*"([^"]+\.py)"', body, re.S)
    if not m:
        raise AssertionError(
            "_launch_hud no longer joins an os.path.join(..., 'hud', "
            "'<script>.py') path — update this extractor with the fix, "
            "and make sure the new launch target renders the boot keys")
    return m.group(1)


def _write_hud_state_call_args(src: str):
    """Yield the argument text of every ``write_hud_state(...)`` call in src
    (matches ``_write_hud_state(`` too), via balanced-paren scanning so the
    multi-line keyword calls the producers use are captured whole."""
    for m in re.finditer(r"write_hud_state\s*\(", src):
        depth, i = 1, m.end()
        while i < len(src) and depth:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
            i += 1
        yield src[m.end():i - 1]


def _producer_boot_keys() -> dict:
    """Scan the tree for write_hud_state(...boot_phase=...) producer calls and
    return {relative_path: {boot_* keyword names written}}. Only keywords
    INSIDE such a call count — a stray ``boot_line = ...`` assignment in the
    same file must not inflate the key set."""
    found: dict = {}
    for root, dirs, files in os.walk(_PROJECT_DIR):
        dirs[:] = [d for d in dirs
                   if d not in _SKIP_DIRS and not d.startswith(".")
                   and not (d == "tests" and root == _PROJECT_DIR)
                   and not (d == "hud" and root == _PROJECT_DIR)]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            src = _read_text(path)
            keys: set = set()
            for args in _write_hud_state_call_args(src):
                if re.search(r"\bboot_phase\s*=", args):
                    keys |= set(re.findall(r"\b(boot_\w+)\s*=", args))
            if keys:
                found[os.path.relpath(path, _PROJECT_DIR)] = keys
    return found


class LaunchedHudRendersBootKeysInvariant(unittest.TestCase):
    """The launched HUD must read every boot key the producers write."""

    def test_producer_scan_finds_the_known_set(self):
        # Sanity floor: if the scan regex rots, this fails LOUDLY rather than
        # the main invariant passing vacuously on an empty producer set.
        found_names = {os.path.basename(p) for p in _producer_boot_keys()}
        self.assertTrue(
            _KNOWN_PRODUCERS <= found_names,
            f"producer scan lost known boot-hook writers: found {found_names}")

    def test_launcher_spawns_an_existing_hud_script(self):
        fn = _launched_hud_filename()
        self.assertTrue(os.path.exists(os.path.join(_HUD_DIR, fn)),
                        f"_launch_hud spawns hud/{fn} which does not exist")

    def test_launched_hud_reads_every_produced_boot_key(self):
        # THE stale-duplicate invariant: whichever HUD file the launcher
        # spawns TODAY must mention every boot_* key any producer writes —
        # a repoint to a HUD without the boot render must fail here.
        producers = _producer_boot_keys()
        all_keys = set().union(*producers.values()) if producers else set()
        self.assertTrue({"boot_phase", "boot_started_at",
                         "boot_duration"} <= all_keys,
                        f"expected the full boot key set, got {all_keys}")
        hud_src = _read_text(os.path.join(_HUD_DIR, _launched_hud_filename()))
        missing = sorted(k for k in all_keys if f'"{k}"' not in hud_src)
        self.assertEqual(
            missing, [],
            f"hud/{_launched_hud_filename()} (the HUD bobert_companion."
            f"_launch_hud actually spawns) never reads the boot key(s) "
            f"{missing} written by {sorted(producers)} — the boot power-up "
            f"animation would render nowhere while the producers still pay "
            f"their MIN_VISIBLE_SECONDS hold for it")


def _load_hud_no_pyqt(testcase, filename, mod_name):
    """Load a HUD source from hud/<filename> with PyQt6 import blocked, so the
    module takes its graceful-degrade path (``_HAS_PYQT6`` False, Qt names
    stubbed). Restores sys.modules + the real importer on cleanup."""
    path = os.path.join(_HUD_DIR, filename)
    real_import = __import__

    def _imp(name, *a, **k):
        if name.split(".")[0] == "PyQt6":
            raise ImportError(f"[test] PyQt6 blocked: {name}")
        return real_import(name, *a, **k)

    hidden = {n: sys.modules.pop(n)
              for n in list(sys.modules) if n.split(".")[0] == "PyQt6"}
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module

    def restore():
        sys.modules.pop(mod_name, None)
        sys.modules.update(hidden)

    testcase.addCleanup(restore)
    with mock.patch("builtins.__import__", side_effect=_imp):
        spec.loader.exec_module(module)
    testcase.assertFalse(module._HAS_PYQT6,
                         "PyQt6 should be blocked → headless degrade path")
    return module


class UnifiedHudBootPhaseBehaviour(unittest.TestCase):
    MOD_NAME = "_ju_hud_bootphase_under_test"

    def setUp(self):
        self.mod = _load_hud_no_pyqt(self, "jarvis_unified_hud.py",
                                     self.MOD_NAME)

    def _bare_hud(self):
        """A UnifiedHud instance without running the Qt-heavy __init__ —
        only the attributes _refresh touches, so the reader can run headless."""
        hud = object.__new__(self.mod.UnifiedHud)
        hud.parent_pid = 0
        hud.frame = 0
        hud.gpu_util = None
        hud._gpu_sampling = False
        hud._gpu_cached_at = time.time() + 3600.0   # cache "fresh" → no thread
        hud._last_net = None
        hud._last_net_at = None
        hud._refresh_camera_preview = lambda: False  # needs isVisible() → stub
        hud.boot_phase = ""
        hud.boot_started_at = 0.0
        hud.boot_duration = 0.0
        return hud

    def _refresh_with_state(self, hud, state: dict) -> bool:
        def fake_read_json(path):
            return dict(state) if path == self.mod.HUD_STATE_FILE else {}
        with mock.patch.object(self.mod, "_read_json",
                               side_effect=fake_read_json), \
             mock.patch.object(self.mod, "_is_parent_alive",
                               return_value=True), \
             mock.patch.object(self.mod, "_control_says_off",
                               return_value=False):
            return hud._refresh()

    # ── _refresh reads the producer keys ──────────────────────────────────
    def test_refresh_reads_boot_keys_and_gate_is_active(self):
        hud = self._bare_hud()
        now = time.time()
        ok = self._refresh_with_state(hud, {
            "boot_phase": "powering",
            "boot_started_at": now,
            "boot_duration": 4.5,
            "state": "Initialising",
        })
        self.assertTrue(ok)
        self.assertEqual(hud.boot_phase, "powering")
        self.assertAlmostEqual(hud.boot_started_at, now, places=3)
        self.assertEqual(hud.boot_duration, 4.5)
        self.assertTrue(self.mod._boot_overlay_active(
            hud.boot_phase, hud.boot_started_at, hud.boot_duration,
            time.time()))

    def test_refresh_clear_write_deactivates_gate(self):
        # The producers' clearing write (boot_phase="", boot_started_at=0.0)
        # must drop the overlay immediately.
        hud = self._bare_hud()
        self._refresh_with_state(hud, {
            "boot_phase": "", "boot_started_at": 0.0, "state": "Idle",
        })
        self.assertEqual(hud.boot_phase, "")
        self.assertFalse(self.mod._boot_overlay_active(
            hud.boot_phase, hud.boot_started_at, hud.boot_duration,
            time.time()))

    def test_refresh_tolerates_garbage_boot_values(self):
        hud = self._bare_hud()
        ok = self._refresh_with_state(hud, {
            "boot_phase": "powering",
            "boot_started_at": "not-a-number",
            "boot_duration": None,
        })
        self.assertTrue(ok)
        self.assertEqual(hud.boot_started_at, 0.0)
        self.assertEqual(hud.boot_duration, 0.0)
        self.assertFalse(self.mod._boot_overlay_active(
            hud.boot_phase, hud.boot_started_at, hud.boot_duration,
            time.time()))

    # ── the pure gate: jarvis_hud.py:1164-1166 contract ───────────────────
    def test_gate_true_within_duration(self):
        now = time.time()
        self.assertTrue(self.mod._boot_overlay_active(
            "powering", now - 1.0, 4.5, now))

    def test_gate_self_clears_after_duration_plus_half_second(self):
        # A crashed producer never writes the clearing boot_phase="" — the
        # overlay must still drop 0.5 s past the advertised duration.
        now = time.time()
        self.assertTrue(self.mod._boot_overlay_active(
            "powering", now - 4.9, 4.5, now))    # inside the +0.5 grace
        self.assertFalse(self.mod._boot_overlay_active(
            "powering", now - 5.1, 4.5, now))    # past it → self-clear

    def test_gate_requires_powering_phase_and_positive_fields(self):
        now = time.time()
        self.assertFalse(self.mod._boot_overlay_active("", now, 4.5, now))
        self.assertFalse(self.mod._boot_overlay_active("suit_up", now, 4.5, now))
        self.assertFalse(self.mod._boot_overlay_active("powering", 0.0, 4.5, now))
        self.assertFalse(self.mod._boot_overlay_active("powering", now, 0.0, now))

    # ── paint wiring ──────────────────────────────────────────────────────
    def test_paint_event_is_wired_to_the_gate_and_overlay(self):
        # Headless we cannot actually paint, so pin the wiring at source
        # level: paintEvent must consult the gate and hand off to the
        # overlay renderer, which must exist as a real method.
        import inspect
        paint_src = inspect.getsource(self.mod.UnifiedHud.paintEvent)
        self.assertIn("_boot_overlay_active", paint_src)
        self.assertIn("_draw_boot_overlay", paint_src)
        self.assertTrue(callable(
            getattr(self.mod.UnifiedHud, "_draw_boot_overlay", None)))


if __name__ == "__main__":
    unittest.main()
