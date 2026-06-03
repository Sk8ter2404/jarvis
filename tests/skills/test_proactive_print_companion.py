"""Logic tests for skills/proactive_print_companion.py.

MCU-flavoured commentary layered on bambu_monitor. High-value targets:

  • pure helpers: _format_eta_clock, _infer_material, _bucket_key,
    _next_hedge_suffix, _milestone_commentary, _completion_offer_line
  • pattern persistence: _record_print_outcome / _historical_failure_rate /
    _maybe_warn_historical_failure (all pointed at a temp patterns file)
  • downstream-availability probes: _light_skill_available,
    _timer_skill_available, _vision_available (degrade to False on any doubt)
  • the print_companion_status / print_companion_history actions

The patterns JSON is redirected to a tempfile via patched _PATTERNS_FILE /
_DATA_DIR so the real data/ file is untouched, and _enqueue_speech is patched
so nothing reaches pending_speech.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_bambu(state=None, announced=None):
    m = types.ModuleType("skill_bambu_monitor")
    m._state_lock = threading.Lock()
    m._state = state if state is not None else {"last_update": 0.0}
    m._announced_milestones = announced if announced is not None else set()
    m.register_state_change_hook = mock.MagicMock()
    return m


class PrintCompanionMixin:
    def _load(self, bambu_state="__absent__", announced=None):
        # Redirect the patterns file to a fresh temp dir before register() runs.
        self.tmpdir = tempfile.mkdtemp()
        self.patterns_path = os.path.join(self.tmpdir, "patterns.json")
        self.addCleanup(self._cleanup_tmp)

        ctx = []
        if bambu_state != "__absent__":
            fake = _fake_bambu(bambu_state, announced)
            p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake})
            p.start(); ctx.append(p)
            self._fake = fake
        else:
            p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": None})
            p.start(); ctx.append(p)
            self._fake = None
        for p in ctx:
            self.addCleanup(p.stop)

        mod, actions = load_skill_isolated("proactive_print_companion")
        # Point persistence at the temp file (register() already ran, but every
        # _load_patterns/_save_patterns reads these module globals live).
        mod._PATTERNS_FILE = self.patterns_path
        mod._DATA_DIR = self.tmpdir
        return mod, actions

    def _cleanup_tmp(self):
        try:
            for f in os.listdir(self.tmpdir):
                os.remove(os.path.join(self.tmpdir, f))
            os.rmdir(self.tmpdir)
        except OSError:
            pass


class PrintCompanionImportGuardTests(unittest.TestCase):
    def test_path_bootstrap_inserts_project_root(self):
        # Re-exec the source with the project root removed from sys.path so the
        # `if _PROJECT_DIR not in sys.path: sys.path.insert(...)` guard runs.
        # core.atomic_io is cached, so the from-import still resolves.
        mod, _ = load_skill_isolated("proactive_print_companion")
        path = mod.__file__
        proj = os.path.dirname(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("ppc_reexec", path)
        m = importlib.util.module_from_spec(spec)
        m.skill_utils = {}
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path
                           if os.path.abspath(p) != os.path.abspath(proj)]
            spec.loader.exec_module(m)
            self.assertIn(m._PROJECT_DIR, sys.path)
        finally:
            sys.path[:] = saved


class PrintCompanionHelperTests(PrintCompanionMixin, unittest.TestCase):
    def test_format_eta_clock(self):
        mod, _a = self._load()
        # 90 minutes from a fixed base → predictable HH:MM.
        base = time.mktime((2026, 6, 1, 10, 0, 0, 0, 0, -1))
        with mock.patch.object(mod.time, "time", return_value=base):
            self.assertEqual(mod._format_eta_clock(90), "11:30")
        self.assertEqual(mod._format_eta_clock(0), "")
        self.assertEqual(mod._format_eta_clock(None), "")

    def test_infer_material_plus_variant_first(self):
        mod, _a = self._load()
        # "pla+" must win over "pla".
        self.assertEqual(mod._infer_material("Bracket_PLA+_BLACK.3mf"), "plaplus")
        self.assertEqual(mod._infer_material("Gear_PETG_4h.3mf"), "petg")
        self.assertEqual(mod._infer_material("Vase_PLA.gcode"), "pla")
        self.assertEqual(mod._infer_material("noname.3mf"), "unknown")
        self.assertEqual(mod._infer_material(""), "unknown")

    def test_bucket_key_coarse_layers(self):
        mod, _a = self._load()
        # 215 and 250 land in the same 200 bucket.
        self.assertEqual(mod._bucket_key("pla", 215), "pla_200")
        self.assertEqual(mod._bucket_key("pla", 250), "pla_200")
        self.assertEqual(mod._bucket_key("petg", None), "petg_0")

    def test_next_hedge_suffix_cycles(self):
        mod, _a = self._load()
        mod._mcu_hedge_idx[0] = 0
        n = len(mod._MCU_HEDGE_SUFFIXES)
        seen = [mod._next_hedge_suffix() for _ in range(n)]
        self.assertEqual(len(set(seen)), n)  # one full unique cycle
        # Wraps back to the first.
        self.assertEqual(mod._next_hedge_suffix(), seen[0])

    def test_milestone_commentary_openers(self):
        mod, _a = self._load()
        base = time.mktime((2026, 6, 1, 10, 0, 0, 0, 0, -1))
        with mock.patch.object(mod.time, "time", return_value=base):
            self.assertIn("Quarter of the way along",
                          mod._milestone_commentary(25, 60))
            self.assertIn("Halfway through", mod._milestone_commentary(50, 60))
            self.assertIn("Three-quarters complete",
                          mod._milestone_commentary(75, 60))

    def test_milestone_commentary_unknown_eta(self):
        mod, _a = self._load()
        out = mod._milestone_commentary(25, None)
        self.assertIn("trajectory still settling", out)


class PrintCompanionCompletionOfferTests(PrintCompanionMixin, unittest.TestCase):
    def test_offer_with_both_lights_and_timer(self):
        mod, _a = self._load()
        with mock.patch.object(mod, "_light_skill_available", return_value=True), \
             mock.patch.object(mod, "_timer_skill_available", return_value=True):
            out = mod._completion_offer_line("cube")
        self.assertIn("cube", out)
        self.assertIn("dim the workshop lights", out)
        self.assertIn("queue a cooldown timer", out)

    def test_offer_lights_only(self):
        mod, _a = self._load()
        with mock.patch.object(mod, "_light_skill_available", return_value=True), \
             mock.patch.object(mod, "_timer_skill_available", return_value=False):
            out = mod._completion_offer_line("cube")
        self.assertIn("dim the workshop lights", out)
        self.assertNotIn("cooldown timer", out)

    def test_offer_neither_falls_back_to_plain(self):
        mod, _a = self._load()
        with mock.patch.object(mod, "_light_skill_available", return_value=False), \
             mock.patch.object(mod, "_timer_skill_available", return_value=False):
            out = mod._completion_offer_line("cube")
        self.assertIn("Print complete", out)
        self.assertNotIn("Shall I", out)

    def test_offer_empty_filename_generic_head(self):
        mod, _a = self._load()
        with mock.patch.object(mod, "_light_skill_available", return_value=True), \
             mock.patch.object(mod, "_timer_skill_available", return_value=False):
            out = mod._completion_offer_line("")   # no pretty filename
        self.assertTrue(out.startswith("Print complete, sir."))
        self.assertIn("dim the workshop lights", out)


class PrintCompanionPatternsTests(PrintCompanionMixin, unittest.TestCase):
    def test_record_outcome_persists_and_failure_rate(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        mod._print_start_ts[0] = time.time() - 600
        state = {"filename": "Gear_PLA.3mf", "total_layer": 150,
                 "print_error": 0}
        mod._record_print_outcome("success", state)
        mod._record_print_outcome("failed", dict(state))
        rate, count = mod._historical_failure_rate("pla", 150)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(rate, 0.5)

    def test_record_outcome_skips_midflight(self):
        mod, _a = self._load()
        mod._print_midflight[0] = True   # mid-flight discovery → no write
        mod._record_print_outcome("success", {"filename": "x_PLA.3mf",
                                              "total_layer": 100})
        rate, count = mod._historical_failure_rate("pla", 100)
        self.assertEqual(count, 0)

    def test_record_outcome_retention_cap(self):
        mod, _a = self._load()
        mod._PER_BUCKET_RETENTION = mod.PER_BUCKET_RETENTION  # use real cap
        mod._print_midflight[0] = False
        for _ in range(mod.PER_BUCKET_RETENTION + 10):
            mod._record_print_outcome("success",
                                      {"filename": "x_PLA.3mf", "total_layer": 100})
        data = mod._load_patterns()
        series = data["buckets"]["pla_100"]
        self.assertEqual(len(series), mod.PER_BUCKET_RETENTION)

    def test_historical_failure_rate_empty_bucket(self):
        mod, _a = self._load()
        self.assertEqual(mod._historical_failure_rate("abs", 999), (0.0, 0))

    def test_warn_historical_failure_fires_above_threshold(self):
        mod, _a = self._load()
        # Seed 4 records, 3 failed → 75% > 30% threshold, count >= 3.
        mod._print_midflight[0] = False
        for outcome in ("failed", "failed", "failed", "success"):
            mod._record_print_outcome(outcome,
                                      {"filename": "Gear_PLA.3mf",
                                       "total_layer": 150})
        mod._warned_historical_failure[0] = False
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._maybe_warn_historical_failure({"filename": "Gear_PLA.3mf",
                                               "total_layer": 150})
        enq.assert_called_once()
        self.assertIn("PLA", enq.call_args.args[0])
        self.assertIn("%", enq.call_args.args[0])

    def test_warn_historical_failure_silent_below_threshold(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        for outcome in ("success", "success", "success", "failed"):  # 25% < 30%
            mod._record_print_outcome(outcome,
                                      {"filename": "Gear_PLA.3mf",
                                       "total_layer": 150})
        mod._warned_historical_failure[0] = False
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._maybe_warn_historical_failure({"filename": "Gear_PLA.3mf",
                                               "total_layer": 150})
        enq.assert_not_called()

    def test_warn_historical_failure_silent_insufficient_history(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        # Only 1 record (< MIN_HISTORY_FOR_WARNING) even though it failed.
        mod._record_print_outcome("failed",
                                  {"filename": "Gear_PLA.3mf", "total_layer": 150})
        mod._warned_historical_failure[0] = False
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._maybe_warn_historical_failure({"filename": "Gear_PLA.3mf",
                                               "total_layer": 150})
        enq.assert_not_called()


class PrintCompanionAvailabilityTests(PrintCompanionMixin, unittest.TestCase):
    def test_light_available_requires_devices(self):
        mod, _a = self._load()
        hue = types.ModuleType("skill_sh_hue")
        hue.is_available = lambda: True
        hue.list_devices = lambda: [{"id": "bulb1"}]
        with mock.patch.dict(sys.modules, {"skill_sh_hue": hue}):
            self.assertTrue(mod._light_skill_available())
        # Available but zero devices → not available.
        hue.list_devices = lambda: []
        with mock.patch.dict(sys.modules, {"skill_sh_hue": hue,
                                           "skill_sh_govee": None}):
            self.assertFalse(mod._light_skill_available())

    def test_light_available_false_when_no_module(self):
        mod, _a = self._load()
        with mock.patch.dict(sys.modules,
                             {"skill_sh_hue": None, "skill_sh_govee": None}):
            self.assertFalse(mod._light_skill_available())

    def test_light_available_swallows_probe_errors(self):
        mod, _a = self._load()
        hue = types.ModuleType("skill_sh_hue")
        hue.is_available = lambda: True
        def _boom():
            raise RuntimeError("bridge offline")
        hue.list_devices = _boom
        with mock.patch.dict(sys.modules, {"skill_sh_hue": hue,
                                           "skill_sh_govee": None}):
            self.assertFalse(mod._light_skill_available())

    def test_timer_available_false_without_module(self):
        mod, _a = self._load()
        with mock.patch.dict(sys.modules, {"skill_timer": None}):
            self.assertFalse(mod._timer_skill_available())

    def test_vision_available_false_without_module(self):
        mod, _a = self._load()
        with mock.patch.dict(sys.modules, {"skill_local_vision": None}):
            self.assertFalse(mod._vision_available())


class PrintCompanionActionTests(PrintCompanionMixin, unittest.TestCase):
    def test_status_when_bambu_absent(self):
        mod, actions = self._load()  # no bambu
        out = actions["print_companion_status"]("")
        self.assertIn("monitor isn't running", out.lower())

    def test_status_armed_no_state(self):
        mod, actions = self._load(bambu_state={"last_update": 0.0})
        out = actions["print_companion_status"]("")
        self.assertIn("no fresh printer", out.lower())

    def test_status_tracking_running(self):
        mod, actions = self._load(bambu_state={
            "last_update": time.time(), "gcode_state": "RUNNING",
            "filename": "widget_PLA.3mf", "total_layer": 150})
        out = actions["print_companion_status"]("")
        self.assertIn("widget", out)
        self.assertIn("RUNNING", out)

    def test_history_empty(self):
        mod, actions = self._load()
        self.assertIn("No print history", actions["print_companion_history"](""))

    def test_history_renders_buckets(self):
        mod, actions = self._load()
        mod._print_midflight[0] = False
        mod._record_print_outcome("success",
                                  {"filename": "a_PLA.3mf", "total_layer": 100})
        mod._record_print_outcome("failed",
                                  {"filename": "b_PLA.3mf", "total_layer": 100})
        out = actions["print_companion_history"]("")
        self.assertIn("pla_100", out)
        self.assertIn("1 success / 1 failed", out)


class _RunNowThread:
    """Thread stand-in whose .start() runs the target synchronously, so a
    test can drive _sample_vision_async's worker without a real thread (and
    without racing the assertions). Mirrors the threading.Thread ctor kwargs
    the module uses (target/daemon/name)."""
    def __init__(self, target=None, daemon=None, name=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


# ─────────────────────────────────────────────────────────────────────────
# _strip_filename / small bambu-bridge helpers
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionStripFilenameTests(PrintCompanionMixin, unittest.TestCase):
    def test_strip_filename_local_fallback_no_bambu(self):
        mod, _a = self._load()  # bambu absent
        # Local fallback: basename, drop ext, collapse separators.
        self.assertEqual(mod._strip_filename("dir/My_Cool-Part.3mf"), "My Cool Part")
        self.assertEqual(mod._strip_filename(""), "")

    def test_strip_filename_local_fallback_truncates_to_60(self):
        mod, _a = self._load()
        long = "x" * 100 + ".gcode"
        self.assertEqual(len(mod._strip_filename(long)), 60)

    def test_strip_filename_prefers_bambu_stripper(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        self._fake._strip_filename = lambda n: "BAMBU-STRIPPED"
        self.assertEqual(mod._strip_filename("anything_PLA.3mf"), "BAMBU-STRIPPED")

    def test_strip_filename_bambu_raises_falls_back_to_local(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        def _boom(_n):
            raise RuntimeError("stripper down")
        self._fake._strip_filename = _boom
        # Falls through to the local regex path.
        self.assertEqual(mod._strip_filename("Gear_Box.3mf"), "Gear Box")

    def test_bambu_already_announced_true(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0}, announced={25})
        self.assertTrue(mod._bambu_already_announced(25))
        self.assertFalse(mod._bambu_already_announced(50))

    def test_bambu_already_announced_false_when_absent(self):
        mod, _a = self._load()  # bambu absent
        self.assertFalse(mod._bambu_already_announced(25))

    def test_bambu_already_announced_non_set_attr(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        self._fake._announced_milestones = "not-a-set"
        self.assertFalse(mod._bambu_already_announced(25))

    def test_bambu_already_announced_getattr_error_swallowed(self):
        # If reading bambu_monitor's _announced_milestones raises, the broad
        # except swallows it and the gate returns False (don't trail).
        mod, _a = self._load()

        class _Boom:
            def __getattr__(self, name):
                raise RuntimeError("module torn down")

        with mock.patch.object(mod, "_get_bambu_module", return_value=_Boom()):
            self.assertFalse(mod._bambu_already_announced(25))

    def test_read_state_swallows_lock_error(self):
        mod, _a = self._load(bambu_state={"last_update": 5.0})
        # A lock whose __enter__ raises → _read_state degrades to None.
        class _BadLock:
            def __enter__(self): raise RuntimeError("lock broken")
            def __exit__(self, *a): return False
        self._fake._state_lock = _BadLock()
        self.assertIsNone(mod._read_state())


# ─────────────────────────────────────────────────────────────────────────
# _infer_material remaining branches
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionInferMaterialBranchTests(PrintCompanionMixin, unittest.TestCase):
    def test_infer_material_abs_plus_and_carbon(self):
        mod, _a = self._load()
        self.assertEqual(mod._infer_material("Mount_ABS+_4h.3mf"), "absplus")
        self.assertEqual(mod._infer_material("Frame_CARBON.gcode"), "carbon")
        self.assertEqual(mod._infer_material("Toy_TPU.3mf"), "tpu")
        self.assertEqual(mod._infer_material("Spool_ABS.3mf"), "abs")


# ─────────────────────────────────────────────────────────────────────────
# _light / _timer / _vision availability — remaining branches
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionAvailabilityBranchTests(PrintCompanionMixin, unittest.TestCase):
    def test_light_available_is_available_false_skips(self):
        mod, _a = self._load()
        hue = types.ModuleType("skill_sh_hue")
        hue.is_available = lambda: False            # reports unavailable
        hue.list_devices = lambda: [{"id": "b"}]    # would-be device, never reached
        with mock.patch.dict(sys.modules, {"skill_sh_hue": hue,
                                           "skill_sh_govee": None}):
            self.assertFalse(mod._light_skill_available())

    def test_light_available_is_available_raises_skips(self):
        mod, _a = self._load()
        hue = types.ModuleType("skill_sh_hue")
        def _boom():
            raise RuntimeError("probe blew up")
        hue.is_available = _boom
        hue.list_devices = lambda: [{"id": "b"}]
        with mock.patch.dict(sys.modules, {"skill_sh_hue": hue,
                                           "skill_sh_govee": None}):
            self.assertFalse(mod._light_skill_available())

    def test_light_available_second_module_govee(self):
        mod, _a = self._load()
        govee = types.ModuleType("skill_sh_govee")
        govee.is_available = lambda: True
        govee.list_devices = lambda: [{"id": "strip"}]
        with mock.patch.dict(sys.modules, {"skill_sh_hue": None,
                                           "skill_sh_govee": govee}):
            self.assertTrue(mod._light_skill_available())

    def test_timer_available_true_with_actions_dict(self):
        mod, _a = self._load()
        timer = types.ModuleType("skill_timer")
        bc = types.ModuleType("bobert_companion")
        bc.ACTIONS = {"set_timer": lambda _s="": "ok"}
        with mock.patch.dict(sys.modules, {"skill_timer": timer,
                                           "bobert_companion": bc}):
            self.assertTrue(mod._timer_skill_available())

    def test_timer_available_true_module_only_fallback(self):
        mod, _a = self._load()
        timer = types.ModuleType("skill_timer")
        # No bobert_companion / __main__ ACTIONS exposed → module-presence path.
        with mock.patch.dict(sys.modules, {"skill_timer": timer,
                                           "bobert_companion": None,
                                           "__main__": types.ModuleType("__main__")}):
            self.assertTrue(mod._timer_skill_available())

    def test_vision_available_true_when_wired(self):
        mod, _a = self._load()
        lv = types.ModuleType("skill_local_vision")
        bc = types.ModuleType("bobert_companion")
        bc.LOCAL_VISION_FALLBACK = True
        bc.LOCAL_VISION_MODEL = "llava:7b"
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": bc}):
            self.assertTrue(mod._vision_available())

    def test_vision_available_false_no_bc(self):
        mod, _a = self._load()
        lv = types.ModuleType("skill_local_vision")
        # Both bobert_companion AND __main__ resolve to None → bc is None branch.
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": None,
                                           "__main__": None}):
            self.assertFalse(mod._vision_available())

    def test_vision_available_false_fallback_off(self):
        mod, _a = self._load()
        lv = types.ModuleType("skill_local_vision")
        bc = types.ModuleType("bobert_companion")
        bc.LOCAL_VISION_FALLBACK = False
        bc.LOCAL_VISION_MODEL = "llava:7b"
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": bc}):
            self.assertFalse(mod._vision_available())

    def test_vision_available_false_no_model(self):
        mod, _a = self._load()
        lv = types.ModuleType("skill_local_vision")
        bc = types.ModuleType("bobert_companion")
        bc.LOCAL_VISION_FALLBACK = True
        bc.LOCAL_VISION_MODEL = ""
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": bc}):
            self.assertFalse(mod._vision_available())


# ─────────────────────────────────────────────────────────────────────────
# _enqueue_speech — every fallback layer
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionEnqueueSpeechTests(PrintCompanionMixin, unittest.TestCase):
    def test_routes_through_bobert_proactive_announce(self):
        mod, _a = self._load()
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(mod.importlib, "import_module", return_value=bc):
            mod._enqueue_speech("hello sir")
        bc.proactive_announce.assert_called_once()
        self.assertEqual(bc.proactive_announce.call_args.args[0], "hello sir")
        self.assertEqual(bc.proactive_announce.call_args.kwargs.get("source"),
                         "print_companion")

    def test_falls_back_to_bambu_enqueue(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        self._fake._enqueue_speech = mock.MagicMock()
        # bobert import fails → bambu path taken.
        with mock.patch.object(mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")):
            mod._enqueue_speech("via bambu")
        self._fake._enqueue_speech.assert_called_once_with("via bambu")

    def test_falls_back_to_queue_file(self):
        mod, _a = self._load()  # bambu absent, no bobert
        qpath = os.path.join(self.tmpdir, "pending_speech.json")
        with mock.patch.object(mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")), \
             mock.patch.object(mod, "_PROJECT_DIR", self.tmpdir):
            mod._enqueue_speech("queued line")
        with open(qpath, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[-1]["message"], "queued line")

    def test_queue_file_appends_to_existing_list(self):
        mod, _a = self._load()
        qpath = os.path.join(self.tmpdir, "pending_speech.json")
        with open(qpath, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        with mock.patch.object(mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")), \
             mock.patch.object(mod, "_PROJECT_DIR", self.tmpdir):
            mod._enqueue_speech("new")
        with open(qpath, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["old", "new"])

    def test_queue_file_corrupt_resets_to_list(self):
        mod, _a = self._load()
        qpath = os.path.join(self.tmpdir, "pending_speech.json")
        with open(qpath, "w", encoding="utf-8") as f:
            f.write("{garbage not json")
        with mock.patch.object(mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")), \
             mock.patch.object(mod, "_PROJECT_DIR", self.tmpdir):
            mod._enqueue_speech("after-corrupt")
        with open(qpath, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, [{"ts": data[0]["ts"], "message": "after-corrupt"}])

    def test_bambu_enqueue_raises_falls_through_to_file(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        def _boom(_m):
            raise RuntimeError("bambu queue broken")
        self._fake._enqueue_speech = _boom
        qpath = os.path.join(self.tmpdir, "pending_speech.json")
        with mock.patch.object(mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")), \
             mock.patch.object(mod, "_PROJECT_DIR", self.tmpdir):
            mod._enqueue_speech("fell through")
        with open(qpath, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[-1]["message"], "fell through")

    def test_proactive_announce_not_callable_falls_through(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = "not-callable"
        self._fake._enqueue_speech = mock.MagicMock()
        with mock.patch.object(mod.importlib, "import_module", return_value=bc):
            mod._enqueue_speech("x")
        self._fake._enqueue_speech.assert_called_once_with("x")

    def test_queue_file_non_list_payload_reset(self):
        mod, _a = self._load()
        qpath = os.path.join(self.tmpdir, "pending_speech.json")
        with open(qpath, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)   # valid JSON, wrong type
        with mock.patch.object(mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")), \
             mock.patch.object(mod, "_PROJECT_DIR", self.tmpdir):
            mod._enqueue_speech("after-dict")
        with open(qpath, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, [{"ts": data[0]["ts"], "message": "after-dict"}])

    def test_queue_file_write_failure_swallowed(self):
        mod, _a = self._load()
        # Last-resort writer raises → caught and printed, never propagates.
        with mock.patch.object(mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")), \
             mock.patch.object(mod, "_PROJECT_DIR", self.tmpdir), \
             mock.patch.object(mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            mod._enqueue_speech("doomed")   # must not raise


# ─────────────────────────────────────────────────────────────────────────
# _sample_vision_async — worker behaviour (run synchronously)
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionVisionSampleTests(PrintCompanionMixin, unittest.TestCase):
    def _wire_vision(self, mod, describe):
        lv = types.ModuleType("skill_local_vision")
        lv.local_describe_screen = describe
        bc = types.ModuleType("bobert_companion")
        bc.LOCAL_VISION_FALLBACK = True
        bc.LOCAL_VISION_MODEL = "llava:7b"
        return lv, bc

    def test_no_op_when_vision_unavailable(self):
        mod, _a = self._load()
        with mock.patch.object(mod, "_vision_available", return_value=False), \
             mock.patch.object(mod.threading, "Thread") as T:
            mod._sample_vision_async(50)
        T.assert_not_called()  # never even spawns the worker
        self.assertEqual(mod._vision_samples, [])

    def test_nominal_reply_records_sample_no_warning(self):
        mod, _a = self._load()
        lv, bc = self._wire_vision(mod, lambda q: "Everything looks nominal.")
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": bc}), \
             mock.patch.object(mod.threading, "Thread", _RunNowThread), \
             mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._sample_vision_async(25)
        self.assertEqual(len(mod._vision_samples), 1)
        self.assertEqual(mod._vision_samples[0]["milestone"], 25)
        enq.assert_not_called()

    def test_failure_keyword_fires_warning(self):
        mod, _a = self._load()
        lv, bc = self._wire_vision(
            mod, lambda q: "I see stringy extrusion across the bed.")
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": bc}), \
             mock.patch.object(mod.threading, "Thread", _RunNowThread), \
             mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._sample_vision_async(75)
        enq.assert_called_once()
        self.assertIn("75%", enq.call_args.args[0])
        self.assertEqual(len(mod._vision_samples), 1)

    def test_empty_reply_records_nothing(self):
        mod, _a = self._load()
        lv, bc = self._wire_vision(mod, lambda q: "   ")
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": bc}), \
             mock.patch.object(mod.threading, "Thread", _RunNowThread), \
             mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._sample_vision_async(50)
        self.assertEqual(mod._vision_samples, [])
        enq.assert_not_called()

    def test_non_callable_describe_records_nothing(self):
        mod, _a = self._load()
        lv, bc = self._wire_vision(mod, "not-callable")
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": bc}), \
             mock.patch.object(mod.threading, "Thread", _RunNowThread):
            mod._sample_vision_async(50)
        self.assertEqual(mod._vision_samples, [])

    def test_describe_raises_is_swallowed(self):
        mod, _a = self._load()
        def _boom(_q):
            raise RuntimeError("ollama down")
        lv, bc = self._wire_vision(mod, _boom)
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": bc}), \
             mock.patch.object(mod.threading, "Thread", _RunNowThread):
            mod._sample_vision_async(50)  # must not raise
        self.assertEqual(mod._vision_samples, [])

    def test_excerpt_truncated_to_240(self):
        mod, _a = self._load()
        lv, bc = self._wire_vision(mod, lambda q: "ok " * 200)  # 600 chars
        with mock.patch.dict(sys.modules, {"skill_local_vision": lv,
                                           "bobert_companion": bc}), \
             mock.patch.object(mod.threading, "Thread", _RunNowThread), \
             mock.patch.object(mod, "_enqueue_speech"):
            mod._sample_vision_async(50)
        self.assertLessEqual(len(mod._vision_samples[0]["excerpt"]), 240)


# ─────────────────────────────────────────────────────────────────────────
# Completion-offer / failed transition handlers
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionTransitionTests(PrintCompanionMixin, unittest.TestCase):
    def test_completion_offer_skipped_if_never_running(self):
        mod, _a = self._load()
        mod._saw_running_this_print[0] = False
        with mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_record_print_outcome") as rec:
            mod._maybe_announce_completion_offer({"filename": "x_PLA.3mf"})
        enq.assert_not_called()
        rec.assert_not_called()
        self.assertFalse(mod._announced_completion_offer[0])

    def test_completion_offer_fires_and_records_success(self):
        mod, _a = self._load()
        mod._saw_running_this_print[0] = True
        mod._announced_completion_offer[0] = False
        with mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_completion_offer_line",
                               return_value="Print complete, sir.") as line, \
             mock.patch.object(mod, "_record_print_outcome") as rec:
            mod._maybe_announce_completion_offer({"filename": "Gear_PLA.3mf",
                                                  "total_layer": 100})
        enq.assert_called_once_with("Print complete, sir.")
        rec.assert_called_once_with("success", {"filename": "Gear_PLA.3mf",
                                                "total_layer": 100})
        self.assertTrue(mod._announced_completion_offer[0])
        line.assert_called_once()

    def test_completion_offer_idempotent_once_announced(self):
        mod, _a = self._load()
        mod._saw_running_this_print[0] = True
        mod._announced_completion_offer[0] = True   # already fired
        with mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_record_print_outcome") as rec:
            mod._maybe_announce_completion_offer({"filename": "x_PLA.3mf"})
        enq.assert_not_called()
        rec.assert_not_called()

    def test_completion_offer_uses_current_filename_fallback(self):
        mod, _a = self._load()
        mod._saw_running_this_print[0] = True
        mod._current_filename[0] = "Fallback_PLA.3mf"
        with mock.patch.object(mod, "_enqueue_speech"), \
             mock.patch.object(mod, "_record_print_outcome"), \
             mock.patch.object(mod, "_strip_filename",
                               return_value="Fallback") as strip:
            mod._maybe_announce_completion_offer({})  # no filename in state
        strip.assert_called_once_with("Fallback_PLA.3mf")

    def test_handle_failed_records_and_marks(self):
        mod, _a = self._load()
        mod._saw_running_this_print[0] = True
        mod._announced_completion_offer[0] = False
        with mock.patch.object(mod, "_record_print_outcome") as rec:
            mod._maybe_handle_failed({"filename": "Gear_PLA.3mf",
                                      "total_layer": 100, "print_error": 83})
        rec.assert_called_once_with("failed", {"filename": "Gear_PLA.3mf",
                                               "total_layer": 100,
                                               "print_error": 83})
        self.assertTrue(mod._announced_completion_offer[0])

    def test_handle_failed_skipped_if_never_running(self):
        mod, _a = self._load()
        mod._saw_running_this_print[0] = False
        with mock.patch.object(mod, "_record_print_outcome") as rec:
            mod._maybe_handle_failed({"filename": "x_PLA.3mf"})
        rec.assert_not_called()
        self.assertFalse(mod._announced_completion_offer[0])

    def test_handle_failed_idempotent(self):
        mod, _a = self._load()
        mod._saw_running_this_print[0] = True
        mod._announced_completion_offer[0] = True
        with mock.patch.object(mod, "_record_print_outcome") as rec:
            mod._maybe_handle_failed({"filename": "x_PLA.3mf"})
        rec.assert_not_called()

    def test_reset_per_print_state_clears_everything(self):
        mod, _a = self._load()
        # Dirty all the per-print globals.
        mod._print_started_pct[0] = 42.0
        mod._print_midflight[0] = True
        mod._saw_running_this_print[0] = True
        mod._milestone_detected_at[25] = 123.0
        mod._milestone_announced.add(50)
        mod._announced_completion_offer[0] = True
        mod._warned_historical_failure[0] = True
        mod._vision_samples.append({"x": 1})
        base = time.time()
        with mock.patch.object(mod.time, "time", return_value=base):
            mod._reset_per_print_state()
        self.assertEqual(mod._print_start_ts[0], base)
        self.assertIsNone(mod._print_started_pct[0])
        self.assertFalse(mod._print_midflight[0])
        self.assertFalse(mod._saw_running_this_print[0])
        self.assertEqual(mod._milestone_detected_at, {})
        self.assertEqual(mod._milestone_announced, set())
        self.assertFalse(mod._announced_completion_offer[0])
        self.assertFalse(mod._warned_historical_failure[0])
        self.assertEqual(mod._vision_samples, [])

    def test_warn_historical_already_warned_short_circuits(self):
        mod, _a = self._load()
        mod._warned_historical_failure[0] = True
        with mock.patch.object(mod, "_historical_failure_rate") as rate, \
             mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._maybe_warn_historical_failure({"filename": "x_PLA.3mf"})
        rate.assert_not_called()
        enq.assert_not_called()

    def test_warn_historical_layer_phrase_from_total(self):
        mod, _a = self._load()
        mod._warned_historical_failure[0] = False
        with mock.patch.object(mod, "_historical_failure_rate",
                               return_value=(0.5, 4)), \
             mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._maybe_warn_historical_failure({"filename": "Gear_PLA.3mf",
                                                "total_layer": 220})
        msg = enq.call_args.args[0]
        self.assertIn("220-layer", msg)
        self.assertTrue(mod._warned_historical_failure[0])

    def test_warn_historical_layer_phrase_when_total_bad(self):
        mod, _a = self._load()
        mod._warned_historical_failure[0] = False
        with mock.patch.object(mod, "_historical_failure_rate",
                               return_value=(0.9, 5)), \
             mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._maybe_warn_historical_failure({"filename": "Gear_PLA.3mf",
                                                "total_layer": "bogus"})
        self.assertIn("this size", enq.call_args.args[0])


# ─────────────────────────────────────────────────────────────────────────
# _record_print_outcome — extra branches (duration, fallback filename)
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionRecordOutcomeBranchTests(PrintCompanionMixin, unittest.TestCase):
    def test_duration_computed_from_start_ts(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        mod._print_start_ts[0] = 1000.0
        with mock.patch.object(mod.time, "time", return_value=1000.0 + 1800):
            mod._record_print_outcome("success",
                                      {"filename": "Gear_PLA.3mf",
                                       "total_layer": 100})
        rec = mod._load_patterns()["buckets"]["pla_100"][-1]
        self.assertEqual(rec["duration_min"], 30)   # 1800s / 60
        self.assertEqual(rec["outcome"], "success")

    def test_duration_none_when_no_start(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        mod._print_start_ts[0] = 0.0   # falsy → no duration
        mod._record_print_outcome("failed",
                                  {"filename": "Gear_PLA.3mf", "total_layer": 100})
        rec = mod._load_patterns()["buckets"]["pla_100"][-1]
        self.assertIsNone(rec["duration_min"])

    def test_uses_current_filename_when_state_lacks_one(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        mod._current_filename[0] = "Tracked_PETG.3mf"
        mod._record_print_outcome("success", {"total_layer": 300})
        # Material inferred from the fallback filename → petg bucket.
        data = mod._load_patterns()
        self.assertIn("petg_300", data["buckets"])

    def test_vision_samples_snapshotted_into_record(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        mod._vision_samples.append({"milestone": 50, "excerpt": "nominal"})
        mod._record_print_outcome("success",
                                  {"filename": "Gear_PLA.3mf", "total_layer": 100})
        rec = mod._load_patterns()["buckets"]["pla_100"][-1]
        self.assertEqual(rec["vision_samples"], [{"milestone": 50,
                                                  "excerpt": "nominal"}])


# ─────────────────────────────────────────────────────────────────────────
# _load_patterns / _save_patterns edge cases
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionPatternsIoTests(PrintCompanionMixin, unittest.TestCase):
    def test_load_patterns_missing_file(self):
        mod, _a = self._load()
        # tmp patterns file doesn't exist yet.
        self.assertEqual(mod._load_patterns(), {"buckets": {}})

    def test_load_patterns_corrupt_file(self):
        mod, _a = self._load()
        with open(self.patterns_path, "w", encoding="utf-8") as f:
            f.write("{ not valid json")
        self.assertEqual(mod._load_patterns(), {"buckets": {}})

    def test_load_patterns_wrong_shape(self):
        mod, _a = self._load()
        with open(self.patterns_path, "w", encoding="utf-8") as f:
            json.dump({"buckets": "not-a-dict"}, f)
        self.assertEqual(mod._load_patterns(), {"buckets": {}})

    def test_save_patterns_swallows_errors(self):
        mod, _a = self._load()
        with mock.patch.object(mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            mod._save_patterns({"buckets": {}})  # must not raise


# ─────────────────────────────────────────────────────────────────────────
# _poll_once_locked — the core state machine
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionPollTests(PrintCompanionMixin, unittest.TestCase):
    def _set_state(self, **kw):
        """Overwrite the fake bambu _state for the next poll."""
        base = {"last_update": time.time()}
        base.update(kw)
        self._fake._state = base

    def test_poll_no_bambu_is_noop(self):
        mod, _a = self._load()  # bambu absent → _read_state None
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._poll_once()
        enq.assert_not_called()

    def test_poll_stale_state_returns(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._poll_once()
        enq.assert_not_called()
        self.assertIsNone(mod._current_filename[0])

    def test_new_print_via_filename_resets_and_tracks(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        # Simulate a prior non-None gcode so this isn't treated as first-poll.
        mod._last_gcode_state[0] = "IDLE"
        self._set_state(gcode_state="RUNNING", filename="NewPart_PLA.3mf",
                        mc_percent=0.0, total_layer=120)
        with mock.patch.object(mod, "_maybe_warn_historical_failure") as warn:
            mod._poll_once()
        self.assertEqual(mod._current_filename[0], "NewPart_PLA.3mf")
        self.assertFalse(mod._print_midflight[0])    # pct 0 < threshold, prev set
        warn.assert_called_once()                    # fresh start → warn checked

    def test_first_poll_is_midflight_even_at_zero(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        # prev gcode None (cold boot) → always midflight, no warn.
        self.assertIsNone(mod._last_gcode_state[0])
        self._set_state(gcode_state="RUNNING", filename="Part_PLA.3mf",
                        mc_percent=0.0, total_layer=120)
        with mock.patch.object(mod, "_maybe_warn_historical_failure") as warn:
            mod._poll_once()
        self.assertTrue(mod._print_midflight[0])
        warn.assert_not_called()

    def test_new_print_midflight_premarks_passed_milestones(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        mod._last_gcode_state[0] = "IDLE"
        self._set_state(gcode_state="RUNNING", filename="Mid_PLA.3mf",
                        mc_percent=60.0, total_layer=120)
        with mock.patch.object(mod, "_maybe_warn_historical_failure") as warn:
            mod._poll_once()
        self.assertTrue(mod._print_midflight[0])             # 60 > 5 threshold
        self.assertEqual(mod._milestone_announced, {25, 50}) # 75 not yet passed
        warn.assert_not_called()                             # midflight → no warn

    def test_new_print_via_gcode_rotation_finish_to_running(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        # Same filename, but gcode rotates FINISH → RUNNING (reprint).
        mod._current_filename[0] = "Repeat_PLA.3mf"
        mod._last_gcode_state[0] = "FINISH"
        self._set_state(gcode_state="RUNNING", filename="Repeat_PLA.3mf",
                        mc_percent=1.0, total_layer=120)
        with mock.patch.object(mod, "_maybe_warn_historical_failure") as warn:
            mod._poll_once()
        # Treated as a new print → per-print state reset, warn evaluated.
        self.assertFalse(mod._print_midflight[0])
        warn.assert_called_once()

    def test_no_tracking_when_current_filename_none(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        mod._last_gcode_state[0] = "IDLE"
        # Empty filename and a non-new-print gcode → falls to the
        # "_current_filename is None" guard and returns.
        self._set_state(gcode_state="IDLE", filename="", mc_percent=None)
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._poll_once()
        enq.assert_not_called()
        self.assertIsNone(mod._current_filename[0])

    def test_running_sets_saw_running_flag(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        mod._current_filename[0] = "Active_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        mod._print_midflight[0] = True   # suppress milestone path
        self._set_state(gcode_state="RUNNING", filename="Active_PLA.3mf",
                        mc_percent=30.0, total_layer=120)
        mod._poll_once()
        self.assertTrue(mod._saw_running_this_print[0])

    def test_milestone_requires_bambu_announced_first(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0}, announced=set())
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        mod._print_midflight[0] = False
        self._set_state(gcode_state="RUNNING", filename="Part_PLA.3mf",
                        mc_percent=30.0, mc_remaining=60, total_layer=120)
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._poll_once()
        # bambu hasn't announced 25 yet → we don't even record detection.
        enq.assert_not_called()
        self.assertNotIn(25, mod._milestone_detected_at)

    def test_milestone_offset_two_poll_sequence(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0}, announced={25})
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        mod._print_midflight[0] = False
        self._set_state(gcode_state="RUNNING", filename="Part_PLA.3mf",
                        mc_percent=30.0, mc_remaining=60, total_layer=120)
        t0 = 5000.0
        # First poll: records detection time, no announcement yet.
        with mock.patch.object(mod.time, "time", return_value=t0), \
             mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_sample_vision_async"):
            mod._poll_once()
        enq.assert_not_called()
        self.assertIn(25, mod._milestone_detected_at)
        # Second poll, well past the offset → fires once + samples vision.
        with mock.patch.object(mod.time, "time",
                               return_value=t0 + mod.MILESTONE_OFFSET_SECONDS + 1), \
             mock.patch.object(mod, "_enqueue_speech") as enq2, \
             mock.patch.object(mod, "_sample_vision_async") as vis:
            mod._poll_once()
        enq2.assert_called_once()
        self.assertIn(25, mod._milestone_announced)
        vis.assert_called_once_with(25)

    def test_milestone_not_fired_before_offset_elapses(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0}, announced={25})
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        mod._print_midflight[0] = False
        mod._milestone_detected_at[25] = 5000.0   # detected already
        self._set_state(gcode_state="RUNNING", filename="Part_PLA.3mf",
                        mc_percent=30.0, mc_remaining=60, total_layer=120)
        with mock.patch.object(mod.time, "time", return_value=5000.0 + 5), \
             mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._poll_once()
        enq.assert_not_called()   # only 5s < 45s offset
        self.assertNotIn(25, mod._milestone_announced)

    def test_milestone_skipped_when_already_announced(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0}, announced={25})
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        mod._print_midflight[0] = False
        mod._milestone_announced.add(25)   # we already spoke 25
        self._set_state(gcode_state="RUNNING", filename="Part_PLA.3mf",
                        mc_percent=30.0, mc_remaining=60, total_layer=120)
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._poll_once()
        enq.assert_not_called()

    def test_milestone_suppressed_for_midflight_print(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0}, announced={25})
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        mod._print_midflight[0] = True    # midflight → milestone block skipped
        self._set_state(gcode_state="RUNNING", filename="Part_PLA.3mf",
                        mc_percent=30.0, mc_remaining=60, total_layer=120)
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._poll_once()
        enq.assert_not_called()

    def test_finish_dispatches_completion_offer(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        self._set_state(gcode_state="FINISH", filename="Part_PLA.3mf",
                        mc_percent=100.0, total_layer=120)
        with mock.patch.object(mod, "_maybe_announce_completion_offer") as off, \
             mock.patch.object(mod, "_maybe_handle_failed") as fail:
            mod._poll_once()
        off.assert_called_once()
        fail.assert_not_called()

    def test_failed_dispatches_failure_handler(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        self._set_state(gcode_state="FAILED", filename="Part_PLA.3mf",
                        mc_percent=42.0, total_layer=120, print_error=83)
        with mock.patch.object(mod, "_maybe_announce_completion_offer") as off, \
             mock.patch.object(mod, "_maybe_handle_failed") as fail:
            mod._poll_once()
        fail.assert_called_once()
        off.assert_not_called()

    def test_poll_bad_pct_value_tolerated(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        mod._print_midflight[0] = False
        # mc_percent unparseable → pct_f None → milestone block skipped, no crash.
        self._set_state(gcode_state="RUNNING", filename="Part_PLA.3mf",
                        mc_percent="??", mc_remaining=60, total_layer=120)
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._poll_once()
        enq.assert_not_called()

    def test_new_print_first_poll_prepare_not_midflight_branch(self):
        # prev_gcode None forces midflight True regardless of PREPARE/pct.
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        self._set_state(gcode_state="PREPARE", filename="Fresh_PLA.3mf",
                        mc_percent=None, total_layer=120)
        with mock.patch.object(mod, "_maybe_warn_historical_failure") as warn:
            mod._poll_once()
        self.assertTrue(mod._print_midflight[0])
        warn.assert_not_called()

    def test_milestone_full_integration_emits_real_line(self):
        # End-to-end with real _enqueue_speech routed to the queue file, to
        # exercise _milestone_commentary + the enqueue path together.
        mod, _a = self._load(bambu_state={"last_update": 0.0}, announced={25, 50})
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        mod._print_midflight[0] = False
        mod._milestone_detected_at[25] = 5000.0   # 25 already past offset window
        self._set_state(gcode_state="RUNNING", filename="Part_PLA.3mf",
                        mc_percent=55.0, mc_remaining=30, total_layer=120)
        with mock.patch.object(mod.time, "time",
                               return_value=5000.0 + mod.MILESTONE_OFFSET_SECONDS + 1), \
             mock.patch.object(mod, "_sample_vision_async"), \
             mock.patch.object(mod, "_PROJECT_DIR", self.tmpdir), \
             mock.patch.object(mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")):
            mod._poll_once()
        # 25 fires first (lowest passed milestone with detection recorded).
        qpath = os.path.join(self.tmpdir, "pending_speech.json")
        with open(qpath, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("Quarter of the way along", data[-1]["message"])
        self.assertIn(25, mod._milestone_announced)


# ─────────────────────────────────────────────────────────────────────────
# Poll loop / lifecycle / hook registration
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionLifecycleTests(PrintCompanionMixin, unittest.TestCase):
    def test_poll_loop_exits_immediately_on_initial_wait(self):
        mod, _a = self._load()
        with mock.patch.object(mod._stop_evt, "wait", return_value=True) as w, \
             mock.patch.object(mod, "_poll_once") as poll:
            mod._poll_loop()
        poll.assert_not_called()        # initial delay interrupted → never polls
        w.assert_called_once_with(mod.INITIAL_DELAY_SECONDS)

    def test_poll_loop_runs_one_iteration_then_stops(self):
        mod, _a = self._load()
        # First wait (initial delay) returns False → proceed; second wait
        # (poll interval) returns True → stop after exactly one _poll_once.
        waits = iter([False, True])
        with mock.patch.object(mod._stop_evt, "wait",
                               side_effect=lambda *_a: next(waits)), \
             mock.patch.object(mod._stop_evt, "is_set", return_value=False), \
             mock.patch.object(mod, "_poll_once") as poll:
            mod._poll_loop()
        poll.assert_called_once()

    def test_poll_loop_swallows_poll_exception(self):
        mod, _a = self._load()
        waits = iter([False, True])
        with mock.patch.object(mod._stop_evt, "wait",
                               side_effect=lambda *_a: next(waits)), \
             mock.patch.object(mod._stop_evt, "is_set", return_value=False), \
             mock.patch.object(mod, "_poll_once",
                               side_effect=RuntimeError("poll boom")):
            mod._poll_loop()   # must not propagate

    def test_start_poller_spawns_thread_and_stop_clears(self):
        mod, _a = self._load()
        mod._thread[0] = None
        started = {}
        # Use the synchronous thread shim but DON'T run the loop body.
        class _NoRunThread(_RunNowThread):
            def start(self):
                started["yes"] = True
                self.daemon_alive = True
            def is_alive(self):
                return True
        with mock.patch.object(mod.threading, "Thread", _NoRunThread):
            mod._start_poller()
        self.assertTrue(started.get("yes"))
        self.assertIsNotNone(mod._thread[0])
        mod.stop_companion()
        self.assertTrue(mod._stop_evt.is_set())
        self.assertIsNone(mod._thread[0])

    def test_start_poller_noop_when_thread_alive(self):
        mod, _a = self._load()
        live = mock.MagicMock()
        live.is_alive.return_value = True
        mod._thread[0] = live
        with mock.patch.object(mod.threading, "Thread") as T:
            mod._start_poller()
        T.assert_not_called()   # already running → no new thread

    def test_on_bambu_state_change_calls_poll(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        with mock.patch.object(mod, "_poll_once") as poll:
            mod._on_bambu_state_change({"gcode_state": "RUNNING"}, "IDLE", "RUNNING")
        poll.assert_called_once()

    def test_on_bambu_state_change_swallows_crash(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        with mock.patch.object(mod, "_poll_once",
                               side_effect=RuntimeError("hook boom")):
            mod._on_bambu_state_change({}, None, "RUNNING")  # must not raise

    def test_register_bambu_hook_success(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        self._fake.register_state_change_hook = mock.MagicMock()
        self.assertTrue(mod._register_bambu_hook())
        self._fake.register_state_change_hook.assert_called_with(
            mod._on_bambu_state_change)

    def test_register_bambu_hook_false_when_absent(self):
        mod, _a = self._load()  # bambu absent
        self.assertFalse(mod._register_bambu_hook())

    def test_register_bambu_hook_false_when_no_api(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        # Strip the hook API off the fake module.
        del self._fake.register_state_change_hook
        self.assertFalse(mod._register_bambu_hook())

    def test_register_bambu_hook_swallows_registration_error(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        def _boom(_cb):
            raise RuntimeError("registry full")
        self._fake.register_state_change_hook = _boom
        self.assertFalse(mod._register_bambu_hook())

    def test_register_wires_actions_and_hook(self):
        # register() ran during _load (threads neutered). Confirm both actions
        # exist and the bambu hook was registered.
        mod, actions = self._load(bambu_state={"last_update": 0.0})
        self.assertIn("print_companion_status", actions)
        self.assertIn("print_companion_history", actions)
        self._fake.register_state_change_hook.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────
# print_companion_status — history-bearing branch
# ─────────────────────────────────────────────────────────────────────────
class PrintCompanionStatusHistoryTests(PrintCompanionMixin, unittest.TestCase):
    def test_status_includes_failure_history(self):
        mod, actions = self._load(bambu_state={
            "last_update": time.time(), "gcode_state": "RUNNING",
            "filename": "widget_PLA.3mf", "total_layer": 150})
        # Seed enough failed history to surface the history clause.
        mod._print_midflight[0] = False
        for outcome in ("failed", "failed", "success"):
            mod._record_print_outcome(outcome, {"filename": "widget_PLA.3mf",
                                                "total_layer": 150})
        out = actions["print_companion_status"]("")
        self.assertIn("failure across", out)
        self.assertIn("3 similar prints", out)


if __name__ == "__main__":
    unittest.main()
