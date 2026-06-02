"""Logic tests for skills/night_owl_mode.py.

Targets the pure pieces and the toggle logic the spec calls out, without
relying on the live monolith (which the skill monkeypatches at runtime):
  • _in_night_window — the wrap-around 23:00–06:00 gate (time controlled).
  • _adjust_rate_string — slow an edge-tts rate string by N percentage points.
  • the TTS-preset wrapper installed by _install_tts_modifier — gain ×0.85 and
    rate −5pp, applied to a stub _resolve_tts_preset (no real TTS).
  • enter/exit state machine + is_night_owl_active().
  • the good_morning action (releases the mode when active; plain greeting else)
    and night_owl_status.

All side effects (TTS install/restore, nudge suppression, overlay dim, prompt
addendum, announcement enqueue) are patched so only state transitions run.
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
import types
import unittest
from unittest import mock
from datetime import datetime

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules (restoring prior
    state — including absence — on exit). Dotted-name leaves are ALSO set as an
    attribute on the already-imported parent package. Mirrors the helper in
    test_self_diagnostic.py so this file stays self-contained and isolated."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


@contextlib.contextmanager
def _neutered(mod):
    with mock.patch.object(mod, "_install_tts_modifier"), \
         mock.patch.object(mod, "_restore_tts_modifier"), \
         mock.patch.object(mod, "_install_nudge_suppressors"), \
         mock.patch.object(mod, "_restore_nudge_suppressors"), \
         mock.patch.object(mod, "_apply_prompt_addendum"), \
         mock.patch.object(mod, "_restore_prompt_addendum"), \
         mock.patch.object(mod, "_set_overlay_dim"), \
         mock.patch.object(mod, "_enqueue_speech"):
        yield


def _dt(hour):
    return datetime(2026, 6, 1, hour, 0, 0)


class NightOwlWindowTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")

    def test_in_window_late_night(self):
        self.assertTrue(self.mod._in_night_window(_dt(23)))
        self.assertTrue(self.mod._in_night_window(_dt(2)))
        self.assertTrue(self.mod._in_night_window(_dt(5)))

    def test_out_of_window_daytime(self):
        self.assertFalse(self.mod._in_night_window(_dt(6)))
        self.assertFalse(self.mod._in_night_window(_dt(12)))
        self.assertFalse(self.mod._in_night_window(_dt(22)))

    def test_in_window_defaults_to_now(self):
        # No arg → uses datetime.now(); patch it to a known late hour.
        fake_now = mock.MagicMock()
        fake_now.now.return_value = _dt(1)
        with mock.patch.object(self.mod, "datetime", fake_now):
            self.assertTrue(self.mod._in_night_window())

    def test_non_wraparound_config_branch(self):
        # Force START <= END so the non-wraparound code path is taken.
        with mock.patch.object(self.mod, "NIGHT_OWL_START_HOUR", 1), \
             mock.patch.object(self.mod, "NIGHT_OWL_END_HOUR", 5):
            self.assertTrue(self.mod._in_night_window(_dt(3)))
            self.assertFalse(self.mod._in_night_window(_dt(6)))
            self.assertFalse(self.mod._in_night_window(_dt(0)))


class NightOwlRateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")

    def test_adjust_rate_string_slows(self):
        self.assertEqual(self.mod._adjust_rate_string("+6%", 5), "+1%")
        self.assertEqual(self.mod._adjust_rate_string("-12%", 5), "-17%")
        self.assertEqual(self.mod._adjust_rate_string("+0%", 5), "-5%")

    def test_adjust_rate_string_handles_malformed(self):
        # Non-percent input → just the negative delta.
        self.assertEqual(self.mod._adjust_rate_string("fast", 5), "-5%")
        self.assertEqual(self.mod._adjust_rate_string("+x%", 5), "-5%")

    def test_adjust_rate_string_non_string_input(self):
        # A non-str rate (e.g. None) is coerced to "+0%" then slowed.
        self.assertEqual(self.mod._adjust_rate_string(None, 5), "-5%")
        self.assertEqual(self.mod._adjust_rate_string(7, 5), "-5%")


class NightOwlTtsWrapperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")
        self.mod._saved_resolve_preset[0] = None

    def test_wrapper_scales_gain_and_slows_rate(self):
        # Provide a fake bobert_companion with a stub _resolve_tts_preset, then
        # let _install_tts_modifier wrap it and inspect the wrapped output.
        import sys
        fake_bc = mock.MagicMock()
        fake_bc._resolve_tts_preset = lambda text, tone: (
            "base", {"rate": "+0%", "gain": 1.0})
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            with mock.patch.object(self.mod.importlib, "import_module",
                                   return_value=fake_bc):
                self.mod._install_tts_modifier()
                name, preset = fake_bc._resolve_tts_preset("hello", None)
        self.assertTrue(name.endswith("_nightowl"))
        self.assertAlmostEqual(preset["gain"], 0.85, places=3)
        self.assertEqual(preset["rate"], "-5%")

    def test_wrapper_idempotent_install(self):
        fake_bc = mock.MagicMock()
        fake_bc._resolve_tts_preset = lambda text, tone: ("base", {})
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=fake_bc):
            self.mod._install_tts_modifier()
            saved = self.mod._saved_resolve_preset[0]
            # A second install must not re-wrap (saved original unchanged).
            self.mod._install_tts_modifier()
        self.assertIs(self.mod._saved_resolve_preset[0], saved)

    def test_install_noop_when_bc_unimportable(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.mod._install_tts_modifier()
        self.assertIsNone(self.mod._saved_resolve_preset[0])

    def test_install_noop_when_resolve_missing(self):
        bc = types.ModuleType("bobert_companion")  # no _resolve_tts_preset
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=bc):
            self.mod._install_tts_modifier()
        self.assertIsNone(self.mod._saved_resolve_preset[0])

    def test_wrapper_falls_back_on_malformed_preset(self):
        # A preset whose gain can't be floated → wrapper returns the ORIGINAL
        # (name, preset) untouched so TTS is never muted.
        bc = types.ModuleType("bobert_companion")
        bc._resolve_tts_preset = lambda text, tone: (
            "base", {"rate": "+0%", "gain": object()})  # gain not floatable
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=bc):
            self.mod._install_tts_modifier()
            name, preset = bc._resolve_tts_preset("hi", None)
        self.assertEqual(name, "base")        # NOT suffixed → fallback taken
        self.assertNotIn("_nightowl", name)

    def test_restore_reverts_wrapper(self):
        bc = types.ModuleType("bobert_companion")
        orig = lambda text, tone: ("base", {"rate": "+0%", "gain": 1.0})
        bc._resolve_tts_preset = orig
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=bc):
            self.mod._install_tts_modifier()
            self.assertIsNot(bc._resolve_tts_preset, orig)   # wrapped
            self.mod._restore_tts_modifier()
        self.assertIs(bc._resolve_tts_preset, orig)          # restored
        self.assertIsNone(self.mod._saved_resolve_preset[0])

    def test_restore_noop_when_nothing_saved(self):
        bc = types.ModuleType("bobert_companion")
        bc._resolve_tts_preset = lambda t, u: ("x", {})
        unchanged = bc._resolve_tts_preset
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=bc):
            self.mod._restore_tts_modifier()   # saved is None → no-op
        self.assertIs(bc._resolve_tts_preset, unchanged)

    def test_restore_clears_saved_when_bc_unimportable(self):
        self.mod._saved_resolve_preset[0] = lambda t, u: ("x", {})
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("gone")):
            self.mod._restore_tts_modifier()
        self.assertIsNone(self.mod._saved_resolve_preset[0])

    def test_install_assignment_failure_rolls_back_saved(self):
        # If assigning the wrapper raises, the cached original is cleared so a
        # later restore won't re-wrap or strand the preset resolver.
        def _orig(text, tone):
            return ("base", {"rate": "+0%", "gain": 1.0})

        class _Locked(types.ModuleType):
            def __setattr__(self, k, v):
                if k == "_resolve_tts_preset" and v is not _orig:
                    raise RuntimeError("frozen module")
                super().__setattr__(k, v)
        bc = _Locked("bobert_companion")
        super(_Locked, bc).__setattr__("_resolve_tts_preset", _orig)
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=bc):
            self.mod._install_tts_modifier()   # logged, no raise
        self.assertIsNone(self.mod._saved_resolve_preset[0])

    def test_restore_assignment_failure_is_swallowed(self):
        def _orig(text, tone):
            return ("base", {})

        class _Locked(types.ModuleType):
            def __setattr__(self, k, v):
                if k == "_resolve_tts_preset":
                    raise RuntimeError("cannot restore")
                super().__setattr__(k, v)
        bc = _Locked("bobert_companion")
        self.mod._saved_resolve_preset[0] = _orig
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=bc):
            self.mod._restore_tts_modifier()   # logged, no raise
        # The finally-block still clears the saved handle.
        self.assertIsNone(self.mod._saved_resolve_preset[0])


class NightOwlStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")
        self.mod._night_owl_active[0] = False
        self.mod._trigger[0] = ""
        self.mod._started_at[0] = 0.0

    def test_enter_activates(self):
        with _neutered(self.mod):
            out = self.mod._enter_night_owl(trigger="manual")
        self.assertTrue(self.mod.is_night_owl_active())
        self.assertIn("engaged", out.lower())

    def test_enter_idempotent(self):
        with _neutered(self.mod):
            self.mod._enter_night_owl(trigger="manual")
            out = self.mod._enter_night_owl(trigger="auto")
        self.assertIn("already engaged", out.lower())

    def test_exit_deactivates(self):
        with _neutered(self.mod):
            self.mod._enter_night_owl(trigger="manual")
            out = self.mod._exit_night_owl(trigger="manual")
        self.assertFalse(self.mod.is_night_owl_active())
        self.assertIn("disengaged", out.lower())

    def test_exit_when_inactive(self):
        with _neutered(self.mod):
            out = self.mod._exit_night_owl(trigger="manual")
        self.assertIn("was not active", out.lower())

    def test_exit_good_morning_message(self):
        with _neutered(self.mod):
            self.mod._enter_night_owl(trigger="auto")
            out = self.mod._exit_night_owl(trigger="phrase_good_morning")
        self.assertIn("good morning", out.lower())


class NightOwlActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")
        self.mod._night_owl_active[0] = False
        self.mod._trigger[0] = ""
        self.mod._started_at[0] = 0.0

    def test_night_owl_on_off_actions(self):
        with _neutered(self.mod):
            on = self.actions["night_owl_on"]("")
            self.assertTrue(self.mod.is_night_owl_active())
            off = self.actions["night_owl_off"]("")
        self.assertIn("engaged", on.lower())
        self.assertIn("disengaged", off.lower())
        self.assertFalse(self.mod.is_night_owl_active())

    def test_good_morning_releases_active_mode(self):
        with _neutered(self.mod):
            self.actions["night_owl_on"]("")
            out = self.actions["good_morning"]("")
        self.assertIn("good morning", out.lower())
        self.assertFalse(self.mod.is_night_owl_active())

    def test_good_morning_plain_when_not_active(self):
        out = self.actions["good_morning"]("")
        self.assertEqual(out, "Good morning, sir.")

    def test_status_inactive(self):
        self.assertIn("not currently engaged",
                      self.actions["night_owl_status"]("").lower())

    def test_status_active_reports_trigger(self):
        with _neutered(self.mod):
            self.actions["night_owl_on"]("")
        out = self.actions["night_owl_status"]("")
        self.assertIn("engaged", out.lower())
        self.assertIn("by voice", out)   # manual trigger label

    def test_status_reports_hours_when_long_running(self):
        # Backdate started_at > 1h so the "Xh Ym" branch renders.
        with _neutered(self.mod):
            self.actions["night_owl_on"]("")
        self.mod._started_at[0] = self.mod.time.time() - (2 * 3600 + 15 * 60)
        out = self.actions["night_owl_status"]("")
        self.assertIn("2h 15m", out)

    def test_status_auto_trigger_label(self):
        with _neutered(self.mod):
            self.mod._enter_night_owl(trigger="auto")
        out = self.actions["night_owl_status"]("")
        self.assertIn("automatically", out)

    def test_register_wires_all_actions_and_aliases(self):
        for name in ("night_owl_on", "night_owl_mode", "enable_night_owl",
                     "night_owl_off", "end_night_owl", "disable_night_owl",
                     "good_morning", "night_owl_status"):
            self.assertIn(name, self.actions)
            self.assertTrue(callable(self.actions[name]))
        # The aliases point at the same underlying callables.
        self.assertIs(self.actions["night_owl_mode"], self.actions["night_owl_on"])
        self.assertIs(self.actions["end_night_owl"], self.actions["night_owl_off"])


# ─── _enqueue_speech (announcer route + atomic fallback) ─────────────────
class NightOwlEnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("night_owl_mode")

    def test_routes_through_proactive_announce_with_volume(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(return_value=True)
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("quiet line", volume_scale=0.7)
        bc.proactive_announce.assert_called_once()
        kw = bc.proactive_announce.call_args.kwargs
        self.assertEqual(kw.get("source"), "night_owl")
        self.assertAlmostEqual(kw.get("volume_scale"), 0.7)

    def test_fallback_atomic_write_carries_volume_scale(self):
        bc = types.ModuleType("bobert_companion")  # no announcer
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_atomic_write_json") as wj:
            self.mod._enqueue_speech("dim", volume_scale=0.5)
        written = wj.call_args[0][1][-1]
        self.assertEqual(written["message"], "dim")
        self.assertAlmostEqual(written["volume_scale"], 0.5)

    def test_fallback_default_volume_omits_field(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_atomic_write_json") as wj:
            self.mod._enqueue_speech("normal")   # volume_scale=1.0 default
        written = wj.call_args[0][1][-1]
        self.assertNotIn("volume_scale", written)

    def test_fallback_reads_existing_then_appends(self):
        import json
        import os
        bc = types.ModuleType("bobert_companion")
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w")
        json.dump([{"ts": 1.0, "message": "old"}], tmp)
        tmp.close()
        self.addCleanup(lambda: os.unlink(tmp.name))
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", tmp.name), \
             mock.patch.object(self.mod, "_atomic_write_json") as wj:
            self.mod._enqueue_speech("new")
        self.assertEqual([d["message"] for d in wj.call_args[0][1]],
                         ["old", "new"])

    def test_announcer_raises_then_falls_back(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(side_effect=RuntimeError("x"))
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_atomic_write_json") as wj:
            self.mod._enqueue_speech("delivered")
        wj.assert_called_once()

    def test_fallback_write_failure_is_swallowed(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk gone")):
            self.mod._enqueue_speech("doomed")   # no raise

    def test_fallback_corrupt_existing_queue_resets_to_list(self):
        import os
        bc = types.ModuleType("bobert_companion")
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w")
        tmp.write("{not valid json")   # unparseable → except branch → data=[]
        tmp.close()
        self.addCleanup(lambda: os.unlink(tmp.name))
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", tmp.name), \
             mock.patch.object(self.mod, "_atomic_write_json") as wj:
            self.mod._enqueue_speech("fresh")
        # Corrupt prior content discarded; only our message remains.
        self.assertEqual([d["message"] for d in wj.call_args[0][1]], ["fresh"])


# ─── overlay dim (HUD state publish) ─────────────────────────────────────
class NightOwlOverlayDimTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("night_owl_mode")

    def test_publishes_dim_on(self):
        bc = types.ModuleType("bobert_companion")
        bc._write_hud_state = mock.MagicMock()
        with inject_modules(bobert_companion=bc):
            self.mod._set_overlay_dim(True)
        bc._write_hud_state.assert_called_once_with(
            night_owl_dim=self.mod.NIGHT_OWL_OVERLAY_DIM)

    def test_publishes_dim_off(self):
        bc = types.ModuleType("bobert_companion")
        bc._write_hud_state = mock.MagicMock()
        with inject_modules(bobert_companion=bc):
            self.mod._set_overlay_dim(False)
        bc._write_hud_state.assert_called_once_with(night_owl_dim=0.0)

    def test_noop_when_bc_unimportable(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.mod._set_overlay_dim(True)   # no raise

    def test_noop_when_writer_missing(self):
        bc = types.ModuleType("bobert_companion")  # no _write_hud_state
        with inject_modules(bobert_companion=bc):
            self.mod._set_overlay_dim(True)   # no raise

    def test_writer_exception_is_swallowed(self):
        bc = types.ModuleType("bobert_companion")
        bc._write_hud_state = mock.MagicMock(side_effect=RuntimeError("locked"))
        with inject_modules(bobert_companion=bc):
            self.mod._set_overlay_dim(True)   # logged, no raise


# ─── nudge suppression (critical allow-list) ─────────────────────────────
class NightOwlNudgeSuppressionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("night_owl_mode")
        self.mod._saved_enqueues.clear()
        self.addCleanup(self.mod._saved_enqueues.clear)

    def _fake_skill(self, name):
        # Real function, not MagicMock — so getattr(fn, "_is_nudge_sink",
        # False) is genuinely falsy.
        m = types.ModuleType(f"skill_{name}")

        def _orig(message, *args, **kwargs):
            pass
        m._enqueue_speech = _orig
        return m

    def test_suppresses_listed_only_and_spares_critical(self):
        fakes = {f"skill_{s}": self._fake_skill(s)
                 for s in self.mod.SUPPRESSED_SKILLS}
        # A critical announcer NOT in the allow-list (e.g. bambu_monitor).
        critical = self._fake_skill("bambu_monitor")
        original_critical = critical._enqueue_speech
        fakes["skill_bambu_monitor"] = critical
        with inject_modules(**fakes):
            self.mod._install_nudge_suppressors()
            for s in self.mod.SUPPRESSED_SKILLS:
                sink = sys.modules[f"skill_{s}"]._enqueue_speech
                self.assertTrue(getattr(sink, "_is_nudge_sink", False))
            # VIP/critical announcer untouched.
            self.assertIs(critical._enqueue_speech, original_critical)
            # The sink accepts extra args/kwargs (night_owl sink signature) and
            # drops silently.
            sink = sys.modules[f"skill_{self.mod.SUPPRESSED_SKILLS[0]}"]._enqueue_speech
            sink("a line", 0.8)
            self.mod._restore_nudge_suppressors()
            for s in self.mod.SUPPRESSED_SKILLS:
                restored = sys.modules[f"skill_{s}"]._enqueue_speech
                self.assertFalse(getattr(restored, "_is_nudge_sink", False))

    def test_missing_module_skipped(self):
        for s in self.mod.SUPPRESSED_SKILLS:
            sys.modules.pop(f"skill_{s}", None)
        self.mod._install_nudge_suppressors()
        self.assertEqual(self.mod._saved_enqueues, {})

    def test_non_callable_skipped(self):
        s0 = self.mod.SUPPRESSED_SKILLS[0]
        m = types.ModuleType(f"skill_{s0}")
        m._enqueue_speech = 123
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._install_nudge_suppressors()
        self.assertNotIn(s0, self.mod._saved_enqueues)

    def test_existing_sink_not_owned(self):
        s0 = self.mod.SUPPRESSED_SKILLS[0]
        m = types.ModuleType(f"skill_{s0}")

        def _other(msg, *a, **k):
            pass
        _other._is_nudge_sink = True
        m._enqueue_speech = _other
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._install_nudge_suppressors()
            self.assertIsNone(self.mod._saved_enqueues[s0])
            self.mod._restore_nudge_suppressors()
            self.assertIs(m._enqueue_speech, _other)   # left in place

    def test_double_install_no_rewrap(self):
        s0 = self.mod.SUPPRESSED_SKILLS[0]
        m = self._fake_skill(s0)
        real_original = m._enqueue_speech
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._install_nudge_suppressors()
            first = m._enqueue_speech
            self.mod._install_nudge_suppressors()
            self.assertIs(m._enqueue_speech, first)
            self.assertIs(self.mod._saved_enqueues[s0], real_original)

    def test_restore_skips_missing_module(self):
        s0 = self.mod.SUPPRESSED_SKILLS[0]
        self.mod._saved_enqueues[s0] = lambda m: None
        sys.modules.pop(f"skill_{s0}", None)
        self.mod._restore_nudge_suppressors()   # no raise
        self.assertEqual(self.mod._saved_enqueues, {})

    def test_wrap_assignment_failure_pops_saved(self):
        s0 = self.mod.SUPPRESSED_SKILLS[0]

        class _Locked(types.ModuleType):
            def __setattr__(self, k, v):
                if k == "_enqueue_speech" and getattr(v, "_is_nudge_sink", False):
                    raise RuntimeError("frozen")
                super().__setattr__(k, v)
        m = _Locked(f"skill_{s0}")
        super(_Locked, m).__setattr__("_enqueue_speech", lambda msg: None)
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._install_nudge_suppressors()
        self.assertNotIn(s0, self.mod._saved_enqueues)

    def test_restore_assignment_failure_is_swallowed(self):
        s0 = self.mod.SUPPRESSED_SKILLS[0]

        class _Locked(types.ModuleType):
            def __setattr__(self, k, v):
                if k == "_enqueue_speech":
                    raise RuntimeError("cannot restore")
                super().__setattr__(k, v)
        m = _Locked(f"skill_{s0}")
        self.mod._saved_enqueues[s0] = lambda msg: None
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._restore_nudge_suppressors()   # no raise
        self.assertEqual(self.mod._saved_enqueues, {})


# ─── prompt addendum ─────────────────────────────────────────────────────
class NightOwlPromptAddendumTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("night_owl_mode")
        self.mod._saved_system_prompt[0] = None
        self.addCleanup(lambda: self.mod._saved_system_prompt.__setitem__(0, None))

    def test_apply_then_restore_roundtrip(self):
        bc = types.ModuleType("bobert_companion")
        bc._system_prompt = "BASE"
        with inject_modules(bobert_companion=bc):
            self.mod._apply_prompt_addendum()
            self.assertIn("[Night-owl mode]", bc._system_prompt)
            self.assertTrue(bc._system_prompt.startswith("BASE"))
            self.mod._restore_prompt_addendum()
            self.assertEqual(bc._system_prompt, "BASE")
            self.assertIsNone(self.mod._saved_system_prompt[0])

    def test_apply_import_failure_silent(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.mod._apply_prompt_addendum()
        self.assertIsNone(self.mod._saved_system_prompt[0])

    def test_apply_assignment_failure_is_swallowed(self):
        class _Locked(types.ModuleType):
            def __setattr__(self, k, v):
                if k == "_system_prompt" and v != "BASE":
                    raise RuntimeError("frozen")
                super().__setattr__(k, v)
        bc = _Locked("bobert_companion")
        super(_Locked, bc).__setattr__("_system_prompt", "BASE")
        with inject_modules(bobert_companion=bc):
            self.mod._apply_prompt_addendum()   # logged, no raise

    def test_restore_noop_when_nothing_saved(self):
        bc = types.ModuleType("bobert_companion")
        bc._system_prompt = "UNCHANGED"
        with inject_modules(bobert_companion=bc):
            self.mod._restore_prompt_addendum()
        self.assertEqual(bc._system_prompt, "UNCHANGED")

    def test_restore_import_failure_silent(self):
        self.mod._saved_system_prompt[0] = "X"
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("gone")):
            self.mod._restore_prompt_addendum()   # no raise


# ─── watcher loop (auto engage/disengage at the window edges) ────────────
class NightOwlWatchLoopTests(unittest.TestCase):
    """The watcher is an infinite loop; we let it run exactly ONE iteration by
    making the trailing sleep raise KeyboardInterrupt (a BaseException the
    loop's `except Exception` does not catch), then assert which transition
    fired. No real sleeping or threads."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("night_owl_mode")
        self.mod._night_owl_active[0] = False
        self.mod._trigger[0] = ""

    @contextlib.contextmanager
    def _one_iteration(self):
        # sleep call #1 is the 8s startup delay; call #2 is the in-loop
        # WATCH_INTERVAL sleep — raise there to exit after one pass.
        state = {"n": 0}

        def _sleep(_secs):
            state["n"] += 1
            if state["n"] >= 2:
                raise KeyboardInterrupt
        with mock.patch.object(self.mod.time, "sleep", side_effect=_sleep):
            yield

    def test_auto_engages_inside_window(self):
        with mock.patch.object(self.mod, "_in_night_window", return_value=True), \
             mock.patch.object(self.mod, "_enter_night_owl") as ent, \
             mock.patch.object(self.mod, "_exit_night_owl") as ext, \
             self._one_iteration():
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_loop()
        ent.assert_called_once_with(trigger="auto")
        ext.assert_not_called()

    def test_auto_disengages_outside_window_when_auto_triggered(self):
        self.mod._night_owl_active[0] = True
        self.mod._trigger[0] = "auto"
        with mock.patch.object(self.mod, "_in_night_window", return_value=False), \
             mock.patch.object(self.mod, "is_night_owl_active", return_value=True), \
             mock.patch.object(self.mod, "_enter_night_owl") as ent, \
             mock.patch.object(self.mod, "_exit_night_owl") as ext, \
             self._one_iteration():
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_loop()
        ext.assert_called_once_with(trigger="auto_morning")
        ent.assert_not_called()

    def test_auto_disengages_stale_manual_session(self):
        # Manual engagement that the clock rolled past 06:00 → still released.
        self.mod._night_owl_active[0] = True
        self.mod._trigger[0] = "manual"
        with mock.patch.object(self.mod, "_in_night_window", return_value=False), \
             mock.patch.object(self.mod, "is_night_owl_active", return_value=True), \
             mock.patch.object(self.mod, "_exit_night_owl") as ext, \
             self._one_iteration():
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_loop()
        ext.assert_called_once_with(trigger="auto_morning")

    def test_no_transition_when_inside_window_already_active(self):
        self.mod._night_owl_active[0] = True
        self.mod._trigger[0] = "manual"
        with mock.patch.object(self.mod, "_in_night_window", return_value=True), \
             mock.patch.object(self.mod, "is_night_owl_active", return_value=True), \
             mock.patch.object(self.mod, "_enter_night_owl") as ent, \
             mock.patch.object(self.mod, "_exit_night_owl") as ext, \
             self._one_iteration():
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_loop()
        ent.assert_not_called()
        ext.assert_not_called()

    def test_iteration_exception_is_logged_and_loop_continues(self):
        # First tick: _in_night_window raises → caught, logged, recovery sleep.
        # That recovery sleep is call #2 → KeyboardInterrupt exits the loop.
        with mock.patch.object(self.mod, "_in_night_window",
                               side_effect=RuntimeError("clock boom")), \
             mock.patch.object(self.mod.logging, "exception") as logexc, \
             self._one_iteration():
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_loop()
        logexc.assert_called()


# ─── enter/exit side-effect wiring + announce toggles ────────────────────
class NightOwlEnterExitWiringTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("night_owl_mode")
        self.mod._night_owl_active[0] = False
        self.mod._trigger[0] = ""
        self.mod._started_at[0] = 0.0

    def test_enter_invokes_all_side_effects(self):
        with mock.patch.object(self.mod, "_install_tts_modifier") as tts, \
             mock.patch.object(self.mod, "_install_nudge_suppressors") as nud, \
             mock.patch.object(self.mod, "_apply_prompt_addendum") as pr, \
             mock.patch.object(self.mod, "_set_overlay_dim") as dim, \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._enter_night_owl(trigger="manual")
        tts.assert_called_once()
        nud.assert_called_once()
        pr.assert_called_once()
        dim.assert_called_once_with(True)
        enq.assert_called_once()

    def test_enter_auto_uses_late_night_copy(self):
        with mock.patch.object(self.mod, "_install_tts_modifier"), \
             mock.patch.object(self.mod, "_install_nudge_suppressors"), \
             mock.patch.object(self.mod, "_apply_prompt_addendum"), \
             mock.patch.object(self.mod, "_set_overlay_dim"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._enter_night_owl(trigger="auto")
        spoken = enq.call_args[0][0]
        self.assertIn("past 11", spoken)

    def test_enter_announce_false_suppresses_speech(self):
        with mock.patch.object(self.mod, "_install_tts_modifier"), \
             mock.patch.object(self.mod, "_install_nudge_suppressors"), \
             mock.patch.object(self.mod, "_apply_prompt_addendum"), \
             mock.patch.object(self.mod, "_set_overlay_dim"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._enter_night_owl(trigger="manual", announce=False)
        enq.assert_not_called()

    def test_exit_invokes_restores_and_dim_off(self):
        with _neutered(self.mod):
            self.mod._enter_night_owl(trigger="manual")
        with mock.patch.object(self.mod, "_restore_tts_modifier") as tts, \
             mock.patch.object(self.mod, "_restore_nudge_suppressors") as nud, \
             mock.patch.object(self.mod, "_restore_prompt_addendum") as pr, \
             mock.patch.object(self.mod, "_set_overlay_dim") as dim, \
             mock.patch.object(self.mod, "_enqueue_speech"):
            self.mod._exit_night_owl(trigger="manual")
        tts.assert_called_once()
        nud.assert_called_once()
        pr.assert_called_once()
        dim.assert_called_once_with(False)

    def test_exit_announce_false_returns_plain_line(self):
        with _neutered(self.mod):
            self.mod._enter_night_owl(trigger="manual")
            out = self.mod._exit_night_owl(trigger="auto_morning", announce=False)
        self.assertEqual(out, "Night-owl mode disengaged, sir.")

    def test_exit_auto_morning_announces_good_morning(self):
        with mock.patch.object(self.mod, "_restore_tts_modifier"), \
             mock.patch.object(self.mod, "_restore_nudge_suppressors"), \
             mock.patch.object(self.mod, "_restore_prompt_addendum"), \
             mock.patch.object(self.mod, "_set_overlay_dim"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._night_owl_active[0] = True
            self.mod._trigger[0] = "auto"
            out = self.mod._exit_night_owl(trigger="auto_morning")
        self.assertIn("Good morning", out)
        self.assertIn("Good morning", enq.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
