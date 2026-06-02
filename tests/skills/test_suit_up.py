"""Logic tests for skills/suit_up.py.

Targets the cinematic boot logic without real TTS, HUD, overlay, or audio
hardware. Every collaborator the skill reaches lazily is faked and removed
again after the test:
  • `memory` (session-end probe) — injected into sys.modules, restored on exit.
  • `bobert_companion` (speaker-name lookup, _speak/_write_hud_state for the
    manual action) — injected, restored. Never the real monolith.
  • `skill_holographic_overlay` (overlay coordination) — injected, restored.
sounddevice is NOT on the CI runner; the skill only touches it via
`_resolve_speaker_name`'s bobert_companion fallback, which we drive with a fake
bobert_companion (or block) so sounddevice is never imported here.

Coverage:
  • _build_diagnostic_lines — the four-line readout, with/without a speaker.
  • _resolve_speaker_name — explicit passthrough, the bobert_companion friendly-
    name path (incl. the "[N] Desc, API" strip), and the fallback.
  • _load_state / _save_state — round-trip, missing-file, corrupt-file, and the
    write path (redirected to a temp file).
  • _last_session_end_ts — ts / iso / date row shapes, empty, and no-memory.
  • _is_warm_restart — the 18-hour window gate (time controlled).
  • _ensure_holo_overlay_up / _dismiss_holo_overlay — launch-when-down,
    skip-when-already-up, missing-module, and error swallowing.
  • _start_animation / _clear_animation — HUD writes, the no-writer short
    circuit, and the elapsed-time hold (time.sleep patched).
  • play_suit_up_sequence — speaks all lines in order, returns the welcome,
    dismisses the overlay only when it launched it, survives a throwing speak.
  • maybe_play_morning_suit_up — the (warm-restart AND not-fired-today) gate.
  • _act_suit_up — the verbal trigger incl. the no-bobert and no-TTS error arms.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install/remove fake modules in ``sys.modules`` (e.g.
    ``memory``, ``bobert_companion``, ``skill_holographic_overlay``) for the
    duration of a block, restoring prior state — including absence — afterwards.
    ``obj=None`` removes the module so a lazy import misses. All names used here
    are flat (non-dotted)."""
    saved: dict[str, object] = {}
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def make_fake_memory(summaries):
    mod = types.ModuleType("memory")
    mod.get_session_summaries = lambda *a, **k: summaries
    return mod


def make_fake_overlay(*, alive=False, launch=(True, "ok"),
                      alive_raises=False, launch_raises=False,
                      shutdown_raises=False):
    """Fake `skill_holographic_overlay` module exposing the private hooks
    suit_up consults: _overlay_is_alive / _launch_overlay / _shutdown_overlay."""
    mod = types.ModuleType("skill_holographic_overlay")
    calls = {"launch": 0, "shutdown": 0}

    def _is_alive():
        if alive_raises:
            raise RuntimeError("alive boom")
        return alive

    def _launch():
        calls["launch"] += 1
        if launch_raises:
            raise RuntimeError("launch boom")
        return launch

    def _shutdown():
        calls["shutdown"] += 1
        if shutdown_raises:
            raise RuntimeError("shutdown boom")

    mod._overlay_is_alive = _is_alive
    mod._launch_overlay = _launch
    mod._shutdown_overlay = _shutdown
    mod._calls = calls
    return mod


# ─── diagnostic-line builder + speaker resolution ────────────────────────
class SuitUpBuilderTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")

    def test_diagnostic_lines_with_speaker(self):
        lines = self.mod._build_diagnostic_lines("Gaming Headset")
        self.assertEqual(lines[0], "Diagnostics: nominal.")
        self.assertEqual(lines[1], "Network: online.")
        self.assertIn("Gaming Headset", lines[2])
        self.assertEqual(lines[3], "Workshop: standing by.")

    def test_diagnostic_lines_without_speaker(self):
        lines = self.mod._build_diagnostic_lines("")
        self.assertEqual(lines[2], "Audio: connected.")

    def test_resolve_speaker_explicit(self):
        self.assertEqual(self.mod._resolve_speaker_name("My Headset"),
                         "My Headset")

    def test_resolve_speaker_blank_explicit_falls_back(self):
        # No explicit name and bobert_companion lookup fails → "system default".
        with inject_modules(bobert_companion=None):
            self.assertEqual(self.mod._resolve_speaker_name(""), "system default")

    def test_resolve_speaker_via_bobert_friendly_name(self):
        # bobert_companion returns a raw "[N] Desc, API" string which the skill
        # strips down before handing to _friendly_device_name.
        bc = types.ModuleType("bobert_companion")
        seen = {}

        def _friendly(raw):
            seen["raw"] = raw
            return "Gaming Headset"
        bc.get_current_speaker_name = lambda: "[3] USB Audio Device, MME"
        bc._friendly_device_name = _friendly
        with inject_modules(bobert_companion=bc):
            out = self.mod._resolve_speaker_name(None)
        self.assertEqual(out, "Gaming Headset")
        # The "[3] ... , API" prefix was stripped before _friendly saw it.
        self.assertEqual(seen["raw"], "USB Audio Device, MME")

    def test_resolve_speaker_friendly_returns_blank_falls_back(self):
        bc = types.ModuleType("bobert_companion")
        bc.get_current_speaker_name = lambda: "Plain Name"
        bc._friendly_device_name = lambda raw: ""   # nothing usable
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._resolve_speaker_name(None), "system default")

    def test_resolve_speaker_bobert_raises_falls_back(self):
        bc = types.ModuleType("bobert_companion")

        def _boom():
            raise RuntimeError("no audio subsystem")
        bc.get_current_speaker_name = _boom
        bc._friendly_device_name = lambda raw: raw
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._resolve_speaker_name(None), "system default")

    def test_resolve_speaker_whitespace_explicit_falls_back(self):
        # An explicit but whitespace-only name is treated as no name → probe.
        with inject_modules(bobert_companion=None):
            self.assertEqual(self.mod._resolve_speaker_name("   "), "system default")


# ─── state-file IO ───────────────────────────────────────────────────────
class SuitUpStateIoTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")
        fd, self.statep = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.statep)   # start with no file
        self._patch = mock.patch.object(self.mod, "_STATE_FILE", self.statep)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._unlink)

    def _unlink(self):
        try:
            os.unlink(self.statep)
        except OSError:
            pass

    def test_load_state_missing_file_returns_empty(self):
        self.assertEqual(self.mod._load_state(), {})

    def test_save_then_load_round_trips(self):
        self.mod._save_state({"last_fired_date": "2026-06-01", "n": 3})
        self.assertTrue(os.path.exists(self.statep))
        self.assertEqual(self.mod._load_state(),
                         {"last_fired_date": "2026-06-01", "n": 3})

    def test_load_state_corrupt_json_returns_empty(self):
        with open(self.statep, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertEqual(self.mod._load_state(), {})

    def test_load_state_null_payload_returns_empty(self):
        # A file literally containing `null` → json.load yields None → {}.
        with open(self.statep, "w", encoding="utf-8") as f:
            f.write("null")
        self.assertEqual(self.mod._load_state(), {})

    def test_save_state_swallows_write_error(self):
        # mkstemp inside _save_state raising must not propagate.
        with mock.patch.object(self.mod.tempfile, "mkstemp",
                               side_effect=OSError("disk full")):
            self.mod._save_state({"x": 1})   # must not raise

    def test_save_state_unlinks_temp_when_dump_fails(self):
        # mkstemp SUCCEEDS but json.dump raises (non-serialisable payload), so
        # the inner except runs os.unlink(tmp) then re-raises into the outer
        # swallow. Assert the temp file was cleaned up and the real state file
        # was never created. A set is not JSON-serialisable.
        with mock.patch.object(self.mod.os, "unlink") as unlink:
            self.mod._save_state({"bad": {1, 2, 3}})   # must not raise
        unlink.assert_called_once()                    # temp file removed
        self.assertFalse(os.path.exists(self.statep))  # no half-written state

    def test_save_state_unlink_failure_also_swallowed(self):
        # Even if the cleanup os.unlink itself raises, nothing propagates.
        with mock.patch.object(self.mod.os, "unlink",
                               side_effect=OSError("locked")):
            self.mod._save_state({"bad": {1, 2, 3}})   # must not raise


# ─── _last_session_end_ts (memory probe) ─────────────────────────────────
class SuitUpSessionEndTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")

    def test_session_end_from_numeric_ts(self):
        with inject_modules(memory=make_fake_memory([{"ts": 1_700_000_000}])):
            self.assertEqual(self.mod._last_session_end_ts(), 1_700_000_000.0)

    def test_session_end_from_iso_end(self):
        with inject_modules(memory=make_fake_memory(
                [{"iso_end": "2026-05-30T20:00:00"}])):
            ts = self.mod._last_session_end_ts()
        self.assertEqual(ts, time.mktime(
            time.strptime("2026-05-30T20:00:00", "%Y-%m-%dT%H:%M:%S")))

    def test_session_end_from_date_only(self):
        with inject_modules(memory=make_fake_memory([{"date": "2026-05-30"}])):
            ts = self.mod._last_session_end_ts()
        self.assertEqual(ts, time.mktime(
            time.strptime("2026-05-30T20:00:00", "%Y-%m-%dT%H:%M:%S")))

    def test_session_end_empty_history_is_zero(self):
        with inject_modules(memory=make_fake_memory([])):
            self.assertEqual(self.mod._last_session_end_ts(), 0.0)

    def test_session_end_row_without_usable_fields_is_zero(self):
        with inject_modules(memory=make_fake_memory([{"unrelated": "x"}])):
            self.assertEqual(self.mod._last_session_end_ts(), 0.0)

    def test_session_end_bad_ts_then_iso_fallback(self):
        # A non-numeric ts is skipped; the iso_start field is used instead.
        row = [{"ts": "notanumber", "iso_start": "2026-05-30T08:00:00"}]
        with inject_modules(memory=make_fake_memory(row)):
            ts = self.mod._last_session_end_ts()
        self.assertEqual(ts, time.mktime(
            time.strptime("2026-05-30T08:00:00", "%Y-%m-%dT%H:%M:%S")))

    def test_session_end_malformed_iso_is_zero(self):
        # iso present but unparseable → inner strptime raises, falls through to
        # the date arm (absent) → 0.0.
        with inject_modules(memory=make_fake_memory(
                [{"iso_end": "not-a-timestamp"}])):
            self.assertEqual(self.mod._last_session_end_ts(), 0.0)

    def test_session_end_malformed_date_is_zero(self):
        # date present but unparseable → inner strptime raises → 0.0.
        with inject_modules(memory=make_fake_memory([{"date": "31/13/2026"}])):
            self.assertEqual(self.mod._last_session_end_ts(), 0.0)

    def test_session_end_no_memory_module_is_zero(self):
        # importlib.import_module('memory') raising → 0.0, no crash. Patch the
        # skill's own importlib so a real `memory` module on the dev box can't
        # satisfy the import.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no memory")):
            self.assertEqual(self.mod._last_session_end_ts(), 0.0)

    def test_session_end_outer_exception_is_zero(self):
        # An error escaping the inner handlers (here: get_session_summaries
        # itself raising) is caught by the outer try/except → 0.0.
        mem = types.ModuleType("memory")

        def _boom(*a, **k):
            raise RuntimeError("db locked")
        mem.get_session_summaries = _boom
        with inject_modules(memory=mem):
            self.assertEqual(self.mod._last_session_end_ts(), 0.0)


# ─── _is_warm_restart window gate ────────────────────────────────────────
class SuitUpWarmRestartTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")

    def test_warm_restart_within_window(self):
        with mock.patch.object(self.mod, "_last_session_end_ts",
                               return_value=time.time() - 2 * 3600):
            self.assertTrue(self.mod._is_warm_restart())

    def test_not_warm_restart_when_too_old(self):
        with mock.patch.object(self.mod, "_last_session_end_ts",
                               return_value=time.time() - 20 * 3600):
            self.assertFalse(self.mod._is_warm_restart())

    def test_not_warm_restart_when_no_prior_session(self):
        with mock.patch.object(self.mod, "_last_session_end_ts",
                               return_value=0.0):
            self.assertFalse(self.mod._is_warm_restart())

    def test_not_warm_restart_when_end_in_future(self):
        # A future timestamp (age <= 0) is not a warm restart.
        with mock.patch.object(self.mod, "_last_session_end_ts",
                               return_value=time.time() + 100):
            self.assertFalse(self.mod._is_warm_restart())


# ─── overlay coordination ────────────────────────────────────────────────
class SuitUpOverlayTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")

    def test_ensure_overlay_launches_when_down(self):
        ov = make_fake_overlay(alive=False, launch=(True, "up"))
        with inject_modules(skill_holographic_overlay=ov):
            launched = self.mod._ensure_holo_overlay_up()
        self.assertTrue(launched)
        self.assertEqual(ov._calls["launch"], 1)

    def test_ensure_overlay_skips_when_already_up(self):
        ov = make_fake_overlay(alive=True)
        with inject_modules(skill_holographic_overlay=ov):
            launched = self.mod._ensure_holo_overlay_up()
        self.assertFalse(launched)          # already alive → we didn't launch
        self.assertEqual(ov._calls["launch"], 0)

    def test_ensure_overlay_missing_module_returns_false(self):
        with inject_modules(skill_holographic_overlay=None):
            self.assertFalse(self.mod._ensure_holo_overlay_up())

    def test_ensure_overlay_is_alive_raises_treated_as_down(self):
        # _overlay_is_alive() raising → treated as not-alive, so we try launch.
        ov = make_fake_overlay(alive_raises=True, launch=(True, "up"))
        with inject_modules(skill_holographic_overlay=ov):
            launched = self.mod._ensure_holo_overlay_up()
        self.assertTrue(launched)

    def test_ensure_overlay_launch_failure_returns_false(self):
        ov = make_fake_overlay(alive=False, launch=(False, "denied"))
        with inject_modules(skill_holographic_overlay=ov):
            self.assertFalse(self.mod._ensure_holo_overlay_up())

    def test_ensure_overlay_launch_raises_returns_false(self):
        ov = make_fake_overlay(alive=False, launch_raises=True)
        with inject_modules(skill_holographic_overlay=ov):
            self.assertFalse(self.mod._ensure_holo_overlay_up())

    def test_dismiss_overlay_calls_shutdown(self):
        ov = make_fake_overlay()
        with inject_modules(skill_holographic_overlay=ov):
            self.mod._dismiss_holo_overlay()
        self.assertEqual(ov._calls["shutdown"], 1)

    def test_dismiss_overlay_missing_module_is_noop(self):
        with inject_modules(skill_holographic_overlay=None):
            self.mod._dismiss_holo_overlay()   # must not raise

    def test_dismiss_overlay_shutdown_raises_swallowed(self):
        ov = make_fake_overlay(shutdown_raises=True)
        with inject_modules(skill_holographic_overlay=ov):
            self.mod._dismiss_holo_overlay()   # must not raise


# ─── HUD animation start/clear ───────────────────────────────────────────
class SuitUpAnimationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")

    def test_start_animation_no_writer_returns_timestamp(self):
        started = self.mod._start_animation(None)
        self.assertIsInstance(started, float)
        self.assertGreater(started, 0.0)

    def test_start_animation_writes_powering_phase(self):
        writes = []
        started = self.mod._start_animation(lambda **kw: writes.append(kw))
        self.assertGreater(started, 0.0)
        self.assertEqual(len(writes), 1)
        kw = writes[0]
        # The HUD hook only matches boot_phase=='powering' literally.
        self.assertEqual(kw["boot_phase"], "powering")
        self.assertEqual(kw["boot_duration"], self.mod.SUIT_UP_ANIMATION_SECONDS)
        self.assertEqual(kw["state"], "Initialising")

    def test_start_animation_writer_error_swallowed(self):
        def boom(**kw):
            raise RuntimeError("hud down")
        # Returns a timestamp despite the writer raising.
        started = self.mod._start_animation(boom)
        self.assertGreater(started, 0.0)

    def test_clear_animation_no_writer_is_noop(self):
        self.mod._clear_animation(None, 0.0)   # must not raise

    def test_clear_animation_writes_idle_phase_after_hold(self):
        writes = []
        slept = []
        # started_at far in the past → elapsed already exceeds the budget, so no
        # sleep; assert the clear write happened with the reset fields.
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=lambda s: slept.append(s)):
            self.mod._clear_animation(lambda **kw: writes.append(kw),
                                      started_at=time.time() - 999)
        self.assertEqual(slept, [])           # no hold needed
        self.assertEqual(writes[-1]["boot_phase"], "")
        self.assertEqual(writes[-1]["state"], "Idle")

    def test_clear_animation_holds_until_budget(self):
        writes = []
        slept = []
        # started_at = now → must sleep ~the full budget before clearing.
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=lambda s: slept.append(s)):
            self.mod._clear_animation(lambda **kw: writes.append(kw),
                                      started_at=time.time())
        self.assertEqual(len(slept), 1)
        self.assertGreater(slept[0], 0.0)
        self.assertLessEqual(slept[0], self.mod.SUIT_UP_ANIMATION_SECONDS)
        self.assertEqual(writes[-1]["boot_phase"], "")

    def test_clear_animation_writer_error_swallowed(self):
        def boom(**kw):
            raise RuntimeError("hud down")
        with mock.patch.object(self.mod.time, "sleep"):
            self.mod._clear_animation(boom, started_at=time.time() - 999)


# ─── play_suit_up_sequence ───────────────────────────────────────────────
class SuitUpSequenceTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")

    def test_sequence_speaks_all_lines_in_order(self):
        spoken = []
        speak_fn = lambda line: spoken.append(line)
        with mock.patch.object(self.mod, "_ensure_holo_overlay_up",
                               return_value=False), \
             mock.patch.object(self.mod, "_dismiss_holo_overlay"), \
             mock.patch.object(self.mod, "_start_animation", return_value=0.0), \
             mock.patch.object(self.mod, "_clear_animation"), \
             mock.patch.object(self.mod, "_resolve_speaker_name",
                               return_value="Gaming Headset"), \
             mock.patch.object(self.mod.time, "sleep"):
            welcome = self.mod.play_suit_up_sequence(speak_fn=speak_fn)
        # 4 diagnostics + 1 welcome.
        self.assertEqual(len(spoken), 5)
        self.assertEqual(spoken[0], "Diagnostics: nominal.")
        self.assertIn("Gaming Headset", spoken[2])
        self.assertEqual(spoken[-1], "Welcome back, sir. Systems are yours.")
        self.assertEqual(welcome, "Welcome back, sir. Systems are yours.")

    def test_sequence_dismisses_overlay_only_if_it_launched_it(self):
        with mock.patch.object(self.mod, "_ensure_holo_overlay_up",
                               return_value=True), \
             mock.patch.object(self.mod, "_dismiss_holo_overlay") as dismiss, \
             mock.patch.object(self.mod, "_start_animation", return_value=0.0), \
             mock.patch.object(self.mod, "_clear_animation"), \
             mock.patch.object(self.mod, "_resolve_speaker_name",
                               return_value="Spk"), \
             mock.patch.object(self.mod.time, "sleep"):
            self.mod.play_suit_up_sequence(speak_fn=lambda _l: None)
        dismiss.assert_called_once()

    def test_sequence_does_not_dismiss_when_user_had_overlay(self):
        with mock.patch.object(self.mod, "_ensure_holo_overlay_up",
                               return_value=False), \
             mock.patch.object(self.mod, "_dismiss_holo_overlay") as dismiss, \
             mock.patch.object(self.mod, "_start_animation", return_value=0.0), \
             mock.patch.object(self.mod, "_clear_animation"), \
             mock.patch.object(self.mod, "_resolve_speaker_name",
                               return_value="Spk"), \
             mock.patch.object(self.mod.time, "sleep"):
            self.mod.play_suit_up_sequence(speak_fn=lambda _l: None)
        dismiss.assert_not_called()

    def test_sequence_survives_speak_failure(self):
        def boom(_line):
            raise RuntimeError("tts down")
        with mock.patch.object(self.mod, "_ensure_holo_overlay_up",
                               return_value=False), \
             mock.patch.object(self.mod, "_start_animation", return_value=0.0), \
             mock.patch.object(self.mod, "_clear_animation"), \
             mock.patch.object(self.mod, "_resolve_speaker_name",
                               return_value="Spk"), \
             mock.patch.object(self.mod.time, "sleep"):
            welcome = self.mod.play_suit_up_sequence(speak_fn=boom)
        self.assertEqual(welcome, "Welcome back, sir. Systems are yours.")

    def test_sequence_passes_writer_through_to_animation(self):
        # The supplied write_hud_state reaches _start_animation/_clear_animation.
        writer = mock.MagicMock()
        with mock.patch.object(self.mod, "_ensure_holo_overlay_up",
                               return_value=False), \
             mock.patch.object(self.mod, "_dismiss_holo_overlay"), \
             mock.patch.object(self.mod, "_start_animation",
                               return_value=0.0) as start, \
             mock.patch.object(self.mod, "_clear_animation") as clear, \
             mock.patch.object(self.mod, "_resolve_speaker_name",
                               return_value="Spk"), \
             mock.patch.object(self.mod.time, "sleep"):
            self.mod.play_suit_up_sequence(speak_fn=lambda _l: None,
                                           write_hud_state=writer)
        self.assertIs(start.call_args[0][0], writer)
        self.assertIs(clear.call_args[0][0], writer)

    def test_sequence_inter_line_sleep_error_swallowed(self):
        # time.sleep raising between readouts must not abort the sequence.
        spoken = []
        with mock.patch.object(self.mod, "_ensure_holo_overlay_up",
                               return_value=False), \
             mock.patch.object(self.mod, "_dismiss_holo_overlay"), \
             mock.patch.object(self.mod, "_start_animation", return_value=0.0), \
             mock.patch.object(self.mod, "_clear_animation"), \
             mock.patch.object(self.mod, "_resolve_speaker_name",
                               return_value="Spk"), \
             mock.patch.object(self.mod.time, "sleep",
                               side_effect=RuntimeError("interrupted")):
            welcome = self.mod.play_suit_up_sequence(
                speak_fn=lambda l: spoken.append(l))
        self.assertEqual(welcome, "Welcome back, sir. Systems are yours.")
        self.assertEqual(len(spoken), 5)


# ─── maybe_play_morning_suit_up gate ─────────────────────────────────────
class SuitUpMorningGateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")
        fd, self.statep = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.statep)   # start with no state file
        self._patch = mock.patch.object(self.mod, "_STATE_FILE", self.statep)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._unlink)

    def _unlink(self):
        try:
            os.unlink(self.statep)
        except OSError:
            pass

    def test_morning_fires_on_warm_restart_first_time(self):
        with mock.patch.object(self.mod, "_is_warm_restart", return_value=True), \
             mock.patch.object(self.mod, "play_suit_up_sequence",
                               return_value="Welcome back, sir. Systems are yours."
                               ) as seq:
            out = self.mod.maybe_play_morning_suit_up(speak_fn=lambda _l: None)
        self.assertIn("Welcome back", out)
        seq.assert_called_once()
        with open(self.statep, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state["last_fired_date"], time.strftime("%Y-%m-%d"))
        self.assertEqual(state["last_reason"], "first warm restart of day")

    def test_morning_skips_when_not_warm_restart(self):
        with mock.patch.object(self.mod, "_is_warm_restart", return_value=False), \
             mock.patch.object(self.mod, "play_suit_up_sequence") as seq:
            out = self.mod.maybe_play_morning_suit_up(speak_fn=lambda _l: None)
        self.assertEqual(out, "")
        seq.assert_not_called()

    def test_morning_skips_when_already_fired_today(self):
        with open(self.statep, "w", encoding="utf-8") as f:
            json.dump({"last_fired_date": time.strftime("%Y-%m-%d")}, f)
        with mock.patch.object(self.mod, "_is_warm_restart", return_value=True), \
             mock.patch.object(self.mod, "play_suit_up_sequence") as seq:
            out = self.mod.maybe_play_morning_suit_up(speak_fn=lambda _l: None)
        self.assertEqual(out, "")
        seq.assert_not_called()

    def test_morning_passes_writer_and_speaker_through(self):
        writer = mock.MagicMock()
        with mock.patch.object(self.mod, "_is_warm_restart", return_value=True), \
             mock.patch.object(self.mod, "play_suit_up_sequence",
                               return_value="Welcome back, sir. Systems are yours."
                               ) as seq:
            self.mod.maybe_play_morning_suit_up(
                speak_fn=lambda _l: None, write_hud_state=writer,
                speaker_name="Studio Monitors")
        _a, kw = seq.call_args
        self.assertIs(kw["write_hud_state"], writer)
        self.assertEqual(kw["speaker_name"], "Studio Monitors")


# ─── _act_suit_up verbal trigger + register ──────────────────────────────
class SuitUpActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")
        fd, self.statep = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.statep)
        self._patch = mock.patch.object(self.mod, "_STATE_FILE", self.statep)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._unlink)

    def _unlink(self):
        try:
            os.unlink(self.statep)
        except OSError:
            pass

    def test_act_suit_up_fires_and_records_manual(self):
        bc = types.ModuleType("bobert_companion")
        bc._speak = mock.MagicMock()
        bc._write_hud_state = mock.MagicMock()
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "play_suit_up_sequence") as seq:
            out = self.actions["suit_up"]("")
        self.assertIn("Welcome back", out)
        seq.assert_called_once()
        # speak_fn + write_hud_state were bound from bobert_companion.
        _a, kw = seq.call_args
        self.assertIs(kw["speak_fn"], bc._speak)
        self.assertIs(kw["write_hud_state"], bc._write_hud_state)
        # A manual fire is journalled to state.
        with open(self.statep, "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["last_reason"], "manual trigger")

    def test_act_suit_up_no_bobert_module_reports_error(self):
        # Patch the skill's importlib so the real bobert_companion file on the
        # dev box can't satisfy the import — the except-arm must return the
        # speech-subsystem error string.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("monolith offline")):
            out = self.actions["suit_up"]("")
        self.assertIn("can't reach the speech subsystem", out)

    def test_act_suit_up_no_speak_fn_reports_tts_offline(self):
        # bobert_companion present but without a callable _speak.
        bc = types.ModuleType("bobert_companion")
        bc._speak = None
        bc._write_hud_state = mock.MagicMock()
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "play_suit_up_sequence") as seq:
            out = self.actions["suit_up"]("")
        self.assertIn("TTS layer is offline", out)
        seq.assert_not_called()

    def test_register_wires_both_aliases(self):
        actions: dict = {}
        self.mod.register(actions)
        self.assertIn("suit_up", actions)
        self.assertIn("suit_up_sequence", actions)
        self.assertIs(actions["suit_up"], actions["suit_up_sequence"])


if __name__ == "__main__":
    unittest.main()
