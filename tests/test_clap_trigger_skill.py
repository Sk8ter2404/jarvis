"""Tests for skills/clap_trigger.py — the live wiring around core/clap_detector.

Loads the skill in isolation (tests/_skill_harness) with a FAKE monolith in
sys.modules: no microphone, no real record_speech, no real actions. Frames are
pushed by hand into the queue the skill registers through the fake
add_record_tap, so the REAL tap -> detector -> gate -> routine path runs.

What is pinned:
  * the gates: flag off, staging, the 60 s cool-down, JARVIS speaking (the
    playback flag AND the core/self_echo playback registry), asleep (unless
    CLAP_TRIGGER_WAKE), Mute Mic, game mode, sustained music;
  * the routine: the morning workspace setup by default, an acknowledgement
    when it is missing or "acknowledge" is configured, NEVER a destructive /
    power / send / delete action whatever the setting says, self-voiced
    actions are not re-spoken, a crashing action is reported, not raised;
  * the worker: shares the main loop's mic through add_record_tap (never a
    stream of its own), resets the detector across a gap in the frames,
    removes its tap when it stops, and stops itself when the flag goes off;
  * the voice toggles + their utterance route + the prompt routing;
  * the settings surface: OFF-safe defaults in core/config.py, the Settings
    schema rows, the shipped template.

stdlib unittest + mock + numpy.
"""
from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated

_SAVED_SETTINGS_ENV = None


def setUpModule():
    # Belt and braces (the toggle tests also patch the writer): any save that
    # escaped would land in a throwaway file, never data/user_settings.json.
    global _SAVED_SETTINGS_ENV
    _SAVED_SETTINGS_ENV = os.environ.get("JARVIS_SETTINGS_PATH")
    d = tempfile.mkdtemp(prefix="jarvis_clap_test_")
    os.environ["JARVIS_SETTINGS_PATH"] = os.path.join(d, "user_settings.json")


def tearDownModule():
    if _SAVED_SETTINGS_ENV is None:
        os.environ.pop("JARVIS_SETTINGS_PATH", None)
    else:
        os.environ["JARVIS_SETTINGS_PATH"] = _SAVED_SETTINGS_ENV


SR = 16000


def _scene(times, seconds=4.0, seed=1, amp=0.5):
    from tests.test_clap_detector import _Scene
    rng = np.random.default_rng(seed)
    return _Scene(seconds, rng).claps(times, amp=amp).samples()


def _fake_bc(*, sleep=False, standby=False, speaking=False, muted=False,
             actions=None, self_voiced=()):
    bc = types.ModuleType("bobert_companion")
    bc._sleep_mode = [bool(sleep)]
    bc._standby_mode = [bool(standby)]
    bc._standby_auto_engage_lock = threading.Lock()
    bc._tts_playback_active = [bool(speaking)]
    bc._mic_muted = [bool(muted)]
    bc._record_speech_sr = [SR]
    bc.SAMPLE_RATE = SR
    bc._is_staging = lambda: False
    bc._DESTRUCTIVE_REPLAY_ACTIONS = frozenset({"run_shell", "reset_memory",
                                                "restart"})
    bc.ACTIONS = dict(actions or {})
    bc._announced = []
    bc.proactive_announce = lambda msg, source="skill", **k: (
        bc._announced.append((source, msg)) or True)
    bc._hud = []
    bc._write_hud_state = lambda **kw: bc._hud.append(kw)
    bc.is_self_voiced = lambda n: n in set(self_voiced)
    bc._taps = []
    bc._removed = []

    def _add(q):
        bc._taps.append(q)
        return True

    def _remove(q):
        bc._removed.append(q)
        if q in bc._taps:
            bc._taps.remove(q)
    bc.add_record_tap = _add
    bc.remove_record_tap = _remove
    return bc


def _event(t_first=1.0, gap=0.3):
    return {"t_first": t_first, "t_second": t_first + gap,
            "t_detect": t_first + gap + 0.71, "interval_s": gap,
            "peak_first": 0.5, "peak_second": 0.5}


class _Base(unittest.TestCase):
    FLAGS = {"CLAP_TRIGGER_ENABLED": True,
             "CLAP_TRIGGER_ACTION": "predictive_morning_setup",
             "CLAP_TRIGGER_WAKE": False,
             "CLAP_TRIGGER_COOLDOWN_S": 60.0,
             "CLAP_TRIGGER_MIN_PEAK": 0.12}

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("clap_trigger",
                                                     register=False)
        from core import config as cfg
        for k, v in self.FLAGS.items():
            p = mock.patch.object(cfg, k, v, create=True)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(self.mod, "_is_staging", lambda: False)
        p.start()
        self.addCleanup(p.stop)
        # Run the routine on the calling thread so a test can assert at once.
        p = mock.patch.object(self.mod, "_spawn", lambda fn, *a: fn(*a))
        p.start()
        self.addCleanup(p.stop)
        # Mid-afternoon (the night gate is pinned on its own), and silent
        # speakers: no test reads the real clock or the real playback meter.
        import datetime as _dt
        self.wall = [_dt.datetime(2026, 10, 2, 14, 0)]
        p = mock.patch.object(self.mod, "_now_local", lambda: self.wall[0])
        p.start()
        self.addCleanup(p.stop)
        self.speaker_peak = [0.0]
        p = mock.patch.object(self.mod, "_speaker_peak",
                              lambda: self.speaker_peak[0])
        p.start()
        self.addCleanup(p.stop)
        self.mod._reset_state_for_tests()
        from core import self_echo
        self_echo._reset_for_tests()
        self.addCleanup(self_echo._reset_for_tests)
        self.setup_calls = []

        def _setup(_=""):
            self.setup_calls.append(_)
            return "Workshop is yours, sir. Apple Music is queued."
        self.setup_fn = _setup

    def flag(self, name, value):
        from core import config as cfg
        p = mock.patch.object(cfg, name, value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def inject(self, name, module):
        old = sys.modules.get(name)
        sys.modules[name] = module
        self.addCleanup(lambda: sys.modules.__setitem__(name, old)
                        if old is not None else sys.modules.pop(name, None))

    def bc(self, **kw):
        kw.setdefault("actions", {"predictive_morning_setup": self.setup_fn})
        return _fake_bc(**kw)


# ─── gates ──────────────────────────────────────────────────────────────────
class GateTests(_Base):
    def test_a_double_clap_runs_the_workspace_setup(self):
        bc = self.bc()
        out = self.mod.handle_double_clap(bc, _event(), now=1000.0)
        self.assertEqual(out, "fired")
        self.assertEqual(len(self.setup_calls), 1)
        self.assertEqual(bc._announced,
                         [("clap", "Workshop is yours, sir. Apple Music is queued.")])

    def test_flag_off_ignores(self):
        self.flag("CLAP_TRIGGER_ENABLED", False)
        bc = self.bc()
        self.assertIn("off", self.mod.handle_double_clap(bc, _event(), now=1.0))
        self.assertEqual(self.setup_calls, [])

    def test_staging_ignores(self):
        bc = self.bc()
        with mock.patch.object(self.mod, "_is_staging", lambda: True):
            out = self.mod.handle_double_clap(bc, _event(), now=1.0)
        self.assertIn("staging", out)
        self.assertEqual(self.setup_calls, [])

    def test_cooldown(self):
        bc = self.bc()
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=500.0),
                         "fired")
        self.assertIn("cool", self.mod.handle_double_clap(bc, _event(),
                                                          now=530.0))
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=561.0),
                         "fired")
        self.assertEqual(len(self.setup_calls), 2)

    def test_never_while_jarvis_is_speaking(self):
        bc = self.bc(speaking=True)
        self.assertIn("speaking", self.mod.handle_double_clap(bc, _event(),
                                                              now=10.0))
        self.assertEqual(self.setup_calls, [])

    def test_never_on_the_echo_of_a_line_that_just_ended(self):
        # The detector reported claps at now-1.0 / now-0.7 (mapped from the
        # stream clock); JARVIS's line ended 0.4 s before the first one.
        from core import self_echo
        now = 2000.0
        tok = self_echo.playback_begin("Good evening, sir.", at=now - 4.0)
        self_echo.playback_end(tok, at=now - 1.4)
        bc = self.bc()
        ev = _event(t_first=10.0)               # t_detect = 11.01
        out = self.mod.handle_double_clap(bc, ev, now=now)
        self.assertIn("speaking", out)
        self.assertEqual(self.setup_calls, [])

    def test_a_line_that_ended_long_before_does_not_block(self):
        from core import self_echo
        now = 3000.0
        tok = self_echo.playback_begin("Hello, sir.", at=now - 20.0)
        self_echo.playback_end(tok, at=now - 15.0)
        bc = self.bc()
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=now),
                         "fired")

    def test_asleep_ignores_unless_clap_to_wake(self):
        bc = self.bc(sleep=True, standby=True)
        self.assertIn("asleep", self.mod.handle_double_clap(bc, _event(),
                                                            now=10.0))
        self.assertEqual(self.setup_calls, [])
        self.assertTrue(bc._sleep_mode[0], "a clap must not wake him unasked")

    def test_clap_to_wake(self):
        self.flag("CLAP_TRIGGER_WAKE", True)
        bc = self.bc(sleep=True, standby=True)
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=10.0),
                         "fired")
        self.assertFalse(bc._sleep_mode[0])
        self.assertFalse(bc._standby_mode[0])
        self.assertEqual(len(self.setup_calls), 1)

    def test_muted_mic_ignores(self):
        bc = self.bc(muted=True)
        self.assertIn("muted", self.mod.handle_double_clap(bc, _event(),
                                                           now=10.0))
        self.assertEqual(self.setup_calls, [])

    def test_game_mode_ignores(self):
        gm = types.ModuleType("skill_game_mode")
        gm._st = types.SimpleNamespace(active=True)
        self.inject("skill_game_mode", gm)
        bc = self.bc()
        self.assertIn("game", self.mod.handle_double_clap(bc, _event(),
                                                          now=10.0))
        self.assertEqual(self.setup_calls, [])

    def test_sustained_music_ignores(self):
        sa = types.ModuleType("skill_standby_audio_detect")
        sa.is_music_currently_playing = lambda: True
        self.inject("skill_standby_audio_detect", sa)
        bc = self.bc()
        self.assertIn("music", self.mod.handle_double_clap(bc, _event(),
                                                           now=10.0))
        self.assertEqual(self.setup_calls, [])

    def test_an_ignored_clap_does_not_start_the_cooldown(self):
        bc = self.bc(speaking=True)
        self.mod.handle_double_clap(bc, _event(), now=100.0)
        bc._tts_playback_active[0] = False
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=105.0),
                         "fired")

    # ── 2026-10-02 review: the gates a false double clap needs ──────────────
    def test_night_hours_ignore_even_with_clap_to_wake(self):
        """A false double clap at 3 a.m. must not wake him, set up the desk
        or speak — the phone pings' quiet hours, read live."""
        import datetime as _dt
        self.flag("CLAP_TRIGGER_WAKE", True)
        for hhmm, fired in (((3, 0), False), ((23, 30), False),
                            ((6, 59), False), ((7, 0), True), ((22, 59), True)):
            with self.subTest(at=hhmm):
                self.mod._reset_state_for_tests()
                self.wall[0] = _dt.datetime(2026, 10, 2, *hhmm)
                bc = self.bc(sleep=True, standby=True)
                out = self.mod.handle_double_clap(bc, _event(), now=10.0)
                self.assertEqual(out == "fired", fired, out)
                self.assertEqual(bc._sleep_mode[0], not fired)
                if not fired:
                    self.assertIn("night", out)
                    self.assertEqual(bc._announced, [])

    def test_night_window_is_the_quiet_hours_setting(self):
        import datetime as _dt
        self.wall[0] = _dt.datetime(2026, 10, 2, 21, 30)
        bc = self.bc()
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=10.0),
                         "fired")
        self.mod._reset_state_for_tests()
        self.flag("PHONE_PING_QUIET_START", "21:00")
        self.assertIn("night", self.mod.handle_double_clap(bc, _event(),
                                                           now=10.0))
        self.mod._reset_state_for_tests()
        self.flag("PHONE_PING_QUIET_END", "21:00")   # start == end: none
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=10.0),
                         "fired")

    def test_focus_mode_ignores_either_kind(self):
        bc = self.bc()
        bc.focus_mode_active = lambda: True
        self.assertIn("focus", self.mod.handle_double_clap(bc, _event(),
                                                           now=10.0))
        self.flag("FOCUS_MODE_ENABLED", False)     # the feature's kill switch
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=10.0),
                         "fired")
        # The automatic one (a CAD / slicer window): skills/dnd_focus_mode.
        self.mod._reset_state_for_tests()
        dnd = types.ModuleType("skill_dnd_focus_mode")
        dnd.is_focus_mode_active = lambda: True
        self.inject("skill_dnd_focus_mode", dnd)
        self.assertIn("focus", self.mod.handle_double_clap(self.bc(), _event(),
                                                           now=10.0))
        self.assertEqual(len(self.setup_calls), 1)

    def test_media_playing_ignores(self):
        """A video with claps in it, Apple Music, a TV: the monolith's
        _ambient_media_is_playing (media session / room music / camera)."""
        bc = self.bc()
        bc._ambient_media_is_playing = lambda: True
        self.assertIn("media", self.mod.handle_double_clap(bc, _event(),
                                                           now=10.0))
        # No full probe on an older monolith: the media session alone.
        bc = self.bc()
        bc._smtc_media_playing = lambda: True
        self.assertIn("media", self.mod.handle_double_clap(bc, _event(),
                                                           now=10.0))
        self.assertEqual(self.setup_calls, [])

    def test_music_jarvis_started_recently_ignores(self):
        bc = self.bc()
        bc._jarvis_played_music_at = [time.time() - 120.0]
        self.assertIn("media", self.mod.handle_double_clap(bc, _event(),
                                                           now=10.0))
        bc._jarvis_played_music_at = [time.time() - 3600.0]
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=10.0),
                         "fired")

    def test_sound_on_the_speakers_ignores(self):
        bc = self.bc()
        self.speaker_peak[0] = 0.3          # a game, a call, a clip
        self.assertIn("speakers", self.mod.handle_double_clap(bc, _event(),
                                                              now=10.0))
        self.speaker_peak[0] = 0.001        # a silent device
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=10.0),
                         "fired")

    def test_an_unreadable_meter_does_not_block(self):
        bc = self.bc()
        self.speaker_peak[0] = None
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=10.0),
                         "fired")

    def test_clap_to_wake_uses_the_monoliths_full_wake(self):
        """2026-10-02 review: the tray's own wake (_force_wake) — the overnight
        flag, the morning chain's wake stamp — silently, not a partial copy."""
        self.flag("CLAP_TRIGGER_WAKE", True)
        bc = self.bc(sleep=True, standby=True)
        calls = []

        def _fw(speak=True, source="tray"):
            calls.append((speak, source))
            bc._sleep_mode[0] = False
            bc._standby_mode[0] = False
        bc._force_wake = _fw
        self.assertEqual(self.mod.handle_double_clap(bc, _event(), now=10.0),
                         "fired")
        self.assertEqual(calls, [(False, "clap")])
        self.assertEqual(len(self.setup_calls), 1)


# ─── the routine ────────────────────────────────────────────────────────────
class RoutineTests(_Base):
    def test_missing_workspace_setup_falls_back_to_an_acknowledgement(self):
        bc = self.bc(actions={})
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertEqual(bc._announced, [("clap", self.mod.ACK_LINE)])

    def test_acknowledge_setting_never_runs_the_setup(self):
        self.flag("CLAP_TRIGGER_ACTION", "acknowledge")
        bc = self.bc()
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertEqual(self.setup_calls, [])
        self.assertEqual(bc._announced, [("clap", self.mod.ACK_LINE)])

    def test_blank_setting_means_the_default(self):
        """The default is the acknowledgement (2026-10-02 review): a blank
        setting never runs the workspace setup."""
        self.flag("CLAP_TRIGGER_ACTION", "  ")
        bc = self.bc()
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertEqual(self.setup_calls, [])
        self.assertEqual(bc._announced, [("clap", self.mod.ACK_LINE)])

    def test_the_morning_briefing_may_run(self):
        calls = []
        self.flag("CLAP_TRIGGER_ACTION", "morning_briefing")
        bc = self.bc(actions={"morning_briefing":
                              lambda _="": calls.append(1) or "Sunny, sir."})
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertEqual(calls, [1])
        self.assertEqual(bc._announced, [("clap", "Sunny, sir.")])

    def test_an_alias_of_the_setup_runs_by_handler(self):
        """A name bound to the SAME handler as an allow-listed action runs; a
        look-alike name bound to something else does not."""
        self.flag("CLAP_TRIGGER_ACTION", "desk_please")
        bc = self.bc(actions={"predictive_morning_setup": self.setup_fn,
                              "desk_please": self.setup_fn})
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertEqual(len(self.setup_calls), 1)
        ran = []
        self.mod._reset_state_for_tests()
        bc = self.bc(actions={"predictive_morning_setup": self.setup_fn,
                              "desk_please": lambda _="": ran.append(1)})
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertEqual(ran, [])
        self.assertIn("won't run", bc._announced[0][1])

    def test_only_the_allow_list_runs_on_a_clap(self):
        """2026-10-02 review: the old word list let these through. An
        allow-list refuses every one of them, however harmless it sounds."""
        for name in ("guard_off", "resume_print", "export_memory",
                     "smart_home_purge_cookie", "scrap_pending_draft",
                     "archive_email", "archive_message", "text_my_phone",
                     "notify_phone", "say_aloud", "stop_pipeline",
                     "web_interface_off", "force_backup", "type", "hotkey",
                     "click", "set_model", "switch_llm", "hibernate_pc",
                     "sleep_pc", "lock_pc", "log_off", "sign_out",
                     "weather_briefing", "unlock_front_door"):
            with self.subTest(action=name):
                ran = []
                self.flag("CLAP_TRIGGER_ACTION", name)
                bc = self.bc(actions={name: lambda _="": ran.append(1) or "x"})
                self.mod._reset_state_for_tests()
                self.mod.handle_double_clap(bc, _event(), now=10.0)
                self.assertEqual(ran, [], f"{name} ran on a clap")
                self.assertEqual(len(bc._announced), 1)
                self.assertIn("won't run", bc._announced[0][1])

    def test_action_risk_is_a_floor_under_the_allow_list(self):
        """Even an allow-listed name is refused if core/action_risk ever
        classes it as risky."""
        with mock.patch("core.action_risk.confirm_reasons",
                        return_value=("spends money",)):
            refusal = self.mod._routine_refusal("predictive_morning_setup",
                                                self.bc())
        self.assertTrue(refusal)
        self.assertIsNone(self.mod._routine_refusal(
            "predictive_morning_setup", self.bc()))

    def test_dangerous_actions_never_run_on_a_clap(self):
        for name in ("shutdown_jarvis", "turn_off_jarvis", "restart",
                     "run_shell", "reset_memory", "delete_file",
                     "send_pending_draft", "buy_now", "kill_process",
                     "forget_face", "close_window", "run_python"):
            with self.subTest(action=name):
                ran = []
                self.flag("CLAP_TRIGGER_ACTION", name)
                bc = self.bc(actions={name: lambda _="": ran.append(1) or "x"})
                self.mod._reset_state_for_tests()
                self.mod.handle_double_clap(bc, _event(), now=10.0)
                self.assertEqual(ran, [], f"{name} ran on a clap")
                self.assertEqual(len(bc._announced), 1)
                self.assertIn("won't run", bc._announced[0][1])

    def test_a_self_voiced_routine_is_not_spoken_twice(self):
        bc = self.bc(self_voiced=("predictive_morning_setup",))
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertEqual(len(self.setup_calls), 1)
        self.assertEqual(bc._announced, [])

    def test_a_crashing_routine_is_reported_not_raised(self):
        def boom(_=""):
            raise RuntimeError("nope")
        bc = self.bc(actions={"predictive_morning_setup": boom})
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertEqual(len(bc._announced), 1)
        self.assertIn("snag", bc._announced[0][1])

    def test_an_empty_result_still_answers(self):
        bc = self.bc(actions={"predictive_morning_setup": lambda _="": ""})
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertEqual(bc._announced, [("clap", self.mod.ACK_LINE)])


# ─── the speaker meter probe (fake pycaw / comtypes; never the real device) ──
class SpeakerMeterProbeTests(unittest.TestCase):
    def setUp(self):
        self.mod, _a = load_skill_isolated("clap_trigger", register=False)
        p = mock.patch.object(self.mod.time, "sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)

    def _fake_audio(self, peaks=None, activate_raises=False):
        comtypes = types.ModuleType("comtypes")
        comtypes.CLSCTX_ALL = 23
        comtypes.inits = []
        comtypes.CoInitialize = lambda: comtypes.inits.append(1)
        pycaw = types.ModuleType("pycaw")
        pp = types.ModuleType("pycaw.pycaw")
        reads = list(peaks or [])

        class _Meter:
            def GetPeakValue(self):
                return reads.pop(0) if reads else 0.0

        class _Iface:
            def QueryInterface(self, _iid):
                return _Meter()

        class _Raw:
            def Activate(self, *_a):
                if activate_raises:
                    raise OSError("no device")
                return _Iface()

        class IAudioMeterInformation:
            _iid_ = "meter"

        class AudioUtilities:
            @staticmethod
            def GetSpeakers():
                return types.SimpleNamespace(_dev=_Raw())
        pp.AudioUtilities = AudioUtilities
        pp.IAudioMeterInformation = IAudioMeterInformation
        pycaw.pycaw = pp
        return {"comtypes": comtypes, "pycaw": pycaw, "pycaw.pycaw": pp}

    def test_off_windows_is_unreadable(self):
        with mock.patch.object(self.mod.sys, "platform", "linux"):
            self.assertIsNone(self.mod._speaker_peak())
            self.assertFalse(self.mod._speaker_output_active())

    def test_reads_the_highest_of_a_few_samples(self):
        with mock.patch.object(self.mod.sys, "platform", "win32"):
            with mock.patch.dict(sys.modules, self._fake_audio(
                    peaks=[0.0, 0.005, 0.4, 0.0])):
                self.assertAlmostEqual(self.mod._speaker_peak(), 0.4)
            with mock.patch.dict(sys.modules, self._fake_audio(
                    peaks=[0.0, 0.005, 0.4, 0.0])):
                self.assertTrue(self.mod._speaker_output_active())

    def test_silence_reads_quiet(self):
        mods = self._fake_audio(peaks=[0.0, 0.001, 0.0, 0.002])
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.dict(sys.modules, mods):
            self.assertLess(self.mod._speaker_peak(), self.mod._OUTPUT_PEAK_MIN)
            self.assertFalse(self.mod._speaker_output_active())

    def test_a_missing_device_is_unreadable_not_a_raise(self):
        mods = self._fake_audio(activate_raises=True)
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.dict(sys.modules, mods):
            self.assertIsNone(self.mod._speaker_peak())
            self.assertFalse(self.mod._speaker_output_active())

    def test_no_pycaw_is_unreadable(self):
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.dict(sys.modules, {"pycaw": None, "pycaw.pycaw": None}):
            self.assertIsNone(self.mod._speaker_peak())


# ─── the worker: tap -> detector -> handler ─────────────────────────────────
class WorkerTests(_Base):
    # A generous gap window so a slow CI scheduler pausing the pushing thread
    # never reads as a hole in the stream; the gap test sets its own.
    GAP_S = 1.0

    def _start(self, bc):
        p = mock.patch.object(self.mod, "_GAP_S", self.GAP_S)
        p.start()
        self.addCleanup(p.stop)
        self.inject("bobert_companion", bc)
        self.assertTrue(self.mod._start_worker())
        self.addCleanup(self.mod._stop_worker)
        deadline = time.monotonic() + 3.0
        while not bc._taps and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(bc._taps), 1, "the worker never tapped the mic")
        return bc._taps[0]

    def _push(self, q, samples, chunk=1024):
        for i in range(0, len(samples), chunk):
            while q.qsize() > 8:
                time.sleep(0.001)
            q.put(samples[i:i + chunk])

    def _wait(self, cond, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not cond() and time.monotonic() < deadline:
            time.sleep(0.01)
        return cond()

    def test_synthetic_double_clap_through_the_tap_runs_the_routine(self):
        bc = self.bc()
        q = self._start(bc)
        self.assertIsInstance(q, queue.Queue)
        self.assertGreater(q.maxsize, 0, "an unbounded tap queue can grow "
                                         "without limit behind a stalled worker")
        self._push(q, _scene([1.5, 1.8]))
        self.assertTrue(self._wait(lambda: self.setup_calls),
                        "the double clap never reached the routine")
        self.assertEqual(len(self.setup_calls), 1)

    def test_a_key_double_tap_never_sets_up_the_desk_by_default(self):
        """2026-10-02 review: a mechanical key's double tap (four 0.8 ms
        sub-impulses over 7 ms plus a short rattle, 0.25 s apart, in a quiet
        room) can pass the detector's click test — the click/clap line sits
        close to it for synthetic sounds. Whatever the detector decides, the
        SHIPPED default routine is only "You rang, sir?": the workspace setup
        never runs."""
        self.flag("CLAP_TRIGGER_ACTION",
                  self._shipped_default("CLAP_TRIGGER_ACTION"))
        rng = np.random.default_rng(1)

        def mech_key(amp):
            k = np.zeros(int(0.03 * SR))
            for off, rel in ((0, 1.0), (0.002, 0.7), (0.004, 0.5),
                             (0.007, 0.35)):
                i, m = int(off * SR), int(0.0008 * SR)
                k[i:i + m] += rng.normal(0, amp * rel, m)
            t = int(0.02 * SR)
            j = int(0.008 * SR)
            k[j:j + t] += (rng.normal(0, amp * 0.2, t)
                           * np.exp(-np.arange(t) / (t / 4)))
            return k

        room = rng.normal(0, 0.002, int(3.0 * SR))
        for at in (1.2, 1.45):
            ev = mech_key(0.1)
            i = int(at * SR)
            room[i:i + ev.size] += ev
        bc = self.bc()
        q = self._start(bc)
        self._push(q, room.astype(np.float32))
        self._wait(lambda: bc._announced, timeout=2.0)
        self.assertEqual(self.setup_calls, [])
        for _src, line in bc._announced:
            self.assertEqual(line, self.mod.ACK_LINE)

    @staticmethod
    def _shipped_default(key):
        import ast
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "core", "config.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None) == key):
                return ast.literal_eval(node.value)
        raise AssertionError(f"{key} missing from core/config.py")

    def test_a_single_clap_through_the_tap_does_nothing(self):
        bc = self.bc()
        q = self._start(bc)
        self._push(q, _scene([1.5]))
        self.assertTrue(self._wait(
            lambda: self.mod.status_snapshot()["claps_heard"] == 1))
        self.assertTrue(self._wait(lambda: q.qsize() == 0))
        time.sleep(0.1)
        self.assertEqual(self.setup_calls, [])

    def test_a_gap_in_the_frames_splits_a_pair(self):
        # record_speech closed between the claps (a turn ran): the stream has a
        # hole, so the two claps are seconds apart in real time.
        self.GAP_S = 0.1
        bc = self.bc()
        q = self._start(bc)
        x = _scene([1.5, 1.8])
        cut = int(1.65 * SR) // 1024 * 1024
        self._push(q, x[:cut])
        self.assertTrue(self._wait(lambda: q.qsize() == 0))
        time.sleep(self.GAP_S * 4)
        self._push(q, x[cut:])
        self.assertTrue(self._wait(lambda: q.qsize() == 0))
        time.sleep(0.3)
        self.assertEqual(self.setup_calls, [])
        # ...and it was the reset, not deafness: the clap after the gap was
        # heard; the one before it was dropped half-analysed with the stream.
        self.assertEqual(self.mod.status_snapshot()["claps_heard"], 1)

    def test_the_worker_stops_and_untaps_when_the_flag_goes_off(self):
        bc = self.bc()
        q = self._start(bc)
        self.flag("CLAP_TRIGGER_ENABLED", False)
        self.assertTrue(self._wait(lambda: q in bc._removed, timeout=5.0),
                        "the tap was left registered after the flag went off")
        self.assertTrue(self._wait(lambda: not self.mod._worker_alive()))

    def test_stop_worker_untaps(self):
        bc = self.bc()
        q = self._start(bc)
        self.mod._stop_worker()
        self.assertIn(q, bc._removed)
        self.assertFalse(self.mod._worker_alive())

    def test_no_worker_in_staging(self):
        bc = self.bc()
        self.inject("bobert_companion", bc)
        with mock.patch.object(self.mod, "_is_staging", lambda: True):
            self.assertFalse(self.mod._start_worker())
        self.assertEqual(bc._taps, [])

    def test_no_tap_api_means_no_worker(self):
        bc = self.bc()
        del bc.add_record_tap
        self.inject("bobert_companion", bc)
        self.assertFalse(self.mod._start_worker())

    def test_never_opens_its_own_stream(self):
        import ast
        with open(self.mod.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for banned in ("InputStream", "sounddevice", "rec"):
            self.assertNotIn(banned, names,
                             f"{banned}: the clap trigger must share the main "
                             f"loop's mic through add_record_tap")


# ─── voice toggles ──────────────────────────────────────────────────────────
class ToggleTests(_Base):
    def _writer(self, initial=None):
        from tools import settings_window as sw
        saved = dict(initial or {})
        p1 = mock.patch.object(sw, "load_settings", lambda *a, **k: dict(saved))
        p2 = mock.patch.object(sw, "save_settings",
                               lambda d, *a, **k: saved.update(d))
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        return saved

    def test_on_persists_flips_live_and_starts_listening(self):
        self.flag("CLAP_TRIGGER_ENABLED", False)
        saved = self._writer()
        with mock.patch.object(self.mod, "_start_worker",
                               return_value=True) as start:
            out = self.mod.clap_trigger_on("")
        from core import config as cfg
        self.assertTrue(cfg.CLAP_TRIGGER_ENABLED)
        self.assertIs(saved.get("CLAP_TRIGGER_ENABLED"), True)
        start.assert_called_once()
        self.assertIn("clap trigger on", out.lower())
        self.assertIn("workspace", out.lower())

    def test_on_with_the_default_routine_explains_the_trial(self):
        """Out of the box a clap only answers, and turning it on says so —
        and how to get the morning setup once it only hears him."""
        self.flag("CLAP_TRIGGER_ENABLED", False)
        self.flag("CLAP_TRIGGER_ACTION", "acknowledge")
        self._writer()
        with mock.patch.object(self.mod, "_start_worker", return_value=True):
            out = self.mod.clap_trigger_on("")
        self.assertIn("You rang, sir?", out)
        self.assertIn("clap trigger runs the morning setup", out)

    def test_routine_sets_and_saves_only_allow_listed_routines(self):
        saved = self._writer()
        from core import config as cfg
        for arg, want in (("morning setup", "predictive_morning_setup"),
                          ("set up my workspace", "predictive_morning_setup"),
                          ("the morning briefing", "morning_briefing"),
                          ("just answer", "acknowledge"),
                          ("acknowledge", "acknowledge")):
            with self.subTest(arg=arg):
                out = self.mod.clap_trigger_routine(arg)
                self.assertEqual(cfg.CLAP_TRIGGER_ACTION, want)
                self.assertEqual(saved.get("CLAP_TRIGGER_ACTION"), want)
                self.assertIn("Done", out)
        before = dict(saved)
        out = self.mod.clap_trigger_routine("shut down the pc")
        self.assertEqual(saved, before)
        self.assertEqual(cfg.CLAP_TRIGGER_ACTION, "acknowledge")
        self.assertIn("which would you like", out)

    def test_routine_mentions_the_trigger_is_off(self):
        self.flag("CLAP_TRIGGER_ENABLED", False)
        self._writer()
        self.assertIn("turn on the clap trigger",
                      self.mod.clap_trigger_routine("morning setup"))

    def test_status_names_a_routine_a_clap_may_not_run(self):
        self.flag("CLAP_TRIGGER_ACTION", "run_shell")
        out = self.mod.clap_trigger_status("")
        self.assertIn("not one a clap may run", out)

    def test_off_persists_and_stops(self):
        saved = self._writer({"CLAP_TRIGGER_ENABLED": True})
        with mock.patch.object(self.mod, "_stop_worker") as stop:
            out = self.mod.clap_trigger_off("")
        from core import config as cfg
        self.assertFalse(cfg.CLAP_TRIGGER_ENABLED)
        self.assertIs(saved.get("CLAP_TRIGGER_ENABLED"), False)
        stop.assert_called_once()
        self.assertIn("off", out.lower())

    def test_on_in_staging_refuses(self):
        self.flag("CLAP_TRIGGER_ENABLED", False)
        saved = self._writer()
        with mock.patch.object(self.mod, "_is_staging", lambda: True):
            out = self.mod.clap_trigger_on("")
        self.assertNotIn("CLAP_TRIGGER_ENABLED", saved)
        self.assertIn("staging", out.lower())

    def test_status_reports_off_and_on(self):
        self.flag("CLAP_TRIGGER_ENABLED", False)
        self.assertIn("off", self.mod.clap_trigger_status("").lower())
        self.flag("CLAP_TRIGGER_ENABLED", True)
        out = self.mod.clap_trigger_status("").lower()
        self.assertIn("on", out)
        self.assertIn("workspace", out)

    def test_status_helps_tune_the_loudness(self):
        det = types.SimpleNamespace(stats={"claps": 0}, last_clap_peak=0.0,
                                    last_transient_peak=0.08,
                                    last_rejection="too quiet")
        self.mod._detector_ref[0] = det
        out = self.mod.clap_trigger_status("")
        self.assertIn("0.08", out)
        self.assertIn("0.12", out)
        det.last_clap_peak, det.stats["claps"] = 0.41, 1
        self.assertIn("peaked at 0.41", self.mod.clap_trigger_status(""))

    def test_status_says_why_the_last_clap_was_ignored(self):
        bc = self.bc(speaking=True)
        self.mod.handle_double_clap(bc, _event(), now=10.0)
        self.assertIn("speaking", self.mod.clap_trigger_status("").lower())


class RouteTests(_Base):
    CASES = {
        "turn on the clap trigger": "[ACTION: clap_trigger_on]",
        "Jarvis, turn on the clap trigger.": "[ACTION: clap_trigger_on]",
        "enable the double clap trigger": "[ACTION: clap_trigger_on]",
        "clap trigger on": "[ACTION: clap_trigger_on]",
        "switch the clap detection on please": "[ACTION: clap_trigger_on]",
        "clap trigger off": "[ACTION: clap_trigger_off]",
        "turn off the clap trigger": "[ACTION: clap_trigger_off]",
        "disable clap detection": "[ACTION: clap_trigger_off]",
        "is the clap trigger on?": "[ACTION: clap_trigger_status]",
        "clap trigger status": "[ACTION: clap_trigger_status]",
        "clap trigger runs the morning setup":
            "[ACTION: clap_trigger_routine, morning setup]",
        "make the clap trigger set up my workspace":
            "[ACTION: clap_trigger_routine, morning setup]",
        "Jarvis, have the clap trigger give me the morning briefing.":
            "[ACTION: clap_trigger_routine, morning briefing]",
        "clap trigger just answers":
            "[ACTION: clap_trigger_routine, acknowledge]",
        "set the clap routine to the briefing":
            "[ACTION: clap_trigger_routine, morning briefing]",
    }

    def test_documented_phrases_route(self):
        for text, token in self.CASES.items():
            with self.subTest(text=text):
                self.assertEqual(self.mod._clap_route(text), token)

    def test_unrelated_phrases_do_not(self):
        for text in ("play eric clapton", "turn on the lights",
                     "clap along with me", "what is a clap trigger",
                     "turn on the trigger", "", None,
                     "make the clap trigger shut down the pc",
                     "clap trigger runs run_shell"):
            with self.subTest(text=text):
                self.assertIsNone(self.mod._clap_route(text))


class RegisterTests(unittest.TestCase):
    def test_register_wires_actions_route_and_speak_set(self):
        from tests._skill_harness import make_fake_skill_utils
        utils = make_fake_skill_utils()
        from core import config as cfg
        with mock.patch.object(cfg, "CLAP_TRIGGER_ENABLED", False, create=True):
            mod, actions = load_skill_isolated("clap_trigger", utils=utils)
        for name in ("clap_trigger_on", "clap_trigger_off",
                     "clap_trigger_status", "clap_trigger_routine"):
            self.assertIn(name, actions)
            self.assertIn(name, mod.SPEAK_VERBATIM_ACTIONS)
        utils["register_utterance_route"].assert_called_once()

    def test_register_starts_no_worker_while_off(self):
        from core import config as cfg
        with mock.patch.object(cfg, "CLAP_TRIGGER_ENABLED", False, create=True):
            mod, _ = load_skill_isolated("clap_trigger", register=False)
            with mock.patch.object(mod, "_start_worker") as start:
                mod.register({})
        start.assert_not_called()

    def test_register_starts_the_worker_when_on(self):
        from core import config as cfg
        with mock.patch.object(cfg, "CLAP_TRIGGER_ENABLED", True, create=True):
            mod, _ = load_skill_isolated("clap_trigger", register=False)
            with mock.patch.object(mod, "_is_staging", lambda: False), \
                    mock.patch.object(mod, "_start_worker") as start:
                mod.register({})
        start.assert_called_once()


# ─── settings surface ───────────────────────────────────────────────────────
class SettingsSurfaceTests(unittest.TestCase):
    KEYS = {"CLAP_TRIGGER_ENABLED": False,
            "CLAP_TRIGGER_ACTION": "acknowledge",
            "CLAP_TRIGGER_WAKE": False,
            "CLAP_TRIGGER_COOLDOWN_S": 60.0,
            "CLAP_TRIGGER_MIN_PEAK": 0.12}

    def _config_literals(self):
        import ast
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "core", "config.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        out = {}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                try:
                    out[node.targets[0].id] = ast.literal_eval(node.value)
                except Exception:
                    pass
        return out

    def test_shipped_defaults_are_off_safe(self):
        lits = self._config_literals()
        for key, want in self.KEYS.items():
            with self.subTest(key=key):
                self.assertTrue(key in lits, f"{key} missing from core/config.py")
                self.assertEqual(lits[key], want)

    def test_settings_rows_exist_with_matching_defaults(self):
        from tools import settings_window as sw
        for key, want in self.KEYS.items():
            with self.subTest(key=key):
                self.assertIn(key, sw.SCHEMA)
                self.assertEqual(sw.SCHEMA[key]["default"], want)
                self.assertIn(key, sw.persisted_keys())
        laid_out = {k for _s, keys in sw.tab_layout(sw.SCHEMA[
            "CLAP_TRIGGER_ENABLED"]["tab"]) for k in keys}
        self.assertTrue(set(self.KEYS) <= laid_out)

    def test_the_routine_setting_offers_only_the_allow_list(self):
        from tools import settings_window as sw
        spec = sw.SCHEMA["CLAP_TRIGGER_ACTION"]
        self.assertEqual(spec["type"], "enum")
        mod, _a = load_skill_isolated("clap_trigger", register=False)
        allowed = set(mod._CLAP_SAFE_ACTIONS) | {"acknowledge"}
        self.assertTrue(set(spec["choices"]) <= allowed, spec["choices"])

    def test_template_carries_the_keys(self):
        import json
        from tools import settings_window as sw
        with open(sw.EXAMPLE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        for key, want in self.KEYS.items():
            self.assertEqual(data.get(key), want, key)


# ─── prompt routing (the LOCAL slim prompt must carry the tokens) ────────────
class PromptRoutingTests(unittest.TestCase):
    def test_trigger_phrases_ship_their_action(self):
        from core import prompt_router as pr
        from core.prompts import PC_CONTROL_PROMPT
        for phrase, action in (("turn on the clap trigger", "clap_trigger_on"),
                               ("clap trigger off", "clap_trigger_off"),
                               ("is the clap trigger on", "clap_trigger_status"),
                               ("turn on clap to wake", "clap_trigger_status"),
                               ("make the clap trigger run the morning setup",
                                "clap_trigger_routine")):
            with self.subTest(phrase=phrase):
                slim = pr.slim_pc_control(phrase, PC_CONTROL_PROMPT)
                self.assertIn(action, slim)

    def test_a_clapton_request_does_not_load_the_section(self):
        from core import prompt_router as pr
        from core.prompts import PC_CONTROL_PROMPT
        _core, sections = pr.split_pc_control(PC_CONTROL_PROMPT)
        inc, _ = pr.select_sections("play eric clapton", sections)
        self.assertNotIn("CLAP TRIGGER", inc)


if __name__ == "__main__":
    unittest.main()
