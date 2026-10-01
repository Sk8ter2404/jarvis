"""Monolith side of the 2026-09-30 tray audit fixes.

The tray (tray.py) is a separate process; everything it shows about JARVIS
arrives through hud_state.json, and everything it asks for goes through
tray_commands.json. These tests drive the REAL bobert_companion handlers for
each fix the audit asked for:

  * results      — a tray request (rid) gets its answer in tray_results.json,
                   sync or via _tray_async (interim + final), never for voice.
  * de-dup       — a command re-appended by the tray's read/replace race (same
                   cid) runs once.
  * restart      — commands clicked during a restart's teardown survive the
                   new session's launch; stale / lifecycle ones do not.
  * HUD off      — hud_state keeps being written for the tray.
  * boot order   — the tray launches early; tray_ready_pid ends "starting…".
  * real state   — standby / ambient-listening / dashboard port are published;
                   the ambient toggle flips what is really running.
  * upgrades off — "Run Upgrade Now" never sleeps JARVIS.
  * way back     — show_tray relaunches a quit tray (voice, spoken).
  * Mute Mic     — a capture in progress stops the moment mute is set, and a
                   transcript that finishes after the mute is dropped.

No real audio device, window or tray is touched: the mic is a fake
InputStream fed from a thread (the tests/monolith/test_monolith_self_echo.py
pattern), subprocess.Popen is mocked, and every file lives in a temp dir.

    python -m unittest tests.monolith.test_monolith_tray_fixes
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from collections import OrderedDict
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_WAIT = 5.0


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.tmp = tempfile.mkdtemp(prefix="tray_fix_mono_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.results_path = os.path.join(self.tmp, "tray_results.json")
        self._p(bc, "TRAY_RESULTS_FILE", self.results_path, create=True)
        self._p(bc, "_tray_seen_cids", OrderedDict(), create=True)
        self._p(bc, "_tray_result_seq", [0], create=True)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _quiet(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()

    def _results(self):
        try:
            with open(self.results_path, encoding="utf-8") as f:
                return json.load(f).get("results", [])
        except OSError:
            return []

    def _wait_for(self, pred, what):
        deadline = time.time() + _WAIT
        while time.time() < deadline:
            if pred():
                return
            time.sleep(0.02)
        self.fail(f"timed out waiting for {what}")


# ── results round-trip ─────────────────────────────────────────────────────
class TrayResultTests(_Base):
    def test_publish_is_atomic_and_capped(self):
        from core import atomic_io
        with mock.patch.object(atomic_io, "_atomic_write_json",
                               wraps=atomic_io._atomic_write_json) as aw:
            for i in range(25):
                self.bc._publish_tray_result(f"r-{i}", "x", f"answer {i}")
        self.assertTrue(aw.called, "must go through core.atomic_io")
        res = self._results()
        self.assertEqual(len(res), self.bc._TRAY_RESULTS_MAX)
        self.assertEqual(res[-1]["rid"], "r-24")
        self.assertEqual(res[-1]["text"], "answer 24")
        self.assertTrue(res[-1]["final"])

    def test_no_rid_publishes_nothing(self):
        self.bc._publish_tray_result("", "x", "voice-triggered")
        self.assertFalse(os.path.exists(self.results_path))

    def test_sync_answer_is_published_final(self):
        fn = mock.Mock(return_value="backend=ollama  model=m")
        with mock.patch.object(self.bc, "ACTIONS", {"show_llm_stats": fn}), \
             mock.patch.object(self.bc, "_HEAVY_ACTIONS", frozenset()):
            self._quiet(self.bc._dispatch_tray_command, "show_llm_stats",
                        {"cmd": "show_llm_stats", "rid": "r1-1"})
        res = self._results()
        self.assertEqual([(r["rid"], r["final"], r["text"]) for r in res],
                         [("r1-1", True, "backend=ollama  model=m")])

    def test_spawned_work_reports_interim_then_final(self):
        bc = self.bc

        def backup(_arg):
            bc._tray_async("force_backup", lambda: "backup -> 20260930_101500")
            return "backup started"
        with mock.patch.object(bc, "ACTIONS", {"force_backup": backup}), \
             mock.patch.object(bc, "_HEAVY_ACTIONS", frozenset()):
            self._quiet(bc._dispatch_tray_command, "force_backup",
                        {"cmd": "force_backup", "rid": "r1-2"})
            self._wait_for(lambda: any(r["final"] for r in self._results()),
                           "the final backup result")
        # The worker can finish before the dispatcher files the interim line;
        # the tray copes with either order (a final answer retires the rid).
        res = {(r["final"], r["text"]) for r in self._results()}
        self.assertEqual(res, {(False, "backup started"),
                               (True, "backup -> 20260930_101500")})

    def test_heavy_action_answer_is_published(self):
        fn = mock.Mock(return_value="12 probes, all OK")
        with mock.patch.object(self.bc, "ACTIONS", {"run_diagnostic": fn}), \
             mock.patch.object(self.bc, "_HEAVY_ACTIONS",
                               frozenset({"run_diagnostic"})):
            self._quiet(self.bc._dispatch_tray_command, "run_diagnostic",
                        {"cmd": "run_diagnostic", "rid": "r1-3"})
            self._wait_for(self._results, "the diagnostic result")
        self.assertEqual(self._results()[-1]["text"], "12 probes, all OK")

    def test_nested_async_inside_a_heavy_action(self):
        # _act_run_diagnostic_tray shape: the heavy worker itself hands the
        # sweep to _tray_async; the request must follow it.
        bc = self.bc

        def diag(_arg):
            bc._tray_async("run_diagnostic", lambda: "sweep: 3 FAIL")
            return "diagnostic sweep started"
        with mock.patch.object(bc, "ACTIONS", {"run_diagnostic": diag}), \
             mock.patch.object(bc, "_HEAVY_ACTIONS",
                               frozenset({"run_diagnostic"})):
            self._quiet(bc._dispatch_tray_command, "run_diagnostic",
                        {"cmd": "run_diagnostic", "rid": "r1-4"})
            self._wait_for(lambda: len(self._results()) >= 2, "two answers")
        res = sorted(self._results(), key=lambda r: r["seq"])
        self.assertEqual({(r["final"], r["text"]) for r in res},
                         {(False, "diagnostic sweep started"),
                          (True, "sweep: 3 FAIL")})

    def test_unknown_and_failing_commands_answer(self):
        with mock.patch.object(self.bc, "ACTIONS",
                               {"boom": mock.Mock(side_effect=RuntimeError("x"))}), \
             mock.patch.object(self.bc, "_HEAVY_ACTIONS", frozenset()):
            self._quiet(self.bc._dispatch_tray_command, "nope", {"rid": "r1-5"})
            self._quiet(self.bc._dispatch_tray_command, "boom", {"rid": "r1-6"})
        texts = {r["rid"]: r["text"] for r in self._results()}
        self.assertIn("isn't available", texts["r1-5"])
        self.assertIn("failed", texts["r1-6"])

    def test_voice_triggered_async_publishes_nothing(self):
        done = threading.Event()
        self._quiet(self.bc._tray_async, "run_smoke_test",
                    lambda: done.set() or "smoke test PASSED")
        self.assertTrue(done.wait(_WAIT))
        time.sleep(0.1)
        self.assertEqual(self._results(), [])

    def test_run_upgrade_now_while_upgrades_are_off(self):
        # The tray path end to end: trigger_overnight -> the real
        # _act_start_overnight_upgrade. It must not sleep JARVIS or arm the
        # 8 h flag, and the refusal is shown to the owner.
        bc = self.bc
        flag = os.path.join(self.tmp, ".overnight_active")
        self._p(bc, "OVERNIGHT_UPGRADE_ENABLED", False)
        self._p(bc, "OVERNIGHT_FLAG_FILE", flag)
        self._p(bc, "_write_hud_state")
        sleep = bc._sleep_mode
        saved = sleep[0]
        self.addCleanup(lambda: sleep.__setitem__(0, saved))
        sleep[0] = False
        run_now = bc._overnight_run_now
        run_now.clear()
        self._quiet(bc._dispatch_tray_command, "trigger_overnight",
                    {"cmd": "trigger_overnight", "rid": "r1-7"})
        self.assertFalse(sleep[0], "JARVIS was put to sleep for a disabled upgrade")
        self.assertFalse(run_now.is_set())
        self.assertFalse(os.path.exists(flag))
        self.assertIn("switched off", self._results()[-1]["text"])


# ── de-duplication ─────────────────────────────────────────────────────────
class TrayCommandDedupeTests(_Base):
    def _inflight(self, cmds):
        path = os.path.join(self.tmp, "tray_commands.json.inflight")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cmds, f)
        return path

    def test_a_reappended_command_runs_once(self):
        bc = self.bc
        muted = bc._tts_muted
        saved = muted[0]
        self.addCleanup(lambda: muted.__setitem__(0, saved))
        muted[0] = False
        self._p(bc, "_write_hud_state")
        entry = {"cmd": "mute_tts_toggle", "ts": time.time(), "cid": "c77-1"}
        self._quiet(bc._process_inflight, self._inflight([entry]))
        self.assertTrue(muted[0])
        # The tray's read raced the claim and wrote the SAME entry back:
        _, log = self._quiet(bc._process_inflight, self._inflight([entry]))
        self.assertTrue(muted[0], "the duplicate toggled mute straight back off")
        self.assertIn("skipped duplicate", log)

    def test_entries_without_a_cid_are_not_deduplicated(self):
        bc = self.bc
        muted = bc._tts_muted
        saved = muted[0]
        self.addCleanup(lambda: muted.__setitem__(0, saved))
        muted[0] = False
        self._p(bc, "_write_hud_state")
        entry = {"cmd": "mute_tts_toggle", "ts": time.time()}
        self._quiet(bc._process_inflight, self._inflight([entry]))
        self._quiet(bc._process_inflight, self._inflight([entry]))
        self.assertFalse(muted[0])


# ── the restart inbox ──────────────────────────────────────────────────────
class LaunchInboxPruneTests(_Base):
    def _inbox(self, cmds):
        path = os.path.join(self.tmp, "tray_commands.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cmds, f)
        return path

    def _read(self, path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_keeps_recent_clicks_drops_stale_and_lifecycle(self):
        now = time.time()
        path = self._inbox([
            {"cmd": "mic_mute_toggle", "ts": now - 5, "cid": "a"},
            {"cmd": "mute_tts_toggle", "ts": now - 3600, "cid": "b"},
            {"cmd": "restart", "ts": now - 2, "cid": "c"},
            {"cmd": "shutdown_jarvis", "ts": now - 1, "cid": "d"},
            {"cmd": "trigger_overnight", "ts": now - 1, "cid": "e"},
            "junk",
        ])
        kept, _ = self._quiet(self.bc._prune_stale_tray_commands, path, now)
        self.assertEqual(kept, 1)
        self.assertEqual([c["cid"] for c in self._read(path)], ["a"])

    def test_all_stale_removes_the_file(self):
        path = self._inbox([{"cmd": "restart", "ts": time.time()}])
        self._quiet(self.bc._prune_stale_tray_commands, path)
        self.assertFalse(os.path.exists(path))

    def test_unreadable_inbox_is_removed(self):
        path = os.path.join(self.tmp, "tray_commands.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{{ not json")
        self._quiet(self.bc._prune_stale_tray_commands, path)
        self.assertFalse(os.path.exists(path))

    def test_launch_keeps_a_click_made_during_the_restart(self):
        bc = self.bc
        path = self._inbox([{"cmd": "mic_mute_toggle", "ts": time.time() - 3,
                             "cid": "keep-me"}])
        proc = mock.Mock(pid=31337)
        self._p(bc, "TRAY_COMMANDS_FILE", path)
        self._p(bc, "TRAY_ENABLED", True)
        self._p(bc, "_tray_process", None)
        self._p(bc, "_publish_tray_boot_info")
        self._p(bc, "_publish_audio_state")
        with mock.patch.object(bc.subprocess, "Popen", return_value=proc):
            self._quiet(bc._launch_tray)
        self.assertEqual([c["cid"] for c in self._read(path)], ["keep-me"])


# ── hud_state for the tray ─────────────────────────────────────────────────
class HudOffTrayStillLiveTests(_Base):
    def test_hud_off_still_feeds_the_tray(self):
        bc = self.bc
        path = os.path.join(self.tmp, "hud_state.json")
        cache = bc._hud_state_cache
        snap = dict(cache)
        self.addCleanup(lambda: (cache.clear(), cache.update(snap)))
        with mock.patch.object(bc, "HUD_ENABLED", False), \
             mock.patch.object(bc, "TRAY_ENABLED", True), \
             mock.patch.object(bc, "HUD_STATE_FILE", path):
            bc._write_hud_state(mic_muted=True)
        with open(path, encoding="utf-8") as f:
            self.assertTrue(json.load(f)["mic_muted"],
                            "unticking the HUD froze the tray")


class BootInfoTests(_Base):
    def test_boot_info_is_published(self):
        bc = self.bc
        writes = []
        self._p(bc, "_write_hud_state", side_effect=lambda **k: writes.append(k))
        self._p(bc, "OVERNIGHT_UPGRADE_ENABLED", False)
        bc._publish_tray_boot_info()
        from core.version import version_string
        info = writes[-1]
        self.assertEqual(info["jarvis_pid"], os.getpid())
        self.assertEqual(info["jarvis_version"], version_string())
        self.assertEqual(info["boot_started_at"], float(bc._session_start_time))
        self.assertIs(info["overnight_upgrade_enabled"], False)

    def test_tray_launches_early_and_ready_marks_the_drainer(self):
        """Boot order, from main()'s source: the tray is launched right after
        setup_logging — before preflight / Whisper / skills (the 11-56 s the
        icon used to be missing) — and its drainer + tray_ready_pid come only
        after load_skills (ACTIONS complete)."""
        import inspect
        src = inspect.getsource(self.bc.main)
        launch = src.index("_launch_tray()")
        self.assertLess(src.index("setup_logging()"), launch)
        for later in ("_startup_preflight()", "_ensure_whisper()",
                      "load_skills()"):
            self.assertLess(launch, src.index(later),
                            f"the tray must launch before {later}")
        drainer = src.index("target=_tray_command_drainer")
        self.assertLess(src.index("load_skills()"), drainer)
        self.assertLess(drainer, src.index("tray_ready_pid=os.getpid()"))


class PublisherTests(_Base):
    def _one_iteration(self, **mods):
        bc = self.bc
        stop = bc._tray_publisher_stop
        stop.clear()

        def _wait(_):
            stop.set()
            return True
        writes = []
        with mock.patch.object(bc, "_write_hud_state",
                               side_effect=lambda **k: writes.append(k)), \
             mock.patch.dict(bc.sys.modules, mods), \
             mock.patch.object(stop, "wait", side_effect=_wait), \
             mock.patch.object(bc, "_hud_cal_last", [time.time()]):
            try:
                bc._tray_state_publisher()
            finally:
                stop.clear()
        merged = {}
        for w in writes:
            merged.update(w)
        return merged

    def test_publishes_standby_ambient_and_the_dashboard(self):
        bc = self.bc
        thread = mock.Mock()
        thread.is_alive.return_value = True
        amb = mock.Mock(_thread=thread)
        web = mock.Mock(_httpd=object(), _bound=("127.0.0.1", 8766))
        sleep, standby = bc._sleep_mode, bc._standby_mode
        saved = (sleep[0], standby[0])
        self.addCleanup(lambda: (sleep.__setitem__(0, saved[0]),
                                 standby.__setitem__(0, saved[1])))
        sleep[0] = standby[0] = True
        cache = bc._hud_state_cache
        snap = dict(cache)
        self.addCleanup(lambda: (cache.clear(), cache.update(snap)))
        for k in ("sleep_mode", "standby_mode", "ambient_listening", "web_port"):
            cache.pop(k, None)
        out = self._one_iteration(skill_ambient_listen=amb,
                                  skill_web_interface=web)
        self.assertIs(out.get("sleep_mode"), True)
        self.assertIs(out.get("standby_mode"), True)
        self.assertIs(out.get("ambient_listening"), True)
        self.assertEqual(out.get("web_port"), 8766)


class AmbientToggleTests(_Base):
    def test_toggle_stops_an_autostarted_daemon(self):
        # AMBIENT_LISTEN_ENABLED started the daemon at boot; the cell is False.
        bc = self.bc
        thread = mock.Mock()
        thread.is_alive.return_value = True
        amb = mock.Mock(_thread=thread)
        start, stop = mock.Mock(return_value="on"), mock.Mock(return_value="off")
        cell = bc._ambient_mode_active
        saved = cell[0]
        self.addCleanup(lambda: cell.__setitem__(0, saved))
        cell[0] = False
        self._p(bc, "_write_hud_state")
        with mock.patch.dict(bc.sys.modules, {"skill_ambient_listen": amb}), \
             mock.patch.dict(bc.ACTIONS, {"ambient_listen_start": start,
                                          "ambient_listen_stop": stop}):
            self._quiet(bc._dispatch_tray_command, "ambient_mode_toggle", {})
        stop.assert_called_once()
        start.assert_not_called()
        self.assertFalse(cell[0])

    def test_falls_back_to_the_cell_without_the_skill(self):
        bc = self.bc
        start = mock.Mock(return_value="on")
        cell = bc._ambient_mode_active
        saved = cell[0]
        self.addCleanup(lambda: cell.__setitem__(0, saved))
        cell[0] = False
        self._p(bc, "_write_hud_state")
        with mock.patch.dict(bc.sys.modules, {"skill_ambient_listen": None}), \
             mock.patch.dict(bc.ACTIONS, {"ambient_listen_start": start}):
            bc.sys.modules.pop("skill_ambient_listen", None)
            self._quiet(bc._dispatch_tray_command, "ambient_mode_toggle", {})
        start.assert_called_once()
        self.assertTrue(cell[0])


class ShowTrayTests(_Base):
    def test_registered_spoken_and_routable(self):
        bc = self.bc
        self.assertIs(bc.ACTIONS.get("show_tray"), bc._act_show_tray)
        self.assertIn("show_tray", bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        from core import prompt_router, prompts
        self.assertIn("show_tray", prompts.PC_CONTROL_PROMPT)
        slim = prompt_router.slim_pc_control("show the tray icon",
                                             prompts.PC_CONTROL_PROMPT)
        self.assertIn("show_tray", slim)

    def test_relaunches_a_quit_tray(self):
        bc = self.bc
        self._p(bc, "TRAY_ENABLED", True)
        self._p(bc, "_tray_process", mock.Mock(poll=mock.Mock(return_value=0)))
        self._p(bc, "_write_hud_state")

        def fake_launch():
            bc._tray_process = mock.Mock(poll=mock.Mock(return_value=None))
        launch = self._p(bc, "_launch_tray", side_effect=fake_launch)
        out = bc._act_show_tray("")
        launch.assert_called_once()
        self.assertIn("back", out)

    def test_already_there(self):
        bc = self.bc
        self._p(bc, "TRAY_ENABLED", True)
        self._p(bc, "_tray_process", mock.Mock(poll=mock.Mock(return_value=None)))
        launch = self._p(bc, "_launch_tray")
        self.assertIn("already", bc._act_show_tray(""))
        launch.assert_not_called()


class RecentFactsIsHeavyTests(_Base):
    def test_recent_facts_runs_off_the_drainer(self):
        # It reads the LTM store (may wait on its lock while it warms).
        self.assertIn("show_recent_facts", self.bc._HEAVY_ACTIONS)


# ── Mute Mic is immediate ──────────────────────────────────────────────────
class MicMuteMidCaptureTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.muted = bc._mic_muted
        saved = self.muted[0]
        self.addCleanup(lambda: self.muted.__setitem__(0, saved))
        self.muted[0] = False
        self._p(bc, "_write_hud_state")
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "should_be_proactive", return_value=False)
        self._p(bc, "MAX_RECORDING_SECS", 8.0)

    def _harness(self, feeder, transcript="turn off the lights"):
        bc = self.bc
        np = bc.np
        voiced = threading.Event()
        self._p(bc, "_prof", side_effect=lambda *a, **k: (
            voiced.set() if a and a[0] == "voiced" else None))

        class FakeStream:
            device = 1

            def __init__(self, *a, callback=None, **k):
                self.cb = callback

            def start(self):
                cb = self.cb

                def push(n, amp):
                    frame = np.full((1024, 1), amp, dtype="float32")
                    for _ in range(n):
                        cb(frame, 1024, None, None)

                def wait_vad():
                    voiced.wait(_WAIT)

                threading.Thread(target=feeder, args=(push, wait_vad),
                                 daemon=True, name="fake-mic").start()

        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "get_input_device", return_value=1)
        self._p(bc, "_safe_close_stream", lambda s: None)
        self._p(bc.sd, "InputStream", FakeStream)
        self._p(bc, "_note_live_capture", lambda *a, **k: None)
        self._p(bc, "_filler_capture_mark", lambda *a, **k: None)
        self._p(bc, "_process_capture_chunk",
                lambda data, sr, skip_ns=False: data)
        self._p(bc, "_spec_stt_should_snapshot", return_value=False)
        self._p(bc, "pause_face_tracking")
        self._p(bc, "resume_face_tracking")
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "_audio_music_feed")
        self.transcribe = self._p(
            bc, "_transcribe_capture",
            return_value=(transcript, {"no_speech_prob": 0.0,
                                       "avg_logprob": -0.1}))
        self._p(bc, "_get_realtime_session", return_value=None)
        self._p(bc, "VAD_THRESHOLD", 0.008)
        bc._utterance_in_progress[0] = False

    def _tray_mute(self):
        self._quiet(self.bc._dispatch_tray_command, "mic_mute_toggle", {})

    def test_mute_clicked_mid_utterance_stops_and_drops_it(self):
        def feeder(push, wait_vad):
            push(6, 0.2)             # the owner starts talking…
            wait_vad()
            self._tray_mute()        # …and clicks Mute Mic mid-sentence
            push(30, 0.2)            # still talking
            push(40, 0.0)            # then silence (would end the utterance)
        self._harness(feeder)
        t0 = time.time()
        cap, log = self._quiet(self.bc._capture_utterance, None, {})
        self.assertIsNone(cap, "a capture muted mid-way was still acted on")
        self.transcribe.assert_not_called()
        self.assertIn("mic muted mid-capture", log)
        self.assertLess(time.time() - t0, _WAIT)
        self.assertTrue(self.muted[0])
        self.assertFalse(self.bc._utterance_in_progress[0])

    def test_mute_during_transcription_drops_the_transcript(self):
        def feeder(push, wait_vad):
            push(6, 0.2)
            wait_vad()
            push(40, 0.0)

        self._harness(feeder)

        def stt(_audio):
            self._tray_mute()        # clicked while Whisper was running
            return ("turn off the lights", {"no_speech_prob": 0.0,
                                            "avg_logprob": -0.1})
        self.transcribe.side_effect = stt
        cap, log = self._quiet(self.bc._capture_utterance, None, {})
        self.assertIsNone(cap)
        self.assertIn("transcript dropped", log)
        self.assertNotIn("lights", log)       # numbers only, never the text

    def test_unmuted_capture_still_works(self):
        def feeder(push, wait_vad):
            push(6, 0.2)
            wait_vad()
            push(40, 0.0)
        self._harness(feeder, transcript="hello")
        cap, _ = self._quiet(self.bc._capture_utterance, None, {})
        self.assertEqual(cap[0], "hello")

    def test_capture_that_starts_muted_opens_nothing(self):
        # 2026-10-01: mute is a capture-ENTRY rule. A capture that STARTED
        # muted (the standby loop, a draft / printer confirmation) used to
        # record as normal; now no stream opens, and the brief idle keeps the
        # caller's loop from spinning.
        self.muted[0] = True
        opened = []

        def feeder(push, wait_vad):
            opened.append(True)
            push(6, 0.2)
            wait_vad()
            push(40, 0.0)
        self._harness(feeder)
        t0 = time.time()
        audio, _ = self._quiet(self.bc.record_speech, 5)
        self.assertIsNone(audio, "a capture that started muted still listened")
        self.assertEqual(opened, [], "a stream was opened while muted")
        self.assertGreaterEqual(time.time() - t0, 0.25)   # idles, not spins


if __name__ == "__main__":
    unittest.main()
