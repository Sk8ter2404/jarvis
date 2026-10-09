"""core/clone_voice_client.py -- the clone voice server client (2026-10-03).

Light tier (stdlib + numpy). Every request goes to a FAKE server on an
ephemeral loopback port (tests/_clone_voice_fake.py); the spawn is a fake
Popen; nothing is played, nothing touches a GPU.

Pins:
  * start: reuse a server that is up; spawn VOICE_CLONE_SERVER_CMD (hidden,
    its own process group, {ref}/{port} filled) when none answers, then wait
    for /health; ONE log line for every outcome; down at boot (no command,
    exit while loading, never ready, wrong voice, unconsented profile,
    something else on the port) never raises and never blocks the caller;
  * render: healthy -> trimmed, loudness-matched float32 audio; text is
    number-normalised before it is sent; slow -> times out inside its
    deadline; a dead server costs the capped connect, not ~2 s;
  * the cool-down (2026-10-04, was a latch for the whole session):
    MAX_FAILURES latency-critical misses in a row rest the clone for 5 min
    (doubling, capped at 30) with one log line per state change, then it is
    tried again; a success resets the count, rearm() ends it at once;
  * probation: right after a cool-down ONE counted miss rests the clone
    again (a dead or still-slow server costs one line per cool-down, not
    three); a success, or a healthy half hour, ends it;
  * the needed-by deadline: a line rendered ahead may wait until it is
    needed (never less than its budget), and -- once the reply speaks in
    the clone (hold) -- up to its budget past that; a look-ahead line given
    up on before it was needed does not count;
  * a stopped reply (cancel) ends a render's wait at once, uncounted;
  * hard errors (refused, HTTP, silent) always count; a background render
    (count=False, the filler clips) never counts and never resets;
  * the 10:36-10:37 replay (the briefing and the two one-line replies that
    latched it, as the server logged them) and a short opener before a long
    line: the live pattern keeps the clone voice;
  * the consent gate: the server is used only while it speaks the active
    consented profile's reference (by hash) -- re-checked after a
    cool-down;
  * the cache answers a repeated line without a request.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
import unittest
from unittest import mock

import numpy as np

from core import clone_voice_client as cvc
from core import voice_clone as vc
from core import sentence_tts as st
from tests._clone_voice_fake import (LIVE_1036_AUDIO_S, LIVE_1036_LINES,
                                     LIVE_1036_RENDER_S, LIVE_1037_AWAY,
                                     LIVE_1037_AWAY_PIECES,
                                     LIVE_1037_AWAY_RENDER_S,
                                     LIVE_1037_MORNING,
                                     LIVE_1037_MORNING_PIECES,
                                     LIVE_1037_MORNING_RENDER_S,
                                     FakeCloneServer, ProfileDir, free_port,
                                     live_1036_server, live_1037_timings,
                                     make_wav)


class _FakeProc:
    def __init__(self, rc=None):
        self.rc = rc

    def poll(self):
        return self.rc


class _Base(unittest.TestCase):
    def setUp(self):
        self.prof = ProfileDir("butler")
        self.addCleanup(self.prof.cleanup)
        p = mock.patch.object(vc, "PROFILES_DIR", self.prof.root)
        p.start()
        self.addCleanup(p.stop)
        self.logs: list = []
        self.servers: list = []

    def server(self, **kw) -> FakeCloneServer:
        kw.setdefault("ref_sha", self.prof.sha)
        s = FakeCloneServer(**kw).start()
        self.servers.append(s)
        self.addCleanup(s.stop)
        return s

    def client(self, **kw) -> cvc.CloneVoiceClient:
        kw.setdefault("log", self.logs.append)
        kw.setdefault("boot_wait_s", 3.0)
        kw.setdefault("sleep", lambda s: time.sleep(min(s, 0.02)))
        return cvc.CloneVoiceClient(**kw)

    def ready_client(self, srv=None, **kw):
        srv = srv or self.server()
        c = self.client(**kw)
        self.assertEqual(c.start(url=srv.url, cmd="", profile="butler"),
                         "ready", self.logs)
        self.logs.clear()
        return c, srv


# ════════════════════════════════════════════════════════════════════════════
#  Pure helpers
# ════════════════════════════════════════════════════════════════════════════
class PureHelperTests(unittest.TestCase):
    def test_parse_url_accepts_only_loopback_http(self):
        self.assertEqual(cvc.parse_url("http://127.0.0.1:8767"),
                         ("127.0.0.1", 8767))
        self.assertEqual(cvc.parse_url("http://[::1]:9000"), ("::1", 9000))
        # The localhost tax: never resolve 'localhost' (::1 first, ~2 s).
        self.assertEqual(cvc.parse_url("http://localhost:8767"),
                         ("127.0.0.1", 8767))
        for bad in ("http://192.168.1.5:8767", "https://127.0.0.1:8767",
                    "http://127.0.0.1", "http://example.com:80", "", None,
                    "127.0.0.1:8767", "ftp://127.0.0.1:21"):
            self.assertIsNone(cvc.parse_url(bad), bad)

    def test_is_server_model(self):
        self.assertTrue(cvc.is_server_model("chatterbox_turbo_server"))
        self.assertTrue(cvc.is_server_model(" Chatterbox_Turbo_Server "))
        for other in ("chatterbox", "", None, 3):
            self.assertFalse(cvc.is_server_model(other))

    def test_budget_grows_only_past_the_base_length(self):
        self.assertEqual(cvc.render_budget_s(10, 2.5), 2.5)
        self.assertEqual(cvc.render_budget_s(cvc.BASE_CHARS, 2.5), 2.5)
        self.assertAlmostEqual(cvc.render_budget_s(cvc.BASE_CHARS + 100, 2.5),
                               2.5 + 100 * cvc.PER_CHAR_S)

    def test_build_command_fills_ref_and_port(self):
        ref = os.path.join("C:" + os.sep, "voice dir", "reference.wav")
        self.assertIsNone(cvc.build_command("", ref, 8767))
        self.assertIsNone(cvc.build_command("   ", ref, 8767))
        self.assertEqual(
            cvc.build_command('["py", "srv.py", "--ref", "{ref}", "--port", '
                              '"{port}"]', ref, 8767),
            ["py", "srv.py", "--ref", ref, "--port", "8767"])
        out = cvc.build_command("py srv.py --ref {ref} --port {port}", ref, 9)
        if os.name == "nt":
            self.assertEqual(out, f'py srv.py --ref "{ref}" --port 9')
            # Already quoted by the owner: not quoted twice.
            self.assertEqual(
                cvc.build_command('py srv.py --ref "{ref}"', ref, 9),
                f'py srv.py --ref "{ref}"')
        else:
            self.assertEqual(out, ["py", "srv.py", "--ref", ref, "--port", "9"])

    def test_decode_wav_round_trip(self):
        a, sr = cvc.decode_wav(make_wav(sr=22050, lead_s=0, speech_s=0.1,
                                        tail_s=0, amp=0.5))
        self.assertEqual(sr, 22050)
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(a.size, 2205)
        self.assertAlmostEqual(float(np.max(np.abs(a))), 0.5, places=2)

    def test_finish_audio_trims_and_matches_loudness(self):
        sr = 24000
        a, _ = cvc.decode_wav(make_wav(sr=sr, lead_s=0.3, speech_s=0.5,
                                       tail_s=0.4, amp=0.03))
        out = cvc.finish_audio(a, sr)
        # Ends trimmed to the kept margins (+/- one sample of tone edge).
        expect = 0.5 + cvc.LEAD_KEEP_S + cvc.TAIL_KEEP_S
        self.assertAlmostEqual(out.size / sr, expect, delta=0.01)
        # The quiet tone is brought up to the target level.
        voiced = out[int(cvc.LEAD_KEEP_S * sr) + 10: -int(cvc.TAIL_KEEP_S * sr) - 10]
        rms = float(np.sqrt(np.mean(voiced ** 2)))
        self.assertAlmostEqual(rms, cvc.TARGET_RMS, delta=0.01)

    def test_finish_audio_caps_peak_and_gain(self):
        sr = 24000
        loud, _ = cvc.decode_wav(make_wav(sr=sr, lead_s=0, speech_s=0.3,
                                          tail_s=0, amp=0.9))
        self.assertLessEqual(float(np.max(np.abs(cvc.finish_audio(loud, sr)))),
                             cvc.PEAK_CAP + 1e-4)
        faint, _ = cvc.decode_wav(make_wav(sr=sr, lead_s=0, speech_s=0.3,
                                           tail_s=0, amp=0.002))
        out = cvc.finish_audio(faint, sr)
        self.assertLessEqual(float(np.max(np.abs(out))),
                             0.002 * cvc.MAX_GAIN + 1e-3)

    def test_finish_audio_refuses_silence(self):
        self.assertIsNone(cvc.finish_audio(np.zeros(2400, np.float32), 24000))
        self.assertIsNone(cvc.finish_audio(np.zeros(0, np.float32), 24000))
        self.assertIsNone(cvc.finish_audio(
            np.full(10, np.nan, dtype=np.float32), 24000))


# ════════════════════════════════════════════════════════════════════════════
#  Start (boot)
# ════════════════════════════════════════════════════════════════════════════
class StartTests(_Base):
    def test_reuses_a_server_that_is_already_up(self):
        srv = self.server()
        ready = []
        c = self.client(popen=mock.Mock(side_effect=AssertionError("spawned")))
        self.assertEqual(c.start(url=srv.url, cmd="never run", profile="butler",
                                 on_ready=lambda: ready.append(1)), "ready")
        self.assertEqual(c.status(), ("ready", ""))
        self.assertEqual(c.server_pid(), 4242)
        self.assertEqual(ready, [1])
        self.assertEqual(len(self.logs), 1, self.logs)
        self.assertIn("already running", self.logs[0])
        self.assertTrue(c.usable_for("butler"))

    def test_spawns_the_command_when_nothing_answers_then_waits(self):
        port = free_port()
        calls = []
        srv_box = []

        def popen(argv, **kw):
            calls.append((argv, kw))
            # The "server" comes up a moment later, on the port it was given.
            def _up():
                time.sleep(0.15)
                s = FakeCloneServer(ref_sha=self.prof.sha)
                srv_box.append(s)
                s.start(port=port)
            threading.Thread(target=_up, daemon=True).start()
            return _FakeProc(None)

        self.addCleanup(lambda: [s.stop() for s in srv_box])
        c = self.client(popen=popen)
        logp = os.path.join(self.prof.root, "logs", "server.log")
        with mock.patch.dict(os.environ, {"PYTHONPATH": "x"}):
            st = c.start(url=f"http://127.0.0.1:{port}",
                         cmd='["srv", "--ref", "{ref}", "--port", "{port}"]',
                         profile="butler", log_path=logp)
        self.assertEqual(st, "ready", self.logs)
        self.assertEqual(len(calls), 1)
        argv, kw = calls[0]
        self.assertEqual(argv, ["srv", "--ref", self.prof.ref, "--port",
                                str(port)])
        self.assertNotIn("PYTHONPATH", kw["env"])
        if os.name == "nt":
            # getattr: tools/run_tests_ci_sim.py removes CREATE_NO_WINDOW to
            # imitate the Linux runner.
            import subprocess
            self.assertEqual(
                kw["creationflags"],
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            self.assertFalse(kw["creationflags"]
                             & getattr(subprocess, "DETACHED_PROCESS", 0))
        else:
            self.assertTrue(kw["start_new_session"])
        self.assertTrue(os.path.isdir(os.path.dirname(logp)))
        self.assertEqual(len(self.logs), 1, self.logs)
        self.assertIn("started", self.logs[0])

    def test_down_at_boot_without_a_command_logs_one_line(self):
        c = self.client(popen=mock.Mock(side_effect=AssertionError("spawned")))
        t0 = time.monotonic()
        st = c.start(url=f"http://127.0.0.1:{free_port()}", cmd="",
                     profile="butler")
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(st, "down")
        self.assertEqual(c.status()[0], "down")
        self.assertEqual(len(self.logs), 1, self.logs)
        self.assertIn("VOICE_CLONE_SERVER_CMD is empty", self.logs[0])
        self.assertIn("Kokoro keeps speaking", self.logs[0])
        self.assertFalse(c.usable_for("butler"))
        self.assertEqual(c.render("Hello.", 2.5).reason, "not-ready")

    def test_server_that_exits_while_loading(self):
        c = self.client(popen=lambda argv, **kw: _FakeProc(3))
        st = c.start(url=f"http://127.0.0.1:{free_port()}", cmd="srv {ref}",
                     profile="butler", log_path=None)
        self.assertEqual(st, "down")
        self.assertEqual(len(self.logs), 1, self.logs)
        self.assertIn("exited with code 3", self.logs[0])

    def test_a_second_instance_that_exits_adopts_the_one_holding_the_port(self):
        # Two starters can race (a JARVIS restart while the server it just
        # spawned is not yet listening): the late spawn cannot bind, exits at
        # once (the real server: code 4, measured 121 ms, before any GPU
        # work) -- and the instance that holds the port is waited for, not
        # written off as "down" for the session.
        port = free_port()              # nothing listening at first...
        url = f"http://127.0.0.1:{port}"
        later = []

        def popen(argv, **kw):
            # ...then the OTHER instance binds the port and loads; ours
            # exits with "cannot bind".
            s = FakeCloneServer(ref_sha=self.prof.sha, health_code=503,
                                ok=False).start(port=port)
            later.append(s)

            def _ready():
                time.sleep(0.15)
                s.health_code, s.ok = 200, True
            threading.Thread(target=_ready, daemon=True).start()
            return _FakeProc(4)

        self.addCleanup(lambda: [s.stop() for s in later])
        c = self.client(popen=popen)
        self.assertEqual(c.start(url=url, cmd="srv {ref} {port}",
                                 profile="butler"), "ready", self.logs)
        self.assertEqual(len(self.logs), 1, self.logs)
        self.assertNotIn("exited", self.logs[0])
        # Not ours: never asked to stop.
        self.assertEqual(later[0].count("POST", "/shutdown"), 0)

    def test_server_that_never_comes_up_is_bounded(self):
        c = self.client(popen=lambda argv, **kw: _FakeProc(None),
                        boot_wait_s=0.3)
        t0 = time.monotonic()
        st = c.start(url=f"http://127.0.0.1:{free_port()}", cmd="srv",
                     profile="butler")
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertEqual(st, "down")
        self.assertEqual(len(self.logs), 1, self.logs)
        self.assertIn("did not come up within", self.logs[0])

    def test_waits_for_a_server_that_is_still_loading(self):
        srv = self.server(health_code=503, ok=False)

        def _ready_later():
            time.sleep(0.15)
            srv.health_code, srv.ok = 200, True
        threading.Thread(target=_ready_later, daemon=True).start()
        c = self.client(popen=mock.Mock(side_effect=AssertionError("spawned")))
        self.assertEqual(c.start(url=srv.url, cmd="x", profile="butler"),
                         "ready")
        self.assertIn("loaded", self.logs[0])

    def test_a_different_voice_is_never_used(self):
        srv = self.server(ref_sha="0" * 64)
        c = self.client()
        self.assertEqual(c.start(url=srv.url, cmd="", profile="butler"), "down")
        self.assertIn("different voice prompt", self.logs[0])
        self.assertEqual(srv.count("POST", "/shutdown"), 0)   # not ours
        self.assertFalse(c.usable_for("butler"))

    def test_a_server_that_hides_its_voice_is_never_used(self):
        srv = self.server(ref_sha="")
        c = self.client()
        self.assertEqual(c.start(url=srv.url, cmd="", profile="butler"), "down")
        self.assertIn("ref_sha256", self.logs[0])

    def test_unconsented_profile_never_starts_anything(self):
        bad = ProfileDir("other", consent=None)
        self.addCleanup(bad.cleanup)
        with mock.patch.object(vc, "PROFILES_DIR", bad.root):
            c = self.client(popen=mock.Mock(side_effect=AssertionError("spawn")))
            st = c.start(url=f"http://127.0.0.1:{free_port()}", cmd="srv",
                         profile="other")
        self.assertEqual(st, "down")
        self.assertIn("no consented voice profile", self.logs[0])

    def test_something_else_on_the_port(self):
        srv = self.server(health_code=404)
        c = self.client()
        self.assertEqual(c.start(url=srv.url, cmd="srv", profile="butler"),
                         "down")
        self.assertIn("something else answers", self.logs[0])

    def test_bad_url_is_down_without_any_request(self):
        c = self.client()
        self.assertEqual(c.start(url="http://10.0.0.5:8767", cmd="srv",
                                 profile="butler"), "down")
        self.assertIn("loopback", self.logs[0])

    def test_start_async_is_single_flight_and_never_blocks(self):
        made = []

        class _T:
            def __init__(self, target=None, kwargs=None, name=None,
                         daemon=None):
                made.append((target, kwargs, name, daemon))

            def start(self):
                pass
        c = self.client()
        self.assertTrue(c.start_async(url="http://127.0.0.1:1", cmd="",
                                      profile="butler", thread_factory=_T))
        self.assertEqual(c.status()[0], "starting")
        self.assertFalse(c.start_async(url="http://127.0.0.1:1", cmd="",
                                       profile="butler", thread_factory=_T))
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0][2], "clone-voice-start")
        self.assertTrue(made[0][3])


# ════════════════════════════════════════════════════════════════════════════
#  Render
# ════════════════════════════════════════════════════════════════════════════
class RenderTests(_Base):
    def test_healthy_render(self):
        c, srv = self.ready_client()
        out = c.render("Very good, sir.", 2.5)
        self.assertTrue(out.ok, out.reason)
        self.assertEqual(out.sr, 24000)
        self.assertEqual(out.audio.dtype, np.float32)
        self.assertFalse(out.cached)
        self.assertEqual(out.server_ms, "12.5")
        # 0.5 s of tone, trimmed ends kept to the margins.
        self.assertLess(out.audio.size / 24000, 0.5 + 0.2)
        self.assertEqual(srv.tts_texts(), ["Very good, sir."])
        tts = [r for r in srv.requests if r[1] == "/tts"]
        self.assertEqual(tts[0][3], "application/json")
        self.assertEqual(c.failures(), 0)

    def test_numbers_are_spelled_before_sending(self):
        c, srv = self.ready_client()
        c.render("It is 2:07 PM.", 2.5)
        self.assertEqual(srv.tts_texts(), ["It is two oh seven p m."])

    def test_slow_render_times_out_inside_its_deadline(self):
        c, srv = self.ready_client(self.server(tts_delay=5.0))
        t0 = time.monotonic()
        out = c.render("Right away, sir.", 0.3)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertFalse(out.ok)
        self.assertTrue(out.attempted)
        self.assertIn("timed out", out.reason)
        self.assertEqual(c.failures(), 1)
        self.assertEqual(c.status()[0], "ready")

    def test_server_gone_costs_the_capped_connect(self):
        c, srv = self.ready_client()
        srv.stop()
        t0 = time.monotonic()
        out = c.render("Hello there.", 2.5)
        self.assertLess(time.monotonic() - t0, cvc.CONNECT_TIMEOUT_S + 0.5)
        self.assertFalse(out.ok)
        self.assertIn("ConnectionError", out.reason)
        self.assertTrue(out.counted)               # a hard error counts
        self.assertEqual(c.failures(), 1)

    def test_http_error_is_a_failure(self):
        c, srv = self.ready_client(self.server(tts_status=500))
        out = c.render("Hello.", 2.5)
        self.assertEqual(out.reason, "http 500")
        self.assertEqual(c.failures(), 1)

    def test_silent_render_is_a_failure(self):
        silent = make_wav(lead_s=0.1, speech_s=0.0, tail_s=0.2)
        c, srv = self.ready_client(self.server(wav=silent))
        out = c.render("Hello.", 2.5)
        self.assertFalse(out.ok)
        self.assertIn("silent", out.reason)
        self.assertTrue(out.counted)
        self.assertEqual(c.failures(), 1)

    def test_hard_errors_count_for_a_line_rendered_ahead_too(self):
        # Refused, silent: the server is broken, however much audio was
        # still queued ahead of the line.
        silent = make_wav(lead_s=0.1, speech_s=0.0, tail_s=0.2)
        c, srv = self.ready_client(self.server(wav=silent))
        out = c.render("Ahead.", 2.5, needed_by=time.monotonic() + 20.0)
        self.assertIn("silent", out.reason)
        self.assertTrue(out.lookahead and out.counted)
        srv.stop()
        out = c.render("Ahead again.", 2.5,
                       needed_by=time.monotonic() + 20.0)
        self.assertIn("ConnectionError", out.reason)
        self.assertTrue(out.lookahead and out.counted)
        self.assertEqual(c.failures(), 2)

    def test_a_server_that_is_gone_rests_the_clone_after_three_lines(self):
        c, srv = self.ready_client()
        srv.stop()
        for i in range(cvc.MAX_FAILURES):
            self.assertEqual(c.status()[0], "ready")
            out = c.render(f"Anyone there {i}?", 2.5)
            self.assertIn("ConnectionError", out.reason)
        self.assertEqual(c.status()[0], "cooldown")
        self.assertEqual(len([m for m in self.logs if "rests for" in m]), 1)

    def test_too_long_is_not_sent_and_not_a_failure(self):
        c, srv = self.ready_client()
        out = c.render("word " * 300, 2.5)
        self.assertEqual(out.reason, "too-long")
        self.assertFalse(out.attempted)
        self.assertEqual(srv.tts_texts(), [])
        self.assertEqual(c.failures(), 0)

    def test_empty_and_not_ready(self):
        c = self.client()
        self.assertEqual(c.render("   ", 2.5).reason, "empty")
        self.assertEqual(c.render("Hello.", 2.5).reason, "not-ready")
        self.assertFalse(c.render("Hello.", 2.5).attempted)

    def test_cache_answers_a_repeated_line_without_a_request(self):
        c, srv = self.ready_client()
        a = c.render("Right away, sir.", 2.5)
        b = c.render("Right away, sir.", 2.5)
        self.assertTrue(b.ok and b.cached)
        self.assertEqual(b.ms, 0)
        np.testing.assert_array_equal(a.audio, b.audio)
        b.audio[:] = 0                         # a copy: the cache is intact
        self.assertTrue(np.any(c.render("Right away, sir.", 2.5).audio))
        c.render("Something else.", 2.5)
        self.assertEqual(srv.tts_texts(), ["Right away, sir.",
                                           "Something else."])

    def test_cache_is_bounded(self):
        c, srv = self.ready_client()
        with mock.patch.object(cvc, "CACHE_MAX_BYTES", 200_000):
            for i in range(6):
                self.assertTrue(c.render(f"Line number {i}.", 2.5).ok)
        self.assertLess(c.cache_len(), 6)


# ════════════════════════════════════════════════════════════════════════════
#  The cool-down (was: latch-off for the session) and rearm
# ════════════════════════════════════════════════════════════════════════════
class LatchTests(_Base):
    def test_max_failures_in_a_row_start_a_cooldown_with_one_line(self):
        c, srv = self.ready_client(self.server(tts_status=500))
        for _ in range(cvc.MAX_FAILURES):
            self.assertFalse(c.render("Hello.", 2.5).ok)
        self.assertEqual(c.status()[0], "cooldown")
        rest_lines = [m for m in self.logs if "rests for 5 min" in m]
        self.assertEqual(len(rest_lines), 1, self.logs)
        self.assertNotIn("for this session", " ".join(self.logs))
        n = srv.count("POST", "/tts")
        out = c.render("Hello.", 2.5)
        self.assertEqual(out.reason, "not-ready")     # no request any more
        self.assertEqual(srv.count("POST", "/tts"), n)
        self.assertFalse(c.usable_for("butler"))
        for _ in range(3):
            c.render("Hello.", 2.5)
        self.assertEqual(len([m for m in self.logs if "rests for" in m]), 1)

    def test_a_success_resets_the_count(self):
        srv = self.server()
        c, _ = self.ready_client(srv)
        srv.tts_status = 500
        for _ in range(cvc.MAX_FAILURES - 1):
            c.render("Fail.", 2.5)
        srv.tts_status = 200
        self.assertTrue(c.render("Works.", 2.5).ok)
        self.assertEqual(c.failures(), 0)
        srv.tts_status = 500
        for _ in range(cvc.MAX_FAILURES - 1):
            c.render("Fail again.", 2.5)
        self.assertEqual(c.status()[0], "ready")

    def test_cache_hits_do_not_reset_the_failure_streak(self):
        # A cached stock line says nothing about the server's health. If a
        # cache hit reset the count, a hung server would never latch while
        # acks ("Very good, sir.") came between the answers, and EVERY new
        # line would wait out its whole deadline for the rest of the session.
        # Since 2026-10-05 a cached take does not OPEN a reply while a miss
        # is pending (it renders, and counts, like any line: a reply must
        # not open in the clone and miss to Kokoro); a LOOK-AHEAD hit -- a
        # reply already in the clone -- is still served. Neither resets it.
        srv = self.server()
        c, _ = self.ready_client(srv)
        self.assertTrue(c.render("Very good, sir.", 2.5).ok)
        srv.tts_status = 500
        for i in range(cvc.MAX_FAILURES):
            self.assertEqual(c.status()[0], "ready")
            self.assertFalse(c.render(f"New line {i}.", 2.5).ok)
            if c.status()[0] == "ready":
                hit = c.render("Very good, sir.", 2.5,
                               needed_by=time.monotonic() + 5.0)
                self.assertTrue(hit.ok and hit.cached)
        self.assertEqual(c.status()[0], "cooldown", c.failures())
        # A hit still reports success to its caller while the streak runs.
        c2, srv2 = self.ready_client(self.server())
        self.assertTrue(c2.render("Right away, sir.", 2.5).ok)
        srv2.tts_status = 500
        c2.render("Fails once.", 2.5)
        self.assertTrue(c2.render("Right away, sir.", 2.5,
                                  needed_by=time.monotonic() + 5.0).cached)
        self.assertEqual(c2.failures(), 1)
        # ...but it does not open a reply now: rendered, and missed.
        out = c2.render("Right away, sir.", 2.5)
        self.assertFalse(out.ok or out.cached)
        self.assertEqual(c2.failures(), 2)

    def test_rearm_from_cooldown_or_down(self):
        c, srv = self.ready_client(self.server(tts_status=500))
        for _ in range(cvc.MAX_FAILURES):
            c.render("Hello.", 2.5)
        self.assertEqual(c.status()[0], "cooldown")
        self.assertTrue(c.rearm())
        self.assertEqual(c.status(), ("idle", ""))
        self.assertEqual(c.failures(), 0)
        self.assertFalse(c.rearm())          # idle: nothing to undo
        c2 = self.client()
        c2.start(url="http://10.0.0.1:1", cmd="", profile="butler")
        self.assertTrue(c2.rearm())


# ════════════════════════════════════════════════════════════════════════════
#  The cool-down's clock: 5 min, doubling, capped at 30, then tried again
# ════════════════════════════════════════════════════════════════════════════
class CooldownTests(_Base):
    """A fake clock: the cool-down is minutes long. The fake server answers
    HTTP 500 at once (a hard error), so no render waits on the clock."""

    def setUp(self):
        super().setUp()
        self.now = [5000.0]
        self.srv = self.server(tts_status=500)
        self.c, _ = self.ready_client(self.srv, clock=lambda: self.now[0])

    def miss(self, n=cvc.MAX_FAILURES):
        for i in range(n):
            self.assertFalse(self.c.render(f"Miss {self.now[0]} {i}.", 2.5).ok)

    def test_the_first_cooldown_is_five_minutes_then_the_clone_is_tried(self):
        self.miss()
        self.assertEqual(self.c.status()[0], "cooldown")
        self.assertAlmostEqual(self.c.cooldown_left_s(), cvc.COOLDOWN_BASE_S)
        self.assertEqual(cvc.COOLDOWN_BASE_S, 300.0)
        self.now[0] += cvc.COOLDOWN_BASE_S - 1
        self.assertFalse(self.c.usable_for("butler"))
        self.assertEqual(self.c.render("Still resting.", 2.5).reason,
                         "not-ready")
        self.now[0] += 1
        self.assertTrue(self.c.usable_for("butler"))    # tried again
        self.assertEqual(self.c.status(), ("ready", ""))
        self.assertEqual(self.c.failures(), 0)
        over = [m for m in self.logs if "cool-down over" in m]
        self.assertEqual(len(over), 1, self.logs)
        self.srv.tts_status = 200
        self.assertTrue(self.c.render("Back again.", 2.5).ok)

    def test_each_further_cooldown_doubles_up_to_thirty_minutes(self):
        seen = []
        for _ in range(6):
            self.miss()
            self.assertEqual(self.c.status()[0], "cooldown")
            left = self.c.cooldown_left_s()
            seen.append(left)
            self.now[0] += left
            self.assertEqual(self.c.status()[0], "ready")
        self.assertEqual(seen, [300.0, 600.0, 1200.0, 1800.0, 1800.0, 1800.0])
        self.assertEqual(cvc.COOLDOWN_MAX_S, 1800.0)

    def test_the_doubling_starts_over_after_a_healthy_half_hour(self):
        self.miss()
        self.now[0] += self.c.cooldown_left_s()
        self.miss()
        self.assertAlmostEqual(self.c.cooldown_left_s(), 600.0)
        self.now[0] += self.c.cooldown_left_s()
        self.assertEqual(self.c.status()[0], "ready")
        self.now[0] += cvc.COOLDOWN_MAX_S            # a healthy half hour
        self.miss()
        self.assertAlmostEqual(self.c.cooldown_left_s(), 300.0)

    def test_one_log_line_per_state_change(self):
        self.miss()
        self.miss()                    # no requests while resting
        self.assertEqual(self.c.status()[0], "cooldown")
        for _ in range(5):
            self.c.status()
            self.c.usable_for("butler")
        self.now[0] += 300
        for _ in range(5):
            self.c.status()
            self.c.usable_for("butler")
        self.assertEqual(len(self.logs), 2, self.logs)
        self.assertIn("rests for 5 min", self.logs[0])
        self.assertIn("Kokoro speaks until then", self.logs[0])
        self.assertIn("cool-down over", self.logs[1])

    def test_rearm_ends_a_cooldown_and_resets_the_doubling(self):
        self.miss()
        self.now[0] += self.c.cooldown_left_s()
        self.miss()
        self.assertAlmostEqual(self.c.cooldown_left_s(), 600.0)
        self.assertTrue(self.c.rearm())
        self.assertEqual(self.c.status()[0], "idle")
        self.assertEqual(self.c.start(url=self.srv.url, cmd="",
                                      profile="butler"), "ready")
        self.miss(1)                                   # no probation
        self.assertEqual(self.c.status()[0], "ready")
        self.miss(cvc.MAX_FAILURES - 1)
        self.assertAlmostEqual(self.c.cooldown_left_s(), 300.0)

    # -- probation (2026-10-04 review) ---------------------------------------
    def test_one_miss_right_after_a_cooldown_rests_the_clone_again(self):
        # A server still slow or broken when the rest ends costs ONE more
        # degraded line, not MAX_FAILURES of them, before the next rest.
        self.miss()
        self.now[0] += self.c.cooldown_left_s()
        self.assertEqual(self.c.status()[0], "ready")
        self.assertEqual(self.c.failures(), 0)
        out = self.c.render("Still broken.", 2.5)
        self.assertTrue(out.counted)
        self.assertEqual(self.c.status()[0], "cooldown")
        self.assertAlmostEqual(self.c.cooldown_left_s(), 600.0)   # doubled
        rests = [m for m in self.logs if "rests for" in m]
        self.assertEqual(len(rests), 2, self.logs)
        self.assertIn("the first line after a cool-down missed too",
                      rests[1])
        self.assertIn("rests for 10 min", rests[1])

    def test_a_success_ends_the_probation(self):
        self.miss()
        self.now[0] += self.c.cooldown_left_s()
        self.srv.tts_status = 200
        self.assertTrue(self.c.render("Back again.", 2.5).ok)
        self.srv.tts_status = 500
        self.miss(cvc.MAX_FAILURES - 1)
        self.assertEqual(self.c.status()[0], "ready")
        self.miss(1)
        self.assertEqual(self.c.status()[0], "cooldown")
        # Still within half an hour of the last rest: the doubling holds.
        self.assertAlmostEqual(self.c.cooldown_left_s(), 600.0)

    def test_the_probation_lapses_after_a_healthy_half_hour(self):
        self.miss()
        self.now[0] += self.c.cooldown_left_s()
        self.assertEqual(self.c.status()[0], "ready")
        self.now[0] += cvc.COOLDOWN_MAX_S
        self.miss(1)
        self.assertEqual(self.c.status()[0], "ready")
        self.miss(cvc.MAX_FAILURES - 1)
        self.assertAlmostEqual(self.c.cooldown_left_s(), 300.0)

    def test_a_server_gone_when_the_rest_ends_costs_one_capped_connect(self):
        self.miss()
        self.srv.stop()                     # the server died meanwhile
        self.now[0] += self.c.cooldown_left_s()
        self.assertEqual(self.c.status()[0], "ready")
        t0 = time.monotonic()
        out = self.c.render("Anyone there?", 2.5)
        self.assertLess(time.monotonic() - t0, cvc.CONNECT_TIMEOUT_S + 0.5)
        self.assertIn("ConnectionError", out.reason)
        self.assertEqual(self.c.status()[0], "cooldown")
        self.assertAlmostEqual(self.c.cooldown_left_s(), 600.0)


class CooldownEdgeTests(_Base):
    def test_a_failure_from_a_render_in_flight_changes_nothing(self):
        # A render that was already waiting when the cool-down began fails
        # afterwards: no second log line, the rest is not doubled.
        srv = self.server(tts_status=500,
                          latency_for=lambda t: 1.0 if t == "In flight." else 0)
        c, _ = self.ready_client(srv)
        box = {}
        th = threading.Thread(
            target=lambda: box.setdefault("out", c.render("In flight.", 2.5)))
        th.start()
        time.sleep(0.1)
        for i in range(cvc.MAX_FAILURES):
            c.render(f"Quick miss {i}.", 2.5)
        self.assertEqual(c.status()[0], "cooldown")
        th.join(5.0)
        self.assertEqual(box["out"].reason, "http 500")
        self.assertEqual(c.status()[0], "cooldown")
        self.assertEqual(len([m for m in self.logs if "rests for" in m]), 1,
                         self.logs)
        self.assertLessEqual(c.cooldown_left_s(), cvc.COOLDOWN_BASE_S)

    def test_consent_revoked_during_a_cooldown_is_still_refused_after_it(self):
        now = [2000.0]
        c, srv = self.ready_client(self.server(tts_status=500),
                                   clock=lambda: now[0])
        for i in range(cvc.MAX_FAILURES):
            c.render(f"Miss {i}.", 2.5)
        self.assertEqual(c.status()[0], "cooldown")
        import json
        meta = os.path.join(self.prof.root, "butler", "meta.json")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"name": "butler", "source": "character"}, f)
        now[0] += cvc.COOLDOWN_BASE_S + cvc.PROFILE_TTL_S + 1
        self.assertEqual(c.status()[0], "ready")      # the rest is over ...
        self.assertFalse(c.usable_for("butler"))       # ... consent is not


# ════════════════════════════════════════════════════════════════════════════
#  The needed-by deadline (2026-10-04)
# ════════════════════════════════════════════════════════════════════════════
class LineDeadlineTests(unittest.TestCase):
    def test_a_first_line_gets_its_latency_budget(self):
        self.assertEqual(cvc.line_deadline_s(2.5, None, 100.0), (2.5, False))

    def test_a_line_rendered_ahead_may_take_until_it_is_needed(self):
        wait, by_need = cvc.line_deadline_s(2.5, 108.0, 100.0)
        self.assertAlmostEqual(wait, 8.0 - cvc.NEEDED_BY_MARGIN_S)
        self.assertTrue(by_need)

    def test_never_less_than_the_budget(self):
        self.assertEqual(cvc.line_deadline_s(2.5, 101.0, 100.0), (2.5, False))
        self.assertEqual(cvc.line_deadline_s(2.5, 90.0, 100.0), (2.5, False))

    def test_never_more_than_the_cap(self):
        wait, by_need = cvc.line_deadline_s(2.5, 1000.0, 100.0)
        self.assertEqual(wait, cvc.LOOKAHEAD_MAX_S)
        self.assertTrue(by_need)
        self.assertEqual(cvc.line_deadline_s(2.5, 1000.0, 100.0, hold=True),
                         (cvc.LOOKAHEAD_MAX_S, True))

    def test_the_cap_never_shortens_the_budget(self):
        # A long line at a long VOICE_CLONE_TIMEOUT_S (30 s allowed) has a
        # budget above the cap; the cap must not cut it.
        budget = cvc.render_budget_s(200, 30.0)
        self.assertGreater(budget, cvc.LOOKAHEAD_MAX_S)
        for hold in (False, True):
            self.assertEqual(
                cvc.line_deadline_s(budget, 140.0, 100.0, hold=hold),
                (budget, False))

    def test_a_bad_needed_by_is_the_budget(self):
        self.assertEqual(cvc.line_deadline_s(2.5, "soon", 100.0), (2.5, False))
        self.assertEqual(cvc.line_deadline_s(2.5, float("nan"), 100.0,
                                             hold=True), (2.5, False))

    def test_held_a_line_may_run_its_budget_past_the_time_it_is_needed(self):
        # The reply already speaks in the clone: a pause no longer than the
        # line's own budget beats a change of voice.
        wait, by_need = cvc.line_deadline_s(2.5, 101.0, 100.0, hold=True)
        self.assertAlmostEqual(wait, 1.0 + 2.5)
        self.assertTrue(by_need)
        # Not held: it must be back NEEDED_BY_MARGIN_S before it is needed.
        self.assertEqual(cvc.line_deadline_s(2.5, 101.0, 100.0), (2.5, False))
        # Needed already (the queue ran dry): the budget from now.
        self.assertEqual(cvc.line_deadline_s(2.5, 99.0, 100.0, hold=True),
                         (2.5, False))


class LookAheadRenderTests(_Base):
    def test_a_line_rendered_ahead_waits_past_its_budget_until_needed(self):
        # The 10:36 case in one line: slower than the budget, but seconds of
        # earlier audio still queued ahead of it.
        c, srv = self.ready_client(self.server(tts_delay=0.6))
        first = c.render("A first line, sir.", 0.3)
        self.assertFalse(first.ok)
        self.assertIn("timed out", first.reason)
        self.assertTrue(first.counted)
        self.assertFalse(first.lookahead)
        self.assertAlmostEqual(first.deadline_s, 0.3)
        out = c.render("A line rendered ahead.", 0.3,
                       needed_by=time.monotonic() + 2.0)
        self.assertTrue(out.ok, out.reason)
        self.assertTrue(out.lookahead and out.by_need)
        self.assertGreater(out.deadline_s, 1.0)
        self.assertGreater(out.ms, 300)
        self.assertEqual(c.failures(), 0)          # a success resets it

    def test_a_lookahead_miss_past_its_needed_by_counts(self):
        c, srv = self.ready_client(self.server(tts_delay=2.0))
        t0 = time.monotonic()
        out = c.render("Needed almost at once.", 0.3,
                       needed_by=t0 + cvc.NEEDED_BY_MARGIN_S + 0.1)
        self.assertLess(time.monotonic() - t0, 1.2)
        self.assertFalse(out.ok)
        self.assertTrue(out.lookahead)
        self.assertFalse(out.by_need)              # the budget was longer
        self.assertTrue(out.counted)
        self.assertEqual(c.failures(), 1)

    def test_a_lookahead_that_waited_until_needed_and_missed_counts(self):
        # The wait was set by the needed-by time (not the budget, not the
        # cap, not held) and the line still did not come back: it was due
        # and Kokoro voices it mid-reply -- a counted miss.
        c, srv = self.ready_client(self.server(tts_delay=2.0))
        t0 = time.monotonic()
        out = c.render("Due and missed.", 0.2,
                       needed_by=t0 + cvc.NEEDED_BY_MARGIN_S + 0.4)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertFalse(out.ok)
        self.assertTrue(out.lookahead and out.by_need)
        self.assertFalse(out.held)
        self.assertAlmostEqual(out.deadline_s, 0.4, delta=0.05)
        self.assertTrue(out.counted)
        self.assertEqual(c.failures(), 1)

    def test_a_lookahead_given_up_before_it_was_needed_does_not_count(self):
        c, srv = self.ready_client(self.server(tts_delay=3.0))
        with mock.patch.object(cvc, "LOOKAHEAD_MAX_S", 0.3):
            for i in range(cvc.MAX_FAILURES + 1):
                out = c.render(f"Far ahead {i}.", 0.1,
                               needed_by=time.monotonic() + 20.0)
                self.assertFalse(out.ok)
                self.assertTrue(out.by_need)
                self.assertFalse(out.counted)
        self.assertEqual(c.failures(), 0)
        self.assertEqual(c.status()[0], "ready")
        self.assertEqual(self.logs, [])

    def test_bad_hints_never_raise(self):
        c, srv = self.ready_client(self.server(tts_delay=0.5))
        out = c.render("A bad hint.", 0.1, needed_by="soon",
                       budget_chars="many")
        self.assertFalse(out.ok)
        self.assertFalse(out.lookahead)            # treated as a first line
        self.assertTrue(out.counted)
        self.assertAlmostEqual(out.deadline_s, 0.1)

    def test_hard_errors_count_for_lines_rendered_ahead_too(self):
        c, srv = self.ready_client(self.server(tts_status=500))
        for i in range(cvc.MAX_FAILURES):
            out = c.render(f"Ahead {i}.", 2.5,
                           needed_by=time.monotonic() + 10.0)
            self.assertTrue(out.counted)
        self.assertEqual(c.status()[0], "cooldown")

    def test_a_held_line_waits_past_the_time_it_is_needed(self):
        # The short-opener case: little audio queued, the line slower than
        # its budget. Not held it goes to Kokoro; held (the reply already
        # speaks in the clone) it is voiced late -- a pause, one voice.
        c, srv = self.ready_client(self.server(tts_delay=0.45))
        first = c.render("Not held.", 0.3, needed_by=time.monotonic() + 0.3)
        self.assertFalse(first.ok)
        self.assertFalse(first.held)
        self.assertAlmostEqual(first.deadline_s, 0.3)
        need = time.monotonic() + 0.3
        out = c.render("Held.", 0.3, needed_by=need, hold=True)
        self.assertTrue(out.ok, out.reason)
        self.assertTrue(out.held and out.by_need)
        self.assertGreater(out.deadline_s, 0.5)
        self.assertGreater(out.late_s, 0.05)        # about 0.15 s of pause
        self.assertEqual(c.failures(), 0)

    def test_a_held_line_that_still_misses_counts(self):
        c, srv = self.ready_client(self.server(tts_delay=3.0))
        t0 = time.monotonic()
        out = c.render("Held but hopeless.", 0.2, needed_by=t0 + 0.2,
                       hold=True)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertFalse(out.ok)
        self.assertTrue(out.held and out.counted)
        self.assertEqual(c.failures(), 1)

    def test_hold_means_nothing_for_a_first_line(self):
        c, srv = self.ready_client(self.server(tts_delay=0.6))
        out = c.render("First line.", 0.3, hold=True)
        self.assertFalse(out.ok)
        self.assertFalse(out.held)
        self.assertAlmostEqual(out.deadline_s, 0.3)

    def test_the_rest_of_a_split_line_keeps_the_whole_lines_budget(self):
        c, srv = self.ready_client(self.server(tts_delay=0.45))
        rest = "the rest of a sentence."               # < BASE_CHARS
        short = c.render(rest, 0.2)
        self.assertFalse(short.ok)
        with mock.patch.object(cvc, "PER_CHAR_S", 0.02):
            whole = c.render("and " + rest, 0.2,
                             budget_chars=cvc.BASE_CHARS + 30)
        self.assertTrue(whole.ok, whole.reason)
        self.assertAlmostEqual(whole.deadline_s, 0.2 + 0.02 * 30)


class BackgroundRenderTests(_Base):
    """count=False (the filler clips, rendered after a turn while nobody
    waits): an outcome that never touches the miss count, the probation or
    the cool-down."""

    def test_background_misses_never_rest_the_clone(self):
        c, srv = self.ready_client(self.server(tts_delay=0.5))
        for i in range(cvc.MAX_FAILURES + 1):
            out = c.render(f"Filler {i}.", 0.1, count=False)
            self.assertIn("timed out", out.reason)
            self.assertFalse(out.counted)
        srv.tts_delay = 0.0
        srv.tts_status = 500
        self.assertFalse(c.render("Filler error.", 2.5, count=False).counted)
        srv.stop()
        self.assertFalse(c.render("Filler gone.", 2.5, count=False).counted)
        self.assertEqual(c.failures(), 0)
        self.assertEqual(c.status(), ("ready", ""))
        self.assertEqual(self.logs, [])

    def test_a_background_success_does_not_reset_the_count(self):
        srv = self.server()
        c, _ = self.ready_client(srv)
        srv.fail_texts = {"Answer 0.", "Answer 1."}
        for i in range(2):
            self.assertTrue(c.render(f"Answer {i}.", 2.5).counted)
        self.assertTrue(c.render("Very good, sir.", 2.5, count=False).ok)
        self.assertEqual(c.failures(), 2)     # says nothing about the answers


class RecheckAfterCooldownTests(_Base):
    """The consent gate across a cool-down: the server is checked again
    before the first render after one, because whatever answers at the
    address may have changed while the clone rested."""

    def setUp(self):
        super().setUp()
        self.now = [5000.0]
        self.srv = self.server(tts_status=500)
        self.c, _ = self.ready_client(self.srv, clock=lambda: self.now[0])
        for i in range(cvc.MAX_FAILURES):
            self.c.render(f"Miss {i}.", 2.5)
        self.assertEqual(self.c.status()[0], "cooldown")
        self.srv.tts_status = 200

    def end_rest(self):
        self.now[0] += self.c.cooldown_left_s()
        self.assertEqual(self.c.status()[0], "ready")
        self.logs.clear()

    def test_a_server_with_another_voice_is_refused_after_the_rest(self):
        self.srv.ref_sha = "f" * 64       # someone else's server, same port
        self.end_rest()
        out = self.c.render("Hello again.", 2.5)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "not-ready")
        self.assertEqual(self.srv.tts_texts()[-1:], ["Miss 2."])  # nothing new
        self.assertEqual(self.c.status()[0], "down")
        self.assertFalse(self.c.usable_for("butler"))
        self.assertEqual(len([m for m in self.logs
                              if "different voice prompt" in m]), 1, self.logs)

    def test_a_server_that_hides_its_voice_is_refused_after_the_rest(self):
        self.srv.ref_sha = ""
        self.end_rest()
        self.assertEqual(self.c.render("Hello again.", 2.5).reason,
                         "not-ready")
        self.assertEqual(self.c.status()[0], "down")

    def test_the_same_voice_is_checked_once_then_used(self):
        self.end_rest()
        n = self.srv.count("GET", "/health")
        self.assertTrue(self.c.render("Back again.", 2.5).ok)
        self.assertTrue(self.c.render("And again.", 2.5).ok)
        self.assertEqual(self.srv.count("GET", "/health"), n + 1)
        self.assertEqual(self.c.status(), ("ready", ""))

    def test_a_cache_hit_needs_no_check(self):
        # The cache holds renders of the voice start() checked; serving one
        # sends nothing, so there is nothing to check yet -- for a LOOK-AHEAD
        # line (its reply already speaks in the clone). A cached take never
        # OPENS a reply before the re-check (2026-10-05): the server could
        # speak another voice now, and the rest of the reply would be
        # Kokoro. That line renders, after the check.
        self.srv.tts_status = 200
        c, srv = self.ready_client(self.server())
        self.assertTrue(c.render("Very good, sir.", 2.5).ok)
        c._recheck = True
        n = srv.count("GET", "/health")
        self.assertTrue(c.render("Very good, sir.", 2.5,
                                 needed_by=time.monotonic() + 5.0).cached)
        self.assertEqual(srv.count("GET", "/health"), n)
        out = c.render("Very good, sir.", 2.5)
        self.assertTrue(out.ok and not out.cached)
        self.assertEqual(srv.count("GET", "/health"), n + 1)   # the re-check
        self.assertFalse(c._recheck)

    def test_a_server_still_loading_is_a_counted_miss_and_checked_again(self):
        self.srv.health_code = 503
        self.end_rest()
        out = self.c.render("Hello again.", 2.5)
        self.assertIn("503", out.reason)
        self.assertTrue(out.counted)
        self.assertEqual(self.c.status()[0], "cooldown")   # probation
        self.srv.health_code = 200
        self.end_rest()
        self.assertTrue(self.c.render("Hello at last.", 2.5).ok)


class CancelTests(_Base):
    """A reply stopped (a barge-in) while a line renders ahead: nothing will
    play that line, so the wait ends at once and nothing is counted."""

    def test_a_stopped_reply_ends_the_wait_at_once(self):
        srv = self.server(latency_for=lambda t: 5.0 if t == "Slow." else 0.0)
        srv.fail_texts = {"Miss 0.", "Miss 1."}
        c, _ = self.ready_client(srv)
        for i in range(2):
            self.assertTrue(c.render(f"Miss {i}.", 2.5).counted)
        stop = threading.Event()
        timer = threading.Timer(0.2, stop.set)
        timer.start()
        self.addCleanup(timer.cancel)
        t0 = time.monotonic()
        out = c.render("Slow.", 2.5, needed_by=t0 + 10.0, cancel=stop)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(out.reason, "cancelled")
        self.assertTrue(out.cancelled)
        self.assertFalse(out.ok or out.counted)
        # Neither a miss nor a success: the streak is exactly where it was.
        self.assertEqual(c.failures(), 2)
        self.assertEqual(c.status()[0], "ready")

    def test_an_already_stopped_reply_sends_nothing(self):
        c, srv = self.ready_client()
        stop = threading.Event()
        stop.set()
        out = c.render("Never sent.", 2.5, needed_by=time.monotonic() + 5,
                       cancel=stop)
        self.assertTrue(out.cancelled)
        time.sleep(0.3)               # a request, had one gone out, lands
        self.assertEqual(srv.tts_texts(), [])
        self.assertEqual(srv.count("POST", "/tts"), 0)

    def test_a_render_that_is_not_stopped_is_unchanged(self):
        c, srv = self.ready_client(self.server(tts_delay=0.2))
        out = c.render("Fine.", 2.5, needed_by=time.monotonic() + 1.0,
                       cancel=threading.Event())
        self.assertTrue(out.ok, out.reason)
        slow = c.render("Too slow.", 0.1, cancel=threading.Event())
        self.assertIn("timed out", slow.reason)
        self.assertTrue(slow.counted)


# ════════════════════════════════════════════════════════════════════════════
#  The live pattern of 2026-10-04 10:36, replayed (client + player)
# ════════════════════════════════════════════════════════════════════════════
class LiveReplayTests(_Base):
    """The replies that changed voice mid-reply and latched the clone off on
    2026-10-04 (tests/_clone_voice_fake: the 10:36 briefing, then the two
    one-line replies of 10:36:58 and 10:37:09): their render times and audio
    lengths as the server logged them, scaled by SCALE, through the REAL
    client against a serial fake server and the REAL
    core.sentence_tts.play_pipelined (and plan_clone_chunks for the later
    two). Kokoro is a marker; play blocks for the audio's length, as the
    speaker does. The synth passes what the monolith's _speak_sentences
    passes: the needed-by time, the chunk's budget, the reply's stop Event,
    and hold once the line before was the clone's (the real wiring is
    replayed in tests/monolith/test_monolith_clone_server).

    What is replayed is the lines rendered AHEAD. The reply's first line
    gets a generous budget (FIRST_TIMEOUT_S): a first line's own budget is
    pinned elsewhere, and on a starved runner (measured: up to ~0.3 s of
    extra round trip at 2x CPU oversubscription) its scaled 0.3 s margin
    would make this test about scheduling instead. Every later line has the
    shipped 2.5 s."""

    SCALE = 0.2
    FIRST_TIMEOUT_S = 10.0
    KOKORO = 0.25                       # Kokoro's marker: a constant level

    def replay(self, srv, lines, timeout_s=2.5, scale=None, client=None):
        s = self.SCALE if scale is None else scale
        c = client if client is not None else self.ready_client(srv)[0]
        voices = []
        prev_clone = [None]

        def synth(text):
            need = st.needed_by()
            budget = timeout_s if need is not None else self.FIRST_TIMEOUT_S
            out = c.render(text, budget * s, needed_by=need,
                           budget_chars=getattr(text, "budget_chars", None),
                           hold=prev_clone[0] is True,
                           cancel=st.reply_stopped())
            prev_clone[0] = out.ok
            if out.ok:
                return out.audio, out.sr
            return np.full(int(24000 * 0.1), self.KOKORO, np.float32), 24000

        def play(audio, sr):
            voices.append("kokoro" if np.allclose(audio, self.KOKORO)
                          else "clone")
            time.sleep(len(audio) / float(sr))

        with mock.patch.object(cvc, "PER_CHAR_S", cvc.PER_CHAR_S * s), \
             mock.patch.object(cvc, "NEEDED_BY_MARGIN_S",
                               cvc.NEEDED_BY_MARGIN_S * s):
            res = st.play_pipelined(list(lines), synth, play, lambda: False)
        return c, res, voices

    def test_a_short_opener_then_a_long_line_keeps_one_voice(self):
        # 10:36 lines 4 and 7 as one reply: a 23-char opener (1.6 s of
        # audio), then the 114-char line that was slower than its budget at
        # the shipped 2.5 s (3.76 s against 3.52 s). Only the opener's audio
        # is queued ahead of it, so waiting until it is needed is not enough
        # (995fad8: Kokoro mid-reply); the reply already speaks in the
        # clone, so the line is held -- about 2 s of pause, then the clone.
        # Scaled by 0.5 so the hold's margin (1.4 s live) survives a starved
        # CI runner; the long line's own clip is cut short (nothing follows
        # it, so its length changes nothing).
        s = 0.5
        lines = (LIVE_1036_LINES[3], LIVE_1036_LINES[6])
        lat = {lines[0]: LIVE_1036_RENDER_S[3] * s,
               lines[1]: LIVE_1036_RENDER_S[6] * s}
        wavs = {lines[0]: make_wav(lead_s=0.0, speech_s=LIVE_1036_AUDIO_S[3] * s,
                                   tail_s=0.0, amp=0.3),
                lines[1]: make_wav(lead_s=0.0, speech_s=0.2, tail_s=0.0,
                                   amp=0.3)}
        srv = self.server(latency_for=lat, wav_for=wavs, serial=True)
        c, res, voices = self.replay(srv, lines, scale=s)
        self.assertEqual(res.sentences_played, 2)
        self.assertEqual(voices, ["clone", "clone"])
        self.assertEqual(c.failures(), 0)
        # The premise: the long line really is slower than its budget, and
        # than the time it was needed.
        self.assertGreater(LIVE_1036_RENDER_S[6],
                           cvc.render_budget_s(len(lines[1]), 2.5))
        self.assertGreater(LIVE_1036_RENDER_S[6], LIVE_1036_AUDIO_S[3])

    def test_the_1036_briefing_keeps_the_clone_and_never_cools_down(self):
        s = self.SCALE
        srv = live_1036_server(self.prof.sha, s).start()
        self.addCleanup(srv.stop)
        c, res, voices = self.replay(srv, LIVE_1036_LINES)
        self.assertEqual(res.sentences_played, 7)
        self.assertEqual(voices, ["clone"] * 7)        # no Kokoro switch
        self.assertEqual(c.failures(), 0)
        self.assertEqual(c.status(), ("ready", ""))   # no latch, no rest
        self.assertEqual(self.logs, [])
        self.assertEqual(srv.tts_texts(), list(LIVE_1036_LINES))
        # The pattern really is the live one: only line 7 rendered slower
        # than the fixed per-line budget -- the one Kokoro voiced live.
        self.assertEqual([len(t) for t in LIVE_1036_LINES],
                         [18, 33, 58, 23, 113, 93, 114])
        budgets = [cvc.render_budget_s(len(t), 2.5) for t in LIVE_1036_LINES]
        over = [i for i, (r, b) in enumerate(zip(LIVE_1036_RENDER_S, budgets))
                if r > b]
        self.assertEqual(over, [6])
        self.assertTrue(all(a > r for a, r in zip(LIVE_1036_AUDIO_S,
                                                  LIVE_1036_RENDER_S)))

    def test_the_two_replies_that_latched_it_keep_one_voice(self):
        # 10:36:58 (one 95-char sentence) and 10:37:09 (two short sentences,
        # 100 chars): each missed its first-line budget live -- misses 2 and
        # 3, the latch. Planned for the clone now (a clause head and the
        # rest; sentence by sentence), the piece rendered ahead keeps the
        # whole line's budget and is held for the clone, so each reply stays
        # in one voice with no miss. (The pieces' render times are
        # estimates, see tests/_clone_voice_fake.)
        s = self.SCALE
        srv = live_1036_server(self.prof.sha, s, later=True).start()
        self.addCleanup(srv.stop)
        c, _ = self.ready_client(srv)
        for whole, pieces in ((LIVE_1037_AWAY, LIVE_1037_AWAY_PIECES),
                              (LIVE_1037_MORNING, LIVE_1037_MORNING_PIECES)):
            chunks = st.plan_clone_chunks(whole)
            self.assertEqual(chunks, list(pieces))
            self.assertEqual(chunks[1].budget_chars, len(whole))
            _c, res, voices = self.replay(srv, chunks, client=c)
            self.assertEqual(voices, ["clone", "clone"], whole)
        self.assertEqual(c.failures(), 0)
        self.assertEqual(c.status(), ("ready", ""))
        self.assertEqual(self.logs, [])
        # The premise: unsplit, each was slower than its first-line budget.
        self.assertEqual((len(LIVE_1037_AWAY), len(LIVE_1037_MORNING)),
                         (95, 100))
        self.assertGreater(LIVE_1037_AWAY_RENDER_S,
                           cvc.render_budget_s(len(LIVE_1037_AWAY), 2.5))
        self.assertGreater(LIVE_1037_MORNING_RENDER_S,
                           cvc.render_budget_s(len(LIVE_1037_MORNING), 2.5))
        # ... while the second piece alone (no whole-line budget, not held)
        # would miss too: the budget and the hold are both load-bearing.
        rest_r, _a = live_1037_timings()[LIVE_1037_MORNING_PIECES[1]]
        self.assertGreater(rest_r, cvc.render_budget_s(
            len(LIVE_1037_MORNING_PIECES[1]), 2.5))


# ════════════════════════════════════════════════════════════════════════════
#  The consent gate after start
# ════════════════════════════════════════════════════════════════════════════
class ConsentGateTests(_Base):
    def test_a_profile_switch_stops_using_the_server(self):
        c, srv = self.ready_client()
        other = ProfileDir("owner", source="owner", wav=b"another voice")
        self.addCleanup(other.cleanup)
        import shutil
        shutil.copytree(os.path.join(other.root, "owner"),
                        os.path.join(self.prof.root, "owner"))
        self.assertTrue(c.usable_for("butler"))
        self.assertFalse(c.usable_for("owner"))
        self.assertFalse(c.usable_for("owner"))
        self.assertFalse(c.usable_for(""))
        mism = [m for m in self.logs if "not the 'owner' profile's" in m]
        self.assertEqual(len(mism), 1, self.logs)

    def test_revoked_consent_is_seen_after_the_ttl(self):
        clock = [1000.0]
        c = self.client(clock=lambda: clock[0])
        srv = self.server()
        self.assertEqual(c.start(url=srv.url, cmd="", profile="butler"),
                         "ready")
        self.assertTrue(c.usable_for("butler"))
        import json
        meta = os.path.join(self.prof.root, "butler", "meta.json")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"name": "butler", "source": "character"}, f)
        self.assertTrue(c.usable_for("butler"))      # memo still fresh
        clock[0] += cvc.PROFILE_TTL_S + 0.1
        self.assertFalse(c.usable_for("butler"))


# ════════════════════════════════════════════════════════════════════════════
#  A replaced reference.wav (2026-10-09)
# ════════════════════════════════════════════════════════════════════════════
class ReferenceSwapTests(_Base):
    """The owner swapped data/voice_profiles/<p>/reference.wav to a new take
    and the server was restarted with it, yet JARVIS spoke Kokoro until a
    full restart: the client still held the OLD hash and refused the server
    whose /health said the new one. The consent check notices the changed
    file (a stat, then a hash) and reads /health again, so the swap takes
    effect without a restart -- and a server in any other voice stays
    refused."""
    TEXT = "Very good, sir."

    def setUp(self):
        super().setUp()
        self.now = [3000.0]

    def ready(self, srv=None):
        return self.ready_client(srv, clock=lambda: self.now[0])

    def swap_reference(self, wav: bytes = b"RIFF the owner's new take") -> str:
        with open(self.prof.ref, "wb") as f:
            f.write(wav)
        return hashlib.sha256(wav).hexdigest()

    def later(self, s: float) -> None:
        self.now[0] += s

    def test_a_replaced_reference_is_used_without_a_restart(self):
        c, srv = self.ready()
        self.assertTrue(c.render(self.TEXT, 2.5).ok)      # cached, old voice
        self.assertEqual(c.cache_len(), 1)
        new = self.swap_reference()
        srv.ref_sha, srv.pid = new, 5151                   # restarted with it
        self.later(cvc.PROFILE_TTL_S + 0.1)
        self.assertTrue(c.usable_for("butler"))            # the next check
        self.assertEqual(c.status(), ("ready", ""))
        self.assertEqual(c.voice_prefix(), new[:16])
        self.assertEqual(c.server_pid(), 5151)
        self.assertEqual(len([m for m in self.logs
                              if "now speaks the 'butler' profile's" in m]),
                         1, self.logs)
        # The old voice's take is gone and never served: the line renders
        # in the new voice.
        self.assertEqual(c.store.prefixes(), set())
        n = len(srv.tts_texts())
        out = c.render(self.TEXT, 2.5)
        self.assertTrue(out.ok and not out.cached)
        self.assertEqual(len(srv.tts_texts()), n + 1)
        self.assertEqual(c.store.prefixes(), {new[:16]})
        self.assertTrue(c.usable_for("butler"))

    def test_a_server_still_in_the_old_voice_stays_refused(self):
        c, srv = self.ready()
        gets = srv.count("GET", "/health")
        new = self.swap_reference()                        # not restarted yet
        self.later(cvc.PROFILE_TTL_S + 0.1)
        self.assertFalse(c.usable_for("butler"))
        self.assertEqual(srv.count("GET", "/health"), gets + 1)   # looked
        for _ in range(5):                                 # rate-limited
            self.assertFalse(c.usable_for("butler"))
        self.assertEqual(srv.count("GET", "/health"), gets + 1)
        self.later(cvc.VOICE_PROBE_S + 0.1)
        self.assertFalse(c.usable_for("butler"))
        self.assertEqual(srv.count("GET", "/health"), gets + 2)
        self.assertEqual(c.status()[0], "ready")           # just refused
        self.assertEqual(c.voice_prefix(), self.prof.sha[:16])
        self.assertEqual(srv.tts_texts(), [])
        # Now it restarts with the new take: used at the next check after.
        srv.ref_sha, srv.pid = new, 5151
        self.later(cvc.VOICE_PROBE_S + 0.1)
        self.assertTrue(c.usable_for("butler"))
        self.assertEqual(c.voice_prefix(), new[:16])

    def test_a_server_restarted_with_the_new_take_first_is_used_once_the_file_matches(self):
        c, srv = self.ready()
        new = hashlib.sha256(b"RIFF the owner's new take").hexdigest()
        srv.ref_sha, srv.pid = new, 5151        # the file is still the old one
        self.assertFalse(c.refresh_health())
        self.assertEqual(c.status()[0], "down")
        self.assertFalse(c.usable_for("butler"))
        self.assertEqual(c.render(self.TEXT, 2.5).reason, "not-ready")
        self.assertEqual(srv.tts_texts(), [])
        self.swap_reference()                    # ...and now the file
        self.later(cvc.PROFILE_TTL_S + 0.1)
        self.assertTrue(c.usable_for("butler"))
        self.assertEqual(c.status(), ("ready", ""))
        self.assertTrue(c.render(self.TEXT, 2.5).ok)

    def test_a_server_in_any_other_voice_is_never_used(self):
        c, srv = self.ready()
        srv.ref_sha, srv.pid = "c" * 64, 5151      # nobody's reference
        self.assertFalse(c.refresh_health())
        self.assertEqual(c.status()[0], "down")
        for step in range(4):
            self.later(cvc.VOICE_PROBE_IDLE_S + 0.1)
            if step == 2:
                self.swap_reference()              # another new take
            self.assertFalse(c.usable_for("butler"), step)
        self.assertEqual(c.status()[0], "down")
        self.assertEqual(srv.tts_texts(), [])
        self.assertEqual(len([m for m in self.logs
                              if "different voice prompt" in m]), 1, self.logs)

    def test_a_reference_replaced_during_a_cool_down_is_followed_after_it(self):
        c, srv = self.ready(self.server(tts_status=500))
        for i in range(cvc.MAX_FAILURES):
            c.render(f"Miss {i}.", 2.5)
        self.assertEqual(c.status()[0], "cooldown")
        new = self.swap_reference()
        srv.ref_sha, srv.pid, srv.tts_status = new, 5151, 200
        self.later(c.cooldown_left_s() + 0.1)
        self.assertTrue(c.usable_for("butler"))
        out = c.render(self.TEXT, 2.5)
        self.assertTrue(out.ok, out)
        self.assertEqual(c.status(), ("ready", ""))
        self.assertEqual(c.voice_prefix(), new[:16])

    def test_the_voice_recheck_after_a_cool_down_follows_the_new_reference(self):
        # The first render after a rest re-checks the voice before it sends
        # anything (_recheck_voice); a reference replaced during the rest,
        # with the server restarted on it, is the consented voice: it passes.
        c, srv = self.ready(self.server(tts_status=500))
        for i in range(cvc.MAX_FAILURES):
            c.render(f"Miss {i}.", 2.5)
        new = self.swap_reference()
        srv.ref_sha, srv.pid, srv.tts_status = new, 5151, 200
        self.later(c.cooldown_left_s() + 0.1)
        self.assertEqual(c.status()[0], "ready")
        out = c.render(self.TEXT, 2.5)          # no consent check before it
        self.assertTrue(out.ok, out)
        self.assertEqual(c.status(), ("ready", ""))
        self.assertEqual(c.voice_prefix(), new[:16])
        self.assertEqual(srv.tts_texts()[-1], self.TEXT)
        # (One restarted in nobody's voice is refused at that check:
        # RecheckAfterCooldownTests.)

    def test_a_server_left_running_in_another_voice_at_boot_is_watched(self):
        new = hashlib.sha256(b"RIFF the owner's new take").hexdigest()
        srv = self.server(ref_sha=new)            # JARVIS restarted mid-swap
        c = self.client(clock=lambda: self.now[0])
        self.assertEqual(c.start(url=srv.url, cmd="", profile="butler"),
                         "down")
        self.assertFalse(c.usable_for("butler"))
        self.swap_reference()
        self.later(cvc.PROFILE_TTL_S + 0.1)
        self.assertTrue(c.usable_for("butler"))
        self.assertEqual(c.voice_prefix(), new[:16])

    def test_a_server_down_for_another_reason_is_not_probed(self):
        c = self.client(clock=lambda: self.now[0])
        self.assertEqual(c.start(url=f"http://127.0.0.1:{free_port()}",
                                 cmd="", profile="butler"), "down")
        with mock.patch.object(c, "_health",
                               side_effect=AssertionError("probed")):
            self.later(cvc.VOICE_PROBE_IDLE_S + 1.0)
            self.assertFalse(c.usable_for("butler"))

    def test_a_same_size_swap_that_kept_the_old_mtime_is_noticed(self):
        # A cheap stat decides whether to hash again: a new take of the same
        # length, copied with its times kept, still has a new file id.
        c, _srv = self.ready()
        old_sha = c._file_sha(self.prof.ref)
        self.assertEqual(old_sha, self.prof.sha)
        st0 = os.stat(self.prof.ref)
        tmp = self.prof.ref + ".new"
        with open(self.prof.ref, "rb") as f:
            wav = bytes(b ^ 0x55 for b in f.read())
        with open(tmp, "wb") as f:
            f.write(wav)
        os.utime(tmp, ns=(st0.st_atime_ns, st0.st_mtime_ns))
        os.replace(tmp, self.prof.ref)
        st1 = os.stat(self.prof.ref)
        self.assertEqual((st1.st_size, st1.st_mtime_ns),
                         (st0.st_size, st0.st_mtime_ns))
        self.assertEqual(c._file_sha(self.prof.ref),
                         hashlib.sha256(wav).hexdigest())
        self.later(cvc.PROFILE_TTL_S + 0.1)
        self.assertEqual(c.profile_sha("butler"),
                         hashlib.sha256(wav).hexdigest())

    def test_the_status_line_says_the_server_speaks_another_reference(self):
        # "What voice are you using?" right after the swap, before the
        # server was restarted with the new take: an honest reason.
        c, srv = self.ready()
        new = self.swap_reference()
        self.later(cvc.PROFILE_TTL_S + 0.1)
        with mock.patch.object(vc, "_cfg_model", return_value=cvc.MODEL_ID), \
             mock.patch.object(vc, "_cfg_enabled", return_value=True), \
             mock.patch.object(vc, "_cfg_profile", return_value="butler"), \
             mock.patch.object(cvc, "CLIENT", c):
            self.assertFalse(vc.is_available())
            self.assertIn("different reference", vc.engine_hint())
            srv.ref_sha, srv.pid = new, 5151
            self.later(cvc.VOICE_PROBE_S + 0.1)
            self.assertTrue(vc.is_available())
            self.assertNotIn("different reference", vc.engine_hint())
            srv.ref_sha = "c" * 64                  # someone else's voice
            self.assertFalse(c.refresh_health())
            self.assertIn("different reference", vc.engine_hint())

    def test_a_stale_memo_never_drops_a_server_in_the_new_voice(self):
        # Something reads /health (the writer, the keeper) within the
        # consent memo's lifetime after the swap: the decision about the
        # server is taken on the file as it is now, not on the memo.
        c, srv = self.ready()
        self.assertTrue(c.usable_for("butler"))     # memo: the old hash
        new = self.swap_reference()
        srv.ref_sha, srv.pid = new, 5151
        self.assertTrue(c.refresh_health())          # no TTL has passed
        self.assertEqual(c.status(), ("ready", ""))
        self.assertEqual(c.voice_prefix(), new[:16])


# ════════════════════════════════════════════════════════════════════════════
#  core.voice_clone with the server engine selected
# ════════════════════════════════════════════════════════════════════════════
class VoiceCloneServerModelTests(_Base):
    def test_is_available_asks_the_client(self):
        c, srv = self.ready_client()
        with mock.patch.object(vc, "_cfg_enabled", return_value=True), \
             mock.patch.object(vc, "_cfg_model",
                               return_value=cvc.MODEL_ID), \
             mock.patch.object(vc, "_cfg_profile", return_value="butler"), \
             mock.patch.object(cvc, "CLIENT", c), \
             mock.patch.object(vc, "_chatterbox_importable",
                               side_effect=AssertionError("in-process probe")):
            self.assertTrue(vc.is_available())
            for _ in range(cvc.MAX_FAILURES):
                c._failed("x", c._clock())
            self.assertFalse(vc.is_available())

    def test_synthesize_never_loads_the_in_process_model(self):
        with mock.patch.object(vc, "_cfg_model", return_value=cvc.MODEL_ID), \
             mock.patch.object(vc, "_load_engine",
                               side_effect=AssertionError("loaded")):
            self.assertIsNone(vc.synthesize(
                "hello", {"consent": True, "source": "owner",
                          "reference_wav": __file__}))

    def test_engine_hint_says_when_the_clone_is_resting(self):
        now = [100.0]
        c, srv = self.ready_client(self.server(tts_status=500),
                                   clock=lambda: now[0])
        with mock.patch.object(vc, "_cfg_model", return_value=cvc.MODEL_ID), \
             mock.patch.object(cvc, "CLIENT", c):
            for i in range(cvc.MAX_FAILURES):
                c.render(f"Miss {i}.", 2.5)
            self.assertIn("resting for about 5 more minutes",
                          vc.engine_hint())
            self.assertIn("kept failing", vc.engine_hint())     # HTTP 500s
            now[0] += 250
            self.assertIn("about 1 more minute,", vc.engine_hint() + ",")
            now[0] += 50
            self.assertIn("isn't running or isn't ready", vc.engine_hint())

    def test_engine_hint_says_why_the_clone_is_resting(self):
        # Spoken by voice_clone_status: slow lines and a server that stopped
        # answering are different stories.
        for reason, words in (("timed out", "was too slow"),
                              ("error (ConnectionError: cannot connect to "
                               "127.0.0.1:1 (ConnectionRefusedError))",
                               "stopped answering"),
                              ("http 500", "kept failing")):
            c = self.client()
            with c._mu:
                c._status, c._reason = "cooldown", reason
                c._cool_until = c._clock() + 120.0
            with mock.patch.object(vc, "_cfg_model",
                                   return_value=cvc.MODEL_ID), \
                 mock.patch.object(cvc, "CLIENT", c):
                hint = vc.engine_hint()
            self.assertIn(words, hint, reason)
            self.assertIn("resting for about 2 more minutes", hint)

    def test_engine_hint_and_rearm(self):
        with mock.patch.object(vc, "_cfg_model", return_value=cvc.MODEL_ID):
            self.assertIn("clone voice server", vc.engine_hint())
        with mock.patch.object(vc, "_cfg_model", return_value="chatterbox"):
            self.assertIn("chatterbox-tts", vc.engine_hint())
            self.assertFalse(vc.rearm())


class FastConnectTests(_Base):
    """Every request connects through _fast_connect (a non-blocking connect
    and select): CPython's own timed connect costs ~20 ms on about a
    quarter of loopback connects on Windows (measured 2026-10-05), which a
    cached opener's /health check -- and every render -- would pay."""

    def test_it_connects_and_the_client_uses_it(self):
        srv = self.server()
        sock = cvc._fast_connect("127.0.0.1", srv.port, 1.0)
        try:
            self.assertIsNotNone(sock.getpeername())
            self.assertTrue(sock.getblocking())
        finally:
            sock.close()
        calls = []
        real = cvc._fast_connect

        def spy(*a):
            calls.append(a)
            return real(*a)
        c, _ = self.ready_client(srv)
        with mock.patch.object(cvc, "_fast_connect", side_effect=spy):
            self.assertTrue(c.render("Hello there.", 2.5).ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ("127.0.0.1", srv.port))

    def test_nothing_listening_is_a_connection_error_within_its_time(self):
        port = free_port()
        t0 = time.monotonic()
        with self.assertRaises(ConnectionError):
            cvc._fast_connect("127.0.0.1", port, 0.3)
        self.assertLess(time.monotonic() - t0, 2.0)
        c = self.client()
        c._host, c._port = "127.0.0.1", port
        with self.assertRaises(ConnectionError):
            c._request("GET", "/health", None, 0.3)
        self.assertEqual(c._health(0.3), (None, {}))


if __name__ == "__main__":
    unittest.main()
