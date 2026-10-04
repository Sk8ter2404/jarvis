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
  * the needed-by deadline: a line rendered ahead may wait until it is
    needed (never less than its budget), and a look-ahead line given up on
    before it was needed does not count;
  * the 10:36 replay: the live briefing pattern keeps the clone voice;
  * the consent gate: the server is used only while it speaks the active
    consented profile's reference (by hash);
  * the cache answers a repeated line without a request.
"""
from __future__ import annotations

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
                                     LIVE_1036_RENDER_S, FakeCloneServer,
                                     ProfileDir, free_port, live_1036_server,
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
        srv = self.server()
        c, _ = self.ready_client(srv)
        self.assertTrue(c.render("Very good, sir.", 2.5).ok)
        srv.tts_status = 500
        for i in range(cvc.MAX_FAILURES):
            self.assertEqual(c.status()[0], "ready")
            self.assertFalse(c.render(f"New line {i}.", 2.5).ok)
            if c.status()[0] == "ready":
                hit = c.render("Very good, sir.", 2.5)
                self.assertTrue(hit.ok and hit.cached)
        self.assertEqual(c.status()[0], "cooldown", c.failures())
        # A hit still reports success to its caller while the streak runs.
        c2, srv2 = self.ready_client(self.server())
        self.assertTrue(c2.render("Right away, sir.", 2.5).ok)
        srv2.tts_status = 500
        c2.render("Fails once.", 2.5)
        self.assertTrue(c2.render("Right away, sir.", 2.5).cached)
        self.assertEqual(c2.failures(), 1)

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
        self.miss()
        self.assertAlmostEqual(self.c.cooldown_left_s(), 300.0)


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

    def test_a_bad_needed_by_is_the_budget(self):
        self.assertEqual(cvc.line_deadline_s(2.5, "soon", 100.0), (2.5, False))


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


# ════════════════════════════════════════════════════════════════════════════
#  The live pattern of 2026-10-04 10:36, replayed (client + player)
# ════════════════════════════════════════════════════════════════════════════
class LiveReplayTests(_Base):
    """The 7-sentence briefing that changed voice mid-reply and latched the
    clone off: its measured render times (1.0-3.8 s) and audio lengths,
    scaled by SCALE, through the REAL client against a serial fake server and
    the REAL core.sentence_tts.play_pipelined. Kokoro is a marker; play
    blocks for the audio's length, as the speaker does."""

    SCALE = 0.15

    def test_the_1036_briefing_keeps_the_clone_and_never_cools_down(self):
        s = self.SCALE
        srv = live_1036_server(self.prof.sha, s).start()
        self.addCleanup(srv.stop)
        c, _ = self.ready_client(srv)
        voices = []
        kokoro = 0.25                   # Kokoro's marker: a constant level

        def synth(text):
            out = c.render(text, 2.5 * s, needed_by=st.needed_by())
            if out.ok:
                return out.audio, out.sr
            return np.full(int(24000 * 0.1), kokoro, np.float32), 24000

        def play(audio, sr):
            voices.append("kokoro" if np.allclose(audio, kokoro) else "clone")
            time.sleep(len(audio) / float(sr))

        with mock.patch.object(cvc, "PER_CHAR_S", cvc.PER_CHAR_S * s), \
             mock.patch.object(cvc, "NEEDED_BY_MARGIN_S",
                               cvc.NEEDED_BY_MARGIN_S * s):
            res = st.play_pipelined(list(LIVE_1036_LINES), synth, play,
                                    lambda: False)
        self.assertEqual(res.sentences_played, 7)
        self.assertEqual(voices, ["clone"] * 7)        # no Kokoro switch
        self.assertEqual(c.failures(), 0)
        self.assertEqual(c.status(), ("ready", ""))   # no latch, no rest
        self.assertEqual(self.logs, [])
        self.assertEqual(srv.tts_texts(), list(LIVE_1036_LINES))
        # The pattern really is the live one: lines 5-7 rendered slower than
        # the fixed per-line budget -- the ones that switched voice live.
        budgets = [cvc.render_budget_s(len(t), 2.5) for t in LIVE_1036_LINES]
        over = [i for i, (r, b) in enumerate(zip(LIVE_1036_RENDER_S, budgets))
                if r > b]
        self.assertEqual(over, [4, 5, 6])
        self.assertTrue(all(a > r for a, r in zip(LIVE_1036_AUDIO_S,
                                                  LIVE_1036_RENDER_S)))


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
            now[0] += 250
            self.assertIn("about 1 more minute,", vc.engine_hint() + ",")
            now[0] += 50
            self.assertIn("isn't running or isn't ready", vc.engine_hint())

    def test_engine_hint_and_rearm(self):
        with mock.patch.object(vc, "_cfg_model", return_value=cvc.MODEL_ID):
            self.assertIn("clone voice server", vc.engine_hint())
        with mock.patch.object(vc, "_cfg_model", return_value="chatterbox"):
            self.assertIn("chatterbox-tts", vc.engine_hint())
            self.assertFalse(vc.rearm())


if __name__ == "__main__":
    unittest.main()
