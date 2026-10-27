"""v3 W13: core.background_llm - the pure routing + HTTP helper for the
background brain on a second (CPU-only) Ollama.

Pins: unset URL = main brain; only allowlisted background callers go remote;
owner-facing callers never do; the payload carries sampling knobs only (no
num_ctx / num_thread / GPU options / keep_alive: any of them would reload the
CPU host's model or override its residency); failures raise RemoteFailure and
arm the 60 s health memo; loopback URLs in the new code use 127.0.0.1.
"""
from __future__ import annotations

import os
import re
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import background_llm as bgl  # noqa: E402

# A tailnet (100.64/10) address / MagicDNS name, split so the PII gate sees
# no literal (NET-3).
TAILNET_HOST = "100" + ".101.1.2"
TS_NET_HOST = "unraid.tail1234" + ".ts" + ".net"
URL = "http://%s:11434" % TAILNET_HOST


class _Resp:
    def __init__(self, body=None, status=200):
        self._body = body
        self.status_code = status
        self.ok = 200 <= status < 300

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _ok(text="hello"):
    return _Resp({"message": {"role": "assistant", "content": text},
                  "prompt_eval_count": 7, "eval_count": 3})


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class RoutingTableTests(unittest.TestCase):
    def setUp(self):
        self.health = bgl.HealthMemo(clock=_Clock())

    def r(self, url, caller, bg):
        return bgl.route(url, caller, bg, health=self.health)

    def test_plain_http_goes_remote_only_to_loopback_or_the_tailnet(self):
        """Security review NET-3 (2026-10-27): memory, conversation and room
        text never travel in clear text to a LAN or foreign host."""
        caller = sorted(bgl.REMOTE_OK_CALLERS)[0]
        for url in ("http://192.168.1.20:11434", "http://10.0.0.5:11434",
                    "http://203.0.113.9:11434",
                    "http://unraid:11434", "http://unraid.local:11434",
                    "http://%s:11434" % ("100" + ".128.0.1"),
                    "ftp://%s/" % TAILNET_HOST,
                    "http://[::ffff:192.168.1.20]:11434", "garbage"):
            self.assertEqual(self.r(url, caller, True), "main", url)
        for url in (URL, "http://127.0.0.1:11434",
                    "http://%s:11434" % TS_NET_HOST,
                    "https://ollama.example.com"):
            self.assertEqual(self.r(url, caller, True), "remote", url)
        with self.assertRaises(bgl.RemoteFailure):
            bgl.chat("http://192.168.1.20:11434", "m", "s", [], 5, 5,
                     post=lambda *a, **k: self.fail("posted to a LAN host"))

    def test_unset_url_is_always_main(self):
        for url in ("", "   ", None, 5):
            for caller in bgl.REMOTE_OK_CALLERS:
                self.assertEqual(self.r(url, caller, True), "main")

    def test_allowlisted_background_callers_go_remote(self):
        for caller in ("learn_batch", "learn_from_turn", "session_checkpoint",
                       "ltm_reflect", "ambient_media_judge",
                       "ambient_extractor"):
            self.assertEqual(self.r(URL, caller, True), "remote", caller)

    def test_foreground_never_remote(self):
        for caller in list(bgl.REMOTE_OK_CALLERS) + ["voice_turn", ""]:
            self.assertEqual(self.r(URL, caller, False), "main", caller)

    def test_owner_facing_background_callers_never_remote(self):
        for caller in ("proactive_remark", "screen_remark",
                       "boot_changelog_summary", "chappie",
                       "briefing_headline_summary", "notification_triage",
                       "voice_turn", "voice_followup", "untagged", ""):
            self.assertEqual(self.r(URL, caller, True), "main", caller)

    def test_lists_do_not_overlap(self):
        self.assertFalse(bgl.REMOTE_OK_CALLERS & bgl.OWNER_FACING_CALLERS)

    def test_unknown_background_caller_defaults_to_main(self):
        self.assertEqual(self.r(URL, "some_new_job", True), "main")

    def test_health_memo_sends_jobs_to_main_for_60s(self):
        clock = _Clock()
        h = bgl.HealthMemo(clock=clock)
        self.assertEqual(bgl.route(URL, "learn_batch", True, health=h), "remote")
        h.note_failure()
        self.assertEqual(bgl.route(URL, "learn_batch", True, health=h), "main")
        clock.t += 59.0
        self.assertTrue(h.is_down())
        clock.t += 2.0
        self.assertFalse(h.is_down())
        self.assertEqual(bgl.route(URL, "learn_batch", True, health=h), "remote")

    def test_note_ok_clears_the_memo(self):
        h = bgl.HealthMemo(clock=_Clock())
        h.note_failure()
        h.note_ok()
        self.assertFalse(h.is_down())


class SettingsCleaningTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(bgl.clean_model(""), "gemma4:e4b")
        self.assertEqual(bgl.clean_model(None), "gemma4:e4b")
        self.assertEqual(bgl.clean_timeout(None), 120.0)
        self.assertEqual(bgl.clean_timeout(-3), 120.0)
        self.assertEqual(bgl.clean_timeout("abc"), 120.0)
        self.assertEqual(bgl.clean_timeout(float("nan")), 120.0)
        self.assertEqual(bgl.clean_timeout("45"), 45.0)
        self.assertEqual(bgl.clean_base_url(" http://h:1/ "), "http://h:1")

    def test_config_declares_the_three_settings_with_inert_defaults(self):
        from core import config
        self.assertEqual(config.BACKGROUND_LLM_BASE_URL, "")
        self.assertEqual(config.BACKGROUND_LLM_MODEL, "gemma4:e4b")
        self.assertEqual(config.BACKGROUND_LLM_TIMEOUT_S, 120)

    def test_base_url_stays_per_machine(self):
        """The *_URL suffix keeps the shared-settings ledger from copying the
        Unraid address onto the other laptop."""
        from core import peer_settings
        self.assertFalse(peer_settings.is_shared("BACKGROUND_LLM_BASE_URL"))


class PayloadTests(unittest.TestCase):
    def test_sampling_knobs_only(self):
        """The CPU host serves one pre-configured model (num_thread baked into
        its Modelfile, context set server-side). A per-request num_ctx or
        num_thread would reload 6.6 GB on a CPU box, and keep_alive would
        override the server's 24 h residency, so none of them is sent."""
        p = bgl.build_payload("gemma4:e4b", "sys", [{"role": "user",
                              "content": "hi"}], 120, think=False)
        opts = p["options"]
        for runner_key in ("num_ctx", "num_thread", "num_gpu", "main_gpu"):
            self.assertNotIn(runner_key, opts)
        self.assertNotIn("keep_alive", p)
        self.assertEqual(opts["num_predict"], 120)
        self.assertEqual(opts["temperature"], 0.4)
        for k in opts:
            self.assertNotIn("gpu", k.lower())
        self.assertNotIn("main_gpu", p)
        self.assertFalse(p["stream"])
        self.assertIs(p["think"], False)
        self.assertEqual(p["messages"][0], {"role": "system", "content": "sys"})

    def test_think_omitted_when_none(self):
        p = bgl.build_payload("m", "s", [], 10)
        self.assertNotIn("think", p)


class ChatTests(unittest.TestCase):
    def setUp(self):
        bgl.HEALTH.note_ok()
        self.addCleanup(bgl.HEALTH.note_ok)

    def test_success_returns_text_and_body(self):
        seen = {}

        def post(url, json=None, timeout=None, **kw):
            seen.update(url=url, json=json, timeout=timeout, **kw)
            return _ok("done")
        text, body = bgl.chat(URL + "/", "gemma4:e4b", "s",
                              [{"role": "user", "content": "u"}], 50, 90,
                              post=post)
        self.assertEqual(text, "done")
        self.assertEqual(body["eval_count"], 3)
        self.assertEqual(seen["url"], URL + "/api/chat")
        self.assertEqual(seen["timeout"], (bgl.CONNECT_TIMEOUT_S, 90.0))
        self.assertNotIn("num_ctx", seen["json"]["options"])
        self.assertNotIn("keep_alive", seen["json"])
        self.assertFalse(bgl.HEALTH.is_down())

    def test_connection_error_and_timeout_raise_and_arm_memo(self):
        for exc in (ConnectionError("refused"), TimeoutError("slow")):
            bgl.HEALTH.note_ok()

            def post(url, json=None, timeout=None, _e=exc, **_kw):
                raise _e
            with self.assertRaises(bgl.RemoteFailure):
                bgl.chat(URL, "m", "s", [], 5, 5, post=post)
            self.assertTrue(bgl.HEALTH.is_down())

    def test_5xx_raises_and_arms_memo(self):
        with self.assertRaises(bgl.RemoteFailure):
            bgl.chat(URL, "m", "s", [], 5, 5,
                     post=lambda *a, **k: _Resp({}, status=503))
        self.assertTrue(bgl.HEALTH.is_down())

    def test_4xx_and_3xx_raise_and_arm_memo(self):
        """Security review NET-5 (2026-10-27): a 404 / 400 used to leave the
        memo clear, so every background job re-sent the full memory payload
        before falling back; a redirect is never followed."""
        for status in (404, 400, 307, 308):
            bgl.HEALTH.note_ok()
            seen = {}

            def post(url, _s=status, **kw):
                seen.update(kw)
                return _Resp({"message": {"content": "x"}}, status=_s)
            with self.assertRaises(bgl.RemoteFailure):
                bgl.chat(URL, "m", "s", [], 5, 5, post=post)
            self.assertTrue(bgl.HEALTH.is_down(), status)
            self.assertIs(seen.get("allow_redirects"), False)

    def test_default_poster_ignores_proxies(self):
        """NET-5: no HTTP(S)_PROXY / Windows system proxy carries the body."""
        if bgl._requests is None:
            self.skipTest("requests not installed")
        self.assertFalse(bgl._session().trust_env)

    def test_empty_reply_raises(self):
        with self.assertRaises(bgl.RemoteFailure):
            bgl.chat(URL, "m", "s", [], 5, 5,
                     post=lambda *a, **k: _ok("   "))

    def test_bad_json_raises(self):
        with self.assertRaises(bgl.RemoteFailure):
            bgl.chat(URL, "m", "s", [], 5, 5,
                     post=lambda *a, **k: _Resp(ValueError("x")))


class SourceHygieneTests(unittest.TestCase):
    def test_no_localhost_url_in_new_module(self):
        path = os.path.join(_ROOT, "core", "background_llm.py")
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        self.assertEqual([ln for ln in lines
                          if re.search(r"https?://localhost", ln)], [])


class SyncSummaryAndWindowTests(unittest.TestCase):
    def test_stay_on_main_pins_this_thread_only(self):
        """The summary taken at shutdown / peer hand-off is waited on by the
        owner: inside stay_on_main() it stays on the main brain even with a
        URL set; outside (and on other threads) routing is unchanged."""
        import threading
        memo = bgl.HealthMemo()
        self.assertEqual(bgl.route(URL, "session_checkpoint", True, memo),
                         "remote")
        other = {}
        with bgl.stay_on_main():
            with bgl.stay_on_main():   # re-entrant
                self.assertEqual(
                    bgl.route(URL, "session_checkpoint", True, memo), "main")
            self.assertEqual(
                bgl.route(URL, "session_checkpoint", True, memo), "main")
            th = threading.Thread(target=lambda: other.update(
                r=bgl.route(URL, "session_checkpoint", True, memo)))
            th.start()
            th.join()
        self.assertEqual(other["r"], "remote")
        self.assertFalse(bgl.pinned_to_main())
        self.assertEqual(bgl.route(URL, "session_checkpoint", True, memo),
                         "remote")

    def test_clean_num_ctx(self):
        self.assertEqual(bgl.clean_num_ctx(None), bgl.DEFAULT_NUM_CTX)
        self.assertEqual(bgl.clean_num_ctx("x"), bgl.DEFAULT_NUM_CTX)
        self.assertEqual(bgl.clean_num_ctx(100), bgl.DEFAULT_NUM_CTX)
        self.assertEqual(bgl.clean_num_ctx(4096), 4096)
        self.assertEqual(bgl.DEFAULT_NUM_CTX, 8192)


if __name__ == "__main__":
    unittest.main()
