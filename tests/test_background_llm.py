"""v3 W13: core.background_llm - the pure routing + HTTP helper for the
background brain on a second (CPU-only) Ollama.

Pins: unset URL = main brain; only allowlisted background callers go remote;
owner-facing callers never do; the payload has no GPU options and a pinned
num_ctx; failures raise RemoteFailure and arm the 60 s health memo; loopback
URLs in the new code use 127.0.0.1.
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
from core import ollama_opts  # noqa: E402

URL = "http://203.0.113.9:11434"


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
    def test_no_gpu_options_and_num_ctx_pinned(self):
        p = bgl.build_payload("gemma4:e4b", "sys", [{"role": "user",
                              "content": "hi"}], 120, think=False)
        opts = p["options"]
        self.assertEqual(opts["num_ctx"], ollama_opts.local_num_ctx("gemma4:e4b"))
        self.assertEqual(opts["num_predict"], 120)
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

        def post(url, json=None, timeout=None):
            seen.update(url=url, json=json, timeout=timeout)
            return _ok("done")
        text, body = bgl.chat(URL + "/", "gemma4:e4b", "s",
                              [{"role": "user", "content": "u"}], 50, 90,
                              post=post)
        self.assertEqual(text, "done")
        self.assertEqual(body["eval_count"], 3)
        self.assertEqual(seen["url"], URL + "/api/chat")
        self.assertEqual(seen["timeout"], (bgl.CONNECT_TIMEOUT_S, 90.0))
        self.assertIn("num_ctx", seen["json"]["options"])
        self.assertFalse(bgl.HEALTH.is_down())

    def test_connection_error_and_timeout_raise_and_arm_memo(self):
        for exc in (ConnectionError("refused"), TimeoutError("slow")):
            bgl.HEALTH.note_ok()

            def post(url, json=None, timeout=None, _e=exc):
                raise _e
            with self.assertRaises(bgl.RemoteFailure):
                bgl.chat(URL, "m", "s", [], 5, 5, post=post)
            self.assertTrue(bgl.HEALTH.is_down())

    def test_5xx_raises_and_arms_memo(self):
        with self.assertRaises(bgl.RemoteFailure):
            bgl.chat(URL, "m", "s", [], 5, 5,
                     post=lambda *a, **k: _Resp({}, status=503))
        self.assertTrue(bgl.HEALTH.is_down())

    def test_4xx_raises_without_arming_memo(self):
        with self.assertRaises(bgl.RemoteFailure):
            bgl.chat(URL, "m", "s", [], 5, 5,
                     post=lambda *a, **k: _Resp({}, status=404))
        self.assertFalse(bgl.HEALTH.is_down())

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


if __name__ == "__main__":
    unittest.main()
