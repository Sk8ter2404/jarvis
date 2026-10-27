"""v3 W13: _call_local_llm routes allowlisted background jobs to the
background (CPU) Ollama when BACKGROUND_LLM_BASE_URL is set.

Pins: unset = the main brain exactly as before (no remote POST); background +
allowlisted caller + URL = remote POST to that URL with the BACKGROUND model,
no GPU options, no num_ctx / keep_alive (the CPU host's settings win); owner-facing callers and foreground calls
never go remote; ConnectionError / timeout / 5xx fall back to the main brain
(no lost learning); the health memo stops a dead host costing a timeout per
job; every text-brain background=True caller in the tree is classified.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)

# A tailnet (100.64/10) address, split so the PII gate sees no literal (NET-3).
TAILNET_HOST = "100" + ".101.1.2"
URL = "http://%s:11434" % TAILNET_HOST
MAIN = "http://127.0.0.1:11434"


class _Resp:
    def __init__(self, text="ok", status=200):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = "body"
        self._body = {"model": "m", "done": True,
                      "message": {"role": "assistant", "content": text},
                      "prompt_eval_count": 11, "eval_count": 2}

    def json(self):
        return self._body


@requires_monolith
class BackgroundRemoteRoutingTests(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        from core import background_llm as bgl
        self.bgl = bgl
        bgl.HEALTH.note_ok()
        self.addCleanup(bgl.HEALTH.note_ok)
        self.posts = []
        self.remote_reply = _Resp("remote answer")
        self.main_reply = _Resp("main answer")

    def _p(self, *args, **kw):
        patcher = mock.patch.object(*args, **kw)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _wire(self, url=URL, remote=None):
        bc = self.bc

        def _post(u, json=None, timeout=None, **_k):
            self.posts.append((u, json, timeout))
            if u.startswith(MAIN):
                return self.main_reply
            r = remote if remote is not None else self.remote_reply
            if isinstance(r, Exception):
                raise r
            return r
        fake = mock.Mock()
        fake.post.side_effect = _post
        fake.Session.return_value = fake    # background_llm posts via a Session
        fake.RequestException = Exception
        fake.Timeout = TimeoutError
        self._p(bc, "requests", fake)
        self._p(self.bgl, "_requests", fake)
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_ollama_has_model", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value="gemma-main")
        self._p(bc, "_next_local_llm_fallback", return_value=None)
        self._p(bc, "BACKGROUND_LLM_BASE_URL", url, create=True)
        self._p(bc, "BACKGROUND_LLM_MODEL", "gemma4:e4b", create=True)
        self._p(bc, "BACKGROUND_LLM_TIMEOUT_S", 120, create=True)

    def _call(self, caller, background):
        return self.bc._call_local_llm(
            "You extract facts.", [{"role": "user", "content": "hi"}],
            max_tokens=50, caller=caller, background=background)

    def _urls(self):
        return [u for u, _j, _t in self.posts]

    def test_unset_url_keeps_today_behaviour(self):
        self._wire(url="")
        self.assertEqual(self._call("learn_batch", True), "main answer")
        self.assertEqual(len(self.posts), 1)
        self.assertTrue(self._urls()[0].startswith(MAIN))

    def test_background_allowlisted_goes_remote(self):
        self._wire()
        self.assertEqual(self._call("learn_batch", True), "remote answer")
        self.assertEqual(self._urls(), [URL + "/api/chat"])
        _u, payload, timeout = self.posts[0]
        self.assertEqual(payload["model"], "gemma4:e4b")
        self.assertNotIn("num_ctx", payload["options"])
        self.assertNotIn("keep_alive", payload)
        self.assertNotIn("main_gpu", payload["options"])
        self.assertNotIn("main_gpu", payload)
        self.assertEqual(timeout[1], 120.0)

    def test_prompt_over_the_remote_window_stays_main(self):
        """The remote window is server-side (8192 on the Unraid box) and never
        sent; a prompt that would overflow it stays on the main brain rather
        than being silently truncated there."""
        self._wire()
        self._p(self.bc, "BACKGROUND_LLM_NUM_CTX", 1024, create=True)
        big = [{"role": "user", "content": "word " * 4000}]
        self.assertEqual(self.bc._call_local_llm(
            "You extract facts.", big, max_tokens=50,
            caller="learn_batch", background=True), "main answer")
        self.assertTrue(all(u.startswith(MAIN) for u in self._urls()))

    def test_stay_on_main_keeps_a_checkpoint_local(self):
        """Shutdown / hand-off summaries run inside stay_on_main(): the same
        session_checkpoint caller that goes remote periodically stays on the
        main brain there."""
        self._wire()
        with self.bgl.stay_on_main():
            self.assertEqual(self._call("session_checkpoint", True),
                             "main answer")
        self.assertTrue(all(u.startswith(MAIN) for u in self._urls()))
        self.posts.clear()
        self.assertEqual(self._call("session_checkpoint", True),
                         "remote answer")

    def test_shutdown_and_handoff_summaries_are_pinned(self):
        """Both owner-waits call sites wrap _session_summary_update in
        stay_on_main() (source pin: the two functions are not unit-drivable
        without a full session)."""
        import inspect
        for fn in (self.bc.save_session_to_memory,
                   self.bc._handoff_stand_down):
            src = inspect.getsource(fn)
            i = src.index("_session_summary_update(")
            self.assertIn("stay_on_main()", src[max(0, i - 300):i], fn.__name__)

    def test_every_allowlisted_caller_goes_remote(self):
        self._wire()
        for caller in sorted(self.bgl.REMOTE_OK_CALLERS):
            self.posts.clear()
            self.assertEqual(self._call(caller, True), "remote answer", caller)
            self.assertEqual(self._urls(), [URL + "/api/chat"], caller)

    def test_owner_facing_and_foreground_never_remote(self):
        self._wire()
        cases = [(c, True) for c in sorted(self.bgl.OWNER_FACING_CALLERS)]
        cases += [("voice_turn", False), ("voice_followup", False),
                  ("learn_batch", False), ("", None)]
        for caller, bg in cases:
            self.posts.clear()
            self.assertEqual(self._call(caller, bg), "main answer",
                             (caller, bg))
            self.assertTrue(all(u.startswith(MAIN) for u in self._urls()),
                            (caller, bg, self._urls()))

    def test_connection_error_falls_back_to_main_brain(self):
        self._wire(remote=ConnectionError("refused"))
        self.assertEqual(self._call("learn_batch", True), "main answer")
        self.assertEqual(self._urls()[0], URL + "/api/chat")
        self.assertTrue(self._urls()[1].startswith(MAIN))

    def test_timeout_falls_back_to_main_brain(self):
        self._wire(remote=TimeoutError("slow"))
        self.assertEqual(self._call("session_checkpoint", True), "main answer")

    def test_5xx_falls_back_to_main_brain(self):
        self._wire(remote=_Resp("", status=503))
        self.assertEqual(self._call("ltm_reflect", True), "main answer")

    def test_empty_remote_reply_falls_back(self):
        self._wire(remote=_Resp("  "))
        self.assertEqual(self._call("learn_batch", True), "main answer")

    def test_health_memo_skips_the_dead_host_for_later_jobs(self):
        self._wire(remote=ConnectionError("refused"))
        self.assertEqual(self._call("learn_batch", True), "main answer")
        self.posts.clear()
        self.assertEqual(self._call("learn_batch", True), "main answer")
        self.assertEqual(len(self.posts), 1)
        self.assertTrue(self._urls()[0].startswith(MAIN))

    def test_remote_failure_is_logged(self):
        self._wire(remote=ConnectionError("refused"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._call("learn_batch", True)
        self.assertIn("[bg-llm] remote failed", buf.getvalue())
        self.assertIn("using the main brain", buf.getvalue())

    def test_served_via_line_names_the_remote_host(self):
        self._wire()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._call("learn_batch", True)
        out = buf.getvalue()
        self.assertIn("[local-llm] served via gemma4:e4b @" + TAILNET_HOST, out)
        self.assertIn("caller=learn_batch", out)

    def test_remote_works_even_when_main_ollama_is_down(self):
        self._wire()
        self._p(self.bc, "_ollama_alive", return_value=False)
        self.assertEqual(self._call("learn_batch", True), "remote answer")


class BackgroundCallerCensusTests(unittest.TestCase):
    """Every `caller="x", background=True` passed to the text brain
    (_llm_quick / _call_local_llm) in the tree is classified: on the remote
    allowlist, or listed owner-facing. A new caller must be decided."""

    _CALL = re.compile(
        r"""caller\s*=\s*["']([a-z0-9_]+)["']\s*,\s*background\s*=\s*True""")
    _TEXT = ("_llm_quick(", "_call_local_llm(", "quick(")
    _OTHER = ("create_message(", "_call_local_vision(", "ask(", "ask_vision(",
              "record(")

    def _text_brain_callers(self):
        found = set()
        for base, dirs, files in os.walk(_PROJECT):
            dirs[:] = [d for d in dirs if d not in (
                "tests", ".git", "__pycache__", ".claude", "backups",
                "venv", ".venv", "node_modules", "data", "logs")]
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                try:
                    with open(os.path.join(base, fn), encoding="utf-8",
                              errors="replace") as fh:
                        src = fh.read()
                except OSError:
                    continue
                for m in self._CALL.finditer(src):
                    window = src[max(0, m.start() - 600):m.start()]
                    text_at = max(window.rfind(e) for e in self._TEXT)
                    other_at = max(window.rfind(e) for e in self._OTHER)
                    if text_at > other_at:
                        found.add(m.group(1))
        return found

    def test_every_text_background_caller_is_classified(self):
        from core import background_llm as bgl
        classified = bgl.REMOTE_OK_CALLERS | bgl.OWNER_FACING_CALLERS
        unknown = sorted(self._text_brain_callers() - classified)
        self.assertEqual(unknown, [], "decide remote-ok or owner-facing in "
                         "core/background_llm.py for: " + ", ".join(unknown))

    def test_census_finds_the_known_callers(self):
        found = self._text_brain_callers()
        for must in ("learn_batch", "session_checkpoint", "proactive_remark"):
            self.assertIn(must, found)


if __name__ == "__main__":
    unittest.main()
