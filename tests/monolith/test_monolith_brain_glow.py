"""Brain glow, monolith side: the brain that actually SERVED a chat turn
reaches hud_state.json (``brain``), so the HUD orb shows local blue / Sonnet
gold / Opus violet per turn — including the fallback turns (local -> cloud on
the local route, cloud -> local on the cloud route).

Pinned here (local full tier; skipped on the light CI runner):

  * a local-route turn served by the local model publishes ("local", tag);
  * a local-route turn the local model could not answer, served by Claude
    instead, publishes ("cloud", CLAUDE_MODEL);
  * a cloud turn publishes ("cloud", CLAUDE_MODEL); a cloud turn whose call
    failed and was answered locally publishes ("local", tag);
  * the AI_BACKEND=ollama branch publishes ("local", tag);
  * a turn NOTHING answered (honest "both down" line) publishes nothing — the
    last real brain keeps glowing;
  * two turns on the same brain = ONE hud_state write (cheap per turn);
  * the per-turn record is THREAD-LOCAL — a background local call on another
    thread (learn_from_turn, ambient extract...) can never recolour the orb;
  * BRAIN_GLOW_ENABLED False = no brain write at all;
  * boot publishes the brain the first turn will use;
  * PARITY: core.brain_glow.expected_brain() names the same brain _call_llm
    really serves for each routing config, so the colour shown after a switch
    is the colour the next turn keeps.

Network is never touched: the local path runs through a canned Ollama body on
a fake ``requests`` (the test_monolith_turn_timing recipe) and the cloud path
through a fake ``_llm_client``.
"""
from __future__ import annotations

import contextlib
import io
import threading
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_LOCAL_TAG = "gemma-test:12b"
_BODY = {"model": _LOCAL_TAG,
         "message": {"role": "assistant", "content": "It is noon, sir."},
         "done": True}


class _Resp:
    def __init__(self, body):
        self._body = body
        self.ok = True
        self.status_code = 200
        self.text = "body"

    def json(self):
        return self._body


class _FakeClient:
    """Stand-in for core.llm_client: complete() answers or raises."""

    def __init__(self, reply="Cloud reply, sir.", exc=None):
        self.reply = reply
        self.exc = exc
        self.calls = []

    def complete(self, **kw):
        self.calls.append(kw)
        if self.exc is not None:
            raise self.exc
        return self.reply

    def stream_text(self, **kw):   # never reached: streaming is patched off
        raise AssertionError("stream_text should not be used in these tests")


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        from core import brain_glow
        self.bg = brain_glow
        brain_glow.PUBLISHER.reset()
        self.addCleanup(brain_glow.PUBLISHER.reset)
        self._hist_len = len(self.bc.conversation_history)
        self.addCleanup(self._restore_hist)
        self.hud = self._p(self.bc, "_write_hud_state")
        self._p(brain_glow, "settings", return_value=(True, 4.0, {}))
        self._p(self.bc, "_system_prompt", "SYS")
        self._p(self.bc, "_mcu_phrases",
                mock.Mock(detect_phrases_in_reply=mock.Mock(return_value={})))
        self._p(self.bc, "_streaming_tts_enabled", return_value=False)
        self._p(self.bc, "CLAUDE_MODEL", "claude-opus-5-5")
        # The turn's TLS slot starts clean for every test.
        self.bc._turn_brain_tls.served = None

    def _restore_hist(self):
        del self.bc.conversation_history[self._hist_len:]

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    # ── route knobs ───────────────────────────────────────────────────────
    def _route(self, chat):
        import core.config as cfg
        self._p(cfg, "model_route", side_effect=lambda f: chat if f == "chat"
                else "auto")

    def _local_ok(self, model=_LOCAL_TAG):
        fake_req = mock.Mock()
        fake_req.post.side_effect = lambda url, json=None, timeout=None, **k: \
            _Resp(dict(_BODY, model=json["model"] if json else model))
        self._p(self.bc, "LOCAL_LLM_FALLBACK", True)
        self._p(self.bc, "_ollama_alive", return_value=True)
        self._p(self.bc, "_ollama_has_model", return_value=True)
        self._p(self.bc, "_get_local_llm_model", return_value=model)
        self._p(self.bc, "_next_local_llm_fallback", return_value=None)
        self._p(self.bc, "requests", fake_req)
        return fake_req

    def _local_down(self):
        self._p(self.bc, "LOCAL_LLM_FALLBACK", True)
        self._p(self.bc, "_ollama_alive", return_value=False)
        self._p(self.bc, "_ollama_selfheal_async")
        self._p(self.bc, "_ollama_install_async")
        self._p(self.bc, "_sac_blocked_local_recently", return_value=False)

    def _cloud(self, client):
        self._p(self.bc, "AI_BACKEND", "claude")
        self._p(self.bc, "_llm_client", client)
        self._p(self.bc, "_claude_reachable", return_value=True)

    def _turn(self, text="what time is it"):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc._call_llm(text)

    def _brains(self):
        """Every brain= value written to hud_state, in order."""
        return [c.kwargs["brain"] for c in self.hud.call_args_list
                if "brain" in c.kwargs]


class PerTurnBrainTests(_Base):
    def test_local_turn_publishes_the_local_model(self):
        self._route("local")
        self._local_ok()
        self.assertEqual(self._turn(), "It is noon, sir.")
        brains = self._brains()
        self.assertEqual(len(brains), 1)
        self.assertEqual((brains[0]["route"], brains[0]["model"]),
                         ("local", _LOCAL_TAG))
        self.assertEqual(brains[0]["tier"], "local")
        self.assertEqual(brains[0]["source"], "turn")

    def test_local_route_cloud_fallback_publishes_claude(self):
        self._route("local")
        self._local_down()
        self._cloud(_FakeClient("From the cloud, sir."))
        self.assertEqual(self._turn(), "From the cloud, sir.")
        brains = self._brains()
        self.assertEqual(len(brains), 1)
        self.assertEqual((brains[0]["route"], brains[0]["model"]),
                         ("cloud", "claude-opus-5-5"))
        self.assertEqual(brains[0]["tier"], "opus")

    def test_cloud_turn_publishes_claude(self):
        self._route("auto")
        client = _FakeClient("Cloud reply, sir.")
        self._cloud(client)
        self.assertEqual(self._turn(), "Cloud reply, sir.")
        self.assertEqual(len(client.calls), 1)
        brains = self._brains()
        self.assertEqual([(b["route"], b["model"]) for b in brains],
                         [("cloud", "claude-opus-5-5")])

    def test_cloud_failure_answered_locally_publishes_local(self):
        self._route("auto")
        self._cloud(_FakeClient(exc=RuntimeError("cloud blew up")))
        self._local_ok()
        self.assertEqual(self._turn(), "It is noon, sir.")
        brains = self._brains()
        self.assertEqual([(b["route"], b["model"]) for b in brains],
                         [("local", _LOCAL_TAG)])

    def test_ollama_backend_branch_publishes_local(self):
        self._route("auto")
        self._p(self.bc, "AI_BACKEND", "ollama")
        self._p(self.bc, "_get_local_llm_model", return_value=_LOCAL_TAG)
        self._p(self.bc, "_ollama_chat_bounded",
                return_value={"message": {"content": "Local branch, sir."}})
        self.assertEqual(self._turn(), "Local branch, sir.")
        self.assertEqual([(b["route"], b["model"]) for b in self._brains()],
                         [("local", _LOCAL_TAG)])

    def test_turn_nothing_answered_publishes_nothing(self):
        self._route("local")
        self._local_down()
        self._p(self.bc, "_claude_reachable", return_value=False)
        reply = self._turn()
        self.assertEqual(reply, self.bc._local_unavailable_message())
        self.assertEqual(self._brains(), [])

    def test_stale_note_from_before_the_turn_is_never_published(self):
        # A generate on the voice thread BEFORE this turn (the previous
        # turn's follow-up round, a voice-thread helper) left a note behind.
        # A turn that nothing answered must not publish that stale brain.
        self.bc._note_turn_brain("local", "stale:1b")
        self._route("local")
        self._local_down()
        self._p(self.bc, "_claude_reachable", return_value=False)
        self._turn()
        self.assertEqual(self._brains(), [])

    def test_two_turns_same_brain_one_write(self):
        self._route("local")
        self._local_ok()
        self._turn("first")
        self._turn("second")
        self._turn("third")
        self.assertEqual(len(self._brains()), 1)

    def test_brain_change_between_turns_writes_again(self):
        self._route("local")
        self._local_ok()
        self._turn("first")
        # Same route, but the local brain was swapped (set_model / game mode).
        self._p(self.bc, "_get_local_llm_model", return_value="qwen3:14b")
        self._turn("second")
        self.assertEqual([b["model"] for b in self._brains()],
                         [_LOCAL_TAG, "qwen3:14b"])

    def test_glow_disabled_writes_no_brain(self):
        self._p(self.bg, "settings", return_value=(False, 4.0, {}))
        self._route("local")
        self._local_ok()
        self._turn()
        self.assertEqual(self._brains(), [])

    def test_publish_failure_never_breaks_the_turn(self):
        self._p(self.bg, "publish", side_effect=RuntimeError("glow broke"))
        self._route("local")
        self._local_ok()
        self.assertEqual(self._turn(), "It is noon, sir.")

    def test_turn_record_is_thread_local(self):
        # A background local call on ANOTHER thread (learn_from_turn, ambient
        # extract, vision) must not become the owner turn's brain.
        self._local_ok(model="background:1b")
        t = threading.Thread(target=lambda: self.bc._call_local_llm(
            "sys", [{"role": "user", "content": "bg"}]))
        with contextlib.redirect_stdout(io.StringIO()):
            t.start()
            t.join()
        self.assertIsNone(getattr(self.bc._turn_brain_tls, "served", None))
        self.bc._publish_turn_brain()
        self.assertEqual(self._brains(), [])

    def test_note_is_cheap(self):
        # The note runs on every local generate (voice thread included): it
        # must stay a single attribute store, far under a millisecond.
        import time as _t
        n = 20000
        t0 = _t.perf_counter()
        for _ in range(n):
            self.bc._note_turn_brain("local", _LOCAL_TAG)
        per_call_us = (_t.perf_counter() - t0) / n * 1e6
        self.assertLess(per_call_us, 50.0)


class BootPublishTests(_Base):
    def test_boot_publishes_the_expected_brain(self):
        self._route("auto")
        self._p(self.bc, "AI_BACKEND", "claude")
        self.bc._publish_boot_brain_glow()
        brains = self._brains()
        self.assertEqual(len(brains), 1)
        self.assertEqual((brains[0]["route"], brains[0]["model"],
                          brains[0]["source"]),
                         ("cloud", "claude-opus-5-5", "boot"))

    def test_boot_publish_never_raises(self):
        self._p(self.bg, "publish_expected", side_effect=RuntimeError("x"))
        self.bc._publish_boot_brain_glow()   # no raise

    def test_tray_restore_calls_the_boot_publish(self):
        import inspect
        src = inspect.getsource(self.bc._restore_tray_toggle_state)
        self.assertIn("_publish_boot_brain_glow()", src)


class ExpectedBrainParityTests(_Base):
    """expected_brain() must name the brain _call_llm REALLY serves when every
    call succeeds — otherwise the colour shown after a switch would flip on
    the very next turn."""

    def _served(self):
        self.hud.reset_mock()
        self.bg.PUBLISHER.reset()
        self._turn()
        brains = self._brains()
        self.assertEqual(len(brains), 1)
        return brains[0]["route"], brains[0]["model"]

    def test_local_route(self):
        self._route("local")
        self._p(self.bc, "AI_BACKEND", "claude")
        self._local_ok()
        self._p(self.bc, "_RESOLVED_LOCAL_LLM_MODEL", [_LOCAL_TAG])
        self.assertEqual(self.bg.expected_brain(self.bc), self._served())

    def test_auto_route_claude_backend(self):
        self._route("auto")
        self._cloud(_FakeClient("ok"))
        self.assertEqual(self.bg.expected_brain(self.bc), self._served())

    def test_cloud_route_claude_backend(self):
        self._route("cloud")
        self._cloud(_FakeClient("ok"))
        self.assertEqual(self.bg.expected_brain(self.bc), self._served())

    def test_auto_route_ollama_backend(self):
        self._route("auto")
        self._p(self.bc, "AI_BACKEND", "ollama")
        self._p(self.bc, "_RESOLVED_LOCAL_LLM_MODEL", [_LOCAL_TAG])
        self._p(self.bc, "_get_local_llm_model", return_value=_LOCAL_TAG)
        self._p(self.bc, "_ollama_chat_bounded",
                return_value={"message": {"content": "ok"}})
        self.assertEqual(self.bg.expected_brain(self.bc), self._served())


if __name__ == "__main__":
    unittest.main()
