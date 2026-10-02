"""Tests for core.claude_model_guard and its wiring into core.llm_client
(2026-10-02, the retired / unknown Claude model guard).

Anthropic answers a request for a retired model, a model id that does not
exist, or one the key's organization cannot use with HTTP 404
``not_found_error`` and a message starting ``model: <id>``. The guard turns
that into ONE log line per model per session, a retry on a configured
successor, or a RuntimeError the caller's local fallback catches — instead of
paying the 404 on every call.

No network: the Anthropic client is a fake, and the real SDK is used only to
BUILD a genuine NotFoundError (skipped where the SDK is absent). The process-
wide GUARD is reset around every test.

    python -B -m unittest tests.test_claude_model_guard
"""
from __future__ import annotations

import contextlib
import io
import threading
import unittest
from unittest import mock

import core.claude_model_guard as mg
import core.llm_client as llm

try:  # the real SDK error, built offline (anthropic depends on httpx)
    import anthropic as _anthropic
    import httpx as _httpx
except Exception:  # pragma: no cover - CI installs anthropic
    _anthropic = None
    _httpx = None


def _not_found_body(model: str) -> dict:
    return {"type": "error",
            "error": {"type": "not_found_error", "message": f"model: {model}"},
            "request_id": "req_test"}


def _sdk_not_found(model: str):
    """A real anthropic.NotFoundError, exactly as the SDK raises it."""
    body = _not_found_body(model)
    req = _httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = _httpx.Response(404, request=req, json=body)
    return _anthropic.NotFoundError(f"Error code: 404 - {body}",
                                    response=resp, body=body)


class _DuckNotFound(Exception):
    """The SDK error's shape without the SDK: status_code + body."""

    def __init__(self, model: str):
        self.status_code = 404
        self.body = _not_found_body(model)
        super().__init__(f"Error code: 404 - {self.body}")


class _ProviderError(Exception):
    """browser-use's ModelProviderError shape: a 404 status_code and the SDK's
    message, raised `from` the SDK error."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class _Clock:
    def __init__(self, t: float = 5000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _guard(clock=None, recheck_after_s=3600.0):
    lines: list[str] = []
    g = mg.ModelGuard(sink=lines.append, clock=clock or _Clock(),
                      recheck_after_s=recheck_after_s)
    return g, lines


# ── is_model_not_found ────────────────────────────────────────────────────

class IsModelNotFoundTests(unittest.TestCase):
    @unittest.skipUnless(_anthropic is not None, "anthropic SDK not installed")
    def test_real_sdk_not_found_error(self):
        err = _sdk_not_found("claude-haiku-4-5")
        self.assertTrue(mg.is_model_not_found(err))
        self.assertTrue(mg.is_model_not_found(err, "claude-haiku-4-5"))

    @unittest.skipUnless(_anthropic is not None, "anthropic SDK not installed")
    def test_real_sdk_other_errors_are_not_not_found(self):
        req = _httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        body = {"type": "error", "error": {
            "type": "invalid_request_error",
            "message": "model: temperature is not supported"}}
        bad = _anthropic.BadRequestError(
            "Error code: 400", response=_httpx.Response(400, request=req,
                                                        json=body), body=body)
        self.assertFalse(mg.is_model_not_found(bad, "claude-x"))
        body = {"type": "error", "error": {"type": "rate_limit_error",
                                           "message": "slow down"}}
        rl = _anthropic.RateLimitError(
            "Error code: 429", response=_httpx.Response(429, request=req,
                                                        json=body), body=body)
        self.assertFalse(mg.is_model_not_found(rl, "claude-x"))

    def test_duck_typed_error(self):
        self.assertTrue(mg.is_model_not_found(_DuckNotFound("claude-x")))

    def test_404_that_is_not_about_a_model_is_not_a_retirement(self):
        class _E(Exception):
            status_code = 404
            body = {"type": "error", "error": {"type": "not_found_error",
                                               "message": "file_abc not found"}}
        self.assertFalse(mg.is_model_not_found(_E("Not Found")))

    def test_wrapped_error_is_followed_through_cause(self):
        try:
            try:
                raise _DuckNotFound("claude-x")
            except _DuckNotFound as inner:
                raise RuntimeError("step failed") from inner
        except RuntimeError as outer:
            self.assertTrue(mg.is_model_not_found(outer, "claude-x"))

    def test_browser_use_provider_error_shape(self):
        msg = f"Error code: 404 - {_not_found_body('claude-sonnet-5')}"
        self.assertTrue(mg.is_model_not_found(_ProviderError(msg, 404),
                                              "claude-sonnet-5"))

    def test_error_text(self):
        text = ("ModelProviderError: Error code: 404 - {'type': 'error', "
                "'error': {'type': 'not_found_error', 'message': "
                "'model: claude-sonnet-5'}}")
        self.assertTrue(mg.is_model_not_found(text, "claude-sonnet-5"))
        self.assertFalse(mg.is_model_not_found("timed out after 30s"))

    def test_plain_failures_and_junk(self):
        for err in (RuntimeError("boom"), ValueError("model: bad value"),
                    None, 404, "", object()):
            with self.subTest(err=err):
                self.assertFalse(mg.is_model_not_found(err, "claude-x"))


# ── successor_for ─────────────────────────────────────────────────────────

class SuccessorForTests(unittest.TestCase):
    TABLE = {"claude-haiku-4-5": "claude-sonnet-5-5",
             "Claude-Old": "claude-new", "same": "same"}

    def test_exact_and_case_insensitive(self):
        self.assertEqual(mg.successor_for("claude-haiku-4-5", self.TABLE),
                         "claude-sonnet-5-5")
        self.assertEqual(mg.successor_for("CLAUDE-OLD", self.TABLE), "claude-new")

    def test_snapshot_id_matches_its_alias_entry(self):
        self.assertEqual(
            mg.successor_for("claude-haiku-4-5-20251001", self.TABLE),
            "claude-sonnet-5-5")

    def test_no_entry_self_entry_or_bad_table(self):
        self.assertEqual(mg.successor_for("claude-other", self.TABLE), "")
        self.assertEqual(mg.successor_for("same", self.TABLE), "")
        self.assertEqual(mg.successor_for("claude-haiku-4-5", None), "")
        self.assertEqual(mg.successor_for("claude-haiku-4-5", ["x"]), "")
        self.assertEqual(mg.successor_for("", self.TABLE), "")


# ── ModelGuard ────────────────────────────────────────────────────────────

class ModelGuardTests(unittest.TestCase):
    def test_one_line_per_model_per_session(self):
        g, lines = _guard()
        self.assertTrue(g.note_not_found("claude-haiku-4-5", where="a 'classify' call"))
        self.assertFalse(g.note_not_found("claude-haiku-4-5"))
        self.assertFalse(g.note_not_found("CLAUDE-HAIKU-4-5"))
        self.assertEqual(len(lines), 1)
        self.assertIn("[model-guard]", lines[0])
        self.assertIn("claude-haiku-4-5", lines[0])
        self.assertIn("not_found", lines[0])
        self.assertIn("local path", lines[0])
        g.note_not_found("claude-sonnet-5")
        self.assertEqual(len(lines), 2)

    def test_line_names_the_successor_when_there_is_one(self):
        g, lines = _guard()
        g.note_not_found("claude-haiku-4-5", successor="claude-sonnet-5-5")
        self.assertIn("claude-sonnet-5-5", lines[0])

    def test_resolve(self):
        g, _ = _guard()
        table = {"claude-haiku-4-5": "claude-sonnet-5-5"}
        self.assertEqual(g.resolve("claude-haiku-4-5", table), "claude-haiku-4-5")
        g.note_not_found("claude-haiku-4-5")
        self.assertEqual(g.resolve("claude-haiku-4-5", table), "claude-sonnet-5-5")
        self.assertIsNone(g.resolve("claude-haiku-4-5", {}))
        g.note_not_found("claude-sonnet-5-5")          # successor gone too
        self.assertIsNone(g.resolve("claude-haiku-4-5", table))
        self.assertIsNone(g.resolve(None))             # blank → itself
        self.assertEqual(g.resolve("", table), "")

    def test_recheck_window_then_silent_remark(self):
        clock = _Clock()
        g, lines = _guard(clock=clock, recheck_after_s=60.0)
        g.note_not_found("claude-x")
        self.assertTrue(g.is_retired("claude-x"))
        clock.t += 59.0
        self.assertTrue(g.is_retired("claude-x"))
        clock.t += 2.0                                   # past the window
        self.assertFalse(g.is_retired("claude-x"))
        self.assertEqual(g.resolve("claude-x"), "claude-x")   # one real try
        g.note_not_found("claude-x")                     # still gone: silent
        self.assertTrue(g.is_retired("claude-x"))
        self.assertEqual(len(lines), 1)

    def test_note_ok_clears_a_mark_with_one_line(self):
        g, lines = _guard()
        g.note_ok("claude-x")                            # never marked: silent
        self.assertEqual(lines, [])
        g.note_not_found("claude-x")
        g.note_ok("claude-x")
        self.assertFalse(g.is_retired("claude-x"))
        self.assertEqual(len(lines), 2)
        self.assertIn("answering again", lines[1])

    def test_reset_and_listing(self):
        g, lines = _guard()
        g.note_not_found("b-model")
        g.note_not_found("a-model")
        self.assertEqual(g.retired_models(), ["a-model", "b-model"])
        g.reset()
        self.assertEqual(g.retired_models(), [])
        g.note_not_found("a-model")
        self.assertEqual(len(lines), 3)                  # announced afresh

    def test_concurrent_reports_log_once(self):
        g, lines = _guard()
        barrier = threading.Barrier(8)

        def _hit():
            barrier.wait()
            g.note_not_found("claude-x")
        threads = [threading.Thread(target=_hit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(len(lines), 1)

    def test_a_raising_sink_never_breaks_the_caller(self):
        def _bad(_line):
            raise OSError("closed stdout")
        g = mg.ModelGuard(sink=_bad, clock=_Clock())
        self.assertTrue(g.note_not_found("claude-x"))

    def test_retired_model_error_is_a_runtime_error_naming_the_model(self):
        e = mg.RetiredModelError("claude-x", "a 'vision' call")
        self.assertIsInstance(e, RuntimeError)
        self.assertEqual(e.model, "claude-x")
        self.assertIn("claude-x", str(e))
        self.assertIn("CLAUDE_MODEL_SUCCESSORS", str(e))


# ── core.llm_client wiring (the one Claude chokepoint) ────────────────────

class _Msg:
    def __init__(self, text="ok"):
        blk = mock.Mock()
        blk.type = "text"
        blk.text = text
        self.content = [blk]
        self.stop_reason = "end_turn"
        self.usage = None


class _Messages:
    """messages.create / .stream: raises `errors[model]` for a listed model,
    answers otherwise. Records every request."""

    def __init__(self, errors=None):
        self.errors = dict(errors or {})
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        err = self.errors.get(kwargs.get("model"))
        if err is not None:
            raise err
        return _Msg(f"answer from {kwargs.get('model')}")

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        err = self.errors.get(kwargs.get("model"))
        outer = self

        class _S:
            text_stream = ["str", "eamed"]

            def __enter__(self):
                if err is not None:
                    raise err            # the request goes out on enter
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return _Msg("streamed")
        _S.calls = outer.calls
        return _S()


class _Client:
    def __init__(self, errors=None):
        self.messages = _Messages(errors)


class _GuardedBase(unittest.TestCase):
    def setUp(self):
        mg.GUARD.reset()
        self.addCleanup(mg.GUARD.reset)
        self.out = io.StringIO()
        redirect = contextlib.redirect_stdout(self.out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def _successors(self, table):
        p = mock.patch.object(llm, "_successor_table", return_value=dict(table))
        p.start()
        self.addCleanup(p.stop)

    def _guard_lines(self):
        return [ln for ln in self.out.getvalue().splitlines()
                if "[model-guard]" in ln]


class CreateMessageGuardTests(_GuardedBase):
    def test_working_model_is_untouched(self):
        client = _Client()
        msg = llm.create_message(client, purpose="classify",
                                 model="claude-haiku-4-5", max_tokens=8,
                                 messages=[{"role": "user", "content": "x"}])
        self.assertEqual(llm.response_text(msg), "answer from claude-haiku-4-5")
        self.assertEqual(len(client.messages.calls), 1)
        self.assertEqual(client.messages.calls[0]["model"], "claude-haiku-4-5")
        self.assertEqual(self._guard_lines(), [])

    def test_first_not_found_propagates_then_calls_skip_the_network(self):
        client = _Client({"claude-gone": _DuckNotFound("claude-gone")})
        kw = dict(purpose="classify", model="claude-gone", max_tokens=8,
                  messages=[{"role": "user", "content": "x"}])
        with self.assertRaises(_DuckNotFound):          # the SDK's own error
            llm.create_message(client, **kw)
        self.assertEqual(len(self._guard_lines()), 1)
        for _ in range(3):
            with self.assertRaises(mg.RetiredModelError):
                llm.create_message(client, **kw)
        self.assertEqual(len(client.messages.calls), 1)  # no repeat 404s
        self.assertEqual(len(self._guard_lines()), 1)    # ONE line

    @unittest.skipUnless(_anthropic is not None, "anthropic SDK not installed")
    def test_real_sdk_error_still_reaches_api_status_handlers(self):
        client = _Client({"claude-gone": _sdk_not_found("claude-gone")})
        with self.assertRaises(_anthropic.APIStatusError) as cm:
            llm.create_message(client, purpose="voice", model="claude-gone",
                               max_tokens=8, messages=[])
        self.assertEqual(cm.exception.status_code, 404)

    def test_configured_successor_answers_and_keeps_answering(self):
        self._successors({"claude-gone": "claude-sonnet-5-5"})
        client = _Client({"claude-gone": _DuckNotFound("claude-gone")})
        kw = dict(purpose="classify", model="claude-gone", max_tokens=8,
                  messages=[{"role": "user", "content": "x"}])
        msg = llm.create_message(client, **kw)
        self.assertEqual(llm.response_text(msg), "answer from claude-sonnet-5-5")
        self.assertEqual([c["model"] for c in client.messages.calls],
                         ["claude-gone", "claude-sonnet-5-5"])
        # The retry is shaped for the SUCCESSOR (Sonnet 5.5: effort + floor).
        retry = client.messages.calls[1]
        self.assertEqual(retry["extra_body"]["output_config"]["effort"], "low")
        self.assertGreaterEqual(retry["max_tokens"], 2048)
        llm.create_message(client, **kw)                 # straight to it now
        self.assertEqual([c["model"] for c in client.messages.calls],
                         ["claude-gone", "claude-sonnet-5-5", "claude-sonnet-5-5"])
        lines = self._guard_lines()
        self.assertEqual(len(lines), 1)
        self.assertIn("claude-sonnet-5-5", lines[0])

    def test_successor_that_is_gone_too_raises(self):
        self._successors({"claude-gone": "claude-also-gone"})
        client = _Client({"claude-gone": _DuckNotFound("claude-gone"),
                          "claude-also-gone": _DuckNotFound("claude-also-gone")})
        kw = dict(purpose="quick", model="claude-gone", max_tokens=8, messages=[])
        with self.assertRaises(_DuckNotFound):
            llm.create_message(client, **kw)
        self.assertTrue(mg.GUARD.is_retired("claude-also-gone"))
        with self.assertRaises(mg.RetiredModelError):
            llm.create_message(client, **kw)
        self.assertEqual(len(client.messages.calls), 2)

    def test_blank_model_is_passed_through_not_called_retired(self):
        # resolve() returns a blank model as itself (None stays None); that
        # must never read as "gone" and raise RetiredModelError.
        for blank in (None, "", "   "):
            client = _Client()
            with self.subTest(model=blank):
                llm.create_message(client, purpose="quick", model=blank,
                                   max_tokens=8, messages=[])
                self.assertEqual(len(client.messages.calls), 1)
        client = _Client()
        llm.stream_message(client, purpose="voice", model=None,
                           max_tokens=8, messages=[])
        self.assertEqual(len(client.messages.calls), 1)

    def test_other_errors_pass_through_and_mark_nothing(self):
        client = _Client({"claude-haiku-4-5": RuntimeError("socket reset")})
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                llm.create_message(client, purpose="classify",
                                   model="claude-haiku-4-5", max_tokens=8,
                                   messages=[])
        self.assertEqual(len(client.messages.calls), 2)
        self.assertEqual(mg.GUARD.retired_models(), [])

    def test_success_after_the_recheck_window_clears_the_mark(self):
        mg.GUARD.note_not_found("claude-back")
        clock = mock.patch.object(mg.GUARD, "is_retired", return_value=False)
        with clock:
            llm.create_message(_Client(), purpose="quick", model="claude-back",
                               max_tokens=8, messages=[])
        self.assertEqual(mg.GUARD.retired_models(), [])
        self.assertTrue(any("answering again" in ln for ln in self._guard_lines()))

    def test_complete_goes_through_the_guard(self):
        client = _Client({"claude-gone": _DuckNotFound("claude-gone")})
        with mock.patch.object(llm, "_client", return_value=client):
            with self.assertRaises(_DuckNotFound):
                llm.complete(model="claude-gone", messages=[])
            with self.assertRaises(mg.RetiredModelError):
                llm.complete(model="claude-gone", messages=[])
        self.assertEqual(len(client.messages.calls), 1)


class StreamGuardTests(_GuardedBase):
    def test_not_found_on_open_is_noted_then_skipped(self):
        client = _Client({"claude-gone": _DuckNotFound("claude-gone")})
        with mock.patch.object(llm, "_client", return_value=client):
            with self.assertRaises(_DuckNotFound):
                llm.stream_text(model="claude-gone", messages=[])
            self.assertTrue(mg.GUARD.is_retired("claude-gone"))
            with self.assertRaises(mg.RetiredModelError):
                llm.stream_text(model="claude-gone", messages=[])
        self.assertEqual(len(client.messages.calls), 1)
        self.assertEqual(len(self._guard_lines()), 1)

    def test_stream_uses_the_successor_and_records_its_usage(self):
        self._successors({"claude-gone": "claude-sonnet-5-5"})
        mg.GUARD.note_not_found("claude-gone")
        client = _Client()
        with mock.patch.object(llm, "_client", return_value=client), \
                mock.patch.object(llm, "_record_session_usage") as rec:
            out = llm.stream_text(model="claude-gone", messages=[])
        self.assertEqual(out, "streamed")
        self.assertEqual(client.messages.calls[0]["model"], "claude-sonnet-5-5")
        self.assertEqual(rec.call_args.args[0], "claude-sonnet-5-5")

    def test_stream_message_raises_for_a_known_gone_model(self):
        mg.GUARD.note_not_found("claude-gone")
        client = _Client()
        with self.assertRaises(mg.RetiredModelError):
            llm.stream_message(client, purpose="voice", model="claude-gone",
                               max_tokens=8, messages=[])
        self.assertEqual(client.messages.calls, [])


class SuccessorTableReaderTests(unittest.TestCase):
    def test_reads_core_config_at_call_time(self):
        import core.config as cfg
        with mock.patch.object(cfg, "CLAUDE_MODEL_SUCCESSORS",
                               {"claude-a": "claude-b"}):
            self.assertEqual(llm._successor_table(), {"claude-a": "claude-b"})
        with mock.patch.object(cfg, "CLAUDE_MODEL_SUCCESSORS", ["not a dict"]):
            self.assertEqual(llm._successor_table(), {})

    def test_shipped_defaults(self):
        import core.config as cfg
        self.assertEqual(cfg.CLAUDE_MODEL_SUCCESSORS, {})
        self.assertEqual(cfg.CLAUDE_FAST_MODEL, "claude-haiku-4-5")


if __name__ == "__main__":
    unittest.main()
