"""Tests for core.llm_client — the shared Anthropic call wrapper. We patch the
private _client() factory with a fake so these run with no network and no
`anthropic` install: they pin the param-shaping, text extraction, streaming
accumulation, and the on_delta-must-not-abort contract."""
import sys
import types
import unittest
from unittest import mock

import core.llm_client as llm


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeMsg:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeStream:
    def __init__(self, chunks):
        self.text_stream = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeMessages:
    def __init__(self, captured, chunks):
        self._captured = captured
        self._chunks = chunks

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeMsg("hello sir")

    def stream(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeStream(self._chunks)


class _FakeClient:
    def __init__(self, captured, chunks):
        self.messages = _FakeMessages(captured, chunks)


def _patch(captured, chunks=("Hel", "lo ", "sir")):
    return mock.patch.object(llm, "_client",
                             return_value=_FakeClient(captured, list(chunks)))


class CompleteTests(unittest.TestCase):
    def test_returns_text_and_passes_params(self):
        cap = {}
        with _patch(cap):
            out = llm.complete(model="m", messages=[{"role": "user", "content": "hi"}],
                               system="sys", max_tokens=123)
        self.assertEqual(out, "hello sir")
        self.assertEqual(cap["model"], "m")
        self.assertEqual(cap["max_tokens"], 123)
        self.assertEqual(cap["system"], "sys")
        self.assertEqual(cap["messages"], [{"role": "user", "content": "hi"}])

    def test_omits_system_when_none(self):
        cap = {}
        with _patch(cap):
            llm.complete(model="m", messages=[{"role": "user", "content": "hi"}])
        self.assertNotIn("system", cap)

    def test_default_max_tokens(self):
        cap = {}
        with _patch(cap):
            llm.complete(model="m", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(cap["max_tokens"], 500)


class StreamTests(unittest.TestCase):
    def test_accumulates_and_calls_on_delta(self):
        cap, deltas = {}, []
        with _patch(cap):
            out = llm.stream_text(model="m", messages=[{"role": "user", "content": "hi"}],
                                  system="sys", on_delta=deltas.append)
        self.assertEqual(out, "Hello sir")
        self.assertEqual(deltas, ["Hel", "lo ", "sir"])

    def test_on_delta_errors_do_not_abort(self):
        cap = {}

        def boom(_chunk):
            raise ValueError("callback blew up")

        with _patch(cap):
            out = llm.stream_text(model="m", messages=[{"role": "user", "content": "hi"}],
                                  on_delta=boom)
        self.assertEqual(out, "Hello sir")  # full text despite the raising callback

    def test_works_without_on_delta(self):
        cap = {}
        with _patch(cap):
            out = llm.stream_text(model="m", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(out, "Hello sir")


class FirstTextTests(unittest.TestCase):
    def test_skips_leading_non_text_block(self):
        class ToolBlock:
            pass

        class Msg:
            content = [ToolBlock(), _FakeBlock("real text")]

        self.assertEqual(llm._first_text(Msg()), "real text")

    def test_falls_back_to_historical_access_when_no_str_text(self):
        # No block exposes a *str* `.text` (here `.text` is None on every block),
        # so the loop finds nothing and the function falls back to the historical
        # `msg.content[0].text` access rather than silently returning "".
        sentinel = object()

        class Blk:
            text = None        # non-str → loop skips it

        class Msg:
            # content[0].text is the historical fallback value.
            content = [Blk()]

        Msg.content[0].text = sentinel
        # With text re-set to a sentinel object (still non-str) the loop skips
        # it, and the fallback returns content[0].text verbatim.
        self.assertIs(llm._first_text(Msg()), sentinel)


class ClientFactoryTests(unittest.TestCase):
    def test_client_lazy_imports_and_constructs(self):
        # Inject a fake `anthropic` module so _client() imports + constructs it
        # without needing the real SDK or an API key. Asserts the timeout is
        # forwarded to the Anthropic() constructor. Restores sys.modules after.
        captured = {}

        class FakeAnthropic:
            def __init__(self, timeout=None):
                captured["timeout"] = timeout

        fake_mod = types.ModuleType("anthropic")
        fake_mod.Anthropic = FakeAnthropic
        with mock.patch.dict(sys.modules, {"anthropic": fake_mod}):
            client = llm._client(12.5)
        self.assertIsInstance(client, FakeAnthropic)
        self.assertEqual(captured["timeout"], 12.5)


class LogCacheUsageTests(unittest.TestCase):
    def test_logs_and_records_usage_fields(self):
        class _U:
            input_tokens = 42
            output_tokens = 7
            cache_read_input_tokens = 19000
            cache_creation_input_tokens = 0
        llm.last_usage.clear()
        llm._log_cache_usage(_U())
        self.assertEqual(llm.last_usage,
                         {"cache_read": 19000, "cache_creation": 0,
                          "input": 42, "output": 7})

    def test_none_and_malformed_usage_are_harmless(self):
        llm.last_usage.clear()
        llm._log_cache_usage(None)          # no-op
        self.assertEqual(llm.last_usage, {})

        class _Weird:
            # attribute access raising must not escape (telemetry only)
            def __getattr__(self, name):
                raise RuntimeError("boom")
        llm._log_cache_usage(_Weird())
        self.assertEqual(llm.last_usage, {})

    def test_missing_cache_fields_default_to_zero(self):
        class _U:
            input_tokens = 5
            output_tokens = 3
            # no cache_* attrs at all (older SDK shapes)
        llm.last_usage.clear()
        llm._log_cache_usage(_U())
        self.assertEqual(llm.last_usage["cache_read"], 0)
        self.assertEqual(llm.last_usage["cache_creation"], 0)


if __name__ == "__main__":
    unittest.main()
