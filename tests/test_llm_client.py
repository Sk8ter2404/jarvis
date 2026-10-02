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


class ResponseTextTests(unittest.TestCase):
    def test_skips_leading_non_text_block(self):
        class ToolBlock:
            pass

        class Msg:
            content = [ToolBlock(), _FakeBlock("real text")]

        self.assertEqual(llm.response_text(Msg()), "real text")

    def test_no_str_text_raises_instead_of_returning_garbage(self):
        # No block exposes a *str* `.text`. The old _first_text fell back to
        # `msg.content[0].text` and handed that object to the caller; on a
        # thinking model block 0 is a thinking block, so that path either
        # crashed with AttributeError or spoke nothing. It now raises the typed
        # CloudEmptyReplyError (a RuntimeError) so the caller's existing
        # cloud-failure fallback runs. 2026-10-01.
        class Blk:
            text = None        # non-str → skipped

        class Msg:
            content = [Blk()]

        with self.assertRaises(llm.CloudEmptyReplyError):
            llm.response_text(Msg())


class ClientFactoryTests(unittest.TestCase):
    def test_client_lazy_imports_and_constructs(self):
        # Inject a fake `anthropic` module so _client() imports + constructs it
        # without needing the real SDK or an API key. Asserts the timeout is
        # forwarded to the Anthropic() constructor. Restores sys.modules after.
        captured = {}

        class FakeAnthropic:
            def __init__(self, timeout=None, max_retries=None):
                captured["timeout"] = timeout
                captured["max_retries"] = max_retries

        fake_mod = types.ModuleType("anthropic")
        fake_mod.Anthropic = FakeAnthropic
        with mock.patch.dict(sys.modules, {"anthropic": fake_mod}):
            client = llm._client(12.5)
        self.assertIsInstance(client, FakeAnthropic)
        self.assertEqual(captured["timeout"], 12.5)
        # `timeout` is PER ATTEMPT and the SDK retries twice by default, so an
        # unpinned client turns a 30 s bound into a ~92 s one on the voice
        # thread. Pin it explicitly. 2026-07-14 audit.
        self.assertEqual(captured["max_retries"], llm.DEFAULT_MAX_RETRIES)

    def test_client_retry_budget_is_pinned_small(self):
        self.assertLessEqual(llm.DEFAULT_MAX_RETRIES, 1)


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



class _Usage:
    def __init__(self, inp=0, out=0, read=0, write=0):
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_read_input_tokens = read
        self.cache_creation_input_tokens = write


class _UsageMsg(_FakeMsg):
    def __init__(self, text, usage):
        super().__init__(text)
        self.usage = usage


class _UsageStream(_FakeStream):
    def __init__(self, chunks, usage):
        super().__init__(chunks)
        self._usage = usage

    def get_final_message(self):
        return _UsageMsg("".join(self.text_stream), self._usage)


class _UsageMessages:
    def __init__(self, usage):
        self._usage = usage

    def create(self, **kwargs):
        return _UsageMsg("hello sir", self._usage)

    def stream(self, **kwargs):
        return _UsageStream(["Hello", " sir"], self._usage)


class _UsageClient:
    def __init__(self, usage):
        self.messages = _UsageMessages(usage)


class SessionUsageTests(unittest.TestCase):
    """session_usage: the per-model token tally running_costs prices."""

    def setUp(self):
        p = mock.patch.object(llm, "session_usage", {})
        p.start()
        self.addCleanup(p.stop)

    def _create(self, model, usage):
        return llm.create_message(_UsageClient(usage), model=model,
                                  max_tokens=10, messages=[])

    def test_create_message_tallies_per_base_model(self):
        self._create("claude-sonnet-5-5-20261001", _Usage(100, 20, 5000, 300))
        self._create("claude-sonnet-5-5", _Usage(50, 10))
        self._create("claude-haiku-4-5", _Usage(7, 3))
        self.assertEqual(llm.session_usage_snapshot(), {
            "claude-sonnet-5-5": {"calls": 2, "input": 150, "output": 30,
                                  "cache_read": 5000, "cache_write": 300},
            "claude-haiku-4-5": {"calls": 1, "input": 7, "output": 3,
                                 "cache_read": 0, "cache_write": 0},
        })

    def test_complete_counts_each_reply_once(self):
        with mock.patch.object(llm, "_client",
                               return_value=_UsageClient(_Usage(40, 4))):
            llm.complete(model="claude-haiku-4-5", messages=[])
        row = llm.session_usage_snapshot()["claude-haiku-4-5"]
        self.assertEqual((row["calls"], row["input"], row["output"]),
                         (1, 40, 4))

    def test_stream_text_counts_the_final_message_once(self):
        with mock.patch.object(llm, "_client",
                               return_value=_UsageClient(_Usage(60, 6, 900))):
            out = llm.stream_text(model="claude-haiku-4-5", messages=[])
        self.assertEqual(out, "Hello sir")
        row = llm.session_usage_snapshot()["claude-haiku-4-5"]
        self.assertEqual((row["calls"], row["input"], row["cache_read"]),
                         (1, 60, 900))

    def test_replies_without_token_counts_record_nothing(self):
        cap = {}
        with _patch(cap):
            llm.complete(model="m", messages=[])         # no usage block
            llm.stream_text(model="m", messages=[])      # no final message
        llm.create_message(_FakeClient(cap, []), model="m", messages=[])
        self._create("m", mock.MagicMock())               # not integers
        self._create("m", _Usage())                       # all zero
        self.assertEqual(llm.session_usage_snapshot(), {})

    def test_snapshot_is_a_copy(self):
        self._create("claude-haiku-4-5", _Usage(1, 1))
        snap = llm.session_usage_snapshot()
        snap["claude-haiku-4-5"]["calls"] = 99
        self.assertEqual(
            llm.session_usage_snapshot()["claude-haiku-4-5"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
