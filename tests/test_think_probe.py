"""v3 W16: the Ollama `think` parameter from the model's own capabilities.

core.ollama_opts.model_can_think asks /api/show once per (server, model); the
monolith's _local_think_param uses it only for families outside its name table.
"""
import unittest

from core import ollama_opts as oo


class _Resp:
    def __init__(self, body=None, status=200):
        self._body = body
        self.status_code = status
        self.ok = 200 <= status < 300

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Poster:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append((url, json, timeout))
        if self.exc:
            raise self.exc
        return self.resp


class ModelCanThinkTests(unittest.TestCase):
    def setUp(self):
        oo._think_probe_cache.clear()

    def test_thinking_capability_true(self):
        p = _Poster(_Resp({"capabilities": ["completion", "vision", "tools", "thinking"]}))
        self.assertIs(oo.model_can_think("bonsai-2:27b", "http://127.0.0.1:11434", post=p), True)
        self.assertEqual(p.calls[0][0], "http://127.0.0.1:11434/api/show")
        self.assertEqual(p.calls[0][1], {"model": "bonsai-2:27b"})

    def test_no_thinking_capability_false(self):
        p = _Poster(_Resp({"capabilities": ["completion", "tools"]}))
        self.assertIs(oo.model_can_think("llama3.1:8b", "http://127.0.0.1:11434", post=p), False)

    def test_probe_failures_are_none(self):
        base = "http://127.0.0.1:11434"
        for p in (_Poster(exc=OSError("down")),
                  _Poster(_Resp({}, status=404)),
                  _Poster(_Resp(ValueError("junk"))),
                  _Poster(_Resp({"capabilities": "thinking"})),
                  _Poster(_Resp({"details": {}}))):
            oo._think_probe_cache.clear()
            self.assertIsNone(oo.model_can_think("m:1b", base, post=p))

    def test_unusable_inputs_never_probe(self):
        p = _Poster(_Resp({"capabilities": ["thinking"]}))
        for model, url in (("", "http://127.0.0.1:11434"), (None, "http://x"),
                           ("m", ""), ("m", None)):
            self.assertIsNone(oo.model_can_think(model, url, post=p))
        self.assertEqual(p.calls, [])

    def test_answer_cached_one_probe_per_model(self):
        p = _Poster(_Resp({"capabilities": ["thinking"]}))
        for _ in range(5):
            self.assertIs(oo.model_can_think("m:1b", "http://127.0.0.1:11434", post=p), True)
        self.assertEqual(len(p.calls), 1)
        oo.model_can_think("m:2b", "http://127.0.0.1:11434", post=p)
        oo.model_can_think("m:1b", "http://10.0.0.5:11434", post=p)
        self.assertEqual(len(p.calls), 3)

    def test_failure_retried_only_after_the_window(self):
        t = [100.0]
        p = _Poster(exc=OSError("down"))
        clock = lambda: t[0]
        for _ in range(3):
            oo.model_can_think("m:1b", "http://127.0.0.1:11434", post=p, clock=clock)
        self.assertEqual(len(p.calls), 1)
        t[0] += oo.THINK_PROBE_RETRY_S + 1
        p.exc, p.resp = None, _Resp({"capabilities": ["thinking"]})
        self.assertIs(oo.model_can_think("m:1b", "http://127.0.0.1:11434", post=p, clock=clock), True)
        self.assertEqual(len(p.calls), 2)

    def test_localhost_spelled_loopback_ip_and_timeout_from_probe_timeout(self):
        p = _Poster(_Resp({"capabilities": []}))
        oo.model_can_think("m:1b", "http://localhost:11434/", post=p)
        self.assertEqual(p.calls[0][0], "http://127.0.0.1:11434/api/show")
        self.assertEqual(p.calls[0][2], oo.probe_timeout("http://127.0.0.1:11434"))
        p2 = _Poster(_Resp({"capabilities": []}))
        oo.model_can_think("m:1b", "http://100.1.2.3:11434", post=p2)
        self.assertEqual(p2.calls[0][2], oo.PROBE_TIMEOUT_REMOTE_S)


if __name__ == "__main__":
    unittest.main()
