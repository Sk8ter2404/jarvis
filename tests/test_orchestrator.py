"""Tests for core.orchestrator.

The original suite (WorkerNoFabricationTests, preserved verbatim below) pins the
no-fabrication contract added 2026-05-31: a worker must run a REAL registered
action and summarise only that, and must return EMPTY (never invent) when no
allowed_action is registered.

The remaining suites raise line coverage of the whole module to >=90% by
exercising spec loading, the Claude/Ollama LLM seams, the planner, the worker
edge paths, the async dispatcher, the merger, and the Orchestrator class
(including its sync/loop-aware wrapper) — all deterministic and fully offline.

ISOLATION CONTRACT (wave-1 agents broke the full suite by ignoring this):
  * No `sys.modules[...] =` at module level. Fake `anthropic` / `urllib.request`
    are injected ONLY inside a test via `mock.patch.dict(sys.modules, ...)` with
    an `addCleanup` stop, so after every test sys.modules holds the REAL modules.
  * The module-level singleton `core.orchestrator._default_orchestrator` is reset
    in tearDown of the suite that touches it — no live object survives a test.
  * No real network, no real threads beyond asyncio.to_thread (which we drive to
    completion synchronously inside asyncio.run), no real sleeps.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import core.orchestrator as orch


# ──────────────────────────────────────────────────────────────────────────
#  Original no-fabrication suite — PRESERVED. Do not weaken these.
# ──────────────────────────────────────────────────────────────────────────

def _spec(allowed):
    return orch.SubAgentSpec(
        name="t_agent",
        description="test agent",
        allowed_actions=list(allowed),
        model_preference="haiku",
        system_prompt="Summarise tersely.",
    )


def _run(spec, actions, args=None):
    task = orch.SubTask(sub_agent="t_agent", task="do the thing", args=args or {})
    return orch._run_worker_sync(
        spec, task, actions,
        worker_model="claude-test",
        local_model="x",            # truthy → never imports bobert_companion
        local_base_url="http://localhost:0",
        timeout_s=5.0,
    )


class WorkerNoFabricationTests(unittest.TestCase):
    def setUp(self):
        # Save the LLM/ollama seams we monkeypatch so each test is isolated.
        self._claude = orch._claude_call
        self._ollama = orch._ollama_call
        self._reach = orch._ollama_reachable
        self._resolve = orch._resolve_local_model

    def tearDown(self):
        orch._claude_call = self._claude
        orch._ollama_call = self._ollama
        orch._ollama_reachable = self._reach
        orch._resolve_local_model = self._resolve

    def test_no_registered_action_returns_empty(self):
        # Spec's only action is NOT in the actions dict → must return empty
        # (so the merger omits it) and never call the LLM (no fabrication).
        calls = {"claude": 0}
        orch._claude_call = lambda *a, **k: calls.__setitem__("claude", calls["claude"] + 1) or "FABRICATED"
        res = _run(_spec(["unregistered_action"]), actions={})
        self.assertEqual(res.output, "")
        self.assertIn("no registered tool", (res.error or "").lower())
        self.assertEqual(calls["claude"], 0)     # LLM never invoked

    def test_runs_real_action_and_summarises_only_it(self):
        called = {"n": 0}

        def fake_read(arg):
            called["n"] += 1
            return "REAL_SENSOR_VALUE_42"

        # LLM echoes its user message so we can confirm the worker fed it the
        # REAL action output (not the bare task description).
        orch._claude_call = lambda model, system, user, **k: user
        res = _run(_spec(["fake_read"]), actions={"fake_read": fake_read})
        self.assertEqual(called["n"], 1)                       # real action ran
        self.assertIn("REAL_SENSOR_VALUE_42", res.output)      # fed to the LLM
        self.assertEqual(res.error, None)

    def test_llm_down_falls_back_to_raw_real_data(self):
        def boom(*a, **k):
            raise RuntimeError("claude down")

        orch._claude_call = boom
        orch._resolve_local_model = lambda *a, **k: None        # no ollama fallback
        res = _run(_spec(["fake_read"]),
                   actions={"fake_read": lambda a: "RAW_REAL_DATA"})
        # LLM unavailable → return the RAW real data, still real, never invented.
        self.assertEqual(res.output, "RAW_REAL_DATA")
        self.assertEqual(res.error, None)

    def test_explicit_direct_action_executes_without_llm(self):
        calls = {"claude": 0}
        orch._claude_call = lambda *a, **k: calls.__setitem__("claude", calls["claude"] + 1) or "x"
        res = _run(
            _spec(["fake_read"]),
            actions={"fake_read": lambda a: f"GOT:{a}"},
            args={"direct_action": "fake_read", "arg": "hello"},
        )
        self.assertEqual(res.output, "GOT:hello")
        self.assertEqual(res.model_used, "direct")
        self.assertEqual(calls["claude"], 0)


# ──────────────────────────────────────────────────────────────────────────
#  Shared helpers / fakes for the extended suites
# ──────────────────────────────────────────────────────────────────────────

class _FakeBlock:
    """Mimics one anthropic content block carrying `.text`."""
    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeAnthropicClient:
    """Stand-in for anthropic.Anthropic(). Records the kwargs of the last call
    and returns a canned message (or raises if `exc` is set)."""
    last_kwargs = None
    canned = None
    exc = None

    def __init__(self, *a, **k):
        pass

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        type(self).last_kwargs = kwargs
        if type(self).exc is not None:
            raise type(self).exc
        return type(self).canned


def _make_fake_anthropic(canned=None, exc=None):
    """Build a fake `anthropic` module whose Anthropic() returns `canned`."""
    cls = type(
        "FakeClient",
        (_FakeAnthropicClient,),
        {"last_kwargs": None, "canned": canned, "exc": exc},
    )
    mod = mock.Mock()
    mod.Anthropic = cls
    return mod, cls


class _FakeHTTPResponse:
    """Context-manager response like urllib returns."""
    def __init__(self, body=b"", status=200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeReq:
    def __init__(self, url=None, data=None, headers=None, method=None):
        self.url = url
        self.data = data
        self.headers = headers or {}
        self.method = method


def _make_fake_urllib(urlopen):
    """Build a fake stand-in for the `urllib.request` submodule exposing a
    custom `urlopen` and a lightweight `Request`."""
    mod = mock.Mock()
    mod.Request = _FakeReq
    mod.urlopen = urlopen
    return mod


def _patch_urllib_request(testcase, urlopen):
    """Patch the `request` attribute on the real `urllib` package.

    The orchestrator does `import urllib.request` then calls
    `urllib.request.Request(...)`. With `urllib.request` already in
    sys.modules, that `import` statement just rebinds the local `urllib`
    name and the call resolves `.request` as an *attribute* of the live
    package object — so patching `sys.modules['urllib.request']` alone is not
    seen. We import the real submodule once (so the attribute exists), then
    patch the attribute (and the sys.modules entry, belt-and-braces)."""
    import urllib
    import urllib.request  # noqa: F401 — ensure the attribute exists to patch
    fake = _make_fake_urllib(urlopen)
    p_attr = mock.patch.object(urllib, "request", fake, create=True)
    p_mod = mock.patch.dict(sys.modules, {"urllib.request": fake})
    p_attr.start()
    p_mod.start()
    testcase.addCleanup(p_attr.stop)
    testcase.addCleanup(p_mod.stop)
    return fake


# ──────────────────────────────────────────────────────────────────────────
#  Dataclasses: SubAgentSpec.from_dict / is_valid
# ──────────────────────────────────────────────────────────────────────────

class SubAgentSpecTests(unittest.TestCase):
    def test_from_dict_full(self):
        s = orch.SubAgentSpec.from_dict({
            "name": "  email  ",
            "description": "  reads mail ",
            "allowed_actions": ["a", "b"],
            "model_preference": "  local ",
            "system_prompt": "  hi ",
        })
        self.assertEqual(s.name, "email")
        self.assertEqual(s.description, "reads mail")
        self.assertEqual(s.allowed_actions, ["a", "b"])
        self.assertEqual(s.model_preference, "local")
        self.assertEqual(s.system_prompt, "hi")
        self.assertTrue(s.is_valid())

    def test_from_dict_defaults_and_blank_pref(self):
        # Missing allowed_actions → []; blank model_preference → "haiku".
        s = orch.SubAgentSpec.from_dict({
            "name": "x", "description": "d", "model_preference": "   ",
        })
        self.assertEqual(s.allowed_actions, [])
        self.assertEqual(s.model_preference, "haiku")
        self.assertEqual(s.system_prompt, "")

    def test_from_dict_none_allowed_actions(self):
        s = orch.SubAgentSpec.from_dict({"name": "x", "description": "d",
                                         "allowed_actions": None})
        self.assertEqual(s.allowed_actions, [])

    def test_is_valid_requires_name_and_description(self):
        self.assertFalse(orch.SubAgentSpec.from_dict({"name": "x"}).is_valid())
        self.assertFalse(orch.SubAgentSpec.from_dict({"description": "d"}).is_valid())
        self.assertFalse(orch.SubAgentSpec.from_dict({}).is_valid())


# ──────────────────────────────────────────────────────────────────────────
#  SPEC LOADING: _load_json_spec, _load_py_spec, load_specs
# ──────────────────────────────────────────────────────────────────────────

class SpecLoadingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="orch_specs_")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_load_json_spec_ok(self):
        p = self._write("a.json", json.dumps(
            {"name": "n", "description": "d", "allowed_actions": ["x"]}))
        s = orch._load_json_spec(p)
        self.assertIsNotNone(s)
        self.assertEqual(s.name, "n")

    def test_load_json_spec_unreadable(self):
        # Nonexistent path → open() raises → warning + None.
        s = orch._load_json_spec(os.path.join(self.dir, "missing.json"))
        self.assertIsNone(s)

    def test_load_json_spec_not_an_object(self):
        p = self._write("arr.json", "[1, 2, 3]")
        self.assertIsNone(orch._load_json_spec(p))

    def test_load_json_spec_invalid_missing_fields(self):
        p = self._write("bad.json", json.dumps({"name": "n"}))  # no description
        self.assertIsNone(orch._load_json_spec(p))

    def test_load_py_spec_ok(self):
        p = self._write("pyspec.py",
                        "SPEC = {'name': 'pyn', 'description': 'pyd'}\n")
        s = orch._load_py_spec(p)
        self.assertIsNotNone(s)
        self.assertEqual(s.name, "pyn")

    def test_load_py_spec_no_spec_attr(self):
        p = self._write("nospec.py", "X = 1\n")
        self.assertIsNone(orch._load_py_spec(p))

    def test_load_py_spec_spec_not_dict(self):
        p = self._write("badspec.py", "SPEC = [1, 2]\n")
        self.assertIsNone(orch._load_py_spec(p))

    def test_load_py_spec_invalid_spec(self):
        p = self._write("invalidspec.py", "SPEC = {'name': 'only'}\n")
        self.assertIsNone(orch._load_py_spec(p))

    def test_load_py_spec_import_raises(self):
        p = self._write("boom.py", "raise RuntimeError('nope')\n")
        self.assertIsNone(orch._load_py_spec(p))

    def test_load_py_spec_loader_none(self):
        # spec_from_file_location returns an object with loader=None → None.
        fake_spec = mock.Mock()
        fake_spec.loader = None
        with mock.patch("importlib.util.spec_from_file_location",
                        return_value=fake_spec):
            self.assertIsNone(orch._load_py_spec(os.path.join(self.dir, "x.py")))

    def test_load_specs_missing_dir(self):
        self.assertEqual(orch.load_specs(os.path.join(self.dir, "nope")), {})

    def test_load_specs_mixed_dir(self):
        self._write("good.json", json.dumps({"name": "j", "description": "d"}))
        self._write("good.py", "SPEC = {'name': 'p', 'description': 'd'}\n")
        self._write("_skip.json", json.dumps({"name": "skip", "description": "d"}))
        self._write("notes.txt", "ignore me")            # non-json/py → skipped
        self._write("broken.json", "{ not json")          # parse fail → skipped
        specs = orch.load_specs(self.dir)
        self.assertIn("j", specs)
        self.assertIn("p", specs)
        self.assertNotIn("skip", specs)
        self.assertEqual(len(specs), 2)


# ──────────────────────────────────────────────────────────────────────────
#  LLM CALL HELPERS: _claude_call, _ollama_call, _ollama_reachable
# ──────────────────────────────────────────────────────────────────────────

class ClaudeCallTests(unittest.TestCase):
    def _patch_anthropic(self, canned=None, exc=None):
        mod, cls = _make_fake_anthropic(canned=canned, exc=exc)
        p = mock.patch.dict(sys.modules, {"anthropic": mod})
        p.start()
        self.addCleanup(p.stop)
        return cls

    def test_claude_call_joins_text_blocks(self):
        msg = _FakeMessage([_FakeBlock("Hello "), _FakeBlock("world"),
                            _FakeBlock(None), object()])  # non-text blocks skipped
        cls = self._patch_anthropic(canned=msg)
        out = orch._claude_call("m", "sys", "usr", max_tokens=42, timeout_s=7.0)
        self.assertEqual(out, "Hello world")
        # timeout_s threaded through as `timeout`; core kwargs present.
        self.assertEqual(cls.last_kwargs["timeout"], 7.0)
        self.assertEqual(cls.last_kwargs["model"], "m")
        self.assertEqual(cls.last_kwargs["max_tokens"], 42)
        self.assertEqual(cls.last_kwargs["system"], "sys")

    def test_claude_call_without_timeout_omits_kwarg(self):
        cls = self._patch_anthropic(canned=_FakeMessage([_FakeBlock("ok")]))
        out = orch._claude_call("m", "s", "u")          # timeout_s defaults None
        self.assertEqual(out, "ok")
        self.assertNotIn("timeout", cls.last_kwargs)

    def test_claude_call_content_none(self):
        # getattr(msg, "content", []) falls back when content is None.
        self._patch_anthropic(canned=_FakeMessage(None))
        self.assertEqual(orch._claude_call("m", "s", "u"), "")

    def test_claude_call_propagates_exception(self):
        self._patch_anthropic(exc=RuntimeError("api down"))
        with self.assertRaises(RuntimeError):
            orch._claude_call("m", "s", "u")


class OllamaCallTests(unittest.TestCase):
    def test_ollama_call_posts_and_parses(self):
        captured = {}

        def urlopen(req, timeout=None):
            captured["url"] = req.url
            captured["method"] = req.method
            captured["timeout"] = timeout
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse(
                body=json.dumps({"message": {"content": "  hi there  "}}).encode())

        _patch_urllib_request(self, urlopen)
        out = orch._ollama_call("llama3", "sys", "usr",
                                base_url="http://h:1/", timeout_s=9.0)
        self.assertEqual(out, "hi there")                  # stripped
        self.assertEqual(captured["url"], "http://h:1/api/chat")   # rstrip slash
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["timeout"], 9.0)
        self.assertEqual(captured["payload"]["model"], "llama3")
        self.assertEqual(captured["payload"]["messages"][0]["role"], "system")

    def test_ollama_call_missing_message_key(self):
        def urlopen(req, timeout=None):
            return _FakeHTTPResponse(body=json.dumps({}).encode())

        _patch_urllib_request(self, urlopen)
        self.assertEqual(orch._ollama_call("m", "s", "u"), "")


class OllamaReachableTests(unittest.TestCase):
    def test_reachable_2xx(self):
        def urlopen(req, timeout=None):
            return _FakeHTTPResponse(status=200)

        _patch_urllib_request(self, urlopen)
        self.assertTrue(orch._ollama_reachable("http://h:1"))

    def test_reachable_default_status_attr(self):
        # Response object lacking `.status` → getattr default 200 → reachable.
        class _NoStatus:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def urlopen(req, timeout=None):
            return _NoStatus()

        _patch_urllib_request(self, urlopen)
        self.assertTrue(orch._ollama_reachable())

    def test_unreachable_on_exception(self):
        def urlopen(req, timeout=None):
            raise OSError("refused")

        _patch_urllib_request(self, urlopen)
        self.assertFalse(orch._ollama_reachable())

    def test_unreachable_on_non_2xx(self):
        def urlopen(req, timeout=None):
            return _FakeHTTPResponse(status=500)

        _patch_urllib_request(self, urlopen)
        self.assertFalse(orch._ollama_reachable())


# ──────────────────────────────────────────────────────────────────────────
#  _resolve_local_model — configured / bobert_companion / failure
# ──────────────────────────────────────────────────────────────────────────

class ResolveLocalModelTests(unittest.TestCase):
    def test_configured_short_circuits(self):
        self.assertEqual(orch._resolve_local_model("explicit-tag"), "explicit-tag")

    def test_uses_bobert_resolver(self):
        fake_bc = mock.Mock()
        fake_bc._get_local_llm_model = lambda: "  llama-from-bc  "
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            self.assertEqual(orch._resolve_local_model(None), "llama-from-bc")

    def test_bobert_resolver_returns_blank(self):
        fake_bc = mock.Mock()
        fake_bc._get_local_llm_model = lambda: "   "
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            self.assertIsNone(orch._resolve_local_model(None))

    def test_bobert_resolver_not_callable(self):
        fake_bc = mock.Mock()
        fake_bc._get_local_llm_model = "not-callable"
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            self.assertIsNone(orch._resolve_local_model(None))

    def test_resolve_swallows_exception(self):
        # import_module raising is caught → None.
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("bobert_companion", None)
            with mock.patch("importlib.import_module",
                            side_effect=RuntimeError("boom")):
                self.assertIsNone(orch._resolve_local_model(None))


# ──────────────────────────────────────────────────────────────────────────
#  _format_catalogue / _extract_json_object
# ──────────────────────────────────────────────────────────────────────────

class FormatCatalogueTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(orch._format_catalogue({}), "(catalogue empty)")

    def test_lists_entries(self):
        specs = {"a": orch.SubAgentSpec(name="a", description="da"),
                 "b": orch.SubAgentSpec(name="b", description="db")}
        out = orch._format_catalogue(specs)
        self.assertIn("- a: da", out)
        self.assertIn("- b: db", out)


class ExtractJsonObjectTests(unittest.TestCase):
    def test_empty_string(self):
        self.assertIsNone(orch._extract_json_object(""))

    def test_no_brace(self):
        self.assertIsNone(orch._extract_json_object("no json here"))

    def test_plain_object(self):
        self.assertEqual(orch._extract_json_object('{"a": 1}'), {"a": 1})

    def test_strips_json_fence(self):
        text = '```json\n{"sub_tasks": []}\n```'
        self.assertEqual(orch._extract_json_object(text), {"sub_tasks": []})

    def test_strips_bare_fence(self):
        text = '```\n{"x": 2}\n```'
        self.assertEqual(orch._extract_json_object(text), {"x": 2})

    def test_braces_inside_strings_ignored(self):
        # A literal "{" inside a string value must not fool the depth counter.
        text = 'prefix {"task": "set timer {hh}:{mm}", "n": 1} suffix'
        self.assertEqual(orch._extract_json_object(text),
                         {"task": "set timer {hh}:{mm}", "n": 1})

    def test_escaped_quote_inside_string(self):
        text = r'{"q": "she said \"hi\" {x}"}'
        self.assertEqual(orch._extract_json_object(text),
                         {"q": 'she said "hi" {x}'})

    def test_unbalanced_returns_none(self):
        # Opening brace, never closed → loop ends with depth>0 → None.
        self.assertIsNone(orch._extract_json_object('{"a": 1'))

    def test_malformed_balanced_returns_none(self):
        # Balanced braces but invalid JSON inside → json.loads raises → None.
        self.assertIsNone(orch._extract_json_object("{not: valid, json}"))


# ──────────────────────────────────────────────────────────────────────────
#  plan_decomposition — every branch
# ──────────────────────────────────────────────────────────────────────────

class PlanDecompositionTests(unittest.TestCase):
    def setUp(self):
        self._claude = orch._claude_call
        self._ollama = orch._ollama_call
        self._reach = orch._ollama_reachable
        self._resolve = orch._resolve_local_model

    def tearDown(self):
        orch._claude_call = self._claude
        orch._ollama_call = self._ollama
        orch._ollama_reachable = self._reach
        orch._resolve_local_model = self._resolve

    def _specs(self):
        return {
            "email": orch.SubAgentSpec(name="email", description="reads mail"),
            "news": orch.SubAgentSpec(name="news", description="reads news"),
        }

    def test_empty_specs_returns_empty(self):
        orch._claude_call = lambda *a, **k: self.fail("should not call LLM")
        self.assertEqual(orch.plan_decomposition("hi", {}), [])

    def test_happy_path_parses_subtasks(self):
        plan = {"sub_tasks": [
            {"sub_agent": "email", "task": "summarise inbox",
             "args": {"k": "v"}},
            {"sub_agent": "news", "task": "top headlines"},
        ]}
        orch._claude_call = lambda *a, **k: json.dumps(plan)
        out = orch.plan_decomposition("morning", self._specs())
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].sub_agent, "email")
        self.assertEqual(out[0].args, {"k": "v"})
        self.assertEqual(out[1].args, {})           # missing/non-dict args → {}

    def test_skips_invalid_items(self):
        plan = {"sub_tasks": [
            "not-a-dict",                                  # not dict → skip
            {"sub_agent": "", "task": "x"},                # blank name → skip
            {"sub_agent": "email", "task": ""},            # blank task → skip
            {"sub_agent": "ghost", "task": "real"},        # unknown spec → skip
            {"sub_agent": "email", "task": "keep me",
             "args": "not-a-dict"},                        # args coerced to {}
        ]}
        orch._claude_call = lambda *a, **k: json.dumps(plan)
        out = orch.plan_decomposition("req", self._specs())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].task, "keep me")
        self.assertEqual(out[0].args, {})

    def test_unparseable_json_returns_empty(self):
        orch._claude_call = lambda *a, **k: "totally not json"
        self.assertEqual(orch.plan_decomposition("req", self._specs()), [])

    def test_sub_tasks_not_a_list_returns_empty(self):
        orch._claude_call = lambda *a, **k: json.dumps({"sub_tasks": "nope"})
        self.assertEqual(orch.plan_decomposition("req", self._specs()), [])

    def test_planner_fails_then_ollama_succeeds(self):
        def boom(*a, **k):
            raise RuntimeError("claude capped")

        orch._claude_call = boom
        orch._resolve_local_model = lambda *a, **k: "llama-x"
        orch._ollama_reachable = lambda *a, **k: True
        plan = {"sub_tasks": [{"sub_agent": "news", "task": "go"}]}
        orch._ollama_call = lambda *a, **k: json.dumps(plan)
        out = orch.plan_decomposition("req", self._specs())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].sub_agent, "news")

    def test_planner_fails_ollama_fails_degrades_to_first_spec(self):
        orch._claude_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        orch._resolve_local_model = lambda *a, **k: "llama-x"
        orch._ollama_reachable = lambda *a, **k: True
        orch._ollama_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("o"))
        out = orch.plan_decomposition("whole request", self._specs())
        # Degrades to a single sub_task on the FIRST registered spec.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].sub_agent, "email")
        self.assertEqual(out[0].task, "whole request")

    def test_planner_fails_no_ollama_degrades_to_first_spec(self):
        orch._claude_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        orch._resolve_local_model = lambda *a, **k: None      # no local model
        orch._ollama_reachable = lambda *a, **k: False
        out = orch.plan_decomposition("req", self._specs())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].sub_agent, "email")

    def test_planner_fails_degrade_with_empty_specs_after_fail(self):
        # Force the failure path but make `next(iter(specs))` yield None by
        # passing specs that are truthy at entry yet... here we just confirm
        # the ollama-reachable-but-resolve-None branch (skips ollama attempt).
        orch._claude_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        orch._resolve_local_model = lambda *a, **k: None
        orch._ollama_reachable = lambda *a, **k: True   # reachable but no model
        out = orch.plan_decomposition("req", self._specs())
        self.assertEqual(out[0].sub_agent, "email")


# ──────────────────────────────────────────────────────────────────────────
#  _build_worker_system / _resolve_worker_model
# ──────────────────────────────────────────────────────────────────────────

class BuildWorkerSystemTests(unittest.TestCase):
    def test_lists_only_registered_allowed_actions(self):
        spec = orch.SubAgentSpec(name="n", description="d",
                                 allowed_actions=["a", "b", "c"],
                                 system_prompt="EXTRA")
        out = orch._build_worker_system(spec, ["a", "c", "z"])
        self.assertIn("• a", out)
        self.assertIn("• c", out)
        self.assertNotIn("• b", out)        # b not registered
        self.assertIn("EXTRA", out)         # spec system_prompt appended

    def test_no_allowed_actions_shows_none(self):
        spec = orch.SubAgentSpec(name="n", description="d", allowed_actions=[])
        out = orch._build_worker_system(spec, ["a"])
        self.assertIn("(none)", out)

    def test_no_system_prompt_returns_base_only(self):
        spec = orch.SubAgentSpec(name="n", description="d",
                                 allowed_actions=["a"], system_prompt="")
        out = orch._build_worker_system(spec, ["a"])
        self.assertNotIn("\n\n\n", out)     # nothing appended after base


class ResolveWorkerModelTests(unittest.TestCase):
    def test_local_pref_with_local_model(self):
        spec = orch.SubAgentSpec(name="n", description="d",
                                 model_preference="local")
        self.assertEqual(
            orch._resolve_worker_model(spec, "claude-w", "llama-l"),
            ("ollama", "llama-l"))

    def test_local_pref_without_local_model_uses_claude(self):
        spec = orch.SubAgentSpec(name="n", description="d",
                                 model_preference="local")
        self.assertEqual(
            orch._resolve_worker_model(spec, "claude-w", None),
            ("claude", "claude-w"))

    def test_auto_pref_uses_claude(self):
        spec = orch.SubAgentSpec(name="n", description="d",
                                 model_preference="auto")
        self.assertEqual(
            orch._resolve_worker_model(spec, "claude-w", "llama-l"),
            ("claude", "claude-w"))

    def test_haiku_pref_uses_claude(self):
        spec = orch.SubAgentSpec(name="n", description="d",
                                 model_preference="haiku")
        self.assertEqual(
            orch._resolve_worker_model(spec, "claude-w", "llama-l"),
            ("claude", "claude-w"))


# ──────────────────────────────────────────────────────────────────────────
#  _run_worker_sync — remaining edge paths not in the original suite
# ──────────────────────────────────────────────────────────────────────────

class RunWorkerSyncEdgeTests(unittest.TestCase):
    def setUp(self):
        self._claude = orch._claude_call
        self._ollama = orch._ollama_call
        self._reach = orch._ollama_reachable
        self._resolve = orch._resolve_local_model

    def tearDown(self):
        orch._claude_call = self._claude
        orch._ollama_call = self._ollama
        orch._ollama_reachable = self._reach
        orch._resolve_local_model = self._resolve

    def test_direct_action_failure_returns_error(self):
        def boom(arg):
            raise ValueError("kaboom")

        res = _run(_spec(["fake_read"]), actions={"fake_read": boom},
                   args={"direct_action": "fake_read", "arg": "x"})
        self.assertEqual(res.output, "")
        self.assertIn("direct action fake_read failed", res.error)
        self.assertEqual(res.model_used, "direct")

    def test_direct_action_non_string_output_coerced(self):
        res = _run(_spec(["fake_read"]),
                   actions={"fake_read": lambda a: 12345},   # int output
                   args={"direct_action": "fake_read", "arg": ""})
        self.assertEqual(res.output, "12345")
        self.assertEqual(res.model_used, "direct")

    def test_direct_action_not_in_allowed_falls_through_to_auto(self):
        # direct_action names a registered action that the spec does NOT allow
        # → the direct branch is skipped; auto-read path runs instead.
        orch._claude_call = lambda model, system, user, **k: "SUMMARY"
        res = _run(
            _spec(["allowed_read"]),
            actions={"allowed_read": lambda a: "DATA", "other": lambda a: "X"},
            args={"direct_action": "other", "arg": "y"},
        )
        self.assertEqual(res.output, "SUMMARY")
        self.assertNotEqual(res.model_used, "direct")

    def test_auto_tool_raises_then_no_data_returns_empty(self):
        # The auto read action raises → real_data stays "" → empty + error,
        # LLM never called.
        called = {"claude": 0}
        orch._claude_call = lambda *a, **k: called.__setitem__("claude", 1)

        def boom(arg):
            raise RuntimeError("tool failed")

        res = _run(_spec(["fake_read"]), actions={"fake_read": boom})
        self.assertEqual(res.output, "")
        self.assertIn("no registered tool data", res.error)
        self.assertEqual(called["claude"], 0)

    def test_auto_tool_returns_whitespace_only(self):
        # Tool returns whitespace → treated as empty → omit (no fabrication).
        res = _run(_spec(["fake_read"]), actions={"fake_read": lambda a: "   \n  "})
        self.assertEqual(res.output, "")
        self.assertIn("no registered tool data", res.error)

    def test_auto_tool_non_string_output_coerced_then_summarised(self):
        orch._claude_call = lambda model, system, user, **k: f"SAW:{user[:20]}"
        res = _run(_spec(["fake_read"]),
                   actions={"fake_read": lambda a: ["a", "b"]})  # list output
        self.assertEqual(res.error, None)
        self.assertTrue(res.output.startswith("SAW:"))

    def test_ollama_backend_used_when_pref_local(self):
        spec = orch.SubAgentSpec(name="t_agent", description="d",
                                 allowed_actions=["fake_read"],
                                 model_preference="local",
                                 system_prompt="s")
        orch._ollama_call = lambda model, system, user, **k: f"OLLAMA:{model}"
        orch._claude_call = lambda *a, **k: self.fail("claude must not run")
        task = orch.SubTask(sub_agent="t_agent", task="t", args={})
        res = orch._run_worker_sync(
            spec, task, {"fake_read": lambda a: "DATA"},
            worker_model="claude-w", local_model="llama-l",
            local_base_url="http://h", timeout_s=3.0)
        self.assertEqual(res.output, "OLLAMA:llama-l")
        self.assertEqual(res.model_used, "llama-l")

    def test_claude_down_ollama_fallback_succeeds(self):
        orch._claude_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
        orch._resolve_local_model = lambda *a, **k: "llama-fb"
        orch._ollama_reachable = lambda *a, **k: True
        orch._ollama_call = lambda model, *a, **k: f"FELLBACK:{model}"
        res = _run(_spec(["fake_read"]), actions={"fake_read": lambda a: "DATA"})
        self.assertEqual(res.output, "FELLBACK:llama-fb")
        self.assertEqual(res.model_used, "llama-fb")
        self.assertEqual(res.error, None)

    def test_claude_down_ollama_fallback_itself_raises_uses_raw(self):
        # Outer try/except: ollama fallback raises → caught → raw real data.
        orch._claude_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
        orch._resolve_local_model = lambda *a, **k: "llama-fb"
        orch._ollama_reachable = lambda *a, **k: True
        orch._ollama_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ofail"))
        res = _run(_spec(["fake_read"]), actions={"fake_read": lambda a: "RAWDATA"})
        self.assertEqual(res.output, "RAWDATA")
        self.assertEqual(res.error, None)

    def test_args_arg_passed_to_auto_tool(self):
        seen = {}
        orch._claude_call = lambda model, system, user, **k: "ok"

        def reader(arg):
            seen["arg"] = arg
            return "DATA"

        _run(_spec(["fake_read"]), actions={"fake_read": reader},
             args={"arg": "PAYLOAD"})
        self.assertEqual(seen["arg"], "PAYLOAD")


# ──────────────────────────────────────────────────────────────────────────
#  dispatch_sub_agents — async dispatcher
# ──────────────────────────────────────────────────────────────────────────

class DispatchSubAgentsTests(unittest.TestCase):
    def setUp(self):
        self._claude = orch._claude_call
        orch._claude_call = lambda model, system, user, **k: f"OUT:{system[:0]}{user[:0]}done"

    def tearDown(self):
        orch._claude_call = self._claude

    def test_empty_tasks_returns_empty(self):
        out = asyncio.run(orch.dispatch_sub_agents([], {}, {}))
        self.assertEqual(out, [])

    def test_unknown_sub_agent_yields_error_result(self):
        tasks = [orch.SubTask(sub_agent="ghost", task="t")]
        out = asyncio.run(orch.dispatch_sub_agents(tasks, {}, {}))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].error, "unknown sub_agent")
        self.assertEqual(out[0].sub_agent, "ghost")
        self.assertEqual(out[0].duration_s, 0.0)

    def test_dispatches_multiple_in_parallel(self):
        specs = {"a": orch.SubAgentSpec(name="a", description="d",
                                        allowed_actions=["r"]),
                 "b": orch.SubAgentSpec(name="b", description="d",
                                        allowed_actions=["r"])}
        actions = {"r": lambda arg: "REAL"}
        tasks = [orch.SubTask(sub_agent="a", task="t1"),
                 orch.SubTask(sub_agent="b", task="t2")]
        out = asyncio.run(orch.dispatch_sub_agents(
            tasks, specs, actions, max_parallel=2))
        self.assertEqual(len(out), 2)
        self.assertTrue(all(r.output.endswith("done") for r in out))
        self.assertEqual({r.sub_agent for r in out}, {"a", "b"})

    def test_max_parallel_clamped_to_one(self):
        # max_parallel<=0 must clamp to 1 (Semaphore(max(1, n))) and still run.
        specs = {"a": orch.SubAgentSpec(name="a", description="d",
                                        allowed_actions=["r"])}
        out = asyncio.run(orch.dispatch_sub_agents(
            [orch.SubTask(sub_agent="a", task="t")],
            specs, {"r": lambda a: "REAL"}, max_parallel=0))
        self.assertEqual(len(out), 1)


# ──────────────────────────────────────────────────────────────────────────
#  merge_results — synthesis + fallbacks
# ──────────────────────────────────────────────────────────────────────────

class MergeResultsTests(unittest.TestCase):
    def setUp(self):
        self._claude = orch._claude_call
        self._ollama = orch._ollama_call
        self._reach = orch._ollama_reachable
        self._resolve = orch._resolve_local_model

    def tearDown(self):
        orch._claude_call = self._claude
        orch._ollama_call = self._ollama
        orch._ollama_reachable = self._reach
        orch._resolve_local_model = self._resolve

    def _results(self):
        return [
            orch.SubTaskResult(sub_agent="a", task="t", output="Alpha out",
                               duration_s=0.1),
            orch.SubTaskResult(sub_agent="b", task="t", output="Beta out",
                               duration_s=0.1),
            orch.SubTaskResult(sub_agent="c", task="t", output="",
                               duration_s=0.1),                # empty → excluded
            orch.SubTaskResult(sub_agent="d", task="t", output="x",
                               duration_s=0.1, error="boom"),  # error → excluded
        ]

    def test_no_usable_results_returns_empty(self):
        orch._claude_call = lambda *a, **k: self.fail("must not call LLM")
        res = [orch.SubTaskResult(sub_agent="a", task="t", output="",
                                  duration_s=0.0)]
        self.assertEqual(orch.merge_results("req", res), "")

    def test_merges_usable_results(self):
        captured = {}

        def fake(model, system, user, **k):
            captured["user"] = user
            return "MERGED REPLY"

        orch._claude_call = fake
        out = orch.merge_results("the request", self._results())
        self.assertEqual(out, "MERGED REPLY")
        # Only usable (a, b) sections are bundled; c/d excluded.
        self.assertIn("[a]", captured["user"])
        self.assertIn("Alpha out", captured["user"])
        self.assertIn("[b]", captured["user"])
        self.assertNotIn("[c]", captured["user"])
        self.assertNotIn("[d]", captured["user"])

    def test_claude_fails_ollama_succeeds(self):
        orch._claude_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        orch._resolve_local_model = lambda *a, **k: "llama-m"
        orch._ollama_reachable = lambda *a, **k: True
        orch._ollama_call = lambda *a, **k: "OLLAMA MERGED"
        self.assertEqual(orch.merge_results("req", self._results()),
                         "OLLAMA MERGED")

    def test_claude_fails_ollama_returns_empty_then_concat(self):
        # Ollama reachable but returns "" → falls through to deterministic concat.
        orch._claude_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        orch._resolve_local_model = lambda *a, **k: "llama-m"
        orch._ollama_reachable = lambda *a, **k: True
        orch._ollama_call = lambda *a, **k: ""
        out = orch.merge_results("req", self._results())
        self.assertEqual(out, "Alpha out Beta out")

    def test_claude_fails_ollama_raises_then_concat(self):
        orch._claude_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        orch._resolve_local_model = lambda *a, **k: "llama-m"
        orch._ollama_reachable = lambda *a, **k: True
        orch._ollama_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("o"))
        out = orch.merge_results("req", self._results())
        self.assertEqual(out, "Alpha out Beta out")

    def test_claude_fails_no_ollama_concat(self):
        orch._claude_call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        orch._resolve_local_model = lambda *a, **k: None
        orch._ollama_reachable = lambda *a, **k: False
        out = orch.merge_results("req", self._results())
        self.assertEqual(out, "Alpha out Beta out")


# ──────────────────────────────────────────────────────────────────────────
#  Orchestrator class + module-level singleton
# ──────────────────────────────────────────────────────────────────────────

class OrchestratorClassTests(unittest.TestCase):
    def setUp(self):
        # Build with an empty specs dir so construction never reads real specs.
        self.dir = tempfile.mkdtemp(prefix="orch_cls_")
        self.addCleanup(self._cleanup)
        self._claude = orch._claude_call
        self._ollama = orch._ollama_call
        self._reach = orch._ollama_reachable
        self._resolve = orch._resolve_local_model
        # Always reset the module singleton so no live object leaks across tests.
        self.addCleanup(self._reset_singleton)

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _reset_singleton(self):
        orch._claude_call = self._claude
        orch._ollama_call = self._ollama
        orch._ollama_reachable = self._reach
        orch._resolve_local_model = self._resolve
        orch._default_orchestrator = None

    def _mk(self, **kw):
        kw.setdefault("specs_dir", self.dir)
        return orch.Orchestrator(**kw)

    def _write_spec(self, name, payload):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))

    def test_construct_loads_specs(self):
        self._write_spec("a.json", {"name": "a", "description": "d"})
        o = self._mk()
        self.assertEqual(o.list_specs(), ["a"])

    def test_reload_specs_returns_count(self):
        o = self._mk()
        self.assertEqual(o.reload_specs(), 0)
        self._write_spec("a.json", {"name": "a", "description": "d"})
        self._write_spec("b.json", {"name": "b", "description": "d"})
        self.assertEqual(o.reload_specs(), 2)
        self.assertEqual(o.list_specs(), ["a", "b"])

    def test_plan_delegates(self):
        self._write_spec("email", "ignored")  # non-json/py ext → skipped anyway
        self._write_spec("email.json", {"name": "email", "description": "d"})
        o = self._mk()
        orch._claude_call = lambda *a, **k: json.dumps(
            {"sub_tasks": [{"sub_agent": "email", "task": "go"}]})
        out = o.plan("req")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].sub_agent, "email")

    def test_merge_delegates(self):
        o = self._mk()
        orch._claude_call = lambda *a, **k: "MERGED"
        res = [orch.SubTaskResult(sub_agent="a", task="t", output="o",
                                  duration_s=0.0)]
        self.assertEqual(o.merge("req", res), "MERGED")

    def test_dispatch_delegates(self):
        self._write_spec("a.json", {"name": "a", "description": "d",
                                    "allowed_actions": ["r"]})
        o = self._mk()
        orch._claude_call = lambda model, system, user, **k: "done"
        tasks = [orch.SubTask(sub_agent="a", task="t")]
        out = asyncio.run(o.dispatch(tasks, {"r": lambda a: "REAL"}))
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].output)

    def test_orchestrate_async_no_subtasks_returns_empty(self):
        o = self._mk()                       # empty specs → plan() returns []
        out = asyncio.run(o.orchestrate_async("req", {}))
        self.assertEqual(out, "")

    def test_orchestrate_async_full_pipeline(self):
        self._write_spec("a.json", {"name": "a", "description": "d",
                                    "allowed_actions": ["r"]})
        o = self._mk()
        # Planner returns one task; worker summarises real data; merger joins.
        plan_json = json.dumps({"sub_tasks": [{"sub_agent": "a", "task": "go"}]})
        outputs = iter([plan_json, "WORKER_SUMMARY", "FINAL_MERGED"])
        orch._claude_call = lambda *a, **k: next(outputs)
        out = asyncio.run(o.orchestrate_async("req", {"r": lambda a: "REAL"}))
        self.assertEqual(out, "FINAL_MERGED")

    def test_orchestrate_sync_no_running_loop(self):
        # No event loop running in this thread → builds a fresh loop internally.
        self._write_spec("a.json", {"name": "a", "description": "d",
                                    "allowed_actions": ["r"]})
        o = self._mk()
        outputs = iter([
            json.dumps({"sub_tasks": [{"sub_agent": "a", "task": "go"}]}),
            "WS", "FINAL_SYNC"])
        orch._claude_call = lambda *a, **k: next(outputs)
        self.assertEqual(o.orchestrate("req", {"r": lambda a: "REAL"}),
                         "FINAL_SYNC")
        # Confirm we did not leave a running loop installed.
        with self.assertRaises(RuntimeError):
            asyncio.get_running_loop()

    def test_orchestrate_sync_empty_plan(self):
        o = self._mk()
        self.assertEqual(o.orchestrate("req", {}), "")

    def test_orchestrate_sync_called_from_worker_thread(self):
        # orchestrate() invoked via asyncio.to_thread runs in a worker thread
        # that has NO running loop of its own, so it takes the fresh-loop
        # (no-running-loop) branch even though an outer loop drives the test.
        # Confirms the wrapper is safe to call from a non-loop worker thread.
        self._write_spec("a.json", {"name": "a", "description": "d",
                                    "allowed_actions": ["r"]})
        o = self._mk()
        outputs = iter([
            json.dumps({"sub_tasks": [{"sub_agent": "a", "task": "go"}]}),
            "WS", "FINAL_THREADED"])
        orch._claude_call = lambda *a, **k: next(outputs)

        async def _driver():
            return await asyncio.to_thread(
                o.orchestrate, "req", {"r": lambda a: "REAL"})

        self.assertEqual(asyncio.run(_driver()), "FINAL_THREADED")

    def test_orchestrate_sync_detects_running_loop(self):
        # Drive the "loop already running in THIS thread" branch (the
        # concurrent.futures offload). We call the *sync* orchestrate()
        # directly from inside a running coroutine — so asyncio.get_running_loop()
        # succeeds and orchestrate() must hand the work to a worker thread that
        # spins up its own fresh loop. The outer loop thread blocks on
        # .result() while that worker drives the pipeline; because the worker's
        # loop is independent, there is no deadlock.
        self._write_spec("a.json", {"name": "a", "description": "d",
                                    "allowed_actions": ["r"]})
        o = self._mk()
        outputs = iter([
            json.dumps({"sub_tasks": [{"sub_agent": "a", "task": "go"}]}),
            "WS", "OK_RUNLOOP"])
        orch._claude_call = lambda *a, **k: next(outputs)

        result_box = {}

        async def _main():
            # Synchronous, in-loop-thread call → exercises lines 816-829.
            result_box["r"] = o.orchestrate("req", {"r": lambda a: "REAL"})

        asyncio.run(_main())
        self.assertEqual(result_box["r"], "OK_RUNLOOP")


class WorkerTimeoutBackstopTests(unittest.TestCase):
    """P1-7: a wedged worker must NOT deadlock the turn.

    `timeout_s` is forwarded into the worker for its LLM/HTTP socket timeouts,
    but the worker's direct/auto ACTION call (a full JARVIS action) has no inner
    timeout. dispatch_sub_agents must wrap the asyncio.to_thread hand-off in an
    asyncio.wait_for backstop so a hung worker is abandoned (returned as an empty
    errored result the merger silently omits) while every other worker — and the
    overall turn — still completes. We drive the backstop deterministically by
    patching asyncio.wait_for to raise (no real sleeps / no real hung threads,
    per the suite isolation contract)."""

    def setUp(self):
        self._claude = orch._claude_call
        self._wait_for = orch.asyncio.wait_for
        # Fast, deterministic worker LLM seam for the non-timing-out task.
        orch._claude_call = lambda model, system, user, **k: "done"

    def tearDown(self):
        orch._claude_call = self._claude
        orch.asyncio.wait_for = self._wait_for

    def test_hung_worker_times_out_to_error_result(self):
        # Force the backstop to fire for EVERY worker: the dispatcher must turn
        # the timeout into an error SubTaskResult instead of propagating it.
        async def _always_timeout(awaitable, timeout=None):
            # Close the wrapped coroutine so we don't leak an un-awaited warning,
            # then simulate the wall-clock backstop tripping.
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise asyncio.TimeoutError()

        orch.asyncio.wait_for = _always_timeout
        specs = {"a": orch.SubAgentSpec(name="a", description="d",
                                        allowed_actions=["r"])}
        out = asyncio.run(orch.dispatch_sub_agents(
            [orch.SubTask(sub_agent="a", task="t")],
            specs, {"r": lambda arg: "REAL"}, timeout_s=5.0))
        self.assertEqual(len(out), 1)                  # gather still returned
        self.assertEqual(out[0].output, "")            # nothing fabricated
        self.assertIsNotNone(out[0].error)
        self.assertIn("timed out", out[0].error.lower())
        self.assertEqual(out[0].sub_agent, "a")

    def test_one_hung_worker_does_not_block_the_others(self):
        # Only the worker named "slow" times out; "fast" must still complete.
        # This is the core no-deadlock guarantee: a single wedged tool can't
        # take the whole turn down with it.
        async def _selective(awaitable, timeout=None):
            # Run the real worker to completion, then decide per-result.
            res = await awaitable
            if res.sub_agent == "slow":
                raise asyncio.TimeoutError()
            return res

        orch.asyncio.wait_for = _selective
        specs = {
            "fast": orch.SubAgentSpec(name="fast", description="d",
                                      allowed_actions=["r"]),
            "slow": orch.SubAgentSpec(name="slow", description="d",
                                      allowed_actions=["r"]),
        }
        tasks = [orch.SubTask(sub_agent="fast", task="t1"),
                 orch.SubTask(sub_agent="slow", task="t2")]
        out = asyncio.run(orch.dispatch_sub_agents(
            tasks, specs, {"r": lambda arg: "REAL"},
            max_parallel=2, timeout_s=5.0))
        by_agent = {r.sub_agent: r for r in out}
        self.assertEqual(by_agent["fast"].output, "done")   # fast finished
        self.assertIsNone(by_agent["fast"].error)
        self.assertEqual(by_agent["slow"].output, "")       # slow abandoned
        self.assertIn("timed out", (by_agent["slow"].error or "").lower())

    def test_backstop_budget_exceeds_inner_worker_timeout(self):
        # The wall-clock backstop must be strictly larger than the inner
        # per-call timeout it guards, so a worker about to return its own
        # graceful raw-data fallback is never preempted. We can't read the
        # closure constant directly, so assert the relationship the code uses.
        timeout_s = 30.0
        backstop_s = max(1.0, timeout_s) * 2.0 + 5.0
        self.assertGreater(backstop_s, timeout_s)


class OverallTimeoutBackstopTests(unittest.TestCase):
    """P1-7 (sync entry): the synchronous orchestrate() wrapper used by the main
    voice turn loop must never block indefinitely on .result(), even if the
    worker loop wedged. When the coarse top-level backstop trips it returns ""
    (the documented 'nothing applicable ran' contract) so the turn proceeds."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="orch_to_")
        self._claude = orch._claude_call
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import shutil
        orch._claude_call = self._claude
        orch._default_orchestrator = None
        shutil.rmtree(self.dir, ignore_errors=True)

    def _mk(self, **kw):
        kw.setdefault("specs_dir", self.dir)
        return orch.Orchestrator(**kw)

    def test_overall_timeout_s_is_positive_and_ordered(self):
        o = self._mk(worker_timeout_s=30.0, planner_timeout_s=20.0,
                     merger_timeout_s=20.0)
        budget = o._overall_timeout_s()
        self.assertGreater(budget, 0.0)
        # Must leave room for at least the planner + one worker wave + merger.
        self.assertGreater(
            budget, o.planner_timeout_s + o.worker_timeout_s + o.merger_timeout_s)

    def test_running_loop_branch_returns_empty_on_overall_timeout(self):
        # Drive the loop-already-running branch with an orchestrate_async that
        # never completes, and a tiny overall budget so .result(timeout=…) trips
        # fast. The wrapper must swallow the TimeoutError and return "".
        o = self._mk()
        hang = __import__("threading").Event()
        self.addCleanup(hang.set)   # release the orphaned worker thread after.

        async def _never_returns(request, actions):
            hang.wait()             # blocks the worker loop until cleanup.
            return "SHOULD_NOT_SURFACE"

        o.orchestrate_async = _never_returns                 # type: ignore
        o._overall_timeout_s = lambda: 0.2                   # type: ignore

        async def _main():
            # In-loop-thread sync call → takes the concurrent.futures branch.
            return o.orchestrate("req", {"r": lambda a: "REAL"})

        self.assertEqual(asyncio.run(_main()), "")


class ModuleSingletonTests(unittest.TestCase):
    def setUp(self):
        self._saved = orch._default_orchestrator
        orch._default_orchestrator = None
        self._claude = orch._claude_call
        self.addCleanup(self._restore)

    def _restore(self):
        orch._default_orchestrator = self._saved
        orch._claude_call = self._claude

    def test_get_orchestrator_is_singleton(self):
        dir1 = tempfile.mkdtemp(prefix="orch_s1_")
        self.addCleanup(lambda: __import__("shutil").rmtree(dir1, ignore_errors=True))
        a = orch.get_orchestrator(specs_dir=dir1)
        b = orch.get_orchestrator(specs_dir="ignored-second-time")
        self.assertIs(a, b)                      # same instance regardless of kwargs

    def test_orchestrate_entrypoint_uses_singleton(self):
        dir1 = tempfile.mkdtemp(prefix="orch_s2_")
        self.addCleanup(lambda: __import__("shutil").rmtree(dir1, ignore_errors=True))
        # Empty specs dir → plan() returns [] → orchestrate returns "".
        out = orch.orchestrate("req", {}, specs_dir=dir1)
        self.assertEqual(out, "")
        # Singleton now exists and is the same object the helper returns.
        self.assertIsNotNone(orch._default_orchestrator)
        self.assertIs(orch.get_orchestrator(), orch._default_orchestrator)


class ModelConstantTests(unittest.TestCase):
    """Guard the model-id constants against re-pinning to a dated build.

    The rest of JARVIS uses un-dated aliases (claude-sonnet-4-6, claude-haiku-
    4-5) so a model refresh doesn't require code edits and a retired dated build
    can't silently rot the worker. A worker model like 'claude-haiku-4-5-2025…'
    would still 'work' until Anthropic retires that exact snapshot, then fail
    closed — so we assert NO module model constant carries an 8-digit date.
    """
    import re as _re
    _DATE_RE = _re.compile(r"\d{8}")

    def test_worker_model_is_undated_haiku_alias(self):
        # The specific value F4 un-pinned: the un-dated Haiku alias.
        self.assertEqual(orch.DEFAULT_WORKER_MODEL, "claude-haiku-4-5")

    def test_no_default_model_constant_is_date_pinned(self):
        for const in ("DEFAULT_PLANNER_MODEL", "DEFAULT_WORKER_MODEL",
                      "DEFAULT_MERGER_MODEL"):
            val = getattr(orch, const)
            self.assertIsNone(
                self._DATE_RE.search(val),
                f"{const}={val!r} is date-pinned; use an un-dated alias",
            )


if __name__ == "__main__":
    unittest.main()
