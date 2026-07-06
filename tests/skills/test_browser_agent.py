"""Logic tests for skills/browser_agent.py.

browser-use (+ Playwright + an Anthropic key) is an optional dep absent from
the test env, so is_available() is False and every action returns the install
hint — the easy, high-value degradation path. The genuinely interesting pure
logic is the SSRF guard, which we test directly (no DNS for literal IPs;
socket.gethostbyname mocked for hostnames), plus intent prefixing and the
truncation/model helpers. No browser, no asyncio loop, no network.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── shared fakes / helpers for the deep (browser-driving) paths ─────────────
#
# browser-use + Playwright are async-only and run on a daemon thread in the
# real skill. None of that is allowed in a test: we never start the bg thread,
# never construct a real Browser, and never sleep. Instead we drive the
# module's coroutines on a throwaway local event loop and feed the import-
# dependent factories a dict of fake classes via mock.patch.object(_bu_imports).
_SENTINEL = object()


def _run_coro(coro):
    """Run a single coroutine to completion on a fresh, immediately-closed
    event loop. Keeps every async path off the skill's real bg thread."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@contextlib.contextmanager
def _inject_modules(**mods):
    """Temporarily install fake modules into sys.modules (e.g. a fake
    ``browser_use`` whose ``Agent`` lives under ``.agent.service``). Dotted
    names also get attached to their parent package so ``from a.b import c``
    resolves. Everything — including prior ABSENCE — is restored on exit, so
    after the test sys.modules holds the REAL modules again. Mirrors the
    save/restore contract used by tests/skills/test_self_diagnostic.py."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


@contextlib.contextmanager
def _block_import(*names):
    """Force ``import <name>`` to raise ImportError inside the block, even when
    the real package is installed on the dev box (browser_use IS installed
    here). Detaches any already-imported target from sys.modules first so the
    import machinery can't satisfy it from cache, then restores on exit."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in blocked or top in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    saved_mod: dict[str, object] = {}
    for name in list(blocked):
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                saved_mod[key] = sys.modules.pop(key)
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        for key, mod in saved_mod.items():
            sys.modules[key] = mod


class _FakeLLM:
    def __init__(self, **kw):
        self.kw = kw


class _FakeBrowser:
    """Stand-in browser handle. Records start/close, can carry a fake page."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        self._page = None

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


class _FakeHistory:
    def __init__(self, final="all done, sir"):
        self._final = final

    def final_result(self):
        return self._final


class _FakeAgent:
    """Captures construction kwargs and yields a configurable history/run."""
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.task = kwargs.get("task")
        self.stopped = False

    async def run(self, **kwargs):
        return _FakeHistory()

    async def stop(self):
        self.stopped = True


def _imports(*, agent=_FakeAgent, browser=_FakeBrowser,
             browser_config=None, chat=_FakeLLM):
    """Build the dict shape _bu_imports() returns, with fakes."""
    return {
        "Agent": agent,
        "Browser": browser,
        "BrowserConfig": browser_config,
        "ChatAnthropic": chat,
    }


class _BrowserAgentTestBase(unittest.TestCase):
    """Loads the skill in isolation and guarantees the module's global _state
    and bg-loop machinery are reset after every test so no real thread / loop
    / browser handle leaks into the rest of the suite."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("browser_agent")
        # Snapshot pristine state to restore in tearDown.
        self._pristine_state = {
            "loop": None, "thread": None, "current_task": None,
            "current_fut": None, "agent": None, "browser": None,
            "step_count": 0, "last_step": None, "started_at": None,
            "finished_at": None, "last_result": None, "last_error": None,
        }

    def tearDown(self):
        # Never leave a live loop/thread/handle behind.
        self.mod._state.clear()
        self.mod._state.update(self._pristine_state)


class IpBlockedTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("browser_agent")

    def test_loopback_blocked(self):
        self.assertTrue(self.mod._ip_is_blocked("127.0.0.1"))

    def test_private_ranges_blocked(self):
        for ip in ("10.0.0.5", "172.16.3.4", "192.168.1.1"):
            self.assertTrue(self.mod._ip_is_blocked(ip), ip)

    def test_cloud_metadata_ip_blocked(self):
        self.assertTrue(self.mod._ip_is_blocked("169.254.169.254"))

    def test_public_ip_allowed(self):
        self.assertFalse(self.mod._ip_is_blocked("8.8.8.8"))

    def test_garbage_not_blocked(self):
        self.assertFalse(self.mod._ip_is_blocked("not-an-ip"))


class HostBlockedTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("browser_agent")

    def test_literal_private_ip(self):
        self.assertTrue(self.mod._host_is_blocked("192.168.0.10"))

    def test_localhost_alias_blocked_without_resolution(self):
        self.assertTrue(self.mod._host_is_blocked("localhost"))
        self.assertTrue(self.mod._host_is_blocked("foo.localhost"))

    def test_bracketed_ipv6_loopback(self):
        self.assertTrue(self.mod._host_is_blocked("[::1]"))

    def test_public_host_resolving_to_public_ip_allowed(self):
        with mock.patch("socket.gethostbyname", return_value="93.184.216.34"):
            self.assertFalse(self.mod._host_is_blocked("example.com"))

    def test_public_name_resolving_to_private_ip_blocked(self):
        # DNS-rebinding style: a public-looking name that resolves to LAN.
        with mock.patch("socket.gethostbyname", return_value="10.1.2.3"):
            self.assertTrue(self.mod._host_is_blocked("sneaky.example.com"))

    def test_resolution_failure_allows_public_name(self):
        with mock.patch("socket.gethostbyname", side_effect=OSError("no DNS")):
            self.assertFalse(self.mod._host_is_blocked("example.com"))


class ExtractHostsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("browser_agent")

    def test_extracts_http_url_hostname(self):
        hosts = self.mod._extract_hosts("please open https://news.ycombinator.com/item?id=1")
        self.assertIn("news.ycombinator.com", hosts)

    def test_bare_host_port(self):
        hosts = self.mod._extract_hosts("localhost:8188")
        self.assertIn("localhost", hosts)

    def test_plain_task_has_no_hosts(self):
        self.assertEqual(self.mod._extract_hosts("book me a haircut friday"), [])


class SsrfGuardTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("browser_agent")

    def test_blocks_metadata_url(self):
        out = self.mod._ssrf_guard("browse to http://169.254.169.254/latest/meta-data/")
        self.assertEqual(out, self.mod._SSRF_REFUSAL)

    def test_blocks_localhost(self):
        # Realistic injection vector: a full http://localhost URL in the task.
        self.assertEqual(self.mod._ssrf_guard("go to http://localhost:8188/admin"),
                         self.mod._SSRF_REFUSAL)

    def test_blocks_bare_localhost_host_token(self):
        # browser_open-style bare 'host:port' as the leading token is also caught.
        self.assertEqual(self.mod._ssrf_guard("localhost:8188"),
                         self.mod._SSRF_REFUSAL)

    def test_allows_public_task(self):
        with mock.patch("socket.gethostbyname", return_value="93.184.216.34"):
            self.assertIsNone(self.mod._ssrf_guard("open https://example.com"))

    def test_no_hosts_allows(self):
        self.assertIsNone(self.mod._ssrf_guard("find the cheapest flight to tokyo"))


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("browser_agent")

    def test_truncate_short(self):
        self.assertEqual(self.mod._truncate("hi", 100), "hi")

    def test_truncate_long_marks(self):
        out = self.mod._truncate("x" * 50, 20)
        self.assertTrue(out.endswith("(truncated)"))
        self.assertLess(len(out), 50)   # shorter than the input

    def test_model_name_from_env(self):
        with mock.patch.dict(os.environ, {"BROWSER_AGENT_MODEL": "claude-test-9"}):
            self.assertEqual(self.mod._model_name(), "claude-test-9")


class DegradationTests(unittest.TestCase):
    """browser-use absent → every action returns the install hint."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("browser_agent")

    def test_is_available_false_without_browser_use(self):
        # _bu_imports returns {} when browser_use isn't importable.
        with mock.patch.object(self.mod, "_bu_imports", return_value={}):
            self.assertFalse(self.mod.is_available())

    def test_browser_task_returns_hint(self):
        with mock.patch.object(self.mod, "is_available", return_value=False):
            out = self.actions["browser_task"]("do a thing")
        self.assertIn("Browser agent is offline", out)
        self.assertIn("pip install browser-use", out)

    def test_browser_task_empty_arg(self):
        out = self.actions["browser_task"]("")
        self.assertIn("Give me a task", out)

    def test_browser_status_before_any_run(self):
        out = self.actions["browser_status"]("")
        self.assertIn("No browser task has run yet", out)

    def test_browser_stop_when_idle(self):
        out = self.actions["browser_stop"]("")
        self.assertIn("No browser task is running", out)


class IntentAndGuardThroughActionsTests(unittest.TestCase):
    """With availability forced True (and the actual runner stubbed), verify
    the SSRF refusal fires through the action and intent prefixing happens."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("browser_agent")

    def test_ssrf_refusal_through_action(self):
        with mock.patch.object(self.mod, "is_available", return_value=True), \
             mock.patch.object(self.mod, "_run_task_sync") as runner:
            out = self.actions["browser_open"]("http://127.0.0.1:8080/admin")
        self.assertEqual(out, self.mod._SSRF_REFUSAL)
        runner.assert_not_called()   # refused before ever submitting

    def test_book_appointment_prefixes_task(self):
        captured = {}

        def _fake_run(task, **kw):
            captured["task"] = task
            return "done"

        with mock.patch.object(self.mod, "is_available", return_value=True), \
             mock.patch.object(self.mod, "_ssrf_guard", return_value=None), \
             mock.patch.object(self.mod, "_run_task_sync", side_effect=_fake_run):
            self.actions["book_appointment"]("a haircut Friday afternoon")
        self.assertTrue(captured["task"].lower().startswith("book me"))
        self.assertIn("haircut Friday afternoon", captured["task"])

    def test_book_appointment_keeps_existing_verb(self):
        captured = {}
        with mock.patch.object(self.mod, "is_available", return_value=True), \
             mock.patch.object(self.mod, "_ssrf_guard", return_value=None), \
             mock.patch.object(self.mod, "_run_task_sync",
                               side_effect=lambda task, **kw: captured.update(task=task)):
            self.actions["book_appointment"]("schedule a dentist visit")
        # Already starts with 'schedule' → not re-prefixed with 'Book me'.
        self.assertTrue(captured["task"].lower().startswith("schedule"))

    def test_browse_for_builds_search_task(self):
        captured = {}
        with mock.patch.object(self.mod, "is_available", return_value=True), \
             mock.patch.object(self.mod, "_ssrf_guard", return_value=None), \
             mock.patch.object(self.mod, "_run_task_sync",
                               side_effect=lambda task, **kw: captured.update(task=task)):
            self.actions["browse_for"]("best mechanical keyboards")
        self.assertIn("best mechanical keyboards", captured["task"])
        self.assertIn("Summarise", captured["task"])


class BuImportsTests(_BrowserAgentTestBase):
    def test_returns_empty_when_browser_use_missing(self):
        # The dev box HAS browser_use installed; force the import to fail.
        with _block_import("browser_use"):
            self.assertEqual(self.mod._bu_imports(), {})

    def test_real_browser_use_present(self):
        # Sanity: with the real package, Agent + a chat wrapper are found.
        # browser_use + langchain_anthropic are absent on the light-deps CI
        # runner, so this real-package check only applies when they're installed.
        import importlib.util
        if importlib.util.find_spec("browser_use") is None:
            self.skipTest("browser_use not installed (absent on the CI runner)")
        out = self.mod._bu_imports()
        self.assertIsNotNone(out.get("Agent"))
        self.assertIsNotNone(out.get("ChatAnthropic"))

    def test_agent_under_agent_service_fallback(self):
        # browser_use top-level lacks Agent → resolved via .agent.service.
        bu = types.ModuleType("browser_use")  # no Agent / Browser attrs
        svc = types.ModuleType("browser_use.agent.service")
        svc.Agent = _FakeAgent
        llm_mod = types.ModuleType("browser_use.llm")
        llm_mod.ChatAnthropic = _FakeLLM
        with _inject_modules(**{
            "browser_use": bu,
            "browser_use.agent": types.ModuleType("browser_use.agent"),
            "browser_use.agent.service": svc,
            "browser_use.llm": llm_mod,
        }):
            out = self.mod._bu_imports()
        self.assertIs(out["Agent"], _FakeAgent)
        self.assertIs(out["ChatAnthropic"], _FakeLLM)

    def test_agent_service_missing_returns_empty(self):
        # No top-level Agent and the .agent.service import fails → {}.
        bu = types.ModuleType("browser_use")
        with _inject_modules(browser_use=bu), \
                _block_import("browser_use.agent.service"):
            out = self.mod._bu_imports()
        self.assertEqual(out, {})

    def test_langchain_anthropic_fallback(self):
        # browser_use.llm absent → ChatAnthropic comes from langchain_anthropic.
        bu = types.ModuleType("browser_use")
        bu.Agent = _FakeAgent
        bu.BrowserSession = _FakeBrowser
        lc = types.ModuleType("langchain_anthropic")
        lc.ChatAnthropic = _FakeLLM
        with _inject_modules(browser_use=bu, langchain_anthropic=lc), \
                _block_import("browser_use.llm"):
            out = self.mod._bu_imports()
        self.assertIs(out["ChatAnthropic"], _FakeLLM)
        self.assertIs(out["Browser"], _FakeBrowser)

    def test_chat_anthropic_none_when_no_wrapper(self):
        # Neither browser_use.llm nor langchain_anthropic importable.
        bu = types.ModuleType("browser_use")
        bu.Agent = _FakeAgent
        with _inject_modules(browser_use=bu), \
                _block_import("browser_use.llm", "langchain_anthropic"):
            out = self.mod._bu_imports()
        self.assertIsNone(out["ChatAnthropic"])


class GuardEdgeTests(_BrowserAgentTestBase):
    def test_host_blocked_empty_string(self):
        self.assertFalse(self.mod._host_is_blocked(""))
        self.assertFalse(self.mod._host_is_blocked("   "))

    def test_extract_hosts_bare_ipv6_bracketed_with_port(self):
        hosts = self.mod._extract_hosts("[::1]:8188/admin")
        self.assertIn("::1", hosts)

    def test_extract_hosts_bare_ipv6_no_brackets(self):
        hosts = self.mod._extract_hosts("fe80::1")
        self.assertIn("fe80::1", hosts)

    def test_extract_hosts_empty(self):
        self.assertEqual(self.mod._extract_hosts(""), [])

    def test_extract_hosts_url_with_unparseable_host(self):
        # urlparse on a malformed URL yields no hostname → skipped, no crash.
        hosts = self.mod._extract_hosts("http://")
        self.assertEqual(hosts, [])

    def test_ssrf_guard_fails_open_on_internal_error(self):
        # If _extract_hosts itself raises, the guard must fail OPEN (None).
        with mock.patch.object(self.mod, "_extract_hosts",
                               side_effect=RuntimeError("boom")):
            self.assertIsNone(self.mod._ssrf_guard("anything"))

    def test_extract_hosts_urlparse_raises_skips_url(self):
        # urlparse blowing up on one matched URL is swallowed (inner except).
        import urllib.parse as _up
        with mock.patch.object(_up, "urlparse",
                               side_effect=ValueError("bad url")):
            hosts = self.mod._extract_hosts("visit https://example.com/x")
        self.assertEqual(hosts, [])

    def test_extract_hosts_outer_exception_swallowed(self):
        # An unexpected failure in the body (re.findall) is caught by the
        # outer guard and yields [] rather than propagating.
        import re as _re
        with mock.patch.object(_re, "findall",
                               side_effect=RuntimeError("regex blew up")):
            self.assertEqual(self.mod._extract_hosts("https://example.com"), [])

    def test_bracketed_ipv6_public_allowed(self):
        # A bracketed *public* IPv6 literal is not blocked.
        self.assertFalse(self.mod._host_is_blocked("[2606:4700:4700::1111]"))


class IsAvailableTests(_BrowserAgentTestBase):
    def test_available_with_imports_and_key(self):
        with mock.patch.object(self.mod, "_bu_imports", return_value=_imports()), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            self.assertTrue(self.mod.is_available())

    def test_unavailable_without_key(self):
        with mock.patch.object(self.mod, "_bu_imports", return_value=_imports()), \
                mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.mod.is_available())

    def test_unavailable_without_chat(self):
        imps = _imports(chat=None)
        with mock.patch.object(self.mod, "_bu_imports", return_value=imps), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            self.assertFalse(self.mod.is_available())


class ModelNameTests(_BrowserAgentTestBase):
    def test_default_when_no_env_no_companion(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                _block_import("bobert_companion"):
            self.assertEqual(self.mod._model_name(), "claude-sonnet-5")

    def test_from_bobert_companion(self):
        fake = types.ModuleType("bobert_companion")
        fake.CLAUDE_MODEL = "claude-from-companion"
        with mock.patch.dict(os.environ, {}, clear=True), \
                _inject_modules(bobert_companion=fake):
            self.assertEqual(self.mod._model_name(), "claude-from-companion")

    def test_companion_without_attr_falls_back(self):
        fake = types.ModuleType("bobert_companion")  # no CLAUDE_MODEL
        with mock.patch.dict(os.environ, {}, clear=True), \
                _inject_modules(bobert_companion=fake):
            self.assertEqual(self.mod._model_name(), "claude-sonnet-5")


class MakeBrowserTests(_BrowserAgentTestBase):
    def test_browser_none_returns_none(self):
        imps = _imports(browser=None)
        self.assertIsNone(_run_coro(self.mod._make_browser(imps, headless=True)))

    def test_constructs_and_starts_browser(self):
        imps = _imports(browser=_FakeBrowser)
        handle = _run_coro(self.mod._make_browser(imps, headless=False))
        self.assertIsInstance(handle, _FakeBrowser)
        self.assertTrue(handle.started)  # .start() was called
        # First kwargs combo that works carries user_data_dir + headless.
        self.assertIn("user_data_dir", handle.kwargs)

    def test_async_start_is_awaited(self):
        class _AsyncStartBrowser(_FakeBrowser):
            async def start(self):
                self.started = "async"
        imps = _imports(browser=_AsyncStartBrowser)
        handle = _run_coro(self.mod._make_browser(imps, headless=False))
        self.assertEqual(handle.started, "async")

    def test_typeerror_falls_through_kwargs_combos(self):
        # Only the no-arg constructor works → loop must reach {} kwargs.
        class _PickyBrowser(_FakeBrowser):
            def __init__(self, **kwargs):
                if kwargs:
                    raise TypeError("no kwargs accepted")
                super().__init__()
        imps = _imports(browser=_PickyBrowser)
        handle = _run_coro(self.mod._make_browser(imps, headless=True))
        self.assertIsInstance(handle, _PickyBrowser)
        self.assertEqual(handle.kwargs, {})

    def test_falls_back_to_browser_config_path(self):
        # Every direct Browser(**kwargs) raises (non-TypeError) → config path.
        made = {}

        class _CfgOnlyBrowser:
            def __init__(self, config=None, **kwargs):
                if config is None:
                    raise RuntimeError("must pass config")
                made["config"] = config

        class _FakeCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        imps = _imports(browser=_CfgOnlyBrowser, browser_config=_FakeCfg)
        handle = _run_coro(self.mod._make_browser(imps, headless=True))
        self.assertIsInstance(handle, _CfgOnlyBrowser)
        self.assertIsInstance(made["config"], _FakeCfg)

    def test_config_path_none_when_no_config_class(self):
        class _AlwaysFails:
            def __init__(self, **kwargs):
                raise RuntimeError("nope")
        imps = _imports(browser=_AlwaysFails, browser_config=None)
        self.assertIsNone(_run_coro(self.mod._make_browser(imps, headless=True)))

    def test_config_path_all_typeerror_returns_none(self):
        class _AlwaysFails:
            def __init__(self, **kwargs):
                raise RuntimeError("direct fails")

        class _CfgTypeError:
            def __init__(self, **kwargs):
                raise TypeError("cfg bad kwargs")
        imps = _imports(browser=_AlwaysFails, browser_config=_CfgTypeError)
        self.assertIsNone(_run_coro(self.mod._make_browser(imps, headless=True)))

    def test_config_path_typeerror_then_succeeds(self):
        # First BrowserConfig kwargs combo TypeErrors; a later one works.
        made = {}

        class _DirectFails:
            def __init__(self, config=None, **kwargs):
                if config is None:
                    raise RuntimeError("need config")
                made["ok"] = True

        class _PickyCfg:
            def __init__(self, **kwargs):
                # Reject the first combo (has user_data_dir + headless only).
                if set(kwargs) == {"user_data_dir", "headless"}:
                    raise TypeError("first combo rejected")
        imps = _imports(browser=_DirectFails, browser_config=_PickyCfg)
        handle = _run_coro(self.mod._make_browser(imps, headless=True))
        self.assertIsInstance(handle, _DirectFails)
        self.assertTrue(made["ok"])

    def test_config_path_generic_exception_then_succeeds(self):
        # First config combo raises a NON-TypeError → except Exception/continue
        # (line 373-374); a later combo works.
        made = {}

        class _DirectFails:
            def __init__(self, config=None, **kwargs):
                if config is None:
                    raise RuntimeError("need config")
                made["ok"] = True

        class _GenericFailCfg:
            def __init__(self, **kwargs):
                if set(kwargs) == {"user_data_dir", "headless"}:
                    raise RuntimeError("first combo non-TypeError fail")
        imps = _imports(browser=_DirectFails, browser_config=_GenericFailCfg)
        handle = _run_coro(self.mod._make_browser(imps, headless=True))
        self.assertIsInstance(handle, _DirectFails)
        self.assertTrue(made["ok"])


class MakeLlmTests(_BrowserAgentTestBase):
    def test_none_when_no_chat_class(self):
        self.assertIsNone(self.mod._make_llm(_imports(chat=None)))

    def test_constructs_llm(self):
        with mock.patch.object(self.mod, "_model_name", return_value="m-1"):
            llm = self.mod._make_llm(_imports(chat=_FakeLLM))
        self.assertIsInstance(llm, _FakeLLM)
        self.assertEqual(llm.kw.get("model"), "m-1")

    def test_model_name_kwarg_variant(self):
        # Class that only accepts model_name= → loop reaches 2nd combo.
        class _ModelNameLLM:
            def __init__(self, model_name=None, temperature=None):
                self.model_name = model_name
        with mock.patch.object(self.mod, "_model_name", return_value="m-2"):
            llm = self.mod._make_llm(_imports(chat=_ModelNameLLM))
        self.assertEqual(llm.model_name, "m-2")

    def test_returns_none_when_all_constructions_fail(self):
        class _BadLLM:
            def __init__(self, **kwargs):
                raise RuntimeError("cannot build")
        with mock.patch.object(self.mod, "_model_name", return_value="m"):
            self.assertIsNone(self.mod._make_llm(_imports(chat=_BadLLM)))


class StepHookTests(_BrowserAgentTestBase):
    def test_increments_step_count(self):
        self.mod._state["step_count"] = 0
        self.mod._step_hook()
        self.assertEqual(self.mod._state["step_count"], 1)

    def test_scrapes_action_description(self):
        class _Obj:
            action = "clicking the submit button"
        self.mod._state["step_count"] = 0
        self.mod._step_hook(_Obj())
        self.assertEqual(self.mod._state["step_count"], 1)
        self.assertEqual(self.mod._state["last_step"], "clicking the submit button")

    def test_handles_none_step_count(self):
        self.mod._state["step_count"] = None
        self.mod._step_hook()
        self.assertEqual(self.mod._state["step_count"], 1)


class RunTaskInnerTests(_BrowserAgentTestBase):
    def _run(self, **patch):
        return _run_coro(self.mod._run_task_inner("do a thing", 5, True))

    def test_install_hint_when_no_agent(self):
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=None)):
            out = self._run()
        self.assertIn("Browser agent is offline", out)

    def test_install_hint_when_no_llm(self):
        with mock.patch.object(self.mod, "_bu_imports", return_value=_imports()), \
                mock.patch.object(self.mod, "_make_llm", return_value=None):
            out = self._run()
        self.assertIn("Browser agent is offline", out)

    def test_happy_path_returns_final_result(self):
        with mock.patch.object(self.mod, "_bu_imports", return_value=_imports()), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertEqual(out, "all done, sir")
        # Agent got the task + the step callback on its first attempt.
        self.assertEqual(_FakeAgent.last_kwargs.get("task"), "do a thing")

    def test_agent_construction_all_fail(self):
        class _BadAgent:
            def __init__(self, **kwargs):
                raise RuntimeError("cannot build agent")
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=_BadAgent)), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertIn("could not construct an Agent", out)

    def test_agent_construction_typeerror_then_succeeds(self):
        # First kwargs combo TypeErrors (browser=), later one (browser_session=)
        # works → exercises the TypeError-continue branch in construction.
        class _SessionAgent(_FakeAgent):
            def __init__(self, **kwargs):
                if "browser" in kwargs:
                    raise TypeError("no 'browser' kwarg")
                super().__init__(**kwargs)
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=_SessionAgent)), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertEqual(out, "all done, sir")

    def test_str_fallback_raises_uses_default_message(self):
        # history has no usable getter AND str(history) raises → last-ditch
        # 'task completed.' default.
        class _Unstringable:
            def __str__(self):
                raise RuntimeError("cannot stringify")

        class _UnstringableAgent(_FakeAgent):
            async def run(self, **kwargs):
                return _Unstringable()
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=_UnstringableAgent)), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertEqual(out, "task completed.")

    def test_run_typeerror_then_succeeds_on_next_kwargs(self):
        # First .run(max_steps=) raises TypeError → retried with steps= etc.
        class _PickyAgent(_FakeAgent):
            async def run(self, **kwargs):
                if "max_steps" in kwargs:
                    raise TypeError("max_steps unsupported")
                return _FakeHistory("recovered result")
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=_PickyAgent)), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertEqual(out, "recovered result")

    def test_run_all_typeerror_returns_failure(self):
        class _AlwaysTypeError(_FakeAgent):
            async def run(self, **kwargs):
                raise TypeError("bad run signature")
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=_AlwaysTypeError)), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertIn(".run() failed", out)

    def test_result_getter_used_when_no_final_result(self):
        class _ResultHistory:
            def result(self):
                return "from result()"

        class _ResultAgent(_FakeAgent):
            async def run(self, **kwargs):
                return _ResultHistory()
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=_ResultAgent)), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertEqual(out, "from result()")

    def test_attribute_output_non_callable(self):
        class _AttrHistory:
            output = "plain attribute output"

        class _AttrAgent(_FakeAgent):
            async def run(self, **kwargs):
                return _AttrHistory()
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=_AttrAgent)), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertEqual(out, "plain attribute output")

    def test_stringifies_history_when_no_getter(self):
        class _Opaque:
            def __str__(self):
                return "opaque history str"

        class _OpaqueAgent(_FakeAgent):
            async def run(self, **kwargs):
                return _Opaque()
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=_OpaqueAgent)), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertEqual(out, "opaque history str")

    def test_getter_raises_then_falls_through(self):
        # final_result() raises → loop continues → str(history) fallback.
        class _RaisingHistory:
            def final_result(self):
                raise RuntimeError("boom")

            def __str__(self):
                return "stringified after raise"

        class _RaisingAgent(_FakeAgent):
            async def run(self, **kwargs):
                return _RaisingHistory()
        with mock.patch.object(self.mod, "_bu_imports",
                               return_value=_imports(agent=_RaisingAgent)), \
                mock.patch.object(self.mod, "_make_llm", return_value=_FakeLLM()), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=_FakeBrowser())):
            out = self._run()
        self.assertEqual(out, "stringified after raise")


class CloseBrowserAsyncTests(_BrowserAgentTestBase):
    def test_noop_when_nothing_open(self):
        self.mod._state["browser"] = None
        self.mod._state["agent"] = None
        _run_coro(self.mod._close_browser_async())  # must not raise
        self.assertIsNone(self.mod._state["browser"])

    def test_closes_agent_and_browser(self):
        agent = _FakeAgent(task="t")
        browser = _FakeBrowser()
        self.mod._state["agent"] = agent
        self.mod._state["browser"] = browser
        _run_coro(self.mod._close_browser_async())
        self.assertTrue(agent.stopped)
        self.assertTrue(browser.closed)
        self.assertIsNone(self.mod._state["agent"])
        self.assertIsNone(self.mod._state["browser"])

    def test_tolerates_close_raising(self):
        class _BadBrowser:
            def close(self):
                raise RuntimeError("close failed")
        self.mod._state["agent"] = None
        self.mod._state["browser"] = _BadBrowser()
        _run_coro(self.mod._close_browser_async())  # swallowed
        self.assertIsNone(self.mod._state["browser"])

    def test_agent_stop_raises_then_tries_next_method(self):
        # agent.stop() raises → except/continue → close() succeeds.
        calls = []

        class _FlakyAgent:
            def stop(self):
                calls.append("stop")
                raise RuntimeError("stop failed")

            def close(self):
                calls.append("close")
        self.mod._state["agent"] = _FlakyAgent()
        self.mod._state["browser"] = None
        _run_coro(self.mod._close_browser_async())
        self.assertEqual(calls, ["stop", "close"])

    def test_async_browser_close_awaited(self):
        closed = {}

        class _AsyncCloseBrowser:
            async def close(self):
                closed["done"] = True
        self.mod._state["agent"] = None
        self.mod._state["browser"] = _AsyncCloseBrowser()
        _run_coro(self.mod._close_browser_async())
        self.assertTrue(closed["done"])

    def test_non_callable_close_attr_skipped(self):
        # browser.close is a non-callable attr → skipped; .stop() used instead.
        used = {}

        class _OddBrowser:
            close = "not callable"

            def stop(self):
                used["stopped"] = True
        self.mod._state["agent"] = None
        self.mod._state["browser"] = _OddBrowser()
        _run_coro(self.mod._close_browser_async())
        self.assertTrue(used["stopped"])


class OrchestrateTests(_BrowserAgentTestBase):
    def test_returns_inner_result_and_sets_finished(self):
        with mock.patch.object(self.mod, "_run_task_inner",
                               new=mock.AsyncMock(return_value="ok result")), \
                mock.patch.object(self.mod, "_close_browser_async",
                                  new=mock.AsyncMock()):
            out = _run_coro(self.mod._orchestrate("task", 5, True))
        self.assertEqual(out, "ok result")
        self.assertIsNotNone(self.mod._state["finished_at"])

    def test_generic_exception_surfaced(self):
        with mock.patch.object(self.mod, "_run_task_inner",
                               new=mock.AsyncMock(side_effect=ValueError("kaboom"))), \
                mock.patch.object(self.mod, "_close_browser_async",
                                  new=mock.AsyncMock()):
            out = _run_coro(self.mod._orchestrate("task", 5, True))
        self.assertIn("Browser agent failed", out)
        self.assertIn("ValueError", out)
        self.assertIsNotNone(self.mod._state["last_error"])

    def test_api_cap_message_for_quota_errors(self):
        with mock.patch.object(self.mod, "_run_task_inner",
                               new=mock.AsyncMock(
                                   side_effect=RuntimeError("usage limit reached"))), \
                mock.patch.object(self.mod, "_close_browser_async",
                                  new=mock.AsyncMock()):
            out = _run_coro(self.mod._orchestrate("task", 5, True))
        self.assertIn("capped until it resets", out)

    def test_cleanup_runs_even_on_success(self):
        closer = mock.AsyncMock()
        with mock.patch.object(self.mod, "_run_task_inner",
                               new=mock.AsyncMock(return_value="done")), \
                mock.patch.object(self.mod, "_close_browser_async", new=closer):
            _run_coro(self.mod._orchestrate("task", 5, True))
        closer.assert_awaited_once()


class _ImmediateFuture:
    """A concurrent.futures-like future that resolves synchronously, standing
    in for asyncio.run_coroutine_threadsafe's return value."""
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self._cancelled = False
        self._callbacks = []

    def done(self):
        return True

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._result

    def cancel(self):
        self._cancelled = True
        return True

    def add_done_callback(self, cb):
        cb(self)


class SubmitAndSyncTests(_BrowserAgentTestBase):
    """Drive _submit / _run_task_sync without the real bg loop: the loop
    thread is stubbed and run_coroutine_threadsafe executes the coroutine on
    a throwaway local loop, returning an immediate future."""

    def _patch_loop(self, result="task result", exc=None):
        sentinel_loop = object()

        def _fake_rcts(coro, loop):
            # Actually drive the coroutine so its side effects (state) run,
            # but isolate it on a local loop. Then hand back a resolved future.
            try:
                got = _run_coro(coro)
            except Exception as e:  # pragma: no cover - defensive
                return _ImmediateFuture(exc=e)
            if exc is not None:
                return _ImmediateFuture(exc=exc)
            return _ImmediateFuture(result=result if result is not None else got)

        return (
            mock.patch.object(self.mod, "_start_loop_thread",
                              return_value=sentinel_loop),
            mock.patch.object(self.mod.asyncio, "run_coroutine_threadsafe",
                              side_effect=_fake_rcts),
        )

    def test_submit_rejects_concurrent_task(self):
        with mock.patch.object(self.mod, "_start_loop_thread",
                               return_value=object()):
            self.mod._state["current_fut"] = _ImmediateFuture(result="x")
            # done() True for our immediate future, so simulate a *running* one:
            running = _ImmediateFuture(result="x")
            running.done = lambda: False
            self.mod._state["current_fut"] = running
            with self.assertRaises(RuntimeError):
                self.mod._submit("task", max_steps=3, headless=True)

    def test_run_task_sync_happy(self):
        p1, p2 = self._patch_loop(result="final answer")
        with p1, p2, mock.patch.object(self.mod, "_orchestrate",
                                       new=mock.AsyncMock(return_value="final answer")):
            out = self.mod._run_task_sync("do it", max_steps=4, timeout=1)
        self.assertEqual(out, "final answer")
        self.assertEqual(self.mod._state["current_task"], "do it")

    def test_run_task_sync_already_running(self):
        running = _ImmediateFuture(result="x")
        running.done = lambda: False
        with mock.patch.object(self.mod, "_start_loop_thread",
                               return_value=object()):
            self.mod._state["current_fut"] = running
            out = self.mod._run_task_sync("second task", timeout=1)
        self.assertIn("already running", out)

    def test_run_task_sync_timeout_message(self):
        class _TimeoutFuture(_ImmediateFuture):
            def result(self, timeout=None):
                raise TimeoutError("still running")

        def _fake_rcts(coro, loop):
            _run_coro(coro)
            return _TimeoutFuture()
        with mock.patch.object(self.mod, "_start_loop_thread",
                               return_value=object()), \
                mock.patch.object(self.mod.asyncio, "run_coroutine_threadsafe",
                                  side_effect=_fake_rcts), \
                mock.patch.object(self.mod, "_orchestrate",
                                  new=mock.AsyncMock(return_value="x")):
            out = self.mod._run_task_sync("long task " * 10, timeout=1)
        self.assertIn("still working", out)
        self.assertIn("browser_status", out)

    def test_run_task_sync_generic_failure(self):
        class _ErrFuture(_ImmediateFuture):
            def result(self, timeout=None):
                raise ValueError("explode")

        def _fake_rcts(coro, loop):
            _run_coro(coro)
            return _ErrFuture()
        with mock.patch.object(self.mod, "_start_loop_thread",
                               return_value=object()), \
                mock.patch.object(self.mod.asyncio, "run_coroutine_threadsafe",
                                  side_effect=_fake_rcts), \
                mock.patch.object(self.mod, "_orchestrate",
                                  new=mock.AsyncMock(return_value="x")):
            out = self.mod._run_task_sync("task", timeout=1)
        self.assertIn("Browser agent failed", out)

    def test_submit_done_callback_records_result(self):
        p1, p2 = self._patch_loop(result="captured")
        with p1, p2, mock.patch.object(self.mod, "_orchestrate",
                                       new=mock.AsyncMock(return_value="captured")):
            fut = self.mod._submit("task", max_steps=2, headless=False)
        # The done callback fired synchronously and stored last_result.
        self.assertEqual(self.mod._state["last_result"], "captured")
        self.assertTrue(fut.done())

    def test_submit_done_callback_records_error(self):
        p1, p2 = self._patch_loop(exc=RuntimeError("run blew up"))
        with p1, p2, mock.patch.object(self.mod, "_orchestrate",
                                       new=mock.AsyncMock(return_value="x")):
            self.mod._submit("task", max_steps=2, headless=False)
        self.assertIn("run blew up", self.mod._state["last_error"])


class ActionFlowTests(_BrowserAgentTestBase):
    """Action bodies past the guard, with is_available True and the runner
    stubbed so the natural-language task is captured but nothing real runs."""

    @contextlib.contextmanager
    def _available(self, capture):
        def _fake_run(task, **kw):
            capture["task"] = task
            capture["kw"] = kw
            return "RAN: " + task
        with mock.patch.object(self.mod, "is_available", return_value=True), \
                mock.patch.object(self.mod, "_ssrf_guard", return_value=None), \
                mock.patch.object(self.mod, "_run_task_sync", side_effect=_fake_run):
            yield

    def test_browser_task_runs(self):
        cap = {}
        with self._available(cap):
            out = self.actions["browser_task"]("find me a flight")
        self.assertIn("find me a flight", cap["task"])
        self.assertTrue(out.startswith("RAN:"))

    def test_fill_form_empty(self):
        out = self.actions["fill_form"]("")
        self.assertIn("what to fill in", out)

    def test_fill_form_builds_task_and_passes_details_to_guard(self):
        cap = {}
        with mock.patch.object(self.mod, "is_available", return_value=True), \
                mock.patch.object(self.mod, "_ssrf_guard",
                                  return_value=None) as guard, \
                mock.patch.object(self.mod, "_run_task_sync",
                                  side_effect=lambda task, **kw: cap.update(task=task)):
            self.actions["fill_form"]("name=Bob, email=bob@x.com")
        self.assertIn("fill it in with these details", cap["task"])
        self.assertIn("Stop short of final submission", cap["task"])
        guard.assert_called_once_with("name=Bob, email=bob@x.com")

    def test_fill_form_hint_when_unavailable(self):
        with mock.patch.object(self.mod, "is_available", return_value=False):
            out = self.actions["fill_form"]("some details")
        self.assertIn("Browser agent is offline", out)

    def test_find_cheapest_empty(self):
        self.assertIn("bargain-hunt", self.actions["find_cheapest"](""))

    def test_find_cheapest_builds_task(self):
        cap = {}
        with self._available(cap):
            self.actions["find_cheapest"]("RTX 4090")
        self.assertIn("RTX 4090", cap["task"])
        self.assertIn("cheapest available option", cap["task"])

    def test_browse_for_empty(self):
        self.assertIn("look up", self.actions["browse_for"](""))

    def test_book_appointment_empty(self):
        self.assertIn("kind of appointment", self.actions["book_appointment"](""))

    def test_browser_open_empty(self):
        self.assertIn("Give me a URL", self.actions["browser_open"](""))

    def test_browser_open_runs_with_max_steps(self):
        cap = {}
        with self._available(cap):
            out = self.actions["browser_open"]("https://example.com")
        self.assertIn("Navigate to https://example.com", cap["task"])
        self.assertEqual(cap["kw"].get("max_steps"), 3)
        self.assertTrue(out.startswith("RAN:"))

    def test_browser_open_hint_when_unavailable(self):
        with mock.patch.object(self.mod, "is_available", return_value=False):
            out = self.actions["browser_open"]("https://example.com")
        self.assertIn("Browser agent is offline", out)

    def test_browse_for_hint_when_unavailable(self):
        with mock.patch.object(self.mod, "is_available", return_value=False):
            self.assertIn("offline", self.actions["browse_for"]("x"))

    def test_find_cheapest_hint_when_unavailable(self):
        with mock.patch.object(self.mod, "is_available", return_value=False):
            self.assertIn("offline", self.actions["find_cheapest"]("x"))

    def test_book_appointment_hint_when_unavailable(self):
        with mock.patch.object(self.mod, "is_available", return_value=False):
            self.assertIn("offline", self.actions["book_appointment"]("a haircut"))

    def test_browse_for_ssrf_blocks(self):
        with mock.patch.object(self.mod, "is_available", return_value=True), \
                mock.patch.object(self.mod, "_ssrf_guard",
                                  return_value=self.mod._SSRF_REFUSAL), \
                mock.patch.object(self.mod, "_run_task_sync") as runner:
            out = self.actions["browse_for"]("http://localhost/x")
        self.assertEqual(out, self.mod._SSRF_REFUSAL)
        runner.assert_not_called()

    def test_all_llm_actions_ssrf_block(self):
        # browser_task / book_appointment / fill_form / find_cheapest each
        # hit `return blocked` when the guard refuses.
        cases = {
            "browser_task": "go to http://127.0.0.1/x",
            "book_appointment": "at http://10.0.0.1/book",
            "fill_form": "form at http://192.168.1.1/f",
            "find_cheapest": "deal at http://169.254.169.254/x",
        }
        for action, arg in cases.items():
            with self.subTest(action=action):
                with mock.patch.object(self.mod, "is_available",
                                       return_value=True), \
                        mock.patch.object(self.mod, "_ssrf_guard",
                                          return_value=self.mod._SSRF_REFUSAL), \
                        mock.patch.object(self.mod, "_run_task_sync") as runner:
                    out = self.actions[action](arg)
                self.assertEqual(out, self.mod._SSRF_REFUSAL)
                runner.assert_not_called()

    def test_browser_task_unavailable_hint(self):
        with mock.patch.object(self.mod, "is_available", return_value=False):
            self.assertIn("offline", self.actions["browser_task"]("x"))


class StatusActionTests(_BrowserAgentTestBase):
    def test_running_status(self):
        running = _ImmediateFuture(result="x")
        running.done = lambda: False
        self.mod._state.update({
            "current_task": "booking a haircut",
            "current_fut": running,
            "step_count": 3,
            "last_step": "clicked submit",
            "started_at": 1000.0,
        })
        with mock.patch.object(self.mod.time, "time", return_value=1005.0):
            out = self.actions["browser_status"]("")
        self.assertIn("running", out)
        self.assertIn("booking a haircut", out)
        self.assertIn("steps:   3", out)
        self.assertIn("elapsed: 5s", out)
        self.assertIn("clicked submit", out)

    def test_finished_status_with_result_and_error(self):
        done = _ImmediateFuture(result="x")  # done() True
        self.mod._state.update({
            "current_task": "some task",
            "current_fut": done,
            "step_count": 7,
            "started_at": 1000.0,
            "last_result": "the result text",
            "last_error": "an error happened",
        })
        with mock.patch.object(self.mod.time, "time", return_value=1010.0):
            out = self.actions["browser_status"]("")
        self.assertIn("finished", out)
        self.assertIn("the result text", out)
        self.assertIn("an error happened", out)


class StopActionTests(_BrowserAgentTestBase):
    def test_stop_idle_when_future_done(self):
        self.mod._state["current_fut"] = _ImmediateFuture(result="x")  # done
        self.assertIn("No browser task is running",
                      self.actions["browser_stop"](""))

    def test_stop_running_task_calls_agent_and_cancels(self):
        running = _ImmediateFuture(result="x")
        running.done = lambda: False
        agent = _FakeAgent(task="t")
        sentinel_loop = object()
        rcts_calls = []

        def _fake_rcts(coro, loop):
            # The agent.stop() coroutine + _close_browser_async coroutine.
            rcts_calls.append(coro)
            _run_coro(coro)
            return _ImmediateFuture(result=None)
        self.mod._state.update({
            "current_fut": running, "loop": sentinel_loop, "agent": agent,
        })
        with mock.patch.object(self.mod.asyncio, "run_coroutine_threadsafe",
                               side_effect=_fake_rcts):
            out = self.actions["browser_stop"]("")
        self.assertIn("Stopping the browser task", out)
        self.assertTrue(running._cancelled)
        # Both the agent.stop() coro and the cleanup coro were scheduled.
        self.assertGreaterEqual(len(rcts_calls), 1)

    def test_stop_with_sync_agent_stop(self):
        # agent.stop() is a plain (non-coroutine) function → no rcts needed.
        running = _ImmediateFuture(result="x")
        running.done = lambda: False

        class _SyncStopAgent:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True
        agent = _SyncStopAgent()
        self.mod._state.update({
            "current_fut": running, "loop": None, "agent": agent,
        })
        out = self.actions["browser_stop"]("")
        self.assertTrue(agent.stopped)
        self.assertIn("Stopping", out)

    def test_stop_agent_stop_raises_tries_next(self):
        # agent.stop() raises → except/continue → request_stop() used.
        running = _ImmediateFuture(result="x")
        running.done = lambda: False
        calls = []

        class _FlakyAgent:
            def stop(self):
                calls.append("stop")
                raise RuntimeError("stop boom")

            def request_stop(self):
                calls.append("request_stop")
        self.mod._state.update({
            "current_fut": running, "loop": None, "agent": _FlakyAgent(),
        })
        out = self.actions["browser_stop"]("")
        self.assertEqual(calls, ["stop", "request_stop"])
        self.assertIn("Stopping", out)

    def test_stop_cleanup_scheduling_exception_swallowed(self):
        # run_coroutine_threadsafe for the cleanup coro raises → swallowed,
        # still returns the 'Stopping' message.
        running = _ImmediateFuture(result="x")
        running.done = lambda: False
        self.mod._state.update({
            "current_fut": running, "loop": object(), "agent": None,
        })

        def _raise_after_closing(coro, loop):
            # Close the passed coroutine to avoid a 'never awaited' warning,
            # then simulate the loop being gone.
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("loop gone")
        with mock.patch.object(self.mod.asyncio, "run_coroutine_threadsafe",
                               side_effect=_raise_after_closing):
            out = self.actions["browser_stop"]("")
        self.assertIn("Stopping", out)
        self.assertTrue(running._cancelled)


class ScreenshotActionTests(_BrowserAgentTestBase):
    def test_unavailable_returns_hint(self):
        with mock.patch.object(self.mod, "is_available", return_value=False):
            self.assertIn("offline", self.actions["browser_screenshot"](""))

    def test_no_browser_running(self):
        with mock.patch.object(self.mod, "is_available", return_value=True):
            self.mod._state["browser"] = None
            out = self.actions["browser_screenshot"]("")
        self.assertIn("Browser is not running", out)

    def test_no_loop_running(self):
        with mock.patch.object(self.mod, "is_available", return_value=True), \
                mock.patch.object(self.mod.os, "makedirs"):
            self.mod._state["browser"] = _FakeBrowser()
            self.mod._state["loop"] = None
            out = self.actions["browser_screenshot"]("")
        self.assertIn("loop is not running", out)

    def test_screenshot_success(self):
        class _Page:
            def __init__(self):
                self.kw = None

            async def screenshot(self, **kw):
                self.kw = kw
        page = _Page()
        browser = _FakeBrowser()
        browser._page = page
        browser.page = page  # attr the skill looks for
        sentinel_loop = object()

        def _fake_rcts(coro, loop):
            res = _run_coro(coro)
            return _ImmediateFuture(result=res)
        with mock.patch.object(self.mod, "is_available", return_value=True), \
                mock.patch.object(self.mod.os, "makedirs"), \
                mock.patch.object(self.mod.asyncio, "run_coroutine_threadsafe",
                                  side_effect=_fake_rcts):
            self.mod._state["browser"] = browser
            self.mod._state["loop"] = sentinel_loop
            out = self.actions["browser_screenshot"]("")
        self.assertIn("Saved screenshot", out)
        self.assertTrue(out.strip().endswith(", sir."))
        self.assertEqual(page.kw.get("full_page"), True)

    def test_screenshot_uses_get_current_page_async(self):
        class _Page:
            async def screenshot(self, **kw):
                return None

        class _GetterBrowser:
            async def get_current_page(self):
                return _Page()

        def _fake_rcts(coro, loop):
            return _ImmediateFuture(result=_run_coro(coro))
        with mock.patch.object(self.mod, "is_available", return_value=True), \
                mock.patch.object(self.mod.os, "makedirs"), \
                mock.patch.object(self.mod.asyncio, "run_coroutine_threadsafe",
                                  side_effect=_fake_rcts):
            self.mod._state["browser"] = _GetterBrowser()
            self.mod._state["loop"] = object()
            out = self.actions["browser_screenshot"]("")
        self.assertIn("Saved screenshot", out)

    def test_screenshot_no_page_raises_handled(self):
        class _NoPageBrowser:
            pass

        def _fake_rcts(coro, loop):
            # Real run_coroutine_threadsafe captures the coroutine's exception
            # into the future; mirror that so .result() re-raises it.
            try:
                return _ImmediateFuture(result=_run_coro(coro))
            except Exception as e:
                return _ImmediateFuture(exc=e)
        with mock.patch.object(self.mod, "is_available", return_value=True), \
                mock.patch.object(self.mod.os, "makedirs"), \
                mock.patch.object(self.mod.asyncio, "run_coroutine_threadsafe",
                                  side_effect=_fake_rcts):
            self.mod._state["browser"] = _NoPageBrowser()
            self.mod._state["loop"] = object()
            out = self.actions["browser_screenshot"]("")
        self.assertIn("Screenshot failed", out)

    def test_screenshot_future_exception(self):
        def _fake_rcts(coro, loop):
            coro.close()  # avoid "never awaited" warning
            return _ImmediateFuture(exc=RuntimeError("shot exploded"))
        with mock.patch.object(self.mod, "is_available", return_value=True), \
                mock.patch.object(self.mod.os, "makedirs"), \
                mock.patch.object(self.mod.asyncio, "run_coroutine_threadsafe",
                                  side_effect=_fake_rcts):
            self.mod._state["browser"] = _FakeBrowser()
            self.mod._state["loop"] = object()
            out = self.actions["browser_screenshot"]("")
        self.assertIn("Screenshot failed", out)


class ResetProfileActionTests(_BrowserAgentTestBase):
    def test_blocks_while_task_running(self):
        running = _ImmediateFuture(result="x")
        running.done = lambda: False
        self.mod._state["current_fut"] = running
        out = self.actions["browser_reset_profile"]("")
        self.assertIn("Stop the running browser task first", out)

    def test_already_clean_when_dir_absent(self):
        self.mod._state["current_fut"] = None
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            out = self.actions["browser_reset_profile"]("")
        self.assertIn("already clean", out)

    def test_wipes_profile(self):
        self.mod._state["current_fut"] = None
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
                mock.patch.object(self.mod.shutil, "rmtree") as rm, \
                mock.patch.object(self.mod.os, "makedirs") as mk:
            out = self.actions["browser_reset_profile"]("")
        rm.assert_called_once()
        mk.assert_called_once()
        self.assertIn("wiped", out)

    def test_wipe_failure_reported(self):
        self.mod._state["current_fut"] = None
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
                mock.patch.object(self.mod.shutil, "rmtree",
                                  side_effect=OSError("locked")):
            out = self.actions["browser_reset_profile"]("")
        self.assertIn("Could not wipe the profile", out)


class RegisterTests(_BrowserAgentTestBase):
    def test_registers_all_actions(self):
        acts: dict = {}
        with mock.patch.object(self.mod, "_bu_imports", return_value={}):
            self.mod.register(acts)
        for name in ("browser_task", "browser_do", "browser_run",
                     "book_appointment", "fill_form", "browse_for",
                     "find_cheapest", "browser_open", "browser_screenshot",
                     "browser_status", "browser_stop", "browser_reset_profile"):
            self.assertIn(name, acts)

    def test_register_does_not_overwrite_existing(self):
        sentinel = lambda a="": "pre-existing"
        acts = {"browser_task": sentinel}
        with mock.patch.object(self.mod, "_bu_imports", return_value={}):
            self.mod.register(acts)
        self.assertIs(acts["browser_task"], sentinel)

    def test_register_quiet_when_no_agent(self):
        # browser-use 'absent' → early return, no profile dir creation.
        with mock.patch.object(self.mod, "_bu_imports", return_value={}), \
                mock.patch.object(self.mod.os, "makedirs") as mk:
            self.mod.register({})
        mk.assert_not_called()

    def test_register_warns_without_api_key(self):
        with mock.patch.object(self.mod, "_bu_imports", return_value=_imports()), \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(self.mod.os, "makedirs") as mk, \
                mock.patch("builtins.print") as pr:
            self.mod.register({})
        mk.assert_not_called()
        # A hint about the missing key is printed.
        printed = " ".join(str(c) for c in pr.call_args_list)
        self.assertIn("ANTHROPIC_API_KEY", printed)

    def test_register_ready_with_key(self):
        with mock.patch.object(self.mod, "_bu_imports", return_value=_imports()), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}), \
                mock.patch.object(self.mod, "_model_name", return_value="m"), \
                mock.patch.object(self.mod.os, "makedirs") as mk, \
                mock.patch("builtins.print") as pr:
            self.mod.register({})
        mk.assert_called_once()  # profile dir ensured
        printed = " ".join(str(c) for c in pr.call_args_list)
        self.assertIn("ready", printed)


class StartLoopThreadTests(_BrowserAgentTestBase):
    def test_returns_existing_open_loop(self):
        class _Loop:
            def is_closed(self):
                return False
        existing = _Loop()
        self.mod._state["loop"] = existing
        self.assertIs(self.mod._start_loop_thread(), existing)

    def test_starts_new_loop_thread(self):
        # Fake Thread actually RUNS the target synchronously so the daemon
        # body (_run: set_event_loop → ready.set → run_forever → close) is
        # exercised — but the fake loop's run_forever returns immediately, so
        # nothing blocks and no real thread/loop is created.
        self.mod._state["loop"] = None
        fake_loop = mock.MagicMock(name="loop")
        fake_loop.is_closed.return_value = False
        fake_loop.run_forever.return_value = None  # returns at once

        started = {}

        class _FakeThread:
            def __init__(self, target=None, name=None, daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon

            def start(self):
                started["ran"] = True
                self.target()  # drive the real _run body inline

        with mock.patch.object(self.mod.asyncio, "new_event_loop",
                               return_value=fake_loop), \
                mock.patch.object(self.mod.asyncio, "set_event_loop") as setloop, \
                mock.patch.object(self.mod.threading, "Thread", _FakeThread):
            loop = self.mod._start_loop_thread()
        self.assertIs(loop, fake_loop)
        self.assertTrue(started["ran"])
        self.assertIs(self.mod._state["loop"], fake_loop)
        setloop.assert_called_once_with(fake_loop)
        fake_loop.run_forever.assert_called_once()
        fake_loop.close.assert_called_once()

    def test_loop_run_body_closes_even_if_close_raises(self):
        # The finally-block in _run swallows a close() error.
        self.mod._state["loop"] = None
        fake_loop = mock.MagicMock(name="loop")
        fake_loop.is_closed.return_value = False
        fake_loop.run_forever.return_value = None
        fake_loop.close.side_effect = RuntimeError("close failed")

        class _FakeThread:
            def __init__(self, target=None, name=None, daemon=None):
                self.target = target

            def start(self):
                self.target()  # must not raise despite close() failing

        with mock.patch.object(self.mod.asyncio, "new_event_loop",
                               return_value=fake_loop), \
                mock.patch.object(self.mod.asyncio, "set_event_loop"), \
                mock.patch.object(self.mod.threading, "Thread", _FakeThread):
            loop = self.mod._start_loop_thread()
        self.assertIs(loop, fake_loop)

    def test_raises_when_thread_never_ready(self):
        self.mod._state["loop"] = None
        fake_loop = mock.MagicMock()

        class _NeverReadyEvent:
            def set(self):
                pass

            def wait(self, timeout=None):
                return False  # thread failed to signal readiness

        class _FakeThread:
            def __init__(self, **kw):
                pass

            def start(self):
                pass

        with mock.patch.object(self.mod.asyncio, "new_event_loop",
                               return_value=fake_loop), \
                mock.patch.object(self.mod.threading, "Thread", _FakeThread), \
                mock.patch.object(self.mod.threading, "Event", _NeverReadyEvent):
            with self.assertRaises(RuntimeError):
                self.mod._start_loop_thread()


if __name__ == "__main__":
    unittest.main()
