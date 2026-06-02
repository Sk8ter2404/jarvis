"""Thorough unit tests for ``core.mcp_client`` — the Model Context Protocol
client manager.

The real module owns a daemon thread running an asyncio event loop, spawns MCP
servers via subprocess/sockets, and bridges sync→async with
``asyncio.run_coroutine_threadsafe``. None of that may actually happen in a
test: every external (the ``mcp`` SDK, the bg loop, ``run_coroutine_threadsafe``,
real threads/timers, the on-disk ``mcp_servers.json``, ``time``) is mocked or
neutered so the suite is fully deterministic and offline.

Strategy
--------
* ``_mcp_imports()`` — the SDK gate — is patched to return a controllable dict
  of fakes (or ``{}`` to simulate "SDK not installed").
* ``_read_config()`` is patched per-test (or the ``_CONFIG_PATH`` constant is
  redirected at a temp file) to control the server config.
* The bg loop is replaced with a ``FakeLoop`` and ``run_coroutine_threadsafe``
  with a fake that runs the coroutine inline (or fails on demand), so no real
  loop/thread is needed for the sync facade.
* The pure-async helpers (``_server_task``, ``_spawn_server``, ``_wait_event``)
  ARE exercised for real via ``asyncio.run`` against in-memory fakes — that is
  the cleanest way to cover their happy/sad branches.
* Module-global ``_state`` is reset around every test for isolation.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import sys
import unittest
from unittest import mock

from core import mcp_client as m


# ──────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────
class FakeTool:
    """Stand-in for an ``mcp.types.Tool``."""

    def __init__(self, name, description="", inputSchema=None):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema if inputSchema is not None else {}


class FakeContent:
    """A single content part of a ``CallToolResult``."""

    def __init__(self, type=None, text=None):
        self.type = type
        self.text = text


class FakeCallResult:
    """Stand-in for an ``mcp.types.CallToolResult``."""

    def __init__(self, content=None, isError=False):
        self.content = content if content is not None else []
        self.isError = isError


class FakeLoop:
    """Minimal asyncio-loop stand-in for the sync facade.

    Records ``call_soon_threadsafe`` invocations and (by default) runs the
    callbacks immediately so ``shutdown()`` event-sets and ``fut.cancel``
    calls take effect deterministically.
    """

    def __init__(self, closed=False, run_callbacks=True):
        self._closed = closed
        self.run_callbacks = run_callbacks
        self.soon_calls = []

    def is_closed(self):
        return self._closed

    def call_soon_threadsafe(self, fn, *args):
        self.soon_calls.append((fn, args))
        if self.run_callbacks:
            fn(*args)


class FakeFuture:
    """Stand-in for the ``concurrent.futures.Future`` that
    ``run_coroutine_threadsafe`` returns."""

    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.cancelled = False

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._result

    def cancel(self):
        self.cancelled = True


def _make_imports(**over):
    """A complete fake ``_mcp_imports()`` dict; override pieces as needed."""
    d = {
        "ClientSession":          mock.MagicMock(name="ClientSession"),
        "stdio_client":           mock.MagicMock(name="stdio_client"),
        "StdioServerParameters":  mock.MagicMock(name="StdioServerParameters"),
        "sse_client":             mock.MagicMock(name="sse_client"),
        "streamable_http_client": mock.MagicMock(name="streamable_http_client"),
    }
    d.update(over)
    return d


# ──────────────────────────────────────────────────────────────────────
# Base: reset global _state around every test
# ──────────────────────────────────────────────────────────────────────
class _StateIsolatedTest(unittest.TestCase):
    def setUp(self):
        self._saved_state = copy.deepcopy(m._state)
        # Start every test from a clean, fully-reset state.
        m._state.update({
            "loop":            None,
            "thread":          None,
            "sessions":        {},
            "tools":           {},
            "tool_index":      {},
            "errors":          {},
            "shutdown_events": {},
            "started_at":      None,
            "bootstrapped":    False,
        })

    def tearDown(self):
        # Restore the exact pre-test state (deepcopy-restored) so nothing leaks.
        m._state.clear()
        m._state.update(self._saved_state)


# ──────────────────────────────────────────────────────────────────────
# _mcp_imports — lazy SDK import with fallbacks
# ──────────────────────────────────────────────────────────────────────
class McpImportsTests(unittest.TestCase):
    """``_mcp_imports`` walks several import paths; drive each branch by
    patching ``sys.modules`` so ``from mcp import X`` resolves to fakes."""

    def _fake_mcp_module(self, **attrs):
        mod = mock.MagicMock(name="mcp")
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    def test_returns_empty_when_sdk_absent(self):
        # No 'mcp' anywhere → every import raises → {}.
        with mock.patch.dict(sys.modules, {}, clear=False):
            for k in list(sys.modules):
                if k == "mcp" or k.startswith("mcp."):
                    sys.modules.pop(k, None)
            # Block re-import from disk so we deterministically get {}.
            with mock.patch.dict(sys.modules, {"mcp": None}):
                self.assertEqual(m._mcp_imports(), {})

    def test_happy_path_primary_imports(self):
        cs = object()
        stdio = object()
        ssp = object()
        sse = object()
        sh = object()
        mcp_mod = self._fake_mcp_module(ClientSession=cs, StdioServerParameters=ssp)
        stdio_mod = self._fake_mcp_module(stdio_client=stdio, StdioServerParameters=ssp)
        sse_mod = self._fake_mcp_module(sse_client=sse)
        shttp_mod = self._fake_mcp_module(streamablehttp_client=sh)
        with mock.patch.dict(sys.modules, {
            "mcp": mcp_mod,
            "mcp.client": mock.MagicMock(),
            "mcp.client.stdio": stdio_mod,
            "mcp.client.sse": sse_mod,
            "mcp.client.streamable_http": shttp_mod,
        }):
            out = m._mcp_imports()
        self.assertIs(out["ClientSession"], cs)
        self.assertIs(out["stdio_client"], stdio)
        self.assertIs(out["StdioServerParameters"], ssp)
        self.assertIs(out["sse_client"], sse)
        self.assertIs(out["streamable_http_client"], sh)

    def test_clientsession_fallback_to_session_submodule(self):
        # `from mcp import ClientSession` fails, but
        # `from mcp.client.session import ClientSession` works.
        cs = object()
        ssp = object()
        stdio = object()
        mcp_mod = self._fake_mcp_module(StdioServerParameters=ssp)
        del mcp_mod.ClientSession  # ensure attribute import raises ImportError
        session_mod = self._fake_mcp_module(ClientSession=cs)
        stdio_mod = self._fake_mcp_module(stdio_client=stdio, StdioServerParameters=ssp)
        with mock.patch.dict(sys.modules, {
            "mcp": mcp_mod,
            "mcp.client": mock.MagicMock(),
            "mcp.client.session": session_mod,
            "mcp.client.stdio": stdio_mod,
            "mcp.client.sse": mock.MagicMock(spec=[]),
            "mcp.client.streamable_http": mock.MagicMock(spec=[]),
        }):
            out = m._mcp_imports()
        self.assertIs(out["ClientSession"], cs)

    def test_returns_empty_when_clientsession_unavailable_everywhere(self):
        mcp_mod = self._fake_mcp_module()
        del mcp_mod.ClientSession
        # mcp.client.session has no ClientSession either.
        session_mod = mock.MagicMock(spec=[])
        with mock.patch.dict(sys.modules, {
            "mcp": mcp_mod,
            "mcp.client": mock.MagicMock(),
            "mcp.client.session": session_mod,
        }):
            self.assertEqual(m._mcp_imports(), {})

    def test_stdio_fallback_split_import(self):
        # First `from mcp.client.stdio import stdio_client, StdioServerParameters`
        # raises (no StdioServerParameters in that submodule), but the split
        # form succeeds.
        cs = object()
        stdio = object()
        ssp = object()
        mcp_mod = self._fake_mcp_module(ClientSession=cs, StdioServerParameters=ssp)
        stdio_mod = self._fake_mcp_module(stdio_client=stdio)
        del stdio_mod.StdioServerParameters
        with mock.patch.dict(sys.modules, {
            "mcp": mcp_mod,
            "mcp.client": mock.MagicMock(),
            "mcp.client.stdio": stdio_mod,
            "mcp.client.sse": mock.MagicMock(spec=[]),
            "mcp.client.streamable_http": mock.MagicMock(spec=[]),
        }):
            out = m._mcp_imports()
        self.assertIs(out["stdio_client"], stdio)
        self.assertIs(out["StdioServerParameters"], ssp)

    def test_returns_empty_when_stdio_unavailable(self):
        cs = object()
        mcp_mod = self._fake_mcp_module(ClientSession=cs)
        # mcp.client.stdio missing stdio_client entirely.
        stdio_mod = mock.MagicMock(spec=[])
        with mock.patch.dict(sys.modules, {
            "mcp": mcp_mod,
            "mcp.client": mock.MagicMock(),
            "mcp.client.stdio": stdio_mod,
        }):
            self.assertEqual(m._mcp_imports(), {})

    def test_sse_and_http_optional_left_none(self):
        # Primary CS + stdio import succeed; sse/http submodules missing →
        # those keys are present but None (transports unsupported).
        cs = object()
        stdio = object()
        ssp = object()
        mcp_mod = self._fake_mcp_module(ClientSession=cs, StdioServerParameters=ssp)
        stdio_mod = self._fake_mcp_module(stdio_client=stdio, StdioServerParameters=ssp)
        with mock.patch.dict(sys.modules, {
            "mcp": mcp_mod,
            "mcp.client": mock.MagicMock(),
            "mcp.client.stdio": stdio_mod,
            "mcp.client.sse": mock.MagicMock(spec=[]),
            "mcp.client.streamable_http": mock.MagicMock(spec=[]),
        }):
            out = m._mcp_imports()
        self.assertIsNone(out["sse_client"])
        self.assertIsNone(out["streamable_http_client"])


# ──────────────────────────────────────────────────────────────────────
# is_available
# ──────────────────────────────────────────────────────────────────────
class IsAvailableTests(_StateIsolatedTest):
    def test_false_when_sdk_missing(self):
        with mock.patch.object(m, "_mcp_imports", return_value={}):
            self.assertFalse(m.is_available())

    def test_false_when_no_config(self):
        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value={}):
            self.assertFalse(m.is_available())

    def test_false_when_all_servers_disabled(self):
        cfg = {"fs": {"enabled": False}, "gh": {"enabled": False}}
        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value=cfg):
            self.assertFalse(m.is_available())

    def test_true_when_one_enabled_server(self):
        cfg = {"fs": {"enabled": False}, "gh": {"command": "npx"}}  # gh defaults enabled
        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value=cfg):
            self.assertTrue(m.is_available())

    def test_ignores_non_dict_entries(self):
        # A stray non-dict value must not crash the any() generator.
        cfg = {"weird": "not-a-dict", "gh": {"command": "npx"}}
        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value=cfg):
            self.assertTrue(m.is_available())

    def test_handles_none_server_value(self):
        # `(s or {}).get(...)` guards None entries.
        cfg = {"fs": None}
        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value=cfg):
            # None is filtered by isinstance(s, dict) → no enabled → False.
            self.assertFalse(m.is_available())


# ──────────────────────────────────────────────────────────────────────
# _read_config
# ──────────────────────────────────────────────────────────────────────
class ReadConfigTests(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        with mock.patch.object(m.os.path, "exists", return_value=False):
            self.assertEqual(m._read_config(), {})

    def test_valid_plain_config(self):
        data = {"fs": {"command": "npx"}}
        with mock.patch.object(m.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(data))):
            self.assertEqual(m._read_config(), data)

    def test_unwraps_mcpServers_key(self):
        inner = {"fs": {"command": "npx"}}
        data = {"mcpServers": inner}
        with mock.patch.object(m.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(data))):
            self.assertEqual(m._read_config(), inner)

    def test_mcpServers_non_dict_not_unwrapped(self):
        data = {"mcpServers": ["x"], "fs": {"command": "npx"}}
        with mock.patch.object(m.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(data))):
            # mcpServers present but not a dict → returned as-is.
            self.assertEqual(m._read_config(), data)

    def test_malformed_json_returns_empty(self):
        with mock.patch.object(m.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{not json")):
            with self.assertLogs("jarvis.mcp", level="WARNING"):
                self.assertEqual(m._read_config(), {})

    def test_non_dict_toplevel_returns_empty(self):
        with mock.patch.object(m.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="[1, 2, 3]")):
            self.assertEqual(m._read_config(), {})

    def test_open_raises_returns_empty(self):
        with mock.patch.object(m.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("boom")):
            with self.assertLogs("jarvis.mcp", level="WARNING"):
                self.assertEqual(m._read_config(), {})


# ──────────────────────────────────────────────────────────────────────
# _start_loop_thread
# ──────────────────────────────────────────────────────────────────────
class StartLoopThreadTests(_StateIsolatedTest):
    def test_returns_existing_open_loop(self):
        existing = FakeLoop(closed=False)
        m._state["loop"] = existing
        # Must NOT create a new loop/thread.
        with mock.patch.object(m.asyncio, "new_event_loop") as new_loop, \
             mock.patch.object(m.threading, "Thread") as thread_cls:
            got = m._start_loop_thread()
        self.assertIs(got, existing)
        new_loop.assert_not_called()
        thread_cls.assert_not_called()

    def test_creates_loop_and_thread_when_none(self):
        fake_loop = mock.MagicMock(name="newloop")
        # Fake Thread whose start() flips the ready Event (the target sets it,
        # but we never actually run the target — so do it from start()).
        created = {}

        def fake_thread(target=None, name=None, daemon=None):
            created["target"] = target
            created["name"] = name
            created["daemon"] = daemon
            t = mock.MagicMock(name="thread")

            def _start():
                # Emulate the run() body's `ready.set()` without run_forever.
                created["target_ran"] = True
            t.start.side_effect = _start
            return t

        # The ready Event must report set() True so the 5s wait passes. We make
        # Event.wait return True regardless.
        with mock.patch.object(m.asyncio, "new_event_loop", return_value=fake_loop), \
             mock.patch.object(m.threading, "Thread", side_effect=fake_thread), \
             mock.patch.object(m.threading, "Event") as event_cls:
            ev = mock.MagicMock()
            ev.wait.return_value = True
            event_cls.return_value = ev
            got = m._start_loop_thread()

        self.assertIs(got, fake_loop)
        self.assertIs(m._state["loop"], fake_loop)
        self.assertEqual(created["name"], "mcp-asyncio")
        self.assertTrue(created["daemon"])

    def test_run_target_body_executes(self):
        """Cover the ``_run`` closure (the daemon thread body) WITHOUT spawning
        a real thread: capture the ``target`` callable handed to Thread and
        invoke it inline against a fake loop whose ``run_forever`` returns at
        once, so it falls through to the ``finally: close()`` cleanup."""
        fake_loop = mock.MagicMock(name="newloop")
        fake_loop.run_forever.return_value = None     # returns immediately
        captured = {}

        def fake_thread(target=None, name=None, daemon=None):
            captured["target"] = target
            return mock.MagicMock(name="thread")

        with mock.patch.object(m.asyncio, "new_event_loop", return_value=fake_loop), \
             mock.patch.object(m.asyncio, "set_event_loop") as set_loop, \
             mock.patch.object(m.threading, "Thread", side_effect=fake_thread), \
             mock.patch.object(m.threading, "Event") as event_cls:
            ev = mock.MagicMock()
            ev.wait.return_value = True
            event_cls.return_value = ev
            m._start_loop_thread()
            # Now run the captured thread body explicitly.
            captured["target"]()

        set_loop.assert_called_once_with(fake_loop)
        ev.set.assert_called()                 # ready.set() inside _run
        fake_loop.run_forever.assert_called_once()
        fake_loop.close.assert_called_once()   # finally branch

    def test_run_target_body_swallows_close_error(self):
        """The ``_run`` finally-block swallows a ``close()`` exception."""
        fake_loop = mock.MagicMock(name="newloop")
        fake_loop.run_forever.return_value = None
        fake_loop.close.side_effect = RuntimeError("close boom")
        captured = {}

        def fake_thread(target=None, name=None, daemon=None):
            captured["target"] = target
            return mock.MagicMock(name="thread")

        with mock.patch.object(m.asyncio, "new_event_loop", return_value=fake_loop), \
             mock.patch.object(m.asyncio, "set_event_loop"), \
             mock.patch.object(m.threading, "Thread", side_effect=fake_thread), \
             mock.patch.object(m.threading, "Event") as event_cls:
            ev = mock.MagicMock()
            ev.wait.return_value = True
            event_cls.return_value = ev
            m._start_loop_thread()
            # Must not raise despite close() raising.
            captured["target"]()
        fake_loop.close.assert_called_once()

    def test_raises_when_loop_never_ready(self):
        fake_loop = mock.MagicMock(name="newloop")

        def fake_thread(target=None, name=None, daemon=None):
            return mock.MagicMock(name="thread")

        with mock.patch.object(m.asyncio, "new_event_loop", return_value=fake_loop), \
             mock.patch.object(m.threading, "Thread", side_effect=fake_thread), \
             mock.patch.object(m.threading, "Event") as event_cls:
            ev = mock.MagicMock()
            ev.wait.return_value = False     # never signalled within timeout
            event_cls.return_value = ev
            with self.assertRaises(RuntimeError):
                m._start_loop_thread()

    def test_recreates_when_existing_loop_closed(self):
        m._state["loop"] = FakeLoop(closed=True)
        fake_loop = mock.MagicMock(name="newloop")
        with mock.patch.object(m.asyncio, "new_event_loop", return_value=fake_loop), \
             mock.patch.object(m.threading, "Thread") as thread_cls, \
             mock.patch.object(m.threading, "Event") as event_cls:
            ev = mock.MagicMock()
            ev.wait.return_value = True
            event_cls.return_value = ev
            thread_cls.return_value = mock.MagicMock()
            got = m._start_loop_thread()
        self.assertIs(got, fake_loop)


# ──────────────────────────────────────────────────────────────────────
# _make_transport_cm
# ──────────────────────────────────────────────────────────────────────
class MakeTransportTests(unittest.TestCase):
    def test_stdio_basic(self):
        imports = _make_imports()
        imports["StdioServerParameters"].return_value = "PARAMS"
        imports["stdio_client"].return_value = "STDIO_CM"
        cfg = {"transport": "stdio", "command": "npx", "args": ["-y", "pkg"]}
        out = m._make_transport_cm(imports, "fs", cfg)
        self.assertEqual(out, "STDIO_CM")
        imports["stdio_client"].assert_called_once_with("PARAMS")
        # No env supplied → env kwarg omitted.
        _, kwargs = imports["StdioServerParameters"].call_args
        self.assertNotIn("env", kwargs)
        self.assertEqual(kwargs["command"], "npx")
        self.assertEqual(kwargs["args"], ["-y", "pkg"])

    def test_stdio_defaults_when_no_transport_key(self):
        imports = _make_imports()
        cfg = {"command": "npx"}  # transport omitted → defaults to stdio
        m._make_transport_cm(imports, "fs", cfg)
        imports["stdio_client"].assert_called_once()

    def test_stdio_missing_command_raises(self):
        imports = _make_imports()
        with self.assertRaises(ValueError):
            m._make_transport_cm(imports, "fs", {"transport": "stdio"})

    def test_stdio_merges_env_over_os_environ(self):
        imports = _make_imports()
        cfg = {"transport": "stdio", "command": "npx",
               "env": {"FOO": "bar", "NUM": 5}}
        with mock.patch.dict(m.os.environ, {"EXISTING": "1"}, clear=True):
            m._make_transport_cm(imports, "fs", cfg)
        _, kwargs = imports["StdioServerParameters"].call_args
        env = kwargs["env"]
        self.assertEqual(env["EXISTING"], "1")     # inherited from os.environ
        self.assertEqual(env["FOO"], "bar")        # override merged
        self.assertEqual(env["NUM"], "5")          # coerced to str

    def test_stdio_typeerror_drops_env_and_retries(self):
        imports = _make_imports()
        # First call (with env) raises TypeError → retried without env.
        ssp = mock.MagicMock()
        ssp.side_effect = [TypeError("no env kwarg"), "PARAMS"]
        imports["StdioServerParameters"] = ssp
        cfg = {"transport": "stdio", "command": "npx", "env": {"A": "B"}}
        m._make_transport_cm(imports, "fs", cfg)
        self.assertEqual(ssp.call_count, 2)
        _, second_kwargs = ssp.call_args_list[1]
        self.assertNotIn("env", second_kwargs)

    def test_stdio_coerces_args_to_str(self):
        imports = _make_imports()
        cfg = {"transport": "stdio", "command": "npx", "args": [1, 2.5, "x"]}
        m._make_transport_cm(imports, "fs", cfg)
        _, kwargs = imports["StdioServerParameters"].call_args
        self.assertEqual(kwargs["args"], ["1", "2.5", "x"])

    def test_sse_basic(self):
        imports = _make_imports()
        imports["sse_client"].return_value = "SSE_CM"
        cfg = {"transport": "sse", "url": "http://x/sse",
               "headers": {"Authorization": "Bearer z"}}
        out = m._make_transport_cm(imports, "brave", cfg)
        self.assertEqual(out, "SSE_CM")
        imports["sse_client"].assert_called_once_with(
            url="http://x/sse", headers={"Authorization": "Bearer z"})

    def test_sse_unsupported_raises(self):
        imports = _make_imports(sse_client=None)
        with self.assertRaises(RuntimeError):
            m._make_transport_cm(imports, "brave", {"transport": "sse", "url": "x"})

    def test_sse_missing_url_raises(self):
        imports = _make_imports()
        with self.assertRaises(ValueError):
            m._make_transport_cm(imports, "brave", {"transport": "sse"})

    def test_sse_typeerror_fallback_positional(self):
        imports = _make_imports()
        sse = mock.MagicMock()
        sse.side_effect = [TypeError("old sig"), "SSE_CM"]
        imports["sse_client"] = sse
        cfg = {"transport": "sse", "url": "http://x/sse"}
        out = m._make_transport_cm(imports, "brave", cfg)
        self.assertEqual(out, "SSE_CM")
        sse.assert_called_with("http://x/sse")  # positional fallback

    def test_http_basic(self):
        imports = _make_imports()
        imports["streamable_http_client"].return_value = "HTTP_CM"
        for transport in ("http", "streamable_http", "streamable-http"):
            imports["streamable_http_client"].reset_mock()
            cfg = {"transport": transport, "url": "http://x/mcp"}
            out = m._make_transport_cm(imports, "srv", cfg)
            self.assertEqual(out, "HTTP_CM")

    def test_http_unsupported_raises(self):
        imports = _make_imports(streamable_http_client=None)
        with self.assertRaises(RuntimeError):
            m._make_transport_cm(imports, "srv", {"transport": "http", "url": "x"})

    def test_http_missing_url_raises(self):
        imports = _make_imports()
        with self.assertRaises(ValueError):
            m._make_transport_cm(imports, "srv", {"transport": "http"})

    def test_http_typeerror_fallback_positional(self):
        imports = _make_imports()
        sh = mock.MagicMock()
        sh.side_effect = [TypeError("old sig"), "HTTP_CM"]
        imports["streamable_http_client"] = sh
        out = m._make_transport_cm(imports, "srv", {"transport": "http", "url": "u"})
        self.assertEqual(out, "HTTP_CM")
        sh.assert_called_with("u")

    def test_unknown_transport_raises(self):
        imports = _make_imports()
        with self.assertRaises(ValueError):
            m._make_transport_cm(imports, "x", {"transport": "carrier-pigeon"})

    def test_transport_case_insensitive(self):
        imports = _make_imports()
        m._make_transport_cm(imports, "fs", {"transport": "STDIO", "command": "npx"})
        imports["stdio_client"].assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# Async helpers: _wait_event, _spawn_server, _server_task
# ──────────────────────────────────────────────────────────────────────
class AsyncCtx:
    """A simple async context manager yielding a preset value (or raising)."""

    def __init__(self, value=None, enter_exc=None):
        self._value = value
        self._enter_exc = enter_exc

    async def __aenter__(self):
        if self._enter_exc is not None:
            raise self._enter_exc
        return self._value

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Async-context-manager ClientSession double."""

    def __init__(self, tools=None, init_exc=None):
        self._tools = tools if tools is not None else []
        self._init_exc = init_exc
        self.initialized = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        if self._init_exc is not None:
            raise self._init_exc
        self.initialized = True

    async def list_tools(self):
        r = mock.MagicMock()
        r.tools = self._tools
        return r


class WaitEventTests(_StateIsolatedTest):
    def test_returns_true_when_set(self):
        async def go():
            ev = asyncio.Event()
            ev.set()
            return await m._wait_event(ev, timeout=1.0)
        self.assertTrue(asyncio.run(go()))

    def test_returns_false_on_timeout(self):
        async def go():
            ev = asyncio.Event()  # never set
            return await m._wait_event(ev, timeout=0.01)
        self.assertFalse(asyncio.run(go()))


class ServerTaskTests(_StateIsolatedTest):
    def _run_task(self, imports, name, cfg, set_shutdown_after_ready=True):
        async def go():
            ready = asyncio.Event()
            shutdown = asyncio.Event()
            task = asyncio.ensure_future(
                m._server_task(imports, name, cfg, ready, shutdown))
            # Wait for the task to publish readiness (success OR failure both
            # set `ready`). Bounded so a bug can't hang the suite.
            await asyncio.wait_for(ready.wait(), timeout=1.0)
            if set_shutdown_after_ready:
                shutdown.set()
            await asyncio.wait_for(task, timeout=1.0)
        asyncio.run(go())

    def test_happy_path_publishes_session_and_tools(self):
        tools = [FakeTool("read_file"), FakeTool("write_file")]
        session = FakeSession(tools=tools)
        imports = _make_imports(ClientSession=lambda r, w: session)
        # transport yields a 2-tuple (read, write).
        with mock.patch.object(m, "_make_transport_cm",
                               return_value=AsyncCtx(value=("R", "W"))):
            self._run_task(imports, "fs", {"transport": "stdio", "command": "x"})
        self.assertIn("fs", m._state["sessions"])
        self.assertEqual(m._state["tools"]["fs"], tools)
        self.assertNotIn("fs", m._state["errors"])
        self.assertTrue(session.initialized)

    def test_clears_prior_error_on_success(self):
        m._state["errors"]["fs"] = "old failure"
        session = FakeSession(tools=[FakeTool("t")])
        imports = _make_imports(ClientSession=lambda r, w: session)
        with mock.patch.object(m, "_make_transport_cm",
                               return_value=AsyncCtx(value=("R", "W"))):
            self._run_task(imports, "fs", {"command": "x"})
        self.assertNotIn("fs", m._state["errors"])

    def test_transport_object_with_read_write_attrs(self):
        # Non-tuple transport exposing .read/.write attributes.
        conn = mock.MagicMock()
        conn.read = "R"
        conn.write = "W"
        session = FakeSession(tools=[])
        imports = _make_imports(ClientSession=lambda r, w: session)
        with mock.patch.object(m, "_make_transport_cm",
                               return_value=AsyncCtx(value=conn)):
            self._run_task(imports, "fs", {"command": "x"})
        self.assertIn("fs", m._state["sessions"])

    def test_transport_short_tuple_records_error(self):
        imports = _make_imports(ClientSession=mock.MagicMock())
        with mock.patch.object(m, "_make_transport_cm",
                               return_value=AsyncCtx(value=("only-one",))):
            self._run_task(imports, "fs", {"command": "x"})
        self.assertIn("fs", m._state["errors"])
        self.assertNotIn("fs", m._state["sessions"])

    def test_transport_unexpected_object_records_error(self):
        # Non-tuple object lacking read/write → RuntimeError → error recorded.
        class Bare:
            pass
        imports = _make_imports(ClientSession=mock.MagicMock())
        with mock.patch.object(m, "_make_transport_cm",
                               return_value=AsyncCtx(value=Bare())):
            self._run_task(imports, "fs", {"command": "x"})
        self.assertIn("fs", m._state["errors"])

    def test_transport_open_failure_records_error_and_sets_ready(self):
        # __aenter__ of the transport raises → except branch records error.
        imports = _make_imports(ClientSession=mock.MagicMock())
        with mock.patch.object(m, "_make_transport_cm",
                               return_value=AsyncCtx(enter_exc=RuntimeError("spawn failed"))):
            # No need to set shutdown; the task ends on its own via except.
            self._run_task(imports, "fs", {"command": "x"},
                           set_shutdown_after_ready=False)
        self.assertIn("fs", m._state["errors"])
        self.assertIn("spawn failed", m._state["errors"]["fs"])
        self.assertNotIn("fs", m._state["sessions"])

    def test_initialize_failure_records_error(self):
        session = FakeSession(init_exc=RuntimeError("init boom"))
        imports = _make_imports(ClientSession=lambda r, w: session)
        with mock.patch.object(m, "_make_transport_cm",
                               return_value=AsyncCtx(value=("R", "W"))):
            self._run_task(imports, "fs", {"command": "x"},
                           set_shutdown_after_ready=False)
        self.assertIn("fs", m._state["errors"])
        self.assertIn("init boom", m._state["errors"]["fs"])

    def test_list_tools_none_yields_empty_list(self):
        # list_tools returns an object whose .tools is None → tools = [].
        class Sess(FakeSession):
            async def list_tools(self):
                r = mock.MagicMock()
                r.tools = None
                return r
        session = Sess()
        imports = _make_imports(ClientSession=lambda r, w: session)
        with mock.patch.object(m, "_make_transport_cm",
                               return_value=AsyncCtx(value=("R", "W"))):
            self._run_task(imports, "fs", {"command": "x"})
        self.assertEqual(m._state["tools"]["fs"], [])


class SpawnServerTests(_StateIsolatedTest):
    def test_creates_events_and_schedules_task(self):
        created = {}

        def fake_create_task(coro):
            created["coro"] = coro
            # Close the coroutine so it isn't flagged as never-awaited.
            coro.close()
            return mock.MagicMock()

        async def go():
            with mock.patch.object(m.asyncio, "create_task",
                                   side_effect=fake_create_task):
                ready, shutdown = await m._spawn_server(_make_imports(), "fs", {})
            self.assertIsInstance(ready, asyncio.Event)
            self.assertIsInstance(shutdown, asyncio.Event)
            self.assertIn("coro", created)
        asyncio.run(go())


# ──────────────────────────────────────────────────────────────────────
# bootstrap
# ──────────────────────────────────────────────────────────────────────
class BootstrapTests(_StateIsolatedTest):
    def test_no_sdk_returns_empty(self):
        with mock.patch.object(m, "_mcp_imports", return_value={}):
            self.assertEqual(m.bootstrap(), [])

    def test_no_config_returns_empty(self):
        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value={}):
            self.assertEqual(m.bootstrap(), [])

    def test_idempotent_second_call_returns_catalog_no_deadlock(self):
        """Regression: bootstrap()'s idempotent branch holds ``_lock`` and calls
        ``_build_catalog()``, which re-acquires the same lock. ``_lock`` is now a
        reentrant ``RLock`` so a second bootstrap() no longer self-deadlocks — it
        just returns the existing catalog. Under a plain Lock this test hangs.
        """
        m._state["bootstrapped"] = True
        m._state["tools"]["fs"] = [FakeTool("read_file", "Read it.")]
        cfg = {"fs": {"command": "npx", "prefix": "fs"}}
        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value=cfg):
            catalog = m.bootstrap()   # would hang forever under a non-reentrant Lock
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0]["action_name"], "mcp_fs_read_file")
        # belt-and-braces: the lock is genuinely reentrant now
        self.assertTrue(m._lock.acquire(blocking=False))
        self.assertTrue(m._lock.acquire(blocking=False))
        m._lock.release(); m._lock.release()

    def test_idempotent_intent_via_build_catalog(self):
        """The idempotent branch is *supposed* to return the existing catalog
        from in-memory state. Verify that contract via ``_build_catalog()``
        directly (the function the branch delegates to) — same observable
        result the branch intends, minus the deadlocking re-lock."""
        m._state["bootstrapped"] = True
        m._state["tools"]["fs"] = [FakeTool("read_file", "Read it.")]
        cfg = {"fs": {"command": "npx", "prefix": "fs"}}
        with mock.patch.object(m, "_read_config", return_value=cfg):
            catalog = m._build_catalog()
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0]["action_name"], "mcp_fs_read_file")

    def test_full_bring_up_one_server(self):
        cfg = {"fs": {"command": "npx", "prefix": "fs"}}
        fake_loop = FakeLoop()
        ready_ev = object()
        shutdown_ev = object()

        # run_coroutine_threadsafe is called twice per server: spawn then wait.
        # Close the passed coroutine to avoid "never awaited" warnings, then
        # return a FakeFuture with the right result.
        results = [(ready_ev, shutdown_ev), True]
        call_idx = {"i": 0}

        def fake_rcts(coro, loop):
            coro.close()
            i = call_idx["i"]
            call_idx["i"] += 1
            return FakeFuture(result=results[i])

        # After "bring-up", pretend the server published a tool so the catalog
        # is non-empty. We patch _build_catalog to read live state, but simpler:
        # seed tools right before catalog build by patching _server_task effect.
        def seed_state(*a, **k):
            m._state["tools"]["fs"] = [FakeTool("read_file", "Read a file.")]
            return fake_loop
        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value=cfg), \
             mock.patch.object(m, "_start_loop_thread", side_effect=seed_state), \
             mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               side_effect=fake_rcts), \
             mock.patch.object(m.time, "time", return_value=123.0):
            catalog = m.bootstrap()

        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0]["server"], "fs")
        self.assertEqual(catalog[0]["tool"], "read_file")
        self.assertIs(m._state["shutdown_events"]["fs"], shutdown_ev)
        self.assertTrue(m._state["bootstrapped"])
        self.assertEqual(m._state["started_at"], 123.0)

    def test_skips_disabled_and_non_dict_servers(self):
        cfg = {
            "off": {"command": "x", "enabled": False},
            "weird": "not-a-dict",
            "fs": {"command": "npx"},
        }
        fake_loop = FakeLoop()

        def fake_rcts(coro, loop):
            coro.close()
            # Return ready/shutdown events for the spawn and any wait calls;
            # the value type doesn't matter for this count-based assertion.
            return FakeFuture(result=(object(), object()))

        # Only 'fs' is enabled AND a dict, so run_coroutine_threadsafe is called
        # exactly twice (spawn + ready-wait). 'off' (disabled) and 'weird'
        # (non-dict) are skipped before any scheduling.
        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value=cfg), \
             mock.patch.object(m, "_start_loop_thread", return_value=fake_loop), \
             mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               side_effect=fake_rcts) as rcts:
            m.bootstrap()
        self.assertEqual(rcts.call_count, 2)

    def test_spawn_failure_records_error_and_continues(self):
        cfg = {"bad": {"command": "x"}, "good": {"command": "y"}}
        fake_loop = FakeLoop()

        # First server's spawn future raises; second server proceeds normally.
        seq = []

        def fake_rcts(coro, loop):
            coro.close()
            seq.append(1)
            n = len(seq)
            if n == 1:
                # 'bad' spawn → raise on .result()
                return FakeFuture(exc=RuntimeError("launch kaboom"))
            # 'good': spawn returns events, then wait returns True
            if n == 2:
                return FakeFuture(result=(object(), object()))
            return FakeFuture(result=True)

        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value=cfg), \
             mock.patch.object(m, "_start_loop_thread", return_value=fake_loop), \
             mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               side_effect=fake_rcts):
            m.bootstrap()
        self.assertIn("bad", m._state["errors"])
        self.assertIn("launch failed", m._state["errors"]["bad"])

    def test_ready_wait_failure_sets_error(self):
        cfg = {"slow": {"command": "x"}}
        fake_loop = FakeLoop()
        seq = []

        def fake_rcts(coro, loop):
            coro.close()
            seq.append(1)
            if len(seq) == 1:
                return FakeFuture(result=(object(), object()))  # spawn ok
            return FakeFuture(exc=RuntimeError("ready timeout"))  # wait fails

        with mock.patch.object(m, "_mcp_imports", return_value=_make_imports()), \
             mock.patch.object(m, "_read_config", return_value=cfg), \
             mock.patch.object(m, "_start_loop_thread", return_value=fake_loop), \
             mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               side_effect=fake_rcts):
            m.bootstrap()
        self.assertIn("slow", m._state["errors"])
        self.assertIn("ready wait failed", m._state["errors"]["slow"])


# ──────────────────────────────────────────────────────────────────────
# _log_bootstrap_summary
# ──────────────────────────────────────────────────────────────────────
class LogBootstrapSummaryTests(_StateIsolatedTest):
    def test_all_up_no_failures(self):
        m._state["sessions"] = {"fs": object(), "gh": object()}
        cfg = {"fs": {"command": "x"}, "gh": {"command": "y"}}
        catalog = [{"action_name": "a"}, {"action_name": "b"}]
        with self.assertLogs("jarvis.mcp", level="INFO") as cm:
            m._log_bootstrap_summary(cfg, catalog)
        joined = "\n".join(cm.output)
        self.assertIn("2/2 servers up", joined)
        self.assertNotIn("failures", joined)

    def test_reports_failures_with_errors(self):
        m._state["sessions"] = {"fs": object()}
        m._state["errors"] = {"gh": "boom"}
        cfg = {"fs": {"command": "x"}, "gh": {"command": "y"}}
        with self.assertLogs("jarvis.mcp", level="INFO") as cm:
            m._log_bootstrap_summary(cfg, [{"action_name": "a"}])
        joined = "\n".join(cm.output)
        self.assertIn("1/2 servers up", joined)
        self.assertIn("failures", joined)
        self.assertIn("gh (boom)", joined)

    def test_failure_without_recorded_error_uses_unknown(self):
        m._state["sessions"] = {}
        cfg = {"gh": {"command": "y"}}
        with self.assertLogs("jarvis.mcp", level="INFO") as cm:
            m._log_bootstrap_summary(cfg, [])
        self.assertIn("unknown error", "\n".join(cm.output))

    def test_disabled_servers_excluded_from_counts(self):
        m._state["sessions"] = {"fs": object()}
        cfg = {"fs": {"command": "x"}, "off": {"command": "y", "enabled": False}}
        with self.assertLogs("jarvis.mcp", level="INFO") as cm:
            m._log_bootstrap_summary(cfg, [{"action_name": "a"}])
        # Only fs counts → 1/1, off ignored.
        self.assertIn("1/1 servers up", "\n".join(cm.output))


# ──────────────────────────────────────────────────────────────────────
# _sanitize_segment
# ──────────────────────────────────────────────────────────────────────
class SanitizeSegmentTests(unittest.TestCase):
    def test_lowercases_and_replaces_nonalnum(self):
        self.assertEqual(m._sanitize_segment("Read File!"), "read_file")

    def test_strips_leading_trailing_underscores(self):
        self.assertEqual(m._sanitize_segment("__Foo--"), "foo")

    def test_collapses_runs_of_separators(self):
        self.assertEqual(m._sanitize_segment("a---b...c"), "a_b_c")

    def test_empty_becomes_x(self):
        self.assertEqual(m._sanitize_segment(""), "x")

    def test_none_becomes_x(self):
        self.assertEqual(m._sanitize_segment(None), "x")

    def test_all_symbols_becomes_x(self):
        self.assertEqual(m._sanitize_segment("!!!"), "x")

    def test_preserves_digits(self):
        self.assertEqual(m._sanitize_segment("server2"), "server2")


# ──────────────────────────────────────────────────────────────────────
# _build_catalog
# ──────────────────────────────────────────────────────────────────────
class BuildCatalogTests(_StateIsolatedTest):
    def test_flattens_tools_with_prefix(self):
        m._state["tools"]["filesystem"] = [
            FakeTool("read_file", "Read a file.", {"type": "object"}),
            FakeTool("write_file", "Write a file."),
        ]
        cfg = {"filesystem": {"prefix": "fs"}}
        with mock.patch.object(m, "_read_config", return_value=cfg):
            out = m._build_catalog()
        names = {r["action_name"] for r in out}
        self.assertEqual(names, {"mcp_fs_read_file", "mcp_fs_write_file"})
        rec = next(r for r in out if r["tool"] == "read_file")
        self.assertEqual(rec["server"], "filesystem")
        self.assertEqual(rec["description"], "Read a file.")
        self.assertEqual(rec["schema"], {"type": "object"})
        # tool_index updated.
        self.assertEqual(m._state["tool_index"]["mcp_fs_read_file"],
                         ("filesystem", "read_file"))

    def test_falls_back_to_server_name_when_no_prefix(self):
        m._state["tools"]["GitHub"] = [FakeTool("create_issue")]
        with mock.patch.object(m, "_read_config", return_value={"GitHub": {}}):
            out = m._build_catalog()
        self.assertEqual(out[0]["action_name"], "mcp_github_create_issue")

    def test_skips_tools_without_name(self):
        m._state["tools"]["fs"] = [FakeTool(None), FakeTool("")]
        with mock.patch.object(m, "_read_config", return_value={"fs": {}}):
            out = m._build_catalog()
        self.assertEqual(out, [])

    def test_description_coerced_to_str(self):
        t = FakeTool("t")
        t.description = 12345  # non-str description
        m._state["tools"]["fs"] = [t]
        with mock.patch.object(m, "_read_config", return_value={"fs": {}}):
            out = m._build_catalog()
        self.assertEqual(out[0]["description"], "12345")
        self.assertIsInstance(out[0]["description"], str)

    def test_none_description_becomes_empty_string(self):
        t = FakeTool("t")
        t.description = None
        m._state["tools"]["fs"] = [t]
        with mock.patch.object(m, "_read_config", return_value={"fs": {}}):
            out = m._build_catalog()
        self.assertEqual(out[0]["description"], "")

    def test_input_schema_fallback_name(self):
        # Tool lacking inputSchema but having input_schema.
        class T:
            name = "t"
            description = "d"
            input_schema = {"k": "v"}
        t = T()
        # Ensure no inputSchema attribute.
        self.assertFalse(hasattr(t, "inputSchema"))
        m._state["tools"]["fs"] = [t]
        with mock.patch.object(m, "_read_config", return_value={"fs": {}}):
            out = m._build_catalog()
        self.assertEqual(out[0]["schema"], {"k": "v"})

    def test_non_dict_schema_becomes_empty_dict(self):
        t = FakeTool("t", inputSchema="not-a-dict")
        m._state["tools"]["fs"] = [t]
        with mock.patch.object(m, "_read_config", return_value={"fs": {}}):
            out = m._build_catalog()
        self.assertEqual(out[0]["schema"], {})

    def test_missing_schema_defaults_empty_dict(self):
        class T:
            name = "t"
            description = "d"
        m._state["tools"]["fs"] = [T()]
        with mock.patch.object(m, "_read_config", return_value={"fs": {}}):
            out = m._build_catalog()
        self.assertEqual(out[0]["schema"], {})

    def test_empty_tool_list_for_server(self):
        m._state["tools"]["fs"] = []
        with mock.patch.object(m, "_read_config", return_value={"fs": {}}):
            out = m._build_catalog()
        self.assertEqual(out, [])

    def test_none_tool_list_handled(self):
        m._state["tools"]["fs"] = None
        with mock.patch.object(m, "_read_config", return_value={"fs": {}}):
            out = m._build_catalog()
        self.assertEqual(out, [])

    def test_server_missing_from_config_uses_empty_cfg(self):
        # tools present for a server with no config entry → prefix = name.
        m._state["tools"]["orphan"] = [FakeTool("do_thing")]
        with mock.patch.object(m, "_read_config", return_value={}):
            out = m._build_catalog()
        self.assertEqual(out[0]["action_name"], "mcp_orphan_do_thing")

    def test_rebuild_clears_old_tool_index(self):
        m._state["tool_index"]["stale_action"] = ("old", "old")
        m._state["tools"]["fs"] = [FakeTool("read_file")]
        with mock.patch.object(m, "_read_config", return_value={"fs": {}}):
            m._build_catalog()
        self.assertNotIn("stale_action", m._state["tool_index"])


# ──────────────────────────────────────────────────────────────────────
# list_servers
# ──────────────────────────────────────────────────────────────────────
class ListServersTests(_StateIsolatedTest):
    def test_reports_status_per_server(self):
        m._state["sessions"] = {"fs": object()}
        m._state["tools"] = {"fs": [FakeTool("a"), FakeTool("b")]}
        m._state["errors"] = {"gh": "down"}
        cfg = {
            "fs": {"command": "x", "transport": "stdio"},
            "gh": {"command": "y", "transport": "sse", "enabled": True},
        }
        with mock.patch.object(m, "_read_config", return_value=cfg):
            out = m.list_servers()
        self.assertTrue(out["fs"]["connected"])
        self.assertEqual(out["fs"]["tool_count"], 2)
        self.assertIsNone(out["fs"]["error"])
        self.assertFalse(out["gh"]["connected"])
        self.assertEqual(out["gh"]["tool_count"], 0)
        self.assertEqual(out["gh"]["error"], "down")
        self.assertEqual(out["gh"]["transport"], "sse")

    def test_transport_defaults_to_stdio(self):
        cfg = {"fs": {"command": "x"}}  # no transport key
        with mock.patch.object(m, "_read_config", return_value=cfg):
            out = m.list_servers()
        self.assertEqual(out["fs"]["transport"], "stdio")

    def test_enabled_defaults_true(self):
        cfg = {"fs": {"command": "x"}}
        with mock.patch.object(m, "_read_config", return_value=cfg):
            out = m.list_servers()
        self.assertTrue(out["fs"]["enabled"])

    def test_skips_non_dict_entries(self):
        cfg = {"weird": "nope", "fs": {"command": "x"}}
        with mock.patch.object(m, "_read_config", return_value=cfg):
            out = m.list_servers()
        self.assertNotIn("weird", out)
        self.assertIn("fs", out)

    def test_empty_config_yields_empty_dict(self):
        with mock.patch.object(m, "_read_config", return_value={}):
            self.assertEqual(m.list_servers(), {})


# ──────────────────────────────────────────────────────────────────────
# _is_closed_session_error
# ──────────────────────────────────────────────────────────────────────
class IsClosedSessionErrorTests(unittest.TestCase):
    def test_matches_known_exc_class_names(self):
        for cls_name in ("ClosedResourceError", "BrokenResourceError",
                         "EndOfStream", "CancelledError"):
            exc = type(cls_name, (Exception,), {})()
            self.assertTrue(m._is_closed_session_error(exc), cls_name)

    def test_runtimeerror_cancel_scope_text(self):
        self.assertTrue(m._is_closed_session_error(
            RuntimeError("Attempted to exit cancel scope in a different task")))

    def test_runtimeerror_closed_text(self):
        self.assertTrue(m._is_closed_session_error(RuntimeError("stream closed")))

    def test_runtimeerror_event_loop_closed_text(self):
        self.assertTrue(m._is_closed_session_error(
            RuntimeError("Event loop is closed")))

    def test_runtimeerror_different_task_text(self):
        self.assertTrue(m._is_closed_session_error(
            RuntimeError("called from a different task")))

    def test_plain_runtimeerror_not_matched(self):
        self.assertFalse(m._is_closed_session_error(RuntimeError("real failure")))

    def test_unrelated_exception_not_matched(self):
        self.assertFalse(m._is_closed_session_error(ValueError("nope")))


# ──────────────────────────────────────────────────────────────────────
# call_tool
# ──────────────────────────────────────────────────────────────────────
class CallToolTests(_StateIsolatedTest):
    def test_non_dict_args_rejected(self):
        res = m.call_tool("fs", "read_file", args=["not", "dict"])
        self.assertFalse(res["ok"])
        self.assertIn("args must be a dict", res["error"])

    def test_not_bootstrapped_when_loop_none(self):
        m._state["loop"] = None
        m._state["sessions"]["fs"] = object()
        res = m.call_tool("fs", "read_file")
        self.assertFalse(res["ok"])
        self.assertIn("not bootstrapped", res["error"])

    def test_server_not_connected(self):
        m._state["loop"] = FakeLoop()
        res = m.call_tool("fs", "read_file")
        self.assertFalse(res["ok"])
        self.assertIn("not connected", res["error"])

    def test_server_not_connected_includes_recorded_error(self):
        m._state["loop"] = FakeLoop()
        m._state["errors"]["fs"] = "spawn died"
        res = m.call_tool("fs", "read_file")
        self.assertFalse(res["ok"])
        self.assertIn("spawn died", res["error"])

    def test_happy_path_formats_result(self):
        session = mock.MagicMock()
        m._state["loop"] = FakeLoop()
        m._state["sessions"]["fs"] = session
        result = FakeCallResult(content=[FakeContent("text", "hello world")])
        fut = FakeFuture(result=result)

        def fake_rcts(coro, loop):
            coro.close() if hasattr(coro, "close") else None
            return fut
        # session.call_tool returns a coroutine-like; we don't await it because
        # rcts is mocked. Make it a MagicMock so .close() exists.
        session.call_tool.return_value = mock.MagicMock()
        with mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               side_effect=fake_rcts):
            res = m.call_tool("fs", "read_file", {"path": "/x"})
        self.assertTrue(res["ok"])
        self.assertEqual(res["text"], "hello world")
        self.assertIs(res["raw"], result)
        session.call_tool.assert_called_once_with("read_file", {"path": "/x"})

    def test_none_args_defaults_to_empty_dict(self):
        session = mock.MagicMock()
        m._state["loop"] = FakeLoop()
        m._state["sessions"]["fs"] = session
        session.call_tool.return_value = mock.MagicMock()
        fut = FakeFuture(result=FakeCallResult(content=[]))
        with mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               return_value=fut):
            m.call_tool("fs", "read_file", None)
        # Called with {} when args is None.
        session.call_tool.assert_called_once_with("read_file", {})

    def test_timeout_concurrent_futures_cancels_future(self):
        session = mock.MagicMock()
        loop = FakeLoop()
        m._state["loop"] = loop
        m._state["sessions"]["fs"] = session
        session.call_tool.return_value = mock.MagicMock()
        fut = FakeFuture(exc=concurrent.futures.TimeoutError())
        with mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               return_value=fut):
            res = m.call_tool("fs", "read_file", {})
        self.assertFalse(res["ok"])
        self.assertIn("timed out", res["error"])
        # The cancel callback should have been scheduled on the loop.
        self.assertTrue(loop.soon_calls)
        self.assertTrue(fut.cancelled)

    def test_timeout_asyncio_error_also_handled(self):
        session = mock.MagicMock()
        m._state["loop"] = FakeLoop()
        m._state["sessions"]["fs"] = session
        session.call_tool.return_value = mock.MagicMock()
        fut = FakeFuture(exc=asyncio.TimeoutError())
        with mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               return_value=fut):
            res = m.call_tool("fs", "read_file", {})
        self.assertIn("timed out", res["error"])

    def test_timeout_cancel_swallows_loop_error(self):
        # If call_soon_threadsafe itself raises, it's swallowed and we still
        # return the timeout message.
        session = mock.MagicMock()
        loop = mock.MagicMock()
        loop.is_closed.return_value = False
        loop.call_soon_threadsafe.side_effect = RuntimeError("loop dead")
        m._state["loop"] = loop
        m._state["sessions"]["fs"] = session
        session.call_tool.return_value = mock.MagicMock()
        fut = FakeFuture(exc=concurrent.futures.TimeoutError())
        with mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               return_value=fut):
            res = m.call_tool("fs", "read_file", {})
        self.assertIn("timed out", res["error"])

    def test_closed_session_error_maps_to_shutdown_message(self):
        session = mock.MagicMock()
        m._state["loop"] = FakeLoop()
        m._state["sessions"]["fs"] = session
        session.call_tool.return_value = mock.MagicMock()
        closed_exc = type("ClosedResourceError", (Exception,), {})()
        fut = FakeFuture(exc=closed_exc)
        with mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               return_value=fut):
            res = m.call_tool("fs", "read_file", {})
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], "server shut down")

    def test_generic_exception_returns_descriptive_error(self):
        session = mock.MagicMock()
        m._state["loop"] = FakeLoop()
        m._state["sessions"]["fs"] = session
        session.call_tool.return_value = mock.MagicMock()
        fut = FakeFuture(exc=ValueError("bad tool args"))
        with mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               return_value=fut):
            res = m.call_tool("fs", "read_file", {})
        self.assertFalse(res["ok"])
        self.assertIn("ValueError", res["error"])
        self.assertIn("bad tool args", res["error"])
        self.assertIn("fs.read_file", res["error"])

    def test_custom_timeout_is_passed_to_future_result(self):
        session = mock.MagicMock()
        m._state["loop"] = FakeLoop()
        m._state["sessions"]["fs"] = session
        session.call_tool.return_value = mock.MagicMock()
        captured = {}

        class CapFuture(FakeFuture):
            def result(self, timeout=None):
                captured["timeout"] = timeout
                return FakeCallResult(content=[])
        with mock.patch.object(m.asyncio, "run_coroutine_threadsafe",
                               return_value=CapFuture()):
            m.call_tool("fs", "read_file", {}, timeout=7.5)
        self.assertEqual(captured["timeout"], 7.5)


# ──────────────────────────────────────────────────────────────────────
# _format_call_result
# ──────────────────────────────────────────────────────────────────────
class FormatCallResultTests(unittest.TestCase):
    def test_text_content_joined(self):
        r = FakeCallResult(content=[
            FakeContent("text", "line1"),
            FakeContent("text", "line2"),
        ])
        out = m._format_call_result(r)
        self.assertTrue(out["ok"])
        self.assertEqual(out["text"], "line1\nline2")
        self.assertIs(out["raw"], r)

    def test_is_error_flag_propagates(self):
        r = FakeCallResult(content=[FakeContent("text", "oops")], isError=True)
        out = m._format_call_result(r)
        self.assertFalse(out["ok"])

    def test_image_and_resource_become_markers(self):
        r = FakeCallResult(content=[
            FakeContent("text", "see attached"),
            FakeContent("image", None),
            FakeContent("resource", None),
        ])
        out = m._format_call_result(r)
        self.assertIn("see attached", out["text"])
        self.assertIn("[image]", out["text"])
        self.assertIn("[resource]", out["text"])

    def test_unknown_type_marker(self):
        r = FakeCallResult(content=[FakeContent("video", None)])
        out = m._format_call_result(r)
        self.assertIn("[video]", out["text"])

    def test_none_type_marker_unknown(self):
        r = FakeCallResult(content=[FakeContent(None, None)])
        out = m._format_call_result(r)
        self.assertIn("[unknown]", out["text"])

    def test_dict_content_text(self):
        # Content item is a plain dict (some SDK builds) with type/text keys.
        r = FakeCallResult(content=[{"type": "text", "text": "dict text"}])
        out = m._format_call_result(r)
        self.assertEqual(out["text"], "dict text")

    def test_dict_content_image_marker(self):
        r = FakeCallResult(content=[{"type": "image"}])
        out = m._format_call_result(r)
        self.assertIn("[image]", out["text"])

    def test_empty_text_skipped(self):
        # A text part with empty/None text contributes nothing.
        r = FakeCallResult(content=[
            FakeContent("text", ""),
            FakeContent("text", None),
            FakeContent("text", "real"),
        ])
        out = m._format_call_result(r)
        self.assertEqual(out["text"], "real")

    def test_no_content_yields_empty_text(self):
        out = m._format_call_result(FakeCallResult(content=[]))
        self.assertTrue(out["ok"])
        self.assertEqual(out["text"], "")

    def test_content_none_treated_as_empty(self):
        r = mock.MagicMock()
        r.isError = False
        r.content = None
        out = m._format_call_result(r)
        self.assertEqual(out["text"], "")

    def test_text_coerced_to_str(self):
        r = FakeCallResult(content=[FakeContent("text", 999)])
        out = m._format_call_result(r)
        self.assertEqual(out["text"], "999")

    def test_extras_only_no_text(self):
        # Only an image part → text is just the marker (no leading newline).
        r = FakeCallResult(content=[FakeContent("image", None)])
        out = m._format_call_result(r)
        self.assertEqual(out["text"], "[image]")


# ──────────────────────────────────────────────────────────────────────
# shutdown
# ──────────────────────────────────────────────────────────────────────
class ShutdownTests(_StateIsolatedTest):
    def test_noop_when_no_loop(self):
        m._state["loop"] = None
        # Should simply return without error.
        m.shutdown()
        self.assertFalse(m._state["bootstrapped"])

    def test_noop_when_loop_closed(self):
        m._state["loop"] = FakeLoop(closed=True)
        m._state["shutdown_events"] = {"fs": mock.MagicMock()}
        m.shutdown()
        # Events were not touched (loop closed) but state preserved as-is here;
        # the function returns early before clearing.
        self.assertIn("fs", m._state["shutdown_events"])

    def test_signals_all_events_and_clears_state(self):
        loop = FakeLoop()
        m._state["loop"] = loop
        ev1, ev2 = mock.MagicMock(), mock.MagicMock()
        m._state["shutdown_events"] = {"fs": ev1, "gh": ev2}
        m._state["sessions"] = {"fs": object()}
        m._state["tools"] = {"fs": [FakeTool("t")]}
        m._state["tool_index"] = {"a": ("fs", "t")}
        m._state["bootstrapped"] = True
        m._state["started_at"] = 123.0

        m.shutdown()

        # Both events' .set were scheduled (FakeLoop runs callbacks inline).
        ev1.set.assert_called_once()
        ev2.set.assert_called_once()
        # State reset for a clean re-bootstrap.
        self.assertFalse(m._state["bootstrapped"])
        self.assertEqual(m._state["sessions"], {})
        self.assertEqual(m._state["tools"], {})
        self.assertEqual(m._state["tool_index"], {})
        self.assertEqual(m._state["shutdown_events"], {})
        self.assertIsNone(m._state["started_at"])

    def test_event_set_failure_is_swallowed(self):
        # call_soon_threadsafe raising for one event must not stop the clear.
        loop = mock.MagicMock()
        loop.is_closed.return_value = False
        loop.call_soon_threadsafe.side_effect = RuntimeError("boom")
        m._state["loop"] = loop
        m._state["shutdown_events"] = {"fs": mock.MagicMock()}
        m._state["bootstrapped"] = True
        with self.assertLogs("jarvis.mcp", level="WARNING"):
            m.shutdown()
        self.assertFalse(m._state["bootstrapped"])
        self.assertEqual(m._state["shutdown_events"], {})

    def test_timeout_kwarg_accepted(self):
        # `timeout` is kept for API compat; passing it must not error.
        m._state["loop"] = FakeLoop()
        m.shutdown(timeout=9.0)
        self.assertFalse(m._state["bootstrapped"])


if __name__ == "__main__":
    unittest.main()
