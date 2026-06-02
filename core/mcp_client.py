"""
core.mcp_client — Model Context Protocol client manager.

Spawns/connects to MCP servers configured in `mcp_servers.json`, holds the
sessions open in a background asyncio event loop, discovers each server's
tools, and exposes a synchronous facade for the JARVIS skill loader.

Why an in-process bg loop?
--------------------------
The MCP Python SDK is async-only and its transports are async context
managers — the ClientSession must remain open between calls. JARVIS skill
actions are synchronous `(arg: str) -> str` callables invoked from the
main thread. So this module owns a daemon thread running its own asyncio
event loop, runs one persistent "keep-alive" coroutine per MCP server,
and bridges sync→async via `asyncio.run_coroutine_threadsafe`.

mcp_servers.json schema (the Claude Desktop "mcpServers" wrapper is also
accepted)::

    {
      "filesystem": {
        "transport": "stdio",
        "command":   "npx",
        "args":      ["-y", "@modelcontextprotocol/server-filesystem",
                       "C:/JARVIS/data"],
        "env":       {"NODE_NO_WARNINGS": "1"},
        "enabled":   true,
        "prefix":    "fs"
      },
      "github": {
        "transport": "stdio",
        "command":   "npx",
        "args":      ["-y", "@modelcontextprotocol/server-github"],
        "env":       {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."}
      },
      "brave": {
        "transport": "sse",
        "url":       "http://localhost:8788/sse",
        "headers":   {"Authorization": "Bearer xyz"}
      }
    }

Public API
----------
    is_available()                   -> True iff mcp SDK is installed AND ≥1 enabled server
    bootstrap()                      -> list of {action_name, server, tool, description, schema}
    call_tool(server, tool, args)    -> {ok, text, raw, error?}
    list_servers()                   -> {server_name: {connected, tool_count, error}}
    shutdown(timeout=5.0)            -> signal every server task to exit

The module is intentionally tolerant of missing deps and missing config —
if `mcp` isn't installed or no servers are configured, every entry point
degrades quietly. Per-server failures are isolated; one broken server
does not prevent the others from coming up.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from typing import Any


_log = logging.getLogger("jarvis.mcp")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_PROJECT_DIR, "mcp_servers.json")

# Per-server bring-up budget. npx will sometimes download the server
# package on first run, which can take a while.
_SERVER_READY_TIMEOUT = 30.0
# Per-call budget for tool dispatch. Individual tools (GitHub PR diff,
# Postgres query) can legitimately take 10s+.
_DEFAULT_CALL_TIMEOUT = 30.0

# Shared global state. Guarded by `_lock` except for the asyncio.Event
# instances, which are created on (and only touched from) the loop thread.
_lock = threading.RLock()  # reentrant: bootstrap()'s idempotent branch calls
#                            _build_catalog(), which re-acquires this lock.
_state: dict[str, Any] = {
    "loop":            None,    # asyncio loop running in bg thread
    "thread":          None,    # daemon thread that owns the loop
    "sessions":        {},      # server_name → live ClientSession
    "tools":           {},      # server_name → list of mcp.types.Tool
    "tool_index":      {},      # action_name → (server_name, tool_name)
    "errors":          {},      # server_name → str (latest error)
    "shutdown_events": {},      # server_name → asyncio.Event (created on loop)
    "started_at":      None,
    "bootstrapped":    False,
}


# ── lazy SDK import ─────────────────────────────────────────────────
def _mcp_imports() -> dict[str, Any]:
    """Try every common import path the `mcp` SDK has shipped under.

    The package layout shifted between 0.x and 1.x; this returns whatever
    pieces are available, or `{}` if the SDK is not installed at all.
    """
    try:
        from mcp import ClientSession  # type: ignore
    except Exception:
        try:
            from mcp.client.session import ClientSession  # type: ignore
        except Exception:
            return {}

    try:
        from mcp.client.stdio import stdio_client, StdioServerParameters  # type: ignore
    except Exception:
        try:
            from mcp.client.stdio import stdio_client  # type: ignore
            from mcp import StdioServerParameters  # type: ignore
        except Exception:
            return {}

    sse_client = None
    streamable_http_client = None
    try:
        from mcp.client.sse import sse_client  # type: ignore
    except Exception:
        pass
    try:
        from mcp.client.streamable_http import (  # type: ignore
            streamablehttp_client as streamable_http_client,
        )
    except Exception:
        pass

    return {
        "ClientSession":          ClientSession,
        "stdio_client":           stdio_client,
        "StdioServerParameters":  StdioServerParameters,
        "sse_client":             sse_client,
        "streamable_http_client": streamable_http_client,
    }


def is_available() -> bool:
    """True iff the SDK is importable AND at least one enabled server is configured."""
    if not _mcp_imports():
        return False
    cfg = _read_config()
    if not cfg:
        return False
    return any((s or {}).get("enabled", True) for s in cfg.values() if isinstance(s, dict))


# ── config ──────────────────────────────────────────────────────────
def _read_config() -> dict[str, dict]:
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        _log.warning("[mcp] could not read %s: %s", _CONFIG_PATH, e)
        return {}
    if not isinstance(data, dict):
        return {}
    # Accept Claude Desktop's "mcpServers" wrapper format too.
    if "mcpServers" in data and isinstance(data["mcpServers"], dict):
        data = data["mcpServers"]
    return data


# ── bg loop ─────────────────────────────────────────────────────────
def _start_loop_thread() -> asyncio.AbstractEventLoop:
    """Start the daemon thread that owns the asyncio event loop. Idempotent."""
    with _lock:
        loop = _state["loop"]
        if loop is not None and not loop.is_closed():
            return loop

    ready = threading.Event()
    new_loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(new_loop)
        ready.set()
        try:
            new_loop.run_forever()
        finally:
            try:
                new_loop.close()
            except Exception:
                pass

    t = threading.Thread(target=_run, name="mcp-asyncio", daemon=True)
    t.start()
    if not ready.wait(timeout=5.0):
        raise RuntimeError("mcp asyncio loop failed to start within 5s")

    with _lock:
        _state["loop"]   = new_loop
        _state["thread"] = t
    return new_loop


# ── transport ───────────────────────────────────────────────────────
def _make_transport_cm(imports: dict, name: str, config: dict):
    """Return the async context manager appropriate for `config['transport']`."""
    transport = (config.get("transport") or "stdio").lower()

    if transport == "stdio":
        cmd = config.get("command")
        if not cmd:
            raise ValueError(f"server '{name}' missing 'command'")
        args = [str(a) for a in (config.get("args") or [])]
        env_cfg = config.get("env") or None
        # Merge into the process env so PATH / NODE_PATH still resolve when
        # the user only specified a couple of overrides.
        env = None
        if env_cfg:
            merged = dict(os.environ)
            merged.update({str(k): str(v) for k, v in env_cfg.items()})
            env = merged
        kw: dict[str, Any] = {"command": str(cmd), "args": args}
        if env is not None:
            kw["env"] = env
        try:
            params = imports["StdioServerParameters"](**kw)
        except TypeError:
            # Older mcp without `env` kwarg — drop it.
            kw.pop("env", None)
            params = imports["StdioServerParameters"](**kw)
        return imports["stdio_client"](params)

    if transport == "sse":
        sse = imports.get("sse_client")
        if sse is None:
            raise RuntimeError("sse transport not supported by installed mcp version")
        url = config.get("url")
        if not url:
            raise ValueError(f"server '{name}' missing 'url'")
        headers = config.get("headers") or None
        try:
            return sse(url=url, headers=headers)
        except TypeError:
            return sse(url)  # very old signature

    if transport in ("http", "streamable_http", "streamable-http"):
        sh = imports.get("streamable_http_client")
        if sh is None:
            raise RuntimeError(
                "streamable_http transport not supported by installed mcp version"
            )
        url = config.get("url")
        if not url:
            raise ValueError(f"server '{name}' missing 'url'")
        headers = config.get("headers") or None
        try:
            return sh(url=url, headers=headers)
        except TypeError:
            return sh(url)

    raise ValueError(f"unknown transport '{transport}' for server '{name}'")


# ── per-server keep-alive coroutine ─────────────────────────────────
async def _server_task(
    imports: dict,
    name: str,
    config: dict,
    ready: asyncio.Event,
    shutdown: asyncio.Event,
) -> None:
    """Hold a server's ClientSession open until `shutdown` is signalled.

    Resolution:
      1. Open the configured transport (stdio | sse | streamable_http).
      2. Open a ClientSession on top of it.
      3. initialize() → list_tools(), publish results into _state.
      4. Set `ready`, then await `shutdown` so the context managers stay
         open and the session remains callable from other threads via
         `call_tool()`.
    """
    ClientSession = imports["ClientSession"]
    try:
        async with _make_transport_cm(imports, name, config) as conn:
            # streamable_http yields 3 items (read, write, get_session_id);
            # stdio/sse yield 2. Unpack defensively.
            if isinstance(conn, tuple):
                if len(conn) >= 2:
                    read, write = conn[0], conn[1]
                else:
                    raise RuntimeError(
                        f"transport for '{name}' returned {len(conn)}-tuple"
                    )
            else:
                # Some SDK builds return a custom object with attributes.
                read = getattr(conn, "read", None)
                write = getattr(conn, "write", None)
                if read is None or write is None:
                    raise RuntimeError(
                        f"transport for '{name}' returned unexpected type {type(conn).__name__}"
                    )

            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                tools = list(getattr(tools_resp, "tools", None) or [])
                with _lock:
                    _state["sessions"][name] = session
                    _state["tools"][name]    = tools
                    _state["errors"].pop(name, None)
                ready.set()
                _log.info("[mcp] '%s' ready with %d tool(s)", name, len(tools))
                await shutdown.wait()
    except Exception as e:
        with _lock:
            _state["errors"][name] = f"{type(e).__name__}: {e}"
            _state["sessions"].pop(name, None)
        _log.warning("[mcp] server '%s' task ended: %s", name, e)
        ready.set()


async def _spawn_server(imports: dict, name: str, config: dict):
    """Create per-server events on the loop's thread, then schedule the task."""
    ready = asyncio.Event()
    shutdown = asyncio.Event()
    asyncio.create_task(_server_task(imports, name, config, ready, shutdown))
    return ready, shutdown


async def _wait_event(ev: asyncio.Event, timeout: float) -> bool:
    try:
        await asyncio.wait_for(ev.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


# ── public bootstrap / call / status ────────────────────────────────
def bootstrap() -> list[dict]:
    """Bring up every enabled MCP server and return a flat tool catalog.

    Idempotent — calling twice after a successful first bootstrap just
    returns the existing catalog from in-memory state.

    Each catalog entry::

        {
          "action_name": "mcp_filesystem_read_file",
          "server":      "filesystem",
          "tool":        "read_file",
          "description": "Read the complete contents of a file.",
          "schema":      {"type": "object", "properties": {...}, "required": [...]}
        }
    """
    imports = _mcp_imports()
    if not imports:
        _log.info("[mcp] mcp SDK not installed; bootstrap skipped")
        return []
    cfg = _read_config()
    if not cfg:
        _log.info("[mcp] no servers configured at %s; skipping", _CONFIG_PATH)
        return []

    with _lock:
        if _state["bootstrapped"]:
            return _build_catalog()
        _state["bootstrapped"] = True
        _state["started_at"]   = time.time()

    loop = _start_loop_thread()

    for name, server_cfg in cfg.items():
        if not isinstance(server_cfg, dict):
            continue
        if not server_cfg.get("enabled", True):
            continue
        # Schedule the per-server task on the loop and wait for ready.
        try:
            spawn_fut = asyncio.run_coroutine_threadsafe(
                _spawn_server(imports, name, server_cfg), loop,
            )
            ready_event, shutdown_event = spawn_fut.result(timeout=10.0)
        except Exception as e:
            _log.warning("[mcp] failed to launch '%s': %s", name, e)
            with _lock:
                _state["errors"][name] = f"launch failed: {e}"
            continue

        with _lock:
            _state["shutdown_events"][name] = shutdown_event

        # Wait for the server to report ready (or fail) before moving on.
        try:
            wait_fut = asyncio.run_coroutine_threadsafe(
                _wait_event(ready_event, _SERVER_READY_TIMEOUT), loop,
            )
            wait_fut.result(timeout=_SERVER_READY_TIMEOUT + 2.0)
        except Exception as e:
            _log.warning("[mcp] '%s' did not become ready: %s", name, e)
            with _lock:
                _state["errors"].setdefault(name, f"ready wait failed: {e}")

    catalog = _build_catalog()
    _log_bootstrap_summary(cfg, catalog)
    return catalog


def _log_bootstrap_summary(cfg: dict[str, dict], catalog: list[dict]) -> None:
    """Emit a one-line summary of bootstrap results so failures surface on boot."""
    enabled = [
        name for name, sc in cfg.items()
        if isinstance(sc, dict) and sc.get("enabled", True)
    ]
    with _lock:
        sessions = dict(_state["sessions"])
        errors   = dict(_state["errors"])
    up = [n for n in enabled if n in sessions]
    failed = [(n, errors.get(n, "unknown error")) for n in enabled if n not in sessions]
    msg = f"[mcp] {len(up)}/{len(enabled)} servers up, {len(catalog)} tools"
    if failed:
        details = ", ".join(f"{n} ({err})" for n, err in failed)
        msg += f", failures: {details}"
    _log.info(msg)


def _sanitize_segment(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "x"


def _build_catalog() -> list[dict]:
    """Flatten per-server tool lists into action-friendly records."""
    cfg = _read_config()
    out: list[dict] = []
    with _lock:
        servers = list(_state["tools"].items())
        # Reset the tool_index — we're about to rebuild it.
        _state["tool_index"].clear()
    for name, tools in servers:
        server_cfg = cfg.get(name) or {}
        prefix = _sanitize_segment(server_cfg.get("prefix") or name)
        for tool in tools or []:
            tool_name = getattr(tool, "name", None)
            if not tool_name:
                continue
            action = f"mcp_{prefix}_{_sanitize_segment(tool_name)}"
            description = getattr(tool, "description", "") or ""
            # mcp.types.Tool uses `inputSchema`; fall back to `input_schema`
            # in case a future SDK rev renames it.
            schema = (
                getattr(tool, "inputSchema", None)
                or getattr(tool, "input_schema", None)
                or {}
            )
            with _lock:
                _state["tool_index"][action] = (name, tool_name)
            out.append({
                "action_name": action,
                "server":      name,
                "tool":        tool_name,
                "description": description if isinstance(description, str) else str(description),
                "schema":      schema if isinstance(schema, dict) else {},
            })
    return out


def list_servers() -> dict[str, dict]:
    """Snapshot of per-server status — useful for the `mcp_status` action."""
    cfg = _read_config()
    with _lock:
        sessions = dict(_state["sessions"])
        tools    = {k: list(v or []) for k, v in _state["tools"].items()}
        errors   = dict(_state["errors"])
    out: dict[str, dict] = {}
    for name, server_cfg in cfg.items():
        if not isinstance(server_cfg, dict):
            continue
        out[name] = {
            "enabled":    server_cfg.get("enabled", True),
            "transport":  server_cfg.get("transport") or "stdio",
            "connected":  name in sessions,
            "tool_count": len(tools.get(name, [])),
            "error":      errors.get(name),
        }
    return out


def _is_closed_session_error(exc: BaseException) -> bool:
    """True when `exc` looks like it came from a session whose transport was
    torn down concurrently (e.g. shutdown() closed the streams mid-call),
    rather than a genuine tool failure. Matched by class name so we don't
    hard-depend on anyio's exception classes, plus the cancel-scope text
    anyio raises when a task group unwinds underneath us."""
    name = type(exc).__name__
    if name in ("ClosedResourceError", "BrokenResourceError",
                "EndOfStream", "CancelledError"):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if ("cancel scope" in msg or "closed" in msg
                or "different task" in msg or "event loop is closed" in msg):
            return True
    return False


def call_tool(
    server: str,
    tool: str,
    args: dict | None = None,
    *,
    timeout: float = _DEFAULT_CALL_TIMEOUT,
) -> dict:
    """Synchronous MCP tool call. Always returns a dict; never raises.

    Result shape::
        {"ok": bool, "text": str, "raw": <CallToolResult>, "error"?: str}
    """
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {"ok": False, "error": f"args must be a dict, got {type(args).__name__}"}

    with _lock:
        session = _state["sessions"].get(server)
        loop    = _state["loop"]
        err     = _state["errors"].get(server)

    if loop is None:
        return {"ok": False, "error": "mcp client not bootstrapped"}
    if session is None:
        msg = f"mcp server '{server}' not connected"
        if err:
            msg += f" ({err})"
        return {"ok": False, "error": msg}

    fut = asyncio.run_coroutine_threadsafe(session.call_tool(tool, args), loop)
    try:
        result = fut.result(timeout=timeout)
    except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
        # Cancel the future on the bg loop so the hung session.call_tool
        # coroutine doesn't keep running forever after we've given up on it.
        try:
            loop.call_soon_threadsafe(fut.cancel)
        except Exception:
            pass
        return {"ok": False, "error": "tool call timed out"}
    except Exception as e:
        # A concurrent shutdown()/server crash can surface as a closed-stream
        # (anyio ClosedResourceError), a RuntimeError, or a cancel-scope error
        # rather than a meaningful tool failure. Map those to a clean message
        # instead of leaking the opaque internal exception to the skill layer.
        if _is_closed_session_error(e):
            return {"ok": False, "error": "server shut down"}
        return {
            "ok": False,
            "error": f"mcp '{server}.{tool}' raised: {type(e).__name__}: {e}",
        }
    return _format_call_result(result)


def _format_call_result(result: Any) -> dict:
    """Normalise an mcp CallToolResult into a flat dict for the skill layer."""
    is_error = bool(getattr(result, "isError", False))
    content  = getattr(result, "content", None) or []
    parts: list[str] = []
    extras: list[str] = []
    for c in content:
        ctype = getattr(c, "type", None)
        if ctype is None and isinstance(c, dict):
            ctype = c.get("type")
        if ctype == "text":
            text = getattr(c, "text", None)
            if text is None and isinstance(c, dict):
                text = c.get("text")
            if text:
                parts.append(str(text))
        elif ctype == "image":
            extras.append("[image]")
        elif ctype == "resource":
            extras.append("[resource]")
        else:
            extras.append(f"[{ctype or 'unknown'}]")
    text = "\n".join(parts).strip()
    if extras:
        text = (text + "\n" + " ".join(extras)).strip()
    return {
        "ok":   not is_error,
        "text": text,
        "raw":  result,
    }


def shutdown(*, timeout: float = 5.0) -> None:
    """Signal every running server task to exit. Best-effort."""
    with _lock:
        loop   = _state["loop"]
        events = list(_state["shutdown_events"].items())
    if loop is None or loop.is_closed():
        return
    for name, ev in events:
        try:
            loop.call_soon_threadsafe(ev.set)
        except Exception as e:
            _log.warning("[mcp] shutdown signal for '%s' failed: %s", name, e)
    # Clear the catalog/connection state so a later bootstrap() reconnects
    # cleanly. Without this, bootstrap()'s `if _state["bootstrapped"]` guard
    # would early-return a stale catalog whose ClientSessions point at server
    # tasks we just told to exit. We leave the bg loop/thread running (the
    # daemon thread dies with the process) to avoid racing in-flight calls.
    with _lock:
        _state["bootstrapped"]    = False
        _state["sessions"]        = {}
        _state["tools"]           = {}
        _state["tool_index"]      = {}
        _state["shutdown_events"] = {}
        _state["started_at"]      = None
    # The daemon thread will die with the process; we don't stop the
    # loop here to avoid racing against in-flight call_tool() requests.
    _ = timeout  # kept for API compatibility
