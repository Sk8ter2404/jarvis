"""
skills/mcp_tools.py — JARVIS skill bridge for Model Context Protocol servers.

On register(), this skill asks `core.mcp_client` to bring up every server
configured in `mcp_servers.json` and discover its tools. Each discovered
tool is registered into JARVIS' ACTIONS dict as

    mcp_<server_prefix>_<tool_name>

so the LLM and the dispatcher can call community-built MCP tools by name
the same way they call any other JARVIS action.

Force multiplier — one bootstrap unlocks the entire MCP ecosystem
(filesystem, GitHub, Slack, Notion, Brave Search, Postgres, Puppeteer, …)
without writing per-service skills.

Management actions
------------------
    mcp_status                — connection + tool summary
    mcp_list_tools [server]   — list discovered tools, optionally filtered
    mcp_call <server> <tool> [json_args]
                              — manual dispatch (debugging / one-offs)
    mcp_reload                — re-read mcp_servers.json (informational; full
                                reconnect needs a JARVIS restart)

If the `mcp` SDK is not installed or no servers are configured, register()
silently no-ops so JARVIS still loads cleanly. Install hint::

    pip install "mcp[cli]"
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable


# ── arg parsing ─────────────────────────────────────────────────────
def _parse_args(raw: str, schema: dict) -> dict | str:
    """Convert a JARVIS-style string arg into the dict an MCP tool expects.

    Strategy:
      1. If the tool takes no args, ignore the input and return {}.
      2. If `raw` parses as a JSON object, use it directly.
      3. If the tool has exactly one (required) property, treat `raw` as
         that property's value.
      4. Otherwise, surface a "Format:" hint listing the expected keys.

    Returns a dict on success, or a user-facing error string.
    """
    raw = (raw or "").strip()
    props = (schema or {}).get("properties") or {}
    required = list((schema or {}).get("required") or [])

    if not props:
        return {}

    if not raw:
        if required:
            return f"Missing required args: {', '.join(required)}."
        return {}

    # Try JSON first — supports the full structured-arg case.
    if raw[:1] in "{[":
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Single-arg shortcut — the most common voice-command case.
    if len(required) == 1:
        return {required[0]: _coerce(raw, props.get(required[0]) or {})}
    if len(props) == 1:
        only = next(iter(props.keys()))
        return {only: _coerce(raw, props[only] or {})}

    keys = list(props.keys())
    preview = ", ".join(sorted(keys)[:6]) + ("..." if len(keys) > 6 else "")
    return (
        f"Format: pass a JSON object with keys {{{preview}}}. "
        f"Required: {', '.join(required) or '(none)'}."
    )


def _coerce(value: str, prop_schema: dict) -> Any:
    """Best-effort coerce a string to the property's expected JSON type."""
    expected = prop_schema.get("type")
    if expected in (None, "string"):
        return value
    if expected == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if expected == "number":
        try:
            return float(value)
        except ValueError:
            return value
    if expected == "boolean":
        v = value.strip().lower()
        if v in ("true", "yes", "on", "1"):
            return True
        if v in ("false", "no", "off", "0"):
            return False
        return value
    if expected in ("array", "object"):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _truncate_text(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


# ── per-tool action factory ─────────────────────────────────────────
def _make_tool_action(
    mcp_client: Any,
    server: str,
    tool: str,
    schema: dict,
) -> Callable[[str], str]:
    def _action(arg: str = "") -> str:
        args = _parse_args(arg, schema)
        if isinstance(args, str):
            return args  # user-facing format hint
        result = mcp_client.call_tool(server, tool, args)
        if not result.get("ok"):
            return result.get("error") or f"{server}.{tool} failed"
        text = result.get("text") or ""
        if not text:
            return f"{server}.{tool} ok (no output)"
        return _truncate_text(text)
    _action.__name__ = f"mcp_{server}_{tool}"
    return _action


# ── management actions ─────────────────────────────────────────────
def _make_status_action(mcp_client: Any) -> Callable[[str], str]:
    def _status(_: str = "") -> str:
        servers = mcp_client.list_servers()
        if not servers:
            return "No MCP servers configured, sir."
        lines = []
        for name, info in servers.items():
            state = "connected" if info.get("connected") else "offline"
            tools = info.get("tool_count") or 0
            line = f"  • {name} ({info.get('transport')}): {state}, {tools} tool(s)"
            if info.get("error"):
                line += f" — {info['error']}"
            lines.append(line)
        return "MCP servers, sir:\n" + "\n".join(lines)
    return _status


def _make_list_action(get_catalog: Callable[[], list[dict]]) -> Callable[[str], str]:
    def _list(arg: str = "") -> str:
        filt = (arg or "").strip().lower()
        catalog = get_catalog()
        items = [
            e for e in catalog
            if not filt or filt in (e.get("server") or "").lower()
        ]
        if not items:
            if filt:
                return f"No MCP tools matching '{filt}', sir."
            return "No MCP tools discovered yet, sir."
        items.sort(key=lambda e: (e.get("server") or "", e.get("tool") or ""))
        out = []
        for e in items[:60]:
            desc = (e.get("description") or "").strip().split("\n", 1)[0]
            if len(desc) > 80:
                desc = desc[:77] + "..."
            out.append(f"  • {e['action_name']} — {desc}" if desc else f"  • {e['action_name']}")
        more = f"\n  ...and {len(items) - 60} more" if len(items) > 60 else ""
        return f"{len(items)} MCP tool(s), sir:\n" + "\n".join(out) + more
    return _list


def _make_call_action(mcp_client: Any) -> Callable[[str], str]:
    """Manual dispatch: `mcp_call <server> <tool> [json_args]`."""
    def _call(arg: str = "") -> str:
        raw = (arg or "").strip()
        if not raw:
            return "Format: mcp_call <server> <tool> [json_args]"
        # Whitespace split (max 3 fields) keeps the JSON blob verbatim —
        # shlex would strip the JSON's quotes and fragment it on spaces.
        parts = raw.split(None, 2)
        # Allow comma-separated input too: "filesystem, read_file, {...}".
        if len(parts) < 2 or parts[0].endswith(","):
            parts = [p.strip() for p in raw.split(",", 2)]
        if len(parts) < 2:
            return "Format: mcp_call <server> <tool> [json_args]"
        server, tool = parts[0], parts[1]
        args: dict[str, Any] = {}
        if len(parts) >= 3 and parts[2]:
            try:
                args = json.loads(parts[2])
                if not isinstance(args, dict):
                    return "json_args must be a JSON object"
            except Exception as e:
                return f"json_args parse failed: {e}"
        result = mcp_client.call_tool(server, tool, args)
        if not result.get("ok"):
            return result.get("error") or f"{server}.{tool} failed"
        return _truncate_text(result.get("text") or f"{server}.{tool} ok (no output)")
    return _call


def _make_reload_action(
    mcp_client: Any,
    actions: dict,
    catalog_holder: list,
) -> Callable[[str], str]:
    """Re-query the MCP client and register any tools that came up after the
    initial async bootstrap (e.g., a slow server that took 20 s to start)."""
    def _reload(_: str = "") -> str:
        # We don't tear down existing sessions — that would race against
        # in-flight tool calls. We just re-query the catalog from already-
        # connected servers and pick up any tools we haven't registered yet.
        try:
            catalog = mcp_client.bootstrap()  # idempotent — returns current catalog
        except Exception as e:
            return f"MCP reload failed, sir: {type(e).__name__}: {e}"
        catalog_holder[0] = catalog
        added = _register_tool_actions(actions, mcp_client, catalog, verbose=False)
        servers = mcp_client.list_servers()
        live    = sum(1 for s in servers.values() if s.get("connected"))
        total   = len(servers)
        msg = f"MCP reload, sir — {live}/{total} server(s) connected, {len(catalog)} tool(s) discovered"
        if added:
            head = ", ".join(added[:5])
            more = f" +{len(added) - 5} more" if len(added) > 5 else ""
            msg += f"; newly registered: {head}{more}"
        msg += ". To pick up edits in mcp_servers.json, restart JARVIS."
        return msg
    return _reload


def _register_tool_actions(
    actions: dict,
    mcp_client: Any,
    catalog: list[dict],
    *,
    verbose: bool = True,
) -> list[str]:
    """Add a per-tool action for every catalog entry not already present.

    Safe to call repeatedly — entries already registered as MCP tools are
    skipped silently; entries colliding with non-MCP actions are reported
    once on the first call (verbose=True).
    """
    registered: list[str] = []
    collisions: list[str] = []
    for entry in catalog:
        action_name = entry.get("action_name")
        if not action_name:
            continue
        if action_name in actions:
            # Already registered (by us on a previous pass, or a name
            # collision with a non-MCP skill). Skip either way.
            existing = actions.get(action_name)
            if verbose and getattr(existing, "__name__", "") != action_name:
                collisions.append(action_name)
            continue
        actions[action_name] = _make_tool_action(
            mcp_client,
            entry["server"],
            entry["tool"],
            entry.get("schema") or {},
        )
        registered.append(action_name)

    if verbose and registered:
        head = ", ".join(registered[:6])
        more = f" +{len(registered) - 6} more" if len(registered) > 6 else ""
        print(f"  [mcp_tools] registered {len(registered)} MCP tool(s): {head}{more}")
    if verbose and collisions:
        print(
            f"  [mcp_tools] {len(collisions)} MCP tool(s) skipped due to name collision: "
            f"{', '.join(collisions[:5])}{'...' if len(collisions) > 5 else ''}"
        )
    return registered


# ── skill entry point ──────────────────────────────────────────────
def register(actions: dict) -> None:
    """Register management actions immediately; bring up MCP servers in a
    background thread so the skill loader does not stall on slow npx spawns.

    `mcp_client.bootstrap()` waits up to ~30 s per server (npx download +
    first init); with several servers configured that blocks JARVIS boot
    for minutes. We register the four management actions synchronously
    so `mcp_status`, `mcp_reload`, etc. are immediately available, then
    spawn a daemon thread that runs bootstrap and registers each tool's
    action into the same `actions` dict when discovery completes. Late
    registration is safe — dict __setitem__ is GIL-atomic, and the
    dispatcher only reads keys when handling a user command.
    """
    try:
        from core import mcp_client  # type: ignore
    except Exception as e:
        print(f"  [mcp_tools] core.mcp_client unavailable: {e}")
        return

    if not mcp_client.is_available():
        # Either the `mcp` SDK is not installed OR mcp_servers.json is
        # absent / empty. Stay silent — this is the default state on a
        # fresh install and we don't want to spam the boot log.
        return

    # Shared one-element holder so the bg thread can publish the catalog
    # without rebinding any name the management actions closed over.
    catalog_holder: list[list[dict]] = [[]]

    # Always expose the management actions immediately, even before any
    # server has come up — so `mcp_status` is reachable while bootstrap
    # is still running, and `mcp_reload` can pick up servers that came
    # online after the initial wait window.
    actions["mcp_status"]     = _make_status_action(mcp_client)
    actions["mcp_list_tools"] = _make_list_action(lambda: catalog_holder[0])
    actions["mcp_call"]       = _make_call_action(mcp_client)
    actions["mcp_reload"]     = _make_reload_action(mcp_client, actions, catalog_holder)

    def _bootstrap_in_background() -> None:
        try:
            catalog = mcp_client.bootstrap()
        except Exception as e:
            print(f"  [mcp_tools] bootstrap raised: {type(e).__name__}: {e}")
            return
        catalog_holder[0] = catalog
        _register_tool_actions(actions, mcp_client, catalog, verbose=True)

    threading.Thread(
        target=_bootstrap_in_background,
        name="mcp-tools-bootstrap",
        daemon=True,
    ).start()
