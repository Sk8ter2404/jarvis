"""Logic tests for skills/mcp_tools.py.

mcp_tools bridges Model Context Protocol servers into JARVIS actions. On a box
without the `mcp` SDK + a servers config, register() no-ops (0 actions), so the
meaningful surface is the pure helpers and the per-tool / management action
FACTORIES, which we drive directly with a fake mcp_client. No real MCP server,
no network, no subprocess.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class ParseArgsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_no_properties_returns_empty_dict(self):
        self.assertEqual(self.mod._parse_args("anything", {"properties": {}}), {})

    def test_missing_required_reports(self):
        schema = {"properties": {"path": {"type": "string"}}, "required": ["path"]}
        out = self.mod._parse_args("", schema)
        self.assertIsInstance(out, str)
        self.assertIn("Missing required args", out)
        self.assertIn("path", out)

    def test_json_object_used_directly(self):
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
        out = self.mod._parse_args('{"a": "x", "b": "y"}', schema)
        self.assertEqual(out, {"a": "x", "b": "y"})

    def test_single_required_shortcut(self):
        schema = {"properties": {"query": {"type": "string"}}, "required": ["query"]}
        self.assertEqual(self.mod._parse_args("hello world", schema),
                         {"query": "hello world"})

    def test_single_property_shortcut(self):
        schema = {"properties": {"q": {"type": "string"}}}
        self.assertEqual(self.mod._parse_args("foo", schema), {"q": "foo"})

    def test_multi_prop_no_json_returns_format_hint(self):
        # Two required props (no single-arg shortcut applies) + bare text that
        # isn't JSON → the skill surfaces a "Format:" hint listing the keys.
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                  "required": ["a", "b"]}
        out = self.mod._parse_args("bare text", schema)
        self.assertIsInstance(out, str)
        self.assertIn("Format:", out)
        self.assertIn("a", out)

    def test_int_coercion_through_single_required(self):
        schema = {"properties": {"count": {"type": "integer"}}, "required": ["count"]}
        self.assertEqual(self.mod._parse_args("42", schema), {"count": 42})


class CoerceTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_string_passthrough(self):
        self.assertEqual(self.mod._coerce("hi", {"type": "string"}), "hi")
        self.assertEqual(self.mod._coerce("hi", {}), "hi")

    def test_integer(self):
        self.assertEqual(self.mod._coerce("7", {"type": "integer"}), 7)

    def test_integer_bad_value_left_as_string(self):
        self.assertEqual(self.mod._coerce("seven", {"type": "integer"}), "seven")

    def test_number(self):
        self.assertEqual(self.mod._coerce("3.5", {"type": "number"}), 3.5)

    def test_boolean_truthy_and_falsy(self):
        self.assertIs(self.mod._coerce("yes", {"type": "boolean"}), True)
        self.assertIs(self.mod._coerce("off", {"type": "boolean"}), False)
        # Unrecognised boolean string falls back to the raw value.
        self.assertEqual(self.mod._coerce("maybe", {"type": "boolean"}), "maybe")

    def test_array_json_parsed(self):
        self.assertEqual(self.mod._coerce("[1, 2, 3]", {"type": "array"}), [1, 2, 3])

    def test_array_bad_json_left_as_string(self):
        self.assertEqual(self.mod._coerce("not json", {"type": "array"}), "not json")


class TruncateTextTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_short_unchanged(self):
        self.assertEqual(self.mod._truncate_text("hello", limit=100), "hello")

    def test_long_truncated(self):
        out = self.mod._truncate_text("x" * 50, limit=10)
        self.assertTrue(out.startswith("x" * 10))
        self.assertIn("truncated", out)

    def test_empty(self):
        self.assertEqual(self.mod._truncate_text("", limit=10), "")


class ToolActionFactoryTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_successful_call_returns_text(self):
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": True, "text": "file contents here"}
        action = self.mod._make_tool_action(client, "filesystem", "read_file",
                                            {"properties": {"path": {"type": "string"}},
                                             "required": ["path"]})
        out = action("/etc/hosts")
        self.assertEqual(out, "file contents here")
        client.call_tool.assert_called_once_with("filesystem", "read_file",
                                                 {"path": "/etc/hosts"})

    def test_failed_call_returns_error(self):
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": False, "error": "permission denied"}
        action = self.mod._make_tool_action(client, "fs", "read",
                                            {"properties": {"p": {"type": "string"}},
                                             "required": ["p"]})
        self.assertEqual(action("x"), "permission denied")

    def test_ok_but_no_output(self):
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": True, "text": ""}
        action = self.mod._make_tool_action(client, "srv", "ping",
                                            {"properties": {"p": {"type": "string"}},
                                             "required": ["p"]})
        self.assertIn("ok (no output)", action("x"))

    def test_bad_args_short_circuits_before_calling(self):
        client = mock.MagicMock()
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                  "required": ["a", "b"]}
        action = self.mod._make_tool_action(client, "srv", "tool", schema)
        out = action("")  # missing required → format hint, never calls client
        self.assertIn("Missing required args", out)
        client.call_tool.assert_not_called()

    def test_action_name_set(self):
        client = mock.MagicMock()
        action = self.mod._make_tool_action(client, "github", "list_repos", {})
        self.assertEqual(action.__name__, "mcp_github_list_repos")


class StatusActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_no_servers(self):
        client = mock.MagicMock()
        client.list_servers.return_value = {}
        out = self.mod._make_status_action(client)("")
        self.assertIn("No MCP servers configured", out)

    def test_renders_connected_and_offline(self):
        client = mock.MagicMock()
        client.list_servers.return_value = {
            "filesystem": {"connected": True, "transport": "stdio", "tool_count": 5},
            "slack": {"connected": False, "transport": "stdio", "tool_count": 0,
                      "error": "spawn failed"},
        }
        out = self.mod._make_status_action(client)("")
        self.assertIn("filesystem", out)
        self.assertIn("connected", out)
        self.assertIn("5 tool(s)", out)
        self.assertIn("slack", out)
        self.assertIn("offline", out)
        self.assertIn("spawn failed", out)


class ListActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def _catalog(self):
        return [
            {"action_name": "mcp_fs_read_file", "server": "fs", "tool": "read_file",
             "description": "Read a file from disk"},
            {"action_name": "mcp_github_list", "server": "github", "tool": "list",
             "description": "List repositories"},
        ]

    def test_empty_catalog(self):
        out = self.mod._make_list_action(lambda: [])("")
        self.assertIn("No MCP tools discovered", out)

    def test_lists_all(self):
        out = self.mod._make_list_action(self._catalog)("")
        self.assertIn("2 MCP tool(s)", out)
        self.assertIn("mcp_fs_read_file", out)
        self.assertIn("Read a file from disk", out)

    def test_filter_by_server(self):
        out = self.mod._make_list_action(self._catalog)("github")
        self.assertIn("mcp_github_list", out)
        self.assertNotIn("mcp_fs_read_file", out)

    def test_filter_no_match(self):
        out = self.mod._make_list_action(self._catalog)("notion")
        self.assertIn("No MCP tools matching 'notion'", out)


class CallActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_empty_arg_format_hint(self):
        client = mock.MagicMock()
        out = self.mod._make_call_action(client)("")
        self.assertIn("Format: mcp_call", out)

    def test_whitespace_parse_server_tool(self):
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": True, "text": "done"}
        out = self.mod._make_call_action(client)("filesystem read_file")
        self.assertEqual(out, "done")
        client.call_tool.assert_called_once_with("filesystem", "read_file", {})

    def test_comma_separated_with_json_args(self):
        # No spaces → whitespace split yields a single token, so the skill
        # falls back to comma-splitting (maxsplit 2) which preserves the JSON
        # arg blob.
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": True, "text": "ok"}
        self.mod._make_call_action(client)('fs,read_file,{"path":"/tmp/x"}')
        client.call_tool.assert_called_once_with("fs", "read_file", {"path": "/tmp/x"})

    def test_bad_json_args_reported(self):
        client = mock.MagicMock()
        out = self.mod._make_call_action(client)("fs read_file {not json}")
        self.assertIn("json_args parse failed", out)
        client.call_tool.assert_not_called()

    def test_non_object_json_args_rejected(self):
        client = mock.MagicMock()
        out = self.mod._make_call_action(client)("fs read_file [1,2,3]")
        self.assertIn("must be a JSON object", out)

    def test_comma_form_with_trailing_comma_on_server(self):
        # "fs, tool" → whitespace split leaves a trailing comma on the server
        # token, which flips the parse to the comma-separated form.
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": True, "text": "done"}
        out = self.mod._make_call_action(client)('fs, read_"file')
        self.assertEqual(out, "done")
        client.call_tool.assert_called_once_with("fs", 'read_"file', {})

    def test_single_token_after_split_format_hint(self):
        # One bare word → neither whitespace nor comma split yields 2 tokens.
        client = mock.MagicMock()
        out = self.mod._make_call_action(client)("onlyserver")
        self.assertIn("Format: mcp_call", out)
        client.call_tool.assert_not_called()

    def test_json_args_quotes_preserved(self):
        # Regression: shlex used to strip the JSON's double quotes, so the
        # documented form `mcp_call filesystem read_file {"path":"..."}`
        # always failed to parse. The JSON tail must reach json.loads verbatim.
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": True, "text": "contents"}
        out = self.mod._make_call_action(client)(
            'filesystem read_file {"path":"C:/temp/x.txt"}')
        self.assertEqual(out, "contents")
        client.call_tool.assert_called_once_with(
            "filesystem", "read_file", {"path": "C:/temp/x.txt"})

    def test_json_args_with_spaces_kept_whole(self):
        # Regression: shlex used to fragment JSON containing spaces across
        # multiple tokens, parsing only the first fragment.
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": True, "text": "ok"}
        self.mod._make_call_action(client)('srv tool {"a": 1, "b": "two words"}')
        client.call_tool.assert_called_once_with(
            "srv", "tool", {"a": 1, "b": "two words"})

    def test_comma_form_with_spaces_and_json_args(self):
        # Regression: "filesystem, read_file, {...}" (spaces after commas)
        # used to mis-parse — shlex yielded >=2 tokens with trailing commas,
        # so the comma-split branch never ran and the server name was wrong.
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": True, "text": "ok"}
        self.mod._make_call_action(client)(
            'filesystem, read_file, {"path": "/tmp/x"}')
        client.call_tool.assert_called_once_with(
            "filesystem", "read_file", {"path": "/tmp/x"})

    def test_failed_call_without_error_uses_generic_message(self):
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": False}   # no 'error' key
        out = self.mod._make_call_action(client)("fs read_file")
        self.assertEqual(out, "fs.read_file failed")


class RegisterToolActionsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_registers_new_actions(self):
        actions = {}
        client = mock.MagicMock()
        catalog = [
            {"action_name": "mcp_fs_read", "server": "fs", "tool": "read", "schema": {}},
            {"action_name": "mcp_fs_write", "server": "fs", "tool": "write", "schema": {}},
        ]
        added = self.mod._register_tool_actions(actions, client, catalog, verbose=False)
        self.assertEqual(sorted(added), ["mcp_fs_read", "mcp_fs_write"])
        self.assertIn("mcp_fs_read", actions)
        self.assertTrue(callable(actions["mcp_fs_write"]))

    def test_skips_collision_with_non_mcp_action(self):
        # A pre-existing non-MCP action with the same name must NOT be clobbered.
        def existing(_):
            return "native"
        actions = {"mcp_fs_read": existing}
        client = mock.MagicMock()
        catalog = [{"action_name": "mcp_fs_read", "server": "fs", "tool": "read",
                    "schema": {}}]
        added = self.mod._register_tool_actions(actions, client, catalog, verbose=False)
        self.assertEqual(added, [])
        self.assertIs(actions["mcp_fs_read"], existing)

    def test_idempotent_second_pass_adds_nothing(self):
        actions = {}
        client = mock.MagicMock()
        catalog = [{"action_name": "mcp_x_y", "server": "x", "tool": "y", "schema": {}}]
        self.mod._register_tool_actions(actions, client, catalog, verbose=False)
        again = self.mod._register_tool_actions(actions, client, catalog, verbose=False)
        self.assertEqual(again, [])

    def test_entry_without_action_name_skipped(self):
        actions = {}
        client = mock.MagicMock()
        catalog = [{"server": "x", "tool": "y", "schema": {}}]   # no action_name
        added = self.mod._register_tool_actions(actions, client, catalog, verbose=False)
        self.assertEqual(added, [])
        self.assertEqual(actions, {})

    def test_verbose_prints_registration_and_collision_summaries(self):
        # verbose=True path with >6 registered (the "+N more" tail) AND a
        # collision against a non-MCP action.
        def native(_):
            return "native"
        actions = {"mcp_collide": native}
        client = mock.MagicMock()
        catalog = [{"action_name": "mcp_collide", "server": "s", "tool": "t",
                    "schema": {}}]
        catalog += [{"action_name": f"mcp_s_t{i}", "server": "s", "tool": f"t{i}",
                     "schema": {}} for i in range(8)]
        added = self.mod._register_tool_actions(actions, client, catalog, verbose=True)
        self.assertEqual(len(added), 8)
        self.assertIs(actions["mcp_collide"], native)   # not clobbered


class ParseArgsExtraBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_empty_input_no_required_returns_empty_dict(self):
        schema = {"properties": {"opt": {"type": "string"}}}   # optional only
        self.assertEqual(self.mod._parse_args("", schema), {})

    def test_json_array_input_falls_through_to_shortcut(self):
        # '['-prefixed input parses to a list (not a dict) → JSON branch is
        # skipped, and the single-property shortcut treats the raw string as
        # the value (coerced to array).
        schema = {"properties": {"items": {"type": "array"}}}
        out = self.mod._parse_args("[1, 2]", schema)
        self.assertEqual(out, {"items": [1, 2]})

    def test_json_object_with_invalid_json_falls_through(self):
        # '{'-prefixed but not valid JSON → the try/except is swallowed and we
        # fall to the single-property shortcut.
        schema = {"properties": {"blob": {"type": "string"}}}
        out = self.mod._parse_args("{not valid", schema)
        self.assertEqual(out, {"blob": "{not valid"})


class CoerceExtraBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_number_bad_value_left_as_string(self):
        self.assertEqual(self.mod._coerce("not-a-number", {"type": "number"}),
                         "not-a-number")

    def test_object_json_parsed(self):
        self.assertEqual(self.mod._coerce('{"a": 1}', {"type": "object"}),
                         {"a": 1})

    def test_unknown_type_passthrough(self):
        self.assertEqual(self.mod._coerce("x", {"type": "weird"}), "x")


class ListActionTruncationTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_caps_at_60_and_reports_remainder(self):
        catalog = [{"action_name": f"mcp_s_t{i:03d}", "server": "s",
                    "tool": f"t{i}", "description": "d"} for i in range(75)]
        out = self.mod._make_list_action(lambda: catalog)("")
        self.assertIn("75 MCP tool(s)", out)
        self.assertIn("...and 15 more", out)        # 75 - 60

    def test_long_description_truncated(self):
        catalog = [{"action_name": "mcp_s_t", "server": "s", "tool": "t",
                    "description": "x" * 200}]
        out = self.mod._make_list_action(lambda: catalog)("")
        self.assertIn("...", out)

    def test_entry_without_description_renders_name_only(self):
        catalog = [{"action_name": "mcp_s_t", "server": "s", "tool": "t",
                    "description": ""}]
        out = self.mod._make_list_action(lambda: catalog)("")
        self.assertIn("mcp_s_t", out)


class ReloadActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def test_reload_reports_servers_and_newly_registered(self):
        client = mock.MagicMock()
        catalog = [{"action_name": "mcp_fs_new", "server": "fs", "tool": "new",
                    "schema": {}}]
        client.bootstrap.return_value = catalog
        client.list_servers.return_value = {
            "fs": {"connected": True}, "slack": {"connected": False}}
        actions = {}
        holder = [[]]
        out = self.mod._make_reload_action(client, actions, holder)("")
        self.assertIn("1/2 server(s) connected", out)
        self.assertIn("1 tool(s) discovered", out)
        self.assertIn("newly registered: mcp_fs_new", out)
        self.assertIn("restart JARVIS", out)
        self.assertIn("mcp_fs_new", actions)
        self.assertEqual(holder[0], catalog)

    def test_reload_with_many_new_tools_truncates_list(self):
        client = mock.MagicMock()
        catalog = [{"action_name": f"mcp_s_t{i}", "server": "s", "tool": f"t{i}",
                    "schema": {}} for i in range(8)]
        client.bootstrap.return_value = catalog
        client.list_servers.return_value = {"s": {"connected": True}}
        out = self.mod._make_reload_action(client, {}, [[]])("")
        self.assertIn("newly registered:", out)
        self.assertIn("+3 more", out)              # 8 listed, head shows 5

    def test_reload_no_new_tools_omits_registered_clause(self):
        client = mock.MagicMock()
        client.bootstrap.return_value = []
        client.list_servers.return_value = {"s": {"connected": True}}
        out = self.mod._make_reload_action(client, {}, [[]])("")
        self.assertNotIn("newly registered", out)
        self.assertIn("0 tool(s) discovered", out)

    def test_reload_bootstrap_failure_reported(self):
        client = mock.MagicMock()
        client.bootstrap.side_effect = RuntimeError("npx blew up")
        out = self.mod._make_reload_action(client, {}, [[]])("")
        self.assertIn("MCP reload failed", out)
        self.assertIn("RuntimeError", out)


class RegisterTests(unittest.TestCase):
    """register() does `from core import mcp_client`. We inject a fake module so
    no real MCP SDK / servers / subprocess are touched."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("mcp_tools")

    def _install_fake_core_mcp(self, fake):
        import sys
        import types
        core = sys.modules.get("core")
        created_core = False
        if core is None:
            core = types.ModuleType("core")
            sys.modules["core"] = core
            created_core = True
        had_attr = hasattr(core, "mcp_client")
        prev = getattr(core, "mcp_client", None)
        core.mcp_client = fake
        sys.modules["core.mcp_client"] = fake
        self.addCleanup(lambda: self._restore_core(core, created_core, had_attr, prev))

    def _restore_core(self, core, created_core, had_attr, prev):
        import sys
        sys.modules.pop("core.mcp_client", None)
        if created_core:
            sys.modules.pop("core", None)
        elif had_attr:
            core.mcp_client = prev
        else:
            try:
                delattr(core, "mcp_client")
            except AttributeError:
                pass

    def test_register_import_failure_noops(self):
        # Force `from core import mcp_client` to raise → silent return, no actions.
        import sys
        actions = {}
        fake = mock.MagicMock()
        # A module-like object whose attribute access raises on mcp_client.
        with mock.patch.dict(sys.modules, {"core.mcp_client": None}):
            # Patch importlib machinery indirectly: easiest is to make `core`
            # raise. Use builtins.__import__ shim scoped to this call.
            real_import = __import__

            def _imp(name, *a, **k):
                if name == "core" and a and "mcp_client" in (a[2] or []):
                    raise ImportError("no mcp sdk")
                return real_import(name, *a, **k)
            with mock.patch("builtins.__import__", _imp):
                self.mod.register(actions)
        self.assertEqual(actions, {})
        del fake

    def test_register_unavailable_client_noops(self):
        fake = mock.MagicMock()
        fake.is_available.return_value = False
        self._install_fake_core_mcp(fake)
        actions = {}
        self.mod.register(actions)
        self.assertEqual(actions, {})

    def test_register_available_wires_management_actions_and_bootstraps(self):
        import threading as _thr
        fake = mock.MagicMock()
        fake.is_available.return_value = True
        fake.bootstrap.return_value = [
            {"action_name": "mcp_fs_read", "server": "fs", "tool": "read",
             "schema": {}}]
        fake.list_servers.return_value = {"fs": {"connected": True}}
        self._install_fake_core_mcp(fake)
        actions = {}
        captured = {}
        with mock.patch.object(_thr.Thread, "start",
                               lambda self: captured.__setitem__("t", self._target)):
            self.mod.register(actions)
        # Management actions are registered synchronously.
        for name in ("mcp_status", "mcp_list_tools", "mcp_call", "mcp_reload"):
            self.assertIn(name, actions)
        # Drive the bootstrap thread body directly → registers the tool action.
        captured["t"]()
        self.assertIn("mcp_fs_read", actions)

    def test_register_bootstrap_thread_swallows_exception(self):
        import threading as _thr
        fake = mock.MagicMock()
        fake.is_available.return_value = True
        fake.bootstrap.side_effect = RuntimeError("bootstrap boom")
        self._install_fake_core_mcp(fake)
        actions = {}
        captured = {}
        with mock.patch.object(_thr.Thread, "start",
                               lambda self: captured.__setitem__("t", self._target)):
            self.mod.register(actions)
        captured["t"]()   # the bg bootstrap raising must be swallowed
        # Management actions still present; no tool actions added.
        self.assertIn("mcp_status", actions)


if __name__ == "__main__":
    unittest.main()
