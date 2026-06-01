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

    def test_shlex_parse_server_tool(self):
        client = mock.MagicMock()
        client.call_tool.return_value = {"ok": True, "text": "done"}
        out = self.mod._make_call_action(client)("filesystem read_file")
        self.assertEqual(out, "done")
        client.call_tool.assert_called_once_with("filesystem", "read_file", {})

    def test_comma_separated_with_json_args(self):
        # No spaces → shlex yields a single token, so the skill falls back to
        # comma-splitting (maxsplit 2) which preserves the JSON arg blob.
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


if __name__ == "__main__":
    unittest.main()
