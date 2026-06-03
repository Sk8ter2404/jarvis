"""Tests for tools/setup_wizard.py — the first-run onboarding flow.

CI-safe: stdlib only; reuses settings_window's load/save (no tkinter needed);
all I/O (stdin, .env, user_settings.json) is injected or pointed at temp files.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import tools.setup_wizard as suw


def _no_env_key():
    """Context manager: env with ANTHROPIC_API_KEY removed (restored on exit)."""
    ctx = mock.patch.dict(os.environ, {}, clear=False)

    class _C:
        def __enter__(self):
            ctx.start()
            os.environ.pop("ANTHROPIC_API_KEY", None)
            return self

        def __exit__(self, *a):
            ctx.stop()
            return False

    return _C()


def _in(*vals):
    it = iter(vals)
    return lambda prompt="": next(it)


def _first_then_blank(first):
    state = {"first": True}

    def fn(prompt=""):
        if state["first"]:
            state["first"] = False
            return first
        return ""
    return fn


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.dir, ignore_errors=True))
        self.env = os.path.join(self.dir, ".env")
        self.settings = os.path.join(self.dir, "user_settings.json")


class EnvKeyTests(_Tmp):
    def test_env_var_present(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk"}, clear=False):
            self.assertTrue(suw.env_has_api_key(self.env))

    def test_dotenv_has_key(self):
        with open(self.env, "w", encoding="utf-8") as f:
            f.write("ANTHROPIC_API_KEY=sk-ant-x\n")
        with _no_env_key():
            self.assertTrue(suw.env_has_api_key(self.env))

    def test_dotenv_empty_key(self):
        with open(self.env, "w", encoding="utf-8") as f:
            f.write("ANTHROPIC_API_KEY=\n")
        with _no_env_key():
            self.assertFalse(suw.env_has_api_key(self.env))

    def test_dotenv_no_key_line(self):
        with open(self.env, "w", encoding="utf-8") as f:
            f.write("OTHER=1\n")
        with _no_env_key():
            self.assertFalse(suw.env_has_api_key(self.env))

    def test_dotenv_missing(self):
        with _no_env_key():
            self.assertFalse(suw.env_has_api_key(self.env))


class UpsertEnvTests(_Tmp):
    def test_add_new_to_missing(self):
        self.assertTrue(suw.upsert_env(self.env, "ANTHROPIC_API_KEY", "sk-1"))
        self.assertIn("ANTHROPIC_API_KEY=sk-1",
                      open(self.env, encoding="utf-8").read())

    def test_update_existing_preserves_others(self):
        with open(self.env, "w", encoding="utf-8") as f:
            f.write("FOO=1\nANTHROPIC_API_KEY=old\nBAR=2\n")
        self.assertTrue(suw.upsert_env(self.env, "ANTHROPIC_API_KEY", "new"))
        c = open(self.env, encoding="utf-8").read()
        self.assertIn("ANTHROPIC_API_KEY=new", c)
        self.assertIn("FOO=1", c)
        self.assertIn("BAR=2", c)
        self.assertNotIn("old", c)

    def test_io_error_returns_false(self):
        with mock.patch.object(suw.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("io")):
            self.assertFalse(suw.upsert_env(self.env, "K", "v"))


class AskValueTests(unittest.TestCase):
    def test_bool_enter_keeps_current(self):
        self.assertTrue(suw._ask_value({"type": "bool", "label": "X"}, True, _in("")))
        self.assertFalse(suw._ask_value({"type": "bool", "label": "X"}, False, _in("")))

    def test_bool_yes_no(self):
        self.assertTrue(suw._ask_value({"type": "bool", "label": "X"}, False, _in("y")))
        self.assertFalse(suw._ask_value({"type": "bool", "label": "X"}, True, _in("n")))

    def test_enum_enter_keeps_current(self):
        spec = {"type": "enum", "label": "Y", "choices": ["a", "b"], "default": "a"}
        self.assertEqual(suw._ask_value(spec, "b", _in("")), "b")

    def test_enum_choice(self):
        spec = {"type": "enum", "label": "Y", "choices": ["a", "b"], "default": "a"}
        self.assertEqual(suw._ask_value(spec, "b", _in("a")), "a")

    def test_str_enter_keeps_current(self):
        self.assertEqual(suw._ask_value({"type": "str", "label": "Z"}, "hi", _in("")), "hi")

    def test_str_value(self):
        self.assertEqual(suw._ask_value({"type": "str", "label": "Z"}, "hi", _in("new")), "new")


class RunTests(_Tmp):
    def test_defaults_key_present(self):
        out = []
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk"}, clear=False):
            code = suw.run(defaults=True, env_path=self.env,
                           settings_path=self.settings, out=out.append)
        self.assertEqual(code, 0)
        self.assertIn("already set", "\n".join(out))
        self.assertTrue(os.path.exists(self.settings))

    def test_defaults_no_key(self):
        out = []
        with _no_env_key():
            code = suw.run(defaults=True, env_path=self.env,
                           settings_path=self.settings, out=out.append)
        self.assertEqual(code, 0)
        self.assertIn("set it in .env", "\n".join(out))

    def test_interactive_saves_key_and_settings(self):
        out = []
        with _no_env_key():
            code = suw.run(input_fn=_first_then_blank("sk-ant-zzz"),
                           env_path=self.env, settings_path=self.settings,
                           out=out.append)
        self.assertEqual(code, 0)
        self.assertIn("ANTHROPIC_API_KEY=sk-ant-zzz",
                      open(self.env, encoding="utf-8").read())
        self.assertIn("saved to .env", "\n".join(out))
        self.assertTrue(os.path.exists(self.settings))

    def test_interactive_blank_key(self):
        out = []
        with _no_env_key():
            code = suw.run(input_fn=_first_then_blank(""), env_path=self.env,
                           settings_path=self.settings, out=out.append)
        self.assertEqual(code, 0)
        self.assertIn("skipped", "\n".join(out))

    def test_key_write_failure(self):
        out = []
        with _no_env_key(), mock.patch.object(suw, "upsert_env", return_value=False):
            suw.run(input_fn=_first_then_blank("sk-ant-x"), env_path=self.env,
                    settings_path=self.settings, out=out.append)
        self.assertIn("could not write .env", "\n".join(out))

    def test_save_failure_returns_1(self):
        out = []
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk"}, clear=False), \
             mock.patch.object(suw.sw, "save_settings", side_effect=OSError("ro")):
            code = suw.run(defaults=True, env_path=self.env,
                           settings_path=self.settings, out=out.append)
        self.assertEqual(code, 1)
        self.assertIn("Could not save", "\n".join(out))


class MainTests(unittest.TestCase):
    def test_defaults_flag(self):
        with mock.patch.object(suw, "run", return_value=0) as r:
            suw.main(["--defaults"])
        self.assertTrue(r.call_args.kwargs["defaults"])

    def test_no_flag(self):
        with mock.patch.object(suw, "run", return_value=0) as r:
            suw.main([])
        self.assertFalse(r.call_args.kwargs["defaults"])


class ClaudeCliTests(unittest.TestCase):
    def test_found_on_path(self):
        with mock.patch("shutil.which", return_value=r"C:\bin\claude.exe"):
            self.assertEqual(suw.find_claude_cli(), r"C:\bin\claude.exe")

    def test_found_in_known_location(self):
        with mock.patch("shutil.which", return_value=None), \
             mock.patch.object(suw.os.path, "exists", return_value=True):
            self.assertIsNotNone(suw.find_claude_cli())

    def test_not_found(self):
        with mock.patch("shutil.which", return_value=None), \
             mock.patch.object(suw.os.path, "exists", return_value=False):
            self.assertIsNone(suw.find_claude_cli())


class ClaudeCheckRunTests(_Tmp):
    def test_reports_present(self):
        out = []
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk"}, clear=False), \
             mock.patch.object(suw, "find_claude_cli", return_value=r"C:\claude.exe"):
            suw.run(defaults=True, env_path=self.env, settings_path=self.settings,
                    out=out.append)
        self.assertIn("Claude Code detected", "\n".join(out))

    def test_reports_absent(self):
        out = []
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk"}, clear=False), \
             mock.patch.object(suw, "find_claude_cli", return_value=None):
            suw.run(defaults=True, env_path=self.env, settings_path=self.settings,
                    out=out.append)
        self.assertIn("install Claude Code", "\n".join(out))


if __name__ == "__main__":
    unittest.main()
