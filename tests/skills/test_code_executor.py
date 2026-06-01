"""Logic tests for skills/code_executor.py.

code_executor runs Python in a sandboxed subprocess. We exercise the parsing,
formatting, config, and the credential-stripping security guard WITHOUT running
untrusted code:
  * _strip_fence / _truncate / _format_result are pure — tested directly.
  * _sandbox_env (the security mitigation that removes credential-shaped env
    vars before a snippet inherits the environment) is tested directly.
  * _config_timeout / _config_backend env+default resolution.
  * run_python dispatch is tested with the subprocess backend MOCKED.

Exactly one end-to-end execution runs a trivial, safe `print(2+2)` snippet to
prove the local-exec path the skill is designed around actually works. Nothing
unsafe is ever executed.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class StripFenceTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_strips_python_fence(self):
        self.assertEqual(self.mod._strip_fence("```python\nprint(1)\n```"), "print(1)")

    def test_strips_bare_fence(self):
        self.assertEqual(self.mod._strip_fence("```\nx = 5\n```"), "x = 5")

    def test_leaves_unfenced_code(self):
        self.assertEqual(self.mod._strip_fence("print(42)"), "print(42)")


class TruncateTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_short_text_unchanged(self):
        self.assertEqual(self.mod._truncate("hello"), "hello")

    def test_long_text_truncated_with_marker(self):
        big = "x" * (self.mod._MAX_OUTPUT_CHARS + 500)
        out = self.mod._truncate(big)
        self.assertLess(len(out), len(big))
        self.assertIn("chars truncated", out)


class FormatResultTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_ok_header_and_stdout(self):
        out = self.mod._format_result("the answer is 4", "", 0, 0.5)
        self.assertIn("[ok in 0.5s]", out)
        self.assertIn("stdout:", out)
        self.assertIn("the answer is 4", out)

    def test_nonzero_exit_header(self):
        out = self.mod._format_result("", "Traceback...", 1, 0.2)
        self.assertIn("[exited with code 1", out)
        self.assertIn("stderr:", out)

    def test_timeout_header(self):
        out = self.mod._format_result("partial", "", None, 30.0)
        self.assertIn("timeout after 30.0s", out)
        self.assertIn("process killed", out)

    def test_no_output_success(self):
        out = self.mod._format_result("", "", 0, 0.1)
        self.assertIn("(no output)", out)


class SandboxEnvTests(unittest.TestCase):
    """The credential-stripping guard is the security-critical bit."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_strips_credential_shaped_vars(self):
        secrets = {
            "ANTHROPIC_API_KEY": "sk-xxx",
            "TELEGRAM_BOT_TOKEN": "123:abc",
            "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_x",
            "MY_SECRET": "shh",
            "DB_PASSWORD": "pw",
            "SOME_PRIVATE_KEY": "----",
            "PORCUPINE_ACCESS_KEY": "ak",
            "PATH": "/usr/bin",          # benign — must survive
            "NORMAL_VAR": "keep me",
        }
        with mock.patch.dict(os.environ, secrets, clear=True):
            env = self.mod._sandbox_env()
        # Every credential-shaped name is gone.
        for k in ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN",
                  "GITHUB_PERSONAL_ACCESS_TOKEN", "MY_SECRET", "DB_PASSWORD",
                  "SOME_PRIVATE_KEY", "PORCUPINE_ACCESS_KEY"):
            self.assertNotIn(k, env)
        # Benign vars survive.
        self.assertEqual(env.get("PATH"), "/usr/bin")
        self.assertEqual(env.get("NORMAL_VAR"), "keep me")

    def test_case_insensitive_match(self):
        # Windows normalises env-var names to uppercase, so compare in a
        # case-insensitive way: the credential-shaped name is stripped, the
        # benign one survives regardless of platform casing.
        with mock.patch.dict(os.environ, {"lowercase_api_key": "x", "benign": "y"},
                             clear=True):
            env = self.mod._sandbox_env()
        upper = {k.upper() for k in env}
        self.assertNotIn("LOWERCASE_API_KEY", upper)
        self.assertIn("BENIGN", upper)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_timeout_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod._config_timeout(), self.mod._DEFAULT_TIMEOUT)

    def test_timeout_from_env(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_TIMEOUT": "12"}, clear=True):
            self.assertEqual(self.mod._config_timeout(), 12)

    def test_timeout_bad_value_falls_back(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_TIMEOUT": "soon"}, clear=True):
            self.assertEqual(self.mod._config_timeout(), self.mod._DEFAULT_TIMEOUT)

    def test_backend_default_is_subprocess(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod._config_backend(), "subprocess")

    def test_backend_invalid_falls_back(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_BACKEND": "docker"}, clear=True):
            self.assertEqual(self.mod._config_backend(), "subprocess")

    def test_backend_jupyter_honoured(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_BACKEND": "JUPYTER"}, clear=True):
            self.assertEqual(self.mod._config_backend(), "jupyter")


class RunPythonDispatchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("code_executor")

    def test_empty_code_returns_format_hint(self):
        out = self.actions["run_python"]("")
        self.assertIn("format:", out)
        self.assertIn("run_python", out)

    def test_dispatches_to_subprocess_backend(self):
        # Force subprocess backend and mock the actual runner so no real
        # process is spawned — we only verify dispatch + fence-stripping.
        with mock.patch.object(self.mod, "_config_backend", return_value="subprocess"), \
             mock.patch.object(self.mod, "_run_subprocess",
                               return_value="[ok in 0.0s]\nstdout:\n4") as runner:
            out = self.actions["run_python"]("```python\nprint(2+2)\n```")
        self.assertIn("4", out)
        # Fence must be stripped before the snippet reaches the runner.
        self.assertEqual(runner.call_args[0][0], "print(2+2)")

    def test_jupyter_backend_dispatch(self):
        with mock.patch.object(self.mod, "_config_backend", return_value="jupyter"), \
             mock.patch.object(self.mod, "_run_jupyter", return_value="[ok in 0.0s]") as jrun:
            self.actions["run_python"]("x=1")
        jrun.assert_called_once()

    def test_reset_kernel_when_none_running(self):
        # No kernel was ever started in this fresh module → friendly message.
        out = self.actions["reset_kernel"]("")
        self.assertIn("no kernel", out.lower())


class RunPythonEndToEndTests(unittest.TestCase):
    """A single, trivial, SAFE real execution to prove the local-exec path
    the skill is designed around works end to end. No untrusted code."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("code_executor")

    def test_trivial_arithmetic_executes(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_BACKEND": "subprocess",
                                          "CODE_EXECUTOR_TIMEOUT": "30"}):
            out = self.actions["run_python"]("print(2 + 2)")
        self.assertIn("[ok", out)
        self.assertIn("4", out)


if __name__ == "__main__":
    unittest.main()
