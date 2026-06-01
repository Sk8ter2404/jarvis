"""Logic tests for skills/browser_agent.py.

browser-use (+ Playwright + an Anthropic key) is an optional dep absent from
the test env, so is_available() is False and every action returns the install
hint — the easy, high-value degradation path. The genuinely interesting pure
logic is the SSRF guard, which we test directly (no DNS for literal IPs;
socket.gethostbyname mocked for hostnames), plus intent prefixing and the
truncation/model helpers. No browser, no asyncio loop, no network.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


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


if __name__ == "__main__":
    unittest.main()
