"""LOCAL_KEEP_ALIVE (2026-10-02): one setting for the keep_alive every
local-brain request sends. "20m" stays the shipped default; the owner sets -1
so the brain never unloads after a break (a 10 s reload was live at 14:47).

    python -m unittest tests.monolith.test_monolith_keep_alive
"""
from __future__ import annotations

import inspect
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class KeepAliveSettingTests(MonolithGlobalsTestCase):
    def _payload(self):
        return self.bc._local_chat_payload(
            "gemma-test", "sys", [{"role": "user", "content": "hi"}])

    def test_default_is_unchanged(self):
        import core.config as cfg
        self.assertEqual(cfg.LOCAL_KEEP_ALIVE, "20m")
        self.assertEqual(self._payload()["keep_alive"], "20m")

    def test_the_setting_reaches_the_chat_payload(self):
        for value in (-1, "2h", 3600):
            with self.subTest(value=value), \
                    mock.patch.object(self.bc, "LOCAL_KEEP_ALIVE", value,
                                      create=True):
                self.assertEqual(self._payload()["keep_alive"], value)

    def test_a_blank_or_broken_value_falls_back(self):
        for value in ("", "  ", None, True):
            with self.subTest(value=value), \
                    mock.patch.object(self.bc, "LOCAL_KEEP_ALIVE", value,
                                      create=True):
                self.assertEqual(self.bc._local_keep_alive(), "20m")

    def test_a_numeric_string_is_sent_as_a_number(self):
        # Live 2026-10-02: -1 saved in the settings arrived as "-1", and Ollama
        # answered HTTP 400 on every local request.
        for value, want in (("-1", -1), (" 3600 ", 3600), ("+60", 60),
                            ("24h", "24h"), ("-1m", "-1m")):
            with self.subTest(value=value), \
                    mock.patch.object(self.bc, "LOCAL_KEEP_ALIVE", value,
                                      create=True):
                self.assertEqual(self.bc._local_keep_alive(), want)

    def test_a_value_ollama_would_reject_falls_back_to_20m(self):
        # Review 2026-10-02: the settings loader stores str(saved value), so
        # a hand edit arrives here as "1.5", "3600.0", "None" (a JSON null),
        # "True", " 24h " or "1d". Each was sent as-is - HTTP 400 on EVERY
        # local request, the 15:05 outage again - and "--1" raised
        # ValueError out of the chat payload builder. A number of seconds
        # goes as a number, a duration Ollama parses goes as itself, anything
        # else falls back to "20m".
        for value, want in (("1.5", 1.5), ("3600.0", 3600), (" 24h ", "24h"),
                            ("1h30m", "1h30m"), ("500ms", "500ms"), ("0", 0),
                            (2.5, 2.5), ("None", "20m"), ("True", "20m"),
                            ("1d", "20m"), ("24H", "20m"), ("--1", "20m"),
                            ("1h 30m", "20m"), ("inf", "20m"),
                            (float("inf"), "20m"), (float("nan"), "20m")):
            with self.subTest(value=value), \
                    mock.patch.object(self.bc, "LOCAL_KEEP_ALIVE", value,
                                      create=True):
                self.assertEqual(self.bc._local_keep_alive(), want)
                self.assertEqual(self._payload()["keep_alive"], want)

    def test_no_local_request_hard_codes_its_own_keep_alive(self):
        src = inspect.getsource(self.bc)
        self.assertNotIn('"keep_alive": "20m"', src)
        self.assertNotIn('keep_alive="20m"', src)
        self.assertNotIn('payload["keep_alive"] = "20m"', src)


if __name__ == "__main__":
    unittest.main()
