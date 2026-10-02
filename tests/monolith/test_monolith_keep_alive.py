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

    def test_no_local_request_hard_codes_its_own_keep_alive(self):
        src = inspect.getsource(self.bc)
        self.assertNotIn('"keep_alive": "20m"', src)
        self.assertNotIn('keep_alive="20m"', src)
        self.assertNotIn('payload["keep_alive"] = "20m"', src)


if __name__ == "__main__":
    unittest.main()
