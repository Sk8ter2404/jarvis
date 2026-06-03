"""Tests for core.config._apply_user_settings — the loader that overrides config
constants from data/user_settings.json (written by the tray Settings GUI).

CI-safe: no monolith, no real file (mocks os.path.exists + open). Mutations to
core.config are snapshotted + restored so the suite isn't polluted.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

import core.config as cfg


class ApplyUserSettingsTests(unittest.TestCase):
    def test_exists_and_missing_file_is_noop(self):
        self.assertTrue(callable(cfg._apply_user_settings))
        with mock.patch("core.config.os.path.exists", return_value=False):
            cfg._apply_user_settings()  # must not raise

    def test_overrides_matching_constants_with_coercion(self):
        # Snapshot every constant the fake file will touch.
        keys = ("AI_BACKEND", "CLAUDE_OPTIONAL", "PUSHBACK_MAX_CLOSE_WINDOWS")
        orig = {k: getattr(cfg, k) for k in keys}
        fake = {
            "AI_BACKEND": "ollama",                 # str
            "CLAUDE_OPTIONAL": False,               # bool
            "PUSHBACK_MAX_CLOSE_WINDOWS": "9",      # str -> int coercion
            "_README": "ignored",                   # underscore key skipped
            "TOTALLY_NOT_A_CONFIG_KEY": 123,        # unknown key skipped
        }
        try:
            m = mock.mock_open(read_data=json.dumps(fake))
            with mock.patch("core.config.os.path.exists", return_value=True), \
                 mock.patch("core.config.open", m, create=True):
                cfg._apply_user_settings()
            self.assertEqual(cfg.AI_BACKEND, "ollama")
            self.assertIs(cfg.CLAUDE_OPTIONAL, False)
            self.assertEqual(cfg.PUSHBACK_MAX_CLOSE_WINDOWS, 9)   # coerced int
            self.assertFalse(hasattr(cfg, "TOTALLY_NOT_A_CONFIG_KEY"))
        finally:
            for k, v in orig.items():
                setattr(cfg, k, v)

    def test_bad_value_keeps_default(self):
        orig = cfg.PUSHBACK_MAX_CLOSE_WINDOWS
        fake = {"PUSHBACK_MAX_CLOSE_WINDOWS": "not-a-number"}
        try:
            m = mock.mock_open(read_data=json.dumps(fake))
            with mock.patch("core.config.os.path.exists", return_value=True), \
                 mock.patch("core.config.open", m, create=True):
                cfg._apply_user_settings()      # int("not-a-number") raises -> skip
            self.assertEqual(cfg.PUSHBACK_MAX_CLOSE_WINDOWS, orig)
        finally:
            cfg.PUSHBACK_MAX_CLOSE_WINDOWS = orig

    def test_non_dict_payload_is_noop(self):
        m = mock.mock_open(read_data=json.dumps(["not", "a", "dict"]))
        with mock.patch("core.config.os.path.exists", return_value=True), \
             mock.patch("core.config.open", m, create=True):
            cfg._apply_user_settings()  # must not raise


if __name__ == "__main__":
    unittest.main()
