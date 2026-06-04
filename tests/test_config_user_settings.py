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

    def test_legacy_ambient_listening_key_aliases_to_new_name(self):
        # v1.20.0 renamed AMBIENT_LISTENING_ENABLED -> AMBIENT_LISTEN_ENABLED.
        # A saved user_settings.json carrying the OLD key must still flip the
        # NEW constant, otherwise the mic-ambient daemon never autostarts and
        # nothing is learned (the user-reported regression). The legacy key is
        # NOT a real config constant, so without the alias the `key not in g`
        # guard would silently drop it.
        orig = cfg.AMBIENT_LISTEN_ENABLED
        self.assertFalse(hasattr(cfg, "AMBIENT_LISTENING_ENABLED"))  # legacy name is gone
        fake = {"AMBIENT_LISTENING_ENABLED": True}
        try:
            cfg.AMBIENT_LISTEN_ENABLED = False
            m = mock.mock_open(read_data=json.dumps(fake))
            with mock.patch("core.config.os.path.exists", return_value=True), \
                 mock.patch("core.config.open", m, create=True):
                cfg._apply_user_settings()
            self.assertIs(cfg.AMBIENT_LISTEN_ENABLED, True)             # alias applied
            self.assertFalse(hasattr(cfg, "AMBIENT_LISTENING_ENABLED"))  # no stray attr created
        finally:
            cfg.AMBIENT_LISTEN_ENABLED = orig


class ModelRoutingTests(unittest.TestCase):
    def test_model_route_default_and_lookup(self):
        self.assertEqual(cfg.model_route("nonexistent_fn"), "auto")
        with mock.patch.object(cfg, "MODEL_ROUTING", {"vision": "local"}):
            self.assertEqual(cfg.model_route("vision"), "local")
            self.assertEqual(cfg.model_route("chat"), "auto")   # missing key -> default

    def test_partial_dict_override_merges(self):
        orig = dict(cfg.MODEL_ROUTING)
        fake = {"MODEL_ROUTING": {"vision": "local"}}   # PARTIAL override
        try:
            m = mock.mock_open(read_data=json.dumps(fake))
            with mock.patch("core.config.os.path.exists", return_value=True), \
                 mock.patch("core.config.open", m, create=True):
                cfg._apply_user_settings()
            self.assertEqual(cfg.MODEL_ROUTING["vision"], "local")       # changed
            self.assertEqual(cfg.MODEL_ROUTING["chat"], orig["chat"])    # kept (merge)
            self.assertEqual(cfg.MODEL_ROUTING["ambient"], orig["ambient"])
        finally:
            cfg.MODEL_ROUTING = orig


if __name__ == "__main__":
    unittest.main()
