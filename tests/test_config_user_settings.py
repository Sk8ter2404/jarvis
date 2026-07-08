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

    def test_microphone_index_lives_in_config(self):
        # The mic picker's blocker fix: MICROPHONE_INDEX (and the preferred-list)
        # must be core/config.py constants so _apply_user_settings can override
        # them. Before the move they were redeclared in bobert_companion.py AFTER
        # `from core.config import *`, silently shadowing any saved value.
        for k in ("MICROPHONE_INDEX", "SPEAKER_INDEX",
                  "PREFERRED_INPUT_DEVICES", "PREFERRED_OUTPUT_DEVICES"):
            self.assertTrue(hasattr(cfg, k), msg=f"{k} missing from core.config")
        self.assertIsNone(cfg.MICROPHONE_INDEX)  # default = auto / preferred lookup

    def _apply(self, fake: dict) -> None:
        m = mock.mock_open(read_data=json.dumps(fake))
        with mock.patch("core.config.os.path.exists", return_value=True), \
             mock.patch("core.config.open", m, create=True):
            cfg._apply_user_settings()

    def test_microphone_index_int_round_trips(self):
        # An int index written by the GUI must reach the runtime constant. This
        # is the exact path that was broken: a None default carries no type, so
        # the old coercion ladder skipped it and the saved value was dropped.
        orig = cfg.MICROPHONE_INDEX
        try:
            cfg.MICROPHONE_INDEX = None
            self._apply({"MICROPHONE_INDEX": 3})
            self.assertEqual(cfg.MICROPHONE_INDEX, 3)
        finally:
            cfg.MICROPHONE_INDEX = orig

    def test_microphone_index_numeric_string_coerces_to_int(self):
        orig = cfg.MICROPHONE_INDEX
        try:
            cfg.MICROPHONE_INDEX = None
            self._apply({"MICROPHONE_INDEX": "5"})
            self.assertEqual(cfg.MICROPHONE_INDEX, 5)
            self.assertIsInstance(cfg.MICROPHONE_INDEX, int)
        finally:
            cfg.MICROPHONE_INDEX = orig

    def test_microphone_index_negative_hard_off_round_trips(self):
        # A negative index is the "no mic / hard-off" contract the monolith
        # honours (bobert_companion._mic_input_disabled). It must survive the
        # override path so the GUI's "Off (no mic)" choice actually takes effect.
        orig = cfg.MICROPHONE_INDEX
        try:
            cfg.MICROPHONE_INDEX = None
            self._apply({"MICROPHONE_INDEX": -1})
            self.assertEqual(cfg.MICROPHONE_INDEX, -1)
        finally:
            cfg.MICROPHONE_INDEX = orig

    def test_microphone_index_null_and_blank_stay_none(self):
        # null / "" => auto (None). The GUI persists null for "System default".
        # At the real apply point the constant is its None default, so a null
        # override must leave it None (not raise, not become some int).
        orig = cfg.MICROPHONE_INDEX
        try:
            for val in (None, ""):
                cfg.MICROPHONE_INDEX = None      # the real default at apply time
                self._apply({"MICROPHONE_INDEX": val})
                self.assertIsNone(cfg.MICROPHONE_INDEX,
                                  msg=f"{val!r} should keep None (auto)")
        finally:
            cfg.MICROPHONE_INDEX = orig

    def test_microphone_index_garbage_keeps_none_default(self):
        # A non-numeric value must not raise and must leave the None default
        # intact (never crashes the import on a hand-edited settings file).
        orig = cfg.MICROPHONE_INDEX
        try:
            cfg.MICROPHONE_INDEX = None
            self._apply({"MICROPHONE_INDEX": "not-a-number"})
            self.assertIsNone(cfg.MICROPHONE_INDEX)
        finally:
            cfg.MICROPHONE_INDEX = orig

    def test_preferred_input_devices_list_override(self):
        orig = list(cfg.PREFERRED_INPUT_DEVICES)
        try:
            self._apply({"PREFERRED_INPUT_DEVICES": ["Yeti", "Realtek"]})
            self.assertEqual(cfg.PREFERRED_INPUT_DEVICES, ["Yeti", "Realtek"])
        finally:
            cfg.PREFERRED_INPUT_DEVICES = orig

    def test_string_bool_is_parsed_not_bare_bool(self):
        # #18: a JSON *string* boolean the Settings GUI accepts ("true"/"false")
        # must be parsed the same way settings_window.coerce_value does. A bare
        # bool("false") is True, which would flip the runtime constant OPPOSITE
        # to the GUI's intent (runtime and GUI silently disagree).
        orig = cfg.CLAUDE_OPTIONAL
        try:
            cfg.CLAUDE_OPTIONAL = True
            self._apply({"CLAUDE_OPTIONAL": "false"})
            self.assertIs(cfg.CLAUDE_OPTIONAL, False)   # NOT True

            cfg.CLAUDE_OPTIONAL = False
            self._apply({"CLAUDE_OPTIONAL": "true"})
            self.assertIs(cfg.CLAUDE_OPTIONAL, True)

            # The wider truthy vocabulary coerce_value honours also applies.
            cfg.CLAUDE_OPTIONAL = True
            self._apply({"CLAUDE_OPTIONAL": "off"})
            self.assertIs(cfg.CLAUDE_OPTIONAL, False)

            cfg.CLAUDE_OPTIONAL = False
            self._apply({"CLAUDE_OPTIONAL": "yes"})
            self.assertIs(cfg.CLAUDE_OPTIONAL, True)

            # A real JSON bool still round-trips unchanged.
            cfg.CLAUDE_OPTIONAL = True
            self._apply({"CLAUDE_OPTIONAL": False})
            self.assertIs(cfg.CLAUDE_OPTIONAL, False)
        finally:
            cfg.CLAUDE_OPTIONAL = orig

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


class DocstringHonestyTests(unittest.TestCase):
    """_apply_user_settings() runs ONCE at import, so a user_settings.json flip
    only takes effect on the next start — a fresh `import core.config` returns
    the already-cached module without re-reading the file. The STREAMING_AUTO_-
    FULLSCREEN / BARGE_IN_ENABLED comments used to overpromise 'takes effect
    without a restart'; this guards the corrected wording so the docs can't
    silently drift back to the false claim.
    """

    @staticmethod
    def _source() -> str:
        import os
        cfg_path = os.path.join(os.path.dirname(cfg.__file__), "config.py")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_no_without_a_restart_overpromise(self):
        self.assertNotIn(
            "without a restart", self._source(),
            msg="core/config.py claims a live flip 'without a restart' but "
                "_apply_user_settings() runs once at import — correct the "
                "comment to 'on the next start'.")

    def test_apply_user_settings_runs_at_import_time(self):
        # The behavioural fact the docs must match: the loader is a module-level
        # call, not re-run per read.
        self.assertIn("_apply_user_settings()", self._source())


if __name__ == "__main__":
    unittest.main()
