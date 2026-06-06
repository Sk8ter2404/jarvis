"""Tests for tools/settings_window.py — the JARVIS settings GUI.

CI-safe: the GUI (tkinter) is imported only lazily inside
settings_window.run_gui(), so importing the module here never needs a display
or the Tk libraries. These tests exercise the NON-GUI surface only — schema /
defaults, load+save round-trip, atomic write, --tab parsing, first-run file
creation, and the no-secret integration-status probe.

pyflakes-clean: no unused imports; optional deps are probed via
importlib.util.find_spec. PRIVACY: only fake fixture values are used — no real
keys, paths, or PII.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock

# Load the target module by file path so the test works whether or not
# `tools` is an importable package on the runner.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_MODULE_PATH = os.path.join(_PROJECT, "tools", "settings_window.py")

_spec = importlib.util.spec_from_file_location("jarvis_settings_window",
                                               _MODULE_PATH)
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)

# Is tkinter importable on this runner? (Used to gate the optional GUI probe.)
_HAS_TK = importlib.util.find_spec("tkinter") is not None


class SchemaTests(unittest.TestCase):
    def test_tab_order_and_labels_consistent(self):
        self.assertEqual(sw.TAB_ORDER,
                         ["voice", "ai", "privacy", "integrations", "advanced"])
        for tab in sw.TAB_ORDER:
            self.assertIn(tab, sw.TAB_LABELS)

    def test_every_field_has_a_known_tab(self):
        for key, spec in sw.SCHEMA.items():
            self.assertIn(spec.get("tab"), sw.TAB_ORDER,
                          msg=f"{key} has an unknown tab")

    def test_every_tab_has_at_least_one_field(self):
        used = {spec.get("tab") for spec in sw.SCHEMA.values()}
        for tab in sw.TAB_ORDER:
            self.assertIn(tab, used, msg=f"tab {tab} has no fields")

    def test_persisted_fields_have_defaults(self):
        for key in sw.persisted_keys():
            self.assertIn("default", sw.SCHEMA[key],
                          msg=f"{key} is persisted but has no default")

    def test_enum_defaults_are_valid_choices(self):
        for key, spec in sw.SCHEMA.items():
            if spec.get("type") == "enum":
                self.assertIn(spec["default"], spec["choices"],
                              msg=f"{key} default not in its choices")

    def test_status_rows_are_not_persisted(self):
        for key, spec in sw.SCHEMA.items():
            if spec.get("type") == "status":
                self.assertNotIn(key, sw.persisted_keys())
                self.assertTrue(key.startswith("_status_"))

    def test_defaults_match_config_known_values(self):
        # Spot-check that a few defaults mirror core/config.py's current values.
        d = sw.default_settings()
        self.assertEqual(d["VOICE_MODE"], "turn_based")
        self.assertEqual(d["TTS_VOICE"], "en-GB-RyanNeural")
        self.assertEqual(d["AI_BACKEND"], "claude")
        self.assertEqual(d["CLAUDE_MODEL"], "claude-sonnet-4-6")
        self.assertIs(d["CLAUDE_OPTIONAL"], True)
        self.assertAlmostEqual(d["VAD_THRESHOLD"], 0.008)

    def test_default_settings_returns_fresh_mutable_copies(self):
        a = sw.default_settings()
        a["SCREENSHOT_PRIVACY_BLOCKLIST"].append("MUTATED")
        b = sw.default_settings()
        self.assertNotIn("MUTATED", b["SCREENSHOT_PRIVACY_BLOCKLIST"])


class CoercionTests(unittest.TestCase):
    def test_bool_coercion_from_strings(self):
        spec = {"type": "bool", "default": False}
        for truthy in ("true", "True", "1", "yes", "on", "y"):
            self.assertIs(sw.coerce_value(spec, truthy), True)
        for falsy in ("false", "0", "no", "off", ""):
            self.assertIs(sw.coerce_value(spec, falsy), False)

    def test_enum_coercion_falls_back_on_bad_value(self):
        spec = {"type": "enum", "choices": ["a", "b"], "default": "a"}
        self.assertEqual(sw.coerce_value(spec, "b"), "b")
        self.assertEqual(sw.coerce_value(spec, "nope"), "a")

    def test_float_and_int_coercion(self):
        self.assertAlmostEqual(
            sw.coerce_value({"type": "float", "default": 0.0}, "1.5"), 1.5)
        self.assertEqual(
            sw.coerce_value({"type": "int", "default": 0}, "7"), 7)

    def test_bad_number_falls_back_to_default(self):
        self.assertAlmostEqual(
            sw.coerce_value({"type": "float", "default": 0.008}, "abc"), 0.008)

    def test_text_coercion_from_newline_string_and_list(self):
        spec = {"type": "text", "default": []}
        self.assertEqual(sw.coerce_value(spec, "a\n b \n\nc"), ["a", "b", "c"])
        self.assertEqual(sw.coerce_value(spec, ["x", "y"]), ["x", "y"])

    def test_routing(self):
        spec = {"type": "routing", "choices": ["auto", "local", "cloud"],
                "default": {"chat": "auto", "vision": "auto", "ambient": "auto"}}
        out = sw.coerce_value(spec, {"vision": "local"})          # partial merges
        self.assertEqual(out["vision"], "local")
        self.assertEqual(out["chat"], "auto")
        out2 = sw.coerce_value(spec, {"vision": "bogus", "nope": "local"})
        self.assertEqual(out2["vision"], "auto")                  # bad route dropped
        self.assertNotIn("nope", out2)                            # unknown fn dropped
        self.assertEqual(sw.coerce_value(spec, "garbage"),        # non-dict -> default
                         {"chat": "auto", "vision": "auto", "ambient": "auto"})


class DeviceTypeTests(unittest.TestCase):
    """The 'device' mic-picker type: coercion + label<->index translation."""

    SPEC = {"type": "device", "default": None}

    def test_device_coercion_int_and_numeric_string(self):
        self.assertEqual(sw.coerce_value(self.SPEC, 3), 3)
        self.assertEqual(sw.coerce_value(self.SPEC, "5"), 5)
        self.assertIsInstance(sw.coerce_value(self.SPEC, "5"), int)

    def test_device_coercion_negative_is_preserved(self):
        # -1 is the GUI's "Off (no mic)" hard-off contract — must survive.
        self.assertEqual(sw.coerce_value(self.SPEC, -1), -1)
        self.assertEqual(sw.coerce_value(self.SPEC, "-1"), -1)

    def test_device_coercion_none_and_blank_become_default(self):
        self.assertIsNone(sw.coerce_value(self.SPEC, None))
        self.assertIsNone(sw.coerce_value(self.SPEC, ""))
        self.assertIsNone(sw.coerce_value(self.SPEC, "   "))

    def test_device_coercion_bad_value_falls_back_to_default(self):
        # Non-numeric string must NOT raise (a hand-edited file can't crash it).
        self.assertIsNone(sw.coerce_value(self.SPEC, "garbage"))
        # And a non-None default is honoured on bad input.
        self.assertEqual(sw.coerce_value({"type": "device", "default": -1},
                                         "garbage"), -1)

    def test_device_coercion_bool_is_not_an_index(self):
        # bool is an int subclass; True must not be persisted as index 1.
        self.assertIsNone(sw.coerce_value(self.SPEC, True))

    def test_mic_choices_always_offer_auto_and_off(self):
        # With no live devices (mocked empty) the two synthetic choices remain.
        with mock.patch.object(sw, "list_input_devices", return_value=[]):
            choices = sw.mic_choices(saved_index=None)
        labels = [lbl for lbl, _ in choices]
        self.assertIn(sw.MIC_AUTO_LABEL, labels)
        self.assertIn(sw.MIC_OFF_LABEL, labels)
        self.assertEqual(dict((l, i) for l, i in choices)[sw.MIC_AUTO_LABEL],
                         None)
        self.assertEqual(dict((l, i) for l, i in choices)[sw.MIC_OFF_LABEL], -1)

    def test_mic_choices_includes_live_devices(self):
        live = [("[0] Mic A", 0), ("[2] Mic B", 2)]
        with mock.patch.object(sw, "list_input_devices", return_value=live):
            choices = sw.mic_choices(saved_index=None)
        self.assertIn(("[0] Mic A", 0), choices)
        self.assertIn(("[2] Mic B", 2), choices)

    def test_mic_choices_preserves_saved_but_absent_index(self):
        # A pinned USB mic that's currently unplugged must stay selectable so
        # the saved index round-trips (mirrors the combo branch keeping an
        # unknown Ollama tag visible).
        live = [("[0] Mic A", 0)]
        with mock.patch.object(sw, "list_input_devices", return_value=live):
            choices = sw.mic_choices(saved_index=7)
        idxs = [i for _, i in choices]
        self.assertIn(7, idxs)

    def test_mic_choices_does_not_duplicate_present_saved_index(self):
        live = [("[0] Mic A", 0), ("[2] Mic B", 2)]
        with mock.patch.object(sw, "list_input_devices", return_value=live):
            choices = sw.mic_choices(saved_index=2)
        self.assertEqual([i for _, i in choices].count(2), 1)

    def test_mic_label_index_translation_round_trip(self):
        live = [("[2] Mic B", 2)]
        with mock.patch.object(sw, "list_input_devices", return_value=live):
            choices = sw.mic_choices(saved_index=None)
        # index -> label -> index for each kind of choice
        for idx in (None, -1, 2):
            label = sw.mic_index_to_label(idx, choices)
            self.assertEqual(sw.mic_label_to_index(label, choices), idx)

    def test_mic_label_to_index_unknown_label_defaults_to_auto(self):
        choices = [(sw.MIC_AUTO_LABEL, None), (sw.MIC_OFF_LABEL, -1)]
        self.assertIsNone(sw.mic_label_to_index("nonexistent label", choices))

    def test_list_input_devices_is_import_safe_without_sounddevice(self):
        # Simulate a runner with no sounddevice: must return [] not raise.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "sounddevice" or name.startswith("sounddevice."):
                raise ImportError("no sounddevice on this runner")
            return real_import(name, *a, **k)

        with mock.patch.object(builtins, "__import__", side_effect=fake_import):
            self.assertEqual(sw.list_input_devices(), [])

    def test_list_input_devices_filters_to_input_capable(self):
        fake_devices = [
            {"name": "Speakers", "max_input_channels": 0},
            {"name": "Yeti", "max_input_channels": 2},
            {"name": "Webcam Mic", "max_input_channels": 1},
        ]
        fake_sd = mock.MagicMock()
        fake_sd.query_devices.return_value = fake_devices
        with mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            devices = sw.list_input_devices()
        self.assertEqual(devices, [("[1] Yeti", 1), ("[2] Webcam Mic", 2)])

    def test_microphone_index_is_a_persisted_device_key(self):
        self.assertEqual(sw.SCHEMA["MICROPHONE_INDEX"]["type"], "device")
        self.assertIn("MICROPHONE_INDEX", sw.persisted_keys())
        self.assertIsNone(sw.default_settings()["MICROPHONE_INDEX"])


class RoundTripTests(unittest.TestCase):
    def test_save_then_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            values = sw.default_settings()
            values["VOICE_MODE"] = "realtime"
            values["TTS_VOICE"] = "en-US-FakeVoice"
            values["VAD_THRESHOLD"] = 0.02
            values["SCREENSHOT_PRIVACY_BLOCKLIST"] = ["fakeapp", "vault"]
            sw.save_settings(values, path)

            loaded = sw.load_settings(path)
            self.assertEqual(loaded["VOICE_MODE"], "realtime")
            self.assertEqual(loaded["TTS_VOICE"], "en-US-FakeVoice")
            self.assertAlmostEqual(loaded["VAD_THRESHOLD"], 0.02)
            self.assertEqual(loaded["SCREENSHOT_PRIVACY_BLOCKLIST"],
                             ["fakeapp", "vault"])

    def test_saved_file_is_valid_json_with_all_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            sw.save_settings(sw.default_settings(), path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in sw.persisted_keys():
                self.assertIn(key, data)

    def test_load_missing_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "does_not_exist.json")
            self.assertEqual(sw.load_settings(path), sw.default_settings())

    def test_load_corrupt_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{ this is not json ]")
            self.assertEqual(sw.load_settings(path), sw.default_settings())

    def test_unknown_keys_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"FUTURE_KEY": "keep-me",
                           "VOICE_MODE": "realtime"}, f)
            loaded = sw.load_settings(path)
            self.assertEqual(loaded["FUTURE_KEY"], "keep-me")
            self.assertEqual(loaded["VOICE_MODE"], "realtime")
            # And a save must not drop the passthrough key.
            sw.save_settings(loaded, path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["FUTURE_KEY"], "keep-me")

    def test_string_values_in_file_coerced_for_typed_fields(self):
        # A hand-edited file storing a bool as the string "false" must load as
        # a real bool, and a numeric string as a real number.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"CLAUDE_OPTIONAL": "false",
                           "VAD_THRESHOLD": "0.05"}, f)
            loaded = sw.load_settings(path)
            self.assertIs(loaded["CLAUDE_OPTIONAL"], False)
            self.assertAlmostEqual(loaded["VAD_THRESHOLD"], 0.05)


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_write_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "deep", "user_settings.json")
            sw.atomic_write_json(path, {"a": 1})
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"a": 1})

    def test_atomic_write_leaves_no_temp_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            sw.atomic_write_json(path, {"a": 1})
            leftovers = [n for n in os.listdir(d) if n.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_atomic_write_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            sw.atomic_write_json(path, {"v": 1})
            sw.atomic_write_json(path, {"v": 2})
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["v"], 2)


class EnsureFileTests(unittest.TestCase):
    def test_ensure_creates_file_on_first_run(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            self.assertFalse(os.path.exists(path))
            result = sw.ensure_settings_file(path)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(result, sw.default_settings())

    def test_ensure_does_not_clobber_existing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            sw.save_settings({**sw.default_settings(),
                              "VOICE_MODE": "realtime"}, path)
            result = sw.ensure_settings_file(path)
            self.assertEqual(result["VOICE_MODE"], "realtime")


class ArgParseTests(unittest.TestCase):
    def test_no_tab_opens_first_tab(self):
        args = sw.parse_args([])
        self.assertIsNone(args.tab)
        self.assertEqual(sw.resolve_start_tab(args.tab), 0)

    def test_each_tab_name_resolves_to_its_index(self):
        for idx, name in enumerate(sw.TAB_ORDER):
            args = sw.parse_args(["--tab", name])
            self.assertEqual(args.tab, name)
            self.assertEqual(sw.resolve_start_tab(name), idx)

    def test_ai_tab_resolves(self):
        self.assertEqual(sw.resolve_start_tab(sw.parse_args(["--tab", "ai"]).tab),
                         sw.TAB_ORDER.index("ai"))

    def test_invalid_tab_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            sw.parse_args(["--tab", "bogus"])

    def test_resolve_unknown_tab_defaults_to_zero(self):
        self.assertEqual(sw.resolve_start_tab("nonexistent"), 0)
        self.assertEqual(sw.resolve_start_tab(None), 0)


class IntegrationStatusTests(unittest.TestCase):
    """The status probe must report PRESENCE only — never leak a secret."""

    def test_present_when_env_set(self):
        spec = {"type": "status", "secret_env": ["JARVIS_FAKE_TEST_KEY"]}
        os.environ["JARVIS_FAKE_TEST_KEY"] = "super-secret-fake-value"
        try:
            present, detail = sw.integration_status(spec)
        finally:
            del os.environ["JARVIS_FAKE_TEST_KEY"]
        self.assertTrue(present)
        # Detail is a status word, NOT the secret value.
        self.assertNotIn("super-secret-fake-value", detail)
        self.assertEqual(detail, "present")

    def test_not_set_when_env_absent(self):
        spec = {"type": "status",
                "secret_env": ["JARVIS_DEFINITELY_UNSET_KEY_XYZ"]}
        os.environ.pop("JARVIS_DEFINITELY_UNSET_KEY_XYZ", None)
        present, detail = sw.integration_status(spec)
        self.assertFalse(present)
        self.assertEqual(detail, "not set")

    def test_match_any_present_with_one_of_many(self):
        spec = {"type": "status", "match": "any",
                "secret_env": ["JARVIS_UNSET_A", "JARVIS_SET_B", "JARVIS_UNSET_C"]}
        for k in ("JARVIS_UNSET_A", "JARVIS_UNSET_C"):
            os.environ.pop(k, None)
        os.environ["JARVIS_SET_B"] = "x"
        try:
            present, _ = sw.integration_status(spec)
        finally:
            del os.environ["JARVIS_SET_B"]
        self.assertTrue(present)

    def test_blank_env_is_not_present(self):
        spec = {"type": "status", "secret_env": ["JARVIS_BLANK_KEY"]}
        os.environ["JARVIS_BLANK_KEY"] = "   "
        try:
            present, _ = sw.integration_status(spec)
        finally:
            del os.environ["JARVIS_BLANK_KEY"]
        self.assertFalse(present)

    def test_config_file_backed_status(self):
        # The Hue row is config-file backed (no env). With no file, not present.
        spec = {"type": "status", "secret_env": [],
                "config_file": "this_file_does_not_exist_fake.json"}
        present, detail = sw.integration_status(spec)
        self.assertFalse(present)

    def test_all_real_status_rows_probe_safely(self):
        # Every status row in the real schema must resolve without raising and
        # must return a (bool, str) where the str carries no secret.
        for key, spec in sw.SCHEMA.items():
            if spec.get("type") != "status":
                continue
            present, detail = sw.integration_status(spec)
            self.assertIsInstance(present, bool)
            self.assertIsInstance(detail, str)


class ExampleTemplateTests(unittest.TestCase):
    """The shipped tools/user_settings.example.json must mirror the schema
    defaults exactly (minus the _README preamble). It had drifted — KINECT_-
    ENABLED and MODEL_ROUTING (both wired keys) were missing, so anyone copying
    the template silently lost real settings (B18). This locks the two together
    so a future schema key can't ship without landing in the template too.
    """

    def _example(self) -> dict:
        with open(sw.EXAMPLE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_example_is_valid_json(self):
        self.assertIsInstance(self._example(), dict)

    def test_example_has_readme_and_no_secrets_shape(self):
        data = self._example()
        self.assertIn("_README", data)
        # No status (_status_*) rows and no secret-bearing keys leak in.
        for key in data:
            self.assertFalse(key.startswith("_status_"),
                             msg=f"{key} status row must not be in the template")

    def test_example_matches_default_settings(self):
        data = self._example()
        data.pop("_README", None)
        self.assertEqual(
            data, sw.default_settings(),
            msg="tools/user_settings.example.json is out of sync with the "
                "schema defaults. Regenerate it from default_settings() "
                "(keep the _README key).")

    def test_example_contains_the_previously_missing_wired_keys(self):
        data = self._example()
        for key in ("KINECT_ENABLED", "MODEL_ROUTING",
                    "MICROPHONE_INDEX", "PREFERRED_INPUT_DEVICES"):
            self.assertIn(key, data, msg=f"{key} missing from the template")


class DeviceRoundTripTests(unittest.TestCase):
    """A device index must survive save -> on-disk JSON -> load as the int."""

    def test_microphone_index_persists_as_int(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            values = sw.default_settings()
            values["MICROPHONE_INDEX"] = 4
            sw.save_settings(values, path)
            with open(path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["MICROPHONE_INDEX"], 4)
            self.assertIsInstance(on_disk["MICROPHONE_INDEX"], int)
            self.assertEqual(sw.load_settings(path)["MICROPHONE_INDEX"], 4)

    def test_microphone_index_off_persists_negative(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            values = sw.default_settings()
            values["MICROPHONE_INDEX"] = sw.MIC_OFF_INDEX  # "Off (no mic)"
            sw.save_settings(values, path)
            self.assertEqual(sw.load_settings(path)["MICROPHONE_INDEX"], -1)

    def test_microphone_index_auto_persists_null(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "user_settings.json")
            values = sw.default_settings()
            values["MICROPHONE_INDEX"] = None  # "System default (auto)"
            sw.save_settings(values, path)
            with open(path, "r", encoding="utf-8") as f:
                self.assertIsNone(json.load(f)["MICROPHONE_INDEX"])


@unittest.skipUnless(_HAS_TK, "tkinter not available on this runner")
class GuiSmokeTests(unittest.TestCase):
    """Optional: only the import-ability of the GUI entry point, no display."""

    def test_run_gui_is_callable(self):
        self.assertTrue(callable(sw.run_gui))

    def test_main_is_callable(self):
        self.assertTrue(callable(sw.main))


if __name__ == "__main__":
    unittest.main(verbosity=2)
