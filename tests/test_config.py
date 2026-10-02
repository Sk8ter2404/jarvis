"""Light sanity tests for core.config.

config.py is almost entirely dumb constants, so this suite stays minimal: it
pins the env-driven knobs (USER_NAME / BAMBU_*) actually read os.getenv with a
blank default, the safety CONFIRM_KEYWORDS list is non-empty, and a couple of
structural invariants other modules rely on (CONSOLE_MONITOR ∈ MONITORS, the
RAG paths expand ~). The env tests reload the module under a patched
environment so they assert the read semantics, not a committed personal value.

stdlib unittest + importlib only.
"""
from __future__ import annotations

import importlib
import json
import os
import unittest
from unittest import mock

from core import config


class EnvDrivenTests(unittest.TestCase):
    """USER_NAME and the BAMBU_* secrets must come from the environment with a
    blank default — no personal value is ever committed to the repo."""

    def _reload_with_env(self, **env):
        with mock.patch.dict(os.environ, env, clear=False):
            return importlib.reload(config)

    def tearDown(self):
        # Restore the module to the ambient (unpatched) environment so a
        # patched value can't leak into later tests in the process.
        importlib.reload(config)

    def test_user_name_reads_env(self):
        mod = self._reload_with_env(JARVIS_USER_NAME="Tony")
        self.assertEqual(mod.USER_NAME, "Tony")

    def test_user_name_blank_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_USER_NAME", None)
            mod = importlib.reload(config)
        self.assertEqual(mod.USER_NAME, "")

    def test_bambu_creds_read_env(self):
        mod = self._reload_with_env(
            BAMBU_PRINTER_IP="192.168.1.50",
            BAMBU_ACCESS_CODE="12345678",
            BAMBU_SERIAL="SN-XYZ",
        )
        self.assertEqual(mod.BAMBU_PRINTER_IP, "192.168.1.50")
        self.assertEqual(mod.BAMBU_ACCESS_CODE, "12345678")
        self.assertEqual(mod.BAMBU_SERIAL, "SN-XYZ")

    def test_bambu_creds_blank_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("BAMBU_PRINTER_IP", "BAMBU_ACCESS_CODE", "BAMBU_SERIAL"):
                os.environ.pop(k, None)
            mod = importlib.reload(config)
        self.assertEqual(mod.BAMBU_PRINTER_IP, "")
        self.assertEqual(mod.BAMBU_ACCESS_CODE, "")
        self.assertEqual(mod.BAMBU_SERIAL, "")

    def test_smart_turn_mode_reads_env_and_ships_shadow(self):
        # Speed plan R7: JARVIS_SMART_TURN picks off / shadow / on (any case,
        # stray spaces); unset or blank is the shipped 'shadow'.
        mod = self._reload_with_env(JARVIS_SMART_TURN=" On ")
        self.assertEqual(mod.SMART_TURN_MODE, "on")
        mod = self._reload_with_env(JARVIS_SMART_TURN="")
        self.assertEqual(mod.SMART_TURN_MODE, "shadow")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_SMART_TURN", None)
            mod = importlib.reload(config)
        self.assertEqual(mod.SMART_TURN_MODE, "shadow")

    def test_env_only_secrets_are_never_applied_from_settings(self):
        # user_settings.json must NOT be able to override the env-only BAMBU
        # secrets — the printer-reconnect flow once wrote BAMBU_PRINTER_IP there,
        # which defeated the env contract and broke these tests. A benign key in
        # the SAME payload still applies, proving the guard is targeted.
        payload = {"BAMBU_PRINTER_IP": "10.10.10.10", "USER_NAME": "SettingsGuy"}
        with mock.patch.dict(os.environ, {"BAMBU_PRINTER_IP": "192.168.5.5"},
                             clear=False):
            mod = importlib.reload(config)
            self.assertEqual(mod.BAMBU_PRINTER_IP, "192.168.5.5")
            # `open` must be mocked too, not just exists + json.load:
            # data/user_settings.json is gitignored (.gitignore `data/*`), so on
            # a clean checkout os.path.exists lies True, open() then raises
            # FileNotFoundError, _apply_user_settings swallows it and returns
            # early -- and USER_NAME stayed "" instead of the settings value.
            # That passed only on a machine where the real file happens to
            # exist, and errored on every CI run.
            with mock.patch("os.path.exists", return_value=True), \
                 mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
                 mock.patch("json.load", return_value=payload):
                mod._apply_user_settings()
            self.assertEqual(mod.BAMBU_PRINTER_IP, "192.168.5.5")  # secret intact
            self.assertEqual(mod.USER_NAME, "SettingsGuy")          # benign applied
        self.assertIn("BAMBU_PRINTER_IP", config._ENV_ONLY_KEYS)


class SafetyConstantTests(unittest.TestCase):
    def test_confirm_keywords_non_empty_list(self):
        self.assertIsInstance(config.CONFIRM_KEYWORDS, list)
        self.assertTrue(config.CONFIRM_KEYWORDS)
        # The destructive verbs the safety layer keys on must be present.
        for kw in ("delete", "format", "transfer"):
            self.assertIn(kw, config.CONFIRM_KEYWORDS)

    def test_confirm_keywords_all_strings(self):
        self.assertTrue(all(isinstance(k, str) for k in config.CONFIRM_KEYWORDS))

    def test_local_vision_fallback_defaults_off(self):
        # Shipped default MUST be False: on a 24 GB card the resident ~21 GB
        # 30B text model leaves no room to co-load the ~7 GB local VLM, so a
        # Claude-vision error (incl. a transient API cap / network blip) that
        # silently pulled the VLM would over-commit and brick the GPU. The
        # safe-for-everyone default is OFF; a box with the VRAM headroom opts
        # in via user_settings.json.
        #
        # 2026-07-07: assert the SHIPPED default only when it is NOT overridden.
        # _apply_user_settings() mutates this constant at import from the live
        # data/user_settings.json, so on the owner's opted-in box (or any deploy
        # tree carrying a settings file) the effective value is legitimately
        # True. A blanket assertFalse then failed in the LIVE tree even though
        # the opt-in is intended — the same test-isolation trap the bambu offline
        # test had. Skip the value assertion when the owner explicitly set the
        # key; a clean checkout / CI (no settings file) still enforces the safe
        # default.
        self.assertIsInstance(config.LOCAL_VISION_FALLBACK, bool)
        settings_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(config.__file__))),
            "data", "user_settings.json")
        overridden = False
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    overridden = "LOCAL_VISION_FALLBACK" in (json.load(f) or {})
            except Exception:
                overridden = False
        if not overridden:
            self.assertFalse(config.LOCAL_VISION_FALLBACK)


class StructuralInvariantTests(unittest.TestCase):
    def test_console_monitor_is_a_known_monitor(self):
        # CONSOLE_MONITOR must name a key in MONITORS (or be blank).
        if config.CONSOLE_MONITOR:
            self.assertIn(config.CONSOLE_MONITOR, config.MONITORS)

    def test_hud_monitor_is_a_known_monitor(self):
        if config.HUD_MONITOR:
            self.assertIn(config.HUD_MONITOR, config.MONITORS)

    def test_monitor_tuples_are_four_ints(self):
        for name, geom in config.MONITORS.items():
            self.assertEqual(len(geom), 4, name)
            self.assertTrue(all(isinstance(v, int) for v in geom), name)

    def test_rag_paths_are_absolute_and_expanded(self):
        # RAG_INDEX_PATHS is the one import-time computed value (expanduser).
        self.assertTrue(config.RAG_INDEX_PATHS)
        for p in config.RAG_INDEX_PATHS:
            self.assertNotIn("~", p)
            self.assertTrue(os.path.isabs(p), p)

    def test_cameras_have_one_primary(self):
        primaries = [c for c in config.CAMERAS if c.get("primary")]
        self.assertEqual(len(primaries), 1)

    def test_answer_first_ships_on(self):
        # 2026-09-29: the short lead-in before a spoken answer is skipped by
        # default; the kill switch is a plain bool.
        self.assertIs(config.ANSWER_FIRST_ENABLED, True)

    def test_processing_filler_ships_off_with_float_delays(self):
        # 2026-09-29: the filler is OFF until proven by ear; both delays are
        # float literals (an int default would make _apply_user_settings
        # truncate a saved 2.5 to 2) and stage 2 comes after stage 1.
        self.assertIs(config.PROCESSING_FILLER_ENABLED, False)
        self.assertIsInstance(config.PROCESSING_FILLER_DELAY, float)
        self.assertIsInstance(config.PROCESSING_FILLER_STILL_DELAY, float)
        self.assertGreater(config.PROCESSING_FILLER_STILL_DELAY,
                           config.PROCESSING_FILLER_DELAY)

    def test_prompt_freeze_settings_ship_on_with_a_float_window(self):
        # 2026-09-29 local prompt-prefix stability: the quiet window is a
        # float literal (an int default would make _apply_user_settings
        # truncate a saved 12.5 to 12) and the idle re-prime ships ON.
        self.assertIsInstance(config.PROMPT_FREEZE_QUIET_S, float)
        self.assertEqual(config.PROMPT_FREEZE_QUIET_S, 30.0)
        self.assertIs(config.LOCAL_PREFIX_REPRIME, True)
        # v2.0.139: the boot warm-up delay is a float literal, ships ON.
        self.assertIsInstance(config.LOCAL_REPRIME_AT_BOOT_S, float)
        self.assertEqual(config.LOCAL_REPRIME_AT_BOOT_S, 20.0)

    def test_background_traffic_settings_ship_on_with_float_windows(self):
        # 2026-09-29 (r6): both knobs are float literals (an int default would
        # make _apply_user_settings truncate a saved 90.5 to 90), both ship
        # ON (a positive window), 0.0 is documented as "off", and the comment
        # promises "next start" like the other import-time knobs.
        self.assertIsInstance(config.LOCAL_BACKGROUND_MAX_DEFER_S, float)
        self.assertEqual(config.LOCAL_BACKGROUND_MAX_DEFER_S, 120.0)
        self.assertIsInstance(config.LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S,
                              float)
        self.assertEqual(config.LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S, 600.0)
        with open(config.__file__, encoding="utf-8") as fh:
            src = fh.read()
        block = src[src.index("Local background traffic (2026-09-29)"):
                    src.index("LOCAL_BACKGROUND_MAX_DEFER_S = 120.0")]
        self.assertIn("0.0 turns the waiting off", block)
        self.assertIn("0.0 turns it off", block)
        self.assertIn("apply on the next start", block)

    def test_sentence_tts_ships_on_and_says_next_start(self):
        # 2026-09-29: per-sentence Kokoro speech is ON by default (a bool, so
        # _apply_user_settings keeps a saved true/false), and its comment
        # promises "next start" like every other import-time knob.
        self.assertIs(config.SENTENCE_TTS_ENABLED, True)
        with open(config.__file__, encoding="utf-8") as fh:
            src = fh.read()
        block = src[src.index("Per-sentence speech (Kokoro)"):
                    src.index("SENTENCE_TTS_ENABLED = True")]
        self.assertIn("apply on", block)
        self.assertIn("the next start", block)

    def test_audio_flap_knobs_are_typed_and_say_next_start(self):
        # 2026-09-29 audio-device flap damping: the three seconds knobs are
        # floats (an int default would make _apply_user_settings truncate a
        # saved 7.5 to 7), the threshold is a plain int, and the block says
        # when a change applies. core/audio_flap.py's own tests pin the
        # shipped VALUES against its defaults.
        for name in ("AUDIO_FLAP_WINDOW_S", "AUDIO_ANNOUNCE_MIN_GAP_S",
                     "AUDIO_REPICK_STABLE_S"):
            self.assertIsInstance(getattr(config, name), float, name)
        self.assertIs(type(config.AUDIO_FLAP_THRESHOLD), int)
        with open(config.__file__, encoding="utf-8") as fh:
            src = fh.read()
        block = src[src.index("Audio-device flap damping"):
                    src.index("AUDIO_REPICK_STABLE_S    =")]
        self.assertIn("apply on the next start", block)

    def test_smart_turn_knobs_are_typed_and_the_model_lives_outside_the_repo(self):
        # Speed plan R7: the three numeric knobs are float literals (an int
        # default would make _apply_user_settings truncate a saved 0.75 to
        # 0) at the plan's values, and the model path never points into the
        # repo (a model is never committed).
        for name, want in (("SMART_TURN_THRESHOLD", 0.7),
                           ("SMART_TURN_MIN_SILENCE_S", 0.256),
                           ("SMART_TURN_MIN_SPEECH_S", 1.0)):
            self.assertIsInstance(getattr(config, name), float, name)
        with open(config.__file__, encoding="utf-8") as fh:
            src = fh.read()
        block = src[src.index("Smart Turn end of turn (speed plan R7"):
                    src.index("SMART_TURN_MODEL = ")]
        self.assertIn("SMART_TURN_THRESHOLD = 0.7\n", block)
        self.assertIn("SMART_TURN_MIN_SILENCE_S = 0.256\n", block)
        self.assertIn("SMART_TURN_MIN_SPEECH_S = 1.0\n", block)
        self.assertIn("applies on the next start", block)
        model = config.SMART_TURN_MODEL
        self.assertTrue(model.endswith("smart-turn-v3.2-cpu.onnx"), model)
        import ntpath          # a Windows path, also when CI runs on Linux
        self.assertTrue(ntpath.isabs(model), model)
        root = os.path.dirname(os.path.dirname(os.path.abspath(
            config.__file__)))
        self.assertFalse(os.path.normcase(model).startswith(
            os.path.normcase(root)), model)


if __name__ == "__main__":
    unittest.main()
