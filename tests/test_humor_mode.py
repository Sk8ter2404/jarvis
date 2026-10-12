"""HUMOR_MODE ("dry" | "sarcastic"): the sarcasm block and the [wry] cadence
swap in core.prompts.base_system_prompt(), and the setting's schema row."""
import json
import os
import unittest
from unittest import mock

import core.config as cfg
from core import prompts
from tools import settings_window as sw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRY_OLD = ("Use it sparingly \u2014 once or twice a session at most; if "
           "every line is wry the cadence stops being funny.")
WRY_NEW = ("Use it on roughly one casual reply in four; if every line is wry "
           "the cadence stops being funny.")


def _build(mode, night_quiet=True):
    with mock.patch.object(cfg, "HUMOR_MODE", mode, create=True):
        return prompts.base_system_prompt(night_quiet=night_quiet)


class HumorPromptTests(unittest.TestCase):
    def test_default_is_dry(self):
        self.assertEqual(cfg.HUMOR_MODE, "dry")

    def test_dry_is_byte_identical(self):
        self.assertEqual(_build("dry"), prompts.BASE_SYSTEM_PROMPT)
        self.assertEqual(_build("nonsense"), prompts.BASE_SYSTEM_PROMPT)
        self.assertEqual(_build("dry", False),
                         _build("dry", False).replace(
                             prompts.SARCASM_BLOCK, ""))
        self.assertNotIn(prompts.SARCASM_BLOCK, _build("dry", False))

    def test_sarcastic_inserts_block_once_before_memory_use(self):
        p = _build("sarcastic")
        self.assertEqual(p.count(prompts.SARCASM_BLOCK), 1)
        self.assertLess(p.index(prompts.SARCASM_BLOCK),
                        p.index("Memory use:"))
        self.assertLess(p.index("Few-shot examples of the dry-wit register"),
                        p.index(prompts.SARCASM_BLOCK))
        self.assertNotIn("\u2014", prompts.SARCASM_BLOCK)

    def test_sarcastic_swaps_wry_sentence(self):
        self.assertIn(WRY_OLD, prompts.BASE_SYSTEM_PROMPT)
        p = _build("sarcastic")
        self.assertIn(WRY_NEW, p)
        self.assertNotIn(WRY_OLD, p)

    def test_sarcastic_also_works_with_night_quiet_off(self):
        p = _build("sarcastic", night_quiet=False)
        self.assertEqual(p.count(prompts.SARCASM_BLOCK), 1)
        self.assertIn(WRY_NEW, p)

    def test_stable_within_a_session(self):
        self.assertEqual(_build("sarcastic"), _build("sarcastic"))
        self.assertIsInstance(prompts.SARCASM_BLOCK, str)

    def test_block_text(self):
        b = prompts.SARCASM_BLOCK
        self.assertTrue(b.startswith("SARCASM AND SHADE (sir asked for it:"))
        self.assertTrue(b.endswith("If in doubt, skip the jab.\n"))
        self.assertIn("  'Done, sir. I live to serve, apparently.'\n", b)


class HumorSettingTests(unittest.TestCase):
    def test_schema_row(self):
        row = sw.SCHEMA["HUMOR_MODE"]
        self.assertEqual(row["type"], "enum")
        self.assertEqual(row["choices"], ["dry", "sarcastic"])
        self.assertEqual(row["default"], cfg.HUMOR_MODE)
        self.assertIn("HUMOR_MODE", [k for _h, rows in
                                     sw.TAB_SECTIONS["voice"] for k in rows])

    def test_example_json(self):
        with open(os.path.join(ROOT, "tools", "user_settings.example.json"),
                  encoding="utf-8") as f:
            self.assertEqual(json.load(f)["HUMOR_MODE"], "dry")


if __name__ == "__main__":
    unittest.main()
