"""Tests for core.model_catalog — model pricing + per-conversation cost estimate.

CI-safe: stdlib only, env mocked.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import core.model_catalog as mc


def _no_conv_env():
    ctx = mock.patch.dict(os.environ, {}, clear=False)

    class _C:
        def __enter__(self):
            ctx.start()
            os.environ.pop("JARVIS_CONV_INPUT_TOKENS", None)
            os.environ.pop("JARVIS_CONV_OUTPUT_TOKENS", None)
            return self

        def __exit__(self, *a):
            ctx.stop()
            return False
    return _C()


class ConvTokensTests(unittest.TestCase):
    def test_defaults(self):
        with _no_conv_env():
            self.assertEqual(mc._conv_tokens(), (12000, 1500))

    def test_env_override(self):
        with mock.patch.dict(os.environ, {"JARVIS_CONV_INPUT_TOKENS": "5000",
                                          "JARVIS_CONV_OUTPUT_TOKENS": "800"}, clear=False):
            self.assertEqual(mc._conv_tokens(), (5000, 800))

    def test_blank_env_uses_default(self):
        with _no_conv_env():
            with mock.patch.dict(os.environ, {"JARVIS_CONV_INPUT_TOKENS": "  "}, clear=False):
                self.assertEqual(mc._conv_tokens()[0], 12000)

    def test_bad_env_uses_default(self):
        with mock.patch.dict(os.environ, {"JARVIS_CONV_INPUT_TOKENS": "abc"}, clear=False):
            self.assertEqual(mc._conv_tokens()[0], 12000)

    def test_negative_clamped_to_zero(self):
        with mock.patch.dict(os.environ, {"JARVIS_CONV_INPUT_TOKENS": "-100"}, clear=False):
            self.assertEqual(mc._conv_tokens()[0], 0)


class CostTests(unittest.TestCase):
    def test_local_is_free(self):
        self.assertEqual(mc.by_id("qwen2.5:14b-instruct").cost_per_conversation(), 0.0)

    def test_cloud_cost_with_defaults(self):
        with _no_conv_env():
            c = mc.by_id("claude-sonnet-4-6").cost_per_conversation()
        # 12000/1e6*3 + 1500/1e6*15 = 0.036 + 0.0225
        self.assertAlmostEqual(c, 0.0585, places=4)

    def test_explicit_tokens(self):
        c = mc.by_id("claude-haiku-4-5").cost_per_conversation(in_tokens=1_000_000,
                                                               out_tokens=0)
        self.assertAlmostEqual(c, 1.0)


class CatalogTests(unittest.TestCase):
    def test_catalog_is_a_copy(self):
        c = mc.catalog()
        c.clear()
        self.assertTrue(mc.catalog())

    def test_by_id_exact(self):
        self.assertEqual(mc.by_id("claude-opus-4-6").label, "Claude Opus")
        self.assertEqual(mc.by_id("claude-sonnet-5").label, "Claude Sonnet 5")
        self.assertEqual(mc.by_id("claude-opus-4-8").label, "Claude Opus 4.8")

    def test_by_id_prefix(self):
        self.assertEqual(mc.by_id("qwen2.5:14b-instruct-q5_K_M").backend, "ollama")

    def test_by_id_missing(self):
        self.assertIsNone(mc.by_id("gpt-9"))

    def test_by_id_empty(self):
        self.assertIsNone(mc.by_id(""))


class FormatTests(unittest.TestCase):
    def test_fmt_usd_buckets(self):
        self.assertIn("local", mc._fmt_usd(0))
        self.assertIn("0.005", mc._fmt_usd(0.005))
        self.assertIn("0.29", mc._fmt_usd(0.29))

    def test_format_catalog_lists_all(self):
        with _no_conv_env():
            s = mc.format_catalog()
        for label in ("Qwen", "Claude Haiku", "Claude Sonnet", "Claude Opus"):
            self.assertIn(label, s)
        self.assertIn(mc.PRICING_AS_OF, s)
        self.assertIn("cheapest first", s)


class ModelActionTests(unittest.TestCase):
    """The core.actions surfaces (light tier; they delegate to model_catalog)."""

    def test_model_costs_action(self):
        import core.actions as A
        out = A._act_model_costs()
        self.assertIn("Claude Sonnet", out)
        self.assertIn("per conversation", out)

    def test_switch_llm_picker_lists_options(self):
        import core.actions as A
        out = A._act_switch_llm_picker()
        self.assertIn("Claude Haiku", out)
        self.assertIn("switch to", out)

    # These exercise the CATALOG (does it price the model right?), using
    # show_llm_stats as the vehicle. They used to patch core.config — but as of
    # the 2026-07-14 audit the action reads the LIVE backend off the monolith,
    # with core.config only as a no-monolith fallback. Patching config here
    # would therefore pass in the light CI tier (no monolith importable → the
    # fallback runs) and FAIL on a dev box (monolith imports → live values win):
    # green-by-environment. Patch the resolver these tests actually depend on.
    def test_show_llm_stats_known_cloud_model(self):
        import core.actions as A
        with mock.patch.object(A, "_live_backend_and_model",
                               return_value=("claude", "claude-sonnet-4-6")):
            out = A._act_show_llm_stats()
        self.assertIn("est.", out)
        self.assertIn("claude-sonnet-4-6", out)
        self.assertIn("/conv", out)

    def test_show_llm_stats_local_is_free(self):
        import core.actions as A
        with mock.patch.object(A, "_live_backend_and_model",
                               return_value=("ollama", "qwen2.5:14b-instruct")):
            out = A._act_show_llm_stats()
        self.assertIn("$0 (local)", out)

    def test_show_llm_stats_unknown_model(self):
        import core.actions as A
        with mock.patch.object(A, "_live_backend_and_model",
                               return_value=("claude", "some-unlisted-model")):
            out = A._act_show_llm_stats()
        self.assertIn("not in the cost catalog", out)


if __name__ == "__main__":
    unittest.main()
