"""Logic tests for skills/gpu_usage.py — the voice action + HUD feed.

The engine (core.gpu_usage) is mocked: these tests cover the SKILL's job —
turning a snapshot into a natural spoken readout, the local-vs-cloud routing
sentence, the spoken model-name shortener, and the HUD publish path (which
calls write_hud_state with gpu_lines / gpu_bar). No GPU, no nvidia-smi, no
Ollama, no threads (the harness neuters Thread.start).

stdlib unittest + unittest.mock only.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated, make_fake_skill_utils


_SNAP = {
    "total_mb": 24576, "used_mb": 20600, "free_mb": 3976,
    "util_pct": 29, "temp_c": 41,
    "models": [
        {"name": "qwen2.5:14b-instruct-q5_K_M", "vram_mb": 13312,
         "size_mb": 13312, "processor": "100% GPU"},
        {"name": "qwen2.5vl:7b", "vram_mb": 7168, "size_mb": 7168,
         "processor": "100% GPU"},
        {"name": "nomic-embed-text:latest", "vram_mb": 323, "size_mb": 323,
         "processor": "100% GPU"},
    ],
    "routing": {"chat": "cloud", "vision": "cloud", "ambient": "local"},
    "ts": 1.0,
}


class _GpuSkillBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("gpu_usage")


# ─── action registration ────────────────────────────────────────────────────

class RegistrationTests(_GpuSkillBase):
    def test_actions_registered(self):
        for name in ("gpu_usage", "vram_status", "show_vram", "gpu_status",
                     "whats_loaded"):
            self.assertIn(name, self.actions)
        # All aliases point at the same callable.
        self.assertIs(self.actions["gpu_usage"], self.actions["vram_status"])


# ─── spoken readout (_build_spoken) ────────────────────────────────────────

class SpokenReadoutTests(_GpuSkillBase):
    def test_build_spoken_full(self):
        out = self.mod._build_spoken(_SNAP)
        # First model named + its VRAM against the total.
        self.assertIn("qwen 14B is using 13 of 24 gigs", out)
        # Vision model and embedder by their friendly names.
        self.assertIn("the vision model 7", out)
        self.assertIn("the embedder", out)
        # Total + the routing clause.
        self.assertIn("total 20.1 of 24", out)
        self.assertIn("Chat and vision are on the cloud, ambient on local.", out)
        self.assertTrue(out.rstrip().endswith("."))

    def test_build_spoken_no_models(self):
        snap = dict(_SNAP, models=[])
        out = self.mod._build_spoken(snap)
        self.assertIn("Nothing is resident on the GPU", out)

    def test_build_spoken_without_total(self):
        # No nvidia-smi total → speak just the gig figures, sum for total.
        snap = {"models": [{"name": "qwen2.5:14b", "vram_mb": 13312,
                            "size_mb": 13312, "processor": "100% GPU"}],
                "routing": {"chat": "auto", "vision": "auto", "ambient": "auto"}}
        out = self.mod._build_spoken(snap)
        self.assertIn("qwen 14B is using 13 gigs", out)
        self.assertIn("total 13.0 gigs", out)

    def test_build_spoken_includes_temp(self):
        self.assertIn("GPU at 41 degrees", self.mod._build_spoken(_SNAP))


# ─── routing sentence (_routing_sentence) ──────────────────────────────────

class RoutingSentenceTests(_GpuSkillBase):
    def test_all_cloud_one_local(self):
        s = self.mod._routing_sentence(
            {"chat": "cloud", "vision": "cloud", "ambient": "local"})
        self.assertEqual(s, "Chat and vision are on the cloud, ambient on local.")

    def test_all_local(self):
        s = self.mod._routing_sentence(
            {"chat": "local", "vision": "local", "ambient": "local"})
        self.assertIn("on local", s)
        self.assertTrue(s.endswith("."))

    def test_mixed_three_buckets(self):
        s = self.mod._routing_sentence(
            {"chat": "cloud", "vision": "local", "ambient": "auto"})
        self.assertIn("cloud", s)
        self.assertIn("local", s)
        self.assertIn("auto", s)

    def test_empty_routing(self):
        self.assertEqual(self.mod._routing_sentence({}), "")


# ─── spoken model-name shortener (_speak_model_name) ───────────────────────

class ModelNameTests(_GpuSkillBase):
    def test_14b(self):
        self.assertEqual(
            self.mod._speak_model_name("qwen2.5:14b-instruct-q5_K_M"),
            "qwen 14B")

    def test_32b(self):
        self.assertEqual(
            self.mod._speak_model_name("qwen2.5:32b-instruct-q4_K_M"),
            "qwen 32B")

    def test_vision(self):
        self.assertEqual(self.mod._speak_model_name("qwen2.5vl:7b"),
                         "the vision model")

    def test_embedder(self):
        self.assertEqual(self.mod._speak_model_name("nomic-embed-text:latest"),
                         "the embedder")

    def test_llama_base(self):
        self.assertEqual(
            self.mod._speak_model_name("llama3.1:8b-instruct-q5_K_M"),
            "llama 8B")

    def test_speak_size_under_a_gig(self):
        self.assertEqual(self.mod._speak_size(0.3), "under a gig")
        self.assertEqual(self.mod._speak_size(13.0), "13")
        self.assertEqual(self.mod._speak_size(7.3), "7.3")


# ─── voice action returns the formatted readout (engine mocked) ────────────

class ActionTests(_GpuSkillBase):
    def test_action_returns_spoken(self):
        with mock.patch.object(self.mod.gpu_usage, "gpu_snapshot",
                               return_value=_SNAP):
            out = self.actions["gpu_usage"]("")
        self.assertIn("qwen 14B is using 13 of 24 gigs", out)
        self.assertIn("ambient on local", out)

    def test_action_graceful_on_engine_error(self):
        with mock.patch.object(self.mod.gpu_usage, "gpu_snapshot",
                               side_effect=RuntimeError("boom")):
            out = self.actions["gpu_usage"]("")
        self.assertIn("failed", out.lower())
        # Must not raise — a string is always returned.
        self.assertIsInstance(out, str)


# ─── HUD publish path (_publish_hud → write_hud_state) ─────────────────────

class HudPublishTests(unittest.TestCase):
    def test_publish_calls_write_hud_state_with_gpu_fields(self):
        captured = {}

        def _writer(**kw):
            captured.update(kw)

        utils = make_fake_skill_utils(write_hud_state=_writer)
        mod, _ = load_skill_isolated("gpu_usage", utils=utils)
        mod._publish_hud(["qwen 14B  13/24 GB", "TOTAL  20.6/24 GB (84%)"],
                         "[####------] 84%")
        self.assertEqual(captured["gpu_lines"][0], "qwen 14B  13/24 GB")
        self.assertEqual(captured["gpu_bar"], "[####------] 84%")
        self.assertIn("gpu_updated_at", captured)

    def test_publish_silent_when_no_writer(self):
        # No write_hud_state available → no exception, just a no-op.
        utils = make_fake_skill_utils()
        utils.pop("write_hud_state", None)
        mod, _ = load_skill_isolated("gpu_usage", utils=utils)
        try:
            mod._publish_hud(["x"], "[--] 0%")   # must not raise
        except Exception as e:   # pragma: no cover - the assert is the point
            self.fail(f"_publish_hud raised without a writer: {e}")


if __name__ == "__main__":
    unittest.main()
