"""Tests for core.vram_budget — the predictive VRAM budget estimator.

The estimator is the engine behind the Settings GUI's "graphics-settings"
VRAM bar: given the chosen model + feature toggles it predicts the TOTAL GPU
memory those settings will load and flags an over-commit (the 32B-plus-vision
"brick"). These tests pin:

  * the calibrated per-model table (the measured 3090 anchors),
  * the unknown-tag estimate from a MOCKED ``ollama list`` and the param-count
    fallback when ollama is absent,
  * predict_budget's component selection + over/pct math for the canonical
    cases (14B+cloud-vision fits; 32B+local-vision over; RAG adds embeddings;
    vision excluded when routed to the cloud or when screen-vision is off),
  * the text renderers (budget_lines / budget_bar / over_warning), and
  * the GUI value→budget bridge (tools.settings_window.budget_from_live_values)
    — the function the live recompute callback calls, tested WITHOUT any Tk.

No real GPU / nvidia-smi / ollama is touched: subprocess is mocked end-to-end.
stdlib unittest + unittest.mock only. PRIVACY: only fake fixture values.
"""
from __future__ import annotations

import importlib.util
import os
import unittest
from unittest import mock

from core import vram_budget as vb

# 24 GB card in MB — passed explicitly to predict_budget so the math is
# deterministic regardless of the test host's real GPU.
_CARD_MB = 24576

# A representative ``ollama list`` capture (header + the JARVIS model set), used
# to drive the unknown-tag disk-size estimate. Mirrors the real column layout:
#   NAME  ID  SIZE  UNIT  MODIFIED…
_OLLAMA_LIST = (
    "NAME                           ID              SIZE      MODIFIED\n"
    "qwen2.5:32b-instruct-q4_K_M    9f13ba1299af    19 GB     5 days ago\n"
    "llama3.1:8b-instruct-q5_K_M    27fe1b0ab52c    5.7 GB    7 days ago\n"
    "qwen2.5:14b-instruct-q5_K_M    7bb3f324cafc    10 GB     7 days ago\n"
    "qwen2.5vl:7b                   5ced39dfa4ba    6.0 GB    7 days ago\n"
    "nomic-embed-text:latest        0a109f422b47    274 MB    7 days ago\n"
    "mistral-small:24b              abcdef012345    14 GB     2 days ago\n"
)


def _ok(stdout):
    """A fake completed subprocess with a 0 exit and the given stdout."""
    return mock.MagicMock(returncode=0, stdout=stdout)


class _CacheResetBase(unittest.TestCase):
    """Reset the module's brief probe caches before each test so a mocked
    nvidia-smi / ollama result in one test never leaks into the next."""

    def setUp(self):
        vb._TOTAL_CACHE[0] = None
        vb._TOTAL_CACHE[1] = 0.0
        vb._OLLAMA_CACHE[0] = None
        vb._OLLAMA_CACHE[1] = 0.0
        self.addCleanup(self._reset)

    def _reset(self):
        vb._TOTAL_CACHE[0] = None
        vb._TOTAL_CACHE[1] = 0.0
        vb._OLLAMA_CACHE[0] = None
        vb._OLLAMA_CACHE[1] = 0.0


# ──────────────────────────────────────────────────────────────────────────
#  model_vram_estimate
# ──────────────────────────────────────────────────────────────────────────
class ModelEstimateTests(_CacheResetBase):
    def test_calibrated_known_tags(self):
        # The measured 3090 anchors (MB). These are the ground truth the GUI
        # bar is calibrated to.
        cases = {
            "qwen2.5:32b-instruct-q4_K_M": 22 * 1024,
            "qwen2.5:14b-instruct-q5_K_M": 13 * 1024,
            "llama3.1:8b-instruct-q5_K_M": 6 * 1024,
            "qwen2.5vl:7b": int(7.3 * 1024),
            "large-v3-turbo": int(1.5 * 1024),
            "nomic-embed-text": int(0.3 * 1024),
        }
        for tag, expect in cases.items():
            self.assertEqual(vb.model_vram_estimate(tag), expect,
                             msg=f"{tag} calibrated VRAM")

    def test_calibrated_match_is_case_insensitive(self):
        self.assertEqual(
            vb.model_vram_estimate("QWEN2.5:14B-INSTRUCT-Q5_K_M"),
            13 * 1024)

    def test_empty_tag_is_zero(self):
        self.assertEqual(vb.model_vram_estimate(""), 0)
        self.assertEqual(vb.model_vram_estimate(None), 0)

    def test_unknown_tag_estimated_from_ollama_list(self):
        # mistral-small:24b isn't calibrated; it IS in the mocked `ollama list`
        # at 14 GB → estimate = 14GB*1.15 + KV allowance.
        def _fake_run(cmd, timeout=2.0):
            return _OLLAMA_LIST if cmd[:2] == ["ollama", "list"] else None
        with mock.patch.object(vb, "_run", side_effect=_fake_run):
            est = vb.model_vram_estimate("mistral-small:24b")
        expect = int(14 * 1024 * vb._DISK_TO_VRAM_FACTOR) + vb._UNKNOWN_KV_ALLOWANCE_MB
        self.assertEqual(est, expect)

    def test_unknown_tag_param_heuristic_when_ollama_absent(self):
        # ollama missing (every _run returns None) AND tag not calibrated →
        # fall back to the param-count heuristic from the "70b" marker.
        with mock.patch.object(vb, "_run", return_value=None):
            est = vb.model_vram_estimate("llama3.3:70b")
        expect = int(70 * 0.7 * 1024) + vb._UNKNOWN_KV_ALLOWANCE_MB
        self.assertEqual(est, expect)

    def test_unknown_no_size_marker_defaults_midrange(self):
        # A wholly unrecognised tag with no "<n>b" and no ollama entry → the
        # 7B-class default, not a crash or zero.
        with mock.patch.object(vb, "_run", return_value=None):
            est = vb.model_vram_estimate("some-weird-model")
        self.assertEqual(est, int(7 * 0.7 * 1024) + vb._UNKNOWN_KV_ALLOWANCE_MB)
        self.assertGreater(est, 0)

    def test_never_raises_on_subprocess_explosion(self):
        # A surprise from subprocess must degrade to the heuristic, never raise.
        with mock.patch.object(vb.subprocess, "run",
                               side_effect=RuntimeError("boom")):
            try:
                est = vb.model_vram_estimate("foo:13b")
            except Exception as e:  # pragma: no cover - the assert is the point
                self.fail(f"model_vram_estimate raised: {e}")
        self.assertGreater(est, 0)


# ──────────────────────────────────────────────────────────────────────────
#  total_vram_mb
# ──────────────────────────────────────────────────────────────────────────
class TotalVramTests(_CacheResetBase):
    def test_parses_nvidia_smi_total(self):
        with mock.patch.object(vb, "_run", return_value="24576\n"):
            self.assertEqual(vb.total_vram_mb(force=True), 24576)

    def test_strips_stray_mib_unit(self):
        # Some drivers print the unit despite --nounits; we still parse the int.
        with mock.patch.object(vb, "_run", return_value="24576 MiB\n"):
            self.assertEqual(vb.total_vram_mb(force=True), 24576)

    def test_multi_gpu_takes_first_card(self):
        with mock.patch.object(vb, "_run", return_value="24576\n8192\n"):
            self.assertEqual(vb.total_vram_mb(force=True), 24576)

    def test_falls_back_to_default_when_absent(self):
        with mock.patch.object(vb, "_run", return_value=None):
            self.assertEqual(vb.total_vram_mb(force=True), vb.DEFAULT_TOTAL_MB)

    def test_falls_back_on_garbage(self):
        with mock.patch.object(vb, "_run", return_value="N/A\n"):
            self.assertEqual(vb.total_vram_mb(force=True), vb.DEFAULT_TOTAL_MB)

    def test_result_is_cached(self):
        run = mock.MagicMock(return_value="24576\n")
        with mock.patch.object(vb, "_run", run):
            vb.total_vram_mb(force=True)   # populates cache
            vb.total_vram_mb()             # served from cache
            vb.total_vram_mb()
        self.assertEqual(run.call_count, 1)


# ──────────────────────────────────────────────────────────────────────────
#  predict_budget  — the heart of the estimator
# ──────────────────────────────────────────────────────────────────────────
class PredictBudgetTests(_CacheResetBase):
    def test_14b_cloud_vision_whisper_fits(self):
        # 14B (13) + Whisper (1.5), vision routed to the cloud (excluded) →
        # ~14.5 GB, comfortably under the 24 GB card. NOT over.
        s = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
             "MODEL_ROUTING": {"vision": "cloud"},
             "LOCAL_VISION_FALLBACK": False,
             "SCREEN_VISION_ENABLED": True}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        self.assertFalse(b["over"])
        self.assertEqual(b["total_mb"], 13 * 1024 + int(1.5 * 1024))
        labels = [c["label"] for c in b["components"]]
        self.assertNotIn("vision", labels)
        self.assertIn("Whisper", labels)
        # ~14.5 GB / 22.5 usable budget → well under 100%.
        self.assertLess(b["pct"], 80.0)

    def test_32b_local_vision_whisper_is_over_the_brick(self):
        # THE BRICK: 32B (22) + vision (7.3) + Whisper (1.5) ≈ 30.8 GB on a
        # 24 GB card → over budget. This is the over-commit the bar must catch.
        s = {"LOCAL_LLM_MODEL": "qwen2.5:32b-instruct-q4_K_M",
             "MODEL_ROUTING": {"vision": "local"},
             "SCREEN_VISION_ENABLED": True}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        self.assertTrue(b["over"])
        expect = 22 * 1024 + int(7.3 * 1024) + int(1.5 * 1024)
        self.assertEqual(b["total_mb"], expect)
        self.assertGreater(b["pct"], 100.0)
        # ~30.8 GB.
        self.assertAlmostEqual(b["total_mb"] / 1024.0, 30.8, delta=0.2)

    def test_rag_adds_embeddings_component(self):
        base = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
                "MODEL_ROUTING": {"vision": "cloud"},
                "LOCAL_VISION_FALLBACK": False}
        without = vb.predict_budget(base, total_mb=_CARD_MB)
        withrag = vb.predict_budget({**base, "RAG_ENABLED": True},
                                    total_mb=_CARD_MB)
        self.assertNotIn("embeddings", [c["label"] for c in without["components"]])
        self.assertIn("embeddings", [c["label"] for c in withrag["components"]])
        self.assertEqual(withrag["total_mb"] - without["total_mb"],
                         vb.EMBED_VRAM_MB)

    def test_vision_excluded_when_routed_cloud(self):
        s = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
             "MODEL_ROUTING": {"vision": "cloud"},
             "LOCAL_VISION_FALLBACK": False,
             "SCREEN_VISION_ENABLED": True}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        self.assertNotIn("vision", [c["label"] for c in b["components"]])

    def test_vision_included_when_routed_local(self):
        s = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
             "MODEL_ROUTING": {"vision": "local"}}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        comp = [c for c in b["components"] if c["label"] == "vision"]
        self.assertEqual(len(comp), 1)
        self.assertTrue(comp[0]["ondemand"])           # marked on-demand
        self.assertEqual(comp[0]["mb"], vb.VISION_VRAM_MB)

    def test_vision_included_on_auto_route(self):
        s = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
             "MODEL_ROUTING": {"vision": "auto"},
             "LOCAL_VISION_FALLBACK": False}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        self.assertIn("vision", [c["label"] for c in b["components"]])

    def test_vision_shared_with_multimodal_chat_costs_zero(self):
        # LOCAL_VISION_MODEL == LOCAL_LLM_MODEL (a multimodal brain like
        # gemma4:26b-a4b): vision re-uses the resident chat model — the
        # component shows as shared at 0 MB, so the co-load brick can't
        # happen and the bar doesn't double-count 17 GB.
        s = {"LOCAL_LLM_MODEL": "gemma4:26b-a4b-it-qat",
             "LOCAL_VISION_MODEL": "gemma4:26b-a4b-it-qat",
             "MODEL_ROUTING": {"vision": "local"}}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        comp = [c for c in b["components"] if c["label"].startswith("vision")]
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["mb"], 0)
        self.assertFalse(b["over"])   # 16 GB chat + 1.5 whisper fits 24 GB

    def test_vision_model_off_excludes_component(self):
        s = {"LOCAL_LLM_MODEL": "gemma4:26b-a4b-it-qat",
             "LOCAL_VISION_MODEL": "off",
             "MODEL_ROUTING": {"vision": "local"}}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        self.assertNotIn("vision",
                         [c["label"] for c in b["components"]])

    def test_vision_distinct_tag_estimated_not_flat(self):
        # A DIFFERENT vision tag is a real co-load — estimated from the
        # calibration table (qwen2.5vl:7b → the flat legacy allowance).
        s = {"LOCAL_LLM_MODEL": "gemma4:26b-a4b-it-qat",
             "LOCAL_VISION_MODEL": "qwen2.5vl:7b",
             "MODEL_ROUTING": {"vision": "local"}}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        comp = [c for c in b["components"] if c["label"] == "vision"]
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["mb"], vb.CALIBRATED_VRAM_MB["qwen2.5vl:7b"])

    def test_vision_included_when_fallback_on_even_if_cloud(self):
        # LOCAL_VISION_FALLBACK pulls the VLM in even when the route is cloud
        # (the cloud call can fail over to local → it can load).
        s = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
             "MODEL_ROUTING": {"vision": "cloud"},
             "LOCAL_VISION_FALLBACK": True,
             "SCREEN_VISION_ENABLED": True}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        self.assertIn("vision", [c["label"] for c in b["components"]])

    def test_vision_excluded_when_screen_vision_off(self):
        # Master screen-vision switch off → no vision VRAM even if routed local.
        s = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
             "MODEL_ROUTING": {"vision": "local"},
             "LOCAL_VISION_FALLBACK": True,
             "SCREEN_VISION_ENABLED": False}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        self.assertNotIn("vision", [c["label"] for c in b["components"]])

    def test_flattened_routing_key_is_honoured(self):
        # The GUI's Tk routing var is the flattened "MODEL_ROUTING::vision".
        s = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
             "MODEL_ROUTING::vision": "local",
             "SCREEN_VISION_ENABLED": True}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        self.assertIn("vision", [c["label"] for c in b["components"]])

    def test_string_bools_coerced(self):
        # Tk StringVars deliver "1"/"0"; the engine must read them as bools.
        s = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
             "MODEL_ROUTING::vision": "cloud",
             "LOCAL_VISION_FALLBACK": "0",
             "SCREEN_VISION_ENABLED": "1"}
        b = vb.predict_budget(s, total_mb=_CARD_MB)
        self.assertNotIn("vision", [c["label"] for c in b["components"]])

    def test_chat_model_always_counted(self):
        b = vb.predict_budget({"LOCAL_LLM_MODEL": "llama3.1:8b-instruct-q5_K_M"},
                              total_mb=_CARD_MB)
        self.assertFalse(b["components"][0]["ondemand"])
        self.assertEqual(b["components"][0]["mb"], 6 * 1024)

    def test_default_chat_model_when_unset(self):
        # No LOCAL_LLM_MODEL → falls back to config's default, now
        # gemma4:26b-a4b-it-qat (~16 GB calibrated; 2026-07-15 P2 phase B, after
        # TTS moved off the GPU freed ~13 GB).
        b = vb.predict_budget({"MODEL_ROUTING": {"vision": "cloud"},
                               "LOCAL_VISION_FALLBACK": False},
                              total_mb=_CARD_MB)
        self.assertEqual(b["components"][0]["mb"], 16 * 1024)

    def test_budget_is_card_minus_headroom(self):
        b = vb.predict_budget({"LOCAL_LLM_MODEL": "llama3.1:8b-instruct-q5_K_M",
                               "MODEL_ROUTING": {"vision": "cloud"},
                               "LOCAL_VISION_FALLBACK": False},
                              total_mb=_CARD_MB)
        self.assertEqual(b["budget_mb"], _CARD_MB - vb.HEADROOM_MB)
        self.assertEqual(b["total_card_mb"], _CARD_MB)
        self.assertEqual(b["headroom_mb"], vb.HEADROOM_MB)

    def test_pct_and_over_math(self):
        b = vb.predict_budget({"LOCAL_LLM_MODEL": "qwen2.5:32b-instruct-q4_K_M",
                               "MODEL_ROUTING": {"vision": "local"}},
                              total_mb=_CARD_MB)
        expect_pct = b["total_mb"] / b["budget_mb"] * 100.0
        self.assertAlmostEqual(b["pct"], expect_pct, places=4)
        self.assertEqual(b["over"], b["total_mb"] > b["budget_mb"])

    def test_uses_probed_total_when_not_passed(self):
        # With no explicit total_mb, predict_budget asks total_vram_mb(), which
        # we mock at the 24 GB default.
        with mock.patch.object(vb, "total_vram_mb", return_value=_CARD_MB):
            b = vb.predict_budget({"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
                                   "MODEL_ROUTING": {"vision": "cloud"},
                                   "LOCAL_VISION_FALLBACK": False})
        self.assertEqual(b["total_card_mb"], _CARD_MB)

    def test_kinect_does_not_change_total(self):
        # KINECT_ENABLED is informational (CPU/USB, no GPU model) — it must not
        # move the VRAM total.
        base = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
                "MODEL_ROUTING": {"vision": "cloud"},
                "LOCAL_VISION_FALLBACK": False}
        off = vb.predict_budget(base, total_mb=_CARD_MB)
        on = vb.predict_budget({**base, "KINECT_ENABLED": True}, total_mb=_CARD_MB)
        self.assertEqual(off["total_mb"], on["total_mb"])

    def test_never_raises_on_malformed_settings(self):
        for bad in (None, "not a dict", 123, [], {"LOCAL_LLM_MODEL": None}):
            try:
                b = vb.predict_budget(bad, total_mb=_CARD_MB)
            except Exception as e:  # pragma: no cover
                self.fail(f"predict_budget raised on {bad!r}: {e}")
            self.assertIn("total_mb", b)
            self.assertGreaterEqual(b["total_mb"], 0)


# ──────────────────────────────────────────────────────────────────────────
#  Text renderers
# ──────────────────────────────────────────────────────────────────────────
class RenderTests(_CacheResetBase):
    BRICK = {"LOCAL_LLM_MODEL": "qwen2.5:32b-instruct-q4_K_M",
             "MODEL_ROUTING": {"vision": "local"},
             "SCREEN_VISION_ENABLED": True}
    FITS = {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
            "MODEL_ROUTING": {"vision": "cloud"},
            "LOCAL_VISION_FALLBACK": False}

    def test_budget_lines_include_breakdown_and_warning(self):
        lines = vb.budget_lines(self.BRICK, total_mb=_CARD_MB)
        joined = "\n".join(lines)
        self.assertIn("VRAM budget", joined)
        self.assertIn("32B", joined)
        self.assertIn("vision", joined)
        self.assertIn("on-demand", joined)
        self.assertIn("Whisper", joined)
        # The over-budget warning is present for the brick.
        self.assertIn("⚠", joined)   # ⚠

    def test_budget_lines_no_warning_when_fits(self):
        joined = "\n".join(vb.budget_lines(self.FITS, total_mb=_CARD_MB))
        self.assertNotIn("⚠", joined)

    def test_over_warning_names_overage_and_fixes(self):
        b = vb.predict_budget(self.BRICK, total_mb=_CARD_MB)
        warn = vb.over_warning(b)
        self.assertIn("⚠", warn)
        self.assertIn("24", warn)        # the card size
        self.assertIn("14B", warn)       # suggested smaller model
        self.assertIn("cloud", warn)     # route vision to the cloud

    def test_budget_bar_marks_over_with_bang(self):
        bar = vb.budget_bar(self.BRICK, width=20, total_mb=_CARD_MB)
        self.assertTrue(bar.startswith("["))
        self.assertTrue(bar.endswith("!"))   # over → trailing '!'
        # Over budget → the bar is fully filled (clamped).
        self.assertIn("#" * 20, bar)

    def test_budget_bar_partial_when_fits(self):
        bar = vb.budget_bar(self.FITS, width=20, total_mb=_CARD_MB)
        self.assertFalse(bar.endswith("!"))
        self.assertIn("#", bar)
        self.assertIn("-", bar)              # some free space remains

    def test_budget_bar_width_floor(self):
        # A nonsense width doesn't crash or produce a negative-length bar.
        bar = vb.budget_bar(self.FITS, width=0, total_mb=_CARD_MB)
        self.assertTrue(bar.startswith("[") and "]" in bar)

    def test_fmt_gb_rounding(self):
        self.assertEqual(vb._fmt_gb(13 * 1024), "13")
        self.assertEqual(vb._fmt_gb(int(7.3 * 1024)), "7.3")

    def test_short_model_label(self):
        self.assertEqual(vb._short_model_label("qwen2.5:32b-instruct-q4_K_M"), "32B")
        self.assertEqual(vb._short_model_label("llama3.1:8b-instruct-q5_K_M"), "8B")
        # No "<n>b" marker → a trimmed base name, not a crash.
        self.assertTrue(vb._short_model_label("nomic-embed-text"))


# ──────────────────────────────────────────────────────────────────────────
#  GUI bridge  (tools.settings_window.budget_from_live_values) — the live
#  recompute callback's value→budget function, tested without any Tk.
# ──────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_SW_PATH = os.path.join(_PROJECT, "tools", "settings_window.py")
_sw_spec = importlib.util.spec_from_file_location("jarvis_settings_window_vram",
                                                  _SW_PATH)
sw = importlib.util.module_from_spec(_sw_spec)
_sw_spec.loader.exec_module(sw)


class GuiBridgeTests(_CacheResetBase):
    def test_watch_keys_cover_the_budget_inputs(self):
        # The keys the GUI traces for live recompute must include every input
        # predict_budget actually reads.
        for k in ("LOCAL_LLM_MODEL", "MODEL_ROUTING::vision",
                  "LOCAL_VISION_FALLBACK", "SCREEN_VISION_ENABLED",
                  "RAG_ENABLED", "KINECT_ENABLED"):
            self.assertIn(k, sw.VRAM_WATCH_KEYS)

    def test_schema_has_rag_and_kinect_toggles(self):
        # The GUI edits these (added for the budget) — they must be persisted
        # bools on the AI tab.
        for k in ("RAG_ENABLED", "KINECT_ENABLED"):
            self.assertIn(k, sw.SCHEMA)
            self.assertEqual(sw.SCHEMA[k]["type"], "bool")
            self.assertIn(k, sw.persisted_keys())

    def test_bridge_brick_is_over(self):
        b = sw.budget_from_live_values(
            {"LOCAL_LLM_MODEL": "qwen2.5:32b-instruct-q4_K_M",
             "MODEL_ROUTING::vision": "local",
             "SCREEN_VISION_ENABLED": True},
            total_mb=_CARD_MB)
        self.assertIsNotNone(b)
        self.assertTrue(b["over"])

    def test_bridge_14b_cloud_fits(self):
        b = sw.budget_from_live_values(
            {"LOCAL_LLM_MODEL": "qwen2.5:14b-instruct-q5_K_M",
             "MODEL_ROUTING::vision": "cloud",
             "LOCAL_VISION_FALLBACK": "0",
             "SCREEN_VISION_ENABLED": "1"},
            total_mb=_CARD_MB)
        self.assertIsNotNone(b)
        self.assertFalse(b["over"])

    def test_bridge_handles_garbage(self):
        # Non-dict input must not raise — returns a valid budget (or None).
        b = sw.budget_from_live_values("nope", total_mb=_CARD_MB)
        self.assertTrue(b is None or "total_mb" in b)

    def test_bridge_returns_none_when_engine_missing(self):
        # If the engine can't import, the bridge returns None (panel hides) —
        # never an exception.
        with mock.patch.object(sw, "_load_vram_budget", return_value=None):
            self.assertIsNone(sw.budget_from_live_values({"LOCAL_LLM_MODEL": "x"}))


if __name__ == "__main__":
    unittest.main()
