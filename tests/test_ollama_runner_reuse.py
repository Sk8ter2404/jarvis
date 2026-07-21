"""Regression guard: every Ollama GENERATION call must pin num_ctx.

THE INCIDENT (2026-07-21, live)
===============================
Ollama keys a loaded runner by (model, options). Chat pinned num_ctx=16384;
`ask_vision` sent only num_predict. Because the shipped design points chat AND
vision at the SAME multimodal tag (gemma4:26b-a4b-it-qat), the vision call did
not reuse the warm runner — it evicted it and reloaded the same weights at the
model's own 262144 default. Ollama's server log::

    llama_context: n_ctx = 262144
    srv load_model: initializing, n_slots = 1, n_ctx_slot = 262144

`ollama ps` then read `16 GB  6%/94% CPU/GPU  CONTEXT 262144` with the 3090 at
24147/24576 MiB, and the next voice turn died on the 50 s read timeout:
"My local model isn't responding and I can't reach the cloud either, sir."
The ambient-extract daemon fires a vision call every 300 s, so this bricked the
primary brain on a five-minute cycle.

`core/orchestrator.py:_ollama_call` had the same defect (no options at all) —
the stale-duplicate class: one rule, three call sites, fixed in one.

These tests fail if any of that regresses. They need no GPU, no Ollama, and no
monolith import, so they run under tools/run_tests_ci_sim.py.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ollama_opts                      # noqa: E402
from core.ollama_opts import chat_options, local_num_ctx  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LocalNumCtxTests(unittest.TestCase):
    """The heuristic itself — behaviour preserved from the monolith original."""

    def test_big_dense_tags_get_the_tight_window(self):
        for tag in ("qwen2.5:32b", "llama3.3:70b", "something:72b-instruct",
                    "x:34b", "y:65b", "z:30b"):
            self.assertEqual(local_num_ctx(tag), ollama_opts.BIG_MODEL_NUM_CTX,
                             f"{tag} should get the tight window")

    def test_moe_active_param_suffix_is_not_mistaken_for_size(self):
        # `qwen3:30b-a3b` must parse as 30 (tight), NOT 3 (wide) — the `a3b`
        # active-param suffix is excluded by the lookbehind.
        self.assertEqual(local_num_ctx("qwen3:30b-a3b-instruct-2507-q4_K_M"),
                         ollama_opts.BIG_MODEL_NUM_CTX)

    def test_production_26b_moe_keeps_the_wide_window(self):
        # The live production tag. 26 < 30 → the 16k window, which is what the
        # boot warm-up loads and therefore what every other caller must match.
        self.assertEqual(local_num_ctx("gemma4:26b-a4b-it-qat"),
                         ollama_opts.DEFAULT_NUM_CTX)

    def test_small_models_keep_the_wide_window(self):
        for tag in ("qwen2.5:14b-instruct-q5_K_M", "llama3.1:8b", "gemma4:12b"):
            self.assertEqual(local_num_ctx(tag), ollama_opts.DEFAULT_NUM_CTX)

    def test_empty_and_none_are_safe(self):
        self.assertEqual(local_num_ctx(""), ollama_opts.DEFAULT_NUM_CTX)
        self.assertEqual(local_num_ctx(None), ollama_opts.DEFAULT_NUM_CTX)  # type: ignore[arg-type]


class ChatOptionsTests(unittest.TestCase):
    def test_num_ctx_always_present(self):
        self.assertIn("num_ctx", chat_options("gemma4:26b-a4b-it-qat"))

    def test_optional_knobs_only_present_when_asked(self):
        opts = chat_options("gemma4:12b")
        self.assertNotIn("num_predict", opts)
        self.assertNotIn("temperature", opts)
        opts = chat_options("gemma4:12b", num_predict=200, temperature=0.4)
        self.assertEqual(opts["num_predict"], 200)
        self.assertEqual(opts["temperature"], 0.4)

    def test_extra_wins_last(self):
        opts = chat_options("gemma4:12b", extra={"num_ctx": 999})
        self.assertEqual(opts["num_ctx"], 999)

    def test_two_callers_for_one_model_agree(self):
        """The actual invariant: same model → identical num_ctx, so the runner
        is REUSED rather than evicted and reloaded."""
        chat = chat_options("gemma4:26b-a4b-it-qat", num_predict=512,
                            temperature=0.4)
        vision = chat_options("gemma4:26b-a4b-it-qat", num_predict=300)
        self.assertEqual(chat["num_ctx"], vision["num_ctx"])


class OrchestratorSendsNumCtxTests(unittest.TestCase):
    """core/orchestrator.py:_ollama_call used to send NO options at all."""

    def test_ollama_call_pins_num_ctx(self):
        from core import orchestrator as orch

        captured = {}

        class _Resp:
            def read(self):
                return json.dumps({"message": {"content": "ok"}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _Resp()

        import urllib.request
        with mock.patch.object(urllib.request, "urlopen", _urlopen):
            out = orch._ollama_call("qwen3:30b-a3b", "sys", "usr")

        self.assertEqual(out, "ok")
        opts = captured["payload"].get("options")
        self.assertIsInstance(opts, dict,
                              "_ollama_call must send an options dict")
        self.assertEqual(opts.get("num_ctx"), local_num_ctx("qwen3:30b-a3b"))


class NoUnpinnedGenerationCallsTests(unittest.TestCase):
    """Tree-wide invariant — the guard that actually stops this coming back.

    Any production POST to Ollama's /api/chat or /api/generate must carry a
    num_ctx. Scans source text rather than importing the monolith, so it is
    cheap and CI-safe.
    """

    # Files that legitimately POST to /api/chat without JARVIS's runner
    # discipline: standalone benchmarking scratch tools, not production paths.
    _EXEMPT = {
        os.path.join("tools", "test_local_prompt.py"),   # pins 16384 inline
    }

    def _production_sources(self):
        for base, dirs, files in os.walk(_PROJECT_ROOT):
            # `.claude` is excluded because it can hold detached git WORKTREES
            # (.claude/worktrees/*) — full snapshots of older revisions. Those
            # are not production source, and scanning them makes this guard
            # report defects that were already fixed on the live tree.
            dirs[:] = [d for d in dirs
                       if d not in ("tests", "__pycache__", ".git", ".claude",
                                    "backups", "_backups", "dist", "models",
                                    "logs", "logs_staging", "data_staging",
                                    "node_modules")]
            for fn in files:
                if fn.endswith(".py"):
                    yield os.path.join(base, fn)

    def test_every_generation_post_pins_num_ctx(self):
        offenders = []
        for path in self._production_sources():
            rel = os.path.relpath(path, _PROJECT_ROOT)
            if rel in self._EXEMPT:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
            except OSError:
                continue
            for m in re.finditer(r"/api/(chat|generate)\b", src):
                # A generation POST builds its payload nearby. Look at a window
                # around the call for the num_ctx pin (directly, or via the
                # canonical builders that always include it).
                lo = max(0, m.start() - 4000)
                window = src[lo:m.end() + 1500]
                if ("num_ctx" in window or "chat_options(" in window
                        or "_local_num_ctx" in window):
                    continue
                # GET probes (/api/tags, /api/ps) never reach here; a bare
                # mention in a comment might, so require an actual post/request.
                if not re.search(r"(requests\.post|urlopen|Request\()", window):
                    continue
                line = src[:m.start()].count("\n") + 1
                offenders.append(f"{rel}:{line}")
        self.assertEqual(
            offenders, [],
            "Ollama generation call(s) without a pinned num_ctx — these EVICT "
            "and reload the warm runner at the model's default context "
            "(262144 for gemma4:26b-a4b), spilling the brain to CPU. Use "
            "core.ollama_opts.chat_options(). Offenders: " + ", ".join(offenders))


if __name__ == "__main__":
    unittest.main()
