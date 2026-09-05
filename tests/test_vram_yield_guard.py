"""Tests for tools/vram_yield_guard.py — the unload-only VRAM latch.

The guard exists because of a measured fact (2026-09-04): the owner shut JARVIS
down at 20:47:08 and `llama-server` was STILL holding 15 GB of the 3090 eight
minutes later, because `keep_alive` is Ollama's state, not JARVIS's. Nothing
inside JARVIS can hold that memory down.

These tests pin the three properties the design argument rests on:

  1. It only ever UNLOADS. There is no restore path, so there is no state it can
     strand if it dies mid-tick. A test asserts the module exposes no such path,
     because the safety argument evaporates the moment someone adds one.
  2. It cannot fight the owner. While the inhibit file is fresh, it unloads
     nothing — it must never drop a brain he just asked for, possibly
     mid-generation.
  3. It is a LATCH, not a one-shot. It re-unloads every tick, which is what
     makes it correct without a request chokepoint — there are five distinct
     Ollama request sites in this tree, and an earlier design that gated two of
     them called it coverage.

Nothing here touches a real game, a real Ollama, or real VRAM.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    """Import tools/vram_yield_guard.py by path, fresh each time.

    By path rather than `from tools import ...` so the module's own
    INHIBIT_FILE constant resolves against the real tree, and so a test that
    patches module globals cannot leak into the next test."""
    path = os.path.join(_ROOT, "tools", "vram_yield_guard.py")
    spec = importlib.util.spec_from_file_location("_vram_yield_guard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Base(unittest.TestCase):
    def setUp(self):
        self.g = _load()


class UnloadOnlyTests(_Base):
    """Property 1 — the safety argument."""

    def test_module_exposes_no_restore_path(self):
        # If someone adds a reload/restore, the "no state it can strand" claim
        # is no longer true and this test should force them to re-argue it.
        banned = [n for n in dir(self.g)
                  if any(w in n.lower() for w in
                         ("restore", "reload", "rewarm", "warm_up", "resume"))]
        self.assertEqual(banned, [],
                         f"guard grew a restore path: {banned}")

    def test_unload_posts_keep_alive_zero(self):
        captured = {}

        class _Resp:
            def read(self): return b"{}"
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def _urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = req.data
            return _Resp()

        with mock.patch.object(self.g.urllib.request, "urlopen", _urlopen):
            self.assertTrue(self.g.unload("gemma4:26b"))
        self.assertIn("/api/generate", captured["url"])
        body = captured["body"].decode()
        self.assertIn('"keep_alive": 0', body)
        self.assertIn("gemma4:26b", body)

    def test_unload_pins_num_ctx(self):
        """Caught by tests/test_ollama_runner_reuse.py on 2026-09-04, and it
        would have made this guard cause the failure it exists to prevent.

        Ollama keys a runner by (model, options). A generate post with no
        num_ctx does not address the RESIDENT runner — it asks for a different
        one at the model's own default window (262144 for gemma4:26b-a4b).
        Measured 2026-07-21: that pinned the 3090 at 24147/24576 MiB with the
        model spilled to CPU and the next voice turn died on a 50 s timeout."""
        captured = {}

        class _Resp:
            def read(self): return b"{}"
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def _urlopen(req, timeout=None):
            captured["body"] = req.data
            return _Resp()

        with mock.patch.object(self.g.urllib.request, "urlopen", _urlopen):
            self.g.unload("gemma4:26b-a4b-it-qat")
        import json as _json
        sent = _json.loads(captured["body"].decode())
        self.assertIn("options", sent)
        self.assertIn("num_ctx", sent["options"])
        self.assertGreater(sent["options"]["num_ctx"], 0)
        # And it must be a real window, never the model's 262144 default.
        self.assertLess(sent["options"]["num_ctx"], 100_000)

    def test_num_ctx_never_returns_zero_or_none_even_if_core_is_missing(self):
        # The fallback must never degrade to "send no options" — that IS the bug.
        with mock.patch.dict(sys.modules, {"core.ollama_opts": None}):
            n = self.g._num_ctx("some:tag")
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)


class InhibitTests(_Base):
    """Property 2 — it must never fight the owner."""

    def test_fresh_inhibit_blocks_unloading(self):
        with mock.patch.object(self.g, "running_games", return_value=["f.exe"]), \
             mock.patch.object(self.g, "inhibited", return_value=True), \
             mock.patch.object(self.g, "loaded_models",
                               return_value=[{"name": "m", "size_vram": 1, "size": 1}]), \
             mock.patch.object(self.g, "unload") as un:
            games, freed = self.g.tick(("f.exe",), dry=False)
        self.assertEqual((games, freed), (1, 0))
        un.assert_not_called()

    def test_inhibit_expires(self):
        now = 1_000_000.0
        with mock.patch.object(self.g.os.path, "getmtime",
                               return_value=now - self.g.INHIBIT_TTL_S - 1):
            self.assertFalse(self.g.inhibited(now))
        with mock.patch.object(self.g.os.path, "getmtime",
                               return_value=now - 1):
            self.assertTrue(self.g.inhibited(now))

    def test_missing_inhibit_file_is_not_inhibited(self):
        # Fail OPEN in the direction of doing its job: absent file = free to act.
        with mock.patch.object(self.g.os.path, "getmtime",
                               side_effect=OSError("nope")):
            self.assertFalse(self.g.inhibited())


class LatchTests(_Base):
    """Property 3 — re-unloads every tick, so no request chokepoint is needed."""

    def test_unloads_again_on_the_next_tick(self):
        # Models come back because five separate call sites can reload them
        # (chat, vision, the BOOT warm-up, the orchestrator, the RAG indexer).
        model = [{"name": "gemma4:26b", "size_vram": 15 << 30, "size": 15 << 30}]
        with mock.patch.object(self.g, "running_games", return_value=["f.exe"]), \
             mock.patch.object(self.g, "inhibited", return_value=False), \
             mock.patch.object(self.g, "loaded_models", return_value=model), \
             mock.patch.object(self.g, "unload", return_value=True) as un:
            self.assertEqual(self.g.tick(("f.exe",), dry=False), (1, 1))
            self.assertEqual(self.g.tick(("f.exe",), dry=False), (1, 1))
        self.assertEqual(un.call_count, 2)

    def test_no_game_means_no_unload(self):
        # It must never touch the brain when he is just working.
        with mock.patch.object(self.g, "running_games", return_value=[]), \
             mock.patch.object(self.g, "unload") as un:
            self.assertEqual(self.g.tick(("f.exe",), dry=False), (0, 0))
        un.assert_not_called()

    def test_dry_run_unloads_nothing(self):
        model = [{"name": "m", "size_vram": 1, "size": 1}]
        with mock.patch.object(self.g, "running_games", return_value=["f.exe"]), \
             mock.patch.object(self.g, "inhibited", return_value=False), \
             mock.patch.object(self.g, "loaded_models", return_value=model), \
             mock.patch.object(self.g, "unload") as un:
            games, freed = self.g.tick(("f.exe",), dry=True)
        self.assertEqual((games, freed), (1, 0))
        un.assert_not_called()


class DetectionTests(_Base):
    def test_game_match_is_case_insensitive(self):
        out = types.SimpleNamespace(
            stdout='"FORTNITECLIENT-WIN64-SHIPPING.EXE","123","Console"\n')
        with mock.patch.object(self.g.subprocess, "run", return_value=out):
            self.assertEqual(
                self.g.running_games(("FortniteClient-Win64-Shipping.exe",)),
                ["FortniteClient-Win64-Shipping.exe"])

    def test_tasklist_failure_is_not_fatal(self):
        # A probe that raises would kill the guard exactly when the owner is
        # gaming — the one time it must not die.
        with mock.patch.object(self.g.subprocess, "run",
                               side_effect=OSError("boom")):
            self.assertEqual(self.g.running_games(("f.exe",)), [])

    def test_loaded_models_survives_a_bad_document(self):
        for bad in (None, {}, {"models": None}, {"models": [{"no_name": 1}]}):
            with mock.patch.object(self.g, "_get", return_value=bad):
                self.assertEqual(self.g.loaded_models(), [])

    def test_loaded_models_parses_the_real_shape(self):
        doc = {"models": [{"name": "gemma4:26b-a4b-it-qat",
                           "size": 16_000_000_000, "size_vram": 15_000_000_000}]}
        with mock.patch.object(self.g, "_get", return_value=doc):
            got = self.g.loaded_models()
        self.assertEqual(got[0]["name"], "gemma4:26b-a4b-it-qat")
        self.assertEqual(got[0]["size_vram"], 15_000_000_000)

    def test_ollama_unreachable_yields_no_models(self):
        # JARVIS down / Ollama stopped must read as "nothing to do", never as
        # an exception that takes the guard with it.
        with mock.patch.object(self.g.urllib.request, "urlopen",
                               side_effect=OSError("refused")):
            self.assertEqual(self.g.loaded_models(), [])


class HoldFullPowerTests(_Base):
    def test_hold_full_power_writes_a_fresh_marker(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(self.g, "INHIBIT_FILE",
                                   os.path.join(d, ".inhibit")):
                self.g.hold_full_power()
                self.assertTrue(os.path.exists(self.g.INHIBIT_FILE))
                self.assertTrue(self.g.inhibited(time.time()))

    def test_hold_full_power_never_raises_on_an_unwritable_path(self):
        with mock.patch.object(self.g, "INHIBIT_FILE",
                               os.path.join(os.sep, "nope", "x", ".inhibit")):
            self.g.hold_full_power()          # must not raise


if __name__ == "__main__":
    unittest.main()
