"""Tests for skills.model_picker — choose the LOCAL Ollama model by voice.

No real Ollama, no real network, no real file writes: ``requests`` is patched
on the module, the running monolith is faked via ``sys.modules['__main__']``
(exactly how the live skill reaches it), and persistence is asserted through a
patched ``_persist_setting`` (with a couple of tests exercising the REAL helper
against a mocked tools.settings_window to prove the GUI-writer reuse).

Stdlib unittest + unittest.mock only (App-Control-safe; no pytest).
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from skills import model_picker as M


# ─── settings-file safety net ───────────────────────────────────────────────
# The PersistReuseTests below patch tools.settings_window.load_settings/
# save_settings at the source module, so the real data/user_settings.json is
# never written. This module-level redirect is a second, independent layer:
# JARVIS_SETTINGS_PATH points at a throwaway file for the whole module, so even
# a future unmocked persistence test can't clobber the owner's real settings.
_SAVED_SETTINGS_ENV: "str | None" = None
_SETTINGS_TMPDIR: "str | None" = None


def setUpModule() -> None:
    global _SAVED_SETTINGS_ENV, _SETTINGS_TMPDIR
    _SAVED_SETTINGS_ENV = os.environ.get("JARVIS_SETTINGS_PATH")
    _SETTINGS_TMPDIR = tempfile.mkdtemp(prefix="jarvis_model_picker_test_")
    os.environ["JARVIS_SETTINGS_PATH"] = os.path.join(
        _SETTINGS_TMPDIR, "test_user_settings.json")


def tearDownModule() -> None:
    if _SAVED_SETTINGS_ENV is None:
        os.environ.pop("JARVIS_SETTINGS_PATH", None)
    else:
        os.environ["JARVIS_SETTINGS_PATH"] = _SAVED_SETTINGS_ENV

# The installed models on the dev box (per the task brief): three CHAT models,
# one VISION model, one EMBEDDING model.
CHAT_32B = "qwen2.5:32b-instruct-q4_K_M"
CHAT_14B = "qwen2.5:14b-instruct-q5_K_M"
CHAT_8B = "llama3.1:8b-instruct-q5_K_M"
VISION = "qwen2.5vl:7b"
EMBED = "nomic-embed-text"
ALL_TAGS = [CHAT_32B, CHAT_14B, CHAT_8B, VISION, EMBED]


# ─── Ollama /api/tags faking ───────────────────────────────────────────────
def _tags_response(tags, ok=True):
    r = mock.Mock()
    r.ok = ok
    r.json = mock.Mock(return_value={"models": [{"name": t} for t in tags]})
    return r


def _patch_tags(tags=ALL_TAGS, ok=True):
    """Patch requests.get on the module to return a fake /api/tags."""
    return mock.patch.object(M.requests, "get",
                             return_value=_tags_response(tags, ok=ok))


def _patch_tags_down():
    """Patch requests.get to raise — Ollama unreachable."""
    return mock.patch.object(M.requests, "get",
                             side_effect=OSError("connection refused"))


# ─── fake monolith (the live skill reaches sys.modules['__main__']) ─────────
def _fake_monolith(resolved=None, local_model=CHAT_14B,
                   routing=None):
    bc = types.ModuleType("__main__")
    bc._RESOLVED_LOCAL_LLM_MODEL = [resolved]
    bc.LOCAL_LLM_MODEL = local_model
    bc.MODEL_ROUTING = dict(routing) if routing is not None else {
        "chat": "auto", "vision": "auto", "ambient": "auto"}
    return bc


class _MonolithCtx:
    """Install a fake monolith as __main__ and drop bobert_companion for the
    duration so _monolith() resolves to our fake."""
    def __init__(self, bc):
        self.bc = bc
        self._patch = None
        self._saved_bc = None

    def __enter__(self):
        self._patch = mock.patch.dict(sys.modules, {"__main__": self.bc})
        self._patch.start()
        self._saved_bc = sys.modules.pop("bobert_companion", None)
        return self.bc

    def __exit__(self, *exc):
        self._patch.stop()
        if self._saved_bc is not None:
            sys.modules["bobert_companion"] = self._saved_bc


def _patch_persist(store):
    """Patch _persist_setting to record (key,value) into `store` and report ok."""
    def _fake(key, value):
        store[key] = value
        return True
    return mock.patch.object(M, "_persist_setting", side_effect=_fake)


# ─── pure helpers ───────────────────────────────────────────────────────────
class HelperTests(unittest.TestCase):
    def test_chat_models_excludes_embed_and_vision(self):
        self.assertEqual(M._chat_models(ALL_TAGS), [CHAT_32B, CHAT_14B, CHAT_8B])

    def test_vision_models_picks_vl(self):
        self.assertEqual(M._vision_models(ALL_TAGS), [VISION])

    def test_short_name(self):
        self.assertEqual(M._short_name(CHAT_32B), "qwen 32B")
        self.assertEqual(M._short_name(CHAT_8B), "llama 8B")

    def test_same_model_distinguishes_sizes(self):
        # Different sizes of the same family are DIFFERENT models.
        self.assertFalse(M._same_model(CHAT_14B, CHAT_32B))
        self.assertTrue(M._same_model(CHAT_32B, CHAT_32B))
        # A bare base name matches a sized tag of that family.
        self.assertTrue(M._same_model("qwen2.5", CHAT_32B))

    def test_resolve_alias_table(self):
        chat = [CHAT_32B, CHAT_14B, CHAT_8B]
        self.assertEqual(M._resolve_alias("32b", chat), CHAT_32B)
        self.assertEqual(M._resolve_alias("14b", chat), CHAT_14B)
        self.assertEqual(M._resolve_alias("8b", chat), CHAT_8B)
        self.assertEqual(M._resolve_alias("big", chat), CHAT_32B)
        self.assertEqual(M._resolve_alias("smartest", chat), CHAT_32B)
        self.assertEqual(M._resolve_alias("best", chat), CHAT_32B)
        self.assertEqual(M._resolve_alias("medium", chat), CHAT_14B)
        self.assertEqual(M._resolve_alias("small", chat), CHAT_8B)
        self.assertEqual(M._resolve_alias("fast", chat), CHAT_8B)
        self.assertEqual(M._resolve_alias("light", chat), CHAT_8B)
        self.assertEqual(M._resolve_alias("qwen", chat), CHAT_32B)   # qwen default = 32B
        self.assertEqual(M._resolve_alias("llama", chat), CHAT_8B)
        self.assertEqual(M._resolve_alias(CHAT_8B, chat), CHAT_8B)   # exact tag
        self.assertIsNone(M._resolve_alias("frobnicator", chat))
        self.assertIsNone(M._resolve_alias("", chat))


# ─── list_models ─────────────────────────────────────────────────────────────
class ListModelsTests(unittest.TestCase):
    def test_lists_chat_only_marks_active_and_mentions_vision(self):
        with _patch_tags(), _MonolithCtx(_fake_monolith(resolved=CHAT_14B)):
            out = M.list_models()
        self.assertIn("qwen 32B", out)
        self.assertIn("qwen 14B (active)", out)
        self.assertIn("llama 8B", out)
        # ONLY the active one is marked.
        self.assertNotIn("qwen 32B (active)", out)
        # Embedding model never appears; vision mentioned separately.
        self.assertNotIn("nomic", out.lower())
        self.assertIn("vision", out.lower())

    def test_ollama_down_is_graceful(self):
        with _patch_tags_down():
            out = M.list_models()
        self.assertIn("can't reach ollama", out.lower())

    def test_no_chat_models_installed(self):
        with _patch_tags([VISION, EMBED]):
            out = M.list_models()
        self.assertIn("don't see any local chat models", out.lower())


# ─── current_model ───────────────────────────────────────────────────────────
class CurrentModelTests(unittest.TestCase):
    def test_reports_resolved_cache(self):
        # The resolved-cache tag is reported (short + full) when the chat route
        # is local; on auto/cloud current_model leads with the active brain
        # (see the route-specific tests below).
        with _MonolithCtx(_fake_monolith(resolved=CHAT_32B, routing={"chat": "local"})):
            out = M.current_model()
        self.assertIn("qwen 32B", out)
        self.assertIn(CHAT_32B, out)

    def test_falls_back_to_local_llm_model_global(self):
        # cache cold → reads LOCAL_LLM_MODEL via _get_local_llm_model absence.
        bc = _fake_monolith(resolved=None, local_model=CHAT_8B)
        with _MonolithCtx(bc):
            out = M.current_model()
        self.assertIn("llama 8B", out)

    def test_reports_claude_when_chat_route_is_cloud(self):
        # 2026-07-04 live-repro: after 'switch to the cloud model', current_model
        # kept saying "qwen 14B locally". It must now reflect the live chat route
        # (MODEL_ROUTING['chat'], the SAME source set_brain writes + _call_llm
        # reads) rather than always naming the local model.
        bc = _fake_monolith(resolved=CHAT_32B, routing={"chat": "cloud"})
        with _MonolithCtx(bc):
            out = M.current_model()
        self.assertIn("Claude", out)
        self.assertNotIn("locally", out.lower())

    def test_reports_local_model_when_chat_route_is_local(self):
        bc = _fake_monolith(resolved=CHAT_32B, routing={"chat": "local"})
        with _MonolithCtx(bc):
            out = M.current_model()
        self.assertIn("qwen 32B", out)
        self.assertIn("locally", out.lower())

    def test_reports_auto_names_both_claude_and_local_fallback(self):
        bc = _fake_monolith(resolved=CHAT_8B, routing={"chat": "auto"})
        with _MonolithCtx(bc):
            out = M.current_model()
        self.assertIn("Claude", out)
        self.assertIn("llama 8B", out)


# ─── set_model: happy paths ─────────────────────────────────────────────────
class SetModelHappyTests(unittest.TestCase):
    def test_alias_32b_switches_live_and_persists(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_14B, local_model=CHAT_14B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            out = M.set_model("32b")
        self.assertIn("qwen 32B", out)
        # LIVE: resolver cache + LOCAL_LLM_MODEL global both repointed.
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_32B)
        self.assertEqual(bc.LOCAL_LLM_MODEL, CHAT_32B)
        # PERSISTED: helper called with LOCAL_LLM_MODEL=new tag.
        self.assertEqual(store.get("LOCAL_LLM_MODEL"), CHAT_32B)

    def test_alias_big_maps_to_32b(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_8B, local_model=CHAT_8B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            M.set_model("big")
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_32B)
        self.assertEqual(store.get("LOCAL_LLM_MODEL"), CHAT_32B)

    def test_alias_fast_maps_to_llama_8b(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_32B, local_model=CHAT_32B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            out = M.set_model("fast")
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_8B)
        self.assertEqual(store.get("LOCAL_LLM_MODEL"), CHAT_8B)
        self.assertIn("llama 8B", out)

    def test_alias_small_maps_to_llama_8b(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_32B, local_model=CHAT_32B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            M.set_model("small")
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_8B)

    def test_exact_tag_switches(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_32B, local_model=CHAT_32B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            M.set_model(CHAT_14B)
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_14B)
        self.assertEqual(store.get("LOCAL_LLM_MODEL"), CHAT_14B)

    def test_already_active_is_idempotent_no_persist(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_32B, local_model=CHAT_32B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            out = M.set_model("32b")
        self.assertIn("already running", out.lower())
        self.assertNotIn("LOCAL_LLM_MODEL", store)   # no write when unchanged
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_32B)


# ─── set_model: invalidates the cache (live switch) ─────────────────────────
class SetModelCacheTests(unittest.TestCase):
    def test_switch_repoints_resolver_cache_so_next_turn_uses_it(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_14B, local_model=CHAT_14B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            M.set_model("8b")
            # The skill's own active-resolver now returns the NEW model — i.e.
            # the very next turn's _get_local_llm_model() (which reads the same
            # cache element) would serve the 8B.
            self.assertEqual(M._active_model(), CHAT_8B)
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_8B)

    def test_persist_called_with_new_model(self):
        bc = _fake_monolith(resolved=CHAT_14B, local_model=CHAT_14B)
        with _patch_tags(), _MonolithCtx(bc), \
                mock.patch.object(M, "_persist_setting", return_value=True) as p:
            M.set_model("32b")
        p.assert_called_once_with("LOCAL_LLM_MODEL", CHAT_32B)


# ─── set_model: rejections (never switch to a non-installed / wrong-slot) ────
class SetModelRejectTests(unittest.TestCase):
    def test_rejects_non_installed_and_lists_options(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_14B, local_model=CHAT_14B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            out = M.set_model("mixtral")
        self.assertIn("couldn't match", out.lower())
        self.assertIn("qwen 32B", out)             # lists installed options
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_14B)  # unchanged
        self.assertNotIn("LOCAL_LLM_MODEL", store)                  # not persisted

    def test_rejects_embedding_model_for_chat_slot(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_14B, local_model=CHAT_14B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            out = M.set_model(EMBED)
        self.assertIn("embedding model", out.lower())
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_14B)
        self.assertNotIn("LOCAL_LLM_MODEL", store)

    def test_rejects_vision_model_for_chat_slot(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_14B, local_model=CHAT_14B)
        with _patch_tags(), _MonolithCtx(bc), _patch_persist(store):
            out = M.set_model(VISION)
        self.assertIn("vision model", out.lower())
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_14B)
        self.assertNotIn("LOCAL_LLM_MODEL", store)

    def test_ollama_down_does_not_switch(self):
        store = {}
        bc = _fake_monolith(resolved=CHAT_14B, local_model=CHAT_14B)
        with _patch_tags_down(), _MonolithCtx(bc), _patch_persist(store):
            out = M.set_model("32b")
        self.assertIn("can't reach ollama", out.lower())
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], CHAT_14B)
        self.assertNotIn("LOCAL_LLM_MODEL", store)

    def test_empty_arg_prompts(self):
        with _patch_tags():
            out = M.set_model("")
        self.assertIn("which model", out.lower())


# ─── set_brain: route switch + persist MODEL_ROUTING.chat ───────────────────
class SetBrainTests(unittest.TestCase):
    def test_local_updates_and_persists_chat_route(self):
        store = {}
        bc = _fake_monolith(routing={"chat": "auto", "vision": "local",
                                     "ambient": "auto"})
        with _MonolithCtx(bc), _patch_persist(store):
            out = M.set_brain("local")
        self.assertIn("local model", out.lower())
        self.assertEqual(bc.MODEL_ROUTING["chat"], "local")
        # Other routes preserved live.
        self.assertEqual(bc.MODEL_ROUTING["vision"], "local")
        self.assertEqual(store.get("MODEL_ROUTING"), {"chat": "local"})

    def test_cloud_route(self):
        store = {}
        bc = _fake_monolith()
        with _MonolithCtx(bc), _patch_persist(store):
            out = M.set_brain("cloud")
        self.assertEqual(bc.MODEL_ROUTING["chat"], "cloud")
        self.assertEqual(store.get("MODEL_ROUTING"), {"chat": "cloud"})
        self.assertIn("claude", out.lower())

    def test_auto_route(self):
        store = {}
        bc = _fake_monolith(routing={"chat": "local", "vision": "auto",
                                     "ambient": "auto"})
        with _MonolithCtx(bc), _patch_persist(store):
            M.set_brain("auto")
        self.assertEqual(bc.MODEL_ROUTING["chat"], "auto")
        self.assertEqual(store.get("MODEL_ROUTING"), {"chat": "auto"})

    def test_synonyms_normalise(self):
        store = {}
        bc = _fake_monolith()
        with _MonolithCtx(bc), _patch_persist(store):
            M.set_brain("ollama")     # → local
        self.assertEqual(bc.MODEL_ROUTING["chat"], "local")
        with _MonolithCtx(bc), _patch_persist(store):
            M.set_brain("Claude")     # → cloud
        self.assertEqual(bc.MODEL_ROUTING["chat"], "cloud")

    def test_status_reports_current_route(self):
        bc = _fake_monolith(routing={"chat": "local", "vision": "auto",
                                     "ambient": "auto"})
        with _MonolithCtx(bc):
            out = M.set_brain("")
        self.assertIn("local", out.lower())
        with _MonolithCtx(bc):
            out2 = M.set_brain("status")
        self.assertIn("local", out2.lower())

    def test_unknown_arg_is_guided_not_persisted(self):
        store = {}
        bc = _fake_monolith()
        with _MonolithCtx(bc), _patch_persist(store):
            out = M.set_brain("banana")
        self.assertIn("local, cloud, or auto", out.lower())
        self.assertNotIn("MODEL_ROUTING", store)


# ─── _persist_setting REUSES the Settings-GUI writer (no real file) ─────────
class PersistReuseTests(unittest.TestCase):
    """Prove the persistence path calls tools.settings_window.save_settings
    (the GUI's own atomic, merge-not-clobber writer) and never hand-rolls a
    json.dump — and that a partial MODEL_ROUTING merges, not clobbers.

    The skill does ``from tools import settings_window as sw`` then calls
    ``sw.load_settings`` / ``sw.save_settings``, so we patch those two functions
    ON the real module (robust whether or not it was already imported — patching
    only sys.modules wouldn't redirect the `from tools import` attribute lookup
    once the package is loaded). No real file is ever written."""

    def test_local_llm_model_written_through_save_settings(self):
        from tools import settings_window as sw
        current = {"LOCAL_LLM_MODEL": CHAT_14B, "VOICE_MODE": "turn_based"}
        with mock.patch.object(sw, "load_settings", return_value=dict(current)), \
                mock.patch.object(sw, "save_settings") as save:
            ok = M._persist_setting("LOCAL_LLM_MODEL", CHAT_32B)
        self.assertTrue(ok)
        save.assert_called_once()
        saved = save.call_args.args[0]
        self.assertEqual(saved["LOCAL_LLM_MODEL"], CHAT_32B)
        # other keys preserved (merge-not-clobber).
        self.assertEqual(saved["VOICE_MODE"], "turn_based")

    def test_partial_routing_merges_into_existing_dict(self):
        from tools import settings_window as sw
        current = {"MODEL_ROUTING": {"chat": "auto", "vision": "local",
                                     "ambient": "auto"}}
        with mock.patch.object(sw, "load_settings", return_value=dict(current)), \
                mock.patch.object(sw, "save_settings") as save:
            M._persist_setting("MODEL_ROUTING", {"chat": "cloud"})
        saved = save.call_args.args[0]
        # chat updated, vision/ambient preserved — NOT replaced by {"chat":...}.
        self.assertEqual(saved["MODEL_ROUTING"],
                         {"chat": "cloud", "vision": "local", "ambient": "auto"})

    def test_persist_failure_returns_false(self):
        from tools import settings_window as sw
        with mock.patch.object(sw, "load_settings", return_value={}), \
                mock.patch.object(sw, "save_settings",
                                  side_effect=OSError("disk full")):
            self.assertFalse(M._persist_setting("LOCAL_LLM_MODEL", CHAT_32B))


# ─── register ───────────────────────────────────────────────────────────────
class RegisterTests(unittest.TestCase):
    def test_registers_four_callables(self):
        actions = {}
        M.register(actions)
        self.assertEqual(set(actions),
                         {"list_models", "current_model", "set_model", "set_brain"})
        for fn in actions.values():
            self.assertTrue(callable(fn))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
