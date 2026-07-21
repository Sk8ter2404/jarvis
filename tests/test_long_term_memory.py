"""Unit tests for core.long_term_memory — the tiered (working / semantic /
episodic) long-term memory store.

The module lazily imports chromadb / sentence_transformers / rank_bm25 / torch
and degrades gracefully when any is absent, so these tests exercise BOTH paths:

  * the DEGRADED path  — every heavy dep reports "absent" via the _try_import_*
    probes, so retrieval falls back to BM25-less / dense-less recency, and
    reflection falls back to exact-text dedupe.
  * the WITH-DEPS path — fully controlled FAKES (a deterministic embedder, an
    in-memory chroma collection doing real cosine ranking, and a tiny BM25) are
    injected so no model ever loads, no real Chroma DB is created, and the dense
    + sparse + blend branches are all driven.

No real model is loaded, nothing is written outside a per-test TemporaryDirectory
(every path constant in the module is repointed at the tmp dir in setUp), and the
reflector's background trigger is driven deterministically. numpy is genuinely
installed and used only as a pure deterministic math lib for cosine similarity.

stdlib unittest + unittest.mock only (no pytest).
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import core.long_term_memory as ltm


# ──────────────────────────────────────────────────────────────────────────
#  FAKES — deterministic, offline, no model / no real chroma
# ──────────────────────────────────────────────────────────────────────────

def _vec_for(text: str):
    """Deterministic small unit vector derived from the text's sha1. Real numpy
    is used (pure math, offline). Same text → same vector, so cosine sim of a
    text with itself is exactly 1.0 and the reflector dedupe path is testable."""
    import numpy as np
    h = hashlib.sha1((text or "").encode("utf-8", "ignore")).digest()
    # 8-dim vector from the first 8 hash bytes, then L2-normalised (the real
    # embedder normalises too, so _cosine_sim == dot product is valid).
    v = np.array([b / 255.0 for b in h[:8]], dtype="float64")
    n = np.linalg.norm(v)
    if n == 0:
        v = np.ones(8, dtype="float64")
        n = np.linalg.norm(v)
    return v / n


class FakeEncoded(list):
    """Stands in for the ndarray-of-vectors the real embedder returns. Supports
    .tolist() (used by _chroma_upsert) and indexing/iteration (used elsewhere)."""
    def tolist(self):
        return [list(v) for v in self]


class FakeEmbedder:
    """Replaces SentenceTransformer — encode() returns deterministic vectors and
    NEVER loads a model. Records calls so a test can assert no double-load."""
    instances = 0

    def __init__(self, *a, **k):
        FakeEmbedder.instances += 1
        self.encode_calls = 0

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        return FakeEncoded(_vec_for(t) for t in texts)


class FakeEmbedderRaises(FakeEmbedder):
    def encode(self, texts, **kwargs):
        raise RuntimeError("encode boom")


class FakeChromaCollection:
    """In-memory stand-in for a Chroma collection. Stores vectors and does a
    REAL cosine ranking on query so the dense branch produces meaningful scores.
    Distances are returned as cosine distance (1 - sim), which the module turns
    back into a similarity."""
    def __init__(self):
        self.store: dict[str, dict] = {}

    def add(self, *, ids, embeddings, documents, metadatas):
        for i, fid in enumerate(ids):
            self.store[fid] = {
                "embedding": list(embeddings[i]),
                "document": documents[i],
                "metadata": metadatas[i],
            }

    def delete(self, *, ids=None):
        for fid in (ids or []):
            self.store.pop(fid, None)

    def query(self, *, query_embeddings, n_results, include=None):
        import numpy as np
        q = np.array(query_embeddings[0], dtype="float64")
        scored = []
        for fid, rec in self.store.items():
            v = np.array(rec["embedding"], dtype="float64")
            sim = float(np.dot(q, v))
            scored.append((fid, 1.0 - sim, rec["metadata"]))
        scored.sort(key=lambda t: t[1])
        scored = scored[:n_results]
        return {
            "ids":       [[fid for fid, _, _ in scored]],
            "distances": [[d for _, d, _ in scored]],
            "metadatas": [[m for _, _, m in scored]],
        }


class FakeChromaCollectionQueryRaises(FakeChromaCollection):
    def query(self, *, query_embeddings, n_results, include=None):
        raise RuntimeError("query boom")


class FakeChromaCollectionWithGet(FakeChromaCollection):
    """Adds the .get() used by _reconcile_chroma_locked() to list stored ids."""
    def get(self, *, include=None, ids=None):
        return {"ids": list(self.store.keys())}


class FakeBM25:
    """Minimal rank_bm25.BM25Okapi stand-in. get_scores() returns a token-overlap
    count per corpus doc — deterministic and good enough to drive the sparse
    branch and the [0,1] normalisation."""
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query_tokens):
        qs = set(query_tokens)
        return [float(sum(1 for t in doc if t in qs)) for doc in self.corpus]


class FakeBM25Raises(FakeBM25):
    def get_scores(self, query_tokens):
        raise RuntimeError("bm25 boom")


# ──────────────────────────────────────────────────────────────────────────
#  BASE FIXTURE — isolate module globals + repoint all paths at a tmp dir
# ──────────────────────────────────────────────────────────────────────────

# Every module-level path constant we repoint, and every cached/global piece of
# state we must reset so tests don't bleed into each other.
_PATH_ATTRS = ("_DATA_DIR", "_CHROMA_DIR", "_FACTS_JSON", "_EPISODE_LOG",
               "_MIGRATE_FLAG", "_LEGACY_BOBERT_MEMORY")
_STATE_ATTRS = ("_chroma_client", "_collection", "_embedder",
                "_embedder_failed_until", "_bm25_index",
                "_bm25_corpus_ids", "_bm25_corpus", "_facts", "_working",
                "_loaded", "_turns_since_reflect", "_writes_since_rotate",
                "_reflector_llm")


class _LtmBase(unittest.TestCase):
    """Repoints LTM's paths into a fresh temp dir and resets every global so each
    test starts from a clean, in-process store. Restores everything on teardown."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        # Snapshot & repoint path constants.
        self._orig_paths = {a: getattr(ltm, a) for a in _PATH_ATTRS}
        ltm._DATA_DIR = os.path.join(d, "ltm")
        ltm._CHROMA_DIR = os.path.join(ltm._DATA_DIR, "chroma")
        ltm._FACTS_JSON = os.path.join(ltm._DATA_DIR, "facts.json")
        ltm._EPISODE_LOG = os.path.join(ltm._DATA_DIR, "episodes.jsonl")
        ltm._MIGRATE_FLAG = os.path.join(ltm._DATA_DIR, "migrated.flag")
        ltm._LEGACY_BOBERT_MEMORY = os.path.join(d, "bobert_memory.json")
        # Snapshot & reset mutable global state.
        self._orig_state = {a: getattr(ltm, a) for a in _STATE_ATTRS}
        ltm._chroma_client = None
        ltm._collection = None
        ltm._embedder = None
        ltm._embedder_failed_until = 0.0
        ltm._bm25_index = None
        ltm._bm25_corpus_ids = []
        ltm._bm25_corpus = []
        ltm._facts = {}
        ltm._working = []
        ltm._loaded = False
        ltm._turns_since_reflect = 0
        ltm._writes_since_rotate = 0
        ltm._reflector_llm = None
        FakeEmbedder.instances = 0

    def tearDown(self):
        for a, v in self._orig_paths.items():
            setattr(ltm, a, v)
        for a, v in self._orig_state.items():
            setattr(ltm, a, v)
        self._tmp.cleanup()

    # — helpers ————————————————————————————————————————————————————————
    def _force_no_deps(self):
        """Make every heavy-dep probe report absent (degraded path)."""
        self._patchers = [
            mock.patch.object(ltm, "_try_import_chroma", lambda: None),
            mock.patch.object(ltm, "_try_import_embedder", lambda: None),
            mock.patch.object(ltm, "_try_import_bm25", lambda: False),
        ]
        for p in self._patchers:
            p.start()
            self.addCleanup(p.stop)

    def _install_fake_embedder(self):
        emb = FakeEmbedder()
        ltm._embedder = emb
        p = mock.patch.object(ltm, "_try_import_embedder", lambda: emb)
        p.start()
        self.addCleanup(p.stop)
        return emb

    def _install_fake_chroma(self, coll=None):
        coll = coll or FakeChromaCollection()
        ltm._collection = coll
        p = mock.patch.object(ltm, "_try_import_chroma", lambda: coll)
        p.start()
        self.addCleanup(p.stop)
        return coll

    def _install_fake_bm25(self, cls=FakeBM25):
        """Make _try_import_bm25 true and the `from rank_bm25 import BM25Okapi`
        inside the rebuild resolve to our fake by injecting a fake module."""
        fake_mod = types.ModuleType("rank_bm25")
        fake_mod.BM25Okapi = cls
        p_mod = mock.patch.dict(sys.modules, {"rank_bm25": fake_mod})
        p_mod.start()
        self.addCleanup(p_mod.stop)
        p_probe = mock.patch.object(ltm, "_try_import_bm25", lambda: True)
        p_probe.start()
        self.addCleanup(p_probe.stop)
        return fake_mod


# ──────────────────────────────────────────────────────────────────────────
#  ensure_loaded / boot / migration
# ──────────────────────────────────────────────────────────────────────────

class EnsureLoadedTests(_LtmBase):
    def test_ensure_loaded_idempotent_and_sets_loaded(self):
        self._force_no_deps()
        self.assertFalse(ltm._loaded)
        ltm.ensure_loaded()
        self.assertTrue(ltm._loaded)
        # Second call is a cheap no-op: patch _load_facts_locked to detect re-run.
        with mock.patch.object(ltm, "_load_facts_locked",
                               side_effect=AssertionError("should not reload")):
            ltm.ensure_loaded()

    def test_ensure_loaded_creates_data_dir_and_migrate_flag(self):
        self._force_no_deps()
        ltm.ensure_loaded()
        self.assertTrue(os.path.isdir(ltm._DATA_DIR))
        # No legacy file → a "no-legacy" marker flag is dropped.
        self.assertTrue(os.path.exists(ltm._MIGRATE_FLAG))
        with open(ltm._MIGRATE_FLAG, encoding="utf-8") as f:
            self.assertIn("no-legacy", f.read())


class MigrationTests(_LtmBase):
    def _write_legacy(self, facts):
        with open(ltm._LEGACY_BOBERT_MEMORY, "w", encoding="utf-8") as f:
            json.dump({"facts": facts}, f)

    def test_migrates_legacy_facts(self):
        self._force_no_deps()
        self._write_legacy(["User likes lofi", "  Cat named Biscuit ",
                            "", 42, "User likes lofi"])
        ltm.ensure_loaded()
        texts = sorted(e["text"] for e in ltm._facts.values())
        # Blank + non-string skipped; the dup-by-text within legacy collapses.
        self.assertEqual(texts, ["Cat named Biscuit", "User likes lofi"])
        for e in ltm._facts.values():
            self.assertEqual(e["source"], "bobert_memory_migration")
            self.assertIn("legacy", e["tags"])
        # Flag records the count, facts mirror was written.
        with open(ltm._MIGRATE_FLAG, encoding="utf-8") as f:
            self.assertIn("migrated=2", f.read())
        self.assertTrue(os.path.exists(ltm._FACTS_JSON))

    def test_migration_skipped_when_flag_exists(self):
        self._force_no_deps()
        os.makedirs(ltm._DATA_DIR, exist_ok=True)
        with open(ltm._MIGRATE_FLAG, "w", encoding="utf-8") as f:
            f.write("already\n")
        self._write_legacy(["should not import"])
        ltm.ensure_loaded()
        self.assertEqual(ltm._facts, {})

    def test_migration_corrupt_legacy_file_is_safe(self):
        self._force_no_deps()
        with open(ltm._LEGACY_BOBERT_MEMORY, "w", encoding="utf-8") as f:
            f.write("{ not json")
        ltm.ensure_loaded()       # must not raise
        self.assertEqual(ltm._facts, {})

    def test_migration_skips_text_already_present(self):
        # Pre-seed a fact whose text matches a legacy entry → not re-imported.
        self._force_no_deps()
        self._write_legacy(["Known fact", "New fact"])
        ltm._facts["pre"] = {"id": "pre", "text": "Known fact", "source": "x",
                             "tags": [], "created_at": 1.0, "updated_at": 1.0}
        # Skip _load_facts (which would clear our pre-seed) by faking the file.
        os.makedirs(ltm._DATA_DIR, exist_ok=True)
        with open(ltm._FACTS_JSON, "w", encoding="utf-8") as f:
            json.dump([ltm._facts["pre"]], f)
        ltm.ensure_loaded()
        texts = sorted(e["text"] for e in ltm._facts.values())
        self.assertEqual(texts, ["Known fact", "New fact"])  # only one of each


# ──────────────────────────────────────────────────────────────────────────
#  Persistence: _save_facts_locked / _load_facts_locked
# ──────────────────────────────────────────────────────────────────────────

class PersistenceTests(_LtmBase):
    def test_save_then_load_roundtrip(self):
        self._force_no_deps()
        ltm.ensure_loaded()
        fid = ltm.add_fact("Persisted fact", source="s", tags=["a", "b"])
        # Reload from disk into a clean dict.
        ltm._facts = {}
        with ltm._lock:
            ltm._load_facts_locked()
        self.assertIn(fid, ltm._facts)
        e = ltm._facts[fid]
        self.assertEqual(e["text"], "Persisted fact")
        self.assertEqual(e["tags"], ["a", "b"])

    def test_load_missing_file_clears(self):
        self._force_no_deps()
        ltm._facts = {"x": {"id": "x"}}
        with ltm._lock:
            ltm._load_facts_locked()           # file doesn't exist yet
        self.assertEqual(ltm._facts, {})

    def test_load_corrupt_json_is_safe(self):
        self._force_no_deps()
        os.makedirs(ltm._DATA_DIR, exist_ok=True)
        with open(ltm._FACTS_JSON, "w", encoding="utf-8") as f:
            f.write("{ not a list or json")
        with ltm._lock:
            ltm._load_facts_locked()
        self.assertEqual(ltm._facts, {})

    def test_load_non_list_json_is_ignored(self):
        self._force_no_deps()
        os.makedirs(ltm._DATA_DIR, exist_ok=True)
        with open(ltm._FACTS_JSON, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)
        with ltm._lock:
            ltm._load_facts_locked()
        self.assertEqual(ltm._facts, {})

    def test_load_skips_bad_entries_and_defaults_missing_keys(self):
        self._force_no_deps()
        os.makedirs(ltm._DATA_DIR, exist_ok=True)
        data = [
            "not a dict",                      # skipped (not dict)
            {"text": "no id"},                 # skipped (no id)
            {"id": "", "text": "blank id"},    # skipped (blank id)
            {"id": "ok", "text": "fine"},      # kept, defaults filled
        ]
        with open(ltm._FACTS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f)
        with ltm._lock:
            ltm._load_facts_locked()
        self.assertEqual(list(ltm._facts), ["ok"])
        e = ltm._facts["ok"]
        for k in ("source", "tags", "created_at", "updated_at"):
            self.assertIn(k, e)

    def test_save_failure_is_swallowed(self):
        self._force_no_deps()
        ltm.ensure_loaded()
        with mock.patch.object(ltm, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            with ltm._lock:
                ltm._save_facts_locked()       # must not raise

    def test_save_leaves_no_tempfiles(self):
        self._force_no_deps()
        ltm.ensure_loaded()
        ltm.add_fact("a fact")
        leftovers = [f for f in os.listdir(ltm._DATA_DIR) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])


# ──────────────────────────────────────────────────────────────────────────
#  add / update / delete / list  (degraded path)
# ──────────────────────────────────────────────────────────────────────────

class FactCrudDegradedTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_add_fact_returns_id_and_stores(self):
        fid = ltm.add_fact("  Spaces trimmed  ", source="src", tags=["t"])
        self.assertTrue(fid.startswith("fact_"))
        e = ltm._facts[fid]
        self.assertEqual(e["text"], "Spaces trimmed")
        self.assertEqual(e["source"], "src")
        self.assertEqual(e["tags"], ["t"])
        self.assertEqual(e["created_at"], e["updated_at"])

    def test_add_fact_empty_raises(self):
        with self.assertRaises(ValueError):
            ltm.add_fact("   ")
        with self.assertRaises(ValueError):
            ltm.add_fact("")

    def test_add_fact_dedupes_exact_text(self):
        a = ltm.add_fact("same text")
        b = ltm.add_fact("same text")
        self.assertEqual(a, b)
        self.assertEqual(len(ltm._facts), 1)

    def test_add_fact_tags_default_empty_list(self):
        fid = ltm.add_fact("no tags given")
        self.assertEqual(ltm._facts[fid]["tags"], [])

    def test_update_fact_changes_text_and_timestamp(self):
        fid = ltm.add_fact("original")
        ltm._facts[fid]["updated_at"] = 0.0    # force a detectable bump
        self.assertTrue(ltm.update_fact(fid, "  updated  "))
        self.assertEqual(ltm._facts[fid]["text"], "updated")
        self.assertGreater(ltm._facts[fid]["updated_at"], 0.0)

    def test_update_fact_unknown_id_false(self):
        self.assertFalse(ltm.update_fact("nope", "x"))

    def test_update_fact_empty_text_false(self):
        fid = ltm.add_fact("keep me")
        self.assertFalse(ltm.update_fact(fid, "   "))
        self.assertEqual(ltm._facts[fid]["text"], "keep me")

    def test_delete_fact_removes(self):
        fid = ltm.add_fact("to delete")
        self.assertTrue(ltm.delete_fact(fid))
        self.assertNotIn(fid, ltm._facts)

    def test_delete_fact_unknown_id_false(self):
        self.assertFalse(ltm.delete_fact("ghost"))

    def test_list_facts_sorted_newest_first_and_limit(self):
        a = ltm.add_fact("first")
        b = ltm.add_fact("second")
        c = ltm.add_fact("third")
        # Make ordering deterministic regardless of clock resolution.
        ltm._facts[a]["updated_at"] = 100.0
        ltm._facts[b]["updated_at"] = 200.0
        ltm._facts[c]["updated_at"] = 300.0
        out = ltm.list_facts()
        self.assertEqual([e["id"] for e in out], [c, b, a])
        self.assertEqual([e["id"] for e in ltm.list_facts(limit=2)], [c, b])
        # Returned dicts are copies, not the live entries.
        out[0]["text"] = "mutated"
        self.assertNotEqual(ltm._facts[c]["text"], "mutated")

    def test_new_fact_id_is_deterministic_prefix(self):
        i1 = ltm._new_fact_id("hello")
        i2 = ltm._new_fact_id("hello")
        # Same sha1 prefix, different random suffix → never collide but share head.
        self.assertEqual(i1.rsplit("_", 1)[0], i2.rsplit("_", 1)[0])
        self.assertNotEqual(i1, i2)


# ──────────────────────────────────────────────────────────────────────────
#  add / update / delete  (WITH chroma+embedder fakes — dense write path)
# ──────────────────────────────────────────────────────────────────────────

class FactCrudWithDepsTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self.emb = self._install_fake_embedder()
        self.coll = self._install_fake_chroma()
        ltm.ensure_loaded()

    def test_add_fact_upserts_to_chroma(self):
        fid = ltm.add_fact("dense me", tags=["x", "y"])
        self.assertIn(fid, self.coll.store)
        # tags flattened to CSV in chroma metadata.
        self.assertEqual(self.coll.store[fid]["metadata"]["tags"], "x,y")
        self.assertEqual(self.coll.store[fid]["document"], "dense me")

    def test_update_fact_reupserts_to_chroma(self):
        fid = ltm.add_fact("v1")
        ltm.update_fact(fid, "v2")
        self.assertEqual(self.coll.store[fid]["document"], "v2")

    def test_delete_fact_removes_from_chroma(self):
        fid = ltm.add_fact("temp")
        ltm.delete_fact(fid)
        self.assertNotIn(fid, self.coll.store)

    def test_chroma_upsert_false_when_no_collection(self):
        with mock.patch.object(ltm, "_try_import_chroma", lambda: None):
            self.assertFalse(ltm._chroma_upsert("id", "t", {}))

    def test_chroma_upsert_false_when_embed_none(self):
        with mock.patch.object(ltm, "_embed", lambda texts: None):
            self.assertFalse(ltm._chroma_upsert("id", "t", {"tags": ["a"]}))

    def test_chroma_upsert_handles_collection_add_raising(self):
        boom = mock.MagicMock()
        boom.delete.side_effect = None
        boom.add.side_effect = RuntimeError("add failed")
        with mock.patch.object(ltm, "_try_import_chroma", lambda: boom):
            self.assertFalse(ltm._chroma_upsert("id", "t", {}))

    def test_chroma_upsert_delete_precheck_swallows(self):
        # The pre-delete inside upsert raising must not abort the add.
        coll = FakeChromaCollection()
        coll.delete = mock.MagicMock(side_effect=RuntimeError("delete boom"))
        with mock.patch.object(ltm, "_try_import_chroma", lambda: coll):
            ok = ltm._chroma_upsert("idz", "txt", {"tags": ["a"]})
        self.assertTrue(ok)
        self.assertIn("idz", coll.store)

    def test_chroma_delete_no_collection_noop(self):
        with mock.patch.object(ltm, "_try_import_chroma", lambda: None):
            ltm._chroma_delete("anything")     # must not raise

    def test_chroma_delete_swallows_errors(self):
        boom = mock.MagicMock()
        boom.delete.side_effect = RuntimeError("nope")
        with mock.patch.object(ltm, "_try_import_chroma", lambda: boom):
            ltm._chroma_delete("id")           # must not raise


# ──────────────────────────────────────────────────────────────────────────
#  _embed  (the encode wrapper)
# ──────────────────────────────────────────────────────────────────────────

class EmbedTests(_LtmBase):
    def test_embed_none_without_embedder(self):
        with mock.patch.object(ltm, "_try_import_embedder", lambda: None):
            self.assertIsNone(ltm._embed(["x"]))

    def test_embed_returns_vectors(self):
        self._install_fake_embedder()
        out = ltm._embed(["a", "b"])
        self.assertEqual(len(out), 2)

    def test_embed_swallows_encode_error(self):
        emb = FakeEmbedderRaises()
        with mock.patch.object(ltm, "_try_import_embedder", lambda: emb):
            self.assertIsNone(ltm._embed(["x"]))


# ──────────────────────────────────────────────────────────────────────────
#  BM25 index build + tokenizer
# ──────────────────────────────────────────────────────────────────────────

class Bm25Tests(_LtmBase):
    def test_tokenize(self):
        self.assertEqual(ltm._tokenize("Hello, WORLD! it's 42"),
                         ["hello", "world", "it's", "42"])
        self.assertEqual(ltm._tokenize(""), [])
        self.assertEqual(ltm._tokenize(None), [])

    def test_rebuild_bm25_no_facts_resets(self):
        self._install_fake_bm25()
        ltm._facts = {}
        with ltm._lock:
            ltm._rebuild_bm25_locked()
        self.assertIsNone(ltm._bm25_index)
        self.assertEqual(ltm._bm25_corpus_ids, [])

    def test_rebuild_bm25_absent_dep_resets(self):
        with mock.patch.object(ltm, "_try_import_bm25", lambda: False):
            ltm._facts = {"a": {"id": "a", "text": "hello world"}}
            with ltm._lock:
                ltm._rebuild_bm25_locked()
        self.assertIsNone(ltm._bm25_index)

    def test_rebuild_bm25_builds_index(self):
        self._install_fake_bm25()
        ltm._facts = {
            "a": {"id": "a", "text": "the cat sat"},
            "b": {"id": "b", "text": ""},            # empty → no tokens, skipped
            "c": {"id": "c", "text": "dog barked"},
        }
        with ltm._lock:
            ltm._rebuild_bm25_locked()
        self.assertIsInstance(ltm._bm25_index, FakeBM25)
        self.assertEqual(sorted(ltm._bm25_corpus_ids), ["a", "c"])

    def test_rebuild_bm25_all_empty_text_no_index(self):
        self._install_fake_bm25()
        ltm._facts = {"a": {"id": "a", "text": "   "}}
        with ltm._lock:
            ltm._rebuild_bm25_locked()
        self.assertIsNone(ltm._bm25_index)

    def test_rebuild_bm25_constructor_raises_safely(self):
        self._install_fake_bm25(cls=_Bm25CtorRaises)
        ltm._facts = {"a": {"id": "a", "text": "hello"}}
        with ltm._lock:
            ltm._rebuild_bm25_locked()
        self.assertIsNone(ltm._bm25_index)


class _Bm25CtorRaises:
    def __init__(self, corpus):
        raise RuntimeError("ctor boom")


# ──────────────────────────────────────────────────────────────────────────
#  retrieve_facts — every branch
# ──────────────────────────────────────────────────────────────────────────

class RetrieveDegradedTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_empty_query_returns_recent(self):
        a = ltm.add_fact("alpha")
        b = ltm.add_fact("beta")
        ltm._facts[a]["updated_at"] = 1.0
        ltm._facts[b]["updated_at"] = 2.0
        out = ltm.retrieve_facts("   ", k=5)
        self.assertEqual([e["id"] for e in out], [b, a])

    def test_no_facts_returns_empty(self):
        self.assertEqual(ltm.retrieve_facts("anything"), [])

    def test_no_scores_falls_back_to_recent(self):
        # Degraded: no chroma, no bm25 → blend has nothing → recency fallback.
        a = ltm.add_fact("one")
        b = ltm.add_fact("two")
        ltm._facts[a]["updated_at"] = 5.0
        ltm._facts[b]["updated_at"] = 9.0
        out = ltm.retrieve_facts("query text", k=1)
        self.assertEqual([e["id"] for e in out], [b])


class RetrieveWithDepsTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self.emb = self._install_fake_embedder()
        self.coll = self._install_fake_chroma()
        self._install_fake_bm25()
        ltm.ensure_loaded()

    def test_hybrid_blend_returns_scored(self):
        fid = ltm.add_fact("the quick brown fox")
        ltm.add_fact("entirely unrelated content")
        out = ltm.retrieve_facts("quick brown fox", k=2)
        self.assertTrue(out)
        # Top hit is the lexically+semantically matching fact, with a score key.
        self.assertEqual(out[0]["id"], fid)
        self.assertIn("score", out[0])
        self.assertIsInstance(out[0]["score"], float)

    def test_dense_only_when_bm25_absent(self):
        # chroma present, bm25 index missing → dense-only branch.
        ltm._bm25_index = None
        ltm._bm25_corpus_ids = []
        fid = ltm.add_fact("dense path only")
        out = ltm.retrieve_facts("dense path only", k=3)
        self.assertTrue(out)
        self.assertEqual(out[0]["id"], fid)

    def test_sparse_only_when_chroma_absent(self):
        ltm.add_fact("lexical match term apple")
        with mock.patch.object(ltm, "_try_import_chroma", lambda: None):
            out = ltm.retrieve_facts("apple", k=3)
        self.assertTrue(out)
        self.assertIn("apple", out[0]["text"])

    def test_chroma_query_error_falls_through_to_sparse(self):
        ltm.add_fact("banana split term")
        bad = FakeChromaCollectionQueryRaises()
        # Re-add into the broken collection so ids exist there too.
        bad.add(ids=["x"], embeddings=[list(_vec_for("banana split term"))],
                documents=["banana split term"], metadatas=[{}])
        with mock.patch.object(ltm, "_try_import_chroma", lambda: bad):
            out = ltm.retrieve_facts("banana", k=3)
        # Dense raised but sparse still produced a hit.
        self.assertTrue(out)

    def test_dense_skipped_when_embed_none(self):
        ltm.add_fact("cherry pie term")
        with mock.patch.object(ltm, "_embed", lambda texts: None):
            out = ltm.retrieve_facts("cherry", k=3)
        self.assertTrue(out)        # sparse carried it

    def test_bm25_score_error_falls_through_to_dense(self):
        fid = ltm.add_fact("date fruit term")
        ltm._bm25_index = FakeBM25Raises(ltm._bm25_corpus)
        out = ltm.retrieve_facts("date", k=3)
        self.assertTrue(out)        # dense carried it
        self.assertEqual(out[0]["id"], fid)

    def test_retrieved_id_missing_from_facts_is_skipped(self):
        # Score a fid that no longer exists in _facts → skipped in output.
        ltm.add_fact("present term zebra")
        with mock.patch.object(ltm, "_try_import_chroma", lambda: None):
            with mock.patch.object(
                ltm._bm25_index, "get_scores",
                lambda toks: [9.0] * len(ltm._bm25_corpus_ids),
            ):
                # Drop one fact from the dict after scoring corpus was built.
                ghost = "ghost_id"
                ltm._bm25_corpus_ids = list(ltm._bm25_corpus_ids) + [ghost]
                ltm._bm25_corpus = list(ltm._bm25_corpus) + [["zebra"]]
                out = ltm.retrieve_facts("zebra", k=5)
        self.assertTrue(all(e["id"] != "ghost_id" for e in out))

    def test_bm25_all_zero_scores_no_sparse(self):
        # Query tokens match nothing → max score 0 → sparse dict stays empty,
        # dense still answers.
        ltm.add_fact("elderberry term")
        out = ltm.retrieve_facts("zzzznomatch", k=3)
        # dense always returns something (cosine over all stored vecs).
        self.assertTrue(out)


# ──────────────────────────────────────────────────────────────────────────
#  working memory + record_turn + episodic log
# ──────────────────────────────────────────────────────────────────────────

class WorkingMemoryTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_record_and_get_window(self):
        ltm.record_turn("user", "hello there")
        ltm.record_turn("assistant", "hi")
        win = ltm.get_working_window(10)
        self.assertEqual([t["role"] for t in win], ["user", "assistant"])
        self.assertEqual(win[0]["text"], "hello there")

    def test_record_turn_empty_dropped(self):
        ltm.record_turn("user", "   ")
        ltm.record_turn("user", "")
        self.assertEqual(ltm._working, [])

    def test_record_turn_defaults_role_and_truncates(self):
        ltm.record_turn("", "x" * 5000)
        e = ltm._working[-1]
        self.assertEqual(e["role"], "user")
        self.assertEqual(len(e["text"]), 2000)

    def test_record_turn_custom_ts_populates_calendar_fields(self):
        ts = time.mktime(time.strptime("2026-05-04 13:00:00",
                                       "%Y-%m-%d %H:%M:%S"))
        ltm.record_turn("user", "dated", ts=ts)
        e = ltm._working[-1]
        self.assertEqual(e["date"], "2026-05-04")
        self.assertEqual(e["hour"], 13)
        self.assertEqual(e["ts"], ts)

    def test_get_working_window_zero_or_negative(self):
        ltm.record_turn("user", "a")
        self.assertEqual(ltm.get_working_window(0), [])
        self.assertEqual(ltm.get_working_window(-3), [])

    def test_working_window_copies(self):
        ltm.record_turn("user", "orig")
        win = ltm.get_working_window(5)
        win[0]["text"] = "mutated"
        self.assertEqual(ltm._working[-1]["text"], "orig")

    def test_working_ring_buffer_trims(self):
        cap = ltm.WORKING_WINDOW * 4
        for i in range(cap + 10):
            ltm.record_turn("user", f"turn {i}")
        self.assertLessEqual(len(ltm._working), cap)
        # Oldest got trimmed; newest retained.
        self.assertEqual(ltm._working[-1]["text"], f"turn {cap + 9}")

    def test_record_turn_appends_to_episode_log(self):
        ltm.record_turn("user", "logged turn")
        self.assertTrue(os.path.exists(ltm._EPISODE_LOG))
        with open(ltm._EPISODE_LOG, encoding="utf-8") as f:
            line = json.loads(f.readline())
        self.assertEqual(line["text"], "logged turn")


class ReflectorTriggerTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_reflector_fires_at_threshold(self):
        with mock.patch.object(ltm, "reflect_and_consolidate") as ref:
            for _ in range(ltm.REFLECTOR_RUN_EVERY_TURNS - 1):
                ltm.record_turn("user", "x")
            ref.assert_not_called()
            ltm.record_turn("user", "trip it")
            ref.assert_called_once()
        self.assertEqual(ltm._turns_since_reflect, 0)   # counter reset

    def test_reflector_exception_is_swallowed(self):
        with mock.patch.object(ltm, "reflect_and_consolidate",
                               side_effect=RuntimeError("reflect boom")):
            for _ in range(ltm.REFLECTOR_RUN_EVERY_TURNS):
                ltm.record_turn("user", "x")     # must not raise
        self.assertEqual(ltm._turns_since_reflect, 0)


class EpisodeRotationTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_episode_write_failure_swallowed(self):
        m = mock.mock_open()
        m.side_effect = OSError("cannot write")
        with mock.patch("builtins.open", m):
            with ltm._lock:
                ltm._append_episode_locked({"text": "x"})   # must not raise

    def test_rotation_trims_to_max_lines(self):
        # Shrink thresholds so the test is fast and deterministic. Rotation only
        # runs on every EPISODE_ROTATE_CHECK_EVERY-th write, trimming to the last
        # EPISODE_MAX_LINES; writes after the final check accumulate on top. With
        # 20 writes / check-every-3 / max-4: the last check fires at write 18
        # (leaving i=14..17), then writes 19,20 append → 6 lines i=14..19. The
        # store stays bounded (never grows unbounded) which is the contract.
        with mock.patch.object(ltm, "EPISODE_ROTATE_CHECK_EVERY", 3), \
             mock.patch.object(ltm, "EPISODE_MAX_LINES", 4):
            for i in range(20):
                with ltm._lock:
                    ltm._append_episode_locked({"i": i})
        with open(ltm._EPISODE_LOG, encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        # Bounded well under the unrotated count of 20.
        self.assertLessEqual(len(lines), ltm.EPISODE_MAX_LINES
                             + ltm.EPISODE_ROTATE_CHECK_EVERY)
        self.assertEqual([e["i"] for e in lines], [14, 15, 16, 17, 18, 19])
        # A trim demonstrably happened: the earliest writes are gone.
        self.assertNotIn(0, [e["i"] for e in lines])

    def test_rotation_check_under_max_keeps_all(self):
        with mock.patch.object(ltm, "EPISODE_ROTATE_CHECK_EVERY", 2), \
             mock.patch.object(ltm, "EPISODE_MAX_LINES", 1000):
            for i in range(5):
                with ltm._lock:
                    ltm._append_episode_locked({"i": i})
        with open(ltm._EPISODE_LOG, encoding="utf-8") as f:
            lines = [x for x in f if x.strip()]
        self.assertEqual(len(lines), 5)

    def test_rotation_read_failure_swallowed(self):
        # Force the rotation check to run, then make the re-read raise.
        ltm._writes_since_rotate = ltm.EPISODE_ROTATE_CHECK_EVERY - 1
        real_open = builtins.open

        def flaky_open(path, mode="r", *a, **k):
            # Allow the append write; fail the subsequent read-for-rotation.
            if "r" in mode and str(path).endswith("episodes.jsonl"):
                raise OSError("read boom")
            return real_open(path, mode, *a, **k)

        with mock.patch("builtins.open", flaky_open):
            with ltm._lock:
                ltm._append_episode_locked({"text": "y"})   # must not raise
        self.assertEqual(ltm._writes_since_rotate, 0)        # counter still reset


# ──────────────────────────────────────────────────────────────────────────
#  episodic search
# ──────────────────────────────────────────────────────────────────────────

class SearchEpisodesTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def _seed(self, rows):
        os.makedirs(ltm._DATA_DIR, exist_ok=True)
        with open(ltm._EPISODE_LOG, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_no_log_returns_empty(self):
        self.assertEqual(ltm.search_episodes("anything"), [])

    def test_substring_match_case_insensitive_newest_first(self):
        self._seed([
            {"ts": 1.0, "date": "2026-05-01", "text": "Bought APPLES today"},
            {"ts": 2.0, "date": "2026-05-02", "text": "talked about apples again"},
            {"ts": 3.0, "date": "2026-05-03", "text": "unrelated banana"},
        ])
        out = ltm.search_episodes("apple")
        self.assertEqual([e["ts"] for e in out], [2.0, 1.0])

    def test_empty_query_returns_all_in_window(self):
        self._seed([{"ts": 1.0, "date": "2026-05-01", "text": "a"},
                    {"ts": 2.0, "date": "2026-05-02", "text": "b"}])
        out = ltm.search_episodes("")
        self.assertEqual(len(out), 2)

    def test_date_window_filtering(self):
        import datetime as dt
        self._seed([
            {"ts": 1.0, "date": "2026-05-01", "text": "early"},
            {"ts": 2.0, "date": "2026-05-15", "text": "mid"},
            {"ts": 3.0, "date": "2026-05-31", "text": "late"},
        ])
        out = ltm.search_episodes("", start=dt.date(2026, 5, 10),
                                  end=dt.date(2026, 5, 20))
        self.assertEqual([e["text"] for e in out], ["mid"])

    def test_rows_without_parseable_date_excluded_when_window_set(self):
        import datetime as dt
        self._seed([
            {"ts": 1.0, "date": "", "text": "no date"},
            {"ts": 2.0, "date": "not-a-date", "text": "bad date"},
            {"ts": 3.0, "date": "2026-05-15", "text": "good"},
        ])
        out = ltm.search_episodes("", start=dt.date(2026, 5, 1))
        self.assertEqual([e["text"] for e in out], ["good"])

    def test_limit_applied(self):
        self._seed([{"ts": float(i), "date": "2026-05-01", "text": f"r{i}"}
                    for i in range(30)])
        self.assertEqual(len(ltm.search_episodes("", limit=5)), 5)

    def test_iter_episodes_skips_corrupt_lines(self):
        os.makedirs(ltm._DATA_DIR, exist_ok=True)
        with open(ltm._EPISODE_LOG, "w", encoding="utf-8") as f:
            f.write('{"ts": 1.0, "date": "2026-05-01", "text": "ok"}\n')
            f.write("\n")                       # blank skipped
            f.write("{ broken json\n")          # corrupt skipped
        out = ltm.search_episodes("")
        self.assertEqual([e["text"] for e in out], ["ok"])

    def test_iter_episodes_open_failure_safe(self):
        os.makedirs(ltm._DATA_DIR, exist_ok=True)
        with open(ltm._EPISODE_LOG, "w", encoding="utf-8") as f:
            f.write('{"ts": 1.0, "text": "x"}\n')
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(list(ltm._iter_episodes()), [])


# ──────────────────────────────────────────────────────────────────────────
#  reflector — _cosine_sim + reflect_and_consolidate
# ──────────────────────────────────────────────────────────────────────────

class CosineSimTests(_LtmBase):
    def test_none_inputs_zero(self):
        self.assertEqual(ltm._cosine_sim(None, [1, 0]), 0.0)
        self.assertEqual(ltm._cosine_sim([1, 0], None), 0.0)

    def test_identical_unit_vectors_one(self):
        v = _vec_for("same")
        self.assertAlmostEqual(ltm._cosine_sim(v, v), 1.0, places=6)

    def test_numpy_absent_returns_zero(self):
        # Simulate numpy import failing inside _cosine_sim.
        real_import = builtins.__import__

        def no_numpy(name, *a, **k):
            if name == "numpy":
                raise ImportError("no numpy")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=no_numpy):
            self.assertEqual(ltm._cosine_sim([1, 0], [1, 0]), 0.0)


class ReflectDegradedTests(_LtmBase):
    """No embedder → exact-text dedupe branch only."""
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_fewer_than_two_facts_noops(self):
        ltm.add_fact("only one")
        summary = ltm.reflect_and_consolidate()
        self.assertEqual(summary["duplicates_removed"], 0)
        self.assertEqual(summary["checked_pairs"], 0)

    def test_exact_text_dedupe_removes_older(self):
        # Bypass add_fact's own dedupe by writing _facts directly.
        ltm._facts = {
            "old": {"id": "old", "text": "dup", "source": "", "tags": [],
                    "created_at": 1.0, "updated_at": 1.0},
            "new": {"id": "new", "text": "dup", "source": "", "tags": [],
                    "created_at": 2.0, "updated_at": 2.0},
            "uniq": {"id": "uniq", "text": "unique", "source": "", "tags": [],
                     "created_at": 1.5, "updated_at": 1.5},
        }
        summary = ltm.reflect_and_consolidate()
        self.assertEqual(summary["duplicates_removed"], 1)
        self.assertNotIn("old", ltm._facts)     # older removed
        self.assertIn("new", ltm._facts)
        self.assertIn("uniq", ltm._facts)

    def test_no_duplicates_no_save(self):
        ltm._facts = {
            "a": {"id": "a", "text": "one", "source": "", "tags": [],
                  "created_at": 1.0, "updated_at": 1.0},
            "b": {"id": "b", "text": "two", "source": "", "tags": [],
                  "created_at": 2.0, "updated_at": 2.0},
        }
        with mock.patch.object(ltm, "_save_facts_locked") as save:
            summary = ltm.reflect_and_consolidate()
            save.assert_not_called()
        self.assertEqual(summary["duplicates_removed"], 0)


class ReflectWithEmbedderTests(_LtmBase):
    """Embedder present → pairwise cosine scan drives dup / contradiction / merge."""
    def setUp(self):
        super().setUp()
        self.emb = self._install_fake_embedder()
        self.coll = self._install_fake_chroma()
        ltm.ensure_loaded()

    def _seed(self, items):
        """items: list of (id, text, created_at)."""
        ltm._facts = {}
        for fid, text, ca in items:
            ltm._facts[fid] = {"id": fid, "text": text, "source": "",
                               "tags": [], "created_at": ca, "updated_at": ca}

    def test_near_dup_removes_older(self):
        # Identical text → cosine 1.0 ≥ REFLECTOR_DUP_SIM → older dropped.
        self._seed([("old", "identical fact", 1.0),
                    ("new", "identical fact", 2.0)])
        summary = ltm.reflect_and_consolidate()
        self.assertGreaterEqual(summary["duplicates_removed"], 1)
        self.assertIn("new", ltm._facts)
        self.assertNotIn("old", ltm._facts)

    def test_near_dup_no_cascade_over_delete(self):
        # 2026-07-07 bug-hunt (LOW-MED): X is a near-dup of BOTH Y and Z, but Y
        # and Z are NOT near-dups of each other. X is deleted in pair (X,Y);
        # the OLD code kept comparing the now-doomed X against Z and deleted Z
        # too (Z was only similar to the condemned X), AND counted 2 removals.
        # After the fix, once X is condemned the inner loop breaks, so Z survives
        # and the count reflects the single real deletion.
        # ids order is by updated_at DESC → [X, Y, Z]; deletion picks the older
        # created_at, so created_at Z(1) < X(2) < Y(3) makes X lose to Y and,
        # under the old cascade, Z would lose to X.
        ltm._facts = {
            "X": {"id": "X", "text": "fact x", "source": "", "tags": [],
                  "created_at": 2.0, "updated_at": 9.0},
            "Y": {"id": "Y", "text": "fact y", "source": "", "tags": [],
                  "created_at": 3.0, "updated_at": 8.0},
            "Z": {"id": "Z", "text": "fact z", "source": "", "tags": [],
                  "created_at": 1.0, "updated_at": 7.0},
        }
        # vecs aligned to the updated_at-desc order [X, Y, Z] = [[0],[1],[2]];
        # sim is high whenever X (vector 0.0) is one of the pair, else 0.
        with mock.patch.object(ltm, "_embed",
                               return_value=[[0.0], [1.0], [2.0]]), \
             mock.patch.object(ltm, "_cosine_sim",
                               side_effect=lambda a, b: 1.0 if 0.0 in (a[0], b[0]) else 0.0), \
             mock.patch.object(ltm, "_save_facts_locked"), \
             mock.patch.object(ltm, "_rebuild_bm25_locked"), \
             mock.patch.object(ltm, "_chroma_delete"):
            summary = ltm.reflect_and_consolidate()
        self.assertNotIn("X", ltm._facts)          # the genuine near-dup, removed
        self.assertIn("Y", ltm._facts)             # kept
        self.assertIn("Z", ltm._facts)             # NOT cascade-deleted
        self.assertEqual(summary["duplicates_removed"], 1)  # not 2

    def test_pairwise_cap_applied(self):
        with mock.patch.object(ltm, "REFLECTOR_MAX_PAIRWISE", 3):
            self._seed([(f"id{i}", f"fact number {i}", float(i))
                        for i in range(10)])
            # Each fact text is distinct → no removals, but the cap path runs.
            summary = ltm.reflect_and_consolidate()
        # With only 3 facts considered, max checked pairs is C(3,2)=3.
        self.assertLessEqual(summary["checked_pairs"], 3)

    def test_contradiction_keep_A_deletes_B(self):
        # Facts are scanned newest-first, so ids[i] is the newer fact ("b",
        # updated_at 2.0) and ids[j] the older ("a"). Verdict 'A' keeps the
        # first presented (ids[i]="b") and deletes ids[j]="a".
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0)])
        with mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            summary = ltm.reflect_and_consolidate(llm_call=lambda p, ctx: "A")
        self.assertEqual(summary["contradictions_resolved"], 1)
        self.assertIn("b", ltm._facts)         # ids[i] kept
        self.assertNotIn("a", ltm._facts)      # ids[j] removed

    def test_contradiction_keep_B_deletes_A(self):
        # Verdict 'B' keeps the second presented (ids[j]="a") and deletes
        # ids[i]="b".
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0)])
        with mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            summary = ltm.reflect_and_consolidate(llm_call=lambda p, ctx: "B")
        self.assertEqual(summary["contradictions_resolved"], 1)
        self.assertIn("a", ltm._facts)         # ids[j] kept
        self.assertNotIn("b", ltm._facts)      # ids[i] removed

    def test_contradiction_blank_verdict_keeps_both(self):
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0)])
        with mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            summary = ltm.reflect_and_consolidate(llm_call=lambda p, ctx: "")
        self.assertEqual(summary["contradictions_resolved"], 0)
        self.assertIn("a", ltm._facts)
        self.assertIn("b", ltm._facts)

    def test_merge_replaces_survivor_and_deletes_other(self):
        # MERGE rewrites the survivor ids[i] (newer = "b") and deletes ids[j].
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0)])
        with mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            summary = ltm.reflect_and_consolidate(
                llm_call=lambda p, ctx: "MERGE: fused fact")
        self.assertEqual(summary["merged"], 1)
        self.assertIn("b", ltm._facts)
        self.assertEqual(ltm._facts["b"]["text"], "fused fact")
        self.assertNotIn("a", ltm._facts)

    def test_merge_empty_text_is_ignored(self):
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0)])
        with mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            summary = ltm.reflect_and_consolidate(
                llm_call=lambda p, ctx: "MERGE:   ")
        self.assertEqual(summary["merged"], 0)
        self.assertIn("a", ltm._facts)
        self.assertIn("b", ltm._facts)

    def test_llm_call_raising_is_swallowed_as_blank(self):
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0)])

        def boom(prompt, ctx):
            raise RuntimeError("llm down")

        with mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            summary = ltm.reflect_and_consolidate(llm_call=boom)
        # Treated as '' → both kept, no contradiction resolved.
        self.assertEqual(summary["contradictions_resolved"], 0)
        self.assertIn("a", ltm._facts)
        self.assertIn("b", ltm._facts)

    def test_mid_band_without_llm_call_skips_contradiction(self):
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0)])
        with mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            summary = ltm.reflect_and_consolidate(llm_call=None)
        self.assertEqual(summary["contradictions_resolved"], 0)
        self.assertEqual(summary["merged"], 0)

    def test_delete_skipped_if_text_changed_under_us(self):
        # to_delete computed, but a concurrent edit changed the survivor's text
        # so the content-guard refuses the delete.
        self._seed([("old", "identical fact", 1.0),
                    ("new", "identical fact", 2.0)])
        real_embed = ltm._embed

        def embed_then_mutate(texts):
            out = real_embed(texts)
            # After embeddings computed, mutate the loser's text so the guard
            # (entry.text == snapshot_text[fid]) fails and it is NOT deleted.
            ltm._facts["old"]["text"] = "changed since snapshot"
            return out

        with mock.patch.object(ltm, "_embed", side_effect=embed_then_mutate):
            ltm.reflect_and_consolidate()
        # Guard should have prevented deletion of the mutated 'old'.
        self.assertIn("old", ltm._facts)


# ──────────────────────────────────────────────────────────────────────────
#  Reflector LLM injection (set_reflector_llm) — 2026-07-21 audit #39
# ──────────────────────────────────────────────────────────────────────────

class ReflectorInjectionTests(_LtmBase):
    """The contradiction pass was DEAD in production: record_turn's periodic
    trigger called reflect_and_consolidate() with no llm_call, and no other
    production caller existed. These pin the injection hook: the trigger now
    passes the installed adjudicator, defaults to the old behavior when none
    is installed, honours the trusted-source guard, and bounds LLM work."""

    def setUp(self):
        super().setUp()
        self._install_fake_embedder()
        self._install_fake_chroma()
        ltm.ensure_loaded()

    def _seed(self, items):
        """items: list of (id, text, created_at[, source])."""
        ltm._facts = {}
        for row in items:
            fid, text, ca = row[0], row[1], row[2]
            src = row[3] if len(row) > 3 else ""
            ltm._facts[fid] = {"id": fid, "text": text, "source": src,
                               "tags": [], "created_at": ca, "updated_at": ca}

    def test_record_turn_trigger_uses_injected_llm(self):
        # REGRESSION (audit #39): the production trigger must no longer call
        # reflect_and_consolidate with llm_call=None — the injected adjudicator
        # must actually resolve a mid-band contradiction from record_turn.
        calls = []

        def spy(prompt, ctx):
            calls.append((prompt, ctx))
            return "B"          # keep the second presented, condemn the first

        ltm.set_reflector_llm(spy)
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0)])
        with mock.patch.object(ltm, "REFLECTOR_RUN_EVERY_TURNS", 3), \
             mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            for _ in range(3):
                ltm.record_turn("user", "another turn")
        self.assertTrue(calls, "injected adjudicator was never invoked "
                               "from record_turn's reflector trigger")
        # ids scan newest-first → ids[i]="b"; verdict 'B' condemns it.
        self.assertNotIn("b", ltm._facts)
        self.assertIn("a", ltm._facts)

    def test_without_injection_defaults_to_old_behavior(self):
        # No adjudicator installed → llm_call=None → contradiction pass
        # skipped and both facts stay (exactly the pre-fix behavior).
        ltm.set_reflector_llm(None)
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0)])
        with mock.patch.object(ltm, "REFLECTOR_RUN_EVERY_TURNS", 3), \
             mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            for _ in range(3):
                ltm.record_turn("user", "another turn")
        self.assertIn("a", ltm._facts)
        self.assertIn("b", ltm._facts)

    def test_trusted_source_guard_keeps_migrated_fact(self):
        # Verdict 'A' condemns ids[j] = the older, migration-sourced fact;
        # the survivor is an untrusted ambient extraction → guard: both stay.
        self._seed([("mig", "user's name is Marcus", 1.0,
                     "bobert_memory_migration"),
                    ("amb", "user's name is Rodney", 2.0, "merge_memory")])
        with mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            summary = ltm.reflect_and_consolidate(llm_call=lambda p, c: "A")
        self.assertEqual(summary["contradictions_resolved"], 0)
        self.assertIn("mig", ltm._facts)
        self.assertIn("amb", ltm._facts)

    def test_untrusted_condemned_still_deleted(self):
        # Complementary arm: condemning the AMBIENT fact in favour of the
        # migrated one is allowed (verdict 'B' condemns ids[i] = "amb").
        self._seed([("mig", "user's name is Marcus", 1.0,
                     "bobert_memory_migration"),
                    ("amb", "user's name is Rodney", 2.0, "merge_memory")])
        with mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            summary = ltm.reflect_and_consolidate(llm_call=lambda p, c: "B")
        self.assertEqual(summary["contradictions_resolved"], 1)
        self.assertIn("mig", ltm._facts)
        self.assertNotIn("amb", ltm._facts)

    def test_llm_pair_cap_bounds_adjudications(self):
        calls = []

        def spy(prompt, ctx):
            calls.append(prompt)
            return ""           # both stay — we only count invocations

        # 3 facts, every pair mid-band → 3 qualifying pairs, cap of 1.
        self._seed([("a", "fact a text", 1.0), ("b", "fact b text", 2.0),
                    ("c", "fact c text", 3.0)])
        with mock.patch.object(ltm, "REFLECTOR_MAX_LLM_PAIRS", 1), \
             mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7):
            ltm.reflect_and_consolidate(llm_call=spy)
        self.assertEqual(len(calls), 1)


# ──────────────────────────────────────────────────────────────────────────
#  reset_all — full wipe with backup (2026-07-21 audit #17)
# ──────────────────────────────────────────────────────────────────────────

class ResetAllTests(_LtmBase):
    """_act_reset_memory claimed a full wipe while the semantic store kept
    every fact and the verbatim episode log. reset_all() is the LTM side of
    that wipe: backup-then-clear of facts + episodes + working window, with
    migrated.flag left in place so legacy facts can't resurrect."""

    def _backup_dirs(self):
        root = os.path.join(ltm._DATA_DIR, "backups")
        if not os.path.isdir(root):
            return []
        return sorted(os.path.join(root, d) for d in os.listdir(root)
                      if d.startswith("pre_reset_"))

    def test_wipes_facts_episodes_working_and_backs_up(self):
        self._force_no_deps()
        ltm.ensure_loaded()
        ltm.add_fact("User's address is 12 Secret Lane")
        ltm.add_fact("User has a dog")
        ltm.record_turn("user", "a private turn")
        ltm.record_turn("assistant", "noted, sir")
        n = ltm.reset_all()
        self.assertEqual(n, 2)
        # (a) store empty from every read path
        self.assertEqual(ltm.list_facts(), [])
        self.assertEqual(ltm.retrieve_facts("address"), [])
        self.assertEqual(ltm.get_working_window(), [])
        self.assertEqual(ltm.search_episodes(""), [])
        self.assertFalse(os.path.exists(ltm._EPISODE_LOG) and
                         os.path.getsize(ltm._EPISODE_LOG) > 0)
        # (b) pre_reset backup holds the OLD facts + episode log
        backups = self._backup_dirs()
        self.assertEqual(len(backups), 1)
        with open(os.path.join(backups[0], "facts.json"),
                  encoding="utf-8") as f:
            old = json.load(f)
        self.assertEqual(sorted(e["text"] for e in old),
                         ["User has a dog", "User's address is 12 Secret Lane"])
        self.assertTrue(os.path.exists(
            os.path.join(backups[0], "episodes.jsonl")))

    def test_backup_failure_refuses_to_wipe(self):
        self._force_no_deps()
        ltm.ensure_loaded()
        fid = ltm.add_fact("keep me")
        with mock.patch.object(ltm.shutil, "copy2",
                               side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                ltm.reset_all()
        self.assertIn(fid, ltm._facts)             # nothing wiped
        self.assertTrue(os.path.exists(ltm._FACTS_JSON))

    def test_leaves_migrate_flag_so_legacy_cannot_resurrect(self):
        self._force_no_deps()
        with open(ltm._LEGACY_BOBERT_MEMORY, "w", encoding="utf-8") as f:
            json.dump({"facts": ["legacy fact"]}, f)
        ltm.ensure_loaded()
        self.assertEqual(len(ltm._facts), 1)
        self.assertEqual(ltm.reset_all(), 1)
        self.assertTrue(os.path.exists(ltm._MIGRATE_FLAG))
        # A later boot must NOT re-import the wiped legacy facts.
        ltm._loaded = False
        ltm.ensure_loaded()
        self.assertEqual(ltm._facts, {})

    def test_chroma_absent_wipe_still_succeeds(self):
        self._force_no_deps()
        ltm.ensure_loaded()
        ltm.add_fact("degraded-path fact")
        self.assertEqual(ltm.reset_all(), 1)
        self.assertEqual(ltm.list_facts(), [])

    def test_chroma_client_collection_recreated(self):
        self._install_fake_embedder()
        self._install_fake_chroma()
        fresh = FakeChromaCollection()
        client = mock.Mock()
        client.get_or_create_collection.return_value = fresh
        ltm._chroma_client = client
        ltm.ensure_loaded()
        ltm.add_fact("wipe me")
        self.assertEqual(ltm.reset_all(), 1)
        client.delete_collection.assert_called_once_with(ltm.LTM_COLLECTION)
        self.assertIs(ltm._collection, fresh)

    def test_chroma_without_client_deletes_per_id(self):
        self._install_fake_embedder()
        coll = self._install_fake_chroma()     # _chroma_client stays None
        ltm.ensure_loaded()
        fid = ltm.add_fact("wipe me")
        self.assertIn(fid, coll.store)
        self.assertEqual(ltm.reset_all(), 1)
        self.assertEqual(coll.store, {})


# ──────────────────────────────────────────────────────────────────────────
#  forget_since — time-window purge (2026-07-21 audit #51)
# ──────────────────────────────────────────────────────────────────────────

class ForgetSinceTests(_LtmBase):
    """'Forget the last hour' left the hour's verbatim turns in
    episodes.jsonl, its facts in the semantic store, and the turns in the
    in-process working window. forget_since purges all three by timestamp."""

    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_purges_recent_only_across_all_three_tiers(self):
        now = time.time()
        old_ts, recent_ts = now - 7200, now - 1800
        ltm.record_turn("user", "old turn about the garden", ts=old_ts)
        ltm.record_turn("user", "recent turn about the surprise party",
                        ts=recent_ts)
        # A legacy/unparseable line must survive the rewrite untouched.
        with open(ltm._EPISODE_LOG, "a", encoding="utf-8") as f:
            f.write("not json at all\n")
        ltm._facts = {
            "fold": {"id": "fold", "text": "old durable fact", "source": "",
                     "tags": [], "created_at": old_ts, "updated_at": old_ts},
            "fnew": {"id": "fnew", "text": "fact learned just now",
                     "source": "", "tags": [], "created_at": recent_ts,
                     "updated_at": recent_ts},
        }
        with mock.patch.object(ltm, "_chroma_delete") as cdel:
            res = ltm.forget_since(now - 3600)
        self.assertEqual(res, {"episodes": 1, "facts": 1, "working": 1})
        # Episodic log: recent line gone, old + unparseable kept, atomic.
        with open(ltm._EPISODE_LOG, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("old turn about the garden", content)
        self.assertIn("not json at all", content)
        self.assertNotIn("surprise party", content)
        self.assertFalse(os.path.exists(ltm._EPISODE_LOG + ".tmp"))
        self.assertEqual(ltm.search_episodes("surprise party"), [])
        self.assertEqual(len(ltm.search_episodes("garden")), 1)
        # Working window: only the old turn remains in prompt context.
        window_texts = [t["text"] for t in ltm.get_working_window()]
        self.assertEqual(window_texts, ["old turn about the garden"])
        # Facts: recent one gone (incl. its chroma vector), old survives.
        self.assertNotIn("fnew", ltm._facts)
        self.assertIn("fold", ltm._facts)
        cdel.assert_called_once_with("fnew")

    def test_nothing_in_window_is_a_counted_noop(self):
        now = time.time()
        ltm.record_turn("user", "ancient turn", ts=now - 7200)
        ltm._facts = {
            "fold": {"id": "fold", "text": "old fact", "source": "",
                     "tags": [], "created_at": now - 7200,
                     "updated_at": now - 7200},
        }
        with mock.patch.object(ltm, "_save_facts_locked") as save:
            res = ltm.forget_since(now - 3600)
            save.assert_not_called()
        self.assertEqual(res, {"episodes": 0, "facts": 0, "working": 0})
        self.assertFalse(os.path.exists(ltm._EPISODE_LOG + ".tmp"))
        self.assertIn("fold", ltm._facts)

    def test_ts_less_fact_treated_as_old_and_kept(self):
        now = time.time()
        ltm._facts = {
            "weird": {"id": "weird", "text": "no created_at", "source": "",
                      "tags": [], "created_at": "garbage",
                      "updated_at": now},
        }
        res = ltm.forget_since(now - 3600)
        self.assertEqual(res["facts"], 0)
        self.assertIn("weird", ltm._facts)


# ──────────────────────────────────────────────────────────────────────────
#  is_available / status / config_summary  (real import probes)
# ──────────────────────────────────────────────────────────────────────────

class AvailabilityTests(_LtmBase):
    def test_is_available_keys_and_types(self):
        a = ltm.is_available()
        for k in ("chromadb", "sentence_transformers", "rank_bm25",
                  "fully_available"):
            self.assertIn(k, a)
            self.assertIsInstance(a[k], bool)
        self.assertEqual(
            a["fully_available"],
            a["chromadb"] and a["sentence_transformers"] and a["rank_bm25"])

    def test_is_available_all_present(self):
        # Inject fake modules so all three probes report True deterministically.
        fakes = {name: types.ModuleType(name)
                 for name in ("chromadb", "sentence_transformers", "rank_bm25")}
        with mock.patch.dict(sys.modules, fakes):
            a = ltm.is_available()
        self.assertTrue(a["chromadb"])
        self.assertTrue(a["sentence_transformers"])
        self.assertTrue(a["rank_bm25"])
        self.assertTrue(a["fully_available"])

    def test_is_available_all_absent(self):
        real_import = builtins.__import__

        def block(name, *a, **k):
            if name in ("chromadb", "sentence_transformers", "rank_bm25"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        # Also drop any cached copies so the import statement re-executes.
        with mock.patch.dict(sys.modules, {}, clear=False):
            for n in ("chromadb", "sentence_transformers", "rank_bm25"):
                sys.modules.pop(n, None)
            with mock.patch("builtins.__import__", side_effect=block):
                a = ltm.is_available()
        self.assertFalse(a["chromadb"])
        self.assertFalse(a["sentence_transformers"])
        self.assertFalse(a["rank_bm25"])
        self.assertFalse(a["fully_available"])


class StatusTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_status_shape(self):
        ltm.add_fact("a fact")
        ltm.record_turn("user", "a turn")
        st = ltm.status()
        self.assertEqual(st["facts"], 1)
        self.assertEqual(st["working"], 1)
        self.assertEqual(st["episodes"], 1)
        self.assertTrue(st["loaded"])
        self.assertIn("available", st)
        self.assertEqual(st["facts_path"], ltm._FACTS_JSON)
        self.assertTrue(st["migrated"])

    def test_status_episode_count_open_failure_safe(self):
        ltm.record_turn("user", "x")
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            st = ltm.status()
        self.assertEqual(st["episodes"], 0)     # counting failed → 0

    def test_status_no_episode_log(self):
        st = ltm.status()
        self.assertEqual(st["episodes"], 0)


class ConfigSummaryTests(_LtmBase):
    def test_config_summary_values(self):
        cs = ltm.config_summary()
        self.assertEqual(cs["LTM_COLLECTION"], ltm.LTM_COLLECTION)
        self.assertEqual(cs["WORKING_WINDOW"], ltm.WORKING_WINDOW)
        self.assertEqual(cs["RETRIEVE_K"], ltm.RETRIEVE_K)
        self.assertAlmostEqual(cs["HYBRID_DENSE_W"] + cs["HYBRID_SPARSE_W"], 1.0)


# ──────────────────────────────────────────────────────────────────────────
#  lazy import probes — exercise the real _try_import_* with injected fakes
# ──────────────────────────────────────────────────────────────────────────

class LazyImportProbeTests(_LtmBase):
    def test_try_import_bm25_true_with_fake(self):
        fake = types.ModuleType("rank_bm25")
        fake.BM25Okapi = FakeBM25
        with mock.patch.dict(sys.modules, {"rank_bm25": fake}):
            self.assertTrue(ltm._try_import_bm25())

    def test_try_import_bm25_false_when_absent(self):
        real_import = builtins.__import__

        def block(name, *a, **k):
            if name == "rank_bm25":
                raise ImportError("absent")
            return real_import(name, *a, **k)

        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("rank_bm25", None)
            with mock.patch("builtins.__import__", side_effect=block):
                self.assertFalse(ltm._try_import_bm25())

    def test_try_import_embedder_fast_path_returns_cached(self):
        sentinel = object()
        ltm._embedder = sentinel
        self.assertIs(ltm._try_import_embedder(), sentinel)

    def test_try_import_embedder_absent_returns_none(self):
        ltm._embedder = None
        real_import = builtins.__import__

        def block(name, *a, **k):
            if name == "sentence_transformers":
                raise ImportError("absent")
            return real_import(name, *a, **k)

        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("sentence_transformers", None)
            with mock.patch("builtins.__import__", side_effect=block):
                self.assertIsNone(ltm._try_import_embedder())

    def test_try_import_embedder_builds_with_fake_module(self):
        ltm._embedder = None
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = FakeEmbedder
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                           "torch": fake_torch}):
            emb = ltm._try_import_embedder()
        self.assertIsInstance(emb, FakeEmbedder)
        self.assertIs(ltm._embedder, emb)

    def test_try_import_embedder_cuda_path(self):
        ltm._embedder = None
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = FakeEmbedder
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                           "torch": fake_torch}):
            emb = ltm._try_import_embedder()
        self.assertIsInstance(emb, FakeEmbedder)

    def test_ltm_embed_device_knob_forces_cpu_even_with_cuda(self):
        # LTM_EMBED_DEVICE="cpu" (config/user_settings) must win over the
        # cuda auto-pick — frees ~0.4GB VRAM for the local LLM. 2026-07-10.
        ltm._embedder = None
        seen = {}

        class RecordingEmbedder(FakeEmbedder):
            def __init__(self, model, device=None, **k):
                super().__init__()
                seen["device"] = device

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = RecordingEmbedder
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        import core.config as _cfg
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                           "torch": fake_torch}), \
                mock.patch.object(_cfg, "LTM_EMBED_DEVICE", "cpu", create=True):
            emb = ltm._try_import_embedder()
        self.assertIsInstance(emb, FakeEmbedder)
        self.assertEqual(seen.get("device"), "cpu")

    def test_try_import_embedder_cuda_load_fail_falls_back_to_cpu(self):
        ltm._embedder = None
        # First construction (cuda) raises; the CPU retry succeeds.
        state = {"calls": 0}

        class CudaFailsThenCpu:
            def __init__(self, model, device=None):
                state["calls"] += 1
                if device == "cuda":
                    raise RuntimeError("cuda OOM")

            def encode(self, texts, **k):
                return FakeEncoded(_vec_for(t) for t in texts)

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = CudaFailsThenCpu
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        # Pin the device knob to auto ("") — the LIVE user_settings.json may
        # set LTM_EMBED_DEVICE="cpu", which would skip the cuda attempt this
        # test exists to exercise.
        import core.config as _cfg
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                           "torch": fake_torch}), \
                mock.patch.object(_cfg, "LTM_EMBED_DEVICE", "", create=True):
            emb = ltm._try_import_embedder()
        self.assertIsInstance(emb, CudaFailsThenCpu)
        self.assertEqual(state["calls"], 2)      # cuda attempt + cpu retry

    def test_try_import_embedder_cuda_and_cpu_both_fail_returns_none(self):
        ltm._embedder = None

        class AlwaysFails:
            def __init__(self, model, device=None):
                raise RuntimeError(f"fail on {device}")

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = AlwaysFails
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                           "torch": fake_torch}):
            self.assertIsNone(ltm._try_import_embedder())

    def test_try_import_embedder_backoff_blocks_hot_retry(self):
        # 2026-07-07 regression guard: a FAILED load must arm a cooldown so the
        # next embed call does NOT re-attempt the full model load (that hot-retry
        # loaded the model 174x in one session and stressed the box).
        ltm._embedder = None
        ltm._embedder_failed_until = 0.0
        attempts = {"n": 0}

        class Fails:
            def __init__(self, model, device=None):
                attempts["n"] += 1
                raise RuntimeError(f"fail on {device}")

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        fail_st = types.ModuleType("sentence_transformers")
        fail_st.SentenceTransformer = Fails
        with mock.patch.dict(sys.modules, {"sentence_transformers": fail_st,
                                           "torch": fake_torch}):
            self.assertIsNone(ltm._try_import_embedder())
        first = attempts["n"]
        self.assertGreater(first, 0)                          # it did try
        self.assertGreater(ltm._embedder_failed_until, 0.0)   # cooldown armed

        # With a WORKING embedder now available, the cooldown STILL blocks — no
        # new construction happens until the cooldown elapses.
        ok_st = types.ModuleType("sentence_transformers")
        ok_st.SentenceTransformer = FakeEmbedder
        with mock.patch.dict(sys.modules, {"sentence_transformers": ok_st,
                                           "torch": fake_torch}):
            self.assertIsNone(ltm._try_import_embedder())     # blocked by cooldown
            self.assertEqual(attempts["n"], first)            # NO re-attempt
            ltm._embedder_failed_until = 0.0                  # cooldown elapsed
            emb = ltm._try_import_embedder()
        self.assertIsInstance(emb, FakeEmbedder)              # now it loads

    def test_try_import_embedder_torch_absent_defaults_cpu(self):
        ltm._embedder = None
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = FakeEmbedder
        real_import = builtins.__import__

        def block_torch(name, *a, **k):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *a, **k)

        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_st}):
            sys.modules.pop("torch", None)
            with mock.patch("builtins.__import__", side_effect=block_torch):
                emb = ltm._try_import_embedder()
        self.assertIsInstance(emb, FakeEmbedder)

    def test_try_import_chroma_fast_path_returns_cached(self):
        sentinel = object()
        ltm._collection = sentinel
        self.assertIs(ltm._try_import_chroma(), sentinel)

    def test_try_import_chroma_absent_returns_none(self):
        ltm._collection = None
        real_import = builtins.__import__

        def block(name, *a, **k):
            if name == "chromadb":
                raise ImportError("absent")
            return real_import(name, *a, **k)

        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("chromadb", None)
            with mock.patch("builtins.__import__", side_effect=block):
                self.assertIsNone(ltm._try_import_chroma())

    def test_try_import_chroma_builds_with_fake_module(self):
        ltm._collection = None
        ltm._chroma_client = None
        coll = FakeChromaCollection()

        class FakeClient:
            def __init__(self, path=None):
                self.path = path

            def get_or_create_collection(self, name=None, metadata=None):
                return coll

        fake_chroma = types.ModuleType("chromadb")
        fake_chroma.PersistentClient = FakeClient
        with mock.patch.dict(sys.modules, {"chromadb": fake_chroma}):
            got = ltm._try_import_chroma()
        self.assertIs(got, coll)
        self.assertIs(ltm._collection, coll)
        # Chroma dir created under the tmp dir.
        self.assertTrue(os.path.isdir(ltm._CHROMA_DIR))

    def test_try_import_chroma_client_error_returns_none(self):
        ltm._collection = None
        ltm._chroma_client = None

        class BoomClient:
            def __init__(self, path=None):
                raise RuntimeError("chroma init failed")

        fake_chroma = types.ModuleType("chromadb")
        fake_chroma.PersistentClient = BoomClient
        with mock.patch.dict(sys.modules, {"chromadb": fake_chroma}):
            self.assertIsNone(ltm._try_import_chroma())


# ──────────────────────────────────────────────────────────────────────────
#  defensive / edge-path coverage (swallowed exceptions, race re-checks)
# ──────────────────────────────────────────────────────────────────────────

class DefensiveEdgeTests(_LtmBase):
    def test_ensure_dirs_swallows_makedirs_error(self):
        with mock.patch.object(ltm.os, "makedirs",
                               side_effect=OSError("denied")):
            ltm._ensure_dirs()             # must not raise

    def test_rebuild_bm25_import_raises_after_probe_true(self):
        # _try_import_bm25() says yes, but the subsequent `from rank_bm25
        # import BM25Okapi` raises (module present but attribute missing).
        broken = types.ModuleType("rank_bm25")   # no BM25Okapi attribute
        with mock.patch.object(ltm, "_try_import_bm25", lambda: True), \
             mock.patch.dict(sys.modules, {"rank_bm25": broken}):
            ltm._facts = {"a": {"id": "a", "text": "hello world"}}
            with ltm._lock:
                ltm._rebuild_bm25_locked()
        self.assertIsNone(ltm._bm25_index)

    def test_migrate_no_legacy_flag_write_failure_swallowed(self):
        self._force_no_deps()
        # No legacy file exists; force the "no-legacy" flag write to fail.
        real_open = builtins.open

        def fail_flag(path, mode="r", *a, **k):
            if str(path).endswith("migrated.flag"):
                raise OSError("cannot write flag")
            return real_open(path, mode, *a, **k)

        with mock.patch("builtins.open", fail_flag):
            with ltm._lock:
                n = ltm._migrate_legacy_locked()    # must not raise
        self.assertEqual(n, 0)

    def test_migrate_flag_write_failure_after_migration_swallowed(self):
        # Legacy facts present → migration runs; the post-migration flag write
        # fails (lines 465-466) but the migrated count is still returned.
        self._force_no_deps()
        with open(ltm._LEGACY_BOBERT_MEMORY, "w", encoding="utf-8") as f:
            json.dump({"facts": ["legacy fact one", "legacy fact two"]}, f)
        real_open = builtins.open

        def fail_flag(path, mode="r", *a, **k):
            if str(path).endswith("migrated.flag") and "w" in mode:
                raise OSError("cannot write flag")
            return real_open(path, mode, *a, **k)

        with mock.patch("builtins.open", fail_flag):
            with ltm._lock:
                n = ltm._migrate_legacy_locked()       # must not raise
        self.assertEqual(n, 2)
        self.assertEqual(len(ltm._facts), 2)

    def test_chroma_fast_path_recheck_under_lock(self):
        # Drive line 163: _collection is None on the unlocked check but another
        # "thread" sets it before/at lock acquisition. We emulate by making the
        # lock's __enter__ populate _collection.
        ltm._collection = None
        sentinel = object()

        class SettingLock:
            def __enter__(self_):
                ltm._collection = sentinel
                return self_

            def __exit__(self_, *a):
                return False

        with mock.patch.object(ltm, "_chroma_lock", SettingLock()):
            got = ltm._try_import_chroma()
        self.assertIs(got, sentinel)

    def test_embedder_fast_path_recheck_under_lock(self):
        # Drive line 193 similarly for the embedder construction lock.
        ltm._embedder = None
        sentinel = object()

        class SettingLock:
            def __enter__(self_):
                ltm._embedder = sentinel
                return self_

            def __exit__(self_, *a):
                return False

        with mock.patch.object(ltm, "_embedder_lock", SettingLock()):
            got = ltm._try_import_embedder()
        self.assertIs(got, sentinel)

    def test_embedder_cpu_load_failure_returns_none(self):
        # Drive lines 222-223: cuda unavailable (dev stays "cpu"), construction
        # raises → non-cuda branch prints and returns None.
        ltm._embedder = None

        class CpuFails:
            def __init__(self, model, device=None):
                raise RuntimeError("cpu load boom")

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = CpuFails
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                           "torch": fake_torch}):
            self.assertIsNone(ltm._try_import_embedder())

    def test_reflector_inner_loop_skips_already_marked_j(self):
        # Drive line 878 (`if ids[j] in to_delete: continue`). Need a fact marked
        # in an earlier i-row to reappear as ids[j] in a later, unmarked i-row.
        # Pairwise call order for ids=[A,B,C] is (A,B),(A,C),(B,C). Returning
        # sims [0.5, 0.95, 0.95]: (A,B) no-op, (A,C) marks C, then (B,C) sees C
        # already in to_delete → the guarded `continue` fires.
        self._install_fake_embedder()
        self._install_fake_chroma()
        ltm.ensure_loaded()
        # updated_at desc fixes scan order ids=[A,B,C]; created_at desc (A newest,
        # C oldest) makes the (A,C) near-dup drop C=ids[j] (the older).
        ltm._facts = {
            "A": {"id": "A", "text": "alpha text", "source": "", "tags": [],
                  "created_at": 10.0, "updated_at": 30.0},
            "B": {"id": "B", "text": "bravo text", "source": "", "tags": [],
                  "created_at": 5.0, "updated_at": 20.0},
            "C": {"id": "C", "text": "charlie text", "source": "", "tags": [],
                  "created_at": 1.0, "updated_at": 10.0},
        }
        sims = iter([0.5, 0.95, 0.95])
        with mock.patch.object(ltm, "_cosine_sim", lambda a, b: next(sims)):
            summary = ltm.reflect_and_consolidate()
        # C (ids[2]) is the near-dup of A that got marked; B and the survivor
        # remain. Exactly one removal despite two ≥-threshold sims (the second
        # short-circuits on the already-marked guard).
        self.assertNotIn("C", ltm._facts)
        self.assertEqual(summary["duplicates_removed"], 1)

    def test_reflector_inner_skip_and_older_is_ids_i(self):
        # Three identical-text facts so the pairwise scan marks one fact that
        # later appears as ids[j] (drives line 878 `if ids[j] in to_delete`).
        # Also arrange updated_at-desc but created_at so ids[i] is the OLDER of
        # a near-dup pair, driving line 887 `to_delete.add(a)`.
        self._install_fake_embedder()
        self._install_fake_chroma()
        ltm.ensure_loaded()    # mark loaded so reflect's ensure_loaded is a no-op
                               # (otherwise it reloads _facts from the empty file)
        # Identical text → every pair is a near-dup (cosine 1.0).
        # Sort key is updated_at desc → order is x(3), y(2), z(1) as ids[0..2].
        # Pair (x,y): created_at x=10 > y=5 → adds y (ids[j]).
        # Pair (x,z): created_at x=10 > z=1 → adds z.
        # Then make a 4th fact w whose updated_at is highest but created_at
        # lowest so for pair (w, x): created_at w=0 <= x=10 → adds w == ids[i].
        ltm._facts = {
            "w": {"id": "w", "text": "dup", "source": "", "tags": [],
                  "created_at": 0.0, "updated_at": 40.0},
            "x": {"id": "x", "text": "dup", "source": "", "tags": [],
                  "created_at": 10.0, "updated_at": 30.0},
            "y": {"id": "y", "text": "dup", "source": "", "tags": [],
                  "created_at": 5.0, "updated_at": 20.0},
            "z": {"id": "z", "text": "dup", "source": "", "tags": [],
                  "created_at": 1.0, "updated_at": 10.0},
        }
        summary = ltm.reflect_and_consolidate()
        # All four share text → three removed, one survives.
        self.assertEqual(len(ltm._facts), 1)
        self.assertGreaterEqual(summary["duplicates_removed"], 1)


# ──────────────────────────────────────────────────────────────────────────
#  end-to-end smoke through the fully-faked stack
# ──────────────────────────────────────────────────────────────────────────

class EndToEndTests(_LtmBase):
    def test_full_lifecycle_with_fakes(self):
        self._install_fake_embedder()
        self._install_fake_chroma()
        self._install_fake_bm25()
        ltm.ensure_loaded()
        fid = ltm.add_fact("User enjoys jazz", source="t", tags=["music"])
        self.assertTrue(ltm.update_fact(fid, "User enjoys jazz and blues"))
        ltm.record_turn("user", "play some blues")
        ltm.record_turn("assistant", "Queueing blues.")
        hits = ltm.retrieve_facts("what music does the user like", k=3)
        self.assertTrue(hits)
        self.assertEqual(ltm.get_working_window(5)[0]["text"], "play some blues")
        self.assertTrue(ltm.search_episodes("blues"))
        self.assertEqual(ltm.status()["facts"], 1)
        self.assertTrue(ltm.delete_fact(fid))
        self.assertEqual(ltm.list_facts(), [])


# ──────────────────────────────────────────────────────────────────────────
#  2026-07-08 bug-fix batch: encode-lock / no-embed-under-_lock / OOM recovery /
#  migration flag gating + reconcile / boot-time episode rotation
# ──────────────────────────────────────────────────────────────────────────

class EncodeLockAndOomTests(_LtmBase):
    """#15/#28 — every encode runs under _encode_lock; #27 — a CUDA/OOM encode
    failure drops the wedged model, frees VRAM and arms the reload cooldown."""

    def test_embed_holds_encode_lock_during_encode(self):
        # #28: the forward pass must be serialised behind _encode_lock, and the
        # lock must be released again afterwards.
        held = {}

        class LockSpyEmbedder:
            def encode(self, texts, **k):
                held["locked"] = ltm._encode_lock.locked()
                return FakeEncoded(_vec_for(t) for t in texts)

        with mock.patch.object(ltm, "_try_import_embedder",
                               lambda: LockSpyEmbedder()):
            out = ltm._embed(["x"])
        self.assertIsNotNone(out)
        self.assertTrue(held.get("locked"))          # held during encode
        self.assertFalse(ltm._encode_lock.locked())  # released after

    def test_embed_cuda_oom_resets_embedder_and_arms_cooldown(self):
        # #27: an OOM-looking encode error drops _embedder, best-effort frees
        # VRAM, and arms _embedder_failed_until so a later call rebuilds it.
        class OOMEmbedder:
            def encode(self, texts, **k):
                raise RuntimeError("CUDA out of memory")

        ltm._embedder = OOMEmbedder()
        ltm._embedder_failed_until = 0.0
        calls = {"empty": 0}
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            empty_cache=lambda: calls.__setitem__("empty", calls["empty"] + 1))
        with mock.patch.object(ltm, "_try_import_embedder",
                               lambda: ltm._embedder), \
             mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertIsNone(ltm._embed(["x"]))
        self.assertIsNone(ltm._embedder)                     # wedged handle dropped
        self.assertGreater(ltm._embedder_failed_until, 0.0)  # cooldown armed
        self.assertEqual(calls["empty"], 1)                  # VRAM freed best-effort

    def test_embed_non_cuda_error_does_not_reset(self):
        # A plain encode error (no cuda/oom signal) is swallowed WITHOUT nuking
        # the embedder — only genuine OOM warrants a rebuild.
        sentinel = FakeEmbedderRaises()          # raises "encode boom"
        ltm._embedder = sentinel
        ltm._embedder_failed_until = 0.0
        with mock.patch.object(ltm, "_try_import_embedder", lambda: sentinel):
            self.assertIsNone(ltm._embed(["x"]))
        self.assertIs(ltm._embedder, sentinel)               # NOT dropped
        self.assertEqual(ltm._embedder_failed_until, 0.0)    # no cooldown


class RetrieveLockDisciplineTests(_LtmBase):
    """#15/#29 — the query embed (a possibly multi-second cold load) must run
    OUTSIDE _lock so it can't stall record_turn and other callers."""
    def setUp(self):
        super().setUp()
        self.emb = self._install_fake_embedder()
        self.coll = self._install_fake_chroma()
        self._install_fake_bm25()
        ltm.ensure_loaded()

    def test_retrieve_does_not_embed_while_holding_lock(self):
        ltm.add_fact("some fact about cats")
        observed = {}
        real_embed = ltm._embed

        def spy(texts):
            # RLock._is_owned() → True only if THIS thread holds _lock.
            observed["lock_held"] = ltm._lock._is_owned()
            return real_embed(texts)

        with mock.patch.object(ltm, "_embed", side_effect=spy):
            ltm.retrieve_facts("cats", k=3)
        self.assertIn("lock_held", observed)     # embed actually ran
        self.assertFalse(observed["lock_held"])  # …but not under _lock


class MigrationFlagGatingTests(_LtmBase):
    """#16 — the migrated.flag is only written once every fact is confirmed into
    Chroma; a reconcile back-fill repairs facts missing from the dense index."""

    def test_migration_defers_flag_when_chroma_upsert_fails(self):
        # Chroma present but every upsert fails → flag NOT written (so a later
        # boot retries) while the JSON mirror is still populated.
        self._install_fake_chroma()                   # chroma_avail = True
        with open(ltm._LEGACY_BOBERT_MEMORY, "w", encoding="utf-8") as f:
            json.dump({"facts": ["legacy one", "legacy two"]}, f)
        with mock.patch.object(ltm, "_chroma_upsert", return_value=False):
            ltm.ensure_loaded()
        self.assertFalse(os.path.exists(ltm._MIGRATE_FLAG))   # deferred
        self.assertEqual(len(ltm._facts), 2)                  # mirror populated

    def test_migration_writes_flag_when_chroma_absent(self):
        # Chroma unavailable → JSON mirror is authoritative, so a failed upsert
        # must NOT block the flag (else we'd re-import every boot forever).
        self._force_no_deps()
        with open(ltm._LEGACY_BOBERT_MEMORY, "w", encoding="utf-8") as f:
            json.dump({"facts": ["legacy one"]}, f)
        ltm.ensure_loaded()
        self.assertTrue(os.path.exists(ltm._MIGRATE_FLAG))
        self.assertEqual(len(ltm._facts), 1)

    def test_reconcile_backfills_missing_chroma_facts(self):
        coll = FakeChromaCollectionWithGet()
        self._install_fake_embedder()
        self._install_fake_chroma(coll)
        ltm.ensure_loaded()
        # A fact that reached the mirror but never made it into Chroma.
        ltm._facts["m1"] = {"id": "m1", "text": "missing fact", "source": "",
                            "tags": [], "created_at": 1.0, "updated_at": 1.0}
        self.assertNotIn("m1", coll.store)
        with ltm._lock:
            n = ltm._reconcile_chroma_locked()
        self.assertEqual(n, 1)
        self.assertIn("m1", coll.store)             # back-filled into dense index

    def test_reconcile_noop_without_chroma(self):
        self._force_no_deps()
        ltm.ensure_loaded()
        ltm._facts["x"] = {"id": "x", "text": "t", "source": "", "tags": [],
                           "created_at": 1.0, "updated_at": 1.0}
        with ltm._lock:
            self.assertEqual(ltm._reconcile_chroma_locked(), 0)   # no chroma → 0


class BootTimeEpisodeRotationTests(_LtmBase):
    """#30 — rotation trims on ACTUAL line count at boot, so a box that restarts
    before the per-process append counter trips can't grow the log unbounded."""
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_boot_time_rotation_trims_oversized_log(self):
        os.makedirs(ltm._DATA_DIR, exist_ok=True)
        with mock.patch.object(ltm, "EPISODE_MAX_LINES", 5):
            with open(ltm._EPISODE_LOG, "w", encoding="utf-8") as f:
                for i in range(20):
                    f.write(json.dumps({"i": i}) + "\n")
            # Simulate a fresh boot: counter is 0, far below CHECK_EVERY, so the
            # in-append gate would NEVER fire — only the boot trim saves us.
            ltm._loaded = False
            ltm._writes_since_rotate = 0
            ltm.ensure_loaded()
        with open(ltm._EPISODE_LOG, encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        self.assertEqual(len(lines), 5)
        self.assertEqual([e["i"] for e in lines], [15, 16, 17, 18, 19])

    def test_rotate_episodes_locked_missing_file_is_noop(self):
        # No episode log yet → must not raise.
        with ltm._lock:
            ltm._rotate_episodes_locked()
        self.assertFalse(os.path.exists(ltm._EPISODE_LOG))


if __name__ == "__main__":
    unittest.main()
