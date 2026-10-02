"""Tests for the MEMORY_EMBED_MODEL switch in core.long_term_memory
(2026-10-02): an opt-in voyage-4-nano memory embedder with a safe re-index.

What they pin:
  * DEFAULT UNCHANGED -- with the shipped setting the embedder is built and
    called exactly as before (bge-small, no prompts, no extra kwargs), the
    legacy collection keeps its metadata, no manifest is written and no
    rebuild thread starts.
  * MISMATCH -> BACKGROUND REBUILD -- a different setting builds the new
    model's OWN collection from the stored fact texts, keeps the old index
    live until the new one is complete (facts written meanwhile are caught
    up), never holds the store lock while it embeds, then swaps, records it
    in embed_index.json and keeps the old index as a .bak. Nothing deleted.
  * MIXING IS IMPOSSIBLE -- an embedder for another profile is refused, a
    vector of the wrong dimension never reaches the index, a swap during a
    query drops the dense half, a collection stamped by another model is
    never used, and a builder model with the wrong dimension is rejected.
  * FALLBACK -- a model that cannot load leaves (or puts) the index on
    bge-small with ONE log line, and an unknown setting does the same.

Everything is FAKE: sentence_transformers, torch and chromadb are stand-ins
injected into sys.modules, so no model is downloaded or loaded and no real
Chroma store is opened; every path is repointed into a per-test temp dir,
and the rebuild thread is recorded, never started -- the tests run its body
synchronously. stdlib unittest + numpy (pure math) only.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

import core.config as _cfg
import core.long_term_memory as ltm
# Imported up front so the sys.modules patch below never drops them: the
# code under test imports both lazily.
from core import local_traffic as _local_traffic, log_filters as _log_filters

_PRELOADED = (_local_traffic, _log_filters)


_BGE = "BAAI/bge-small-en-v1.5"
_VOYAGE = "voyageai/voyage-4-nano"
_VOYAGE_COLL = ltm.LTM_COLLECTION + "__voyage-4-nano-512"
_QP = ltm._EMBED_PROFILES["voyage-4-nano"]["query_prefix"]
_DP = ltm._EMBED_PROFILES["voyage-4-nano"]["doc_prefix"]


# ──────────────────────────────────────────────────────────────────────────
#  FAKES
# ──────────────────────────────────────────────────────────────────────────

def _vec(model: str, text: str, dim: int):
    """Deterministic unit vector per (model, text). The voyage prompts are
    stripped first, so a query equal to a stored fact ranks it first under
    either model -- while the SAME text gets unrelated vectors from the two
    models, as real ones would."""
    for p in (_QP, _DP):
        if text.startswith(p):
            text = text[len(p):]
    seed = int.from_bytes(
        hashlib.sha1(f"{model}|{text}".encode("utf-8")).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


class FakeST:
    """Stands in for sentence_transformers.SentenceTransformer."""
    native_dim = {_BGE: 384, _VOYAGE: 2048}
    instances: list = []
    fail_models: set = set()
    dim_override: dict = {}

    def __init__(self, model, device=None, **kwargs):
        if model in FakeST.fail_models:
            raise RuntimeError(f"cannot load {model}")
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.calls: list = []
        self.hook = None
        FakeST.instances.append(self)

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        hook, self.hook = self.hook, None     # fires once
        if hook is not None:
            hook(self, texts)
        dim = (FakeST.dim_override.get(self.model)
               or self.kwargs.get("truncate_dim")
               or FakeST.native_dim[self.model])
        return np.stack([_vec(self.model, t, dim) for t in texts])

    @classmethod
    def of(cls, model):
        return [i for i in cls.instances if i.model == model]


class FakeColl:
    """In-memory Chroma collection. Like the real one, the first add fixes
    the dimension and a vector of another dimension raises."""

    def __init__(self, client, name, metadata):
        self._client = client
        self.name = name
        self.metadata = dict(metadata) if metadata else None
        self.store: dict = {}
        self.dim = None
        # Counted on ENTRY, before the dimension check below, so a test can
        # tell "never offered to the index" from "the index refused it".
        self.queries = 0
        self.add_calls = 0

    def _check(self, v):
        if self.dim is not None and len(v) != self.dim:
            raise ValueError(f"Collection expecting embedding with dimension "
                             f"of {self.dim}, got {len(v)}")

    def add(self, *, ids, embeddings, documents, metadatas):
        self.add_calls += 1
        for i, fid in enumerate(ids):
            v = [float(x) for x in embeddings[i]]
            self._check(v)
            self.dim = len(v)
            self.store[fid] = (v, documents[i], metadatas[i])

    def delete(self, *, ids=None):
        for fid in (ids or []):
            self.store.pop(fid, None)

    def query(self, *, query_embeddings, n_results, include=None):
        self.queries += 1
        q = np.asarray(query_embeddings[0], dtype="float64")
        self._check(q)
        scored = sorted(
            ((fid, 1.0 - float(np.dot(q, np.asarray(rec[0]))))
             for fid, rec in self.store.items()), key=lambda t: t[1])
        scored = scored[:n_results]
        return {"ids": [[f for f, _ in scored]],
                "distances": [[d for _, d in scored]],
                "metadatas": [[self.store[f][2] for f, _ in scored]]}

    def get(self, *, include=None, ids=None, limit=None):
        keys = list(self.store)
        out = {"ids": keys}
        if include and "documents" in include:
            out["documents"] = [self.store[k][1] for k in keys]
        return out

    def count(self):
        return len(self.store)

    def modify(self, name=None, metadata=None):
        if name is not None:
            if name in self._client.colls:
                raise ValueError(f"collection {name} already exists")
            del self._client.colls[self.name]
            self.name = name
            self._client.colls[name] = self
        if metadata is not None:
            self.metadata = dict(metadata)

    def dims(self):
        return {len(rec[0]) for rec in self.store.values()}

    def docs(self):
        return sorted(rec[1] for rec in self.store.values())


class FakeClient:
    def __init__(self):
        self.colls: dict = {}
        self.deleted: list = []

    def get_or_create_collection(self, name, metadata=None, **_):
        if name not in self.colls:
            self.colls[name] = FakeColl(self, name, metadata)
        return self.colls[name]

    def get_collection(self, name, **_):
        if name not in self.colls:
            raise KeyError(name)
        return self.colls[name]

    def list_collections(self, *_, **__):
        return [types.SimpleNamespace(name=n) for n in self.colls]

    def delete_collection(self, name):
        self.deleted.append(name)
        self.colls.pop(name, None)


class _FakeClock:
    """Stands in for the `time` module inside core.long_term_memory: a
    monotonic() the test moves by hand, everything else the real module."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class _RecordedThread:
    """threading.Thread stand-in: records the rebuild thread, never runs it."""
    made: list = []

    def __init__(self, target=None, name=None, daemon=None, **_):
        self.target, self.name, self.daemon = target, name, daemon
        self.started = False
        _RecordedThread.made.append(self)

    def start(self):
        self.started = True

    def is_alive(self):
        return False


# ──────────────────────────────────────────────────────────────────────────
#  BASE FIXTURE
# ──────────────────────────────────────────────────────────────────────────

_PATH_ATTRS = ("_DATA_DIR", "_CHROMA_DIR", "_FACTS_JSON", "_EPISODE_LOG",
               "_MIGRATE_FLAG", "_LEGACY_BOBERT_MEMORY")
_STATE_ATTRS = ("_chroma_client", "_collection", "_embedder",
                "_embedder_failed_until", "_bm25_index", "_bm25_corpus_ids",
                "_bm25_corpus", "_facts", "_working", "_loaded",
                "_turns_since_reflect", "_writes_since_rotate",
                "_reflector_llm", "_reflector_sink", "_index_profile_key",
                "_collection_name", "_index_dim", "_embedder_key",
                "_binding_gen", "_embed_fallback_from",
                "_embed_notes_printed", "_reindex_thread", "_reindex_state")


class _SwitchBase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        d = self._tmp.name
        saved = {a: getattr(ltm, a) for a in _PATH_ATTRS + _STATE_ATTRS}
        self.addCleanup(lambda: [setattr(ltm, a, v) for a, v in saved.items()])
        ltm._DATA_DIR = os.path.join(d, "ltm")
        ltm._CHROMA_DIR = os.path.join(ltm._DATA_DIR, "chroma")
        ltm._FACTS_JSON = os.path.join(ltm._DATA_DIR, "facts.json")
        ltm._EPISODE_LOG = os.path.join(ltm._DATA_DIR, "episodes.jsonl")
        ltm._MIGRATE_FLAG = os.path.join(ltm._DATA_DIR, "migrated.flag")
        ltm._LEGACY_BOBERT_MEMORY = os.path.join(d, "bobert_memory.json")
        self._fresh_process()
        ltm._facts = {}
        ltm._working = []
        ltm._reflector_llm = None
        ltm._reflector_sink = None
        # The persisted "disk": one FakeClient per Chroma path, so a
        # simulated restart reopens the same store.
        self.clients: dict = {}
        chromadb = types.ModuleType("chromadb")
        chromadb.PersistentClient = (
            lambda path: self.clients.setdefault(path, FakeClient()))
        st = types.ModuleType("sentence_transformers")
        st.SentenceTransformer = FakeST
        torch = types.ModuleType("torch")
        torch.float32 = "torch.float32"
        torch.cuda = types.SimpleNamespace(is_available=lambda: False,
                                           empty_cache=lambda: None)
        FakeST.instances = []
        FakeST.fail_models = set()
        FakeST.dim_override = {}
        _RecordedThread.made = []
        for p in (
            mock.patch.dict(sys.modules, {"chromadb": chromadb,
                                          "sentence_transformers": st,
                                          "torch": torch}),
            mock.patch.object(_cfg, "MEMORY_EMBED_MODEL", _BGE, create=True),
            mock.patch.object(_cfg, "LTM_EMBED_DEVICE", "cpu", create=True),
            mock.patch.object(ltm, "_try_import_bm25", lambda: False),
            mock.patch.object(ltm, "threading",
                              types.SimpleNamespace(Thread=_RecordedThread)),
            mock.patch.object(ltm, "_REINDEX_PAUSE_S", 0.0),
        ):
            p.start()
            self.addCleanup(p.stop)
        self._out = io.StringIO()
        redirect = contextlib.redirect_stdout(self._out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    # — helpers ————————————————————————————————————————————————————————
    def _fresh_process(self):
        """Module state as a fresh process has it (the store on disk and
        the fake Chroma "disk" survive)."""
        ltm._chroma_client = None
        ltm._collection = None
        ltm._embedder = None
        ltm._embedder_failed_until = 0.0
        ltm._bm25_index = None
        ltm._bm25_corpus_ids = []
        ltm._bm25_corpus = []
        ltm._loaded = False
        ltm._turns_since_reflect = 0
        ltm._writes_since_rotate = 0
        ltm._index_profile_key = ltm._DEFAULT_EMBED_PROFILE
        ltm._collection_name = ltm.LTM_COLLECTION
        ltm._index_dim = None
        ltm._embedder_key = ltm._DEFAULT_EMBED_PROFILE
        ltm._binding_gen = 0
        ltm._embed_fallback_from = None
        ltm._embed_notes_printed = set()
        ltm._reindex_thread = None
        ltm._reindex_state = {"state": "idle"}

    def restart(self):
        self._fresh_process()
        ltm._facts = {}
        ltm._working = []
        _RecordedThread.made = []

    def want(self, value):
        _cfg.MEMORY_EMBED_MODEL = value

    @property
    def client(self) -> FakeClient:
        return self.clients[ltm._CHROMA_DIR]

    def log(self) -> str:
        return self._out.getvalue()

    def manifest(self):
        path = os.path.join(ltm._DATA_DIR, ltm._EMBED_INDEX_FILE)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def seed_legacy(self, texts=("User has a dog", "User drinks coffee black",
                                 "User's sister lives in Ohio")):
        """A store as it exists today: facts embedded by bge-small into the
        legacy, unstamped collection."""
        ltm.ensure_loaded()
        ids = [ltm.add_fact(t, source="test") for t in texts]
        return ids

    def switch_and_build(self, value="voyage-4-nano"):
        self.want(value)
        ltm._reindex_worker()


# ──────────────────────────────────────────────────────────────────────────
#  default unchanged
# ──────────────────────────────────────────────────────────────────────────

class DefaultUnchangedTests(_SwitchBase):

    def test_shipped_default_is_todays_model(self):
        # _SHIPPED_* is captured before user_settings.json can override the
        # public constant, so this holds on a box that opted in.
        self.assertEqual(_cfg._SHIPPED_MEMORY_EMBED_MODEL, _BGE)
        self.assertEqual(ltm.LTM_EMBED_MODEL, _BGE)
        self.assertEqual(
            ltm._EMBED_PROFILE_ALIASES[_cfg._SHIPPED_MEMORY_EMBED_MODEL.lower()],
            ltm._DEFAULT_EMBED_PROFILE)
        prof = ltm._EMBED_PROFILES[ltm._DEFAULT_EMBED_PROFILE]
        self.assertEqual(prof["model"], _BGE)
        self.assertEqual(prof["collection"], ltm.LTM_COLLECTION)
        self.assertEqual((prof["query_prefix"], prof["doc_prefix"]), ("", ""))
        self.assertEqual(prof["load_kwargs"], {})
        self.assertFalse(prof["fp32"])

    def test_default_boot_and_use_are_exactly_as_before(self):
        self.seed_legacy()
        hits = ltm.retrieve_facts("User has a dog", k=3)
        # The embedder: SentenceTransformer(model, device=dev), nothing else.
        (emb,) = FakeST.instances
        self.assertEqual((emb.model, emb.device, emb.kwargs), (_BGE, "cpu", {}))
        # Raw texts in, raw query in: no prompts on the default model.
        self.assertEqual(emb.calls[-1], ["User has a dog"])
        self.assertIn(["User drinks coffee black"], emb.calls)
        self.assertEqual(hits[0]["text"], "User has a dog")
        # The legacy collection, with its historical metadata only.
        self.assertEqual(sorted(self.client.colls), [ltm.LTM_COLLECTION])
        coll = self.client.colls[ltm.LTM_COLLECTION]
        self.assertEqual(coll.metadata, {"hnsw:space": "cosine"})
        self.assertEqual(coll.dims(), {384})
        # No manifest, no rebuild thread, nothing dropped.
        self.assertIsNone(self.manifest())
        self.assertEqual(_RecordedThread.made, [])
        self.assertEqual(self.client.deleted, [])
        self.assertEqual(ltm._index_profile_key, ltm._DEFAULT_EMBED_PROFILE)

    def test_default_text_shaping_is_identity(self):
        self.assertEqual(ltm._query_text("what is my dog called"),
                         "what is my dog called")
        self.assertEqual(ltm._doc_text("User has a dog"), "User has a dog")
        self.assertTrue(ltm._dims_ok([0.0] * 7))     # no dimension recorded
        self.assertTrue(ltm._reflector_semantic_ok())

    def test_spellings_of_the_default_resolve_to_it(self):
        for v in (_BGE, "bge-small", "", "  BAAI/BGE-small-en-v1.5 "):
            self.want(v)
            self.assertEqual(ltm._desired_profile_key(),
                             ltm._DEFAULT_EMBED_PROFILE, v)
        self.assertNotIn("not a known memory embedder", self.log())


# ──────────────────────────────────────────────────────────────────────────
#  mismatch -> background rebuild
# ──────────────────────────────────────────────────────────────────────────

class RebuildTests(_SwitchBase):

    def test_switch_starts_the_rebuild_in_the_background(self):
        self.seed_legacy()
        self.restart()
        self.want("voyage-4-nano")
        ltm.ensure_loaded()
        (t,) = _RecordedThread.made
        self.assertIs(t.target, ltm._reindex_worker)
        self.assertEqual(t.name, "ltm-reindex")
        self.assertTrue(t.daemon)
        self.assertTrue(t.started)
        # ensure_loaded() returned without building anything: the live index
        # is still the legacy one and no voyage model was even constructed.
        self.assertEqual(ltm._index_profile_key, ltm._DEFAULT_EMBED_PROFILE)
        self.assertEqual(FakeST.of(_VOYAGE), [])

    def test_rebuild_swaps_in_a_new_index_and_keeps_the_old_as_bak(self):
        self.seed_legacy()
        old = self.client.colls[ltm.LTM_COLLECTION]
        old_store = dict(old.store)
        self.switch_and_build()
        # The research recipe, verbatim.
        (voy,) = FakeST.of(_VOYAGE)
        self.assertEqual(voy.device, "cpu")
        self.assertEqual(voy.kwargs, {
            "revision": "67fabc9bef010dabc5f6024aa1b1b6b93410426f",
            "trust_remote_code": True, "truncate_dim": 512,
            "model_kwargs": {"dtype": "torch.float32"}})
        built = [t for call in voy.calls for t in call]
        self.assertTrue(built and all(t.startswith(_DP) for t in built))
        # New, stamped collection with every fact at 512-d, raw documents.
        new = self.client.colls[_VOYAGE_COLL]
        self.assertEqual(new.metadata, {"hnsw:space": "cosine",
                                        "embed_model": _VOYAGE,
                                        "embed_dim": 512})
        self.assertEqual(new.dims(), {512})
        self.assertEqual(new.docs(), sorted(e["text"]
                                            for e in ltm._facts.values()))
        # The old index: renamed .bak, every vector untouched, not deleted.
        self.assertIs(self.client.colls[ltm.LTM_COLLECTION + ".bak"], old)
        self.assertEqual(old.store, old_store)
        self.assertNotIn(ltm.LTM_COLLECTION, self.client.colls)
        self.assertEqual(self.client.deleted, [])
        # Live binding + manifest.
        self.assertEqual(ltm._index_profile_key, "voyage-4-nano")
        self.assertIs(ltm._collection, new)
        self.assertIs(ltm._embedder, voy)
        man = self.manifest()
        self.assertEqual(man["active"]["profile"], "voyage-4-nano")
        self.assertEqual(man["active"]["collection"], _VOYAGE_COLL)
        self.assertEqual(man["active"]["dim"], 512)
        self.assertEqual(man["active"]["facts"], 3)
        self.assertEqual([(b["collection"], b["model"]) for b in man["backups"]],
                         [(ltm.LTM_COLLECTION + ".bak", _BGE)])
        self.assertIn("memory index rebuilt with voyageai/voyage-4-nano",
                      self.log())

    def test_after_the_swap_recall_uses_the_query_prompt(self):
        self.seed_legacy()
        self.switch_and_build()
        hits = ltm.retrieve_facts("User drinks coffee black", k=2)
        (voy,) = FakeST.of(_VOYAGE)
        self.assertEqual(voy.calls[-1], [_QP + "User drinks coffee black"])
        self.assertEqual(hits[0]["text"], "User drinks coffee black")
        fid = ltm.add_fact("User owns a 3D printer")
        self.assertEqual(voy.calls[-1], [_DP + "User owns a 3D printer"])
        self.assertEqual(len(ltm._collection.store[fid][0]), 512)

    def test_restart_binds_the_rebuilt_index_from_the_manifest(self):
        self.seed_legacy()
        self.switch_and_build()
        self.restart()
        ltm.ensure_loaded()
        self.assertEqual(ltm._index_profile_key, "voyage-4-nano")
        self.assertEqual(ltm._collection_name, _VOYAGE_COLL)
        self.assertEqual(ltm._index_dim, 512)
        self.assertEqual(_RecordedThread.made, [])   # nothing to rebuild
        hits = ltm.retrieve_facts("User has a dog", k=1)
        self.assertEqual(hits[0]["text"], "User has a dog")
        self.assertEqual(FakeST.of(_VOYAGE)[-1].calls[-1],
                         [_QP + "User has a dog"])
        self.assertEqual(FakeST.of(_BGE)[1:], [])    # bge never reloaded

    def test_old_index_serves_until_the_new_one_is_complete(self):
        self.seed_legacy()
        old = self.client.colls[ltm.LTM_COLLECTION]
        bge = FakeST.of(_BGE)[0]
        seen = {}

        def mid_build(model, texts):
            # (a) the builder is not holding the store lock while it embeds
            got = []

            def probe():
                ok = ltm._lock.acquire(timeout=2.0)
                got.append(ok)
                if ok:
                    ltm._lock.release()     # by the thread that holds it

            th = threading.Thread(target=probe, daemon=True)
            th.start()
            th.join(5.0)
            seen["lock_free"] = got == [True]
            # (b) a voice turn's recall still runs on the OLD index, with
            #     the OLD model and the raw query
            seen["hits"] = ltm.retrieve_facts("User has a dog", k=1)
            seen["query"] = bge.calls[-1]
            # (c) a fact learned now goes to the old index...
            seen["fid"] = ltm.add_fact("User likes green tea")
            seen["old_dims"] = old.dims()

        self.want("voyage-4-nano")
        real_load = ltm._load_profile_model

        def load_with_hook(key, dev):
            m = real_load(key, dev)
            m.hook = mid_build
            return m

        with mock.patch.object(ltm, "_load_profile_model", load_with_hook):
            ltm._reindex_worker()
        self.assertTrue(seen["lock_free"])
        self.assertEqual(seen["hits"][0]["text"], "User has a dog")
        self.assertEqual(seen["query"], ["User has a dog"])
        self.assertEqual(seen["old_dims"], {384})
        self.assertIn(seen["fid"], old.store)
        # ...and is caught up into the new index before the swap.
        new = self.client.colls[_VOYAGE_COLL]
        self.assertIn(seen["fid"], new.store)
        self.assertEqual(new.dims(), {512})
        self.assertIs(ltm._collection, new)

    def test_fact_removed_mid_build_is_not_in_the_new_index(self):
        ids = self.seed_legacy()
        doomed = ids[1]

        def mid_build(model, texts):
            ltm.delete_fact(doomed)

        real_load = ltm._load_profile_model

        def load_with_hook(key, dev):
            m = real_load(key, dev)
            m.hook = mid_build
            return m

        self.want("voyage-4-nano")
        with mock.patch.object(ltm, "_load_profile_model", load_with_hook):
            ltm._reindex_worker()
        new = self.client.colls[_VOYAGE_COLL]
        self.assertNotIn(doomed, new.store)
        self.assertEqual(set(new.store), set(ltm._facts))

    def test_an_interrupted_build_is_resumed_not_redone(self):
        ids = self.seed_legacy()
        part = self.client.get_or_create_collection(
            _VOYAGE_COLL, metadata=ltm._collection_metadata(
                "voyage-4-nano", stamp=True))
        for fid in ids[:2]:
            t = ltm._facts[fid]["text"]
            part.add(ids=[fid], embeddings=[_vec(_VOYAGE, t, 512)],
                     documents=[t], metadatas=[{}])
        self.switch_and_build()
        (voy,) = FakeST.of(_VOYAGE)
        self.assertEqual(voy.calls, [[_DP + ltm._facts[ids[2]]["text"]]])
        self.assertIs(ltm._collection, part)
        self.assertEqual(set(part.store), set(ids))

    def test_a_foreign_collection_under_the_target_name_is_moved_aside(self):
        self.seed_legacy()
        alien = self.client.get_or_create_collection(
            _VOYAGE_COLL, metadata={"hnsw:space": "cosine",
                                    "embed_model": "someone/else",
                                    "embed_dim": 512})
        alien.add(ids=["x"], embeddings=[[1.0] + [0.0] * 511],
                  documents=["kept"], metadatas=[{}])
        self.switch_and_build()
        self.assertIs(self.client.colls[_VOYAGE_COLL + ".bak"], alien)
        self.assertEqual(alien.docs(), ["kept"])
        self.assertIsNot(ltm._collection, alien)
        backups = {b["collection"]: b["model"]
                   for b in self.manifest()["backups"]}
        self.assertEqual(backups[_VOYAGE_COLL + ".bak"], "someone/else")
        self.assertEqual(self.client.deleted, [])

    def test_an_older_bak_is_never_overwritten(self):
        self.seed_legacy()
        older = self.client.get_or_create_collection(
            ltm.LTM_COLLECTION + ".bak", metadata={"hnsw:space": "cosine"})
        self.switch_and_build()
        self.assertIs(self.client.colls[ltm.LTM_COLLECTION + ".bak"], older)
        retired = [n for n in self.client.colls
                   if n.startswith(ltm.LTM_COLLECTION + ".bak-")]
        self.assertEqual(len(retired), 1)
        self.assertEqual(self.client.colls[retired[0]].dims(), {384})

    def test_out_of_time_build_keeps_the_old_index_live(self):
        self.seed_legacy()
        old = self.client.colls[ltm.LTM_COLLECTION]
        self.want("voyage-4-nano")
        with mock.patch.object(ltm, "_REINDEX_BUDGET_S", -1.0):
            self.assertFalse(ltm._run_reindex("voyage-4-nano"))
        self.assertIs(ltm._collection, old)
        self.assertEqual(ltm._index_profile_key, ltm._DEFAULT_EMBED_PROFILE)
        self.assertIsNone(self.manifest())
        self.assertIn(ltm.LTM_COLLECTION, self.client.colls)
        self.assertEqual(ltm._reindex_state["state"], "failed")
        self.assertEqual(ltm.retrieve_facts("User has a dog", k=1)[0]["text"],
                         "User has a dog")

    def test_the_build_budget_starts_after_the_model_load(self):
        # 2026-10-02 review: the first load downloads ~0.7 GB. A budget that
        # started before it was spent on the download, so a slow link threw
        # the finished download away at the first batch.
        self.seed_legacy()
        clock = _FakeClock()
        real_load = ltm._load_profile_model

        def slow_load(key, dev):
            clock.now += ltm._REINDEX_BUDGET_S + 60.0     # a long download
            return real_load(key, dev)

        self.want("voyage-4-nano")
        with mock.patch.object(ltm, "time", clock), \
                mock.patch.object(ltm, "_load_profile_model", slow_load):
            self.assertTrue(ltm._run_reindex("voyage-4-nano"))
        self.assertEqual(ltm._index_profile_key, "voyage-4-nano")
        self.assertEqual(ltm._reindex_state["state"], "done")

    def test_switching_back_rebuilds_bge_and_keeps_both_backups(self):
        self.seed_legacy()
        self.switch_and_build()
        self.switch_and_build(_BGE)
        self.assertEqual(ltm._index_profile_key, ltm._DEFAULT_EMBED_PROFILE)
        live = self.client.colls[ltm.LTM_COLLECTION]
        self.assertIs(ltm._collection, live)
        self.assertEqual(live.dims(), {384})
        self.assertEqual(live.metadata["embed_model"], _BGE)
        self.assertEqual(ltm._index_dim, 384)
        names = {b["collection"] for b in self.manifest()["backups"]}
        self.assertEqual(names, {ltm.LTM_COLLECTION + ".bak",
                                 _VOYAGE_COLL + ".bak"})
        self.assertEqual(self.client.deleted, [])

    def test_forget_since_also_purges_retired_indexes(self):
        ltm.ensure_loaded()
        keep = ltm.add_fact("User has a dog")
        gone = ltm.add_fact("User's door code is new")
        ltm._facts[keep]["created_at"] = 1_000_000.0
        ltm._facts[gone]["created_at"] = 2_000_000.0
        self.switch_and_build()
        bak = self.client.colls[ltm.LTM_COLLECTION + ".bak"]
        self.assertIn(gone, bak.store)
        ltm.forget_since(1_500_000.0)
        self.assertNotIn(gone, bak.store)
        self.assertNotIn(gone, ltm._collection.store)
        self.assertIn(keep, bak.store)

    def test_status_reports_the_binding(self):
        self.seed_legacy()
        self.want("voyage-4-nano")
        st = ltm.status()["embedder"]
        self.assertEqual(st["index_profile"], "bge-small")
        self.assertEqual(st["wanted_profile"], "voyage-4-nano")
        self.switch_and_build()
        st = ltm.status()["embedder"]
        self.assertEqual(st["index_profile"], "voyage-4-nano")
        self.assertEqual(st["rebuild"]["state"], "done")


# ──────────────────────────────────────────────────────────────────────────
#  mixing is impossible
# ──────────────────────────────────────────────────────────────────────────

class MixingImpossibleTests(_SwitchBase):

    def test_an_embedder_loaded_for_another_profile_is_refused(self):
        ltm._embedder = FakeST(_BGE, device="cpu")
        ltm._embedder_key = "bge-small"
        ltm._index_profile_key = "voyage-4-nano"
        self.assertIsNone(ltm._try_import_embedder())
        self.assertIsNone(ltm._embed(["anything"]))

    def test_a_wrong_dimension_vector_never_reaches_the_index(self):
        self.seed_legacy()
        self.switch_and_build()
        coll = ltm._collection
        before = dict(coll.store)
        adds = coll.add_calls
        with mock.patch.object(
                ltm, "_embed",
                lambda texts: np.stack([_vec(_BGE, t, 384) for t in texts])):
            self.assertFalse(ltm._chroma_upsert("f_x", "a fact", {}))
            hits = ltm.retrieve_facts("User has a dog", k=2)
        self.assertEqual(coll.store, before)
        self.assertEqual(coll.add_calls, adds)     # never offered to it
        self.assertEqual(coll.queries, 0)          # never searched with it
        self.assertTrue(all("score" not in h for h in hits))   # recency only

    def test_an_index_swap_during_a_query_drops_the_dense_half(self):
        self.seed_legacy()
        coll = ltm._collection
        real = ltm._embed

        def embed_while_swapping(texts):
            out = real(texts)
            ltm._binding_gen += 1          # the background swap lands here
            return out

        with mock.patch.object(ltm, "_embed", embed_while_swapping):
            hits = ltm.retrieve_facts("User has a dog", k=2)
        self.assertEqual(coll.queries, 0)
        self.assertTrue(all("score" not in h for h in hits))

    def test_a_load_that_straddles_a_swap_never_installs_the_stale_model(self):
        # 2026-10-02 review: a bge load in flight when the background swap
        # landed installed itself over the swapped-in voyage model; the key
        # check then refused it on every later call, so dense recall stayed
        # off for the rest of the session.
        self.seed_legacy()
        ltm._embedder = None         # dropped after an encode error / cold
        self.want("voyage-4-nano")
        st = sys.modules["sentence_transformers"]

        def load_while_the_swap_lands(model, device=None, **kwargs):
            if model == _BGE:
                ltm._reindex_worker()    # the rebuild completes mid-load
            return FakeST(model, device=device, **kwargs)

        with mock.patch.object(st, "SentenceTransformer",
                               load_while_the_swap_lands):
            got = ltm._try_import_embedder()
        (voy,) = FakeST.of(_VOYAGE)
        self.assertIs(got, voy)
        self.assertIs(ltm._try_import_embedder(), voy)
        self.assertEqual(ltm.retrieve_facts("User has a dog", k=1)[0]["text"],
                         "User has a dog")
        self.assertEqual(voy.calls[-1], [_QP + "User has a dog"])
        self.assertIn("discarded: the index moved to voyage-4-nano",
                      self.log())

    def test_a_collection_stamped_by_another_model_is_never_used(self):
        self.seed_legacy()
        self.switch_and_build()
        live = self.client.colls[_VOYAGE_COLL]
        live.metadata["embed_model"] = "someone/else"    # hand-edited store
        self.restart()
        ltm.ensure_loaded()
        self.assertIsNone(ltm._try_import_chroma())
        self.assertIsNone(ltm._try_import_embedder())
        self.assertEqual(live.queries, 0)
        self.assertIn("semantic recall is off", self.log())
        # ...and the background rebuild replaces it (moved aside, kept).
        (t,) = _RecordedThread.made
        ltm._reindex_worker()
        self.assertIs(self.client.colls[_VOYAGE_COLL + ".bak"], live)
        self.assertEqual(ltm._index_profile_key, "voyage-4-nano")
        self.assertIsNot(ltm._collection, live)
        self.assertEqual(ltm.retrieve_facts("User has a dog", k=1)[0]["text"],
                         "User has a dog")

    def test_a_builder_model_with_the_wrong_dimension_is_rejected(self):
        self.seed_legacy()
        old = self.client.colls[ltm.LTM_COLLECTION]
        FakeST.dim_override[_VOYAGE] = 384
        self.want("voyage-4-nano")
        self.assertFalse(ltm._run_reindex("voyage-4-nano"))
        self.assertIs(ltm._collection, old)
        self.assertIsNone(self.manifest())
        self.assertEqual(self.client.colls[_VOYAGE_COLL].count(), 0)


# ──────────────────────────────────────────────────────────────────────────
#  fallback
# ──────────────────────────────────────────────────────────────────────────

class FallbackTests(_SwitchBase):

    def _fallback_lines(self):
        return [ln for ln in self.log().splitlines()
                if "falling back to " + _BGE in ln]

    def test_unloadable_model_falls_back_with_one_clear_line(self):
        self.seed_legacy()
        old = self.client.colls[ltm.LTM_COLLECTION]
        FakeST.fail_models.add(_VOYAGE)
        self.want("voyage-4-nano")
        ltm._reindex_worker()
        ltm._reindex_worker()                   # a second attempt: silent
        (line,) = self._fallback_lines()
        self.assertIn(_VOYAGE, line)
        self.assertIn("cannot load", line)
        self.assertEqual(ltm._desired_profile_key(), ltm._DEFAULT_EMBED_PROFILE)
        self.assertIs(ltm._collection, old)
        self.assertIsNone(self.manifest())
        self.assertEqual(ltm.retrieve_facts("User has a dog", k=1)[0]["text"],
                         "User has a dog")

    def test_voyage_index_unloadable_at_boot_moves_back_to_bge(self):
        self.seed_legacy()
        self.switch_and_build()
        voyage_coll = ltm._collection
        self.restart()
        FakeST.fail_models.add(_VOYAGE)
        ltm.ensure_loaded()
        # Never a bge stand-in on the voyage index.
        self.assertIsNone(ltm._try_import_embedder())
        self.assertEqual(FakeST.of(_BGE)[1:], [])
        self.assertEqual(len(self._fallback_lines()), 1)
        (t,) = _RecordedThread.made
        self.assertIs(t.target, ltm._reindex_worker)
        ltm._reindex_worker()
        self.assertEqual(ltm._index_profile_key, ltm._DEFAULT_EMBED_PROFILE)
        live = self.client.colls[ltm.LTM_COLLECTION]
        self.assertIs(ltm._collection, live)
        self.assertEqual(live.dims(), {384})
        self.assertIs(self.client.colls[_VOYAGE_COLL + ".bak"], voyage_coll)
        self.assertEqual(ltm.retrieve_facts("User has a dog", k=1)[0]["text"],
                         "User has a dog")
        self.assertEqual(len(self._fallback_lines()), 1)

    def test_unknown_setting_uses_the_default_with_one_line(self):
        self.seed_legacy()
        self.want("voyage-9-mega")
        self.assertEqual(ltm._desired_profile_key(), ltm._DEFAULT_EMBED_PROFILE)
        self.assertEqual(ltm._desired_profile_key(), ltm._DEFAULT_EMBED_PROFILE)
        self.assertFalse(ltm._maybe_start_reindex())
        lines = [ln for ln in self.log().splitlines()
                 if "not a known memory embedder" in ln]
        self.assertEqual(len(lines), 1)
        self.assertEqual(_RecordedThread.made, [])


# ──────────────────────────────────────────────────────────────────────────
#  reflector on an uncalibrated model
# ──────────────────────────────────────────────────────────────────────────

class ReflectorTests(_SwitchBase):

    def test_uncalibrated_model_keeps_near_duplicates(self):
        self.seed_legacy(("User has a dog", "User has a dog named Rex",
                          "User drinks tea"))
        self.switch_and_build()
        with mock.patch.object(ltm, "_cosine_sim", lambda a, b: 0.99):
            summary = ltm.reflect_and_consolidate()
        self.assertEqual(summary["duplicates_removed"], 0)
        self.assertEqual(len(ltm._facts), 3)
        self.assertIn("similarity passes are off", self.log())

    def test_calibrated_default_still_dedupes(self):
        self.seed_legacy(("User has a dog", "User has a dog named Rex"))
        with mock.patch.object(ltm, "_cosine_sim", lambda a, b: 0.99):
            summary = ltm.reflect_and_consolidate()
        self.assertEqual(summary["duplicates_removed"], 1)


if __name__ == "__main__":
    unittest.main()
