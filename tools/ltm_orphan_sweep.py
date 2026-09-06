#!/usr/bin/env python3
"""Reconcile the JARVIS semantic VECTOR store down to the fact store.

WHY THIS EXISTS (2026-09-06)
============================
``core.long_term_memory`` reconciles in exactly ONE direction. ``ensure_loaded()``
calls ``_reconcile_chroma_locked()``, which re-upserts facts that are present in
``facts.json`` but missing from Chroma. There is no sweep the other way, so a
row that is present in CHROMA but absent from ``facts.json`` — an ORPHAN — is
never noticed and never removed.

That asymmetry is what let a fabricated belief survive its own purge. The five
synthetic "test utterance" facts (the piano pair, the coffee/Rex/Denver batch)
were removed from ``facts.json`` by hand, which is the store a human reads and
audits — but RAG recalls from Chroma, so the belief kept being retrieved and
kept being restated. A fact store that looks clean while the index still answers
with the deleted belief is worse than one that is visibly dirty.

Deleting an orphan is safe in a way that deleting a *fact* is not: an orphan is
defined mechanically (in the index, not in the store of record), so removing it
cannot lose a fact the owner actually stated — the fact store is the authority
and it already does not have it. Nothing here decides what is true.

Every delete goes through chromadb's OWN collection API (``collection.delete(
ids=...)``) — the same call ``core.long_term_memory._chroma_delete()`` makes.
Chroma keeps an embedding row, a metadata table, an fts index, a content shadow
table and an append-only write queue that must stay mutually consistent; raw SQL
DELETEs against those tables corrupt recall in ways that only surface later, as
a query returning garbage.

Usage
-----
    python tools/ltm_orphan_sweep.py            # report only (default, safe)
    python tools/ltm_orphan_sweep.py --apply    # delete the orphans

Stop JARVIS before ``--apply``: the store is live and the running process both
reads and writes it.
"""
from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):  # direct `python tools/ltm_orphan_sweep.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import long_term_memory as ltm


def _collection_contents(coll) -> dict[str, str]:
    """{id: document} for everything in the collection.

    Chroma's ``get()`` honours ``include``; the in-memory fakes used by the test
    suite return ids only. Fall back so this works against both.
    """
    try:
        got = coll.get(include=["documents"])
    except Exception:
        got = coll.get(include=[])
    ids = list((got or {}).get("ids") or [])
    docs = list((got or {}).get("documents") or [])
    if len(docs) != len(ids):
        docs = [""] * len(ids)
    return dict(zip(ids, docs))


def find_orphans(coll=None) -> list[tuple[str, str]]:
    """Rows present in the vector store but absent from the fact store.

    Returns [(fact_id, document_text), ...]. Read-only.
    """
    ltm.ensure_loaded()
    if coll is None:
        coll = ltm._try_import_chroma()
    if coll is None:
        return []
    contents = _collection_contents(coll)
    with ltm._lock:
        live = set(ltm._facts.keys())
    return [(fid, doc) for fid, doc in contents.items() if fid not in live]


def find_missing(coll=None) -> list[str]:
    """Facts present in the fact store but absent from the vector store.

    The direction ``ensure_loaded()`` already repairs; reported for symmetry so
    a caller can see the store is genuinely in sync rather than assuming it.
    """
    ltm.ensure_loaded()
    if coll is None:
        coll = ltm._try_import_chroma()
    if coll is None:
        return []
    contents = _collection_contents(coll)
    with ltm._lock:
        live = set(ltm._facts.keys())
    return sorted(live - set(contents))


def purge_orphans(coll=None, *, expect: set[str] | None = None) -> list[tuple[str, str]]:
    """Delete every orphan through the chromadb collection API.

    ``expect`` optionally pins the exact set of orphan DOCUMENT TEXTS the caller
    intends to remove. If the store does not match, nothing is deleted and
    ValueError is raised — a purge that quietly removes more than it was asked
    to is how a real memory gets lost.

    Returns the list of removed (id, document) pairs.
    """
    if coll is None:
        coll = ltm._try_import_chroma()
    if coll is None:
        return []
    orphans = find_orphans(coll)
    if not orphans:
        return []
    if expect is not None:
        found = {doc for _, doc in orphans}
        if found != expect:
            raise ValueError(
                "orphan set does not match the approved batch; refusing to "
                f"delete. unexpected={sorted(found - expect)!r} "
                f"absent={sorted(expect - found)!r}")
    coll.delete(ids=[fid for fid, _ in orphans])
    return orphans


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="actually delete the orphans (default: report only)")
    args = ap.parse_args(argv)

    coll = ltm._try_import_chroma()
    if coll is None:
        print("chromadb unavailable — no vector store to sweep.")
        return 0

    orphans = find_orphans(coll)
    missing = find_missing(coll)
    print(f"vector rows : {len(_collection_contents(coll))}")
    with ltm._lock:
        print(f"facts       : {len(ltm._facts)}")
    print(f"orphans     : {len(orphans)}  (in the index, not in the fact store)")
    for fid, doc in orphans:
        print(f"   {fid}  ->  {doc!r}")
    print(f"missing     : {len(missing)}  (in the fact store, not in the index)")

    if not orphans:
        print("\nin sync — nothing to do.")
        return 0
    if not args.apply:
        print("\nreport only. re-run with --apply to delete (stop JARVIS first).")
        return 1
    removed = purge_orphans(coll)
    print(f"\ndeleted {len(removed)} orphan row(s).")
    print(f"orphans remaining: {len(find_orphans(coll))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
