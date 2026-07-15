"""One-time backfill: push the 120 structured facts (bobert_memory.json, store A)
into the SEMANTIC LTM store (chroma + facts.json + BM25), which had been frozen
at 23 since the 2026-05-28 migration. add_fact embeds (bge-small) + upserts with
exact-text dedupe, so the 23 already present are skipped — NON-DESTRUCTIVE.

RUN WITH THE LIVE JARVIS STOPPED (two processes racing the same chroma sqlite +
the embedder can corrupt/lock). JARVIS_STAGING must be unset (writes the real
store). Back up data/long_term_memory/ first (done by the caller).
"""
import io, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import long_term_memory as ltm  # noqa: E402

MEM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "bobert_memory.json")


def main() -> int:
    ltm.ensure_loaded()
    before = len(ltm.list_facts())
    print(f"[backfill] semantic store BEFORE: {before} facts")
    # bobert_memory.json has non-cp1252 bytes → force UTF-8.
    mem = json.load(io.open(MEM, encoding="utf-8"))
    submitted = 0
    for f in mem.get("facts", []):
        if isinstance(f, str) and f.strip():
            try:
                ltm.add_fact(f.strip(), source="bobert_memory_backfill",
                             tags=["backfill"])
                submitted += 1
            except Exception as e:
                print(f"  [backfill] fact failed ({e}): {f[:60]!r}")
    for p in mem.get("projects", []):
        if isinstance(p, str) and p.strip():
            try:
                ltm.add_fact(p.strip(), source="bobert_memory_backfill",
                             tags=["backfill", "project"])
                submitted += 1
            except Exception as e:
                print(f"  [backfill] project failed ({e}): {p[:60]!r}")
    after = len(ltm.list_facts())
    print(f"[backfill] submitted {submitted} items; semantic store AFTER: {after} "
          f"facts (+{after - before} new after dedupe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
