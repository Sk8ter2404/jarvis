"""
command_autocorrect.py — fuzzy-match middleware for unrecognised action names.

When the LLM emits `[ACTION: enable_ambient_learning_mode]` and the dispatcher
fails to find that key in ACTIONS, we'd rather silently route it to the closest
registered action than spit back "unknown action" and force the user to rephrase.

The scorer blends three signals:
  * Levenshtein-based string similarity (catches typos and minor extras)
  * Token-level overlap weighted by token-pair Levenshtein
    (catches reorderings and "enable_X" / "start_X" / "X_mode" wrappers)
  * Embedding cosine similarity from Ollama's /api/embeddings endpoint
    (catches semantic equivalence like 'see_screen' ↔ 'look_at_display')

When embeddings are available the weights are 0.4125 Lev + 0.3375 tok + 0.25 emb.
When the Ollama endpoint is down the embedding signal is dropped and the
remaining two signals are renormalised to the original 0.55 / 0.45 split.

A small phonetic/substring boost lets pairs like 'ambient' ↔ 'ambiant' or
'enable_ambient_learning_mode' ↔ 'ambient_listening' clear the 0.75 floor
without false-positiving on unrelated short names.

The dispatcher also wants top-2 candidates so it can ask 'did you mean X or Y'
when two distinct actions both clear the threshold within `AMBIGUITY_GAP`.
Use `autocorrect_command_choice()` for that flow; `autocorrect_command()` keeps
the legacy single-best-or-None contract.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from typing import Iterable


_FILLER_TOKENS = frozenset({
    "the", "a", "an", "please", "now", "mode", "enable", "disable",
    "start", "stop", "begin", "end", "turn", "on", "off", "to",
    "into", "go", "make", "be", "is", "it",
})


# ──────────────────────────────────────────────────────────────────────────
#  Embedding-similarity layer (Ollama /api/embeddings)
# ──────────────────────────────────────────────────────────────────────────
# All network state is module-global so the dispatcher's per-utterance call
# doesn't pay the latency of a fresh probe each time. Candidate embeddings
# are cached by name forever (the action registry only grows during a run);
# the unknown-side cache is a tiny LRU keyed by the raw string. When the
# endpoint fails we mark it dead for a cool-off so subsequent dispatches
# fall through to the Levenshtein-only path immediately.

EMBED_ENDPOINT = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL    = "nomic-embed-text"
EMBED_TIMEOUT  = 1.5     # seconds — keep dispatch latency capped
EMBED_COOLDOWN = 60.0    # seconds — how long we treat Ollama as dead
EMBED_WEIGHT   = 0.25    # blend weight when embeddings are available

_embed_lock           = threading.Lock()
_embed_cache: dict[str, list[float]] = {}
_embed_dead_until: float = 0.0
_embed_disabled: bool   = False   # explicit user opt-out


def disable_embeddings() -> None:
    """Force the embedding pathway off (e.g. in unit tests). Reversible."""
    global _embed_disabled
    _embed_disabled = True


def enable_embeddings() -> None:
    global _embed_disabled, _embed_dead_until
    _embed_disabled = False
    _embed_dead_until = 0.0


def _embeddings_available() -> bool:
    if _embed_disabled:
        return False
    if _embed_dead_until and time.time() < _embed_dead_until:
        return False
    return True


def _mark_embeddings_dead(reason: str | None = None) -> None:
    global _embed_dead_until
    _embed_dead_until = time.time() + EMBED_COOLDOWN
    if reason:
        # One log line per cooldown window — the dispatcher prints its own
        # autocorrect lines so we stay terse here.
        print(f"  [autocorrect] embeddings unavailable ({reason}); "
              f"falling back to Levenshtein for {EMBED_COOLDOWN:.0f}s")


def _fetch_embedding(text: str) -> list[float] | None:
    """One POST to /api/embeddings. Returns None and trips the cool-off on
    any failure so callers don't have to know about HTTP."""
    body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        EMBED_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # RESIDENCY GATE (2026-07-21). This layer is a NICE-TO-HAVE that blends a
    # 0.25-weight embedding score into a Levenshtein match, and it must never
    # cost more than it is worth. JARVIS persists OLLAMA_MAX_LOADED_MODELS=1 so
    # Ollama EVICTS rather than co-loads, which means a request naming
    # nomic-embed-text while the 16 GB voice brain is resident evicts the brain
    # and cold-loads the embedder. EMBED_TIMEOUT is 1.5 s, which cannot cover
    # that load — so this path ALWAYS timed out (the similarity layer was dead
    # in production, permanently degraded to Levenshtein-only) AND abandoning
    # the response did NOT cancel the swap the server had already begun. Net
    # effect: a feature that never worked, charging a brain reload every
    # EMBED_COOLDOWN seconds. Same class as the RAG boot probe and the
    # ask_vision num_ctx bug — see core/ollama_opts.
    #
    # Ask first, fire second: if the embedder is not already loaded, skip and
    # let Levenshtein handle it. On a box with room to co-load (or with the
    # embedder genuinely resident) the layer works exactly as before.
    try:
        from core.ollama_opts import model_resident
        if not model_resident(EMBED_MODEL):
            _mark_embeddings_dead(f"{EMBED_MODEL} not resident; refusing to "
                                  f"evict the voice brain for autocorrect")
            return None
    except Exception:
        # Never let the guard itself break dispatch — fall through to the
        # original behaviour if core.ollama_opts is unavailable.
        pass
    try:
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        _mark_embeddings_dead(repr(e))
        return None
    except Exception as e:  # noqa: BLE001 — defensive against shape changes
        _mark_embeddings_dead(repr(e))
        return None
    emb = payload.get("embedding") or []
    if not emb:
        _mark_embeddings_dead("empty embedding")
        return None
    return [float(x) for x in emb]


def _embed(text: str) -> list[float] | None:
    """Cache-aware single-text embedding. Returns None when unavailable."""
    if not _embeddings_available():
        return None
    key = _normalise(text)
    with _embed_lock:
        cached = _embed_cache.get(key)
    if cached is not None:
        return cached
    vec = _fetch_embedding(key)
    if vec is None:
        return None
    # Pre-normalise so the cosine reduces to a dot product downstream.
    vec = _l2_normalise(vec)
    with _embed_lock:
        _embed_cache[key] = vec
    return vec


def _l2_normalise(vec: list[float]) -> list[float]:
    s = 0.0
    for x in vec:
        s += x * x
    if s <= 0.0:
        return vec
    inv = s ** -0.5
    return [x * inv for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
    # Vectors are pre-normalised; clip to [0, 1] (cosine of identical
    # texts is 1.0, near-orthogonal is ~0; negative values become 0
    # so the blend stays well-defined).
    if dot < 0.0:
        return 0.0
    if dot > 1.0:
        return 1.0
    return dot


def _embedding_similarity(unknown: str, candidate: str) -> float | None:
    """Cosine between unknown and candidate. None when the layer is down."""
    u = _embed(unknown)
    if u is None:
        return None
    c = _embed(candidate)
    if c is None:
        return None
    return _cosine(u, c)


# ──────────────────────────────────────────────────────────────────────────
#  Lexical-similarity layer (Levenshtein + token overlap)
# ──────────────────────────────────────────────────────────────────────────


def _levenshtein(a: str, b: str) -> int:
    """Classic DP edit distance — insertions, deletions, substitutions cost 1."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]


def _lev_ratio(a: str, b: str) -> float:
    """Edit-distance normalised to [0,1]. 1.0 = identical, 0.0 = fully different."""
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - (_levenshtein(a, b) / m)


def _normalise(name: str) -> str:
    """Lowercase, swap separators to underscores, collapse runs."""
    out = name.lower().strip()
    for ch in (" ", "-", ".", "/"):
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _tokens(name: str) -> list[str]:
    """Split a normalised name into meaningful tokens, dropping filler words."""
    raw = [t for t in _normalise(name).split("_") if t]
    kept = [t for t in raw if t not in _FILLER_TOKENS]
    # If everything filtered out (e.g. "turn_on_mode"), keep the originals so
    # we still have something to match on.
    return kept or raw


def _token_similarity(a: str, b: str) -> float:
    """
    Token-level semantic score.

    For each token in the shorter list, find its best Levenshtein-ratio match
    among the longer list. Average the best-matches and apply a small penalty
    when the lists differ in length so 'ambient' alone doesn't score 1.0
    against 'ambient_listen_start'.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    best_scores: list[float] = []
    for s in short:
        best = 0.0
        for l in long_:
            r = _lev_ratio(s, l)
            if r > best:
                best = r
        best_scores.append(best)
    avg = sum(best_scores) / len(best_scores)
    # Length-mismatch penalty: 1.0 when equal, gracefully decays.
    coverage = len(short) / len(long_)
    return avg * (0.5 + 0.5 * coverage)


def _phonetic_boost(a: str, b: str) -> float:
    """
    Small additive nudge for cases the edit-distance and token scores both
    under-rate. Captures: shared prefix, one name contained in the other, and
    high token-overlap when the strings have wildly different lengths.
    """
    if not a or not b:
        return 0.0
    boost = 0.0
    # Shared prefix is a strong phonetic signal ("ambien" matches both
    # 'ambient' and 'ambiant').
    prefix = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix += 1
        else:
            break
    if prefix >= 4:
        boost += min(0.10, prefix / max(len(a), len(b)) * 0.15)
    # Substring containment — 'ambient' fully inside 'ambient_listen'.
    if a in b or b in a:
        boost += 0.05
    # Shared meaningful tokens — 'ambient' shows up in both token lists.
    shared = set(_tokens(a)) & set(_tokens(b))
    if shared:
        boost += min(0.08, 0.04 * len(shared))
    return min(0.15, boost)


def _lexical_score(unknown_norm: str, candidate_norm: str) -> float:
    """Original lexical-only signal — Lev + token similarity + phonetic boost.

    Returned unblended so the choice layer can re-weight when an embedding
    component is also available.
    """
    lev = _lev_ratio(unknown_norm, candidate_norm)
    tok = _token_similarity(unknown_norm, candidate_norm)
    u_core = "_".join(_tokens(unknown_norm))
    c_core = "_".join(_tokens(candidate_norm))
    lev_core = _lev_ratio(u_core, c_core) if u_core and c_core else lev
    lev_best = max(lev, lev_core)
    # 0.55 / 0.45 split between Lev-best and token similarity (matches the
    # pre-embedding-era blend exactly so existing callers see identical
    # scores when embeddings are off).
    base = 0.55 * lev_best + 0.45 * tok
    boosted = base + _phonetic_boost(unknown_norm, candidate_norm)
    return min(1.0, boosted)


def _score(unknown: str, candidate: str,
           *, use_embeddings: bool = True) -> float:
    """Combined confidence score in [0, 1]."""
    u = _normalise(unknown)
    c = _normalise(candidate)
    if u == c:
        return 1.0
    lex = _lexical_score(u, c)
    emb = _embedding_similarity(unknown, candidate) if use_embeddings else None
    if emb is None:
        return lex
    # 0.25 weight on the embedding signal; the remaining 0.75 keeps the
    # original 0.55 / 0.45 internal split inside the lexical bucket so
    # adding embeddings can only slow boundary cases down a little, never
    # leapfrog an obviously-wrong candidate.
    blended = (1.0 - EMBED_WEIGHT) * lex + EMBED_WEIGHT * emb
    return min(1.0, blended)


# ──────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────


# When the dispatcher passes a large ACTIONS dict (~300 entries) the naive
# "score every candidate with embeddings" loop would issue one network round-
# trip per candidate. That's fine when Ollama is down (we fast-fail via the
# cooldown), but an alive-but-slow Ollama could stall the dispatcher for many
# seconds. We pre-rank by lexical score (cheap, in-process) and only blend
# the embedding signal for the top RERANK_FLOOR candidates, bounding worst-
# case fetches per autocorrect call regardless of ACTIONS size.
RERANK_FLOOR = 8


def _rank_candidates(
    unknown: str,
    registered: Iterable[str],
    *,
    use_embeddings: bool,
    rerank_top: int,
) -> list[tuple[str, float]]:
    """Lexical pre-pass + bounded embedding rerank. Returns all candidates
    sorted by (possibly blended) score descending — callers slice as needed.
    """
    u_norm = _normalise(unknown)
    lexical: list[tuple[str, float]] = []
    for cand in registered:
        if not cand:
            continue
        c_norm = _normalise(cand)
        if u_norm == c_norm:
            lex = 1.0
        else:
            lex = _lexical_score(u_norm, c_norm)
        lexical.append((cand, lex))
    lexical.sort(key=lambda kv: kv[1], reverse=True)
    if not use_embeddings or not _embeddings_available() or not lexical:
        return lexical
    cutoff = max(rerank_top, 1)
    blended: list[tuple[str, float]] = []
    for cand, lex in lexical[:cutoff]:
        if lex >= 1.0:
            blended.append((cand, lex))
            continue
        emb = _embedding_similarity(unknown, cand)
        if emb is None:
            # Embeddings just went dead mid-batch; lexical score stands and
            # the remaining tail keeps its lexical-only scores below.
            blended.append((cand, lex))
            continue
        score = min(1.0, (1.0 - EMBED_WEIGHT) * lex + EMBED_WEIGHT * emb)
        blended.append((cand, score))
    # Tail keeps its lexical-only scores. Re-sort the union so a rerank that
    # demoted a high-lexical candidate can drop below an untouched one.
    merged = blended + lexical[cutoff:]
    merged.sort(key=lambda kv: kv[1], reverse=True)
    return merged


def autocorrect_command(
    unknown: str,
    registered: Iterable[str],
    threshold: float = 0.75,
    *,
    use_embeddings: bool = True,
) -> tuple[str | None, float]:
    """
    Find the best-matching registered action for an unrecognised command.

    Returns (best_match, confidence). If no candidate clears the threshold,
    returns (None, best_seen_confidence) — callers can log the near-miss
    without acting on it.
    """
    if not unknown:
        return (None, 0.0)
    ranked = _rank_candidates(
        unknown, registered,
        use_embeddings=use_embeddings,
        rerank_top=RERANK_FLOOR,
    )
    if not ranked:
        return (None, 0.0)
    best_name, best_score = ranked[0]
    if best_score >= threshold:
        return (best_name, best_score)
    return (None, best_score)


def autocorrect_command_topk(
    unknown: str,
    registered: Iterable[str],
    k: int = 2,
    *,
    use_embeddings: bool = True,
) -> list[tuple[str, float]]:
    """Return the top-k (name, score) pairs sorted by score descending.

    Does NOT apply a threshold — callers can inspect the gap between #1
    and #2 to decide whether the match is confident, ambiguous, or absent.
    """
    if not unknown or k <= 0:
        return []
    ranked = _rank_candidates(
        unknown, registered,
        use_embeddings=use_embeddings,
        rerank_top=max(RERANK_FLOOR, k * 3),
    )
    return ranked[:k]


# Result dict returned by autocorrect_command_choice. Documented here so the
# dispatcher contract is in one place:
#   status:     "silent"     → top1 ≥ threshold and clearly beats top2; just route
#               "ambiguous"  → top1 ≥ threshold AND top2 ≥ threshold AND
#                              gap (top1 - top2) ≤ ambiguity_gap; ask the user
#               "none"       → top1 < threshold; treat as unknown action
#   primary:    (name, score) of the best candidate (always present when
#               status != "none"; for "none" still set to the best-seen so
#               callers can log it)
#   secondary:  (name, score) of the runner-up; None when there isn't one
#               that matters (only populated for "ambiguous")
def autocorrect_command_choice(
    unknown: str,
    registered: Iterable[str],
    threshold: float = 0.75,
    ambiguity_gap: float = 0.10,
    *,
    use_embeddings: bool = True,
) -> dict:
    """Pick between silent route, disambiguation prompt, and no-match."""
    top = autocorrect_command_topk(
        unknown, registered, k=2, use_embeddings=use_embeddings,
    )
    if not top:
        return {"status": "none", "primary": None, "secondary": None}
    top1 = top[0]
    top2 = top[1] if len(top) > 1 else None
    if top1[1] < threshold:
        return {"status": "none", "primary": top1, "secondary": None}
    if (top2 is not None
            and top2[1] >= threshold
            and (top1[1] - top2[1]) <= ambiguity_gap
            and top2[0] != top1[0]):
        return {"status": "ambiguous", "primary": top1, "secondary": top2}
    return {"status": "silent", "primary": top1, "secondary": None}


# Convenience alias matching the planner's call name.
autocorrect = autocorrect_command


if __name__ == "__main__":
    # Quick sanity probe — run `python command_autocorrect.py` to eyeball
    # scoring on the canonical examples from jarvis_todo.md. Embeddings
    # are forced off so this works without a running Ollama.
    disable_embeddings()
    # All 10 Task #66 ambient-mode aliases (ambient_mode, ambient_mode_on/off,
    # silent_learning, silent_learning_on/off, ambient_listening,
    # start/stop_eavesdropping, chappie_mode) plus the skill-registered
    # ambient_listen_start/stop pair the dispatcher routes through. Keeping
    # them all in the sample list ensures the autocorrect probe exercises the
    # ambiguity branch on close variants like "ambient" / "ambient_listening".
    sample_actions = [
        "ambient_mode", "ambient_mode_on", "ambient_mode_off",
        "silent_learning", "silent_learning_on", "silent_learning_off",
        "ambient_listening", "start_eavesdropping", "stop_eavesdropping",
        "chappie_mode",
        "ambient_listen_start", "ambient_listen_stop",
        "screenshot", "open_url", "launch_app", "shutdown_jarvis",
        "focus_mode", "night_owl_mode", "see_screen",
    ]
    for probe in [
        "ambient_learning_mode", "enable_ambient_learning_mode",
        "ambiant_mode", "ambient", "screen_shot",
        "totally_unrelated_command", "ambient_mode_please",
    ]:
        choice = autocorrect_command_choice(probe, sample_actions)
        prim = choice["primary"]
        sec = choice["secondary"]
        prim_s = f"{prim[0]!r} ({prim[1]:.3f})" if prim else "—"
        sec_s = f"{sec[0]!r} ({sec[1]:.3f})" if sec else "—"
        print(f"  {probe!r:42} [{choice['status']:9}] {prim_s:38} vs {sec_s}")
