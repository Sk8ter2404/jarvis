"""Canonical Ollama request-option builders — ONE source of truth.

WHY THIS MODULE EXISTS
======================
Ollama keys a loaded runner by (model, options). Two callers that ask for the
same model with DIFFERENT options do not share the warm runner: the second one
EVICTS the first and reloads the weights under its own config. The context
length is part of that key, so a single call site that forgets ``num_ctx``
silently reloads the model at the model's own default window.

That is not theoretical. Live on 2026-07-21, with chat and vision both pointed
at the same multimodal tag (``gemma4:26b-a4b-it-qat`` — the v2.0.33 design
where ONE model serves both), the chat path pinned ``num_ctx=16384`` while
``ask_vision`` sent only ``num_predict``. Ollama's own server log recorded the
consequence::

    llama_context: n_ctx = 262144
    srv load_model: initializing, n_slots = 1, n_ctx_slot = 262144

``ollama ps`` then showed ``16 GB  6%/94% CPU/GPU  CONTEXT 262144`` with the
3090 pinned at 24147/24576 MiB: a 256K KV cache does not fit in 24 GB, so
llama.cpp spilled the model to CPU. The next voice turn died on the 50 s read
timeout and JARVIS said "My local model isn't responding and I can't reach the
cloud either, sir." The ambient-extract daemon fires a vision call every 300 s,
so the primary brain was being bricked on a five-minute cycle.

The heuristic below used to live only as ``bobert_companion._local_num_ctx``.
Non-monolith callers (``core/orchestrator.py``) could not import it without
booting a second JARVIS, so they sent no options at all — the stale-duplicate
bug class this codebase keeps paying for. It lives here now: pure, importable
from anywhere in ``core``/``skills``, and re-exported by the monolith so every
existing caller and test keeps working.
"""
from __future__ import annotations

import re

# The window every model that comfortably fits gets. Measured on the 3090.
DEFAULT_NUM_CTX = 16384
# The tighter window for 30B-class-and-up tags. MEASURED on this box (RTX 3090,
# 24 GB): a 32B-class q4_K_M at 16384 spills ~5 % to CPU and runs ~28 tok/s
# (fragile); at 12288 it stays 100 % on the GPU and runs ~49 tok/s (stable).
BIG_MODEL_NUM_CTX = 12288

# Tags that are unambiguously 30B-class or larger. `30b` covers the qwen3:30b-a3b
# MoE, which previously fell through to the 16k window (~40 % slower + a CPU
# spill every turn).
_BIG_TAGS = ("30b", "32b", "34b", "65b", "70b", "72b")

# Digit-runs immediately followed by `b` (e.g. the `30` in `30b`), but NOT the
# active-param `a3b` MoE suffix — the leading `a` is excluded by the lookbehind
# so `qwen3:30b-a3b` parses as 30, not 3.
_SIZE_RE = re.compile(r"(?<![a-z0-9])(\d+)b\b")


def local_num_ctx(model: str) -> int:
    """Pick the Ollama ``num_ctx`` for a model so it fits 100 % on the 3090.

    Smaller models (14B/8B/26B-class) have headroom to spare and keep the larger
    16k window; any tag that looks 30B-class or bigger gets the tighter 12k one.
    """
    tag = (model or "").lower()
    if any(b in tag for b in _BIG_TAGS):
        return BIG_MODEL_NUM_CTX
    # General param-parse so any FUTURE >=30B tag also gets the tight window
    # without needing a literal added above.
    try:
        sizes = [int(n) for n in _SIZE_RE.findall(tag)]
        if sizes and max(sizes) >= 30:
            return BIG_MODEL_NUM_CTX
    except Exception:
        pass
    return DEFAULT_NUM_CTX


def model_resident(model: str, base_url: str = "http://127.0.0.1:11434",
                   timeout_s: float = 1.5) -> bool:
    """True iff ``model`` is ALREADY loaded in Ollama right now.

    The guard for optional, latency-sensitive extras (autocorrect embeddings,
    reachability pings). With OLLAMA_MAX_LOADED_MODELS=1 — the setting JARVIS
    persists so Ollama EVICTS rather than co-loads — any request naming a
    model that is not resident silently evicts whatever IS resident, i.e. the
    voice brain. A 1.5 s client timeout does not protect you: giving up on the
    response does not cancel the load the server already started.

    So: nice-to-have callers must ask this FIRST and skip themselves when the
    answer is False, rather than firing a request that costs a brain reload.
    Cheap GET of /api/ps; never raises.
    """
    tag = (model or "").strip()
    if not tag:
        return False
    import json as _json
    import urllib.request as _url
    try:
        req = _url.Request(f"{base_url.rstrip('/')}/api/ps", method="GET")
        with _url.urlopen(req, timeout=timeout_s) as resp:
            payload = _json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return False
    for m in (payload.get("models") or []):
        name = (m or {}).get("name") or (m or {}).get("model") or ""
        if not name:
            continue
        # Ollama reports fully-qualified tags ("nomic-embed-text:latest");
        # accept a bare-name configuration too.
        if name == tag or name.split(":", 1)[0] == tag.split(":", 1)[0]:
            return True
    return False


def chat_options(model: str, *, num_predict: int | None = None,
                 temperature: float | None = None,
                 extra: dict | None = None) -> dict:
    """Build an Ollama ``options`` dict that is RUNNER-COMPATIBLE with every
    other JARVIS call for the same model.

    ``num_ctx`` is always present — that is the whole point. Callers add their
    own knobs on top; anything in ``extra`` wins last so a caller can still
    override deliberately (and take the reload it implies).
    """
    opts: dict = {"num_ctx": local_num_ctx(model)}
    if num_predict is not None:
        opts["num_predict"] = num_predict
    if temperature is not None:
        opts["temperature"] = temperature
    if extra:
        opts.update(extra)
    return opts
