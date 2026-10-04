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
from urllib.parse import urlsplit

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

# ── Probe timeouts: sized to WHERE the server is ───────────────────────────
# A liveness or inventory probe of the LLM server (GET /api/tags, /api/ps) needs
# a timeout that fits the server's location. 2 s is right for loopback. It is far
# too tight for a REMOTE brain: measured live 2026-09-09 on an edge-node install
# whose brain was a GPU box on the tailnet, a healthy server mid-generation blew
# the 2 s probe, JARVIS declared the brain dead and answered "my local model
# isn't responding" while the remote box was serving fine. Every probe of the
# LLM server takes its timeout from probe_timeout() -- one knob, not six copies
# (a static test fails if a probe pins a literal timeout again).
PROBE_TIMEOUT_LOCAL_S = 2
PROBE_TIMEOUT_REMOTE_S = 8


def endpoint_is_remote(base_url) -> bool:
    """True when ``base_url`` points OFF this machine.

    Loopback is the whole 127.0.0.0/8 block, ``localhost``, ``::1`` and the
    wildcard ``0.0.0.0``. Anything unparseable returns False: treating junk as
    local keeps the pre-existing behaviour (short probe, local self-heal allowed)
    instead of inventing a remote brain. Never raises."""
    try:
        s = str(base_url or "").strip()
        if not s:
            return False
        host = urlsplit(s if "//" in s else "//" + s).hostname
        if not host:
            return False
        host = host.lower()
        if host in ("localhost", "::1", "0.0.0.0") or host.startswith("127."):
            return False
        return True
    except Exception:
        return False


def probe_timeout(base_url) -> float:
    """Seconds to wait on a liveness/inventory probe of ``base_url``."""
    return PROBE_TIMEOUT_REMOTE_S if endpoint_is_remote(base_url) else PROBE_TIMEOUT_LOCAL_S


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
                   timeout_s: float = 1.5, *, exact: bool = False) -> bool:
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

    ``exact=True`` (for a caller whose own request would BE the load, e.g.
    the idle re-prime): only the same tag counts, or a bare name against its
    ":latest" form. The default family match ("fam:12b" loaded satisfies
    "fam:26b") is fine for a nice-to-have ping but not as a no-cold-load
    gate — two tags of one family are two different ~GB loads.
    """
    tag = (model or "").strip()
    if not tag:
        return False
    names = resident_models(base_url, timeout_s)
    for name in names or ():
        if exact:
            if same_tag(name, tag):
                return True
            continue
        # Ollama reports fully-qualified tags ("nomic-embed-text:latest");
        # accept a bare-name configuration too.
        if name == tag or name.split(":", 1)[0] == tag.split(":", 1)[0]:
            return True
    return False


def same_tag(resident_name: str, tag: str) -> bool:
    """Is ``resident_name`` (as /api/ps reports it) exactly ``tag``? A bare
    name matches only its ":latest" form, in either direction. Mirrors
    skills/game_mode.py _keep_warm._same."""
    a = (resident_name or "").strip()
    b = (tag or "").strip()
    if not a or not b:
        return False
    return (a == b
            or (":" not in b and a == f"{b}:latest")
            or (":" not in a and b == f"{a}:latest"))


def resident_models(base_url: str = "http://127.0.0.1:11434",
                    timeout_s: float = 1.5) -> "list[str] | None":
    """The model tags Ollama has loaded right now (GET /api/ps), in its
    order; None when the list cannot be read (server down, timeout, junk).
    Loads nothing. Never raises."""
    import json as _json
    import urllib.request as _url
    try:
        req = _url.Request(f"{str(base_url).rstrip('/')}/api/ps", method="GET")
        with _url.urlopen(req, timeout=timeout_s) as resp:
            payload = _json.loads(resp.read().decode("utf-8", errors="replace"))
        out = []
        for m in (payload.get("models") or []):
            name = (m or {}).get("name") or (m or {}).get("model") or ""
            if name:
                out.append(str(name))
        return out
    except Exception:
        return None


# ── Brain-eviction guard (2026-10-04) ──────────────────────────────────────
# With OLLAMA_MAX_LOADED_MODELS=1 (the value JARVIS persists, see the
# monolith's _ensure_ollama_single_model_env) ANY request naming a model that
# is not loaded unloads whatever is: on 10-02 15:05:40 and 10-03 17:35:41 the
# RAG boot scan's nomic-embed-text requests unloaded the voice brain, and the
# next brain loads took 54 s and 9 s (Ollama server.log). A caller whose work
# can wait asks eviction_risk() first and waits when it says no.
MAX_LOADED_ENV = "OLLAMA_MAX_LOADED_MODELS"


def max_loaded_models(environ=None) -> int:
    """How many models the Ollama server holds at once, as JARVIS runs it:
    OLLAMA_MAX_LOADED_MODELS from this process's environment (the value
    JARVIS persists to the User environment, which the server reads at its
    own start), else 1. A blank, zero, negative or non-integer value reads
    as 1 - the evicting behaviour JARVIS enforces. Never raises."""
    import os as _os
    env = _os.environ if environ is None else environ
    try:
        n = int(str(env.get(MAX_LOADED_ENV, "1")).strip())
    except Exception:
        return 1
    return n if n >= 1 else 1


def eviction_risk(model: str, base_url: str = "http://127.0.0.1:11434",
                  timeout_s: "float | None" = None, *,
                  max_loaded: "int | None" = None) -> str:
    """'' when a request naming ``model`` cannot unload another model: it is
    already loaded (exact tag, see same_tag), or fewer than ``max_loaded``
    (default max_loaded_models()) models are loaded. Otherwise a short reason
    naming what it would unload. When /api/ps cannot be read the answer is a
    reason too: a caller that can wait must not gamble a ~15 GB brain reload
    on an unknown. Loads nothing. Never raises."""
    tag = (model or "").strip()
    if not tag:
        return "no model named"
    cap = max_loaded_models() if max_loaded is None else max(1, int(max_loaded))
    t = probe_timeout(base_url) if timeout_s is None else timeout_s
    names = resident_models(base_url, t)
    if names is None:
        return f"could not read {str(base_url).rstrip('/')}/api/ps"
    if any(same_tag(n, tag) for n in names):
        return ""
    if len(names) < cap:
        return ""
    return (f"loading {tag} would unload {', '.join(names)} "
            f"(Ollama holds {cap} model{'s' if cap != 1 else ''})")


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
