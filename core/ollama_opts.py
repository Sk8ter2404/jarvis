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
#
# Review 2026-10-04: the cap the guard trusts is the RUNNING SERVER's, not
# this process's environment. The server reads OLLAMA_MAX_LOADED_MODELS once,
# at its own start, and logs it ('msg="server config" env="map[...
# OLLAMA_MAX_LOADED_MODELS:1 ...]"'); a value changed in the User
# environment reaches JARVIS at its next launch but the server only at ITS
# next restart. Trusting this process alone failed OPEN: set it to 2, restart
# only JARVIS, and the boot scan unloaded the brain exactly as before. So the
# cap is min(this process's value, the server's logged value), and 1 when the
# server's cannot be read (fail closed: indexing waits, the brain stays). One
# more case the log cannot show: a server that may hold 2 models still
# unloads one when the new one does not fit in VRAM - the first co-load that
# does unload something (note_coload) pins the cap to 1 for the process.
MAX_LOADED_ENV = "OLLAMA_MAX_LOADED_MODELS"


def max_loaded_models(environ=None) -> int:
    """OLLAMA_MAX_LOADED_MODELS from this process's environment (the value
    JARVIS persists to the User environment), else 1. A blank, zero,
    negative or non-integer value reads as 1 - the evicting behaviour JARVIS
    enforces. What the guard trusts is effective_max_loaded(). Never
    raises."""
    import os as _os
    env = _os.environ if environ is None else environ
    try:
        n = int(str(env.get(MAX_LOADED_ENV, "1")).strip())
    except Exception:
        return 1
    return n if n >= 1 else 1


_SERVER_CONFIG_MARK = b'msg="server config"'
_SERVER_CAP_RE = re.compile(rb"OLLAMA_MAX_LOADED_MODELS:(\d+)")
# Incremental read of the server log: {path: (bytes scanned, last value)}.
_server_log_scan: dict = {}
# True once a co-load unloaded a model anyway (note_coload).
_coload_evicted = [False]


def ollama_server_log(environ=None) -> "str | None":
    """Where the local Ollama server writes its log: %LOCALAPPDATA%/Ollama/
    server.log on Windows, ~/.ollama/logs/server.log elsewhere; None when
    neither exists. Never raises."""
    import os as _os
    env = _os.environ if environ is None else environ
    try:
        cands = []
        if env.get("LOCALAPPDATA"):
            cands.append(_os.path.join(env["LOCALAPPDATA"], "Ollama",
                                       "server.log"))
        cands.append(_os.path.join(_os.path.expanduser("~"), ".ollama",
                                   "logs", "server.log"))
        for c in cands:
            if _os.path.isfile(c):
                return c
    except Exception:
        pass
    return None


def server_max_loaded_models(base_url: str = "http://127.0.0.1:11434",
                             log_path: "str | None" = None) -> "int | None":
    """OLLAMA_MAX_LOADED_MODELS as the RUNNING local server read it: the last
    "server config" line of its log (``log_path``, default
    ollama_server_log()). Read incrementally - only bytes appended since the
    last call (from 0 again when the log shrank: a new log). None for a
    remote ``base_url``, a missing log, no such line, or a blank / zero value
    (the server's own default). Never raises."""
    import os as _os
    try:
        if endpoint_is_remote(base_url):
            return None
        path = log_path or ollama_server_log()
        if not path:
            return None
        size = _os.path.getsize(path)
        done, value = _server_log_scan.get(path, (0, None))
        if size < done:
            done, value = 0, None
        if size > done:
            with open(path, "rb") as fh:
                fh.seek(done)
                chunk = fh.read(size - done)
            cut = chunk.rfind(b"\n")
            if cut >= 0:
                for line in chunk[:cut].split(b"\n"):
                    if _SERVER_CONFIG_MARK in line:
                        m = _SERVER_CAP_RE.search(line)
                        value = int(m.group(1)) if m else None
                done += cut + 1
            _server_log_scan[path] = (done, value)
        return value if value and value >= 1 else None
    except Exception:
        return None


def effective_max_loaded(base_url: str = "http://127.0.0.1:11434",
                         environ=None, log_path: "str | None" = None) -> int:
    """How many models the guard may assume the server holds at once: 1 when
    this process's value is 1 (no log read) or a co-load already unloaded
    something (note_coload), else min(this process's value, the running
    server's logged one), and 1 when the server's is unknown. Never
    raises."""
    try:
        own = max_loaded_models(environ)
        if own <= 1 or _coload_evicted[0]:
            return 1
        srv = server_max_loaded_models(base_url, log_path)
        return min(own, srv) if srv else 1
    except Exception:
        return 1


def note_coload(before, after, model: str = "") -> bool:
    """After a request that co-loaded ``model`` next to the models ``before``
    (a /api/ps list): True, and the guard's cap is 1 from now on, when any
    of them is missing from ``after`` - the server unloaded it anyway (its
    VRAM, or a cap the log did not show). ``after`` None (unreadable) proves
    nothing. Never raises."""
    try:
        if after is None:
            return False
        gone = [n for n in (before or ())
                if not any(same_tag(a, n) for a in after)]
        if not gone:
            return False
        if not _coload_evicted[0]:
            print(f"  [ollama-guard] loading {model or 'a model'} unloaded "
                  f"{', '.join(gone)} although the server may hold more "
                  f"than one model - treating it as one from now on")
        _coload_evicted[0] = True
        return True
    except Exception:
        return False


def eviction_risk(model: str, base_url: str = "http://127.0.0.1:11434",
                  timeout_s: "float | None" = None, *,
                  max_loaded: "int | None" = None,
                  resident: "list | None" = None) -> str:
    """'' when a request naming ``model`` cannot unload another model: it is
    already loaded (exact tag, see same_tag), or fewer than ``max_loaded``
    (default effective_max_loaded()) models are loaded. Otherwise a short
    reason naming what it would unload. When /api/ps cannot be read the
    answer is a reason too: a caller that can wait must not gamble a ~15 GB
    brain reload on an unknown. ``resident``: the /api/ps list the caller
    already read (no second GET). Loads nothing. Never raises."""
    tag = (model or "").strip()
    if not tag:
        return "no model named"
    try:
        cap = (effective_max_loaded(base_url) if max_loaded is None
               else max(1, int(max_loaded)))
    except Exception:
        cap = 1
    if resident is None:
        t = probe_timeout(base_url) if timeout_s is None else timeout_s
        names = resident_models(base_url, t)
    else:
        names = list(resident)
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
