"""model_picker — choose which LOCAL Ollama model JARVIS runs on, BY VOICE.

JARVIS's always-on baseline brain is a local Ollama model (the 3090 rig). This
skill lets the owner pick WHICH installed chat model serves turns — live (no
restart) and persistently (survives a restart) — plus flip the chat ROUTE
(local / cloud / auto) by voice.

Actions registered (all `fn(arg: str) -> str`, spoken-friendly):
  * list_models    — list the installed CHAT models, marking the active one
                     (excludes the embedding + vision models). "what models
                     do you have / what can you run on".
  * current_model  — report the ACTIVE local model in plain speech. "what
                     model are you using / what's your brain".
  * set_model, <name|alias>
                   — switch the active local chat model. Fuzzy/alias match
                     against INSTALLED chat models (exact tag, 32b/14b/8b,
                     size words big/medium/small/fast, family words
                     qwen/llama). VALIDATES it's installed (never switches to a
                     non-installed model, and rejects the embed/vision models
                     for the chat slot). Switches LIVE (resolver cache +
                     LOCAL_LLM_MODEL) so the very next turn uses it, and
                     PERSISTS to data/user_settings.json so it survives restart.
  * set_brain, <local|cloud|auto>
                   — switch the chat ROUTE; updates + persists
                     MODEL_ROUTING["chat"]. No/`status` arg reports the route.

Reaching the running monolith
─────────────────────────────
The live model selector lives in the monolith (`_get_local_llm_model()` reads
the module cache `_RESOLVED_LOCAL_LLM_MODEL[0]`, falling through to the
`LOCAL_LLM_MODEL` global). We reach that module the same way
skills/itunes_library.py does — ``sys.modules.get("__main__") or
sys.modules.get("bobert_companion")`` — because the monolith runs as __main__
and is aliased to the importable name at boot. Setting the cache's element 0 is
the documented lock-free mutation pattern (core/state.py style) and short-
circuits the resolver ahead of the env-var / preference-chain branches, so the
switch takes effect on the next turn with no restart.

Persistence reuses the Settings-GUI writer (tools/settings_window:
load_settings + save_settings), which is the SAME atomic, merge-not-clobber
path the GUI's Save button uses — so writing LOCAL_LLM_MODEL never clobbers the
owner's other saved knobs. See `_persist_setting`.

Network-resilient: every Ollama call is wrapped; if Ollama is unreachable the
actions degrade to a spoken-friendly line instead of raising.
"""
from __future__ import annotations

import sys

import requests

# Ollama base URL + the names of the NON-chat models to hide from / reject for
# the chat slot. Pulled from core.config so a host that moved Ollama still
# works; falls back to the well-known defaults if config can't be imported.
try:
    from core.config import LOCAL_LLM_BASE_URL as _BASE_URL
except Exception:  # pragma: no cover - config always importable in practice
    _BASE_URL = "http://127.0.0.1:11434"

# Substrings that mark a tag as NOT a chat model. The embedding model can't
# chat at all; the VL (vision-language) model is reserved for the vision slot
# (LOCAL_VISION_MODEL) and must never be selected as the chat brain.
_EMBED_MARKERS = ("nomic-embed", "embed-text", "-embed", "bge-", "all-minilm")
_VISION_MARKERS = ("vl:", "-vl", "vision", "llava", "moondream", "bakllava")

_TIMEOUT = (3, 5)  # (connect, read) — /api/tags is tiny and local


# ─── monolith reach (mirrors skills/itunes_library._apple_music_fallback) ──
def _monolith():
    """The running monolith module, or None. It runs as __main__ and is also
    importable as bobert_companion (aliased at boot)."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


# ─── Ollama tag listing ────────────────────────────────────────────────────
def _list_tags() -> list[str] | None:
    """All installed Ollama tags via GET /api/tags, or None if Ollama is
    unreachable (so callers can tell 'no models' from 'Ollama down')."""
    try:
        r = requests.get(f"{_BASE_URL}/api/tags", timeout=_TIMEOUT)
        if not r.ok:
            return None
        return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return None


def _is_embed(tag: str) -> bool:
    t = (tag or "").lower()
    return any(m in t for m in _EMBED_MARKERS)


def _is_vision(tag: str) -> bool:
    t = (tag or "").lower()
    return any(m in t for m in _VISION_MARKERS)


def _chat_models(tags: list[str]) -> list[str]:
    """Installed tags that are usable as the CHAT brain — drops embedding and
    vision models. Order preserved as Ollama returned them."""
    return [t for t in tags if t and not _is_embed(t) and not _is_vision(t)]


def _vision_models(tags: list[str]) -> list[str]:
    return [t for t in tags if t and _is_vision(t)]


# ─── active-model resolution ───────────────────────────────────────────────
def _active_model() -> str | None:
    """The model the monolith would use for the next local turn: the resolver
    cache if warm, else what `_get_local_llm_model()` resolves, else the
    LOCAL_LLM_MODEL constant. None only if the monolith isn't reachable AND we
    can't fall back to config."""
    bc = _monolith()
    if bc is not None:
        try:
            cache = getattr(bc, "_RESOLVED_LOCAL_LLM_MODEL", None)
            if isinstance(cache, list) and cache and cache[0]:
                return cache[0]
        except Exception:
            pass
        getter = getattr(bc, "_get_local_llm_model", None)
        if callable(getter):
            try:
                got = getter()
                if isinstance(got, str) and got:
                    return got
            except Exception:
                pass
        const = getattr(bc, "LOCAL_LLM_MODEL", None)
        if isinstance(const, str) and const:
            return const
    try:
        from core.config import LOCAL_LLM_MODEL as _cfg_model
        return _cfg_model or None
    except Exception:
        return None


def _short_name(tag: str) -> str:
    """A spoken-friendly label for an Ollama tag, e.g.
    'qwen2.5:32b-instruct-q4_K_M' → 'qwen 32B'; 'llama3.1:8b-...' → 'llama 8B'.
    Falls back to the bare base name when no size token is present."""
    t = (tag or "").lower()
    family = "qwen" if "qwen" in t else "llama" if "llama" in t else \
        "mistral" if "mistral" in t else "gemma" if "gemma" in t else \
        "phi" if "phi" in t else (tag.split(":", 1)[0] if tag else tag)
    size = ""
    for s in ("72b", "70b", "65b", "34b", "32b", "14b", "13b", "8b", "7b", "3b", "1.5b"):
        if s in t:
            size = s.upper()
            break
    return f"{family} {size}".strip() if size else (tag.split(":", 1)[0] or tag)


def _same_model(a: str, b: str) -> bool:
    """True when two tags denote the SAME model. Exact (case-insensitive) tag
    match, OR same base name where neither carries a DIFFERENT explicit size —
    so a bare 'qwen2.5' matches 'qwen2.5:32b-...', but 'qwen2.5:14b' does NOT
    match 'qwen2.5:32b' (different sizes of the same family are distinct
    models). Distinguishing 14B from 32B is the whole point of the feature, so
    a plain base-name comparison would be wrong here."""
    if not a or not b:
        return False
    al, bl = a.lower(), b.lower()
    if al == bl:
        return True
    if a.split(":", 1)[0].lower() != b.split(":", 1)[0].lower():
        return False
    # Same base name: same model only if their size tokens don't conflict.
    ra, rb = _size_rank(a), _size_rank(b)
    unknown = len(_SIZE_ORDER)
    if ra == unknown or rb == unknown:   # one side has no size token → treat same
        return True
    return ra == rb


# ─── vision lockstep (chat + vision share ONE multimodal brain) ────────────
def _is_multimodal(tag: str) -> bool:
    """True when `tag` can serve VISION as well as chat. Asks Ollama's
    /api/show for the model's declared capabilities; on any error falls back
    to the family markers (_VISION_MARKERS plus gemma4, the known multimodal
    CHAT family). Never raises."""
    t = (tag or "").strip()
    if not t:
        return False
    try:
        r = requests.post(f"{_BASE_URL}/api/show", json={"model": t},
                          timeout=_TIMEOUT)
        if r.ok:
            return "vision" in (r.json().get("capabilities") or [])
    except Exception:
        pass
    tl = t.lower()
    return any(m in tl for m in _VISION_MARKERS) or "gemma4" in tl


def _sync_vision_to_chat(old_tag, new_tag, persist: bool = True, bc=None) -> bool:
    """Keep LOCAL_VISION_MODEL in LOCKSTEP with a chat-model switch.

    core/config.py mandates the lockstep ("promoting the brain never forks
    vision onto a second VLM"), but until the 2026-07-21 audit every switch
    site repointed only the CHAT tag — leaving vision on the OLD tag, so the
    co-load guard refused every local vision call ("REFUSING co-load: big
    model ... is resident") until restart.

    Syncs ONLY when the current vision tag equals the OLD chat tag (the
    shared-brain config) AND the new tag is vision-capable:
      * a user-pinned separate VLM (vision != old chat tag) is never touched;
      * a switch to a TEXT-ONLY tag leaves vision on the old multimodal tag
        (the residency/co-load guards then degrade with a printed reason
        instead of silently blinding local vision on a chat-only model).

    `bc` is the monolith module (resolved via _monolith() when omitted).
    `persist=False` for callers that don't persist the chat tag either —
    persisting vision alone would desync user_settings.json on restart.
    Returns True when vision was repointed. Never raises."""
    if bc is None:
        bc = _monolith()
    vision = getattr(bc, "LOCAL_VISION_MODEL", None) if bc is not None else None
    if not vision:
        try:
            import core.config as _cfg
            vision = getattr(_cfg, "LOCAL_VISION_MODEL", None)
        except Exception:
            vision = None
    if not (isinstance(vision, str) and vision and old_tag and new_tag):
        return False
    if not _same_model(vision, old_tag):
        return False   # separate pinned VLM — never touch it
    if _same_model(vision, new_tag):
        return False   # already in lockstep — nothing to do
    if not _is_multimodal(new_tag):
        return False   # text-only brain — keep the old multimodal vision tag
    if bc is not None:
        try:
            setattr(bc, "LOCAL_VISION_MODEL", new_tag)
        except Exception:
            pass
    try:
        import core.config as _cfg
        _cfg.LOCAL_VISION_MODEL = new_tag
    except Exception:
        pass
    if persist:
        _persist_setting("LOCAL_VISION_MODEL", new_tag)
    return True


# ─── persistence (reuse the Settings-GUI atomic, merge-not-clobber writer) ──
def _persist_setting(key: str, value) -> bool:
    """Write {key: value} into data/user_settings.json WITHOUT clobbering the
    owner's other saved settings.

    REUSES the Settings GUI's own writer: ``settings_window.save_settings`` is
    the exact atomic temp-file + os.replace, schema-coercing write its Save
    button uses, and ``load_settings`` returns the full merged document
    (every persisted schema key + any passthrough keys) — so loading it,
    overlaying our key, and saving it back is byte-for-byte the GUI's
    merge-not-clobber path.

    When ``value`` is a dict (e.g. a PARTIAL ``MODEL_ROUTING`` update like
    {"chat": "local"}), it is MERGED into the existing dict so the other
    sub-keys (vision / ambient routes) are preserved — never replaced. Returns
    True on success, False on any error (the live switch already took effect;
    persistence is best-effort)."""
    try:
        from tools import settings_window as sw
    except Exception:
        return False
    try:
        current = sw.load_settings()
        if not isinstance(current, dict):
            current = {}
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            merged = dict(current[key])
            merged.update(value)
            current[key] = merged
        else:
            current[key] = value
        sw.save_settings(current)
        return True
    except Exception:
        return False


# ─── alias / fuzzy resolution ──────────────────────────────────────────────
# Size words → which end of the installed-chat-model size order to pick.
_BIG_WORDS = (
    "big", "bigger", "biggest", "large", "largest", "smart", "smartest",
    "best", "strong", "strongest", "sharp", "sharpest", "powerful", "max",
    "maximum", "heavy", "full",
)
_MED_WORDS = ("medium", "mid", "middle", "balanced", "moderate", "regular", "standard")
_SMALL_WORDS = (
    "small", "smaller", "smallest", "fast", "faster", "fastest", "light",
    "lighter", "lightest", "lightweight", "lite", "quick", "quicker",
    "quickest", "snappy", "tiny", "mini", "low",
)

# Size tokens, largest→smallest, used to rank chat tags for big/medium/small.
_SIZE_ORDER = ("72b", "70b", "65b", "34b", "32b", "14b", "13b", "8b", "7b", "3b", "1.5b")


def _size_rank(tag: str) -> int:
    """Index of a tag's size in _SIZE_ORDER (lower = bigger). Unknown → end."""
    t = (tag or "").lower()
    for i, s in enumerate(_SIZE_ORDER):
        if s in t:
            return i
    return len(_SIZE_ORDER)


def _by_size(chat: list[str]):
    """Chat models sorted largest→smallest by their size token."""
    return sorted(chat, key=_size_rank)


def _resolve_alias(arg: str, chat: list[str]) -> str | None:
    """Resolve a spoken arg to one INSTALLED chat tag, or None if nothing fits.

    Match order (first hit wins):
      1. exact tag (case-insensitive).
      2. explicit size token in the arg: 32b/34b → ~32B, 14b/13b → ~14B,
         8b/7b → ~8B (matched against installed tags).
      3. size WORDS: big/large/smart/best/strong → largest installed chat;
         medium → the middle one; small/fast/light/quick → smallest chat.
      4. family WORDS: qwen → the qwen default (prefer 32B, else first qwen);
         llama → the first llama.
      5. loose substring of the arg against a tag (last resort).
    """
    if not chat:
        return None
    raw = (arg or "").strip()
    low = raw.lower()
    if not low:
        return None

    # 1. exact tag (case-insensitive), incl. bare base name ("qwen2.5").
    for t in chat:
        if t.lower() == low:
            return t
    for t in chat:
        if t.split(":", 1)[0].lower() == low:
            return t

    words = set(low.replace("-", " ").replace(":", " ").split())

    # 2. explicit size token anywhere in the arg.
    def _first_with(*sizes):
        for t in _by_size(chat):
            tl = t.lower()
            if any(s in tl for s in sizes):
                return t
        return None

    if "32b" in low or "34b" in low or "30b" in low:
        hit = _first_with("32b", "34b")
        if hit:
            return hit
    if "14b" in low or "13b" in low:
        hit = _first_with("14b", "13b")
        if hit:
            return hit
    if "8b" in low or "7b" in low:
        hit = _first_with("8b", "7b")
        if hit:
            return hit

    ordered = _by_size(chat)  # largest → smallest

    # 3. size words.
    if words & set(_BIG_WORDS):
        return ordered[0]
    if words & set(_SMALL_WORDS):
        return ordered[-1]
    if words & set(_MED_WORDS):
        return ordered[len(ordered) // 2]

    # 4. family words. qwen → prefer the 32B default, else the first qwen.
    if "qwen" in low:
        for t in ordered:
            if "qwen" in t.lower() and "32b" in t.lower():
                return t
        for t in chat:
            if "qwen" in t.lower():
                return t
    if "llama" in low:
        for t in chat:
            if "llama" in t.lower():
                return t

    # 5. loose substring either direction.
    for t in chat:
        tl = t.lower()
        if low in tl or tl.split(":", 1)[0] in low:
            return t
    return None


def _spoken_options(chat: list[str]) -> str:
    """A short spoken list of installed chat models by friendly name."""
    if not chat:
        return "none"
    return ", ".join(_short_name(t) for t in _by_size(chat))


# ─── actions ───────────────────────────────────────────────────────────────
def list_models(arg: str = "") -> str:
    tags = _list_tags()
    if tags is None:
        return ("I can't reach Ollama right now, sir — the local model server "
                "may be offline.")
    chat = _chat_models(tags)
    if not chat:
        return ("I don't see any local chat models installed, sir. Pull one "
                "with Ollama — for example qwen2.5:14b-instruct-q5_K_M.")
    active = _active_model()
    parts = []
    for t in _by_size(chat):
        label = _short_name(t)
        if active and _same_model(t, active):
            label += " (active)"
        parts.append(label)
    out = f"Installed local chat models, sir: {', '.join(parts)}."
    vis = _vision_models(tags)
    if vis:
        out += f" I also have {_short_name(vis[0])} for vision."
    return out


def _chat_route() -> str:
    """The active CHAT route — 'local' | 'cloud' | 'auto' — from
    MODEL_ROUTING['chat'], the SAME source set_brain writes and _call_llm reads
    (bobert_companion `model_route("chat")`). 'auto' = Claude when reachable,
    local on failure. Falls back to the config default when the monolith isn't
    reachable."""
    bc = _monolith()
    if bc is not None:
        try:
            routing = getattr(bc, "MODEL_ROUTING", None)
            if isinstance(routing, dict) and routing.get("chat"):
                return str(routing["chat"]).lower()
        except Exception:
            pass
    try:
        from core.config import MODEL_ROUTING as _mr
        return str(_mr.get("chat", "auto")).lower()
    except Exception:
        return "auto"


def current_model(arg: str = "") -> str:
    """Report the brain the NEXT chat turn will actually use. Reads the live
    chat route (MODEL_ROUTING['chat'], what set_brain sets) so a 'switch to
    Claude' is reflected — previously this always named the local model even
    when Claude was handling every turn (2026-07-04 live-repro: after 'switch
    to the cloud model' it still said 'qwen 14B locally, sir')."""
    route = _chat_route()
    active = _active_model()
    if route == "cloud":
        return "I'm running on Claude in the cloud, sir."
    if route == "auto":
        if active:
            return (f"I'm on auto, sir — Claude when it's reachable, with "
                    f"{_short_name(active)} locally as the fallback.")
        return "I'm on auto, sir — Claude when reachable, local as the fallback."
    # route == "local" (or anything unexpected): report the local model.
    if not active:
        return ("I'm not sure which local model is active, sir — I can't reach "
                "the model selector right now.")
    return f"I'm running on {_short_name(active)} locally, sir ({active})."


def set_model(arg: str) -> str:
    want = (arg or "").strip()
    if not want:
        return ("Which model would you like, sir? Say something like 'switch to "
                "the 32B' or 'use the fast one'.")
    tags = _list_tags()
    if tags is None:
        return ("I can't reach Ollama to switch models right now, sir — the "
                "local server may be offline.")
    chat = _chat_models(tags)
    if not chat:
        return ("There are no local chat models installed to switch to, sir. "
                "Pull one with Ollama first.")

    # If the user explicitly named the embed/vision model (by its exact tag or
    # bare base name) for the CHAT slot, reject it clearly — never silently fall
    # through to a fuzzy chat match. Only an EXACT name counts, so a family word
    # like "qwen" (which also matches a chat model) still routes to chat.
    low = want.lower()

    def _names_tag(t: str) -> bool:
        return t.lower() == low or t.split(":", 1)[0].lower() == low

    named_non_chat = [t for t in tags if (_is_embed(t) or _is_vision(t)) and _names_tag(t)]
    named_chat = [t for t in chat if _names_tag(t)]
    if named_non_chat and not named_chat:
        kind = "an embedding" if _is_embed(named_non_chat[0]) else "a vision"
        return (f"{_short_name(named_non_chat[0])} is {kind} model, sir — not a "
                f"chat brain. Installed chat models: {_spoken_options(chat)}.")

    target = _resolve_alias(want, chat)
    if not target:
        return (f"I couldn't match '{want}' to an installed chat model, sir. "
                f"I have: {_spoken_options(chat)}.")

    # Validate (belt-and-suspenders — _resolve_alias only returns from `chat`).
    if target not in tags or _is_embed(target) or _is_vision(target):
        return (f"I can only switch to a model that's actually installed, sir. "
                f"I have: {_spoken_options(chat)}.")

    active = _active_model()
    if active and _same_model(target, active):
        return f"I'm already running on {_short_name(target)}, sir."

    # ── LIVE switch: prime the resolver cache + the LOCAL_LLM_MODEL global so
    # the very next turn uses `target` with no restart. Setting the cache's
    # element 0 short-circuits _get_local_llm_model ahead of its env-var and
    # preference-chain branches (the documented lock-free mutation pattern).
    switched_live = False
    bc = _monolith()
    if bc is not None:
        try:
            cache = getattr(bc, "_RESOLVED_LOCAL_LLM_MODEL", None)
            if isinstance(cache, list):
                cache[0] = target          # invalidate→repoint the resolver cache
                switched_live = True
            else:
                setattr(bc, "_RESOLVED_LOCAL_LLM_MODEL", [target])
                switched_live = True
        except Exception:
            switched_live = False
        try:
            setattr(bc, "LOCAL_LLM_MODEL", target)   # fallback path + pull msgs
        except Exception:
            pass
    # Keep core.config in sync too, so anything reading the constant directly
    # (and a future _apply_user_settings re-read) sees the new value.
    try:
        import core.config as _cfg
        _cfg.LOCAL_LLM_MODEL = target
    except Exception:
        pass

    persisted = _persist_setting("LOCAL_LLM_MODEL", target)

    # Vision LOCKSTEP (2026-07-21 audit): when vision shares the OLD chat tag
    # (the shipped one-multimodal-brain config), carry LOCAL_VISION_MODEL
    # along — live AND persisted — so the switch can't fork vision onto a
    # second VLM / dead tag. No-op for a pinned separate VLM or a text-only
    # target (see _sync_vision_to_chat).
    _sync_vision_to_chat(active, target, persist=True, bc=bc)

    # Confirmation, with a one-beat character note on the trade-off.
    note = ""
    rank = _size_rank(target)
    if rank <= _SIZE_ORDER.index("32b"):
        note = " — sharper, a touch slower"
    elif rank >= _SIZE_ORDER.index("8b"):
        note = " — faster, slightly less sharp"
    msg = f"Switching to {_short_name(target)}, sir{note}."
    if not switched_live and bc is None:
        # No live monolith (e.g. invoked outside a running session): the
        # persisted value will take effect on the next start.
        msg = (f"Set the local model to {_short_name(target)}, sir — it'll take "
               f"effect on the next start.")
    elif not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


def set_brain(arg: str = "") -> str:
    """Switch the CHAT route: local | cloud | auto. Updates + persists
    MODEL_ROUTING['chat']. Empty / 'status' arg reports the current route."""
    want = (arg or "").strip().lower()

    def _current_route() -> str:
        bc = _monolith()
        if bc is not None:
            try:
                routing = getattr(bc, "MODEL_ROUTING", None)
                if isinstance(routing, dict) and routing.get("chat"):
                    return str(routing["chat"])
            except Exception:
                pass
        try:
            from core.config import MODEL_ROUTING as _mr
            return str(_mr.get("chat", "auto"))
        except Exception:
            return "auto"

    if not want or want in ("status", "current", "what", "which"):
        cur = _current_route()
        human = {"local": "the local model (free)", "cloud": "Claude",
                 "auto": "auto — Claude when available, local on failure"}.get(cur, cur)
        return f"My chat brain is set to {human}, sir."

    # Normalise common phrasings to the three canonical routes.
    if want in ("local", "offline", "ollama", "free", "on-device", "on device"):
        route = "local"
    elif want in ("cloud", "claude", "online", "anthropic", "remote"):
        route = "cloud"
    elif want in ("auto", "automatic", "default", "hybrid", "smart"):
        route = "auto"
    else:
        return ("Say local, cloud, or auto, sir — local is the free on-device "
                "model, cloud is Claude, auto uses Claude with a local fallback.")

    # ── live update of MODEL_ROUTING['chat'] (merge, never replace the dict).
    bc = _monolith()
    if bc is not None:
        try:
            routing = getattr(bc, "MODEL_ROUTING", None)
            if isinstance(routing, dict):
                routing["chat"] = route
            else:
                setattr(bc, "MODEL_ROUTING", {"chat": route})
        except Exception:
            pass
    try:
        import core.config as _cfg
        if isinstance(getattr(_cfg, "MODEL_ROUTING", None), dict):
            _cfg.MODEL_ROUTING["chat"] = route
    except Exception:
        pass

    persisted = _persist_setting("MODEL_ROUTING", {"chat": route})

    human = {"local": "the local model, sir — $0 per turn",
             "cloud": "Claude, sir",
             "auto": "auto, sir — Claude when available, local on failure"}[route]
    msg = f"Chat brain set to {human}."
    if not persisted:
        msg += " (Couldn't save it, so it'll revert on restart.)"
    return msg


def register(actions: dict) -> None:
    actions["list_models"] = list_models
    actions["current_model"] = current_model
    actions["set_model"] = set_model
    actions["set_brain"] = set_brain
