#!/usr/bin/env python3
"""model_lockstep — the ONE copy of the chat↔vision model lockstep rule.

core/config.py mandates that ``LOCAL_VISION_MODEL`` moves WITH
``LOCAL_LLM_MODEL`` ("Kept in lockstep with LOCAL_LLM_MODEL so promoting the
brain never forks vision onto a second VLM"): the shipped config points both
at the SAME multimodal tag, so a vision call re-uses the already-resident
chat model instead of co-loading a second one on top of it.

Why this module exists
──────────────────────
Until the 2026-08-20 audit the rule had FOUR switch sites and ONE copy:

  * ``skills.model_picker._sync_vision_to_chat``  — the copy (voice switch),
    also reused by ``core.actions``' backend switch;
  * ``tools/settings_window.py``  — the Settings GUI's Save, and
  * ``tools/web_interface.py``    — POST /api/settings,

and the two settings writers repointed the chat tag with NO vision
consideration at all. A GUI/dashboard save therefore forked the pair — and
once forked neither UI could repair it, because the voice path reads a
mismatched vision tag as a deliberate user pin and refuses to touch it
(``_sync_vision_to_chat``'s "separate pinned VLM — never touch it" branch).

That is this repo's #1 bug class (the stale duplicate), so the DECISION now
lives here, once, and every switch site imports it. The callers still own the
assignment/persistence — this module never writes anything.

Contract
────────
Pure, stdlib-only, no network, never raises. Nothing here imports the
monolith, ``requests``, or ``core.config`` at module scope, so the
deliberately dependency-light Settings GUI and the stdlib-only VRAM engine
can both use it. ``is_multimodal`` is injectable so the voice path can keep
asking Ollama for real capabilities while the offline UIs fall back to the
family markers.
"""
from __future__ import annotations

import re

# Tag substrings that mark a model as VISION-capable. Single source of truth:
# skills/model_picker.py and core/vram_budget.py import this rather than each
# keeping their own copy (they had three identical tuples before 2026-08-20).
VISION_MARKERS = ("vl:", "-vl", "vision", "llava", "moondream", "bakllava")

# Chat families that are multimodal WITHOUT carrying a vision marker in the
# tag — gemma4 is the shipped one-brain default (text + images).
MULTIMODAL_CHAT_FAMILIES = ("gemma4",)

# Size tokens, largest→smallest. Index = rank (lower is bigger); a tag with no
# recognised token ranks last (see size_rank). RANKING ONLY — identity is
# decided by the declared parameter count (declared_size_b), because this list
# can never enumerate every size that ships (it had no entry for the 26B
# default brain or its 12B sibling, which made same_model() call them the SAME
# model).
SIZE_ORDER = ("72b", "70b", "65b", "34b", "32b", "14b", "13b", "8b", "7b",
              "3b", "1.5b")

# The parameter count a tag DECLARES, e.g. the 26 in "gemma4:26b-a4b-it-qat".
# Digit-runs immediately followed by `b`, but NOT the active-param `a4b` MoE
# suffix — the leading `a` is excluded by the lookbehind, so `qwen3:30b-a3b`
# parses as 30, not 3. Same pattern core/ollama_opts.py uses to size a tag.
_SIZE_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b\b")

# Decision reasons returned by vision_lockstep_decision(). Stable strings —
# callers log/render them and the tests pin them.
LOCKSTEP_SYNC = "sync"              # vision must be repointed to the new tag
LOCKSTEP_ALREADY = "already"        # vision already denotes the new tag
LOCKSTEP_PINNED = "pinned"          # user pinned a separate VLM — never touch
LOCKSTEP_TEXT_ONLY = "text-only"    # new chat tag can't serve vision
LOCKSTEP_INCOMPLETE = "incomplete"  # a tag is missing — nothing to decide


def size_rank(tag: str) -> int:
    """Index of a tag's size in SIZE_ORDER (lower = bigger). Unknown → end."""
    t = (tag or "").lower()
    for i, s in enumerate(SIZE_ORDER):
        if s in t:
            return i
    return len(SIZE_ORDER)


def declared_size_b(tag: str) -> float | None:
    """The parameter count in billions a tag declares, or None when it doesn't.

    'gemma4:26b-a4b-it-qat' → 26.0 (not 4 — the MoE active-param suffix is
    excluded), 'qwen2.5:14b-instruct-q5_K_M' → 14.0, 'gemma4:latest' → None."""
    m = _SIZE_RE.search((tag or "").lower())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:              # pragma: no cover - regex guarantees numeric
        return None


def same_model(a: str, b: str) -> bool:
    """True when two tags denote the SAME model. Exact (case-insensitive) tag
    match, OR same base name where neither carries a DIFFERENT explicit size —
    so a bare 'qwen2.5' matches 'qwen2.5:32b-...', but 'qwen2.5:14b' does NOT
    match 'qwen2.5:32b' (different sizes of the same family are distinct
    models). Distinguishing 14B from 32B is the whole point of the feature, so
    a plain base-name comparison would be wrong here.

    The size comparison reads the tag's DECLARED parameter count rather than
    looking it up in SIZE_ORDER: that hardcoded token list carried no entry for
    26b or 12b, so 'gemma4:26b-a4b-it-qat' and 'gemma4:12b' — the shipped
    default brain and the model right below it in the fallback chain — matched
    as the same model. That made the vision lockstep answer "already in
    lockstep" for the single most likely switch on this box (and made the voice
    switch reply "I'm already running on gemma 12B" while running the 26B).
    2026-08-20 audit."""
    if not a or not b:
        return False
    al, bl = a.lower(), b.lower()
    if al == bl:
        return True
    if a.split(":", 1)[0].lower() != b.split(":", 1)[0].lower():
        return False
    # Same base name: same model only if their declared sizes don't conflict.
    # One side without a declared size ('qwen2.5', 'gemma4:latest') carries no
    # contradicting information → treat as the same model, as before.
    sa, sb = declared_size_b(a), declared_size_b(b)
    if sa is None or sb is None:
        return True
    return sa == sb


def is_multimodal_tag(tag: str) -> bool:
    """OFFLINE guess at whether `tag` can serve vision as well as chat: the
    vision family markers plus the known multimodal CHAT families.

    This is the fallback the voice path already used when Ollama's /api/show
    was unreachable; the settings writers use it as their only check because a
    Save must not block on a network call."""
    t = (tag or "").strip().lower()
    if not t:
        return False
    return (any(m in t for m in VISION_MARKERS)
            or any(f in t for f in MULTIMODAL_CHAT_FAMILIES))


def vision_lockstep_decision(old_chat, new_chat, cur_vision,
                             is_multimodal=None) -> tuple[str | None, str]:
    """Decide what LOCAL_VISION_MODEL must become after a chat-model switch.

    Returns ``(new_vision_tag_or_None, reason)``. The tag is non-None ONLY for
    ``LOCKSTEP_SYNC``; every other reason means "leave vision alone", and the
    reason exists so a caller can SAY why rather than degrade silently.

    The three conditions are the ones skills/model_picker has enforced since
    the 2026-07-21 audit, unchanged:
      * sync only when the CURRENT vision tag denotes the OLD chat tag (the
        shared-brain config) — a user-pinned separate VLM is never touched;
      * nothing to do when vision already denotes the new tag;
      * a switch to a TEXT-ONLY brain leaves vision on the old multimodal tag
        (the monolith's residency/co-load guards then degrade with a printed
        reason instead of blinding local vision on a chat-only model).

    ``is_multimodal`` overrides the offline family check (the voice path
    passes model_picker's Ollama /api/show probe). Never raises: a probe that
    throws is treated as "unknown" and falls back to the offline check."""
    if not (isinstance(cur_vision, str) and cur_vision
            and isinstance(old_chat, str) and old_chat
            and isinstance(new_chat, str) and new_chat):
        return (None, LOCKSTEP_INCOMPLETE)
    if not same_model(cur_vision, old_chat):
        return (None, LOCKSTEP_PINNED)
    if same_model(cur_vision, new_chat):
        return (None, LOCKSTEP_ALREADY)
    check = is_multimodal or is_multimodal_tag
    try:
        capable = bool(check(new_chat))
    except Exception:
        capable = is_multimodal_tag(new_chat)
    if not capable:
        return (None, LOCKSTEP_TEXT_ONLY)
    return (new_chat, LOCKSTEP_SYNC)


def vision_tag_after_chat_switch(old_chat, new_chat, cur_vision,
                                 is_multimodal=None) -> str | None:
    """The tag LOCAL_VISION_MODEL must take after a chat switch, or None when
    the lockstep must NOT fire. Thin wrapper over vision_lockstep_decision for
    callers that don't need the reason."""
    return vision_lockstep_decision(old_chat, new_chat, cur_vision,
                                    is_multimodal=is_multimodal)[0]


def config_default(key: str, default=None):
    """The value core/config.py ships for ``key``, or ``default``.

    The settings DOCUMENT is not the whole story: ``core.config`` keeps a
    constant for every knob and ``_apply_user_settings`` only overrides the
    keys actually present in the file, so a key absent from
    data/user_settings.json is still LIVE at its config value. Anything that
    reasons about the effective settings (the GUI's VRAM budget, the lockstep
    writers) must resolve absent keys here rather than treating them as unset
    — that phantom-unset key is what made the Settings VRAM bar charge a
    7.3 GB VLM that the shipped config never loads.

    Imported lazily so this module stays import-cheap and stdlib-only for
    callers that never need it. Never raises."""
    try:
        import core.config as _cfg      # lazy: keeps this module dependency-free
    except Exception:
        return default
    try:
        val = getattr(_cfg, key, None)
    except Exception:
        return default
    return default if val is None else val
