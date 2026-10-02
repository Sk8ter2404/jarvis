"""core/brain_glow.py — which brain is answering, as a HUD colour + label.

The owner's ask (2026-10-02, from the Instagram JARVIS survey): when the active
brain changes — the local model, Claude Sonnet 5.5, Opus 5.5, any other model,
and per TURN when one turn is served by the cloud and the next locally — the
HUD orb / arc reactor glow takes that brain's colour (local = blue, Sonnet =
gold, Opus = violet...) and a brief label names it. The HUDs draw it as a
glowing RING of its own around the reactor: the halo keeps the state colour
(listening / thinking / speaking), so Sonnet's gold never reads as the
"thinking" gold (2026-10-02 review), and no ring is drawn asleep / in
standby, where the dim reactor is the cue.

Two halves, one module, stdlib only (imported by the HUD subprocesses, which
must import cleanly with nothing but the standard library):

  MAIN PROCESS (writer)
    * brain_state(model, route) — the ``brain`` dict written into
      hud_state.json: name, tier, color, model, route, source, changed_at,
      label_until, label.
    * BrainGlowPublisher / publish() — writes through _write_hud_state ONLY
      when the brain CHANGES. A turn on the same brain costs a tuple compare,
      never a file write (and the label does not re-flash every turn).
    * expected_brain(bc) — the brain the NEXT chat turn tries first, decided
      the way _call_llm decides it: the route predicate
      (_chat_takes_local_branch) first, then AI_BACKEND. Used at boot and by
      every switch site (set_model / set_brain / switch_llm), so the colour
      changes the moment the owner switches. The per-turn publish (in
      bobert_companion._call_llm) then shows the brain that REALLY answered —
      including a local turn the cloud had to answer and vice versa.
    * describe_for_voice() — the sentence current_model appends ("The
      reactor's glowing gold for Sonnet 5.5.").

  HUD SUBPROCESSES (reader)
    * hud_brain(hud_state, now) -> HudBrain | None. None for a missing, None
      or garbage key or an invalid colour, so a HUD with no brain information
      simply renders its normal state-coloured look. label_alpha fades the
      brief label out over the last LABEL_FADE_S seconds before label_until.

Settings (core/config.py, read live on every publish):
    BRAIN_GLOW_ENABLED  — master switch (default True). Off = nothing is
                          written; a glow already published is cleared once.
    BRAIN_GLOW_LABEL_S  — seconds the name label shows after a change
                          (default 4; 0 = colour only, never a label).
    BRAIN_GLOW_COLORS   — {tier: "#RRGGBB"} overrides, merged over
                          DEFAULT_COLORS (tiers: local, haiku, sonnet, opus,
                          fable, cloud). Bad values are ignored.
"""
from __future__ import annotations

import colorsys
import os
import re
import threading
import time
from collections import namedtuple
from typing import Any, Callable, Dict, Optional, Tuple

# ─── tiers + colours ─────────────────────────────────────────────────────────
#: Every tier, in the order the settings help lists them.
TIERS = ("local", "haiku", "sonnet", "opus", "fable", "cloud")

#: Default glow per tier. The alert red (#ff5b5b) is never a brain colour. A
#: tier may share a hue with a state colour (Sonnet gold ~ the "thinking"
#: gold): the HUDs draw the brain on a ring of its own and keep the halo for
#: the state, so the two never compete. Haiku is TEAL, not green (owner's
#: call, 2026-10-02): its old green #36D399 sat ~13 degrees of hue from the
#: unified HUD's listening green #78EBA8 and read as "listening"; #20B2AA is
#: ~32 degrees from that green and ~39 from the local blue
#: (tests/test_brain_glow.HaikuTealTests).
DEFAULT_COLORS: Dict[str, str] = {
    "local":  "#3D8BFF",   # blue   — the on-device model, $0 per turn
    "haiku":  "#20B2AA",   # teal   — fast / cheap cloud
    "sonnet": "#FFC233",   # gold   — the default cloud brain
    "opus":   "#B05CFF",   # violet — deep work
    "fable":  "#FF5C8A",   # rose   — the priciest
    "cloud":  "#C8D4E3",   # silver — any other cloud model
}

#: Spoken names of the default colours (an override is named by its hue).
_DEFAULT_WORDS = {
    "#3D8BFF": "blue", "#20B2AA": "teal", "#FFC233": "gold",
    "#B05CFF": "violet", "#FF5C8A": "rose", "#C8D4E3": "silver",
}

DEFAULT_LABEL_S = 4.0
LABEL_FADE_S = 1.0          # the label fades over its last second
_NAME_MAX = 24

_CLOUD_FAMILIES = ("haiku", "sonnet", "opus", "fable")
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def valid_color(value: Any) -> Optional[str]:
    """``"#RRGGBB"`` (upper-cased) when ``value`` is a 6-digit hex colour
    string, else None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v.upper() if _HEX_RE.match(v) else None


def resolve_colors(overrides: Any = None) -> Dict[str, str]:
    """DEFAULT_COLORS with the valid entries of ``overrides`` (a
    {tier: "#RRGGBB"} dict) merged over it. Unknown tiers and bad colours are
    dropped; a non-dict override is ignored entirely."""
    cols = {t: c.upper() for t, c in DEFAULT_COLORS.items()}
    if isinstance(overrides, dict):
        for tier, col in overrides.items():
            key = str(tier).strip().lower()
            ok = valid_color(col)
            if key in cols and ok:
                cols[key] = ok
    return cols


# ─── route / tier / name ─────────────────────────────────────────────────────
def _looks_cloud(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("claude") or any(f in m for f in _CLOUD_FAMILIES)


def normalize_route(route: Any, model: Any = "") -> str:
    """'local' or 'cloud'. ``route`` may be local/ollama or
    cloud/claude/anthropic (any case); a blank route is inferred from the
    model name (a claude-* / family-named model is cloud, anything else is a
    local Ollama tag)."""
    r = str(route or "").strip().lower()
    if r in ("local", "ollama"):
        return "local"
    if r in ("cloud", "claude", "anthropic"):
        return "cloud"
    return "cloud" if _looks_cloud(str(model or "")) else "local"


def tier_for(model: Any, route: Any = None) -> str:
    """The colour tier for a brain: 'local' for anything served locally, else
    the Claude family (haiku / sonnet / opus / fable), else 'cloud'."""
    m = str(model or "")
    if normalize_route(route, m) == "local":
        return "local"
    low = m.lower()
    for fam in _CLOUD_FAMILIES:
        if fam in low:
            return fam
    return "cloud"


_CLAUDE_RE = re.compile(r"claude-([a-z]+)-(\d{1,2})(?:-(\d{1,2}))?(?=$|-)")
_SIZE_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)b(?![a-z0-9])", re.I)


def display_name(model: Any, route: Any = None) -> str:
    """The short human name the HUD label and the voice line use:
    'claude-sonnet-5-5' -> 'Sonnet 5.5', 'gemma4:12b' -> 'gemma4 12B'.
    Bounded to 24 characters; never empty."""
    m = str(model or "").strip()
    if normalize_route(route, m) == "cloud":
        if not m:
            return "Claude"
        hit = _CLAUDE_RE.search(m.lower())
        if hit:
            fam, major, minor = hit.group(1), hit.group(2), hit.group(3)
            ver = f"{major}.{minor}" if minor else major
            return f"{fam.title()} {ver}"[:_NAME_MAX]
        for fam in _CLOUD_FAMILIES:
            if fam in m.lower():
                return fam.title()
        return m[:_NAME_MAX]
    if not m:
        return "local model"
    base, _, rest = m.partition(":")
    base = base.rstrip("/").rsplit("/", 1)[-1] or base
    size = _SIZE_RE.search(rest) if rest else None
    name = f"{base} {size.group(1)}B" if size else base
    return name[:_NAME_MAX]


def color_for(model: Any, route: Any = None,
              colors: Optional[Dict[str, str]] = None) -> str:
    """The glow colour ("#RRGGBB") for a brain. ``colors`` is a resolved
    table (resolve_colors); None = the defaults."""
    cols = colors if isinstance(colors, dict) else resolve_colors()
    tier = tier_for(model, route)
    return valid_color(cols.get(tier)) or DEFAULT_COLORS[tier].upper()


def color_word(color: Any) -> str:
    """A spoken word for a colour: the default palette by name, anything else
    by its hue ('red', 'orange', 'gold', 'green', 'teal', 'blue', 'violet',
    'pink', or 'white' / 'silver' for the unsaturated). '' when invalid."""
    c = valid_color(color)
    if c is None:
        return ""
    if c in _DEFAULT_WORDS:
        return _DEFAULT_WORDS[c]
    r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    h, lum, sat = colorsys.rgb_to_hls(r, g, b)
    if sat < 0.2 or lum > 0.92:
        return "white" if lum > 0.8 else "silver"
    deg = h * 360.0
    for limit, word in ((15, "red"), (40, "orange"), (65, "gold"),
                        (160, "green"), (195, "teal"), (250, "blue"),
                        (290, "violet"), (345, "pink")):
        if deg < limit:
            return word
    return "red"


# ─── the hud_state ``brain`` dict ────────────────────────────────────────────
def brain_state(model: Any, route: Any = None, *, source: str = "turn",
                label_s: float = DEFAULT_LABEL_S,
                colors: Optional[Dict[str, Any]] = None,
                now: Optional[float] = None) -> Dict[str, Any]:
    """The JSON-safe dict published as hud_state.json's ``brain``."""
    m = str(model or "").strip()
    rt = normalize_route(route, m)
    cols = resolve_colors(colors)
    name = display_name(m, rt)
    t = time.time() if now is None else float(now)
    try:
        ls = max(0.0, float(label_s))
    except (TypeError, ValueError):
        ls = DEFAULT_LABEL_S
    label = name.upper() if rt == "cloud" else f"LOCAL · {name.upper()}"
    return {
        "name": name,
        "tier": tier_for(m, rt),
        "color": color_for(m, rt, cols),
        "model": m,
        "route": rt,
        "source": str(source or "turn"),
        "changed_at": t,
        "label_until": t + ls,
        "label": label,
    }


# ─── HUD-side reader ─────────────────────────────────────────────────────────
HudBrain = namedtuple("HudBrain", "color name label label_alpha tier")


def hud_brain(hud: Any, now: Optional[float] = None) -> Optional[HudBrain]:
    """Parse hud_state's ``brain`` for a HUD. None (= draw the normal look)
    for a missing / None / non-dict key or a missing / invalid colour; never
    raises. label_alpha is 1.0 until LABEL_FADE_S before label_until, fades
    linearly to 0.0 at label_until, and stays 0.0 after it."""
    try:
        if not isinstance(hud, dict):
            return None
        b = hud.get("brain")
        if not isinstance(b, dict):
            return None
        color = valid_color(b.get("color"))
        if color is None:
            return None
        name = b.get("name")
        name = name[:_NAME_MAX] if isinstance(name, str) else ""
        label = b.get("label")
        label = label[:32] if isinstance(label, str) and label else name.upper()
        tier = b.get("tier")
        tier = tier if isinstance(tier, str) else ""
        t = time.time() if now is None else float(now)
        try:
            until = float(b.get("label_until"))
        except (TypeError, ValueError):
            until = 0.0
        if until != until:          # NaN
            until = 0.0
        remaining = until - t
        if remaining <= 0.0:
            alpha = 0.0
        elif remaining >= LABEL_FADE_S:
            alpha = 1.0
        else:
            alpha = remaining / LABEL_FADE_S
        return HudBrain(color, name, label, alpha, tier)
    except Exception:
        return None


# ─── settings ────────────────────────────────────────────────────────────────
def settings() -> Tuple[bool, float, Dict[str, Any]]:
    """(enabled, label_s, color_overrides) read LIVE from core.config, so a
    runtime flip of the constants takes effect on the next publish. Garbage
    values fall back to the defaults; never raises."""
    enabled, label_s, colors = True, DEFAULT_LABEL_S, {}
    try:
        import core.config as cfg
        enabled = bool(getattr(cfg, "BRAIN_GLOW_ENABLED", True))
        raw = getattr(cfg, "BRAIN_GLOW_LABEL_S", DEFAULT_LABEL_S)
        try:
            label_s = max(0.0, float(raw))
        except (TypeError, ValueError):
            label_s = DEFAULT_LABEL_S
        raw_cols = getattr(cfg, "BRAIN_GLOW_COLORS", {})
        colors = dict(raw_cols) if isinstance(raw_cols, dict) else {}
    except Exception:
        pass
    return enabled, label_s, colors


# ─── publisher ───────────────────────────────────────────────────────────────
class BrainGlowPublisher:
    """Writes the ``brain`` key only when the brain CHANGES.

    The writer (the monolith's _write_hud_state) runs under this object's lock
    so the file and last() can never disagree: two racing publishers could
    otherwise leave brain A on disk while last() says B, and every later
    publish of B would be skipped — a stale colour forever."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: Optional[Tuple[str, str]] = None
        self._published = False

    def last(self) -> Optional[Tuple[str, str]]:
        """(route, model) of the brain currently on the HUD, or None."""
        return self._last

    def reset(self) -> None:
        with self._lock:
            self._last = None
            self._published = False

    def publish(self, writer: Optional[Callable[..., Any]], model: Any,
                route: Any = None, *, source: str = "turn",
                enabled: bool = True, label_s: float = DEFAULT_LABEL_S,
                colors: Optional[Dict[str, Any]] = None,
                now: Optional[float] = None) -> bool:
        """Publish (route, model). True when hud_state was written. Never
        raises."""
        if not callable(writer):
            return False
        if not enabled:
            with self._lock:
                if not self._published:
                    return False
                self._published = False
                self._last = None
                try:
                    writer(brain=None)
                except Exception:
                    pass
            return True
        m = model.strip() if isinstance(model, str) else ""
        if not m:
            return False
        key = (normalize_route(route, m), m)
        with self._lock:
            if key == self._last:
                return False
            try:
                state = brain_state(m, key[0], source=source, label_s=label_s,
                                    colors=colors, now=now)
                writer(brain=state)
            except Exception:
                return False
            self._last = key
            self._published = True
            return True


PUBLISHER = BrainGlowPublisher()


def publish(writer: Optional[Callable[..., Any]], model: Any, route: Any = None,
            source: str = "turn", now: Optional[float] = None) -> bool:
    """Publish through the process-wide PUBLISHER with the live settings."""
    try:
        enabled, label_s, colors = settings()
        return PUBLISHER.publish(writer, model, route, source=source,
                                 enabled=enabled, label_s=label_s,
                                 colors=colors, now=now)
    except Exception:
        return False


# ─── the brain the next turn will use ────────────────────────────────────────
def _str_attr(obj: Any, name: str) -> str:
    v = getattr(obj, name, None) if obj is not None else None
    return v.strip() if isinstance(v, str) else ""


def _local_model(bc: Any) -> str:
    """The local tag the next local turn uses, in the resolver's own order
    minus its Ollama probe (the same order core.prompts.self_knowledge_facts
    reads): the resolver cache when warm, else the JARVIS_LOCAL_LLM_MODEL
    override (which the resolver puts first — at boot the cache is still
    cold), else the configured LOCAL_LLM_MODEL. Never a network call."""
    cache = getattr(bc, "_RESOLVED_LOCAL_LLM_MODEL", None) if bc is not None else None
    if isinstance(cache, list) and cache and isinstance(cache[0], str) and cache[0]:
        return cache[0]
    env = (os.environ.get("JARVIS_LOCAL_LLM_MODEL") or "").strip()
    if env:
        return env
    tag = _str_attr(bc, "LOCAL_LLM_MODEL")
    if tag:
        return tag
    try:
        import core.config as cfg
        return str(getattr(cfg, "LOCAL_LLM_MODEL", "") or "")
    except Exception:
        return ""


def _takes_local(bc: Any) -> bool:
    """_call_llm's route predicate: the monolith's own
    _chat_takes_local_branch() when there is one (the ONE copy of the rule),
    else MODEL_ROUTING['chat'] == 'local' from the monolith or core.config."""
    fn = getattr(bc, "_chat_takes_local_branch", None) if bc is not None else None
    if callable(fn):
        try:
            got = fn()
            if isinstance(got, bool):
                return got
        except Exception:
            pass
    routing = getattr(bc, "MODEL_ROUTING", None) if bc is not None else None
    if isinstance(routing, dict) and isinstance(routing.get("chat"), str):
        return routing["chat"].strip().lower() == "local"
    try:
        from core.config import model_route
        return model_route("chat") == "local"
    except Exception:
        return False


def expected_brain(bc: Any) -> Optional[Tuple[str, str]]:
    """(route, model) the NEXT chat turn tries first — _call_llm's order:
    the local route branch, else AI_BACKEND == 'claude' (cloud), else
    AI_BACKEND == 'ollama' (local). None when no backend is configured."""
    try:
        if _takes_local(bc):
            m = _local_model(bc)
            return ("local", m) if m else None
        # The monolith's wildcard-copied globals are the live values
        # (switch_llm mutates bc.AI_BACKEND, never core.config); core.config
        # only fills in what a partial stand-in lacks.
        backend = _str_attr(bc, "AI_BACKEND").lower()
        claude = _str_attr(bc, "CLAUDE_MODEL")
        if not backend or not claude:
            import core.config as cfg
            backend = backend or str(getattr(cfg, "AI_BACKEND", "") or "").lower()
            claude = claude or str(getattr(cfg, "CLAUDE_MODEL", "") or "")
        if backend == "claude":
            return ("cloud", claude) if claude else None
        if backend == "ollama":
            m = _local_model(bc)
            return ("local", m) if m else None
        return None
    except Exception:
        return None


def publish_expected(bc: Any, source: str = "switch") -> bool:
    """Publish expected_brain(bc) through bc._write_hud_state — the call
    every switch site and the boot path make. Never raises."""
    try:
        writer = getattr(bc, "_write_hud_state", None) if bc is not None else None
        if not callable(writer):
            return False
        exp = expected_brain(bc)
        if exp is None:
            return False
        return publish(writer, exp[1], exp[0], source=source)
    except Exception:
        return False


# ─── voice ───────────────────────────────────────────────────────────────────
def describe_for_voice(bc: Any = None, route: Any = None) -> str:
    """The sentence current_model appends (leading space), or '' when the
    glow is off. Names the colour on the HUD right now and the brain it
    stands for; when the last answer came from a different brain than the
    one configured (a fallback turn), says so. On the 'auto' route with the
    cloud glowing, adds the local colour too."""
    try:
        enabled, _label_s, overrides = settings()
        if not enabled:
            return ""
        cols = resolve_colors(overrides)
        expected = expected_brain(bc)
        last = PUBLISHER.last()
        current = last or expected
        if not current:
            return ""
        c_route, c_model = current
        word = color_word(color_for(c_model, c_route, cols))
        name = display_name(c_model, c_route)
        if not word:
            return ""
        if last and expected and last != expected:
            out = f" The reactor's glowing {word} right now: {name} answered last."
        else:
            out = f" The reactor's glowing {word} for {name}."
        if str(route or "").strip().lower() == "auto" and c_route == "cloud":
            local_word = color_word(cols.get("local"))
            if local_word and local_word != word:
                out += f" It turns {local_word} when the local model answers instead."
        return out
    except Exception:
        return ""
