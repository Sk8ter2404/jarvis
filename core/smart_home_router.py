"""
core.smart_home_router — dispatch smart-home commands to per-brand skills.

Reads the canonical device catalog at `data/smart_home_devices.json` produced
by the `skills.smart_home_discover` wizard, identifies which brand controller
skills are actually needed, dynamic-imports them, then dispatches voice
utterances (`'turn off the office light'`, `'set bedroom to 65'`,
`'dim the kitchen lights to 30%'`) to the correct brand skill's uniform
`set_state` / `get_state` / `list_devices` API.

Resolution order for a target device:
  1. Direct brand skill (skills/sh_<brand>.py) — preferred.
  2. Alexa cookie fallback via `alexapy.AlexaAPI.set_appliance_state` — only
     used if every direct path raises or returns an error.

If a discovered device's brand has no matching `skills/sh_<brand>.py`, a one-
liner `[TODO: build skill for brand X]` is logged and a self-implementing
task is appended to `jarvis_todo.md` (idempotent — duplicate markers are
skipped) so the upgrade pipeline can build the missing skill next pass.

Registered actions
------------------
    smart_home_control       — main entry; freeform utterance
    control_device           — alias
    smart_home_devices       — list catalog devices grouped by room
    smart_home_router_status — short status (brands loaded, fallback ready)
    refresh_smart_home_router — reload the catalog and brand modules

The router is intentionally tolerant: missing catalog, missing brand skill,
missing alexapy install, missing dependency for one brand skill — each
case degrades gracefully and surfaces a clear message to the LLM/user
rather than raising.
"""
from __future__ import annotations

import datetime
import importlib
import json
import os
import re
import threading
import time
from typing import Any, Callable


_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR     = os.path.join(_PROJECT_DIR, "data")
_CATALOG_PATH = os.path.join(_DATA_DIR, "smart_home_devices.json")
_TODO_PATH    = os.path.join(_PROJECT_DIR, "jarvis_todo.md")

# Brand → controller skill module name. Multiple brand strings can map to
# the same skill (e.g. Signify, Hue, Philips Hue → sh_hue).
_BRAND_TO_SKILL = {
    "philips hue":  "sh_hue",
    "signify":      "sh_hue",
    "hue":          "sh_hue",
    "tp-link":      "sh_kasa",
    "tplink":       "sh_kasa",
    "kasa":         "sh_kasa",
    "tapo":         "sh_kasa",
    "lifx":         "sh_lifx",
    "govee":        "sh_govee",
    "ecobee":       "sh_ecobee",
    "nest":         "sh_nest",
    "google nest":  "sh_nest",
    "ring":         "sh_ring",
}

# Coarse rgb tuples used when the user names a color. Keyed by the lowercase
# english name. Used for the color setpoint translation in `_extract_color`.
_NAMED_COLORS = {
    "red":     (255, 0, 0),
    "orange":  (255, 140, 0),
    "yellow":  (255, 230, 0),
    "green":   (0, 200, 0),
    "blue":    (0, 60, 255),
    "indigo":  (75, 0, 130),
    "violet":  (148, 0, 211),
    "purple":  (160, 32, 240),
    "pink":    (255, 105, 180),
    "magenta": (255, 0, 255),
    "cyan":    (0, 255, 255),
    "teal":    (0, 128, 128),
    "white":   (255, 255, 255),
    "warm":    (255, 187, 120),
    "cool":    (200, 220, 255),
}

# Lock state recognition / phrase templates.
_LOCK_VERBS   = {"lock"}
_UNLOCK_VERBS = {"unlock"}
# Tokens that imply a device should be turned off (or otherwise zeroed).
_OFF_TOKENS = {"off", "kill", "disable", "shut"}
_ON_TOKENS  = {"on", "enable", "activate", "switch"}

# How many seconds of grace before we refresh the catalog on every call.
_CATALOG_TTL_SECS = 5.0


# ── catalog loader ──────────────────────────────────────────────────
_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "catalog":      None,        # parsed JSON dict or None
    "loaded_at":    0.0,         # monotonic time
    "modules":      {},          # skill_name → imported module
    "missing_logged": set(),     # brands we already TODO'd this session
    "alexapy_login": None,       # restored alexapy login (lazy)
}


def _load_catalog_raw() -> dict | None:
    if not os.path.exists(_CATALOG_PATH):
        return None
    try:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [sh-router] catalog read failed: {e}")
        return None


def _ensure_catalog(force: bool = False) -> dict | None:
    """Return the in-memory catalog, reloading from disk if stale."""
    with _state_lock:
        cat = _state["catalog"]
        age = time.monotonic() - _state["loaded_at"]
        if cat is not None and not force and age < _CATALOG_TTL_SECS:
            return cat
        fresh = _load_catalog_raw()
        if fresh is None:
            _state["catalog"] = None
            return None
        _state["catalog"]   = fresh
        _state["loaded_at"] = time.monotonic()
        return fresh


# ── brand → skill resolution ────────────────────────────────────────
def _controller_for(brand: str) -> str | None:
    if not brand:
        return None
    b = brand.lower()
    for key, skill in _BRAND_TO_SKILL.items():
        if key in b:
            return skill
    return None


def _import_skill(skill_name: str) -> Any:
    """Import skills.sh_<brand>. Cached. Returns the module or None on
    failure (e.g. missing optional dependency)."""
    with _state_lock:
        mod = _state["modules"].get(skill_name)
        if mod is not None:
            return mod
    try:
        mod = importlib.import_module(f"skills.{skill_name}")
    except Exception as e:
        print(f"  [sh-router] could not import skills.{skill_name}: {e}")
        return None
    with _state_lock:
        _state["modules"][skill_name] = mod
    return mod


def _present_brands(catalog: dict) -> set[str]:
    """Set of unique brand strings present in the catalog."""
    out: set[str] = set()
    for d in catalog.get("devices", []) or []:
        b = (d.get("brand") or "").strip()
        if b:
            out.add(b)
    return out


def _present_skills(catalog: dict) -> dict[str, list[dict]]:
    """skill_name → [device_record, ...] for brands actually in the catalog
    that map to a known controller skill."""
    grouped: dict[str, list[dict]] = {}
    for d in catalog.get("devices", []) or []:
        skill = d.get("controller_skill") or _controller_for(d.get("brand") or "")
        if not skill:
            continue
        grouped.setdefault(skill, []).append(d)
    return grouped


def _log_missing_brand(brand: str) -> None:
    """Print the `[TODO: build skill for brand X]` line and append a self-
    implementing task to `jarvis_todo.md` (idempotent within the session
    AND against the file's existing contents)."""
    if not brand:
        return
    key = brand.strip().lower()
    with _state_lock:
        if key in _state["missing_logged"]:
            return
        _state["missing_logged"].add(key)
    print(f"  [sh-router] [TODO: build skill for brand {brand}]")
    if not os.path.exists(_TODO_PATH):
        return
    try:
        with open(_TODO_PATH, "r", encoding="utf-8") as f:
            existing = f.read()
    except Exception:
        return
    marker = f"[sh-router] Build controller skill for brand '{brand}'"
    if marker in existing:
        return
    slug = re.sub(r"[^a-z0-9]+", "_", brand.lower()).strip("_") or "unknown"
    today = datetime.date.today().isoformat()
    line = (
        f"\n- [ ] **{today} sh-router** - {marker}. Add `skills/sh_{slug}.py` "
        f"with the uniform set_state / get_state / list_devices interface. "
        f"Brand observed in `data/smart_home_devices.json` at runtime but no "
        f"controller skill exists yet — discovery wizard found it via Alexa "
        f"but direct LAN control is missing.\n"
    )
    try:
        with open(_TODO_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"  [sh-router] todo append failed: {e}")


# ── utterance parsing ───────────────────────────────────────────────
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}

# Filler words stripped before device-name matching so 'turn off the office
# light' and 'turn off office light' both resolve identically.
_FILLER_PREFIXES = (
    "could you ", "can you ", "please ", "jarvis ", "hey jarvis ",
    "would you ", "for me ", "the ", "a ", "an ",
)


def _strip_filler(s: str) -> str:
    out = s.strip().lower()
    changed = True
    while changed:
        changed = False
        for f in _FILLER_PREFIXES:
            if out.startswith(f):
                out = out[len(f):]
                changed = True
                break
    return out.strip()


def _parse_number(token: str) -> int | None:
    """Translate one number token (digit string OR english word) to int."""
    t = token.strip().lower().rstrip("°%")
    if not t:
        return None
    if t.isdigit():
        try:
            return int(t)
        except ValueError:
            return None
    return _NUMBER_WORDS.get(t)


def _parse_spoken_number(value: str) -> int | None:
    """Parse a possibly-COMPOUND spoken number: '65', 'sixty five' → 65,
    'seventy two' → 72, 'one hundred' / 'a hundred' / 'hundred' → 100.

    2026-07-07 bug-hunt (MED): the bare-number branch parsed only value.split()[0],
    so a spelled-out compound like 'set the bedroom to sixty five' collapsed to 60
    (a thermostat/brightness set to the WRONG value — spelled-out numbers are the
    norm from a voice front-end). Folds tens+ones and the 'hundred' multiplier;
    stops at the first non-number token so trailing units don't corrupt it. Returns
    None when no leading number word is present."""
    if not value:
        return None
    current = 0
    saw = False
    for tok in value.strip().lower().split():
        n = _parse_number(tok)
        if n is None:
            break                      # 'sixty five foo' → 65 (stop at 'foo')
        saw = True
        if n == 100:
            current = max(current, 1) * 100   # 'one hundred' → 100; 'hundred' → 100
        else:
            current += n               # 'sixty' then 'five' → 65
    return current if saw else None


def _extract_percent(text: str) -> int | None:
    """Find a percent expression: '30%', '30 percent', 'thirty percent'."""
    # `\b` must guard ONLY the word 'percent'. A literal '%' is itself a
    # non-word char, so a trailing `\b` after it can never match — '30%',
    # '30% now' and '50%.' all sit at a non-word/non-word edge (no boundary),
    # which silently killed the '%' branch and dropped brightness from
    # phrasings like 'set the office to 75%'.
    m = re.search(r"(\d{1,3})\s*(?:%|percent\b)", text)
    if m:
        try:
            n = int(m.group(1))
            return max(0, min(100, n))
        except ValueError:  # pragma: no cover - unreachable: int() on a \d+ regex group (Unicode Nd) always parses; only superscript/No digits raise and \d never matches those
            return None  # pragma: no cover - see above
    m = re.search(r"\b([a-z\-]+)\s+percent\b", text)
    if m:
        n = _parse_number(m.group(1))
        if n is not None:
            return max(0, min(100, n))
    return None


def _extract_temperature(text: str) -> int | None:
    """Find a thermostat temperature: '65', '65 degrees', '72 F', 'sixty five'."""
    m = re.search(r"(?:to|at)\s+(\d{2,3})\s*(?:°|degrees?|deg|f|fahrenheit|c|celsius)?\b", text)
    if m:
        try:
            n = int(m.group(1))
            if 40 <= n <= 110:
                return n
        except ValueError:  # pragma: no cover - unreachable: int() on a \d{2,3} regex group (Unicode Nd) always parses; \d never matches superscript/No digits
            pass  # pragma: no cover - see above
    m = re.search(r"\b(\d{2,3})\s*(?:°|degrees?|deg)\b", text)
    if m:
        try:
            n = int(m.group(1))
            if 40 <= n <= 110:
                return n
        except ValueError:  # pragma: no cover - unreachable: int() on a \d{2,3} regex group (Unicode Nd) always parses; \d never matches superscript/No digits
            pass  # pragma: no cover - see above
    return None


def _extract_color(text: str) -> tuple[str, tuple[int, int, int]] | None:
    """If the user mentioned a named color, return (name, rgb). None otherwise."""
    for name, rgb in _NAMED_COLORS.items():
        if re.search(rf"\b{name}\b", text):
            return (name, rgb)
    return None


def _extract_color_temperature(text: str) -> int | None:
    """Find a color-temperature expression: '2700K', '2700 K', '2700 kelvin',
    '4000-kelvin'. Returns Kelvin clamped to the Hue/Govee safe range
    2000..6500. None if no match."""
    if not text:
        return None
    m = re.search(r"\b(\d{4})\s*-?\s*(?:k|kelvin)\b", text, re.IGNORECASE)
    if not m:
        return None
    try:
        k = int(m.group(1))
    except ValueError:  # pragma: no cover - unreachable: int() on a \d{4} regex group (Unicode Nd) always parses; \d never matches superscript/No digits
        return None  # pragma: no cover - see above
    return max(2000, min(6500, k))


def _classify_action(utterance: str) -> dict[str, Any]:
    """Convert a freeform utterance into a structured request.

    Returns a dict with keys (any may be None / absent):
        verb        : 'on' | 'off' | 'set' | 'lock' | 'unlock' | 'scene'
        brightness  : 0..100 or None
        temperature : int or None
        color       : (name, rgb) or None
        descriptor  : the rest of the utterance after action words, used
                      to match against device names/rooms.
    """
    raw = _strip_filler(utterance or "")
    if not raw:
        return {"verb": None, "descriptor": ""}

    out: dict[str, Any] = {"verb": None, "descriptor": raw}

    # INTERROGATIVE GUARD (2026-07-07 bug-hunt, HIGH). A STATUS QUESTION —
    # "are the lights on", "is the office light on" — must NEVER be parsed as an
    # on/off COMMAND. The bare "(.+?)\s+(on|off)$" suffix pattern below otherwise
    # matches "are the lights on" → verb='on' and ACTUALLY SWITCHES THE DEVICE ON
    # when the user only ASKED. (_strip_filler removes "the/please/…" prefixes but
    # not "are/is", so the interrogative survived to the suffix match; voice
    # transcription frequently drops the trailing '?', so the punctuation-less
    # form is the common one.) A leading question word or a trailing '?' marks a
    # query → verb='query', so smart_home_control reports rather than toggling.
    q = re.match(r"^(?:are|is|was|were|do|does|did|has|have|can|could|"
                 r"what'?s|what\s+is|how'?s|how\s+is|whats)\b\s*(.*)$", raw)
    if q is not None or raw.endswith("?"):
        rest = (q.group(1) if q is not None else raw).rstrip("?").strip()
        # Drop a trailing state word ("… lights on?") + re-strip filler so the
        # descriptor is just the device/room to match against the catalog.
        rest = re.sub(r"\s+(on|off|locked|unlocked|open|closed)$", "", rest)
        out["verb"] = "query"
        out["descriptor"] = _strip_filler(rest)
        return out

    # Lock first (so 'lock the front door' doesn't get reduced to 'on').
    if re.match(r"^(?:lock)\b", raw):
        out["verb"] = "lock"
        out["descriptor"] = re.sub(r"^lock\s+", "", raw)
        return out
    if re.match(r"^(?:unlock)\b", raw):
        out["verb"] = "unlock"
        out["descriptor"] = re.sub(r"^unlock\s+", "", raw)
        return out

    # Scene / 'run' / 'activate <scene name>'
    m = re.match(r"^(?:run|activate|trigger|start)\s+(?:the\s+)?(.+?)(?:\s+scene)?$", raw)
    if m and "scene" in raw:
        out["verb"] = "scene"
        out["descriptor"] = m.group(1)
        return out

    # 'turn on/off X' or 'X on/off'
    m = re.match(r"^turn\s+(on|off)\s+(.+)$", raw)
    if m:
        out["verb"] = "on" if m.group(1) == "on" else "off"
        out["descriptor"] = m.group(2)
        return out
    m = re.match(r"^(?:switch|flip)\s+(on|off)\s+(.+)$", raw)
    if m:
        out["verb"] = "on" if m.group(1) == "on" else "off"
        out["descriptor"] = m.group(2)
        return out
    m = re.match(r"^(.+?)\s+(on|off)$", raw)
    if m:
        out["verb"] = "on" if m.group(2) == "on" else "off"
        out["descriptor"] = m.group(1)
        return out

    # 'dim X to N%' / 'brighten X'
    m = re.match(r"^(?:dim|brighten)\s+(.+)$", raw)
    if m:
        rest = m.group(1)
        pct = _extract_percent(rest)
        out["verb"] = "set"
        out["brightness"] = pct if pct is not None else (30 if raw.startswith("dim") else 100)
        # Pick up color_temperature and color BEFORE stripping the percent
        # expression — the strip below would eat trailing modifiers like
        # 'warm 2700K' that come after the percentage.
        kelvin = _extract_color_temperature(rest)
        if kelvin is not None:
            out["color_temperature"] = kelvin
        else:
            color = _extract_color(rest)
            if color:
                out["color"] = color
        # Strip the percent expression from descriptor for matching.
        desc = re.sub(r"\s+(?:to\s+)?\d{1,3}\s*(?:%|percent).*$", "", rest).strip()
        out["descriptor"] = desc
        return out

    # 'set X to Y' — Y can be temperature, brightness%, or color
    m = re.match(r"^(?:set|change|adjust|put)\s+(.+?)\s+(?:to|on)\s+(.+)$", raw)
    if m:
        desc, value = m.group(1).strip(), m.group(2).strip()
        out["verb"] = "set"
        out["descriptor"] = desc
        # Try color first ('set the bedroom to blue')
        color = _extract_color(value)
        if color:
            out["color"] = color
            return out
        pct = _extract_percent(value)
        if pct is not None:
            out["brightness"] = pct
            return out
        temp = _extract_temperature("to " + value)
        if temp is not None:
            out["temperature"] = temp
            return out
        # Bare number → could be temperature (>=40) or brightness (<=100).
        # Compound-aware so 'sixty five' → 65 (not 60 from the first token).
        n = _parse_spoken_number(value)
        if n is not None:
            if 40 <= n <= 110:
                out["temperature"] = n
            else:
                out["brightness"] = max(0, min(100, n))
            return out
        return out

    # 'make X blue' / 'change X to blue' (already handled above for 'change to')
    m = re.match(r"^make\s+(.+)$", raw)
    if m:
        rest = m.group(1)
        color = _extract_color(rest)
        if color:
            out["verb"] = "set"
            out["color"] = color
            out["descriptor"] = re.sub(rf"\b{color[0]}\b", "", rest).strip()
            return out
        pct = _extract_percent(rest)
        if pct is not None:
            out["verb"] = "set"
            out["brightness"] = pct
            out["descriptor"] = re.sub(r"\s+(?:to\s+)?\d{1,3}\s*(?:%|percent).*$", "", rest).strip()
            return out

    return out


# ── device matching ─────────────────────────────────────────────────
_STOPWORDS = {
    "the", "a", "an", "my", "our", "please", "all", "every", "lights",
    "light", "bulb", "bulbs", "lamp", "lamps", "switch", "switches",
    "plug", "plugs",
}


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower())]


def _device_text(d: dict) -> str:
    bits = [
        d.get("name") or "",
        d.get("alexa_room") or "",
        " ".join(d.get("alexa_groups") or []),
        d.get("type") or "",
    ]
    return " ".join(b for b in bits if b)


def _match_score(descriptor: str, device: dict) -> float:
    """Token-overlap score in [0, 1+] between descriptor and device text.
    A bonus is added for exact name substring match so 'kitchen light'
    beats 'office kitchen light' for a device literally named 'kitchen'."""
    desc_tokens = [t for t in _tokens(descriptor) if t not in _STOPWORDS]
    if not desc_tokens:
        return 0.0
    dev_tokens = set(_tokens(_device_text(device)))
    if not dev_tokens:
        return 0.0
    overlap = sum(1 for t in desc_tokens if t in dev_tokens)
    score = overlap / max(1, len(desc_tokens))
    # Bonus for tight name match.
    name_low = (device.get("name") or "").lower()
    desc_low = (descriptor or "").lower()
    if name_low and name_low in desc_low:
        score += 0.5
    return score


def _resolve_devices(descriptor: str, catalog: dict,
                     want_type: str | None = None) -> list[dict]:
    """Return zero or more catalog device dicts that match `descriptor`.
    Multiple results are returned when the user clearly addressed a whole
    room ('turn off the kitchen lights')."""
    devices = catalog.get("devices", []) or []
    if not devices:
        return []
    scored = [(d, _match_score(descriptor, d)) for d in devices]
    scored = [(d, s) for d, s in scored if s > 0.3]
    if not scored:
        return []
    if want_type:
        typed = [(d, s) for d, s in scored if d.get("type") == want_type]
        if typed:
            scored = typed
    scored.sort(key=lambda ds: -ds[1])
    # If the descriptor is plural (mentions "lights"/"all"/"every"), fan out to
    # the whole matching room: every matched device sharing the top match's
    # room and type. (Don't filter on score equality — the name-match bonus in
    # _match_score means roommates can score differently yet still belong.)
    plural = any(w in (descriptor or "").lower()
                 for w in ("lights", "lamps", "all ", "every "))
    if plural:
        room_hint = None
        # Pick the room of the top match; widen to every device in that room
        # whose type matches the top match's type.
        top = scored[0][0]
        room_hint = (top.get("alexa_room") or "").lower()
        want_type2 = top.get("type")
        wide = [d for d, _ in scored
                if (d.get("alexa_room") or "").lower() == room_hint
                and d.get("type") == want_type2]
        if len(wide) >= 1:
            return wide
    return [scored[0][0]]


# ── brand-skill dispatch ────────────────────────────────────────────
def _call_skill(skill_name: str, device: dict, **kwargs) -> dict:
    """Invoke skills.<skill_name>.set_state(device, **kwargs). Returns the
    dict the skill returned, or {'error': ...} if anything went sideways."""
    mod = _import_skill(skill_name)
    if mod is None:
        return {"error": f"skill {skill_name} not loadable"}
    fn = getattr(mod, "set_state", None)
    if not callable(fn):
        return {"error": f"skill {skill_name} has no set_state()"}
    try:
        result = fn(device, **kwargs)
    except Exception as e:
        return {"error": f"{skill_name}.set_state raised: {e}"}
    if not isinstance(result, dict):
        return {"ok": True, "raw": str(result)}
    return result


def _action_to_kwargs(action: dict) -> dict[str, Any]:
    """Translate the parsed action dict into the kwargs each brand skill's
    set_state expects."""
    out: dict[str, Any] = {}
    verb = action.get("verb")
    if verb == "on":
        out["on"] = True
    elif verb == "off":
        out["on"] = False
    elif verb == "lock":
        out["locked"] = True
    elif verb == "unlock":
        out["locked"] = False
    if "brightness" in action and action["brightness"] is not None:
        out["brightness"] = int(action["brightness"])
        out.setdefault("on", out.get("on", True) if action.get("brightness", 0) > 0 else False)
    if "temperature" in action and action["temperature"] is not None:
        out["temperature"] = int(action["temperature"])
    if action.get("color"):
        name, rgb = action["color"]
        out["color"] = rgb
        out["color_name"] = name
    if action.get("color_temperature") is not None:
        out["color_temperature"] = int(action["color_temperature"])
    return out


# ── Alexa fallback ──────────────────────────────────────────────────
def _alexa_login() -> Any:
    """Lazy-restore the alexapy login from the cached cookie. Cached for
    the lifetime of the process; if the cookie is bad we don't keep
    re-trying it on every command."""
    with _state_lock:
        cached = _state.get("alexapy_login")
        if cached is not None:
            return cached if cached != "_failed" else None
    try:
        from skills import smart_home_discover as _sh  # type: ignore
    except Exception as e:
        print(f"  [sh-router] alexa fallback unavailable: {e}")
        with _state_lock:
            _state["alexapy_login"] = "_failed"
        return None
    try:
        login = _sh._restore_login_from_cookie()
    except Exception as e:
        print(f"  [sh-router] alexa fallback login failed: {e}")
        login = None
    with _state_lock:
        _state["alexapy_login"] = login if login is not None else "_failed"
    return login


def _alexa_set_state(device: dict, kwargs: dict) -> dict:
    """Last-resort: drive the device through Alexa's smart-home graph using
    the cached cookie. Only on_off / brightness / setpoint can be sent via
    this path reliably; color/scene fall back to a polite refusal."""
    entity_id = device.get("alexa_entity_id")
    if not entity_id:
        return {"error": "no alexa entity id on device"}
    login = _alexa_login()
    if login is None:
        return {"error": "alexa fallback unavailable (no cookie / alexapy)"}
    try:
        import alexapy  # type: ignore
    except Exception as e:
        return {"error": f"alexapy not importable: {e}"}
    AlexaAPI = getattr(alexapy, "AlexaAPI", None)
    if AlexaAPI is None:
        return {"error": "alexapy.AlexaAPI missing"}

    # Pick the simplest legal call shape for what was requested.
    target: str | None = None
    if kwargs.get("on") is True:
        target = "ON"
    elif kwargs.get("on") is False:
        target = "OFF"
    if target is None and "brightness" in kwargs:
        target = "ON"

    # alexapy is async; reuse the discover skill's runner so we don't
    # repeat the event-loop dance here.
    try:
        from skills.smart_home_discover import _run_async as _run  # type: ignore
    except Exception:
        return {"error": "asyncio runner unavailable"}

    async def _go() -> dict:
        try:
            if target and hasattr(AlexaAPI, "set_appliance_state"):
                await AlexaAPI.set_appliance_state(login, entity_id, target)
                return {"ok": True, "path": "alexa", "set": target}
            return {"error": "no compatible alexapy call available"}
        except Exception as e:
            return {"error": f"alexa call failed: {e}"}

    try:
        return _run(_go())
    except Exception as e:
        return {"error": f"alexa fallback runner failed: {e}"}


# ── one-device dispatch ─────────────────────────────────────────────
def _dispatch_one(device: dict, action: dict) -> dict:
    """Drive one device through its brand skill, with Alexa fallback on
    any failure."""
    kwargs = _action_to_kwargs(action)
    if not kwargs:
        return {"error": "nothing to do (action had no recognised parameters)"}

    brand = device.get("brand") or ""
    skill_name = device.get("controller_skill") or _controller_for(brand)
    if not skill_name:
        _log_missing_brand(brand)
        result = _alexa_set_state(device, kwargs)
        result["device"] = device.get("name")
        result["path"] = result.get("path") or ("alexa" if "ok" in result else "alexa-failed")
        return result

    result = _call_skill(skill_name, device, **kwargs)
    if "error" not in result:
        result["device"] = device.get("name")
        result["path"] = f"direct/{skill_name}"
        return result

    # Direct path failed → try Alexa.
    direct_err = result["error"]
    fallback = _alexa_set_state(device, kwargs)
    if "error" not in fallback:
        fallback["device"]   = device.get("name")
        fallback["path"]     = f"alexa-after-{skill_name}-fail"
        fallback["direct_error"] = direct_err
        return fallback
    return {
        "error":  f"direct ({direct_err}); alexa ({fallback.get('error')})",
        "device": device.get("name"),
        "path":   f"failed/{skill_name}",
    }


# ── public API ──────────────────────────────────────────────────────
# Reentrancy guard for the pointing hook: skills/kinect_pointing fires the
# resolved command back through this same function ('turn on desk lamp'), and
# that resolved utterance is never a pronoun so the hook below won't re-enter —
# but a thread-local flag makes that guarantee explicit and cheap so a future
# phrasing change can't introduce a loop.
_pointing_hook_active = threading.local()


def _try_pointing_resolution(utterance: str) -> str | None:
    """Best-effort: when `utterance` is an ambiguous pronoun on/off command
    ('turn that on', 'that one off') AND point-to-control is active and the user
    is pointing at a calibrated device, execute it via pointing and return the
    spoken result. Returns None otherwise so the caller's normal (named-device /
    ask-which) path is completely unchanged. Never raises.

    Imported lazily so the router has no hard dependency on the pointing skill,
    and guarded against reentry from the skill's own call back into us."""
    if getattr(_pointing_hook_active, "on", False):
        return None
    try:
        kp = importlib.import_module("skills.kinect_pointing")
    except Exception:
        return None
    checker = getattr(kp, "is_pronoun_device_command", None)
    resolver = getattr(kp, "resolve_pointing_command", None)
    if not callable(checker) or not callable(resolver):
        return None
    try:
        if not checker(utterance):
            return None
    except Exception:
        return None
    _pointing_hook_active.on = True
    try:
        return resolver(utterance)
    except Exception:
        return None
    finally:
        _pointing_hook_active.on = False


def smart_home_control(utterance: str = "") -> str:
    """Voice / LLM entry point. Parses `utterance`, finds the device(s),
    dispatches via the right brand skill (with Alexa fallback)."""
    if not utterance or not utterance.strip():
        return ("I need something to do, sir — try 'turn off the office "
                "light' or 'set the bedroom to 65'.")

    # Point-to-control hook: an ambiguous "turn that on" resolves via the Kinect
    # pointing direction BEFORE we fall back to asking which device. Best-effort
    # and non-breaking — only fires for bare pronoun commands when point-control
    # is enabled and the point resolves to a calibrated target; otherwise None
    # and the existing flow proceeds untouched.
    pointed = _try_pointing_resolution(utterance)
    if pointed is not None:
        return pointed

    catalog = _ensure_catalog()
    if catalog is None or not catalog.get("devices"):
        return ("No smart-home catalog yet, sir. "
                "Say 'discover smart home devices' to run the wizard.")

    action = _classify_action(utterance)

    # STATUS QUERY ("are the lights on?"): never toggle — resolve the device and
    # answer honestly. The router has no live state-read path across brands yet,
    # so we confirm we found the device and offer to act, rather than switching
    # it (the 2026-07-07 HIGH bug was answering this question by turning it ON).
    if action.get("verb") == "query":
        descriptor = action.get("descriptor") or ""
        devices = _resolve_devices(descriptor, catalog)
        if not devices:
            return (f"I don't see anything in the catalog matching "
                    f"'{descriptor}', sir.")
        name = devices[0].get("name") or descriptor or "that device"
        # Phrasing deliberately AVOIDS the failure-marker substrings in
        # core/failure_markers.FAILURE_MARKERS ("can't"/"couldn't"/…). This is a
        # SUCCESS reply (an honest status answer), but it is spoken via the
        # verbatim path and classified by _is_failure/_is_failure_result on
        # those substrings; "can't read its live state" used to match "can't",
        # so the honest answer was suppressed AND misclassified as a failed
        # action (extra LLM round-trip). "I'm not able to" says the same thing
        # with no marker. 2026-07-08.
        return (f"I found {name}, sir, but I'm not able to read its live state "
                f"from here yet — I can turn it on or off if you'd like.")

    if not action.get("verb"):
        return f"I couldn't parse that as a smart-home command, sir: '{utterance}'"

    descriptor = action.get("descriptor") or ""

    # Type hint helps with multi-device rooms ('set bedroom to 65' →
    # thermostat, not the bedroom lamp).
    want_type = None
    if "temperature" in action and action["temperature"] is not None:
        want_type = "thermostat"
    elif action.get("verb") in ("lock", "unlock"):
        want_type = "lock"
    elif "color" in action or "brightness" in action or "color_temperature" in action:
        want_type = "light"

    devices = _resolve_devices(descriptor, catalog, want_type=want_type)
    if not devices:
        return f"I don't see anything in the catalog matching '{descriptor}', sir."

    results = [_dispatch_one(d, action) for d in devices]
    return _summarize_results(action, results)


def _summarize_results(action: dict, results: list[dict]) -> str:
    """One-line summary suitable for TTS."""
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    n = len(results)
    if not ok and bad:
        first = bad[0]
        return (f"That didn't work, sir — {first.get('device','device')} "
                f"reported: {first['error']}")
    verb = action.get("verb")
    descriptor = action.get("descriptor") or "device"
    if verb == "on" and "brightness" not in action:
        if n == 1:
            return f"On, sir — {ok[0].get('device', descriptor)}."
        return f"On, sir — {n} {descriptor} devices."
    if verb == "off":
        if n == 1:
            return f"Off, sir — {ok[0].get('device', descriptor)}."
        return f"Off, sir — {n} {descriptor} devices."
    if "brightness" in action:
        return (f"Set to {action['brightness']}%, sir "
                f"— {len(ok)}/{n} device(s).")
    if "temperature" in action:
        return f"Setpoint at {action['temperature']}, sir."
    if action.get("color"):
        return f"Color set to {action['color'][0]}, sir."
    if verb == "lock":
        return f"Locked, sir — {ok[0].get('device', descriptor)}."
    if verb == "unlock":
        return f"Unlocked, sir — {ok[0].get('device', descriptor)}."
    if verb == "scene":
        return f"Scene running, sir — {ok[0].get('device', descriptor)}."
    return f"Done, sir — {len(ok)}/{n} device(s)."


def smart_home_devices(_: str = "") -> str:
    """Speakable summary of the cached catalog grouped by room."""
    catalog = _ensure_catalog()
    if catalog is None:
        return "No smart-home catalog yet, sir — run 'discover smart home devices'."
    rooms: dict[str, list[str]] = {}
    for d in catalog.get("devices", []) or []:
        room = d.get("alexa_room") or "(unassigned)"
        rooms.setdefault(room, []).append(d.get("name") or "(unnamed)")
    if not rooms:
        return "The catalog is empty, sir."
    parts = [f"{len(names)} in {room}" for room, names in sorted(rooms.items())]
    return f"{catalog.get('device_count', 0)} devices, sir: " + ", ".join(parts) + "."


def smart_home_router_status(_: str = "") -> str:
    """Short status — which brand skills are loaded and the fallback state."""
    catalog = _ensure_catalog()
    if catalog is None:
        return "Catalog not loaded, sir."
    grouped = _present_skills(catalog)
    loaded = []
    missing_skill = []
    for skill_name, devs in grouped.items():
        mod = _import_skill(skill_name)
        avail = False
        if mod is not None:
            fn = getattr(mod, "is_available", None)
            try:
                avail = bool(fn()) if callable(fn) else True
            except Exception:
                avail = False
        if avail:
            loaded.append(f"{skill_name}({len(devs)})")
        else:
            missing_skill.append(f"{skill_name}({len(devs)})")
    missing_brands = sorted({
        (d.get("brand") or "?") for d in catalog.get("devices", []) or []
        if not (d.get("controller_skill") or _controller_for(d.get("brand") or ""))
    })
    bits = []
    if loaded:
        bits.append("active: " + ", ".join(loaded))
    if missing_skill:
        bits.append("dep-missing: " + ", ".join(missing_skill))
    if missing_brands:
        bits.append("no-skill: " + ", ".join(missing_brands))
    return "Smart-home router — " + ("; ".join(bits) or "no devices, sir.")


def refresh_smart_home_router(_: str = "") -> str:
    with _state_lock:
        _state["catalog"]      = None
        _state["loaded_at"]    = 0.0
        _state["modules"]      = {}
        _state["alexapy_login"] = None
    cat = _ensure_catalog(force=True)
    if cat is None:
        return "No catalog on disk yet, sir."
    grouped = _present_skills(cat)
    return (f"Router refreshed, sir: {cat.get('device_count', 0)} devices "
            f"across {len(grouped)} brand controller(s).")


def warm_up() -> None:
    """Pre-import the controller skills needed by the catalog. Called at
    JARVIS start-up via `register()` so the first voice command isn't
    delayed by a cold import."""
    catalog = _ensure_catalog()
    if catalog is None:
        return
    grouped = _present_skills(catalog)
    for skill_name, devs in grouped.items():
        _import_skill(skill_name)
    # Log TODOs for brands present in the catalog but with no skill.
    for d in catalog.get("devices", []) or []:
        if d.get("controller_skill") or _controller_for(d.get("brand") or ""):
            continue
        _log_missing_brand((d.get("brand") or "").strip())


# ── action registration ─────────────────────────────────────────────
def register(actions: dict[str, Callable[[str], str]]) -> None:
    """Called by skills/smart_home_router_skill.py (or directly from
    bobert_companion at boot)."""
    actions["smart_home_control"]        = smart_home_control
    actions["control_device"]            = smart_home_control
    actions["control_smart_home"]        = smart_home_control
    actions["smart_home_devices"]        = smart_home_devices
    actions["smart_home_list"]           = smart_home_devices
    actions["smart_home_router_status"]  = smart_home_router_status
    actions["refresh_smart_home_router"] = refresh_smart_home_router
    try:
        warm_up()
    except Exception as e:
        print(f"  [sh-router] warm_up failed: {e}")
