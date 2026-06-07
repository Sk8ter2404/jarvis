"""
Chappie Mode v1 — JARVIS continuous self-learning.

Like the robot in the film, JARVIS quietly absorbs what happens around him,
forms episodes from the raw ambient transcripts, distills facts about the
people and things in the user's life, and never volunteers what he's learned
unless explicitly asked.

Three background layers run on a daemon thread:
  • Layer 2  — quality filter on raw transcripts (Whisper noise dropped).
  • Layer 3  — episode grouping every EPISODE_INTERVAL_SEC, written to
               data/chappie_episodes.jsonl with a one-line Claude summary +
               topics + mood + new_entities.
  • Layer 4  — entity-level fact accumulation every FACT_INTERVAL_SEC,
               written to data/chappie_facts.json. Open questions sit on
               each entity as the "curiosity flag" the spec described.

Recall is silent unless invoked. Two actions are exported:
  • chappie_recall_entity   — "what do you know about X"
  • chappie_recall_today    — "what did you overhear today / did I mention X"

NO proactive_announce calls. NO TTS queueing. Ever. Chappie writes to his
own brain files; he never touches the user-facing memory.json. A bug here
can only cost API budget; it can't corrupt the rest of JARVIS.

Design doc: docs/CHAPPIE_MODE_SPEC.md
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime

# ─── paths ──────────────────────────────────────────────────────────────
_PROJECT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRANSCRIPTS_FILE = os.path.join(_PROJECT_DIR, "data", "ambient_transcripts.jsonl")
_EPISODES_FILE    = os.path.join(_PROJECT_DIR, "data", "chappie_episodes.jsonl")
_FACTS_FILE       = os.path.join(_PROJECT_DIR, "data", "chappie_facts.json")
_CURSORS_FILE     = os.path.join(_PROJECT_DIR, "data", "chappie_cursors.json")

# ─── config ─────────────────────────────────────────────────────────────
EPISODE_INTERVAL_SEC = 300       # build episodes every 5 min
FACT_INTERVAL_SEC    = 1800      # roll facts every 30 min
EPISODE_GAP_SEC      = 30        # utterances within 30s collapse into one episode
EPISODE_MAX_UTT      = 40        # cap utterances per episode to keep prompts small
# Hard cap on Claude spend per UTC day. Sourced from core/config.py so the
# Settings GUI / user_settings.json override takes effect; falls back to the
# historical 1.0 literal when core.config can't be imported (bare test import).
try:
    from core.config import DAILY_BUDGET_USD as DAILY_BUDGET_USD
except Exception:
    DAILY_BUDGET_USD = 1.0
# CHAPPIE_ENABLED gates the spending background daemon. Default False so a bare
# import (incl. a test) or the live skill loader registers the recall actions
# WITHOUT spinning the Claude-spending thread. Sourced from core/config.py so
# the Settings GUI / user_settings.json override takes effect; falls back to
# False when core.config can't be imported (bare test import).
try:
    from core.config import CHAPPIE_ENABLED as CHAPPIE_ENABLED
except Exception:
    CHAPPIE_ENABLED = False
APPROX_COST_PER_CALL = 0.0012    # Haiku ~$0.25/$1.25 per Mtok; 1k tokens ≈ this
SLEEP_TICK_SEC       = 30        # main loop wakeup; cheap

# Whisper-noise blocklist on top of the quality gates. The capture layer
# already drops some; this catches the rest before they make it to episodes.
_NOISE_PHRASES = frozenset({
    "thank you", "thanks", "you", "bye", "goodbye", "hello", "hi",
    "okay", "ok", "yeah", "yes", "no", "uh", "um", "mm", "hm",
    "right", "sure", "alright", "...",
})

# ─── state ──────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_thread_started = [False]
_last_episode_run = [0.0]
_last_fact_run    = [0.0]


# ─── small helpers ──────────────────────────────────────────────────────

def _atomic_write_json(path: str, data) -> None:
    """Local fallback if core.atomic_io isn't importable. tempfile + replace."""
    try:
        from core.atomic_io import _atomic_write_json as _real
        _real(path, data)
        return
    except Exception:
        pass
    import tempfile
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".chappie_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _today_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


# ─── cursors + budget ───────────────────────────────────────────────────

def _default_cursors() -> dict:
    return {
        "transcripts_offset": 0,    # byte offset into ambient_transcripts.jsonl
        "episodes_count":     0,    # episodes processed by fact layer
        "budget_date":        "",
        "budget_used_usd":    0.0,
        "version":            1,
    }


def _load_cursors() -> dict:
    with _state_lock:
        return _load_json(_CURSORS_FILE, _default_cursors())


def _save_cursors(c: dict) -> None:
    with _state_lock:
        try:
            _atomic_write_json(_CURSORS_FILE, c)
        except Exception as e:
            print(f"  [chappie] cursor save failed: {e}")


def _check_budget_and_reset(cursors: dict) -> bool:
    """Return True if we still have budget for a call today; reset on UTC date roll."""
    today = _today_utc()
    if cursors.get("budget_date") != today:
        cursors["budget_date"] = today
        cursors["budget_used_usd"] = 0.0
    return cursors.get("budget_used_usd", 0.0) < DAILY_BUDGET_USD


def _charge_budget(cursors: dict, usd: float = APPROX_COST_PER_CALL) -> None:
    cursors["budget_used_usd"] = cursors.get("budget_used_usd", 0.0) + usd


# ─── Layer 2: filter ────────────────────────────────────────────────────

def _passes_filter(entry: dict) -> bool:
    text = (entry.get("text") or "").strip()
    if len(text) < 3:
        return False
    try:
        if float(entry.get("no_speech_prob", 0)) > 0.5:
            return False
        if float(entry.get("avg_logprob", 0)) < -1.0:
            return False
        if float(entry.get("rms", 0)) < 0.003:
            return False
    except (TypeError, ValueError):
        return False
    norm = re.sub(r"[\.\!\?\,\-—]+$", "", text.lower()).strip()
    if norm in _NOISE_PHRASES:
        return False
    return True


# ─── claude wrapper ─────────────────────────────────────────────────────

def _llm(system: str, user: str, max_tokens: int = 600) -> str:
    """Run Chappie's autonomous consciousness passes on the LOCAL Ollama
    model ONLY. These are background, non-conversational LLM calls, so per the
    user's standing rule — Claude API credits are for conversational turns
    only; everything else stays on Max/local — they must never spend cloud
    credits. We call _call_local_llm directly rather than _llm_quick (which
    tries Claude first). Returns "" when the local model is unreachable.
    2026-05-30 audit."""
    try:
        bc = sys.modules.get("bobert_companion") or importlib.import_module("bobert_companion")
        out = bc._call_local_llm(system, [{"role": "user", "content": user}],
                                 max_tokens=max_tokens)
        return (out or "").strip()
    except Exception as e:
        print(f"  [chappie] llm call failed: {e}")
        return ""


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ─── Layer 3: episode building ─────────────────────────────────────────

def _read_new_transcripts(offset: int) -> tuple[list[dict], int]:
    """Read transcript entries written after `offset`. Returns (entries, new_offset)."""
    if not os.path.exists(_TRANSCRIPTS_FILE):
        return [], offset
    entries: list[dict] = []
    try:
        with open(_TRANSCRIPTS_FILE, "r", encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
            new_offset = f.tell()
        return entries, new_offset
    except Exception as e:
        print(f"  [chappie] transcript read failed: {e}")
        return [], offset


def _group_episodes(filtered: list[dict]) -> list[dict]:
    """Collapse consecutive utterances within EPISODE_GAP_SEC + same window
    context into one episode. Returns raw episode dicts without summaries
    (those get added by Claude after)."""
    if not filtered:
        return []
    episodes: list[dict] = []
    bucket: list[dict] = [filtered[0]]
    for prev, curr in zip(filtered, filtered[1:]):
        gap = float(curr.get("ts", 0)) - float(prev.get("ts", 0))
        same_window = (curr.get("window", "") == prev.get("window", ""))
        if gap <= EPISODE_GAP_SEC and same_window and len(bucket) < EPISODE_MAX_UTT:
            bucket.append(curr)
        else:
            episodes.append(_bucket_to_episode(bucket))
            bucket = [curr]
    if bucket:
        episodes.append(_bucket_to_episode(bucket))
    return episodes


def _bucket_to_episode(bucket: list[dict]) -> dict:
    start = float(bucket[0].get("ts", time.time()))
    end   = float(bucket[-1].get("ts", start))
    window = (bucket[0].get("window") or "").strip()
    in_call = bool(re.search(r"teams|zoom|meet|webex|skype", window, re.I))
    utt_texts = [e.get("text", "").strip() for e in bucket if e.get("text")]
    return {
        "id":        f"ep_{int(start)}_{len(utt_texts)}",
        "start_ts":  start,
        "end_ts":    end,
        "duration_s": round(end - start, 2),
        "window":    window,
        "in_call":   in_call,
        "utterance_count": len(utt_texts),
        "utterances": utt_texts,
        # Filled by _enrich_episode:
        "summary":      "",
        "topics":       [],
        "mood":         "",
        "new_entities": [],
    }


def _enrich_episode(ep: dict, cursors: dict) -> dict:
    """Call Claude (Haiku) to produce summary + topics + mood + new_entities."""
    if not _check_budget_and_reset(cursors):
        return ep  # out of budget — leave summary empty, store anyway
    utt_str = "\n".join(f"- {u}" for u in ep["utterances"])
    _uname = os.getenv("JARVIS_USER_NAME", "").strip() or "the user"
    system = (
        "You are JARVIS's internal observer. You receive a short stretch of "
        f"things {_uname} said (or that were said near them). Output ONLY valid JSON:\n"
        f'{{"summary": "one sentence in third person about {_uname}", '
        '"topics": ["1-3 short topic tags"], '
        '"mood": "1-3 words like \\"focused\\", \\"frustrated\\", \\"casual\\"", '
        '"new_entities": ["proper nouns: people, companies, products, projects"]}\n\n'
        "Rules:\n"
        "  - summary refers to the user by their name, not 'the user'. Be specific.\n"
        "  - If utterances are clearly a phone call / meeting, mention that.\n"
        "  - new_entities are ONLY proper nouns clearly mentioned. No invented names.\n"
        "  - If the snippet is too noisy / no real content, return\n"
        '    {"summary": "", "topics": [], "mood": "", "new_entities": []}.'
    )
    context_line = f"Window context: {ep['window']}\nIn call: {ep['in_call']}\n\n"
    user = context_line + "Utterances:\n" + utt_str
    raw = _llm(system, user, max_tokens=350)
    _charge_budget(cursors)
    data = _extract_json(raw) or {}
    ep["summary"]      = (data.get("summary") or "").strip()
    ep["topics"]       = [str(t).strip() for t in (data.get("topics") or []) if t]
    ep["mood"]         = (data.get("mood") or "").strip()
    ep["new_entities"] = [str(n).strip() for n in (data.get("new_entities") or []) if n]
    return ep


def _append_episodes(eps: list[dict]) -> None:
    if not eps:
        return
    try:
        os.makedirs(os.path.dirname(_EPISODES_FILE), exist_ok=True)
        with open(_EPISODES_FILE, "a", encoding="utf-8") as f:
            for ep in eps:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [chappie] episode write failed: {e}")


def _process_episodes_once(cursors: dict) -> int:
    """Read new transcripts → filter → group → enrich → append. Returns
    the number of episodes produced."""
    raw, new_offset = _read_new_transcripts(cursors.get("transcripts_offset", 0))
    if not raw:
        cursors["transcripts_offset"] = new_offset
        return 0
    filtered = [e for e in raw if _passes_filter(e)]
    eps = _group_episodes(filtered)
    enriched = [_enrich_episode(ep, cursors) for ep in eps]
    # Skip episodes Claude flagged as empty (no_speech_prob noise that snuck through)
    keep = [ep for ep in enriched if ep["summary"]]
    _append_episodes(keep)
    cursors["transcripts_offset"] = new_offset
    return len(keep)


# ─── Layer 4: fact accumulation ─────────────────────────────────────────

def _load_facts() -> dict:
    return _load_json(_FACTS_FILE, {})


def _save_facts(facts: dict) -> None:
    try:
        _atomic_write_json(_FACTS_FILE, facts)
    except Exception as e:
        print(f"  [chappie] facts save failed: {e}")


def _read_episodes_since(count_cursor: int) -> tuple[list[dict], int]:
    """Return episodes after the count_cursor-th line. Returns (eps, new_cursor)."""
    if not os.path.exists(_EPISODES_FILE):
        return [], count_cursor
    eps: list[dict] = []
    new_cursor = count_cursor
    try:
        with open(_EPISODES_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < count_cursor:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    eps.append(json.loads(line))
                    new_cursor = i + 1
                except Exception:
                    continue
        return eps, new_cursor
    except Exception as e:
        print(f"  [chappie] episode read failed: {e}")
        return [], count_cursor


def _update_facts_from_episodes(eps: list[dict], facts: dict, cursors: dict) -> int:
    """For each new episode with entities, ask Claude to write 1-2 sentence
    fact records keyed by entity. Returns count of entity-touches."""
    if not eps:
        return 0
    if not _check_budget_and_reset(cursors):
        return 0

    # Build a compact batch prompt — one Claude call per episode (cheap)
    # would balloon costs; batching keeps total spend bounded.
    summaries_block = []
    for ep in eps:
        if not ep.get("summary") or not ep.get("new_entities"):
            continue
        summaries_block.append({
            "ep_id":        ep["id"],
            "ts":           ep["start_ts"],
            "summary":      ep["summary"],
            "topics":       ep.get("topics", []),
            "mood":         ep.get("mood", ""),
            "entities":     ep.get("new_entities", []),
        })
    if not summaries_block:
        return 0

    # Snapshot existing entities so the model knows what's already on file
    existing_names = list(facts.keys())[:60]

    _uname = os.getenv("JARVIS_USER_NAME", "").strip() or "the user"
    system = (
        "You are JARVIS's quiet observer. You receive episode summaries he gathered "
        "and you update his private fact records about people, companies, products, "
        f"and projects in {_uname}'s life. Output ONLY valid JSON:\n"
        '{"updates": [{"entity": "Acme", "type": "person|company|product|project|other", '
        '"new_fact": "one sentence Chappie now believes about this entity, in third person", '
        '"open_question": "one specific thing Chappie does NOT yet know — empty string if none"}, ...]}\n\n'
        "Rules:\n"
        "  - Only output entities that actually appear in the summaries. Don't invent.\n"
        "  - Prefer adding a new_fact that ADDS information, not restates what's known.\n"
        "  - open_question is the curiosity flag — something a smart observer would "
        "still wonder. Empty string if nothing genuinely open.\n"
        f"  - Don't include facts about {_uname} unless they meaningfully define them "
        "(e.g. 'works in IT sales'). Trivia ('said hello') is not a fact.\n"
        "  - Skip noisy fragments. If summaries are all noise, return {\"updates\": []}."
    )
    user = (
        f"Existing entity names on file (don't duplicate type info): {existing_names}\n\n"
        f"New episodes:\n{json.dumps(summaries_block, ensure_ascii=False, indent=2)}"
    )
    raw = _llm(system, user, max_tokens=900)
    _charge_budget(cursors, usd=APPROX_COST_PER_CALL * 2)  # bigger output
    data = _extract_json(raw) or {}
    updates = data.get("updates") or []
    touched = 0
    now = time.time()
    for u in updates:
        name = (u.get("entity") or "").strip()
        if not name:
            continue
        rec = facts.get(name) or {
            "first_observed":   now,
            "type":             (u.get("type") or "other").strip(),
            "observation_count": 0,
            "facts":            [],
            "open_questions":   [],
        }
        rec["last_observed"]     = now
        rec["observation_count"] = int(rec.get("observation_count", 0)) + 1
        # Only append novel-looking facts (cheap substring dedupe)
        nf = (u.get("new_fact") or "").strip()
        if nf and not any(nf.lower() in (existing.get("text", "") or "").lower()
                          for existing in rec["facts"]):
            rec["facts"].append({"text": nf, "ts": now})
            rec["facts"] = rec["facts"][-20:]  # cap history
        oq = (u.get("open_question") or "").strip()
        if oq and oq not in rec["open_questions"]:
            rec["open_questions"].append(oq)
            rec["open_questions"] = rec["open_questions"][-5:]
        if not rec.get("type") or rec["type"] == "other":
            t = (u.get("type") or "").strip()
            if t and t != "other":
                rec["type"] = t
        facts[name] = rec
        touched += 1
    return touched


def _process_facts_once(cursors: dict) -> int:
    """Read new episodes → merge entity updates → save facts."""
    eps, new_cursor = _read_episodes_since(cursors.get("episodes_count", 0))
    if not eps:
        cursors["episodes_count"] = new_cursor
        return 0
    facts = _load_facts()
    touched = _update_facts_from_episodes(eps, facts, cursors)
    if touched:
        _save_facts(facts)
    cursors["episodes_count"] = new_cursor
    return touched


# ─── background daemon ─────────────────────────────────────────────────

def _chappie_loop():
    print("  [chappie] consciousness thread online")
    while True:
        try:
            time.sleep(SLEEP_TICK_SEC)
            now = time.time()
            cursors = _load_cursors()
            ran = False
            if now - _last_episode_run[0] >= EPISODE_INTERVAL_SEC:
                produced = _process_episodes_once(cursors)
                _last_episode_run[0] = now
                if produced:
                    print(f"  [chappie] +{produced} episode(s)")
                ran = True
            if now - _last_fact_run[0] >= FACT_INTERVAL_SEC:
                touched = _process_facts_once(cursors)
                _last_fact_run[0] = now
                if touched:
                    print(f"  [chappie] +{touched} fact-update(s)")
                ran = True
            if ran:
                _save_cursors(cursors)
        except Exception as e:
            # broad catch — a single bad tick can't take the thread down.
            print(f"  [chappie] tick error (ignored): {e}")


def _ensure_thread_started() -> None:
    # Spend gate: the daemon makes Claude calls, so it only ever starts when the
    # user has explicitly opted in via CHAPPIE_ENABLED (core/config.py, default
    # False). Without this, merely registering the skill — or importing the
    # module — would spin a live spending thread.
    if not CHAPPIE_ENABLED:
        return
    if _thread_started[0]:
        return
    _thread_started[0] = True
    t = threading.Thread(target=_chappie_loop, name="chappie", daemon=True)
    t.start()


# ─── recall actions (silent unless called) ─────────────────────────────

# Voice_mood layer opt-in tag (bobert_companion._parse_mood_tag) — Chappie's
# observational recall lands in the 'dry_amused' register: a slight smile in
# the voice, gently mocking rather than flat. The tag is stripped from the
# spoken text and the spoken text alone, so it shows up in transcripts but
# never reads aloud. Bobert with no mood-tag support strips an unknown
# leading bracket-tag harmlessly.
_DRY_AMUSED_TAG = "[mood:dry_amused] "


def chappie_recall_entity(arg: str = "") -> str:
    """'What do you know about X' — look up an entity in Chappie's facts."""
    name = (arg or "").strip().strip("?.,")
    if not name:
        return _DRY_AMUSED_TAG + "I'd need a name to think about, sir."
    facts = _load_facts()
    # Exact match first, then case-insensitive substring
    rec = facts.get(name)
    if not rec:
        lname = name.lower()
        for k, v in facts.items():
            if lname in k.lower() or k.lower() in lname:
                rec = v
                name = k
                break
    if not rec:
        return _DRY_AMUSED_TAG + f"Nothing on file for {name}, sir — I haven't picked up anything yet."
    bits = []
    if rec.get("facts"):
        latest = rec["facts"][-3:]
        bits.append("; ".join(f.get("text", "") for f in latest if f.get("text")))
    if rec.get("observation_count"):
        bits.append(f"({rec['observation_count']} observation(s))")
    if rec.get("open_questions"):
        bits.append("Still unclear: " + "; ".join(rec["open_questions"][:2]))
    body = (f"{name}, sir — " + " ".join(bits)) if bits else f"{name} is on file but I don't have specifics yet, sir."
    return _DRY_AMUSED_TAG + body


def chappie_recall_today(arg: str = "") -> str:
    """'What did you overhear today / did I mention X today' — pull from
    today's episodes, optionally filtered by a keyword in arg."""
    if not os.path.exists(_EPISODES_FILE):
        return _DRY_AMUSED_TAG + "Nothing on the record from today, sir."
    keyword = (arg or "").strip().lower().strip("?.,")
    start_of_day = time.mktime(datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0).timetuple())
    eps_today: list[dict] = []
    try:
        with open(_EPISODES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except Exception:
                    continue
                if float(ep.get("start_ts", 0)) >= start_of_day:
                    eps_today.append(ep)
    except Exception as e:
        return f"I tripped reading the day's notes, sir: {e}"
    if not eps_today:
        return _DRY_AMUSED_TAG + "Nothing on the record from today, sir."
    if keyword:
        matches = [
            ep for ep in eps_today
            if keyword in (ep.get("summary", "") + " " + " ".join(ep.get("topics", []))
                           + " " + " ".join(ep.get("new_entities", []))).lower()
        ]
        if not matches:
            return _DRY_AMUSED_TAG + f"Nothing on '{keyword}' from today, sir."
        eps_today = matches[-3:]
    # Compact summary — last few moments, dry
    lines = []
    for ep in eps_today[-4:]:
        t = datetime.fromtimestamp(ep.get("start_ts", 0)).strftime("%H:%M")
        lines.append(f"{t} — {ep.get('summary', '(no summary)')}")
    return _DRY_AMUSED_TAG + "Today, sir: " + " | ".join(lines)


def chappie_status(_: str = "") -> str:
    """Quick health check — episodes captured, facts on file, budget."""
    cursors = _load_cursors()
    facts = _load_facts()
    ep_count = 0
    if os.path.exists(_EPISODES_FILE):
        try:
            with open(_EPISODES_FILE, "r", encoding="utf-8") as f:
                ep_count = sum(1 for _ in f)
        except Exception:
            pass
    used = cursors.get("budget_used_usd", 0.0)
    return (
        f"Chappie status, sir — {ep_count} episode(s) on file, "
        f"{len(facts)} entity record(s), "
        f"today's spend ${used:.3f} of ${DAILY_BUDGET_USD:.2f}."
    )


# ─── skill registration hook ────────────────────────────────────────────

def register(actions: dict) -> None:
    """Called by the skill loader. Adds three silent recall actions and, only
    when CHAPPIE_ENABLED is set, starts the background consciousness thread.

    NOTE: the loader only ever calls a hook named `register` — the old name
    `register_actions` meant these three actions were NEVER registered (dead
    feature). Renamed 2026-05-30 audit; alias kept below for any direct
    callers/tests."""
    actions["chappie_recall_entity"] = chappie_recall_entity
    actions["chappie_recall_today"]  = chappie_recall_today
    actions["chappie_status"]        = chappie_status
    _ensure_thread_started()


# Backwards-compat alias for any caller/test using the old name.
register_actions = register


# NO module-load thread start. The Claude-spending daemon must never spin from a
# bare import (e.g. a test importing this module). It starts only when the skill
# loader calls register() AND CHAPPIE_ENABLED is set — see _ensure_thread_started.
