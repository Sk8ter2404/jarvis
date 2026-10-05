"""core/clone_seed.py -- which clone voice lines recur, and rendering them into
the persistent cache while the owner is quiet (voice architecture C6,
2026-10-05).

WHY
---
A write-through cache only helps the SECOND time a line is said in the clone
voice. Lines that already recur in the owner's history ("Certainly, sir.",
"One moment, sir.", the Kinect notice) can be rendered ahead of time, so the
first time he hears them in this voice they are instant too. The critic of
the voice plan measured ~106 texts said at least twice in history (~70 s of
GPU, ~14 MB): seeding only a fixed top-50 list reaches ~20 % of first
chunks, seeding every recurring text is what the ~42-46 % estimate assumed.

THE LEDGER (<data dir>/clone_cache/lines.json, gitignored)
---------------------------------------------------------
Every line the clone voices for a listener (not the filler warm, not a seed)
is counted under a hash of its text. Its TEXT is written down only once it
has been said twice: a one-off line (a message, a calendar entry) is never
stored in plain text by this module. On first use the ledger is bootstrapped
ONCE from what JARVIS has said before -- the "JARVIS:" lines of the session
logs and the assistant replies in the episode store -- cleaned and chunked
by the same planner the voice uses (the monolith supplies both), counted per
source and merged by MAXIMUM (a reply that is in both sources is one reply,
not two). Nothing of this is committed to the repo.

SEEDING (only with VOICE_CLONE_CACHE 'on')
------------------------------------------
CacheKeeper runs on one daemon ('clone-cache-keeper') and, each tick:
  * purges takes of any voice that is no longer a consented profile's
    reference (consent revoked, reference replaced) -- in EVERY mode;
  * saves the ledger, the take gate and the seed budget when they changed;
  * reads /health now and then (fast-decode alert, C3);
  * renders ONE recurring line that is not cached yet, only when ALL hold:
    the monolith's gate says the owner has been quiet >= QUIET_S and nothing
    is speaking, recording or mid-turn; the client is ready, speaking the
    consented voice, with the fast (cuda-graph) decoder on; the server's GPU
    is under GPU_BUSY_PCT utilisation (no game, no brain work); this voice
    has used less than VOICE_CLONE_SEED_GPU_S of server render time today. The
    wait is abandoned the moment the owner starts talking (``abort_fn``); a
    render already on the GPU finishes there (<= ~1.3 s), long before his
    turn reaches its first render (the end-of-turn silence alone is ~1.2 s).
    Most frequent first, shortest first; a line tried today is not retried
    until tomorrow; a line the owner asked to forget is never seeded.
A seed render is a background render (count=False): it never touches the
clone's miss count or cool-down. Seeds go through the same take gate as any
render. Two log lines per seeding burst at most.

Never raises out of a tick. Stdlib only (the client does the audio).
Tests: tests/test_clone_seed.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from typing import Callable, Iterable, List, Optional

__all__ = ["LEDGER_FILE", "STATE_FILE", "LineLedger", "SeedBudget",
           "CacheKeeper", "text_hash", "log_replies", "episode_replies",
           "reply_units", "count_units", "bootstrap_counts", "seed_gpu_s"]

LEDGER_FILE = "lines.json"
STATE_FILE = "seed_state.json"
LEDGER_MAX = 4000
# Longer lines are never seeded (they recur verbatim too rarely to pay).
TEXT_MAX_CHARS = 300
SEED_MIN_COUNT = 2
DEFAULT_SEED_GPU_S = 90.0
QUIET_S = 60.0
# Between two seed renders (the 3090 is shared with the brain and DWM).
SEED_GAP_S = 2.0
IDLE_TICK_S = 30.0
# /health is read before seeding when older than this, and otherwise at most
# this often while the owner is quiet (the fast-decode alert, C3).
HEALTH_SEED_MAX_AGE_S = 60.0
HEALTH_IDLE_S = 300.0
# A seed render waits at most this long (it is nobody's line).
SEED_TIMEOUT_S = 4.0
# History read for the bootstrap, at most this much per file.
HISTORY_MAX_BYTES = 32 * 1024 * 1024
# No seeding while the voice server's GPU is this busy (NVML utilisation, no
# CUDA context): a game, the brain at work. Idle, the 3090 reads a few
# percent (the desktop, the Kinect).
GPU_BUSY_PCT = 30

_LOG_LINE_RE = re.compile(
    r"^\[\d\d:\d\d:\d\d\]\s+JARVIS(?: \(spoken\))?:\s+(.+?)\s*$")
_SPACE_RE = re.compile(r"\s+")


def _cfg(name: str, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def seed_gpu_s() -> float:
    """VOICE_CLONE_SEED_GPU_S: server render seconds per voice per day the
    seeding may use (0 = no seeding), clamped to 0..600."""
    try:
        v = float(_cfg("VOICE_CLONE_SEED_GPU_S", DEFAULT_SEED_GPU_S))
        if not v == v:
            v = DEFAULT_SEED_GPU_S
    except Exception:
        v = DEFAULT_SEED_GPU_S
    return max(0.0, min(600.0, v))


def _norm(text) -> str:
    try:
        return _SPACE_RE.sub(" ", str(text or "")).strip()
    except Exception:
        return ""


def text_hash(text) -> str:
    """sha256 of the whitespace-normalised text (the ledger's key)."""
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()


def gpu_util_pct(index=None):
    """NVML utilisation of GPU `index` (PCI order), or the busiest GPU when
    `index` is None; None when it cannot be read. No CUDA context
    (core.gpu_probe). Never raises."""
    try:
        from core import gpu_probe
        vals = [g.get("util_pct") for g in gpu_probe.gpus()
                if index is None or g.get("index") == index]
        vals = [int(v) for v in vals if v is not None]
        return max(vals) if vals else None
    except Exception:
        return None


def _today(wall: Callable[[], float]) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.localtime(wall()))
    except Exception:
        return ""


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, obj) -> bool:
    try:
        from core.atomic_io import _atomic_write_json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _atomic_write_json(path, obj, indent=None)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  The ledger
# ═══════════════════════════════════════════════════════════════════════════
class LineLedger:
    """How often each clone line was voiced. Thread-safe; never raises.
    ``path`` None keeps it in memory only."""

    def __init__(self, path: Optional[str] = None, *,
                 wall: Callable[[], float] = time.time) -> None:
        self._mu = threading.Lock()
        self.path = path
        self._wall = wall
        self._lines: dict = {}
        self.bootstrapped = False
        self.dirty = False
        if path:
            self._load()

    def _load(self) -> None:
        obj = _read_json(self.path)
        if not isinstance(obj, dict) or obj.get("v") != 1:
            return
        lines = obj.get("lines")
        if not isinstance(lines, dict):
            return
        clean = {}
        for h, e in lines.items():
            if not (isinstance(h, str) and len(h) == 64 and isinstance(e, dict)):
                continue
            try:
                n = int(e.get("n", 0))
            except Exception:
                continue
            if n <= 0:
                continue
            ent = {"n": n, "d": str(e.get("d") or "")}
            t = e.get("t")
            if isinstance(t, str) and t and text_hash(t) == h:
                ent["t"] = t
            if e.get("f"):
                ent["f"] = 1
            clean[h] = ent
        with self._mu:
            self._lines = clean
            self.bootstrapped = bool(obj.get("bootstrapped"))

    def save_if_dirty(self) -> bool:
        if not self.path:
            return False
        with self._mu:
            if not self.dirty:
                return False
            obj = {"v": 1, "bootstrapped": self.bootstrapped,
                   "lines": {h: dict(e) for h, e in self._lines.items()}}
            self.dirty = False
        ok = _write_json(self.path, obj)
        if not ok:
            with self._mu:
                self.dirty = True
        return ok

    def _prune(self) -> None:
        """Caller holds the lock. Over LEDGER_MAX: one-off lines go first,
        oldest first; then the least said."""
        over = len(self._lines) - LEDGER_MAX
        if over <= 0:
            return
        order = sorted(self._lines.items(),
                       key=lambda kv: (kv[1].get("n", 0) > 1,
                                       kv[1].get("n", 0), kv[1].get("d", "")))
        for h, _e in order[:over]:
            self._lines.pop(h, None)

    def record(self, text) -> None:
        """One more time this line was voiced for a listener."""
        try:
            t = _norm(text)
            if not t:
                return
            h = text_hash(t)
            day = _today(self._wall)
            with self._mu:
                e = self._lines.get(h)
                if e is None:
                    e = self._lines[h] = {"n": 0, "d": day}
                e["n"] = int(e.get("n", 0)) + 1
                e["d"] = day
                if e["n"] >= SEED_MIN_COUNT and len(t) <= TEXT_MAX_CHARS:
                    e["t"] = t
                self.dirty = True
                self._prune()
        except Exception:
            pass

    def merge_counts(self, counts: dict) -> int:
        """Fold history counts in (max, not sum: the history and the live
        ledger may describe the same replies). Returns how many lines now
        recur."""
        try:
            day = _today(self._wall)
            with self._mu:
                for t, n in (counts or {}).items():
                    t = _norm(t)
                    try:
                        n = int(n)
                    except Exception:
                        continue
                    if not t or n <= 0:
                        continue
                    h = text_hash(t)
                    e = self._lines.get(h)
                    if e is None:
                        if n < SEED_MIN_COUNT:
                            continue      # a one-off: not even its hash
                        e = self._lines[h] = {"n": 0, "d": day}
                    e["n"] = max(int(e.get("n", 0)), n)
                    if e["n"] >= SEED_MIN_COUNT and len(t) <= TEXT_MAX_CHARS:
                        e["t"] = t
                self.bootstrapped = True
                self.dirty = True
                self._prune()
                return sum(1 for e in self._lines.values() if "t" in e)
        except Exception:
            return 0

    def forget(self, text) -> None:
        """The owner asked to forget this take: never seed it again."""
        try:
            t = _norm(text)
            if not t:
                return
            h = text_hash(t)
            with self._mu:
                e = self._lines.get(h)
                if e is None:
                    e = self._lines[h] = {"n": 0, "d": _today(self._wall)}
                e["f"] = 1
                self.dirty = True
        except Exception:
            pass

    def is_forgotten(self, text) -> bool:
        try:
            with self._mu:
                e = self._lines.get(text_hash(text))
                return bool(e and e.get("f"))
        except Exception:
            return False

    def count(self, text) -> int:
        try:
            with self._mu:
                e = self._lines.get(text_hash(text))
                return int(e.get("n", 0)) if e else 0
        except Exception:
            return 0

    def seeds(self, min_count: int = SEED_MIN_COUNT) -> List[str]:
        """Texts said at least `min_count` times, never forgotten: most said
        first, then shortest first."""
        try:
            with self._mu:
                items = [(e.get("n", 0), e["t"]) for e in self._lines.values()
                         if "t" in e and not e.get("f")
                         and e.get("n", 0) >= min_count]
            items.sort(key=lambda it: (-it[0], len(it[1]), it[1]))
            return [t for _n, t in items]
        except Exception:
            return []

    def __len__(self) -> int:
        with self._mu:
            return len(self._lines)


# ═══════════════════════════════════════════════════════════════════════════
#  History (the one-time bootstrap)
# ═══════════════════════════════════════════════════════════════════════════
def _read_tail(path: str, max_bytes: int) -> str:
    """The last `max_bytes` of a text file (whole lines), '' on any error."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def log_replies(paths: Iterable[str],
                max_bytes: int = HISTORY_MAX_BYTES) -> List[str]:
    """The text of every "JARVIS:" / "JARVIS (spoken):" line in these session
    logs, in order. Never raises."""
    out: List[str] = []
    for p in paths or ():
        for line in _read_tail(p, max_bytes).splitlines():
            m = _LOG_LINE_RE.match(line)
            if m:
                out.append(m.group(1))
    return out


def episode_replies(path: str,
                    max_bytes: int = HISTORY_MAX_BYTES) -> List[str]:
    """The text of every assistant reply in an episodes.jsonl store. Never
    raises."""
    out: List[str] = []
    if not path:
        return out
    for line in _read_tail(path, max_bytes).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("role") == "assistant":
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                out.append(t)
    return out


def reply_units(text, clean: Optional[Callable[[str], str]] = None,
                plan: Optional[Callable[[str], list]] = None) -> List[str]:
    """The units one reply is voiced in by the clone (its planned chunks)
    plus its first sentence (what the cache-aware planner looks for), each
    once. Never raises."""
    try:
        t = clean(text) if clean is not None else str(text or "")
        t = _norm(t)
        if not t:
            return []
        if plan is not None:
            chunks = [_norm(c) for c in plan(t)]
        else:
            chunks = [t]
        units: List[str] = []
        try:
            from core.sentence_tts import split_sentences
            first = _norm(split_sentences(t)[0]) if t else ""
        except Exception:
            first = ""
        for u in chunks + ([first] if first else []):
            if u and u not in units:
                units.append(u)
        return units
    except Exception:
        return []


def count_units(replies: Iterable[str], clean=None, plan=None) -> dict:
    """text -> how many replies voiced it."""
    counts: dict = {}
    for r in replies or ():
        for u in reply_units(r, clean, plan):
            counts[u] = counts.get(u, 0) + 1
    return counts


def bootstrap_counts(sources: Iterable[Iterable[str]], clean=None,
                     plan=None) -> dict:
    """Counts per source, merged by MAXIMUM: the session logs and the
    episode store record many of the same replies, so a sum would make
    every one-off reply look like a repeat."""
    merged: dict = {}
    for replies in sources or ():
        for t, n in count_units(replies, clean, plan).items():
            if n > merged.get(t, 0):
                merged[t] = n
    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  The daily budget
# ═══════════════════════════════════════════════════════════════════════════
class SeedBudget:
    """Seed render time used per voice today, and the lines tried today.
    Resets when the date changes. Thread-safe; never raises."""

    def __init__(self, path: Optional[str] = None, *,
                 wall: Callable[[], float] = time.time) -> None:
        self._mu = threading.Lock()
        self.path = path
        self._wall = wall
        self._date = _today(wall)
        self._voices: dict = {}
        self.dirty = False
        obj = _read_json(path) if path else None
        if isinstance(obj, dict) and obj.get("date") == self._date and \
                isinstance(obj.get("voices"), dict):
            for p, v in obj["voices"].items():
                if not isinstance(v, dict):
                    continue
                try:
                    ms = float(v.get("gpu_ms", 0.0))
                except Exception:
                    ms = 0.0
                tried = [h for h in (v.get("tried") or [])
                         if isinstance(h, str) and len(h) == 64]
                self._voices[str(p)] = {"gpu_ms": max(0.0, ms),
                                        "tried": set(tried)}

    def _roll(self) -> None:
        """Caller holds the lock: a new day starts a new budget."""
        day = _today(self._wall)
        if day != self._date:
            self._date = day
            self._voices = {}
            self.dirty = True

    def _v(self, prefix: str) -> dict:
        v = self._voices.get(prefix)
        if v is None:
            v = self._voices[prefix] = {"gpu_ms": 0.0, "tried": set()}
        return v

    def used_s(self, prefix: str) -> float:
        with self._mu:
            self._roll()
            return self._v(prefix)["gpu_ms"] / 1000.0

    def add(self, prefix: str, ms) -> None:
        try:
            with self._mu:
                self._roll()
                self._v(prefix)["gpu_ms"] += max(0.0, float(ms or 0.0))
                self.dirty = True
        except Exception:
            pass

    def tried(self, prefix: str, text) -> bool:
        with self._mu:
            self._roll()
            return text_hash(text) in self._v(prefix)["tried"]

    def mark_tried(self, prefix: str, text) -> None:
        with self._mu:
            self._roll()
            self._v(prefix)["tried"].add(text_hash(text))
            self.dirty = True

    def save_if_dirty(self) -> bool:
        if not self.path:
            return False
        with self._mu:
            if not self.dirty:
                return False
            obj = {"date": self._date,
                   "voices": {p: {"gpu_ms": round(v["gpu_ms"], 1),
                                  "tried": sorted(v["tried"])}
                              for p, v in self._voices.items()}}
            self.dirty = False
        ok = _write_json(self.path, obj)
        if not ok:
            with self._mu:
                self.dirty = True
        return ok


# ═══════════════════════════════════════════════════════════════════════════
#  The keeper daemon
# ═══════════════════════════════════════════════════════════════════════════
class _Abort:
    """An Event-shaped view of a predicate (the client's render polls
    ``is_set()`` while it waits). A predicate that raises counts as set."""

    def __init__(self, fn: Optional[Callable[[], bool]]) -> None:
        self._fn = fn

    def is_set(self) -> bool:
        if self._fn is None:
            return False
        try:
            return bool(self._fn())
        except Exception:
            return True


class CacheKeeper:
    """Housekeeping and quiet-time seeding for one CloneVoiceClient.

    ``gate_fn()``     -> None when the owner has been quiet long enough and
                         nothing speaks / records / is mid-turn, else a
                         short reason (the monolith's view of the room);
    ``abort_fn()``    -> True the moment a seed render must stop waiting
                         (the owner started talking, a line wants the
                         speakers);
    ``history_fn()``  -> a list of reply lists (one per source) for the
                         one-time bootstrap, or None;
    ``clean`` / ``plan`` -> the monolith's speech cleaner and the clone's
                         chunk planner (for the bootstrap)."""

    def __init__(self, client, *, gate_fn: Callable[[], Optional[str]],
                 abort_fn: Optional[Callable[[], bool]] = None,
                 history_fn: Optional[Callable[[], list]] = None,
                 clean: Optional[Callable[[str], str]] = None,
                 plan: Optional[Callable[[str], list]] = None,
                 log: Optional[Callable[[str], None]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Optional[Callable[[float], None]] = None,
                 tick_s: float = IDLE_TICK_S) -> None:
        self.client = client
        self._gate_fn = gate_fn
        self._abort = _Abort(abort_fn)
        self._history_fn = history_fn
        self._clean = clean
        self._plan = plan
        self._log_fn = log
        self._clock = clock
        self._sleep = sleep or time.sleep
        self._tick_s = float(tick_s)
        self._thread: Optional[threading.Thread] = None
        self._mu = threading.Lock()
        self._burst = 0                 # lines seeded in the current burst
        self._burst_announced = False
        self._last_health = float("-inf")

    def _log(self, msg: str) -> None:
        try:
            (self._log_fn or print)(msg)
        except Exception:
            pass

    def start(self) -> bool:
        """Start the daemon once. True when this call started it."""
        with self._mu:
            if self._thread is not None:
                return False
            self._thread = threading.Thread(target=self._loop,
                                            name="clone-cache-keeper",
                                            daemon=True)
            th = self._thread
        th.start()
        return True

    def _loop(self) -> None:
        while True:
            try:
                did = self.tick()
            except Exception:
                did = "error"
            self._sleep(SEED_GAP_S if did == "seeded" else self._tick_s)

    def _gate(self) -> Optional[str]:
        try:
            return self._gate_fn()
        except Exception as e:
            return f"gate error ({type(e).__name__})"

    def _end_burst(self, why: str) -> None:
        if self._burst:
            try:
                prefix = self.client.voice_prefix()
                used = self.client.budget.used_s(prefix) if prefix else 0.0
            except Exception:
                used = 0.0
            self._log(f"  [clone-cache] seeded {self._burst} line"
                      f"{'s' if self._burst != 1 else ''} ({why}; "
                      f"{used:.0f} s of {seed_gpu_s():.0f} s seed render "
                      f"time used today)")
        self._burst = 0
        self._burst_announced = False

    def _persist(self) -> None:
        for obj in (getattr(self.client, "ledger", None),
                    getattr(self.client, "budget", None)):
            try:
                if obj is not None:
                    obj.save_if_dirty()
            except Exception:
                pass
        try:
            self.client.store.save_gate()
        except Exception:
            pass

    def tick(self) -> str:
        """One round; returns what it did or why it did nothing (tests read
        it). Never raises."""
        from core import clone_render_cache as _crc
        try:
            self.client.purge_unconsented()
        except Exception:
            pass
        self._persist()
        m = _crc.mode()
        if m == "off":
            self._end_burst("cache off")
            return "off"
        why = self._gate()
        if why:
            self._end_burst("you were busy" if self._burst else why)
            return str(why)
        ledger = getattr(self.client, "ledger", None)
        if ledger is None:
            return "no-ledger"
        if not ledger.bootstrapped and self._history_fn is not None:
            return self._bootstrap(ledger)
        now = self._clock()
        if now - self._last_health >= HEALTH_IDLE_S:
            self._last_health = now
            try:
                self.client.refresh_health()
            except Exception:
                pass
        if m != "on":
            return m
        return self._seed_one(ledger)

    def _bootstrap(self, ledger: LineLedger) -> str:
        try:
            sources = self._history_fn() or []
            counts = bootstrap_counts(sources, self._clean, self._plan)
            n_replies = sum(len(s) for s in sources)
        except Exception:
            counts, n_replies = {}, 0
        recur = ledger.merge_counts(counts)
        ledger.save_if_dirty()
        self._log(f"  [clone-cache] line history: {recur} line"
                  f"{'s' if recur != 1 else ''} recur in {n_replies} earlier "
                  f"replies; they are rendered into the cache while you're "
                  f"quiet")
        return "bootstrapped"

    def _seed_one(self, ledger: LineLedger) -> str:
        cap = seed_gpu_s()
        if cap <= 0.0:
            self._end_burst("seeding off")
            return "seed-off"
        try:
            why = self.client.seed_ready(HEALTH_SEED_MAX_AGE_S)
        except Exception as e:
            why = f"client error ({type(e).__name__})"
        if why:
            self._end_burst(why)
            return str(why)
        util = gpu_util_pct(self.client.server_gpu_index())
        if util is not None and util >= GPU_BUSY_PCT:
            self._end_burst("the GPU got busy")
            return "gpu busy"
        prefix = self.client.voice_prefix()
        budget = self.client.budget
        if budget.used_s(prefix) >= cap:
            self._end_burst("today's seed budget is spent")
            return "budget"
        pick = None
        left = 0
        for t in ledger.seeds():
            if budget.tried(prefix, t) or self.client.is_cached(t):
                continue
            left += 1
            if pick is None:
                pick = t
        if pick is None:
            self._end_burst("every recurring line is cached")
            return "done"
        if not self._burst_announced:
            self._burst_announced = True
            self._log(f"  [clone-cache] seeding {left} recurring line"
                      f"{'s' if left != 1 else ''} into the clone cache, one "
                      f"at a time while you're quiet")
        if self._abort.is_set():
            return "aborted"
        budget.mark_tried(prefix, pick)
        t0 = self._clock()
        out = self.client.render(pick, SEED_TIMEOUT_S, count=False,
                                 cancel=self._abort)
        spent_ms = (out.server_ms if getattr(out, "server_ms", None)
                    else (self._clock() - t0) * 1000.0)
        try:
            spent_ms = float(spent_ms)
        except Exception:
            spent_ms = (self._clock() - t0) * 1000.0
        budget.add(prefix, spent_ms)
        if out.ok and not out.cached:
            self._burst += 1
            return "seeded"
        if getattr(out, "cancelled", False):
            self._end_burst("you started talking")
            return "aborted"
        return "seed-failed"
