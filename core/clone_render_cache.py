"""core/clone_render_cache.py -- the clone voice's render cache, kept on disk
(voice architecture C4, 2026-10-05).

WHY THIS EXISTS
---------------
JARVIS says the same short lines again and again ("Certainly, sir.", "One
moment, sir."), and every one costs a render on the clone voice server
(~0.5-0.9 s on the 3090) before the first word. The client's in-memory
cache only helped within one run, and JARVIS restarts about ten times a day,
so it almost never hit. This store keeps finished takes across restarts:
a small memory tier (16 MB, the client's CACHE_MAX_BYTES) in front of int16
.npy files under <data dir>/clone_cache/ (VOICE_CLONE_CACHE_MB, default 64).

VOICE_CLONE_CACHE (core/config.py), read at call time:
  * 'off'    -- memory only, exactly the client's old per-run cache; nothing
                is read from or written to disk (a purge on consent revoke
                still runs: those files are the cloned voice);
  * 'shadow' -- the memory tier serves as before; every server render is
                also written to disk, and a line the memory tier misses logs
                whether the disk WOULD have served it (proves the hit rate);
  * 'on'     -- memory, then disk, then the server; written through.

THE KEY
-------
sha256 of JSON ["clone-v1", ref_sha256, model, t3_dtype, sample_rate, text]:
the server's voice prompt hash and the model facts it reports on /health,
plus the exact text sent (numbers spelled out, whitespace collapsed). A new
reference.wav, another model, another T3 precision or sample rate gives new
keys, so a take is only ever served for the voice and model that made it.
The text itself is never stored here (the file name is the key).

FILES
-----
<prefix>_<key>.npy, where <prefix> is the first 16 hex characters of the
voice's ref_sha256: a revoked or replaced voice is purged by prefix
(purge_except). Each file is the server's own int16 samples, unprocessed:
the client trims and loudness-matches on every read (finish_audio), so a
change to those settings never leaves stale audio behind. A file that is not
a 1-D int16 array of sane length is never served and is deleted. Writes go
through ONE writer thread: the caller's ``keep()`` check first (the client
re-checks there that the take came from the server and voice it knows), then
a temporary file flushed to the disk (fsync) and renamed over the take, then
the folder is trimmed to the cap, least recently used first. A write still
queued when its take is forgotten or its voice purged never lands, and the
temporary file of a write cut off by a crash is swept at the next attach (and
by a purge of its voice).

THE TAKE GATE
-------------
Sampling is random, so a cached take freezes one performance. Only takes
whose length fits the text are kept on disk: the ratio of the audio length
to the length expected for that many characters must lie within the p1-p99
range of this voice's own recent renders (a fixed band until 50 have been
seen), and the voice's own range can only be TIGHTER than the fixed band,
never wider: every measured take joins the history (keeping only admitted
ones would ratchet the band shut), so without that clamp a few runaway takes
would stretch p99 until runaways were admitted. The audio per SPEECH token is
constant (40 ms a token by construction), so it cannot flag anything; per
character of text is what a runaway or truncated take changes. Rejected
takes still play (and stay in the memory tier for this run, as before); they
are just never persisted.

Nothing here raises: a cache fault costs the cache, never a line of speech.
Stdlib + numpy; no monolith import. Tests: tests/test_clone_render_cache.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Iterable, Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is present wherever audio is
    np = None  # type: ignore

__all__ = ["MODES", "DEFAULT_MODE", "DEFAULT_DISK_MB", "SUBDIR", "KEY_VERSION",
           "mode", "disk_cap_bytes", "normalise_text", "voice_prefix",
           "make_key", "take_ratio", "TakeGate", "CloneRenderCache"]

MODES = ("off", "shadow", "on")
# The owner said yes to a persistent cache (2026-10-04); the serve rules are
# the old in-memory cache's (ready + consented voice only), so 'on' ships.
DEFAULT_MODE = "on"
DEFAULT_DISK_MB = 64
MAX_DISK_MB = 2048
DEFAULT_MEM_BYTES = 16 * 1024 * 1024
SUBDIR = "clone_cache"
KEY_VERSION = "clone-v1"
PREFIX_HEX = 16
GATE_FILE = "gate.json"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_RE = re.compile(r"^([0-9a-f]{16})_([0-9a-f]{64})\.npy$")
# A write's temporary file (_write): the take's name plus the writer's pid and
# thread. One left by a crash is swept at attach and by a purge of its voice.
_TMP_RE = re.compile(r"^([0-9a-f]{16})_([0-9a-f]{64})\.npy\.\d+\.\d+\.tmp$")
# How many forgotten keys are remembered (a queued write of one never lands).
_FORGOT_MAX = 256
# A cached take longer than this is refused (a 4 MB float32 entry is ~43 s at
# 24 kHz; nothing JARVIS caches comes close).
MAX_TAKE_S = 45.0
_WRITE_QUEUE_MAX = 64
# Queued by close(): the writer thread returns when it reaches it.
_STOP_WRITER = object()

# The take gate. Expected audio for a line of N characters (a straight-line
# fit over 1,141 live renders of 2026-10-03..05, all voices): 281 ms + 57.6
# ms per character. The ratio audio / expected had p1 0.756 and p99 1.50
# there; the fixed band is that, widened, until the voice has its own.
GATE_FIXED_MS = 281.0
GATE_MS_PER_CHAR = 57.6
GATE_DEFAULT_BAND = (0.70, 1.60)
GATE_MIN_SAMPLES = 50
GATE_HISTORY = 400
GATE_LO_Q = 0.01
GATE_HI_Q = 0.99


def _cfg(name: str, default):
    """A core.config knob read at call time; `default` on any error."""
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def mode() -> str:
    """VOICE_CLONE_CACHE as one of MODES; anything unknown is 'off'."""
    m = str(_cfg("VOICE_CLONE_CACHE", DEFAULT_MODE) or "off").strip().lower()
    return m if m in MODES else "off"


def disk_cap_bytes() -> int:
    """VOICE_CLONE_CACHE_MB in bytes, clamped to 0..MAX_DISK_MB."""
    try:
        mb = float(_cfg("VOICE_CLONE_CACHE_MB", DEFAULT_DISK_MB))
        if not mb == mb:            # NaN
            mb = float(DEFAULT_DISK_MB)
    except Exception:
        mb = float(DEFAULT_DISK_MB)
    mb = max(0.0, min(float(MAX_DISK_MB), mb))
    return int(mb * 1024 * 1024)


def normalise_text(text) -> str:
    """The text as it is keyed and sent: stripped, whitespace runs collapsed
    to one space. Case and punctuation are kept (they change the take)."""
    try:
        return " ".join(str(text or "").split())
    except Exception:
        return ""


def voice_prefix(ref_sha) -> str:
    """The file-name prefix for a voice: the first PREFIX_HEX hex characters
    of its reference hash, or '' when that is not a sha256 hex digest."""
    s = str(ref_sha or "").strip().lower()
    return s[:PREFIX_HEX] if _HEX64_RE.match(s) else ""


def make_key(ref_sha, model, t3_dtype, sample_rate, text) -> Optional[str]:
    """The cache key for one take, or None (no usable voice hash, no text):
    no key, no caching."""
    try:
        ref = str(ref_sha or "").strip().lower()
        t = str(text or "")
        if not _HEX64_RE.match(ref) or not t:
            return None
        material = json.dumps(
            [KEY_VERSION, ref, str(model or ""), str(t3_dtype or ""),
             int(sample_rate or 0), t], ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
    except Exception:
        return None


def take_ratio(audio_ms, chars) -> Optional[float]:
    """Audio length over the length expected for `chars` characters; None
    when either is unusable."""
    try:
        a = float(audio_ms)
        n = int(chars)
        if not a > 0.0 or n <= 0:
            return None
        return a / (GATE_FIXED_MS + GATE_MS_PER_CHAR * n)
    except Exception:
        return None


def _quantile(sorted_vals: list, q: float) -> float:
    """Linear-interpolated quantile of an already sorted, non-empty list."""
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class TakeGate:
    """Per-voice history of take ratios and the admission band. Thread-safe;
    never raises."""

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._hist: dict = {}       # prefix -> deque of ratios
        self.dirty = False

    def band(self, prefix: str) -> tuple:
        """(lo, hi, samples) for `prefix`: p1-p99 of its history once it has
        GATE_MIN_SAMPLES, clamped INSIDE GATE_DEFAULT_BAND (a voice's own
        range may only tighten the fixed band: a run of runaway takes in the
        history must never widen it), else GATE_DEFAULT_BAND."""
        with self._mu:
            h = list(self._hist.get(prefix) or ())
        if len(h) < GATE_MIN_SAMPLES:
            return GATE_DEFAULT_BAND[0], GATE_DEFAULT_BAND[1], len(h)
        h.sort()
        return (max(GATE_DEFAULT_BAND[0], _quantile(h, GATE_LO_Q)),
                min(GATE_DEFAULT_BAND[1], _quantile(h, GATE_HI_Q)), len(h))

    def admit(self, prefix: str, audio_ms, chars) -> tuple:
        """(admitted, ratio, (lo, hi)). The band is the one BEFORE this take
        joins the history; every measurable take joins it."""
        try:
            r = take_ratio(audio_ms, chars)
            lo, hi, _n = self.band(prefix)
            if r is None or not prefix:
                return False, r, (lo, hi)
            with self._mu:
                d = self._hist.get(prefix)
                if d is None:
                    d = self._hist[prefix] = deque(maxlen=GATE_HISTORY)
                d.append(round(float(r), 4))
                self.dirty = True
            return (lo <= r <= hi), r, (lo, hi)
        except Exception:
            return False, None, GATE_DEFAULT_BAND

    def drop_except(self, keep: set) -> None:
        with self._mu:
            for p in [p for p in self._hist if p not in keep]:
                self._hist.pop(p, None)
                self.dirty = True

    def to_json(self) -> dict:
        with self._mu:
            return {"v": 1, "hist": {p: list(d) for p, d in self._hist.items()}}

    def load_json(self, obj) -> None:
        try:
            hist = obj.get("hist") if isinstance(obj, dict) else None
            if not isinstance(hist, dict):
                return
            with self._mu:
                for p, vals in hist.items():
                    if not (isinstance(p, str) and len(p) == PREFIX_HEX
                            and isinstance(vals, list)):
                        continue
                    d = deque(maxlen=GATE_HISTORY)
                    for v in vals[-GATE_HISTORY:]:
                        try:
                            f = float(v)
                        except Exception:
                            continue
                        if 0.0 < f < 100.0:
                            d.append(f)
                    self._hist[p] = d
        except Exception:
            pass


class CloneRenderCache:
    """Memory tier (finished float32 takes, LRU, capped by ``mem_cap_fn()``
    bytes, one entry at most a quarter of it) in front of an optional disk
    tier (raw int16 takes, ``attach``). Thread-safe; no method raises.

    ``sync_writes`` makes disk writes happen on the caller's thread (tests);
    otherwise ONE daemon writer thread does them, started on first use."""

    def __init__(self, *, mem_cap_fn: Optional[Callable[[], int]] = None,
                 sync_writes: bool = False) -> None:
        self._mu = threading.Lock()
        self._mem: "OrderedDict[str, tuple]" = OrderedDict()
        self._mem_bytes = 0
        self._mem_cap_fn = mem_cap_fn or (lambda: DEFAULT_MEM_BYTES)
        self._dir: Optional[str] = None
        self._index: dict = {}      # file name -> size in bytes
        self._sync = bool(sync_writes)
        self._q: Optional["queue.Queue"] = None
        self._writer: Optional[threading.Thread] = None
        self._pending = 0
        self._idle = threading.Event()
        self._idle.set()
        # Bumped by every forget / purge; a queued write carries the value
        # it was queued at, so one queued BEFORE its take was forgotten (or
        # its voice purged) never lands (_stale).
        self._gen = 0
        self._forgot: "OrderedDict[str, int]" = OrderedDict()
        self._purge: Optional[tuple] = None     # (gen, frozenset of kept)
        # The gen of the last memory wipe (wipe): every write queued before
        # it is stale.
        self._wiped = 0
        self.gate = TakeGate()
        self.counters = {"mem_hits": 0, "disk_hits": 0, "misses": 0,
                         "shadow_would_hit": 0, "shadow_would_miss": 0,
                         "writes": 0, "write_errors": 0, "rejected": 0,
                         "corrupt": 0, "purged": 0, "forgotten": 0,
                         "dropped_writes": 0, "not_kept": 0,
                         "stale_writes": 0, "leftovers": 0,
                         "refused": 0, "rejected_on_read": 0}

    # ── setup ────────────────────────────────────────────────────────────
    def attach(self, disk_dir: str) -> bool:
        """Use `disk_dir` as the disk tier: index its files and load the
        take gate's history. True when the folder is usable."""
        try:
            d = str(disk_dir or "")
            if not d:
                return False
            os.makedirs(d, exist_ok=True)
            index = {}
            swept = 0
            for name in os.listdir(d):
                if _FILE_RE.match(name):
                    try:
                        index[name] = os.path.getsize(os.path.join(d, name))
                    except OSError:
                        pass
                elif _TMP_RE.match(name):
                    # A write cut off by a crash (nothing writes before the
                    # folder is attached): cloned-voice audio outside the
                    # index, the cap and the purge -- removed.
                    try:
                        os.remove(os.path.join(d, name))
                        swept += 1
                    except OSError:
                        pass
            gate_obj = None
            try:
                with open(os.path.join(d, GATE_FILE), "r",
                          encoding="utf-8") as f:
                    gate_obj = json.load(f)
            except Exception:
                gate_obj = None
            with self._mu:
                self._dir = d
                self._index = index
                self.counters["leftovers"] += swept
            if gate_obj is not None:
                self.gate.load_json(gate_obj)
            return True
        except Exception:
            return False

    @property
    def disk_dir(self) -> Optional[str]:
        with self._mu:
            return self._dir

    def _count(self, name: str, n: int = 1) -> None:
        with self._mu:
            self.counters[name] = self.counters.get(name, 0) + n

    # ── memory tier ──────────────────────────────────────────────────────
    def mem_get(self, key):
        """(a COPY of the finished take, sr) or None."""
        try:
            with self._mu:
                v = self._mem.get(key)
                if v is None:
                    return None
                self._mem.move_to_end(key)
                return v[0].copy(), v[1]
        except Exception:
            return None

    def mem_has(self, key) -> bool:
        try:
            with self._mu:
                return key in self._mem
        except Exception:
            return False

    def mem_put(self, key, audio, sr: int, prefix: str = "") -> bool:
        """Keep a COPY of a finished float32 take. False when it is too big
        for the memory tier (a quarter of its cap) or unusable."""
        try:
            if not key or audio is None:
                return False
            a = np.array(audio, dtype=np.float32, copy=True).reshape(-1)
            cap = int(self._mem_cap_fn())
            if not a.size or a.nbytes > cap // 4:
                return False
            with self._mu:
                old = self._mem.pop(key, None)
                if old is not None:
                    self._mem_bytes -= int(old[0].nbytes)
                self._mem[key] = (a, int(sr), str(prefix or ""))
                self._mem_bytes += int(a.nbytes)
                while self._mem_bytes > cap and self._mem:
                    _k, gone = self._mem.popitem(last=False)
                    self._mem_bytes -= int(gone[0].nbytes)
            return True
        except Exception:
            return False

    def mem_len(self) -> int:
        with self._mu:
            return len(self._mem)

    # ── disk tier ────────────────────────────────────────────────────────
    @staticmethod
    def _name(key: str, prefix: str) -> Optional[str]:
        name = f"{prefix}_{key}.npy"
        return name if _FILE_RE.match(name) else None

    def disk_has(self, key, prefix) -> bool:
        """The disk tier holds a file for this take (by the in-memory index:
        no file I/O). Never raises."""
        try:
            name = self._name(str(key or ""), str(prefix or ""))
            with self._mu:
                return bool(name) and self._dir is not None and \
                    name in self._index
        except Exception:
            return False

    def disk_get(self, key, prefix, sr: int):
        """The raw int16 take for this key, or None (absent, no disk tier,
        corrupt, wrong shape or type, absurd length). A bad file is deleted
        (so it is never tried again, and never sits outside the cap; the
        next render of the line replaces it)."""
        name = None
        try:
            name = self._name(str(key or ""), str(prefix or ""))
            with self._mu:
                d = self._dir
                known = bool(name) and name in self._index
            if d is None or not known:
                return None
            path = os.path.join(d, name)
            try:
                a = np.load(path, allow_pickle=False)
            except FileNotFoundError:
                with self._mu:
                    self._index.pop(name, None)
                return None
            max_n = int(MAX_TAKE_S * max(1, int(sr or 0)))
            if (not isinstance(a, np.ndarray) or a.dtype != np.int16
                    or a.ndim != 1 or not 0 < a.size <= max_n):
                self._drop_bad(d, name)
                return None
            try:
                os.utime(path)              # recently used: trimmed last
            except OSError:
                pass
            return np.ascontiguousarray(a)
        except Exception:
            try:
                with self._mu:
                    d = self._dir
                if name:
                    self._drop_bad(d, name)
            except Exception:
                pass
            return None

    def _drop_bad(self, d: Optional[str], name: str) -> None:
        """A take file that cannot be served: out of the index and off the
        disk, counted 'corrupt'. Never raises."""
        try:
            with self._mu:
                self._index.pop(name, None)
                self.counters["corrupt"] += 1
            if d is not None:
                os.remove(os.path.join(d, name))
        except Exception:
            pass

    def disk_put(self, key, prefix, pcm16, *, sr: Optional[int] = None,
                 keep: Optional[Callable[[], bool]] = None) -> bool:
        """Persist a raw int16 take (queued to the writer thread unless
        sync_writes). ``sr`` given: a take longer than MAX_TAKE_S (which
        disk_get would never serve) is refused. ``keep()`` runs on the
        writer just before the write: False (or raising) and the take is not
        kept. False when there is no disk tier or nothing to write."""
        try:
            k = str(key or "")
            p = str(prefix or "")
            name = self._name(k, p)
            with self._mu:
                d = self._dir
                gen = self._gen
            if d is None or not name or pcm16 is None:
                return False
            a = np.ascontiguousarray(np.asarray(pcm16, dtype=np.int16)
                                     .reshape(-1))
            if not a.size:
                return False
            if sr is not None and a.size > int(MAX_TAKE_S * max(1, int(sr))):
                return False
            item = (d, name, a, keep, gen, k, p)
            if self._sync:
                self._write_item(item)
                return True
            self._ensure_writer()
            with self._mu:
                self._pending += 1
                self._idle.clear()
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self._done_one()
                self._count("dropped_writes")
                return False
            return True
        except Exception:
            return False

    def _stale(self, gen: int, key: str, prefix: str) -> bool:
        """Caller holds the lock: a write queued at `gen` whose take has been
        forgotten, or whose voice purged, since."""
        if gen < self._wiped:
            return True
        fg = self._forgot.get(key)
        if fg is not None and fg > gen:
            return True
        pg = self._purge
        return pg is not None and pg[0] > gen and prefix not in pg[1]

    def _write_item(self, item) -> None:
        """One queued write: skipped when stale, or when its keep() says no;
        else written. Never raises."""
        d, name, a, keep, gen, key, prefix = item
        try:
            with self._mu:
                stale = self._stale(gen, key, prefix)
            if stale:
                self._count("stale_writes")
                return
            if keep is not None:
                try:
                    ok = bool(keep())
                except Exception:
                    ok = False
                if not ok:
                    self._count("not_kept")
                    return
            self._write(d, name, a, gen, key, prefix)
        except Exception:
            pass

    def _ensure_writer(self) -> None:
        with self._mu:
            if self._writer is not None:
                return
            self._q = queue.Queue(maxsize=_WRITE_QUEUE_MAX)
            th = threading.Thread(target=self._writer_loop, args=(self._q,),
                                  name="clone-cache-writer", daemon=True)
            self._writer = th
        th.start()

    def _done_one(self) -> None:
        with self._mu:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._idle.set()

    def _writer_loop(self, q=None) -> None:
        q = q if q is not None else self._q
        while True:
            try:
                item = q.get()
            except Exception:
                time.sleep(0.1)
                continue
            if item is _STOP_WRITER:
                return
            try:
                self._write_item(item)
            except Exception:
                pass
            finally:
                self._done_one()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait (bounded) until every queued write has landed. Tests."""
        return self._idle.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Stop the writer thread once the writes queued before this call
        have landed; a later write starts a new one. The process keeps ONE
        cache for its life, so this is for a cache that is done with (a
        test's client): each writer is a daemon that otherwise waits on
        its queue forever, and a suite that built hundreds of clients kept
        one thread per client alive (50 at once in the 2026-10-09 rel-182
        run, which pushed a real faulthandler dump past its 100-thread
        limit). Never raises."""
        try:
            with self._mu:
                th, q = self._writer, self._q
                if th is None or q is None:
                    return
                self._writer = None
            q.put(_STOP_WRITER, timeout=max(0.0, float(timeout)))
            th.join(max(0.0, float(timeout)))
        except Exception:
            pass

    def _write(self, d: str, name: str, a, gen: int = -1, key: str = "",
               prefix: str = "") -> None:
        """Write one take: a temporary file flushed to the disk (fsync, so a
        crash cannot leave a renamed take whose samples never reached it)
        and renamed over the take; then trim the folder to the cap. A take
        forgotten (or its voice purged) while it was being written is
        removed again, so a forget / purge always wins the race."""
        path = os.path.join(d, name)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            os.makedirs(d, exist_ok=True)
            with open(tmp, "wb") as f:
                np.save(f, a, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            size = os.path.getsize(path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            self._count("write_errors")
            return
        with self._mu:
            if self._dir != d:          # re-attached elsewhere meanwhile
                return
            stale = self._stale(gen, key, prefix)
            if not stale:
                self._index[name] = size
                self.counters["writes"] += 1
            else:
                self.counters["stale_writes"] += 1
        if stale:
            try:
                os.remove(path)
            except OSError:
                pass
            return
        self._trim(d, disk_cap_bytes())

    def _trim(self, d: str, cap: int) -> None:
        """Remove the least recently used files until the folder fits."""
        try:
            with self._mu:
                total = sum(self._index.values())
                names = list(self._index)
            if total <= cap:
                return
            files = []
            for name in names:
                try:
                    files.append((os.stat(os.path.join(d, name)).st_mtime_ns,
                                  name))
                except OSError:
                    with self._mu:
                        total -= self._index.pop(name, 0)
            for _mt, name in sorted(files):
                if total <= cap:
                    break
                try:
                    os.remove(os.path.join(d, name))
                except FileNotFoundError:
                    pass
                except OSError:
                    continue
                with self._mu:
                    total -= self._index.pop(name, 0)
        except Exception:
            pass

    def disk_len(self) -> int:
        with self._mu:
            return len(self._index)

    def disk_bytes(self) -> int:
        with self._mu:
            return sum(self._index.values())

    # ── removal ──────────────────────────────────────────────────────────
    def forget(self, keys: Iterable[str], count: bool = True) -> int:
        """Drop these takes from both tiers (any voice); a write of one still
        queued never lands. Returns how many entries went. ``count`` False:
        an internal drop (not the owner's "forget that line")."""
        n = 0
        try:
            want = {str(k) for k in keys if k}
            if not want:
                return 0
            with self._mu:
                self._gen += 1
                for k in want:
                    self._forgot[k] = self._gen
                    self._forgot.move_to_end(k)
                while len(self._forgot) > _FORGOT_MAX:
                    self._forgot.popitem(last=False)
                for k in [k for k in self._mem if k in want]:
                    gone = self._mem.pop(k)
                    self._mem_bytes -= int(gone[0].nbytes)
                    n += 1
                d = self._dir
                names = [nm for nm in self._index
                         if _FILE_RE.match(nm).group(2) in want]
            for nm in names:
                if self._remove_file(d, nm):
                    n += 1
            if count:
                self._count("forgotten", n)
        except Exception:
            pass
        return n

    def _remove_file(self, d: Optional[str], name: str) -> bool:
        if d is None:
            return False
        try:
            os.remove(os.path.join(d, name))
        except FileNotFoundError:
            pass
        except OSError:
            return False
        with self._mu:
            self._index.pop(name, None)
        return True

    def purge_except(self, keep_prefixes: Iterable[str]) -> int:
        """Drop every take (memory and disk) whose voice prefix is not in
        `keep_prefixes`: a revoked or replaced voice. Returns how many went.
        Files on disk that are not in the index are found by a listing
        (a crash's temporary files of those voices too), and a write of
        theirs still queued never lands."""
        n = 0
        try:
            keep = {str(p) for p in keep_prefixes if p}
            with self._mu:
                self._gen += 1
                self._purge = (self._gen, frozenset(keep))
                for k in [k for k, v in self._mem.items() if v[2] not in keep]:
                    gone = self._mem.pop(k)
                    self._mem_bytes -= int(gone[0].nbytes)
                    n += 1
                d = self._dir
            self.gate.drop_except(keep)
            if d is not None:
                try:
                    names = os.listdir(d)
                except OSError:
                    names = []
                for name in names:
                    m = _FILE_RE.match(name) or _TMP_RE.match(name)
                    if m and m.group(1) not in keep:
                        if self._remove_file(d, name):
                            n += 1
            self._count("purged", n)
        except Exception:
            pass
        return n

    def wipe(self, since: Optional[float] = None,
             until: Optional[float] = None) -> int:
        """A memory wipe (core.actions reset_memory / forget_last_hour,
        review 2026-10-09). ``since`` None: every take, both tiers. Else
        every take on disk written or played in [``since``, ``until``]
        (time.time(); ``until`` None = no end) is removed, and the memory
        tier is emptied too - it keeps no times, so it cannot tell that
        hour's takes from older ones (an older one is read from the disk
        again); those are not counted. A write queued before the wipe never
        lands (a temporary file mid-write is removed by its writer). The
        take gate (lengths only, no words) is kept. Returns how many takes
        went (one per line, whichever tiers held it). Never raises."""
        gone: set = set()
        try:
            lo = None if since is None else float(since)
            hi = float("inf") if until is None else float(until)
            with self._mu:
                self._gen += 1
                self._wiped = self._gen
                if lo is None:
                    gone.update(self._mem)
                self._mem.clear()
                self._mem_bytes = 0
                d = self._dir
            if d is not None:
                try:
                    names = os.listdir(d)
                except OSError:
                    names = []
                for name in names:
                    m = _FILE_RE.match(name)
                    if not m:
                        continue
                    if lo is not None:
                        try:
                            mt = os.stat(os.path.join(d, name)).st_mtime
                        except OSError:
                            continue
                        if not lo <= mt <= hi:
                            continue
                    if self._remove_file(d, name):
                        gone.add(m.group(2))
            self._count("wiped", len(gone))
        except Exception:
            pass
        return len(gone)

    def prefixes(self) -> set:
        """Every voice prefix held, in either tier."""
        try:
            with self._mu:
                out = {v[2] for v in self._mem.values() if v[2]}
                out |= {_FILE_RE.match(nm).group(1) for nm in self._index}
            return out
        except Exception:
            return set()

    def save_gate(self) -> bool:
        """Persist the take gate's history when it changed. Never raises."""
        try:
            with self._mu:
                d = self._dir
            if d is None or not self.gate.dirty:
                return False
            from core.atomic_io import _atomic_write_json
            self.gate.dirty = False
            _atomic_write_json(os.path.join(d, GATE_FILE), self.gate.to_json(),
                               indent=None)
            return True
        except Exception:
            return False

    def stats(self) -> dict:
        with self._mu:
            out = dict(self.counters)
            out.update({"mem_entries": len(self._mem),
                        "mem_bytes": self._mem_bytes,
                        "disk_entries": len(self._index),
                        "disk_bytes": sum(self._index.values()),
                        "disk": self._dir is not None})
        return out
