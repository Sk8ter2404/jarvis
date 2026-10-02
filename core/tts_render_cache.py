"""core/tts_render_cache.py — finished Kokoro renders, served again on a repeat
(speed plan R4, 2026-10-02).

WHY THIS EXISTS
---------------
JARVIS says the same short lines over and over ("Of course, sir.", "Right
away, sir."), and every one costs a full CPU Kokoro render (~0.5 s) before the
first word. A render is a pure function of model, voice, language, speed and
text, so a repeat can play straight from memory.

KOKORO_RENDER_CACHE (core/config.py), read at CALL time:
  * 'off'    — core/kokoro_tts.synthesize() never consults this module;
  * 'shadow' — renders are stored and every lookup is logged as a would-hit or
               would-miss, but nothing is served (prove the hit rate first);
  * 'on'     — a hit is returned instead of rendering.

Contracts (core/kokoro_tts relies on every one):
  * the key is a sha256 over model size + mtime, voices-file mtime, voice,
    language, speed (3 dp) and the normalised text — the text itself is never
    stored, in memory or on disk;
  * entries are raw float32 mono arrays; put() stores a COPY and get()
    returns a COPY, because callers scale audio in place;
  * memory is capped at KOKORO_RENDER_CACHE_MB, least recently used out first;
  * KOKORO_RENDER_CACHE_PERSIST also keeps each entry as <key>.npy under the
    JARVIS data dir's tts_cache/ (same cap; written off the caller's thread).
    A corrupt or foreign file there is ignored, never served;
  * nothing here raises: a cache fault costs the cache, never a line of speech.

numpy is imported lazily (it is only needed once there is audio to store).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import OrderedDict

MODES = ("off", "shadow", "on")
SUBDIR = "tts_cache"
_DEFAULT_MB = 64
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")

# Short lines JARVIS opens with often enough to be worth rendering ahead of
# time (prefill_openers). Plain strings — the number normaliser leaves them be.
OPENERS = (
    "Of course, sir.",
    "Right away, sir.",
    "Certainly, sir.",
    "Very good, sir.",
    "One moment, sir.",
    "Just a moment, sir.",
    "On it, sir.",
    "Done, sir.",
    "Yes, sir.",
    "Yes, sir?",
    "Understood, sir.",
    "Noted, sir.",
    "As you wish, sir.",
    "Consider it done, sir.",
    "At once, sir.",
    "Good morning, sir.",
    "Good afternoon, sir.",
    "Good evening, sir.",
    "Welcome back, sir.",
    "You're welcome, sir.",
)


def _cfg(name: str, default):
    """A core.config knob read at call time; `default` on any error."""
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def mode() -> str:
    """KOKORO_RENDER_CACHE as one of MODES; anything unknown is 'off'."""
    m = str(_cfg("KOKORO_RENDER_CACHE", "off") or "off").strip().lower()
    return m if m in MODES else "off"


def _cap_bytes() -> int:
    try:
        mb = float(_cfg("KOKORO_RENDER_CACHE_MB", _DEFAULT_MB))
    except Exception:
        mb = float(_DEFAULT_MB)
    return max(0, int(mb * 1024 * 1024))


def _disk_dir():
    """<data dir>/tts_cache when KOKORO_RENDER_CACHE_PERSIST is on, else None.
    Resolved through core.paths so a staging / test redirect is honoured."""
    if not bool(_cfg("KOKORO_RENDER_CACHE_PERSIST", False)):
        return None
    try:
        from core.paths import data_dir
        return os.path.join(data_dir(), SUBDIR)
    except Exception:
        return None


def make_key(text: str, speed: float, voice: str, lang: str,
             model_path: str, voices_path: str):
    """The cache key for one render, or None when the model files can't be
    stat'ed (no key, no caching). A new model or voices file, another voice,
    language or speed, or any change to the text gives a different key."""
    try:
        m = os.stat(model_path)
        v = os.stat(voices_path)
        material = json.dumps(
            [m.st_size, m.st_mtime_ns, v.st_mtime_ns, str(voice), str(lang),
             round(float(speed), 3), str(text)], ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
    except Exception:
        return None


def _as_audio(a):
    """A float32 1-D non-empty COPY of `a`, or None."""
    import numpy as np
    a = np.array(a, dtype=np.float32, copy=True).reshape(-1)
    return a if a.size else None


class RenderCache:
    """Byte-capped LRU of float32 renders keyed by make_key(), with optional
    .npy persistence. Thread-safe; no method raises."""

    def __init__(self, async_writes: bool = True):
        self._lock = threading.Lock()
        self._mem: "OrderedDict[str, object]" = OrderedDict()
        self._bytes = 0
        self._async_writes = async_writes
        self.hits = 0
        self.misses = 0

    # -- memory ----------------------------------------------------------
    def _remember(self, key: str, a, cap: int) -> bool:
        """Store `a` (already our own copy) under `key`, evicting the least
        recently used entries until the total fits `cap`."""
        if a.nbytes > cap:
            return False
        with self._lock:
            old = self._mem.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._mem[key] = a
            self._bytes += a.nbytes
            while self._bytes > cap and self._mem:
                _, gone = self._mem.popitem(last=False)
                self._bytes -= gone.nbytes
        return True

    # -- disk ------------------------------------------------------------
    def _load(self, key: str):
        """The persisted entry for `key`, or None (absent, corrupt, not a
        float32 mono array, persistence off)."""
        d = _disk_dir()
        if d is None:
            return None
        path = os.path.join(d, key + ".npy")
        try:
            if not os.path.isfile(path):
                return None
            import numpy as np
            a = np.load(path, allow_pickle=False)
            if (not isinstance(a, np.ndarray) or a.dtype != np.float32
                    or a.ndim != 1 or not a.size):
                return None
            try:
                os.utime(path)              # recently used: trimmed last
            except OSError:
                pass
            return np.ascontiguousarray(a)
        except Exception:
            return None

    def _write(self, d: str, key: str, a, cap: int) -> None:
        """Write `a` as <d>/<key>.npy (atomic replace), then trim the folder
        to `cap`, oldest first."""
        tmp = None
        try:
            import numpy as np
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, key + ".npy")
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                np.save(f, a, allow_pickle=False)
            os.replace(tmp, path)
        except Exception:
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return
        try:
            files = []
            for name in os.listdir(d):
                if name.endswith(".npy"):
                    p = os.path.join(d, name)
                    st = os.stat(p)
                    files.append((st.st_mtime_ns, st.st_size, p))
            total = sum(f[1] for f in files)
            for _, size, p in sorted(files):
                if total <= cap:
                    break
                try:
                    os.remove(p)
                    total -= size
                except OSError:
                    pass
        except Exception:
            pass

    # -- public ----------------------------------------------------------
    def get(self, key):
        """A COPY of the cached render for `key`, or None. Never raises."""
        if not key or not _KEY_RE.match(str(key)):
            return None
        try:
            with self._lock:
                a = self._mem.get(key)
                if a is not None:
                    self._mem.move_to_end(key)
                    return a.copy()
            a = self._load(key)
            if a is None:
                return None
            self._remember(key, a, _cap_bytes())
            return a.copy()
        except Exception:
            return None

    def contains(self, key) -> bool:
        """True when get(key) would find something (memory, or a readable
        persisted file). Never raises."""
        if not key or not _KEY_RE.match(str(key)):
            return False
        try:
            with self._lock:
                if key in self._mem:
                    return True
            return self._load(key) is not None
        except Exception:
            return False

    def put(self, key, audio) -> bool:
        """Store a COPY of `audio` (float32 mono) under `key`. True when it was
        kept. Never raises."""
        if not key or not _KEY_RE.match(str(key)):
            return False
        try:
            a = _as_audio(audio)
            if a is None:
                return False
            cap = _cap_bytes()
            if not self._remember(key, a, cap):
                return False
            d = _disk_dir()
            if d is not None:
                if self._async_writes:
                    threading.Thread(target=self._write, args=(d, key, a, cap),
                                     daemon=True, name="tts-cache-write").start()
                else:
                    self._write(d, key, a, cap)
            return True
        except Exception:
            return False

    def lookup(self, key, serve: bool):
        """One synthesize() lookup: counts a hit or a miss. serve=True ('on')
        returns a COPY of a hit; serve=False ('shadow') logs would-hit /
        would-miss and always returns None. Never raises."""
        try:
            if serve:
                a = self.get(key)
                hit = a is not None
            else:
                a = None
                hit = self.contains(key)
            with self._lock:
                if hit:
                    self.hits += 1
                else:
                    self.misses += 1
                h, n = self.hits, self.hits + self.misses
            if not serve:
                print(f"  [tts-cache] shadow {'would-hit' if hit else 'would-miss'}"
                      f" ({h}/{n} would have been served)")
            return a
        except Exception:
            return None

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._mem), "bytes": self._bytes,
                    "hits": self.hits, "misses": self.misses}

    def clear(self) -> None:
        """Forget every in-memory entry and the counters (disk files stay)."""
        with self._lock:
            self._mem.clear()
            self._bytes = 0
            self.hits = 0
            self.misses = 0


CACHE = RenderCache()

_PREFILL_LOCK = threading.Lock()


def _stop_requested(stop) -> bool:
    """`stop` is a threading.Event, a no-arg callable, or None. A stop signal
    that can't be read counts as a stop."""
    try:
        if stop is None:
            return False
        is_set = getattr(stop, "is_set", None)
        if callable(is_set):
            return bool(is_set())
        return bool(stop()) if callable(stop) else False
    except Exception:
        return True


def prefill_openers(stop_event_or_callable, speed: float = 1.0) -> int:
    """Render each OPENERS line into the cache ahead of time, one at a time,
    skipping lines already cached, and stop as soon as `stop_event_or_callable`
    says so (an Event that is set, or a callable returning True). Single-
    flight: a call while another prefill runs returns 0 at once. Does nothing
    while the cache is 'off'. Returns how many lines it rendered. Never
    raises. Not wired into boot — the caller decides when (it costs one CPU
    render per line)."""
    if not _PREFILL_LOCK.acquire(blocking=False):
        return 0
    done = 0
    try:
        if mode() == "off":
            return 0
        from core import kokoro_tts as _k
        for line in OPENERS:
            if _stop_requested(stop_event_or_callable):
                break
            if _k.fill_cache(line, speed=speed):
                done += 1
    except Exception as e:
        print(f"  [tts-cache] prefill stopped ({type(e).__name__}: {e})")
    finally:
        _PREFILL_LOCK.release()
    return done
