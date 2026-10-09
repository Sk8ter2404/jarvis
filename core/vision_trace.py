"""core/vision_trace.py - the owner-approved record of what JARVIS's eyes
were given and what they answered (2026-10-05).

WHY THIS EXISTS
===============
Live 00:28-00:31: five vision calls, six brain rounds, zero clicks landed,
and afterwards nobody could say what the model had actually been SHOWN -
the four 1024-px monitor shots, the crop, the question, the raw answer.
At 00:32:48 the owner asked for exactly that ("so Claude can see what you
see, that way it can learn") and confirmed it in the Claude chat as "Yes,
with limits". This is that trace, inside his limits:

  * LOCAL ONLY. ``data/vision_trace/`` through core.paths (``data/*`` is
    gitignored). Nothing here uploads, sends or syncs anything.
  * BOUNDED. The writer itself rotates out, oldest first, everything older
    than VISION_TRACE_DAYS (7), then beyond VISION_TRACE_MAX_ENTRIES (300),
    then beyond VISION_TRACE_MAX_MB (300) - on every write, files deleted
    with their entries.
  * PRIVATE IS NOT SAVED. A step over a SCREENSHOT_PRIVACY_BLOCKLIST /
    sensitive / sign-in / password / owner-excluded window is a text-only
    entry ``{"privacy": "skipped: private"}`` - no title, URL, prompt,
    answer or image.
  * VISION_TRACE = "on" (images + text) | "text" (no images) | "off".

One entry per vision step (see_screen, find_on_screen, each click tier
attempt, verify, undo, recall_screen, page reads): the exact images as
sent (WebP q85, after downscale and masking), the prompt summary and its
sha1, the raw answer, parsed coordinates, the candidates, the action taken
and the verified outcome. ``tools/vision_trace_report.py`` summarises the
index as TEXT; a debugging session opens an image only with the owner's
go-ahead.

Mechanics: a contextvar holds the CURRENT step (``step(...)``), opened only
by an owner-turn step owner; core-level hooks (the local vision chokepoint)
add model calls to it. Background callers with no step are never traced.
One never-exiting writer thread with a bounded queue does every disk write
and every WebP encode, off the voice path; overflow is counted, never
blocking. Never raises at the public API.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import io
import json
import os
import queue
import threading
import time
import uuid

__all__ = [
    "Step", "step", "current", "note_model_call", "note_image", "record",
    "mode", "trace_dir", "index_path", "flush", "purge", "prune",
    "read_index", "stats", "set_context_provider", "PRIVATE_SKIP",
    "PAUSED_SKIP", "GUEST_SKIP",
]

PRIVATE_SKIP = "skipped: private"
# While the owner's "stop watching" pause runs (core.screen_memory.
# owner_paused) a step keeps nothing but this marker - no title, URL,
# prompt, answer or image (review 2026-10-05: the pause stopped only the
# background watcher, and every click kept tracing).
PAUSED_SKIP = "skipped: paused"
_PAUSED = "paused by the owner"
# Guest mode (core.guest_mode: visitors are in the room, nothing said is
# kept) keeps the same bare marker - no utterance, title, URL, prompt,
# answer or image (review 2026-10-09: every look and click traced the
# owner's - or a guest's - words for 7 days with guest mode on).
GUEST_SKIP = "skipped: guest mode"
_GUEST = "guest mode"


def _paused_reason() -> str:
    try:
        from core import guest_mode as _gm
        if _gm.is_on():
            return _GUEST
    except Exception:
        pass
    try:
        from core import screen_memory as _sm
        return _PAUSED if _sm.owner_paused() else ""
    except Exception:
        return ""


_QUEUE_MAX = 64
_PENDING_IMAGE_BYTES_MAX = 48 * 1024 * 1024
_COMPACT_EVERY = 25
_PROMPT_MAX = 600

_current: contextvars.ContextVar = contextvars.ContextVar(
    "jarvis_vision_trace_step", default=None)
_q: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAX)
_state_lock = threading.Lock()
_state = {"thread": None, "writes": 0, "overflow": 0, "overflow_logged": 0.0,
          "pending_bytes": 0, "loaded": False, "entries": 0, "bytes": 0,
          "errors": 0}
_ctx_provider = [None]
_disk_lock = threading.Lock()


def _cfg(name, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def mode() -> str:
    """"on" | "text" | "off" (VISION_TRACE; anything else reads as "on" -
    the owner's choice - except a falsy value, which is "off")."""
    try:
        v = _cfg("VISION_TRACE", "on")
        if v is False or v is None:
            return "off"
        s = str(v).strip().lower()
        if s in ("off", "false", "0", "no", "none", "disabled"):
            return "off"
        if s in ("text", "text-only", "textonly"):
            return "text"
        return "on"
    except Exception:
        return "on"


def _limits() -> tuple:
    def _num(name, default, cast):
        try:
            v = cast(_cfg(name, default))
            return v if v > 0 else default
        except Exception:
            return default
    return (_num("VISION_TRACE_DAYS", 7, float),
            _num("VISION_TRACE_MAX_ENTRIES", 300, int),
            _num("VISION_TRACE_MAX_MB", 300, float))


def trace_dir(create: bool = True) -> str:
    from core.paths import data_dir
    d = os.path.join(data_dir(create=create), "vision_trace")
    if create:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
    return d


def index_path() -> str:
    return os.path.join(trace_dir(), "index.jsonl")


def set_context_provider(fn) -> None:
    """``fn() -> {"session_log", "log_offset", "version", "turn_id"}`` (the
    monolith registers one at boot). Never raises."""
    _ctx_provider[0] = fn if callable(fn) else None


def _context() -> dict:
    out = {}
    try:
        from core.version import version_string
        out["version"] = version_string()
    except Exception:
        pass
    try:
        fn = _ctx_provider[0]
        if fn is not None:
            got = fn() or {}
            if isinstance(got, dict):
                out.update({k: got[k] for k in ("session_log", "log_offset",
                                                 "version", "turn_id")
                            if k in got})
    except Exception:
        pass
    return out


class Step:
    """One vision step being recorded. Fill it, then ``finish``."""

    def __init__(self, name, *, utterance="", source="", scope=None,
                 privacy=None):
        self.id = f"vt-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.t0 = time.time()
        self.entry: dict = {
            "id": self.id, "ts": round(self.t0, 3), "step": str(name),
            "utterance": str(utterance or ""), "source": str(source or ""),
            "scope": dict(scope or {}), "images": [], "prompt": "",
            "prompt_sha1": "", "raw_answer": "", "parsed": None,
            "candidates": [], "chosen": None, "action": None,
            "precheck": None, "outcome": None, "evidence": "",
            "latency_ms": {}, "cloud": False, "privacy": "",
            "model_calls": [],
        }
        self.private = str(privacy or "") or _paused_reason()
        self._images: list = []         # (payload, meta) - encoded by writer
        self._done = False
        self.entry.update(_context())

    # -- filling ------------------------------------------------------
    def set(self, **fields) -> "Step":
        try:
            for k, v in fields.items():
                if k == "latency_ms" and isinstance(v, dict):
                    self.entry["latency_ms"].update(v)
                elif k == "scope" and isinstance(v, dict):
                    self.entry["scope"].update(v)
                else:
                    self.entry[k] = v
        except Exception:
            pass
        return self

    def mark_private(self, reason) -> "Step":
        self.private = str(reason or "private")
        return self

    def add_image(self, img, *, as_sent="", native_region=None, scale=1.0,
                  est_tokens=None, label="") -> "Step":
        """Attach an image exactly as the model got it: PNG/JPEG bytes or a
        PIL image (copied). Encoded to WebP by the writer thread."""
        try:
            if mode() != "on" or self.private:
                return self
            payload = img
            if hasattr(img, "copy") and not isinstance(img, (bytes, bytearray)):
                payload = img.copy()
            elif isinstance(img, (bytes, bytearray)):
                payload = bytes(img)
            else:
                return self
            meta = {"as_sent": str(as_sent), "native_region": native_region,
                    "scale": round(float(scale or 1.0), 4),
                    "est_tokens": est_tokens, "label": str(label or "")}
            self._images.append((payload, meta))
        except Exception:
            pass
        return self

    def model_call(self, prompt, images=(), answer="", ms=None,
                   cloud=False) -> "Step":
        """A vision-model call made inside this step (from the chokepoint
        hook): the prompt, the images as sent, the raw answer."""
        try:
            p = str(prompt or "")
            call = {"prompt": p[:_PROMPT_MAX],
                    "prompt_sha1": hashlib.sha1(p.encode("utf-8",
                                                         "replace")).hexdigest(),
                    "raw_answer": str(answer or ""),
                    "n_images": len(list(images or ())),
                    "ms": None if ms is None else int(ms), "cloud": bool(cloud)}
            self.entry["model_calls"].append(call)
            if not self.entry["prompt"]:
                self.entry["prompt"] = call["prompt"]
                self.entry["prompt_sha1"] = call["prompt_sha1"]
            self.entry["raw_answer"] = call["raw_answer"]
            if cloud:
                self.entry["cloud"] = True
            for i, png in enumerate(images or ()):
                size = ""
                try:
                    from PIL import Image
                    with Image.open(io.BytesIO(png)) as im:
                        size = f"{im.size[0]}x{im.size[1]}"
                        est = _est(im.size)
                except Exception:
                    est = None
                self.add_image(png, as_sent=size, est_tokens=est,
                               label=f"call{len(self.entry['model_calls'])}"
                                     f"_img{i + 1}")
        except Exception:
            pass
        return self

    # -- closing ------------------------------------------------------
    def finish(self, outcome=None, evidence="", **fields) -> str:
        """Queue the entry (once). Returns the id ('' when not traced)."""
        try:
            if self._done:
                return self.id
            self._done = True
            if outcome is not None:
                self.entry["outcome"] = str(outcome)
            if evidence:
                self.entry["evidence"] = str(evidence)[:600]
            self.set(**fields)
            self.entry["latency_ms"].setdefault(
                "total", int((time.time() - self.t0) * 1000))
            return _enqueue(self._final_entry(), self._images)
        except Exception:
            return ""

    def _final_entry(self) -> dict:
        if self.private:
            return {"id": self.id, "ts": self.entry["ts"],
                    "step": self.entry["step"],
                    "outcome": self.entry.get("outcome"),
                    "privacy": (PAUSED_SKIP if self.private == _PAUSED
                                else GUEST_SKIP if self.private == _GUEST
                                else PRIVATE_SKIP
                                if "excluded" not in self.private
                                else "skipped: excluded")}
        e = dict(self.entry)
        e["privacy"] = ""
        e["candidates"] = list(e.get("candidates") or [])[:20]
        return e


def _est(size):
    try:
        from core.vision_grounding import est_tokens
        return est_tokens(*size)
    except Exception:
        return None


class _NullStep:
    """What step() yields when nothing is traced: every method a no-op."""
    id = ""
    private = ""
    entry: dict = {}

    def __bool__(self):
        return False

    def set(self, **_k):
        return self

    def mark_private(self, _r):
        return self

    def add_image(self, *_a, **_k):
        return self

    def model_call(self, *_a, **_k):
        return self

    def finish(self, *_a, **_k):
        return ""


NULL_STEP = _NullStep()


@contextlib.contextmanager
def step(name, *, utterance="", source="", scope=None, privacy=None,
         force: bool = False):
    """Open a step for the duration of a ``with`` block (it becomes
    current()). Nothing is opened - a falsy no-op step is yielded - when
    the trace is off, or when there is no owner utterance and not ``force``
    (background callers are never traced). The step is finished on exit if
    the block did not finish it."""
    if mode() == "off" or (not str(utterance or "").strip() and not force):
        yield NULL_STEP
        return
    st = Step(name, utterance=utterance, source=source, scope=scope,
              privacy=privacy)
    token = _current.set(st)
    try:
        yield st
    finally:
        try:
            _current.reset(token)
        except Exception:
            _current.set(None)
        if not st._done:
            st.finish(st.entry.get("outcome") or "unfinished")


def current():
    """The step open in this context, else None."""
    try:
        return _current.get()
    except Exception:
        return None


def note_model_call(prompt, images=(), answer="", ms=None, cloud=False) -> None:
    """Hook for the vision chokepoint: attach the call to the current step,
    if one is open. A background call (no step) is not traced."""
    st = current()
    if st is not None:
        st.model_call(prompt, images, answer, ms=ms, cloud=cloud)


def note_image(img, **meta) -> None:
    st = current()
    if st is not None:
        st.add_image(img, **meta)


def record(name, *, utterance="", outcome=None, privacy=None, images=(),
           **fields) -> str:
    """One-shot entry (no ``with``). Returns its id ('' when not traced)."""
    try:
        if mode() == "off" or not str(utterance or "").strip():
            return ""
        st = Step(name, utterance=utterance, privacy=privacy)
        for img in images or ():
            st.add_image(img)
        return st.finish(outcome, **fields)
    except Exception:
        return ""


# ── writer ───────────────────────────────────────────────────────────────
def _payload_bytes(images) -> int:
    n = 0
    for payload, _meta in images or ():
        if isinstance(payload, (bytes, bytearray)):
            n += len(payload)
        else:
            try:
                w, h = payload.size
                n += w * h * 3
            except Exception:
                n += 1 << 20
    return n


def _enqueue(entry: dict, images) -> str:
    try:
        images = list(images or ()) if mode() == "on" else []
        nbytes = _payload_bytes(images)
        with _state_lock:
            if _state["pending_bytes"] + nbytes > _PENDING_IMAGE_BYTES_MAX:
                images, nbytes = [], 0
                entry = dict(entry, images_dropped="writer queue full")
            _state["pending_bytes"] += nbytes
        _ensure_thread()
        try:
            _q.put_nowait((entry, images, nbytes))
        except queue.Full:
            with _state_lock:
                _state["overflow"] += 1
                _state["pending_bytes"] -= nbytes
                n = _state["overflow"]
                log = time.time() - _state["overflow_logged"] > 60
                if log:
                    _state["overflow_logged"] = time.time()
            if log:
                print(f"  [vision-trace] writer queue full - {n} entr"
                      f"{'y' if n == 1 else 'ies'} dropped so far", flush=True)
            return ""
        out = entry.get("outcome")
        print(f"  [vision-trace] id={entry['id']} step={entry.get('step')} "
              f"source={entry.get('source') or '-'} outcome={out}"
              + (f" privacy={entry['privacy']}" if entry.get("privacy") else ""),
              flush=True)
        return entry["id"]
    except Exception:
        return ""


def _ensure_thread() -> None:
    with _state_lock:
        t = _state["thread"]
        if t is not None and t.is_alive():
            return
        t = threading.Thread(target=_writer_loop, name="vision-trace",
                             daemon=True)
        _state["thread"] = t
        t.start()


def _writer_loop() -> None:          # never exits (the ct2_host lesson)
    while True:
        try:
            item = _q.get()
        except Exception:
            time.sleep(0.5)
            continue
        try:
            entry, images, nbytes = item
            try:
                _write(entry, images)
            finally:
                with _state_lock:
                    _state["pending_bytes"] = max(
                        0, _state["pending_bytes"] - nbytes)
        except Exception as e:
            with _state_lock:
                _state["errors"] += 1
            try:
                print(f"  [vision-trace] write failed: {type(e).__name__}: {e}",
                      flush=True)
            except Exception:
                pass
        finally:
            try:
                _q.task_done()
            except Exception:
                pass


def _encode_webp(payload) -> bytes:
    from PIL import Image
    if isinstance(payload, (bytes, bytearray)):
        img = Image.open(io.BytesIO(payload))
        img.load()
    else:
        img = payload
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=85, method=4)
    return buf.getvalue()


def _write(entry: dict, images) -> None:
    with _disk_lock:
        base = trace_dir()
        _load_counts(base)
        files = []
        added = 0
        if images and not entry.get("privacy"):
            day = time.strftime("%Y%m%d", time.localtime(entry.get("ts")
                                                          or time.time()))
            ddir = os.path.join(base, day)
            os.makedirs(ddir, exist_ok=True)
            for n, (payload, meta) in enumerate(images, 1):
                try:
                    data = _encode_webp(payload)
                except Exception:
                    continue
                rel = f"{day}/{entry['id']}_{n}.webp"
                with open(os.path.join(base, *rel.split("/")), "wb") as f:
                    f.write(data)
                added += len(data)
                files.append(dict(meta, file=rel, bytes=len(data)))
        entry = dict(entry)
        if not entry.get("privacy"):
            entry["images"] = files
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with open(os.path.join(base, "index.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(line)
        with _state_lock:
            _state["writes"] += 1
            _state["entries"] += 1
            _state["bytes"] += added + len(line.encode("utf-8"))
            writes = _state["writes"]
            entries, nbytes = _state["entries"], _state["bytes"]
        days, max_entries, max_mb = _limits()
        if (writes % _COMPACT_EVERY == 0 or entries > max_entries
                or nbytes > max_mb * 1024 * 1024):
            _compact(base)


def _load_counts(base) -> None:
    with _state_lock:
        if _state["loaded"]:
            return
    entries = _read_entries(base)
    nbytes = _dir_bytes(base)
    with _state_lock:
        _state["entries"] = len(entries)
        _state["bytes"] = nbytes
        _state["loaded"] = True


def _dir_bytes(base) -> int:
    total = 0
    for root, _dirs, fs in os.walk(base):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _read_entries(base) -> list:
    out = []
    try:
        with open(os.path.join(base, "index.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if isinstance(e, dict) and e.get("id"):
                        out.append(e)
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    except Exception:
        return out
    return out


def _entry_bytes(base, e) -> int:
    n = len(json.dumps(e, ensure_ascii=False, default=str).encode("utf-8")) + 1
    for im in e.get("images") or ():
        try:
            n += int(im.get("bytes") or os.path.getsize(
                os.path.join(base, *str(im["file"]).split("/"))))
        except Exception:
            pass
    return n


def _delete_files(base, e) -> None:
    for im in e.get("images") or ():
        try:
            os.remove(os.path.join(base, *str(im["file"]).split("/")))
        except Exception:
            pass


def _rewrite_index(base, keep) -> None:
    tmp = os.path.join(base, f"index.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in keep:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, os.path.join(base, "index.jsonl"))


def _remove_empty_dirs(base) -> None:
    try:
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if os.path.isdir(p) and not os.listdir(p):
                os.rmdir(p)
    except Exception:
        pass


def _compact(base, now=None) -> dict:
    """Rotate out (oldest first): age > days, then count > max entries, then
    bytes > max MB. Files go with their entries. Caller holds _disk_lock."""
    days, max_entries, max_mb = _limits()
    now = time.time() if now is None else float(now)
    entries = _read_entries(base)
    entries.sort(key=lambda e: float(e.get("ts") or 0))
    dropped = {"age": 0, "count": 0, "size": 0}
    keep = []
    for e in entries:
        if now - float(e.get("ts") or 0) > days * 86400:
            _delete_files(base, e)
            dropped["age"] += 1
        else:
            keep.append(e)
    while len(keep) > max_entries:
        _delete_files(base, keep.pop(0))
        dropped["count"] += 1
    sizes = [_entry_bytes(base, e) for e in keep]
    total = sum(sizes)
    cap = max_mb * 1024 * 1024
    while keep and total > cap:
        total -= sizes.pop(0)
        _delete_files(base, keep.pop(0))
        dropped["size"] += 1
    _rewrite_index(base, keep)
    _remove_empty_dirs(base)
    with _state_lock:
        _state["entries"] = len(keep)
        _state["bytes"] = _dir_bytes(base)
    if any(dropped.values()):
        print(f"  [vision-trace] rotated out {sum(dropped.values())} entr"
              f"{'y' if sum(dropped.values()) == 1 else 'ies'} "
              f"(age {dropped['age']}, count {dropped['count']}, size "
              f"{dropped['size']})", flush=True)
    return dropped


def prune(now=None) -> dict:
    """Apply the retention limits now. Never raises."""
    try:
        with _disk_lock:
            base = trace_dir()
            if not os.path.exists(os.path.join(base, "index.jsonl")):
                return {"age": 0, "count": 0, "size": 0}
            return _compact(base, now=now)
    except Exception:
        return {"age": 0, "count": 0, "size": 0}


def purge(since_ts=None, until_ts=None) -> int:
    """Delete every entry (and its images) with ts in [since, until] - the
    "forget the last hour" path. Returns how many. Never raises."""
    try:
        flush(2.0)
        with _disk_lock:
            base = trace_dir()
            entries = _read_entries(base)
            lo = float(since_ts) if since_ts is not None else float("-inf")
            hi = float(until_ts) if until_ts is not None else float("inf")
            keep, gone = [], 0
            for e in entries:
                ts = float(e.get("ts") or 0)
                if lo <= ts <= hi:
                    _delete_files(base, e)
                    gone += 1
                else:
                    keep.append(e)
            if gone:
                _rewrite_index(base, keep)
                _remove_empty_dirs(base)
                with _state_lock:
                    _state["entries"] = len(keep)
                    _state["bytes"] = _dir_bytes(base)
            return gone
    except Exception:
        return 0


def flush(timeout: float = 5.0) -> bool:
    """Wait (bounded) until every queued entry is written. True when the
    queue drained."""
    deadline = time.time() + max(0.0, float(timeout))
    while time.time() < deadline:
        if _q.unfinished_tasks == 0:
            return True
        time.sleep(0.02)
    return _q.unfinished_tasks == 0


def read_index(limit=None) -> list:
    """Entries, oldest first (text only - images are file names)."""
    try:
        with _disk_lock:
            out = _read_entries(trace_dir(create=False))
        return out[-int(limit):] if limit else out
    except Exception:
        return []


def stats() -> dict:
    with _state_lock:
        return {k: _state[k] for k in ("writes", "overflow", "entries",
                                        "bytes", "errors", "pending_bytes")}


def _reset_for_tests() -> None:
    """Forget the cached counts (a test pointed JARVIS_DATA_DIR elsewhere)."""
    flush(5.0)
    with _state_lock:
        _state.update(writes=0, overflow=0, loaded=False, entries=0, bytes=0,
                      errors=0, pending_bytes=0)
