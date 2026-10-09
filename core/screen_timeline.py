"""core/screen_timeline.py - JARVIS's text memory of what was on the screen
(2026-10-05).

WHY THIS EXISTS
===============
Live 00:29:49 "I wanted the one that was on screen at the time": JARVIS had
no record of what had been on screen, only five cached screenshots, so
recall_screen re-asked the vision model and it invented "the Kai Cenat
video"; the next three rounds hunted for a video that never existed. The
owner then asked (00:34) for JARVIS to watch the screen constantly "so he
has a constant memory of what's going on", cheaply.

This is that memory, and it is TEXT ONLY: window titles, the browser's
address (query string stripped), and the words UI Automation / OCR read on
the page - each row timestamped, per monitor and window. recall_screen
answers from these rows with their times and says "I have no record" when
there is none; it never asks a model who something was.

Store: ``data/screen_timeline.db`` (core.paths; ``data/*`` is gitignored),
SQLite WAL, secure_delete on, incremental auto-vacuum, an FTS5 index over
(title, url, text). Every line passes core.memory_guards._is_secret_fact (a
line that looks like a credential is dropped). Retention: older than
SCREEN_TIMELINE_DAYS (7) or beyond SCREEN_TIMELINE_MAX_MB (200) is pruned
at start-up and hourly. "Forget the last hour" deletes a span.

Rows come from on-demand looks, clicks and scene freezes (always), and from
the continuous watcher (core.screen_memory) when AMBIENT_SCREEN_ENABLED is on.
One writer thread fed by a queue does every write. Never raises.
"""
from __future__ import annotations

import os
import queue
import re
import sqlite3
import threading
import time
import urllib.parse

__all__ = ["Timeline", "get", "add", "clean_url", "SOURCES", "set_enabled",
           "start_pruner", "prune_all"]

SOURCES = ("focus", "title", "uia", "ocr", "look", "click", "scene", "action")
_KEEP_QUERY = ("v", "list", "t", "q", "search_query")
_MAX_TEXT = 4000
_QUEUE_MAX = 512
# Queued by Timeline.close(): the writer thread returns when it reaches it.
_STOP_WRITER = object()


def _cfg(name, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def clean_url(url) -> str:
    """``url`` with fragment and query dropped, except the parameters that
    NAME what is shown (v, list, t, q, search_query). Never raises."""
    try:
        s = str(url or "").strip()
        if not s:
            return ""
        parts = urllib.parse.urlsplit(s if "://" in s else "https://" + s)
        q = urllib.parse.parse_qsl(parts.query, keep_blank_values=False)
        kept = [(k, v) for k, v in q if k in _KEEP_QUERY]
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path,
                                        urllib.parse.urlencode(kept), ""))
    except Exception:
        return ""


def _secret(text) -> bool:
    try:
        from core.memory_guards import _is_secret_fact
        return bool(_is_secret_fact(str(text or "")))
    except Exception:
        return False


def _redact(text) -> str:
    """Lines that look like secrets dropped; whitespace collapsed."""
    out = []
    for line in str(text or "").splitlines() or [str(text or "")]:
        line = " ".join(line.split())
        if line and not _secret(line):
            out.append(line)
    return "\n".join(out)[:_MAX_TEXT]


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS obs(id INTEGER PRIMARY KEY, ts REAL NOT NULL,"
    " monitor TEXT, hwnd INTEGER, process TEXT, title TEXT, url TEXT,"
    " source TEXT, text TEXT)",
    "CREATE INDEX IF NOT EXISTS obs_ts ON obs(ts)",
    "CREATE VIRTUAL TABLE IF NOT EXISTS obs_fts USING fts5(title, url, text,"
    " content='obs', content_rowid='id')",
    "CREATE TRIGGER IF NOT EXISTS obs_ai AFTER INSERT ON obs BEGIN"
    " INSERT INTO obs_fts(rowid, title, url, text)"
    " VALUES (new.id, new.title, new.url, new.text); END",
    "CREATE TRIGGER IF NOT EXISTS obs_ad AFTER DELETE ON obs BEGIN"
    " INSERT INTO obs_fts(obs_fts, rowid, title, url, text)"
    " VALUES ('delete', old.id, old.title, old.url, old.text); END",
    "CREATE TRIGGER IF NOT EXISTS obs_au AFTER UPDATE ON obs BEGIN"
    " INSERT INTO obs_fts(obs_fts, rowid, title, url, text)"
    " VALUES ('delete', old.id, old.title, old.url, old.text);"
    " INSERT INTO obs_fts(rowid, title, url, text)"
    " VALUES (new.id, new.title, new.url, new.text); END",
)


class Timeline:
    """One SQLite timeline. Reads on the caller's thread (own connection,
    short); writes on one writer thread (``add``) or directly
    (``add_now``, tests and the writer itself)."""

    def __init__(self, path=None):
        if path is None:
            from core.paths import data_file
            path = data_file("screen_timeline.db")
        self.path = str(path)
        self._lock = threading.RLock()
        self._q: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread = None
        self._dropped = 0
        self._ready = False

    # -- connection ---------------------------------------------------
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=2.0, check_same_thread=False)
        con.execute("PRAGMA busy_timeout=2000")
        return con

    def _init(self) -> None:
        with self._lock:
            if self._ready:
                return
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            con = self._connect()
            try:
                # auto_vacuum must be set before the first table exists.
                con.execute("PRAGMA auto_vacuum=INCREMENTAL")
                con.execute("PRAGMA journal_mode=WAL")
                con.execute("PRAGMA secure_delete=ON")
                for stmt in _SCHEMA:
                    con.execute(stmt)
                con.commit()
            finally:
                con.close()
            self._ready = True

    # -- writing ------------------------------------------------------
    @staticmethod
    def _row(ts=None, monitor=None, hwnd=None, process="", title="", url="",
             source="look", text="") -> "tuple | None":
        src = str(source or "look")
        if src not in SOURCES:
            src = "look"
        title = _redact(title)[:400]
        text = _redact(text)
        url = clean_url(url)
        if not (title or text or url):
            return None
        return (float(time.time() if ts is None else ts),
                str(monitor) if monitor else None,
                int(hwnd) if isinstance(hwnd, int) else None,
                str(process or "")[:80], title, url, src, text)

    def add_now(self, **row) -> bool:
        """Insert one row on THIS thread. False when it carried nothing or
        failed."""
        try:
            r = self._row(**row)
            if r is None:
                return False
            self._init()
            with self._lock:
                con = self._connect()
                try:
                    con.execute("PRAGMA secure_delete=ON")
                    con.execute("INSERT INTO obs(ts, monitor, hwnd, process,"
                                " title, url, source, text)"
                                " VALUES (?,?,?,?,?,?,?,?)", r)
                    con.commit()
                finally:
                    con.close()
            return True
        except Exception:
            return False

    def add(self, **row) -> bool:
        """Queue one row for the writer thread (never blocks)."""
        try:
            self._ensure_writer()
            self._q.put_nowait(row)
            return True
        except queue.Full:
            self._dropped += 1
            return False
        except Exception:
            return False

    def _ensure_writer(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._writer_loop,
                                            name="screen-timeline",
                                            daemon=True)
            self._thread.start()

    def _writer_loop(self) -> None:          # exits only on close()
        while True:
            try:
                row = self._q.get()
            except Exception:
                time.sleep(0.5)
                continue
            try:
                if row is _STOP_WRITER:
                    return
                self.add_now(**row)
            except Exception:
                pass
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass

    def close(self, timeout: float = 2.0) -> None:
        """Stop the writer thread after the rows queued before this call; a
        later add() starts a new one. For a timeline that is done with:
        get() closes the one it replaces when the data dir changes (never
        in a running JARVIS; every test that redirects JARVIS_DATA_DIR -
        32 writer threads were left waiting in the 2026-10-09 rel-182
        run). ``timeout`` 0 = signal, do not wait. Never raises."""
        try:
            with self._lock:
                th = self._thread
                if th is None or not th.is_alive():
                    return
                self._thread = None
            t = max(0.0, float(timeout))
            if t:
                self._q.put(_STOP_WRITER, timeout=t)
                th.join(t)
            else:
                self._q.put_nowait(_STOP_WRITER)
        except Exception:
            pass

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + max(0.0, float(timeout))
        while time.time() < deadline:
            if self._q.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return self._q.unfinished_tasks == 0

    # -- reading ------------------------------------------------------
    def query(self, text=None, since=None, until=None, monitor=None,
              sources=None, url_like=None, limit: int = 20) -> list:
        """Rows (dicts, newest first) matching every given filter. ``text``
        is an FTS match over title / url / text (words AND-ed, each a
        prefix). Never raises."""
        try:
            if not os.path.exists(self.path):
                return []
            self._init()
            where, args = [], []
            if since is not None:
                where.append("o.ts >= ?")
                args.append(float(since))
            if until is not None:
                where.append("o.ts <= ?")
                args.append(float(until))
            if monitor:
                where.append("o.monitor = ?")
                args.append(str(monitor))
            if sources:
                srcs = [s for s in sources if s in SOURCES]
                if srcs:
                    where.append("o.source IN (%s)" % ",".join("?" * len(srcs)))
                    args.extend(srcs)
            if url_like:
                where.append("o.url LIKE ?")
                args.append(f"%{url_like}%")
            match = _fts_query(text)
            if match:
                sql = ("SELECT o.id, o.ts, o.monitor, o.hwnd, o.process, o.title,"
                       " o.url, o.source, o.text FROM obs o JOIN obs_fts f ON"
                       " f.rowid = o.id WHERE obs_fts MATCH ?")
                args.insert(0, match)
                if where:
                    sql += " AND " + " AND ".join(where)
            else:
                sql = ("SELECT o.id, o.ts, o.monitor, o.hwnd, o.process, o.title,"
                       " o.url, o.source, o.text FROM obs o")
                if where:
                    sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY o.ts DESC LIMIT ?"
            args.append(max(1, int(limit)))
            with self._lock:
                con = self._connect()
                try:
                    rows = con.execute(sql, args).fetchall()
                finally:
                    con.close()
            keys = ("id", "ts", "monitor", "hwnd", "process", "title", "url",
                    "source", "text")
            return [dict(zip(keys, r)) for r in rows]
        except Exception:
            return []

    def count(self, since=None) -> int:
        try:
            if not os.path.exists(self.path):
                return 0
            self._init()
            with self._lock:
                con = self._connect()
                try:
                    if since is None:
                        return int(con.execute(
                            "SELECT COUNT(*) FROM obs").fetchone()[0])
                    return int(con.execute(
                        "SELECT COUNT(*) FROM obs WHERE ts >= ?",
                        (float(since),)).fetchone()[0])
                finally:
                    con.close()
        except Exception:
            return 0

    # -- forgetting ---------------------------------------------------
    def forget(self, since=None, until=None, hwnd=None) -> int:
        """Delete rows in [since, until] (and of ``hwnd`` only, when given).
        Returns how many. Never raises."""
        try:
            if not os.path.exists(self.path):
                return 0
            self.flush(2.0)
            self._init()
            where, args = [], []
            if since is not None:
                where.append("ts >= ?")
                args.append(float(since))
            if until is not None:
                where.append("ts <= ?")
                args.append(float(until))
            if hwnd is not None:
                where.append("hwnd = ?")
                args.append(int(hwnd))
            sql = "DELETE FROM obs" + (" WHERE " + " AND ".join(where)
                                       if where else "")
            with self._lock:
                con = self._connect()
                try:
                    con.execute("PRAGMA secure_delete=ON")
                    n = con.execute(sql, args).rowcount
                    con.commit()
                    con.execute("PRAGMA incremental_vacuum")
                    con.commit()
                finally:
                    con.close()
            return int(n or 0)
        except Exception:
            return 0

    def size_mb(self) -> float:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                total += os.path.getsize(self.path + suffix)
            except OSError:
                pass
        return total / (1024 * 1024)

    def prune(self, now=None, days=None, max_mb=None) -> dict:
        """Retention: rows older than ``days``, then the oldest until the
        file is under ``max_mb``. Never raises."""
        out = {"age": 0, "size": 0}
        try:
            if not os.path.exists(self.path):
                return out
            now = time.time() if now is None else float(now)
            days = float(days if days is not None
                         else _cfg("SCREEN_TIMELINE_DAYS", 7))
            max_mb = float(max_mb if max_mb is not None
                           else _cfg("SCREEN_TIMELINE_MAX_MB", 200))
            out["age"] = self.forget(until=now - days * 86400)
            guard = 0
            while self.size_mb() > max_mb and guard < 50:
                guard += 1
                with self._lock:
                    con = self._connect()
                    try:
                        total = con.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
                        if not total:
                            break
                        cut = max(1, total // 10)
                        con.execute("PRAGMA secure_delete=ON")
                        n = con.execute(
                            "DELETE FROM obs WHERE id IN (SELECT id FROM obs"
                            " ORDER BY ts ASC LIMIT ?)", (cut,)).rowcount
                        con.commit()
                        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        con.execute("PRAGMA incremental_vacuum")
                        con.commit()
                    finally:
                        con.close()
                out["size"] += int(n or 0)
        except Exception:
            return out
        return out


def _fts_query(text) -> str:
    """An FTS5 MATCH string: each word a quoted prefix term, AND-ed."""
    words = [w for w in re.findall(r"[A-Za-z0-9$]+", str(text or ""))
             if len(w) >= 2][:8]
    if not words:
        return ""
    return " AND ".join(f'"{w}"*' for w in words)


_singleton = {"tl": None, "enabled": True}
_s_lock = threading.Lock()


def set_enabled(on: bool) -> None:
    """The monolith turns writes off in a process that does not own the
    live data (a staging / test instance writes to its own data dir
    anyway)."""
    _singleton["enabled"] = bool(on)


def get() -> Timeline:
    """The process-wide timeline at the CURRENT data dir (a changed
    JARVIS_DATA_DIR gets a new one)."""
    from core.paths import data_file
    path = data_file("screen_timeline.db")
    old = None
    with _s_lock:
        tl = _singleton["tl"]
        if tl is None or tl.path != path:
            old = tl
            tl = Timeline(path)
            _singleton["tl"] = tl
    if old is not None:
        old.close(0)          # its queued rows still land, then it stops
    return tl


def add(**row) -> bool:
    """Queue one row on the process-wide timeline - not while the owner's
    "stop watching" pause runs (core.screen_memory.owner_paused: a click
    or look he asks for then is not kept either). Never raises."""
    try:
        if not _singleton["enabled"]:
            return False
        try:
            from core import screen_memory as _sm
            if _sm.owner_paused():
                return False
        except Exception:
            pass
        return get().add(**row)
    except Exception:
        return False


def prune_all(now=None) -> dict:
    """Apply the retention of the screen timeline AND the vision trace
    (start-up and hourly). Never raises."""
    out = {}
    try:
        out["timeline"] = get().prune(now=now)
    except Exception:
        out["timeline"] = {}
    try:
        from core import vision_trace as _vt
        out["trace"] = _vt.prune(now=now)
    except Exception:
        out["trace"] = {}
    return out


_pruner = {"thread": None}


def start_pruner(interval_s: float = 3600.0) -> bool:
    """Prune now (off this thread) and then every hour, on ONE never-exiting
    daemon thread. Idempotent. Never raises."""
    try:
        with _s_lock:
            t = _pruner["thread"]
            if t is not None and t.is_alive():
                return False

            def _loop():                   # never exits
                while True:
                    try:
                        res = prune_all()
                        n = sum(sum(v.values()) for v in res.values()
                                if isinstance(v, dict))
                        if n:
                            print(f"  [screen-memory] pruned {n} old "
                                  f"row(s) / trace entr(ies)", flush=True)
                    except Exception:
                        pass
                    time.sleep(max(60.0, float(interval_s)))
            t = threading.Thread(target=_loop, daemon=True,
                                 name="screen-memory-prune")
            _pruner["thread"] = t
            t.start()
        return True
    except Exception:
        return False
