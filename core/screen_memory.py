"""core/screen_memory.py - continuous screen memory, TEXT ONLY and cheap
(2026-10-05).

WHY THIS EXISTS
===============
00:34 the owner asked: "it would be cool if you could watch my screen
constantly ... that way he has a constant memory of what's going on - if it
doesn't take up too much resources on the computer"; "I'll tell him if I
don't want him to watch something".

The old AMBIENT_SCREEN_ENABLED loop (skills/ambient_listen) sent a
screenshot to the vision model every changed minute - on the 3090 the brain
and the clone voice share. This watcher makes NO model calls, stores NO
images and uses NO GPU. Every SCREEN_MEMORY_INTERVAL_S (5 s) it:

  1. pauses (logging only the transitions) while a game is fullscreen
     (game mode, or Windows reports a D3D full-screen / presentation app),
     the session is locked, the CPU is busy (> 60% over three ticks), the
     owner said "stop watching", or its own governor stepped in;
  2. reads the window inventory (EnumWindows: title, process, monitor -
     under 1 ms) and records opened / closed / retitled windows and focus
     changes, and the address of a browser window whose title changed;
  3. grabs each monitor at 160x90 (one GDI StretchBlt, only when there was
     input, the windows changed, or 30 s passed) and diffs a 16x9 grid of
     cells; a cell that keeps changing (a video) is "animated" and stops
     triggering;
  4. on a change only: reads the changed browser windows' text through UI
     Automation (<= 1 read per window per 10 s), and OCRs the changed region
     when UIA gave little (<= 1 OCR per 10 s);
  5. stores NEW lines only (plus a full read every 5 minutes of continuous
     viewing) as timestamped rows in core.screen_timeline.

Never: a vision / LLM call, a stored image, an Edit value (the address bar
excepted, its query string stripped), a password box, a private or excluded
window, or fact extraction into long-term memory.

Budget: target <= 1% of one core (measured by the watcher itself: its own
thread time + the UIA / OCR work it asked for); above 1% the interval
doubles (5 -> 10 -> 20 -> 30 s), above 3% it pauses for 5 minutes. Every 10
minutes it logs cpu / rss / threads / handles / rows.

Controls (core.dispatcher routes + the tray): "don't watch this" (the
foreground window, and its site, until it closes; its last 5 minutes of
rows deleted), "don't watch <app>" (persisted in data/screen_memory_state.
json), "stop watching (for N minutes)", "you can watch again", "are you
watching?", "forget the last hour". Never autostarts under
JARVIS_TEST_MODE. Never raises at the public API.
"""
from __future__ import annotations

import collections
import os
import threading
import time

__all__ = ["Watcher", "ChangeDetector", "get", "start", "stop", "pause",
           "unpause", "status_line", "exclude_foreground", "exclude_app",
           "forget", "is_running", "state_path", "load_state",
           "owner_paused", "set_state_publisher"]

THUMB_W, THUMB_H = 160, 90
GRID_W, GRID_H = 16, 9
CELL_MAD = 24.0
CHANGED_FRACTION = 0.02
MIN_CHANGED_CELLS = 3
ANIMATED_TICKS = 3
STABLE_TICKS = 2
REGRAB_S = 30.0
UIA_PER_WINDOW_S = 10.0
OCR_EVERY_S = 10.0
FULL_READ_S = 300.0
HEALTH_EVERY_S = 600.0
GOVERNOR_WINDOW_S = 60.0
GOVERNOR_PAUSE_S = 300.0
MAX_INTERVAL_S = 30.0
MAX_LINES = 60
MAX_TEXT = 2048
_BROWSERS = ("chrome.exe", "msedge.exe", "firefox.exe", "brave.exe",
             "opera.exe", "vivaldi.exe")
# SHQueryUserNotificationState
QUNS_NOT_PRESENT, QUNS_BUSY, QUNS_D3D_FULL = 1, 2, 3
QUNS_PRESENTATION = 4


def _cfg(name, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _g(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ── the change detector (pure numpy) ────────────────────────────────────
class ChangeDetector:
    """Per-monitor 160x90 grey thumbnails -> changed region, with animated
    cells (a playing video) masked out until they settle."""

    def __init__(self):
        self.prev: dict = {}           # monitor -> np.ndarray (90, 160)
        self.streak: dict = {}         # monitor -> np.ndarray (9, 16) ints
        self.calm: dict = {}           # monitor -> np.ndarray (9, 16) ints
        self.animated: dict = {}       # monitor -> np.ndarray (9, 16) bool

    @staticmethod
    def to_grey(bgra: bytes, w: int = THUMB_W, h: int = THUMB_H):
        import numpy as np
        a = np.frombuffer(bgra, dtype=np.uint8).reshape(h, w, 4)
        return (a[..., 2] * 0.299 + a[..., 1] * 0.587
                + a[..., 0] * 0.114).astype(np.float32)

    def update(self, monitor, grey, mon_rect):
        """(changed_cells, region) for one monitor: ``region`` is the native
        (x, y, w, h) bounding box of the changed, non-animated cells, or
        None when the monitor did not change."""
        import numpy as np
        prev = self.prev.get(monitor)
        self.prev[monitor] = grey
        if prev is None or prev.shape != grey.shape:
            self.streak[monitor] = np.zeros((GRID_H, GRID_W), np.int32)
            self.calm[monitor] = np.zeros((GRID_H, GRID_W), np.int32)
            self.animated[monitor] = np.zeros((GRID_H, GRID_W), bool)
            return 0, None
        ch, cw = grey.shape[0] // GRID_H, grey.shape[1] // GRID_W
        diff = np.abs(grey[:ch * GRID_H, :cw * GRID_W]
                      - prev[:ch * GRID_H, :cw * GRID_W])
        mad = diff.reshape(GRID_H, ch, GRID_W, cw).mean(axis=(1, 3))
        changed = mad > CELL_MAD
        streak = self.streak[monitor]
        calm = self.calm[monitor]
        anim = self.animated[monitor]
        streak[changed] += 1
        streak[~changed] = 0
        calm[~changed] += 1
        calm[changed] = 0
        anim |= streak > ANIMATED_TICKS
        anim &= ~(calm >= STABLE_TICKS)
        live = changed & ~anim
        n = int(live.sum())
        if n < max(MIN_CHANGED_CELLS, CHANGED_FRACTION * GRID_W * GRID_H):
            return n, None
        ys, xs = np.nonzero(live)
        mx, my, mw, mh = (float(v) for v in mon_rect)
        cwn, chn = mw / GRID_W, mh / GRID_H
        x0, x1 = xs.min() * cwn, (xs.max() + 1) * cwn
        y0, y1 = ys.min() * chn, (ys.max() + 1) * chn
        return n, (mx + x0, my + y0, x1 - x0, y1 - y0)


# ── persisted exclusions ────────────────────────────────────────────────
def state_path() -> str:
    from core.paths import data_file
    return data_file("screen_memory_state.json")


def load_state() -> dict:
    try:
        import json
        with open(state_path(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_state(d: dict) -> bool:
    try:
        from core.atomic_io import _atomic_write_json
        _atomic_write_json(state_path(), d)
        return True
    except Exception:
        return False


# ── the production environment ──────────────────────────────────────────
class Env:
    """Real windows, GDI thumbnails, UIA, OCR, CPU. Tests pass their own."""

    def windows(self):
        from core import screen_scope as _sc
        return _sc.visible_windows()

    def foreground(self):
        try:
            import ctypes
            return int(ctypes.WinDLL("user32").GetForegroundWindow() or 0)
        except Exception:
            return 0

    def monitors(self):
        try:
            from core.config import MONITORS
            return dict(MONITORS)
        except Exception:
            return {}

    def read_url(self, hwnd):
        from core import screen_text as _st
        return _st.read_url(hwnd, timeout_s=0.3)

    def snapshot(self, win):
        from core import screen_text as _st
        return _st.snapshot(win.hwnd, win_rect=win.rect, title=win.title,
                            process=win.process, pid=win.pid,
                            monitor=win.monitor, budget_ms=300,
                            want_hrefs=False, retry_chrome=False)

    def last_input_ms(self):
        try:
            import ctypes
            from ctypes import wintypes

            class LII(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]
            lii = LII()
            lii.cbSize = ctypes.sizeof(LII)
            if ctypes.WinDLL("user32").GetLastInputInfo(ctypes.byref(lii)):
                return int(lii.dwTime)
        except Exception:
            pass
        return None

    def notification_state(self):
        try:
            import ctypes
            st = ctypes.c_int(0)
            if ctypes.WinDLL("shell32").SHQueryUserNotificationState(
                    ctypes.byref(st)) == 0:
                return int(st.value)
        except Exception:
            pass
        return None

    def cpu_percent(self):
        try:
            import psutil
            return float(psutil.cpu_percent(None))
        except Exception:
            return 0.0

    def game_mode(self):
        try:
            import sys
            bc = sys.modules.get("bobert_companion")
            fn = getattr(bc, "_game_mode_active", None)
            return bool(fn()) if callable(fn) else False
        except Exception:
            return False

    def grab_thumb(self, rect):
        """BGRA bytes of ``rect`` scaled to 160x90 (one GDI StretchBlt into
        a memory DIB). Never stored. None on failure."""
        try:
            from core.screen_privacy import reads_blocked
            if reads_blocked():
                return None
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.WinDLL("user32")
            gdi32 = ctypes.WinDLL("gdi32")
            gdi32.CreateCompatibleDC.restype = wintypes.HDC
            gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
            user32.GetDC.restype = wintypes.HDC
            gdi32.CreateDIBSection.restype = wintypes.HBITMAP
            gdi32.SelectObject.restype = wintypes.HGDIOBJ
            gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
            gdi32.StretchBlt.argtypes = [wintypes.HDC] + [ctypes.c_int] * 4 + [
                wintypes.HDC] + [ctypes.c_int] * 4 + [wintypes.DWORD]
            gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
            gdi32.DeleteDC.argtypes = [wintypes.HDC]
            user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]

            class BIH(ctypes.Structure):
                _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                            ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                            ("biBitCount", wintypes.WORD),
                            ("biCompression", wintypes.DWORD),
                            ("biSizeImage", wintypes.DWORD),
                            ("biXPelsPerMeter", wintypes.LONG),
                            ("biYPelsPerMeter", wintypes.LONG),
                            ("biClrUsed", wintypes.DWORD),
                            ("biClrImportant", wintypes.DWORD)]
            bih = BIH(ctypes.sizeof(BIH), THUMB_W, -THUMB_H, 1, 32, 0, 0, 0,
                      0, 0, 0)
            bits = ctypes.c_void_p()
            screen = user32.GetDC(None)
            mem = gdi32.CreateCompatibleDC(screen)
            dib = gdi32.CreateDIBSection(mem, ctypes.byref(bih), 0,
                                         ctypes.byref(bits), None, 0)
            old = gdi32.SelectObject(mem, dib)
            try:
                gdi32.SetStretchBltMode(mem, 3)           # COLORONCOLOR
                x, y, w, h = (int(v) for v in rect)
                ok = gdi32.StretchBlt(mem, 0, 0, THUMB_W, THUMB_H, screen, x,
                                      y, w, h, 0x00CC0020)
                if not ok or not bits.value:
                    return None
                return ctypes.string_at(bits.value, THUMB_W * THUMB_H * 4)
            finally:
                gdi32.SelectObject(mem, old)
                gdi32.DeleteObject(dib)
                gdi32.DeleteDC(mem)
                user32.ReleaseDC(None, screen)
        except Exception:
            return None

    def capture(self, rect, windows, urls=None):
        """Privacy-gated native pixels of ``rect`` (for OCR), or None. A
        browser window is judged with its address (``urls``: hwnd -> the
        address read this tick); one whose address is unknown is masked
        (fail closed)."""
        try:
            from core import screen_privacy as _priv
            if _priv.reads_blocked():
                return None
            urls = urls or {}
            infos = [{"hwnd": w.hwnd, "rect": w.rect, "title": w.title,
                      "process": w.process,
                      "private": _priv.live_private(
                          w, url=urls.get(w.hwnd) or None, fail_closed=True)}
                     for w in windows]
            gate = _priv.region_gate(rect, infos, None)
            if not gate.allowed:
                return None
            import mss
            from PIL import Image
            x, y, w, h = (int(v) for v in rect)
            cls = getattr(mss, "MSS", mss.mss)
            with cls() as sct:
                raw = sct.grab({"left": x, "top": y, "width": w, "height": h})
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            return _priv.apply_masks(img, gate.masks, origin=(x, y))
        except Exception:
            return None

    def ocr(self, img):
        from core import screen_ocr as _ocr
        return _ocr.ocr_image(img)

    def ocr_cpu_s(self):
        try:
            from core import screen_ocr as _ocr
            pid = _ocr.status().get("pid")
            if not pid:
                return 0.0
            import psutil
            t = psutil.Process(pid).cpu_times()
            return float(t.user + t.system)
        except Exception:
            return 0.0

    def health(self):
        try:
            import psutil
            p = psutil.Process(os.getpid())
            return {"rss_mb": round(p.memory_info().rss / 1048576, 1),
                    "threads": p.num_threads(),
                    "handles": getattr(p, "num_handles", lambda: None)()}
        except Exception:
            return {}

    def set_idle_priority(self):
        try:
            import ctypes
            k = ctypes.WinDLL("kernel32")
            k.SetThreadPriority(k.GetCurrentThread(), -15)  # THREAD_PRIORITY_IDLE
        except Exception:
            pass


# ── the watcher ─────────────────────────────────────────────────────────
class Watcher:
    def __init__(self, env=None, timeline=None, clock=None):
        self.env = env or Env()
        self._timeline = timeline
        self.clock = clock or time.time
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        self.wake = threading.Event()
        self.det = ChangeDetector()
        self.owner_pause_until = None      # float ts, float("inf") or None
        self.gov_pause_until = 0.0
        self.interval = None
        self.paused_reason = ""
        self.last_inventory: dict = {}
        self.last_fg = None
        self.fg_since = 0.0
        self.last_grab: dict = {}
        self.last_input = None
        self.last_uia: dict = {}
        self.last_ocr = 0.0
        self.last_full: dict = {}
        self.seen_lines: dict = {}
        self.cpu_hist: "collections.deque" = collections.deque(maxlen=3)
        self.cost: "collections.deque" = collections.deque()
        self.rows_today = 0
        self.rows_day = time.strftime("%Y%m%d")
        self.last_health = 0.0
        self.ticks = 0
        self.excluded_hwnds: dict = {}

    # -- plumbing --------------------------------------------------------
    @property
    def timeline(self):
        if self._timeline is None:
            from core import screen_timeline as _tl
            return _tl.get()
        return self._timeline

    def base_interval(self) -> float:
        try:
            v = float(_cfg("SCREEN_MEMORY_INTERVAL_S", 5.0))
            return v if v >= 1.0 else 5.0
        except Exception:
            return 5.0

    def _add(self, **row) -> None:
        try:
            if self._timeline is not None:
                ok = self._timeline.add_now(**row)
            else:
                ok = self.timeline.add(**row)
            if ok:
                day = time.strftime("%Y%m%d")
                if day != self.rows_day:
                    self.rows_day, self.rows_today = day, 0
                self.rows_today += 1
        except Exception:
            pass

    # -- lifecycle -------------------------------------------------------
    def start(self) -> bool:
        with self.lock:
            from core.screen_privacy import reads_blocked
            if reads_blocked() and isinstance(self.env, Env):
                return False
            self.running = True
            self.interval = self.base_interval()
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, daemon=True,
                                               name="screen-memory")
                self.thread.start()
            self.wake.set()
        print("  [screen-memory] on - text only, no AI calls", flush=True)
        _publish()
        return True

    def stop(self) -> None:
        with self.lock:
            self.running = False
        print("  [screen-memory] off", flush=True)
        _publish()

    def owner_paused(self) -> bool:
        """True while the owner's "stop watching" pause runs."""
        try:
            until = self.owner_pause_until
            return until is not None and self.clock() < until
        except Exception:
            return False

    def _loop(self) -> None:               # never exits
        try:
            self.env.set_idle_priority()
        except Exception:
            pass
        while True:
            try:
                self.wake.wait(self.interval or self.base_interval())
                self.wake.clear()
                if not self.running:
                    continue
                self.tick()
            except Exception as e:
                try:
                    print(f"  [screen-memory] tick failed: {type(e).__name__}: "
                          f"{e}", flush=True)
                except Exception:
                    pass
                time.sleep(1.0)

    # -- pause rules -----------------------------------------------------
    def pause_reason(self, fg_is_browser: bool) -> str:
        now = self.clock()
        if self.owner_pause_until is not None:
            if now < self.owner_pause_until:
                return "you asked me to stop watching"
            self.owner_pause_until = None
        if now < self.gov_pause_until:
            return "my own CPU use was over budget"
        if self.env.game_mode():
            return "a game is running"
        st = self.env.notification_state()
        if st == QUNS_NOT_PRESENT:
            return "the session is locked"
        if st in (QUNS_D3D_FULL, QUNS_PRESENTATION):
            return "a full-screen app is running"
        if st == QUNS_BUSY and not fg_is_browser:
            return "a full-screen app is running"
        self.cpu_hist.append(self.env.cpu_percent())
        limit = float(_cfg("SCREEN_MEMORY_CPU_PAUSE_PCT", 60) or 60)
        if len(self.cpu_hist) == self.cpu_hist.maxlen and (
                sum(self.cpu_hist) / len(self.cpu_hist)) > limit:
            return "the CPU is busy"
        return ""

    # -- the tick --------------------------------------------------------
    def tick(self) -> dict:
        """One pass. Returns a small report (tests). Never raises."""
        t_cpu0 = time.thread_time()
        ocr0 = self.env.ocr_cpu_s()
        report = {"paused": "", "rows": 0, "grabs": 0, "uia": 0, "ocr": 0}
        uia_wall = 0.0
        try:
            self.ticks += 1
            wins = list(self.env.windows())
            fg = self.env.foreground()
            fgw = next((w for w in wins if _g(w, "hwnd") == fg), None)
            fg_browser = bool(fgw and str(_g(fgw, "process", "")).lower()
                              in _BROWSERS)
            why = self.pause_reason(fg_browser)
            if why != self.paused_reason:
                print(f"  [screen-memory] {'paused: ' + why if why else 'resumed'}",
                      flush=True)
                self.paused_reason = why
            if why:
                report["paused"] = why
                return report
            from core import screen_privacy as _priv
            urls = self._addresses(wins)
            # A browser page is private by its ADDRESS too (a bank whose tab
            # title says only "Accounts Overview"); a browser whose address
            # could not be read is skipped this tick - fail closed (review
            # 2026-10-05).
            visible = [w for w in wins if not _priv.live_private(
                w, url=urls.get(_g(w, "hwnd")) or None, fail_closed=True)]
            rows0 = self.rows_today
            self._inventory(visible, fg, urls)
            # Fullscreen browser / video: titles and now-playing only.
            st = self.env.notification_state()
            if st == QUNS_BUSY and fg_browser:
                report["rows"] = self.rows_today - rows0
                return report
            changed = self._detect(visible, report)
            for mon, region in changed.items():
                t0 = time.monotonic()
                self._extract(mon, region, visible, urls, report, wins)
                uia_wall += time.monotonic() - t0
            self._full_read(fgw, visible, urls, report)
            report["rows"] = self.rows_today - rows0
            return report
        finally:
            cost = (time.thread_time() - t_cpu0) + max(
                0.0, self.env.ocr_cpu_s() - ocr0)
            self._govern(cost, uia_wall)
            self._health()

    def _addresses(self, wins) -> dict:
        """{hwnd: address} of every browser window: read again when its
        title changed (or it is new), else the address read then. "" when
        it could not be read. Never raises."""
        out = {}
        cache = getattr(self, "url_cache", None)
        if cache is None:
            cache = self.url_cache = {}
        live = set()
        for w in wins:
            h = _g(w, "hwnd")
            if str(_g(w, "process", "")).lower() not in _BROWSERS:
                continue
            live.add(h)
            title = _g(w, "title", "")
            old = cache.get(h)
            if old is not None and old[0] == title and old[1]:
                out[h] = old[1]
                continue
            try:
                url = self.env.read_url(h) or ""
            except Exception:
                url = ""
            cache[h] = (title, url)
            out[h] = url
        for h in [h for h in cache if h not in live]:
            cache.pop(h, None)
        return out

    def _inventory(self, wins, fg, urls) -> None:
        now = self.clock()
        cur = {}
        for w in wins:
            cur[_g(w, "hwnd")] = (_g(w, "title", ""), _g(w, "monitor"),
                                  _g(w, "process", ""))
        for h, (title, mon, proc) in cur.items():
            old = self.last_inventory.get(h)
            if old is None or old[0] != title:
                url = urls.get(h, "") if str(proc).lower() in _BROWSERS else ""
                self._add(ts=now, monitor=mon, hwnd=h, process=proc,
                          title=title, url=url,
                          source="title", text=("opened" if old is None
                                                else "now showing"))
        for h, (title, mon, proc) in self.last_inventory.items():
            if h not in cur:
                self.seen_lines.pop(h, None)
                self.last_uia.pop(h, None)
        if fg != self.last_fg:
            info = cur.get(fg)
            if info is not None:
                self._add(ts=now, monitor=info[1], hwnd=fg, process=info[2],
                          title=info[0], source="focus", text="in front")
            self.last_fg = fg
            self.fg_since = now
        self.last_inventory = cur

    @staticmethod
    def _private_now(w, url) -> bool:
        """The window, judged again with the address its page read just
        returned (fresher than the tick's): private, or a browser page
        whose address is still unknown. Never raises (True on failure)."""
        try:
            from core import screen_privacy as _priv
            return bool(_priv.live_private(w, url=url or None,
                                           fail_closed=True))
        except Exception:
            return True

    def _detect(self, wins, report) -> dict:
        """{monitor: changed native region} (input / window gated)."""
        now = self.clock()
        inp = self.env.last_input_ms()
        had_input = inp is not None and inp != self.last_input
        self.last_input = inp
        out = {}
        mons = self.env.monitors()
        win_mons = {_g(w, "monitor") for w in wins}
        for mon, rect in mons.items():
            due = (had_input or now - self.last_grab.get(mon, 0.0) >= REGRAB_S
                   or self.ticks <= 1)
            if not due or mon not in win_mons:
                continue
            raw = self.env.grab_thumb(rect)
            self.last_grab[mon] = now
            if raw is None:
                continue
            report["grabs"] += 1
            try:
                grey = ChangeDetector.to_grey(raw)
            except Exception:
                continue
            _n, region = self.det.update(mon, grey, rect)
            if region is not None:
                out[mon] = region
        return out

    def _lines_from_snapshot(self, snap) -> list:
        els = [e for e in snap.elements
               if (e.in_document or snap.doc_rect is None)
               and not e.is_password and e.ctype in (
                   "Hyperlink", "Text", "Button", "TabItem", "ListItem",
                   "MenuItem", "Image", "DataItem", "TreeItem")]
        els.sort(key=lambda e: (round(e.rect[1] / 20), e.rect[0]))
        out, total = [], 0
        for e in els:
            t = " ".join(e.name.split())
            if len(t) < 3 or t in out:
                continue
            out.append(t)
            total += len(t) + 1
            if len(out) >= MAX_LINES or total >= MAX_TEXT:
                break
        return out

    def _new_lines(self, hwnd, lines) -> list:
        seen = self.seen_lines.setdefault(hwnd, collections.OrderedDict())
        new = [ln for ln in lines if ln not in seen]
        for ln in lines:
            seen[ln] = True
            seen.move_to_end(ln)
        while len(seen) > 600:
            seen.popitem(last=False)
        return new

    def _intersects(self, a, b) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah

    def _extract(self, mon, region, wins, urls, report,
                 all_wins=None) -> None:
        """Read the windows of ``wins`` (the non-private ones) that the
        changed ``region`` touches. The OCR capture's privacy gate sees
        ``all_wins`` - EVERY window, so a private one above the region is
        masked (it was handed the filtered list, which could not mask
        what it had never been shown)."""
        now = self.clock()
        mode = str(_cfg("SCREEN_UIA_NONBROWSER", "on_demand")).lower()
        for w in wins:
            if _g(w, "monitor") != mon or not self._intersects(
                    _g(w, "rect"), region):
                continue
            hwnd = _g(w, "hwnd")
            browser = str(_g(w, "process", "")).lower() in _BROWSERS
            text_chars = 0
            if browser or mode == "always":
                if now - self.last_uia.get(hwnd, 0.0) >= UIA_PER_WINDOW_S:
                    self.last_uia[hwnd] = now
                    snap = self.env.snapshot(w)
                    report["uia"] += 1
                    if snap is not None and not snap.has_password:
                        url = snap.url or urls.get(hwnd) or ""
                        if self._private_now(w, url):
                            continue
                        lines = self._lines_from_snapshot(snap)
                        text_chars = sum(len(x) for x in lines)
                        new = self._new_lines(hwnd, lines)
                        if new:
                            self._add(ts=now, monitor=mon, hwnd=hwnd,
                                      process=_g(w, "process", ""),
                                      title=_g(w, "title", ""), url=url,
                                      source="uia", text="\n".join(new))
                    elif snap is not None and snap.has_password:
                        continue
                else:
                    continue
            if text_chars >= 200:
                continue
            if now - self.last_ocr < OCR_EVERY_S:
                continue
            x0 = max(region[0], _g(w, "rect")[0])
            y0 = max(region[1], _g(w, "rect")[1])
            x1 = min(region[0] + region[2], _g(w, "rect")[0] + _g(w, "rect")[2])
            y1 = min(region[1] + region[3], _g(w, "rect")[1] + _g(w, "rect")[3])
            if x1 - x0 < 32 or y1 - y0 < 16:
                continue
            if self._private_now(w, urls.get(hwnd, "")):
                continue
            self.last_ocr = now
            img = self.env.capture((x0, y0, x1 - x0, y1 - y0),
                                   all_wins if all_wins is not None else wins,
                                   urls)
            if img is None:
                continue
            lines = self.env.ocr(img)
            report["ocr"] += 1
            if not lines:
                continue
            texts = [" ".join(str(ln.get("t") or "").split()) for ln in lines]
            new = self._new_lines(hwnd, [t for t in texts if len(t) >= 3])
            if new:
                self._add(ts=now, monitor=mon, hwnd=hwnd,
                          process=_g(w, "process", ""),
                          title=_g(w, "title", ""), url=urls.get(hwnd, ""),
                          source="ocr", text="\n".join(new[:MAX_LINES]))

    def _full_read(self, fgw, wins, urls, report) -> None:
        if fgw is None:
            return
        now = self.clock()
        hwnd = _g(fgw, "hwnd")
        if now - self.fg_since < FULL_READ_S:
            return
        if now - self.last_full.get(hwnd, 0.0) < FULL_READ_S:
            return
        if str(_g(fgw, "process", "")).lower() not in _BROWSERS:
            return
        self.last_full[hwnd] = now
        snap = self.env.snapshot(fgw)
        report["uia"] += 1
        if snap is None or snap.has_password:
            return
        if self._private_now(fgw, snap.url or urls.get(hwnd, "")):
            return
        lines = self._lines_from_snapshot(snap)
        if lines:
            self._add(ts=now, monitor=_g(fgw, "monitor"), hwnd=hwnd,
                      process=_g(fgw, "process", ""),
                      title=_g(fgw, "title", ""),
                      url=urls.get(hwnd) or snap.url, source="uia",
                      text="\n".join(lines))

    # -- governor + health -----------------------------------------------
    def _govern(self, cpu_s, uia_wall_s) -> None:
        now = self.clock()
        self.cost.append((now, float(cpu_s) + float(uia_wall_s)))
        while self.cost and now - self.cost[0][0] > GOVERNOR_WINDOW_S:
            self.cost.popleft()
        span = max(self.base_interval(), (now - self.cost[0][0])
                   if len(self.cost) > 1 else self.base_interval())
        pct = 100.0 * sum(c for _t, c in self.cost) / span
        cap = float(_cfg("SCREEN_MEMORY_MAX_CORE_PCT", 1.0) or 1.0)
        if pct > 3 * cap and len(self.cost) >= 3:
            self.gov_pause_until = now + GOVERNOR_PAUSE_S
            self.cost.clear()
            print(f"  [screen-memory] used {pct:.1f}% of a core - pausing 5 "
                  "minutes", flush=True)
        elif pct > cap and len(self.cost) >= 3:
            old = self.interval or self.base_interval()
            new = min(MAX_INTERVAL_S, old * 2)
            if new != self.interval:
                print(f"  [screen-memory] {pct:.2f}% of a core - interval "
                      f"{old:.0f} -> {new:.0f} s", flush=True)
                self.interval = new
        self.last_pct = pct

    def _health(self) -> None:
        now = self.clock()
        if now - self.last_health < HEALTH_EVERY_S:
            return
        self.last_health = now
        h = self.env.health()
        print(f"  [screen-memory] cpu={getattr(self, 'last_pct', 0.0):.2f}% "
              f"rss={h.get('rss_mb')}MB threads={h.get('threads')} "
              f"handles={h.get('handles')} rows={self.rows_today} "
              f"interval={self.interval or self.base_interval():.0f} "
              f"paused={self.paused_reason or 'no'}", flush=True)

    # -- controls --------------------------------------------------------
    def pause(self, minutes=None) -> str:
        with self.lock:
            now = self.clock()
            self.owner_pause_until = (now + float(minutes) * 60.0
                                      if minutes else float("inf"))
        if minutes:
            m = int(round(float(minutes)))
            return (f"Very good, sir — I won't watch your screen for "
                    f"{m} minute{'s' if m != 1 else ''}.")
        return ("Very good, sir — I've stopped watching your screen "
                "until you tell me I can watch again.")

    def unpause(self) -> str:
        with self.lock:
            self.owner_pause_until = None
            self.gov_pause_until = 0.0
        if not self.running:
            return ("Screen memory is switched off in Settings, sir, so "
                    "I'm not watching either way.")
        self.wake.set()
        return "Watching again, sir — text only, as before."

    def status(self) -> str:
        if not self.running:
            on = "off"
        elif self.owner_pause_until is not None and \
                self.clock() < self.owner_pause_until:
            on = "paused"
        else:
            on = "on"
        apps = sorted(load_state().get("apps") or [])
        n = 0
        try:
            n = self.timeline.count(since=time.time() - 86400)
        except Exception:
            n = self.rows_today
        if on == "off":
            head = "I'm not watching your screen, sir — screen memory is off."
        elif on == "paused":
            until = self.owner_pause_until
            head = ("I've paused watching, sir"
                    + ("" if until == float("inf") else
                       f", until {time.strftime('%H:%M', time.localtime(until))}")
                    + ".")
        else:
            head = ("Yes, sir — I'm keeping a text-only record of your "
                    "screen" + (f" ({self.paused_reason} right now, so I'm "
                                "holding off)" if self.paused_reason else "")
                    + ".")
        tail = f" {n} entries in the last day."
        if apps:
            tail += f" I don't watch: {', '.join(apps)}."
        return head + tail

    def exclude_foreground(self) -> str:
        from core import screen_privacy as _priv
        fg = self.env.foreground()
        win = next((w for w in self.env.windows() if _g(w, "hwnd") == fg),
                   None)
        if win is None:
            return "I can't tell which window you mean, sir."
        url = ""
        if str(_g(win, "process", "")).lower() in _BROWSERS:
            url = self.env.read_url(fg) or ""
        _priv.exclude_window(fg, url)
        n = 0
        try:
            n = self.timeline.forget(since=self.clock() - 300, hwnd=fg)
        except Exception:
            n = 0
        self.seen_lines.pop(fg, None)
        title = str(_g(win, "title", "")).split(" - ")[0][:60]
        return (f"Understood, sir — I won't watch '{title}' while it's "
                f"open" + (f", and I've forgotten {n} note"
                           f"{'s' if n != 1 else ''} from the last five "
                           "minutes" if n else "") + ".")

    def exclude_app(self, name) -> str:
        from core import screen_privacy as _priv
        app = " ".join(str(name or "").split()).lower()
        if not app:
            return "Which app shouldn't I watch, sir?"
        st = load_state()
        apps = sorted(set(st.get("apps") or []) | {app})
        st["apps"] = apps
        _save_state(st)
        _priv.exclude_app(app)
        return f"Understood, sir — I won't watch {name} from now on."


_singleton = {"w": None}
_s_lock = threading.Lock()


def get() -> Watcher:
    with _s_lock:
        if _singleton["w"] is None:
            _singleton["w"] = Watcher()
            try:
                from core import screen_privacy as _priv
                _priv.set_exclusions_provider(load_state)
            except Exception:
                pass
        return _singleton["w"]


def is_running() -> bool:
    w = _singleton["w"]
    return bool(w and w.running)


def owner_paused() -> bool:
    """True while the owner's "stop watching (for N minutes)" runs - with
    screen memory on OR off. Every screen RECORD honours it, not only the
    watcher (review 2026-10-05): core.screen_timeline.add drops rows and
    core.vision_trace keeps a bare "skipped: paused" entry. A click or look
    he asks for still reads the screen; it just is not kept. Never
    raises."""
    try:
        w = _singleton["w"]
        return bool(w is not None and w.owner_paused())
    except Exception:
        return False


# The tray's "Screen Memory" checkmark (hud_state.screen_memory): published
# on EVERY start / stop - boot autostart, voice, settings, tray - by the
# watcher itself (review 2026-10-05: only the tray's own toggle wrote it, so
# a boot autostart showed unchecked while recording).
_publisher = [None]


def set_state_publisher(fn) -> None:
    """``fn(on: bool)`` - called now and on every start / stop. Never
    raises."""
    _publisher[0] = fn if callable(fn) else None
    _publish()


def _publish() -> None:
    try:
        fn = _publisher[0]
        if fn is not None:
            fn(is_running())
    except Exception:
        pass


def start() -> bool:
    return get().start()


def stop() -> None:
    w = _singleton["w"]
    if w is not None:
        w.stop()


def pause(minutes=None) -> str:
    return get().pause(minutes)


def unpause() -> str:
    return get().unpause()


def status_line() -> str:
    return get().status()


def exclude_foreground() -> str:
    return get().exclude_foreground()


def exclude_app(name) -> str:
    return get().exclude_app(name)


def forget(span: dict, now=None) -> str:
    """Purge timeline rows, trace entries, the scene ring and the screen
    cache for a span ({"seconds": N} / {"today": True} / {"all": True}).
    Returns the spoken line with the counts. Never raises."""
    try:
        t = float(now or time.time())
        if span.get("all"):
            since, what = None, "everything I'd seen"
        elif span.get("today"):
            lt = time.localtime(t)
            since = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0,
                                 0, 0, -1))
            what = "everything from today"
        else:
            secs = float(span.get("seconds") or 3600)
            since = t - secs
            mins = int(round(secs / 60))
            what = ("the last hour" if mins == 60 else
                    f"the last {mins // 60} hours" if mins % 60 == 0 and mins > 60
                    else f"the last {mins} minute{'s' if mins != 1 else ''}")
        from core import screen_timeline as _tl
        rows = _tl.get().forget(since=since, until=t)
        traces = 0
        try:
            from core import vision_trace as _vt
            traces = _vt.purge(since, t)
        except Exception:
            traces = 0
        scenes = 0
        try:
            from core import grounded_click as _gc
            scenes = _gc.forget_scenes(since, t)
        except Exception:
            scenes = 0
        cached = 0
        try:
            import sys
            bc = sys.modules.get("bobert_companion")
            lock = getattr(bc, "_screen_cache_lock", None)
            cache = getattr(bc, "_screen_cache", None)
            if lock is not None and isinstance(cache, list):
                with lock:
                    keep = [e for e in cache
                            if since is not None and e.get("ts", 0) < since]
                    cached = len(cache) - len(keep)
                    cache[:] = keep
        except Exception:
            cached = 0
        w = _singleton["w"]
        if w is not None:
            w.seen_lines.clear()
        print(f"  [screen-memory] forgot {what}: {rows} rows, {traces} trace "
              f"entries, {scenes} scenes, {cached} cached looks", flush=True)
        return (f"Done, sir — I've forgotten {what} on your screen: "
                f"{rows} note{'s' if rows != 1 else ''}"
                + (f", {traces} vision record{'s' if traces != 1 else ''}"
                   if traces else "")
                + (f" and {cached} saved look{'s' if cached != 1 else ''}"
                   if cached else "") + ".")
    except Exception as e:
        return f"I couldn't forget that, sir ({type(e).__name__})."
