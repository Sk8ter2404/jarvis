"""core/screen_ocr.py - read the words in an image with Windows' own OCR
(2026-10-05).

WHY THIS EXISTS
===============
UI Automation names the links of a web page, but not everything: a heavy
page (2,400 elements took 430-580 ms), an app that exposes no tree, a game
launcher, a video's burnt-in title. Windows ships an OCR engine
(Windows.Media.Ocr): on the research's synthetic 2560x1440 pages it read
every title the resolver needed (OCR dev set 33/33) in ~15 ms per window
crop of engine time.

The in-process WinRT projection (winrt-Windows.Media.Ocr) is NOT installed
on this box (owner decision D1: ~330 KB, three wheels). Until it is, a small
PowerShell worker (tools/ocr_worker.ps1) does it with no install: started
lazily, hidden, at idle priority, it takes raw gray pixels on stdin and
returns word boxes on stdout. It exits after 120 s idle (84-174 MB while
alive). A request waits at most 3 s; a timeout kills and restarts the
worker with backoff, and after 3 restarts in 10 minutes OCR is off for the
session. Text under 12 px median height is read again at 1.5x.

Only pixels the privacy gate already allowed reach this module, and nothing
it reads is written anywhere by it. Under JARVIS_TEST_MODE no worker is
spawned unless a test injected a backend. Never raises.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import statistics
import subprocess
import threading
import time

__all__ = ["ocr_image", "backend_name", "set_backend", "lines_to_cands",
           "status", "shutdown"]

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKER = os.path.join(_PROJECT, "tools", "ocr_worker.ps1")
_REQUEST_TIMEOUT_S = 3.0
_IDLE_EXIT_S = 120.0
_MAX_RESTARTS = 3
_RESTART_WINDOW_S = 600.0
_SMALL_TEXT_PX = 12
_IDLE_PRIORITY = 0x00000040
_NO_WINDOW = 0x08000000

_lock = threading.Lock()
_backend = [None]                   # tests: fn(PIL image) -> lines
_state = {"proc": None, "out": None, "last_used": 0.0, "restarts": [],
          "off": False, "off_logged": False, "reaper": None, "requests": 0,
          "engine_ms": 0.0}


def _cfg(name, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _winrt_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("winrt.windows.media.ocr") is not None
    except Exception:
        return False


def backend_name() -> str:
    """"fake" (a test), "off", "winrt" or "powershell"."""
    if _backend[0] is not None:
        return "fake"
    mode = str(_cfg("SCREEN_OCR_BACKEND", "auto") or "auto").strip().lower()
    if mode == "off" or _state["off"]:
        return "off"
    if mode in ("auto", "winrt") and _winrt_available():
        return "winrt"
    if mode == "winrt":
        return "off"
    return "powershell"


def set_backend(fn) -> None:
    """Tests: ``fn(PIL image) -> [{"t": text, "words": [[w, x, y, w, h]]}]``
    replaces the engine; None restores it."""
    _backend[0] = fn


def _test_mode() -> bool:
    from core.screen_privacy import reads_blocked
    return reads_blocked()


# ── the PowerShell worker ───────────────────────────────────────────────
def _reader(proc, out_q):
    try:
        for line in proc.stdout:
            out_q.put(line)
    except Exception:
        pass
    finally:
        out_q.put(None)


def _start_worker():
    if _test_mode():
        raise RuntimeError("OCR worker refused in a test process")
    if not os.path.exists(_WORKER):
        raise FileNotFoundError(_WORKER)
    flags = (_IDLE_PRIORITY | _NO_WINDOW) if os.name == "nt" else 0
    proc = subprocess.Popen(
        ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive",
         "-ExecutionPolicy", "Bypass", "-File", _WORKER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1, encoding="utf-8",
        errors="replace", creationflags=flags)
    out_q: "queue.Queue" = queue.Queue()
    threading.Thread(target=_reader, args=(proc, out_q), daemon=True,
                     name="ocr-worker-out").start()
    try:
        first = out_q.get(timeout=15.0)
    except queue.Empty:
        _kill(proc)
        raise TimeoutError("OCR worker did not start")
    if not first or '"ready":true' not in first.replace(" ", ""):
        _kill(proc)
        raise RuntimeError(f"OCR worker failed: {str(first)[:120]}")
    return proc, out_q


def _kill(proc) -> None:
    try:
        proc.kill()
    except Exception:
        pass


def _ensure_reaper() -> None:
    t = _state["reaper"]
    if t is not None and t.is_alive():
        return

    def _loop():                       # never exits
        while True:
            time.sleep(10.0)
            try:
                with _lock:
                    p = _state["proc"]
                    idle = time.time() - _state["last_used"]
                    if p is not None and idle > _IDLE_EXIT_S:
                        _state["proc"] = None
                        _state["out"] = None
                    else:
                        p = None
                if p is not None:
                    try:
                        p.stdin.write("quit\n")
                        p.stdin.flush()
                        p.wait(3)
                    except Exception:
                        _kill(p)
            except Exception:
                pass
    t = threading.Thread(target=_loop, daemon=True, name="ocr-reaper")
    _state["reaper"] = t
    t.start()


def _note_restart() -> None:
    now = time.time()
    rs = [t for t in _state["restarts"] if now - t < _RESTART_WINDOW_S]
    rs.append(now)
    _state["restarts"] = rs
    if len(rs) > _MAX_RESTARTS:
        _state["off"] = True
        if not _state["off_logged"]:
            _state["off_logged"] = True
            print("  [ocr] off for this session: the worker hung "
                  f"{len(rs)} times in 10 minutes", flush=True)


def _powershell_ocr(arr_bytes, w, h):
    with _lock:
        if _state["off"]:
            return None
        if _state["proc"] is None or _state["proc"].poll() is not None:
            try:
                _state["proc"], _state["out"] = _start_worker()
            except Exception as e:
                print(f"  [ocr] worker unavailable: {type(e).__name__}: {e}",
                      flush=True)
                _note_restart()
                return None
            _ensure_reaper()
        proc, out_q = _state["proc"], _state["out"]
        _state["last_used"] = time.time()
        payload = json.dumps({"w": int(w), "h": int(h), "fmt": "gray8",
                              "b64": base64.b64encode(arr_bytes).decode()})
        try:
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
            line = out_q.get(timeout=_REQUEST_TIMEOUT_S)
        except (queue.Empty, OSError, ValueError):
            line = None
        if not line:
            print("  [ocr] request timed out - restarting the worker",
                  flush=True)
            _kill(proc)
            _state["proc"] = None
            _state["out"] = None
            _note_restart()
            return None
        _state["last_used"] = time.time()
    try:
        out = json.loads(line)
    except Exception:
        return None
    if "error" in out:
        return None
    _state["requests"] += 1
    _state["engine_ms"] += float(out.get("ms") or 0.0)
    return out.get("lines") or []


def _winrt_ocr(img):                  # pragma: no cover - needs the D1 wheels
    import asyncio
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.graphics.imaging import (SoftwareBitmap,
                                                BitmapPixelFormat)
    from winrt.windows.storage.streams import DataWriter
    g = img.convert("L")
    w, h = g.size
    dw = DataWriter()
    dw.write_bytes(list(g.tobytes()))
    bmp = SoftwareBitmap.create_copy_from_buffer(
        dw.detach_buffer(), BitmapPixelFormat.GRAY8, w, h)
    engine = OcrEngine.try_create_from_user_profile_languages()

    async def _go():
        return await engine.recognize_async(bmp)
    res = asyncio.run(_go())
    lines = []
    for ln in res.lines:
        words = [[wd.text, int(wd.bounding_rect.x), int(wd.bounding_rect.y),
                  int(wd.bounding_rect.width), int(wd.bounding_rect.height)]
                 for wd in ln.words]
        lines.append({"t": ln.text, "words": words})
    return lines


def _run_engine(img):
    fake = _backend[0]
    if fake is not None:
        return fake(img)
    name = backend_name()
    if name == "off":
        return None
    if name == "winrt":
        try:
            return _winrt_ocr(img)
        except Exception:
            pass
    g = img.convert("L")
    w, h = g.size
    return _powershell_ocr(g.tobytes(), w, h)


def _median_height(lines) -> float:
    hs = [wd[4] for ln in lines or () for wd in ln.get("words") or ()
          if len(wd) >= 5]
    return statistics.median(hs) if hs else 0.0


def ocr_image(img):
    """OCR lines of a PIL image: [{"t": text, "rect": (x, y, w, h),
    "words": [...]}] in IMAGE pixels, or None when OCR is unavailable.
    Small text (< 12 px median) is read again at 1.5x. Never raises."""
    try:
        if img is None:
            return None
        lines = _run_engine(img)
        if lines is None:
            return None
        if lines and _median_height(lines) < _SMALL_TEXT_PX:
            from PIL import Image
            big = img.resize((int(img.size[0] * 1.5), int(img.size[1] * 1.5)),
                             Image.LANCZOS)
            again = _run_engine(big)
            if again:
                for ln in again:
                    for wd in ln.get("words") or ():
                        for k in (1, 2, 3, 4):
                            wd[k] = wd[k] / 1.5
                lines = again
        out = []
        for ln in lines:
            ws = [wd for wd in ln.get("words") or () if len(wd) >= 5]
            if not ws:
                continue
            x0 = min(w[1] for w in ws)
            y0 = min(w[2] for w in ws)
            x1 = max(w[1] + w[3] for w in ws)
            y1 = max(w[2] + w[4] for w in ws)
            out.append({"t": str(ln.get("t") or ""),
                        "rect": (float(x0), float(y0), float(x1 - x0),
                                 float(y1 - y0)),
                        "words": ws})
        return out
    except Exception:
        return None


def lines_to_cands(lines, origin=(0.0, 0.0), scale: float = 1.0) -> list:
    """OCR lines -> resolver candidates in SCREEN pixels (``origin`` = the
    screen position of the image's top-left, ``scale`` = image px per
    screen px)."""
    out = []
    try:
        ox, oy = origin
        s = float(scale or 1.0)
        for ln in lines or ():
            x, y, w, h = ln["rect"]
            out.append({"text": ln["t"], "type": "ocr",
                        "rect": [ox + x / s, oy + y / s, w / s, h / s]})
    except Exception:
        return out
    return out


def status() -> dict:
    with _lock:
        p = _state["proc"]
        return {"backend": backend_name(), "alive": bool(p and p.poll() is None),
                "requests": _state["requests"], "off": _state["off"],
                "engine_ms_total": round(_state["engine_ms"], 1),
                "pid": getattr(p, "pid", None) if p else None}


def shutdown() -> None:
    with _lock:
        p = _state["proc"]
        _state["proc"] = None
        _state["out"] = None
    if p is not None:
        try:
            p.stdin.write("quit\n")
            p.stdin.flush()
            p.wait(3)
        except Exception:
            _kill(p)
