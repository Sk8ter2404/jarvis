"""core/grounded_click.py - "click that MrBeast video": look, resolve, guard,
act, VERIFY, report - in code, once (2026-10-05).

WHY THIS EXISTS
===============
Live 00:28:23-00:30:26 (session_2026-10-05_00-21-55): "click that Mr. Beast
video" became a YouTube SEARCH that played a different video in a new
window on the TOP monitor while the page he meant was on the MIDDLE one;
"that's not the right video" / "the one that was on screen at the time"
then ran recall_screen (an invented "Kai Cenat video"), find_on_screen, see_
screen, click, see_screen, click - 41 s, six brain rounds, five vision
calls, zero clicks landed - until the repeat-failure rule stopped it.

This module is the ONE deterministic executor every description click goes
through (click_on_screen, the aliased [ACTION: click, <description>],
find_on_screen, local_click_target_by_description, "the second one" picks):

  scope   the owner-named monitor (hard) > the window JARVIS opened > the
          foreground window > the top window of each other monitor
          (core.screen_scope), never the whole desktop, never JARVIS's own
          windows;
  L0      already playing? (the media session / a YouTube watch window) ->
          "That one's already playing", no click;
  locate  T0 the scene frozen when the turn began, T1 UI Automation names +
          the resolver (core.screen_resolve), T2 OCR of the window crop, T3 a
          question naming the real options, T4 set-of-mark and T5 box_2d on
          the local model (<= 960 image tokens, core.vision_grounding);
  guard   private window -> sign-in page (core.auth_guard) -> JARVIS's own
          host -> send / delete / buy ... held for a yes -> a HIT TEST (the
          element under the point must be the target) -> scroll into view;
  act     the real pointer (bc.ui_click, reticle and failsafe unchanged);
          Invoke only as the one retry, or when a JARVIS window covers it;
  verify  in the SAME window within CLICK_VERIFY_TIMEOUT_S: the URL / title
          / a new tab or window / the element's state / the region's pixels;
  report  one sentence spoken word for word, a vision-trace entry, a screen
          timeline row and an undo record.

Limits per request: <= 4 UIA snapshots, <= 2 model calls (<= 4 per turn),
<= 2 act attempts, and a wall budget enforced by a bounded join on the
worker (4 s text path, 12 s with the local model). A worker that outlives
its budget never clicks afterwards. Never raises at the public API.
"""
from __future__ import annotations

import collections
import contextvars
import io
import re
import threading
import time
import urllib.parse
from typing import NamedTuple, Optional

from core import onscreen_refs as _refs
from core import screen_privacy as _priv
from core import screen_resolve as _R
from core import screen_scope as _scope
from core import vision_trace as _vt

__all__ = [
    "Result", "Backend", "run", "run_bounded", "undo", "pick",
    "freeze_scene", "freeze_scene_async", "scene_match", "pending_choice",
    "clear_pending", "last_ui_action", "note_ui_action", "recent_scenes",
    "forget_scenes", "reset_state", "safe_label",
    "VERIFIED", "NAVIGATED", "NO_CHANGE", "CHANGED_WRONG", "NOT_FOUND",
    "AMBIGUOUS", "TIMED_OUT",
    "ALREADY_PLAYING", "REFUSED_AUTH", "REFUSED_PRIVATE", "ASKED", "FAILED",
    "FOUND",
]

VERIFIED = "verified"
NAVIGATED = "navigated"          # it went somewhere this check cannot judge
# Result.tier of a run that ran out of time (NOT_FOUND, but nothing was
# judged absent: the page may hold it).
TIMED_OUT = "timeout"
NO_CHANGE = "no_change"
CHANGED_WRONG = "changed_wrong"
NOT_FOUND = "not_found"
AMBIGUOUS = "ambiguous"
ALREADY_PLAYING = "already_playing"
REFUSED_AUTH = "refused_auth"
REFUSED_PRIVATE = "refused_private"
ASKED = "asked"
FAILED = "failed"
FOUND = "found"

UI_ACTION_TTL_S = 120.0
PENDING_TTL_S = 90.0
SCENE_FRESH_S = 20.0
SCENE_MAX_AGE_S = 600.0
SCENE_RING = 8
MAX_SNAPSHOTS = 4
MAX_VISION_CALLS = 2
MAX_VISION_PER_TURN = 4
BUDGET_TEXT_S = 4.0
BUDGET_VISION_S = 12.0
CROSS_WINDOW_LEAD = 0.15
_POLL_S = 0.15

_DESTRUCTIVE_RE = re.compile(
    r"\b(send|delete|remove|buy|pay|purchase|confirm|unsubscribe|sign\s*out|"
    r"log\s*out|place\s+order|checkout|check\s+out|discard|erase|format|"
    r"uninstall|transfer)\b", re.IGNORECASE)
_VISUAL_RE = re.compile(
    r"\b(red|blue|green|yellow|orange|purple|pink|black|white|grey|gray|"
    r"gold|silver|colou?r(?:ed)?|picture|image|photo|logo|icon\s+of|"
    r"thumbnail\s+with|the\s+one\s+with|looks?\s+like|face|wearing|"
    r"drawing|cartoon)\b", re.IGNORECASE)
_YT_TITLE_SUFFIX_RE = re.compile(
    r"\s+-\s+YouTube(?:\s+-\s+(?:Google\s+Chrome|Microsoft\s*Edge|Brave|"
    r"Opera|Vivaldi|Mozilla\s+Firefox))?\s*$", re.IGNORECASE)
_BROWSER_SUFFIX_RE = re.compile(
    r"\s+-\s+(?:Google\s+Chrome|Microsoft\s*Edge|Brave|Opera|Vivaldi|"
    r"Mozilla\s+Firefox)\s*$", re.IGNORECASE)


class Result(NamedTuple):
    text: str                 # the one line to speak / return
    outcome: str
    label: str = ""
    monitor: str = ""
    tier: str = ""
    fact: str = ""            # the log fact line
    trace_id: str = ""
    failed: bool = False      # a real failure (UIA down and no fallback ...)


def _cfg(name, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def safe_label(label, max_len: int = 90) -> str:
    """A label to quote in a spoken result: whitespace collapsed, trimmed,
    and its ASCII apostrophes made typographic, so a title such as "You
    Can't Win" never reads as a failure marker ("can't") to the speak
    contract."""
    s = " ".join(str(label or "").split()).replace("'", "’")
    if len(s) > max_len:
        s = s[:max_len - 3].rstrip() + "..."
    return s


def _q(label) -> str:
    return f"'{safe_label(label)}'"


def _mon(m) -> str:
    return f"the {m} monitor" if m else "the screen"


# ── state ────────────────────────────────────────────────────────────────
_lock = threading.RLock()
_scenes: "collections.deque" = collections.deque(maxlen=SCENE_RING)
_state = {"last_ui": None, "pending": None, "uia_down_said": False,
          "last_click_ts": 0.0}
# The run on THIS thread (its Deadline): _set_pending and the act step ask
# it whether the caller already gave up.
_tls = threading.local()


def reset_state() -> None:
    """Tests: forget scenes, the pending question and the last UI action."""
    with _lock:
        _scenes.clear()
        _state.update(last_ui=None, pending=None, uia_down_said=False,
                      last_click_ts=0.0)


def pending_choice(now=None) -> Optional[dict]:
    """The open "which one?" (<= PENDING_TTL_S old), else None."""
    with _lock:
        p = _state["pending"]
        if p is None:
            return None
        if (now or time.time()) - p["ts"] > PENDING_TTL_S:
            _state["pending"] = None
            return None
        return p


def clear_pending() -> None:
    with _lock:
        _state["pending"] = None


def _set_pending(options, said, referent, kind="which", allow_yes=False,
                 **extra):
    """Open a "which one?" / "shall I press it?" for the owner's next
    words. Never from a run its caller already gave up on: that question
    was never spoken, and a later plain "yes" would act on it (review
    2026-10-05: a timed-out worker left "shall I press 'Delete account'?"
    armed). The check and the write share the lock run_bounded takes to
    give up, so the two cannot interleave."""
    dl = getattr(_tls, "deadline", None)
    with _lock:
        if dl is not None and dl.cancelled:
            return False
        _state["pending"] = dict(extra, ts=time.time(), options=list(options),
                                 said=str(said or ""),
                                 referent=str(referent or ""), kind=kind,
                                 allow_yes=bool(allow_yes), owner=dl)
        return True


def last_ui_action(now=None) -> Optional[dict]:
    """JARVIS's own last UI action (<= UI_ACTION_TTL_S old), else None."""
    with _lock:
        rec = _state["last_ui"]
        if rec is None:
            return None
        if (now or time.time()) - rec["ts"] > UI_ACTION_TTL_S:
            return None
        return rec


def note_ui_action(rec: dict) -> None:
    with _lock:
        _state["last_ui"] = dict(rec, ts=rec.get("ts") or time.time())
        _state["last_click_ts"] = _state["last_ui"]["ts"]


def _ledger_undoable(led, b=None) -> bool:
    """May "go back" close the page JARVIS opened (``led``, an
    opened_ledger entry)? Only while nothing happened in it since: no click
    of JARVIS's came after the open (a click - even one taken back - means
    "go back" is about that page, not the open), and its window still shows
    what was opened (the owner has not navigated it himself - review
    2026-10-05: "go back" closed a YouTube window he had clicked a video
    in). A window whose title cannot be read counts as moved on. Never
    raises."""
    try:
        if led is None:
            return False
        with _lock:
            if _state["last_click_ts"] and _state["last_click_ts"] >= led.at:
                return False
        if b is None or led.hwnd is None:
            return False
        if not b.window_alive(led.hwnd):
            return False
        now_title = str(b.window_title(led.hwnd) or "")
        return bool(led.title) and now_title == led.title
    except Exception:
        return False


def undoable(backend=None) -> bool:
    """Is there a JARVIS UI action "go back" / "not that one" may take back
    right now (a click <= UI_ACTION_TTL_S old, or an open nothing happened
    in since)? The screen routes ask this - the ONE rule _undo applies.
    Never raises."""
    try:
        if last_ui_action() is not None:
            return True
        from core import opened_ledger as _ol
        led = _ol.last_opened(UI_ACTION_TTL_S)
        return _ledger_undoable(led, _backend(backend))
    except Exception:
        return False


def recent_scenes(max_age_s: float = SCENE_MAX_AGE_S, now=None) -> list:
    """Scenes (newest first) no older than ``max_age_s``."""
    t = now or time.time()
    with _lock:
        return [s for s in reversed(_scenes) if t - s["ts"] <= max_age_s]


def forget_scenes(since=None, until=None) -> int:
    with _lock:
        lo = float(since) if since is not None else float("-inf")
        hi = float(until) if until is not None else float("inf")
        keep = [s for s in _scenes if not lo <= s["ts"] <= hi]
        n = len(_scenes) - len(keep)
        _scenes.clear()
        _scenes.extend(keep)
        return n


class Deadline:
    def __init__(self, budget_s: float):
        self.t0 = time.monotonic()
        self.budget = float(budget_s)
        self.cancelled = False
        self.acted = None          # (label, monitor) once a press was sent

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.t0)

    def expired(self) -> bool:
        return self.cancelled or self.remaining() <= 0


# ── the production backend ───────────────────────────────────────────────
class Backend:
    """Real windows, UIA, OCR, pointer and the local model. Tests pass an
    object with the same methods instead."""

    def _bc(self):
        import sys
        return sys.modules.get("bobert_companion")

    def windows(self, include_jarvis=False):
        return _scope.visible_windows(include_jarvis=include_jarvis)

    def foreground(self):
        try:
            import ctypes
            return int(ctypes.WinDLL("user32").GetForegroundWindow() or 0)
        except Exception:
            return None

    def ledger(self):
        """(hwnd, monitor, target URL, via, title when opened) of the page
        JARVIS opened last (core.opened_ledger, PAGE_MAX_AGE_S), else
        None."""
        try:
            from core import opened_ledger as _ol
            e = _ol.last_opened(_ol.PAGE_MAX_AGE_S)
            if e is None:
                return None
            return (e.hwnd, e.monitor, e.target, e.via, e.title)
        except Exception:
            return None

    def snapshot(self, win, budget_ms=600):
        from core import screen_text as _st
        return _st.snapshot(win.hwnd, win_rect=win.rect, title=win.title,
                            process=win.process, pid=win.pid,
                            monitor=win.monitor, budget_ms=budget_ms)

    def read_url(self, hwnd):
        from core import screen_text as _st
        return _st.read_url(hwnd)

    def element_at(self, x, y):
        from core import screen_text as _st
        return _st.element_at(x, y)

    def root_at(self, x, y):
        """The top-level window under a screen point (Win32, no UIA) - the
        hit test's answer when UI Automation has none (review 2026-10-05:
        only the test fake had this, so every OCR / box / pixel click was
        "something else is under it" exactly when UIA was down)."""
        from core import screen_text as _st
        return _st._root_at(x, y)

    def href_of(self, el):
        """A link's address, read on demand (one UIA call)."""
        from core import screen_text as _st
        return _st.href_of(el) if getattr(el, "ref", ()) else ""

    def invoke(self, el):
        """True pressed / False not / None sent but unconfirmed (it may
        still land - never follow it with a second press or a key)."""
        from core import screen_text as _st
        if el is None or not getattr(el, "ref", ()):
            return False
        return _st.invoke(el)

    def scroll_into_view(self, el):
        from core import screen_text as _st
        return _st.scroll_into_view(el) if getattr(el, "ref", ()) else None

    def toggle_state(self, el):
        from core import screen_text as _st
        return _st.toggle_state(el) if getattr(el, "ref", ()) else (None, None)

    def tabs(self, hwnd):
        from core import screen_text as _st
        return _st.tab_items(hwnd)

    def back(self, hwnd):
        from core import screen_text as _st
        return _st.back_button(hwnd)

    def close_tab(self, hwnd, name):
        from core import screen_text as _st
        return _st.close_tab(hwnd, name)

    def window_title(self, hwnd):
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(512)
            ctypes.WinDLL("user32").GetWindowTextW(int(hwnd), buf, 512)
            return buf.value or ""
        except Exception:
            return ""

    def window_alive(self, hwnd):
        try:
            import ctypes
            return bool(ctypes.WinDLL("user32").IsWindow(int(hwnd)))
        except Exception:
            return False

    def top_hwnds(self):
        return {w.hwnd for w in _scope.visible_windows(include_jarvis=True)}

    def close_window(self, hwnd):
        """WM_CLOSE (never a kill)."""
        try:
            import ctypes
            ctypes.WinDLL("user32").PostMessageW(int(hwnd), 0x0010, 0, 0)
            return True
        except Exception:
            return False

    def close_last_opened(self):
        from core import actions as _a
        return _a._act_close_last_opened("")

    def focus(self, hwnd):
        try:
            bc = self._bc()
            fn = getattr(bc, "_focus_hwnd", None)
            if callable(fn):
                return bool(fn(hwnd))
            import ctypes
            return bool(ctypes.WinDLL("user32").SetForegroundWindow(int(hwnd)))
        except Exception:
            return False

    def hotkey(self, *keys):
        bc = self._bc()
        bc.ui_hotkey(*keys)

    def click(self, x, y):
        bc = self._bc()
        bc.ui_click(int(x), int(y))

    def now_playing(self):
        try:
            from core import media_now_playing as _np
            return _np.get_now_playing()
        except Exception:
            return None

    def capture(self, rect, target_hwnd=None, windows=None):
        """Native pixels of ``rect`` (PIL), privacy-gated (core.screen_
        privacy.region_gate over every visible window), or None."""
        try:
            if _priv.reads_blocked():
                return None
            bc = self._bc()
            if bc is not None and bc.screenshot_privacy_block_reason():
                return None
            wins = windows if windows is not None else self.windows(True)
            gate = _priv.region_gate(rect, _capture_infos(
                wins, rect, target_hwnd, self.read_url), target_hwnd)
            if not gate.allowed:
                return None
            import mss
            from PIL import Image
            x, y, w, h = (int(v) for v in rect)
            cls = getattr(mss, "MSS", mss.mss)
            with cls() as sct:
                raw = sct.grab({"left": x, "top": y, "width": w, "height": h})
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            if gate.masks:
                img = _priv.apply_masks(img, gate.masks, origin=(x, y))
            return img
        except Exception:
            return None

    def ocr(self, img):
        from core import screen_ocr as _ocr
        return _ocr.ocr_image(img)

    def vision(self, prompt, png):
        """The LOCAL model only (never routed to the cloud)."""
        bc = self._bc()
        if bc is None:
            return None
        try:
            return bc._call_local_vision(prompt, [png], max_tokens=80)
        except Exception:
            return None

    def vision_usable(self):
        try:
            bc = self._bc()
            return bool(bc is not None and bc.SCREEN_VISION_ENABLED
                        and bc._local_vision_usable())
        except Exception:
            return False

    def legacy_find(self, desc, monitor):
        bc = self._bc()
        if bc is None:
            return None
        return bc.find_click_target(desc, monitor=monitor)

    def is_self_close(self, desc):
        try:
            return bool(self._bc()._is_self_close_attempt(desc))
        except Exception:
            return False

    def _frame(self):
        """This owner turn's grounding frame (run_bounded hands it to its
        worker thread - see turn_frame / adopt_frame), else None."""
        try:
            return getattr(self._bc()._turn_grounding, "frame", None)
        except Exception:
            return None

    def turn_frame(self):
        """The calling thread's turn frame, for a worker to adopt."""
        return self._frame()

    def adopt_frame(self, frame):
        """Make ``frame`` this (worker) thread's turn frame; returns the
        previous one. The frame is a per-thread record, so without this a
        click on its worker saw no turn at all (review 2026-10-05: the
        4-calls-per-turn vision cap read 0 and the sign-in guard saw no
        screen texts)."""
        try:
            tg = self._bc()._turn_grounding
            prev = getattr(tg, "frame", None)
            tg.frame = frame
            return prev
        except Exception:
            return None

    def turn_vision(self, add=0):
        """Vision calls made this owner turn (+ ``add``)."""
        try:
            frame = self._frame()
            if frame is None:
                return 0
            frame["vision_calls"] = int(frame.get("vision_calls", 0)) + add
            return frame["vision_calls"]
        except Exception:
            return 0

    def screen_texts(self):
        try:
            fn = getattr(self._bc(), "_turn_screen_texts", None)
            if callable(fn):
                return list(fn())
            frame = self._frame()
            return list((frame or {}).get("screen") or [])
        except Exception:
            return []

    def auth_context(self):
        """What the sign-in guard judges a click by this turn, from the
        same seams core.actions uses (core.auth_guard's _auth_context when
        the module has it): the owner's words, the screen texts, what a
        find looked for, and whether an input was already refused. The
        page JARVIS OPENED is not added here - the caller adds it only for
        that page's own window, while it still shows what was opened."""
        out = {"owner_text": "", "screen_texts": [], "looked_for": [],
               "refused_before": False}
        try:
            bc = self._bc()
            frame = self._frame() or {}
            try:
                said = bc._turn_user_text()
                out["owner_text"] = said if isinstance(said, str) else ""
            except Exception:
                out["owner_text"] = str(frame.get("user_text") or "")
            out["screen_texts"] = [s for s in self.screen_texts()
                                   if isinstance(s, str)]
            fn = getattr(bc, "_turn_click_targets", None)
            looked = fn() if callable(fn) else frame.get("looked_for")
            out["looked_for"] = [s for s in (looked or ())
                                 if isinstance(s, str)]
            fn = getattr(bc, "_turn_auth_refused", None)
            out["refused_before"] = ((fn() is True) if callable(fn)
                                     else bool(frame.get("auth_refused")))
        except Exception:
            pass
        return out

    def note_auth_refused(self):
        """Mark the turn: an input was refused, so the rest of its clicks
        and keys are refused too (core.auth_guard's turn rule)."""
        try:
            fn = getattr(self._bc(), "_turn_note_auth_refused", None)
            if callable(fn):
                fn()
                return
            frame = self._frame()
            if frame is not None:
                frame["auth_refused"] = True
        except Exception:
            pass

    def sleep(self, s):
        time.sleep(s)


_default_backend = [None]


def _backend(b=None):
    if b is not None:
        return b
    if _default_backend[0] is None:
        _default_backend[0] = Backend()
    return _default_backend[0]


def set_default_backend(b) -> None:
    """Tests: make ``b`` the backend for calls that pass none."""
    _default_backend[0] = b


# ── small helpers ────────────────────────────────────────────────────────
def _g(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _center(rect):
    x, y, w, h = rect
    return (int(round(x + w / 2.0)), int(round(y + h / 2.0)))


def _inside(pt, rect, pad=0.0):
    x, y, w, h = rect
    return (x - pad <= pt[0] <= x + w + pad) and (y - pad <= pt[1] <= y + h + pad)


def _overlap(a, b) -> float:
    """Intersection area / area of ``a``."""
    try:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        return (ix * iy) / max(1.0, aw * ah)
    except Exception:
        return 0.0


def _area(r) -> float:
    return max(0.0, float(r[2])) * max(0.0, float(r[3]))


def _page_title(title: str) -> str:
    t = _YT_TITLE_SUFFIX_RE.sub("", str(title or ""))
    return _BROWSER_SUFFIX_RE.sub("", t).strip()


def _norm_url(url) -> str:
    """YouTube by its video id; a local file by its path (the address bar
    shows "C:/dir/page.html" for file:///C:/dir/page.html); anything else
    host (no www) + path."""
    try:
        s = str(url or "").strip()
        if not s:
            return ""
        if s.lower().startswith("file:"):
            path = urllib.parse.unquote(urllib.parse.urlsplit(s).path)
            return "file:" + path.replace("\\", "/").lstrip("/").lower()
        if re.match(r"^[A-Za-z]:[\\/]", s):
            return "file:" + urllib.parse.unquote(
                s.replace("\\", "/")).lower()
        p = urllib.parse.urlsplit(s if "://" in s else "https://" + s)
        host = (p.hostname or "").lower().removeprefix("www.").removeprefix(
            "m.")
        qs = urllib.parse.parse_qs(p.query)
        if host.endswith("youtube.com") or host == "youtu.be":
            if qs.get("v"):
                return "yt:" + qs["v"][0]
            if host == "youtu.be":
                return "yt:" + p.path.strip("/")
        if qs.get("v"):                    # any site's watch?v=<id> page
            return host + p.path.rstrip("/") + "?v=" + qs["v"][0]
        return host + p.path.rstrip("/")
    except Exception:
        return ""


def _parse_arg(arg) -> tuple:
    """(model monitor, description) from "monitor:top|the video ..."."""
    s = str(arg or "").strip()
    m = re.match(r"^\s*monitor:([\w-]+)\s*\|\s*(.*)$", s, re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    return None, s


def _clean_desc(desc) -> str:
    """Strip quotes and a leading 'the video' wrapper the brain adds."""
    s = " ".join(str(desc or "").split())
    s = s.strip(" .")
    s = re.sub(r"[\"“”]", "", s)
    return s


def _kind_of(cand, grid_hit: bool) -> str:
    t = str(_g(cand, "type") or "")
    if grid_hit:
        return "video"
    if t == "Hyperlink":
        return "link"
    if t == "TabItem":
        return "tab"
    if t in ("CheckBox", "RadioButton"):
        return "toggle"
    if t in ("Button", "MenuItem", "ListItem", "SplitButton", "TreeItem",
             "DataItem", "ComboBox"):
        return "button"
    return "unknown"


def _visible_list(cands, n=3) -> list:
    """The first few page titles / link names (reading order) to name when
    nothing matched."""
    try:
        prepared = _R.prepare(cands)
        grid = _R.video_cards(prepared)
        if grid:
            return [_R.label_of(_R.card_target(u))
                    for u in _R.reading_order(grid)][:n]
        names = []
        for c in sorted(prepared, key=lambda c: (c["rect"][1], c["rect"][0])):
            lab = _R.label_of(c)
            if (len(lab) >= 4 and _g(c, "type") in ("Hyperlink", "Button",
                                                      "TabItem", "ocr")
                    and lab not in names):
                names.append(lab)
            if len(names) >= n:
                break
        return names
    except Exception:
        return []


def _cands_from_snapshot(snap) -> list:
    out = []
    try:
        browser_doc = snap.doc_rect is not None
        for el in snap.elements:
            if el.is_password:
                continue
            if browser_doc and not el.in_document and el.ctype != "TabItem":
                continue
            out.append({"text": el.name, "rect": list(el.rect),
                        "type": el.ctype, "href": el.href, "el": el,
                        "hwnd": snap.hwnd})
    except Exception:
        return out
    return out


def _win_dict(win, url="", has_password=False) -> dict:
    return {"hwnd": _g(win, "hwnd"), "title": _g(win, "title", ""),
            "process": _g(win, "process", ""), "url": url or _g(win, "url", ""),
            "rect": _g(win, "rect"), "has_password": has_password}


def _rects_meet(a, b) -> bool:
    try:
        ax, ay, aw, ah = (float(v) for v in a)
        bx, by, bw, bh = (float(v) for v in b)
        return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
    except Exception:
        return True


def _capture_infos(wins, rect, target_hwnd, url_of) -> list:
    """The region gate's view of every visible window: private by title /
    process, and - for a browser window the capture touches - by its
    current ADDRESS (a bank whose tab title does not say so). A browser
    above the target whose address cannot be read is masked (fail closed);
    the target itself is checked by its address bar after OCR instead
    (screen_privacy.address_bar_private). Never raises."""
    infos = []
    for w in wins or ():
        try:
            priv = _priv.window_private(w)
            if (not priv and _priv.is_browser_process(_g(w, "process"))
                    and _rects_meet(_g(w, "rect"), rect)):
                priv = _priv.live_private(
                    w, url_of=url_of,
                    fail_closed=(_g(w, "hwnd") != target_hwnd))
        except Exception:
            priv = "privacy check failed"
        infos.append({"hwnd": _g(w, "hwnd"), "rect": _g(w, "rect"),
                      "title": _g(w, "title"), "process": _g(w, "process"),
                      "private": priv})
    return infos


def _uia_allowed(w, said, fg, mode=None) -> bool:
    """May UI Automation read this window? A browser: yes. Another app
    (SCREEN_UIA_NONBROWSER): "off" never, "always" yes, "on_demand" only the
    app the click or look is about - the window in front, or one the
    owner's words name (review 2026-10-05: every non-browser window in
    scope was read, and an Electron app read through UIA can switch into
    screen-reader mode). Never raises."""
    try:
        if _scope.is_browser(w):
            return True
        m = str(mode or _cfg("SCREEN_UIA_NONBROWSER", "on_demand")).lower()
        if m == "off":
            return False
        if m == "always":
            return True
        if fg is not None and _g(w, "hwnd") == fg:
            return True
        return _names_app(said, w)
    except Exception:
        return False


def _names_app(said, w) -> bool:
    """Do the owner's words name this window's app ("... in Notepad",
    "the Discord one")? Never raises."""
    try:
        words = set(re.findall(r"[a-z0-9]+", str(said or "").lower()))
        if not words:
            return False
        proc = str(_g(w, "process", "") or "").lower()
        stem = proc[:-4] if proc.endswith(".exe") else proc
        if len(stem) >= 3 and stem in words:
            return True
        parts = re.split(r"\s+[-–—|]\s+", str(_g(w, "title", "") or ""))
        app = parts[-1].lower() if parts else ""
        toks = [t for t in re.findall(r"[a-z0-9]+", app) if len(t) >= 4]
        return bool(toks) and all(t in words for t in toks)
    except Exception:
        return False


# ── scene freeze (T0) ────────────────────────────────────────────────────
def _scope_now(said, model_monitor, b, windows=None):
    wins = windows if windows is not None else b.windows()
    led = b.ledger()
    led_hwnd = led[0] if led else None
    fg = b.foreground()
    return _scope.scope_for(said, model_monitor, led_hwnd, fg, windows=wins,
                            url_of=b.read_url), led


def freeze_scene(said, backend=None, referent=None) -> bool:
    """Snapshot (text only) the in-scope windows for this turn, into the
    scene ring. Returns True when something was frozen. Never raises."""
    try:
        if not _cfg("SCREEN_UIA_ENABLED", True):
            return False
        b = _backend(backend)
        sc, _led = _scope_now(said, None, b)
        fg = b.foreground()
        snaps = {}
        for w in list(sc.windows)[:MAX_SNAPSHOTS]:
            if not _uia_allowed(w, said, fg):
                continue
            # The address FIRST, so a page private by its address (a bank
            # tab titled "Accounts Overview") is never read at all.
            url = (b.read_url(w.hwnd) or "") if _scope.is_browser(w) else ""
            if _priv.live_private(w, url=url or None):
                continue
            s = b.snapshot(w, budget_ms=250)
            if s is None:
                continue
            if s.has_password or _priv.live_private(
                    _win_dict(w, s.url or url, s.has_password)):
                continue
            snaps[w.hwnd] = {"snap": s, "win": w}
        if not snaps:
            return False
        scene = {"ts": time.time(), "said": str(said or ""),
                 "referent": (referent if referent is not None
                              else _refs.referent_phrase(said)),
                 "windows": snaps}
        with _lock:
            _scenes.append(scene)
        try:
            from core import screen_timeline as _tl
            for v in snaps.values():
                s = v["snap"]
                vis = _visible_list(_cands_from_snapshot(s), n=12)
                _tl.add(source="scene", monitor=s.monitor, hwnd=s.hwnd,
                        process=s.process, title=s.title, url=s.url,
                        text="\n".join(vis))
        except Exception:
            pass
        return True
    except Exception:
        return False


def freeze_scene_async(said, backend=None) -> bool:
    """Freeze the scene on a short worker (bounded by the UIA host's own
    timeouts) so the turn never waits for it."""
    try:
        t = threading.Thread(target=freeze_scene, args=(said, backend),
                             daemon=True, name="scene-freeze")
        t.start()
        return True
    except Exception:
        return False


def scene_match(referent, said="", backend=None, fresh_only=True) -> dict:
    """Resolve ``referent`` against the latest frozen scene (<= SCENE_FRESH_S
    when ``fresh_only``). {"status", "score", "target", "monitor", "hwnd",
    "options"}; status "none" when there is no scene or no match. Used by
    the rewrite guard. Never raises."""
    try:
        scenes = recent_scenes(SCENE_FRESH_S if fresh_only else SCENE_MAX_AGE_S)
        if not scenes:
            return {"status": "none"}
        scene = scenes[0]
        best = None
        for v in scene["windows"].values():
            res = _R.resolve(referent, _cands_from_snapshot(v["snap"]))
            if res["status"] == "none":
                continue
            score = float(res.get("score") or 0.0)
            item = dict(res, monitor=v["snap"].monitor, hwnd=v["snap"].hwnd)
            if best is None or score > best.get("score", 0):
                best = item
        return best or {"status": "none"}
    except Exception:
        return {"status": "none"}


# ── L0: already playing ─────────────────────────────────────────────────
def _playing_titles(windows, b) -> list:
    """[(title, monitor)] of what is playing: YouTube watch windows (by
    their title) and the media session's title."""
    out = []
    try:
        for w in windows or ():
            t = str(_g(w, "title") or "")
            if _YT_TITLE_SUFFIX_RE.search(t):
                pt = _page_title(t)
                if pt and pt.lower() not in ("youtube",):
                    out.append((pt, _g(w, "monitor")))
        np = b.now_playing() or {}
        t = str(np.get("title") or "").strip()
        if t and not any(_R.squash(t) == _R.squash(o[0]) for o in out):
            out.append((t, None))
    except Exception:
        return out
    return out


_PLAYING_WORDS_RE = re.compile(r"\b(?:playing|already|currently|on\s+now)\b",
                               re.IGNORECASE)


def _playing_referent(desc, playing, hard_monitor=None) -> Optional[tuple]:
    """(title, monitor) for "the one that's playing" (playing words, one
    thing playing there), else None. Never raises."""
    try:
        if not _PLAYING_WORDS_RE.search(str(desc or "")):
            return None
        here = [(t, m) for t, m in playing
                if not hard_monitor or m in (hard_monitor, None)]
        located = [(t, m) for t, m in here if m]
        if len(located) == 1:
            return located[0]
        if len(here) == 1:
            return here[0]
    except Exception:
        return None
    return None


def _already_playing(desc, playing, hard_monitor=None) -> Optional[tuple]:
    """(title, monitor) when ``desc`` names what is already playing: "the
    one that's playing", or a strong title match (not just its channel).
    A monitor the owner named is a HARD filter: only what plays there
    counts, and a media-session title (monitor unknown) does not (review
    2026-10-05: "... on the middle monitor" answered "already playing on
    the top monitor"). The title match is asked only AFTER the page was
    read and matched nothing (see _run). Never raises."""
    try:
        pr = _playing_referent(desc, playing, hard_monitor)
        if pr is not None:
            return pr
        if not _R.content_tokens(desc):
            return None
        for title, mon in playing:
            if hard_monitor and mon != hard_monitor:
                continue
            res = _R.resolve(desc, [{"text": title, "rect": [0, 0, 400, 20],
                                     "type": "Text"}])
            if res["status"] == "ok" and float(res["score"]) >= _R.REWRITE_MIN:
                return (title, mon)
    except Exception:
        return None
    return None


def _is_playing(label, monitor, playing) -> Optional[tuple]:
    """(title, monitor) when the matched element ``label`` IS what is
    playing on ``monitor`` (or in the media session), else None."""
    try:
        sq = _R.squash(label)
        if not sq:
            return None
        for title, mon in playing:
            if _R.squash(title) == sq and mon in (monitor, None):
                return (title, mon or monitor)
    except Exception:
        return None
    return None


def _already_playing_result(title, mon, st) -> "Result":
    st.set(source="L0")
    st.finish(ALREADY_PLAYING, evidence=f"playing: {title}")
    where = f" on the {mon} monitor" if mon else ""
    _timeline("click", {"monitor": mon}, f"asked to click '{title}': "
              "already playing")
    return Result(f"That one's already playing{where}, sir.",
                  ALREADY_PLAYING, label=title, monitor=mon or "",
                  tier="L0", fact=f"already playing: {title!r}")


# ── sign-in guard (core.auth_guard when present) ───────────────────────
_READY_LINE = ("The sign-in page is up and ready for you, sir - I'll leave "
               "the signing in to you.")


def _takes(fn, *names) -> bool:
    """Does callable ``fn`` accept every keyword in ``names``? Never
    raises."""
    try:
        import inspect
        params = inspect.signature(fn).parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return True
        return all(n in params for n in names)
    except Exception:
        return False


def _auth_refusal(label, said, urls=(), titles=(), screen_texts=(),
                  looked_for=(), refused_before=False) -> str:
    try:
        from core import auth_guard as _ag
        kw = {"urls": urls, "titles": titles, "screen_texts": screen_texts}
        # The turn rule's arguments only where this auth_guard takes them
        # (asked of its signature - a TypeError raised INSIDE the guard must
        # not be read as "old signature" and retried).
        if _takes(_ag.click_refusal, "looked_for", "refused_before"):
            kw.update(looked_for=looked_for, refused_before=refused_before)
        return _ag.click_refusal(label, said, **kw) or ""
    except ImportError:
        pass
    except Exception:
        return ""
    # Fallback until core.auth_guard is on main: a sign-in control (an
    # e-mail address, "sign in", "continue with ...") or a sign-in page -
    # or an input already refused this turn - unless the owner's own words
    # click that very thing. Superseded by core.auth_guard once merged.
    try:
        from core.failure_markers import TERMINAL_FAILURE_PREFIX
        lab = str(label or "")
        control = bool(re.search(
            r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\b(?:sign[\s-]?in|log[\s-]?in|"
            r"login|continue\s+(?:with|as)\s+\w+|choose\s+an\s+account|"
            r"use\s+another\s+account|password|passkey|authori[sz]e)\b",
            lab, re.IGNORECASE))
        page = (any(_priv.auth_reason(t, "") for t in titles or ())
                or any(_priv.auth_reason("", u) for u in urls or () if u))
        if not (control or page or refused_before):
            return ""
        s = str(said or "").lower()
        words = set(re.findall(r"[a-z0-9]+", lab.lower())) - {
            "the", "a", "an", "to", "with", "button", "link"}
        if (re.search(r"\b(?:click|press|tap|select|choose|pick|hit)\b", s)
                and words and words & set(re.findall(r"[a-z0-9]+", s))):
            return ""
        return TERMINAL_FAILURE_PREFIX + _READY_LINE
    except Exception:
        return ""


def _screen_texts(snap, b) -> list:
    texts = []
    try:
        texts.extend(b.screen_texts() or [])
    except Exception:
        pass
    try:
        if snap is not None:
            heads = [el.name for el in snap.elements
                     if el.ctype == "Text" and el.in_document][:12]
            if heads:
                texts.append(" | ".join(heads))
            if snap.has_password:
                texts.append("a sign in form: enter your password")
    except Exception:
        pass
    return texts


# ── locate ───────────────────────────────────────────────────────────────
def _vision_mode() -> str:
    v = str(_cfg("VISION_GROUNDING_FORMAT", "box2d") or "box2d").lower()
    return v if v in ("box2d", "pixel", "off") else "box2d"


def _vision_on(b) -> bool:
    try:
        return _vision_mode() != "off" and bool(b.vision_usable())
    except Exception:
        return False


def _decide(per, priors):
    """('ok', row) | ('ambiguous', [(win, snap, cand, cands), ...]) |
    ('none', None) across windows (SPEC 4.1)."""
    oks, ambs = [], []
    for row in per:
        w, snap, cands, res, tier = row
        p = float(priors.get(_g(w, "hwnd"), 0.0))
        st = res.get("status")
        if st == "ok":
            oks.append((float(res.get("score") or 0) + p, row))
        elif st == "ambiguous":
            ambs.append((float(res.get("score") or 0) + p, row))
    if oks:
        oks.sort(key=lambda x: -x[0])
        rivals = [s for s, _r in oks[1:]] + [s for s, _r in ambs]
        if not rivals or oks[0][0] - max(rivals) >= CROSS_WINDOW_LEAD:
            return "ok", oks[0][1]
        opts = [(r[0], r[1], r[3]["target"], r[2]) for _s, r in oks]
        opts += [(r[0], r[1], o, r[2])
                 for _s, r in sorted(ambs, key=lambda x: -x[0])
                 for o in r[3]["options"]]
        return "ambiguous", opts
    if ambs:
        ambs.sort(key=lambda x: -x[0])
        return "ambiguous", [(r[0], r[1], o, r[2]) for _s, r in ambs
                             for o in r[3]["options"]]
    return "none", None


def _option_dict(w, cand, cands=None) -> dict:
    pos = _R.describe_position(cand, cands) if cands else ""
    return {"label": _R.label_of(cand), "rect": list(cand["rect"]),
            "hwnd": _g(w, "hwnd"), "monitor": _g(w, "monitor"),
            "type": cand.get("type"), "href": cand.get("href", ""),
            "window_title": _g(w, "title", ""), "position": pos}


def _question(options, said, referent, n_total=None, lead="") -> Result:
    """T3: "I see 2 matches on the middle monitor - 'A' (top row, 2nd) or
    'B' (...). Which one, sir?" + the pending choice."""
    options = list(options)[:3]
    _set_pending(options, said, referent)
    mons = {o.get("monitor") for o in options}
    n = n_total or len(options)
    parts = []
    for o in options:
        bit = _q(o["label"])
        if len(mons) > 1 and o.get("monitor"):
            bit += f" on the {o['monitor']} monitor"
        elif o.get("position"):
            bit += f" ({o['position']})"
        parts.append(bit)
    names = parts[0] if len(parts) == 1 else (", ".join(parts[:-1]) + " or "
                                              + parts[-1])
    only = next(iter(mons)) if len(mons) == 1 else None
    where = f" on the {only} monitor" if only else ""
    if lead and len(parts) == 1:
        text = f"{lead} Did you mean {names}?"
    elif lead:
        text = f"{lead} Which one did you mean: {names}?"
    elif n == 1:
        text = f"Did you mean {names}, sir?"
    else:
        text = f"I see {n} matches{where} — {names}. Which one, sir?"
    return Result(text, AMBIGUOUS, tier="ask",
                  fact=f"asked: {n} match(es) for {referent!r}")


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_BROWSER_TOOLBAR_PX = 88


def _ocr_tier(desc, w, snap, b, st, playing_title):
    """T2: OCR of the window crop. (cands, result) or (None, None); for a
    browser window whose ADDRESS bar (as OCR read it) is private - a bank,
    a sign-in page - (None, {"status": "private", ...}) and nothing of it
    is kept, traced or named (review 2026-10-05: this fallback runs exactly
    when the address could not be read through UI Automation)."""
    try:
        img = b.capture(_g(w, "rect"), target_hwnd=_g(w, "hwnd"))
        if img is None:
            return None, None
        lines = b.ocr(img)
        if lines is None:
            return None, None
        x0, y0 = _g(w, "rect")[0], _g(w, "rect")[1]
        top = snap.doc_rect[1] if (snap is not None and snap.doc_rect) else None
        if top is None and _scope.is_browser(w):
            top = y0 + _BROWSER_TOOLBAR_PX            # the browser's toolbar
        if _scope.is_browser(w):
            why = _priv.address_bar_private(lines, (top or y0) - y0)
            if why:
                return None, {"status": "private", "why": why}
        from core.screen_ocr import lines_to_cands
        cands = lines_to_cands(lines, origin=(x0, y0))
        if top is not None:
            cands = [c for c in cands if c["rect"][1] >= top - 2]
        for c in cands:
            c["hwnd"] = _g(w, "hwnd")
        try:
            st.add_image(img.convert("L"),
                         as_sent=f"{img.size[0]}x{img.size[1]}",
                         native_region=list(_g(w, "rect")), label="ocr")
        except Exception:
            pass
        return cands, _R.resolve(desc, cands, playing_title=playing_title)
    except Exception:
        return None, None


def _vision_tiers(desc, w, cands, b, dl, st):
    """T4 set-of-mark (candidate boxes exist) or T5 two-stage box_2d on the
    window crop. Returns (target cand, tier) or (None, why)."""
    from core import vision_grounding as _vgr
    if dl.remaining() < 2.5:
        return None, "no time"
    if b.turn_vision() >= MAX_VISION_PER_TURN:
        return None, "turn budget"
    img = b.capture(_g(w, "rect"), target_hwnd=_g(w, "hwnd"))
    if img is None:
        return None, "no capture"
    wx, wy = _g(w, "rect")[0], _g(w, "rect")[1]
    fit, scale = _vgr.fit_for_vlm(img)
    boxes = [c for c in (cands or []) if c.get("rect")
             and _area(c["rect"]) >= 64][:30]
    calls = 0
    if len(boxes) >= 2:
        rects = [((c["rect"][0] - wx) * scale, (c["rect"][1] - wy) * scale,
                  c["rect"][2] * scale, c["rect"][3] * scale) for c in boxes]
        marked, numbers = _vgr.draw_marks(fit, rects)
        ans = b.vision(_vgr.prompt_mark(desc), _png(marked))
        b.turn_vision(1)
        calls += 1
        parsed = _vgr.parse_reply(ans, "mark")
        st.set(parsed=parsed)
        if parsed.get("kind") == "mark":
            n = parsed["mark"]
            if 1 <= n <= len(boxes) and numbers[n - 1] == n:
                return dict(boxes[n - 1], agree=True), "mark"
        if parsed.get("kind") == "none":
            return None, "model: not there"
    if (calls >= MAX_VISION_CALLS or dl.remaining() < 2.0
            or b.turn_vision() >= MAX_VISION_PER_TURN):
        return None, "vision budget"
    ans = b.vision(_vgr.prompt_box2d(desc), _png(fit))
    b.turn_vision(1)
    calls += 1
    p1 = _vgr.parse_reply(ans, "box2d")
    st.set(parsed=p1)
    if p1.get("kind") != "box":
        return None, f"model: {p1.get('kind')}"
    bx, by, bw, bh = _vgr.box_to_rect(p1["box"], fit.size[0], fit.size[1])
    r1 = (wx + bx / scale, wy + by / scale, bw / scale, bh / scale)
    c1 = _center(r1)
    if (calls >= MAX_VISION_CALLS or b.turn_vision() >= MAX_VISION_PER_TURN
            or dl.remaining() < 2.0):
        return {"text": desc, "rect": list(r1), "type": "box2d",
                "agree": False}, "box2d"
    # Stage 2: a native zoom of max(3x box, 640x640) around it.
    zw = max(640.0, r1[2] * 3)
    zh = max(640.0, r1[3] * 3)
    iw, ih = img.size
    zx0 = min(max(0.0, (c1[0] - wx) - zw / 2), max(0.0, iw - zw))
    zy0 = min(max(0.0, (c1[1] - wy) - zh / 2), max(0.0, ih - zh))
    crop = img.crop((int(zx0), int(zy0), int(min(iw, zx0 + zw)),
                     int(min(ih, zy0 + zh))))
    cfit, cscale = _vgr.fit_for_vlm(crop)
    ans2 = b.vision(_vgr.prompt_box2d(desc), _png(cfit))
    b.turn_vision(1)
    p2 = _vgr.parse_reply(ans2, "box2d")
    if p2.get("kind") != "box":
        return {"text": desc, "rect": list(r1), "type": "box2d",
                "agree": False}, "box2d"
    bx2, by2, bw2, bh2 = _vgr.box_to_rect(p2["box"], cfit.size[0],
                                         cfit.size[1])
    r2 = (wx + zx0 + bx2 / cscale, wy + zy0 + by2 / cscale, bw2 / cscale,
          bh2 / cscale)
    c2 = _center(r2)
    agree = abs(c1[0] - c2[0]) <= 40 and abs(c1[1] - c2[1]) <= 40
    return {"text": desc, "rect": list(r2), "type": "box2d",
            "agree": agree}, "box2d"


# ── guard, act, verify ───────────────────────────────────────────────────
def _jarvis_hwnds(b) -> set:
    try:
        return {w.hwnd for w in b.windows(True) if _g(w, "jarvis")}
    except Exception:
        return set()


def _hit_test(pt, rect, w, b, el, label):
    """('ok', pt) | ('jarvis', None) | ('covered', title) | ('mismatch',
    None). The element under the point must be the target (same name,
    inside it, or the card around it) and its top-level window the target
    window; four inset points are tried before giving up."""
    hwnd = _g(w, "hwnd")
    x, y, rw, rh = rect
    points = [pt, (int(x + rw * 0.25), int(y + rh / 2)),
              (int(x + rw * 0.75), int(y + rh / 2)),
              (int(x + rw / 2), int(y + rh * 0.25)),
              (int(x + rw / 2), int(y + rh * 0.75))]
    covered_by = None
    for p in points:
        ea = b.element_at(*p)
        root = int((ea or {}).get("root_hwnd") or 0) if ea else 0
        if ea is None:
            # No UIA answer at all: accept only when the window under the
            # point (Win32) is the target.
            try:
                r2 = int(b.root_at(*p) or 0)
            except Exception:
                r2 = 0
            if r2 and r2 == hwnd:
                return "ok", p
            if r2 and r2 != hwnd:
                covered_by = r2
            continue
        if root and root != hwnd:
            covered_by = root
            continue
        er = ea.get("rect")
        name = str(ea.get("name") or "")
        same = bool(name) and _R.squash(name) == _R.squash(label)
        contained = bool(er) and _overlap(er, rect) >= 0.8
        card = bool(er) and _overlap(rect, er) >= 0.9 and _area(er) <= 8 * max(
            1.0, _area(rect))
        if same or contained or card or el is None:
            return "ok", p
    if covered_by:
        if covered_by in _jarvis_hwnds(b):
            return "jarvis", None
        try:
            title = next((wi.title for wi in b.windows(True)
                          if wi.hwnd == covered_by), "")
        except Exception:
            title = ""
        return "covered", title
    return "mismatch", None


def _before_state(w, b, url) -> dict:
    hwnd = _g(w, "hwnd")
    st = {"url": url or "", "title": b.window_title(hwnd) or _g(w, "title", ""),
          "tops": set(b.top_hwnds() or ()), "tabs": None, "np": None}
    if _scope.is_browser(w):
        st["tabs"] = b.tabs(hwnd)
    try:
        st["np"] = (b.now_playing() or {}).get("title")
    except Exception:
        st["np"] = None
    return st


def _title_fits(title, label) -> bool:
    """Does a page / window title name ``label`` (the thing clicked)? Most
    of its content words are in the title, or it is there whole. Never
    raises."""
    try:
        t = _page_title(title)
        lab = str(label or "")
        if not t or not lab:
            return False
        sl, st = _R.squash(lab), _R.squash(t)
        if len(sl) >= 4 and sl in st:
            return True
        want = _R.content_tokens(lab)
        if not want:
            return False
        have = _R.toks(t)
        hit = sum(1 for q in want if any(_R.tok_sim(q, x) >= 0.8 for x in have))
        return hit / len(want) >= 0.6
    except Exception:
        return False


def _landing(url_n, href_n) -> str:
    """How the address a click landed on relates to the link's own
    (both _norm_url): "match"; "redirect" (the same site, the link's path
    inside the new one - /docs/install -> /docs/install/latest/ or
    /en-US/docs/install - or a root link); "wrong" (the same site, another
    page / video); "other_site" (a shortener, an ad link, a sign-in hop:
    where it went says nothing about whether it was right); "nolink" when
    the link's address is unknown."""
    if not href_n:
        return "nolink"
    if url_n == href_n:
        return "match"
    if url_n.startswith("yt:") and href_n.startswith("yt:"):
        return "wrong"
    uh, _s, up = url_n.partition("/")
    hh, _s2, hp = href_n.partition("/")
    if uh and uh == hh:
        upath, _q1, uq = up.partition("?")
        hpath, _q2, hq = hp.partition("?")
        upath, hpath = "/" + upath.strip("/"), "/" + hpath.strip("/")
        if upath == hpath:
            return "match" if uq == hq or not hq else "wrong"
        if (hpath == "/" or upath.startswith(hpath + "/")
                or upath.endswith(hpath)
                or (hpath + "/") in (upath + "/")):
            return "redirect"
        return "wrong"
    return "other_site"


def _new_windows(b, before, proc) -> list:
    """[(hwnd, title)] of the top-level windows that appeared since the
    click and belong to the clicked window's app - never JARVIS's own, never
    another app's (review 2026-10-05: a chat pop-up appearing meanwhile was
    "verified", and "go back" then closed it). Never raises."""
    try:
        new = set(b.top_hwnds() or ()) - before["tops"]
        if not new:
            return []
        return [(wi.hwnd, wi.title or "") for wi in b.windows(True)
                if wi.hwnd in new and not _g(wi, "jarvis")
                and str(_g(wi, "process", "") or "").lower() == proc]
    except Exception:
        return []


def _verify(kind, cand, w, before, b, dl, label, el=None, before_img=None):
    """(outcome, evidence, after) - see SPEC 4.4. VERIFIED needs evidence
    from the clicked window's own app: its address became the link's (or a
    redirect inside the same site), a new window / tab of that app, a
    changed state, title or pixels. A VIDEO also needs its title (or the
    link's own address). An address on another site is NAVIGATED (said, not
    undone); the same site's OTHER page is CHANGED_WRONG (taken back)."""
    hwnd = _g(w, "hwnd")
    browser = _scope.is_browser(w)
    proc = str(_g(w, "process", "") or "").lower()
    timeout = min(float(_cfg("CLICK_VERIFY_TIMEOUT_S", 2.5)),
                  max(0.3, dl.remaining()))
    href_n = _norm_url(cand.get("href"))
    t_end = time.monotonic() + timeout
    landed = None
    video = kind == "video"
    toggle0 = b.toggle_state(el) if (el is not None and kind in (
        "tab", "toggle")) else (None, None)
    polls = 0
    while True:
        b.sleep(_POLL_S)
        polls += 1
        url = b.read_url(hwnd) if browser else None
        title = b.window_title(hwnd) or ""
        if url and _norm_url(url) != _norm_url(before["url"]):
            verdict = _landing(_norm_url(url), href_n)
            if verdict == "match" or (verdict in ("redirect", "nolink")
                                      and not video):
                return (VERIFIED, f"the address is now {_short_url(url)}",
                        {"url": url, "title": title})
            if title != before["title"] and _title_fits(title, label):
                return (VERIFIED, f"the page is now {_q(_page_title(title))}",
                        {"url": url, "title": title})
            landed = (url, verdict)        # the title may still be loading
        for nh, nt in _new_windows(b, before, proc):
            if not video or _title_fits(nt, label):
                return (VERIFIED, "a new window opened",
                        {"url": url, "title": nt, "new_hwnds": [nh]})
        if browser and polls % 2 == 1:
            tabs = b.tabs(hwnd)
            if (tabs is not None and before.get("tabs") is not None
                    and len(tabs) > len(before["tabs"])):
                extra = [t for t in tabs if t not in before["tabs"]]
                if not video or any(_title_fits(t, label) for t in extra):
                    return (VERIFIED, "a new tab opened",
                            {"url": url, "title": title, "tabs": tabs})
        if kind in ("tab", "toggle") and el is not None:
            now_state = b.toggle_state(el)
            if now_state != toggle0 and any(v is not None for v in now_state):
                return VERIFIED, "its state changed", {"title": title}
        if (title and title != before["title"] and landed is None
                and (not href_n or not url)
                and (not video or _title_fits(title, label))):
            return VERIFIED, "the window title changed", {"title": title,
                                                          "url": url}
        if kind in ("button", "unknown") and el is not None:
            ea = b.element_at(*_center(cand["rect"]))
            if ea is not None and _R.squash(ea.get("name")) != _R.squash(
                    label):
                return VERIFIED, "the control changed", {"title": title}
        if before_img is not None and kind in ("unknown", "ocr", "box2d",
                                               "mark"):
            after = b.capture(_pad_rect(cand["rect"], 200), _g(w, "hwnd"))
            if after is not None and _mad(before_img, after) > 6.0:
                return VERIFIED, "the screen changed there", {"title": title}
        if time.monotonic() >= t_end or dl.expired():
            break
    if landed:
        url, verdict = landed
        title_now = b.window_title(hwnd) or ""
        if verdict == "wrong":
            return (CHANGED_WRONG, f"it opened {_short_url(url)}",
                    {"url": url, "title": title_now})
        return (NAVIGATED, f"it opened {_short_url(url)}",
                {"url": url, "title": title_now})
    return NO_CHANGE, "", {}


def _short_url(url) -> str:
    s = str(url or "")
    s = re.sub(r"^https?://", "", s)
    return s[:80]


def _pad_rect(rect, pad):
    x, y, w, h = rect
    return (x - pad, y - pad, w + 2 * pad, h + 2 * pad)


def _mad(a, b) -> float:
    """Mean absolute difference of two images as grey 64x64."""
    try:
        from PIL import ImageChops, ImageStat
        a2 = a.convert("L").resize((64, 64))
        b2 = b.convert("L").resize((64, 64))
        return float(ImageStat.Stat(ImageChops.difference(a2, b2)).mean[0])
    except Exception:
        return 0.0


def _timeline(source, w, text, url="") -> None:
    try:
        from core import screen_timeline as _tl
        _tl.add(source=source, monitor=_g(w, "monitor"), hwnd=_g(w, "hwnd"),
                process=_g(w, "process", ""), title=_g(w, "title", ""),
                url=url, text=text)
    except Exception:
        pass


def _guarded_click(cand, w, snap, cands, said, desc, tier, b, dl, st, *,
                   others=(), confirmed=False, grid_hit=False) -> Result:
    """The single chokepoint every description click goes through (SPEC
    4.3): privacy, sign-in, self-target, destructive hold, hit test, scroll
    into view, undo record, act, verify, report."""
    label = _R.label_of(cand) or desc
    el = cand.get("el")
    hwnd = _g(w, "hwnd")
    mon = _g(w, "monitor")
    url = (snap.url if snap is not None and snap.url
           else (b.read_url(hwnd) if _scope.is_browser(w) else "")) or ""
    has_pw = bool(snap is not None and snap.has_password)
    st.set(scope={"monitor": mon, "window_title": _g(w, "title", ""),
                  "process": _g(w, "process", ""),
                  "url_host": urllib.parse.urlsplit(
                      url if "://" in url else "https://" + url).hostname
                  if url else ""},
           chosen=label)
    # 1. privacy (with the page's address: a bank whose title doesn't say)
    why = _priv.live_private(_win_dict(w, url, has_pw), url=url or None)
    if why:
        st.mark_private(why)
        st.finish(REFUSED_PRIVATE)
        return Result("That's in a private window, sir — I'll leave it "
                      "to you.", REFUSED_PRIVATE, label=label, monitor=mon or "")
    # 2. sign-in guard, on the RESOLVED label, judged by THIS window's
    # page (its address and title) and this turn's context - the address
    # JARVIS opened counts only for that page's own window, while it still
    # shows what was opened, and only when the window's own address could
    # not be read (review 2026-10-05: a /login page opened on the top
    # monitor refused a YouTube click on the middle one for ten minutes).
    ctx = b.auth_context() or {}
    urls = [url] if url else []
    led = b.ledger()
    if (not url and led and led[0] == hwnd and led[2]
            and (len(led) < 5 or not led[4]
                 or (b.window_title(hwnd) or "") == led[4])):
        urls.append(led[2])
    refusal = _auth_refusal(label, said or ctx.get("owner_text") or "",
                            urls=urls, titles=[_g(w, "title", "")],
                            screen_texts=_screen_texts(snap, b),
                            looked_for=ctx.get("looked_for") or (),
                            refused_before=bool(ctx.get("refused_before")))
    if refusal:
        b.note_auth_refused()
        st.finish(REFUSED_AUTH, evidence="sign-in guard")
        print("  [click] sign-in guard: not clicking (the owner did not ask "
              "for that click this turn)", flush=True)
        return Result(refusal, REFUSED_AUTH, label=label, monitor=mon or "")
    # 3. self-target
    if b.is_self_close(label):
        st.finish(REFUSED_PRIVATE, evidence="self-target")
        return Result(f"REFUSED: '{label}' looks like an attempt to close the "
                      "terminal or Python process running me.", FAILED,
                      label=label, failed=True)
    # 4. destructive hold
    m = _DESTRUCTIVE_RE.search(label)
    if m and not confirmed:
        verb = m.group(1).split()[0].lower()
        if not re.search(r"\b" + re.escape(verb), str(said or "").lower()):
            opt = _option_dict(w, cand, cands)
            _set_pending([opt], said, desc, kind="confirm", allow_yes=True)
            st.finish(ASKED, evidence=f"destructive: {verb}")
            return Result(f"That's the {_q(label)} button on {_mon(mon)}, sir "
                          "— shall I press it?", ASKED, label=label,
                          monitor=mon or "")
    # 5. point, off-viewport, hit test
    rect = tuple(float(v) for v in cand["rect"])
    wr = _g(w, "rect")
    if el is not None and wr and not _inside(_center(rect), wr):
        nr = b.scroll_into_view(el)
        if nr:
            rect = tuple(float(v) for v in nr)
    pt = _center(rect)
    if cand.get("type") == "box2d" and not cand.get("agree"):
        ea = b.element_at(*pt)
        if not (ea and ea.get("rect") and _overlap(ea["rect"], rect) >= 0.5):
            st.finish(ASKED, evidence="vision answers disagreed")
            return Result(f"I'm not sure exactly where {_q(desc)} is on "
                          f"{_mon(mon)}, sir — could you describe it "
                          "another way?", ASKED, label=label,
                          monitor=mon or "")
    hit, extra = _hit_test(pt, rect, w, b, el, label)
    st.set(precheck=hit)
    if hit == "covered":
        opt = _option_dict(w, cand, cands)
        _set_pending([opt], said, desc, kind="bring_forward", allow_yes=True)
        st.finish(ASKED, evidence=f"covered by {extra!r}")
        return Result(f"{_q(label)} is behind {_q(extra or 'another window')}, "
                      "sir — shall I bring it forward?", ASKED,
                      label=label, monitor=mon or "")
    if hit == "mismatch" and not (el is not None and el.invokable):
        st.finish(ASKED, evidence="hit test: something else under the point")
        return Result(f"Something else is under {_q(label)} on {_mon(mon)} "
                      "right now, sir, so I left it alone.", ASKED,
                      label=label, monitor=mon or "")
    if extra is not None and hit == "ok":
        pt = extra
    # 6. undo record + before state. A link's address is read now, for the
    # one element clicked, when the page read skipped it (its budget).
    kind = _kind_of(cand, grid_hit)
    if (el is not None and not cand.get("href")
            and str(getattr(el, "ctype", "")) == "Hyperlink"):
        try:
            href = b.href_of(el) or ""
        except Exception:
            href = ""
        if href:
            cand = dict(cand, href=href)
    before = _before_state(w, b, url)
    before_img = None
    if kind == "unknown":
        before_img = b.capture(_pad_rect(rect, 200), hwnd)
    # 7. act (never after the budget ran out, never once the caller gave
    # up: the check and the "acted" mark share run_bounded's lock)
    with _lock:
        if dl.expired():
            late = True
        else:
            late = False
            dl.acted = (label, mon or "")
    if late:
        st.finish(FAILED, evidence="out of time before acting")
        return Result("That took too long to find, sir — I stopped "
                      "before clicking anything.", NOT_FOUND, label=label,
                      tier=TIMED_OUT)
    how = "mouse"
    try:
        if hit in ("jarvis", "mismatch"):
            how = "invoke"
            if b.invoke(el) is False:
                st.finish(FAILED, evidence="invoke failed")
                return Result(f"My own window is over {_q(label)}, sir, and I "
                              "could not press it through.", FAILED,
                              label=label, failed=True)
        else:
            b.click(pt[0], pt[1])
    except Exception as e:
        st.finish(FAILED, evidence=str(e)[:120])
        return Result(str(e) or "Click failed, sir.", FAILED, label=label,
                      failed=True)
    st.set(action={"how": how, "x": pt[0], "y": pt[1]})
    # 8. verify (same window)
    outcome, evidence, after = _verify(kind, cand, w, before, b, dl, label,
                                       el=el, before_img=before_img)
    if outcome == NO_CHANGE and el is not None and el.invokable \
            and how == "mouse" and not dl.expired():
        if b.invoke(el) is not False:
            how = "mouse+invoke"
            outcome, evidence, after = _verify(kind, cand, w, before, b, dl,
                                               label, el=el)
    title_now = (after or {}).get("title") or ""
    fact = (f"clicked '{label}' ({tier}, {mon}, '{_g(w, 'title', '')}') at "
            f"({pt[0]},{pt[1]}) - {outcome}: {evidence or '-'}")
    print(f"  [click] {fact}", flush=True)
    note_ui_action({
        "ts": time.time(), "kind": "click", "hwnd": hwnd, "monitor": mon,
        "label": label, "referent": _refs.referent_phrase(said) or desc,
        "url_before": before["url"], "url_after": (after or {}).get("url"),
        "title_before": before["title"], "tabs_before": before["tabs"],
        "tops_before": sorted(before["tops"]),
        "new_hwnds": (after or {}).get("new_hwnds") or [],
        "tabs_after": (after or {}).get("tabs"), "outcome": outcome,
        "options": [_R.label_of(o) for o in others][:3], "said": said,
        "process": _g(w, "process", "")})
    _timeline("click", w, f"clicked '{label}' - {outcome}"
              + (f" ({evidence})" if evidence else ""),
              url=(after or {}).get("url") or url)
    st.set(outcome=outcome, evidence=evidence)
    tid = st.finish(outcome, evidence=evidence)
    if outcome == VERIFIED:
        if kind == "video":
            text = f"Playing {_q(label)} on {_mon(mon)}, sir."
        else:
            text = f"Done, sir — clicked {_q(label)} on {_mon(mon)}."
    elif outcome == NO_CHANGE:
        text = (f"I clicked {_q(label)} on {_mon(mon)}, but nothing changed, "
                "sir.")
    elif outcome == NAVIGATED:
        got = _page_title(title_now) or _short_url((after or {}).get("url"))
        text = (f"I clicked {_q(label)} on {_mon(mon)}, sir — it opened "
                f"{_q(got)}.")
    else:                                    # changed_wrong: go back, ask
        got = _page_title(title_now) or _short_url((after or {}).get("url"))
        # Taking it back is an action too: never once the caller gave up.
        back = False
        if not dl.cancelled:
            u = undo(False, said, backend=b, quiet=True)
            back = u.outcome == VERIFIED
        lead = (f"That opened {_q(got)}, not {_q(label)}, sir"
                + (" — I've gone back." if back else "."))
        alts = [o for o in others
                if _R.squash(_R.label_of(o)) != _R.squash(label)]
        if alts:
            q = _question([_option_dict(w, o, cands) for o in alts], said,
                          desc, lead=lead)
            return q._replace(outcome=CHANGED_WRONG, label=label,
                              monitor=mon or "", tier=tier, fact=fact,
                              trace_id=tid)
        text = lead
    return Result(text, outcome, label=label, monitor=mon or "", tier=tier,
                  fact=fact, trace_id=tid)


# ── the executor ─────────────────────────────────────────────────────────
def _find_line(cand, w, tier) -> str:
    return (f"found '{_R.label_of(cand)}' on {_mon(_g(w, 'monitor'))} in "
            f"'{_g(w, 'title', '')}' [{tier}]")


def _not_found(desc, sc_windows, per, said, b, mode, st, dl=None) -> Result:
    """Nothing matched: name what IS there; or the last time it WAS on
    screen (the scene ring) with an offer to open it. Never an invented
    target. The scene look-back stops when the run's budget does (review
    2026-10-05: it re-matched every saved scene after the caller had given
    up). In mode "find" the line carries "could not find" - a failure the
    follow-up chain's "same action failed twice" rule counts."""
    # Seen earlier (a frozen scene), not on screen now?
    try:
        for scene in recent_scenes():
            if dl is not None and dl.expired():
                break
            for v in scene["windows"].values():
                if dl is not None and dl.expired():
                    break
                res = _R.resolve(desc, _cands_from_snapshot(v["snap"]))
                if res["status"] == "ok":
                    tgt = res["target"]
                    lab = _R.label_of(tgt)
                    when = time.strftime("%H:%M", time.localtime(scene["ts"]))
                    mon = v["snap"].monitor
                    href = tgt.get("href") or ""
                    if mode == "find":
                        st.finish(NOT_FOUND, evidence="seen earlier")
                        return Result(
                            f"could not find '{desc}' on screen now; it was "
                            f"there at {when} on {_mon(mon)} ('{lab}')",
                            NOT_FOUND, label=lab, monitor=mon or "")
                    if href and mode == "click":
                        _set_pending([{"label": lab, "rect": tgt["rect"],
                                       "hwnd": v["snap"].hwnd, "monitor": mon,
                                       "href": href, "type": tgt.get("type")}],
                                     said, desc, kind="open_href",
                                     allow_yes=True)
                        st.finish(NOT_FOUND, evidence="seen earlier; offered")
                        return Result(
                            f"It's not on screen now, sir; I last saw "
                            f"{_q(lab)} at {when} on {_mon(mon)} — shall "
                            "I open it?", NOT_FOUND, label=lab,
                            monitor=mon or "")
                    st.finish(NOT_FOUND, evidence="seen earlier")
                    return Result(f"It's not on screen now, sir; I last saw "
                                  f"{_q(lab)} at {when} on {_mon(mon)}.",
                                  NOT_FOUND, label=lab, monitor=mon or "")
    except Exception:
        pass
    names = []
    for row in per:
        names.extend(n for n in _visible_list(row[2]) if n not in names)
        if len(names) >= 3:
            break
    mons = sorted({_g(w, "monitor") for w in sc_windows if _g(w, "monitor")})
    where = (_mon(mons[0]) if len(mons) == 1 else
             ("your screens" if mons else "the screen"))
    text = f"I don't see {_q(desc)} on {where}, sir."
    if names:
        text += " On screen: " + ", ".join(_q(n) for n in names[:3]) + "."
    if mode == "find":
        st.finish(NOT_FOUND)
        return Result(f"could not find '{desc}' on {where}"
                      + (f"; visible: {', '.join(names[:5])}" if names else ""),
                      NOT_FOUND)
    st.finish(NOT_FOUND, evidence="; ".join(names[:3]))
    return Result(text, NOT_FOUND, fact=f"not found: {desc!r}")


def run(arg, said="", mode="click", backend=None, deadline=None) -> Result:
    """Find (and, mode "click", click) what ``arg`` describes. See the
    module docstring. Never raises."""
    b = _backend(backend)
    dl = deadline or Deadline(BUDGET_VISION_S)
    prev_dl = getattr(_tls, "deadline", None)
    _tls.deadline = dl
    try:
        model_mon, desc = _parse_arg(arg)
        desc = _clean_desc(desc)
        low = desc.lower()
        m = re.match(r"^pick\s*:\s*(-?\d+)$", low)
        if m:
            return pick(int(m.group(1)), said, backend=b, deadline=dl)
        if low in ("scene:previous", "scene:before", "previous"):
            return scene_back(said, backend=b, deadline=dl)
        if not desc:
            return Result("What should I click, sir?", ASKED)
        if mode == "click" and b.is_self_close(desc):
            return Result(f"REFUSED: '{desc}' looks like an attempt to close "
                          "the terminal or Python process running me. Closing "
                          "it would kill my session.", FAILED, failed=True)
        from core import uia_host as _uh
        notice = _uh.disabled_notice()
        with _vt.step("find" if mode == "find" else "click", utterance=said,
                      source="uia") as st:
            r = _run(desc, model_mon, said, mode, b, dl, st)
        if notice and r.text:
            r = r._replace(text=f"{notice} {r.text}")
        return r
    except Exception as e:
        if mode == "find":
            return Result(f"could not finish looking for that "
                          f"({type(e).__name__})", FAILED, failed=True)
        return Result(f"Something went wrong while I was looking for that, "
                      f"sir ({type(e).__name__}).", FAILED, failed=True)
    finally:
        _tls.deadline = prev_dl


def run_bounded(arg, said="", mode="click", backend=None,
                budget_s=None) -> Result:
    """run() on a worker thread with a bounded join (4 s text path, 12 s
    with the local model). The worker runs inside the CALLER's owner turn
    (its turn frame - the per-turn vision cap, the screen texts and the
    sign-in guard's turn rules live there). A worker that outlives the
    budget is cancelled: it never clicks, asks or takes anything back
    afterwards, a question it already armed is withdrawn, and if a press
    had already gone out the reply says so instead of "nothing clicked".
    Never raises."""
    b = _backend(backend)
    if budget_s is None:
        budget_s = BUDGET_VISION_S if _vision_on(b) else BUDGET_TEXT_S
    dl = Deadline(budget_s)
    box: list = [None]
    ctx = contextvars.copy_context()
    try:
        frame = b.turn_frame()
    except Exception:
        frame = None

    def work():
        prev = None
        if frame is not None:
            try:
                prev = b.adopt_frame(frame)
            except Exception:
                prev = None
        try:
            box[0] = run(arg, said, mode, backend=b, deadline=dl)
        finally:
            if frame is not None:
                try:
                    b.adopt_frame(prev)
                except Exception:
                    pass
    t = threading.Thread(target=lambda: ctx.run(work), daemon=True,
                         name="grounded-click")
    t.start()
    t.join(budget_s + 1.5)
    with _lock:
        if box[0] is not None:
            return box[0]
        dl.cancelled = True
        p = _state["pending"]
        if p is not None and p.get("owner") is dl:
            _state["pending"] = None
        acted = dl.acted
    if acted:
        label, mon = acted
        print(f"  [click] gave up after {budget_s:.0f} s - a press on "
              f"{label!r} had already been sent", flush=True)
        return Result(f"I pressed {_q(label)} on {_mon(mon)}, sir, but it took "
                      "too long to confirm what it did.", NO_CHANGE,
                      label=label, monitor=mon)
    print(f"  [click] gave up after {budget_s:.0f} s - nothing clicked",
          flush=True)
    if mode == "find":
        return Result("could not finish looking in time - nothing found yet",
                      NOT_FOUND, tier=TIMED_OUT)
    return Result("That took too long to find, sir — I stopped "
                  "before clicking anything.", NOT_FOUND, tier=TIMED_OUT)


def _run(desc, model_mon, said, mode, b, dl, st) -> Result:
    windows = b.windows()
    sc, _led = _scope_now(said, model_mon, b, windows)
    st.set(utterance=said, candidates=[])
    if not sc.windows:
        st.finish(NOT_FOUND, evidence="no window in scope")
        where = (f" on the {sc.hard_monitor} monitor" if sc.hard_monitor
                 else "")
        if mode == "find":
            return Result(f"could not find a window to look in{where}",
                          NOT_FOUND)
        return Result(f"I don't see a window I may look in{where}, sir.",
                      NOT_FOUND)
    # L0a - "the one that's playing" names what plays, not words to match.
    playing = _playing_titles(windows, b)
    if mode == "click":
        pr = _playing_referent(desc, playing, sc.hard_monitor)
        if pr is not None:
            return _already_playing_result(pr[0], pr[1], st)
    playing_title = next((t for t, m in playing
                          if not sc.hard_monitor or m == sc.hard_monitor),
                         None)
    # T0 / T1 per window
    per = []
    unverified = set()          # browser windows whose address is unknown
    uia_on = bool(_cfg("SCREEN_UIA_ENABLED", True))
    scene = recent_scenes(SCENE_FRESH_S)
    frozen = scene[0]["windows"] if scene else {}
    snaps_used = 0
    fg = b.foreground()
    for w in list(sc.windows):
        if dl.expired():
            break
        hwnd = _g(w, "hwnd")
        browser = _scope.is_browser(w)
        # The address FIRST: a page private by its address alone (a bank
        # tab titled "Accounts Overview") is never read, OCR'd, traced or
        # named (review 2026-10-05).
        url = (b.read_url(hwnd) or "") if browser else ""
        why = _priv.live_private(w, url=url or None)
        if why:
            per.append((w, None, [], {"status": "private", "why": why},
                        "private"))
            continue
        snap, tier = None, "uia"
        fz = frozen.get(hwnd)
        if fz is not None and fz["snap"].title == (b.window_title(hwnd)
                                                   or _g(w, "title")):
            snap, tier = fz["snap"], "scene"
        elif (uia_on and snaps_used < MAX_SNAPSHOTS
              and _uia_allowed(w, said, fg)):
            snap = b.snapshot(w)
            snaps_used += 1
        cands = _cands_from_snapshot(snap) if snap is not None else []
        if snap is not None:
            url = snap.url or url
            why = _priv.live_private(_win_dict(w, url, snap.has_password),
                                     url=url or None)
            if why:
                per.append((w, None, [], {"status": "private", "why": why},
                            "private"))
                continue
        if browser and not url:
            unverified.add(hwnd)
        res = (_R.resolve(desc, cands, playing_title=playing_title)
               if cands else {"status": "none"})
        sparse = (snap is None or len(cands) < 10
                  or sum(len(c["text"]) for c in cands) < 150
                  or bool(getattr(snap, "heavy", False)))
        if res["status"] == "none" and sparse and not dl.expired():
            ocands, ores = _ocr_tier(desc, w, snap, b, st, playing_title)
            if ores is not None and ores.get("status") == "private":
                per.append((w, None, [], ores, "private"))
                continue
            if ores is not None:
                unverified.discard(hwnd)     # its address bar was checked
                cands = cands + (ocands or [])
                res, tier = ores, "ocr"
        per.append((w, snap, cands, res, tier))
    st.set(candidates=[{"window": ("(private)" if r[3].get("status") ==
                                   "private" else _g(r[0], "title", "")[:60]),
                        "status": r[3].get("status"),
                        "score": round(float(r[3].get("score") or 0), 3),
                        "target": _R.label_of(r[3].get("target") or {})}
                       for r in per])
    verdict, row = _decide(per, sc.priors)
    if verdict == "ok":
        w, snap, cands, res, tier = row
        tgt = res["target"]
        # L0b - the thing matched IS what is already playing there.
        lp = (_is_playing(_R.label_of(tgt), _g(w, "monitor"), playing)
              if mode == "click" else None)
        if lp is not None:
            return _already_playing_result(lp[0], lp[1], st)
        grid_hit = bool(_R.video_cards(_R.prepare(cands))) and bool(
            re.search(r"\b(video|one|clip|watch|play)\b", desc.lower()))
        others = list(res.get("others") or [])
        if mode == "find":
            st.finish(FOUND, evidence=_R.label_of(tgt))
            return Result(_find_line(tgt, w, tier), FOUND,
                          label=_R.label_of(tgt), monitor=_g(w, "monitor", ""),
                          tier=tier)
        st.set(source=tier)
        return _guarded_click(tgt, w, snap, cands, said, desc, tier, b, dl,
                              st, others=others, grid_hit=grid_hit)
    if verdict == "ambiguous":
        options = [_option_dict(w, o, cands) for (w, _s, o, cands) in row]
        if mode == "find":
            st.finish(AMBIGUOUS)
            return Result("several matches: " + "; ".join(
                f"'{o['label']}' on the {o['monitor']} monitor"
                + (f" ({o['position']})" if o.get("position") else "")
                for o in options[:5]), AMBIGUOUS)
        st.finish(AMBIGUOUS, evidence=", ".join(o["label"] for o in options[:3]))
        return _question(options, said, desc, n_total=len(options))
    # L0c - nothing on the page matched: is it what is already playing (on
    # the monitor he named, when he named one)?
    if mode == "click":
        lp = _already_playing(desc, playing, sc.hard_monitor)
        if lp is not None:
            return _already_playing_result(lp[0], lp[1], st)
    # T4 / T5 - the local model, on the best window (visual referents, or
    # nothing matched by text). Never a browser window whose address is
    # unknown (its privacy could not be checked).
    vision_rows = [r for r in per if r[3].get("status") == "none"
                   and r[0] is not None and _g(r[0], "hwnd") not in unverified]
    if vision_rows and _vision_on(b) and _vision_mode() == "box2d":
        w, snap, cands, _res, _t = vision_rows[0]
        if _VISUAL_RE.search(desc) or len(cands) >= 2 or not cands:
            with _vt.step("vision", utterance=said, source="vision") as vst:
                tgt, why = _vision_tiers(desc, w, cands, b, dl, vst)
                vst.finish(FOUND if tgt else NOT_FOUND, evidence=str(why))
            if tgt is not None:
                if mode == "find":
                    st.finish(FOUND, evidence=why)
                    return Result(_find_line(tgt, w, why), FOUND,
                                  label=_R.label_of(tgt),
                                  monitor=_g(w, "monitor", ""), tier=why)
                return _guarded_click(tgt, w, snap, cands, said, desc, why,
                                      b, dl, st)
    # Legacy pixel two-pass (find_click_target) pinned to ONE monitor - only
    # when the new vision tiers are switched to "pixel".
    private_hwnds = {_g(r[0], "hwnd") for r in per
                     if r[3].get("status") == "private"}
    pix = [w for w in sc.windows if _g(w, "hwnd") not in unverified
           and _g(w, "hwnd") not in private_hwnds]
    if (mode == "click" and _vision_mode() == "pixel" and _vision_on(b)
            and pix and not dl.expired()):
        w = pix[0]
        pt = b.legacy_find(desc, _g(w, "monitor"))
        if pt is not None:
            cand = {"text": desc, "rect": [pt[0] - 6, pt[1] - 6, 12, 12],
                    "type": "pixel"}
            return _guarded_click(cand, w, None, [], said, desc, "pixel", b,
                                  dl, st)
    # Nothing found, and windows in scope were private. A sign-in page is
    # the owner's to fill in (core.auth_guard's line) - when the thing asked
    # for could be ON it (an account, a sign-in button, his e-mail; or he
    # is signing in this turn), or when nothing else is in scope: never "I
    # don't see it" while it is right there. Any other request with readable
    # windows in scope is an honest "I don't see it" naming what IS there
    # (review 2026-10-05: a sign-in page on the top monitor must not answer
    # "click the dragon video").
    priv = [r[3].get("why") for r in per if r[3].get("status") == "private"]
    if priv and mode == "click":
        all_private = len(priv) == len(per)
        signin = any("sign-in" in str(w) or "sign in" in str(w)
                     for w in priv)
        if signin and (all_private or _sign_in_request(desc, said)):
            from core.failure_markers import TERMINAL_FAILURE_PREFIX
            st.mark_private("sign-in page")
            st.finish(REFUSED_AUTH, evidence="sign-in page in scope")
            return Result(TERMINAL_FAILURE_PREFIX + _READY_LINE, REFUSED_AUTH)
        if all_private:
            st.mark_private("private window")
            st.finish(REFUSED_PRIVATE)
            return Result("That's in a private window, sir — I'll leave "
                          "it to you.", REFUSED_PRIVATE)
    return _not_found(desc, sc.windows, per, said, b, mode, st, dl)


_SIGN_IN_THING_RE = re.compile(
    r"@|\b(?:accounts?|e-?mail|user\s*name|username|profile|sign[\s-]?(?:in|"
    r"on|up)|signin|log[\s-]?(?:in|on)|login|password|passkey|authori[sz]e|"
    r"continue\s+(?:with|as))\b", re.IGNORECASE)
_SIGNING_IN_RE = re.compile(r"\b(?:sign|log)(?:ging)?[\s-]?(?:me\s+)?in\b",
                            re.IGNORECASE)


def _sign_in_request(desc, said) -> bool:
    """Could ``desc`` be a thing ON a sign-in page (an account entry, a
    sign-in button, an e-mail address), or is the owner signing in this
    turn ("... so I can sign in")? Never raises."""
    try:
        try:
            from core import auth_guard as _ag
            if _ag.auth_control(desc):
                return True
        except Exception:
            pass
        return bool(_SIGN_IN_THING_RE.search(str(desc or ""))
                    or _SIGNING_IN_RE.search(str(said or "")))
    except Exception:
        return False


# ── "the second one", "yes", "the one that was on screen" ──────────────
def _live_match(opt, b):
    """(win, snap, cand, cands) for a pending option, found again NOW in its
    window by its label (nearest to where it was), else None."""
    hwnd = opt.get("hwnd")
    win = next((w for w in b.windows() if _g(w, "hwnd") == hwnd), None)
    if win is None:
        return None
    snap = b.snapshot(win)
    cands = _cands_from_snapshot(snap) if snap is not None else []
    want = _R.squash(opt.get("label"))
    hits = [c for c in _R.prepare(cands)
            if _R.squash(c.get("text")) == want
            or _R.squash(c.get("raw")) == want]
    if not hits:
        return None
    ox, oy = _center(opt.get("rect") or (0, 0, 0, 0))
    hits.sort(key=lambda c: abs(_center(c["rect"])[0] - ox)
              + abs(_center(c["rect"])[1] - oy))
    return win, snap, hits[0], cands


def pick(n, said="", backend=None, deadline=None) -> Result:
    """Act on option ``n`` (1-based; -1 = last) of the open question."""
    b = _backend(backend)
    dl = deadline or Deadline(BUDGET_TEXT_S)
    p = pending_choice()
    if p is None:
        return Result("I'm not waiting on a choice just now, sir.", NOT_FOUND)
    opts = p["options"]
    i = (len(opts) - 1) if n == -1 else n - 1
    if not 0 <= i < len(opts):
        return Result(f"I only offered {len(opts)}, sir.", ASKED)
    opt = opts[i]
    clear_pending()
    with _vt.step("pick", utterance=said, source="pick") as st:
        if p["kind"] == "open_href":
            href = opt.get("href") or ""
            try:
                from core import actions as _a
                out = _a._act_open_url(href)
            except Exception as e:
                out = f"failed: {type(e).__name__}"
            st.finish(VERIFIED if href else FAILED, evidence=out[:120])
            return Result(f"Opening {_q(opt['label'])}, sir.", VERIFIED,
                          label=opt["label"])
        if p["kind"] == "bring_forward":
            b.focus(opt.get("hwnd"))
            b.sleep(0.35)
        found = _live_match(opt, b)
        if found is None:
            st.finish(NOT_FOUND, evidence="option gone")
            return Result(f"{_q(opt['label'])} isn't on screen any more, sir.",
                          NOT_FOUND, label=opt["label"])
        win, snap, cand, cands = found
        return _guarded_click(cand, win, snap, cands, p.get("said") or said,
                              p.get("referent") or opt["label"], "pick", b,
                              dl, st, confirmed=(p["kind"] == "confirm"),
                              grid_hit=bool(_R.video_cards(_R.prepare(cands))))


def _scene_before(ts) -> Optional[dict]:
    """The newest scene frozen at or before ``ts`` (the turn that led to
    JARVIS's last action), else the newest scene with a referent."""
    scenes = recent_scenes()
    if ts is not None:
        for s in scenes:
            if s["ts"] <= ts + 1.0 and s.get("referent"):
                return s
    for s in scenes:
        if s.get("referent"):
            return s
    return scenes[0] if scenes else None


def _scene_options(scene, referent, exclude_label=""):
    """(status, [(win, cand, cands)]) for ``referent`` in a frozen scene."""
    oks, ambs = [], []
    for v in scene["windows"].values():
        cands = _cands_from_snapshot(v["snap"])
        res = _R.resolve(referent, cands)
        if res["status"] == "ok":
            oks.append((float(res["score"]), v["win"], res["target"], cands))
        elif res["status"] == "ambiguous":
            for o in res["options"]:
                ambs.append((float(res["score"]), v["win"], o, cands))
    rows = sorted(oks, key=lambda r: -r[0]) + sorted(ambs, key=lambda r: -r[0])
    ex = _R.squash(exclude_label)
    rows = [r for r in rows if not ex or _R.squash(_R.label_of(r[2])) != ex]
    return [(r[1], r[2], r[3]) for r in rows]


def _last_action_ts(b):
    rec = last_ui_action()
    led_at = None
    try:
        from core import opened_ledger as _ol
        e = _ol.last_opened(UI_ACTION_TTL_S)
        led_at = e.at if e is not None else None
    except Exception:
        led_at = None
    ts = [t for t in ((rec or {}).get("ts"), led_at) if t]
    return max(ts) if ts else None


def scene_back(said="", backend=None, deadline=None) -> Result:
    """"I wanted the one that was on screen at the time": the referent of
    the turn before JARVIS's last action, resolved in the scene frozen then;
    clicked if it is still on screen, else offered. Never an invention."""
    b = _backend(backend)
    dl = deadline or Deadline(BUDGET_TEXT_S)
    with _vt.step("scene", utterance=said, source="scene") as st:
        scene = _scene_before(_last_action_ts(b))
        if scene is None or not scene.get("referent"):
            st.finish(NOT_FOUND, evidence="no scene")
            return Result("I have no record of what was on screen then, sir "
                          "— which one do you mean?", NOT_FOUND)
        rec = last_ui_action() or {}
        opts = _scene_options(scene, scene["referent"],
                              exclude_label=rec.get("label", ""))
        if not opts:
            st.finish(NOT_FOUND, evidence="referent not in scene")
            return Result("I have no record of that on screen, sir.",
                          NOT_FOUND)
        if len(opts) > 1:
            st.finish(AMBIGUOUS)
            return _question([_option_dict(w, c, cs) for (w, c, cs) in opts],
                             said, scene["referent"])
        w, cand, cands = opts[0]
        opt = _option_dict(w, cand, cands)
        live = _live_match(opt, b)
        if live is not None:
            win, snap, c2, cs2 = live
            return _guarded_click(c2, win, snap, cs2, said, scene["referent"],
                                  "scene", b, dl, st,
                                  grid_hit=bool(_R.video_cards(
                                      _R.prepare(cs2))))
        when = time.strftime("%H:%M", time.localtime(scene["ts"]))
        if opt.get("href"):
            _set_pending([opt], said, scene["referent"], kind="open_href",
                         allow_yes=True)
            st.finish(NOT_FOUND, evidence="offered to open")
            return Result(f"It's not on screen now; I last saw {_q(opt['label'])} "
                          f"at {when} on {_mon(opt['monitor'])} — shall I "
                          "open it?", NOT_FOUND, label=opt["label"])
        st.finish(NOT_FOUND, evidence="not on screen now")
        return Result(f"It's not on screen now, sir; I last saw "
                      f"{_q(opt['label'])} at {when} on {_mon(opt['monitor'])}.",
                      NOT_FOUND, label=opt["label"])


# ── undo ─────────────────────────────────────────────────────────────────
def _wait(pred, b, timeout=2.0) -> bool:
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        try:
            if pred():
                return True
        except Exception:
            pass
        b.sleep(_POLL_S)
    try:
        return bool(pred())
    except Exception:
        return False


def undo(other=False, said="", backend=None, quiet=False) -> Result:
    """Take back JARVIS's OWN last UI action (<= 120 s): close the window /
    tab it opened, or Back in the window it navigated. ``other``: then ask
    (or, for "the other one" with exactly one other, click) among the
    options from the scene captured BEFORE that action. Never raises."""
    b = _backend(backend)
    try:
        with _vt.step("undo", utterance=said or ("(auto)" if quiet else ""),
                      source="undo", force=quiet) as st:
            return _undo(other, said, b, st)
    except Exception as e:
        return Result(f"I couldn't take that back, sir ({type(e).__name__}).",
                      FAILED, failed=True)


def _undo(other, said, b, st) -> Result:
    rec = last_ui_action()
    try:
        from core import opened_ledger as _ol
        led = _ol.last_opened(UI_ACTION_TTL_S)
    except Exception:
        led = None
    use_ledger = (led is not None and (rec is None or led.at > rec["ts"])
                  and _ledger_undoable(led, b))
    undone, title, action_ts = False, "", None
    if use_ledger:
        action_ts = led.at
        msg = str(b.close_last_opened() or "")
        undone = msg.lower().startswith(("closed", "the "))
        title = _page_title(led.title) or led.target
        st.set(evidence=msg[:160])
        try:
            fg = b.foreground()
            title = _page_title(b.window_title(fg)) if fg else title
        except Exception:
            pass
    elif rec is not None:
        action_ts = rec["ts"]
        hwnd = rec["hwnd"]
        # Only a window the click itself opened: still there, and the
        # clicked app's own (never another app's window that appeared
        # meanwhile, never JARVIS's).
        proc = str(rec.get("process") or "").lower()
        own = {wi.hwnd for wi in b.windows(True)
               if not _g(wi, "jarvis")
               and str(_g(wi, "process", "") or "").lower() == proc}
        new = [h for h in rec.get("new_hwnds") or []
               if h in own and b.window_alive(h)]
        if new:
            for h in new:
                b.close_window(h)
            undone = _wait(lambda: not any(b.window_alive(h) for h in new), b)
            title = rec.get("title_before", "")
        elif (rec.get("tabs_after") is not None and rec.get("tabs_before")
              is not None and len(rec["tabs_after"]) > len(rec["tabs_before"])):
            extra = [t for t in rec["tabs_after"]
                     if t not in rec["tabs_before"]]
            ok = b.close_tab(hwnd, extra[-1]) if extra else False
            if ok is False:               # None = sent, may land: no key
                b.focus(hwnd)
                b.hotkey("ctrl", "w")
            undone = _wait(lambda: len(b.tabs(hwnd) or []) < len(
                rec["tabs_after"]), b, timeout=3.0 if ok is None else 2.0)
            title = rec.get("title_before", "")
        elif rec.get("url_after") and _norm_url(rec["url_after"]) != _norm_url(
                rec.get("url_before")):
            pressed = b.back(hwnd)
            if pressed is False:          # None = sent, may land: no key
                b.focus(hwnd)
                b.hotkey("alt", "left")
            want = _norm_url(rec.get("url_before"))
            undone = _wait(lambda: _norm_url(b.read_url(hwnd)) == want, b,
                           timeout=3.0 if pressed is None else 2.0)
            title = _page_title(b.window_title(hwnd)) or rec.get(
                "title_before", "")
        else:
            undone = False
        with _lock:
            _state["last_ui"] = None
    else:
        st.finish(NOT_FOUND, evidence="nothing to undo")
        if led is not None:
            return Result("You've moved on from the page I opened, sir, so "
                          "I've left it as it is.", NOT_FOUND)
        return Result("There's nothing of mine from the last two minutes to "
                      "take back, sir.", NOT_FOUND)
    outcome = VERIFIED if undone else NO_CHANGE
    st.finish(outcome)
    head = (f"Back on {_q(title)}, sir." if undone and title else
            "Done, sir — I've taken that back." if undone else
            "I couldn't take that back, sir.")
    if not other:
        return Result(head, outcome, failed=not undone)
    scene = _scene_before(action_ts)
    referent = (rec or {}).get("referent") or (scene or {}).get("referent")
    if scene is None or not referent:
        return Result(f"{head} Which one did you mean?", ASKED)
    opts = _scene_options(scene, referent,
                          exclude_label=(rec or {}).get("label", ""))
    if not opts:
        return Result(f"{head} Which one did you mean?", ASKED)
    options = [_option_dict(w, c, cs) for (w, c, cs) in opts]
    if (len(options) == 1 and re.search(r"\bother\s+one\b", str(said or ""),
                                        re.IGNORECASE)):
        _set_pending(options, said, referent)
        r = pick(1, said, backend=b)
        return r._replace(text=f"{head} {r.text}")
    return _question(options, said, referent, lead=head)
