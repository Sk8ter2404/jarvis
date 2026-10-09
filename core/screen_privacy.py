"""core/screen_privacy.py - the ONE rule for which windows JARVIS may read,
capture, store or trace (2026-10-05).

WHY THIS EXISTS
===============
Three copies of "is this window private?" had drifted apart:

  * bobert_companion.screenshot_privacy_block_reason / _privacy_blocklist_
    match - a substring of SCREENSHOT_PRIVACY_BLOCKLIST in the FOCUSED
    window's title (the hard gate inside take_screenshot);
  * skills/ambient_listen._DEFAULT_SCREEN_BLOCKLIST / _is_sensitive_window -
    regexes (password managers, bank hosts, "Authenticator") over title +
    process, plus AMBIENT_SCREEN_BLOCKLIST;
  * core.auth_guard.auth_page (the 2026-10-05 live-turn branch) - sign-in
    hosts, paths and titles.

None looked at windows that were NOT focused, the browser's URL (a bank
whose title doesn't say "banking"), or a password box on the page. Screen
memory (core.screen_memory), the click executor, the vision trace and the
OCR reader all ask this module, so they cannot disagree.

Levels:
  * title_reason(title, process, url)  - owner blocklist + the sensitive
    defaults. This is what the focused-window hard gate in take_screenshot
    uses (an auth page is NOT refused there: the streaming sign-in-wall
    check has to look at sign-in walls);
  * auth_reason(title, url)            - a sign-in page (core.auth_guard,
    when importable);
  * window_private(win)                - either of those, a password box
    (UIA IsPassword) in the window, or an owner exclusion. Private windows
    are never read by UIA or OCR, never stored, never traced;
  * region_gate(rect, windows, target) - every VISIBLE window intersecting a
    capture: masks for the private ones above the target, or a refusal.

Pure stdlib; never raises at the public API.
"""
from __future__ import annotations

import os
import re
import threading
import time
import urllib.parse
from typing import Iterable, NamedTuple, Optional

__all__ = [
    "reads_blocked", "DEFAULT_PATTERNS", "owner_blocklist", "compiled_patterns",
    "title_reason", "auth_reason", "window_private", "region_gate",
    "Gate", "exclude_window", "exclude_app", "excluded", "clear_exclusions",
    "set_exclusions_provider", "BROWSER_PROCESSES", "is_browser_process",
    "url_private", "live_private", "address_bar_private", "UNKNOWN_ADDRESS",
    "visible_private",
    "app_open",
]

# Browsers: their window title does not say which SITE is showing, so a page
# is private by its ADDRESS too (review 2026-10-05: a bank page titled
# "Accounts Overview - Google Chrome" at secure.chase.com was stored, spoken
# and traced because only the title was checked).
BROWSER_PROCESSES = frozenset({"chrome.exe", "msedge.exe", "firefox.exe",
                               "brave.exe", "opera.exe", "vivaldi.exe"})
# The reason for a browser window whose address could not be read, where a
# caller must fail closed (screen memory, a window above a capture target).
UNKNOWN_ADDRESS = "its address could not be read"


def is_browser_process(process) -> bool:
    try:
        return str(process or "").strip().lower() in BROWSER_PROCESSES
    except Exception:
        return False

def reads_blocked() -> bool:
    """True in a test process (tests/__init__.py sets JARVIS_NO_SCREEN_READ=1)
    or under JARVIS_TEST_MODE: nothing may enumerate, read (UI Automation,
    OCR) or capture the owner's real windows - a test injects fakes. The ONE
    switch every screen reader asks (core.screen_scope, core.uia_host,
    core.screen_ocr, the click / watch capture paths)."""
    return (os.environ.get("JARVIS_NO_SCREEN_READ", "").strip() == "1"
            or os.environ.get("JARVIS_TEST_MODE", "").strip() == "1")


# The sensitive-window defaults (moved from skills/ambient_listen, which now
# delegates here). Regexes over "title process url".
DEFAULT_PATTERNS = (
    r"(?i)\b1password\b",
    r"(?i)\bbitwarden\b",
    r"(?i)\bkeepass(?:xc)?\b",
    r"(?i)\blastpass\b",
    r"(?i)\bdashlane\b",
    r"(?i)\bbanking\b",
    r"(?i)\bchase\.com\b",
    r"(?i)\bcapitalone\b",
    r"(?i)\bbankofamerica\b",
    r"(?i)\bwellsfargo\b",
    r"(?i)\b(visa|mastercard)\.com\b",
    r"(?i)\bpaypal\.com\b",
    r"(?i)\bvenmo\.com\b",
    r"(?i)\bcoinbase\b",
    r"(?i)\bcredit\s*card\b",
    # Generic auth screens
    r"(?i)\b(sign\s*in|log\s*in|login).*(password|2fa|otp)\b",
    r"(?i)\bauthenticator\b",
    r"(?i)\bone\s*time\s*passcode\b",
    r"(?i)\bsocial\s*security\b",
    r"(?i)\bssn\b",
)

_lock = threading.Lock()
_compiled: dict = {"key": None, "pats": []}


def _cfg(name, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def owner_blocklist(blocklist=None) -> tuple:
    """SCREENSHOT_PRIVACY_BLOCKLIST entries (stripped, non-blank)."""
    try:
        src = (_cfg("SCREENSHOT_PRIVACY_BLOCKLIST", ()) if blocklist is None
               else blocklist)
        return tuple(str(e).strip() for e in (src or ()) if str(e).strip())
    except Exception:
        return ()


def compiled_patterns(extra=None) -> list:
    """DEFAULT_PATTERNS + AMBIENT_SCREEN_BLOCKLIST (+ ``extra``), compiled
    and cached; a bad pattern is skipped."""
    try:
        if extra is None:
            extra = _cfg("AMBIENT_SCREEN_BLOCKLIST", ())
            if not extra:
                try:
                    import sys
                    bc = sys.modules.get("bobert_companion")
                    extra = getattr(bc, "AMBIENT_SCREEN_BLOCKLIST", ()) or ()
                except Exception:
                    extra = ()
        key = tuple(str(x) for x in (extra or ()))
        with _lock:
            if _compiled["key"] == key and _compiled["pats"]:
                return list(_compiled["pats"])
        pats = []
        for src in tuple(DEFAULT_PATTERNS) + key:
            try:
                pats.append(re.compile(src))
            except re.error:
                continue
        with _lock:
            _compiled["key"] = key
            _compiled["pats"] = pats
        return list(pats)
    except Exception:
        return []


def _host(url) -> str:
    try:
        s = str(url or "")
        if not s:
            return ""
        return (urllib.parse.urlsplit(s if "://" in s else "https://" + s)
                .hostname or "").lower()
    except Exception:
        return ""


def title_reason(title="", process="", url="", blocklist=None) -> Optional[str]:
    """The reason a window is private by its title / process / URL: the
    owner's blocklist entry it contains (case-insensitive), or the default
    pattern it matches; None when it is not. Never raises."""
    try:
        t = str(title or "")
        p = str(process or "")
        u = str(url or "")
        blob = f"{t} {p} {u}".lower()
        for entry in owner_blocklist(blocklist):
            if entry.lower() in blob:
                return entry
        if not blob.strip():
            return None
        for pat in compiled_patterns():
            if pat.search(f"{t} {p} {u}"):
                return "sensitive window"
    except Exception:
        return None
    return None


def auth_reason(title="", url="", screen_texts: Iterable = ()) -> Optional[str]:
    """Why the window is a sign-in page (core.auth_guard.auth_page, when the
    module is present), else None. Never raises."""
    try:
        from core import auth_guard as _ag
    except Exception:
        return _auth_reason_fallback(title, url)
    try:
        why = _ag.auth_page(urls=[url] if url else (),
                            titles=[title] if title else (),
                            screen_texts=tuple(screen_texts or ()))
        return why or None
    except Exception:
        return None


# Until core.auth_guard lands on main (claude/live-turn-fixes-1005), the
# strongest sign-in signals only: the identity-provider hosts and a title
# that starts "Sign in" / "Log in". auth_guard supersedes it when present.
_AUTH_HOSTS = ("accounts.google.com", "login.microsoftonline.com",
               "login.live.com", "account.live.com", "login.microsoft.com",
               "appleid.apple.com", "idmsa.apple.com")
_AUTH_TITLE_RE = re.compile(r"^\s*(?:sign\s*in|log\s*in|login|choose\s+an\s+"
                            r"account)\b|\bgoogle\s+accounts\b", re.IGNORECASE)
_AUTH_PATH_RE = re.compile(r"/(?:login|signin|sign-in|oauth\d?|authorize|"
                           r"accountchooser|choose-account)(?:[/?#]|$)",
                           re.IGNORECASE)


def _auth_reason_fallback(title, url) -> Optional[str]:
    try:
        h = _host(url)
        if h and any(h == a or h.endswith("." + a) for a in _AUTH_HOSTS):
            return "its address is a sign-in page"
        if url and _AUTH_PATH_RE.search(urllib.parse.urlsplit(
                str(url) if "://" in str(url) else "https://" + str(url)).path
                or ""):
            return "its address is a sign-in page"
        if title and _AUTH_TITLE_RE.search(str(title)):
            return "its title is a sign-in page"
    except Exception:
        return None
    return None


# ── owner exclusions ("don't watch this", "don't watch Discord") ────────
_excl_lock = threading.Lock()
_excl = {"hwnds": {}, "hosts": set(), "apps": set()}
_provider = [None]


def set_exclusions_provider(fn) -> None:
    """A callable returning {"apps": [...]} persisted exclusions
    (core.screen_memory registers it). Never raises."""
    _provider[0] = fn if callable(fn) else None


def exclude_window(hwnd, url="", ttl_s: float = 12 * 3600.0) -> None:
    """Exclude one window (until it closes or ``ttl_s``), and for a browser
    the URL host of what it shows."""
    try:
        with _excl_lock:
            if isinstance(hwnd, int):
                _excl["hwnds"][hwnd] = time.time() + float(ttl_s)
            h = _host(url)
            if h:
                _excl["hosts"].add(h)
    except Exception:
        pass


def exclude_app(name) -> None:
    try:
        n = str(name or "").strip().lower()
        if n:
            with _excl_lock:
                _excl["apps"].add(n)
    except Exception:
        pass


def clear_exclusions() -> None:
    with _excl_lock:
        _excl["hwnds"].clear()
        _excl["hosts"].clear()
        _excl["apps"].clear()


def _app_names() -> set:
    names = set()
    with _excl_lock:
        names |= set(_excl["apps"])
    try:
        fn = _provider[0]
        if fn is not None:
            for a in (fn() or {}).get("apps", ()) or ():
                if str(a).strip():
                    names.add(str(a).strip().lower())
    except Exception:
        pass
    return names


def excluded(win) -> Optional[str]:
    """"excluded by the owner" when ``win`` (a dict / object with hwnd,
    title, process, url) is excluded, else None."""
    try:
        hwnd = _get(win, "hwnd")
        now = time.time()
        with _excl_lock:
            for h, until in list(_excl["hwnds"].items()):
                if until < now:
                    _excl["hwnds"].pop(h, None)
            if isinstance(hwnd, int) and hwnd in _excl["hwnds"]:
                return "excluded by the owner"
            host = _host(_get(win, "url"))
            if host and host in _excl["hosts"]:
                return "excluded by the owner"
        proc = str(_get(win, "process") or "").lower()
        title = str(_get(win, "title") or "").lower()
        host = _host(_with_scheme(_get(win, "url")))
        for app in _app_names():
            if app and _app_matches(app, proc, title, host):
                return "excluded by the owner"
    except Exception:
        return None
    return None


def app_open(name, windows=None) -> bool:
    """Is an app the owner names ("Discord", "Gmail", "my bank") open on the
    screen right now - a visible window's process or title part? The
    screen routes ask before claiming "don't watch <name>" (core.dispatcher:
    "stop watching the room" is guard mode's). ``windows`` for tests; the
    real ones otherwise (core.screen_scope). Never raises."""
    try:
        app = " ".join(str(name or "").split()).lower()
        if not app:
            return False
        if windows is None:
            from core import screen_scope as _sc
            windows = _sc.visible_windows()
        for w in windows or ():
            if _app_matches(app, str(_get(w, "process") or "").lower(),
                            str(_get(w, "title") or "").lower(),
                            _host(_with_scheme(_get(w, "url")))):
                return True
    except Exception:
        return False
    return False


_TITLE_PART_SPLIT_RE = re.compile(r"\s+[-–—|·]\s+")


def _app_matches(app: str, proc: str, title: str, host: str) -> bool:
    """Does the owner's "don't watch <app>" name this window? Its process
    ("discord" = Discord.exe), any part of its title ("Gmail" in "Inbox -
    me@x - Gmail - Google Chrome", "YouTube" in "Video - YouTube - Google
    Chrome" - review 2026-10-05: only the LAST part was checked, which for a
    browser is always "Google Chrome"), or its site's address ("bank" in
    mybank.example.com). Whole words in a title, so "bank" does not match
    "Banking basics". Never raises."""
    try:
        a = " ".join(app.split()).lower()
        if not a:
            return False
        stem = proc[:-4] if proc.endswith(".exe") else proc
        if a in (stem, proc) or (len(a) >= 4 and a in stem):
            return True
        word = re.compile(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])")
        browser = is_browser_process(proc)
        parts = [p.strip() for p in _TITLE_PART_SPLIT_RE.split(title)
                 if p.strip()]
        if browser and parts and re.match(
                r"^(?:google chrome|microsoft\s*edge|brave|opera|vivaldi|"
                r"mozilla firefox)$", parts[-1]):
            parts = parts[:-1]              # the browser's own name
        if any(word.search(p) for p in parts):
            return True
        squashed = re.sub(r"[^a-z0-9]", "", a)
        if host and len(squashed) >= 4 and squashed in host.replace("-", ""):
            return True
    except Exception:
        return False
    return False


def _get(win, key, default=None):
    try:
        if isinstance(win, dict):
            return win.get(key, default)
        return getattr(win, key, default)
    except Exception:
        return default


def window_private(win, has_password: bool = False) -> Optional[str]:
    """Why ``win`` (hwnd, title, process, url) must not be read, captured,
    stored or traced: an owner exclusion, a blocklist / sensitive match, a
    sign-in page, or a password box on it; else None. Judges only the
    fields it is given - for a browser window use live_private, which adds
    the CURRENT address. Never raises."""
    try:
        why = excluded(win)
        if why:
            return why
        title = _get(win, "title") or ""
        proc = _get(win, "process") or ""
        url = _get(win, "url") or ""
        why = title_reason(title, proc, url)
        if why:
            return why
        why = auth_reason(title, _with_scheme(url))
        if why:
            return why
        if has_password or _get(win, "has_password"):
            return "a password box is on it"
    except Exception:
        return "privacy check failed"
    return None


def _with_scheme(url) -> str:
    """An address as the address bar shows it ("secure.example.com/x") with a
    scheme, so address rules written for "https://..." apply. Never
    raises."""
    try:
        u = str(url or "").strip()
        if not u:
            return ""
        if "://" in u or u.lower().startswith(("about:", "chrome:", "edge:",
                                               "file:", "data:")):
            return u
        if re.match(r"^[A-Za-z]:[\\/]", u):
            return u
        return "https://" + u
    except Exception:
        return ""


def url_private(url) -> Optional[str]:
    """Why an ADDRESS alone makes a page private (the owner's blocklist and
    the sensitive defaults over the address - a bank's host - or a sign-in
    address), else None. Takes the address with or without its scheme.
    Never raises."""
    try:
        u = _with_scheme(url)
        if not u:
            return None
        return title_reason("", "", u) or auth_reason("", u)
    except Exception:
        return "privacy check failed"


def live_private(win, url=None, url_of=None, has_password: bool = False,
                 fail_closed: bool = False) -> Optional[str]:
    """window_private, with a browser window's CURRENT address: ``url`` when
    the caller has it, else ``url_of(hwnd)`` (a UI Automation read of the
    address bar). A browser whose address cannot be read is judged on its
    title alone - or, with ``fail_closed``, is private (UNKNOWN_ADDRESS):
    screen memory and windows covering a capture use that. Never raises."""
    try:
        why = window_private(win, has_password=has_password)
        if why:
            return why
        if not is_browser_process(_get(win, "process")):
            return None
        u = url if url else (_get(win, "url") or "")
        if not u and callable(url_of):
            try:
                u = url_of(_get(win, "hwnd")) or ""
            except Exception:
                u = ""
        if not u:
            return UNKNOWN_ADDRESS if fail_closed else None
        return window_private({"hwnd": _get(win, "hwnd"),
                               "title": _get(win, "title") or "",
                               "process": _get(win, "process") or "",
                               "url": u}, has_password=has_password)
    except Exception:
        return "privacy check failed"


def visible_private(windows=None, url_of=None) -> Optional[str]:
    """Why a picture of WHOLE monitors must not be KEPT (the vision trace of
    a see_screen look): a visible window is private - an owner exclusion,
    the blocklist / sensitive defaults by title or process, a sign-in page,
    or a browser page by its current address. take_screenshot's gate only
    judges the FOCUSED window, so a bank tab on another monitor reached
    the model; the trace keeps only "skipped: private" for it (review
    2026-10-05). ``windows`` / ``url_of(hwnd)`` for tests; the real windows
    and UI Automation address reads otherwise. None when none is private
    (or nothing could be listed); "privacy check failed" on an error.
    Never raises."""
    try:
        if windows is None:
            from core import screen_scope as _sc
            windows = _sc.visible_windows()
        if url_of is None:
            def url_of(h):
                from core import screen_text as _st
                return _st.read_url(h, timeout_s=0.2) or ""
        for w in windows or ():
            why = live_private(w, url_of=url_of)
            if why:
                return why
    except Exception:
        return "privacy check failed"
    return None


# The address-bar row of a browser window: below the tab strip, above the
# page (Chrome at 100%: tabs 0-40 px, address bar ~46-80 px).
_TAB_STRIP_PX = 36
_ADDRESS_TOKEN_RE = re.compile(r"[\w.-]+\.[a-z]{2,}(?:[/:?#][^\s]*)?",
                               re.IGNORECASE)


def address_bar_private(ocr_lines, page_top: float) -> Optional[str]:
    """Why the address bar, as OCR read it from a browser-window crop, makes
    the page private. ``ocr_lines`` are dicts with "t" and "rect" [x, y, w,
    h] relative to the crop's top-left; ``page_top`` is where the page
    starts in the crop (the toolbar's bottom). The second line of defence
    when the address could not be read through UI Automation. Never
    raises (a failure is private)."""
    try:
        for ln in ocr_lines or ():
            r = ln.get("rect") if isinstance(ln, dict) else None
            if not r or len(r) < 4:
                continue
            cy = float(r[1]) + float(r[3]) / 2.0
            if not (_TAB_STRIP_PX <= cy <= float(page_top)):
                continue
            for tok in _ADDRESS_TOKEN_RE.findall(str(ln.get("t") or "")):
                why = url_private(tok)
                if why:
                    return why
    except Exception:
        return "privacy check failed"
    return None


# ── captures that span several windows ──────────────────────────────────
class Gate(NamedTuple):
    allowed: bool
    masks: list            # [(x, y, w, h)] screen rects to black out
    reason: str


def _intersect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _area(r) -> float:
    return max(0.0, float(r[2])) * max(0.0, float(r[3]))


def region_gate(rect, windows, target_hwnd=None, *, pad: int = 8,
                whole_desktop: bool = False) -> Gate:
    """May ``rect`` (x, y, w, h, screen pixels) be captured, given the
    visible ``windows`` (z-order, topmost first; dicts with hwnd, rect,
    title, process, url, private)?

      * whole_desktop (legacy callers): refused if ANY private window is
        visible;
      * window-scoped (``target_hwnd``): a private window ABOVE the target
        that intersects is masked (its rect + ``pad``); a private target is
        refused;
      * monitor-scoped (no target): every intersecting private window is
        masked.
    Refused when masks cover more than half of ``rect``. Never raises
    (fails closed)."""
    try:
        rect = tuple(float(v) for v in rect)
        total = _area(rect) or 1.0
        masks = []
        above = True
        for w in windows or ():
            hwnd = _get(w, "hwnd")
            wr = _get(w, "rect")
            if not wr:
                continue
            priv = _get(w, "private")
            if priv is None:
                priv = window_private(w)
            if target_hwnd is not None and hwnd == target_hwnd:
                if priv:
                    return Gate(False, [], f"the window is private ({priv})")
                above = False
                continue
            if not priv:
                continue
            if whole_desktop:
                return Gate(False, [], f"a private window is visible ({priv})")
            if target_hwnd is not None and not above:
                continue          # below the target: covered by it
            x, y, ww, hh = (float(v) for v in wr)
            padded = (x - pad, y - pad, ww + 2 * pad, hh + 2 * pad)
            hit = _intersect(rect, padded)
            if hit:
                masks.append(hit)
        covered = sum(_area(m) for m in masks)
        if covered > 0.5 * total:
            return Gate(False, masks, "private windows cover most of it")
        return Gate(True, masks, "")
    except Exception:
        return Gate(False, [], "privacy check failed")


def apply_masks(img, masks, origin=(0, 0), scale: float = 1.0):
    """``img`` (PIL) with each screen-rect mask blacked out; ``origin`` is
    the screen position of the image's top-left, ``scale`` image px per
    screen px. Never raises (returns None on failure: fail closed)."""
    try:
        if not masks:
            return img
        from PIL import ImageDraw
        out = img.copy()
        d = ImageDraw.Draw(out)
        ox, oy = origin
        for (x, y, w, h) in masks:
            d.rectangle([(x - ox) * scale, (y - oy) * scale,
                         (x - ox + w) * scale, (y - oy + h) * scale],
                        fill=(0, 0, 0))
        return out
    except Exception:
        return None
