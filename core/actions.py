"""JARVIS action handlers (Phase 4 modularisation).

Each `_act_*` function takes one string argument (the LLM-supplied
argument body) and returns a string the dispatcher feeds back to the
LLM as the action result. The whitelist `ACTIONS` dict in
bobert_companion.py maps action names to these handlers.

Phase 4 strategy — deferred-import pattern:

  • Helpers and module-level state still live in bobert_companion.py.
    Moving them all in one pass would cascade across thousands of
    references; instead each handler that needs a bobert_companion
    helper grabs it via ``bc = _bc()`` at function-body level. The
    import is lazy — bobert_companion is fully loaded by the time
    any handler is called.

  • Handlers with ZERO bobert_companion references (e.g. ``_act_get_time``,
    ``_act_youtube``) don't need the late-bind at all.

  • Handlers with ONE-OR-TWO references prefix with ``bc.``: e.g.
    ``bc.take_screenshot()``, ``bc._get_pyautogui()``,
    ``bc._media_key_with_focus(...)``.

  • The wildcard ``from core.actions import *`` in bobert_companion.py
    re-exports each handler so ``ACTIONS = {"get_time": _act_get_time}``
    still resolves by bare name.

Future migrations: move additional handlers here over time. The
pipeline reviewer's diff size for an action-only fix drops to this
file's size (currently small, grows as Phase 4 progresses) instead of
the full bobert_companion.py.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser

from core import window_scope as _window_scope


def _bc():
    """Late-bound reference to bobert_companion module. Breaks the
    circular import between this module and bobert_companion.py — the
    handler caller-chain only resolves this at runtime, by which time
    bobert_companion is fully loaded."""
    import bobert_companion as _bc_mod
    return _bc_mod


def _apple_music_app():
    """Late-bound reference to the audio.apple_music_app bridge — the lazy,
    never-raises controller for the new UWP Apple Music app. Imported here
    rather than at module top so a missing dep inside the bridge can never
    break core.actions import. Returns None if the bridge can't be imported
    at all (so callers degrade gracefully)."""
    try:
        from audio import apple_music_app as _amapp
        return _amapp
    except Exception:
        return None


# ─── Browser + search basics (zero-or-low bobert_companion deps) ───────

def _site_shortcut_url(name: str) -> "str | None":
    """SITE_SHORTCUTS (core/stt_vocab.py, 2026-10-01): the URL for a spoken site
    name ("accelo"), read live from core.config; None when unset or no match."""
    try:
        from core import config as _cfg
        from core import stt_vocab as _sv
        return _sv.site_shortcut(name, getattr(_cfg, "SITE_SHORTCUTS", None))
    except Exception:
        return None


def _is_data_dir_page(url: str) -> bool:
    """True for a ``file:`` URI naming an existing .html/.htm page inside
    JARVIS's own data dir (a page a skill wrote, e.g. skills/site_builder.py).
    Only those open as-is: any other local file could be an executable, and on
    Windows webbrowser.open hands the URI to os.startfile. Never raises."""
    try:
        import urllib.request
        from core.paths import data_dir
        p = urllib.parse.urlparse(url)
        if (p.scheme.lower() != "file" or p.netloc not in ("", "localhost")
                or p.params or p.query or p.fragment):
            return False
        root = os.path.realpath(data_dir(create=False))
        path = os.path.realpath(urllib.request.url2pathname(p.path))
        return (path.lower().endswith((".html", ".htm"))
                and os.path.commonpath([root, path]) == root
                and os.path.isfile(path))
    except Exception:
        return False


def _loaded_bc():
    """The monolith through _bc() when it is ALREADY loaded, else None -
    never imports it (a light-tier run has no heavy deps, and a failed import
    there is a multi-MB code object that coverage pins forever). Going
    through _bc() keeps a test's patch of it in force."""
    if "bobert_companion" not in sys.modules:
        return None
    try:
        return _bc()
    except Exception:
        return None


def _foreground_snapshot():
    """(hwnd, title, rect) of the foreground window through the running
    monolith's reader; (None, "", None) without one. Never raises."""
    bc = _loaded_bc()
    if bc is None:
        return None, "", None
    try:
        hwnd, title, rect = bc._read_focused_window()
        return hwnd, (title if isinstance(title, str) else ""), rect
    except Exception:
        return None, "", None


def _note_browser_tab_open(via: str, url: str, before, page=None) -> None:
    """Record into core.opened_ledger the browser tab a webbrowser.open just
    brought to the front (S1, 2026-10-02): the foreground window AFTER the
    open, when it is a browser window and something changed in front of the
    owner (``before`` is the snapshot taken before the open). Nothing changed
    in front (the tab landed in a background window, or nothing opened) =
    nothing recorded: "close that" then honestly says it has no record,
    instead of closing whatever tab he has in front. ``page`` is what
    _await_opened_page already confirmed (hwnd, title, rect): recorded
    without a second read (2026-10-05). Never raises."""
    try:
        bc = _loaded_bc()
        if bc is None:
            return
        if page is not None:
            hwnd, title, rect = page[0], page[1], page[2]
        else:
            hwnd, title, rect = _foreground_snapshot()
        if not isinstance(hwnd, int) or not title:
            return
        if (hwnd, title) == (before[0], before[1]):
            return
        page_title = _browser_page_title(bc, title)
        if page_title is None or not _title_fits_url(page_title, url):
            return
        from core.config import MONITORS
        from core import monitor_geometry as _mg
        from core import opened_ledger as _ol
        mon = _mg.monitor_for_rect(*rect, MONITORS) if rect else None
        _ol.note_opened(via, url, hwnd=hwnd, kind="tab", monitor=mon,
                        title=title)
    except Exception:
        pass


# How long open_url / web_search wait for the page they opened (2026-10-05:
# was a fixed 3 s sleep). The wait ends as soon as the browser window in
# front shows a title that names the page (its site or its search words) and
# is not just the address Chrome shows while loading; it gives up early when
# nothing new comes to the front at all (the tab went to a background
# window). Bounded by poll COUNT, so a test's no-op sleep never spins.
_OPEN_WAIT_S = 6.0
_OPEN_POLL_S = 0.15
_OPEN_NOTHING_NEW_S = 2.0
_LOADING_TITLE_RE = re.compile(
    r"^(?:untitled|new tab|loading\.*|about:blank|"
    r"(?:https?://)?[\w.-]+\.[a-z]{2,}(?:[/?#]\S*)?)$", re.IGNORECASE)


def _await_opened_page(url: str, before):
    """(hwnd, title, rect, monitor, page_url) of the browser window that
    came to the front showing ``url``'s page, or None (still loading / it
    landed in a background window). Never raises."""
    try:
        bc = _loaded_bc()
        polls = int(_OPEN_WAIT_S / _OPEN_POLL_S)
        idle_polls = int(_OPEN_NOTHING_NEW_S / _OPEN_POLL_S)
        changed = False
        for i in range(polls):
            time.sleep(_OPEN_POLL_S)
            hwnd, title, rect = _foreground_snapshot()
            if not isinstance(hwnd, int) or not title or \
                    (hwnd, title) == (before[0], before[1]):
                if not changed and i >= idle_polls:
                    return None
                continue
            changed = True
            if bc is None:
                continue
            page = _browser_page_title(bc, title)
            if page is None or _LOADING_TITLE_RE.match(page.strip()):
                continue
            if not _title_fits_url(page, url):
                continue
            page_url = ""
            try:
                from core import screen_text as _st
                page_url = _st.read_url(hwnd, timeout_s=0.3) or ""
            except Exception:
                page_url = ""
            mon = None
            try:
                from core.config import MONITORS
                from core import monitor_geometry as _mg
                mon = _mg.monitor_for_rect(*rect, MONITORS) if rect else None
            except Exception:
                mon = None
            return (hwnd, title, rect, mon, page_url)
    except Exception:
        return None
    return None


def _opened_line(url: str, found, note: str = "") -> str:
    """"opened <url> on the middle monitor - page title '<title>'" or
    "opened <url> (still loading)"."""
    extra = f" ({note})" if note else ""
    if found is None:
        return f"opened {url}{extra} (still loading)"
    bc = _loaded_bc()
    title = _browser_page_title(bc, found[1]) if bc is not None else None
    where = f" on the {found[3]} monitor" if found[3] else ""
    return (f"opened {url}{extra}{where} \u2014 page title "
            f"'{title or found[1]}'")


def _search_results_line(found) -> str:
    """The first result headings (<= 5, <= 600 chars) of the search page in
    window ``found`` - privacy-gated, through UI Automation. "" when they
    can't be read. Never raises."""
    try:
        if found is None:
            return ""
        from core import screen_privacy as _sp
        from core import screen_text as _st
        hwnd, title = found[0], found[1]
        if _sp.window_private({"hwnd": hwnd, "title": title,
                               "url": found[4] or ""}):
            return ""
        snap = _st.snapshot(hwnd, title=title, process="chrome.exe",
                            budget_ms=500, want_hrefs=False)
        if snap is None or snap.has_password:
            return ""
        heads = []
        for e in sorted(snap.elements, key=lambda e: (e.rect[1], e.rect[0])):
            if (e.in_document and e.ctype == "Hyperlink"
                    and 12 <= len(e.name) <= 160 and e.rect[3] <= 60
                    and e.name not in heads):
                heads.append(e.name)
            if len(heads) >= 5:
                break
        out = ", ".join(f'"{h}"' for h in heads)
        return out[:600]
    except Exception:
        return ""


# Host labels that say nothing about which site a page is.
_GENERIC_HOST_LABELS = frozenset({
    "www", "com", "net", "org", "edu", "gov", "co", "uk", "us", "io", "app",
    "play", "web", "m", "en", "tv",
})


def _title_fits_url(page: str, url: str) -> bool:
    """True when browser page title ``page`` plausibly IS the page at ``url``:
    it names the site (a host label such as "youtube" / "google", or the
    streaming service's name), or carries the search words. Review
    2026-10-02: a foreground title that merely CHANGED (his own mail tab
    ticking from "(2)" to "(3)" while the open landed nowhere visible) was
    recorded as the tab JARVIS opened, so "close that" would have closed it.
    An unknown title records nothing - "close that" then says it has no
    record instead of guessing. Never raises."""
    try:
        low = (page or "").lower()
        squashed = re.sub(r"[^a-z0-9]", "", low)
        if not squashed:
            return False
        parts = urllib.parse.urlsplit(url if "://" in url else "https://" + url)
        host = (parts.hostname or "").lower()
        for label in host.split("."):
            if (len(label) >= 3 and label not in _GENERIC_HOST_LABELS
                    and label in squashed):
                return True
        try:
            from core import streaming_search as _ss
            key = _ss.service_for_url(url)
            if key and re.sub(r"[^a-z0-9]", "",
                              _ss.service_name(key).lower()) in squashed:
                return True
        except Exception:
            pass
        query = " ".join(v for vals in urllib.parse.parse_qs(parts.query).values()
                         for v in vals)
        words = [w for w in re.findall(r"[a-z0-9']+", query.lower())
                 if len(w) >= 3]
        if words and sum(w in low for w in words) * 2 >= len(words):
            return True
    except Exception:
        return False
    return False


def _streaming_url_fix(url: str, bare_names: bool = True) -> "tuple[str, str]":
    """(url, note) through core.streaming_search.fix_search_url: a search
    link the brain GUESSED for a streaming service (live 2026-10-02:
    hbomax.com/search?q=... is a 404) becomes the verified one, or the
    service's home page when it has none. ``bare_names=False`` leaves a bare
    name ("netflix") alone - open_on_monitor launches that as an APP.
    Never raises."""
    try:
        from core import streaming_search as _ss
        if not bare_names and "." not in url and "/" not in url:
            return url, ""
        fix = _ss.fix_search_url(url)
        if fix.note:
            print(f"  [streaming-url] {url!r} -> {fix.url!r}", flush=True)
        return fix.url, fix.note
    except Exception:
        return url, ""


def _act_open_url(url: str) -> str:
    url = _site_shortcut_url(url) or url
    url, _fix_note = _streaming_url_fix(url)
    if not (url.startswith(("http://", "https://")) or _is_data_dir_page(url)):
        url = "https://" + url
    _before = _foreground_snapshot()
    webbrowser.open(url)
    # Wait (bounded, up to _OPEN_WAIT_S) until the page is in front and
    # named, instead of a blind 3 s sleep; the result says where it opened
    # and what it is, so the next round needs no screenshot (2026-10-05).
    _found = _await_opened_page(url, _before)
    _note_browser_tab_open("open_url", url, _before,
                           page=_found[:3] if _found else None)
    return _opened_line(url, _found, _fix_note)


def _act_web_search(query: str) -> str:
    """Google search. Video-intent shortcut: if the query mentions
    YouTube/video/watch/etc., fetch the SERP programmatically, pull the
    first YouTube watch URL out, and open THAT directly. This kills the
    old loop where JARVIS would screenshot the Google results page and
    then guess at a video URL."""
    bc = _bc()
    q_lower = query.lower()
    video_intent = any(hint in q_lower for hint in bc._VIDEO_QUERY_HINTS)
    if video_intent:
        yt_url = bc._extract_youtube_url_from_search(query)
        if yt_url:
            _before = _foreground_snapshot()
            webbrowser.open(yt_url)
            _await_opened_page(yt_url, _before)
            return (
                f"opened {yt_url} (extracted from Google results for '{query}') — "
                f"video is now playing, no further action needed"
            )
        # Extraction failed (network, rate-limit, parse miss). Fall through.

    url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
    _before = _foreground_snapshot()
    webbrowser.open(url)
    _found = _await_opened_page(url, _before)
    _note_browser_tab_open("web_search", url, _before,
                           page=_found[:3] if _found else None)
    where = f" on the {_found[3]} monitor" if _found and _found[3] else ""
    results = _search_results_line(_found)
    if results:
        return (f"opened Google search for '{query}'{where} \u2014 top "
                f"results: {results}")
    return (f"opened Google search for '{query}'{where}"
            + ("" if _found else " (still loading)"))


def _act_youtube(query: str) -> str:
    """Search YouTube without auto-playing — for when the user wants
    results, not playback. For 'play X on YouTube', use the youtube
    action via the streaming auto-play pipeline (route through
    _act_play_streaming)."""
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    webbrowser.open(url)
    return f"searching YouTube for {query}"


def _act_get_time(_: str = "") -> str:
    # Include the real calendar date (month/day/year), not just the weekday, so
    # "what's the date" is grounded in the system clock instead of an LLM guess
    # (an ungrounded date freehands an off-by-one). Same single real-clock read.
    return time.strftime("current time is %I:%M %p on %A, %B %d, %Y")


# ─── Screenshot + media keys (single bobert_companion dep each) ────────

def _act_screenshot(_: str = "") -> str:
    bc = _bc()
    # Privacy gate: refuse before the PowerShell fallback below — that path
    # captures the screen directly and would otherwise bypass the blocklist
    # that take_screenshot() enforces.
    if bc.screenshot_privacy_block_reason():
        return bc.SCREENSHOT_PRIVACY_REFUSAL
    out_dir = os.path.join(os.path.dirname(os.path.abspath(bc.__file__)), "screenshots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime("screenshot_%Y%m%d_%H%M%S.png"))
    # Use the same mss-backed take_screenshot() that see_screen uses — avoids
    # the PIL ImageGrab.grab(all_screens=True) segfault on some multi-monitor
    # Windows display driver configs.
    png = bc.take_screenshot()   # primary monitor, max_dim=1568 (good for saving)
    if png is not None:
        try:
            with open(path, "wb") as f:
                f.write(png)
            return f"screenshot saved to {path}"
        except Exception as e:
            return f"screenshot capture ok but save failed: {e}"
    # Fallback: PowerShell (no extra deps required)
    if sys.platform == "win32":
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "Add-Type -AssemblyName System.Drawing; "
            "$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height; "
            "$g = [System.Drawing.Graphics]::FromImage($bmp); "
            "$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size); "
            f"$bmp.Save('{path}')"
        )
        # getattr (not subprocess.CREATE_NO_WINDOW directly): the flag is a
        # Windows-only attribute, and reading it survives a test that has left
        # `subprocess` mocked — never AttributeErrors, real 0x08000000 on Win.
        #
        # VERIFY BEFORE CLAIMING SUCCESS (2026-07-14 audit). This used to return
        # "screenshot saved" unconditionally: capture_output=True threw stderr
        # away, check= wasn't passed, and the CompletedProcess was discarded —
        # so a PowerShell error (assembly load failure, a locked/at-capacity
        # disk, an invalid path) reported success while no file existed, and any
        # downstream vision step then read a stale or missing image. Inspect the
        # exit code AND confirm the file actually landed.
        try:
            r = subprocess.run(
                ["powershell", "-Command", ps], capture_output=True, timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return (f"screenshot failed: PowerShell timed out after 60s; "
                    f"no file written to {path}")
        if r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0:
            return f"screenshot saved to {path}"
        _err = (r.stderr or b"")
        if isinstance(_err, bytes):
            _err = _err.decode(errors="replace")
        return (f"screenshot failed (powershell exit {r.returncode}): "
                f"{_err.strip()[:200] or 'no file was written to ' + path}")
    return "screenshot not supported (install Pillow + mss: pip install pillow mss)"


def _act_media_next(_: str = "") -> str:
    bc = _bc()
    return bc._media_key_with_focus(
        "nexttrack",
        "next track / skip forward button in the music player controls",
        "media next pressed",
    )


def _act_media_prev(_: str = "") -> str:
    bc = _bc()
    return bc._media_key_with_focus(
        "prevtrack",
        "previous track / skip back button in the music player controls",
        "media previous pressed",
    )


def _act_media_playpause(_: str = "") -> str:
    bc = _bc()
    return bc._media_key_with_focus(
        "playpause",
        "play or pause button in the music player controls",
        "media play/pause pressed",
    )


def _act_volume_up(_: str = "") -> str:
    bc = _bc()
    pag = bc._get_pyautogui()
    if pag:
        pag.press("volumeup")
        return "volume up"
    return "pyautogui unavailable"


def _act_volume_down(_: str = "") -> str:
    bc = _bc()
    pag = bc._get_pyautogui()
    if pag:
        pag.press("volumedown")
        return "volume down"
    return "pyautogui unavailable"


def _endpoint_volume():
    """The default render device's IAudioEndpointVolume (pycaw). Raises when
    pycaw / COM / an output device is unavailable — callers degrade.

    Shared by set_volume and the explicit mute/unmute actions (2026-10-01:
    lifted out of _act_set_volume so mute can READ the state, not toggle it)."""
    from pycaw.pycaw import AudioUtilities

    dev = AudioUtilities.GetSpeakers()
    # Modern pycaw returns an AudioDevice wrapper with a ready-made
    # EndpointVolume property (verified on-box 2026-07-10); older releases
    # return the raw COM device needing the Activate+cast dance.
    vol = getattr(dev, "EndpointVolume", None)
    if vol is None:
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import IAudioEndpointVolume

        iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        vol = cast(iface, POINTER(IAudioEndpointVolume))
    return vol


def _set_system_mute(want_muted: bool) -> str:
    """Put the system output INTO the requested mute state and say what
    happened. Idempotent: "mute" on an already-muted PC stays muted.

    2026-10-01 (B091): volume_mute used to press VK_VOLUME_MUTE, which is a
    TOGGLE — "JARVIS, mute" while Windows was already muted (the owner mutes
    it when family is home) turned the speakers back ON while the chain path
    still said "muted". pycaw's GetMute/SetMute set the state outright. Only
    when the endpoint can't be read do we fall back to the key, and then we
    say plainly that it may have toggled."""
    word = "muted" if want_muted else "unmuted"
    try:
        vol = _endpoint_volume()
        if bool(vol.GetMute()) == want_muted:
            return f"system audio was already {word}, sir"
        vol.SetMute(1 if want_muted else 0, None)
        return f"system audio {word}, sir"
    except Exception:
        pass
    pag = _bc()._get_pyautogui()
    if pag:
        pag.press("volumemute")
        # No failure-marker words here (core/failure_markers.py): a "failed"
        # result re-prompts the LLM, which could send volume_mute again and
        # toggle the key a second time (2026-10-01, actions-a review).
        return ("mute key pressed — the mute state wasn't readable, so it "
                "may have toggled the other way, sir")
    return "pyautogui unavailable"


def _act_volume_mute(_: str = "") -> str:
    return _set_system_mute(True)


def _act_volume_unmute(_: str = "") -> str:
    """'unmute' / 'turn the sound back on'. Added 2026-10-01 alongside the
    mute fix: unmuting used to work only because mute was a toggle."""
    return _set_system_mute(False)


def _act_set_volume(arg: str = "") -> str:
    """Set the MASTER system volume to an absolute percent (0-100).

    Added 2026-07-10: "set the volume to 30 percent" had NO matching action
    (only volume_up/down/mute existed), so the local model routed it to a
    single volume_down nudge. Accepts digits ("30", "30%") or spoken numbers
    ("thirty") via the monolith's _parse_spoken_number. Uses pycaw (already a
    JARVIS dependency — audio ducking uses it) on the default render device."""
    bc = _bc()
    raw = (arg or "").strip().rstrip("%").strip()
    n = None
    try:
        n = int(float(raw))
    except (TypeError, ValueError):
        try:
            n = bc._parse_spoken_number(raw)      # "thirty" → 30
        except Exception:
            n = None
    if n is None or not (0 <= n <= 100):
        return (f"couldn't parse a volume percent from {arg!r} — "
                "give a number from 0 to 100")
    try:
        vol = _endpoint_volume()
        vol.SetMasterVolumeLevelScalar(n / 100.0, None)
        return f"volume set to {n} percent, sir"
    except Exception as e:
        return f"couldn't set the volume ({type(e).__name__}: {e})"


# ─── Streaming auto-play (Phase 4B) ────────────────────────────────────
# Each is a one-liner delegating to bc._streaming_auto_play(service, q).

def _act_netflix(query: str) -> str:
    return _bc()._streaming_auto_play("netflix", query)


def _act_prime_video(query: str) -> str:
    return _bc()._streaming_auto_play("prime_video", query)


def _act_disney_plus(query: str) -> str:
    return _bc()._streaming_auto_play("disney_plus", query)


def _act_hulu(query: str) -> str:
    return _bc()._streaming_auto_play("hulu", query)


def _act_max(query: str) -> str:
    return _bc()._streaming_auto_play("max", query)


def _act_spotify(query: str) -> str:
    return _bc()._streaming_auto_play("spotify", query)


def _act_youtube_play(query: str) -> str:
    """Auto-play on YouTube: opens search, clicks the first real video."""
    return _bc()._streaming_auto_play("youtube", query)


# ─── HUD visibility toggles (Phase 4B) ─────────────────────────────────

def _act_hide_hud(_: str = "") -> str:
    """Hide the on-screen HUD without killing the subprocess. JARVIS can
    bring it back any time with show_hud."""
    _bc()._write_hud_state(visible=False)
    return "HUD hidden, sir. Say 'show HUD' when you want it back."


def _set_unified_hud_hidden(hidden: bool) -> bool:
    """Set the unified HUD's own ✕-button 'hidden' flag in its control file so
    its close-button hide and the voice show/hide commands stay in sync.

    Returns True when the latch actually reached disk, False when it did not.
    2026-08-20: it used to swallow every failure and return None, and
    _act_show_hud said "HUD restored, sir." on the strength of that None -- a
    spoken success nothing had verified. The swallow itself is deliberate and
    pinned (tests/test_actions_sec1.py
    ::test_replace_and_cleanup_both_fail_still_swallowed): this runs on the
    voice path and must never raise into it. What changed is that the caller
    can now TELL, and one log line names the failure instead of nothing at
    all. Unlike _write_hud_state's ~30 Hz callers this helper only runs on an
    explicit user command, so a log line here cannot spam."""
    try:
        import os as _os
        import json as _json
        ctrl = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "unified_hud_state.json",
        )
        data = {}
        if _os.path.exists(ctrl):
            try:
                with open(ctrl, "r", encoding="utf-8") as _f:
                    data = _json.load(_f) or {}
            except Exception:
                data = {}
        data["hidden"] = bool(hidden)
        # Unique per-write temp. The unified HUD subprocess
        # (hud/jarvis_unified_hud.py:_write_control) writes this SAME control
        # file; a fixed shared ".tmp" name let the two processes truncate each
        # other's half-written temp and race the os.replace, corrupting
        # unified_hud_state.json. A mkstemp temp keeps each writer's file whole
        # (last replace wins; the single bool re-syncs on the next interaction).
        import tempfile as _tf
        _fd, _tmp = _tf.mkstemp(dir=_os.path.dirname(ctrl) or ".",
                                prefix=".uhud_", suffix=".tmp")
        try:
            with _os.fdopen(_fd, "w", encoding="utf-8") as _f:
                _json.dump(data, _f)
            _os.replace(_tmp, ctrl)
        except Exception:
            try:
                if _os.path.exists(_tmp):
                    _os.remove(_tmp)
            except Exception:
                pass
            raise
        return True
    except Exception as e:
        print(f"  [hud] couldn't write the ✕ hide latch "
              f"(unified_hud_state.json): {type(e).__name__}: {e}")
        return False


def _hud_child_state(bc) -> str:
    """What bobert's ``_hud_process`` handle says about the HUD subprocess:
    ``"running"`` / ``"dead"`` / ``"never-launched"``.

    Deliberately asymmetric — it reports death only from POSITIVE evidence.
    ``Popen.poll()`` is None while the child lives and an int returncode once it
    has exited, so an int means dead and None means alive; anything else (a test
    double, an unexpected object) is NOT evidence of death and reads as running.
    Claiming a failure we did not verify is the same sin as claiming a success
    we did not verify. Uses poll(), never psutil.pid_exists — that returns True
    for both Windows dead states. Never raises."""
    try:
        proc = getattr(bc, "_hud_process", None)
    except Exception:
        return "running"
    if proc is None:
        # Boot-time launch failed, HUD_ENABLED was off at boot, or
        # _shutdown_hud() ran. Either way there is nothing on screen.
        return "never-launched"
    try:
        rc = proc.poll()
    except Exception:
        return "running"
    return "dead" if isinstance(rc, int) else "running"


def _restore_hud(bc) -> str:
    """Shared body of show_hud / toggle_hud's un-hide branch.

    2026-08-20 honest-failure fix. This used to write two latches and say "HUD
    restored, sir." unconditionally — including when HUD_ENABLED was off (both
    writes are no-ops then), when the latch write failed (swallowed, silent),
    and when the HUD subprocess was DEAD, which is precisely the state the owner
    is in when he says "turn on the HUD" (core/prompts.py documents that phrase
    as a show_hud trigger). Nothing polls the child, so the parent can believe a
    HUD is up long after a Qt crash, a missing PyQt6 (the HUD exits 2), or the
    owner killing the window.

    Now: relaunch when the handle says the child is gone, and report honestly
    when the HUD cannot actually be put back. The relaunch is only safe because
    bobert_companion._launch_hud() is idempotent as of the same date — without
    that guard this call would orphan a live HUD on the tray route, which
    already calls _act_show_hud and then _launch_hud itself."""
    if not getattr(bc, "HUD_ENABLED", True):
        return ("The HUD is switched off in Settings, sir — turn 'On-screen "
                "HUD' back on there and I'll bring it up.")
    bc._write_hud_state(visible=True)
    latched = _set_unified_hud_hidden(False)   # clear a ✕-button hide too
    if _hud_child_state(bc) != "running":
        try:
            bc._launch_hud()
        except Exception as e:
            print(f"  [hud] relaunch from show_hud failed: "
                  f"{type(e).__name__}: {e}")
    if _hud_child_state(bc) != "running":
        return ("The HUD process isn't running, sir, and I couldn't restart "
                "it — the tray's Open HUD is the other way in.")
    if not latched:
        return ("I brought the HUD back, sir, but I couldn't clear its close-"
                "button latch — it may hide itself again on the next tick.")
    return "HUD restored, sir."


def _act_show_hud(_: str = "") -> str:
    """Re-display the HUD after a previous hide_hud (or a ✕-button close)."""
    return _restore_hud(_bc())


def _act_toggle_hud(_: str = "") -> str:
    """Toggle HUD visibility — useful as a single voice command."""
    bc = _bc()
    try:
        with bc._hud_state_lock:
            currently_visible = bool(bc._hud_state_cache.get("visible", True))
    except Exception:
        currently_visible = True
    if currently_visible:
        bc._write_hud_state(visible=False)
        return "HUD hidden, sir."
    # Toggling back to visible must also clear a ✕-button hide, exactly like
    # _act_show_hud — otherwise the persisted 'hidden' latch keeps the window
    # down and the toggle silently fails to bring it back. Delegating to
    # _restore_hud keeps ONE owner of the un-hide rule (this pair drifting apart
    # is what tests/monolith/test_monolith_tray_contract.py exists to stop) and
    # inherits the deliverability check.
    return _restore_hud(bc)


# ─── Self-diagnostic probes (Phase 4B) ─────────────────────────────────

def _act_test_mic(_: str = "") -> str:
    return _bc()._probe_via_selfdiag("mic",    "_probe_microphone")


def _act_test_tts(_: str = "") -> str:
    return _bc()._probe_via_selfdiag("tts",    "_probe_tts")


def _act_test_vision(_: str = "") -> str:
    return _bc()._probe_via_selfdiag("vision", "_probe_webcam")


# ─── Task queue + restart + session resume (Phase 4C) ─────────────────

def _act_clear_tasks(_: str = "") -> str:
    """Wipe the task queue (after the user confirms via the safety system).

    Unlike a silent delete, snapshot the current queue to backups/ first —
    mirrors _act_reset_memory so a cleared queue stays recoverable. The
    backup timestamp is derived from the file's own mtime (no live-clock
    dependency); if the backup fails we refuse to wipe."""
    bc = _bc()
    if not os.path.exists(bc.TODO_FILE):
        return "no task file to clear"
    todo_dir = os.path.dirname(os.path.abspath(bc.TODO_FILE)) or "."
    backup_dir = os.path.join(todo_dir, "backups")
    try:
        os.makedirs(backup_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(bc.TODO_FILE)))
        backup_path = os.path.join(backup_dir, f"jarvis_todo_{ts}.md")
        # Avoid clobbering an existing snapshot from the same mtime second.
        if os.path.exists(backup_path):
            n = 1
            while os.path.exists(os.path.join(backup_dir, f"jarvis_todo_{ts}_{n}.md")):
                n += 1
            backup_path = os.path.join(backup_dir, f"jarvis_todo_{ts}_{n}.md")
        shutil.copy2(bc.TODO_FILE, backup_path)
    except Exception as e:
        return f"backup failed, refused to clear: {e}"
    os.remove(bc.TODO_FILE)
    return f"task queue cleared (backup -> backups/{os.path.basename(backup_path)})"


def _act_session_resume(_: str = "") -> str:
    """Verbal trigger: 'Where did we leave off?' / 'Pick up where we left off'.
    Always returns a JARVIS-voice reply — falls back to a candid 'no
    recollection' note when the previous session is stale or missing."""
    text, _details = _bc()._build_session_resume(force=True)
    if not text:
        return ("I'm afraid I have no clear recollection of where we left "
                "off, sir.")
    return text


def _successor_env(bc):
    """The environment for a self-restart's successor: ours, plus the owner's
    REAL power plan (2026-09-29).

    The restart path hard-exits without restoring the power plan (it must not
    block before the successor exists - see _act_restart), so the successor
    boots with High Performance already active and used to record THAT as the
    plan to restore: a self-restart left the PC on High Performance
    (verified live). Handing the plan down lets
    bobert_companion._activate_high_performance_plan adopt it. None (inherit
    unchanged) when there is nothing to hand down. NEVER raises."""
    try:
        prior = getattr(bc, "_prior_power_plan_guid", None)
        hp = str(getattr(bc, "_HIGH_PERF_GUID", "") or "")
        var = getattr(bc, "_PRIOR_POWER_PLAN_ENV", "JARVIS_PRIOR_POWER_PLAN")
        if not prior or str(prior).lower() == hp.lower():
            return None
        env = dict(os.environ)
        env[var] = str(prior)
        return env
    except Exception:
        return None


def _act_restart(_: str = "") -> str:
    """Relaunch bobert_companion.py in a fresh process and exit this one."""
    import threading
    bc = _bc()
    # No processing-filler clip may start once a restart is decided (the
    # 1.5 s spawn delay and the native release below would otherwise race a
    # pending 'still working' stage). Latches it off. 2026-09-29.
    _filler_teardown_via_bc(bc, "restart")
    script = os.path.abspath(bc.__file__)

    def _do_restart():
        time.sleep(1.5)
        # ORDER IS LOAD-BEARING (rewritten 2026-07-14 after a live failure).
        #
        # The replacement MUST be spawned before anything that can block, and
        # the failsafe timer must be armed only AFTER it exists. Previously
        # this released the natives FIRST: voice_clone.unload() calls
        # torch.cuda.synchronize(), which waits for in-flight CUDA work — and
        # the "Restarting now, sir" line was STILL BEING SYNTHESISED on the
        # GPU (chatterbox, ~1000 sampling steps). The release outlasted the
        # 20s failsafe, the failsafe hard-exited the process, and the Popen
        # below NEVER RAN: old dead, no replacement, JARVIS simply gone
        # (live 2026-07-14 10:49). The natives still get released — just
        # after the successor is already booting, which is safe: it needs
        # ~60s of skill/model loading before it touches the mic or cameras.
        #
        # 1) free the singleton + port so the successor can boot and bind
        try:
            _stop_web_interface_quietly()
        except Exception as e:
            print(f"  [restart] web release failed: {e}")
        try:
            bc._release_singleton()
        except Exception as e:
            print(f"  [restart] singleton release failed: {e}")
        # 2) SPAWN THE SUCCESSOR — nothing above this line may block on a GPU
        spawned = False
        try:
            # Detached + WINDOWLESS relaunch. CREATE_NEW_CONSOLE (the old value)
            # forced a visible console window on the relaunched instance even
            # though JARVIS runs as pythonw (GUI, no console) — a "ghost window"
            # on every restart. DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP keeps
            # the new process alive after this one exits, with no window
            # (mirrors _ensure_ollama_running's detached spawn). 2026-07-10.
            _flags = 0
            if sys.platform == "win32":
                _flags = (subprocess.DETACHED_PROCESS
                          | subprocess.CREATE_NEW_PROCESS_GROUP)
            subprocess.Popen(
                [sys.executable, script],
                creationflags=_flags,
                close_fds=True,
                env=_successor_env(bc),
            )
            spawned = True
            print("  [restart] successor spawned; releasing native resources")
        except Exception as e:
            print(f"  [restart] relaunch failed: {e}")
        # 3) NOW arm the failsafe — the successor is already coming up, so a
        #    hard kill here can no longer orphan the machine. No clean flag:
        #    if the spawn failed, the watchdog SHOULD resurrect us.
        _fs = threading.Timer(45.0, _hard_exit_via_bc, args=(bc, 0, False))
        _fs.daemon = True
        _fs.start()
        # 4) release CUDA/Kinect/WASAPI/cameras so THIS process can actually
        #    die instead of corpse-ing with ~5GB of VRAM pinned (v2.0.57 —
        #    proven live: exit released 4.5GB and left no corpse).
        _release_native_resources(bc)
        time.sleep(2.0)                   # let camera caps + streams close
        if not spawned:
            print("  [restart] NO successor was spawned — exiting unclean so "
                  "the watchdog resurrects us")
        _hard_exit_via_bc(bc, 0, clean=False)
    threading.Thread(target=_do_restart, daemon=True).start()
    return "Restarting now, sir."


# ─── LLM picker stubs (Phase 4C) ───────────────────────────────────────

def _act_switch_llm_picker(_: str = "") -> str:
    """The tray menu's 'Other...' entry — list the model options with their
    estimated cost PER CONVERSATION so the user can pick how fast it burns
    credit, then switch via the AI submenu / 'switch to <model>'."""
    from core import model_catalog
    return (model_catalog.format_catalog()
            + "\nSwitch via the AI submenu, or say 'switch to haiku / sonnet / "
              "opus / local'.")


def _act_model_costs(_: str = "") -> str:
    """Report every model option + its estimated cost per conversation, so the
    user can choose how fast it burns through credit ('what does each model
    cost', 'how much does each model burn', 'model prices', 'compare models')."""
    from core import model_catalog
    return model_catalog.format_catalog()


def _act_running_costs(_: str = "") -> str:
    """What it costs to RUN JARVIS ('how much does it cost to run you', 'what
    do you cost', 'running costs'): an electricity estimate from the live GPU
    + CPU draw over the hours run today / this month, this session's Claude
    spend, and a one-line verdict (core.running_costs). Not the account
    balance; that is check_credits."""
    from core import running_costs
    return running_costs.report()


def _live_backend_and_model() -> tuple[str, str]:
    """The backend and model JARVIS is ACTUALLY using this second.

    Both reporting actions used to answer this by importing AI_BACKEND /
    OLLAMA_MODEL from core.config at call time, which is wrong twice over:

      * core.config holds the BOOT values. `switch_llm` mutates the monolith's
        wildcard-copied globals (`bc.AI_BACKEND`) and deliberately leaves
        core.config alone — _act_switch_llm's own docstring says so. So after a
        switch, both actions kept cheerfully reporting the old brain.
      * core.config.OLLAMA_MODEL is still the shipped default `"llama3"` — a tag
        that was RETIRED from this box. The live tag comes from the monolith's
        resolver. So on the local backend these didn't merely name the wrong
        model, they named a model that isn't installed.

    Read both off the live monolith, which is the authority. core.config is kept
    as a fallback for the one case where there IS no live value to read — the
    monolith isn't importable — because a status readout should degrade to the
    boot defaults, not raise in the user's face. 2026-07-14 audit."""
    from core import config as cfg
    try:
        bc = _bc()
    except Exception:
        return cfg.AI_BACKEND, (cfg.CLAUDE_MODEL if cfg.AI_BACKEND == "claude"
                                else cfg.OLLAMA_MODEL)
    backend = getattr(bc, "AI_BACKEND", None) or cfg.AI_BACKEND
    if backend == "claude":
        return backend, (getattr(bc, "CLAUDE_MODEL", None) or cfg.CLAUDE_MODEL)
    resolver = getattr(bc, "_get_local_llm_model", None)
    if callable(resolver):
        try:
            return backend, resolver()
        except Exception:
            pass
    return backend, (getattr(bc, "OLLAMA_MODEL", None) or cfg.OLLAMA_MODEL)


def _act_show_llm_stats(_: str = "") -> str:
    """The active backend + model, with an estimated cost per conversation for
    the current model (from core.model_catalog)."""
    from core import model_catalog
    backend, model = _live_backend_and_model()
    entry = model_catalog.by_id(model)
    if entry is not None:
        c = entry.cost_per_conversation()
        cost = "$0 (local)" if c <= 0 else f"~${c:.2f}/conv"
        return (f"backend={backend}  model={model}  est. {cost} ({entry.tier})."
                f" Say 'model costs' to compare the options.")
    return (f"backend={backend}  model={model}  "
            f"(not in the cost catalog — say 'model costs' for priced options).")


# ─── UI automation primitives (Phase 4C) ───────────────────────────────

def _act_press(key: str) -> str:
    bc = _bc()
    # Never Enter / Tab / Space on a sign-in page the owner did not ask for
    # (core.auth_guard, review 2026-10-05).
    refusal = _input_auth_refusal(bc, "press", key)
    if refusal:
        return refusal
    try:
        bc.ui_press(key.strip().lower())
    except bc.UIFailsafeError as e:
        return str(e)
    return f"pressed {key}"


def _act_scroll(args: str) -> str:
    bc = _bc()
    try:
        amount = int(args.strip())
    except ValueError:
        return "scroll amount must be an integer (positive = up, negative = down)"
    try:
        bc.ui_scroll(amount)
    except bc.UIFailsafeError as e:
        return str(e)
    return f"scrolled {amount}"


# ─── Skills directory listing (Phase 4C) ───────────────────────────────

def _act_list_skills(_: str = "") -> str:
    bc = _bc()
    if not os.path.isdir(bc.SKILLS_DIR):
        return "no skills directory yet"
    files = [f[:-3] for f in os.listdir(bc.SKILLS_DIR)
             if f.endswith(".py") and not f.startswith("_")]
    if not files:
        return "no skills installed yet"
    return f"installed skills: {', '.join(files)}"


# ─── Apple Music search-vs-playlist routing (Phase 4C) ─────────────────

def _act_apple_music(query: str) -> str:
    """Open Apple Music in browser and start playing `query` — finds the
    first match in search results, opens it, then clicks play/shuffle.
    Playlist requests ('X playlist', 'my X playlist', 'playlist:X',
    'playlist called X') route to Library > Playlists directly instead of
    search."""
    bc = _bc()
    is_playlist, name = bc._looks_like_playlist_request(query)
    if is_playlist and name:
        return bc._apple_music_play_playlist(name)
    return bc._streaming_auto_play("apple_music", query)


# ─── App launching (Phase 4D) ──────────────────────────────────────────

# Spoken names that mean "Apple Music". They open the web player (see
# _act_open_apple_music) rather than a doomed exe / startfile lookup.
_APPLE_MUSIC_LAUNCH_ALIASES = frozenset({
    "apple music", "apple music app", "music app", "applemusic",
    "the apple music app",
})

# The Apple Music web player, the owner's player. Same default and the same
# JARVIS_APPLE_MUSIC_URL override as tray.py's APPLE_MUSIC_WEB_URL (the tray's
# "Open Apple Music" item), so voice and tray open the same page. Not imported
# from tray.py (the tray is its own process and module); a test keeps the two
# equal.
_APPLE_MUSIC_WEB_URL = "https://music.apple.com/"


def _apple_music_web_url() -> str:
    return (os.environ.get("JARVIS_APPLE_MUSIC_URL", "").strip()
            or _APPLE_MUSIC_WEB_URL)


def _act_launch_app(name: str) -> str:
    # A named site shortcut ("open Accelo") is a page, not a program.
    shortcut = _site_shortcut_url(name)
    if shortcut:
        return _act_open_url(shortcut)
    bc = _bc()
    # 0) Apple Music special-case — the web player, like open_apple_music
    #    (2026-10-01, B089: this used to launch the Store app by AUMID, which
    #    the owner doesn't use). Must run before the generic resolution:
    #    os.startfile("apple music") fails.
    if re.sub(r"\s+", " ", (name or "").strip().lower()) in _APPLE_MUSIC_LAUNCH_ALIASES:
        return _act_open_apple_music()

    # 1) Known-app table for things shutil.which / os.startfile can't resolve
    #    (e.g. Bambu Studio, which installs to Program Files without a
    #    PATH-friendly name).
    known = bc._resolve_known_app(name)
    if known:
        try:
            subprocess.Popen([known], close_fds=True)
            return f"launched {name}"
        except Exception as e:
            return f"could not launch {name}: {e}"

    # 2) Try to find the executable on PATH; if not, hand it to the OS shell
    exe = shutil.which(name)
    try:
        if exe:
            subprocess.Popen([exe], close_fds=True)
        elif sys.platform == "win32":
            os.startfile(name)        # works for installed apps, shortcuts
        else:
            subprocess.Popen([name], close_fds=True)
        return f"launched {name}"
    except Exception as e:
        return f"could not launch {name}: {e}"


# ─── Music pause/resume/now-playing — media keys, COM is dead (Phase 4D) ─
#
# Classic iTunes is gone (iTunes.Application COM not registered, iTunes.exe
# absent), so the old _get_itunes() / app.Pause() / app.Play() / CurrentTrack
# paths are dead. Transport drives the Windows media session (SMTC) first —
# pause only pauses, resume only resumes, on the session actually playing
# (2026-10-01, B029). Only without SMTC does it fall back to OS-level MEDIA
# KEYS: the Apple Music web app (browser-active fast path) OR the new UWP
# Apple Music app (apple_music_app.is_active_media_app()). Only when NOTHING
# is playing/running do we return an honest line.

_NOTHING_PLAYING_MSG = (
    "Nothing seems to be playing, sir — open Apple Music and I'll take it "
    "from there."
)

# 2026-10-01 (B029): what each transport op says, keyed by the outcome of
# core.media_now_playing.transport(). {app} is the session's friendly name.
_TRANSPORT_REPLIES = {
    "pause": {"done": "paused {app}, sir",
              "already": "{app} is already paused, sir"},
    "play": {"done": "resumed {app}, sir",
             "already": "{app} is already playing, sir"},
    "next": {"done": "skipped to the next track in {app}, sir"},
    "prev": {"done": "went back a track in {app}, sir"},
}


def _web_player_titles(bc) -> tuple:
    """The live Apple Music web-player window titles (the monolith's
    _apple_music_web_player_titles), or () when unreadable. Never raises."""
    try:
        titles = bc._apple_music_web_player_titles()
    except Exception:
        return ()
    if not isinstance(titles, (list, tuple)):
        return ()
    return tuple(t for t in titles if isinstance(t, str) and t)


def _smtc_transport_reply(op: str) -> "str | None":
    """Drive pause/resume/next/previous through the Windows media session
    (SMTC) and return the spoken result, or None when SMTC is unavailable
    (CI / Linux / no winrt) so the caller falls back to its media-key path.

    2026-10-01 (B029): the media-key path is a blind TOGGLE sent to whatever
    session Windows calls current. With the Apple Music app merely running it
    toggled an HBO video in Chrome on "pause the music", STARTED a paused
    player on "pause", paused a playing one on "resume", and skipped a video
    on "next song". transport() picks the session by its real state and calls
    the idempotent TryPause / TryPlay / TrySkip* on it — and it never focuses
    a window just to send a key."""
    try:
        from core.media_now_playing import transport as _smtc_transport
        res = _smtc_transport(op, web_player_titles=_web_player_titles(_bc()))
    except Exception:
        return None
    if res is None:
        return None
    outcome, app = res
    app = app or "the media player"
    if outcome == "none":
        return _NOTHING_PLAYING_MSG
    if outcome == "not_music":
        # next/prev with only a browser VIDEO to skip (2026-10-01 review):
        # skipping it would jump the HBO / YouTube video to its next episode.
        return (f"What's playing in {app} doesn't look like music, sir, so I "
                f"left it alone.")
    if outcome == "failed":
        if app == "the media player":
            return "I couldn't reach the Windows media controls, sir."
        return f"{app} didn't accept that, sir."
    line = _TRANSPORT_REPLIES.get(op, {}).get(outcome)
    return line.format(app=app) if line else _NOTHING_PLAYING_MSG


def _act_pause_music(_: str = "") -> str:
    smtc = _smtc_transport_reply("pause")
    if smtc is not None:
        return smtc
    # No SMTC (winrt missing): the legacy media-key path below.
    bc = _bc()
    # Browser Apple Music → media key (fast path, unchanged).
    if bc._apple_music_chrome_active():
        return _act_media_playpause()
    # New UWP Apple Music app running → media key. pause/resume both map to
    # the single OS playpause toggle.
    amapp = _apple_music_app()
    if amapp is not None and amapp.is_active_media_app():
        return _act_media_playpause()
    return _NOTHING_PLAYING_MSG


def _act_resume_music(_: str = "") -> str:
    smtc = _smtc_transport_reply("play")
    if smtc is not None:
        return smtc
    bc = _bc()
    if bc._apple_music_chrome_active():
        return _act_media_playpause()
    amapp = _apple_music_app()
    if amapp is not None and amapp.is_active_media_app():
        return _act_media_playpause()
    return _NOTHING_PLAYING_MSG


def _act_now_playing(_: str = "") -> str:
    bc = _bc()
    # 0) The Windows media session (SMTC) — source-agnostic and reliable; names
    #    the real track from Chrome / Spotify / the Apple Music app / YouTube
    #    without scraping a window title. Falls through when nothing is playing.
    try:
        from core.media_now_playing import get_now_playing as _smtc_get
        _snap = _smtc_get()
    except Exception:
        _snap = None
    if _snap and _snap.get("title"):
        _t = _snap["title"]
        _a = _snap.get("artist") or ""
        _src = _snap.get("app") or "your media player"
        _verb = "playing" if _snap.get("playing") else "paused"
        if _a:
            return f"{_t} by {_a} — {_verb} in {_src}, sir."
        return f"{_t} — {_verb} in {_src}, sir."
    # 1) New UWP Apple Music app — best-effort read of its window title.
    amapp = _apple_music_app()
    if amapp is not None:
        try:
            np = amapp.now_playing()
        except Exception:
            np = None
        if np:
            return f"Apple Music: {np}"
    # 2) Browser Apple Music tab — the tab title carries the song while a
    #    track is playing ("<Song> — <Artist>"). Parse the REAL track out of
    #    it rather than echoing the raw window title (which is a bare
    #    "Apple Music" when idle, giving the useless "Apple Music: Apple
    #    Music"). _apple_music_title_now_playing returns None when no track
    #    title is present, so we fall through to an honest line.
    if bc._apple_music_chrome_active():
        try:
            track = bc._apple_music_title_now_playing()
        except Exception:
            track = None
        if track:
            return f"Apple Music: {track}"
        # The modern web player keeps a PAGE title ("<Song> - Song by <Artist>
        # - Apple Music") even while playing, which the strict confirm parser
        # rejects. Fall back to parsing that page title so we can still name
        # the loaded/current track instead of claiming nothing is playing.
        loaded = None
        try:
            loaded = bc._apple_music_loaded_track_from_title()
        except Exception:
            loaded = None
        if loaded:
            return f"Apple Music: {loaded}"
        return ("Apple Music is the active player, sir, but nothing seems to "
                "be playing right now — start a song and I'll read it back.")
    # 3) The app is running but its title gave us nothing useful.
    if amapp is not None and amapp.is_active_media_app():
        return ("Apple Music is open, sir, but it isn't telling me the track "
                "name right now.")
    return _NOTHING_PLAYING_MSG


# ─── Open / status for Apple Music (web player first) ──────────────────────

def _act_open_apple_music(_: str = "") -> str:
    """Open the Apple Music WEB PLAYER in a real browser. 'open Apple Music'.

    2026-10-01 (B089): this launched the Microsoft-Store app by AUMID. The
    owner listens in Chrome and had the Store app's autostart/keep-open turned
    off; the tray's "Open Apple Music" was moved to the web player in v2.0.144
    and this voice copy was left behind (a stale duplicate). Goes through the
    monolith's _open_url_in_browser, NOT webbrowser.open: the Store app
    registers itself as the music.apple.com handler, so the default handler
    can open the app instead of the page."""
    bc = _bc()
    # Already open in a browser window right now: don't stack another tab or
    # window on it (each one muddies the later media-session / title reads).
    # The LIVE scan only: the 5-minute sighting cache would refuse to reopen
    # a player the owner closed a minute ago. (2026-10-01, actions-a review.)
    if _web_player_titles(bc):
        return "Apple Music is already open in the browser, sir."
    url = _apple_music_web_url()
    try:
        how = bc._open_url_in_browser(url)
    except Exception as e:
        return f"could not open Apple Music in the browser: {e}"
    if how == "default":
        return ("no Chrome or Edge found, sir, so I handed Apple Music to the "
                "default link handler — that may have opened the app instead.")
    return "opened Apple Music in the browser, sir"


def _act_music_status(_: str = "") -> str:
    """Is Apple Music open, and what is playing? 'is Apple Music open' /
    'music status'.

    2026-10-01 (B089): this read ONLY the Store app — "doesn't appear to be
    running" while the web player played, and "running, now playing <some
    Chrome tab's title>" when the app was open. Sources now go in the owner's
    order: the Windows media session (what is really playing, any player),
    the browser web player, and only then the Store app.

    2026-10-01 (actions-a review): the media session is only called Apple
    Music's when it IS the web player (core.media_now_playing's
    is_web_player_session) or the Store app — an HBO video in the same
    Chrome is named as what it is, never as Apple Music's track. And "open
    in the browser" means a live web-player window; a sighting from the
    5-minute cache is reported as just that."""
    bc = _bc()
    try:
        from core.media_now_playing import get_now_playing as _smtc_get
        from core.media_now_playing import is_web_player_session as _is_web
        snap = _smtc_get()
    except Exception:
        snap, _is_web = None, None
    playing = None
    if snap and snap.get("title"):
        what = snap["title"]
        if snap.get("artist"):
            what = f"{what} by {snap['artist']}"
        verb = "playing" if snap.get("playing") else "paused"
        playing = f"{what} is {verb} in {snap.get('app') or 'your media player'}"

    live = _web_player_titles(bc)
    try:
        snap_web = bool(playing and _is_web and _is_web(snap, live))
    except Exception:
        snap_web = False
    snap_app = bool(playing and snap.get("app") == "Apple Music")

    head = None
    if live:
        head = "Apple Music is open in the browser, sir"
    else:
        try:
            recent = bool(bc._apple_music_chrome_active())
        except Exception:
            recent = False
        if recent:
            head = "I saw Apple Music in the browser a few minutes ago, sir"
    if head:
        if snap_web:
            return f"{head} — {playing}."
        for reader in ("_apple_music_title_now_playing",
                       "_apple_music_loaded_track_from_title"):
            try:
                track = getattr(bc, reader)()
            except Exception:
                track = None
            if track:
                return f"{head} — now playing {track}."
        other = f" — {playing}" if playing else ""
        return f"{head}, but nothing is playing in it right now{other}."
    if snap_web:
        return f"{playing}, sir — that looks like the Apple Music web player."

    amapp = _apple_music_app()
    try:
        running = amapp is not None and bool(amapp.is_running())
    except Exception:
        running = False
    if running:
        if snap_app:
            return f"The Apple Music app is running, sir — {playing}."
        try:
            np = amapp.now_playing()
        except Exception:
            np = None
        if np:
            return f"The Apple Music app is running, sir — now playing {np}."
        other = f" — {playing}" if playing else ""
        return ("The Apple Music app is running, sir, but nothing is playing "
                f"in it right now{other}.")
    if playing:
        return f"Apple Music isn't open, sir — but {playing}."
    return ("Apple Music isn't open, sir — say 'open Apple Music' and I'll "
            "open the web player.")


# ─── Task queue add (Phase 4D) ─────────────────────────────────────────

def _act_queue_task(args: str) -> str:
    """Append a task to jarvis_todo.md. The user can later hand this file
    (or specific entries) to Claude Code as a worklist."""
    bc = _bc()
    if not args.strip():
        return "format: queue_task, <description of the task>"
    ts = time.strftime("%Y-%m-%d %H:%M")
    entry = f"- [ ] **{ts}** — {args.strip()}\n"
    if not os.path.exists(bc.TODO_FILE):
        with open(bc.TODO_FILE, "w", encoding="utf-8") as f:
            f.write(
                "# JARVIS Task Queue\n\n"
                "Things the user wants Claude Code to build, fix, or investigate later.\n"
                "Tick items as you complete them; archive when the file gets big.\n\n"
            )
    with open(bc.TODO_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
    return f"queued: {args.strip()[:80]}"


# ─── Window management (Phase 4D) ──────────────────────────────────────

def _act_list_windows(_: str = "") -> str:
    """Return the owner's open window titles: never JARVIS's own windows or
    the shell's (core.window_scope - live 2026-10-03 the listing handed the
    brain "JARVIS HUD", "JARVIS Reticle", "Program Manager" and "Windows
    Input Experience", and it minimized all four)."""
    try:
        import pygetwindow as gw
    except ImportError:
        return "pygetwindow not available — pip install pygetwindow"
    titles = sorted({w.title for w in _window_scope.user_windows(gw.getAllWindows())
                     if w.title and w.title.strip()})
    if not titles:
        return "no windows visible"
    return "Open windows:\n" + "\n".join(f"  - {t}" for t in titles)


def _act_focus_window(query: str) -> str:
    """Bring a window to the foreground by partial title match, else by the
    app its process is ("focus Claude" while its title is a chat's name).
    A name nothing answers to that is one open name misheard ("Claw" for
    Claude - _window_name_suggestion) focuses that one and says so: bringing
    a window forward harms nothing (2026-10-05)."""
    bc = _bc()
    if not query.strip():
        return "format: focus_window, <window title>"
    matches = bc._find_windows_by_title(query)
    if not matches:
        matches = _seam_list(bc, "_find_app_windows", query)
    heard_as = ""
    if not matches:
        sugg = _seam_str(bc, "_window_name_suggestion", query)
        if sugg:
            matches = (bc._find_windows_by_title(sugg)
                       or _seam_list(bc, "_find_app_windows", sugg))
            heard_as = f" (I heard '{query.strip()}')" if matches else ""
    if not matches:
        return f"no window matching '{query}'"
    target = matches[0]
    try:
        target.activate()
        bc._flash_window_reticle(target, "focus")
        return f"focused '{target.title}'{heard_as}"
    except Exception as e:
        # pygetwindow on Windows raises a generic exception even on success
        # (Win32 SetForegroundWindow returns false in some allowed cases).
        # If the error message is "operation completed successfully" it
        # actually worked. Restore + minimize-toggle as a defense-in-depth.
        msg = str(e).lower()
        if "operation completed successfully" in msg or "error code from windows: 0" in msg:
            bc._flash_window_reticle(target, "focus")
            return f"focused '{target.title}'{heard_as}"
        # Try the restore trick as a fallback (works around some flag-set quirks)
        try:
            target.minimize()
            target.restore()
            bc._flash_window_reticle(target, "focus")
            return f"focused '{target.title}' (via restore){heard_as}"
        except Exception:
            pass
        return f"could not focus '{target.title}': {e}"


def _act_minimize_window(query: str) -> str:
    """Minimize a window by partial title match."""
    bc = _bc()
    if not query.strip():
        return "format: minimize_window, <window title>"
    matches = bc._find_windows_by_title(query)
    if not matches:
        return f"no window matching '{query}'"
    done = []
    for w in matches:
        try:
            w.minimize()
            done.append(w.title)
        except Exception:
            pass
    return f"minimized: {', '.join(done)}" if done else "could not minimize"


# ── close_window targets the owner can't see in a title (2026-10-02) ───────
# Live 12:00:18-12:01:04 the model asked close_window for "taskmgr.exe" (the
# PROCESS name) and for "Task Manager | left" (the move_window_to_monitor
# format), and both were "no window matching". A query that names a process
# resolves to that process's windows; a trailing "| <monitor>" names the
# monitor, which narrows several matches.
_EXE_QUERY_RE = re.compile(r"^[\w .()&+-]+\.exe$", re.IGNORECASE)
# PROCESS_QUERY_LIMITED_INFORMATION / TOKEN_QUERY / TokenElevation.
_PQLI, _TOKEN_QUERY, _TOKEN_ELEVATION = 0x1000, 0x0008, 20


def _window_pid(w) -> "int | None":
    """The process id owning pygetwindow window ``w``, or None (no native
    handle, not Windows, any fault)."""
    hwnd = getattr(w, "_hWnd", None)
    if not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        pid = wintypes.DWORD(0)
        ctypes.windll.user32.GetWindowThreadProcessId(
            wintypes.HWND(hwnd), ctypes.byref(pid))
        return int(pid.value) or None
    except Exception:
        return None


def _window_process_name(w) -> "str | None":
    """The executable name ("Taskmgr.exe") of the process owning ``w``, or
    None when it can't be read."""
    pid = _window_pid(w)
    if not pid:
        return None
    try:
        import psutil
        return psutil.Process(pid).name()
    except Exception:
        return None


def _process_token_elevated(pid: int) -> "bool | None":
    """True / False when the process's token says it is / isn't elevated,
    None when it can't be read. Query-only handles, always closed."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        adv = ctypes.windll.advapi32
        hproc = k32.OpenProcess(_PQLI, False, int(pid))
        if not hproc:
            return None
        try:
            htok = wintypes.HANDLE()
            if not adv.OpenProcessToken(hproc, _TOKEN_QUERY,
                                        ctypes.byref(htok)):
                return None
            try:
                elevated = wintypes.DWORD(0)
                size = wintypes.DWORD(0)
                if not adv.GetTokenInformation(
                        htok, _TOKEN_ELEVATION, ctypes.byref(elevated),
                        ctypes.sizeof(elevated), ctypes.byref(size)):
                    return None
                return bool(elevated.value)
            finally:
                k32.CloseHandle(htok)
        finally:
            k32.CloseHandle(hproc)
    except Exception:
        return None


def _self_is_elevated() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _window_is_elevated(w) -> bool:
    """True when ``w`` belongs to an elevated ("run as administrator")
    process and JARVIS itself is not elevated - Windows' UIPI then refuses
    our WM_CLOSE. False when unknown."""
    pid = _window_pid(w)
    if not pid or _self_is_elevated():
        return False
    return _process_token_elevated(pid) is True


def _close_refused(err) -> bool:
    """True when a close attempt failed because Windows denied it
    (ERROR_ACCESS_DENIED: UIPI blocking a message to a higher-integrity
    window). pygetwindow raises "Error code from Windows: 5 - Access is
    denied."."""
    if isinstance(err, PermissionError):
        return True
    if getattr(err, "winerror", None) == 5:
        return True
    msg = str(err).lower()
    return "access is denied" in msg or "error code from windows: 5 " in msg


# The shell's own windows ("Program Manager" - the desktop, explorer.exe -
# "Windows Input Experience", ...). WM_CLOSE on the desktop opens Windows'
# "Shut Down Windows" dialog, so "close_window, explorer.exe" must reach File
# Explorer windows only (review 2026-10-02). ONE list: core.window_scope.
_SHELL_WINDOW_TITLES = _window_scope.SYSTEM_WINDOW_TITLES


def _windows_of_process(exe: str) -> list:
    """Titled windows whose process executable is ``exe`` (case-insensitive),
    among the owner's windows (core.window_scope.user_windows): never the
    shell's desktop window or one of JARVIS's own."""
    try:
        import pygetwindow as gw
        windows = _window_scope.user_windows(gw.getAllWindows())
    except Exception:
        return []
    want = exe.strip().lower()
    out = []
    for w in windows:
        title = (getattr(w, "title", "") or "").strip()
        if not title:
            continue
        name = _window_process_name(w)
        if name and name.strip().lower() == want:
            out.append(w)
    return out


def _seam_list(bc, name: str, *args) -> list:
    """``bc.<name>(*args)`` when it returns a list, else [] (a missing seam,
    a fault, or a test's Mock monolith)."""
    try:
        out = getattr(bc, name)(*args)
    except Exception:
        return []
    return out if isinstance(out, list) else []


def _seam_str(bc, name: str, *args) -> str:
    """``bc.<name>(*args)`` when it returns a str, else ""."""
    try:
        out = getattr(bc, name)(*args)
    except Exception:
        return ""
    return out.strip() if isinstance(out, str) else ""


# ── Names the owner says vs. names the windows carry (2026-10-05) ──────────
# Live 00:24:21-00:25:43 the owner named File Explorer and Google Chrome four
# times and the brain answered with close_last_opened (only what JARVIS
# itself opened); and Parakeet heard "Claude" as "Claw". close_window now
# also finds the app the owner NAMED by its process when no title carries the
# name, and a name no window answers to gets ONE "did you mean" candidate
# (core.name_suggest) from the names actually open. Every lookup goes through
# core.window_scope.user_windows: never JARVIS's own windows, never the
# shell's.

def _owner_windows() -> list:
    """The owner's open windows (core.window_scope.user_windows), [] when
    window control is unavailable. Never raises."""
    try:
        import pygetwindow as gw
        return _window_scope.user_windows(gw.getAllWindows())
    except Exception:
        return []


def _app_process_windows(windows, query) -> list:
    """The windows among ``windows`` whose PROCESS is the app ``query``
    names (_exe_names_app: "Claude" -> claude.exe, "File Explorer" ->
    explorer.exe). A packaged-app host process names no app. Never
    raises."""
    try:
        words = _app_words(_keep_key(query))
        if not words:
            return []
        out = []
        for w in list(windows or ()):
            if not (getattr(w, "title", "") or "").strip():
                continue
            proc = _window_process_name(w)
            if (proc and proc.strip().lower() not in _APP_HOST_PROCESSES
                    and _exe_names_app(proc, words)):
                out.append(w)
        return out
    except Exception:
        return []


def _find_app_windows(query) -> list:
    """The owner's windows whose process is the app ``query`` names - for a
    single close / focus by name when no window TITLE carries the name.
    Terminals included: whether a close by name may close one is
    _resolve_named_close's call (review 2026-10-05). Never raises."""
    try:
        return _app_process_windows(_owner_windows(), query)
    except Exception:
        return []


def _title_names_as_app(w, words) -> bool:
    """True when window ``w``'s title names ``words`` the way a window
    names ITSELF, every word whole: its last part ("Downloads - File
    Explorer", "notes.txt - Notepad", "Claude") or, in a browser window, its
    page's ("Home - YouTube - Google Chrome"). A word inside a document,
    folder or page text ("Doorbell camera" for "door") is not. Never
    raises."""
    try:
        parts = _TITLE_PART_SPLIT_RE.split(
            (getattr(w, "title", "") or "").strip())
        tails = [parts[-1]]
        if len(parts) > 1 and _is_browser_process(w):
            tails.append(parts[-2])
        for tail in tails:
            have = set(re.findall(r"[a-z0-9+#]+", tail.lower()))
            if words and all(t in have for t in words):
                return True
    except Exception:
        return False
    return False


def _names_open_window(name, bulk=None) -> bool:
    """True when a close BY NAME of ``name`` (_resolve_named_close) would
    close at least one window and every one of them is the owner's (never
    one of JARVIS's own): the bar for routing "close <name>" to close_window
    without the brain. The route and the action resolve the same windows, so
    "close Claude" routed is the Claude app alone - never the Claude Code
    terminal or a "Claude Code" folder (review 2026-10-05) - and "close the
    door" is never a "Doorbell" tab. ``bulk``: the bulk close a "you forgot
    X" follows (_resolve_after_bulk). Never raises."""
    try:
        res = _resolve_after_bulk(_bc(), name, bulk)
        if not res or not res[0]:
            return False
        return len(_window_scope.user_windows(res[0])) == len(res[0])
    except Exception:
        return False


def _named_close_state(name) -> str:
    """How an open window answers to ``name``: "named" (a close by name
    resolves it - _resolve_named_close - even if only to a terminal it
    leaves open), "loose" (only close_window's title-substring lookup or the
    app's process finds something: a word inside a title) or "none".
    bobert_companion._enforce_named_close rewrites the brain's
    close_last_opened only on "named" / "none" (review 2026-10-05: "close the
    song" closed "Song lyrics draft.txt - Notepad" when the rewrite ignored
    this). Never raises: "loose" on a fault, so nothing is rewritten."""
    try:
        bc = _bc()
        res = _resolve_named_close(bc, name)
        if res and (res[0] or res[1]):
            return "named"
        if (bc._find_windows_by_title(name)
                or _seam_list(bc, "_find_app_windows", name)):
            return "loose"
        return "none"
    except Exception:
        return "loose"


# Invisible marks in titles: core.window_scope's ONE list.
_NAME_INVISIBLE_RE = _window_scope._INVISIBLE_RE
_PATHLIKE_RE = re.compile(r"[\\/]|^[a-z]:", re.IGNORECASE)


def _window_name_candidates(windows) -> list:
    """The names the open windows answer to, as a person would say them:
    each part of each title ("Claude", "Google Chrome", "File Explorer"; a
    folder path is no name) and each window's app by its process
    ("Claude" for claude.exe). De-duplicated, in order. Never raises."""
    out: list = []
    seen: set = set()

    def _add(name):
        n = " ".join(str(name or "").split()).strip(_KEEP_QUOTES)
        n = re.sub(r"^administrator\s*:\s*", "", n, flags=re.IGNORECASE)
        if len(n) >= 3 and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)

    try:
        for w in list(windows or ()):
            title = _NAME_INVISIBLE_RE.sub(
                "", str(getattr(w, "title", "") or "")).replace("\xa0", " ")
            for part in _TITLE_PART_SPLIT_RE.split(title.strip()):
                p = part.strip()
                if p and len(p) <= 60 and not _PATHLIKE_RE.search(p):
                    _add(p)
            proc = _window_process_name(w)
            if proc and proc.strip().lower() not in _APP_HOST_PROCESSES:
                stem = re.sub(r"\.exe$", "", proc.strip(), flags=re.IGNORECASE)
                _add(stem.capitalize() if stem.islower() else stem)
    except Exception:
        return out
    return out


def _suggest_window_name(query, windows) -> str:
    """The one open window / app name ``query`` was most likely a
    mishearing of (core.name_suggest), "" when none. Never raises."""
    try:
        from core import name_suggest as _ns
        return _ns.suggest(_keep_key(query),
                           _window_name_candidates(windows)) or ""
    except Exception:
        return ""


def _window_name_suggestion(query) -> str:
    """"Claude" for a heard "Claw" while Claude is open: _suggest_window_name
    against the owner's open windows. "" when none. Never raises."""
    return _suggest_window_name(query, _owner_windows())


# ── Which windows a close BY NAME closes (review 2026-10-05) ───────────────
# "close Claude" was routed to close_window, whose title-SUBSTRING lookup
# also took the Claude Code terminal (WM_CLOSE ends what runs in it, unsaved)
# and a "Claude Code" folder - with no question, since that was under the
# "too many windows" bar. The brain, now taught close_window <name>, and the
# yes to a "did you mean" reached the same broad close. A query that is a
# NAME (a few words: not a "<document> - <app>" title, a path, a file or an
# exe) now closes the windows that ARE that name, in this order:
#   1. the app whose executable carries EVERY word of the name (claude.exe
#      for "Claude", Code.exe for "Code", chrome.exe for "Google Chrome") -
#      not the windows that merely mention it;
#   2. else the windows whose title names it as ITSELF (_title_names_as_app:
#      "Downloads - File Explorer", "Deck1 - PowerPoint", a browser page's
#      "Lo-fi - YouTube - Google Chrome");
#   3. else, when no title carries the name at all, the app by its process
#      the looser way (_exe_names_app: "File Explorer" -> explorer.exe; an
#      explorer.exe window that is not a folder window - a Properties sheet,
#      a copy dialog - is not File Explorer).
# A console / terminal window in that set is closed only when the owner named
# the terminal itself - its program ("PowerShell") or its whole title ("npm
# run dev"); one that a title word swept in is left open and named. JARVIS's
# own windows only when none of the owner's answers ("close settings" is
# Windows Settings, not JARVIS Settings). No window qualifies: close_window's
# title-substring match, exactly as before.
_FOLDER_WINDOW_CLASS = "cabinetwclass"
_FILE_NAME_RE = re.compile(r"\.[a-z0-9]{1,5}$", re.IGNORECASE)
_NAME_QUERY_MAX_WORDS = 4


def _name_query_words(query) -> list:
    """The app words of ``query`` when it is a NAME ("File Explorer", "the
    Claude app"), [] when it is a window title, a path, a file, an exe or
    longer than a name. Never raises."""
    try:
        q = " ".join(_NAME_INVISIBLE_RE.sub("", str(query or "")).split())
        q = _keep_key(q)
        if (not q or len(q.split()) > _NAME_QUERY_MAX_WORDS
                or _EXE_QUERY_RE.match(q) or _TITLE_PART_SPLIT_RE.search(q)
                or _PATHLIKE_RE.search(q) or _FILE_NAME_RE.search(q)):
            return []
        return _app_words(q)
    except Exception:
        return []


def _proc_is_named_app(w, words) -> bool:
    """True when window ``w``'s executable carries EVERY word of the name
    (claude.exe for "Claude"; not claude.exe for "Claude Code", whose words
    it does not all carry). A packaged-app host names no app. Never
    raises."""
    try:
        proc = _window_process_name(w)
        if not proc or proc.strip().lower() in _APP_HOST_PROCESSES:
            return False
        stem = _exe_stem(proc)
        own = [c for c in (_compact(t) for t in words or ()) if c]
        return bool(stem and own) and all(t in stem for t in own)
    except Exception:
        return False


def _folder_window_ok(w) -> bool:
    """False for an explorer.exe window whose class is known and is not a
    folder window ("CabinetWClass"): a Properties sheet or a copy dialog is
    not "File Explorer". True for every other window. Never raises."""
    try:
        proc = _window_process_name(w)
        if not proc or _exe_stem(proc) != "explorer":
            return True
        cls = _window_scope.probe(w).class_name
        return not cls or cls == _FOLDER_WINDOW_CLASS
    except Exception:
        return True


def _terminal_named(w, words, name) -> bool:
    """True when terminal window ``w`` is what the owner named: the terminal
    program itself (every word of the name in a terminal executable -
    "PowerShell", "Windows Terminal") or its whole title / title part ("npm
    run dev", "Command Prompt"). Never raises."""
    try:
        proc = _window_process_name(w)
        if (proc and _exe_stem(proc) in _TERMINAL_PROCESS_STEMS
                and _proc_is_named_app(w, words)):
            return True
        want = _compact(_keep_key(name))
        title = _NAME_INVISIBLE_RE.sub("", str(getattr(w, "title", "") or ""))
        title = re.sub(r"^\s*administrator\s*:\s*", "", title,
                       flags=re.IGNORECASE).strip()
        parts = _TITLE_PART_SPLIT_RE.split(title)
        return bool(want) and want in {_compact(title), _compact(parts[0]),
                                       _compact(parts[-1])}
    except Exception:
        return False


def _resolve_named_close(bc, title_q, exclude=frozenset()) -> "tuple | None":
    """(windows to close, terminals left open) for a close BY NAME of
    ``title_q`` - see the block comment above - or None when the query is
    not a name or no window answers to it as one (close_window then matches
    by title, as before). ``exclude``: window keys (_window_key) never to
    touch. Raises what bc._find_windows_by_title raises."""
    words = _name_query_words(title_q)
    if not words:
        return None
    by_title = [w for w in (bc._find_windows_by_title(title_q) or [])
                if _window_key(w) not in exclude]
    # The app by its process only when no title carries the name (the
    # 2026-10-05 fallback): a test that fakes the title lookup never reaches
    # a real desktop through this.
    by_proc = [] if by_title else [
        w for w in _seam_list(bc, "_find_app_windows", title_q)
        if _window_key(w) not in exclude]
    seen: set = set()
    cands = []
    for w in by_title + by_proc:
        if _window_key(w) not in seen:
            seen.add(_window_key(w))
            cands.append(w)
    pool = [w for w in cands if _proc_is_named_app(w, words)]
    if not pool:
        pool = [w for w in by_title if _title_names_as_app(w, words)]
    if not pool:
        pool = [w for w in by_proc if _folder_window_ok(w)]
    if not pool:
        return None
    mine = _window_scope.user_windows(pool)
    if mine:
        pool = mine
    targets, left = [], []
    for w in pool:
        if _is_terminal_window(w) and not _terminal_named(w, words, title_q):
            left.append(w)
        else:
            targets.append(w)
    return targets, left


def _spoken_title(w) -> str:
    """A window's title as it can be said: no leading marks ("✳ Claude
    Code" -> "Claude Code"), no "Administrator:"."""
    t = _NAME_INVISIBLE_RE.sub("", str(getattr(w, "title", "") or ""))
    t = re.sub(r"^\s*administrator\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^[^\w]+", "", t).strip()
    return t[:60].strip() or "That window"


def _terminal_left_line(left) -> str:
    """The TERMINAL line for terminals a close by name left open."""
    from core.failure_markers import TERMINAL_FAILURE_PREFIX
    labels: list = []
    for w in left:
        label = _spoken_title(w)
        if label not in labels:
            labels.append(label)
    one = len(labels) == 1
    return (TERMINAL_FAILURE_PREFIX
            + f"{_spoken_list(labels)} {'is a terminal' if one else 'are terminals'}"
            f", sir; closing {'it' if one else 'them'} would end whatever "
            f"runs in {'it' if one else 'them'}, so I've left "
            f"{'it' if one else 'them'} open.")


def _bulk_exclusions(bulk) -> list:
    """The window-key sets a close right after bulk close ``bulk`` must
    spare, strictest first: every window it kept or spared; then - when that
    leaves nothing - only the windows it kept by name and the ones it spared,
    so a browser window it kept ONLY for a page ("Google Chrome stays open
    for its Claude page") is what "you forgot Google Chrome" closes."""
    if bulk is None:
        return [frozenset()]
    out = [bulk.keys]
    if bulk.page_kept_keys:
        out.append(bulk.loose_keys)
    return out


def _resolve_after_bulk(bc, title_q, bulk=None) -> "tuple | None":
    """_resolve_named_close, sparing what bulk close ``bulk`` kept or spared
    (_bulk_exclusions, strictest first). Review 2026-10-05: "you forgot
    Chrome" closed the Chrome window kept for "YouTube", and a terminal the
    bulk close had said it left open. Raises what bc._find_windows_by_title
    raises."""
    res = None
    for exclude in _bulk_exclusions(bulk):
        res = _resolve_named_close(bc, title_q, exclude)
        if res and res[0]:
            return res
    return res


def _close_window_matches(bc, query) -> tuple:
    """(windows close_window(``query``) would close, terminals it would
    leave open, the "you forgot X" bulk-close record or None) - the ONE
    resolution behind close_window and its pushback count. Raises what
    bc._find_windows_by_title raises."""
    title_q, monitor = _split_close_query(query)
    bulk = _forgot_bulk_close()
    named = _resolve_after_bulk(bc, title_q, bulk)
    if named is not None:
        matches, left = named
    else:
        left = []
        matches = []
        for exclude in _bulk_exclusions(bulk):
            matches = [w for w in (bc._find_windows_by_title(title_q) or [])
                       if _window_key(w) not in exclude]
            if not matches and _EXE_QUERY_RE.match(title_q):
                matches = [w for w in _windows_of_process(title_q)
                           if _window_key(w) not in exclude]
            if matches:
                break
    if monitor and len(matches) > 1:
        on_it = [w for w in matches if _on_monitor(w, monitor)]
        matches = on_it or matches
    return matches, left, bulk


def _close_window_preview(arg) -> list:
    """Titles of the windows close_window(``arg``) would close now - the
    pushback's "that will close N windows" count
    (bobert_companion._jarvis_pushback), so the number asked about is the
    number closed, process matches included. [] on any fault."""
    try:
        bc = _bc()
        try:
            forbidden = [str(t).lower() for t in bc.FORBIDDEN_TARGETS if t]
        except Exception:
            forbidden = []
        matches, _left, _bulk = _close_window_matches(bc, str(arg or ""))
        out = []
        for w in matches:
            title = getattr(w, "title", "") or ""
            if not any(t in title.lower() for t in forbidden):
                out.append(title)
        return out
    except Exception:
        return []


def _split_close_query(query: str) -> "tuple[str, str | None]":
    """(title query, monitor key) from "<title> | <monitor>" (either order);
    the whole query and None when the pipe does not name a monitor."""
    if "|" in query:
        a, b = (s.strip() for s in query.split("|", 1))
        for title, mon in ((a, b), (b, a)):
            key = _resolve_monitor(mon)
            if key is not None and title:
                return title, key
    return query.strip(), None


def _on_monitor(w, key: str) -> bool:
    """True when the centre of window ``w`` lies on monitor ``key`` (the one
    rule, core.monitor_geometry.monitor_for_rect)."""
    try:
        from core.config import MONITORS
        from core.monitor_geometry import monitor_for_rect
        return key in MONITORS and monitor_for_rect(
            w.left, w.top, w.width, w.height, MONITORS) == key
    except Exception:
        return False


def _app_label(title: str) -> str:
    """A short spoken name for a window: "Administrator: Command Prompt" ->
    "Command Prompt", "Report - Some App" -> "Some App"."""
    t = re.sub(r"^\s*administrator\s*:\s*", "", str(title or ""),
               flags=re.IGNORECASE).strip()
    if " - " in t:
        t = t.rsplit(" - ", 1)[-1].strip() or t
    return t[:48].strip() or "That window"


def _elevated_close_line(denied: list, n_closed: int) -> str:
    """The terminal result for windows Windows would not let JARVIS close."""
    from core.failure_markers import TERMINAL_FAILURE_PREFIX
    name = _app_label(denied[0])
    if n_closed:
        other = ("the other window" if n_closed == 1
                 else f"the other {n_closed} windows")
        line = (f"I closed {other}, sir, but {name} runs as administrator; "
                "Windows won't let me close it from here.")
    else:
        line = (f"{name} runs as administrator, sir; Windows won't let me "
                "close it from here.")
    return TERMINAL_FAILURE_PREFIX + line


def _send_close(w) -> str:
    """THE close step for one window: pygetwindow's close(), which posts
    WM_CLOSE - exactly the window's X button, so an app can ask to save
    first. Never a kill. "closed", "denied" (Windows refused: an elevated
    window, 2026-10-02 live - Task Manager) or "error"."""
    try:
        w.close()
        return "closed"
    except Exception as e:
        # Checked only after a failed close, so a close that works never
        # queries the process.
        if _close_refused(e) or _window_is_elevated(w):
            return "denied"
        return "error"


def _act_close_window(query: str) -> str:
    """Close a window by partial title match. Refuses to close Bobert's host.
    A process name ("taskmgr.exe") and a "<title> | <monitor>" query resolve
    too; a window Windows won't let JARVIS close (elevated) gets one honest
    TERMINAL line (core.failure_markers) that ends the follow-up chain."""
    bc = _bc()
    if not query.strip():
        return "format: close_window, <window title>"
    # Self-preservation: refuse if title matches one of the forbidden targets
    if any(target in query.lower() for target in bc.FORBIDDEN_TARGETS):
        return (
            f"REFUSED: '{query}' looks like your own host process. "
            f"Closing it would kill the session. Ask the user to close it manually."
        )
    title_q, _monitor = _split_close_query(query)
    # A NAME closes the windows that ARE that name, never every title that
    # mentions it, and no terminal it merely swept in; a "you forgot X"
    # right after a bulk close never touches what that close kept or spared
    # (review 2026-10-05: _close_window_matches).
    matches, left, bulk = _close_window_matches(bc, query)
    if not matches and left:
        return _terminal_left_line(left)
    if not matches and bulk is not None:
        why = _forgot_spared_line(bc, title_q, bulk)
        if why:
            return why
    if not matches:
        sugg = _seam_str(bc, "_window_name_suggestion", title_q)
        if sugg:
            return (f"no window matching '{query}' - did you mean {sugg}? "
                    "Ask him; never close it on a guess")
        return f"no window matching '{query}'"
    closed = []
    tabs = []
    skipped = []
    browser_only = []
    denied = []
    for w in matches:
        # Defence in depth: also check the actual window title we found
        if any(target in (w.title or "").lower() for target in bc.FORBIDDEN_TARGETS):
            continue
        # A browser window's title is its ACTIVE TAB's title, so "close
        # YouTube" matched the owner's whole Chrome window and WM_CLOSE took
        # every tab in it (2026-10-01, B040 — the same data loss
        # skills/youtube_search.py fixed on 2026-07-14). When the query
        # matched the PAGE part of a browser title, close just that tab.
        # Only a query that matched the browser part itself ("close chrome")
        # still closes the window.
        page = _browser_page_title(bc, w.title)
        if page is not None and _query_names_page(bc, query, page):
            ok = _close_browser_tab(bc, w)
            (tabs if ok else skipped).append(w.title)
            continue
        # Matched ONLY through the browser's own suffix, and the query
        # doesn't name a browser: "close Google" matched " - Google Chrome"
        # and WM_CLOSEd every tab while YouTube was in front (2026-10-01,
        # actions-a review). Close nothing; ask.
        if page is not None and not _query_names_browser(query):
            browser_only.append(w.title)
            continue
        outcome = _send_close(w)
        if outcome == "closed":
            closed.append(w.title)
        elif outcome == "denied":
            denied.append(w.title)
    if denied:
        return _elevated_close_line(denied, len(closed) + len(tabs))
    parts = []
    if closed:
        parts.append(f"closed: {', '.join(closed)}")
    if tabs:
        parts.append("closed just the tab: " + ", ".join(tabs)
                     + " (the rest of the browser window is untouched)")
    if skipped:
        parts.append("couldn't bring " + ", ".join(repr(t) for t in skipped)
                     + " to the front, so I left that window open")
    if browser_only:
        # Worded as a failure on purpose: the follow-up loop then reports it
        # (and the prompt tells the LLM to ASK, not to retry with the
        # browser's name), instead of a silent "Done.".
        parts.append("didn't close " + ", ".join(repr(t) for t in browser_only)
                     + f" — '{query.strip()}' only matched the browser's own "
                     "name, and closing that window would close every tab in "
                     "it. Say 'close Chrome' (or the browser's name) for the "
                     "whole window, or name the tab")
    if parts and left:
        parts.append("left " + ", ".join(repr(_spoken_title(w)) for w in left)
                     + " open: a terminal - closing it would end whatever "
                     "runs in it")
    return "; ".join(parts) if parts else "could not close"


# Words that name a browser itself: only these close a WHOLE browser window
# when the query matched nothing but the window's browser suffix.
_BROWSER_NAME_WORDS = frozenset({
    "chrome", "edge", "firefox", "mozilla", "brave", "opera", "vivaldi",
    "chromium", "browser",
})


def _query_names_browser(query: str) -> bool:
    """True when ``query`` names a browser ("chrome", "close the browser"),
    so closing the whole browser window is what was asked for."""
    words = re.findall(r"[a-z]+", (query or "").lower())
    return any(w in _BROWSER_NAME_WORDS for w in words)


def _browser_page_title(bc, title: str) -> "str | None":
    """The page / tab part of a browser window title ("Home - YouTube" for
    "Home - YouTube - Google Chrome"), or None when the window isn't a
    browser. Reuses the monolith's single browser-suffix list."""
    t = bc._strip_bidi_and_nbsp(title or "")
    low = t.lower()
    for suf in bc._BROWSER_CHROME_SUFFIXES:
        if low.endswith(suf):
            return t[: len(t) - len(suf)].strip()
    return None


def _query_names_page(bc, query: str, page: str) -> bool:
    """True when ``query`` matched the tab (page) part of a browser title
    rather than only the browser's own name. A query that carries a browser
    suffix itself (an LLM copying "YouTube - Google Chrome" from
    list_windows) is judged by its page part too."""
    q = _browser_page_title(bc, query)
    q = (q if q is not None else bc._strip_bidi_and_nbsp(query or "")).strip().lower()
    return bool(q) and q in page.lower()


def _close_browser_tab(bc, w) -> bool:
    """Close the ACTIVE tab of browser window ``w``: focus it, confirm it
    really is the foreground window, then Ctrl+W. Returns False — closing
    NOTHING — when it can't be confirmed in front: Ctrl+W sent to whatever
    else has focus would close the wrong tab or document."""
    hwnd = getattr(w, "_hWnd", None)
    if hwnd is None:
        return False
    try:
        w.activate()
    except Exception as e:
        # pygetwindow raises even on success in some allowed cases (see
        # _act_focus_window); the foreground check below is the real test.
        msg = str(e).lower()
        if not ("operation completed successfully" in msg
                or "error code from windows: 0" in msg):
            return False
    time.sleep(0.25)
    try:
        fg = bc._read_focused_window()[0]
    except Exception:
        fg = None
    if fg != hwnd:
        return False
    if not bc._get_pyautogui():   # ui_hotkey would silently send nothing
        return False
    try:
        bc.ui_hotkey("ctrl", "w")
    except Exception:
        return False
    return True


# ─── "close / minimize all windows except X" (2026-10-03) ─────────────────
# Live 17:22 the owner asked to close every window but the Claude app: no
# action did that, so the brain ran list_windows and minimize_window six times -
# JARVIS's own HUD and Reticle and two shell windows among them - and closed
# nothing. ONE action does it now. Every one of the owner's windows
# (core.window_scope.user_windows: never JARVIS's own, never the shell's)
# except the ones he named goes through _send_close (WM_CLOSE, the X button:
# an app may ask to save; never a kill) or is minimized. One spoken summary
# ("Closed 4 windows, sir; kept Claude."); an elevated window is reported
# once with close_window's terminal line. Closing more than
# PUSHBACK_MAX_CLOSE_WINDOWS windows asks first
# (bobert_companion._jarvis_pushback, via _close_all_windows_except_preview).
# A bulk close never sends WM_CLOSE to a console / terminal window (that ends
# what runs in it unsaved - _is_terminal_window) or to a window that may host
# JARVIS (FORBIDDEN_TARGETS): those are left open and named.

# Several names to keep: "Claude, Spotify", "Claude and Spotify".
_KEEP_SPLIT_RE = re.compile(
    r"\s*(?:[,;|&+/]|\band\b|\bor\b|\bplus\b|\bas\s+well\s+as\b)\s*",
    re.IGNORECASE)
_KEEP_LEAD_RE = re.compile(
    r"^(?:(?:except|but|for|keep|leave|leaving)\s+)+", re.IGNORECASE)
_KEEP_TAIL_RE = re.compile(
    r"(?:\s+(?:open|alone|running|please))+$", re.IGNORECASE)
# Matching also drops the words that do not name the app ("the Claude app").
_KEEP_ARTICLE_RE = re.compile(r"^(?:(?:the|my|our)\s+)+", re.IGNORECASE)
_KEEP_KIND_RE = re.compile(
    r"(?:\s+(?:app|apps|application|program|window|windows))+$",
    re.IGNORECASE)
# "except this one": the window in front.
_KEEP_FRONT_RE = re.compile(
    r"^(?:this|that|(?:the\s+)?(?:current|active|focused|front|foreground)|"
    r"(?:(?:the\s+)?one|what)\s+i(?:'?m|\s+am)\s+(?:on|in|using|looking\s+at|"
    r"working\s+(?:in|on)))(?:\s+(?:one|window|app))?$", re.IGNORECASE)
_KEEP_FRONT = "\x00front"
_KEEP_QUOTES = " .!?\"'‘’“”"


def _keep_key(name) -> str:
    """What a keep name is matched by: "the Claude app" -> "Claude"."""
    if name == _KEEP_FRONT:
        return name
    key = _KEEP_KIND_RE.sub("", _KEEP_ARTICLE_RE.sub("", str(name or "")))
    return key.strip(_KEEP_QUOTES) or str(name or "")


def _keep_names(arg) -> list:
    """The names to keep from a close/minimize_all_windows_except argument,
    as said, in order, de-duplicated ("Claude and Spotify" -> ["Claude",
    "Spotify"]); _KEEP_FRONT for "this one" / "the current window"."""
    names: list = []
    keys: set = set()
    for piece in _KEEP_SPLIT_RE.split(str(arg or "")):
        p = " ".join(piece.strip(_KEEP_QUOTES).split())
        if not p:
            continue
        if _KEEP_FRONT_RE.match(p):
            if _KEEP_FRONT not in names:
                names.append(_KEEP_FRONT)
            continue
        p = _KEEP_TAIL_RE.sub("", _KEEP_LEAD_RE.sub("", p)).strip(_KEEP_QUOTES)
        key = _keep_key(p).lower()
        if len(key) >= 2 and key not in keys:
            keys.add(key)
            names.append(p)
    return names


def _keep_is_app_window(w, key) -> bool:
    """True when window ``w`` is the APP keep name ``key`` names: its process
    is that app's (_exe_names_app, the v2.0.176 process-name rule: "except
    Claude" keeps claude.exe whatever its windows are titled), its title ends
    with the app's name (_title_names_app: "Claude", "Claude Code", "Deck1 -
    PowerPoint"), or - a browser window - its page's does ("Claude - Google
    Chrome"). Never raises."""
    try:
        words = _app_words(key)
        if not words:
            return False
        proc = _window_process_name(w)
        if (proc and proc.strip().lower() not in _APP_HOST_PROCESSES
                and _exe_names_app(proc, words)):
            return True
        title = (getattr(w, "title", "") or "").strip()
        if _title_names_app(title, words):
            return True
        parts = _TITLE_PART_SPLIT_RE.split(title)
        return (len(parts) > 1 and _is_browser_process(w)
                and _title_names_app(parts[-2], words))
    except Exception:
        return False


def _keep_mentions_window(w, key) -> bool:
    """True when window ``w``'s title mentions keep name ``key`` anywhere (a
    document, a folder, a page)."""
    return str(key or "").lower() in (getattr(w, "title", "") or "").lower()


def _keep_is_app_process(w, key) -> bool:
    """True when window ``w``'s PROCESS is the app keep name ``key`` names
    (claude.exe for "Claude"). Never raises."""
    return bool(_app_process_windows([w], key))


def _kept_windows(scoped, name, front_hwnd) -> list:
    """The windows among ``scoped`` (the owner's) that keep name ``name``
    covers. "this one": the window in front. A name that is a RUNNING APP
    keeps that app's own windows, by process, and nothing else; a name no
    running app answers to keeps the windows whose title names it as an app
    (_keep_is_app_window: "Claude Code", "Claude - Google Chrome"), else
    every window whose title mentions it.

    Review 2026-10-03: a bare title-substring keep held on to the live
    turn's File Explorer window because its folder PATH mentioned "Claude".
    Live 2026-10-05 00:25:12: "close everything except for Claude" closed 5
    windows and kept Claude - and a Chrome window, because its tab was a
    Claude page; the owner had to say "you forgot Google Chrome". The Claude
    app (claude.exe) was running, so the app's own windows are what he
    meant."""
    return _kept_windows_how(scoped, name, front_hwnd)[0]


def _kept_windows_how(scoped, name, front_hwnd) -> tuple:
    """(_kept_windows' windows, how they were matched): "front", "process"
    (the running app's own windows), "title" (no app of that name runs; the
    title names it as an app or a browser page) or "mention" (anywhere in a
    title: a document, a folder)."""
    if name == _KEEP_FRONT:
        return ([w for w in scoped if front_hwnd is not None
                 and getattr(w, "_hWnd", None) == front_hwnd], "front")
    key = _keep_key(name)
    own = [w for w in scoped if _keep_is_app_process(w, key)]
    if own:
        return own, "process"
    app = [w for w in scoped if _keep_is_app_window(w, key)]
    if app:
        return app, "title"
    return [w for w in scoped if _keep_mentions_window(w, key)], "mention"


def _kept_by_title_only(w, key) -> bool:
    """True when window ``w`` is a BROWSER window that keep name ``key``
    kept only through its page ("Claude - Google Chrome" for "Claude" while
    no Claude app runs): the owner then hears which browser stayed open. A
    window whose own title names the app (POWERPNT.EXE's "Deck1 -
    PowerPoint", olk.exe's "Inbox - Outlook") IS that app - review
    2026-10-05: those got "I kept PowerPoint only because its title mentions
    PowerPoint" - and an unknown process says nothing. Never raises."""
    try:
        if not _is_browser_process(w):
            return False
        words = _app_words(key)
        return bool(words) and not _title_names_app(
            getattr(w, "title", "") or "", words)
    except Exception:
        return False


# Console and terminal windows. WM_CLOSE on one ends every program running in
# it with no chance to save - a kill, not the X-button close this action
# promises - and its title is whatever runs in it, so close_window's
# title-only host rule (FORBIDDEN_TARGETS) misses a terminal titled by a
# build, an ssh login or a coding session (review 2026-10-03). A bulk close
# leaves them open and names them; minimizing one is harmless.
_TERMINAL_WINDOW_CLASSES = frozenset({
    "consolewindowclass", "cascadia_hosting_window_class",
    "pseudoconsolewindow", "virtualconsoleclass", "mintty", "putty",
    "org.wezfurlong.wezterm",
})
_TERMINAL_PROCESS_STEMS = frozenset({
    "windowsterminal", "wt", "conhost", "openconsole", "cmd", "powershell",
    "pwsh", "wezterm", "weztermgui", "alacritty", "mintty", "conemu",
    "conemu64", "putty", "kitty", "hyper", "tabby", "warp",
})


def _is_terminal_window(w) -> bool:
    """True for a console / terminal window (its window class or its
    process). Never raises; False when unknown."""
    try:
        if _window_scope.probe(w).class_name in _TERMINAL_WINDOW_CLASSES:
            return True
    except Exception:
        pass
    try:
        proc = _window_process_name(w)
        return bool(proc) and _exe_stem(proc) in _TERMINAL_PROCESS_STEMS
    except Exception:
        return False


def _host_label(w) -> str:
    """A spoken name for a window a bulk close left open."""
    try:
        proc = _window_process_name(w)
        if proc and _exe_stem(proc) == "windowsterminal":
            return "Windows Terminal"
    except Exception:
        pass
    return _app_label(getattr(w, "title", "") or "")


def _is_browser_process(w) -> bool:
    """True when ``w``'s process is a web browser (_BROWSER_PROCESS_WORDS);
    False when unknown."""
    proc = _window_process_name(w)
    return bool(proc) and any(b in _exe_stem(proc)
                              for b in _BROWSER_PROCESS_WORDS)


def _spoken_list(items, last="and") -> str:
    items = [str(i) for i in items if str(i)]
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + f" {last} " + items[-1]


class _AllExceptPlan:
    """What close/minimize_all_windows_except would do right now."""

    def __init__(self, names):
        self.names = names            # keep names, as said
        self.targets: list = []       # windows to close / minimize
        self.hosts: list = []         # left alone: may host JARVIS (close)
        self.matched: set = set()     # keep names that matched a window
        self.labels: dict = {}        # keep name -> spoken label
        # keep name -> browser windows kept only by their page
        self.by_title: dict = {}
        self.kept: list = []          # every window kept by a name
        self.scoped: list = []        # the owner's windows the plan saw

    @property
    def front_missing(self) -> bool:
        return _KEEP_FRONT in self.names and _KEEP_FRONT not in self.matched

    @property
    def kept_label(self) -> str:
        return _spoken_list([self.labels.get(n, n) for n in self.names
                             if n in self.matched])

    @property
    def unmatched(self) -> list:
        return [n for n in self.names
                if n not in self.matched and n != _KEEP_FRONT]

    @property
    def unmatched_keys(self) -> list:
        """The unmatched names as apps ("Spotify", not "the Spotify app")."""
        return [_keep_key(n) for n in self.unmatched]

    @property
    def title_only_note(self) -> str:
        """"Google Chrome stays open for its Claude page." - one sentence per
        keep name that kept a browser window by its page alone
        (_kept_by_title_only); "" when none did."""
        notes = []
        for n in self.names:
            wins = self.by_title.get(n) or []
            if not wins:
                continue
            labels: list = []
            for w in wins:
                label = _host_label(w)
                if label not in labels:
                    labels.append(label)
            one = len(labels) == 1
            notes.append(f"{_spoken_list(labels)} "
                         f"{'stays' if one else 'stay'} open for "
                         f"{'its' if one else 'their'} {_keep_key(n)} "
                         f"{'page' if one else 'pages'}.")
        return " ".join(notes)


def _all_except_plan(bc, arg, closing: bool):
    """The _AllExceptPlan for ``arg``, or None when it names nothing to keep.
    Raises ImportError when pygetwindow is missing."""
    names = _keep_names(arg)
    if not names:
        return None
    import pygetwindow as gw
    every = list(gw.getAllWindows())
    plan = _AllExceptPlan(names)
    front = None
    if _KEEP_FRONT in names:
        try:
            front = bc._read_focused_window()[0]
        except Exception:
            front = None
    try:
        forbidden = [str(t).lower() for t in bc.FORBIDDEN_TARGETS if t]
    except Exception:
        forbidden = []
    scoped = _window_scope.user_windows(every)
    plan.scoped = scoped
    kept_by: dict = {}
    how_by: dict = {}
    for n in names:
        wins, how = _kept_windows_how(scoped, n, front)
        kept_by[n] = {id(w) for w in wins}
        how_by[n] = how
    for w in scoped:
        title = getattr(w, "title", "") or ""
        hit = [n for n in names if id(w) in kept_by[n]]
        if hit:
            plan.kept.append(w)
            for n in hit:
                plan.matched.add(n)
                if n == _KEEP_FRONT:
                    plan.labels[n] = _app_label(title)
            # A browser window kept only by its page ("Claude - Google
            # Chrome" while no Claude app runs) is reported (2026-10-05):
            # the owner then hears which browser stayed open. A document /
            # folder named in the keep, or an app's own title, is not.
            if all(how_by[n] == "title"
                   and _kept_by_title_only(w, _keep_key(n)) for n in hit):
                plan.by_title.setdefault(hit[0], []).append(w)
            continue
        if closing and (_is_terminal_window(w) or (
                any(t in title.lower() for t in forbidden)
                and not _is_browser_process(w))):
            if any(n != _KEEP_FRONT and _keep_is_app_window(w, _keep_key(n))
                   for n in names):
                # A terminal titled by the app named in the keep ("Claude
                # Code" for "except Claude") while the app itself runs: it
                # stays, as before, without a "may be running me" note.
                continue
            # A console / terminal window: WM_CLOSE would end what runs in
            # it unsaved. And close_window's own self-preservation rule: a
            # terminal / python / editor window may be the one running
            # JARVIS. (A browser tab that merely mentions "python" is not.)
            plan.hosts.append(w)
            continue
        if not closing and getattr(w, "isMinimized", False) is True:
            continue
        plan.targets.append(w)
    # "except the HUD": JARVIS's own windows are always kept, so a name that
    # names one of them is kept, not "not found".
    jarvis_titles = [getattr(w, "title", "") or "" for w in every
                     if _window_scope.is_jarvis_title(getattr(w, "title", ""))]
    for n in plan.unmatched:
        if any(_window_scope.names_jarvis_window(n, t) for t in jarvis_titles):
            plan.matched.add(n)
    return plan


def _close_all_windows_except_preview(arg) -> list:
    """Titles of the windows close_all_windows_except(``arg``) would close
    right now; [] when it would close nothing or refuse. The pushback count
    (bobert_companion._jarvis_pushback). Never raises."""
    try:
        plan = _all_except_plan(_bc(), arg, True)
    except Exception:
        return []
    if plan is None or not plan.matched or plan.front_missing:
        return []
    return [getattr(w, "title", "") or "" for w in plan.targets]


def _send_minimize(w) -> str:
    try:
        w.minimize()
        return "minimized"
    except Exception:
        return "error"


def _all_windows_except(arg: str, closing: bool) -> str:
    action = ("close_all_windows_except" if closing
              else "minimize_all_windows_except")
    verb = "close" if closing else "minimize"
    usage = f"format: {action}, <window(s) to keep>"
    if not str(arg or "").strip():
        return usage
    bc = _bc()
    try:
        plan = _all_except_plan(bc, arg, closing)
    except ImportError:
        return "pygetwindow not available — pip install pygetwindow"
    if plan is None:
        return usage
    from core.failure_markers import TERMINAL_FAILURE_PREFIX
    # Nothing to anchor on: closing "everything except <a window that isn't
    # there>" would close everything. Say so and close nothing; a terminal
    # line, so no follow-up round improvises its own closes.
    if plan.front_missing:
        return (TERMINAL_FAILURE_PREFIX + "I can't tell which window is in "
                f"front, sir, so I've {verb}d nothing.")
    if not plan.matched:
        # A misheard name ("Claw" for Claude, live 2026-10-05) gets the one
        # open name it most likely was - asked, never acted on: the bulk
        # close's pushback (_close_name_question) holds the corrected
        # command for a yes before this action ever runs.
        fix = _all_except_correction(plan)
        ask = f" Did you mean {fix[2]}?" if fix else ""
        return (TERMINAL_FAILURE_PREFIX + "I don't see a "
                f"{_spoken_list(plan.unmatched_keys, 'or')} window to keep, "
                f"sir, so I've {verb}d nothing.{ask}")
    done, denied, errors = [], [], []
    for w in plan.targets:
        outcome = _send_close(w) if closing else _send_minimize(w)
        title = getattr(w, "title", "") or ""
        if outcome in ("closed", "minimized"):
            done.append(title)
        elif outcome == "denied":
            denied.append(title)
        else:
            errors.append(title)
    kept = plan.kept_label
    n = len(done)
    if denied:
        # v2.0.176's terminal line, said once for every refused window.
        line = _elevated_close_line(denied, n) + f" I kept {kept}."
    elif n:
        line = (f"{'Closed' if closing else 'Minimized'} {n} "
                f"window{'' if n == 1 else 's'}, sir; kept {kept}.")
    else:
        line = f"Nothing else to {verb}, sir; kept {kept}."
    extra = []
    if plan.hosts:
        labels: list = []
        for w in plan.hosts:
            label = _host_label(w)
            if label not in labels:
                labels.append(label)
        extra.append(f"I left {_spoken_list(labels)} open; "
                     f"{'it' if len(labels) == 1 else 'they'} may be running "
                     "me.")
    if plan.unmatched:
        extra.append(
            f"I saw no {_spoken_list(plan.unmatched_keys, 'or')} window.")
    note = plan.title_only_note
    if note:
        extra.append(note)
    if closing:
        _note_bulk_close(plan, done)
    if errors:
        more = len(errors) - 1
        extra.append(f"{_app_label(errors[0])}"
                     + (f" and {more} other window{'' if more == 1 else 's'}"
                        if more else "")
                     + f" wouldn't {verb}.")
        if not denied:
            line = TERMINAL_FAILURE_PREFIX + line
    return " ".join([line] + extra)


def _all_except_correction(plan) -> "tuple | None":
    """(corrected names, heard, suggested) when at least one keep name that
    matched no window is one open name misheard (core.name_suggest against
    the windows the plan saw); else None. "Claw" -> (["Claude"], "Claw",
    "Claude"); "Excel, Claw" -> (["Excel", "Claude"], "Claw", "Claude"): the
    names that matched stay, and a name with no suggestion stays as said (it
    is reported as not found when the corrected close runs). Review
    2026-10-05: with one name matched and one misheard, the bulk close went
    ahead and closed the app the owner meant to keep. Never raises."""
    try:
        if plan is None or plan.front_missing:
            return None
        fixed, heard, said = [], [], []
        for n in plan.names:
            if n == _KEEP_FRONT:
                fixed.append("this one")
                continue
            if n in plan.matched:
                fixed.append(n)
                continue
            s = _suggest_window_name(n, plan.scoped)
            if s:
                fixed.append(s)
                heard.append(_keep_key(n))
                said.append(s)
            else:
                fixed.append(n)
        if not said:
            return None
        return fixed, _spoken_list(heard, "or"), _spoken_list(said, "and")
    except Exception:
        return None


def _close_all_windows_except_suggestion(arg) -> "tuple | None":
    """For a close_all_windows_except(``arg``) with a keep name that matches
    no window but is an open name misheard: (corrected arg, titles it would
    then close, heard, suggested); else None. The pushback asks "Did you mean
    Claude?" with it and holds the corrected command for a yes. Never
    raises."""
    try:
        bc = _bc()
        plan = _all_except_plan(bc, arg, True)
        fix = _all_except_correction(plan)
        if fix is None:
            return None
        fixed_arg = ", ".join(fix[0])
        again = _all_except_plan(bc, fixed_arg, True)
        if again is None or not again.matched or again.front_missing:
            return None
        return (fixed_arg,
                [getattr(w, "title", "") or "" for w in again.targets],
                fix[1], fix[2])
    except Exception:
        return None


def _close_name_question(name, arg) -> "tuple | None":
    """(question, reason, corrected arg) when a close names a window that is
    not open but one that IS open sounds like it - "close everything except
    Claw" while Claude is open (live 2026-10-05) - else None. A close on a
    guess would close the wrong window(s), so bobert_companion._jarvis_
    pushback asks it and queues the CORRECTED command: a yes runs it.

      * close_window: no window title carries the name, no running app is
        it (_find_app_windows), and _window_name_suggestion has one;
      * close_all_windows_except: a keep name that matches no window is a
        misheard open name (_close_all_windows_except_suggestion) - whether
        or not the other keep names matched. The question carries the
        count of the corrected close, so the yes needs no second one.
    A lookup that fails asks nothing. Never raises."""
    try:
        bc = _bc()
        raw = str(arg or "").strip()
        nm = str(name or "").strip().lower()
        if not raw:
            return None
        if nm == "close_window":
            title_q, _mon = _split_close_query(raw)
            if not title_q or _EXE_QUERY_RE.match(title_q):
                return None
            try:
                found = bc._find_windows_by_title(title_q)
            except Exception:
                return None
            if found or _seam_list(bc, "_find_app_windows", title_q):
                return None
            sugg = _seam_str(bc, "_window_name_suggestion", title_q)
            if not sugg:
                return None
            heard = _keep_key(title_q)
            return (f"I don't see a {heard} window, sir. Did you mean "
                    f"{sugg}?",
                    f"close_window: no {heard!r} window; did you mean "
                    f"{sugg!r}", sugg)
        if nm == "close_all_windows_except":
            # Asked BEFORE anything closes, also when other keep names
            # matched (review 2026-10-05): "except Excel and Claw" would
            # otherwise close Claude.
            fix = bc._close_all_windows_except_suggestion(raw)
            if not isinstance(fix, tuple) or len(fix) != 4:
                return None
            fixed_arg, would_close, heard, sugg = fix
            n = len(would_close)
            if n:
                tail = (f" Say yes and I'll close the other {n} "
                        f"window{'' if n == 1 else 's'}.")
            else:
                tail = " Nothing else is open to close."
            return (f"I don't see a {heard} window, sir. Did you mean "
                    f"{sugg}?{tail}",
                    f"close_all_windows_except: no {heard!r} window; did you "
                    f"mean {sugg!r} ({n} to close)", fixed_arg)
    except Exception:
        return None
    return None


# ── "You forgot X" right after a bulk close (2026-10-05) ──────────────────
# Live 00:25:27, right after "Closed 5 windows, sir; kept Claude." the owner
# said "you forgot Google Chrome". The brain read it as one more name to KEEP
# - close_all_windows_except(Claude, Chrome) - and answered "Nothing else to
# close, sir; kept Claude and Chrome": the opposite of what he meant. The
# last bulk close is remembered for a short while, so the monolith's utterance
# route (core.dispatcher.forgot_close_target) can turn "you forgot X" / "X is
# still open" into close_window X.
BULK_CLOSE_FOLLOWUP_S = 120.0


class _BulkClose:
    """One bulk close that ran: when, the keep names, the titles it closed,
    and the windows it KEPT and SPARED (left open on purpose - a terminal, a
    window that may host JARVIS), by _window_key. Review 2026-10-05: "you
    forgot Chrome" closed the Chrome window kept for "YouTube", and "you
    forgot <the dev-server terminal>" closed the terminal the bulk close had
    just said it left open."""
    __slots__ = ("at", "keep", "closed", "kept_keys", "spared_keys",
                 "page_kept_keys", "labels")

    def __init__(self, at, keep, closed, kept_keys=(), spared_keys=(),
                 labels=None, page_kept_keys=()):
        self.at = at
        self.keep = tuple(keep)
        self.closed = tuple(closed)
        self.kept_keys = frozenset(kept_keys)
        self.spared_keys = frozenset(spared_keys)
        # Browser windows kept ONLY for a page (the owner heard "Google
        # Chrome stays open for its Claude page").
        self.page_kept_keys = frozenset(page_kept_keys) & self.kept_keys
        self.labels = dict(labels or {})

    @property
    def keys(self) -> frozenset:
        """Every window the bulk close kept or spared."""
        return self.kept_keys | self.spared_keys

    @property
    def loose_keys(self) -> frozenset:
        """The same, less the browser windows kept only for a page."""
        return (self.kept_keys - self.page_kept_keys) | self.spared_keys


_LAST_BULK_CLOSE: list = [None]


def _note_bulk_close(plan, closed) -> None:
    """Remember a bulk close that ran: its keep names, what it closed, and
    which of the owner's windows it kept or spared."""
    try:
        kept = {_window_key(w) for w in plan.kept}
        targets = {_window_key(w) for w in plan.targets}
        spared = {_window_key(w) for w in plan.scoped
                  if _window_key(w) not in kept
                  and _window_key(w) not in targets}
        labels = {_window_key(w): _spoken_title(w)
                  for w in plan.scoped if _window_key(w) in kept | spared}
        page_kept = {_window_key(w) for wins in plan.by_title.values()
                     for w in wins}
        _LAST_BULK_CLOSE[0] = _BulkClose(
            time.monotonic(),
            [_keep_key(n) for n in plan.names if n != _KEEP_FRONT],
            list(closed or ()), kept, spared, labels, page_kept)
    except Exception:
        pass


def _last_bulk_close(max_age_s: float = BULK_CLOSE_FOLLOWUP_S):
    """The last bulk close when it ran no more than ``max_age_s`` ago, else
    None. Never raises."""
    try:
        rec = _LAST_BULK_CLOSE[0]
        if rec is None:
            return None
        age = time.monotonic() - rec.at
        return rec if 0 <= age <= max_age_s else None
    except Exception:
        return None


def _forgot_bulk_close():
    """The fresh bulk close (_last_bulk_close) when the owner's words this
    turn are "you forgot X" / "X is still open"
    (core.dispatcher.forgot_close_target), else None: a close in that turn
    must not touch what the bulk close kept or spared. Never raises."""
    try:
        bulk = _last_bulk_close()
        if bulk is None:
            return None
        said = _seam_str(_bc(), "_turn_user_text")
        if not said:
            return None
        from core.dispatcher import forgot_close_target
        return bulk if forgot_close_target(said) else None
    except Exception:
        return None


def _forgot_spared_line(bc, title_q, bulk) -> str:
    """The TERMINAL line when the only windows ``title_q`` names are ones the
    last bulk close kept or spared on purpose; "" otherwise. Never
    raises."""
    try:
        from core.failure_markers import TERMINAL_FAILURE_PREFIX
        hits = list(bc._find_windows_by_title(title_q) or [])
        if not hits:
            hits = _seam_list(bc, "_find_app_windows", title_q)
        hits = [w for w in hits if _window_key(w) in bulk.keys]
        if not hits:
            return ""
        key = _window_key(hits[0])
        label = bulk.labels.get(key) or _spoken_title(hits[0])
        if key in bulk.kept_keys:
            return (TERMINAL_FAILURE_PREFIX + f"You asked me to keep {label} "
                    "open, sir, so I've left it.")
        return (TERMINAL_FAILURE_PREFIX + f"I left {label} open on purpose, "
                "sir: closing it could end whatever runs in it - me "
                "included.")
    except Exception:
        return ""


def _act_close_all_windows_except(arg: str) -> str:
    """close_all_windows_except, <names to keep>: close every one of the
    owner's windows except the named ones (an open app's windows by process
    or title, else any title that mentions the name). JARVIS's own windows
    and the shell's are always kept; console / terminal windows and windows
    that may host JARVIS are left open and named. WM_CLOSE only, never a
    kill."""
    return _all_windows_except(arg, closing=True)


def _act_minimize_all_windows_except(arg: str) -> str:
    """minimize_all_windows_except, <names to keep>: the same, minimizing."""
    return _all_windows_except(arg, closing=False)


# ─── "close that": the window / tab JARVIS opened last (S1, 2026-10-02) ──

def _same_page_title(bc, a: str, b: str) -> bool:
    """True when two browser titles name the same page (browser suffix,
    invisible marks and spacing ignored)."""
    def _norm(t):
        t = bc._strip_bidi_and_nbsp(t or "")
        page = _browser_page_title(bc, t)
        return " ".join((page if page is not None else t).lower().split())
    try:
        return bool(_norm(a)) and _norm(a) == _norm(b)
    except Exception:
        return False


def _act_close_last_opened(_arg: str = "") -> str:
    """Close the window or tab JARVIS ITSELF opened last
    (core.opened_ledger), and nothing else: "close that" right after JARVIS
    opened something, and the close half of "close that and open X instead".

      * a window JARVIS made (open_on_monitor, the streaming actions) is
        closed by its handle;
      * a tab JARVIS added to an existing browser window (open_url,
        web_search) is closed only while that window still shows the page
        JARVIS opened - otherwise the tab in front may be the owner's, and it
        is left alone;
      * no record (nothing opened in the last CLOSE_MAX_AGE_S) closes nothing
        and says so, so the follow-up round can ask which window he means.
    Never closes JARVIS's own host (FORBIDDEN_TARGETS)."""
    from core import opened_ledger as _ol
    entry = _ol.last_opened()
    if entry is None:
        mins = int(_ol.CLOSE_MAX_AGE_S // 60)
        return ("couldn't close it: I have no record of opening a window or "
                f"tab in the last {mins} minutes, so I left every window as "
                "it is - ask which one he means")
    bc = _bc()
    label = _ol.describe(entry)
    try:
        import pygetwindow as gw
        wins = gw.getAllWindows()
    except Exception:
        return (f"couldn't close the {label} I opened: window control "
                "(pygetwindow) isn't available")
    win = None
    if entry.hwnd is not None:
        win = next((w for w in wins
                    if getattr(w, "_hWnd", None) == entry.hwnd), None)
    if win is None:
        _ol.forget(entry)
        return f"the {label} I opened is already closed"
    title = getattr(win, "title", "") or ""
    if any(t in title.lower() for t in bc.FORBIDDEN_TARGETS):
        _ol.forget(entry)
        return (f"REFUSED: '{title}' looks like my own host process, so I "
                "didn't close it")
    # A WEB PAGE JARVIS opened only ever lives in a browser window. A record
    # whose window is now something else - open_on_monitor took an unrelated
    # fresh window (a reminder popup in the same 2 s), or Windows reused the
    # handle - is not the page he means (review 2026-10-02).
    if _ol.is_web_target(entry.target) and _browser_page_title(bc, title) is None:
        _ol.forget(entry)
        return (f"didn't close it: '{title}' isn't the browser window I "
                f"opened the {label} in, so I left it; ask him to name the "
                "window")
    if entry.kind == "tab":
        if (_browser_page_title(bc, title) is None
                or not _same_page_title(bc, title, entry.title)):
            return (f"didn't close it: the tab in front of that browser "
                    f"window is no longer the {label} I opened, so it may be "
                    "his own - I left it; ask him to name the tab")
        if not _close_browser_tab(bc, win):
            return (f"couldn't bring the {label} I opened to the front, so "
                    "I left it open")
        _ol.forget(entry)
        return f"closed the {label} I opened (just that tab)"
    try:
        win.close()
    except Exception as e:
        if _close_refused(e) or _window_is_elevated(win):
            return _elevated_close_line([title], 0)
        return f"couldn't close the {label} I opened: {e}"
    _ol.forget(entry)
    return f"closed the {label} I opened"


# ─── UI type (Phase 4D) ────────────────────────────────────────────────

def _act_type(text: str) -> str:
    # If this looks like a shell command and no terminal is focused, the LLM
    # is trying to "execute" it by typing into whatever window has focus —
    # which could be a chat, a code editor, a browser address bar, anything.
    # Refuse and tell the LLM to use run_shell instead.
    bc = _bc()
    # Never typing on a sign-in page the owner did not ask for - an address,
    # a code (core.auth_guard, review 2026-10-05).
    refusal = _input_auth_refusal(bc, "type", text)
    if refusal:
        return refusal
    if bc._looks_like_shell_command(text) and not bc._active_window_is_terminal():
        preview = text.strip().splitlines()[0][:80]
        return (
            f"REFUSED: that looks like a shell command, sir — and no terminal "
            f"is focused. Use [ACTION: run_shell, {preview}] to run it as a "
            f"subprocess instead of typing it into whatever happens to have focus."
        )
    try:
        bc.ui_type(text)
    except bc.UIFailsafeError as e:
        return str(e)
    return f"typed: {text[:60]}{'...' if len(text) > 60 else ''}"


# ─── Music skip/back (Phase 4E) ────────────────────────────────────────

def _act_next_song(_: str = "") -> str:
    # The Windows media session first (B029, 2026-10-01): skips the music
    # player's session, never whatever video Windows calls current.
    smtc = _smtc_transport_reply("next")
    if smtc is not None:
        return smtc
    # No SMTC → media keys. Browser Apple Music or the new UWP app both
    # respond to the OS nexttrack key.
    bc = _bc()
    if bc._apple_music_chrome_active():
        return _act_media_next()
    amapp = _apple_music_app()
    if amapp is not None and amapp.is_active_media_app():
        return _act_media_next()
    return _NOTHING_PLAYING_MSG


def _act_previous_song(_: str = "") -> str:
    smtc = _smtc_transport_reply("prev")
    if smtc is not None:
        return smtc
    bc = _bc()
    if bc._apple_music_chrome_active():
        return _act_media_prev()
    amapp = _apple_music_app()
    if amapp is not None and amapp.is_active_media_app():
        return _act_media_prev()
    return _NOTHING_PLAYING_MSG


# ─── Task queue read (Phase 4E) ────────────────────────────────────────

def _act_show_tasks(_: str = "") -> str:
    """Return the current task queue contents so JARVIS can read them aloud."""
    bc = _bc()
    if not os.path.exists(bc.TODO_FILE):
        return "no tasks queued yet"
    with open(bc.TODO_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()
    pending = [ln.strip() for ln in lines if ln.strip().startswith("- [ ]")]
    done    = [ln.strip() for ln in lines if ln.strip().startswith("- [x]")]
    if not pending and not done:
        return "the file exists but no tasks are in it"
    if not pending:
        return f"all {len(done)} task(s) are done — nothing left to do"
    summary = f"{len(pending)} pending task(s)"
    if done:
        summary += f" ({len(done)} already done)"
    return summary + ":\n" + "\n".join(pending)


# ─── Ambient mode setter (Phase 4E) ────────────────────────────────────

def _ambient_start_refused(result) -> bool:
    """Did ambient_listen_start REFUSE? It says so by RETURNING a line, never
    by raising: "Ambient mode failed to start, sir: <err>." when the worker
    died (carries the canonical "failed" FAILURE_MARKER), and "Ambient mode
    requires an exclusive mic connection — stop the wake-word listener first,
    sir." when the wake-word detector owns the mic (carries none, so it is
    named here; tests/test_audit_c5_honesty.py drives the real skill to keep
    the two in step). Never raises."""
    try:
        if not isinstance(result, str):
            return False
        from core.failure_markers import FAILURE_MARKERS
        low = result.lower()
        return ("requires an exclusive mic" in low
                or any(m in low for m in FAILURE_MARKERS))
    except Exception:
        return False


def _act_ambient_mode_set(active: bool) -> str:
    """Force ambient (silent-learning) mode on or off. Mirrors the tray
    dispatcher's ambient_mode_toggle branch so voice and tray follow the
    same code path. Persists ambient_mode_active to hud_state so the
    setting survives a JARVIS bounce.

    "Ambient mode" is meant to LEARN from what it overhears, so turning it on
    must do two things, not one:
      1. start the passive mic-transcription daemon (ambient_listen_start),
         which now SHARES the main loop's mic via the record_speech tap so the
         wake word keeps working, and
      2. start the multimodal fact-EXTRACTOR daemon, which is what actually
         distils the rolling transcripts into bobert_memory.json. Without (2)
         the mic captured audio but nothing was ever learned — the user's
         "i don't think it's even learning" symptom. The extractor is the same
         one _act_ambient_learning_set starts; we skip it in staging so test
         injects never write real memory.

    The choice is also AMBIENT_LISTEN_ENABLED, live and in user_settings.json
    (2026-10-01). The boot autostart (skills/ambient_listen.register) reads
    only that key, never hud_state, so a voice "stop eavesdropping" lasted one
    session and the room mic came back on at the next restart — the bug the
    wake-word setter had until 2026-07-21. The live flag also gates
    _ambient_learn_from_gated, which kept learning after "off". Saved through
    the Settings GUI's single-key merge writer; skipped in staging, like the
    extractor below.

    A refused start is reported, not announced (A44, 2026-10-02):
    ambient_listen_start refuses by RETURNING a line (see
    _ambient_start_refused), and that used to fall through to "listening
    quietly and learning" with the flag already saved ON for every later boot.
    Now an ON is saved only once the daemon took it; a refusal puts the cell,
    hud_state and the live flags back, skips the extractor and returns the
    daemon's own reason with a FAILURE_MARKER so the follow-up loop says so."""
    bc = _bc()
    _was_flag = getattr(bc, "AMBIENT_LISTEN_ENABLED", False)
    bc._ambient_mode_active[0] = bool(active)
    bc._write_hud_state(ambient_mode_active=bool(bc._ambient_mode_active[0]))
    _on = bool(bc._ambient_mode_active[0])
    bc.AMBIENT_LISTEN_ENABLED = _on
    _was_cfg = False
    try:
        import core.config as _cfg
        _was_cfg = getattr(_cfg, "AMBIENT_LISTEN_ENABLED", False)
        _cfg.AMBIENT_LISTEN_ENABLED = _on
    except Exception:
        _cfg = None
    _staging = getattr(bc, "_is_staging", lambda: False)

    def _save() -> str:
        if _staging():
            return ""
        try:
            from tools import settings_window as sw
            sw.update_settings({"AMBIENT_LISTEN_ENABLED": _on})
        except Exception:
            return " (though I couldn't save that for next boot)"
        return ""

    def _refused(why) -> str:
        bc._ambient_mode_active[0] = False
        bc._write_hud_state(ambient_mode_active=False)
        bc.AMBIENT_LISTEN_ENABLED = _was_flag
        if _cfg is not None:
            _cfg.AMBIENT_LISTEN_ENABLED = _was_cfg
        return f"ambient daemon refused: {why}"

    # An OFF is saved before the stop, so it holds for the next boot even if
    # the stop raises; an ON only after the daemon accepted it (below).
    caveat = "" if _on else _save()
    action_name = "ambient_listen_start" if bc._ambient_mode_active[0] else "ambient_listen_stop"
    fn = bc.ACTIONS.get(action_name)
    if fn is not None:
        try:
            rv = fn("")
        except Exception as e:
            return _refused(e) if _on else f"ambient daemon refused: {e}"
        if _on and _ambient_start_refused(rv):
            return _refused(str(rv).strip())
    if _on:
        caveat = _save()
    # Start / stop the fact-extractor alongside the mic daemon so ambient mode
    # genuinely folds overheard speech into long-term memory.
    if not _staging():
        _ext = sys.modules.get("skill_ambient_multimodal_extract")
        if _ext is not None:
            _ext_action = ("ambient_extract_start" if bc._ambient_mode_active[0]
                           else "ambient_extract_stop")
            _ext_fn = getattr(_ext, _ext_action, None)
            if callable(_ext_fn):
                try:
                    _ext_fn("")
                except Exception:
                    pass
    state_word = "active" if bc._ambient_mode_active[0] else "off"
    return f"Ambient mode {state_word}, sir — Chappie is {'listening quietly and learning' if bc._ambient_mode_active[0] else 'standing down'}{caveat}."


def _act_greet_new_people_set(on: bool) -> str:
    """Flip GREET_NEW_PEOPLE_ENABLED live, in-process — the proactive 'who are
    all these new people?' greeting fired by skills/face_tracker when it sees
    multiple UNRECOGNISED faces (friends over). Mirrors the ambient_mode_on/off
    setter's idempotent live-toggle shape.

    We set the flag on core.config so the face-tracker poller (which re-reads
    core.config every tick) picks it up WITHOUT a restart, AND save it to
    user_settings.json through the Settings GUI's single-key merge writer
    (2026-10-01). It used to be a session-only toggle on the premise that a
    boot reverts to the opt-in default — but an owner file holding
    GREET_NEW_PEOPLE_ENABLED=true reverts it to ON, so "stop greeting people"
    silently came back at the next restart (the wake-word setter's 2026-07-21
    bug). Saving is best-effort: on failure the flip holds for this session
    and the reply says so. Honest about the face-ID dependency: the greeting
    needs the webcams to actually recognise faces, so it nudges the user to
    enable FACE_ID_ENABLED when that's still off."""
    try:
        import core.config as _cfg
        _cfg.GREET_NEW_PEOPLE_ENABLED = bool(on)
        face_id_on = bool(getattr(_cfg, "FACE_ID_ENABLED", False))
    except Exception as e:   # pragma: no cover - core.config import never fails here
        return f"I couldn't change the new-people greeting, sir — {e}."
    try:
        from tools import settings_window as sw
        sw.update_settings({"GREET_NEW_PEOPLE_ENABLED": bool(on)})
        caveat = ""
    except Exception:
        caveat = " (though I couldn't save that for next boot)"
    if not on:
        return f"Noted, sir — I'll stop announcing new faces{caveat}."
    msg = ("Will do, sir — when a few unfamiliar faces turn up I'll say hello "
           f"once{caveat}.")
    if not face_id_on:
        msg += (" Note face recognition is still off, so I won't actually spot "
                "them until you enable it.")
    return msg


# ─── Skills reload (Phase 4E) ──────────────────────────────────────────

def _act_reload_skills(_: str = "") -> str:
    """Re-import every file in skills/ and let each call register(ACTIONS).
    Spawned on a daemon because the side modules can pull heavy deps
    (chroma, numpy) on first import."""
    bc = _bc()
    from core.config import SKILLS_ENABLED
    if not SKILLS_ENABLED:
        return "skills disabled"

    def _do():
        try:
            before = len(bc.ACTIONS)
            bc.load_skills()
            after = len(bc.ACTIONS)
            return f"skills reloaded ({after - before:+d} new actions, total={after})"
        except Exception as e:
            return f"reload_skills failed: {e}"
    bc._tray_async("reload_skills", _do)
    return "reloading skills"


# ─── Memory introspection (Phase 4E) ───────────────────────────────────

_RECENT_FACTS_WINDOW_S = 24 * 3600
_RECENT_FACTS_MAX = 40


def _act_show_recent_facts(_: str = "") -> str:
    """Facts learned in the LAST 24 HOURS, newest first, as text (the tray
    shows it). Read from the tiered long-term store, whose facts carry a
    created_at; bobert_memory.json's facts are bare strings with no time at
    all, which is why this used to print "the last 10" whatever their age —
    to a console nobody sees (2026-09-30 audit)."""
    try:
        from core import long_term_memory as ltm
        facts = ltm.list_facts()
    except Exception as e:
        return f"I couldn't read the long-term memory store: {e}"
    cutoff = time.time() - _RECENT_FACTS_WINDOW_S

    def _created(entry) -> float:
        try:
            return float(entry.get("created_at") or 0.0)
        except (TypeError, ValueError, AttributeError):
            return 0.0
    recent = [f for f in facts if isinstance(f, dict) and _created(f) >= cutoff]
    recent.sort(key=_created, reverse=True)
    if not recent:
        return (f"No new facts in the last 24 hours ({len(facts)} on file "
                "in total).")
    lines = [f"{len(recent)} fact(s) learned in the last 24 hours "
             f"({len(facts)} on file):", ""]
    for entry in recent[:_RECENT_FACTS_MAX]:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(_created(entry)))
        lines.append(f"  {when}  {str(entry.get('text') or '').strip()}")
    if len(recent) > _RECENT_FACTS_MAX:
        lines.append(f"  … and {len(recent) - _RECENT_FACTS_MAX} more")
    return "\n".join(lines)


def _act_export_memory(_: str = "") -> str:
    """Copy bobert_memory.json to backups/memory_export_<ts>.json.
    Read-only — original file untouched."""
    bc = _bc()
    try:
        with bc._memory_lock:
            if not os.path.exists(bc.MEMORY_FILE):
                return "no memory file to export"
            ts = time.strftime("%Y%m%d_%H%M%S")
            mem_dir = os.path.dirname(os.path.abspath(bc.MEMORY_FILE)) or "."
            export_dir = os.path.join(mem_dir, "backups")
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(export_dir, f"memory_export_{ts}.json")
            shutil.copy2(bc.MEMORY_FILE, export_path)
        return f"memory exported -> backups/{os.path.basename(export_path)}"
    except Exception as e:
        return f"export_memory failed: {e}"


# ─── Self-diagnostic tray wrappers (Phase 4E) ──────────────────────────

def _act_run_diagnostic_tray(_: str = "") -> str:
    """Tray wrapper that runs the self_diagnostic sweep on a daemon
    thread so the drainer isn't blocked for the 30-60s a full sweep
    can take. If self_diagnostic isn't loaded, reports so."""
    bc = _bc()
    sd = bc._selfdiag_module()
    if sd is None or not hasattr(sd, "run_diagnostic"):
        return "self_diagnostic skill not loaded"
    bc._tray_async("run_diagnostic", lambda: sd.run_diagnostic(""))
    return "diagnostic sweep started"


def _act_show_last_diagnostic(_: str = "") -> str:
    """Wraps self_diagnostic.last_diagnostic_run() so the tray's
    'Show Last Diagnostic Run' button gets a synchronous answer (a
    JSON dump of the most recent sweep)."""
    bc = _bc()
    sd = bc._selfdiag_module()
    if sd is None or not hasattr(sd, "last_diagnostic_run"):
        return "self_diagnostic skill not loaded"
    try:
        out = sd.last_diagnostic_run("") or ""
        # Return the run itself (the tray opens long answers as a text file);
        # it used to print one line to a console nobody sees.
        return out.strip() or "No diagnostic run on record yet."
    except Exception as e:
        return f"show_last_diagnostic failed: {e}"


# ─── Generic streaming dispatcher (Phase 4F) ───────────────────────────

def _act_play_streaming(args: str) -> str:
    """Generic streaming play. Two formats:
        play_streaming, <service>|<title>
        play_streaming, <title>                 (defaults to YouTube)

    Use this when the user says 'play X' without a service preference, or
    when the LLM wants to centralize service dispatch. Service names:
        netflix, prime_video (amazon, prime), apple_music (apple),
        spotify, youtube, disney_plus (disney), hulu, max (hbo, hbo_max)
    """
    bc = _bc()
    if "|" in args:
        raw_service, query = (s.strip() for s in args.split("|", 1))
        service = bc._normalize_service(raw_service)
        if service not in bc._STREAMING_SERVICES:
            return (
                f"unknown service '{raw_service}'. Known: "
                + ", ".join(sorted(bc._STREAMING_SERVICES.keys()))
            )
        return bc._streaming_auto_play(service, query)
    # No service specified — default to YouTube as the universal fallback
    return bc._streaming_auto_play("youtube", args.strip())


def _act_streaming_search(args: str) -> str:
    """streaming_search, <service> | <title> - open the title's search page
    on a video service from the VERIFIED table (core/streaming_search.py),
    or the service's home page, said so, when it has no verified search link
    (Disney+). Never a guessed URL (S2, 2026-10-02: hbomax.com/search?q= was
    a 404). Stops on a sign-in wall."""
    bc = _bc()
    from core import streaming_search as _ss
    if "|" in args:
        raw_service, query = (s.strip() for s in args.split("|", 1))
    elif "," in args:
        raw_service, query = (s.strip() for s in args.split(",", 1))
    else:
        return "format: streaming_search, <service> | <title>"
    key = _ss.canon_service(raw_service)
    if key is None:
        return (f"unknown streaming service '{raw_service}'. Known: "
                + ", ".join(sorted(_ss.SERVICES)))
    return bc._streaming_open_search(key, query)


# ─── UI click + hotkey (Phase 4F) ──────────────────────────────────────

def _act_click(args: str) -> str:
    """args: 'x,y' or 'x,y,right' for right-click, or a description to find+click.
    Coords can be negative (for monitors to the left of the primary, e.g. -2215,249).
    Prefix with 'monitor:NAME|' to prefer that monitor:
        click, monitor:left|the play button

    A DESCRIPTION click goes through the grounded executor (core.
    grounded_click, 2026-10-05) - the same chokepoint as click_on_screen:
    read the windows by name, resolve, guard, click, verify. Live 00:28-00:30
    the old path photographed a whole monitor (or all four) and clicked
    nothing in five tries."""
    bc = _bc()
    # Optional monitor prefix
    monitor, args = bc._parse_monitor_prefix(args)

    m = re.match(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*(?:,\s*(left|right|middle))?\s*$", args)
    # Never a sign-in click the owner did not ask for (2026-10-05): see
    # core.auth_guard. A coordinate click on a sign-in page counts too.
    refusal = _click_auth_refusal(bc, "" if m else args)
    if refusal:
        return refusal
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        button = m.group(3) or "left"
        try:
            bc.ui_click(x, y, button)
        except bc.UIFailsafeError as e:
            return str(e)
        return f"clicked {button} at ({x},{y})"

    # Description-based click — refuse if it's targeting Bobert's own host
    if bc._is_self_close_attempt(args):
        return (
            f"REFUSED: '{args}' looks like an attempt to close the terminal or "
            f"Python process running me. Closing it would kill my session. "
            f"Ask the user to close it manually if they really want to."
        )
    target = f"monitor:{monitor}|{args}" if monitor else args
    return _click_on_screen(target, _turn_said(bc))


def _opened_page_context(fg_hwnd=None) -> "tuple[str, str] | None":
    """(the URL / app JARVIS opened last - "" once its page has moved on -,
    the CURRENT title of its window) for core.opened_ledger's newest entry
    no older than PAGE_MAX_AGE_S, when that window still exists and is the
    window in front (``fg_hwnd``; unknown counts as in front). None
    otherwise. Review 2026-10-05: a /login address JARVIS opened kept
    refusing every click for ten minutes, whatever was in front and after
    he had signed in. The address counts only while the window still shows
    the title it was opened with. Never raises."""
    try:
        from core import opened_ledger as _ol
        entry = _ol.last_opened(_ol.PAGE_MAX_AGE_S)
        if entry is None or entry.hwnd is None:
            return None
        if isinstance(fg_hwnd, int) and fg_hwnd != entry.hwnd:
            return None
        import pygetwindow as gw
        win = next((w for w in gw.getAllWindows()
                    if getattr(w, "_hWnd", None) == entry.hwnd), None)
        title = str(getattr(win, "title", "") or "") if win else ""
        if not title:
            return None
        url = (str(entry.target or "")
               if entry.title and title == entry.title else "")
        return url, title
    except Exception:
        return None


def _seam_true(bc, name: str, *args) -> bool:
    """``bc.<name>(*args) is True`` (a Mock monolith is never True)."""
    try:
        return getattr(bc, name)(*args) is True
    except Exception:
        return False


def _auth_context(bc) -> dict:
    """What core.auth_guard judges an input by: the owner's words this
    turn, the page in front (the focused window's title; the page JARVIS
    opened, while it is in front), what this turn's looks at the screen
    said, what its find_on_screen looked for, and whether an input was
    already refused this turn. Never raises."""
    ctx = {"owner_text": _seam_str(bc, "_turn_user_text"), "urls": [],
           "titles": [], "screen_texts": [], "looked_for": [],
           "refused_before": False}
    try:
        fg_hwnd = None
        try:
            fg = bc._read_focused_window()
            if isinstance(fg, tuple) and len(fg) >= 2:
                if isinstance(fg[0], int):
                    fg_hwnd = fg[0]
                if isinstance(fg[1], str):
                    ctx["titles"].append(fg[1])
        except Exception:
            pass
        page = _opened_page_context(fg_hwnd)
        if page is not None:
            if page[0]:
                ctx["urls"].append(page[0])
            if page[1] not in ctx["titles"]:
                ctx["titles"].append(page[1])
        ctx["screen_texts"] = [s for s in _seam_list(bc, "_turn_screen_texts")
                               if isinstance(s, str)]
        ctx["looked_for"] = [s for s in _seam_list(bc, "_turn_click_targets")
                             if isinstance(s, str)]
        ctx["refused_before"] = _seam_true(bc, "_turn_auth_refused")
    except Exception:
        pass
    return ctx


def _note_auth_refusal(bc, what: str, why: str) -> None:
    """Log a refused input (never its target: an account entry carries the
    owner's name and e-mail address) and mark the turn, so the rest of the
    reply's clicks and keys are refused too."""
    print(f"  [auth-guard] not {what} ({why or 'an earlier refusal'}): the "
          "owner did not ask for it this turn", flush=True)
    try:
        bc._turn_note_auth_refused()
    except Exception:
        pass


def _click_auth_refusal(bc, description: str) -> str:
    """The TERMINAL refusal (core.auth_guard.click_refusal) when this click
    would pick an account, sign in or grant access - or land on a sign-in
    page - and the owner's own words this turn did not ask for that exact
    click; "" when the click may go ahead.

    Live 2026-10-05 00:14:18: the owner asked for the console page "so I can
    sign in"; JARVIS looked at the screen and clicked his Google account
    entry by itself. Never raises."""
    try:
        from core import auth_guard as _ag
        ctx = _auth_context(bc)
        line = _ag.click_refusal(description, **ctx)
        if line:
            why = (_ag.auth_control(description)
                   or _ag.auth_page(ctx["urls"], ctx["titles"],
                                    ctx["screen_texts"])
                   or _ag.auth_overlay(ctx["screen_texts"])
                   or ("it looked for a sign-in control"
                       if ctx["looked_for"] else ""))
            _note_auth_refusal(bc, "clicking", why)
        return line
    except Exception:
        return ""


def _input_auth_refusal(bc, kind: str, value: str) -> str:
    """The TERMINAL refusal (core.auth_guard.input_refusal) for typing or a
    submit key on a sign-in page / under a sign-in pop-up, or after an input
    was refused this turn, that the owner did not ask for this turn; "" when
    it may go ahead. Review 2026-10-05: the click guard alone left
    "[ACTION: type, <address>] [ACTION: press, enter]" free to sign him in.
    Never raises."""
    try:
        from core import auth_guard as _ag
        if kind != "type" and not _ag.is_submit_key(value):
            return ""
        ctx = _auth_context(bc)
        ctx.pop("looked_for", None)
        line = _ag.input_refusal(kind, value, **ctx)
        if line:
            why = (_ag.auth_page(ctx["urls"], ctx["titles"],
                                 ctx["screen_texts"])
                   or _ag.auth_overlay(ctx["screen_texts"]))
            _note_auth_refusal(bc, "typing" if kind == "type"
                               else "pressing that key", why)
        return line
    except Exception:
        return ""


def _turn_said(bc=None) -> str:
    """The owner's words this turn (the grounding ledger), '' outside one."""
    try:
        bc = bc if bc is not None else _loaded_bc()
        said = bc._turn_user_text() if bc is not None else ""
        return said if isinstance(said, str) else ""
    except Exception:
        return ""


def _act_click_on_screen(args: str) -> str:
    """click_on_screen, <what> | pick:<n> | scene:previous - find the thing
    on screen BY NAME and click it, verified (core.grounded_click). Spoken
    word for word: a verified fact, a question naming the real options, or
    an honest "I don't see it".

    The same sign-in pre-check as _act_click (core.auth_guard via
    _click_auth_refusal) runs first for a described target: the monolith
    hands every DESCRIPTION "click" here (bobert_companion._click_alias),
    and the screen route sends "click that X" here, so without it the
    v2.0.181 guard on the request itself (the page in front, this turn's
    looks, an input already refused) would apply to a direct _act_click
    only. grounded_click then judges the RESOLVED label again, by the
    target window's own page. A "pick:<n>" / "scene:previous" answer has
    no description of its own: grounded_click judges the option it
    resolves to."""
    desc = re.sub(r"^\s*monitor:[\w-]+\s*\|", "", str(args or "")).strip()
    if desc and not desc.lower().startswith(("pick:", "scene:")):
        refusal = _click_auth_refusal(_loaded_bc(), desc)
        if refusal:
            return refusal
    return _click_on_screen(args, _turn_said())


def _click_on_screen(args: str, said: str) -> str:
    from core import grounded_click as _gc
    r = _gc.run_bounded(args, said=said, mode="click")
    _note_screen_look("click_on_screen", r.text)
    return _youtube_when_not_on_screen(r, args, said) or r.text


def _youtube_when_not_on_screen(r, args: str, said: str) -> str:
    """"play that MrBeast video on YouTube" when NO such video is on the
    screen: he named YouTube, so its search-and-play runs (main did that;
    review 2026-10-05: the screen route answered "I don't see it" and
    nothing played). Only on a plain "not on screen": never after a time-out
    (the page may hold it), a failure, an open question or a pick / scene
    answer; and only for that exact shape (core.onscreen_refs.
    youtube_play_query). "" otherwise. Never raises."""
    try:
        from core import grounded_click as _gc
        from core import onscreen_refs as _or
        if (r.outcome != _gc.NOT_FOUND or r.failed
                or r.tier == _gc.TIMED_OUT):
            return ""
        if str(args or "").strip().lower().startswith(("pick:", "scene:")):
            return ""
        if _gc.pending_choice() is not None:
            return ""
        q = _or.youtube_play_query(said)
        if not q:
            return ""
        fn = getattr(_bc(), "ACTIONS", {}).get("youtube_play")
        if not callable(fn):
            return ""
        print(f"  [click] {q!r} is not on screen and he named YouTube - "
              "playing it from a search", flush=True)
        res = str(fn(q) or "")
        from core.failure_markers import FAILURE_MARKERS
        low = res.lower()
        if not res or any(m in low for m in FAILURE_MARKERS):
            return ("I don't see that on screen, sir, and the YouTube "
                    f"search didn't play it: {res or 'no answer'}")
        return (f"I don't see that on screen, sir, so I've put '{q}' on "
                "from YouTube.")
    except Exception:
        return ""


def _act_undo_click(args: str = "") -> str:
    """undo_click[, other] - take back JARVIS's own last UI action (the
    window / tab it opened, or Back in the page it navigated), within two
    minutes; "other" then asks which one he meant, naming the options that
    were on screen BEFORE that action."""
    from core import grounded_click as _gc
    other = "other" in str(args or "").lower()
    r = _gc.undo(other=other, said=_turn_said())
    return r.text


def _act_note_for_claude(args: str = "") -> str:
    """note_for_claude, <note> - the owner's words for the developer,
    appended to data/notes_for_claude.jsonl (core.dev_notes). JARVIS never
    claims to have relayed it: there is no channel to Claude."""
    from core import dev_notes as _dn
    from core import onscreen_refs as _or
    said = _turn_said()
    note = " ".join(str(args or "").split()) or (_or.claude_note(said) or said)
    if not note:
        return "What should I note for Claude, sir?"
    try:
        from core import guest_mode as _gm
        if _gm.is_on():
            return _dn.GUEST_LINE
    except Exception:
        pass
    rec = _dn.add_note(note, utterance=said or note)
    if rec is None:
        return ("I couldn't save that note for Claude, sir \u2014 the notes "
                "file isn't writable.")
    return _dn.SPOKEN_LINE


def _act_screen_memory(args: str = "") -> str:
    """screen_memory, pause [N] | unpause | status | exclude_this |
    exclude <app> - the continuous screen memory's controls (core.
    screen_memory)."""
    from core import screen_memory as _sw
    a = " ".join(str(args or "").split())
    low = a.lower()
    m = re.match(r"^(?:pause|stop)(?:\s+(\d+(?:\.\d+)?))?", low)
    if m:
        return _sw.pause(float(m.group(1)) if m.group(1) else None)
    if low.startswith(("unpause", "resume", "start", "again", "on")):
        return _sw.unpause()
    if low.startswith(("status", "are you")):
        return _sw.status_line()
    if low.startswith(("exclude_this", "exclude this", "this")):
        return _sw.exclude_foreground()
    m = re.match(r"^exclude\s+(.+)$", a, re.IGNORECASE)
    if m:
        return _sw.exclude_app(m.group(1))
    return ("format: screen_memory, pause [minutes] | unpause | status | "
            "exclude_this | exclude <app>")


def _act_forget_screen(args: str = "") -> str:
    """forget_screen, <span> - "forget the last hour / 10 minutes / today /
    everything you saw": timeline rows, vision-trace entries, the scene ring
    and the cached looks in that span."""
    # ltm-exempt: screen memory is never written to long-term memory - the
    # screen timeline / vision trace are their own stores (core.screen_memory
    # never fact-extracts), so there is no LTM copy to purge.
    from core import onscreen_refs as _or
    from core import screen_memory as _sw
    a = " ".join(str(args or "").split()).lower()
    span = None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(m|min|mins|minutes?|h|hours?)?$", a)
    if m:
        n = float(m.group(1))
        span = {"seconds": n * (3600.0 if (m.group(2) or "m").startswith("h")
                                else 60.0)}
    elif a in ("all", "everything"):
        span = {"all": True}
    elif a == "today":
        span = {"today": True}
    elif a in ("hour", "last hour", "1 hour"):
        span = {"seconds": 3600.0}
    if span is None:
        span = _or.forget_span(_turn_said()) or {"seconds": 3600.0}
    return _sw.forget(span)


def _note_screen_look(name: str, text: str) -> None:
    """Feed what a screen action found into this turn's screen texts (the
    sign-in guard reads them; core.auth_guard's _turn_screen_texts seam,
    when the monolith has it). Never raises.

    ONE entry per look (review 2026-10-09): the monolith's
    _note_turn_action_ran also records a look action's result after it
    runs. ``frame["screen_fed"]`` names the action that fed the frame here,
    so that run is not recorded a second time - two entries per look had
    halved the guard's window (the last 4 looks became the last 2) and a
    sign-in page seen early in a chain was forgotten three actions later.
    This entry is the one kept: see_screen's raw page text, not an answer
    built from it."""
    try:
        bc = _loaded_bc()
        frame = getattr(getattr(bc, "_turn_grounding", None), "frame", None)
        if frame is not None and text:
            seen = frame.setdefault("screen", [])
            seen.append(str(text)[:4000])
            del seen[:-4]
            frame["screen_fed"] = str(name or "").strip().lower()
    except Exception:
        pass


def _opened_page_now():
    """(entry, monitor) for the page JARVIS opened last (core.opened_ledger,
    PAGE_MAX_AGE_S) while its window STILL EXISTS - the monitor it is on
    NOW (he may have moved it), else the one it opened on - or None. A
    record whose window is gone (or a test's fake handle) aims nothing.
    Never raises."""
    try:
        from core.config import MONITORS
        from core import monitor_geometry as _mg
        from core import opened_ledger as _ol
        entry = _ol.last_opened(_ol.PAGE_MAX_AGE_S)
        if entry is None or entry.hwnd is None:
            return None
        import pygetwindow as gw
        win = next((w for w in gw.getAllWindows()
                    if getattr(w, "_hWnd", None) == entry.hwnd), None)
        if win is None:
            return None
        mon = _mg.monitor_for_rect(win.left, win.top, win.width, win.height,
                                   MONITORS) or entry.monitor
        return (entry, mon) if mon else None
    except Exception:
        return None


def _act_hotkey(args: str) -> str:
    bc = _bc()
    keys = [bc._normalize_key(k) for k in args.split("+")]
    refusal = _input_auth_refusal(
        bc, "hotkey", "+".join(k for k in keys if isinstance(k, str)))
    if refusal:
        return refusal
    # Refuse alt+f4 if the currently focused window is our host process
    if set(keys) == {"alt", "f4"}:
        try:
            import pygetwindow as gw
            active = gw.getActiveWindow()
            if active and active.title:
                title_lower = active.title.lower()
                if any(t in title_lower for t in bc.FORBIDDEN_TARGETS):
                    return (
                        f"REFUSED: alt+f4 with '{active.title}' focused would "
                        f"kill my own host process. Ask the user to do it manually."
                    )
        except Exception:
            pass
    try:
        bc.ui_hotkey(*keys)
        return f"pressed {'+'.join(keys)}"
    except bc.UIFailsafeError as e:
        return str(e)
    except Exception as e:
        return f"hotkey failed: {e}"


# ─── Pipeline + backup + memory reset (Phase 4F) ───────────────────────

def _act_stop_pipeline(_: str = "") -> str:
    """Quiet the overnight engine. Clears the pending immediate-fire flag,
    drops sleep mode, and removes the on-disk overnight flag so the engine
    won't re-arm on the next boot. Does NOT kill an in-progress upgrade
    subprocess — once upgrade_jarvis.py is running it owns the process
    lifecycle; the most we can do here is stop new cycles from starting."""
    bc = _bc()
    cleared = False
    try:
        if bc._overnight_run_now.is_set():
            bc._overnight_run_now.clear()
            cleared = True
    except Exception:
        pass
    try:
        if bc._sleep_mode[0]:
            bc._sleep_mode[0] = False
            cleared = True
    except Exception:
        pass
    try:
        if os.path.exists(bc.OVERNIGHT_FLAG_FILE):
            os.remove(bc.OVERNIGHT_FLAG_FILE)
            cleared = True
            try: bc._write_hud_state(overnight_expiry=0.0)
            except Exception: pass
    except Exception:
        pass
    return "overnight engine quieted" if cleared else "nothing pending to halt"


def _act_force_backup(_: str = "") -> str:
    """Snapshot the codebase via upgrade_jarvis.backup_codebase() on a
    daemon thread. Returns immediately so the drainer isn't blocked."""
    bc = _bc()

    def _do():
        try:
            search_dir = os.path.dirname(os.path.abspath(bc.__file__))
            upgrade_path = os.path.join(search_dir, "upgrade_jarvis.py")
            if not os.path.exists(upgrade_path):
                return "upgrade_jarvis.py not found"
            import importlib.util as _ilu
            spec = _ilu.spec_from_file_location("_force_backup_uj", upgrade_path)
            mod  = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            dest = mod.backup_codebase()
            return f"backup -> {os.path.basename(dest)}"
        except Exception as e:
            return f"backup failed: {e}"
    bc._tray_async("force_backup", _do)
    return "backup started"


def _refresh_live_prompt_after_wipe(bc) -> str:
    """A memory wipe must reach the LIVE system prompt at once (2026-10-01).
    Both wipes run from the spoken "yes" (handle_confirmation_response) or
    the tray, neither of which rebuilds the prompt, and the next post-turn
    rebuild is deferred while the owner keeps talking (the prompt freeze) --
    so "what do you know about me?" right after "Done." recited the erased
    facts. Returns '' or a disclosure for the reply."""
    try:
        bc._rebuild_prompt_now()
        return ""
    except Exception as e:
        return (f"the live prompt was NOT refreshed ({e}); the old facts may "
                f"be used until the next rebuild")


def _forget_learning_in_flight(bc, cutoff) -> None:
    """Nothing queued for learning before a wipe may land after it
    (2026-10-01): the learner waits up to two minutes for the owner to go
    quiet, so the turns spoken just before "forget the last hour" / "yes"
    were extracted AFTER the purge and written back. Call holding
    bc._memory_lock (merge_memory re-checks under it). Never raises."""
    try:
        _inv = getattr(bc, "_learn_invalidate", None)
        if callable(_inv):
            _inv(cutoff)
    except Exception as e:
        print(f"  [memory] learn queue not invalidated: {e}")


def _forget_screen_and_voice(span: dict, since) -> "tuple[list, list]":
    """The screen memory's and the clone voice's share of a memory wipe
    (review 2026-10-09: both wipes left them, so after a confirmed "forget
    the last hour" the screen timeline still answered "what was on my
    screen", the vision trace still held his words, and the clone cache's
    ledger kept a line said twice in plain text). ``span``: core.
    screen_memory's ({"seconds": 3600} / {"all": True}); ``since``: the
    same cutoff on time.time() (None = everything). Returns (bits,
    failures) for the reply - a store that could not be purged is
    DISCLOSED, never silent. Never raises."""
    bits: list = []
    failures: list = []
    try:
        from core import screen_memory as _sm
        c = _sm.forget_counts(span)
        n = int(c.get("rows", 0)) + int(c.get("traces", 0)) + int(
            c.get("cached", 0))
        if n:
            bits.append(f"{n} screen record(s)")
    except Exception as e:
        failures.append(f"what I saw on the screen was NOT forgotten ({e})")
    try:
        from core import clone_voice_client as _cvc
        client = getattr(_cvc, "CLIENT", None)
        if client is not None:
            v = client.wipe(since)
            n = max(int(v.get("takes", 0)), int(v.get("lines", 0)))
            if n:
                bits.append(f"{n} cached voice line(s)")
    except Exception as e:
        failures.append(f"the cloned voice's cache was NOT cleared ({e})")
    return bits, failures


def _act_reset_memory(_: str = "") -> str:
    """Snapshot bobert_memory.json to backups/, then re-initialise it
    to the empty schema. Destructive — but the backup is unconditional,
    so the user can restore by copying the file back.

    Also wipes the tiered long-term store (core/long_term_memory:
    facts.json + chroma + episodes.jsonl, with its own pre_reset backup):
    _ltm_context() retrieves from that store every turn, so leaving it
    intact made a confirmed wipe a lie — JARVIS kept reciting the facts it
    just claimed to have erased (2026-07-21 audit). A failed LTM wipe is
    DISCLOSED in the reply, never silent.

    2026-10-01: also wiped -- each one survived and was read back after a
    confirmed reset: turns still queued for learning, the session-summary
    index and the verbatim voice-command log (backed up first), this
    process's conversation history, and the LIVE system prompt.

    2026-10-09: and the screen memory (timeline, vision trace, scenes,
    cached looks) and the cloned voice's cache (its takes and the line
    ledger's plain text) - _forget_screen_and_voice."""
    bc = _bc()
    try:
        with bc._memory_lock:
            ts = time.strftime("%Y%m%d_%H%M%S")
            mem_dir = os.path.dirname(os.path.abspath(bc.MEMORY_FILE)) or "."
            backup_dir = os.path.join(mem_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"memory_pre_reset_{ts}.json")
            existed = os.path.exists(bc.MEMORY_FILE)
            if existed:
                try:
                    shutil.copy2(bc.MEMORY_FILE, backup_path)
                except Exception as e:
                    return f"backup failed, refused to wipe: {e}"
            _forget_learning_in_flight(bc, None)
            bc.save_memory(bc._empty_memory())
        # Long-term store wipe runs OUTSIDE bc._memory_lock so it can't nest
        # with long_term_memory._lock (reset_all takes its own lock).
        try:
            from core import long_term_memory as ltm
            n = ltm.reset_all()
            ltm_note = (f" + {n} long-term fact(s) cleared "
                        f"(backup -> data/long_term_memory/backups)")
        except Exception as le:
            ltm_note = (f" — WARNING: the long-term semantic store was NOT "
                        f"cleared ({le}); recorded facts and the "
                        f"conversation log remain")
        # This session's opening-utterance record ("what was the first thing
        # I asked you", v2.0.148) is wiped with everything else.
        try:
            _forget_opening = getattr(bc, "_forget_session_opening_since",
                                      None)
            if callable(_forget_opening):
                _forget_opening(None)
        except Exception as oe:
            ltm_note += (f" — WARNING: this session's opening-utterance "
                         f"record was NOT cleared ({oe})")
        # This process's conversation history (and the running session
        # summary built from it) -- in every LLM call until now. BEFORE the
        # session-summary purge below (2026-10-01): this bumps the summary
        # generation, so a checkpoint whose LLM call returns during the purge
        # discards its summary instead of writing the session's row back.
        try:
            _forget_live = getattr(bc, "_forget_live_conversation", None)
            if callable(_forget_live):
                _forget_live(None)
        except Exception as ce:
            ltm_note += (f" — WARNING: this conversation's history was NOT "
                         f"cleared ({ce})")
        # The pattern store's conversation logs: the session-summary index
        # ("what did we do yesterday") and the verbatim voice-command log
        # (the "where did we leave off" greeting). Backed up first.
        try:
            bc.pattern_memory.reset_conversation_logs(backup_dir)
        except Exception as pe:
            ltm_note += (f" — WARNING: the session summaries and the "
                         f"voice-command log were NOT cleared ({pe})")
        # What the screen memory saw and the cloned voice's cache of what
        # was said (review 2026-10-09).
        _sv_bits, _sv_fail = _forget_screen_and_voice({"all": True}, None)
        if _sv_bits:
            ltm_note += " + " + " + ".join(_sv_bits) + " cleared"
        for _f in _sv_fail:
            ltm_note += f" — WARNING: {_f}"
        _pw = _refresh_live_prompt_after_wipe(bc)
        if _pw:
            ltm_note += f" — WARNING: {_pw}"
        if existed:
            return (f"memory reset (backup -> backups/"
                    f"{os.path.basename(backup_path)}){ltm_note}")
        return f"memory was already empty{ltm_note}"
    except Exception as e:
        return f"reset_memory failed: {e}"


# ─── Version info (Phase 4G) ───────────────────────────────────────────

def _act_version_info(_: str = "") -> str:
    """Report the current version + a human-friendly rendering of the last
    upgrade time (e.g. 'this morning at 8:43 AM').

    SINGLE SOURCE — do NOT "fix" this to report data/version.json's own
    ``version`` key.  That key is the self-upgrade pipeline's internal counter
    and has been frozen at an old value for months (1.0.17 while the release
    was 2.0.104); reporting it would make JARVIS state his version wrong by a
    whole major series.  The release version comes from core/version.py (the
    VERSION file).  Only ``last_upgrade_at`` is read out of the JSON, and even
    that yields to the release's git date when that is newer — see the inline
    comments below."""
    bc = _bc()
    try:
        from datetime import datetime as _dt
        try:
            from core.version import __version__ as release_ver
        except Exception:
            release_ver = "unknown"
        _root = os.path.dirname(os.path.abspath(bc.__file__))
        _ver_path = os.path.join(_root, "data", "version.json")
        data = {}
        if os.path.exists(_ver_path):
            with open(_ver_path, "r", encoding="utf-8") as _vf:
                data = json.load(_vf)
        ver = release_ver  # single-source release version (core/version.py),
        #                    not the self-upgrade pipeline's internal counter
        ts_iso = data.get("last_upgrade_at") or ""
        # last_upgrade_at is written ONLY by the self-upgrade pipeline —
        # releases deployed via git checkout never touch version.json, so
        # the reported date went stale (live bug: v1.99.0 announced as
        # "last updated on May 30"). The release's own date comes from git
        # (the v<VERSION> tag / the commit that set VERSION; the VERSION mtime
        # only outside a checkout — core.version.release_timestamp), and
        # whichever of the two is newer wins. It used to be the VERSION mtime
        # alone, and only when version.json existed: a box with no pipeline
        # history never heard a date at all (2026-10-02).
        ts = None
        try:
            ts = _dt.fromisoformat(ts_iso) if ts_iso else None
        except Exception:
            ts = None
        try:
            from core.version import release_timestamp
            _rel = release_timestamp(_root)
            if _rel is not None:
                _rel_dt = _dt.fromtimestamp(_rel)
                if ts is None or _rel_dt > ts:
                    ts = _rel_dt
        except Exception:
            pass
        if ts is None:
            if ts_iso:
                return f"I'm on version {ver}, last updated {ts_iso}."
            if not os.path.exists(_ver_path):
                return f"I'm on version {ver}, sir."
            return f"I'm on version {ver}, sir — no upgrade timestamp on file."
        now = _dt.now()
        same_day = (ts.date() == now.date())
        yesterday = ((now.date() - ts.date()).days == 1)
        hour = ts.hour
        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 21:
            period = "evening"
        else:
            period = "night"
        clock = ts.strftime("%I:%M %p").lstrip("0")
        if same_day:
            when = f"this {period} at {clock}"
        elif yesterday:
            when = f"yesterday {period} at {clock}"
        else:
            days_ago = (now.date() - ts.date()).days
            if days_ago < 7:
                when = f"{ts.strftime('%A')} {period} at {clock}"
            else:
                when = f"on {ts.strftime('%B %d')} at {clock}"
        return f"I'm on version {ver}, last updated {when}, sir."
    except Exception as e:
        return f"could not read version info: {e}"


def _act_check_for_updates(_: str = "") -> str:
    """Check GitHub for a NEWER published release and report it conversationally.

    Delegates to core.update_checker, which is total (never raises) and degrades
    gracefully when there's no GitHub token or no network — so this action always
    returns a sentence, never an error."""
    from core import update_checker as uc
    return uc.update_message(uc.check_for_update())


def _act_report_bug(description: str = "") -> str:
    """Log a bug the USER is reporting — scrubbed of personal data LOCALLY — and
    open a pre-filled GitHub issue for them to review + submit (consent-gated; no
    auto-send). 'report a bug: the timer never fired', 'jarvis that was wrong'."""
    desc = (description or "").strip()
    if not desc:
        return ("Tell me what went wrong and I'll log it — e.g. 'report a bug: "
                "the timer never fired'.")
    from core import bug_reporter as br
    rep = br.record_bug("user", desc, context={"source": "voice"})
    # Autonomous API submission when opted in (JARVIS_BUG_AUTO_SUBMIT=1);
    # otherwise the consent-gated browser path below.
    if br.auto_submit_enabled():
        issue = br.api_submit_issue(rep)
        if issue:
            return f"Logged it (scrubbed of personal info) and filed a GitHub issue: {issue}"
        return ("Logged it locally (scrubbed) — auto-submit is on but the GitHub "
                "API call didn't go through, so it's saved in the outbox.")
    url = br.browser_submit_url(rep)
    try:
        import webbrowser
        opened = bool(webbrowser.open(url))
    except Exception:
        opened = False
    if opened:
        return ("Logged it, scrubbed of personal info, and opened a pre-filled "
                "GitHub issue — review it and hit submit to send it.")
    return ("Logged it locally, scrubbed of personal info, and saved to the bug "
            "outbox to file when you're ready.")


# ─── Smoke test + skills selftest (Phase 4G) ───────────────────────────

def _act_run_smoke_test(_: str = "") -> str:
    """Lightweight in-process smoke test — py_compile sweep over the main
    entry point + every loaded skill file. Does NOT spawn a staging
    instance (that's the upgrade pipeline's job). Async so the multi-MB
    skills/ directory doesn't stall the drainer."""
    bc = _bc()

    def _do():
        try:
            import py_compile
            root = os.path.dirname(os.path.abspath(bc.__file__))
            targets = [
                os.path.join(root, "bobert_companion.py"),
                os.path.join(root, "tray.py"),
                os.path.join(root, "upgrade_jarvis.py"),
                os.path.join(root, "overnight_upgrade.py"),
            ]
            skills_dir = os.path.join(root, "skills")
            if os.path.isdir(skills_dir):
                for fn in sorted(os.listdir(skills_dir)):
                    if fn.endswith(".py") and not fn.startswith("_"):
                        targets.append(os.path.join(skills_dir, fn))
            errors = []
            checked = 0
            for path in targets:
                if not os.path.exists(path):
                    continue
                checked += 1
                try:
                    py_compile.compile(path, doraise=True)
                except py_compile.PyCompileError as exc:
                    errors.append(f"{os.path.basename(path)}: {exc}")
                except OSError as exc:
                    errors.append(f"{os.path.basename(path)}: {exc!r}")
            if errors:
                return f"smoke test FAILED ({len(errors)}/{checked}): {errors[0]}"
            return f"smoke test PASSED ({checked} files clean)"
        except Exception as e:
            return f"smoke test errored: {e}"
    bc._tray_async("run_smoke_test", _do)
    return "smoke test running"


def _act_test_each_skill(_: str = "") -> str:
    """Sweep every loaded skill module and call its `selftest()` if it
    exposes one. Skills without a selftest are counted but not exercised
    — so the report says exactly which ones are silent. Async because
    selftests can do real I/O (file probes, API pings)."""
    bc = _bc()

    def _do():
        loaded = sorted(n for n in sys.modules if n.startswith("skill_"))
        if not loaded:
            return "no skills loaded"
        ok, fail, silent = [], [], []
        for name in loaded:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            fn = getattr(mod, "selftest", None)
            short = name[len("skill_"):]
            if fn is None:
                silent.append(short)
                continue
            try:
                r = fn()
                if (isinstance(r, dict) and r.get("ok") is False) or r is False:
                    fail.append(f"{short}({r})")
                else:
                    ok.append(short)
            except Exception as e:
                fail.append(f"{short}({e})")
        return (f"skills: {len(ok)} OK, {len(fail)} FAIL, "
                f"{len(silent)} no selftest "
                + (f"— failed: {', '.join(fail[:4])}" if fail else ""))
    bc._tray_async("test_each_skill", _do)
    return f"testing {len([n for n in sys.modules if n.startswith('skill_')])} skill module(s)"


# ─── Memory forget + LLM latency benchmark (Phase 4G) ──────────────────

def _entry_ts(entry: dict) -> float:
    """Numeric epoch ts of a topic/session entry, or 0.0 if absent/garbage.
    Defaulting to 0.0 means legacy entries written before the ts field are
    treated as old and thus survive a 'forget the last hour'."""
    try:
        return float(entry.get("ts", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _act_forget_last_hour(_: str = "") -> str:
    """Drop the last hour's traces from EVERY conversation store:
    bobert_memory.json topics/sessions and hidden topic sightings, the tiered
    LTM store (verbatim episodes.jsonl turn log, semantic facts created in the
    window, the in-process working turns), the voice-command pattern log and
    the monolith's in-process record of this session's opening utterances.
    Older facts/projects in bobert_memory are durable knowledge and kept;
    the ones LEARNED in the window (the LTM facts forget_since drops were
    mirrored from them) are dropped too (2026-10-01: the reply counted them
    as forgotten while every prompt still carried them). The bobert prune
    is held under _memory_lock so it can't race with learn_from_turn; the
    LTM purge runs outside it (long_term_memory takes its own lock). A
    failed purge of any store is DISCLOSED in the reply — silently leaving
    the hour on disk was the 2026-07-21 audit bug.

    2026-10-01: also forgotten -- each one survived and was read back after
    a confirmed forget: turns still queued for learning, the session-summary
    index, this process's conversation history (and the running session
    summary), and the LIVE system prompt is rebuilt at once.

    2026-10-09: and the hour of the screen memory and of the cloned voice's
    cache (_forget_screen_and_voice)."""
    bc = _bc()
    try:
        # Numeric epoch cutoff. Entries carry a float ts=time.time() written
        # at learn time; the old "%Y-%m-%d" date string compared lexically
        # against a datetime cutoff and so could NEVER drop a same-day entry
        # (a date-only prefix always sorts before "<date> HH:MM"). Keep only
        # entries strictly OLDER than one hour; anything within the window is
        # forgotten. Missing/legacy ts defaults to 0 -> treated as old -> kept.
        cutoff = time.time() - 3600
        with bc._memory_lock:
            _forget_learning_in_flight(bc, cutoff)
            mem = bc.load_memory()
            old_topics  = list(mem.get("topics") or [])
            old_sessions = list(mem.get("sessions") or [])
            kept_topics  = [t for t in old_topics
                            if _entry_ts(t) < cutoff]
            kept_sessions = [s for s in old_sessions
                             if _entry_ts(s) < cutoff]
            # The hidden topic/project SIGHTINGS (core/topic_hygiene.py) are
            # conversation traces too: left in place, a forgotten hour could
            # still promote a topic on its next mention. Mutates mem in place.
            try:
                from core import topic_hygiene as _th
                sightings = _th.forget_sightings_since(mem, cutoff)
            except Exception:
                sightings = 0
            removed = (len(old_topics) - len(kept_topics)
                       + len(old_sessions) - len(kept_sessions)
                       + sightings)
            if removed:
                mem["topics"]  = kept_topics
                mem["sessions"] = kept_sessions
                bc.save_memory(mem)
        # LTM + voice-command purges are deliberately NOT gated behind
        # removed == 0: bobert_memory can have nothing recent while the
        # episode log still holds the whole hour verbatim.
        failures = []
        ltm_counts = {}
        try:
            from core import long_term_memory as ltm
            ltm_counts = ltm.forget_since(cutoff)
        except Exception as le:
            failures.append(f"the conversation log was NOT purged ({le})")
        fcs = int(ltm_counts.get("facts", 0) or 0)
        # The facts the LTM purge dropped were mirrored from bobert_memory's
        # facts/projects (merge_memory -> _ltm_learn_facts): drop the same
        # texts there, or the "forgotten" fact stays in every system prompt.
        _gone = {t.strip().lower() for t in (ltm_counts.get("fact_texts")
                                               or [])
                 if isinstance(t, str) and t.strip()}
        if _gone:
            try:
                with bc._memory_lock:
                    mem2 = bc.load_memory()
                    changed = False
                    for key in ("facts", "projects"):
                        items = mem2.get(key)
                        if not isinstance(items, list):
                            continue
                        kept = [x for x in items
                                if not (isinstance(x, str)
                                        and x.strip().lower() in _gone)]
                        if len(kept) != len(items):
                            mem2[key] = kept
                            changed = True
                    if changed:
                        bc.save_memory(mem2)
            except Exception as fe:
                failures.append(f"the facts learned in the last hour were NOT "
                                f"removed from my main memory ({fe})")
                fcs = 0
        vc_removed = 0
        try:
            vc_removed = int(
                bc.pattern_memory.forget_voice_commands_since(cutoff))
        except Exception as ve:
            failures.append(f"the voice-command log was NOT purged ({ve})")
        # The in-process record of this session's opening utterances ("what
        # was the first thing I asked you", v2.0.148) is a conversation store
        # too, and it lives for the whole process: left alone, JARVIS would
        # confirm the forget and then recite the forgotten first request word
        # for word.
        opening_removed = 0
        try:
            _forget_opening = getattr(bc, "_forget_session_opening_since",
                                      None)
            if callable(_forget_opening):
                _n = _forget_opening(cutoff)
                if isinstance(_n, int) and not isinstance(_n, bool):
                    opening_removed = _n
        except Exception as oe:
            failures.append(
                f"this session's opening-utterance record was NOT purged "
                f"({oe})")
        # This process's conversation history -- in every LLM call, and what
        # "summarise what we talked about" reads -- and the running session
        # summary, which the next checkpoint would have re-written. BEFORE
        # the session-summary purge (2026-10-01): this bumps the summary
        # generation, so a checkpoint whose LLM call returns during the purge
        # discards its summary instead of writing the forgotten row back.
        live_removed = 0
        try:
            _forget_live = getattr(bc, "_forget_live_conversation", None)
            if callable(_forget_live):
                _n = _forget_live(cutoff)
                if isinstance(_n, int) and not isinstance(_n, bool):
                    live_removed = _n
        except Exception as ce:
            failures.append(
                f"this conversation's history was NOT cleared ({ce})")
        # The session-summary index ("what did we do this afternoon"): its
        # current-session row is re-written by the 10-minute checkpoint.
        ss_removed = 0
        try:
            _n = bc.pattern_memory.forget_session_summaries_since(cutoff)
            if isinstance(_n, int) and not isinstance(_n, bool):
                ss_removed = _n
        except Exception as se:
            failures.append(f"the session summaries were NOT purged ({se})")
        # The screen memory and the cloned voice's cache (review
        # 2026-10-09): the same hour.
        _sv_bits, _sv_fail = _forget_screen_and_voice({"seconds": 3600.0},
                                                      cutoff)
        failures.extend(_sv_fail)
        _pw = _refresh_live_prompt_after_wipe(bc)
        if _pw:
            failures.append(_pw)

        bits = []
        if removed + ss_removed:
            bits.append(f"{removed + ss_removed} item(s)")
        eps = int(ltm_counts.get("episodes", 0) or 0)
        if eps:
            bits.append(f"{eps} logged turn(s)")
        if fcs:
            bits.append(f"{fcs} fact(s)")
        if vc_removed:
            bits.append(f"{vc_removed} voice command(s)")
        bits.extend(_sv_bits)
        if opening_removed and not bits:
            # Normally the same utterances are already counted as logged
            # turns; say so only when nothing else was.
            bits.append(f"{opening_removed} recorded utterance(s)")
        warn = (" — WARNING: " + "; ".join(failures)) if failures else ""
        if not bits:
            if live_removed:
                # The history messages carry no timestamps, so the WHOLE
                # in-context conversation was cleared, however old: say that,
                # not "forgot N message(s) ... from the last hour" (2026-10-01).
                return (f"cleared this conversation's context "
                        f"({live_removed} message(s)); nothing else was "
                        f"recent enough to forget" + warn)
            return "nothing recent enough to forget" + warn
        return "forgot " + ", ".join(bits) + " from the last hour" + warn
    except Exception as e:
        return f"forget_last_hour failed: {e}"


def _act_latency_benchmark(_: str = "") -> str:
    """Time a single one-shot LLM round-trip via _llm_quick() to give
    the user a feel for current backend latency. Async — Claude
    typically replies in ~1s but Ollama on a cold model can take 5-30s."""
    bc = _bc()

    def _do():
        try:
            # Resolve the backend label INSIDE the worker, not at dispatch. The
            # old code closed over core.config's boot values, while _llm_quick
            # picks its backend live — so a local round-trip could be timed and
            # then labelled "claude/<CLAUDE_MODEL>". A benchmark that misnames
            # what it just measured is worse than no benchmark.
            t0 = time.time()
            reply = bc._llm_quick(
                system="Reply with exactly the word 'pong' and nothing else.",
                user="ping",
                max_tokens=8,
            )
            ms = (time.time() - t0) * 1000
            backend, model = _live_backend_and_model()
            head = (reply or "").strip().splitlines()[0] if reply else "(no reply)"
            return f"{backend}/{model}: {ms:.0f}ms — reply={head[:40]!r}"
        except Exception as e:
            return f"latency_benchmark failed: {e}"
    bc._tray_async("latency_benchmark", _do)
    return "latency benchmark running"


# ─── Music: iTunes search-and-play with browser-Apple-Music routing (Phase 4H) ──

def _act_play_music(args: str) -> str:
    """Play a song / artist / album by name.

    The classic local iTunes library + COM is GONE on this machine, so a
    "play <query>" request can no longer search a local library. It now
    routes to the EXISTING browser ``apple_music`` action, which plays on
    music.apple.com (already working). Field prefixes are handled gracefully:

      play_music, Earth Song              → apple_music("Earth Song")
      play_music, artist:Michael Jackson  → apple_music("Michael Jackson")
      play_music, song:Smooth Criminal    → apple_music("Smooth Criminal")
      play_music, album:Thriller          → apple_music("Thriller")
      play_music, library:Earth Song      → honest note (local library gone)
                                            + plays via Apple Music instead

    The dead ``_play_music_core`` (iTunes COM search) is intentionally NOT
    the primary path anymore — it only ever returns the COM-unavailable
    error if hit.
    """
    stripped = args.strip()
    if not stripped:
        return "format: play_music, <song/artist/album name>"

    # `library:` used to FORCE the local iTunes library. That library is gone,
    # so say so honestly, strip the prefix, and stream it via Apple Music.
    m = re.match(r"^library:\s*(.+)$", stripped, re.IGNORECASE)
    if m:
        query = m.group(1).strip()
        am_reply = _act_apple_music(query)
        return (
            "Your local iTunes library is no longer available, sir — playing "
            f"it via Apple Music instead. {am_reply}"
        )

    # Strip an artist/song/album/track field prefix (the browser action
    # searches all fields anyway) and route to music.apple.com.
    m = re.match(r"^(?:artist|song|album|track):\s*(.+)$", stripped,
                 re.IGNORECASE)
    query = m.group(1).strip() if m else stripped
    return _act_apple_music(query)


# ─── Webcam awareness (Phase 4H) ───────────────────────────────────────

def _act_where_is_user(_: str = "") -> str:
    """Returns which cameras can currently see the user's face."""
    bc = _bc()
    from core.config import CAMERAS
    if not CAMERAS:
        return "no cameras configured"
    with bc._camera_state_lock:
        now = time.time()
        # Snapshot: for each configured camera, when did it last see the user?
        report = []
        for cam in CAMERAS:
            seen = bc._camera_last_seen.get(cam["index"], 0.0)
            age = now - seen if seen else None
            if age is None:
                state = "never seen user"
            elif age < 3.0:
                state = "sees user NOW (face visible)"
            elif age < 10.0:
                state = f"saw user {age:.1f}s ago"
            else:
                state = f"no face for {age:.0f}s"
            err = bc._camera_last_read_error.get(cam["index"])
            if err:
                err_at = bc._camera_last_read_error_at.get(cam["index"], 0.0)
                err_age = now - err_at if err_at else 0.0
                state = f"{state} (I/O issue {err_age:.0f}s ago: {err})"
            report.append(f"  {cam['label']} (index {cam['index']}): {state}")

    # Summarize current direction
    visible = [cam for cam in CAMERAS
               if bc._camera_last_seen.get(cam["index"], 0.0) > now - 3.0]
    if not visible:
        summary = "User is NOT currently visible to any camera."
    elif len(visible) == len(CAMERAS):
        summary = "User is visible to ALL cameras — likely facing forward (center monitor)."
    else:
        labels = [cam["label"] for cam in visible]
        summary = f"User is visible only to: {', '.join(labels)}"

    return summary + "\n\nPer-camera detail:\n" + "\n".join(report)


# ─── Vision: see_screen with multi-monitor capture (Phase 4H) ──────────

# A see_screen "question" that is really a file reference: the model chained
# [ACTION: screenshot] → [ACTION: see_screen, screenshot_20261001_204510.png]
# (live 2026-10-01 20:45, "read this page for me and see if there's any
# issues"); the vision model then answered "you haven't provided the image
# file ...", JARVIS said the screenshot didn't come through, and the chain
# re-captured until its depth cap. The capture is always fresh, so a file
# name carries no meaning here — ask the owner's own words instead.
_SEE_SCREEN_FILE_REF_RE = re.compile(
    r"(?ix) (?:^|[\s\\/\"'(])"
    r"(?: screenshot_\d{8}_\d{6}(?:\.\w+)?"
    r"  | [\w.-]+\.(?:png|jpe?g|bmp|gif|webp|tiff?) )"
    r"[\s\"').,]*$"
    r"| ^\s*[a-z]:[\\/]")
_SEE_SCREEN_DEFAULT_Q = (
    "Describe in detail what is currently on the screen, and point out "
    "anything that looks wrong (errors, warnings, typos, broken layout).")


# S3 (2026-10-02). Live 16:13-16:14: the model passed a bare URL as the
# question ([ACTION: see_screen, https://...]) and vision answered "which
# monitor has this URL" - from the URL text in JARVIS's own console window;
# then "Jarvis, continue" with no question became 'The owner asked: "Jarvis
# continued."', and vision answered from the CHAT WINDOW on another monitor.
# A bare URL, a control word or nothing at all is not a question about the
# page: ask about the page JARVIS opened instead.
_SEE_SCREEN_URL_RE = re.compile(
    r"(?i)^\s*(?:https?://\S+|www\.\S+|[\w-]+(?:\.[\w-]+)+(?:[/?#]\S*)?)\s*$")
# Words that steer the turn but ask nothing ("continue", "go on", "try
# again"); Parakeet writes "continue" as "continued".
_CONTROL_WORDS = frozenset({
    "continue", "continued", "continuing", "go", "on", "ahead", "keep",
    "going", "carry", "try", "again", "proceed", "next", "resume", "do", "it",
    "okay", "ok", "yes", "yeah", "yep", "sure", "please", "now", "then",
    "and", "jarvis", "sir", "alright", "right", "so", "well", "same",
    "one", "more", "time",
})
# Words that alone are only an address, never a steer.
_CONTROL_ADDRESS = frozenset({"jarvis", "sir", "and"})
# Whole steering phrases the word list can't hold without swallowing real
# questions (review 2026-10-02): "keep trying", "continue what you were
# doing", "carry on with it", "pick up where you left off", "finish it".
_CONTROL_PHRASE_RE = re.compile(
    r"(?i)^(?:(?:jarvis|sir|please|ok(?:ay)?|alright|yes|yeah|and|so|now|"
    r"then|just)\W+)*"
    r"(?:continue|carry\s+on|go\s+on|go\s+ahead|keep\s+(?:going|trying|at\s+it)"
    r"|proceed|resume|try\s+(?:it\s+|that\s+)?again|"
    r"finish\s+(?:it|that|up|the\s+job)|"
    r"pick\s+(?:it\s+)?up\s+where\s+you\s+left\s+off|do\s+it)"
    r"(?:\W+(?:with\s+(?:it|that|this)|what\s+you\s+were\s+doing|"
    r"where\s+you\s+left\s+off|from\s+there|please|sir|jarvis|then|now|"
    r"again))*\W*$")
_SEE_SCREEN_PAGE_Q = (
    "What is on the screen in the browser window showing {page}? Read the "
    "main content, search results or error messages.")
_VISION_CHAT_GUARD = (
    " Ignore chat and assistant windows (the Claude app, the JARVIS console, "
    "terminals) unless the question is about them.")
# A question ABOUT a chat / messaging / terminal window keeps the note out
# (the Teams nudger asks about Teams' chat sidebar). Not "jarvis" (review
# 2026-10-02): in wake-word mode nearly every owner question starts "Jarvis,
# ...", which dropped the note from exactly the looks live vision answered
# from the chat window ("the JARVIS console" still counts, via "console").
# Not a bare "log" either: "log in" is a page's button, not a log window.
_CHAT_TOPIC_RE = re.compile(
    r"(?i)\b(?:chats?|assistant|claude|console|terminals?|powershell|"
    r"command\s+prompt|transcript|logs|log\s+(?:file|window|output)|teams|"
    r"slack|discord|messenger|e-?mail|inbox|mail)\b")


def _is_control_utterance(text) -> bool:
    """True for words that steer the turn but ask nothing: "Jarvis,
    continue.", "go on", "try again please". Never raises."""
    try:
        s = " ".join(str(text or "").split())
        words = re.findall(r"[a-z']+", s.lower())
        if (words and all(w in _CONTROL_WORDS for w in words)
                and any(w not in _CONTROL_ADDRESS for w in words)):
            return True
        return bool(words) and bool(_CONTROL_PHRASE_RE.match(s))
    except Exception:
        return False


def _see_screen_plan(raw: str, user_text: str = "", opened=None):
    """(question, page) for see_screen. ``page`` is the URL / name of the
    page the question is about when the question was REWRITTEN to the page
    question (a bare URL, a control word or nothing, with a page to ask
    about), else None. ``opened``: the core.opened_ledger entry of the page
    JARVIS opened last, or None.

      * a real question passes through untouched;
      * a bare URL -> the page question about that URL;
      * empty / a file name / a control word -> the owner's own words when
        they ask something; for a control word ("continue") the page JARVIS
        opened last; else the generic describe-the-screen default."""
    q = " ".join((raw or "").split())
    target = getattr(opened, "target", "") or ""
    file_ref = bool(q) and bool(_SEE_SCREEN_FILE_REF_RE.search(q))
    if q and not file_ref and _SEE_SCREEN_URL_RE.match(q):
        return _SEE_SCREEN_PAGE_Q.format(page=q), q
    if q and not file_ref and not _is_control_utterance(q):
        return q, None
    ut = " ".join((user_text or "").split())
    if ut and not _is_control_utterance(ut):
        return (f'The owner asked: "{ut}". Answer that from what is on '
                "the screen."), None
    if target and (ut or q):
        return _SEE_SCREEN_PAGE_Q.format(page=target), target
    return _SEE_SCREEN_DEFAULT_Q, None


def _see_screen_question(raw: str, user_text: str = "", opened=None) -> str:
    """The question to ask vision (see _see_screen_plan)."""
    return _see_screen_plan(raw, user_text, opened)[0]


def _with_chat_guard(q: str) -> str:
    """``q`` plus the ignore-the-chat-windows note, unless ``q`` is about a
    chat / assistant / terminal window itself."""
    if _CHAT_TOPIC_RE.search(q or ""):
        return q
    return (q or "").rstrip() + _VISION_CHAT_GUARD


def _page_wall_result(answer, opened, bc=None, png=None) -> str:
    """S5: when the page JARVIS opened on an account streaming service is a
    sign-in wall ("Sign In", "Log in") or an error page ("Oops ... isn't
    working"), the one plain TERMINAL line that ends the turn - no clicking
    around a page that cannot play; else "". Never raises.

    The free ``answer`` is only a HINT (core.streaming_search.wall_kind
    counts any "Sign In", a footer link under real results included - review
    2026-10-02). A hint is CONFIRMED with the strict SIGNIN / ERROR / OK look
    (wall_question) at the same image ``png``, the one _streaming_page_wall
    asks. Only a service whose _STREAMING_SERVICES entry carries
    "sign_in_check" is checked: YouTube plays signed out and always shows
    "Sign in", so a look at a YouTube page is never a wall."""
    try:
        from core import streaming_search as _ss
        from core.failure_markers import TERMINAL_FAILURE_PREFIX
        key = _ss.service_for_url(getattr(opened, "target", "") or "")
        if not key or bc is None or png is None:
            return ""
        table = getattr(bc, "_STREAMING_SERVICES", None)
        cfg = table.get(key) if isinstance(table, dict) else None
        if not (isinstance(cfg, dict) and cfg.get("sign_in_check") is True):
            return ""
        if not _ss.wall_kind(answer):
            return ""
        kind = _ss.parse_wall_verdict(bc.ask_vision(_ss.wall_question(key), png))
        if kind not in ("sign_in", "error"):
            return ""
        print(f"  [vision] {_ss.service_name(key)} page is a {kind} wall - "
              "stopping", flush=True)
        return TERMINAL_FAILURE_PREFIX + _ss.wall_line(key, kind)
    except Exception:
        return ""


# "Read this page" means the window he is looking at (NEW #11, 2026-10-02).
# Live 20:45-21:00 (2026-10-01) see_screen logged "Capturing all 4 monitors"
# and sent four 1024-px shots to local vision in one call: the page he meant
# was shrunk to a corner of a composite, a poor way to read text. When his
# words name a page / window / tab / article / document, only the focused
# window is captured, through _capture_focused_window_png (privacy-gated), at
# the size of a maximised window on a 2560-px monitor (its rect includes the
# frame) - no downscale. Anything else keeps the all-monitor capture.
_SEE_SCREEN_FOCUSED_RE = re.compile(
    r"(?i)\b(?:this|that|the|my|current|open)\s+(?:current\s+|open\s+)?"
    r"(?:web\s?page|web\s+site|website|page|window|tab|article|site|"
    r"document|doc|pdf|e-?mail|post|thread)s?\b")
_SEE_SCREEN_WINDOW_MAX_DIM = 2600
_SEE_SCREEN_WINDOW_LABEL = "the focused window"


def _see_screen_wants_focused_window(user_text: str, question: str) -> bool:
    """True when the owner's words (else the model's question) name the
    page / window he is on. Never raises."""
    try:
        said = " ".join(str(user_text or "").split())
        return bool(_SEE_SCREEN_FOCUSED_RE.search(said or str(question or "")))
    except Exception:
        return False


def _focused_window_is_jarvis(bc) -> bool:
    """True when the focused window is JARVIS's own (a request typed into
    the dashboard): its page is not the one he means. Never raises."""
    try:
        title = None
        try:
            _h, title, _r = bc._read_focused_window()
        except Exception:
            title = None
        if not isinstance(title, str) or not title:
            title = (bc._focused_window_state or {}).get("title")
        return isinstance(title, str) and "jarvis" in title.lower()
    except Exception:
        return False


def _see_screen_focused_window(bc, q: str):
    """Ask vision about the focused window only. The answer, or None when
    the capture is unavailable (the caller then captures every monitor)."""
    if _focused_window_is_jarvis(bc):
        print("  [vision] focused window is JARVIS's own - capturing all "
              "monitors instead", flush=True)
        return None
    try:
        png = bc._capture_focused_window_png(
            max_dim=_SEE_SCREEN_WINDOW_MAX_DIM)
    except Exception as e:
        print(f"  [vision] focused-window capture failed: {e}", flush=True)
        return None
    if not isinstance(png, (bytes, bytearray)) or not png:
        print("  [vision] focused-window capture unavailable - capturing all "
              "monitors instead", flush=True)
        return None
    print("  [vision] Capturing the focused window (full resolution)...",
          flush=True)
    result = bc.ask_vision(q, png)
    print(f"  [vision] Got answer ({len(result)} chars)", flush=True)
    bc._push_screen_context(_SEE_SCREEN_WINDOW_LABEL, q, result,
                            {_SEE_SCREEN_WINDOW_LABEL: png})
    return result


# ── see_screen reads TEXT first (2026-10-05) ────────────────────────────
# Live 00:28:01 see_screen sent four monitor shots to the local model and
# got "the YouTube page displays several video thumbnails and categories" -
# not one title, while every title sat in the page's accessibility tree. A
# READING question (anything not about colours / pictures / layout) is now
# answered from the windows' own text (core.screen_digest: UI Automation,
# OCR when thin): no vision call, not counted against the see_screen budget.
# A VISUAL question still gets ONE look, with that text beside it.
_screen_digest_backend: list = [None]       # tests inject a fake backend


def _see_screen_text(bc, monitor, said, question, page_now):
    """(text, scope, hwnds) - the digest answer for a reading question, or
    None when nothing could be read (the caller falls back to vision)."""
    try:
        from core import screen_digest as _sd
        from core import monitor_geometry as _mg
        from core.config import MONITORS
        mon = monitor or _mg.monitor_named_in(said or question or "", MONITORS)
        backend = _screen_digest_backend[0]
        if mon:
            d = _sd.digest("monitor", said=said, monitor=mon, backend=backend)
            scope = f"the {mon} monitor"
        elif _see_screen_wants_focused_window(said, question) and \
                not _focused_window_is_jarvis(bc):
            hwnd = None
            try:
                hwnd = bc._read_focused_window()[0]
            except Exception:
                hwnd = None
            d = _sd.digest("window", said=said, hwnd=hwnd, backend=backend)
            scope = "the focused window"
        elif page_now is not None and getattr(page_now[0], "hwnd", None):
            d = _sd.digest("window", said=said, hwnd=page_now[0].hwnd,
                           backend=backend)
            scope = "the page I opened"
        else:
            d = _sd.digest("overview", said=said, backend=backend)
            scope = "every monitor"
        if not d.get("windows"):
            return None
        return d["text"], scope, d["windows"]
    except Exception:
        return None


def _record_look(bc, said, scope, text, source="look") -> None:
    """A look's text into the screen timeline, the vision trace and the
    turn's screen texts. Never raises."""
    try:
        from core import screen_timeline as _tl
        _tl.add(source="look", title=f"see_screen: {scope}", text=text)
    except Exception:
        pass
    try:
        from core import vision_trace as _vt
        _vt.record("see_screen", utterance=said, outcome="read",
                   source=source, scope={"monitor": scope}, raw_answer=text)
    except Exception:
        pass
    _note_screen_look("see_screen", text)


def _act_see_screen(question: str) -> str:
    bc = _bc()
    # Privacy gate: refuse (spoken) before spending the per-intent budget if a
    # SCREENSHOT_PRIVACY_BLOCKLIST window is focused. take_screenshot() also
    # hard-blocks, but checking here returns the in-character line instead of
    # the generic "could not capture any monitor".
    if bc.screenshot_privacy_block_reason():
        return bc.SCREENSHOT_PRIVACY_REFUSAL
    from core.config import MONITORS
    monitor, question = bc._parse_monitor_prefix(question)
    try:
        _ut = bc._turn_user_text()
    except Exception:
        _ut = ""
    # The page JARVIS opened last, while its window still exists (S3/S4/S5).
    _page_now = _opened_page_now()
    _opened = _page_now[0] if _page_now else None
    q, _page = _see_screen_plan(question, _ut if isinstance(_ut, str) else "",
                                _opened)
    _said = _ut if isinstance(_ut, str) else ""

    # Text first (2026-10-05): a READING question is answered from the
    # windows' own text, with no picture. A sign-in-checked streaming page
    # JARVIS opened keeps its look (S5's wall check reads the image).
    from core.screen_digest import VISUAL_Q_RE
    _visual = bool(VISUAL_Q_RE.search(q or "") or VISUAL_Q_RE.search(_said))
    _wall_page = False
    try:
        from core import streaming_search as _ss
        _key = _ss.service_for_url(getattr(_opened, "target", "") or "")
        _cfgs = getattr(bc, "_STREAMING_SERVICES", None)
        _wall_page = bool(_key and isinstance(_cfgs, dict)
                          and (_cfgs.get(_key) or {}).get("sign_in_check"))
    except Exception:
        _wall_page = False
    _digest = None
    try:
        from core import config as _cfg_mod
        _uia_on = bool(getattr(_cfg_mod, "SCREEN_UIA_ENABLED", True))
    except Exception:
        _uia_on = True
    if _uia_on and not _wall_page:
        _digest = _see_screen_text(bc, monitor, _said, question, _page_now)
    if _digest is not None and not _visual:
        text, scope, _hw = _digest
        print(f"  [vision] read {scope} as text ({len(text)} chars, no "
              "picture)", flush=True)
        bc._push_screen_context(scope, q, text, {})
        _record_look(bc, _said, scope, text, source="uia")
        return ("Read from the screen's own text (exact titles; quote them, "
                "invent nothing):\n" + text)
    if _digest is not None:
        # A visual question: ONE look, with the page's text beside it.
        q = (q + "\n\nThe windows' own text (quote titles exactly from "
             "this):\n" + _digest[0][:1500])

    # Per-intent budget guard. parse_and_run_actions resets the counter at
    # the start of every dispatch; once exhausted, refuse with a hint that
    # steers the LLM toward recall_screen or drafting from cached data
    # instead of re-capturing the same screen for the Nth time.
    used = getattr(bc._see_screen_budget_state, "used", 0)
    if used >= bc.SEE_SCREEN_BUDGET_PER_INTENT:
        print(
            f"  [vision] see_screen budget exhausted "
            f"({used}/{bc.SEE_SCREEN_BUDGET_PER_INTENT}) — refusing fresh capture",
            flush=True,
        )
        return (
            f"see_screen budget for this intent is exhausted "
            f"({used}/{bc.SEE_SCREEN_BUDGET_PER_INTENT} captures used). "
            "Use recall_screen to query the cached visual state from the "
            "captures already taken, or draft your reply from the data you "
            "already have rather than re-capturing the screen. If the user "
            "issues a fresh request the budget will reset."
        )
    bc._see_screen_budget_state.used = used + 1

    # "Read this page" / "this window": the focused window only, at full
    # resolution (NEW #11). Falls through to every monitor when unavailable.
    if monitor is None and _see_screen_wants_focused_window(
            _ut if isinstance(_ut, str) else "", question):
        result = _see_screen_focused_window(bc, q)
        if result is not None:
            return result

    # The page question about the page JARVIS opened: only the monitor that
    # page is on (S4) - one image, no chat window beside it to answer from.
    _about_opened = bool(_page and _opened is not None
                         and _same_site(_page, _opened.target))
    if monitor is None and _about_opened and _page_now[1] in MONITORS:
        monitor = _page_now[1]
        print(f"  [vision] looking at the {monitor} monitor (where I opened "
              "the page)", flush=True)

    # Default behaviour: no specific monitor requested -> capture every
    # monitor in MONITORS and send them all to vision in one call.
    if monitor is None:
        print(f"  [vision] Capturing all {len(MONITORS)} monitors...", flush=True)
        images = bc.take_all_monitor_screenshots()
        if not images:
            return "could not capture any monitor"
        print(
            f"  [vision] Asking Claude about {', '.join(images.keys())}...",
            flush=True,
        )
        result = bc.ask_vision_multi(q, images)
        print(f"  [vision] Got answer ({len(result)} chars)", flush=True)
        bc._push_screen_context(None, q, result, images)
        return result

    print(f"  [vision] Capturing screen ({monitor} monitor)...", flush=True)
    png = bc.take_screenshot(monitor=monitor)
    if png is None:
        return "could not capture screen"
    print("  [vision] Asking Claude (this takes a few seconds)...", flush=True)
    result = bc.ask_vision(_with_chat_guard(q), png)
    print(f"  [vision] Got answer ({len(result)} chars)", flush=True)
    bc._push_screen_context(monitor, q, result, {monitor: png})
    if _about_opened:
        _wall = _page_wall_result(result, _opened, bc=bc, png=png)
        if _wall:
            return _wall
    return result


def _same_site(a, b) -> bool:
    """True when two URLs (or a URL and a bare host) are on the same site
    (the same streaming service, or the same host without "www.")."""
    try:
        from core import streaming_search as _ss
        ka, kb = _ss.service_for_url(a), _ss.service_for_url(b)
        if ka or kb:
            return ka == kb
        ha = urllib.parse.urlsplit(a if "://" in a else "https://" + a).hostname
        hb = urllib.parse.urlsplit(b if "://" in b else "https://" + b).hostname
        strip = (lambda h: (h or "").lower().removeprefix("www."))
        return bool(ha) and strip(ha) == strip(hb)
    except Exception:
        return False


# ─── Replay last non-destructive action (Phase 4H) ─────────────────────

# "do that again" re-fires only an action this recent (seconds). Two minutes
# covers "do that again on the left monitor" right after the first run, and
# no more (2026-10-01).
_REPLAY_MAX_AGE_S = 120.0


def _act_replay_last_action(arg: str = "") -> str:
    """Re-fire the most recent non-destructive action.

    Triggered by the voice phrases 'do that again' / 'replay that' / 'do it
    again'. If `arg` is non-empty it's treated as a monitor identifier
    ('left', 'right', etc.) and substituted into the target action's arg
    where the shape is known.

    Destructive actions (close_window, kill_process, restart, upgrade,
    start_overnight_upgrade, run_shell) are refused — the user must re-issue
    the command so it goes through the normal confirmation/pushback path.

    Only a RECENT action replays (2026-10-01): the history kept no age limit,
    so "do that again" could re-fire whatever ran hours earlier. A draft send
    is refused too: this path calls the handler directly, past the read-back
    gate parse_and_run_actions applies (the monolith also stops recording
    them; this is the second lock on the same door).

    The handler runs through the monolith's one runner, _run_draft_gated
    (2026-10-05). Returns "" when a self-voiced action did its own talking
    (nothing more to say) and a terminal failure whole (the owner's
    sentence, voiced word for word by every caller).
    """
    bc = _bc()
    with bc._action_history_lock:
        if not bc._action_history:
            return "no previous action to replay"
        last = dict(bc._action_history[-1])

    name = last.get("action", "")
    orig_arg = last.get("arg", "") or ""

    try:
        age = time.time() - float(last.get("at") or 0.0)
    except (TypeError, ValueError):
        age = float("inf")
    if age > _REPLAY_MAX_AGE_S:
        return "nothing recent to replay, sir — please say the command again"

    if name in bc._DESTRUCTIVE_REPLAY_ACTIONS:
        return (f"refusing to replay destructive action '{name}' without "
                "confirmation — please re-issue the command explicitly")

    try:
        from core.draft_preview_gate import should_gate as _is_draft_send
        _draft_send = _is_draft_send(name)
    except Exception:
        _draft_send = str(name).lower().startswith("send_")
    if _draft_send:
        return (f"refusing to replay '{name}' — a draft is only sent after "
                "it has been read back; please ask me to send it again")

    fn = bc.ACTIONS.get(name)
    if fn is None:
        return f"cannot replay '{name}' — action no longer registered"

    new_arg = bc._substitute_monitor_in_arg(name, orig_arg, arg) if arg else orig_arg
    try:
        # The monolith's one runner (2026-10-05), as every path that runs an
        # action it was handed: a self-voiced action (a device chat) gets
        # its bounded wait for another chat and its honest line when it
        # says nothing. A bare fn() here bypassed both.
        res = bc._run_draft_gated(name, new_arg, fn)
    except Exception as e:
        return f"replay of '{name}' failed: {e}"
    # A terminal failure is already the owner's sentence: handed back whole,
    # so every caller voices it word for word - never "replayed x: failed
    # (final): ...". And a self-voiced action that did its own talking has
    # nothing more to say: its result is bookkeeping, never read aloud.
    from core.failure_markers import terminal_failure_text
    if terminal_failure_text(res):
        return res
    if bc._self_voiced_did_talk(name, res):
        return ""
    suffix = f" on monitor {arg.strip().lower()}" if arg else ""
    summary = res if isinstance(res, str) else str(res)
    head = summary.split("\n", 1)[0]
    if len(head) > 160:
        head = head[:157].rstrip() + "..."
    return f"replayed {name}{suffix}: {head}"


# ─── Shell command execution (Phase 4H) ────────────────────────────────

def _act_run_shell(command: str) -> str:
    """Execute a shell command via PowerShell and return its output.

    Use this when the LLM wants to run a shell command (Get-Process, python
    something.py, git status, etc.) — much safer than typing the command into
    whatever window happens to have focus via [ACTION: type, ...].
    """
    bc = _bc()
    cmd = command.strip()
    if not cmd:
        return "format: run_shell, <command>"

    low = cmd.lower()
    for bad in bc._SHELL_FORBIDDEN_PATTERNS:
        if bad in low:
            return (
                f"REFUSED: '{bad.strip()}' is on the destructive-commands blocklist. "
                f"If you really need this, ask the user to run it manually."
            )

    # Hidden console so we don't pop a terminal window on every call.
    creationflags = 0
    try:
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    except AttributeError:
        creationflags = 0

    try:
        result = subprocess.run(
            ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=bc.RUN_SHELL_TIMEOUT_SEC,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        return f"run_shell timed out after {bc.RUN_SHELL_TIMEOUT_SEC}s — command was: {cmd[:120]}"
    except FileNotFoundError:
        return "run_shell failed: powershell.exe not on PATH"
    except Exception as e:
        return f"run_shell failed: {e}"

    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if len(out) > bc.RUN_SHELL_OUTPUT_MAX_CHARS:
        out = out[:bc.RUN_SHELL_OUTPUT_MAX_CHARS] + f"\n...(truncated, {len(result.stdout)} chars total)"
    if len(err) > bc.RUN_SHELL_OUTPUT_MAX_CHARS:
        err = err[:bc.RUN_SHELL_OUTPUT_MAX_CHARS] + f"\n...(truncated, {len(result.stderr)} chars total)"

    parts = [f"exit code: {result.returncode}"]
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    if not out and not err:
        parts.append("(no output)")
    return "\n".join(parts)


# ─── Webcam snapshot + vision (Phase 4I) ───────────────────────────────

def _act_see_user(camera_hint: str = "") -> str:
    """Take a snapshot from a webcam and ask Claude vision to describe the user."""
    import cv2
    bc = _bc()
    from core.config import CAMERAS
    if not CAMERAS:
        return "no cameras configured"

    with bc._camera_state_lock:
        # Pick which camera frame to use: prefer the one that most recently saw a face
        best_idx = None
        best_seen = 0.0
        for cam in CAMERAS:
            seen = bc._camera_last_seen.get(cam["index"], 0.0)
            if seen > best_seen:
                best_seen = seen
                best_idx = cam["index"]
        # Fallback: any camera with a cached frame
        if best_idx is None and bc._camera_latest_frame:
            best_idx = next(iter(bc._camera_latest_frame))
        if best_idx is None:
            # Surface the most recent I/O failure so the LLM gets context
            last_err = None
            for idx, msg in bc._camera_last_read_error.items():
                last_err = (idx, msg)
                break
            if last_err is not None:
                return (f"no webcam frames available yet — camera {last_err[0]} "
                        f"last reported: {last_err[1]}")
            return "no webcam frames available yet — face tracker may not have started"

        frame = bc._camera_latest_frame.get(best_idx)
        last_frame_at = bc._camera_last_frame_at.get(best_idx, 0.0)
        last_err      = bc._camera_last_read_error.get(best_idx)
        if frame is None:
            return "no frame cached for that camera"
        frame = frame.copy()
        frame_age = time.time() - last_frame_at if last_frame_at else None

    print(f"  [vision] Looking at user via camera {best_idx}...", flush=True)
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        return "failed to encode webcam frame"
    png_bytes = buf.tobytes()
    question = camera_hint.strip() or (
        "Describe the person visible in this webcam image — what they're doing, "
        "their posture, expression, what they're wearing, and anything notable "
        "in the background."
    )
    print("  [vision] Analyzing user image...", flush=True)
    result = bc.ask_vision(question, png_bytes)
    print("  [vision] Got description", flush=True)
    if frame_age is not None and frame_age > 5.0:
        note = (f"(note: camera {best_idx} frame is {frame_age:.1f}s old; "
                f"last read error: {last_err})" if last_err else
                f"(note: camera {best_idx} frame is {frame_age:.1f}s old)")
        result = f"{result}\n\n{note}"
    return result


def _kinect_gaze_which_monitor() -> str | None:
    """If Kinect head-direction gaze (KINECT_GAZE_ENABLED) has a FRESH read of
    which monitor the owner faces, return a 'facing X monitor' string; else
    None. This is the PRIMARY which-monitor path — it works with the WEBCAMS OFF
    because it reads the owner's head yaw from the Kinect skeleton, not a camera.

    Delegates to the face_tracker skill (the single source of truth for the
    yaw→monitor mapping + calibration + freshness window). NEVER raises — any
    miss returns None so the caller falls back to the camera heuristic below."""
    try:
        ft = sys.modules.get("skill_face_tracker")
        if ft is None:
            return None
        getter = getattr(ft, "_kinect_gaze_monitor", None)
        if not callable(getter):
            return None
        monitor = getter(time.time())
        if not monitor or monitor == "away":
            return None
        from core.config import MONITORS
        suffix = f" ({monitor})" if monitor in (MONITORS or {}) else ""
        return f"facing {monitor.upper()} monitor{suffix} (Kinect head-direction)"
    except Exception:
        return None


def _act_which_monitor(_: str = "") -> str:
    """Determine which monitor the user is currently looking at.
    Strategy:
      • PRIMARY (KINECT_GAZE_ENABLED): the Kinect reads the owner's HEAD
        DIRECTION (facing yaw) and maps it to a monitor — works with BOTH
        WEBCAMS OFF. Used whenever it has a fresh reading.
      • FALLBACK (webcam heuristic), used when gaze is off / the Kinect has no
        body in view:
          • If only the LEFT camera sees the face   -> "left" monitor
          • If only the RIGHT camera sees the face  -> "right" monitor
          • If BOTH cameras see the face            -> middle area, then use
            Claude vision to check if the user is tilting their head UP toward
            the top monitor or looking forward at the middle monitor.
          • If NO camera sees the face              -> "user not visible"
    """
    # PRIMARY: Kinect head-direction gaze (webcam-free).
    gaze = _kinect_gaze_which_monitor()
    if gaze is not None:
        return gaze

    import cv2
    bc = _bc()
    from core.config import CAMERAS, MONITORS
    if not MONITORS:
        return "no MONITORS configured (run --list-monitors and add them to the script)"

    now = time.time()
    with bc._camera_state_lock:
        visible_indexes = [
            cam["index"] for cam in CAMERAS
            if bc._camera_last_seen.get(cam["index"], 0.0) > now - 3.0
        ]
        frame_for_vision = None
        if visible_indexes:
            frame_for_vision = bc._camera_latest_frame.get(visible_indexes[0])
            if frame_for_vision is not None:
                frame_for_vision = frame_for_vision.copy()

    if not visible_indexes:
        return "user is not visible to any camera — can't determine monitor"

    # Use the monolith's canonical side rule (label-first, look_x<=0.5 = left),
    # not a raw `look_x < 0.5` — that was the 4th surviving copy of the stale
    # rule the other three call sites already migrated off, and it mislabels the
    # live LEFT camera whose look_x is exactly 0.5. 2026-07-14 bug-hunt.
    _side = getattr(bc, "_percam_side", None)
    if callable(_side):
        cam_sides = {cam["index"]: _side(cam) for cam in CAMERAS}
    else:
        cam_sides = {
            cam["index"]: ("left" if cam.get("look_x", 0.5) <= 0.5 else "right")
            for cam in CAMERAS
        }
    sides_seen = {cam_sides[i] for i in visible_indexes}

    if sides_seen == {"left"}:
        target = "left" if "left" in MONITORS else None
        return "facing LEFT monitor" + (f" ({target})" if target else "")
    if sides_seen == {"right"}:
        target = "right" if "right" in MONITORS else None
        return "facing RIGHT monitor" + (f" ({target})" if target else "")

    # Both sides see user -> middle area. Use vision to disambiguate middle vs top.
    if frame_for_vision is None or "top" not in MONITORS:
        return "facing middle/forward (top monitor not configured)"

    print("  [vision] Checking head tilt for top monitor...", flush=True)
    ok, buf = cv2.imencode(".png", frame_for_vision)
    if not ok:
        return "facing middle (couldn't check head tilt)"
    answer = bc.ask_vision(
        "Look at this person's head and eyes. Are they looking STRAIGHT FORWARD "
        "at the camera level, looking UPWARD (head tilted up, eyes looking high), "
        "or looking DOWNWARD? Reply with exactly one word: UP, FORWARD, or DOWN.",
        buf.tobytes(),
    )
    answer_upper = answer.upper()
    if "UP" in answer_upper:
        return "facing TOP monitor (head tilted up)"
    return "facing MIDDLE monitor (looking forward)"


# ─── Session memory recall (Phase 4I) ──────────────────────────────────

# "Summarise / recap what we talked about", "what have we discussed" — a request
# about THIS conversation, which is in memory (conversation_history), not in the
# prior-session index. 2026-09-29, live: "summarize what we talked about today"
# got "I can only recall specific past conversations if you ask me about them
# directly" — a false decline for a conversation JARVIS was holding.
_CONVO_SUMMARY_RE = re.compile(
    r"\b(?:summari[sz]e|summary|recap|sum\s+up|go\s+over|run\s+(?:me\s+)?through)\b"
    r".{0,40}?\b(?:talk(?:ed|ing)?|discuss(?:ed|ing)?|conversation|chat(?:ted)?|"
    r"said|covered)\b"
    r"|\bwhat\s+(?:have|did)\s+we\s+(?:been\s+)?"
    r"(?:talk(?:ed|ing)?\s+about|discuss(?:ed|ing)?|cover(?:ed)?)\b",
    re.IGNORECASE)

# A window BEFORE this session — those questions stay on the summary index.
_PAST_WINDOW_RE = re.compile(
    r"\b(?:yesterday|last\s+(?:night|week|time|session)|days?\s+ago|"
    r"the\s+other\s+day|monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday|this\s+week|previous\s+session)\b",
    re.IGNORECASE)

_ACTION_TOKEN_RE = re.compile(r"\[\s*ACTION\s*:[^\]]*\]", re.IGNORECASE)


def _conversation_summary_requested(*texts: str) -> bool:
    """True when any of `texts` asks for a summary of the CURRENT
    conversation — a summary request with no reference to an earlier window."""
    joined = " ".join(t for t in texts if t)
    return (bool(joined) and any(_CONVO_SUMMARY_RE.search(t) for t in texts if t)
            and not _PAST_WINDOW_RE.search(joined))


def _summarise_current_conversation(bc, texts: tuple) -> str:
    """Summarise this session's conversation_history in JARVIS voice.

    The request that produced this action is dropped from the transcript first
    (its own user line and the assistant reply carrying the token), so "we just
    started" is judged on what came BEFORE it. When he says "today" (or "this
    morning", "earlier"), earlier session summaries from today are folded in,
    because a restart empties conversation_history but not his day.

    Same LLM path as every other recall here and as the in-session checkpoint
    (bc._llm_quick — local-first per model_route('ambient')), which already
    summarises this same transcript every 10 minutes; nothing new leaves the box."""
    hist = getattr(bc, "conversation_history", None)
    msgs = [m for m in (list(hist) if isinstance(hist, list) else [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str) and m["content"].strip()]
    if msgs and msgs[-1]["role"] == "assistant" \
            and "session_memory_recall" in msgs[-1]["content"]:
        msgs.pop()
    if msgs and msgs[-1]["role"] == "user" and (
            msgs[-1]["content"].strip() in texts
            or _conversation_summary_requested(msgs[-1]["content"])):
        msgs.pop()

    earlier: list = []
    if any(re.search(r"\b(?:today|this\s+(?:morning|afternoon)|earlier)\b",
                     t or "", re.IGNORECASE) for t in texts):
        try:
            got = bc.pattern_memory.get_session_summaries("today", limit=8)
            earlier = [s for s in got if isinstance(s, dict)
                       and isinstance(s.get("summary"), str)
                       and s["summary"].strip()] if isinstance(got, list) else []
        except Exception:
            earlier = []

    if not any(m["role"] == "user" for m in msgs) and not earlier:
        return ("We've only just started this session, sir — there's nothing "
                "to summarise yet.")

    lines = []
    for m in msgs:
        text = " ".join(_ACTION_TOKEN_RE.sub("", m["content"]).split())
        if text:
            lines.append(f"{m['role'].title()}: {text[:500]}")
    context = "Conversation so far this session (oldest first):\n" + (
        "\n".join(lines) if lines else "(nothing yet — this session just started)")
    if earlier:
        context += "\n\nEarlier sessions today (summaries, newest first):\n" + \
            "\n".join(f"- {s['summary'].strip()}" for s in earlier)

    system = (
        "You are J.A.R.V.I.S. summarising, for the user (sir), the conversation "
        "you have had with him. Two to four sentences in JARVIS voice — "
        "composed, British, dry. Cover the main topics in order and anything "
        "decided or left open. Use ONLY what is in the transcript and notes "
        "provided; never invent a topic. If there is very little, say so in one "
        "sentence. No preamble, no bullet points, no closing question."
    )
    try:
        reply = (bc._llm_quick(system=system, user=context, max_tokens=220)
                 or "").strip()
    except Exception as e:
        return f"conversation summary LLM call failed: {e}"
    if not reply:
        return "I couldn't produce a summary of our conversation just now, sir."
    return reply


def _act_session_memory_recall(args: str = "") -> str:
    """Query the session_summaries.json index and return a one-line
    JARVIS-voice answer about what the user was doing in a given time window.

    Triggered by phrases like 'what did we do yesterday', 'remind me what I
    was working on last night', 'what happened this morning'. The free-text
    query (typically the user's full utterance) is parsed for a time
    reference; matching session summaries are then handed to the LLM with a
    JARVIS-voice prompt for a 1-2 sentence reply.

    SUMMARY MODE (2026-09-29): 'summarize what we talked about (today)',
    'recap our conversation', 'what have we discussed' summarise THIS
    session's conversation_history instead (see
    _summarise_current_conversation). Decided from the argument AND the
    owner's own words for this turn, so a bare token still gets it right.

    "What did I just ask you" / "what was my last question" is about THIS
    conversation, not the session index: it is answered deterministically from
    conversation_history with the most recent PRIOR owner utterance. By the
    time this action runs on the LLM path, _call_llm has already appended the
    CURRENT utterance, so that entry is skipped (fast_paths.recall_turn_-
    recorded), as is any earlier "what did I just ask" question. The index +
    LLM route recalled the current question itself ("You just asked me what
    you had previously asked me, sir.", live 2026-09-29)."""
    bc = _bc()
    from core.owner_turn import current_owner_utterance
    query = (args or "").strip()
    utterance = current_owner_utterance(bc)
    if _conversation_summary_requested(query, utterance):
        return _summarise_current_conversation(bc, (query, utterance))
    if not query:
        # A bare token: his own words carry the time reference.
        query = utterance
    try:
        from core import fast_paths as _fp
        # "What was the first thing I asked you (today)" (v2.0.148): the
        # session's opening utterances the monolith records, not the trimmed
        # history (live v2.0.140 this route said it had no access). A
        # paraphrase in the argument counts; so do his own words.
        _turns = getattr(bc, "_session_opening_turns", None)
        if isinstance(_turns, list) and (
                _fp.is_first_utterance_question(query, loose=True)
                or _fp.is_first_utterance_question(utterance)):
            turns = [t for t in _turns if isinstance(t, str)]
            # main() records THIS turn before dispatch, so on the LLM route
            # the question being asked can sit in the record as its newest
            # entry — and, right after a start, as the session's only real
            # one ("what did I ask you at the start of today" slips past the
            # recall detectors). Never recall the current question as the
            # first thing he asked: drop it, the same rule as skip_newest for
            # "what did I just ask you".
            if (utterance and turns
                    and _fp.normalize(turns[-1]) == _fp.normalize(utterance)):
                turns = turns[:-1]
            _hist = bc.conversation_history
            _hist = list(_hist) if isinstance(_hist, list) else []
            # The monolith's latch: a forget / reset purged the start, or a
            # handoff came without it — no later turn is the first thing.
            _lost = getattr(bc, "_session_opening_lost", None)
            _lost = (isinstance(_lost, list) and bool(_lost)
                     and _lost[0] is True)
            return _fp.first_utterance_reply(
                utterance if _fp.first_recall_verb(utterance) else query,
                turns, _hist,
                skip_newest=_fp.recall_turn_recorded(_hist),
                start_lost=_lost)
        if _fp.is_last_utterance_question(query, loose=True):
            history = list(bc.conversation_history)
            return _fp.last_utterance_reply(
                query, history,
                skip_newest=_fp.recall_turn_recorded(history))
    except Exception:
        pass   # fall back to the session-index recall below
    try:
        sessions = bc.pattern_memory.get_session_summaries(query, limit=8)
    except Exception as e:
        return f"session recall failed: {e}"

    if not sessions:
        window = bc.pattern_memory.describe_window(query)
        return (f"I'm afraid I have no recollection {window}, sir — "
                f"the session log is empty for that period.")

    lines = []
    for s in sessions:
        date = s.get("date", "")
        day  = s.get("day", "")
        h_s  = s.get("hour_started", -1)
        h_e  = s.get("hour_ended", -1)
        when = date
        if day:
            when = f"{day} {date}"
        if isinstance(h_s, int) and h_s >= 0 and isinstance(h_e, int) and h_e >= 0:
            when += f" {h_s:02d}:00-{h_e:02d}:00"
        summary = s.get("summary", "").strip()
        if summary:
            lines.append(f"- {when}: {summary}")
    context_block = "\n".join(lines)

    system = (
        "You are J.A.R.V.I.S. recalling what the user (sir) was working on. "
        "You will be given a short list of prior session summaries. "
        "Produce ONE or TWO sentences in JARVIS voice — composed, British, "
        "dry — that synthesise the relevant activity for the time window the "
        "user asked about. Lead with the time reference ('Yesterday evening, "
        "sir,' / 'Earlier this morning, sir,'). Mention specific work "
        "(projects, features, fixes) by name from the summaries. If the "
        "summaries cover multiple sessions, fold them together rather than "
        "listing them. Do not invent details that aren't in the summaries. "
        "No preamble, no bullet points, no closing question — just the recall."
    )
    user_msg = f"User asked: {query!r}\n\nSession summaries (newest first):\n{context_block}"
    try:
        reply = bc._llm_quick(system=system, user=user_msg, max_tokens=200)
        reply = reply.strip()
    except Exception as e:
        return f"session recall LLM call failed: {e}"
    if not reply:
        return f"recalled {len(sessions)} session(s) but the recall LLM returned nothing"
    return reply


# ─── Cached-screen recall (Phase 4I) ───────────────────────────────────

_RECALL_STOP = frozenset("""
what was that the a an on in at of to my your screen screens monitor
monitors video videos page window tab thing one i me you were was is are
looking look watching watch reading read doing did see saw seen earlier
before ago minutes minute mins min hours hour seconds second just there
which who about from with it this time when then while recently few couple
left right top middle main bottom open opened jarvis sir please
""".split())


def _recall_window(question: str, now: float):
    """(since, until, label) for the time words in a recall question.
    Default: the last 15 minutes."""
    q = (question or "").lower()
    m = re.search(r"\b(\d+|a|an|one|two|three|five|ten|fifteen|twenty|"
                  r"thirty|few|couple(?:\s+of)?)\s+(minutes?|mins?|hours?)\s+"
                  r"ago\b", q)
    words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "five": 5,
             "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "few": 3,
             "couple": 2, "couple of": 2}
    if m:
        n = float(words.get(m.group(1), m.group(1)) if not
                  m.group(1).isdigit() else m.group(1))
        secs = n * (3600.0 if m.group(2).startswith("h") else 60.0)
        pad = max(120.0, secs * 0.3)
        return now - secs - pad, now - secs + pad, f"about {m.group(0)}"
    if "this morning" in q:
        lt = time.localtime(now)
        start = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 5, 0, 0, 0,
                             0, -1))
        return start, start + 7 * 3600, "this morning"
    if re.search(r"\btoday\b", q):
        lt = time.localtime(now)
        start = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0,
                             0, -1))
        return start, now, "today"
    if re.search(r"\b(?:earlier|before|a while ago|last hour)\b", q):
        return now - 3600.0, now, "in the last hour"
    return now - 900.0, now, "in the last 15 minutes"


def _act_recall_screen(question: str) -> str:
    """What WAS on the screen, from JARVIS's text record (core.
    screen_timeline - looks, clicks, scene freezes and, when screen memory
    is on, the watcher), with times. Never re-asks the vision model who or
    what something was: live 00:29:49 that invented "the Kai Cenat video".
    Only a VISUAL follow-up ("what colour was it") re-examines the last
    cached look, and says it is a reduced snapshot."""
    bc = _bc()
    q = " ".join(str(question or "").split())
    said = _turn_said(bc)
    from core.screen_digest import VISUAL_Q_RE
    if q and VISUAL_Q_RE.search(q):
        recent = bc._recent_screen_contexts()
        entry = next((e for e in recent if e.get("images")), None)
        if entry is not None:
            age = time.time() - entry["ts"]
            age_str = bc._format_screen_age(age)
            images = entry["images"]
            print(f"  [vision] visual follow-up on a cached look "
                  f"({age_str})", flush=True)
            if len(images) == 1:
                ans = bc.ask_vision(f"This is a cached screenshot from "
                                    f"{age_str}. {q}", next(iter(images.values())))
            else:
                ans = bc.ask_vision_multi(f"These screenshots are cached from "
                                          f"{age_str}. {q}", images)
            return (f"From a reduced snapshot {int(age)} s ago (only how it "
                    f"looked, not what it was called): {ans}")
    from core import screen_timeline as _tl
    from core import monitor_geometry as _mg
    from core.config import MONITORS
    now = time.time()
    since, until, when = _recall_window(q or said, now)
    mon = _mg.monitor_named_in(q or said, MONITORS)
    words = [w for w in re.findall(r"[a-z0-9$']+", (q or "").lower())
             if w not in _RECALL_STOP and len(w) >= 3]
    video = bool(re.search(r"\b(?:video|watch|watching|youtube|clip)\b",
                           (q + " " + said).lower()))
    tl = _tl.get()
    tl.flush(1.0)
    rows = []
    if words:
        rows = tl.query(text=" ".join(words), since=since, until=until,
                        monitor=mon, limit=12)
        if not rows:
            rows = tl.query(text=" ".join(words), since=now - 7 * 86400,
                            monitor=mon, limit=8)
    if not rows:
        rows = tl.query(since=since, until=until, monitor=mon, limit=40)
        if video:
            vids = [r for r in rows
                    if "watch?v=" in (r.get("url") or "")
                    or "youtube" in (r.get("title") or "").lower()
                    or r.get("source") in ("scene", "click")]
            rows = vids or rows
        rows = rows[:12]
    if not rows:
        print(f"  [recall] no record {when}", flush=True)
        return (f"I have no record of that, sir \u2014 nothing on screen "
                f"{when} matches.")
    lines = []
    for r in sorted(rows, key=lambda r: r["ts"]):
        stamp = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        host = ""
        try:
            host = (urllib.parse.urlsplit(r["url"]).hostname or "").removeprefix(
                "www.") if r.get("url") else ""
        except Exception:
            host = ""
        head = ", ".join(b for b in (stamp, r.get("monitor") or "",
                                     (r.get("process") or "").replace(
                                         ".exe", ""), host) if b)
        body = " ".join(str(r.get("text") or "").split())[:300]
        title = str(r.get("title") or "")[:120]
        lines.append(f"[{head}] '{title}'" + (f" - {body}" if body else ""))
    print(f"  [recall] {len(lines)} recorded row(s) {when}", flush=True)
    return ("Recorded - quote exactly; if the answer is not here, say you "
            "have no record:\n" + "\n".join(lines[-12:]))


# ─── Changelog read + summarise (Phase 4I) ─────────────────────────────

def _act_read_changelog(args: str = "") -> str:
    """Read CHANGELOG.md and either speak a concise summary of the most
    recent entry (default) or up to 3 entries if the user asks 'what has
    changed lately'. For long entries (> ~6000 chars) open the file in the
    default editor and speak a brief pointer instead of summarising in
    voice."""
    bc = _bc()
    try:
        _changelog_path = os.path.join(
            os.path.dirname(os.path.abspath(bc.__file__)), "CHANGELOG.md")
        if not os.path.exists(_changelog_path):
            return "I don't have a changelog file yet, sir."
        with open(_changelog_path, "r", encoding="utf-8") as _cf:
            text = _cf.read()
        arg_lower = (args or "").strip().lower()
        want_history = any(k in arg_lower for k in (
            "lately", "recent", "recently", "history", "past few",
            "last few", "several"))
        n_entries = 3 if want_history else 1
        headers = list(re.finditer(r"^## v.+$", text, re.MULTILINE))
        if not headers:
            return "The changelog is empty, sir."
        entries: list[str] = []
        for i, h in enumerate(headers[:n_entries]):
            start = h.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            chunk = text[start:end].strip()
            chunk = re.sub(r"\n---\s*$", "", chunk).strip()
            entries.append(chunk)
        combined = "\n\n".join(entries)
        # Long-entry branch — open file in default editor and give a short
        # spoken pointer.
        if len(combined) > 6000:
            try:
                if sys.platform == "win32":
                    os.startfile(_changelog_path)
                else:
                    subprocess.Popen(["xdg-open", _changelog_path],
                                     close_fds=True)
            except Exception:
                pass
            return ("The latest changelog entry is sizeable, sir — I've "
                    "opened CHANGELOG.md for you to read in full.")
        plural = "entries" if n_entries > 1 else "entry"
        system = (
            "You are J.A.R.V.I.S. Summarise the following CHANGELOG.md "
            f"{plural} into ONE to THREE concise sentences spoken in your "
            "own voice for the user (he/sir). Call out new capabilities by "
            "name and the rough number of bug fixes if visible. Plain "
            "prose only — no markdown, no bullets, no headers."
        )
        user = f"Changelog content:\n\n{combined[:8000]}"
        try:
            summary = (bc._llm_quick(system, user, max_tokens=300) or "").strip()
        except Exception as e:
            return f"could not summarise the changelog: {e}"
        if not summary:
            return ("I couldn't produce a summary just now, sir — the full "
                    f"changelog is at {_changelog_path}.")
        return summary
    except Exception as e:
        return f"could not read the changelog: {e}"


# ─── Overnight engine kick (Phase 4J) ──────────────────────────────────

def _act_start_overnight_upgrade(_: str = "") -> str:
    """Trigger the built-in overnight upgrade engine immediately.
    Sets the run-now flag so the background thread skips the idle wait and
    starts generating improvements straight away. Also enters sleep mode.

    Writes a persistence flag (.overnight_active) so the engine keeps
    re-triggering across JARVIS restarts for the next OVERNIGHT_MODE_HOURS
    hours — important because each completed upgrade kills + relaunches
    JARVIS, and a fresh process wouldn't otherwise know it's still
    overnight time."""
    bc = _bc()
    from core.config import OVERNIGHT_MODE_HOURS
    # Switched off (OVERNIGHT_UPGRADE_ENABLED = False): the engine thread is
    # never started at boot, so the run-now flag below would be set for
    # nothing — yet JARVIS went to sleep and wrote an 8 h .overnight_active
    # flag that re-armed sleep across restarts (2026-09-30 audit, tray "Run
    # Upgrade Now"). Refuse plainly and change nothing. The monolith's own
    # global is the authority (staging forces it off).
    try:
        from core import config as _cfg
        enabled = bool(getattr(bc, "OVERNIGHT_UPGRADE_ENABLED",
                               getattr(_cfg, "OVERNIGHT_UPGRADE_ENABLED", False)))
    except Exception:
        enabled = False
    if not enabled:
        return ("The overnight upgrade engine is switched off, sir "
                "(OVERNIGHT_UPGRADE_ENABLED), so there's nothing to run — "
                "I'm staying awake.")
    bc._overnight_run_now.set()
    bc._sleep_mode[0] = True

    # Write the persistence flag — overnight mode survives restarts until
    # this expiry. New JARVIS sessions check the flag at startup and
    # re-arm the engine automatically.
    try:
        expiry = time.time() + OVERNIGHT_MODE_HOURS * 3600
        with open(bc.OVERNIGHT_FLAG_FILE, "w", encoding="utf-8") as _f:
            _f.write(str(expiry))
        bc._write_hud_state(overnight_expiry=expiry)
        print(f"  [overnight] persistence flag set, active until "
              f"{time.strftime('%H:%M', time.localtime(expiry))}")
    except Exception as _e:
        print(f"  [overnight] couldn't write persistence flag: {_e}")

    return (
        "On it, sir. I'll start generating improvements right away "
        "and stand by quietly — say 'JARVIS' when you need me again."
    )


# ─── Window placement (Phase 4J) ───────────────────────────────────────

# Monitor words the owner (and the local model) use that are not MONITORS keys
# (2026-10-01: "put it on the main monitor" failed as "unknown monitor").
# "main" / "primary" mean the Windows primary display: the one at the origin.
_MONITOR_FILLER_RE = re.compile(r"\b(?:the|my|monitor|screen|display)\b")
_PRIMARY_MONITOR_WORDS = ("main", "primary", "center", "centre")


def _resolve_monitor(name) -> "str | None":
    """A MONITORS key for `name` ('left', 'the main monitor', 'Primary'...), else None."""
    from core.config import MONITORS
    s = " ".join(_MONITOR_FILLER_RE.sub(" ", str(name or "").lower()).split())
    if s in MONITORS:
        return s
    if s in _PRIMARY_MONITOR_WORDS:
        for key, rect in MONITORS.items():
            if tuple(rect[:2]) == (0, 0):
                return key
    return None


def _split_monitor_args(args: str, monitor_first: bool) -> "tuple[str | None, str]":
    """(monitor key, the other part) from '<a> | <b>' -- or the comma form the
    local model writes every time ('left, youtube, cello'; live 2026-10-01: two
    open_on_monitor calls in a row died on "format:"). The monitor is normally on
    the documented side; if that side is not a monitor but the other end is, the
    model swapped them. The monitor is None when neither end names one."""
    if "|" in args:
        a, b = (s.strip() for s in args.split("|", 1))
    else:
        parts = [p.strip() for p in str(args or "").split(",") if p.strip()]
        if len(parts) < 2:
            return None, ""
        if monitor_first:
            a, b = parts[0], " ".join(parts[1:])
        else:
            a, b = ", ".join(parts[:-1]), parts[-1]
    mon, other = (a, b) if monitor_first else (b, a)
    key = _resolve_monitor(mon)
    if key is None:
        swapped = _resolve_monitor(other)
        if swapped is not None:
            return swapped, mon
    return key, other


_YOUTUBE_SEARCH_RE = re.compile(r"^(?:youtube|you tube)\s+(?:for\s+)?(.+)$", re.IGNORECASE)


# Seconds open_on_monitor waits for a fresh window whose title MATCHES the
# target before settling for a fresh window that doesn't (yet).
_OPEN_ON_MONITOR_GRACE_S = 2.0
# Seconds with no fresh window, while a pre-existing window matches the
# target, before open_on_monitor concludes the app reused that window.
_OPEN_ON_MONITOR_REUSE_S = 4.0
# How long open_on_monitor waits on the voice turn for the new window.
_OPEN_ON_MONITOR_WAIT_S = 15.0
# A slow-starting app (2026-10-02 live, 14:47:52: Teams, closed since the
# morning, showed no window inside the 15 s wait, and was still windowless 46 s
# after the launch) is then watched for this much longer OFF the voice thread,
# and its window is moved when it appears. Bounded; one log line either way.
_OPEN_ON_MONITOR_WATCH_S = 30.0
_OPEN_ON_MONITOR_WATCH_POLL_S = 0.5
# Words that name a vendor or a kind of program, not one app: "Microsoft
# Teams" is ms-teams.exe, never every "Microsoft ..." window on the desktop.
_APP_GENERIC_WORDS = frozenset({
    "microsoft", "google", "adobe", "apple", "the", "app", "application",
    "desktop", "client", "new", "classic", "windows", "for", "and",
})
# The browser executables' name words (the process-name half of
# _is_browser_window; the title half is the monolith's suffix list).
_BROWSER_PROCESS_WORDS = ("chrome", "msedge", "firefox", "brave", "opera",
                          "vivaldi", "chromium", "iexplore")
# Processes that host OTHER apps' windows (packaged apps such as Calculator):
# their name says nothing about the app, so the title decides.
_APP_HOST_PROCESSES = frozenset({"applicationframehost.exe"})
# The background watches by target: a newer request for the same app takes
# over (the older one stops), so one window is never moved twice.
_OPEN_WATCHES: dict = {}
_OPEN_WATCHES_LOCK = threading.Lock()


def _window_key(w):
    """A stable identity for a pygetwindow window: its native handle (titles
    change under us), or the object itself where there is none (held in the
    set, so its identity can't be recycled the way a bare id() can)."""
    hwnd = getattr(w, "_hWnd", None)
    return hwnd if hwnd is not None else w


def _app_words(target) -> list:
    """The words of an app name that identify THAT app ("Microsoft Teams" ->
    ["teams"]); all its words when every one is generic."""
    toks = [t for t in re.split(r"[^a-z0-9+#]+", str(target or "").lower())
            if len(t) >= 3]
    own = [t for t in toks if t not in _APP_GENERIC_WORDS]
    return own or toks


def _compact(s) -> str:
    return re.sub(r"[^a-z0-9+#]", "", str(s or "").lower())


def _exe_stem(proc) -> str:
    return _compact(re.sub(r"\.exe$", "", str(proc or "").strip(), flags=re.I))


def _is_browser_window(bc, w) -> bool:
    """True when ``w`` is a web browser's window (its process, or its title's
    browser suffix). Never raises: on a fault True, so a window that may be
    the owner's stream is never moved (B092)."""
    try:
        proc = _window_process_name(w)
        if proc and proc.strip().lower() not in _APP_HOST_PROCESSES:
            return any(b in _exe_stem(proc) for b in _BROWSER_PROCESS_WORDS)
        return _browser_page_title(bc, getattr(w, "title", "") or "") is not None
    except Exception:
        return True


def _exe_names_app(proc, words) -> bool:
    """True when executable ``proc`` is the app ``words`` names. Review
    2026-10-02: ANY one word inside the exe name was enough, so "Visual
    Studio Code" matched bambu-studio.exe and AtmelStudio.exe through
    "studio". Now the exe carries EVERY word ("ms-teams" <- teams,
    "StreamDeck" <- stream deck), or is part of the whole name ("Code" <-
    visual studio code, "explorer" <- file explorer), or starts with its
    first word and adds a short tail ("obs64" <- obs studio, "Taskmgr" <-
    task manager). Never raises."""
    try:
        stem = _exe_stem(proc)
        own = [c for c in (_compact(t) for t in words or ()) if c]
        if not stem or not own:
            return False
        if all(t in stem for t in own):
            return True
        if len(stem) >= 4 and stem in "".join(own):
            return True
        return (len(own) > 1 and len(own[0]) >= 3
                and stem.startswith(own[0]) and len(stem) - len(own[0]) <= 4)
    except Exception:
        return False


# A window title's parts: "Inbox - someone - Outlook", "Chat | Microsoft Teams".
_TITLE_PART_SPLIT_RE = re.compile(r"\s+[-–—|]\s+")


def _title_names_app(title, words) -> bool:
    """True when the LAST part of ``title`` carries every word - where an
    app puts its own name ("Deck1 - PowerPoint", "Task Manager"); a document
    or folder that merely shares the name puts it first ("Teams - File
    Explorer", "teams notes.txt - Notepad"). Never raises."""
    try:
        last = _TITLE_PART_SPLIT_RE.split(str(title or "").strip())[-1].lower()
        return bool(last) and all(t in last for t in words or ())
    except Exception:
        return False


def _is_app_window(bc, w, words) -> bool:
    """True when window ``w`` belongs to the app ``words`` names
    (_app_words): its process executable is that app's (_exe_names_app, the
    v2.0.176 process-name match - "ms-teams.exe" for "Microsoft Teams"), or
    its title ends with the app's name (_title_names_app: the exe name of
    olk.exe / POWERPNT.EXE doesn't say "Outlook" / "PowerPoint", and a
    packaged app's window belongs to ApplicationFrameHost.exe). A browser's
    window and its pages are the browser, never another app (unless the
    words name the browser itself), and the shell's desktop window is never
    an app's. Never raises."""
    try:
        if not words:
            return False
        title = (getattr(w, "title", "") or "").strip()
        if title.lower() in _SHELL_WINDOW_TITLES:
            return False
        proc = _window_process_name(w)
        if proc and proc.strip().lower() not in _APP_HOST_PROCESSES:
            if _exe_names_app(proc, words):
                return True
            if any(b in _exe_stem(proc) for b in _BROWSER_PROCESS_WORDS):
                return False
        if not _title_names_app(title, words):
            return False
        if _browser_page_title(bc, title) is not None:
            return _query_names_browser(" ".join(words))
        return True
    except Exception:
        return False


def _usable_window(w) -> bool:
    """A titled window big enough to be an app's own (not a splash, tooltip
    or tray stub) - or minimized, which Windows reports as a tiny icon rect.
    Never raises."""
    try:
        if not (getattr(w, "title", "") or "").strip():
            return False
        if getattr(w, "isMinimized", False) is True:
            return True
        return not (w.width < 200 or w.height < 200)
    except Exception:
        return True


def _place_on_monitor(w, mx, my, sleep=None) -> None:
    """Restore ``w``, move it onto the monitor whose origin is (mx, my) and
    maximize it there. Raises what pygetwindow raises."""
    sleep = sleep or time.sleep
    w.restore()
    sleep(0.1)
    w.moveTo(mx + 50, my + 50)
    sleep(0.1)
    w.maximize()


def _watch_for_app_window(gw, bc, target, words, hwnds_before, monitor_name,
                          rect, *, waited_s=0.0, timeout_s=None, poll_s=None,
                          clock=None, sleep=None, cancel=None) -> str:
    """Wait (on a background thread) up to ``timeout_s`` for a NEW window of
    the app ``target`` (one not in ``hwnds_before``, _is_app_window) and move
    it to ``monitor_name`` (``rect`` = MONITORS[monitor_name]). Prints one
    line. Returns "moved", "move-failed", "timeout" or "cancelled" (a newer
    request took over: ``cancel`` set). ``clock`` / ``sleep`` are injectable
    for tests. Never raises."""
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    timeout_s = _OPEN_ON_MONITOR_WATCH_S if timeout_s is None else timeout_s
    poll_s = _OPEN_ON_MONITOR_WATCH_POLL_S if poll_s is None else poll_s
    tag = f"  [open-on-monitor] {target}:"
    try:
        start = clock()
        deadline = start + float(timeout_s)
        while clock() < deadline:
            if cancel is not None and cancel.is_set():
                print(f"{tag} a newer request took over the window watch")
                return "cancelled"
            sleep(poll_s)
            try:
                windows = gw.getAllWindows()
            except Exception:
                windows = []
            for w in windows:
                if (_window_key(w) in hwnds_before or not _usable_window(w)
                        or not _is_app_window(bc, w, words)):
                    continue
                # A newer request (or the owner's own move) may have landed
                # while this poll slept: it owns the window now.
                if cancel is not None and cancel.is_set():
                    print(f"{tag} a newer request took over the window watch")
                    return "cancelled"
                after = float(waited_s) + (clock() - start)
                try:
                    _place_on_monitor(w, rect[0], rect[1], sleep)
                except Exception as e:
                    print(f"{tag} its window appeared {after:.0f} s after the "
                          f"launch but could not be moved: {e}")
                    return "move-failed"
                print(f"{tag} its window appeared {after:.0f} s after the "
                      f"launch - moved it to the {monitor_name} monitor")
                return "moved"
        print(f"{tag} no window within {float(waited_s) + float(timeout_s):.0f}"
              f" s of the launch - left it to open where it opens")
        return "timeout"
    except Exception as e:
        print(f"{tag} the window watch stopped: {e}")
        return "timeout"


def _watch_key(words, target="") -> str:
    return " ".join(words or ()) or str(target or "").lower()


def _cancel_open_watch(words, target="") -> bool:
    """Stop the background watch for this app, if one is running (review
    2026-10-02: a newer open_on_monitor for the same app used to stop it only
    when ITS OWN 15 s wait ended, so the old watch could still move the window
    to the old monitor after the new request had put it on the new one).
    Returns True when one was stopped. Never raises."""
    try:
        with _OPEN_WATCHES_LOCK:
            old = _OPEN_WATCHES.pop(_watch_key(words, target), None)
        if old is None:
            return False
        old.set()
        return True
    except Exception:
        return False


def _cancel_watches_for_window(bc, w) -> None:
    """The owner had window ``w`` moved (move_window_to_monitor): a watch
    still waiting to move a window of the same app would undo that, so it
    stops. Never raises."""
    try:
        with _OPEN_WATCHES_LOCK:
            keys = list(_OPEN_WATCHES)
        for key in keys:
            if _is_app_window(bc, w, key.split()):
                if _cancel_open_watch(key.split()):
                    print(f"  [open-on-monitor] {key}: moved by request - "
                          f"stopped its window watch")
    except Exception:
        pass


def _start_open_watch(gw, bc, target, words, hwnds_before, monitor_name,
                      rect, waited_s) -> "threading.Thread | None":
    """Run _watch_for_app_window on a daemon thread - never on the voice
    turn. A watch already running for the same app is told to stop first.
    Returns the thread, None when it could not start. Never raises."""
    key = _watch_key(words, target)
    cancel = threading.Event()
    with _OPEN_WATCHES_LOCK:
        old = _OPEN_WATCHES.get(key)
        if old is not None:
            old.set()
        _OPEN_WATCHES[key] = cancel

    def _run():
        try:
            _watch_for_app_window(gw, bc, target, words, hwnds_before,
                                  monitor_name, rect, waited_s=waited_s,
                                  cancel=cancel)
        finally:
            with _OPEN_WATCHES_LOCK:
                if _OPEN_WATCHES.get(key) is cancel:
                    del _OPEN_WATCHES[key]

    try:
        t = threading.Thread(target=_run, name="open-on-monitor-watch",
                             daemon=True)
        t.start()
        return t
    except Exception:
        with _OPEN_WATCHES_LOCK:
            if _OPEN_WATCHES.get(key) is cancel:
                del _OPEN_WATCHES[key]
        return None


def _launch_failed(result) -> bool:
    """True when _act_launch_app's result says the launch itself failed."""
    try:
        if not isinstance(result, str):
            return False
        from core.failure_markers import FAILURE_MARKERS
        low = result.lower()
        return any(m in low for m in FAILURE_MARKERS)
    except Exception:
        return False


def _act_open_on_monitor(args: str) -> str:
    """args format: '<monitor_name> | <url-or-app-name>' (or the comma form,
    see _split_monitor_args). Opens the URL or launches the app, then moves the
    resulting window to the named monitor and maximizes it.

    An app whose window has not appeared when the wait ends is watched in the
    background (_start_open_watch) and moved when it does; the result says
    so, and never calls the app "already open" unless a window of it was
    found BEFORE the launch - that window is then moved at once (2026-10-02).
    A browser's existing window is never moved (B092: it may be the owner's
    stream); that case still only offers."""
    bc = _bc()
    from core.config import MONITORS
    if "|" not in args and "," not in args:
        return "format: open_on_monitor, <monitor_name> | <url-or-app>"
    monitor_name, target = _split_monitor_args(args, monitor_first=True)
    if monitor_name is None:
        asked = (args.split("|", 1)[0] if "|" in args else args.split(",", 1)[0]).strip().lower()
        return f"unknown monitor '{asked}'. Available: {list(MONITORS.keys())}"
    if not target:
        return "format: open_on_monitor, <monitor_name> | <url-or-app>"
    # "youtube cello" (the model's "youtube, cello") is a search, not an app.
    target = _site_shortcut_url(target) or target
    # A guessed streaming search link / a bare service name ("HBO Max") ->
    # the verified link or the service's home page (S2, 2026-10-02).
    target, _fix_note = _streaming_url_fix(target, bare_names=False)
    m = _YOUTUBE_SEARCH_RE.match(target)
    if m:
        target = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(m.group(1).strip())
    mx, my, mw, mh = MONITORS[monitor_name]

    try:
        import pygetwindow as gw
    except ImportError:
        return "pygetwindow not available — pip install pygetwindow"
    # Snapshot window HANDLES, not titles (2026-10-01, B092): a title snapshot
    # let a pre-existing window count as "new" the moment its title changed
    # (a browser tab moving on to the next video), and the old loop also
    # accepted ANY pre-existing window whose title matched the target — so
    # "open Chrome on the left monitor" restored, moved and maximised the
    # owner's existing Chrome window (e.g. his stream) on the first 0.2 s
    # poll, before the new window existed, and left the new one where it
    # opened. Only a window that did not exist before the launch is moved.
    windows_before = list(gw.getAllWindows())
    hwnds_before = {_window_key(w) for w in windows_before}

    # Launch the target. Treat as URL if explicit scheme or recognisable
    # domain suffix; otherwise treat as an app name.
    _URL_HINT = re.compile(
        r"^(?:https?://|[\w\-]+\.(?:com|net|org|io|gov|edu|co|app|dev|me|tv|ai|so|xyz)(?:/|$))",
        re.IGNORECASE,
    )
    is_url = bool(_URL_HINT.match(target))
    # The app's OWN windows that were open before the launch (process-name
    # matched, _is_app_window): the only evidence that it was "already open".
    app_words = [] if is_url else _app_words(target)
    # This request owns the app's window now: a watch an earlier request
    # left running must not move it after this one does (review 2026-10-02).
    if app_words:
        _cancel_open_watch(app_words, target)
    existing = {_window_key(w) for w in windows_before
                if _usable_window(w) and _is_app_window(bc, w, app_words)}
    # Wait for a window matching the target to appear.
    target_tokens = [
        tok for tok in re.split(r"[\s_\-]+", target.lower()) if len(tok) >= 3
    ]

    def _page_names_target(w) -> bool:
        # A browser window whose ACTIVE page carries every word of the
        # target ("Apple Music - Google Chrome"): a web app's tab.
        try:
            page = _browser_page_title(bc, getattr(w, "title", "") or "")
            return (page is not None and bool(target_tokens)
                    and all(tok in page.lower() for tok in target_tokens))
        except Exception:
            return False

    # Browser windows whose page already named the app before the launch
    # (Teams on the web while the desktop app starts): not where it went.
    named_before = {_window_key(w) for w in windows_before
                    if _page_names_target(w)}
    if is_url:
        # monitor=: the new window's own placement (visible + maximized,
        # 2026-10-01) targets the same monitor this action then moves it to.
        if not bc._open_url_new_window(target, monitor=monitor_name):
            webbrowser.open(target if target.startswith(("http://", "https://"))
                            else "https://" + target)
    else:
        launched = _act_launch_app(target)
        if _launch_failed(launched):
            return launched
        if "already open" in str(launched or "").lower() and not existing:
            # The launcher itself found the app running where no window of
            # it is ours to move - Apple Music's web player in the owner's
            # browser (review 2026-10-02): no new window is coming, and a
            # browser window may be his stream (B092). Say so; never promise
            # a move nothing will make.
            return (f"{target} is already open in your browser, so I didn't "
                    f"move that window — ask me to move it to the "
                    f"{monitor_name} monitor if you want it there")

    def _matches_target(w) -> bool:
        t = (w.title or "").lower()
        if target_tokens and any(tok in t for tok in target_tokens):
            return True
        return not is_url and _is_app_window(bc, w, app_words)

    def _reusable(w) -> bool:
        # A URL may become a tab of a window that was already open; an app
        # may bring forward its own window from before the launch, or (a web
        # app such as the Apple Music player) open as a NEW tab in a browser
        # window that was already open - its page names the app only now.
        if is_url:
            return _matches_target(w)
        key = _window_key(w)
        return key in existing or (key not in named_before
                                   and _page_names_target(w)
                                   and _is_browser_window(bc, w))

    new_window = None
    fallback = None   # a FRESH window that doesn't (yet) match the target
    reused = None     # a PRE-EXISTING window the launch may have reused
    started = time.time()
    deadline = started + _OPEN_ON_MONITOR_WAIT_S
    while time.time() < deadline:
        time.sleep(0.2)
        fresh = []
        for w in gw.getAllWindows():
            if not w.title:
                continue
            if _window_key(w) in hwnds_before:
                if reused is None and _reusable(w):
                    reused = w
                continue
            try:
                if w.width < 200 or w.height < 200:
                    continue   # ignore tiny splash/tooltip windows
            except Exception:
                pass
            fresh.append(w)
        matched = [w for w in fresh if _matches_target(w)]
        if matched:
            new_window = matched[0]
            break
        if fresh and fallback is None:
            fallback = fresh[0]
        # A URL target's page title rarely contains its domain token, and a
        # new window's first title is often "New Tab": after a short grace,
        # take the fresh window we saw instead of waiting out the deadline.
        if fallback is not None and time.time() - started >= _OPEN_ON_MONITOR_GRACE_S:
            break
        # Single-instance apps (VS Code, Spotify, Teams) and a URL that
        # became a tab reuse a window that was already open: no fresh window
        # will ever come, and waiting out the full 15 s only to say so was a
        # UX regression (2026-10-01, actions-a review).
        if (fallback is None and reused is not None
                and time.time() - started >= _OPEN_ON_MONITOR_REUSE_S):
            break
    if new_window is None:
        new_window = fallback   # still a window from AFTER the launch, never before

    if new_window:
        try:
            _place_on_monitor(new_window, mx, my)
        except Exception as e:
            return f"opened {target} but failed to move window: {e}"
        # The window JARVIS made: "close that" closes exactly this one, and a
        # click / a look at "the page" with no monitor named aims here (S1/S4).
        try:
            from core import opened_ledger as _ol
            _ol.note_opened("open_on_monitor", target,
                            hwnd=getattr(new_window, "_hWnd", None),
                            kind="window", monitor=monitor_name,
                            title=getattr(new_window, "title", "") or "")
        except Exception:
            pass
        if _fix_note:
            return (f"opened '{target}' on {monitor_name} monitor "
                    f"(at {mx},{my}) - {_fix_note}")
        return f"opened '{target}' on {monitor_name} monitor (at {mx},{my})"

    if reused is not None:
        if not is_url and not _is_browser_window(bc, reused):
            # The app was open before the launch and brought its own window
            # forward: that is the window the owner meant - move it now.
            try:
                _place_on_monitor(reused, mx, my)
            except Exception as e:
                return (f"{target} was already open, but I failed to move its "
                        f"'{reused.title}' window: {e}")
            return (f"{target} was already open, so I moved its "
                    f"'{reused.title}' window to the {monitor_name} monitor")
        # A browser window that was already open (a URL became one of its
        # tabs) may be the owner's stream (B092): offer, never move.
        return (f"launched {target}, but it reused your existing "
                f"'{reused.title}' window rather than opening a new one, so "
                f"I didn't move it — ask me to move '{reused.title}' to the "
                f"{monitor_name} monitor if you want it there")
    if is_url:
        # Only what was seen: a browser window open before the launch is
        # where the page most likely went.
        if any(_usable_window(w) and _is_browser_window(bc, w)
               for w in windows_before):
            return (f"opened {target}, but couldn't find new window to move "
                    f"it — it probably opened as a tab in a browser window "
                    f"that was already open; ask me to move that window to "
                    f"the {monitor_name} monitor if you want it there")
        return f"opened {target}, but couldn't find new window to move it"
    # The app is still starting (no window of it existed before the launch,
    # none has appeared yet): keep watching off the voice thread.
    if _start_open_watch(gw, bc, target, app_words, hwnds_before,
                         monitor_name, (mx, my, mw, mh),
                         waited_s=time.time() - started) is None:
        return (f"launched {target}, but couldn't find its window to move it "
                f"to the {monitor_name} monitor yet")
    return (f"launched {target}; I'll move it to the {monitor_name} monitor "
            f"when its window appears")


def _act_move_window_to_monitor(args: str) -> str:
    """Move an existing window to a named monitor using win32 SetWindowPos.

    args format: '<window_title> | <monitor_name>'

    Reliable alternative to win+shift+arrow hotkeys, which depend on the
    target window having focus AND the right monitor being adjacent. This
    one resolves the window handle by title (partial match) and sets the
    position directly to the target monitor's top-left coordinates, then
    maximizes the window so it fills that screen.
    """
    bc = _bc()
    from core.config import MONITORS
    if "|" not in args and "," not in args:
        return "format: move_window_to_monitor, <window_title> | <monitor_name>"
    monitor_name, title = _split_monitor_args(args, monitor_first=False)
    if monitor_name is None:
        asked = (args.split("|", 1)[1] if "|" in args else args.rsplit(",", 1)[-1]).strip().lower()
        return f"unknown monitor '{asked}'. Available: {list(MONITORS.keys())}"
    mx, my, mw, mh = MONITORS[monitor_name]

    if not title:
        return "format: move_window_to_monitor, <window_title> | <monitor_name>"

    matches = bc._find_windows_by_title(title)
    if not matches:
        return f"no window matching '{title}'"
    target = matches[0]

    try:
        import win32gui
        import win32con
    except ImportError:
        # pywin32 missing — fall back to pygetwindow's higher-level API
        try:
            if getattr(target, "isMaximized", False):
                target.restore()
                time.sleep(0.1)
            target.moveTo(mx, my)
            time.sleep(0.1)
            target.resizeTo(mw, mh)
            time.sleep(0.1)
            target.maximize()
            # An open_on_monitor watch for this app would undo the move.
            if _OPEN_WATCHES:
                _cancel_watches_for_window(bc, target)
            return f"moved '{target.title}' to {monitor_name} monitor (pygetwindow)"
        except Exception as e:
            return f"could not move '{target.title}': {e}"

    try:
        hwnd = target._hWnd
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.1)
        flags = 0x0004 | 0x0010  # SWP_NOZORDER | SWP_NOACTIVATE
        win32gui.SetWindowPos(hwnd, 0, mx, my, mw, mh, flags)
        time.sleep(0.1)
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        if _OPEN_WATCHES:
            _cancel_watches_for_window(bc, target)
        return f"moved '{target.title}' to {monitor_name} monitor"
    except Exception as e:
        return f"could not move '{target.title}': {e}"


# ─── LLM-authored skill creation (Phase 4J) ────────────────────────────

def _act_create_skill(args: str) -> str:
    """
    args format: '<name> | <description of what it should do>'
    Bobert asks the LLM to write a Python skill module, saves it to
    ./pending_skills/<name>.py and asks the user to move + restart.
    """
    bc = _bc()
    # SKILLS_ENABLED is genuinely boot-time, so core.config is the right source
    # for it. The BACKEND is not: switch_llm only ever writes bc.AI_BACKEND, so
    # reading core.config here got it wrong in BOTH directions — after "switch to
    # local" the frozen "claude" let this through and it went on spending Claude
    # credits authoring the skill (the exact thing the switch was meant to
    # prevent), and a box whose user_settings pin "ollama" could never create a
    # skill even with Claude live. Same stale-gate class as the vision gates.
    # 2026-07-14 audit.
    from core.config import SKILLS_ENABLED
    backend, _model = _live_backend_and_model()
    if not SKILLS_ENABLED or backend != "claude":
        return "skill creation requires SKILLS_ENABLED + Claude backend"

    if "|" not in args:
        return "format: create_skill, <name> | <what it should do>"

    raw_name, desc = (s.strip() for s in args.split("|", 1))
    name = re.sub(r"[^a-z0-9_]", "_", raw_name.lower()) or "unnamed"
    path = os.path.join(bc.PENDING_SKILLS_DIR, f"{name}.py")

    system_prompt = (
        "You write Python skill modules for the JARVIS AI assistant.\n\n"
        "Requirements:\n"
        "1. Define a function `register(actions)` that adds one or more callable\n"
        "   actions to the actions dict. Each action takes ONE string argument\n"
        "   and returns a string result.\n"
        "2. Use only the standard library + these utilities (already injected\n"
        "   as `skill_utils` at module scope):\n"
        "     skill_utils['ask_vision'](question)  -> string description of screen\n"
        "     skill_utils['find_click_target'](description) -> (x,y) or None\n"
        "     skill_utils['click'](x, y)\n"
        "     skill_utils['type_text'](text)\n"
        "     skill_utils['press_key'](key)\n"
        "     skill_utils['hotkey']('ctrl', 'l')\n"
        "     skill_utils['sleep'](seconds)\n"
        "     skill_utils['launch_app'](name)\n"
        "     skill_utils['open_url'](url)\n"
        "3. Sleep between UI interactions so apps have time to respond.\n"
        "4. NEVER include final purchase / payment confirmation steps.\n"
        "   Always stop one step BEFORE the actual money-spending click and\n"
        "   return a string asking the user to confirm manually.\n"
        "5. Output ONLY the Python code. No markdown fences, no commentary."
    )

    print(f"\n  [skill] Generating new skill '{name}'...")
    try:
        code = bc._llm_quick(system=system_prompt, user=desc, max_tokens=1500)
    except Exception as e:
        return f"skill generation failed: {e}"

    code = re.sub(r"^```(?:python)?\s*", "", code.strip())
    code = re.sub(r"\s*```$", "", code)

    try:
        compile(code, f"<skill:{name}>", "exec")
    except SyntaxError as e:
        print("-" * 60)
        print(code)
        print("-" * 60)
        print(f"  [skill] SyntaxError in generated code: {e}")
        return (
            f"skill '{name}' rejected — generated code failed syntax check "
            f"({e.msg} at line {e.lineno}). Nothing was written."
        )

    print("-" * 60)
    print(code)
    print("-" * 60)
    print(f"  [skill] Saved as PENDING (not auto-loaded): {path}")

    os.makedirs(bc.PENDING_SKILLS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    return (
        f"skill '{name}' written to pending_skills/{name}.py for review. "
        f"To activate it, review the code then MOVE the file to skills/ and "
        f"restart me. Nothing has been auto-installed."
    )


# ─── Upgrade pipeline kickoff (Phase 4K) ───────────────────────────────

def _act_upgrade(_: str = "") -> str:
    """Hand off pending queue items to Claude Code for implementation.
    Spawns upgrade_jarvis.py which kills JARVIS, runs Claude Code on the
    queue, and relaunches JARVIS with the new code.

    Guarded by OVERNIGHT_UPGRADE_ENABLED so a stray voice approval ('go
    ahead', 'do it', 'give it a shot') after an unrelated 'upgrade'
    mention can't accidentally bounce JARVIS into a pipeline. To run
    upgrades, re-enable with 'start overnight upgrade' or flip the
    config flag and bounce."""
    import logging
    import threading
    bc = _bc()
    from core.config import OVERNIGHT_UPGRADE_ENABLED
    if not OVERNIGHT_UPGRADE_ENABLED:
        return ("Upgrades are disabled, sir. Say 'start overnight upgrade' "
                "to enable, or flip OVERNIGHT_UPGRADE_ENABLED in config.")
    # Find upgrade_jarvis.py — search upward from bobert_companion's __file__.
    search_dir = os.path.dirname(os.path.abspath(bc.__file__))
    upgrade_script = None
    for _ in range(4):
        candidate = os.path.join(search_dir, "upgrade_jarvis.py")
        if os.path.exists(candidate):
            upgrade_script = candidate
            break
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    if upgrade_script is None:
        return ("upgrade_jarvis.py not found anywhere near my script. "
                "Looked starting from " + os.path.abspath(bc.__file__))

    pending = 0
    if os.path.exists(bc.TODO_FILE):
        try:
            with open(bc.TODO_FILE, "r", encoding="utf-8") as f:
                pending = sum(1 for line in f if line.strip().startswith("- [ ]"))
        except Exception:
            pass

    if pending == 0:
        return "queue is empty - nothing to upgrade right now"

    # Spawn upgrade pipeline in a new visible PowerShell window so the user
    # can watch the work — and the window stays open even if the script errors.
    try:
        project_dir = os.path.dirname(os.path.abspath(bc.__file__))
        # Strip ANTHROPIC_API_KEY so Claude Code bills to Max, not API credits.
        ps_cmd = (
            f"$env:ANTHROPIC_API_KEY=''; "
            f"cd '{project_dir}'; "
            f"Write-Host '=== JARVIS UPGRADE PIPELINE ===' -ForegroundColor Cyan; "
            f"python '{upgrade_script}' --relaunch"
        )
        _env = os.environ.copy()
        _env.pop("ANTHROPIC_API_KEY", None)
        subprocess.Popen(
            ["powershell", "-Command", ps_cmd],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            env=_env,
            close_fds=True,
        )
    except Exception as e:
        return f"failed to spawn upgrade: {e}"

    def _self_exit():
        # The LAST os._exit in the tree (2026-07-14 audit). Everything else was
        # converted in v2.0.51/57: os._exit → ExitProcess walks
        # DLL_PROCESS_DETACH under the loader lock, a thread parked in the CUDA
        # driver holds it, and the process becomes a kernel-stuck "terminating
        # forever" corpse that pins ~5GB of VRAM and its handles until Windows
        # reboots. This path ALSO skipped the native release, the web socket,
        # and the singleton — so the upgrade's relaunched JARVIS would boot into
        # a held :8766 and a held singleton. Same hardened sequence the restart
        # uses. clean=True: upgrade_jarvis.py relaunches JARVIS itself, so the
        # watchdog must NOT also resurrect us (that would double-boot).
        try:
            time.sleep(3.0)
            _fs = threading.Timer(30.0, _hard_exit_via_bc, args=(bc, 0, True))
            _fs.daemon = True
            _fs.start()
            _stop_web_interface_quietly()
            try:
                bc._release_singleton()
            except Exception as e:
                print(f"  [upgrade] singleton release failed: {e}")
            _release_native_resources(bc)
            time.sleep(1.0)
        except Exception:
            logging.exception("_self_exit teardown failed")
        _hard_exit_via_bc(bc, 0, clean=True)
    threading.Thread(target=_self_exit, daemon=True).start()

    return (
        f"upgrade initiated for {pending} pending task(s) - "
        f"I'll shut down so Claude Code can take over. "
        f"A new window will open showing the work, and I'll relaunch automatically when done."
    )


# ─── Graceful shutdown (Phase 4K) ──────────────────────────────────────

def _hard_exit_via_bc(bc, code: int = 0, clean: bool = False) -> None:
    """Un-deadlockable exit via the monolith's _hard_exit (TerminateProcess).
    os._exit → ExitProcess walks DLL_PROCESS_DETACH under the LOADER LOCK —
    a thread wedged in a CUDA/driver DLL holds it and the process becomes an
    immortal zombie (pid 14608 lived 22h past its 'Session ended' banner;
    the very next restart reproduced it — 2026-07-12 py-spy dumps). Falls
    back to os._exit if the helper is absent (older monolith)."""
    fn = getattr(bc, "_hard_exit", None)
    if fn is not None:
        try:
            fn(code, clean=clean)
            # The real helper never returns; a test double does — don't
            # fall through and kill the TEST process.
            return
        except Exception:
            pass
    os._exit(code)


def _release_audio_streams(bc, budget_s: float = 3.0) -> None:
    """Release the PortAudio/WASAPI streams this process actually owns, before
    TerminateProcess lands. Bounded by ``budget_s``; never raises.

    H-3, FIFTH COPY (2026-08-20). This step used to be one line:

        try: bc.sd.stop()                 # WASAPI streams
        except Exception: pass

    Four copies of that escape hatch were deleted on 2026-08-20
    (bobert_companion._safe_close_stream, core/wake_word, skills/ambient_listen,
    skills/enroll_voice); this one was missed, and it is the only one whose
    stated purpose DEPENDED on the claim those deletions disproved. Module-level
    ``sd.stop()`` (sounddevice 0.5.5, sounddevice.py:406-418) does not touch
    "WASAPI streams": it stops+closes ``_last_callback`` and nothing else, and
    ``_last_callback`` is published in exactly one place —
    ``_CallbackContext.start_stream``, reached only from
    sd.play()/sd.rec()/sd.playrec(). Every stream alive at shutdown here is an
    explicitly constructed InputStream/OutputStream (the wake-word detector's
    persistent stream, ambient listen's mic + WASAPI-loopback streams,
    record_speech's stream), so the call was a NO-OP for all of them — the
    function went on to TerminateProcess with threads still inside the WASAPI
    driver, which is precisely the kernel-stuck 'terminating forever' corpse
    _release_native_resources exists to prevent.

    And when ``_last_callback`` was NOT empty it was worse than a no-op: the
    goodbye line is spoken two seconds earlier, so a queued proactive/timer
    _speak can be mid-playback with play_with_lipsync's single-toucher reaper
    owning that stream — sd.stop() would then stop+close it from a SECOND
    thread (0xc0000374, mid-teardown). Since H-7 the reaper's stream is
    detached from ``_last_callback`` at handoff, so the call is now provably a
    no-op in every case: it can only ever act on a stream nobody owns.

    Teardown does NOT get an exemption from the H-3 rule. What it gets instead
    is this: ask each owner to close its own stream, then wait — bounded — for
    the ownership flags to actually drop. The signalling runs on a daemon so a
    wedged owner costs ``budget_s``, not the caller's whole failsafe window
    (the ambient stops join 3 s each internally); an abandoned daemon dies with
    the process, the same idiom as every other abandonable close here."""
    done = {}

    def _signal():
        # (0) the playback keeper's silent speaker stream (PLAYBACK_KEEPER,
        #     2026-10-05): latch it off; its own thread closes the stream and
        #     drops _tts_keeper_active, which the wait below watches. FIRST
        #     (review 2026-10-09): shutdown() only sets a latch and never
        #     blocks, so its ~10 ms close then runs while (a)-(c) stop -- the
        #     ambient stops below join up to 3 s each, and as the last step
        #     the keeper's close could still be inside the driver when
        #     TerminateProcess lands.
        try:
            fn = getattr(bc, "_playback_keeper_shutdown", None)
            if callable(fn):
                fn()
                done["playback_keeper"] = True
        except Exception:
            pass
        # (a) record_speech / the main loop. The watchdog reset signal is the
        #     documented way to make it wake from audio_q.get, close its
        #     InputStream and return (see _main_loop_watchdog_thread).
        try:
            bc._watchdog_reset_signal.set()
            done["record_speech"] = True
        except Exception:
            pass
        # (b) the wake-word detector's persistent InputStream — the one stream
        #     that is open for the entire session.
        try:
            wl = sys.modules.get("skill_wake_listener")
            det = getattr(wl, "_detector", None) if wl is not None else None
            stop = getattr(det, "stop", None)
            if callable(stop):
                stop()
                done["wake_word"] = True
        except Exception:
            pass
        # (c) ambient listen's mic and WASAPI-loopback workers. Both entry
        #     points are already bounded (3 s joins) and idempotent when the
        #     worker is not running.
        try:
            al = sys.modules.get("skill_ambient_listen")
            for name in ("ambient_listen_stop", "ambient_audio_stop"):
                fn = getattr(al, name, None) if al is not None else None
                if callable(fn):
                    fn("")
                    done[name] = True
        except Exception:
            pass

    try:
        import threading as _threading
        t = _threading.Thread(target=_signal, name="audio-release", daemon=True)
        t.start()
        t.join(timeout=budget_s)
    except Exception:
        # Could not even spawn the helper (thread exhaustion): do it inline
        # rather than skipping the release entirely.
        try:
            _signal()
        except Exception:
            pass

    # Now wait — bounded — for the owner flags to actually drop, so
    # TerminateProcess lands with nothing inside the driver. A flag that never
    # clears means a native call really is still running; we report that
    # instead of pretending the release worked.
    deadline = time.time() + budget_s
    live = True
    while time.time() < deadline:
        try:
            with bc._mic_lock:
                live = bool(bc._pa_streams_live())
        except Exception:
            live = False        # no gate on this host — nothing to wait for
            break
        if not live:
            break
        time.sleep(0.05)
    if live:
        print("  [teardown] audio streams did NOT all release in "
              f"{budget_s:.1f}s (a native call is still in flight) — "
              f"terminating anyway")
    elif done:
        print(f"  [teardown] audio streams released ({', '.join(sorted(done))})")


def _filler_teardown_via_bc(bc, reason: str) -> None:
    """Latch the monolith's processing filler off (bc._filler_teardown) on a
    restart / shutdown / teardown entry. Tolerates a host without the hook
    (older monoliths, test doubles). Never raises. 2026-09-29."""
    try:
        fn = getattr(bc, "_filler_teardown", None)
        if callable(fn):
            fn(reason)
    except Exception:
        pass


def _release_native_resources(bc) -> None:
    """Best-effort release of every native/driver resource BEFORE process
    termination. TerminateProcess (v2.0.51) prevents the ExitProcess
    loader-lock deadlock, but a thread parked in an uncompletable DRIVER
    call (CUDA / WASAPI / Kinect) still can't be reaped — the process
    becomes a kernel-stuck 'terminating forever' corpse that pins its VRAM
    and handles until Windows reboots. Live 2026-07-13: two same-day
    restarts left two such corpses holding ~11GB of the 3090, starving the
    local brain into 50s generate timeouts. Releasing the drivers first
    gives termination nothing to snag on. Every step is guarded; the
    caller's failsafe Timer still guarantees death regardless."""
    # Ctrl-C / every hardened path: no processing-filler clip may start while
    # the natives are being released. 2026-09-29.
    _filler_teardown_via_bc(bc, "teardown")
    # WAIT FOR IN-FLIGHT GPU WORK FIRST (2026-07-14). unload() drops the cached
    # model and empties the CUDA cache — but it cannot release a model that is
    # still LOADING. TerminateProcess then lands while a thread sits inside the
    # CUDA driver (from_pretrained streaming weights to the GPU, or a generate
    # mid-flight) and that thread cannot be reaped: the process becomes the
    # kernel-stuck corpse this whole teardown exists to prevent.
    #
    # Live proof: a restart issued 83 SECONDS after boot — while chatterbox was
    # still warming — corpsed the instance even WITH the v2.0.57 release,
    # whereas a restart of a settled instance released 4.5GB and left nothing.
    # The voice-clone single-flight guard already tracks exactly this state, so
    # wait for it (bounded — the caller's failsafe timer still guarantees death).
    try:
        inflight = getattr(bc, "_voice_clone_inflight", None)
        if inflight is not None:
            deadline = time.time() + 20.0
            waited = False
            while inflight[0] and time.time() < deadline:
                waited = True
                time.sleep(0.25)
            if waited:
                state = "finished" if not inflight[0] else "STILL RUNNING"
                print(f"  [teardown] waited for the voice-clone GPU worker "
                      f"({state})")
    except Exception:
        pass
    try:
        from core import voice_clone as _vc
        _vc.unload()                      # chatterbox CUDA context
    except Exception:
        pass
    try:
        from audio import kinect_bridge as _kb
        # final=True: clear the enable flag FIRST so an in-flight 30 Hz pump
        # tick can't re-open the sensor (and spawn a fresh pump) microseconds
        # before TerminateProcess — a thread holding a live Kinect driver
        # handle at termination is exactly what corpses the process.
        # 2026-07-14 audit.
        try:
            _kb.close(final=True)
        except TypeError:                 # older bridge without the kwarg
            _kb.set_enabled(False)
    except Exception:
        pass
    _release_audio_streams(bc)
    try:
        bc._face_track_stop.set()         # camera caps (thread releases them)
    except Exception:
        pass
    _flush_persistent_stores()


def _flush_persistent_stores(timeout: float = 1.5) -> None:
    """Land the session's queued disk writes before TerminateProcess
    (review 2026-10-09): the clone cache's line ledger, seed budget and
    take gate (otherwise saved only after 60 s of quiet, or every 5 min)
    and its queued takes, the screen timeline's queued rows and the vision
    trace's queued entries - a session that ended mid-conversation lost
    them all. Last in the teardown, after every driver is released, and
    bounded by ``timeout`` in total (the caller's failsafe timer still
    guarantees death). The writer daemons hold no driver handle, so they
    are not stopped. Only modules already loaded are touched. Never
    raises."""
    deadline = time.time() + max(0.0, float(timeout))

    def left() -> float:
        return max(0.0, deadline - time.time())

    try:
        cvc = sys.modules.get("core.clone_voice_client")
        client = getattr(cvc, "CLIENT", None) if cvc is not None else None
        if client is not None:
            for obj in (getattr(client, "ledger", None),
                        getattr(client, "budget", None)):
                try:
                    if obj is not None:
                        obj.save_if_dirty()
                except Exception:
                    pass
            try:
                client.store.save_gate()
            except Exception:
                pass
            try:
                client.store.flush(left())
            except Exception:
                pass
    except Exception:
        pass
    try:
        tlm = sys.modules.get("core.screen_timeline")
        holder = getattr(tlm, "_singleton", None) if tlm is not None else None
        tl = holder.get("tl") if isinstance(holder, dict) else None
        if tl is not None:
            tl.flush(left())
    except Exception:
        pass
    try:
        vt = sys.modules.get("core.vision_trace")
        if vt is not None:
            vt.flush(left())
    except Exception:
        pass


def _stop_web_interface_quietly() -> None:
    """Release the web dashboard's listening socket BEFORE process exit.
    A kernel-stuck terminating process keeps its HANDLES open — live
    2026-07-12: a restarted-away instance held :8766 as a corpse, the
    replacement's autostart refused to co-bind, and the dashboard stayed
    dead until reboot. skills.web_interface._stop() is time-boxed (5s) so
    this can never stall a teardown. Best-effort, never raises."""
    try:
        mod = sys.modules.get("skill_web_interface")
        if mod is not None and getattr(mod, "_httpd", None) is not None:
            mod._stop()
            print("  [shutdown] web-interface socket released")
    except Exception:
        pass


def _act_shutdown_jarvis(_: str = "") -> str:
    """Graceful full shutdown — speak goodbye, terminate every JARVIS
    subprocess we spawned, flush state, release the singleton lock, then
    hard-exit (TerminateProcess — see _hard_exit_via_bc)."""
    import random
    import threading
    bc = _bc()
    bc._sleep_mode[0] = True
    _filler_teardown_via_bc(bc, "shutdown")

    try:
        line = random.choice(bc.SHUTDOWN_GOODBYE_LINES)
        bc._speak(line)
    except Exception as _e:
        print(f"  [shutdown_jarvis] goodbye TTS failed: {_e}")

    def _do_shutdown():
        # FAILSAFE: the teardown steps below are not time-bounded (the audio
        # release and the HUD kills can block in native code forever, and the
        # CUDA in-flight wait alone allows 20 s) — arm an independent
        # hard kill so a wedged step can never strand a half-shut-down
        # immortal process. clean=True: this path only runs on an
        # INTENTIONAL stop, so the watchdog must not resurrect.
        _fs = threading.Timer(25.0, _hard_exit_via_bc, args=(bc, 0, True))
        _fs.daemon = True
        _fs.start()
        try:
            time.sleep(2.0)
            print("  [shutdown_jarvis] beginning graceful teardown")
            _stop_web_interface_quietly()
            # CUDA/Kinect first — a thread parked in a driver at terminate
            # time corpse-pins the VRAM until reboot (2026-07-13).
            _release_native_resources(bc)
            # (No `bc.sd.stop()` here: it was a second copy of a call that
            # cannot free any stream this process owns, and
            # _release_native_resources already ran the real release.
            # _face_track_stop is re-set because it costs nothing and the
            # helper's own attempt is inside a swallowing try.)
            try: bc._face_track_stop.set()
            except Exception: pass
            try: bc._focus_tracker_stop.set()
            except Exception: pass
            try:
                from core import diagnostic_daemons as _diag_daemons
                _diag_daemons.stop_diagnostic_daemons()
            except Exception as _e:
                print(f"  [shutdown_jarvis] diag daemons stop failed: {_e}")
            try: bc.set_state("sleep")
            except Exception: pass
            for _hud_kill in (bc._shutdown_hud, bc._shutdown_tray,
                              bc._shutdown_reticle_overlay):
                try: _hud_kill()
                except Exception as _e:
                    print(f"  [shutdown_jarvis] {_hud_kill.__name__} failed: {_e}")
            try: bc.save_session_pattern()
            except Exception as _e:
                print(f"  [shutdown_jarvis] save_session_pattern failed: {_e}")
            try:
                _mem_snapshot = bc.load_memory()
                saver = threading.Thread(
                    target=bc.save_session_to_memory, args=(_mem_snapshot,),
                    daemon=True,
                )
                saver.start()
                saver.join(timeout=8)
                if saver.is_alive():
                    print("  [shutdown_jarvis] session save timed out — exiting anyway")
            except Exception as _e:
                print(f"  [shutdown_jarvis] session save spawn failed: {_e}")
            try: bc._restore_prior_power_plan()
            except Exception: pass
            try: bc._release_singleton()
            except Exception as _e:
                print(f"  [shutdown_jarvis] _release_singleton failed: {_e}")
            try: bc.close_log()
            except Exception: pass
            print("  [shutdown_jarvis] clean exit complete — hard-exiting")
        finally:
            _hard_exit_via_bc(bc, 0, clean=True)

    threading.Thread(target=_do_shutdown, daemon=True).start()
    return "Going dark, sir."


# ─── LLM backend switching (Phase 4K) ──────────────────────────────────

def _apply_chat_brain(bc, route: str, backend: str | None) -> None:
    """Point BOTH chat-brain knobs at one brain, live (2026-10-01).

    _call_llm picks its branch from MODEL_ROUTING['chat'] FIRST
    (_chat_takes_local_branch -> core.config.model_route) and only then from
    AI_BACKEND, while _claude_reachable() lets a cloud call through on
    AI_BACKEND alone. switch_llm used to set only AI_BACKEND — so the tray's
    "Switch to Claude" ticked Claude, said "switched to claude" and every turn
    stayed local (now with a paid cloud fallback) — and model_picker.set_brain
    set only the route, so "use Claude" landed on the ollama branch. Both call
    this now. The routing dicts are mutated IN PLACE (model_route reads
    core.config's; model_picker reads the monolith's star-imported alias),
    never replaced. `bc` may be None (no monolith): only core.config moves.
    `backend` None moves the route only and leaves AI_BACKEND as it is —
    set_brain('local'): route=local alone keeps chat local, and AI_BACKEND is
    the GLOBAL cloud switch (vision, the orchestrator, create_skill...)."""
    targets = []
    if bc is not None:
        try:
            if backend is not None:
                bc.AI_BACKEND = backend
            routing = getattr(bc, "MODEL_ROUTING", None)
            if isinstance(routing, dict):
                targets.append(routing)
            else:
                setattr(bc, "MODEL_ROUTING", {"chat": route})
        except Exception:
            pass
    try:
        import core.config as _cfg
        routing = getattr(_cfg, "MODEL_ROUTING", None)
        if isinstance(routing, dict) and all(routing is not t for t in targets):
            targets.append(routing)
    except Exception:
        pass
    for routing in targets:
        routing["chat"] = route


# switch_llm is runtime-only; its reply says so (2026-10-01, see below).
_SWITCH_SESSION_NOTE = " for this session (a restart returns to the saved backend)"


def _act_switch_llm(arg: str = "") -> str:
    """Switch AI_BACKEND between Claude and a local Ollama model.
    arg formats: 'claude' | 'anthropic' | '<ollama-model-tag>' (e.g.
    'qwen2.5:14b'). Validates against a known-tag allowlist before
    mutating the global — unknown tags are rejected so a typo can't
    silently put JARVIS into a backend that won't reply.

    Mutation contract: bobert_companion.py does `from core.config import *`
    at boot, which copies AI_BACKEND + OLLAMA_MODEL into its own
    namespace. Setting `bc.AI_BACKEND = "ollama"` here mutates that
    namespace; every other read in bobert_companion sees the new value
    via its own globals. core.config.AI_BACKEND stays at the boot value.
    The chat ROUTE, though, IS read from core.config on every turn
    (model_route('chat')), so the switch moves MODEL_ROUTING['chat'] with
    the backend — see _apply_chat_brain (2026-10-01). Runtime-only: neither
    knob is persisted, so a restart returns to the saved pair together — and
    the reply says so (_SWITCH_SESSION_NOTE): the web Settings panel shows
    the live route against the file as "pending restart", which reads as if
    a restart would APPLY the switch when it undoes it.
    """
    bc = _bc()
    from core.config import CLAUDE_MODEL
    tag = (arg or "").strip().lower()
    if not tag:
        backend = bc.AI_BACKEND
        model = CLAUDE_MODEL if backend == "claude" else bc.OLLAMA_MODEL
        return f"current backend: {backend} (model: {model})"
    # Publish the active backend to hud_state so the tray's AI submenu shows a
    # checkmark on the live model. The tray reads `llm_backend`: "anthropic" for
    # Claude, otherwise the ollama tag it matches via .startswith() (qwen…/llama…).
    def _publish_backend(value: str) -> None:
        try:
            bc._write_hud_state(llm_backend=value)
        except Exception:
            pass
        # Brain glow: the HUD takes the colour of the brain the next turn
        # uses (core/brain_glow; a no-op when nothing changed).
        try:
            from core.brain_glow import publish_expected
            publish_expected(bc, source="switch")
        except Exception:
            pass
    if tag in ("claude", "anthropic"):
        _apply_chat_brain(bc, "cloud", "claude")
        _publish_backend("anthropic")
        return f"switched to claude ({CLAUDE_MODEL}){_SWITCH_SESSION_NOTE}"
    def _resolved_local() -> str:
        # What the local brain will ACTUALLY use next turn (the resolver cache),
        # not the vestigial OLLAMA_MODEL constant. 2026-07-14 bug-hunt.
        try:
            return bc._get_local_llm_model()
        except Exception:
            return getattr(bc, "OLLAMA_MODEL", "")

    if tag == "ollama":
        _apply_chat_brain(bc, "local", "ollama")
        model = _resolved_local()
        _publish_backend(model)
        return f"switched to ollama (model: {model}){_SWITCH_SESSION_NOTE}"
    # explicit model tag — resolve it against what Ollama has INSTALLED first.
    # The tray's Local Model picker lists Ollama's own /api/tags, so every tag
    # it sends is installed; the family-prefix allowlist below used to be the
    # only gate, and refused installed models outside it (gpt-oss:20b,
    # laguna-xs-2.1) with "unknown backend tag" while the voice picker
    # (skills/model_picker.set_model) switched to them (2026-10-01). The
    # allowlist now only decides whether an UNINSTALLED, free-typed tag is
    # worth a background pull. An embedding model is never a chat brain.
    concrete = None
    try:
        concrete = bc._ollama_resolve_model(tag)
    except Exception:
        concrete = None
    if not isinstance(concrete, str) or not concrete:
        concrete = None
    else:
        try:
            from skills.model_picker import _is_embed
            embed = _is_embed(concrete)
        except Exception:
            embed = "embed" in concrete.lower()
        if embed:
            concrete = None
    if concrete or tag in bc._KNOWN_OLLAMA_MODELS or any(tag.startswith(p) for p in
            ("llama", "qwen", "mistral", "mixtral", "phi", "gemma",
             "deepseek", "codellama")):
        _apply_chat_brain(bc, "local", "ollama")
        bc.OLLAMA_MODEL = tag
        # MAKE THE PICK AUTHORITATIVE (2026-07-14 bug-hunt). Setting OLLAMA_MODEL
        # alone had NO effect — every generation resolves through
        # _get_local_llm_model(), which reads the _RESOLVED_LOCAL_LLM_MODEL cache
        # and never looks at OLLAMA_MODEL. So "switch to qwen2.5:14b" reported
        # success while the box kept answering on gemma4:12b. Repoint the
        # resolver cache the same way skills/model_picker.set_model does — but
        # ONLY at the CONCRETE installed tag (2026-07-21 audit): the old
        # base-name installed check passed for any sibling tag, pinning e.g.
        # 'qwen2.5:14b' while the box only has 'qwen2.5:14b-instruct-q5_K_M' —
        # a 404 on every later turn with no recovery until restart. Resolve
        # the request to what Ollama actually has; if nothing matches, kick a
        # background pull and leave the working model in place.
        old = _resolved_local()   # the OLD chat tag, before any cache repoint
        if concrete:
            cache = getattr(bc, "_RESOLVED_LOCAL_LLM_MODEL", None)
            if isinstance(cache, list):
                cache[0] = concrete
            try:
                bc.LOCAL_LLM_MODEL = concrete
            except Exception:
                pass
            # Vision LOCKSTEP (2026-07-21 audit, same rule as set_model):
            # when vision shared the OLD chat tag, carry it to the new tag —
            # live only (persist=False: this branch doesn't persist the chat
            # tag either, so persisting vision alone would desync
            # user_settings.json on restart). Reuses the ONE helper rather
            # than minting another copy of the rule.
            try:
                from skills.model_picker import _sync_vision_to_chat
                _sync_vision_to_chat(old, concrete, persist=False, bc=bc)
            except Exception:
                pass
            _publish_backend(concrete)
            return f"switched to ollama / {concrete}{_SWITCH_SESSION_NOTE}"
        # Not installed yet — pull in the background, keep the current model.
        try:
            bc._ollama_pull_async(tag)
        except Exception:
            pass
        _publish_backend(_resolved_local())
        return (f"'{tag}' isn't installed yet, sir — pulling it in the "
                f"background. Staying on {_resolved_local()} until it's ready.")
    # Derive the suggestion from config so it can't drift from the shipped
    # default brain (2026-07-21 audit — it used to name retired tags).
    from core.config import LOCAL_LLM_MODEL as _local_default
    return (f"unknown backend tag: {tag!r}. "
            f"Use 'claude', 'ollama' (local default: {_local_default}), or an "
            f"installed tag (qwen.../llama.../gemma...)")


# ─── Vision: find on screen (Phase 4C) ─────────────────────────────────

def _act_find_on_screen(description: str) -> str:
    """find_on_screen, <what> - FIND (never click) the thing by name in the
    visible windows (core.grounded_click, mode "find"): facts the follow-up
    round quotes - "found '<label>' on the middle monitor in '<window>'
    [uia]", several matches by name, or not found plus what IS there."""
    from core import grounded_click as _gc
    bc = _bc()
    monitor, description = bc._parse_monitor_prefix(description)
    target = f" on {monitor} monitor" if monitor else ""
    print(f"  [vision] Looking for '{description}'{target}...", flush=True)
    arg = f"monitor:{monitor}|{description}" if monitor else description
    r = _gc.run_bounded(arg, said=_turn_said(bc), mode="find")
    print(f"  [vision] {r.text[:160]}", flush=True)
    _note_screen_look("find_on_screen", r.text)
    return r.text


# ─── Cache / mode toggles (Phase 4B) ───────────────────────────────────

def _act_clear_llm_cache(_: str = "") -> str:
    """The pipeline does not maintain an in-process LLM response cache —
    Claude's prompt cache is server-side and Ollama caches at the daemon
    layer. Nothing for us to evict here; report so the user knows."""
    return "no in-process LLM cache to clear (Claude/Ollama cache server-side)"


def _act_ambient_mode_toggle(_: str = "") -> str:
    """Flip the ambient mode flag — the natural 'ambient mode' voice command."""
    # Call _act_ambient_mode_set DIRECTLY (same module) rather than via
    # bc._act_ambient_mode_set. The bc.* form only resolved because
    # bobert_companion does `from core.actions import *`; if that wildcard were
    # ever dropped the voice toggle would silently break (2026-05-30 audit).
    # The runtime flag still comes from bc — it's the shared state slot.
    # Flip from what is REALLY running, the tray's rule (2026-10-01): with
    # AMBIENT_LISTEN_ENABLED the daemon auto-starts while _ambient_mode_active
    # stays False, so flipping the cell turned a "stop" into a no-op "start"
    # ("Ambient mode active, sir") and left the room mic transcribing. The tray
    # copy was fixed on 2026-09-30; this one was missed. One helper now.
    return _act_ambient_mode_set(not _bc()._ambient_effective_on())


__all__ = [
    # Phase 4A
    "_act_open_url",
    "_act_web_search",
    "_act_youtube",
    "_act_get_time",
    "_act_screenshot",
    "_act_media_next",
    "_act_media_prev",
    "_act_media_playpause",
    "_act_volume_up",
    "_act_volume_down",
    "_act_volume_mute",
    "_act_volume_unmute",
    "_act_set_volume",
    # Phase 4B — streaming
    "_act_netflix",
    "_act_prime_video",
    "_act_disney_plus",
    "_act_hulu",
    "_act_max",
    "_act_spotify",
    "_act_youtube_play",
    # Phase 4B — HUD
    "_act_hide_hud",
    "_act_show_hud",
    "_act_toggle_hud",
    # Phase 4B — diagnostics
    "_act_test_mic",
    "_act_test_tts",
    "_act_test_vision",
    # Phase 4B — misc
    "_act_clear_llm_cache",
    "_act_ambient_mode_toggle",
    # Phase 4C — task / restart / session
    "_act_clear_tasks",
    "_act_session_resume",
    "_act_restart",
    # Phase 4C — LLM picker stubs
    "_act_switch_llm_picker",
    "_act_show_llm_stats",
    "_act_model_costs",
    "_act_running_costs",
    # Phase 4C — UI primitives
    "_act_press",
    "_act_scroll",
    # Phase 4C — misc
    "_act_list_skills",
    "_act_apple_music",
    "_act_find_on_screen",
    # 2026-10-05 - grounded screen clicks, undo, developer notes, screen memory
    "_act_click_on_screen",
    "_act_undo_click",
    "_act_note_for_claude",
    "_act_screen_memory",
    "_act_forget_screen",
    # Phase 4D — app launching
    "_act_launch_app",
    # Phase 4D — Apple Music transport (media keys; classic iTunes COM is dead)
    "_act_pause_music",
    "_act_resume_music",
    "_act_now_playing",
    # New UWP Apple Music app — launch + status
    "_act_open_apple_music",
    "_act_music_status",
    # Phase 4D — task queue add
    "_act_queue_task",
    # Phase 4D — window management
    "_act_list_windows",
    "_act_focus_window",
    "_act_minimize_window",
    "_act_close_window",
    "_act_close_last_opened",
    # "close / minimize all windows except X" (2026-10-03) + the pushback count
    "_act_close_all_windows_except",
    "_act_minimize_all_windows_except",
    "_close_all_windows_except_preview",
    # Named closes, misheard names and "you forgot X" (2026-10-05)
    "_find_app_windows",
    "_names_open_window",
    "_named_close_state",
    "_close_window_preview",
    "_window_name_suggestion",
    "_close_all_windows_except_suggestion",
    "_close_name_question",
    "_last_bulk_close",
    # Phase 4D — UI type (with shell-cmd refusal)
    "_act_type",
    # Phase 4E — music skip/back
    "_act_next_song",
    "_act_previous_song",
    # Phase 4E — task queue read
    "_act_show_tasks",
    # Phase 4E — ambient mode setter (called by ambient_mode_toggle above)
    "_act_ambient_mode_set",
    # New-people greeting live toggle (GREET_NEW_PEOPLE_ENABLED)
    "_act_greet_new_people_set",
    # Phase 4E — skills reload
    "_act_reload_skills",
    # Phase 4E — memory introspection
    "_act_show_recent_facts",
    "_act_export_memory",
    # Phase 4E — diagnostic tray wrappers
    "_act_run_diagnostic_tray",
    "_act_show_last_diagnostic",
    # Phase 4F — streaming dispatcher
    "_act_play_streaming",
    "_act_streaming_search",
    # Phase 4F — UI click + hotkey
    "_act_click",
    "_act_hotkey",
    # Phase 4F — pipeline + backup + memory reset
    "_act_stop_pipeline",
    "_act_force_backup",
    "_act_reset_memory",
    # Phase 4G — version + smoke test + selftest + memory forget + latency
    "_act_version_info",
    "_act_check_for_updates",
    "_act_report_bug",
    "_act_run_smoke_test",
    "_act_test_each_skill",
    "_act_forget_last_hour",
    "_act_latency_benchmark",
    # Phase 4H — music routing, webcam, vision, replay, shell
    "_act_play_music",
    "_act_where_is_user",
    "_act_see_screen",
    "_act_replay_last_action",
    "_act_run_shell",
    # Phase 4I — webcam vision + session recall + cached-screen + changelog
    "_act_see_user",
    "_act_which_monitor",
    "_act_session_memory_recall",
    "_act_recall_screen",
    "_act_read_changelog",
    # Phase 4J — overnight kick, window placement, skill creation
    "_act_start_overnight_upgrade",
    "_act_open_on_monitor",
    "_act_move_window_to_monitor",
    "_act_create_skill",
    # Phase 4K — final 3: upgrade kickoff, graceful shutdown, LLM switch
    "_act_upgrade",
    "_act_shutdown_jarvis",
    "_act_switch_llm",
]
