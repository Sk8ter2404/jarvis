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
import time
import urllib.parse
import webbrowser


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

def _act_open_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    # Small wait so the page has time to start loading before any follow-up
    # see_screen is triggered by the informative-action follow-up loop.
    time.sleep(3.0)
    return f"opened {url} — use see_screen to read what loaded"


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
            webbrowser.open(yt_url)
            time.sleep(3.0)
            return (
                f"opened {yt_url} (extracted from Google results for '{query}') — "
                f"video is now playing, no further action needed"
            )
        # Extraction failed (network, rate-limit, parse miss). Fall through.

    url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
    webbrowser.open(url)
    # Brief wait for page load before follow-up see_screen captures the results.
    time.sleep(3.0)
    return f"opened Google search for '{query}' — use see_screen to read the results"


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
        subprocess.run(["powershell", "-Command", ps], capture_output=True, timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return f"screenshot saved to {path}"
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


def _act_volume_mute(_: str = "") -> str:
    bc = _bc()
    pag = bc._get_pyautogui()
    if pag:
        pag.press("volumemute")
        return "mute toggled"
    return "pyautogui unavailable"


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


def _set_unified_hud_hidden(hidden: bool) -> None:
    """Set the unified HUD's own ✕-button 'hidden' flag in its control file so
    its close-button hide and the voice show/hide commands stay in sync."""
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
    except Exception:
        pass


def _act_show_hud(_: str = "") -> str:
    """Re-display the HUD after a previous hide_hud (or a ✕-button close)."""
    _bc()._write_hud_state(visible=True)
    _set_unified_hud_hidden(False)   # clear a ✕-button hide too
    return "HUD restored, sir."


def _act_toggle_hud(_: str = "") -> str:
    """Toggle HUD visibility — useful as a single voice command."""
    bc = _bc()
    try:
        with bc._hud_state_lock:
            currently_visible = bool(bc._hud_state_cache.get("visible", True))
    except Exception:
        currently_visible = True
    bc._write_hud_state(visible=not currently_visible)
    if currently_visible:
        return "HUD hidden, sir."
    # Toggling back to visible must also clear a ✕-button hide, exactly like
    # _act_show_hud — otherwise the persisted 'hidden' latch keeps the window
    # down and the toggle silently fails to bring it back.
    _set_unified_hud_hidden(False)
    return "HUD restored, sir."


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


def _act_restart(_: str = "") -> str:
    """Relaunch bobert_companion.py in a fresh process and exit this one."""
    import threading
    bc = _bc()
    script = os.path.abspath(bc.__file__)

    def _do_restart():
        time.sleep(1.5)
        # FAILSAFE: if anything below wedges, die anyway — a lingering old
        # instance is exactly what turned the 2026-07-12 restart into a
        # zombie + a singleton-suicided replacement (nothing left running).
        # NO clean flag: if the relaunch failed too, the watchdog SHOULD
        # resurrect us within its 5-minute poll.
        _fs = threading.Timer(20.0, _hard_exit_via_bc, args=(bc, 0, False))
        _fs.daemon = True
        _fs.start()
        # Release the web socket BEFORE spawning: a kernel-stuck corpse keeps
        # its handles, and the replacement's dashboard autostart refuses to
        # co-bind a port a dead process still holds (live 2026-07-12).
        _stop_web_interface_quietly()
        # Release the singleton BEFORE spawning: the replacement boots in
        # ~1s and its singleton check must not see this (possibly wedged,
        # not-yet-dead) process and suicide — live failure mode 2026-07-12:
        # old hung in ExitProcess, new exited via singleton, JARVIS gone.
        try:
            bc._release_singleton()
        except Exception as e:
            print(f"  [restart] singleton release failed: {e}")
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
            )
        except Exception as e:
            print(f"  [restart] relaunch failed: {e}")
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


def _act_show_llm_stats(_: str = "") -> str:
    """The active backend + model, with an estimated cost per conversation for
    the current model (from core.model_catalog)."""
    from core.config import AI_BACKEND, CLAUDE_MODEL, OLLAMA_MODEL
    from core import model_catalog
    model = CLAUDE_MODEL if AI_BACKEND == "claude" else OLLAMA_MODEL
    entry = model_catalog.by_id(model)
    if entry is not None:
        c = entry.cost_per_conversation()
        cost = "$0 (local)" if c <= 0 else f"~${c:.2f}/conv"
        return (f"backend={AI_BACKEND}  model={model}  est. {cost} ({entry.tier})."
                f" Say 'model costs' to compare the options.")
    return (f"backend={AI_BACKEND}  model={model}  "
            f"(not in the cost catalog — say 'model costs' for priced options).")


# ─── UI automation primitives (Phase 4C) ───────────────────────────────

def _act_press(key: str) -> str:
    bc = _bc()
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

# Spoken names that should launch the new UWP Apple Music app via its AUMID
# (explorer shell:AppsFolder) rather than a doomed exe / startfile lookup —
# the Store app has no PATH-friendly executable.
_APPLE_MUSIC_LAUNCH_ALIASES = frozenset({
    "apple music", "apple music app", "music app", "applemusic",
    "the apple music app",
})


def _act_launch_app(name: str) -> str:
    bc = _bc()
    # 0) Apple Music (UWP) special-case — launch via AUMID. The Store app
    #    isn't on PATH and os.startfile("apple music") fails, so route it
    #    through the apple_music_app bridge before the generic resolution.
    if re.sub(r"\s+", " ", (name or "").strip().lower()) in _APPLE_MUSIC_LAUNCH_ALIASES:
        amapp = _apple_music_app()
        if amapp is not None:
            ok, err = amapp.launch()
            if ok:
                return "launched Apple Music"
            return f"could not launch Apple Music: {err}"
        # Bridge unimportable — fall through to the generic path below.

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
# paths are dead. Transport now drives whatever player is live with OS-level
# MEDIA KEYS: the Apple Music web app (browser-active fast path) OR the new
# UWP Apple Music app (apple_music_app.is_active_media_app()). Only when
# NOTHING is playing/running do we return an honest line.

_NOTHING_PLAYING_MSG = (
    "Nothing seems to be playing, sir — open Apple Music and I'll take it "
    "from there."
)


def _act_pause_music(_: str = "") -> str:
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


# ─── Open / status for the new UWP Apple Music app ─────────────────────────

def _act_open_apple_music(_: str = "") -> str:
    """Launch the new UWP Apple Music app via its AUMID. 'open Apple Music'."""
    amapp = _apple_music_app()
    if amapp is None:
        return "the Apple Music bridge isn't available, sir."
    if amapp.is_running():
        return "Apple Music is already open, sir."
    ok, err = amapp.launch()
    if ok:
        return "launched Apple Music"
    return f"could not launch Apple Music: {err}"


def _act_music_status(_: str = "") -> str:
    """Report whether the Apple Music app is installed / running and what (if
    anything) it's playing. 'is Apple Music open' / 'music status'."""
    amapp = _apple_music_app()
    if amapp is None:
        return "the Apple Music bridge isn't available, sir."
    running = amapp.is_running()
    if not running:
        # is_installed() is a best-effort PowerShell check; treat False as
        # 'unknown' rather than a hard claim it's missing.
        if amapp.is_installed():
            return "Apple Music is installed but not running, sir."
        return ("Apple Music doesn't appear to be running, sir — say 'open "
                "Apple Music' and I'll start it.")
    try:
        np = amapp.now_playing()
    except Exception:
        np = None
    if np:
        return f"Apple Music is running, sir — now playing {np}."
    return "Apple Music is running, sir, but nothing is playing right now."


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
    """Return all open window titles."""
    try:
        import pygetwindow as gw
    except ImportError:
        return "pygetwindow not available — pip install pygetwindow"
    titles = sorted({w.title for w in gw.getAllWindows() if w.title and w.title.strip()})
    if not titles:
        return "no windows visible"
    return "Open windows:\n" + "\n".join(f"  - {t}" for t in titles)


def _act_focus_window(query: str) -> str:
    """Bring a window to the foreground by partial title match."""
    bc = _bc()
    if not query.strip():
        return "format: focus_window, <window title>"
    matches = bc._find_windows_by_title(query)
    if not matches:
        return f"no window matching '{query}'"
    target = matches[0]
    try:
        target.activate()
        bc._flash_window_reticle(target, "focus")
        return f"focused '{target.title}'"
    except Exception as e:
        # pygetwindow on Windows raises a generic exception even on success
        # (Win32 SetForegroundWindow returns false in some allowed cases).
        # If the error message is "operation completed successfully" it
        # actually worked. Restore + minimize-toggle as a defense-in-depth.
        msg = str(e).lower()
        if "operation completed successfully" in msg or "error code from windows: 0" in msg:
            bc._flash_window_reticle(target, "focus")
            return f"focused '{target.title}'"
        # Try the restore trick as a fallback (works around some flag-set quirks)
        try:
            target.minimize()
            target.restore()
            bc._flash_window_reticle(target, "focus")
            return f"focused '{target.title}' (via restore)"
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


def _act_close_window(query: str) -> str:
    """Close a window by partial title match. Refuses to close Bobert's host."""
    bc = _bc()
    if not query.strip():
        return "format: close_window, <window title>"
    # Self-preservation: refuse if title matches one of the forbidden targets
    if any(target in query.lower() for target in bc.FORBIDDEN_TARGETS):
        return (
            f"REFUSED: '{query}' looks like your own host process. "
            f"Closing it would kill the session. Ask the user to close it manually."
        )
    matches = bc._find_windows_by_title(query)
    if not matches:
        return f"no window matching '{query}'"
    closed = []
    for w in matches:
        # Defence in depth: also check the actual window title we found
        if any(target in (w.title or "").lower() for target in bc.FORBIDDEN_TARGETS):
            continue
        try:
            w.close()
            closed.append(w.title)
        except Exception:
            pass
    return f"closed: {', '.join(closed)}" if closed else "could not close"


# ─── UI type (Phase 4D) ────────────────────────────────────────────────

def _act_type(text: str) -> str:
    # If this looks like a shell command and no terminal is focused, the LLM
    # is trying to "execute" it by typing into whatever window has focus —
    # which could be a chat, a code editor, a browser address bar, anything.
    # Refuse and tell the LLM to use run_shell instead.
    bc = _bc()
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
    # COM is dead → media keys. Browser Apple Music or the new UWP app both
    # respond to the OS nexttrack key.
    bc = _bc()
    if bc._apple_music_chrome_active():
        return _act_media_next()
    amapp = _apple_music_app()
    if amapp is not None and amapp.is_active_media_app():
        return _act_media_next()
    return _NOTHING_PLAYING_MSG


def _act_previous_song(_: str = "") -> str:
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
         injects never write real memory."""
    bc = _bc()
    bc._ambient_mode_active[0] = bool(active)
    bc._write_hud_state(ambient_mode_active=bool(bc._ambient_mode_active[0]))
    action_name = "ambient_listen_start" if bc._ambient_mode_active[0] else "ambient_listen_stop"
    fn = bc.ACTIONS.get(action_name)
    if fn is not None:
        try:
            fn("")
        except Exception as e:
            return f"ambient daemon refused: {e}"
    # Start / stop the fact-extractor alongside the mic daemon so ambient mode
    # genuinely folds overheard speech into long-term memory.
    _staging = getattr(bc, "_is_staging", lambda: False)
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
    return f"Ambient mode {state_word}, sir — Chappie is {'listening quietly and learning' if bc._ambient_mode_active[0] else 'standing down'}."


def _act_greet_new_people_set(on: bool) -> str:
    """Flip GREET_NEW_PEOPLE_ENABLED live, in-process — the proactive 'who are
    all these new people?' greeting fired by skills/face_tracker when it sees
    multiple UNRECOGNISED faces (friends over). Mirrors the ambient_mode_on/off
    setter's idempotent live-toggle shape.

    We set the flag on core.config so the face-tracker poller (which re-reads
    core.config every tick) picks it up WITHOUT a restart. Deliberately NOT
    persisted to user_settings.json — this is a live, session toggle (like the
    wake-word-mode setter), so it cleanly reverts on the next boot to the
    opt-in default. Honest about the face-ID dependency: the greeting needs the
    webcams to actually recognise faces, so it nudges the user to enable
    FACE_ID_ENABLED when that's still off."""
    try:
        import core.config as _cfg
        _cfg.GREET_NEW_PEOPLE_ENABLED = bool(on)
        face_id_on = bool(getattr(_cfg, "FACE_ID_ENABLED", False))
    except Exception as e:   # pragma: no cover - core.config import never fails here
        return f"I couldn't change the new-people greeting, sir — {e}."
    if not on:
        return "Noted, sir — I'll stop announcing new faces."
    msg = ("Will do, sir — when a few unfamiliar faces turn up I'll say hello "
           "once.")
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

def _act_show_recent_facts(_: str = "") -> str:
    """Tail of bobert_memory.json facts list (newest are appended last)."""
    bc = _bc()
    try:
        with bc._memory_lock:
            mem = bc.load_memory()
        facts = mem.get("facts") or []
        if not facts:
            return "no facts in memory yet, sir"
        recent = facts[-10:]
        lines = [f"  {i+1}. {f}" for i, f in enumerate(recent)]
        print("\n[tray] recent facts:")
        for ln in lines:
            print(ln)
        return f"showed {len(recent)} recent fact(s) of {len(facts)} total — see console"
    except Exception as e:
        return f"show_recent_facts failed: {e}"


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
        head = out.split("\n", 1)[0]
        if len(head) > 200:
            head = head[:197] + "..."
        print(f"\n[tray] last diagnostic run (first line): {head}")
        return f"printed last run ({len(out)} chars total) — see console"
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


# ─── UI click + hotkey (Phase 4F) ──────────────────────────────────────

def _act_click(args: str) -> str:
    """args: 'x,y' or 'x,y,right' for right-click, or a description to find+click.
    Coords can be negative (for monitors to the left of the primary, e.g. -2215,249).
    Prefix with 'monitor:NAME|' to restrict vision search to that monitor:
        click, monitor:left|the play button"""
    bc = _bc()
    # Optional monitor prefix
    monitor, args = bc._parse_monitor_prefix(args)

    m = re.match(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*(?:,\s*(left|right|middle))?\s*$", args)
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

    coords = bc.find_click_target(args, monitor=monitor)
    if coords is None:
        target = f"'{args}' on {monitor} monitor" if monitor else f"'{args}'"
        return f"could not locate {target} on screen"
    try:
        bc.ui_click(coords[0], coords[1])
    except bc.UIFailsafeError as e:
        return str(e)
    return f"clicked '{args}' at {coords}"


def _act_hotkey(args: str) -> str:
    bc = _bc()
    keys = [bc._normalize_key(k) for k in args.split("+")]
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


def _act_reset_memory(_: str = "") -> str:
    """Snapshot bobert_memory.json to backups/, then re-initialise it
    to the empty schema. Destructive — but the backup is unconditional,
    so the user can restore by copying the file back."""
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
            bc.save_memory(bc._empty_memory())
        if existed:
            return f"memory reset (backup -> backups/{os.path.basename(backup_path)})"
        return "memory was already empty"
    except Exception as e:
        return f"reset_memory failed: {e}"


# ─── Version info (Phase 4G) ───────────────────────────────────────────

def _act_version_info(_: str = "") -> str:
    """Read data/version.json and report current version + a human-friendly
    rendering of last_upgrade_at (e.g. 'this morning at 8:43 AM')."""
    bc = _bc()
    try:
        from datetime import datetime as _dt
        try:
            from core.version import __version__ as release_ver
        except Exception:
            release_ver = "unknown"
        _ver_path = os.path.join(
            os.path.dirname(os.path.abspath(bc.__file__)),
            "data", "version.json")
        if not os.path.exists(_ver_path):
            return f"I'm on version {release_ver}, sir."
        with open(_ver_path, "r", encoding="utf-8") as _vf:
            data = json.load(_vf)
        ver = release_ver  # single-source release version (core/version.py),
        #                    not the self-upgrade pipeline's internal counter
        ts_iso = data.get("last_upgrade_at") or ""
        # last_upgrade_at is written ONLY by the self-upgrade pipeline —
        # releases deployed via git checkout never touch version.json, so
        # the reported date went stale (live bug: v1.99.0 announced as
        # "last updated on May 30"). The VERSION file's mtime IS the deploy
        # moment (checkout rewrites it on every release), so use whichever
        # of the two is newer.
        ts = None
        try:
            ts = _dt.fromisoformat(ts_iso) if ts_iso else None
        except Exception:
            ts = None
        try:
            _version_file = os.path.join(
                os.path.dirname(os.path.abspath(bc.__file__)), "VERSION")
            _mtime = _dt.fromtimestamp(os.path.getmtime(_version_file))
            if ts is None or _mtime > ts:
                ts = _mtime
        except Exception:
            pass
        if ts is None:
            if ts_iso:
                return f"I'm on version {ver}, last updated {ts_iso}."
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
    """Drop topics/sessions whose timestamp falls in the last hour.
    Facts/projects are intentionally NOT touched — those are durable
    knowledge, not session traces. Held under _memory_lock so it can't
    race with learn_from_turn."""
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
            mem = bc.load_memory()
            old_topics  = list(mem.get("topics") or [])
            old_sessions = list(mem.get("sessions") or [])
            kept_topics  = [t for t in old_topics
                            if _entry_ts(t) < cutoff]
            kept_sessions = [s for s in old_sessions
                             if _entry_ts(s) < cutoff]
            removed = (len(old_topics) - len(kept_topics)
                       + len(old_sessions) - len(kept_sessions))
            if removed == 0:
                return "nothing recent enough to forget"
            mem["topics"]  = kept_topics
            mem["sessions"] = kept_sessions
            bc.save_memory(mem)
        return f"forgot {removed} item(s) from the last hour"
    except Exception as e:
        return f"forget_last_hour failed: {e}"


def _act_latency_benchmark(_: str = "") -> str:
    """Time a single one-shot LLM round-trip via _llm_quick() to give
    the user a feel for current backend latency. Async — Claude
    typically replies in ~1s but Ollama on a cold model can take 5-30s."""
    bc = _bc()
    from core.config import AI_BACKEND, CLAUDE_MODEL, OLLAMA_MODEL

    def _do():
        try:
            t0 = time.time()
            reply = bc._llm_quick(
                system="Reply with exactly the word 'pong' and nothing else.",
                user="ping",
                max_tokens=8,
            )
            ms = (time.time() - t0) * 1000
            backend = AI_BACKEND
            model = CLAUDE_MODEL if backend == "claude" else OLLAMA_MODEL
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
    q = question.strip() or "Describe in detail what is currently on the screen."

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
    result = bc.ask_vision(q, png)
    print(f"  [vision] Got answer ({len(result)} chars)", flush=True)
    bc._push_screen_context(monitor, q, result, {monitor: png})
    return result


# ─── Replay last non-destructive action (Phase 4H) ─────────────────────

def _act_replay_last_action(arg: str = "") -> str:
    """Re-fire the most recent non-destructive action.

    Triggered by the voice phrases 'do that again' / 'replay that' / 'do it
    again'. If `arg` is non-empty it's treated as a monitor identifier
    ('left', 'right', etc.) and substituted into the target action's arg
    where the shape is known.

    Destructive actions (close_window, kill_process, restart, upgrade,
    start_overnight_upgrade, run_shell) are refused — the user must re-issue
    the command so it goes through the normal confirmation/pushback path.
    """
    bc = _bc()
    with bc._action_history_lock:
        if not bc._action_history:
            return "no previous action to replay"
        last = dict(bc._action_history[-1])

    name = last.get("action", "")
    orig_arg = last.get("arg", "") or ""

    if name in bc._DESTRUCTIVE_REPLAY_ACTIONS:
        return (f"refusing to replay destructive action '{name}' without "
                "confirmation — please re-issue the command explicitly")

    fn = bc.ACTIONS.get(name)
    if fn is None:
        return f"cannot replay '{name}' — action no longer registered"

    new_arg = bc._substitute_monitor_in_arg(name, orig_arg, arg) if arg else orig_arg
    try:
        res = fn(new_arg)
    except Exception as e:
        return f"replay of '{name}' failed: {e}"
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

    cam_sides = {
        cam["index"]: ("left" if cam["look_x"] < 0.5 else "right")
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

def _act_session_memory_recall(args: str = "") -> str:
    """Query the session_summaries.json index and return a one-line
    JARVIS-voice answer about what the user was doing in a given time window.

    Triggered by phrases like 'what did we do yesterday', 'remind me what I
    was working on last night', 'what happened this morning'. The free-text
    query (typically the user's full utterance) is parsed for a time
    reference; matching session summaries are then handed to the LLM with a
    JARVIS-voice prompt for a 1-2 sentence reply."""
    bc = _bc()
    query = (args or "").strip()
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

def _act_recall_screen(question: str) -> str:
    """Reference the cached screen context from a recent see_screen.

    Empty question -> returns a JARVIS-style summary of the most recent
    capture ('I last saw your screen 47 seconds ago: ...'). With a question ->
    re-asks vision against the SAME cached images (no recapture, ~instant)
    so the user can ask follow-ups against the visual state JARVIS already
    has in memory. Falls back to a polite refusal when nothing is cached
    or the newest entry is older than SCREEN_CACHE_TTL_SECONDS."""
    bc = _bc()
    recent = bc._recent_screen_contexts()
    if not recent:
        return (
            "I'm afraid I haven't seen the screen in the last 5 minutes, sir — "
            "use see_screen first if you'd like me to take a fresh look."
        )

    entry = recent[0]
    age = time.time() - entry["ts"]
    mon_label = entry["monitor"] or "all monitors"
    age_str = bc._format_screen_age(age)

    q = question.strip()
    if not q:
        # Summary mode: read back what was last seen without burning a vision call.
        history_lines = []
        for e in recent[:3]:
            e_age = bc._format_screen_age(time.time() - e["ts"])
            e_mon = e["monitor"] or "all monitors"
            snippet = e["answer"].strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            history_lines.append(f"- {e_age}, {e_mon}: {snippet}")
        head = (
            f"I last looked at {mon_label} {age_str}, sir. "
            f"What I saw then:"
        )
        return head + "\n" + "\n".join(history_lines)

    # Follow-up mode: re-vision against the cached images so we get a fresh
    # answer to a NEW question without re-capturing.
    images = entry["images"]
    if not images:
        return f"I have the answer from {age_str} cached but no image to re-examine, sir."

    print(
        f"  [vision] Recalling cached screen ({mon_label}, "
        f"{age_str}) — re-asking without recapture...",
        flush=True,
    )
    if len(images) == 1:
        only_png = next(iter(images.values()))
        contextual_q = (
            f"This is a cached screenshot from {age_str}. {q}"
        )
        result = bc.ask_vision(contextual_q, only_png)
    else:
        contextual_q = (
            f"These screenshots are cached from {age_str}. {q}"
        )
        result = bc.ask_vision_multi(contextual_q, images)
    print(f"  [vision] Got cached-recall answer ({len(result)} chars)", flush=True)
    return result


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

def _act_open_on_monitor(args: str) -> str:
    """args format: '<monitor_name> | <url-or-app-name>'
    Opens the URL or launches the app, then moves the resulting window to
    the named monitor and maximizes it."""
    bc = _bc()
    from core.config import MONITORS
    if "|" not in args:
        return "format: open_on_monitor, <monitor_name> | <url-or-app>"
    monitor_name, target = (s.strip() for s in args.split("|", 1))
    monitor_name = monitor_name.lower()

    if monitor_name not in MONITORS:
        return f"unknown monitor '{monitor_name}'. Available: {list(MONITORS.keys())}"
    mx, my, mw, mh = MONITORS[monitor_name]

    try:
        import pygetwindow as gw
    except ImportError:
        return "pygetwindow not available — pip install pygetwindow"
    titles_before = {w.title for w in gw.getAllWindows() if w.title}

    # Launch the target. Treat as URL if explicit scheme or recognisable
    # domain suffix; otherwise treat as an app name.
    _URL_HINT = re.compile(
        r"^(?:https?://|[\w\-]+\.(?:com|net|org|io|gov|edu|co|app|dev|me|tv|ai|so|xyz)(?:/|$))",
        re.IGNORECASE,
    )
    if _URL_HINT.match(target):
        if not bc._open_url_new_window(target):
            webbrowser.open(target if target.startswith(("http://", "https://"))
                            else "https://" + target)
    else:
        _act_launch_app(target)

    # Wait for a window matching the target to appear.
    target_tokens = [
        tok for tok in re.split(r"[\s_\-]+", target.lower()) if len(tok) >= 3
    ]

    def _matches_target(title: str) -> bool:
        t = (title or "").lower()
        return any(tok in t for tok in target_tokens) if target_tokens else False

    new_window = None
    deadline = time.time() + 15.0
    while time.time() < deadline:
        time.sleep(0.2)
        candidates = []
        for w in gw.getAllWindows():
            if not w.title:
                continue
            try:
                if w.width < 200 or w.height < 200:
                    continue   # ignore tiny splash/tooltip windows
            except Exception:
                pass
            is_new   = w.title not in titles_before
            is_match = _matches_target(w.title)
            if is_new or is_match:
                candidates.append((w, is_match, is_new))
        if candidates:
            candidates.sort(key=lambda c: (c[1] and c[2], c[1], c[2]), reverse=True)
            new_window = candidates[0][0]
            break

    if not new_window:
        return f"launched {target}, but couldn't find new window to move it"

    try:
        new_window.restore()
        time.sleep(0.1)
        new_window.moveTo(mx + 50, my + 50)
        time.sleep(0.1)
        new_window.maximize()
    except Exception as e:
        return f"opened {target} but failed to move window: {e}"

    return f"opened '{target}' on {monitor_name} monitor (at {mx},{my})"


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
    if "|" not in args:
        return "format: move_window_to_monitor, <window_title> | <monitor_name>"
    title, monitor_name = (s.strip() for s in args.split("|", 1))
    monitor_name = monitor_name.lower()

    if monitor_name not in MONITORS:
        return f"unknown monitor '{monitor_name}'. Available: {list(MONITORS.keys())}"
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
    from core.config import SKILLS_ENABLED, AI_BACKEND
    if not SKILLS_ENABLED or AI_BACKEND != "claude":
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
        try:
            time.sleep(3.0)
            os._exit(0)
        except Exception:
            logging.exception("_self_exit failed")
            os._exit(0)
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

    try:
        line = random.choice(bc.SHUTDOWN_GOODBYE_LINES)
        bc._speak(line)
    except Exception as _e:
        print(f"  [shutdown_jarvis] goodbye TTS failed: {_e}")

    def _do_shutdown():
        # FAILSAFE: the teardown steps below are not time-bounded (sd.stop /
        # HUD kills can block in native code forever) — arm an independent
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
            try: bc.sd.stop()
            except Exception: pass
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
    via its own globals. core.config.AI_BACKEND stays at the boot value
    (read-only after import) — that's fine: nothing reads from
    core.config at runtime, only the wildcard-copied bobert_companion
    attribute matters.
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
    if tag in ("claude", "anthropic"):
        bc.AI_BACKEND = "claude"
        _publish_backend("anthropic")
        return f"switched to claude ({CLAUDE_MODEL})"
    if tag == "ollama":
        bc.AI_BACKEND = "ollama"
        _publish_backend(bc.OLLAMA_MODEL)
        return f"switched to ollama (model: {bc.OLLAMA_MODEL})"
    # explicit model tag — verify it's one we recognise
    if tag in bc._KNOWN_OLLAMA_MODELS or any(tag.startswith(p) for p in
            ("llama", "qwen", "mistral", "mixtral", "phi", "gemma",
             "deepseek", "codellama")):
        bc.AI_BACKEND = "ollama"
        bc.OLLAMA_MODEL = tag
        _publish_backend(tag)
        return f"switched to ollama / {tag}"
    return (f"unknown backend tag: {tag!r}. "
            f"Use 'claude' or one of: qwen2.5:14b, llama3.1:8b, mistral, ...")


# ─── Vision: find on screen (Phase 4C) ─────────────────────────────────

def _act_find_on_screen(description: str) -> str:
    bc = _bc()
    monitor, description = bc._parse_monitor_prefix(description)
    target = f" on {monitor} monitor" if monitor else ""
    print(f"  [vision] Looking for '{description}'{target}...", flush=True)
    coords = bc.find_click_target(description, monitor=monitor)
    if coords is None:
        print("  [vision] Not found", flush=True)
        return f"could not find '{description}' on screen"
    print(f"  [vision] Found at {coords}", flush=True)
    return f"found at {coords[0]},{coords[1]}"


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
    return _act_ambient_mode_set(not _bc()._ambient_mode_active[0])


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
    # Phase 4C — UI primitives
    "_act_press",
    "_act_scroll",
    # Phase 4C — misc
    "_act_list_skills",
    "_act_apple_music",
    "_act_find_on_screen",
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
