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

def _site_shortcut_url(name: str) -> "str | None":
    """SITE_SHORTCUTS (core/stt_vocab.py, 2026-10-01): the URL for a spoken site
    name ("accelo"), read live from core.config; None when unset or no match."""
    try:
        from core import config as _cfg
        from core import stt_vocab as _sv
        return _sv.site_shortcut(name, getattr(_cfg, "SITE_SHORTCUTS", None))
    except Exception:
        return None


def _act_open_url(url: str) -> str:
    url = _site_shortcut_url(url) or url
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
    tabs = []
    skipped = []
    browser_only = []
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
        try:
            w.close()
            closed.append(w.title)
        except Exception:
            pass
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
    extractor below."""
    bc = _bc()
    bc._ambient_mode_active[0] = bool(active)
    bc._write_hud_state(ambient_mode_active=bool(bc._ambient_mode_active[0]))
    _on = bool(bc._ambient_mode_active[0])
    bc.AMBIENT_LISTEN_ENABLED = _on
    try:
        import core.config as _cfg
        _cfg.AMBIENT_LISTEN_ENABLED = _on
    except Exception:
        pass
    _staging = getattr(bc, "_is_staging", lambda: False)
    caveat = ""
    if not _staging():
        try:
            from tools import settings_window as sw
            sw.update_settings({"AMBIENT_LISTEN_ENABLED": _on})
        except Exception:
            caveat = " (though I couldn't save that for next boot)"
    action_name = "ambient_listen_start" if bc._ambient_mode_active[0] else "ambient_listen_stop"
    fn = bc.ACTIONS.get(action_name)
    if fn is not None:
        try:
            fn("")
        except Exception as e:
            return f"ambient daemon refused: {e}"
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
    process's conversation history, and the LIVE system prompt."""
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
    that yields to the VERSION file's mtime when the mtime is newer — see the
    inline comments below."""
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
    summary), and the LIVE system prompt is rebuilt at once."""
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
            # then labelled "claude/claude-sonnet-5". A benchmark that misnames
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


def _window_key(w):
    """A stable identity for a pygetwindow window: its native handle (titles
    change under us), or the object itself where there is none (held in the
    set, so its identity can't be recycled the way a bare id() can)."""
    hwnd = getattr(w, "_hWnd", None)
    return hwnd if hwnd is not None else w


def _act_open_on_monitor(args: str) -> str:
    """args format: '<monitor_name> | <url-or-app-name>' (or the comma form,
    see _split_monitor_args). Opens the URL or launches the app, then moves the
    resulting window to the named monitor and maximizes it."""
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
    hwnds_before = {_window_key(w) for w in gw.getAllWindows()}

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
    fallback = None   # a FRESH window that doesn't (yet) match the target
    reused = None     # a PRE-EXISTING window that matches the target
    started = time.time()
    deadline = started + 15.0
    while time.time() < deadline:
        time.sleep(0.2)
        fresh = []
        for w in gw.getAllWindows():
            if not w.title:
                continue
            if _window_key(w) in hwnds_before:
                if reused is None and _matches_target(w.title):
                    reused = w
                continue
            try:
                if w.width < 200 or w.height < 200:
                    continue   # ignore tiny splash/tooltip windows
            except Exception:
                pass
            fresh.append(w)
        matched = [w for w in fresh if _matches_target(w.title)]
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
        # UX regression (2026-10-01, actions-a review). Stop early and offer
        # the move instead of making it: that window may be the owner's
        # stream (B092).
        if (fallback is None and reused is not None
                and time.time() - started >= _OPEN_ON_MONITOR_REUSE_S):
            break
    if new_window is None:
        new_window = fallback   # still a window from AFTER the launch, never before

    if not new_window and reused is not None:
        return (f"launched {target}, but it reused your existing "
                f"'{reused.title}' window rather than opening a new one, so "
                f"I didn't move it — ask me to move '{reused.title}' to the "
                f"{monitor_name} monitor if you want it there")
    if not new_window:
        return (f"launched {target}, but couldn't find new window to move it "
                f"— if it reused a window that was already open, ask me to "
                f"move that window to the {monitor_name} monitor")

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
