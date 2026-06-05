"""
YouTube search-to-direct-URL resolver (skills/youtube_search.py).

Uses yt-dlp under the hood to resolve a search query straight to a
`https://www.youtube.com/watch?v=...` URL without scraping a Google SERP
or having JARVIS vision-click the first result.

Why this exists:
  The legacy `_act_youtube` action opens the YouTube search-results page and
  relies on vision to figure out where the user wanted to go. The
  vision-based `youtube_play` pipeline does the same thing more aggressively
  (it actually clicks the first result and presses play). Both fail badly
  on flaky vision or unusual layouts and tend to fall into "see_screen →
  guess" loops. This skill short-circuits the lookup by asking yt-dlp for
  the canonical video URL, then opens it directly in the default browser.

Actions added:
  youtube_search_direct, <query>
      Resolve <query> to the first matching YouTube watch URL and open it
      in the default browser. Returns either the URL string or a friendly
      install/troubleshooting hint.

Module-level helper:
  find_direct_url(query) -> str | None
      Same resolution logic without opening a browser. Used by
      bobert_companion._extract_youtube_url_from_search() so the
      web_search video-intent shortcut benefits from the same fast path.
      Returns None on any failure so callers can fall back cleanly.

yt-dlp lookup:
  We prefer the `yt-dlp` executable on PATH, then fall back to
  `python -m yt_dlp`. Either form accepts the `ytsearch1:<query>` URL
  scheme to mean "first YouTube search hit" and `--print` to emit a
  single line containing only the resolved URL. No video download.

Optional dep:
  yt-dlp must be installed (pip install yt-dlp) OR a `yt-dlp` binary must
  be on PATH. The skill registers cleanly without it; the action returns a
  one-line install hint and the helper returns None.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import webbrowser

# Single-call budget. yt-dlp's search backend usually answers in under 3 s
# on a warm connection; 10 s is generous but still short enough that a
# failing lookup doesn't strand the voice loop.
_YTDLP_TIMEOUT_S = 10.0

# Cache the resolved invocation so we don't re-probe PATH / `python -m
# yt_dlp` on every call. None means "not probed yet"; an empty list means
# "probed and not available".
_YTDLP_CMD: list[str] | None = None


def _probe_ytdlp() -> list[str]:
    """Return the argv prefix to invoke yt-dlp, or [] if it can't be found.

    Tries the `yt-dlp` binary on PATH first (fastest startup), then falls
    back to `python -m yt_dlp` so a pure `pip install yt-dlp` works even
    when the user hasn't restarted their shell to refresh PATH.
    """
    global _YTDLP_CMD
    if _YTDLP_CMD is not None:
        return _YTDLP_CMD

    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if exe:
        _YTDLP_CMD = [exe]
        return _YTDLP_CMD

    # Fall back to the Python module form. Confirm the import is available
    # before claiming success — otherwise the subprocess call would emit a
    # ModuleNotFoundError every time and we'd never reach the friendlier
    # install hint.
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import yt_dlp"],
            capture_output=True,
            timeout=5.0,
        )
        if probe.returncode == 0:
            _YTDLP_CMD = [sys.executable, "-m", "yt_dlp"]
            return _YTDLP_CMD
    except Exception:
        pass

    _YTDLP_CMD = []
    return _YTDLP_CMD


def _missing_dep_hint() -> str:
    return ("yt-dlp is not installed, sir — pip install yt-dlp (or put the "
            "yt-dlp binary on PATH) to enable direct YouTube search.")


def find_direct_url(query: str) -> str | None:
    """Resolve `query` to a YouTube watch URL via yt-dlp.

    Returns the watch URL string on success, or None if yt-dlp isn't
    available, the search returned no hits, or the subprocess errored /
    timed out. Never raises — callers chain this with a SERP fallback.
    """
    query = (query or "").strip()
    if not query:
        return None

    cmd = _probe_ytdlp()
    if not cmd:
        return None

    # ytsearch1:<query> → first matching video.
    # --print emits one line per item; the format string yields only the
    #     watch URL so we don't have to parse JSON.
    # --skip-download / -s keep the subprocess from grabbing any media.
    # --no-warnings keeps stderr clean enough that we can log it on failure.
    argv = [
        *cmd,
        "--print", "https://www.youtube.com/watch?v=%(id)s",
        "--no-playlist",
        "--skip-download",
        "--no-warnings",
        "--quiet",
        f"ytsearch1:{query}",
    ]

    try:
        # Hide a console window on Windows so headless boots don't flash a
        # black flicker for every lookup.
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_YTDLP_TIMEOUT_S,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("https://www.youtube.com/watch?v="):
            return line
    return None


def _close_prior_youtube_windows() -> None:
    """Best-effort: close any existing YouTube browser window so a new video
    REUSES one tab instead of piling up (the user wants a single video tab).
    No-op without pygetwindow, and never raises."""
    try:
        import pygetwindow as gw
        wins = gw.getAllWindows()
    except Exception:
        return
    for w in wins:
        t = (getattr(w, "title", "") or "").lower()
        if "youtube" in t and any(b in t for b in
                                  ("chrome", "edge", "firefox", "brave", "opera")):
            try:
                w.close()
            except Exception:
                pass


def youtube_search_direct(query: str) -> str:
    """Action: resolve `query` to a YouTube watch URL and open it.

    Falls back to the YouTube results page (no auto-click) when yt-dlp
    isn't installed or returns no hits — that path is still better than
    nothing because the user can pick a result manually instead of being
    stuck in a vision retry loop.
    """
    query = (query or "").strip()
    if not query:
        return "no query given, sir"

    # Reuse one video tab: close any prior YouTube window before opening this
    # one so videos don't pile up (mirrors the music single-tab behaviour).
    _close_prior_youtube_windows()

    cmd = _probe_ytdlp()
    if not cmd:
        # Open the search results page as a graceful fallback so the user
        # at least lands somewhere useful.
        import urllib.parse
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        return (
            f"{_missing_dep_hint()} Opened the YouTube results page for "
            f"'{query}' instead."
        )

    url = find_direct_url(query)
    if not url:
        # Fall back to the search-results page so the user isn't stranded.
        import urllib.parse
        results = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
        try:
            webbrowser.open(results)
        except Exception:
            pass
        return (
            f"yt-dlp couldn't find a direct match for '{query}', sir — "
            f"opened the YouTube results page instead."
        )

    try:
        webbrowser.open(url)
    except Exception as e:
        return f"resolved {url} but couldn't open the browser: {e}"

    # Brief settle so any follow-up see_screen captures the player and not
    # a blank tab. Mirrors the timing used by _act_web_search / _act_youtube.
    time.sleep(3.0)
    return (
        f"opened {url} (resolved via yt-dlp for '{query}') — video is now "
        f"playing, no further action needed"
    )


def register(actions: dict) -> None:
    """Register the action with bobert_companion's dispatcher."""
    actions["youtube_search_direct"] = youtube_search_direct
    # Common phrasing aliases so the LLM has a few names to land on.
    actions["youtube_direct"]        = youtube_search_direct
    actions["yt_direct"]             = youtube_search_direct
