"""Single source of truth for the JARVIS release version.

The public release/distribution version lives in the top-level ``VERSION``
file and is read once here. This is intentionally SEPARATE from the
self-upgrade pipeline's internal CHANGELOG counter (which bumps a patch
number every pipeline run); ``__version__`` is the version JARVIS reports as
its shareable build (e.g. "2.0.29").

Zero side effects at import beyond a single small file read, so this stays in
the import-light tier and is safe to import anywhere (incl. bare CI).
"""
from __future__ import annotations

import os

_VERSION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION"
)

_FALLBACK = "0.0.0-dev"


def _read_version() -> str:
    try:
        with open(_VERSION_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip() or _FALLBACK
    except OSError:
        return _FALLBACK


__version__ = _read_version()
VERSION = __version__


def version_string() -> str:
    """Human-facing release string, e.g. ``2.0.29``."""
    return __version__


def _git_out(root: str, *args: str) -> "str | None":
    """stdout of ``git -C root <args>`` ('' when git answers with an error),
    or None when git could not be asked at all (missing, timed out). Never
    raises."""
    import subprocess
    import sys
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        r = subprocess.run(["git", "-C", root, *args], capture_output=True,
                           text=True, timeout=3, creationflags=flags)
    except Exception:
        return None
    return (r.stdout or "").strip() if r.returncode == 0 else ""


# ONE ANSWER PER RELEASE ON DISK (2026-10-02 review). "What version are you"
# runs on the action path, and an uncached answer is 2-3 git subprocesses
# (~50-300 ms, up to 3 s apiece when git stalls). Keyed by the directory and
# the VERSION file's mtime and size, so a pull or checkout that changes the
# release is read afresh. An answer git did not give (timed out, no git) is
# never kept: the next ask tries git again.
_release_ts_cache: dict = {}


def release_timestamp(project_dir: str | None = None) -> float | None:
    """Epoch seconds of the release on disk in ``project_dir`` (default: this
    checkout), or None. Call-time only: importing this module stays one read.

    A release is a git commit that bumps VERSION, tagged ``v<VERSION>``. It
    never writes data/version.json: that file is the self-upgrade pipeline's
    own counter and timestamp, and it sat at 1.0.17 / 2026-05-30 for four
    months while the releases moved on to 2.0.x (2026-10-02). So the date
    comes from git: the tag's commit date, else the date of the last commit
    that changed VERSION. A tree that is not its own git checkout (a copy, a
    zip, a temp dir inside some other repo) uses the VERSION file's mtime."""
    root = os.path.abspath(project_dir or os.path.dirname(_VERSION_FILE))
    vfile = os.path.join(root, "VERSION")
    try:
        st = os.stat(vfile)
        key = (os.path.normcase(root), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _release_ts_cache:
        return _release_ts_cache[key]
    ts, git_answered = _release_timestamp_uncached(root, vfile)
    if key is not None and git_answered:
        _release_ts_cache[key] = ts
    return ts


def _release_timestamp_uncached(root: str, vfile: str):
    """(release_timestamp's answer, whether git was able to give it)."""
    top = _git_out(root, "rev-parse", "--show-toplevel")
    asked = top is not None
    if top and (os.path.normcase(os.path.realpath(top))
                == os.path.normcase(os.path.realpath(root))):
        try:
            with open(vfile, "r", encoding="utf-8") as fh:
                ver = fh.read().strip()
        except OSError:
            ver = ""
        queries = []
        if ver and all(c.isalnum() or c in ".-+_" for c in ver):
            queries.append(("log", "-1", "--format=%ct", f"refs/tags/v{ver}", "--"))
        queries.append(("log", "-1", "--format=%ct", "--", "VERSION"))
        for q in queries:
            out = _git_out(root, *q)
            if out is None:
                asked = False
                continue
            try:
                return float(out.splitlines()[0]), asked
            except (IndexError, ValueError):
                continue
    try:
        return os.path.getmtime(vfile), asked
    except OSError:
        return None, asked
