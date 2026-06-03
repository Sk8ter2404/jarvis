"""Update checker — compare the running build against the latest GitHub release.

WHY THIS MODULE EXISTS
----------------------
JARVIS already reports its own release version (``core/version.py`` reads the
top-level ``VERSION`` file). This module answers the next question — "is there a
NEWER published version?" — so a running instance can tell its user. It is the
shared core behind three surfaces:

  * the boot-time "a new version is available" nudge,
  * the spoken "check for updates" action,
  * the update wizard (``tools/update_wizard.py``), which acts on the result.

CI-SAFETY CONTRACT
------------------
* Import-light: stdlib only (+ ``core.version``). The network call uses urllib
  (stdlib) and is performed inside ``fetch_latest_release`` so importing this
  module never touches the network.
* Every public function is TOTAL — it returns a structured value for any input
  and never propagates an exception — so the boot path can call
  ``check_for_update()`` on a background thread with no guard.
* No token / no network / private-repo 401-403 → the check degrades to
  ``checked=False`` with a human-readable reason. Never an error, never a crash,
  never a nag the user can't action.

PRIVATE-REPO NOTE
-----------------
The repo is private, so the GitHub Releases API needs a token. The checker reads
``JARVIS_GITHUB_TOKEN`` (preferred) or ``GITHUB_TOKEN``; with neither it reports
"can't check (no token)" and stops. Owner + collaborators set the token once.
Repo coordinates are env-overridable (``JARVIS_GITHUB_OWNER`` /
``JARVIS_GITHUB_REPO``) so a fork can point the check at its own repo.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, Tuple

from core.version import __version__ as LOCAL_VERSION

_DEFAULT_OWNER = "Sk8ter2404"
_DEFAULT_REPO = "jarvis"

# A small, permissive semver parser: MAJOR.MINOR.PATCH with an optional
# -prerelease tag (e.g. 1.0.0-beta.1). A leading 'v' is tolerated. Build
# metadata (+...) is intentionally ignored.
_SEMVER_RE = re.compile(r"^\s*v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?\s*$")


def _owner_repo() -> Tuple[str, str]:
    """Repo coordinates, env-overridable for forks."""
    owner = (os.environ.get("JARVIS_GITHUB_OWNER") or _DEFAULT_OWNER).strip()
    repo = (os.environ.get("JARVIS_GITHUB_REPO") or _DEFAULT_REPO).strip()
    return owner or _DEFAULT_OWNER, repo or _DEFAULT_REPO


def _token() -> Optional[str]:
    """The GitHub token for the API call, or None. ``JARVIS_GITHUB_TOKEN`` wins
    over the conventional ``GITHUB_TOKEN``; blank/whitespace counts as absent."""
    for name in ("JARVIS_GITHUB_TOKEN", "GITHUB_TOKEN"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return None


def parse_version(text) -> Optional[Tuple[int, int, int, Optional[str]]]:
    """Parse 'v1.2.3' / '1.2.3-beta.1' → (major, minor, patch, prerelease|None).

    Returns None for anything unparseable (non-string, empty, 'latest', …).
    Never raises."""
    if not isinstance(text, str):
        return None
    m = _SEMVER_RE.match(text)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4))


def _pre_key(pre: Optional[str]):
    """Ordering key for the prerelease field. A release (``None``) outranks any
    prerelease (1.0.0 > 1.0.0-beta), and prereleases compare by dot-separated
    identifiers: numeric identifiers compare as ints and rank below
    alphanumeric ones, matching the semver spec closely enough for beta.N
    tags."""
    if pre is None:
        return (1,)  # release sorts AFTER any prerelease
    parts = []
    for ident in pre.split("."):
        if ident.isdigit():
            parts.append((0, int(ident), ""))
        else:
            parts.append((1, 0, ident))
    return (0, tuple(parts))


def compare_versions(a, b) -> Optional[int]:
    """-1 if a<b, 0 if a==b, 1 if a>b. Returns None when EITHER side is
    unparseable so the caller can treat the comparison as 'unknown' rather than
    guessing. Prerelease sorts below the matching release."""
    pa, pb = parse_version(a), parse_version(b)
    if pa is None or pb is None:
        return None
    ka = (pa[0], pa[1], pa[2], _pre_key(pa[3]))
    kb = (pb[0], pb[1], pb[2], _pre_key(pb[3]))
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


def _http_get_json(url: str, headers: dict, timeout: float) -> dict:
    """GET ``url`` and parse the JSON body. Isolated as its own function so
    tests patch THIS (rather than urllib internals). Raises on any
    network/HTTP/decode error — callers wrap it."""
    import urllib.request

    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def fetch_latest_release(timeout: float = 6.0) -> Optional[dict]:
    """Return the latest GitHub release JSON dict, or None on ANY failure (no
    token, network down, 401/404, malformed body). Never raises."""
    token = _token()
    if not token:
        return None
    owner, repo = _owner_repo()
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": f"jarvis-update-checker/{LOCAL_VERSION}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        data = _http_get_json(url, headers, timeout)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def check_for_update(timeout: float = 6.0, current: Optional[str] = None) -> dict:
    """The single entry point. Compare the local build to the latest release.

    Returns a dict (never raises):
      ``current``          – the running version string
      ``latest``           – latest release tag (str) or None if not checked
      ``update_available`` – True iff ``latest`` is newer than ``current``
      ``release_url``      – HTML URL of the latest release (or None)
      ``release_name``     – the release's name/title (or None)
      ``published_at``     – ISO timestamp of the release (or None)
      ``checked``          – True iff we reached the API and parsed a version
      ``detail``           – short human-readable status / reason
    """
    cur = current or LOCAL_VERSION
    result = {
        "current": cur, "latest": None, "update_available": False,
        "release_url": None, "release_name": None, "published_at": None,
        "checked": False, "detail": "",
    }
    if not _token():
        result["detail"] = (
            "no GitHub token set (JARVIS_GITHUB_TOKEN / GITHUB_TOKEN); "
            "update check skipped")
        return result
    rel = fetch_latest_release(timeout=timeout)
    if not rel:
        result["detail"] = "couldn't reach the GitHub releases API"
        return result
    tag = rel.get("tag_name") or rel.get("name") or ""
    result["latest"] = tag or None
    result["release_url"] = rel.get("html_url")
    result["release_name"] = rel.get("name") or tag or None
    result["published_at"] = rel.get("published_at")
    cmp = compare_versions(cur, tag)
    if cmp is None:
        result["detail"] = f"couldn't compare versions ({cur!r} vs {tag!r})"
        return result
    result["checked"] = True
    if cmp < 0:
        result["update_available"] = True
        result["detail"] = f"update available: {cur} -> {tag}"
    else:
        result["detail"] = f"up to date ({cur})"
    return result


def update_message(result: dict) -> str:
    """Render a one-line, speakable status from a ``check_for_update`` result.
    Total — accepts any dict shaped like the checker's output."""
    if result.get("update_available"):
        latest = result.get("latest") or "a newer version"
        return (f"A new version of me is available, sir — {latest} "
                f"(you're on {result.get('current')}).")
    if result.get("checked"):
        return f"I'm up to date, sir — running {result.get('current')}."
    return (f"I couldn't check for updates, sir — {result.get('detail') or 'unknown reason'}. "
            f"Running {result.get('current')}.")


# ──────────────────────────────────────────────────────────────────────
#  Throttled cache + boot-time nudge
#  The boot check must not hammer the GitHub API on every restart, so the
#  result is cached in data/update_check.json and reused within a TTL.
# ──────────────────────────────────────────────────────────────────────

def default_cache_path() -> str:
    """data/update_check.json beside the repo (gitignored runtime state)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data", "update_check.json")


def read_cache(path: str) -> Optional[dict]:
    """Return the cached result dict, or None if missing/unreadable/not a dict.
    Never raises."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_cache(path: str, payload: dict) -> bool:
    """Atomically write `payload` as JSON. Returns False on any failure (read-
    only FS, perms) instead of raising — the check is best-effort."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def cached_check(ttl_hours: float = 24.0, timeout: float = 6.0,
                 path: Optional[str] = None,
                 now: Optional[float] = None) -> dict:
    """``check_for_update`` with a throttle. Reuses a cached result younger than
    ``ttl_hours``; otherwise re-checks and rewrites the cache. The returned dict
    is the check result plus ``checked_at`` (epoch) and ``cached`` (bool).
    ``now`` is injectable for deterministic tests. Total — never raises."""
    p = path or default_cache_path()
    t = now if now is not None else time.time()
    cached = read_cache(p)
    if cached and isinstance(cached.get("checked_at"), (int, float)):
        age_h = (t - cached["checked_at"]) / 3600.0
        if 0 <= age_h < ttl_hours:
            fresh = dict(cached)
            fresh["cached"] = True
            return fresh
    res = check_for_update(timeout=timeout)
    res["checked_at"] = t
    res["cached"] = False
    write_cache(p, res)
    return res


def boot_nudge(announce, ttl_hours: float = 24.0, enabled: bool = True,
               path: Optional[str] = None,
               now: Optional[float] = None) -> Optional[dict]:
    """Throttled boot check that speaks ONE nudge when a newer release exists.

    ``announce`` is a ``callable(str) -> Any`` — the monolith passes a wrapper
    around ``proactive_announce`` so the message lands in the pending-speech
    queue and the main loop speaks it at the next turn boundary. Total: any
    error (including a raising ``announce``) is swallowed so the boot path is
    never affected. Returns the check result, or None when disabled/errored."""
    if not enabled:
        return None
    try:
        res = cached_check(ttl_hours=ttl_hours, path=path, now=now)
    except Exception:
        return None
    if res.get("update_available"):
        try:
            announce(update_message(res))
        except Exception:
            pass
    return res


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import pprint
    res = check_for_update()
    pprint.pprint(res)
    print(update_message(res))
