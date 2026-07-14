#!/usr/bin/env python3
"""PII / secret leak gate for the JARVIS repo.

Scans the files that git would actually track (``git ls-files``) — or every
text file under a directory you pass — for owner-identifying personal data and
secrets that must NEVER be committed or shipped. This is the hard gate run
before the baseline commit, in CI, and inside ``tools/build_release.py``.

    python tools/check_no_pii.py            # scan git-tracked / staged files
    python tools/check_no_pii.py <dir>      # scan every text file under <dir>
    python tools/check_no_pii.py --strict   # promote WARN findings to failures

Exit code 0 = clean (no HARD findings), 1 = leak(s) found.

Zero external deps (stdlib only) so it runs in the App-Control-locked
environment and on a bare CI runner. Add new patterns as the owner's personal
data surface grows — keep this file itself free of real secrets (the patterns
below are deliberately partial / well-known formats, not live values).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Extensions we never try to read as text (binary / large assets).
_SKIP_EXT = {
    ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp",
    ".stl", ".3mf", ".obj", ".zip", ".gz", ".7z", ".pdf", ".xlsx", ".docx",
    ".wav", ".mp3", ".flac", ".ogg", ".npy", ".npz", ".bin", ".pt", ".onnx",
    ".woff", ".woff2", ".ttf", ".mp4", ".mov",
}

# This scanner itself names the patterns it hunts for, so exclude it (and the
# plan/notes) from being self-flagged.
_SELF = {
    os.path.normcase(os.path.abspath(__file__)),
    # gitignored owner-pattern file (loaded below) — never scan it for PII,
    # it holds the real values on purpose.
    os.path.normcase(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pii_local.py")),
}

# (label, compiled regex). HARD findings fail the gate; WARN are reported only
# (unless --strict). Keep HARD reserved for genuinely identifying data.
def _rx(p: str) -> "re.Pattern[str]":
    return re.compile(p)


HARD = [
    # Generic secret/key FORMATS only. Owner-specific identifiers (real name,
    # email, LAN subnet, device codes) load from the gitignored
    # tools/pii_local.py below, so this shipped scanner contains no real
    # personal values. A fresh clone has no such data to catch; the owner's
    # working copy keeps full detection via that local file.
    ("anthropic-key",     _rx(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    ("openai-key",        _rx(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("aws-access-key",    _rx(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google-api-key",    _rx(r"\bAIza[0-9A-Za-z_\-]{32,}\b")),
    ("private-key-block", _rx(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
]

WARN = [
    # Generic secret-shaped assignments are advisory only — as HARD they
    # false-positive on test fixtures (fake creds) and the .env.example
    # template. Run `check_no_pii.py --strict` (e.g. in the release build) to
    # block on these. Owner-specific advisory tokens load from pii_local.py.
    ("secret-literal",    _rx(r"(?i)(password|passwd|access[_-]?code|secret|api[_-]?key|"
                              r"auth[_-]?token)\s*[:=]\s*[\"'][^\"']{6,}[\"']")),
    ("private-ip",        _rx(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b")),
]


def _local_pattern_candidates() -> list[str]:
    """Ordered, de-duped (by realpath) paths to probe for pii_local.py.

    The module-relative sibling is gitignored, so in a `git worktree add`
    checkout or a CI runner it is absent. Without a fallback the owner HARD/WARN
    patterns would silently never register and the gate would degrade to generic
    key formats — a no-op precisely where it matters. So after the local probe
    we fall back to the canonical owner checkout and an explicit override env var.
    """
    raw = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pii_local.py"),
        r"C:/JARVIS/tools/pii_local.py",
        os.environ.get("JARVIS_PII_LOCAL", ""),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for p in raw:
        if not p:
            continue
        try:
            key = os.path.normcase(os.path.realpath(p))
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# Set by _load_local_patterns() when a pii_local.py EXISTS but fails to load —
# the "degraded scanner" case. main() turns a non-None value into exit 2 so the
# git gate fails closed, WITHOUT importing this module ever exiting the process.
_LOAD_ERROR: str | None = None


def _load_local_patterns() -> None:
    """Extend HARD/WARN from a gitignored tools/pii_local.py if present.

    Keeps owner-identifying literals (name, email, device codes, contacts)
    OUT of this shipped, committed scanner. The local file is gitignored and
    skipped from scanning (see _SELF); absent on a fresh clone and in CI,
    where there is no owner PII to match anyway. We try several canonical
    locations (see _local_pattern_candidates) so a worktree/CI checkout that
    lacks the sibling still gets full owner coverage when the owner file is
    reachable; if none is found we say so on stderr rather than passing
    silently.
    """
    for path in _local_pattern_candidates():
        if not os.path.exists(path):
            continue
        ns: dict = {}
        try:
            with open(path, encoding="utf-8") as fh:
                exec(compile(fh.read(), path, "exec"), ns)
        except Exception as e:
            # FAIL CLOSED — but NOT by exiting at IMPORT time (2026-07-14 audit).
            # A pii_local.py that EXISTS but won't parse/exec is more dangerous
            # than "no file present": the owner-PII patterns did NOT load, so the
            # scanner has silently degraded to generic key formats exactly when
            # it is meant to catch owner literals — and this repo syncs through
            # Nextcloud, so a leaked commit is effectively irreversible. The gate
            # must still refuse. But this module is ALSO imported by
            # core/bug_reporter (to scrub by the same rules), and a bare
            # sys.exit(2) at module scope would terminate the interpreter of
            # whatever imported it — bug_reporter's whole contract is "never
            # crash the host". So RECORD the degraded state and let main() (the
            # gate/hook entry) turn it into exit 2; importers stay alive with the
            # generic patterns they managed to load.
            global _LOAD_ERROR
            _LOAD_ERROR = (
                f"{path} is present but failed to load "
                f"({e.__class__.__name__}: {e}). Owner-PII patterns did not "
                f"load; refusing to run with a degraded scanner. Fix or remove "
                f"the file.")
            sys.stderr.write(f"[check_no_pii] ERROR: {_LOAD_ERROR}\n")
            return
        for lbl, pat in ns.get("HARD", []):
            HARD.append((lbl, _rx(pat)))
        for lbl, pat in ns.get("WARN", []):
            WARN.append((lbl, _rx(pat)))
        return
    sys.stderr.write(
        "[check_no_pii] note: no pii_local.py found (owner-PII patterns "
        "unavailable; running with generic key formats only)\n"
    )


_load_local_patterns()


def _tracked_files() -> list[str] | None:
    """Files git would commit (staged + tracked). None if not a git repo."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    files = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    return [os.path.join(_PROJECT_ROOT, f) for f in files]


def _walk_text_files(root: str) -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune obvious non-source dirs
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", "backups", "_backups",
                                    "node_modules", "data_staging", "logs_staging"}]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in _SKIP_EXT:
                continue
            found.append(os.path.join(dirpath, fn))
    return found


def _redact(s: str) -> str:
    # Sanitise to ASCII so printing never crashes on a cp1252 console / pipe.
    s = s.strip().encode("ascii", "replace").decode("ascii")
    return (s[:120] + " ...") if len(s) > 120 else s


def _scan_file(path: str, rules: list) -> list[tuple]:
    # Skip this scanner's own source wherever it lives — its pattern strings
    # would self-match (e.g. when scanning a release-build copy in dist/).
    if (os.path.normcase(os.path.abspath(path)) in _SELF
            or os.path.basename(path) == "check_no_pii.py"):
        return []
    if os.path.splitext(path)[1].lower() in _SKIP_EXT:
        return []
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return []
    if b"\x00" in raw[:4096]:  # binary
        return []
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - decode(errors="replace") never raises
        return []
    hits: list[tuple] = []
    for i, line in enumerate(text.splitlines(), 1):
        for label, rx in rules:
            if rx.search(line):
                hits.append((label, i, _redact(line)))
    return hits


def main(argv: list[str]) -> int:
    # Honor the fail-closed degraded-scanner state HERE (the gate/hook entry),
    # not at import. Returning 2 → the __main__ wrapper exits 2, so the git hook
    # still blocks a commit when owner patterns couldn't load — while `import
    # tools.check_no_pii` from bug_reporter never terminates its host. 2026-07-14.
    if _LOAD_ERROR is not None:
        sys.stderr.write(f"[check_no_pii] refusing to scan: {_LOAD_ERROR}\n")
        return 2

    strict = "--strict" in argv
    positional = [a for a in argv if not a.startswith("-")]

    if positional:
        root = os.path.abspath(positional[0])
        files = _walk_text_files(root)
        scope = f"directory {root}"
    else:
        files = _tracked_files()
        if files is None:
            files = _walk_text_files(os.path.join(_PROJECT_ROOT, "core")) \
                + _walk_text_files(os.path.join(_PROJECT_ROOT, "skills")) \
                + _walk_text_files(os.path.join(_PROJECT_ROOT, "tools"))
            scope = "core/+skills/+tools/ (no git repo found)"
        else:
            scope = "git-tracked files"

    hard_hits: list[tuple] = []
    warn_hits: list[tuple] = []
    for path in files:
        rel = os.path.relpath(path, _PROJECT_ROOT)
        for label, ln, snip in _scan_file(path, HARD):
            hard_hits.append((rel, ln, label, snip))
        for label, ln, snip in _scan_file(path, WARN):
            warn_hits.append((rel, ln, label, snip))

    print(f"[check_no_pii] scanned {len(files)} files ({scope})")
    if warn_hits:
        print(f"\n  WARN ({len(warn_hits)}) — review before public release:")
        for rel, ln, label, snip in warn_hits[:200]:
            print(f"    ~ {rel}:{ln}  [{label}]  {snip}")
    if hard_hits:
        print(f"\n  HARD ({len(hard_hits)}) — MUST be removed before commit/ship:")
        for rel, ln, label, snip in hard_hits[:200]:
            print(f"    ! {rel}:{ln}  [{label}]  {snip}")

    failed = bool(hard_hits) or (strict and bool(warn_hits))
    if failed:
        print(f"\n[check_no_pii] FAIL — {len(hard_hits)} hard"
              f"{' + ' + str(len(warn_hits)) + ' warn (strict)' if strict and warn_hits else ''}"
              " finding(s).")
        return 1
    print(f"[check_no_pii] OK — no hard findings"
          f"{' (' + str(len(warn_hits)) + ' warn)' if warn_hits else ''}.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main(sys.argv[1:]))
