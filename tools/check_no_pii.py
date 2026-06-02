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


def _load_local_patterns() -> None:
    """Extend HARD/WARN from a gitignored tools/pii_local.py if present.

    Keeps owner-identifying literals (name, email, device codes, contacts)
    OUT of this shipped, committed scanner. The local file is gitignored and
    skipped from scanning (see _SELF); absent on a fresh clone and in CI,
    where there is no owner PII to match anyway.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pii_local.py")
    if not os.path.exists(path):
        return
    ns: dict = {}
    try:
        with open(path, encoding="utf-8") as fh:
            exec(compile(fh.read(), path, "exec"), ns)
    except Exception:
        return
    for lbl, pat in ns.get("HARD", []):
        HARD.append((lbl, _rx(pat)))
    for lbl, pat in ns.get("WARN", []):
        WARN.append((lbl, _rx(pat)))


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
