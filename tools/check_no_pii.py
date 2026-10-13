#!/usr/bin/env python3
"""PII / secret leak gate for the JARVIS repo.

Scans the files that git would actually track (``git ls-files``) — or every
text file under a directory you pass — for owner-identifying personal data and
secrets that must NEVER be committed or shipped. This is the hard gate run
before the baseline commit, in CI, and inside ``tools/build_release.py``.

    python tools/check_no_pii.py            # scan git-tracked / staged files
    python tools/check_no_pii.py <dir>      # scan every text file under <dir>
    python tools/check_no_pii.py --strict   # promote WARN findings to failures
    python tools/check_no_pii.py --metadata [RANGE]
                                            # scan commit author/committer
                                            # identities + messages (default
                                            # range: upstream..HEAD)
    python tools/check_no_pii.py --commit-msg FILE
                                            # scan a commit message + the
                                            # identity git would commit as
                                            # (the commit-msg hook)

Exit code 0 = clean (no HARD findings), 1 = leak(s) found, 2 = could not scan
safely.

Zero external deps (stdlib only) so it runs in the App-Control-locked
environment and on a bare CI runner. Add new patterns as the owner's personal
data surface grows — keep this file itself free of real secrets (the patterns
below are deliberately partial / well-known formats, not live values).

SINGLE SOURCE OF TRUTH — and the DIRECTION matters. ``core/bug_reporter.py``
IMPORTS ``HARD``/``WARN`` from this module (its ``_gate_patterns()``) and applies
them on top of its own built-in scrub rules, so a credential shape added HERE
reaches the runtime scrubber for free. The reverse is NOT true — shapes added
only to ``bug_reporter._SCRUB_RULES`` never flow back, and that asymmetry is
exactly how this gate silently fell behind the scrubber on GitHub tokens
(found 2026-08-20). New credential shapes therefore belong in ``HARD`` below
FIRST. ``tests/test_gate_scrubber_parity.py`` fails if the two drift apart
again.
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

# Binary files cannot be read as text but can still carry personal data (a
# desktop screenshot shows the taskbar, open apps, hostnames, the wallpaper).
# They are no longer silently skipped: every tracked file with one of these
# extensions must be on the reviewed allow-list below or the gate fails with a
# "binary-review" finding, so a new image / document / archive is a deliberate,
# looked-at addition. To add one: open it, confirm it shows nothing personal,
# then append its repo-relative path (forward slashes) to _REVIEWED_BINARIES.
_REVIEW_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp", ".pdf",
    ".xlsx", ".docx", ".zip", ".gz", ".7z", ".wav", ".mp3", ".flac", ".ogg",
    ".mp4", ".mov", ".npy", ".npz", ".pt", ".onnx", ".bin", ".stl", ".3mf",
    ".obj",
}
_REVIEWED_BINARIES = frozenset({
    "assets/jarvis_icon.png",
    "assets/jarvis_icon.ico",
    # HUD crops: weather/clock widgets on a blank background, no names.
    "tools/corner.png",
    "tools/hud_check.png",
    "tools/hud_live.png",
    "tools/hud_now.png",
    "tools/hud_wide.png",
    "tools/unified_hud_preview.png",
})

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


class _FilteredRx:
    """A regex plus an allow-list predicate, quacking like ``re.Pattern`` for the
    calls the gate and core/bug_reporter make (``search`` / ``sub`` /
    ``finditer`` / ``match``). A match the predicate allows (an obvious
    placeholder such as an example.com address or an all-zero MAC) is treated as
    no match at all, so the allow-list lives in readable Python instead of a
    lookahead maze, and the runtime bug-report scrubber honours it too."""

    def __init__(self, pattern: str, allow, flags: int = 0) -> None:
        self.rx = re.compile(pattern, flags)
        self.pattern = pattern
        self.flags = self.rx.flags
        self._allow = allow

    def finditer(self, text, *a):
        for m in self.rx.finditer(text, *a):
            if not self._allow(m):
                yield m

    def search(self, text, *a):
        return next(self.finditer(text, *a), None)

    def match(self, text, *a):
        m = self.rx.match(text, *a)
        return None if m is None or self._allow(m) else m

    def sub(self, repl, text, count: int = 0):
        def _one(m):
            if self._allow(m):
                return m.group(0)
            return repl(m) if callable(repl) else m.expand(repl)
        return self.rx.sub(_one, text, count)


# --- allow-lists for the privacy categories ---------------------------------
# Obvious placeholders only. Anything that merely LOOKS like a placeholder but
# could be a real address (a plausible corporate domain, a believable MAC) is
# NOT allowed: rewrite the fixture to a reserved example value instead.
_EMAIL_OK_DOMAINS = frozenset({
    "example.com", "example.org", "example.net", "example.edu",
    "users.noreply.github.com", "noreply.github.com", "github.com",
    "anthropic.com", "localhost",
})
_EMAIL_OK_SUFFIXES = (".example", ".test", ".invalid", ".localhost", ".local",
                      ".example.com", ".example.org", ".example.net")
# One-letter / generic placeholder domains used by tests (a@x.com ...).
_EMAIL_OK_LABELS = frozenset({"x", "y", "z", "a", "b", "c", "e", "t", "foo",
                              "bar", "baz", "domain", "yourdomain", "your-domain",
                              "company", "mydomain", "synth"})


# Role / placeholder local parts. A no-reply or "shipment-tracking" sender at a
# retailer, or "user@..." in an example, names no person.
_EMAIL_OK_LOCALS = frozenset({
    "noreply", "no-reply", "donotreply", "do-not-reply", "shipment-tracking",
    "auto-confirm", "ship", "deals", "order-update", "orders", "info",
    "support", "admin", "sales", "contact", "hello", "billing", "newsletter",
    "notifications", "alerts", "user", "you", "name", "someone", "me", "email",
    "username", "test", "foo", "bar", "alice", "bob", "carol", "yourname",
    "your.name", "first.last", "firstname.lastname",
})


def _email_allowed(m) -> bool:
    local, _, domain = m.group(0).lower().rpartition("@")
    if local in _EMAIL_OK_LOCALS:
        return True
    if domain in _EMAIL_OK_DOMAINS or domain.endswith(_EMAIL_OK_SUFFIXES):
        return True
    labels = domain.split(".")
    if "example" in labels:
        return True
    return len(labels) == 2 and labels[0] in _EMAIL_OK_LABELS


_MAC_PLACEHOLDERS = frozenset({
    "aa:bb:cc:dd:ee:ff", "00:11:22:33:44:55", "11:22:33:44:55:66",
    "12:34:56:78:9a:bc", "de:ad:be:ef:00:01", "de:ad:be:ef:de:ad",
    "00:1a:2b:3c:4d:5e",
})
# Last three octets that are obviously typed by hand for a fixture.
_MAC_PLACEHOLDER_TAILS = frozenset({
    ("aa", "bb", "cc"), ("dd", "ee", "ff"), ("11", "22", "33"),
    ("00", "11", "22"), ("44", "55", "66"), ("3c", "4d", "5e"),
})


def _mac_allowed(m) -> bool:
    octets = re.split(r"[:\-]", m.group(0).lower())
    if len(set(octets)) == 1:                    # 00:00:.. / ff:ff:.. / aa:aa:..
        return True
    if octets[3:5] == ["00", "00"]:              # <OUI>:00:00:xx test convention
        return True
    if ":".join(octets) in _MAC_PLACEHOLDERS:
        return True
    if tuple(octets[3:]) in _MAC_PLACEHOLDER_TAILS:
        return True
    return octets[:2] in (["aa", "bb"], ["00", "11"], ["01", "23"])


_USER_PATH_OK = frozenset({
    "public", "default", "default user", "all users", "user", "users",
    "username", "you", "your", "yourname", "me", "name", "someone", "example",
    "alice", "bob", "carol", "test", "testuser", "tester", "runner",
    "runneradmin", "ubuntu", "root", "pi", "jarvis", "foo", "bar", "x", "u",
    "owner", "admin", "person", "someuser", "some_user", "dev", "developer", "home", "..", ".",
    "your-name", "your_name", "user_name", "yourusername",
})


def _user_path_allowed(m) -> bool:
    name = m.group(1).strip().lower()
    # A one-letter name or a run of dots ("C:\Users\...") is a placeholder.
    return name in _USER_PATH_OK or len(name) <= 1 or not name.strip(".")


def _phone_allowed(m) -> bool:
    digits = re.sub(r"\D", "", m.group(0))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    # 555 exchange = reserved fictional block; 000/123/555 area codes and the
    # 123-456-7890 run are the usual placeholders.
    return (digits[3:6] == "555" or digits[:3] in ("000", "123", "555")
            or digits in ("1234567890", "0123456789"))


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
    # GitHub tokens. This was the gate's one genuinely REACHABLE leak: the
    # TRACKED .env.example ships a `JARVIS_GITHUB_TOKEN=` line the user is told
    # to fill in (core/bug_reporter._issue_token reads exactly that variable and
    # sends it as `Authorization: Bearer`), so filling the TEMPLATE instead of
    # the gitignored .env used to sail past the pre-commit hook, CI, and the
    # release scan with nothing — not even a WARN — firing. A classic token is a
    # fixed prefix + 36 base62 chars; a fine-grained one is `github_pat_` + a
    # 22-char id + `_` + a 59-char secret.
    ("github-token",       _rx(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github-fg-token",    _rx(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    # OpenAI's modern project / service-account keys do NOT match the legacy
    # `sk-` + 32-base62 rule above (they carry `-` and `_`), so they need their
    # own shape. Kept PREFIXED rather than widening "openai-key", which would
    # start matching ordinary hyphenated identifiers that begin "sk-".
    ("openai-project-key", _rx(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{20,}\b")),
    # Slack + JWT. Deliberately TIGHTER than core/bug_reporter._SCRUB_RULES'
    # equivalents, so this repo’s own synthetic fixtures cannot HARD-fail the
    # gate (tests/test_bug_reporter.py carries a literal "xoxb-12345678abcd" and
    # a .../T00/B00/... webhook): a REAL Slack token has three dash-separated
    # groups and a real webhook full T…/B…/24-char path segments. Both still
    # catch every well-formed live credential.
    ("slack-token",        _rx(r"\bxox[bpoas]-\d{9,}-\d{9,}-[0-9A-Za-z]{20,}\b")),
    ("slack-webhook",      _rx(r"https://hooks\.slack\.com/services/"
                               r"T[A-Z0-9]{6,}/B[A-Z0-9]{6,}/[A-Za-z0-9]{20,}")),
    ("jwt",                _rx(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
                               r"\.[A-Za-z0-9_\-]{10,}\b")),
    # --- Personal-data categories (2026-10-12 privacy audit). The shipped gate
    # used to know generic KEY shapes only; an e-mail address, phone number,
    # MAC, overlay-network address or profile path was caught solely by the
    # gitignored pii_local.py, i.e. never in GitHub CI or on a contributor's
    # clone. These are format rules with obvious-placeholder allow-lists
    # (example.com, all-zero MACs, 555 numbers, a "<user>" path ...).
    ("email-address",      _FilteredRx(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b",
        _email_allowed)),
    ("phone-number",       _FilteredRx(
        r"(?<![\w.\-/])(?:\+?1[ .\-]?)?(?:\(\d{3}\)[ .\-]?|\d{3}[ .\-])"
        r"\d{3}[ .\-]\d{4}(?![\w\-])", _phone_allowed)),
    ("mac-address",        _FilteredRx(
        r"(?<![0-9A-Fa-f:\-])[0-9A-Fa-f]{2}([:\-])(?:[0-9A-Fa-f]{2}\1){4}"
        r"[0-9A-Fa-f]{2}(?![0-9A-Fa-f:\-])", _mac_allowed)),
    # Tailscale / CGNAT range 100.64.0.0/10 and MagicDNS names.
    ("tailscale-address",  _rx(r"(?<![\d.])100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])"
                              r"\.\d{1,3}\.\d{1,3}(?![\d.]*\d)")),
    ("tailscale-host",     _rx(r"(?i)\b[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.ts\.net\b")),
    # A real person's profile folder: C:\Users\<name>, /Users/<name>,
    # /home/<name>. Reserved/placeholder names pass (Public, runner, ...).
    ("user-profile-path",  _FilteredRx(
        r"(?i)(?:\b[A-Za-z]:[\\/]+Users|(?<![\w.~])/(?:Users|home))[\\/]+"
        r"([A-Za-z0-9_.\-]{1,40})(?![A-Za-z0-9_.\-])",
        _user_path_allowed)),
]

WARN = [
    # Generic secret-shaped assignments are advisory only — as HARD they
    # false-positive on test fixtures (fake creds) and the .env.example
    # template. Run `check_no_pii.py --strict` (e.g. in the release build) to
    # block on these. Owner-specific advisory tokens load from pii_local.py.
    # The `auth` prefix on `token` is OPTIONAL on purpose: requiring it was the
    # odd one out of three copies of this name list (core/bug_reporter.py and
    # tools/audit_codebase.py both match a bare `token`), and it meant a quoted
    # `GITHUB_TOKEN = "…"` produced no finding at all. 2026-08-20.
    ("secret-literal",    _rx(r"(?i)(password|passwd|access[_-]?code|secret|api[_-]?key|"
                              r"(?:auth[_-]?)?token)\s*[:=]\s*[\"'][^\"']{6,}[\"']")),
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
            # a synced folder, so a leaked commit is effectively irreversible. The gate
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
            ext = os.path.splitext(fn)[1].lower()
            # Binaries are not read as text, but they ARE listed so the
            # binary-review rule (_binary_review_hits) can see them.
            if ext in _SKIP_EXT and ext not in _REVIEW_EXT:
                continue
            found.append(os.path.join(dirpath, fn))
    return found


def _binary_review_hits(files: list[str], root: str) -> list[tuple]:
    """(rel, 0, "binary-review", "") for every file whose extension is a
    review-worthy binary and whose path relative to ``root`` is not on the
    reviewed allow-list."""
    hits: list[tuple] = []
    for path in files:
        if os.path.splitext(path)[1].lower() not in _REVIEW_EXT:
            continue
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        if rel not in _REVIEWED_BINARIES:
            hits.append((rel, 0, "binary-review", ""))
    return hits


# --- commit metadata (author / committer identity + message) -----------------
_ZERO_SHA = "0" * 40


def _git_out(args: list[str]) -> str | None:
    try:
        out = subprocess.run(["git"] + args, cwd=_PROJECT_ROOT,
                             capture_output=True, text=True, timeout=60,
                             encoding="utf-8", errors="replace")
    except Exception:
        return None
    return out.stdout if out.returncode == 0 else None


def _scan_blob(text: str, rules: list) -> list[tuple]:
    """[(label, line_no)] for every rule hit in free text (no snippets)."""
    hits: list[tuple] = []
    for i, line in enumerate(text.splitlines(), 1):
        for label, rx in rules:
            if rx.search(line):
                hits.append((label, i))
    return hits


def _default_rev_args() -> list[str]:
    """Commits not yet on the upstream / origin/main; else just HEAD."""
    for base in ("@{upstream}", "origin/main"):
        if _git_out(["rev-parse", "--verify", "-q", base]) is not None:
            return [f"{base}..HEAD"]
    return ["-n", "1", "HEAD"]


def _commit_metadata_hits(rev_args: list[str]) -> list[tuple] | None:
    """Scan author / committer identity lines and the message body of the
    commits named by ``rev_args`` with every HARD rule. Returns
    [(short_sha, field, label, line_no)], or None if git could not be asked."""
    fmt = "--format=%H%x1f%an <%ae>%x1f%cn <%ce>%x1f%B%x1e"
    out = _git_out(["log", fmt] + rev_args)
    if out is None:
        return None
    sep_f, sep_r = chr(0x1f), chr(0x1e)     # what git printed for %x1f / %x1e
    hits: list[tuple] = []
    for rec in out.split(sep_r):
        rec = rec.strip("\n")
        if not rec:
            continue
        parts = rec.split(sep_f, 3)
        if len(parts) < 4:
            continue
        sha, author, committer, body = parts
        for field, text in (("author", author), ("committer", committer),
                            ("message", body)):
            for label, ln in _scan_blob(text, HARD):
                hits.append((sha[:7], field, label, ln))
    return hits


def _pre_push_rev_args(stdin_text: str) -> list[list[str]]:
    """One rev-arg list per pushed ref from a pre-push hook's stdin lines
    (``<local ref> <local sha> <remote ref> <remote sha>``). A new remote ref
    scans only the commits no remote-tracking ref already has."""
    revs: list[list[str]] = []
    for line in stdin_text.splitlines():
        f = line.split()
        if len(f) < 4 or f[1] == _ZERO_SHA:     # delete push: nothing to scan
            continue
        local, remote = f[1], f[3]
        if remote == _ZERO_SHA:
            revs.append([local, "--not", "--remotes"])
        else:
            revs.append([f"{remote}..{local}"])
    return revs


def _commit_msg_hits(msg_path: str) -> list[tuple] | None:
    """Scan the message in ``msg_path`` (comment lines dropped) and the
    author / committer identity git is about to record."""
    try:
        with open(msg_path, encoding="utf-8", errors="replace") as fh:
            msg = chr(10).join(l for l in fh.read().splitlines()
                               if not l.startswith("#"))
    except OSError:
        return None
    hits = [("message", label, ln) for label, ln in _scan_blob(msg, HARD)]
    for field, var in (("author", "GIT_AUTHOR_IDENT"),
                       ("committer", "GIT_COMMITTER_IDENT")):
        ident = _git_out(["var", var])
        if ident is None:
            continue
        hits += [(field, label, ln) for label, ln in _scan_blob(ident, HARD)]
    return hits


def _redact(s: str) -> str:
    # Sanitise to ASCII so printing never crashes on a cp1252 console / pipe.
    s = s.strip().encode("ascii", "replace").decode("ascii")
    return (s[:120] + " ...") if len(s) > 120 else s


def _mask(line: str, rules: list) -> str:
    """`line` with every span any of `rules` matches replaced by `<label>`, so
    a finding's snippet never carries the secret it caught (audit P3-5). An
    empty match is left as it is."""
    for label, rx in rules:
        line = rx.sub(lambda m, _l=label: f"<{_l}>" if m.group(0) else "", line)
    return line


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
                hits.append((label, i, _redact(_mask(line, rules))))
    return hits


def main(argv: list[str]) -> int:
    # Honor the fail-closed degraded-scanner state HERE (the gate/hook entry),
    # not at import. Returning 2 → the __main__ wrapper exits 2, so the git hook
    # still blocks a commit when owner patterns couldn't load — while `import
    # tools.check_no_pii` from bug_reporter never terminates its host. 2026-07-14.
    if _LOAD_ERROR is not None:
        sys.stderr.write(f"[check_no_pii] refusing to scan: {_LOAD_ERROR}\n")
        return 2

    if "--commit-msg" in argv:
        i = argv.index("--commit-msg")
        if i + 1 >= len(argv):
            sys.stderr.write("[check_no_pii] --commit-msg needs a file path\n")
            return 2
        mh = _commit_msg_hits(argv[i + 1])
        if mh is None:
            sys.stderr.write("[check_no_pii] could not read the commit message\n")
            return 2
        for field, label, ln in mh:
            print(f"    ! commit {field}:{ln}  [{label}]")
        if mh:
            print(f"\n[check_no_pii] FAIL - {len(mh)} hard finding(s) in the commit "
                  "identity/message. Use a noreply identity "
                  "(git config user.email <id>+<user>@users.noreply.github.com).")
            return 1
        print("[check_no_pii] OK - commit identity and message are clean.")
        return 0

    if "--pre-push" in argv or "--metadata" in argv:
        if "--pre-push" in argv:
            ranges = _pre_push_rev_args(sys.stdin.read())
            if not ranges:
                print("[check_no_pii] nothing to push-scan.")
                return 0
        else:
            i = argv.index("--metadata")
            tail = [a for a in argv[i + 1:] if a != "--strict"]
            ranges = [tail or _default_rev_args()]
        total = 0
        for rev_args in ranges:
            mh = _commit_metadata_hits(rev_args)
            if mh is None:
                sys.stderr.write("[check_no_pii] could not read git history for "
                                 f"{' '.join(rev_args)}\n")
                return 2
            for sha, field, label, ln in mh:
                print(f"    ! commit {sha} {field}:{ln}  [{label}]")
            total += len(mh)
        if total:
            print(f"\n[check_no_pii] FAIL - {total} hard finding(s) in commit "
                  "metadata. This tool never rewrites history: fix the "
                  "identity (noreply address) and re-create the commit(s) "
                  "before pushing.")
            return 1
        print("[check_no_pii] OK - commit authors, committers and messages are clean.")
        return 0

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
    # Unreviewed binaries: a gate failure for the files git would commit, an
    # advisory in a directory (release-output) scan whose layout may differ.
    bin_root = os.path.abspath(positional[0]) if positional else _PROJECT_ROOT
    for rel, ln, label, snip in _binary_review_hits(files, bin_root):
        (warn_hits if positional else hard_hits).append((rel, ln, label, snip))
    for path in files:
        rel = os.path.relpath(path, _PROJECT_ROOT)
        for label, ln, snip in _scan_file(path, HARD):
            hard_hits.append((rel, ln, label, snip))
        for label, ln, snip in _scan_file(path, WARN):
            warn_hits.append((rel, ln, label, snip))

    # A finding is path:line [label] and NOTHING from the line itself. CI runs
    # this gate on a PUBLIC GitHub Actions log, where printing the matching
    # line (even cut to 120 chars) reprinted the very secret it had just
    # caught. path:line is enough to find it locally. Audit P3-5, 2026-10-01.
    print(f"[check_no_pii] scanned {len(files)} files ({scope})")
    if warn_hits:
        print(f"\n  WARN ({len(warn_hits)}) — review before public release:")
        for rel, ln, label, _snip in warn_hits[:200]:
            print(f"    ~ {rel}:{ln}  [{label}]")
    if hard_hits:
        print(f"\n  HARD ({len(hard_hits)}) — MUST be removed before commit/ship:")
        for rel, ln, label, _snip in hard_hits[:200]:
            print(f"    ! {rel}:{ln}  [{label}]")

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
