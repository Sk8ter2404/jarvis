#!/usr/bin/env python3
"""Capture, scrub, and (consent-gated) submit JARVIS bug reports.

Two sources feed this:
  * USER-reported  — the user tells JARVIS something is wrong (a `report_bug`
    action), and
  * SELF-detected  — JARVIS catches its own unhandled exception/error and files
    an automatic report.

Every report is SCRUBBED of personal data BEFORE it is written to the local
outbox or offered for submission — because on a shared/other-user instance a raw
traceback would otherwise carry THAT person's paths, secrets, and data. The
scrubber is deliberately conservative (it over-redacts rather than risk a leak):
home-dir/usernames, emails, IPv4 addresses, API-key/token shapes, and
env-style ``NAME = secret`` values all go before anything is persisted.

Nothing leaves the machine on its own: `browser_submit_url` returns a pre-filled
GitHub issue link the user opens and clicks (no token, no auto-send). The
autonomous API path + rate-limiting land in a later, consent-flagged increment.

import-light (stdlib only; `core.version` imported lazily), so it stays in the
CI light tier and is unit-testable without a JARVIS boot.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback as _tb
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OWNER = os.environ.get("JARVIS_GITHUB_OWNER", "Sk8ter2404")
_REPO = os.environ.get("JARVIS_GITHUB_REPO", "jarvis")
_OUTBOX = os.path.join(_ROOT, "data", "bug_reports.jsonl")


def _luhn(digits: str) -> bool:
    """True if `digits` (a run of 0-9) satisfies the Luhn checksum. This gate is
    what lets the credit-card rule redact real cards without eating arbitrary
    long digit runs (order numbers, timestamps): only a number that actually
    checksums as a card is removed."""
    total = 0
    # Walk right-to-left, doubling every second digit (the Luhn algorithm).
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_sub(m: "re.Match[str]") -> str:
    """Redact a 13-19 digit candidate to <CARD> only when it passes Luhn;
    otherwise leave the original text untouched (it isn't a card number)."""
    digits = m.group(0).replace(" ", "").replace("-", "")
    return "<CARD>" if _luhn(digits) else m.group(0)


# (compiled pattern, replacement) applied IN ORDER. Specific secret shapes run
# before the generic catch-alls so a token isn't half-redacted by a broad rule.
# A replacement may be a string OR a callable (re.sub semantics) — the card rule
# uses a callable so the Luhn check can veto a non-card digit run.
_SCRUB_RULES: List[Tuple[Any, Any]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"), "<KEY>"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "<KEY>"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "<KEY>"),
    (re.compile(r"gho_[A-Za-z0-9]{20,}"), "<KEY>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<KEY>"),
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "<KEY>"),                # Google API key
    (re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
     "<KEY>"),                                                       # JWT
    (re.compile(r"xox[bpoas]-[0-9A-Za-z-]{8,}"), "<KEY>"),
    (re.compile(r"https://hooks\.slack\.com/services/\S+"), "<KEY>"),  # Slack webhook
    (re.compile(r"-----BEGIN[^-]+PRIVATE KEY-----.*?-----END[^-]+PRIVATE KEY-----",
                re.DOTALL), "<KEY>"),                                # PEM private key
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{10,}"), "Bearer <KEY>"),
    # NAME=value / NAME: value for sensitive names — underscored names too
    # (OPENAI_API_KEY=, AWS_SECRET_ACCESS_KEY=, client_secret=) -> redact value.
    (re.compile(r"(?i)([A-Za-z0-9_]*(?:api[_-]?key|token|secret|password|passwd|"
                r"pwd|access[_-]?code)[A-Za-z0-9_]*)(\s*[=:]\s*)\S+"),
     r"\1\2<REDACTED>"),
    # connection-string password: scheme://user:pass@host  (user may be empty)
    (re.compile(r"(://[^:/@\s]*):[^@/\s]+@"), r"\1:<REDACTED>@"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    (re.compile(r"([A-Za-z]:\\Users\\)[^\\/\r\n\"']+"), r"\1<USER>"),
    (re.compile(r"(/(?:home|Users)/)[^/\r\n\"']+"), r"\1<USER>"),
    # MAC before the IP rules so its hex groups aren't partially eaten.
    (re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"), "<MAC>"),
    # Payment-card numbers: a 13-19 digit run (optionally space/dash grouped),
    # redacted ONLY when it Luhn-validates so order IDs/timestamps stay intact.
    # Before the IP/HEX rules (a spaced/contiguous card never reaches <HEX>,
    # which needs a contiguous 32+ hex run).
    (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _card_sub),
    # North-American phone numbers — conservative: a separator between the
    # 3-3-4 groups is REQUIRED, so a bare 10-digit id/timestamp isn't caught.
    (re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}"
                r"[\s.-]\d{4}(?!\d)"), "<PHONE>"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<IP>"),
    (re.compile(r"(?i)\b(?:[a-f0-9]{1,4}:){3,7}[a-f0-9]{1,4}\b"), "<IP>"),  # IPv6
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "<HEX>"),
]


def scrub(text: str) -> str:
    """Redact personal data/secrets from `text`. Conservative by design."""
    if not text:
        return ""
    out = str(text)
    for pat, repl in _SCRUB_RULES:
        out = pat.sub(repl, out)
    return out


def _version() -> str:
    try:
        from core.version import version_string
        return version_string()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def make_report(kind: str, summary: str, *, detail: str = "", tb: str = "",
                context: Optional[Dict[str, Any]] = None,
                version: Optional[str] = None,
                ts: Optional[float] = None) -> Dict[str, Any]:
    """Build a fully SCRUBBED report dict. `kind` normalises to 'auto'/'user'."""
    ctx: Dict[str, str] = {}
    for k, v in (context or {}).items():
        ctx[str(k)] = scrub(str(v))
    return {
        "kind": "auto" if kind == "auto" else "user",
        "summary": scrub(summary)[:300],
        "detail": scrub(detail)[:4000],
        "traceback": scrub(tb)[:6000],
        "context": ctx,
        "version": version if version is not None else _version(),
        "ts": ts if ts is not None else time.time(),
    }


def capture_exception(exc: BaseException, *, context: Optional[Dict[str, Any]] = None,
                      where: str = "") -> Dict[str, Any]:
    """Build a scrubbed 'auto' report from a caught exception + its traceback."""
    tb = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
    summary = f"{type(exc).__name__}: {exc}"
    if where:
        summary = f"[{where}] {summary}"
    return make_report("auto", summary, tb=tb, context=context)


def append_outbox(report: Dict[str, Any], path: str = _OUTBOX) -> bool:
    """Append one report as a JSON line to the local outbox. Never raises."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(report, ensure_ascii=False) + "\n")
        return True
    except OSError:
        return False


def record_bug(kind: str, summary: str, *, detail: str = "", tb: str = "",
               context: Optional[Dict[str, Any]] = None,
               outbox: str = _OUTBOX) -> Dict[str, Any]:
    """Make a scrubbed report, persist it to the outbox, and return it."""
    rep = make_report(kind, summary, detail=detail, tb=tb, context=context)
    append_outbox(rep, outbox)
    return rep


# Rate-limiter state for SELF-detected reports: signature -> last unix time.
_recent_auto: Dict[str, float] = {}


def auto_capture(exc: BaseException, *, where: str = "",
                 context: Optional[Dict[str, Any]] = None,
                 now: Optional[float] = None, window: float = 300.0,
                 outbox: str = _OUTBOX) -> Optional[Dict[str, Any]]:
    """Rate-limited SELF-detect capture. Build a scrubbed 'auto' report and
    record it to the outbox — UNLESS the same (exception-type, where) pair was
    already recorded within `window` seconds, so a recurring error can't spam
    the outbox. Returns the report, or None when suppressed.

    NEVER raises: a self-reporter that crashes the very thing it is reporting on
    would be worse than the original bug, so every failure path returns None."""
    try:
        t = now if now is not None else time.time()
        sig = f"{type(exc).__name__}|{where}"
        last = _recent_auto.get(sig)
        if last is not None and (t - last) < window:
            return None
        _recent_auto[sig] = t
        rep = capture_exception(exc, where=where, context=context)
        append_outbox(rep, outbox)
        return rep
    except Exception:  # pragma: no cover - the reporter must never raise
        return None


def format_issue(report: Dict[str, Any]) -> Tuple[str, str]:
    """Render a report as a (title, markdown-body) GitHub issue."""
    kind = report.get("kind", "user")
    tag = "auto-detected" if kind == "auto" else "user-reported"
    title = f"[{tag}] {report.get('summary', 'bug report')}"[:120]
    lines = [f"**Source:** {tag}",
             f"**Version:** {report.get('version', 'unknown')}", ""]
    if report.get("detail"):
        lines += ["**Details**", "", report["detail"], ""]
    if report.get("traceback"):
        lines += ["**Traceback**", "", "```", report["traceback"], "```", ""]
    ctx = report.get("context") or {}
    if ctx:
        lines += ["**Context**", ""]
        lines += [f"- {k}: {v}" for k, v in ctx.items()]
        lines.append("")
    lines.append("_Filed by the JARVIS bug reporter. Personal data was scrubbed "
                 "locally before submission._")
    return title, "\n".join(lines)


def browser_submit_url(report: Dict[str, Any], owner: str = _OWNER,
                       repo: str = _REPO) -> str:
    """A pre-filled GitHub 'new issue' URL — the user opens it and clicks submit
    (no token, explicit consent). Works for anyone on a public repo."""
    title, body = format_issue(report)
    q = urllib.parse.urlencode({"title": title, "body": body, "labels": "bug"})
    return f"https://github.com/{owner}/{repo}/issues/new?{q}"


def _issue_token() -> Optional[str]:
    for k in ("JARVIS_GITHUB_TOKEN", "GITHUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def auto_submit_enabled() -> bool:
    """Opt-in (default OFF): only when JARVIS_BUG_AUTO_SUBMIT=1 are reports POSTed
    to the GitHub API automatically. Otherwise submission stays the consent-gated
    browser-click path — nothing leaves the machine without a human."""
    return os.environ.get("JARVIS_BUG_AUTO_SUBMIT", "0") == "1"


def _issue_opener(url: str, payload: bytes, token: str) -> Optional[str]:  # pragma: no cover - real network
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json; charset=utf-8",
                 "User-Agent": "jarvis-bug-reporter"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("html_url")
    except (urllib.error.URLError, ValueError, OSError):
        return None


def api_submit_issue(report: Dict[str, Any], *, owner: str = _OWNER,
                     repo: str = _REPO, token: Optional[str] = None,
                     opener=None) -> Optional[str]:
    """POST `report` as a GitHub issue via the API; return the issue URL, or None
    (no token / API error). Total. `opener` is injectable for tests."""
    tok = token if token is not None else _issue_token()
    if not tok:
        return None
    title, body = format_issue(report)
    payload = json.dumps({"title": title, "body": body,
                          "labels": ["bug"]}).encode("utf-8")
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    fn = opener if opener is not None else _issue_opener
    try:
        return fn(url, payload, tok)
    except Exception:
        return None
