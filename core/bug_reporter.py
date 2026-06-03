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
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OWNER = os.environ.get("JARVIS_GITHUB_OWNER", "Sk8ter2404")
_REPO = os.environ.get("JARVIS_GITHUB_REPO", "jarvis")
_OUTBOX = os.path.join(_ROOT, "data", "bug_reports.jsonl")

# (compiled pattern, replacement) applied IN ORDER. Specific secret shapes run
# before the generic catch-alls so a token isn't half-redacted by a broad rule.
_SCRUB_RULES: List[Tuple[Any, str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"), "<KEY>"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "<KEY>"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "<KEY>"),
    (re.compile(r"gho_[A-Za-z0-9]{20,}"), "<KEY>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<KEY>"),
    (re.compile(r"xox[bpoas]-[0-9A-Za-z-]{8,}"), "<KEY>"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{10,}"), "Bearer <KEY>"),
    # env-style "NAME = value" for sensitive names -> redact only the value
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|pwd|"
                r"access[_-]?code)\b(\s*[=:]\s*)\S+"), r"\1\2<REDACTED>"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    (re.compile(r"([A-Za-z]:\\Users\\)[^\\/\r\n\"']+"), r"\1<USER>"),
    (re.compile(r"(/(?:home|Users)/)[^/\r\n\"']+"), r"\1<USER>"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<IP>"),
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
