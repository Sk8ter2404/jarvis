"""JARVIS-voice failure phrases keyed by failure class.

The dispatcher in bobert_companion.parse_and_run_actions() calls
classify_failure(exc, action_name) to pick a class, then jarvis_failure_line()
to draw a random in-character line from that class's bank. The result string
the dispatcher hands back to the LLM leads with the JARVIS line so the
follow-up reply already reads in-character, and carries the raw exception
text as a diagnostic suffix so the model can still reason about what broke.

To keep imports light this module is pure stdlib and classifies entirely
from the exception's class name + message — no dependency on any specific
network / COM / UI library.
"""
from __future__ import annotations
import random
import re
from typing import Iterable


_LINES: dict[str, list[str]] = {
    # 5
    "network": [
        "It seems the internet has opinions today, sir.",
        "I'm afraid the network is being uncooperative, sir.",
        "The connection appears to have stepped out for tea, sir.",
        "Slight problem, sir — the server isn't answering.",
        "I'm afraid I can't reach that just now, sir. Network unavailable.",
    ],
    # 4
    "permission": [
        "Windows is being precious about that one, sir.",
        "I'm afraid that's above my permissions, sir.",
        "Access denied — Windows is feeling protective today, sir.",
        "I'm afraid that file is locked, sir.",
    ],
    # 4
    "parse": [
        "I didn't quite catch the intent — say again, in shorter words if you can bear it.",
        "I'm afraid I couldn't parse that, sir.",
        "Slight problem, sir — the response wasn't quite what I expected.",
        "The data appears malformed, sir. I'm afraid I can't make sense of it.",
    ],
    # 4
    "app_not_found": [
        "I'm afraid that application isn't installed, sir. Shall I find an alternative?",
        "I can't seem to locate that program, sir.",
        "I'm afraid that one's missing from your Programs list, sir.",
        "That application doesn't appear to be installed, sir.",
    ],
    # 4
    "timeout": [
        "That's taking longer than I'd like. I'll keep watching.",
        "I'm afraid that's timed out, sir.",
        "No response, sir — I waited rather a while.",
        "It seems to be ignoring me, sir. Possibly hung.",
    ],
    # 3
    "com": [
        "I'm afraid the application isn't responding to me, sir.",
        "Slight problem, sir — the COM interface refused the call.",
        "iTunes is being temperamental again, sir.",
    ],
    # 3
    "ui_automation": [
        "I'm afraid I couldn't find that control on screen, sir.",
        "The window I needed appears to have vanished, sir.",
        "Slight problem, sir — I lost track of the cursor.",
    ],
    # 3
    "io": [
        "I'm afraid the file went missing, sir.",
        "Slight problem with the disk, sir.",
        "I couldn't write to that location, sir.",
    ],
    # 4
    "unknown": [
        "Slight problem, sir.",
        "I'm afraid that didn't quite go to plan, sir.",
        "Something rather unexpected happened, sir.",
        "I'm afraid I've hit a snag, sir.",
    ],
}

# Total: 5+4+4+4+4+3+3+3+4 = 34 entries.


# Ordered most-specific → most-general. The classifier walks this list and
# returns the first class whose regex matches against "{exc_class}: {msg} {action_name}".
_PATTERNS: list[tuple[str, str]] = [
    (r"\btimed[\s_-]?out\b|\btimeout\b|TimeoutError|TimeoutExpired|ReadTimeout|ConnectTimeout", "timeout"),
    (r"WinError\s*5\b|access (?:is )?denied|permission denied|operation not permitted|PermissionError|EACCES", "permission"),
    (r"ConnectionError|ConnectionRefused|ConnectionReset|ConnectionAborted|URLError|HTTPError|getaddrinfo failed|name or service not known|max retries exceeded|name resolution|\bDNS\b|gaierror|RemoteDisconnected|SSLError", "network"),
    (r"JSONDecodeError|ExpatError|XML(?:Syntax|Parse)Error|malformed|invalid literal|could not convert", "parse"),
    (r"COMError|com_error|CoInitialize|HRESULT|0x8[0-9a-f]{7}|-21\d{8}", "com"),
    (r"pyautogui|pygetwindow|ImageNotFoundException|FailSafeException|NoSuchWindow|window not found", "ui_automation"),
    (r"OSError|IOError|disk full|no space left|read[- ]only file system|too many open files", "io"),
]

# FileNotFoundError is ambiguous between app-not-found and generic IO. These
# tokens in the action name or message bump it into the app_not_found bucket.
_APP_HINT_RE = re.compile(
    r"\.exe\b|\.app\b|\.lnk\b|\bexecutable\b|launch|open_app|start_app|"
    r"play_music|spotify|apple_music|itunes|netflix|disney|chrome|edge|firefox|"
    r"focus_window|streaming",
    re.IGNORECASE,
)


def classify_failure(exc_or_msg, action_name: str = "") -> str:
    """Return one of the keys in _LINES for the given exception or message."""
    if isinstance(exc_or_msg, BaseException):
        cls_name = type(exc_or_msg).__name__
        msg = f"{cls_name}: {exc_or_msg}"
    else:
        msg = str(exc_or_msg or "")

    haystack = f"{msg} {action_name}".strip()
    haystack_lower = haystack.lower()

    # FileNotFoundError is context-dependent.
    if (
        "filenotfounderror" in haystack_lower
        or "winerror 2" in haystack_lower
        or "cannot find the file" in haystack_lower
        or "no such file" in haystack_lower
    ):
        if _APP_HINT_RE.search(haystack):
            return "app_not_found"
        return "io"

    for pat, klass in _PATTERNS:
        if re.search(pat, haystack, re.IGNORECASE):
            return klass

    return "unknown"


def jarvis_failure_line(failure_class: str) -> str:
    """Pick a random in-character line for the given class."""
    bank = _LINES.get(failure_class) or _LINES["unknown"]
    return random.choice(bank)


def failure_message(exc_or_msg, action_name: str = "") -> tuple[str, str, str]:
    """Convenience: classify + draw a line.

    Returns (failure_class, jarvis_line, technical_detail). The dispatcher
    typically renders these into:
        "{jarvis_line} (action failed; class={class}; {technical})"
    so the leading text reads in JARVIS voice while the suffix keeps the raw
    error visible to the follow-up LLM call.
    """
    klass = classify_failure(exc_or_msg, action_name)
    line = jarvis_failure_line(klass)
    if isinstance(exc_or_msg, BaseException):
        technical = f"{type(exc_or_msg).__name__}: {exc_or_msg}"
    else:
        technical = str(exc_or_msg or "")
    return klass, line, technical


def all_classes() -> Iterable[str]:
    """Diagnostic helper — list every registered class."""
    return list(_LINES.keys())
