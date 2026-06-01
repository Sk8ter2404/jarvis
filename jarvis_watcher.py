#!/usr/bin/env python3
"""
JARVIS Queue Watcher
─────────────────────
Polls jarvis_todo.md every 30 seconds. When a new unticked task appears
(one that wasn't there last check), fires a Windows toast notification
so you know JARVIS captured something for you to action.

This is the simplest version — it just notifies. Wiring autonomous
Claude Code execution is a separate piece (you'd want to spawn the
`claude` CLI with the task as initial prompt, in a way that depends on
how your CLI is set up).

Run alongside JARVIS:
    python jarvis_watcher.py

Quit with Ctrl-C. Add a Windows shortcut to run on startup if you want it
to always be on.
"""

import hashlib
import os
import re
import subprocess
import sys
import time

POLL_SECONDS = 30
TODO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_todo.md")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jarvis_watcher_state")

# Regex to find unticked tasks: "- [ ] **date** — text"
TASK_RE = re.compile(r"^- \[ \] (.+)$", re.MULTILINE)


def get_unticked_tasks() -> list[str]:
    if not os.path.exists(TODO_FILE):
        return []
    try:
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []
    return TASK_RE.findall(content)


def hash_tasks(tasks: list[str]) -> str:
    """Hash the set of pending tasks so we can detect changes."""
    joined = "\n".join(sorted(tasks))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_seen_hash() -> str:
    if not os.path.exists(STATE_FILE):
        return ""
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def save_seen_hash(h: str):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write(h)
    except Exception:
        pass


def toast(title: str, body: str):
    """Show a Windows toast notification. Falls back to console on error."""
    if sys.platform == "win32":
        # Use PowerShell + BurntToast if available, else fall back to msg
        # Simplest universal approach: Win32 MessageBox via ctypes (non-blocking)
        ps_script = (
            "[Windows.UI.Notifications.ToastNotificationManager, "
            "Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null; "
            "$template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02; "
            "$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template); "
            f"$nodes = $xml.GetElementsByTagName('text'); "
            f"$nodes.Item(0).AppendChild($xml.CreateTextNode('{title.replace(chr(39), chr(39)*2)}')) | Out-Null; "
            f"$nodes.Item(1).AppendChild($xml.CreateTextNode('{body.replace(chr(39), chr(39)*2)}')) | Out-Null; "
            "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml); "
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('JARVIS').Show($toast)"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_script],
                timeout=5, capture_output=True,
            )
            return
        except Exception:
            pass
    # Fallback: just print
    print(f"\n  ╔══════════════════════════════╗")
    print(f"  ║ {title}")
    print(f"  ║ {body[:60]}")
    print(f"  ╚══════════════════════════════╝\n")


def main():
    print(f"JARVIS Queue Watcher — polling {TODO_FILE} every {POLL_SECONDS}s")
    print("Press Ctrl-C to stop.\n")

    seen_hash = load_seen_hash()
    initial = get_unticked_tasks()
    if not seen_hash:
        # First run: just note what's there, don't notify
        seen_hash = hash_tasks(initial)
        save_seen_hash(seen_hash)
        print(f"  [init] {len(initial)} pending task(s) — won't notify on these")
    else:
        print(f"  [init] {len(initial)} pending task(s) currently in queue")

    while True:
        try:
            time.sleep(POLL_SECONDS)
            tasks = get_unticked_tasks()
            h = hash_tasks(tasks)
            if h != seen_hash:
                # Find newly-added tasks (compared to last seen)
                prev = set()
                if seen_hash:
                    # Best-effort: we don't store the actual tasks, only the hash
                    # So just notify about the full pending set
                    pass
                seen_hash = h
                save_seen_hash(h)
                if tasks:
                    msg = tasks[-1] if len(tasks) <= 1 else f"{tasks[-1][:80]} (+{len(tasks)-1} more pending)"
                    toast("JARVIS task queued", msg)
                    print(f"  [{time.strftime('%H:%M:%S')}] queue changed — {len(tasks)} pending")
                    for t in tasks:
                        print(f"    - {t[:100]}")
        except KeyboardInterrupt:
            print("\nStopping watcher.")
            sys.exit(0)
        except Exception as e:
            print(f"  [error] {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
