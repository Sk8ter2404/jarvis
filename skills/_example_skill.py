"""
Example skill — shows the format Bobert uses when writing new skills himself.
File starts with "_" so the loader ignores it, but it's a working reference.

To turn this into a real skill, copy/rename to a non-underscore name like:
    skills/open_dev_tabs.py
and restart Bobert.
"""

import time


def open_morning_tabs(_: str = "") -> str:
    """Open the user's usual morning browser tabs."""
    skill_utils["open_url"]("https://gmail.com")
    time.sleep(0.5)
    skill_utils["open_url"]("https://calendar.google.com")
    time.sleep(0.5)
    skill_utils["open_url"]("https://news.ycombinator.com")
    return "opened gmail, calendar, and hacker news"


def vscode_command_palette(query: str) -> str:
    """Open VS Code's command palette and type a query."""
    skill_utils["hotkey"]("ctrl", "shift", "p")
    time.sleep(0.4)
    skill_utils["type_text"](query)
    time.sleep(0.2)
    skill_utils["press_key"]("enter")
    return f"ran '{query}' in command palette"


def register(actions: dict):
    """Required: this function is called at load time to register actions."""
    actions["morning_tabs"]      = open_morning_tabs
    actions["vscode_command"]    = vscode_command_palette
