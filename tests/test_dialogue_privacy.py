"""Privacy guard for the device-dialogue hooks (2026-09-29).

The dialogue API is GENERIC: the device a private skill drives, its name, its
maker and its network address must never appear in this public repo. This
test scans every file the feature added or touched for the private words and
any LAN address. The patterns are assembled from fragments so this file does
not contain them itself.

stdlib unittest only; CI-safe.
    python -m unittest tests.test_dialogue_privacy
"""
from __future__ import annotations

import os
import re
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files the device-dialogue feature added or changed.
_FILES = (
    "core/dialogue.py",
    "core/device_speech_filter.py",
    "core/dispatcher.py",
    "core/services.py",
    "skills/standby_audio_detect.py",
    "skills/ambient_listen.py",
    "tests/test_dialogue.py",
    "tests/test_device_speech_filter_expect.py",
    "tests/test_dialogue_privacy.py",
    "tests/monolith/test_monolith_dialogue.py",
    "tests/_monolith_harness.py",
    "tests/_skill_harness.py",
    "tests/test_services.py",
    "tests/test_dispatcher.py",
    "tests/skills/test_ambient_listen.py",
    "tests/skills/test_standby_audio_detect.py",
)

_PATTERNS = [re.compile(p, re.I) for p in (
    "but" + "ter",                 # the device's product name and its bits
    "but" + "ler",                 # what the device calls JARVIS
    "circuit" + r"\s*" + "mess",   # the maker
    r"192\.168\.",                 # any LAN address
    "rick" + r"\s+and\s+" + "morty",
)]


class DialoguePrivacyTests(unittest.TestCase):
    def test_new_and_touched_files_carry_no_private_names(self):
        for rel in _FILES:
            path = os.path.join(_ROOT, *rel.split("/"))
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for pat in _PATTERNS:
                m = pat.search(text)
                self.assertIsNone(
                    m, f"{rel}: private pattern #{_PATTERNS.index(pat)} "
                       f"at offset {m.start() if m else -1}")

    def test_monolith_dialogue_block_is_generic(self):
        path = os.path.join(_ROOT, "bobert_companion.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        start = src.index("#  DEVICE DIALOGUES (core/dialogue.py)")
        end = src.index("def _do_proactive_turn(memory: dict):", start)
        block = src[start:end]
        for pat in _PATTERNS:
            self.assertIsNone(pat.search(block))

    def test_config_dialogue_block_is_generic(self):
        # core/config.py as a whole predates this feature (it carries older
        # generic robot placeholders), so only the dialogue block is scanned.
        path = os.path.join(_ROOT, "core", "config.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        start = src.index("# Device dialogues (core/dialogue.py)")
        end = src.index("DIALOGUE_LOST_HOLD_S", start)
        block = src[start:end + 40]
        for pat in _PATTERNS:
            self.assertIsNone(pat.search(block))

    def test_changelog_section_if_present(self):
        path = os.path.join(_ROOT, "CHANGELOG.md")
        if not os.path.exists(path):
            self.skipTest("no CHANGELOG.md in this tree")
        with open(os.path.join(_ROOT, "VERSION"), encoding="utf-8") as f:
            version = f.read().strip()
        with open(path, encoding="utf-8") as f:
            text = f.read()
        idx = text.find(version)
        if idx < 0:
            self.skipTest("no section for this version")
        nxt = text.find("\n## ", idx + 1)
        section = text[idx:nxt if nxt > 0 else len(text)]
        for pat in _PATTERNS:
            self.assertIsNone(pat.search(section))


if __name__ == "__main__":
    unittest.main()
