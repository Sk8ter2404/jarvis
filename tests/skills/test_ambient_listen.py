"""Logic tests for skills/ambient_listen.py.

ambient_listen is the passive-transcription daemon (mic + system-audio +
screen). It's heavy (sounddevice / whisper / mss), so tests target the
deterministic, mockable helpers and the action surface — never a real stream:
  • wake-phrase regex compilation with word boundaries ('jar visit' ≠ jarvis),
  • screen blocklist compilation + sensitive-window matching (1Password,
    banking, auth screens),
  • the daily vision-budget tracker (remaining / charge / 14-day prune),
  • the average-hash + hamming distance helpers,
  • the rolling-buffer time-window + hard-cap trim,
  • wake-listener-active gating + the start/stop/status actions.

Disk paths (state file, budget, jsonl) are redirected to a temp dir. The
worker threads are neutered by the harness (Thread.start no-op), so the
start actions never open a device.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class AmbientListenHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")

    # ── _compile_wake_pattern ────────────────────────────────────────────
    def test_wake_pattern_word_boundary(self):
        bc = mock.MagicMock()
        bc.WAKE_PHRASES = {"jarvis", "hey jarvis"}
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            pat = self.mod._compile_wake_pattern()
        self.assertTrue(pat.search("ok jarvis are you there"))
        self.assertTrue(pat.search("hey jarvis"))
        # Word-boundary guard: 'jar visit' / 'jarvisible' must NOT match.
        self.assertIsNone(pat.search("jar visit the store"))
        self.assertIsNone(pat.search("jarvisible spectrum"))

    def test_wake_pattern_defaults_when_empty(self):
        bc = mock.MagicMock()
        bc.WAKE_PHRASES = set()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            pat = self.mod._compile_wake_pattern()
        self.assertTrue(pat.search("jarvis"))

    # ── blocklist / sensitive window ─────────────────────────────────────
    def test_sensitive_window_matches_password_managers(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            bl = self.mod._compile_blocklist()
        self.assertTrue(self.mod._is_sensitive_window("1Password — Vault", "1password.exe", bl))
        self.assertTrue(self.mod._is_sensitive_window("Chase Online Banking", "chrome.exe", bl))
        self.assertTrue(self.mod._is_sensitive_window("Bitwarden", "bitwarden.exe", bl))

    def test_sensitive_window_matches_auth_screens(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            bl = self.mod._compile_blocklist()
        self.assertTrue(self.mod._is_sensitive_window("Authenticator app", "", bl))
        self.assertTrue(self.mod._is_sensitive_window("Enter your SSN", "", bl))

    def test_non_sensitive_window_passes(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            bl = self.mod._compile_blocklist()
        self.assertFalse(self.mod._is_sensitive_window("report.docx - Word", "winword.exe", bl))

    def test_blocklist_tolerates_bad_extra_regex(self):
        bc = mock.MagicMock()
        bc.AMBIENT_SCREEN_BLOCKLIST = ("(",)   # invalid regex → skipped, not raised
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            bl = self.mod._compile_blocklist()
        # The defaults still compiled even though the extra one was bad.
        self.assertTrue(any(p.search("1password") for p in bl))

    # ── _hamming / _phash64 ──────────────────────────────────────────────
    def test_hamming_distance(self):
        self.assertEqual(self.mod._hamming(0b1010, 0b1010), 0)
        self.assertEqual(self.mod._hamming(0b1111, 0b1010), 2)

    def test_phash_identical_images_zero_distance(self):
        # Two solid-grey images hash identically → hamming 0.
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, format="PNG")
        png = buf.getvalue()
        h1 = self.mod._phash64(png)
        h2 = self.mod._phash64(png)
        self.assertIsNotNone(h1)
        self.assertEqual(self.mod._hamming(h1, h2), 0)

    def test_phash_returns_none_on_garbage(self):
        self.assertIsNone(self.mod._phash64(b"not a png"))


class AmbientListenBudgetTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self.tmp = tempfile.mkdtemp(prefix="ambient_budget_")
        self.addCleanup(self._cleanup)
        self.mod._DATA_DIR = self.tmp
        self.mod._BUDGET_PATH = os.path.join(self.tmp, "budget.json")

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def test_budget_remaining_starts_at_cap(self):
        bc = mock.MagicMock()
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertAlmostEqual(self.mod._vision_budget_remaining(), 1.0)

    def test_budget_charge_reduces_remaining(self):
        bc = mock.MagicMock()
        bc.AMBIENT_VISION_BUDGET_USD = 1.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._vision_budget_charge(0.30)
            self.assertAlmostEqual(self.mod._vision_budget_remaining(), 0.70)

    def test_budget_never_negative(self):
        bc = mock.MagicMock()
        bc.AMBIENT_VISION_BUDGET_USD = 0.50
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._vision_budget_charge(5.0)   # massive overspend
            self.assertEqual(self.mod._vision_budget_remaining(), 0.0)

    def test_budget_prunes_to_14_days(self):
        # Pre-seed 20 days of history; a charge should prune to the last 14.
        old = {f"2026-05-{d:02d}": 0.01 for d in range(1, 21)}
        with mock.patch.object(self.mod, "_save_budget") as save, \
             mock.patch.object(self.mod, "_load_budget", return_value=dict(old)):
            self.mod._vision_budget_charge(0.01)
        saved = save.call_args[0][0]
        self.assertLessEqual(len(saved), 14)


class AmbientListenStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self.mod._buffer.clear()

    # ── _trim_buffer ─────────────────────────────────────────────────────
    def test_trim_buffer_drops_old_entries(self):
        bc = mock.MagicMock()
        bc.AMBIENT_LISTEN_BUFFER_MINUTES = 10
        now = time.time()
        self.mod._buffer.append({"ts": now - 9999, "text": "old"})   # >10 min
        self.mod._buffer.append({"ts": now - 60, "text": "recent"})
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._trim_buffer(now)
        texts = [e["text"] for e in self.mod._buffer]
        self.assertEqual(texts, ["recent"])

    def test_trim_buffer_enforces_hard_cap(self):
        bc = mock.MagicMock()
        bc.AMBIENT_LISTEN_BUFFER_MINUTES = 100000   # window won't drop anything
        now = time.time()
        for i in range(self.mod._HARD_ENTRY_CAP + 50):
            self.mod._buffer.append({"ts": now, "text": str(i)})
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._trim_buffer(now)
        self.assertEqual(len(self.mod._buffer), self.mod._HARD_ENTRY_CAP)

    # ── _maybe_nudge_wake ────────────────────────────────────────────────
    def test_maybe_nudge_announces_on_match(self):
        import re
        bc = mock.MagicMock()
        bc._sleep_mode = [False]
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = 0.0
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("hey jarvis you there")
        bc.proactive_announce.assert_called_once()
        self.assertIn("heard my name", bc.proactive_announce.call_args[0][0])

    def test_maybe_nudge_debounces(self):
        import re
        bc = mock.MagicMock()
        bc._sleep_mode = [False]
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)
        self.mod._last_wake_at = time.time()   # just fired → debounce
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.mod._maybe_nudge_wake("jarvis again")
        bc.proactive_announce.assert_not_called()

    # ── _wake_listener_active ────────────────────────────────────────────
    def test_wake_listener_active_true(self):
        import sys
        wl = mock.MagicMock()
        wl._detector.is_running.return_value = True
        with mock.patch.dict(sys.modules, {"skill_wake_listener": wl}):
            self.assertTrue(self.mod._wake_listener_active())

    def test_wake_listener_inactive_when_module_absent(self):
        import sys
        sys.modules.pop("skill_wake_listener", None)
        self.assertFalse(self.mod._wake_listener_active())


class AmbientListenActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        # Ensure no daemon is recorded as alive between tests.
        self.mod._thread = None
        self.mod._audio_thread = None
        self.mod._screen_thread = None

    def test_listen_stop_when_not_running(self):
        self.assertIn("not running", self.actions["ambient_listen_stop"](""))

    def test_listen_start_refused_when_wake_listener_owns_mic(self):
        with mock.patch.object(self.mod, "_wake_listener_active", return_value=True):
            out = self.actions["ambient_listen_start"]("")
        self.assertIn("stop the wake-word listener first", out)

    def test_audio_start_windows_only_guard(self):
        # On non-Windows the action refuses immediately. On Windows it would
        # try to start (thread neutered); assert whichever branch this host hits.
        import sys
        out = self.actions["ambient_audio_start"]("")
        if sys.platform != "win32":
            self.assertIn("Windows-only", out)
        else:
            self.assertIn("System-audio capture", out)

    def test_screen_stop_when_not_running(self):
        self.assertIn("not running", self.actions["ambient_screen_stop"](""))

    def test_status_action_renders(self):
        out = self.actions["ambient_listen_status"]("")
        # Status should mention the mic listening state in some form.
        self.assertTrue(len(out) > 0)
        self.assertIn("sir", out.lower())


if __name__ == "__main__":
    unittest.main()
