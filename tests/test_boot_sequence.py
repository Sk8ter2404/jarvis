"""Unit tests for boot_sequence.py — the JARVIS "coming online" moment.

boot_sequence has no heavy / optional dependencies (just stdlib ``random`` and
``time``) so it runs unmodified on the reduced CI runner. The module is
deliberately decoupled from bobert_companion's globals: the caller passes in the
``speak`` function, the HUD-state writer, and the inventory data, so every test
drives it with simple fakes:

  * ``speak_fn``         — a callable that records what was spoken (or raises, to
                           exercise the swallow-exception paths),
  * ``write_hud_state``  — a callable that records the HUD field dicts,
  * ``rng``              — a ``random.Random`` with a pinned seed (or a tiny stub)
                           so phrase selection is deterministic.

``time.sleep`` is patched out in the one path that would otherwise block
(MIN_VISIBLE_SECONDS padding) so the suite never actually waits. ``time.time`` is
patched where elapsed-time arithmetic is asserted.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest). No personal data.
"""
from __future__ import annotations

import random
import unittest
from unittest import mock

import boot_sequence as bs


# ──────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────
class RecordingSpeaker:
    """Callable that records every line it is asked to speak."""

    def __init__(self, raise_on=None):
        self.spoken: list[str] = []
        self._raise_on = raise_on  # a substring → raise when seen

    def __call__(self, text):
        if self._raise_on is not None and self._raise_on in text:
            raise RuntimeError(f"tts boom on {text!r}")
        self.spoken.append(text)


class RecordingHud:
    """Callable that records each write_hud_state(**fields) call."""

    def __init__(self, raise_always=False):
        self.calls: list[dict] = []
        self._raise_always = raise_always

    def __call__(self, **fields):
        if self._raise_always:
            raise RuntimeError("hud boom")
        self.calls.append(dict(fields))


class _FixedRng:
    """Minimal rng stub exposing only .choice, returning a pinned element."""

    def __init__(self, index=0):
        self.index = index
        self.seen = None

    def choice(self, seq):
        self.seen = list(seq)
        return seq[self.index]


# ──────────────────────────────────────────────────────────────────────
# _humanise_ago
# ──────────────────────────────────────────────────────────────────────
class HumaniseAgoTests(unittest.TestCase):
    NOW = 1_000_000.0

    def _ago(self, seconds):
        return bs._humanise_ago(self.NOW - seconds, now=self.NOW)

    def test_zero_timestamp_is_never(self):
        self.assertEqual(bs._humanise_ago(0.0, now=self.NOW), "never")

    def test_negative_timestamp_is_never(self):
        self.assertEqual(bs._humanise_ago(-5.0, now=self.NOW), "never")

    def test_none_timestamp_is_never(self):
        # ``not then_ts`` covers None too.
        self.assertEqual(bs._humanise_ago(None, now=self.NOW), "never")

    def test_under_90s_is_moments_ago(self):
        self.assertEqual(self._ago(30), "moments ago")
        self.assertEqual(self._ago(89), "moments ago")

    def test_minutes_branch(self):
        self.assertEqual(self._ago(90), "1 minutes ago")
        self.assertEqual(self._ago(600), "10 minutes ago")

    def test_one_hour_singular(self):
        # 3600s rounds to exactly one hour → singular phrasing.
        self.assertEqual(self._ago(3600), "an hour ago")

    def test_multiple_hours_plural(self):
        self.assertEqual(self._ago(3600 * 5), "5 hours ago")

    def test_hours_rounding_to_one_is_singular(self):
        # 50 minutes < 3600 → still the minutes branch; 70 min rounds to ~1h.
        self.assertEqual(self._ago(3600 + 600), "an hour ago")  # 1.16h rounds→1

    def test_yesterday_branch(self):
        # Between 1 and 2 days → "yesterday".
        self.assertEqual(self._ago(86400 + 100), "yesterday")

    def test_days_branch_plural(self):
        self.assertEqual(self._ago(86400 * 3), "3 days ago")

    def test_default_now_uses_wall_clock(self):
        # When now is omitted the function reads time.time(); pin it.
        with mock.patch.object(bs.time, "time", return_value=self.NOW):
            self.assertEqual(bs._humanise_ago(self.NOW - 30), "moments ago")

    def test_boundary_3600_exact_is_hour_not_minutes(self):
        # delta == 3600 is NOT < 3600, so it falls to the hours branch.
        self.assertEqual(self._ago(3600), "an hour ago")

    def test_boundary_86400_exact_is_day_not_hours(self):
        # delta == 86400 is NOT < 86400, so it lands in the yesterday branch.
        self.assertEqual(self._ago(86400), "yesterday")


# ──────────────────────────────────────────────────────────────────────
# build_inventory_line
# ──────────────────────────────────────────────────────────────────────
class BuildInventoryLineTests(unittest.TestCase):
    def test_actions_and_skills(self):
        line = bs.build_inventory_line(12, 8, "", "", 0.0)
        self.assertIn("12 actions and 8 skills standing by, sir.", line)

    def test_actions_only(self):
        line = bs.build_inventory_line(5, 0, "", "", 0.0)
        self.assertIn("5 actions standing by, sir.", line)
        self.assertNotIn("skills", line)

    def test_skills_only(self):
        line = bs.build_inventory_line(0, 3, "", "", 0.0)
        self.assertIn("3 skills standing by, sir.", line)

    def test_neither_count(self):
        line = bs.build_inventory_line(0, 0, "", "", 0.0)
        self.assertTrue(line.startswith("Inventory loaded, sir."))

    def test_mic_and_distinct_speaker(self):
        line = bs.build_inventory_line(1, 1, "USB Mic", "Studio Monitors", 0.0)
        self.assertIn("Microphone on the USB Mic", line)
        self.assertIn("speakers on the Studio Monitors", line)

    def test_speaker_same_as_mic_is_collapsed(self):
        # When the speaker name matches the mic (case-insensitively), the
        # second clause is suppressed so we don't say the device twice.
        line = bs.build_inventory_line(1, 1, "Realtek Audio", "realtek audio", 0.0)
        self.assertIn("Microphone on the Realtek Audio", line)
        self.assertNotIn("speakers on", line)

    def test_mic_only_no_speaker(self):
        line = bs.build_inventory_line(1, 0, "Blue Yeti", "", 0.0)
        self.assertIn("Microphone on the Blue Yeti.", line)
        self.assertNotIn("speakers on", line)

    def test_speaker_only_no_mic(self):
        # No mic name but a speaker name: mic clause absent, but the speaker
        # clause still requires the speaker to differ from the (empty) mic.
        line = bs.build_inventory_line(1, 0, "", "Soundbar", 0.0)
        self.assertIn("speakers on the Soundbar", line)
        self.assertNotIn("Microphone on", line)

    def test_whitespace_device_names_ignored(self):
        line = bs.build_inventory_line(1, 1, "   ", "  ", 0.0)
        # Both strip to empty → no device clause at all.
        self.assertNotIn("Microphone", line)
        self.assertNotIn("speakers", line)

    def test_session_line_present_for_recent(self):
        with mock.patch.object(bs.time, "time", return_value=10_000.0):
            line = bs.build_inventory_line(1, 1, "", "", 10_000.0 - 30)
        self.assertIn("Last session was moments ago.", line)

    def test_session_line_omitted_when_never(self):
        line = bs.build_inventory_line(1, 1, "", "", 0.0)
        self.assertNotIn("Last session", line)

    def test_full_line_assembles_all_three_pieces(self):
        with mock.patch.object(bs.time, "time", return_value=10_000.0):
            line = bs.build_inventory_line(
                4, 2, "USB Mic", "Headphones", 10_000.0 - 600)
        self.assertEqual(
            line,
            "4 actions and 2 skills standing by, sir. "
            "Microphone on the USB Mic; speakers on the Headphones. "
            "Last session was 10 minutes ago.",
        )


# ──────────────────────────────────────────────────────────────────────
# pick_boot_line
# ──────────────────────────────────────────────────────────────────────
class PickBootLineTests(unittest.TestCase):
    def test_returns_a_known_line(self):
        self.assertIn(bs.pick_boot_line(), bs.BOOT_LINES)

    def test_uses_supplied_rng(self):
        rng = _FixedRng(index=2)
        self.assertEqual(bs.pick_boot_line(rng), bs.BOOT_LINES[2])
        self.assertEqual(rng.seen, bs.BOOT_LINES)

    def test_seeded_random_is_deterministic(self):
        # A seeded stdlib Random gives a stable pick run-to-run.
        a = bs.pick_boot_line(random.Random(123))
        b = bs.pick_boot_line(random.Random(123))
        self.assertEqual(a, b)

    def test_default_rng_path(self):
        # rng=None → uses the module-level ``random``; patch its choice.
        with mock.patch.object(bs.random, "choice",
                               return_value=bs.BOOT_LINES[0]) as ch:
            out = bs.pick_boot_line()
        ch.assert_called_once_with(bs.BOOT_LINES)
        self.assertEqual(out, bs.BOOT_LINES[0])


# ──────────────────────────────────────────────────────────────────────
# play_boot_sequence — orchestration / ordering
# ──────────────────────────────────────────────────────────────────────
class PlayBootSequenceTests(unittest.TestCase):
    def setUp(self):
        # Pin the chosen boot line so assertions are stable.
        self.rng = _FixedRng(index=0)
        self.boot_line = bs.BOOT_LINES[0]
        # Patch sleep globally for this class so MIN_VISIBLE padding never waits.
        p = mock.patch.object(bs.time, "sleep")
        self.sleep = p.start()
        self.addCleanup(p.stop)

    def test_returns_boot_and_inventory_lines(self):
        spk = RecordingSpeaker()
        boot, inv = bs.play_boot_sequence(
            spk, n_actions=3, n_skills=2, rng=self.rng)
        self.assertEqual(boot, self.boot_line)
        self.assertEqual(inv, bs.build_inventory_line(3, 2, "", "", 0.0))

    def test_speaks_boot_then_inventory_in_order(self):
        spk = RecordingSpeaker()
        boot, inv = bs.play_boot_sequence(
            spk, n_actions=1, n_skills=1, rng=self.rng)
        self.assertEqual(spk.spoken, [boot, inv])

    def test_hud_publish_start_and_clear(self):
        spk = RecordingSpeaker()
        hud = RecordingHud()
        bs.play_boot_sequence(spk, write_hud_state=hud, rng=self.rng)
        # Two HUD writes: powering then clear.
        self.assertEqual(len(hud.calls), 2)
        start, clear = hud.calls
        self.assertEqual(start["boot_phase"], "powering")
        self.assertEqual(start["boot_duration"], bs.BOOT_ANIMATION_SECONDS)
        self.assertEqual(start["state"], "Initialising")
        self.assertGreater(start["boot_started_at"], 0.0)
        self.assertEqual(clear["boot_phase"], "")
        self.assertEqual(clear["boot_started_at"], 0.0)
        self.assertEqual(clear["state"], "Idle")

    def test_no_hud_when_write_hud_state_none(self):
        spk = RecordingSpeaker()
        # Should not raise and should still speak both lines.
        boot, inv = bs.play_boot_sequence(spk, write_hud_state=None, rng=self.rng)
        self.assertEqual(spk.spoken, [boot, inv])

    def test_staging_skips_hud_entirely(self):
        spk = RecordingSpeaker()
        hud = RecordingHud()
        bs.play_boot_sequence(spk, write_hud_state=hud, staging=True, rng=self.rng)
        # Staging suppresses both the start and clear HUD publishes.
        self.assertEqual(hud.calls, [])

    def test_staging_skips_min_visible_sleep(self):
        spk = RecordingSpeaker()
        hud = RecordingHud()
        bs.play_boot_sequence(spk, write_hud_state=hud, staging=True, rng=self.rng)
        self.sleep.assert_not_called()

    def test_min_visible_sleep_when_speech_fast(self):
        # Fast TTS: elapsed is tiny → pad up to MIN_VISIBLE_SECONDS.
        spk = RecordingSpeaker()
        hud = RecordingHud()
        # started_at and the post-speech now differ by ~0 → full pad.
        times = iter([100.0, 100.0])  # time() for started_at, then elapsed calc

        def _time():
            try:
                return next(times)
            except StopIteration:
                return 100.0
        with mock.patch.object(bs.time, "time", side_effect=_time):
            bs.play_boot_sequence(spk, write_hud_state=hud, rng=self.rng)
        self.sleep.assert_called_once()
        (delay,), _ = self.sleep.call_args
        self.assertAlmostEqual(delay, bs.MIN_VISIBLE_SECONDS, places=3)

    def test_no_min_visible_sleep_when_speech_slow(self):
        # Slow TTS: elapsed already exceeds MIN_VISIBLE_SECONDS → no pad.
        spk = RecordingSpeaker()
        hud = RecordingHud()
        times = iter([100.0, 100.0 + bs.MIN_VISIBLE_SECONDS + 5])

        def _time():
            try:
                return next(times)
            except StopIteration:
                return 100.0 + bs.MIN_VISIBLE_SECONDS + 5
        with mock.patch.object(bs.time, "time", side_effect=_time):
            bs.play_boot_sequence(spk, write_hud_state=hud, rng=self.rng)
        self.sleep.assert_not_called()

    # ── failure / edge paths ─────────────────────────────────────────
    def test_boot_line_speak_failure_is_swallowed_and_inventory_still_speaks(self):
        # speak raises on the boot line; inventory must still be attempted and
        # the function must still return both strings.
        spk = RecordingSpeaker(raise_on=self.boot_line)
        boot, inv = bs.play_boot_sequence(spk, n_actions=1, rng=self.rng)
        self.assertEqual(boot, self.boot_line)
        # Boot raised (not recorded) but inventory was spoken.
        self.assertEqual(spk.spoken, [inv])

    def test_inventory_speak_failure_is_swallowed(self):
        inv = bs.build_inventory_line(1, 1, "", "", 0.0)
        spk = RecordingSpeaker(raise_on=inv)
        boot, got_inv = bs.play_boot_sequence(
            spk, n_actions=1, n_skills=1, rng=self.rng)
        # Boot spoke fine; inventory raised but was caught.
        self.assertEqual(spk.spoken, [boot])
        self.assertEqual(got_inv, inv)

    def test_hud_start_failure_is_swallowed(self):
        # HUD writer raises on the FIRST call; speech still proceeds.
        spk = RecordingSpeaker()
        hud = RecordingHud(raise_always=True)
        boot, inv = bs.play_boot_sequence(spk, write_hud_state=hud, rng=self.rng)
        self.assertEqual(spk.spoken, [boot, inv])

    def test_hud_clear_failure_is_swallowed(self):
        # HUD start succeeds, clear raises. Use a writer that raises only on the
        # clear (second) call.
        spk = RecordingSpeaker()

        class _HudFailClear:
            def __init__(self):
                self.n = 0

            def __call__(self, **fields):
                self.n += 1
                if self.n >= 2:
                    raise RuntimeError("clear boom")

        hud = _HudFailClear()
        boot, inv = bs.play_boot_sequence(spk, write_hud_state=hud, rng=self.rng)
        self.assertEqual(spk.spoken, [boot, inv])
        self.assertEqual(hud.n, 2)  # both publishes attempted

    def test_all_speak_failures_still_returns_lines(self):
        # Both speak calls raise → still returns the computed strings.
        spk = RecordingSpeaker(raise_on="")  # "" is in every string → always raise
        boot, inv = bs.play_boot_sequence(spk, n_actions=2, n_skills=1, rng=self.rng)
        self.assertEqual(boot, self.boot_line)
        self.assertEqual(inv, bs.build_inventory_line(2, 1, "", "", 0.0))
        self.assertEqual(spk.spoken, [])  # nothing recorded; both raised

    def test_default_kwargs_minimal_invocation(self):
        # Only speak_fn supplied; all counts default to 0, no HUD, no rng.
        spk = RecordingSpeaker()
        boot, inv = bs.play_boot_sequence(spk)
        self.assertIn(boot, bs.BOOT_LINES)
        self.assertTrue(inv.startswith("Inventory loaded, sir."))


# ──────────────────────────────────────────────────────────────────────
# Module constants sanity
# ──────────────────────────────────────────────────────────────────────
class ModuleConstantsTests(unittest.TestCase):
    def test_boot_lines_has_at_least_three_variations(self):
        # Spec: at least three variations so the same line doesn't always fire.
        self.assertGreaterEqual(len(bs.BOOT_LINES), 3)
        self.assertEqual(len(set(bs.BOOT_LINES)), len(bs.BOOT_LINES))

    def test_timing_constants_are_positive(self):
        self.assertGreater(bs.BOOT_ANIMATION_SECONDS, 0)
        self.assertGreater(bs.MIN_VISIBLE_SECONDS, 0)


if __name__ == "__main__":
    unittest.main()
