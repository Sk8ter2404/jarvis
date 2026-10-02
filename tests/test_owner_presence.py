"""Light-tier tests for core/owner_presence.py (NEW #5, 2026-10-02).

The pure presence rule and the return-drain plan behind the speech-queue hold
in bobert_companion._speak_pending. Live 2026-10-01 the owner left at 18:27
and the wellness nudge (19:05), the credits nag (19:33) and a GPU pulse
(19:36) were spoken into the empty room; a wellness nudge also talked over a
live conversation (21:16). Stdlib only, no clock reads: every age is passed in.

    python -m unittest tests.test_owner_presence
"""
from __future__ import annotations

import unittest

from core import owner_presence as op

W = dict(voice_window_s=600, face_window_s=120, input_window_s=300)
INF = float("inf")


class PresenceVerdictTests(unittest.TestCase):
    def test_live_19_05_empty_room_is_away(self):
        # Last physical input 18:27:01 (38.5 min), no face, voice 45 min ago.
        present, why = op.presence_verdict(voice_age_s=45 * 60, face_age_s=None,
                                           input_age_s=38 * 60 + 29, **W)
        self.assertFalse(present)
        self.assertIn("owner voice 45 min ago", why)
        self.assertIn("no face", why)
        self.assertIn("physical input 38 min ago", why)

    def test_each_signal_alone_is_presence(self):
        for kw, label in ((dict(voice_age_s=30, face_age_s=None, input_age_s=INF),
                           "owner spoke"),
                          (dict(voice_age_s=None, face_age_s=20, input_age_s=INF),
                           "face seen"),
                          (dict(voice_age_s=None, face_age_s=None, input_age_s=9),
                           "physical input")):
            with self.subTest(label=label):
                present, why = op.presence_verdict(**kw, **W)
                self.assertTrue(present)
                self.assertIn(label, why)

    def test_bad_ages_are_unknown_not_present(self):
        for bad in (None, INF, float("nan"), -5, "x"):
            with self.subTest(bad=bad):
                present, _ = op.presence_verdict(voice_age_s=bad, face_age_s=bad,
                                                 input_age_s=bad, **W)
                self.assertFalse(present)


class RoomTalkTests(unittest.TestCase):
    def test_window(self):
        self.assertTrue(op.room_talk_recent(10, 45))
        self.assertFalse(op.room_talk_recent(46, 45))
        self.assertFalse(op.room_talk_recent(None, 45))

    def test_noise_words_are_not_talk(self):
        for noise in ("Thank you.", "You", "Bye bye.", "", None):
            with self.subTest(noise=noise):
                self.assertFalse(op.counts_as_room_talk(noise))
        self.assertTrue(op.counts_as_room_talk(
            "so anyway I told him we'd be there by eight"))


class RecapTests(unittest.TestCase):
    def test_fragment_strips_the_vocative_and_cuts_at_the_clause(self):
        self.assertEqual(op.recap_fragment(
            "GPU pinned at 100 percent, sir. CPU 12 percent."),
            "GPU pinned at 100 percent")
        self.assertEqual(op.recap_fragment(
            "Sir, the credits check needs a login - the page asked me."),
            "the credits check needs a login")
        self.assertEqual(op.recap_fragment(
            "[intent:urgent] Switched to your headset, sir."),
            "Switched to your headset")
        self.assertEqual(op.recap_fragment("The fan is at 99.5 percent."),
                         "The fan is at 99.5 percent")

    def test_fragment_is_bounded_at_a_word(self):
        frag = op.recap_fragment("word " * 40)
        self.assertLessEqual(len(frag), 70)
        self.assertFalse(frag.endswith(" "))

    def test_recap_line_shape(self):
        self.assertEqual(op.recap_line([]), "")
        self.assertEqual(op.recap_line(["A"]), "While you were away, sir: A.")
        self.assertEqual(op.recap_line(["A", "B", "C"], extra=2),
                         "While you were away, sir: A; B; and C, plus 2 more.")


class ReturnDrainTests(unittest.TestCase):
    NOW = 1_000_000.0

    def _e(self, msg, src, age):
        return {"ts": self.NOW - age, "message": msg, "source": src}

    def test_stale_status_folds_into_one_recap_in_place(self):
        timer = self._e("Reminder, sir — tea", "timer", 3000)
        items = [timer,
                 self._e("Hydration, perhaps?", "wellness", 2400),
                 self._e("GPU pinned at 99 percent, sir.", "pulse", 2000),
                 self._e("GPU pinned at 100 percent, sir.", "pulse", 1900),
                 self._e("Memory at 91 percent, sir.", "system_monitor", 1800),
                 self._e("Your print is done, sir.", "bambu", 30)]
        out, dropped, folded = op.plan_return_drain(items, now=self.NOW,
                                                    stale_s=600)
        self.assertEqual(out[0], timer)                  # exempt: as it was
        self.assertEqual(out[1]["source"], "recap:presence")
        recap = out[1]["message"]
        # The NEWEST line per source is named; one recap for all of them.
        self.assertIn("GPU pinned at 100 percent", recap)
        self.assertNotIn("99 percent", recap)
        self.assertIn("Memory at 91 percent", recap)
        self.assertEqual(out[2]["message"], "Your print is done, sir.")
        self.assertEqual(len(out), 3)
        self.assertEqual([d["source"] for d in dropped], ["wellness"])
        self.assertEqual(len(folded), 3)

    def test_keep_whole_and_fresh_lines_are_untouched(self):
        items = [self._e("Good evening, sir. Busy day.", "evening", 3600),
                 self._e("GPU pinned, sir.", "pulse", 60),
                 {"message": "no ts at all", "source": "pulse"}]
        out, dropped, folded = op.plan_return_drain(items, now=self.NOW,
                                                    stale_s=600)
        self.assertEqual(out, items)
        self.assertEqual((dropped, folded), ([], []))

    def test_every_briefing_source_is_kept_whole(self):
        # The morning arrival cold-open (skills/morning_arrival, source
        # "arrival") is a briefing like the others, never a recap fragment.
        for src in ("morning", "arrival", "evening", "daily", "news",
                    "handoff", "recap"):
            with self.subTest(src=src):
                self.assertIn(src, op.KEEP_WHOLE_SOURCES)
                self.assertNotIn(src, op.EXPIRE_QUIETLY_SOURCES)
        self.assertFalse(op.KEEP_WHOLE_SOURCES & op.EXPIRE_QUIETLY_SOURCES)
        self.assertFalse(op.EXEMPT_SOURCES & op.EXPIRE_QUIETLY_SOURCES)

    def test_named_fragments_are_capped(self):
        items = [self._e(f"Thing {i} happened, sir.", f"src{i}", 4000)
                 for i in range(6)]
        out, _, folded = op.plan_return_drain(items, now=self.NOW, stale_s=600)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["message"].endswith(", plus 3 more."))
        self.assertEqual(len(folded), 6)

    def test_never_raises(self):
        out, d, f = op.plan_return_drain([object(), 5, None], now="bad",
                                         stale_s=None)
        self.assertEqual(len(out), 3)

    def test_source_root(self):
        self.assertEqual(op.source_root({"source": "promise:memory"}), "promise")
        self.assertEqual(op.source_root({}), "")
        self.assertEqual(op.source_root("x"), "")


if __name__ == "__main__":
    unittest.main()
