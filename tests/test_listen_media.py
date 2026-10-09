"""Listening over media (2026-10-05): the pure policy (core/listen_media.py),
the wake rule's "hay" lead and sentence re-anchor (core/wake_prefix.py), the
STT word timing the re-anchor cuts by (core/stt_parakeet.word_times,
listen_media.segment_word_times) and the settings' four copies.

Owner, 10-05 ~02:00: "it seems like he's constantly listening, especially
when videos are playing, and he can't hear me - even in wake word mode."

Stdlib unittest, light tier (the monolith is never imported here;
tests/monolith/test_monolith_listen_media.py drives the real record_speech).

    python -m unittest tests.test_listen_media
"""
from __future__ import annotations

import ast
import json
import os
import re
import unittest

from core import listen_media as lm
from core import stt_parakeet as sp
from core import wake_prefix as wp

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(*parts) -> str:
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as f:
        return f.read()


# ── A1: "Hay Jarvis" ──────────────────────────────────────────────────────
class HayLeadTests(unittest.TestCase):
    """Parakeet writes "Hey Jarvis" as "Hay Jarvis": 18 of its 22 refusals
    on clean synthetic commands."""

    def test_hay_jarvis_is_addressed(self):
        for text in ("Hay Jarvis, pause the music.", "hay jarvis what time",
                     "Hay, Jarvis.", "Hay Jarvis"):
            with self.subTest(text=text):
                self.assertTrue(wp.has_wake_prefix(text))

    def test_hay_is_read_exactly_as_hey(self):
        # A legacy lead: no mention guard, like "hey Jarvis says hi".
        self.assertIn("hay", wp._LEGACY_LEADS)
        self.assertIn("hay", wp.WAKE_LEAD_FILLERS)
        for tail in ("says hi", "is it raining", "said what"):
            with self.subTest(tail=tail):
                self.assertEqual(wp.has_wake_prefix(f"hay Jarvis {tail}"),
                                 wp.has_wake_prefix(f"hey Jarvis {tail}"))

    def test_hay_is_handed_on_as_hey(self):
        # Every downstream stripper knows "hey jarvis", none "hay jarvis".
        self.assertEqual(wp.canonical_wake_text("Hay Jarvis, pause the music."),
                         "Hey Jarvis, pause the music.")
        self.assertEqual(wp.strip_wake_lead("Hay Jarvis, pause the music."),
                         "pause the music.")
        # The other legacy forms are still never rewritten.
        self.assertEqual(wp.canonical_wake_text("hey Jarvis, pause"),
                         "hey Jarvis, pause")

    def test_hay_is_not_a_spelling_of_the_name(self):
        # Fuzzy spellings of the NAME stay out (they would widen the rule
        # for every caller).
        for text in ("Jervis, pause", "Jarva pause", "hay Jervis pause"):
            with self.subTest(text=text):
                self.assertFalse(wp.has_wake_prefix(text))

    def test_the_dashboard_mirror_takes_hay_too(self):
        m = re.search(r"const WAKE_WORD_RE = /(.+)/i;",
                      _src("tools", "web_interface.py"))
        rx = re.compile(m.group(1), re.IGNORECASE)
        self.assertTrue(rx.search("Hay Jarvis, pause"))


# ── A2: the sentence re-anchor ────────────────────────────────────────────
class ReanchorTests(unittest.TestCase):

    def test_a_buried_command_is_found(self):
        r = wp.reanchor("Okay so that was close. Jarvis, pause the music.")
        self.assertIsNotNone(r)
        self.assertEqual(r.command, "Jarvis, pause the music.")
        self.assertEqual(r.sentence, 1)
        self.assertEqual((r.start_word, r.name_word, r.words), (5, 5, 9))

    def test_a_lead_before_the_name_is_kept(self):
        r = wp.reanchor("and that is how you do it! Hey Jarvis what time is it?")
        self.assertEqual(r.command, "Hey Jarvis what time is it?")
        self.assertEqual((r.start_word, r.name_word), (7, 8))

    def test_the_first_addressed_sentence_wins(self):
        r = wp.reanchor("one. two. Jarvis, stop. Jarvis, play jazz.")
        self.assertEqual(r.command, "Jarvis, stop. Jarvis, play jazz.")
        self.assertEqual(r.sentence, 2)

    def test_nothing_to_re_anchor(self):
        for text in ("Jarvis, pause",              # already addressed
                     "the video continues. So Jarvis said no.",  # mention
                     "price is 3.5 dollars Jarvis pause",  # no sentence end
                     "the video continues. I asked Jarvis.",
                     "no wake here. none here either.", "", None, 42):
            with self.subTest(text=text):
                self.assertIsNone(wp.reanchor(text))

    def test_quotes_and_ellipses_end_a_sentence(self):
        self.assertEqual(wp.reanchor('"Great." Jarvis, next track').command,
                         "Jarvis, next track")
        self.assertEqual(wp.reanchor("Wow… Jarvis, stop.").command,
                         "Jarvis, stop.")

    def test_the_name_word_index(self):
        self.assertEqual(wp.name_word_index("Jarvis pause"), 0)
        self.assertEqual(wp.name_word_index("Hey Jarvis, pause"), 1)
        self.assertEqual(wp.name_word_index("um okay Jarvis go"), 2)
        self.assertIsNone(wp.name_word_index("I asked Jarvis"))

    def test_the_rule_has_one_home(self):
        # The re-anchor rule lives in core/wake_prefix.py only; the monolith
        # calls it (never a copy of the sentence split).
        mono = _src("bobert_companion.py")
        self.assertTrue("_wake_prefix.reanchor(text)" in mono)
        self.assertFalse("_SENTENCE_END_RE" in mono)


class CutTimingTests(unittest.TestCase):
    """reanchor_cut_s: the audio is cut at the name (its STT timing) minus
    0.3 s - and never without timing."""

    def _r(self):
        return wp.reanchor("Okay so that was close. Jarvis, pause the music.")

    def test_cut_at_the_name(self):
        conf = {"word_t": [[i, 0.5 * i] for i in range(9)], "n_words": 9}
        self.assertAlmostEqual(lm.reanchor_cut_s(self._r(), conf), 2.2)

    def test_whisper_segment_start_stands_in(self):
        # Whisper gives the first word of each segment only; the sentence's
        # first word is the name here.
        conf = {"word_t": [[0, 0.0], [5, 9.75]], "n_words": 9}
        self.assertAlmostEqual(lm.reanchor_cut_s(self._r(), conf), 9.45)

    def test_no_timing_no_cut(self):
        for conf in (None, {}, {"word_t": [[0, 0.0]], "n_words": 9},
                     {"word_t": [[5, 3.0]], "n_words": 8}):
            with self.subTest(conf=conf):
                self.assertIsNone(lm.reanchor_cut_s(self._r(), conf))

    def test_never_negative(self):
        r = wp.reanchor("Hm. Jarvis go")
        self.assertEqual(lm.reanchor_cut_s(r, {"word_t": [[1, 0.1]],
                                               "n_words": 3}), 0.0)


class WordTimeTests(unittest.TestCase):

    def test_parakeet_tokens_become_word_times(self):
        toks = [" Ok", "ay", ",", " so", " that", " was", " close", ".",
                " Jar", "vis", ",", " pause", " the", " music", "."]
        ts = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 2.0, 2.1, 2.2, 2.4,
              2.6, 2.8, 3.0]
        text = "Okay, so that was close. Jarvis, pause the music."
        wt = sp.word_times(toks, ts, text)
        self.assertEqual(len(wt), len(text.split()))
        self.assertEqual(wt[5], [5, 2.0])          # "Jarvis,"
        self.assertEqual(wt[0], [0, 0.1])

    def test_mismatched_tokens_give_no_times(self):
        self.assertIsNone(sp.word_times([" a", " b"], [0.1, 0.2], "a b c"))
        self.assertIsNone(sp.word_times([" a"], [0.1, 0.2], "a"))
        self.assertIsNone(sp.word_times(None, None, "a"))

    def test_transcribe_carries_numbers_only(self):
        class Res:
            text = " Hay Jarvis, pause."
            tokens = [" Hay", " Jar", "vis", ",", " pause", "."]
            timestamps = [0.0, 0.32, 0.4, 0.48, 0.64, 0.96]
            logprobs = [-0.01] * 6

        class Eng:
            def recognize(self, a, sample_rate):
                return Res()

        import numpy as np
        text, conf = sp.transcribe(Eng(), np.zeros(16000, np.float32))
        self.assertEqual(text, "Hay Jarvis, pause.")
        self.assertEqual(conf["word_t"], [[0, 0.0], [1, 0.32], [2, 0.64]])
        self.assertEqual(conf["n_words"], 3)
        self.assertTrue(all(isinstance(v, (int, float))
                            for pair in conf["word_t"] for v in pair))

    def test_whisper_segments_give_their_first_words(self):
        segs = [("So that was close.", 0.0), ("Jarvis, pause the music.", 2.4)]
        self.assertEqual(lm.segment_word_times(
            segs, "So that was close. Jarvis, pause the music."),
            [[0, 0.0], [4, 2.4]])
        # A replacement that changed the word count drops the times.
        self.assertIsNone(lm.segment_word_times(segs, "So close. Jarvis go"))
        self.assertIsNone(lm.segment_word_times([("x", None)], "x"))


# ── B2 / D1 / D2 / D3 policy ──────────────────────────────────────────────
class PolicyTests(unittest.TestCase):

    def test_modes(self):
        self.assertEqual(lm.mode3("ON "), "on")
        self.assertEqual(lm.mode3("bogus"), "off")
        self.assertEqual(lm.mode2(True), "on")
        self.assertEqual(lm.mode2("shadow"), "off")
        self.assertEqual(lm.threshold_setting("x"), 0.15)
        self.assertEqual(lm.threshold_setting(0.01), 0.10)
        self.assertEqual(lm.threshold_setting(0.9), 0.50)

    def test_segments_only_when_the_echo_is_not_cancelled(self):
        on = dict(bus_on=True, wake_mode=True, media=True, aec_mode="off",
                  segment_s=12.0)
        self.assertTrue(lm.segment_active(**on))
        for k, v in (("bus_on", False), ("wake_mode", False),
                     ("media", False), ("aec_mode", "on"), ("segment_s", 0)):
            with self.subTest(k=k):
                self.assertFalse(lm.segment_active(**{**on, k: v}))

    def test_threshold_from_a_week_of_peaks(self):
        self.assertEqual(lm.threshold_from_peaks([0.9] * 5), 0.15)
        self.assertAlmostEqual(
            lm.threshold_from_peaks([0.2 + 0.01 * i for i in range(20)]),
            0.2 + 0.019, places=3)
        self.assertEqual(lm.threshold_from_peaks([0.05] * 30), 0.10)
        self.assertEqual(lm.threshold_from_peaks([0.9] * 30), 0.30)

    def test_trigger_refractory_and_flood_bump(self):
        tr = lm.PregateTrigger(0.15, rate_per_min=3)
        self.assertTrue(tr.offer(0.0, 0.5))
        self.assertFalse(tr.offer(0.5, 0.9))       # one per word
        self.assertTrue(tr.offer(2.0, 0.2))
        self.assertTrue(tr.offer(4.0, 0.2))
        self.assertFalse(tr.offer(6.0, 0.9))       # a 4th in a minute: flood
        self.assertEqual(tr.flooded, 1)
        self.assertAlmostEqual(tr.effective_threshold(10.0), 0.20)
        self.assertFalse(tr.offer(30.0, 0.19))     # still bumped
        self.assertTrue(tr.offer(130.0, 0.16))     # bump expired
        ev = tr.take_events()
        self.assertEqual(ev[0.15], 6)        # 0, 2, 4, 6, 30, 130
        self.assertEqual(tr.take_events()[0.15], 0)

    def test_veto_window(self):
        loop = lm.ScoreTrack()
        loop.add(10.0, 0.05)
        loop.add(10.5, 0.42)
        self.assertTrue(lm.vetoed_by(loop, 11.4))
        self.assertFalse(lm.vetoed_by(loop, 11.6))
        self.assertFalse(lm.vetoed_by(loop, 9.0))
        self.assertFalse(lm.vetoed_by(None, 10.5))
        weak = lm.ScoreTrack()
        weak.add(10.5, 0.29)
        self.assertFalse(lm.vetoed_by(weak, 10.5))

    def test_headset_bypass(self):
        self.assertTrue(lm.output_is_headset(
            "Headset Earphone (ACME HS-1 Wireless)", ""))
        self.assertTrue(lm.output_is_headset("Speakers (ACME HS-1)",
                                             "acme hs-1"))
        self.assertFalse(lm.output_is_headset("Speakers (Desk USB Audio)",
                                              "ACME HS-1"))
        self.assertFalse(lm.output_is_headset("", "x"))

    def test_score_track_window(self):
        tr = lm.ScoreTrack(history_s=5)
        for i in range(10):
            tr.add(float(i), i / 10.0)
        self.assertIsNone(tr.max_between(0, 3))    # forgotten
        self.assertAlmostEqual(tr.max_between(5, 7), 0.7)


class MinuteLineTests(unittest.TestCase):

    def test_one_numbers_only_line_per_media_minute(self):
        now = [0.0]
        c = lm.MinuteCounter(clock=lambda: now[0])
        c.note("captures", 3)
        c.note("refused", 2)
        c.note("mic_closed_s", 4.5)
        c.gauge("erle_db", 23.4)
        c.mark_media()
        now[0] = 30.0
        self.assertIsNone(c.tick())
        now[0] = 61.0
        line = c.tick()
        self.assertTrue(line.startswith("[listen-media] 61 s with media"))
        self.assertIn("captures 3", line)
        self.assertIn("refused 2", line)
        self.assertIn("mic closed 4.5 s", line)
        self.assertIn("aec erle 23.4 dB", line)
        # No media in the next minute: no line.
        c.note("captures")
        now[0] = 130.0
        self.assertIsNone(c.tick())

    def test_ambient_hits_the_main_loop_lost(self):
        a = lm.AmbientMatch(window_s=10)
        a.ambient_hit(100.0)
        a.main_turn(103.0)               # matched
        a.ambient_hit(200.0)             # never matched
        self.assertEqual(a.settle(150.0), 0)
        self.assertEqual(a.settle(215.0), 1)


# ── the settings' four copies (the stale-duplicate rule) ──────────────────
class SettingsCopiesTests(unittest.TestCase):

    def test_config_schema_example_and_fallbacks_agree(self):
        from core import config as cfg
        from tools import settings_window as sw
        with open(os.path.join(_ROOT, "tools", "user_settings.example.json"),
                  encoding="utf-8") as f:
            example = json.load(f)
        for key, want in lm.DEFAULTS.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(cfg, key), want)
                if key in lm.OWNER_FACING:
                    self.assertEqual(sw.SCHEMA[key]["default"], want)
                    self.assertEqual(example[key], want)
                else:
                    self.assertNotIn(key, sw.SCHEMA)   # a constant
        # The shipped defaults the spec names.
        self.assertEqual(lm.DEFAULTS["WAKE_REANCHOR_MODE"], "shadow")
        # Off: 'shadow' loads openWakeWord into the live process (2026-10-09
        # review) - its own canary first.
        self.assertEqual(lm.DEFAULTS["WAKE_PREGATE_MODE"], "off")
        self.assertEqual(lm.DEFAULTS["MIC_BUS_MODE"], "off")
        self.assertEqual(lm.DEFAULTS["MEDIA_AEC_MODE"], "off")

    def test_every_setting_has_a_reader_outside_config(self):
        # Dead-config guard: a key nothing reads would pass a hasattr test.
        mono = _src("bobert_companion.py")
        for key in lm.DEFAULTS:
            with self.subTest(key=key):
                self.assertTrue(f'globals().get("{key}"' in mono,
                                f"nothing outside config reads {key}")

    def test_the_monolith_fallbacks_read_the_one_copy(self):
        mono = _src("bobert_companion.py")
        tree = ast.parse(mono)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "get"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in lm.DEFAULTS
                    and isinstance(getattr(node.func, "value", None), ast.Call)
                    and getattr(node.func.value.func, "id", "") == "globals"):
                with self.subTest(key=node.args[0].value):
                    self.assertEqual(len(node.args), 2)
                    self.assertNotIsInstance(node.args[1], ast.Constant,
                                             "a literal fallback is a copy")

    def test_wake_listener_autostart_has_one_default(self):
        from core import config as cfg
        self.assertIs(cfg.WAKE_LISTENER_AUTOSTART, False)
        src = _src("skills", "wake_listener.py")
        self.assertIn("WAKE_WORD_AUTOSTART: bool = _config_autostart()", src)
        self.assertIn('"WAKE_LISTENER_AUTOSTART"', src)


# ── no test opens a real mic or loopback (spec §8) ────────────────────────
class RealCaptureGuardTests(unittest.TestCase):
    """tools/hermetic_guard.py refuses a REAL microphone / loopback capture
    from a test: sounddevice's InputStream, soundcard's recorder and
    pyaudiowpatch's open - the streams this feature opens in a live
    JARVIS."""

    def test_the_capture_entry_points_are_effect_targets(self):
        from tools import hermetic_guard as hg
        for target in (("sounddevice", None, "InputStream"),
                       ("soundcard.mediafoundation", "_Microphone",
                        "recorder"),
                       ("soundcard.mediafoundation", "_Microphone", "record"),
                       ("pyaudiowpatch", "PyAudio", "open")):
            with self.subTest(target=target):
                self.assertIn(target, hg._EFFECT_TARGETS)

    def test_a_real_input_stream_is_refused_before_it_opens(self):
        import importlib.util
        from tools import hermetic_guard as hg
        if importlib.util.find_spec("sounddevice") is None:
            self.skipTest("sounddevice is not installed here")
        if "input" in hg.unarmed_guards():
            self.skipTest("the input guard is not armed in this run")
        import sounddevice as sd
        self.assertTrue(hg._marked(sd.InputStream))
        # Keep this deliberate refusal out of the run's offender report.
        with hg._lock:
            saved = list(hg._refusals)
        try:
            with self.assertRaises(hg.InputGuardError):
                sd.InputStream(samplerate=16000, channels=1)
        finally:
            with hg._lock:
                hg._refusals[:] = saved


if __name__ == "__main__":
    unittest.main()
