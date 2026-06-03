"""Tests for core.tts — the tone/preset selection layer, quip layer, the wry
helpers, and the new render cache. The pure helpers had only a __main__
self-test before; these promote them to real assertions, and the cache tests
pin the copy-safety + LRU + char-guard contract synthesise() relies on."""
import datetime
import json
import os
import tempfile
import unittest
from unittest import mock

import core.tts as tts


class SelectPresetTests(unittest.TestCase):
    def test_wry_wins(self):
        self.assertEqual(tts.select_preset("Quite, sir.", wry=True), "wry")

    def test_intent_passthrough(self):
        self.assertEqual(tts.select_preset("Right away.", intent="urgent"), "urgent")

    def test_emotion_mapping(self):
        self.assertEqual(tts.select_preset("Working.", emotion_label="stressed"), "calm")
        self.assertEqual(tts.select_preset("Yay.", emotion_label="excited"), "amused")
        self.assertEqual(tts.select_preset("Review.", emotion_label="focused"), "briefing")

    def test_bad_news_blocks_amused(self):
        # 'excited' would map to 'amused', but a bad_news line keeps its gravitas
        got = tts.select_preset("I'm afraid that's failed.",
                                emotion_label="excited", text_emotion="bad_news")
        self.assertEqual(got, "bad_news")

    def test_default_neutral(self):
        self.assertEqual(tts.select_preset("Hello."), "neutral")


class ResolveTtsPresetTests(unittest.TestCase):
    """The pure preset-priority resolver moved out of bobert_companion. The
    monolith keeps a thin _resolve_tts_preset(text, user_tone) shim that feeds
    the live _last_* state in as keyword args; these pin the priority chain."""

    # A daytime clock + a path that doesn't exist keep the context-preset
    # branch (emergency / late-hour / vocal-stress) inert, so the lower-tier
    # branches under test are deterministic regardless of when the suite runs.
    DAY = datetime.datetime(2026, 5, 31, 14, 0, 0)
    NO_STATE = "C:/__jarvis_no_such_anticipation_state__.json"

    def _resolve(self, text="", user_tone=None, **kw):
        kw.setdefault("now", self.DAY)
        kw.setdefault("state_path", self.NO_STATE)
        return tts.resolve_tts_preset(text, user_tone, **kw)

    def test_wry_wins_over_everything(self):
        name, preset = self._resolve("Quite, sir.", "excited", wry=True,
                                     intent_override="urgent", mood="urgent_clipped")
        self.assertEqual(name, "wry")
        self.assertEqual(preset, tts._TTS_EMOTION_PRESETS["wry"])

    def test_intent_override_and_alias(self):
        self.assertEqual(self._resolve("Now.", intent_override="urgent")[0], "urgent_alert")
        self.assertEqual(self._resolve("Now.", intent_override="dry")[0], "dry_wit")

    def test_mood_layer_between_intent_and_fallbacks(self):
        # mood loses to intent...
        self.assertEqual(self._resolve("x", intent_override="urgent",
                                       mood="dry_amused")[0], "urgent_alert")
        # ...but wins over emotion/user_tone when no intent is set.
        self.assertEqual(self._resolve("x", "excited", mood="concerned_soft",
                                       emotion_preset="amused")[0], "concerned_soft")

    def test_emotion_preset_wins_over_user_tone(self):
        name, preset = self._resolve("Working on it.", "excited", emotion_preset="calm")
        self.assertEqual(name, "calm")
        self.assertEqual(preset, tts._TTS_EMOTION_PRESETS["calm"])

    def test_bad_news_text_blocks_amused_emotion(self):
        # 'amused' would normally win, but a bad_news line keeps its gravitas.
        self.assertEqual(
            self._resolve("I'm afraid that's failed.", emotion_preset="amused")[0],
            "bad_news")

    def test_user_tone_mapping(self):
        self.assertEqual(self._resolve("ok", "stressed")[0], "bad_news")
        self.assertEqual(self._resolve("ok", "excited")[0], "excited")
        self.assertEqual(self._resolve("ok", "playful")[0], "confirmation")

    def test_user_tone_excited_blocked_on_bad_news(self):
        self.assertEqual(self._resolve("I'm afraid that failed.", "excited")[0],
                         "bad_news")

    def test_text_keyword_fallback(self):
        self.assertEqual(self._resolve("Very good, sir.")[0], "confirmation")

    def test_default_neutral(self):
        name, preset = self._resolve("Hello there.")
        self.assertEqual(name, "neutral")
        self.assertEqual(preset, tts._TTS_EMOTION_PRESETS["neutral"])

    def test_context_emergency_keyword_wins(self):
        # An emergency keyword in the USER's text beats the emotion/tone tiers
        # (and is clock-independent).
        self.assertEqual(
            self._resolve("ok", "excited", user_text="help me now",
                          emotion_preset="amused")[0],
            "brisk_alert")

    def test_context_vocal_stress(self):
        self.assertEqual(self._resolve("ok", user_text=None, peak_rms=0.25)[0],
                         "calm_low")

    def test_unknown_intent_falls_through(self):
        # An intent string not in _INTENT_PRESETS is ignored, not crashed on.
        self.assertEqual(self._resolve("Very good, sir.",
                                       intent_override="bogus_intent")[0],
                         "confirmation")


class WryTagTests(unittest.TestCase):
    def test_leading_tag_stripped(self):
        self.assertEqual(tts.parse_wry_tag("[wry] Quite, sir."), (True, "Quite, sir."))
        self.assertEqual(tts.parse_wry_tag("[ WRY ]   Indeed."), (True, "Indeed."))

    def test_no_tag(self):
        self.assertEqual(tts.parse_wry_tag("Just text"), (False, "Just text"))
        self.assertEqual(tts.parse_wry_tag("mid [wry] tag"), (False, "mid [wry] tag"))


class SplitForWryPauseTests(unittest.TestCase):
    def test_splits_two_sentences(self):
        head, tail = tts.split_for_wry_pause("Quite, sir. I had wondered when you'd ask.")
        self.assertTrue(head and tail)

    def test_no_split_for_short(self):
        self.assertEqual(tts.split_for_wry_pause("Short."), ("Short.", None))


class QuipLayerTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(tts.classify_action_for_quip("upgrade_jarvis"), "destructive")
        self.assertEqual(tts.classify_action_for_quip("play_music"), "media")
        self.assertEqual(tts.classify_action_for_quip("focus_window"), "ui")
        self.assertEqual(tts.classify_action_for_quip("see_screen"), "default")
        self.assertEqual(tts.classify_action_for_quip(None), "default")

    def test_long_text_never_quips(self):
        import random
        long_text = "I will close all the windows for you now."
        out = tts.jarvis_quip_layer(long_text, "close_all_windows",
                                    rng=random.Random(0), probability=1.0)
        self.assertEqual(out, long_text)

    def test_probability_zero_never_quips(self):
        import random
        out = tts.jarvis_quip_layer("Done.", "upgrade_jarvis",
                                    rng=random.Random(0), probability=0.0)
        self.assertEqual(out, "Done.")

    def test_probability_one_appends(self):
        import random
        out = tts.jarvis_quip_layer("Done.", "play_music",
                                    rng=random.Random(0), probability=1.0)
        self.assertNotEqual(out, "Done.")
        self.assertTrue(out.startswith("Done."))


class EmergencyKeywordTests(unittest.TestCase):
    def test_whole_word_only(self):
        self.assertTrue(tts.detect_emergency_keywords("help me now"))
        self.assertFalse(tts.detect_emergency_keywords("that was helpful"))
        self.assertFalse(tts.detect_emergency_keywords(""))


class RenderCacheTests(unittest.TestCase):
    def setUp(self):
        tts.tts_cache_clear()

    def test_put_get_roundtrip(self):
        self.assertTrue(tts.tts_cache_put("Yes, sir?", "v", "+0%", "+0Hz", [1, 2, 3], 24000))
        got = tts.tts_cache_get("Yes, sir?", "v", "+0%", "+0Hz")
        self.assertEqual(got, ([1, 2, 3], 24000))

    def test_miss_returns_none(self):
        self.assertIsNone(tts.tts_cache_get("never stored", "v", "+0%", "+0Hz"))

    def test_key_includes_rate_pitch_voice(self):
        tts.tts_cache_put("hi", "v1", "+0%", "+0Hz", [1], 24000)
        self.assertIsNone(tts.tts_cache_get("hi", "v2", "+0%", "+0Hz"))      # voice differs
        self.assertIsNone(tts.tts_cache_get("hi", "v1", "+5%", "+0Hz"))      # rate differs
        self.assertIsNotNone(tts.tts_cache_get("hi", "v1", "+0%", "+0Hz"))

    def test_copy_safety_on_return(self):
        tts.tts_cache_put("hi", "v", "+0%", "+0Hz", [1, 2, 3], 24000)
        audio, _ = tts.tts_cache_get("hi", "v", "+0%", "+0Hz")
        audio.append(999)                       # mutate the returned buffer
        audio2, _ = tts.tts_cache_get("hi", "v", "+0%", "+0Hz")
        self.assertEqual(audio2, [1, 2, 3])     # cache entry untouched

    def test_copy_safety_on_store(self):
        src = [1, 2, 3]
        tts.tts_cache_put("hi", "v", "+0%", "+0Hz", src, 24000)
        src.append(999)                         # mutate the source after storing
        audio, _ = tts.tts_cache_get("hi", "v", "+0%", "+0Hz")
        self.assertEqual(audio, [1, 2, 3])

    def test_overlong_text_not_cached(self):
        long_text = "x" * (tts.TTS_CACHE_MAX_CHARS + 1)
        self.assertFalse(tts.tts_cache_put(long_text, "v", "+0%", "+0Hz", [1], 24000))
        self.assertIsNone(tts.tts_cache_get(long_text, "v", "+0%", "+0Hz"))

    def test_empty_text_not_cached(self):
        self.assertFalse(tts.tts_cache_put("   ", "v", "+0%", "+0Hz", [1], 24000))

    def test_lru_eviction(self):
        for i in range(tts.TTS_CACHE_MAX_ENTRIES + 5):
            tts.tts_cache_put(f"phrase {i}", "v", "+0%", "+0Hz", [i], 24000)
        stats = tts.tts_cache_stats()
        self.assertEqual(stats["entries"], tts.TTS_CACHE_MAX_ENTRIES)
        self.assertIsNone(tts.tts_cache_get("phrase 0", "v", "+0%", "+0Hz"))   # oldest evicted
        self.assertIsNotNone(tts.tts_cache_get(
            f"phrase {tts.TTS_CACHE_MAX_ENTRIES + 4}", "v", "+0%", "+0Hz"))    # newest kept

    def test_stats_hit_rate(self):
        tts.tts_cache_put("hi", "v", "+0%", "+0Hz", [1], 24000)
        tts.tts_cache_get("hi", "v", "+0%", "+0Hz")   # hit
        tts.tts_cache_get("nope", "v", "+0%", "+0Hz")  # miss
        stats = tts.tts_cache_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)

    def test_stats_hit_rate_zero_when_no_lookups(self):
        # No gets at all → total 0 → hit_rate defaults to 0.0 (no ZeroDiv).
        s = tts.tts_cache_stats()
        self.assertEqual(s["hit_rate"], 0.0)

    def test_clear_resets_counters(self):
        tts.tts_cache_put("hi", "v", "+0%", "+0Hz", [1], 24000)
        tts.tts_cache_get("hi", "v", "+0%", "+0Hz")
        tts.tts_cache_clear()
        s = tts.tts_cache_stats()
        self.assertEqual(s["entries"], 0)
        self.assertEqual(s["hits"], 0)
        self.assertEqual(s["stores"], 0)


class IsMutedTests(unittest.TestCase):
    """is_muted() reads the MUTE_TTS env var fresh on every call."""

    def _set_env(self, value):
        # None → remove the var; else set it. Auto-restore on cleanup.
        old = os.environ.get(tts._MUTE_TTS_ENV)
        if value is None:
            os.environ.pop(tts._MUTE_TTS_ENV, None)
        else:
            os.environ[tts._MUTE_TTS_ENV] = value

        def restore():
            if old is None:
                os.environ.pop(tts._MUTE_TTS_ENV, None)
            else:
                os.environ[tts._MUTE_TTS_ENV] = old
        self.addCleanup(restore)

    def test_unset_is_not_muted(self):
        self._set_env(None)
        self.assertFalse(tts.is_muted())

    def test_truthy_values_muted(self):
        for v in ("1", "true", "yes", "on", "anything"):
            self._set_env(v)
            self.assertTrue(tts.is_muted(), f"{v!r} should mute")

    def test_explicit_falsey_values_not_muted(self):
        for v in ("", "0", "false", "no", "off", "  OFF  "):
            self._set_env(v)
            self.assertFalse(tts.is_muted(), f"{v!r} should not mute")


class ParseWryEmptyTests(unittest.TestCase):
    def test_empty_text_returns_false(self):
        # Line: `if not text: return False, text`
        self.assertEqual(tts.parse_wry_tag(""), (False, ""))
        self.assertEqual(tts.parse_wry_tag(None), (False, None))


class SplitForWryPauseBranchTests(unittest.TestCase):
    def test_empty_text_returns_none_tail(self):
        self.assertEqual(tts.split_for_wry_pause(""), ("", None))

    def test_comma_fallback_when_no_sentence_end(self):
        # No sentence-ending punctuation, but a comma with >=2 words each side
        # → the comma-fallback branch splits there.
        head, tail = tts.split_for_wry_pause(
            "between you and me, the diagnostics look grim")
        self.assertEqual(head, "between you and me")
        self.assertEqual(tail, "the diagnostics look grim")

    def test_comma_too_close_to_edges_no_split(self):
        # Comma present but one side has < 2 words → no usable split.
        self.assertEqual(tts.split_for_wry_pause("ok, go"), ("ok, go", None))

    def test_sentence_end_preferred_over_comma(self):
        head, tail = tts.split_for_wry_pause(
            "First clause, still going. Second sentence here.")
        # Splits on the sentence end, not the comma.
        self.assertEqual(head, "First clause, still going.")
        self.assertEqual(tail, "Second sentence here.")


class DetectLateHourTests(unittest.TestCase):
    """detect_late_hour combines a wall-clock check with an
    anticipation_state.json 'late_hour' trigger freshness check."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jv_tts_")
        self.state = os.path.join(self.tmp, "anticipation_state.json")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_state(self, obj):
        with open(self.state, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def test_late_wall_clock_true(self):
        # 23:30 local → at/after threshold (23) → True regardless of state file.
        late = datetime.datetime(2026, 5, 31, 23, 30, 0)
        self.assertTrue(tts.detect_late_hour(state_path=self.state, now=late))

    def test_early_clock_no_state_file_false(self):
        early = datetime.datetime(2026, 5, 31, 14, 0, 0)
        # state file absent → open() raises → returns False.
        self.assertFalse(tts.detect_late_hour(state_path=self.state, now=early))

    def test_state_trigger_fresh_true(self):
        early = datetime.datetime(2026, 5, 31, 14, 0, 0)
        import time as _t
        self._write_state({"last_trigger": "late_hour",
                           "last_proactive_at": _t.time()})
        self.assertTrue(tts.detect_late_hour(state_path=self.state, now=early))

    def test_state_trigger_stale_false(self):
        early = datetime.datetime(2026, 5, 31, 14, 0, 0)
        import time as _t
        self._write_state({"last_trigger": "late_hour",
                           "last_proactive_at": _t.time() - 99999})
        self.assertFalse(tts.detect_late_hour(state_path=self.state, now=early))

    def test_state_wrong_trigger_false(self):
        early = datetime.datetime(2026, 5, 31, 14, 0, 0)
        self._write_state({"last_trigger": "something_else",
                           "last_proactive_at": 123.0})
        self.assertFalse(tts.detect_late_hour(state_path=self.state, now=early))

    def test_state_non_dict_false(self):
        early = datetime.datetime(2026, 5, 31, 14, 0, 0)
        self._write_state(["not", "a", "dict"])
        self.assertFalse(tts.detect_late_hour(state_path=self.state, now=early))

    def test_state_unparseable_last_at_false(self):
        early = datetime.datetime(2026, 5, 31, 14, 0, 0)
        self._write_state({"last_trigger": "late_hour",
                           "last_proactive_at": "not-a-number"})
        self.assertFalse(tts.detect_late_hour(state_path=self.state, now=early))

    def test_now_none_uses_wall_clock(self):
        # now=None → datetime.now() is used. Patch it to a late hour.
        late = datetime.datetime(2026, 5, 31, 23, 45, 0)
        with mock.patch.object(tts.datetime, "datetime") as mdt:
            mdt.now.return_value = late
            self.assertTrue(tts.detect_late_hour(state_path=self.state))

    def test_hour_compare_exception_swallowed(self):
        # If the hour comparison raises (broken `now`), the except passes and
        # the function falls through to the state-file check (absent → False).
        class BadNow:
            @property
            def hour(self):
                raise RuntimeError("no hour")
        self.assertFalse(
            tts.detect_late_hour(state_path=self.state, now=BadNow()))


class DetectStressFromRmsTests(unittest.TestCase):
    def test_explicit_peak_above_threshold(self):
        self.assertTrue(tts.detect_stress_from_rms(0.5))

    def test_explicit_peak_below_threshold(self):
        self.assertFalse(tts.detect_stress_from_rms(0.01))

    def test_zero_or_negative_peak_false(self):
        self.assertFalse(tts.detect_stress_from_rms(0.0))
        self.assertFalse(tts.detect_stress_from_rms(-1.0))

    def test_unparseable_peak_false(self):
        self.assertFalse(tts.detect_stress_from_rms("loud"))   # TypeError/ValueError

    def test_none_peak_reads_audio_processor(self):
        # peak_rms=None → imports core.audio_processor.recent_peak_rms.
        fake_ap = mock.MagicMock()
        fake_ap.recent_peak_rms.return_value = 0.9
        with mock.patch.dict("sys.modules",
                             {"core.audio_processor": fake_ap}):
            self.assertTrue(tts.detect_stress_from_rms(None))

    def test_none_peak_import_failure_false(self):
        # If recent_peak_rms import/call raises, returns False (no crash).
        fake_ap = mock.MagicMock()
        fake_ap.recent_peak_rms.side_effect = RuntimeError("no audio")
        with mock.patch.dict("sys.modules",
                             {"core.audio_processor": fake_ap}):
            self.assertFalse(tts.detect_stress_from_rms(None))


class DetectContextPresetTests(unittest.TestCase):
    NO_STATE = "C:/__jv_no_anticipation__.json"
    DAY = datetime.datetime(2026, 5, 31, 14, 0, 0)

    def test_emergency_wins(self):
        self.assertEqual(
            tts.detect_context_preset("help!", peak_rms=0.9,
                                      now=self.DAY, state_path=self.NO_STATE),
            "brisk_alert")

    def test_late_hour_second_priority(self):
        late = datetime.datetime(2026, 5, 31, 23, 30, 0)
        self.assertEqual(
            tts.detect_context_preset("calm text", peak_rms=0.0,
                                      now=late, state_path=self.NO_STATE),
            "hushed_late")

    def test_vocal_stress_third_priority(self):
        self.assertEqual(
            tts.detect_context_preset("calm text", peak_rms=0.9,
                                      now=self.DAY, state_path=self.NO_STATE),
            "calm_low")

    def test_none_when_nothing_applies(self):
        self.assertIsNone(
            tts.detect_context_preset("calm text", peak_rms=0.0,
                                      now=self.DAY, state_path=self.NO_STATE))


class SelectPresetUserToneTests(unittest.TestCase):
    def test_user_tone_branch_maps(self):
        # Lines 432-434: emotion_label None, user_tone set → _USER_TONE_TO_PRESET.
        self.assertEqual(
            tts.select_preset("ok", user_tone="rushed"), "confirmation")
        self.assertEqual(
            tts.select_preset("ok", user_tone="late_night"), "concerned")

    def test_user_tone_bad_news_blocks_amused(self):
        # user_tone maps to 'amused' but text_emotion bad_news → skip it,
        # fall through to text_emotion return.
        self.assertEqual(
            tts.select_preset("ok", user_tone="playful",
                              text_emotion="bad_news"), "bad_news")

    def test_text_emotion_fallback_when_no_tone(self):
        self.assertEqual(
            tts.select_preset("ok", text_emotion="warning"), "warning")


class QuipLayerBranchTests(unittest.TestCase):
    def test_empty_text_returned_as_is(self):
        import random
        self.assertEqual(
            tts.jarvis_quip_layer("", "play_music",
                                  rng=random.Random(0), probability=1.0), "")

    def test_empty_pool_returns_text(self):
        import random
        # Patch the category pool to empty so the `if not pool` guard fires.
        with mock.patch.dict(tts._QUIP_POOLS, {"default": ()}):
            out = tts.jarvis_quip_layer("Done.", "see_screen",
                                        rng=random.Random(0), probability=1.0)
        self.assertEqual(out, "Done.")

    def test_adds_period_when_missing_terminal_punct(self):
        import random
        # "Done" (no trailing punctuation) → a period is inserted before aside.
        out = tts.jarvis_quip_layer("Done", "play_music",
                                    rng=random.Random(0), probability=1.0)
        self.assertTrue(out.startswith("Done. "))

    def test_keeps_existing_terminal_punct(self):
        import random
        out = tts.jarvis_quip_layer("Done!", "play_music",
                                    rng=random.Random(0), probability=1.0)
        self.assertTrue(out.startswith("Done! "))


class DetectTtsEmotionTests(unittest.TestCase):
    def test_empty_text_neutral(self):
        self.assertEqual(tts.detect_tts_emotion(""), "neutral")

    def test_bad_news_keyword(self):
        self.assertEqual(tts.detect_tts_emotion("I'm afraid not."), "bad_news")

    def test_no_keyword_neutral(self):
        self.assertEqual(tts.detect_tts_emotion("a plain sentence"), "neutral")


class ResolveContextExceptionTests(unittest.TestCase):
    def test_context_detector_exception_swallowed(self):
        # If detect_context_preset raises, resolve_tts_preset must swallow it
        # (ctx_preset=None) and continue to the lower tiers.
        with mock.patch.object(tts, "detect_context_preset",
                               side_effect=RuntimeError("ctx boom")):
            name, preset = tts.resolve_tts_preset(
                "Very good, sir.", None,
                now=datetime.datetime(2026, 5, 31, 14, 0, 0))
        self.assertEqual(name, "confirmation")


if __name__ == "__main__":
    unittest.main()
