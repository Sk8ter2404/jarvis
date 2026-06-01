"""Tests for core.tts — the tone/preset selection layer, quip layer, the wry
helpers, and the new render cache. The pure helpers had only a __main__
self-test before; these promote them to real assertions, and the cache tests
pin the copy-safety + LRU + char-guard contract synthesise() relies on."""
import datetime
import unittest

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


if __name__ == "__main__":
    unittest.main()
