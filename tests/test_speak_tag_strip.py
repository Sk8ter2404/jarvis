"""v3 W01: a stray [intent:x] / [mood:x] / [wry] is never spoken.

_speak parses only LEADING tags; the model also writes them mid-reply (8 live
lines 2026-10-19 19:43-19:59, 2026-10-26 20:06:52; replay c041 / c042), and
the rest went to the synthesiser as words. Pure tests of core.speech_hygiene
plus the monolith's own strip chain (the same one _speak runs).

    python -m unittest tests.test_speak_tag_strip
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import speech_hygiene as sh
from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

C041 = ("[intent:briefing] I'm afraid I'll need a specific measurement or "
        "object to convert for you, sir. [intent:concerned] Without a "
        "reference point, I'm essentially scanning the void.")
C042 = ("[intent:briefing] I'm afraid I can't provide a specific cost without "
        "knowing which model you've settled on, sir. [intent:briefing] If you "
        "have a particular unit in mind, I can run the numbers for you.")
# The 10-19 19:44:19 shape: two sentences, a second tag between them.
L194419 = ("[intent:briefing] Most consumer models scan up to roughly 125 "
           "feet, sir. [intent:dry_wit] Industrial units reach 1,500 feet.")


class StripStrayTagsTests(unittest.TestCase):
    def test_mid_reply_tags_are_removed(self):
        for raw in (C041, C042, L194419):
            with self.subTest(raw=raw[:30]):
                body = raw.split("] ", 1)[1]        # past the leading tag
                out = sh.strip_stray_tags(body)
                self.assertNotIn("[intent", out)
                self.assertNotIn("  ", out)

    def test_words_stay_in_order_with_one_space(self):
        self.assertEqual(
            sh.strip_stray_tags("One, sir. [intent:concerned] Two, sir."),
            "One, sir. Two, sir.")

    def test_all_three_forms(self):
        self.assertEqual(sh.strip_stray_tags("a [mood:calm_efficient] b"), "a b")
        self.assertEqual(sh.strip_stray_tags("a [wry] b"), "a b")
        self.assertEqual(sh.strip_stray_tags("a [ Intent : bad_news ] b"), "a b")

    def test_no_space_before_closing_punctuation_or_at_the_ends(self):
        self.assertEqual(sh.strip_stray_tags("done [wry]."), "done.")
        self.assertEqual(sh.strip_stray_tags("[wry] done"), "done")
        self.assertEqual(sh.strip_stray_tags("done [intent:dry]"), "done")

    def test_action_and_other_brackets_are_never_altered(self):
        for t in ("On it. [ACTION: set_timer, 5]", "Press [Enter] now",
                  "[ACTION: mute_music] done [x]",
                  "see [intent] and [mood] and [wry thing]"):
            with self.subTest(t=t):
                self.assertEqual(sh.strip_stray_tags(t), t)

    def test_text_without_a_tag_is_returned_unchanged(self):
        for t in ("", "plain  text  with  gaps", None):
            self.assertEqual(sh.strip_stray_tags(t), t)

    def test_aliases_point_at_real_presets(self):
        from core import tts
        for alias, target in sh.INTENT_ALIASES.items():
            self.assertIn(target, tts._INTENT_PRESETS, alias)


@requires_monolith
class SpeakChainTests(MonolithGlobalsTestCase):
    """The strip chain _speak runs before synthesis (tag parse, then
    _strip_markdown_for_speech)."""

    def heard(self, text):
        bc = self.bc
        intent, t = bc._parse_intent_tag(text)
        _m, t = bc._parse_mood_tag(t)
        wry, t = bc._tts_layer.parse_wry_tag(t)
        if wry and not intent:
            intent, t = bc._parse_intent_tag(t)
        if wry:
            _m, t = bc._parse_mood_tag(t)
        return intent, bc._strip_markdown_for_speech(t).strip()

    def test_c041_c042_and_the_19_44_19_line_have_no_tag_text(self):
        for raw in (C041, C042, L194419):
            with self.subTest(raw=raw[:40]):
                intent, heard = self.heard(raw)
                self.assertEqual(intent, "briefing")
                self.assertNotIn("[intent", heard)
                self.assertNotIn("intent", heard.lower())
                self.assertNotIn("[", heard)

    def test_underscore_tag_is_gone_before_the_snake_case_rule(self):
        _i, heard = self.heard("[intent:briefing] One. [intent:bad_news] Two.")
        self.assertEqual(heard, "One. Two.")

    def test_action_text_is_never_altered(self):
        bc = self.bc
        t = "Right away [ACTION: set_timer, 5 minutes] [intent:dry] ok"
        on = bc._strip_markdown_for_speech(t)
        with mock.patch.object(bc, "SPEECH_HYGIENE", False):
            off = bc._strip_markdown_for_speech(t)
        self.assertIn("[ACTION:", on)
        # the only difference is the stray tag itself
        self.assertEqual(on, off.replace(" [intent:dry]", ""))

    def test_setting_off_is_the_old_behaviour(self):
        bc = self.bc
        with mock.patch.object(bc, "SPEECH_HYGIENE", False):
            _i, heard = self.heard(C041)
        self.assertIn("[intent:concerned]", heard)

    def test_leading_tag_routing_is_unchanged(self):
        bc = self.bc
        self.assertEqual(bc._parse_intent_tag("[intent:dry_wit] hi"),
                         ("dry_wit", "hi"))
        self.assertEqual(bc._parse_intent_tag("[intent:nonsense] hi"),
                         (None, "hi"))
        self.assertEqual(bc._parse_intent_tag("no tag"), (None, "no tag"))

    def test_concern_and_observation_route_to_a_preset(self):
        bc = self.bc
        self.assertEqual(bc._parse_intent_tag("[intent:concern] hi"),
                         ("concerned", "hi"))
        i, rest = bc._parse_intent_tag("[intent:observation] hi")
        self.assertIn(i, bc._INTENT_PRESETS)
        self.assertEqual(rest, "hi")
        with mock.patch.object(bc, "SPEECH_HYGIENE", False):
            self.assertEqual(bc._parse_intent_tag("[intent:concern] hi"),
                             (None, "hi"))


if __name__ == "__main__":
    unittest.main()
