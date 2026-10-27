"""core.speech_plan.hold_head - the head hold (v3 W02, 2026-10-27).

The prose a reply writes BEFORE a held action token (one whose result will
still speak) is judged sentence by sentence. Fixtures are the live heads from
the 2026-10-16 / 10-19 / 10-26 session logs (JARVIS-Research
v3_overhaul_20261027/proto_head_hold.json, 02_act_then_speak.md) and the 52
pre-token heads of the brain A/B run (brain_ab_20261027 case_replies_pass1).

Pure: stdlib + core.claim_validator. Run:
    python -m unittest tests.test_speech_plan
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import speech_plan as sp

# (id-model, owner's words, head before the first token, first action)
AB_HEADS = (
    ('c002-qwen', 'Jarvis, how are you feeling with the new update?', "Very good, sir. I'm running smoothly, though I confess I remain skeptical of any software that promises to improve my wit.", 'version_info'),
    ('c004-qwen', 'Jarvis, is this the week that I need to put in hours?', '[intent:briefing] One moment, sir.', 'hours_status'),
    ('c006-gemma', 'Jarvis, get out of the way for just a second.', '[intent:confirmation] On it, sir.', 'game_mode_on'),
    ('c006-qwen', 'Jarvis, get out of the way for just a second.', '[intent:confirmation] On it, sir.', 'game_mode_on'),
    ('c007-gemma', 'Jarvis, what are you running now?', '[intent:briefing] Certainly, sir.', 'game_mode_status'),
    ('c008-gemma', 'Jarvis, do I have anything to do for school tonight?', '[intent:briefing] One moment, sir.', 'calendar_today'),
    ('c008-qwen', 'Jarvis, do I have anything to do for school tonight?', 'One moment, sir.', 'calendar_today'),
    ('c009-gemma', 'Jarvis, play some Skrillex. Shuffle all songs by Skrillex.', '[intent:confirmation] Certainly, sir.', 'play_music'),
    ('c009-qwen', 'Jarvis, play some Skrillex. Shuffle all songs by Skrillex.', '[intent:confirmation] Very good, sir.', 'play_music'),
    ('c010-gemma', 'Jarvis, you never started playing music.', '[intent:confirmation] My apologies, sir.', 'youtube_play'),
    ('c010-qwen', 'Jarvis, you never started playing music.', "[intent:bad_news] I'm afraid the previous command failed to execute, sir. Let me try again.", 'youtube_play'),
    ('c011-gemma', "Jarvis is downloading something. I don't know what he's downloading.", '[intent:briefing] One moment, sir.', 'check_system'),
    ('c012-gemma', 'Jarvis, turn on a video on YouTube that you think I might like on the left monitor.', '[intent:amused] A bold choice, sir.', 'open_on_monitor'),
    ('c013-gemma', 'Jarvis, pause the video.', '[intent:confirmation] Certainly, sir.', 'media_playpause'),
    ('c013-qwen', 'Jarvis, pause the video.', 'Very good, sir.', 'media_playpause'),
    ('c014-gemma', 'Jarvis open you to his homepage.', '[intent:confirmation] As you wish.', 'open_url'),
    ('c015-gemma', 'Jarvis, click that Omnitrix video for me.', '[intent:confirmation] Certainly, sir.', 'click'),
    ('c018-gemma', 'Jarvis knows everything about me, so he can pretty much just fill out the resume for me.', '[intent:confirmation] Certainly, sir.', 'browser_task'),
    ('c019-gemma', 'Jarvis, go ahead and take my resume to documents folder.', '[intent:confirmation] Certainly, sir.', 'move_file'),
    ('c020-gemma', 'Jarvis find the model for AA battery to C battery converter on Google.', '[intent:confirmation] Right away, sir.', 'web_search'),
    ('c020-qwen', 'Jarvis find the model for AA battery to C battery converter on Google.', '[intent:confirmation] Right away, sir.', 'web_search'),
    ('c021-gemma', 'Jarvis Jarvis now.', '[intent:confirmation] On it, sir.', 'shutdown'),
    ('c022-gemma', "Jarvis, how's the Bat and Blaze upload doing?", '[intent:briefing] One moment, sir.', 'backblaze_status'),
    ('c022-qwen', "Jarvis, how's the Bat and Blaze upload doing?", 'One moment, sir.', 'backblaze_status'),
    ('c023-gemma', 'Jarvis, what do you have linked to my Tuya account?', '[intent:briefing] One moment, sir.', 'tuya_list'),
    ('c023-qwen', 'Jarvis, what do you have linked to my Tuya account?', '[intent:briefing] One moment, sir.', 'tuya_list'),
    ('c024-gemma', 'Jarvis, I turned Blackblaze up to 100 threads just to get the Backblaze upload done.', '[intent:confirmation] On it, sir.', 'backblaze_status'),
    ('c025-gemma', "Jarvis, I want you to be able to do all this stuff that I'm saying. Pass that over to Claude as well.", '[intent:confirmation] Understood, sir.', 'note_for_claude'),
    ('c025-qwen', "Jarvis, I want you to be able to do all this stuff that I'm saying. Pass that over to Claude as well.", '[intent:confirmation] Very good, sir.', 'note_for_claude'),
    ('c026-gemma', 'Jarvis opened YouTube on the left monitor.', '[intent:confirmation] Very good, sir.', 'open_on_monitor'),
    ('c027-gemma', 'Jarvis, go ahead and click on that video that says, Can I find the rare daddy bear?', '[intent:confirmation] Certainly, sir.', 'click'),
    ('c027-qwen', 'Jarvis, go ahead and click on that video that says, Can I find the rare daddy bear?', 'Certainly, sir.', 'click'),
    ('c028-gemma', "Jarvis, it's on the left monitor left monitor.", '[intent:confirmation] Certainly, sir.', 'move_window_to_monitor'),
    ('c028-qwen', "Jarvis, it's on the left monitor left monitor.", '[intent:confirmation] Understood, sir.', 'click'),
    ('c029-gemma', 'Jarvis open teams on the top monitor.', '[intent:confirmation] Right away, sir.', 'open_on_monitor'),
    ('c029-qwen', 'Jarvis open teams on the top monitor.', '[intent:confirmation] Right away, sir.', 'open_on_monitor'),
    ('c030-gemma', 'Jarvis, turn on all smart home devices.', '[intent:confirmation] Certainly, sir.', 'list_smart_home_devices'),
    ('c030-qwen', 'Jarvis, turn on all smart home devices.', '[intent:confirmation] Very good, sir.', 'smart_home_control'),
    ('c032-gemma', "Jarvis, the headset is on though. I'm using it right now.", '[intent:confirmation] On it, sir.', 'which_mic_is_active'),
    ('c032-qwen', "Jarvis, the headset is on though. I'm using it right now.", '[intent:confirmation] Understood, sir.', 'which_mic_is_active'),
    ('c033-gemma', "Jarvis, how long until that data rebuild's done?", '[intent:briefing] Shall I run the numbers, sir?', 'server_status'),
    ('c033-qwen', "Jarvis, how long until that data rebuild's done?", 'One moment, sir.', 'server_status'),
    ('c036-gemma', 'Jarvis changed the lamp to red.', '[intent:confirmation] Very good, sir.', 'smart_home_control'),
    ('c037-qwen', 'Jarvis, how is the rebuild looking? Is it healthy?', '[intent:briefing] One moment, sir.', 'server_status'),
    ('c038-gemma', 'Jarvis, is there any free any resources I can free up?', '[intent:briefing] Certainly, sir.', 'game_mode_status'),
    ('c038-qwen', 'Jarvis, is there any free any resources I can free up?', '[intent:briefing] One moment, sir.', 'check_system'),
    ('c043-gemma', 'Jarvis, close all of that out for me.', '[intent:confirmation] As you wish, sir.', 'close_all_windows_except'),
    ('c045-qwen', 'yes', '[intent:confirmation] Understood, sir.', 'shutdown_jarvis'),
    ('c048-gemma', 'Yes, try to dampen the audio if you can.', '[intent:confirmation] Right away, sir.', 'volume_down'),
    ('c048-qwen', 'Yes, try to dampen the audio if you can.', '[intent:confirmation] Right away, sir.', 'volume_down'),
    ('c050-gemma', "Jarvis booted back up and it still has a flashing screen, like you're viewing my screen. I need Fortnite opened up.", '[intent:confirmation] Right away, sir.', 'launch_app'),
    ('c050-qwen', "Jarvis booted back up and it still has a flashing screen, like you're viewing my screen. I need Fortnite opened up.", 'Right away, sir.', 'launch_app'),
)


def _hold(head, held, result="ok", **kw):
    kw.setdefault("results", [(held, result, True)])
    return sp.hold_head(head, held_name=held, **kw)


class LiveHeadsTests(unittest.TestCase):
    """The five heads the prototype fired on, plus the plan's other two."""

    def test_200548_the_finding_before_the_look_is_dropped(self):
        head = ("I've taken a look at the screen, sir \u2014 it appears the "
                "video hasn't loaded or is simply stuck on a black frame.")
        kept, dropped = _hold(head, "local_describe_screen",
                              "A YouTube page.", ran_before=["screenshot"],
                              pending_lookup=True)
        self.assertEqual(kept, "")
        self.assertEqual([w for _s, w in dropped], ["finding"])

    def test_200540_the_inability_before_the_check_is_dropped(self):
        head = ("[intent:briefing] I'm afraid I've no way to check if the "
                "video is actually playing without running a scan, sir.")
        kept, dropped = _hold(head, "screenshot",
                              "screenshot saved to C:/shots/s.png")
        self.assertEqual(kept, "")
        self.assertEqual([w for _s, w in dropped], ["inability"])

    def test_104611_only_the_inability_is_dropped(self):
        head = ("I'm afraid I can't check that directly, sir; I can only "
                "confirm that your voice has reached me. To see which "
                "microphone is currently active, I'll need to run a "
                "diagnostic.")
        kept, dropped = _hold(head, "which_mic_is_active",
                              "Your active microphone is the headset.")
        self.assertEqual(kept, "To see which microphone is currently "
                               "active, I'll need to run a diagnostic.")
        self.assertEqual([w for _s, w in dropped], ["inability"])

    def test_101459_done_before_a_verbatim_read_out_is_dropped(self):
        kept, dropped = _hold("Done, sir.", "hand_mouse_off",
                              "Hand mouse off, sir.")
        self.assertEqual(kept, "")
        self.assertEqual(dropped, [("Done, sir.", "claim")])

    def test_200602_a_read_back_of_a_result_already_back_is_kept(self):
        head = ("I've looked at the screen locally, sir \u2014 the video is "
                "sitting right there on your top monitor, but it appears "
                "you've only managed to pull up the search results.")
        kept, dropped = _hold(head, "click_on_screen", "clicked",
                              ran_before=["screenshot",
                                          "local_describe_screen"],
                              pending_lookup=False)
        self.assertIs(kept, head)
        self.assertEqual(dropped, [])

    def test_194105_the_moot_offer_is_dropped(self):
        kept, dropped = _hold("[intent:briefing] Shall I run the numbers, "
                              "sir?", "system_pulse", "CPU 12%, memory 40%.")
        self.assertEqual(kept, "")
        self.assertEqual([w for _s, w in dropped], ["moot offer"])

    def test_194439_prepared_is_a_known_miss(self):
        # No "prepare" claim family in core.claim_validator (protected): the
        # head is kept, as the plan records.
        kept, dropped = _hold("I've prepared a few options, sir.",
                              "web_search", "searched")
        self.assertEqual(dropped, [])


class KeepTests(unittest.TestCase):
    def test_acknowledgements_and_progress_lines_are_kept(self):
        for head in ("Let me take a look, sir.", "One moment, sir.",
                     "Checking the printer now, sir.",
                     "[intent:confirmation] On it, sir.",
                     "Right away, sir. Let me check that for you."):
            with self.subTest(head=head):
                kept, dropped = _hold(head, "see_screen", "A text editor.")
                self.assertIs(kept, head)
                self.assertEqual(dropped, [])

    def test_a_question_that_is_not_the_last_sentence_is_kept(self):
        head = "Shall I run the numbers, sir? One moment."
        kept, _dropped = _hold(head, "system_pulse")
        self.assertIs(kept, head)

    def test_a_question_to_the_owner_is_kept_when_the_look_failed(self):
        head = "Would you like me to open it, sir?"
        kept, dropped = _hold(head, "see_screen",
                              "see_screen failed: no capture")
        self.assertIs(kept, head)
        self.assertEqual(dropped, [])

    def test_the_offer_is_kept_when_the_held_action_failed(self):
        head = "Shall I run the numbers, sir?"
        kept, _dropped = _hold(head, "system_pulse", held_failed=True)
        self.assertIs(kept, head)

    def test_an_inability_is_kept_when_the_check_failed(self):
        head = "I'm afraid I can't check the printer from here, sir."
        kept, dropped = _hold(head, "how_is_the_print",
                              "could not reach the printer")
        self.assertIs(kept, head)
        self.assertEqual(dropped, [])

    def test_the_leading_tag_survives_a_partial_drop(self):
        head = ("[intent:briefing] One moment, sir. It appears the print has "
                "stalled.")
        kept, dropped = _hold(head, "how_is_the_print", "Printing, 40%.")
        self.assertEqual(kept, "[intent:briefing] One moment, sir.")
        self.assertEqual([w for _s, w in dropped], ["finding"])

    def test_a_grounded_reading_is_kept(self):
        head = "The volume is at 40 percent, sir."
        kept, _dropped = _hold(head, "see_screen",
                               grounding_before=["Volume: 40%"],
                               pending_lookup=False)
        self.assertIs(kept, head)

    def test_an_ungrounded_reading_is_dropped(self):
        head = "The volume is at 40 percent, sir."
        kept, dropped = _hold(head, "audio_check", "Volume 55%.")
        self.assertEqual(kept, "")
        self.assertEqual([w for _s, w in dropped], ["reading"])


class WholeProseTests(unittest.TestCase):
    """An AMBIGUOUS / UNCLEAR autocorrect deferral: nothing ran."""

    def test_questions_stay_and_the_guessed_finding_goes(self):
        prose = ("Which one did you mean, sir? It appears the price is "
                 "about forty dollars.")
        kept, dropped = sp.hold_head(prose, whole=True, held_name="")
        self.assertEqual(kept, "Which one did you mean, sir?")
        self.assertEqual([w for _s, w in dropped], ["finding"])

    def test_no_moot_offer_or_inability_in_whole_mode(self):
        prose = "I can't see it from here, sir. Shall I look?"
        kept, _dropped = sp.hold_head(prose, whole=True, held_name="",
                                      pending_lookup=False)
        self.assertIs(kept, prose)


class AbHeadsTests(unittest.TestCase):
    def test_the_52_ab_heads_are_untouched_but_the_moot_offer(self):
        self.assertEqual(len(AB_HEADS), 52)
        fired = {}
        for rid, user, head, action in AB_HEADS:
            kept, dropped = sp.hold_head(
                head, results=[(action, "ok", True)], user_text=user,
                held_name=action)
            if dropped:
                fired[rid] = [w for _s, w in dropped]
            else:
                self.assertIs(kept, head, rid)
        # c033 is the 10-19 19:41:05 live pattern itself ("Shall I run the
        # numbers, sir?" with the token straight after).
        self.assertEqual(fired, {"c033-gemma": ["moot offer"]})

    def test_rules_1_to_4_fire_on_none_of_them(self):
        for rid, user, head, action in AB_HEADS:
            _kept, dropped = sp.hold_head(
                head, results=[(action, "ok", True)], user_text=user,
                held_name=action, held_failed=True)
            self.assertEqual(dropped, [], rid)


class FaultTests(unittest.TestCase):
    def test_a_fault_returns_the_input_unchanged(self):
        head = "It appears the video hasn't loaded, sir."
        with mock.patch.object(sp._cv, "find_completed_claim",
                               side_effect=RuntimeError("boom")):
            kept, dropped = _hold(head, "see_screen")
        self.assertIs(kept, head)
        self.assertEqual(dropped, [])

    def test_empty_and_odd_input(self):
        self.assertEqual(sp.hold_head("", held_name="x"), ("", []))
        self.assertEqual(sp.hold_head(None, held_name="x"), (None, []))
        self.assertEqual(sp.hold_head("[intent:briefing]", held_name="x"),
                         ("[intent:briefing]", []))
        # Results that cannot be read are a fault: the head is kept.
        self.assertEqual(sp.hold_head("Done.", results=object(),
                                      held_name="x"), ("Done.", []))


if __name__ == "__main__":
    unittest.main()
