"""core/onscreen_refs.py + the screen routes (core.dispatcher.screen_route)
+ the router gate (core.prompt_router) - 2026-10-05.

The live utterances (session 2026-10-05 00:21-00:33) are PARAPHRASED here,
keeping the shape that broke:
  * a compound "go ahead and click that Mr. Beast video and then go back
    into ... wake word mode" (the brain answered youtube_play);
  * "that's not the right video" / "the one that was on screen at the time";
  * "also tell Claude to ... research the screen vision ..." (the brain ran
    web_search) and "tell Claude that I want it to watch ..." (the brain
    claimed it relayed it).

    python -m unittest tests.test_onscreen_refs
"""
from __future__ import annotations

import unittest

from core import onscreen_refs as O
from core import prompt_router as PR
from core.dispatcher import screen_route, youtube_play_route
from core.prompts import PC_CONTROL_PROMPT

LIVE_COMPOUND = ("Jarvis, go ahead and click that Mr. Beast video and then go "
                 "back into, what's it called, wake word mode.")
LIVE_NOTE_1 = ("Jarvis, also tell Claude to go ahead and do some research on "
               "the screen vision. Your screen vision isn't working properly, "
               "you can't see things.")
LIVE_NOTE_2 = ("Jarvis, tell Claude that I want it to be able to watch what "
               "you see, so it can learn.")


class OnscreenReferenceTests(unittest.TestCase):
    POSITIVE = (
        LIVE_COMPOUND, "click that MrBeast video", "click the Save button",
        "play that MrBeast video on YouTube", "open that link",
        "watch this video", "pick the one on the left monitor",
        "the thing on my screen", "what's that right there",
        "the one you just opened", "click it again",
    )
    NEGATIVE = (
        "play that song again", "play a lofi mix on YouTube",
        "close that window", "play Eminem on YouTube",
        "search YouTube for MrBeast", "start the timer",
        "open that page again", "what time is it", "play some music",
        "put on that playlist",
    )

    def test_positives(self):
        for t in self.POSITIVE:
            self.assertTrue(O.is_onscreen_reference(t), t)

    def test_negatives(self):
        for t in self.NEGATIVE:
            self.assertFalse(O.is_onscreen_reference(t), t)

    def test_referent_phrase(self):
        self.assertEqual(O.referent_phrase(LIVE_COMPOUND), "that Mr. Beast video")
        self.assertEqual(O.referent_phrase("click on the Save button"),
                         "the Save button")
        self.assertEqual(O.referent_phrase("click that one"), "")


class ClickTargetTests(unittest.TestCase):
    def test_whole_requests(self):
        cases = {
            "click that MrBeast video": "that MrBeast video",
            "Jarvis, go ahead and click that Mr. Beast video":
                "that Mr. Beast video",
            "click the Save button": "the Save button",
            "play that MrBeast video on YouTube": "that MrBeast video",
            "click the video about bridges on the middle monitor":
                "the video about bridges",
            "tap the Subscribe button please": "the Subscribe button please",
        }
        for t, want in cases.items():
            self.assertEqual(O.onscreen_click_target(t), want, t)

    def test_left_to_the_brain(self):
        for t in (LIVE_COMPOUND, "click that one", "click that video",
                  "click it", "play that song", "click the Save button and "
                  "then close it", "play Eminem on YouTube",
                  "click [ACTION: x]"):
            self.assertIsNone(O.onscreen_click_target(t), t)


class CorrectionTests(unittest.TestCase):
    def test_other(self):
        for t in ("not that one", "Jarvis, that's not the right video.",
                  "wrong one", "the other one", "no, not that one",
                  "you clicked the wrong video", "that's not what I meant"):
            self.assertEqual(O.is_ui_correction(t), "other", t)

    def test_undo(self):
        for t in ("go back", "undo that", "Jarvis, go back please",
                  "take that back"):
            self.assertEqual(O.is_ui_correction(t), "undo", t)

    def test_not_corrections(self):
        for t in ("go back to sleep", "go back into wake word mode",
                  "that's not bad", "what is the other thing"):
            self.assertIsNone(O.is_ui_correction(t), t)

    def test_scene_back_reference(self):
        self.assertTrue(O.is_scene_back_reference(
            "Jarvis, I wanted the one that was on screen at the time."))
        self.assertTrue(O.is_scene_back_reference("the video that was there"))
        self.assertFalse(O.is_scene_back_reference("what's on screen now"))


class PendingChoiceTests(unittest.TestCase):
    OPTS = [{"label": "I Survived 7 Days In An Abandoned City",
             "rect": [320, 469, 284, 23]},
            {"label": "Ranking Every Burger", "rect": [774, 469, 296, 23]}]

    def test_ordinals_and_numbers(self):
        self.assertEqual(O.pending_choice_answer("the second one", self.OPTS), 1)
        self.assertEqual(O.pending_choice_answer("number one", self.OPTS), 0)
        self.assertEqual(O.pending_choice_answer("the last one", self.OPTS), 1)
        self.assertIsNone(O.pending_choice_answer("the third one", self.OPTS))

    def test_positions_and_words(self):
        self.assertEqual(O.pending_choice_answer("the left one", self.OPTS), 0)
        self.assertEqual(O.pending_choice_answer("the burger one", self.OPTS), 1)
        self.assertEqual(O.pending_choice_answer("the abandoned city one",
                                                 self.OPTS), 0)

    def test_yes_only_for_a_single_offered_option(self):
        self.assertIsNone(O.pending_choice_answer("yes", self.OPTS,
                                                  allow_yes=True))
        self.assertEqual(O.pending_choice_answer("yes please", self.OPTS[:1],
                                                 allow_yes=True), 0)
        self.assertIsNone(O.pending_choice_answer("yes", self.OPTS[:1]))


class RecallAndWatchTests(unittest.TestCase):
    def test_recall_requests(self):
        for t in ("what was that video on the middle monitor",
                  "what was I looking at ten minutes ago",
                  "the one that was on screen", "what was I watching"):
            self.assertTrue(O.is_screen_recall_request(t), t)
        self.assertFalse(O.is_screen_recall_request("what's on my screen"))

    def test_watch_commands(self):
        self.assertEqual(O.screen_memory_command("stop watching"),
                         {"op": "pause", "minutes": None})
        self.assertEqual(O.screen_memory_command(
            "stop watching my screen for 10 minutes"),
            {"op": "pause", "minutes": 10.0})
        self.assertEqual(O.screen_memory_command("stop watching for half an hour"),
                         {"op": "pause", "minutes": 30.0})
        self.assertEqual(O.screen_memory_command("don't watch this"),
                         {"op": "exclude_this"})
        self.assertEqual(O.screen_memory_command("don't watch Discord"),
                         {"op": "exclude_app", "app": "discord"})
        self.assertEqual(O.screen_memory_command("you can watch again"),
                         {"op": "resume"})
        self.assertEqual(O.screen_memory_command("are you watching?"),
                         {"op": "status"})
        self.assertIsNone(O.screen_memory_command("watch the game"))

    def test_forget_spans(self):
        self.assertEqual(O.forget_span("forget the last hour of what you saw"),
                         {"seconds": 3600.0})
        self.assertEqual(O.forget_span("forget the last 10 minutes of screen"),
                         {"seconds": 600.0})
        self.assertEqual(O.forget_span("forget what you saw today"),
                         {"today": True})
        self.assertEqual(O.forget_span("forget everything you saw on my screen"),
                         {"all": True})
        # memory's own "forget the last hour" is not claimed
        self.assertIsNone(O.forget_span("forget the last hour"))


class ClaudeNoteTests(unittest.TestCase):
    def test_both_live_utterances_are_notes(self):
        n1 = O.claude_note(LIVE_NOTE_1)
        self.assertTrue(n1 and "research" in n1, n1)
        n2 = O.claude_note(LIVE_NOTE_2)
        self.assertTrue(n2 and "watch" in n2, n2)

    def test_other_shapes(self):
        self.assertEqual(O.claude_note("leave Claude a note: the click is off"),
                         "the click is off")
        self.assertEqual(O.claude_note("let Claude know your clicking is broken"),
                         "your clicking is broken")

    def test_exclusions(self):
        for t in ("how many credits does Claude have left",
                  "what does Claude cost", "switch to Claude", "use Claude",
                  "open Claude", "ask Claude what the capital of France is",
                  "tell Claude hi", "use Claude Code to fix it"):
            self.assertIsNone(O.claude_note(t), t)

    def test_boundary_with_the_claude_route_predicate(self):
        # claude/router-arg-examples adds prompt_router.is_claude_route_request
        # ("use / switch to Claude"); whichever merges first, a note and a
        # route request never overlap.
        pred = getattr(PR, "is_claude_route_request", None)
        for t in ("use Claude", "switch back to Claude", "stop using Claude"):
            self.assertIsNone(O.claude_note(t), t)
            if pred is not None:
                self.assertTrue(pred(t), t)
        for t in (LIVE_NOTE_1, LIVE_NOTE_2):
            if pred is not None:
                self.assertFalse(pred(t), t)


class ScreenRouteTests(unittest.TestCase):
    def test_whole_click_request(self):
        self.assertEqual(screen_route("click that MrBeast video", {}),
                         "[ACTION: click_on_screen, that MrBeast video]")
        self.assertEqual(screen_route("play that MrBeast video on YouTube", {}),
                         "[ACTION: click_on_screen, that MrBeast video]")

    def test_compound_is_left_to_the_brain(self):
        self.assertIsNone(screen_route(LIVE_COMPOUND, {}))

    def test_corrections_need_a_recent_ui_action(self):
        self.assertIsNone(screen_route("that's not the right video", {}))
        self.assertEqual(screen_route("that's not the right video",
                                      {"recent_ui": True}),
                         "[ACTION: undo_click, other]")
        self.assertEqual(screen_route("go back", {"recent_ui": True}),
                         "[ACTION: undo_click]")
        self.assertIsNone(screen_route("go back", {}))

    def test_pending_choice(self):
        p = {"options": PendingChoiceTests.OPTS}
        self.assertEqual(screen_route("the second one", {"pending": p}),
                         "[ACTION: click_on_screen, pick:2]")
        self.assertIsNone(screen_route("the second one", {}))

    def test_scene_back_reference(self):
        self.assertEqual(screen_route(
            "I wanted the one that was on screen at the time", {"scenes": True}),
            "[ACTION: click_on_screen, scene:previous]")
        self.assertIsNone(screen_route(
            "I wanted the one that was on screen at the time", {}))

    def test_notes(self):
        tok = screen_route(LIVE_NOTE_1, {})
        self.assertTrue(tok.startswith("[ACTION: note_for_claude, "), tok)
        self.assertNotIn("[", tok[1:-1])

    def test_watch_controls(self):
        self.assertEqual(screen_route("stop watching", {"watching": True}),
                         "[ACTION: screen_memory, pause]")
        # bare "stop watching" with the watcher off / guard armed: brain's
        self.assertIsNone(screen_route("stop watching", {}))
        self.assertIsNone(screen_route("stop watching", {"watching": True,
                                                         "other_watch": True}))
        self.assertEqual(screen_route("stop watching my screen for 10 minutes",
                                      {}), "[ACTION: screen_memory, pause 10]")
        self.assertEqual(screen_route("you can watch again", {}),
                         "[ACTION: screen_memory, unpause]")
        self.assertEqual(screen_route("don't watch this", {}),
                         "[ACTION: screen_memory, exclude_this]")

    def test_forget(self):
        self.assertEqual(screen_route("forget the last hour of what you saw", {}),
                         "[ACTION: forget_screen, 60m]")
        self.assertIsNone(screen_route("forget the last hour", {}))

    def test_click_route_switch(self):
        self.assertIsNone(screen_route("click the Save button",
                                       {"click_route": False}))

    def test_youtube_route_declines_demonstratives(self):
        self.assertIsNone(youtube_play_route("play that MrBeast video on YouTube"))
        self.assertIsNone(youtube_play_route("play this one on YouTube"))
        self.assertEqual(youtube_play_route("play Eminem on YouTube"),
                         "[ACTION: youtube_play, Eminem]")


class RouterGateTests(unittest.TestCase):
    """The THIRD gate: the token must survive slim_pc_control."""

    def _slim(self, t):
        return PR.slim_pc_control(t, PC_CONTROL_PROMPT)

    def test_compound_utterance_ships_the_click_token(self):
        s = self._slim(LIVE_COMPOUND)
        self.assertIn("click, <description>", s)
        self.assertIn("never youtube_play", s)

    def test_recovery_and_memory_and_notes_ship(self):
        self.assertIn("undo_click", self._slim("not that one"))
        self.assertIn("undo_click", self._slim("go back"))
        self.assertIn("screen_memory", self._slim("stop watching my screen"))
        self.assertIn("forget_screen",
                      self._slim("forget the last hour of what you saw"))
        self.assertIn("note_for_claude", self._slim(LIVE_NOTE_1))
        self.assertIn("note_for_claude", self._slim(LIVE_NOTE_2))
        self.assertIn("recall_screen",
                      self._slim("what was that video on the middle monitor"))

    def test_new_arrow_examples_round_trip(self):
        # Every new "'phrase' -> [ACTION: x, ...]" example keeps x.
        for phrase, action in (
                ("click that MrBeast video", "click"),
                ("go back", "undo_click"), ("not that one", "undo_click"),
                ("stop watching my screen", "screen_memory"),
                ("you can watch again", "screen_memory"),
                ("forget the last hour of what you saw", "forget_screen"),
                ("tell Claude to fix the clicking", "note_for_claude"),
                ("play a lofi mix on YouTube", "youtube_play")):
            self.assertIn(action, self._slim(phrase), phrase)

    def test_no_demonstrative_teaches_search(self):
        self.assertNotIn("'play that lofi mix'", PC_CONTROL_PROMPT)
        self.assertNotIn("default to the PRIMARY (middle)", PC_CONTROL_PROMPT)

    def test_turn_block_growth_is_bounded_and_reported(self):
        # The per-turn block rides after the KV-cached prefix; > ~1,024 tokens
        # of divergence costs a re-eval (~2.5 s). Report the sizes (chars).
        sizes = {}
        for t in (LIVE_COMPOUND, "click that MrBeast video", "go back",
                  "stop watching my screen", LIVE_NOTE_1):
            sizes[t[:30]] = len(PR.turn_pc_block(t, PC_CONTROL_PROMPT))
        print(f"\n  [turn_pc_block chars] {sizes}")
        for t, n in sizes.items():
            self.assertLess(n, 16000, t)


if __name__ == "__main__":
    unittest.main()
