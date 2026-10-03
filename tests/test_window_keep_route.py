"""core.dispatcher.window_keep_route - "close all windows except X" without
the brain (2026-10-03).

Live 17:22 a request to close every window but the Claude app reached the
model, which had no action for it: it ran list_windows and minimize_window six
times (JARVIS's own HUD and Reticle among them) and closed nothing. The
monolith's built-in utterance route asks window_keep_route, so a whole
"close / minimize all windows except X" request becomes ONE
close_all_windows_except / minimize_all_windows_except token - never a
minimize the model improvised.

It is deliberately NOT a chain / Controlled-mode rule (_INTENT_RULES): those
paths run their steps directly and would skip the "Close N windows?"
pushback. Phrases are paraphrased, names generic.

    python -m unittest tests.test_window_keep_route
"""
from __future__ import annotations

import unittest

from core import dispatcher

CLOSE = "close_all_windows_except"
MINIMIZE = "minimize_all_windows_except"


class WindowKeepRouteTests(unittest.TestCase):
    def route(self, text):
        return dispatcher.window_keep_route(text)

    def test_the_live_request_is_one_close_token(self):
        # The live sentence's shape: wake word, "all windows", "except for".
        self.assertEqual(self.route("Jarvis close all the windows except for "
                                    "Claude."),
                         f"[ACTION: {CLOSE}, Claude]")

    def test_close_phrasings(self):
        cases = {
            "close everything but Claude": "Claude",
            "Jarvis, close every window except Claude": "Claude",
            "close all but Claude and Spotify": "Claude and Spotify",
            "can you close all the windows but leave Claude open?": "Claude",
            "please close all windows and keep Claude open": "Claude",
            "close all of my apps apart from Claude": "Claude",
            "close every other window except Claude": "Claude",
            "close all open windows other than Claude and Notepad":
                "Claude and Notepad",
            "close everything else except Claude": "Claude",
            "close all windows except this one": "this one",
            # Parakeet writes the imperative as "closed" after the wake word.
            "Jarvis closed all windows except Claude": "Claude",
        }
        for text, keep in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.route(text),
                                 f"[ACTION: {CLOSE}, {keep}]")

    def test_minimize_phrasings_never_become_a_close(self):
        for text in ("minimize everything except Spotify",
                     "minimise all windows but Spotify",
                     "hide all apps except Spotify please"):
            with self.subTest(text=text):
                self.assertEqual(self.route(text),
                                 f"[ACTION: {MINIMIZE}, Spotify]")

    def test_close_never_becomes_a_minimize(self):
        tok = self.route("close every window but Claude")
        self.assertNotIn("minimize", tok)

    def test_left_to_the_model(self):
        for text in (
                "close all windows",                     # nothing to keep
                "close all tabs except this one",        # tabs, not windows
                "close everything except Claude and play some jazz",
                "close every window except Claude then open Spotify",
                "close everything except it",            # needs context
                "what windows are open except Claude",   # a question
                "don't close every window except Claude",
                "close Claude",
                "",
                None,
                42):
            with self.subTest(text=text):
                self.assertIsNone(self.route(text))

    def test_a_trailing_vocative_or_thanks_is_not_a_name_to_keep(self):
        # Review 2026-10-03: "..., Jarvis" was kept as a second name, so any
        # of the owner's windows titled with the word (a JARVIS folder or
        # workspace) stayed open and the summary said "kept Claude and
        # Jarvis"; "thanks" became "I saw no thanks window".
        for text in ("close all windows except Claude, Jarvis.",
                     "close everything but Claude, thanks",
                     "close every window except Claude thank you",
                     "close all windows except Claude, sir"):
            with self.subTest(text=text):
                self.assertEqual(self.route(text),
                                 f"[ACTION: {CLOSE}, Claude]")
        # "except Jarvis" itself is the model's to resolve.
        self.assertIsNone(self.route("close all windows except Jarvis"))

    def test_but_dont_close_x_keeps_x(self):
        self.assertEqual(self.route("close all windows but don't close "
                                    "Claude"), f"[ACTION: {CLOSE}, Claude]")
        self.assertEqual(self.route("minimize everything but do not "
                                    "minimize Claude"),
                         f"[ACTION: {MINIMIZE}, Claude]")

    def test_a_second_window_command_is_left_to_the_model(self):
        # Review 2026-10-03: the chain splitter's verb list has no window
        # verbs, so "..., minimize Spotify" rode inside the keep: Spotify
        # was KEPT (its name was in the keep) and never minimized, while the
        # route claimed the turn.
        for text in ("close all windows except Claude, minimize Spotify",
                     "close everything but Claude and hide Spotify",
                     "close all windows except Claude and focus Chrome",
                     "close everything except Claude and move Chrome to "
                     "the left monitor",
                     "minimize everything but Claude and maximize Excel"):
            with self.subTest(text=text):
                self.assertIsNone(self.route(text))

    def test_a_keep_that_names_no_window_is_left_to_the_model(self):
        for text in ("close all but one", "close every window but the one",
                     "close all windows except yours"):
            with self.subTest(text=text):
                self.assertIsNone(self.route(text))

    def test_minimize_everything_and_close_this_window_are_not_claimed(self):
        # Neither names a window to keep: "minimize everything" must never
        # become a close, and "close this window" is close_window's.
        for text in ("minimize everything", "minimise all windows",
                     "hide everything", "close this window",
                     "Jarvis, close this window please", "close that window"):
            with self.subTest(text=text):
                self.assertIsNone(self.route(text))

    def test_the_brain_sees_the_actions_for_these_requests(self):
        # When the route declines (a second command rides along), the model
        # must still have the token: the local prompt is sliced per turn, so
        # being in core/prompts.py is not enough (jarvis routing gate 3).
        from core.prompt_router import slim_pc_control
        from core.prompts import PC_CONTROL_PROMPT
        for text, name in (
                ("close everything except Claude and play some jazz", CLOSE),
                ("Jarvis close all the windows except for Claude", CLOSE),
                ("close all but Claude", CLOSE),
                ("minimise everything but Spotify", MINIMIZE),
                ("hide all apps except Chrome", MINIMIZE)):
            with self.subTest(text=text):
                self.assertIn(name, slim_pc_control(text, PC_CONTROL_PROMPT))

    def test_the_prompt_forbids_the_live_improvisation(self):
        from core.prompts import PC_CONTROL_PROMPT
        flat = " ".join(PC_CONTROL_PROMPT.split())
        self.assertIn(f"[ACTION: {CLOSE}, Claude]", flat)
        self.assertIn("never minimize when the user said close", flat)

    def test_close_is_not_a_chain_or_controlled_mode_rule(self):
        # Those paths run the handler directly - no pushback confirmation.
        actions = {CLOSE: lambda a: "", MINIMIZE: lambda a: "",
                   "play_music": lambda a: ""}
        self.assertIsNone(dispatcher.match_single_intent(
            "close every window except Claude", actions))
        rules = {r["action"] for r in dispatcher._INTENT_RULES}
        self.assertNotIn(CLOSE, rules)


if __name__ == "__main__":
    unittest.main()
