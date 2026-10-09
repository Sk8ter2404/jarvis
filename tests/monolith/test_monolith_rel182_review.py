"""Cross-branch findings of the v2.0.182 integration review (2026-10-09),
through the real monolith:

  1. one look at the screen is ONE entry in the turn's screen texts (main's
     _note_turn_action_ran and screen-vision's _note_screen_look both fed
     it, so the sign-in guard's last-4 window held only the last 2 looks);
  2. JARVIS's open screen question ("shall I press 'Delete account'?") is
     answered by the owner's next turn or not at all - a declined one used
     to stay armed for 90 s, and a later "yes" to a question the brain
     asked in prose pressed it;
  3. "do that again" never re-fires forget_screen / forget_voice_line;
  4. no seeding of the clone cache in guest mode (the take would not be
     kept, and its line would count as tried for good).

    python -m unittest tests.monolith.test_monolith_rel182_review
"""
from __future__ import annotations

import collections
import threading
import time
import types
from unittest import mock

from core import guest_mode as GM
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

SIGN_IN = ("Chrome - Sign in to continue. Choose an account to continue to "
           "Example.")


@requires_monolith
class ScreenLookRecordedOnceTests(_Base):
    def setUp(self):
        super().setUp()
        import core.actions as A
        self.A = A
        prev = self.bc._begin_turn_grounding(
            "Jarvis, open the console and make a project")
        self.addCleanup(self.bc._end_turn_grounding, prev)

    def _feeding(self, name, answer):
        """An action that feeds its own look, like core.actions'
        _act_see_screen (_record_look), _act_find_on_screen and
        _click_on_screen."""
        A = self.A

        def fn(arg=""):
            text = answer(arg) if callable(answer) else answer
            A._note_screen_look(name, text)
            return text
        self._actions[name] = fn

    def _run(self, token):
        return self._quiet(self.bc.parse_and_run_actions, token)

    def test_one_see_screen_is_one_entry(self):
        self._feeding("see_screen", SIGN_IN)
        self._run("[ACTION: see_screen, what is on my screen]")
        self.assertEqual(self.bc._turn_screen_texts(), [SIGN_IN])

    def test_one_find_on_screen_is_one_entry(self):
        found = "found 'Sign in' on the middle monitor in 'Example' [uia]"
        self._feeding("find_on_screen", found)
        self._run("[ACTION: find_on_screen, the sign in button]")
        self.assertEqual(self.bc._turn_screen_texts(), [found])

    def test_a_look_that_does_not_feed_itself_is_still_recorded(self):
        # main's own path (a vision answer, local_describe_screen, ...).
        self._stub("see_screen", "The screen shows an inbox, sir.")
        self._run("[ACTION: see_screen, what is on my screen]")
        self.assertEqual(self.bc._turn_screen_texts(),
                         ["The screen shows an inbox, sir."])

    def test_a_click_mark_does_not_hide_the_next_look(self):
        # A click fed outside the parse loop leaves its mark; the next look
        # (another handler) is recorded all the same.
        self.A._note_screen_look("click_on_screen",
                                 "I clicked Create project, sir.")
        self._stub("see_screen", "The screen shows an inbox, sir.")
        self._run("[ACTION: see_screen, what is on my screen]")
        self.assertEqual(self.bc._turn_screen_texts(),
                         ["I clicked Create project, sir.",
                          "The screen shows an inbox, sir."])

    def test_a_sign_in_page_survives_three_more_screen_actions(self):
        from core import auth_guard as ag
        self._feeding("see_screen", SIGN_IN)
        self._feeding("find_on_screen",
                      lambda q: f"found '{q}' on the middle monitor in "
                                f"'Example' [uia]")
        self._feeding("click_on_screen", lambda q: f"I clicked {q}, sir.")
        self._run("[ACTION: see_screen, what is up]")
        self._run("[ACTION: find_on_screen, Create project]")
        self._run("[ACTION: click_on_screen, Create project]")
        self._run("[ACTION: find_on_screen, Name field]")
        texts = self.bc._turn_screen_texts()
        self.assertEqual(len(texts), 4)
        self.assertEqual(texts[0], SIGN_IN)
        self.assertTrue(ag.auth_page([], [], texts))


@requires_monolith
class ScreenQuestionOneTurnTests(_Base):
    DELETE = {"label": "Delete account", "rect": [100, 100, 80, 30],
              "monitor": "middle", "hwnd": 1}

    def setUp(self):
        super().setUp()
        from core import grounded_click as G
        self.G = G
        G.reset_state()
        self.addCleanup(G.reset_state)
        self._p(self.bc, "_prev_owner_turn_at", [0.0])
        self._p(self.bc, "_last_owner_turn_at", [0.0])
        self._p(self.bc, "_pending_confirmation", [])
        self.clicks = []
        self._actions["click_on_screen"] = (
            lambda a="": self.clicks.append(a) or "Pressed, sir.")
        self._stub("undo_click", "Gone back, sir.")

    def _turn(self, at):
        """The two stamps the main loop's _note_owner_turn moves."""
        bc = self.bc
        bc._prev_owner_turn_at[0] = bc._last_owner_turn_at[0]
        bc._last_owner_turn_at[0] = float(at)

    def _ask(self, options=None, kind="confirm"):
        """Owner turn N, during which JARVIS asks. Returns when it asked
        (core.grounded_click's own stamp)."""
        self._turn(time.monotonic() - 5.0)
        self.G._set_pending(list(options or [self.DELETE]),
                            "click that red button", "that red button",
                            kind=kind, allow_yes=(kind != "which"))
        p = self.G.pending_choice()
        return float(p.get("mono") or time.monotonic())

    def _route(self, text):
        return self._quiet(self.bc._utterance_route_reply, text)

    def test_the_next_turns_yes_still_answers_it(self):
        asked = self._ask()
        self._turn(asked + 1.0)
        self.assertEqual(self._route("yes"),
                         "[ACTION: click_on_screen, pick:1]")

    def test_the_next_turns_pick_still_answers_a_which_one(self):
        two = [dict(self.DELETE, label="Video one"),
               dict(self.DELETE, label="Video two", rect=[300, 100, 80, 30])]
        asked = self._ask(two, kind="which")
        self._turn(asked + 1.0)
        self.assertEqual(self._route("the second one"),
                         "[ACTION: click_on_screen, pick:2]")

    def test_a_turn_that_does_not_answer_closes_it(self):
        for said in ("no", "never mind", "play some jazz", "what time is it",
                     "stop"):
            with self.subTest(said=said):
                asked = self._ask()
                self._turn(asked + 1.0)
                self.assertIsNone(self._route(said))
                self.assertIsNone(self.G.pending_choice())
                # ... so the next turn's "yes" answers something else.
                self._turn(asked + 2.0)
                self.assertIsNone(self._route("yes"))

    def test_a_yes_two_turns_later_is_not_an_answer(self):
        asked = self._ask()
        self._turn(asked + 1.0)        # a turn handled before the routes
        self._turn(asked + 2.0)
        self.assertIsNone(self._route("yes"))
        self.assertIsNone(self.G.pending_choice())

    def test_a_declined_press_is_not_pressed_by_a_later_yes(self):
        # The review's live shape, through the real dispatch.
        self._stub("lights_on", "The lights are on, sir.")
        asked = self._ask()
        self._turn(asked + 1.0)
        self._dispatch("no", "[intent:confirmation] Very good, sir.")
        self._turn(asked + 2.0)
        self._dispatch("it's dark in here",
                       "[intent:helpful] Shall I turn the lights on, sir?")
        self._turn(asked + 3.0)
        self._dispatch("yes", "[intent:confirmation] [ACTION: lights_on]")
        self.assertEqual(self.clicks, [])
        self.assertEqual(self.calls["lights_on"], [""])


@requires_monolith
class ReplayRefusesTheNewForgetsTests(_Base):
    def test_forget_screen_and_forget_voice_line_are_not_replayed(self):
        import core.actions as A
        bc = self.bc
        self._p(bc, "_action_history", collections.deque(maxlen=5))
        for name in ("forget_screen", "forget_voice_line"):
            with self.subTest(name=name):
                ran = []
                self._actions[name] = (
                    lambda a="", ran=ran: ran.append(a) or "Done, sir.")
                bc._action_history.clear()
                bc._action_history.append({"action": name, "arg": "60m",
                                           "result": "Done, sir.",
                                           "at": time.time()})
                out = self._quiet(A._act_replay_last_action)
                self.assertIn("refusing to replay destructive action", out)
                self.assertEqual(ran, [])
                self.assertTrue(bc._autocorrect_protected(name))


@requires_monolith
class CloneSeedingGuestModeTests(_Base):
    def test_no_seed_render_in_guest_mode(self):
        import time as _t
        bc = self.bc
        clock = types.SimpleNamespace(**{k: getattr(_t, k) for k in dir(_t)
                                         if not k.startswith("_")})
        clock.monotonic = lambda: 1_000_000.0
        clock.time = lambda: 2_000_000_000.0
        self._p(bc, "time", clock)
        for name, value in (("_turn_in_progress", [False]),
                            ("_utterance_in_progress", [False]),
                            ("_tts_playback_active", [False]),
                            ("_filler_on_device", [False]),
                            ("_main_loop_started_at", [1.0]),
                            ("_last_convo_activity", [0.0]),
                            ("_last_owner_turn_at", [0.0]),
                            ("_last_owner_voice_at", [0.0])):
            self._p(bc, name, value)
        self._p(bc, "last_speech_time", 0.0, create=True)
        self._p(bc, "_SPEAK_LOCK", threading.Lock())
        self._p(bc, "_tts_engine_kind", return_value="clone")
        game = mock.patch.dict("sys.modules", {"skill_game_mode": None})
        game.start()
        self.addCleanup(game.stop)
        self.assertIsNone(bc._clone_seed_gate())
        GM.set_on(True)
        self.addCleanup(GM.set_on, False)
        self.assertEqual(bc._clone_seed_gate(), "guest mode")
