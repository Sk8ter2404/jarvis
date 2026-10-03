"""Cross-branch interactions of the v2.0.177 release candidate (2026-10-02).

Seven reviewed branches meet in this release, and each one's own tests ran
without the others' code. These drive the REAL dispatch where two of them
meet - every model reply canned, every action stubbed or faked, pygetwindow
a fake module - so nothing real is opened, closed, captured or spoken, and
no LLM, audio or network is touched. Paraphrased fixtures, no owner words.

  A  offer-yes x "close that and open X" (streaming S1): a yes to JARVIS's
     own offer "Shall I close that and open <service>?" is the same request
     as the owner saying it, so the close is still the close of what JARVIS
     opened last (core.opened_ledger) - never a window the brain guessed at,
     and never dropped. Before the fix the close-then-open enforcer read only
     the owner's words ("Jarvis, yes."), so the brain's open-only reply lost
     the close (the live S1 failure) and a guessed close_window ran.
  B  offer-yes x a streaming TERMINAL line: an offer the brain made before a
     sign-in wall stopped the turn is not what a later yes answers.
  C  the streaming terminal lines (sign-in wall, no verified link) x the
     turn checker in 'on' mode x live-fixes F1 x instant actions 'on': the
     line is spoken once, never withheld as an unverified claim, never
     followed by a follow-up round and never retried on Claude.
  D  guest mode x the learn worker's deferred batch (R5 background slot):
     guest mode is judged when the batch is WRITTEN, not when it was queued.
  E  "diagnostic status" (a fresh sweep) x paused diagnostic daemons: the
     answer is spoken verbatim, once, and still says they are paused.

    python -m unittest tests.monolith.test_monolith_rel177_interactions
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base
from tests.monolith.test_monolith_offer_yes import QUIP, _OfferBase
from tests.monolith.test_monolith_turn_check import (_ASK, _CLAIM,
                                                     _TurnCheckBase)

SUFFIX = " - Google Chrome"
SIGN_IN_LINE = ("HBO Max isn't signed in on this browser, sir - sign in once "
                "and I can take it from there.")
NO_LINK_LINE = ("I don't have a verified search link for Disney+, sir, so "
                "I've opened its home page - search for Bluey there.")
MAX_HOME = "https://play.hbomax.com"


def _terminal(line: str) -> str:
    from core.failure_markers import TERMINAL_FAILURE_PREFIX
    return TERMINAL_FAILURE_PREFIX + line


class _Win:
    def __init__(self, title, hwnd):
        self.title = title
        self._hWnd = hwnd
        self.left, self.top, self.width, self.height = 0, 0, 2560, 1400
        self.closed = False

    def close(self):
        self.closed = True


# ════════════════════════════════════════════════════════════════════════════
#  A - a yes to "Shall I close that and open X?" closes only what JARVIS opened
# ════════════════════════════════════════════════════════════════════════════
CLOSE_OFFER = "Shall I close that and open HBO Max instead?"
CLOSE_OFFER_REPLY = ("[intent:dry_wit] Middling reviews, I'm afraid, sir. "
                     + CLOSE_OFFER)


@requires_monolith
class OfferYesCloseThenOpenTests(_OfferBase):
    def setUp(self):
        super().setUp()
        from core import opened_ledger
        self.ledger = opened_ledger
        opened_ledger.reset()
        self.addCleanup(opened_ledger.reset)
        # His own stream, and the results page JARVIS put up a moment ago.
        self.his = _Win("His Stream" + SUFFIX, 0x100)
        self.mine = _Win("Some Show - YouTube" + SUFFIX, 0x200)
        self.windows = [self.his, self.mine]
        fake = types.SimpleNamespace(getAllWindows=lambda: [
            w for w in self.windows if not w.closed])
        p = mock.patch.dict(sys.modules, {"pygetwindow": fake})
        p.start()
        self.addCleanup(p.stop)
        self.ledger.note_opened(
            "open_on_monitor",
            "https://www.youtube.com/results?search_query=some+show",
            hwnd=0x200, kind="window", monitor="middle")
        self.order: list = []
        real_close = self._actions["close_last_opened"]

        def _close(arg=""):
            self.order.append("close_last_opened")
            return real_close(arg)
        self._actions["close_last_opened"] = _close
        for name in ("open_url", "play_streaming"):
            self._actions[name] = (lambda n: lambda arg="": (
                self.order.append(f"{n}:{arg}") or f"opened {arg}"))(name)
        self._stub("close_window", "closed: His Stream")

    def _offer_then_yes(self, act, offer_reply=CLOSE_OFFER_REPLY,
                        offer=CLOSE_OFFER):
        self.brain = self.obedient(offer_reply=offer_reply, act=act,
                                   offer=offer)
        printed = self._turn("Jarvis, is that show any good?")
        self.assertIn("[offer-yes] open offer:", printed)
        self.assertEqual(self.order, [])
        printed = self._turn("Jarvis, yes.")
        self.assertIn("[offer-yes] sir said yes to", printed)
        return printed

    def test_an_open_only_reply_still_closes_what_jarvis_opened_first(self):
        # The live S1 shape, reached through the yes: the brain wrote ONE
        # token, the open.
        printed = self._offer_then_yes(
            f"[intent:confirmation] Very good, sir. [ACTION: open_url, {MAX_HOME}]")
        self.assertEqual(self.order,
                         ["close_last_opened", f"open_url:{MAX_HOME}"])
        self.assertTrue(self.mine.closed)
        self.assertFalse(self.his.closed)
        self.assertIsNone(self.ledger.last_opened())
        self.assertIn("closing what I opened last before the open", printed)

    def test_a_guessed_close_is_replaced_by_the_window_jarvis_opened(self):
        self._offer_then_yes(
            "Right away, sir. [ACTION: close_window, His Stream] "
            f"[ACTION: open_url, {MAX_HOME}]")
        self.assertEqual(self.calls["close_window"], [])
        self.assertFalse(self.his.closed)
        self.assertTrue(self.mine.closed)
        self.assertEqual(self.order,
                         ["close_last_opened", f"open_url:{MAX_HOME}"])

    def test_a_close_only_reply_gets_the_open_as_the_dropped_step(self):
        self.followups = [f"Of course, sir. [ACTION: open_url, {MAX_HOME}]"]
        self._offer_then_yes("Done, sir. [ACTION: close_last_opened]")
        self.assertTrue(self.gfr.called)
        self.assertIn("_dropped_step", self._followup_names(0))
        self.assertEqual(self.order,
                         ["close_last_opened", f"open_url:{MAX_HOME}"])
        self.assertFalse(self.his.closed)

    def test_a_named_window_in_the_offer_is_closed_as_named(self):
        # He said yes to closing a window JARVIS named: that close stands.
        offer = "Shall I close His Stream and open HBO Max instead?"
        self._offer_then_yes(
            "Right away, sir. [ACTION: close_window, His Stream] "
            f"[ACTION: open_url, {MAX_HOME}]",
            offer_reply="It is buffering again, sir. " + offer, offer=offer)
        self.assertEqual(self.calls["close_window"], ["His Stream"])
        self.assertEqual(self.order, [f"open_url:{MAX_HOME}"])
        self.assertFalse(self.mine.closed)

    def test_an_offer_without_a_close_closes_nothing(self):
        offer = "Shall I open HBO Max instead?"
        self._offer_then_yes(
            f"Of course, sir. [ACTION: open_url, {MAX_HOME}]",
            offer_reply="That one is not on YouTube, sir. " + offer,
            offer=offer)
        self.assertEqual(self.order, [f"open_url:{MAX_HOME}"])
        self.assertFalse(self.mine.closed)

    def test_the_offer_rides_only_its_own_turn(self):
        # The next owner turn is his own words again: an open-only reply to a
        # plain "open" closes nothing.
        self._offer_then_yes(
            f"Very good, sir. [ACTION: open_url, {MAX_HOME}]")
        self.ledger.note_opened("open_url", MAX_HOME, hwnd=0x300, kind="tab",
                                monitor="middle")
        self.brain = lambda p: "Of course, sir. [ACTION: open_url, netflix.com]"
        self._turn("Jarvis, open Netflix.")
        self.assertEqual(self.order[-1], "open_url:netflix.com")
        self.assertEqual(self.order.count("close_last_opened"), 1)


# ════════════════════════════════════════════════════════════════════════════
#  B - an offer JARVIS spoke before a sign-in wall stopped the turn
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class OfferBeforeTerminalLineTests(_OfferBase):
    def test_the_wall_line_is_the_last_word_and_a_yes_answers_no_offer(self):
        self._stub("play_streaming", _terminal(SIGN_IN_LINE))
        self._stub("smart_home_control", "lights dimmed")
        state = {"n": 0}

        def brain(prompt):
            state["n"] += 1
            if state["n"] == 1:
                return ("Right away, sir. [ACTION: play_streaming, max|Some "
                        "Show] Shall I also dim the lights?")
            if "OFFER ACCEPTED" in prompt:
                return "[ACTION: smart_home_control, dim the lights]"
            return QUIP
        self.brain = brain
        self._turn("Jarvis, get Some Show going for me.")
        self.assertEqual(self.spoken.count(SIGN_IN_LINE), 1, self.spoken)
        self.assertEqual(self.gfr.call_count, 0)
        printed = self._turn("Jarvis, yes.")
        self.assertIn("said something else after the offer", printed)
        self.assertNotIn("OFFER ACCEPTED", self.prompts[-1])
        self.assertEqual(self.calls["smart_home_control"], [])


# ════════════════════════════════════════════════════════════════════════════
#  C - a terminal line: spoken once, never withheld, never retried
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class TerminalLineTurnCheckTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.assertIsNotNone(bc._turn_checker, "no turn checker to test")
        self._p(bc, "_turn_check_mode", return_value="on")
        self._p(bc, "_chat_cloud_allowed", return_value=True)
        self._p(bc, "_turn_check_escalate_target",
                return_value=("claude-test", "claude-test"))
        self.escalate = self._p(bc, "_turn_check_escalate",
                                return_value="spoken")
        self.rows: list = []
        self._p(bc, "_turn_check_write_row",
                side_effect=lambda row, fn=None: self.rows.append(row) or True)
        self._p(bc, "_instant_actions_mode", return_value="on")
        self.instant_rows = self._p(bc._instant_actions, "append_row")
        self._stub("play_streaming", _terminal(SIGN_IN_LINE))

    def _run(self, user_text, reply, followups=()):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self._p(self.bc, "get_response_with_animation",
                    return_value=reply)
            self.gfr = self._p(self.bc, "get_followup_response",
                               side_effect=list(followups) + [None] * 8)
            self.bc._run_llm_dispatch(user_text)
        return out.getvalue()

    def _assert_once_and_settled(self, line, printed):
        self.assertEqual(self.spoken.count(line), 1, self.spoken)
        self.assertNotIn("_unverified_claim", printed)
        self.assertNotIn("hallucinated execution", printed)
        self.escalate.assert_not_called()
        self.assertTrue(self.rows, "the turn was never checked")
        self.assertEqual(self.rows[-1]["kind"], "ok", self.rows[-1])

    def test_the_brains_play_hits_a_sign_in_wall(self):
        printed = self._run("Jarvis, get Some Show going on the telly.",
                            "[ACTION: play_streaming, max|Some Show]")
        self._assert_once_and_settled(SIGN_IN_LINE, printed)
        self.gfr.assert_not_called()

    def test_a_reply_that_claimed_the_play_does_not_withhold_the_line(self):
        printed = self._run(
            "Jarvis, get Some Show going on the telly.",
            "Playing Some Show on HBO Max now, sir. "
            "[ACTION: play_streaming, max|Some Show]")
        self._assert_once_and_settled(SIGN_IN_LINE, printed)
        self.gfr.assert_not_called()

    def test_the_no_verified_link_line_is_said_once_and_ends_the_turn(self):
        self._stub("play_streaming", _terminal(NO_LINK_LINE))
        printed = self._run("Jarvis, get Bluey going for the kids.",
                            "[ACTION: play_streaming, disney_plus|Bluey]")
        self._assert_once_and_settled(NO_LINK_LINE, printed)
        self.gfr.assert_not_called()

    def test_a_wall_in_the_follow_up_round_stops_the_chain_there(self):
        self._stub("see_screen", "a streaming home page with a search box")
        printed = self._run(
            "Jarvis, what's on the screen, and get Some Show going.",
            "One moment, sir. [ACTION: see_screen]",
            ["[ACTION: play_streaming, max|Some Show]",
             "[ACTION: click, the first result]"])
        self._assert_once_and_settled(SIGN_IN_LINE, printed)
        self.assertEqual(self.gfr.call_count, 1)
        self.assertNotIn("click", self.calls)

    def test_a_routed_play_never_touches_the_instant_layer(self):
        printed = self._run("play Some Show on HBO Max", "SHOULD NOT BE USED")
        self.assertEqual(self.spoken, [SIGN_IN_LINE])
        self.assertEqual(self.calls["play_streaming"], ["max|Some Show"])
        self.bc.get_response_with_animation.assert_not_called()
        self.instant_rows.assert_not_called()
        self.escalate.assert_not_called()
        self.assertNotIn("_unverified_claim", printed)

    def test_the_harness_can_escalate_a_real_failure(self):
        # Mutation guard for the class: the same setup DOES retry a turn
        # that failed (a made-up action), so "not escalated" above is the
        # checker's verdict, not a dead switch.
        self._run("Jarvis, get Some Show going on the telly.",
                  "[ACTION: frobnicate_the_widget]")
        self.escalate.assert_called_once()
        self.assertEqual(self.rows[-1]["kind"], "made_up_action")


@requires_monolith
class TerminalLineInTheTurnCheckRetryTests(_TurnCheckBase):
    """TURN_CHECK_MODE 'on': the local turn failed and the Claude retry ran
    play_streaming into a sign-in wall. The main follow-up loop ends a chain
    on a terminal line; the retry's one read-back must too - before the fix
    it re-worded the line it had just said word for word."""

    READ_BACK = "It seems HBO Max wants you to sign in first, sir."

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "TURN_CHECK_MODE", "on")
        self.plays: list = []
        bc.ACTIONS["play_streaming"] = lambda a="": (
            self.plays.append(a) or _terminal(SIGN_IN_LINE))
        self.oneshot.return_value = "[ACTION: play_streaming, max|Some Show]"
        # Round 1 of the local chain (the claim check's self-correction)
        # gets nothing; a read-back of the retry would get READ_BACK.
        self.followup.side_effect = ["", self.READ_BACK]

    def test_the_wall_line_is_said_once_and_not_read_back(self):
        out = self._run(_CLAIM, text=_ASK)
        self.assertEqual(self.oneshot.call_count, 1, out)   # it did retry
        self.assertEqual(self.plays, ["max|Some Show"])
        self.assertEqual(self.spoken.count(SIGN_IN_LINE), 1, self.spoken)
        self.assertNotIn(self.READ_BACK, self.spoken)
        self.assertEqual(self.followup.call_count, 1)
        # ... and, as in the main loop, it is the last thing JARVIS said.
        self.assertEqual(self.assistant_msgs()[-1], SIGN_IN_LINE)


# ════════════════════════════════════════════════════════════════════════════
#  D - guest mode is judged when a deferred learn batch is WRITTEN
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class GuestModeDeferredLearnTests(MonolithGlobalsTestCase):
    """The learn worker takes its batch inside the background slot - after
    R5's wait for the owner to go quiet - so a batch queued before guest mode
    can be extracted after it came on, and the guests' own turns must never
    be learned once it goes off."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.saved: list = []
        self.mirrored: list = []
        for name, value in (
                ("load_memory", bc._empty_memory),
                ("save_memory", lambda mem: self.saved.append(mem)),
                ("_ltm_learn_facts",
                 lambda f, p=None, provenance=None: self.mirrored.append(f)),
                ("_ltm_enabled", lambda: True),
                ("LEARN_ONLY_FROM_OWNER", False),
                ("LEARN_EVERY_TURN", True),
                ("_dialogue_gate_active", lambda: False),
                ("_rebuild_after_learning", lambda: None),
                ("_llm_quick_goes_local", lambda: False)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.llm = mock.Mock(return_value='{"new_facts": ["User owns a '
                                          'canoe"], "new_projects": [], '
                                          '"topic": ""}')
        p = mock.patch.object(bc, "_llm_quick", self.llm)
        p.start()
        self.addCleanup(p.stop)
        saved_q = list(bc._learn_pending)
        bc._learn_pending.clear()
        self.addCleanup(lambda: (bc._learn_pending.clear(),
                                 bc._learn_pending.extend(saved_q)))
        # Queue, never start a thread: the worker body runs here.
        p = mock.patch.object(bc, "_learn_enqueue",
                              side_effect=bc._learn_pending.append)
        p.start()
        self.addCleanup(p.stop)

    def _drain(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._learn_worker()

    def test_a_batch_queued_before_guest_mode_writes_nothing_during_it(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I bought a canoe", "A fine vessel, sir.", {})
        self.assertEqual(len(bc._learn_pending), 1)
        bc._guest_mode.set_on(True)          # the guests arrive mid-wait
        self._drain()
        self.assertEqual(self.saved, [])
        self.assertEqual(self.mirrored, [])

    def test_a_guests_turn_is_never_learned_after_they_leave(self):
        bc = self.bc
        bc._guest_mode.set_on(True)
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("my name is Orlo and I sail", "Welcome.", {})
        bc._guest_mode.set_on(False)          # gone before the worker ran
        self._drain()
        self.llm.assert_not_called()
        self.assertEqual(self.saved, [])
        # ... while the owner's next turn is learned as usual.
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I bought a canoe", "A fine vessel, sir.", {})
        self._drain()
        self.llm.assert_called_once()
        self.assertEqual(self.mirrored, [["User owns a canoe"]])


# ════════════════════════════════════════════════════════════════════════════
#  E - "diagnostic status" while the background diagnostics are paused
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class FreshDiagnosticStatusPausedTests(_Base):
    SWEEP = ("Sir, one issue — webcam. Severity breakdown: 1 medium. "
             "(2.0s sweep.)")

    def setUp(self):
        super().setUp()
        from core import diagnostic_daemons as dd
        self.dd = dd
        self._p(dd, "_read_state", return_value={"paused": True})
        self.sweep = self._stub("run_diagnostic", self.SWEEP)
        self._actions["diagnostic_status"] = self.bc._fresh_diagnostic_status

    def test_the_fresh_sweep_is_spoken_once_and_says_paused(self):
        out = self._dispatch("Jarvis, diagnostic status.",
                             "[ACTION: diagnostic_status]")
        want = self.SWEEP + " The background diagnostics are still paused."
        self.assertEqual(self.spoken, [want])
        self.assertEqual(self.calls["run_diagnostic"], [""])
        self.gfr.assert_not_called()
        self.assertNotIn("_unverified_claim", out)

    def test_no_sweep_reads_the_counters_and_says_paused_once(self):
        self._actions.pop("run_diagnostic")
        for name in self.bc._SELF_CHECK_ACTIONS:
            self._actions.pop(name, None)
        self._dispatch("Jarvis, diagnostic status.",
                       "[ACTION: diagnostic_status]")
        self.assertEqual(len(self.spoken), 1, self.spoken)
        said = self.spoken[0]
        self.assertTrue(said.startswith("The self-check is not loaded, sir"),
                        said)
        self.assertEqual(said.count("paused"), 1, said)
        self.gfr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
