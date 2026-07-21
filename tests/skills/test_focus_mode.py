"""Logic tests for skills/focus_mode.py — the do-not-disturb control surface.

The skill reaches the monolith (bobert_companion) through small NEVER-RAISES
helpers to flip the focus flag and read the "missed" buffer. Here we inject a
FAITHFUL FAKE bobert_companion into sys.modules that reimplements exactly that
helper surface (focus_mode_active / set_focus_mode / focus_missed_count /
_build_focus_recap + the _focus_until cell), so the skill's control flow — arm
on, recap-and-clear on off, status-without-clear — is exercised end to end
without importing the ~1M-line monolith.

Covers: actions registered; on/off toggle the shared flag; duration parsing
(spoken + numeric + bare + indefinite); whats_missed does NOT clear; off returns
a recap AND clears; the auto-resume Timer is armed for a timed block and not for
an indefinite one; EVERY engage alias (focus_mode / focus_mode_on /
do_not_disturb / quiet_mode) chains dnd_focus_mode's prior engage and EVERY
disengage alias (end_focus_mode / focus_mode_off / resume) chains its prior
teardown — exactly once per call, enumerated from the registered actions dict
so a future un-chained alias fails here rather than in a grep.

stdlib unittest + unittest.mock only. Headless: the resume Timer is mocked so no
real thread sleeps, and load_skill_isolated neuters Thread.start at import.
"""
from __future__ import annotations

import sys
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── a faithful fake of the monolith's focus helper surface ──────────────────

def _make_fake_bc():
    """Build a stand-in `bobert_companion` module exposing only the focus API
    the skill calls. Mirrors the real monolith semantics: the missed buffer is
    a plain list of (message, source, ts); focus_mode_active() self-resumes a
    lapsed timed block; _build_focus_recap summarises + optionally clears."""
    bc = types.ModuleType("bobert_companion")
    bc._focus_mode = [False]
    bc._focus_until = [0.0]
    bc._focus_missed_buffer = []
    bc._proactive_calls = []   # test spy: what got enqueued once un-gated

    def focus_mode_active():
        if not bc._focus_mode[0]:
            return False
        until = bc._focus_until[0]
        if until and time.time() >= until:
            bc._focus_mode[0] = False
            bc._focus_until[0] = 0.0
            recap = _build_focus_recap(clear=True, prefix="Focus time's up, sir")
            if recap:
                proactive_announce(recap, source="focus_mode")
            return False
        return True

    def set_focus_mode(on, *, until=0.0):
        bc._focus_mode[0] = bool(on)
        bc._focus_until[0] = float(until) if on else 0.0

    def focus_missed_count():
        return len(bc._focus_missed_buffer)

    def focus_missed_snapshot():
        return list(bc._focus_missed_buffer)

    def clear_focus_missed():
        bc._focus_missed_buffer.clear()

    def _build_focus_recap(*, clear, prefix="While you were focused, sir"):
        items = list(bc._focus_missed_buffer)
        if clear:
            bc._focus_missed_buffer.clear()
        n = len(items)
        if n == 0:
            return f"{prefix} — nothing came up while you were focused."
        frags = [(m or "").strip().rstrip(".") for m, _s, _t in items[:3]]
        noun = "thing" if n == 1 else "things"
        listed = (frags[0] if len(frags) == 1
                  else ", ".join(frags[:-1]) + ", and " + frags[-1])
        rem = n - len(frags)
        if rem > 0:
            listed += f", plus {rem} more"
        return f"{prefix}: {n} {noun} — {listed}."

    def proactive_announce(message, source="skill", *, mood=None, volume_scale=1.0):
        # Mimic the real gate: while focused, HOLD into the missed buffer and
        # return True; otherwise record the enqueue (the spy the auto-resume
        # test inspects).
        if focus_mode_active():
            bc._focus_missed_buffer.append((message, source, time.time()))
            return True
        bc._proactive_calls.append((message, source))
        return True

    bc.focus_mode_active = focus_mode_active
    bc.set_focus_mode = set_focus_mode
    bc.focus_missed_count = focus_missed_count
    bc.focus_missed_snapshot = focus_missed_snapshot
    bc.clear_focus_missed = clear_focus_missed
    bc._build_focus_recap = _build_focus_recap
    bc.proactive_announce = proactive_announce
    return bc


class _FocusSkillBase(unittest.TestCase):
    def setUp(self):
        # Inject the fake monolith under BOTH names the skill's _bc() probes.
        self.fake_bc = _make_fake_bc()
        self._saved_main = sys.modules.get("__main__")
        self._saved_bc = sys.modules.get("bobert_companion")
        sys.modules["bobert_companion"] = self.fake_bc
        # Remove __main__ so _bc() falls through to bobert_companion (the real
        # test-runner __main__ has no focus helpers).
        sys.modules.pop("__main__", None)
        self.addCleanup(self._restore_modules)

        self.mod, self.actions = load_skill_isolated("focus_mode")
        # Neuter the real resume Timer in every test by default — deterministic,
        # no thread sleeps. Individual tests re-patch to assert it was armed.
        self._arm_patch = mock.patch.object(self.mod, "_arm_resume_timer")
        self.arm_mock = self._arm_patch.start()
        self.addCleanup(self._arm_patch.stop)

    def _restore_modules(self):
        if self._saved_main is not None:
            sys.modules["__main__"] = self._saved_main
        if self._saved_bc is not None:
            sys.modules["bobert_companion"] = self._saved_bc
        else:
            sys.modules.pop("bobert_companion", None)


# ─── registration ────────────────────────────────────────────────────────────

class RegistrationTests(_FocusSkillBase):
    def test_all_actions_registered(self):
        for name in ("focus_mode_on", "do_not_disturb", "quiet_mode",
                     "focus_mode_off", "resume", "whats_missed",
                     "end_focus_mode", "focus_mode_status"):
            self.assertIn(name, self.actions)

    def test_aliases_share_the_on_handler(self):
        self.assertIs(self.actions["focus_mode_on"], self.actions["do_not_disturb"])
        self.assertIs(self.actions["focus_mode_on"], self.actions["quiet_mode"])

    def test_actions_are_callable_and_return_str(self):
        for name in ("focus_mode_on", "focus_mode_off", "whats_missed"):
            out = self.actions[name]("")
            self.assertIsInstance(out, str)
            self.assertTrue(out)


# ─── on / off toggle the shared flag ─────────────────────────────────────────

class ToggleTests(_FocusSkillBase):
    def test_on_engages_focus(self):
        self.assertFalse(self.fake_bc._focus_mode[0])
        msg = self.actions["focus_mode_on"]("")
        self.assertTrue(self.fake_bc._focus_mode[0])
        self.assertIn("focus mode on", msg.lower())

    def test_off_disengages_focus(self):
        self.actions["focus_mode_on"]("")
        self.assertTrue(self.fake_bc._focus_mode[0])
        self.actions["focus_mode_off"]("")
        self.assertFalse(self.fake_bc._focus_mode[0])

    def test_indefinite_on_sets_no_deadline(self):
        self.actions["focus_mode_on"]("")   # no duration
        self.assertEqual(self.fake_bc._focus_until[0], 0.0)
        # No auto-resume timer for an indefinite block.
        self.arm_mock.assert_not_called()

    def test_do_not_disturb_and_quiet_mode_engage(self):
        self.actions["do_not_disturb"]("")
        self.assertTrue(self.fake_bc._focus_mode[0])
        self.actions["focus_mode_off"]("")
        self.actions["quiet_mode"]("")
        self.assertTrue(self.fake_bc._focus_mode[0])


# ─── duration parsing + timed engage ─────────────────────────────────────────

class DurationTests(_FocusSkillBase):
    def test_parse_variants(self):
        p = self.mod._parse_duration_seconds
        self.assertEqual(p("30 minutes"), 1800)
        self.assertEqual(p("45m"), 2700)
        self.assertEqual(p("1 hour 30 min"), 5400)
        self.assertEqual(p("2h"), 7200)
        self.assertEqual(p("an hour"), 3600)
        self.assertEqual(p("half an hour"), 1800)
        self.assertEqual(p("ninety minutes"), 5400)
        self.assertEqual(p("30"), 1800)          # bare → minutes
        self.assertIsNone(p(""))                 # blank → indefinite
        self.assertIsNone(p("please focus"))     # unparseable → indefinite

    def test_timed_on_sets_deadline_and_arms_timer(self):
        before = time.time()
        msg = self.actions["focus_mode_on"]("30 minutes")
        until = self.fake_bc._focus_until[0]
        # Deadline ~30 min out.
        self.assertGreater(until, before + 1700)
        self.assertLess(until, before + 1900)
        # Auto-resume timer armed with ~1800s.
        self.arm_mock.assert_called_once()
        armed_secs = self.arm_mock.call_args.args[0]
        self.assertAlmostEqual(armed_secs, 1800, delta=5)
        self.assertIn("30 minutes", msg)

    def test_on_confirmation_mentions_duration(self):
        msg = self.actions["focus_mode_on"]("an hour")
        self.assertIn("1 hour", msg)


# ─── whats_missed is a status query — never clears ───────────────────────────

class WhatsMissedTests(_FocusSkillBase):
    def test_reports_held_count_without_clearing(self):
        self.actions["focus_mode_on"]("")
        # Simulate three held announcements arriving via proactive_announce.
        for m in ("the print finished", "a weather alert", "a Teams message"):
            self.fake_bc.proactive_announce(m, source="skill")
        self.assertEqual(self.fake_bc.focus_missed_count(), 3)

        status = self.actions["whats_missed"]("")
        # It reports the count …
        self.assertIn("3", status)
        self.assertIn("focus mode is on", status.lower())
        # … and does NOT clear the buffer (still 3 held afterward).
        self.assertEqual(self.fake_bc.focus_missed_count(), 3)

    def test_status_off_when_not_engaged(self):
        out = self.actions["whats_missed"]("")
        self.assertIn("off", out.lower())

    def test_status_reports_remaining_time_for_timed_block(self):
        self.actions["focus_mode_on"]("30 minutes")
        out = self.actions["whats_missed"]("")
        self.assertIn("left", out.lower())


# ─── off returns a recap AND clears ──────────────────────────────────────────

class RecapTests(_FocusSkillBase):
    def _hold_three(self):
        self.actions["focus_mode_on"]("")
        for m in ("the print finished", "a weather alert", "a Teams message"):
            self.fake_bc.proactive_announce(m, source="skill")

    def test_off_returns_recap_of_missed(self):
        self._hold_three()
        recap = self.actions["focus_mode_off"]("")
        self.assertIn("3 things", recap)
        # The first few items are named in the recap.
        self.assertIn("the print finished", recap)

    def test_off_clears_the_buffer(self):
        self._hold_three()
        self.assertEqual(self.fake_bc.focus_missed_count(), 3)
        self.actions["focus_mode_off"]("")
        self.assertEqual(self.fake_bc.focus_missed_count(), 0)

    def test_resume_is_alias_for_off(self):
        self._hold_three()
        recap = self.actions["resume"]("")
        self.assertIn("3 things", recap)
        self.assertFalse(self.fake_bc._focus_mode[0])

    def test_off_when_not_active_says_so(self):
        out = self.actions["focus_mode_off"]("")
        self.assertIn("wasn't on", out.lower())

    def test_off_cancels_pending_resume_timer(self):
        # Real timer path: engage timed, then off must cancel it. We assert the
        # skill calls _cancel_resume_timer via a spy.
        with mock.patch.object(self.mod, "_cancel_resume_timer") as cancel:
            self.actions["focus_mode_off"]("")
            cancel.assert_called()


# ─── auto-resume recap fires once the block lapses ───────────────────────────

class AutoResumeTests(_FocusSkillBase):
    def test_expired_block_recaps_via_proactive_announce(self):
        # Engage a block that's already expired (until in the past) so the
        # fake's focus_mode_active() self-resumes and enqueues the recap.
        self.actions["focus_mode_on"]("")   # engage indefinitely first
        # Hold two items while active.
        for m in ("the print finished", "a weather alert"):
            self.fake_bc.proactive_announce(m, source="skill")
        # Now force a lapsed timed deadline.
        self.fake_bc._focus_until[0] = time.time() - 1
        self.fake_bc._focus_mode[0] = True
        # Reading the flag self-resumes and enqueues the recap normally.
        self.assertFalse(self.fake_bc.focus_mode_active())
        self.assertFalse(self.fake_bc._focus_mode[0])
        self.assertTrue(self.fake_bc._proactive_calls)
        recap_msg = self.fake_bc._proactive_calls[-1][0]
        self.assertIn("Focus time's up", recap_msg)


# ─── every alias chains the pre-existing dnd_focus_mode handlers ─────────────

class ChainingTests(unittest.TestCase):
    def setUp(self):
        self.fake_bc = _make_fake_bc()
        self._saved_main = sys.modules.get("__main__")
        self._saved_bc = sys.modules.get("bobert_companion")
        sys.modules["bobert_companion"] = self.fake_bc
        sys.modules.pop("__main__", None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved_main is not None:
            sys.modules["__main__"] = self._saved_main
        if self._saved_bc is not None:
            sys.modules["bobert_companion"] = self._saved_bc
        else:
            sys.modules.pop("bobert_companion", None)

    def test_end_focus_mode_calls_prior_handler_first(self):
        # Pre-seed the shared ACTIONS dict with a dnd_focus_mode-style handler,
        # then load our skill INTO the same dict so register() chains it.
        prior_called = []

        def prior_end(_=""):
            prior_called.append(True)
            return "Focus mode disengaged, sir."   # dnd_focus_mode's teardown line

        actions = {"end_focus_mode": prior_end, "focus_mode_status": lambda _="": "OS DND state."}
        mod, actions = load_skill_isolated("focus_mode", actions=actions)
        with mock.patch.object(mod, "_arm_resume_timer"):
            # Engage, hold one, then end via the chained handler.
            actions["focus_mode_on"]("")
            self.fake_bc.proactive_announce("a Teams message", source="skill")
            out = actions["end_focus_mode"]("")
        # Prior dnd teardown ran …
        self.assertTrue(prior_called)
        # … and OUR recap is returned.
        self.assertIn("1 thing", out)
        self.assertFalse(self.fake_bc._focus_mode[0])

    def test_focus_mode_status_composes_prior_and_ours(self):
        actions = {"focus_mode_status": lambda _="": "Windows DND is on, sir."}
        mod, actions = load_skill_isolated("focus_mode", actions=actions)
        with mock.patch.object(mod, "_arm_resume_timer"):
            actions["focus_mode_on"]("")
            out = actions["focus_mode_status"]("")
        self.assertIn("Windows DND is on", out)   # prior text preserved
        self.assertIn("focus mode is on", out.lower())  # ours appended

    def test_shared_focus_mode_engage_chains_prior_and_engages_our_state(self):
        # 2026-07-07 fix: the phrase "focus mode" routes to the SHARED `focus_mode`
        # (dnd's engage). Our override must chain dnd's OS-level engage AND turn on
        # OUR announcement-holding, so notifications are actually held and status
        # is consistent (the live bug was: dnd on, our gate off).
        prior_called = []

        def prior_engage(_=""):
            prior_called.append(True)
            return "Windows Focus Assist on, sir."     # dnd_focus_mode's engage line

        actions = {"focus_mode": prior_engage}
        mod, actions = load_skill_isolated("focus_mode", actions=actions)
        with mock.patch.object(mod, "_arm_resume_timer"):
            out = actions["focus_mode"]("")            # the phrase "focus mode"
            # dnd's OS engage ran …
            self.assertTrue(prior_called)
            # … and OUR gate is now ON, so a proactive announcement is HELD.
            self.assertTrue(self.fake_bc._focus_mode[0])
            self.fake_bc.proactive_announce("a weather alert", source="skill")
            self.assertEqual(self.fake_bc.focus_missed_count(), 1)
        self.assertIn("hold notifications", out.lower())  # our confirmation spoken

    # ── 2026-07-21 fix: disengage symmetry ───────────────────────────────────
    # "resume" / "focus_mode_off" used to skip dnd_focus_mode's teardown,
    # leaving Windows Focus Assist, Teams DND and the nudge suppressors stuck
    # on until dnd's own expiry thread fired (up to an hour later).

    def test_focus_mode_off_and_resume_chain_prior_end(self):
        end_calls = []

        def prior_end(a=""):
            end_calls.append(a)
            return "Focus mode disengaged, sir."   # dnd_focus_mode's teardown line

        actions = {"end_focus_mode": prior_end}
        mod, actions = load_skill_isolated("focus_mode", actions=actions)
        with mock.patch.object(mod, "_arm_resume_timer"):
            for alias in ("focus_mode_off", "resume"):
                before = len(end_calls)
                actions["focus_mode_on"]("")
                self.fake_bc.proactive_announce("a Teams message", source="skill")
                out = actions[alias]("")
                # dnd's OS-level teardown ran, exactly once …
                self.assertEqual(len(end_calls), before + 1, alias)
                # … and OUR recap is still returned, gate off.
                self.assertIn("1 thing", out, alias)
                self.assertFalse(self.fake_bc._focus_mode[0], alias)

    def test_all_engage_aliases_chain_prior_engage(self):
        # Mirror gap: "do not disturb" / "quiet mode" used to engage only OUR
        # gate, never dnd's Focus Assist / Teams DND.
        engage_calls = []

        def prior_engage(a=""):
            engage_calls.append(a)
            return "Windows Focus Assist on, sir."

        actions = {"focus_mode": prior_engage}
        mod, actions = load_skill_isolated("focus_mode", actions=actions)
        with mock.patch.object(mod, "_arm_resume_timer"):
            for alias in ("focus_mode_on", "do_not_disturb", "quiet_mode"):
                self.fake_bc.set_focus_mode(False)
                before = len(engage_calls)
                actions[alias]("")
                self.assertEqual(len(engage_calls), before + 1, alias)
                self.assertTrue(self.fake_bc._focus_mode[0], alias)

    def test_shared_names_chain_exactly_once_per_call(self):
        # Guards the naive-fix regression: chaining in BOTH a shared-name
        # wrapper and the base handler would fire dnd's teardown twice per
        # "end focus mode" and silently extend dnd's deadline on every
        # "focus mode" (its engage extends when already active).
        engage_calls, end_calls = [], []
        actions = {
            "focus_mode":     lambda a="": engage_calls.append(a) or "dnd on",
            "end_focus_mode": lambda a="": end_calls.append(a) or "dnd off",
        }
        mod, actions = load_skill_isolated("focus_mode", actions=actions)
        with mock.patch.object(mod, "_arm_resume_timer"):
            actions["focus_mode"]("")
            self.assertEqual(len(engage_calls), 1)
            self.assertEqual(len(end_calls), 0)
            actions["end_focus_mode"]("")
            self.assertEqual(len(end_calls), 1)
            self.assertEqual(len(engage_calls), 1)

    def test_every_registered_alias_chains_its_direction(self):
        # Stale-duplicate invariant: enumerate the names from the ACTIONS dict
        # itself and classify each handler by its observed effect on the gate —
        # any handler that ENGAGES must have fired dnd's prior engage exactly
        # once, any handler that DISENGAGES must have fired dnd's prior end
        # exactly once. A future alias registered without chaining (or chained
        # twice) fails here rather than in a grep.
        engage_calls, end_calls = [], []
        seeded = {
            "focus_mode":        lambda a="": engage_calls.append(a) or "dnd on",
            "end_focus_mode":    lambda a="": end_calls.append(a) or "dnd off",
            "focus_mode_status": lambda a="": "OS DND state.",
        }
        mod, actions = load_skill_isolated("focus_mode", actions=seeded)
        with mock.patch.object(mod, "_arm_resume_timer"):
            for name in sorted(actions):
                handler = actions[name]
                # From OFF: does this handler engage?
                self.fake_bc.set_focus_mode(False)
                e0, d0 = len(engage_calls), len(end_calls)
                handler("")
                if self.fake_bc._focus_mode[0]:
                    # Engage-direction alias.
                    self.assertEqual(len(engage_calls), e0 + 1,
                                     f"{name}: engage alias must chain dnd's "
                                     f"engage exactly once")
                    self.assertEqual(len(end_calls), d0,
                                     f"{name}: engage alias must not chain "
                                     f"dnd's teardown")
                    continue
                # From ON: does this handler disengage?
                self.fake_bc.set_focus_mode(True)
                e1, d1 = len(engage_calls), len(end_calls)
                handler("")
                if not self.fake_bc._focus_mode[0]:
                    # Disengage-direction alias.
                    self.assertEqual(len(end_calls), d1 + 1,
                                     f"{name}: disengage alias must chain "
                                     f"dnd's teardown exactly once")
                    self.assertEqual(len(engage_calls), e1,
                                     f"{name}: disengage alias must not chain "
                                     f"dnd's engage")
                # else: pure status alias — no chaining requirement.


if __name__ == "__main__":
    unittest.main()
