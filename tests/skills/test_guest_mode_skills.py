"""Guest mode outside the monolith (2026-10-02, core/guest_mode.py).

While visitors are here JARVIS answers normally but keeps nothing. The
monolith's learners are pinned in tests/monolith/test_monolith_fact_provenance;
these are the writers that live elsewhere:
  * skills/ambient_listen: the room / PC-audio / screen logs are not appended
    (they feed the fact extractor and Chappie's episodes);
  * skills/ambient_multimodal_extract: a pass skips its LLM call;
  * skills/chappie_consciousness: no episode or fact pass runs;
  * hud/jarvis_unified_hud: the HUD shows a GUEST MODE badge.

Every path is a temp dir or a mock; no daemon, LLM, mic or Qt window starts.
"""
from __future__ import annotations

import os
import tempfile
import time
import types
import unittest
from unittest import mock

from core import guest_mode
from tests._skill_harness import load_skill_isolated
from tests.skills.test_hud_brain_glow import _HUD_DIR, _load


class _GuestOn(unittest.TestCase):
    def setUp(self):
        self.addCleanup(guest_mode.set_on, False)


class AmbientListenLogTests(_GuestOn):
    def setUp(self):
        super().setUp()
        self.mod, _ = load_skill_isolated("ambient_listen")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "ambient_transcripts.jsonl")

    def test_nothing_is_appended_while_guests_are_here(self):
        guest_mode.set_on(True)
        self.mod._append_jsonl(self.path, {"ts": 1.0, "text": "visitor talk"})
        self.assertFalse(os.path.exists(self.path))

    def test_it_appends_again_once_they_have_gone(self):
        guest_mode.set_on(True)
        self.mod._append_jsonl(self.path, {"text": "dropped"})
        guest_mode.set_on(False)
        self.mod._append_jsonl(self.path, {"text": "kept"})
        with open(self.path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("kept", lines[0])


class AmbientExtractTests(_GuestOn):
    def setUp(self):
        super().setUp()
        self.mod, _ = load_skill_isolated("ambient_multimodal_extract")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mod._DATA_DIR = self.tmp.name
        self.mod._EXTRACT_JSONL = os.path.join(self.tmp.name, "x.jsonl")
        bc = mock.MagicMock()
        bc.AMBIENT_EXTRACT_BATCH = 50
        bc.AMBIENT_EXTRACT_INTERVAL_S = 300.0
        bc.LEARN_ONLY_FROM_OWNER = False
        self.bc = bc

    def _run(self):
        audio = [{"ts": time.time(), "source": "mic",
                  "text": "my friend Orlo is staying over", "window": ""}]
        with mock.patch.object(self.mod, "_get_bobert", return_value=self.bc), \
             mock.patch.object(self.mod, "_tail_jsonl",
                               side_effect=[audio, []]), \
             mock.patch.object(self.mod, "_llm_extract",
                               return_value={}) as llm:
            summary = self.mod._run_once()
        return summary, llm

    def test_a_pass_skips_its_llm_call_in_guest_mode(self):
        guest_mode.set_on(True)
        summary, llm = self._run()
        llm.assert_not_called()
        self.assertEqual(summary.get("skipped"), "guest mode")

    def test_off_it_runs_as_before(self):
        summary, llm = self._run()
        llm.assert_called_once()
        self.assertNotIn("skipped", summary)


class _Stop(BaseException):
    """Ends the infinite daemon loop (its own except Exception would swallow
    an ordinary exception)."""


class ChappieLoopTests(_GuestOn):
    def setUp(self):
        super().setUp()
        self.mod, _ = load_skill_isolated("chappie_consciousness")

    def _one_tick(self):
        sleeps = [None, _Stop()]

        def _sleep(_s):
            nxt = sleeps.pop(0)
            if nxt is not None:
                raise nxt

        # The skill's own `time` name only: patching time.sleep itself would
        # hit every other thread in the test process.
        fake_time = types.SimpleNamespace(sleep=_sleep, time=time.time,
                                          monotonic=time.monotonic)
        with mock.patch.object(self.mod, "time", fake_time), \
             mock.patch.object(self.mod, "_load_cursors", return_value={}), \
             mock.patch.object(self.mod, "_save_cursors"), \
             mock.patch.object(self.mod, "_process_episodes_once",
                               return_value=0) as eps, \
             mock.patch.object(self.mod, "_process_facts_once",
                               return_value=0) as facts, \
             mock.patch.object(self.mod, "_last_episode_run", [0.0]), \
             mock.patch.object(self.mod, "_last_fact_run", [0.0]), \
             mock.patch("builtins.print"):
            with self.assertRaises(_Stop):
                self.mod._chappie_loop()
        return eps, facts

    def test_no_episode_or_fact_pass_in_guest_mode(self):
        guest_mode.set_on(True)
        eps, facts = self._one_tick()
        eps.assert_not_called()
        facts.assert_not_called()

    def test_off_both_passes_run(self):
        eps, facts = self._one_tick()
        eps.assert_called_once()
        facts.assert_called_once()


class WakeListenerGuestTests(_GuestOn):
    """skills/wake_listener had its own, narrower "guest mode" (voice-gate
    bypass only, under the SAME action names). One switch now: the owner's
    guest mode opens the wake gate, and the skill never registers its
    toggles over the monolith's."""

    def setUp(self):
        super().setUp()
        from tests.skills.test_wake_listener import (
            inject_modules, make_fake_voice_id)
        self.mod, _ = load_skill_isolated("wake_listener")
        self.inject, self.fake_vid = inject_modules, make_fake_voice_id
        saved = self.mod.VOICE_BIOMETRIC_ENABLED
        self.addCleanup(setattr, self.mod, "VOICE_BIOMETRIC_ENABLED", saved)
        self.addCleanup(setattr, self.mod, "GUEST_MODE_ENABLED", False)
        self.mod.VOICE_BIOMETRIC_ENABLED = True
        self.mod.GUEST_MODE_ENABLED = False

    def _strict(self):
        with self.inject(**{"core.voice_id": self.fake_vid(
                available=True, enrolled=("alice",))}):
            return self.mod._gate_is_strict()

    def test_the_owners_guest_mode_opens_the_wake_gate(self):
        self.assertTrue(self._strict())
        # The boot seed alone (a saved guest mode after a restart) is the
        # memory half only: the gate stays strict, as the old per-boot
        # bypass did.
        guest_mode.set_on(True)
        self.assertTrue(self._strict())
        # A live "guest mode on" this run opens it.
        guest_mode.set_voices_open(True)
        self.assertFalse(self._strict())
        guest_mode.set_on(False)
        self.assertTrue(self._strict())

    def test_register_keeps_the_monoliths_switch(self):
        mono_on, mono_off = mock.Mock(), mock.Mock()
        actions = {"guest_mode_on": mono_on, "guest_mode_off": mono_off}
        self.mod.register(actions)
        self.assertIs(actions["guest_mode_on"], mono_on)
        self.assertIs(actions["guest_mode_off"], mono_off)
        # Alone (no monolith), the skill still offers its own toggles.
        alone = {}
        self.mod.register(alone)
        self.assertIs(alone["guest_mode_on"], self.mod.guest_mode_on)


class UnifiedHudGuestBadgeTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load(self, "jarvis_unified_hud.py", "_ju_guest_badge_ut",
                         True)

    def test_the_badge_shows_only_for_a_real_true(self):
        m = self.mod
        self.assertEqual(m._hud_guest_badge({"guest_mode": True}),
                         m.GUEST_BADGE_TEXT)
        self.assertIn("GUEST MODE", m.GUEST_BADGE_TEXT)
        for bad in ({}, {"guest_mode": False}, {"guest_mode": "yes"},
                    {"guest_mode": 1}, None, "x"):
            with self.subTest(bad=bad):
                self.assertEqual(m._hud_guest_badge(bad), "")

    def test_refresh_reads_it_from_hud_state(self):
        m = self.mod
        hud = object.__new__(m.UnifiedHud)
        hud.parent_pid = 0
        hud.frame = 0
        hud.gpu_util = None
        hud._gpu_sampling = False
        hud._gpu_cached_at = time.time() + 3600.0
        hud._last_net = None
        hud._last_net_at = None
        hud._refresh_camera_preview = lambda: False
        hud.brain = None
        for state, want in (({"state": "Idle", "guest_mode": True},
                             m.GUEST_BADGE_TEXT),
                            ({"state": "Idle"}, "")):
            with self.subTest(state=state), \
                 mock.patch.object(m, "_read_json", side_effect=lambda p, s=state:
                                   dict(s) if p == m.HUD_STATE_FILE else {}), \
                 mock.patch.object(m, "_is_parent_alive", return_value=True), \
                 mock.patch.object(m, "_control_says_off", return_value=False):
                self.assertTrue(hud._refresh())
                self.assertEqual(hud.guest_badge, want)

    def test_paint_draws_the_badge(self):
        with open(os.path.join(_HUD_DIR, "jarvis_unified_hud.py"),
                  encoding="utf-8") as f:
            src = f.read()
        paint = src[src.index("def paintEvent("):]
        paint = paint[:paint.index("\n    def ", 10)]
        self.assertIn("if self.guest_badge:", paint)
        self.assertIn("self.guest_badge)", paint)


if __name__ == "__main__":
    unittest.main()
