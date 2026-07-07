"""Monolith seam test: bobert_companion.synthesise() ↔ core.voice_clone.

Proves the integration contract:
  * When core.voice_clone.is_available() is False, synthesise() IGNORES the
    clone and falls through to the EXISTING edge-tts ladder unchanged.
  * When is_available() is True and synthesize() returns a waveform,
    synthesise() returns THAT waveform (the clone wins) and never calls the
    edge-tts renderer.
  * When is_available() is True but synthesize() returns None (a render
    failure), synthesise() still falls back to the edge-tts ladder — a clone
    failure never silences JARVIS.
  * The master switch (VOICE_CLONE_ENABLED) gates the whole thing: OFF means
    the clone code path is never even entered.

LOCAL full tier only (@requires_monolith): the monolith top-level-imports heavy
deps absent on the CI runner, so this skips there and runs locally. core.voice_-
clone is patched at the module level — no chatterbox, no torch, no GPU, ever.
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

import core.voice_clone as vc


@requires_monolith
class SynthesiseVoiceCloneTests(MonolithGlobalsTestCase):
    # cls.bc is loaded by MonolithGlobalsTestCase.setUpClass; the base also
    # deep-restores mutated monolith globals after every test.

    def setUp(self):
        bc = self.bc
        # synthesise() reads these single-element state cells for prosody.
        self._p(bc, "_last_voice_route", [{"addendum": "", "mood": "casual"}])
        self._p(bc, "_last_user_tone", [None])
        self._p(bc, "_last_mood", [None])
        # A distinctive edge-tts sentinel so we can tell the ladder ran.
        self._edge_sentinel = (np.full(64, 0.25, dtype=np.float32), 24000)
        self._p(bc, "_render_edge_tts",
                lambda text, rate, pitch: self._edge_sentinel)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    # ── fallback when the clone is unavailable ───────────────────────────────
    def test_falls_back_to_edge_when_clone_unavailable(self):
        bc = self.bc
        with mock.patch.object(bc, "VOICE_CLONE_ENABLED", True), \
             mock.patch.object(vc, "is_available", return_value=False), \
             mock.patch.object(vc, "synthesize",
                               side_effect=AssertionError("must not render")):
            audio, sr = bc.synthesise("hello sir")
        # Got the edge sentinel — the clone was skipped.
        self.assertEqual(sr, 24000)
        self.assertTrue(np.allclose(audio, 0.25))

    def test_master_switch_off_never_touches_clone(self):
        bc = self.bc
        # is_available() must NEVER be consulted when the master switch is off.
        with mock.patch.object(bc, "VOICE_CLONE_ENABLED", False), \
             mock.patch.object(vc, "is_available",
                               side_effect=AssertionError("gate bypassed")):
            audio, sr = bc.synthesise("hello sir")
        self.assertEqual(sr, 24000)
        self.assertTrue(np.allclose(audio, 0.25))

    # ── clone wins when available + renders ──────────────────────────────────
    def test_uses_clone_waveform_when_available(self):
        bc = self.bc
        clone_wave = (np.full(128, -0.5, dtype=np.float32), 22050)
        edge_called = {"n": 0}

        def _edge(text, rate, pitch):
            edge_called["n"] += 1
            return self._edge_sentinel

        with mock.patch.object(bc, "VOICE_CLONE_ENABLED", True), \
             mock.patch.object(bc, "_render_edge_tts", _edge), \
             mock.patch.object(vc, "is_available", return_value=True), \
             mock.patch.object(vc, "synthesize", return_value=clone_wave):
            audio, sr = bc.synthesise("hello sir")
        # Got the CLONE waveform, and the edge ladder was never invoked.
        self.assertEqual(sr, 22050)
        self.assertTrue(np.allclose(audio, -0.5))
        self.assertEqual(edge_called["n"], 0)

    # ── clone available but render returns None → still falls back ───────────
    def test_falls_back_when_clone_returns_none(self):
        bc = self.bc
        with mock.patch.object(bc, "VOICE_CLONE_ENABLED", True), \
             mock.patch.object(vc, "is_available", return_value=True), \
             mock.patch.object(vc, "synthesize", return_value=None):
            audio, sr = bc.synthesise("hello sir")
        self.assertEqual(sr, 24000)
        self.assertTrue(np.allclose(audio, 0.25))

    # ── clone raising is swallowed → falls back (never silences JARVIS) ──────
    def test_clone_exception_is_swallowed_and_falls_back(self):
        bc = self.bc
        with mock.patch.object(bc, "VOICE_CLONE_ENABLED", True), \
             mock.patch.object(vc, "is_available",
                               side_effect=RuntimeError("engine exploded")):
            audio, sr = bc.synthesise("hello sir")
        self.assertEqual(sr, 24000)
        self.assertTrue(np.allclose(audio, 0.25))


if __name__ == "__main__":
    unittest.main()
