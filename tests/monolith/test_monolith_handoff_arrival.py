"""Monolith wiring for the 2026-10-20 "identical copy" handoff: the copy that
stands down hands over the conversation in progress, and the arriving copy
(_handoff_arrived, run by core/peer_handoff after "stood down") takes it.

    python -m unittest tests.monolith.test_monolith_handoff_arrival
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

TURNS = [{"role": "user", "content": "Jarvis, how's the server?"},
         {"role": "assistant", "content": "All four services answer, sir."}]


@requires_monolith
class HandoffArrivalTests(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.spoken: list[str] = []
        self.history: list[dict] = []
        self._p(bc, "_speak", side_effect=lambda t, *a, **k: self.spoken.append(t))
        self._p(bc, "conversation_history", self.history)
        self._p(bc, "_trim_conversation_history", lambda *a, **k: 0)
        from core import peer_handoff, peer_memory
        self.pm = peer_memory
        self._p(peer_handoff, "peer_name", return_value="desktop")
        self._p(peer_memory, "own_version", return_value="2.0.227")

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _arrive(self, got):
        with mock.patch.object(self.pm, "arrival_handover", return_value=got), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.bc._handoff_arrived()
        return out.getvalue()

    def test_a_live_conversation_is_picked_up_and_announced(self):
        self._arrive({"turns": TURNS, "age_s": 20.0, "version": "2.0.227", "host": "DESK"})
        self.assertEqual(self.history, TURNS)
        self.assertEqual(self.spoken, ["Picking up where we left off on the desktop, sir."])

    def test_an_older_hand_over_is_taken_quietly(self):
        self._arrive({"turns": TURNS, "age_s": 600.0, "version": "2.0.227", "host": "DESK"})
        self.assertEqual(self.history, TURNS)
        self.assertEqual(self.spoken, [])

    def test_a_conversation_already_started_here_is_left_alone(self):
        here = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"}]
        self.history.extend(here)
        self._arrive({"turns": TURNS, "age_s": 5.0, "version": "2.0.227", "host": "DESK"})
        self.assertEqual(self.history, here)
        self.assertEqual(self.spoken, [])

    def test_nothing_handed_over_is_quiet_and_a_version_gap_is_logged(self):
        out = self._arrive({})
        self.assertEqual((self.history, self.spoken), ([], []))
        out = self._arrive({"turns": TURNS, "age_s": 900.0, "version": "2.0.226", "host": "DESK"})
        self.assertIn("version differs", out)


if __name__ == "__main__":
    unittest.main()
