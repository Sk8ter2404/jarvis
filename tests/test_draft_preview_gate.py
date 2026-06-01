"""SECURITY tests for core.draft_preview_gate — the send_* middleware gate.

This is the CORE gate (distinct from skills/draft_preview_gate.py, which is
covered separately under tests/skills/). run_with_gate(name, arg, fn) wraps a
send_* action: with no pending draft it is a transparent pass-through; with a
pending draft it reads the body aloud and only calls fn(arg) on an explicit
spoken confirmation. Everything else — cancel, silence, ambiguity, a readback
failure — holds the draft and returns a status string WITHOUT calling fn.

The companion (bobert_companion) and the per-skill pending-draft providers are
mocked, so no audio runs and nothing is sent. stdlib unittest + mock only.
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import draft_preview_gate as gate


class ShouldGateTests(unittest.TestCase):
    def test_send_prefix_qualifies(self):
        self.assertTrue(gate.should_gate("send_draft"))
        self.assertTrue(gate.should_gate("send_pending_draft"))
        self.assertTrue(gate.should_gate("SEND_VIP_REPLY"))   # case-insensitive

    def test_non_send_actions_skip(self):
        self.assertFalse(gate.should_gate("play_music"))
        self.assertFalse(gate.should_gate("resend"))           # not a prefix
        self.assertFalse(gate.should_gate(""))


class MatchesAnyTests(unittest.TestCase):
    def test_whole_word_and_phrases(self):
        self.assertTrue(gate._matches_any("yes do it", gate._CONFIRM_KEYWORDS))
        self.assertTrue(gate._matches_any("send it now", gate._CONFIRM_KEYWORDS))
        self.assertTrue(gate._matches_any("do not", gate._CANCEL_KEYWORDS))
        self.assertFalse(gate._matches_any("noted", gate._CANCEL_KEYWORDS))
        self.assertFalse(gate._matches_any("", gate._CONFIRM_KEYWORDS))


class ReadbackTextTests(unittest.TestCase):
    def test_includes_to_and_subject(self):
        out = gate._readback_text({"to": "Sam", "subject": "Lunch", "body": "noon?"})
        self.assertIn("to Sam", out)
        self.assertIn("subject Lunch", out)
        self.assertIn("noon?", out)

    def test_body_only(self):
        out = gate._readback_text({"body": "just the body"})
        self.assertIn("just the body", out)
        self.assertNotIn("subject", out)


class _GateHarness(unittest.TestCase):
    """Helpers to drive run_with_gate with a controllable pending draft and a
    controllable transcription result, without any real audio."""

    def _run(self, *, pending, heard, action_name="send_draft", arg="x"):
        fn = mock.MagicMock(return_value="SENT")
        with mock.patch.object(gate, "_get_pending", return_value=pending), \
             mock.patch.object(gate, "_speak") as speak, \
             mock.patch.object(gate, "_capture_and_transcribe", return_value=heard):
            result = gate.run_with_gate(action_name, arg, fn)
        return result, fn, speak


class PassThroughTests(_GateHarness):
    def test_no_pending_draft_is_transparent(self):
        # Gate must call fn immediately and not speak anything.
        result, fn, speak = self._run(pending=None, heard="")
        fn.assert_called_once_with("x")
        self.assertEqual(result, "SENT")
        speak.assert_not_called()


class FailClosedTests(_GateHarness):
    PENDING = {"to": "Sam", "subject": "Hi", "body": "ship it"}

    def test_silence_holds_draft(self):
        result, fn, _ = self._run(pending=self.PENDING, heard="")
        fn.assert_not_called()
        self.assertIn("No confirmation", result)

    def test_cancel_holds_draft(self):
        result, fn, _ = self._run(pending=self.PENDING, heard="no cancel that")
        fn.assert_not_called()
        self.assertIn("Holding the draft", result)

    def test_ambiguous_holds_draft(self):
        result, fn, _ = self._run(pending=self.PENDING, heard="what time is it")
        fn.assert_not_called()
        self.assertIn("Couldn't tell", result)

    def test_readback_failure_holds_draft(self):
        # If _speak raises, the gate must abort the send, not fall through.
        fn = mock.MagicMock(return_value="SENT")
        with mock.patch.object(gate, "_get_pending", return_value=self.PENDING), \
             mock.patch.object(gate, "_speak", side_effect=RuntimeError("tts boom")), \
             mock.patch.object(gate, "_capture_and_transcribe", return_value="yes"):
            result = gate.run_with_gate("send_draft", "x", fn)
        fn.assert_not_called()
        self.assertIn("holding the send", result.lower())


class ConfirmTests(_GateHarness):
    PENDING = {"to": "Sam", "body": "ship it"}

    def test_explicit_yes_sends(self):
        result, fn, speak = self._run(pending=self.PENDING, heard="yes")
        fn.assert_called_once_with("x")
        self.assertEqual(result, "SENT")
        # The draft body was read aloud + the prompt line spoken.
        self.assertEqual(speak.call_count, 2)

    def test_confirm_synonyms_send(self):
        for word in ("confirm", "send it", "go ahead", "ship it", "affirmative"):
            result, fn, _ = self._run(pending=self.PENDING, heard=f"yeah {word}")
            fn.assert_called_once_with("x")
            self.assertEqual(result, "SENT")

    def test_cancel_beats_confirm_when_both_present(self):
        # Cancel is checked before confirm, so a reply containing both holds.
        result, fn, _ = self._run(pending=self.PENDING, heard="no wait yes")
        fn.assert_not_called()
        self.assertIn("Holding the draft", result)


class GetPendingRoutingTests(unittest.TestCase):
    """_get_pending routes vip_* actions to skills.vip_intercept and
    everything else to skills.email_triage, tolerating import / fetch errors
    by returning None (→ pass-through)."""

    def test_returns_none_when_provider_import_fails(self):
        with mock.patch.object(gate.importlib, "import_module",
                               side_effect=ImportError("no module")):
            self.assertIsNone(gate._get_pending("send_draft"))

    def test_returns_pending_from_email_triage(self):
        fake = mock.MagicMock()
        fake.get_pending_draft.return_value = {"body": "hi"}
        with mock.patch.object(gate.importlib, "import_module", return_value=fake):
            self.assertEqual(gate._get_pending("send_draft"), {"body": "hi"})

    def test_getter_raising_yields_none(self):
        fake = mock.MagicMock()
        fake.get_pending_draft.side_effect = RuntimeError("boom")
        # No _get_pending fallback attribute on the mock spec either.
        del fake._get_pending
        with mock.patch.object(gate.importlib, "import_module", return_value=fake):
            self.assertIsNone(gate._get_pending("send_draft"))


class CaptureAndTranscribeTests(unittest.TestCase):
    def test_no_companion_returns_empty(self):
        with mock.patch.object(gate, "_import_companion", return_value=None):
            self.assertEqual(gate._capture_and_transcribe(1.0), "")

    def test_record_none_returns_empty(self):
        bc = mock.MagicMock()
        bc.record_speech.return_value = None
        with mock.patch.object(gate, "_import_companion", return_value=bc):
            self.assertEqual(gate._capture_and_transcribe(1.0), "")

    def test_happy_lowercased_and_stripped(self):
        bc = mock.MagicMock()
        bc.record_speech.return_value = object()
        bc.transcribe.return_value = ("  YES Please  ", {})
        with mock.patch.object(gate, "_import_companion", return_value=bc):
            self.assertEqual(gate._capture_and_transcribe(1.0), "yes please")


if __name__ == "__main__":
    unittest.main()
