"""SECURITY tests for core.draft_confirm — the imperative pre-send gate.

draft_confirm(text, recipient) reads a draft aloud and waits for an explicit
spoken yes/no. The whole point is FAIL-CLOSED: anything other than an
unambiguous confirmation keyword within the window returns False, so a skill
firing an outbound message never auto-sends on silence, ambiguity, a cancel
word, or a broken mic / TTS / whisper path.

These tests mock the companion (`bobert_companion`) so no real audio I/O runs,
and point the pending-draft file at a tempdir so nothing writes into the real
data/ directory. stdlib unittest + unittest.mock only.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from core import draft_confirm as dc


def _fake_companion(*, speak_ok=True, heard=None, record_returns=object(),
                    transcribe_meta=None):
    """Build a stand-in bobert_companion module.

    speak_ok        — whether _speak succeeds (False ⇒ TTS down).
    heard           — the transcribed text returned by transcribe(); when None,
                      record_speech() returns None (silence in the window).
    record_returns  — sentinel audio object handed to transcribe().
    """
    bc = mock.MagicMock(name="bobert_companion")

    if speak_ok:
        bc._speak.return_value = None
    else:
        bc._speak.side_effect = RuntimeError("tts down")

    if heard is None:
        bc.record_speech.return_value = None          # silence
    else:
        bc.record_speech.return_value = record_returns
        bc.transcribe.return_value = (heard, transcribe_meta or {})
    return bc


class DraftConfirmTestBase(unittest.TestCase):
    def setUp(self):
        # Redirect the pending-draft persistence file so we never touch data/.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        pending = os.path.join(self._tmp.name, "draft_confirm_pending.json")
        p = mock.patch.object(dc, "_PENDING_FILE", pending)
        p.start()
        self.addCleanup(p.stop)
        self.pending_file = pending

    def _run(self, text, recipient="", *, companion):
        with mock.patch.object(dc, "_import_companion", return_value=companion):
            return dc.draft_confirm(text, recipient)


class MatchesAnyTests(unittest.TestCase):
    """The whole-word matcher is the security-critical primitive — a false
    positive here is an unwanted send."""

    def test_single_token_whole_word(self):
        self.assertTrue(dc._matches_any("yes please", dc._CONFIRM_KEYWORDS))
        # 'noted' must NOT trip the cancel 'no'.
        self.assertFalse(dc._matches_any("noted", dc._CANCEL_KEYWORDS))

    def test_multiword_phrase_substring(self):
        self.assertTrue(dc._matches_any("go on, send it now", dc._CONFIRM_KEYWORDS))
        self.assertTrue(dc._matches_any("please do not send", dc._CANCEL_KEYWORDS))

    def test_empty_text_matches_nothing(self):
        self.assertFalse(dc._matches_any("", dc._CONFIRM_KEYWORDS))
        self.assertFalse(dc._matches_any("", dc._CANCEL_KEYWORDS))

    def test_confirm_does_not_match_unrelated(self):
        self.assertFalse(dc._matches_any("the weather is nice", dc._CONFIRM_KEYWORDS))


class PromptLineTests(unittest.TestCase):
    def test_includes_recipient_when_present(self):
        line = dc._prompt_line("see you at 5", "Sam")
        self.assertIn("Sam", line)
        self.assertIn("see you at 5", line)
        self.assertIn("send it", line.lower())

    def test_generic_when_no_recipient(self):
        line = dc._prompt_line("body here", "")
        self.assertNotIn("for ", line.split(":")[0])  # no "for <name>" head
        self.assertIn("body here", line)


class FailClosedTests(DraftConfirmTestBase):
    def test_empty_body_refused_without_touching_audio(self):
        bc = _fake_companion(heard="yes")
        # Even though the (would-be) reply is "yes", an empty draft is refused
        # and no speech/record happens.
        with mock.patch.object(dc, "_import_companion", return_value=bc):
            self.assertFalse(dc.draft_confirm("   ", "Sam"))
        bc._speak.assert_not_called()
        bc.record_speech.assert_not_called()

    def test_tts_down_fails_closed(self):
        bc = _fake_companion(speak_ok=False, heard="yes")
        self.assertFalse(self._run("ship the build", "Sam", companion=bc))
        # Never even gets to recording once speech failed.
        bc.record_speech.assert_not_called()

    def test_no_speaker_callable_fails_closed(self):
        bc = mock.MagicMock()
        bc._speak = "not callable"      # getattr returns a non-callable
        self.assertFalse(self._run("hi", "x", companion=bc))

    def test_companion_absent_fails_closed(self):
        # _import_companion returns None ⇒ _speak returns False ⇒ abort.
        self.assertFalse(self._run("hi", "x", companion=None))

    def test_silence_in_window_fails_closed(self):
        bc = _fake_companion(heard=None)   # record_speech returns None
        self.assertFalse(self._run("are you home?", "Sam", companion=bc))
        bc._speak.assert_called_once()     # it did read the draft aloud

    def test_record_raises_fails_closed(self):
        bc = _fake_companion(heard="yes")
        bc.record_speech.side_effect = RuntimeError("mic gone")
        self.assertFalse(self._run("hi", "x", companion=bc))

    def test_transcribe_raises_fails_closed(self):
        bc = _fake_companion(heard="yes")
        bc.transcribe.side_effect = RuntimeError("whisper unloaded")
        self.assertFalse(self._run("hi", "x", companion=bc))

    def test_explicit_cancel_returns_false(self):
        bc = _fake_companion(heard="no, cancel that")
        self.assertFalse(self._run("send the email", "boss", companion=bc))

    def test_ambiguous_reply_fails_closed(self):
        # Heard something, but it's neither yes-shaped nor no-shaped.
        bc = _fake_companion(heard="what time is it")
        self.assertFalse(self._run("send the email", "boss", companion=bc))

    def test_empty_transcription_is_abort(self):
        # transcribe returns "" (heard nothing intelligible) → fail closed.
        bc = _fake_companion(heard="")
        self.assertFalse(self._run("send the email", "boss", companion=bc))

    def test_record_missing_returns_false(self):
        # Companion present but lacks record_speech/transcribe callables.
        bc = mock.MagicMock()
        bc._speak.return_value = None
        bc.record_speech = None
        bc.transcribe = None
        self.assertFalse(self._run("hi", "x", companion=bc))


class HappyPathTests(DraftConfirmTestBase):
    def test_explicit_yes_returns_true(self):
        bc = _fake_companion(heard="yes")
        self.assertTrue(self._run("ship it", "Sam", companion=bc))
        # The composed prompt was actually spoken, with the body in it.
        spoken = bc._speak.call_args[0][0]
        self.assertIn("ship it", spoken)

    def test_confirm_keyword_variants(self):
        for word in ("confirm", "send it", "go ahead", "affirmative", "do it"):
            bc = _fake_companion(heard=f"okay {word}")
            self.assertTrue(self._run("body", "x", companion=bc),
                            f"{word!r} should confirm")

    def test_cancel_takes_priority_over_confirm(self):
        # If both a cancel and a confirm word appear, cancel must win (the
        # check order in draft_confirm puts cancel first).
        bc = _fake_companion(heard="no don't, well, yes")
        self.assertFalse(self._run("body", "x", companion=bc))


class PendingFilePersistenceTests(DraftConfirmTestBase):
    def test_pending_cleared_after_run(self):
        # After a completed gate, the active pending record is reset to None
        # (the finally branch). We assert the file ends with active=None.
        import json
        bc = _fake_companion(heard="yes")
        self._run("body text", "Sam", companion=bc)
        with open(self.pending_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsNone(data.get("active"))

    def test_write_pending_swallows_write_failure(self):
        # _write_pending is best-effort: a write error (here os.makedirs raises)
        # is logged and swallowed so a full disk can't abort a confirmation.
        with mock.patch.object(dc.os, "makedirs", side_effect=OSError("no dir")):
            dc._write_pending({"body": "x"})       # must not raise
            dc._write_pending(None)                 # None branch, also swallowed


class ImportCompanionTests(unittest.TestCase):
    """Exercise the REAL _import_companion / _capture_and_transcribe helpers
    (the other suites stub _import_companion, so its body never runs)."""

    def test_import_companion_returns_none_on_failure(self):
        # bobert_companion is the heavy monolith and is NOT importable on CI;
        # force the import to fail so the except→None path runs deterministically
        # on every host.
        with mock.patch.object(dc.importlib, "import_module",
                               side_effect=ImportError("no monolith")):
            self.assertIsNone(dc._import_companion())

    def test_import_companion_returns_module_on_success(self):
        sentinel = mock.MagicMock(name="fake_companion")
        with mock.patch.object(dc.importlib, "import_module",
                               return_value=sentinel):
            self.assertIs(dc._import_companion(), sentinel)

    def test_capture_returns_none_when_companion_absent(self):
        # _capture_and_transcribe short-circuits to None (hard failure) when the
        # companion can't be imported — this is the fail-closed mic path.
        with mock.patch.object(dc, "_import_companion", return_value=None):
            self.assertIsNone(dc._capture_and_transcribe(1.0))


class GateRaisesTests(DraftConfirmTestBase):
    def test_unexpected_error_in_gate_fails_closed(self):
        # An unexpected exception inside the locked try-block (here the
        # transcribe step raises a surprise) is caught by the outer except and
        # the gate fails closed (False) rather than propagating.
        bc = _fake_companion(heard="yes")
        with mock.patch.object(dc, "_capture_and_transcribe",
                               side_effect=RuntimeError("surprise")):
            self.assertFalse(self._run("ship it", "Sam", companion=bc))


if __name__ == "__main__":
    unittest.main()
