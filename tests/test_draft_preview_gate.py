"""SECURITY tests for core.draft_preview_gate — the send_* middleware gate.

This is the CORE gate (distinct from skills/draft_preview_gate.py, which is
covered separately under tests/skills/). run_with_gate(name, arg, fn) wraps a
send_* action: with no pending draft it is a transparent pass-through; with a
pending draft it reads the body aloud and only calls fn(arg) on an explicit
spoken confirmation. Everything else — cancel, silence, ambiguity, a readback
failure — holds the draft and returns a status string WITHOUT calling fn.

"Readback failure" is the subtle one, and the reason this file was rewritten:
``bobert_companion._speak`` reports a failed/suppressed readback by RETURNING
(None when muted — a tray toggle that persists across reboots — None on a
staging instance, None when nothing audible survives tag/markdown stripping,
False on a playback error), never by raising. An earlier revision only tested
the raising case and patched ``gate._speak`` with a bare MagicMock whose truthy
return stood in for "voiced", so the suite was green while a muted JARVIS would
still open the 8-second yes-window on a draft nobody had heard. Model the return
values, not an exception.

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

    def _run(self, *, pending, heard, action_name="send_draft", arg="x",
             speak_returns=True):
        """Drive run_with_gate.

        ``speak_returns`` is the value the patched ``gate._speak`` hands back.
        It defaults to literal ``True`` — "the line was genuinely voiced" —
        because that is the ONLY value production returns on success. A bare
        ``mock.patch.object(gate, "_speak")`` yields a truthy MagicMock, which
        is exactly the false-green that let this gate ship fail-open; pass an
        explicit ``None`` / ``False`` to model a readback the owner never
        heard.
        """
        fn = mock.MagicMock(return_value="SENT")
        with mock.patch.object(gate, "_get_pending", return_value=pending), \
             mock.patch.object(gate, "_speak",
                               return_value=speak_returns) as speak, \
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

    def test_unvoiced_readback_holds_draft_without_raising(self):
        # THE REGRESSION THIS FILE EXISTS FOR. _speak returns False (muted TTS,
        # staging instance, nothing audible after stripping, playback failure)
        # WITHOUT raising. The owner heard nothing, so the 8-second yes-window
        # must never open and fn must never fire — even with a confirm word
        # sitting in the mic.
        for ret in (False, None):
            with self.subTest(speak_returns=ret):
                result, fn, _ = self._run(pending=self.PENDING, heard="yes",
                                          speak_returns=ret)
                fn.assert_not_called()
                self.assertIn("holding the send", result.lower())

    def test_muted_readback_never_opens_the_mic(self):
        # Stronger than the above: the capture step is not even reached, so a
        # stray "okay" in the room cannot be heard, let alone matched.
        fn = mock.MagicMock(return_value="SENT")
        with mock.patch.object(gate, "_get_pending", return_value=self.PENDING), \
             mock.patch.object(gate, "_speak", return_value=False), \
             mock.patch.object(gate, "_capture_and_transcribe") as cap:
            result = gate.run_with_gate("send_draft", "x", fn)
        cap.assert_not_called()
        fn.assert_not_called()
        self.assertIn("holding the send", result.lower())

    def test_prompt_line_unvoiced_holds_draft(self):
        # The body read back fine but the "Shall I send it, sir?" prompt did
        # not — the owner never heard the question, so still fail closed.
        fn = mock.MagicMock(return_value="SENT")
        with mock.patch.object(gate, "_get_pending", return_value=self.PENDING), \
             mock.patch.object(gate, "_speak", side_effect=[True, False]), \
             mock.patch.object(gate, "_capture_and_transcribe") as cap:
            result = gate.run_with_gate("send_draft", "x", fn)
        cap.assert_not_called()
        fn.assert_not_called()
        self.assertIn("holding the send", result.lower())

    def test_truthy_non_true_readback_holds_draft(self):
        # A truthy-but-not-True _speak (e.g. an unconfigured MagicMock) must
        # NOT count as voiced. This is the guard against `if _speak(...)`.
        result, fn, _ = self._run(pending=self.PENDING, heard="yes",
                                  speak_returns=mock.MagicMock(name="truthy"))
        fn.assert_not_called()
        self.assertIn("holding the send", result.lower())

    def test_readback_text_raising_holds_draft(self):
        # _readback_text still raises on a truthy non-dict pending; the
        # try/except must keep that fail-closed too.
        fn = mock.MagicMock(return_value="SENT")
        with mock.patch.object(gate, "_get_pending", return_value="not-a-dict"), \
             mock.patch.object(gate, "_capture_and_transcribe") as cap:
            result = gate.run_with_gate("send_draft", "x", fn)
        cap.assert_not_called()
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

    def test_record_missing_returns_empty(self):
        # Companion present but no callable record_speech/transcribe → "".
        bc = mock.MagicMock()
        bc.record_speech = None
        with mock.patch.object(gate, "_import_companion", return_value=bc):
            self.assertEqual(gate._capture_and_transcribe(1.0), "")

    def test_record_raises_returns_empty(self):
        bc = mock.MagicMock()
        bc.record_speech.side_effect = RuntimeError("mic down")
        with mock.patch.object(gate, "_import_companion", return_value=bc):
            self.assertEqual(gate._capture_and_transcribe(1.0), "")

    def test_transcribe_raises_returns_empty(self):
        bc = mock.MagicMock()
        bc.record_speech.return_value = object()
        bc.transcribe.side_effect = RuntimeError("whisper down")
        with mock.patch.object(gate, "_import_companion", return_value=bc):
            self.assertEqual(gate._capture_and_transcribe(1.0), "")


# ─────────────────────────────────────────────────────────────────────────
# Coverage-completion: _import_companion, _speak, and the vip routing branch.
# ─────────────────────────────────────────────────────────────────────────

class ImportCompanionTests(unittest.TestCase):
    def test_returns_module_on_success(self):
        sentinel = object()
        with mock.patch.object(gate.importlib, "import_module",
                               return_value=sentinel):
            self.assertIs(gate._import_companion(), sentinel)

    def test_returns_none_on_import_error(self):
        with mock.patch.object(gate.importlib, "import_module",
                               side_effect=ImportError("no companion")):
            self.assertIsNone(gate._import_companion())


class SpeakTests(unittest.TestCase):
    """``gate._speak`` returns True IFF the line was actually voiced.

    ``run_with_gate`` gates the whole send on this bool, so every value other
    than literal ``True`` has to come back False. ``bobert_companion._speak``
    signals "not voiced" by RETURNING, not by raising: ``None`` when the tray
    mute toggle is on (persisted across reboots), ``None`` on a staging
    instance, ``None`` when nothing audible survives tag/markdown stripping,
    and ``False`` when synthesis/playback fails."""

    def _speak_with(self, companion, text="hello sir"):
        with mock.patch.object(gate, "_import_companion",
                               return_value=companion):
            return gate._speak(text)

    def test_routes_to_companion_speak_and_returns_true_when_voiced(self):
        bc = mock.MagicMock()
        bc._speak.return_value = True
        self.assertIs(self._speak_with(bc), True)
        bc._speak.assert_called_once_with("hello sir")

    def test_returns_false_when_companion_speak_returns_none(self):
        # Muted TTS / staging / nothing audible after stripping.
        bc = mock.MagicMock()
        bc._speak.return_value = None
        self.assertIs(self._speak_with(bc), False)

    def test_returns_false_when_companion_speak_returns_false(self):
        # Synthesis / playback failure (PortAudio, edge-tts, SAPI5).
        bc = mock.MagicMock()
        bc._speak.return_value = False
        self.assertIs(self._speak_with(bc), False)

    def test_truthy_non_true_return_is_false(self):
        # Guards against a `bool(ok)` / `if ok:` regression — an unconfigured
        # MagicMock is truthy, which is precisely what kept this suite green
        # while the gate was fail-open.
        for truthy in (mock.MagicMock(name="mock"), 1, "voiced", ["x"]):
            with self.subTest(truthy=truthy):
                bc = mock.MagicMock()
                bc._speak.return_value = truthy
                self.assertIs(self._speak_with(bc), False)

    def test_exception_returns_false(self):
        bc = mock.MagicMock()
        bc._speak.side_effect = RuntimeError("tts boom")
        self.assertIs(self._speak_with(bc), False)

    def test_no_companion_prints_but_still_returns_false(self):
        # The console fallback keeps the line visible in headless contexts,
        # but a print is NOT a readback the owner heard — a send gate must
        # treat it as a refusal.
        with mock.patch.object(gate, "_import_companion", return_value=None), \
                mock.patch("builtins.print") as mprint:
            out = gate._speak("fallback line")
        self.assertIs(out, False)
        self.assertTrue(
            any("fallback line" in str(c) for c in mprint.call_args_list))

    def test_speak_missing_prints_but_still_returns_false(self):
        bc = mock.MagicMock()
        bc._speak = None        # not callable → console fallback path
        with mock.patch.object(gate, "_import_companion", return_value=bc), \
                mock.patch("builtins.print") as mprint:
            out = gate._speak("no speaker")
        self.assertIs(out, False)
        self.assertTrue(
            any("no speaker" in str(c) for c in mprint.call_args_list))


class GetPendingVipRoutingTests(unittest.TestCase):
    def test_vip_action_tries_vip_intercept_first(self):
        # A send_vip_* action should consult skills.vip_intercept first.
        order = []
        vip = mock.MagicMock()
        vip.get_pending_draft.return_value = {"body": "vip draft"}

        def fake_import(name):
            order.append(name)
            if name == "skills.vip_intercept":
                return vip
            raise ImportError("only vip available")

        with mock.patch.object(gate.importlib, "import_module",
                               side_effect=fake_import):
            out = gate._get_pending("send_vip_reply")
        self.assertEqual(out, {"body": "vip draft"})
        self.assertEqual(order[0], "skills.vip_intercept")

    def test_first_provider_import_fails_falls_through(self):
        # vip import fails → continue to email_triage which returns a draft.
        triage = mock.MagicMock()
        triage.get_pending_draft.return_value = {"body": "triage draft"}

        def fake_import(name):
            if name == "skills.vip_intercept":
                raise ImportError("no vip")
            return triage

        with mock.patch.object(gate.importlib, "import_module",
                               side_effect=fake_import):
            out = gate._get_pending("send_vip_reply")
        self.assertEqual(out, {"body": "triage draft"})

    def test_getter_absent_skips_provider(self):
        # Provider module present but exposes neither getter → returns None.
        mod = mock.MagicMock()
        del mod.get_pending_draft
        del mod._get_pending
        with mock.patch.object(gate.importlib, "import_module", return_value=mod):
            self.assertIsNone(gate._get_pending("send_draft"))

    def test_falls_back_to_private_getter(self):
        # No public get_pending_draft, but a private _get_pending exists.
        mod = mock.MagicMock()
        del mod.get_pending_draft
        mod._get_pending.return_value = {"body": "via private"}
        with mock.patch.object(gate.importlib, "import_module", return_value=mod):
            self.assertEqual(gate._get_pending("send_draft"), {"body": "via private"})


if __name__ == "__main__":
    unittest.main()
