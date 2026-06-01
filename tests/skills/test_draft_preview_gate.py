"""Logic tests for skills/draft_preview_gate.py — SECURITY-relevant.

This skill is the unified outbound-message confirmation gate. The tests below
exercise the parts that actually decide whether a message goes out:

  • recipient resolution for the spoken "I have a reply for X" prompt,
  • body extraction across the {body|message|text} draft shapes,
  • the FAIL-CLOSED contract: no body, sleep/standby active, core gate not
    imported, or draft_confirm raising → gate_outbound_message returns False
    (the send must NOT proceed),
  • the FAIL-OPEN sleep check (lookup error ⇒ treat user as awake),
  • the explicit-confirm happy path returns True only when draft_confirm does,
  • the status action's health reporting.

No monolith boot. core.draft_confirm.draft_confirm is mocked so no TTS / mic /
whisper runs and nothing is actually sent.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class DraftPreviewGateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("draft_preview_gate")

    # ── _resolve_recipient ───────────────────────────────────────────────
    def test_recipient_prefers_explicit_recipient(self):
        out = self.mod._resolve_recipient("teams", {"recipient": "Sam", "to": "x"})
        self.assertEqual(out, "Sam")

    def test_recipient_falls_back_to_to_field(self):
        self.assertEqual(self.mod._resolve_recipient("email", {"to": "boss@x"}), "boss@x")

    def test_recipient_uses_channel_label(self):
        # Known channel with no explicit recipient → spoken label.
        self.assertEqual(self.mod._resolve_recipient("teams", {}), "Teams")
        self.assertEqual(self.mod._resolve_recipient("sms", {}), "text")

    def test_recipient_unknown_channel_is_informative(self):
        # Unknown channel still yields a non-empty phrase, never "".
        self.assertEqual(self.mod._resolve_recipient("carrierpigeon", {}), "carrierpigeon")
        self.assertEqual(self.mod._resolve_recipient("", {}), "the recipient")

    # ── _draft_body ──────────────────────────────────────────────────────
    def test_draft_body_priority_body_over_message_over_text(self):
        d = {"body": "B", "message": "M", "text": "T"}
        self.assertEqual(self.mod._draft_body(d), "B")
        self.assertEqual(self.mod._draft_body({"message": "M", "text": "T"}), "M")
        self.assertEqual(self.mod._draft_body({"text": "T"}), "T")

    def test_draft_body_empty_when_no_known_keys(self):
        self.assertEqual(self.mod._draft_body({"subject": "hi"}), "")
        self.assertEqual(self.mod._draft_body({"body": "   "}), "")  # whitespace-only

    # ── gate_outbound_message: FAIL-CLOSED paths ─────────────────────────
    def test_gate_refuses_non_dict_non_str_draft(self):
        self.assertFalse(self.mod.gate_outbound_message("teams", 12345))

    def test_gate_refuses_empty_body(self):
        # No body to read aloud → refuse rather than silently approve.
        with mock.patch.object(self.mod, "_draft_confirm_mod") as cm:
            self.assertFalse(self.mod.gate_outbound_message("teams", {"subject": "x"}))
            cm.draft_confirm.assert_not_called()

    def test_gate_refuses_when_asleep(self):
        # Regression guard: never prompt while asleep — a sleep-talk "yes"
        # must not send. draft_confirm must not even be called.
        with mock.patch.object(self.mod, "_is_asleep_or_standby", return_value=True), \
             mock.patch.object(self.mod, "_draft_confirm_mod") as cm:
            self.assertFalse(self.mod.gate_outbound_message("teams", "hello there"))
            cm.draft_confirm.assert_not_called()

    def test_gate_refuses_when_core_confirm_unavailable(self):
        # Broken import of the core gate ⇒ fail closed.
        with mock.patch.object(self.mod, "_is_asleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_draft_confirm_mod", None):
            self.assertFalse(self.mod.gate_outbound_message("teams", "hello there"))

    def test_gate_refuses_when_confirm_raises(self):
        cm = mock.MagicMock()
        cm.draft_confirm.side_effect = RuntimeError("mic exploded")
        with mock.patch.object(self.mod, "_is_asleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_draft_confirm_mod", cm):
            self.assertFalse(self.mod.gate_outbound_message("teams", "hello there"))

    # ── gate_outbound_message: happy path ────────────────────────────────
    def test_gate_returns_true_only_on_explicit_confirm(self):
        cm = mock.MagicMock()
        cm.draft_confirm.return_value = True
        with mock.patch.object(self.mod, "_is_asleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_draft_confirm_mod", cm):
            self.assertTrue(
                self.mod.gate_outbound_message("teams", {"body": "ship it",
                                                         "recipient": "Sam"}))
        # The body + resolved recipient are forwarded to the core confirm.
        _, kwargs = cm.draft_confirm.call_args
        self.assertEqual(cm.draft_confirm.call_args[0][0], "ship it")
        self.assertEqual(kwargs.get("recipient"), "Sam")

    def test_gate_returns_false_when_user_declines(self):
        cm = mock.MagicMock()
        cm.draft_confirm.return_value = False     # user said no / timed out
        with mock.patch.object(self.mod, "_is_asleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_draft_confirm_mod", cm):
            self.assertFalse(self.mod.gate_outbound_message("sms", "are you home?"))

    def test_gate_accepts_string_draft_as_body(self):
        cm = mock.MagicMock()
        cm.draft_confirm.return_value = True
        with mock.patch.object(self.mod, "_is_asleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_draft_confirm_mod", cm):
            self.assertTrue(self.mod.gate_outbound_message("telegram", "raw body string"))
        self.assertEqual(cm.draft_confirm.call_args[0][0], "raw body string")

    # ── _is_asleep_or_standby ────────────────────────────────────────────
    def test_sleep_check_reads_companion_flags(self):
        fake_bc = mock.MagicMock()
        fake_bc._sleep_mode = [True]
        fake_bc._standby_mode = [False]
        with mock.patch.object(self.mod, "_companion", return_value=fake_bc):
            self.assertTrue(self.mod._is_asleep_or_standby())
        fake_bc._sleep_mode = [False]
        fake_bc._standby_mode = [True]
        with mock.patch.object(self.mod, "_companion", return_value=fake_bc):
            self.assertTrue(self.mod._is_asleep_or_standby())

    def test_sleep_check_fails_open_when_companion_absent(self):
        # No companion loaded → treat user as awake (prompt), the safer default.
        with mock.patch.object(self.mod, "_companion", return_value=None):
            self.assertFalse(self.mod._is_asleep_or_standby())

    def test_sleep_check_false_when_flags_clear(self):
        fake_bc = mock.MagicMock()
        fake_bc._sleep_mode = [False]
        fake_bc._standby_mode = [False]
        with mock.patch.object(self.mod, "_companion", return_value=fake_bc):
            self.assertFalse(self.mod._is_asleep_or_standby())

    # ── status action ────────────────────────────────────────────────────
    def test_status_reports_loaded_gates(self):
        cm = mock.MagicMock()
        cm.CONFIRM_TIMEOUT_S = 12
        with mock.patch.object(self.mod, "_core_preview_mod", mock.MagicMock()), \
             mock.patch.object(self.mod, "_draft_confirm_mod", cm), \
             mock.patch.object(self.mod, "_is_asleep_or_standby", return_value=False):
            out = self.actions["draft_preview_gate_status"]("")
        self.assertIn("core preview gate: ok", out)
        self.assertIn("12s window", out)
        self.assertIn("awake", out)
        self.assertTrue(out.endswith("sir."))

    def test_status_reports_broken_gate_and_standby(self):
        with mock.patch.object(self.mod, "_core_preview_mod", None), \
             mock.patch.object(self.mod, "_draft_confirm_mod", None), \
             mock.patch.object(self.mod, "_is_asleep_or_standby", return_value=True):
            out = self.actions["draft_preview_gate_status"]("")
        self.assertIn("core preview gate: NOT loaded", out)
        self.assertIn("core draft confirm: NOT loaded", out)
        self.assertIn("auto-refused", out)

    def test_status_alias_registered(self):
        # Both names route to the same handler.
        self.assertIs(self.actions["outbound_gate_status"],
                      self.actions["draft_preview_gate_status"])


if __name__ == "__main__":
    unittest.main()
