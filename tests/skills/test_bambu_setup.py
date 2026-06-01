"""Logic tests for skills/bambu_setup.py.

The first-time printer wizard is mostly pure parsing helpers plus a guarded
inline-args entry point. We cover:

  • _parse_bambu_packet  — NOTIFY-shaped UDP payload → {ip, serial, model, name}
  • _voice_to_digits     — spoken digits / numbers → digit string
  • _voice_to_ip         — dotted / spoken IPv4 extraction
  • _affirmative / _negative — yes/no intent
  • _format_digits_for_speech / _humanise_printer
  • _parse_inline_args    — the non-voice "ip access [serial]" shortcut
  • _wizard_pick_printer  — single / multi / index / model-name selection
  • setup_printer         — inline-args happy path + the "already running" lock,
                            with _persist_credentials and _restart_monitor stubbed
                            so NO real bobert_companion.py source is rewritten.

register() only wires actions (no thread, no I/O), so it loads cleanly. The
wizard's voice helpers (_say/_listen) are patched wherever a path reaches them.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class BambuSetupParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    # ── _parse_bambu_packet ──────────────────────────────────────────────
    def test_parse_packet_extracts_fields(self):
        payload = (
            b"NOTIFY * HTTP/1.1\r\n"
            b"Location: 192.168.1.42\r\n"
            b"USN: 01P00A123456789\r\n"
            b"DevModel.bambu.com: BL-P001\r\n"
            b"DevName.bambu.com: My H2D\r\n"
            b"From: bambulab\r\n"
        )
        out = self.mod._parse_bambu_packet(payload)
        self.assertEqual(out["ip"], "192.168.1.42")
        self.assertEqual(out["serial"], "01P00A123456789")
        self.assertEqual(out["model"], "BL-P001")
        self.assertEqual(out["name"], "My H2D")

    def test_parse_packet_rejects_non_bambu(self):
        self.assertEqual(
            self.mod._parse_bambu_packet(b"Location: 10.0.0.1\r\nrandom upnp"),
            {})

    def test_parse_packet_requires_ip(self):
        # bambu signature present but no Location → useless, empty dict.
        self.assertEqual(
            self.mod._parse_bambu_packet(b"USN: abc\r\nfrom bambulab device"),
            {})

    # ── _voice_to_digits ─────────────────────────────────────────────────
    def test_voice_to_digits_words(self):
        self.assertEqual(self.mod._voice_to_digits("one two three four"), "1234")

    def test_voice_to_digits_mixed_and_homophones(self):
        # "to"/"for" homophones map to 2/4; literal digits pass through.
        self.assertEqual(self.mod._voice_to_digits("to for 5 6"), "2456")

    def test_voice_to_digits_compound_numbers(self):
        # "twenty" → "20", "three" → "3"
        self.assertEqual(self.mod._voice_to_digits("twenty three"), "203")

    def test_voice_to_digits_empty(self):
        self.assertEqual(self.mod._voice_to_digits(""), "")
        self.assertEqual(self.mod._voice_to_digits("hello there"), "")

    # ── _voice_to_ip ─────────────────────────────────────────────────────
    def test_voice_to_ip_dotted_direct(self):
        self.assertEqual(self.mod._voice_to_ip("it's 192.168.1.42 sir"),
                         "192.168.1.42")

    def test_voice_to_ip_spoken_with_dot(self):
        out = self.mod._voice_to_ip("one nine two dot one six eight dot one dot four two")
        self.assertEqual(out, "192.168.1.42")

    def test_voice_to_ip_rejects_out_of_range_octet(self):
        # 300 is not a valid octet → no four-octet result.
        self.assertEqual(
            self.mod._voice_to_ip("three zero zero dot one dot one dot one"), "")

    def test_voice_to_ip_empty_when_unparseable(self):
        self.assertEqual(self.mod._voice_to_ip("no numbers here"), "")

    # ── intent helpers ───────────────────────────────────────────────────
    def test_affirmative(self):
        for w in ("yes", "Yeah", "correct", "that's right", "go ahead"):
            self.assertTrue(self.mod._affirmative(w))
        self.assertFalse(self.mod._affirmative("no"))
        self.assertFalse(self.mod._affirmative(""))

    def test_negative(self):
        for w in ("no", "Nope", "wrong", "cancel"):
            self.assertTrue(self.mod._negative(w))
        self.assertFalse(self.mod._negative("yes"))
        self.assertFalse(self.mod._negative(""))

    # ── small formatters ─────────────────────────────────────────────────
    def test_format_digits_for_speech(self):
        self.assertEqual(self.mod._format_digits_for_speech("12345678"),
                         "1-2-3-4-5-6-7-8")

    def test_humanise_printer(self):
        self.assertEqual(
            self.mod._humanise_printer({"name": "My H2D", "ip": "10.0.0.5"}),
            "My H2D at 10.0.0.5")
        # Falls back to model, then "printer".
        self.assertEqual(
            self.mod._humanise_printer({"model": "BL-P001", "ip": "10.0.0.5"}),
            "BL-P001 at 10.0.0.5")
        self.assertEqual(self.mod._humanise_printer({"ip": "10.0.0.5"}),
                         "printer at 10.0.0.5")

    # ── _parse_inline_args ───────────────────────────────────────────────
    def test_inline_args_three_tokens(self):
        self.assertEqual(
            self.mod._parse_inline_args("192.168.1.5 12345678 01P00A99"),
            ("192.168.1.5", "12345678", "01P00A99"))

    def test_inline_args_two_tokens_serial_blank(self):
        self.assertEqual(self.mod._parse_inline_args("192.168.1.5 12345678"),
                         ("192.168.1.5", "12345678", ""))

    def test_inline_args_rejects_bad_ip(self):
        self.assertIsNone(self.mod._parse_inline_args("not-an-ip 12345678"))

    def test_inline_args_rejects_bad_access_code(self):
        # access code must be 6-12 digits
        self.assertIsNone(self.mod._parse_inline_args("192.168.1.5 abc"))

    def test_inline_args_rejects_wrong_token_count(self):
        self.assertIsNone(self.mod._parse_inline_args("192.168.1.5"))
        self.assertIsNone(self.mod._parse_inline_args(""))


class BambuSetupWizardPickTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_single_printer_confirmed(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="yes"):
            chosen = self.mod._wizard_pick_printer([{"ip": "10.0.0.1",
                                                     "name": "H2D"}])
        self.assertEqual(chosen["ip"], "10.0.0.1")

    def test_single_printer_declined(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="no"):
            self.assertIsNone(
                self.mod._wizard_pick_printer([{"ip": "10.0.0.1"}]))

    def test_multiple_pick_by_index(self):
        printers = [{"ip": "10.0.0.1", "name": "A"},
                    {"ip": "10.0.0.2", "name": "B"}]
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="two"):
            chosen = self.mod._wizard_pick_printer(printers)
        self.assertEqual(chosen["ip"], "10.0.0.2")

    def test_multiple_pick_by_model_name(self):
        printers = [{"ip": "10.0.0.1", "name": "Prusa"},
                    {"ip": "10.0.0.2", "name": "H2D"}]
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               return_value="the h2d please"):
            chosen = self.mod._wizard_pick_printer(printers)
        self.assertEqual(chosen["ip"], "10.0.0.2")


class BambuSetupActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_setup_inline_args_persists_and_restarts(self):
        # Stub the two side-effecting helpers so NO real source file is
        # rewritten and no monitor is actually started.
        with mock.patch.object(self.mod, "_persist_credentials",
                               return_value=True) as persist, \
             mock.patch.object(self.mod, "_restart_monitor", return_value=True):
            out = self.actions["setup_printer"](
                "192.168.1.50 12345678 01P00A0001")
        persist.assert_called_once_with("192.168.1.50", "12345678", "01P00A0001")
        self.assertIn("online", out.lower())

    def test_setup_inline_args_persist_ok_monitor_idle(self):
        with mock.patch.object(self.mod, "_persist_credentials",
                               return_value=True), \
             mock.patch.object(self.mod, "_restart_monitor", return_value=False):
            out = self.actions["setup_printer"]("192.168.1.50 12345678")
        self.assertIn("next launch", out.lower())

    def test_setup_inline_args_persist_failure(self):
        with mock.patch.object(self.mod, "_persist_credentials",
                               return_value=False), \
             mock.patch.object(self.mod, "_restart_monitor", return_value=True):
            out = self.actions["setup_printer"]("192.168.1.50 12345678")
        self.assertIn("couldn't write the credentials", out.lower())

    def test_setup_lock_blocks_concurrent_run(self):
        # Hold the wizard lock so the action reports the busy message instead of
        # entering the voice flow.
        self.assertTrue(self.mod._wizard_lock.acquire(blocking=False))
        try:
            out = self.actions["setup_printer"]("")
        finally:
            self.mod._wizard_lock.release()
        self.assertIn("already running", out.lower())

    def test_all_action_aliases_registered(self):
        for name in ("setup_printer", "setup_bambu", "configure_printer",
                     "bambu_setup", "first_time_printer_setup"):
            self.assertIn(name, self.actions)


if __name__ == "__main__":
    unittest.main()
