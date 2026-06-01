"""Logic tests for jarvis_failure_lines.py — the error→class classifier and
in-character line banks the dispatcher uses to narrate failures.

Coverage targets the real decision logic: which failure CLASS a given
exception/message maps to (timeouts, permission, network, parse, COM, UI,
IO, app-not-found, the FileNotFoundError disambiguation, and the generic
fallback), the most-specific-first precedence of the pattern table, and that
every drawn line actually belongs to its class's bank.

jarvis_failure_line() draws randomly, so tests assert membership in the
correct bank rather than a single fixed string.
"""
from __future__ import annotations

import unittest

import jarvis_failure_lines as fl


class ClassifyByExceptionTypeTests(unittest.TestCase):
    """Real exception instances → class name + message drive the classifier."""

    def test_timeout(self):
        self.assertEqual(fl.classify_failure(TimeoutError("took too long")),
                         "timeout")

    def test_permission(self):
        self.assertEqual(fl.classify_failure(PermissionError("access is denied")),
                         "permission")

    def test_network(self):
        self.assertEqual(fl.classify_failure(ConnectionError("connection refused")),
                         "network")

    def test_parse(self):
        # ValueError text 'invalid literal' is the parse signal.
        self.assertEqual(fl.classify_failure(ValueError("invalid literal for int")),
                         "parse")

    def test_io(self):
        self.assertEqual(fl.classify_failure(OSError("disk full")), "io")

    def test_generic_unknown(self):
        self.assertEqual(fl.classify_failure(RuntimeError("something weird")),
                         "unknown")


class ClassifyByStringTests(unittest.TestCase):
    """Raw error strings (no exception object) classify off text alone."""

    def test_timeout_variants(self):
        self.assertEqual(fl.classify_failure("TimeoutExpired: cmd"), "timeout")
        self.assertEqual(fl.classify_failure("operation timed out"), "timeout")

    def test_permission_winerror5(self):
        self.assertEqual(fl.classify_failure("WinError 5 access denied"),
                         "permission")

    def test_network_dns(self):
        self.assertEqual(fl.classify_failure("getaddrinfo failed"), "network")
        self.assertEqual(fl.classify_failure("Max retries exceeded with url"),
                         "network")

    def test_parse_json(self):
        self.assertEqual(fl.classify_failure("JSONDecodeError: line 1 column 1"),
                         "parse")

    def test_com(self):
        self.assertEqual(fl.classify_failure("com_error HRESULT 0x80004005"),
                         "com")

    def test_ui_automation(self):
        self.assertEqual(
            fl.classify_failure("pyautogui ImageNotFoundException"),
            "ui_automation")

    def test_empty_and_none_are_unknown(self):
        self.assertEqual(fl.classify_failure(""), "unknown")
        self.assertEqual(fl.classify_failure(None), "unknown")
        self.assertEqual(fl.classify_failure("totally benign message"), "unknown")


class FileNotFoundDisambiguationTests(unittest.TestCase):
    """FileNotFoundError is ambiguous; an app hint in the action/message bumps
    it to app_not_found, otherwise it stays IO."""

    def test_app_hint_via_action_name(self):
        self.assertEqual(
            fl.classify_failure(FileNotFoundError("cannot find the file"),
                                "play_music"),
            "app_not_found")

    def test_app_hint_via_exe_in_message(self):
        self.assertEqual(
            fl.classify_failure("WinError 2 cannot find the file",
                                "launch chrome.exe"),
            "app_not_found")

    def test_plain_file_miss_is_io(self):
        self.assertEqual(
            fl.classify_failure(FileNotFoundError("no such file"), "read_log"),
            "io")

    def test_winerror2_without_app_hint_is_io(self):
        self.assertEqual(
            fl.classify_failure("WinError 2 system cannot find", "save_state"),
            "io")


class PrecedenceTests(unittest.TestCase):
    """The pattern table is ordered most-specific → most-general; the first
    matching class wins. These pin the documented ordering."""

    def test_timeout_beats_network(self):
        # Text carries BOTH a timeout and a connection signal; timeout is
        # listed first, so it must win.
        self.assertEqual(fl.classify_failure("ConnectionError: read timed out"),
                         "timeout")

    def test_permission_beats_network(self):
        self.assertEqual(
            fl.classify_failure("permission denied while ConnectionRefused"),
            "permission")


class LineBankTests(unittest.TestCase):
    def test_every_class_draws_from_its_own_bank(self):
        for klass in fl.all_classes():
            line = fl.jarvis_failure_line(klass)
            self.assertIn(line, fl._LINES[klass],
                          msg=f"line for {klass!r} not in its bank")

    def test_unknown_class_falls_back_to_unknown_bank(self):
        line = fl.jarvis_failure_line("no_such_class")
        self.assertIn(line, fl._LINES["unknown"])

    def test_all_classes_lists_every_bank(self):
        self.assertEqual(
            set(fl.all_classes()),
            {"network", "permission", "parse", "app_not_found", "timeout",
             "com", "ui_automation", "io", "unknown"},
        )

    def test_banks_are_nonempty(self):
        for klass in fl.all_classes():
            self.assertTrue(fl._LINES[klass],
                            msg=f"{klass} bank is empty")

    def test_lines_in_character(self):
        # Spot-check the JARVIS voice: most network lines address 'sir'.
        self.assertTrue(any("sir" in ln for ln in fl._LINES["network"]))


class FailureMessageTests(unittest.TestCase):
    """failure_message() = classify + draw line + technical detail tuple."""

    def test_tuple_for_exception(self):
        klass, line, technical = fl.failure_message(TimeoutError("boom"),
                                                    "do_x")
        self.assertEqual(klass, "timeout")
        self.assertIn(line, fl._LINES["timeout"])
        # Technical suffix preserves the raw exception type + message.
        self.assertEqual(technical, "TimeoutError: boom")

    def test_tuple_for_string(self):
        klass, line, technical = fl.failure_message("plain string error", "act")
        self.assertEqual(klass, "unknown")
        self.assertIn(line, fl._LINES["unknown"])
        self.assertEqual(technical, "plain string error")

    def test_none_message_technical_is_empty(self):
        klass, line, technical = fl.failure_message(None, "")
        self.assertEqual(klass, "unknown")
        self.assertEqual(technical, "")

    def test_permission_exception_routes_and_keeps_detail(self):
        klass, line, technical = fl.failure_message(
            PermissionError("WinError 5"), "delete_file")
        self.assertEqual(klass, "permission")
        self.assertIn(line, fl._LINES["permission"])
        self.assertIn("PermissionError", technical)


if __name__ == "__main__":
    unittest.main()
