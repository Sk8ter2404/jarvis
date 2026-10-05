"""tools/hermetic_guard.py - the 2026-10-05 screen READERS: a UI Automation
client (it reads every window's link names and titles) and the screen OCR
worker are refused in a test process, like a screen capture.

    python -m unittest tests.test_hermetic_guard_screen_readers
"""
from __future__ import annotations

import unittest

from tools import hermetic_guard as H


class UiaVerdictTests(unittest.TestCase):
    def test_the_uia_classes_are_refused(self):
        class CUIAutomation8:
            pass

        class CUIAutomation:
            pass
        self.assertTrue(H.uia_verdict((CUIAutomation8,)))
        self.assertTrue(H.uia_verdict((CUIAutomation,)))
        self.assertTrue(H.uia_verdict(
            ("{E22AD333-B25F-460C-83D0-0581107395C9}",)))

    def test_other_com_objects_pass(self):
        self.assertIsNone(H.uia_verdict(("Shell.Application",)))
        self.assertIsNone(H.uia_verdict(()))

    def test_the_create_object_entry_point_is_wrapped_under_the_screen_guard(self):
        self.assertIn(("screen", "comtypes.client", None, "CreateObject"),
                      H._wrapped_targets())
        self.assertIn("comtypes.client", H._wrapped_modules())
        self.assertIs(H._verdict_for("comtypes.client", None, "CreateObject"),
                      H.uia_verdict)


class OcrWorkerVerdictTests(unittest.TestCase):
    def test_the_ocr_worker_is_refused(self):
        why = H.probe_verdict("powershell", [
            "powershell", "-NoLogo", "-File", "C:/x/tools/ocr_worker.ps1"])
        self.assertIn("OCR worker", why)

    def test_other_powershell_runs_are_unchanged(self):
        self.assertIsNone(H.probe_verdict("powershell", [
            "powershell", "-Command", "Get-Date"]))


if __name__ == "__main__":
    unittest.main()
