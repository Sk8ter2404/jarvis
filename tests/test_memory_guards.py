"""Tests for core.memory_guards — the write-time filters that keep secrets and
internal noise OUT of bobert_memory.json (which is sent to the cloud LLM every
turn). Security-critical: a regression here re-opens the credential-leak hole
the guards were added to close, so these pin the exact contract."""
import unittest

import core.memory_guards as mg


class SecretFactTests(unittest.TestCase):
    def test_catches_credentials(self):
        for fact in [
            "User's password for Deco login is hunter2",
            "the API key is sk-abc123",
            "my api_key: xyz",
            "my SSN is 123-45-6789",
            "his social security number is 000",
            "wifi passphrase is opensesame",
            "the access code is 4821",
            "credit card 4111 1111 1111 1111",
            "the cvv is 123",
            "account number 000123456",
            "routing number 021000021",
            "auth token xyz",
            "store this secret value",
            "her login credential",
        ]:
            self.assertTrue(mg._is_secret_fact(fact), f"should be secret: {fact!r}")

    def test_allows_normal_facts(self):
        for fact in [
            "User's name is Alex",
            "User is building a REPO animatronic robot",
            "User likes lofi music while working",
            "User has a Bambu H2D printer",
            "",
        ]:
            self.assertFalse(mg._is_secret_fact(fact), f"should be allowed: {fact!r}")


class InternalNoiseTests(unittest.TestCase):
    def test_catches_internal_artifacts(self):
        for text in [
            "Running diagnostics on anomaly-1780175157",
            "anomaly-5 investigation",
            "[regression] fix the wake word",
            "[overnight] upgrade pass",
            "[self-diag] probe",
            "unhandled exception in foo",
            "traceback most recent call last",
            "None", "n/a", "  null  ", "unknown", "",
        ]:
            self.assertTrue(mg._is_internal_noise_fact(text), f"should be noise: {text!r}")

    def test_allows_real_memory(self):
        for text in [
            "Building a REPO animatronic robot",
            "User uses Apple Music",
            "Alex works late nights on builds",
        ]:
            self.assertFalse(mg._is_internal_noise_fact(text), f"should be kept: {text!r}")


if __name__ == "__main__":
    unittest.main()
