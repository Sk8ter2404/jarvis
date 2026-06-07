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


class ClampFactLenTests(unittest.TestCase):
    def test_normal_fact_unchanged(self):
        # A real durable fact is well under the cap and must pass through
        # byte-for-byte, with no truncation marker appended.
        for fact in [
            "User's name is Alex",
            "User is building a REPO animatronic robot",
            "User has a Bambu H2D printer",
            "",
        ]:
            self.assertEqual(mg._clamp_fact_len(fact), fact)

    def test_overlong_fact_is_truncated(self):
        long_fact = "word " * 200  # ~1000 chars of real words
        out = mg._clamp_fact_len(long_fact)
        self.assertLessEqual(len(out), mg.MAX_FACT_LEN + 1)  # +1 for the '…'
        self.assertTrue(out.endswith("…"))
        self.assertNotEqual(out, long_fact)

    def test_truncates_on_word_boundary_when_close(self):
        # The window ends mid-word; the cut should back up to the last space so
        # no half-word is stored. Use a long unique trailing word we can detect.
        text = ("alpha " * 60) + "TRAILINGWORDTHATSHOULDBECUT"
        out = mg._clamp_fact_len(text)
        self.assertNotIn("TRAILINGWORD", out)   # the chopped word is gone whole
        self.assertFalse(out[:-1].endswith(" "))  # trailing space stripped
        self.assertTrue(out.endswith("…"))

    def test_unbroken_token_hard_cut(self):
        # A single 500-char garbage token with no usable word boundary must
        # still be capped (hard cut), not left at full length.
        token = "x" * 500
        out = mg._clamp_fact_len(token)
        self.assertEqual(len(out), mg.MAX_FACT_LEN + 1)  # 300 chars + '…'
        self.assertTrue(out.endswith("…"))

    def test_respects_explicit_cap(self):
        out = mg._clamp_fact_len("y" * 50, cap=10)
        self.assertEqual(out, ("y" * 10) + "…")

    def test_non_string_passthrough(self):
        # Defensive: never raise on unexpected types (e.g. a stray None in the
        # facts list); return as-is so the caller's own type filtering applies.
        for val in (None, 123, ["a"]):
            self.assertEqual(mg._clamp_fact_len(val), val)


if __name__ == "__main__":
    unittest.main()
