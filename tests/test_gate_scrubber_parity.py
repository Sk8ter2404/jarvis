"""Parity gate: tools/check_no_pii.py HARD vs core/bug_reporter.py's scrubber.

WHY THIS FILE EXISTS
--------------------
``tools/check_no_pii.py`` is the ONLY thing standing between the owner's private
data and a public push (pre-commit hook, CI step, ``tools/build_release.py``).
``core/bug_reporter.py`` claims in its own module docstring that "the gate and
the reporter redact by the same rules" -- but the sharing is ONE-WAY:
bug_reporter IMPORTS ``HARD``/``WARN`` from the gate (``_gate_patterns()``), so
shapes added to the gate reach the scrubber, while shapes added only to
``bug_reporter._SCRUB_RULES`` never flow back.

That asymmetry is how the gate silently fell behind on GitHub tokens (found
2026-08-20): the scrubber knew ``ghp_`` / ``gho_`` / ``github_pat_`` / ``xox*``
/ JWT / Slack-webhook shapes and the gate knew NONE of them -- not even as an
advisory WARN -- while the TRACKED ``.env.example`` ships a
``JARVIS_GITHUB_TOKEN=`` line the user is told to fill in and
``bug_reporter._issue_token()`` reads exactly that variable.

These tests fail the moment the two drift apart again. That is the whole point:
this repo's #1 bug class is a rule fixed in one copy while the others rot.

PRIVACY / SELF-GATE SAFETY
--------------------------
The gate scans ``tests/`` too, so every credential-shaped fixture below is
ASSEMBLED AT RUNTIME from harmless pieces. No line of this source is itself a
HARD match -- that convention is already used by ``tests/test_check_no_pii.py``
and ``tests/test_bug_reporter.py`` and must be preserved here.

CI-safety: stdlib only, plus two import-light project modules. No monolith boot,
no hardware, no network.
"""
from __future__ import annotations

import unittest

import tools.check_no_pii as cnp
from core import bug_reporter

_SEED = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"


def _b62(n: int) -> str:
    """A deterministic n-char base62 run (never a real secret)."""
    return (_SEED * ((n // len(_SEED)) + 1))[:n]


# --- runtime-assembled fixtures: label -> sample -------------------------------
# Each is a WELL-FORMED example of a live credential shape the runtime scrubber
# redacts. Every one of these must ALSO be a HARD finding for the commit gate.
CREDENTIAL_SHAPES = {
    "github-classic-pat":  "ghp" + "_" + _b62(36),
    "github-oauth":        "gho" + "_" + _b62(36),
    "github-server-token": "ghs" + "_" + _b62(36),
    "github-fine-grained": "github" + "_pat_" + _b62(22) + "_" + _b62(59),
    "anthropic-key":       "sk-" + "ant-api03-" + _b62(24),
    "openai-legacy-key":   "sk-" + _b62(48),
    "openai-project-key":  "sk-" + "proj-" + _b62(48),
    "openai-svcacct-key":  "sk-" + "svcacct-" + _b62(48),
    "aws-access-key":      "AKIA" + "QWERTYUIOPASDFGH",
    "google-api-key":      "AIza" + _b62(35),
    "slack-bot-token":     "xox" + "b-" + "123456789012" + "-" + "1234567890123"
                           + "-" + _b62(24),
    "slack-webhook":       "https://hooks.slack.com/" + "services/"
                           + "T00ABC123" + "/" + "B01DEF456" + "/" + _b62(24),
    "jwt":                 "ey" + "JhbGciOiJIUzI1NiJ9" + "."
                           + "ey" + "JzdWIiOiIxMjM0NTY3ODkwIn0" + "."
                           + "dBjftJeZ4CVPmB92K27uhbUJU1p1r-wW1gFWFOEjXk",
}

# Text that must never trip a HARD rule. Includes the near-miss forms this repo
# already carries as SOURCE literals, so a future widening that would break the
# commit gate fails HERE instead of on the owner's next push.
BENIGN = [
    "the quick brown fox writes alice to 10.0.0.5",
    "sk-project-manager is a job title, not a key",
    "ghp" + "_short",
    "github" + "_pat_placeholder",
    "xox" + "b-12345678abcd",                                    # test_bug_reporter fixture
    "https://hooks.slack.com/" + "services/T00/B00/abcXYZ123",   # ditto
    "eyJ is just base64 for an opening brace",
    "JARVIS_GITHUB_TOKEN=",                                      # the .env.example line
]


def _hard_labels(text):
    return [label for label, rx in cnp.HARD if rx.search(text)]


class GateCoversEveryScrubbedShape(unittest.TestCase):
    """Direction 1: anything the runtime scrubber hides, the commit gate blocks."""

    def test_every_credential_shape_is_a_hard_finding(self):
        for name, sample in CREDENTIAL_SHAPES.items():
            with self.subTest(shape=name):
                self.assertTrue(
                    _hard_labels(sample),
                    "%s: no HARD pattern in tools/check_no_pii.py matches this "
                    "credential shape, so it could be committed to a PUBLIC "
                    "repo. Add the shape to check_no_pii.HARD -- NOT only to "
                    "bug_reporter._SCRUB_RULES, that direction does not flow "
                    "back." % name)

    def test_scrubber_redacts_every_credential_shape(self):
        """Direction 2: and the reporter still hides everything the gate detects."""
        for name, sample in CREDENTIAL_SHAPES.items():
            with self.subTest(shape=name):
                out = bug_reporter.scrub("context before %s context after" % sample)
                self.assertNotIn(sample, out,
                                 "%s: survived core.bug_reporter.scrub()" % name)

    def test_gate_hard_patterns_are_the_objects_the_reporter_uses(self):
        """The "one source of truth" claim is wiring, not prose -- prove the wiring.

        ``bug_reporter._gate_patterns()[1]`` must contain the very same compiled
        pattern OBJECTS as ``cnp.HARD``; identity (not equality) is what proves
        the reporter is reading the gate rather than holding a private copy.
        """
        hard_only = bug_reporter._gate_patterns()[1]
        by_id = set(id(rx) for rx in hard_only)
        missing = [label for label, rx in cnp.HARD if id(rx) not in by_id]
        self.assertEqual(
            missing, [],
            "these gate HARD patterns never reached core.bug_reporter -- the "
            "import inside _gate_patterns() is broken, or its cache was warmed "
            "before they were registered")


class NoFalsePositives(unittest.TestCase):
    """A gate that cries wolf gets bypassed with --no-verify. Keep it precise."""

    def test_benign_text_produces_no_hard_finding(self):
        for text in BENIGN:
            with self.subTest(text=text[:48]):
                self.assertEqual(
                    _hard_labels(text), [],
                    "this benign string would HARD-fail the commit gate")

    def test_repo_fixture_shapes_stay_below_the_hard_bar(self):
        """The two literals tests/test_bug_reporter.py writes as SOURCE must stay
        non-matching, or these rules break every commit in the repo."""
        self.assertEqual(_hard_labels("xox" + "b-12345678abcd"), [])
        self.assertEqual(
            _hard_labels("https://hooks.slack.com/"
                         + "services/T00/B00/abcXYZ123"), [])


class WarnCoversBareTokenNames(unittest.TestCase):
    """``GITHUB_TOKEN = "..."`` used to produce NOTHING: the WARN alternation
    required an ``auth`` prefix that neither sibling copy of the same name list
    (core/bug_reporter.py, tools/audit_codebase.py) requires."""

    def _secret_literal(self):
        for label, rx in cnp.WARN:
            if label == "secret-literal":
                return rx
        self.fail("WARN label 'secret-literal' not present")

    def test_bare_token_name_is_advisory(self):
        rx = self._secret_literal()
        for name in ("GITHUB_TOKEN", "token", "auth_token", "AUTH-TOKEN",
                     "api_key", "password", "access-code", "secret"):
            with self.subTest(name=name):
                self.assertTrue(rx.search(name + ' = "longenoughvalue"'),
                                "%s should raise an advisory WARN" % name)

    def test_short_value_still_ignored(self):
        # Unchanged behaviour: quoted values under 6 chars stay below the bar.
        self.assertFalse(self._secret_literal().search('token = "12345"'))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    unittest.main()
