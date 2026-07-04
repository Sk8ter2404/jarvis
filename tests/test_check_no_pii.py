"""Tests for tools/check_no_pii.py — the PII / secret leak gate.

This is the hard gate that greps the git-tracked / staged set (or a directory)
for secret + owner-PII regex patterns and exits 1 on a HARD hit. It runs before
the baseline commit, in CI, and inside tools/build_release.py, so a regression
here could let real credentials ship — it is load-bearing.

CI-safety: the module top-level-imports stdlib only, so importing it is safe on
the bare Linux runner. Every test here either mocks subprocess (no real git) or
works entirely inside a TemporaryDirectory (no real-repo mutation). No test
depends on the host OS; the few Windows-path-specific assertions are guarded.

Privacy: this test file must itself survive the repo's own check_no_pii gate
(it scans tests/ in CI). So any string that would match a HARD pattern is built
at runtime via concatenation, never written as a source literal. Non-secret
fixtures use "alice" / "10.0.0.5".
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import tools.check_no_pii as cnp


# --- runtime-built fixture strings (so no HARD literal sits in this source) ---
# Each is assembled from harmless pieces so the file itself never trips the gate.
FAKE_OPENAI_KEY = "sk-" + ("A1b2" * 8)                  # sk- + 32 chars  -> openai-key
FAKE_ANTHROPIC_KEY = "sk-ant-" + ("Z9y8x7w6" * 3)        # sk-ant- + 24    -> anthropic-key
FAKE_AWS_KEY = "AKIA" + ("Q" * 16)                       # AKIA + 16       -> aws-access-key
FAKE_GOOGLE_KEY = "AIza" + ("k" * 35)                    # AIza + 35       -> google-api-key
FAKE_SECRET_ASSIGN = 'password = "' + ("hunter2plus") + '"'   # secret-literal WARN
# (private-key-block and private-ip are exercised inline / via shipped-regex
#  variant tests below, so no module-level fixture is needed for them.)


def _isolated_rules():
    """A small, deterministic rule set independent of the host's pii_local.py.

    The dev box loads tools/pii_local.py at import (extending HARD/WARN), so we
    never assert against the module-level lists' contents/length; instead we
    build our own (label, compiled-regex) rules via the public-ish _rx helper.
    """
    return [
        ("fake-openai", cnp._rx(r"\bsk-[A-Za-z0-9]{32,}\b")),
        ("fake-token", cnp._rx(r"TOKEN_[A-Z0-9]{6,}")),
    ]


class RxHelperTests(unittest.TestCase):
    def test_rx_compiles_pattern(self):
        rx = cnp._rx(r"ab+c")
        self.assertTrue(rx.search("xxabbbcyy"))
        self.assertFalse(rx.search("xxac... wait no"))

    def test_rx_returns_pattern_object(self):
        rx = cnp._rx(r"\d+")
        self.assertIsInstance(rx, type(cnp.re.compile("")))


class RedactTests(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(cnp._redact("   hello world   "), "hello world")

    def test_non_ascii_replaced_not_crashed(self):
        out = cnp._redact("café — 日本語 — 🤖")
        # Every char must be ASCII (encode('ascii') would otherwise raise).
        out.encode("ascii")
        self.assertNotIn("é", out)
        self.assertIn("?", out)

    def test_short_line_not_truncated(self):
        s = "a" * 120
        self.assertEqual(cnp._redact(s), s)
        self.assertFalse(cnp._redact(s).endswith(" ..."))

    def test_long_line_truncated_with_ellipsis(self):
        s = "b" * 200
        out = cnp._redact(s)
        self.assertTrue(out.endswith(" ..."))
        self.assertEqual(out, "b" * 120 + " ...")

    def test_truncation_boundary_is_121(self):
        # len==120 -> kept; len==121 -> truncated (strictly greater-than).
        self.assertFalse(cnp._redact("c" * 120).endswith(" ..."))
        self.assertTrue(cnp._redact("c" * 121).endswith(" ..."))


class ScanFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, data, mode="w", encoding="utf-8"):
        path = os.path.join(self.dir, name)
        if "b" in mode:
            with open(path, mode) as fh:
                fh.write(data)
        else:
            with open(path, mode, encoding=encoding) as fh:
                fh.write(data)
        return path

    def test_clean_file_no_hits(self):
        path = self._write("clean.py", "x = 1\nname = 'alice'\nip = '10.0.0.5'\n")
        self.assertEqual(cnp._scan_file(path, _isolated_rules()), [])

    def test_hit_reports_label_line_and_snippet(self):
        path = self._write("leak.txt", "line one ok\nkey = " + FAKE_OPENAI_KEY + "\n")
        hits = cnp._scan_file(path, _isolated_rules())
        self.assertEqual(len(hits), 1)
        label, lineno, snip = hits[0]
        self.assertEqual(label, "fake-openai")
        self.assertEqual(lineno, 2)
        self.assertIn("sk-", snip)

    def test_multiple_rules_and_lines(self):
        body = "a TOKEN_ABC123 here\nnothing\nb " + FAKE_OPENAI_KEY + "\n"
        path = self._write("multi.txt", body)
        hits = cnp._scan_file(path, _isolated_rules())
        labels = sorted(h[0] for h in hits)
        self.assertEqual(labels, ["fake-openai", "fake-token"])

    def test_line_numbers_are_one_based(self):
        path = self._write("nums.txt", "\n\nTOKEN_AAAAAA\n")
        hits = cnp._scan_file(path, _isolated_rules())
        self.assertEqual(hits[0][1], 3)

    def test_skips_own_basename_check_no_pii(self):
        # A file literally named check_no_pii.py is always skipped, even if its
        # body would match — its pattern strings self-match in dist/ copies.
        path = self._write("check_no_pii.py", "TOKEN_ZZZZZZ\n" + FAKE_OPENAI_KEY + "\n")
        self.assertEqual(cnp._scan_file(path, _isolated_rules()), [])

    def test_skips_self_path(self):
        # The real module path is in _SELF and must be skipped regardless of rules.
        self.assertEqual(cnp._scan_file(cnp.__file__, _isolated_rules()), [])

    def test_skips_pii_local_path_in_self(self):
        # pii_local.py (sibling of the module) is registered in _SELF.
        pii_local = os.path.join(os.path.dirname(os.path.abspath(cnp.__file__)),
                                 "pii_local.py")
        norm = os.path.normcase(os.path.abspath(pii_local))
        self.assertIn(norm, cnp._SELF)

    def test_skips_binary_extension(self):
        # .png is in _SKIP_EXT -> returned empty before any read.
        path = self._write("image.png", "TOKEN_ABCDEF\n")
        self.assertEqual(cnp._scan_file(path, _isolated_rules()), [])

    def test_skip_ext_is_case_insensitive(self):
        path = self._write("ASSET.PNG", "TOKEN_ABCDEF\n")
        self.assertEqual(cnp._scan_file(path, _isolated_rules()), [])

    def test_binary_null_byte_content_skipped(self):
        # A .txt whose first 4KB contains a NUL is treated as binary -> skipped.
        path = self._write("blob.txt", b"TOKEN_ABCDEF\x00more\n", mode="wb")
        self.assertEqual(cnp._scan_file(path, _isolated_rules()), [])

    def test_null_byte_after_4096_still_scanned(self):
        # NUL only past the 4096-byte sniff window -> still scanned as text.
        prefix = ("x" * 5000) + "\nTOKEN_LATER1\n"   # 6 chars after underscore
        data = prefix.encode("utf-8") + b"\x00tail"
        path = self._write("late_null.txt", data, mode="wb")
        hits = cnp._scan_file(path, _isolated_rules())
        self.assertTrue(any(h[0] == "fake-token" for h in hits))

    def test_unreadable_path_returns_empty(self):
        # Nonexistent path -> open() raises OSError -> [] (no crash).
        missing = os.path.join(self.dir, "does_not_exist.txt")
        self.assertEqual(cnp._scan_file(missing, _isolated_rules()), [])

    def test_invalid_utf8_bytes_replaced_not_crashed(self):
        # Lone continuation bytes decode with errors="replace"; still matchable.
        data = b"\xff\xfe bad bytes\nTOKEN_GOODXY\n"   # 6 chars after underscore
        path = self._write("badutf8.txt", data, mode="wb")
        hits = cnp._scan_file(path, _isolated_rules())
        self.assertTrue(any(h[0] == "fake-token" for h in hits))


class HardPatternTests(unittest.TestCase):
    """Exercise each shipped HARD regex with a runtime-built fake match.

    We pull the compiled pattern out of cnp.HARD by label so we test the real
    shipped regex, but we never write a matching literal into this source.
    """
    def _rx_for(self, label):
        for lbl, rx in cnp.HARD:
            if lbl == label:
                return rx
        self.fail(f"HARD label {label!r} not present")

    def test_anthropic_key_matches(self):
        self.assertTrue(self._rx_for("anthropic-key").search(FAKE_ANTHROPIC_KEY))

    def test_openai_key_matches(self):
        self.assertTrue(self._rx_for("openai-key").search(FAKE_OPENAI_KEY))

    def test_openai_key_too_short_no_match(self):
        # 31 chars after sk- -> below the {32,} floor.
        self.assertFalse(self._rx_for("openai-key").search("sk-" + ("a" * 31)))

    def test_aws_key_matches(self):
        self.assertTrue(self._rx_for("aws-access-key").search(FAKE_AWS_KEY))

    def test_aws_key_lowercase_no_match(self):
        # AKIA body must be [0-9A-Z]; lowercase tail should not match.
        self.assertFalse(self._rx_for("aws-access-key").search("AKIA" + ("q" * 16)))

    def test_google_key_matches(self):
        self.assertTrue(self._rx_for("google-api-key").search(FAKE_GOOGLE_KEY))

    def test_private_key_block_matches_variants(self):
        rx = self._rx_for("private-key-block")
        for kind in ("", "RSA ", "EC ", "OPENSSH ", "DSA "):
            self.assertTrue(rx.search("-----BEGIN " + kind + "PRIVATE KEY-----"),
                            f"variant {kind!r} should match")

    def test_hard_clean_text_no_match(self):
        clean = "the quick brown fox writes alice to 10.0.0.5"
        for _lbl, rx in cnp.HARD:
            self.assertFalse(rx.search(clean))


class WarnPatternTests(unittest.TestCase):
    def _rx_for(self, label):
        for lbl, rx in cnp.WARN:
            if lbl == label:
                return rx
        self.fail(f"WARN label {label!r} not present")

    def test_secret_literal_assignment_matches(self):
        self.assertTrue(self._rx_for("secret-literal").search(FAKE_SECRET_ASSIGN))

    def test_secret_literal_short_value_no_match(self):
        # value must be 6+ chars inside quotes; 5 chars -> no match.
        self.assertFalse(self._rx_for("secret-literal").search('secret = "12345"'))

    def test_secret_literal_keyword_variants(self):
        rx = self._rx_for("secret-literal")
        for kw in ("password", "api_key", "api-key", "auth_token", "access-code"):
            self.assertTrue(rx.search(kw + ' = "longenough"'), f"{kw} should match")

    def test_private_ip_10_block(self):
        self.assertTrue(self._rx_for("private-ip").search("host 10.1.2.3 end"))

    def test_private_ip_192_168_block(self):
        self.assertTrue(self._rx_for("private-ip").search("gw 192.168.0.1"))

    def test_private_ip_172_16_31_block(self):
        rx = self._rx_for("private-ip")
        self.assertTrue(rx.search("172.16.5.5"))
        self.assertTrue(rx.search("172.31.5.5"))

    def test_private_ip_public_no_match(self):
        rx = self._rx_for("private-ip")
        # 172.15 and 172.32 are outside the private 16-31 range; 8.8.8.8 public.
        self.assertFalse(rx.search("172.15.0.1"))
        self.assertFalse(rx.search("172.32.0.1"))
        self.assertFalse(rx.search("8.8.8.8"))


class WalkTextFilesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, *parts):
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("data\n")
        return path

    def test_collects_text_files(self):
        self._touch("a.py")
        self._touch("sub", "b.txt")
        found = cnp._walk_text_files(self.root)
        names = sorted(os.path.basename(p) for p in found)
        self.assertEqual(names, ["a.py", "b.txt"])

    def test_skips_binary_extensions(self):
        self._touch("keep.py")
        self._touch("drop.png")
        self._touch("drop.zip")
        found = [os.path.basename(p) for p in cnp._walk_text_files(self.root)]
        self.assertIn("keep.py", found)
        self.assertNotIn("drop.png", found)
        self.assertNotIn("drop.zip", found)

    def test_prunes_excluded_dirs(self):
        self._touch("real.py")
        self._touch(".git", "config")
        self._touch("__pycache__", "x.py")
        self._touch("node_modules", "pkg.js")
        self._touch("backups", "old.py")
        found = [os.path.relpath(p, self.root) for p in cnp._walk_text_files(self.root)]
        self.assertIn("real.py", found)
        for bad in (".git", "__pycache__", "node_modules", "backups"):
            self.assertFalse(any(f.startswith(bad + os.sep) for f in found),
                             f"{bad} should be pruned; got {found}")


class TrackedFilesTests(unittest.TestCase):
    """_tracked_files() shells out to git — fully mocked here (no real git)."""

    def test_parses_git_output_to_abs_paths(self):
        fake = mock.Mock(returncode=0, stdout="core/a.py\n  skills/b.py  \n\n")
        with mock.patch.object(cnp.subprocess, "run", return_value=fake) as run:
            files = cnp._tracked_files()
        run.assert_called_once()
        # blank line dropped; entries joined onto the project root.
        self.assertEqual(len(files), 2)
        self.assertTrue(all(os.path.isabs(p) for p in files))
        # git emits forward slashes; os.path.join keeps them verbatim on the
        # tail, so compare on basename + presence of the relative segment.
        self.assertEqual(os.path.basename(files[0]), "a.py")
        self.assertEqual(os.path.basename(files[1]), "b.py")
        self.assertIn("core", files[0])
        self.assertIn("skills", files[1])
        # the leading whitespace on the 2nd entry must have been stripped.
        self.assertNotIn("  ", files[1])

    def test_uses_expected_git_args(self):
        fake = mock.Mock(returncode=0, stdout="")
        with mock.patch.object(cnp.subprocess, "run", return_value=fake) as run:
            cnp._tracked_files()
        args = run.call_args[0][0]
        self.assertEqual(args[:2], ["git", "ls-files"])
        self.assertIn("--cached", args)
        self.assertIn("--others", args)
        self.assertIn("--exclude-standard", args)

    def test_nonzero_returncode_returns_none(self):
        fake = mock.Mock(returncode=128, stdout="")
        with mock.patch.object(cnp.subprocess, "run", return_value=fake):
            self.assertIsNone(cnp._tracked_files())

    def test_subprocess_exception_returns_none(self):
        with mock.patch.object(cnp.subprocess, "run",
                               side_effect=FileNotFoundError("no git")):
            self.assertIsNone(cnp._tracked_files())

    def test_empty_output_returns_empty_list(self):
        fake = mock.Mock(returncode=0, stdout="\n  \n")
        with mock.patch.object(cnp.subprocess, "run", return_value=fake):
            self.assertEqual(cnp._tracked_files(), [])


class LoadLocalPatternsTests(unittest.TestCase):
    """_load_local_patterns reads a gitignored sibling and extends HARD/WARN.

    We never touch the real lists destructively: each test snapshots and
    restores cnp.HARD / cnp.WARN, and redirects the read at the source by
    patching os.path.* + builtins.open so no real file is involved.
    """
    def setUp(self):
        self._hard = list(cnp.HARD)
        self._warn = list(cnp.WARN)

    def tearDown(self):
        cnp.HARD[:] = self._hard
        cnp.WARN[:] = self._warn

    def test_absent_file_is_noop(self):
        with mock.patch.object(cnp.os.path, "exists", return_value=False):
            cnp._load_local_patterns()
        self.assertEqual(cnp.HARD, self._hard)
        self.assertEqual(cnp.WARN, self._warn)

    def test_present_file_extends_lists(self):
        src = (
            "HARD = [('x-local-hard', r'LOCALHARD_[0-9]+')]\n"
            "WARN = [('x-local-warn', r'LOCALWARN_[0-9]+')]\n"
        )
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(cnp.os.path, "exists", return_value=True), \
                 mock.patch.object(cnp.os.path, "dirname", return_value=d), \
                 mock.patch("builtins.open", mock.mock_open(read_data=src)):
                cnp._load_local_patterns()
        hard_labels = [l for l, _ in cnp.HARD]
        warn_labels = [l for l, _ in cnp.WARN]
        self.assertIn("x-local-hard", hard_labels)
        self.assertIn("x-local-warn", warn_labels)
        # and the appended regex actually compiled + works
        rx = dict((l, r) for l, r in cnp.HARD)["x-local-hard"]
        self.assertTrue(rx.search("LOCALHARD_42"))

    def test_present_but_unreadable_fails_closed(self):
        # v1.81.0 FAIL-CLOSED: a pii_local.py that EXISTS but can't be read is a
        # different, more dangerous case than "no file present" (the legit
        # fail-open covered by test_absent_file_is_noop). The owner-PII patterns
        # did not load, so rather than pass with a silently degraded scanner the
        # gate exits non-zero to block the commit — the repo syncs via Nextcloud,
        # so a leaked commit is effectively irreversible.
        with mock.patch.object(cnp.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("boom")):
            with self.assertRaises(SystemExit) as cm:
                cnp._load_local_patterns()
        self.assertEqual(cm.exception.code, 2)
        # HARD/WARN are restored by tearDown regardless of the early exit.

    def test_present_but_malformed_fails_closed(self):
        # A present pii_local.py that won't compile/exec must ALSO fail closed
        # (exit non-zero), not silently degrade to generic-only key formats.
        with mock.patch.object(cnp.os.path, "exists", return_value=True), \
             mock.patch("builtins.open",
                        mock.mock_open(read_data="this is not valid python <<<")):
            with self.assertRaises(SystemExit) as cm:
                cnp._load_local_patterns()
        self.assertEqual(cm.exception.code, 2)

    def test_missing_keys_default_to_empty(self):
        # File defines neither HARD nor WARN -> .get(..., []) -> no change.
        with mock.patch.object(cnp.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="OTHER = 1\n")):
            cnp._load_local_patterns()
        self.assertEqual(cnp.HARD, self._hard)
        self.assertEqual(cnp.WARN, self._warn)

    # --- worktree / CI fallback coverage (regression: the gate used to provide
    #     NO owner-PII coverage in a git worktree, where the gitignored sibling
    #     pii_local.py is absent and the single-path probe returned early) ------

    def test_candidates_include_canonical_and_env_fallbacks(self):
        # Beyond the module-relative sibling, the canonical owner checkout and an
        # explicit override env var must be probed so a worktree/CI checkout that
        # lacks the sibling can still load owner patterns from a reachable copy.
        with mock.patch.dict(cnp.os.environ,
                             {"JARVIS_PII_LOCAL": r"D:/elsewhere/pii_local.py"}):
            cands = cnp._local_pattern_candidates()
        joined = [c.replace("\\", "/") for c in cands]
        # sibling first (so a real local file still wins), then the fallbacks
        self.assertTrue(joined[0].endswith("/pii_local.py"))
        self.assertTrue(any(c.endswith("C:/JARVIS/tools/pii_local.py")
                            for c in joined),
                        f"canonical fallback missing from {joined}")
        self.assertIn("D:/elsewhere/pii_local.py", joined)

    def test_candidates_deduped_by_realpath(self):
        # If the env override resolves to the same realpath as the sibling, it
        # must appear only once (no double-loading of the same owner file).
        sibling = os.path.join(
            os.path.dirname(os.path.abspath(cnp.__file__)), "pii_local.py")
        with mock.patch.dict(cnp.os.environ, {"JARVIS_PII_LOCAL": sibling}):
            cands = cnp._local_pattern_candidates()
        keys = [os.path.normcase(os.path.realpath(c)) for c in cands]
        self.assertEqual(len(keys), len(set(keys)), f"dupe realpath in {cands}")

    def test_empty_env_override_skipped(self):
        # An unset/empty JARVIS_PII_LOCAL must not become a "" candidate.
        with mock.patch.dict(cnp.os.environ, {"JARVIS_PII_LOCAL": ""}):
            cands = cnp._local_pattern_candidates()
        self.assertNotIn("", cands)

    def test_fallback_loads_when_sibling_absent(self):
        # The core regression: sibling missing (worktree/CI) but a fallback path
        # exists -> owner patterns STILL register (gate is not a silent no-op).
        src = "HARD = [('fb-hard', r'FALLBACKHIT_[0-9]+')]\n"
        real_exists = os.path.exists
        fb = os.path.join(tempfile.gettempdir(), "pii_local_fallback_probe.py")

        def fake_exists(p):
            # sibling (and anything else) absent; only our fallback path exists
            return os.path.normcase(os.path.abspath(p)) == \
                os.path.normcase(os.path.abspath(fb))

        with mock.patch.object(cnp, "_local_pattern_candidates",
                               return_value=["/nope/pii_local.py", fb]), \
             mock.patch.object(cnp.os.path, "exists", side_effect=fake_exists), \
             mock.patch("builtins.open", mock.mock_open(read_data=src)):
            cnp._load_local_patterns()
        self.assertIn("fb-hard", [l for l, _ in cnp.HARD])
        # nothing earlier in the list silently shadowed the real fallback
        self.assertFalse(real_exists("/nope/pii_local.py"))

    def test_no_candidate_found_warns_on_stderr(self):
        # When NO candidate exists, the gate must say so (visible degradation)
        # instead of silently passing with generic-only patterns.
        import io
        buf = io.StringIO()
        with mock.patch.object(cnp.os.path, "exists", return_value=False), \
             mock.patch.object(cnp.sys, "stderr", buf):
            cnp._load_local_patterns()
        self.assertIn("no pii_local.py found", buf.getvalue())
        # and it remained a no-op for the pattern lists
        self.assertEqual(cnp.HARD, self._hard)
        self.assertEqual(cnp.WARN, self._warn)


class MainDirectoryScanTests(unittest.TestCase):
    """End-to-end main() over a directory, with HARD/WARN swapped for fixtures
    and _PROJECT_ROOT pointed at the temp tree (so relpath is self-contained and
    git is never consulted)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self._hard = list(cnp.HARD)
        self._warn = list(cnp.WARN)
        self._root_attr = cnp._PROJECT_ROOT
        cnp._PROJECT_ROOT = self.root
        # deterministic, local-pattern-independent rule set
        cnp.HARD[:] = [("fx-hard", cnp._rx(r"HARDHIT_[0-9]+"))]
        cnp.WARN[:] = [("fx-warn", cnp._rx(r"WARNHIT_[0-9]+"))]

    def tearDown(self):
        cnp.HARD[:] = self._hard
        cnp.WARN[:] = self._warn
        cnp._PROJECT_ROOT = self._root_attr
        self._tmp.cleanup()

    def _write(self, name, body):
        path = os.path.join(self.root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        return path

    def test_clean_dir_returns_zero(self):
        self._write("ok.py", "value = 'alice'\n")
        with mock.patch("builtins.print") as p:
            rc = cnp.main([self.root])
        self.assertEqual(rc, 0)
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("OK", joined)

    def test_hard_hit_returns_one(self):
        self._write("leak.py", "token = HARDHIT_99\n")
        with mock.patch("builtins.print") as p:
            rc = cnp.main([self.root])
        self.assertEqual(rc, 1)
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("HARD", joined)
        self.assertIn("FAIL", joined)

    def test_warn_only_returns_zero_without_strict(self):
        self._write("warn.py", "ip = WARNHIT_7\n")
        with mock.patch("builtins.print"):
            rc = cnp.main([self.root])
        self.assertEqual(rc, 0)

    def test_warn_only_returns_one_with_strict(self):
        self._write("warn.py", "ip = WARNHIT_7\n")
        with mock.patch("builtins.print") as p:
            rc = cnp.main([self.root, "--strict"])
        self.assertEqual(rc, 1)
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("strict", joined)

    def test_strict_flag_not_treated_as_path(self):
        # "--strict" starts with '-' so it must not become the positional dir.
        self._write("ok.py", "x = 1\n")
        with mock.patch("builtins.print"):
            rc = cnp.main(["--strict", self.root])
        self.assertEqual(rc, 0)

    def test_reports_relative_path_and_scope(self):
        self._write(os.path.join("sub", "leak.py"), "HARDHIT_1\n")
        with mock.patch("builtins.print") as p:
            cnp.main([self.root])
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("directory", joined)
        self.assertIn("leak.py", joined)


class MainTrackedScanTests(unittest.TestCase):
    """main() with no positional arg -> goes through _tracked_files (mocked)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self._hard = list(cnp.HARD)
        self._warn = list(cnp.WARN)
        self._root_attr = cnp._PROJECT_ROOT
        cnp._PROJECT_ROOT = self.root
        cnp.HARD[:] = [("fx-hard", cnp._rx(r"HARDHIT_[0-9]+"))]
        cnp.WARN[:] = [("fx-warn", cnp._rx(r"WARNHIT_[0-9]+"))]

    def tearDown(self):
        cnp.HARD[:] = self._hard
        cnp.WARN[:] = self._warn
        cnp._PROJECT_ROOT = self._root_attr
        self._tmp.cleanup()

    def _write(self, name, body):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        return path

    def test_uses_tracked_files_when_no_arg(self):
        leak = self._write("leak.py", "HARDHIT_5\n")
        with mock.patch.object(cnp, "_tracked_files", return_value=[leak]), \
             mock.patch("builtins.print") as p:
            rc = cnp.main([])
        self.assertEqual(rc, 1)
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("git-tracked files", joined)

    def test_falls_back_to_walk_when_not_a_repo(self):
        # _tracked_files returns None (not a git repo) -> walk core/+skills/+tools.
        os.makedirs(os.path.join(self.root, "core"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "skills"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "tools"), exist_ok=True)
        with open(os.path.join(self.root, "core", "x.py"), "w", encoding="utf-8") as fh:
            fh.write("HARDHIT_3\n")
        with mock.patch.object(cnp, "_tracked_files", return_value=None), \
             mock.patch("builtins.print") as p:
            rc = cnp.main([])
        self.assertEqual(rc, 1)
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("no git repo found", joined)

    def test_empty_tracked_set_is_clean(self):
        with mock.patch.object(cnp, "_tracked_files", return_value=[]), \
             mock.patch("builtins.print") as p:
            rc = cnp.main([])
        self.assertEqual(rc, 0)
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("scanned 0 files", joined)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
