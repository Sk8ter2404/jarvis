"""Tests for the personal-data categories added to tools/check_no_pii.py
(2026-10-12 privacy audit): e-mail, phone, MAC, Tailscale, profile path, the
reviewed-binary rule, and the commit identity / message scans.

Self-gate safety: this file is scanned by the gate it tests, so every
personal-data-shaped fixture is ASSEMBLED AT RUNTIME from harmless pieces and no
source line is itself a HARD match. Hermetic: git is mocked, nothing touches the
network, a GPU or a device.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import tools.check_no_pii as cnp

AT = "@"
BS = chr(92)

REAL_EMAIL = "jane.doe" + AT + "gmail" + ".com"
REAL_MAC = ":".join(["3c", "22", "fb", "9a", "71", "0d"])
REAL_DASH_MAC = "-".join(["3c", "22", "fb", "9a", "71", "0d"])
TS_ADDR = ".".join(["100", "101", "47", "9"])
REAL_PHONE = "-".join(["901", "412", "7788"])
REAL_PROFILE = "C:" + BS + "Users" + BS + "jdoe" + BS + "Documents"
UNIX_PROFILE = "/home/" + "jdoe" + "/project"


def _write_bytes(path, data=b"\x89PNG"):
    with open(path, "wb") as fh:
        fh.write(data)


def _hard(text):
    return [label for label, rx in cnp.HARD if rx.search(text)]


class EmailRule(unittest.TestCase):
    def test_real_looking_address_is_hard(self):
        self.assertIn("email-address", _hard("contact " + REAL_EMAIL + " now"))

    def test_placeholder_domains_pass(self):
        for dom in ("example.com", "example.org", "school.example.edu",
                    "mail.test", "x.com", "users.noreply.github.com"):
            with self.subTest(dom=dom):
                self.assertNotIn("email-address", _hard("a.b" + AT + dom))

    def test_role_and_placeholder_locals_pass(self):
        for local in ("noreply", "shipment-tracking", "user", "you"):
            with self.subTest(local=local):
                self.assertNotIn("email-address",
                                 _hard(local + AT + "retailer" + ".com"))

    def test_co_author_trailer_address_passes(self):
        self.assertNotIn("email-address",
                         _hard("Co-Authored-By: Claude <noreply" + AT + "anthropic.com>"))

    def test_decorator_and_at_mention_are_not_emails(self):
        self.assertEqual(_hard("@pytest.fixture and @user mention"), [])


class MacRule(unittest.TestCase):
    def test_real_mac_is_hard(self):
        self.assertIn("mac-address", _hard("device " + REAL_MAC))
        self.assertIn("mac-address", _hard("device " + REAL_DASH_MAC))

    def test_placeholders_pass(self):
        for mac in ("00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff", "AA:BB:CC:DD:EE:FF",
                    "D0:73:D5:00:00:01", "50:C7:BF:11:22:33"):
            with self.subTest(mac=mac):
                self.assertNotIn("mac-address", _hard("mac " + mac))

    def test_time_ranges_are_not_macs(self):
        # mixed ':' and '-' separators: a transcript time range, not a MAC
        self.assertNotIn("mac-address", _hard("clip 00:24:21-00:25:43 ok"))


class NetworkRules(unittest.TestCase):
    def test_tailscale_cgnat_range(self):
        self.assertIn("tailscale-address", _hard("ssh " + TS_ADDR))
        self.assertIn("tailscale-address", _hard("ssh " + ".".join(["100", "64", "0", "1"])))
        self.assertIn("tailscale-address", _hard("ssh " + ".".join(["100", "127", "255", "254"])))

    def test_other_100_x_addresses_pass(self):
        for ip in ("100.63.1.1", "100.128.0.1", "100.0.0.1", "1100.101.47.9"):
            with self.subTest(ip=ip):
                self.assertNotIn("tailscale-address", _hard("host " + ip))

    def test_magicdns_name(self):
        self.assertIn("tailscale-host", _hard("box.tail1234.ts" + ".net"))


class PhoneRule(unittest.TestCase):
    def test_real_number_is_hard(self):
        self.assertIn("phone-number", _hard("call " + REAL_PHONE))
        self.assertIn("phone-number", _hard("call (" + "901" + ") 412-" + "7788"))
        self.assertIn("phone-number", _hard("call +1 " + " ".join(["901", "412", "7788"])))

    def test_fictional_and_placeholder_numbers_pass(self):
        for num in ("555-123-4567", "(555) 010-0100", "123-456-7890",
                    "800-555-0199"):
            with self.subTest(num=num):
                self.assertNotIn("phone-number", _hard("call " + num))

    def test_versions_dates_and_ips_are_not_phones(self):
        for text in ("version 3.14.2026", "2026-10-12", "192.168.100.2000",
                     "id 1234567890", "pid-123-456-7890123"):
            with self.subTest(text=text):
                self.assertNotIn("phone-number", _hard(text))


class ProfilePathRule(unittest.TestCase):
    def test_real_profile_is_hard(self):
        self.assertIn("user-profile-path", _hard(REAL_PROFILE))
        self.assertIn("user-profile-path", _hard(UNIX_PROFILE))
        self.assertIn("user-profile-path",
                      _hard("C:" + BS + BS + "Users" + BS + BS + "jdoe"))

    def test_placeholders_pass(self):
        for name in ("Public", "someuser", "runner", "<user>", "%USERNAME%",
                     "...", "B"):
            with self.subTest(name=name):
                self.assertNotIn("user-profile-path",
                                 _hard("C:" + BS + "Users" + BS + name + BS + "x"))

    def test_project_paths_pass(self):
        self.assertNotIn("user-profile-path", _hard("C:" + BS + "JARVIS" + BS + "data"))
        self.assertNotIn("user-profile-path", _hard("see /usr/home-dir/notes"))


class FilteredRxBehaviour(unittest.TestCase):
    def test_sub_skips_allowed_matches_and_redacts_the_rest(self):
        rx = next(r for label, r in cnp.HARD if label == "email-address")
        text = "a" + AT + "example.com and " + REAL_EMAIL
        out = rx.sub("<E>", text)
        self.assertIn("a" + AT + "example.com", out)
        self.assertNotIn(REAL_EMAIL, out)
        self.assertIn("<E>", out)

    def test_scrubber_picks_the_new_rules_up(self):
        from core import bug_reporter
        out = bug_reporter.scrub("mail " + REAL_EMAIL + " from " + TS_ADDR)
        self.assertNotIn(REAL_EMAIL, out)
        self.assertNotIn(TS_ADDR, out)


class BinaryReview(unittest.TestCase):
    def test_unreviewed_png_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "screenshot.png")
            _write_bytes(p)
            hits = cnp._binary_review_hits([p], d)
        self.assertEqual([h[2] for h in hits], ["binary-review"])
        self.assertEqual(hits[0][0], "screenshot.png")

    def test_reviewed_asset_passes(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "assets"))
            p = os.path.join(d, "assets", "jarvis_icon.png")
            _write_bytes(p)
            self.assertEqual(cnp._binary_review_hits([p], d), [])

    def test_text_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.py")
            _write_bytes(p, b"x = 1\n")
            self.assertEqual(cnp._binary_review_hits([p], d), [])

    def test_png_is_no_longer_skipped_by_the_directory_walk(self):
        with tempfile.TemporaryDirectory() as d:
            _write_bytes(os.path.join(d, "shot.png"))
            names = [os.path.basename(f) for f in cnp._walk_text_files(d)]
        self.assertIn("shot.png", names)

    def test_gate_fails_on_new_binary_in_tracked_set(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shot.png")
            _write_bytes(p)
            with mock.patch.object(cnp, "_tracked_files", return_value=[p]), \
                 mock.patch.object(cnp, "_PROJECT_ROOT", d), \
                 mock.patch("builtins.print"):
                self.assertEqual(cnp.main([]), 1)

    def test_directory_scan_only_warns(self):
        with tempfile.TemporaryDirectory() as d:
            _write_bytes(os.path.join(d, "shot.png"))
            with mock.patch("builtins.print"):
                self.assertEqual(cnp.main([d]), 0)

    def test_every_tracked_binary_in_the_repo_is_reviewed(self):
        # The real tree: the gate must not fail on what is committed today.
        files = cnp._tracked_files() or []
        self.assertEqual(cnp._binary_review_hits(files, cnp._PROJECT_ROOT), [])


def _log_record(sha, author, committer, body):
    f, r = chr(0x1f), chr(0x1e)
    return sha + f + author + f + committer + f + body + r


class CommitMetadata(unittest.TestCase):
    NOREPLY = "Dev <1234+dev" + AT + "users.noreply.github.com>"

    def _hits(self, log_out):
        with mock.patch.object(cnp, "_git_out", return_value=log_out):
            return cnp._commit_metadata_hits(["HEAD"])

    def test_noreply_identity_is_clean(self):
        out = _log_record("a" * 40, self.NOREPLY, self.NOREPLY, "Fix a thing\n")
        self.assertEqual(self._hits(out), [])

    def test_personal_author_address_is_flagged(self):
        out = _log_record("b" * 40, "Dev <" + REAL_EMAIL + ">", self.NOREPLY, "msg\n")
        self.assertEqual(self._hits(out), [("bbbbbbb", "author", "email-address", 1)])

    def test_personal_committer_address_is_flagged(self):
        out = _log_record("c" * 40, self.NOREPLY, "Dev <" + REAL_EMAIL + ">", "msg\n")
        self.assertEqual([h[1] for h in self._hits(out)], ["committer"])

    def test_message_body_is_scanned(self):
        out = _log_record("d" * 40, self.NOREPLY, self.NOREPLY,
                          "Subject\n\nreach me at " + REAL_EMAIL + "\n")
        hits = self._hits(out)
        self.assertEqual(hits[0][1:3], ("message", "email-address"))
        self.assertEqual(hits[0][3], 3)

    def test_co_author_trailer_is_clean(self):
        out = _log_record("e" * 40, self.NOREPLY, self.NOREPLY,
                          "Subject\n\nCo-Authored-By: Claude <noreply" + AT
                          + "anthropic.com>\n")
        self.assertEqual(self._hits(out), [])

    def test_git_failure_returns_none(self):
        with mock.patch.object(cnp, "_git_out", return_value=None):
            self.assertIsNone(cnp._commit_metadata_hits(["HEAD"]))

    def test_findings_never_print_the_value(self):
        out = _log_record("f" * 40, "Dev <" + REAL_EMAIL + ">", self.NOREPLY, "m\n")
        with mock.patch.object(cnp, "_git_out", return_value=out), \
             mock.patch("builtins.print") as p:
            rc = cnp.main(["--metadata", "HEAD"])
        self.assertEqual(rc, 1)
        shown = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertNotIn(REAL_EMAIL, shown)
        self.assertIn("[email-address]", shown)

    def test_metadata_clean_exit_zero(self):
        out = _log_record("a" * 40, self.NOREPLY, self.NOREPLY, "ok\n")
        with mock.patch.object(cnp, "_git_out", return_value=out), \
             mock.patch("builtins.print"):
            self.assertEqual(cnp.main(["--metadata", "HEAD"]), 0)

    def test_metadata_git_failure_exits_2(self):
        with mock.patch.object(cnp, "_git_out", return_value=None):
            self.assertEqual(cnp.main(["--metadata", "HEAD"]), 2)


class PreRevArgs(unittest.TestCase):
    def test_existing_ref_scans_remote_to_local(self):
        line = "refs/heads/m " + "1" * 40 + " refs/heads/m " + "2" * 40
        self.assertEqual(cnp._pre_push_rev_args(line),
                         [["2" * 40 + ".." + "1" * 40]])

    def test_new_ref_scans_only_commits_no_remote_has(self):
        line = "refs/heads/m " + "1" * 40 + " refs/heads/m " + "0" * 40
        self.assertEqual(cnp._pre_push_rev_args(line),
                         [["1" * 40, "--not", "--remotes"]])

    def test_delete_push_scans_nothing(self):
        line = "(delete) " + "0" * 40 + " refs/heads/m " + "2" * 40
        self.assertEqual(cnp._pre_push_rev_args(line), [])

    def test_pre_push_mode_reads_stdin(self):
        stdin = "refs/heads/m " + "1" * 40 + " refs/heads/m " + "2" * 40 + "\n"
        out = _log_record("a" * 40, CommitMetadata.NOREPLY, CommitMetadata.NOREPLY, "ok\n")
        with mock.patch.object(cnp.sys, "stdin") as si, \
             mock.patch.object(cnp, "_git_out", return_value=out), \
             mock.patch("builtins.print"):
            si.read.return_value = stdin
            self.assertEqual(cnp.main(["--pre-push"]), 0)


class CommitMsgHook(unittest.TestCase):
    def _run(self, message, ident):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "COMMIT_EDITMSG")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(message)
            with mock.patch.object(cnp, "_git_out", return_value=ident), \
                 mock.patch("builtins.print"):
                return cnp.main(["--commit-msg", path])

    def test_clean_message_and_identity(self):
        ident = "Dev <1+dev" + AT + "users.noreply.github.com> 1700000000 +0000\n"
        self.assertEqual(self._run("Fix\n\n# a comment\n", ident), 0)

    def test_personal_identity_blocks(self):
        ident = "Dev <" + REAL_EMAIL + "> 1700000000 +0000\n"
        self.assertEqual(self._run("Fix\n", ident), 1)

    def test_personal_data_in_message_blocks(self):
        ident = "Dev <1+dev" + AT + "users.noreply.github.com> 1700000000 +0000\n"
        self.assertEqual(self._run("Call " + REAL_PHONE + "\n", ident), 1)

    def test_comment_lines_are_not_scanned(self):
        ident = "Dev <1+dev" + AT + "users.noreply.github.com> 1700000000 +0000\n"
        self.assertEqual(self._run("Fix\n# " + REAL_EMAIL + "\n", ident), 0)

    def test_missing_file_exits_2(self):
        with mock.patch("builtins.print"):
            self.assertEqual(cnp.main(["--commit-msg", "/nonexistent/x"]), 2)

    def test_missing_argument_exits_2(self):
        self.assertEqual(cnp.main(["--commit-msg"]), 2)


class TreeStaysClean(unittest.TestCase):
    def test_tracked_tree_has_no_new_category_hits(self):
        """The shipped tree itself must be clean under the new rules."""
        names = {"email-address", "phone-number", "mac-address",
                 "tailscale-address", "tailscale-host", "user-profile-path"}
        rules = [(l, r) for l, r in cnp.HARD if l in names]
        bad = []
        for path in cnp._tracked_files() or []:
            if os.path.basename(path) == "check_no_pii.py":
                continue
            for label, ln, _snip in cnp._scan_file(path, rules):
                bad.append("%s:%d [%s]" % (os.path.relpath(path, cnp._PROJECT_ROOT),
                                           ln, label))
        self.assertEqual(bad, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
