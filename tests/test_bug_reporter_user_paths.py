"""Audit A48: the bug-report scrubber must redact the Windows username in EVERY
shape a path takes on its way into a public bug report.

The rule used to match exactly ONE backslash either side of ``Users``. The most
common exception text is ``str(OSError)``, which shows the filename through
repr() and so DOUBLES every backslash -- and that form went through untouched,
so the account name leaked into both the summary and the traceback of a report
meant for a public issue. The table below covers the single, repr-doubled,
JSON-of-repr, forward-slash, lower-case (``os.path.normcase``), mixed, UNC and
extended-length forms, repr()/ascii() escapes of a non-ASCII name, and a
profile that lives outside ``C:\\Users`` (the expanded value of
``%USERPROFILE%`` / ``$HOME``).

Every fixture uses the made-up account name ``wombat42``. Backslashes are built
from ``BS`` so the doubling is explicit rather than hidden in escape rules.
stdlib only; import-light (core.bug_reporter needs no JARVIS boot).
"""
from __future__ import annotations

import errno
import os
import time
import unittest
from unittest import mock

from core import bug_reporter

BS = chr(92)
NAME = "wombat42"


def _join(sep: str, *parts: str) -> str:
    return sep.join(parts)


# (label, text). Every one carries NAME inside a home-directory path.
USERS_PATH_CASES = [
    ("single backslash",
     _join(BS, "C:", "Users", NAME, "cfg.json")),
    ("repr-doubled backslashes",
     _join(BS * 2, "C:", "Users", NAME, "cfg.json")),
    ("JSON of a repr (quadrupled)",
     _join(BS * 4, "C:", "Users", NAME, "cfg.json")),
    ("repr() of the path, quotes included",
     repr(_join(BS, "C:", "Users", NAME, "cfg.json"))),
    ("forward slashes",
     _join("/", "C:", "Users", NAME, "cfg.json")),
    ("lower case, as os.path.normcase writes it",
     _join(BS, "c:", "users", NAME, "appdata", "cfg.json")),
    ("lower case, forward slashes",
     _join("/", "c:", "users", NAME, "cfg.json")),
    ("mixed separators",
     "C:" + BS + "Users" + "/" + NAME + "/" + "cfg.json"),
    ("UNC admin share",
     BS * 2 + _join(BS, "fileserver", "c$", "Users", NAME, "cfg.json")),
    ("UNC admin share, repr-doubled",
     BS * 4 + _join(BS * 2, "fileserver", "c$", "Users", NAME, "cfg.json")),
    ("extended-length prefix",
     BS * 2 + "?" + BS + _join(BS, "C:", "Users", NAME, "cfg.json")),
    ("extended-length prefix, repr-doubled",
     BS * 4 + "?" + BS * 2 + _join(BS * 2, "C:", "Users", NAME, "cfg.json")),
    ("name is the last segment",
     "profile dir is " + _join(BS * 2, "C:", "Users", NAME)),
    ("inside an OSError message",
     "[Errno 2] No such file or directory: '"
     + _join(BS * 2, "C:", "Users", NAME, "cfg.json") + "'"),
]


class UsersPathShapesTests(unittest.TestCase):
    def test_every_shape_loses_the_username(self):
        for label, text in USERS_PATH_CASES:
            with self.subTest(shape=label):
                self.assertIn(NAME, text)          # the fixture really carries it
                out = bug_reporter.scrub(text)
                self.assertNotIn(NAME, out)
                self.assertIn("<USER>", out)

    def test_the_rest_of_the_path_survives(self):
        # Only the account segment goes; the file name stays useful.
        for label, text in USERS_PATH_CASES:
            if "cfg.json" not in text:
                continue
            with self.subTest(shape=label):
                self.assertIn("cfg.json", bug_reporter.scrub(text))

    def test_separators_are_kept_as_written(self):
        doubled = _join(BS * 2, "C:", "Users", NAME, "cfg.json")
        self.assertEqual(bug_reporter.scrub(doubled),
                         _join(BS * 2, "C:", "Users", "<USER>", "cfg.json"))
        single = _join(BS, "C:", "Users", NAME, "cfg.json")
        self.assertEqual(bug_reporter.scrub(single),
                         _join(BS, "C:", "Users", "<USER>", "cfg.json"))

    def test_a_name_with_a_space_goes_whole(self):
        text = _join(BS * 2, "C:", "Users", "Test Qwertyuiop", "cfg.json")
        out = bug_reporter.scrub(text)
        self.assertNotIn("Qwertyuiop", out)
        self.assertIn("cfg.json", out)

    def test_escaped_non_ascii_name_leaves_no_fragment(self):
        # repr() of a BYTES path and ascii() of a str path both turn a
        # non-ASCII letter into a backslash escape, which used to end the
        # match early and leak the rest of the name.
        name = "w" + chr(0xF6) + "mbatq"
        path = _join(BS, "C:", "Users", name, "cfg.json")
        for label, text in (("repr of bytes", repr(path.encode("utf-8"))),
                            ("ascii of str", ascii(path))):
            with self.subTest(form=label):
                self.assertIn(BS + "x", text)      # an escape really is there
                out = bug_reporter.scrub(text)
                self.assertNotIn("mbatq", out)
                self.assertIn("<USER>", out)
                self.assertIn("cfg.json", out)


class UsersPathFalsePositiveTests(unittest.TestCase):
    """The widening must not start eating text that is not a home directory."""

    def test_plain_text_is_left_alone(self):
        for text in (
            "https://api.github.com/users/octocat/repos",
            _join(BS, "C:", "Program Files", "JARVIS", "core.py"),
            "add alice to the Users group",
            _join(BS, "C:", "MyUsers", "notes.txt"),
        ):
            with self.subTest(text=text):
                self.assertEqual(bug_reporter.scrub(text), text)


class LongSeparatorRunTests(unittest.TestCase):
    """The run-of-separators rules must stay linear. Tried from every position
    of a long run of backslashes or slashes, a greedy separator run that is not
    followed by the expected text backtracks across the whole run, so scrub()
    went quadratic: tens of seconds for 60k backslashes, where the old rule
    took well under a second. Each run is now tried from its first character
    only. The bound is generous; the linear scrub takes milliseconds."""

    def test_a_long_separator_run_scrubs_quickly(self):
        env = {"USERPROFILE": "", "HOME": "/home/" + NAME,
               "HOMEDRIVE": "", "HOMEPATH": ""}
        with mock.patch.dict(os.environ, env):
            for label, text in (("backslashes", BS * 60_000),
                                ("slashes", "/" * 60_000)):
                with self.subTest(run=label):
                    t0 = time.perf_counter()
                    out = bug_reporter.scrub(text)
                    elapsed = time.perf_counter() - t0
                    self.assertEqual(out, text)
                    self.assertLess(elapsed, 2.0)

    def test_a_long_run_before_users_still_redacts(self):
        text = BS * 5_000 + "Users" + BS + NAME + BS + "cfg.json"
        out = bug_reporter.scrub(text)
        self.assertNotIn(NAME, out)
        self.assertTrue(out.endswith("Users" + BS + "<USER>" + BS + "cfg.json"))


class ProfileEnvExpansionTests(unittest.TestCase):
    """A profile outside C:\\Users (a redirected or roaming profile, a Linux
    home under /srv) is caught through the EXPANDED value of %USERPROFILE%,
    $HOME and %HOMEDRIVE%%HOMEPATH%, in any separator shape."""

    def _env(self, **values):
        base = {"USERPROFILE": "", "HOME": "", "HOMEDRIVE": "", "HOMEPATH": ""}
        base.update(values)
        return mock.patch.dict(os.environ, base)

    def test_relocated_userprofile(self):
        profile = _join(BS, "D:", "Profiles", NAME)
        with self._env(USERPROFILE=profile, HOME=profile):
            for label, text in (
                ("single", _join(BS, "D:", "Profiles", NAME, "cfg.json")),
                ("doubled", _join(BS * 2, "D:", "Profiles", NAME, "cfg.json")),
                ("forward", _join("/", "D:", "Profiles", NAME, "cfg.json")),
                ("lower case", _join(BS, "d:", "profiles", NAME, "cfg.json")),
            ):
                with self.subTest(shape=label):
                    out = bug_reporter.scrub(text)
                    self.assertNotIn(NAME, out)
                    self.assertIn("<USER>", out)
                    self.assertIn("cfg.json", out)

    def test_home_outside_slash_home(self):
        home = "/srv/homes/" + NAME
        with self._env(HOME=home, USERPROFILE=home):
            out = bug_reporter.scrub("open " + home + "/data.db failed")
        self.assertNotIn(NAME, out)
        self.assertIn("data.db", out)

    def test_homedrive_plus_homepath(self):
        with self._env(HOMEDRIVE="E:", HOMEPATH=BS + "Staff" + BS + NAME):
            out = bug_reporter.scrub(_join(BS * 2, "E:", "Staff", NAME, "x.log"))
        self.assertNotIn(NAME, out)
        self.assertIn("x.log", out)

    def test_a_longer_name_sharing_the_prefix_is_not_cut_in_half(self):
        profile = _join(BS, "D:", "Profiles", NAME)
        with self._env(USERPROFILE=profile, HOME=profile):
            out = bug_reporter.scrub(_join(BS, "D:", "Profiles", NAME + "x", "a"))
        self.assertNotIn("<USER>x", out)


class CaptureExceptionEndToEndTests(unittest.TestCase):
    """The audit's own repro: a real OSError for a home-dir path, through
    capture_exception, must not leak the name into summary OR traceback."""

    def test_file_not_found_report_is_clean(self):
        path = _join(BS, "C:", "Users", NAME, "nope_cfg.json")
        try:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        except FileNotFoundError as exc:
            # Blindness guard: the exception text really carries the DOUBLED form.
            self.assertIn(BS * 2 + NAME, str(exc))
            rep = bug_reporter.capture_exception(exc, where="config-load")
        for field in ("summary", "traceback"):
            with self.subTest(field=field):
                self.assertNotIn(NAME, rep[field])
                self.assertIn("<USER>", rep[field])

    def test_relocated_profile_report_is_clean(self):
        profile = _join(BS, "D:", "Profiles", NAME)
        env = {"USERPROFILE": profile, "HOME": profile,
               "HOMEDRIVE": "", "HOMEPATH": ""}
        with mock.patch.dict(os.environ, env):
            try:
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES),
                                      profile + BS + "cfg.json")
            except PermissionError as exc:
                rep = bug_reporter.capture_exception(exc)
        self.assertNotIn(NAME, rep["summary"])
        self.assertNotIn(NAME, rep["traceback"])


if __name__ == "__main__":
    unittest.main()
