"""skills.file_move security review fixes (v3 W22, 2026-10-27).

FM-1  the dry run never touches a UNC / WebDAV / device path: refusal runs
      first, and those forms are refused outright, so no isdir / isfile /
      stat / realpath call ever reaches a remote host (an SMB or WebDAV
      lookup hands the owner's NTLM response to that host).
FM-2  the question names a non-known folder with its parent, so
      Nextcloud\\Documents never reads like the local Documents.
FM-3  the PowerShell profile folders under Documents are refused.
FM-4  the pushback reason (printed to the session log) holds no paths.

The recorders below NEVER call the real filesystem for a remote-looking
argument: they record it and fail closed, so the tests touch no network.

    python -m unittest tests.test_file_move_unc
"""
from __future__ import annotations

import contextlib
import os
import unittest
from unittest import mock

from skills import file_move as FM
from tests.skills.test_file_move import FakeProfile


def _remote(arg) -> bool:
    try:
        s = os.fspath(arg)
    except TypeError:
        return False
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    return s.replace("/", "\\").startswith("\\\\")


class _Recorder:
    """Wraps os.path.isdir / isfile / exists / lexists / islink /
    realpath and os.stat / os.scandir: a remote argument is recorded and
    never passed on."""

    def __init__(self):
        self.seen = []

    @contextlib.contextmanager
    def active(self):
        real = {"isdir": os.path.isdir, "isfile": os.path.isfile,
                "exists": os.path.exists, "lexists": os.path.lexists,
                "islink": os.path.islink, "realpath": os.path.realpath,
                "stat": os.stat, "scandir": os.scandir}

        def wrap(name, fail):
            def inner(p, *a, **k):
                if _remote(p):
                    self.seen.append((name, os.fspath(p)))
                    return fail(p)
                return real[name](p, *a, **k)
            return inner

        def _oserror(p):
            raise OSError(f"recorder: refused remote {p!r}")

        with contextlib.ExitStack() as st:
            for name in ("isdir", "isfile", "exists", "lexists", "islink"):
                st.enter_context(mock.patch.object(os.path, name,
                                                   wrap(name, lambda p: False)))
            st.enter_context(mock.patch.object(os.path, "realpath",
                                               wrap("realpath", lambda p: os.fspath(p))))
            st.enter_context(mock.patch.object(os, "stat", wrap("stat", _oserror)))
            st.enter_context(mock.patch.object(os, "scandir", wrap("scandir", _oserror)))
            yield self


class NoRemoteTouch(unittest.TestCase):
    """FM-1."""

    def setUp(self):
        self.p = FakeProfile(self)
        self.p.dl("resume.pdf")

    def _check(self, fn, *args):
        rec = _Recorder()
        with rec.active():
            out = fn(*args)
        self.assertEqual(rec.seen, [], f"{fn.__name__}{args!r} touched a remote path")
        return out

    def test_webdav_destination(self):
        plan, why = self._check(FM.plan_move, r"resume to \\evil.example@SSL\dav\x")
        self.assertIsNone(plan)
        self.assertIn("outside your own folders", why)

    def test_forward_slash_unc_source(self):
        plan, why = self._check(FM.plan_move, "//evil/share/a.pdf to documents")
        self.assertIsNone(plan)
        self.assertIn("outside your own folders", why)

    def test_unc_from_hint(self):
        plan, _why = self._check(FM.plan_move, r"resume from \\evil\s to documents")
        # The hint names no allowed folder, so the words are read as part of
        # the file's name: nothing matches, and no remote lookup happened.
        self.assertIsNone(plan)

    def test_confirm_question_dry_run(self):
        self.assertIsNone(self._check(FM.confirm_question, "move_file",
                                      r"\\evil\s\a.pdf to documents"))

    def test_device_and_long_forms_refused_outright(self):
        for arg in (r"\\?\C:\Windows\x.pdf to documents",
                    r"\\.\pipe\x to documents",
                    r"resume to \\?\UNC\evil\share",
                    "resume to /\\evil/share"):
            plan, why = self._check(FM.plan_move, arg)
            self.assertIsNone(plan, arg)
            self.assertIn("outside your own folders", why, arg)

    def test_resolvers_directly(self):
        for fn in (FM.resolve_folder, FM.resolve_source):
            got, why = self._check(fn, r"\\evil\share\x")
            self.assertIsNone(got)
            self.assertIn("outside your own folders", why)

    def test_local_paths_still_work(self):
        src = os.path.join(self.p.home, "Downloads", "resume.pdf")
        plan, why = FM.plan_move(f"{src} to {self.p.known['documents']}")
        self.assertEqual(why, "")
        self.assertEqual(plan["src"], src)


class QuestionNamesTheParent(unittest.TestCase):
    """FM-2."""

    def setUp(self):
        self.p = FakeProfile(self)
        self.p.dl("return.pdf")

    def test_nextcloud_documents_is_not_the_local_documents(self):
        q_nc = FM.confirm_question("move_file", "return.pdf to nextcloud documents")[0]
        q_local = FM.confirm_question("move_file", "return.pdf to documents")[0]
        self.assertIn("to Documents in Nextcloud, sir?", q_nc)
        self.assertEqual(q_local, "Move 'return.pdf' from Downloads to Documents, sir?")
        self.assertNotEqual(q_nc, q_local)

    def test_subfolder_names_its_known_parent(self):
        taxes = os.path.join(self.p.known["documents"], "Taxes")
        os.makedirs(taxes)
        self.assertEqual(FM.folder_label(taxes), "Taxes in Documents")
        self.assertEqual(FM.folder_label(self.p.nc), "Nextcloud")
        self.assertEqual(FM.folder_label(self.p.known["downloads"]), "Downloads")


class PowerShellProfileFolders(unittest.TestCase):
    """FM-3."""

    def setUp(self):
        self.p = FakeProfile(self)
        for name in ("WindowsPowerShell", "PowerShell"):
            os.makedirs(os.path.join(self.p.known["documents"], name))
        os.makedirs(os.path.join(self.p.nc, "Documents", "WindowsPowerShell"))
        self.p.dl("Microsoft.PowerShell_profile.ps1")

    def test_by_name(self):
        for word in ("WindowsPowerShell", "PowerShell"):
            got, why = FM.resolve_folder(word)
            self.assertIsNone(got, word)
            self.assertIn("runs whenever PowerShell starts", why)

    def test_by_path_including_the_nextcloud_copy(self):
        for d in (os.path.join(self.p.known["documents"], "WindowsPowerShell"),
                  os.path.join(self.p.nc, "Documents", "WindowsPowerShell")):
            got, why = FM.resolve_folder(d)
            self.assertIsNone(got, d)
            self.assertIn("runs whenever PowerShell starts", why)

    def test_plan_refused(self):
        plan, why = FM.plan_move("Microsoft.PowerShell_profile.ps1 to WindowsPowerShell")
        self.assertIsNone(plan)
        self.assertIn("runs whenever PowerShell starts", why)

    def test_a_powershell_folder_elsewhere_is_still_fine(self):
        d = os.path.join(self.p.known["desktop"], "PowerShell")
        os.makedirs(d)
        self.assertEqual(FM.resolve_folder(d), (d, ""))


class ReasonHoldsNoPath(unittest.TestCase):
    """FM-4."""

    def setUp(self):
        self.p = FakeProfile(self)
        self.p.dl("2025 return.pdf")

    def test_reason(self):
        q = FM.confirm_question("move_file", "2025 return to documents")
        self.assertIsNotNone(q)
        self.assertNotIn(os.sep, q[1])
        self.assertNotIn("/", q[1])
        self.assertNotIn("return", q[1])
        self.assertTrue(q[1].startswith("move_file"))


if __name__ == "__main__":
    unittest.main()
