"""Tests for tools/build_release.py — the shareable-release builder.

build_release exports ONLY the git-tracked tree (so gitignored personal data is
excluded by construction), runs check_no_pii on the OUTPUT as a hard gate,
stamps VERSION into the name, and zips it; if the leak scan fails it deletes the
artifact and returns 1. A regression could ship runtime/personal data, so this
is load-bearing.

CI-safety: the module top-level-imports stdlib only (safe to import on the bare
Linux runner). Every test patches the module's _ROOT to a TemporaryDirectory and
mocks subprocess.run so NO real git runs and the real check_no_pii subprocess is
never spawned. zipfile/shutil operate only inside the temp tree. Nothing here is
OS-specific (paths are built with os.path.join); no guards are needed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import tools.build_release as br


def _completed(returncode=0, stdout="", stderr=""):
    """A stand-in for subprocess.CompletedProcess (only the attrs main() reads)."""
    return mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)


class VersionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self._root_attr = br._ROOT
        br._ROOT = self.root

    def tearDown(self):
        br._ROOT = self._root_attr
        self._tmp.cleanup()

    def _write_version(self, text):
        with open(os.path.join(self.root, "VERSION"), "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_reads_version_file(self):
        self._write_version("1.2.3\n")
        self.assertEqual(br._version(), "1.2.3")

    def test_strips_surrounding_whitespace(self):
        self._write_version("   9.9.9-rc1  \n\n")
        self.assertEqual(br._version(), "9.9.9-rc1")

    def test_empty_version_file_falls_back(self):
        self._write_version("   \n")
        self.assertEqual(br._version(), "0.0.0-dev")

    def test_missing_version_file_falls_back(self):
        # No VERSION file in the temp root -> OSError -> default.
        self.assertEqual(br._version(), "0.0.0-dev")


class MainBuildTests(unittest.TestCase):
    """Drive main() end-to-end with git + check_no_pii fully mocked.

    Real filesystem work happens only inside the temp _ROOT: source files are
    written, then copied to dist/<name>/, then zipped — all under the tempdir.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self._root_attr = br._ROOT
        br._ROOT = self.root
        # A couple of tracked source files to export.
        self._src("VERSION", "2.0.0-test\n")
        self._src("README.md", "hello alice\n")
        self._src(os.path.join("core", "app.py"), "x = 1\n")
        self.tracked = ["VERSION", "README.md", os.path.join("core", "app.py")]

    def tearDown(self):
        br._ROOT = self._root_attr
        self._tmp.cleanup()

    def _src(self, rel, body):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path) or self.root, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        return path

    def _run_dispatch(self, pii_returncode=0, pii_stdout="[check_no_pii] OK\n",
                      ls_files_lines=None, ls_check_raises=None):
        """Return a subprocess.run side-effect that distinguishes the two calls.

        Call 1: ['git', 'ls-files']      -> tracked-file listing (or raises)
        Call 2: [python, check_no_pii.py, out_dir] -> the leak gate result
        """
        lines = self.tracked if ls_files_lines is None else ls_files_lines

        def _side(cmd, *a, **kw):
            if cmd[:2] == ["git", "ls-files"]:
                if ls_check_raises is not None:
                    raise ls_check_raises
                return _completed(0, "\n".join(lines) + ("\n" if lines else ""))
            # otherwise: the check_no_pii invocation
            self.assertIn("check_no_pii.py", cmd[1])
            return _completed(pii_returncode, pii_stdout)

        return _side

    # ---- happy path -------------------------------------------------------
    def test_successful_build_returns_zero_and_writes_zip(self):
        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch()), \
             mock.patch("builtins.print"):
            rc = br.main([])
        self.assertEqual(rc, 0)
        zip_path = os.path.join(self.root, "dist", "jarvis-2.0.0-test.zip")
        self.assertTrue(os.path.isfile(zip_path), "release zip should exist")

    def test_zip_contains_tracked_files_under_named_top_dir(self):
        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch()), \
             mock.patch("builtins.print"):
            br.main([])
        zip_path = os.path.join(self.root, "dist", "jarvis-2.0.0-test.zip")
        with zipfile.ZipFile(zip_path) as z:
            names = set(z.namelist())
        top = "jarvis-2.0.0-test"
        # arcnames use os.path.join, so normalise separators for the assert.
        norm = {n.replace("\\", "/") for n in names}
        self.assertIn(f"{top}/VERSION", norm)
        self.assertIn(f"{top}/README.md", norm)
        self.assertIn(f"{top}/core/app.py", norm)

    def test_default_removes_unzipped_dir(self):
        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch()), \
             mock.patch("builtins.print"):
            br.main([])
        out_dir = os.path.join(self.root, "dist", "jarvis-2.0.0-test")
        self.assertFalse(os.path.isdir(out_dir),
                         "unzipped staging dir should be cleaned by default")

    def test_keep_flag_leaves_unzipped_dir(self):
        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch()), \
             mock.patch("builtins.print"):
            rc = br.main(["--keep"])
        self.assertEqual(rc, 0)
        out_dir = os.path.join(self.root, "dist", "jarvis-2.0.0-test")
        self.assertTrue(os.path.isdir(out_dir), "--keep should retain staging dir")
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "VERSION")))

    def test_copied_files_have_expected_contents(self):
        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch()), \
             mock.patch("builtins.print"):
            br.main(["--keep"])
        copied = os.path.join(self.root, "dist", "jarvis-2.0.0-test", "core", "app.py")
        with open(copied, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "x = 1\n")

    # ---- leak gate fails --------------------------------------------------
    def test_leak_scan_failure_aborts_and_removes_artifacts(self):
        with mock.patch.object(
                br.subprocess, "run",
                side_effect=self._run_dispatch(pii_returncode=1,
                                               pii_stdout="[check_no_pii] FAIL\n")), \
             mock.patch("builtins.print") as p:
            rc = br.main([])
        self.assertEqual(rc, 1)
        out_dir = os.path.join(self.root, "dist", "jarvis-2.0.0-test")
        zip_path = os.path.join(self.root, "dist", "jarvis-2.0.0-test.zip")
        self.assertFalse(os.path.isdir(out_dir), "staging dir must be removed on leak")
        self.assertFalse(os.path.isfile(zip_path), "no zip should be written on leak")
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("LEAK SCAN FAILED", joined)

    # ---- git failure ------------------------------------------------------
    def test_git_ls_files_failure_returns_one(self):
        from subprocess import CalledProcessError
        with mock.patch.object(
                br.subprocess, "run",
                side_effect=self._run_dispatch(
                    ls_check_raises=CalledProcessError(1, ["git", "ls-files"]))), \
             mock.patch("builtins.print") as p:
            rc = br.main([])
        self.assertEqual(rc, 1)
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("git ls-files failed", joined)

    def test_git_ls_files_uses_check_true(self):
        captured = {}

        def _side(cmd, *a, **kw):
            if cmd[:2] == ["git", "ls-files"]:
                captured["check"] = kw.get("check")
                captured["cwd"] = kw.get("cwd")
                return _completed(0, "VERSION\n")
            return _completed(0, "[check_no_pii] OK\n")

        with mock.patch.object(br.subprocess, "run", side_effect=_side), \
             mock.patch("builtins.print"):
            br.main([])
        self.assertTrue(captured.get("check"), "git ls-files must pass check=True")
        self.assertEqual(captured.get("cwd"), self.root)

    # ---- copy edge cases --------------------------------------------------
    def test_skips_tracked_entries_that_are_not_files(self):
        # 'core' (a directory) appears in ls-files output -> isfile() False -> skipped.
        lines = ["VERSION", "core"]
        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch(ls_files_lines=lines)), \
             mock.patch("builtins.print"):
            rc = br.main(["--keep"])
        self.assertEqual(rc, 0)
        out_dir = os.path.join(self.root, "dist", "jarvis-2.0.0-test")
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "VERSION")))
        # 'core' should not have been created as a bare file.
        self.assertFalse(os.path.isfile(os.path.join(out_dir, "core")))

    def test_missing_tracked_source_is_skipped(self):
        # A tracked path that doesn't exist on disk -> isfile() False -> skipped, no crash.
        lines = ["VERSION", "ghost.py"]
        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch(ls_files_lines=lines)), \
             mock.patch("builtins.print"):
            rc = br.main(["--keep"])
        self.assertEqual(rc, 0)
        out_dir = os.path.join(self.root, "dist", "jarvis-2.0.0-test")
        self.assertFalse(os.path.exists(os.path.join(out_dir, "ghost.py")))

    def test_copy_oserror_is_warned_not_fatal(self):
        # shutil.copy2 raising OSError for one file must be caught and reported.
        real_copy2 = br.shutil.copy2

        def _flaky_copy(src, dst, *a, **kw):
            if os.path.basename(src) == "README.md":
                raise OSError("disk full (simulated)")
            return real_copy2(src, dst, *a, **kw)

        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch()), \
             mock.patch.object(br.shutil, "copy2", side_effect=_flaky_copy), \
             mock.patch("builtins.print") as p:
            rc = br.main(["--keep"])
        self.assertEqual(rc, 0)
        joined = "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("could not copy", joined)

    # ---- pre-existing output handling ------------------------------------
    def test_preexisting_out_dir_is_replaced(self):
        # Stale staging dir with junk -> removed before re-export.
        stale = os.path.join(self.root, "dist", "jarvis-2.0.0-test")
        os.makedirs(stale, exist_ok=True)
        with open(os.path.join(stale, "STALE.txt"), "w", encoding="utf-8") as fh:
            fh.write("old\n")
        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch()), \
             mock.patch("builtins.print"):
            rc = br.main(["--keep"])
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.isfile(os.path.join(stale, "STALE.txt")),
                         "stale staging contents must be cleared")

    def test_preexisting_zip_is_overwritten(self):
        dist = os.path.join(self.root, "dist")
        os.makedirs(dist, exist_ok=True)
        zip_path = os.path.join(dist, "jarvis-2.0.0-test.zip")
        with open(zip_path, "w", encoding="utf-8") as fh:
            fh.write("not a real zip")
        with mock.patch.object(br.subprocess, "run",
                               side_effect=self._run_dispatch()), \
             mock.patch("builtins.print"):
            rc = br.main([])
        self.assertEqual(rc, 0)
        # It should now be a valid zip (overwritten), not the placeholder text.
        self.assertTrue(zipfile.is_zipfile(zip_path))

    # ---- check_no_pii invocation shape -----------------------------------
    def test_invokes_check_no_pii_on_output_dir_with_current_interpreter(self):
        seen = {}

        def _side(cmd, *a, **kw):
            if cmd[:2] == ["git", "ls-files"]:
                return _completed(0, "VERSION\n")
            seen["cmd"] = cmd
            return _completed(0, "[check_no_pii] OK\n")

        with mock.patch.object(br.subprocess, "run", side_effect=_side), \
             mock.patch("builtins.print"):
            br.main([])
        cmd = seen["cmd"]
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(cmd[1].endswith(os.path.join("tools", "check_no_pii.py")))
        # last arg is the staging output dir under dist/
        self.assertTrue(cmd[2].endswith(os.path.join("dist", "jarvis-2.0.0-test")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
