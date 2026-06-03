"""Tests for core.version — the single source of truth for the release string.

The module reads the top-level VERSION file once at import and exposes it as
``__version__`` / ``VERSION`` / ``version_string()``. Pins: the import-time read
returns the live VERSION-file contents, ``version_string()`` echoes
``__version__``, and the ``_read_version`` helper degrades to the ``0.0.0-dev``
fallback when the file is missing/unreadable or empty (so a packaging slip never
crashes the import-light tier). The real VERSION file is never modified; the
helper is exercised against a temp path or a patched ``open``.

stdlib unittest + unittest.mock only.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from core import version as ver


class VersionConstantsTests(unittest.TestCase):
    def test_module_attrs_present_and_consistent(self):
        # __version__, VERSION and version_string() all agree.
        self.assertIsInstance(ver.__version__, str)
        self.assertEqual(ver.VERSION, ver.__version__)
        self.assertEqual(ver.version_string(), ver.__version__)

    def test_version_is_nonempty(self):
        # Whatever the VERSION file holds (or the fallback), it's never blank.
        self.assertTrue(ver.version_string().strip())


class ReadVersionTests(unittest.TestCase):
    def test_reads_file_contents_stripped(self):
        # A VERSION file with surrounding whitespace reads back trimmed.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "VERSION")
            with open(path, "w", encoding="utf-8") as f:
                f.write("  1.2.3-test\n")
            with mock.patch.object(ver, "_VERSION_FILE", path):
                self.assertEqual(ver._read_version(), "1.2.3-test")

    def test_empty_file_falls_back(self):
        # A present-but-empty VERSION file → the _FALLBACK sentinel.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "VERSION")
            with open(path, "w", encoding="utf-8") as f:
                f.write("   \n")
            with mock.patch.object(ver, "_VERSION_FILE", path):
                self.assertEqual(ver._read_version(), ver._FALLBACK)

    def test_missing_file_falls_back(self):
        # No VERSION file at the configured path → fallback, no raise.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "does-not-exist", "VERSION")
            with mock.patch.object(ver, "_VERSION_FILE", path):
                self.assertEqual(ver._read_version(), ver._FALLBACK)

    def test_oserror_on_open_falls_back(self):
        # Any OSError opening the file (permissions, etc.) degrades to fallback.
        with mock.patch.object(ver, "open", create=True,
                               side_effect=OSError("nope")):
            self.assertEqual(ver._read_version(), ver._FALLBACK)


if __name__ == "__main__":
    unittest.main()
