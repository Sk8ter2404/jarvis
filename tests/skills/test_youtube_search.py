"""Logic tests for skills/youtube_search.py.

Resolves a search query to a canonical YouTube watch URL via yt-dlp. We never
spawn yt-dlp or open a browser:
  * subprocess.run is mocked to return canned CompletedProcess-likes,
  * webbrowser.open is mocked (asserted, never actually opens),
  * time.sleep is mocked so the post-open settle doesn't slow the test.

Covered: the PATH/`python -m` probe + its cache, URL extraction from stdout,
returncode / timeout / empty-output failure paths, the missing-dep hint, and
both graceful-fallback branches of the action (no yt-dlp, and yt-dlp found no
match) which still open the results page.
"""
from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_WATCH = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def _completed(stdout="", returncode=0):
    cp = mock.MagicMock()
    cp.stdout = stdout
    cp.returncode = returncode
    return cp


class YtdlpProbeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("youtube_search")
        self.mod._YTDLP_CMD = None  # reset the probe cache each test

    def test_probe_prefers_path_binary(self):
        with mock.patch.object(self.mod.shutil, "which",
                               side_effect=lambda n: r"C:\bin\yt-dlp.exe"
                               if n == "yt-dlp" else None):
            cmd = self.mod._probe_ytdlp()
        self.assertEqual(cmd, [r"C:\bin\yt-dlp.exe"])

    def test_probe_falls_back_to_python_module(self):
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod.subprocess, "run",
                               return_value=_completed(returncode=0)):
            cmd = self.mod._probe_ytdlp()
        self.assertEqual(cmd[1:], ["-m", "yt_dlp"])  # [sys.executable, -m, yt_dlp]

    def test_probe_returns_empty_when_absent(self):
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod.subprocess, "run",
                               return_value=_completed(returncode=1)):
            self.assertEqual(self.mod._probe_ytdlp(), [])

    def test_probe_is_cached(self):
        # Once probed, a second call must not touch shutil.which again.
        self.mod._YTDLP_CMD = ["yt-dlp"]
        with mock.patch.object(self.mod.shutil, "which") as which:
            self.assertEqual(self.mod._probe_ytdlp(), ["yt-dlp"])
        which.assert_not_called()

    def test_probe_subprocess_exception_treated_as_absent(self):
        # The `import yt_dlp` confirmation subprocess raises (e.g. OSError on a
        # locked-down host) -> the broad except swallows it and the probe
        # reports unavailable.
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("exec denied")):
            self.assertEqual(self.mod._probe_ytdlp(), [])


class FindDirectUrlTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("youtube_search")
        self.mod._YTDLP_CMD = ["yt-dlp"]  # pretend yt-dlp is available

    def test_empty_query_returns_none(self):
        self.assertIsNone(self.mod.find_direct_url("   "))

    def test_no_ytdlp_returns_none(self):
        self.mod._YTDLP_CMD = []  # probed → unavailable
        self.assertIsNone(self.mod.find_direct_url("never gonna give you up"))

    def test_extracts_watch_url_from_stdout(self):
        with mock.patch.object(self.mod.subprocess, "run",
                               return_value=_completed(stdout=_WATCH + "\n")):
            url = self.mod.find_direct_url("rick astley")
        self.assertEqual(url, _WATCH)

    def test_ignores_non_watch_lines(self):
        noisy = "WARNING: something\n" + _WATCH + "\nextra\n"
        with mock.patch.object(self.mod.subprocess, "run",
                               return_value=_completed(stdout=noisy)):
            self.assertEqual(self.mod.find_direct_url("q"), _WATCH)

    def test_nonzero_returncode_returns_none(self):
        with mock.patch.object(self.mod.subprocess, "run",
                               return_value=_completed(stdout=_WATCH, returncode=1)):
            self.assertIsNone(self.mod.find_direct_url("q"))

    def test_timeout_returns_none(self):
        with mock.patch.object(self.mod.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("yt-dlp", 10)):
            self.assertIsNone(self.mod.find_direct_url("q"))

    def test_generic_subprocess_exception_returns_none(self):
        # A non-timeout spawn failure (OSError) hits the broad `except
        # Exception` and degrades to None.
        with mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("spawn failed")):
            self.assertIsNone(self.mod.find_direct_url("q"))

    def test_empty_stdout_returns_none(self):
        with mock.patch.object(self.mod.subprocess, "run",
                               return_value=_completed(stdout="")):
            self.assertIsNone(self.mod.find_direct_url("q"))


class YoutubeSearchDirectActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("youtube_search")
        self.mod._YTDLP_CMD = ["yt-dlp"]

    def test_no_query(self):
        self.assertIn("no query", self.actions["youtube_search_direct"]("").lower())

    def test_happy_path_opens_resolved_url(self):
        with mock.patch.object(self.mod, "find_direct_url", return_value=_WATCH), \
             mock.patch.object(self.mod.webbrowser, "open") as wopen, \
             mock.patch.object(self.mod.time, "sleep"):
            out = self.actions["youtube_search_direct"]("rick astley")
        wopen.assert_called_once_with(_WATCH)
        self.assertIn(_WATCH, out)
        self.assertIn("playing", out)

    def test_no_ytdlp_falls_back_to_results_page(self):
        self.mod._YTDLP_CMD = []  # unavailable
        with mock.patch.object(self.mod.webbrowser, "open") as wopen:
            out = self.actions["youtube_search_direct"]("lofi beats")
        # Opens the SERP with the url-encoded query as a graceful fallback.
        self.assertIn("results page", out)
        opened = wopen.call_args[0][0]
        self.assertIn("results?search_query=lofi", opened)
        self.assertIn("yt-dlp is not installed", out)

    def test_no_match_falls_back_to_results_page(self):
        with mock.patch.object(self.mod, "find_direct_url", return_value=None), \
             mock.patch.object(self.mod.webbrowser, "open") as wopen:
            out = self.actions["youtube_search_direct"]("zzz nonexistent zzz")
        self.assertIn("couldn't find a direct match", out)
        self.assertIn("results?search_query=", wopen.call_args[0][0])

    def test_browser_open_failure_is_reported(self):
        with mock.patch.object(self.mod, "find_direct_url", return_value=_WATCH), \
             mock.patch.object(self.mod.webbrowser, "open",
                               side_effect=RuntimeError("no browser")), \
             mock.patch.object(self.mod.time, "sleep"):
            out = self.actions["youtube_search_direct"]("q")
        self.assertIn("couldn't open the browser", out)
        self.assertIn(_WATCH, out)

    def test_no_ytdlp_fallback_swallows_browser_error(self):
        # yt-dlp absent AND the results-page open() raises -> the except swallows
        # it and the install-hint message is still returned.
        self.mod._YTDLP_CMD = []
        with mock.patch.object(self.mod.webbrowser, "open",
                               side_effect=RuntimeError("no browser")):
            out = self.actions["youtube_search_direct"]("lofi beats")
        self.assertIn("yt-dlp is not installed", out)
        self.assertIn("results page", out)

    def test_no_match_fallback_swallows_browser_error(self):
        # yt-dlp found no match AND the results-page open() raises -> swallowed;
        # the "couldn't find a direct match" message is still returned.
        with mock.patch.object(self.mod, "find_direct_url", return_value=None), \
             mock.patch.object(self.mod.webbrowser, "open",
                               side_effect=RuntimeError("no browser")):
            out = self.actions["youtube_search_direct"]("zzz nope zzz")
        self.assertIn("couldn't find a direct match", out)


if __name__ == "__main__":
    unittest.main()
