"""web_search's "top results" line (live 2026-10-17 21:38).

For "AA battery to C battery converter" it reported the Google doodle "WNBA
Finals 2026" and the page's header links as the top results; the owner's
Google Account link (name + email) sat among those links. The elements below
mirror that page's real UI Automation tree (data/vision_trace, 10-17 21:38:40):
header controls first, then the "Search Results" heading, then the results.
stdlib unittest + mock; no window, browser or UI Automation is touched.
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import actions as A
from core import screen_text as _st


def _el(name, y, ctype="Hyperlink", h=24, in_document=True):
    return _st.El(name=name, ctype=ctype, rect=(100, y, 400, h),
                  in_document=in_document)


def _snap(elements):
    snap = mock.MagicMock()
    snap.elements = elements
    snap.has_password = False
    return snap


HEADER = [
    _el("Google Account: Some Owner (owner@example.com)", 10),
    _el("WNBA Finals 2026", 12),
    _el("Search by voice", 60),
    _el("Search by image", 60),
    _el("Skip to main content", 5),
    _el("Short videos", 120),
    _el("Power Adapters & Chargers filter. Not selected.", 150),
]
RESULTS = [
    _el("Search Results", 200, ctype="Text"),
    _el("Powerlion C Size Battery Adapters, AA to C Size - 4 Pack", 260),
    _el("AA to C Size Battery Spacer Converter Case", 330),
    _el("How to use AA batteries in a C battery device", 400),
]
FOUND = (0x123, "AA battery to C battery converter - Google Search - Google Chrome",
         (0, 0, 2560, 1440), "middle", "https://www.google.com/search?q=x")


class SearchResultsLineTests(unittest.TestCase):
    def _line(self, elements):
        with mock.patch("core.screen_privacy.window_private", return_value=False), \
                mock.patch.object(_st, "snapshot", return_value=_snap(elements)):
            return A._search_results_line(FOUND)

    def test_only_results_below_the_search_results_heading(self):
        out = self._line(HEADER + RESULTS)
        self.assertTrue(out.startswith('"Powerlion C Size Battery Adapters'), out)
        self.assertIn("Spacer Converter Case", out)
        self.assertNotIn("WNBA", out)
        self.assertNotIn("Search by", out)

    def test_never_the_account_link_or_an_email(self):
        out = self._line(HEADER + RESULTS)
        self.assertNotIn("Google Account", out)
        self.assertNotIn("@", out)

    def test_a_page_without_the_heading_still_skips_the_controls(self):
        no_anchor = HEADER + RESULTS[1:]
        out = self._line(no_anchor)
        self.assertNotIn("Google Account", out)
        self.assertNotIn("Search by voice", out)
        self.assertNotIn("filter.", out)
        self.assertNotIn("Short videos", out)
        self.assertIn("Powerlion", out)

    def test_a_result_that_starts_with_a_tab_word_is_kept(self):
        out = self._line(HEADER + RESULTS[:1] + [
            _el("All about AA to C battery adapters", 260),
            _el("News on rechargeable C batteries for 2026", 330)])
        self.assertIn("All about AA to C battery adapters", out)
        self.assertIn("News on rechargeable C batteries", out)

    def test_a_private_window_reads_nothing(self):
        with mock.patch("core.screen_privacy.window_private", return_value=True), \
                mock.patch.object(_st, "snapshot") as snap:
            self.assertEqual(A._search_results_line(FOUND), "")
        snap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
