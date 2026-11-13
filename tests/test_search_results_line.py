"""web_search's "top results" line (live 2026-10-17 21:38).

For "AA battery to C battery converter" it reported the Google doodle "WNBA
Finals 2026" and the page's header links as the top results; the owner's
Google Account link (name + email) sat among those links. The elements below
mirror that page's real UI Automation tree (data/vision_trace, 10-17 21:38:40):
header controls first, then the "Search Results" heading, then the results.
stdlib unittest + mock; no window, browser or UI Automation is touched.
"""
from __future__ import annotations

import time
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
                mock.patch.object(A, "_RESULTS_RETRY_GAP_S", 0.0), \
                mock.patch.object(_st, "snapshot", return_value=_snap(elements)):
            return A._search_results_line(FOUND)

    def test_a_page_that_has_not_loaded_offers_nothing(self):
        # Live 2026-11-10 16:29:54: the "top results" were "Go to Google Home".
        self.assertEqual(self._line([_el("Go to Google Home", 40)]), "")
        self.assertEqual(self._line(HEADER + [_el("Go to Google Home", 40)]), "")
        # One stray link on a page with no results heading is not a result.
        self.assertEqual(self._line(HEADER + [
            _el("A single link that is not a result at all", 300)]), "")

    def test_it_waits_for_the_results_to_arrive(self):
        loading = _snap([_el("Go to Google Home", 40)])
        loaded = _snap(HEADER + RESULTS)
        with mock.patch("core.screen_privacy.window_private", return_value=False), \
                mock.patch.object(A, "_RESULTS_RETRY_GAP_S", 0.0), \
                mock.patch.object(_st, "snapshot",
                                  side_effect=[loading, None, loaded]) as snap:
            out = A._search_results_line(FOUND)
        self.assertEqual(snap.call_count, 3)
        self.assertTrue(out.startswith('"Powerlion C Size Battery Adapters'), out)

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


    def test_google_puts_the_title_site_and_address_in_one_tall_link(self):
        # Live 2026-11-13 11:12: a loaded results page read as nothing. Every
        # real result was taller than the 60 px limit (71 to 270 px), and its
        # name carries the site and the address after the title.
        page = HEADER + [
            _el("Search Results", 223, ctype="Text"),
            _el("Outlook down? Current outages and problems | Downdetector US", 300, h=122),
            _el("CNBC", 337, h=20),
            _el("Microsoft hit by global outages: What happened and why? "
                "8 hours ago Al Jazeera", 378, h=270),
            _el("Microsoft outage live updates: Email service is back CNN "
                "https://www.cnn.com › microsoft-outage", 851, h=71),
            _el("Read more", 949, h=17),
            # The same story again, further down the page.
            _el("Outlook down? Current outages and problems | Downdetector US "
                "https://downdetector.com › status", 993, h=71),
            _el("A full-page promo block that is no result at all here", 1200, h=900),
        ]
        self.assertEqual(
            self._line(page),
            '"Outlook down? Current outages and problems | Downdetector US", '
            '"Microsoft hit by global outages: What happened and why? '
            '8 hours ago Al Jazeera", '
            '"Microsoft outage live updates: Email service is back CNN"')

    def test_a_very_long_title_is_cut_at_a_word(self):
        long = "Microsoft Outlook outage updates and " + "more words " * 20
        out = self._line(RESULTS[:1] + [_el(long, 300, h=100)])
        self.assertTrue(out.endswith('..."'), out)
        self.assertLess(len(out), A._RESULT_TITLE_MAX + 10)


class ReadSearchResultsTests(unittest.TestCase):
    """2026-11-13 11:13: "what are the results?" had no action behind it. He
    was told there was no way to check, the search ran again, and the news
    briefing was read instead."""

    def setUp(self):
        saved = dict(A._last_search)
        self.addCleanup(A._last_search.update, saved)
        p = mock.patch("core.screen_privacy.window_private", return_value=False)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(A, "_RESULTS_RETRY_GAP_S", 0.0)
        p.start()
        self.addCleanup(p.stop)

    def test_nothing_searched_says_so(self):
        A._last_search.update(query="", found=None, at=0.0)
        self.assertIn("No web search is open", A._act_read_search_results(""))
        A._last_search.update(query="old", found=FOUND,
                              at=time.monotonic() - A._LAST_SEARCH_MAX_AGE_S - 5)
        self.assertIn("No web search is open", A._act_read_search_results(""))

    def test_it_reads_the_page_the_search_opened(self):
        A._last_search.update(query="aa to c", found=FOUND, at=time.monotonic())
        with mock.patch.object(_st, "snapshot", return_value=_snap(HEADER + RESULTS)),                 mock.patch.object(A, "_search_window") as find:
            out = A._act_read_search_results("")
        find.assert_not_called()
        # A finished sentence, spoken as written: numbered, three at most.
        self.assertEqual(
            out,
            "The top results for 'aa to c', sir. "
            "One: Powerlion C Size Battery Adapters, AA to C Size - 4 Pack. "
            "Two: AA to C Size Battery Spacer Converter Case. "
            "Three: How to use AA batteries in a C battery device.")
        for word in ("couldn't", "could not", "failed", "unable", "error"):
            self.assertNotIn(word, out.lower())

    def test_titles_are_cleaned_for_speech(self):
        # Live 2026-11-13 12:39:50: a "|" in a title and a card that was only
        # "X \u00b7 ..." came back.
        page = [_el("Search Results", 200, ctype="Text"),
                _el("Outlook down? Current outages and problems | Downdetector US", 300, h=120),
                _el("X \u00b7 Microsoft 365 Status", 420, h=118),
                _el("Microsoft hit by global outages: What happened and why? "
                    "8 hours ago Al Jazeera", 560, h=270)]
        A._last_search.update(query="outlook outage", found=FOUND, at=time.monotonic())
        with mock.patch.object(_st, "snapshot", return_value=_snap(page)):
            out = A._act_read_search_results("")
        self.assertIn("One: Outlook down? Current outages and problems, "
                      "Downdetector US.", out)
        self.assertNotIn("|", out)
        self.assertNotIn("\u00b7", out)

    def test_a_search_that_was_still_loading_is_found_by_its_title(self):
        A._last_search.update(query="aa to c", found=None, at=time.monotonic())
        with mock.patch.object(_st, "snapshot", return_value=_snap(HEADER + RESULTS)),                 mock.patch.object(A, "_search_window",
                                  return_value=(0x9, "aa to c - Google Search - Google Chrome")):
            out = A._act_read_search_results("")
        self.assertIn("AA to C Size Battery Spacer Converter Case", out)

    def test_a_page_that_cannot_be_read_says_so_and_opens_nothing(self):
        A._last_search.update(query="aa to c", found=None, at=time.monotonic())
        with mock.patch.object(A, "_search_window", return_value=None),                 mock.patch.object(A, "_webbrowser_open") as opened:
            out = A._act_read_search_results("")
        self.assertIn("The results page for 'aa to c' is not open", out)
        opened.assert_not_called()

    def test_web_search_remembers_what_it_searched(self):
        bc = mock.MagicMock()
        bc._VIDEO_QUERY_HINTS = ()
        with mock.patch.object(A, "_bc", return_value=bc),                 mock.patch.object(A, "_foreground_snapshot", return_value=(None, "", None)),                 mock.patch.object(A, "_webbrowser_open"),                 mock.patch.object(A, "_await_opened_page", return_value=FOUND),                 mock.patch.object(A, "_note_browser_tab_open"),                 mock.patch.object(_st, "snapshot", return_value=_snap(HEADER + RESULTS)):
            out = A._act_web_search("aa to c")
        self.assertIn("top results: \"Powerlion", out)
        self.assertEqual(A._last_search["query"], "aa to c")
        self.assertEqual(A._last_search["found"], FOUND)


if __name__ == "__main__":
    unittest.main()
