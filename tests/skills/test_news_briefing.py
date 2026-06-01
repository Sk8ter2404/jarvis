"""Logic tests for skills/news_briefing.py.

Covers the pure RSS/Atom plumbing (HTML stripping, stdlib feed parsing, config
normalisation, round-robin headline gathering) and the briefing assembly with
network + LLM mocked out. get_news_text()'s graceful-degradation contract
(disabled / no feeds / all feeds fail → "") is verified explicitly.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class NewsBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("news_briefing")

    # ── _strip_html (pure) ───────────────────────────────────────────────
    def test_strip_html_tags_and_entities(self):
        s = self.mod._strip_html("<b>Big</b> &amp; bold&#39;s   tale")
        self.assertEqual(s, "Big & bold's tale")

    def test_strip_html_empty(self):
        self.assertEqual(self.mod._strip_html(""), "")
        self.assertEqual(self.mod._strip_html(None), "")

    # ── _parse_with_stdlib (pure RSS + Atom) ─────────────────────────────
    def test_parse_rss_items(self):
        xml = (
            "<rss><channel>"
            "<item><title>First headline</title>"
            "<description>Body one</description></item>"
            "<item><title>Second headline</title></item>"
            "</channel></rss>"
        )
        items = self.mod._parse_with_stdlib(xml)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["title"], "First headline")
        self.assertEqual(items[0]["description"], "Body one")
        self.assertEqual(items[1]["title"], "Second headline")

    def test_parse_atom_with_namespace(self):
        xml = (
            '<feed xmlns="http://www.w3.org/2005/Atom">'
            "<entry><title>Atom story</title>"
            "<summary>Atom summary</summary></entry>"
            "</feed>"
        )
        items = self.mod._parse_with_stdlib(xml)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "Atom story")
        self.assertEqual(items[0]["description"], "Atom summary")

    def test_parse_skips_titleless_and_handles_bad_xml(self):
        self.assertEqual(self.mod._parse_with_stdlib("<<not xml>>"), [])
        # An item with no title is dropped.
        xml = "<rss><channel><item><description>orphan</description></item></channel></rss>"
        self.assertEqual(self.mod._parse_with_stdlib(xml), [])

    # ── _read_config normalisation ───────────────────────────────────────
    def test_read_config_normalises_feeds(self):
        feeds_raw = [
            "https://plain.example/rss",                      # str form
            {"name": " Tech ", "url": "https://tech.example"},  # dict form
            {"name": "broken"},                                # no url → dropped
        ]
        with mock.patch.object(self.mod, "_config",
                               side_effect=lambda name, default: feeds_raw
                               if name == "NEWS_BRIEFING_FEEDS" else default):
            cfg = self.mod._read_config()
        self.assertEqual(len(cfg["feeds"]), 2)
        self.assertEqual(cfg["feeds"][0], {"name": "", "url": "https://plain.example/rss"})
        self.assertEqual(cfg["feeds"][1], {"name": "Tech", "url": "https://tech.example"})

    def test_read_config_count_clamped_and_ttl_seconds(self):
        def fake(name, default):
            return {"NEWS_BRIEFING_HEADLINE_COUNT": 0,    # → clamped up to 1
                    "NEWS_BRIEFING_CACHE_MINUTES": 30}.get(name, default)
        with mock.patch.object(self.mod, "_config", side_effect=fake):
            cfg = self.mod._read_config()
        self.assertEqual(cfg["count"], 1)
        self.assertEqual(cfg["ttl"], 30 * 60)

    # ── _gather_headlines round-robin ────────────────────────────────────
    def test_gather_headlines_round_robins_across_feeds(self):
        cfg = {
            "feeds": [{"name": "a", "url": "ua"}, {"name": "b", "url": "ub"}],
            "count": 3, "timeout": 1.0, "ttl": 0,
        }
        feed_data = {
            "ua": [{"title": "A1", "description": ""}, {"title": "A2", "description": ""}],
            "ub": [{"title": "B1", "description": ""}],
        }
        with mock.patch.object(self.mod, "_fetch_feed_cached",
                               side_effect=lambda url, t, ttl: list(feed_data[url])):
            out = self.mod._gather_headlines(cfg)
        titles = [h["title"] for h in out]
        # Round-robin: A1, B1, then back to A2 (B exhausted).
        self.assertEqual(titles, ["A1", "B1", "A2"])
        self.assertEqual(out[0]["feed_name"], "a")

    # ── get_news_text graceful degradation ───────────────────────────────
    def test_get_news_text_disabled(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"enabled": False, "feeds": [{"url": "x"}],
                                             "count": 1, "summarize": False, "ttl": 0,
                                             "timeout": 1.0}):
            self.assertEqual(self.mod.get_news_text(), "")

    def test_get_news_text_no_headlines(self):
        cfg = {"enabled": True, "feeds": [{"name": "", "url": "x"}],
               "count": 1, "summarize": False, "ttl": 0, "timeout": 1.0}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_gather_headlines", return_value=[]):
            self.assertEqual(self.mod.get_news_text(), "")

    def test_get_news_text_assembles_paragraph_no_llm(self):
        cfg = {"enabled": True, "feeds": [{"name": "", "url": "x"}],
               "count": 2, "summarize": False, "ttl": 0, "timeout": 1.0}
        heads = [{"title": "Rates rise", "description": "", "feed_name": ""},
                 {"title": "Storm clears", "description": "", "feed_name": ""}]
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_gather_headlines", return_value=heads):
            text = self.mod.get_news_text()
        self.assertTrue(text.startswith("Today's headlines"))
        self.assertIn("Rates rise.", text)
        self.assertIn("Storm clears.", text)

    # ── _summarize_via_llm degradation (backend down → title verbatim) ────
    def test_summarize_returns_title_when_backend_unavailable(self):
        # Stub bobert_companion so the LLM paths are inert: non-claude backend
        # skips the Anthropic call, and _call_local_llm returns None → the
        # documented fallback hands the raw title back. No real network/GPU.
        fake_bc = mock.MagicMock()
        fake_bc.AI_BACKEND = "other"
        fake_bc._call_local_llm.return_value = None
        import sys
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            out = self.mod._summarize_via_llm("Raw title", "desc")
        self.assertEqual(out, "Raw title")

    # ── news_briefing action ─────────────────────────────────────────────
    def test_action_no_feeds_message(self):
        with mock.patch.object(self.mod, "get_news_text", return_value=""):
            out = self.actions["news_briefing"]("")
        self.assertIn("No news feeds responded", out)

    def test_action_enqueues_with_intent_tag(self):
        with mock.patch.object(self.mod, "get_news_text", return_value="Today's headlines, sir. X."), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            out = self.actions["news_briefing"]("")
        self.assertIn("Today's headlines", out)
        enq.assert_called_once()
        self.assertTrue(enq.call_args[0][0].startswith("[intent:briefing]"))


if __name__ == "__main__":
    unittest.main()
