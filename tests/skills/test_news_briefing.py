"""Logic tests for skills/news_briefing.py.

Covers the pure RSS/Atom plumbing (HTML stripping, stdlib feed parsing, config
normalisation, round-robin headline gathering) and the briefing assembly with
network + LLM mocked out. get_news_text()'s graceful-degradation contract
(disabled / no feeds / all feeds fail → "") is verified explicitly.
"""
from __future__ import annotations

import contextlib
import json
import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules, restoring the prior
    state (including absence) on exit; dotted leaves are also set on the parent
    package. Mirrors tests/skills/test_self_diagnostic.py's isolation contract."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                with contextlib.suppress(AttributeError):
                    delattr(parent, leaf)
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


class _FakeResp:
    """Context-manager stand-in for urllib.request.urlopen()'s return."""
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


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


class NewsConfigTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("news_briefing")

    def test_config_returns_default_when_bc_unimportable(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.assertEqual(self.mod._config("ANYTHING", "fallback"), "fallback")

    def test_config_reads_attr_from_bc(self):
        bc = types.ModuleType("bobert_companion")
        bc.NEWS_BRIEFING_HEADLINE_COUNT = 9
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._config("NEWS_BRIEFING_HEADLINE_COUNT", 3), 9)

    def test_read_config_defaults_when_no_bc(self):
        # No NEWS_* overrides → built-in defaults; the default feed list is used.
        with mock.patch.object(self.mod, "_config",
                               side_effect=lambda name, default: default):
            cfg = self.mod._read_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["count"], self.mod._DEFAULT_HEADLINE_COUNT)
        self.assertEqual(len(cfg["feeds"]), len(self.mod._DEFAULT_FEEDS))
        self.assertEqual(cfg["ttl"], self.mod._DEFAULT_CACHE_MINUTES * 60)

    def test_read_config_empty_feeds_list(self):
        def fake(name, default):
            return [] if name == "NEWS_BRIEFING_FEEDS" else default
        with mock.patch.object(self.mod, "_config", side_effect=fake):
            cfg = self.mod._read_config()
        self.assertEqual(cfg["feeds"], [])


class NewsFeedparserParseTests(unittest.TestCase):
    """_parse_with_feedparser with a fake feedparser injected (the real one is
    on CI, but we control the parse result and avoid any network)."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("news_briefing")

    def _fake_feedparser(self, entries, parse_raises=False):
        fp = types.ModuleType("feedparser")
        if parse_raises:
            fp.parse = mock.MagicMock(side_effect=RuntimeError("bad bytes"))
        else:
            fp.parse = mock.MagicMock(
                return_value=types.SimpleNamespace(entries=entries))
        return fp

    def test_parse_extracts_title_and_summary(self):
        entries = [{"title": " <b>Hello</b> ", "summary": "<p>world</p>"}]
        with inject_modules(feedparser=self._fake_feedparser(entries)):
            out = self.mod._parse_with_feedparser("<rss/>")
        self.assertEqual(out, [{"title": "Hello", "description": "world"}])

    def test_parse_falls_back_to_description_field(self):
        entries = [{"title": "T", "description": "desc-field"}]
        with inject_modules(feedparser=self._fake_feedparser(entries)):
            out = self.mod._parse_with_feedparser("<rss/>")
        self.assertEqual(out[0]["description"], "desc-field")

    def test_parse_skips_entries_without_title(self):
        entries = [{"summary": "orphan"}, {"title": "Keep"}]
        with inject_modules(feedparser=self._fake_feedparser(entries)):
            out = self.mod._parse_with_feedparser("<rss/>")
        self.assertEqual([e["title"] for e in out], ["Keep"])

    def test_parse_returns_empty_when_feedparser_absent(self):
        # Simulate feedparser not installed: import inside the function raises.
        real_import = __import__

        def _imp(name, *a, **k):
            if name == "feedparser":
                raise ImportError("not installed")
            return real_import(name, *a, **k)
        with inject_modules(feedparser=None), \
             mock.patch("builtins.__import__", side_effect=_imp):
            self.assertEqual(self.mod._parse_with_feedparser("<rss/>"), [])

    def test_parse_returns_empty_when_parse_raises(self):
        with inject_modules(feedparser=self._fake_feedparser([], parse_raises=True)):
            self.assertEqual(self.mod._parse_with_feedparser("<rss/>"), [])

    def test_parse_empty_entries(self):
        with inject_modules(feedparser=self._fake_feedparser([])):
            self.assertEqual(self.mod._parse_with_feedparser("<rss/>"), [])


class NewsStdlibContentTests(unittest.TestCase):
    """Atom <content> path + child.itertext() fallback in _parse_with_stdlib."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("news_briefing")

    def test_atom_content_with_nested_html(self):
        # <content> begins with a child element (no direct text) so the parser
        # falls back to itertext(), stitching the nested markup together.
        xml = (
            '<feed xmlns="http://www.w3.org/2005/Atom">'
            "<entry><title>Story</title>"
            '<content type="html"><b>bold</b> tail</content></entry>'
            "</feed>"
        )
        items = self.mod._parse_with_stdlib(xml)
        self.assertEqual(items[0]["title"], "Story")
        self.assertIn("bold", items[0]["description"])

    def test_title_with_nested_markup_uses_itertext(self):
        # Title element has no direct .text but nested children → itertext path.
        xml = (
            "<rss><channel><item>"
            "<title><b>Nested</b> title</title>"
            "<description>d</description>"
            "</item></channel></rss>"
        )
        items = self.mod._parse_with_stdlib(xml)
        self.assertEqual(items[0]["title"], "Nested title")


class NewsFetchTests(unittest.TestCase):
    """_fetch_feed (urllib mocked) + _fetch_feed_cached TTL/stale logic."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("news_briefing")
        # Start every fetch test from an empty in-process cache and restore it.
        self._saved_cache = dict(self.mod._feed_cache)
        self.mod._feed_cache.clear()
        self.addCleanup(self._restore_cache)

    def _restore_cache(self):
        self.mod._feed_cache.clear()
        self.mod._feed_cache.update(self._saved_cache)

    _RSS = (b"<rss><channel><item><title>Fetched headline</title>"
            b"<description>body</description></item></channel></rss>")

    def test_fetch_feed_parses_via_stdlib_when_no_feedparser(self):
        # feedparser path returns [] (absent) → stdlib parser handles the bytes.
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_FakeResp(self._RSS)), \
             mock.patch.object(self.mod, "_parse_with_feedparser", return_value=[]):
            items = self.mod._fetch_feed("http://x/rss", 1.0)
        self.assertEqual(items[0]["title"], "Fetched headline")

    def test_fetch_feed_prefers_feedparser_result(self):
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_FakeResp(self._RSS)), \
             mock.patch.object(self.mod, "_parse_with_feedparser",
                               return_value=[{"title": "FP", "description": ""}]):
            items = self.mod._fetch_feed("http://x/rss", 1.0)
        self.assertEqual(items[0]["title"], "FP")

    def test_fetch_feed_http_error_returns_empty(self):
        err = self.mod.urllib.error.HTTPError(
            "http://x/rss", 503, "Service Unavailable", {}, None)
        with mock.patch.object(self.mod.urllib.request, "urlopen", side_effect=err):
            self.assertEqual(self.mod._fetch_feed("http://x/rss", 1.0), [])

    def test_fetch_feed_generic_error_returns_empty(self):
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=OSError("connection refused")):
            self.assertEqual(self.mod._fetch_feed("http://x/rss", 1.0), [])

    def test_fetch_feed_decode_error_returns_empty(self):
        # raw.decode raises → the decode-guard returns [].
        bad = mock.MagicMock()
        bad.decode.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "boom")
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = bad
        resp.__exit__.return_value = False
        with mock.patch.object(self.mod.urllib.request, "urlopen", return_value=resp):
            self.assertEqual(self.mod._fetch_feed("http://x/rss", 1.0), [])

    # ── _fetch_feed_cached ────────────────────────────────────────────────
    def test_cached_returns_fresh_without_refetch(self):
        self.mod._feed_cache["u"] = {"fetched_at": self.mod.time.time(),
                                     "items": [{"title": "cached", "description": ""}]}
        with mock.patch.object(self.mod, "_fetch_feed") as fetch:
            out = self.mod._fetch_feed_cached("u", 1.0, ttl=999)
        fetch.assert_not_called()
        self.assertEqual(out[0]["title"], "cached")

    def test_cached_refetches_when_expired(self):
        self.mod._feed_cache["u"] = {"fetched_at": self.mod.time.time() - 10_000,
                                     "items": [{"title": "old", "description": ""}]}
        with mock.patch.object(self.mod, "_fetch_feed",
                               return_value=[{"title": "new", "description": ""}]):
            out = self.mod._fetch_feed_cached("u", 1.0, ttl=60)
        self.assertEqual(out[0]["title"], "new")
        # Cache updated to the fresh items.
        self.assertEqual(self.mod._feed_cache["u"]["items"][0]["title"], "new")

    def test_cached_falls_back_to_stale_on_fetch_failure(self):
        self.mod._feed_cache["u"] = {"fetched_at": self.mod.time.time() - 10_000,
                                     "items": [{"title": "stale", "description": ""}]}
        with mock.patch.object(self.mod, "_fetch_feed", return_value=[]):
            out = self.mod._fetch_feed_cached("u", 1.0, ttl=60)
        self.assertEqual(out[0]["title"], "stale")

    def test_cached_miss_then_fetch_populates_cache(self):
        with mock.patch.object(self.mod, "_fetch_feed",
                               return_value=[{"title": "fresh", "description": ""}]):
            out = self.mod._fetch_feed_cached("brand-new", 1.0, ttl=60)
        self.assertEqual(out[0]["title"], "fresh")
        self.assertIn("brand-new", self.mod._feed_cache)

    def test_cached_miss_fetch_fails_returns_empty(self):
        with mock.patch.object(self.mod, "_fetch_feed", return_value=[]):
            out = self.mod._fetch_feed_cached("nada", 1.0, ttl=60)
        self.assertEqual(out, [])
        self.assertNotIn("nada", self.mod._feed_cache)


class NewsSummarizeTests(unittest.TestCase):
    """_summarize_via_llm — Claude primary path (fake anthropic) + local
    fallback + cleaning + degradation. No real network/LLM."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("news_briefing")

    def _fake_anthropic(self, text=None, raises=False):
        anthropic = types.ModuleType("anthropic")

        class _Anthropic:
            def __init__(self, *a, **k):
                self.messages = types.SimpleNamespace(create=self._create)

            def _create(self, **kwargs):
                if raises:
                    raise RuntimeError("api down")
                block = types.SimpleNamespace(text=text)
                return types.SimpleNamespace(content=[block])
        anthropic.Anthropic = _Anthropic
        return anthropic

    def test_summarize_claude_path_returns_clean_sentence(self):
        bc = types.ModuleType("bobert_companion")
        bc.AI_BACKEND = "claude"
        bc.CLAUDE_MODEL = "claude-test"
        with inject_modules(bobert_companion=bc,
                            anthropic=self._fake_anthropic(text="Rates climbed today.")):
            out = self.mod._summarize_via_llm("Rates up", "desc")
        self.assertEqual(out, "Rates climbed today.")

    def test_summarize_claude_strips_stray_intent_tag(self):
        bc = types.ModuleType("bobert_companion")
        bc.AI_BACKEND = "claude"
        bc.CLAUDE_MODEL = "claude-test"
        tagged = "[intent:briefing] The market dipped."
        with inject_modules(bobert_companion=bc,
                            anthropic=self._fake_anthropic(text=tagged)):
            out = self.mod._summarize_via_llm("Market", "")
        self.assertEqual(out, "The market dipped.")   # tag removed

    def test_summarize_claude_empty_reply_falls_back_to_title(self):
        bc = types.ModuleType("bobert_companion")
        bc.AI_BACKEND = "claude"
        bc.CLAUDE_MODEL = "claude-test"
        # Claude returns blank → _clean returns the title; local not consulted
        # because the claude branch already returned.
        with inject_modules(bobert_companion=bc,
                            anthropic=self._fake_anthropic(text="   ")):
            out = self.mod._summarize_via_llm("Fallback title", "")
        self.assertEqual(out, "Fallback title")

    def test_summarize_claude_error_then_local_succeeds(self):
        bc = types.ModuleType("bobert_companion")
        bc.AI_BACKEND = "claude"
        bc.CLAUDE_MODEL = "claude-test"
        bc._call_local_llm = mock.MagicMock(return_value="Local sentence.")
        with inject_modules(bobert_companion=bc,
                            anthropic=self._fake_anthropic(raises=True)):
            out = self.mod._summarize_via_llm("Title", "body")
        self.assertEqual(out, "Local sentence.")
        bc._call_local_llm.assert_called_once()

    def test_summarize_non_claude_backend_uses_local(self):
        bc = types.ModuleType("bobert_companion")
        bc.AI_BACKEND = "ollama"
        bc._call_local_llm = mock.MagicMock(return_value="From local model.")
        with inject_modules(bobert_companion=bc):
            out = self.mod._summarize_via_llm("Title", "")
        self.assertEqual(out, "From local model.")

    def test_summarize_local_raises_returns_title(self):
        bc = types.ModuleType("bobert_companion")
        bc.AI_BACKEND = "ollama"
        bc._call_local_llm = mock.MagicMock(side_effect=RuntimeError("ollama gone"))
        with inject_modules(bobert_companion=bc):
            out = self.mod._summarize_via_llm("Resilient title", "")
        self.assertEqual(out, "Resilient title")

    def test_summarize_bc_import_fails_returns_title(self):
        # bobert_companion not importable at all → immediate title fallback.
        with inject_modules(bobert_companion=None), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            out = self.mod._summarize_via_llm("Just the title", "x")
        self.assertEqual(out, "Just the title")

    def test_summarize_truncates_long_description(self):
        # A >500 char description is truncated in the user prompt. Verify the
        # local model receives the trimmed text.
        bc = types.ModuleType("bobert_companion")
        bc.AI_BACKEND = "ollama"
        captured = {}

        def _local(system, messages, max_tokens=120):
            captured["content"] = messages[0]["content"]
            return "ok."
        bc._call_local_llm = _local
        long_desc = "x" * 1000
        with inject_modules(bobert_companion=bc):
            self.mod._summarize_via_llm("Title", long_desc)
        # 500-char cap + the "Description: " prefix; never the full 1000.
        self.assertLess(len(captured["content"]), 600)
        self.assertIn("Title", captured["content"])


class NewsAssemblyTests(unittest.TestCase):
    """_gather_headlines edge cases + get_news_text with summarize on, plus the
    no-feeds short-circuit and the empty-sentence guard."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("news_briefing")

    def test_gather_stops_when_all_feeds_empty(self):
        cfg = {"feeds": [{"name": "a", "url": "ua"}, {"name": "b", "url": "ub"}],
               "count": 10, "timeout": 1.0, "ttl": 0}
        with mock.patch.object(self.mod, "_fetch_feed_cached", return_value=[]):
            out = self.mod._gather_headlines(cfg)
        self.assertEqual(out, [])

    def test_gather_skips_empty_feed_slot(self):
        # Feed "a" (index 0) yields nothing while "b" has items: the round-robin
        # must skip the empty slot and still collect from the non-empty feed.
        cfg = {"feeds": [{"name": "a", "url": "ua"}, {"name": "b", "url": "ub"}],
               "count": 2, "timeout": 1.0, "ttl": 0}
        data = {"ua": [], "ub": [{"title": "B1", "description": ""},
                                 {"title": "B2", "description": ""}]}
        with mock.patch.object(self.mod, "_fetch_feed_cached",
                               side_effect=lambda u, t, ttl: list(data[u])):
            out = self.mod._gather_headlines(cfg)
        self.assertEqual([h["title"] for h in out], ["B1", "B2"])

    def test_gather_single_feed_caps_at_count(self):
        cfg = {"feeds": [{"name": "a", "url": "ua"}], "count": 2,
               "timeout": 1.0, "ttl": 0}
        data = [{"title": f"H{i}", "description": ""} for i in range(5)]
        with mock.patch.object(self.mod, "_fetch_feed_cached",
                               side_effect=lambda u, t, ttl: list(data)):
            out = self.mod._gather_headlines(cfg)
        self.assertEqual([h["title"] for h in out], ["H0", "H1"])

    def test_get_news_text_no_feeds_short_circuits(self):
        cfg = {"enabled": True, "feeds": [], "count": 1, "summarize": False,
               "ttl": 0, "timeout": 1.0}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg):
            self.assertEqual(self.mod.get_news_text(), "")

    def test_get_news_text_summarizes_each_headline(self):
        cfg = {"enabled": True, "feeds": [{"name": "", "url": "x"}],
               "count": 2, "summarize": True, "ttl": 0, "timeout": 1.0}
        heads = [{"title": "Raw one", "description": "d1", "feed_name": ""},
                 {"title": "Raw two", "description": "d2", "feed_name": ""}]
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_gather_headlines", return_value=heads), \
             mock.patch.object(self.mod, "_summarize_via_llm",
                               side_effect=lambda t, d: f"Summary of {t}"):
            text = self.mod.get_news_text()
        self.assertIn("Summary of Raw one.", text)
        self.assertIn("Summary of Raw two.", text)

    def test_get_news_text_drops_blank_summaries(self):
        cfg = {"enabled": True, "feeds": [{"name": "", "url": "x"}],
               "count": 2, "summarize": True, "ttl": 0, "timeout": 1.0}
        heads = [{"title": "Good", "description": "", "feed_name": ""},
                 {"title": "Blank", "description": "", "feed_name": ""}]

        def _summ(t, d):
            return "" if t == "Blank" else "Kept"
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_gather_headlines", return_value=heads), \
             mock.patch.object(self.mod, "_summarize_via_llm", side_effect=_summ):
            text = self.mod.get_news_text()
        self.assertIn("Kept.", text)
        self.assertNotIn("Blank", text)

    def test_get_news_text_all_blank_returns_empty(self):
        cfg = {"enabled": True, "feeds": [{"name": "", "url": "x"}],
               "count": 1, "summarize": True, "ttl": 0, "timeout": 1.0}
        heads = [{"title": "x", "description": "", "feed_name": ""}]
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_gather_headlines", return_value=heads), \
             mock.patch.object(self.mod, "_summarize_via_llm", return_value="   "):
            self.assertEqual(self.mod.get_news_text(), "")


class NewsEnqueueTests(unittest.TestCase):
    """_enqueue_speech — announcer path + atomic-write fallback variants."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("news_briefing")

    def test_enqueue_uses_bc_announcer(self):
        bc = types.ModuleType("bobert_companion")
        calls = []
        bc.proactive_announce = lambda msg, source=None: calls.append((msg, source))
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("[intent:briefing] hi")
        self.assertEqual(calls, [("[intent:briefing] hi", "news")])

    def test_enqueue_falls_back_to_atomic_write(self):
        bc = types.ModuleType("bobert_companion")   # no proactive_announce
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("hello")
        wr.assert_called_once()
        path, data = wr.call_args[0][0], wr.call_args[0][1]
        self.assertEqual(path, self.mod._SPEECH_QUEUE)
        self.assertEqual(data[-1]["message"], "hello")

    def test_enqueue_appends_to_existing_queue(self):
        bc = types.ModuleType("bobert_companion")
        existing = json.dumps([{"ts": 1.0, "message": "old"}])
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=existing)), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("new")
        data = wr.call_args[0][1]
        self.assertEqual([d["message"] for d in data], ["old", "new"])

    def test_enqueue_corrupt_queue_resets_to_list(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{garbage")), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("fresh")
        data = wr.call_args[0][1]
        self.assertEqual([d["message"] for d in data], ["fresh"])

    def test_enqueue_non_list_json_resets_to_list(self):
        # Existing file decodes to a dict (not a list) → data stays [] and only
        # the new message is written.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data='{"not": "a list"}')), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("solo")
        data = wr.call_args[0][1]
        self.assertEqual([d["message"] for d in data], ["solo"])

    def test_enqueue_empty_existing_file(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="   ")), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("only")
        data = wr.call_args[0][1]
        self.assertEqual([d["message"] for d in data], ["only"])

    def test_enqueue_read_raises_resets_to_list(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("after-error")
        data = wr.call_args[0][1]
        self.assertEqual([d["message"] for d in data], ["after-error"])

    def test_enqueue_announcer_raises_falls_through(self):
        bc = types.ModuleType("bobert_companion")

        def _boom(*_a, **_k):
            raise RuntimeError("announcer broke")
        bc.proactive_announce = _boom
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("still queued")
        wr.assert_called_once()

    def test_enqueue_atomic_write_failure_swallowed(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only share")):
            self.mod._enqueue_speech("resilient")   # must not raise

    def test_enqueue_announcer_not_callable_falls_through(self):
        # proactive_announce present but not callable → fall through to write.
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = "not a function"
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("written")
        wr.assert_called_once()


class NewsRegisterTests(unittest.TestCase):
    def test_register_enabled_adds_action(self):
        mod, _ = load_skill_isolated("news_briefing", register=False)
        actions: dict = {}
        cfg = {"enabled": True, "feeds": [{"name": "", "url": "x"}], "count": 3,
               "summarize": True, "ttl": 0, "timeout": 1.0}
        with mock.patch.object(mod, "_read_config", return_value=cfg):
            mod.register(actions)
        self.assertIn("news_briefing", actions)

    def test_register_disabled_still_adds_action(self):
        mod, _ = load_skill_isolated("news_briefing", register=False)
        actions: dict = {}
        cfg = {"enabled": False, "feeds": [], "count": 3, "summarize": True,
               "ttl": 0, "timeout": 1.0}
        with mock.patch.object(mod, "_read_config", return_value=cfg):
            mod.register(actions)
        self.assertIn("news_briefing", actions)

    def test_action_enqueues_real_path(self):
        # End-to-end action with get_news_text stubbed and enqueue going through
        # the announcer (no file written).
        mod, actions = load_skill_isolated("news_briefing")
        bc = types.ModuleType("bobert_companion")
        sent = []
        bc.proactive_announce = lambda msg, source=None: sent.append(msg)
        with mock.patch.object(mod, "get_news_text",
                               return_value="Today's headlines, sir. X."), \
             inject_modules(bobert_companion=bc):
            out = actions["news_briefing"]("")
        self.assertIn("Today's headlines", out)
        self.assertTrue(sent[0].startswith("[intent:briefing]"))


if __name__ == "__main__":
    unittest.main()
