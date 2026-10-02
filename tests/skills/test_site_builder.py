"""Logic tests for skills/site_builder.py ("build a website for <business>").

Fakes only: the fact lookup is a stand-in dossier module in sys.modules, the
cloud call is a patched core.llm_client.complete, the local model is the fake
skill_utils["local_complete"], the browser is the fake skill_utils["open_url"],
the announce path is a stand-in bobert_companion.proactive_announce and the
data dir is a temp dir via JARVIS_DATA_DIR. No network, no LLM, no browser.

The build runs on a background worker. Every test replaces the module's
_start_worker seam with a recorder, then runs the recorded worker itself,
synchronously — so no real thread is ever started (the one test of the real
seam mocks threading.Thread). Fixtures use generic business names only.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tests._skill_harness import load_skill_isolated, make_fake_skill_utils

_PAGE = (
    "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<title>Blue Door Bakery</title>\n<style>body{margin:0}</style>\n</head>\n"
    "<body>\n<section id=\"hero\"><h1>Blue Door Bakery</h1></section>\n"
    "</body>\n</html>"
)
_FACTS = "Blue Door Bakery is a neighbourhood bakery known for sourdough."
_STARTED = ("Building a site for Blue Door Bakery now, sir — I'll tell you "
            "when it's ready.")
_SENTINEL = object()


@contextlib.contextmanager
def _module(name, obj):
    """Install (or, with obj=None, remove) sys.modules[name]; restore after."""
    prev = sys.modules.get(name, _SENTINEL)
    if obj is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = obj
    try:
        yield
    finally:
        if prev is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


def _dossier(facts=_FACTS):
    mod = types.ModuleType("skill_dossier")
    mod._gather_web = mock.MagicMock(return_value=facts)
    return mod


class _Run:
    """One build_website call plus its worker: the immediate reply, every
    announced line, the cloud-call mock and the captured log."""

    def __init__(self, reply, announced, complete, out):
        self.reply, self.announced = reply, announced
        self.complete, self.out = complete, out


class _SiteBuilderCase(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="site_builder_test_")
        self.addCleanup(shutil.rmtree, self.data, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.data,
                                           "ANTHROPIC_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)
        self.utils = make_fake_skill_utils()
        self.mod, self.actions = load_skill_isolated(
            "site_builder", utils=self.utils)
        # The executor seam: record the worker instead of starting a thread.
        self.real_start_worker = self.mod._start_worker
        self.started = []
        self.mod._start_worker = (
            lambda target, *args: self.started.append((target, args)))
        # The announce path: a stand-in monolith with proactive_announce.
        self.announce = mock.MagicMock(name="proactive_announce",
                                       return_value=True)
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = self.announce
        cm = _module("bobert_companion", bc)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def run_workers(self):
        while self.started:
            target, args = self.started.pop(0)
            target(*args)

    def run_build(self, arg, *, cloud=True, cloud_reply=_PAGE,
                  local_reply=None, dossier=_SENTINEL):
        """build_website(arg) (skipped for arg=None), then every queued
        worker, with the gate, the cloud call and the lookup faked."""
        self.utils["local_complete"].return_value = local_reply
        if dossier is _SENTINEL:
            dossier = _dossier()
        before = self.announce.call_count
        out = io.StringIO()
        with mock.patch("core.cloud_gate.chat_cloud_allowed",
                        return_value=cloud), \
                mock.patch("core.llm_client.complete",
                           return_value=cloud_reply) as complete, \
                _module("skill_dossier", dossier), \
                contextlib.redirect_stdout(out):
            reply = (None if arg is None
                     else self.actions["build_website"](arg))
            self.run_workers()
        announced = [c.args[0] for c in
                     self.announce.call_args_list[before:]]
        return _Run(reply, announced, complete, out.getvalue())

    def site_file(self, slug):
        return os.path.join(self.data, "sites", slug, "index.html")


class RegisterTests(_SiteBuilderCase):
    def test_registers_build_website_spoken_verbatim(self):
        self.assertIs(self.actions["build_website"], self.mod.build_website)
        self.assertIn("build_website", self.mod.SPEAK_VERBATIM_ACTIONS)

    def test_no_longer_marked_long_running(self):
        # It returns at once now; the mid-task "still working" line would
        # only ever fire on a stuck enqueue.
        fake_bc = types.ModuleType("bobert_companion")
        fake_bc.LONG_RUNNING_ACTIONS = set()
        with _module("bobert_companion", fake_bc):
            self.mod.register({})
        self.assertNotIn("build_website", fake_bc.LONG_RUNNING_ACTIONS)


class BackgroundTests(_SiteBuilderCase):
    def test_returns_at_once_before_any_work(self):
        with mock.patch("core.llm_client.complete") as complete, \
                _module("skill_dossier", _dossier()):
            reply = self.actions["build_website"]("Blue Door Bakery")
        self.assertEqual(reply, _STARTED)
        self.assertEqual(len(self.started), 1)
        complete.assert_not_called()
        self.utils["local_complete"].assert_not_called()
        self.announce.assert_not_called()
        self.utils["open_url"].assert_not_called()
        self.assertFalse(os.path.exists(os.path.join(self.data, "sites")))

    def test_single_flight(self):
        first = self.actions["build_website"]("Blue Door Bakery")
        second = self.actions["build_website"]("Green Gate Garage")
        self.assertEqual(first, _STARTED)
        self.assertEqual(second,
                         "I'm still building the Blue Door Bakery site, sir.")
        self.assertEqual(len(self.started), 1, "a second worker was started")
        r = self.run_build(None)          # run only the one queued worker
        self.assertEqual(len(r.announced), 1)
        self.assertIn("Blue Door Bakery is ready", r.announced[0])
        # Finished: the next request starts a new build.
        self.assertEqual(self.actions["build_website"]("Green Gate Garage"),
                         "Building a site for Green Gate Garage now, sir — "
                         "I'll tell you when it's ready.")
        self.assertEqual(len(self.started), 1)

    def test_completion_announces_once_then_opens_the_page(self):
        order = mock.Mock()
        order.attach_mock(self.announce, "announce")
        order.attach_mock(self.utils["open_url"], "open_url")
        r = self.run_build("Blue Door Bakery")
        self.assertEqual(r.reply, _STARTED)
        self.assertEqual(len(r.announced), 1)
        self.assertIn("The website for Blue Door Bakery is ready, sir",
                      r.announced[0])
        self.assertIn("sites/blue-door-bakery", r.announced[0])
        self.assertEqual(self.announce.call_args.kwargs,
                         {"source": "site_builder"})
        uri = Path(self.site_file("blue-door-bakery")).as_uri()
        self.utils["open_url"].assert_called_once_with(uri)
        self.assertEqual([c[0] for c in order.mock_calls],
                         ["announce", "open_url"])

    def test_failure_announces_honestly_once_and_opens_nothing(self):
        r = self.run_build("Blue Door Bakery", cloud=False, local_reply=None)
        self.assertEqual(r.reply, _STARTED)
        self.assertEqual(r.announced, [
            "I couldn't build the Blue Door Bakery website, sir — Claude "
            "isn't available for this, and the local model couldn't write "
            "the page."])
        self.utils["open_url"].assert_not_called()
        self.assertFalse(os.path.exists(os.path.join(self.data, "sites")))

    def test_a_crashing_build_is_announced_and_frees_the_slot(self):
        with mock.patch.object(self.mod, "_build_site",
                               side_effect=RuntimeError("boom")):
            r = self.run_build("Blue Door Bakery")
        self.assertEqual(r.announced, [
            "I couldn't build the Blue Door Bakery website, sir — something "
            "went wrong partway through."])
        self.assertNotIn("boom", r.out)
        self.assertEqual(self.actions["build_website"]("Blue Door Bakery"),
                         _STARTED)

    def test_nothing_is_spoken_without_an_announce_path(self):
        for label, bc in (
                ("no monolith", None),
                ("no proactive_announce", types.ModuleType("bobert_companion")),
                ("enqueue refused", types.SimpleNamespace(
                    proactive_announce=mock.MagicMock(return_value=False))),
                ("enqueue raises", types.SimpleNamespace(
                    proactive_announce=mock.MagicMock(
                        side_effect=OSError("disk"))))):
            with self.subTest(label), _module("bobert_companion", bc):
                self.utils["open_url"].reset_mock()
                r = self.run_build("Blue Door Bakery")
                self.assertEqual(r.reply, _STARTED)
                self.assertIn("not spoken", r.out)
                self.assertIn("The website for Blue Door Bakery is ready",
                              r.out)
                # The page is still built and opened.
                self.assertTrue(os.path.isfile(
                    self.site_file("blue-door-bakery")))
                self.utils["open_url"].assert_called_once()
        self.announce.assert_not_called()

    def test_start_failure_frees_the_slot(self):
        self.mod._start_worker = mock.MagicMock(side_effect=RuntimeError("x"))
        with contextlib.redirect_stdout(io.StringIO()):
            reply = self.actions["build_website"]("Blue Door Bakery")
        self.assertEqual(reply, "I couldn't start building that website, sir.")
        self.mod._start_worker = (
            lambda target, *args: self.started.append((target, args)))
        self.assertEqual(self.actions["build_website"]("Blue Door Bakery"),
                         _STARTED)

    def test_real_seam_starts_one_daemon_thread(self):
        target = mock.MagicMock()
        with mock.patch.object(self.mod.threading, "Thread") as thread:
            self.real_start_worker(target, "a", "b", "c")
        thread.assert_called_once_with(target=target, args=("a", "b", "c"),
                                       daemon=True, name="site-builder")
        thread.return_value.start.assert_called_once_with()
        target.assert_not_called()


class SlugTests(_SiteBuilderCase):
    def test_slugs(self):
        s = self.mod._slugify
        self.assertEqual(s("Blue Door Bakery"), "blue-door-bakery")
        self.assertEqual(s("The Baker's Dozen"), "the-bakers-dozen")
        self.assertEqual(s("Corner Café & Bar!"), "corner-cafe-and-bar")
        self.assertEqual(s("  --Hello   World--  "), "hello-world")
        self.assertEqual(s("../../etc/passwd"), "etc-passwd")
        self.assertEqual(s(""), "site")
        self.assertEqual(s("!!!"), "site")
        self.assertEqual(s("Con"), "con-site")       # Windows device name

    def test_long_names_are_capped_without_a_trailing_hyphen(self):
        slug = self.mod._slugify("word " * 40)
        self.assertLessEqual(len(slug), self.mod._SLUG_MAX_LEN)
        self.assertFalse(slug.endswith("-"))

    def test_arg_parsing(self):
        p = self.mod._parse_arg
        self.assertEqual(p("Blue Door Bakery"), ("Blue Door Bakery", "", ""))
        self.assertEqual(p("for Blue Door Bakery | Springfield | sourdough"),
                         ("Blue Door Bakery", "Springfield", "sourdough"))
        self.assertEqual(p("'Blue Door' | | a | b"), ("Blue Door", "", "a | b"))
        self.assertEqual(p(""), ("", "", ""))


class SaveLocationTests(_SiteBuilderCase):
    def test_page_is_written_in_the_data_dir(self):
        self.run_build("Blue Door Bakery")
        path = self.site_file("blue-door-bakery")
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), _PAGE + "\n")
        # Nothing but the one page under the data dir.
        self.assertEqual(os.listdir(os.path.join(self.data, "sites")),
                         ["blue-door-bakery"])

    def test_save_refuses_a_folder_outside_the_sites_dir(self):
        for slug in ("..", "../escape", ""):
            with self.assertRaises(ValueError):
                self.mod._save_site(slug, _PAGE)
        self.assertFalse(os.path.exists(os.path.join(self.data, "escape")))

    def test_open_failure_is_logged_and_frees_the_slot(self):
        self.utils["open_url"].side_effect = RuntimeError("no browser")
        r = self.run_build("Blue Door Bakery")
        self.assertTrue(os.path.isfile(self.site_file("blue-door-bakery")))
        self.assertEqual(len(r.announced), 1)
        self.assertIn("open failed", r.out)
        self.assertEqual(self.actions["build_website"]("Blue Door Bakery"),
                         _STARTED)


class HtmlSanityTests(_SiteBuilderCase):
    def test_fenced_reply_with_chatter_is_unwrapped(self):
        raw = f"Here is your site:\n```html\n{_PAGE}\n```\nEnjoy!"
        self.assertEqual(self.mod._extract_html(raw), _PAGE + "\n")

    def test_external_scripts_are_stripped_and_viewport_added(self):
        raw = ("<html><head><title>x</title>"
               "<script src=\"https://cdn.example.com/x.js\"></script></head>"
               "<body><p>hi</p></body></html>")
        html = self.mod._extract_html(raw)
        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        self.assertNotIn("<script", html)
        self.assertIn('name="viewport"', html)
        self.assertTrue(html.rstrip().endswith("</html>"))

    def test_no_active_content_survives(self):
        # The page is written partly from fetched web text, so a prompt
        # injection could ask for script. The prompt already says "No
        # JavaScript"; the extractor enforces it: inline scripts, event-handler
        # attributes, javascript: links and embedded frames are all removed.
        raw = ("<html><head><title>x</title><script>steal()</script></head>"
               "<body onload=\"steal()\"><p onclick='x()'>hi</p>"
               "<a href=\"javascript:steal()\">menu</a>"
               "<a href=\"https://maps.example.com/?q=shop\">map</a>"
               "<iframe src=\"https://evil.example.com\"></iframe>"
               "<object data=\"x.swf\"></object><embed src=\"x.swf\">"
               "<SCRIPT type=module>later()</SCRIPT></body></html>")
        html = self.mod._extract_html(raw)
        low = html.lower()
        for bad in ("<script", "onload", "onclick", "javascript:", "<iframe",
                    "<object", "<embed", "steal", "later()"):
            self.assertNotIn(bad, low)
        self.assertIn("https://maps.example.com/?q=shop", html)
        self.assertIn("<p>hi</p>", html)

    def test_non_html_replies_are_rejected(self):
        for raw in (None, "", "Sorry, I can't help with that.",
                    "<html><head></head></html>", "</html> <html><body>"):
            self.assertIsNone(self.mod._extract_html(raw), raw)

    def test_prompt_asks_for_every_section_and_forbids_invention(self):
        r = self.run_build("Blue Door Bakery | Springfield")
        kw = r.complete.call_args.kwargs
        system = kw["system"].lower()
        for part in ("hero", "about", "menu or services", "hours", "location",
                     "contact", "call to action", "no javascript",
                     "never invent", "responsive"):
            self.assertIn(part, system)
        user = kw["messages"][0]["content"]
        self.assertIn("Business name: Blue Door Bakery", user)
        self.assertIn("City: Springfield", user)
        self.assertIn("https://www.google.com/maps/search/?api=1&query="
                      "Blue+Door+Bakery+Springfield", user)

    def test_unusable_model_output_saves_nothing(self):
        r = self.run_build("Blue Door Bakery", cloud_reply="no page here",
                           local_reply="still no page")
        self.assertFalse(os.path.exists(os.path.join(self.data, "sites")))
        self.utils["open_url"].assert_not_called()
        self.assertEqual(r.announced, [
            "I couldn't build the Blue Door Bakery website, sir — neither "
            "Claude nor the local model produced a usable page."])


class ModelRoutingTests(_SiteBuilderCase):
    def test_cloud_path_uses_opus_deep(self):
        r = self.run_build("Blue Door Bakery")
        r.complete.assert_called_once()
        kw = r.complete.call_args.kwargs
        self.assertEqual(kw["model"], "claude-opus-5-5")
        self.assertEqual(kw["purpose"], "deep")
        self.utils["local_complete"].assert_not_called()
        self.assertNotIn("local model", r.announced[0])

    def test_cloud_disabled_uses_the_local_long_reply_path(self):
        r = self.run_build("Blue Door Bakery", cloud=False, local_reply=_PAGE)
        r.complete.assert_not_called()
        call = self.utils["local_complete"].call_args
        self.assertEqual(call.kwargs["max_tokens"], self.mod.LOCAL_MAX_TOKENS)
        self.assertEqual(call.kwargs["timeout_s"], self.mod.LOCAL_TIMEOUT_S)
        self.assertGreaterEqual(self.mod.LOCAL_MAX_TOKENS, 4096)
        self.assertTrue(os.path.isfile(self.site_file("blue-door-bakery")))
        self.assertIn("built with the local model", r.announced[0])

    def test_no_key_never_calls_the_cloud(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            r = self.run_build("Blue Door Bakery", local_reply=_PAGE)
        r.complete.assert_not_called()

    def test_cloud_failure_falls_back_to_local(self):
        r = self.run_build("Blue Door Bakery", cloud_reply=None,
                           local_reply=_PAGE)
        r.complete.assert_called_once()
        self.utils["local_complete"].assert_called_once()
        self.assertTrue(os.path.isfile(self.site_file("blue-door-bakery")))
        self.assertIn("built with the local model", r.announced[0])


class FactsTests(_SiteBuilderCase):
    def test_facts_reach_the_prompt_but_never_the_log(self):
        dossier = _dossier()
        r = self.run_build("Blue Door Bakery | Springfield", dossier=dossier)
        dossier._gather_web.assert_called_once_with(
            "Blue Door Bakery Springfield")
        self.assertIn(_FACTS,
                      r.complete.call_args.kwargs["messages"][0]["content"])
        self.assertNotIn(_FACTS, r.out)
        self.assertNotIn("<section", r.out)        # nor the page itself
        self.assertNotIn("couldn't find anything", r.announced[0])

    def test_no_facts_builds_from_the_name_alone_and_says_so(self):
        r = self.run_build("Blue Door Bakery", dossier=_dossier(""))
        self.assertIn("nothing could be fetched",
                      r.complete.call_args.kwargs["messages"][0]["content"])
        self.assertTrue(os.path.isfile(self.site_file("blue-door-bakery")))
        self.assertIn("couldn't find anything about them online",
                      r.announced[0])
        self.assertIn("the name alone", r.announced[0])

    def test_no_lookup_helper_loaded_counts_as_no_facts(self):
        r = self.run_build("Blue Door Bakery | | sourdough", dossier=None)
        self.assertIn("couldn't find anything about them online",
                      r.announced[0])
        self.assertIn("your notes", r.announced[0])

    def test_failing_lookup_counts_as_no_facts(self):
        dossier = _dossier()
        dossier._gather_web.side_effect = OSError("offline")
        r = self.run_build("Blue Door Bakery", dossier=dossier)
        self.assertIn("couldn't find anything about them online",
                      r.announced[0])

    def test_empty_arg_asks_which_business(self):
        r = self.run_build("  ")
        self.assertIn("Which business", r.reply)
        self.assertEqual(self.started, [])
        r.complete.assert_not_called()
        self.utils["local_complete"].assert_not_called()
        self.announce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
