"""Logic tests for skills/site_builder.py ("build a website for <business>").

Fakes only: the fact lookup is a stand-in dossier module in sys.modules, the
cloud call is a patched core.llm_client.complete, the local model is the fake
skill_utils["local_complete"], the browser is the fake skill_utils["open_url"]
and the data dir is a temp dir via JARVIS_DATA_DIR. No network, no LLM, no
browser. Fixtures use generic business names only.
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

    def run_action(self, arg, *, cloud=True, cloud_reply=_PAGE,
                   local_reply=None, dossier=_SENTINEL):
        """Run build_website with the gate, the cloud call and the lookup
        faked. Returns (reply, complete_mock, stdout)."""
        self.utils["local_complete"].return_value = local_reply
        if dossier is _SENTINEL:
            dossier = _dossier()
        out = io.StringIO()
        with mock.patch("core.cloud_gate.chat_cloud_allowed",
                        return_value=cloud), \
                mock.patch("core.llm_client.complete",
                           return_value=cloud_reply) as complete, \
                _module("skill_dossier", dossier), \
                contextlib.redirect_stdout(out):
            reply = self.actions["build_website"](arg)
        return reply, complete, out.getvalue()

    def site_file(self, slug):
        return os.path.join(self.data, "sites", slug, "index.html")


class RegisterTests(_SiteBuilderCase):
    def test_registers_build_website_spoken_verbatim(self):
        self.assertIs(self.actions["build_website"], self.mod.build_website)
        self.assertIn("build_website", self.mod.SPEAK_VERBATIM_ACTIONS)

    def test_marks_itself_long_running_on_the_monolith(self):
        fake_bc = types.ModuleType("bobert_companion")
        fake_bc.LONG_RUNNING_ACTIONS = set()
        with _module("bobert_companion", fake_bc):
            self.mod.register({})
        self.assertIn("build_website", fake_bc.LONG_RUNNING_ACTIONS)


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
    def test_page_is_written_in_the_data_dir_and_opened(self):
        reply, _complete, _out = self.run_action("Blue Door Bakery")
        path = self.site_file("blue-door-bakery")
        self.assertTrue(os.path.isfile(path), reply)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), _PAGE + "\n")
        self.utils["open_url"].assert_called_once_with(Path(path).as_uri())
        self.assertIn("opened it in your browser", reply)
        self.assertIn("sites/blue-door-bakery", reply)
        # Nothing but the one page under the data dir.
        self.assertEqual(os.listdir(os.path.join(self.data, "sites")),
                         ["blue-door-bakery"])

    def test_save_refuses_a_folder_outside_the_sites_dir(self):
        for slug in ("..", "../escape", ""):
            with self.assertRaises(ValueError):
                self.mod._save_site(slug, _PAGE)
        self.assertFalse(os.path.exists(os.path.join(self.data, "escape")))

    def test_open_failure_still_reports_the_saved_page(self):
        self.utils["open_url"].side_effect = RuntimeError("no browser")
        reply, _c, _o = self.run_action("Blue Door Bakery")
        self.assertTrue(os.path.isfile(self.site_file("blue-door-bakery")))
        self.assertIn("couldn't open the browser", reply)


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

    def test_non_html_replies_are_rejected(self):
        for raw in (None, "", "Sorry, I can't help with that.",
                    "<html><head></head></html>", "</html> <html><body>"):
            self.assertIsNone(self.mod._extract_html(raw), raw)

    def test_prompt_asks_for_every_section_and_forbids_invention(self):
        _r, complete, _o = self.run_action("Blue Door Bakery | Springfield")
        kw = complete.call_args.kwargs
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
        reply, _c, _o = self.run_action("Blue Door Bakery",
                                        cloud_reply="no page here",
                                        local_reply="still no page")
        self.assertFalse(os.path.exists(os.path.join(self.data, "sites")))
        self.utils["open_url"].assert_not_called()
        self.assertIn("couldn't build the website", reply)


class ModelRoutingTests(_SiteBuilderCase):
    def test_cloud_path_uses_opus_deep(self):
        reply, complete, _o = self.run_action("Blue Door Bakery")
        complete.assert_called_once()
        kw = complete.call_args.kwargs
        self.assertEqual(kw["model"], "claude-opus-5-5")
        self.assertEqual(kw["purpose"], "deep")
        self.utils["local_complete"].assert_not_called()
        self.assertNotIn("local model", reply)

    def test_cloud_disabled_uses_the_local_long_reply_path(self):
        reply, complete, _o = self.run_action(
            "Blue Door Bakery", cloud=False, local_reply=_PAGE)
        complete.assert_not_called()
        call = self.utils["local_complete"].call_args
        self.assertEqual(call.kwargs["max_tokens"], self.mod.LOCAL_MAX_TOKENS)
        self.assertEqual(call.kwargs["timeout_s"], self.mod.LOCAL_TIMEOUT_S)
        self.assertGreaterEqual(self.mod.LOCAL_MAX_TOKENS, 4096)
        self.assertTrue(os.path.isfile(self.site_file("blue-door-bakery")))
        self.assertIn("with the local model", reply)

    def test_no_key_never_calls_the_cloud(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            _r, complete, _o = self.run_action("Blue Door Bakery",
                                               local_reply=_PAGE)
        complete.assert_not_called()

    def test_cloud_disabled_and_no_local_model_declines_honestly(self):
        reply, complete, _o = self.run_action(
            "Blue Door Bakery", cloud=False, local_reply=None)
        complete.assert_not_called()
        self.assertFalse(os.path.exists(os.path.join(self.data, "sites")))
        self.utils["open_url"].assert_not_called()
        self.assertIn("Claude isn't available", reply)
        self.assertIn("local model couldn't write the page", reply)

    def test_cloud_failure_falls_back_to_local(self):
        reply, complete, _o = self.run_action(
            "Blue Door Bakery", cloud_reply=None, local_reply=_PAGE)
        complete.assert_called_once()
        self.utils["local_complete"].assert_called_once()
        self.assertTrue(os.path.isfile(self.site_file("blue-door-bakery")))
        self.assertIn("with the local model", reply)


class FactsTests(_SiteBuilderCase):
    def test_facts_reach_the_prompt_but_never_the_log(self):
        dossier = _dossier()
        reply, complete, out = self.run_action(
            "Blue Door Bakery | Springfield", dossier=dossier)
        dossier._gather_web.assert_called_once_with(
            "Blue Door Bakery Springfield")
        self.assertIn(_FACTS, complete.call_args.kwargs["messages"][0]["content"])
        self.assertNotIn(_FACTS, out)
        self.assertNotIn("<section", out)          # nor the page itself
        self.assertNotIn("couldn't find anything", reply)

    def test_no_facts_builds_from_the_name_alone_and_says_so(self):
        reply, complete, _o = self.run_action("Blue Door Bakery",
                                              dossier=_dossier(""))
        self.assertIn("nothing could be fetched",
                      complete.call_args.kwargs["messages"][0]["content"])
        self.assertTrue(os.path.isfile(self.site_file("blue-door-bakery")))
        self.assertIn("couldn't find anything about them online", reply)
        self.assertIn("the name alone", reply)

    def test_no_lookup_helper_loaded_counts_as_no_facts(self):
        reply, _c, _o = self.run_action("Blue Door Bakery | | sourdough",
                                        dossier=None)
        self.assertIn("couldn't find anything about them online", reply)
        self.assertIn("your notes", reply)

    def test_failing_lookup_counts_as_no_facts(self):
        dossier = _dossier()
        dossier._gather_web.side_effect = OSError("offline")
        reply, _c, _o = self.run_action("Blue Door Bakery", dossier=dossier)
        self.assertIn("couldn't find anything about them online", reply)

    def test_empty_arg_asks_which_business(self):
        reply, complete, _o = self.run_action("  ")
        self.assertIn("Which business", reply)
        complete.assert_not_called()
        self.utils["local_complete"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
