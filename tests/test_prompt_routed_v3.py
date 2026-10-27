"""Prompt v3 routed sections (v3 overhaul W07): PROMPT_ROUTED_V3.

One bool setting fixes wrong-but-registered choices the routed prompt
sections taught (report 06, E11 + E13, plus the index gap):

  * OFF (the default until the brain harness gate passes): the previous
    text, byte for byte - PC_CONTROL_PROMPT is only ever changed by exact
    swaps and the old text survives as PC_CONTROL_PROMPT_V2; the index and
    the session-summaries header are the old ones;
  * ON: click targets are the distinctive words, the verb decides in SMART
    HOME, open_url waits for a URL sir said, MULTI-MONITOR leads with PLAY
    versus SEARCH, "pause the video" is a pause, "read the results" is
    see_screen, the ACTION INDEX names search_my_files and
    whats_using_bandwidth, and the session summaries are framed as
    background.

The pure-text tests need no monolith; the index / summaries tests do.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import prompt_router as pr  # noqa: E402
from core import prompts  # noqa: E402
from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)


def _v3() -> str:
    return prompts._apply_swaps(prompts.PC_CONTROL_PROMPT_V2,
                                prompts._ROUTED_V3_SWAPS)


def _section(text: str, header: str) -> str:
    core, sections = pr.split_pc_control(text)
    for h, body in sections:
        if h.strip().upper() == header:
            return body
    raise AssertionError("no section " + header)


class SwapTableTests(unittest.TestCase):
    def test_every_swap_finds_its_old_text_exactly_once(self):
        for old, _new in prompts._ROUTED_V3_SWAPS:
            self.assertEqual(prompts.PC_CONTROL_PROMPT_V2.count(old), 1,
                             old[:60])

    def test_off_path_is_the_untouched_text(self):
        if not prompts._routed_v3_on():
            self.assertIs(prompts.PC_CONTROL_PROMPT,
                          prompts.PC_CONTROL_PROMPT_V2)

    def test_swaps_change_only_what_they_name(self):
        v3 = _v3()
        self.assertNotEqual(v3, prompts.PC_CONTROL_PROMPT_V2)
        back = v3
        for old, new in prompts._ROUTED_V3_SWAPS:
            self.assertIn(new, back)
            back = back.replace(new, old, 1)
        self.assertEqual(back, prompts.PC_CONTROL_PROMPT_V2)

    def test_ui_automation_targets_are_short(self):
        v3 = _v3()
        ui = _section(v3, "UI AUTOMATION")
        self.assertIn("[ACTION: click, MrBeast video]", ui)
        self.assertNotIn("[ACTION: click, the MrBeast video]", ui)
        self.assertIn("Target = the distinctive words only", ui)
        self.assertIn("'for me'", ui)
        self.assertIn("'on the X monitor'", ui)
        # The old rule lines survive.
        self.assertIn("never youtube_play, web_search or open_url for it", ui)

    def test_smart_home_the_verb_decides(self):
        sh = _section(_v3(), "SMART HOME")
        self.assertIn("'turn on all the smart home\n    devices'", sh)
        self.assertIn("The verb decides", sh)
        self.assertIn("= smart_home_control", sh)
        self.assertIn("only 'what / list ... do I have' is a list request",
                      sh)

    def test_open_url_waits_for_a_url_sir_said(self):
        v3 = _v3()
        self.assertIn("Only a URL sir said or a site you are", v3)
        self.assertIn("otherwise web_search.", v3)
        # In the always-on core preamble, not a routed section.
        core, _sections = pr.split_pc_control(v3)
        always = core + _section(v3, "MULTI-MONITOR APP LAUNCHING")
        self.assertIn("Only a URL sir said", always)

    def test_multi_monitor_leads_with_play_versus_search(self):
        body = _section(_v3(), "MULTI-MONITOR APP LAUNCHING")
        first = body.split("\n", 2)[1].strip()
        self.assertEqual(
            first, "PLAY on a monitor = youtube_play, then "
                   "move_window_to_monitor.")
        self.assertLess(body.index("PLAY on a monitor"),
                        body.index("open_on_monitor, <monitor> | <url"))
        self.assertIn("(plays nothing)", body)
        # The old worked examples are still there.
        self.assertIn("[ACTION: move_window_to_monitor, chrome | left]", body)

    def test_pause_and_read_teaching_lines(self):
        v3 = _v3()
        self.assertIn("'pause the video' / 'pause it' → [ACTION: pause_music]",
                      v3)
        self.assertIn("media_prev\n    and media_next are track skips", v3)
        self.assertIn("'read the results' / 'is the video playing'", v3)
        self.assertIn("never screenshot", v3)

    def test_no_file_move_line_in_this_item(self):
        # File moves belong to W15 (move_file); W07 teaches nothing about them.
        v3 = _v3()
        self.assertNotIn("No action moves, copies or deletes files", v3)
        self.assertNotIn("move_file", v3)

    def test_the_router_still_splits_the_same_sections(self):
        _c0, s0 = pr.split_pc_control(prompts.PC_CONTROL_PROMPT_V2)
        _c1, s1 = pr.split_pc_control(_v3())
        self.assertEqual([h for h, _ in s0], [h for h, _ in s1])


class SlimIndexKeepTests(unittest.TestCase):
    """slim_stable_pc_block(keep=...) - pure, no monolith."""

    PC = (
        "core preamble\n\n"
        "ALPHA THINGS:\n"
        "  [ACTION: first_name] runs the handler.\n\n"
        "BETA THINGS:\n"
        "  second_alias is the same handler.\n")
    GROUP = {"first_name": ("fn", 1), "second_alias": ("fn", 1)}
    KNOWN = {"first_name", "second_alias"}

    def _rows(self, **kw):
        out = pr.slim_stable_pc_block(self.PC, "", known=self.KNOWN,
                                      group=self.GROUP, **kw)
        return out[out.index("ACTION INDEX"):]

    def test_an_alias_is_dropped_by_default(self):
        self.assertNotIn("second_alias", self._rows())
        self.assertNotIn("second_alias", self._rows(keep=None))

    def test_a_kept_alias_stays_listed(self):
        rows = self._rows(keep=("second_alias",))
        self.assertIn("BETA THINGS: second_alias", rows)
        self.assertIn("ALPHA THINGS: first_name", rows)

    def test_keep_of_an_unknown_name_changes_nothing(self):
        self.assertEqual(self._rows(keep=("nothing_here",)), self._rows())

    def test_real_prompt_keep_lists_the_two_names(self):
        known = {"search_my_files", "rag_search", "whats_using_bandwidth",
                 "network_usage", "bandwidth_hogs"}
        group = {"search_my_files": ("fn", "rag"), "rag_search": ("fn", "rag"),
                 "whats_using_bandwidth": ("fn", "net"),
                 "network_usage": ("fn", "net"),
                 "bandwidth_hogs": ("fn", "hogs")}
        v3 = _v3()
        off = pr.slim_stable_pc_block(v3, "", known=known, group=group)
        on = pr.slim_stable_pc_block(
            v3, "", known=known, group=group,
            keep=("search_my_files", "whats_using_bandwidth"))
        for name in ("search_my_files", "whats_using_bandwidth"):
            self.assertNotIn(name, off[off.index("ACTION INDEX"):])
            self.assertIn(name, on[on.index("ACTION INDEX"):])


@requires_monolith
class MonolithTests(MonolithGlobalsTestCase):
    def test_default_is_off_in_the_shipped_config(self):
        import core.config as cfg
        self.assertIn("PROMPT_ROUTED_V3 = False",
                      open(cfg.__file__, encoding="utf-8").read())

    def test_setting_is_a_turn_flag_key_on_its_own_line(self):
        self.assertIn("PROMPT_ROUTED_V3", self.bc._TURN_FLAG_KEYS)
        src = open(self.bc.__file__, encoding="utf-8").read()
        self.assertIn('\n    "PROMPT_ROUTED_V3",\n', src)

    def test_stable_index_names_the_two_gap_actions_only_when_on(self):
        bc = self.bc
        # Two registered aliases that share a handler with a name documented
        # earlier (the live registry has them via the personal_rag and
        # network_deco skills, which the monolith harness does not load).
        def _rag(*_a, **_k):
            return "rag"

        def _net(*_a, **_k):
            return "net"

        fake = {"rag_search": _rag, "search_my_files": _rag,
                "network_usage": _net, "whats_using_bandwidth": _net}
        for flag in (False, True):
            with mock.patch.dict(bc.ACTIONS, fake), \
                    mock.patch.object(bc, "PROMPT_ROUTED_V3", flag,
                                      create=True):
                bc._SLIM_PC_CACHE.clear()
                try:
                    out = bc._slim_stable_pc("", None)
                finally:
                    bc._SLIM_PC_CACHE.clear()
            idx = out[out.index("ACTION INDEX"):]
            for name in ("search_my_files", "whats_using_bandwidth"):
                self.assertEqual(name in idx, flag, (name, flag))

    def test_summaries_header_follows_the_setting(self):
        bc = self.bc
        mem = bc._empty_memory()
        mem["sessions"] = [{"date": "2026-10-20", "location": "desk",
                            "summary": "talked about the print"}]
        for flag in (False, True):
            with mock.patch.object(bc, "PROMPT_ROUTED_V3", flag,
                                   create=True):
                prompt = bc.build_system_prompt(mem)
            if flag:
                self.assertIn("Recent conversation summaries (background "
                              "only: never bring one up or act on it unless "
                              "sir does):\n- 2026-10-20", prompt)
                self.assertNotIn("Recent conversation summaries:\n", prompt)
            else:
                self.assertIn("\n\nRecent conversation summaries:\n- 2026-10-20",
                              prompt)
                self.assertNotIn("background only", prompt)

    def test_standing_rules_substring_untouched(self):
        # test_monolith_sec1 pins "STANDING RULES"; W07 never edits it.
        src = open(self.bc.__file__, encoding="utf-8").read()
        self.assertIn("STANDING RULES", src)


if __name__ == "__main__":
    unittest.main()
