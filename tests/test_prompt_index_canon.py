"""Prompt diet and canonical index (v3 overhaul W12): PROMPT_INDEX_CANON.

One bool setting shrinks the stable prompt without taking any action away:

  * OFF (the default before 3.0.0's fair prompt gate): the previous
    prompt, byte for byte - BASE_SYSTEM_PROMPT_V2 is the untouched persona
    and slim_stable_pc_block() with no ``hide`` / ``collapse`` is the old
    index;
  * ON: (1) the "Signature phrases" list is cut - the 20-line pool and the
    phrasebook carry the same lines - but the "Minimal acknowledgements ...
    These are deliberate" span stays; (2) the ACTION INDEX leaves out the
    registered near-duplicates in core.action_aliases.HIDDEN_DUPLICATES while
    their canonical name is visible; (3) rare / setup rows fold into a few
    routed lines (prompt_router.COLLAPSED_ROWS) that name the capability.

Nothing is unregistered: every hidden name is still a real action that keeps
its risk tier, and every collapsed capability still routes from its own
header words, so the full section (exact action names, arguments) rides the
turn that names it.

The pure tests need no monolith; the monolith tests skip on the light tier.

    python -m unittest tests.test_prompt_index_canon
"""
from __future__ import annotations

import ast
import glob
import importlib.util
import os
import re
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import action_aliases as aa  # noqa: E402
from core import action_risk  # noqa: E402
from core import prompt_router as pr  # noqa: E402
from core import prompts  # noqa: E402
from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)

_REGISTRY: "dict | None" = None


def _registry() -> dict:
    """name -> (file, handler symbol) for every scanned registration (the
    same scan tests/test_action_aliases.py uses)."""
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    spec = importlib.util.spec_from_file_location(
        "registration_scan_w12",
        os.path.join(_PROJECT, "tools", "registration_scan.py"))
    rs = importlib.util.module_from_spec(spec)
    sys.modules["registration_scan_w12"] = rs    # before exec (dataclasses)
    spec.loader.exec_module(rs)
    srcs = ([os.path.join(_PROJECT, "bobert_companion.py")]
            + sorted(glob.glob(os.path.join(_PROJECT, "skills", "*.py")))
            + sorted(glob.glob(os.path.join(_PROJECT, "skills", "*", "*.py")))
            + sorted(glob.glob(os.path.join(_PROJECT, "core", "*.py"))))
    reg: dict = {}
    for path in srcs:
        rel = os.path.relpath(path, _PROJECT).replace("\\", "/")
        targets = (("ACTIONS",) if rel == "bobert_companion.py"
                   else rs.DEFAULT_TARGETS)
        try:
            found = rs.scan_file(path, targets=targets, filename=rel)
        except (SyntaxError, OSError):
            continue
        for name, r in found.items():
            reg.setdefault(name, (rel, getattr(r, "symbol", "")))
    _REGISTRY = reg
    return reg


def _confirm_tier(name: str) -> bool:
    """bobert_companion._needs_confirmation's word rule on the name alone."""
    from core.config import CONFIRM_KEYWORDS
    hay = name.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(k.lower())}", hay)
               for k in CONFIRM_KEYWORDS)


def _tier(name: str) -> tuple:
    return (name in action_risk.SELF_TERMINATING_ACTIONS,
            action_risk.guess_protected(name),
            _confirm_tier(name),
            tuple(action_risk.confirm_reasons(name)))


def _word_in(name: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(name)
                     + r"(?![A-Za-z0-9_])", text) is not None


# ─────────────────────────────────────────────────────────────────────────
#  HIDDEN_DUPLICATES: pure data, checked against the registration scan
# ─────────────────────────────────────────────────────────────────────────
class HiddenDuplicatesTableTests(unittest.TestCase):
    def test_registry_scan_is_not_blind(self):
        reg = _registry()
        self.assertGreater(len(reg), 500)
        self.assertIn("next_song", reg)

    def test_the_table_is_a_real_diet(self):
        # PLAN W12 aimed at about 60 names; the review pass removed every
        # pair whose canonical name could not do the hidden name's job, so
        # the verified table is smaller, but still a real diet.
        self.assertGreaterEqual(len(aa.HIDDEN_DUPLICATES), 28)

    def test_rows_are_lower_snake_case(self):
        for hid, canon in aa.HIDDEN_DUPLICATES.items():
            for n in (hid, canon):
                self.assertRegex(n, r"^[a-z][a-z0-9_]*$")
            self.assertNotEqual(hid, canon)

    def test_every_hidden_name_is_still_a_registered_action(self):
        reg = _registry()
        for hid in aa.HIDDEN_DUPLICATES:
            with self.subTest(hidden=hid):
                self.assertIn(hid, reg, "a hidden name must stay registered "
                              "(it is hidden from the prompt, never removed)")

    def test_every_canonical_name_is_registered_and_not_hidden(self):
        reg = _registry()
        for hid, canon in aa.HIDDEN_DUPLICATES.items():
            with self.subTest(hidden=hid, canonical=canon):
                self.assertIn(canon, reg)
                self.assertNotIn(canon, aa.HIDDEN_DUPLICATES)

    def test_a_hidden_name_is_never_an_alias_source(self):
        # ALIASES rewrite invented names; a hidden name is real and never
        # rewritten.
        for hid in aa.HIDDEN_DUPLICATES:
            with self.subTest(hidden=hid):
                self.assertNotIn(hid, aa.ALIASES)
                self.assertIsNone(aa.alias_for(hid, "x", set(_registry())))

    def test_nothing_never_aliased_or_self_terminating_is_hidden(self):
        for hid, canon in aa.HIDDEN_DUPLICATES.items():
            with self.subTest(hidden=hid, canonical=canon):
                self.assertFalse(aa.is_never_alias(hid))
                for n in (hid, canon):
                    self.assertNotIn(n, action_risk.SELF_TERMINATING_ACTIONS)

    def test_a_hidden_name_keeps_the_risk_tier_of_its_canonical(self):
        # The model will say the canonical name instead of the hidden one, so
        # a hidden name must not carry a stricter gate than what replaces it
        # (that would route around a confirmation), nor a looser one.
        for hid, canon in aa.HIDDEN_DUPLICATES.items():
            with self.subTest(hidden=hid, canonical=canon):
                self.assertEqual(_tier(hid), _tier(canon))

    # Pairs whose handlers are different functions but where the canonical
    # name demonstrably covers the hidden name's output. Every other pair must
    # share one handler (same registration file and symbol).
    REVIEWED_PAIRS = {
        "media_next": "raw media key; next_song is the media-session skip",
        "media_prev": "raw media key; previous_song is the media-session back",
        "media_playpause": "raw media key; pause_music toggles the session",
        "music_status": "now_playing reads the same session",
        "prime_video": "wrapper: play_streaming takes 'prime|title'",
        "netflix": "wrapper: play_streaming takes 'netflix|title'",
        "disney_plus": "wrapper: play_streaming takes 'disney|title'",
        "hulu": "wrapper: play_streaming takes 'hulu|title'",
        "max": "wrapper: play_streaming takes 'max|title'",
        "spotify": "wrapper: play_streaming takes 'spotify|title'",
        "hue_list_devices": "list_smart_home_devices lists every brand",
        "govee_list_devices": "list_smart_home_devices lists every brand",
        "lifx_list_devices": "list_smart_home_devices lists every brand",
        "kasa_list_devices": "list_smart_home_devices lists every brand",
        "ecobee_list_devices": "list_smart_home_devices lists every brand",
        "nest_list_devices": "list_smart_home_devices lists every brand",
        "ring_list_devices": "list_smart_home_devices lists every brand",
        "magichome_list_devices": "list_smart_home_devices lists every brand",
        "tuya_list": "list_smart_home_devices lists every brand",
        "check_system": "both speak the machine's CPU / RAM / disk posture",
        "system_check": "self_diagnostic registers run_diagnostic under both",
        "diagnostic_daemon_status": "diagnostic_status falls back to the "
                                    "daemon counters (owner's 2026-10-02 call)",
        "last_gate_result": "same skill, same last-gate record",
        "which_mic_is_active": "current_mic names the live mic and the "
                               "preferred-mic override",
        "focus_mode_on": "focus_mode_on is the chained base of focus_mode",
        "focus_mode_off": "focus_mode_off is the chained base of "
                          "end_focus_mode",
        "make_picture": "one-line wrapper: return generate_image(prompt)",
    }

    def test_each_pair_shares_a_handler_or_is_reviewed(self):
        reg = _registry()
        for hid, canon in aa.HIDDEN_DUPLICATES.items():
            with self.subTest(hidden=hid, canonical=canon):
                if reg[hid] == reg[canon]:
                    continue
                self.assertIn(hid, self.REVIEWED_PAIRS,
                              "a hidden name whose handler differs from its "
                              "canonical's must be reviewed: the canonical "
                              "must do the hidden name's whole job")
                self.assertTrue(self.REVIEWED_PAIRS[hid].strip())

    def test_the_reviewed_list_has_no_stale_or_needless_rows(self):
        reg = _registry()
        for hid in self.REVIEWED_PAIRS:
            with self.subTest(hidden=hid):
                self.assertIn(hid, aa.HIDDEN_DUPLICATES)
                self.assertNotEqual(reg[hid],
                                    reg[aa.HIDDEN_DUPLICATES[hid]],
                                    "same handler: needs no review row")

    def test_pairs_the_review_found_different_stay_out(self):
        # Each of these has a handler that does a different job from the
        # canonical name the table once gave it.
        for n in ("schedule_status", "is_printer_online", "morning_handoff",
                  "morning_arrival", "toggle_hud", "toggle_holographic_overlay",
                  "bambu_overlay_toggle", "stark_status_ring",
                  "arc_reactor_status_status", "stark_status_ring_status",
                  "hand_mouse_off", "hand_mouse_status",
                  "last_diagnostic_run", "rag_search_quiet"):
            with self.subTest(name=n):
                self.assertNotIn(n, aa.HIDDEN_DUPLICATES)

    def test_hidden_duplicate_of_is_a_plain_lookup(self):
        self.assertEqual(aa.hidden_duplicate_of("media_next"), "next_song")
        self.assertEqual(aa.hidden_duplicate_of(" Netflix "), "play_streaming")
        self.assertIsNone(aa.hidden_duplicate_of("next_song"))
        self.assertIsNone(aa.hidden_duplicate_of(""))
        self.assertIsNone(aa.hidden_duplicate_of(None))

    def test_the_clusters_named_in_the_plan_are_covered(self):
        # 8 pause / next names, 11 play / video, 12 smart-home lists,
        # list_schedule beside list_schedules.
        h = aa.HIDDEN_DUPLICATES
        for n in ("media_next", "media_prev", "media_playpause"):
            self.assertIn(n, h)
        for n in ("netflix", "hulu", "disney_plus", "prime_video", "max",
                  "spotify"):
            self.assertEqual(h[n], "play_streaming")
        brands = [n for n in h if n.endswith("_list_devices")]
        self.assertGreaterEqual(len(brands), 7)
        self.assertEqual(h["list_schedule"], "list_schedules")


# ─────────────────────────────────────────────────────────────────────────
#  prompt_router.slim_stable_pc_block(hide=, collapse=): pure, synthetic
# ─────────────────────────────────────────────────────────────────────────
class IndexHideCollapseTests(unittest.TestCase):
    PC = (
        "core preamble\n\n"
        "PLAYBACK:\n"
        "  [ACTION: next_song] skips; [ACTION: media_next] is the raw key.\n\n"
        "STREAMING:\n"
        "  [ACTION: play_streaming, service|title]; [ACTION: stream_a], "
        "[ACTION: stream_b]\n\n"
        "RARE SETUP:\n"
        "  [ACTION: setup_widget] and [ACTION: widget_status]\n\n"
        "LAST THINGS:\n"
        "  [ACTION: last_one]\n")
    KNOWN = {"next_song", "media_next", "play_streaming", "stream_a",
             "stream_b", "setup_widget", "widget_status", "last_one"}
    HIDE = {"media_next": "next_song", "stream_a": "play_streaming",
            "stream_b": "play_streaming", "last_one": "widget_status"}

    def _idx(self, **kw):
        out = pr.slim_stable_pc_block(self.PC, "", known=self.KNOWN, **kw)
        return out[out.index("ACTION INDEX"):]

    def test_no_new_arguments_is_the_old_index(self):
        base = self._idx()
        self.assertEqual(base, self._idx(hide=None, collapse=None))
        self.assertEqual(base, self._idx(hide={}, collapse=()))
        for n in ("media_next", "stream_a", "stream_b", "setup_widget"):
            self.assertIn(n, base)
        self.assertNotIn("ROUTED ON REQUEST", base)

    def test_a_hidden_duplicate_leaves_the_rows_when_its_canonical_shows(self):
        idx = self._idx(hide=self.HIDE)
        self.assertIn("PLAYBACK: next_song", idx)
        self.assertNotIn("media_next", idx)
        self.assertIn("STREAMING: play_streaming", idx)
        self.assertNotIn("stream_a", idx)
        self.assertNotIn("stream_b", idx)

    def test_a_name_whose_canonical_is_not_visible_stays_listed(self):
        # widget_status is only documented in a COLLAPSED row, so last_one
        # (hidden -> widget_status) must stay listed.
        idx = self._idx(hide=self.HIDE, collapse=(("RARE", ("RARE SETUP",)),))
        self.assertIn("last_one", idx)
        self.assertNotIn("widget_status", idx)

    def test_hide_only_touches_registered_names(self):
        out = pr.slim_stable_pc_block(
            self.PC, "", known={"next_song"},
            hide={"media_next": "next_song"})
        self.assertIn("media_next", out[out.index("ACTION INDEX"):])
        out = pr.slim_stable_pc_block(self.PC, "", hide=self.HIDE)
        self.assertIn("media_next", out[out.index("ACTION INDEX"):])

    def test_a_canonical_that_is_itself_hidden_never_counts(self):
        idx = self._idx(hide={"media_next": "stream_a", "stream_a": "stream_b",
                              "stream_b": "media_next"})
        for n in ("media_next", "stream_a", "stream_b"):
            self.assertIn(n, idx)

    def test_a_self_terminating_name_is_never_hidden_or_listed(self):
        out = pr.slim_stable_pc_block(
            self.PC, "", known=self.KNOWN | {"shutdown_jarvis"},
            hide={"shutdown_jarvis": "next_song"})
        self.assertNotIn("shutdown_jarvis", out[out.index("ACTION INDEX"):])

    def test_collapse_names_the_capability_not_its_actions(self):
        idx = self._idx(collapse=(("SETUP LINE", ("RARE SETUP",)),))
        self.assertNotIn("setup_widget", idx)
        self.assertNotIn("widget_status", idx)
        self.assertNotIn("\nRARE SETUP", idx)
        self.assertIn(pr._COLLAPSED_NOTE, idx)
        self.assertIn("SETUP LINE: RARE SETUP", idx)
        # the other rows are untouched
        self.assertIn("PLAYBACK: next_song", idx)
        self.assertIn("LAST THINGS: last_one", idx)

    def test_a_collapse_naming_no_present_header_changes_nothing(self):
        self.assertEqual(self._idx(),
                         self._idx(collapse=(("X", ("NOT A HEADER",)),)))

    def test_collapse_is_case_insensitive_and_groups_by_label(self):
        idx = self._idx(collapse=(("ONE", ("rare setup",)),
                                  ("TWO", ("Last Things",))))
        self.assertIn("ONE: RARE SETUP", idx)
        self.assertIn("TWO: LAST THINGS", idx)

    def test_pure_and_deterministic(self):
        a = self._idx(hide=self.HIDE, collapse=pr.COLLAPSED_ROWS)
        b = self._idx(hide=self.HIDE, collapse=pr.COLLAPSED_ROWS)
        self.assertEqual(a, b)


# ─────────────────────────────────────────────────────────────────────────
#  COLLAPSED_ROWS: every header must still route from its own words
# ─────────────────────────────────────────────────────────────────────────
def _skill_parts() -> list:
    """[(name, text)] for every skill PROMPT_EXAMPLES in the tree, read by
    AST (no import), split the way the monolith joins them."""
    out = []
    paths = (sorted(glob.glob(os.path.join(_PROJECT, "skills", "*.py")))
             + sorted(glob.glob(os.path.join(_PROJECT, "skills", "*",
                                             "__init__.py"))))
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name)
                            and t.id == "PROMPT_EXAMPLES"
                            for t in node.targets)):
                try:
                    val = ast.literal_eval(node.value)
                except Exception:
                    continue
                if isinstance(val, str) and val.strip():
                    txt = val.strip()
                    out.append((pr._skill_part_name(txt), txt))
    return out


class CollapsedRowsRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core, cls.sections = pr.split_pc_control(prompts.PC_CONTROL_PROMPT)
        cls.pc_headers = [h.strip() for h, _ in cls.sections]
        cls.parts = _skill_parts()
        cls.headers = [h for _label, hs in pr.COLLAPSED_ROWS for h in hs]
        cls.known = set(_registry())

    def test_labels_and_headers_are_unique(self):
        labels = [label for label, _ in pr.COLLAPSED_ROWS]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(self.headers), len(set(self.headers)))
        self.assertLessEqual(len(labels), 6, "a few routed lines, not a wall")

    def test_enough_rows_fold_to_matter(self):
        self.assertGreaterEqual(len(self.headers), 30)

    def test_every_header_exists_in_the_prompt(self):
        names = set(self.pc_headers) | {n for n, _ in self.parts}
        for h in self.headers:
            with self.subTest(header=h):
                self.assertIn(h, names, "stale header: no such section")

    def test_every_pc_section_routes_from_its_own_header_words(self):
        for h in self.headers:
            if h not in self.pc_headers:
                continue
            with self.subTest(header=h):
                q = h.lower().replace("—", " ")
                inc, _dropped = pr.select_sections(q, self.sections)
                self.assertIn(h, inc)

    def test_every_skill_part_routes_from_its_own_header_words(self):
        for h in self.headers:
            for name, text in self.parts:
                if name != h:
                    continue
                with self.subTest(header=h):
                    q = h.lower().replace("—", " ")
                    self.assertIn(text, pr.select_skill_examples(q,
                                                                 self.parts))

    # ── the rule itself: the section's own phrases must route ────────────
    # A folded section's action names leave the index, so every phrase the
    # prompt quotes for it (a trigger, an example, an in-row "say ..." list)
    # must bring back a section that documents one of its actions. Quotes
    # that are NOT something the owner says (the model's own spoken line, an
    # argument value) are exempt, each with its reason.
    _NOT_OWNER_PHRASES = {
        "GLOBE": {"lat, lon": "argument form of globe_pin"},
        "STATUS READ-BACKS": {"find out": "verb in an instruction"},
        "CLAUDE CREDITS": {"one moment": "the model's spoken line"},
        "CLAP TRIGGER": {
            "You rang, sir?": "the model's spoken line",
            "morning setup": "argument of clap_trigger_routine",
            "morning briefing": "argument of clap_trigger_routine"},
        "SUIT-UP CINEMATIC": {"Welcome back, sir": "the model's spoken line"},
    }
    _PHRASE = re.compile(r"(?<![\w])'([^'\n]{3,90}?)'(?![\w])")

    @classmethod
    def _routed_actions(cls, phrase):
        """Action names documented by every PC section and skill example that
        ``phrase`` (as the owner's whole message) routes to."""
        inc, _ = pr.select_sections(phrase, cls.sections)
        bodies = {h.strip(): b for h, b in cls.sections}
        texts = [bodies.get(h.strip(), "") for h in inc]
        texts += pr.select_skill_examples(phrase, cls.parts)
        return set(pr.documented_actions(" ".join(texts), cls.known))

    @classmethod
    def _folded_texts(cls):
        """[(header, text)] of every folded PC section and skill part."""
        out = [(h.strip(), b) for h, b in cls.sections
               if h.strip() in cls.headers]
        out += [(n, t) for n, t in cls.parts if n in cls.headers]
        return out

    def test_every_quoted_phrase_of_a_folded_section_routes_back(self):
        checked = 0
        for h, text in self._folded_texts():
            own = set(pr.documented_actions(text, self.known))
            exempt = self._NOT_OWNER_PHRASES.get(h, {})
            flat = re.sub(r"\s+", " ", text)
            for m in self._PHRASE.finditer(flat):
                p = m.group(1).strip()
                if ("[ACTION" in p or "→" in p or "<" in p
                        or len(p.split()) < 2 or p in exempt):
                    continue
                checked += 1
                with self.subTest(header=h, phrase=p):
                    self.assertTrue(
                        self._routed_actions(p) & own,
                        "folds the index row but the phrase routes to no "
                        "section that documents any of its actions: un-fold "
                        "it (COLLAPSED_ROWS) or fix the routing")
        self.assertGreater(checked, 100, "the phrase scan went blind")

    def test_every_example_line_routes_to_its_own_action(self):
        # "'phrase' -> [ACTION: name ...]": the routed prompt must document
        # exactly that name, not just a sibling.
        seen = 0
        for h, text in self._folded_texts():
            for ln in text.split("\n"):
                if "→" not in ln and "->" not in ln:
                    continue
                left, _, right = ln.replace("->", "→").partition("→")
                names = pr.documented_actions(right)
                if not names:
                    continue
                for m in self._PHRASE.finditer(left):
                    p = m.group(1).strip()
                    if "<" in p or len(p.split()) < 2:
                        continue
                    seen += 1
                    with self.subTest(header=h, phrase=p, action=names[0]):
                        self.assertIn(names[0], self._routed_actions(p))
        self.assertGreater(seen, 30, "the example scan went blind")

    def test_the_sections_the_review_found_unreachable_stay_unfolded(self):
        # Their trigger phrases ("what's the GPU temp", "use the cloud",
        # "go back to auto", "switch to the 32B", "play my Friday night
        # vibe", "what's new", "what did I miss") do not route back, and
        # WEB INTERFACE hosts hardware_sensors / check_budget.
        for h in ("LOCAL MODEL SELECTION", "TASTE-AWARE MUSIC",
                  "CHANGELOG / VERSION", "PHONE NOTIFICATIONS",
                  "WEB INTERFACE", "SUIT DIAGNOSTICS", "SELF DIAGNOSTIC",
                  "MEMORY MAINTENANCE", "GUARD MODE", "FACE RECOGNITION"):
            with self.subTest(header=h):
                self.assertNotIn(h, self.headers)

    def test_no_collapsed_section_documents_a_self_terminating_action(self):
        bodies = {h.strip(): b for h, b in self.sections}
        for h in self.headers:
            with self.subTest(header=h):
                self.assertFalse(
                    pr.documents_self_terminating_action(bodies.get(h, "")))

    def test_the_always_on_and_safety_sections_never_collapse(self):
        upper = {h.upper() for h in self.headers}
        for keep in pr._ALWAYS | pr._SLIM_HEAD_SECTIONS:
            self.assertNotIn(keep.upper(), upper)
        self.assertNotIn("SHUTDOWN ALIASES", upper)

    def test_the_real_prompt_folds_and_loses_no_header(self):
        plain = pr.slim_stable_pc_block(prompts.PC_CONTROL_PROMPT)
        folded = pr.slim_stable_pc_block(prompts.PC_CONTROL_PROMPT,
                                         collapse=pr.COLLAPSED_ROWS)
        self.assertLess(len(folded), len(plain) - 800)
        for h in self.headers:
            if h in self.pc_headers:
                with self.subTest(header=h):
                    self.assertIn(h, folded)


# ─────────────────────────────────────────────────────────────────────────
#  The persona: the "Signature phrases" block goes, the minimal acks stay
# ─────────────────────────────────────────────────────────────────────────
class SignatureBlockTests(unittest.TestCase):
    V2 = prompts.BASE_SYSTEM_PROMPT_V2

    def test_the_block_is_in_the_untouched_text(self):
        self.assertIn(prompts._SIGNATURE_BLOCK_START, self.V2)
        self.assertIn(prompts._SIGNATURE_BLOCK_END, self.V2)

    def test_the_cut_removes_the_block_and_nothing_else(self):
        cut = prompts._strip_signature_block(self.V2)
        a = self.V2.index(prompts._SIGNATURE_BLOCK_START)
        b = self.V2.index(prompts._SIGNATURE_BLOCK_END)
        self.assertEqual(cut, self.V2[:a] + prompts._SIGNATURE_GUARD
                         + self.V2[b:])
        self.assertGreaterEqual(len(self.V2) - len(cut), 1800)
        self.assertNotIn("Signature phrases", cut)
        self.assertNotIn("  Acknowledgements: 'Very good, sir.'", cut)
        self.assertNotIn("Pushback / disagreement:", cut)
        self.assertNotIn("Initiative: 'I've taken the liberty of...'", cut)

    def test_the_status_and_concern_guards_survive_the_cut(self):
        cut = prompts._strip_signature_block(self.V2)
        self.assertIn("ONLY in a reply whose [ACTION:] token does that work",
                      cut)
        self.assertIn("never as a stand-in for an answer", cut)
        self.assertIn("never claim a calculation, search or comparison you "
                      "did not run", cut)
        self.assertIn("always leads with 'I'm afraid'", cut)
        # and the untouched text has the very guard the cut keeps
        self.assertIn("ONLY in a reply whose [ACTION:] token does that work",
                      self.V2)

    def test_the_minimal_acknowledgements_span_stays(self):
        cut = prompts._strip_signature_block(self.V2)
        i = cut.index("Minimal acknowledgements")
        j = cut.index("These are deliberate", i)
        seg = cut[i:j]
        self.assertEqual(sorted(re.findall(r"'([^']+)'", seg)),
                         sorted(["Working.", "Working on it.", "One moment.",
                                 "Quite.", "Indeed, sir.", "Naturally, sir."]))
        self.assertIn("deadpan replies", cut[j:j + 200])

    def test_the_pool_and_the_opener_rules_stay(self):
        cut = prompts._strip_signature_block(self.V2)
        for needle in ("MATCH THE OPENER TO THE CONTENT",
                       "Adjacent-fact volunteering",
                       "Few-shot examples of the dry-wit register",
                       "DELIVER WHAT WAS ASKED",
                       "Canonical MCU openings"):
            with self.subTest(needle=needle):
                self.assertIn(needle, cut)

    def test_a_missing_marker_leaves_the_text_alone(self):
        self.assertEqual(prompts._strip_signature_block("nothing here"),
                         "nothing here")
        only_start = "x " + prompts._SIGNATURE_BLOCK_START + " y"
        self.assertEqual(prompts._strip_signature_block(only_start),
                         only_start)
        swapped = (prompts._SIGNATURE_BLOCK_END + " a "
                   + prompts._SIGNATURE_BLOCK_START)
        self.assertEqual(prompts._strip_signature_block(swapped), swapped)

    def test_on_the_live_text_drops_the_block(self):
        # Default ON since 3.0.0 (fair prompt gate, PLAN.md: golden 90.1% -> 94.0%):
        # the live persona no longer carries the block.
        import core.config as cfg
        if not getattr(cfg, "PROMPT_INDEX_CANON", False):
            self.skipTest("setting is off in this checkout")
        self.assertNotIn(prompts._SIGNATURE_BLOCK_START,
                         prompts.BASE_SYSTEM_PROMPT)


class ConfigTests(unittest.TestCase):
    def test_default_is_on_in_the_shipped_config(self):
        # Default ON since 3.0.0 (fair prompt gate, PLAN.md: golden 90.1% -> 94.0%).
        import core.config as cfg
        self.assertIn("PROMPT_INDEX_CANON = True",
                      open(cfg.__file__, encoding="utf-8").read())

    def test_the_setting_is_a_bool(self):
        import core.config as cfg
        self.assertIsInstance(cfg.PROMPT_INDEX_CANON, bool)


# ─────────────────────────────────────────────────────────────────────────
#  The monolith side
# ─────────────────────────────────────────────────────────────────────────
@requires_monolith
class MonolithTests(MonolithGlobalsTestCase):
    def test_setting_is_a_turn_flag_key_on_its_own_line(self):
        self.assertIn("PROMPT_INDEX_CANON", self.bc._TURN_FLAG_KEYS)
        src = open(self.bc.__file__, encoding="utf-8").read()
        self.assertIn('\n    "PROMPT_INDEX_CANON",\n', src)

    def _slim(self, flag):
        bc = self.bc
        with mock.patch.object(bc, "PROMPT_INDEX_CANON", flag, create=True):
            bc._SLIM_PC_CACHE.clear()
            try:
                return bc._slim_stable_pc("", None)
            finally:
                bc._SLIM_PC_CACHE.clear()

    def test_off_kwargs_are_empty_and_the_index_is_the_old_one(self):
        bc = self.bc
        with mock.patch.object(bc, "PROMPT_INDEX_CANON", False, create=True):
            self.assertEqual(bc._canon_index_kwargs(), {})
        known = frozenset(bc.ACTIONS)
        group = {n: ("fn", id(getattr(f, "__func__", f)))
                 for n, f in bc.ACTIONS.items()}
        old = (bc._LOCAL_NEVER_GUESS_GUARD + "\n"
               + pr.slim_stable_pc_block(bc.PC_CONTROL_PROMPT, "",
                                         known=known, group=group,
                                         skill_texts=()))
        off = self._slim(False)
        if getattr(bc, "PROMPT_DIRECTIVE_V3", False):
            off = off.lstrip("\n")
        self.assertEqual(off, old)

    def test_on_kwargs_carry_the_two_tables(self):
        bc = self.bc
        with mock.patch.object(bc, "PROMPT_INDEX_CANON", True, create=True):
            kw = bc._canon_index_kwargs()
        self.assertEqual(kw["hide"], aa.HIDDEN_DUPLICATES)
        self.assertIs(kw["collapse"], pr.COLLAPSED_ROWS)

    def test_on_the_stable_index_is_smaller_and_folds_rows(self):
        off, on = self._slim(False), self._slim(True)
        self.assertLess(len(on), len(off) - 1000)
        self.assertIn(pr._COLLAPSED_NOTE, on)
        self.assertNotIn(pr._COLLAPSED_NOTE, off)
        idx = on[on.index("ACTION INDEX"):]
        for h in ("FACE RECOGNITION", "KINECT DEPTH SENSOR", "TV DETECTION"):
            self.assertIn(h, idx)

    def test_on_hidden_names_leave_the_index_but_stay_registered(self):
        bc = self.bc
        on = self._slim(True)
        idx = on[on.index("ACTION INDEX"):]
        present = [h for h in aa.HIDDEN_DUPLICATES if h in bc.ACTIONS]
        self.assertGreaterEqual(len(present), 5, "harness registry too thin")
        for hid in present:
            with self.subTest(hidden=hid):
                self.assertTrue(callable(bc.ACTIONS[hid]))
                if hid == "max":
                    continue   # a common word; checked by name elsewhere
                self.assertFalse(_word_in(hid, idx))

    def test_hidden_names_keep_their_risk_tier_in_the_live_gate(self):
        bc = self.bc
        before = {h: bc._needs_confirmation(h, "x")
                  for h in aa.HIDDEN_DUPLICATES}
        self._slim(True)
        after = {h: bc._needs_confirmation(h, "x")
                 for h in aa.HIDDEN_DUPLICATES}
        self.assertEqual(before, after)
        for hid, canon in aa.HIDDEN_DUPLICATES.items():
            self.assertEqual(bc._needs_confirmation(hid, "x"),
                             bc._needs_confirmation(canon, "x"), (hid, canon))

    def test_on_every_documented_action_is_named_hidden_or_routed(self):
        bc = self.bc
        skills = (bc._system_prompt_blocks[0] or {}).get("skills") or ""
        on = self._slim(True)
        core, sections = pr.split_pc_control(bc.PC_CONTROL_PROMPT)
        collapsed_bodies = " ".join(
            b for h, b in sections
            if h.strip().upper() in {x.upper() for _l, hs in pr.COLLAPSED_ROWS
                                     for x in hs})
        routed = set(pr.documented_actions(collapsed_bodies, set(bc.ACTIONS)))
        documented = pr.documented_actions(bc.PC_CONTROL_PROMPT + skills,
                                           set(bc.ACTIONS))

        def handler(n):
            f = bc.ACTIONS.get(n)
            return id(getattr(f, "__func__", f)) if f is not None else n
        present = {handler(n) for n in bc.ACTIONS if _word_in(n, on)}
        missing = [a for a in documented
                   if a not in action_risk.SELF_TERMINATING_ACTIONS
                   and not _word_in(a, on) and handler(a) not in present
                   and a not in aa.HIDDEN_DUPLICATES and a not in routed]
        self.assertEqual(missing, [])

    def test_the_two_w07_gap_names_survive_the_fold(self):
        bc = self.bc

        def _rag(*_a, **_k):
            return "rag"

        def _net(*_a, **_k):
            return "net"

        fake = {"rag_search": _rag, "search_my_files": _rag,
                "network_usage": _net, "whats_using_bandwidth": _net}
        with mock.patch.dict(bc.ACTIONS, fake), \
                mock.patch.object(bc, "PROMPT_ROUTED_V3", True, create=True), \
                mock.patch.object(bc, "PROMPT_INDEX_CANON", True, create=True):
            bc._SLIM_PC_CACHE.clear()
            try:
                out = bc._slim_stable_pc("", None)
            finally:
                bc._SLIM_PC_CACHE.clear()
        idx = out[out.index("ACTION INDEX"):]
        for name in ("search_my_files", "whats_using_bandwidth"):
            self.assertIn(name, idx)

    def test_off_and_on_agree_on_the_head(self):
        off, on = self._slim(False), self._slim(True)
        self.assertEqual(off[:off.index("ACTION INDEX")],
                         on[:on.index("ACTION INDEX")])


if __name__ == "__main__":
    unittest.main()
