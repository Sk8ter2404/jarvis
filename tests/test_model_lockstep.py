"""Tests for core/model_lockstep.py — the ONE copy of the chat↔vision lockstep.

core/config.py ships ONE multimodal brain: LOCAL_VISION_MODEL and
LOCAL_LLM_MODEL point at the same tag, so a vision call re-uses the resident
chat model instead of co-loading a second VLM on top of it. Promoting the
brain must therefore carry vision along.

The 2026-08-20 audit found that rule implemented at ONE of FOUR switch sites:
skills/model_picker had it (and core/actions reuses that helper), but the
Settings GUI's Save and the web dashboard's POST /api/settings each repointed
LOCAL_LLM_MODEL with no vision consideration at all — forking the pair, and
permanently, because the voice path then reads the mismatch as a deliberately
pinned VLM and refuses to repair it. This suite pins:

  * the shared decision rule itself (sync / pinned / already / text-only), and
    the tag-identity fix it depends on — 'gemma4:26b-a4b-it-qat' vs
    'gemma4:12b' used to compare EQUAL (neither size was in the hardcoded
    SIZE_ORDER token list), which made the rule answer "already in lockstep"
    for the single most likely switch on this box;
  * every switch site applying it: the voice path, the Settings GUI's collect
    step, and the web dashboard's settings writer (behavioural, on a temp
    settings file — no Tk, no HTTP server, no network);
  * SOURCE-SCAN INVARIANTS so a FOURTH site fails this suite instead of
    shipping: any file that writes the chat tag must reference the shared
    helper, the set of schema-driven settings writers is pinned, and the
    vision-marker tuple may exist in exactly one place.

stdlib unittest only. No network: the offline family check decides, and the
Ollama /api/show probe is never reached (the settings writers don't use it).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from unittest import mock

from core import model_lockstep as ml
from tools import settings_window as sw
from tools import web_interface as wi

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)

# The shipped one-brain pair and the models around it (see core/config.py and
# settings_window.OLLAMA_MODEL_FALLBACK).
BRAIN_26B = "gemma4:26b-a4b-it-qat"     # shipped chat AND vision tag
BRAIN_12B = "gemma4:12b"                # multimodal sibling, one step down
TEXT_32B = "qwen2.5:32b-instruct-q4_K_M"   # text-only
PINNED_VLM = "qwen2.5vl:7b"             # a user-pinned separate VLM


# ═══════════════════════════════════════════════════════════════════════════
#  The rule itself
# ═══════════════════════════════════════════════════════════════════════════
class TagIdentityTests(unittest.TestCase):
    """same_model decides "is vision still on the old brain?" — the whole
    lockstep hangs off it."""

    def test_declared_size_ignores_the_moe_active_param_suffix(self):
        self.assertEqual(ml.declared_size_b(BRAIN_26B), 26.0)   # not 4 (a4b)
        self.assertEqual(
            ml.declared_size_b("qwen3:30b-a3b-instruct-2507-q4_K_M"), 30.0)
        self.assertEqual(ml.declared_size_b(BRAIN_12B), 12.0)
        self.assertEqual(ml.declared_size_b("qwen2:1.5b"), 1.5)
        self.assertIsNone(ml.declared_size_b("gemma4:latest"))
        self.assertIsNone(ml.declared_size_b(""))

    def test_same_family_different_size_is_a_different_model(self):
        # THE regression: 26b and 12b are both absent from SIZE_ORDER, so the
        # old token-table comparison called them the same model and the
        # lockstep concluded "already in lockstep" for the most likely switch
        # this box will ever see.
        self.assertFalse(ml.same_model(BRAIN_26B, BRAIN_12B))
        self.assertFalse(ml.same_model("qwen2.5:14b-instruct-q5_K_M", TEXT_32B))

    def test_exact_and_size_free_tags_still_match(self):
        self.assertTrue(ml.same_model(BRAIN_26B, BRAIN_26B))
        self.assertTrue(ml.same_model(BRAIN_26B, BRAIN_26B.upper()))
        # A bare base name carries no contradicting size information.
        self.assertTrue(ml.same_model("qwen2.5", TEXT_32B))
        self.assertTrue(ml.same_model(BRAIN_26B, "gemma4:latest"))
        # Different families never match.
        self.assertFalse(ml.same_model(BRAIN_26B, TEXT_32B))
        self.assertFalse(ml.same_model("", BRAIN_26B))

    def test_offline_multimodal_check(self):
        self.assertTrue(ml.is_multimodal_tag(BRAIN_26B))     # gemma4 family
        self.assertTrue(ml.is_multimodal_tag(PINNED_VLM))    # vl: marker
        self.assertFalse(ml.is_multimodal_tag(TEXT_32B))
        self.assertFalse(ml.is_multimodal_tag(""))


class LockstepDecisionTests(unittest.TestCase):
    def test_shared_brain_switch_moves_vision(self):
        self.assertEqual(
            ml.vision_lockstep_decision(BRAIN_26B, BRAIN_12B, BRAIN_26B),
            (BRAIN_12B, ml.LOCKSTEP_SYNC))
        self.assertEqual(
            ml.vision_tag_after_chat_switch(BRAIN_26B, BRAIN_12B, BRAIN_26B),
            BRAIN_12B)

    def test_pinned_separate_vlm_is_never_touched(self):
        tag, reason = ml.vision_lockstep_decision(BRAIN_26B, BRAIN_12B,
                                                  PINNED_VLM)
        self.assertIsNone(tag)
        self.assertEqual(reason, ml.LOCKSTEP_PINNED)
        # A pin inside the SAME family is still a pin (26B brain, 12B VLM).
        self.assertEqual(
            ml.vision_lockstep_decision(BRAIN_26B, "gemma4:latest",
                                        BRAIN_12B)[1], ml.LOCKSTEP_PINNED)

    def test_text_only_target_leaves_vision_alone_with_a_reason(self):
        tag, reason = ml.vision_lockstep_decision(BRAIN_26B, TEXT_32B,
                                                  BRAIN_26B)
        self.assertIsNone(tag)
        self.assertEqual(reason, ml.LOCKSTEP_TEXT_ONLY)

    def test_no_change_and_missing_tags(self):
        self.assertEqual(
            ml.vision_lockstep_decision(BRAIN_26B, BRAIN_26B, BRAIN_26B)[1],
            ml.LOCKSTEP_ALREADY)
        for args in ((None, BRAIN_12B, BRAIN_26B), (BRAIN_26B, "", BRAIN_26B),
                     (BRAIN_26B, BRAIN_12B, None)):
            self.assertEqual(ml.vision_lockstep_decision(*args)[1],
                             ml.LOCKSTEP_INCOMPLETE)

    def test_injected_capability_probe_overrides_the_family_check(self):
        # The voice path injects model_picker._is_multimodal (Ollama's declared
        # capabilities); a probe that says "no" must veto the sync...
        self.assertIsNone(ml.vision_tag_after_chat_switch(
            BRAIN_26B, BRAIN_12B, BRAIN_26B, is_multimodal=lambda t: False))
        # ...and one that says "yes" enables a tag the markers don't know.
        self.assertEqual(ml.vision_tag_after_chat_switch(
            BRAIN_26B, "minicpm-v:8b", BRAIN_26B,
            is_multimodal=lambda t: True), "minicpm-v:8b")
        # A probe that RAISES falls back to the offline check, never escapes.
        def _boom(_tag):
            raise OSError("ollama down")
        self.assertEqual(ml.vision_tag_after_chat_switch(
            BRAIN_26B, BRAIN_12B, BRAIN_26B, is_multimodal=_boom), BRAIN_12B)

    def test_config_default_resolves_absent_keys(self):
        import core.config as cfg
        self.assertEqual(ml.config_default("LOCAL_VISION_MODEL"),
                         cfg.LOCAL_VISION_MODEL)
        self.assertIsNone(ml.config_default("NO_SUCH_CONSTANT_XYZ"))
        self.assertEqual(ml.config_default("NO_SUCH_CONSTANT_XYZ", "fb"), "fb")


# ═══════════════════════════════════════════════════════════════════════════
#  Site 1/3 — the Settings GUI's Save (tools/settings_window.py)
# ═══════════════════════════════════════════════════════════════════════════
class SettingsGuiLockstepTests(unittest.TestCase):
    """apply_vision_lockstep is exactly what _collect() runs before Save, with
    ``base`` = the document on disk and ``out`` = the document about to be
    written. No Tk involved."""

    def test_chat_switch_carries_vision(self):
        base = {"LOCAL_LLM_MODEL": BRAIN_26B, "LOCAL_VISION_MODEL": BRAIN_26B}
        out = dict(base, LOCAL_LLM_MODEL=BRAIN_12B)
        tag, reason = sw.apply_vision_lockstep(base, out)
        self.assertEqual((tag, reason), (BRAIN_12B, ml.LOCKSTEP_SYNC))
        self.assertEqual(out["LOCAL_VISION_MODEL"], BRAIN_12B)

    def test_fresh_install_uses_the_config_defaults_for_the_absent_keys(self):
        # LOCAL_VISION_MODEL has no schema row, so a fresh data/user_settings
        # .json contains NEITHER key — yet both constants are live at their
        # config values, so the switch must still be caught.
        out = {"LOCAL_LLM_MODEL": BRAIN_12B}
        tag, reason = sw.apply_vision_lockstep({}, out)
        self.assertEqual((tag, reason), (BRAIN_12B, ml.LOCKSTEP_SYNC))
        self.assertEqual(out["LOCAL_VISION_MODEL"], BRAIN_12B)

    def test_pinned_vlm_survives_a_save(self):
        base = {"LOCAL_LLM_MODEL": BRAIN_26B, "LOCAL_VISION_MODEL": PINNED_VLM}
        out = dict(base, LOCAL_LLM_MODEL=BRAIN_12B)
        tag, reason = sw.apply_vision_lockstep(base, out)
        self.assertIsNone(tag)
        self.assertEqual(reason, ml.LOCKSTEP_PINNED)
        self.assertEqual(out["LOCAL_VISION_MODEL"], PINNED_VLM)

    def test_text_only_switch_reports_why_it_did_nothing(self):
        base = {"LOCAL_LLM_MODEL": BRAIN_26B, "LOCAL_VISION_MODEL": BRAIN_26B}
        out = dict(base, LOCAL_LLM_MODEL=TEXT_32B)
        tag, reason = sw.apply_vision_lockstep(base, out)
        self.assertIsNone(tag)
        self.assertEqual(reason, ml.LOCKSTEP_TEXT_ONLY)
        self.assertEqual(out["LOCAL_VISION_MODEL"], BRAIN_26B)

    def test_an_unrelated_save_touches_nothing(self):
        base = {"LOCAL_LLM_MODEL": BRAIN_26B, "LOCAL_VISION_MODEL": BRAIN_26B,
                "TTS_VOICE": "en-GB-RyanNeural"}
        out = dict(base, TTS_VOICE="en-GB-ThomasNeural")
        tag, reason = sw.apply_vision_lockstep(base, out)
        self.assertIsNone(tag)
        self.assertEqual(reason, ml.LOCKSTEP_ALREADY)
        self.assertEqual(out["LOCAL_VISION_MODEL"], BRAIN_26B)

    def test_written_document_round_trips_through_save_settings(self):
        """The synced key must actually LAND on disk: LOCAL_VISION_MODEL has no
        schema row, so this pins save_settings' passthrough behaviour (the same
        property skills/model_picker._persist_setting relies on)."""
        d = tempfile.mkdtemp()
        path = os.path.join(d, "user_settings.json")
        base = {"LOCAL_LLM_MODEL": BRAIN_26B, "LOCAL_VISION_MODEL": BRAIN_26B}
        out = dict(base, LOCAL_LLM_MODEL=BRAIN_12B)
        sw.apply_vision_lockstep(base, out)
        sw.save_settings(out, path)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["LOCAL_LLM_MODEL"], BRAIN_12B)
        self.assertEqual(doc["LOCAL_VISION_MODEL"], BRAIN_12B)


# ═══════════════════════════════════════════════════════════════════════════
#  Site 2/3 — the web dashboard (POST /api/settings → _write_settings)
# ═══════════════════════════════════════════════════════════════════════════
class WebSettingsLockstepTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "user_settings.json")

    def _seed(self, doc):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(doc, f)

    def _doc(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def test_chat_switch_carries_vision_and_is_reported(self):
        self._seed({"LOCAL_LLM_MODEL": BRAIN_26B,
                    "LOCAL_VISION_MODEL": BRAIN_26B})
        applied = wi._write_settings({"LOCAL_LLM_MODEL": BRAIN_12B}, self.path)
        # Written...
        self.assertEqual(self._doc()["LOCAL_VISION_MODEL"], BRAIN_12B)
        # ...and REPORTED: the API answer must name every key it changed, or
        # the panel silently rewrites a model tag behind the caller's back.
        self.assertEqual(applied.get("LOCAL_VISION_MODEL"), BRAIN_12B)

    def test_fresh_file_without_either_key(self):
        self._seed({"TTS_VOICE": "en-GB-RyanNeural"})
        applied = wi._write_settings({"LOCAL_LLM_MODEL": BRAIN_12B}, self.path)
        self.assertEqual(applied.get("LOCAL_VISION_MODEL"), BRAIN_12B)
        doc = self._doc()
        self.assertEqual(doc["LOCAL_VISION_MODEL"], BRAIN_12B)
        self.assertEqual(doc["TTS_VOICE"], "en-GB-RyanNeural")  # preserved

    def test_pinned_vlm_and_text_only_targets_are_left_alone(self):
        self._seed({"LOCAL_LLM_MODEL": BRAIN_26B,
                    "LOCAL_VISION_MODEL": PINNED_VLM})
        applied = wi._write_settings({"LOCAL_LLM_MODEL": BRAIN_12B}, self.path)
        self.assertNotIn("LOCAL_VISION_MODEL", applied)
        self.assertEqual(self._doc()["LOCAL_VISION_MODEL"], PINNED_VLM)

        self._seed({"LOCAL_LLM_MODEL": BRAIN_26B,
                    "LOCAL_VISION_MODEL": BRAIN_26B})
        applied = wi._write_settings({"LOCAL_LLM_MODEL": TEXT_32B}, self.path)
        self.assertNotIn("LOCAL_VISION_MODEL", applied)
        self.assertEqual(self._doc()["LOCAL_VISION_MODEL"], BRAIN_26B)

    def test_a_write_that_does_not_touch_the_chat_model_is_unaffected(self):
        self._seed({"LOCAL_LLM_MODEL": BRAIN_26B,
                    "LOCAL_VISION_MODEL": BRAIN_26B})
        applied = wi._write_settings({"WAKE_WORD_AUTOSTART": True}, self.path)
        self.assertEqual(list(applied), ["WAKE_WORD_AUTOSTART"])
        self.assertEqual(self._doc()["LOCAL_VISION_MODEL"], BRAIN_26B)

    def test_degraded_lockstep_is_logged_not_silent(self):
        """A text-only switch is a legitimate degrade — but it must SAY so."""
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            tag = wi._lockstep_vision({"LOCAL_LLM_MODEL": BRAIN_26B,
                                       "LOCAL_VISION_MODEL": BRAIN_26B},
                                      TEXT_32B)
        self.assertIsNone(tag)
        self.assertIn("vision", buf.getvalue().lower())
        self.assertIn(TEXT_32B, buf.getvalue())


# ═══════════════════════════════════════════════════════════════════════════
#  Site 3/3 — the voice path still shares the one rule
# ═══════════════════════════════════════════════════════════════════════════
class VoicePathUsesTheSharedRuleTests(unittest.TestCase):
    def test_model_picker_imports_the_shared_primitives(self):
        from skills import model_picker as mp
        self.assertIs(mp._same_model, ml.same_model)
        self.assertIs(mp._size_rank, ml.size_rank)
        self.assertIs(mp._VISION_MARKERS, ml.VISION_MARKERS)

    def test_sync_vision_to_chat_obeys_the_shared_decision(self):
        """The voice helper still owns only the ASSIGNMENT — the decision is
        the shared rule. requests.post is patched to fail so the Ollama
        capability probe is never made (no network in this suite) and the
        offline family check decides, exactly as on a box with Ollama down."""
        from skills import model_picker as mp
        import core.config as cfg

        class _BC:
            LOCAL_VISION_MODEL = BRAIN_26B

        bc = _BC()
        saved = cfg.LOCAL_VISION_MODEL
        with mock.patch.object(mp.requests, "post",
                               side_effect=OSError("ollama down")):
            try:
                # Text-only target → declined, nothing repointed.
                self.assertFalse(mp._sync_vision_to_chat(
                    BRAIN_26B, TEXT_32B, persist=False, bc=bc))
                self.assertEqual(bc.LOCAL_VISION_MODEL, BRAIN_26B)
                # Pinned separate VLM → declined.
                bc.LOCAL_VISION_MODEL = PINNED_VLM
                self.assertFalse(mp._sync_vision_to_chat(
                    BRAIN_26B, BRAIN_12B, persist=False, bc=bc))
                self.assertEqual(bc.LOCAL_VISION_MODEL, PINNED_VLM)
                # Shared brain → repointed live (persist=False = no disk write).
                bc.LOCAL_VISION_MODEL = BRAIN_26B
                self.assertTrue(mp._sync_vision_to_chat(
                    BRAIN_26B, BRAIN_12B, persist=False, bc=bc))
                self.assertEqual(bc.LOCAL_VISION_MODEL, BRAIN_12B)
            finally:
                cfg.LOCAL_VISION_MODEL = saved


# ═══════════════════════════════════════════════════════════════════════════
#  Source-scan invariants — a FOURTH site must fail this suite
# ═══════════════════════════════════════════════════════════════════════════
def _sources(roots=("skills", "core", "tools")):
    """(relpath, source) for every non-test .py under `roots`."""
    out = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(os.path.join(_PROJECT,
                                                                 root)):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".claude", "backups",
                                        "worktrees")]
            for fn in filenames:
                if not fn.endswith(".py") or fn.startswith("test_"):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        out.append((os.path.relpath(path, _PROJECT), f.read()))
                except OSError:
                    continue
    return out


# Names that prove a file reaches the ONE rule rather than re-deriving it.
_HELPER_NAMES = ("vision_lockstep_decision", "vision_tag_after_chat_switch",
                 "_sync_vision_to_chat", "_lockstep_vision",
                 "apply_vision_lockstep")

# A chat-model switch site repoints the LIVE resolver cache...
_CACHE_WRITE_RES = (
    re.compile(r"cache\[0\]\s*="),
    re.compile(r"setattr\([^)\n]*_RESOLVED_LOCAL_LLM_MODEL"),
)
# ...or writes the tag itself (a constant, or the persisted document). The
# 2026-07-21 invariant knew only the cache form, which is exactly why the two
# settings writers — which PERSIST the tag — were invisible to it.
_TAG_WRITE_RES = (
    re.compile(r"_persist_setting\(\s*[\"']LOCAL_LLM_MODEL"),
    re.compile(r"\[\s*[\"']LOCAL_LLM_MODEL[\"']\s*\]\s*="),
    re.compile(r"\.LOCAL_LLM_MODEL\s*=[^=]"),
)


def _writes_chat_tag(src: str) -> bool:
    """True when this source can change which local chat model JARVIS runs."""
    if "_RESOLVED_LOCAL_LLM_MODEL" in src and any(rx.search(src)
                                                  for rx in _CACHE_WRITE_RES):
        return True                      # cache[0] = <tag> (live switch)
    return any(rx.search(src) for rx in _TAG_WRITE_RES)


class ChatTagWriterInvariantTests(unittest.TestCase):
    def test_every_chat_tag_writer_reaches_the_shared_lockstep(self):
        """The 2026-07-21 version of this scan walked only skills/ and core/
        and keyed off the resolver CACHE — so both settings writers (which
        persist the tag instead) were invisible to it and shipped the fork.
        tools/ is in scope now, and persisting the tag counts as a switch."""
        offenders = []
        for rel, src in _sources():
            if os.path.normpath(rel) == os.path.join("core",
                                                     "model_lockstep.py"):
                continue                       # the rule itself
            if not _writes_chat_tag(src):
                continue
            if not any(name in src for name in _HELPER_NAMES):
                offenders.append(rel)
        self.assertEqual(
            offenders, [],
            f"these files repoint LOCAL_LLM_MODEL without the chat<->vision "
            f"lockstep from core.model_lockstep: {offenders}. A chat switch "
            f"that leaves LOCAL_VISION_MODEL behind forks the shipped "
            f"one-multimodal-brain config onto a second VLM.")

    def test_the_known_switch_sites_are_visible_to_the_scan(self):
        """Self-check: the scan must still SEE the four known switch sites, or
        it has rotted into a no-op that passes because it matches nothing."""
        seen = {rel for rel, src in _sources() if _writes_chat_tag(src)}
        for rel in (os.path.join("skills", "model_picker.py"),
                    os.path.join("core", "actions.py")):
            self.assertIn(rel, seen, f"{rel} is no longer detected as a "
                                     f"chat-model switch site")


class SettingsWriterRegistryTests(unittest.TestCase):
    """The web dashboard hid from the old invariant because it never mentions
    LOCAL_LLM_MODEL: it writes whatever key the SCHEMA accepts. So the set of
    schema-driven settings writers is pinned here — a NEW one fails this test
    and its author has to decide whether it can move the chat model."""

    # relpath -> must it apply the lockstep?
    _KNOWN = {
        os.path.join("tools", "settings_window.py"): True,   # the GUI's Save
        os.path.join("tools", "web_interface.py"): True,     # POST /api/settings
        os.path.join("tools", "setup_wizard.py"): False,     # fixed key list
    }

    @staticmethod
    def _is_schema_writer(rel, src):
        if os.path.normpath(rel) == os.path.join("tools", "settings_window.py"):
            return "def save_settings(" in src
        return (("SCHEMA" in src or "build_settings_schema" in src)
                and ("save_settings(" in src or "_write_settings(" in src))

    def test_schema_driven_settings_writers_are_the_known_set(self):
        found = {rel for rel, src in _sources() if self._is_schema_writer(rel,
                                                                         src)}
        self.assertEqual(
            found, set(self._KNOWN),
            "the set of schema-driven settings writers changed. A writer that "
            "can persist LOCAL_LLM_MODEL must apply the chat<->vision lockstep "
            "(core.model_lockstep); one that writes a fixed key list may be "
            "added to _KNOWN with False.")

    def test_writers_that_can_move_the_chat_model_apply_the_lockstep(self):
        for rel, needs in self._KNOWN.items():
            with open(os.path.join(_PROJECT, rel), encoding="utf-8") as f:
                src = f.read()
            if needs:
                self.assertTrue(
                    any(name in src for name in _HELPER_NAMES),
                    f"{rel} can persist LOCAL_LLM_MODEL but never reaches the "
                    f"vision lockstep")
            else:
                self.assertNotIn(
                    "LOCAL_LLM_MODEL", src,
                    f"{rel} is registered as unable to write the chat model, "
                    f"but it names LOCAL_LLM_MODEL — re-check and flip its "
                    f"_KNOWN flag to True (plus wire the lockstep) if it can "
                    f"now persist it")


class NoDuplicateRuleTests(unittest.TestCase):
    """The rule and the tag semantics it rests on may exist ONCE.

    Three files each carried an identical private vision-marker tuple before
    2026-08-20 (core/vram_budget, skills/model_picker, tools/settings_window).
    Two now import it; the third keeps a deliberate SUPERSET (vision PLUS
    embedding markers) for its chat dropdown and is asserted to stay a
    superset below, so drift fails a test instead of going unnoticed."""

    _MARKER_HOLDERS = {
        os.path.join("core", "model_lockstep.py"),      # the definition
        os.path.join("tools", "settings_window.py"),    # superset mirror
    }

    def test_vision_markers_are_defined_in_one_place_plus_one_mirror(self):
        literal = re.compile(r'"moondream"')
        defs = {os.path.normpath(rel) for rel, src in _sources()
                if literal.search(src)}
        self.assertEqual(
            defs, {os.path.normpath(p) for p in self._MARKER_HOLDERS},
            f"the vision-marker tuple moved or was duplicated: {sorted(defs)}. "
            f"It is defined once in core/model_lockstep.py (VISION_MARKERS) "
            f"and imported from there; the only permitted second copy is the "
            f"Settings GUI's superset, which the next test pins.")

    def test_settings_gui_chat_filter_still_covers_every_vision_marker(self):
        for marker in ml.VISION_MARKERS:
            self.assertIn(marker, sw._NON_CHAT_MARKERS,
                          f"the chat-model dropdown would offer {marker!r} "
                          f"tags as chat brains")

    def test_gui_reason_literal_matches_the_shared_constant(self):
        """The Save button explains the text-only degrade by comparing against
        a literal (settings_window must not import core at import time). Pin
        the two together so the message can't rot into never firing."""
        self.assertEqual(sw.LOCKSTEP_TEXT_ONLY_REASON, ml.LOCKSTEP_TEXT_ONLY)


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
