"""Cross-branch interactions of the v2.0.177 release candidate (2026-10-02),
light tier (no monolith import). Each class sits where two of the seven
merged branches meet, which neither branch's own tests could reach.

  * fact provenance x the memory index re-index (voyage-embed) x guest mode:
    the background rebuild never copies a fact's provenance (the sentence it
    was learned from) into Chroma metadata, and guest mode adds nothing to
    what it rebuilds;
  * the rag-excludes secret floor x personal_rag's register()/rag_configure
    x the owner's kind of RAG_INDEX_PATHS (several Windows roots, spaces,
    one root nested in another's parent, a root whose own name holds an
    excluded word);
  * the Kinect stale-stream reset (it now holds the open lock while it
    closes the old runtime) x game mode's set_enabled(False)/(True) x the
    camera gate;
  * "diagnostic status" (now a fresh sweep) x paused diagnostic daemons.

Everything is faked: no model, sensor, index, network or settings file is
touched.

    python -m unittest tests.test_rel177_interactions
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import threading
import unittest
from unittest import mock

import core.long_term_memory as ltm
from audio import kinect_bridge as kb
from core import camera_gate as cg
from core import diagnostic_daemons as dd
from core import guest_mode
from tests import test_diagnostic_daemons as tdd
from tests.skills.test_personal_rag import _load_rag_skill
from tests.test_kinect_real_frame_loss import _BridgeBase, _Clock
from tests.test_ltm_embed_switch import _VOYAGE_COLL, _SwitchBase
from tests.test_rag_excludes import _shipped_excludes

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SENTENCE = "my cousin Quillon keeps three ferrets in the garage"


# ════════════════════════════════════════════════════════════════════════════
#  fact provenance x the background re-index x guest mode
# ════════════════════════════════════════════════════════════════════════════
class ReindexProvenanceGuestTests(_SwitchBase):
    def setUp(self):
        super().setUp()
        guest_mode.set_on(False)
        self.addCleanup(guest_mode.set_on, False)

    def _prov(self, utterance=_SENTENCE):
        return ltm.make_provenance(speaker="owner", source="voice",
                                   utterance=utterance, ts=1_759_400_000.0)

    def test_the_rebuild_never_copies_a_facts_sentence_into_chroma(self):
        ltm.ensure_loaded()
        fid = ltm.add_fact("User's cousin keeps three ferrets",
                           source="merge_memory", tags=["learned"],
                           provenance=self._prov())
        legacy = ltm.add_fact("User drinks coffee black", source="test")
        self.switch_and_build("voyage-4-nano")
        coll = self.client.colls[_VOYAGE_COLL]
        self.assertEqual(set(coll.store), {fid, legacy},
                         "the rebuild did not cover every fact")
        for fact_id, (_vec, _doc, meta) in coll.store.items():
            self.assertNotIn("provenance", meta, fact_id)
            self.assertNotIn("Quillon", json.dumps(meta), fact_id)
        # ... and the sentence is still where it belongs: facts.json.
        self.assertEqual(ltm.fact_provenance(ltm._facts[fid])["utterance"],
                         _SENTENCE)
        with open(ltm._FACTS_JSON, encoding="utf-8") as f:
            self.assertIn("Quillon", f.read())

    def test_a_fact_written_while_the_rebuild_runs_is_caught_up_without_it(self):
        # The rebuild re-reads the store each round: a fact learned between
        # rounds reaches the new index - through _chroma_meta, sentence-free.
        ltm.ensure_loaded()
        ltm.add_fact("User drinks coffee black", source="test")
        late: list = []
        real_encode = ltm._reindex_encode

        def _encode(model, prof, texts):
            if not late:
                late.append(ltm.add_fact(
                    "User's cousin keeps three ferrets", source="merge_memory",
                    provenance=self._prov()))
            return real_encode(model, prof, texts)
        with mock.patch.object(ltm, "_reindex_encode", _encode):
            self.switch_and_build("voyage-4-nano")
        coll = self.client.colls[_VOYAGE_COLL]
        self.assertIn(late[0], coll.store)
        self.assertNotIn("provenance", coll.store[late[0]][2])

    def test_guest_mode_adds_nothing_the_rebuild_could_pick_up(self):
        ltm.ensure_loaded()
        kept = ltm.add_fact("User drinks coffee black", source="test")
        guest_mode.set_on(True)
        self.assertEqual(ltm.add_fact("Guest Orlo sails a ketch",
                                      source="merge_memory",
                                      provenance=self._prov("I sail a ketch")),
                         "")
        # The rebuild itself still runs in guest mode (it learns nothing).
        self.switch_and_build("voyage-4-nano")
        coll = self.client.colls[_VOYAGE_COLL]
        self.assertEqual(set(coll.store), {kept})
        self.assertNotIn("Orlo", json.dumps(ltm._facts))
        self.assertEqual(ltm._index_profile_key, "voyage-4-nano")


# ════════════════════════════════════════════════════════════════════════════
#  the rag-excludes floor x personal_rag x the owner's kind of index roots
# ════════════════════════════════════════════════════════════════════════════
# Synthetic, but the owner's SHAPE: user folders, a synced folder, project
# folders with spaces, and one root whose own name holds "pass".
_ROOTS = [r"C:\Users\Someone\Documents", r"C:\Users\Someone\OneDrive",
          r"D:\Cloud\Documents\Senior Project", r"D:\Cloud\Documents\School",
          r"D:\Cloud\Documents\Compass Lab"]


class RagExcludesOwnerRootsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)
        # The REAL indexer, privately: the skill harness's stub owns
        # sys.modules["core.rag_indexer"], and nothing else may see this copy.
        spec = importlib.util.spec_from_file_location(
            "_rag_indexer_rel177", os.path.join(_ROOT, "core", "rag_indexer.py"))
        self.rag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.rag)
        self.shipped = _shipped_excludes()
        p = mock.patch.object(self.mod, "_rag", return_value=self.rag)
        p.start()
        self.addCleanup(p.stop)
        self._boot()

    def _boot(self):
        """register() at boot: core.config's RAG_* knobs reach the indexer."""
        from core import config as real_cfg
        with mock.patch.object(real_cfg, "RAG_ENABLED", True, create=True), \
             mock.patch.object(real_cfg, "RAG_INDEX_PATHS", list(_ROOTS),
                               create=True), \
             mock.patch.object(real_cfg, "RAG_EXCLUDE_GLOBS",
                               list(self.shipped), create=True), \
             mock.patch.object(self.mod, "RAG_AUTOSTART", False):
            self.mod.register({})

    def _x(self, path):
        return self.rag._is_excluded(path)

    def test_the_boot_push_gives_every_owner_root_the_secret_floor(self):
        self.assertEqual(self.rag.RAG_INDEX_PATHS, _ROOTS)
        self.assertEqual(self.rag.RAG_EXCLUDE_GLOBS, self.shipped)
        for root in _ROOTS:
            with self.subTest(root=root):
                self.assertTrue(self._x(root + r"\Router passwords.txt"))
                self.assertTrue(self._x(root + r"\Secrets\bank.md"))
                self.assertTrue(self._x(root + r"\exports\devices.csv"))
                self.assertFalse(self._x(root + r"\meeting notes.md"))
                self.assertFalse(self._x(root + r"\Lab 4\report.docx"))

    def test_a_roots_own_name_is_never_tested_but_its_folders_are(self):
        lab = r"D:\Cloud\Documents\Compass Lab"
        self.assertFalse(self._x(lab + r"\notes.md"))
        self.assertFalse(self._x(lab + r"\Week 3\notes.md"))
        self.assertTrue(self._x(lab + r"\Passwords\notes.md"))

    def test_a_nested_synced_folder_uses_its_own_root(self):
        # "Documents" inside OneDrive: the floor sits at the OneDrive root.
        self.assertTrue(self._x(
            r"C:\Users\Someone\OneDrive\Documents\Credentials\vpn.txt"))
        self.assertFalse(self._x(
            r"C:\Users\Someone\OneDrive\Documents\Senior Project\plan.md"))

    def test_rag_configure_index_paths_keeps_the_secret_floor(self):
        out = self.mod.rag_configure(
            r"index_paths=E:\Archive\Passage Notes, D:\Cloud\Documents\School")
        self.assertIn("index_paths", out)
        self.assertEqual(self.rag.RAG_INDEX_PATHS,
                         [r"E:\Archive\Passage Notes",
                          r"D:\Cloud\Documents\School"])
        self.assertEqual(self.rag.RAG_EXCLUDE_GLOBS, self.shipped)
        # The NEW root's floor applies at once (walk, watcher and search).
        self.assertFalse(self._x(r"E:\Archive\Passage Notes\journal.md"))
        self.assertTrue(self._x(r"E:\Archive\Passage Notes\Tokens\gh.md"))
        self.assertTrue(self._x(r"E:\Archive\Passage Notes\id_rsa"))

    def test_a_spoken_exclude_list_lasts_only_until_the_next_boot(self):
        # rag_configure REPLACES the list (documented), and the next boot's
        # register() puts core.config's list back.
        self.mod.rag_configure("exclude_globs=*.journal")
        self.assertEqual(self.rag.RAG_EXCLUDE_GLOBS, ["*.journal"])
        self._boot()
        self.assertEqual(self.rag.RAG_EXCLUDE_GLOBS, self.shipped)
        self.assertTrue(self._x(_ROOTS[0] + r"\passwords.txt"))


# ════════════════════════════════════════════════════════════════════════════
#  the Kinect reopen lock x game mode's set_enabled x the camera gate
# ════════════════════════════════════════════════════════════════════════════
class KinectResetGameModeTests(_BridgeBase):
    """The stale-stream reset holds the open lock from dropping the dead
    runtime until it is closed. Game mode turns the Kinect off (and later
    back on) from another thread, at any moment."""

    def setUp(self):
        super().setUp()
        self.clk = _Clock()
        self.gate = cg.CameraGate(clock=self.clk, log=lambda _l: None,
                                  announce=lambda _m: None)
        kb.set_open_gate(self.gate)
        self.addCleanup(kb.set_open_gate, None)
        self.pump_starts = mock.Mock(return_value=False)
        p = mock.patch.object(kb, "start_body_pump", self.pump_starts)
        p.start()
        self.addCleanup(p.stop)

    def _flip_mid_reset(self, enabled: bool, poll: bool = False):
        """Run a real stale reset; when it tells the gate (it holds the open
        lock then) game mode flips the bridge from ANOTHER thread, which
        must finish on its own - and, with ``poll``, a poller asks for the
        runtime right after. Returns (reset result, poll answers)."""
        old = self._open()
        self.pump_starts.reset_mock()           # the open's own pump check
        now = self._stall(old)
        started, flipped, answers = [], [], []
        real_gate_call = kb._gate_call

        def _hook(method, *a, **k):
            # Only the reset's own call: set_enabled's close() unholds too.
            if method == "unhold" and not started:
                started.append(True)
                t = threading.Thread(target=lambda: (
                    kb.set_enabled(enabled), flipped.append(True)),
                    daemon=True)
                t.start()
                t.join(5.0)
                self.assertTrue(flipped, "set_enabled blocked on the reset")
                if poll:
                    answers.append(kb.get_runtime())
            return real_gate_call(method, *a, **k)
        with mock.patch.object(kb, "_gate_call", side_effect=_hook):
            did, _out = self._reset(now)
        self.assertTrue(did)
        self.assertTrue(old.closed)
        return answers

    def test_game_mode_off_mid_reset_leaves_the_sensor_closed(self):
        self._flip_mid_reset(False)
        self.assertFalse(self.sensor.is_open)
        self.assertIsNone(kb._runtime[0])
        self.assertEqual(len(self.runtimes), 1)
        rt, err = kb.get_runtime()
        self.assertIsNone(rt)
        self.assertIn("disabled", err)
        self.assertEqual(len(self.runtimes), 1, "reopened after game mode")
        self.pump_starts.assert_not_called()
        self.assertEqual(self.gate._dev[kb._GATE_KEY]["held_by"], "")

    def test_game_mode_back_on_mid_reset_reopens_on_a_closed_sensor(self):
        answers = self._flip_mid_reset(True, poll=True)
        self.assertEqual(answers[0][0], None)
        self.assertEqual(answers[0][1], kb._RESET_IN_PROGRESS)
        self.assertIsNone(kb._open_error[0], "the mid-reset answer latched")
        rt = self._open()                       # the next tick
        self.assertFalse(rt.closed)
        self.assertTrue(all(self.built_on_closed),
                        "a runtime was built on the sensor the reset closed")
        self.assertTrue(self.sensor.is_open)

    def test_game_mode_closing_a_fresh_stream_is_not_a_death_on_open(self):
        # Game mode can switch the Kinect off seconds after an open; that
        # close must never count toward the dies-on-open verdict.
        for _ in range(cg.DIES_ON_OPEN_COUNT + 1):
            self._open()
            self.clk.advance(2.0)
            kb.set_enabled(False)
            did, _out = self._reset(10_000.0)   # long after any frame
            self.assertFalse(did)
            kb.set_enabled(True)
            self.clk.advance(120.0)
        rec = self.gate._dev[kb._GATE_KEY]
        self.assertEqual(rec["doo_count"], 0)
        self.assertEqual(rec["doo_retry_s"], 0.0)


# ════════════════════════════════════════════════════════════════════════════
#  "diagnostic status" (a fresh sweep) x paused diagnostic daemons
# ════════════════════════════════════════════════════════════════════════════
class FreshStatusLeavesThePauseAloneTests(tdd._Base):
    def _paused_state(self):
        with open(dd.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"paused": True,
                       "self_diag": {"last_run_iso": "2026-06-01T00:00:00",
                                     "runs": 7}}, f)
        with open(dd.STATE_FILE, "rb") as f:
            return f.read()

    def test_a_fresh_sweep_while_paused_never_touches_the_pause(self):
        before = self._paused_state()
        sweep = mock.Mock(return_value="All systems nominal, sir. "
                                       "(2.0s sweep, 20 probes.)")
        with contextlib.redirect_stdout(io.StringIO()):
            out = dd.fresh_diagnostic_status(sweep)
        sweep.assert_called_once_with("")
        self.assertTrue(out.endswith("The background diagnostics are still "
                                     "paused."), out)
        with open(dd.STATE_FILE, "rb") as f:
            self.assertEqual(f.read(), before, "the status wrote the state")
        # The daemons' loops still read paused, and boot would not see drift.
        self.assertTrue(dd._read_state().get("paused"))
        self.assertFalse(dd.reconcile_paused(True))

    def test_resumed_daemons_get_no_pause_note(self):
        self._paused_state()
        with contextlib.redirect_stdout(io.StringIO()):
            dd.act_resume_diagnostics()
            out = dd.fresh_diagnostic_status(mock.Mock(return_value="Done."))
        self.assertEqual(out, "Done.")


if __name__ == "__main__":
    unittest.main()
