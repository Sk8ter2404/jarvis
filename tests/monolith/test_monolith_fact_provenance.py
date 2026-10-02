"""Monolith wiring for fact provenance and guest mode (2026-10-02).

FACT PROVENANCE. Every learn path that writes a fact into the semantic store
(learn_from_turn's worker, the ambient learner, a spoken "remember that ...",
which is an ordinary owner turn) records where it came from:
  * each queued turn carries an origin -- the voice verdict as the speaker
    ("owner" / "unknown"), how it arrived ("voice" / "typed" / "web" /
    "ambient"), the sentence and when;
  * merge_memory gives each fact the origin of the turn it most likely came
    from and hands the records to _ltm_learn_facts -> add_fact;
  * _ltm_context remembers which facts the last recall put in the prompt,
    and "where did you learn that" (where_learned) reports who / how / when
    for the one JARVIS just used -- never the sentence itself.

GUEST MODE. "guest mode on" / "we have guests": JARVIS answers normally but
writes nothing -- no fact, project, topic, LTM episode or session summary --
until "guest mode off". The flip is saved like wake-word mode, re-applied at
boot, published to hud_state.json, and reachable from the dashboard's tray
control plane.

GENERIC fixtures only (made-up people and facts); no real memory file,
settings file, voiceprint or audio is touched.

    python -m unittest tests.monolith.test_monolith_fact_provenance
"""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_SENTENCE_1 = "my cousin Quillon keeps three ferrets"
_SENTENCE_2 = "and my sister Wenna moved to a lighthouse"


def _printed(p) -> str:
    return " ".join(" ".join(str(a) for a in c.args) for c in p.call_args_list)


def _join_learn_worker(timeout: float = 3.0) -> None:
    for t in threading.enumerate():
        if t.name == "ltm-learn":
            t.join(timeout=timeout)


@requires_monolith
class _MemoryStoreBase(MonolithGlobalsTestCase):
    """merge_memory against an in-memory bobert_memory.json; the semantic
    mirror (_ltm_learn_facts) is captured, never run."""

    def setUp(self):
        bc = self.bc
        self.store = bc._empty_memory() if hasattr(bc, "_empty_memory") else {}
        for k in ("facts", "projects", "topics", "sessions"):
            self.store.setdefault(k, [])
        self.saved = []
        self.mirrored = []

        def _load():
            return copy.deepcopy(self.store)

        def _save(mem):
            self.saved.append(copy.deepcopy(mem))
            self.store = copy.deepcopy(mem)

        def _mirror(facts, projects=None, provenance=None):
            self.mirrored.append((list(facts or []), list(projects or []),
                                  provenance))

        for name, value in (("load_memory", _load), ("save_memory", _save),
                            ("_ltm_learn_facts", _mirror),
                            ("_ltm_enabled", lambda: True),
                            ("LEARN_ONLY_FROM_OWNER", False)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)


# ═══════════════════════════════════════════════════════════════════════════
#  Fact provenance
# ═══════════════════════════════════════════════════════════════════════════

@requires_monolith
class LearnOriginTests(MonolithGlobalsTestCase):
    """learn_from_turn queues each turn with its origin (6th field)."""

    def setUp(self):
        bc = self.bc
        self.enqueued = []
        for name, value in (("LEARN_ONLY_FROM_OWNER", False),
                            ("LEARN_EVERY_TURN", True),
                            ("_dialogue_gate_active", lambda: False),
                            ("_learn_enqueue", self.enqueued.append)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _origin(self, *args, **kw):
        self.enqueued.clear()
        self.bc.learn_from_turn(*args, **kw)
        self.assertEqual(len(self.enqueued), 1)
        return self.enqueued[0][5]

    def test_a_spoken_turn_is_voice(self):
        before = time.time()
        o = self._origin(_SENTENCE_1, "Noted, sir.", {})
        self.assertEqual(o["source"], "voice")
        self.assertEqual(o["utterance"], _SENTENCE_1)
        self.assertEqual(o["speaker"], "unknown")   # no voiceprint verdict
        self.assertGreaterEqual(o["ts"], before)

    def test_a_typed_turn_is_typed_and_the_web_page_is_web(self):
        bc = self.bc
        bc._last_inject_source[0] = ""
        self.assertEqual(self._origin("x", "", {}, injected=True)["source"],
                         "typed")
        bc._last_inject_source[0] = "web"
        self.assertEqual(self._origin("x", "", {}, injected=True)["source"],
                         "web")

    def test_overheard_speech_is_ambient_and_the_owner_voice_is_owner(self):
        o = self._origin("we should repaint", "", {}, owner_directed=False,
                         voice=self.bc._learn_gate_mod.OWNER)
        self.assertEqual((o["source"], o["speaker"]), ("ambient", "owner"))

    def test_an_explicit_remember_that_is_an_owner_turn_like_any_other(self):
        # There is no separate "remember" action: the owner's "remember
        # that ..." is answered, then learned through this same path, so
        # its facts carry the same origin.
        said = "Jarvis, remember that my cousin keeps three ferrets"
        o = self._origin(said, "I'll remember that, sir.", {})
        self.assertEqual((o["source"], o["utterance"]), ("voice", said))

    def test_an_explicit_channel_wins(self):
        self.assertEqual(self._origin("x", "", {}, channel="web")["source"],
                         "web")

    def test_with_the_learn_gate_on_the_speaker_is_the_verdict(self):
        bc = self.bc
        lg = bc._learn_gate_mod
        bc._learn_gate_state[0] = lg.LearnGate(90)
        verdicts = [(lg.OWNER, 0.82)]
        with mock.patch.object(bc, "LEARN_ONLY_FROM_OWNER", True), \
             mock.patch.object(bc, "_learn_gate_submit",
                               bc._learn_gate_classify), \
             mock.patch.object(bc, "_learn_voice_verdict",
                               lambda a, s: verdicts.pop(0)):
            o = self._origin(_SENTENCE_1, "Noted.", {}, audio=object(),
                             sample_rate=16000)
            self.assertEqual((o["speaker"], o["source"]), ("owner", "voice"))
            # A typed turn has no voice to check: never "owner".
            o = self._origin("I take the train", "", {}, injected=True)
            self.assertEqual((o["speaker"], o["source"]),
                             ("unknown", "typed"))

    def test_the_batch_provenance_lists_every_origin(self):
        bc = self.bc
        o1 = bc._learn_origin(_SENTENCE_1, "voice")
        o2 = bc._learn_origin(_SENTENCE_2, "typed")
        prov = bc._learn_provenance([("a", "", True, None, None, o1),
                                     ("b", "", True, None, None, o2)])
        self.assertEqual(prov["origins"], [o1, o2])
        # Older 4- / 5-tuples simply have none.
        self.assertEqual(bc._learn_provenance(
            [("a", "", True, None)])["origins"], [])


class FactProvenanceMergeTests(_MemoryStoreBase):
    def _origin(self, text, source="voice", speaker="owner", ts=1.79e9):
        return {"speaker": speaker, "source": source, "utterance": text,
                "ts": ts}

    def test_each_fact_records_the_turn_it_came_from(self):
        bc = self.bc
        batch = [("u1", "", True, None, None,
                  self._origin(_SENTENCE_1, "voice", "owner")),
                 ("u2", "", True, None, None,
                  self._origin(_SENTENCE_2, "web", "unknown"))]
        added, _ = bc.merge_memory(
            new_facts=["User's sister Wenna lives in a lighthouse",
                       "User's cousin keeps three ferrets"],
            provenance=bc._learn_provenance(batch))
        self.assertEqual(len(added), 2)
        (_f, _p, prov), = self.mirrored
        wenna = prov["User's sister Wenna lives in a lighthouse"]
        ferrets = prov["User's cousin keeps three ferrets"]
        self.assertEqual((wenna["source"], wenna["speaker"],
                          wenna["utterance"]), ("web", "unknown", _SENTENCE_2))
        self.assertEqual((ferrets["source"], ferrets["speaker"],
                          ferrets["utterance"]),
                         ("voice", "owner", _SENTENCE_1))
        self.assertEqual(ferrets["ts"], 1.79e9)
        # bobert_memory.json still holds bare strings.
        self.assertTrue(all(isinstance(f, str)
                            for f in self.saved[-1]["facts"]))

    def test_a_learner_without_origins_gets_one_from_its_flags(self):
        # The ambient multimodal extractor's provenance shape.
        self.bc.merge_memory(new_facts=["User owns a canoe"],
                             provenance={"owner_directed": False,
                                         "source": "ambient extractor"})
        (_f, _p, prov), = self.mirrored
        rec = prov["User owns a canoe"]
        self.assertEqual((rec["speaker"], rec["source"], rec["utterance"]),
                         ("unknown", "ambient", ""))

    def test_a_deliberate_write_records_no_provenance(self):
        self.bc.merge_memory(new_facts=["User owns a canoe"])
        self.assertIsNone(self.mirrored[0][2])

    def test_the_extraction_path_end_to_end(self):
        bc = self.bc
        batch = [(_SENTENCE_1, "Noted.", True, None, None,
                  bc._learn_origin(_SENTENCE_1, "voice"))]
        with mock.patch.object(bc, "_rebuild_after_learning"), \
             mock.patch("builtins.print") as p:
            bc._learn_apply(json.dumps({
                "new_facts": ["User's cousin keeps three ferrets"],
                "new_projects": [], "topic": ""}), batch)
        prov = self.mirrored[0][2]
        self.assertEqual(prov["User's cousin keeps three ferrets"]["source"],
                         "voice")
        # The fact may be logged ([learned]); the sentence never is.
        self.assertNotIn("Quillon", _printed(p))


@requires_monolith
class LtmLearnFactsProvenanceTests(MonolithGlobalsTestCase):
    def test_each_text_reaches_add_fact_with_its_record(self):
        bc = self.bc
        fake = mock.Mock()
        rec = {"speaker": "owner", "source": "voice", "utterance": "u",
               "ts": 1.79e9}
        with mock.patch.object(bc, "_ltm_enabled", return_value=True), \
             mock.patch.object(bc, "_ltm_module", return_value=fake):
            bc._ltm_learn_facts(["User likes tea"], ["Building a kite"],
                                provenance={"User likes tea": rec})
            _join_learn_worker()
        got = {c.args[0]: c.kwargs.get("provenance")
               for c in fake.add_fact.call_args_list}
        self.assertEqual(got, {"User likes tea": rec,
                               "Building a kite": None})


@requires_monolith
class RecallRecordTests(MonolithGlobalsTestCase):
    FACT = {"id": "fact_a", "text": "User's cousin keeps three ferrets",
            "tags": ["learned"],
            "provenance": {"speaker": "owner", "source": "voice",
                           "utterance": _SENTENCE_1, "ts": 1.79e9}}

    def _recall(self, user_text):
        bc = self.bc
        fake = mock.Mock()
        fake.retrieve_facts.return_value = [dict(self.FACT)]
        with mock.patch.object(bc, "_ltm_enabled", return_value=True), \
             mock.patch.object(bc, "_ltm_module", return_value=fake):
            return bc._ltm_context(user_text)

    def test_the_recall_remembers_what_it_put_in_the_prompt(self):
        out = self._recall("what does my cousin keep")
        self.assertIn("three ferrets", out)
        # Privacy: the prompt carries the fact, never the sentence.
        self.assertNotIn("Quillon", out)
        rec = self.bc._ltm_last_recalled[0]
        self.assertEqual([f["id"] for f in rec["facts"]], ["fact_a"])

    def test_the_provenance_question_keeps_the_previous_record(self):
        bc = self.bc
        bc._ltm_last_recalled[0] = {"ts": time.time(),
                                    "facts": [{"id": "earlier",
                                               "text": "User likes tea"}]}
        self._recall("where did you learn that")
        self.assertEqual(bc._ltm_last_recalled[0]["facts"][0]["id"],
                         "earlier")

    def test_which_turns_ask_where_a_fact_came_from(self):
        yes = ("where did you learn that", "Jarvis, who told you that?",
               "how do you know that", "how did you know that?",
               "where'd you hear that",
               "what's your source for that", "how did you find out")
        no = ("where is the nearest pharmacy", "do you know that song",
              "tell me what you learned today", "who is on the wifi")
        for t in yes:
            self.assertTrue(self.bc.is_where_learned_question(t), t)
        for t in no:
            self.assertFalse(self.bc.is_where_learned_question(t), t)


def _noon_today() -> float:
    """Today's local noon: a provenance stamp that reads "today" whatever
    the hour the suite runs (a now-minus-N stamp crosses midnight)."""
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 12, 0, 0, 0, 0,
                        -1))


@requires_monolith
class WhereLearnedActionTests(MonolithGlobalsTestCase):
    FERRETS = {"id": "fact_f", "text": "User's cousin keeps three ferrets",
               "provenance": {"speaker": "owner", "source": "voice",
                              "utterance": _SENTENCE_1, "ts": None}}
    TEA = {"id": "fact_t", "text": "User drinks green tea",
           "provenance": {"speaker": "unknown", "source": "typed",
                          "utterance": "I drink green tea", "ts": None}}
    LEGACY = {"id": "fact_l", "text": "User rides a blue bicycle",
              "source": "bobert_memory_migration"}

    def _ask(self, facts, reply, age=5.0):
        bc = self.bc
        stamped = []
        for f in facts:
            f = copy.deepcopy(f)
            if isinstance(f.get("provenance"), dict):
                f["provenance"]["ts"] = _noon_today()
            stamped.append(f)
        bc._ltm_last_recalled[0] = {"ts": time.time() - age,
                                    "facts": stamped}
        bc.conversation_history[:] = [
            {"role": "user", "content": "what does my cousin keep"},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "where did you learn that"},
        ]
        return bc._act_where_learned("")

    def test_reports_who_how_and_when_for_the_fact_just_used(self):
        out = self._ask([self.TEA, self.FERRETS],
                        "Your cousin keeps three ferrets, sir.")
        self.assertIn("three ferrets", out)
        self.assertIn("you told me by voice, today at", out)
        self.assertIn("exact words", out)
        self.assertNotIn("Quillon", out)            # never the sentence

    def test_without_a_matching_reply_the_top_recall_is_used(self):
        out = self._ask([self.TEA, self.FERRETS], "Certainly, sir.")
        self.assertIn("green tea", out)
        self.assertIn("typed to me", out)

    def test_a_fact_from_before_provenance_says_so(self):
        out = self._ask([self.LEGACY], "You ride a blue bicycle, sir.")
        self.assertIn("blue bicycle", out)
        self.assertIn("don't have a note", out)

    def test_nothing_recalled_or_too_long_ago(self):
        bc = self.bc
        bc._ltm_last_recalled[0] = None
        self.assertIn("nothing to trace", bc._act_where_learned(""))
        self.assertIn("nothing to trace",
                      self._ask([self.FERRETS], "ferrets", age=3600))

    def test_registered_and_spoken_verbatim(self):
        bc = self.bc
        self.assertIs(bc.ACTIONS["where_learned"], bc._act_where_learned)
        self.assertIn("where_learned", bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        # Never re-summarised by the LLM (its result never reaches a prompt).
        self.assertNotIn("where_learned", bc.INFORMATIVE_ACTIONS)


# ═══════════════════════════════════════════════════════════════════════════
#  Guest mode
# ═══════════════════════════════════════════════════════════════════════════

@requires_monolith
class GuestModeActionTests(MonolithGlobalsTestCase):
    def setUp(self):
        import core.config as cfg
        bc = self.bc
        self.cfg = cfg
        saved = cfg.GUEST_MODE
        self.addCleanup(setattr, cfg, "GUEST_MODE", saved)
        cfg.GUEST_MODE = False
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.settings_file = os.path.join(tmp.name, "user_settings.json")
        env = mock.patch.dict(os.environ,
                              {"JARVIS_SETTINGS_PATH": self.settings_file})
        env.start()
        self.addCleanup(env.stop)
        self.hud = []
        p = mock.patch.object(bc, "_write_hud_state",
                              lambda **k: self.hud.append(k))
        p.start()
        self.addCleanup(p.stop)

    def _settings(self):
        with open(self.settings_file, encoding="utf-8") as f:
            return json.load(f)

    def test_on_sets_the_flag_config_settings_and_hud(self):
        bc = self.bc
        msg = bc._act_guest_mode_set(True)
        self.assertTrue(bc._guest_mode.is_on())
        self.assertIs(self.cfg.GUEST_MODE, True)
        self.assertIs(self._settings()["GUEST_MODE"], True)
        self.assertIn({"guest_mode": True}, self.hud)
        self.assertIn("guest mode on", msg.lower())
        self.assertNotIn("couldn't save", msg)

    def test_off_clears_all_of_them(self):
        bc = self.bc
        bc._act_guest_mode_set(True)
        msg = bc._act_guest_mode_set(False)
        self.assertFalse(bc._guest_mode.is_on())
        self.assertIs(self.cfg.GUEST_MODE, False)
        self.assertIs(self._settings()["GUEST_MODE"], False)
        self.assertEqual(self.hud[-1], {"guest_mode": False})
        self.assertIn("guest mode off", msg.lower())

    def test_it_survives_a_restart_until_turned_off(self):
        bc = self.bc
        with open(self.settings_file, "w", encoding="utf-8") as f:
            json.dump({"SOME_OTHER_KEY": "keep"}, f)
        bc._act_guest_mode_set(True)
        self.assertEqual(self._settings(),
                         {"SOME_OTHER_KEY": "keep", "GUEST_MODE": True})
        # "Restart": a fresh process has the flag off and config re-read
        # from the file (core.config._apply_user_settings, pinned in
        # tests/test_fact_provenance.py); the boot seed turns it back on.
        bc._guest_mode.set_on(False)
        self.cfg.GUEST_MODE = self._settings()["GUEST_MODE"]
        self.hud.clear()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            bc._guest_mode_boot()
        self.assertTrue(bc._guest_mode.is_on())
        self.assertEqual(self.hud, [{"guest_mode": True}])
        self.assertIn("[guest-mode] ON", out.getvalue())
        # Off is saved too, so the next boot stays off.
        bc._act_guest_mode_set(False)
        self.cfg.GUEST_MODE = self._settings()["GUEST_MODE"]
        bc._guest_mode_boot()
        self.assertFalse(bc._guest_mode.is_on())

    def test_a_failed_save_still_flips_and_says_so(self):
        from tools import settings_window as sw
        with mock.patch.object(sw, "save_settings",
                               side_effect=OSError("disk full")):
            msg = self.bc._act_guest_mode_set(True)
        self.assertTrue(self.bc._guest_mode.is_on())
        self.assertIn("couldn't save", msg)

    def test_status(self):
        bc = self.bc
        self.assertIn("off", bc._act_guest_mode_status())
        bc._guest_mode.set_on(True)
        self.assertIn("Guest mode is on", bc._act_guest_mode_status())

    def test_the_actions_are_registered(self):
        bc = self.bc
        for name in ("guest_mode_on", "guest_mode_off", "guest_mode_status"):
            self.assertIn(name, bc.ACTIONS)
        bc.ACTIONS["guest_mode_on"]("")
        self.assertTrue(bc._guest_mode.is_on())
        bc.ACTIONS["guest_mode_off"]("")
        self.assertFalse(bc._guest_mode.is_on())
        self.assertIn("guest_mode_status", bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        # The on/off flips are side effects: the reply confirms them.
        for name in ("guest_mode_on", "guest_mode_off"):
            self.assertNotIn(name, bc.SPEAK_RESULT_VERBATIM_ACTIONS)

    def test_visitors_voices_pass_the_media_voice_gate(self):
        # The older wake-listener "guest mode" (voice-gate bypass) and the
        # owner's guest mode are one switch: either lets visitors talk.
        bc = self.bc
        with mock.patch.object(bc, "GUEST_MODE_ENABLED", False), \
             mock.patch.dict(bc.sys.modules, {"skill_wake_listener": None}):
            self.assertFalse(bc._media_guest_mode())
            with contextlib.redirect_stdout(io.StringIO()):
                bc._act_guest_mode_set(True)
            self.assertTrue(bc._media_guest_mode())

    def test_a_restart_keeps_the_memory_half_but_closes_the_voice_gates(self):
        """The voice-ID bypass was always per boot (the wake listener's
        GUEST_MODE_ENABLED resets on every restart). Guest mode's memory half
        survives a restart; visitors' voices pass the gates again only after
        a live "guest mode on" in that run (review 2026-10-02)."""
        bc = self.bc
        with mock.patch.object(bc, "GUEST_MODE_ENABLED", False), \
             mock.patch.dict(bc.sys.modules, {"skill_wake_listener": None}), \
             contextlib.redirect_stdout(io.StringIO()):
            bc._act_guest_mode_set(True)
            self.assertTrue(bc._media_guest_mode())
            # "Restart": a fresh flag, re-seeded from the saved setting.
            bc._guest_mode.set_on(False)
            self.cfg.GUEST_MODE = True
            bc._guest_mode_boot()
            self.assertTrue(bc._guest_mode.is_on())    # still remembers nothing
            self.assertFalse(bc._media_guest_mode())   # but the gates are shut
            bc._act_guest_mode_set(True)               # said again this run
            self.assertTrue(bc._media_guest_mode())
            bc._act_guest_mode_set(False)
            self.assertFalse(bc._media_guest_mode())

    def _history(self, *lines):
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
                for i, t in enumerate(lines)]

    def test_after_the_guests_leave_no_summary_holds_their_turns(self):
        """Checkpoints run every 10 minutes, so a short visit may see none
        while guest mode is on. Turning it off must still keep the guests'
        turns out of every later summary (review 2026-10-02)."""
        bc = self.bc
        hist = bc.conversation_history
        hist[:] = self._history("how is the printer", "The printer is idle.")
        bc._session_running_summary[0] = "Checked the printer."
        bc._session_summary_marker[0] = hist[-1]
        with contextlib.redirect_stdout(io.StringIO()):
            bc._act_guest_mode_set(True)
            hist.extend(self._history("Orlo here, I collect stamps",
                                      "Nice to meet you, Orlo."))
            bc._act_guest_mode_set(False)
        hist.extend(self._history("owner line after", "owner reply after"))
        with mock.patch.object(bc, "_llm_quick",
                               return_value="Printer, then more.") as llm, \
             mock.patch.object(bc.pattern_memory, "record_session_summary"):
            bc._session_summary_update(bc._session_summary_pending())
        sent = llm.call_args.kwargs.get("user", "")
        self.assertIn("owner line after", sent)
        self.assertNotIn("Orlo", sent)

    def test_a_snapshot_taken_in_guest_mode_is_never_summarised(self):
        """The checkpoint snapshots, then waits (bounded) for a quiet moment:
        if the guests leave meanwhile, their turns must not go to the model
        or into the summary."""
        bc = self.bc
        hist = bc.conversation_history
        bc._session_running_summary[0] = "Checked the printer."
        with contextlib.redirect_stdout(io.StringIO()):
            bc._act_guest_mode_set(True)
            hist[:] = self._history("Orlo here", "Hello, Orlo.",
                                    "I collect stamps", "Splendid.")
            pending = bc._session_summary_pending()
            self.assertIsNotNone(pending)
            bc._act_guest_mode_set(False)
        with mock.patch.object(bc, "_llm_quick",
                               return_value="Orlo collects stamps.") as llm, \
             mock.patch.object(bc.pattern_memory,
                               "record_session_summary") as rec:
            self.assertEqual(bc._session_summary_update(pending), "")
        llm.assert_not_called()
        rec.assert_not_called()
        self.assertEqual(bc._session_running_summary[0],
                         "Checked the printer.")

    def test_the_dashboard_buttons_go_through_the_tray_plane(self):
        bc = self.bc
        with mock.patch.object(bc, "_publish_tray_result") as pub, \
             contextlib.redirect_stdout(io.StringIO()):
            bc._dispatch_tray_command("guest_mode_on",
                                      {"cmd": "guest_mode_on", "rid": "r9"})
            self.assertTrue(bc._guest_mode.is_on())
            bc._dispatch_tray_command("guest_mode_off",
                                      {"cmd": "guest_mode_off"})
        self.assertFalse(bc._guest_mode.is_on())
        pub.assert_any_call("r9", "guest_mode_on", mock.ANY)

    def test_the_model_saying_it_without_the_token_runs_it(self):
        bc = self.bc
        got = bc._detect_preemptive_hallucination(
            "Entering guest mode, sir. I won't remember a thing.")
        self.assertEqual(got[:2], ("inject", "guest_mode_on"))
        got = bc._detect_preemptive_hallucination(
            "Turning off guest mode, sir.")
        self.assertEqual(got[:2], ("inject", "guest_mode_off"))
        got = bc._detect_preemptive_hallucination(
            "I'll greet the guests as they arrive, sir.")
        self.assertNotIn(got and got[1], ("guest_mode_on", "guest_mode_off"))

    def test_boot_seeds_before_the_skills_load(self):
        """main() re-applies the saved mode BEFORE load_skills(): the ambient
        daemons a skill starts must see it from their first write."""
        with open(self.bc.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [n.func.id for n in ast.walk(main)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("_guest_mode_boot", calls)
        lines = {}
        for n in ast.walk(main):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in ("_guest_mode_boot", "load_skills")):
                lines[n.func.id] = min(lines.get(n.func.id, n.lineno),
                                       n.lineno)
        self.assertLess(lines["_guest_mode_boot"], lines["load_skills"])


class GuestModeWritesNothingTests(_MemoryStoreBase):
    def setUp(self):
        super().setUp()
        self.bc._guest_mode.set_on(True)

    def test_merge_memory_keeps_nothing_and_logs_counts_only(self):
        bc = self.bc
        with mock.patch("builtins.print") as p:
            out = bc.merge_memory(new_facts=["User's guest Orlo likes rum"],
                                  new_projects=["Orlo's boat"],
                                  new_topic="rum tasting",
                                  provenance={"owner_directed": True})
        self.assertEqual(out, ([], []))
        self.assertEqual(self.saved, [])
        self.assertEqual(self.mirrored, [])
        logged = _printed(p)
        self.assertIn("[guest-mode] 3 memory item(s) not kept", logged)
        self.assertNotIn("Orlo", logged)

    def test_learn_from_turn_queues_nothing(self):
        bc = self.bc
        for gate_on in (False, True):
            with self.subTest(gate_on=gate_on), \
                 mock.patch.object(bc, "LEARN_ONLY_FROM_OWNER", gate_on), \
                 mock.patch.object(bc, "LEARN_EVERY_TURN", True), \
                 mock.patch.object(bc, "_dialogue_gate_active",
                                   lambda: False), \
                 mock.patch.object(bc, "_learn_enqueue") as enq, \
                 mock.patch.object(bc, "_learn_gate_submit") as sub:
                bc.learn_from_turn("my name is Orlo", "Hello, Orlo.", {})
            enq.assert_not_called()
            sub.assert_not_called()

    def test_no_turn_reaches_the_episode_log(self):
        bc = self.bc
        with mock.patch.object(bc, "queue") as q, \
             mock.patch.object(bc.threading, "Thread") as th:
            bc._ltm_enqueue("user", "my name is Orlo")
        q.Queue.assert_not_called()
        th.assert_not_called()
        self.assertTrue(bc._ltm_queue is None
                        or bc._ltm_queue.qsize() == 0)

    def test_the_ambient_learner_skips_before_its_llm_judge(self):
        bc = self.bc
        with mock.patch.object(bc, "AMBIENT_LISTEN_ENABLED", True), \
             mock.patch.object(bc, "_dialogue_gate_active", lambda: False), \
             mock.patch.object(bc, "_ambient_media_is_playing") as media, \
             mock.patch.object(bc, "_ambient_content_is_media") as judge, \
             mock.patch.object(bc, "learn_from_turn") as lft:
            bc._ambient_learn_from_gated("Orlo said the boat is moored", {})
        media.assert_not_called()
        judge.assert_not_called()
        lft.assert_not_called()

    def test_the_ambient_feed_writes_no_transcript(self):
        bc = self.bc
        with mock.patch("builtins.open",
                        side_effect=AssertionError("opened a file")), \
             mock.patch.object(bc.os, "makedirs") as mk:
            bc._ambient_learning_feed("Orlo said the boat is moored nearby")
        mk.assert_not_called()

    def test_the_session_summary_skips_the_guests_conversation(self):
        bc = self.bc
        hist = bc.conversation_history
        hist[:] = [{"role": "user", "content": f"guest line {i}"}
                   if i % 2 == 0 else
                   {"role": "assistant", "content": f"reply {i}"}
                   for i in range(6)]
        bc._session_running_summary[0] = "Morning: fixed the printer."
        pending = bc._session_summary_pending()
        with mock.patch.object(bc, "_llm_quick") as llm, \
             mock.patch.object(bc.pattern_memory,
                               "record_session_summary") as rec:
            self.assertEqual(bc._session_summary_update(pending), "")
            llm.assert_not_called()
            rec.assert_not_called()
            self.assertEqual(bc._session_running_summary[0],
                             "Morning: fixed the printer.")
            # After the guests leave, only the NEW turns are summarised.
            bc._guest_mode.set_on(False)
            hist.append({"role": "user", "content": "owner line"})
            hist.append({"role": "assistant", "content": "owner reply"})
            llm.return_value = "Fixed the printer, then the owner line."
            bc._session_summary_update(bc._session_summary_pending())
        sent = llm.call_args.kwargs.get("user", "")
        self.assertIn("owner line", sent)
        self.assertNotIn("guest line", sent)

    def test_the_shutdown_summary_writes_nothing(self):
        bc = self.bc
        bc._session_running_summary[0] = "Morning: fixed the printer."
        with mock.patch.object(bc.pattern_memory,
                               "record_session_summary") as rec, \
             contextlib.redirect_stdout(io.StringIO()):
            bc.save_session_to_memory(bc._empty_memory())
        self.assertEqual(self.saved, [])
        rec.assert_not_called()

    def test_learning_resumes_when_it_is_turned_off(self):
        bc = self.bc
        bc._guest_mode.set_on(False)
        added, _ = bc.merge_memory(new_facts=["User owns a canoe"])
        self.assertEqual(added, ["User owns a canoe"])
        self.assertEqual(len(self.saved), 1)


if __name__ == "__main__":
    unittest.main()
