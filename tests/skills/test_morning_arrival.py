"""Logic tests for skills/morning_arrival.py.

Covers the data-section formatters (time, weather C→F, first-meeting phrasing),
the "Three things require your attention" clause builder, full briefing
composition, the TTS-budget estimator + progressive section dropping, the
overnight-merge log parser, overnight-anomaly extraction, the silence-gate
helper, same-day suppression, the chain entry's silence gate, and the manual
action. ThreadPoolExecutor sections are mocked away; no network/hardware.

The second half of the file (the *Extended* test classes) pushes line+branch
coverage of every remaining section formatter and orchestration path:
  • _import_skill resolution order (live skill_<name> > skills.<name> > flat),
  • _enqueue_speech (proactive_announce funnel + disk fallback),
  • _load_state / _save_state / _mark_fired round-trip on a temp file,
  • _silence_hours_since_last_speech pre-wake vs last_speech_time fallbacks,
  • the Teams unread callout (1 / 2 / 3 senders, vision-top-3 vs nudge fallback),
  • _teams_top_senders_via_vision parsing + every early-out,
  • _section_print_phrase (finish / failed / running / pause / stale / disk),
  • _section_top_headline (config-gated, summarised vs raw),
  • _section_first_meeting_phrase organizer/subject/PM/minute branches,
  • _gather_sections parallel collection + per-section timeout/crash handling,
  • _compose_within_budget trimming, _build_briefing, and the full fire path.

Isolation contract (wave-1/2 lessons): all fakes live inside ``with`` blocks
or are torn down via addCleanup; no module-level sys.modules writes survive a
test; wall-clock-sensitive paths freeze the skill's ``time`` module; fixtures
use generic names only.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _rmtree(path):
    """Best-effort recursive cleanup of a temp dir created by a test."""
    try:
        for fn in os.listdir(path):
            try:
                os.unlink(os.path.join(path, fn))
            except OSError:
                pass
        os.rmdir(path)
    except OSError:
        pass


class MorningArrivalTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_arrival")

    # ── _section_time_phrase (pure) ──────────────────────────────────────
    def test_section_time_phrase_format(self):
        out = self.mod._section_time_phrase()
        self.assertRegex(out, r"^\d{1,2}:\d{2} (AM|PM)$")

    # ── _section_weather_phrase (C→F via briefing_sources) ───────────────
    def test_weather_phrase_converts_to_fahrenheit(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"temp_c": 18, "desc": "Clear"}  # 18C → 64F
        with mock.patch.object(self.mod, "_import_skill", return_value=bs):
            out = self.mod._section_weather_phrase()
        self.assertEqual(out, "64 degrees and clear")

    def test_weather_phrase_degraded(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_weather_phrase(), "")

    # ── _section_first_meeting_phrase ────────────────────────────────────
    def test_first_meeting_on_the_hour(self):
        ms = mock.MagicMock()
        ms.get_first_meeting.return_value = {
            "start": datetime.datetime(2026, 6, 1, 10, 0),
            "subject": "Sam sync", "organizer": "Alex Morgan",
        }
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_first_meeting_phrase()
        # On-the-hour morning meeting drops ":00" → "Sam sync at 10".
        self.assertEqual(out, "Sam sync at 10")

    def test_first_meeting_none(self):
        ms = mock.MagicMock()
        ms.get_first_meeting.return_value = None
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(), "")

    # ── _attention_clause (pure) ─────────────────────────────────────────
    def test_attention_clause_counts(self):
        a = self.mod._attention_clause
        self.assertEqual(a([]), "")
        self.assertIn("One item requires your attention: x.", a(["x"]))
        self.assertIn("Two things require your attention: x, and y.", a(["x", "y"]))
        three = a(["x", "y", "z"])
        self.assertIn("Three things require your attention", three)
        self.assertIn("x, y, and z.", three)

    # ── _compose_briefing (pure) ─────────────────────────────────────────
    def test_compose_briefing_groups_attention_items(self):
        parts = {"time": "7:42 AM", "weather": "64 degrees and clear",
                 "teams": "one new Teams chat from Sam",
                 "print": "the H2D finished your bracket print",
                 "anomalies": "", "meeting": "a sync at 10",
                 "claude": "Claude Code merged 3 improvements", "headline": "Markets up"}
        out = self.mod._compose_briefing(parts)
        self.assertTrue(out.startswith("[intent:briefing] Good morning, sir."))
        self.assertIn("It's 7:42 AM, 64 degrees and clear.", out)
        self.assertIn("Two things require your attention", out)
        self.assertIn("You have a sync at 10.", out)
        self.assertIn("Overnight, Claude Code merged 3 improvements.", out)
        self.assertIn("In the news, Markets up.", out)

    def test_compose_briefing_minimal(self):
        parts = {k: "" for k in ("time", "weather", "meeting", "claude",
                                 "print", "teams", "anomalies", "headline")}
        out = self.mod._compose_briefing(parts)
        self.assertEqual(out, "[intent:briefing] Good morning, sir.")

    # ── _estimate_tts_seconds + _compose_within_budget ───────────────────
    def test_estimate_tts_strips_intent_tag(self):
        # 30 chars of body at 15 chars/s → 2.0 s. Tag not counted.
        secs = self.mod._estimate_tts_seconds("[intent:briefing] " + ("x" * 30))
        self.assertAlmostEqual(secs, 2.0, places=3)

    def test_compose_within_budget_drops_low_priority(self):
        # Force a tiny budget so everything droppable gets dropped, but the
        # three attention items (teams/print/anomalies) must survive.
        long = "x" * 200
        parts = {"time": "7 AM", "weather": long, "meeting": long,
                 "claude": long, "print": "print done", "teams": "teams ping",
                 "anomalies": "gpu anomaly", "headline": long}
        with mock.patch.object(self.mod, "TTS_BUDGET_SECONDS", 8.0):
            out = self.mod._compose_within_budget(parts)
        # Attention spine retained.
        self.assertIn("Three things require your attention", out)
        self.assertIn("teams ping", out)
        self.assertIn("print done", out)
        # Droppable long sections gone.
        self.assertNotIn("In the news", out)

    # ── _count_overnight_merges (log parsing) ────────────────────────────
    def test_count_overnight_merges_counts_decrements(self):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")  # in-window (now)
        log = (
            f"=== upgrade loop started {ts} ===\n"
            "[loop] iteration 1 of 5 — 5 task(s) remain\n"
            "[loop] iteration 2 of 5 — 3 task(s) remain\n"   # -2
            "[loop] iteration 3 of 5 — 2 task(s) remain\n"   # -1
        )
        m = mock.mock_open(read_data=log.encode())
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os.path, "getsize", return_value=len(log)), \
             mock.patch("builtins.open", m):
            n = self.mod._count_overnight_merges()
        self.assertEqual(n, 3)   # (5→3) + (3→2)

    def test_count_overnight_merges_missing_log(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._count_overnight_merges(), 0)

    def test_section_claude_code_phrase(self):
        with mock.patch.object(self.mod, "_count_overnight_merges", return_value=3):
            self.assertEqual(self.mod._section_claude_code_phrase(),
                             "Claude Code merged 3 improvements")
        with mock.patch.object(self.mod, "_count_overnight_merges", return_value=1):
            self.assertEqual(self.mod._section_claude_code_phrase(),
                             "Claude Code merged one improvement")
        with mock.patch.object(self.mod, "_count_overnight_merges", return_value=0):
            self.assertEqual(self.mod._section_claude_code_phrase(), "")

    # ── _section_overnight_anomalies ─────────────────────────────────────
    def test_overnight_anomalies_gpu_and_disk(self):
        history = [{"ts": time.time(),
                    "probes": {"gpu": {"ok": False, "severity": "HIGH", "error": "ECC"},
                               "disk": {"ok": False, "severity": "MEDIUM", "error": "SMART"}}}]
        import json
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(history))):
            out = self.mod._section_overnight_anomalies()
        self.assertEqual(out, "GPU and disk anomalies overnight")

    def test_overnight_anomalies_skips_low_severity(self):
        history = [{"ts": time.time(),
                    "probes": {"gpu": {"ok": False, "severity": "LOW", "error": "blip"}}}]
        import json
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(history))):
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    def test_overnight_anomalies_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    # ── _silence_hours_since_last_speech + suppression ───────────────────
    def test_silence_hours_prefers_pre_wake(self):
        fake_bc = mock.MagicMock()
        fake_bc._pre_wake_silence_seconds = [8 * 3600]  # 8h
        import sys
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            self.assertAlmostEqual(self.mod._silence_hours_since_last_speech(), 8.0, places=2)

    def test_already_fired_today(self):
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod, "_load_state", return_value={"last_fired_date": today}):
            self.assertTrue(self.mod._arrival_already_fired_today())
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            self.assertFalse(self.mod._arrival_already_fired_today())

    # ── _fire_from_chain silence gate ────────────────────────────────────
    def test_fire_from_chain_blocked_by_short_silence(self):
        with mock.patch.object(self.mod, "_arrival_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_silence_hours_since_last_speech", return_value=2.0), \
             mock.patch.object(self.mod, "_fire_arrival") as fire:
            out = self.mod._fire_from_chain("chain")
        self.assertEqual(out, "")
        fire.assert_not_called()

    def test_fire_from_chain_fires_when_silent_enough(self):
        with mock.patch.object(self.mod, "_arrival_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_silence_hours_since_last_speech", return_value=9.0), \
             mock.patch.object(self.mod.time, "sleep"), \
             mock.patch.object(self.mod, "_fire_arrival", return_value="briefing!") as fire:
            out = self.mod._fire_from_chain("chain")
        self.assertEqual(out, "briefing!")
        fire.assert_called_once()

    def test_fire_from_chain_degrades_open_when_silence_unknown(self):
        # silence_h None (bobert unreachable) → fire anyway.
        with mock.patch.object(self.mod, "_arrival_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_silence_hours_since_last_speech", return_value=None), \
             mock.patch.object(self.mod.time, "sleep"), \
             mock.patch.object(self.mod, "_fire_arrival", return_value="briefing!") as fire:
            out = self.mod._fire_from_chain("chain")
        self.assertEqual(out, "briefing!")
        fire.assert_called_once()

    # ── morning_arrival action ───────────────────────────────────────────
    def test_action_returns_built_text(self):
        mod, actions = load_skill_isolated("morning_arrival")
        with mock.patch.object(mod, "_build_briefing", return_value="[intent:briefing] Good morning."), \
             mock.patch.object(mod, "_enqueue_speech"), \
             mock.patch.object(mod, "_mark_fired"):
            out = actions["morning_arrival"]("")
        self.assertIn("Good morning", out)

    def test_action_no_content_message(self):
        mod, actions = load_skill_isolated("morning_arrival")
        with mock.patch.object(mod, "_build_briefing", return_value=""), \
             mock.patch.object(mod, "_enqueue_speech"), \
             mock.patch.object(mod, "_mark_fired"):
            out = actions["morning_arrival"]("")
        self.assertIn("no content", out.lower())


# ─────────────────────────────────────────────────────────────────────────
# Extended coverage: import resolution, speech enqueue, and state I/O.
# ─────────────────────────────────────────────────────────────────────────
class ImportAndEnqueueTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    # ── _import_skill resolution order ───────────────────────────────────
    def test_import_skill_prefers_live_registered(self):
        live = types.ModuleType("skill_demo_src")
        with mock.patch.dict(sys.modules, {"skill_demo_src": live}):
            self.assertIs(self.mod._import_skill("demo_src"), live)

    def test_import_skill_falls_back_to_skills_package(self):
        pkg_mod = types.ModuleType("skills.demo_src")
        # No live skill_demo_src registered → tries importlib.import_module.
        with mock.patch.dict(sys.modules, {}, clear=False), \
             mock.patch.object(self.mod.importlib, "import_module",
                               return_value=pkg_mod) as imp:
            sys.modules.pop("skill_demo_src", None)
            out = self.mod._import_skill("demo_src")
        self.assertIs(out, pkg_mod)
        imp.assert_called_once_with("skills.demo_src")

    def test_import_skill_returns_none_when_unresolvable(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("nope")):
            sys.modules.pop("skill_does_not_exist", None)
            self.assertIsNone(self.mod._import_skill("does_not_exist"))

    # ── _enqueue_speech ──────────────────────────────────────────────────
    def test_enqueue_speech_uses_proactive_announce(self):
        bc = types.SimpleNamespace(proactive_announce=mock.MagicMock())
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.mod._enqueue_speech("hello sir")
        bc.proactive_announce.assert_called_once_with("hello sir", source="arrival")

    def test_enqueue_speech_falls_back_to_disk_when_no_announcer(self):
        # bobert_companion importable but without a callable proactive_announce →
        # the direct atomic-write fallback runs. Patch the writer so nothing
        # touches the real pending_speech.json.
        bc = types.SimpleNamespace(proactive_announce=None)
        captured = {}

        def _fake_write(path, data):
            captured["path"] = path
            captured["data"] = data

        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json", _fake_write):
            self.mod._enqueue_speech("disk path message")
        self.assertEqual(captured["path"], self.mod._SPEECH_QUEUE)
        self.assertEqual(captured["data"][-1]["message"], "disk path message")

    def test_enqueue_speech_appends_to_existing_queue(self):
        # Existing queue on a real temp file is read, the new message appended.
        # A real file is used (not mock_open) so the source's open()/json.load
        # round-trip behaves exactly as in production.
        bc = types.SimpleNamespace(proactive_announce=None)
        tmp = tempfile.mkdtemp(prefix="arrival_q_")
        self.addCleanup(lambda: _rmtree(tmp))
        queue = os.path.join(tmp, "pending_speech.json")
        with open(queue, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        captured = {}
        with mock.patch.object(self.mod, "_SPEECH_QUEUE", queue), \
             mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d: captured.update(data=d)):
            self.mod._enqueue_speech("new")
        msgs = [d["message"] for d in captured["data"]]
        self.assertEqual(msgs, ["old", "new"])

    def test_enqueue_speech_disk_fallback_when_bobert_import_raises(self):
        # bobert_companion import itself raising is swallowed; the direct
        # atomic-write fallback still queues the message to disk.
        captured = {}
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bobert yet")), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d: captured.update(data=d)):
            self.mod._enqueue_speech("early boot message")
        self.assertEqual(captured["data"][-1]["message"], "early boot message")

    def test_enqueue_speech_swallows_write_failure(self):
        # A failing atomic write must not raise out of the briefing path.
        bc = types.SimpleNamespace(proactive_announce=None)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.mod._enqueue_speech("boom")  # must not raise

    def test_enqueue_speech_recovers_from_corrupt_queue(self):
        # Unparseable JSON on disk → treated as empty list, message still queued.
        bc = types.SimpleNamespace(proactive_announce=None)
        tmp = tempfile.mkdtemp(prefix="arrival_q_")
        self.addCleanup(lambda: _rmtree(tmp))
        queue = os.path.join(tmp, "pending_speech.json")
        with open(queue, "w", encoding="utf-8") as f:
            f.write("{not json")
        captured = {}
        with mock.patch.object(self.mod, "_SPEECH_QUEUE", queue), \
             mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d: captured.update(data=d)):
            self.mod._enqueue_speech("fresh")
        self.assertEqual(captured["data"][-1]["message"], "fresh")


class StateFileTests(unittest.TestCase):
    """_load_state / _save_state / _mark_fired round-trip against a real temp
    file so the atomic-write path executes without touching the project's
    morning_arrival_state.json."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")
        self.tmp = tempfile.mkdtemp(prefix="arrival_state_")
        self.state_file = os.path.join(self.tmp, "morning_arrival_state.json")
        self._orig_state_file = self.mod._STATE_FILE
        self.mod._STATE_FILE = self.state_file
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        self.mod._STATE_FILE = self._orig_state_file
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def test_load_state_missing_returns_empty(self):
        self.assertEqual(self.mod._load_state(), {})

    def test_save_then_load_roundtrip(self):
        self.mod._save_state({"last_fired_date": "2026-06-01", "x": 1})
        self.assertEqual(self.mod._load_state(),
                         {"last_fired_date": "2026-06-01", "x": 1})

    def test_load_state_corrupt_returns_empty(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write("{ broken json")
        self.assertEqual(self.mod._load_state(), {})

    def test_load_state_null_payload_returns_empty(self):
        # json.load → None (the literal `null`) must collapse to {}.
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write("null")
        self.assertEqual(self.mod._load_state(), {})

    def test_mark_fired_records_date_reason_ts(self):
        with mock.patch.object(self.mod.time, "strftime", return_value="2026-06-01"), \
             mock.patch.object(self.mod.time, "time", return_value=123.0):
            self.mod._mark_fired("manual trigger")
        state = self.mod._load_state()
        self.assertEqual(state["last_fired_date"], "2026-06-01")
        self.assertEqual(state["last_reason"], "manual trigger")
        self.assertEqual(state["last_fired_ts"], 123.0)

    def test_save_state_swallows_write_error(self):
        # A failing writer is logged, not raised.
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("nope")):
            self.mod._save_state({"a": 1})  # must not raise


# ─────────────────────────────────────────────────────────────────────────
# Extended coverage: the silence-gate helper's fallback ladder.
# ─────────────────────────────────────────────────────────────────────────
class SilenceHoursTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    def test_returns_none_when_bobert_unreachable(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.assertIsNone(self.mod._silence_hours_since_last_speech())

    def test_falls_back_to_last_speech_time(self):
        # No usable _pre_wake_silence_seconds → compute from last_speech_time.
        fake_bc = types.SimpleNamespace(
            _pre_wake_silence_seconds=None,
            last_speech_time=1000.0)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=fake_bc), \
             mock.patch.object(self.mod.time, "time", return_value=1000.0 + 3 * 3600):
            self.assertAlmostEqual(self.mod._silence_hours_since_last_speech(),
                                   3.0, places=3)

    def test_pre_wake_zero_falls_through_to_last_speech(self):
        # pre_wake present but <= 0 → ignored, last_speech_time used instead.
        fake_bc = types.SimpleNamespace(
            _pre_wake_silence_seconds=[0],
            last_speech_time=2000.0)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=fake_bc), \
             mock.patch.object(self.mod.time, "time", return_value=2000.0 + 7200):
            self.assertAlmostEqual(self.mod._silence_hours_since_last_speech(),
                                   2.0, places=3)

    def test_pre_wake_non_numeric_falls_through(self):
        # A bad pre_wake value raises in float() → swallowed, last_speech used.
        fake_bc = types.SimpleNamespace(
            _pre_wake_silence_seconds=["not-a-number"],
            last_speech_time=3000.0)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=fake_bc), \
             mock.patch.object(self.mod.time, "time", return_value=3000.0 + 3600):
            self.assertAlmostEqual(self.mod._silence_hours_since_last_speech(),
                                   1.0, places=3)

    def test_returns_none_when_no_speech_recorded(self):
        # Neither signal usable (last_speech_time <= 0) → None ("can't decide").
        fake_bc = types.SimpleNamespace(
            _pre_wake_silence_seconds=[],
            last_speech_time=0.0)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=fake_bc):
            self.assertIsNone(self.mod._silence_hours_since_last_speech())


# ─────────────────────────────────────────────────────────────────────────
# Extended coverage: the Teams unread callout phrasing + vision parser.
# ─────────────────────────────────────────────────────────────────────────
class TeamsPhraseTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    # ── _section_teams_phrase via the vision top-3 path ──────────────────
    def test_one_sender(self):
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=["Alice"]):
            self.assertEqual(self.mod._section_teams_phrase(),
                             "one new Teams chat from Alice")

    def test_two_senders(self):
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=["Alice", "Sam"]):
            self.assertEqual(self.mod._section_teams_phrase(),
                             "two new Teams chats from Alice and Sam")

    def test_three_senders(self):
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=["Alice", "Sam", "Pat"]):
            self.assertEqual(
                self.mod._section_teams_phrase(),
                "three new Teams chats from Alice, Sam, and Pat")

    # ── _section_teams_phrase nudge fallback (vision found nothing) ───────
    def test_fallback_no_nudge_skill(self):
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=[]), \
             mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_teams_phrase(), "")

    def test_fallback_single_unread_with_sender(self):
        tn = mock.MagicMock()
        tn._ask_vision_for_teams_state.return_value = (True, 1, "Sam")
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=[]), \
             mock.patch.object(self.mod, "_import_skill", return_value=tn):
            self.assertEqual(self.mod._section_teams_phrase(),
                             "one unread Teams message from Sam")

    def test_fallback_single_unread_no_sender(self):
        tn = mock.MagicMock()
        tn._ask_vision_for_teams_state.return_value = (True, 1, "")
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=[]), \
             mock.patch.object(self.mod, "_import_skill", return_value=tn):
            self.assertEqual(self.mod._section_teams_phrase(),
                             "one unread Teams message")

    def test_fallback_multi_unread_with_sender(self):
        tn = mock.MagicMock()
        tn._ask_vision_for_teams_state.return_value = (True, 4, "Sam")
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=[]), \
             mock.patch.object(self.mod, "_import_skill", return_value=tn):
            self.assertEqual(
                self.mod._section_teams_phrase(),
                "4 unread Teams messages including one from Sam")

    def test_fallback_multi_unread_no_sender(self):
        tn = mock.MagicMock()
        tn._ask_vision_for_teams_state.return_value = (True, 3, "")
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=[]), \
             mock.patch.object(self.mod, "_import_skill", return_value=tn):
            self.assertEqual(self.mod._section_teams_phrase(),
                             "3 unread Teams messages")

    def test_fallback_nothing_unread(self):
        tn = mock.MagicMock()
        tn._ask_vision_for_teams_state.return_value = (False, 0, "")
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=[]), \
             mock.patch.object(self.mod, "_import_skill", return_value=tn):
            self.assertEqual(self.mod._section_teams_phrase(), "")

    def test_fallback_vision_lookup_raises(self):
        tn = mock.MagicMock()
        tn._ask_vision_for_teams_state.side_effect = RuntimeError("vision down")
        with mock.patch.object(self.mod, "_teams_top_senders_via_vision",
                               return_value=[]), \
             mock.patch.object(self.mod, "_import_skill", return_value=tn):
            self.assertEqual(self.mod._section_teams_phrase(), "")


class TeamsVisionParserTests(unittest.TestCase):
    """_teams_top_senders_via_vision: drive bobert_companion's screenshot +
    vision plumbing through a fake and assert the strict-format parser."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    def _bc(self, *, answer="", images=("img",), take=True, ask=True,
            take_raises=False, ask_raises=False):
        ns = types.SimpleNamespace()
        if take:
            def _take():
                if take_raises:
                    raise RuntimeError("screenshot boom")
                return list(images)
            ns.take_all_monitor_screenshots = _take
        if ask:
            def _ask(prompt, imgs):
                if ask_raises:
                    raise RuntimeError("vision boom")
                return answer
            ns.ask_vision_multi = _ask
        return ns

    def test_parses_three_names(self):
        bc = self._bc(answer="TOP: Alice | Sam | Pat")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(),
                             ["Alice", "Sam", "Pat"])

    def test_parses_single_name(self):
        bc = self._bc(answer="TOP: Alice")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), ["Alice"])

    def test_caps_at_three_names(self):
        bc = self._bc(answer="TOP: A | B | C | D | E")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(),
                             ["A", "B", "C"])

    def test_long_name_truncated_to_first_word(self):
        bc = self._bc(answer="TOP: Alexandria Bartholomew Smith")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            out = self.mod._teams_top_senders_via_vision()
        # >14 chars → collapses to the first word capped at 14 chars.
        self.assertEqual(out, ["Alexandria"])

    def test_drops_placeholder_tokens(self):
        # "NONE"/"UNKNOWN" tokens are filtered out; only the real name remains.
        bc = self._bc(answer="TOP: NONE | Sam | UNKNOWN")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), ["Sam"])

    def test_na_token_leaks_due_to_slash_strip_quirk(self):
        # NOTE (source quirk, not fixed): the name-sanitiser strips '/' BEFORE
        # the placeholder check, so "N/A" becomes "NA", which is NOT in the
        # {"NONE","N/A","UNKNOWN"} reject set — it survives as a bogus sender.
        bc = self._bc(answer="TOP: N/A | Sam")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), ["NA", "Sam"])

    def test_none_answer_returns_empty(self):
        bc = self._bc(answer="NONE")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])

    def test_unparseable_answer_returns_empty(self):
        bc = self._bc(answer="I see a Teams window with some chats")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])

    def test_non_string_answer_returns_empty(self):
        bc = self._bc(answer=12345)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])

    def test_top_marker_but_empty_payload_returns_empty(self):
        bc = self._bc(answer="TOP:")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])

    def test_empty_answer_returns_empty(self):
        bc = self._bc(answer="")
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])

    def test_no_screenshots_returns_empty(self):
        bc = self._bc(answer="TOP: Alice", images=[])
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])

    def test_missing_helpers_returns_empty(self):
        bc = self._bc(take=False, ask=False)  # neither callable present
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])

    def test_screenshot_raises_returns_empty(self):
        bc = self._bc(answer="TOP: Alice", take_raises=True)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])

    def test_vision_call_raises_returns_empty(self):
        bc = self._bc(answer="TOP: Alice", ask_raises=True)
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])

    def test_bobert_import_fails_returns_empty(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.assertEqual(self.mod._teams_top_senders_via_vision(), [])


# ─────────────────────────────────────────────────────────────────────────
# Extended coverage: the Bambu print-status section.
# ─────────────────────────────────────────────────────────────────────────
class PrintPhraseTests(unittest.TestCase):
    """_section_print_phrase reads bambu_monitor in-process state with an
    on-disk overlay fallback. ``_NOW`` is frozen so the age window is
    deterministic; the in-process bambu_monitor is mocked away so no MQTT
    poller is touched."""
    _NOW = 1_700_000_000.0

    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")
        # Freeze the skill's clock for all age math in this section.
        self._time_patch = mock.patch.object(self.mod.time, "time",
                                              return_value=self._NOW)
        self._time_patch.start()
        self.addCleanup(self._time_patch.stop)

    def _bm(self, state):
        """Fake bambu_monitor with a real lock + a _strip_filename helper."""
        import threading
        bm = types.SimpleNamespace()
        bm._state_lock = threading.Lock()
        bm._state = state
        bm._strip_filename = lambda raw: (
            raw.replace(".3mf", "").replace("_", " ").strip())
        return bm

    def test_finished_named_print(self):
        st = {"last_update": self._NOW - 100, "gcode_state": "FINISH",
              "filename": "bracket.3mf"}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D finished your bracket print")

    def test_finished_unnamed_print(self):
        st = {"last_update": self._NOW - 100, "gcode_state": "FINISH",
              "filename": ""}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D finished its overnight print")

    def test_failed_print(self):
        st = {"last_update": self._NOW - 50, "gcode_state": "FAILED",
              "filename": "thing.3mf"}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D's overnight print failed")

    def test_running_with_percent(self):
        st = {"last_update": self._NOW - 10, "gcode_state": "RUNNING",
              "mc_percent": 47}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D is mid-print at 47 percent")

    def test_running_without_percent(self):
        st = {"last_update": self._NOW - 10, "gcode_state": "PRINTING",
              "mc_percent": None}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D is mid-print")

    def test_running_bad_percent_value(self):
        st = {"last_update": self._NOW - 10, "gcode_state": "PREPARE",
              "mc_percent": "n/a"}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D is mid-print")

    def test_paused_print(self):
        st = {"last_update": self._NOW - 10, "gcode_state": "PAUSE"}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D is paused mid-print")

    def test_unknown_state_returns_empty(self):
        st = {"last_update": self._NOW - 10, "gcode_state": "IDLE"}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(), "")

    def test_stale_state_dropped(self):
        # > 36h old → misleading, dropped.
        st = {"last_update": self._NOW - 40 * 3600, "gcode_state": "FINISH",
              "filename": "old.3mf"}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(), "")

    def test_finish_older_than_overnight_window_not_announced(self):
        # FINISH but the finish is older than the overnight window (yet within
        # 36h) → neither the FINISH branch nor any other matches → "".
        st = {"last_update": self._NOW - 20 * 3600, "gcode_state": "FINISH",
              "filename": "x.3mf"}
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)):
            self.assertEqual(self.mod._section_print_phrase(), "")

    def test_no_last_update_returns_empty(self):
        st = {"gcode_state": "RUNNING", "mc_percent": 10}  # no last_update
        # Without last_update the function falls back to bambu_overlay_state.json
        # on disk — on a live box that real runtime file exists (a real finished
        # print), so stub it away to keep the test hermetic regardless of host.
        with mock.patch.object(self.mod, "_import_skill", return_value=self._bm(st)), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._section_print_phrase(), "")

    def test_disk_overlay_fallback_when_inprocess_empty(self):
        # bambu_monitor present but in-process _state empty → read overlay JSON.
        empty_bm = self._bm({})
        overlay = {"last_update": self._NOW - 100, "gcode_state": "RUNNING",
                   "mc_percent": 80, "filename": "y.3mf"}
        with mock.patch.object(self.mod, "_import_skill", return_value=empty_bm), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open",
                        mock.mock_open(read_data=json.dumps(overlay))):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D is mid-print at 80 percent")

    def test_no_bambu_monitor_uses_disk_and_regex_strip(self):
        # bm is None → filename stripped via the local regex fallback branch.
        overlay = {"last_update": self._NOW - 100, "gcode_state": "FINISH",
                   "filename": "left_bracket.gcode"}
        with mock.patch.object(self.mod, "_import_skill", return_value=None), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open",
                        mock.mock_open(read_data=json.dumps(overlay))):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D finished your left bracket print")

    def test_bm_state_lock_raises_then_disk_fallback(self):
        # Accessing bm._state under its lock raises → state stays {} → disk read.
        bad_bm = types.SimpleNamespace()

        class _BoomLock:
            def __enter__(self):
                raise RuntimeError("lock boom")

            def __exit__(self, *a):
                return False

        bad_bm._state_lock = _BoomLock()
        bad_bm._state = {"last_update": self._NOW}
        bad_bm._strip_filename = lambda raw: raw
        overlay = {"last_update": self._NOW - 100, "gcode_state": "PAUSE"}
        with mock.patch.object(self.mod, "_import_skill", return_value=bad_bm), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open",
                        mock.mock_open(read_data=json.dumps(overlay))):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D is paused mid-print")

    def test_corrupt_overlay_json_returns_empty(self):
        empty_bm = self._bm({})
        with mock.patch.object(self.mod, "_import_skill", return_value=empty_bm), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{bad")):
            self.assertEqual(self.mod._section_print_phrase(), "")

    def test_strip_filename_raises_keeps_going(self):
        # bm._strip_filename raising is swallowed; FINISH w/o name still speaks.
        bm = self._bm({"last_update": self._NOW - 100, "gcode_state": "FINISH",
                       "filename": "z.3mf"})
        bm._strip_filename = mock.MagicMock(side_effect=RuntimeError("strip boom"))
        with mock.patch.object(self.mod, "_import_skill", return_value=bm):
            self.assertEqual(self.mod._section_print_phrase(),
                             "the H2D finished its overnight print")


# ─────────────────────────────────────────────────────────────────────────
# Extended coverage: the news-headline closer.
# ─────────────────────────────────────────────────────────────────────────
class HeadlineSectionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    def _nb(self, *, enabled=True, feeds=True, summarize=False,
            headlines=None, summary="summarised line"):
        nb = mock.MagicMock()
        nb._read_config.return_value = {
            "enabled": enabled,
            "feeds": ["http://feed"] if feeds else [],
            "summarize": summarize,
        }
        nb._gather_headlines.return_value = headlines if headlines is not None else [
            {"title": "Markets rally today.", "description": "desc"}]
        nb._summarize_via_llm.return_value = summary
        return nb

    def test_no_news_skill(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_top_headline(), "")

    def test_disabled_in_config(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=self._nb(enabled=False)):
            self.assertEqual(self.mod._section_top_headline(), "")

    def test_no_feeds_configured(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=self._nb(feeds=False)):
            self.assertEqual(self.mod._section_top_headline(), "")

    def test_raw_title_when_not_summarising(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=self._nb()):
            # Trailing period is stripped by the formatter.
            self.assertEqual(self.mod._section_top_headline(), "Markets rally today")

    def test_summarised_line_when_enabled(self):
        nb = self._nb(summarize=True, summary="Stocks climb on earnings.")
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            self.assertEqual(self.mod._section_top_headline(), "Stocks climb on earnings")

    def test_summary_raises_falls_back_to_title(self):
        nb = self._nb(summarize=True)
        nb._summarize_via_llm.side_effect = RuntimeError("llm down")
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            self.assertEqual(self.mod._section_top_headline(), "Markets rally today")

    def test_no_headlines_returns_empty(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=self._nb(headlines=[])):
            self.assertEqual(self.mod._section_top_headline(), "")

    def test_config_read_raises_returns_empty(self):
        nb = mock.MagicMock()
        nb._read_config.side_effect = RuntimeError("config boom")
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            self.assertEqual(self.mod._section_top_headline(), "")

    def test_forces_count_one(self):
        nb = self._nb()
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            self.mod._section_top_headline()
        passed_cfg = nb._gather_headlines.call_args[0][0]
        self.assertEqual(passed_cfg["count"], 1)


# ─────────────────────────────────────────────────────────────────────────
# Extended coverage: weather edge branches + meeting organizer/PM/minute.
# ─────────────────────────────────────────────────────────────────────────
class WeatherEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    def test_temp_only_when_desc_blank(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"temp_c": 0, "desc": "  "}  # 0C → 32F
        with mock.patch.object(self.mod, "_import_skill", return_value=bs):
            self.assertEqual(self.mod._section_weather_phrase(), "32 degrees")

    def test_fetch_raises_returns_empty(self):
        bs = mock.MagicMock()
        bs.get_weather_data.side_effect = RuntimeError("wttr down")
        with mock.patch.object(self.mod, "_import_skill", return_value=bs):
            self.assertEqual(self.mod._section_weather_phrase(), "")

    def test_empty_payload_returns_empty(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {}
        with mock.patch.object(self.mod, "_import_skill", return_value=bs):
            self.assertEqual(self.mod._section_weather_phrase(), "")

    def test_missing_temp_key_returns_empty(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"desc": "clear"}  # no temp_c
        with mock.patch.object(self.mod, "_import_skill", return_value=bs):
            self.assertEqual(self.mod._section_weather_phrase(), "")

    def test_non_numeric_temp_returns_empty(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"temp_c": "warm", "desc": "clear"}
        with mock.patch.object(self.mod, "_import_skill", return_value=bs):
            self.assertEqual(self.mod._section_weather_phrase(), "")


class MeetingPhraseTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    def _ms(self, meeting):
        ms = mock.MagicMock()
        ms.get_first_meeting.return_value = meeting
        return ms

    def test_no_graph_skill(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_first_meeting_phrase(), "")

    def test_graph_fetch_raises(self):
        ms = mock.MagicMock()
        ms.get_first_meeting.side_effect = RuntimeError("graph down")
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(), "")

    def test_meeting_without_start_attr(self):
        # 'start' present but not a datetime (no .hour) → "".
        ms = self._ms({"start": "10:00", "subject": "Standup"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(), "")

    def test_pm_meeting_keeps_suffix_and_minutes(self):
        ms = self._ms({"start": datetime.datetime(2026, 6, 1, 14, 30),
                       "subject": "Review", "organizer": "Sam Lee"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(),
                             "Review at 2:30 PM")

    def test_organizer_used_when_subject_long(self):
        # Subject > 40 chars → falls through to "a sync with <organizer-first>".
        long_subject = "x" * 41
        ms = self._ms({"start": datetime.datetime(2026, 6, 1, 9, 0),
                       "subject": long_subject, "organizer": "Sam Lee <sam@x.io>"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(),
                             "a sync with Sam at 9")

    def test_self_organizer_dropped_falls_back_to_subject(self):
        # Organizer == JARVIS_USER_NAME → who_label dropped. With a (long)
        # subject present, the source's `elif subject` branch still uses the
        # subject verbatim — only an EMPTY subject reaches "a meeting".
        long_subject = "y" * 50
        ms = self._ms({"start": datetime.datetime(2026, 6, 1, 9, 0),
                       "subject": long_subject, "organizer": "Owner Person"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms), \
             mock.patch.dict(self.mod.os.environ, {"JARVIS_USER_NAME": "Owner Person"}):
            self.assertEqual(self.mod._section_first_meeting_phrase(),
                             f"{long_subject} at 9")

    def test_self_organizer_empty_subject_is_generic_meeting(self):
        # Self-organizer (who_label dropped via env) AND empty subject → the
        # final `else` branch → "a meeting".
        ms = self._ms({"start": datetime.datetime(2026, 6, 1, 9, 0),
                       "subject": "", "organizer": "Owner Person"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms), \
             mock.patch.dict(self.mod.os.environ, {"JARVIS_USER_NAME": "Owner Person"}):
            self.assertEqual(self.mod._section_first_meeting_phrase(),
                             "a meeting at 9")

    def test_organizer_email_only_and_empty_subject_generic(self):
        # Organizer first token contains '@' → rejected as who_label; with an
        # empty subject the result is the generic "a meeting".
        ms = self._ms({"start": datetime.datetime(2026, 6, 1, 9, 0),
                       "subject": "", "organizer": "sam@example.io"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(),
                             "a meeting at 9")

    def test_early_hour_drops_minutes_no_suffix_branch(self):
        # On-the-hour at 7 AM: disp_hour (7) < 8 so the special "drop :00 and
        # suffix" branch is skipped, but minute_part is already "" → the phrase
        # is "Early sync at 7 AM" (suffix kept, no :00).
        ms = self._ms({"start": datetime.datetime(2026, 6, 1, 7, 0),
                       "subject": "Early sync"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(),
                             "Early sync at 7 AM")

    def test_long_subject_no_organizer_uses_subject(self):
        # Long subject + no organizer → `elif subject` branch keeps the long
        # subject verbatim (NOT collapsed to "a meeting").
        long_subject = "q" * 60
        ms = self._ms({"start": datetime.datetime(2026, 6, 1, 10, 0),
                       "subject": long_subject, "organizer": ""})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(),
                             f"{long_subject} at 10")

    def test_short_subject_preferred_over_organizer(self):
        ms = self._ms({"start": datetime.datetime(2026, 6, 1, 11, 15),
                       "subject": "1:1", "organizer": "Sam Lee"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(),
                             "1:1 at 11:15 AM")


# ─────────────────────────────────────────────────────────────────────────
# Extended coverage: overnight-merge tail handling + anomaly edge branches.
# ─────────────────────────────────────────────────────────────────────────
class MergeCountTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    def test_out_of_window_loop_ignored(self):
        # An old loop-start (epoch 0) is out of window → its iterations skipped.
        log = (
            "=== upgrade loop started 1970-01-01 00:00:00 ===\n"
            "[loop] iteration 1 of 5 — 5 task(s) remain\n"
            "[loop] iteration 2 of 5 — 2 task(s) remain\n"
        )
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os.path, "getsize", return_value=len(log)), \
             mock.patch("builtins.open", mock.mock_open(read_data=log.encode())):
            self.assertEqual(self.mod._count_overnight_merges(), 0)

    def test_large_log_seeks_tail(self):
        # A real >512KB file forces the seek-to-tail + partial-leading-line skip
        # branch. We pad the head with junk so the seek lands mid-file and the
        # readline() that discards the partial line runs for real; the in-window
        # loop markers live in the final <512KB the source reads.
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        tail = (
            f"=== upgrade loop started {ts} ===\n"
            "[loop] iteration 1 of 9 — 9 task(s) remain\n"
            "[loop] iteration 2 of 9 — 7 task(s) remain\n"  # -2
        )
        tmp = tempfile.mkdtemp(prefix="arrival_log_")
        self.addCleanup(lambda: _rmtree(tmp))
        logpath = os.path.join(tmp, "upgrade_stream.log")
        with open(logpath, "wb") as f:
            f.write(b"OLD NOISE LINE that gets truncated by the tail seek\n"
                    * 20000)              # ~1MB of head padding (> 512KB cap)
            f.write(tail.encode("utf-8"))
        with mock.patch.object(self.mod, "_UPGRADE_LOG", logpath):
            n = self.mod._count_overnight_merges()
        self.assertEqual(n, 2)

    def test_read_error_returns_zero(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os.path, "getsize", return_value=10), \
             mock.patch("builtins.open", side_effect=OSError("read boom")):
            self.assertEqual(self.mod._count_overnight_merges(), 0)

    def test_unparseable_timestamp_drops_loop(self):
        # A malformed loop-start timestamp → current_in_window False, skipped.
        log = (
            "=== upgrade loop started not-a-timestamp ===\n"
            "[loop] iteration 1 of 5 — 5 task(s) remain\n"
            "[loop] iteration 2 of 5 — 1 task(s) remain\n"
        )
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os.path, "getsize", return_value=len(log)), \
             mock.patch("builtins.open", mock.mock_open(read_data=log.encode())):
            self.assertEqual(self.mod._count_overnight_merges(), 0)

    def test_regex_shaped_but_invalid_date_drops_loop(self):
        # Timestamp matches the \d{4}-\d{2}-\d{2} ... shape but is not a real
        # calendar date → strptime raises → that loop window is treated as
        # out-of-window (exercises the mktime/strptime except branch).
        log = (
            "=== upgrade loop started 2026-13-45 99:99:99 ===\n"
            "[loop] iteration 1 of 5 — 5 task(s) remain\n"
            "[loop] iteration 2 of 5 — 1 task(s) remain\n"
        )
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os.path, "getsize", return_value=len(log)), \
             mock.patch("builtins.open", mock.mock_open(read_data=log.encode())):
            self.assertEqual(self.mod._count_overnight_merges(), 0)

    def test_in_window_noise_lines_skipped(self):
        # In-window lines that are neither loop-start nor iteration markers hit
        # the `if not m: continue` skip without affecting the merge count.
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        log = (
            f"=== upgrade loop started {ts} ===\n"
            "[loop] iteration 1 of 5 — 5 task(s) remain\n"
            "some unrelated stdout chatter from a subprocess\n"   # noise
            "[loop] iteration 2 of 5 — 4 task(s) remain\n"        # -1
        )
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os.path, "getsize", return_value=len(log)), \
             mock.patch("builtins.open", mock.mock_open(read_data=log.encode())):
            self.assertEqual(self.mod._count_overnight_merges(), 1)

    def test_increase_in_remaining_not_counted(self):
        # Remaining going UP (a new batch appended) is not a merge.
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        log = (
            f"=== upgrade loop started {ts} ===\n"
            "[loop] iteration 1 of 5 — 3 task(s) remain\n"
            "[loop] iteration 2 of 5 — 5 task(s) remain\n"  # +2, ignored
            "[loop] iteration 3 of 5 — 4 task(s) remain\n"  # -1
        )
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os.path, "getsize", return_value=len(log)), \
             mock.patch("builtins.open", mock.mock_open(read_data=log.encode())):
            self.assertEqual(self.mod._count_overnight_merges(), 1)


class AnomalyEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    def _read(self, payload):
        return (mock.patch.object(self.mod.os.path, "exists", return_value=True),
                mock.patch("builtins.open",
                           mock.mock_open(read_data=json.dumps(payload))))

    def test_single_gpu_anomaly(self):
        history = [{"ts": time.time(),
                    "probes": {"gpu": {"ok": False, "severity": "HIGH",
                                       "error": "ECC errors"}}}]
        p1, p2 = self._read(history)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(),
                             "a GPU anomaly overnight")

    def test_single_disk_anomaly(self):
        history = [{"ts": time.time(),
                    "probes": {"disk": {"ok": False, "severity": "MEDIUM",
                                        "error": "SMART pre-fail"}}}]
        p1, p2 = self._read(history)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(),
                             "a disk anomaly overnight")

    def test_runs_dict_shape_with_runs_key(self):
        # {"runs": [...]} shape is accepted as well as a bare list.
        payload = {"runs": [{"ts": time.time(),
                             "probes": {"gpu": {"ok": False, "severity": "HIGH",
                                                "error": "fault"}}}]}
        p1, p2 = self._read(payload)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(),
                             "a GPU anomaly overnight")

    def test_ok_probe_not_flagged(self):
        history = [{"ts": time.time(),
                    "probes": {"gpu": {"ok": True}, "disk": {"ok": True}}}]
        p1, p2 = self._read(history)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    def test_failure_without_error_text_not_flagged(self):
        # Probe failed but no error string → not enough signal, skipped.
        history = [{"ts": time.time(),
                    "probes": {"gpu": {"ok": False, "severity": "HIGH",
                                       "error": "   "}}}]
        p1, p2 = self._read(history)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    def test_out_of_window_run_ignored(self):
        history = [{"ts": 1.0,  # ancient → before cutoff
                    "probes": {"gpu": {"ok": False, "severity": "HIGH",
                                       "error": "old"}}}]
        p1, p2 = self._read(history)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    def test_bad_ts_treated_as_zero_and_skipped(self):
        history = [{"ts": "not-a-number",
                    "probes": {"gpu": {"ok": False, "severity": "HIGH",
                                       "error": "x"}}}]
        p1, p2 = self._read(history)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    def test_non_dict_run_skipped(self):
        history = ["junk", {"ts": time.time(),
                            "probes": {"disk": {"ok": False, "severity": "HIGH",
                                                "error": "bad"}}}]
        p1, p2 = self._read(history)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(),
                             "a disk anomaly overnight")

    def test_history_not_a_list_returns_empty(self):
        # A JSON object that isn't the {"runs": ...} shape → unwraps to [] → "".
        p1, p2 = self._read({"unexpected": "shape"})
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    def test_history_scalar_json_returns_empty(self):
        # Top-level JSON that is neither dict nor list (a bare string) → the
        # `not isinstance(history, list)` guard returns "".
        p1, p2 = self._read("just a string")
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    def test_component_already_flagged_skipped_in_later_run(self):
        # gpu fails in TWO in-window runs; the second time the component is
        # already in `seen`, so the loop's `if seen.get(component): continue`
        # fires. Output is still a single GPU anomaly.
        now = time.time()
        history = [
            {"ts": now - 20,
             "probes": {"gpu": {"ok": False, "severity": "HIGH", "error": "e1"}}},
            {"ts": now - 5,
             "probes": {"gpu": {"ok": False, "severity": "HIGH", "error": "e2"}}},
        ]
        p1, p2 = self._read(history)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(),
                             "a GPU anomaly overnight")

    def test_corrupt_history_file_returns_empty(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{bad json")):
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    def test_first_run_low_then_later_high_still_flags(self):
        # An earlier LOW result must not latch the component as "seen"/clean —
        # a later in-window HIGH failure for the same component still flags.
        now = time.time()
        history = [
            {"ts": now - 10,
             "probes": {"gpu": {"ok": False, "severity": "LOW", "error": "blip"}}},
            {"ts": now,
             "probes": {"gpu": {"ok": False, "severity": "HIGH", "error": "ECC"}}},
        ]
        p1, p2 = self._read(history)
        with p1, p2:
            self.assertEqual(self.mod._section_overnight_anomalies(),
                             "a GPU anomaly overnight")


# ─────────────────────────────────────────────────────────────────────────
# Extended coverage: parallel orchestration, budget trim, and the fire path.
# ─────────────────────────────────────────────────────────────────────────
class GatherSectionsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    def test_collects_all_sections(self):
        # Patch every section formatter to a fast deterministic stub so the
        # real ThreadPoolExecutor runs but nothing touches the network.
        stubs = {
            "_section_time_phrase": "7:42 AM",
            "_section_weather_phrase": "64 degrees and clear",
            "_section_first_meeting_phrase": "a sync at 10",
            "_section_claude_code_phrase": "Claude Code merged 2 improvements",
            "_section_print_phrase": "the H2D is mid-print at 50 percent",
            "_section_teams_phrase": "one new Teams chat from Alice",
            "_section_overnight_anomalies": "a GPU anomaly overnight",
            "_section_top_headline": "Markets up",
        }
        patches = [mock.patch.object(self.mod, name, return_value=val)
                   for name, val in stubs.items()]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        out = self.mod._gather_sections()
        self.assertEqual(out["time"], "7:42 AM")
        self.assertEqual(out["weather"], "64 degrees and clear")
        self.assertEqual(out["meeting"], "a sync at 10")
        self.assertEqual(out["claude"], "Claude Code merged 2 improvements")
        self.assertEqual(out["print"], "the H2D is mid-print at 50 percent")
        self.assertEqual(out["teams"], "one new Teams chat from Alice")
        self.assertEqual(out["anomalies"], "a GPU anomaly overnight")
        self.assertEqual(out["headline"], "Markets up")

    def test_section_crash_becomes_empty(self):
        # A raising section is caught and mapped to '' without failing others.
        with mock.patch.object(self.mod, "_section_time_phrase", return_value="7 AM"), \
             mock.patch.object(self.mod, "_section_weather_phrase",
                               side_effect=RuntimeError("weather boom")), \
             mock.patch.object(self.mod, "_section_first_meeting_phrase", return_value=""), \
             mock.patch.object(self.mod, "_section_claude_code_phrase", return_value=""), \
             mock.patch.object(self.mod, "_section_print_phrase", return_value=""), \
             mock.patch.object(self.mod, "_section_teams_phrase", return_value=""), \
             mock.patch.object(self.mod, "_section_overnight_anomalies", return_value=""), \
             mock.patch.object(self.mod, "_section_top_headline", return_value=""):
            out = self.mod._gather_sections()
        self.assertEqual(out["time"], "7 AM")
        self.assertEqual(out["weather"], "")  # crashed → empty

    def test_section_timeout_becomes_empty(self):
        # Simulate a section overrun: fut.result(timeout=...) raises
        # FutureTimeoutError, which the collector swallows to ''. Patch the
        # executor's Future.result on the module's symbol path.
        real_results = {
            "_section_time_phrase": "7 AM",
            "_section_weather_phrase": "",
            "_section_first_meeting_phrase": "",
            "_section_claude_code_phrase": "",
            "_section_print_phrase": "",
            "_section_teams_phrase": "",
            "_section_overnight_anomalies": "",
            "_section_top_headline": "",
        }
        patches = [mock.patch.object(self.mod, n, return_value=v)
                   for n, v in real_results.items()]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        # Force every Future.result to raise a timeout to exercise that branch.
        with mock.patch("concurrent.futures.Future.result",
                        side_effect=self.mod.FutureTimeoutError()):
            out = self.mod._gather_sections()
        # All sections timed out → every value is ''.
        self.assertTrue(all(v == "" for v in out.values()))


class ComposeWithinBudgetTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival")

    def test_returns_untrimmed_when_under_budget(self):
        parts = {"time": "7 AM", "weather": "", "meeting": "", "claude": "",
                 "print": "", "teams": "", "anomalies": "", "headline": ""}
        out = self.mod._compose_within_budget(parts)
        self.assertEqual(out, self.mod._compose_briefing(parts))

    def test_stops_dropping_once_within_budget(self):
        # headline is first in TTS_DROP_ORDER; dropping just it should suffice,
        # leaving weather/meeting/claude intact.
        big = "x" * 120
        parts = {"time": "7 AM", "weather": "w", "meeting": "m",
                 "claude": "c", "print": "print done", "teams": "teams ping",
                 "anomalies": "", "headline": big}
        with mock.patch.object(self.mod, "TTS_BUDGET_SECONDS", 8.0):
            out = self.mod._compose_within_budget(parts)
        self.assertNotIn(big, out)
        self.assertIn("teams ping", out)
        self.assertIn("print done", out)

    def test_drops_all_droppables_but_still_over_budget(self):
        # Only the non-droppable attention items are huge → even after dropping
        # every TTS_DROP_ORDER section the estimate stays over budget, so the
        # loop runs to completion (no early break). Some drop keys are already
        # empty, exercising the `if not trimmed.get(drop_key): continue` skip.
        huge = "x" * 600
        parts = {"time": "", "weather": "", "meeting": "",   # droppables empty
                 "claude": "", "print": huge, "teams": huge,  # spine: huge
                 "anomalies": huge, "headline": ""}
        with mock.patch.object(self.mod, "TTS_BUDGET_SECONDS", 5.0):
            out = self.mod._compose_within_budget(parts)
        # The attention spine is never dropped, so it survives over-budget.
        self.assertIn("Three things require your attention", out)
        self.assertGreater(self.mod._estimate_tts_seconds(out),
                           self.mod.TTS_BUDGET_SECONDS)

    def test_estimate_empty_body_is_zero(self):
        self.assertEqual(self.mod._estimate_tts_seconds("[intent:briefing] "), 0.0)
        self.assertEqual(self.mod._estimate_tts_seconds(""), 0.0)

    def test_build_briefing_wires_gather_and_budget(self):
        parts = {"time": "7:42 AM", "weather": "", "meeting": "", "claude": "",
                 "print": "", "teams": "one new Teams chat from Alice",
                 "anomalies": "", "headline": ""}
        with mock.patch.object(self.mod, "_gather_sections", return_value=parts):
            out = self.mod._build_briefing()
        self.assertIn("It's 7:42 AM.", out)
        self.assertIn("One item requires your attention", out)


class FirePathTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_arrival")

    def test_fire_arrival_suppressed_when_already_fired(self):
        with mock.patch.object(self.mod, "_arrival_already_fired_today",
                               return_value=True), \
             mock.patch.object(self.mod, "_build_briefing") as build:
            out = self.mod._fire_arrival("auto")
        self.assertEqual(out, "")
        build.assert_not_called()

    def test_fire_arrival_forced_bypasses_suppression(self):
        with mock.patch.object(self.mod, "_arrival_already_fired_today",
                               return_value=True), \
             mock.patch.object(self.mod, "_build_briefing",
                               return_value="[intent:briefing] hi"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.mod._fire_arrival("manual", force=True)
        self.assertEqual(out, "[intent:briefing] hi")
        enq.assert_called_once()
        mark.assert_called_once_with("manual")

    def test_fire_arrival_build_failure_returns_empty(self):
        with mock.patch.object(self.mod, "_arrival_already_fired_today",
                               return_value=False), \
             mock.patch.object(self.mod, "_build_briefing",
                               side_effect=RuntimeError("build boom")), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.mod._fire_arrival("auto")
        self.assertEqual(out, "")
        enq.assert_not_called()
        mark.assert_not_called()

    def test_fire_arrival_happy_path_enqueues_and_marks(self):
        with mock.patch.object(self.mod, "_arrival_already_fired_today",
                               return_value=False), \
             mock.patch.object(self.mod, "_build_briefing",
                               return_value="[intent:briefing] morning"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.mod._fire_arrival("auto")
        self.assertEqual(out, "[intent:briefing] morning")
        enq.assert_called_once_with("[intent:briefing] morning")
        mark.assert_called_once_with("auto")

    # ── _fire_from_chain re-check branches ───────────────────────────────
    def test_chain_bails_when_already_fired_upfront(self):
        with mock.patch.object(self.mod, "_arrival_already_fired_today",
                               return_value=True), \
             mock.patch.object(self.mod, "_silence_hours_since_last_speech") as sil:
            out = self.mod._fire_from_chain("chain")
        self.assertEqual(out, "")
        sil.assert_not_called()

    def test_chain_bails_on_recheck_after_delay(self):
        # Passes the upfront check + silence gate, then a parallel fire marks it
        # fired during the sleep → the post-delay re-check bails.
        calls = {"n": 0}

        def _already_fired():
            calls["n"] += 1
            return calls["n"] >= 2  # False first (upfront), True after the sleep

        with mock.patch.object(self.mod, "_arrival_already_fired_today",
                               side_effect=_already_fired), \
             mock.patch.object(self.mod, "_silence_hours_since_last_speech",
                               return_value=9.0), \
             mock.patch.object(self.mod.time, "sleep"), \
             mock.patch.object(self.mod, "_fire_arrival") as fire:
            out = self.mod._fire_from_chain("chain")
        self.assertEqual(out, "")
        fire.assert_not_called()

    # ── registered actions + alias ───────────────────────────────────────
    def test_actions_registered_with_alias(self):
        self.assertIn("morning_arrival", self.actions)
        self.assertIn("arrival_briefing", self.actions)
        self.assertIs(self.actions["morning_arrival"], self.actions["arrival_briefing"])

    def test_action_handles_internal_exception(self):
        mod, actions = load_skill_isolated("morning_arrival")
        with mock.patch.object(mod, "_fire_arrival",
                               side_effect=RuntimeError("kaboom")):
            out = actions["morning_arrival"]("")
        self.assertIn("failed", out.lower())
        self.assertIn("kaboom", out)


if __name__ == "__main__":
    unittest.main()
