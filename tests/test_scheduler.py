"""Unit tests for ``core.scheduler`` — the APScheduler-backed cron / interval /
one-shot / conditional job engine.

Design of this suite
---------------------
APScheduler *is* importable in the test env but SQLAlchemy is not, so the
module's ``_aps_imports`` naturally resolves ``SQLAlchemyJobStore`` to None and
falls back to the in-memory jobstore. We never let a real BackgroundScheduler
thread or the condition-poller daemon actually run:

  * Job construction / listing / cancel / fire tests inject a lightweight
    ``FakeScheduler`` straight into ``_state["scheduler"]`` — the public
    ``schedule_*`` helpers only need ``add_job`` / ``get_jobs`` / ``remove_job``
    / ``get_job`` to exist. Real apscheduler trigger classes are still used so
    the trigger-construction kwargs are validated for real.
  * ``bootstrap`` is exercised against a fully faked ``_aps_imports`` payload
    (fake BackgroundScheduler + fake/raising SQLAlchemyJobStore) with
    ``threading.Thread`` patched to a no-op, so no daemon thread is spawned.
  * The condition poller's rising-edge logic is driven by calling
    ``_evaluate_conditions_once`` directly — deterministic, no sleeps.

Every test redirects the module's data-file paths into a per-test temp dir and
restores ``_state`` afterwards so the suite is order-independent and offline.

stdlib ``unittest`` only.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

import core.scheduler as sched


# ── test doubles ────────────────────────────────────────────────────
class FakeJob:
    """Minimal stand-in for an apscheduler Job."""

    def __init__(self, id, trigger, kwargs=None, name=""):
        self.id = id
        self.trigger = trigger
        self.kwargs = kwargs or {}
        self.name = name
        self.next_run_time = None


class FakeScheduler:
    """Records add_job calls and supports the read/remove surface the module
    touches. Not thread-backed — nothing actually fires."""

    def __init__(self):
        self.jobs: dict = {}
        self.running = True
        self.add_calls: list = []
        self.removed: list = []
        self.raise_on_get_jobs = False

    def add_job(self, func, trigger=None, kwargs=None, id=None,
                replace_existing=True, name=""):
        self.add_calls.append({
            "func": func, "trigger": trigger, "kwargs": kwargs,
            "id": id, "replace_existing": replace_existing, "name": name,
        })
        self.jobs[id] = FakeJob(id, trigger, kwargs, name)
        return self.jobs[id]

    def get_jobs(self):
        if self.raise_on_get_jobs:
            raise RuntimeError("jobstore offline")
        return list(self.jobs.values())

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def remove_job(self, job_id):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        del self.jobs[job_id]

    def shutdown(self, wait=False):
        self.running = False


class FakeTrigger:
    """Generic trigger double whose class name drives ``_describe_trigger``."""

    def __init__(self, **kw):
        self.kw = kw


# ── base fixture: isolate module state + data paths ─────────────────
class SchedulerTestBase(unittest.TestCase):
    # Keys whose values may hold un-serializable live objects (a real
    # apscheduler BackgroundScheduler, the action callables, the poller
    # Thread/Event). deepcopy-ing a live scheduler raises "Schedulers cannot be
    # serialized" — and another test (test_skills_smoke loads schedule_manager,
    # which bootstraps a REAL scheduler into the global _state) can leave one
    # here. So these are snapshot by REFERENCE and never deep-copied.
    _LIVE_OBJECT_KEYS = ("scheduler", "actions", "cond_thread", "cond_stop")

    def setUp(self):
        # Snapshot mutable module-level state so each test is hermetic. Detach
        # the live-object keys BEFORE deepcopy so a real scheduler left in the
        # global state by another test can't crash the (un-serializable)
        # deepcopy; back them up by reference instead.
        live_backup = {k: sched._state.get(k) for k in self._LIVE_OBJECT_KEYS}
        for k in self._LIVE_OBJECT_KEYS:
            sched._state[k] = None
        self._state_backup = copy.deepcopy(sched._state)
        self._state_backup.update(live_backup)
        # If another test leaked a live scheduler, shut it down now so its
        # daemon thread/jobstore doesn't linger for the rest of this module.
        leaked = live_backup.get("scheduler")
        if leaked is not None and hasattr(leaked, "shutdown"):
            try:
                leaked.shutdown(wait=False)
            except Exception:
                pass
        leaked_stop = live_backup.get("cond_stop")
        if leaked_stop is not None and hasattr(leaked_stop, "set"):
            try:
                leaked_stop.set()
            except Exception:
                pass
        # Start every test from a clean baseline for the live-object + data keys.
        sched._state["scheduler"] = None
        sched._state["actions"] = None
        sched._state["started_at"] = None
        sched._state["conditions"] = {}
        sched._state["cond_thread"] = None
        sched._state["cond_stop"] = None
        sched._state["cond_state"] = {}
        sched._state["last_error"] = None

        # Redirect every data-file path into a throwaway temp dir.
        self._tmp = tempfile.mkdtemp(prefix="sched_test_")
        self._p_conditions = mock.patch.object(
            sched, "_CONDITIONS_PATH", os.path.join(self._tmp, "conds.json"))
        self._p_datadir = mock.patch.object(
            sched, "_DATA_DIR", os.path.join(self._tmp, "data"))
        self._p_db = mock.patch.object(
            sched, "_DB_PATH", os.path.join(self._tmp, "scheduler.db"))
        self._p_conditions.start()
        self._p_datadir.start()
        self._p_db.start()

    def tearDown(self):
        self._p_conditions.stop()
        self._p_datadir.stop()
        self._p_db.stop()
        # If a test installed/created a live scheduler (or condition poller),
        # tear it down so neither a real apscheduler daemon thread nor an
        # un-serializable scheduler object leaks into the next test's setUp
        # deepcopy. FakeScheduler.shutdown() is a harmless no-op flag flip.
        cur_sched = sched._state.get("scheduler")
        if cur_sched is not None and hasattr(cur_sched, "shutdown"):
            try:
                cur_sched.shutdown(wait=False)
            except Exception:
                pass
        cur_stop = sched._state.get("cond_stop")
        if cur_stop is not None and hasattr(cur_stop, "set"):
            try:
                cur_stop.set()
            except Exception:
                pass
        # Restore state. Live-object keys are restored BY REFERENCE from the
        # backup (never deep-copied — a real scheduler can't be); the rest are
        # deep-copied so plain data can't bleed between tests.
        for k, v in self._state_backup.items():
            if k in self._LIVE_OBJECT_KEYS:
                sched._state[k] = v
            else:
                sched._state[k] = copy.deepcopy(v)
        sched._state["conditions"] = {}
        sched._state["cond_state"] = {}
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    # helper: install a FakeScheduler as the live scheduler
    def _install_fake_scheduler(self):
        fs = FakeScheduler()
        sched._state["scheduler"] = fs
        return fs


# ── is_available / imports ──────────────────────────────────────────
class AvailabilityTests(SchedulerTestBase):
    def test_is_available_true_when_imports_resolve(self):
        # apscheduler is installed in this env, so this is genuinely True.
        self.assertTrue(sched.is_available())

    def test_aps_imports_cached(self):
        first = sched._aps_imports()
        second = sched._aps_imports()
        self.assertIs(first, second)
        # Core trigger classes are present.
        for key in ("BackgroundScheduler", "CronTrigger",
                    "IntervalTrigger", "DateTrigger"):
            self.assertIn(key, first)

    def test_is_available_false_on_import_failure(self):
        # Simulate a minimal env where apscheduler isn't importable: clear the
        # cache and make the lazy importer fail.
        with mock.patch.dict(sched._imports, {}, clear=True):
            with mock.patch.object(sched, "_imports", {}):
                with mock.patch(
                    "builtins.__import__",
                    side_effect=ImportError("no apscheduler"),
                ):
                    self.assertFalse(sched.is_available())
        # _import_error string should have been recorded.
        self.assertIsNotNone(sched._import_error)


# ── run_action dispatch ─────────────────────────────────────────────
class RunActionTests(SchedulerTestBase):
    def test_run_action_before_bootstrap_is_a_noop_message(self):
        sched._state["actions"] = None
        out = sched.run_action("weather", "today")
        self.assertIn("not bootstrapped", out)
        self.assertIn("weather", out)

    def test_run_action_dispatches_and_formats_string_result(self):
        sched._state["actions"] = {"weather": lambda a: f"sunny:{a}"}
        out = sched.run_action("weather", "today")
        self.assertEqual(out, "weather: sunny:today")

    def test_run_action_non_string_result_reports_ok(self):
        sched._state["actions"] = {"ping": lambda a: 12345}
        out = sched.run_action("ping", "")
        self.assertEqual(out, "ping: ok")

    def test_run_action_unregistered_action_is_reported(self):
        sched._state["actions"] = {"known": lambda a: "x"}
        out = sched.run_action("unknown", "")
        self.assertEqual(out, "unknown: not registered")

    def test_run_action_catches_action_exception(self):
        def boom(_):
            raise ValueError("kaboom")
        sched._state["actions"] = {"bad": boom}
        out = sched.run_action("bad", "")
        self.assertIn("bad: ValueError: kaboom", out)

    def test_run_action_chain_runs_each_step_in_order(self):
        order = []
        sched._state["actions"] = {
            "a": lambda x: order.append("a") or "ra",
            "b": lambda x: order.append("b") or "rb",
            "c": lambda x: order.append("c") or "rc",
        }
        out = sched.run_action(
            "a", "",
            chain=[{"action": "b", "arg": "y"}, {"action": "c"}],
        )
        self.assertEqual(order, ["a", "b", "c"])
        self.assertEqual(out, "a: ra | b: rb | c: rc")

    def test_run_action_chain_ignores_malformed_entries(self):
        sched._state["actions"] = {"a": lambda x: "ra"}
        # Non-dict entry and a dict lacking "action" are both skipped.
        out = sched.run_action(
            "a", "",
            chain=["not a dict", {"arg": "no action key"}],
        )
        self.assertEqual(out, "a: ra")


# ── condition registry ──────────────────────────────────────────────
class ConditionRegistryTests(SchedulerTestBase):
    def test_register_condition_adds_predicate(self):
        sched.register_condition("custom_flag", lambda: True)
        self.assertIn("custom_flag", sched.available_conditions())
        self.assertIs(sched._resolve_condition("custom_flag")(), True)

    def test_register_condition_rejects_bad_args(self):
        with self.assertRaises(ValueError):
            sched.register_condition("", lambda: True)
        with self.assertRaises(ValueError):
            sched.register_condition("x", "not callable")

    def test_available_conditions_merges_builtins_and_custom_sorted(self):
        sched.register_condition("zzz_custom", lambda: False)
        avail = sched.available_conditions()
        self.assertEqual(avail, sorted(avail))
        for builtin in sched._BUILTIN_CONDITIONS:
            self.assertIn(builtin, avail)
        self.assertIn("zzz_custom", avail)

    def test_resolve_condition_prefers_custom_over_builtin(self):
        sentinel = lambda: True
        sched.register_condition("disk_low", sentinel)  # shadow a builtin
        self.assertIs(sched._resolve_condition("disk_low"), sentinel)

    def test_resolve_condition_unknown_returns_none(self):
        self.assertIsNone(sched._resolve_condition("does_not_exist"))


# ── built-in condition predicates ───────────────────────────────────
class BuiltinConditionTests(SchedulerTestBase):
    def _write_overlay(self, payload):
        path = os.path.join(sched._PROJECT_DIR, "bambu_overlay_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

    def test_bambu_finished_true_on_finish_state(self):
        self._write_overlay({"gcode_state": "finish"})
        self.assertTrue(sched._cond_bambu_print_finished())

    def test_bambu_finished_false_on_other_state(self):
        self._write_overlay({"gcode_state": "running"})
        self.assertFalse(sched._cond_bambu_print_finished())

    def test_bambu_failed_true_on_failed_state(self):
        self._write_overlay({"gcode_state": "FAILED"})
        self.assertTrue(sched._cond_bambu_print_failed())

    def test_bambu_failed_true_on_error_field(self):
        self._write_overlay({"gcode_state": "running", "print_error": "117"})
        self.assertTrue(sched._cond_bambu_print_failed())

    def test_bambu_failed_false_on_zero_error(self):
        self._write_overlay({"gcode_state": "running", "print_error": "0"})
        self.assertFalse(sched._cond_bambu_print_failed())

    def test_bambu_started_true_on_running(self):
        self._write_overlay({"gcode_state": "PRINTING"})
        self.assertTrue(sched._cond_bambu_print_started())

    def test_read_json_safe_missing_file_returns_empty(self):
        self.assertEqual(sched._read_json_safe(
            os.path.join(self._tmp, "nope.json")), {})

    def test_read_json_safe_bad_json_returns_empty(self):
        bad = os.path.join(self._tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertEqual(sched._read_json_safe(bad), {})

    def test_disk_low_true_when_under_threshold(self):
        fake_usage = mock.Mock(return_value=(100, 99, 500 * 1024 * 1024))
        with mock.patch("shutil.disk_usage", fake_usage):
            self.assertTrue(sched._cond_disk_low())

    def test_disk_low_false_when_plenty_free(self):
        fake_usage = mock.Mock(return_value=(100, 1, 50 * 1024 * 1024 * 1024))
        with mock.patch("shutil.disk_usage", fake_usage):
            self.assertFalse(sched._cond_disk_low())

    def test_disk_low_false_on_exception(self):
        with mock.patch("shutil.disk_usage", side_effect=OSError("boom")):
            self.assertFalse(sched._cond_disk_low())

    def test_ram_high_true_when_over_threshold(self):
        fake_psutil = mock.Mock()
        fake_psutil.virtual_memory.return_value = mock.Mock(percent=95.0)
        with mock.patch.dict("sys.modules", {"psutil": fake_psutil}):
            self.assertTrue(sched._cond_ram_high())

    def test_ram_high_false_when_under_threshold(self):
        fake_psutil = mock.Mock()
        fake_psutil.virtual_memory.return_value = mock.Mock(percent=42.0)
        with mock.patch.dict("sys.modules", {"psutil": fake_psutil}):
            self.assertFalse(sched._cond_ram_high())

    def test_ram_high_false_when_psutil_missing(self):
        # Force the import of psutil to fail.
        with mock.patch.dict("sys.modules", {"psutil": None}):
            self.assertFalse(sched._cond_ram_high())


# ── conditional-trigger persistence helpers ─────────────────────────
class ConditionPersistenceTests(SchedulerTestBase):
    def test_read_conditions_missing_file_returns_empty(self):
        self.assertEqual(sched._read_conditions(), [])

    def test_write_then_read_roundtrip_list(self):
        data = [{"id": "t1", "condition": "disk_low", "action": "warn"}]
        sched._write_conditions(data)
        self.assertEqual(sched._read_conditions(), data)

    def test_read_conditions_dict_with_triggers_key(self):
        with open(sched._CONDITIONS_PATH, "w", encoding="utf-8") as f:
            json.dump({"triggers": [{"id": "x"}]}, f)
        self.assertEqual(sched._read_conditions(), [{"id": "x"}])

    def test_read_conditions_unexpected_shape_returns_empty(self):
        with open(sched._CONDITIONS_PATH, "w", encoding="utf-8") as f:
            json.dump({"no_triggers_key": 1}, f)
        self.assertEqual(sched._read_conditions(), [])

    def test_read_conditions_corrupt_json_returns_empty(self):
        with open(sched._CONDITIONS_PATH, "w", encoding="utf-8") as f:
            f.write("[[[ broken")
        self.assertEqual(sched._read_conditions(), [])


# ── _evaluate_conditions_once (rising-edge poller core) ─────────────
class ConditionEvaluationTests(SchedulerTestBase):
    def setUp(self):
        super().setUp()
        self.fired = []
        sched._state["actions"] = {
            "notify": lambda a: self.fired.append(a) or "notified",
        }

    def _put_triggers(self, triggers):
        sched._write_conditions(triggers)

    def test_no_triggers_is_a_noop(self):
        # Returns early; cond_state stays empty.
        sched._evaluate_conditions_once()
        self.assertEqual(sched._state["cond_state"], {})

    def test_rising_edge_fires_action(self):
        flag = {"v": False}
        sched.register_condition("myflag", lambda: flag["v"])
        self._put_triggers([{
            "id": "t1", "condition": "myflag", "action": "notify", "arg": "hi",
        }])
        # First sweep: condition is False (and seeded to current value) → no fire.
        sched._evaluate_conditions_once()
        self.assertEqual(self.fired, [])
        # Flip True → next sweep fires (False→True edge).
        flag["v"] = True
        sched._evaluate_conditions_once()
        self.assertEqual(self.fired, ["hi"])
        # Stays True → no re-fire (debounced by cond_state).
        sched._evaluate_conditions_once()
        self.assertEqual(self.fired, ["hi"])

    def test_already_true_on_first_sweep_does_not_fire(self):
        sched.register_condition("hot", lambda: True)
        self._put_triggers([{
            "id": "t1", "condition": "hot", "action": "notify",
        }])
        sched._evaluate_conditions_once()
        # Seeded from current value (True) so no spurious rising edge.
        self.assertEqual(self.fired, [])
        self.assertTrue(sched._state["cond_state"]["t1"])

    def test_trigger_missing_required_fields_is_skipped(self):
        self._put_triggers([
            {"id": "t1"},                                   # no condition/action
            {"condition": "disk_low", "action": "notify"},  # no id
        ])
        sched._evaluate_conditions_once()
        self.assertEqual(self.fired, [])

    def test_unknown_condition_preserves_prev_state(self):
        # Seed a previous value, then reference an unresolvable condition: the
        # prior edge-state for that id must survive the sweep.
        sched._state["cond_state"] = {"t1": True}
        self._put_triggers([{
            "id": "t1", "condition": "ghost_condition", "action": "notify",
        }])
        sched._evaluate_conditions_once()
        self.assertEqual(self.fired, [])
        self.assertEqual(sched._state["cond_state"].get("t1"), True)

    def test_condition_raising_preserves_prev_state(self):
        def boom():
            raise RuntimeError("sensor down")
        sched.register_condition("flaky", boom)
        sched._state["cond_state"] = {"t1": False}
        self._put_triggers([{
            "id": "t1", "condition": "flaky", "action": "notify",
        }])
        sched._evaluate_conditions_once()
        self.assertEqual(self.fired, [])
        self.assertEqual(sched._state["cond_state"].get("t1"), False)

    def test_one_shot_trigger_removed_after_firing(self):
        flag = {"v": False}
        sched.register_condition("once_flag", lambda: flag["v"])
        self._put_triggers([{
            "id": "os1", "condition": "once_flag", "action": "notify",
            "one_shot": True,
        }])
        sched._evaluate_conditions_once()       # seed False
        flag["v"] = True
        sched._evaluate_conditions_once()       # fire + drop
        self.assertEqual(self.fired, [""])
        # Trigger removed from disk and edge-state.
        self.assertEqual(sched._read_conditions(), [])
        self.assertNotIn("os1", sched._state["cond_state"])

    def test_deleted_trigger_ids_are_pruned_from_state(self):
        # cond_state holds an id no longer present on disk → it should not be
        # carried forward after a sweep that doesn't include it.
        sched.register_condition("present", lambda: False)
        sched._state["cond_state"] = {"stale": True}
        self._put_triggers([{
            "id": "present_id", "condition": "present", "action": "notify",
        }])
        sched._evaluate_conditions_once()
        self.assertNotIn("stale", sched._state["cond_state"])
        self.assertIn("present_id", sched._state["cond_state"])

    def test_chain_passed_through_on_fire(self):
        flag = {"v": False}
        sched.register_condition("cflag", lambda: flag["v"])
        seen = []
        sched._state["actions"] = {
            "first": lambda a: seen.append("first") or "r1",
            "second": lambda a: seen.append("second") or "r2",
        }
        self._put_triggers([{
            "id": "t1", "condition": "cflag", "action": "first",
            "chain": [{"action": "second"}],
        }])
        sched._evaluate_conditions_once()
        flag["v"] = True
        sched._evaluate_conditions_once()
        self.assertEqual(seen, ["first", "second"])

    def test_fire_dispatch_exception_is_caught(self):
        # run_action normally swallows its own action errors, but if dispatch
        # itself raises the rising-edge handler must log-and-continue, still
        # recording the new edge-state value.
        flag = {"v": False}
        sched.register_condition("cflag", lambda: flag["v"])
        self._put_triggers([{
            "id": "t1", "condition": "cflag", "action": "notify",
        }])
        sched._evaluate_conditions_once()       # seed False
        flag["v"] = True
        with mock.patch.object(sched, "run_action",
                               side_effect=RuntimeError("dispatch boom")):
            sched._evaluate_conditions_once()   # fire path raises, is caught
        # Sweep survived and the edge value advanced to True.
        self.assertTrue(sched._state["cond_state"]["t1"])

    def test_one_shot_prune_write_failure_is_swallowed(self):
        flag = {"v": False}
        sched.register_condition("once_flag", lambda: flag["v"])
        self._put_triggers([{
            "id": "os1", "condition": "once_flag", "action": "notify",
            "one_shot": True,
        }])
        sched._evaluate_conditions_once()       # seed False
        flag["v"] = True
        with mock.patch.object(sched, "_write_conditions",
                               side_effect=OSError("disk full")):
            sched._evaluate_conditions_once()   # fires, prune-write fails, swallowed
        self.assertEqual(self.fired, [""])      # action still fired
        # Edge-state still pruned of the one-shot id despite the write failure.
        self.assertNotIn("os1", sched._state["cond_state"])

    def test_malformed_trigger_entry_does_not_crash_sweep(self):
        # A trigger whose .get() blows up (non-dict) is caught by the inner
        # try/except so the rest of the sweep proceeds.
        flag = {"v": True}
        sched.register_condition("good", lambda: flag["v"])
        # Manually write a list with a bad (non-dict) element alongside a good
        # trigger; _read_conditions returns it verbatim.
        with open(sched._CONDITIONS_PATH, "w", encoding="utf-8") as f:
            json.dump([
                "i am not a dict",
                {"id": "g1", "condition": "good", "action": "notify"},
            ], f)
        # Seed g1 False so the good trigger fires on this sweep.
        sched._state["cond_state"] = {"g1": False}
        sched._evaluate_conditions_once()
        self.assertEqual(self.fired, [""])


# ── bootstrap / shutdown ────────────────────────────────────────────
class _FakeBackgroundScheduler:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.running = False
        self.shutdown_called = False
        _FakeBackgroundScheduler.instances.append(self)

    def start(self):
        self.started = True
        self.running = True

    def shutdown(self, wait=False):
        self.shutdown_called = True
        self.running = False


class _FakeSQLAlchemyJobStore:
    last_url = None

    def __init__(self, url=None):
        _FakeSQLAlchemyJobStore.last_url = url


class _RaisingSQLAlchemyJobStore:
    def __init__(self, url=None):
        raise RuntimeError("sqlite open failed")


class BootstrapTests(SchedulerTestBase):
    def setUp(self):
        super().setUp()
        _FakeBackgroundScheduler.instances = []
        # No real daemon thread, ever.
        self._p_thread = mock.patch.object(sched.threading, "Thread",
                                           return_value=mock.Mock())
        self._p_thread.start()
        self.addCleanup(self._p_thread.stop)

    def _imports_payload(self, jobstore_cls):
        return {
            "BackgroundScheduler": _FakeBackgroundScheduler,
            "CronTrigger": object,
            "IntervalTrigger": object,
            "DateTrigger": object,
            "SQLAlchemyJobStore": jobstore_cls,
        }

    def test_bootstrap_unavailable_records_error_and_returns_false(self):
        with mock.patch.object(sched, "is_available", return_value=False):
            with mock.patch.object(sched, "_import_error", "boom"):
                ok = sched.bootstrap({"a": lambda x: "x"})
        self.assertFalse(ok)
        self.assertIn("APScheduler unavailable", sched._state["last_error"])

    def test_bootstrap_with_sqlalchemy_jobstore_uses_sqlite_url(self):
        payload = self._imports_payload(_FakeSQLAlchemyJobStore)
        with mock.patch.object(sched, "is_available", return_value=True), \
             mock.patch.object(sched, "_aps_imports", return_value=payload):
            ok = sched.bootstrap({"a": lambda x: "x"})
        self.assertTrue(ok)
        inst = _FakeBackgroundScheduler.instances[-1]
        self.assertTrue(inst.started)
        # jobstores kwarg was passed because the store constructed cleanly.
        self.assertIn("jobstores", inst.kwargs)
        self.assertTrue(_FakeSQLAlchemyJobStore.last_url.startswith("sqlite:///"))

    def test_bootstrap_jobstore_construction_failure_falls_back_to_memory(self):
        payload = self._imports_payload(_RaisingSQLAlchemyJobStore)
        with mock.patch.object(sched, "is_available", return_value=True), \
             mock.patch.object(sched, "_aps_imports", return_value=payload):
            ok = sched.bootstrap({"a": lambda x: "x"})
        self.assertTrue(ok)
        inst = _FakeBackgroundScheduler.instances[-1]
        # Failed store → no jobstores kwarg → in-memory MemoryJobStore.
        self.assertNotIn("jobstores", inst.kwargs)

    def test_bootstrap_no_sqlalchemy_jobstore_omits_kwarg(self):
        payload = self._imports_payload(None)
        with mock.patch.object(sched, "is_available", return_value=True), \
             mock.patch.object(sched, "_aps_imports", return_value=payload):
            ok = sched.bootstrap({"a": lambda x: "x"})
        self.assertTrue(ok)
        inst = _FakeBackgroundScheduler.instances[-1]
        self.assertNotIn("jobstores", inst.kwargs)
        # job_defaults always passed (coalesce/misfire grace).
        self.assertEqual(inst.kwargs["job_defaults"]["misfire_grace_time"],
                         60 * 60)
        self.assertTrue(inst.kwargs["job_defaults"]["coalesce"])

    def test_bootstrap_is_idempotent(self):
        payload = self._imports_payload(None)
        with mock.patch.object(sched, "is_available", return_value=True), \
             mock.patch.object(sched, "_aps_imports", return_value=payload):
            self.assertTrue(sched.bootstrap({"a": lambda x: "1"}))
            count_after_first = len(_FakeBackgroundScheduler.instances)
            # Second call with a NEW actions dict: rebinds actions, does NOT
            # build a second scheduler.
            self.assertTrue(sched.bootstrap({"b": lambda x: "2"}))
        self.assertEqual(len(_FakeBackgroundScheduler.instances),
                         count_after_first)
        self.assertIn("b", sched._state["actions"])

    def test_bootstrap_start_failure_records_error(self):
        class _BoomScheduler(_FakeBackgroundScheduler):
            def start(self):
                raise RuntimeError("cannot start")
        payload = self._imports_payload(None)
        payload["BackgroundScheduler"] = _BoomScheduler
        with mock.patch.object(sched, "is_available", return_value=True), \
             mock.patch.object(sched, "_aps_imports", return_value=payload):
            ok = sched.bootstrap({"a": lambda x: "x"})
        self.assertFalse(ok)
        self.assertIn("BackgroundScheduler.start failed",
                      sched._state["last_error"])
        self.assertIsNone(sched._state["scheduler"])

    def test_bootstrap_starts_condition_poller_thread(self):
        payload = self._imports_payload(None)
        with mock.patch.object(sched, "is_available", return_value=True), \
             mock.patch.object(sched, "_aps_imports", return_value=payload):
            sched.bootstrap({"a": lambda x: "x"})
        # The patched Thread was constructed with the poller target and started.
        sched.threading.Thread.assert_called_once()
        _, kwargs = sched.threading.Thread.call_args
        self.assertIs(kwargs["target"], sched._condition_poller)
        self.assertTrue(kwargs["daemon"])
        sched._state["cond_thread"].start.assert_called_once()
        self.assertIsNotNone(sched._state["cond_stop"])

    def test_shutdown_sets_stop_event_and_shuts_scheduler(self):
        fake_sched = _FakeBackgroundScheduler()
        stop = mock.Mock()
        sched._state["scheduler"] = fake_sched
        sched._state["cond_stop"] = stop
        sched.shutdown(wait=True)
        stop.set.assert_called_once()
        self.assertTrue(fake_sched.shutdown_called)
        self.assertIsNone(sched._state["scheduler"])

    def test_shutdown_when_nothing_running_is_safe(self):
        sched._state["scheduler"] = None
        sched._state["cond_stop"] = None
        sched.shutdown()  # must not raise
        self.assertIsNone(sched._state["scheduler"])

    def test_shutdown_swallows_scheduler_errors(self):
        bad = mock.Mock()
        bad.shutdown.side_effect = RuntimeError("already dead")
        stop = mock.Mock()
        stop.set.side_effect = RuntimeError("stop boom")
        sched._state["scheduler"] = bad
        sched._state["cond_stop"] = stop
        sched.shutdown()  # both errors swallowed
        self.assertIsNone(sched._state["scheduler"])


# ── condition poller thread (driven without real sleeping) ──────────
class ConditionPollerThreadTests(SchedulerTestBase):
    def test_poller_exits_immediately_if_stopped_during_boot_delay(self):
        stop = mock.Mock()
        stop.wait.return_value = True   # boot-delay wait returns True → exit
        sched._state["cond_stop"] = stop
        with mock.patch.object(sched, "_evaluate_conditions_once") as ev:
            sched._condition_poller()
        ev.assert_not_called()
        stop.wait.assert_called_once_with(sched._CONDITION_BOOT_DELAY)

    def test_poller_runs_one_iteration_then_stops(self):
        stop = mock.Mock()
        # boot-delay False (proceed), is_set False (enter loop), poll-wait True (exit)
        stop.wait.side_effect = [False, True]
        stop.is_set.return_value = False
        sched._state["cond_stop"] = stop
        with mock.patch.object(sched, "_evaluate_conditions_once") as ev:
            sched._condition_poller()
        ev.assert_called_once()

    def test_poller_survives_iteration_exception(self):
        stop = mock.Mock()
        stop.wait.side_effect = [False, True]
        stop.is_set.return_value = False
        sched._state["cond_stop"] = stop
        with mock.patch.object(sched, "_evaluate_conditions_once",
                               side_effect=RuntimeError("iter boom")) as ev:
            sched._condition_poller()  # exception logged, then exits cleanly
        ev.assert_called_once()


# ── schedule_cron / interval / once (job construction) ──────────────
class ScheduleJobTests(SchedulerTestBase):
    def setUp(self):
        super().setUp()
        self.fs = self._install_fake_scheduler()

    def test_schedule_cron_adds_job_with_real_cron_trigger(self):
        jid = sched.schedule_cron(action="brief", arg="emails",
                                  hour=8, minute=0, job_id="morning")
        self.assertEqual(jid, "morning")
        call = self.fs.add_calls[-1]
        self.assertIs(call["func"], sched.run_action)
        self.assertEqual(call["id"], "morning")
        self.assertEqual(call["name"], "cron:brief")
        self.assertEqual(call["kwargs"], {"action": "brief", "arg": "emails"})
        # A real apscheduler CronTrigger was built.
        self.assertEqual(type(call["trigger"]).__name__, "CronTrigger")

    def test_schedule_cron_autogenerates_id_when_omitted(self):
        jid = sched.schedule_cron(action="x", hour=9)
        self.assertTrue(jid.startswith("cron_"))
        self.assertEqual(len(jid), len("cron_") + 8)

    def test_schedule_cron_includes_chain_in_kwargs(self):
        sched.schedule_cron(action="a", chain=[{"action": "b"}], job_id="c1")
        self.assertEqual(self.fs.add_calls[-1]["kwargs"]["chain"],
                         [{"action": "b"}])

    def test_schedule_cron_bad_field_raises(self):
        # An out-of-range cron field is rejected by the real CronTrigger.
        with self.assertRaises(ValueError):
            sched.schedule_cron(action="x", hour=99)

    def test_schedule_interval_adds_job(self):
        jid = sched.schedule_interval(action="poll", minutes=15, job_id="iv")
        self.assertEqual(jid, "iv")
        call = self.fs.add_calls[-1]
        self.assertEqual(call["name"], "interval:poll")
        self.assertEqual(type(call["trigger"]).__name__, "IntervalTrigger")

    def test_schedule_interval_autogenerates_id(self):
        jid = sched.schedule_interval(action="poll", seconds=30)
        self.assertTrue(jid.startswith("intv_"))

    def test_schedule_interval_zero_duration_raises(self):
        with self.assertRaises(ValueError) as cm:
            sched.schedule_interval(action="x", seconds=0, minutes=0, hours=0)
        self.assertIn("at least one", str(cm.exception))

    def test_schedule_once_with_datetime(self):
        when = datetime(2030, 1, 1, 8, 0, 0)
        jid = sched.schedule_once(action="remind", run_at=when, job_id="o1")
        self.assertEqual(jid, "o1")
        call = self.fs.add_calls[-1]
        self.assertEqual(call["name"], "once:remind")
        self.assertEqual(type(call["trigger"]).__name__, "DateTrigger")

    def test_schedule_once_with_epoch_float(self):
        epoch = datetime(2030, 6, 1, 12, 0, 0).timestamp()
        jid = sched.schedule_once(action="remind", run_at=epoch)
        self.assertTrue(jid.startswith("once_"))
        self.assertEqual(type(self.fs.add_calls[-1]["trigger"]).__name__,
                         "DateTrigger")

    def test_schedule_helpers_raise_when_not_bootstrapped(self):
        sched._state["scheduler"] = None
        with self.assertRaises(RuntimeError):
            sched.schedule_cron(action="x", hour=8)
        with self.assertRaises(RuntimeError):
            sched.schedule_interval(action="x", seconds=5)
        with self.assertRaises(RuntimeError):
            sched.schedule_once(action="x", run_at=datetime(2030, 1, 1))


# ── schedule_when (conditional triggers) ────────────────────────────
class ScheduleWhenTests(SchedulerTestBase):
    def test_schedule_when_persists_trigger(self):
        sched.register_condition("flag", lambda: False)
        name = sched.schedule_when(name="t1", condition="flag", action="warn",
                                   arg="low")
        self.assertEqual(name, "t1")
        stored = sched._read_conditions()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["id"], "t1")
        self.assertEqual(stored[0]["condition"], "flag")
        self.assertEqual(stored[0]["action"], "warn")
        self.assertEqual(stored[0]["arg"], "low")
        self.assertFalse(stored[0]["one_shot"])

    def test_schedule_when_unknown_condition_raises(self):
        with self.assertRaises(ValueError) as cm:
            sched.schedule_when(name="t1", condition="nope", action="warn")
        self.assertIn("unknown condition", str(cm.exception))

    def test_schedule_when_replaces_existing_by_id(self):
        sched.register_condition("flag", lambda: False)
        sched.schedule_when(name="dup", condition="flag", action="a")
        sched.schedule_when(name="dup", condition="flag", action="b")
        stored = sched._read_conditions()
        self.assertEqual(len(stored), 1)        # replaced, not duplicated
        self.assertEqual(stored[0]["action"], "b")

    def test_schedule_when_seeds_cond_state_from_current_value(self):
        sched.register_condition("hot", lambda: True)
        sched.schedule_when(name="t1", condition="hot", action="warn")
        # Seeded True so the poller won't treat first observation as an edge.
        self.assertTrue(sched._state["cond_state"]["t1"])

    def test_schedule_when_seeds_false_when_predicate_raises(self):
        def boom():
            raise RuntimeError("x")
        sched.register_condition("flaky", boom)
        sched.schedule_when(name="t1", condition="flaky", action="warn")
        self.assertFalse(sched._state["cond_state"]["t1"])

    def test_schedule_when_one_shot_flag_persisted(self):
        sched.register_condition("flag", lambda: False)
        sched.schedule_when(name="t1", condition="flag", action="a",
                            one_shot=True, chain=[{"action": "b"}])
        stored = sched._read_conditions()[0]
        self.assertTrue(stored["one_shot"])
        self.assertEqual(stored["chain"], [{"action": "b"}])


# ── listing / inspection ────────────────────────────────────────────
class ListingTests(SchedulerTestBase):
    def test_describe_trigger_cron(self):
        from apscheduler.triggers.cron import CronTrigger
        kind, summary = sched._describe_trigger(CronTrigger(hour=8, minute=30))
        self.assertEqual(kind, "cron")
        self.assertIn("hour=8", summary)
        self.assertIn("minute=30", summary)

    def test_describe_trigger_cron_all_default(self):
        from apscheduler.triggers.cron import CronTrigger
        kind, summary = sched._describe_trigger(CronTrigger())
        self.assertEqual(kind, "cron")
        self.assertEqual(summary, "cron(*)")

    def test_describe_trigger_interval(self):
        from apscheduler.triggers.interval import IntervalTrigger
        kind, summary = sched._describe_trigger(IntervalTrigger(minutes=5))
        self.assertEqual(kind, "interval")
        self.assertIn("every", summary)

    def test_describe_trigger_date(self):
        from apscheduler.triggers.date import DateTrigger
        kind, summary = sched._describe_trigger(
            DateTrigger(run_date=datetime(2030, 1, 1, 8, 0).astimezone()))
        self.assertEqual(kind, "date")
        self.assertIn("at", summary)

    def test_describe_trigger_unknown_type(self):
        kind, summary = sched._describe_trigger(FakeTrigger(x=1))
        self.assertEqual(kind, "faketrigger")

    def test_describe_trigger_cron_exception_path(self):
        # A CronTrigger-named object whose .fields blows up hits the except.
        class CronTrigger:  # noqa: N801 - deliberately shadow the name
            @property
            def fields(self):
                raise RuntimeError("boom")
        kind, summary = sched._describe_trigger(CronTrigger())
        self.assertEqual(kind, "cron")

    def test_describe_trigger_interval_exception_path(self):
        class IntervalTrigger:  # noqa: N801
            @property
            def interval(self):
                raise RuntimeError("boom")
        kind, _ = sched._describe_trigger(IntervalTrigger())
        self.assertEqual(kind, "interval")

    def test_describe_trigger_date_exception_path(self):
        class DateTrigger:  # noqa: N801
            @property
            def run_date(self):
                raise RuntimeError("boom")
        kind, _ = sched._describe_trigger(DateTrigger())
        self.assertEqual(kind, "date")

    def test_list_jobs_empty_when_no_scheduler(self):
        sched._state["scheduler"] = None
        self.assertEqual(sched.list_jobs(), [])

    def test_list_jobs_returns_summaries(self):
        from apscheduler.triggers.cron import CronTrigger
        fs = self._install_fake_scheduler()
        nrt = datetime(2030, 1, 1, 8, 0).astimezone()
        job = FakeJob("j1", CronTrigger(hour=8),
                      kwargs={"action": "brief", "arg": "x",
                              "chain": [{"action": "b"}]},
                      name="cron:brief")
        job.next_run_time = nrt
        fs.jobs["j1"] = job
        summaries = sched.list_jobs()
        self.assertEqual(len(summaries), 1)
        s = summaries[0]
        self.assertEqual(s["id"], "j1")
        self.assertEqual(s["kind"], "cron")
        self.assertEqual(s["action"], "brief")
        self.assertEqual(s["arg"], "x")
        self.assertEqual(s["chain"], [{"action": "b"}])
        self.assertEqual(s["next_run"], nrt.isoformat())
        self.assertEqual(s["name"], "cron:brief")

    def test_list_jobs_handles_job_with_no_next_run(self):
        from apscheduler.triggers.date import DateTrigger
        fs = self._install_fake_scheduler()
        job = FakeJob("j2", DateTrigger(
            run_date=datetime(2030, 1, 1).astimezone()))
        job.next_run_time = None
        fs.jobs["j2"] = job
        s = sched.list_jobs()[0]
        self.assertIsNone(s["next_run"])
        self.assertEqual(s["arg"], "")
        self.assertEqual(s["chain"], [])

    def test_list_jobs_swallows_get_jobs_error(self):
        fs = self._install_fake_scheduler()
        fs.raise_on_get_jobs = True
        self.assertEqual(sched.list_jobs(), [])

    def test_list_conditions_merges_current_value(self):
        sched._write_conditions([
            {"id": "c1", "condition": "disk_low", "action": "warn"},
            {"id": "c2", "condition": "ram_high", "action": "warn"},
        ])
        sched._state["cond_state"] = {"c1": True}
        conds = sched.list_conditions()
        by_id = {c["id"]: c for c in conds}
        self.assertEqual(by_id["c1"]["current_value"], True)
        self.assertIsNone(by_id["c2"]["current_value"])


# ── cancel_job ──────────────────────────────────────────────────────
class CancelJobTests(SchedulerTestBase):
    def test_cancel_removes_apscheduler_job(self):
        fs = self._install_fake_scheduler()
        fs.jobs["j1"] = FakeJob("j1", FakeTrigger())
        self.assertTrue(sched.cancel_job("j1"))
        self.assertNotIn("j1", fs.jobs)

    def test_cancel_removes_conditional_trigger(self):
        self._install_fake_scheduler()  # job not in store
        sched._write_conditions([
            {"id": "cond1", "condition": "disk_low", "action": "warn"},
            {"id": "keep", "condition": "ram_high", "action": "warn"},
        ])
        sched._state["cond_state"] = {"cond1": True}
        self.assertTrue(sched.cancel_job("cond1"))
        remaining = [t["id"] for t in sched._read_conditions()]
        self.assertEqual(remaining, ["keep"])
        self.assertNotIn("cond1", sched._state["cond_state"])

    def test_cancel_unknown_id_returns_false(self):
        self._install_fake_scheduler()
        sched._write_conditions([{"id": "other", "condition": "disk_low",
                                  "action": "warn"}])
        self.assertFalse(sched.cancel_job("nonexistent"))

    def test_cancel_works_without_scheduler(self):
        sched._state["scheduler"] = None
        sched._write_conditions([{"id": "c1", "condition": "disk_low",
                                  "action": "warn"}])
        self.assertTrue(sched.cancel_job("c1"))

    def test_cancel_handles_condition_rewrite_failure(self):
        self._install_fake_scheduler()
        sched._write_conditions([{"id": "c1", "condition": "disk_low",
                                  "action": "warn"}])
        with mock.patch.object(sched, "_write_conditions",
                               side_effect=OSError("disk full")):
            # Rewrite fails, but the call must not raise; removed stays False
            # because the APScheduler removal also missed.
            result = sched.cancel_job("c1")
        self.assertFalse(result)


# ── fire_now ────────────────────────────────────────────────────────
class FireNowTests(SchedulerTestBase):
    def setUp(self):
        super().setUp()
        self.fired = []
        sched._state["actions"] = {
            "act": lambda a: self.fired.append(a) or f"did:{a}",
        }

    def test_fire_now_runs_apscheduler_job_action(self):
        fs = self._install_fake_scheduler()
        fs.jobs["j1"] = FakeJob("j1", FakeTrigger(),
                                kwargs={"action": "act", "arg": "now"})
        out = sched.fire_now("j1")
        self.assertEqual(out, "act: did:now")
        self.assertEqual(self.fired, ["now"])

    def test_fire_now_falls_back_to_conditional_trigger(self):
        self._install_fake_scheduler()  # no matching job in store
        sched._write_conditions([
            {"id": "cond1", "condition": "disk_low", "action": "act",
             "arg": "cval"},
        ])
        out = sched.fire_now("cond1")
        self.assertEqual(out, "act: did:cval")
        self.assertEqual(self.fired, ["cval"])

    def test_fire_now_unknown_job_reports_not_found(self):
        self._install_fake_scheduler()
        out = sched.fire_now("ghost")
        self.assertIn("not found", out)
        self.assertIn("ghost", out)

    def test_fire_now_without_scheduler_checks_conditions(self):
        sched._state["scheduler"] = None
        sched._write_conditions([
            {"id": "c1", "condition": "disk_low", "action": "act", "arg": "z"},
        ])
        out = sched.fire_now("c1")
        self.assertEqual(out, "act: did:z")

    def test_fire_now_handles_get_job_exception(self):
        fs = self._install_fake_scheduler()
        with mock.patch.object(fs, "get_job",
                               side_effect=RuntimeError("store error")):
            # get_job raises → job treated as None → falls to conditions →
            # not found.
            out = sched.fire_now("whatever")
        self.assertIn("not found", out)


# ── status ──────────────────────────────────────────────────────────
class StatusTests(SchedulerTestBase):
    def test_status_when_not_running(self):
        sched._state["scheduler"] = None
        st = sched.status()
        self.assertTrue(st["available"])     # apscheduler installed
        self.assertFalse(st["running"])
        self.assertEqual(st["job_count"], 0)
        self.assertEqual(st["condition_count"], 0)
        self.assertEqual(st["uptime_seconds"], 0)
        self.assertIn("registered_conditions", st)

    def test_status_running_with_jobs_and_conditions(self):
        fs = self._install_fake_scheduler()
        fs.jobs["j1"] = FakeJob("j1", FakeTrigger(),
                                kwargs={"action": "a"})
        sched._state["started_at"] = 1000.0
        sched._write_conditions([{"id": "c1", "condition": "disk_low",
                                  "action": "warn"}])
        with mock.patch.object(sched.time, "time", return_value=1050.0):
            st = sched.status()
        self.assertTrue(st["running"])
        self.assertEqual(st["job_count"], 1)
        self.assertEqual(st["condition_count"], 1)
        self.assertEqual(st["uptime_seconds"], 50.0)

    def test_status_reports_last_error(self):
        sched._state["scheduler"] = None
        sched._state["last_error"] = "something broke"
        self.assertEqual(sched.status()["last_error"], "something broke")


# ── time-string parsing helpers ─────────────────────────────────────
class ParseClockTests(SchedulerTestBase):
    def test_parse_clock_variants(self):
        cases = {
            "8am": (8, 0),
            "8:30 am": (8, 30),
            "8 pm": (20, 0),
            "12am": (0, 0),
            "12pm": (12, 0),
            "20:15": (20, 15),
            "9": (9, 0),
        }
        for text, expected in cases.items():
            self.assertEqual(sched.parse_clock(text), expected, text)

    def test_parse_clock_strips_dots_in_ampm(self):
        # "a.m." → "am" after dot-stripping; the colon must remain for the
        # minutes group to parse (dot-stripping "8.30" yields "830", which the
        # regex rejects — so dotted times only work as the am/pm marker).
        self.assertEqual(sched.parse_clock("8:30 p.m."), (20, 30))

    def test_parse_clock_empty_returns_none(self):
        self.assertIsNone(sched.parse_clock(""))

    def test_parse_clock_garbage_returns_none(self):
        self.assertIsNone(sched.parse_clock("not a time"))

    def test_parse_clock_out_of_range_returns_none(self):
        self.assertIsNone(sched.parse_clock("25:00"))
        self.assertIsNone(sched.parse_clock("8:99"))


class ParseDowTests(SchedulerTestBase):
    def test_parse_dow_daily_means_any(self):
        for t in ("daily", "everyday", "every day", "any"):
            self.assertIsNone(sched.parse_dow(t))

    def test_parse_dow_weekdays_and_weekends(self):
        self.assertEqual(sched.parse_dow("weekdays"), "mon-fri")
        self.assertEqual(sched.parse_dow("weekday"), "mon-fri")
        self.assertEqual(sched.parse_dow("weekend"), "sat,sun")
        self.assertEqual(sched.parse_dow("weekends"), "sat,sun")

    def test_parse_dow_single_and_aliases(self):
        self.assertEqual(sched.parse_dow("monday"), "mon")
        self.assertEqual(sched.parse_dow("tues"), "tue")
        self.assertEqual(sched.parse_dow("thurs"), "thu")

    def test_parse_dow_multiple_separators(self):
        self.assertEqual(sched.parse_dow("mon, wed, fri"), "mon,wed,fri")
        self.assertEqual(sched.parse_dow("sat/sun"), "sat,sun")
        self.assertEqual(sched.parse_dow("mon and tue"), "mon,tue")

    def test_parse_dow_empty_returns_none(self):
        self.assertIsNone(sched.parse_dow(""))

    def test_parse_dow_unrecognised_returns_none(self):
        self.assertIsNone(sched.parse_dow("someday"))


class ParseEveryTests(SchedulerTestBase):
    def test_parse_every_seconds(self):
        self.assertEqual(sched.parse_every("45 seconds"), {"seconds": 45})
        self.assertEqual(sched.parse_every("30 sec"), {"seconds": 30})

    def test_parse_every_minutes(self):
        self.assertEqual(sched.parse_every("30 minutes"), {"minutes": 30})
        self.assertEqual(sched.parse_every("5 min"), {"minutes": 5})

    def test_parse_every_hours(self):
        self.assertEqual(sched.parse_every("2 hours"), {"hours": 2})
        self.assertEqual(sched.parse_every("3 hr"), {"hours": 3})

    def test_parse_every_empty_returns_none(self):
        self.assertIsNone(sched.parse_every(""))

    def test_parse_every_garbage_returns_none(self):
        self.assertIsNone(sched.parse_every("a while"))
        self.assertIsNone(sched.parse_every("5 fortnights"))


class ParseWhenTests(SchedulerTestBase):
    def test_parse_when_in_minutes(self):
        before = datetime.now().astimezone()
        result = sched.parse_when("in 30 minutes")
        self.assertIsNotNone(result)
        delta = result - before
        self.assertGreater(delta, timedelta(minutes=29))
        self.assertLess(delta, timedelta(minutes=31))

    def test_parse_when_in_seconds_and_hours(self):
        r_sec = sched.parse_when("in 45 seconds")
        r_hr = sched.parse_when("in 2 hours")
        self.assertIsNotNone(r_sec)
        self.assertIsNotNone(r_hr)
        # Aware datetimes.
        self.assertIsNotNone(r_sec.tzinfo)

    def test_parse_when_tomorrow_at_clock(self):
        result = sched.parse_when("tomorrow 8am")
        self.assertIsNotNone(result)
        tomorrow = (datetime.now().astimezone() + timedelta(days=1)).date()
        self.assertEqual(result.date(), tomorrow)
        self.assertEqual((result.hour, result.minute), (8, 0))

    def test_parse_when_tomorrow_bad_clock_returns_none(self):
        self.assertIsNone(sched.parse_when("tomorrow whenever"))

    def test_parse_when_today_future_time(self):
        # Pin "now" so the requested time is unambiguously in the future today.
        fixed_now = datetime(2030, 6, 1, 6, 0, 0).astimezone()
        with mock.patch.object(sched, "datetime") as dt:
            dt.now.return_value = fixed_now
            dt.strptime = datetime.strptime
            result = sched.parse_when("today 8am")
        self.assertEqual((result.hour, result.minute), (8, 0))
        self.assertEqual(result.date(), fixed_now.date())

    def test_parse_when_today_past_time_rolls_to_tomorrow(self):
        fixed_now = datetime(2030, 6, 1, 10, 0, 0).astimezone()
        with mock.patch.object(sched, "datetime") as dt:
            dt.now.return_value = fixed_now
            dt.strptime = datetime.strptime
            result = sched.parse_when("today 8am")   # 8am already passed
        self.assertEqual(result.date(),
                         (fixed_now + timedelta(days=1)).date())
        self.assertEqual((result.hour, result.minute), (8, 0))

    def test_parse_when_iso_formats(self):
        self.assertEqual(
            sched.parse_when("2026-06-01T08:00:00").replace(tzinfo=None),
            datetime(2026, 6, 1, 8, 0, 0))
        self.assertEqual(
            sched.parse_when("2026-06-01 08:00").replace(tzinfo=None),
            datetime(2026, 6, 1, 8, 0))
        self.assertEqual(
            sched.parse_when("2026-06-01").replace(tzinfo=None),
            datetime(2026, 6, 1, 0, 0))

    def test_parse_when_empty_returns_none(self):
        self.assertIsNone(sched.parse_when(""))

    def test_parse_when_unparseable_returns_none(self):
        self.assertIsNone(sched.parse_when("sometime next century"))


# ── _make_aware / helpers ───────────────────────────────────────────
class HelperTests(SchedulerTestBase):
    def test_make_aware_naive_gets_tzinfo(self):
        result = sched._make_aware(datetime(2030, 1, 1, 8, 0))
        self.assertIsNotNone(result.tzinfo)

    def test_make_aware_already_aware_unchanged(self):
        aware = datetime(2030, 1, 1, 8, 0).astimezone()
        self.assertIs(sched._make_aware(aware), aware)

    def test_job_kwargs_minimal(self):
        self.assertEqual(sched._job_kwargs("act", "", None),
                         {"action": "act", "arg": ""})

    def test_job_kwargs_with_arg_and_chain(self):
        kw = sched._job_kwargs("act", "x", [{"action": "b"}])
        self.assertEqual(kw["action"], "act")
        self.assertEqual(kw["arg"], "x")
        self.assertEqual(kw["chain"], [{"action": "b"}])

    def test_short_id_shape(self):
        sid = sched._short_id("pre")
        self.assertTrue(sid.startswith("pre_"))
        self.assertEqual(len(sid), len("pre_") + 8)
        # Two calls differ.
        self.assertNotEqual(sched._short_id("pre"), sched._short_id("pre"))

    def test_scheduler_raises_when_unbootstrapped(self):
        sched._state["scheduler"] = None
        with self.assertRaises(RuntimeError):
            sched._scheduler()

    def test_scheduler_returns_live_instance(self):
        fs = self._install_fake_scheduler()
        self.assertIs(sched._scheduler(), fs)


# ── atomic write fallback ───────────────────────────────────────────
class AtomicWriteTests(SchedulerTestBase):
    def test_atomic_write_json_writes_file(self):
        path = os.path.join(self._tmp, "out.json")
        sched._atomic_write_json(path, {"a": 1})
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1})


if __name__ == "__main__":
    unittest.main()
